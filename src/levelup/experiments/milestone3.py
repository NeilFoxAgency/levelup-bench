"""Milestone 3: a tiny, reproducible optimality-transfer sanity experiment.

This experiment is intentionally synthetic. It validates the protocol for asking whether
information about *how* stronger trajectories differ from weaker ones can improve search on
held-out tasks. It is not evidence of cross-game or human-to-TAS transfer.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from levelup.core.experiment import DiscoveryPoint, DiscoveryRun, ExposureManifest
from levelup.core.trajectory import ActionRecord, Trajectory, TrajectoryStep
from levelup.envs.macrotrack import (
    STAGE_LABELS,
    VALID_ACTIONS,
    MacroTrack,
    MacroTrackBundle,
    macro_track_bundle,
    optimum_value,
)
from levelup.evaluation import evaluate_trajectory

TRAIN_DISTANCES = (6, 8, 9, 10, 11, 12)
HELD_OUT_DISTANCES = (13, 14, 15, 16)
DEFAULT_BUDGETS = (1, 10, 100, 300)


@dataclass(frozen=True, slots=True)
class ActionPrior:
    """A tiny learned proposal model over action names."""

    weights: dict[str, float]

    def __post_init__(self) -> None:
        if set(self.weights) != set(VALID_ACTIONS):
            raise ValueError("action prior must assign every valid action exactly one weight")
        if any(weight <= 0 for weight in self.weights.values()):
            raise ValueError("action-prior weights must be positive")

    def choose(self, available_actions: list[str], rng: random.Random) -> str:
        legal = [name for name in available_actions if name in self.weights]
        if not legal:
            raise RuntimeError("no permitted action is available")
        return rng.choices(legal, weights=[self.weights[name] for name in legal], k=1)[0]


def _normalized_frequency(actions: list[str]) -> dict[str, float]:
    counts = Counter(actions)
    total = len(actions)
    return {name: counts[name] / total for name in VALID_ACTIONS}


def uniform_prior() -> ActionPrior:
    return ActionPrior({name: 1.0 for name in VALID_ACTIONS})


def fit_stage_imitation_prior(
    bundles: tuple[MacroTrackBundle, ...],
    stage_label: str,
    *,
    smoothing: float = 1.0,
) -> ActionPrior:
    """Count-based behavioral imitation of exactly one exposed skill stage."""

    if stage_label not in STAGE_LABELS:
        raise ValueError(f"unknown stage label: {stage_label!r}")
    counts = Counter({name: smoothing for name in VALID_ACTIONS})
    for bundle in bundles:
        trajectory = bundle.trajectory_for(stage_label)
        counts.update(step.action.name for step in trajectory.steps)
    return ActionPrior({name: float(counts[name]) for name in VALID_ACTIONS})


def fit_pooled_imitation_prior(
    bundles: tuple[MacroTrackBundle, ...],
    stage_labels: tuple[str, ...],
    *,
    smoothing: float = 1.0,
) -> ActionPrior:
    """Imitate a pool of stages while deliberately discarding their ordering."""

    counts = Counter({name: smoothing for name in VALID_ACTIONS})
    for bundle in bundles:
        for label in stage_labels:
            trajectory = bundle.trajectory_for(label)
            counts.update(step.action.name for step in trajectory.steps)
    return ActionPrior({name: float(counts[name]) for name in VALID_ACTIONS})


def fit_transition_delta_prior(
    bundles: tuple[MacroTrackBundle, ...],
    *,
    from_stage: str = "frontier",
    to_stage: str = "optimum",
    smoothing: float = 0.05,
) -> ActionPrior:
    """Learn which actions increase when trajectories cross an optimality gap.

    For each training task, action frequencies are normalized within the two trajectories.
    Only positive frequency changes are accumulated. This deliberately tiny learner represents
    the narrow hypothesis "what became more common when the run got better?" without seeing any
    held-out optimum trajectory.
    """

    weights = {name: smoothing for name in VALID_ACTIONS}
    for bundle in bundles:
        before = [step.action.name for step in bundle.trajectory_for(from_stage).steps]
        after = [step.action.name for step in bundle.trajectory_for(to_stage).steps]
        before_frequency = _normalized_frequency(before)
        after_frequency = _normalized_frequency(after)
        for name in VALID_ACTIONS:
            weights[name] += max(0.0, after_frequency[name] - before_frequency[name])
    return ActionPrior(weights)


def _structured_forbidden_actions(environment: MacroTrack) -> set[str]:
    """Use structured constraint metadata so this milestone does not test language parsing."""

    forbidden: set[str] = set()
    for constraint in environment.task_spec.constraints:
        if constraint.verifier_id == "never_use_action":
            action = constraint.verifier_config.get("action")
            if isinstance(action, str):
                forbidden.add(action)
    return forbidden


def _sample_trajectory(
    distance: int,
    prior: ActionPrior,
    rng: random.Random,
    trajectory_id: str,
) -> tuple[Trajectory, float]:
    """Generate one candidate and return its benchmark-side measured performance."""

    environment = MacroTrack(distance)
    outcome = environment.reset()
    forbidden = _structured_forbidden_actions(environment)
    actions: list[str] = []
    while not outcome.completed:
        raw_available = outcome.observation.get("available_actions")
        if not isinstance(raw_available, list):
            raise RuntimeError("MacroTrack observation is missing available_actions")
        available = [
            name
            for name in raw_available
            if isinstance(name, str) and name not in forbidden
        ]
        action = prior.choose(available, rng)
        actions.append(action)
        outcome = environment.step(ActionRecord(name=action))

    trajectory = Trajectory(
        trajectory_id=trajectory_id,
        task_id=environment.task_spec.task_id,
        source="agent",
        steps=tuple(
            TrajectoryStep(index=index, action=ActionRecord(name=name))
            for index, name in enumerate(actions)
        ),
    )
    return trajectory, environment.objective_value()


def discovery_run(
    distance: int,
    condition_id: str,
    prior: ActionPrior,
    *,
    seed: int,
    max_episodes: int = 1000,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
) -> DiscoveryRun:
    """Search one held-out task and record its best-valid discovery curve."""

    if not budgets or tuple(sorted(set(budgets))) != budgets or budgets[-1] > max_episodes:
        raise ValueError("budgets must be strictly increasing and not exceed max_episodes")
    optimum = optimum_value(distance)
    rng = random.Random(seed)
    best: float | None = None
    first_optimum: int | None = None
    points: list[DiscoveryPoint] = []
    budget_set = set(budgets)

    for episode in range(1, max_episodes + 1):
        trajectory, measured_performance = _sample_trajectory(
            distance,
            prior,
            rng,
            trajectory_id=f"search:{condition_id}:d{distance}:s{seed}:e{episode}",
        )
        if best is None or measured_performance < best:
            # Only frontier-changing candidates are replayed through the independent evaluator.
            # This keeps large search sweeps fast without trusting the learner's own score.
            result = evaluate_trajectory(MacroTrack(distance), trajectory)
            if not result.performance_eligible_for(MacroTrack(distance).task_spec):
                raise RuntimeError("structured-rule sampler produced an invalid candidate")
            if result.performance_value != measured_performance:
                raise RuntimeError("candidate measurement disagrees with deterministic replay")
            best = measured_performance
            if measured_performance == optimum and first_optimum is None:
                first_optimum = episode

        if episode in budget_set:
            points.append(
                DiscoveryPoint(
                    budget=episode,
                    best_performance=best,
                    optimum_found=first_optimum is not None,
                )
            )
        if first_optimum is not None:
            break

    recorded = {point.budget for point in points}
    for budget in budgets:
        if budget not in recorded:
            points.append(
                DiscoveryPoint(
                    budget=budget,
                    best_performance=best,
                    optimum_found=first_optimum is not None,
                )
            )
    points.sort(key=lambda point: point.budget)
    return DiscoveryRun(
        condition_id=condition_id,
        task_id=MacroTrack(distance).task_spec.task_id,
        seed=seed,
        optimum_value=optimum,
        first_optimum_episode=first_optimum,
        points=tuple(points),
    )


def _manifest(
    condition_id: str,
    train_bundles: tuple[MacroTrackBundle, ...],
    held_out_ids: tuple[str, ...],
    labels: tuple[str, ...],
) -> ExposureManifest:
    exposed: list[str] = []
    for bundle in train_bundles:
        for label in labels:
            exposed.append(bundle.ladder.stage(label).trajectory_id)
    return ExposureManifest(
        condition_id=condition_id,
        train_task_ids=(
            tuple(bundle.ladder.task_id for bundle in train_bundles) if labels else ()
        ),
        held_out_task_ids=held_out_ids,
        exposed_trajectory_ids=tuple(exposed),
        exposed_stage_labels=labels,
        privileged_state_access=False,
        structured_constraint_access=True,
        metadata={"synthetic_sanity_experiment": True},
    )


def build_conditions() -> tuple[
    tuple[MacroTrackBundle, ...],
    tuple[str, ...],
    dict[str, tuple[ActionPrior, ExposureManifest]],
]:
    train_bundles = tuple(macro_track_bundle(distance) for distance in TRAIN_DISTANCES)
    held_out_ids = tuple(MacroTrack(distance).task_spec.task_id for distance in HELD_OUT_DISTANCES)

    conditions: dict[str, tuple[ActionPrior, ExposureManifest]] = {
        "uniform": (
            uniform_prior(),
            _manifest("uniform", train_bundles, held_out_ids, ()),
        ),
        "imitate_frontier": (
            fit_stage_imitation_prior(train_bundles, "frontier"),
            _manifest("imitate_frontier", train_bundles, held_out_ids, ("frontier",)),
        ),
        "imitate_optimum": (
            fit_stage_imitation_prior(train_bundles, "optimum"),
            _manifest("imitate_optimum", train_bundles, held_out_ids, ("optimum",)),
        ),
        "pooled_frontier_optimum": (
            fit_pooled_imitation_prior(train_bundles, ("frontier", "optimum")),
            _manifest(
                "pooled_frontier_optimum",
                train_bundles,
                held_out_ids,
                ("frontier", "optimum"),
            ),
        ),
        "frontier_to_optimum_delta": (
            fit_transition_delta_prior(train_bundles),
            _manifest(
                "frontier_to_optimum_delta",
                train_bundles,
                held_out_ids,
                ("frontier", "optimum"),
            ),
        ),
    }
    return train_bundles, held_out_ids, conditions


def validate_training_ladders(train_bundles: tuple[MacroTrackBundle, ...]) -> None:
    """Replay every exposed demonstration before allowing it into an experiment."""

    for bundle in train_bundles:
        for stage in bundle.ladder.stages:
            trajectory = bundle.trajectories[stage.trajectory_id]
            result = evaluate_trajectory(MacroTrack(bundle.distance), trajectory)
            if not result.performance_eligible_for(MacroTrack(bundle.distance).task_spec):
                raise RuntimeError(f"invalid training demonstration: {stage.stage_id}")
            if result.performance_value != stage.performance_value:
                raise RuntimeError(f"training demonstration drift: {stage.stage_id}")


def run_experiment(
    *,
    replicates: int = 20,
    max_episodes: int = 300,
    base_seed: int = 17_000_000,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
) -> dict[str, Any]:
    """Run the deterministic Milestone 3 study and return JSON-serializable results."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    train_bundles, _, conditions = build_conditions()
    validate_training_ladders(train_bundles)

    report: dict[str, Any] = {
        "experiment": "milestone3_microtransfer_v1",
        "train_distances": list(TRAIN_DISTANCES),
        "held_out_distances": list(HELD_OUT_DISTANCES),
        "replicates": replicates,
        "max_episodes": max_episodes,
        "budgets": list(budgets),
        "conditions": {},
        "note": (
            "Synthetic in-family transfer sanity check. A positive effect here is not evidence "
            "of cross-game, human-to-TAS, or real-world optimality transfer."
        ),
    }

    condition_reports: dict[str, dict[str, Any]] = {}
    for condition_id, (prior, manifest) in conditions.items():
        totals: list[int] = []
        per_task_hits: dict[int, list[int]] = {distance: [] for distance in HELD_OUT_DISTANCES}
        per_task_runs: dict[int, list[DiscoveryRun]] = {
            distance: [] for distance in HELD_OUT_DISTANCES
        }
        for replicate in range(replicates):
            total = 0
            for task_index, distance in enumerate(HELD_OUT_DISTANCES):
                seed = base_seed + replicate * 100 + task_index
                run = discovery_run(
                    distance,
                    condition_id,
                    prior,
                    seed=seed,
                    max_episodes=max_episodes,
                    budgets=budgets,
                )
                per_task_runs[distance].append(run)
                hit = run.first_optimum_episode or (max_episodes + 1)
                per_task_hits[distance].append(hit)
                total += hit
            totals.append(total)

        task_summary: dict[str, dict[str, Any]] = {}
        for distance, hits in per_task_hits.items():
            curve: list[dict[str, Any]] = []
            for budget in budgets:
                budget_points = [
                    next(point for point in run.points if point.budget == budget)
                    for run in per_task_runs[distance]
                ]
                best_values = [
                    point.best_performance
                    for point in budget_points
                    if point.best_performance is not None
                ]
                curve.append(
                    {
                        "budget": budget,
                        "optimum_rate": (
                            sum(point.optimum_found for point in budget_points)
                            / len(budget_points)
                        ),
                        "median_best_performance": statistics.median(best_values),
                    }
                )
            task_summary[str(distance)] = {
                "median_episodes_to_optimum": statistics.median(hits),
                "success_rate_at_budget": sum(hit <= max_episodes for hit in hits) / len(hits),
                "discovery_curve": curve,
            }
        condition_reports[condition_id] = {
            "prior_weights": prior.weights,
            "exposure_manifest": manifest.model_dump(mode="json"),
            "median_total_episodes_across_held_out_tasks": statistics.median(totals),
            "tasks": task_summary,
        }

    report["conditions"] = condition_reports
    delta_median = condition_reports["frontier_to_optimum_delta"][
        "median_total_episodes_across_held_out_tasks"
    ]
    report["comparisons"] = {
        f"delta_sample_efficiency_vs_{baseline}": (
            condition_reports[baseline]["median_total_episodes_across_held_out_tasks"]
            / delta_median
        )
        for baseline in (
            "uniform",
            "imitate_frontier",
            "imitate_optimum",
            "pooled_frontier_optimum",
        )
    }
    return report


def main(output: str | None = None) -> None:
    report = run_experiment()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if output is None:
        print(rendered)
    else:
        Path(output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
