"""Milestone 4: neural optimality transfer across mechanically different synthetic tasks."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from levelup.core.experiment import DiscoveryPoint, DiscoveryRun, ExposureManifest
from levelup.core.trajectory import ActionRecord, Trajectory, TrajectoryStep
from levelup.envs.mechanictrack import (
    FEATURE_NAMES,
    HELD_OUT_FAMILY,
    TRAIN_FAMILIES,
    ActionMechanic,
    MechanicTrack,
    MechanicTrackBundle,
    collect_bundles,
    held_out_tasks,
)
from levelup.evaluation import evaluate_trajectory

TRAIN_TASKS_PER_FAMILY = 40
HELD_OUT_TASK_COUNT = 8
TRAIN_SEED_BASE = 900
HELD_OUT_GENERATOR_SEED = 1337
MODEL_SEED = 42
DEFAULT_REPLICATES = 20
DEFAULT_MAX_EPISODES = 150
DEFAULT_BUDGETS = (1, 10, 50, 150)
DEFAULT_SEARCH_SEED = 950_000
DEFAULT_TEMPERATURE = 0.9
MODEL_EPOCHS = 250

ConditionKind = Literal[
    "uniform",
    "frontier_to_optimum_delta",
    "shuffled_transition_direction",
    "pooled_frontier_optimum",
    "imitate_optimum",
]


class FeatureScorer(nn.Module):
    """Tiny MLP that never receives action aliases or mechanic-family IDs."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass(frozen=True, slots=True)
class TrainedCondition:
    condition_id: ConditionKind
    model: FeatureScorer | None
    manifest: ExposureManifest
    training_loss: float | None


def _frequency(trajectory: Trajectory) -> dict[str, float]:
    counts = Counter(step.action.name for step in trajectory.steps)
    total = len(trajectory.steps)
    return {name: count / total for name, count in counts.items()}


def _feature_tensor(environment: MechanicTrack, action: ActionMechanic) -> torch.Tensor:
    return torch.tensor(action.feature_vector(environment.target), dtype=torch.float32)


def _training_examples(
    bundles: tuple[MechanicTrackBundle, ...],
    condition_id: ConditionKind,
    *,
    label_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if condition_id == "uniform":
        raise ValueError("uniform condition has no training examples")
    rng = random.Random(label_seed)
    rows: list[torch.Tensor] = []
    targets: list[float] = []
    for bundle in bundles:
        frontier = bundle.trajectory_for("frontier")
        optimum = bundle.trajectory_for("optimum")
        frontier_frequency = _frequency(frontier)
        optimum_frequency = _frequency(optimum)
        direction = 1.0
        if condition_id == "shuffled_transition_direction" and rng.random() >= 0.5:
            direction = -1.0

        for action in bundle.environment.valid_actions:
            before = frontier_frequency.get(action.alias, 0.0)
            after = optimum_frequency.get(action.alias, 0.0)
            if condition_id in (
                "frontier_to_optimum_delta",
                "shuffled_transition_direction",
            ):
                target = (after - before) * direction
            elif condition_id == "pooled_frontier_optimum":
                target = (before + after) / 2.0
            elif condition_id == "imitate_optimum":
                target = after
            else:
                raise ValueError(f"unsupported training condition: {condition_id!r}")
            rows.append(_feature_tensor(bundle.environment, action))
            targets.append(target)

    return torch.stack(rows), torch.tensor(targets, dtype=torch.float32)


def _manifest(
    condition_id: ConditionKind,
    bundles: tuple[MechanicTrackBundle, ...],
    held_out_ids: tuple[str, ...],
) -> ExposureManifest:
    if condition_id == "uniform":
        labels: tuple[str, ...] = ()
    elif condition_id == "imitate_optimum":
        labels = ("optimum",)
    else:
        labels = ("frontier", "optimum")

    exposed: list[str] = []
    for bundle in bundles:
        for label in labels:
            exposed.append(bundle.ladder.stage(label).trajectory_id)
    return ExposureManifest(
        condition_id=condition_id,
        train_task_ids=(
            tuple(bundle.ladder.task_id for bundle in bundles) if labels else ()
        ),
        held_out_task_ids=held_out_ids,
        exposed_trajectory_ids=tuple(exposed),
        exposed_stage_labels=labels,
        privileged_state_access=False,
        structured_constraint_access=True,
        metadata={
            "milestone": 4,
            "neural_model": condition_id != "uniform",
            "action_aliases_visible_to_model": False,
            "mechanic_family_id_visible_to_model": False,
        },
    )


def _train_model(
    bundles: tuple[MechanicTrackBundle, ...],
    condition_id: ConditionKind,
    *,
    model_seed: int = MODEL_SEED,
) -> tuple[FeatureScorer, float]:
    torch.manual_seed(model_seed)
    model = FeatureScorer()
    features, targets = _training_examples(
        bundles,
        condition_id,
        label_seed=model_seed + 111,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(MODEL_EPOCHS):
        optimizer.zero_grad()
        prediction = model(features)
        loss = torch.mean((prediction - targets) ** 2)
        loss.backward()
        optimizer.step()
    model.eval()
    return model, float(loss.detach().item())


def build_training_bundles() -> tuple[MechanicTrackBundle, ...]:
    bundles: list[MechanicTrackBundle] = []
    for family_index, family in enumerate(TRAIN_FAMILIES):
        bundles.extend(
            collect_bundles(
                family,
                TRAIN_TASKS_PER_FAMILY,
                TRAIN_SEED_BASE + family_index * 100,
            )
        )
    return tuple(bundles)


def validate_training_ladders(bundles: tuple[MechanicTrackBundle, ...]) -> None:
    for bundle in bundles:
        for stage in bundle.ladder.stages:
            trajectory = bundle.trajectories[stage.trajectory_id]
            result = evaluate_trajectory(bundle.environment.fresh(), trajectory)
            if not result.performance_eligible_for(bundle.environment.task_spec):
                raise RuntimeError(f"invalid training demonstration: {stage.stage_id}")
            if result.performance_value != stage.performance_value:
                raise RuntimeError(f"training demonstration drift: {stage.stage_id}")


def build_conditions(
    bundles: tuple[MechanicTrackBundle, ...],
    held_out_ids: tuple[str, ...],
) -> dict[ConditionKind, TrainedCondition]:
    conditions: dict[ConditionKind, TrainedCondition] = {}
    for condition_id in (
        "uniform",
        "frontier_to_optimum_delta",
        "shuffled_transition_direction",
        "pooled_frontier_optimum",
        "imitate_optimum",
    ):
        manifest = _manifest(condition_id, bundles, held_out_ids)
        if condition_id == "uniform":
            model = None
            loss = None
        else:
            model, loss = _train_model(bundles, condition_id)
        conditions[condition_id] = TrainedCondition(
            condition_id=condition_id,
            model=model,
            manifest=manifest,
            training_loss=loss,
        )
    return conditions


def _manifest_summary(manifest: ExposureManifest) -> dict[str, Any]:
    payload = manifest.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "condition_id": manifest.condition_id,
        "train_task_count": len(manifest.train_task_ids),
        "held_out_task_ids": list(manifest.held_out_task_ids),
        "exposed_trajectory_count": len(manifest.exposed_trajectory_ids),
        "exposed_stage_labels": list(manifest.exposed_stage_labels),
        "privileged_state_access": manifest.privileged_state_access,
        "structured_constraint_access": manifest.structured_constraint_access,
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _structured_forbidden_aliases(environment: MechanicTrack) -> set[str]:
    aliases: set[str] = set()
    for constraint in environment.task_spec.constraints:
        if constraint.verifier_id == "never_use_action":
            alias = constraint.verifier_config.get("action")
            if isinstance(alias, str):
                aliases.add(alias)
    return aliases


def _action_weights(
    environment: MechanicTrack,
    model: FeatureScorer | None,
    *,
    temperature: float,
) -> dict[str, float]:
    if model is None:
        return {action.alias: 1.0 for action in environment.actions}

    scored_actions = environment.valid_actions
    features = torch.stack(
        [_feature_tensor(environment, action) for action in scored_actions]
    )
    with torch.no_grad():
        scores = model(features)
    standard_deviation = scores.std(unbiased=False)
    if len(scores) > 1 and float(standard_deviation) > 1e-6:
        normalized = (scores - scores.mean()) / standard_deviation
    else:
        normalized = torch.zeros_like(scores)
    probabilities = torch.softmax(normalized / temperature, dim=0)
    weights = {
        action.alias: float(probability)
        for action, probability in zip(scored_actions, probabilities.tolist())
    }
    for action in environment.actions:
        weights.setdefault(action.alias, 0.0)
    return weights


def _sample_candidate(
    environment: MechanicTrack,
    weights: dict[str, float],
    rng: random.Random,
    trajectory_id: str,
) -> tuple[Trajectory, float] | None:
    outcome = environment.reset()
    forbidden = _structured_forbidden_aliases(environment)
    actions: list[str] = []
    for _ in range(environment.target * 4):
        if outcome.completed:
            break
        raw_available = outcome.observation.get("available_actions")
        if not isinstance(raw_available, list):
            raise RuntimeError("MechanicTrack observation is missing available_actions")
        aliases = [
            item.get("alias")
            for item in raw_available
            if isinstance(item, dict)
            and isinstance(item.get("alias"), str)
            and item.get("alias") not in forbidden
        ]
        if not aliases:
            return None
        alias = rng.choices(aliases, weights=[weights[name] for name in aliases], k=1)[0]
        actions.append(alias)
        outcome = environment.step(ActionRecord(name=alias))
    if not outcome.completed:
        return None
    trajectory = Trajectory(
        trajectory_id=trajectory_id,
        task_id=environment.task_spec.task_id,
        source="agent",
        steps=tuple(
            TrajectoryStep(index=index, action=ActionRecord(name=alias))
            for index, alias in enumerate(actions)
        ),
    )
    return trajectory, environment.objective_value()


def discovery_run(
    environment: MechanicTrack,
    optimum_value: float,
    condition: TrainedCondition,
    *,
    seed: int,
    max_episodes: int = DEFAULT_MAX_EPISODES,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> DiscoveryRun:
    if not budgets or tuple(sorted(set(budgets))) != budgets or budgets[-1] > max_episodes:
        raise ValueError("budgets must be strictly increasing and not exceed max_episodes")
    rng = random.Random(seed)
    weights = _action_weights(environment, condition.model, temperature=temperature)
    best: float | None = None
    first_optimum: int | None = None
    points: list[DiscoveryPoint] = []
    budget_set = set(budgets)

    for episode in range(1, max_episodes + 1):
        sampled = _sample_candidate(
            environment.fresh(),
            weights,
            rng,
            trajectory_id=(
                f"search:{condition.condition_id}:{environment.task_spec.task_id}:"
                f"s{seed}:e{episode}"
            ),
        )
        if sampled is not None:
            trajectory, measured_performance = sampled
            if best is None or measured_performance < best:
                result = evaluate_trajectory(environment.fresh(), trajectory)
                if not result.performance_eligible_for(environment.task_spec):
                    raise RuntimeError("search candidate failed independent validity replay")
                if result.performance_value != measured_performance:
                    raise RuntimeError("search measurement disagrees with deterministic replay")
                best = measured_performance
                if measured_performance == optimum_value and first_optimum is None:
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
        condition_id=condition.condition_id,
        task_id=environment.task_spec.task_id,
        seed=seed,
        optimum_value=optimum_value,
        first_optimum_episode=first_optimum,
        points=tuple(points),
    )


def run_experiment(
    *,
    replicates: int = DEFAULT_REPLICATES,
    max_episodes: int = DEFAULT_MAX_EPISODES,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    base_seed: int = DEFAULT_SEARCH_SEED,
    held_out_count: int = HELD_OUT_TASK_COUNT,
) -> dict[str, Any]:
    """Train neural conditions and evaluate on the entirely held-out heat mechanic family."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    torch.set_num_threads(1)
    bundles = build_training_bundles()
    validate_training_ladders(bundles)
    held_out = held_out_tasks(held_out_count, HELD_OUT_GENERATOR_SEED)
    held_out_ids = tuple(environment.task_spec.task_id for environment, _ in held_out)
    conditions = build_conditions(bundles, held_out_ids)

    report: dict[str, Any] = {
        "experiment": "milestone4_neural_cross_mechanic_v1",
        "train_families": list(TRAIN_FAMILIES),
        "held_out_family": HELD_OUT_FAMILY,
        "train_tasks_per_family": TRAIN_TASKS_PER_FAMILY,
        "held_out_task_count": held_out_count,
        "replicates": replicates,
        "max_episodes": max_episodes,
        "budgets": list(budgets),
        "model": {
            "input_features": list(FEATURE_NAMES),
            "hidden_widths": [32, 16],
            "epochs": MODEL_EPOCHS,
            "model_seed": MODEL_SEED,
            "action_alias_input": False,
            "family_id_input": False,
        },
        "note": (
            "Synthetic cross-mechanic neural transfer experiment. Positive transfer here is not "
            "evidence of cross-game human-to-TAS transfer; the model still receives structured "
            "numeric action descriptors and structured constraint access."
        ),
        "conditions": {},
    }

    condition_reports: dict[str, dict[str, Any]] = {}
    paired_totals: dict[str, list[int]] = {}
    for condition_id, condition in conditions.items():
        totals: list[int] = []
        task_hits: dict[str, list[int]] = {
            environment.task_spec.task_id: [] for environment, _ in held_out
        }
        task_runs: dict[str, list[DiscoveryRun]] = {
            environment.task_spec.task_id: [] for environment, _ in held_out
        }
        for replicate in range(replicates):
            total = 0
            for task_index, (environment, optimum) in enumerate(held_out):
                seed = base_seed + replicate * 100 + task_index
                run = discovery_run(
                    environment,
                    optimum,
                    condition,
                    seed=seed,
                    max_episodes=max_episodes,
                    budgets=budgets,
                )
                task_runs[environment.task_spec.task_id].append(run)
                hit = run.first_optimum_episode or (max_episodes + 1)
                task_hits[environment.task_spec.task_id].append(hit)
                total += hit
            totals.append(total)
        paired_totals[condition_id] = totals

        task_summary: dict[str, dict[str, Any]] = {}
        for task_id, hits in task_hits.items():
            runs = task_runs[task_id]
            curve: list[dict[str, Any]] = []
            for budget in budgets:
                budget_points = [
                    next(point for point in run.points if point.budget == budget)
                    for run in runs
                ]
                valid_best = [
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
                        "median_best_performance": (
                            statistics.median(valid_best) if valid_best else None
                        ),
                    }
                )
            task_summary[task_id] = {
                "median_episodes_to_optimum": statistics.median(hits),
                "success_rate_at_budget": sum(hit <= max_episodes for hit in hits) / len(hits),
                "discovery_curve": curve,
            }

        condition_reports[condition_id] = {
            "training_loss": condition.training_loss,
            "exposure": _manifest_summary(condition.manifest),
            "median_total_episodes_across_held_out_tasks": statistics.median(totals),
            "mean_total_episodes_across_held_out_tasks": statistics.mean(totals),
            "held_out_task_success_rate": (
                sum(hit <= max_episodes for hits in task_hits.values() for hit in hits)
                / (replicates * len(held_out))
            ),
            "tasks": task_summary,
        }

    report["conditions"] = condition_reports
    delta = condition_reports["frontier_to_optimum_delta"][
        "median_total_episodes_across_held_out_tasks"
    ]
    report["comparisons"] = {
        f"delta_sample_efficiency_vs_{baseline}": (
            condition_reports[baseline]["median_total_episodes_across_held_out_tasks"] / delta
        )
        for baseline in (
            "uniform",
            "shuffled_transition_direction",
            "pooled_frontier_optimum",
            "imitate_optimum",
        )
    }
    report["paired_replicates"] = {
        "delta_beats_pooled_count": sum(
            delta_total < pooled_total
            for delta_total, pooled_total in zip(
                paired_totals["frontier_to_optimum_delta"],
                paired_totals["pooled_frontier_optimum"],
            )
        ),
        "delta_beats_shuffled_count": sum(
            delta_total < shuffled_total
            for delta_total, shuffled_total in zip(
                paired_totals["frontier_to_optimum_delta"],
                paired_totals["shuffled_transition_direction"],
            )
        ),
        "replicates": replicates,
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
