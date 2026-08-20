"""Milestone 5: learn action affordances through interaction before optimizing.

Reward/model choices are selected only on development mechanic families. The final Combo family
is evaluated after the method is frozen. An earlier Overdrive diagnostic is not treated as a
pristine final result.
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from levelup.core.experiment import ExposureManifest
from levelup.core.trajectory import ActionRecord, Trajectory, TrajectoryStep
from levelup.envs.adaptive_track import (
    DEVELOPMENT_FAMILIES,
    AdaptiveTrack,
    AdaptiveTrackBundle,
    collect_adaptive_bundles,
)
from levelup.envs.challenge_track import FINAL_CHALLENGE_FAMILY, held_out_combo_tasks
from levelup.evaluation import evaluate_trajectory
from levelup.learning.interaction import (
    PROBE_FEATURE_COUNT,
    InteractionScorer,
    ProbeResult,
    action_weights,
    available_aliases,
    probe_action_effects,
    structured_forbidden_aliases,
    train_interaction_model,
)

DEVELOPMENT_TASKS_PER_FAMILY = 30
FINAL_TASK_COUNT = 8
PROBES_PER_ACTION = 6
MODEL_SEED = 42
CV_MODEL_EPOCHS = 120
FINAL_MODEL_EPOCHS = 180
DEFAULT_REPLICATES = 20
DEFAULT_MAX_EPISODES = 150
DEFAULT_SEARCH_SEED = 1_900_000
DEFAULT_TEMPERATURE = 0.9
FINAL_GENERATOR_SEED = 2026
FINAL_FAMILY = FINAL_CHALLENGE_FAMILY


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    first_optimum_episode: int | None
    best_performance: float | None
    search_actions: int


def build_development_bundles(
    tasks_per_family: int = DEVELOPMENT_TASKS_PER_FAMILY,
) -> dict[str, tuple[AdaptiveTrackBundle, ...]]:
    return {
        family: collect_adaptive_bundles(
            family,
            tasks_per_family,
            900 + index * 100,
        )
        for index, family in enumerate(DEVELOPMENT_FAMILIES)
    }


def validate_training_ladders(bundles: tuple[AdaptiveTrackBundle, ...]) -> None:
    for bundle in bundles:
        for stage in bundle.ladder.stages:
            trajectory = bundle.trajectories[stage.trajectory_id]
            result = evaluate_trajectory(bundle.environment.fresh(), trajectory)
            if not result.performance_eligible_for(bundle.environment.task_spec):
                raise RuntimeError(f"invalid training demonstration: {stage.stage_id}")
            if result.performance_value != stage.performance_value:
                raise RuntimeError(f"training demonstration drift: {stage.stage_id}")


def build_probe_cache(
    bundles: tuple[AdaptiveTrackBundle, ...],
    *,
    probes_per_action: int,
) -> dict[str, ProbeResult]:
    cache: dict[str, ProbeResult] = {}
    for bundle in bundles:
        environment = bundle.environment
        seed = 700_000 + environment.generator_seed * 100 + environment.task_index
        cache[environment.task_spec.task_id] = probe_action_effects(
            environment,
            seed=seed,
            probes_per_action=probes_per_action,
        )
    return cache


def _sample_candidate(
    environment: Any,
    weights: dict[str, float],
    rng: random.Random,
    trajectory_id: str,
) -> tuple[Trajectory | None, float | None, int]:
    """Return a candidate plus every environment action spent generating it."""

    outcome = environment.reset()
    forbidden = structured_forbidden_aliases(environment)
    actions: list[str] = []
    for _ in range(environment.target * 4):
        if outcome.completed:
            break
        aliases = available_aliases(outcome.observation, forbidden)
        if not aliases:
            return None, None, len(actions)
        proposal = [weights.get(alias, 0.0) for alias in aliases]
        if sum(proposal) <= 0:
            proposal = [1.0] * len(aliases)
        alias = rng.choices(aliases, weights=proposal, k=1)[0]
        actions.append(alias)
        outcome = environment.step(ActionRecord(name=alias))

    if not outcome.completed:
        return None, None, len(actions)
    trajectory = Trajectory(
        trajectory_id=trajectory_id,
        task_id=environment.task_spec.task_id,
        source="agent",
        steps=tuple(
            TrajectoryStep(index=index, action=ActionRecord(name=alias))
            for index, alias in enumerate(actions)
        ),
    )
    return trajectory, environment.objective_value(), len(actions)


def search_for_optimum(
    environment: Any,
    optimum_value: float,
    weights: dict[str, float],
    *,
    seed: int,
    max_episodes: int,
    condition_id: str,
) -> SearchOutcome:
    rng = random.Random(seed)
    best: float | None = None
    first_optimum: int | None = None
    search_actions = 0

    for episode in range(1, max_episodes + 1):
        trajectory, measured, action_count = _sample_candidate(
            environment.fresh(),
            weights,
            rng,
            trajectory_id=(
                f"search:{condition_id}:{environment.task_spec.task_id}:"
                f"s{seed}:e{episode}"
            ),
        )
        search_actions += action_count
        if trajectory is None or measured is None:
            continue
        if best is None or measured < best:
            result = evaluate_trajectory(environment.fresh(), trajectory)
            if not result.performance_eligible_for(environment.task_spec):
                raise RuntimeError("candidate failed independent validity replay")
            if result.performance_value != measured:
                raise RuntimeError("candidate measurement disagrees with replay")
            best = measured
        if measured == optimum_value:
            first_optimum = episode
            break

    return SearchOutcome(first_optimum, best, search_actions)


def _simplex_grid() -> tuple[dict[str, float], ...]:
    values = (0.0, 0.25, 0.5, 0.75, 1.0)
    mixes: list[dict[str, float]] = []
    for optimum in values:
        for pooled in values:
            delta = round(1.0 - optimum - pooled, 2)
            if delta in values:
                mixes.append(
                    {"optimum": optimum, "pooled": pooled, "delta": delta}
                )
    return tuple(mixes)


def _evaluate_mix(
    tasks: tuple[tuple[Any, float], ...],
    models: dict[str, InteractionScorer],
    mix: dict[str, float],
    *,
    family_index: int,
    replicates: int,
    max_episodes: int,
    probes_per_action: int,
) -> dict[str, float]:
    totals: list[int] = []
    interactions: list[int] = []
    successes = 0
    base = 1_500_000 + family_index * 10_000

    for replicate in range(replicates):
        episode_total = 0
        interaction_total = 0
        for task_index, (environment, optimum) in enumerate(tasks):
            probe = probe_action_effects(
                environment,
                seed=base + replicate * 1_000 + task_index + 123,
                probes_per_action=probes_per_action,
            )
            weights = action_weights(
                environment,
                probe,
                models,
                mix,
                temperature=DEFAULT_TEMPERATURE,
            )
            search = search_for_optimum(
                environment,
                optimum,
                weights,
                seed=base + replicate * 100 + task_index,
                max_episodes=max_episodes,
                condition_id="cv",
            )
            hit = search.first_optimum_episode or (max_episodes + 1)
            episode_total += hit
            interaction_total += probe.interactions + search.search_actions
            successes += search.first_optimum_episode is not None
        totals.append(episode_total)
        interactions.append(interaction_total)

    return {
        "median_total_episodes": float(statistics.median(totals)),
        "median_total_interactions": float(statistics.median(interactions)),
        "success_rate": successes / (replicates * len(tasks)),
    }


def select_robust_mix(
    development: dict[str, tuple[AdaptiveTrackBundle, ...]],
    probe_cache: dict[str, ProbeResult],
    *,
    validation_tasks: int,
    replicates: int,
    max_episodes: int,
    probes_per_action: int,
    model_epochs: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Choose a mixture without inspecting the final family.

    The primary selection criterion is worst-family median environment interactions, not average
    benchmark score. This deliberately penalizes brittle methods.
    """

    fold_models: dict[str, dict[str, InteractionScorer]] = {}
    for holdout_family in DEVELOPMENT_FAMILIES:
        training = tuple(
            bundle
            for family in DEVELOPMENT_FAMILIES
            if family != holdout_family
            for bundle in development[family]
        )
        fold_models[holdout_family] = {
            kind: train_interaction_model(
                training,
                probe_cache,
                kind,
                epochs=model_epochs,
                model_seed=MODEL_SEED,
            )
            for kind in ("optimum", "pooled", "delta")
        }

    candidates: list[dict[str, Any]] = []
    for mix in _simplex_grid():
        folds: list[dict[str, Any]] = []
        for family_index, family in enumerate(DEVELOPMENT_FAMILIES):
            tasks = tuple(
                (
                    bundle.environment,
                    bundle.ladder.stage("optimum").performance_value,
                )
                for bundle in development[family][:validation_tasks]
            )
            result = _evaluate_mix(
                tasks,
                fold_models[family],
                mix,
                family_index=family_index,
                replicates=replicates,
                max_episodes=max_episodes,
                probes_per_action=probes_per_action,
            )
            folds.append({"family": family, **result})

        worst = max(fold["median_total_interactions"] for fold in folds)
        mean_interactions = statistics.mean(
            fold["median_total_interactions"] for fold in folds
        )
        mean_success = statistics.mean(fold["success_rate"] for fold in folds)
        candidates.append(
            {
                "mix": mix,
                "worst_family_median_interactions": worst,
                "mean_family_median_interactions": mean_interactions,
                "mean_success_rate": mean_success,
                "folds": folds,
            }
        )

    candidates.sort(
        key=lambda row: (
            row["worst_family_median_interactions"],
            row["mean_family_median_interactions"],
            -row["mean_success_rate"],
        )
    )
    return dict(candidates[0]["mix"]), candidates


def _manifest(
    condition_id: str,
    bundles: tuple[AdaptiveTrackBundle, ...],
    final_ids: tuple[str, ...],
    labels: tuple[str, ...],
    probes_per_action: int,
) -> ExposureManifest:
    exposed = tuple(
        bundle.ladder.stage(label).trajectory_id
        for bundle in bundles
        for label in labels
    )
    return ExposureManifest(
        condition_id=condition_id,
        train_task_ids=(tuple(bundle.ladder.task_id for bundle in bundles) if labels else ()),
        held_out_task_ids=final_ids,
        exposed_trajectory_ids=exposed,
        exposed_stage_labels=labels,
        privileged_state_access=False,
        structured_constraint_access=True,
        metadata={
            "milestone": 5,
            "action_descriptors_visible": False,
            "interaction_probe_access": True,
            "probes_per_action": probes_per_action,
            "final_family_used_for_model_selection": False,
        },
    )


def _paired(left: list[int], right: list[int]) -> dict[str, int]:
    return {
        "left_wins": sum(a < b for a, b in zip(left, right)),
        "ties": sum(a == b for a, b in zip(left, right)),
        "right_wins": sum(a > b for a, b in zip(left, right)),
    }


def run_experiment(
    *,
    development_tasks_per_family: int = DEVELOPMENT_TASKS_PER_FAMILY,
    final_task_count: int = FINAL_TASK_COUNT,
    replicates: int = DEFAULT_REPLICATES,
    max_episodes: int = DEFAULT_MAX_EPISODES,
    probes_per_action: int = PROBES_PER_ACTION,
    cv_validation_tasks: int = 4,
    cv_replicates: int = 2,
    cv_max_episodes: int = 60,
    cv_model_epochs: int = CV_MODEL_EPOCHS,
    final_model_epochs: int = FINAL_MODEL_EPOCHS,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    development = build_development_bundles(development_tasks_per_family)
    all_development = tuple(
        bundle for family in DEVELOPMENT_FAMILIES for bundle in development[family]
    )
    validate_training_ladders(all_development)
    training_probes = build_probe_cache(
        all_development,
        probes_per_action=probes_per_action,
    )

    selected_mix, candidates = select_robust_mix(
        development,
        training_probes,
        validation_tasks=min(cv_validation_tasks, development_tasks_per_family),
        replicates=cv_replicates,
        max_episodes=cv_max_episodes,
        probes_per_action=probes_per_action,
        model_epochs=cv_model_epochs,
    )

    final_tasks = held_out_combo_tasks(final_task_count, FINAL_GENERATOR_SEED)
    final_ids = tuple(environment.task_spec.task_id for environment, _ in final_tasks)
    if any(bundle.environment.family == FINAL_FAMILY for bundle in all_development):
        raise RuntimeError("final family leaked into development data")

    models = {
        kind: train_interaction_model(
            all_development,
            training_probes,
            kind,
            epochs=final_model_epochs,
            model_seed=MODEL_SEED,
        )
        for kind in ("delta", "shuffled", "pooled", "optimum")
    }
    conditions: dict[str, tuple[dict[str, float] | None, tuple[str, ...]]] = {
        "uniform": (None, ()),
        "frontier_to_optimum_delta": ({"delta": 1.0}, ("frontier", "optimum")),
        "shuffled_transition_direction": ({"shuffled": 1.0}, ("frontier", "optimum")),
        "pooled_frontier_optimum": ({"pooled": 1.0}, ("frontier", "optimum")),
        "imitate_optimum": ({"optimum": 1.0}, ("optimum",)),
        "robust_selected_mix": (selected_mix, ("frontier", "optimum")),
    }

    final_probes: dict[tuple[int, str], ProbeResult] = {}
    for replicate in range(replicates):
        for task_index, (environment, _) in enumerate(final_tasks):
            final_probes[(replicate, environment.task_spec.task_id)] = probe_action_effects(
                environment,
                seed=DEFAULT_SEARCH_SEED + replicate * 1_000 + task_index + 123,
                probes_per_action=probes_per_action,
            )

    reports: dict[str, dict[str, Any]] = {}
    paired_totals: dict[str, list[int]] = {}
    for condition_id, (mix, labels) in conditions.items():
        episode_totals: list[int] = []
        interaction_totals: list[int] = []
        successes = 0
        probe_counts: list[int] = []

        for replicate in range(replicates):
            episode_total = 0
            interaction_total = 0
            for task_index, (environment, optimum) in enumerate(final_tasks):
                probe = final_probes[(replicate, environment.task_spec.task_id)]
                probe_counts.append(probe.interactions)
                weights = action_weights(
                    environment,
                    probe,
                    None if mix is None else models,
                    mix,
                    temperature=DEFAULT_TEMPERATURE,
                )
                search = search_for_optimum(
                    environment,
                    optimum,
                    weights,
                    seed=DEFAULT_SEARCH_SEED + replicate * 100 + task_index,
                    max_episodes=max_episodes,
                    condition_id=condition_id,
                )
                hit = search.first_optimum_episode or (max_episodes + 1)
                episode_total += hit
                interaction_total += probe.interactions + search.search_actions
                successes += search.first_optimum_episode is not None
            episode_totals.append(episode_total)
            interaction_totals.append(interaction_total)

        paired_totals[condition_id] = episode_totals
        reports[condition_id] = {
            "mix": mix,
            "median_total_episodes": float(statistics.median(episode_totals)),
            "mean_total_episodes": statistics.mean(episode_totals),
            "held_out_task_success_rate": successes / (replicates * len(final_tasks)),
            "median_total_environment_interactions": float(
                statistics.median(interaction_totals)
            ),
            "mean_probe_interactions_per_task": statistics.mean(probe_counts),
            "exposure_manifest": _manifest(
                condition_id,
                all_development,
                final_ids,
                labels,
                probes_per_action,
            ).model_dump(mode="json"),
        }

    delta = paired_totals["frontier_to_optimum_delta"]
    comparisons = {
        f"delta_vs_{condition_id}": _paired(delta, totals)
        for condition_id, totals in paired_totals.items()
        if condition_id != "frontier_to_optimum_delta"
    }

    return {
        "experiment": "milestone5_interaction_inferred_transfer_v1",
        "development_families": list(DEVELOPMENT_FAMILIES),
        "final_family": FINAL_FAMILY,
        "development_tasks_per_family": development_tasks_per_family,
        "final_task_count": final_task_count,
        "replicates": replicates,
        "max_episodes": max_episodes,
        "probes_per_action": probes_per_action,
        "probe_feature_count": PROBE_FEATURE_COUNT,
        "model": {
            "hidden_widths": [48, 24],
            "model_seed": MODEL_SEED,
            "action_descriptors_visible": False,
            "action_alias_input": False,
            "family_id_input": False,
        },
        "method_selection": {
            "criterion": (
                "minimize worst leave-one-development-family median environment interactions; "
                "tie-break mean interactions, then mean success"
            ),
            "selected_mix": selected_mix,
            "top_candidates": candidates[:3],
            "final_family_consulted": False,
        },
        "conditions": reports,
        "comparisons": comparisons,
        "note": (
            "Synthetic interaction-inference experiment. Oracle action descriptors are hidden, "
            "but structured constraint access and compact state observations remain."
        ),
    }


def main(output: str | None = None) -> None:
    report = run_experiment()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if output is None:
        print(rendered)
    else:
        Path(output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
