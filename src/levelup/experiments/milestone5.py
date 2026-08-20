"""Milestone 5: infer action affordances from interaction before learning to improve.

The learner never receives hidden action descriptors. It gets opaque aliases, observable state,
and a fixed probe budget. Probe transitions are summarized into empirical effect vectors, which
feed the same small neural scoring family used for controlled reward/target comparisons.

Method selection is restricted to development mechanic families. The final Overdrive family is
excluded until the reward mixture has been frozen.
"""

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

from levelup.core.experiment import ExposureManifest
from levelup.core.trajectory import ActionRecord, Trajectory, TrajectoryStep
from levelup.envs.adaptive_track import (
    DEVELOPMENT_FAMILIES,
    FINAL_FAMILY,
    AdaptiveTrack,
    AdaptiveTrackBundle,
    collect_adaptive_bundles,
    held_out_adaptive_tasks,
)
from levelup.evaluation import evaluate_trajectory

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

TRANSITION_FEATURE_NAMES = (
    "pre_progress_fraction",
    "pre_remaining_fraction",
    "pre_resource_fraction",
    "pre_pressure_fraction",
    "delta_progress_scaled",
    "delta_ticks_scaled",
    "delta_resource_fraction",
    "delta_pressure_fraction",
    "post_resource_fraction",
    "post_pressure_fraction",
    "completed",
    "available_action_fraction",
)
SUMMARY_STATISTICS = ("mean", "std", "min", "max")
PROBE_FEATURE_COUNT = len(TRANSITION_FEATURE_NAMES) * len(SUMMARY_STATISTICS) + 1

TargetKind = Literal["delta", "shuffled", "pooled", "optimum"]


class InteractionScorer(nn.Module):
    """Score actions from empirically observed effects, not hidden action semantics."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(PROBE_FEATURE_COUNT, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    features: dict[str, tuple[float, ...]]
    interactions: int


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    first_optimum_episode: int | None
    best_performance: float | None
    search_actions: int


def _stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _structured_forbidden_aliases(environment: AdaptiveTrack) -> set[str]:
    aliases: set[str] = set()
    for constraint in environment.task_spec.constraints:
        if constraint.verifier_id == "never_use_action":
            alias = constraint.verifier_config.get("action")
            if isinstance(alias, str):
                aliases.add(alias)
    return aliases


def _available_aliases(observation: Any, forbidden: set[str]) -> list[str]:
    if not isinstance(observation, dict):
        raise RuntimeError("AdaptiveTrack observation must be a dictionary")
    raw = observation.get("available_actions")
    if not isinstance(raw, list):
        raise RuntimeError("AdaptiveTrack observation is missing available_actions")
    aliases: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeError("available action must be an object")
        if set(item) != {"alias"}:
            raise RuntimeError("Milestone 5 forbids structured action descriptors")
        alias = item.get("alias")
        if isinstance(alias, str) and alias not in forbidden:
            aliases.append(alias)
    return aliases


def _state_fields(observation: Any) -> tuple[float, float, float, float, float]:
    if not isinstance(observation, dict):
        raise RuntimeError("AdaptiveTrack observation must be a dictionary")
    progress = observation.get("progress")
    target = observation.get("target")
    elapsed = observation.get("elapsed_ticks")
    resource = observation.get("resource_fraction")
    pressure = observation.get("pressure_fraction")
    if not isinstance(progress, int) or not isinstance(target, int) or target <= 0:
        raise RuntimeError("invalid progress observation")
    if not isinstance(elapsed, int):
        raise RuntimeError("invalid elapsed-tick observation")
    if not isinstance(resource, (int, float)) or not isinstance(pressure, (int, float)):
        raise RuntimeError("invalid resource or pressure observation")
    return (
        float(progress),
        float(target),
        float(elapsed),
        float(resource),
        float(pressure),
    )


def _transition_vector(
    before: Any,
    after: Any,
    *,
    completed: bool,
    available_count: int,
) -> tuple[float, ...]:
    pre_progress, target, pre_elapsed, pre_resource, pre_pressure = _state_fields(before)
    post_progress, _, post_elapsed, post_resource, post_pressure = _state_fields(after)
    return (
        pre_progress / target,
        (target - pre_progress) / target,
        pre_resource,
        pre_pressure,
        (post_progress - pre_progress) / 3.0,
        (post_elapsed - pre_elapsed) / 13.0,
        post_resource - pre_resource,
        post_pressure - pre_pressure,
        post_resource,
        post_pressure,
        float(completed),
        available_count / 5.0,
    )


def _summarize(rows: list[tuple[float, ...]], expected: int) -> tuple[float, ...]:
    if not rows:
        raise RuntimeError("probe produced no observations for an action")
    tensor = torch.tensor(rows, dtype=torch.float32)
    summary = torch.cat(
        (
            tensor.mean(dim=0),
            tensor.std(dim=0, unbiased=False),
            tensor.min(dim=0).values,
            tensor.max(dim=0).values,
            torch.tensor([len(rows) / expected], dtype=torch.float32),
        )
    )
    if len(summary) != PROBE_FEATURE_COUNT:
        raise RuntimeError("unexpected probe feature width")
    return tuple(float(value) for value in summary.tolist())


def probe_action_effects(
    environment: AdaptiveTrack,
    *,
    seed: int,
    probes_per_action: int = PROBES_PER_ACTION,
) -> ProbeResult:
    """Actively infer each permitted action's observable consequences.

    Random prefixes create different states. The target action is then executed only when it is
    available. Prefix and target actions both count toward interaction cost.
    """

    if probes_per_action < 1:
        raise ValueError("probes_per_action must be positive")
    rng = random.Random(seed)
    forbidden = _structured_forbidden_aliases(environment)
    target_aliases = tuple(
        alias for alias in environment.valid_action_aliases if alias not in forbidden
    )
    observations: dict[str, list[tuple[float, ...]]] = {
        alias: [] for alias in target_aliases
    }
    interactions = 0

    for target_alias in target_aliases:
        attempts = 0
        while len(observations[target_alias]) < probes_per_action and attempts < 400:
            attempts += 1
            probe_env = environment.fresh()
            outcome = probe_env.reset()
            prefix_length = rng.randrange(0, max(1, environment.target // 2 + 1))

            for _ in range(prefix_length):
                if outcome.completed:
                    break
                aliases = _available_aliases(outcome.observation, forbidden)
                if not aliases:
                    break
                alias = rng.choice(aliases)
                outcome = probe_env.step(ActionRecord(name=alias))
                interactions += 1

            if outcome.completed:
                continue
            aliases = _available_aliases(outcome.observation, forbidden)
            if target_alias not in aliases:
                continue

            before = outcome.observation
            before_count = len(aliases)
            after = probe_env.step(ActionRecord(name=target_alias))
            interactions += 1
            observations[target_alias].append(
                _transition_vector(
                    before,
                    after.observation,
                    completed=after.completed,
                    available_count=before_count,
                )
            )

        if len(observations[target_alias]) != probes_per_action:
            raise RuntimeError(
                f"probe could not collect {probes_per_action} transitions for {target_alias!r}"
            )

    return ProbeResult(
        features={
            alias: _summarize(rows, probes_per_action)
            for alias, rows in observations.items()
        },
        interactions=interactions,
    )


def _frequency(trajectory: Trajectory) -> dict[str, float]:
    counts = Counter(step.action.name for step in trajectory.steps)
    total = len(trajectory.steps)
    return {alias: count / total for alias, count in counts.items()}


def _development_seed(family_index: int) -> int:
    return 900 + family_index * 100


def build_development_bundles(
    tasks_per_family: int = DEVELOPMENT_TASKS_PER_FAMILY,
) -> dict[str, tuple[AdaptiveTrackBundle, ...]]:
    return {
        family: collect_adaptive_bundles(
            family,
            tasks_per_family,
            _development_seed(index),
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
    probes_per_action: int = PROBES_PER_ACTION,
) -> dict[str, ProbeResult]:
    cache: dict[str, ProbeResult] = {}
    for bundle in bundles:
        task_id = bundle.environment.task_spec.task_id
        seed = 700_000 + bundle.environment.generator_seed * 100 + bundle.environment.task_index
        cache[task_id] = probe_action_effects(
            bundle.environment,
            seed=seed,
            probes_per_action=probes_per_action,
        )
    return cache


def _training_examples(
    bundles: tuple[AdaptiveTrackBundle, ...],
    probe_cache: dict[str, ProbeResult],
    target_kind: TargetKind,
    *,
    label_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(label_seed)
    rows: list[tuple[float, ...]] = []
    targets: list[float] = []

    for bundle in bundles:
        frontier_frequency = _frequency(bundle.trajectory_for("frontier"))
        optimum_frequency = _frequency(bundle.trajectory_for("optimum"))
        reverse = target_kind == "shuffled" and rng.random() >= 0.5
        probe = probe_cache[bundle.environment.task_spec.task_id]

        for alias in bundle.environment.valid_action_aliases:
            before = frontier_frequency.get(alias, 0.0)
            after = optimum_frequency.get(alias, 0.0)
            delta = after - before
            if target_kind == "delta":
                target = delta
            elif target_kind == "shuffled":
                target = -delta if reverse else delta
            elif target_kind == "pooled":
                target = (before + after) / 2.0
            elif target_kind == "optimum":
                target = after
            else:
                raise ValueError(f"unsupported target kind: {target_kind!r}")
            rows.append(probe.features[alias])
            targets.append(target)

    return (
        torch.tensor(rows, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )


def train_model(
    bundles: tuple[AdaptiveTrackBundle, ...],
    probe_cache: dict[str, ProbeResult],
    target_kind: TargetKind,
    *,
    epochs: int,
    model_seed: int = MODEL_SEED,
) -> InteractionScorer:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(model_seed)
    model = InteractionScorer()
    features, targets = _training_examples(
        bundles,
        probe_cache,
        target_kind,
        label_seed=model_seed + 111,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = model(features)
        loss = torch.mean((prediction - targets) ** 2)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _normalized_scores(model: InteractionScorer, features: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        scores = model(features)
    deviation = scores.std(unbiased=False)
    if len(scores) > 1 and float(deviation) > 1e-6:
        return (scores - scores.mean()) / deviation
    return torch.zeros_like(scores)


def action_weights(
    environment: AdaptiveTrack,
    probe: ProbeResult,
    models: dict[str, InteractionScorer] | None,
    mix: dict[str, float] | None,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, float]:
    aliases = environment.valid_action_aliases
    if models is None or mix is None:
        return {alias: 1.0 for alias in aliases}

    features = torch.tensor(
        [probe.features[alias] for alias in aliases],
        dtype=torch.float32,
    )
    combined = torch.zeros(len(aliases), dtype=torch.float32)
    for target_kind, weight in mix.items():
        if weight <= 0:
            continue
        combined += weight * _normalized_scores(models[target_kind], features)
    probabilities = torch.softmax(combined / temperature, dim=0)
    return {
        alias: float(probability)
        for alias, probability in zip(aliases, probabilities.tolist())
    }


def _sample_candidate(
    environment: AdaptiveTrack,
    weights: dict[str, float],
    rng: random.Random,
    trajectory_id: str,
) -> tuple[Trajectory, float, int] | None:
    outcome = environment.reset()
    forbidden = _structured_forbidden_aliases(environment)
    actions: list[str] = []

    for _ in range(environment.target * 4):
        if outcome.completed:
            break
        aliases = _available_aliases(outcome.observation, forbidden)
        if not aliases:
            return None
        alias_weights = [weights.get(alias, 0.0) for alias in aliases]
        if sum(alias_weights) <= 0:
            alias_weights = [1.0] * len(aliases)
        alias = rng.choices(aliases, weights=alias_weights, k=1)[0]
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
    return trajectory, environment.objective_value(), len(actions)


def search_for_optimum(
    environment: AdaptiveTrack,
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
        sampled = _sample_candidate(
            environment.fresh(),
            weights,
            rng,
            trajectory_id=(
                f"search:{condition_id}:{environment.task_spec.task_id}:"
                f"s{seed}:e{episode}"
            ),
        )
        if sampled is None:
            continue
        trajectory, measured, action_count = sampled
        search_actions += action_count
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

    return SearchOutcome(
        first_optimum_episode=first_optimum,
        best_performance=best,
        search_actions=search_actions,
    )


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


def _evaluate_mix_on_tasks(
    tasks: tuple[tuple[AdaptiveTrack, float], ...],
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
        total = 0
        interaction_total = 0
        for task_index, (environment, optimum) in enumerate(tasks):
            probe = probe_action_effects(
                environment,
                seed=base + replicate * 1_000 + task_index + 123,
                probes_per_action=probes_per_action,
            )
            weights = action_weights(environment, probe, models, mix)
            outcome = search_for_optimum(
                environment,
                optimum,
                weights,
                seed=base + replicate * 100 + task_index,
                max_episodes=max_episodes,
                condition_id="cv",
            )
            hit = outcome.first_optimum_episode or (max_episodes + 1)
            total += hit
            interaction_total += probe.interactions + outcome.search_actions
            if outcome.first_optimum_episode is not None:
                successes += 1
        totals.append(total)
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
    validation_tasks: int = 4,
    replicates: int = 2,
    max_episodes: int = 60,
    probes_per_action: int = PROBES_PER_ACTION,
    model_epochs: int = CV_MODEL_EPOCHS,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Select a reward mixture without consulting the final family.

    Selection minimizes the worst leave-one-family-out median interaction count. Mean interaction
    count and then mean success rate break ties. This favors robust transfer rather than a method
    that is spectacular on one familiar mechanic and brittle on another.
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
            kind: train_model(
                training,
                probe_cache,
                kind,
                epochs=model_epochs,
            )
            for kind in ("optimum", "pooled", "delta")
        }

    candidates: list[dict[str, Any]] = []
    for mix in _simplex_grid():
        folds: list[dict[str, Any]] = []
        for family_index, family in enumerate(DEVELOPMENT_FAMILIES):
            validation = tuple(
                (
                    bundle.environment,
                    bundle.ladder.stage("optimum").performance_value,
                )
                for bundle in development[family][:validation_tasks]
            )
            result = _evaluate_mix_on_tasks(
                validation,
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
    exposed: list[str] = []
    for bundle in bundles:
        for label in labels:
            exposed.append(bundle.ladder.stage(label).trajectory_id)
    return ExposureManifest(
        condition_id=condition_id,
        train_task_ids=(
            tuple(bundle.ladder.task_id for bundle in bundles) if labels else ()
        ),
        held_out_task_ids=final_ids,
        exposed_trajectory_ids=tuple(exposed),
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


def _paired_comparison(left: list[int], right: list[int]) -> dict[str, int]:
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
    """Run development-only method selection, then a frozen final-family evaluation."""

    if replicates < 1 or max_episodes < 1:
        raise ValueError("replicates and max_episodes must be positive")
    torch.set_num_threads(1)

    development = build_development_bundles(development_tasks_per_family)
    all_development = tuple(
        bundle for family in DEVELOPMENT_FAMILIES for bundle in development[family]
    )
    validate_training_ladders(all_development)
    development_probe_cache = build_probe_cache(
        all_development,
        probes_per_action=probes_per_action,
    )

    selected_mix, cv_candidates = select_robust_mix(
        development,
        development_probe_cache,
        validation_tasks=min(cv_validation_tasks, development_tasks_per_family),
        replicates=cv_replicates,
        max_episodes=cv_max_episodes,
        probes_per_action=probes_per_action,
        model_epochs=cv_model_epochs,
    )

    final_tasks = held_out_adaptive_tasks(
        FINAL_FAMILY,
        final_task_count,
        FINAL_GENERATOR_SEED,
    )
    final_ids = tuple(environment.task_spec.task_id for environment, _ in final_tasks)
    if any(FINAL_FAMILY in bundle.environment.family for bundle in all_development):
        raise RuntimeError("final family leaked into development data")

    final_models = {
        kind: train_model(
            all_development,
            development_probe_cache,
            kind,
            epochs=final_model_epochs,
        )
        for kind in ("delta", "shuffled", "pooled", "optimum")
    }

    conditions: dict[str, tuple[dict[str, float] | None, tuple[str, ...]]] = {
        "uniform": (None, ()),
        "frontier_to_optimum_delta": ({"delta": 1.0}, ("frontier", "optimum")),
        "shuffled_transition_direction": (
            {"shuffled": 1.0},
            ("frontier", "optimum"),
        ),
        "pooled_frontier_optimum": ({"pooled": 1.0}, ("frontier", "optimum")),
        "imitate_optimum": ({"optimum": 1.0}, ("optimum",)),
        "robust_selected_mix": (selected_mix, ("frontier", "optimum")),
    }

    probe_cache: dict[tuple[int, str], ProbeResult] = {}
    for replicate in range(replicates):
        for task_index, (environment, _) in enumerate(final_tasks):
            probe_cache[(replicate, environment.task_spec.task_id)] = probe_action_effects(
                environment,
                seed=DEFAULT_SEARCH_SEED + replicate * 1_000 + task_index + 123,
                probes_per_action=probes_per_action,
            )

    condition_reports: dict[str, dict[str, Any]] = {}
    paired_totals: dict[str, list[int]] = {}
    for condition_id, (mix, labels) in conditions.items():
        total_episodes: list[int] = []
        total_interactions: list[int] = []
        successes = 0
        task_hits: dict[str, list[int]] = {
            environment.task_spec.task_id: [] for environment, _ in final_tasks
        }
        probe_counts: list[int] = []

        for replicate in range(replicates):
            episode_total = 0
            interaction_total = 0
            for task_index, (environment, optimum) in enumerate(final_tasks):
                probe = probe_cache[(replicate, environment.task_spec.task_id)]
                probe_counts.append(probe.interactions)
                weights = action_weights(
                    environment,
                    probe,
                    None if mix is None else final_models,
                    mix,
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
                task_hits[environment.task_spec.task_id].append(hit)
                episode_total += hit
                interaction_total += probe.interactions + search.search_actions
                if search.first_optimum_episode is not None:
                    successes += 1
            total_episodes.append(episode_total)
            total_interactions.append(interaction_total)

        paired_totals[condition_id] = total_episodes
        manifest = _manifest(
            condition_id,
            all_development,
            final_ids,
            labels,
            probes_per_action,
        )
        condition_reports[condition_id] = {
            "mix": mix,
            "median_total_episodes": float(statistics.median(total_episodes)),
            "mean_total_episodes": statistics.mean(total_episodes),
            "held_out_task_success_rate": successes / (replicates * len(final_tasks)),
            "median_total_environment_interactions": float(
                statistics.median(total_interactions)
            ),
            "mean_probe_interactions_per_task": statistics.mean(probe_counts),
            "per_task_median_episodes": {
                task_id: float(statistics.median(hits))
                for task_id, hits in task_hits.items()
            },
            "exposure_manifest": manifest.model_dump(mode="json"),
        }

    delta_totals = paired_totals["frontier_to_optimum_delta"]
    comparisons = {
        f"delta_vs_{condition}": _paired_comparison(delta_totals, totals)
        for condition, totals in paired_totals.items()
        if condition != "frontier_to_optimum_delta"
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
            "cv_epochs": cv_model_epochs,
            "final_epochs": final_model_epochs,
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
            "top_candidates": cv_candidates[:3],
            "final_family_consulted": False,
        },
        "conditions": condition_reports,
        "comparisons": comparisons,
        "note": (
            "Synthetic interaction-inference experiment. It removes oracle action descriptors "
            "but still uses structured constraint access and compact state observations."
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
