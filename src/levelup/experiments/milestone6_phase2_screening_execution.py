"""Execute one frozen, development-only Phase 2 screening unit.

The preparation pass owns training-data and model-artifact creation.  This module is
the deliberately small held-out boundary: it may load an already materialized model,
pay the declared probe, generate a complete fixed-budget candidate batch without an
evaluator, replay that batch independently, and only then ask the reporting oracle
for the post-hoc exact-optimum classification.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from typing import Any, Protocol

from levelup.envs.adaptive_track import optimal_path as adaptive_optimal_path
from levelup.envs.challenge_track import optimal_path as combo_optimal_path
from levelup.experiments.milestone6_baselines import (
    IndependentCandidateEvaluator,
    classify_exact_optimum,
    discover_affordances,
    evaluate_generated_search,
    generate_candidates_with_observable_policy,
    trajectory_content_sha256,
)
from levelup.experiments.milestone6_phase2 import (
    _condition,
    _environment,
    _forbidden_aliases,
    _task,
)
from levelup.experiments.milestone6_phase2_screening import (
    LEARNED_BASES,
    base_condition_id,
    validate_screening_child_config,
)
from levelup.experiments.milestone6_phase2_screening_execution_artifacts import (
    ScreeningModelCache,
    prepare_unit_model,
)
from levelup.experiments.runner.config import ExperimentConfig, scientific_config_sha256
from levelup.experiments.runner.records import (
    PhaseAccounting,
    PlannedUnit,
    ResourceAccounting,
    SharedArtifactReference,
    UnitOutcome,
    UnitPayload,
)
from levelup.experiments.runner.storage import RunStore
from levelup.learning.state_conditioned import AffordanceTable

PreparationEvent = Callable[[str], None]
OracleProvider = Callable[[Any, str], float]

SCREENING_CANDIDATE_EPISODES = 150
SCREENING_ADAPTATION_ACTION_CAP = 2048
SCREENING_MAX_ACTIONS_PER_EPISODE = 64
SCREENING_PROBE_ACTIONS_PER_ATTEMPT = 16
SCREENING_PROBE_ACTION_CAP = 64
FAILURE_SENTINEL = SCREENING_ADAPTATION_ACTION_CAP + 1


class _Fold(Protocol):
    family_id: str
    config: ExperimentConfig
    store: RunStore


def _default_optimum_provider(environment: Any, family_id: str) -> float:
    """Reporting-only optimum lookup; called strictly after replay completes."""

    if family_id == "combo":
        return float(combo_optimal_path(environment)[0])
    return float(adaptive_optimal_path(environment)[0])


def _generated_candidates_sha256(generated: Any) -> str:
    """Hash only generated candidates, never post-hoc replay or oracle values."""

    body = [
        {
            "episode": item.episode,
            "adaptation_actions": item.adaptation_actions,
            "trajectory_sha256": trajectory_content_sha256(item.trajectory),
        }
        for item in generated.candidates
    ]
    from levelup.experiments.runner.config import canonical_json_bytes

    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _cache_instance() -> ScreeningModelCache:
    return ScreeningModelCache()


def _prepare_model(
    fold: _Fold,
    planned: PlannedUnit,
    condition: Any,
    *,
    model_cache: ScreeningModelCache,
) -> Any:
    """Call the typed artifact gate without allowing training in this module."""
    return prepare_unit_model(fold, planned, condition, model_cache)


def _report_value(report: Any, name: str) -> int:
    value = getattr(report, name, 0) if report is not None else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"screening model report has invalid {name}")
    return value


def _validate_references(
    condition_id: str,
    prepared: Any,
) -> tuple[SharedArtifactReference, ...]:
    base = base_condition_id(condition_id)
    references = tuple(getattr(prepared, "references", ())) if prepared is not None else ()
    if base is None:
        if prepared is not None:
            raise RuntimeError("non-learned screening control received a model")
        if references:
            raise RuntimeError("screening control received shared model references")
        return ()
    if base not in LEARNED_BASES or prepared is None:
        raise RuntimeError("learned screening unit lacks its prepared model")
    if len(references) != 3 or not all(isinstance(item, SharedArtifactReference) for item in references):
        raise RuntimeError("learned screening unit must expose exactly three typed references")
    if tuple(item.kind for item in references) != (
        "training_data_evidence",
        "training_data_view",
        "training_artifact",
    ):
        raise RuntimeError("learned screening references are not in canonical kind order")
    return references


def _validate_fold_and_unit(fold: _Fold, planned: PlannedUnit) -> tuple[Any, Any, Any]:
    if not bool(getattr(fold.store, "_execution_ready", False)):
        raise RuntimeError("screening fold store is not execution-ready")
    config = fold.config
    if not isinstance(config, ExperimentConfig):
        raise RuntimeError("screening fold has no typed experiment config")
    if config.split.final_tasks or any(
        "final" in condition.execution_phases for condition in config.conditions
    ):
        raise RuntimeError("screening execution cannot contain final tasks")
    validate_screening_child_config(config)
    if scientific_config_sha256(config) != getattr(fold.store, "config_sha256", None):
        raise RuntimeError("screening fold store config identity differs from its config")
    if planned.key.phase != "validation":
        raise RuntimeError("screening execution accepts validation units only")
    expected = getattr(fold.store, "expected", None)
    if expected is None or getattr(expected, "config_sha256", None) != scientific_config_sha256(config):
        raise RuntimeError("screening fold expected-unit authority is not exact")
    try:
        authoritative = fold.store.planned_unit(planned.unit_id)
    except (AttributeError, KeyError, RuntimeError, ValueError) as exc:
        raise RuntimeError("screening unit is not in the fold expected matrix") from exc
    if authoritative != planned:
        raise RuntimeError("screening planned unit differs from fold expected authority")
    family_id = str(config.parameters.get("heldout_family_id"))
    if (
        str(getattr(fold, "family_id", "")) != family_id
        or planned.key.family_id != family_id
        or planned.key.task_id not in {task.task_id for task in config.split.validation_tasks}
    ):
        raise RuntimeError("screening unit does not belong to this held-out fold")
    task = _task(config, planned.key.task_id)
    if (
        task.family_id != family_id
        or task.task_id != planned.key.task_id
        or task.task_index != planned.key.task_index
        or planned.seeds.environment_seed != task.environment_reset_seed
    ):
        raise RuntimeError("screening unit task identity differs from fold authority")
    condition = _condition(config, planned.key.condition_id)
    if condition.execution_phases != ("validation",):
        raise RuntimeError("screening condition is not validation-only")
    if config.parameters.get("candidate_episodes") != SCREENING_CANDIDATE_EPISODES:
        raise RuntimeError("screening candidate episode budget differs from frozen protocol")
    if config.parameters.get("adaptation_action_cap") != SCREENING_ADAPTATION_ACTION_CAP:
        raise RuntimeError("screening adaptation budget differs from frozen protocol")
    if config.parameters.get("maximum_actions_per_candidate_episode") != SCREENING_MAX_ACTIONS_PER_EPISODE:
        raise RuntimeError("screening episode action budget differs from frozen protocol")
    if config.parameters.get("probe_actions_per_attempt") != SCREENING_PROBE_ACTIONS_PER_ATTEMPT:
        raise RuntimeError("screening probe attempt budget differs from frozen protocol")
    if config.parameters.get("probe_action_cap") != SCREENING_PROBE_ACTION_CAP:
        raise RuntimeError("screening probe cap differs from frozen condition protocol")
    if condition.condition_id in {"A0-no-probe-uniform", "A1-paid-probe-uniform"}:
        expected_probe_cap = (
            0
            if condition.condition_id == "A0-no-probe-uniform"
            else SCREENING_PROBE_ACTION_CAP
        )
        if condition.parameters.get("probe_action_cap") != expected_probe_cap:
            raise RuntimeError("screening condition probe cap differs from frozen protocol")
    if condition.condition_id == "A0-no-probe-uniform" and condition.exposure.probe_interaction_access:
        raise RuntimeError("A0 cannot receive probe access")
    if condition.condition_id != "A0-no-probe-uniform" and not condition.exposure.probe_interaction_access:
        raise RuntimeError("paid-probe screening condition lacks probe access")
    return task, condition, config


def execute_screening_unit(
    fold: _Fold,
    planned: PlannedUnit,
    *,
    optimum_provider: OracleProvider = _default_optimum_provider,
    event: PreparationEvent | None = None,
    model_cache: ScreeningModelCache | None = None,
) -> UnitPayload:
    """Execute exactly one validated development screening unit."""

    task, condition, config = _validate_fold_and_unit(fold, planned)
    setup_started = time.perf_counter()
    environment = _environment(task)
    forbidden = _forbidden_aliases(environment)
    cache = model_cache if model_cache is not None else _cache_instance()
    prepared = _prepare_model(fold, planned, condition, model_cache=cache)
    setup_wall = time.perf_counter() - setup_started
    model = getattr(prepared, "model", None) if prepared is not None else None
    report = getattr(prepared, "report", None) if prepared is not None else None
    references = _validate_references(condition.condition_id, prepared)

    probe = None
    if condition.exposure.probe_interaction_access:
        probe = discover_affordances(
            environment,
            task_id=task.task_id,
            forbidden_aliases=forbidden,
            seed=planned.seeds.probe_seed,
            action_cap=SCREENING_PROBE_ACTION_CAP,
            target_samples_per_alias=int(config.parameters["probe_coverage_target_samples_per_alias"]),
            actions_per_attempt=SCREENING_PROBE_ACTIONS_PER_ATTEMPT,
        )
        affordances = probe.affordances
        prior_actions = probe.accounting.actions
    else:
        affordances = AffordanceTable(features={}, sample_counts={})
        prior_actions = 0

    temperature = float(condition.parameters.get("search_temperature", 0.9))
    generated = generate_candidates_with_observable_policy(
        environment,
        task_id=task.task_id,
        forbidden_aliases=forbidden,
        affordances=affordances,
        model=model,
        seed=planned.seeds.search_seed,
        temperature=temperature,
        max_episodes=SCREENING_CANDIDATE_EPISODES,
        max_actions_per_episode=SCREENING_MAX_ACTIONS_PER_EPISODE,
        total_adaptation_action_cap=SCREENING_ADAPTATION_ACTION_CAP,
        prior_adaptation_actions=prior_actions,
        condition_id=condition.condition_id,
    )
    generated_sha256 = _generated_candidates_sha256(generated)
    if event is not None:
        event("generation_complete")

    search = evaluate_generated_search(
        generated,
        IndependentCandidateEvaluator(environment),
    )
    if event is not None:
        event("candidate_evaluation_complete")
        event("optimum_oracle")
    oracle_started = time.perf_counter()
    optimum_performance = float(optimum_provider(environment, task.family_id))
    oracle_wall = time.perf_counter() - oracle_started
    exact = classify_exact_optimum(search, optimum_performance=optimum_performance)

    probe_attempts = probe.accounting.attempts if probe is not None else 0
    probe_resets = probe.accounting.resets if probe is not None else 0
    probe_actions = probe.accounting.actions if probe is not None else 0
    probe_wall = probe.accounting.wall_seconds if probe is not None else 0.0
    valid = bool(search.evaluated_candidates)
    completed = bool(generated.candidates)
    success = bool(exact.success)
    return UnitPayload(
        outcome=UnitOutcome(
            evaluator_ran=True,
            valid=valid,
            completed=completed,
            success=success,
            performance_metric_id="performance_value",
            performance_value=search.best_performance,
            performance_direction="minimize",
            first_valid_completion_episode=search.first_valid_episode,
            first_optimum_episode=exact.first_episode,
            first_optimum_adaptation_actions=exact.first_adaptation_actions,
            censored=not success,
            # The typed outcome records the actual fixed endpoint (2048).  The
            # selection reducer applies the declared +1 sentinel when it turns
            # censored failures into the scalar action metric.
            censoring_budget=None if success else SCREENING_ADAPTATION_ACTION_CAP,
            censoring_reason=None if success else "fixed_endpoint",
        ),
        accounting=ResourceAccounting(
            setup=PhaseAccounting(calls=1, wall_seconds=setup_wall),
            probes=PhaseAccounting(
                calls=probe_attempts,
                actions=probe_actions,
                environment_steps=probe_actions,
                resets=probe_resets,
                wall_seconds=probe_wall,
            ),
            training=PhaseAccounting(),
            search=PhaseAccounting(
                calls=1,
                episodes=search.accounting.episodes,
                actions=search.accounting.actions,
                environment_steps=search.accounting.actions,
                resets=search.accounting.resets,
                forward_passes=search.accounting.forward_passes,
                wall_seconds=search.accounting.generation_wall_seconds,
            ),
            replay=PhaseAccounting(
                calls=search.accounting.evaluator_calls,
                actions=search.accounting.evaluator_replay_actions,
                environment_steps=search.accounting.evaluator_replay_actions,
                resets=search.accounting.evaluator_calls,
                wall_seconds=search.accounting.evaluator_wall_seconds,
            ),
            evaluator=PhaseAccounting(calls=1, wall_seconds=oracle_wall),
        ),
        shared_artifacts=references,
        candidate_generation_sha256=generated_sha256,
        diagnostics={
            "development_screening": True,
            "first_optimum_adaptation_actions": exact.first_adaptation_actions,
            "unknown_affordance_decisions": search.accounting.unknown_affordance_decisions,
            "trainable_parameters": _report_value(report, "trainable_parameters"),
            "training_examples": _report_value(report, "training_examples"),
            "oracle_setup_calls": 1,
            "shared_training_artifact": base_condition_id(condition.condition_id) in LEARNED_BASES,
            "search_temperature": temperature,
        },
    )


__all__ = [
    "FAILURE_SENTINEL",
    "SCREENING_ADAPTATION_ACTION_CAP",
    "SCREENING_CANDIDATE_EPISODES",
    "SCREENING_MAX_ACTIONS_PER_EPISODE",
    "SCREENING_PROBE_ACTIONS_PER_ATTEMPT",
    "SCREENING_PROBE_ACTION_CAP",
    "ScreeningModelCache",
    "_default_optimum_provider",
    "_generated_candidates_sha256",
    "execute_screening_unit",
    "prepare_unit_model",
]
