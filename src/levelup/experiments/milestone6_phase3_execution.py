"""Execute one frozen, development-only Phase 3 unit.

The execution boundary is intentionally narrow.  A caller supplies an opaque
validated Phase 3 plan, the published model authority, the prepared model-store
root, and a canonical resolver for the held-out task.  Everything that affects
the scientific unit (condition, tuple, temperature, seeds, and budgets) is
derived from the planned unit.  Candidate generation is completed before an
independent replay, and the exact-optimum oracle is consulted only for the
post-hoc report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from levelup.envs.adaptive_track import optimal_path as adaptive_optimal_path
from levelup.envs.challenge_track import optimal_path as combo_optimal_path
from levelup.experiments.milestone6_baselines import (
    IndependentCandidateEvaluator,
    classify_exact_optimum,
    discover_affordances,
    evaluate_generated_search,
)
from levelup.experiments.milestone6_phase2 import (
    _environment,
    _forbidden_aliases,
)
from levelup.experiments.milestone6_phase2_screening import screening_child_configs
from levelup.experiments.milestone6_phase3_execution_models import (
    AuthorizedPhase3LoadedModel,
    open_authorized_phase3_model,
)
from levelup.experiments.milestone6_phase3_generation import (
    FROZEN_CANDIDATE_EPISODES,
    FROZEN_HISTORY_SHUFFLE_BASE,
    FROZEN_MAX_ACTIONS_PER_EPISODE,
    FROZEN_PROBE_ACTIONS,
    FROZEN_TOTAL_ADAPTATION_ACTION_CAP,
    H4_SHUFFLED_CONDITION,
    generate_phase3_candidates_with_observable_policy,
)
from levelup.experiments.milestone6_phase3_model_authority import (
    Phase3ModelArtifactAuthority,
)
from levelup.experiments.milestone6_phase3_plan import (
    Phase3PlannedUnit,
    ValidatedPhase3Plan,
)
from levelup.experiments.runner.config import TaskIdentity
from levelup.experiments.runner.records import (
    PhaseAccounting,
    ResourceAccounting,
    SharedArtifactReference,
    UnitOutcome,
    UnitPayload,
)

PROBE_ACTIONS_PER_ATTEMPT = 16
PROBE_COVERAGE_TARGET_SAMPLES_PER_ALIAS = 8
FAILURE_SENTINEL = FROZEN_TOTAL_ADAPTATION_ACTION_CAP + 1


class Phase3ExecutionEvent(Protocol):
    def __call__(self, name: str) -> None: ...


@dataclass(frozen=True, slots=True)
class Phase3ExecutionContext:
    """Typed authorities needed to execute one unit.

    The environment is reconstructed from the canonical Phase 2 development
    configs.  No task resolver or environment factory is caller-supplied, so a
    caller cannot substitute a generator seed or final task.
    """

    authority: Phase3ModelArtifactAuthority
    plan: ValidatedPhase3Plan
    artifact_output_root: str | Path


def _default_optimum_provider(environment: Any, family_id: str) -> float:
    """Reporting-only oracle; never called until independent replay completes."""

    if family_id == "combo":
        return float(combo_optimal_path(environment)[0])
    return float(adaptive_optimal_path(environment)[0])


def _resolve_planned_unit(
    context: Phase3ExecutionContext,
    planned_unit: Phase3PlannedUnit,
) -> Phase3PlannedUnit:
    if type(context) is not Phase3ExecutionContext:
        raise TypeError("Phase 3 execution requires the canonical typed context")
    if type(context.plan) is not ValidatedPhase3Plan:
        raise TypeError("Phase 3 execution requires an opaque validated plan")
    if type(planned_unit) is not Phase3PlannedUnit:
        raise TypeError("Phase 3 execution accepts one canonical Phase3PlannedUnit")
    if planned_unit.unit.key.phase != "validation":
        raise ValueError("Phase 3 execution accepts validation units only")
    if context.plan.plan.final_family_access:
        raise ValueError("Phase 3 execution plan permits final-family access")
    try:
        context.plan.require_unit(planned_unit)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("planned unit is not in the validated Phase 3 plan") from exc
    matches = [
        item
        for item in context.plan.plan.units
        if item.unit.unit_id == planned_unit.unit.unit_id
    ]
    if len(matches) != 1 or matches[0] != planned_unit:
        raise ValueError("planned unit identity differs from the frozen Phase 3 plan")
    return planned_unit


@lru_cache(maxsize=1)
def _canonical_validation_tasks() -> tuple[TaskIdentity, ...]:
    configs = screening_child_configs()
    if len(configs) != 6 or any(config.split.final_tasks for config in configs):
        raise ValueError("canonical Phase 3 task authority is not development-only")
    tasks = tuple(task for config in configs for task in config.split.validation_tasks)
    identities = {(task.family_id, task.task_id) for task in tasks}
    if len(tasks) != 48 or len(identities) != len(tasks):
        raise ValueError("canonical Phase 3 validation task matrix differs")
    return tasks


def _resolve_task(planned: Phase3PlannedUnit) -> TaskIdentity:
    matches = [
        task
        for task in _canonical_validation_tasks()
        if task.task_id == planned.unit.key.task_id
        and task.family_id == planned.heldout_family
    ]
    if len(matches) != 1:
        raise ValueError("frozen Phase 3 unit has no exact canonical validation task")
    task = matches[0]
    key = planned.unit.key
    if (
        task.task_id != key.task_id
        or task.family_id != key.family_id
        or task.task_index != key.task_index
        or task.family_id != planned.heldout_family
        or task.environment_reset_seed != planned.unit.seeds.environment_seed
        or task.environment_reset_seed != 0
    ):
        raise ValueError("resolved held-out task differs from the frozen unit")
    return task


def _temperature(planned: Phase3PlannedUnit) -> float:
    suffix = planned.tuple_id.rsplit("-", 1)[-1]
    values = {"t0p6": 0.6, "t0p9": 0.9, "t1p2": 1.2}
    try:
        return values[suffix]
    except KeyError as exc:
        raise ValueError("Phase 3 tuple has an unknown search temperature") from exc


def _diagnostics(
    generated: Any,
    probe: Any,
    model_report: Any,
    oracle_wall: float,
) -> dict[str, bool | int | float | None]:
    accounting = generated.accounting
    result: dict[str, bool | int | float | None] = {
        "development_phase3": True,
        "recurrent_steps": int(getattr(accounting, "recurrent_steps", 0)),
        "unknown_affordance_decisions": int(
            getattr(accounting, "unknown_affordance_decisions", 0)
        ),
        "model_trainable_parameters": int(model_report.trainable_parameters),
        "model_optimizer_steps": int(model_report.optimizer_steps),
        "model_forward_passes": int(model_report.forward_passes),
        "model_recurrent_steps": int(model_report.recurrent_steps),
        "model_training_examples": int(model_report.training_examples),
        "history_shuffle_claim_eligible": None,
        "history_shuffle_eligible_windows": 0,
        "history_shuffle_map_nonidentity_windows": 0,
        "history_shuffle_effective_tensor_changed_windows": 0,
        "history_shuffle_duplicate_vector_no_effect_windows": 0,
        "history_shuffle_unchanged_short_windows": 0,
        "oracle_wall_seconds": oracle_wall,
    }
    shuffle = getattr(generated, "history_shuffle", None)
    if shuffle is not None:
        result.update(
            {
                "history_shuffle_claim_eligible": bool(shuffle.claim_eligible),
                "history_shuffle_eligible_windows": int(shuffle.eligible_windows),
                "history_shuffle_map_nonidentity_windows": int(
                    shuffle.map_nonidentity_windows
                ),
                "history_shuffle_effective_tensor_changed_windows": int(
                    shuffle.effective_tensor_changed_windows
                ),
                "history_shuffle_duplicate_vector_no_effect_windows": int(
                    shuffle.duplicate_vector_no_effect_windows
                ),
                "history_shuffle_unchanged_short_windows": int(
                    shuffle.unchanged_short_windows
                ),
            }
        )
    if probe is not None:
        result["probe_attempts"] = int(probe.accounting.attempts)
    return result


def execute_phase3_unit(
    context: Phase3ExecutionContext,
    planned_unit: Phase3PlannedUnit,
    *,
    event: Phase3ExecutionEvent | None = None,
) -> UnitPayload:
    """Execute exactly one frozen development unit."""

    planned = _resolve_planned_unit(context, planned_unit)
    task = _resolve_task(planned)
    setup_started = time.perf_counter()
    environment = _environment(task)
    forbidden_aliases = _forbidden_aliases(environment)
    temperature = _temperature(planned)

    with open_authorized_phase3_model(
        context.authority,
        context.plan,
        planned,
        context.artifact_output_root,
    ) as model:
        if type(model) is not AuthorizedPhase3LoadedModel:
            raise TypeError("Phase 3 execution requires an authorized loaded model")
        model_report = model.key.report
        model_reference = SharedArtifactReference(
            key_id=model.key.key_id,
            artifact_id=model.index.artifact_id,
            cost_id=model.cost.cost_id,
        )
        setup_wall = time.perf_counter() - setup_started
        probe_started = time.perf_counter()
        probe = discover_affordances(
            environment,
            task_id=task.task_id,
            forbidden_aliases=forbidden_aliases,
            seed=planned.unit.seeds.probe_seed,
            action_cap=FROZEN_PROBE_ACTIONS,
            target_samples_per_alias=PROBE_COVERAGE_TARGET_SAMPLES_PER_ALIAS,
            actions_per_attempt=PROBE_ACTIONS_PER_ATTEMPT,
        )
        probe_wall = time.perf_counter() - probe_started
        if probe.accounting.actions != FROZEN_PROBE_ACTIONS:
            raise RuntimeError("Phase 3 paid probe did not consume the frozen 64 actions")
        generated = generate_phase3_candidates_with_observable_policy(
            environment,
            task_id=task.task_id,
            forbidden_aliases=forbidden_aliases,
            affordances=probe.affordances,
            model=model,
            seed=planned.unit.seeds.search_seed,
            temperature=temperature,
            max_episodes=FROZEN_CANDIDATE_EPISODES,
            max_actions_per_episode=FROZEN_MAX_ACTIONS_PER_EPISODE,
            total_adaptation_action_cap=FROZEN_TOTAL_ADAPTATION_ACTION_CAP,
            prior_adaptation_actions=FROZEN_PROBE_ACTIONS,
            condition_id=planned.base_condition_id,
            fold_id=planned.fold_id,
            replicate=planned.unit.key.replicate,
            phase=planned.unit.key.phase,
            history_shuffle_base=FROZEN_HISTORY_SHUFFLE_BASE,
            unit_id=planned.unit.unit_id,
            planned_unit=planned,
            plan_authority=context.plan,
            model_authority=context.authority,
        )
        if event is not None:
            event("generation_complete")

    if (planned.base_condition_id == H4_SHUFFLED_CONDITION) != (
        generated.history_shuffle is not None
    ):
        raise RuntimeError("Phase 3 search shuffle diagnostics differ from the condition")
    if (
        generated.accounting.episodes > FROZEN_CANDIDATE_EPISODES
        or generated.accounting.actions
        > FROZEN_TOTAL_ADAPTATION_ACTION_CAP - FROZEN_PROBE_ACTIONS
    ):
        raise RuntimeError("Phase 3 generation exceeded the frozen search budget")
    replay = evaluate_generated_search(
        generated,
        IndependentCandidateEvaluator(environment),
    )
    if event is not None:
        event("candidate_evaluation_complete")

    # This call is deliberately after the model context and independent replay.
    if event is not None:
        event("optimum_oracle")
    oracle_started = time.perf_counter()
    optimum = float(_default_optimum_provider(environment, task.family_id))
    oracle_wall = time.perf_counter() - oracle_started
    exact = classify_exact_optimum(replay, optimum_performance=optimum)

    valid = bool(replay.evaluated_candidates)
    completed = bool(generated.candidates)
    success = bool(exact.success)
    return UnitPayload(
        outcome=UnitOutcome(
            evaluator_ran=True,
            valid=valid,
            completed=completed,
            success=success,
            performance_metric_id="performance_value",
            performance_value=replay.best_performance,
            performance_direction="minimize",
            first_valid_completion_episode=replay.first_valid_episode,
            first_optimum_episode=exact.first_episode,
            first_optimum_adaptation_actions=exact.first_adaptation_actions,
            censored=not success,
            censoring_budget=None if success else FROZEN_TOTAL_ADAPTATION_ACTION_CAP,
            censoring_reason=None if success else "fixed_endpoint",
        ),
        accounting=ResourceAccounting(
            setup=PhaseAccounting(calls=1, wall_seconds=setup_wall),
            probes=PhaseAccounting(
                calls=probe.accounting.attempts,
                actions=probe.accounting.actions,
                environment_steps=probe.accounting.actions,
                resets=probe.accounting.resets,
                wall_seconds=probe_wall,
            ),
            training=PhaseAccounting(),
            search=PhaseAccounting(
                calls=1,
                episodes=replay.accounting.episodes,
                actions=replay.accounting.actions,
                environment_steps=replay.accounting.actions,
                resets=replay.accounting.resets,
                forward_passes=replay.accounting.forward_passes,
                wall_seconds=replay.accounting.generation_wall_seconds,
            ),
            replay=PhaseAccounting(
                calls=replay.accounting.evaluator_calls,
                actions=replay.accounting.evaluator_replay_actions,
                environment_steps=replay.accounting.evaluator_replay_actions,
                resets=replay.accounting.evaluator_calls,
                wall_seconds=replay.accounting.evaluator_wall_seconds,
            ),
            evaluator=PhaseAccounting(calls=1, wall_seconds=oracle_wall),
        ),
        shared_artifact=model_reference,
        shared_artifacts=(),
        candidate_generation_sha256=generated.candidate_generation_sha256,
        history_shuffle_permutation_map_sha256=(
            generated.history_shuffle.permutation_map_sha256
            if generated.history_shuffle is not None
            else None
        ),
        diagnostics=_diagnostics(generated, probe, model_report, oracle_wall),
    )


__all__ = [
    "FAILURE_SENTINEL",
    "Phase3ExecutionContext",
    "PROBE_ACTIONS_PER_ATTEMPT",
    "PROBE_COVERAGE_TARGET_SAMPLES_PER_ALIAS",
    "execute_phase3_unit",
]
