"""Execute one frozen, development-only outcome diagnostic unit.

This module is deliberately only the held-out execution boundary.  Model and
probe capabilities are minted by the descriptor-pinned execution-model and
generation modules; this layer cannot train a model, substitute a task, or
provide an optimum to candidate generation.  The reporting oracle is called
only after the generated candidate batch has been independently replayed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from levelup.envs.adaptive_track import optimal_path as adaptive_optimal_path
from levelup.envs.challenge_track import optimal_path as combo_optimal_path
from levelup.experiments.milestone6_baselines import (
    IndependentCandidateEvaluator,
    classify_exact_optimum,
    discover_affordances,
    evaluate_generated_search,
)
from levelup.experiments.milestone6_phase2 import _environment, _forbidden_aliases
from levelup.experiments.milestone6_phase2_screening import screening_child_configs
from levelup.experiments.milestone6_phase3_outcome_diagnostic_execution_models import (
    AuthorizedOutcomeExecutionModel,
    OutcomeDiagnosticExecutionAuthorityCache,
    OutcomeDiagnosticExecutionModelError,
    build_outcome_diagnostic_execution_authority_cache,
    load_authorized_outcome_model_from_pinned_store,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_generation import (
    FROZEN_CANDIDATE_EPISODES,
    FROZEN_MAX_ACTIONS_PER_EPISODE,
    FROZEN_PROBE_ACTIONS,
    FROZEN_TOTAL_ADAPTATION_ACTION_CAP,
    authorize_outcome_probe_context,
    generate_outcome_group_candidates_with_observable_policy,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
    OutcomeDiagnosticModelArtifactAuthority,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    OutcomePlannedUnit,
    ValidatedOutcomePlan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    CONDITIONS,
    OutcomeDiagnosticProtocolSnapshot,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_readiness import (
    OutcomeDiagnosticModelReadinessSnapshot,
)
from levelup.experiments.runner.config import TaskIdentity
from levelup.experiments.runner.records import (
    PhaseAccounting,
    ResourceAccounting,
    UnitOutcome,
    UnitPayload,
)

PROBE_ACTIONS_PER_ATTEMPT = 16
PROBE_COVERAGE_TARGET_SAMPLES_PER_ALIAS = 8
FAILURE_SENTINEL = FROZEN_TOTAL_ADAPTATION_ACTION_CAP + 1
_CONTEXT_TOKEN = object()


class OutcomeDiagnosticExecutionError(ValueError):
    """Raised when a unit is not authorized by the frozen development plan."""


class OutcomeDiagnosticExecutionEvent(Protocol):
    def __call__(self, name: str) -> None: ...


@dataclass(frozen=True, slots=True, init=False)
class OutcomeDiagnosticExecutionContext:
    """Immutable execution authorities derived from one active readiness snapshot."""

    snapshot: OutcomeDiagnosticModelReadinessSnapshot
    authority: OutcomeDiagnosticModelArtifactAuthority
    plan: ValidatedOutcomePlan
    protocol: OutcomeDiagnosticProtocolSnapshot
    authority_cache: OutcomeDiagnosticExecutionAuthorityCache

    def __init__(
        self,
        snapshot: OutcomeDiagnosticModelReadinessSnapshot,
        authority: OutcomeDiagnosticModelArtifactAuthority,
        plan: ValidatedOutcomePlan,
        protocol: OutcomeDiagnosticProtocolSnapshot,
        authority_cache: OutcomeDiagnosticExecutionAuthorityCache,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _CONTEXT_TOKEN:
            raise OutcomeDiagnosticExecutionError(
                "outcome execution contexts require canonical readiness"
            )
        object.__setattr__(self, "snapshot", snapshot)
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "authority_cache", authority_cache)

    @classmethod
    def canonical(
        cls, snapshot: OutcomeDiagnosticModelReadinessSnapshot
    ) -> "OutcomeDiagnosticExecutionContext":
        if type(snapshot) is not OutcomeDiagnosticModelReadinessSnapshot:
            raise OutcomeDiagnosticExecutionError(
                "canonical outcome model readiness snapshot is required"
            )
        try:
            snapshot.lease.require_active()
            cache = build_outcome_diagnostic_execution_authority_cache(snapshot)
            cache.require_active()
            authority = snapshot.authority
            protocol = snapshot.protocol
            if type(authority) is not OutcomeDiagnosticModelArtifactAuthority:
                raise OutcomeDiagnosticExecutionError("typed outcome model authority is required")
            if type(protocol) is not OutcomeDiagnosticProtocolSnapshot:
                raise OutcomeDiagnosticExecutionError("typed outcome protocol is required")
            plan = cache.validated_plan
            if (
                type(plan) is not ValidatedOutcomePlan
                or cache.authority is not authority
                or authority.final_family_access
                or authority.final
                or not authority.development_only
                or plan.plan.final_family_access
                or authority.protocol_sha256 != protocol.sha256
                or plan.plan.protocol_sha256 != protocol.sha256
            ):
                raise OutcomeDiagnosticExecutionError(
                    "outcome authority/cache is not development-only and protocol-bound"
                )
        except OutcomeDiagnosticExecutionError:
            raise
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise OutcomeDiagnosticExecutionError(
                "outcome readiness snapshot cannot authorize execution"
            ) from exc
        return cls(snapshot, authority, plan, protocol, cache, _token=_CONTEXT_TOKEN)


def _default_optimum_provider(environment: Any, family_id: str) -> float:
    """Reporting-only oracle; it is intentionally private and post-replay."""

    if family_id == "combo":
        return float(combo_optimal_path(environment)[0])
    return float(adaptive_optimal_path(environment)[0])


@lru_cache(maxsize=1)
def _canonical_validation_tasks() -> tuple[TaskIdentity, ...]:
    configs = screening_child_configs()
    if len(configs) != 6 or any(config.split.final_tasks for config in configs):
        raise OutcomeDiagnosticExecutionError("canonical task authority is not development-only")
    tasks = tuple(task for config in configs for task in config.split.validation_tasks)
    if len(tasks) != 48 or len({(task.family_id, task.task_id) for task in tasks}) != 48:
        raise OutcomeDiagnosticExecutionError("canonical validation task matrix differs")
    return tasks


def _resolve_task(unit: OutcomePlannedUnit) -> TaskIdentity:
    matches = tuple(
        task
        for task in _canonical_validation_tasks()
        if task.family_id == unit.heldout_family and task.task_id == unit.task_id
    )
    if len(matches) != 1:
        raise OutcomeDiagnosticExecutionError("unit has no canonical validation task")
    task = matches[0]
    if (
        task.family_id != unit.heldout_family
        or task.task_id != unit.task_id
        or task.task_index != unit.task_index
        or task.environment_reset_seed != unit.environment_seed
        or task.environment_reset_seed != 0
    ):
        raise OutcomeDiagnosticExecutionError("resolved task identity differs from the unit")
    return task


def _resolve_unit(
    context: OutcomeDiagnosticExecutionContext,
    unit: OutcomePlannedUnit,
) -> OutcomePlannedUnit:
    if type(context) is not OutcomeDiagnosticExecutionContext:
        raise OutcomeDiagnosticExecutionError("canonical outcome execution context is required")
    if type(unit) is not OutcomePlannedUnit:
        raise OutcomeDiagnosticExecutionError("canonical outcome planned unit is required")
    try:
        context.authority_cache.require_active()
        planned = context.authority_cache.resolve_unit(unit)
    except (AttributeError, TypeError, ValueError, OutcomeDiagnosticExecutionModelError) as exc:
        raise OutcomeDiagnosticExecutionError("planned unit differs from frozen authority") from exc
    if (
        planned.final_family_access
        or planned.condition_id not in CONDITIONS
        or planned.environment_seed != 0
        or planned.candidate_episodes_per_task != FROZEN_CANDIDATE_EPISODES
        or planned.adaptation_actions_per_task != FROZEN_TOTAL_ADAPTATION_ACTION_CAP
        or planned.probe_actions_per_task != FROZEN_PROBE_ACTIONS
        or planned.maximum_actions_per_candidate_episode != FROZEN_MAX_ACTIONS_PER_EPISODE
    ):
        raise OutcomeDiagnosticExecutionError("unit condition, seed, or budget is not frozen")
    return planned


def _report_diagnostics(
    generated: Any,
    probe: Any,
    model: AuthorizedOutcomeExecutionModel,
    oracle_wall: float,
) -> dict[str, bool | int | float | None]:
    accounting = generated.accounting
    training = model.training_accounting
    return {
        "development_outcome_diagnostic": True,
        "model_trainable_parameters": 3_841,
        "model_optimizer_steps": int(training.optimizer_steps),
        "model_forward_passes": int(training.forward_passes),
        "model_training_examples": int(training.training_examples),
        "model_serialization_calls": int(training.serialization_calls),
        "model_recurrent_steps": 0,
        "unknown_affordance_decisions": int(accounting.unknown_affordance_decisions),
        "probe_attempts": int(probe.accounting.attempts),
        "search_forward_passes": int(accounting.forward_passes),
        "oracle_wall_seconds": oracle_wall,
    }


def execute_outcome_diagnostic_unit(
    context: OutcomeDiagnosticExecutionContext,
    planned_unit: OutcomePlannedUnit,
    *,
    event: OutcomeDiagnosticExecutionEvent | None = None,
) -> UnitPayload:
    """Execute exactly one canonical development validation unit."""

    planned = _resolve_unit(context, planned_unit)
    task = _resolve_task(planned)
    setup_started = time.perf_counter()
    environment = _environment(task)
    forbidden_aliases = frozenset(_forbidden_aliases(environment))

    with load_authorized_outcome_model_from_pinned_store(context.snapshot, planned) as model:
        if type(model) is not AuthorizedOutcomeExecutionModel:
            raise OutcomeDiagnosticExecutionError("authorized outcome model capability is invalid")
        model.require_active()
        setup_wall = time.perf_counter() - setup_started
        probe_started = time.perf_counter()
        probe = discover_affordances(
            environment,
            task_id=task.task_id,
            forbidden_aliases=forbidden_aliases,
            seed=planned.probe_seed,
            action_cap=FROZEN_PROBE_ACTIONS,
            target_samples_per_alias=PROBE_COVERAGE_TARGET_SAMPLES_PER_ALIAS,
            actions_per_attempt=PROBE_ACTIONS_PER_ATTEMPT,
        )
        if probe.accounting.actions != FROZEN_PROBE_ACTIONS:
            raise OutcomeDiagnosticExecutionError("paid probe did not spend exactly 64 actions")
        probe_context = authorize_outcome_probe_context(
            environment,
            task,
            probe,
            planned,
            context.plan,
            context.protocol,
        )
        probe_wall = time.perf_counter() - probe_started
        generated = generate_outcome_group_candidates_with_observable_policy(
            model=model.authorized_model,
            probe=probe_context,
            planned_unit=planned,
            plan=context.plan,
            protocol_snapshot=context.protocol,
        )
        accounting = generated.accounting
        if (
            accounting.planned_episode_cap != FROZEN_CANDIDATE_EPISODES
            or accounting.prior_probe_actions != FROZEN_PROBE_ACTIONS
            or accounting.episodes > FROZEN_CANDIDATE_EPISODES
            or accounting.actions + FROZEN_PROBE_ACTIONS
            > FROZEN_TOTAL_ADAPTATION_ACTION_CAP
        ):
            raise OutcomeDiagnosticExecutionError("outcome generation accounting exceeded frozen budget")
        if event is not None:
            event("generation_complete")

    # The authorized model context is closed before replay.  Replay therefore
    # cannot accidentally observe a live store/model descriptor.
    replay = evaluate_generated_search(
        generated,
        IndependentCandidateEvaluator(environment),
    )
    if event is not None:
        event("candidate_evaluation_complete")
        event("optimum_oracle")
    oracle_started = time.perf_counter()
    optimum = _default_optimum_provider(environment, task.family_id)
    oracle_wall = time.perf_counter() - oracle_started
    exact = classify_exact_optimum(replay, optimum_performance=optimum)
    success = bool(exact.success)
    valid = bool(replay.evaluated_candidates)
    completed = bool(generated.candidates)
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
                wall_seconds=getattr(probe.accounting, "wall_seconds", probe_wall),
            ),
            training=PhaseAccounting(),
            search=PhaseAccounting(
                calls=1,
                episodes=replay.accounting.episodes,
                actions=replay.accounting.actions,
                environment_steps=replay.accounting.actions,
                resets=replay.accounting.resets,
                forward_passes=replay.accounting.forward_passes,
                wall_seconds=getattr(replay.accounting, "generation_wall_seconds", 0.0),
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
        # Prepared model costs are shared by the owner and are intentionally
        # not fabricated as a per-unit shared-artifact reference here.
        shared_artifact=None,
        shared_artifacts=(),
        candidate_generation_sha256=generated.candidate_generation_sha256,
        diagnostics=_report_diagnostics(generated, probe, model, oracle_wall),
    )


__all__ = [
    "FAILURE_SENTINEL",
    "OutcomeDiagnosticExecutionContext",
    "OutcomeDiagnosticExecutionError",
    "PROBE_ACTIONS_PER_ATTEMPT",
    "PROBE_COVERAGE_TARGET_SAMPLES_PER_ALIAS",
    "execute_outcome_diagnostic_unit",
]
