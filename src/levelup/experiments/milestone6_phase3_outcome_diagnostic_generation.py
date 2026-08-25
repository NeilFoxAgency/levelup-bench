"""Oracle-free generation for the frozen Phase 3 outcome-channel diagnostic.

This module is intentionally separate from the historical Phase 3 dispatcher.  It
only supports the two additive outcome masks, consumes a canonical validated
diagnostic unit, and returns a complete observable candidate batch.  Replay,
evaluation, and exact-optimum classification are deliberately absent.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from levelup.core.trajectory import ActionRecord, Trajectory, TrajectoryStep
from levelup.experiments.milestone6_baselines import ObservableEnvironment
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
    AuthorizedOutcomeModelArtifact,
    OutcomeDiagnosticModelArtifactRecord,
    OutcomeStateTensorPayload,
    PinnedOutcomeModelState,
    inspect_outcome_model_state,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    OutcomePlannedUnit,
    ValidatedOutcomePlan,
    validate_outcome_diagnostic_plan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    CONDITIONS,
    OutcomeDiagnosticProtocolSnapshot,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.learning.state_conditioned import (
    AffordanceTable,
    DecisionExample,
    ObservableTrace,
    StateConditionedScorer,
    apply_progress_elapsed_completion_mask,
    apply_resource_pressure_mask,
    candidate_tensor,
    outcome_group_optimum_example_views,
    parse_observation,
)

if len(CONDITIONS) != 2:
    raise RuntimeError("outcome diagnostic protocol must define exactly two conditions")
RP_CONDITION, PEC_CONDITION = CONDITIONS
OUTCOME_CONDITIONS = CONDITIONS

FROZEN_PROBE_ACTIONS = 64
FROZEN_CANDIDATE_EPISODES = 150
FROZEN_TOTAL_ADAPTATION_ACTION_CAP = 2_048
FROZEN_MAX_ACTIONS_PER_EPISODE = 64
FROZEN_PARAMETER_COUNT = 3_841
_TEMPERATURE_BY_SUFFIX = {"t0p6": 0.6, "t0p9": 0.9, "t1p2": 1.2}
_MODEL_TOKEN = object()
_PROBE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class OutcomeGenerationAccounting:
    """Observable resource counters; no evaluator/oracle counters exist here."""

    planned_episode_cap: int
    episodes: int
    resets: int
    actions: int
    forward_passes: int
    unknown_affordance_decisions: int
    prior_probe_actions: int
    probe_seed: int
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class OutcomeGeneratedCandidate:
    episode: int
    adaptation_actions: int
    trajectory: Trajectory


@dataclass(frozen=True, slots=True)
class OutcomeGeneratedSearch:
    candidates: tuple[OutcomeGeneratedCandidate, ...]
    accounting: OutcomeGenerationAccounting
    candidate_generation_sha256: str


@dataclass(frozen=True, slots=True, init=False)
class AuthorizedOutcomeGenerationModel:
    """Opaque pre-executor lease for one artifact-validated diagnostic model."""

    model: StateConditionedScorer
    unit_id: str
    owner_id: str
    view_id: str
    model_identity_sha256: str
    model_state_sha256: str
    artifact_record_id: str
    _token: object

    def __init__(
        self,
        model: StateConditionedScorer,
        unit: OutcomePlannedUnit,
        model_state_sha256: str,
        artifact_record_id: str,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _MODEL_TOKEN:
            raise ValueError("authorized outcome models require the artifact validator")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "unit_id", unit.unit_id)
        object.__setattr__(self, "owner_id", unit.model_owner_id)
        object.__setattr__(self, "view_id", unit.view_id)
        object.__setattr__(self, "model_identity_sha256", unit.model_identity_sha256)
        object.__setattr__(self, "model_state_sha256", model_state_sha256)
        object.__setattr__(self, "artifact_record_id", artifact_record_id)
        object.__setattr__(self, "_token", _MODEL_TOKEN)


@dataclass(frozen=True, slots=True, init=False)
class AuthorizedOutcomeProbeContext:
    """Opaque exact-task context produced after the canonical 64-action probe."""

    environment: ObservableEnvironment
    unit_id: str
    task_id: str
    task_index: int
    family_id: str
    fold_id: str
    replicate: int
    environment_seed: int
    probe_seed: int
    probe_actions: int
    forbidden_aliases: frozenset[str]
    affordances: AffordanceTable
    affordance_sha256: str
    environment_generation_identity_sha256: str
    probe_context_sha256: str
    _token: object

    def __init__(
        self,
        unit: OutcomePlannedUnit,
        environment: ObservableEnvironment,
        affordances: AffordanceTable,
        forbidden_aliases: frozenset[str],
        environment_generation_identity_sha256: str,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _PROBE_TOKEN:
            raise ValueError("authorized probe contexts require the canonical probe validator")
        if not all(
            callable(getattr(environment, name, None)) for name in ("fresh", "reset", "step")
        ):
            raise ValueError("authorized probe context requires an observable environment")
        affordance_sha = _affordance_identity(affordances)
        if (
            not isinstance(environment_generation_identity_sha256, str)
            or len(environment_generation_identity_sha256) != 64
        ):
            raise ValueError("canonical environment generation identity is required")
        environment_identity = environment_generation_identity_sha256
        context_sha = hashlib.sha256(
            canonical_json_bytes(
                {
                    "unit_id": unit.unit_id,
                    "environment_generation_identity_sha256": environment_identity,
                    "probe_seed": unit.probe_seed,
                    "probe_actions": FROZEN_PROBE_ACTIONS,
                    "forbidden_aliases": sorted(forbidden_aliases),
                    "affordance_sha256": affordance_sha,
                }
            )
        ).hexdigest()
        object.__setattr__(self, "environment", environment)
        object.__setattr__(self, "unit_id", unit.unit_id)
        object.__setattr__(self, "task_id", unit.task_id)
        object.__setattr__(self, "task_index", unit.task_index)
        object.__setattr__(self, "family_id", unit.heldout_family)
        object.__setattr__(self, "fold_id", unit.fold_id)
        object.__setattr__(self, "replicate", unit.replicate)
        object.__setattr__(self, "environment_seed", unit.environment_seed)
        object.__setattr__(self, "probe_seed", unit.probe_seed)
        object.__setattr__(self, "probe_actions", FROZEN_PROBE_ACTIONS)
        object.__setattr__(self, "forbidden_aliases", forbidden_aliases)
        object.__setattr__(self, "affordances", affordances)
        object.__setattr__(self, "affordance_sha256", affordance_sha)
        object.__setattr__(self, "environment_generation_identity_sha256", environment_identity)
        object.__setattr__(self, "probe_context_sha256", context_sha)
        object.__setattr__(self, "_token", _PROBE_TOKEN)


def _mask_for_condition(condition_id: str):
    if condition_id == RP_CONDITION:
        return apply_resource_pressure_mask
    if condition_id == PEC_CONDITION:
        return apply_progress_elapsed_completion_mask
    raise ValueError("unknown outcome diagnostic condition")


def outcome_group_training_examples(
    samples: Sequence[tuple[ObservableTrace, AffordanceTable]],
    condition_id: str,
) -> tuple[DecisionExample, ...]:
    """Build one outcome view from one exact, ordered T-example source tuple."""

    # Deliberately make one call: all views must share the same source examples,
    # labels, candidate ordering, and example ordering.
    views = outcome_group_optimum_example_views(samples)
    if condition_id == RP_CONDITION:
        return views.resource_pressure
    if condition_id == PEC_CONDITION:
        return views.progress_elapsed_completion
    raise ValueError("unknown outcome diagnostic condition")


def outcome_group_optimum_examples(
    samples: Sequence[tuple[ObservableTrace, AffordanceTable]],
    condition_id: str,
) -> tuple[DecisionExample, ...]:
    """Compatibility spelling for :func:`outcome_group_training_examples`."""

    return outcome_group_training_examples(samples, condition_id)


def _masked_visible_action_weights(
    model: StateConditionedScorer,
    state: Any,
    affordances: AffordanceTable,
    *,
    condition_id: str,
    temperature: float,
) -> tuple[dict[str, float], int]:
    """Score visible aliases using the exact outcome mask and S fallback semantics."""

    raw_model = getattr(model, "model", model)
    if type(raw_model) is not StateConditionedScorer:
        raise ValueError("outcome diagnostic generation requires StateConditionedScorer")
    if sum(parameter.numel() for parameter in raw_model.parameters()) != FROZEN_PARAMETER_COUNT:
        raise ValueError("outcome diagnostic scorer capacity drifted")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    aliases, features, unknown = candidate_tensor(state, affordances)
    masked = _mask_for_condition(condition_id)(features)
    with torch.no_grad():
        scores = raw_model(masked)
    if unknown:
        known_indices = [
            index for index, alias in enumerate(aliases) if affordances.for_alias(alias) is not None
        ]
        neutral_score = (
            scores[known_indices].mean()
            if known_indices
            else torch.tensor(0.0, dtype=scores.dtype, device=scores.device)
        )
        scores = scores.clone()
        for index, alias in enumerate(aliases):
            if affordances.for_alias(alias) is None:
                scores[index] = neutral_score
    probabilities = torch.softmax(scores / temperature, dim=0).tolist()
    return dict(zip(aliases, (float(value) for value in probabilities))), unknown


def outcome_group_visible_action_weights(
    model: StateConditionedScorer,
    state: Any,
    affordances: AffordanceTable,
    *,
    condition_id: str,
    temperature: float,
) -> tuple[dict[str, float], int]:
    """Public typed scorer used by both tests and the later executor."""

    return _masked_visible_action_weights(
        model,
        state,
        affordances,
        condition_id=condition_id,
        temperature=temperature,
    )


def masked_visible_action_weights(
    model: StateConditionedScorer,
    state: Any,
    affordances: AffordanceTable,
    *,
    condition_id: str,
    temperature: float,
) -> tuple[dict[str, float], int]:
    """Short alias emphasizing that scores are computed after masking."""

    return outcome_group_visible_action_weights(
        model,
        state,
        affordances,
        condition_id=condition_id,
        temperature=temperature,
    )


def _model_state_identity(model: StateConditionedScorer) -> str:
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        header = canonical_json_bytes(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        )
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


model_state_sha256 = _model_state_identity


def _affordance_identity(affordances: AffordanceTable) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "features": {
                    alias: list(affordances.features[alias])
                    for alias in sorted(affordances.features)
                },
                "sample_counts": {
                    alias: affordances.sample_counts[alias]
                    for alias in sorted(affordances.sample_counts)
                },
            }
        )
    ).hexdigest()


def _temperature_for_unit(unit: OutcomePlannedUnit) -> float:
    suffix = unit.tuple_id.rsplit("-", 1)[-1]
    try:
        return _TEMPERATURE_BY_SUFFIX[suffix]
    except KeyError as exc:
        raise ValueError("outcome diagnostic tuple has unknown temperature") from exc


def _validate_unit(
    unit: OutcomePlannedUnit,
    plan: ValidatedOutcomePlan,
    model: AuthorizedOutcomeGenerationModel,
    probe: AuthorizedOutcomeProbeContext,
    protocol_snapshot: OutcomeDiagnosticProtocolSnapshot,
) -> tuple[StateConditionedScorer, str, float]:
    if type(plan) is not ValidatedOutcomePlan or type(unit) is not OutcomePlannedUnit:
        raise ValueError("generation requires canonical outcome unit and validated plan")
    if plan.plan.final_family_access or unit.final_family_access:
        raise ValueError("outcome diagnostic generation cannot access final families")
    if type(protocol_snapshot) is not OutcomeDiagnosticProtocolSnapshot:
        raise ValueError("generation requires the canonical diagnostic protocol snapshot")
    _validate_canonical_plan(plan, protocol_snapshot)
    try:
        plan.require_unit(unit)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("outcome unit differs from the validated plan") from exc
    if unit.condition_id not in OUTCOME_CONDITIONS:
        raise ValueError("outcome unit condition is not one of the frozen masks")
    expected_temperature = _temperature_for_unit(unit)
    if (
        unit.candidate_episodes_per_task != FROZEN_CANDIDATE_EPISODES
        or unit.adaptation_actions_per_task != FROZEN_TOTAL_ADAPTATION_ACTION_CAP
        or unit.probe_actions_per_task != FROZEN_PROBE_ACTIONS
        or unit.maximum_actions_per_candidate_episode != FROZEN_MAX_ACTIONS_PER_EPISODE
    ):
        raise ValueError("outcome unit budgets differ from the frozen diagnostic")
    if unit.environment_seed != 0:
        raise ValueError("outcome diagnostic requires the canonical environment seed")
    if (
        type(model) is not AuthorizedOutcomeGenerationModel
        or model._token is not _MODEL_TOKEN
        or model.unit_id != unit.unit_id
        or model.owner_id != unit.model_owner_id
        or model.view_id != unit.view_id
        or model.model_identity_sha256 != unit.model_identity_sha256
        or model.model_state_sha256 != _model_state_identity(model.model)
    ):
        raise ValueError("outcome diagnostic model authorization differs from the unit")
    if (
        type(probe) is not AuthorizedOutcomeProbeContext
        or probe._token is not _PROBE_TOKEN
        or probe.unit_id != unit.unit_id
        or (
            probe.task_id,
            probe.task_index,
            probe.family_id,
            probe.fold_id,
            probe.replicate,
            probe.environment_seed,
            probe.probe_seed,
            probe.probe_actions,
        )
        != (
            unit.task_id,
            unit.task_index,
            unit.heldout_family,
            unit.fold_id,
            unit.replicate,
            unit.environment_seed,
            unit.probe_seed,
            FROZEN_PROBE_ACTIONS,
        )
        or probe.affordance_sha256 != _affordance_identity(probe.affordances)
        or not all(
            callable(getattr(probe.environment, name, None)) for name in ("fresh", "reset", "step")
        )
        or probe.probe_context_sha256
        != hashlib.sha256(
            canonical_json_bytes(
                {
                    "unit_id": unit.unit_id,
                    "environment_generation_identity_sha256": (
                        probe.environment_generation_identity_sha256
                    ),
                    "probe_seed": unit.probe_seed,
                    "probe_actions": FROZEN_PROBE_ACTIONS,
                    "forbidden_aliases": sorted(probe.forbidden_aliases),
                    "affordance_sha256": probe.affordance_sha256,
                }
            )
        ).hexdigest()
    ):
        raise ValueError("outcome diagnostic probe authorization differs from the unit")
    return model.model, unit.condition_id, expected_temperature


_VALIDATED_PLAN_KEYS: set[tuple[int, int]] = set()


def _validate_canonical_plan(
    plan: ValidatedOutcomePlan, snapshot: OutcomeDiagnosticProtocolSnapshot
) -> None:
    key = (id(plan), id(snapshot))
    if key not in _VALIDATED_PLAN_KEYS:
        validate_outcome_diagnostic_plan(plan.plan, snapshot=snapshot)
        _VALIDATED_PLAN_KEYS.add(key)


def authorize_outcome_generation_model(
    model: StateConditionedScorer,
    authorization: AuthorizedOutcomeModelArtifact,
    unit: OutcomePlannedUnit,
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
) -> AuthorizedOutcomeGenerationModel:
    """Bind exact loaded weights to a semantically validated artifact capability."""

    _validate_canonical_plan(plan, snapshot)
    plan.require_unit(unit)
    if (
        type(model) is not StateConditionedScorer
        or model.training
        or any(parameter.requires_grad for parameter in model.parameters())
    ):
        raise ValueError("artifact model must be the exact frozen eval/no-grad scorer")
    if type(authorization) is not AuthorizedOutcomeModelArtifact:
        raise ValueError("validated outcome artifact authorization is required")
    record = authorization.record
    if not isinstance(record, OutcomeDiagnosticModelArtifactRecord):
        raise ValueError("authorized outcome artifact record is invalid")
    state_payload = PinnedOutcomeModelState(
        tuple(
            OutcomeStateTensorPayload(
                name,
                tuple(tensor.shape),
                tensor.detach().cpu().contiguous().numpy().tobytes(order="C"),
            )
            for name, tensor in sorted(model.state_dict().items())
        )
    )
    key = record.key
    _schema, state_sha = inspect_outcome_model_state(state_payload)
    if (
        authorization.owner_id != key.owner_id
        or authorization.view_id != key.view_id
        or authorization.model_state_sha256 != key.model_state_sha256
        or key.plan_id != plan.plan.plan_id
        or key.protocol_sha256 != snapshot.sha256
        or key.condition_id != unit.condition_id
        or key.view_id != unit.view_id
        or key.owner_id != unit.model_owner_id
        or key.heldout_family != unit.heldout_family
        or key.fold_id != unit.fold_id
        or key.replicate != unit.replicate
        or key.training_tuple_id != unit.training_tuple_id
        or key.model_seed != unit.model_seed
        or key.data_order_seed != unit.data_order_seed
        or key.feature_mask_sha256 != unit.feature_mask_sha256
        or key.transformation_sha256 != unit.transformation_sha256
        or key.model_identity_sha256 != unit.model_identity_sha256
        or key.model_state_sha256 != state_sha
    ):
        raise ValueError("artifact record/model differs from the canonical outcome unit")
    return AuthorizedOutcomeGenerationModel(
        model, unit, state_sha, record.record_id, _token=_MODEL_TOKEN
    )


def _authorize_outcome_generation_model_for_test(
    model: StateConditionedScorer,
    unit: OutcomePlannedUnit,
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
) -> AuthorizedOutcomeGenerationModel:
    """Private pre-executor adapter for exact synthetic test weights only."""

    _validate_canonical_plan(plan, snapshot)
    if unit.final_family_access:
        raise ValueError("test model authorization cannot access final families")
    plan.require_unit(unit)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state_sha = _model_state_identity(model)
    return AuthorizedOutcomeGenerationModel(
        model, unit, state_sha, "test-only", _token=_MODEL_TOKEN
    )


def _authorize_outcome_probe_context_for_test(
    environment: ObservableEnvironment,
    unit: OutcomePlannedUnit,
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    *,
    affordances: AffordanceTable,
    forbidden_aliases: frozenset[str],
) -> AuthorizedOutcomeProbeContext:
    """Private pre-executor adapter for a synthetic exact-task paid probe."""

    _validate_canonical_plan(plan, snapshot)
    if unit.final_family_access:
        raise ValueError("test probe authorization cannot access final families")
    plan.require_unit(unit)
    if not isinstance(affordances, AffordanceTable) or not isinstance(forbidden_aliases, frozenset):
        raise ValueError("typed probe affordances and forbidden aliases are required")
    environment_identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "test_only": True,
                "environment_type": (
                    f"{type(environment).__module__}.{type(environment).__qualname__}"
                ),
                "task_id": unit.task_id,
                "task_index": unit.task_index,
                "family_id": unit.heldout_family,
                "fold_id": unit.fold_id,
                "replicate": unit.replicate,
                "environment_seed": unit.environment_seed,
            }
        )
    ).hexdigest()
    return AuthorizedOutcomeProbeContext(
        unit,
        environment,
        affordances,
        forbidden_aliases,
        environment_identity,
        _token=_PROBE_TOKEN,
    )


def _candidate_generation_sha256(
    candidates: tuple[OutcomeGeneratedCandidate, ...],
    *,
    unit: OutcomePlannedUnit,
    model_state_sha256: str,
    artifact_record_id: str,
    probe: AuthorizedOutcomeProbeContext,
    accounting: OutcomeGenerationAccounting,
) -> str:
    payload = {
        "schema_version": "milestone6.phase3.outcome-group-candidate-generation.v1",
        "condition_id": unit.condition_id,
        "unit_id": unit.unit_id,
        "task_id": unit.task_id,
        "task_index": unit.task_index,
        "tuple_id": unit.tuple_id,
        "fold_id": unit.fold_id,
        "heldout_family": unit.heldout_family,
        "replicate": unit.replicate,
        "view_id": unit.view_id,
        "model_owner_id": unit.model_owner_id,
        "exposure_manifest_sha256": unit.exposure_manifest_sha256,
        "model_identity_sha256": unit.model_identity_sha256,
        "model_state_sha256": model_state_sha256,
        "artifact_record_id": artifact_record_id,
        "feature_mask_sha256": unit.feature_mask_sha256,
        "transformation_sha256": unit.transformation_sha256,
        "environment_generation_identity_sha256": (probe.environment_generation_identity_sha256),
        "probe_context_sha256": probe.probe_context_sha256,
        "forbidden_aliases": sorted(probe.forbidden_aliases),
        "affordance_sha256": probe.affordance_sha256,
        "accounting_schema": {
            "prior_probe_actions": FROZEN_PROBE_ACTIONS,
            "search_actions_exclude_probes": True,
            "candidate_endpoint_includes_probes": True,
        },
        "accounting": {
            "planned_episode_cap": accounting.planned_episode_cap,
            "attempted_episodes": accounting.episodes,
            "resets": accounting.resets,
            "search_actions": accounting.actions,
            "forward_passes": accounting.forward_passes,
            "unknown_affordance_decisions": accounting.unknown_affordance_decisions,
            "prior_probe_actions": accounting.prior_probe_actions,
            "probe_seed": accounting.probe_seed,
        },
        "seeds": {
            "model": unit.model_seed,
            "environment": unit.environment_seed,
            "probe": unit.probe_seed,
            "search": unit.search_seed,
            "data_order": unit.data_order_seed,
        },
        "budgets": {
            "probe_actions": FROZEN_PROBE_ACTIONS,
            "candidate_episodes": FROZEN_CANDIDATE_EPISODES,
            "adaptation_actions": FROZEN_TOTAL_ADAPTATION_ACTION_CAP,
            "max_actions_per_episode": FROZEN_MAX_ACTIONS_PER_EPISODE,
        },
        "candidates": [
            {
                "episode": item.episode,
                "adaptation_actions": item.adaptation_actions,
                "trajectory": item.trajectory.model_dump(mode="json"),
            }
            for item in candidates
        ],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def generate_outcome_group_candidates_with_observable_policy(
    *,
    model: AuthorizedOutcomeGenerationModel,
    probe: AuthorizedOutcomeProbeContext,
    planned_unit: OutcomePlannedUnit,
    plan: ValidatedOutcomePlan,
    protocol_snapshot: OutcomeDiagnosticProtocolSnapshot,
) -> OutcomeGeneratedSearch:
    """Generate a fixed candidate batch without evaluator, oracle, or stopping feedback.

    Task, condition, seeds, temperatures, and budgets come from ``planned_unit``;
    model/probe inputs are opaque validator products, and the environment is
    held by the probe capability.  The signature has no evaluator, oracle,
    optimum, raw scorer, raw environment, or caller-supplied task identity.
    """

    raw_model, condition, resolved_temperature = _validate_unit(
        planned_unit,
        plan,
        model,
        probe,
        protocol_snapshot=protocol_snapshot,
    )
    rng = random.Random(planned_unit.search_seed)
    started = time.perf_counter()
    candidates: list[OutcomeGeneratedCandidate] = []
    episodes = resets = actions = forward_passes = unknown_decisions = 0
    environment = probe.environment

    for episode in range(1, FROZEN_CANDIDATE_EPISODES + 1):
        if FROZEN_PROBE_ACTIONS + actions >= FROZEN_TOTAL_ADAPTATION_ACTION_CAP:
            break
        episodes = episode
        candidate_environment = environment.fresh()
        outcome = candidate_environment.reset(seed=planned_unit.environment_seed)
        resets += 1
        aliases_taken: list[str] = []
        for _ in range(FROZEN_MAX_ACTIONS_PER_EPISODE):
            if outcome.completed:
                break
            if FROZEN_PROBE_ACTIONS + actions >= FROZEN_TOTAL_ADAPTATION_ACTION_CAP:
                break
            state = parse_observation(
                outcome.observation, forbidden_aliases=probe.forbidden_aliases
            )
            if not state.available_aliases:
                break
            weights, unknown = _masked_visible_action_weights(
                raw_model,
                state,
                probe.affordances,
                condition_id=condition,
                temperature=resolved_temperature,
            )
            forward_passes += 1
            unknown_decisions += unknown
            alias = rng.choices(tuple(weights), weights=tuple(weights.values()), k=1)[0]
            aliases_taken.append(alias)
            outcome = candidate_environment.step(ActionRecord(name=alias))
            actions += 1
        if not outcome.completed:
            continue
        trajectory = Trajectory(
            trajectory_id=(
                f"search:{planned_unit.unit_id}:{planned_unit.task_id}:"
                f"s{planned_unit.search_seed}:e{episode}"
            ),
            task_id=planned_unit.task_id,
            source="agent",
            environment_seed=planned_unit.environment_seed,
            steps=tuple(
                TrajectoryStep(index=index, action=ActionRecord(name=alias))
                for index, alias in enumerate(aliases_taken)
            ),
        )
        candidates.append(
            OutcomeGeneratedCandidate(
                episode=episode,
                adaptation_actions=FROZEN_PROBE_ACTIONS + actions,
                trajectory=trajectory,
            )
        )

    result = tuple(candidates)
    accounting = OutcomeGenerationAccounting(
        planned_episode_cap=FROZEN_CANDIDATE_EPISODES,
        episodes=episodes,
        resets=resets,
        actions=actions,
        forward_passes=forward_passes,
        unknown_affordance_decisions=unknown_decisions,
        prior_probe_actions=FROZEN_PROBE_ACTIONS,
        probe_seed=planned_unit.probe_seed,
        wall_seconds=time.perf_counter() - started,
    )
    return OutcomeGeneratedSearch(
        candidates=result,
        accounting=accounting,
        candidate_generation_sha256=_candidate_generation_sha256(
            result,
            unit=planned_unit,
            model_state_sha256=model.model_state_sha256,
            artifact_record_id=model.artifact_record_id,
            probe=probe,
            accounting=accounting,
        ),
    )


generate_candidates_with_outcome_diagnostic_policy = (
    generate_outcome_group_candidates_with_observable_policy
)


__all__ = [
    "FROZEN_CANDIDATE_EPISODES",
    "FROZEN_MAX_ACTIONS_PER_EPISODE",
    "FROZEN_PARAMETER_COUNT",
    "FROZEN_PROBE_ACTIONS",
    "FROZEN_TOTAL_ADAPTATION_ACTION_CAP",
    "OUTCOME_CONDITIONS",
    "PEC_CONDITION",
    "RP_CONDITION",
    "AuthorizedOutcomeGenerationModel",
    "AuthorizedOutcomeProbeContext",
    "OutcomeGeneratedCandidate",
    "OutcomeGeneratedSearch",
    "OutcomeGenerationAccounting",
    "generate_candidates_with_outcome_diagnostic_policy",
    "generate_outcome_group_candidates_with_observable_policy",
    "authorize_outcome_generation_model",
    "masked_visible_action_weights",
    "model_state_sha256",
    "outcome_group_optimum_examples",
    "outcome_group_training_examples",
    "outcome_group_visible_action_weights",
]
