"""Development-only Phase 3 candidate generation.

This module is an additive execution boundary for the four new representation
conditions.  It deliberately has no evaluator, oracle, optimum threshold, or
result aggregation input. Generation stops only at the caller's frozen observable
episode/action caps; validity and performance classification belong to a later
independent replay boundary.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from typing import Any

import torch

from levelup.core.trajectory import ActionRecord, Trajectory, TrajectoryStep
from levelup.experiments.milestone6_baselines import ObservableEnvironment
from levelup.experiments.milestone6_phase3_execution_models import (
    AuthorizedPhase3LoadedModel,
    validate_authorized_phase3_loaded_model,
)
from levelup.experiments.milestone6_phase3_model_authority import (
    Phase3ModelArtifactAuthority,
)
from levelup.experiments.milestone6_phase3_plan import (
    Phase3PlannedUnit,
    ValidatedPhase3Plan,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.learning.state_conditioned import (
    AffordanceTable,
    HistoryConditionedScorer,
    HistoryPermutationIdentity,
    ObservedTransition,
    StateConditionedScorer,
    deterministic_history_derangement,
    history_visible_action_weights,
    parse_observation,
    permutation_map_sha256,
    state_availability_visible_action_weights,
    transition_features,
)

S_CONDITION = "S-state-availability-listwise-optimum"
H0_CONDITION = "H0-null-history-transition-listwise-optimum"
H4_CONDITION = "H4-causal-history-transition-listwise-optimum"
H4_SHUFFLED_CONDITION = "H4-shuffled-history-transition-listwise-optimum"
HISTORY_CONDITIONS = frozenset({H0_CONDITION, H4_CONDITION, H4_SHUFFLED_CONDITION})
HISTORY_LENGTH = 4
FROZEN_CANDIDATE_EPISODES = 150
FROZEN_MAX_ACTIONS_PER_EPISODE = 64
FROZEN_TOTAL_ADAPTATION_ACTION_CAP = 2_048
FROZEN_PROBE_ACTIONS = 64
FROZEN_HISTORY_SHUFFLE_BASE = 6_700_000
_TEMPERATURE_BY_SUFFIX = {"t0p6": 0.6, "t0p9": 0.9, "t1p2": 1.2}


@dataclass(frozen=True, slots=True)
class Phase3HistoryShuffleDiagnostics:
    """Search-time shuffle coverage and effective-change diagnostics."""

    eligible_windows: int
    map_nonidentity_windows: int
    effective_tensor_changed_windows: int
    duplicate_vector_no_effect_windows: int
    unchanged_short_windows: int
    permutation_map_sha256: str

    @property
    def effective_change_fraction(self) -> float:
        return (
            self.effective_tensor_changed_windows / self.eligible_windows
            if self.eligible_windows
            else 1.0
        )

    @property
    def claim_eligible(self) -> bool:
        return self.eligible_windows > 0 and self.effective_change_fraction >= 0.80


@dataclass(frozen=True, slots=True)
class Phase3GenerationAccounting:
    """Deterministic resource counters; wall time is diagnostic only."""

    episodes: int
    resets: int
    actions: int
    forward_passes: int
    recurrent_steps: int
    unknown_affordance_decisions: int
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class Phase3GeneratedCandidate:
    episode: int
    adaptation_actions: int
    trajectory: Trajectory


@dataclass(frozen=True, slots=True)
class Phase3GeneratedSearch:
    candidates: tuple[Phase3GeneratedCandidate, ...]
    accounting: Phase3GenerationAccounting
    candidate_generation_sha256: str
    history_shuffle: Phase3HistoryShuffleDiagnostics | None = None


def _model_and_lineage(
    model: Any,
    condition_id: str,
    *,
    planned_unit: Phase3PlannedUnit | None,
    plan_authority: ValidatedPhase3Plan | None,
    model_authority: Phase3ModelArtifactAuthority | None,
    task_id: str,
    seed: int,
    temperature: float,
    max_episodes: int,
    max_actions_per_episode: int,
    total_adaptation_action_cap: int,
    prior_adaptation_actions: int,
    fold_id: str,
    replicate: int,
    phase: str,
    history_shuffle_base: int,
    unit_id: str | None,
    allow_test_model: bool,
) -> Any:
    """Unwrap an authorized Phase 3 model and enforce frozen execution identity.

    Test doubles may be duck-typed, but real state/history scorer instances are
    checked strictly.  An authorized model owner, when present, must name the same
    condition as the execution request.
    """

    if not allow_test_model:
        if type(model) is not AuthorizedPhase3LoadedModel:
            raise ValueError("production generation requires an authorized Phase 3 model")
        if planned_unit is None:
            raise ValueError("production generation requires a frozen Phase 3 unit")
        if not isinstance(plan_authority, ValidatedPhase3Plan):
            raise ValueError("production generation requires validated plan authority")
        if not isinstance(model_authority, Phase3ModelArtifactAuthority):
            raise ValueError("production generation requires model authority")
        validate_authorized_phase3_loaded_model(
            model,
            model_authority,
            plan_authority,
            planned_unit,
        )
        plan_authority.require_unit(planned_unit)
        planned = planned_unit.unit
        if unit_id != planned.unit_id:
            raise ValueError("generation unit identity differs from the frozen unit")
        expected_variant = f"{condition_id}--{planned_unit.tuple_id}"
        if (
            planned_unit.base_condition_id != condition_id
            or planned.key.condition_id != expected_variant
            or planned.key.task_id != task_id
            or planned.key.family_id != planned_unit.heldout_family
            or planned.key.replicate != replicate
            or planned.key.phase != phase
            or planned.seeds.search_seed != seed
            or planned_unit.fold_id != fold_id
        ):
            raise ValueError("generation request differs from the frozen unit")
        temperature_suffix = planned_unit.tuple_id.rsplit("-", 1)[-1]
        expected_temperature = _TEMPERATURE_BY_SUFFIX.get(temperature_suffix)
        if expected_temperature is None or temperature != expected_temperature:
            raise ValueError("generation temperature differs from the frozen tuple")
        if (
            max_episodes != FROZEN_CANDIDATE_EPISODES
            or max_actions_per_episode != FROZEN_MAX_ACTIONS_PER_EPISODE
            or total_adaptation_action_cap != FROZEN_TOTAL_ADAPTATION_ACTION_CAP
            or prior_adaptation_actions != FROZEN_PROBE_ACTIONS
            or history_shuffle_base != FROZEN_HISTORY_SHUFFLE_BASE
        ):
            raise ValueError("generation budget differs from the frozen Phase 3 protocol")
        if model.planned_unit != planned_unit:
            raise ValueError("authorized model unit differs from the generation unit")
        owner = model.owner
        key = model.key
        if (
            owner.owner_id != planned_unit.model_owner_id
            or owner.view_id != planned_unit.view_id
            or owner.condition_id != condition_id
            or owner.fold_id != fold_id
            or owner.heldout_family != planned_unit.heldout_family
            or owner.replicate != replicate
            or owner.training_tuple_id != planned_unit.training_tuple_id
            or planned_unit.tuple_id not in owner.search_temperature_ids
            or key.owner_id != owner.owner_id
            or key.view_id != owner.view_id
            or key.condition_id != owner.condition_id
            or key.fold_id != owner.fold_id
            or key.heldout_family != owner.heldout_family
            or key.replicate != owner.replicate
            or key.training_tuple_id != owner.training_tuple_id
            or key.model_seed != owner.model_seed
        ):
            raise ValueError("authorized model owner/key differs from the frozen unit")
    owner = getattr(model, "owner", None)
    owner_condition = getattr(owner, "condition_id", None)
    if owner_condition is not None and owner_condition != condition_id:
        raise ValueError("authorized model condition lineage does not match generation")
    raw = getattr(model, "model", model)
    declared = getattr(raw, "condition_id", None)
    if declared is not None and declared != condition_id:
        raise ValueError("model condition lineage does not match generation")
    if condition_id == S_CONDITION:
        if not allow_test_model and type(raw) is not StateConditionedScorer:
            raise ValueError("S generation requires the exact frozen scorer class")
        if isinstance(raw, HistoryConditionedScorer):
            raise ValueError("S generation requires the state-availability model")
        if isinstance(raw, StateConditionedScorer):
            expected = 3841
            if sum(parameter.numel() for parameter in raw.parameters()) != expected:
                raise ValueError("S model capacity drifted")
    elif condition_id in HISTORY_CONDITIONS:
        if not allow_test_model and type(raw) is not HistoryConditionedScorer:
            raise ValueError("history generation requires the exact frozen scorer class")
        if isinstance(raw, StateConditionedScorer):
            raise ValueError("history generation requires the history-conditioned model")
        if isinstance(raw, HistoryConditionedScorer):
            if sum(parameter.numel() for parameter in raw.parameters()) != 3889:
                raise ValueError("history model capacity drifted")
    else:
        raise ValueError("unsupported Phase 3 generation condition")
    return raw


def _candidate_sha256(
    candidates: tuple[Phase3GeneratedCandidate, ...],
    *,
    condition_id: str,
    task_id: str,
    seed: int,
    max_episodes: int,
    max_actions_per_episode: int,
    total_adaptation_action_cap: int,
    prior_adaptation_actions: int,
    fold_id: str,
    replicate: int,
    phase: str,
    unit_id: str | None,
    history_shuffle_sha256: str | None,
) -> str:
    payload = {
        "schema_version": "milestone6.phase3.candidate-generation.v1",
        "condition_id": condition_id,
        "task_id": task_id,
        "seed": seed,
        "fold_id": fold_id,
        "replicate": replicate,
        "phase": phase,
        "unit_id": unit_id,
        "max_episodes": max_episodes,
        "max_actions_per_episode": max_actions_per_episode,
        "total_adaptation_action_cap": total_adaptation_action_cap,
        "prior_adaptation_actions": prior_adaptation_actions,
        "history_shuffle_sha256": history_shuffle_sha256,
        "candidates": [
            {
                "episode": candidate.episode,
                "adaptation_actions": candidate.adaptation_actions,
                "trajectory": candidate.trajectory.model_dump(mode="json"),
            }
            for candidate in candidates
        ],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _generate_phase3_candidates_impl(
    environment: ObservableEnvironment,
    *,
    task_id: str,
    forbidden_aliases: frozenset[str],
    affordances: AffordanceTable,
    model: Any,
    seed: int,
    temperature: float,
    max_episodes: int,
    max_actions_per_episode: int,
    total_adaptation_action_cap: int,
    prior_adaptation_actions: int,
    condition_id: str,
    fold_id: str = "phase3",
    replicate: int = 0,
    phase: str = "validation",
    history_shuffle_base: int = 6_700_000,
    unit_id: str | None = None,
    planned_unit: Phase3PlannedUnit | None = None,
    plan_authority: ValidatedPhase3Plan | None = None,
    model_authority: Phase3ModelArtifactAuthority | None = None,
    allow_test_model: bool,
) -> Phase3GeneratedSearch:
    """Generate a complete fixed-budget candidate batch for one Phase 3 unit.

    No argument represents an optimum, evaluator, or stopping threshold.  Every
    episode is reset and attempted, including after an earlier episode happens
    to reach an observable completed state. As in the locked Phase 2 search,
    incomplete attempts consume budget but only completed candidates are returned
    for independent replay.
    """

    if not task_id or not fold_id or not phase:
        raise ValueError("task, fold, and phase identities must be nonempty")
    if unit_id is not None and not unit_id:
        raise ValueError("unit identity must be nonempty when supplied")
    if replicate < 0 or max_episodes < 1 or max_actions_per_episode < 1:
        raise ValueError("replicate and episode/action budgets are invalid")
    if not 0 <= prior_adaptation_actions <= total_adaptation_action_cap:
        raise ValueError("prior adaptation actions exceed the total cap")
    if not allow_test_model and unit_id is None:
        raise ValueError("production generation requires the frozen unit identity")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    raw_model = _model_and_lineage(
        model,
        condition_id,
        planned_unit=planned_unit,
        plan_authority=plan_authority,
        model_authority=model_authority,
        task_id=task_id,
        seed=seed,
        temperature=temperature,
        max_episodes=max_episodes,
        max_actions_per_episode=max_actions_per_episode,
        total_adaptation_action_cap=total_adaptation_action_cap,
        prior_adaptation_actions=prior_adaptation_actions,
        fold_id=fold_id,
        replicate=replicate,
        phase=phase,
        history_shuffle_base=history_shuffle_base,
        unit_id=unit_id,
        allow_test_model=allow_test_model,
    )
    rng = random.Random(seed)
    started = time.perf_counter()
    candidates: list[Phase3GeneratedCandidate] = []
    episodes = resets = actions = forward_passes = recurrent_steps = unknown_decisions = 0
    permutation_records: list[dict[str, Any]] = []
    eligible = map_nonidentity = effective_changed = duplicate_no_effect = short = 0

    for episode in range(1, max_episodes + 1):
        if prior_adaptation_actions + actions >= total_adaptation_action_cap:
            break
        episodes = episode
        candidate_environment = environment.fresh()
        outcome = candidate_environment.reset(seed=0)
        resets += 1
        history: list[tuple[float, ...]] = []
        aliases_taken: list[str] = []
        trace_anchor = unit_id or condition_id
        trace_id = f"search:{trace_anchor}:{task_id}:s{seed}:e{episode}"
        for decision_index in range(max_actions_per_episode):
            if outcome.completed:
                break
            if prior_adaptation_actions + actions >= total_adaptation_action_cap:
                break
            state = parse_observation(outcome.observation, forbidden_aliases=forbidden_aliases)
            if not state.available_aliases:
                break

            if condition_id == S_CONDITION:
                weights, unknown = state_availability_visible_action_weights(
                    raw_model, state, affordances, temperature=temperature
                )
            else:
                window = history[-HISTORY_LENGTH:]
                input_indices = tuple(
                    range(max(0, decision_index - HISTORY_LENGTH), decision_index)
                )
                output_indices = input_indices
                if condition_id == H0_CONDITION:
                    history_tensor = torch.zeros(
                        (len(window), 12), dtype=torch.float32
                    )
                elif condition_id == H4_SHUFFLED_CONDITION:
                    identity = {
                        "fold_id": fold_id,
                        "replicate": replicate,
                        "task_id": task_id,
                        "phase": phase,
                        "trace_or_episode_id": trace_id,
                        "decision_index": decision_index,
                    }
                    permutation = deterministic_history_derangement(
                        len(input_indices),
                        history_shuffle_base=history_shuffle_base,
                        identity=HistoryPermutationIdentity(**identity),
                    )
                    output_indices = tuple(input_indices[index] for index in permutation)
                    history_tensor = (
                        torch.tensor(
                            [history[index] for index in output_indices],
                            dtype=torch.float32,
                        )
                        if output_indices
                        else torch.zeros((0, 12), dtype=torch.float32)
                    )
                    record = {
                        **identity,
                        "input_transition_indices": list(input_indices),
                        "permuted_transition_indices": list(output_indices),
                    }
                    permutation_records.append(record)
                    if len(input_indices) < 2:
                        short += 1
                    else:
                        eligible += 1
                        if input_indices != output_indices:
                            map_nonidentity += 1
                        original = torch.tensor(
                            [history[index] for index in input_indices], dtype=torch.float32
                        )
                        if torch.equal(original, history_tensor):
                            duplicate_no_effect += 1
                        else:
                            effective_changed += 1
                else:
                    history_tensor = (
                        torch.tensor(
                            [history[index] for index in input_indices],
                            dtype=torch.float32,
                        )
                        if input_indices
                        else torch.zeros((0, 12), dtype=torch.float32)
                    )
                weights, unknown = history_visible_action_weights(
                    raw_model, state, affordances, history_tensor, temperature=temperature
                )
                recurrent_steps += int(history_tensor.shape[0])

            forward_passes += 1
            unknown_decisions += unknown
            alias = rng.choices(
                tuple(weights), weights=tuple(weights.values()), k=1
            )[0]
            aliases_taken.append(alias)
            before = state
            outcome = candidate_environment.step(ActionRecord(name=alias))
            after = parse_observation(
                outcome.observation, forbidden_aliases=forbidden_aliases
            )
            history.append(
                transition_features(
                    ObservedTransition(before, alias, after, outcome.completed)
                )
            )
            actions += 1

        if not outcome.completed:
            continue
        trajectory = Trajectory(
            trajectory_id=trace_id,
            task_id=task_id,
            source="agent",
            environment_seed=0,
            steps=tuple(
                TrajectoryStep(index=index, action=ActionRecord(name=alias))
                for index, alias in enumerate(aliases_taken)
            ),
        )
        candidates.append(
            Phase3GeneratedCandidate(
                episode=episode,
                adaptation_actions=prior_adaptation_actions + actions,
                trajectory=trajectory,
            )
        )

    result_candidates = tuple(candidates)
    diagnostics = None
    if condition_id == H4_SHUFFLED_CONDITION:
        diagnostics = Phase3HistoryShuffleDiagnostics(
            eligible,
            map_nonidentity,
            effective_changed,
            duplicate_no_effect,
            short,
            permutation_map_sha256(permutation_records),
        )
    accounting = Phase3GenerationAccounting(
        episodes=episodes,
        resets=resets,
        actions=actions,
        forward_passes=forward_passes,
        recurrent_steps=recurrent_steps,
        unknown_affordance_decisions=unknown_decisions,
        wall_seconds=time.perf_counter() - started,
    )
    return Phase3GeneratedSearch(
        candidates=result_candidates,
        accounting=accounting,
        candidate_generation_sha256=_candidate_sha256(
            result_candidates,
            condition_id=condition_id,
            task_id=task_id,
            seed=seed,
            max_episodes=max_episodes,
            max_actions_per_episode=max_actions_per_episode,
            total_adaptation_action_cap=total_adaptation_action_cap,
            prior_adaptation_actions=prior_adaptation_actions,
            fold_id=fold_id,
            replicate=replicate,
            phase=phase,
            unit_id=unit_id,
            history_shuffle_sha256=(diagnostics.permutation_map_sha256 if diagnostics else None),
        ),
        history_shuffle=diagnostics,
    )


def generate_phase3_candidates_with_observable_policy(
    environment: ObservableEnvironment,
    *,
    task_id: str,
    forbidden_aliases: frozenset[str],
    affordances: AffordanceTable,
    model: AuthorizedPhase3LoadedModel,
    seed: int,
    temperature: float,
    max_episodes: int,
    max_actions_per_episode: int,
    total_adaptation_action_cap: int,
    prior_adaptation_actions: int,
    condition_id: str,
    fold_id: str = "phase3",
    replicate: int = 0,
    phase: str = "validation",
    history_shuffle_base: int = FROZEN_HISTORY_SHUFFLE_BASE,
    unit_id: str | None = None,
    planned_unit: Phase3PlannedUnit | None = None,
    plan_authority: ValidatedPhase3Plan | None = None,
    model_authority: Phase3ModelArtifactAuthority | None = None,
) -> Phase3GeneratedSearch:
    """Run the strict production path; no caller-visible test bypass exists."""

    return _generate_phase3_candidates_impl(
        environment,
        task_id=task_id,
        forbidden_aliases=forbidden_aliases,
        affordances=affordances,
        model=model,
        seed=seed,
        temperature=temperature,
        max_episodes=max_episodes,
        max_actions_per_episode=max_actions_per_episode,
        total_adaptation_action_cap=total_adaptation_action_cap,
        prior_adaptation_actions=prior_adaptation_actions,
        condition_id=condition_id,
        fold_id=fold_id,
        replicate=replicate,
        phase=phase,
        history_shuffle_base=history_shuffle_base,
        unit_id=unit_id,
        planned_unit=planned_unit,
        plan_authority=plan_authority,
        model_authority=model_authority,
        allow_test_model=False,
    )


def _generate_phase3_candidates_with_test_model(
    environment: ObservableEnvironment,
    **kwargs: Any,
) -> Phase3GeneratedSearch:
    """Private adapter for bounded unit tests with synthetic models and budgets."""

    forbidden = {"allow_test_model", "_allow_test_model"}.intersection(kwargs)
    if forbidden:
        raise TypeError("test-model mode is selected only by the private adapter")
    return _generate_phase3_candidates_impl(
        environment,
        **kwargs,
        allow_test_model=True,
    )


# A short alias keeps call sites parallel with the historical Phase 2 helper.
generate_candidates_with_phase3_observable_policy = (
    generate_phase3_candidates_with_observable_policy
)
