from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from levelup.core.trajectory import ActionRecord
from levelup.experiments.milestone6_baselines import ProbeEvidence, discover_affordances
from levelup.experiments.milestone6_phase2 import _environment, _forbidden_aliases
from levelup.experiments.milestone6_phase3_execution import _canonical_validation_tasks
from levelup.experiments.milestone6_phase3_outcome_diagnostic_generation import (
    FROZEN_CANDIDATE_EPISODES,
    FROZEN_PROBE_ACTIONS,
    FROZEN_TOTAL_ADAPTATION_ACTION_CAP,
    PEC_CONDITION,
    RP_CONDITION,
    AuthorizedOutcomeGenerationModel,
    AuthorizedOutcomeProbeContext,
    _authorize_outcome_generation_model_for_test,
    _authorize_outcome_probe_context_for_test,
    _candidate_generation_sha256,
    authorize_outcome_generation_model,
    authorize_outcome_probe_context,
    generate_outcome_group_candidates_with_observable_policy,
    masked_visible_action_weights,
    outcome_group_training_examples,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    OutcomePlannedUnit,
    ValidatedOutcomePlan,
    bind_validated_outcome_diagnostic_plan,
    build_outcome_group_diagnostic_plan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    load_outcome_group_diagnostic_protocol,
)
from levelup.learning.state_conditioned import (
    AffordanceTable,
    ObservableState,
    ObservableTrace,
    ObservedTransition,
    StateConditionedScorer,
    apply_progress_elapsed_completion_mask,
    apply_resource_pressure_mask,
    candidate_tensor,
)


@pytest.fixture(scope="module")
def protocol_snapshot():
    return load_outcome_group_diagnostic_protocol()


@pytest.fixture(scope="module")
def validated_plan(protocol_snapshot) -> ValidatedOutcomePlan:
    return bind_validated_outcome_diagnostic_plan(
        build_outcome_group_diagnostic_plan(protocol_snapshot), snapshot=protocol_snapshot
    )


def _unit(plan: ValidatedOutcomePlan, condition: str = RP_CONDITION) -> OutcomePlannedUnit:
    return next(item for item in plan.plan.units if item.condition_id == condition)


def _state(progress: int, aliases: tuple[str, ...] = ("a", "b")) -> ObservableState:
    return ObservableState(
        progress / 10,
        (10 - progress) / 10,
        progress / 10,
        0.4 + progress / 100,
        0.2 + progress / 100,
        aliases,
    )


def _affordances() -> AffordanceTable:
    return AffordanceTable(
        features={"a": (0.1,) * 49, "b": (0.2,) * 49},
        sample_counts={"a": 2, "b": 2},
    )


def _generation_kwargs(
    unit: OutcomePlannedUnit,
    plan: ValidatedOutcomePlan,
    snapshot,
) -> dict:
    environment = _Environment(complete_after=1)
    model = _authorize_outcome_generation_model_for_test(
        StateConditionedScorer(), unit, plan, snapshot
    )
    probe = _authorize_outcome_probe_context_for_test(
        environment,
        unit,
        plan,
        snapshot,
        affordances=_affordances(),
        forbidden_aliases=frozenset(),
    )
    return {
        "model": model,
        "probe": probe,
        "planned_unit": unit,
        "plan": plan,
        "protocol_snapshot": snapshot,
    }


class _Outcome:
    def __init__(self, observation: dict, completed: bool = False) -> None:
        self.observation = observation
        self.completed = completed


def _observation(progress: int) -> dict:
    return {
        "progress": progress,
        "target": 10,
        "elapsed_ticks": progress,
        "resource_fraction": 0.4,
        "pressure_fraction": 0.2,
        "available_actions": [{"alias": "a"}, {"alias": "b"}],
    }


class _Environment:
    def __init__(self, complete_after: int | None = 1) -> None:
        self.complete_after = complete_after
        self.steps = 0

    def fresh(self) -> "_Environment":
        return _Environment(self.complete_after)

    def reset(self, seed: int | None = None) -> _Outcome:
        assert seed == 0
        self.steps = 0
        return _Outcome(_observation(0))

    def step(self, action: ActionRecord) -> _Outcome:
        assert action.name in {"a", "b"}
        self.steps += 1
        return _Outcome(_observation(self.steps), self.complete_after == self.steps)


def test_mask_dispatch_uses_one_shared_example_source_and_preserves_labels_order() -> None:
    trace = ObservableTrace(
        (
            ObservedTransition(_state(0), "a", _state(1), False),
            ObservedTransition(_state(1), "b", _state(2), True),
        )
    )
    samples = ((trace, _affordances()),)
    rp = outcome_group_training_examples(samples, RP_CONDITION)
    pec = outcome_group_training_examples(samples, PEC_CONDITION)
    assert [example.selected_index for example in rp] == [0, 1]
    assert [example.selected_index for example in pec] == [0, 1]
    source, _, _ = candidate_tensor(_state(0), _affordances())
    del source
    assert torch.equal(
        rp[0].candidate_features,
        apply_resource_pressure_mask(
            # Reconstruct the exact source candidate tensor, then compare mask bytes.
            candidate_tensor(_state(0), _affordances())[1]
        ),
    )
    assert torch.equal(
        pec[0].candidate_features,
        apply_progress_elapsed_completion_mask(candidate_tensor(_state(0), _affordances())[1]),
    )


def test_masked_logits_and_unknown_alias_fallback_match_s_semantics() -> None:
    model = StateConditionedScorer().eval()
    state = _state(0, ("a", "unknown"))
    weights, unknown = masked_visible_action_weights(
        model,
        state,
        _affordances(),
        condition_id=RP_CONDITION,
        temperature=1.2,
    )
    assert unknown == 1
    assert set(weights) == {"a", "unknown"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["unknown"] == pytest.approx(weights["a"])


def test_generation_is_deterministic_fixed_endpoint_and_hash_has_no_oracle_channel(
    validated_plan: ValidatedOutcomePlan,
    protocol_snapshot,
) -> None:
    unit = _unit(validated_plan)
    plan = validated_plan
    kwargs = _generation_kwargs(unit, plan, protocol_snapshot)
    first = generate_outcome_group_candidates_with_observable_policy(
        **kwargs,
    )
    second = generate_outcome_group_candidates_with_observable_policy(
        **kwargs,
    )
    assert first.candidates == second.candidates
    assert first.candidate_generation_sha256 == second.candidate_generation_sha256
    assert first.accounting.episodes == FROZEN_CANDIDATE_EPISODES
    assert first.accounting.actions == FROZEN_CANDIDATE_EPISODES
    assert first.accounting.actions < FROZEN_TOTAL_ADAPTATION_ACTION_CAP
    assert len(first.candidates) == FROZEN_CANDIDATE_EPISODES
    pec_unit = _unit(validated_plan, PEC_CONDITION)
    pec = generate_outcome_group_candidates_with_observable_policy(
        **_generation_kwargs(pec_unit, validated_plan, protocol_snapshot),
    )
    assert len(pec.candidates) == FROZEN_CANDIDATE_EPISODES
    with pytest.raises(TypeError):
        generate_outcome_group_candidates_with_observable_policy(  # type: ignore[call-arg]
            **_generation_kwargs(unit, plan, protocol_snapshot),
            oracle=SimpleNamespace(),
        )


def test_generation_rejects_final_units_and_identity_overrides(
    validated_plan: ValidatedOutcomePlan,
    protocol_snapshot,
) -> None:
    unit = replace(_unit(validated_plan), final_family_access=True)
    with pytest.raises(ValueError, match="final"):
        generate_outcome_group_candidates_with_observable_policy(
            **_generation_kwargs(unit, validated_plan, protocol_snapshot),
        )
    unit = _unit(validated_plan)
    plan = validated_plan
    kwargs = _generation_kwargs(unit, plan, protocol_snapshot)
    with pytest.raises(ValueError, match="artifact validator"):
        AuthorizedOutcomeGenerationModel(StateConditionedScorer(), unit, "a" * 64, "forged")
    with pytest.raises(ValueError, match="probe validator"):
        AuthorizedOutcomeProbeContext(unit, _Environment(), _affordances(), frozenset(), "a" * 64)
    with pytest.raises((TypeError, ValueError)):
        generate_outcome_group_candidates_with_observable_policy(
            **{**kwargs, "model": StateConditionedScorer()},
        )
    raw_model = StateConditionedScorer().eval()
    for parameter in raw_model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="validated outcome artifact"):
        authorize_outcome_generation_model(
            raw_model,
            SimpleNamespace(record=SimpleNamespace()),
            unit,
            plan,
            protocol_snapshot,
        )
    tampered_model = _generation_kwargs(unit, plan, protocol_snapshot)
    with torch.no_grad():
        next(tampered_model["model"].model.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="model authorization"):
        generate_outcome_group_candidates_with_observable_policy(**tampered_model)
    tampered_probe = _generation_kwargs(unit, plan, protocol_snapshot)
    tampered_probe["probe"].affordances.features["a"] = (0.9,) * 49
    with pytest.raises(ValueError, match="probe authorization"):
        generate_outcome_group_candidates_with_observable_policy(**tampered_probe)
    with pytest.raises(TypeError):
        generate_outcome_group_candidates_with_observable_policy(
            **kwargs,
            forbidden_aliases=frozenset(),
        )
    with pytest.raises(TypeError):
        generate_outcome_group_candidates_with_observable_policy(
            **kwargs,
            environment=_Environment(),
        )


def test_production_probe_authorization_binds_canonical_task_and_paid_evidence(
    validated_plan: ValidatedOutcomePlan,
    protocol_snapshot,
) -> None:
    unit = _unit(validated_plan)
    task = next(item for item in _canonical_validation_tasks() if item.task_id == unit.task_id)
    environment = _environment(task)
    forbidden_aliases = _forbidden_aliases(environment)
    evidence = discover_affordances(
        environment,
        task_id=task.task_id,
        forbidden_aliases=forbidden_aliases,
        seed=unit.probe_seed,
        action_cap=FROZEN_PROBE_ACTIONS,
        target_samples_per_alias=8,
        actions_per_attempt=16,
    )
    context = authorize_outcome_probe_context(
        environment,
        task,
        evidence,
        unit,
        validated_plan,
        protocol_snapshot,
    )
    assert context.unit_id == unit.unit_id
    assert context.task_id == task.task_id
    assert context.task_index == task.task_index
    assert context.family_id == task.family_id
    assert context.environment_seed == task.environment_reset_seed == 0
    assert context.probe_seed == unit.probe_seed
    assert context.probe_actions == FROZEN_PROBE_ACTIONS
    assert context.forbidden_aliases == forbidden_aliases
    assert context.affordance_sha256 == _affordance_identity_for_test(evidence)


def _affordance_identity_for_test(evidence: ProbeEvidence) -> str:
    """Keep the production identity assertion independent of private context state."""

    import hashlib

    from levelup.experiments.runner.config import canonical_json_bytes

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "features": {
                    alias: list(evidence.affordances.features[alias])
                    for alias in sorted(evidence.affordances.features)
                },
                "sample_counts": {
                    alias: evidence.affordances.sample_counts[alias]
                    for alias in sorted(evidence.affordances.sample_counts)
                },
            }
        )
    ).hexdigest()


def test_production_probe_authorization_rejects_foreign_identity_budget_and_forgery(
    validated_plan: ValidatedOutcomePlan,
    protocol_snapshot,
) -> None:
    unit = _unit(validated_plan)
    task = next(item for item in _canonical_validation_tasks() if item.task_id == unit.task_id)
    environment = _environment(task)
    forbidden_aliases = _forbidden_aliases(environment)
    evidence = discover_affordances(
        environment,
        task_id=task.task_id,
        forbidden_aliases=forbidden_aliases,
        seed=unit.probe_seed,
        action_cap=FROZEN_PROBE_ACTIONS,
        target_samples_per_alias=8,
        actions_per_attempt=16,
    )
    args = (environment, task, evidence, unit, validated_plan, protocol_snapshot)
    foreign_task = next(item for item in _canonical_validation_tasks() if item.family_id != task.family_id)
    with pytest.raises(ValueError, match="task identity|environment"):
        authorize_outcome_probe_context(environment, foreign_task, evidence, unit, validated_plan, protocol_snapshot)
    foreign_environment = _environment(foreign_task)
    with pytest.raises(ValueError, match="task|environment"):
        authorize_outcome_probe_context(foreign_environment, task, evidence, unit, validated_plan, protocol_snapshot)
    with pytest.raises(ValueError, match="plan|unit|seed"):
        authorize_outcome_probe_context(
            *args[:3],
            replace(unit, probe_seed=unit.probe_seed + 1),
            validated_plan,
            protocol_snapshot,
        )
    with pytest.raises(ValueError, match="64"):
        authorize_outcome_probe_context(
            environment,
            task,
            replace(evidence, accounting=replace(evidence.accounting, actions=63)),
            unit,
            validated_plan,
            protocol_snapshot,
        )
    with pytest.raises(ValueError, match="typed"):
        authorize_outcome_probe_context(
            environment,
            task,
            SimpleNamespace(**evidence.__dict__) if hasattr(evidence, "__dict__") else SimpleNamespace(),
            unit,
            validated_plan,
            protocol_snapshot,
        )
    with pytest.raises(ValueError, match="unit|plan"):
        authorize_outcome_probe_context(
            environment,
            task,
            evidence,
            replace(unit, final_family_access=True),
            validated_plan,
            protocol_snapshot,
        )


def test_candidate_hash_binds_deterministic_search_accounting(
    validated_plan: ValidatedOutcomePlan,
    protocol_snapshot,
) -> None:
    unit = _unit(validated_plan)
    kwargs = _generation_kwargs(unit, validated_plan, protocol_snapshot)
    search = generate_outcome_group_candidates_with_observable_policy(**kwargs)
    changed = replace(search.accounting, actions=search.accounting.actions + 1)
    assert changed.wall_seconds == search.accounting.wall_seconds
    assert (
        _candidate_generation_sha256(
            search.candidates,
            unit=unit,
            model_state_sha256=kwargs["model"].model_state_sha256,
            artifact_record_id=kwargs["model"].artifact_record_id,
            probe=kwargs["probe"],
            accounting=changed,
        )
        != search.candidate_generation_sha256
    )
    assert search.accounting.planned_episode_cap == FROZEN_CANDIDATE_EPISODES
