from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import levelup.experiments.milestone6_phase3_execution_models as execution_models
import levelup.experiments.milestone6_phase3_generation as generation
from levelup.core.trajectory import ActionRecord
from levelup.experiments.milestone6_phase3_generation import (
    H0_CONDITION,
    H4_CONDITION,
    H4_SHUFFLED_CONDITION,
    S_CONDITION,
)
from levelup.experiments.milestone6_phase3_generation import (
    _generate_phase3_candidates_with_test_model as generate_phase3_candidates_with_observable_policy,
)
from levelup.experiments.milestone6_phase3_model_authority import (
    load_phase3_model_artifact_authority_bytes,
)
from levelup.experiments.milestone6_phase3_plan import (
    bind_validated_phase3_plan,
    build_phase3_plan,
)
from levelup.learning.state_conditioned import AffordanceTable, StateConditionedScorer


def _observation(progress: int, *, actions: list[str] | None = None) -> dict:
    return {
        "progress": progress,
        "target": 99,
        "elapsed_ticks": progress,
        "resource_fraction": 0.5,
        "pressure_fraction": 0.25,
        "available_actions": [{"alias": alias} for alias in (actions or ["a", "b"])],
    }


class _FakeOutcome:
    def __init__(self, observation: dict, completed: bool = False) -> None:
        self.observation = observation
        self.completed = completed


class _FakeEnvironment:
    def __init__(self, *, complete_after: int | None = None) -> None:
        self.complete_after = complete_after
        self.steps = 0
        self.reset_count = 0

    def fresh(self) -> "_FakeEnvironment":
        return _FakeEnvironment(complete_after=self.complete_after)

    def reset(self, seed: int | None = None) -> _FakeOutcome:
        del seed
        self.steps = 0
        self.reset_count += 1
        return _FakeOutcome(_observation(0))

    def step(self, action: ActionRecord) -> _FakeOutcome:
        assert action.name in {"a", "b"}
        self.steps += 1
        completed = self.complete_after is not None and self.steps >= self.complete_after
        return _FakeOutcome(_observation(self.steps), completed)


class _FakeStateModel:
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(features.shape[0], dtype=torch.float32)


class _FakeHistoryModel:
    def __call__(
        self, features: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        assert history.ndim == 2
        return torch.zeros(features.shape[0], dtype=torch.float32)


def _affordances() -> AffordanceTable:
    return AffordanceTable(
        features={"a": (0.1,) * 49, "b": (0.2,) * 49},
        sample_counts={"a": 2, "b": 2},
    )


def _generate(condition: str, *, complete_after: int | None = None, episodes: int = 3):
    return generate_phase3_candidates_with_observable_policy(
        _FakeEnvironment(complete_after=complete_after),
        task_id="task-0",
        forbidden_aliases=frozenset(),
        affordances=_affordances(),
        model=_FakeStateModel() if condition == S_CONDITION else _FakeHistoryModel(),
        seed=17,
        temperature=0.9,
        max_episodes=episodes,
        max_actions_per_episode=5,
        total_adaptation_action_cap=episodes * 5,
        prior_adaptation_actions=0,
        condition_id=condition,
        fold_id="fold-0",
        replicate=0,
        phase="validation",
    )


def test_fixed_batch_is_not_stopped_by_observable_completion() -> None:
    generated = _generate(S_CONDITION, complete_after=1, episodes=4)
    assert len(generated.candidates) == 4
    assert generated.accounting.episodes == 4
    assert generated.accounting.resets == 4
    assert generated.accounting.actions == 4
    assert all(len(candidate.trajectory.steps) == 1 for candidate in generated.candidates)


def test_generation_is_evaluator_free_and_hashes_complete_batch() -> None:
    generated = _generate(S_CONDITION, episodes=2)
    repeated = _generate(S_CONDITION, episodes=2)
    assert generated.candidate_generation_sha256 == repeated.candidate_generation_sha256
    assert generated.accounting.forward_passes == generated.accounting.actions
    assert generated.accounting.recurrent_steps == 0
    assert all(candidate.trajectory.source == "agent" for candidate in generated.candidates)


def test_history_conditions_have_matched_recurrent_resource_counts() -> None:
    h0 = _generate(H0_CONDITION, episodes=1)
    h4 = _generate(H4_CONDITION, episodes=1)
    shuffled = _generate(H4_SHUFFLED_CONDITION, episodes=1)
    assert (
        h0.accounting.recurrent_steps
        == h4.accounting.recurrent_steps
        == shuffled.accounting.recurrent_steps
    )
    assert h0.accounting.recurrent_steps == 10  # windows: 0, 1, 2, 3, 4
    assert h0.accounting.forward_passes == h4.accounting.forward_passes == 5


def test_shuffled_history_is_deterministic_and_reports_effective_changes() -> None:
    first = _generate(H4_SHUFFLED_CONDITION, episodes=1)
    second = _generate(H4_SHUFFLED_CONDITION, episodes=1)
    assert first.candidate_generation_sha256 == second.candidate_generation_sha256
    assert first.history_shuffle == second.history_shuffle
    assert first.history_shuffle is not None
    assert first.history_shuffle.eligible_windows == 3
    assert first.history_shuffle.map_nonidentity_windows == 3
    assert first.history_shuffle.permutation_map_sha256


def test_unknown_affordances_use_phase2_neutral_fallback() -> None:
    generated = generate_phase3_candidates_with_observable_policy(
        _FakeEnvironment(),
        task_id="task-0",
        forbidden_aliases=frozenset(),
        affordances=AffordanceTable(
            features={"a": (0.1,) * 49}, sample_counts={"a": 2}
        ),
        model=_FakeStateModel(),
        seed=1,
        temperature=1.0,
        max_episodes=1,
        max_actions_per_episode=1,
        total_adaptation_action_cap=1,
        prior_adaptation_actions=0,
        condition_id=S_CONDITION,
    )
    assert generated.accounting.unknown_affordance_decisions == 1


def test_observable_action_cap_is_the_only_batch_stopping_rule() -> None:
    generated = generate_phase3_candidates_with_observable_policy(
        _FakeEnvironment(),
        task_id="task-0",
        forbidden_aliases=frozenset(),
        affordances=_affordances(),
        model=_FakeStateModel(),
        seed=1,
        temperature=1.0,
        max_episodes=5,
        max_actions_per_episode=5,
        total_adaptation_action_cap=7,
        prior_adaptation_actions=0,
        condition_id=S_CONDITION,
    )
    assert generated.accounting.episodes == 2
    assert generated.accounting.resets == 2
    assert generated.accounting.actions == 7
    assert generated.candidates == ()


def test_prior_probe_actions_are_included_in_candidate_endpoint_and_hash() -> None:
    common = dict(
        environment=_FakeEnvironment(complete_after=2),
        task_id="task-0",
        forbidden_aliases=frozenset(),
        affordances=_affordances(),
        model=_FakeStateModel(),
        seed=1,
        temperature=1.0,
        max_episodes=3,
        max_actions_per_episode=5,
        condition_id=S_CONDITION,
    )
    with_probe = generate_phase3_candidates_with_observable_policy(
        **common,
        total_adaptation_action_cap=66,
        prior_adaptation_actions=64,
    )
    without_probe = generate_phase3_candidates_with_observable_policy(
        **common,
        total_adaptation_action_cap=2,
        prior_adaptation_actions=0,
    )
    assert with_probe.accounting.actions == 2
    assert with_probe.candidates[0].adaptation_actions == 66
    assert without_probe.candidates[0].adaptation_actions == 2
    assert (
        with_probe.candidate_generation_sha256
        != without_probe.candidate_generation_sha256
    )


def test_production_lineage_accepts_only_authorized_wrapper_without_view_adapter(
    monkeypatch,
) -> None:
    authority = load_phase3_model_artifact_authority_bytes(
        Path("configs/milestone6/phase3_model_artifact_authority.json").read_bytes()
    )
    plan = bind_validated_phase3_plan(build_phase3_plan())
    planned = next(
        item for item in plan.plan.units if item.base_condition_id == S_CONDITION
    )
    owner = next(
        item for item in plan.plan.model_owners if item.owner_id == planned.model_owner_id
    )
    raw = StateConditionedScorer().eval()
    for parameter in raw.parameters():
        parameter.requires_grad_(False)
    key = SimpleNamespace(
        owner_id=owner.owner_id,
        view_id=owner.view_id,
        condition_id=owner.condition_id,
        fold_id=owner.fold_id,
        heldout_family=owner.heldout_family,
        replicate=owner.replicate,
        training_tuple_id=owner.training_tuple_id,
        model_seed=owner.model_seed,
    )
    loaded = execution_models.AuthorizedPhase3LoadedModel(
        model=raw,
        planned_unit=planned,
        owner=owner,
        key=key,  # type: ignore[arg-type]
        index=object(),  # type: ignore[arg-type]
        cost=object(),  # type: ignore[arg-type]
        manifest=object(),  # type: ignore[arg-type]
        _construction_token=execution_models._CONSTRUCTION_TOKEN,
    )
    monkeypatch.setattr(generation, "validate_authorized_phase3_loaded_model", lambda *_: None)
    temperature = {"t0p6": 0.6, "t0p9": 0.9, "t1p2": 1.2}[
        planned.tuple_id.rsplit("-", 1)[-1]
    ]
    resolved = generation._model_and_lineage(
        loaded,
        S_CONDITION,
        planned_unit=planned,
        plan_authority=plan,
        model_authority=authority,
        task_id=planned.unit.key.task_id,
        seed=planned.unit.seeds.search_seed,
        temperature=temperature,
        max_episodes=generation.FROZEN_CANDIDATE_EPISODES,
        max_actions_per_episode=generation.FROZEN_MAX_ACTIONS_PER_EPISODE,
        total_adaptation_action_cap=generation.FROZEN_TOTAL_ADAPTATION_ACTION_CAP,
        prior_adaptation_actions=generation.FROZEN_PROBE_ACTIONS,
        fold_id=planned.fold_id,
        replicate=planned.unit.key.replicate,
        phase="validation",
        history_shuffle_base=generation.FROZEN_HISTORY_SHUFFLE_BASE,
        unit_id=planned.unit.unit_id,
        allow_test_model=False,
    )
    assert resolved is raw
    with pytest.raises(ValueError, match="authorized"):
        generation._model_and_lineage(
            raw,
            S_CONDITION,
            planned_unit=planned,
            plan_authority=plan,
            model_authority=authority,
            task_id=planned.unit.key.task_id,
            seed=planned.unit.seeds.search_seed,
            temperature=temperature,
            max_episodes=generation.FROZEN_CANDIDATE_EPISODES,
            max_actions_per_episode=generation.FROZEN_MAX_ACTIONS_PER_EPISODE,
            total_adaptation_action_cap=generation.FROZEN_TOTAL_ADAPTATION_ACTION_CAP,
            prior_adaptation_actions=generation.FROZEN_PROBE_ACTIONS,
            fold_id=planned.fold_id,
            replicate=planned.unit.key.replicate,
            phase="validation",
            history_shuffle_base=generation.FROZEN_HISTORY_SHUFFLE_BASE,
            unit_id=planned.unit.unit_id,
            allow_test_model=False,
        )


def test_production_generation_signature_has_no_test_bypass() -> None:
    parameters = inspect.signature(
        generation.generate_phase3_candidates_with_observable_policy
    ).parameters
    assert "_allow_test_model" not in parameters
    assert "allow_test_model" not in parameters
