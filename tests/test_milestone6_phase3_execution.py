from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from levelup.experiments import milestone6_phase3_execution as execution
from levelup.experiments.milestone6_phase3_model_authority import (
    load_phase3_model_artifact_authority_bytes,
)
from levelup.experiments.milestone6_phase3_plan import (
    bind_validated_phase3_plan,
    build_phase3_plan,
)


def _planned(condition: str = execution.S_CONDITION if hasattr(execution, "S_CONDITION") else "S-state-availability-listwise-optimum"):
    return SimpleNamespace(
        unit=SimpleNamespace(
            unit_id="a" * 64,
            key=SimpleNamespace(
                phase="validation",
                condition_id=f"{condition}--lr0p003-e120-t0p9",
                family_id="plain",
                task_id="plain-validation-0",
                task_index=0,
                replicate=0,
            ),
            seeds=SimpleNamespace(
                model_seed=1,
                environment_seed=2,
                probe_seed=3,
                search_seed=4,
                data_order_seed=5,
            ),
        ),
        base_condition_id=condition,
        tuple_id="lr0p003-e120-t0p9",
        training_tuple_id="lr0p003-e120",
        fold_id="fold-plain",
        heldout_family="plain",
        model_owner_id="b" * 64,
        view_id="c" * 64,
    )


def _context():
    plan = SimpleNamespace(
        plan=SimpleNamespace(final_family_access=False, units=()),
        require_unit=lambda unit: None,
    )
    return execution.Phase3ExecutionContext(
        authority=object(),
        plan=plan,
        artifact_output_root="/tmp/phase3-models",
    )


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    success: bool = False,
    condition: str = "S-state-availability-listwise-optimum",
):
    planned = _planned(condition)
    order: list[str] = []
    monkeypatch.setattr(execution, "_resolve_planned_unit", lambda context, unit: planned)
    monkeypatch.setattr(execution, "_resolve_task", lambda unit: SimpleNamespace(
        task_id="plain-validation-0", family_id="plain", task_index=0
    ))
    environment = SimpleNamespace(fresh=lambda: environment)
    monkeypatch.setattr(execution, "_environment", lambda task: environment)
    monkeypatch.setattr(execution, "_forbidden_aliases", lambda env: frozenset())
    monkeypatch.setattr(
        execution,
        "discover_affordances",
        lambda *args, **kwargs: SimpleNamespace(
            affordances=object(),
            accounting=SimpleNamespace(
                attempts=4, actions=64, resets=4, wall_seconds=0.0
            ),
        ),
    )

    class FakeModel:
        key = SimpleNamespace(
            key_id="1" * 64,
            report=SimpleNamespace(
                trainable_parameters=3841,
                optimizer_steps=120,
                forward_passes=1000,
                recurrent_steps=0,
                training_examples=10,
            )
        )
        index = SimpleNamespace(artifact_id="2" * 64)
        cost = SimpleNamespace(cost_id="3" * 64)

    monkeypatch.setattr(execution, "AuthorizedPhase3LoadedModel", FakeModel)

    @contextmanager
    def open_model(*args, **kwargs):
        order.append("model_open")
        try:
            yield FakeModel()
        finally:
            order.append("model_closed")

    monkeypatch.setattr(execution, "open_authorized_phase3_model", open_model)
    shuffle = (
        SimpleNamespace(
            claim_eligible=True,
            eligible_windows=10,
            map_nonidentity_windows=10,
            effective_tensor_changed_windows=8,
            duplicate_vector_no_effect_windows=2,
            unchanged_short_windows=3,
            permutation_map_sha256="e" * 64,
        )
        if condition == "H4-shuffled-history-transition-listwise-optimum"
        else None
    )
    generated = SimpleNamespace(
        candidates=(SimpleNamespace(episode=1, adaptation_actions=65, trajectory=object()),),
        accounting=SimpleNamespace(
            episodes=1,
            actions=65,
            resets=1,
            forward_passes=2,
            generation_wall_seconds=0.0,
            recurrent_steps=3,
            unknown_affordance_decisions=1,
        ),
        candidate_generation_sha256="d" * 64,
        history_shuffle=shuffle,
    )
    calls = []

    def generate(*args, **kwargs):
        order.append("generation")
        calls.append(kwargs)
        return generated

    monkeypatch.setattr(execution, "generate_phase3_candidates_with_observable_policy", generate)
    replay = SimpleNamespace(
        evaluated_candidates=(SimpleNamespace(episode=1, adaptation_actions=65),)
        if success
        else (),
        first_valid_episode=1 if success else None,
        best_performance=3.0 if success else None,
        accounting=SimpleNamespace(
            episodes=1,
            actions=65,
            resets=1,
            forward_passes=2,
            evaluator_calls=1,
            evaluator_replay_actions=2,
            generation_wall_seconds=0.0,
            evaluator_wall_seconds=0.0,
        ),
    )
    def evaluate(*args, **kwargs):
        order.append("replay")
        return replay

    monkeypatch.setattr(execution, "evaluate_generated_search", evaluate)
    monkeypatch.setattr(
        execution,
        "classify_exact_optimum",
        lambda *a, **k: SimpleNamespace(
            first_episode=1 if success else None,
            first_adaptation_actions=65 if success else None,
            success=success,
        ),
    )
    def optimum(*args, **kwargs):
        order.append("oracle")
        return 3.0

    monkeypatch.setattr(execution, "_default_optimum_provider", optimum)
    return planned, calls, order


def test_one_unit_uses_frozen_budgets_and_replay_before_oracle(monkeypatch: pytest.MonkeyPatch):
    planned, calls, order = _patch_runtime(monkeypatch, success=True)
    events: list[str] = []
    payload = execution.execute_phase3_unit(
        _context(), planned, event=events.append
    )
    assert events == ["generation_complete", "candidate_evaluation_complete", "optimum_oracle"]
    assert order == ["model_open", "generation", "model_closed", "replay", "oracle"]
    assert calls[0]["seed"] == planned.unit.seeds.search_seed
    assert calls[0]["temperature"] == pytest.approx(0.9)
    assert calls[0]["max_episodes"] == 150
    assert calls[0]["max_actions_per_episode"] == 64
    assert calls[0]["total_adaptation_action_cap"] == 2048
    assert calls[0]["prior_adaptation_actions"] == 64
    assert payload.accounting.training.calls == 0
    assert payload.shared_artifact is not None
    assert payload.shared_artifact.key_id == "1" * 64
    assert payload.diagnostics["model_optimizer_steps"] == 120
    assert payload.diagnostics["model_forward_passes"] == 1000
    assert payload.diagnostics["model_recurrent_steps"] == 0
    assert payload.outcome.success is True
    assert payload.candidate_generation_sha256 == "d" * 64
    assert payload.history_shuffle_permutation_map_sha256 is None


def test_failed_unit_is_typed_fixed_endpoint(monkeypatch: pytest.MonkeyPatch):
    planned, _, _ = _patch_runtime(monkeypatch, success=False)
    payload = execution.execute_phase3_unit(_context(), planned)
    assert payload.outcome.censored is True
    assert payload.outcome.censoring_budget == 2048
    assert payload.outcome.censoring_reason == "fixed_endpoint"


def test_shuffled_unit_persists_permutation_map_digest(monkeypatch: pytest.MonkeyPatch):
    planned, _, _ = _patch_runtime(
        monkeypatch,
        condition="H4-shuffled-history-transition-listwise-optimum",
    )
    payload = execution.execute_phase3_unit(_context(), planned)
    assert payload.history_shuffle_permutation_map_sha256 == "e" * 64
    assert payload.diagnostics["history_shuffle_claim_eligible"] is True


def test_oracle_substitution_cannot_change_candidate_hash(monkeypatch: pytest.MonkeyPatch):
    planned, _, _ = _patch_runtime(monkeypatch, success=False)
    first = execution.execute_phase3_unit(_context(), planned)
    monkeypatch.setattr(execution, "_default_optimum_provider", lambda *_: 4.0)
    second = execution.execute_phase3_unit(_context(), planned)
    assert first.candidate_generation_sha256 == second.candidate_generation_sha256


def test_final_unit_is_rejected_before_any_runtime() -> None:
    authority = load_phase3_model_artifact_authority_bytes(
        Path("configs/milestone6/phase3_model_artifact_authority.json").read_bytes()
    )
    plan = bind_validated_phase3_plan(build_phase3_plan())
    planned = plan.plan.units[0]
    final_key = planned.unit.key.model_copy(update={"phase": "final"})
    final_unit = planned.unit.model_copy(update={"key": final_key})
    forged = replace(planned, unit=final_unit)
    context = execution.Phase3ExecutionContext(authority, plan, "/tmp/not-opened")
    with pytest.raises(ValueError, match="validation"):
        execution._resolve_planned_unit(context, forged)


def test_task_and_oracle_are_not_caller_override_surfaces() -> None:
    assert "task_resolver" not in execution.Phase3ExecutionContext.__dataclass_fields__
    assert "_test_oracle_provider" not in inspect.signature(
        execution.execute_phase3_unit
    ).parameters
    plan = bind_validated_phase3_plan(build_phase3_plan())
    planned = plan.plan.units[0]
    task = execution._resolve_task(planned)
    assert (
        task.task_id,
        task.family_id,
        task.task_index,
        task.environment_reset_seed,
    ) == (
        planned.unit.key.task_id,
        planned.unit.key.family_id,
        planned.unit.key.task_index,
        planned.unit.seeds.environment_seed,
    )
    assert isinstance(task.generator_seed, int)
