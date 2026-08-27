from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from levelup.experiments import milestone6_phase3_outcome_diagnostic_execution as execution
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import OutcomePlannedUnit


def _unit() -> OutcomePlannedUnit:
    # The executor's canonical resolver is patched in these tests; the payload
    # itself remains a real typed unit so generation receives no loose identity.
    return OutcomePlannedUnit(
        unit_id="a" * 64,
        condition_id="RP-resource-pressure-transition-listwise-optimum",
        tuple_id="lr0p003-e120-t0p9",
        training_tuple_id="lr0p003-e120",
        fold_id="lofo-plain",
        heldout_family="plain",
        task_id="micro.adaptive_track.plain.s900.i1.v1",
        task_index=0,
        replicate=0,
        model_owner_id="b" * 64,
        view_id="c" * 64,
        model_seed=1,
        environment_seed=0,
        probe_seed=2,
        search_seed=3,
        data_order_seed=4,
        exposure_manifest_sha256="d" * 64,
        feature_mask_sha256="e" * 64,
        transformation_sha256="f" * 64,
        model_identity_sha256="0" * 64,
        candidate_episodes_per_task=150,
        adaptation_actions_per_task=2048,
        probe_actions_per_task=64,
        maximum_actions_per_candidate_episode=64,
    )


class _FakeModel:
    authorized_model = object()
    training_accounting = SimpleNamespace(
        optimizer_steps=120,
        forward_passes=1200,
        training_examples=10,
        serialization_calls=1,
    )

    def require_active(self):
        return self


def _context() -> object:
    # Runtime tests focus on the sequencing boundary.  Canonical snapshot/cache
    # construction is covered by execution_models tests.
    return SimpleNamespace(snapshot=object(), plan=object(), protocol=object())


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, *, success: bool = True):
    unit = _unit()
    order: list[str] = []
    monkeypatch.setattr(execution, "_resolve_unit", lambda _context, value: value)
    monkeypatch.setattr(
        execution,
        "_resolve_task",
        lambda _unit: SimpleNamespace(
            task_id=_unit.task_id,
            family_id=_unit.heldout_family,
        ),
    )
    monkeypatch.setattr(execution, "_environment", lambda _task: object())
    monkeypatch.setattr(execution, "_forbidden_aliases", lambda _environment: frozenset())
    monkeypatch.setattr(
        execution,
        "AuthorizedOutcomeExecutionModel",
        _FakeModel,
    )

    @contextmanager
    def load_model(_snapshot, _planned):
        order.append("model_open")
        try:
            yield _FakeModel()
        finally:
            order.append("model_closed")

    monkeypatch.setattr(execution, "load_authorized_outcome_model_from_pinned_store", load_model)
    probe = SimpleNamespace(
        affordances=object(),
        accounting=SimpleNamespace(
            attempts=4,
            actions=64,
            resets=4,
            wall_seconds=0.0,
        ),
    )
    monkeypatch.setattr(execution, "discover_affordances", lambda *_a, **_k: probe)
    monkeypatch.setattr(
        execution,
        "authorize_outcome_probe_context",
        lambda *_a, **_k: object(),
    )
    generated = SimpleNamespace(
        candidates=(SimpleNamespace(episode=1, adaptation_actions=65),),
        candidate_generation_sha256="1" * 64,
        accounting=SimpleNamespace(
            planned_episode_cap=150,
            prior_probe_actions=64,
            episodes=1,
            actions=1,
            forward_passes=2,
            unknown_affordance_decisions=1,
        ),
    )

    def generate(**kwargs):
        order.append("generation")
        assert kwargs["model"] is _FakeModel.authorized_model
        assert kwargs["planned_unit"] == unit
        return generated

    monkeypatch.setattr(execution, "generate_outcome_group_candidates_with_observable_policy", generate)
    replay = SimpleNamespace(
        evaluated_candidates=(object(),) if success else (),
        best_performance=3.0 if success else None,
        first_valid_episode=1 if success else None,
        accounting=SimpleNamespace(
            episodes=1,
            actions=1,
            evaluator_calls=1,
            evaluator_replay_actions=2,
            evaluator_wall_seconds=0.0,
            generation_wall_seconds=0.0,
            resets=1,
            forward_passes=2,
        ),
    )

    def evaluate(*_args, **_kwargs):
        order.append("replay")
        return replay

    monkeypatch.setattr(execution, "evaluate_generated_search", evaluate)
    monkeypatch.setattr(
        execution,
        "classify_exact_optimum",
        lambda *_a, **_k: SimpleNamespace(
            success=success,
            first_episode=1 if success else None,
            first_adaptation_actions=65 if success else None,
        ),
    )

    def optimum(*_args):
        order.append("oracle")
        return 3.0

    monkeypatch.setattr(execution, "_default_optimum_provider", optimum)
    return unit, order, generated


def test_success_replays_before_reporting_oracle_and_counts_no_training(monkeypatch):
    unit, order, generated = _patch_runtime(monkeypatch, success=True)
    events: list[str] = []
    payload = execution.execute_outcome_diagnostic_unit(_context(), unit, event=events.append)
    assert order == ["model_open", "generation", "model_closed", "replay", "oracle"]
    assert events == ["generation_complete", "candidate_evaluation_complete", "optimum_oracle"]
    assert payload.outcome.success is True
    assert payload.outcome.first_optimum_adaptation_actions == 65
    assert payload.accounting.training.calls == 0
    assert payload.shared_artifact is None
    assert payload.shared_artifacts == ()
    assert payload.candidate_generation_sha256 == generated.candidate_generation_sha256
    assert payload.diagnostics["model_optimizer_steps"] == 120
    assert payload.diagnostics["model_serialization_calls"] == 1


def test_failed_unit_has_typed_fixed_endpoint_and_no_shared_reference(monkeypatch):
    unit, _order, _generated = _patch_runtime(monkeypatch, success=False)
    payload = execution.execute_outcome_diagnostic_unit(_context(), unit)
    assert payload.outcome.success is False
    assert payload.outcome.censored is True
    assert payload.outcome.censoring_budget == 2048
    assert payload.outcome.censoring_reason == "fixed_endpoint"
    assert payload.outcome.first_optimum_episode is None
    assert payload.shared_artifact is None


def test_oracle_is_unreachable_when_generation_fails(monkeypatch):
    unit, order, _generated = _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        execution,
        "generate_outcome_group_candidates_with_observable_policy",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("generation failed")),
    )
    with pytest.raises(RuntimeError, match="generation failed"):
        execution.execute_outcome_diagnostic_unit(_context(), unit)
    assert order == ["model_open", "model_closed"]


def test_oracle_is_unreachable_when_replay_fails(monkeypatch):
    unit, order, _generated = _patch_runtime(monkeypatch)
    def failing_replay(*_args, **_kwargs):
        order.append("replay")
        raise RuntimeError("replay failed")

    monkeypatch.setattr(execution, "evaluate_generated_search", failing_replay)
    with pytest.raises(RuntimeError, match="replay failed"):
        execution.execute_outcome_diagnostic_unit(_context(), unit)
    assert order == ["model_open", "generation", "model_closed", "replay"]


def test_context_requires_canonical_construction_and_unit_budgets_fail_closed() -> None:
    with pytest.raises(execution.OutcomeDiagnosticExecutionError, match="canonical readiness"):
        execution.OutcomeDiagnosticExecutionContext(
            object(), object(), object(), object(), object()  # type: ignore[arg-type]
        )

    class Cache:
        def require_active(self) -> None:
            return None

        def resolve_unit(self, unit):
            return unit

    context = execution.OutcomeDiagnosticExecutionContext(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        Cache(),  # type: ignore[arg-type]
        _token=execution._CONTEXT_TOKEN,
    )
    unit = _unit()
    for forged in (
        replace(unit, final_family_access=True),
        replace(unit, environment_seed=1),
        replace(unit, candidate_episodes_per_task=149),
        replace(unit, adaptation_actions_per_task=2_047),
        replace(unit, probe_actions_per_task=63),
        replace(unit, maximum_actions_per_candidate_episode=63),
    ):
        with pytest.raises(execution.OutcomeDiagnosticExecutionError, match="seed|budget"):
            execution._resolve_unit(context, forged)
