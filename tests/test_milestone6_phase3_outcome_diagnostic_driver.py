"""Mechanical tests for the development-only outcome diagnostic driver."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from levelup.experiments import milestone6_phase3_outcome_diagnostic_driver as driver
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import OutcomePlannedUnit
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import CONDITIONS, FAMILIES
from levelup.experiments.runner.records import (
    PhaseAccounting,
    ResourceAccounting,
    UnitOutcome,
    UnitPayload,
)


def _planned(index: int, family: str, condition: str) -> OutcomePlannedUnit:
    return OutcomePlannedUnit(
        unit_id=f"{index + 1:064x}",
        condition_id=condition,
        tuple_id=f"tuple-{index}",
        training_tuple_id="lr0p003-e120",
        fold_id=f"fold-{family}",
        heldout_family=family,
        task_id=f"task-{index}",
        task_index=index % 8,
        replicate=index % 5,
        model_owner_id=f"{index + 1:064x}",
        view_id=f"{index + 1:064x}",
        model_seed=index,
        environment_seed=0,
        probe_seed=index + 1,
        search_seed=index + 2,
        data_order_seed=index + 3,
        exposure_manifest_sha256="a" * 64,
        feature_mask_sha256="b" * 64,
        transformation_sha256="c" * 64,
        model_identity_sha256="d" * 64,
        candidate_episodes_per_task=150,
        adaptation_actions_per_task=2048,
        probe_actions_per_task=64,
        maximum_actions_per_candidate_episode=64,
    )


def _matrix() -> tuple[OutcomePlannedUnit, ...]:
    values: list[OutcomePlannedUnit] = []
    for index in range(driver.EXPECTED_TOTAL_UNIT_COUNT):
        family = FAMILIES[index // 960]
        condition = CONDITIONS[(index // 480) % 2]
        values.append(_planned(index, family, condition))
    return tuple(values)


class _Family:
    def __init__(self, family_id: str, units: tuple[OutcomePlannedUnit, ...]) -> None:
        self.family_id = family_id
        self.run_id = f"run-{family_id}"
        self.config_sha256 = "e" * 64
        self.units = units
        self.completed: dict[str, object] = {}
        self.attempts: list[object] = []

    def completed_unit_ids(self) -> tuple[str, ...]:
        return tuple(self.completed)

    def attempt_records(self) -> tuple[object, ...]:
        return tuple(self.attempts)

    def write_completed(self, record: object) -> bool:
        unit_id = record.unit_id
        if unit_id in self.completed:
            return False
        self.completed[unit_id] = record
        return True

    def write_attempt(self, record: object) -> bool:
        self.attempts.append(record)
        return True


class _Expected:
    def __init__(self, units: tuple[OutcomePlannedUnit, ...]) -> None:
        self.units = units
        self.stores = tuple(
            SimpleNamespace(
                family_id=family,
                units=tuple(item for item in units if item.heldout_family == family),
            )
            for family in FAMILIES
        )

    def store_for_family(self, family_id: str) -> SimpleNamespace:
        return next(store for store in self.stores if store.family_id == family_id)


def _payload() -> UnitPayload:
    return UnitPayload(
        outcome=UnitOutcome(
            evaluator_ran=True,
            valid=False,
            completed=False,
            success=False,
            performance_metric_id="performance_value",
            performance_direction="minimize",
            censored=True,
            censoring_budget=2048,
            censoring_reason="fixed_endpoint",
        ),
        accounting=ResourceAccounting(
            probes=PhaseAccounting(actions=64, environment_steps=64),
            search=PhaseAccounting(actions=1, environment_steps=1, episodes=1),
        ),
    )


def _fixture(monkeypatch: pytest.MonkeyPatch):
    units = _matrix()
    families = tuple(
        _Family(family, tuple(item for item in units if item.heldout_family == family))
        for family in FAMILIES
    )
    monkeypatch.setattr(driver, "_enforce_cpu_single_thread", lambda: None)
    return units, families, _Expected(units), SimpleNamespace()


def test_all_complete_resume_executes_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    units, families, expected, context = _fixture(monkeypatch)
    for family in families:
        for unit in family.units:
            family.completed[unit.unit_id] = object()
    calls: list[str] = []
    monkeypatch.setattr(driver, "execute_outcome_diagnostic_unit", lambda _ctx, unit: calls.append(unit.unit_id))
    result = driver._execute_loop(SimpleNamespace(stores=families), context, expected)
    assert result == {
        "completed": 0,
        "skipped": driver.EXPECTED_TOTAL_UNIT_COUNT,
        "failed": 0,
        "interrupted": 0,
        "complete": True,
    }
    assert calls == []


def test_one_missing_unit_publishes_tuple_qualified_record(monkeypatch: pytest.MonkeyPatch) -> None:
    units, families, expected, context = _fixture(monkeypatch)
    missing = units[0]
    for family in families:
        for unit in family.units:
            if unit.unit_id != missing.unit_id:
                family.completed[unit.unit_id] = object()
    calls: list[str] = []

    def execute(_ctx: object, unit: OutcomePlannedUnit) -> UnitPayload:
        calls.append(unit.unit_id)
        return _payload()

    monkeypatch.setattr(driver, "execute_outcome_diagnostic_unit", execute)
    result = driver._execute_loop(SimpleNamespace(stores=families), context, expected)
    assert result["completed"] == 1
    assert result["skipped"] == driver.EXPECTED_TOTAL_UNIT_COUNT - 1
    assert result["complete"] is True
    assert calls == [missing.unit_id]
    record = families[0].completed[missing.unit_id]
    assert record.key.condition_id == f"{missing.condition_id}--{missing.tuple_id}"
    assert record.key.phase == "validation"
    assert record.seeds.environment_seed == missing.environment_seed


def test_nonretryable_attempt_refuses_reexecution(monkeypatch: pytest.MonkeyPatch) -> None:
    units, families, expected, context = _fixture(monkeypatch)
    for family in families:
        for unit in family.units:
            if unit.unit_id != units[0].unit_id:
                family.completed[unit.unit_id] = object()
    families[0].attempts.append(
        driver.AttemptRecord(
            run_id=families[0].run_id,
            config_sha256=families[0].config_sha256,
            unit_id=units[0].unit_id,
            attempt=1,
            key=driver._expected_key(units[0]),
            seeds=driver._expected_seeds(units[0]),
            status="failed",
            stage="payload-validation",
            exception_type="ValueError",
            sanitized_message="payload-validation raised ValueError",
            retryable=False,
            started_at_utc=driver.utc_now(),
            finished_at_utc=driver.utc_now(),
            elapsed_wall_seconds=0,
        )
    )
    monkeypatch.setattr(driver, "execute_outcome_diagnostic_unit", lambda *_args: pytest.fail("reexecuted"))
    with pytest.raises(driver.OutcomeDiagnosticDriverError, match="non-retryable"):
        driver._execute_loop(SimpleNamespace(stores=families), context, expected)


def test_failure_and_interrupt_publish_typed_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    units, families, expected, context = _fixture(monkeypatch)
    for family in families:
        for unit in family.units:
            if unit.unit_id != units[0].unit_id:
                family.completed[unit.unit_id] = object()

    monkeypatch.setattr(driver, "execute_outcome_diagnostic_unit", lambda *_args: (_ for _ in ()).throw(ValueError("secret")))
    with pytest.raises(ValueError):
        driver._execute_loop(SimpleNamespace(stores=families), context, expected)
    failed = families[0].attempts[-1]
    assert failed.status == "failed"
    assert failed.retryable is False
    assert failed.sanitized_message == "execution raised ValueError"
    assert "secret" not in failed.sanitized_message

    families[0].attempts.clear()
    monkeypatch.setattr(driver, "execute_outcome_diagnostic_unit", lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        driver._execute_loop(SimpleNamespace(stores=families), context, expected)
    interrupted = families[0].attempts[-1]
    assert interrupted.status == "interrupted"
    assert interrupted.retryable is True


def _public_driver_fixture(monkeypatch: pytest.MonkeyPatch):
    preflights: list[str] = []
    lease = object()

    class Snapshot:
        def preflight(self, *, expected_git_commit: str) -> None:
            preflights.append(expected_git_commit)

    snapshot = Snapshot()

    @contextmanager
    def hold_for_activation(*, expected_git_commit: str):
        preflights.append(f"hold:{expected_git_commit}")
        yield lease

    snapshot.base = SimpleNamespace(hold_for_activation=hold_for_activation)
    context = SimpleNamespace(
        plan=SimpleNamespace(plan=SimpleNamespace(plan_id="plan-id")),
        protocol=SimpleNamespace(sha256="f" * 64),
        authority=SimpleNamespace(
            authority_sha256="a" * 64,
            expected_authority_sha256="a" * 64,
            development_only=True,
            final=False,
            final_family_access=False,
        ),
    )
    expected = SimpleNamespace()
    monkeypatch.setattr(driver, "OutcomeDiagnosticModelReadinessSnapshot", Snapshot)
    monkeypatch.setattr(
        driver.OutcomeDiagnosticExecutionContext,
        "canonical",
        lambda observed: context if observed is snapshot else pytest.fail("wrong snapshot"),
    )
    monkeypatch.setattr(driver, "_validate_exact_matrix", lambda observed: expected)
    return snapshot, context, expected, lease, preflights


def test_public_validate_only_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot, _context, expected, lease, preflights = _public_driver_fixture(monkeypatch)
    validations: list[tuple[object, object, object, bool]] = []
    monkeypatch.setattr(
        driver,
        "_validate_stores_for_readiness",
        lambda observed, matrix, held, *, validate_only: validations.append(
            (observed, matrix, held, validate_only)
        ),
    )
    monkeypatch.setattr(
        driver,
        "activate_outcome_diagnostic_result_stores",
        lambda *_args, **_kwargs: pytest.fail("validate-only activated stores"),
    )
    monkeypatch.setattr(
        driver,
        "_execute_loop",
        lambda *_args, **_kwargs: pytest.fail("validate-only executed units"),
    )

    summary = driver.run_outcome_diagnostic_development(
        snapshot, expected_git_commit="1" * 40, validate_only=True
    )

    assert validations == [(snapshot, expected, lease, True)]
    assert preflights == ["1" * 40, f"hold:{'1' * 40}"]
    assert summary["validate_only"] is True
    assert summary["complete"] is False
    assert summary["expected_total"] == driver.EXPECTED_TOTAL_UNIT_COUNT


def test_public_execute_rechecks_before_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot, context, expected, lease, preflights = _public_driver_fixture(monkeypatch)
    stores = (object(),)
    batch = object()
    activations: list[tuple[object, object, object, str]] = []

    monkeypatch.setattr(
        driver,
        "_validate_stores_for_readiness",
        lambda observed, matrix, held, *, validate_only: stores
        if (observed, matrix, held, validate_only) == (snapshot, expected, lease, False)
        else pytest.fail("wrong execution store validation"),
    )

    @contextmanager
    def activate(observed_stores, matrix, held, *, expected_git_commit: str):
        activations.append((observed_stores, matrix, held, expected_git_commit))
        yield batch

    monkeypatch.setattr(driver, "activate_outcome_diagnostic_result_stores", activate)
    monkeypatch.setattr(
        driver,
        "_execute_loop",
        lambda observed_batch, observed_context, matrix: {
            "completed": driver.EXPECTED_TOTAL_UNIT_COUNT,
            "skipped": 0,
            "failed": 0,
            "interrupted": 0,
            "complete": True,
        }
        if (observed_batch, observed_context, matrix) == (batch, context, expected)
        else pytest.fail("wrong execution inputs"),
    )

    summary = driver.run_outcome_diagnostic_development(
        snapshot, expected_git_commit="2" * 40
    )

    assert preflights == ["2" * 40, f"hold:{'2' * 40}", "2" * 40]
    assert activations == [(stores, expected, lease, "2" * 40)]
    assert summary["complete"] is True
    assert summary["completed"] == driver.EXPECTED_TOTAL_UNIT_COUNT
