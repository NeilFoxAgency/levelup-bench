from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase3_execution_driver as driver
from levelup.experiments.milestone6_phase3_plan import Phase3PlannedUnit
from levelup.experiments.runner.records import (
    PhaseAccounting,
    PlannedUnit,
    ResourceAccounting,
    UnitKey,
    UnitOutcome,
    UnitPayload,
    UnitSeeds,
)


def _planned(family: str, index: int) -> Phase3PlannedUnit:
    key = UnitKey(
        phase="validation",
        condition_id="S-state-availability-listwise-optimum--lr0p003-e120",
        family_id=family,
        task_id=f"task-{family}-{index}",
        task_index=index,
        replicate=0,
    )
    unit = PlannedUnit(
        unit_id=f"{index + 1:064x}",
        key=key,
        seeds=UnitSeeds(
            model_seed=1,
            environment_seed=0,
            probe_seed=2,
            search_seed=3,
            data_order_seed=4,
        ),
        exposure_manifest_sha256="a" * 64,
    )
    return Phase3PlannedUnit(
        unit=unit,
        base_condition_id="S-state-availability-listwise-optimum",
        tuple_id="S-state-availability-listwise-optimum--lr0p003-e120",
        training_tuple_id="lr0p003-e120",
        fold_id=f"fold-{family}",
        heldout_family=family,
        model_owner_id="b" * 64,
        view_id="c" * 64,
    )


def _payload() -> UnitPayload:
    return UnitPayload(
        outcome=UnitOutcome(
            evaluator_ran=True,
            valid=True,
            completed=True,
            success=False,
            performance_metric_id="performance_value",
            performance_value=1.0,
            performance_direction="minimize",
            censored=True,
            censoring_budget=2048,
            censoring_reason="fixed_endpoint",
        ),
        accounting=ResourceAccounting(
            probes=PhaseAccounting(actions=64, environment_steps=64),
            search=PhaseAccounting(actions=1, environment_steps=1, episodes=1),
            replay=PhaseAccounting(actions=1, environment_steps=1),
        ),
        candidate_generation_sha256="d" * 64,
    )


class _Family:
    def __init__(self, family_id: str) -> None:
        self.family_id = family_id
        self.run_id = f"run-{family_id}"
        self.config_sha256 = "e" * 64
        self.completed: dict[str, object] = {}
        self.attempts: list[object] = []

    def load_completed(self, unit_id: str):
        return self.completed.get(unit_id)

    def completed_unit_ids(self) -> tuple[str, ...]:
        return tuple(self.completed)

    def write_completed(self, record) -> bool:
        if record.unit_id in self.completed:
            return False
        self.completed[record.unit_id] = record
        return True

    def attempt_records(self):
        return tuple(self.attempts)

    def next_attempt_number(self, unit_id: str) -> int:
        return 1 + sum(item.unit_id == unit_id for item in self.attempts)

    def write_attempt(self, record) -> bool:
        self.attempts.append(record)
        return True


def _expected(planned: tuple[Phase3PlannedUnit, ...]):
    stores = tuple(
        SimpleNamespace(
            family_id=family,
            units=tuple(item for item in planned if item.heldout_family == family),
        )
        for family in driver.FAMILIES
    )
    return SimpleNamespace(
        plan_id="f" * 64,
        protocol_sha256="1" * 64,
        model_authority_sha256="2" * 64,
        final_family_access=False,
        family_order=driver.FAMILIES,
        stores=stores,
        units=planned,
    )


def _patch_execution_context(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, object, Path]]:
    calls: list[tuple[object, object, Path]] = []

    class Context:
        @classmethod
        def canonical(cls, authority, plan, model_root):
            calls.append((authority, plan, model_root))
            return object()

    monkeypatch.setattr(driver, "Phase3ExecutionContext", Context)
    return calls


def test_retryability_and_attempt_message_are_bounded() -> None:
    family = _Family("plain")
    planned = _planned("plain", 0)
    exc = ValueError("secret path and a long traceback should not be persisted")
    driver._attempt(
        family,
        planned,
        exc,
        attempt_number=1,
        stage="payload-validation",
        retryable=False,
        started_at=driver.utc_now(),
        elapsed=0.1,
    )
    attempt = family.attempts[0]
    assert attempt.exception_type == "ValueError"
    assert attempt.sanitized_message == "payload-validation raised ValueError"
    assert "secret" not in attempt.sanitized_message
    assert driver._retryable(exc) is False
    assert driver._retryable(RuntimeError("transient")) is True


def test_execute_loop_is_canonical_and_resumable(monkeypatch: pytest.MonkeyPatch) -> None:
    planned = tuple(_planned(family, index) for index, family in enumerate(driver.FAMILIES))
    families = tuple(_Family(family) for family in driver.FAMILIES)
    batch = SimpleNamespace(stores=families)
    expected = _expected(planned)
    context_calls = _patch_execution_context(monkeypatch)
    monkeypatch.setattr(driver, "execute_phase3_unit", lambda *_args, **_kwargs: _payload())

    first = driver._execute_loop(
        batch,
        object(),
        object(),
        expected,
        Path("/tmp/phase3-models"),
    )
    assert first == {
        "completed": 6,
        "skipped": 0,
        "failed": 0,
        "interrupted": 0,
        "complete": True,
    }

    resumed = driver._execute_loop(
        batch,
        object(),
        object(),
        expected,
        Path("/tmp/phase3-models"),
    )
    assert resumed == {
        "completed": 0,
        "skipped": 6,
        "failed": 0,
        "interrupted": 0,
        "complete": True,
    }
    assert len(context_calls) == 2


def test_nonretryable_attempt_cannot_be_treated_as_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned = tuple(_planned(family, index) for index, family in enumerate(driver.FAMILIES))
    families = tuple(_Family(family) for family in driver.FAMILIES)
    attempt = SimpleNamespace(unit_id=planned[0].unit.unit_id, attempt=1, retryable=False)
    families[0].attempts.append(attempt)
    _patch_execution_context(monkeypatch)
    monkeypatch.setattr(driver, "execute_phase3_unit", lambda *_args, **_kwargs: _payload())
    with pytest.raises(driver.Phase3ExecutionDriverError, match="non-retryable"):
        driver._execute_loop(
            SimpleNamespace(stores=families),
            object(),
            object(),
            _expected(planned),
            Path("/tmp/phase3-models"),
        )


def test_retry_attempt_number_comes_from_single_cached_attempt_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned = tuple(_planned(family, index) for index, family in enumerate(driver.FAMILIES))
    families = tuple(_Family(family) for family in driver.FAMILIES)
    families[0].attempts.append(
        SimpleNamespace(unit_id=planned[0].unit.unit_id, attempt=7, retryable=True)
    )

    def forbidden_attempt_scan(_unit_id: str) -> int:
        raise AssertionError("production driver must not rescan attempt filenames")

    families[0].next_attempt_number = forbidden_attempt_scan
    _patch_execution_context(monkeypatch)

    def fail(*_args, **_kwargs):
        raise RuntimeError("transient")

    monkeypatch.setattr(driver, "execute_phase3_unit", fail)
    with pytest.raises(RuntimeError, match="transient"):
        driver._execute_loop(
            SimpleNamespace(stores=families),
            object(),
            object(),
            _expected(planned),
            Path("/tmp/phase3-models"),
        )
    assert families[0].attempts[-1].attempt == 8


def test_validate_only_preflight_never_enters_activation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "authority"
    repository.mkdir()
    model_root = repository / "runs" / "milestone6" / "artifact-store"
    model_root.mkdir(parents=True)
    result_root = tmp_path / "results"
    result_root.mkdir()
    monkeypatch.setattr(driver, "ROOT", repository)
    expected = _expected(())
    authority = SimpleNamespace(artifact_store_id="artifact-store")
    monkeypatch.setattr(driver, "_load_authorities", lambda _repo: (object(), authority, expected))
    monkeypatch.setattr(driver, "_load_prepared_stores", lambda *_args: ())
    monkeypatch.setattr(driver, "capture_phase3_readiness", lambda *_args, **_kwargs: object())
    activation_called = False

    @contextmanager
    def forbidden_activation(*_args, **_kwargs):
        nonlocal activation_called
        activation_called = True
        yield None

    monkeypatch.setattr(driver, "phase3_activation", forbidden_activation)
    result = driver.run_phase3_development(
        repository,
        result_root,
        expected_git_commit="a" * 40,
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert result["expected_total"] == driver.EXPECTED_TOTAL_UNITS
    assert activation_called is False


def test_prepared_store_loading_uses_noncreating_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = _expected(())
    calls: list[Path] = []
    stores = tuple(SimpleNamespace(family_id=family) for family in driver.FAMILIES)
    monkeypatch.setattr(driver, "_require_existing_store_tree", lambda *_args: False)

    def load(root, _validated, _authority):
        calls.append(root)
        return stores

    monkeypatch.setattr(driver, "load_phase3_result_stores", load)
    assert driver._load_prepared_stores(tmp_path, object(), object(), expected) == stores
    assert calls == [tmp_path]


def test_driver_does_not_expose_reducer_or_analysis_modules() -> None:
    names = set(vars(driver))
    assert not any("reducer" in name or "analysis" in name for name in names)
