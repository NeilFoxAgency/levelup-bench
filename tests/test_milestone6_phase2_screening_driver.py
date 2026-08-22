"""Tests for the development-only screening execution boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase2_screening_driver as driver


def _canonical_paths(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    return repository / driver.CANONICAL_READINESS_PATH, repository


def _runtime(*, bad_phase: str | None = None, final_tasks: tuple[object, ...] = ()):
    units = tuple(
        SimpleNamespace(
            key=SimpleNamespace(phase=bad_phase if index == 0 and bad_phase else "validation")
        )
        for index in range(driver.EXPECTED_UNITS_PER_FOLD)
    )
    folds = tuple(
        SimpleNamespace(
            family_id=family,
            config=SimpleNamespace(
                split=SimpleNamespace(final_tasks=final_tasks),
                conditions=(),
            ),
            store=SimpleNamespace(
                expected=SimpleNamespace(units=units),
                _execution_ready=False,
                missing_units=lambda: (),
            ),
        )
        for family in driver.CANONICAL_FAMILY_ORDER
    )
    manifest = SimpleNamespace(family_order=driver.CANONICAL_FAMILY_ORDER)
    calls = SimpleNamespace(recheck=0)

    def recheck() -> None:
        calls.recheck += 1
        for fold in folds:
            fold.store._execution_ready = True

    return SimpleNamespace(manifest=manifest, folds=folds, recheck_before_execution=recheck), calls


def test_validate_only_loads_without_recheck_or_execution(monkeypatch, tmp_path):
    runtime, calls = _runtime()
    loaded: list[tuple[object, ...]] = []

    def load(*args, **kwargs):
        loaded.append((args, kwargs))
        return runtime

    monkeypatch.setattr(driver, "load_screening_runtime", load)

    class UnexpectedRunner:
        def __init__(self, _store):
            raise AssertionError("validate-only must execute zero units")

    monkeypatch.setattr(driver, "ExperimentRunner", UnexpectedRunner)
    manifest_path, repository = _canonical_paths(tmp_path)
    result = driver.run_development_screening(
        manifest_path,
        "a" * 64,
        "raw-root",
        repository,
        dry_run=True,
    )
    assert calls.recheck == 0
    assert result["dry_run"] is True
    assert result["total"]["completed"] == 0
    assert loaded[0][1]["manifest_bytes_sha256"] == "a" * 64


def test_execute_rechecks_once_and_uses_one_cache_per_fold(monkeypatch, tmp_path):
    runtime, calls = _runtime()
    monkeypatch.setattr(driver, "load_screening_runtime", lambda *args, **kwargs: runtime)
    cache_ids: list[int] = []
    caches: list[object] = []
    executed: list[str] = []

    def fake_unit(fold, _planned, *, model_cache):
        caches.append(model_cache)
        cache_ids.append(id(model_cache))
        executed.append(fold.family_id)
        return object()

    monkeypatch.setattr(driver, "execute_screening_unit", fake_unit)

    class FakeRunner:
        def __init__(self, store):
            self.store = store

        def execute(self, executor, **kwargs):
            assert kwargs == {
                "resume": True,
                "retry_failed": True,
                "fail_fast": True,
                "phases": ("validation",),
                "allow_final": False,
            }
            executor(self.store.expected.units[0])
            return {
                "completed": 1,
                "skipped": 1519,
                "unselected": 0,
                "failed": 0,
                "interrupted": 0,
            }

    monkeypatch.setattr(driver, "ExperimentRunner", FakeRunner)
    manifest_path, repository = _canonical_paths(tmp_path)
    result = driver.run_development_screening(
        manifest_path, "b" * 64, "raw", repository
    )

    assert calls.recheck == 1
    assert executed == list(driver.CANONICAL_FAMILY_ORDER)
    assert len(cache_ids) == 6
    assert len(set(cache_ids)) == 6
    assert result["total"] == {
        "completed": 6,
        "skipped": 9_114,
        "unselected": 0,
        "failed": 0,
        "interrupted": 0,
    }


def test_execute_fails_closed_if_runner_counts_leave_units_missing(monkeypatch, tmp_path):
    runtime, calls = _runtime()
    runtime.folds[0].store.missing_units = lambda: (runtime.folds[0].store.expected.units[0],)
    monkeypatch.setattr(driver, "load_screening_runtime", lambda *args, **kwargs: runtime)

    class FakeRunner:
        def __init__(self, _store):
            pass

        def execute(self, _executor, **_kwargs):
            return {
                "completed": 0,
                "skipped": driver.EXPECTED_UNITS_PER_FOLD,
                "unselected": 0,
                "failed": 0,
                "interrupted": 0,
            }

    monkeypatch.setattr(driver, "ExperimentRunner", FakeRunner)
    monkeypatch.setattr(driver, "execute_screening_unit", lambda *args, **kwargs: object())
    manifest_path, repository = _canonical_paths(tmp_path)
    with pytest.raises(RuntimeError, match="still has missing units"):
        driver.run_development_screening(
            manifest_path, "e" * 64, "raw", repository
        )
    assert calls.recheck == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bad_phase": "final"}, "validation units only"),
        ({"final_tasks": (object(),)}, "final tasks"),
    ],
)
def test_matrix_validation_fails_closed_before_recheck(
    monkeypatch, tmp_path, kwargs, message
):
    runtime, calls = _runtime(**kwargs)
    monkeypatch.setattr(driver, "load_screening_runtime", lambda *args, **kw: runtime)
    manifest_path, repository = _canonical_paths(tmp_path)
    with pytest.raises(RuntimeError, match=message):
        driver.run_development_screening(
            manifest_path, "c" * 64, "raw", repository
        )
    assert calls.recheck == 0


def test_driver_rejects_noncanonical_manifest_before_loading(monkeypatch, tmp_path):
    loaded = False

    def load(*_args, **_kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("noncanonical manifest must fail before runtime loading")

    monkeypatch.setattr(driver, "load_screening_runtime", load)
    repository = tmp_path / "repository"
    with pytest.raises(RuntimeError, match="canonical committed readiness manifest"):
        driver.run_development_screening(
            tmp_path / "copied-readiness.json",
            "f" * 64,
            "raw",
            repository,
            dry_run=True,
        )
    assert loaded is False


def test_cli_requires_explicit_mode(monkeypatch, capsys):
    called: list[bool] = []
    monkeypatch.setattr(
        driver,
        "run_development_screening",
        lambda *args, **kwargs: called.append(kwargs["dry_run"])
        or {"dry_run": kwargs["dry_run"], "total": {"completed": 0}},
    )
    assert (
        driver.main(
            [
                "--manifest-path",
                "manifest",
                "--manifest-sha256",
                "d" * 64,
                "--raw-root",
                "raw",
                "--repository",
                "repo",
                "--validate-only",
            ]
        )
        == 0
    )
    assert called == [True]
    assert '"dry_run": true' in capsys.readouterr().out
