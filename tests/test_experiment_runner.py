from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from levelup.experiments.phase1_smoke import smoke_executor
from levelup.experiments.runner import (
    ExperimentConfig,
    ExperimentRunner,
    PhaseAccounting,
    ResourceAccounting,
    RunStore,
    UnitOutcome,
    UnitPayload,
    aggregate_run,
    load_experiment_config,
    run_id_for,
    scientific_config_sha256,
)
from levelup.experiments.runner.aggregate import IncompleteRunError
from levelup.experiments.runner.config import DevicePolicy, canonical_json_bytes
from levelup.experiments.runner.provenance import capture_system_provenance
from levelup.experiments.runner.records import (
    PlannedUnit,
    SystemProvenance,
    UnitRecord,
)
from levelup.experiments.runner.storage import (
    ArtifactValidationError,
    _atomic_write_json,
    plan_expected_units,
)


@pytest.fixture(autouse=True)
def _controlled_runner_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "levelup.experiments.runner.storage.apply_runtime_policy",
        lambda policy: policy.requested_device,
    )
    monkeypatch.setattr(
        "levelup.experiments.runner.storage.capture_system_provenance",
        lambda repository, policy: _provenance(),
    )


def _config(*, conditions: int = 2, tasks: int = 2, replicates: int = 2) -> ExperimentConfig:
    raw = {
        "schema_version": "phase1.config.v1",
        "experiment_id": "runner contract test",
        "method_revision": "test-v1",
        "split": {
            "development_tasks": [
                {
                    "family_id": f"family-{index}",
                    "task_id": f"task-{index}",
                    "task_index": index,
                    "generator_seed": 100 + index,
                    "trajectory_catalog": [
                        {
                            "stage_label": "reference",
                            "trajectory_id": f"task-{index}.reference",
                            "source": "synthetic-reference",
                            "provenance": {"fixture": True},
                        }
                    ],
                }
                for index in range(tasks)
            ],
            "validation_tasks": [],
            "final_tasks": [],
        },
        "conditions": [
            {
                "condition_id": f"condition-{index}",
                "learner_id": "test-learner",
                "exposure": {
                    "train_task_ids": [f"task-{task_index}" for task_index in range(tasks)],
                    "exposed_trajectories": [
                        {
                            "task_id": f"task-{task_index}",
                            "stage_label": "reference",
                            "trajectory_id": f"task-{task_index}.reference",
                        }
                        for task_index in range(tasks)
                    ],
                    "observable_state_access": "current",
                    "action_history_access": False,
                    "action_descriptors_access": False,
                    "probe_interaction_access": True,
                    "search_feedback_access": True,
                    "evaluator_output_access": False,
                    "optimum_threshold_access": False,
                    "privileged_state_access": False,
                    "structured_constraint_access": True,
                    "metadata": {"tier": index},
                },
                "parameters": {"alpha": index + 1},
            }
            for index in range(conditions)
        ],
        "replicates": replicates,
        "seed_policy": {
            "model_seed_base": 10,
            "probe_seed_base": 20,
            "search_seed_base": 30,
            "data_order_seed_base": 40,
        },
        "device_policy": {
            "requested_device": "cpu",
            "torch_threads": 1,
        },
        "metrics": [
            {
                "metric_id": "performance",
                "direction": "minimize",
                "unit": "ticks",
                "description": "Independent test performance.",
            }
        ],
        "selection": {
            "phases": ["development"],
            "primary_metric": "performance",
            "rule": "Lowest paired mean.",
        },
        "diagnostic_fields": ["test"],
    }
    return ExperimentConfig.model_validate(raw)


def _provenance() -> SystemProvenance:
    return SystemProvenance(
        git_commit_sha="a" * 40,
        git_dirty=False,
        python_version="3.11-test",
        packages={
            "levelup-bench": "test",
            "numpy": "test",
            "pydantic": "test",
            "torch": "test",
        },
        installed_packages_sha256="c" * 64,
        os="test-os",
        architecture="test-arch",
        cpu="test-cpu",
        cpu_count=1,
        memory_bytes=1024,
        requested_device="cpu",
        resolved_device="cpu",
        requested_torch_threads=1,
        actual_torch_threads=1,
        requested_torch_interop_threads=1,
        actual_torch_interop_threads=1,
        deterministic_algorithms_requested=False,
        deterministic_algorithms_actual=False,
        processes=1,
        captured_at_utc="2026-01-01T00:00:00+00:00",
    )


def _payload(planned: PlannedUnit) -> UnitPayload:
    value = float(planned.key.task_index + planned.key.replicate)
    return UnitPayload(
        outcome=UnitOutcome(
            evaluator_ran=True,
            valid=True,
            completed=True,
            success=True,
            performance_metric_id="performance",
            performance_value=value,
            performance_direction="minimize",
            first_valid_completion_episode=1,
        ),
        accounting=ResourceAccounting(
            probes=PhaseAccounting(actions=2),
            search=PhaseAccounting(actions=3, forward_passes=4),
            replay=PhaseAccounting(actions=5),
            training=PhaseAccounting(optimizer_steps=6),
        ),
        diagnostics={"test": True},
    )


def _store(tmp_path: Path, config: ExperimentConfig | None = None) -> RunStore:
    store = RunStore(tmp_path, config or _config(), repository=tmp_path)
    store.initialize()
    return store


def _record(store: RunStore, planned: PlannedUnit) -> UnitRecord:
    payload = _payload(planned)
    return UnitRecord(
        run_id=store.run_id,
        config_sha256=store.config_sha256,
        unit_id=planned.unit_id,
        key=planned.key,
        seeds=planned.seeds,
        exposure_manifest_sha256=planned.exposure_manifest_sha256,
        started_at_utc="2026-01-01T00:00:00+00:00",
        finished_at_utc="2026-01-01T00:00:01+00:00",
        elapsed_wall_seconds=1.0,
        outcome=payload.outcome,
        accounting=payload.accounting,
        diagnostics=payload.diagnostics,
    )


def test_run_identity_is_canonical_but_changes_with_scientific_inputs(tmp_path: Path) -> None:
    config = _config()
    reordered = config.model_dump(mode="json")
    reordered["split"]["development_tasks"].reverse()
    reordered["conditions"].reverse()
    for condition in reordered["conditions"]:
        condition["exposure"]["train_task_ids"].reverse()
        condition["exposure"]["exposed_trajectories"].reverse()
    reordered_config = ExperimentConfig.model_validate(reordered)

    assert scientific_config_sha256(config) == scientific_config_sha256(reordered_config)
    assert run_id_for(config) == run_id_for(reordered_config)
    assert RunStore(tmp_path / "one", config, repository=tmp_path).run_id == RunStore(
        tmp_path / "two", config, repository=tmp_path
    ).run_id

    changed = config.model_dump(mode="json")
    changed["conditions"][0]["parameters"]["alpha"] = 999
    assert scientific_config_sha256(config) != scientific_config_sha256(
        ExperimentConfig.model_validate(changed)
    )


def test_config_rejects_final_selection_overlap_and_empty_exposure() -> None:
    raw = _config().model_dump(mode="json")
    raw["selection"]["phases"] = ["final"]
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)

    raw = _config().model_dump(mode="json")
    raw["split"]["final_tasks"] = [raw["split"]["development_tasks"][0]]
    with pytest.raises(ValidationError, match="overlap"):
        ExperimentConfig.model_validate(raw)

    raw = _config().model_dump(mode="json")
    raw["conditions"][0]["exposure"] = {}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)


def test_validation_selection_requires_validation_tasks() -> None:
    raw = _config().model_dump(mode="json")
    raw["selection"]["phases"] = ["development", "validation"]
    with pytest.raises(ValidationError, match="validation tasks"):
        ExperimentConfig.model_validate(raw)

    raw = _config().model_dump(mode="json")
    raw["device_policy"]["processes"] = 2
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)


def test_exposure_must_match_a_development_trajectory_catalog() -> None:
    raw = _config().model_dump(mode="json")
    raw["conditions"][0]["exposure"]["exposed_trajectories"][0][
        "trajectory_id"
    ] = "unknown-trajectory"
    with pytest.raises(ValidationError, match="development task catalog"):
        ExperimentConfig.model_validate(raw)

    raw = _config().model_dump(mode="json")
    exposed = raw["conditions"][0]["exposure"]["exposed_trajectories"][0]
    exposed["task_id"] = "task-1"
    with pytest.raises(ValidationError, match="development task catalog"):
        ExperimentConfig.model_validate(raw)

    raw = _config().model_dump(mode="json")
    final_task = {
        "family_id": "final-family",
        "task_id": "final-task",
        "task_index": 0,
        "generator_seed": 999,
        "trajectory_catalog": [],
    }
    raw["split"]["final_tasks"] = [final_task]
    exposure = raw["conditions"][0]["exposure"]
    exposure["train_task_ids"].append("final-task")
    exposure["exposed_trajectories"].append(
        {
            "task_id": "final-task",
            "stage_label": "optimum",
            "trajectory_id": "final-task.optimum",
        }
    )
    with pytest.raises(ValidationError, match="only on development tasks"):
        ExperimentConfig.model_validate(raw)


def test_expected_matrix_is_complete_deterministic_and_seed_paired() -> None:
    config = _config()
    first = plan_expected_units(config)
    second = plan_expected_units(config)

    assert first == second
    assert len(first.units) == 2 * 2 * 2
    groups: dict[tuple[str, int], set[str]] = {}
    for unit in first.units:
        group = (unit.key.task_id, unit.key.replicate)
        groups.setdefault(group, set()).add(unit.seeds.model_dump_json())
    assert all(len(seeds) == 1 for seeds in groups.values())
    assert all(len(unit.exposure_manifest_sha256) == 64 for unit in first.units)


def test_validity_requires_an_independent_evaluator() -> None:
    with pytest.raises(ValidationError, match="independent evaluator"):
        UnitOutcome(
            evaluator_ran=False,
            valid=True,
            completed=True,
            success=False,
            performance_metric_id="performance",
            performance_value=1.0,
            performance_direction="minimize",
        )


def test_initialize_snapshots_are_immutable_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    snapshots = {
        path.name: path.read_bytes()
        for path in store.run_dir.iterdir()
        if path.is_file()
    }

    monkeypatch.setattr(
        "levelup.experiments.runner.storage.capture_system_provenance",
        lambda repository, policy: _provenance().model_copy(
            update={"captured_at_utc": "2030-01-01T00:00:00+00:00"}
        ),
    )
    store.initialize()

    assert snapshots == {
        path.name: path.read_bytes()
        for path in store.run_dir.iterdir()
        if path.is_file()
    }

    changed = _provenance().model_copy(update={"git_commit_sha": "b" * 40})
    monkeypatch.setattr(
        "levelup.experiments.runner.storage.capture_system_provenance",
        lambda repository, policy: changed,
    )
    with pytest.raises(ArtifactValidationError, match="stored provenance"):
        store.initialize()


def test_atomic_write_cleans_temporary_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "artifact.json"

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr("levelup.experiments.runner.storage.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        _atomic_write_json(target, {"complete": True})

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_replace_failure_preserves_an_existing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "artifact.json"
    _atomic_write_json(target, {"version": 1})
    original = target.read_bytes()

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr("levelup.experiments.runner.storage.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        _atomic_write_json(target, {"version": 2})

    assert target.read_bytes() == original
    assert [path.name for path in tmp_path.iterdir()] == ["artifact.json"]


def test_corrupt_unexpected_and_mismatched_results_are_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    planned = store.expected.units[0]
    path = store.units_dir / f"{planned.unit_id}.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="invalid artifact"):
        store.load_completed(planned.unit_id)

    path.unlink()
    (store.units_dir / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="unexpected completed"):
        store.completed_records()

    (store.units_dir / "unexpected.json").unlink()
    record = _record(store, planned)
    mismatched = record.model_copy(
        update={"key": record.key.model_copy(update={"condition_id": "wrong-condition"})}
    )
    with pytest.raises(ArtifactValidationError, match="does not match"):
        store.write_completed(mismatched)

    undeclared_metric = record.model_copy(
        update={
            "outcome": record.outcome.model_copy(
                update={"performance_metric_id": "undeclared"}
            )
        }
    )
    with pytest.raises(ArtifactValidationError, match="undeclared performance metric"):
        store.write_completed(undeclared_metric)

    secret_diagnostic = record.model_copy(
        update={"diagnostics": {"test": "super-secret-token"}}
    )
    with pytest.raises(ArtifactValidationError) as error:
        store.write_completed(secret_diagnostic)
    assert "super-secret-token" not in str(error.value)
    assert error.value.__cause__ is None


def test_malformed_artifact_error_does_not_echo_secret_values(tmp_path: Path) -> None:
    store = _store(tmp_path)
    planned = store.expected.units[0]
    path = store.units_dir / f"{planned.unit_id}.json"
    path.write_text('{"schema_version":"super-secret-token"}', encoding="utf-8")

    with pytest.raises(ArtifactValidationError) as error:
        store.load_completed(planned.unit_id)

    assert "super-secret-token" not in str(error.value)
    assert error.value.__cause__ is None


def test_failure_is_sanitized_retained_and_retryable(tmp_path: Path) -> None:
    store = _store(tmp_path, _config(conditions=1, tasks=1, replicates=1))
    runner = ExperimentRunner(store)

    def fail_with_secret(planned: PlannedUnit) -> UnitPayload:
        raise RuntimeError("secret-token=/private/path")

    assert runner.execute(fail_with_secret, fail_fast=False) == {
        "completed": 0,
        "skipped": 0,
        "unselected": 0,
        "failed": 1,
        "interrupted": 0,
    }
    attempt = store.attempt_records()[0]
    assert attempt.status == "failed"
    assert attempt.sanitized_message == "executor raised RuntimeError"
    assert "secret" not in attempt.model_dump_json()

    assert runner.execute(_payload) == {
        "completed": 1,
        "skipped": 0,
        "unselected": 0,
        "failed": 0,
        "interrupted": 0,
    }
    assert len(store.attempt_records()) == 1
    assert len(store.completed_records()) == 1


@pytest.mark.parametrize(
    "bad_payload",
    [
        lambda planned: _payload(planned).model_copy(
            update={
                "outcome": _payload(planned).outcome.model_copy(
                    update={"performance_metric_id": "undeclared"}
                )
            }
        ),
        lambda planned: _payload(planned).model_copy(
            update={"diagnostics": {"test": float("nan")}}
        ),
    ],
)
def test_contract_failures_are_recorded_and_not_automatically_retried(
    tmp_path: Path,
    bad_payload: Callable[[PlannedUnit], UnitPayload],
) -> None:
    store = _store(tmp_path, _config(conditions=1, tasks=1, replicates=1))
    runner = ExperimentRunner(store)

    result = runner.execute(bad_payload, fail_fast=False)

    assert result["failed"] == 1
    assert not store.completed_records()
    attempt = store.attempt_records()[0]
    assert not attempt.retryable
    assert runner.execute(_payload)["skipped"] == 1


def test_keyboard_interrupt_is_recorded_then_reraised_and_retryable(tmp_path: Path) -> None:
    store = _store(tmp_path, _config(conditions=1, tasks=1, replicates=1))
    runner = ExperimentRunner(store)

    def interrupt(planned: PlannedUnit) -> UnitPayload:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        runner.execute(interrupt)
    attempt = store.attempt_records()[0]
    assert attempt.status == "interrupted"
    assert attempt.retryable

    assert runner.execute(_payload)["completed"] == 1


def test_resume_is_idempotent_and_does_not_rewrite_completed_units(tmp_path: Path) -> None:
    store = _store(tmp_path)
    runner = ExperimentRunner(store)
    first = runner.execute(_payload)
    unit_bytes = {path.name: path.read_bytes() for path in store.units_dir.glob("*.json")}

    second = runner.execute(_payload)

    assert first["completed"] == len(store.expected.units)
    assert second == {
        "completed": 0,
        "skipped": len(store.expected.units),
        "unselected": 0,
        "failed": 0,
        "interrupted": 0,
    }
    assert unit_bytes == {
        path.name: path.read_bytes() for path in store.units_dir.glob("*.json")
    }


def test_aggregation_is_strict_deterministic_and_read_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ExperimentRunner(store).execute(_payload)
    before = {
        str(path.relative_to(store.run_dir)): path.read_bytes()
        for path in store.run_dir.rglob("*")
        if path.is_file()
    }

    first = aggregate_run(store, strict=True, write=False)
    second = aggregate_run(store, strict=True, write=False)

    after = {
        str(path.relative_to(store.run_dir)): path.read_bytes()
        for path in store.run_dir.rglob("*")
        if path.is_file()
    }
    assert canonical_json_bytes(first.model_dump(mode="json")) == canonical_json_bytes(
        second.model_dump(mode="json")
    )
    assert before == after
    assert not store.aggregate_path.exists()
    assert first.complete
    assert first.paired_seed_audit_passed
    assert first.inventory.completed == first.inventory.expected

    wrong = first.model_copy(update={"run_id": "wrong-run"})
    with pytest.raises(ArtifactValidationError, match="validated raw records"):
        store.write_aggregate(wrong)
    assert store.write_aggregate(first)
    assert not store.write_aggregate(first)


def test_incomplete_aggregate_can_be_monotonically_finalized_after_resume(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, _config(conditions=1, tasks=2, replicates=1))
    first_unit = store.expected.units[0]
    store.write_completed(_record(store, first_unit))
    incomplete = aggregate_run(store, strict=False, write=True)
    assert not incomplete.complete
    assert incomplete.inventory.completed == 1

    ExperimentRunner(store).execute(_payload)
    complete = aggregate_run(store, strict=True, write=True)

    assert complete.complete
    assert complete.inventory.completed == 2
    assert store.aggregate_path.exists()


def test_aggregation_never_merges_development_and_validation_slices(
    tmp_path: Path,
) -> None:
    raw = _config(conditions=1, tasks=1, replicates=1).model_dump(mode="json")
    raw["split"]["validation_tasks"] = [
        {
            "family_id": "family-0",
            "task_id": "validation-task",
            "task_index": 1,
            "generator_seed": 999,
            "trajectory_catalog": [],
        }
    ]
    store = _store(tmp_path, ExperimentConfig.model_validate(raw))
    ExperimentRunner(store).execute(_payload)

    aggregate = aggregate_run(store, strict=True)

    assert aggregate.by_phase_condition["development"]["condition-0"].completed_units == 1
    assert aggregate.by_phase_condition["validation"]["condition-0"].completed_units == 1
    assert aggregate.by_phase_family["development"]["family-0"].completed_units == 1
    assert aggregate.by_phase_family["validation"]["family-0"].completed_units == 1


def test_strict_aggregation_rejects_missing_units(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.units_dir / ".stale-unit.tmp").write_text("partial", encoding="utf-8")
    with pytest.raises(IncompleteRunError, match="incomplete units"):
        aggregate_run(store, strict=True, write=False)
    assert len(store.missing_units()) == len(store.expected.units)


def test_attempt_filename_must_match_validated_identity(tmp_path: Path) -> None:
    store = _store(tmp_path, _config(conditions=1, tasks=1, replicates=1))

    def fail(planned: PlannedUnit) -> UnitPayload:
        raise RuntimeError("failure")

    ExperimentRunner(store).execute(fail, fail_fast=False)
    path = next(store.attempts_dir.glob("*.json"))
    path.rename(store.attempts_dir / "renamed.json")
    with pytest.raises(ArtifactValidationError, match="identity mismatch"):
        store.attempt_records()


def test_final_units_require_an_explicit_execution_boundary(tmp_path: Path) -> None:
    raw = _config(conditions=1, tasks=1, replicates=1).model_dump(mode="json")
    raw["split"]["final_tasks"] = [
        {
            "family_id": "final-family",
            "task_id": "final-task",
            "task_index": 0,
            "generator_seed": 999,
            "trajectory_catalog": [],
        }
    ]
    store = _store(tmp_path, ExperimentConfig.model_validate(raw))
    seen: list[str] = []

    def observe(planned: PlannedUnit) -> UnitPayload:
        seen.append(planned.key.phase)
        return _payload(planned)

    development = ExperimentRunner(store).execute(observe)
    assert seen == ["development"]
    assert development["unselected"] == 1

    with pytest.raises(ValueError, match="allow_final"):
        ExperimentRunner(store).execute(observe, phases=("final",))
    assert seen == ["development"]

    final = ExperimentRunner(store).execute(
        observe,
        phases=("final",),
        allow_final=True,
    )
    assert seen == ["development", "final"]
    assert final["completed"] == 1


def test_read_only_initialization_cannot_execute_units(tmp_path: Path) -> None:
    store = RunStore(
        tmp_path,
        _config(conditions=1, tasks=1, replicates=1),
        repository=tmp_path,
    )
    store.initialize(for_execution=False)

    with pytest.raises(RuntimeError, match="for_execution=True"):
        ExperimentRunner(store).execute(_payload)


def test_dirty_provenance_hash_includes_untracked_contents(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.email", "test@example.invalid"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Runner Test"), cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=tmp_path, check=True)
    untracked = tmp_path / "new-code.py"
    untracked.write_text("value = 1\n", encoding="utf-8")
    policy = DevicePolicy(requested_device="cpu", torch_threads=1)

    first = capture_system_provenance(tmp_path, policy)
    untracked.write_text("value = 2\n", encoding="utf-8")
    second = capture_system_provenance(tmp_path, policy)

    assert first.git_dirty and second.git_dirty
    assert first.git_diff_sha256 != second.git_diff_sha256
    assert "new-code.py" not in first.model_dump_json()


def test_committed_phase1_smoke_config_executes_only_development_data(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    config = load_experiment_config(repository / "configs/milestone6/phase1_smoke.json")
    store = RunStore(tmp_path, config, repository=repository)
    store.initialize()

    execution = ExperimentRunner(store).execute(
        lambda planned: smoke_executor(config, planned)
    )
    aggregate = aggregate_run(store, strict=True)

    assert not config.split.validation_tasks
    assert not config.split.final_tasks
    assert execution["completed"] == 8
    assert execution["unselected"] == 0
    assert aggregate.complete
    assert aggregate.inventory.expected == 8
    assert all(record.key.phase == "development" for record in store.completed_records())
