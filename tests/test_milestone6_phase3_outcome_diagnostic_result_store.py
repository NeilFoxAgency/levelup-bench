"""Adversarial tests for the inert outcome-diagnostic result-store boundary."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from levelup.experiments import milestone6_phase3_outcome_diagnostic_readiness as readiness
from levelup.experiments import milestone6_phase3_outcome_diagnostic_result_store as result_store
from levelup.experiments import milestone6_phase3_readiness as phase3_readiness
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    ValidatedOutcomePlan,
    bind_validated_outcome_diagnostic_plan,
    build_outcome_group_diagnostic_plan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    CONDITIONS,
    FAMILIES,
    load_outcome_group_diagnostic_protocol,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_result_store import (
    ACTIVATION_INTENT_NAME,
    EXPECTED_FAMILY_UNIT_COUNT,
    EXPECTED_NAMESPACE_UNIT_COUNT,
    EXPECTED_TOTAL_UNIT_COUNT,
    OutcomeDiagnosticResultStoreError,
    activate_outcome_diagnostic_result_stores,
    build_outcome_diagnostic_expected_plan,
    load_outcome_diagnostic_result_stores,
    prepare_outcome_diagnostic_result_stores,
)
from levelup.experiments.runner.records import (
    AttemptRecord,
    ResourceAccounting,
    UnitKey,
    UnitOutcome,
    UnitRecord,
    UnitSeeds,
)

REPOSITORY = Path(readiness.ROOT)
_MODEL_STORE_ID = "phase3-model-preparation-cc08207"
_MODEL_METADATA_FIXTURE = Path(__file__).parent / "fixtures" / "phase3_model_preparation_metadata"


@pytest.fixture(scope="module", autouse=True)
def _materialize_metadata_only_model_store():
    """Provide CI exact published metadata without fabricating checkpoints."""

    model_root = REPOSITORY / "runs" / "milestone6" / _MODEL_STORE_ID
    if model_root.exists():
        yield
        return
    candidate_parents = [model_root.parent.parent, model_root.parent]
    created_parents = [path for path in candidate_parents if not path.exists()]
    model_root.mkdir(parents=True)
    created_directories = [
        model_root / "phase3-model-artifact-keys",
        model_root / "phase3-model-artifact-costs",
        model_root / "phase3-model-artifacts",
    ]
    for directory in created_directories:
        directory.mkdir()
    expected_hashes = {
        phase3_readiness.PREPARATION_PROVENANCE_NAME: (
            "c1c302db1f88b62902628c839cd566ade6102bdb0716bcb505d09a5a49737679"
        ),
        phase3_readiness.PREPARATION_PROGRESS_NAME: (
            "e5ff3c385c6f32ca9e5dac04b4a81e229c0bfb073300ac4505edfd419ff7d11b"
        ),
    }
    created_files = []
    for name, expected in expected_hashes.items():
        source = _MODEL_METADATA_FIXTURE / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
        destination = model_root / name
        shutil.copyfile(source, destination)
        created_files.append(destination)
    try:
        yield
    finally:
        for path in reversed(created_files):
            path.unlink(missing_ok=True)
        for path in reversed(created_directories):
            path.rmdir()
        model_root.rmdir()
        for path in reversed(created_parents):
            path.rmdir()


def _empty_root(root: Path) -> None:
    for child in root.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


@pytest.fixture(scope="module")
def authorities(tmp_path_factory: pytest.TempPathFactory):
    root = REPOSITORY / "runs" / "milestone6" / "phase3-outcome-group-diagnostic-test"
    root.mkdir(parents=True, exist_ok=True)
    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        readiness,
        "DIAGNOSTIC_OUTPUT_ROOT_RELATIVE",
        root.relative_to(REPOSITORY).as_posix(),
    )
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=REPOSITORY, text=True
    ).strip()
    patcher.setattr(phase3_readiness, "_git_state", lambda _repo: (commit, False))
    protocol = load_outcome_group_diagnostic_protocol()
    plan = bind_validated_outcome_diagnostic_plan(
        build_outcome_group_diagnostic_plan(protocol), snapshot=protocol
    )
    snapshot = readiness.capture_outcome_group_diagnostic_readiness(
        repository=REPOSITORY, output_root=root, expected_git_commit=commit
    )
    try:
        yield snapshot, plan
    finally:
        shutil.rmtree(root, ignore_errors=True)
        patcher.undo()


def test_expected_matrix_is_exact_and_development_only(authorities) -> None:
    snapshot, plan = authorities
    expected = build_outcome_diagnostic_expected_plan(plan, snapshot.protocol)
    assert expected.family_order == FAMILIES
    assert expected.condition_order == CONDITIONS
    assert tuple(len(store.units) for store in expected.stores) == (EXPECTED_FAMILY_UNIT_COUNT,) * 6
    assert len(expected.units) == EXPECTED_TOTAL_UNIT_COUNT
    assert all(
        len(namespace.units) == EXPECTED_NAMESPACE_UNIT_COUNT
        for store in expected.stores
        for namespace in store.namespaces
    )
    assert all(store.final_family_access is False for store in expected.stores)


def test_preparation_is_idempotent_and_creates_only_metadata(authorities) -> None:
    snapshot, plan = authorities
    _empty_root(snapshot.output_root)
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        first = prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)
        second = prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)
        loaded = load_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)
    assert tuple(store.run_id for store in first) == tuple(store.run_id for store in second)
    assert tuple(store.run_id for store in loaded) == tuple(store.run_id for store in first)
    assert all(not store.execution_ready for store in loaded)
    assert all(
        (store.root / family).is_dir() for family, store in zip(FAMILIES, first, strict=True)
    )
    assert not list(
        (
            first[0].root / FAMILIES[0] / first[0].run_id / "namespaces" / CONDITIONS[0] / "records"
        ).iterdir()
    )
    assert (first[0].root / FAMILIES[0] / first[0].run_id / ACTIVATION_INTENT_NAME).is_file()
    with pytest.raises(TypeError):
        first[0].namespace_identities[CONDITIONS[0]] = (0, 0)  # type: ignore[index]
    with pytest.raises(TypeError):
        first[0].record_namespace_identities[CONDITIONS[0]] = (0, 0)  # type: ignore[index]


def test_path_only_and_forged_or_closed_lease_fail_closed(authorities) -> None:
    snapshot, plan = authorities
    _empty_root(snapshot.output_root)
    with pytest.raises(OutcomeDiagnosticResultStoreError):
        load_outcome_diagnostic_result_stores(None, snapshot.protocol, plan)  # type: ignore[arg-type]
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        lease.close()
        with pytest.raises(OutcomeDiagnosticResultStoreError, match="lease"):
            load_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)
    with pytest.raises(OutcomeDiagnosticResultStoreError, match="canonical readiness"):
        prepare_outcome_diagnostic_result_stores(object(), snapshot.protocol, plan)  # type: ignore[arg-type]


def test_protocol_snapshot_and_plan_identity_are_pinned(authorities) -> None:
    snapshot, plan = authorities
    _empty_root(snapshot.output_root)
    forged_protocol = replace(snapshot.protocol, sha256="0" * 64)
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        with pytest.raises(OutcomeDiagnosticResultStoreError):
            prepare_outcome_diagnostic_result_stores(lease, forged_protocol, plan)
        forged_plan = replace(plan.plan, final_family_access=True)
        forged_validated = ValidatedOutcomePlan(
            forged_plan,
            {item.unit_id: item for item in forged_plan.units},
            _construction_token=plan._construction_token,
        )
        with pytest.raises(OutcomeDiagnosticResultStoreError, match="development|authority"):
            prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, forged_validated)


@pytest.mark.parametrize("mutation", ["extra", "partial", "symlink", "records"])
def test_existing_foreign_or_partial_layout_fails_closed(authorities, mutation: str) -> None:
    snapshot, plan = authorities
    root = snapshot.output_root

    _empty_root(root)

    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        if mutation == "extra":
            (root / "foreign").mkdir()
        elif mutation == "partial":
            (root / FAMILIES[0]).mkdir()
        elif mutation == "symlink":
            target = root / "target"
            target.mkdir()
            (root / FAMILIES[0]).symlink_to(target, target_is_directory=True)
        else:
            prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)
            records = next(root.glob(f"{FAMILIES[0]}/*/namespaces/*/records"))
            (records / "foreign.json").write_text("{}")
        with pytest.raises(OutcomeDiagnosticResultStoreError):
            prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)
    _empty_root(root)


def test_descriptor_substitution_is_detected_on_reload(authorities) -> None:
    snapshot, plan = authorities
    _empty_root(snapshot.output_root)
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        stores = prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)
        family = stores[0]
        family_path = family.root / family.family_id
        moved = family.root / (family.family_id + ".old")
        family_path.rename(moved)
        family_path.mkdir()
        try:
            with pytest.raises(OutcomeDiagnosticResultStoreError):
                load_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)
        finally:
            import shutil

            shutil.rmtree(family_path, ignore_errors=True)
            moved.rename(family_path)


def test_expected_plan_never_reopens_protocol_authority(
    authorities, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, plan = authorities
    import levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol as protocol

    monkeypatch.setattr(
        protocol,
        "load_outcome_group_diagnostic_protocol",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("authority reopened")),
    )
    expected = build_outcome_diagnostic_expected_plan(plan, snapshot.protocol)
    assert len(expected.units) == EXPECTED_TOTAL_UNIT_COUNT


@pytest.mark.parametrize("mutation", ["insert", "replace"])
def test_final_pre_intent_scan_rejects_records_races(
    authorities, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    snapshot, plan = authorities
    _empty_root(snapshot.output_root)
    original = result_store._write_or_verify
    fired = False

    def racing_write(fd: int, name: str, value: object) -> None:
        nonlocal fired
        original(fd, name, value)
        if name != "namespace.json" or fired:
            return
        fired = True
        records = next(snapshot.output_root.glob(f"{FAMILIES[0]}/*/namespaces/*/records"))
        if mutation == "insert":
            (records / "foreign.json").write_text("{}")
        else:
            moved = records.with_name("records-old")
            records.rename(moved)
            records.mkdir()

    monkeypatch.setattr(result_store, "_write_or_verify", racing_write)
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        with pytest.raises(OutcomeDiagnosticResultStoreError, match="records"):
            prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)
    _empty_root(snapshot.output_root)


def test_open_pinned_rejects_namespaces_parent_swap(authorities) -> None:
    snapshot, plan = authorities
    _empty_root(snapshot.output_root)
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        store = prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)[0]
        namespaces = store.root / store.family_id / store.run_id / "namespaces"
        moved = namespaces.with_name("namespaces-old")
        namespaces.rename(moved)
        namespaces.mkdir()
        try:
            with pytest.raises(OutcomeDiagnosticResultStoreError, match="namespaces parent"):
                with store.open_pinned(lease):
                    pass
        finally:
            namespaces.rmdir()
            moved.rename(namespaces)


def test_open_pinned_owns_root_fd_and_rechecks_root_path(authorities) -> None:
    snapshot, plan = authorities
    _empty_root(snapshot.output_root)
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        store = prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)[0]
        root = snapshot.output_root
        moved = root.with_name(root.name + "-old")
        try:
            with pytest.raises(OutcomeDiagnosticResultStoreError, match="root path identity"):
                with store.open_pinned(lease) as descriptors:
                    root.rename(moved)
                    root.mkdir()
                    os.fstat(descriptors["root"])
        finally:
            if root.exists():
                root.rmdir()
            moved.rename(root)


def test_open_pinned_survives_inner_lease_close_without_fd_leak(authorities) -> None:
    snapshot, plan = authorities
    _empty_root(snapshot.output_root)
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        store = prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)[0]
        before = len(tuple(Path("/dev/fd").iterdir()))
        with pytest.raises(OutcomeDiagnosticResultStoreError, match="lease closed"):
            with store.open_pinned(lease) as descriptors:
                lease.close()
                os.fstat(descriptors[f"records:{CONDITIONS[0]}"])
        after = len(tuple(Path("/dev/fd").iterdir()))
        assert after <= before


def test_runtime_activation_writes_canonical_records_and_unbounded_attempts(authorities) -> None:
    snapshot, plan = authorities
    _empty_root(snapshot.output_root)
    now = datetime.now(timezone.utc)
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        stores = prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)
        expected = build_outcome_diagnostic_expected_plan(plan, snapshot.protocol)
        with activate_outcome_diagnostic_result_stores(
            stores, expected, lease, expected_git_commit=snapshot.git_commit_sha
        ) as batch:
            family = batch.stores[0]
            planned = family.planned_unit(stores[0].spec.units[0].unit_id)
            record = UnitRecord(
                run_id=family.run_id,
                config_sha256=family.config_sha256,
                unit_id=planned.unit_id,
                key=UnitKey(
                    phase="validation",
                    condition_id=f"{planned.condition_id}--{planned.tuple_id}",
                    family_id=planned.heldout_family,
                    task_id=planned.task_id,
                    task_index=planned.task_index,
                    replicate=planned.replicate,
                ),
                seeds=UnitSeeds(
                    model_seed=planned.model_seed,
                    environment_seed=planned.environment_seed,
                    probe_seed=planned.probe_seed,
                    search_seed=planned.search_seed,
                    data_order_seed=planned.data_order_seed,
                ),
                exposure_manifest_sha256=planned.exposure_manifest_sha256,
                started_at_utc=now,
                finished_at_utc=now,
                elapsed_wall_seconds=0.0,
                outcome=UnitOutcome(
                    evaluator_ran=False,
                    valid=False,
                    completed=False,
                    success=False,
                    performance_metric_id="performance_value",
                    performance_direction="minimize",
                ),
                accounting=ResourceAccounting(),
            )
            assert family.write_completed(record) is True
            assert family.write_completed(record) is False
            assert family.load_completed(planned.unit_id) == record
            assert record.key.condition_id == f"{planned.condition_id}--{planned.tuple_id}"
            attempt = AttemptRecord(
                run_id=family.run_id,
                config_sha256=family.config_sha256,
                unit_id=planned.unit_id,
                attempt=10_000,
                key=record.key,
                seeds=record.seeds,
                status="failed",
                stage="test",
                exception_type="RuntimeError",
                sanitized_message="test",
                retryable=True,
                started_at_utc=now,
                finished_at_utc=now,
                elapsed_wall_seconds=0.0,
            )
            assert family.write_attempt(attempt) is True
            assert family.next_attempt_number(planned.unit_id) == 10_001
            assert family.last_attempt_retryable(planned.unit_id) is True
            assert planned.unit_id in family.completed_unit_ids()
        # The durable activation marker is write-once and can be reopened
        # idempotently for resume while the readiness lease remains live.
        with activate_outcome_diagnostic_result_stores(
            stores, expected, lease, expected_git_commit=snapshot.git_commit_sha
        ) as resumed:
            assert planned.unit_id in resumed.completed_unit_ids(FAMILIES[0])
            assert resumed.store_for_family(FAMILIES[0]).next_attempt_number(planned.unit_id) == 10_001


def test_runtime_rejects_same_byte_marker_path_replacement(authorities) -> None:
    snapshot, plan = authorities
    _empty_root(snapshot.output_root)
    expected = build_outcome_diagnostic_expected_plan(plan, snapshot.protocol)
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        stores = prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)
        with pytest.raises(
            OutcomeDiagnosticResultStoreError, match=r"marker (?:identity )?changed"
        ):
            with activate_outcome_diagnostic_result_stores(
                stores, expected, lease, expected_git_commit=snapshot.git_commit_sha
            ) as batch:
                marker = snapshot.output_root / result_store.RUNTIME_ACTIVATION_MARKER_NAME
                content = marker.read_bytes()
                marker.unlink()
                marker.write_bytes(content)
                batch.completed_unit_ids()


def test_runtime_rejects_descendant_directory_path_substitution(authorities) -> None:
    snapshot, plan = authorities
    _empty_root(snapshot.output_root)
    expected = build_outcome_diagnostic_expected_plan(plan, snapshot.protocol)
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        stores = prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)
        with pytest.raises(OutcomeDiagnosticResultStoreError, match="path identity changed"):
            with activate_outcome_diagnostic_result_stores(
                stores, expected, lease, expected_git_commit=snapshot.git_commit_sha
            ) as batch:
                records = next(
                    snapshot.output_root.glob(
                        f"{FAMILIES[0]}/*/namespaces/{CONDITIONS[0]}/records"
                    )
                )
                records.rename(records.with_name("records-original"))
                records.mkdir()
                batch.completed_unit_ids()


def test_resume_inspector_accepts_prepared_and_activated_trees_without_parsing_results(
    authorities, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, plan = authorities
    _empty_root(snapshot.output_root)
    now = datetime.now(timezone.utc)
    expected = build_outcome_diagnostic_expected_plan(plan, snapshot.protocol)
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        stores = prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)
        prepared = result_store.inspect_outcome_diagnostic_resume_tree_at(
            lease.output_root_fd,
            snapshot.output_root,
            expected,
            output_state="prepared",
        )
        assert prepared.stores == stores
        assert prepared.records == ()
        assert prepared.marker_sha256 is None
        with pytest.raises(OutcomeDiagnosticResultStoreError, match="layout"):
            result_store.inspect_outcome_diagnostic_resume_tree_at(
                lease.output_root_fd,
                snapshot.output_root,
                expected,
                output_state="activated",
            )

        with activate_outcome_diagnostic_result_stores(
            stores, expected, lease, expected_git_commit=snapshot.git_commit_sha
        ) as batch:
            family = batch.stores[0]
            planned = family.planned_unit(stores[0].spec.units[0].unit_id)
            record = UnitRecord(
                run_id=family.run_id,
                config_sha256=family.config_sha256,
                unit_id=planned.unit_id,
                key=UnitKey(
                    phase="validation",
                    condition_id=f"{planned.condition_id}--{planned.tuple_id}",
                    family_id=planned.heldout_family,
                    task_id=planned.task_id,
                    task_index=planned.task_index,
                    replicate=planned.replicate,
                ),
                seeds=UnitSeeds(
                    model_seed=planned.model_seed,
                    environment_seed=planned.environment_seed,
                    probe_seed=planned.probe_seed,
                    search_seed=planned.search_seed,
                    data_order_seed=planned.data_order_seed,
                ),
                exposure_manifest_sha256=planned.exposure_manifest_sha256,
                started_at_utc=now,
                finished_at_utc=now,
                elapsed_wall_seconds=0.0,
                outcome=UnitOutcome(
                    evaluator_ran=False,
                    valid=False,
                    completed=False,
                    success=False,
                    performance_metric_id="performance_value",
                    performance_direction="minimize",
                ),
                accounting=ResourceAccounting(),
            )
            family.write_completed(record)

        monkeypatch.setattr(
            result_store,
            "_runtime_parse_record",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("readiness parsed comparative result content")
            ),
        )
        activated = result_store.inspect_outcome_diagnostic_resume_tree_at(
            lease.output_root_fd,
            snapshot.output_root,
            expected,
            output_state="activated",
        )
        assert len(activated.records) == 1
        assert activated.records[0].name == f"{planned.unit_id}.json"
        assert activated.marker_sha256 is not None
        with pytest.raises(OutcomeDiagnosticResultStoreError, match="layout"):
            result_store.inspect_outcome_diagnostic_resume_tree_at(
                lease.output_root_fd,
                snapshot.output_root,
                expected,
                output_state="prepared",
            )
        record_path = next(
            snapshot.output_root.glob(
                f"{planned.heldout_family}/*/namespaces/{planned.condition_id}/records/"
                f"{planned.unit_id}.json"
            )
        )
        original_snapshot = result_store._resume_file_snapshot
        replaced = False

        def replace_record_after_first_snapshot(directory_fd: int, name: str):
            nonlocal replaced
            observed = original_snapshot(directory_fd, name)
            if name == f"{planned.unit_id}.json" and not replaced:
                replaced = True
                content = record_path.read_bytes()
                record_path.unlink()
                record_path.write_bytes(content)
            return observed

        monkeypatch.setattr(
            result_store, "_resume_file_snapshot", replace_record_after_first_snapshot
        )
        with pytest.raises(OutcomeDiagnosticResultStoreError, match="record changed"):
            result_store.inspect_outcome_diagnostic_resume_tree_at(
                lease.output_root_fd,
                snapshot.output_root,
                expected,
                output_state="activated",
            )


def test_resume_inspector_rejects_same_byte_marker_replacement(
    authorities, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, plan = authorities
    _empty_root(snapshot.output_root)
    expected = build_outcome_diagnostic_expected_plan(plan, snapshot.protocol)
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        stores = prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)
        with activate_outcome_diagnostic_result_stores(
            stores, expected, lease, expected_git_commit=snapshot.git_commit_sha
        ):
            pass
        marker = snapshot.output_root / result_store.RUNTIME_ACTIVATION_MARKER_NAME
        original_snapshot = result_store._resume_file_snapshot
        replaced = False

        def replace_after_first_snapshot(directory_fd: int, name: str):
            nonlocal replaced
            observed = original_snapshot(directory_fd, name)
            if name == result_store.RUNTIME_ACTIVATION_MARKER_NAME and not replaced:
                replaced = True
                content = marker.read_bytes()
                marker.unlink()
                marker.write_bytes(content)
            return observed

        monkeypatch.setattr(result_store, "_resume_file_snapshot", replace_after_first_snapshot)
        with pytest.raises(OutcomeDiagnosticResultStoreError, match="marker changed"):
            result_store.inspect_outcome_diagnostic_resume_tree_at(
                lease.output_root_fd,
                snapshot.output_root,
                expected,
                output_state="activated",
            )


def test_readiness_explicitly_recaptures_prepared_and_activated_resume_states(
    authorities,
) -> None:
    snapshot, plan = authorities
    _empty_root(snapshot.output_root)
    expected = build_outcome_diagnostic_expected_plan(plan, snapshot.protocol)
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, plan)

    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="empty"):
        readiness.capture_outcome_group_diagnostic_readiness(
            repository=REPOSITORY,
            output_root=snapshot.output_root,
            expected_git_commit=snapshot.git_commit_sha,
        )
    prepared = readiness.capture_outcome_group_diagnostic_readiness(
        repository=REPOSITORY,
        output_root=snapshot.output_root,
        expected_git_commit=snapshot.git_commit_sha,
        output_state="prepared",
    )
    assert prepared.output_state == "prepared"
    assert prepared.resume_baseline.output_state == "prepared"
    with prepared.hold_for_activation(expected_git_commit=prepared.git_commit_sha) as lease:
        with activate_outcome_diagnostic_result_stores(
            prepared.resume_baseline.stores,
            expected,
            lease,
            expected_git_commit=prepared.git_commit_sha,
        ):
            pass

    activated = readiness.capture_outcome_group_diagnostic_readiness(
        repository=REPOSITORY,
        output_root=snapshot.output_root,
        expected_git_commit=snapshot.git_commit_sha,
        output_state="activated",
    )
    assert activated.output_state == "activated"
    assert activated.resume_baseline.output_state == "activated"
    with activated.hold_for_activation(expected_git_commit=activated.git_commit_sha) as lease:
        with activate_outcome_diagnostic_result_stores(
            activated.resume_baseline.stores,
            expected,
            lease,
            expected_git_commit=activated.git_commit_sha,
        ) as resumed:
            assert resumed.completed_unit_ids() == ()
