"""Adversarial tests for the outcome-diagnostic readiness boundary."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from levelup.experiments import milestone6_phase3_outcome_diagnostic_readiness as readiness
from levelup.experiments import milestone6_phase3_readiness as phase3

REPOSITORY = Path(readiness.ROOT)
OUTPUT_ROOT = REPOSITORY / readiness.DIAGNOSTIC_OUTPUT_ROOT_RELATIVE
_MODEL_STORE_ID = "phase3-model-preparation-cc08207"
_MODEL_METADATA_FIXTURE = Path(__file__).parent / "fixtures" / "phase3_model_preparation_metadata"


@pytest.fixture(scope="module", autouse=True)
def _materialize_metadata_only_model_store() -> Iterator[None]:
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
        phase3.PREPARATION_PROVENANCE_NAME: (
            "c1c302db1f88b62902628c839cd566ade6102bdb0716bcb505d09a5a49737679"
        ),
        phase3.PREPARATION_PROGRESS_NAME: (
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


@pytest.fixture
def snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[readiness.OutcomeDiagnosticReadinessSnapshot]:
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=REPOSITORY, text=True
    ).strip()
    monkeypatch.setattr(phase3, "_git_state", lambda _repository: (commit, False))
    output_root = OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    assert not tuple(output_root.iterdir())
    try:
        captured = readiness.capture_outcome_group_diagnostic_readiness(
            repository=REPOSITORY,
            output_root=output_root,
            expected_git_commit=commit,
        )
        yield captured
    finally:
        output_root.rmdir()


def test_snapshot_retains_diagnostic_and_recursive_authorities(snapshot) -> None:
    paths = {item.relative_path for item in snapshot.files}
    assert "configs/milestone6/phase3_outcome_group_diagnostic.json" in paths
    assert "configs/milestone6/phase3_development_selection.json" in paths
    assert len(snapshot.directories) == 4
    assert snapshot.source_result_lock_commit_sha == "9ff9596bf64ef341d69759c0e0db680c51e768f9"


def test_expected_commit_is_explicit_lowercase_and_exact(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    for value in (None, "a" * 39, "A" * 40, "g" * 40):
        with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="expected_git_commit"):
            readiness.capture_outcome_group_diagnostic_readiness(
                repository=REPOSITORY,
                output_root=output,
                expected_git_commit=value,  # type: ignore[arg-type]
            )


def test_byte_mutation_is_detected(snapshot, monkeypatch: pytest.MonkeyPatch) -> None:
    target = REPOSITORY / "configs" / "milestone6" / "phase3_outcome_group_diagnostic.json"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b" ")
        with pytest.raises(readiness.OutcomeDiagnosticReadinessError):
            snapshot.recheck(expected_git_commit=snapshot.git_commit_sha)
    finally:
        target.write_bytes(original)


def test_same_byte_inode_replacement_is_detected(snapshot) -> None:
    target = REPOSITORY / "configs" / "milestone6" / "phase3_outcome_group_diagnostic.json"
    original = target.read_bytes()
    replacement = target.with_name(target.name + ".replacement")
    try:
        replacement.write_bytes(original)
        os.replace(replacement, target)
        with pytest.raises(
            readiness.OutcomeDiagnosticReadinessError, match="source changed|bytes changed"
        ):
            snapshot.recheck(expected_git_commit=snapshot.git_commit_sha)
    finally:
        replacement.unlink(missing_ok=True)
        target.write_bytes(original)


def test_repository_output_and_ancestor_symlinks_are_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    child = real / "child"
    child.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError):
        readiness._reject_lexical_symlinks(link / "child", "authority")
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError):
        readiness._reject_lexical_symlinks(real / "missing" / "child", "output root")


def test_output_root_replacement_is_detected(snapshot) -> None:
    original = snapshot.output_root
    moved = original.with_name(original.name + ".old")
    try:
        original.rename(moved)
        original.mkdir()
        with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="output root identity"):
            snapshot.recheck(expected_git_commit=snapshot.git_commit_sha)
    finally:
        original.rmdir()
        moved.rename(original)


def test_existing_phase3_result_namespace_is_rejected(monkeypatch) -> None:
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=REPOSITORY, text=True
    ).strip()
    monkeypatch.setattr(phase3, "_git_state", lambda _repository: (commit, False))
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="canonical inert"):
        readiness.capture_outcome_group_diagnostic_readiness(
            repository=REPOSITORY,
            output_root=REPOSITORY / "runs" / "milestone6" / "phase3-development-results-b9f50db",
            expected_git_commit=commit,
        )


def test_commit_dirty_mismatch_and_drift_are_fail_closed(monkeypatch, snapshot) -> None:
    monkeypatch.setattr(phase3, "_git_state", lambda _repository: (snapshot.git_commit_sha, True))
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="provenance|clean"):
        snapshot.recheck(expected_git_commit=snapshot.git_commit_sha)
    monkeypatch.setattr(phase3, "_git_state", lambda _repository: ("0" * 40, False))
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="provenance"):
        snapshot.recheck(expected_git_commit=snapshot.git_commit_sha)


def test_final_family_authority_is_rejected(monkeypatch, snapshot) -> None:
    boundary = dict(snapshot.protocol.payload["execution_boundary"])
    boundary["final_family_access"] = True
    protocol = replace(
        snapshot.protocol, payload={**snapshot.protocol.payload, "execution_boundary": boundary}
    )
    monkeypatch.setattr(
        readiness, "load_outcome_group_diagnostic_protocol", lambda *_args, **_kwargs: protocol
    )
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="final-family"):
        readiness.capture_outcome_group_diagnostic_readiness(
            repository=REPOSITORY,
            output_root=OUTPUT_ROOT,
            expected_git_commit=snapshot.git_commit_sha,
        )


def test_activation_lease_is_live_and_double_close_safe(monkeypatch, snapshot) -> None:
    monkeypatch.setattr(phase3, "_git_state", lambda _repository: (snapshot.git_commit_sha, False))
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        assert lease.active
        lease.require_active()
        lease.close()
        assert not lease.active
        lease.close()
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="no longer active"):
        lease.require_active()


def test_hold_rejects_ancestor_identity_substitution(monkeypatch, snapshot) -> None:
    original = readiness._open_absolute_directory

    def substituted(path: Path, stack=None):
        fd, ancestors = original(path, stack)
        if stack is not None and path == snapshot.repository:
            return fd, (*ancestors, ("/race", (1, 1)))
        return fd, ancestors

    monkeypatch.setattr(readiness, "_open_absolute_directory", substituted)
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="ancestors"):
        with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha):
            pass


def test_lease_fd_and_map_tampering_fails_closed(monkeypatch, snapshot) -> None:
    monkeypatch.setattr(phase3, "_git_state", lambda _repository: (snapshot.git_commit_sha, False))
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        with pytest.raises(AttributeError):
            lease.repository_fd = lease.output_root_fd  # type: ignore[misc]
        with pytest.raises(TypeError):
            lease.file_descriptors["forged"] = lease.repository_fd  # type: ignore[index]
        object.__setattr__(lease, "_repository_fd", lease.output_root_fd)
        with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="identity"):
            lease.require_active()


def test_lease_snapshot_reassignment_fails_closed(monkeypatch, snapshot) -> None:
    monkeypatch.setattr(phase3, "_git_state", lambda _repository: (snapshot.git_commit_sha, False))
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        object.__setattr__(lease, "_snapshot", object())
        with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="reassigned"):
            lease.require_active()
        with pytest.raises(AttributeError):
            lease.snapshot = snapshot  # type: ignore[misc]


def test_reordered_file_manifest_is_rejected(monkeypatch, snapshot) -> None:
    monkeypatch.setattr(phase3, "_git_state", lambda _repository: (snapshot.git_commit_sha, False))
    object.__setattr__(snapshot, "files", tuple(reversed(snapshot.files)))
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="file manifest"):
        snapshot.recheck(expected_git_commit=snapshot.git_commit_sha)


def test_reordered_directory_manifest_is_rejected(monkeypatch, snapshot) -> None:
    monkeypatch.setattr(phase3, "_git_state", lambda _repository: (snapshot.git_commit_sha, False))
    object.__setattr__(snapshot, "directories", tuple(reversed(snapshot.directories)))
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="namespace manifest"):
        snapshot.recheck(expected_git_commit=snapshot.git_commit_sha)


def test_hold_empty_check_uses_held_output_descriptor(monkeypatch, snapshot) -> None:
    monkeypatch.setattr(phase3, "_git_state", lambda _repository: (snapshot.git_commit_sha, False))

    def path_reopen_forbidden(*_args, **_kwargs):
        raise AssertionError("output path was reopened for emptiness")

    monkeypatch.setattr(readiness, "_require_empty_diagnostic_output_root", path_reopen_forbidden)
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha):
        pass


def test_snapshot_protocol_payload_is_immutable(snapshot) -> None:
    with pytest.raises(TypeError):
        snapshot.protocol.payload["execution_boundary"]["final_family_access"] = True  # type: ignore[index]


def test_conflicting_duplicate_authority_paths_are_rejected(snapshot) -> None:
    original = snapshot.files[0]
    conflicting = replace(original, content=original.content + b"x")
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="conflicting"):
        readiness._merge_files(iter((conflicting,)), diagnostic=original)


def test_git_drift_after_descriptor_acquisition_is_rejected(monkeypatch, snapshot) -> None:
    calls = 0

    def drift(_repository):
        nonlocal calls
        calls += 1
        if calls == 1:
            return snapshot.git_commit_sha, False
        return "0" * 40, False

    monkeypatch.setattr(phase3, "_git_state", drift)
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="provenance|authorised"):
        with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha):
            pass
    assert calls >= 2


def test_activation_lease_cannot_be_forged(snapshot) -> None:
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="forged"):
        readiness.OutcomeDiagnosticActivationReadinessLease(
            snapshot,
            1,
            2,
            {},
            {},
            object(),  # type: ignore[arg-type]
        )


def test_readiness_does_not_use_path_inventory_walkers(monkeypatch, snapshot) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("path inventory walker called")

    monkeypatch.setattr(Path, "rglob", fail)
    monkeypatch.setattr(Path, "glob", fail)
    snapshot.recheck(expected_git_commit=snapshot.git_commit_sha)


def test_outcome_model_authority_path_is_exact_and_repository_relative(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    canonical = repository / readiness.OUTCOME_MODEL_AUTHORITY_RELATIVE
    canonical.parent.mkdir(parents=True)
    assert readiness._outcome_model_authority_path(repository, None) == canonical
    assert readiness._outcome_model_authority_path(repository, canonical) == canonical
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="canonical path"):
        readiness._outcome_model_authority_path(repository, "configs/other.json")


def test_model_store_readiness_lease_rechecks_complete_identity_without_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import levelup.experiments.milestone6_phase3_outcome_diagnostic_model_store as model_store

    first = object()
    second = object()
    current = [first]

    class FakeStore:
        def recheck(self) -> None:
            return None

    monkeypatch.setattr(
        model_store,
        "snapshot_outcome_model_store_identities_at",
        lambda _store, _owners: current[0],
    )
    stack = ExitStack()
    stack.__enter__()
    lease = readiness.OutcomeDiagnosticModelReadinessLease(
        FakeStore(), stack, tuple(f"{i:064x}" for i in range(240)), first, _token=readiness._LEASE_TOKEN
    )
    assert lease.require_active().active
    current[0] = second
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="identities changed"):
        lease.require_active()
    lease.close()
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="no longer active"):
        lease.require_active()


def test_model_store_lease_and_snapshot_cannot_be_forged() -> None:
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="cannot be forged"):
        readiness.OutcomeDiagnosticModelReadinessLease(
            SimpleNamespace(recheck=lambda: None),
            ExitStack(),
            (),
            object(),
        )


def _fake_model_readiness_inputs(tmp_path: Path):
    """Build a complete mocked authority/store surface under ``tmp_path`` only."""

    from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
        CONDITIONS,
        outcome_artifact_store_id,
    )

    repository = tmp_path / "repository"
    authority_path = repository / readiness.OUTCOME_MODEL_AUTHORITY_RELATIVE
    authority_path.parent.mkdir(parents=True)
    authority_path.write_bytes(b"authority-bytes")
    store_root = repository / "runs" / "milestone6" / outcome_artifact_store_id("1" * 64)
    store_root.mkdir(parents=True)
    protocol = SimpleNamespace(payload={}, sha256="p" * 64)
    base = SimpleNamespace(repository=repository, protocol=protocol, recheck=lambda **_: None)

    families = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
    evidence_rows = []
    evidence_by_key = {}
    for family in families:
        for replicate in range(5):
            body = {
                "family_id": family,
                "replicate": replicate,
                "payload_sha256": "b" * 64,
                "payload_bytes": 10,
                "ordered_training_task_ids": [f"task-{i}" for i in range(40)],
            }
            raw = readiness.canonical_json_bytes(body)
            evidence_rows.append(raw)
            evidence_by_key[(family, replicate)] = body
    views = []
    owners = []
    for condition_index, condition in enumerate(CONDITIONS):
        for family in families:
            for replicate in range(5):
                view_id = f"{len(views) + 1:064x}"
                view = SimpleNamespace(
                    view_id=view_id,
                    condition_id=condition,
                    heldout_family=family,
                    replicate=replicate,
                    feature_mask_sha256="f" * 64,
                    transformation_sha256="t" * 64,
                    representation_sha256="r" * 64,
                    data_order_seed=100 + len(views),
                )
                views.append(view)
                for _ in range(4):
                    owner_id = f"{len(owners) + 1:064x}"
                    owners.append(
                        SimpleNamespace(
                            owner_id=owner_id,
                            view_id=view_id,
                            condition_id=condition,
                            heldout_family=family,
                            fold_id=f"lofo-{family}",
                            replicate=replicate,
                            training_tuple_id="tuple",
                            model_seed=1,
                            feature_mask_sha256="f" * 64,
                            transformation_sha256="t" * 64,
                            model_identity_sha256="m" * 64,
                    )
                    )
    raw_by_key = {
        (__import__("json").loads(raw)["family_id"], __import__("json").loads(raw)["replicate"]): raw
        for raw in evidence_rows
    }
    for view in views:
        view.evidence_row_sha256 = __import__("hashlib").sha256(
            raw_by_key[(view.heldout_family, view.replicate)]
        ).hexdigest()
    artifacts = [
        SimpleNamespace(
            owner_id=owner.owner_id,
            view_id=owner.view_id,
            condition_id=owner.condition_id,
            heldout_family=owner.heldout_family,
            fold_id=owner.fold_id,
            replicate=owner.replicate,
            training_tuple_id=owner.training_tuple_id,
            model_seed=owner.model_seed,
            data_order_seed=views[int(owner.view_id, 16) - 1].data_order_seed,
            feature_mask_sha256=owner.feature_mask_sha256,
            transformation_sha256=owner.transformation_sha256,
            representation_sha256="r" * 64,
            model_identity_sha256=owner.model_identity_sha256,
            consumer_unit_ids_sha256="u" * 64,
            consumer_seed_lineage_sha256="s" * 64,
            record_id="d" * 64,
            key_id="k" * 64,
            model_state_sha256="z" * 64,
        )
        for owner in owners
    ]
    evidence = [
        SimpleNamespace(
            heldout_family=family,
            replicate=replicate,
            evidence_row_sha256=__import__("hashlib").sha256(
                next(raw for raw in evidence_rows if __import__("json").loads(raw)["family_id"] == family and __import__("json").loads(raw)["replicate"] == replicate)
            ).hexdigest(),
            evidence_payload_sha256="b" * 64,
            evidence_payload_bytes=10,
            ordered_training_task_ids=tuple(f"task-{i}" for i in range(40)),
        )
        for family in families
        for replicate in range(5)
    ]
    authority = SimpleNamespace(
        plan_id="1" * 64,
        plan_parent_commit_sha="c" * 40,
        protocol_sha256="q" * 64,
        protocol_self_sha256="h" * 64,
        protocol_file_sha256="p" * 64,
        artifact_store_id=outcome_artifact_store_id("1" * 64),
        condition_ids=CONDITIONS,
        views=tuple(views),
        evidence=tuple(evidence),
        artifacts=tuple(artifacts),
    )
    manifest_entries = tuple(
        SimpleNamespace(
            owner_id=row.owner_id,
            record_id=row.record_id,
            key_id=row.key_id,
            model_state_sha256=row.model_state_sha256,
        )
        for row in artifacts
    )
    manifest = SimpleNamespace(entries=manifest_entries)
    return repository, store_root, base, authority, manifest


def test_model_readiness_capture_success_pins_complete_mock_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, store_root, base, authority, manifest = _fake_model_readiness_inputs(tmp_path)
    monkeypatch.setattr(readiness, "capture_outcome_group_diagnostic_readiness", lambda **_: base)
    monkeypatch.setattr(readiness, "_validate_outcome_model_authority", lambda *_: authority)

    class FakeStore:
        def recheck(self) -> None:
            return None

    @contextmanager
    def open_store(_root):
        yield FakeStore()

    import levelup.experiments.milestone6_phase3_outcome_diagnostic_model_store as model_store

    identities = object()
    monkeypatch.setattr(model_store, "open_existing_outcome_model_store", open_store)
    monkeypatch.setattr(model_store, "load_outcome_model_manifest_at", lambda _store: manifest)
    monkeypatch.setattr(model_store, "snapshot_outcome_model_store_identities_at", lambda *_: identities)
    monkeypatch.setattr(readiness, "_validate_store_payloads_against_authority", lambda *_: None)
    captured = readiness.capture_outcome_group_diagnostic_model_readiness(
        repository=repository,
        output_root=tmp_path / "output",
        model_store_root=store_root,
        expected_git_commit="a" * 40,
    )
    assert captured.model_store_root == store_root
    assert len(captured.owner_ids) == 240
    assert captured.authority_bytes == b"authority-bytes"
    captured.close()


def test_model_readiness_capture_closes_store_on_manifest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, store_root, base, authority, _manifest = _fake_model_readiness_inputs(tmp_path)
    monkeypatch.setattr(readiness, "capture_outcome_group_diagnostic_readiness", lambda **_: base)
    monkeypatch.setattr(readiness, "_validate_outcome_model_authority", lambda *_: authority)
    closed = []

    class FakeStore:
        def recheck(self) -> None:
            return None

    @contextmanager
    def open_store(_root):
        try:
            yield FakeStore()
        finally:
            closed.append(True)

    import levelup.experiments.milestone6_phase3_outcome_diagnostic_model_store as model_store

    monkeypatch.setattr(model_store, "open_existing_outcome_model_store", open_store)
    monkeypatch.setattr(model_store, "load_outcome_model_manifest_at", lambda _store: SimpleNamespace(entries=()))
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="manifest"):
        readiness.capture_outcome_group_diagnostic_model_readiness(
            repository=repository, output_root=tmp_path / "output", expected_git_commit="a" * 40
        )
    assert closed == [True]


def test_model_readiness_rejects_payload_lineage_mismatch_and_closes_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, store_root, base, authority, manifest = _fake_model_readiness_inputs(tmp_path)
    monkeypatch.setattr(readiness, "capture_outcome_group_diagnostic_readiness", lambda **_: base)
    monkeypatch.setattr(readiness, "_validate_outcome_model_authority", lambda *_: authority)
    closed = []

    class FakeStore:
        def recheck(self) -> None:
            return None

    @contextmanager
    def open_store(_root):
        try:
            yield FakeStore()
        finally:
            closed.append(True)

    import levelup.experiments.milestone6_phase3_outcome_diagnostic_model_store as model_store

    monkeypatch.setattr(model_store, "open_existing_outcome_model_store", open_store)
    monkeypatch.setattr(model_store, "load_outcome_model_manifest_at", lambda _store: manifest)
    monkeypatch.setattr(model_store, "snapshot_outcome_model_store_identities_at", lambda *_: object())
    monkeypatch.setattr(
        readiness,
        "_validate_store_payloads_against_authority",
        lambda *_: (_ for _ in ()).throw(
            readiness.OutcomeDiagnosticReadinessError("payload lineage mismatch")
        ),
    )
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="payload lineage"):
        readiness.capture_outcome_group_diagnostic_model_readiness(
            repository=repository,
            output_root=tmp_path / "output",
            model_store_root=store_root,
            expected_git_commit="a" * 40,
        )
    assert closed == [True]


def test_model_readiness_recheck_detects_authority_content_and_inode_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, store_root, base, authority, manifest = _fake_model_readiness_inputs(tmp_path)
    monkeypatch.setattr(readiness, "capture_outcome_group_diagnostic_readiness", lambda **_: base)
    monkeypatch.setattr(readiness, "_validate_outcome_model_authority", lambda *_: authority)

    class FakeStore:
        def recheck(self) -> None:
            return None

    @contextmanager
    def open_store(_root):
        yield FakeStore()

    import levelup.experiments.milestone6_phase3_outcome_diagnostic_model_store as model_store

    monkeypatch.setattr(model_store, "open_existing_outcome_model_store", open_store)
    monkeypatch.setattr(model_store, "load_outcome_model_manifest_at", lambda _store: manifest)
    identities = object()
    monkeypatch.setattr(model_store, "snapshot_outcome_model_store_identities_at", lambda *_: identities)
    monkeypatch.setattr(readiness, "_validate_store_payloads_against_authority", lambda *_: None)
    captured = readiness.capture_outcome_group_diagnostic_model_readiness(
        repository=repository, output_root=tmp_path / "output", expected_git_commit="a" * 40
    )
    authority_path = repository / readiness.OUTCOME_MODEL_AUTHORITY_RELATIVE
    authority_path.write_bytes(b"changed")
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="authority bytes"):
        captured.recheck(expected_git_commit="a" * 40)
    captured.close()


def test_model_readiness_rejects_missing_partial_and_symlink_store_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, store_root, base, authority, _manifest = _fake_model_readiness_inputs(tmp_path)
    monkeypatch.setattr(readiness, "capture_outcome_group_diagnostic_readiness", lambda **_: base)
    monkeypatch.setattr(readiness, "_validate_outcome_model_authority", lambda *_: authority)
    import levelup.experiments.milestone6_phase3_outcome_diagnostic_model_store as model_store

    original = tuple(sorted(path.name for path in store_root.iterdir()))
    store_root_backup = store_root.with_name(store_root.name + ".backup")
    store_root.rename(store_root_backup)
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError):
        readiness.capture_outcome_group_diagnostic_model_readiness(
            repository=repository, output_root=tmp_path / "output", expected_git_commit="a" * 40
        )
    store_root_backup.rename(store_root)
    assert tuple(sorted(path.name for path in store_root.iterdir())) == original

    monkeypatch.setattr(
        model_store,
        "open_existing_outcome_model_store",
        lambda _root: (_ for _ in ()).throw(RuntimeError("partial namespace")),
    )
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError):
        readiness.capture_outcome_group_diagnostic_model_readiness(
            repository=repository, output_root=tmp_path / "output", expected_git_commit="a" * 40
        )
    assert tuple(sorted(path.name for path in store_root.iterdir())) == original

    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    store_root.rmdir()
    store_root.symlink_to(symlink_target, target_is_directory=True)
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError, match="symlink"):
        readiness.capture_outcome_group_diagnostic_model_readiness(
            repository=repository, output_root=tmp_path / "output", expected_git_commit="a" * 40
        )
    store_root.unlink()


def test_outcome_authority_validator_rejects_tampered_evidence_view_and_owner_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _store_root, _base, fake, _manifest = _fake_model_readiness_inputs(tmp_path)
    import levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts as artifacts
    import levelup.experiments.milestone6_phase3_outcome_diagnostic_plan as plan_module

    protocol = SimpleNamespace(
        payload={"diagnostic_protocol_sha256": "h" * 64}, sha256="p" * 64
    )
    plan = SimpleNamespace(
        plan_id="1" * 64,
        parent_commit_sha="c" * 40,
        protocol_sha256="q" * 64,
        condition_ids=fake.condition_ids,
        views=fake.views,
        model_owners=fake.artifacts,
        evidence_lineage_rows=tuple(
            readiness.canonical_json_bytes(
                {
                    "family_id": family,
                    "replicate": replicate,
                    "payload_sha256": "b" * 64,
                    "payload_bytes": 10,
                    "ordered_training_task_ids": [f"task-{i}" for i in range(40)],
                }
            )
            for family in ("plain", "battery", "cooldown", "heat", "momentum", "combo")
            for replicate in range(5)
        ),
    )
    authority = artifacts.OutcomeDiagnosticModelArtifactAuthority.model_construct(
        plan_id=fake.plan_id,
        plan_parent_commit_sha=fake.plan_parent_commit_sha,
        protocol_sha256=fake.protocol_sha256,
        protocol_self_sha256="h" * 64,
        protocol_file_sha256="p" * 64,
        artifact_store_id=fake.artifact_store_id,
        condition_ids=fake.condition_ids,
        views=fake.views,
        evidence=fake.evidence,
        artifacts=fake.artifacts,
    )
    monkeypatch.setattr(artifacts, "load_outcome_model_artifact_authority_bytes", lambda _: authority)
    monkeypatch.setattr(artifacts, "canonical_outcome_model_artifact_authority_bytes", lambda _: b"authority-bytes")
    monkeypatch.setattr(plan_module, "build_outcome_group_diagnostic_plan", lambda _: plan)
    monkeypatch.setattr(plan_module, "bind_validated_outcome_diagnostic_plan", lambda *_args, **_kwargs: SimpleNamespace(plan=plan))
    assert readiness._validate_outcome_model_authority(b"authority-bytes", protocol, repository) is authority

    def clone(item, **updates):
        return SimpleNamespace(**{**vars(item), **updates})

    tampered_evidence = authority.evidence[:-1] + (clone(authority.evidence[-1], evidence_payload_bytes=11),)
    monkeypatch.setattr(artifacts, "load_outcome_model_artifact_authority_bytes", lambda _: authority.model_copy(update={"evidence": tampered_evidence}))
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError) as evidence_error:
        readiness._validate_outcome_model_authority(b"authority-bytes", protocol, repository)
    assert "evidence universe" in str(evidence_error.value.__cause__)

    tampered_views = authority.views[:-1] + (clone(authority.views[-1], representation_sha256="x" * 64),)
    monkeypatch.setattr(artifacts, "load_outcome_model_artifact_authority_bytes", lambda _: authority.model_copy(update={"views": tampered_views}))
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError) as view_error:
        readiness._validate_outcome_model_authority(b"authority-bytes", protocol, repository)
    assert "view universe" in str(view_error.value.__cause__)

    tampered_artifacts = authority.artifacts[:-1] + (clone(authority.artifacts[-1], model_identity_sha256="x" * 64),)
    monkeypatch.setattr(artifacts, "load_outcome_model_artifact_authority_bytes", lambda _: authority.model_copy(update={"artifacts": tampered_artifacts}))
    with pytest.raises(readiness.OutcomeDiagnosticReadinessError) as owner_error:
        readiness._validate_outcome_model_authority(b"authority-bytes", protocol, repository)
    assert "owner universe" in str(owner_error.value.__cause__)
