"""Adversarial tests for the read-only Phase 3 authority preflight."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from levelup.experiments import milestone6_phase3_readiness as readiness
from levelup.experiments.runner.config import canonical_json_bytes

_MODEL_STORE_ID = "phase3-model-preparation-cc08207"
_MODEL_METADATA_FIXTURE = Path(__file__).parent / "fixtures" / "phase3_model_preparation_metadata"


@pytest.fixture(scope="module", autouse=True)
def _materialize_metadata_only_model_store() -> Iterator[None]:
    """Provide CI with exact small metadata, never fake checkpoint contents."""

    model_root = Path(readiness.ROOT) / "runs" / "milestone6" / _MODEL_STORE_ID
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
    created_files = [
        model_root / readiness.PREPARATION_PROVENANCE_NAME,
        model_root / readiness.PREPARATION_PROGRESS_NAME,
    ]
    for destination in created_files:
        source = _MODEL_METADATA_FIXTURE / destination.name
        expected = {
            readiness.PREPARATION_PROVENANCE_NAME: (
                "c1c302db1f88b62902628c839cd566ade6102bdb0716bcb505d09a5a49737679"
            ),
            readiness.PREPARATION_PROGRESS_NAME: (
                "e5ff3c385c6f32ca9e5dac04b4a81e229c0bfb073300ac4505edfd419ff7d11b"
            ),
        }[destination.name]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
        shutil.copyfile(source, destination)
    assert all(not any(path.iterdir()) for path in created_directories)
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


def _report_snapshot(
    snapshot: readiness.Phase3ReadinessSnapshot,
) -> readiness.AuthorityFileSnapshot:
    return next(
        item
        for item in snapshot.files
        if item.relative_path == readiness.PHASE3_TRAINING_SHUFFLE_REPORT_RELATIVE
    )


def _snapshot_with_report_bytes(
    snapshot: readiness.Phase3ReadinessSnapshot,
    content: bytes,
) -> readiness.Phase3ReadinessSnapshot:
    original = _report_snapshot(snapshot)
    replacement = replace(
        original,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
    )
    return replace(
        snapshot,
        files=tuple(replacement if item is original else item for item in snapshot.files),
    )


def _canonical_report_body(snapshot: readiness.Phase3ReadinessSnapshot) -> dict[str, object]:
    body = json.loads(_report_snapshot(snapshot).content)
    unsigned = dict(body)
    unsigned.pop("report_sha256", None)
    body["report_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    return body


def test_metadata_only_fixture_matches_published_authority() -> None:
    provenance = _MODEL_METADATA_FIXTURE / readiness.PREPARATION_PROVENANCE_NAME
    progress = _MODEL_METADATA_FIXTURE / readiness.PREPARATION_PROGRESS_NAME
    assert hashlib.sha256(provenance.read_bytes()).hexdigest() == (
        "c1c302db1f88b62902628c839cd566ade6102bdb0716bcb505d09a5a49737679"
    )
    assert hashlib.sha256(progress.read_bytes()).hexdigest() == (
        "e5ff3c385c6f32ca9e5dac04b4a81e229c0bfb073300ac4505edfd419ff7d11b"
    )


def test_readiness_captures_all_published_development_authorities() -> None:
    snapshot = readiness.capture_phase3_readiness()
    paths = {item.relative_path for item in snapshot.files}
    assert readiness.PHASE3_PROTOCOL_RELATIVE in paths
    assert readiness.PHASE3_PLAN_LOCK_RELATIVE in paths
    assert readiness.PHASE3_ANCHOR_RELATIVE in paths
    assert readiness.PHASE3_EVIDENCE_RELATIVE in paths
    assert readiness.PHASE3_MODEL_AUTHORITY_RELATIVE in paths
    assert readiness.PHASE3_TRAINING_SHUFFLE_REPORT_RELATIVE in paths
    assert readiness.PHASE3_ANCHOR_SELECTION_METRICS_RELATIVE in paths
    assert len(paths) == 13
    assert len(snapshot.directories) == 4
    assert {Path(item.relative_path).name for item in snapshot.directories} >= {
        "phase3-model-artifact-keys",
        "phase3-model-artifact-costs",
        "phase3-model-artifacts",
    }
    assert snapshot.plan_id == readiness.PHASE3_PLAN_ID
    assert (
        snapshot.training_shuffle_report_sha256 == readiness.PHASE3_TRAINING_SHUFFLE_REPORT_SHA256
    )
    assert (
        snapshot.training_shuffle_report_file_sha256
        == readiness.PHASE3_TRAINING_SHUFFLE_REPORT_FILE_SHA256
    )
    snapshot.recheck()


def test_training_shuffle_report_missing_is_rejected() -> None:
    snapshot = readiness.capture_phase3_readiness()
    missing = replace(
        snapshot,
        files=tuple(
            item
            for item in snapshot.files
            if item.relative_path != readiness.PHASE3_TRAINING_SHUFFLE_REPORT_RELATIVE
        ),
    )
    with pytest.raises(readiness.Phase3ReadinessError, match="report is missing"):
        readiness._validate_authority_files(missing)


def test_training_shuffle_report_schema_is_rejected() -> None:
    snapshot = readiness.capture_phase3_readiness()
    body = _canonical_report_body(snapshot)
    body.pop("scope")
    content = canonical_json_bytes(body)
    invalid = _snapshot_with_report_bytes(snapshot, content)
    with pytest.raises(readiness.Phase3ReadinessError, match="not canonical"):
        readiness._validate_authority_files(invalid)


def test_training_shuffle_report_self_hash_is_rejected() -> None:
    snapshot = readiness.capture_phase3_readiness()
    body = json.loads(_report_snapshot(snapshot).content)
    body["report_sha256"] = "0" * 64
    invalid = _snapshot_with_report_bytes(snapshot, canonical_json_bytes(body))
    with pytest.raises(readiness.Phase3ReadinessError, match="not canonical"):
        readiness._validate_authority_files(invalid)


def test_training_shuffle_report_view_lineage_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = readiness.capture_phase3_readiness()
    body = _canonical_report_body(snapshot)
    for view in body["views"]:
        view["plan_id"] = "0" * 64
    unsigned = dict(body)
    unsigned.pop("report_sha256")
    body["report_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    content = canonical_json_bytes(body)
    monkeypatch.setattr(
        readiness,
        "PHASE3_TRAINING_SHUFFLE_REPORT_SHA256",
        body["report_sha256"],
    )
    monkeypatch.setattr(
        readiness,
        "PHASE3_TRAINING_SHUFFLE_REPORT_FILE_SHA256",
        hashlib.sha256(content).hexdigest(),
    )
    invalid = _snapshot_with_report_bytes(snapshot, content)
    with pytest.raises(readiness.Phase3ReadinessError, match="view lineage"):
        readiness._validate_authority_files(invalid)


def test_training_shuffle_report_noncanonical_bytes_are_rejected() -> None:
    snapshot = readiness.capture_phase3_readiness()
    invalid = _snapshot_with_report_bytes(snapshot, _report_snapshot(snapshot).content + b"\n")
    with pytest.raises(readiness.Phase3ReadinessError, match="not canonical"):
        readiness._validate_authority_files(invalid)


def test_training_shuffle_report_same_byte_replacement_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = readiness.capture_phase3_readiness()
    original_read = readiness._read_source
    report_path = readiness.PHASE3_TRAINING_SHUFFLE_REPORT_RELATIVE

    def replaced(repository: Path, relative_path: str) -> readiness.AuthorityFileSnapshot:
        current = original_read(repository, relative_path)
        if relative_path == report_path:
            return replace(
                current,
                file_identity=(current.file_identity[0], current.file_identity[1] + 1),
            )
        return current

    monkeypatch.setattr(readiness, "_read_source", replaced)
    with pytest.raises(readiness.Phase3ReadinessError, match="source changed"):
        snapshot.recheck()


def test_source_byte_mutation_is_detected(tmp_path: Path) -> None:
    source = tmp_path / "authority.json"
    source.write_bytes(b'{"x":1}')
    original = readiness._read_source(tmp_path, "authority.json")
    source.write_bytes(b'{"x":2}')
    changed = readiness._read_source(tmp_path, "authority.json")
    assert changed.content != original.content
    assert changed.sha256 != original.sha256


def test_same_byte_inode_replacement_is_detected(tmp_path: Path) -> None:
    source = tmp_path / "authority.json"
    source.write_bytes(b"same bytes")
    original = readiness._read_source(tmp_path, "authority.json")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(original.content)
    os.replace(replacement, source)
    changed = readiness._read_source(tmp_path, "authority.json")
    assert changed.content == original.content
    assert changed.file_identity != original.file_identity


def test_symlink_substitution_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    link = tmp_path / "authority.json"
    link.symlink_to(target)
    with pytest.raises(readiness.Phase3ReadinessError):
        readiness._read_source(tmp_path, "authority.json")


def test_directory_same_path_replacement_is_detected(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    original = readiness._read_directory(tmp_path, "model")
    replaced = tmp_path / "old-model"
    model.rename(replaced)
    model.mkdir()
    changed = readiness._read_directory(tmp_path, "model")
    assert changed.identity != original.identity


def test_directory_symlink_substitution_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "model"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(readiness.Phase3ReadinessError):
        readiness._read_directory(tmp_path, "model")


def test_recheck_rejects_repository_commit_or_dirty_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = readiness.capture_phase3_readiness()
    snapshot = replace(snapshot, git_dirty=False)
    monkeypatch.setattr(
        readiness,
        "_git_state",
        lambda _repository: ("0" * len(snapshot.git_commit_sha), False),
    )
    with pytest.raises(readiness.Phase3ReadinessError, match="provenance"):
        snapshot.recheck()


def test_execution_preflight_rejects_dirty_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = replace(readiness.capture_phase3_readiness(), git_dirty=True)
    monkeypatch.setattr(
        readiness, "_git_state", lambda _repository: (snapshot.git_commit_sha, True)
    )
    with pytest.raises(readiness.Phase3ReadinessError, match="clean repository"):
        snapshot.preflight(expected_git_commit=snapshot.git_commit_sha)


def test_execution_preflight_requires_exact_authorized_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = readiness.capture_phase3_readiness()
    snapshot = replace(snapshot, git_dirty=False)
    monkeypatch.setattr(
        readiness,
        "_git_state",
        lambda _repository: (snapshot.git_commit_sha, False),
    )
    with pytest.raises(readiness.Phase3ReadinessError, match="not authorized"):
        snapshot.preflight(expected_git_commit="0" * len(snapshot.git_commit_sha))


def test_activation_lease_holds_all_sources_and_deactivates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = replace(readiness.capture_phase3_readiness(), git_dirty=False)
    monkeypatch.setattr(
        readiness,
        "_git_state",
        lambda _repository: (snapshot.git_commit_sha, False),
    )
    with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
        assert lease.active is True
        assert set(lease.file_descriptors) == {item.relative_path for item in snapshot.files}
        assert set(lease.directory_descriptors) == {
            item.relative_path for item in snapshot.directories
        }
        lease.require_active()
    assert lease.active is False
    with pytest.raises(readiness.Phase3ReadinessError, match="no longer active"):
        lease.require_active()


def test_activation_lease_rechecks_report_after_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = replace(readiness.capture_phase3_readiness(), git_dirty=False)
    monkeypatch.setattr(
        readiness,
        "_git_state",
        lambda _repository: (snapshot.git_commit_sha, False),
    )
    original_read = readiness._read_source
    report_path = readiness.PHASE3_TRAINING_SHUFFLE_REPORT_RELATIVE

    def replaced(
        repository: Path,
        relative_path: str,
    ) -> readiness.AuthorityFileSnapshot:
        current = original_read(repository, relative_path)
        if relative_path == report_path:
            return replace(
                current,
                file_identity=(current.file_identity[0], current.file_identity[1] + 1),
            )
        return current

    with pytest.raises(readiness.Phase3ReadinessError, match="source changed"):
        with snapshot.hold_for_activation(expected_git_commit=snapshot.git_commit_sha) as lease:
            assert report_path in lease.file_descriptors
            monkeypatch.setattr(readiness, "_read_source", replaced)


def test_activation_lease_cannot_be_forged() -> None:
    snapshot = readiness.capture_phase3_readiness()
    with pytest.raises(readiness.Phase3ReadinessError, match="canonical"):
        readiness.Phase3ActivationReadinessLease(snapshot, {}, {})
