"""Adversarial tests for the read-only Phase 3 authority preflight."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from levelup.experiments import milestone6_phase3_readiness as readiness


def test_readiness_captures_all_published_development_authorities() -> None:
    snapshot = readiness.capture_phase3_readiness()
    paths = {item.relative_path for item in snapshot.files}
    assert readiness.PHASE3_PROTOCOL_RELATIVE in paths
    assert readiness.PHASE3_PLAN_LOCK_RELATIVE in paths
    assert readiness.PHASE3_ANCHOR_RELATIVE in paths
    assert readiness.PHASE3_EVIDENCE_RELATIVE in paths
    assert readiness.PHASE3_MODEL_AUTHORITY_RELATIVE in paths
    assert len(paths) == 11
    assert len(snapshot.directories) == 4
    assert {Path(item.relative_path).name for item in snapshot.directories} >= {
        "phase3-model-artifact-keys",
        "phase3-model-artifact-costs",
        "phase3-model-artifacts",
    }
    assert snapshot.plan_id == readiness.PHASE3_PLAN_ID
    snapshot.recheck()


def test_source_byte_mutation_is_detected(tmp_path: Path) -> None:
    source = tmp_path / "authority.json"
    source.write_bytes(b"{\"x\":1}")
    original = readiness._read_source(tmp_path, "authority.json")
    source.write_bytes(b"{\"x\":2}")
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
    snapshot = readiness.capture_phase3_readiness()
    monkeypatch.setattr(readiness, "_git_state", lambda _repository: (snapshot.git_commit_sha, True))
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
    with snapshot.hold_for_activation(
        expected_git_commit=snapshot.git_commit_sha
    ) as lease:
        assert lease.active is True
        assert set(lease.file_descriptors) == {
            item.relative_path for item in snapshot.files
        }
        assert set(lease.directory_descriptors) == {
            item.relative_path for item in snapshot.directories
        }
        lease.require_active()
    assert lease.active is False
    with pytest.raises(readiness.Phase3ReadinessError, match="no longer active"):
        lease.require_active()


def test_activation_lease_cannot_be_forged() -> None:
    snapshot = readiness.capture_phase3_readiness()
    with pytest.raises(readiness.Phase3ReadinessError, match="canonical"):
        readiness.Phase3ActivationReadinessLease(snapshot, {}, {})
