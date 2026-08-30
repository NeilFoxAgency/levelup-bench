"""Development-only readiness tests for local-affordance raw publication."""

from __future__ import annotations

import copy
import hashlib
import os
import shutil
from pathlib import Path

import pytest

from levelup.experiments import milestone6_phase3_local_affordance_readiness as readiness
from levelup.experiments.milestone6_phase3_local_affordance_raw_authority import (
    require_expected_raw_probe_authority,
)

ROOT = Path(__file__).resolve().parents[1]
CLEAN_COMMIT = "a" * 40


@pytest.fixture
def authority_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    for relative in readiness.SOURCE_RELATIVE_PATHS:
        source = ROOT / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return repository


@pytest.fixture
def clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(readiness, "_git_state", lambda _repository: (CLEAN_COMMIT, False))


def _capture(
    repository: Path,
    parent: Path,
) -> readiness.LocalAffordanceReadinessSnapshot:
    parent.mkdir(exist_ok=True)
    return readiness.capture_local_affordance_readiness(
        repository,
        raw_publication_destination=parent / "raw-authority",
    )


def test_diagnostic_capture_is_exact_and_side_effect_free(
    authority_repository: Path,
    tmp_path: Path,
    clean_git: None,
) -> None:
    parent = tmp_path / "raw-parent"
    snapshot = _capture(authority_repository, parent)

    assert readiness.require_local_affordance_readiness_snapshot(snapshot) is snapshot
    assert tuple(item.relative_path for item in snapshot.sources) == readiness.SOURCE_RELATIVE_PATHS
    assert all(item.sha256 == hashlib.sha256(item.content).hexdigest() for item in snapshot.sources)
    assert require_expected_raw_probe_authority(snapshot.authority) is snapshot.authority
    assert snapshot.destination_parent == parent
    assert snapshot.destination_name == "raw-authority"
    assert snapshot.git_commit_sha == CLEAN_COMMIT
    assert snapshot.git_dirty is False
    snapshot.preflight(for_execution=False)

    for forbidden in (
        "RunStore",
        "apply_runtime_policy",
        "resolve_device",
        "capture_system_provenance",
        "evaluator",
        "oracle",
        "search",
    ):
        assert not hasattr(readiness, forbidden)
    assert not (parent / "raw-authority").exists()


@pytest.mark.parametrize("same_bytes", [False, True])
def test_source_byte_or_same_byte_inode_replacement_is_rejected(
    authority_repository: Path,
    tmp_path: Path,
    clean_git: None,
    same_bytes: bool,
) -> None:
    snapshot = _capture(authority_repository, tmp_path / "raw-parent")
    target = authority_repository / readiness.SOURCE_RELATIVE_PATHS[0]
    replacement = target.with_name("replacement.json")
    replacement.write_bytes(target.read_bytes() if same_bytes else b"{}")
    os.replace(replacement, target)

    with pytest.raises(readiness.LocalAffordanceReadinessError, match="source drifted"):
        snapshot.recheck()


def test_repository_and_destination_parent_symlinks_are_rejected(
    authority_repository: Path,
    tmp_path: Path,
    clean_git: None,
) -> None:
    repository_link = tmp_path / "repository-link"
    repository_link.symlink_to(authority_repository, target_is_directory=True)
    parent = tmp_path / "raw-parent"
    parent.mkdir()
    with pytest.raises(readiness.LocalAffordanceReadinessError, match="repository"):
        readiness.capture_local_affordance_readiness(
            repository_link,
            raw_publication_destination=parent / "raw-authority",
        )

    actual_parent = tmp_path / "actual-parent"
    actual_parent.mkdir()
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(actual_parent, target_is_directory=True)
    with pytest.raises(readiness.LocalAffordanceReadinessError, match="parent"):
        readiness.capture_local_affordance_readiness(
            authority_repository,
            raw_publication_destination=parent_link / "raw-authority",
        )


def test_destination_parent_replacement_is_rejected(
    authority_repository: Path,
    tmp_path: Path,
    clean_git: None,
) -> None:
    parent = tmp_path / "raw-parent"
    snapshot = _capture(authority_repository, parent)
    parent.rename(tmp_path / "old-parent")
    parent.mkdir()

    with pytest.raises(readiness.LocalAffordanceReadinessError, match="parent identity"):
        snapshot.recheck()


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_destination_appearance_after_capture_is_rejected(
    authority_repository: Path,
    tmp_path: Path,
    clean_git: None,
    kind: str,
) -> None:
    parent = tmp_path / "raw-parent"
    snapshot = _capture(authority_repository, parent)
    destination = parent / "raw-authority"
    if kind == "file":
        destination.write_bytes(b"occupied")
    elif kind == "directory":
        destination.mkdir()
    else:
        target = parent / "target"
        target.mkdir()
        destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(readiness.LocalAffordanceReadinessError, match="already exists"):
        snapshot.recheck()


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_existing_destination_is_rejected_during_capture(
    authority_repository: Path,
    tmp_path: Path,
    clean_git: None,
    kind: str,
) -> None:
    parent = tmp_path / "raw-parent"
    parent.mkdir()
    destination = parent / "raw-authority"
    if kind == "file":
        destination.write_bytes(b"occupied")
    elif kind == "directory":
        destination.mkdir()
    else:
        target = parent / "target"
        target.mkdir()
        destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(readiness.LocalAffordanceReadinessError, match="absent"):
        readiness.capture_local_affordance_readiness(
            authority_repository,
            raw_publication_destination=destination,
        )


def test_execution_preflight_requires_clean_authorized_unchanged_commit(
    authority_repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(readiness, "_git_state", lambda _repository: (CLEAN_COMMIT, False))
    snapshot = _capture(authority_repository, tmp_path / "raw-parent")
    snapshot.preflight(for_execution=True, expected_git_commit=CLEAN_COMMIT)

    with pytest.raises(readiness.LocalAffordanceReadinessError, match="not authorized"):
        snapshot.preflight(for_execution=True, expected_git_commit="b" * 40)

    monkeypatch.setattr(readiness, "_git_state", lambda _repository: (CLEAN_COMMIT, True))
    with pytest.raises(readiness.LocalAffordanceReadinessError, match="git state changed"):
        snapshot.preflight(for_execution=True, expected_git_commit=CLEAN_COMMIT)


def test_capture_for_execution_rejects_dirty_repository(
    authority_repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(readiness, "_git_state", lambda _repository: (CLEAN_COMMIT, True))
    parent = tmp_path / "raw-parent"
    parent.mkdir()
    with pytest.raises(readiness.LocalAffordanceReadinessError, match="clean repository"):
        readiness.capture_local_affordance_readiness(
            authority_repository,
            raw_publication_destination=(parent / "raw-authority"),
            for_execution=True,
            expected_git_commit=CLEAN_COMMIT,
        )


def test_snapshot_rebound_and_direct_construction_fail_closed(
    authority_repository: Path,
    tmp_path: Path,
    clean_git: None,
) -> None:
    snapshot = _capture(authority_repository, tmp_path / "raw-parent")
    rebound = copy.copy(snapshot)
    object.__setattr__(rebound, "destination_name", "forged")
    with pytest.raises(readiness.LocalAffordanceReadinessError, match="forged or rebound"):
        rebound.require_sealed()
    with pytest.raises((readiness.LocalAffordanceReadinessError, TypeError)):
        readiness.LocalAffordanceReadinessSnapshot()


def test_activation_holds_exact_descriptors_and_expires_lease(
    authority_repository: Path,
    tmp_path: Path,
    clean_git: None,
) -> None:
    snapshot = _capture(authority_repository, tmp_path / "raw-parent")
    with snapshot.activation(expected_git_commit=CLEAN_COMMIT) as lease:
        assert lease.require_active() is lease
        assert require_expected_raw_probe_authority(lease.authority) is lease.authority
        repository_root_fd = lease.repository_root_fd
        parent_fd = lease.destination_parent_fd
        source_fds = tuple(lease._source_fds.values())
        os.fstat(parent_fd)
        for fd in source_fds:
            os.fstat(fd)

        rebound = copy.copy(lease)
        object.__setattr__(rebound, "destination_name", "forged")
        with pytest.raises(readiness.LocalAffordanceReadinessError, match="expired or forged"):
            rebound.require_active()

    with pytest.raises(readiness.LocalAffordanceReadinessError, match="expired or forged"):
        lease.require_active()
    for fd in (repository_root_fd, parent_fd, *source_fds):
        with pytest.raises(OSError):
            os.fstat(fd)
    with pytest.raises((readiness.LocalAffordanceReadinessError, TypeError)):
        readiness.LocalAffordanceActivationLease()


def test_activation_rechecks_git_immediately_before_yield(
    authority_repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(readiness, "_git_state", lambda _repository: (CLEAN_COMMIT, False))
    snapshot = _capture(authority_repository, tmp_path / "raw-parent")
    calls = 0

    def changed_during_activation(_repository: Path) -> tuple[str, bool]:
        nonlocal calls
        calls += 1
        return (CLEAN_COMMIT, calls > 1)

    monkeypatch.setattr(readiness, "_git_state", changed_during_activation)
    with pytest.raises(readiness.LocalAffordanceReadinessError, match="authorization changed"):
        with snapshot.activation(expected_git_commit=CLEAN_COMMIT):
            pytest.fail("activation must not yield after git drift")


def test_active_lease_rechecks_held_source_content(
    authority_repository: Path,
    tmp_path: Path,
    clean_git: None,
) -> None:
    snapshot = _capture(authority_repository, tmp_path / "raw-parent")
    with snapshot.activation(expected_git_commit=CLEAN_COMMIT) as lease:
        target = authority_repository / readiness.SOURCE_RELATIVE_PATHS[0]
        target.write_bytes(b"{}")
        with pytest.raises(readiness.LocalAffordanceReadinessError, match="expired or forged"):
            lease.require_active()
