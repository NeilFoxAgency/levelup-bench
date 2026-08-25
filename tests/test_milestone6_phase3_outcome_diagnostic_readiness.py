"""Adversarial tests for the outcome-diagnostic readiness boundary."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from levelup.experiments import milestone6_phase3_outcome_diagnostic_readiness as readiness
from levelup.experiments import milestone6_phase3_readiness as phase3

REPOSITORY = Path(readiness.ROOT)
OUTPUT_ROOT = REPOSITORY / readiness.DIAGNOSTIC_OUTPUT_ROOT_RELATIVE


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
