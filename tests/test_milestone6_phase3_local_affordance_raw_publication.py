"""Atomic publication tests for the Phase 3 raw-evidence authority.

These tests exercise only the synthetic known-development artifact fixture.  No
comparative outcomes, final families, or learner-facing capabilities are used.
"""

from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path

import pytest

from levelup.experiments import milestone6_phase3_local_affordance_readiness as readiness

authority = pytest.importorskip(
    "levelup.experiments.milestone6_phase3_local_affordance_raw_authority"
)
publication = pytest.importorskip(
    "levelup.experiments.milestone6_phase3_local_affordance_raw_publication"
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "milestone6"


def _artifact_rows(key):
    # Reuse the complete synthetic task/artifact construction, without
    # importing any result or final-family data during module collection.
    from test_milestone6_phase3_local_affordance_raw_authority import _artifact

    return _artifact(key)


def _expected():
    return authority.build_expected_raw_probe_authority(
        local_affordance_protocol_bytes=(
            CONFIG / "phase3_local_affordance_protocol.json"
        ).read_bytes(),
        development_protocol_bytes=(CONFIG / "development_protocol.json").read_bytes(),
        development_tasks_bytes=(CONFIG / "development_tasks.json").read_bytes(),
        phase3_evidence_lock_bytes=(CONFIG / "phase3_evidence_lock.json").read_bytes(),
    )


@pytest.fixture(scope="session")
def expected_authority():
    return _expected()


@pytest.fixture(scope="session")
def artifacts(expected_authority):
    return tuple(
        authority.PersistedRawProbeArtifact.model_validate(_artifact_rows(key)[0])
        for key in expected_authority.keys
    )


def _publish(destination: Path, expected_authority, artifacts):
    return publication.publish_raw_probe_store(
        destination,
        expected=expected_authority,
        artifacts=artifacts,
    )


def _assert_one_empty_failed_stage_skeleton(parent: Path) -> None:
    stages = tuple(path for path in parent.iterdir() if ".staging-" in path.name.lower())
    assert len(stages) == 1
    assert stages[0].is_dir()
    assert sorted(path.name for path in stages[0].iterdir()) == [
        "artifacts",
        "heldout-bindings",
        "keys",
        "training-folds",
    ]
    assert not any(path.is_file() or path.is_symlink() for path in stages[0].rglob("*"))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("evidence_lock_file_sha256", "0" * 64),
        ("selected_tasks", ()),
        ("keys", ()),
        ("key_filenames", ("forged.json",) * 240),
    ],
)
def test_rebound_expected_authority_cannot_publish(
    tmp_path, expected_authority, artifacts, field, replacement
):
    rebound = copy.copy(expected_authority)
    object.__setattr__(rebound, field, replacement)
    with pytest.raises(publication.RawProbePublicationError, match="(?i)(frozen|authority)"):
        _publish(tmp_path / "raw-authority", rebound, artifacts)


def test_failed_stage_open_never_deletes_a_substituted_directory(monkeypatch, tmp_path):
    parent_fd = publication.secure_fs.open_directory_chain(tmp_path)
    replacement = tmp_path / ".raw-authority.staging-fixed"

    def substitute_then_fail(directory_fd, name):
        os.rmdir(name, dir_fd=directory_fd)
        os.mkdir(name, 0o700, dir_fd=directory_fd)
        raise publication.secure_fs.SecureFilesystemError("injected substitution")

    monkeypatch.setattr(publication.secrets, "token_hex", lambda _length: "fixed")
    monkeypatch.setattr(
        publication.secure_fs,
        "open_child_directory",
        substitute_then_fail,
    )
    try:
        with pytest.raises(publication.RawProbePublicationError):
            publication._allocate_staging(parent_fd, "raw-authority")
    finally:
        os.close(parent_fd)
    assert replacement.is_dir()


def test_exact_canonical_publication_returns_authority_snapshot(
    tmp_path, expected_authority, artifacts
):
    destination = tmp_path / "raw-authority"
    snapshot = _publish(destination, expected_authority, artifacts)

    assert isinstance(snapshot, authority.RawProbeAuthoritySnapshot)
    assert authority.require_raw_probe_authority_snapshot(snapshot) is snapshot
    assert snapshot.manifest == expected_authority.manifest
    assert len(snapshot.artifact_files) == 240
    assert len(snapshot.key_files) == 240
    assert len(snapshot.training_fold_files) == 30
    assert len(snapshot.heldout_binding_files) == 240
    assert len(snapshot.authority_content_sha256) == 64
    assert snapshot.manifest.execution_authorized is False
    assert sorted(path.name for path in destination.iterdir()) == [
        "artifacts",
        "heldout-bindings",
        "keys",
        "manifest.json",
        "training-folds",
    ]
    assert not any(".staging-" in path.name.lower() for path in tmp_path.iterdir())


def test_existing_identical_destination_is_refused_without_resume(
    tmp_path, expected_authority, artifacts
):
    destination = tmp_path / "raw-authority"
    _publish(destination, expected_authority, artifacts)
    before = {path: path.read_bytes() for path in destination.rglob("*") if path.is_file()}

    with pytest.raises(
        publication.RawProbePublicationError,
        match="(?i)(exist|publish|destination|resume)",
    ):
        _publish(destination, expected_authority, artifacts)

    after = {path: path.read_bytes() for path in destination.rglob("*") if path.is_file()}
    assert after == before


def test_existing_different_destination_is_refused_and_not_overwritten(
    tmp_path, expected_authority, artifacts
):
    destination = tmp_path / "raw-authority"
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_bytes(b"keep me")

    with pytest.raises(
        publication.RawProbePublicationError,
        match="(?i)(exist|publish|destination|resume)",
    ):
        _publish(destination, expected_authority, artifacts)
    assert sentinel.read_bytes() == b"keep me"
    assert tuple(destination.iterdir()) == (sentinel,)


def test_symlink_destination_is_refused(tmp_path, expected_authority, artifacts):
    target = tmp_path / "target"
    target.mkdir()
    destination = tmp_path / "raw-authority"
    destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(
        publication.RawProbePublicationError,
        match="(?i)(symlink|unsafe|destination|root)",
    ):
        _publish(destination, expected_authority, artifacts)
    assert tuple(target.iterdir()) == ()


def test_pre_activation_write_failure_leaves_no_final_and_only_empty_stage_root(
    monkeypatch, tmp_path, expected_authority, artifacts
):
    destination = tmp_path / "raw-authority"
    original = getattr(publication, "_write_staged_file", None)
    if original is None:
        pytest.fail("publication must expose _write_staged_file for failure injection")
    calls = 0

    def fail_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected pre-activation write failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(publication, "_write_staged_file", fail_after_first)
    with pytest.raises(
        publication.RawProbePublicationError,
        match="(?i)(write|stage|publish|injected|failed closed)",
    ):
        _publish(destination, expected_authority, artifacts)

    assert not destination.exists()
    _assert_one_empty_failed_stage_skeleton(tmp_path)


def test_destination_race_is_refused_and_staged_tree_is_cleaned(
    monkeypatch, tmp_path, expected_authority, artifacts
):
    destination = tmp_path / "raw-authority"
    original = getattr(publication, "_activate_staged_store", None)
    if original is None:
        pytest.fail("publication must expose _activate_staged_store for race injection")

    def race(parent_fd, source, target, *args, **kwargs):
        # Simulate a competing creator between the publisher's final absence
        # check and its no-replace activation syscall.
        import os

        os.mkdir(target, dir_fd=parent_fd)
        return original(parent_fd, source, target, *args, **kwargs)

    monkeypatch.setattr(publication, "_activate_staged_store", race)
    with pytest.raises(
        publication.RawProbePublicationError,
        match="(?i)(race|exist|destination|publish)",
    ):
        _publish(destination, expected_authority, artifacts)

    assert destination.is_dir()
    assert not (destination / "manifest.json").exists()
    _assert_one_empty_failed_stage_skeleton(tmp_path)


def test_active_readiness_lease_binds_exact_publication_parent(monkeypatch, tmp_path, artifacts):
    commit = "a" * 40
    monkeypatch.setattr(readiness, "_git_state", lambda _repository: (commit, False))
    parent = tmp_path / "raw-parent"
    parent.mkdir()
    destination = parent / "raw-authority"
    snapshot = readiness.capture_local_affordance_readiness(
        ROOT,
        raw_publication_destination=destination,
    )

    with snapshot.activation(expected_git_commit=commit) as lease:
        published = publication.publish_raw_probe_store_from_readiness(
            lease,
            artifacts=artifacts,
        )
        assert published.manifest == snapshot.authority.manifest
        assert published.manifest.execution_authorized is False
    assert destination.is_dir()

    with pytest.raises(publication.RawProbePublicationError, match="lease"):
        publication.publish_raw_probe_store_from_readiness(lease, artifacts=artifacts)


def test_readiness_publication_rejects_lexical_parent_substitution(
    monkeypatch, tmp_path, artifacts
):
    commit = "a" * 40
    monkeypatch.setattr(readiness, "_git_state", lambda _repository: (commit, False))
    parent = tmp_path / "raw-parent"
    parent.mkdir()
    snapshot = readiness.capture_local_affordance_readiness(
        ROOT,
        raw_publication_destination=parent / "raw-authority",
    )

    with snapshot.activation(expected_git_commit=commit) as lease:
        parent.rename(tmp_path / "old-parent")
        parent.mkdir()
        with pytest.raises(publication.RawProbePublicationError, match="parent identity"):
            publication.publish_raw_probe_store_from_readiness(lease, artifacts=artifacts)


def test_readiness_publication_rejects_same_byte_source_replacement(
    monkeypatch, tmp_path, artifacts
):
    commit = "a" * 40
    monkeypatch.setattr(readiness, "_git_state", lambda _repository: (commit, False))
    repository = tmp_path / "repository"
    for relative in readiness.SOURCE_RELATIVE_PATHS:
        source = ROOT / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    parent = tmp_path / "raw-parent"
    parent.mkdir()
    destination = parent / "raw-authority"
    snapshot = readiness.capture_local_affordance_readiness(
        repository,
        raw_publication_destination=destination,
    )

    with snapshot.activation(expected_git_commit=commit) as lease:
        source = repository / readiness.SOURCE_RELATIVE_PATHS[0]
        replacement = source.with_name("replacement.json")
        replacement.write_bytes(source.read_bytes())
        os.replace(replacement, source)
        with pytest.raises(publication.RawProbePublicationError, match="lease"):
            publication.publish_raw_probe_store_from_readiness(lease, artifacts=artifacts)
    assert not destination.exists()
