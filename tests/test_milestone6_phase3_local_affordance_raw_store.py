from __future__ import annotations

import hashlib
import os

import pytest
from pydantic import ValidationError

import levelup.experiments.milestone6_phase3_local_affordance_raw_store as raw_store
from levelup.experiments.milestone6_phase3_local_affordance_evidence import (
    RawProbeArtifactKey,
)
from levelup.experiments.milestone6_phase3_local_affordance_raw_store import (
    ARTIFACTS_DIR,
    HELDOUT_BINDINGS_DIR,
    KEYS_DIR,
    TRAINING_FOLDS_DIR,
    HeldoutProbeBinding,
    PinnedRawProbeStoreReader,
    RawProbeStoreError,
    RawProbeStoreManifest,
    RawProbeTaskKeyIndex,
    RawProbeTaskReference,
    StableFileSnapshot,
    TrainingFoldManifest,
    _read_stable_canonical_json_at,
    _stable_file_snapshot_at,
    open_existing_raw_probe_store,
)
from levelup.experiments.runner.config import canonical_json_bytes

HASHES = {
    "local_affordance_protocol_sha256": "1" * 64,
    "development_protocol_sha256": "2" * 64,
    "development_tasks_sha256": "3" * 64,
    "phase3_evidence_lock_sha256": "4" * 64,
    "probe_policy_sha256": "5" * 64,
}


def _key(family: str, replicate: int, task_index: int) -> RawProbeArtifactKey:
    return RawProbeArtifactKey(
        **HASHES,
        family_id=family,
        replicate=replicate,
        task_index=task_index,
        task_id=f"{family}-task-{task_index}",
        generator_seed=10 + task_index,
        probe_seed=20 + task_index,
        environment_seed=30 + task_index,
    )


def _reference(family: str, replicate: int, task_index: int) -> RawProbeTaskReference:
    key = _key(family, replicate, task_index)
    return RawProbeTaskReference(
        artifact_id=hashlib.sha256(f"artifact:{key.key_id}".encode()).hexdigest(),
        key_id=key.key_id,
        key=key,
    )


def test_manifest_is_self_hashed_and_has_exact_counts() -> None:
    manifest = RawProbeStoreManifest.from_authority_hashes(**HASHES)
    assert manifest.raw_artifact_count == 240
    assert manifest.training_fold_count == 30
    assert manifest.heldout_binding_count == 240
    assert manifest.scope == "known-development-only"
    assert manifest.execution_authorized is False
    assert manifest.manifest_id == manifest.expected_manifest_id
    with pytest.raises(ValidationError):
        RawProbeStoreManifest.model_validate(
            {**manifest.model_dump(mode="json"), "raw_artifact_count": 239}
        )
    with pytest.raises(ValidationError):
        RawProbeStoreManifest.model_validate(
            {**manifest.model_dump(mode="json"), "raw_artifact_count": 240.0}
        )
    with pytest.raises(ValidationError):
        RawProbeStoreManifest.model_validate(
            {**manifest.model_dump(mode="json"), "execution_authorized": 0}
        )


def test_strict_key_identity_and_actual_manifest_task_index() -> None:
    key = _key("momentum", 0, 126)
    assert key.task_index == 126
    reference = _reference("momentum", 0, 126)
    index = RawProbeTaskKeyIndex(key_id=key.key_id, artifact_id=reference.artifact_id, key=key)
    assert index.key.task_index == 126
    with pytest.raises(ValidationError):
        RawProbeTaskKeyIndex(
            key_id=key.key_id,
            artifact_id=reference.artifact_id,
            key=_key("momentum", 0, 127),
        )
    with pytest.raises(ValidationError):
        RawProbeTaskKeyIndex(
            key_id=key.key_id,
            artifact_id=reference.artifact_id,
            key=key,
            bad=True,
        )


def test_training_fold_is_exact_40_reference_lofo_factorization() -> None:
    refs = tuple(
        _reference(family, 2, i)
        for family in ("battery", "cooldown", "heat", "momentum", "combo")
        for i in range(8)
    )
    fold = TrainingFoldManifest(
        fold_id="plain",
        heldout_family="plain",
        replicate=2,
        task_references=refs,
    )
    assert len(fold.task_references) == 40
    assert fold.task_references[-1].task_index == 7
    with pytest.raises(ValidationError):
        TrainingFoldManifest(
            fold_id="plain",
            heldout_family="plain",
            replicate=2,
            task_references=refs[:-1],
        )
    with pytest.raises(ValidationError, match="canonical order"):
        TrainingFoldManifest(
            fold_id="plain",
            heldout_family="plain",
            replicate=2,
            task_references=tuple(reversed(refs)),
        )


def test_heldout_binding_requires_fold_family_equality() -> None:
    ref = _reference("combo", 4, 126)
    binding = HeldoutProbeBinding(
        fold_id="combo", family_id="combo", replicate=4, task_reference=ref
    )
    assert binding.reference.task_index == 126
    with pytest.raises(ValidationError):
        HeldoutProbeBinding(
            fold_id="plain", family_id="combo", replicate=4, task_reference=ref
        )


def test_stable_canonical_read_rejects_noncanonical_and_symlink(tmp_path) -> None:
    payload = {"a": 1, "b": [2, 3]}
    canonical = canonical_json_bytes(payload) + b"\n"
    (tmp_path / "ok.json").write_bytes(canonical)
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        snapshot = _stable_file_snapshot_at(directory_fd, "ok.json")
        assert isinstance(snapshot, StableFileSnapshot)
        assert snapshot.canonical_bytes == canonical
        assert _read_stable_canonical_json_at(directory_fd, "ok.json") == canonical
    finally:
        os.close(directory_fd)

    (tmp_path / "pretty.json").write_text('{"b": [2, 3], "a": 1}\n', encoding="utf-8")
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(RawProbeStoreError, match="canonical"):
            _read_stable_canonical_json_at(directory_fd, "pretty.json")
    finally:
        os.close(directory_fd)

    (tmp_path / "target.json").write_bytes(canonical)
    (tmp_path / "link.json").symlink_to("target.json")
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(RawProbeStoreError):
            _read_stable_canonical_json_at(directory_fd, "link.json")
    finally:
        os.close(directory_fd)


def test_stable_canonical_read_detects_path_replacement(tmp_path, monkeypatch) -> None:
    live = tmp_path / "live.json"
    replacement = tmp_path / "replacement.json"
    live.write_bytes(canonical_json_bytes({"value": "first"}) + b"\n")
    replacement.write_bytes(canonical_json_bytes({"value": "other"}) + b"\n")
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    original_read = raw_store.os.read
    replaced = False

    def replacing_read(file_fd: int, amount: int) -> bytes:
        nonlocal replaced
        chunk = original_read(file_fd, amount)
        if chunk and not replaced:
            replaced = True
            os.replace(replacement, live)
        return chunk

    monkeypatch.setattr(raw_store.os, "read", replacing_read)
    try:
        with pytest.raises(RawProbeStoreError, match="changed"):
            _stable_file_snapshot_at(directory_fd, "live.json")
    finally:
        os.close(directory_fd)


def test_pinned_reader_detects_directory_substitution_and_has_no_lookup_api(tmp_path) -> None:
    for name in (ARTIFACTS_DIR, KEYS_DIR, TRAINING_FOLDS_DIR, HELDOUT_BINDINGS_DIR):
        (tmp_path / name).mkdir()
    with open_existing_raw_probe_store(tmp_path) as reader:
        assert isinstance(reader, PinnedRawProbeStoreReader)
        assert not hasattr(reader, "lookup")
        assert not hasattr(reader, "list_artifacts")
        assert not hasattr(reader, "enumerate")
        replacement = tmp_path.parent / "replacement"
        replacement.mkdir()
        for name in (ARTIFACTS_DIR, KEYS_DIR, TRAINING_FOLDS_DIR, HELDOUT_BINDINGS_DIR):
            (replacement / name).mkdir()
        moved = tmp_path.parent / "old-root"
        os.rename(tmp_path, moved)
        os.rename(replacement, tmp_path)
        with pytest.raises(RawProbeStoreError, match="replaced"):
            reader.recheck()
