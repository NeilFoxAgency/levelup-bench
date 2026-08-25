from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import levelup.experiments.milestone6_phase3_outcome_diagnostic_model_store as store


def _sha(value: object) -> str:
    return hashlib.sha256(store.canonical_json_bytes(value)).hexdigest()


def test_empty_store_is_descriptor_pinned_and_manifest_is_stable(tmp_path: Path):
    root = tmp_path / "models"
    with store.open_outcome_model_store(root) as pinned:
        assert pinned.reader.root_path == root
        manifest = store._load_manifest(pinned.reader)
        assert store.load_outcome_model_manifest_at(pinned.reader) == manifest
        assert manifest.entries == ()
        assert manifest.manifest_id == _sha({"schema_version": store.SCHEMA_VERSION, "entries": []})
        pinned.recheck()


def test_pinned_store_authority_cannot_be_publicly_forged_or_rebound(tmp_path: Path):
    root = tmp_path / "models"
    with store.open_outcome_model_store(root) as pinned:
        reader = pinned.reader
        with pytest.raises(store.OutcomeModelStoreError, match="canonical"):
            store.PinnedOutcomeModelStoreReader(
                reader.root_fd,
                reader.records_fd,
                reader.states_fd,
                reader.staging_fd,
                reader.root_path,
                reader.identities,
            )
        with pytest.raises(store.OutcomeModelStoreError, match="canonical"):
            store.PinnedOutcomeModelStore(reader)
        rebound = store.PinnedOutcomeModelStoreReader(
            reader.root_fd,
            reader.states_fd,
            reader.records_fd,
            reader.staging_fd,
            reader.root_path,
            reader.identities,
            _token=store._STORE_TOKEN,
        )
        with pytest.raises(store.OutcomeModelStoreError, match="held"):
            rebound.recheck()


def test_symlinked_store_root_is_rejected(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "models"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(store.OutcomeModelStoreError, match="symlink"):
        with store.open_outcome_model_store(link):
            pass


def test_manifest_symlink_and_noncanonical_bytes_fail_closed(tmp_path: Path):
    root = tmp_path / "models"
    with store.open_outcome_model_store(root) as pinned:
        reader = pinned.reader
        target = tmp_path / "manifest-target"
        target.write_bytes(b"{}")
        os.symlink(target, root / store.MANIFEST_NAME)
        with pytest.raises(store.OutcomeModelStoreError):
            store._load_manifest(reader)
        os.unlink(root / store.MANIFEST_NAME)
        (root / store.MANIFEST_NAME).write_bytes(b"{}\n")
        with pytest.raises(store.OutcomeModelStoreError):
            store._load_manifest(reader)


def test_state_index_self_hash_and_tensor_inventory_are_strict():
    body = {
        "schema_version": store.SCHEMA_VERSION,
        "index_id": "0" * 64,
        "owner_id": "1" * 64,
        "record_id": "2" * 64,
        "model_state_sha256": "3" * 64,
        "tensors": [
            {
                "name": "network.0.bias",
                "filename": "0000.bin",
                "shape": [48],
                "byte_length": 192,
                "sha256": "4" * 64,
            }
        ],
    }
    body["index_id"] = _sha({key: value for key, value in body.items() if key != "index_id"})
    index = store.OutcomeModelStateIndex.model_validate(body)
    assert index.expected_index_id == index.index_id
    with pytest.raises(ValueError, match="self-hash"):
        store.OutcomeModelStateIndex.model_validate({**body, "index_id": "5" * 64})
    with pytest.raises(ValueError, match="filenames"):
        store.OutcomeModelStateIndex.model_validate(
            {
                **body,
                "index_id": _sha(
                    {
                        **{key: value for key, value in body.items() if key != "index_id"},
                        "tensors": [{**body["tensors"][0], "filename": "state.bin"}],
                    }
                ),
                "tensors": [{**body["tensors"][0], "filename": "state.bin"}],
            }
        )


def test_inventory_rejects_partial_foreign_owner_universe(tmp_path: Path):
    root = tmp_path / "models"
    with store.open_outcome_model_store(root) as pinned:
        reader = pinned.reader
        body = {
            "schema_version": store.SCHEMA_VERSION,
            "manifest_id": "0" * 64,
            "entries": [],
        }
        body["manifest_id"] = _sha(
            {key: value for key, value in body.items() if key != "manifest_id"}
        )
        store._write_new(
            reader.root_fd,
            store.MANIFEST_NAME,
            store._canonical(store.OutcomeModelStoreManifest.model_validate(body)),
            reader.staging_fd,
        )
        with pytest.raises(store.OutcomeModelStoreError, match="240-owner"):
            store.scan_outcome_model_inventory_at(
                reader,
                ("1" * 64,),
                {},
                None,
                None,
                preparation_git_commit_sha="c" * 40,
                preparation_provenance_sha256="d" * 64,
            )


def test_record_json_substitution_is_not_accepted_as_an_inventory_record(tmp_path: Path):
    root = tmp_path / "models"
    with store.open_outcome_model_store(root) as pinned:
        reader = pinned.reader
        # A foreign record-shaped JSON file is not enough to create a valid
        # inventory entry; semantic record parsing remains mandatory.
        owner = "1" * 64
        (root / store.RECORDS_DIR / f"{owner}.json").write_text(json.dumps({"owner_id": owner}))
        with pytest.raises(store.OutcomeModelStoreError):
            store.load_outcome_model_artifact_at(
                reader,
                owner,
                None,
                None,
                None,
                "c" * 40,
                "d" * 64,
            )  # type: ignore[arg-type]


def test_state_index_requires_canonical_bytes_and_directory_owner(tmp_path: Path):
    owner = "1" * 64
    root = tmp_path / "models"
    with store.open_outcome_model_store(root) as pinned:
        reader = pinned.reader
        os.mkdir(owner, dir_fd=reader.states_fd)
        state_fd = os.open(owner, os.O_RDONLY | os.O_DIRECTORY, dir_fd=reader.states_fd)
        try:
            os.mkdir(store.TENSORS_DIR, dir_fd=state_fd)
            body = {
                "schema_version": store.SCHEMA_VERSION,
                "index_id": "0" * 64,
                "owner_id": owner,
                "record_id": "2" * 64,
                "model_state_sha256": "3" * 64,
                "tensors": [],
            }
            body["index_id"] = _sha(
                {key: value for key, value in body.items() if key != "index_id"}
            )
            # Valid JSON with spaces is deliberately not the canonical form.
            fd = os.open("state.json", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=state_fd)
            try:
                os.write(fd, (json.dumps(body, indent=2) + "\n").encode())
            finally:
                os.close(fd)
        finally:
            os.close(state_fd)
        with pytest.raises(store.OutcomeModelStoreError, match="index"):
            store._state_index_and_payload(reader, owner)


def test_state_index_cannot_be_rebound_to_a_different_owner(tmp_path: Path):
    owner = "1" * 64
    other = "2" * 64
    root = tmp_path / "models"
    with store.open_outcome_model_store(root) as pinned:
        reader = pinned.reader
        os.mkdir(owner, dir_fd=reader.states_fd)
        state_fd = os.open(owner, os.O_RDONLY | os.O_DIRECTORY, dir_fd=reader.states_fd)
        try:
            os.mkdir(store.TENSORS_DIR, dir_fd=state_fd)
            body = {
                "schema_version": store.SCHEMA_VERSION,
                "index_id": "0" * 64,
                "owner_id": other,
                "record_id": "2" * 64,
                "model_state_sha256": "3" * 64,
                "tensors": [],
            }
            body["index_id"] = _sha(
                {key: value for key, value in body.items() if key != "index_id"}
            )
            fd = os.open("state.json", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=state_fd)
            try:
                os.write(fd, store.canonical_json_bytes(body) + b"\n")
            finally:
                os.close(fd)
        finally:
            os.close(state_fd)
        with pytest.raises(store.OutcomeModelStoreError, match="owner"):
            store._state_index_and_payload(reader, owner)


def test_record_claim_never_overwrites_a_competing_record(tmp_path: Path):
    with store.open_outcome_model_store(tmp_path / "models") as pinned:
        reader = pinned.reader
        store._claim_or_match(reader.records_fd, "owner.json", b"first", reader.staging_fd)
        with pytest.raises(store.OutcomeModelStoreError, match="different model record"):
            store._claim_or_match(reader.records_fd, "owner.json", b"second", reader.staging_fd)


def _synthetic_manifest(
    count: int = 240,
) -> tuple[store.OutcomeModelStoreManifest, tuple[str, ...]]:
    owners = tuple(f"{index:064x}" for index in range(count))
    entries = tuple(
        store.OutcomeModelStoreEntry(
            owner_id=owner,
            record_id="1" * 64,
            key_id="2" * 64,
            record_sha256="3" * 64,
            state_index_id="4" * 64,
            model_state_sha256="5" * 64,
        )
        for owner in owners
    )
    body = {
        "schema_version": store.SCHEMA_VERSION,
        "manifest_id": "0" * 64,
        "entries": [item.model_dump(mode="json") for item in entries],
    }
    body["manifest_id"] = _sha({key: value for key, value in body.items() if key != "manifest_id"})
    return store.OutcomeModelStoreManifest.model_validate(body), owners


def test_inventory_rejects_extra_records_and_state_directories(tmp_path: Path):
    manifest, owners = _synthetic_manifest()
    root = tmp_path / "models"
    with store.open_outcome_model_store(root) as pinned:
        reader = pinned.reader
        store._write_new(
            reader.root_fd,
            store.MANIFEST_NAME,
            store._canonical(manifest),
            reader.staging_fd,
        )
        for owner in owners:
            (root / store.RECORDS_DIR / f"{owner}.json").write_bytes(b"foreign")
            os.mkdir(owner, dir_fd=reader.states_fd)
        (root / store.RECORDS_DIR / f"{'f' * 64}.json").write_bytes(b"extra")
        with pytest.raises(store.OutcomeModelStoreError, match="record inventory"):
            store.scan_outcome_model_inventory_at(
                reader,
                owners,
                {},
                None,
                None,
                preparation_git_commit_sha="c" * 40,
                preparation_provenance_sha256="d" * 64,
            )

    root = tmp_path / "models-state-extra"
    with store.open_outcome_model_store(root) as pinned:
        reader = pinned.reader
        store._write_new(
            reader.root_fd,
            store.MANIFEST_NAME,
            store._canonical(manifest),
            reader.staging_fd,
        )
        for owner in owners:
            (root / store.RECORDS_DIR / f"{owner}.json").write_bytes(b"foreign")
            os.mkdir(owner, dir_fd=reader.states_fd)
        os.mkdir("extra", dir_fd=reader.states_fd)
        with pytest.raises(store.OutcomeModelStoreError, match="state inventory"):
            store.scan_outcome_model_inventory_at(
                reader,
                owners,
                {},
                None,
                None,
                preparation_git_commit_sha="c" * 40,
                preparation_provenance_sha256="d" * 64,
            )


def test_inventory_rejects_structurally_complete_store_without_semantic_authority(
    tmp_path: Path,
):
    manifest, owners = _synthetic_manifest()
    root = tmp_path / "models"
    with store.open_outcome_model_store(root) as pinned:
        reader = pinned.reader
        store._write_new(
            reader.root_fd,
            store.MANIFEST_NAME,
            store._canonical(manifest),
            reader.staging_fd,
        )
        for owner in owners:
            (root / store.RECORDS_DIR / f"{owner}.json").write_bytes(b"foreign")
            os.mkdir(owner, dir_fd=reader.states_fd)
        with pytest.raises(store.OutcomeModelStoreError, match="semantic inventory authority"):
            store.scan_outcome_model_inventory_at(
                reader,
                owners,
                {},
                None,
                None,
                preparation_git_commit_sha="c" * 40,
                preparation_provenance_sha256="d" * 64,
            )


def _materialize_identity_snapshot_store(tmp_path: Path):
    manifest, owners = _synthetic_manifest()
    root = tmp_path / "identity-models"
    with store.open_outcome_model_store(root) as pinned:
        reader = pinned.reader
        store._write_new(
            reader.root_fd,
            store.MANIFEST_NAME,
            store._canonical(manifest),
            reader.staging_fd,
        )
        for metadata_name in store.ROOT_METADATA_FILES:
            store._write_new(reader.root_fd, metadata_name, b"{}\n", reader.staging_fd)
        for owner in owners:
            store._write_new(reader.records_fd, f"{owner}.json", b"same bytes\n", reader.staging_fd)
            os.mkdir(owner, dir_fd=reader.states_fd)
            state_fd = os.open(owner, os.O_RDONLY | os.O_DIRECTORY, dir_fd=reader.states_fd)
            try:
                os.mkdir(store.TENSORS_DIR, dir_fd=state_fd)
                body = {
                    "schema_version": store.SCHEMA_VERSION,
                    "index_id": "0" * 64,
                    "owner_id": owner,
                    "record_id": "1" * 64,
                    "model_state_sha256": "2" * 64,
                    "tensors": [],
                }
                body["index_id"] = _sha(
                    {key: value for key, value in body.items() if key != "index_id"}
                )
                store._write_new(
                    state_fd,
                    store.STATE_MANIFEST_NAME,
                    store.canonical_json_bytes(body) + b"\n",
                    reader.staging_fd,
                )
            finally:
                os.close(state_fd)
        first = store.snapshot_outcome_model_store_identities_at(reader, owners)
    return root, owners, first


def test_identity_snapshot_captures_complete_store_and_rejects_same_byte_record_replacement(
    tmp_path: Path,
):
    root, owners, first = _materialize_identity_snapshot_store(tmp_path)
    assert first.root_metadata_file_identities == tuple(
        (name, first.root_metadata_file_identities[index][1])
        for index, name in enumerate(store.ROOT_METADATA_FILES)
    )
    assert len(first.record_file_identities) == 240
    assert len(first.state_identities) == 240
    assert all(not item.tensor_file_identities for item in first.state_identities)
    with store.open_outcome_model_store(root) as pinned:
        record_path = root / store.RECORDS_DIR / f"{owners[0]}.json"
        replacement = tmp_path / "same-record.json"
        replacement.write_bytes(record_path.read_bytes())
        os.replace(replacement, record_path)
        second = store.snapshot_outcome_model_store_identities_at(pinned.reader, owners)
        assert second != first


def test_identity_snapshot_rejects_same_byte_owner_directory_replacement(tmp_path: Path):
    root, owners, _first = _materialize_identity_snapshot_store(tmp_path)
    owner = owners[0]
    with store.open_outcome_model_store(root) as pinned:
        reader = pinned.reader
        replacement = tmp_path / "replaced-owner"
        os.rename(owner, replacement, src_dir_fd=reader.states_fd)
        os.mkdir(owner, dir_fd=reader.states_fd)
        replacement_fd = os.open(replacement, os.O_RDONLY | os.O_DIRECTORY)
        owner_fd = os.open(owner, os.O_RDONLY | os.O_DIRECTORY, dir_fd=reader.states_fd)
        try:
            os.mkdir(store.TENSORS_DIR, dir_fd=owner_fd)
            state_bytes = (replacement / store.STATE_MANIFEST_NAME).read_bytes()
            store._write_new(owner_fd, store.STATE_MANIFEST_NAME, state_bytes, reader.staging_fd)
        finally:
            os.close(replacement_fd)
            os.close(owner_fd)
        os.rename(replacement, tmp_path / "replaced-owner-backup")
        second = store.snapshot_outcome_model_store_identities_at(reader, owners)
        assert second != _first
