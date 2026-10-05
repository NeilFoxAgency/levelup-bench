"""Tests for payload-free revalidation of the frozen Phase 2 result inventory."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase2_result_snapshot_publication as publication
import levelup.experiments.milestone6_phase2_screening_runtime as runtime
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import TrainingDataArtifactError

FAMILIES = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
SELECTION_SHA256 = "a" * 64


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _artifact() -> bytes:
    folds = []
    runtime_snapshot = []
    for family_index, family in enumerate(FAMILIES):
        run_id = f"run-{family}"
        namespaces = []
        snapshot_namespaces = []
        for namespace_index, namespace in enumerate(("units", "attempts")):
            directory = {
                "device": 1,
                "inode": 100 + family_index * 2 + namespace_index,
                "ctime_ns": 11,
                "mtime_ns": 12,
                "file_type": stat.S_IFDIR,
            }
            file_rows = []
            snapshot_files = []
            if namespace == "units":
                for unit_index in range(1520):
                    name = f"{unit_index:064x}.json"
                    payload_sha = f"{unit_index + 1:064x}"
                    fstat = {
                        "device": 1,
                        "inode": 100_000 + family_index * 2000 + unit_index,
                        "ctime_ns": 21,
                        "mtime_ns": 22,
                        "size": 64,
                        "file_type": stat.S_IFREG,
                    }
                    file_rows.append(
                        {"name": name, "payload_sha256": payload_sha, "fstat": fstat}
                    )
                    snapshot_files.append([name, payload_sha])
            namespaces.append(
                {"name": namespace, "directory": directory, "files": file_rows}
            )
            snapshot_namespaces.append(
                [
                    namespace,
                    [
                        directory["device"],
                        directory["inode"],
                        directory["ctime_ns"],
                        directory["mtime_ns"],
                        directory["file_type"],
                        snapshot_files,
                    ],
                ]
            )
        folds.append(
            {
                "family_id": family,
                "run_id": run_id,
                "expected_units": 1520,
                "namespaces": namespaces,
            }
        )
        runtime_snapshot.append([run_id, snapshot_namespaces])
    result_digest = _sha(canonical_json_bytes(runtime_snapshot))
    body = {
        "schema_version": "milestone6.phase2.result-namespace-identity.v1",
        "purpose": "trusted_administrative_post_selection_identity_capture",
        "use_boundary": {
            "development_only": True,
            "final_family_access": False,
            "learner_input": False,
            "administrative_metadata_authority": True,
            "comparative_inspection_performed_here": False,
            "outcome_values_serialized": False,
        },
        "frozen_selection": {
            "selection_lock_sha256": SELECTION_SHA256,
            "selection_lock_parent_identity": [1, 2],
            "selection_lock_fstat": {
                "device": 1,
                "inode": 3,
                "ctime_ns": 4,
                "mtime_ns": 5,
                "size": 6,
                "file_type": stat.S_IFREG,
            },
            "frozen_result_namespace_snapshot_sha256": result_digest,
        },
        "readiness": {
            "manifest_relative_path": "experiments/milestone6_phase2_screening_readiness.json",
            "manifest_bytes_sha256": "b" * 64,
            "manifest_bytes_length": 100,
            "manifest_parent_identity": [1, 2],
            "manifest_file_identity": [1, 3, 4, 5, 100],
            "prepared_tree_sha256": "c" * 64,
        },
        "repositories": {},
        "development_matrix": {
            "family_order": list(FAMILIES),
            "family_count": 6,
            "units_per_fold": 1520,
            "total_units": 9120,
            "final_families": [],
        },
        "result_namespace_snapshot_sha256": result_digest,
        "runtime_result_namespace_snapshot": runtime_snapshot,
        "folds": folds,
    }
    document = body | {"identity_sha256": _sha(canonical_json_bytes(body))}
    return canonical_json_bytes(document) + b"\n"


def test_snapshot_artifact_is_complete_canonical_and_selection_bound():
    payload = _artifact()
    digest, folds = runtime._parse_result_snapshot_artifact(
        payload, selection_lock_sha256=SELECTION_SHA256
    )
    assert len(digest) == 64
    assert tuple(run_id for run_id, _namespaces in folds) == tuple(
        f"run-{family}" for family in FAMILIES
    )
    assert len(folds[0][1][0][2]) == 1520
    typed_snapshot = runtime._typed_result_namespace_snapshot(payload, digest)
    assert _sha(canonical_json_bytes(typed_snapshot)) == digest
    assert typed_snapshot[0][1][0][0] == "units"

    with pytest.raises(TrainingDataArtifactError, match="exact frozen selection"):
        runtime._parse_result_snapshot_artifact(
            payload, selection_lock_sha256="d" * 64
        )


def test_metadata_loader_retains_the_frozen_typed_snapshot_without_payload_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repository = tmp_path / "repository"
    committed = repository / runtime.CANONICAL_READINESS_PATH
    committed.parent.mkdir(parents=True)
    provenance = SimpleNamespace(git_commit_sha="0" * 40)

    class Manifest:
        family_order = FAMILIES
        child_run_ids = tuple(f"run-{family}" for family in FAMILIES)
        children = tuple(
            SimpleNamespace(run_id=f"run-{family}", heldout_family_id=family)
            for family in FAMILIES
        )
        expected_total_units = 9120
        final_family_access = False

        def model_dump(self, **_kwargs):
            return {"schema_version": "runtime-test", "family_order": list(FAMILIES)}

    manifest = Manifest()
    manifest.provenance = provenance
    manifest_bytes = canonical_json_bytes(manifest.model_dump()) + b"\n"
    committed.write_bytes(manifest_bytes)
    manifest_file_bytes, manifest_parent_identity, manifest_file_identity = (
        runtime._read_pinned_file(committed, label="test manifest")
    )
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    for run_id in manifest.child_run_ids:
        (raw_root / run_id / "units").mkdir(parents=True)
        (raw_root / run_id / "attempts").mkdir()
    (raw_root / "phase2-screening-readiness.json").write_bytes(manifest_bytes)

    artifact = json.loads(_artifact())
    artifact["readiness"].update(
        {
            "manifest_bytes_sha256": _sha(manifest_bytes),
            "manifest_bytes_length": len(manifest_bytes),
            "manifest_parent_identity": list(manifest_parent_identity),
            "manifest_file_identity": list(manifest_file_identity),
        }
    )
    unsigned = {key: value for key, value in artifact.items() if key != "identity_sha256"}
    artifact["identity_sha256"] = _sha(canonical_json_bytes(unsigned))
    artifact_bytes = canonical_json_bytes(artifact) + b"\n"
    expected_digest = artifact["result_namespace_snapshot_sha256"]

    source_snapshots = ()
    configs = tuple(
        SimpleNamespace(
            device_policy=SimpleNamespace(requested_device="cpu"),
            parameters={"heldout_family_id": family},
        )
        for family in FAMILIES
    )
    folds = list(
        SimpleNamespace(
            family_id=family,
            config=config,
            store=SimpleNamespace(run_id=f"run-{family}", _execution_ready=False),
            data_keys=object(),
            data=object(),
            model_keys=object(),
            models=object(),
            shared_plan=object(),
        )
        for family, config in zip(FAMILIES, configs, strict=True)
    )
    monkeypatch.setattr(runtime, "ROOT", repository)
    monkeypatch.setattr(runtime, "canonical_screening_repository", lambda path, **_kwargs: Path(path))
    monkeypatch.setattr(runtime, "_manifest_bytes", lambda *_args: (
        manifest_file_bytes,
        manifest,
        manifest_parent_identity,
        manifest_file_identity,
    ))
    monkeypatch.setattr(runtime, "_authority_sources", lambda _manifest: source_snapshots)
    monkeypatch.setattr(runtime, "_assert_authority_source_paths", lambda *_args: None)
    monkeypatch.setattr(runtime, "_assert_development_manifest", lambda *_args: None)
    monkeypatch.setattr(runtime, "_assert_tree_shape_at", lambda *_args: None)
    monkeypatch.setattr(runtime, "build_screening_plan", lambda: object())
    monkeypatch.setattr(runtime, "validate_screening_plan", lambda *_args: None)
    monkeypatch.setattr(runtime, "screening_child_configs", lambda: configs)
    monkeypatch.setattr(runtime, "apply_runtime_policy", lambda *_args: None)
    monkeypatch.setattr(runtime, "capture_system_provenance", lambda *_args: provenance)
    monkeypatch.setattr(runtime, "validate_screening_provenance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime, "_load_fold", lambda *_args: folds.pop(0))
    monkeypatch.setattr(runtime, "_assert_global_inventory", lambda *_args: None)
    monkeypatch.setattr(runtime, "_recheck_manifest_and_tree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime, "_recheck_metadata_selection_authority", lambda *_args: None)

    real_read_bytes_at = runtime.secure_fs.read_bytes_at

    def reject_result_reads(directory_fd, name):
        if name != "phase2-screening-readiness.json":
            raise AssertionError(f"unexpected payload read in metadata-only load: {name}")
        return real_read_bytes_at(directory_fd, name)

    monkeypatch.setattr(runtime.secure_fs, "read_bytes_at", reject_result_reads)
    loaded = runtime.load_screening_runtime_metadata_only(
        committed,
        raw_root,
        repository,
        manifest_bytes_sha256=_sha(manifest_bytes),
        result_snapshot_bytes=artifact_bytes,
        selection_lock_sha256=SELECTION_SHA256,
    )
    assert _sha(canonical_json_bytes(loaded.result_namespace_snapshot)) == expected_digest


def test_metadata_validator_never_reads_payloads_and_detects_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    run_dir = tmp_path / "run"
    units_dir = run_dir / "units"
    attempts_dir = run_dir / "attempts"
    units_dir.mkdir(parents=True)
    attempts_dir.mkdir()
    unit_id = "1" * 64
    result_file = units_dir / f"{unit_id}.json"
    result_file.write_bytes(b"opaque result payload")

    namespace_rows = []
    for name, path in (("units", units_dir), ("attempts", attempts_dir)):
        directory_stat = path.stat()
        directory_identity = (
            directory_stat.st_dev,
            directory_stat.st_ino,
            directory_stat.st_ctime_ns,
            directory_stat.st_mtime_ns,
            stat.S_IFMT(directory_stat.st_mode),
        )
        file_rows = []
        if name == "units":
            file_stat = result_file.stat()
            file_rows.append(
                (
                    result_file.name,
                    (
                        file_stat.st_dev,
                        file_stat.st_ino,
                        file_stat.st_ctime_ns,
                        file_stat.st_mtime_ns,
                        file_stat.st_size,
                        stat.S_IFMT(file_stat.st_mode),
                    ),
                )
            )
        namespace_rows.append((name, directory_identity, tuple(file_rows)))

    class Store:
        run_id = "test-run"
        expected = SimpleNamespace(units=(SimpleNamespace(unit_id=unit_id),))

        @contextmanager
        def _open_result_namespace(self, namespace):
            fd = os.open(run_dir / namespace, os.O_RDONLY | os.O_DIRECTORY)
            try:
                yield None, fd
            finally:
                os.close(fd)

        def completed_records(self):
            raise AssertionError("metadata-only path parsed completed records")

        def attempt_records(self):
            raise AssertionError("metadata-only path parsed attempt records")

    monkeypatch.setattr(
        runtime.secure_fs,
        "read_bytes_at",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("metadata-only path read a result payload")
        ),
    )
    fold = SimpleNamespace(store=Store())
    snapshot = (("test-run", tuple(namespace_rows)),)
    runtime._validate_result_namespace_metadata((fold,), snapshot)

    result_file.write_bytes(b"x" * result_file.stat().st_size)
    with pytest.raises(TrainingDataArtifactError, match="file identity changed"):
        runtime._validate_result_namespace_metadata((fold,), snapshot)

    extra_attempt = attempts_dir / f"{unit_id}.attempt-0001.json"
    extra_attempt.write_bytes(b"not parsed")
    with pytest.raises(TrainingDataArtifactError):
        runtime._validate_result_namespace_metadata((fold,), snapshot)
    extra_attempt.unlink()

    unsafe_link = attempts_dir / "unsafe.json"
    unsafe_link.symlink_to(result_file)
    with pytest.raises(TrainingDataArtifactError):
        runtime._validate_result_namespace_metadata((fold,), snapshot)
    unsafe_link.unlink()

    result_file.unlink()
    with pytest.raises(TrainingDataArtifactError):
        runtime._validate_result_namespace_metadata((fold,), snapshot)


def test_historical_default_validator_still_parses_strict_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    run_dir = tmp_path / "run"
    for namespace in ("units", "attempts"):
        (run_dir / namespace).mkdir(parents=True, exist_ok=True)
    (run_dir / "units" / f"{'2' * 64}.json").write_bytes(b"{}")
    calls: list[str] = []

    class Store:
        run_id = "test-run"
        _result_directory_identities = ("pinned",)
        expected = SimpleNamespace(units=(SimpleNamespace(unit_id="2" * 64),))

        @contextmanager
        def _open_result_namespace(self, namespace):
            fd = os.open(run_dir / namespace, os.O_RDONLY | os.O_DIRECTORY)
            try:
                yield None, fd
            finally:
                os.close(fd)

        def _capture_result_directory_identities(self):
            return self._result_directory_identities

        def completed_records(self):
            calls.append("completed")
            return ()

        def attempt_records(self):
            calls.append("attempts")
            return ()

    store = Store()
    store.run_dir = run_dir
    fold = SimpleNamespace(store=store)
    runtime._validate_result_namespaces((fold,))
    assert calls == ["completed", "attempts"]


def test_metadata_recheck_uses_all_folds_without_parsing_result_payloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repository = tmp_path / "repository"
    manifest_path = repository / runtime.CANONICAL_READINESS_PATH
    manifest_path.parent.mkdir(parents=True)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    provenance = SimpleNamespace(git_commit_sha="0" * 40)
    children = tuple(
        SimpleNamespace(run_id=f"run-{family}", family_id=family)
        for family in FAMILIES
    )

    class Manifest:
        family_order = FAMILIES
        child_run_ids = tuple(child.run_id for child in children)
        expected_total_units = 9120
        final_family_access = False
        development_only = True

        def __init__(self):
            self.children = children

        def model_dump(self, **_kwargs):
            return {"schema_version": "metadata-recheck-test", "family_order": list(FAMILIES)}

    manifest = Manifest()
    manifest.provenance = provenance
    manifest_bytes = canonical_json_bytes(manifest.model_dump()) + b"\n"
    manifest_path.write_bytes(manifest_bytes)
    raw_manifest = raw_root / "phase2-screening-readiness.json"
    raw_manifest.write_bytes(manifest_bytes)

    folds = []
    metadata_folds = []
    for index, family in enumerate(FAMILIES):
        run_id = f"run-{family}"
        run_dir = raw_root / run_id
        units_dir = run_dir / "units"
        attempts_dir = run_dir / "attempts"
        units_dir.mkdir(parents=True)
        attempts_dir.mkdir()
        unit_id = f"{index + 1:064x}"
        unit_file = units_dir / f"{unit_id}.json"
        unit_file.write_bytes(b"private outcome must not be opened")

        namespace_metadata = []
        for namespace, path in (("units", units_dir), ("attempts", attempts_dir)):
            directory_stat = path.stat()
            directory_identity = (
                directory_stat.st_dev,
                directory_stat.st_ino,
                directory_stat.st_ctime_ns,
                directory_stat.st_mtime_ns,
                stat.S_IFMT(directory_stat.st_mode),
            )
            file_rows = ()
            if namespace == "units":
                file_stat = unit_file.stat()
                file_rows = (
                    (
                        unit_file.name,
                        (
                            file_stat.st_dev,
                            file_stat.st_ino,
                            file_stat.st_ctime_ns,
                            file_stat.st_mtime_ns,
                            file_stat.st_size,
                            stat.S_IFMT(file_stat.st_mode),
                        ),
                    ),
                )
            namespace_metadata.append((namespace, directory_identity, file_rows))

        class Store:
            def __init__(self):
                self.run_id = run_id
                self.run_dir = run_dir
                self.expected = SimpleNamespace(
                    units=(SimpleNamespace(unit_id=unit_id),)
                )

            @contextmanager
            def _open_result_namespace(self, namespace):
                descriptor = os.open(
                    self.run_dir / namespace,
                    os.O_RDONLY | os.O_DIRECTORY,
                )
                try:
                    yield None, descriptor
                finally:
                    os.close(descriptor)

            def completed_records(self):
                raise AssertionError("metadata recheck parsed a completed record")

            def attempt_records(self):
                raise AssertionError("metadata recheck parsed an attempt record")

        config = SimpleNamespace(
            family_id=family,
            child=children[index],
            data_keys=object(),
            model_keys=object(),
            shared_plan=object(),
        )
        folds.append(
            SimpleNamespace(
                family_id=family,
                config=config,
                store=Store(),
                data_keys=config.data_keys,
                data=SimpleNamespace(manifests={}),
                model_keys=config.model_keys,
                models=object(),
                shared_plan=config.shared_plan,
            )
        )
        metadata_folds.append((run_id, tuple(namespace_metadata)))
    folds = tuple(folds)
    metadata_snapshot = tuple(metadata_folds)
    snapshot_sha256 = "e" * 64

    committed_bytes, manifest_parent_identity, manifest_file_identity = (
        runtime._read_pinned_file(manifest_path, label="test manifest")
    )
    raw_fd = runtime.secure_fs.open_directory_chain(raw_root)
    try:
        raw_root_identity, child_identities = runtime._tree_identities_at(raw_fd, manifest)
        prepared_tree_sha256 = runtime._walk_tree_digest_at(
            raw_fd, child_identities, canonical_child_ids=manifest.child_run_ids
        )
    finally:
        os.close(raw_fd)
    repository_fd = runtime.secure_fs.open_directory_chain(repository)
    try:
        repository_identity = runtime.secure_fs.directory_identity(repository_fd)
    finally:
        os.close(repository_fd)
    monkeypatch.setattr(runtime, "_authority_sources", lambda _manifest: ())
    monkeypatch.setattr(runtime, "capture_system_provenance", lambda *_args: provenance)
    monkeypatch.setattr(runtime, "validate_screening_provenance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime, "_recheck_authority_repository", lambda *_args: None)
    monkeypatch.setattr(runtime, "build_screening_data_keys", lambda config, _prov: config.data_keys)
    monkeypatch.setattr(
        runtime,
        "build_screening_model_keys",
        lambda config, _data_keys, _manifests: config.model_keys,
    )
    monkeypatch.setattr(
        runtime,
        "build_screening_shared_plan",
        lambda config, _data_keys, _manifests, _models: config.shared_plan,
    )
    monkeypatch.setattr(runtime, "_child_manifest", lambda config, *_args: config.child)
    monkeypatch.setattr(runtime, "_assert_global_inventory", lambda *_args: None)
    monkeypatch.setattr(
        runtime,
        "_parse_result_snapshot_artifact",
        lambda *_args, **_kwargs: (snapshot_sha256, metadata_snapshot),
    )
    monkeypatch.setattr(runtime, "_recheck_metadata_selection_authority", lambda *_args: None)
    monkeypatch.setattr(runtime, "_assert_tree_shape_at", lambda *_args: None)

    real_read_bytes_at = runtime.secure_fs.read_bytes_at

    def reject_outcome_reads(directory_fd, name):
        if name != "phase2-screening-readiness.json":
            raise AssertionError(f"metadata recheck read a result payload: {name}")
        return real_read_bytes_at(directory_fd, name)

    monkeypatch.setattr(runtime.secure_fs, "read_bytes_at", reject_outcome_reads)
    handle = runtime.ScreeningRuntime(
        manifest_path=manifest_path,
        raw_root=raw_root,
        repository=repository,
        device_policy=object(),
        manifest_bytes=committed_bytes,
        manifest=manifest,
        authority_sources=(),
        provenance=provenance,
        folds=folds,
        tree_sha256=prepared_tree_sha256,
        raw_root_identity=raw_root_identity,
        child_identities=child_identities,
        manifest_parent_identity=manifest_parent_identity,
        manifest_file_identity=manifest_file_identity,
        result_namespace_snapshot=(),
        authority_repository=repository,
        authority_repository_identity=repository_identity,
        authority_provenance=provenance,
        metadata_only_snapshot_bytes=b"canonical artifact fixture",
        metadata_only_snapshot_sha256=snapshot_sha256,
        metadata_only_selection_lock_sha256=SELECTION_SHA256,
    )
    runtime.recheck_screening_runtime_metadata_only(handle)


def test_selection_authority_allows_a_distinct_clean_consumer_checkout_but_binds_sources(
    monkeypatch: pytest.MonkeyPatch,
):
    document = json.loads(_artifact())
    screening_provenance = SimpleNamespace(
        git_commit_sha="1" * 40,
        provenance_sha256="2" * 64,
    )
    consumer_authority_provenance = SimpleNamespace(
        git_commit_sha="3" * 40,
        provenance_sha256="4" * 64,
    )
    manifest_bytes = b"frozen manifest bytes"
    source_rows = [{"label": "protocol", "sha256": "5" * 64}]
    tree_digest = "c" * 64
    document["readiness"].update(
        {
            "manifest_bytes_sha256": _sha(manifest_bytes),
            "manifest_bytes_length": len(manifest_bytes),
            "manifest_parent_identity": [6, 7],
            "manifest_file_identity": [6, 8, 9, 10, len(manifest_bytes)],
            "prepared_tree_sha256": tree_digest,
        }
    )
    document["repositories"] = {
        "screening": {
            "path": "/historical/screening-checkout",
            "git_commit_sha": screening_provenance.git_commit_sha,
            "git_dirty": False,
            "provenance_sha256": screening_provenance.provenance_sha256,
        },
        "authority": {
            "path": "/tmp/publisher-authority-checkout",
            "git_commit_sha": "a" * 40,
            "git_dirty": False,
            "provenance_sha256": "b" * 64,
            "directory_identity": [101, 102],
            "source_sha256": source_rows,
        },
    }
    unsigned = {key: value for key, value in document.items() if key != "identity_sha256"}
    document["identity_sha256"] = _sha(canonical_json_bytes(unsigned))
    artifact_bytes = canonical_json_bytes(document) + b"\n"
    frozen_snapshot_digest = document["result_namespace_snapshot_sha256"]
    runtime._parse_result_snapshot_artifact(
        artifact_bytes, selection_lock_sha256=SELECTION_SHA256
    )

    lock = {
        "analysis": {"result_namespace_snapshot_sha256": frozen_snapshot_digest},
        "authority": {
            "readiness_manifest_bytes_sha256": _sha(manifest_bytes),
            "source_git_commit_sha": screening_provenance.git_commit_sha,
            "source_provenance_sha256": screening_provenance.provenance_sha256,
            "prepared_tree_sha256": tree_digest,
        },
    }
    current_parent_identity = (201, 202)
    current_file_identity = {
        "device": 9,
        "inode": 203,
        "ctime_ns": 204,
        "mtime_ns": 205,
        "size": 206,
        "file_type": stat.S_IFREG,
    }
    monkeypatch.setattr(
        publication,
        "_canonical_selection_lock",
        lambda _runtime, digest: (
            lock if digest == SELECTION_SHA256 else None,
            current_parent_identity,
            current_file_identity,
        ),
    )
    monkeypatch.setattr(
        runtime,
        "provenance_identity_sha256",
        lambda provenance: provenance.provenance_sha256,
    )
    consumer = SimpleNamespace(
        metadata_only_selection_lock_sha256=SELECTION_SHA256,
        metadata_only_snapshot_bytes=artifact_bytes,
        manifest_bytes=manifest_bytes,
        manifest_parent_identity=(6, 7),
        manifest_file_identity=(6, 8, 9, 10, len(manifest_bytes)),
        tree_sha256=tree_digest,
        repository=Path("/historical/screening-checkout"),
        provenance=screening_provenance,
        authority_repository=Path("/clean/consumer-authority-checkout"),
        authority_provenance=consumer_authority_provenance,
        authority_repository_identity=(301, 302),
        authority_digests=(("protocol", "5" * 64),),
    )

    # The publisher's selection inode and authority checkout differ from the
    # consumer's, while the exact frozen selection and authority source bytes agree.
    runtime._recheck_metadata_selection_authority(consumer, frozen_snapshot_digest)

    consumer.authority_digests = (("protocol", "d" * 64),)
    with pytest.raises(TrainingDataArtifactError, match="source or provenance bindings"):
        runtime._recheck_metadata_selection_authority(consumer, frozen_snapshot_digest)
