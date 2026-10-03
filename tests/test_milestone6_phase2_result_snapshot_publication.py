from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase2_result_snapshot_publication as publication
from levelup.experiments.milestone6_phase2_screening_runtime import (
    ScreeningRuntime,
    ScreeningRuntimeFold,
)
from levelup.experiments.runner.config import DevicePolicy, canonical_json_bytes
from levelup.experiments.runner.records import SystemProvenance
from levelup.experiments.runner.training_data_artifacts import TrainingDataArtifactError

PROVENANCE = SystemProvenance(
    git_commit_sha="0" * 40,
    git_dirty=False,
    python_version="test",
    packages={"levelup-bench": "test"},
    installed_packages_sha256="a" * 64,
    os="test",
    architecture="test",
    cpu="test",
    cpu_count=1,
    memory_bytes=1,
    requested_device="cpu",
    resolved_device="cpu",
    requested_torch_threads=1,
    actual_torch_threads=1,
    requested_torch_interop_threads=1,
    actual_torch_interop_threads=1,
    deterministic_algorithms_requested=True,
    deterministic_algorithms_actual=True,
    processes=1,
    captured_at_utc=datetime(2026, 10, 3, tzinfo=UTC),
)


@contextmanager
def _open_namespace(store, namespace):
    run_descriptor = os.open(store.run_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    descriptor = os.open(store.run_dir / namespace, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        yield run_descriptor, descriptor
    finally:
        os.close(descriptor)
        os.close(run_descriptor)


def _make_runtime(tmp_path, monkeypatch, *, foreign_key_family: str | None = None):
    monkeypatch.setattr(publication, "EXPECTED_FOLD_UNITS", 1)
    monkeypatch.setattr(publication, "EXPECTED_TOTAL_UNITS", 6)
    monkeypatch.setattr(publication, "recheck_screening_runtime_readonly", lambda _runtime: None)
    repository = tmp_path / "repo"
    lock_path = repository / publication.SELECTION_LOCK_PATH
    lock_path.parent.mkdir(parents=True)
    (repository / "experiments").mkdir()

    lock = {
        "schema_version": "milestone6.phase2.selection-lock.v1",
        "scientific_boundary": {
            "development_only": True,
            "final_family_access": False,
            "final_method_selection": False,
            "claims_deferred": [
                "transition_information",
                "history_or_sequence",
                "frontier_optimum_pairing",
            ],
        },
        "matrix": {"families": 6, "units": 6},
        "authority": {
            "readiness_manifest_bytes_sha256": hashlib.sha256(
                b"readiness-manifest-bytes"
            ).hexdigest(),
            "source_git_commit_sha": PROVENANCE.git_commit_sha,
            "source_provenance_sha256": publication.provenance_identity_sha256(
                PROVENANCE
            ),
            "prepared_tree_sha256": "b" * 64,
        },
        "analysis": {"result_namespace_snapshot_sha256": "f" * 64},
    }
    lock_content = json.dumps(lock, indent=2).encode() + b"\n"
    lock_path.write_bytes(lock_content)
    selection_lock_sha256 = hashlib.sha256(lock_content).hexdigest()

    folds = []
    for index, family in enumerate(publication.FAMILIES):
        run_dir = tmp_path / f"run-{family}"
        (run_dir / "units").mkdir(parents=True)
        (run_dir / "attempts").mkdir()
        unit_id = f"{index + 1:064x}"
        (run_dir / "units" / f"{unit_id}.json").write_bytes(
            b'{"private_outcome_value":"DO_NOT_PUBLISH"}'
        )
        planned = SimpleNamespace(
            unit_id=unit_id,
            key=SimpleNamespace(
                family_id=family,
                phase="validation",
            ),
        )
        store = SimpleNamespace(
            run_dir=run_dir,
            run_id=f"run-{family}",
            expected=SimpleNamespace(units=(planned,)),
        )
        # Bind this fold's store in the closure instead of the loop's final value.
        store._open_result_namespace = lambda namespace, bound=store: _open_namespace(bound, namespace)
        folds.append(
            ScreeningRuntimeFold(
                family_id=family,
                config=object(),
                store=store,
                data_keys=object(),
                data=object(),
                model_keys=object(),
                models=object(),
                shared_plan=object(),
            )
        )
    folds_tuple = tuple(folds)
    snapshots = tuple(
        (fold.store.run_id, publication._capture_fold_namespaces(fold)[0])
        for fold in folds_tuple
    )
    lock["analysis"]["result_namespace_snapshot_sha256"] = hashlib.sha256(
        canonical_json_bytes(snapshots)
    ).hexdigest()
    lock_content = json.dumps(lock, indent=2).encode() + b"\n"
    lock_path.write_bytes(lock_content)
    selection_lock_sha256 = hashlib.sha256(lock_content).hexdigest()
    if foreign_key_family is not None:
        folds_tuple[0].store.expected.units[0].key.family_id = foreign_key_family
    manifest = SimpleNamespace(
        family_order=publication.FAMILIES,
        development_only=True,
        final_family_access=False,
        expected_total_units=6,
    )
    runtime = ScreeningRuntime(
        manifest_path=repository / "experiments/milestone6_phase2_screening_readiness.json",
        raw_root=tmp_path / "raw",
        repository=repository,
        device_policy=DevicePolicy(requested_device="cpu", torch_threads=1),
        manifest_bytes=b"readiness-manifest-bytes",
        manifest=manifest,
        authority_sources=(),
        provenance=PROVENANCE,
        folds=folds_tuple,
        tree_sha256="b" * 64,
        raw_root_identity=(1, 2),
        child_identities=(),
        manifest_parent_identity=(3, 4),
        manifest_file_identity=(5, 6, 7, 8, 9),
        result_namespace_snapshot=snapshots,
        authority_repository=repository,
        authority_repository_identity=(repository.stat().st_dev, repository.stat().st_ino),
        authority_provenance=PROVENANCE,
    )
    return runtime, selection_lock_sha256


def test_publication_is_self_hashed_identity_only_and_covers_development_units(
    tmp_path, monkeypatch
):
    runtime, lock_sha = _make_runtime(tmp_path, monkeypatch)
    payload = publication.publish_phase2_result_namespace_identity(
        runtime, selection_lock_sha256=lock_sha
    )
    document = json.loads(payload)
    digest = document.pop("identity_sha256")
    assert digest == hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    assert document["development_matrix"]["total_units"] == 6
    assert document["development_matrix"]["final_families"] == []
    assert document["use_boundary"]["learner_input"] is False
    assert document["use_boundary"]["administrative_metadata_authority"] is True
    assert document["frozen_selection"]["selection_lock_sha256"] == lock_sha
    assert document["result_namespace_snapshot_sha256"]
    assert b"DO_NOT_PUBLISH" not in payload
    assert b"private_outcome_value" not in payload
    assert b"exact_optimum_success_rate" not in payload


def test_publication_rejects_changed_result_payload_digest(tmp_path, monkeypatch):
    runtime, lock_sha = _make_runtime(tmp_path, monkeypatch)
    file_path = runtime.folds[0].store.run_dir / "units" / f"{1:064x}.json"
    file_path.write_bytes(b'{"private_outcome_value":"changed"}')
    with pytest.raises(TrainingDataArtifactError, match="namespace changed"):
        publication.publish_phase2_result_namespace_identity(
            runtime, selection_lock_sha256=lock_sha
        )


def test_publication_rejects_self_consistent_selection_lock_with_wrong_snapshot(
    tmp_path, monkeypatch
):
    runtime, _lock_sha = _make_runtime(tmp_path, monkeypatch)
    lock_path = runtime.authority_repository / publication.SELECTION_LOCK_PATH
    lock = json.loads(lock_path.read_bytes())
    lock["analysis"]["result_namespace_snapshot_sha256"] = "f" * 64
    changed = json.dumps(lock, indent=2).encode() + b"\n"
    lock_path.write_bytes(changed)
    with pytest.raises(TrainingDataArtifactError, match="frozen selection snapshot"):
        publication.publish_phase2_result_namespace_identity(
            runtime, selection_lock_sha256=hashlib.sha256(changed).hexdigest()
        )


def test_publication_rejects_missing_or_foreign_unit_filename(tmp_path, monkeypatch):
    runtime, lock_sha = _make_runtime(tmp_path, monkeypatch)
    units = runtime.folds[0].store.run_dir / "units"
    original = units / f"{1:064x}.json"
    original.rename(units / f"{99:064x}.json")
    with pytest.raises(TrainingDataArtifactError, match="completed-unit names"):
        publication.publish_phase2_result_namespace_identity(
            runtime, selection_lock_sha256=lock_sha
        )


def test_publication_rejects_foreign_family_unit_key(tmp_path, monkeypatch):
    runtime, lock_sha = _make_runtime(tmp_path, monkeypatch, foreign_key_family="final-family")
    with pytest.raises(TrainingDataArtifactError, match="foreign family"):
        publication.publish_phase2_result_namespace_identity(
            runtime, selection_lock_sha256=lock_sha
        )


def test_publication_rejects_selection_lock_sha_mismatch(tmp_path, monkeypatch):
    runtime, _lock_sha = _make_runtime(tmp_path, monkeypatch)
    with pytest.raises(TrainingDataArtifactError, match="selection lock differs"):
        publication.publish_phase2_result_namespace_identity(
            runtime, selection_lock_sha256="0" * 64
        )


def test_selection_lock_is_resolved_from_pinned_authority_checkout(tmp_path, monkeypatch):
    runtime, lock_sha = _make_runtime(tmp_path, monkeypatch)
    historical_checkout = tmp_path / "historical-screening-checkout"
    historical_checkout.mkdir()
    runtime = replace(runtime, repository=historical_checkout)
    payload = publication.publish_phase2_result_namespace_identity(
        runtime, selection_lock_sha256=lock_sha
    )
    assert json.loads(payload)["frozen_selection"]["selection_lock_sha256"] == lock_sha


def test_selection_lock_must_bind_the_readiness_manifest_bytes(tmp_path, monkeypatch):
    runtime, lock_sha = _make_runtime(tmp_path, monkeypatch)
    runtime = replace(runtime, manifest_bytes=b"substituted-readiness-bytes")
    with pytest.raises(TrainingDataArtifactError, match="manifest, provenance, and prepared tree"):
        publication.publish_phase2_result_namespace_identity(
            runtime, selection_lock_sha256=lock_sha
        )
