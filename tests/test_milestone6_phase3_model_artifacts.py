from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import levelup.experiments.milestone6_phase3_model_artifacts as artifacts
from levelup.experiments.milestone6_phase3_model_artifacts import (
    Phase3ModelArtifactCost,
    Phase3ModelArtifactError,
    Phase3ModelArtifactKey,
    load_phase3_model_bundle_from_at,
    load_phase3_model_index_at,
    load_phase3_model_manifest,
    load_phase3_model_manifest_at,
    open_phase3_model_artifact_reader,
    open_phase3_model_artifact_reader_at,
    open_phase3_model_output,
    write_phase3_model_artifact,
)
from levelup.experiments.milestone6_phase3_model_preparation import _scan_existing
from levelup.experiments.milestone6_phase3_models import _model_state_sha256
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.records import (
    PhaseAccounting,
    TrainingPreparationAccounting,
)


@dataclass(frozen=True)
class _Prep:
    owner: object
    view: object
    model: torch.nn.Module
    report: object
    training_spec: object
    model_state_sha256: str
    model_identity_sha256: str


def _preparation() -> _Prep:
    torch.manual_seed(123)
    model = torch.nn.Linear(2, 1)
    state_sha = _model_state_sha256(model)
    view = SimpleNamespace(
        view=SimpleNamespace(
            view_id="b" * 64,
            condition_id="S-state-availability-listwise-optimum",
            fold_id="fold-0",
            heldout_family="combo",
            replicate=0,
        ),
        evidence_payload_sha256="c" * 64,
        evidence_payload_bytes=12,
    )
    owner = SimpleNamespace(
        owner_id="d" * 64,
        condition_id=view.view.condition_id,
        fold_id=view.view.fold_id,
        heldout_family=view.view.heldout_family,
        replicate=0,
        training_tuple_id="lr0p003-e120",
        model_seed=4,
    )
    report = SimpleNamespace(
        trainable_parameters=3841,
        optimizer_steps=120,
        forward_passes=120,
        training_examples=1,
        recurrent_steps=0,
    )
    training_spec = SimpleNamespace(learning_rate=0.003, weight_decay=0.0001)
    return _Prep(owner, view, model, report, training_spec, state_sha, "e" * 64)


def _write(tmp_path: Path) -> tuple[Path, object]:
    # The preparation validator is tested at the model-preparation boundary;
    # this storage test supplies a small deterministic fixture and exercises the
    # persistence boundary independently.
    original = artifacts.validate_phase3_model_preparation
    artifacts.validate_phase3_model_preparation = lambda *_args, **_kwargs: None
    try:
        manifest = write_phase3_model_artifact(
            tmp_path,
            preparation=_preparation(),
            plan_id="1" * 64,
            protocol_sha256="2" * 64,
            evidence_lock_sha256="3" * 64,
            preparation_git_commit_sha="4" * 64,
            preparation_provenance_sha256="5" * 64,
            accounting=TrainingPreparationAccounting(
                training=PhaseAccounting(
                    optimizer_steps=120,
                    forward_passes=120,
                ),
                serialization=PhaseAccounting(calls=1),
            ),
        )
    finally:
        artifacts.validate_phase3_model_preparation = original
    return tmp_path, manifest


def test_round_trip_reloads_typed_bundle(tmp_path: Path) -> None:
    root, manifest = _write(tmp_path)
    loaded = load_phase3_model_manifest(root, manifest.artifact_id)
    assert loaded == manifest
    with open_phase3_model_artifact_reader(root) as reader:
        loaded_index = load_phase3_model_index_at(reader, manifest.key.key_id)
        index, cost, descriptor_manifest, state = load_phase3_model_bundle_from_at(
            reader, manifest.key
        )
    assert loaded_index == index
    assert index.artifact_id == manifest.artifact_id
    assert cost.artifact_id == manifest.artifact_id
    assert descriptor_manifest == manifest
    assert set(state) == {"bias", "weight"}


def test_artifact_key_carries_preparation_provenance_identity(tmp_path: Path) -> None:
    original = artifacts.validate_phase3_model_preparation
    artifacts.validate_phase3_model_preparation = lambda *_args, **_kwargs: None
    try:
        manifest = write_phase3_model_artifact(
            tmp_path,
            preparation=_preparation(),
            plan_id="1" * 64,
            protocol_sha256="2" * 64,
            evidence_lock_sha256="3" * 64,
            preparation_git_commit_sha="4" * 64,
            preparation_provenance_sha256="5" * 64,
        )
    finally:
        artifacts.validate_phase3_model_preparation = original
    assert manifest.key.preparation_git_commit_sha == "4" * 64
    assert manifest.key.preparation_provenance_sha256 == "5" * 64


def test_writer_output_is_consumed_as_an_indexed_key_bundle(tmp_path: Path) -> None:
    root, manifest = _write(tmp_path)
    existing = _scan_existing(root, {manifest.key.owner_id})
    assert existing == {manifest.key.owner_id: manifest.key}


@pytest.mark.parametrize(
    "missing",
    [
        (artifacts.KEYS_DIR, "{key_id}.json"),
        (artifacts.COSTS_DIR, "{key_id}.json"),
    ],
)
def test_incomplete_publication_is_repaired_by_second_writer(
    tmp_path: Path, missing: tuple[str, str]
) -> None:
    root, manifest = _write(tmp_path)
    namespace, filename = missing
    target = root / namespace / filename.format(key_id=manifest.key.key_id)
    target.unlink()
    assert _scan_existing(root, {manifest.key.owner_id}) == {}
    _, resumed = _write(tmp_path)
    assert resumed == manifest
    assert _scan_existing(root, {manifest.key.owner_id}) == {
        manifest.key.owner_id: manifest.key
    }


def test_legacy_index_without_cost_is_repaired(tmp_path: Path) -> None:
    root, manifest = _write(tmp_path)
    cost = root / artifacts.COSTS_DIR / f"{manifest.key.key_id}.json"
    cost.unlink()
    assert _scan_existing(root, {manifest.key.owner_id}) == {}
    _write(tmp_path)
    assert _scan_existing(root, {manifest.key.owner_id}) == {
        manifest.key.owner_id: manifest.key
    }


def test_artifact_only_publication_is_repaired(tmp_path: Path) -> None:
    root, manifest = _write(tmp_path)
    (root / artifacts.KEYS_DIR / f"{manifest.key.key_id}.json").unlink()
    (root / artifacts.COSTS_DIR / f"{manifest.key.key_id}.json").unlink()
    assert _scan_existing(root, {manifest.key.owner_id}) == {}
    _write(tmp_path)
    assert _scan_existing(root, {manifest.key.owner_id}) == {
        manifest.key.owner_id: manifest.key
    }


def test_partial_publication_for_unselected_owner_fails_closed(tmp_path: Path) -> None:
    root, manifest = _write(tmp_path)
    (root / artifacts.KEYS_DIR / f"{manifest.key.key_id}.json").unlink()
    (root / artifacts.COSTS_DIR / f"{manifest.key.key_id}.json").unlink()
    with pytest.raises(Exception, match="unselected owner"):
        _scan_existing(
            root,
            {manifest.key.owner_id},
            repairable_owner_ids=set(),
        )


def test_strict_scan_of_successful_writer_is_exact(tmp_path: Path) -> None:
    root, manifest = _write(tmp_path)
    assert _scan_existing(
        root,
        {manifest.key.owner_id},
        repairable_owner_ids=set(),
    ) == {manifest.key.owner_id: manifest.key}


def test_scan_rejects_key_namespace_payload_that_is_not_an_index(tmp_path: Path) -> None:
    root, manifest = _write(tmp_path)
    path = root / artifacts.KEYS_DIR / f"{manifest.key.key_id}.json"
    path.write_bytes(
        artifacts.canonical_json_bytes(manifest.key.model_dump(mode="json")) + b"\n"
    )
    with pytest.raises(Exception, match="index"):
        _scan_existing(root, {manifest.key.owner_id})


def test_scan_rejects_orphaned_cost_or_index(tmp_path: Path) -> None:
    root, manifest = _write(tmp_path)
    artifact = root / artifacts.ARTIFACTS_DIR / manifest.artifact_id
    artifact.rename(root / artifacts.ARTIFACTS_DIR / ("0" * 64))
    with pytest.raises(Exception, match="index|cost|artifact"):
        _scan_existing(root, {manifest.key.owner_id})


def test_stale_preparation_staging_does_not_poison_idempotent_resume(
    tmp_path: Path,
) -> None:
    root, manifest = _write(tmp_path)
    stale = root / artifacts.STAGING_DIR / "interrupted-publication"
    stale.mkdir()
    (stale / "partial").write_bytes(b"not-authority")
    resumed_root, resumed = _write(tmp_path)
    assert resumed_root == root
    assert resumed == manifest


def test_tensor_substitution_and_hash_drift_fail_closed(tmp_path: Path) -> None:
    root, manifest = _write(tmp_path)
    tensor = root / artifacts.ARTIFACTS_DIR / manifest.artifact_id / artifacts.TENSORS_DIR / "0000.bin"
    original = tensor.read_bytes()
    tensor.write_bytes(original + b"x")
    with pytest.raises(Phase3ModelArtifactError, match="tensor"):
        load_phase3_model_manifest(root, manifest.artifact_id)


def test_symlinked_tensor_is_rejected(tmp_path: Path) -> None:
    root, manifest = _write(tmp_path)
    tensor = root / artifacts.ARTIFACTS_DIR / manifest.artifact_id / artifacts.TENSORS_DIR / "0000.bin"
    target = tensor.with_name("target.bin")
    target.write_bytes(tensor.read_bytes())
    tensor.unlink()
    tensor.symlink_to(target.name)
    with pytest.raises(Phase3ModelArtifactError):
        load_phase3_model_manifest(root, manifest.artifact_id)


def test_report_and_lineage_drift_are_schema_errors(tmp_path: Path) -> None:
    root, manifest = _write(tmp_path)
    manifest_path = root / artifacts.ARTIFACTS_DIR / manifest.artifact_id / artifacts.MANIFEST_NAME
    raw = manifest_path.read_text()
    manifest_path.write_text(raw.replace('"training_examples":1', '"training_examples":2'))
    with pytest.raises(Phase3ModelArtifactError):
        load_phase3_model_manifest(root, manifest.artifact_id)


def test_key_rejects_training_tuple_and_recurrent_accounting_drift(
    tmp_path: Path,
) -> None:
    _, manifest = _write(tmp_path)
    value = manifest.key.model_dump(mode="json")
    value["optimizer"]["learning_rate"] = 0.01
    with pytest.raises(ValueError, match="training tuple"):
        Phase3ModelArtifactKey.model_validate(value)
    value = manifest.key.model_dump(mode="json")
    value["recurrent_steps"] = 1
    value["report"]["recurrent_steps"] = 1
    with pytest.raises(ValueError, match="state-only"):
        Phase3ModelArtifactKey.model_validate(value)


def test_noncanonical_manifest_bytes_are_rejected(tmp_path: Path) -> None:
    root, manifest = _write(tmp_path)
    manifest_path = (
        root
        / artifacts.ARTIFACTS_DIR
        / manifest.artifact_id
        / artifacts.MANIFEST_NAME
    )
    manifest_path.write_bytes(b" " + manifest_path.read_bytes())
    with pytest.raises(Phase3ModelArtifactError, match="non-canonical"):
        load_phase3_model_manifest(root, manifest.artifact_id)


def test_cost_rejects_nontraining_resource_drift(tmp_path: Path) -> None:
    _, manifest = _write(tmp_path)
    value = {
        "schema_version": "milestone6.phase3.model-cost.v1",
        "key_id": manifest.key.key_id,
        "artifact_id": manifest.artifact_id,
        "scope": "phase3_model_preparation",
        "key": manifest.key.model_dump(mode="json"),
        "accounting": TrainingPreparationAccounting(
            training_probes=PhaseAccounting(actions=1),
            training=PhaseAccounting(
                optimizer_steps=120,
                forward_passes=120,
            ),
            serialization=PhaseAccounting(calls=1),
        ).model_dump(mode="json"),
    }
    value["cost_id"] = artifacts._digest(value)
    with pytest.raises(ValueError, match="cost accounting"):
        Phase3ModelArtifactCost.model_validate(value)


def test_extra_artifact_entry_is_rejected(tmp_path: Path) -> None:
    root, manifest = _write(tmp_path)
    artifact_dir = root / artifacts.ARTIFACTS_DIR / manifest.artifact_id
    (artifact_dir / "unexpected").write_bytes(b"drift")
    with pytest.raises(Phase3ModelArtifactError, match="inventory"):
        load_phase3_model_manifest(root, manifest.artifact_id)


def test_pinned_reader_survives_path_substitution(tmp_path: Path) -> None:
    root, manifest = _write(tmp_path)
    root_fd = secure_fs.open_directory_chain(root)
    try:
        with open_phase3_model_artifact_reader_at(root_fd) as reader:
            artifacts_root = root / artifacts.ARTIFACTS_DIR
            replacement = root / f"{artifacts.ARTIFACTS_DIR}.old"
            os.rename(artifacts_root, replacement)
            artifacts_root.symlink_to(replacement.name, target_is_directory=True)
            # The already-pinned descriptor continues to name the original
            # directory, while a fresh path traversal fails closed.
            assert load_phase3_model_manifest_at(reader, manifest.artifact_id) == manifest
            with pytest.raises(Phase3ModelArtifactError):
                load_phase3_model_manifest(root, manifest.artifact_id)
    finally:
        os.close(root_fd)


def test_pinned_writer_fails_closed_after_namespace_substitution(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(artifacts, "validate_phase3_model_preparation", lambda *_args, **_kwargs: None)
    with open_phase3_model_output(tmp_path) as output:
        old = tmp_path / artifacts.COSTS_DIR
        replacement = tmp_path / f"{artifacts.COSTS_DIR}.old"
        old.rename(replacement)
        old.symlink_to(replacement.name, target_is_directory=True)
        with pytest.raises(Phase3ModelArtifactError, match="replaced"):
            write_phase3_model_artifact(
                tmp_path,
                preparation=_preparation(),
                plan_id="1" * 64,
                protocol_sha256="2" * 64,
                evidence_lock_sha256="3" * 64,
                preparation_git_commit_sha="4" * 64,
                preparation_provenance_sha256="5" * 64,
                pinned_output=output,
            )
        assert not tuple(replacement.iterdir())
