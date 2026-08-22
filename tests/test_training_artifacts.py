from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path

import pytest
import torch

import levelup.experiments.runner.secure_fs as secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import PhaseAccounting, ResourceAccounting
from levelup.experiments.runner.training_artifacts import (
    MODEL_IDS,
    TrainingArtifactKey,
    TrainingReportMetadata,
    load_training_cost,
    load_training_cost_at,
    load_training_key_index,
    load_training_key_index_at,
    load_training_manifest,
    load_training_manifest_at,
    load_training_model,
    load_training_model_at,
    write_training_artifact,
)
from levelup.learning.state_conditioned import (
    GlobalAffordanceScorer,
    StateConditionedScorer,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _key(*, model_id: str = "state_conditioned_mlp_listwise_v1") -> TrainingArtifactKey:
    return TrainingArtifactKey(
        screening_candidates_sha256=_hash("screening-candidates"),
        protocol_sha256=_hash("protocol"),
        task_manifest_sha256=_hash("tasks"),
        expected_unit_plan_sha256=_hash("expected-units"),
        exposure_sha256=_hash("exposure"),
        training_data_sha256=_hash("training-data"),
        provenance_sha256=_hash("provenance"),
        fold_id="fold-plain",
        heldout_family_id="combo",
        ordered_training_task_ids=("task-a", "task-b"),
        ordered_heldout_task_ids=("task-c",),
        condition_id="C-state-conditioned-listwise-optimum",
        learner_id="state-affordance-mlp-listwise-v1",
        objective_id="listwise_optimum",
        backbone_id=model_id,
        training_tuple_id="lr0p003-e120",
        replicate=0,
        model_seed=10,
        data_order_seed=11,
        probe_seeds=(12, 13),
        environment_seeds=(0, 0, 0),
        probe_spec_sha256=_hash("probe-spec"),
        training_config_sha256=_hash("training-config-without-search-temperature"),
        capacity_spec_sha256=_hash("capacity-spec"),
    )


def _report(parameters: int = 3841) -> TrainingReportMetadata:
    return TrainingReportMetadata(
        trainable_parameters=parameters,
        optimizer_steps=2,
        forward_passes=8,
        training_examples=4,
    )


def _factory(model_id: str) -> torch.nn.Module:
    if model_id == "state_conditioned_mlp_listwise_v1":
        return StateConditionedScorer()
    if model_id in {
        "global_affordance_mlp_frequency_v1",
        "global_affordance_mlp_listwise_v1",
    }:
        return GlobalAffordanceScorer()
    raise AssertionError(model_id)


@pytest.mark.parametrize(
    ("model_id", "factory"),
    [
        ("global_affordance_mlp_frequency_v1", _factory),
        ("global_affordance_mlp_listwise_v1", _factory),
        ("state_conditioned_mlp_listwise_v1", _factory),
    ],
)
def test_current_mlps_round_trip_exact_outputs_without_pickle(
    tmp_path: Path,
    model_id: str,
    factory: object,
) -> None:
    torch.manual_seed(91)
    model = _factory(model_id)
    assert model_id in MODEL_IDS
    parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    inputs = torch.randn(5, 49 if model_id.startswith("global") else 54)
    expected = model(inputs).detach()
    manifest = write_training_artifact(
        tmp_path,
        key=_key(model_id=model_id),
        model_id=model_id,
        model=model,
        accounting=ResourceAccounting(),
        report=_report(parameters),
    )
    loaded, _ = load_training_model(
        tmp_path,
        manifest.artifact_id,
        expected_key=_key(model_id=model_id),
        model_factory=factory,  # type: ignore[arg-type]
    )
    assert torch.equal(expected, loaded(inputs).detach())
    assert not list(tmp_path.rglob("*.pkl"))
    assert not list(tmp_path.rglob("*.pickle"))
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.pth"))


def test_write_is_idempotent_and_manifest_is_content_addressed(tmp_path: Path) -> None:
    model = StateConditionedScorer()
    first = write_training_artifact(
        tmp_path,
        key=_key(),
        model_id="state_conditioned_mlp_listwise_v1",
        model=model,
        accounting=ResourceAccounting(),
        report=_report(),
    )
    second = write_training_artifact(
        tmp_path,
        key=_key(),
        model_id="state_conditioned_mlp_listwise_v1",
        model=model,
        accounting=ResourceAccounting(),
        report=_report(),
    )
    assert first == second
    assert (tmp_path / "training-artifacts" / first.artifact_id / "manifest.json").is_file()
    index = load_training_key_index(tmp_path, _key())
    assert index.artifact_id == first.artifact_id


def test_report_parameter_count_must_match_model(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="parameter count"):
        write_training_artifact(
            tmp_path,
            key=_key(),
            model_id="state_conditioned_mlp_listwise_v1",
            model=StateConditionedScorer(),
            accounting=ResourceAccounting(),
            report=_report(3601),
        )


def test_same_key_different_wall_time_keeps_first_cost_record(tmp_path: Path) -> None:
    model = StateConditionedScorer()
    first_cost = ResourceAccounting(setup=PhaseAccounting(wall_seconds=1.0))
    second_cost = ResourceAccounting(setup=PhaseAccounting(wall_seconds=2.0))
    first = write_training_artifact(
        tmp_path,
        key=_key(),
        model_id="state_conditioned_mlp_listwise_v1",
        model=model,
        accounting=first_cost,
        report=_report(),
    )
    second = write_training_artifact(
        tmp_path,
        key=_key(),
        model=model,
        model_id="state_conditioned_mlp_listwise_v1",
        accounting=second_cost,
        report=_report(),
    )
    assert first.artifact_id == second.artifact_id
    loaded = load_training_cost(tmp_path, _key()).accounting
    assert hasattr(loaded, "as_resource_accounting")
    assert loaded.as_resource_accounting() == first_cost


def test_tampered_cost_record_is_rejected(tmp_path: Path) -> None:
    write_training_artifact(
        tmp_path,
        key=_key(),
        model_id="state_conditioned_mlp_listwise_v1",
        model=StateConditionedScorer(),
        accounting=ResourceAccounting(),
        report=_report(),
    )
    path = tmp_path / "training-artifact-costs" / f"{_key().key_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["accounting"]["setup"]["calls"] = 99
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="cost"):
        load_training_cost(tmp_path, _key())


def test_version_one_cost_record_remains_readable(tmp_path: Path) -> None:
    write_training_artifact(
        tmp_path,
        key=_key(),
        model_id="state_conditioned_mlp_listwise_v1",
        model=StateConditionedScorer(),
        accounting=ResourceAccounting(),
        report=_report(),
    )
    path = tmp_path / "training-artifact-costs" / f"{_key().key_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = "runner.training-artifact-cost.v1"
    raw.pop("scope")
    preparation = raw["accounting"]
    raw["accounting"] = {
        "setup": preparation["setup"],
        "probes": preparation["training_probes"],
        "training": preparation["training"],
        "search": PhaseAccounting().model_dump(mode="json"),
        "replay": preparation["reference_replay"],
        "evaluator": PhaseAccounting().model_dump(mode="json"),
        "serialization": preparation["serialization"],
    }
    raw["cost_id"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in raw.items() if key != "cost_id"})
    ).hexdigest()
    path.write_bytes(canonical_json_bytes(raw) + b"\n")
    assert load_training_cost(tmp_path, _key()).schema_version.endswith("v1")


def test_training_preparation_rejects_search_or_evaluator_cost(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="search or evaluator"):
        write_training_artifact(
            tmp_path,
            key=_key(),
            model_id="state_conditioned_mlp_listwise_v1",
            model=StateConditionedScorer(),
            accounting=ResourceAccounting(search=PhaseAccounting(actions=1)),
            report=_report(),
        )


def test_tampered_tensor_is_rejected(tmp_path: Path) -> None:
    model = StateConditionedScorer()
    manifest = write_training_artifact(
        tmp_path,
        key=_key(),
        model_id="state_conditioned_mlp_listwise_v1",
        model=model,
        accounting=ResourceAccounting(),
        report=_report(),
    )
    tensor = next((tmp_path / "training-artifacts" / manifest.artifact_id / "tensors").iterdir())
    tensor.write_bytes(tensor.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="integrity"):
        load_training_manifest(tmp_path, manifest.artifact_id)


def test_manifest_body_change_is_rejected_without_id_update(tmp_path: Path) -> None:
    manifest = write_training_artifact(
        tmp_path,
        key=_key(),
        model_id="state_conditioned_mlp_listwise_v1",
        model=StateConditionedScorer(),
        accounting=ResourceAccounting(),
        report=_report(),
    )
    path = tmp_path / "training-artifacts" / manifest.artifact_id / "manifest.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["report"]["training_examples"] += 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact|manifest"):
        load_training_manifest(tmp_path, manifest.artifact_id)


def test_tensor_and_self_declared_hash_change_cannot_retain_artifact_id(
    tmp_path: Path,
) -> None:
    manifest = write_training_artifact(
        tmp_path,
        key=_key(),
        model_id="state_conditioned_mlp_listwise_v1",
        model=StateConditionedScorer(),
        accounting=ResourceAccounting(),
        report=_report(),
    )
    artifact_dir = tmp_path / "training-artifacts" / manifest.artifact_id
    tensor_path = artifact_dir / "tensors" / manifest.tensors[0].filename
    payload = bytearray(tensor_path.read_bytes())
    payload[0] ^= 1
    tensor_path.write_bytes(payload)
    manifest_path = artifact_dir / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["tensors"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact|manifest"):
        load_training_manifest(tmp_path, manifest.artifact_id)


def test_wrong_key_index_is_rejected(tmp_path: Path) -> None:
    manifest = write_training_artifact(
        tmp_path,
        key=_key(),
        model_id="state_conditioned_mlp_listwise_v1",
        model=StateConditionedScorer(),
        accounting=ResourceAccounting(),
        report=_report(),
    )
    path = tmp_path / "training-artifact-keys" / f"{_key().key_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["key"]["fold_id"] = "wrong-fold"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="index"):
        load_training_key_index(tmp_path, _key())
    assert manifest.artifact_id


def test_incomplete_staging_directory_does_not_block_write(tmp_path: Path) -> None:
    staging = tmp_path / "training-artifacts" / ".interrupted.staging-dead"
    (staging / "tensors").mkdir(parents=True)
    (staging / "tensors" / "0000.bin").write_bytes(b"partial")
    write_training_artifact(
        tmp_path,
        key=_key(),
        model_id="state_conditioned_mlp_listwise_v1",
        model=StateConditionedScorer(),
        accounting=ResourceAccounting(),
        report=_report(),
    )


def test_symlinked_roots_artifact_and_tensor_directories_are_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "real-root"
    write_training_artifact(
        real_root,
        key=_key(),
        model_id="state_conditioned_mlp_listwise_v1",
        model=StateConditionedScorer(),
        accounting=ResourceAccounting(),
        report=_report(),
    )
    linked_root = tmp_path / "linked-root"
    os.symlink(real_root, linked_root)
    with pytest.raises(RuntimeError, match="symlink"):
        load_training_manifest(linked_root, next(real_root.glob("training-artifacts/*")).name)

    artifact = next((real_root / "training-artifacts").iterdir())
    artifact_id = artifact.name
    moved = tmp_path / "moved-artifact"
    shutil.move(str(artifact), moved)
    os.symlink(moved, artifact)
    with pytest.raises(RuntimeError, match="symlink"):
        load_training_manifest(real_root, artifact_id)

    index_path = real_root / "training-artifact-keys" / f"{_key().key_id}.json"
    moved_index = tmp_path / "moved-index.json"
    shutil.move(str(index_path), moved_index)
    os.symlink(moved_index, index_path)
    with pytest.raises(RuntimeError, match="symlink"):
        load_training_key_index(real_root, _key())

    artifact.unlink()
    shutil.move(str(moved), artifact)
    tensor_dir = artifact / "tensors"
    moved_tensors = tmp_path / "moved-tensors"
    shutil.move(str(tensor_dir), moved_tensors)
    os.symlink(moved_tensors, tensor_dir)
    with pytest.raises(RuntimeError, match="symlink"):
        load_training_manifest(real_root, artifact_id)


@pytest.mark.parametrize("wrong", ["provenance_sha256", "training_config_sha256"])
def test_wrong_expected_key_is_rejected(tmp_path: Path, wrong: str) -> None:
    model = StateConditionedScorer()
    key = _key()
    manifest = write_training_artifact(
        tmp_path,
        key=key,
        model_id="state_conditioned_mlp_listwise_v1",
        model=model,
        accounting=ResourceAccounting(),
        report=_report(),
    )
    changed = key.model_copy(update={wrong: _hash("changed")})
    with pytest.raises(RuntimeError, match="key"):
        load_training_model(
            tmp_path,
            manifest.artifact_id,
            expected_key=changed,
            model_factory=_factory,
        )


def test_wrong_factory_model_shape_and_manifest_dtype_are_rejected(tmp_path: Path) -> None:
    model = StateConditionedScorer()
    manifest = write_training_artifact(
        tmp_path,
        key=_key(),
        model_id="state_conditioned_mlp_listwise_v1",
        model=model,
        accounting=ResourceAccounting(),
        report=_report(),
    )
    with pytest.raises(RuntimeError, match="state dict"):
        load_training_model(
            tmp_path,
            manifest.artifact_id,
            expected_key=_key(),
            model_factory=lambda _: GlobalAffordanceScorer(),
        )
    manifest_path = tmp_path / "training-artifacts" / manifest.artifact_id / "manifest.json"
    raw = manifest.model_dump(mode="json")
    raw["tensors"][0]["dtype"] = "float64"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_training_manifest(tmp_path, manifest.artifact_id)


@pytest.mark.parametrize(
    ("field", "value"),
    [("model_id", "global_affordance_mlp_listwise_v1"), ("shape", [1])],
)
def test_manifest_rejects_wrong_model_id_or_shape(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    model = StateConditionedScorer()
    manifest = write_training_artifact(
        tmp_path,
        key=_key(),
        model_id="state_conditioned_mlp_listwise_v1",
        model=model,
        accounting=ResourceAccounting(),
        report=_report(),
    )
    manifest_path = tmp_path / "training-artifacts" / manifest.artifact_id / "manifest.json"
    raw = manifest.model_dump(mode="json")
    if field == "model_id":
        raw[field] = value
    else:
        raw["tensors"][0][field] = value
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_training_manifest(tmp_path, manifest.artifact_id)


def test_descriptor_loaders_match_path_loaders(tmp_path: Path) -> None:
    key = _key()
    manifest = write_training_artifact(
        tmp_path,
        key=key,
        model_id=key.backbone_id,
        model=StateConditionedScorer(),
        accounting=ResourceAccounting(),
        report=_report(),
    )
    run_fd = secure_fs.open_directory_chain(tmp_path)
    try:
        assert load_training_manifest_at(run_fd, manifest.artifact_id) == load_training_manifest(
            tmp_path, manifest.artifact_id
        )
        assert load_training_key_index_at(run_fd, key) == load_training_key_index(tmp_path, key)
        assert load_training_cost_at(run_fd, key) == load_training_cost(tmp_path, key)
        fd_model, fd_manifest = load_training_model_at(
            run_fd,
            key,
            model_factory=lambda _: StateConditionedScorer(),
        )
        path_model, path_manifest = load_training_model(
            tmp_path,
            manifest.artifact_id,
            expected_key=key,
            model_factory=lambda _: StateConditionedScorer(),
        )
        assert fd_manifest == path_manifest
        assert all(
            torch.equal(fd_model.state_dict()[name], value)
            for name, value in path_model.state_dict().items()
        )
    finally:
        os.close(run_fd)


@pytest.mark.parametrize("entry_kind", ["artifact", "tensor", "index", "cost"])
def test_descriptor_loaders_reject_symlinked_entries(tmp_path: Path, entry_kind: str) -> None:
    key = _key()
    manifest = write_training_artifact(
        tmp_path,
        key=key,
        model_id=key.backbone_id,
        model=StateConditionedScorer(),
        accounting=ResourceAccounting(),
        report=_report(),
    )
    artifact = tmp_path / "training-artifacts" / manifest.artifact_id
    if entry_kind == "artifact":
        moved = tmp_path / "moved-artifact"
        artifact.rename(moved)
        artifact.symlink_to(moved, target_is_directory=True)

        def loader(fd: int) -> object:
            return load_training_manifest_at(fd, manifest.artifact_id)

    elif entry_kind == "tensor":
        tensors = artifact / "tensors"
        moved = tmp_path / "moved-tensors"
        tensors.rename(moved)
        tensors.symlink_to(moved, target_is_directory=True)

        def loader(fd: int) -> object:
            return load_training_manifest_at(fd, manifest.artifact_id)

    elif entry_kind == "index":
        path = tmp_path / "training-artifact-keys" / f"{key.key_id}.json"
        moved = tmp_path / "moved-index.json"
        path.rename(moved)
        path.symlink_to(moved)

        def loader(fd: int) -> object:
            return load_training_key_index_at(fd, key)

    else:
        path = tmp_path / "training-artifact-costs" / f"{key.key_id}.json"
        moved = tmp_path / "moved-cost.json"
        path.rename(moved)
        path.symlink_to(moved)

        def loader(fd: int) -> object:
            return load_training_cost_at(fd, key)
    run_fd = secure_fs.open_directory_chain(tmp_path)
    try:
        with pytest.raises(RuntimeError):
            loader(run_fd)
    finally:
        os.close(run_fd)


def test_descriptor_loader_rejects_nonregular_or_extra_tensor_entry(tmp_path: Path) -> None:
    key = _key()
    manifest = write_training_artifact(
        tmp_path,
        key=key,
        model_id=key.backbone_id,
        model=StateConditionedScorer(),
        accounting=ResourceAccounting(),
        report=_report(),
    )
    tensors = tmp_path / "training-artifacts" / manifest.artifact_id / "tensors"
    first = tensors / manifest.tensors[0].filename
    first.unlink()
    first.mkdir()
    run_fd = secure_fs.open_directory_chain(tmp_path)
    try:
        with pytest.raises(RuntimeError):
            load_training_manifest_at(run_fd, manifest.artifact_id)
    finally:
        os.close(run_fd)

    shutil.rmtree(first)
    first.write_bytes(b"invalid")
    (tensors / "extra.bin").write_bytes(b"unexpected")
    run_fd = secure_fs.open_directory_chain(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="unexpected tensor"):
            load_training_manifest_at(run_fd, manifest.artifact_id)
    finally:
        os.close(run_fd)


def test_manifest_and_tensors_share_one_pinned_artifact_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _key()
    manifest = write_training_artifact(
        tmp_path,
        key=key,
        model_id=key.backbone_id,
        model=StateConditionedScorer(),
        accounting=ResourceAccounting(),
        report=_report(),
    )
    artifact = tmp_path / "training-artifacts" / manifest.artifact_id
    detached = tmp_path / "detached-artifact"
    replacement = tmp_path / "replacement-artifact"
    replacement.mkdir()
    (replacement / "manifest.json").write_text("not-json", encoding="utf-8")
    original_read = secure_fs.read_bytes_at
    substituted = False

    def read_bytes(directory_fd: int, name: str) -> bytes:
        nonlocal substituted
        payload = original_read(directory_fd, name)
        if name == "manifest.json" and not substituted:
            substituted = True
            artifact.rename(detached)
            artifact.symlink_to(replacement, target_is_directory=True)
        return payload

    monkeypatch.setattr(secure_fs, "read_bytes_at", read_bytes)
    run_fd = secure_fs.open_directory_chain(tmp_path)
    try:
        assert load_training_manifest_at(run_fd, manifest.artifact_id) == manifest
    finally:
        os.close(run_fd)
    assert substituted
    assert (replacement / "manifest.json").read_text(encoding="utf-8") == "not-json"


@pytest.mark.parametrize("substitute_ancestor", [False, True])
def test_descriptor_loader_is_anchored_across_rename_and_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitute_ancestor: bool,
) -> None:
    parent = tmp_path / "parent"
    root = parent / "run"
    root.mkdir(parents=True)
    key = _key()
    manifest = write_training_artifact(
        root,
        key=key,
        model_id=key.backbone_id,
        model=StateConditionedScorer(),
        accounting=ResourceAccounting(),
        report=_report(),
    )
    external = tmp_path / "external"
    external_artifact = external / "run" / "training-artifacts" / manifest.artifact_id
    external_artifact.mkdir(parents=True)
    (external_artifact / "manifest.json").write_text("not-json", encoding="utf-8")

    barrier = threading.Barrier(2)
    original_read = secure_fs.read_bytes_at
    gated = False

    def read_bytes(directory_fd: int, name: str) -> bytes:
        nonlocal gated
        if name == "manifest.json" and not gated:
            gated = True
            barrier.wait(timeout=5)
            barrier.wait(timeout=5)
        return original_read(directory_fd, name)

    monkeypatch.setattr(secure_fs, "read_bytes_at", read_bytes)
    run_fd = secure_fs.open_directory_chain(root)

    def substitute() -> None:
        barrier.wait(timeout=5)
        if substitute_ancestor:
            detached = tmp_path / "parent-detached"
            parent.rename(detached)
            parent.symlink_to(external, target_is_directory=True)
        else:
            detached = tmp_path / "run-detached"
            root.rename(detached)
            root.symlink_to(external / "run", target_is_directory=True)
        barrier.wait(timeout=5)

    replacer = threading.Thread(target=substitute)
    replacer.start()
    try:
        loaded = load_training_manifest_at(run_fd, manifest.artifact_id)
    finally:
        os.close(run_fd)
    replacer.join(timeout=10)
    assert not replacer.is_alive()
    assert loaded == manifest
    assert (external_artifact / "manifest.json").read_text(encoding="utf-8") == "not-json"
