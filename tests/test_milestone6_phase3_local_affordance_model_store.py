from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

import levelup.experiments.milestone6_phase3_local_affordance_model_store as store
import levelup.experiments.milestone6_phase3_local_affordance_models as training
from levelup.experiments.milestone6_phase3_local_affordance_model_records import build_model_record
from levelup.experiments.milestone6_phase3_local_affordance_plan import build_local_affordance_plan
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.learning.state_conditioned import (
    DecisionExample,
    GlobalAffordanceScorer,
    GlobalDecisionExample,
    TrainingReport,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _examples(condition: str):
    if condition == training.CONDITION_B2:
        return (
            GlobalDecisionExample(torch.tensor([[0.0] * 49, [1.0] * 49]), 1),
            GlobalDecisionExample(torch.tensor([[1.0] * 49, [0.0] * 49]), 0),
        )
    return (
        DecisionExample(torch.tensor([[0.0] * 54, [1.0] * 54]), 1),
        DecisionExample(torch.tensor([[1.0] * 54, [0.0] * 54]), 0),
    )


@pytest.fixture(scope="module")
def plan():
    return build_local_affordance_plan()


@pytest.fixture
def trained_model(plan, monkeypatch):
    owner = next(item for item in plan.model_owners if item.condition_id == training.CONDITION_B2)
    view = next(item for item in plan.views if item.view_id == owner.view_id)
    pair = _examples(owner.condition_id)
    examples = tuple(pair[index % len(pair)] for index in range(40))
    training_view = training._seal_local_affordance_training_view(
        view=view,
        examples=examples,
        task_example_counts=(1,) * 40,
    )

    def train_global(ordered_examples, *, training, model_seed):
        torch.manual_seed(model_seed)
        model = GlobalAffordanceScorer()
        return model, TrainingReport(
            3601,
            training.epochs,
            training.epochs * len(ordered_examples),
            len(ordered_examples),
        )

    monkeypatch.setattr(training, "train_global_listwise_optimum_model", train_global)
    prepared = training.prepare_local_affordance_model(
        plan=plan,
        view=view,
        owner=owner,
        training_view=training_view,
    )
    payload = store.serialize_local_affordance_model(plan, prepared)
    record = build_model_record(
        plan=plan,
        owner_id=owner.owner_id,
        training_examples=prepared.report.training_examples,
        training_examples_sha256=prepared.training_examples_sha256,
        model_state_sha256=prepared.model_state_sha256,
        model_identity_sha256=prepared.model_identity_sha256,
        artifact_bytes=len(payload),
        artifact_sha256=_sha_bytes(payload),
        preparation_git_commit_sha="b" * 40,
        preparation_provenance_sha256="c" * 64,
    )
    return prepared, record, payload


def _write(root: Path, plan, prepared, record):
    return store.write_local_affordance_model(
        root,
        plan,
        prepared,
        record,
        preparation_git_commit_sha="b" * 40,
        preparation_provenance_sha256="c" * 64,
    )


def test_single_owner_store_roundtrips_safe_tensor_bytes_and_resumes(tmp_path, plan, trained_model):
    prepared, record, payload = trained_model
    root = tmp_path / "local-affordance-models"
    entry = _write(root, plan, prepared, record)
    assert entry.owner_id == record.key.owner_id
    assert (root / store.MANIFEST_NAME).is_file()
    assert (root / store.MODELS_DIR / f"{entry.owner_id}.model").read_bytes() == payload

    with store.open_local_affordance_model_store(root) as pinned:
        loaded_record, tensors = store.load_local_affordance_model_at(
            pinned,
            entry.owner_id,
            plan=plan,
            preparation_git_commit_sha="b" * 40,
            preparation_provenance_sha256="c" * 64,
        )
        assert loaded_record == record
        assert tuple(tensor.name for tensor in tensors) == tuple(
            sorted(prepared.model.state_dict())
        )
        assert _write(root, plan, prepared, record) == entry


def test_concurrent_writer_fails_before_claiming_any_owner(tmp_path, plan, trained_model):
    prepared, record, _payload = trained_model
    root = tmp_path / "store"
    with store.open_local_affordance_model_store(root, exclusive_writer=True):
        with pytest.raises(store.LocalAffordanceModelStoreError, match="another model writer"):
            _write(root, plan, prepared, record)
    assert _write(root, plan, prepared, record).owner_id == prepared.owner.owner_id


def test_safe_tensor_encoding_is_deterministic_and_has_no_pickle_signature(plan, trained_model):
    prepared, _record, payload = trained_model
    assert store.serialize_local_affordance_model(plan, prepared) == payload
    assert payload.startswith(store.MODEL_MAGIC)
    assert b"pickle" not in payload.lower()
    assert not payload.startswith(b"\x80\x04")


def test_missing_manifest_with_payload_is_rejected_as_orphan(tmp_path, plan, trained_model):
    prepared, record, _payload = trained_model
    root = tmp_path / "store"
    _write(root, plan, prepared, record)
    (root / store.MANIFEST_NAME).unlink()
    with store.open_local_affordance_model_store(root) as pinned:
        with pytest.raises(store.LocalAffordanceModelStoreError, match="orphan"):
            store._manifest(pinned)


def test_retry_recovers_exact_selected_owner_record_only_after_first_claim(
    tmp_path, plan, trained_model, monkeypatch
):
    prepared, record, _payload = trained_model
    root = tmp_path / "store"
    original_claim = store._claim
    calls = 0

    def crash_after_record(directory_fd, staging_fd, name, content):
        nonlocal calls
        original_claim(directory_fd, staging_fd, name, content)
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated crash after first owner file")

    monkeypatch.setattr(store, "_claim", crash_after_record)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _write(root, plan, prepared, record)
    monkeypatch.setattr(store, "_claim", original_claim)

    assert (root / store.RECORDS_DIR / f"{record.key.owner_id}.json").is_file()
    assert not (root / store.MANIFEST_NAME).exists()
    entry = _write(root, plan, prepared, record)
    with store.open_local_affordance_model_store(root) as pinned:
        loaded, _ = store.load_local_affordance_model_at(
            pinned,
            entry.owner_id,
            plan=plan,
            preparation_git_commit_sha="b" * 40,
            preparation_provenance_sha256="c" * 64,
        )
    assert loaded == record


def test_retry_recovers_exact_owner_pair_after_claims_before_manifest(
    tmp_path, plan, trained_model, monkeypatch
):
    prepared, record, _payload = trained_model
    root = tmp_path / "store"
    original_write = store._write_temp

    def crash_before_commit(*args, **kwargs):
        raise RuntimeError("simulated crash before manifest commit")

    monkeypatch.setattr(store, "_write_temp", crash_before_commit)
    with pytest.raises(RuntimeError, match="before manifest"):
        _write(root, plan, prepared, record)
    monkeypatch.setattr(store, "_write_temp", original_write)

    assert (root / store.RECORDS_DIR / f"{record.key.owner_id}.json").is_file()
    assert (root / store.MODELS_DIR / f"{record.key.owner_id}.model").is_file()
    assert not (root / store.MANIFEST_NAME).exists()
    assert _write(root, plan, prepared, record).owner_id == record.key.owner_id


def test_recovery_rejects_changed_selected_owner_or_foreign_orphans(
    tmp_path, plan, trained_model
):
    prepared, record, _payload = trained_model
    root = tmp_path / "store"
    model_path = root / store.MODELS_DIR / f"{record.key.owner_id}.model"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"not the selected model")
    with pytest.raises(store.LocalAffordanceModelStoreError, match="differs"):
        _write(root, plan, prepared, record)

    model_path.unlink()
    foreign = root / store.RECORDS_DIR / ("f" * 64 + ".json")
    foreign.write_bytes(b"foreign orphan")
    with pytest.raises(store.LocalAffordanceModelStoreError, match="inventory"):
        _write(root, plan, prepared, record)


def test_extra_model_and_symlink_are_rejected(tmp_path, plan, trained_model):
    prepared, record, _payload = trained_model
    root = tmp_path / "store"
    _write(root, plan, prepared, record)
    (root / store.MODELS_DIR / ("f" * 64 + ".model")).write_bytes(b"extra")
    with store.open_local_affordance_model_store(root) as pinned:
        with pytest.raises(store.LocalAffordanceModelStoreError, match="inventory"):
            store._validate_shape(pinned, store._manifest(pinned))
    (root / store.MODELS_DIR / ("f" * 64 + ".model")).unlink()
    symlink = root / store.RECORDS_DIR / ("f" * 64 + ".json")
    symlink.symlink_to(root / store.MANIFEST_NAME)
    with store.open_local_affordance_model_store(root) as pinned:
        with pytest.raises(store.LocalAffordanceModelStoreError):
            store._validate_shape(pinned, store._manifest(pinned))


def test_in_place_artifact_mutation_and_provenance_mismatch_fail_closed(tmp_path, plan, trained_model):
    prepared, record, _payload = trained_model
    root = tmp_path / "store"
    _write(root, plan, prepared, record)
    path = root / store.MODELS_DIR / record.artifact.artifact_name
    changed = bytearray(path.read_bytes())
    changed[-1] ^= 1
    path.write_bytes(changed)
    with store.open_local_affordance_model_store(root) as pinned:
        with pytest.raises(store.LocalAffordanceModelStoreError, match="hash"):
            store.load_local_affordance_model_at(
                pinned,
                record.key.owner_id,
                plan=plan,
                preparation_git_commit_sha="b" * 40,
                preparation_provenance_sha256="c" * 64,
            )
        with pytest.raises(store.LocalAffordanceModelStoreError, match="provenance"):
            store.load_local_affordance_model_at(
                pinned,
                record.key.owner_id,
                plan=plan,
                preparation_git_commit_sha="d" * 40,
                preparation_provenance_sha256="c" * 64,
            )


def test_symlinked_store_root_is_rejected(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "store"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(store.LocalAffordanceModelStoreError, match="symlink"):
        with store.open_local_affordance_model_store(link):
            pass


def test_noncanonical_manifest_is_rejected(tmp_path, plan, trained_model):
    prepared, record, _payload = trained_model
    root = tmp_path / "store"
    _write(root, plan, prepared, record)
    manifest_path = root / store.MANIFEST_NAME
    manifest_path.write_bytes(canonical_json_bytes({"wrong": True}) + b"\n")
    with store.open_local_affordance_model_store(root) as pinned:
        with pytest.raises(store.LocalAffordanceModelStoreError, match="manifest"):
            store._manifest(pinned)


def test_model_mutation_after_preparation_is_rejected_before_write(tmp_path, plan, trained_model):
    prepared, record, _payload = trained_model
    with torch.no_grad():
        next(iter(prepared.model.parameters())).add_(1.0)
    with pytest.raises(store.LocalAffordanceModelStoreError, match="sealed preparation"):
        _write(tmp_path / "store", plan, prepared, record)


def test_forged_unsealed_preparation_is_rejected_before_serialization_or_write(
    tmp_path, plan, trained_model
):
    prepared, record, _payload = trained_model
    forged = object.__new__(type(prepared))
    object.__setattr__(forged, "_token", None)
    with pytest.raises(store.LocalAffordanceModelStoreError, match="sealed preparation"):
        store.serialize_local_affordance_model(plan, forged)
    with pytest.raises(store.LocalAffordanceModelStoreError, match="sealed preparation"):
        _write(tmp_path / "store", plan, forged, record)


def test_forged_training_and_model_identity_metadata_is_rejected(tmp_path, plan, trained_model):
    prepared, record, payload = trained_model
    forged = build_model_record(
        plan=plan,
        owner_id=record.key.owner_id,
        training_examples=record.cost.training_examples,
        training_examples_sha256="d" * 64,
        model_state_sha256=record.key.model_state_sha256,
        model_identity_sha256="e" * 64,
        artifact_bytes=len(payload),
        artifact_sha256=_sha_bytes(payload),
        preparation_git_commit_sha="b" * 40,
        preparation_provenance_sha256="c" * 64,
    )
    with pytest.raises(store.LocalAffordanceModelStoreError, match="record does not match"):
        _write(tmp_path / "store", plan, prepared, forged)
