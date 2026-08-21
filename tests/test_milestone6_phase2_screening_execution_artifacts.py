from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from levelup.experiments import milestone6_phase2_screening_execution_artifacts as artifacts


def _digest(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


class _Key:
    def __init__(self, *, key_id: str, condition_id: str, tuple_id: str, replicate: int):
        self.key_id = key_id
        self.condition_id = condition_id
        self.training_tuple_id = tuple_id
        self.replicate = replicate
        self.fold_id = "lofo-plain"
        self.heldout_family_id = "plain"

    def model_dump(self, *, mode: str) -> dict[str, str | int]:
        return {"key_id": self.key_id, "condition_id": self.condition_id}


def _fixture(monkeypatch, tmp_path):
    base = "B1-global-frequency"
    tuple_id = "lr0p003-e120"
    replicate = 0
    model_key_id = _digest("m")
    view_key_id = _digest("v")
    evidence_key_id = _digest("e")
    model_artifact_id = _digest("a")
    view_artifact_id = _digest("b")
    evidence_artifact_id = _digest("c")
    model_cost_id = _digest("d")
    view_cost_id = _digest("f")
    evidence_cost_id = _digest("g")
    condition_ids = tuple(f"{base}--t{temp}" for temp in ("0p6", "0p9", "1p2"))
    candidate = {
        "tuple_id": "candidate-1",
        "training_tuple_id": tuple_id,
        "learning_rate": 0.003,
        "training_epochs": 120,
        "search_temperature": 0.6,
    }
    model_key = _Key(
        key_id=model_key_id,
        condition_id=base,
        tuple_id=tuple_id,
        replicate=replicate,
    )
    report = SimpleNamespace(trainable_parameters=4, optimizer_steps=1, forward_passes=1)
    model_manifest = SimpleNamespace(
        key=model_key,
        artifact_id=model_artifact_id,
        report=report,
        model_id="global_affordance_mlp_frequency_v1",
    )
    model_cost = SimpleNamespace(
        key_id=model_key_id,
        artifact_id=model_artifact_id,
        cost_id=model_cost_id,
        key=model_key.model_dump(mode="json"),
    )
    view_key = SimpleNamespace(
        key_id=view_key_id,
        fold_id="lofo-plain",
        heldout_family_id="plain",
        condition_id=base,
        replicate=replicate,
    )
    view_manifest = SimpleNamespace(key=view_key, artifact_id=view_artifact_id)
    evidence_key = SimpleNamespace(
        key_id=evidence_key_id,
        fold_id="lofo-plain",
        heldout_family_id="plain",
        replicate=replicate,
    )
    evidence_manifest = SimpleNamespace(key=evidence_key, evidence_id=evidence_artifact_id)

    def shared(kind, key_id, group):
        return SimpleNamespace(
            kind=kind,
            key_id=key_id,
            owner_family_id="plain",
            owner_fold_id="lofo-plain",
            owner_replicate=0,
            owner_group_id=group,
            consumer_phase="validation",
            consumer_condition_ids=condition_ids,
            consumer_unit_ids=("unit-1", "unit-2", "unit-3"),
        )

    validation_calls = []
    store = SimpleNamespace(
        run_id="fold-run",
        run_dir=tmp_path,
        validate_shared_reference_set=lambda unit, refs: validation_calls.append(refs),
    )
    fold = SimpleNamespace(
        family_id="plain",
        config=SimpleNamespace(
            parameters={"fold_id": "lofo-plain"},
            conditions=(SimpleNamespace(condition_id=condition_ids[0]),),
        ),
        store=store,
        data_keys=SimpleNamespace(evidence={0: evidence_key}, views={(base, 0): view_key}),
        data=SimpleNamespace(
            manifests=SimpleNamespace(evidence={0: evidence_manifest}, views={(base, 0): view_manifest}),
            evidence_cost_ids={0: evidence_cost_id},
            view_cost_ids={(base, 0): view_cost_id},
        ),
        model_keys=SimpleNamespace(models={(base, tuple_id, 0): model_key}),
        models=SimpleNamespace(
            manifests={(base, tuple_id, 0): model_manifest},
            costs={(base, tuple_id, 0): model_cost},
        ),
        shared_plan=SimpleNamespace(
            artifacts=(
                shared("training_data_evidence", evidence_key_id, "canonical-evidence"),
                shared("training_data_view", view_key_id, base),
                shared("training_artifact", model_key_id, base),
            )
        ),
    )

    class _UnitKey:
        phase = "validation"
        condition_id = condition_ids[0]
        family_id = "plain"
        replicate = 0

    planned = SimpleNamespace(unit_id="unit-1", key=_UnitKey())
    condition = SimpleNamespace(
        condition_id=condition_ids[0],
        parameters={
            "base_condition_id": base,
            "candidate_tuple_id": "candidate-1",
            "training_tuple_id": tuple_id,
            "learning_rate": 0.003,
            "training_epochs": 120,
            "search_temperature": 0.6,
        },
    )
    monkeypatch.setattr(artifacts, "base_condition_id", lambda _: base)
    monkeypatch.setattr(
        artifacts, "_expected_model_id", lambda _: "global_affordance_mlp_frequency_v1"
    )
    def candidate_for(condition_id):
        value = {"t0p6": 0.6, "t0p9": 0.9, "t1p2": 1.2}[condition_id.rsplit("--", 1)[1]]
        return {**candidate, "search_temperature": value}

    monkeypatch.setattr(artifacts, "candidate_for_condition", candidate_for)
    loader_calls = []

    def load_index(root, key):
        return SimpleNamespace(artifact_id=model_artifact_id)

    def load_model(root, artifact_id, *, expected_key, model_factory):
        loader_calls.append(artifact_id)
        return torch.nn.Linear(1, 1).eval(), model_manifest

    monkeypatch.setattr(artifacts, "load_training_key_index", load_index)
    monkeypatch.setattr(artifacts, "load_training_model", load_model)
    return fold, planned, condition, validation_calls, loader_calls, condition_ids, base, tuple_id


def test_fixed_conditions_do_not_touch_shared_refs_or_models(monkeypatch, tmp_path):
    fold, planned, _, validation_calls, loader_calls, *_ = _fixture(monkeypatch, tmp_path)
    for condition_id in ("A0-no-probe-uniform", "A1-paid-probe-uniform"):
        condition = SimpleNamespace(condition_id=condition_id, parameters={})
        planned.key.condition_id = condition_id
        assert artifacts.prepare_unit_model(fold, planned, condition) is None
    assert validation_calls == []
    assert loader_calls == []


def test_three_temperature_variants_share_one_loaded_model(monkeypatch, tmp_path):
    fold, planned, condition, validation_calls, loader_calls, condition_ids, base, tuple_id = _fixture(
        monkeypatch, tmp_path
    )
    cache = artifacts.ScreeningModelCache()
    first = artifacts.prepare_unit_model(fold, planned, condition, cache)
    assert first is not None
    assert first.identity == ("fold-run", base, tuple_id, 0)
    assert [reference.kind for reference in first.references] == [
        "training_data_evidence",
        "training_data_view",
        "training_artifact",
    ]
    assert len(validation_calls) == 1
    assert len(loader_calls) == 1

    for index, temperature in enumerate(("0p9", "1p2"), start=1):
        condition_id = condition_ids[index]
        condition.condition_id = condition_id
        condition.parameters["search_temperature"] = {"0p9": 0.9, "1p2": 1.2}[temperature]
        planned.key.condition_id = condition_id
        planned.unit_id = f"unit-{index + 1}"
        fold.config.conditions = (SimpleNamespace(condition_id=condition_id),)
        result = artifacts.prepare_unit_model(fold, planned, condition, cache)
        assert result is not None
        assert result.model is first.model
        assert result.identity == first.identity
    assert len(validation_calls) == 3
    assert len(loader_calls) == 1
    assert len(cache) == 1


def test_cross_fold_shared_authorization_is_rejected(monkeypatch, tmp_path):
    fold, planned, condition, *_ = _fixture(monkeypatch, tmp_path)
    fold.family_id = "other-family"
    try:
        artifacts.prepare_unit_model(fold, planned, condition)
    except Exception as exc:  # noqa: BLE001 - exact storage exception is implementation detail
        assert "another fold" in str(exc) or "authorization" in str(exc)
    else:
        raise AssertionError("cross-fold unit unexpectedly accepted")


def test_cache_hit_rejects_manifest_substitution(monkeypatch, tmp_path):
    fold, planned, condition, *_ = _fixture(monkeypatch, tmp_path)
    cache = artifacts.ScreeningModelCache()
    assert artifacts.prepare_unit_model(fold, planned, condition, cache) is not None
    identity = next(iter(fold.models.manifests))
    manifest = fold.models.manifests[identity]
    fold.models.manifests[identity] = SimpleNamespace(
        key=manifest.key,
        artifact_id=_digest("substituted-artifact"),
        report=manifest.report,
        model_id=manifest.model_id,
    )
    fold.models.costs[identity].artifact_id = fold.models.manifests[identity].artifact_id
    with pytest.raises(Exception, match="cached model|identity drifted"):
        artifacts.prepare_unit_model(fold, planned, condition, cache)


def test_candidate_parameter_equality_is_type_exact() -> None:
    assert artifacts._same_value(1.0, 1.0)
    assert not artifacts._same_value(True, 1)
    assert not artifacts._same_value(1, 1.0)
