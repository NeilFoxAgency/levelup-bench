from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import levelup.experiments.milestone6_phase2_screening_models as models
import levelup.experiments.milestone6_phase2_screening_preparation as preparation
from levelup.experiments.milestone6_phase2_screening import build_screening_child_config
from levelup.experiments.milestone6_phase2_screening_preparation import (
    MaterializedScreeningData,
    ScreeningDataKeys,
    ScreeningDataManifests,
    ScreeningModelKeys,
    build_screening_data_keys,
    load_screening_data_inventory,
)
from levelup.experiments.runner.records import (
    PhaseAccounting,
    SystemProvenance,
    TrainingPreparationAccounting,
)
from levelup.experiments.runner.training_artifacts import TrainingReportMetadata
from levelup.experiments.runner.training_data_artifacts import TrainingDataArtifactError

PROVENANCE = SystemProvenance(
    git_commit_sha="0" * 40,
    git_dirty=False,
    python_version="test-python",
    packages={"levelup-bench": "test"},
    installed_packages_sha256="a" * 64,
    os="test-os",
    architecture="test-arch",
    cpu="test-cpu",
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
    captured_at_utc=datetime(2026, 8, 21, tzinfo=UTC),
)


def _keys() -> tuple[object, ScreeningDataKeys]:
    config = build_screening_child_config("plain")
    return config, build_screening_data_keys(config, PROVENANCE)


def _namespace_skeleton(root: Path, keys: ScreeningDataKeys) -> None:
    names = {
        "screening-data-intents": {f"{key.key_id}.json" for key in keys.evidence.values()},
        "training-data-evidence-costs": {
            f"{key.key_id}.json" for key in keys.evidence.values()
        },
        "training-data-view-costs": {f"{key.key_id}.json" for key in keys.views.values()},
        "training-data-artifact-keys": {
            f"{key.key_id}.json" for key in keys.views.values()
        },
    }
    for namespace, entries in names.items():
        directory = root / namespace
        directory.mkdir()
        for entry in entries:
            (directory / entry).write_bytes(b"placeholder")


def test_data_inventory_is_read_only_and_rejects_missing_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, keys = _keys()
    before = tuple(sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")))

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("data inventory attempted materialization")

    monkeypatch.setattr(preparation, "_screening_training_batch", forbidden)
    monkeypatch.setattr(preparation, "write_training_data_artifact", forbidden)
    with pytest.raises(TrainingDataArtifactError, match="existing directory"):
        load_screening_data_inventory(config, keys, tmp_path / "missing")
    after = tuple(sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")))
    assert after == before


def test_data_inventory_rejects_extra_namespace_without_materializing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, keys = _keys()
    _namespace_skeleton(tmp_path, keys)
    extra = tmp_path / "training-data-view-costs" / "unexpected.json"
    extra.write_bytes(b"extra")
    before = tuple(sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")))

    monkeypatch.setattr(
        preparation,
        "_screening_training_batch",
        lambda *args, **kwargs: pytest.fail("inventory loader probed or replayed"),
    )
    monkeypatch.setattr(
        preparation,
        "write_training_data_artifact",
        lambda *args, **kwargs: pytest.fail("inventory loader wrote data"),
    )
    with pytest.raises(TrainingDataArtifactError, match="inventory drifted"):
        load_screening_data_inventory(config, keys, tmp_path)
    after = tuple(sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")))
    assert after == before


def test_model_inventory_missing_root_is_read_only_and_never_trains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, keys = _keys()
    data = MaterializedScreeningData(
        manifests=ScreeningDataManifests(evidence={}, views={}),
        evidence_cost_ids={},
        view_cost_ids={},
    )
    model_keys = ScreeningModelKeys(models={})
    before = tuple(sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")))

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("model inventory attempted training, writing, or inference")

    monkeypatch.setattr(models, "_train", forbidden)
    monkeypatch.setattr(models, "write_training_artifact", forbidden)
    with pytest.raises(TrainingDataArtifactError, match="existing directory"):
        models.load_screening_model_inventory(
            config, keys, data, model_keys, tmp_path / "missing"
        )
    after = tuple(sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")))
    assert after == before


def _fake_payload() -> object:
    return SimpleNamespace(
        samples=tuple(
            SimpleNamespace(
                affordances=SimpleNamespace(sample_counts={"a": 64}),
                trace=SimpleNamespace(transitions=(object(),) * (5 + index % 4)),
            )
            for index in range(40)
        )
    )


def _canonical_evidence_accounting(payload: object) -> TrainingPreparationAccounting:
    replay_actions = 2 * sum(
        len(sample.trace.transitions) for sample in payload.samples
    )
    return TrainingPreparationAccounting(
        setup=PhaseAccounting(calls=40, wall_seconds=0.1),
        training_probes=PhaseAccounting(
            calls=160,
            actions=2560,
            environment_steps=2560,
            resets=160,
            wall_seconds=0.2,
        ),
        reference_replay=PhaseAccounting(
            calls=40,
            actions=replay_actions,
            environment_steps=replay_actions,
            resets=80,
            wall_seconds=0.3,
        ),
        serialization=PhaseAccounting(calls=1),
    )


def test_evidence_accounting_is_recomputed_from_sanitized_payload() -> None:
    config, _ = _keys()
    payload = _fake_payload()
    canonical = _canonical_evidence_accounting(payload)
    preparation._validate_evidence_accounting(config, payload, canonical)

    coherent_tamper = canonical.model_copy(
        update={
            "reference_replay": canonical.reference_replay.model_copy(
                update={
                    "actions": canonical.reference_replay.actions + 2,
                    "environment_steps": canonical.reference_replay.environment_steps + 2,
                }
            )
        }
    )
    with pytest.raises(TrainingDataArtifactError, match="not canonical"):
        preparation._validate_evidence_accounting(config, payload, coherent_tamper)


def test_model_report_is_derived_from_architecture_payload_and_frozen_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_id = "a" * 64
    key_id = "b" * 64
    artifact_dir = tmp_path / "training-artifacts" / artifact_id
    (artifact_dir / "tensors").mkdir(parents=True)
    (artifact_dir / "manifest.json").write_bytes(b"{}")
    key = SimpleNamespace(
        key_id=key_id,
        condition_id=models.B1,
        training_tuple_id="tuple",
        objective_id="optimum_frequency",
        training_data_sha256="c" * 64,
    )
    data_manifest = SimpleNamespace(artifact_id="c" * 64)
    model = torch.nn.Linear(2, 1)
    model.eval()
    tampered_report = TrainingReportMetadata(
        trainable_parameters=3,
        training_examples=7,
        optimizer_steps=120,
        forward_passes=121,
    )
    manifest = SimpleNamespace(
        artifact_id=artifact_id,
        key=key,
        model_id="global_affordance_mlp_frequency_v1",
        report=tampered_report,
    )
    accounting = TrainingPreparationAccounting(
        setup=PhaseAccounting(calls=1, wall_seconds=0.1),
        training=PhaseAccounting(
            calls=1,
            optimizer_steps=120,
            forward_passes=121,
            wall_seconds=0.2,
        ),
        serialization=PhaseAccounting(calls=1),
    )
    cost = SimpleNamespace(artifact_id=artifact_id, key_id=key_id, accounting=accounting)
    monkeypatch.setattr(models, "load_training_key_index", lambda *args: SimpleNamespace(artifact_id=artifact_id))
    monkeypatch.setattr(models, "load_training_cost", lambda *args: cost)
    monkeypatch.setattr(models, "load_training_model", lambda *args, **kwargs: (model, manifest))
    monkeypatch.setattr(models, "learner_samples", lambda payload: payload)
    monkeypatch.setattr(
        models,
        "global_frequency_optimum_examples",
        lambda samples: (torch.zeros((7, 1)), torch.zeros(7)),
    )
    monkeypatch.setattr(
        models,
        "_training_parameters",
        lambda *args: SimpleNamespace(epochs=120),
    )
    monkeypatch.setattr(
        model,
        "forward",
        lambda *args: pytest.fail("forward inference ran"),
    )
    with pytest.raises(TrainingDataArtifactError, match="canonically derived"):
        models._load_one_readonly(
            tmp_path,
            SimpleNamespace(),
            key,
            data_manifest,
            object(),
        )
