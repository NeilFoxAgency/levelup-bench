from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase2_screening_models as models
import levelup.experiments.milestone6_phase2_screening_runtime as runtime
from levelup.experiments.milestone6_phase2_screening import (
    B1,
    B2,
    C,
    build_screening_child_config,
)
from levelup.experiments.milestone6_phase2_screening_preparation import (
    MaterializedScreeningData,
    ScreeningDataManifests,
    build_screening_data_keys,
    build_screening_model_keys,
    build_screening_shared_plan,
    materialize_screening_data,
)
from levelup.experiments.milestone6_phase2_screening_readiness import _child_manifest
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import (
    canonical_json_bytes,
    run_id_for,
    scientific_config_value,
)
from levelup.experiments.runner.records import (
    PhaseAccounting,
    ResourceAccounting,
    SystemProvenance,
)
from levelup.experiments.runner.secure_fs import (
    open_child_directory,
    open_directory_chain,
)
from levelup.experiments.runner.selection_metric import within_parameter_tolerance
from levelup.experiments.runner.storage import RunStore
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataArtifactError,
    TrainingDataArtifactManifest,
    TrainingDataEvidenceManifest,
)
from levelup.learning.state_conditioned import TrainingReport

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
BASES = (B1, B2, C)
TRAINING_TUPLES = (
    "lr0p003-e120",
    "lr0p003-e180",
    "lr0p01-e120",
    "lr0p01-e180",
)


def _digest(*parts: object) -> str:
    value: object = parts[0] if len(parts) == 1 else parts
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fake_data_manifests(config, data_keys) -> ScreeningDataManifests:
    evidence = {}
    views = {}
    family = config.parameters["heldout_family_id"]
    for replicate, key in data_keys.evidence.items():
        body = {
            "schema_version": "runner.training-data-evidence.v1",
            "evidence_key_id": key.key_id,
            "key": key.model_dump(mode="json"),
            "payload_sha256": _digest("payload", family, replicate),
            "payload_bytes": 1,
            "sample_task_ids": key.ordered_training_task_ids,
        }
        evidence[replicate] = TrainingDataEvidenceManifest(
            evidence_id=_digest(body),
            **body,
        )
    for identity, key in data_keys.views.items():
        evidence_manifest = evidence[identity[1]]
        body = {
            "schema_version": "runner.training-data-manifest.v1",
            "evidence_id": evidence_manifest.evidence_id,
            "key_id": key.key_id,
            "key": key.model_dump(mode="json"),
            "payload_sha256": evidence_manifest.payload_sha256,
            "payload_bytes": evidence_manifest.payload_bytes,
            "sample_task_ids": key.ordered_training_task_ids,
        }
        views[identity] = TrainingDataArtifactManifest(
            artifact_id=_digest(body),
            **body,
        )
    return ScreeningDataManifests(evidence=evidence, views=views)


def _fake_preparation():
    config = build_screening_child_config("plain")
    data_keys = build_screening_data_keys(config, PROVENANCE)
    manifests = _fake_data_manifests(config, data_keys)
    model_keys = build_screening_model_keys(config, data_keys, manifests)
    data = MaterializedScreeningData(
        manifests=manifests,
        evidence_cost_ids={replicate: _digest("evidence-cost", replicate) for replicate in range(5)},
        view_cost_ids={identity: _digest("view-cost", *identity) for identity in data_keys.views},
    )
    return config, data_keys, data, model_keys


def _fake_train(base, payload, key, training):
    model = models._model_factory(key.backbone_id)
    report = TrainingReport(
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        optimizer_steps=training.epochs,
        forward_passes=training.epochs * (1 if base == B1 else 2),
        training_examples=7,
    )
    accounting = ResourceAccounting(
        setup=PhaseAccounting(calls=1),
        training=PhaseAccounting(
            calls=1,
            forward_passes=report.forward_passes,
            optimizer_steps=report.optimizer_steps,
            wall_seconds=0.001,
        ),
        serialization=PhaseAccounting(calls=1),
    )
    return model, report, accounting


def test_model_factory_has_actual_symmetric_capacity_match() -> None:
    b2_parameters = sum(parameter.numel() for parameter in models._model_factory(
        "global_affordance_mlp_listwise_v1"
    ).parameters())
    c_parameters = sum(parameter.numel() for parameter in models._model_factory(
        "state_conditioned_mlp_listwise_v1"
    ).parameters())
    assert (b2_parameters, c_parameters) == (3601, 3841)
    assert within_parameter_tolerance(b2_parameters, c_parameters, tolerance=0.1)
    assert within_parameter_tolerance(c_parameters, b2_parameters, tolerance=0.1)


def test_model_inventory_and_temperature_reuse_are_exact() -> None:
    config, data_keys, data, model_keys = _fake_preparation()
    assert len(model_keys.models) == 60
    assert set(model_keys.models) == {
        (base, training_tuple, replicate)
        for base in BASES
        for training_tuple in TRAINING_TUPLES
        for replicate in range(5)
    }
    for base in BASES:
        for training_tuple in TRAINING_TUPLES:
            consumers = tuple(
                condition
                for condition in config.conditions
                if condition.parameters.get("base_condition_id") == base
                and condition.parameters.get("training_tuple_id") == training_tuple
            )
            assert len(consumers) == 3
            assert {item.parameters["search_temperature"] for item in consumers} == {
                0.6,
                0.9,
                1.2,
            }
            assert all(
                "search_temperature" not in model_keys.models[
                    (base, training_tuple, replicate)
                ].model_dump(mode="json")
                for replicate in range(5)
            )
            assert all(
                model_keys.models[(base, training_tuple, replicate)].training_data_sha256
                == data.manifests.views[(base, replicate)].artifact_id
                for replicate in range(5)
            )
    assert data_keys.provenance == PROVENANCE


def test_b1_is_reported_as_unmatched_frequency_baseline() -> None:
    config, _, _, model_keys = _fake_preparation()
    assert {
        model_keys.models[(base, TRAINING_TUPLES[0], 0)].objective_id for base in BASES
    } == {"optimum_frequency", "listwise_optimum"}
    assert config.conditions[2].parameters["base_condition_id"] == B1
    assert model_keys.models[(B1, TRAINING_TUPLES[0], 0)].backbone_id != model_keys.models[
        (C, TRAINING_TUPLES[0], 0)
    ].backbone_id


def test_fake_materialization_writes_exact_60_lineage_and_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, data_keys, data, model_keys = _fake_preparation()
    monkeypatch.setattr(models, "_validate_data", lambda *args: {
        identity: object() for identity in data_keys.views
    })
    monkeypatch.setattr(models, "_train", _fake_train)
    monkeypatch.setattr(models, "_validate_matched_pairs", lambda *args: None)
    persisted = {}

    def fake_write(root, *, key, model_id, model, accounting, report):
        artifact_id = _digest("artifact", key.key_id)
        manifest = SimpleNamespace(
            artifact_id=artifact_id,
            key=key,
            model_id=model_id,
            report=report,
        )
        cost = SimpleNamespace(
            cost_id=_digest("cost", key.key_id),
            key_id=key.key_id,
            artifact_id=artifact_id,
            accounting=accounting,
        )
        persisted[key.key_id] = (manifest, cost, models.ModelComputeReport(
            model_id=model_id,
            objective_id=key.objective_id,
            trainable_parameters=report.trainable_parameters,
            training_examples=report.training_examples,
            optimizer_steps=report.optimizer_steps,
            forward_passes=report.forward_passes,
            training_wall_seconds=accounting.training.wall_seconds,
        ))
        return manifest

    def fake_preflight(root, config, keys, manifests):
        return {
            identity: persisted[key.key_id]
            for identity, key in keys.models.items()
            if key.key_id in persisted
        }

    monkeypatch.setattr(models, "write_training_artifact", fake_write)
    monkeypatch.setattr(models, "_preflight", fake_preflight)
    events: list[str] = []
    materialized = models.materialize_screening_models(
        config, data_keys, data, model_keys, tmp_path, event=events.append
    )
    assert len(materialized.manifests) == len(materialized.costs) == len(materialized.compute) == 60
    assert len({cost.cost_id for cost in materialized.costs.values()}) == 60
    assert sum(item.startswith(f"model_train:{B1}:lr0p003-e120:0") for item in events) == 1
    assert sum(item.startswith("model_train:") for item in events) == 60
    for identity, manifest in materialized.manifests.items():
        key = model_keys.models[identity]
        assert manifest.key == key
        assert key.training_data_sha256 == data.manifests.views[(identity[0], identity[2])].artifact_id
        cost = materialized.costs[identity]
        assert cost.accounting.training.calls == 1
        assert cost.accounting.search == PhaseAccounting()
        assert cost.accounting.evaluator == PhaseAccounting()
        assert cost.accounting.probes == PhaseAccounting()
        assert cost.accounting.replay == PhaseAccounting()
    for tuple_id in TRAINING_TUPLES:
        for replicate in range(5):
            b1 = materialized.compute[(B1, tuple_id, replicate)]
            b2 = materialized.compute[(B2, tuple_id, replicate)]
            assert b1.trainable_parameters == b2.trainable_parameters == 3601
            assert b1.optimizer_steps == b2.optimizer_steps
            assert b1.forward_passes != b2.forward_passes


def test_clean_resume_loads_all_models_without_retraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, data_keys, data, model_keys = _fake_preparation()
    monkeypatch.setattr(models, "_validate_data", lambda *args: {
        identity: object() for identity in data_keys.views
    })
    monkeypatch.setattr(models, "_train", _fake_train)
    monkeypatch.setattr(models, "_validate_matched_pairs", lambda *args: None)
    first = models.materialize_screening_models(config, data_keys, data, model_keys, tmp_path)
    monkeypatch.setattr(models, "_train", lambda *args: pytest.fail("resume retrained a model"))
    events: list[str] = []
    second = models.materialize_screening_models(
        config, data_keys, data, model_keys, tmp_path, event=events.append
    )
    assert second == first
    assert len(events) == 60
    assert all(item.startswith("model_loaded:") for item in events)
    first_manifest = next(iter(first.manifests.values()))
    unexpected = (
        tmp_path
        / "training-artifacts"
        / first_manifest.artifact_id
        / "unexpected.json"
    )
    unexpected.write_text("{}", encoding="utf-8")
    with pytest.raises(TrainingDataArtifactError, match="artifact inventory"):
        models.materialize_screening_models(
            config,
            data_keys,
            data,
            model_keys,
            tmp_path,
        )


def test_interrupted_intent_blocks_before_second_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, data_keys, data, model_keys = _fake_preparation()
    monkeypatch.setattr(models, "_validate_data", lambda *args: {
        identity: object() for identity in data_keys.views
    })
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("injected interruption")

    monkeypatch.setattr(models, "_train", interrupted)
    with pytest.raises(TrainingDataArtifactError):
        models.materialize_screening_models(config, data_keys, data, model_keys, tmp_path)
    with pytest.raises(Exception):
        models.materialize_screening_models(config, data_keys, data, model_keys, tmp_path)
    assert calls == 1


def test_concurrent_loser_or_orphan_or_symlink_fails_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, data_keys, data, model_keys = _fake_preparation()
    monkeypatch.setattr(models, "_validate_data", lambda *args: {
        identity: object() for identity in data_keys.views
    })
    monkeypatch.setattr(models, "_claim_intent", lambda *args: False)
    monkeypatch.setattr(models, "_train", lambda *args: pytest.fail("loser trained"))
    with pytest.raises(TrainingDataArtifactError, match="claimed concurrently"):
        models.materialize_screening_models(config, data_keys, data, model_keys, tmp_path)

    for name in ("training-artifacts", "training-artifact-keys", "training-artifact-costs"):
        root = tmp_path / name
        root.mkdir(parents=True, exist_ok=True)
        (root / "staging").mkdir()
        with pytest.raises(TrainingDataArtifactError):
            models.materialize_screening_models(config, data_keys, data, model_keys, tmp_path)
        for child in root.iterdir():
            if child.is_dir():
                child.rmdir()
        root.rmdir()

    external = tmp_path.parent / "screening-model-external"
    external.mkdir(exist_ok=True)
    (tmp_path / "screening-model-intents").symlink_to(external, target_is_directory=True)
    with pytest.raises(TrainingDataArtifactError):
        models.materialize_screening_models(config, data_keys, data, model_keys, tmp_path)


def test_preflight_accepts_canonical_json_intent_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, data_keys, data, model_keys = _fake_preparation()
    identity, key = next(iter(model_keys.models.items()))
    intent = models._intent_path(tmp_path, key.key_id)
    intent.parent.mkdir(parents=True)
    intent.write_bytes(
        canonical_json_bytes(
            models._intent_body(config, key, data.manifests.views[(identity[0], identity[2])])
        )
        + b"\n"
    )
    fake_manifest = SimpleNamespace(artifact_id=_digest("artifact"), key=key)
    fake_cost = SimpleNamespace(
        artifact_id=fake_manifest.artifact_id,
        key_id=key.key_id,
        accounting=ResourceAccounting(),
    )
    fake_compute = models.ModelComputeReport(
        model_id=key.backbone_id,
        objective_id=key.objective_id,
        trainable_parameters=3601,
        training_examples=1,
        optimizer_steps=1,
        forward_passes=1,
        training_wall_seconds=0.0,
    )
    monkeypatch.setattr(
        models,
        "_load_one",
        lambda *args: (fake_manifest, fake_cost, fake_compute),
    )
    for name in ("training-artifact-keys", "training-artifact-costs"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / f"{key.key_id}.json").write_text("{}", encoding="utf-8")
    (tmp_path / "training-artifacts" / fake_manifest.artifact_id).mkdir(parents=True)
    loaded = models._preflight(tmp_path, config, model_keys, data.manifests)
    assert loaded[identity][0] is fake_manifest


def test_real_b1_b2_c_tuple_uses_same_data_and_objective_matched_budget(
    tmp_path: Path,
) -> None:
    config = build_screening_child_config("plain")
    data_keys = build_screening_data_keys(config, PROVENANCE)
    data = materialize_screening_data(config, data_keys, tmp_path)
    model_keys = build_screening_model_keys(config, data_keys, data.manifests)
    payloads = models._validate_data(tmp_path, data_keys, data)
    tuple_id = TRAINING_TUPLES[0]
    reports = {}
    for base in BASES:
        key = model_keys.models[(base, tuple_id, 0)]
        _, report, accounting = models._train(
            base,
            payloads[(base, 0)],
            key,
            models._training_parameters(config, base, tuple_id),
        )
        reports[base] = report
        assert report.trainable_parameters == sum(
            parameter.numel() for parameter in models._model_factory(key.backbone_id).parameters()
        )
        assert accounting.training.calls == 1
        assert accounting.search == PhaseAccounting()
        assert accounting.evaluator == PhaseAccounting()
    assert reports[B2].training_examples == reports[C].training_examples
    assert reports[B2].optimizer_steps == reports[C].optimizer_steps
    assert reports[B2].forward_passes == reports[C].forward_passes
    assert within_parameter_tolerance(
        reports[B2].trainable_parameters, reports[C].trainable_parameters, tolerance=0.1
    )
    assert reports[B1].training_examples != 0


def test_real_fold_model_load_stays_on_detached_pinned_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_screening_child_config("plain")
    raw_root = tmp_path / "raw"
    run_dir = raw_root / run_id_for(config)
    data_keys = build_screening_data_keys(config, PROVENANCE)
    data = materialize_screening_data(config, data_keys, run_dir)
    model_keys = build_screening_model_keys(config, data_keys, data.manifests)
    expected = models.materialize_screening_models(
        config, data_keys, data, model_keys, run_dir
    )
    shared = build_screening_shared_plan(config, data_keys, data.manifests, model_keys)
    store = RunStore(
        raw_root,
        config,
        repository=tmp_path,
        shared_artifacts=tuple(shared.artifacts),
    )
    (run_dir / "units").mkdir()
    (run_dir / "attempts").mkdir()
    for name, value in (
        ("config.json", scientific_config_value(config)),
        ("expected-units.json", store.expected.model_dump(mode="json")),
        (
            "expected-shared-artifacts.json",
            store.expected_shared.model_dump(mode="json"),
        ),
        ("provenance.json", PROVENANCE.model_dump(mode="json")),
    ):
        (run_dir / name).write_bytes(canonical_json_bytes(value) + b"\n")
    child_manifest = _child_manifest(
        config,
        data_keys,
        data,
        model_keys,
        expected,
        shared,
        PROVENANCE,
    )
    real_data_loader = runtime.load_screening_data_inventory_at
    sentinel: Path | None = None

    def detach_after_real_data_load(*args, **kwargs):
        nonlocal sentinel
        loaded = real_data_loader(*args, **kwargs)
        detached = tmp_path / "detached-child"
        os.rename(run_dir, detached)
        run_dir.mkdir()
        sentinel = run_dir / "external-sentinel"
        sentinel.write_text("must never be read", encoding="utf-8")
        return loaded

    monkeypatch.setattr(
        runtime, "load_screening_data_inventory_at", detach_after_real_data_load
    )
    raw_fd = open_directory_chain(raw_root)
    try:
        child_fd = open_child_directory(raw_fd, child_manifest.run_id)
        try:
            child_identity = secure_fs.directory_identity(child_fd)
        finally:
            os.close(child_fd)
        loaded = runtime._load_fold(
            config,
            child_manifest,
            raw_root,
            raw_fd,
            child_identity,
            tmp_path,
            PROVENANCE,
        )
    finally:
        os.close(raw_fd)

    assert loaded.models == expected
    assert sentinel is not None
    assert sentinel.read_text(encoding="utf-8") == "must never be read"
