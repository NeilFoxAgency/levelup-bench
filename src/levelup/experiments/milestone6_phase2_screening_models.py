"""Materialize the frozen development-screening model artifacts.

This module is deliberately a preparation boundary.  It consumes only the already
materialized learner-visible training views, trains the three declared development
baselines, and stores immutable model artifacts.  It never performs search, replay of
held-out tasks, evaluation, oracle lookup, or final-family access.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from levelup.experiments.milestone6_phase2_screening import (
    B1,
    B2,
    C,
    validate_screening_child_config,
)
from levelup.experiments.milestone6_phase2_screening_preparation import (
    MaterializedScreeningData,
    ScreeningDataKeys,
    ScreeningDataManifests,
    ScreeningModelKeys,
    build_screening_data_keys,
    build_screening_model_keys,
    load_screening_data_inventory,
)
from levelup.experiments.runner import within_parameter_tolerance
from levelup.experiments.runner.config import (
    ExperimentConfig,
    canonical_json_bytes,
    run_id_for,
    scientific_config_sha256,
)
from levelup.experiments.runner.records import (
    PhaseAccounting,
    ResourceAccounting,
    TrainingArtifactCostRecord,
    TrainingPreparationAccounting,
)
from levelup.experiments.runner.training_artifacts import (
    TrainingArtifactKey,
    TrainingArtifactManifest,
    TrainingReportMetadata,
    load_training_cost,
    load_training_key_index,
    load_training_model,
    write_training_artifact,
)
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataArtifactError,
    TrainingDataArtifactManifest,
    learner_samples,
    load_training_data_artifact,
    load_training_data_evidence_cost,
    load_training_data_view_cost,
)
from levelup.learning.state_conditioned import (
    TrainingReport,
    TrainingSpec,
    global_frequency_optimum_examples,
    global_listwise_optimum_examples,
    optimum_imitation_examples,
    train_global_frequency_optimum_model,
    train_global_listwise_optimum_model,
    train_state_conditioned_optimum_model,
)

PreparationEvent = Callable[[str], None]
ModelIdentity = tuple[str, str, int]


@dataclass(frozen=True, slots=True)
class ModelComputeReport:
    """Typed, persisted training accounting exposed for screening audits."""

    model_id: str
    objective_id: str
    trainable_parameters: int
    training_examples: int
    optimizer_steps: int
    forward_passes: int
    training_wall_seconds: float


@dataclass(frozen=True, slots=True)
class MaterializedScreeningModels:
    """The exact 60 model manifests and their first-writer compute reports."""

    manifests: dict[ModelIdentity, TrainingArtifactManifest]
    costs: dict[ModelIdentity, TrainingArtifactCostRecord]
    compute: dict[ModelIdentity, ModelComputeReport]
    b1_compute: dict[ModelIdentity, ModelComputeReport]


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _model_factory(model_id: str) -> torch.nn.Module:
    from levelup.learning.state_conditioned import (
        GlobalAffordanceScorer,
        StateConditionedScorer,
    )

    if model_id in {
        "global_affordance_mlp_frequency_v1",
        "global_affordance_mlp_listwise_v1",
    }:
        return GlobalAffordanceScorer()
    if model_id == "state_conditioned_mlp_listwise_v1":
        return StateConditionedScorer()
    raise ValueError("unsupported screening model ID")


def _key_identity(key: TrainingArtifactKey) -> ModelIdentity:
    return (key.condition_id, key.training_tuple_id, key.replicate)


def _expected_model_id(base: str) -> str:
    return {
        B1: "global_affordance_mlp_frequency_v1",
        B2: "global_affordance_mlp_listwise_v1",
        C: "state_conditioned_mlp_listwise_v1",
    }[base]


def _intent_path(run_dir: Path, key_id: str) -> Path:
    return run_dir / "screening-model-intents" / f"{key_id}.json"


def _intent_body(
    config: ExperimentConfig,
    key: TrainingArtifactKey,
    data_manifest: TrainingDataArtifactManifest,
) -> dict[str, Any]:
    return {
        "schema_version": "milestone6.screening-model-intent.v1",
        "run_id": run_id_for(config),
        "config_sha256": scientific_config_sha256(config),
        "key_id": key.key_id,
        "model_id": key.backbone_id,
        "condition_id": key.condition_id,
        "training_tuple_id": key.training_tuple_id,
        "replicate": key.replicate,
        "training_data_sha256": data_manifest.artifact_id,
    }


def _claim_intent(path: Path, body: dict[str, Any]) -> bool:
    """Atomically claim one model; an existing claim must match byte-for-byte."""

    payload = canonical_json_bytes(body) + b"\n"
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.parent.is_symlink():
            raise TrainingDataArtifactError("screening model intent is unsafe")
        if path.read_bytes() != payload:
            raise TrainingDataArtifactError("screening model intent conflicts")
        return False
    if path.parent.is_symlink() or path.parent.parent.is_symlink():
        raise TrainingDataArtifactError("screening model intent path is unsafe")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise TrainingDataArtifactError("screening model intent directory is unsafe")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise TrainingDataArtifactError("concurrent screening model intent conflicts")
            return False
        return True
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _check_inventory(
    run_dir: Path,
    expected: dict[str, tuple[set[str], str]],
) -> None:
    for name, (names, kind) in expected.items():
        directory = run_dir / name
        if not names:
            if os.path.lexists(directory):
                raise TrainingDataArtifactError("screening model namespace is partial")
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise TrainingDataArtifactError("screening model namespace is unsafe")
        entries = tuple(directory.iterdir())
        if {item.name for item in entries} != names:
            raise TrainingDataArtifactError("screening model namespace inventory drifted")
        for item in entries:
            if item.is_symlink() or (kind == "file" and not item.is_file()) or (
                kind == "directory" and not item.is_dir()
            ):
                raise TrainingDataArtifactError("screening model namespace contains a link or wrong type")


def _validate_data(
    run_dir: Path,
    data_keys: ScreeningDataKeys,
    data: MaterializedScreeningData,
) -> dict[tuple[str, int], Any]:
    manifests = data.manifests
    if set(manifests.evidence) != set(data_keys.evidence) or set(manifests.views) != set(
        data_keys.views
    ):
        raise TrainingDataArtifactError("screening data manifest inventory is not exact")
    payloads: dict[tuple[str, int], Any] = {}
    for identity, key in data_keys.views.items():
        expected = manifests.views[identity]
        view_cost = load_training_data_view_cost(run_dir, key)
        if view_cost.cost_id != data.view_cost_ids.get(identity):
            raise TrainingDataArtifactError("screening view cost identity drifted")
        loaded, payload = load_training_data_artifact(run_dir, expected_key=key)
        if loaded != expected:
            raise TrainingDataArtifactError("screening view manifest reload drifted")
        payloads[identity] = payload
    for replicate, key in data_keys.evidence.items():
        evidence_cost = load_training_data_evidence_cost(run_dir, key)
        if evidence_cost.cost_id != data.evidence_cost_ids.get(replicate):
            raise TrainingDataArtifactError("screening evidence cost inventory is incomplete")
    return payloads


def _training_parameters(config: ExperimentConfig, base: str, tuple_id: str) -> TrainingSpec:
    rows = tuple(
        condition
        for condition in config.conditions
        if condition.parameters.get("base_condition_id") == base
        and condition.parameters.get("training_tuple_id") == tuple_id
    )
    if len(rows) != 3 or {row.parameters.get("search_temperature") for row in rows} != {
        0.6,
        0.9,
        1.2,
    }:
        raise ValueError("screening model training tuple is not temperature-complete")
    learning_rates = {row.parameters.get("learning_rate") for row in rows}
    epochs = {row.parameters.get("training_epochs") for row in rows}
    if len(learning_rates) != 1 or len(epochs) != 1:
        raise ValueError("screening model training tuple hyperparameters drifted")
    return TrainingSpec(
        epochs=int(next(iter(epochs))),
        learning_rate=float(next(iter(learning_rates))),
        weight_decay=float(config.parameters["weight_decay"]),
    )


def _train(
    base: str,
    payload: Any,
    key: TrainingArtifactKey,
    training: TrainingSpec,
) -> tuple[torch.nn.Module, TrainingReport, ResourceAccounting]:
    samples = learner_samples(payload)
    setup_started = time.perf_counter()
    if base == B1:
        features, targets = global_frequency_optimum_examples(samples)
        example_count = int(features.shape[0])
        setup_wall = time.perf_counter() - setup_started
        train_started = time.perf_counter()
        model, report = train_global_frequency_optimum_model(
            features, targets, training=training, model_seed=key.model_seed
        )
    elif base == B2:
        examples = global_listwise_optimum_examples(samples)
        example_count = len(examples)
        setup_wall = time.perf_counter() - setup_started
        train_started = time.perf_counter()
        model, report = train_global_listwise_optimum_model(
            examples, training=training, model_seed=key.model_seed
        )
    elif base == C:
        examples = optimum_imitation_examples(samples)
        example_count = len(examples)
        setup_wall = time.perf_counter() - setup_started
        train_started = time.perf_counter()
        model, report = train_state_conditioned_optimum_model(
            examples, training=training, model_seed=key.model_seed
        )
    else:
        raise ValueError("unsupported screening baseline")
    training_wall = time.perf_counter() - train_started
    if report.training_examples != example_count:
        raise RuntimeError("training report example count drifted")
    accounting = ResourceAccounting(
        setup=PhaseAccounting(calls=1, wall_seconds=setup_wall),
        training=PhaseAccounting(
            calls=1,
            forward_passes=report.forward_passes,
            optimizer_steps=report.optimizer_steps,
            wall_seconds=training_wall,
        ),
        serialization=PhaseAccounting(calls=1),
    )
    return model, report, accounting


def _load_one(
    run_dir: Path,
    config: ExperimentConfig,
    key: TrainingArtifactKey,
    data_manifest: TrainingDataArtifactManifest,
) -> tuple[TrainingArtifactManifest, TrainingArtifactCostRecord, ModelComputeReport]:
    index = load_training_key_index(run_dir, key)
    cost = load_training_cost(run_dir, key)
    _, manifest = load_training_model(
        run_dir, index.artifact_id, expected_key=key, model_factory=_model_factory
    )
    artifact_dir = run_dir / "training-artifacts" / manifest.artifact_id
    artifact_entries = tuple(artifact_dir.iterdir())
    if (
        artifact_dir.is_symlink()
        or {item.name for item in artifact_entries} != {"manifest.json", "tensors"}
        or any(item.is_symlink() for item in artifact_entries)
        or not (artifact_dir / "manifest.json").is_file()
        or not (artifact_dir / "tensors").is_dir()
    ):
        raise TrainingDataArtifactError("screening model artifact inventory drifted")
    if manifest.model_id != _expected_model_id(key.condition_id) or manifest.key != key:
        raise TrainingDataArtifactError("screening model manifest identity drifted")
    if cost.artifact_id != manifest.artifact_id or cost.key_id != key.key_id:
        raise TrainingDataArtifactError("screening model cost lineage drifted")
    if not isinstance(cost.accounting, TrainingPreparationAccounting):
        raise TrainingDataArtifactError("screening model cost has the wrong preparation schema")
    accounting = cost.accounting
    expected_training = PhaseAccounting(
        calls=1,
        optimizer_steps=manifest.report.optimizer_steps,
        forward_passes=manifest.report.forward_passes,
        wall_seconds=accounting.training.wall_seconds,
    )
    expected_setup = PhaseAccounting(
        calls=1,
        wall_seconds=accounting.setup.wall_seconds,
    )
    expected_serialization = PhaseAccounting(
        calls=1,
        wall_seconds=accounting.serialization.wall_seconds,
    )
    if accounting.training != expected_training:
        raise TrainingDataArtifactError("screening model training accounting drifted")
    if accounting.setup != expected_setup or accounting.serialization != expected_serialization:
        raise TrainingDataArtifactError("screening model preparation accounting drifted")
    if (
        accounting.training_probes != PhaseAccounting()
        or accounting.reference_replay != PhaseAccounting()
    ):
        raise TrainingDataArtifactError("screening model includes unearned interaction cost")
    if key.training_data_sha256 != data_manifest.artifact_id:
        raise TrainingDataArtifactError("screening model references the wrong training view")
    return manifest, cost, ModelComputeReport(
        model_id=manifest.model_id,
        objective_id=key.objective_id,
        trainable_parameters=manifest.report.trainable_parameters,
        training_examples=manifest.report.training_examples,
        optimizer_steps=manifest.report.optimizer_steps,
        forward_passes=manifest.report.forward_passes,
        training_wall_seconds=accounting.training.wall_seconds,
    )


def _preflight(
    run_dir: Path,
    config: ExperimentConfig,
    model_keys: ScreeningModelKeys,
    data_manifests: ScreeningDataManifests,
) -> dict[ModelIdentity, tuple[TrainingArtifactManifest, TrainingArtifactCostRecord, ModelComputeReport]]:
    expected_by_id = {_key.key_id: (_identity, _key) for _identity, _key in model_keys.models.items()}
    intent_root = run_dir / "screening-model-intents"
    claimed: dict[str, Path] = {}
    if os.path.lexists(intent_root):
        if intent_root.is_symlink() or not intent_root.is_dir():
            raise TrainingDataArtifactError("screening model intent namespace is unsafe")
        for path in intent_root.iterdir():
            if path.name not in {f"{key_id}.json" for key_id in expected_by_id}:
                raise TrainingDataArtifactError("screening model intent inventory drifted")
            if path.is_symlink() or not path.is_file():
                raise TrainingDataArtifactError("screening model intent contains an unsafe entry")
            claimed[path.stem] = path
    loaded: dict[ModelIdentity, tuple[TrainingArtifactManifest, TrainingArtifactCostRecord, ModelComputeReport]] = {}
    for key_id, path in claimed.items():
        identity, key = expected_by_id[key_id]
        manifest = data_manifests.views[(identity[0], identity[2])]
        if path.read_bytes() != canonical_json_bytes(_intent_body(config, key, manifest)) + b"\n":
            raise TrainingDataArtifactError("screening model intent content drifted")
        loaded[identity] = _load_one(run_dir, config, key, manifest)
    completed_ids = {item[0].artifact_id for item in loaded.values()}
    completed_keys = {f"{key_id}.json" for key_id in claimed}
    _check_inventory(
        run_dir,
        {
            "screening-model-intents": (completed_keys, "file"),
            "training-artifact-keys": (completed_keys, "file"),
            "training-artifact-costs": (completed_keys, "file"),
            "training-artifacts": (completed_ids, "directory"),
        },
    )
    return loaded


def _validate_matched_pairs(
    reports: dict[ModelIdentity, ModelComputeReport],
    payloads: dict[tuple[str, int], Any],
    model_keys: ScreeningModelKeys,
) -> None:
    for replicate in range(5):
        for tuple_id in {
            identity[1] for identity in model_keys.models if identity[2] == replicate
        }:
            b2_id = (B2, tuple_id, replicate)
            c_id = (C, tuple_id, replicate)
            b1_id = (B1, tuple_id, replicate)
            b2_examples = global_listwise_optimum_examples(
                learner_samples(payloads[(B2, replicate)])
            )
            c_examples = optimum_imitation_examples(learner_samples(payloads[(C, replicate)]))
            b2 = reports[b2_id]
            c = reports[c_id]
            b1 = reports[b1_id]
            if (
                b2.training_examples != c.training_examples
                or b2.optimizer_steps != c.optimizer_steps
                or b2.forward_passes != c.forward_passes
                or len(b2_examples) != len(c_examples)
                or _digest([(x.selected_index, int(x.candidate_features.shape[0])) for x in b2_examples])
                != _digest([(x.selected_index, int(x.candidate_features.shape[0])) for x in c_examples])
            ):
                raise TrainingDataArtifactError("B2/C matched listwise training budget drifted")
            same_affordances = all(
                torch.equal(b2_example.candidate_features, c_example.candidate_features[:, 5:])
                for b2_example, c_example in zip(b2_examples, c_examples)
            )
            if not same_affordances:
                raise TrainingDataArtifactError("B2/C affordance rows are not same-data matched")
            if not within_parameter_tolerance(
                b2.trainable_parameters, c.trainable_parameters, tolerance=0.1
            ):
                raise TrainingDataArtifactError("B2/C capacity tolerance failed")
            if (
                b1.optimizer_steps != b2.optimizer_steps
                or b1.trainable_parameters != b2.trainable_parameters
            ):
                raise TrainingDataArtifactError(
                    "B1 continuity baseline received a smaller update or capacity budget"
                )


def materialize_screening_models(
    config: ExperimentConfig,
    data_keys: ScreeningDataKeys,
    data: MaterializedScreeningData,
    model_keys: ScreeningModelKeys,
    run_dir: str | Path,
    *,
    event: PreparationEvent | None = None,
) -> MaterializedScreeningModels:
    """Train or load exactly 60 development-screening models, fail-closed."""

    validate_screening_child_config(config)
    if config.split.final_tasks:
        raise ValueError("screening model materialization cannot receive final tasks")
    if data_keys != build_screening_data_keys(config, data_keys.provenance):
        raise ValueError("screening model materialization received noncanonical data keys")
    expected_model_keys = build_screening_model_keys(config, data_keys, data.manifests)
    if model_keys != expected_model_keys or len(model_keys.models) != 60:
        raise ValueError("screening model-key inventory is not the frozen 60-model matrix")
    payloads = _validate_data(Path(run_dir), data_keys, data)
    root = Path(run_dir)
    if root.is_symlink():
        raise TrainingDataArtifactError("screening model run root cannot be a symlink")
    loaded = _preflight(root, config, model_keys, data.manifests)
    for identity, key in model_keys.models.items():
        if identity in loaded:
            if event is not None:
                event(f"model_loaded:{identity[0]}:{identity[1]}:{identity[2]}")
            continue
        data_manifest = data.manifests.views[(identity[0], identity[2])]
        intent = _intent_path(root, key.key_id)
        if not _claim_intent(intent, _intent_body(config, key, data_manifest)):
            raise TrainingDataArtifactError("screening model was claimed concurrently")
        if event is not None:
            event(f"model_train:{identity[0]}:{identity[1]}:{identity[2]}")
        try:
            training = _training_parameters(config, identity[0], identity[1])
            model, report, accounting = _train(
                identity[0], payloads[(identity[0], identity[2])], key, training
            )
            write_training_artifact(
                root,
                key=key,
                model_id=key.backbone_id,
                model=model,
                accounting=accounting,
                report=TrainingReportMetadata(
                    trainable_parameters=report.trainable_parameters,
                    optimizer_steps=report.optimizer_steps,
                    forward_passes=report.forward_passes,
                    training_examples=report.training_examples,
                ),
            )
        except Exception as exc:
            raise TrainingDataArtifactError(
                "screening model materialization interrupted; intent remains fail-closed"
            ) from exc
        loaded = _preflight(root, config, model_keys, data.manifests)
    if len(loaded) != 60:
        raise TrainingDataArtifactError("screening model materialization is incomplete")
    reports = {identity: item[2] for identity, item in loaded.items()}
    _validate_matched_pairs(reports, payloads, model_keys)
    return MaterializedScreeningModels(
        manifests={identity: item[0] for identity, item in loaded.items()},
        costs={identity: item[1] for identity, item in loaded.items()},
        compute=reports,
        b1_compute={identity: report for identity, report in reports.items() if identity[0] == B1},
    )


def _load_one_readonly(
    run_dir: Path,
    config: ExperimentConfig,
    key: TrainingArtifactKey,
    data_manifest: TrainingDataArtifactManifest,
    payload: Any,
) -> tuple[TrainingArtifactManifest, TrainingArtifactCostRecord, ModelComputeReport]:
    """Load one artifact and recompute its report without a forward execution."""

    index = load_training_key_index(run_dir, key)
    cost = load_training_cost(run_dir, key)
    model, manifest = load_training_model(
        run_dir,
        index.artifact_id,
        expected_key=key,
        model_factory=_model_factory,
    )
    if model.training:
        raise TrainingDataArtifactError("screening model loader did not return eval mode")
    artifact_dir = run_dir / "training-artifacts" / manifest.artifact_id
    try:
        artifact_entries = tuple(artifact_dir.iterdir())
    except OSError as exc:
        raise TrainingDataArtifactError("screening model artifact is unreadable") from exc
    if (
        artifact_dir.is_symlink()
        or {item.name for item in artifact_entries} != {"manifest.json", "tensors"}
        or any(item.is_symlink() for item in artifact_entries)
        or not (artifact_dir / "manifest.json").is_file()
        or not (artifact_dir / "tensors").is_dir()
    ):
        raise TrainingDataArtifactError("screening model artifact inventory drifted")
    if manifest.model_id != _expected_model_id(key.condition_id) or manifest.key != key:
        raise TrainingDataArtifactError("screening model manifest identity drifted")
    if cost.artifact_id != manifest.artifact_id or cost.key_id != key.key_id:
        raise TrainingDataArtifactError("screening model cost lineage drifted")
    if not isinstance(cost.accounting, TrainingPreparationAccounting):
        raise TrainingDataArtifactError("screening model cost has the wrong preparation schema")
    actual_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    samples = learner_samples(payload)
    if key.condition_id == B1:
        features, targets = global_frequency_optimum_examples(samples)
        if len(features) != len(targets):
            raise TrainingDataArtifactError("B1 examples and targets are not aligned")
        training_examples = len(features)
    elif key.condition_id == B2:
        training_examples = len(global_listwise_optimum_examples(samples))
    elif key.condition_id == C:
        training_examples = len(optimum_imitation_examples(samples))
    else:
        raise TrainingDataArtifactError("screening model has an unsupported condition")
    epochs = _training_parameters(
        config, key.condition_id, key.training_tuple_id
    ).epochs
    expected_forward_passes = (
        epochs if key.condition_id == B1 else epochs * training_examples
    )
    expected_report = TrainingReportMetadata(
        trainable_parameters=actual_parameters,
        training_examples=training_examples,
        optimizer_steps=epochs,
        forward_passes=expected_forward_passes,
    )
    if manifest.report != expected_report:
        raise TrainingDataArtifactError("screening model report is not canonically derived")
    accounting = cost.accounting
    if accounting.setup.wall_seconds <= 0 or accounting.training.wall_seconds <= 0:
        raise TrainingDataArtifactError("screening model working-phase wall time is absent")
    expected_training = PhaseAccounting(
        calls=1,
        optimizer_steps=epochs,
        forward_passes=expected_forward_passes,
        wall_seconds=accounting.training.wall_seconds,
    )
    expected_setup = PhaseAccounting(calls=1, wall_seconds=accounting.setup.wall_seconds)
    expected_serialization = PhaseAccounting(calls=1)
    if accounting.training != expected_training:
        raise TrainingDataArtifactError("screening model training accounting drifted")
    if accounting.setup != expected_setup or accounting.serialization != expected_serialization:
        raise TrainingDataArtifactError("screening model preparation accounting drifted")
    if (
        accounting.training_probes != PhaseAccounting()
        or accounting.reference_replay != PhaseAccounting()
    ):
        raise TrainingDataArtifactError("screening model includes unearned interaction cost")
    if key.training_data_sha256 != data_manifest.artifact_id:
        raise TrainingDataArtifactError("screening model references the wrong training view")
    return manifest, cost, ModelComputeReport(
        model_id=manifest.model_id,
        objective_id=key.objective_id,
        trainable_parameters=actual_parameters,
        training_examples=training_examples,
        optimizer_steps=epochs,
        forward_passes=expected_forward_passes,
        training_wall_seconds=accounting.training.wall_seconds,
    )


def load_screening_model_inventory(
    config: ExperimentConfig,
    data_keys: ScreeningDataKeys,
    data: MaterializedScreeningData,
    model_keys: ScreeningModelKeys,
    run_dir: str | Path,
) -> MaterializedScreeningModels:
    """Read and validate all 60 model artifacts without training or inference."""

    validate_screening_child_config(config)
    if config.split.final_tasks:
        raise ValueError("screening model inventory cannot receive final tasks")
    if data_keys != build_screening_data_keys(config, data_keys.provenance):
        raise ValueError("screening model inventory received noncanonical data keys")
    if len(data_keys.evidence) != 5 or len(data_keys.views) != 15:
        raise ValueError("screening model inventory received incomplete data keys")
    expected_data = load_screening_data_inventory(config, data_keys, run_dir)
    if expected_data != data:
        raise TrainingDataArtifactError("screening model inventory data has drifted")
    expected_model_keys = build_screening_model_keys(config, data_keys, data.manifests)
    if model_keys != expected_model_keys or len(model_keys.models) != 60:
        raise ValueError("screening model-key inventory is not the frozen 60-model matrix")

    root = Path(run_dir)
    if root.is_symlink() or not root.exists() or not root.is_dir():
        raise TrainingDataArtifactError("screening model inventory root is missing or unsafe")
    for parent in (root, *root.parents):
        if parent.is_symlink():
            raise TrainingDataArtifactError("screening model inventory has a symlinked ancestor")

    expected_ids = {key.key_id for key in model_keys.models.values()}
    expected_names = {f"{key_id}.json" for key_id in expected_ids}
    _check_inventory(
        root,
        {
            "screening-model-intents": (expected_names, "file"),
            "training-artifact-keys": (expected_names, "file"),
            "training-artifact-costs": (expected_names, "file"),
        },
    )
    for identity, key in model_keys.models.items():
        data_manifest = data.manifests.views[(identity[0], identity[2])]
        intent_path = root / "screening-model-intents" / f"{key.key_id}.json"
        if intent_path.read_bytes() != canonical_json_bytes(
            _intent_body(config, key, data_manifest)
        ) + b"\n":
            raise TrainingDataArtifactError("screening model intent content drifted")

    payloads = _validate_data(root, data_keys, data)
    loaded: dict[ModelIdentity, tuple[TrainingArtifactManifest, TrainingArtifactCostRecord, ModelComputeReport]] = {}
    artifact_ids: set[str] = set()
    for identity, key in model_keys.models.items():
        data_manifest = data.manifests.views[(identity[0], identity[2])]
        item = _load_one_readonly(
            root,
            config,
            key,
            data_manifest,
            payloads[(identity[0], identity[2])],
        )
        loaded[identity] = item
        artifact_ids.add(item[0].artifact_id)
    _check_inventory(root, {"training-artifacts": (artifact_ids, "directory")})
    reports = {identity: item[2] for identity, item in loaded.items()}
    _validate_matched_pairs(reports, payloads, model_keys)
    return MaterializedScreeningModels(
        manifests={identity: item[0] for identity, item in loaded.items()},
        costs={identity: item[1] for identity, item in loaded.items()},
        compute=reports,
        b1_compute={identity: report for identity, report in reports.items() if identity[0] == B1},
    )


prepare_screening_models = materialize_screening_models
