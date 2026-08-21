"""Immutable learner-visible training-data artifacts.

Only sanitized observable traces and paid-probe affordance tables are accepted.  The
format deliberately contains no environment, trajectory, evaluator, or Torch objects.
Writers are sequential-only; publication uses a private staging directory followed by
an atomic directory rename and an exclusive key-index claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from levelup.experiments.runner.config import canonical_json_bytes

HEX64 = re.compile(r"^[0-9a-f]{64}$")
AFFORDANCE_FEATURE_COUNT = 49
_CANONICAL_SANITIZED_DATA_TOKEN = object()


class TrainingDataArtifactError(RuntimeError):
    """Raised for invalid, incomplete, or conflicting persisted data."""


HASH = Field(pattern=r"^[0-9a-f]{64}$")


class TrainingDataArtifactKey(BaseModel):
    """Every declared input that can change learner-visible training data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runner.training-data-key.v1"] = "runner.training-data-key.v1"
    screening_candidates_sha256: str = HASH
    protocol_sha256: str = HASH
    task_manifest_sha256: str = HASH
    expected_unit_plan_sha256: str = HASH
    provenance_sha256: str = HASH
    reference_exposure_sha256: str = HASH
    representation_sha256: str = HASH
    probe_policy_sha256: str = HASH
    fold_id: str = Field(min_length=1)
    heldout_family_id: str = Field(min_length=1)
    ordered_training_task_ids: tuple[str, ...]
    ordered_heldout_task_ids: tuple[str, ...]
    condition_id: str = Field(min_length=1)
    objective_id: str = Field(min_length=1)
    replicate: int = Field(ge=0)
    data_order_seed: int
    probe_seeds: tuple[int, ...]
    environment_seeds: tuple[int, ...]

    @model_validator(mode="after")
    def task_ids_are_valid(self) -> "TrainingDataArtifactKey":
        all_ids = (*self.ordered_training_task_ids, *self.ordered_heldout_task_ids)
        if (
            not self.ordered_training_task_ids
            or not self.ordered_heldout_task_ids
            or any(not value for value in all_ids)
        ):
            raise ValueError("training and held-out task IDs must be non-empty")
        if len(set(self.ordered_training_task_ids)) != len(self.ordered_training_task_ids):
            raise ValueError("training task IDs must be unique")
        if len(set(self.ordered_heldout_task_ids)) != len(self.ordered_heldout_task_ids):
            raise ValueError("held-out task IDs must be unique")
        if set(self.ordered_training_task_ids) & set(self.ordered_heldout_task_ids):
            raise ValueError("training and held-out task IDs must be disjoint")
        if len(self.probe_seeds) != len(self.ordered_training_task_ids):
            raise ValueError("one probe seed is required per ordered training task")
        if len(self.environment_seeds) != len(self.ordered_training_task_ids):
            raise ValueError("one environment seed is required per ordered training task")
        return self

    @property
    def key_id(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


class TrainingDataEvidenceKey(BaseModel):
    """Condition-independent identity for one paid-probe/reference evidence build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runner.training-data-evidence-key.v1"] = (
        "runner.training-data-evidence-key.v1"
    )
    screening_candidates_sha256: str = HASH
    protocol_sha256: str = HASH
    task_manifest_sha256: str = HASH
    expected_unit_plan_sha256: str = HASH
    provenance_sha256: str = HASH
    reference_exposure_sha256: str = HASH
    probe_policy_sha256: str = HASH
    fold_id: str = Field(min_length=1)
    heldout_family_id: str = Field(min_length=1)
    ordered_training_task_ids: tuple[str, ...]
    ordered_heldout_task_ids: tuple[str, ...]
    replicate: int = Field(ge=0)
    data_order_seed: int
    probe_seeds: tuple[int, ...]
    environment_seeds: tuple[int, ...]

    @model_validator(mode="after")
    def task_ids_and_seeds_are_valid(self) -> "TrainingDataEvidenceKey":
        if not self.ordered_training_task_ids or not self.ordered_heldout_task_ids:
            raise ValueError("evidence requires ordered training and held-out tasks")
        if len(set(self.ordered_training_task_ids)) != len(
            self.ordered_training_task_ids
        ) or len(set(self.ordered_heldout_task_ids)) != len(
            self.ordered_heldout_task_ids
        ):
            raise ValueError("evidence task identities must be unique")
        if set(self.ordered_training_task_ids) & set(self.ordered_heldout_task_ids):
            raise ValueError("evidence training and held-out tasks must be disjoint")
        if len(self.probe_seeds) != len(self.ordered_training_task_ids) or len(
            self.environment_seeds
        ) != len(self.ordered_training_task_ids):
            raise ValueError("evidence seeds must align with ordered training tasks")
        return self

    @property
    def key_id(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


def evidence_key_for(key: TrainingDataArtifactKey) -> TrainingDataEvidenceKey:
    """Drop condition/objective/representation fields from a training-data view key."""

    return TrainingDataEvidenceKey(
        screening_candidates_sha256=key.screening_candidates_sha256,
        protocol_sha256=key.protocol_sha256,
        task_manifest_sha256=key.task_manifest_sha256,
        expected_unit_plan_sha256=key.expected_unit_plan_sha256,
        provenance_sha256=key.provenance_sha256,
        reference_exposure_sha256=key.reference_exposure_sha256,
        probe_policy_sha256=key.probe_policy_sha256,
        fold_id=key.fold_id,
        heldout_family_id=key.heldout_family_id,
        ordered_training_task_ids=key.ordered_training_task_ids,
        ordered_heldout_task_ids=key.ordered_heldout_task_ids,
        replicate=key.replicate,
        data_order_seed=key.data_order_seed,
        probe_seeds=key.probe_seeds,
        environment_seeds=key.environment_seeds,
    )


class ObservableStateRecord(BaseModel):
    """The complete current observable state permitted to the learner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    progress_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    remaining_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    elapsed_per_target: float = Field(ge=0, allow_inf_nan=False)
    resource_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    pressure_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    available_aliases: tuple[str, ...]

    @model_validator(mode="after")
    def aliases_are_safe(self) -> "ObservableStateRecord":
        if any(not alias or any(ord(char) < 32 for char in alias) for alias in self.available_aliases):
            raise ValueError("observable aliases must be nonempty and control-character free")
        if len(set(self.available_aliases)) != len(self.available_aliases):
            raise ValueError("observable aliases must be unique")
        return self

    def features(self) -> tuple[float, ...]:
        return (
            self.progress_fraction,
            self.remaining_fraction,
            self.elapsed_per_target,
            self.resource_fraction,
            self.pressure_fraction,
        )


class ObservedTransitionRecord(BaseModel):
    """One ordered learner-visible transition, with no hidden state fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    before: ObservableStateRecord
    action_alias: str = Field(min_length=1)
    after: ObservableStateRecord
    completed: bool

    @model_validator(mode="after")
    def action_alias_is_safe(self) -> "ObservedTransitionRecord":
        if any(ord(char) < 32 for char in self.action_alias):
            raise ValueError("transition action alias contains a control character")
        return self


class ObservableTraceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transitions: tuple[ObservedTransitionRecord, ...]

    @model_validator(mode="after")
    def is_contiguous(self) -> "ObservableTraceRecord":
        if not self.transitions:
            raise ValueError("observable trace must contain at least one transition")
        for index, transition in enumerate(self.transitions):
            if transition.action_alias not in transition.before.available_aliases:
                raise ValueError("trace action is unavailable in its preceding state")
            if index and self.transitions[index - 1].after != transition.before:
                raise ValueError("trace transitions are not contiguous")
            if index and self.transitions[index - 1].completed:
                raise ValueError("trace continues after completion")
        if not self.transitions[-1].completed:
            raise ValueError("optimum training trace must end in observable completion")
        return self


class AffordanceTableRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    features: dict[str, tuple[float, ...]]
    sample_counts: dict[str, int]

    @model_validator(mode="after")
    def table_is_valid(self) -> "AffordanceTableRecord":
        if set(self.features) != set(self.sample_counts) or not self.features:
            raise ValueError("affordance features and counts must have the same non-empty aliases")
        if any(len(row) != AFFORDANCE_FEATURE_COUNT for row in self.features.values()):
            raise ValueError("affordance rows have the wrong feature width")
        if any(count < 1 for count in self.sample_counts.values()):
            raise ValueError("affordance sample counts must be positive")
        if any(
            not alias or any(ord(char) < 32 for char in alias)
            for alias in self.features
        ):
            raise ValueError("affordance aliases must be nonempty and control-character free")
        if any(
            any(not math.isfinite(value) for value in row)
            for row in self.features.values()
        ):
            raise ValueError("affordance values must be finite")
        return self

    def for_alias(self, alias: str) -> tuple[float, ...] | None:
        return self.features.get(alias)


class TrainingDataSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    trace: ObservableTraceRecord
    affordances: AffordanceTableRecord

    @model_validator(mode="after")
    def task_id_is_safe(self) -> "TrainingDataSample":
        if any(ord(char) < 32 for char in self.task_id):
            raise ValueError("training sample task ID contains a control character")
        return self


class TrainingDataPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runner.training-data-payload.v1"] = "runner.training-data-payload.v1"
    samples: tuple[TrainingDataSample, ...]

    @model_validator(mode="after")
    def samples_are_unique(self) -> "TrainingDataPayload":
        ids = [sample.task_id for sample in self.samples]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("training samples must be non-empty and task-unique")
        return self


@dataclass(frozen=True, slots=True, init=False)
class SanitizedTrainingData:
    """Opaque batch that can only be produced by the canonical boundary sanitizer."""

    samples: tuple[TrainingDataSample, ...]
    _construction_token: object

    def __init__(
        self,
        samples: tuple[TrainingDataSample, ...],
        *,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _CANONICAL_SANITIZED_DATA_TOKEN:
            raise ValueError("sanitized training data requires the canonical sanitizer")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "_construction_token", _construction_token)


def sanitize_clean_optimum_samples(samples: Sequence[Any]) -> SanitizedTrainingData:
    """Convert canonical baseline samples without importing their Torch module.

    The generation-only import verifies the canonical paid-probe construction token.
    This boundary then copies only ``reference.trace`` and ``probe.affordances`` plus
    the task ID, so audit fields cannot enter the learner payload.
    """

    from levelup.experiments.milestone6_baselines import (
        CleanOptimumTrainingSample,
        optimum_only_training_samples,
    )

    if not samples or any(
        not isinstance(sample, CleanOptimumTrainingSample) for sample in samples
    ):
        raise ValueError("training-data sanitization requires canonical clean samples")
    optimum_only_training_samples(tuple(samples))

    converted: list[TrainingDataSample] = []
    for source in samples:
        reference = source.reference
        probe = source.probe
        if reference.stage_label != "optimum":
            raise ValueError("training-data artifacts require optimum reference samples")
        if reference.task_id != probe.task_id:
            raise ValueError("reference and paid-probe task identities must match")

        def state_record(state: Any) -> ObservableStateRecord:
            return ObservableStateRecord(
                progress_fraction=state.progress_fraction,
                remaining_fraction=state.remaining_fraction,
                elapsed_per_target=state.elapsed_per_target,
                resource_fraction=state.resource_fraction,
                pressure_fraction=state.pressure_fraction,
                available_aliases=tuple(state.available_aliases),
            )

        converted.append(
            TrainingDataSample(
                task_id=reference.task_id,
                trace=ObservableTraceRecord(
                    transitions=tuple(
                        ObservedTransitionRecord(
                            before=state_record(transition.before),
                            action_alias=transition.action_alias,
                            after=state_record(transition.after),
                            completed=transition.completed,
                        )
                        for transition in reference.trace.transitions
                    )
                ),
                affordances=AffordanceTableRecord(
                    features={key: tuple(values) for key, values in probe.affordances.features.items()},
                    sample_counts=dict(probe.affordances.sample_counts),
                ),
            )
        )
    return SanitizedTrainingData(
        tuple(converted),
        _construction_token=_CANONICAL_SANITIZED_DATA_TOKEN,
    )


class TrainingDataArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runner.training-data-manifest.v1"] = "runner.training-data-manifest.v1"
    artifact_id: str = HASH
    evidence_id: str = HASH
    key_id: str = HASH
    key: TrainingDataArtifactKey
    payload_sha256: str = HASH
    payload_bytes: int = Field(gt=0)
    sample_task_ids: tuple[str, ...]

    @model_validator(mode="after")
    def identity_is_valid(self) -> "TrainingDataArtifactManifest":
        if self.key_id != self.key.key_id:
            raise ValueError("training-data manifest key identity mismatch")
        expected = hashlib.sha256(
            canonical_json_bytes(self.model_dump(mode="json", exclude={"artifact_id"}))
        ).hexdigest()
        if self.artifact_id != expected:
            raise ValueError("training-data manifest artifact identity mismatch")
        if self.sample_task_ids != self.key.ordered_training_task_ids:
            raise ValueError("training-data manifest tasks do not match its training fold")
        return self


class TrainingDataEvidenceManifest(BaseModel):
    """Condition-independent content manifest shared by equivalent views."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runner.training-data-evidence.v1"] = "runner.training-data-evidence.v1"
    evidence_id: str = HASH
    evidence_key_id: str = HASH
    key: TrainingDataEvidenceKey
    payload_sha256: str = HASH
    payload_bytes: int = Field(gt=0)
    sample_task_ids: tuple[str, ...]

    @model_validator(mode="after")
    def identity_is_valid(self) -> "TrainingDataEvidenceManifest":
        if self.evidence_key_id != self.key.key_id:
            raise ValueError("training-data evidence key identity mismatch")
        if self.sample_task_ids != self.key.ordered_training_task_ids:
            raise ValueError("training-data evidence tasks do not match its training fold")
        expected = hashlib.sha256(
            canonical_json_bytes(self.model_dump(mode="json", exclude={"evidence_id"}))
        ).hexdigest()
        if self.evidence_id != expected:
            raise ValueError("training-data evidence identity mismatch")
        return self


class TrainingDataKeyIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runner.training-data-index.v1"] = "runner.training-data-index.v1"
    key_id: str = HASH
    key: TrainingDataArtifactKey
    artifact_id: str = HASH

    @model_validator(mode="after")
    def identity_is_valid(self) -> "TrainingDataKeyIndex":
        if self.key_id != self.key.key_id:
            raise ValueError("training-data key index identity mismatch")
        return self


def _read_json(path: Path) -> Any:
    if path.is_symlink():
        raise TrainingDataArtifactError("symlinks are not permitted in training-data artifacts")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingDataArtifactError(f"invalid training-data artifact file: {path.name}") from exc


def _validate_model(model_type: type[BaseModel], raw: Any, label: str) -> BaseModel:
    try:
        return model_type.model_validate(raw)
    except (TypeError, ValueError) as exc:
        raise TrainingDataArtifactError(f"invalid {label} schema") from exc


def _safe_child(root: Path, child: Path) -> Path:
    if root.is_symlink() or child.is_symlink():
        raise TrainingDataArtifactError("symlinks are not permitted in training-data artifacts")
    try:
        child.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise TrainingDataArtifactError("training-data path escapes root") from exc
    return child


def _atomic_json(path: Path, value: Any) -> None:
    if path.is_symlink() or path.parent.is_symlink():
        raise TrainingDataArtifactError("refusing symlink training-data publication path")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(canonical_json_bytes(value).decode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _validate_loaded(root: Path, artifact_id: str) -> TrainingDataArtifactManifest:
    if not HEX64.fullmatch(artifact_id):
        raise TrainingDataArtifactError("invalid training-data artifact ID")
    artifact_dir = _safe_child(
        _safe_child(root, root / "training-data-artifacts"),
        root / "training-data-artifacts" / artifact_id,
    )
    if not artifact_dir.is_dir() or {path.name for path in artifact_dir.iterdir()} != {
        "manifest.json"
    }:
        raise TrainingDataArtifactError("training-data view has unexpected files")
    manifest_path = _safe_child(artifact_dir, artifact_dir / "manifest.json")
    manifest = _validate_model(
        TrainingDataArtifactManifest,
        _read_json(manifest_path),
        "training-data manifest",
    )
    assert isinstance(manifest, TrainingDataArtifactManifest)
    if manifest.artifact_id != artifact_id:
        raise TrainingDataArtifactError("training-data manifest identity mismatch")
    return manifest


def _load_evidence(
    root: Path, evidence_id: str
) -> tuple[TrainingDataEvidenceManifest, TrainingDataPayload]:
    if not HEX64.fullmatch(evidence_id):
        raise TrainingDataArtifactError("invalid evidence ID")
    evidence_root = _safe_child(root, root / "training-data-evidence")
    evidence_dir = _safe_child(evidence_root, evidence_root / evidence_id)
    if not evidence_dir.is_dir() or {path.name for path in evidence_dir.iterdir()} != {
        "manifest.json",
        "samples.json",
    }:
        raise TrainingDataArtifactError("training-data evidence has unexpected files")
    manifest_path = _safe_child(evidence_dir, evidence_dir / "manifest.json")
    evidence_manifest = _validate_model(
        TrainingDataEvidenceManifest,
        _read_json(manifest_path),
        "training-data evidence manifest",
    )
    assert isinstance(evidence_manifest, TrainingDataEvidenceManifest)
    payload_path = _safe_child(evidence_dir, evidence_dir / "samples.json")
    try:
        payload_bytes = payload_path.read_bytes()
        payload = TrainingDataPayload.model_validate(json.loads(payload_bytes))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TrainingDataArtifactError("invalid training-data evidence payload") from exc
    observed_hash = hashlib.sha256(payload_bytes).hexdigest()
    observed_tasks = tuple(sample.task_id for sample in payload.samples)
    if (
        evidence_manifest.evidence_id != evidence_id
        or evidence_manifest.payload_sha256 != observed_hash
        or evidence_manifest.payload_bytes != len(payload_bytes)
        or evidence_manifest.sample_task_ids != observed_tasks
    ):
        raise TrainingDataArtifactError("training-data evidence integrity mismatch")
    return evidence_manifest, payload


def write_training_data_artifact(
    root: str | Path,
    key: TrainingDataArtifactKey,
    data: SanitizedTrainingData,
) -> TrainingDataArtifactManifest:
    """Publish one immutable sanitized dataset; repeated identical writes are idempotent."""

    if (
        not isinstance(data, SanitizedTrainingData)
        or data._construction_token is not _CANONICAL_SANITIZED_DATA_TOKEN
    ):
        raise TrainingDataArtifactError(
            "training-data publication requires the canonical sanitized batch"
        )
    root_path = Path(root)
    if root_path.is_symlink():
        raise TrainingDataArtifactError("training-data artifact root cannot be a symlink")
    root_path.mkdir(parents=True, exist_ok=True)
    artifact_root = root_path / "training-data-artifacts"
    evidence_root = root_path / "training-data-evidence"
    index_root = root_path / "training-data-artifact-keys"
    for directory in (artifact_root, evidence_root, index_root):
        if directory.exists() and directory.is_symlink():
            raise TrainingDataArtifactError("training-data artifact root cannot be a symlink")
        directory.mkdir(exist_ok=True)
    payload = TrainingDataPayload(samples=data.samples)
    sample_task_ids = tuple(sample.task_id for sample in payload.samples)
    if sample_task_ids != key.ordered_training_task_ids:
        raise TrainingDataArtifactError(
            "training-data payload tasks do not exactly match the ordered training fold"
        )
    payload_bytes = canonical_json_bytes(payload.model_dump(mode="json"))
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    evidence_key = evidence_key_for(key)
    evidence_body = {
        "schema_version": "runner.training-data-evidence.v1",
        "evidence_key_id": evidence_key.key_id,
        "key": evidence_key.model_dump(mode="json"),
        "payload_sha256": payload_sha256,
        "payload_bytes": len(payload_bytes),
        "sample_task_ids": sample_task_ids,
    }
    evidence_id = hashlib.sha256(canonical_json_bytes(evidence_body)).hexdigest()
    evidence_manifest = TrainingDataEvidenceManifest(
        evidence_id=evidence_id, **evidence_body
    )
    evidence_dir = evidence_root / evidence_id
    if evidence_dir.exists():
        loaded_evidence, loaded_payload = _load_evidence(root_path, evidence_id)
        if loaded_evidence != evidence_manifest or loaded_payload != payload:
            raise TrainingDataArtifactError("conflicting training-data evidence")
    else:
        staging = Path(tempfile.mkdtemp(prefix=".training-evidence.", dir=evidence_root))
        try:
            _atomic_json(staging / "samples.json", payload.model_dump(mode="json"))
            _atomic_json(staging / "manifest.json", evidence_manifest.model_dump(mode="json"))
            os.rename(staging, evidence_dir)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    manifest_body = {
        "schema_version": "runner.training-data-manifest.v1",
        "evidence_id": evidence_id,
        "key_id": key.key_id,
        "key": key.model_dump(mode="json"),
        "payload_sha256": payload_sha256,
        "payload_bytes": len(payload_bytes),
        "sample_task_ids": sample_task_ids,
    }
    artifact_id = hashlib.sha256(canonical_json_bytes(manifest_body)).hexdigest()
    manifest = TrainingDataArtifactManifest(artifact_id=artifact_id, **manifest_body)
    artifact_dir = artifact_root / artifact_id
    if artifact_dir.exists():
        existing = _validate_loaded(root_path, artifact_id)
        if existing != manifest:
            raise TrainingDataArtifactError("conflicting training-data artifact")
    else:
        staging = Path(tempfile.mkdtemp(prefix=".training-data.", dir=artifact_root))
        try:
            _atomic_json(staging / "manifest.json", manifest.model_dump(mode="json"))
            os.rename(staging, artifact_dir)
        except FileExistsError:
            if artifact_dir.exists():
                _validate_loaded(root_path, artifact_id)
            else:
                raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    index = TrainingDataKeyIndex(key_id=key.key_id, key=key, artifact_id=artifact_id)
    index_path = index_root / f"{key.key_id}.json"
    if index_path.exists():
        existing = _validate_model(
            TrainingDataKeyIndex,
            _read_json(index_path),
            "training-data key index",
        )
        assert isinstance(existing, TrainingDataKeyIndex)
        if existing != index:
            raise TrainingDataArtifactError("training-data key index conflict")
    else:
        temp = index_root / f".{key.key_id}.tmp"
        _atomic_json(temp, index.model_dump(mode="json"))
        try:
            os.link(temp, index_path)
        except FileExistsError:
            existing = _validate_model(
                TrainingDataKeyIndex,
                _read_json(index_path),
                "training-data key index",
            )
            assert isinstance(existing, TrainingDataKeyIndex)
            if existing != index:
                raise TrainingDataArtifactError("training-data key index conflict")
        finally:
            if temp.exists():
                temp.unlink()
    return manifest


def load_training_data_artifact(
    root: str | Path,
    artifact_id: str | None = None,
    *,
    expected_key: TrainingDataArtifactKey | None = None,
) -> tuple[TrainingDataArtifactManifest, TrainingDataPayload]:
    root_path = Path(root)
    if root_path.is_symlink():
        raise TrainingDataArtifactError("training-data artifact root cannot be a symlink")
    if expected_key is not None:
        index_path = _safe_child(root_path / "training-data-artifact-keys", root_path / "training-data-artifact-keys" / f"{expected_key.key_id}.json")
        index = _validate_model(
            TrainingDataKeyIndex,
            _read_json(index_path),
            "training-data key index",
        )
        assert isinstance(index, TrainingDataKeyIndex)
        if index.key != expected_key or index.key_id != expected_key.key_id:
            raise TrainingDataArtifactError("training-data key index mismatch")
        artifact_id = index.artifact_id
    if artifact_id is None:
        raise TrainingDataArtifactError("artifact ID or expected key is required")
    manifest = _validate_loaded(root_path, artifact_id)
    evidence_manifest, payload = _load_evidence(root_path, manifest.evidence_id)
    if (
        evidence_manifest.key != evidence_key_for(manifest.key)
        or manifest.payload_sha256 != evidence_manifest.payload_sha256
        or manifest.payload_bytes != evidence_manifest.payload_bytes
        or manifest.sample_task_ids != evidence_manifest.sample_task_ids
    ):
        raise TrainingDataArtifactError("training-data view/evidence mismatch")
    if expected_key is not None and manifest.key != expected_key:
        raise TrainingDataArtifactError("training-data key mismatch")
    return manifest, payload


def learner_samples(
    payload: TrainingDataPayload,
) -> tuple[tuple[ObservableTraceRecord, AffordanceTableRecord], ...]:
    """Expose only the trace/table pairs accepted by the existing learner builders."""

    return tuple((sample.trace, sample.affordances) for sample in payload.samples)
