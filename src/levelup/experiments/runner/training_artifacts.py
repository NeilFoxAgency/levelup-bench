"""Content-addressed, non-pickle storage for completed Torch training artifacts.

Artifacts are immutable once published.  The writer is intentionally sequential-only:
callers that need concurrent preparation must provide an external lock.  A manifest is
published last, after every tensor file has been flushed and hashed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn

import levelup.experiments.runner.secure_fs as secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import (
    ResourceAccounting,
    TrainingArtifactCostRecord,
    TrainingPreparationAccounting,
)
from levelup.experiments.runner.storage import ArtifactValidationError

HASH_FIELD = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PinnedTrainingArtifactReader:
    """Descriptors retained across one model cost/index/tensor transaction."""

    keys_fd: int
    costs_fd: int
    artifacts_fd: int


@contextmanager
def open_training_artifact_reader(
    run_fd: int,
) -> Iterator[PinnedTrainingArtifactReader]:
    """Pin all model-artifact namespaces below an already-pinned run fd."""

    with ExitStack() as stack:
        try:
            descriptors: dict[str, int] = {}
            for field, name in (
                ("keys_fd", "training-artifact-keys"),
                ("costs_fd", "training-artifact-costs"),
                ("artifacts_fd", "training-artifacts"),
            ):
                descriptor = secure_fs.open_child_directory(run_fd, name)
                stack.callback(os.close, descriptor)
                descriptors[field] = descriptor
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ArtifactValidationError(
                "training artifact namespaces contain a symlink or cannot be securely pinned"
            ) from exc
        yield PinnedTrainingArtifactReader(**descriptors)
MODEL_IDS = frozenset(
    {
        "global_affordance_mlp_frequency_v1",
        "global_affordance_mlp_listwise_v1",
        "state_conditioned_mlp_listwise_v1",
    }
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
LOCAL_TENSOR = re.compile(r"^[0-9]{4}\.bin$")


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _safe_resolved_child(root: Path, child: Path) -> Path:
    """Reject links/traversal and require a path to remain beneath root."""

    if root.is_symlink() or child.is_symlink():
        raise ArtifactValidationError(f"refusing symlink path: {child}")
    root_resolved = root.resolve()
    child_resolved = child.resolve()
    try:
        child_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ArtifactValidationError(f"path escapes artifact root: {child}") from exc
    return child


class TrainingArtifactKey(BaseModel):
    """All scientific inputs that can change a completed training artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runner.training-key.v1"] = "runner.training-key.v1"
    screening_candidates_sha256: str = HASH_FIELD
    protocol_sha256: str = HASH_FIELD
    task_manifest_sha256: str = HASH_FIELD
    expected_unit_plan_sha256: str = HASH_FIELD
    exposure_sha256: str = HASH_FIELD
    training_data_sha256: str = HASH_FIELD
    provenance_sha256: str = HASH_FIELD
    fold_id: str = Field(min_length=1)
    heldout_family_id: str = Field(min_length=1)
    ordered_training_task_ids: tuple[str, ...]
    ordered_heldout_task_ids: tuple[str, ...]
    condition_id: str = Field(min_length=1)
    learner_id: str = Field(min_length=1)
    objective_id: str = Field(min_length=1)
    backbone_id: str = Field(min_length=1)
    training_tuple_id: str = Field(min_length=1)
    replicate: int = Field(ge=0)
    model_seed: int
    data_order_seed: int
    probe_seeds: tuple[int, ...]
    environment_seeds: tuple[int, ...]
    probe_spec_sha256: str = HASH_FIELD
    training_config_sha256: str = HASH_FIELD
    capacity_spec_sha256: str = HASH_FIELD

    @model_validator(mode="after")
    def ids_are_nonempty(self) -> "TrainingArtifactKey":
        if not self.ordered_training_task_ids:
            raise ValueError("training artifact requires training task IDs")
        if any(
            not item for item in (*self.ordered_training_task_ids, *self.ordered_heldout_task_ids)
        ):
            raise ValueError("training and held-out task IDs must be non-empty")
        return self

    @property
    def key_id(self) -> str:
        return _digest(self.model_dump(mode="json"))


class TensorMetadata(BaseModel):
    """Integrity metadata for one raw little-endian float32 tensor file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    shape: tuple[int, ...]
    dtype: Literal["float32"] = "float32"
    byte_length: int = Field(ge=0)
    sha256: str = HASH_FIELD

    @model_validator(mode="after")
    def shape_is_valid(self) -> "TensorMetadata":
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("tensor dimensions must be nonnegative")
        return self


class TrainingReportMetadata(BaseModel):
    """Numeric training report stored with a completed shared artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trainable_parameters: int = Field(ge=0)
    optimizer_steps: int = Field(ge=0)
    forward_passes: int = Field(ge=0)
    training_examples: int = Field(ge=0)


class TrainingArtifactManifest(BaseModel):
    """Immutable manifest published after all model tensor files."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runner.training-artifact.v1"] = "runner.training-artifact.v1"
    artifact_id: str = HASH_FIELD
    key: TrainingArtifactKey
    model_id: str = Field(min_length=1)
    tensors: tuple[TensorMetadata, ...]
    report: TrainingReportMetadata

    @model_validator(mode="after")
    def manifest_is_canonical(self) -> "TrainingArtifactManifest":
        if self.artifact_id != self.expected_artifact_id:
            raise ValueError("artifact ID does not match canonical manifest body")
        if self.model_id not in MODEL_IDS:
            raise ValueError("model ID is not allowlisted")
        if self.model_id != self.key.backbone_id:
            raise ValueError("manifest model ID differs from artifact backbone")
        names = [tensor.name for tensor in self.tensors]
        files = [tensor.filename for tensor in self.tensors]
        if not names or len(names) != len(set(names)) or names != sorted(names):
            raise ValueError("tensor names must be unique and sorted")
        if len(files) != len(set(files)) or any(
            item != f"{index:04d}.bin" or not LOCAL_TENSOR.fullmatch(item)
            for index, item in enumerate(files)
        ):
            raise ValueError("tensor filenames must be unique and local")
        if any(any(ord(char) < 32 for char in value) for value in names):
            raise ValueError("tensor names cannot contain control characters")
        return self

    @property
    def expected_artifact_id(self) -> str:
        body = self.model_dump(mode="json", exclude={"artifact_id"})
        return _digest(body)


class TrainingArtifactKeyIndex(BaseModel):
    """Immutable key-to-artifact binding, claimed after artifact publication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runner.training-artifact-key.v1"] = "runner.training-artifact-key.v1"
    key_id: str = HASH_FIELD
    key: TrainingArtifactKey
    artifact_id: str = HASH_FIELD
    manifest_sha256: str = HASH_FIELD

    @model_validator(mode="after")
    def binding_is_exact(self) -> "TrainingArtifactKeyIndex":
        if self.key_id != self.key.key_id:
            raise ValueError("key index ID does not match key")
        return self


def _atomic_bytes(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise ArtifactValidationError(f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _exclusive_claim(path: Path, payload: bytes) -> bool:
    """Claim a path without replacement; callers validate an existing winner."""

    if path.is_symlink():
        raise ArtifactValidationError(f"refusing to claim symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".claim", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise ArtifactValidationError("training artifacts require CPU float32 tensors")
    if not bool(torch.isfinite(tensor).all()):
        raise ArtifactValidationError("training artifact tensor contains non-finite values")
    array = tensor.detach().contiguous().numpy().astype("<f4", copy=False)
    return array.tobytes(order="C")


def _state_tensors(model: nn.Module) -> dict[str, tuple[tuple[int, ...], bytes]]:
    state = model.state_dict()
    if not state:
        raise ArtifactValidationError("cannot publish a model with an empty state dict")
    result: dict[str, tuple[tuple[int, ...], bytes]] = {}
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ArtifactValidationError(f"state entry is not a tensor: {name}")
        result[name] = (tuple(int(item) for item in tensor.shape), _tensor_bytes(tensor))
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ArtifactValidationError(f"refusing to read symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("manifest must be an object")
        return value
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"invalid training artifact manifest: {type(exc).__name__}"
        ) from None


def _validate_tensor_file(path: Path, metadata: TensorMetadata) -> torch.Tensor:
    if path.is_symlink():
        raise ArtifactValidationError(f"refusing to read symlink: {path}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ArtifactValidationError(f"cannot read tensor file: {path.name}") from exc
    if (
        len(payload) != metadata.byte_length
        or hashlib.sha256(payload).hexdigest() != metadata.sha256
    ):
        raise ArtifactValidationError(f"tensor integrity mismatch: {metadata.name}")
    expected_bytes = 4
    for dimension in metadata.shape:
        expected_bytes *= dimension
    if len(payload) != expected_bytes:
        raise ArtifactValidationError(f"tensor byte length does not match shape: {metadata.name}")
    values = np.frombuffer(payload, dtype="<f4").reshape(metadata.shape).copy()
    tensor = torch.from_numpy(values)
    if not bool(torch.isfinite(tensor).all()):
        raise ArtifactValidationError(f"tensor contains non-finite values: {metadata.name}")
    return tensor


def write_training_artifact(
    output_root: str | Path,
    *,
    key: TrainingArtifactKey,
    model_id: str,
    model: nn.Module,
    accounting: ResourceAccounting,
    report: TrainingReportMetadata,
) -> TrainingArtifactManifest:
    """Write or idempotently validate one artifact (sequential callers only)."""

    if model_id not in MODEL_IDS:
        raise ArtifactValidationError("model ID is not allowlisted")
    if model_id != key.backbone_id:
        raise ArtifactValidationError("model ID differs from artifact backbone")
    actual_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if report.trainable_parameters != actual_parameters:
        raise ArtifactValidationError("training report parameter count does not match model")
    if accounting.search != ResourceAccounting().search or accounting.evaluator != ResourceAccounting().evaluator:
        raise ArtifactValidationError(
            "training preparation cannot include search or evaluator accounting"
        )
    preparation_accounting = TrainingPreparationAccounting(
        setup=accounting.setup,
        training_probes=accounting.probes,
        reference_replay=accounting.replay,
        training=accounting.training,
        serialization=accounting.serialization,
    )
    tensors = _state_tensors(model)
    output = Path(output_root)
    if output.is_symlink():
        raise ArtifactValidationError("refusing symlink artifact output root")
    artifacts_root = output / "training-artifacts"
    keys_root = output / "training-artifact-keys"
    if artifacts_root.is_symlink() or keys_root.is_symlink():
        raise ArtifactValidationError("refusing to use symlink training-artifacts directory")
    artifact_id = "pending"
    metadata: list[TensorMetadata] = []
    for index, name in enumerate(sorted(tensors)):
        shape, payload = tensors[name]
        metadata.append(
            TensorMetadata(
                name=name,
                filename=f"{index:04d}.bin",
                shape=shape,
                byte_length=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    body = {
        "schema_version": "runner.training-artifact.v1",
        "key": key.model_dump(mode="json"),
        "model_id": model_id,
        "tensors": [item.model_dump(mode="json") for item in metadata],
        "report": report.model_dump(mode="json"),
    }
    artifact_id = _digest(body)
    manifest = TrainingArtifactManifest(
        artifact_id=artifact_id,
        key=key,
        model_id=model_id,
        tensors=tuple(metadata),
        report=report,
    )
    artifact_dir = output / "training-artifacts" / artifact_id
    if artifact_dir.is_symlink():
        raise ArtifactValidationError("refusing to use symlink artifact directory")
    manifest_path = artifact_dir / "manifest.json"
    loaded: TrainingArtifactManifest
    if manifest_path.exists():
        existing = load_training_manifest(output, artifact_id)
        if existing != manifest:
            raise ArtifactValidationError(
                "existing training artifact conflicts with requested content"
            )
        _validate_artifact_files(artifact_dir, existing)
        loaded = existing
    else:
        if artifact_dir.exists():
            raise ArtifactValidationError("training artifact path is unexpectedly present")
        artifacts_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{artifact_id}.staging-", dir=artifacts_root))
        tensor_dir = staging / "tensors"
        try:
            tensor_dir.mkdir()
            for item in metadata:
                _atomic_bytes(tensor_dir / item.filename, tensors[item.name][1])
            _atomic_bytes(
                staging / "manifest.json",
                canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n",
            )
            os.replace(staging, artifact_dir)
        except FileExistsError:
            if staging.exists():
                shutil.rmtree(staging)
            loaded = load_training_manifest(output, artifact_id)
            if loaded != manifest:
                raise ArtifactValidationError("racing artifact conflicts with requested content")
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        else:
            loaded = load_training_manifest(output, artifact_id)
    index = TrainingArtifactKeyIndex(
        key_id=key.key_id,
        key=key,
        artifact_id=artifact_id,
        manifest_sha256=_digest(loaded.model_dump(mode="json")),
    )
    index_path = keys_root / f"{key.key_id}.json"
    claimed = _exclusive_claim(
        index_path, canonical_json_bytes(index.model_dump(mode="json")) + b"\n"
    )
    if not claimed:
        winner = load_training_key_index(output, key)
        if winner.artifact_id != artifact_id:
            raise ArtifactValidationError("different artifact already won key index race")
    cost_body = {
        "schema_version": "runner.training-artifact-cost.v2",
        "key_id": key.key_id,
        "artifact_id": artifact_id,
        "scope": "training_preparation",
        "key": key.model_dump(mode="json"),
        "accounting": preparation_accounting.model_dump(mode="json"),
    }
    cost = TrainingArtifactCostRecord.model_validate({"cost_id": _digest(cost_body), **cost_body})
    costs_root = output / "training-artifact-costs"
    if costs_root.is_symlink():
        raise ArtifactValidationError("refusing symlink training-artifact-costs directory")
    cost_path = costs_root / f"{key.key_id}.json"
    claimed_cost = _exclusive_claim(
        cost_path,
        canonical_json_bytes(cost.model_dump(mode="json")) + b"\n",
    )
    if not claimed_cost:
        load_training_cost(output, key)
    return loaded


def load_training_manifest(output_root: str | Path, artifact_id: str) -> TrainingArtifactManifest:
    """Load and validate an immutable manifest and every referenced tensor file."""

    if not HEX64.fullmatch(artifact_id):
        raise ArtifactValidationError("invalid training artifact ID")
    output = Path(output_root)
    artifact_dir, _ = _validated_artifact_paths(output, artifact_id)
    raw = _load_json(artifact_dir / "manifest.json")
    try:
        manifest = TrainingArtifactManifest.model_validate(raw)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError("invalid training artifact manifest schema") from exc
    if manifest.artifact_id != artifact_id or manifest.expected_artifact_id != artifact_id:
        raise ArtifactValidationError("training artifact ID mismatch")
    _validate_artifact_files(artifact_dir, manifest)
    return manifest


def _fd_child(stack: ExitStack, parent_fd: int, *components: str) -> int:
    try:
        child_fd = secure_fs.open_child_chain(parent_fd, *components)
    except secure_fs.SecureFilesystemError as exc:
        raise ArtifactValidationError("cannot securely open training artifact") from exc
    stack.callback(os.close, child_fd)
    return child_fd


def _fd_json(directory_fd: int, name: str) -> dict[str, Any]:
    try:
        value = secure_fs.load_json_at(directory_fd, name)
    except secure_fs.SecureFilesystemError as exc:
        raise ArtifactValidationError(f"invalid training artifact manifest: {name}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError("training artifact manifest must be an object")
    return value


def _fd_model(model_type: type[BaseModel], raw: Any, label: str) -> BaseModel:
    try:
        return model_type.model_validate(raw)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"invalid {label} schema") from exc


def _validate_tensor_file_at(
    tensors_fd: int, filename: str, metadata: TensorMetadata
) -> torch.Tensor:
    try:
        payload = secure_fs.read_bytes_at(tensors_fd, filename)
    except secure_fs.SecureFilesystemError as exc:
        raise ArtifactValidationError(f"cannot read tensor file: {filename}") from exc
    if (
        len(payload) != metadata.byte_length
        or hashlib.sha256(payload).hexdigest() != metadata.sha256
    ):
        raise ArtifactValidationError(f"tensor integrity mismatch: {metadata.name}")
    expected_bytes = 4
    for dimension in metadata.shape:
        expected_bytes *= dimension
    if len(payload) != expected_bytes:
        raise ArtifactValidationError(f"tensor byte length does not match shape: {metadata.name}")
    values = np.frombuffer(payload, dtype="<f4").reshape(metadata.shape).copy()
    tensor = torch.from_numpy(values)
    if not bool(torch.isfinite(tensor).all()):
        raise ArtifactValidationError(f"tensor contains non-finite values: {metadata.name}")
    return tensor


def _load_artifact_state_at(
    artifact_fd: int, manifest: TrainingArtifactManifest
) -> dict[str, torch.Tensor]:
    with ExitStack() as stack:
        try:
            with os.scandir(artifact_fd) as iterator:
                entries = {
                    entry.name: (
                        entry.is_symlink(),
                        entry.is_file(follow_symlinks=False),
                        entry.is_dir(follow_symlinks=False),
                    )
                    for entry in iterator
                }
        except OSError as exc:
            raise ArtifactValidationError("training artifact inventory is unreadable") from exc
        if entries != {
            "manifest.json": (False, True, False),
            "tensors": (False, False, True),
        }:
            raise ArtifactValidationError("training artifact has unexpected files")
        tensors_fd = _fd_child(stack, artifact_fd, "tensors")
        try:
            observed = set(secure_fs.regular_entries_at(tensors_fd))
        except secure_fs.SecureFilesystemError as exc:
            raise ArtifactValidationError("training artifact has invalid tensor inventory") from exc
        expected = {item.filename for item in manifest.tensors}
        if observed != expected:
            raise ArtifactValidationError("training artifact has unexpected tensor files")
        return {
            item.name: _validate_tensor_file_at(tensors_fd, item.filename, item)
            for item in manifest.tensors
        }


def _load_manifest_state_from_at(
    reader: PinnedTrainingArtifactReader, artifact_id: str
) -> tuple[TrainingArtifactManifest, dict[str, torch.Tensor]]:
    if not HEX64.fullmatch(artifact_id):
        raise ArtifactValidationError("invalid training artifact ID")
    with ExitStack() as stack:
        artifact_fd = _fd_child(stack, reader.artifacts_fd, artifact_id)
        manifest = _fd_model(
            TrainingArtifactManifest,
            _fd_json(artifact_fd, "manifest.json"),
            "training artifact manifest",
        )
        assert isinstance(manifest, TrainingArtifactManifest)
        if manifest.artifact_id != artifact_id or manifest.expected_artifact_id != artifact_id:
            raise ArtifactValidationError("training artifact ID mismatch")
        state = _load_artifact_state_at(artifact_fd, manifest)
    return manifest, state


def load_training_manifest_from_at(
    reader: PinnedTrainingArtifactReader, artifact_id: str
) -> TrainingArtifactManifest:
    """Load a manifest and validate its tensors through retained namespaces."""

    manifest, _ = _load_manifest_state_from_at(reader, artifact_id)
    return manifest


def load_training_manifest_at(
    run_fd: int, artifact_id: str
) -> TrainingArtifactManifest:
    """Load a training manifest and tensors from an already-pinned run fd."""

    with open_training_artifact_reader(run_fd) as reader:
        return load_training_manifest_from_at(reader, artifact_id)


def _load_training_key_index_record_from_at(
    reader: PinnedTrainingArtifactReader,
    expected_key: TrainingArtifactKey,
) -> TrainingArtifactKeyIndex:
    index = _fd_model(
        TrainingArtifactKeyIndex,
        _fd_json(reader.keys_fd, f"{expected_key.key_id}.json"),
        "training artifact key index",
    )
    assert isinstance(index, TrainingArtifactKeyIndex)
    if index.key != expected_key or index.key_id != expected_key.key_id:
        raise ArtifactValidationError("training artifact key index does not match expected key")
    return index


def load_training_key_index_from_at(
    reader: PinnedTrainingArtifactReader,
    expected_key: TrainingArtifactKey,
) -> TrainingArtifactKeyIndex:
    """Resolve a key through retained key/artifact namespace descriptors."""

    index = _load_training_key_index_record_from_at(reader, expected_key)
    manifest = load_training_manifest_from_at(reader, index.artifact_id)
    if _digest(manifest.model_dump(mode="json")) != index.manifest_sha256:
        raise ArtifactValidationError("training artifact key index manifest digest mismatch")
    return index


def load_training_key_index_at(
    run_fd: int, expected_key: TrainingArtifactKey
) -> TrainingArtifactKeyIndex:
    """Resolve a training key using only descendants of an already-pinned fd."""

    with open_training_artifact_reader(run_fd) as reader:
        return load_training_key_index_from_at(reader, expected_key)


def _load_training_cost_record_from_at(
    reader: PinnedTrainingArtifactReader,
    expected_key: TrainingArtifactKey,
) -> TrainingArtifactCostRecord:
    record = _fd_model(
        TrainingArtifactCostRecord,
        _fd_json(reader.costs_fd, f"{expected_key.key_id}.json"),
        "training artifact cost record",
    )
    assert isinstance(record, TrainingArtifactCostRecord)
    if record.key != expected_key.model_dump(mode="json") or record.key_id != expected_key.key_id:
        raise ArtifactValidationError("training artifact cost key mismatch")
    if not HEX64.fullmatch(record.artifact_id):
        raise ArtifactValidationError("invalid cost artifact ID")
    return record


def load_training_bundle_from_at(
    reader: PinnedTrainingArtifactReader,
    expected_key: TrainingArtifactKey,
    *,
    model_factory: Callable[[str], nn.Module],
) -> tuple[
    nn.Module,
    TrainingArtifactManifest,
    TrainingArtifactKeyIndex,
    TrainingArtifactCostRecord,
]:
    """Load cost, index, manifest, and tensors under one pinned fd bundle."""

    index = _load_training_key_index_record_from_at(reader, expected_key)
    cost = _load_training_cost_record_from_at(reader, expected_key)
    if cost.artifact_id != index.artifact_id:
        raise ArtifactValidationError("training artifact cost points to the wrong artifact")
    manifest, state = _load_manifest_state_from_at(reader, index.artifact_id)
    if _digest(manifest.model_dump(mode="json")) != index.manifest_sha256:
        raise ArtifactValidationError("training artifact key index manifest digest mismatch")
    if manifest.key != expected_key or manifest.key.key_id != expected_key.key_id:
        raise ArtifactValidationError("training artifact key does not match expected key")
    if manifest.model_id not in MODEL_IDS:
        raise ArtifactValidationError("model ID is not allowlisted")
    try:
        model = model_factory(manifest.model_id)
    except Exception as exc:
        raise ArtifactValidationError("model factory rejected artifact model ID") from exc
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ArtifactValidationError("model state dict does not match artifact") from exc
    model.eval()
    return model, manifest, index, cost


def load_training_lineage_from_at(
    reader: PinnedTrainingArtifactReader,
    expected_key: TrainingArtifactKey,
) -> tuple[
    TrainingArtifactManifest,
    TrainingArtifactKeyIndex,
    TrainingArtifactCostRecord,
]:
    """Validate one cost/index/manifest/tensor lineage without constructing a model."""

    index = _load_training_key_index_record_from_at(reader, expected_key)
    cost = _load_training_cost_record_from_at(reader, expected_key)
    if cost.artifact_id != index.artifact_id:
        raise ArtifactValidationError("training artifact cost points to the wrong artifact")
    manifest = load_training_manifest_from_at(reader, index.artifact_id)
    if _digest(manifest.model_dump(mode="json")) != index.manifest_sha256:
        raise ArtifactValidationError("training artifact key index manifest digest mismatch")
    if manifest.key != expected_key:
        raise ArtifactValidationError("training artifact key does not match expected key")
    return manifest, index, cost


def load_training_cost_from_at(
    reader: PinnedTrainingArtifactReader,
    expected_key: TrainingArtifactKey,
) -> TrainingArtifactCostRecord:
    """Load one complete training lineage through retained namespace fds."""

    _, _, cost = load_training_lineage_from_at(reader, expected_key)
    return cost


def load_training_cost_by_id_from_at(
    reader: PinnedTrainingArtifactReader,
    key_id: str,
) -> TrainingArtifactCostRecord:
    """Resolve a typed training key from a retained cost namespace."""

    if not HEX64.fullmatch(key_id):
        raise ArtifactValidationError("invalid training artifact cost key")
    raw = _fd_model(
        TrainingArtifactCostRecord,
        _fd_json(reader.costs_fd, f"{key_id}.json"),
        "training artifact cost record",
    )
    assert isinstance(raw, TrainingArtifactCostRecord)
    try:
        expected_key = TrainingArtifactKey.model_validate(raw.key)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError("invalid training artifact cost key schema") from exc
    if expected_key.key_id != key_id:
        raise ArtifactValidationError("training artifact cost key identity mismatch")
    return load_training_cost_from_at(reader, expected_key)


def load_training_model_at(
    run_fd: int,
    expected_key: TrainingArtifactKey,
    *,
    model_factory: Callable[[str], nn.Module],
) -> tuple[nn.Module, TrainingArtifactManifest]:
    """Load a fresh model through one pinned run descriptor."""

    with open_training_artifact_reader(run_fd) as reader:
        model, manifest, _, _ = load_training_bundle_from_at(
            reader, expected_key, model_factory=model_factory
        )
        return model, manifest


def load_training_cost_at(
    run_fd: int, expected_key: TrainingArtifactKey
) -> TrainingArtifactCostRecord:
    """Load a training cost and all artifacts it claims from a pinned run fd."""

    with open_training_artifact_reader(run_fd) as reader:
        index = _load_training_key_index_record_from_at(reader, expected_key)
        record = _load_training_cost_record_from_at(reader, expected_key)
        if record.artifact_id != index.artifact_id:
            raise ArtifactValidationError("training artifact cost points to the wrong artifact")
        manifest = load_training_manifest_from_at(reader, record.artifact_id)
        if _digest(manifest.model_dump(mode="json")) != index.manifest_sha256:
            raise ArtifactValidationError("training artifact key index manifest digest mismatch")
        return record


def _validate_artifact_files(artifact_dir: Path, manifest: TrainingArtifactManifest) -> None:
    _, tensor_dir = _validated_artifact_paths(artifact_dir.parent.parent, artifact_dir.name)
    expected = {item.filename for item in manifest.tensors}
    observed = {path.name for path in tensor_dir.iterdir()}
    if observed != expected:
        raise ArtifactValidationError("training artifact has unexpected tensor files")
    for item in manifest.tensors:
        _validate_tensor_file(tensor_dir / item.filename, item)


def _validated_artifact_paths(output: Path, artifact_id: str) -> tuple[Path, Path]:
    if output.is_symlink() or (output / "training-artifacts").is_symlink():
        raise ArtifactValidationError("refusing symlink artifact root")
    artifacts_root = output / "training-artifacts"
    artifact_dir = _safe_resolved_child(artifacts_root, artifacts_root / artifact_id)
    if artifact_dir.is_symlink():
        raise ArtifactValidationError("refusing symlink artifact directory")
    tensor_dir = artifact_dir / "tensors"
    if tensor_dir.is_symlink():
        raise ArtifactValidationError("refusing symlink tensor directory")
    if not tensor_dir.is_dir():
        raise ArtifactValidationError("training artifact tensor directory is invalid")
    _safe_resolved_child(artifact_dir, tensor_dir)
    return artifact_dir, tensor_dir


def load_training_model(
    output_root: str | Path,
    artifact_id: str,
    *,
    expected_key: TrainingArtifactKey,
    model_factory: Callable[[str], nn.Module],
) -> tuple[nn.Module, TrainingArtifactManifest]:
    """Load a fresh allowlisted model using a caller-supplied expected factory."""

    manifest = load_training_manifest(output_root, artifact_id)
    if manifest.key != expected_key or manifest.key.key_id != expected_key.key_id:
        raise ArtifactValidationError("training artifact key does not match expected key")
    if manifest.model_id not in MODEL_IDS:
        raise ArtifactValidationError("model ID is not allowlisted")
    try:
        model = model_factory(manifest.model_id)
    except Exception as exc:
        raise ArtifactValidationError("model factory rejected artifact model ID") from exc
    state: dict[str, torch.Tensor] = {}
    artifact_dir, tensor_dir = _validated_artifact_paths(Path(output_root), artifact_id)
    _validate_artifact_files(artifact_dir, manifest)
    for item in manifest.tensors:
        state[item.name] = _validate_tensor_file(tensor_dir / item.filename, item)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ArtifactValidationError("model state dict does not match artifact") from exc
    model.eval()
    return model, manifest


def load_training_key_index(
    output_root: str | Path, expected_key: TrainingArtifactKey
) -> TrainingArtifactKeyIndex:
    """Resolve an expected key to its immutable artifact without retraining."""

    output = Path(output_root)
    if output.is_symlink() or (output / "training-artifact-keys").is_symlink():
        raise ArtifactValidationError("refusing symlink key-index root")
    index_path = _safe_resolved_child(
        output / "training-artifact-keys",
        output / "training-artifact-keys" / f"{expected_key.key_id}.json",
    )
    raw = _load_json(index_path)
    try:
        index = TrainingArtifactKeyIndex.model_validate(raw)
    except (TypeError, ValueError):
        raise ArtifactValidationError("invalid training artifact key index") from None
    if index.key != expected_key or index.key_id != expected_key.key_id:
        raise ArtifactValidationError("training artifact key index does not match expected key")
    manifest = load_training_manifest(output, index.artifact_id)
    if _digest(manifest.model_dump(mode="json")) != index.manifest_sha256:
        raise ArtifactValidationError("training artifact key index manifest digest mismatch")
    return index


def load_training_cost(
    output_root: str | Path, expected_key: TrainingArtifactKey
) -> TrainingArtifactCostRecord:
    """Load the first-writer cost record for an expected training key."""

    output = Path(output_root)
    costs_root = output / "training-artifact-costs"
    if output.is_symlink() or costs_root.is_symlink():
        raise ArtifactValidationError("refusing symlink cost root")
    path = _safe_resolved_child(costs_root, costs_root / f"{expected_key.key_id}.json")
    raw = _load_json(path)
    try:
        record = TrainingArtifactCostRecord.model_validate(raw)
    except (TypeError, ValueError):
        raise ArtifactValidationError("invalid training artifact cost record") from None
    if record.key != expected_key.model_dump(mode="json") or record.key_id != expected_key.key_id:
        raise ArtifactValidationError("training artifact cost key mismatch")
    if not HEX64.fullmatch(record.artifact_id):
        raise ArtifactValidationError("invalid cost artifact ID")
    index = load_training_key_index(output, expected_key)
    if record.artifact_id != index.artifact_id:
        raise ArtifactValidationError("training artifact cost points to the wrong artifact")
    load_training_manifest(output, record.artifact_id)
    return record
