"""Immutable, descriptor-safe persistence for one Phase 3 trained model.

This module is deliberately narrower than the legacy training-artifact store.  A
Phase 3 artifact carries the complete representation/model lineage (plan,
evidence, view, owner, tuple, optimizer, report, and exact tensor identity) and
is usable only after its manifest, index, cost record, and every tensor have
been revalidated.  It never runs an environment, search, replay, evaluator, or
oracle and it never reads outcome artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from levelup.experiments.milestone6_phase3_models import (
    H0_CONDITION,
    H4_CONDITION,
    H4_SHUFFLED_CONDITION,
    S_CONDITION,
    Phase3ModelPreparation,
    _model_state_sha256,
    validate_phase3_model_preparation,
)
from levelup.experiments.milestone6_phase3_plan import FAMILIES, TRAINING_TUPLE_IDS
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import (
    PhaseAccounting,
    SystemProvenance,
    TrainingPreparationAccounting,
)
from levelup.experiments.runner.storage import ArtifactValidationError, provenance_identity_sha256

HEX64 = r"^[0-9a-f]{64}$"
ARTIFACTS_DIR = "phase3-model-artifacts"
KEYS_DIR = "phase3-model-artifact-keys"
COSTS_DIR = "phase3-model-artifact-costs"
MANIFEST_NAME = "manifest.json"
TENSORS_DIR = "tensors"
STAGING_DIR = ".phase3-model-staging"
PREPARATION_PROVENANCE_NAME = "phase3-model-preparation-provenance.json"
TRAINING_PARAMETERS = {
    "lr0p003-e120": (0.003, 120),
    "lr0p003-e180": (0.003, 180),
    "lr0p01-e120": (0.01, 120),
    "lr0p01-e180": (0.01, 180),
}


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_fd_bytes(directory_fd: int, name: str, payload: bytes, *, staging_fd: int) -> None:
    """Write bytes by fd-relative temp + fsync + hard-link claim."""
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    fd: int | None = None
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=staging_fd,
        )
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short descriptor-relative write")
            view = view[written:]
        os.fsync(fd)
    finally:
        if fd is not None:
            os.close(fd)
    try:
        os.link(
            temporary,
            name,
            src_dir_fd=staging_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileExistsError:
        raise
    finally:
        try:
            os.unlink(temporary, dir_fd=staging_fd)
        except FileNotFoundError:
            pass
        os.fsync(staging_fd)
    os.fsync(directory_fd)


def _exclusive_claim_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    staging_fd: int,
) -> bool:
    try:
        _write_fd_bytes(directory_fd, name, payload, staging_fd=staging_fd)
        return True
    except FileExistsError:
        return False


def _mkdir_unique_at(parent_fd: int, prefix: str) -> str:
    for _ in range(32):
        name = f".{prefix}.{uuid.uuid4().hex}"
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
            return name
        except FileExistsError:
            continue
    raise OSError("cannot allocate unique descriptor-relative staging directory")


def _remove_tree_at(parent_fd: int, name: str) -> None:
    """Remove a private staging tree relative to a held parent fd."""
    try:
        child_fd = secure_fs.open_child_directory(parent_fd, name)
    except secure_fs.SecureFilesystemError:
        try:
            os.unlink(name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        return
    try:
        with os.scandir(child_fd) as iterator:
            entries = tuple(iterator)
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                _remove_tree_at(child_fd, entry.name)
            else:
                os.unlink(entry.name, dir_fd=child_fd)
    finally:
        os.close(child_fd)
    os.rmdir(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


class Phase3ModelArtifactError(ArtifactValidationError):
    """Raised when a Phase 3 model artifact is malformed or substituted."""


class Phase3OptimizerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    optimizer_id: Literal["adam"] = "adam"
    learning_rate: StrictFloat = Field(gt=0)
    weight_decay: StrictFloat = Field(ge=0)


class Phase3TrainingReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trainable_parameters: StrictInt = Field(ge=1)
    optimizer_steps: StrictInt = Field(ge=1)
    forward_passes: StrictInt = Field(ge=1)
    training_examples: StrictInt = Field(ge=1)
    recurrent_steps: StrictInt = Field(ge=0)


class Phase3ModelArtifactKey(BaseModel):
    """Scientific identity of one temperature-independent trained model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["milestone6.phase3.model-key.v1"] = (
        "milestone6.phase3.model-key.v1"
    )
    plan_id: str = Field(pattern=HEX64)
    protocol_sha256: str = Field(pattern=HEX64)
    evidence_lock_sha256: str = Field(pattern=HEX64)
    evidence_payload_sha256: str = Field(pattern=HEX64)
    evidence_payload_bytes: StrictInt = Field(ge=1)
    view_id: str = Field(pattern=HEX64)
    owner_id: str = Field(pattern=HEX64)
    condition_id: str = Field(min_length=1)
    fold_id: str = Field(min_length=1)
    heldout_family: str = Field(min_length=1)
    replicate: StrictInt = Field(ge=0, le=4)
    training_tuple_id: str = Field(min_length=1)
    model_seed: StrictInt
    architecture_id: Literal["state-availability-mlp-v1", "causal-history-gru-mlp-v1"]
    capacity_parameters: StrictInt = Field(ge=1)
    optimizer: Phase3OptimizerSpec
    report: Phase3TrainingReport
    recurrent_steps: StrictInt = Field(ge=0)
    model_identity_sha256: str = Field(pattern=HEX64)
    model_state_sha256: str = Field(pattern=HEX64)
    preparation_git_commit_sha: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    preparation_provenance_sha256: str = Field(pattern=HEX64)

    @model_validator(mode="after")
    def identity_is_consistent(self) -> "Phase3ModelArtifactKey":
        if set(self.preparation_git_commit_sha) == {"0"}:
            raise ValueError("preparation commit provenance is required")
        if set(self.preparation_provenance_sha256) == {"0"}:
            raise ValueError("preparation provenance identity is required")
        if self.condition_id not in {
            S_CONDITION,
            H0_CONDITION,
            H4_CONDITION,
            H4_SHUFFLED_CONDITION,
        }:
            raise ValueError("condition is not in the frozen Phase 3 universe")
        if self.heldout_family not in FAMILIES:
            raise ValueError("held-out family is not in the frozen Phase 3 universe")
        if self.training_tuple_id not in TRAINING_TUPLE_IDS:
            raise ValueError("training tuple is not in the frozen Phase 3 universe")
        expected_arch = (
            "state-availability-mlp-v1"
            if self.condition_id == S_CONDITION
            else "causal-history-gru-mlp-v1"
        )
        expected_capacity = 3841 if self.architecture_id == "state-availability-mlp-v1" else 3889
        if self.architecture_id != expected_arch:
            raise ValueError("architecture does not match Phase 3 condition")
        if self.capacity_parameters != expected_capacity:
            raise ValueError("Phase 3 model capacity drifted")
        if self.report.trainable_parameters != self.capacity_parameters:
            raise ValueError("report parameter count differs from capacity")
        if self.report.recurrent_steps != self.recurrent_steps:
            raise ValueError("report recurrent steps differ from key")
        learning_rate, epochs = TRAINING_PARAMETERS[self.training_tuple_id]
        if (
            self.optimizer.learning_rate != learning_rate
            or self.optimizer.weight_decay != 0.0001
            or self.report.optimizer_steps != epochs
            or self.report.forward_passes
            != epochs * self.report.training_examples
        ):
            raise ValueError("training tuple or report accounting drifted")
        if self.condition_id == S_CONDITION and self.recurrent_steps != 0:
            raise ValueError("state-only model cannot report recurrent steps")
        return self

    @property
    def key_id(self) -> str:
        return _digest(self.model_dump(mode="json"))


class Phase3TensorMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    filename: str = Field(pattern=r"^[0-9]{4}\.bin$")
    shape: tuple[int, ...]
    dtype: Literal["float32"] = "float32"
    byte_length: StrictInt = Field(ge=1)
    sha256: str = Field(pattern=HEX64)

    @model_validator(mode="after")
    def shape_matches_bytes(self) -> "Phase3TensorMetadata":
        if not self.shape or any(dimension < 1 for dimension in self.shape):
            raise ValueError("tensor shape must be non-empty and positive")
        expected = 4
        for dimension in self.shape:
            expected *= dimension
        if expected != self.byte_length:
            raise ValueError("tensor shape and byte length differ")
        return self


class Phase3ModelArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["milestone6.phase3.model-artifact.v1"] = (
        "milestone6.phase3.model-artifact.v1"
    )
    artifact_id: str = Field(pattern=HEX64)
    key: Phase3ModelArtifactKey
    tensors: tuple[Phase3TensorMetadata, ...]
    state_sha256: str = Field(pattern=HEX64)

    @property
    def expected_artifact_id(self) -> str:
        return _digest(self.model_dump(mode="json", exclude={"artifact_id"}))

    @model_validator(mode="after")
    def manifest_is_canonical(self) -> "Phase3ModelArtifactManifest":
        if self.artifact_id != self.expected_artifact_id:
            raise ValueError("artifact ID does not match canonical manifest")
        if self.state_sha256 != self.key.model_state_sha256:
            raise ValueError("manifest state hash differs from key")
        names = [item.name for item in self.tensors]
        filenames = [item.filename for item in self.tensors]
        if not names or names != sorted(names) or len(set(names)) != len(names):
            raise ValueError("tensor names must be unique and sorted")
        if filenames != [f"{index:04d}.bin" for index in range(len(filenames))]:
            raise ValueError("tensor filenames must be contiguous and canonical")
        if len(set(filenames)) != len(filenames):
            raise ValueError("tensor filenames must be unique")
        return self


class Phase3ModelArtifactIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["milestone6.phase3.model-index.v1"] = (
        "milestone6.phase3.model-index.v1"
    )
    key_id: str = Field(pattern=HEX64)
    key: Phase3ModelArtifactKey
    artifact_id: str = Field(pattern=HEX64)
    manifest_sha256: str = Field(pattern=HEX64)

    @model_validator(mode="after")
    def index_is_exact(self) -> "Phase3ModelArtifactIndex":
        if self.key_id != self.key.key_id:
            raise ValueError("model index key ID differs")
        return self


class Phase3ModelArtifactCost(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["milestone6.phase3.model-cost.v1"] = (
        "milestone6.phase3.model-cost.v1"
    )
    cost_id: str = Field(pattern=HEX64)
    key_id: str = Field(pattern=HEX64)
    artifact_id: str = Field(pattern=HEX64)
    scope: Literal["phase3_model_preparation"] = "phase3_model_preparation"
    key: Phase3ModelArtifactKey
    accounting: TrainingPreparationAccounting

    @property
    def expected_cost_id(self) -> str:
        return _digest(self.model_dump(mode="json", exclude={"cost_id"}))

    @model_validator(mode="after")
    def cost_is_exact(self) -> "Phase3ModelArtifactCost":
        if self.key_id != self.key.key_id:
            raise ValueError("cost key ID differs")
        if (
            self.accounting.setup != PhaseAccounting()
            or self.accounting.training_probes != PhaseAccounting()
            or self.accounting.reference_replay != PhaseAccounting()
            or self.accounting.training
            != PhaseAccounting(
                optimizer_steps=self.key.report.optimizer_steps,
                forward_passes=self.key.report.forward_passes,
            )
            or self.accounting.serialization != PhaseAccounting(calls=1)
        ):
            raise ValueError("model cost accounting differs from the training report")
        if self.cost_id != self.expected_cost_id:
            raise ValueError("cost ID does not match canonical cost")
        return self


class Phase3PreparationProvenance(BaseModel):
    """Write-once provenance for the code/environment that trained models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["milestone6.phase3.model-preparation-provenance.v1"] = (
        "milestone6.phase3.model-preparation-provenance.v1"
    )
    provenance: SystemProvenance
    provenance_sha256: str = Field(pattern=HEX64)

    @model_validator(mode="after")
    def identity_is_exact(self) -> "Phase3PreparationProvenance":
        if self.provenance_sha256 != provenance_identity_sha256(self.provenance):
            raise ValueError("Phase 3 preparation provenance identity differs")
        if self.provenance.git_dirty:
            raise ValueError("Phase 3 preparation provenance must be clean")
        return self


@dataclass(frozen=True, slots=True)
class PinnedPhase3ModelArtifactReader:
    keys_fd: int
    costs_fd: int
    artifacts_fd: int


@dataclass(frozen=True, slots=True)
class PinnedPhase3ModelOutput:
    """One output-root pin held for the complete preparation publication."""

    root_fd: int
    keys_fd: int
    costs_fd: int
    artifacts_fd: int
    staging_fd: int
    root_path: Path
    identities: tuple[tuple[int, int], ...]

    @property
    def reader(self) -> PinnedPhase3ModelArtifactReader:
        return PinnedPhase3ModelArtifactReader(
            keys_fd=self.keys_fd,
            costs_fd=self.costs_fd,
            artifacts_fd=self.artifacts_fd,
        )

    def recheck(self) -> None:
        """Fail closed if the lexical root or any namespace was replaced."""
        try:
            current = secure_fs.open_directory_chain(self.root_path)
        except secure_fs.SecureFilesystemError as exc:
            raise Phase3ModelArtifactError(
                "Phase 3 output root or namespace was replaced"
            ) from exc
        try:
            observed = [secure_fs.directory_identity(current)]
            for name in (KEYS_DIR, COSTS_DIR, ARTIFACTS_DIR, STAGING_DIR):
                try:
                    child = secure_fs.open_child_directory(current, name)
                except secure_fs.SecureFilesystemError as exc:
                    raise Phase3ModelArtifactError(
                        "Phase 3 output root or namespace was replaced"
                    ) from exc
                try:
                    observed.append(secure_fs.directory_identity(child))
                finally:
                    os.close(child)
        finally:
            os.close(current)
        if tuple(observed) != self.identities:
            raise Phase3ModelArtifactError("Phase 3 output root or namespace was replaced")


def _mkdir_child(parent_fd: int, name: str) -> None:
    try:
        os.mkdir(name, 0o755, dir_fd=parent_fd)
    except FileExistsError:
        pass
    else:
        os.fsync(parent_fd)


def _open_or_create_directory_chain(path: Path) -> int:
    """Open an absolute directory chain, creating missing components safely."""
    absolute = Path(os.path.abspath(path))
    fd = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=fd,
                )
            except FileNotFoundError:
                os.mkdir(component, 0o755, dir_fd=fd)
                os.fsync(fd)
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=fd,
                )
            os.close(fd)
            fd = child
        result = fd
        fd = -1
        return result
    finally:
        if fd != -1:
            os.close(fd)


@contextmanager
def open_phase3_model_output(
    output_root: str | Path,
) -> Iterator[PinnedPhase3ModelOutput]:
    """Create and pin all Phase 3 output namespaces with directory fds."""
    root_path = Path(os.path.abspath(output_root))
    if root_path.exists() and root_path.is_symlink():
        raise Phase3ModelArtifactError("refusing symlink artifact output root")
    root_fd: int | None = None
    descriptors: list[int] = []
    try:
        root_fd = _open_or_create_directory_chain(root_path)
        for name in (KEYS_DIR, COSTS_DIR, ARTIFACTS_DIR, STAGING_DIR):
            _mkdir_child(root_fd, name)
        keys_fd = secure_fs.open_child_directory(root_fd, KEYS_DIR)
        descriptors.append(keys_fd)
        costs_fd = secure_fs.open_child_directory(root_fd, COSTS_DIR)
        descriptors.append(costs_fd)
        artifacts_fd = secure_fs.open_child_directory(root_fd, ARTIFACTS_DIR)
        descriptors.append(artifacts_fd)
        staging_fd = secure_fs.open_child_directory(root_fd, STAGING_DIR)
        descriptors.append(staging_fd)
        identities = (secure_fs.directory_identity(root_fd),) + tuple(
            secure_fs.directory_identity(fd)
            for fd in (keys_fd, costs_fd, artifacts_fd, staging_fd)
        )
        pinned = PinnedPhase3ModelOutput(
            root_fd=root_fd,
            keys_fd=keys_fd,
            costs_fd=costs_fd,
            artifacts_fd=artifacts_fd,
            staging_fd=staging_fd,
            root_path=root_path,
            identities=identities,
        )
        pinned.recheck()
        yield pinned
    except (secure_fs.SecureFilesystemError, OSError, TypeError, ValueError) as exc:
        if isinstance(exc, Phase3ModelArtifactError):
            raise
        raise Phase3ModelArtifactError("cannot securely pin Phase 3 output") from exc
    finally:
        for fd in reversed(descriptors):
            os.close(fd)
        if root_fd is not None:
            os.close(root_fd)


@contextmanager
def open_phase3_model_artifact_reader(
    output_root: str | Path,
) -> Iterator[PinnedPhase3ModelArtifactReader]:
    """Pin the three Phase 3 artifact namespaces with O_NOFOLLOW descriptors."""

    try:
        root_fd = secure_fs.open_directory_chain(output_root)
    except (secure_fs.SecureFilesystemError, OSError, TypeError, ValueError) as exc:
        raise Phase3ModelArtifactError("cannot securely pin Phase 3 model artifacts") from exc
    try:
        with open_phase3_model_artifact_reader_at(root_fd) as reader:
            yield reader
    finally:
        os.close(root_fd)


@contextmanager
def open_phase3_model_artifact_reader_at(
    root_fd: int,
) -> Iterator[PinnedPhase3ModelArtifactReader]:
    """Pin Phase 3 namespaces below an already-pinned run/output fd."""

    with ExitStack() as stack:
        try:
            descriptors: dict[str, int] = {}
            for field, name in (
                ("keys_fd", KEYS_DIR),
                ("costs_fd", COSTS_DIR),
                ("artifacts_fd", ARTIFACTS_DIR),
            ):
                child = secure_fs.open_child_directory(root_fd, name)
                stack.callback(os.close, child)
                descriptors[field] = child
        except (secure_fs.SecureFilesystemError, OSError, TypeError, ValueError) as exc:
            raise Phase3ModelArtifactError(
                "cannot securely pin Phase 3 model namespaces"
            ) from exc
        yield PinnedPhase3ModelArtifactReader(**descriptors)


def _state_tensors(model: torch.nn.Module) -> dict[str, tuple[tuple[int, ...], bytes]]:
    result: dict[str, tuple[tuple[int, ...], bytes]] = {}
    for name in sorted(model.state_dict()):
        tensor = model.state_dict()[name]
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise Phase3ModelArtifactError("Phase 3 model state must be float32 tensors")
        tensor = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(tensor).all()):
            raise Phase3ModelArtifactError("Phase 3 model contains non-finite state")
        raw = tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
        result[name] = (tuple(int(item) for item in tensor.shape), raw)
    if not result:
        raise Phase3ModelArtifactError("Phase 3 model state is empty")
    return result


def _key_from_preparation(
    preparation: Phase3ModelPreparation,
    *,
    plan_id: str,
    protocol_sha256: str,
    evidence_lock_sha256: str,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
) -> Phase3ModelArtifactKey:
    if (
        not isinstance(preparation_git_commit_sha, str)
        or len(preparation_git_commit_sha) < 40
        or set(preparation_git_commit_sha) == {"0"}
        or not isinstance(preparation_provenance_sha256, str)
        or len(preparation_provenance_sha256) != 64
        or set(preparation_provenance_sha256) == {"0"}
    ):
        raise Phase3ModelArtifactError(
            "Phase 3 model artifacts require nonzero preparation provenance"
        )
    owner = preparation.owner
    view = preparation.view
    model = preparation.model
    architecture = (
        "state-availability-mlp-v1"
        if owner.condition_id == S_CONDITION
        else "causal-history-gru-mlp-v1"
    )
    capacity = 3841 if architecture == "state-availability-mlp-v1" else 3889
    report = Phase3TrainingReport(
        trainable_parameters=preparation.report.trainable_parameters,
        optimizer_steps=preparation.report.optimizer_steps,
        forward_passes=preparation.report.forward_passes,
        training_examples=preparation.report.training_examples,
        recurrent_steps=preparation.report.recurrent_steps,
    )
    observed_state = _model_state_sha256(model)
    if observed_state != preparation.model_state_sha256:
        raise Phase3ModelArtifactError("prepared model state hash changed")
    return Phase3ModelArtifactKey(
        plan_id=plan_id,
        protocol_sha256=protocol_sha256,
        evidence_lock_sha256=evidence_lock_sha256,
        evidence_payload_sha256=view.evidence_payload_sha256,
        evidence_payload_bytes=view.evidence_payload_bytes,
        view_id=view.view.view_id,
        owner_id=owner.owner_id,
        condition_id=owner.condition_id,
        fold_id=owner.fold_id,
        heldout_family=owner.heldout_family,
        replicate=owner.replicate,
        training_tuple_id=owner.training_tuple_id,
        model_seed=owner.model_seed,
        architecture_id=architecture,
        capacity_parameters=capacity,
        optimizer=Phase3OptimizerSpec(
            learning_rate=float(preparation.training_spec.learning_rate),
            weight_decay=float(preparation.training_spec.weight_decay),
        ),
        report=report,
        recurrent_steps=preparation.report.recurrent_steps,
        model_identity_sha256=preparation.model_identity_sha256,
        model_state_sha256=preparation.model_state_sha256,
        preparation_git_commit_sha=preparation_git_commit_sha,
        preparation_provenance_sha256=preparation_provenance_sha256,
    )


def write_phase3_model_artifact(
    output_root: str | Path,
    *,
    preparation: Phase3ModelPreparation,
    plan_id: str,
    protocol_sha256: str,
    evidence_lock_sha256: str,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
    accounting: TrainingPreparationAccounting | None = None,
    pinned_output: PinnedPhase3ModelOutput | None = None,
) -> Phase3ModelArtifactManifest:
    """Atomically publish one validated, content-addressed Phase 3 model."""

    try:
        validate_phase3_model_preparation(preparation, require_authority=True)
    except Exception as exc:
        raise Phase3ModelArtifactError("preparation is not an authority-validated Phase 3 model") from exc
    key = _key_from_preparation(
        preparation,
        plan_id=plan_id,
        protocol_sha256=protocol_sha256,
        evidence_lock_sha256=evidence_lock_sha256,
        preparation_git_commit_sha=preparation_git_commit_sha,
        preparation_provenance_sha256=preparation_provenance_sha256,
    )
    tensors = _state_tensors(preparation.model)
    metadata = tuple(
        Phase3TensorMetadata(
            name=name,
            filename=f"{index:04d}.bin",
            shape=shape,
            byte_length=len(payload),
            sha256=_sha_bytes(payload),
        )
        for index, (name, (shape, payload)) in enumerate(tensors.items())
    )
    body = {
        "schema_version": "milestone6.phase3.model-artifact.v1",
        "key": key.model_dump(mode="json"),
        "tensors": [item.model_dump(mode="json") for item in metadata],
        "state_sha256": key.model_state_sha256,
    }
    artifact_id = _digest(body)
    manifest = Phase3ModelArtifactManifest(
        artifact_id=artifact_id,
        key=key,
        tensors=metadata,
        state_sha256=key.model_state_sha256,
    )
    own_pin = pinned_output is None
    with (open_phase3_model_output(output_root) if own_pin else nullcontext(pinned_output)) as output:
        assert output is not None
        output.recheck()
        reader = output.reader
        try:
            artifact_fd: int | None = None
            try:
                artifact_fd = secure_fs.open_child_directory(output.artifacts_fd, artifact_id)
            except secure_fs.SecureFilesystemError:
                artifact_fd = None
            if artifact_fd is not None:
                os.close(artifact_fd)
                loaded = load_phase3_model_manifest_at(reader, artifact_id)
                if loaded != manifest:
                    raise Phase3ModelArtifactError("existing Phase 3 artifact conflicts")
            else:
                staging_name = _mkdir_unique_at(output.staging_fd, artifact_id)
                staging_fd: int | None = None
                try:
                    staging_fd = secure_fs.open_child_directory(output.staging_fd, staging_name)
                    tensor_name = TENSORS_DIR
                    _mkdir_child(staging_fd, tensor_name)
                    tensor_fd = secure_fs.open_child_directory(staging_fd, tensor_name)
                    try:
                        for item in metadata:
                            _write_fd_bytes(
                                tensor_fd,
                                item.filename,
                                tensors[item.name][1],
                                staging_fd=output.staging_fd,
                            )
                        os.fsync(tensor_fd)
                    finally:
                        os.close(tensor_fd)
                    _write_fd_bytes(
                        staging_fd,
                        MANIFEST_NAME,
                        canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n",
                        staging_fd=output.staging_fd,
                    )
                    os.fsync(staging_fd)
                    output.recheck()
                    try:
                        os.rename(
                            staging_name,
                            artifact_id,
                            src_dir_fd=output.staging_fd,
                            dst_dir_fd=output.artifacts_fd,
                        )
                    except FileExistsError:
                        pass
                    else:
                        os.fsync(output.artifacts_fd)
                finally:
                    if staging_fd is not None:
                        os.close(staging_fd)
                    try:
                        _remove_tree_at(output.staging_fd, staging_name)
                    except (FileNotFoundError, OSError):
                        pass
                loaded = load_phase3_model_manifest_at(reader, artifact_id)
                if loaded != manifest:
                    raise Phase3ModelArtifactError("racing Phase 3 artifact conflicts")
            output.recheck()
            # The key index is the commit marker. Artifact + cost are written
            # before the index is claimed last.
            cost_accounting = accounting or TrainingPreparationAccounting(
                training=PhaseAccounting(
                    optimizer_steps=key.report.optimizer_steps,
                    forward_passes=key.report.forward_passes,
                ),
                serialization=PhaseAccounting(calls=1),
            )
            cost_body = {
                "schema_version": "milestone6.phase3.model-cost.v1",
                "key_id": key.key_id,
                "artifact_id": artifact_id,
                "scope": "phase3_model_preparation",
                "key": key.model_dump(mode="json"),
                "accounting": cost_accounting.model_dump(mode="json"),
            }
            cost = Phase3ModelArtifactCost.model_validate({"cost_id": _digest(cost_body), **cost_body})
            if not _exclusive_claim_at(
                output.costs_fd,
                f"{key.key_id}.json",
                canonical_json_bytes(cost.model_dump(mode="json")) + b"\n",
                staging_fd=output.staging_fd,
            ):
                loaded_cost = _fd_model(
                    Phase3ModelArtifactCost,
                    _fd_json(output.costs_fd, f"{key.key_id}.json"),
                    "model cost",
                )
                if loaded_cost.artifact_id != artifact_id:
                    raise Phase3ModelArtifactError("different cost won Phase 3 key race")
            output.recheck()
            index = Phase3ModelArtifactIndex(
                key_id=key.key_id,
                key=key,
                artifact_id=artifact_id,
                manifest_sha256=_digest(loaded.model_dump(mode="json")),
            )
            if not _exclusive_claim_at(
                output.keys_fd,
                f"{key.key_id}.json",
                canonical_json_bytes(index.model_dump(mode="json")) + b"\n",
                staging_fd=output.staging_fd,
            ):
                winner = _fd_model(
                    Phase3ModelArtifactIndex,
                    _fd_json(output.keys_fd, f"{key.key_id}.json"),
                    "model index",
                )
                if winner.artifact_id != artifact_id:
                    raise Phase3ModelArtifactError("different artifact won Phase 3 key race")
            output.recheck()
            _, _, final_manifest, _ = load_phase3_model_bundle_from_at(reader, key)
            return final_manifest
        except (secure_fs.SecureFilesystemError, OSError) as exc:
            raise Phase3ModelArtifactError("descriptor-relative Phase 3 publication failed") from exc


def load_phase3_model_manifest(
    output_root: str | Path, artifact_id: str
) -> Phase3ModelArtifactManifest:
    if len(artifact_id) != 64 or any(c not in "0123456789abcdef" for c in artifact_id):
        raise Phase3ModelArtifactError("invalid Phase 3 artifact ID")
    # Resolve the path only once, then keep all descendants pinned by fd.  This
    # is also the implementation used by execution authorities; the path form
    # is retained as a convenience for preparation tooling.
    with open_phase3_model_artifact_reader(output_root) as reader:
        return load_phase3_model_manifest_at(reader, artifact_id)


def _fd_json(directory_fd: int, name: str) -> dict[str, Any]:
    try:
        content = secure_fs.read_bytes_at(directory_fd, name)
        value = json.loads(content)
    except (secure_fs.SecureFilesystemError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase3ModelArtifactError(f"invalid Phase 3 JSON: {name}") from exc
    if not isinstance(value, dict):
        raise Phase3ModelArtifactError(f"Phase 3 JSON is not an object: {name}")
    if content != canonical_json_bytes(value) + b"\n":
        raise Phase3ModelArtifactError(f"non-canonical Phase 3 JSON: {name}")
    return value


def _fd_model(model_type: type[BaseModel], raw: Any, label: str) -> BaseModel:
    try:
        return model_type.model_validate(raw)
    except (TypeError, ValueError) as exc:
        raise Phase3ModelArtifactError(f"invalid Phase 3 {label}") from exc


def _load_manifest_at(reader: PinnedPhase3ModelArtifactReader, artifact_id: str) -> Phase3ModelArtifactManifest:
    try:
        artifact_fd = secure_fs.open_child_directory(reader.artifacts_fd, artifact_id)
    except secure_fs.SecureFilesystemError as exc:
        raise Phase3ModelArtifactError("cannot securely open Phase 3 artifact") from exc
    try:
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
            raise Phase3ModelArtifactError("Phase 3 artifact inventory is unreadable") from exc
        if entries != {
            MANIFEST_NAME: (False, True, False),
            TENSORS_DIR: (False, False, True),
        }:
            raise Phase3ModelArtifactError("Phase 3 artifact inventory differs")
        manifest = _fd_model(
            Phase3ModelArtifactManifest,
            _fd_json(artifact_fd, MANIFEST_NAME),
            "model manifest",
        )
        assert isinstance(manifest, Phase3ModelArtifactManifest)
        if manifest.artifact_id != artifact_id:
            raise Phase3ModelArtifactError("Phase 3 artifact ID mismatch")
        tensors_fd = secure_fs.open_child_directory(artifact_fd, TENSORS_DIR)
        try:
            try:
                observed = set(secure_fs.regular_entries_at(tensors_fd))
            except secure_fs.SecureFilesystemError as exc:
                raise Phase3ModelArtifactError(
                    "Phase 3 tensor inventory is not regular"
                ) from exc
            expected = {item.filename for item in manifest.tensors}
            if observed != expected:
                raise Phase3ModelArtifactError("Phase 3 tensor inventory differs")
            for item in manifest.tensors:
                try:
                    payload = secure_fs.read_bytes_at(tensors_fd, item.filename)
                except secure_fs.SecureFilesystemError as exc:
                    raise Phase3ModelArtifactError("cannot read Phase 3 tensor") from exc
                if len(payload) != item.byte_length or _sha_bytes(payload) != item.sha256:
                    raise Phase3ModelArtifactError(f"Phase 3 tensor hash mismatch: {item.name}")
        finally:
            os.close(tensors_fd)
        return manifest
    finally:
        os.close(artifact_fd)


def load_phase3_model_manifest_at(
    reader: PinnedPhase3ModelArtifactReader, artifact_id: str
) -> Phase3ModelArtifactManifest:
    return _load_manifest_at(reader, artifact_id)


def load_phase3_model_index(
    output_root: str | Path, expected_key: Phase3ModelArtifactKey
) -> Phase3ModelArtifactIndex:
    with open_phase3_model_artifact_reader(output_root) as reader:
        index = _fd_model(
            Phase3ModelArtifactIndex,
            _fd_json(reader.keys_fd, f"{expected_key.key_id}.json"),
            "model index",
        )
        assert isinstance(index, Phase3ModelArtifactIndex)
        if index.key != expected_key:
            raise Phase3ModelArtifactError("Phase 3 model index key mismatch")
        manifest = _load_manifest_at(reader, index.artifact_id)
        if _digest(manifest.model_dump(mode="json")) != index.manifest_sha256:
            raise Phase3ModelArtifactError("Phase 3 model index manifest digest mismatch")
        return index


def load_phase3_model_index_at(
    reader: PinnedPhase3ModelArtifactReader,
    key_id: str,
) -> Phase3ModelArtifactIndex:
    """Load one canonical key-index commit marker through pinned descriptors."""

    if len(key_id) != 64 or any(c not in "0123456789abcdef" for c in key_id):
        raise Phase3ModelArtifactError("invalid Phase 3 key ID")
    index = _fd_model(
        Phase3ModelArtifactIndex,
        _fd_json(reader.keys_fd, f"{key_id}.json"),
        "model index",
    )
    assert isinstance(index, Phase3ModelArtifactIndex)
    if index.key_id != key_id or index.key.key_id != key_id:
        raise Phase3ModelArtifactError("Phase 3 model index key ID mismatch")
    return index


def load_phase3_model_cost(
    output_root: str | Path, expected_key: Phase3ModelArtifactKey
) -> Phase3ModelArtifactCost:
    with open_phase3_model_artifact_reader(output_root) as reader:
        cost = _fd_model(
            Phase3ModelArtifactCost,
            _fd_json(reader.costs_fd, f"{expected_key.key_id}.json"),
            "model cost",
        )
        assert isinstance(cost, Phase3ModelArtifactCost)
        if cost.key != expected_key:
            raise Phase3ModelArtifactError("Phase 3 model cost key mismatch")
        return cost


def load_phase3_model_bundle_from_at(
    reader: PinnedPhase3ModelArtifactReader,
    expected_key: Phase3ModelArtifactKey,
) -> tuple[Phase3ModelArtifactIndex, Phase3ModelArtifactCost, Phase3ModelArtifactManifest, dict[str, torch.Tensor]]:
    """Reload index, cost, manifest, and tensors through one pinned namespace set."""

    index = _fd_model(
        Phase3ModelArtifactIndex,
        _fd_json(reader.keys_fd, f"{expected_key.key_id}.json"),
        "model index",
    )
    assert isinstance(index, Phase3ModelArtifactIndex)
    if index.key != expected_key:
        raise Phase3ModelArtifactError("Phase 3 model index key mismatch")
    cost = _fd_model(
        Phase3ModelArtifactCost,
        _fd_json(reader.costs_fd, f"{expected_key.key_id}.json"),
        "model cost",
    )
    assert isinstance(cost, Phase3ModelArtifactCost)
    if cost.key != expected_key or cost.artifact_id != index.artifact_id:
        raise Phase3ModelArtifactError("Phase 3 model cost lineage mismatch")
    manifest = _load_manifest_at(reader, index.artifact_id)
    if manifest.key != expected_key or _digest(manifest.model_dump(mode="json")) != index.manifest_sha256:
        raise Phase3ModelArtifactError("Phase 3 model manifest lineage mismatch")
    with ExitStack() as stack:
        artifact_fd = secure_fs.open_child_directory(reader.artifacts_fd, index.artifact_id)
        stack.callback(os.close, artifact_fd)
        tensors_fd = secure_fs.open_child_directory(artifact_fd, TENSORS_DIR)
        stack.callback(os.close, tensors_fd)
        state: dict[str, torch.Tensor] = {}
        for item in manifest.tensors:
            try:
                payload = secure_fs.read_bytes_at(tensors_fd, item.filename)
            except secure_fs.SecureFilesystemError as exc:
                raise Phase3ModelArtifactError("cannot reload Phase 3 tensor") from exc
            values = np.frombuffer(payload, dtype="<f4").reshape(item.shape).copy()
            if not bool(np.isfinite(values).all()):
                raise Phase3ModelArtifactError(
                    f"Phase 3 tensor contains non-finite values: {item.name}"
                )
            state[item.name] = torch.from_numpy(values)
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        header = canonical_json_bytes({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)})
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    if digest.hexdigest() != manifest.state_sha256:
        raise Phase3ModelArtifactError("Phase 3 reconstructed state hash mismatch")
    return index, cost, manifest, state


def load_phase3_model_from_at(
    reader: PinnedPhase3ModelArtifactReader,
    expected_key: Phase3ModelArtifactKey,
    *,
    model_factory: Callable[[str], torch.nn.Module],
) -> tuple[torch.nn.Module, Phase3ModelArtifactIndex, Phase3ModelArtifactCost, Phase3ModelArtifactManifest]:
    """Reconstruct one model only after the complete pinned bundle validates."""

    index, cost, manifest, state = load_phase3_model_bundle_from_at(reader, expected_key)
    try:
        model = model_factory(manifest.key.architecture_id)
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise Phase3ModelArtifactError("Phase 3 model architecture/state mismatch") from exc
    if _model_state_sha256(model) != manifest.state_sha256:
        raise Phase3ModelArtifactError("Phase 3 loaded model state hash mismatch")
    return model, index, cost, manifest


__all__ = [
    "ARTIFACTS_DIR",
    "COSTS_DIR",
    "KEYS_DIR",
    "Phase3ModelArtifactCost",
    "Phase3ModelArtifactError",
    "Phase3ModelArtifactIndex",
    "Phase3ModelArtifactKey",
    "Phase3PreparationProvenance",
    "PREPARATION_PROVENANCE_NAME",
    "Phase3ModelArtifactManifest",
    "Phase3OptimizerSpec",
    "Phase3TensorMetadata",
    "Phase3TrainingReport",
    "PinnedPhase3ModelArtifactReader",
    "PinnedPhase3ModelOutput",
    "STAGING_DIR",
    "load_phase3_model_bundle_from_at",
    "load_phase3_model_cost",
    "load_phase3_model_index",
    "load_phase3_model_manifest",
    "load_phase3_model_manifest_at",
    "load_phase3_model_from_at",
    "load_phase3_model_index_at",
    "open_phase3_model_artifact_reader",
    "open_phase3_model_artifact_reader_at",
    "open_phase3_model_output",
    "write_phase3_model_artifact",
]
