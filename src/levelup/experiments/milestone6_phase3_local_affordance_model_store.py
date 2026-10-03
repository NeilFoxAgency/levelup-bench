"""Crash-conscious storage for one trained local-affordance model.

This store is intentionally owner-at-a-time.  It persists only safe tensor
bytes, the typed model record, and a canonical manifest.  A manifest replace is
the commit marker; no training, search, outcome, evaluator, oracle, or final
family API is exposed here.  Full 480-owner inventory/activation remains a
separate batch gate and is not implied by this single-owner store.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from levelup.experiments.milestone6_phase3_local_affordance_model_records import (
    LocalAffordanceModelRecord,
    validate_model_record,
)
from levelup.experiments.milestone6_phase3_local_affordance_models import (
    ARCHITECTURE_ID_BY_CONDITION,
    CONDITION_B2,
    PARAMETERS_BY_CONDITION,
    WEIGHT_DECAY,
    LocalAffordanceModelPreparation,
    _model_state_sha256,
    require_local_affordance_model_preparation,
)
from levelup.experiments.milestone6_phase3_local_affordance_plan import LocalAffordancePlan
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.learning.state_conditioned import (
    GlobalAffordanceScorer,
    StateConditionedScorer,
    TrainingReport,
)

SCHEMA_VERSION = "milestone6.phase3.local-affordance-model-store.v1"
MANIFEST_NAME = "manifest.json"
RECORDS_DIR = "records"
MODELS_DIR = "models"
STAGING_DIR = "staging"
MODEL_MAGIC = b"LAFSAFE1\n"
HEX64 = r"^[0-9a-f]{64}$"
_STORE_TOKEN = object()


class LocalAffordanceModelStoreError(ValueError):
    """Raised when local-affordance model storage is incomplete or unsafe."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object) -> str:
    return _sha_bytes(canonical_json_bytes(value))


def _canonical(model: BaseModel) -> bytes:
    return canonical_json_bytes(model.model_dump(mode="json")) + b"\n"


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    if not stat.S_ISREG(value.st_mode):
        raise LocalAffordanceModelStoreError("store entry is not a regular file")
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read_stable(directory_fd: int, name: str) -> bytes:
    try:
        with secure_fs.open_regular_file_at(directory_fd, name) as fd:
            before = _file_identity(os.fstat(fd))
            if before != _file_identity(os.stat(name, dir_fd=directory_fd, follow_symlinks=False)):
                raise LocalAffordanceModelStoreError("store file identity changed before read")
            chunks: list[bytes] = []
            while chunk := os.read(fd, 1024 * 1024):
                chunks.append(chunk)
            content = b"".join(chunks)
            after = _file_identity(os.fstat(fd))
            if (
                before != after
                or after != _file_identity(os.stat(name, dir_fd=directory_fd, follow_symlinks=False))
                or len(content) != after[3]
            ):
                raise LocalAffordanceModelStoreError("store file changed during read")
            return content
    except LocalAffordanceModelStoreError:
        raise
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise LocalAffordanceModelStoreError(f"store file is missing or unsafe: {name}") from exc


def _stable_file_identity(directory_fd: int, name: str) -> tuple[int, int, int, int, int, int]:
    try:
        with secure_fs.open_regular_file_at(directory_fd, name) as fd:
            first = _file_identity(os.fstat(fd))
            if first != _file_identity(os.stat(name, dir_fd=directory_fd, follow_symlinks=False)):
                raise LocalAffordanceModelStoreError("store file was replaced")
            last = _file_identity(os.fstat(fd))
            if first != last:
                raise LocalAffordanceModelStoreError("store file changed in place")
            return first
    except LocalAffordanceModelStoreError:
        raise
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise LocalAffordanceModelStoreError(f"store file is missing or unsafe: {name}") from exc


def _caused_by_missing(value: BaseException) -> bool:
    current: BaseException | None = value
    while current is not None:
        if isinstance(current, FileNotFoundError):
            return True
        current = current.__cause__
    return False


def _write_temp(directory_fd: int, staging_fd: int, name: str, content: bytes) -> None:
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    fd: int | None = None
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=staging_fd)
        view = memoryview(content)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError("short write")
            view = view[count:]
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary, name, src_dir_fd=staging_fd, dst_dir_fd=directory_fd)
        os.fsync(staging_fd)
        os.fsync(directory_fd)
    except OSError as exc:
        raise LocalAffordanceModelStoreError(f"cannot publish store file: {name}") from exc
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary, dir_fd=staging_fd)
        except FileNotFoundError:
            pass


def _claim(directory_fd: int, staging_fd: int, name: str, content: bytes) -> None:
    temporary = f".{name}.{uuid.uuid4().hex}.claim"
    fd: int | None = None
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=staging_fd)
        view = memoryview(content)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError("short write")
            view = view[count:]
        os.fsync(fd)
        try:
            os.link(temporary, name, src_dir_fd=staging_fd, dst_dir_fd=directory_fd, follow_symlinks=False)
        except FileExistsError:
            if _read_stable(directory_fd, name) != content:
                raise LocalAffordanceModelStoreError("different artifact already owns this model ID")
        else:
            os.fsync(directory_fd)
    except LocalAffordanceModelStoreError:
        raise
    except OSError as exc:
        raise LocalAffordanceModelStoreError(f"cannot claim store file: {name}") from exc
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary, dir_fd=staging_fd)
        except FileNotFoundError:
            pass


class LocalAffordanceStoreEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    owner_id: str = Field(pattern=HEX64)
    record_sha256: str = Field(pattern=HEX64)
    key_sha256: str = Field(pattern=HEX64)
    cost_sha256: str = Field(pattern=HEX64)
    artifact_sha256: str = Field(pattern=HEX64)


class LocalAffordanceStoreManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    manifest_sha256: str = Field(pattern=HEX64)
    entries: tuple[LocalAffordanceStoreEntry, ...] = ()

    @model_validator(mode="after")
    def canonical_identity(self) -> "LocalAffordanceStoreManifest":
        if self.manifest_sha256 != _sha(self.model_dump(mode="json", exclude={"manifest_sha256"})):
            raise ValueError("store manifest self-hash mismatch")
        ids = tuple(item.owner_id for item in self.entries)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("store manifest entries are not sorted and unique")
        return self


@dataclass(frozen=True, slots=True)
class LocalAffordanceTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes


@dataclass(frozen=True, slots=True, init=False)
class PinnedLocalAffordanceModelStore:
    root_fd: int
    records_fd: int
    models_fd: int
    staging_fd: int
    root_path: Path
    identities: tuple[tuple[int, int], ...]
    _token: object

    def __init__(self, root_fd: int, records_fd: int, models_fd: int, staging_fd: int, root_path: Path,
                 identities: tuple[tuple[int, int], ...], *, _token: object | None = None) -> None:
        if _token is not _STORE_TOKEN:
            raise LocalAffordanceModelStoreError("store requires canonical descriptor pinning")
        for name, value in (("root_fd", root_fd), ("records_fd", records_fd), ("models_fd", models_fd), ("staging_fd", staging_fd), ("root_path", root_path), ("identities", identities)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_token", _STORE_TOKEN)

    def recheck(self) -> None:
        if self._token is not _STORE_TOKEN:
            raise LocalAffordanceModelStoreError("store authority is invalid")
        try:
            held = tuple(secure_fs.directory_identity(fd) for fd in (self.root_fd, self.records_fd, self.models_fd, self.staging_fd))
            path_fd = secure_fs.open_directory_chain(self.root_path)
            try:
                on_path = [secure_fs.directory_identity(path_fd)]
                for name in (RECORDS_DIR, MODELS_DIR, STAGING_DIR):
                    child_fd = secure_fs.open_child_directory(path_fd, name)
                    try:
                        on_path.append(secure_fs.directory_identity(child_fd))
                    finally:
                        os.close(child_fd)
            finally:
                os.close(path_fd)
        except (OSError, secure_fs.SecureFilesystemError) as exc:
            raise LocalAffordanceModelStoreError("store root or namespace was replaced") from exc
        if held != self.identities or tuple(on_path) != self.identities:
            raise LocalAffordanceModelStoreError("store root or namespace identity changed")


@contextmanager
def open_local_affordance_model_store(
    root: str | Path, *, exclusive_writer: bool = False, create: bool = True
) -> Iterator[PinnedLocalAffordanceModelStore]:
    """Pin the dedicated store, optionally refusing to create any directory."""
    path = Path(os.path.abspath(root))
    if os.path.lexists(path) and path.is_symlink():
        raise LocalAffordanceModelStoreError("refusing symlink store root")
    parent = path.parent
    if not parent.exists():
        raise LocalAffordanceModelStoreError("store parent must already exist")
    try:
        parent_fd = secure_fs.open_directory_chain(parent)
        try:
            if create:
                try:
                    os.mkdir(path.name, 0o700, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                except FileExistsError:
                    pass
        finally:
            os.close(parent_fd)
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise LocalAffordanceModelStoreError("cannot safely create store root") from exc
    with ExitStack() as stack:
        try:
            root_fd = secure_fs.open_directory_chain(path)
            stack.callback(os.close, root_fd)
            child: dict[str, int] = {}
            for name in (RECORDS_DIR, MODELS_DIR, STAGING_DIR):
                if create:
                    try:
                        os.mkdir(name, 0o700, dir_fd=root_fd)
                        os.fsync(root_fd)
                    except FileExistsError:
                        pass
                child[name] = secure_fs.open_child_directory(root_fd, name)
                stack.callback(os.close, child[name])
            identities = tuple(secure_fs.directory_identity(fd) for fd in (root_fd, child[RECORDS_DIR], child[MODELS_DIR], child[STAGING_DIR]))
            pinned = PinnedLocalAffordanceModelStore(root_fd, child[RECORDS_DIR], child[MODELS_DIR], child[STAGING_DIR], path, identities, _token=_STORE_TOKEN)
            pinned.recheck()
            if exclusive_writer:
                try:
                    fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise LocalAffordanceModelStoreError(
                        "another model writer owns this store"
                    ) from exc
                stack.callback(fcntl.flock, root_fd, fcntl.LOCK_UN)
            yield pinned
        except LocalAffordanceModelStoreError:
            raise
        except (OSError, TypeError, ValueError, secure_fs.SecureFilesystemError) as exc:
            raise LocalAffordanceModelStoreError("cannot securely open model-store namespaces") from exc


def _manifest(
    reader: PinnedLocalAffordanceModelStore, *, allow_missing_for_recovery: bool = False
) -> LocalAffordanceStoreManifest:
    try:
        raw = _read_stable(reader.root_fd, MANIFEST_NAME)
    except LocalAffordanceModelStoreError as exc:
        if _caused_by_missing(exc):
            if not allow_missing_for_recovery and any(
                secure_fs.strict_regular_entries(fd)
                for fd in (reader.records_fd, reader.models_fd, reader.staging_fd)
            ):
                raise LocalAffordanceModelStoreError("store has orphaned files without a commit manifest") from exc
            body = {"schema_version": SCHEMA_VERSION, "manifest_sha256": "0" * 64, "entries": []}
            body["manifest_sha256"] = _sha({key: value for key, value in body.items() if key != "manifest_sha256"})
            return LocalAffordanceStoreManifest.model_validate(body)
        raise
    try:
        value = json.loads(raw)
        if canonical_json_bytes(value) + b"\n" != raw:
            raise ValueError
        return LocalAffordanceStoreManifest.model_validate(value)
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalAffordanceModelStoreError("manifest is not canonical") from exc


def _validate_preparation(plan: LocalAffordancePlan, prepared: LocalAffordanceModelPreparation) -> None:
    try:
        require_local_affordance_model_preparation(prepared, plan=plan)
    except (TypeError, ValueError, AttributeError) as exc:
        raise LocalAffordanceModelStoreError(
            "model store requires a canonical sealed preparation"
        ) from exc
    if type(prepared) is not LocalAffordanceModelPreparation or prepared.plan_id != plan.plan_id:
        raise LocalAffordanceModelStoreError("typed preparation is not bound to this plan")
    owners = tuple(owner for owner in plan.model_owners if owner.owner_id == prepared.owner.owner_id)
    views = tuple(view for view in plan.views if view.view_id == prepared.view.view_id)
    if owners != (prepared.owner,) or views != (prepared.view,) or prepared.owner.view_id != prepared.view.view_id:
        raise LocalAffordanceModelStoreError("preparation owner/view is not uniquely plan-bound")
    model_type = GlobalAffordanceScorer if prepared.owner.condition_id == CONDITION_B2 else StateConditionedScorer
    parameters = PARAMETERS_BY_CONDITION[prepared.owner.condition_id]
    expected_report = TrainingReport(parameters, prepared.owner.training_epochs,
                                     prepared.owner.training_epochs * prepared.report.training_examples,
                                     prepared.report.training_examples)
    if (
        type(prepared.model) is not model_type
        or sum(parameter.numel() for parameter in prepared.model.parameters()) != parameters
        or prepared.report != expected_report
        or prepared.training_spec.epochs != prepared.owner.training_epochs
        or prepared.training_spec.learning_rate != prepared.owner.learning_rate
        or prepared.training_spec.weight_decay != WEIGHT_DECAY
        or prepared.search_temperature_ids != prepared.owner.search_temperature_ids
        or prepared.model_state_sha256 != _model_state_sha256(prepared.model)
    ):
        raise LocalAffordanceModelStoreError("prepared model architecture, cost, or state hash differs")
    identity = _sha({
        "schema_version": "milestone6.phase3.local-affordance-model-preparation.v1",
        "plan_id": prepared.plan_id,
        "owner_id": prepared.owner.owner_id,
        "view_id": prepared.view.view_id,
        "condition_id": prepared.owner.condition_id,
        "training_tuple_id": prepared.owner.training_tuple_id,
        "model_seed": prepared.owner.model_seed,
        "training_spec": {"epochs": prepared.training_spec.epochs, "learning_rate": prepared.training_spec.learning_rate, "weight_decay": prepared.training_spec.weight_decay},
        "architecture_id": ARCHITECTURE_ID_BY_CONDITION[prepared.owner.condition_id],
        "trainable_parameters": parameters,
        "training_examples_sha256": prepared.training_examples_sha256,
        "training_report": {"optimizer_steps": prepared.report.optimizer_steps, "forward_passes": prepared.report.forward_passes, "training_examples": prepared.report.training_examples},
        "model_state_sha256": prepared.model_state_sha256,
    })
    if identity != prepared.model_identity_sha256:
        raise LocalAffordanceModelStoreError("preparation identity hash differs")


def _encode_model(prepared: LocalAffordanceModelPreparation) -> bytes:
    descriptors: list[dict[str, object]] = []
    bodies: list[bytes] = []
    offset = 0
    for name, tensor in sorted(prepared.model.state_dict().items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise LocalAffordanceModelStoreError("model state contains an unsupported tensor")
        value = tensor.detach().cpu().contiguous()
        try:
            data = value.numpy().tobytes(order="C")
        except (TypeError, RuntimeError) as exc:
            raise LocalAffordanceModelStoreError("model tensor cannot be represented as safe bytes") from exc
        descriptors.append({"name": name, "dtype": str(value.dtype), "shape": list(value.shape), "offset": offset,
                            "byte_length": len(data), "sha256": _sha_bytes(data)})
        bodies.append(data)
        offset += len(data)
    header = canonical_json_bytes({"schema_version": "milestone6.phase3.local-affordance-safe-tensors.v1", "tensors": descriptors})
    return MODEL_MAGIC + len(header).to_bytes(8, "big") + header + b"".join(bodies)


def serialize_local_affordance_model(
    plan: LocalAffordancePlan, prepared: LocalAffordanceModelPreparation
) -> bytes:
    """Return the deterministic, non-pickle payload used by record metadata."""
    try:
        _validate_preparation(plan, prepared)
        return _encode_model(prepared)
    except LocalAffordanceModelStoreError:
        raise
    except (TypeError, ValueError, AttributeError) as exc:
        raise LocalAffordanceModelStoreError("model preparation is not plan-authorized") from exc


def _decode_model(payload: bytes, expected_state_sha256: str) -> tuple[LocalAffordanceTensor, ...]:
    if not payload.startswith(MODEL_MAGIC) or len(payload) < len(MODEL_MAGIC) + 8:
        raise LocalAffordanceModelStoreError("model artifact header is invalid")
    cursor = len(MODEL_MAGIC)
    header_size = int.from_bytes(payload[cursor:cursor + 8], "big")
    cursor += 8
    if header_size <= 0 or cursor + header_size > len(payload):
        raise LocalAffordanceModelStoreError("model artifact header length is invalid")
    header_bytes = payload[cursor:cursor + header_size]
    try:
        header = json.loads(header_bytes)
        if canonical_json_bytes(header) != header_bytes or header["schema_version"] != "milestone6.phase3.local-affordance-safe-tensors.v1":
            raise ValueError
        descriptions = header["tensors"]
        if not isinstance(descriptions, list) or not descriptions:
            raise ValueError
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalAffordanceModelStoreError("model tensor index is invalid") from exc
    data_start = cursor + header_size
    tensors: list[LocalAffordanceTensor] = []
    digest = hashlib.sha256()
    expected_offset = 0
    previous_name = ""
    for item in descriptions:
        try:
            name, dtype, shape = item["name"], item["dtype"], item["shape"]
            offset, byte_length, sha = item["offset"], item["byte_length"], item["sha256"]
            if (
                not isinstance(name, str) or not name or name <= previous_name
                or dtype != "torch.float32" or not isinstance(shape, list)
                or not shape or any(type(dim) is not int or dim <= 0 for dim in shape)
                or type(offset) is not int or offset != expected_offset
                or type(byte_length) is not int or byte_length != 4 * _shape_size(shape)
                or not isinstance(sha, str) or len(sha) != 64
            ):
                raise ValueError
            data = payload[data_start + offset:data_start + offset + byte_length]
            if len(data) != byte_length or _sha_bytes(data) != sha:
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise LocalAffordanceModelStoreError("model tensor bytes or schema differ") from exc
        tensor = LocalAffordanceTensor(name, dtype, tuple(shape), data)
        tensors.append(tensor)
        header_tensor = canonical_json_bytes({"name": name, "dtype": dtype, "shape": shape})
        digest.update(len(header_tensor).to_bytes(8, "big"))
        digest.update(header_tensor)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        expected_offset += byte_length
        previous_name = name
    if data_start + expected_offset != len(payload) or digest.hexdigest() != expected_state_sha256:
        raise LocalAffordanceModelStoreError("model state digest or trailing bytes differ")
    return tuple(tensors)


def _shape_size(shape: list[int]) -> int:
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def _entry(record: LocalAffordanceModelRecord) -> LocalAffordanceStoreEntry:
    return LocalAffordanceStoreEntry(owner_id=record.key.owner_id, record_sha256=record.record_sha256,
                                     key_sha256=record.key.key_sha256, cost_sha256=record.cost.cost_sha256,
                                     artifact_sha256=record.artifact.artifact_sha256)


def _recorded_model_identity(record: LocalAffordanceModelRecord) -> str:
    """Recompute the preparation identity envelope using only frozen record fields."""
    key = record.key
    condition = key.condition_id
    architecture_id = ARCHITECTURE_ID_BY_CONDITION.get(condition)
    if architecture_id is None:
        raise LocalAffordanceModelStoreError("record condition has no frozen architecture")
    return _sha({
        "schema_version": "milestone6.phase3.local-affordance-model-preparation.v1",
        "plan_id": key.plan_id,
        "owner_id": key.owner_id,
        "view_id": key.view_id,
        "condition_id": condition,
        "training_tuple_id": key.training_tuple_id,
        "model_seed": key.model_seed,
        "training_spec": {
            "epochs": key.optimizer_epochs,
            "learning_rate": key.learning_rate,
            "weight_decay": WEIGHT_DECAY,
        },
        "architecture_id": architecture_id,
        "trainable_parameters": key.trainable_parameters,
        "training_examples_sha256": key.training_examples_sha256,
        "training_report": {
            "optimizer_steps": record.cost.optimizer_steps,
            "forward_passes": record.cost.forward_passes,
            "training_examples": record.cost.training_examples,
        },
        "model_state_sha256": key.model_state_sha256,
    })


def _validate_shape(reader: PinnedLocalAffordanceModelStore, manifest: LocalAffordanceStoreManifest) -> None:
    expected_records = {f"{item.owner_id}.json" for item in manifest.entries}
    expected_models = {f"{item.owner_id}.model" for item in manifest.entries}
    try:
        with os.scandir(reader.root_fd) as iterator:
            root_entries = {
                item.name: (item.is_symlink(), item.is_dir(follow_symlinks=False), item.is_file(follow_symlinks=False))
                for item in iterator
            }
        expected_root = {
            RECORDS_DIR: (False, True, False),
            MODELS_DIR: (False, True, False),
            STAGING_DIR: (False, True, False),
        }
        if manifest.entries or MANIFEST_NAME in root_entries:
            expected_root[MANIFEST_NAME] = (False, False, True)
        if root_entries != expected_root:
            raise LocalAffordanceModelStoreError("store root contains missing, extra, or unsafe entries")
        if set(secure_fs.strict_regular_entries(reader.records_fd)) != expected_records:
            raise LocalAffordanceModelStoreError("record inventory has missing or orphan entries")
        if set(secure_fs.strict_regular_entries(reader.models_fd)) != expected_models:
            raise LocalAffordanceModelStoreError("model inventory has missing or orphan entries")
        if secure_fs.strict_regular_entries(reader.staging_fd):
            raise LocalAffordanceModelStoreError("store contains incomplete staging files")
    except secure_fs.SecureFilesystemError as exc:
        raise LocalAffordanceModelStoreError("store contains a symlink or non-regular entry") from exc


def _validate_selected_owner_recovery_shape(
    reader: PinnedLocalAffordanceModelStore,
    manifest: LocalAffordanceStoreManifest,
    *,
    owner_id: str,
    record_bytes: bytes,
    model_bytes: bytes,
) -> None:
    """Allow only the selected owner's exact pre-manifest partial publication."""
    expected_records = {f"{item.owner_id}.json" for item in manifest.entries}
    expected_models = {f"{item.owner_id}.model" for item in manifest.entries}
    try:
        with os.scandir(reader.root_fd) as iterator:
            root_entries = {
                entry.name: (entry.is_symlink(), entry.is_dir(follow_symlinks=False), entry.is_file(follow_symlinks=False))
                for entry in iterator
            }
        if root_entries != {
            RECORDS_DIR: (False, True, False),
            MODELS_DIR: (False, True, False),
            STAGING_DIR: (False, True, False),
            **(
                {MANIFEST_NAME: (False, False, True)}
                if _contains_regular(reader.root_fd, MANIFEST_NAME)
                else {}
            ),
        }:
            raise LocalAffordanceModelStoreError("store root has foreign, extra, or unsafe entries")
        actual_records = set(secure_fs.strict_regular_entries(reader.records_fd))
        actual_models = set(secure_fs.strict_regular_entries(reader.models_fd))
        if secure_fs.strict_regular_entries(reader.staging_fd):
            raise LocalAffordanceModelStoreError("store has stale staging entries")
        allowed_records = expected_records | {f"{owner_id}.json"}
        allowed_models = expected_models | {f"{owner_id}.model"}
        if not expected_records <= actual_records or not actual_records <= allowed_records:
            raise LocalAffordanceModelStoreError("record inventory has foreign or missing entries")
        if not expected_models <= actual_models or not actual_models <= allowed_models:
            raise LocalAffordanceModelStoreError("model inventory has foreign or missing entries")
        if owner_id in {entry.owner_id for entry in manifest.entries}:
            raise LocalAffordanceModelStoreError("committed owner cannot be treated as a recovery orphan")
        if f"{owner_id}.json" in actual_records and _read_stable(reader.records_fd, f"{owner_id}.json") != record_bytes:
            raise LocalAffordanceModelStoreError("recovery record differs from selected owner")
        if f"{owner_id}.model" in actual_models and _read_stable(reader.models_fd, f"{owner_id}.model") != model_bytes:
            raise LocalAffordanceModelStoreError("recovery model differs from selected owner")
    except secure_fs.SecureFilesystemError as exc:
        raise LocalAffordanceModelStoreError("recovery store contains symlinks or non-regular entries") from exc


def _contains_regular(directory_fd: int, name: str) -> bool:
    try:
        _stable_file_identity(directory_fd, name)
        return True
    except LocalAffordanceModelStoreError as exc:
        if _caused_by_missing(exc):
            return False
        raise


def write_local_affordance_model(
    root: str | Path,
    plan: LocalAffordancePlan,
    prepared: LocalAffordanceModelPreparation,
    record: LocalAffordanceModelRecord,
    *,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
) -> LocalAffordanceStoreEntry:
    """Persist one already-trained model; repeated identical writes are resumable."""
    try:
        _validate_preparation(plan, prepared)
        validate_model_record(record, plan=plan)
    except LocalAffordanceModelStoreError:
        raise
    except (TypeError, ValueError, AttributeError) as exc:
        raise LocalAffordanceModelStoreError("model preparation or record is not plan-authorized") from exc
    if (
        record.key.owner_id != prepared.owner.owner_id
        or record.key.model_state_sha256 != prepared.model_state_sha256
        or record.key.model_identity_sha256 != prepared.model_identity_sha256
        or record.key.training_examples_sha256 != prepared.training_examples_sha256
        or record.cost.training_examples != prepared.report.training_examples
        or record.cost.forward_passes != prepared.report.forward_passes
        or record.artifact.artifact_name != f"{prepared.owner.owner_id}.model"
        or record.provenance.preparation_git_commit_sha != preparation_git_commit_sha
        or record.provenance.preparation_provenance_sha256 != preparation_provenance_sha256
    ):
        raise LocalAffordanceModelStoreError("record does not match prepared model or exact provenance")
    model_bytes = _encode_model(prepared)
    if record.artifact.artifact_bytes != len(model_bytes) or record.artifact.artifact_sha256 != _sha_bytes(model_bytes):
        raise LocalAffordanceModelStoreError("record artifact metadata does not match safe tensor bytes")
    record_bytes = _canonical(record)
    owner = prepared.owner.owner_id
    with open_local_affordance_model_store(root, exclusive_writer=True) as reader:
        reader.recheck()
        manifest = _manifest(reader, allow_missing_for_recovery=True)
        existing = next((item for item in manifest.entries if item.owner_id == owner), None)
        entry = _entry(record)
        if existing is not None:
            _validate_shape(reader, manifest)
            loaded_record, _ = load_local_affordance_model_at(reader, owner, plan=plan,
                preparation_git_commit_sha=preparation_git_commit_sha,
                preparation_provenance_sha256=preparation_provenance_sha256)
            if existing != entry or loaded_record != record:
                raise LocalAffordanceModelStoreError("different model already committed for this owner")
            return existing
        record_name = f"{owner}.json"
        model_name = f"{owner}.model"
        _validate_selected_owner_recovery_shape(
            reader, manifest, owner_id=owner, record_bytes=record_bytes, model_bytes=model_bytes
        )
        for directory_fd, name, content in ((reader.records_fd, record_name, record_bytes), (reader.models_fd, model_name, model_bytes)):
            try:
                _claim(directory_fd, reader.staging_fd, name, content)
            except LocalAffordanceModelStoreError:
                raise
        entries = tuple(sorted((*manifest.entries, entry), key=lambda item: item.owner_id))
        body = {"schema_version": SCHEMA_VERSION, "manifest_sha256": "0" * 64,
                "entries": [item.model_dump(mode="json") for item in entries]}
        body["manifest_sha256"] = _sha({key: value for key, value in body.items() if key != "manifest_sha256"})
        reader.recheck()
        _write_temp(reader.root_fd, reader.staging_fd, MANIFEST_NAME,
                    _canonical(LocalAffordanceStoreManifest.model_validate(body)))
        reader.recheck()
        return entry


def load_local_affordance_model_at(
    reader: PinnedLocalAffordanceModelStore,
    owner_id: str,
    *,
    plan: LocalAffordancePlan,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
) -> tuple[LocalAffordanceModelRecord, tuple[LocalAffordanceTensor, ...]]:
    """Load and validate one committed record/tensor payload through a held pin."""
    if type(reader) is not PinnedLocalAffordanceModelStore or reader._token is not _STORE_TOKEN:
        raise LocalAffordanceModelStoreError("canonical pinned store is required")
    reader.recheck()
    manifest_identity = _stable_file_identity(reader.root_fd, MANIFEST_NAME)
    manifest = _manifest(reader)
    _validate_shape(reader, manifest)
    entry = next((item for item in manifest.entries if item.owner_id == owner_id), None)
    if entry is None:
        raise LocalAffordanceModelStoreError("owner is not committed in manifest")
    record_name = f"{owner_id}.json"
    record_identity = _stable_file_identity(reader.records_fd, record_name)
    raw_record = _read_stable(reader.records_fd, record_name)
    try:
        value = json.loads(raw_record)
        if canonical_json_bytes(value) + b"\n" != raw_record:
            raise ValueError
        record = LocalAffordanceModelRecord.model_validate(value)
        validate_model_record(record, plan=plan)
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalAffordanceModelStoreError("stored model record is not canonical or plan-bound") from exc
    if (
        entry != _entry(record)
        or record.key.model_identity_sha256 != _recorded_model_identity(record)
        or record.provenance.preparation_git_commit_sha != preparation_git_commit_sha
        or record.provenance.preparation_provenance_sha256 != preparation_provenance_sha256
    ):
        raise LocalAffordanceModelStoreError("stored manifest or provenance lineage differs")
    model_identity = _stable_file_identity(reader.models_fd, record.artifact.artifact_name)
    payload = _read_stable(reader.models_fd, record.artifact.artifact_name)
    if len(payload) != record.artifact.artifact_bytes or _sha_bytes(payload) != record.artifact.artifact_sha256:
        raise LocalAffordanceModelStoreError("stored model artifact hash differs")
    tensors = _decode_model(payload, record.key.model_state_sha256)
    if (
        manifest_identity != _stable_file_identity(reader.root_fd, MANIFEST_NAME)
        or record_identity != _stable_file_identity(reader.records_fd, record_name)
        or model_identity != _stable_file_identity(reader.models_fd, record.artifact.artifact_name)
    ):
        raise LocalAffordanceModelStoreError("model-store file identity changed during reload")
    reader.recheck()
    return record, tensors


__all__ = [
    "LocalAffordanceModelStoreError",
    "LocalAffordanceStoreEntry",
    "LocalAffordanceStoreManifest",
    "LocalAffordanceTensor",
    "PinnedLocalAffordanceModelStore",
    "load_local_affordance_model_at",
    "open_local_affordance_model_store",
    "serialize_local_affordance_model",
    "write_local_affordance_model",
]
