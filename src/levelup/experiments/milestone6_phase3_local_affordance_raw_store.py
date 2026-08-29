"""Read-only descriptor-pinned schemas for Phase 3 raw probe evidence.

This module deliberately stops at the storage boundary.  It defines the
metadata needed to bind raw-probe artifacts and a reader which pins the raw
store directories, but it does not create files, enumerate artifacts, expose
learner lookups, run environments, search, or authorize any execution.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from levelup.experiments.milestone6_phase3_local_affordance_evidence import (
    FAMILY_ORDER,
    RawProbeArtifactKey,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes

SCHEMA_VERSION = "milestone6.phase3.local-affordance-raw-store.v1"
STORE_MANIFEST_VERSION = "milestone6.phase3.local-affordance-raw-store-manifest.v1"
TASK_KEY_INDEX_VERSION = "milestone6.phase3.local-affordance-task-key-index.v1"
TRAINING_FOLD_VERSION = "milestone6.phase3.local-affordance-training-fold.v1"
HELDOUT_BINDING_VERSION = "milestone6.phase3.local-affordance-heldout-binding.v1"
HEX64 = r"^[0-9a-f]{64}$"

# The names are part of the descriptor-pinning contract.  No other namespace
# is opened by this read-only slice.
ARTIFACTS_DIR = "artifacts"
KEYS_DIR = "keys"
TRAINING_FOLDS_DIR = "training-folds"
HELDOUT_BINDINGS_DIR = "heldout-bindings"
RAW_STORE_NAMESPACES = (
    ARTIFACTS_DIR,
    KEYS_DIR,
    TRAINING_FOLDS_DIR,
    HELDOUT_BINDINGS_DIR,
)

RAW_ARTIFACT_COUNT = 240
TRAINING_FOLD_COUNT = 30
HELDOUT_BINDING_COUNT = 240
TASKS_PER_TRAINING_FOLD = 40
TASKS_PER_FAMILY = 8

_STORE_TOKEN = object()


class RawProbeStoreError(ValueError):
    """Raised when raw-store metadata or descriptor identities fail closed."""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


class RawProbeStoreManifest(BaseModel):
    """Self-hashed authority binding the exact frozen raw-artifact universe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[STORE_MANIFEST_VERSION] = STORE_MANIFEST_VERSION
    scope: Literal["known-development-only"] = "known-development-only"
    execution_authorized: StrictBool = False
    manifest_id: str = Field(pattern=HEX64)
    local_affordance_protocol_sha256: str = Field(pattern=HEX64)
    development_protocol_sha256: str = Field(pattern=HEX64)
    development_tasks_sha256: str = Field(pattern=HEX64)
    phase3_evidence_lock_sha256: str = Field(pattern=HEX64)
    probe_policy_sha256: str = Field(pattern=HEX64)
    raw_artifact_count: StrictInt = RAW_ARTIFACT_COUNT
    training_fold_count: StrictInt = TRAINING_FOLD_COUNT
    heldout_binding_count: StrictInt = HELDOUT_BINDING_COUNT

    @property
    def expected_manifest_id(self) -> str:
        return _sha(self.model_dump(mode="json", exclude={"manifest_id"}))

    # Descriptive aliases keep callers from confusing the three exact counts.
    @property
    def artifact_count(self) -> int:
        return self.raw_artifact_count

    @property
    def fold_count(self) -> int:
        return self.training_fold_count

    @model_validator(mode="after")
    def canonical(self) -> "RawProbeStoreManifest":
        if self.execution_authorized is not False:
            raise ValueError("raw-store schema cannot authorize execution")
        if (
            self.raw_artifact_count != RAW_ARTIFACT_COUNT
            or self.training_fold_count != TRAINING_FOLD_COUNT
            or self.heldout_binding_count != HELDOUT_BINDING_COUNT
        ):
            raise ValueError("raw-store manifest counts differ from frozen authority")
        if self.manifest_id != self.expected_manifest_id:
            raise ValueError("raw-store manifest self-hash mismatch")
        return self

    @classmethod
    def from_authority_hashes(
        cls,
        *,
        local_affordance_protocol_sha256: str,
        development_protocol_sha256: str,
        development_tasks_sha256: str,
        phase3_evidence_lock_sha256: str,
        probe_policy_sha256: str,
    ) -> "RawProbeStoreManifest":
        unsigned = {
            "schema_version": STORE_MANIFEST_VERSION,
            "scope": "known-development-only",
            "execution_authorized": False,
            "local_affordance_protocol_sha256": local_affordance_protocol_sha256,
            "development_protocol_sha256": development_protocol_sha256,
            "development_tasks_sha256": development_tasks_sha256,
            "phase3_evidence_lock_sha256": phase3_evidence_lock_sha256,
            "probe_policy_sha256": probe_policy_sha256,
            "raw_artifact_count": RAW_ARTIFACT_COUNT,
            "training_fold_count": TRAINING_FOLD_COUNT,
            "heldout_binding_count": HELDOUT_BINDING_COUNT,
        }
        return cls(manifest_id=_sha(unsigned), **unsigned)


class RawProbeTaskKeyIndex(BaseModel):
    """Index row binding one exact artifact id to its typed artifact key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[TASK_KEY_INDEX_VERSION] = TASK_KEY_INDEX_VERSION
    key_id: str = Field(pattern=HEX64)
    artifact_id: str = Field(pattern=HEX64)
    key: RawProbeArtifactKey

    @model_validator(mode="after")
    def key_identity_is_exact(self) -> "RawProbeTaskKeyIndex":
        if self.key_id != self.key.key_id:
            raise ValueError("raw-store task key index key_id mismatch")
        return self


# Short aliases are intentional: these names read naturally in authority code
# while retaining one canonical model implementation.
TaskKeyIndex = RawProbeTaskKeyIndex
RawProbeTaskKeyReference = RawProbeTaskKeyIndex


class RawProbeTaskReference(BaseModel):
    """A fold/binding reference retaining the source manifest task index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["milestone6.phase3.local-affordance-task-reference.v1"] = (
        "milestone6.phase3.local-affordance-task-reference.v1"
    )
    artifact_id: str = Field(pattern=HEX64)
    key_id: str = Field(pattern=HEX64)
    key: RawProbeArtifactKey

    @property
    def family_id(self) -> str:
        return self.key.family_id

    @property
    def replicate(self) -> int:
        return self.key.replicate

    @property
    def task_id(self) -> str:
        return self.key.task_id

    @property
    def task_index(self) -> int:
        return self.key.task_index

    @model_validator(mode="after")
    def key_identity_is_exact(self) -> "RawProbeTaskReference":
        if self.key_id != self.key.key_id:
            raise ValueError("raw-store task reference key_id mismatch")
        return self


TaskReference = RawProbeTaskReference


class TrainingFoldManifest(BaseModel):
    """One development-only leave-one-family-out training manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[TRAINING_FOLD_VERSION] = TRAINING_FOLD_VERSION
    fold_id: str = Field(min_length=1)
    heldout_family: str = Field(min_length=1)
    replicate: StrictInt = Field(ge=0, le=4)
    task_references: tuple[RawProbeTaskReference, ...]

    @property
    def heldout_family_id(self) -> str:
        return self.heldout_family

    @property
    def references(self) -> tuple[RawProbeTaskReference, ...]:
        return self.task_references

    @model_validator(mode="after")
    def exact_lofo_factorization(self) -> "TrainingFoldManifest":
        if self.fold_id != self.heldout_family:
            raise ValueError("training fold_id must equal heldout family")
        if self.heldout_family not in FAMILY_ORDER:
            raise ValueError("training fold family is not a development family")
        if len(self.task_references) != TASKS_PER_TRAINING_FOLD:
            raise ValueError("training folds require exactly 40 task references")
        uniqueness_views = (
            {ref.key_id for ref in self.task_references},
            {ref.artifact_id for ref in self.task_references},
            {(ref.family_id, ref.task_id) for ref in self.task_references},
            {(ref.family_id, ref.task_index) for ref in self.task_references},
        )
        if any(len(values) != TASKS_PER_TRAINING_FOLD for values in uniqueness_views):
            raise ValueError("training fold task references must be uniquely bound")
        if any(ref.family_id == self.heldout_family for ref in self.task_references):
            raise ValueError("training fold references cannot include heldout family")
        if any(ref.replicate != self.replicate for ref in self.task_references):
            raise ValueError("training fold references must match the fold replicate")
        counts = {family: 0 for family in FAMILY_ORDER if family != self.heldout_family}
        for ref in self.task_references:
            if ref.family_id not in counts:
                raise ValueError("training fold reference family is not in the LOFO complement")
            counts[ref.family_id] += 1
        if any(count != TASKS_PER_FAMILY for count in counts.values()):
            raise ValueError("training folds require eight references from each other family")
        expected_order = tuple(
            sorted(
                self.task_references,
                key=lambda ref: (
                    FAMILY_ORDER.index(ref.family_id),
                    ref.task_index,
                    ref.task_id,
                    ref.key_id,
                ),
            )
        )
        if self.task_references != expected_order:
            raise ValueError("training fold task references are not in canonical order")
        authority_hashes = {
            (
                ref.key.local_affordance_protocol_sha256,
                ref.key.development_protocol_sha256,
                ref.key.development_tasks_sha256,
                ref.key.phase3_evidence_lock_sha256,
                ref.key.probe_policy_sha256,
            )
            for ref in self.task_references
        }
        if len(authority_hashes) != 1:
            raise ValueError("training fold references mix raw-evidence authorities")
        return self


TrainingEvidenceFoldManifest = TrainingFoldManifest


class HeldoutProbeBinding(BaseModel):
    """One development heldout task binding; it carries exactly one reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[HELDOUT_BINDING_VERSION] = HELDOUT_BINDING_VERSION
    fold_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    replicate: StrictInt = Field(ge=0, le=4)
    task_reference: RawProbeTaskReference

    @property
    def reference(self) -> RawProbeTaskReference:
        return self.task_reference

    @model_validator(mode="after")
    def binding_identity_is_exact(self) -> "HeldoutProbeBinding":
        if self.fold_id != self.family_id:
            raise ValueError("heldout binding fold and family must be equal")
        if self.family_id not in FAMILY_ORDER:
            raise ValueError("heldout binding family is not a development family")
        ref = self.task_reference
        if ref.family_id != self.family_id or ref.replicate != self.replicate:
            raise ValueError("heldout binding reference does not match fold/family/replicate")
        return self


HeldoutTaskBinding = HeldoutProbeBinding


class StableDirectoryIdentity(BaseModel):
    """Device/inode identity for one held directory descriptor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device: StrictInt = Field(ge=0)
    inode: StrictInt = Field(ge=0)


class StableFileIdentity(BaseModel):
    """Identity and size snapshot for a regular file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device: StrictInt = Field(ge=0)
    inode: StrictInt = Field(ge=0)
    mode: StrictInt = Field(ge=0)
    byte_length: StrictInt = Field(ge=0)
    mtime_ns: StrictInt = Field(ge=0)
    ctime_ns: StrictInt = Field(ge=0)


class StableFileSnapshot(BaseModel):
    """Stable identity plus canonical bytes captured from a descriptor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: StableFileIdentity
    byte_length: StrictInt = Field(ge=0)
    canonical_bytes: bytes
    sha256: str = Field(pattern=HEX64)

    @model_validator(mode="after")
    def bytes_match_identity(self) -> "StableFileSnapshot":
        if self.byte_length != len(self.canonical_bytes):
            raise ValueError("stable snapshot byte length mismatch")
        if self.byte_length != self.identity.byte_length:
            raise ValueError("stable snapshot identity length mismatch")
        if hashlib.sha256(self.canonical_bytes).hexdigest() != self.sha256:
            raise ValueError("stable snapshot byte digest mismatch")
        return self


def _file_identity(value: os.stat_result) -> StableFileIdentity:
    if not stat.S_ISREG(value.st_mode):
        raise RawProbeStoreError("raw-store entry is not a regular file")
    return StableFileIdentity(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        byte_length=int(value.st_size),
        mtime_ns=int(value.st_mtime_ns),
        ctime_ns=int(value.st_ctime_ns),
    )


def _directory_identity_model(directory_fd: int) -> StableDirectoryIdentity:
    device, inode = secure_fs.directory_identity(directory_fd)
    return StableDirectoryIdentity(device=device, inode=inode)


def _stable_file_snapshot_at(directory_fd: int, name: str) -> StableFileSnapshot:
    """Read canonical JSON bytes while proving descriptor/path identity stability."""

    try:
        with secure_fs.open_regular_file_at(directory_fd, name) as file_fd:
            before = _file_identity(os.fstat(file_fd))
            path_before = _file_identity(
                os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            )
            if before != path_before:
                raise RawProbeStoreError("raw-store file identity changed before read")
            chunks: list[bytes] = []
            while chunk := os.read(file_fd, 1024 * 1024):
                chunks.append(chunk)
            content = b"".join(chunks)
            after = _file_identity(os.fstat(file_fd))
            path_after = _file_identity(os.stat(name, dir_fd=directory_fd, follow_symlinks=False))
            if before != after or after != path_after or len(content) != after.byte_length:
                raise RawProbeStoreError("raw-store file changed while being read")
            try:
                value = json.loads(content)
            except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise RawProbeStoreError("raw-store file is not valid JSON") from exc
            canonical = canonical_json_bytes(value) + b"\n"
            if content != canonical:
                raise RawProbeStoreError("raw-store JSON bytes are not canonical")
            return StableFileSnapshot(
                identity=after,
                byte_length=len(content),
                canonical_bytes=content,
                sha256=hashlib.sha256(content).hexdigest(),
            )
    except RawProbeStoreError:
        raise
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        if isinstance(exc.__cause__, RawProbeStoreError):
            raise exc.__cause__
        raise RawProbeStoreError(f"cannot stably read raw-store file: {name}") from exc


def _read_stable_canonical_json_at(directory_fd: int, name: str) -> bytes:
    """Return canonical bytes from one descriptor-relative regular JSON file."""

    return _stable_file_snapshot_at(directory_fd, name).canonical_bytes


_read_stable_json_at = _read_stable_canonical_json_at


@dataclass(frozen=True, slots=True, init=False)
class PinnedRawProbeStoreReader:
    """Read-only pin of the root and four exact raw-store namespaces.

    No artifact lookup, enumeration, mutation, learner capability, evaluator,
    search, or execution API is intentionally present on this type.
    """

    root_fd: int
    artifacts_fd: int
    keys_fd: int
    training_folds_fd: int
    heldout_bindings_fd: int
    root_path: Path
    identities: tuple[StableDirectoryIdentity, ...]
    _token: object
    _closed: bool

    def __init__(
        self,
        root_fd: int,
        artifacts_fd: int,
        keys_fd: int,
        training_folds_fd: int,
        heldout_bindings_fd: int,
        root_path: Path,
        identities: tuple[StableDirectoryIdentity, ...],
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _STORE_TOKEN:
            raise RawProbeStoreError("raw-store readers require canonical descriptor pinning")
        if len(identities) != len(RAW_STORE_NAMESPACES) + 1:
            raise RawProbeStoreError("raw-store reader identity snapshot is incomplete")
        object.__setattr__(self, "root_fd", root_fd)
        object.__setattr__(self, "artifacts_fd", artifacts_fd)
        object.__setattr__(self, "keys_fd", keys_fd)
        object.__setattr__(self, "training_folds_fd", training_folds_fd)
        object.__setattr__(self, "heldout_bindings_fd", heldout_bindings_fd)
        object.__setattr__(self, "root_path", Path(root_path))
        object.__setattr__(self, "identities", tuple(identities))
        object.__setattr__(self, "_token", _STORE_TOKEN)
        object.__setattr__(self, "_closed", False)

    def _check_open(self) -> None:
        if self._closed or self._token is not _STORE_TOKEN:
            raise RawProbeStoreError("raw-store reader is closed or unauthorized")

    def recheck(self) -> None:
        """Fail closed if held descriptors or lexical namespaces were replaced."""

        self._check_open()
        fds = (
            self.root_fd,
            self.artifacts_fd,
            self.keys_fd,
            self.training_folds_fd,
            self.heldout_bindings_fd,
        )
        try:
            held = tuple(_directory_identity_model(fd) for fd in fds)
            if held != self.identities:
                raise RawProbeStoreError("held raw-store descriptors changed")
            lexical_root = secure_fs.open_directory_chain(self.root_path)
            try:
                observed = [_directory_identity_model(lexical_root)]
                for name in RAW_STORE_NAMESPACES:
                    child = secure_fs.open_child_directory(lexical_root, name)
                    try:
                        observed.append(_directory_identity_model(child))
                    finally:
                        os.close(child)
            finally:
                os.close(lexical_root)
        except RawProbeStoreError:
            raise
        except (OSError, secure_fs.SecureFilesystemError) as exc:
            raise RawProbeStoreError("raw-store root or namespace was replaced") from exc
        if tuple(observed) != self.identities:
            raise RawProbeStoreError("raw-store root or namespace was replaced")

    def close(self) -> None:
        """Close all held descriptors exactly once."""

        if self._closed:
            return
        object.__setattr__(self, "_closed", True)
        for fd in (
            self.heldout_bindings_fd,
            self.training_folds_fd,
            self.keys_fd,
            self.artifacts_fd,
            self.root_fd,
        ):
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> "PinnedRawProbeStoreReader":
        self._check_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@contextmanager
def open_existing_raw_probe_store(root: str | Path) -> Iterator[PinnedRawProbeStoreReader]:
    """Pin an existing raw store without creating or enumerating any entries."""

    root_path = Path(os.path.abspath(root))
    fds: list[int] = []
    try:
        root_fd = secure_fs.open_directory_chain(root_path)
        fds.append(root_fd)
        namespace_fds: list[int] = []
        for name in RAW_STORE_NAMESPACES:
            child_fd = secure_fs.open_child_directory(root_fd, name)
            namespace_fds.append(child_fd)
            # Append immediately so a later namespace-open failure cannot leak earlier fds.
            fds.append(child_fd)
        identities = tuple(_directory_identity_model(fd) for fd in fds)
        reader = PinnedRawProbeStoreReader(
            root_fd,
            namespace_fds[0],
            namespace_fds[1],
            namespace_fds[2],
            namespace_fds[3],
            root_path,
            identities,
            _token=_STORE_TOKEN,
        )
        reader.recheck()
        yield reader
    except RawProbeStoreError:
        raise
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise RawProbeStoreError("cannot pin raw-store root or namespaces") from exc
    finally:
        if fds:
            # Reader owns descriptors after construction; on normal context
            # exit it closes them.  If setup failed, clean up here.
            if "reader" not in locals():
                for fd in reversed(fds):
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            else:
                reader.close()


open_raw_probe_store_reader = open_existing_raw_probe_store


__all__ = [
    "ARTIFACTS_DIR",
    "FAMILY_ORDER",
    "HELDOUT_BINDINGS_DIR",
    "HELDOUT_BINDING_COUNT",
    "HeldoutProbeBinding",
    "HeldoutTaskBinding",
    "KEYS_DIR",
    "PinnedRawProbeStoreReader",
    "RAW_ARTIFACT_COUNT",
    "RAW_STORE_NAMESPACES",
    "RawProbeStoreError",
    "RawProbeStoreManifest",
    "RawProbeTaskKeyIndex",
    "RawProbeTaskKeyReference",
    "RawProbeTaskReference",
    "StableDirectoryIdentity",
    "StableFileIdentity",
    "StableFileSnapshot",
    "TASKS_PER_TRAINING_FOLD",
    "TASKS_PER_FAMILY",
    "TaskKeyIndex",
    "TaskReference",
    "TRAINING_FOLD_COUNT",
    "TRAINING_FOLDS_DIR",
    "TrainingEvidenceFoldManifest",
    "TrainingFoldManifest",
    "_read_stable_canonical_json_at",
    "_stable_file_snapshot_at",
    "open_existing_raw_probe_store",
    "open_raw_probe_store_reader",
]
