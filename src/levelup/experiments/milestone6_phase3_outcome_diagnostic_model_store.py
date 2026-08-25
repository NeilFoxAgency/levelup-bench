"""Descriptor-pinned, resumable storage for outcome-diagnostic model states.

This module stores only preparation artifacts: a canonical model record and the
float32 state bytes which that record names.  It has no training, environment,
result, evaluator, oracle, or final-family authority.  All reads and writes are
relative to descriptors held by :func:`open_outcome_model_store`.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from levelup.experiments.milestone6_phase3_model_artifacts import (
    ARTIFACTS_DIR as PHASE3_ARTIFACTS_DIR,
)
from levelup.experiments.milestone6_phase3_model_artifacts import (
    COSTS_DIR as PHASE3_COSTS_DIR,
)
from levelup.experiments.milestone6_phase3_model_artifacts import (
    KEYS_DIR as PHASE3_KEYS_DIR,
)
from levelup.experiments.milestone6_phase3_model_artifacts import (
    STAGING_DIR as PHASE3_STAGING_DIR,
)
from levelup.experiments.milestone6_phase3_model_artifacts import (
    open_phase3_model_output,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
    AuthorizedOutcomeModelArtifact,
    OutcomeDiagnosticModelArtifactError,
    OutcomeDiagnosticModelArtifactRecord,
    OutcomeStateTensorPayload,
    PinnedOutcomeModelState,
    PinnedOutcomeTrainingEvidence,
    canonical_outcome_model_artifact_record_bytes,
    inspect_outcome_model_state,
    load_outcome_model_artifact_record_bytes,
    validate_outcome_model_artifact_against_plan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    EXPECTED_MODEL_OWNERS,
    ValidatedOutcomePlan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    OutcomeDiagnosticProtocolSnapshot,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes

SCHEMA_VERSION = "milestone6.phase3.outcome-diagnostic-model-store.v1"
MANIFEST_NAME = "outcome-model-store.json"
RECORDS_DIR = "records"
STATES_DIR = "states"
STAGING_DIR = "staging"
STATE_MANIFEST_NAME = "state.json"
TENSORS_DIR = "tensors"
ROOT_METADATA_FILES = (
    "preparation-progress.json",
    "outcome-model-preparation-provenance.json",
)
HEX64 = r"^[0-9a-f]{64}$"
_STORE_TOKEN = object()


class OutcomeModelStoreError(ValueError):
    """Raised when model-store metadata or descriptors fail closed."""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _canonical(value: BaseModel) -> bytes:
    return canonical_json_bytes(value.model_dump(mode="json")) + b"\n"


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    if not stat.S_ISREG(value.st_mode):
        raise OutcomeModelStoreError("model-store entry is not a regular file")
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_stable(directory_fd: int, name: str) -> bytes:
    try:
        with secure_fs.open_regular_file_at(directory_fd, name) as file_fd:
            before = _identity(os.fstat(file_fd))
            path_before = _identity(os.stat(name, dir_fd=directory_fd, follow_symlinks=False))
            if before != path_before:
                raise OutcomeModelStoreError("model-store file identity changed")
            chunks: list[bytes] = []
            while chunk := os.read(file_fd, 1024 * 1024):
                chunks.append(chunk)
            content = b"".join(chunks)
            after = _identity(os.fstat(file_fd))
            path_after = _identity(os.stat(name, dir_fd=directory_fd, follow_symlinks=False))
            if before != after or after != path_after or len(content) != after[3]:
                raise OutcomeModelStoreError("model-store file changed while being read")
            return content
    except OutcomeModelStoreError:
        raise
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise OutcomeModelStoreError(f"cannot read model-store file: {name}") from exc


def _is_missing(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, FileNotFoundError):
            return True
        current = current.__cause__
    return False


def _write_new(parent_fd: int, name: str, content: bytes, staging_fd: int) -> None:
    temp = f".{name}.{uuid.uuid4().hex}.tmp"
    fd: int | None = None
    try:
        fd = os.open(
            temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=staging_fd
        )
        offset = 0
        while offset < len(content):
            offset += os.write(fd, content[offset:])
        os.fsync(fd)
    except OSError as exc:
        raise OutcomeModelStoreError(f"cannot stage model-store file: {name}") from exc
    finally:
        if fd is not None:
            os.close(fd)
    try:
        os.replace(temp, name, src_dir_fd=staging_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as exc:
        raise OutcomeModelStoreError(f"cannot publish model-store file: {name}") from exc
    finally:
        try:
            os.unlink(temp, dir_fd=staging_fd)
        except FileNotFoundError:
            pass


def _claim_or_match(parent_fd: int, name: str, content: bytes, staging_fd: int) -> None:
    """Publish one file without replacing a competing record."""
    temp = f".{name}.{uuid.uuid4().hex}.claim"
    fd: int | None = None
    try:
        fd = os.open(
            temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=staging_fd
        )
        offset = 0
        while offset < len(content):
            offset += os.write(fd, content[offset:])
        os.fsync(fd)
        try:
            os.link(
                temp,
                name,
                src_dir_fd=staging_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            observed = _read_stable(parent_fd, name)
            if observed != content:
                raise OutcomeModelStoreError("different model record already owns this owner")
        else:
            os.fsync(parent_fd)
    except OutcomeModelStoreError:
        raise
    except OSError as exc:
        raise OutcomeModelStoreError(f"cannot claim model-store file: {name}") from exc
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temp, dir_fd=staging_fd)
        except FileNotFoundError:
            pass


def _remove_tree_at(parent_fd: int, name: str) -> None:
    """Remove a private staging tree relative to its held parent fd."""
    try:
        child_fd = secure_fs.open_child_directory(parent_fd, name)
    except (OSError, secure_fs.SecureFilesystemError):
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


class OutcomeModelTensorIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    shape: tuple[StrictInt, ...]
    byte_length: StrictInt = Field(gt=0)
    sha256: str = Field(pattern=HEX64)


class OutcomeModelStateIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    index_id: str = Field(pattern=HEX64)
    owner_id: str = Field(pattern=HEX64)
    record_id: str = Field(pattern=HEX64)
    model_state_sha256: str = Field(pattern=HEX64)
    tensors: tuple[OutcomeModelTensorIndex, ...]

    @property
    def expected_index_id(self) -> str:
        return _sha(self.model_dump(mode="json", exclude={"index_id"}))

    @model_validator(mode="after")
    def canonical(self) -> "OutcomeModelStateIndex":
        if self.index_id != self.expected_index_id:
            raise ValueError("model state index self-hash mismatch")
        if tuple(item.filename for item in self.tensors) != tuple(
            f"{i:04d}.bin" for i in range(len(self.tensors))
        ):
            raise ValueError("model state tensor filenames are not canonical")
        return self


class OutcomeModelStoreEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    owner_id: str = Field(pattern=HEX64)
    record_id: str = Field(pattern=HEX64)
    key_id: str = Field(pattern=HEX64)
    record_sha256: str = Field(pattern=HEX64)
    state_index_id: str = Field(pattern=HEX64)
    model_state_sha256: str = Field(pattern=HEX64)


class OutcomeModelStoreManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    manifest_id: str = Field(pattern=HEX64)
    entries: tuple[OutcomeModelStoreEntry, ...] = ()

    @property
    def expected_manifest_id(self) -> str:
        return _sha(self.model_dump(mode="json", exclude={"manifest_id"}))

    @model_validator(mode="after")
    def canonical(self) -> "OutcomeModelStoreManifest":
        if self.manifest_id != self.expected_manifest_id:
            raise ValueError("model-store manifest self-hash mismatch")
        owners = tuple(entry.owner_id for entry in self.entries)
        if owners != tuple(sorted(owners)) or len(set(owners)) != len(owners):
            raise ValueError("model-store manifest owner inventory is not canonical")
        return self


@dataclass(frozen=True, slots=True)
class OutcomeModelStateIdentitySnapshot:
    """Descriptor identities for one owner's complete state tree."""

    owner_id: str
    state_directory_identity: tuple[int, int]
    manifest_identity: tuple[int, int, int, int, int, int]
    tensors_directory_identity: tuple[int, int]
    tensor_file_identities: tuple[tuple[str, tuple[int, int, int, int, int, int]], ...]


@dataclass(frozen=True, slots=True)
class OutcomeModelStoreIdentitySnapshot:
    """Immutable identity snapshot of a complete, descriptor-pinned model store.

    Every identity is collected through the descriptors held by a
    :class:`PinnedOutcomeModelStoreReader`.  Comparing two snapshots therefore
    detects same-byte replacement of any manifest, record, state directory,
    state manifest, tensor directory, or tensor file.
    """

    root_identity: tuple[int, int]
    records_identity: tuple[int, int]
    states_identity: tuple[int, int]
    staging_identity: tuple[int, int]
    manifest_identity: tuple[int, int, int, int, int, int]
    root_metadata_file_identities: tuple[
        tuple[str, tuple[int, int, int, int, int, int]], ...
    ]
    record_file_identities: tuple[tuple[str, tuple[int, int, int, int, int, int]], ...]
    state_identities: tuple[OutcomeModelStateIdentitySnapshot, ...]


@dataclass(frozen=True, slots=True, init=False)
class PinnedOutcomeModelStoreReader:
    root_fd: int
    records_fd: int
    states_fd: int
    staging_fd: int
    root_path: Path
    identities: tuple[tuple[int, int], ...]
    _token: object

    def __init__(
        self,
        root_fd: int,
        records_fd: int,
        states_fd: int,
        staging_fd: int,
        root_path: Path,
        identities: tuple[tuple[int, int], ...],
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _STORE_TOKEN:
            raise OutcomeModelStoreError("model-store readers require canonical descriptor pinning")
        object.__setattr__(self, "root_fd", root_fd)
        object.__setattr__(self, "records_fd", records_fd)
        object.__setattr__(self, "states_fd", states_fd)
        object.__setattr__(self, "staging_fd", staging_fd)
        object.__setattr__(self, "root_path", root_path)
        object.__setattr__(self, "identities", identities)
        object.__setattr__(self, "_token", _STORE_TOKEN)

    def recheck(self) -> None:
        if getattr(self, "_token", None) is not _STORE_TOKEN:
            raise OutcomeModelStoreError("model-store reader authority is invalid")
        try:
            held = tuple(
                secure_fs.directory_identity(fd)
                for fd in (
                    self.root_fd,
                    self.records_fd,
                    self.states_fd,
                    self.staging_fd,
                )
            )
        except secure_fs.SecureFilesystemError as exc:
            raise OutcomeModelStoreError("held model-store descriptors changed") from exc
        if held != self.identities:
            raise OutcomeModelStoreError("held model-store descriptors changed")
        try:
            root = secure_fs.open_directory_chain(self.root_path)
            observed = [secure_fs.directory_identity(root)]
            for name in (RECORDS_DIR, STATES_DIR, STAGING_DIR):
                child = secure_fs.open_child_directory(root, name)
                try:
                    observed.append(secure_fs.directory_identity(child))
                finally:
                    os.close(child)
        except (OSError, secure_fs.SecureFilesystemError) as exc:
            raise OutcomeModelStoreError("model-store root or namespace was replaced") from exc
        finally:
            try:
                os.close(root)
            except UnboundLocalError:
                pass
        if tuple(observed) != self.identities:
            raise OutcomeModelStoreError("model-store root or namespace was replaced")


@dataclass(frozen=True, slots=True, init=False)
class PinnedOutcomeModelStore:
    reader: PinnedOutcomeModelStoreReader
    _token: object

    def __init__(
        self,
        reader: PinnedOutcomeModelStoreReader,
        *,
        _token: object | None = None,
    ) -> None:
        if (
            _token is not _STORE_TOKEN
            or type(reader) is not PinnedOutcomeModelStoreReader
            or getattr(reader, "_token", None) is not _STORE_TOKEN
        ):
            raise OutcomeModelStoreError("model stores require canonical descriptor pinning")
        object.__setattr__(self, "reader", reader)
        object.__setattr__(self, "_token", _STORE_TOKEN)

    def recheck(self) -> None:
        if getattr(self, "_token", None) is not _STORE_TOKEN:
            raise OutcomeModelStoreError("model-store authority is invalid")
        self.reader.recheck()


def _require_reader(
    value: PinnedOutcomeModelStoreReader | PinnedOutcomeModelStore,
) -> PinnedOutcomeModelStoreReader:
    if type(value) is PinnedOutcomeModelStore:
        if getattr(value, "_token", None) is not _STORE_TOKEN:
            raise OutcomeModelStoreError("model-store authority is invalid")
        reader = value.reader
    elif type(value) is PinnedOutcomeModelStoreReader:
        reader = value
    else:
        raise OutcomeModelStoreError("canonical pinned model-store reader is required")
    if getattr(reader, "_token", None) is not _STORE_TOKEN:
        raise OutcomeModelStoreError("model-store reader authority is invalid")
    reader.recheck()
    return reader


def _mkdir(parent_fd: int, name: str) -> None:
    try:
        os.mkdir(name, 0o755, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except FileExistsError:
        pass


@contextmanager
def open_outcome_model_store(root: str | Path) -> Iterator[PinnedOutcomeModelStore]:
    """Create and pin the model store namespaces for the complete operation."""
    root_path = Path(os.path.abspath(root))
    if root_path.exists() and root_path.is_symlink():
        raise OutcomeModelStoreError("refusing symlink model-store root")
    with open_phase3_model_output(root_path) as output:
        output.root_fd  # keep the phase-3 primitive pin alive for this scope
        for name in (RECORDS_DIR, STATES_DIR, STAGING_DIR):
            _mkdir(output.root_fd, name)
        descriptors: list[int] = []
        try:
            records_fd = secure_fs.open_child_directory(output.root_fd, RECORDS_DIR)
            descriptors.append(records_fd)
            states_fd = secure_fs.open_child_directory(output.root_fd, STATES_DIR)
            descriptors.append(states_fd)
            staging_fd = secure_fs.open_child_directory(output.root_fd, STAGING_DIR)
            descriptors.append(staging_fd)
            identities = (secure_fs.directory_identity(output.root_fd),) + tuple(
                secure_fs.directory_identity(fd) for fd in (records_fd, states_fd, staging_fd)
            )
            reader = PinnedOutcomeModelStoreReader(
                output.root_fd,
                records_fd,
                states_fd,
                staging_fd,
                root_path,
                identities,
                _token=_STORE_TOKEN,
            )
            reader.recheck()
            yield PinnedOutcomeModelStore(reader, _token=_STORE_TOKEN)
        finally:
            for fd in reversed(descriptors):
                os.close(fd)


@contextmanager
def open_existing_outcome_model_store(root: str | Path) -> Iterator[PinnedOutcomeModelStore]:
    """Open an already-created model store without creating or mutating it.

    This is the authority-generation boundary.  Unlike
    :func:`open_outcome_model_store`, it never calls a mkdir-capable output
    helper: the root and all three outcome namespaces must already exist as
    real directories.  Every descriptor is opened with ``O_NOFOLLOW`` and is
    held for the complete operation, so a later replacement of the path cannot
    redirect reads to a different tree.
    """

    root_path = Path(os.path.abspath(root))
    try:
        if root_path.is_symlink():
            raise OutcomeModelStoreError("refusing symlink model-store root")
        if not root_path.exists():
            raise OutcomeModelStoreError("model-store root does not exist")
    except OSError as exc:
        raise OutcomeModelStoreError("cannot inspect model-store root") from exc

    with ExitStack() as stack:
        try:
            root_fd = secure_fs.open_directory_chain(root_path)
            stack.callback(os.close, root_fd)
            child_fds: dict[str, int] = {}
            for name in (RECORDS_DIR, STATES_DIR, STAGING_DIR):
                child_fd = secure_fs.open_child_directory(root_fd, name)
                child_fds[name] = child_fd
                stack.callback(os.close, child_fd)
            identities = (secure_fs.directory_identity(root_fd),) + tuple(
                secure_fs.directory_identity(child_fds[name])
                for name in (RECORDS_DIR, STATES_DIR, STAGING_DIR)
            )
            reader = PinnedOutcomeModelStoreReader(
                root_fd,
                child_fds[RECORDS_DIR],
                child_fds[STATES_DIR],
                child_fds[STAGING_DIR],
                root_path,
                identities,
                _token=_STORE_TOKEN,
            )
            reader.recheck()
            yield PinnedOutcomeModelStore(reader, _token=_STORE_TOKEN)
        except (OSError, TypeError, ValueError, secure_fs.SecureFilesystemError) as exc:
            raise OutcomeModelStoreError(
                "cannot securely open existing model-store namespaces"
            ) from exc


def _record_name(owner_id: str) -> str:
    if len(owner_id) != 64 or any(c not in "0123456789abcdef" for c in owner_id):
        raise OutcomeModelStoreError("invalid model owner ID")
    return f"{owner_id}.json"


def _load_manifest(reader: PinnedOutcomeModelStoreReader) -> OutcomeModelStoreManifest:
    try:
        raw = _read_stable(reader.root_fd, MANIFEST_NAME)
    except OutcomeModelStoreError as exc:
        if _is_missing(exc):
            return OutcomeModelStoreManifest(
                manifest_id=_sha({"schema_version": SCHEMA_VERSION, "entries": []}), entries=()
            )
        raise
    try:
        value = json.loads(raw)
        if canonical_json_bytes(value) + b"\n" != raw:
            raise ValueError
        return OutcomeModelStoreManifest.model_validate(value)
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise OutcomeModelStoreError("model-store manifest is not canonical") from exc


def _reader_for_identity_snapshot(
    value: PinnedOutcomeModelStoreReader | PinnedOutcomeModelStore,
) -> PinnedOutcomeModelStoreReader:
    """Validate only held descriptors, deliberately never reopening ``root_path``."""

    if type(value) is PinnedOutcomeModelStore:
        if getattr(value, "_token", None) is not _STORE_TOKEN:
            raise OutcomeModelStoreError("model-store authority is invalid")
        reader = value.reader
    elif type(value) is PinnedOutcomeModelStoreReader:
        reader = value
    else:
        raise OutcomeModelStoreError("canonical pinned model-store reader is required")
    if getattr(reader, "_token", None) is not _STORE_TOKEN:
        raise OutcomeModelStoreError("model-store reader authority is invalid")
    try:
        held = tuple(
            secure_fs.directory_identity(fd)
            for fd in (reader.root_fd, reader.records_fd, reader.states_fd, reader.staging_fd)
        )
    except secure_fs.SecureFilesystemError as exc:
        raise OutcomeModelStoreError("held model-store descriptors changed") from exc
    if held != reader.identities:
        raise OutcomeModelStoreError("held model-store descriptors changed")
    return reader


def _stable_identity_at(directory_fd: int, name: str) -> tuple[int, int, int, int, int, int]:
    """Return one stable regular-file identity relative to a held descriptor."""

    try:
        with secure_fs.open_regular_file_at(directory_fd, name) as file_fd:
            before = _identity(os.fstat(file_fd))
            path_before = _identity(os.stat(name, dir_fd=directory_fd, follow_symlinks=False))
            after = _identity(os.fstat(file_fd))
            path_after = _identity(os.stat(name, dir_fd=directory_fd, follow_symlinks=False))
            if before != path_before or before != after or after != path_after:
                raise OutcomeModelStoreError("model-store file identity changed")
            return before
    except OutcomeModelStoreError:
        raise
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise OutcomeModelStoreError(f"model-store file is missing or unsafe: {name}") from exc


def _directory_shape_at(
    directory_fd: int,
    expected: Mapping[str, tuple[bool, bool]],
    *,
    message: str,
) -> None:
    """Require an exact descriptor-relative directory shape.

    The booleans describe ``(is_regular_file, is_directory)``.  Symlinks are
    always rejected explicitly, even when their target has the expected shape.
    """

    try:
        observed: dict[str, tuple[bool, bool, bool]] = {}
        with os.scandir(directory_fd) as iterator:
            for entry in iterator:
                observed[entry.name] = (
                    entry.is_symlink(),
                    entry.is_file(follow_symlinks=False),
                    entry.is_dir(follow_symlinks=False),
                )
    except OSError as exc:
        raise OutcomeModelStoreError(message) from exc
    expected_shape = {
        name: (False, is_file, is_directory)
        for name, (is_file, is_directory) in expected.items()
    }
    if observed != expected_shape:
        raise OutcomeModelStoreError(message)


def snapshot_outcome_model_store_identities_at(
    reader_or_store: PinnedOutcomeModelStoreReader | PinnedOutcomeModelStore,
    expected_owner_ids: tuple[str, ...] | list[str] | set[str],
    *,
    required_root_files: tuple[str, ...] = ROOT_METADATA_FILES,
) -> OutcomeModelStoreIdentitySnapshot:
    """Capture a complete descriptor-relative identity snapshot.

    This is intentionally an identity/shape operation, not a semantic model
    authorization operation.  It requires the exact 240-owner universe,
    canonical no-symlink inventory, and an empty staging namespace.  It never
    reopens ``reader.root_path``; every descendant is resolved from the held
    root/records/states/staging descriptors.
    """

    reader = _reader_for_identity_snapshot(reader_or_store)
    try:
        expected = tuple(sorted(expected_owner_ids))
    except (TypeError, ValueError) as exc:
        raise OutcomeModelStoreError("model-store owner universe is invalid") from exc
    if len(expected) != EXPECTED_MODEL_OWNERS or len(set(expected)) != EXPECTED_MODEL_OWNERS:
        raise OutcomeModelStoreError("identity snapshot requires exact 240-owner authority")
    try:
        for owner_id in expected:
            _record_name(owner_id)
    except OutcomeModelStoreError as exc:
        raise OutcomeModelStoreError("identity snapshot owner universe is not canonical") from exc

    try:
        required_root_files = tuple(required_root_files)
    except (TypeError, ValueError) as exc:
        raise OutcomeModelStoreError("required root metadata files are invalid") from exc
    if any(
        not isinstance(name, str)
        or name not in ROOT_METADATA_FILES
        or name in {MANIFEST_NAME, RECORDS_DIR, STATES_DIR, STAGING_DIR}
        for name in required_root_files
    ) or len(set(required_root_files)) != len(required_root_files):
        raise OutcomeModelStoreError("required root metadata files are invalid")
    try:
        with os.scandir(reader.root_fd) as iterator:
            root_entries = {
                entry.name: (
                    entry.is_symlink(),
                    entry.is_file(follow_symlinks=False),
                    entry.is_dir(follow_symlinks=False),
                )
                for entry in iterator
            }
    except OSError as exc:
        raise OutcomeModelStoreError("model-store root inventory is unreadable") from exc
    expected_root_entries = {
        MANIFEST_NAME: (False, True, False),
        RECORDS_DIR: (False, False, True),
        STATES_DIR: (False, False, True),
        STAGING_DIR: (False, False, True),
        PHASE3_KEYS_DIR: (False, False, True),
        PHASE3_COSTS_DIR: (False, False, True),
        PHASE3_ARTIFACTS_DIR: (False, False, True),
        PHASE3_STAGING_DIR: (False, False, True),
    }
    for name in ROOT_METADATA_FILES:
        observed = root_entries.get(name)
        if observed is not None:
            expected_root_entries[name] = (False, True, False)
    if root_entries != expected_root_entries or any(
        name not in root_entries for name in required_root_files
    ):
        raise OutcomeModelStoreError("model-store root inventory differs")
    _directory_shape_at(
        reader.staging_fd,
        {},
        message="model-store staging namespace is not empty",
    )
    _directory_shape_at(
        reader.records_fd,
        {f"{owner_id}.json": (True, False) for owner_id in expected},
        message="model-store record inventory differs",
    )
    _directory_shape_at(
        reader.states_fd,
        {owner_id: (False, True) for owner_id in expected},
        message="model-store state inventory differs",
    )

    manifest_raw = _read_stable(reader.root_fd, MANIFEST_NAME)
    try:
        value = json.loads(manifest_raw)
        if canonical_json_bytes(value) + b"\n" != manifest_raw:
            raise ValueError("non-canonical manifest bytes")
        manifest = OutcomeModelStoreManifest.model_validate(value)
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise OutcomeModelStoreError("model-store manifest is invalid") from exc
    if tuple(entry.owner_id for entry in manifest.entries) != expected:
        raise OutcomeModelStoreError("model-store manifest owner inventory differs")
    manifest_identity = _stable_identity_at(reader.root_fd, MANIFEST_NAME)
    root_metadata_file_identities = tuple(
        (name, _stable_identity_at(reader.root_fd, name))
        for name in ROOT_METADATA_FILES
        if name in root_entries
    )

    record_file_identities = tuple(
        (f"{owner_id}.json", _stable_identity_at(reader.records_fd, f"{owner_id}.json"))
        for owner_id in expected
    )
    state_identities: list[OutcomeModelStateIdentitySnapshot] = []
    for owner_id in expected:
        state_fd: int | None = None
        try:
            state_fd = secure_fs.open_child_directory(reader.states_fd, owner_id)
            _directory_shape_at(
                state_fd,
                {STATE_MANIFEST_NAME: (True, False), TENSORS_DIR: (False, True)},
                message="model state directory shape differs",
            )
            state_raw = _read_stable(state_fd, STATE_MANIFEST_NAME)
            try:
                value = json.loads(state_raw)
                if canonical_json_bytes(value) + b"\n" != state_raw:
                    raise ValueError("non-canonical state index bytes")
                index = OutcomeModelStateIndex.model_validate(value)
            except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
                raise OutcomeModelStoreError("model state index is invalid") from exc
            if index.owner_id != owner_id:
                raise OutcomeModelStoreError("model state index owner differs from directory")
            tensors_fd = secure_fs.open_child_directory(state_fd, TENSORS_DIR)
            try:
                expected_tensors = tuple(item.filename for item in index.tensors)
                _directory_shape_at(
                    tensors_fd,
                    {name: (True, False) for name in expected_tensors},
                    message="model state tensor inventory differs",
                )
                tensor_file_identities = tuple(
                    (name, _stable_identity_at(tensors_fd, name)) for name in expected_tensors
                )
                tensors_identity = secure_fs.directory_identity(tensors_fd)
            finally:
                os.close(tensors_fd)
            state_identities.append(
                OutcomeModelStateIdentitySnapshot(
                    owner_id=owner_id,
                    state_directory_identity=secure_fs.directory_identity(state_fd),
                    manifest_identity=_stable_identity_at(state_fd, STATE_MANIFEST_NAME),
                    tensors_directory_identity=tensors_identity,
                    tensor_file_identities=tensor_file_identities,
                )
            )
        except (OSError, secure_fs.SecureFilesystemError) as exc:
            raise OutcomeModelStoreError("model state identity inventory is unsafe") from exc
        finally:
            if state_fd is not None:
                os.close(state_fd)

    return OutcomeModelStoreIdentitySnapshot(
        root_identity=reader.identities[0],
        records_identity=reader.identities[1],
        states_identity=reader.identities[2],
        staging_identity=reader.identities[3],
        manifest_identity=manifest_identity,
        root_metadata_file_identities=root_metadata_file_identities,
        record_file_identities=record_file_identities,
        state_identities=tuple(state_identities),
    )


def _state_index_and_payload(
    reader: PinnedOutcomeModelStoreReader, owner_id: str
) -> tuple[OutcomeModelStateIndex, PinnedOutcomeModelState]:
    try:
        state_fd = secure_fs.open_child_directory(reader.states_fd, owner_id)
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise OutcomeModelStoreError("model state directory is missing or unsafe") from exc
    try:
        raw = _read_stable(state_fd, STATE_MANIFEST_NAME)
        try:
            value = json.loads(raw)
            if canonical_json_bytes(value) + b"\n" != raw:
                raise ValueError("non-canonical state index bytes")
            index = OutcomeModelStateIndex.model_validate(value)
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise OutcomeModelStoreError("model state index is invalid") from exc
        if index.owner_id != owner_id:
            raise OutcomeModelStoreError("model state index owner differs from directory")
        try:
            with os.scandir(state_fd) as iterator:
                state_entries = {
                    entry.name: (
                        entry.is_symlink(),
                        entry.is_file(follow_symlinks=False),
                        entry.is_dir(follow_symlinks=False),
                    )
                    for entry in iterator
                }
        except OSError as exc:
            raise OutcomeModelStoreError("model state inventory is unreadable") from exc
        if state_entries != {
            STATE_MANIFEST_NAME: (False, True, False),
            TENSORS_DIR: (False, False, True),
        }:
            raise OutcomeModelStoreError("model state inventory differs")
        tensors_fd = secure_fs.open_child_directory(state_fd, TENSORS_DIR)
        try:
            entries = set(secure_fs.regular_entries_at(tensors_fd))
            expected = {item.filename for item in index.tensors}
            if entries != expected:
                raise OutcomeModelStoreError("model state tensor inventory differs")
            tensors: list[OutcomeStateTensorPayload] = []
            for item in index.tensors:
                content = _read_stable(tensors_fd, item.filename)
                if (
                    len(content) != item.byte_length
                    or hashlib.sha256(content).hexdigest() != item.sha256
                ):
                    raise OutcomeModelStoreError("model state tensor hash mismatch")
                tensors.append(OutcomeStateTensorPayload(item.name, item.shape, content))
            payload = PinnedOutcomeModelState(tuple(tensors))
            _schema, digest = inspect_outcome_model_state(payload)
            if digest != index.model_state_sha256:
                raise OutcomeModelStoreError("reconstructed model state hash mismatch")
            return index, payload
        finally:
            os.close(tensors_fd)
    finally:
        os.close(state_fd)


def write_outcome_model_artifact(
    root: str | Path,
    record: OutcomeDiagnosticModelArtifactRecord,
    state_payload: PinnedOutcomeModelState,
    *,
    pinned_output: PinnedOutcomeModelStore | PinnedOutcomeModelStoreReader | None = None,
) -> OutcomeModelStoreEntry:
    """Atomically publish one canonical record and exact state tensor bytes."""
    try:
        record = OutcomeDiagnosticModelArtifactRecord.model_validate(record.model_dump(mode="json"))
        record_bytes = canonical_outcome_model_artifact_record_bytes(record)
        schemas, state_sha = inspect_outcome_model_state(state_payload)
    except (AttributeError, TypeError, ValueError, OutcomeDiagnosticModelArtifactError) as exc:
        raise OutcomeModelStoreError("model artifact is not canonical") from exc
    if state_sha != record.key.model_state_sha256 or tuple(schemas) != tuple(
        record.key.state_schema
    ):
        raise OutcomeModelStoreError("model record and state bytes differ")
    own = pinned_output is None
    context = open_outcome_model_store(root) if own else nullcontext(pinned_output)
    with context as store:
        reader = _require_reader(store)
        if Path(os.path.abspath(root)) != reader.root_path:
            raise OutcomeModelStoreError("model-store root differs from the held output descriptor")
        reader.recheck()
        owner = record.key.owner_id
        name = _record_name(owner)
        manifest = _load_manifest(reader)
        existing = next((entry for entry in manifest.entries if entry.owner_id == owner), None)
        if existing is not None and (
            existing.record_id != record.record_id or existing.model_state_sha256 != state_sha
        ):
            raise OutcomeModelStoreError("different model artifact already owns this owner")
        try:
            prior_record = _read_stable(reader.records_fd, name)
        except OutcomeModelStoreError as exc:
            if not _is_missing(exc):
                raise
        else:
            if prior_record != record_bytes:
                raise OutcomeModelStoreError("different model record already owns this owner")
        # Publish complete state directory under a staging descriptor first.
        stage_name = f"state-{owner}-{uuid.uuid4().hex}"
        os.mkdir(stage_name, 0o700, dir_fd=reader.staging_fd)
        stage_fd = secure_fs.open_child_directory(reader.staging_fd, stage_name)
        try:
            _mkdir(stage_fd, TENSORS_DIR)
            tensors_fd = secure_fs.open_child_directory(stage_fd, TENSORS_DIR)
            try:
                tensor_indexes = tuple(
                    OutcomeModelTensorIndex(
                        name=tensor.name,
                        filename=f"{i:04d}.bin",
                        shape=tensor.shape,
                        byte_length=len(tensor.data),
                        sha256=hashlib.sha256(tensor.data).hexdigest(),
                    )
                    for i, tensor in enumerate(state_payload.tensors)
                )
                for item, tensor in zip(tensor_indexes, state_payload.tensors):
                    _write_new(tensors_fd, item.filename, tensor.data, reader.staging_fd)
            finally:
                os.close(tensors_fd)
            index_body = {
                "schema_version": SCHEMA_VERSION,
                "index_id": "0" * 64,
                "owner_id": owner,
                "record_id": record.record_id,
                "model_state_sha256": state_sha,
                "tensors": [item.model_dump(mode="json") for item in tensor_indexes],
            }
            index_body["index_id"] = _sha({k: v for k, v in index_body.items() if k != "index_id"})
            index = OutcomeModelStateIndex.model_validate(index_body)
            _write_new(stage_fd, STATE_MANIFEST_NAME, _canonical(index), reader.staging_fd)
            os.fsync(stage_fd)
            reader.recheck()
            try:
                os.rename(
                    stage_name, owner, src_dir_fd=reader.staging_fd, dst_dir_fd=reader.states_fd
                )
            except FileExistsError:
                loaded, _ = _state_index_and_payload(reader, owner)
                if loaded != index:
                    raise OutcomeModelStoreError("different model state won publication race")
            else:
                os.fsync(reader.states_fd)
        finally:
            os.close(stage_fd)
            try:
                _remove_tree_at(reader.staging_fd, stage_name)
            except OSError:
                raise OutcomeModelStoreError("cannot clean losing model state staging tree")
        try:
            existing_record = _read_stable(reader.records_fd, name)
        except OutcomeModelStoreError as exc:
            if not _is_missing(exc):
                raise
            _claim_or_match(reader.records_fd, name, record_bytes, reader.staging_fd)
        else:
            if existing_record != record_bytes:
                raise OutcomeModelStoreError("different model record already owns this owner")
        entry = OutcomeModelStoreEntry(
            owner_id=owner,
            record_id=record.record_id,
            key_id=record.key.key_id,
            record_sha256=_sha(json.loads(record_bytes)),
            state_index_id=index.index_id,
            model_state_sha256=state_sha,
        )
        entries = tuple(
            sorted(
                (*[item for item in manifest.entries if item.owner_id != owner], entry),
                key=lambda item: item.owner_id,
            )
        )
        body = {
            "schema_version": SCHEMA_VERSION,
            "manifest_id": "0" * 64,
            "entries": [item.model_dump(mode="json") for item in entries],
        }
        body["manifest_id"] = _sha({k: v for k, v in body.items() if k != "manifest_id"})
        _write_new(
            reader.root_fd,
            MANIFEST_NAME,
            _canonical(OutcomeModelStoreManifest.model_validate(body)),
            reader.staging_fd,
        )
        reader.recheck()
        return entry


def load_outcome_model_artifact_at(
    reader: PinnedOutcomeModelStoreReader | PinnedOutcomeModelStore,
    owner_id: str,
    training_evidence: PinnedOutcomeTrainingEvidence,
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
) -> tuple[
    OutcomeDiagnosticModelArtifactRecord, PinnedOutcomeModelState, AuthorizedOutcomeModelArtifact
]:
    """Load and semantically authorize one model through a held descriptor pin."""
    reader = _require_reader(reader)
    record, index, state = load_outcome_model_artifact_payload_at(reader, owner_id)
    try:
        authorization = validate_outcome_model_artifact_against_plan(
            record,
            state,
            training_evidence,
            plan,
            snapshot,
            preparation_git_commit_sha=preparation_git_commit_sha,
            preparation_provenance_sha256=preparation_provenance_sha256,
        )
    except (TypeError, ValueError, OutcomeDiagnosticModelArtifactError) as exc:
        raise OutcomeModelStoreError("stored model failed canonical plan validation") from exc
    reader.recheck()
    return record, state, authorization


def load_outcome_model_artifact_payload_at(
    reader: PinnedOutcomeModelStoreReader | PinnedOutcomeModelStore,
    owner_id: str,
) -> tuple[OutcomeDiagnosticModelArtifactRecord, OutcomeModelStateIndex, PinnedOutcomeModelState]:
    """Load one stored payload after checking its complete local lineage.

    This is deliberately non-authorizing: it does not inspect a plan, training
    evidence, outcomes, or any evaluator.  It only proves that the descriptor-
    pinned manifest, record, state index, and tensor bytes form one canonical
    owner payload.  Callers needing scientific authorization must apply their
    separate plan/evidence checks after this boundary.
    """

    reader = _require_reader(reader)
    manifest = _load_manifest(reader)
    entry = next((item for item in manifest.entries if item.owner_id == owner_id), None)
    if entry is None:
        raise OutcomeModelStoreError("model owner is absent from store manifest")
    record_raw = _read_stable(reader.records_fd, _record_name(owner_id))
    try:
        record = load_outcome_model_artifact_record_bytes(record_raw)
    except (TypeError, ValueError, OutcomeDiagnosticModelArtifactError) as exc:
        raise OutcomeModelStoreError("stored model record is invalid") from exc
    if (
        record.record_id != entry.record_id
        or record.key.owner_id != owner_id
        or record.key.key_id != entry.key_id
        or record.key.model_state_sha256 != entry.model_state_sha256
        or _sha(json.loads(record_raw)) != entry.record_sha256
    ):
        raise OutcomeModelStoreError("model record manifest lineage differs")
    index, state = _state_index_and_payload(reader, owner_id)
    if (
        index.index_id != entry.state_index_id
        or index.owner_id != owner_id
        or index.record_id != record.record_id
        or index.model_state_sha256 != record.key.model_state_sha256
    ):
        raise OutcomeModelStoreError("model state manifest lineage differs")
    reader.recheck()
    return record, index, state


def load_outcome_model_manifest_at(
    reader: PinnedOutcomeModelStoreReader | PinnedOutcomeModelStore,
) -> OutcomeModelStoreManifest:
    """Load the canonical store manifest through an existing descriptor pin."""
    reader = _require_reader(reader)
    manifest = _load_manifest(reader)
    reader.recheck()
    return manifest


def scan_outcome_model_inventory_at(
    reader: PinnedOutcomeModelStoreReader | PinnedOutcomeModelStore,
    expected_owner_ids: tuple[str, ...] | list[str] | set[str],
    training_evidence_by_view: Mapping[str, PinnedOutcomeTrainingEvidence],
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    *,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
) -> OutcomeModelStoreManifest:
    """Require exact inventory and semantically authorize every stored owner."""
    reader = _require_reader(reader)
    manifest = _load_manifest(reader)
    expected = tuple(sorted(expected_owner_ids))
    if len(expected) != EXPECTED_MODEL_OWNERS or len(set(expected)) != EXPECTED_MODEL_OWNERS:
        raise OutcomeModelStoreError("inventory requires exact 240-owner authority")
    if tuple(item.owner_id for item in manifest.entries) != expected:
        raise OutcomeModelStoreError("model-store owner inventory is partial or foreign")
    try:
        with os.scandir(reader.records_fd) as iterator:
            record_entries = {
                entry.name: (
                    entry.is_symlink(),
                    entry.is_file(follow_symlinks=False),
                    entry.is_dir(follow_symlinks=False),
                )
                for entry in iterator
            }
        with os.scandir(reader.states_fd) as iterator:
            state_entries = {
                entry.name: (
                    entry.is_symlink(),
                    entry.is_file(follow_symlinks=False),
                    entry.is_dir(follow_symlinks=False),
                )
                for entry in iterator
            }
    except OSError as exc:
        raise OutcomeModelStoreError("model-store inventory is unreadable") from exc
    expected_record_names = {f"{owner}.json" for owner in expected}
    if set(record_entries) != expected_record_names or any(
        record_entries[name] != (False, True, False) for name in expected_record_names
    ):
        raise OutcomeModelStoreError("model-store record inventory has extra or unsafe entries")
    if set(state_entries) != set(expected) or any(
        state_entries[name] != (False, False, True) for name in expected
    ):
        raise OutcomeModelStoreError("model-store state inventory has extra or unsafe entries")
    if (
        type(plan) is not ValidatedOutcomePlan
        or not isinstance(snapshot, OutcomeDiagnosticProtocolSnapshot)
        or not isinstance(training_evidence_by_view, Mapping)
        or set(training_evidence_by_view) != {view.view_id for view in plan.plan.views}
    ):
        raise OutcomeModelStoreError(
            "model-store semantic inventory authority is partial or foreign"
        )
    for entry in manifest.entries:
        record_raw = _read_stable(reader.records_fd, _record_name(entry.owner_id))
        record = load_outcome_model_artifact_record_bytes(record_raw)
        if (
            record.record_id != entry.record_id
            or record.key.owner_id != entry.owner_id
            or record.key.key_id != entry.key_id
            or record.key.model_state_sha256 != entry.model_state_sha256
            or getattr(record.key, "preparation_git_commit_sha", None) != preparation_git_commit_sha
            or getattr(record.key, "preparation_provenance_sha256", None)
            != preparation_provenance_sha256
            or _sha(json.loads(record_raw)) != entry.record_sha256
        ):
            raise OutcomeModelStoreError("model-store record inventory differs")
        index, state = _state_index_and_payload(reader, entry.owner_id)
        if (
            index.index_id != entry.state_index_id
            or index.owner_id != entry.owner_id
            or index.record_id != entry.record_id
            or index.model_state_sha256 != entry.model_state_sha256
        ):
            raise OutcomeModelStoreError("model-store state inventory differs")
        try:
            validate_outcome_model_artifact_against_plan(
                record,
                state,
                training_evidence_by_view[record.key.view_id],
                plan,
                snapshot,
                preparation_git_commit_sha=preparation_git_commit_sha,
                preparation_provenance_sha256=preparation_provenance_sha256,
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            OutcomeDiagnosticModelArtifactError,
        ) as exc:
            raise OutcomeModelStoreError(
                "model-store owner failed semantic inventory validation"
            ) from exc
    reader.recheck()
    return manifest


__all__ = [
    "MANIFEST_NAME",
    "ROOT_METADATA_FILES",
    "OutcomeModelStateIdentitySnapshot",
    "OutcomeModelStateIndex",
    "OutcomeModelStoreIdentitySnapshot",
    "OutcomeModelStoreEntry",
    "OutcomeModelStoreError",
    "OutcomeModelStoreManifest",
    "PinnedOutcomeModelStore",
    "PinnedOutcomeModelStoreReader",
    "load_outcome_model_artifact_at",
    "load_outcome_model_artifact_payload_at",
    "load_outcome_model_manifest_at",
    "open_existing_outcome_model_store",
    "open_outcome_model_store",
    "scan_outcome_model_inventory_at",
    "snapshot_outcome_model_store_identities_at",
    "write_outcome_model_artifact",
]
