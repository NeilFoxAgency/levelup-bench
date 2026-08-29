"""Atomic once-only publication of the complete local-affordance raw store.

The publisher accepts only 240 already-sanitized, capability-free persisted
artifacts.  It derives every key index, LOFO manifest, and held-out binding,
builds the complete tree in a private sibling staging directory, validates the
staged authority, and activates it with a no-replace rename.  Existing roots are
always conflicts, including byte-identical roots.  No resume, overwrite,
environment, search, evaluator, or execution behavior exists here.
"""

from __future__ import annotations

import ctypes
import errno
import os
import secrets
import sys
from pathlib import Path
from typing import Iterable

from levelup.experiments.milestone6_phase3_local_affordance_raw_authority import (
    ExpectedRawProbeAuthority,
    PersistedRawProbeArtifact,
    RawProbeAuthoritySnapshot,
    require_expected_raw_probe_authority,
    validate_complete_raw_probe_authority,
)
from levelup.experiments.milestone6_phase3_local_affordance_raw_store import (
    ARTIFACTS_DIR,
    FAMILY_ORDER,
    HELDOUT_BINDINGS_DIR,
    KEYS_DIR,
    TRAINING_FOLDS_DIR,
    HeldoutProbeBinding,
    RawProbeTaskKeyIndex,
    RawProbeTaskReference,
    TrainingFoldManifest,
    open_existing_raw_probe_store,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes

_STAGING_ATTEMPTS = 32


class RawProbePublicationError(RuntimeError):
    """Raised when an immutable raw-store publication fails closed."""


def _canonical_file(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return canonical_json_bytes(value) + b"\n"


def _entry_exists(parent_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RawProbePublicationError(f"cannot inspect publication destination: {name}") from exc
    return True


def _write_all(file_fd: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(file_fd, view)
        if written <= 0:
            raise OSError("short raw-store write")
        view = view[written:]


def _write_staged_file(directory_fd: int, name: str, content: bytes) -> None:
    """Write one final-name file inside a private, unpublished staging tree."""

    file_fd: int | None = None
    try:
        file_fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        _write_all(file_fd, content)
        os.fsync(file_fd)
    except (OSError, TypeError, ValueError) as exc:
        raise RawProbePublicationError(f"cannot stage raw-store file: {name}") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)


def _mkdir_child(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        return secure_fs.open_child_directory(parent_fd, name)
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise RawProbePublicationError(f"cannot create raw-store namespace: {name}") from exc


def _allocate_staging(parent_fd: int, target_name: str) -> tuple[str, int, tuple[int, int]]:
    for _ in range(_STAGING_ATTEMPTS):
        name = f".{target_name}.staging-{secrets.token_hex(12)}"
        descriptor: int | None = None
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            descriptor = secure_fs.open_child_directory(parent_fd, name)
            identity = secure_fs.directory_identity(descriptor)
            lexical = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (lexical.st_dev, lexical.st_ino) != identity:
                raise RawProbePublicationError("raw-store staging root was substituted")
            return name, descriptor, identity
        except FileExistsError:
            continue
        except (OSError, secure_fs.SecureFilesystemError) as exc:
            if descriptor is not None:
                os.close(descriptor)
            # Once mkdirat succeeds but descriptor pinning fails, ownership of
            # the lexical name is unknowable.  Never delete that name: a local
            # race may already have substituted an unrelated directory.
            raise RawProbePublicationError("cannot allocate raw-store staging root") from exc
    raise RawProbePublicationError("cannot allocate a unique raw-store staging root")


def _erase_staging_contents_at(
    parent_fd: int,
    name: str,
    *,
    expected_identity: tuple[int, int],
    namespace_identities: dict[str, tuple[int, int]],
) -> None:
    """Erase expected files through pinned dirs while retaining every dir."""

    root_fd: int | None = None
    try:
        root_fd = secure_fs.open_child_directory(parent_fd, name)
        if secure_fs.directory_identity(root_fd) != expected_identity:
            raise RawProbePublicationError("refusing to clean a substituted staging root")
        with os.scandir(root_fd) as iterator:
            entries = tuple(iterator)
        for entry in entries:
            if entry.is_dir(follow_symlinks=False) and not entry.is_symlink():
                expected_child = namespace_identities.get(entry.name)
                if expected_child is None:
                    raise RawProbePublicationError(
                        "refusing to clean an unexpected staging namespace"
                    )
                namespace_fd = secure_fs.open_child_directory(root_fd, entry.name)
                try:
                    if secure_fs.directory_identity(namespace_fd) != expected_child:
                        raise RawProbePublicationError(
                            "refusing to clean a substituted staging namespace"
                        )
                    with os.scandir(namespace_fd) as iterator:
                        children = tuple(iterator)
                    if any(
                        child.is_dir(follow_symlinks=False) and not child.is_symlink()
                        for child in children
                    ):
                        raise RawProbePublicationError(
                            "refusing to recurse into an unexpected staging directory"
                        )
                    for child in children:
                        os.unlink(child.name, dir_fd=namespace_fd)
                    os.fsync(namespace_fd)
                finally:
                    os.close(namespace_fd)
            elif entry.name == "manifest.json":
                os.unlink(entry.name, dir_fd=root_fd)
            else:
                raise RawProbePublicationError("refusing to clean an unexpected staging entry")
        os.fsync(root_fd)
    except FileNotFoundError:
        return
    finally:
        if root_fd is not None:
            os.close(root_fd)


def _lexical_parent_identity(parent: Path) -> tuple[int, int]:
    descriptor = secure_fs.open_directory_chain(parent)
    try:
        return secure_fs.directory_identity(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(parent_fd: int, source: str, destination: str) -> None:
    """Atomically activate a directory while refusing every existing target."""

    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:
            raise RawProbePublicationError("renameat2 is required for no-replace activation")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(parent_fd, source_bytes, parent_fd, destination_bytes, 1)
    elif sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is None:
            raise RawProbePublicationError("renameatx_np is required for no-replace activation")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        # Darwin's RENAME_EXCL is the no-replace operation.
        result = function(parent_fd, source_bytes, parent_fd, destination_bytes, 0x00000004)
    else:
        raise RawProbePublicationError(
            "raw-store activation requires Linux renameat2 or Darwin renameatx_np"
        )
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise RawProbePublicationError("raw-store destination already exists")
        raise RawProbePublicationError(
            f"cannot atomically activate raw-store authority: errno {error}"
        )


def _activate_staged_store(parent_fd: int, source: str, destination: str) -> None:
    """Narrow test seam around the platform no-replace activation syscall."""

    _rename_noreplace(parent_fd, source, destination)


def _validated_artifacts(
    expected: ExpectedRawProbeAuthority,
    artifacts: Iterable[PersistedRawProbeArtifact],
) -> tuple[PersistedRawProbeArtifact, ...]:
    try:
        supplied = tuple(artifacts)
    except TypeError as exc:
        raise RawProbePublicationError("raw artifacts must be a finite iterable") from exc
    if len(supplied) != 240 or any(
        type(artifact) is not PersistedRawProbeArtifact for artifact in supplied
    ):
        raise RawProbePublicationError("publication requires 240 exact persisted artifacts")
    validated: list[PersistedRawProbeArtifact] = []
    try:
        for artifact in supplied:
            validated.append(
                PersistedRawProbeArtifact.model_validate(artifact.model_dump(mode="json"))
            )
    except (TypeError, ValueError) as exc:
        raise RawProbePublicationError("persisted raw artifact is invalid") from exc
    by_key = {artifact.key.key_id: artifact for artifact in validated}
    if len(by_key) != 240 or set(by_key) != {key.key_id for key in expected.keys}:
        raise RawProbePublicationError("persisted artifact key universe differs from authority")
    ordered = tuple(by_key[key.key_id] for key in expected.keys)
    if len({artifact.manifest.artifact_id for artifact in ordered}) != 240:
        raise RawProbePublicationError("persisted artifact identities are duplicated")
    return ordered


def _derived_manifests(
    expected: ExpectedRawProbeAuthority,
    artifacts: tuple[PersistedRawProbeArtifact, ...],
) -> tuple[
    tuple[RawProbeTaskKeyIndex, ...],
    tuple[TrainingFoldManifest, ...],
    tuple[HeldoutProbeBinding, ...],
]:
    indices = tuple(
        RawProbeTaskKeyIndex(
            key_id=artifact.key.key_id,
            artifact_id=artifact.manifest.artifact_id,
            key=artifact.key,
        )
        for artifact in artifacts
    )
    index_by_key = {index.key_id: index for index in indices}

    def reference(key_id: str) -> RawProbeTaskReference:
        index = index_by_key[key_id]
        return RawProbeTaskReference(
            artifact_id=index.artifact_id,
            key_id=index.key_id,
            key=index.key,
        )

    # The deliberately explicit loops below keep the scientific order visible
    # and avoid deriving families from arbitrary artifact content.
    folds = tuple(
        TrainingFoldManifest(
            fold_id=heldout,
            heldout_family=heldout,
            replicate=replicate,
            task_references=tuple(
                reference(key.key_id)
                for key in expected.keys
                if key.replicate == replicate and key.family_id != heldout
            ),
        )
        for heldout in FAMILY_ORDER
        for replicate in range(5)
    )
    bindings = tuple(
        HeldoutProbeBinding(
            fold_id=key.family_id,
            family_id=key.family_id,
            replicate=key.replicate,
            task_reference=reference(key.key_id),
        )
        for key in expected.keys
    )
    return indices, folds, bindings


def publish_raw_probe_store(
    destination: str | Path,
    *,
    expected: ExpectedRawProbeAuthority,
    artifacts: Iterable[PersistedRawProbeArtifact],
) -> RawProbeAuthoritySnapshot:
    """Publish and validate one exact, immutable development-only raw store."""

    try:
        require_expected_raw_probe_authority(expected)
    except ValueError as exc:
        raise RawProbePublicationError(
            "publication requires the frozen raw authority expectation"
        ) from exc
    target = Path(os.path.abspath(destination))
    if not target.name or target == target.parent:
        raise RawProbePublicationError("raw-store destination must name one child root")
    ordered_artifacts = _validated_artifacts(expected, artifacts)
    indices, folds, bindings = _derived_manifests(expected, ordered_artifacts)
    parent_fd: int | None = None
    staging_fd: int | None = None
    namespace_fds: list[int] = []
    namespace_identities: dict[str, tuple[int, int]] = {}
    staging_name: str | None = None
    staging_identity: tuple[int, int] | None = None
    activated = False
    try:
        parent_fd = secure_fs.open_directory_chain(target.parent)
        parent_identity = secure_fs.directory_identity(parent_fd)
        if _entry_exists(parent_fd, target.name):
            raise RawProbePublicationError("raw-store destination already exists")
        staging_name, staging_fd, staging_identity = _allocate_staging(parent_fd, target.name)
        namespace_by_name: dict[str, int] = {}
        for name in (
            ARTIFACTS_DIR,
            KEYS_DIR,
            TRAINING_FOLDS_DIR,
            HELDOUT_BINDINGS_DIR,
        ):
            descriptor = _mkdir_child(staging_fd, name)
            namespace_fds.append(descriptor)
            namespace_by_name[name] = descriptor
            namespace_identities[name] = secure_fs.directory_identity(descriptor)
        _write_staged_file(
            staging_fd,
            "manifest.json",
            _canonical_file(expected.manifest),
        )
        for artifact in ordered_artifacts:
            _write_staged_file(
                namespace_by_name[ARTIFACTS_DIR],
                f"{artifact.manifest.artifact_id}.json",
                _canonical_file(artifact),
            )
        for index in indices:
            _write_staged_file(
                namespace_by_name[KEYS_DIR],
                f"{index.key_id}.json",
                _canonical_file(index),
            )
        for fold in folds:
            _write_staged_file(
                namespace_by_name[TRAINING_FOLDS_DIR],
                f"{fold.fold_id}.r{fold.replicate}.json",
                _canonical_file(fold),
            )
        for binding in bindings:
            reference = binding.task_reference
            _write_staged_file(
                namespace_by_name[HELDOUT_BINDINGS_DIR],
                (f"{binding.family_id}.r{binding.replicate}.task-{reference.task_index}.json"),
                _canonical_file(binding),
            )
        for descriptor in namespace_fds:
            os.fsync(descriptor)
        os.fsync(staging_fd)
        for descriptor in reversed(namespace_fds):
            os.close(descriptor)
        namespace_fds.clear()

        staging_path = target.parent / staging_name
        with open_existing_raw_probe_store(staging_path) as reader:
            if (reader.identities[0].device, reader.identities[0].inode) != staging_identity:
                raise RawProbePublicationError("staging path was substituted")
            staged_snapshot = validate_complete_raw_probe_authority(reader, expected=expected)
        if _lexical_parent_identity(target.parent) != parent_identity:
            raise RawProbePublicationError("publication parent path was substituted")
        if _entry_exists(parent_fd, target.name):
            raise RawProbePublicationError("raw-store destination appeared during staging")
        current_stage = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
        if (current_stage.st_dev, current_stage.st_ino) != staging_identity:
            raise RawProbePublicationError("staging root changed before activation")
        _activate_staged_store(parent_fd, staging_name, target.name)
        activated = True
        os.fsync(parent_fd)
        if _lexical_parent_identity(target.parent) != parent_identity:
            raise RawProbePublicationError("publication parent changed after activation")
        with open_existing_raw_probe_store(target) as reader:
            if (reader.identities[0].device, reader.identities[0].inode) != staging_identity:
                raise RawProbePublicationError("activated raw-store root was substituted")
            final_snapshot = validate_complete_raw_probe_authority(reader, expected=expected)
        if final_snapshot.authority_content_sha256 != staged_snapshot.authority_content_sha256:
            raise RawProbePublicationError("raw-store content changed during activation")
        return final_snapshot
    except RawProbePublicationError:
        raise
    except (OSError, TypeError, ValueError, secure_fs.SecureFilesystemError) as exc:
        raise RawProbePublicationError("raw-store publication failed closed") from exc
    finally:
        cleanup_error: BaseException | None = None
        for descriptor in reversed(namespace_fds):
            try:
                os.close(descriptor)
            except OSError:
                pass
        if parent_fd is not None:
            if staging_name is not None and staging_identity is not None and not activated:
                try:
                    _erase_staging_contents_at(
                        parent_fd,
                        staging_name,
                        expected_identity=staging_identity,
                        # POSIX has no portable unlink-directory-by-fd
                        # operation. Retain the empty hidden directory
                        # skeleton rather than path-rmdir any namespace.
                        namespace_identities=namespace_identities,
                    )
                    os.fsync(parent_fd)
                except (
                    OSError,
                    RawProbePublicationError,
                    secure_fs.SecureFilesystemError,
                ) as exc:
                    cleanup_error = exc
        if staging_fd is not None:
            try:
                os.close(staging_fd)
            except OSError:
                pass
        if parent_fd is not None:
            os.close(parent_fd)
        if cleanup_error is not None:
            raise RawProbePublicationError(
                "cannot clean failed raw-store staging tree"
            ) from cleanup_error


__all__ = [
    "RawProbePublicationError",
    "publish_raw_probe_store",
]
