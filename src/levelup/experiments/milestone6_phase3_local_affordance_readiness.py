"""Read-only readiness boundary for local-affordance raw publication.

This module is intentionally narrower than the older Phase 3 readiness gate.
It pins only the four committed development authorities and an empty raw-store
destination.  Diagnostic capture performs no model/runtime work.  Activation
holds descriptor-pinned source files and the destination parent until a later
publisher consumes the lease.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from levelup.experiments.milestone6_phase3_local_affordance_raw_authority import (
    ExpectedRawProbeAuthority,
    RawProbeAuthorityError,
    build_expected_raw_probe_authority,
    require_expected_raw_probe_authority,
)
from levelup.experiments.milestone6_phase3_protocol import ROOT
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes


class LocalAffordanceReadinessError(ValueError):
    """Raised when local-affordance readiness cannot be proven."""


SOURCE_RELATIVE_PATHS = (
    "configs/milestone6/phase3_local_affordance_protocol.json",
    "configs/milestone6/development_protocol.json",
    "configs/milestone6/development_tasks.json",
    "configs/milestone6/phase3_evidence_lock.json",
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_LEASE_TOKEN = object()
_SNAPSHOT_TOKEN = object()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity(value: os.stat_result) -> tuple[int, int]:
    if not stat.S_ISREG(value.st_mode) and not stat.S_ISDIR(value.st_mode):
        raise LocalAffordanceReadinessError("authority path contains a non-regular entry")
    return int(value.st_dev), int(value.st_ino)


def _descriptor_sha256(file_fd: int) -> tuple[tuple[int, int], str]:
    before = os.fstat(file_fd)
    os.lseek(file_fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(file_fd, 1024 * 1024):
        digest.update(chunk)
    after = os.fstat(file_fd)
    os.lseek(file_fd, 0, os.SEEK_SET)
    before_identity = _identity(before)
    if before_identity != _identity(after):
        raise LocalAffordanceReadinessError("held source changed during digest")
    return before_identity, digest.hexdigest()


def _relative(value: str | os.PathLike[str]) -> str:
    pure = PurePosixPath(str(value))
    if pure.is_absolute() or not pure.parts or "." in pure.parts or ".." in pure.parts:
        raise LocalAffordanceReadinessError("source path must be repository-relative")
    if any(not part or "\\" in part or "\x00" in part for part in pure.parts):
        raise LocalAffordanceReadinessError("source path contains an unsafe component")
    return "/".join(pure.parts)


def _absolute(repository: Path, value: str | os.PathLike[str], label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repository / path
    return Path(os.path.abspath(path))


def _reject_lexical_symlinks(path: Path, label: str) -> None:
    """Reject symlinks in every existing lexical ancestor."""

    absolute = Path(os.path.abspath(path))
    chain = list(reversed(absolute.parents)) + [absolute]
    for component in chain:
        try:
            observed = component.lstat()
        except FileNotFoundError:
            # A destination itself may be absent; all ancestors are checked by
            # opening the parent descriptor below.
            continue
        except OSError as exc:
            raise LocalAffordanceReadinessError(
                f"{label} path cannot be inspected: {component}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode):
            raise LocalAffordanceReadinessError(
                f"{label} path contains a lexical symlink: {component}"
            )


def _read_file_from_root(
    root_fd: int, relative_path: str, *, stack: ExitStack | None = None
) -> "LocalAffordanceSourceSnapshot":
    components = _relative(relative_path).split("/")
    parent_fd = root_fd
    opened: list[int] = []
    ancestors: list[tuple[str, tuple[int, int]]] = [("", _identity(os.fstat(root_fd)))]
    try:
        for index, component in enumerate(components[:-1]):
            child_fd = secure_fs.open_child_directory(parent_fd, component)
            opened.append(child_fd)
            ancestors.append(("/".join(components[: index + 1]), _identity(os.fstat(child_fd))))
            parent_fd = child_fd
        parent_before = os.fstat(parent_fd)
        parent_identity = _identity(parent_before)
        with secure_fs.open_regular_file_at(parent_fd, components[-1]) as file_fd:
            before = os.fstat(file_fd)
            chunks: list[bytes] = []
            while chunk := os.read(file_fd, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(file_fd)
        if _identity(before) != _identity(after) or _identity(parent_before) != _identity(
            os.fstat(parent_fd)
        ):
            raise LocalAffordanceReadinessError("source changed during read")
        return LocalAffordanceSourceSnapshot(
            relative_path=_relative(relative_path),
            content=b"".join(chunks),
            sha256=_sha256(b"".join(chunks)),
            parent_identity=parent_identity,
            file_identity=_identity(before),
            ancestor_identities=tuple(ancestors),
        )
    finally:
        if stack is not None:
            for fd in opened:
                stack.callback(os.close, fd)
        else:
            for fd in reversed(opened):
                os.close(fd)


def _open_root(repository: Path) -> tuple[int, tuple[int, int]]:
    _reject_lexical_symlinks(repository, "repository")
    try:
        fd = secure_fs.open_directory_chain(repository)
        identity = _identity(os.fstat(fd))
        return fd, identity
    except (OSError, RuntimeError, ValueError, secure_fs.SecureFilesystemError) as exc:
        raise LocalAffordanceReadinessError("repository is not a safe real directory") from exc


def _open_directory(path: Path) -> tuple[int, tuple[tuple[str, tuple[int, int]], ...]]:
    _reject_lexical_symlinks(path, "destination parent")
    current_fd: int | None = None
    try:
        absolute = Path(os.path.abspath(path))
        current = Path(absolute.anchor)
        current_fd = secure_fs.open_directory_chain(current)
        ancestors: list[tuple[str, tuple[int, int]]] = [
            (str(current), _identity(os.fstat(current_fd)))
        ]
        for component in absolute.parts[1:]:
            child_fd = secure_fs.open_child_directory(current_fd, component)
            os.close(current_fd)
            current_fd = child_fd
            current /= component
            ancestors.append((str(current), _identity(os.fstat(current_fd))))
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise LocalAffordanceReadinessError("destination parent is not a directory")
        result = current_fd
        current_fd = None
        return result, tuple(ancestors)
    except (OSError, RuntimeError, ValueError, secure_fs.SecureFilesystemError) as exc:
        raise LocalAffordanceReadinessError("destination parent is not safely openable") from exc
    finally:
        if current_fd is not None:
            os.close(current_fd)


def _git_state(repository: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=repository, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain", "--untracked-files=all"),
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LocalAffordanceReadinessError("cannot capture repository git state") from exc
    if _COMMIT_RE.fullmatch(commit) is None:
        raise LocalAffordanceReadinessError("repository commit identity is malformed")
    return commit, dirty


def _hold_source_from_root(
    root_fd: int,
    expected: "LocalAffordanceSourceSnapshot",
    stack: ExitStack,
) -> int:
    """Open, verify, and retain one exact source descriptor for activation."""

    components = expected.relative_path.split("/")
    parent_fd = root_fd
    ancestors: list[tuple[str, tuple[int, int]]] = [("", _identity(os.fstat(root_fd)))]
    for index, component in enumerate(components[:-1]):
        child_fd = secure_fs.open_child_directory(parent_fd, component)
        stack.callback(os.close, child_fd)
        ancestors.append(("/".join(components[: index + 1]), _identity(os.fstat(child_fd))))
        parent_fd = child_fd
    parent_identity = _identity(os.fstat(parent_fd))
    source_fd = os.open(
        components[-1],
        os.O_RDONLY | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    stack.callback(os.close, source_fd)
    before = os.fstat(source_fd)
    if not stat.S_ISREG(before.st_mode):
        raise LocalAffordanceReadinessError("held source is not a regular file")
    chunks: list[bytes] = []
    while chunk := os.read(source_fd, 1024 * 1024):
        chunks.append(chunk)
    after = os.fstat(source_fd)
    if (
        _identity(before) != expected.file_identity
        or _identity(after) != expected.file_identity
        or parent_identity != expected.parent_identity
        or tuple(ancestors) != expected.ancestor_identities
        or b"".join(chunks) != expected.content
    ):
        raise LocalAffordanceReadinessError(f"held source differs: {expected.relative_path}")
    os.lseek(source_fd, 0, os.SEEK_SET)
    return source_fd


@dataclass(frozen=True, slots=True)
class LocalAffordanceSourceSnapshot:
    relative_path: str
    content: bytes
    sha256: str
    parent_identity: tuple[int, int]
    file_identity: tuple[int, int]
    ancestor_identities: tuple[tuple[str, tuple[int, int]], ...]


def _source_matrix_digest(
    sources: tuple[LocalAffordanceSourceSnapshot, ...],
) -> str:
    return _sha256(
        canonical_json_bytes(
            tuple(
                {
                    "relative_path": item.relative_path,
                    "content": item.content.hex(),
                    "sha256": item.sha256,
                    "parent_identity": item.parent_identity,
                    "file_identity": item.file_identity,
                    "ancestor_identities": item.ancestor_identities,
                }
                for item in sources
            )
        )
    )


@dataclass(frozen=True, slots=True)
class _ReadinessSeal:
    digest: str
    token: object


def _seal_digest(snapshot: "LocalAffordanceReadinessSnapshot") -> str:
    return _sha256(
        canonical_json_bytes(
            {
                "repository": str(snapshot.repository),
                "repository_identity": snapshot.repository_identity,
                "sources": [
                    {
                        "relative_path": item.relative_path,
                        "sha256": item.sha256,
                        "content": item.content.hex(),
                        "parent_identity": item.parent_identity,
                        "file_identity": item.file_identity,
                        "ancestor_identities": item.ancestor_identities,
                    }
                    for item in snapshot.sources
                ],
                "destination_parent": str(snapshot.destination_parent),
                "destination_name": snapshot.destination_name,
                "destination_parent_identity": snapshot.destination_parent_identity,
                "destination_ancestors": snapshot.destination_ancestors,
                "git_commit_sha": snapshot.git_commit_sha,
                "git_dirty": snapshot.git_dirty,
                "authority": snapshot.authority.manifest.model_dump(mode="json"),
            }
        )
    )


@dataclass(frozen=True, slots=True, init=False)
class LocalAffordanceReadinessSnapshot:
    repository: Path
    sources: tuple[LocalAffordanceSourceSnapshot, ...]
    authority: ExpectedRawProbeAuthority
    destination_parent: Path
    destination_name: str
    destination_parent_identity: tuple[int, int]
    destination_ancestors: tuple[tuple[str, tuple[int, int]], ...]
    repository_identity: tuple[int, int]
    git_commit_sha: str
    git_dirty: bool
    _seal: _ReadinessSeal
    _token: object

    def __init__(self, *, _token: object | None = None, **kwargs: Any) -> None:
        if _token is not _SNAPSHOT_TOKEN:
            raise LocalAffordanceReadinessError("readiness snapshots require canonical capture")
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, "_token", _SNAPSHOT_TOKEN)
        object.__setattr__(self, "_seal", _ReadinessSeal("", _SNAPSHOT_TOKEN))
        object.__setattr__(self, "_seal", _ReadinessSeal(_seal_digest(self), _SNAPSHOT_TOKEN))

    def require_sealed(self) -> "LocalAffordanceReadinessSnapshot":
        try:
            valid = (
                type(self) is LocalAffordanceReadinessSnapshot
                and self._token is _SNAPSHOT_TOKEN
                and type(self._seal) is _ReadinessSeal
                and self._seal.token is _SNAPSHOT_TOKEN
                and self._seal.digest == _seal_digest(self)
            )
        except (AttributeError, TypeError, ValueError):
            valid = False
        if not valid:
            raise LocalAffordanceReadinessError("readiness snapshot is forged or rebound")
        try:
            require_expected_raw_probe_authority(self.authority)
        except RawProbeAuthorityError as exc:
            raise LocalAffordanceReadinessError("readiness authority seal is invalid") from exc
        return self

    @property
    def source_by_path(self) -> Mapping[str, LocalAffordanceSourceSnapshot]:
        return {item.relative_path: item for item in self.sources}

    def _check_destination(self, parent_fd: int | None = None) -> None:
        fd: int | None = None
        close = False
        try:
            if parent_fd is None:
                fd, _ = _open_directory(self.destination_parent)
                close = True
            else:
                fd = parent_fd
            try:
                os.stat(self.destination_name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                return
            raise LocalAffordanceReadinessError("raw publication destination already exists")
        except LocalAffordanceReadinessError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise LocalAffordanceReadinessError(
                "cannot inspect raw publication destination"
            ) from exc
        finally:
            if close and fd is not None:
                os.close(fd)

    def recheck(
        self, *, for_execution: bool = False, expected_git_commit: str | None = None
    ) -> None:
        self.require_sealed()
        fd, repository_identity = _open_root(self.repository)
        try:
            if repository_identity != self.repository_identity:
                raise LocalAffordanceReadinessError("repository identity changed")
            current_sources = tuple(
                _read_file_from_root(fd, item.relative_path) for item in self.sources
            )
        finally:
            os.close(fd)
        for expected, current in zip(self.sources, current_sources, strict=True):
            if current != expected:
                raise LocalAffordanceReadinessError(f"source drifted: {expected.relative_path}")
        parent_fd, ancestors = _open_directory(self.destination_parent)
        try:
            parent_identity = _identity(os.fstat(parent_fd))
            if (
                parent_identity != self.destination_parent_identity
                or ancestors != self.destination_ancestors
            ):
                raise LocalAffordanceReadinessError("destination parent identity changed")
            self._check_destination(parent_fd)
        finally:
            os.close(parent_fd)
        commit, dirty = _git_state(self.repository)
        if commit != self.git_commit_sha or dirty != self.git_dirty:
            raise LocalAffordanceReadinessError("repository git state changed")
        if for_execution:
            if dirty:
                raise LocalAffordanceReadinessError(
                    "execution activation requires a clean repository"
                )
            if expected_git_commit is None or expected_git_commit != commit:
                raise LocalAffordanceReadinessError("execution activation commit is not authorized")

    def preflight(
        self, *, for_execution: bool = False, expected_git_commit: str | None = None
    ) -> None:
        self.recheck(for_execution=for_execution, expected_git_commit=expected_git_commit)

    @contextmanager
    def activation(self, *, expected_git_commit: str) -> Iterator["LocalAffordanceActivationLease"]:
        self.recheck(for_execution=True, expected_git_commit=expected_git_commit)
        stack = ExitStack()
        stack.__enter__()
        lease: LocalAffordanceActivationLease | None = None
        try:
            root_fd = secure_fs.open_directory_chain(self.repository)
            stack.callback(os.close, root_fd)
            if _identity(os.fstat(root_fd)) != self.repository_identity:
                raise LocalAffordanceReadinessError("repository changed during activation")
            source_fds: dict[str, int] = {}
            for expected in self.sources:
                source_fds[expected.relative_path] = _hold_source_from_root(
                    root_fd,
                    expected,
                    stack,
                )
            parent_fd, ancestors = _open_directory(self.destination_parent)
            stack.callback(os.close, parent_fd)
            parent_identity = _identity(os.fstat(parent_fd))
            if (
                parent_identity != self.destination_parent_identity
                or ancestors != self.destination_ancestors
            ):
                raise LocalAffordanceReadinessError("destination parent changed during activation")
            self._check_destination(parent_fd)
            commit, dirty = _git_state(self.repository)
            if dirty or commit != expected_git_commit or commit != self.git_commit_sha:
                raise LocalAffordanceReadinessError(
                    "repository authorization changed during activation"
                )
            lease = LocalAffordanceActivationLease(
                authority=self.authority,
                repository=self.repository,
                repository_root_fd=root_fd,
                repository_identity=self.repository_identity,
                git_commit_sha=self.git_commit_sha,
                destination_parent=self.destination_parent,
                destination_parent_fd=parent_fd,
                destination_parent_identity=parent_identity,
                destination_name=self.destination_name,
                _source_fds=source_fds,
                _source_sha256={item.relative_path: item.sha256 for item in self.sources},
                _sources=self.sources,
                _token=_LEASE_TOKEN,
            )
            yield lease.require_active()
        except LocalAffordanceReadinessError:
            raise
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            secure_fs.SecureFilesystemError,
        ) as exc:
            raise LocalAffordanceReadinessError(
                "cannot activate local-affordance readiness"
            ) from exc
        finally:
            if lease is not None:
                lease._deactivate()
            stack.close()


@dataclass(frozen=True, slots=True)
class _ActivationLeaseSeal:
    authority: ExpectedRawProbeAuthority
    repository: Path
    repository_root_fd: int
    repository_identity: tuple[int, int]
    git_commit_sha: str
    destination_parent: Path
    destination_parent_fd: int
    destination_parent_identity: tuple[int, int]
    destination_name: str
    source_descriptors: tuple[tuple[str, int, tuple[int, int], str], ...]
    sources: tuple[LocalAffordanceSourceSnapshot, ...]
    source_matrix_digest: str
    token: object


@dataclass(slots=True, init=False, repr=False)
class LocalAffordanceActivationLease:
    authority: ExpectedRawProbeAuthority
    repository: Path
    repository_root_fd: int
    repository_identity: tuple[int, int]
    git_commit_sha: str
    destination_parent: Path
    destination_parent_fd: int
    destination_parent_identity: tuple[int, int]
    destination_name: str
    _source_fds: Mapping[str, int]
    _source_sha256: Mapping[str, str]
    _sources: tuple[LocalAffordanceSourceSnapshot, ...]
    _seal: _ActivationLeaseSeal
    _active: bool
    _token: object

    def __init__(self, *, _token: object | None = None, **kwargs: Any) -> None:
        if _token is not _LEASE_TOKEN:
            raise LocalAffordanceReadinessError("activation lease requires canonical context")
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)
        source_fds = MappingProxyType(dict(self._source_fds))
        source_sha256 = MappingProxyType(dict(self._source_sha256))
        sources = tuple(self._sources)
        if set(source_fds) != set(source_sha256):
            raise LocalAffordanceReadinessError("activation source digest matrix differs")
        if tuple(item.relative_path for item in sources) != tuple(source_fds):
            raise LocalAffordanceReadinessError("activation source identity matrix differs")
        object.__setattr__(self, "_source_fds", source_fds)
        object.__setattr__(self, "_source_sha256", source_sha256)
        object.__setattr__(self, "_sources", sources)
        source_descriptors = tuple(
            (name, fd, *_descriptor_sha256(fd)) for name, fd in sorted(source_fds.items())
        )
        if any(
            digest != source_sha256[name]
            for name, _fd, _identity_value, digest in source_descriptors
        ):
            raise LocalAffordanceReadinessError("activation source content differs")
        object.__setattr__(
            self,
            "_seal",
            _ActivationLeaseSeal(
                authority=self.authority,
                repository=self.repository,
                repository_root_fd=self.repository_root_fd,
                repository_identity=self.repository_identity,
                git_commit_sha=self.git_commit_sha,
                destination_parent=self.destination_parent,
                destination_parent_fd=self.destination_parent_fd,
                destination_parent_identity=self.destination_parent_identity,
                destination_name=self.destination_name,
                source_descriptors=source_descriptors,
                sources=sources,
                source_matrix_digest=_source_matrix_digest(sources),
                token=_LEASE_TOKEN,
            ),
        )
        object.__setattr__(self, "_active", True)
        object.__setattr__(self, "_token", _LEASE_TOKEN)

    def require_active(self) -> "LocalAffordanceActivationLease":
        try:
            seal = self._seal
            source_descriptors = tuple(
                (name, fd, *_descriptor_sha256(fd)) for name, fd in sorted(self._source_fds.items())
            )
            valid = (
                type(self) is LocalAffordanceActivationLease
                and self._active
                and self._token is _LEASE_TOKEN
                and type(seal) is _ActivationLeaseSeal
                and seal.token is _LEASE_TOKEN
                and self.authority is seal.authority
                and self.repository == seal.repository
                and self.repository_root_fd == seal.repository_root_fd
                and self.repository_identity == seal.repository_identity
                and self.git_commit_sha == seal.git_commit_sha
                and self.destination_parent == seal.destination_parent
                and self.destination_parent_fd == seal.destination_parent_fd
                and self.destination_parent_identity == seal.destination_parent_identity
                and self.destination_name == seal.destination_name
                and source_descriptors == seal.source_descriptors
                and self._sources is seal.sources
                and _source_matrix_digest(self._sources) == seal.source_matrix_digest
                and all(
                    digest == self._source_sha256[name]
                    for name, _fd, _identity_value, digest in source_descriptors
                )
            )
        except (AttributeError, OSError, TypeError, ValueError):
            valid = False
        if not valid:
            raise LocalAffordanceReadinessError("activation lease is expired or forged")
        try:
            require_expected_raw_probe_authority(self.authority)
            if _identity(os.fstat(self.repository_root_fd)) != seal.repository_identity:
                raise LocalAffordanceReadinessError("activation repository descriptor changed")
            lexical_root_fd, lexical_identity = _open_root(self.repository)
            os.close(lexical_root_fd)
            if lexical_identity != seal.repository_identity:
                raise LocalAffordanceReadinessError("activation repository path changed")
            current_sources = tuple(
                _read_file_from_root(self.repository_root_fd, item.relative_path)
                for item in self._sources
            )
            if current_sources != self._sources:
                raise LocalAffordanceReadinessError("activation source path identity changed")
            commit, dirty = _git_state(self.repository)
            if dirty or commit != seal.git_commit_sha:
                raise LocalAffordanceReadinessError("activation repository authorization changed")
            if _identity(os.fstat(self.destination_parent_fd)) != seal.destination_parent_identity:
                raise LocalAffordanceReadinessError("activation parent descriptor changed")
        except LocalAffordanceReadinessError:
            raise
        except (
            OSError,
            RawProbeAuthorityError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            secure_fs.SecureFilesystemError,
        ) as exc:
            raise LocalAffordanceReadinessError("activation lease is invalid") from exc
        return self

    def _deactivate(self) -> None:
        object.__setattr__(self, "_active", False)


def capture_local_affordance_readiness(
    repository: str | os.PathLike[str] = ROOT,
    *,
    raw_publication_destination: str | os.PathLike[str],
    for_execution: bool = False,
    expected_git_commit: str | None = None,
) -> LocalAffordanceReadinessSnapshot:
    """Capture committed sources and an absent raw-publication destination."""

    repo = Path(os.path.abspath(repository))
    root_fd, repository_identity = _open_root(repo)
    try:
        sources = tuple(_read_file_from_root(root_fd, path) for path in SOURCE_RELATIVE_PATHS)
    finally:
        os.close(root_fd)
    try:
        authority = build_expected_raw_probe_authority(
            local_affordance_protocol_bytes=sources[0].content,
            development_protocol_bytes=sources[1].content,
            development_tasks_bytes=sources[2].content,
            phase3_evidence_lock_bytes=sources[3].content,
        )
        require_expected_raw_probe_authority(authority)
    except (RawProbeAuthorityError, TypeError, ValueError) as exc:
        raise LocalAffordanceReadinessError("committed raw authority is invalid") from exc
    destination = _absolute(repo, raw_publication_destination, "raw publication destination")
    parent = destination.parent
    if destination.name in {"", ".", ".."} or "/" in destination.name or "\\" in destination.name:
        raise LocalAffordanceReadinessError("raw publication destination must be one direct child")
    parent_fd, ancestors = _open_directory(parent)
    try:
        parent_identity = _identity(os.fstat(parent_fd))
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise LocalAffordanceReadinessError("raw publication destination must be absent")
    finally:
        os.close(parent_fd)
    commit, dirty = _git_state(repo)
    snapshot = LocalAffordanceReadinessSnapshot(
        repository=repo,
        sources=sources,
        authority=authority,
        destination_parent=parent,
        destination_name=destination.name,
        destination_parent_identity=parent_identity,
        destination_ancestors=ancestors,
        repository_identity=repository_identity,
        git_commit_sha=commit,
        git_dirty=dirty,
        _token=_SNAPSHOT_TOKEN,
    )
    snapshot.recheck(
        for_execution=for_execution,
        expected_git_commit=expected_git_commit,
    )
    return snapshot


def require_local_affordance_readiness_snapshot(
    snapshot: LocalAffordanceReadinessSnapshot,
) -> LocalAffordanceReadinessSnapshot:
    return snapshot.require_sealed()


__all__ = [
    "LocalAffordanceActivationLease",
    "LocalAffordanceReadinessError",
    "LocalAffordanceReadinessSnapshot",
    "LocalAffordanceSourceSnapshot",
    "SOURCE_RELATIVE_PATHS",
    "capture_local_affordance_readiness",
    "require_local_affordance_readiness_snapshot",
]
