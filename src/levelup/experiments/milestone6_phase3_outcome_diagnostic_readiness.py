"""Descriptor-safe readiness for the Phase 3 outcome-group diagnostic.

This is an additive boundary around the already frozen Phase 3 authorities.  It
captures the diagnostic protocol, every authority file required by that
protocol (including the published Phase 3 selection and model-preparation
lineage), and the identities of the model namespace and execution output root.
It does not create a store, prepare an activation, execute a task, or inspect a
result.  A later executor may consume the held descriptor lease only after a
clean, explicitly authorised commit has been checked again.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    PHASE3_OUTCOME_DIAGNOSTIC_PROTOCOL_PATH,
    OutcomeDiagnosticProtocolError,
    OutcomeDiagnosticProtocolSnapshot,
    load_outcome_group_diagnostic_protocol,
)
from levelup.experiments.milestone6_phase3_protocol import ROOT
from levelup.experiments.milestone6_phase3_readiness import (
    AuthorityDirectorySnapshot,
    AuthorityFileSnapshot,
    Phase3ReadinessError,
    _hold_directory_from_root,
    _hold_file_from_root,
    _identity,
    _read_directory,
    _read_source,
    _relative,
    capture_phase3_readiness,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes


class OutcomeDiagnosticReadinessError(ValueError):
    """Raised when diagnostic authority or filesystem identities are unsafe."""


_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_LEASE_TOKEN = object()
_SNAPSHOT_TOKEN = object()
DIAGNOSTIC_OUTPUT_ROOT_RELATIVE = "runs/milestone6/phase3-outcome-group-diagnostic"


def _require_commit(value: str | None) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise OutcomeDiagnosticReadinessError(
            "expected_git_commit must be 40-64 lowercase hexadecimal characters"
        )
    return value


def _absolute_path(repository: Path, value: str | os.PathLike[str], label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repository / path
    return Path(os.path.abspath(path))


def _reject_lexical_symlinks(path: Path, label: str) -> None:
    """Reject a symlink at the path or any lexical ancestor before resolution."""

    absolute = Path(os.path.abspath(path))
    chain = list(reversed(absolute.parents)) + [absolute]
    for component in chain:
        try:
            observed = component.lstat()
        except OSError as exc:
            raise OutcomeDiagnosticReadinessError(
                f"{label} path is unavailable: {component}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode):
            raise OutcomeDiagnosticReadinessError(
                f"{label} path contains a lexical symlink: {component}"
            )


def _open_absolute_directory(
    path: Path, stack: ExitStack | None = None
) -> tuple[int, tuple[tuple[str, tuple[int, int]], ...]]:
    """Open every component with O_NOFOLLOW and retain ancestor identities."""

    _reject_lexical_symlinks(path, "directory")
    absolute = Path(os.path.abspath(path))
    current = secure_fs.open_directory_chain(Path(absolute.anchor))
    opened: list[int] = [current]
    ancestors: list[tuple[str, tuple[int, int]]] = [
        (str(Path(absolute.anchor)), _identity(os.fstat(current)))
    ]
    try:
        for component in absolute.parts[1:]:
            child = secure_fs.open_child_directory(current, component)
            opened.append(child)
            current = child
            ancestors.append(
                (str(Path(*absolute.parts[: len(ancestors) + 1])), _identity(os.fstat(child)))
            )
        if stack is not None:
            for fd in opened:
                stack.callback(os.close, fd)
            return current, tuple(ancestors)
        for fd in opened[:-1]:
            os.close(fd)
        return current, tuple(ancestors)
    except (OSError, RuntimeError, TypeError, ValueError, secure_fs.SecureFilesystemError) as exc:
        for fd in reversed(opened):
            try:
                os.close(fd)
            except OSError:
                pass
        raise OutcomeDiagnosticReadinessError(f"cannot securely open {path}") from exc


def _directory_snapshot(
    path: Path, label: str
) -> tuple[tuple[int, int], tuple[tuple[str, tuple[int, int]], ...]]:
    _reject_lexical_symlinks(path, label)
    fd, ancestors = _open_absolute_directory(path)
    try:
        return _identity(os.fstat(fd)), ancestors
    finally:
        os.close(fd)


def _require_empty_diagnostic_output_root_fd(fd: int) -> None:
    try:
        entries = secure_fs.strict_regular_entries(fd)
    except secure_fs.SecureFilesystemError as exc:
        raise OutcomeDiagnosticReadinessError(
            "diagnostic output root contains a non-regular or symlink child"
        ) from exc
    if entries:
        raise OutcomeDiagnosticReadinessError(
            "diagnostic output root must be an empty inert namespace"
        )


def _require_empty_diagnostic_output_root(path: Path) -> None:
    fd, _ = _open_absolute_directory(path)
    try:
        _require_empty_diagnostic_output_root_fd(fd)
    finally:
        os.close(fd)


def _authority_paths(protocol: OutcomeDiagnosticProtocolSnapshot) -> tuple[str, ...]:
    authority = protocol.payload.get("authority")
    if not isinstance(authority, Mapping):
        raise OutcomeDiagnosticReadinessError("diagnostic authority mapping is missing")
    paths: list[str] = []
    for name, source in authority.items():
        if not isinstance(source, Mapping):
            raise OutcomeDiagnosticReadinessError(f"diagnostic authority {name} is malformed")
        try:
            paths.append(_relative(source["path"]))
        except (KeyError, TypeError, ValueError, Phase3ReadinessError) as exc:
            raise OutcomeDiagnosticReadinessError(
                f"diagnostic authority {name} path is unsafe"
            ) from exc
    return tuple(dict.fromkeys(paths))


def _merge_files(
    *groups: Iterator[AuthorityFileSnapshot], diagnostic: AuthorityFileSnapshot
) -> tuple[AuthorityFileSnapshot, ...]:
    by_path: dict[str, AuthorityFileSnapshot] = {}

    def add(item: AuthorityFileSnapshot) -> None:
        previous = by_path.get(item.relative_path)
        if previous is not None and previous != item:
            raise OutcomeDiagnosticReadinessError(
                f"conflicting authority snapshots for {item.relative_path}"
            )
        by_path[item.relative_path] = item

    add(diagnostic)
    for group in groups:
        for item in group:
            add(item)
    return tuple(by_path[path] for path in sorted(by_path))


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_thaw(item) for item in value]
    return value


def _freeze_protocol(
    protocol: OutcomeDiagnosticProtocolSnapshot,
) -> OutcomeDiagnosticProtocolSnapshot:
    payload = _freeze(protocol.payload)
    if not isinstance(payload, Mapping):
        raise OutcomeDiagnosticReadinessError("diagnostic protocol payload is not a mapping")
    return replace(protocol, payload=payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True, init=False)
class OutcomeDiagnosticReadinessSnapshot:
    """Immutable bytes and identities required by a future diagnostic run."""

    repository: Path
    output_root: Path
    protocol: OutcomeDiagnosticProtocolSnapshot
    files: tuple[AuthorityFileSnapshot, ...]
    directories: tuple[AuthorityDirectorySnapshot, ...]
    repository_identity: tuple[int, int]
    repository_ancestor_identities: tuple[tuple[str, tuple[int, int]], ...]
    output_root_identity: tuple[int, int]
    output_root_ancestor_identities: tuple[tuple[str, tuple[int, int]], ...]
    git_commit_sha: str
    git_dirty: bool
    source_result_lock_commit_sha: str

    def __init__(
        self,
        repository: Path,
        output_root: Path,
        protocol: OutcomeDiagnosticProtocolSnapshot,
        files: tuple[AuthorityFileSnapshot, ...],
        directories: tuple[AuthorityDirectorySnapshot, ...],
        repository_identity: tuple[int, int],
        repository_ancestor_identities: tuple[tuple[str, tuple[int, int]], ...],
        output_root_identity: tuple[int, int],
        output_root_ancestor_identities: tuple[tuple[str, tuple[int, int]], ...],
        git_commit_sha: str,
        git_dirty: bool,
        source_result_lock_commit_sha: str,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _SNAPSHOT_TOKEN:
            raise OutcomeDiagnosticReadinessError(
                "diagnostic readiness snapshot requires the canonical capture boundary"
            )
        object.__setattr__(self, "repository", repository)
        object.__setattr__(self, "output_root", output_root)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "directories", directories)
        object.__setattr__(self, "repository_identity", repository_identity)
        object.__setattr__(self, "repository_ancestor_identities", repository_ancestor_identities)
        object.__setattr__(self, "output_root_identity", output_root_identity)
        object.__setattr__(self, "output_root_ancestor_identities", output_root_ancestor_identities)
        object.__setattr__(self, "git_commit_sha", git_commit_sha)
        object.__setattr__(self, "git_dirty", git_dirty)
        object.__setattr__(self, "source_result_lock_commit_sha", source_result_lock_commit_sha)

    @property
    def files_by_path(self) -> Mapping[str, AuthorityFileSnapshot]:
        return {item.relative_path: item for item in self.files}

    @property
    def directories_by_path(self) -> Mapping[str, AuthorityDirectorySnapshot]:
        return {item.relative_path: item for item in self.directories}

    def recheck(self, *, expected_git_commit: str) -> None:
        expected = _require_commit(expected_git_commit)
        _reject_lexical_symlinks(self.repository, "repository")
        _reject_lexical_symlinks(self.output_root, "output root")
        try:
            repo_id, repo_ancestors = _directory_snapshot(self.repository, "repository")
            output_id, output_ancestors = _directory_snapshot(self.output_root, "output root")
        except OutcomeDiagnosticReadinessError:
            raise
        if (
            repo_id != self.repository_identity
            or repo_ancestors != self.repository_ancestor_identities
        ):
            raise OutcomeDiagnosticReadinessError("repository identity changed")
        if (
            output_id != self.output_root_identity
            or output_ancestors != self.output_root_ancestor_identities
        ):
            raise OutcomeDiagnosticReadinessError("output root identity changed")
        for expected_file in self.files:
            try:
                current = _read_source(self.repository, expected_file.relative_path)
            except (Phase3ReadinessError, OSError, RuntimeError, ValueError) as exc:
                raise OutcomeDiagnosticReadinessError(
                    f"authority source cannot be reopened: {expected_file.relative_path}"
                ) from exc
            if (
                current.content != expected_file.content
                or current.sha256 != expected_file.sha256
                or current.parent_identity != expected_file.parent_identity
                or current.file_identity != expected_file.file_identity
                or current.ancestor_identities != expected_file.ancestor_identities
            ):
                raise OutcomeDiagnosticReadinessError(
                    f"authority source changed: {expected_file.relative_path}"
                )
        for expected_directory in self.directories:
            try:
                current = _read_directory(self.repository, expected_directory.relative_path)
            except (Phase3ReadinessError, OSError, RuntimeError, ValueError) as exc:
                raise OutcomeDiagnosticReadinessError(
                    f"model namespace cannot be reopened: {expected_directory.relative_path}"
                ) from exc
            if (
                current.identity != expected_directory.identity
                or current.ancestor_identities != expected_directory.ancestor_identities
            ):
                raise OutcomeDiagnosticReadinessError(
                    f"model namespace changed: {expected_directory.relative_path}"
                )
        from levelup.experiments.milestone6_phase3_readiness import _git_state

        commit, dirty = _git_state(self.repository)
        if commit != self.git_commit_sha or dirty != self.git_dirty:
            raise OutcomeDiagnosticReadinessError(
                "repository provenance changed since readiness capture"
            )
        if dirty or commit != expected:
            raise OutcomeDiagnosticReadinessError(
                "execution requires a clean explicitly authorised commit"
            )
        try:
            current_protocol = load_outcome_group_diagnostic_protocol(
                self.protocol.path, repository=self.repository
            )
            canonical_phase3 = capture_phase3_readiness(repository=self.repository)
        except OutcomeDiagnosticProtocolError as exc:
            raise OutcomeDiagnosticReadinessError("diagnostic protocol authority changed") from exc
        except (Phase3ReadinessError, OSError, RuntimeError, ValueError) as exc:
            raise OutcomeDiagnosticReadinessError("Phase 3 authority manifest changed") from exc
        if (
            current_protocol.path != self.protocol.path
            or current_protocol.content != self.protocol.content
            or current_protocol.sha256 != self.protocol.sha256
            or canonical_json_bytes(_thaw(current_protocol.payload))
            != canonical_json_bytes(_thaw(self.protocol.payload))
            or current_protocol.authority_bytes != self.protocol.authority_bytes
        ):
            raise OutcomeDiagnosticReadinessError("diagnostic protocol bytes changed")
        expected_paths = {item.relative_path for item in canonical_phase3.files}
        expected_paths.update(_authority_paths(current_protocol))
        expected_paths.add(current_protocol.path.relative_to(self.repository).as_posix())
        expected_paths.add("configs/milestone6/phase3_development_selection.json")
        expected_file_order = tuple(sorted(expected_paths))
        actual_paths = [item.relative_path for item in self.files]
        if tuple(actual_paths) != expected_file_order:
            raise OutcomeDiagnosticReadinessError("diagnostic authority file manifest changed")
        expected_directories = tuple(item.relative_path for item in canonical_phase3.directories)
        actual_directories = [item.relative_path for item in self.directories]
        if tuple(actual_directories) != expected_directories:
            raise OutcomeDiagnosticReadinessError("diagnostic model namespace manifest changed")

    def preflight(self, *, expected_git_commit: str) -> None:
        self.recheck(expected_git_commit=expected_git_commit)

    @contextmanager
    def hold_for_activation(
        self, *, expected_git_commit: str
    ) -> Iterator["OutcomeDiagnosticActivationReadinessLease"]:
        self.preflight(expected_git_commit=expected_git_commit)
        stack = ExitStack()
        stack.__enter__()
        lease: OutcomeDiagnosticActivationReadinessLease | None = None
        try:
            repo_fd, repo_ancestors = _open_absolute_directory(self.repository, stack)
            output_fd, output_ancestors = _open_absolute_directory(self.output_root, stack)
            if _identity(os.fstat(repo_fd)) != self.repository_identity:
                raise OutcomeDiagnosticReadinessError("held repository identity changed")
            if _identity(os.fstat(output_fd)) != self.output_root_identity:
                raise OutcomeDiagnosticReadinessError("held output root identity changed")
            if repo_ancestors != self.repository_ancestor_identities:
                raise OutcomeDiagnosticReadinessError("held repository ancestors changed")
            if output_ancestors != self.output_root_ancestor_identities:
                raise OutcomeDiagnosticReadinessError("held output root ancestors changed")
            _require_empty_diagnostic_output_root_fd(output_fd)
            file_descriptors = {
                item.relative_path: _hold_file_from_root(repo_fd, item, stack)
                for item in self.files
            }
            directory_descriptors = {
                item.relative_path: _hold_directory_from_root(repo_fd, item, stack)
                for item in self.directories
            }
            lease = OutcomeDiagnosticActivationReadinessLease(
                self,
                repo_fd,
                output_fd,
                file_descriptors,
                directory_descriptors,
                stack,
                _token=_LEASE_TOKEN,
            )
            self.preflight(expected_git_commit=expected_git_commit)
            yield lease.require_active()
        except OutcomeDiagnosticReadinessError:
            raise
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            secure_fs.SecureFilesystemError,
        ) as exc:
            raise OutcomeDiagnosticReadinessError(
                "cannot hold diagnostic readiness descriptors"
            ) from exc
        finally:
            if lease is not None:
                lease.close()
            else:
                stack.close()
            self.preflight(expected_git_commit=expected_git_commit)


@dataclass(slots=True, init=False)
class OutcomeDiagnosticActivationReadinessLease:
    """Live descriptor lease; construction is intentionally unforgeable."""

    _snapshot: OutcomeDiagnosticReadinessSnapshot
    _repository_fd: int
    _output_root_fd: int
    _file_descriptors: Mapping[str, int]
    _directory_descriptors: Mapping[str, int]
    _stack: ExitStack
    _active: bool
    _sealed: bool
    _expected_repository_identity: tuple[int, int]
    _expected_output_root_identity: tuple[int, int]
    _expected_file_identities: Mapping[str, tuple[int, int]]
    _expected_directory_identities: Mapping[str, tuple[int, int]]
    _expected_snapshot: OutcomeDiagnosticReadinessSnapshot

    def __init__(
        self,
        snapshot: OutcomeDiagnosticReadinessSnapshot,
        repository_fd: int,
        output_root_fd: int,
        file_descriptors: Mapping[str, int],
        directory_descriptors: Mapping[str, int],
        stack: ExitStack,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _LEASE_TOKEN:
            raise OutcomeDiagnosticReadinessError("diagnostic activation lease cannot be forged")
        object.__setattr__(self, "_snapshot", snapshot)
        object.__setattr__(self, "_repository_fd", repository_fd)
        object.__setattr__(self, "_output_root_fd", output_root_fd)
        object.__setattr__(self, "_file_descriptors", MappingProxyType(dict(file_descriptors)))
        object.__setattr__(
            self, "_directory_descriptors", MappingProxyType(dict(directory_descriptors))
        )
        self._stack = stack
        self._active = True
        self._expected_repository_identity = snapshot.repository_identity
        self._expected_output_root_identity = snapshot.output_root_identity
        self._expected_file_identities = MappingProxyType(
            {item.relative_path: item.file_identity for item in snapshot.files}
        )
        self._expected_directory_identities = MappingProxyType(
            {item.relative_path: item.identity for item in snapshot.directories}
        )
        self._expected_snapshot = snapshot
        self._sealed = True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False) and name != "_active":
            raise AttributeError("diagnostic activation lease is immutable while active")
        object.__setattr__(self, name, value)

    @property
    def snapshot(self) -> OutcomeDiagnosticReadinessSnapshot:
        return self._snapshot

    @property
    def repository_fd(self) -> int:
        return self._repository_fd

    @property
    def output_root_fd(self) -> int:
        return self._output_root_fd

    @property
    def file_descriptors(self) -> Mapping[str, int]:
        return self._file_descriptors

    @property
    def directory_descriptors(self) -> Mapping[str, int]:
        return self._directory_descriptors

    @property
    def active(self) -> bool:
        return self._active

    def require_active(self) -> "OutcomeDiagnosticActivationReadinessLease":
        if not self._active:
            raise OutcomeDiagnosticReadinessError("diagnostic activation lease is no longer active")
        if self._snapshot is not self._expected_snapshot:
            raise OutcomeDiagnosticReadinessError("diagnostic readiness snapshot was reassigned")
        expected_files = self._expected_file_identities
        expected_directories = self._expected_directory_identities
        if set(self.file_descriptors) != set(expected_files):
            raise OutcomeDiagnosticReadinessError(
                "diagnostic readiness file descriptor keys changed"
            )
        if set(self.directory_descriptors) != set(expected_directories):
            raise OutcomeDiagnosticReadinessError(
                "diagnostic readiness directory descriptor keys changed"
            )
        checks = (
            (self.repository_fd, self._expected_repository_identity),
            (self.output_root_fd, self._expected_output_root_identity),
            *((self.file_descriptors[path], expected_files[path]) for path in expected_files),
            *(
                (self.directory_descriptors[path], expected_directories[path])
                for path in expected_directories
            ),
        )
        for fd, expected_identity in checks:
            try:
                if _identity(os.fstat(fd)) != expected_identity:
                    raise OutcomeDiagnosticReadinessError(
                        "diagnostic readiness descriptor identity changed"
                    )
            except OSError as exc:
                raise OutcomeDiagnosticReadinessError(
                    "diagnostic readiness descriptor closed unexpectedly"
                ) from exc
        return self

    def close(self) -> None:
        if not self._active:
            return
        self._active = False
        self._stack.close()


def capture_outcome_group_diagnostic_readiness(
    repository: str | os.PathLike[str] = ROOT,
    *,
    output_root: str | os.PathLike[str],
    expected_git_commit: str,
) -> OutcomeDiagnosticReadinessSnapshot:
    """Capture the frozen development diagnostic and validate execution readiness."""

    expected = _require_commit(expected_git_commit)
    repo = _absolute_path(Path.cwd(), repository, "repository")
    _reject_lexical_symlinks(repo, "repository")
    if not repo.is_dir():
        raise OutcomeDiagnosticReadinessError("repository must be an existing directory")
    output = _absolute_path(repo, output_root, "output root")
    _reject_lexical_symlinks(output, "output root")
    if not output.is_dir():
        raise OutcomeDiagnosticReadinessError("output root must already exist as a directory")
    canonical_output = repo / DIAGNOSTIC_OUTPUT_ROOT_RELATIVE
    if output != canonical_output:
        raise OutcomeDiagnosticReadinessError(
            "output root must be the canonical inert diagnostic namespace"
        )
    _require_empty_diagnostic_output_root(output)
    try:
        protocol = load_outcome_group_diagnostic_protocol(
            PHASE3_OUTCOME_DIAGNOSTIC_PROTOCOL_PATH,
            repository=repo,
        )
        phase3 = capture_phase3_readiness(repository=repo)
    except (
        OutcomeDiagnosticProtocolError,
        Phase3ReadinessError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        if isinstance(exc, OutcomeDiagnosticReadinessError):
            raise
        raise OutcomeDiagnosticReadinessError(
            "frozen diagnostic authorities cannot be captured"
        ) from exc
    boundary = protocol.payload.get("execution_boundary")
    if not isinstance(boundary, Mapping) or any(
        boundary.get(field) is not False
        for field in (
            "final_family_access",
            "final_method_selection",
            "advancement_to_paired_objectives",
        )
    ):
        raise OutcomeDiagnosticReadinessError(
            "diagnostic authority permits final-family or final-method access"
        )
    diagnostic = _read_source(repo, protocol.path.relative_to(repo).as_posix())
    direct_paths = _authority_paths(protocol)
    direct = tuple(_read_source(repo, path) for path in direct_paths)
    selection_path = "configs/milestone6/phase3_development_selection.json"
    if selection_path not in {item.relative_path for item in phase3.files}:
        direct += (_read_source(repo, selection_path),)
    files = _merge_files(iter(phase3.files), iter(direct), diagnostic=diagnostic)
    repo_id, repo_ancestors = _directory_snapshot(repo, "repository")
    output_id, output_ancestors = _directory_snapshot(output, "output root")
    from levelup.experiments.milestone6_phase3_readiness import _git_state

    commit, dirty = _git_state(repo)
    if dirty or commit != expected:
        raise OutcomeDiagnosticReadinessError("readiness requires a clean exact authorised commit")
    snapshot = OutcomeDiagnosticReadinessSnapshot(
        repository=repo,
        output_root=output,
        protocol=_freeze_protocol(protocol),
        files=files,
        directories=phase3.directories,
        repository_identity=repo_id,
        repository_ancestor_identities=repo_ancestors,
        output_root_identity=output_id,
        output_root_ancestor_identities=output_ancestors,
        git_commit_sha=commit,
        git_dirty=dirty,
        source_result_lock_commit_sha=str(
            protocol.payload["freeze_record"]["source_result_lock_commit_sha"]
        ),
        _token=_SNAPSHOT_TOKEN,
    )
    snapshot.preflight(expected_git_commit=expected)
    return snapshot


__all__ = [
    "DIAGNOSTIC_OUTPUT_ROOT_RELATIVE",
    "OutcomeDiagnosticActivationReadinessLease",
    "OutcomeDiagnosticReadinessError",
    "OutcomeDiagnosticReadinessSnapshot",
    "capture_outcome_group_diagnostic_readiness",
]
