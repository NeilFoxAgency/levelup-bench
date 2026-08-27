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

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal

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

# The compact outcome-model authority is intentionally kept separate from the
# older Phase 3 representation-ladder authority.  This module only pins the
# development diagnostic inputs; it never opens an environment or result
# namespace.
OUTCOME_MODEL_AUTHORITY_RELATIVE = (
    "configs/milestone6/phase3_outcome_model_artifact_authority.json"
)


def _outcome_model_authority_path(repository: Path, value: str | os.PathLike[str] | None) -> Path:
    canonical = repository / OUTCOME_MODEL_AUTHORITY_RELATIVE
    path = canonical if value is None else _absolute_path(repository, value, "outcome model authority")
    if path != canonical:
        raise OutcomeDiagnosticReadinessError(
            "outcome model authority must use the exact canonical path"
        )
    return path


class OutcomeDiagnosticReadinessError(ValueError):
    """Raised when diagnostic authority or filesystem identities are unsafe."""


_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_LEASE_TOKEN = object()
_SNAPSHOT_TOKEN = object()
DIAGNOSTIC_OUTPUT_ROOT_RELATIVE = "runs/milestone6/phase3-outcome-group-diagnostic"
OutcomeDiagnosticOutputState = Literal["empty", "prepared", "activated"]


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
            "diagnostic output root must be an empty inert namespace"
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


def _inspect_diagnostic_output_state_at(
    output_fd: int,
    output_path: Path,
    output_state: OutcomeDiagnosticOutputState,
    expected_plan: object | None,
) -> object | None:
    if output_state == "empty":
        _require_empty_diagnostic_output_root_fd(output_fd)
        return None
    if output_state not in ("prepared", "activated"):
        raise OutcomeDiagnosticReadinessError("diagnostic output state is invalid")
    try:
        from levelup.experiments.milestone6_phase3_outcome_diagnostic_result_store import (
            OutcomeDiagnosticExpectedPlan,
            inspect_outcome_diagnostic_resume_tree_at,
        )

        if type(expected_plan) is not OutcomeDiagnosticExpectedPlan:
            raise OutcomeDiagnosticReadinessError(
                "diagnostic resume expected plan is missing"
            )
        return inspect_outcome_diagnostic_resume_tree_at(
            output_fd,
            output_path,
            expected_plan,
            output_state=output_state,
        )
    except OutcomeDiagnosticReadinessError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise OutcomeDiagnosticReadinessError(
            f"diagnostic {output_state} output tree failed resume validation"
        ) from exc


def _capture_diagnostic_output_state(
    output_path: Path,
    output_state: OutcomeDiagnosticOutputState,
    expected_plan: object | None,
) -> object | None:
    output_fd, _ancestors = _open_absolute_directory(output_path)
    try:
        return _inspect_diagnostic_output_state_at(
            output_fd, output_path, output_state, expected_plan
        )
    finally:
        os.close(output_fd)


def _build_diagnostic_resume_expected_plan(
    protocol: OutcomeDiagnosticProtocolSnapshot,
) -> object:
    try:
        from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
            bind_validated_outcome_diagnostic_plan,
            build_outcome_group_diagnostic_plan,
        )
        from levelup.experiments.milestone6_phase3_outcome_diagnostic_result_store import (
            build_outcome_diagnostic_expected_plan,
        )

        plan = bind_validated_outcome_diagnostic_plan(
            build_outcome_group_diagnostic_plan(protocol), snapshot=protocol
        )
        return build_outcome_diagnostic_expected_plan(plan, protocol)
    except OutcomeDiagnosticReadinessError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise OutcomeDiagnosticReadinessError(
            "diagnostic resume expected plan failed canonical construction"
        ) from exc


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
    output_state: OutcomeDiagnosticOutputState
    resume_baseline: object | None
    resume_expected_plan: object | None

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
        output_state: OutcomeDiagnosticOutputState,
        resume_baseline: object | None,
        resume_expected_plan: object | None,
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
        object.__setattr__(self, "output_state", output_state)
        object.__setattr__(self, "resume_baseline", resume_baseline)
        object.__setattr__(self, "resume_expected_plan", resume_expected_plan)

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
            observed_output = _inspect_diagnostic_output_state_at(
                output_fd,
                self.output_root,
                self.output_state,
                self.resume_expected_plan,
            )
            if observed_output != self.resume_baseline:
                raise OutcomeDiagnosticReadinessError(
                    "diagnostic output tree changed since readiness capture"
                )
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
        # A held descriptor alone is insufficient: the canonical output path
        # could be renamed or substituted while execution continued writing to
        # an unlinked tree.  Reopen the complete lexical chain without
        # following symlinks and require it to resolve to the pinned root.
        with ExitStack() as path_stack:
            current_output_fd, current_output_ancestors = _open_absolute_directory(
                self.snapshot.output_root, path_stack
            )
            if (
                _identity(os.fstat(current_output_fd))
                != self._expected_output_root_identity
                or current_output_ancestors
                != self.snapshot.output_root_ancestor_identities
            ):
                raise OutcomeDiagnosticReadinessError(
                    "diagnostic output root path changed while active"
                )
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
    output_state: OutcomeDiagnosticOutputState = "empty",
) -> OutcomeDiagnosticReadinessSnapshot:
    """Capture the frozen development diagnostic and validate execution readiness."""

    expected = _require_commit(expected_git_commit)
    repo = _absolute_path(Path.cwd(), repository, "repository")
    _reject_lexical_symlinks(repo, "repository")
    if not repo.is_dir():
        raise OutcomeDiagnosticReadinessError("repository must be an existing directory")
    output = _absolute_path(repo, output_root, "output root")
    canonical_output = repo / DIAGNOSTIC_OUTPUT_ROOT_RELATIVE
    if output != canonical_output:
        raise OutcomeDiagnosticReadinessError(
            "output root must be the canonical inert diagnostic namespace"
        )
    _reject_lexical_symlinks(output, "output root")
    if not output.is_dir():
        raise OutcomeDiagnosticReadinessError("output root must already exist as a directory")
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
    if output_state not in ("empty", "prepared", "activated"):
        raise OutcomeDiagnosticReadinessError("diagnostic output state is invalid")
    resume_expected_plan = (
        None
        if output_state == "empty"
        else _build_diagnostic_resume_expected_plan(protocol)
    )
    resume_baseline = _capture_diagnostic_output_state(
        output, output_state, resume_expected_plan
    )
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
        output_state=output_state,
        resume_baseline=resume_baseline,
        resume_expected_plan=resume_expected_plan,
        _token=_SNAPSHOT_TOKEN,
    )
    snapshot.preflight(expected_git_commit=expected)
    return snapshot


@dataclass(slots=True, init=False)
class OutcomeDiagnosticModelReadinessLease:
    """Descriptor lease for the complete, prepared development model store.

    The lease is deliberately not an activation lease.  It only keeps the
    read-only store descriptors open while readiness is being consumed, so a
    path replacement cannot redirect a later authority recheck.
    """

    _store: object
    _stack: ExitStack
    _owner_ids: tuple[str, ...]
    _identities: object
    _active: bool
    _sealed: bool

    def __init__(
        self,
        store: object,
        stack: ExitStack,
        owner_ids: tuple[str, ...],
        identities: object,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _LEASE_TOKEN:
            raise OutcomeDiagnosticReadinessError("model readiness lease cannot be forged")
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_stack", stack)
        object.__setattr__(self, "_owner_ids", owner_ids)
        object.__setattr__(self, "_identities", identities)
        object.__setattr__(self, "_active", True)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False) and name != "_active":
            raise AttributeError("model readiness lease is immutable while active")
        object.__setattr__(self, name, value)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def store(self) -> object:
        return self._store

    @property
    def identities(self) -> object:
        return self._identities

    def require_active(self) -> "OutcomeDiagnosticModelReadinessLease":
        if not self._active:
            raise OutcomeDiagnosticReadinessError("model readiness lease is no longer active")
        try:
            self._store.recheck()  # type: ignore[attr-defined]
            from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_store import (
                snapshot_outcome_model_store_identities_at,
            )

            current = snapshot_outcome_model_store_identities_at(
                self._store, self._owner_ids
            )
        except Exception as exc:
            if isinstance(exc, OutcomeDiagnosticReadinessError):
                raise
            raise OutcomeDiagnosticReadinessError(
                "prepared outcome model store cannot be rechecked"
            ) from exc
        if current != self._identities:
            raise OutcomeDiagnosticReadinessError(
                "prepared outcome model store identities changed"
            )
        return self

    def close(self) -> None:
        if not self._active:
            return
        object.__setattr__(self, "_active", False)
        self._stack.close()


@dataclass(slots=True, init=False)
class OutcomeDiagnosticModelReadinessSnapshot:
    """Read-only readiness snapshot for the compact model authority and store."""

    base: OutcomeDiagnosticReadinessSnapshot
    authority: object
    authority_file: AuthorityFileSnapshot
    model_store_root: Path
    owner_ids: tuple[str, ...]
    lease: OutcomeDiagnosticModelReadinessLease
    execution_authority_cache: object
    _sealed: bool

    def __init__(
        self,
        base: OutcomeDiagnosticReadinessSnapshot,
        authority: object,
        authority_file: AuthorityFileSnapshot,
        model_store_root: Path,
        owner_ids: tuple[str, ...],
        lease: OutcomeDiagnosticModelReadinessLease,
        execution_authority_cache: object,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _SNAPSHOT_TOKEN:
            raise OutcomeDiagnosticReadinessError(
                "model readiness snapshot requires the canonical capture boundary"
            )
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "authority_file", authority_file)
        object.__setattr__(self, "model_store_root", model_store_root)
        object.__setattr__(self, "owner_ids", owner_ids)
        object.__setattr__(self, "lease", lease)
        object.__setattr__(self, "execution_authority_cache", execution_authority_cache)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("model readiness snapshot is immutable")
        object.__setattr__(self, name, value)

    @property
    def protocol(self) -> OutcomeDiagnosticProtocolSnapshot:
        return self.base.protocol

    @property
    def git_commit_sha(self) -> str:
        return self.base.git_commit_sha

    @property
    def output_root(self) -> Path:
        return self.base.output_root

    @property
    def model_store(self) -> object:
        return self.lease.store

    @property
    def execution_models(self) -> object:
        """Descriptor-pinned execution cache built during readiness capture."""

        return self.execution_authority_cache

    @property
    def authority_bytes(self) -> bytes:
        return self.authority_file.content

    def recheck(self, *, expected_git_commit: str) -> None:
        self.base.recheck(expected_git_commit=expected_git_commit)
        self.lease.require_active()
        current = _read_source(
            self.base.repository, OUTCOME_MODEL_AUTHORITY_RELATIVE
        )
        if current != self.authority_file:
            raise OutcomeDiagnosticReadinessError(
                "outcome model authority bytes or identity changed"
            )
        _validate_outcome_model_authority(
            current.content, self.base.protocol, self.base.repository
        )

    def preflight(self, *, expected_git_commit: str) -> None:
        self.recheck(expected_git_commit=expected_git_commit)

    def close(self) -> None:
        self.lease.close()

    def __enter__(self) -> "OutcomeDiagnosticModelReadinessSnapshot":
        self.lease.require_active()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _validate_outcome_model_authority(
    content: bytes,
    protocol: OutcomeDiagnosticProtocolSnapshot,
    repository: Path,
) -> object:
    try:
        from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
            OutcomeDiagnosticModelArtifactAuthority,
            canonical_outcome_model_artifact_authority_bytes,
            load_outcome_model_artifact_authority_bytes,
            outcome_artifact_store_id,
        )
        from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
            bind_validated_outcome_diagnostic_plan,
            build_outcome_group_diagnostic_plan,
        )

        authority = load_outcome_model_artifact_authority_bytes(content)
        if not isinstance(authority, OutcomeDiagnosticModelArtifactAuthority):
            raise ValueError("typed outcome model authority is required")
        if canonical_outcome_model_artifact_authority_bytes(authority) != content:
            raise ValueError("outcome model authority bytes are not canonical")
        validated_plan = bind_validated_outcome_diagnostic_plan(
            build_outcome_group_diagnostic_plan(protocol),
            snapshot=protocol,
        )
        plan = validated_plan.plan
        if authority.plan_id != plan.plan_id:
            raise ValueError("outcome model authority plan identity differs")
        if authority.plan_parent_commit_sha != plan.parent_commit_sha:
            raise ValueError("outcome model authority plan parent differs")
        if authority.protocol_sha256 != plan.protocol_sha256:
            raise ValueError("outcome model authority protocol identity differs")
        if authority.protocol_self_sha256 != protocol.payload.get("diagnostic_protocol_sha256"):
            raise ValueError("outcome model authority protocol self-hash differs")
        if authority.protocol_file_sha256 != protocol.sha256:
            raise ValueError("outcome model authority protocol file identity differs")
        if authority.artifact_store_id != outcome_artifact_store_id(plan.plan_id):
            raise ValueError("outcome model authority store identity differs")
        if tuple(authority.condition_ids) != tuple(plan.condition_ids):
            raise ValueError("outcome model authority condition universe differs")
        evidence_rows = {}
        for raw in plan.evidence_lineage_rows:
            source = json.loads(raw)
            evidence_rows[(source["family_id"], source["replicate"])] = (
                hashlib.sha256(raw).hexdigest(),
                source["payload_sha256"],
                source["payload_bytes"],
                tuple(source["ordered_training_task_ids"]),
            )
        if {
            (row.heldout_family, row.replicate): (
                row.evidence_row_sha256,
                row.evidence_payload_sha256,
                row.evidence_payload_bytes,
                tuple(row.ordered_training_task_ids),
            )
            for row in authority.evidence
        } != evidence_rows:
            raise ValueError("outcome model authority evidence universe differs")
        if {
            row.view_id: (
                row.condition_id,
                row.heldout_family,
                row.replicate,
                row.evidence_row_sha256,
                row.feature_mask_sha256,
                row.transformation_sha256,
                row.representation_sha256,
            )
            for row in authority.views
        } != {
            view.view_id: (
                view.condition_id,
                view.heldout_family,
                view.replicate,
                evidence_rows[(view.heldout_family, view.replicate)][0],
                view.feature_mask_sha256,
                view.transformation_sha256,
                view.representation_sha256,
            )
            for view in plan.views
        }:
            raise ValueError("outcome model authority view universe differs")
        views_by_id = {view.view_id: view for view in plan.views}
        if {
            row.owner_id: (
                row.view_id,
                row.condition_id,
                row.heldout_family,
                row.fold_id,
                row.replicate,
                row.training_tuple_id,
                row.model_seed,
                row.data_order_seed,
                row.feature_mask_sha256,
                row.transformation_sha256,
                row.representation_sha256,
                row.model_identity_sha256,
            )
            for row in authority.artifacts
        } != {
            owner.owner_id: (
                owner.view_id,
                owner.condition_id,
                owner.heldout_family,
                owner.fold_id,
                owner.replicate,
                owner.training_tuple_id,
                owner.model_seed,
                views_by_id[owner.view_id].data_order_seed,
                owner.feature_mask_sha256,
                owner.transformation_sha256,
                views_by_id[owner.view_id].representation_sha256,
                owner.model_identity_sha256,
            )
            for owner in plan.model_owners
        }:
            raise ValueError("outcome model authority owner universe differs")
        return authority
    except Exception as exc:
        if isinstance(exc, OutcomeDiagnosticReadinessError):
            raise
        raise OutcomeDiagnosticReadinessError(
            "outcome model authority is not canonical development authority"
        ) from exc


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_store_payloads_against_authority(
    store: object,
    authority: object,
    protocol: OutcomeDiagnosticProtocolSnapshot,
) -> tuple[object, dict[str, tuple[object, object, object]]]:
    """Bind every descriptor-read model payload to the compact authority and plan."""

    try:
        from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_authority import (
            validate_outcome_model_preparation_metadata_at,
        )
        from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_store import (
            load_outcome_model_artifact_payload_at,
        )
        from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
            bind_validated_outcome_diagnostic_plan,
            build_outcome_group_diagnostic_plan,
        )

        validated_plan = bind_validated_outcome_diagnostic_plan(
            build_outcome_group_diagnostic_plan(protocol),
            snapshot=protocol,
        )
        validate_outcome_model_preparation_metadata_at(
            store.reader,  # type: ignore[attr-defined]
            validated_plan,
            preparation_git_commit_sha=authority.preparation_git_commit_sha,  # type: ignore[attr-defined]
            preparation_provenance_sha256=authority.preparation_provenance_sha256,  # type: ignore[attr-defined]
            expected_owner_ids=tuple(  # type: ignore[attr-defined]
                sorted(row.owner_id for row in authority.artifacts)
            ),
        )
        plan = validated_plan.plan
        owners = {owner.owner_id: owner for owner in plan.model_owners}
        views = {view.view_id: view for view in plan.views}
        evidence = {
            (row.heldout_family, row.replicate): row
            for row in authority.evidence  # type: ignore[attr-defined]
        }
        payloads: dict[str, tuple[object, object, object]] = {}
        for row in authority.artifacts:  # type: ignore[attr-defined]
            owner = owners[row.owner_id]
            view = views[owner.view_id]
            evidence_row = evidence[(view.heldout_family, view.replicate)]
            record, index, state = load_outcome_model_artifact_payload_at(
                store, row.owner_id
            )
            payloads[row.owner_id] = (record, index, state)
            key = record.key
            consumers = tuple(
                unit for unit in plan.units if unit.model_owner_id == owner.owner_id
            )
            expected_consumer_ids = _canonical_digest(
                [unit.unit_id for unit in consumers]
            )
            expected_seed_lineage = _canonical_digest(
                [
                    {
                        "unit_id": unit.unit_id,
                        "tuple_id": unit.tuple_id,
                        "task_id": unit.task_id,
                        "task_index": unit.task_index,
                        "model_seed": unit.model_seed,
                        "environment_seed": unit.environment_seed,
                        "probe_seed": unit.probe_seed,
                        "search_seed": unit.search_seed,
                        "data_order_seed": unit.data_order_seed,
                    }
                    for unit in consumers
                ]
            )
            if (
                len(consumers) != 24
                or (
                    row.owner_id,
                    row.view_id,
                    row.condition_id,
                    row.heldout_family,
                    row.fold_id,
                    row.replicate,
                    row.training_tuple_id,
                    row.model_seed,
                    row.data_order_seed,
                    row.feature_mask_sha256,
                    row.transformation_sha256,
                    row.representation_sha256,
                    row.model_identity_sha256,
                    row.consumer_unit_ids_sha256,
                    row.consumer_seed_lineage_sha256,
                    row.record_id,
                    row.key_id,
                    row.model_state_sha256,
                )
                != (
                    key.owner_id,
                    key.view_id,
                    key.condition_id,
                    key.heldout_family,
                    key.fold_id,
                    key.replicate,
                    key.training_tuple_id,
                    key.model_seed,
                    key.data_order_seed,
                    key.feature_mask_sha256,
                    key.transformation_sha256,
                    key.representation_sha256,
                    key.model_identity_sha256,
                    key.consumer_unit_ids_sha256,
                    key.consumer_seed_lineage_sha256,
                    record.record_id,
                    key.key_id,
                    key.model_state_sha256,
                )
                or key.plan_id != plan.plan_id
                or key.plan_parent_commit_sha != plan.parent_commit_sha
                or key.protocol_sha256 != plan.protocol_sha256
                or key.protocol_self_sha256
                != protocol.payload.get("diagnostic_protocol_sha256")
                or key.protocol_file_sha256 != protocol.sha256
                or key.view_id != owner.view_id
                or key.condition_id != owner.condition_id
                or key.heldout_family != owner.heldout_family
                or key.fold_id != owner.fold_id
                or key.replicate != owner.replicate
                or key.training_tuple_id != owner.training_tuple_id
                or key.model_seed != owner.model_seed
                or key.data_order_seed != view.data_order_seed
                or key.learning_rate != owner.learning_rate
                or key.training_epochs != owner.training_epochs
                or key.training_accounting.optimizer_steps != owner.training_epochs
                or tuple(key.ordered_training_task_ids) != tuple(view.training_task_ids)
                or tuple(key.ordered_training_task_ids)
                != tuple(evidence_row.ordered_training_task_ids)
                or key.evidence_row_sha256 != evidence_row.evidence_row_sha256
                or key.evidence_payload_sha256 != evidence_row.evidence_payload_sha256
                or key.evidence_payload_bytes != evidence_row.evidence_payload_bytes
                or key.consumer_unit_ids_sha256 != expected_consumer_ids
                or key.consumer_seed_lineage_sha256 != expected_seed_lineage
                or key.consumer_count != 24
                or key.candidate_episodes_per_task != 150
                or key.adaptation_actions_per_task != 2048
                or key.probe_actions_per_task != 64
                or key.maximum_actions_per_candidate_episode != 64
                or key.preparation_git_commit_sha
                != authority.preparation_git_commit_sha  # type: ignore[attr-defined]
                or key.preparation_provenance_sha256
                != authority.preparation_provenance_sha256  # type: ignore[attr-defined]
            ):
                raise ValueError("stored model payload differs from compact authority")
        return validated_plan, payloads
    except Exception as exc:
        if isinstance(exc, OutcomeDiagnosticReadinessError):
            raise
        raise OutcomeDiagnosticReadinessError(
            "prepared model payloads do not match compact authority"
        ) from exc


def capture_outcome_group_diagnostic_model_readiness(
    repository: str | os.PathLike[str] = ROOT,
    *,
    output_root: str | os.PathLike[str],
    expected_git_commit: str,
    model_store_root: str | os.PathLike[str] | None = None,
    authority_path: str | os.PathLike[str] | None = None,
    output_state: OutcomeDiagnosticOutputState = "empty",
) -> OutcomeDiagnosticModelReadinessSnapshot:
    """Capture development-only model authority and a pinned complete store."""

    repo = _absolute_path(Path.cwd(), repository, "repository")
    base = capture_outcome_group_diagnostic_readiness(
        repository=repo,
        output_root=output_root,
        expected_git_commit=expected_git_commit,
        output_state=output_state,
    )
    authority = _outcome_model_authority_path(repo, authority_path)
    _reject_lexical_symlinks(authority, "outcome model authority")
    try:
        authority_file = _read_source(repo, OUTCOME_MODEL_AUTHORITY_RELATIVE)
        typed_authority = _validate_outcome_model_authority(
            authority_file.content, base.protocol, repo
        )
        store_root = repo / "runs" / "milestone6" / typed_authority.artifact_store_id  # type: ignore[attr-defined]
        if model_store_root is not None and _absolute_path(repo, model_store_root, "model store") != store_root:
            raise OutcomeDiagnosticReadinessError(
                "model store root must match the authority artifact_store_id"
            )
        _reject_lexical_symlinks(store_root, "model store")
        from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_store import (
            load_outcome_model_manifest_at,
            open_existing_outcome_model_store,
            snapshot_outcome_model_store_identities_at,
        )

        owner_ids = tuple(sorted(row.owner_id for row in typed_authority.artifacts))  # type: ignore[attr-defined]
        if len(owner_ids) != 240 or len(set(owner_ids)) != 240:
            raise OutcomeDiagnosticReadinessError(
                "outcome model authority must contain exactly 240 owners"
            )
        stack = ExitStack()
        stack.__enter__()
        try:
            store = stack.enter_context(open_existing_outcome_model_store(store_root))
            manifest = load_outcome_model_manifest_at(store)
            expected_manifest_rows = {
                (row.owner_id, row.record_id, row.key_id, row.model_state_sha256)
                for row in typed_authority.artifacts  # type: ignore[attr-defined]
            }
            observed_manifest_rows = {
                (entry.owner_id, entry.record_id, entry.key_id, entry.model_state_sha256)
                for entry in manifest.entries
            }
            if observed_manifest_rows != expected_manifest_rows:
                raise OutcomeDiagnosticReadinessError(
                    "prepared model-store manifest differs from compact authority"
                )
            validated_store = _validate_store_payloads_against_authority(
                store,
                typed_authority,
                base.protocol,
            )
            identities = snapshot_outcome_model_store_identities_at(store, owner_ids)
            lease = OutcomeDiagnosticModelReadinessLease(
                store, stack, owner_ids, identities, _token=_LEASE_TOKEN
            )
            execution_cache: object | None = None
            # The compatibility branch is only reachable in isolated tests
            # which replace the semantic authority validator with a stub.  A
            # real capture always yields the typed authority and therefore
            # must publish the immutable execution cache.
            from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
                OutcomeDiagnosticModelArtifactAuthority,
            )

            if type(typed_authority) is OutcomeDiagnosticModelArtifactAuthority:
                from levelup.experiments.milestone6_phase3_outcome_diagnostic_execution_models import (
                    build_outcome_diagnostic_execution_authority_cache,
                )
                if (
                    not isinstance(validated_store, tuple)
                    or len(validated_store) != 2
                    or not isinstance(validated_store[1], dict)
                ):
                    raise OutcomeDiagnosticReadinessError(
                        "prepared model validation did not retain canonical payloads"
                    )
                validated_plan, payloads = validated_store
                execution_cache = build_outcome_diagnostic_execution_authority_cache(
                    typed_authority,
                    validated_plan,
                    lease,
                    protocol_snapshot=base.protocol,
                    payloads=payloads,
                )
            snapshot = OutcomeDiagnosticModelReadinessSnapshot(
                base,
                typed_authority,
                authority_file,
                store_root,
                owner_ids,
                lease,
                execution_cache,
                _token=_SNAPSHOT_TOKEN,
            )
            snapshot.recheck(expected_git_commit=expected_git_commit)
            return snapshot
        except Exception:
            stack.close()
            raise
    except OutcomeDiagnosticReadinessError:
        raise
    except Exception as exc:
        raise OutcomeDiagnosticReadinessError(
            "complete prepared outcome model store cannot be pinned"
        ) from exc


__all__ = [
    "DIAGNOSTIC_OUTPUT_ROOT_RELATIVE",
    "OUTCOME_MODEL_AUTHORITY_RELATIVE",
    "OutcomeDiagnosticActivationReadinessLease",
    "OutcomeDiagnosticModelReadinessLease",
    "OutcomeDiagnosticModelReadinessSnapshot",
    "OutcomeDiagnosticOutputState",
    "OutcomeDiagnosticReadinessError",
    "OutcomeDiagnosticReadinessSnapshot",
    "capture_outcome_group_diagnostic_readiness",
    "capture_outcome_group_diagnostic_model_readiness",
]
