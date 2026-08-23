"""Immutable, development-only Phase 3 authority preflight.

This module is deliberately a *read-only* boundary.  It does not create a
``RunStore``, activate a namespace, execute a task, read a result, or apply a
device policy.  It records the exact bytes and filesystem identities of every
authority input used by the Phase 3 executor.  A caller can then recheck the
same object immediately before a future activation; both byte drift and an
inode replacement (including a same-byte replacement) fail closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from levelup.experiments.milestone6_phase3_model_artifacts import (
    ARTIFACTS_DIR,
    COSTS_DIR,
    KEYS_DIR,
)
from levelup.experiments.milestone6_phase3_model_authority import (
    load_phase3_model_artifact_authority_bytes,
)
from levelup.experiments.milestone6_phase3_protocol import ROOT
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes


class Phase3ReadinessError(ValueError):
    """Raised when a Phase 3 authority or its filesystem identity is unsafe."""


PHASE3_PLAN_ID = "2457657f77e3ef67708f1abc195bfa4ad31c554cfd314526e1bb26113cc4c9d1"
PHASE3_MODEL_AUTHORITY_SHA256 = (
    "8771eb52433faf15d6e5e935902a5c935526ec0e6b8e34621c3d6a922aea1a52"
)
PHASE3_MODEL_AUTHORITY_FILE_SHA256 = (
    "eecd68707e2cdfa34e9e9b30f787fd17b87ae767db63b659944e420cb7255388"
)
PHASE3_PROTOCOL_RELATIVE = "configs/milestone6/phase3_representation_ladder.json"
PHASE3_PLAN_LOCK_RELATIVE = "configs/milestone6/phase3_plan_lock.json"
PHASE3_ANCHOR_RELATIVE = "configs/milestone6/phase3_anchor_manifest.json"
PHASE3_EVIDENCE_RELATIVE = "configs/milestone6/phase3_evidence_lock.json"
PHASE3_MODEL_AUTHORITY_RELATIVE = "configs/milestone6/phase3_model_artifact_authority.json"
PREPARATION_PROVENANCE_NAME = "phase3-model-preparation-provenance.json"
PREPARATION_PROGRESS_NAME = "phase3-model-preparation-progress.json"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _identity(value: os.stat_result) -> tuple[int, int]:
    if not stat.S_ISREG(value.st_mode) and not stat.S_ISDIR(value.st_mode):
        raise Phase3ReadinessError("authority path contains a non-regular entry")
    return int(value.st_dev), int(value.st_ino)


def _relative(value: str | os.PathLike[str]) -> str:
    pure = PurePosixPath(str(value))
    if pure.is_absolute() or not pure.parts or "." in pure.parts or ".." in pure.parts:
        raise Phase3ReadinessError("authority source path must be repository-relative")
    if any(not part or "\\" in part or "\x00" in part for part in pure.parts):
        raise Phase3ReadinessError("authority source path contains an unsafe component")
    return "/".join(pure.parts)


@dataclass(frozen=True, slots=True)
class AuthorityFileSnapshot:
    """Exact bytes and descriptor identities for one authority file."""

    relative_path: str
    content: bytes
    sha256: str
    parent_identity: tuple[int, int]
    file_identity: tuple[int, int]
    ancestor_identities: tuple[tuple[str, tuple[int, int]], ...]

    @property
    def bytes(self) -> bytes:
        """Compatibility spelling for callers that refer to retained bytes."""

        return self.content


@dataclass(frozen=True, slots=True)
class AuthorityDirectorySnapshot:
    """Exact identity of one model-store directory and its ancestor chain."""

    relative_path: str
    identity: tuple[int, int]
    ancestor_identities: tuple[tuple[str, tuple[int, int]], ...]


_ACTIVATION_LEASE_TOKEN = object()


@dataclass(slots=True, init=False)
class Phase3ActivationReadinessLease:
    """Unforgeable lease whose authority descriptors remain open and pinned."""

    snapshot: "Phase3ReadinessSnapshot"
    file_descriptors: Mapping[str, int]
    directory_descriptors: Mapping[str, int]
    _active: bool

    def __init__(
        self,
        snapshot: "Phase3ReadinessSnapshot",
        file_descriptors: Mapping[str, int],
        directory_descriptors: Mapping[str, int],
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _ACTIVATION_LEASE_TOKEN:
            raise Phase3ReadinessError(
                "activation readiness lease requires the canonical held-descriptor context"
            )
        self.snapshot = snapshot
        self.file_descriptors = dict(file_descriptors)
        self.directory_descriptors = dict(directory_descriptors)
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def require_active(self) -> "Phase3ActivationReadinessLease":
        if not self._active:
            raise Phase3ReadinessError("activation readiness lease is no longer active")
        for fd in (*self.file_descriptors.values(), *self.directory_descriptors.values()):
            try:
                os.fstat(fd)
            except OSError as exc:
                raise Phase3ReadinessError(
                    "activation readiness descriptor closed unexpectedly"
                ) from exc
        return self

    def _deactivate(self) -> None:
        self._active = False


@contextmanager
def _open_source(repository: Path, relative_path: str) -> Iterator[AuthorityFileSnapshot]:
    """Read one source under a held descriptor chain and retain its identities."""

    relative = _relative(relative_path)
    components = relative.split("/")
    root_fd: int | None = None
    parent_fd: int | None = None
    opened: list[int] = []
    try:
        root_fd = secure_fs.open_directory_chain(repository)
        root_stat = os.fstat(root_fd)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise Phase3ReadinessError("authority repository is not a directory")
        parent_fd = root_fd
        ancestors: list[tuple[str, tuple[int, int]]] = [("", _identity(root_stat))]
        for index, component in enumerate(components[:-1]):
            child_fd = secure_fs.open_child_directory(parent_fd, component)
            opened.append(child_fd)
            child_stat = os.fstat(child_fd)
            ancestors.append(("/".join(components[: index + 1]), _identity(child_stat)))
            parent_fd = child_fd
        parent_stat_before = os.fstat(parent_fd)
        parent_id = _identity(parent_stat_before)
        if not stat.S_ISDIR(parent_stat_before.st_mode):
            raise Phase3ReadinessError("authority source parent is not a directory")
        with secure_fs.open_regular_file_at(parent_fd, components[-1]) as file_fd:
            before = os.fstat(file_fd)
            content = b""
            chunks: list[bytes] = []
            while True:
                chunk = os.read(file_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            content = b"".join(chunks)
            after = os.fstat(file_fd)
            if _identity(before) != _identity(after):
                raise Phase3ReadinessError("authority file changed during read")
            parent_after = os.fstat(parent_fd)
            if _identity(parent_stat_before) != _identity(parent_after):
                raise Phase3ReadinessError("authority parent changed during read")
            yield AuthorityFileSnapshot(
                relative_path=relative,
                content=content,
                sha256=_sha256(content),
                parent_identity=parent_id,
                file_identity=_identity(before),
                ancestor_identities=tuple(ancestors),
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, Phase3ReadinessError):
            raise
        raise Phase3ReadinessError(f"cannot safely read authority source: {relative}") from exc
    finally:
        for fd in reversed(opened):
            os.close(fd)
        if root_fd is not None:
            os.close(root_fd)


def _read_source(repository: Path, relative_path: str) -> AuthorityFileSnapshot:
    with _open_source(repository, relative_path) as snapshot:
        return snapshot


def _read_directory(repository: Path, relative_path: str) -> AuthorityDirectorySnapshot:
    """Open one repository-relative directory through one held descriptor chain."""

    relative = _relative(relative_path)
    components = relative.split("/")
    root_fd: int | None = None
    opened: list[int] = []
    try:
        root_fd = secure_fs.open_directory_chain(repository)
        root_stat = os.fstat(root_fd)
        ancestors: list[tuple[str, tuple[int, int]]] = [("", _identity(root_stat))]
        parent_fd = root_fd
        for index, component in enumerate(components):
            child_fd = secure_fs.open_child_directory(parent_fd, component)
            opened.append(child_fd)
            child_stat = os.fstat(child_fd)
            identity = _identity(child_stat)
            if index < len(components) - 1:
                ancestors.append(("/".join(components[: index + 1]), identity))
            parent_fd = child_fd
        return AuthorityDirectorySnapshot(
            relative_path=relative,
            identity=identity,
            ancestor_identities=tuple(ancestors),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, Phase3ReadinessError):
            raise
        raise Phase3ReadinessError(
            f"cannot safely open authority directory: {relative}"
        ) from exc
    finally:
        for fd in reversed(opened):
            os.close(fd)
        if root_fd is not None:
            os.close(root_fd)


def _hold_file_from_root(
    root_fd: int,
    expected: AuthorityFileSnapshot,
    stack: ExitStack,
) -> int:
    components = expected.relative_path.split("/")
    parent_fd = root_fd
    ancestors: list[tuple[str, tuple[int, int]]] = [
        ("", _identity(os.fstat(root_fd)))
    ]
    for index, component in enumerate(components[:-1]):
        child_fd = secure_fs.open_child_directory(parent_fd, component)
        stack.callback(os.close, child_fd)
        ancestors.append(
            ("/".join(components[: index + 1]), _identity(os.fstat(child_fd)))
        )
        parent_fd = child_fd
    file_fd = os.open(
        components[-1],
        os.O_RDONLY | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    stack.callback(os.close, file_fd)
    before = os.fstat(file_fd)
    if not stat.S_ISREG(before.st_mode):
        raise Phase3ReadinessError("held authority source is not a regular file")
    chunks: list[bytes] = []
    while True:
        chunk = os.read(file_fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(file_fd)
    if (
        _identity(before) != expected.file_identity
        or _identity(after) != expected.file_identity
        or _identity(os.fstat(parent_fd)) != expected.parent_identity
        or tuple(ancestors) != expected.ancestor_identities
        or b"".join(chunks) != expected.content
    ):
        raise Phase3ReadinessError(
            f"held authority source differs: {expected.relative_path}"
        )
    os.lseek(file_fd, 0, os.SEEK_SET)
    return file_fd


def _hold_directory_from_root(
    root_fd: int,
    expected: AuthorityDirectorySnapshot,
    stack: ExitStack,
) -> int:
    components = expected.relative_path.split("/")
    parent_fd = root_fd
    ancestors: list[tuple[str, tuple[int, int]]] = [
        ("", _identity(os.fstat(root_fd)))
    ]
    for index, component in enumerate(components):
        child_fd = secure_fs.open_child_directory(parent_fd, component)
        stack.callback(os.close, child_fd)
        identity = _identity(os.fstat(child_fd))
        if index < len(components) - 1:
            ancestors.append(("/".join(components[: index + 1]), identity))
        parent_fd = child_fd
    if identity != expected.identity or tuple(ancestors) != expected.ancestor_identities:
        raise Phase3ReadinessError(
            f"held authority directory differs: {expected.relative_path}"
        )
    return parent_fd


def _json(
    snapshot: AuthorityFileSnapshot, label: str, *, canonical: bool = False
) -> dict[str, Any]:
    try:
        value = json.loads(snapshot.content)
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase3ReadinessError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict) or (canonical and canonical_json_bytes(value) != snapshot.content):
        raise Phase3ReadinessError(f"{label} is not canonical JSON")
    return value


def _require_self_hash(body: dict[str, Any], field: str, label: str) -> str:
    supplied = body.get(field)
    unsigned = dict(body)
    unsigned.pop(field, None)
    if not isinstance(supplied, str) or len(supplied) != 64 or _sha256(canonical_json_bytes(unsigned)) != supplied:
        raise Phase3ReadinessError(f"{label} self-hash is invalid")
    return supplied


def _reject_final(body: Mapping[str, Any], label: str) -> None:
    forbidden = (
        ("final_family_access", False),
        ("final_family_accessed", False),
        ("final", False),
        ("development_only", True),
        ("scope", "known-development-only"),
    )
    for key, expected in forbidden:
        if key in body and body[key] != expected:
            raise Phase3ReadinessError(f"{label} has final or non-development scope")
    final_results = body.get("final_results")
    if final_results not in (None, [], {}):
        raise Phase3ReadinessError(f"{label} contains final results")


@dataclass(frozen=True, slots=True)
class Phase3ReadinessSnapshot:
    """Immutable authority snapshot consumed by a future activation gate."""

    repository: Path
    files: tuple[AuthorityFileSnapshot, ...]
    directories: tuple[AuthorityDirectorySnapshot, ...]
    repository_identity: tuple[int, int]
    git_commit_sha: str
    git_dirty: bool
    plan_id: str
    model_authority_sha256: str
    preparation_git_commit_sha: str
    generation_git_commit_sha: str

    @property
    def files_by_path(self) -> Mapping[str, AuthorityFileSnapshot]:
        return {item.relative_path: item for item in self.files}

    @property
    def directories_by_path(self) -> Mapping[str, AuthorityDirectorySnapshot]:
        return {item.relative_path: item for item in self.directories}

    def recheck(self, *, execution_preflight: bool = False, expected_git_commit: str | None = None) -> None:
        """Revalidate every retained byte, inode, ancestor, and authority link."""

        try:
            current_repository = secure_fs.open_directory_chain(self.repository)
            try:
                current_identity = _identity(os.fstat(current_repository))
            finally:
                os.close(current_repository)
        except (OSError, RuntimeError, ValueError) as exc:
            raise Phase3ReadinessError("authority repository cannot be reopened safely") from exc
        if current_identity != self.repository_identity:
            raise Phase3ReadinessError("authority repository identity changed")
        for expected in self.files:
            current = _read_source(self.repository, expected.relative_path)
            if (
                current.content != expected.content
                or current.sha256 != expected.sha256
                or current.parent_identity != expected.parent_identity
                or current.file_identity != expected.file_identity
                or current.ancestor_identities != expected.ancestor_identities
            ):
                raise Phase3ReadinessError(f"authority source changed: {expected.relative_path}")
        for expected in self.directories:
            current = _read_directory(self.repository, expected.relative_path)
            if (
                current.identity != expected.identity
                or current.ancestor_identities != expected.ancestor_identities
            ):
                raise Phase3ReadinessError(
                    f"authority directory changed: {expected.relative_path}"
                )
        commit, dirty = _git_state(self.repository)
        if commit != self.git_commit_sha or dirty != self.git_dirty:
            raise Phase3ReadinessError("repository provenance changed since readiness capture")
        if execution_preflight:
            if dirty:
                raise Phase3ReadinessError("execution preflight requires a clean repository")
            if expected_git_commit is None:
                raise Phase3ReadinessError(
                    "execution preflight requires an explicit authorized commit"
                )
            if commit != expected_git_commit:
                raise Phase3ReadinessError("execution preflight repository commit is not authorized")
        _validate_authority_files(self)

    def preflight(self, *, expected_git_commit: str) -> None:
        self.recheck(execution_preflight=True, expected_git_commit=expected_git_commit)

    @contextmanager
    def hold_for_activation(
        self, *, expected_git_commit: str
    ) -> Iterator[Phase3ActivationReadinessLease]:
        """Keep all checked authority descriptors open through activation.

        Future activation code must consume the yielded live lease.  Calling
        :meth:`preflight` alone is diagnostic and never sufficient authority.
        """

        self.preflight(expected_git_commit=expected_git_commit)
        stack = ExitStack()
        stack.__enter__()
        lease: Phase3ActivationReadinessLease | None = None
        try:
            root_fd = secure_fs.open_directory_chain(self.repository)
            stack.callback(os.close, root_fd)
            if _identity(os.fstat(root_fd)) != self.repository_identity:
                raise Phase3ReadinessError("held authority repository identity changed")
            file_descriptors = {
                expected.relative_path: _hold_file_from_root(root_fd, expected, stack)
                for expected in self.files
            }
            directory_descriptors = {
                expected.relative_path: _hold_directory_from_root(
                    root_fd, expected, stack
                )
                for expected in self.directories
            }
            lease = Phase3ActivationReadinessLease(
                self,
                file_descriptors,
                directory_descriptors,
                _token=_ACTIVATION_LEASE_TOKEN,
            )
            yield lease.require_active()
        except Phase3ReadinessError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise Phase3ReadinessError(
                "cannot hold Phase 3 readiness descriptors for activation"
            ) from exc
        finally:
            if lease is not None:
                lease._deactivate()
            stack.close()
            self.preflight(expected_git_commit=expected_git_commit)

    def __enter__(self) -> "Phase3ReadinessSnapshot":
        return self

    def __exit__(self, *_: object) -> None:
        return None


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
        raise Phase3ReadinessError("cannot capture repository provenance") from exc
    if not commit or any(character not in "0123456789abcdef" for character in commit.lower()):
        raise Phase3ReadinessError("repository commit identity is malformed")
    return commit, dirty


def _validate_authority_files(snapshot: Phase3ReadinessSnapshot) -> None:
    files = snapshot.files_by_path
    protocol = _json(files[PHASE3_PROTOCOL_RELATIVE], "Phase 3 protocol")
    _reject_final(protocol, "Phase 3 protocol")
    if protocol.get("schema_version") != "milestone6.phase3_representation_ladder.v1":
        raise Phase3ReadinessError("Phase 3 protocol schema drifted")
    authority = protocol.get("authority")
    if not isinstance(authority, dict):
        raise Phase3ReadinessError("Phase 3 protocol authority is missing")
    for key in ("development_protocol", "development_tasks", "phase2_candidates", "phase2_selection_lock"):
        source = authority.get(key)
        if not isinstance(source, dict) or source.get("sha256") != files[_relative(source.get("path", ""))].sha256:
            raise Phase3ReadinessError(f"Phase 3 linked authority hash drifted: {key}")
    selection = _json(files[_relative(authority["phase2_selection_lock"]["path"])], "Phase 2 selection lock")
    if selection.get("analysis", {}).get("analysis_sha256") != authority["phase2_selection_lock"].get("analysis_sha256"):
        raise Phase3ReadinessError("Phase 2 selection analysis hash drifted")
    _reject_final(selection, "Phase 2 selection lock")

    plan = _json(files[PHASE3_PLAN_LOCK_RELATIVE], "Phase 3 plan lock", canonical=True)
    plan_lock_sha = _require_self_hash(plan, "plan_lock_sha256", "Phase 3 plan lock")
    if plan.get("plan_id") != PHASE3_PLAN_ID or plan.get("protocol_sha256") != files[PHASE3_PROTOCOL_RELATIVE].sha256:
        raise Phase3ReadinessError("Phase 3 plan identity or protocol link drifted")
    if plan.get("final_family_access") is not False:
        raise Phase3ReadinessError("Phase 3 plan permits final-family access")

    anchor = _json(files[PHASE3_ANCHOR_RELATIVE], "Phase 3 anchor manifest", canonical=True)
    anchor_sha = _require_self_hash(anchor, "anchor_manifest_sha256", "Phase 3 anchor manifest")
    _reject_final(anchor, "Phase 3 anchor manifest")
    evidence = _json(files[PHASE3_EVIDENCE_RELATIVE], "Phase 3 evidence lock", canonical=True)
    evidence_sha = _require_self_hash(evidence, "evidence_lock_sha256", "Phase 3 evidence lock")
    _reject_final(evidence, "Phase 3 evidence lock")
    for label, payload in (("anchor", anchor), ("evidence", evidence)):
        lineage = payload.get("lineage")
        if not isinstance(lineage, dict) or lineage.get("phase3_protocol_sha256") != files[PHASE3_PROTOCOL_RELATIVE].sha256:
            raise Phase3ReadinessError(f"Phase 3 {label} protocol lineage drifted")
        if lineage.get("phase3_plan_id") not in (None, PHASE3_PLAN_ID):
            raise Phase3ReadinessError(f"Phase 3 {label} plan lineage drifted")
    if evidence.get("lineage", {}).get("phase3_plan_lock_sha256") not in (None, plan_lock_sha):
        raise Phase3ReadinessError("Phase 3 evidence plan-lock lineage drifted")

    authority = load_phase3_model_artifact_authority_bytes(files[PHASE3_MODEL_AUTHORITY_RELATIVE].content)
    if _sha256(files[PHASE3_MODEL_AUTHORITY_RELATIVE].content) != PHASE3_MODEL_AUTHORITY_FILE_SHA256:
        raise Phase3ReadinessError("published Phase 3 model authority bytes differ")
    if (
        authority.authority_sha256 != PHASE3_MODEL_AUTHORITY_SHA256
        or authority.plan_id != PHASE3_PLAN_ID
        or authority.protocol_sha256 != files[PHASE3_PROTOCOL_RELATIVE].sha256
        or authority.plan_file_sha256 != files[PHASE3_PLAN_LOCK_RELATIVE].sha256
        or authority.anchor_manifest_sha256 != anchor_sha
        or authority.anchor_file_sha256 != files[PHASE3_ANCHOR_RELATIVE].sha256
        or authority.evidence_lock_sha256 != evidence_sha
        or authority.evidence_file_sha256 != files[PHASE3_EVIDENCE_RELATIVE].sha256
    ):
        raise Phase3ReadinessError("published Phase 3 model authority lineage drifted")
    prep = files.get(f"runs/milestone6/{authority.artifact_store_id}/{PREPARATION_PROVENANCE_NAME}")
    progress = files.get(f"runs/milestone6/{authority.artifact_store_id}/{PREPARATION_PROGRESS_NAME}")
    if prep is None or progress is None:
        raise Phase3ReadinessError("published model authority provenance inputs are missing")
    if prep.sha256 != authority.provenance_file_sha256:
        raise Phase3ReadinessError("preparation provenance digest drifted")
    if progress.sha256 != authority.progress_sha256:
        raise Phase3ReadinessError("preparation progress digest drifted")
    prep_body = _json(prep, "Phase 3 preparation provenance")
    progress_body = _json(progress, "Phase 3 preparation progress")
    if (
        prep_body.get("provenance_sha256") != authority.preparation_provenance_sha256
        or not isinstance(prep_body.get("provenance"), dict)
        or prep_body["provenance"].get("git_commit_sha") != authority.preparation_git_commit_sha
        or prep_body["provenance"].get("git_dirty") is not False
        or progress_body.get("preparation_provenance_sha256") != authority.preparation_provenance_sha256
        or progress_body.get("preparation_git_commit_sha") != authority.preparation_git_commit_sha
    ):
        raise Phase3ReadinessError("preparation git provenance lineage drifted")


def capture_phase3_readiness(
    repository: str | os.PathLike[str] = ROOT,
    *,
    model_store_root: str | os.PathLike[str] | None = None,
    execution_preflight: bool = False,
    expected_git_commit: str | None = None,
) -> Phase3ReadinessSnapshot:
    """Capture and validate all committed Phase 3 development authorities."""

    repo = Path(repository).absolute()
    try:
        root_stat = repo.lstat()
    except OSError as exc:
        raise Phase3ReadinessError("authority repository is unavailable") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise Phase3ReadinessError("authority repository must be a real directory")
    protocol_snapshot = _read_source(repo, PHASE3_PROTOCOL_RELATIVE)
    protocol = _json(protocol_snapshot, "Phase 3 protocol")
    authority = protocol.get("authority", {})
    files: list[AuthorityFileSnapshot] = [protocol_snapshot]
    for key in ("development_protocol", "development_tasks", "phase2_candidates", "phase2_selection_lock"):
        files.append(_read_source(repo, _relative(authority[key]["path"])))
    files.extend(
        _read_source(repo, path)
        for path in (
            PHASE3_PLAN_LOCK_RELATIVE,
            PHASE3_ANCHOR_RELATIVE,
            PHASE3_EVIDENCE_RELATIVE,
            PHASE3_MODEL_AUTHORITY_RELATIVE,
        )
    )
    model_authority = load_phase3_model_artifact_authority_bytes(files[-1].content)
    canonical_model_root = repo / "runs" / "milestone6" / model_authority.artifact_store_id
    model_root = (
        Path(model_store_root).absolute()
        if model_store_root is not None
        else canonical_model_root
    )
    if model_root != canonical_model_root:
        raise Phase3ReadinessError("model artifact root differs from published authority")
    try:
        model_relative = model_root.relative_to(repo)
    except ValueError as exc:
        raise Phase3ReadinessError("model artifact root must be inside authority repository") from exc
    for name in (PREPARATION_PROVENANCE_NAME, PREPARATION_PROGRESS_NAME):
        rel = "/".join((*model_relative.parts, name))
        files.append(_read_source(repo, rel))
    model_relative_text = "/".join(model_relative.parts)
    directories = [
        _read_directory(repo, model_relative_text),
        *(
            _read_directory(repo, f"{model_relative_text}/{name}")
            for name in (KEYS_DIR, COSTS_DIR, ARTIFACTS_DIR)
        ),
    ]
    commit, dirty = _git_state(repo)
    snapshot = Phase3ReadinessSnapshot(
        repository=repo,
        files=tuple(files),
        directories=tuple(directories),
        repository_identity=_identity(root_stat),
        git_commit_sha=commit,
        git_dirty=dirty,
        plan_id=model_authority.plan_id,
        model_authority_sha256=model_authority.authority_sha256,
        preparation_git_commit_sha=model_authority.preparation_git_commit_sha,
        generation_git_commit_sha=model_authority.generation_git_commit_sha,
    )
    _validate_authority_files(snapshot)
    snapshot.recheck(execution_preflight=execution_preflight, expected_git_commit=expected_git_commit)
    return snapshot


@contextmanager
def phase3_readiness(**kwargs: Any) -> Iterator[Phase3ReadinessSnapshot]:
    """Context-manager spelling for callers preparing a later activation gate."""

    snapshot = capture_phase3_readiness(**kwargs)
    try:
        yield snapshot
    finally:
        pass


__all__ = [
    "AuthorityDirectorySnapshot",
    "AuthorityFileSnapshot",
    "PHASE3_MODEL_AUTHORITY_SHA256",
    "PHASE3_MODEL_AUTHORITY_FILE_SHA256",
    "PHASE3_PLAN_ID",
    "Phase3ActivationReadinessLease",
    "Phase3ReadinessError",
    "Phase3ReadinessSnapshot",
    "capture_phase3_readiness",
    "phase3_readiness",
]
