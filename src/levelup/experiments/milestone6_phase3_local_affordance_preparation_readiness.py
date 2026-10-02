"""Readiness gate for local-affordance model preparation.

This is a development-only authority boundary, not a model trainer.  It binds
the frozen plan lock, its exact source bytes, the raw-capture summary, and the
complete raw-probe store.  Activation revalidates those authorities while
holding descriptor-pinned inputs and a clean repository commit.  Callers get
only immutable, already-validated data; this module exposes no execution,
search, evaluator, oracle, or result-selection API.  The captured readiness
snapshot retains only raw-store hashes and identities.  Full raw payload bytes
exist only in the active lease, where they are consumed by the fold-capability
issuer, and are dropped when the lease expires.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from levelup.experiments import milestone6_phase3_local_affordance_capabilities as _capabilities
from levelup.experiments import milestone6_phase3_local_affordance_plan as plan_module
from levelup.experiments.milestone6_phase3_local_affordance_raw_authority import (
    ExpectedRawProbeAuthority,
    RawProbeAuthorityError,
    RawProbeAuthoritySnapshot,
    build_expected_raw_probe_authority,
    require_expected_raw_probe_authority,
    require_raw_probe_authority_snapshot,
    validate_complete_raw_probe_authority,
)
from levelup.experiments.milestone6_phase3_local_affordance_raw_store import (
    PinnedRawProbeStoreReader,
    RawProbeStoreError,
    open_existing_raw_probe_store,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes


class LocalAffordancePreparationReadinessError(ValueError):
    """Raised when development model-preparation authority is not exact."""


ROOT = plan_module.ROOT
PLAN_LOCK_RELATIVE_PATH = "configs/milestone6/phase3_local_affordance_plan_lock.json"
RAW_CAPTURE_SUMMARY_RELATIVE_PATH = "experiments/milestone6_phase3_local_affordance_raw_capture.json"
SOURCE_RELATIVE_PATHS = (
    "configs/milestone6/phase3_local_affordance_protocol.json",
    "configs/milestone6/development_protocol.json",
    "configs/milestone6/development_tasks.json",
    "configs/milestone6/phase3_evidence_lock.json",
    "configs/milestone6/phase3_representation_ladder.json",
    PLAN_LOCK_RELATIVE_PATH,
    RAW_CAPTURE_SUMMARY_RELATIVE_PATH,
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_SNAPSHOT_TOKEN = object()
_LEASE_TOKEN = object()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _identity(value: os.stat_result) -> tuple[int, int]:
    if not (stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode)):
        raise LocalAffordancePreparationReadinessError("authority contains a non-regular path")
    return int(value.st_dev), int(value.st_ino)


def _relative(value: str) -> str:
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or "." in pure.parts
        or ".." in pure.parts
        or any(not part or "\\" in part or "\x00" in part for part in pure.parts)
    ):
        raise LocalAffordancePreparationReadinessError("authority path is not repository-relative")
    return "/".join(pure.parts)


def _open_source(root_fd: int, relative_path: str, stack: ExitStack) -> tuple[int, int, tuple[tuple[str, tuple[int, int]], ...]]:
    parts = _relative(relative_path).split("/")
    parent_fd = root_fd
    ancestors: list[tuple[str, tuple[int, int]]] = [("", _identity(os.fstat(root_fd)))]
    for index, part in enumerate(parts[:-1]):
        child_fd = secure_fs.open_child_directory(parent_fd, part)
        stack.callback(os.close, child_fd)
        parent_fd = child_fd
        ancestors.append(("/".join(parts[: index + 1]), _identity(os.fstat(child_fd))))
    parent_identity = _identity(os.fstat(parent_fd))
    try:
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError as exc:
        raise LocalAffordancePreparationReadinessError(
            f"cannot open authority source: {relative_path}"
        ) from exc
    stack.callback(os.close, file_fd)
    if not stat.S_ISREG(os.fstat(file_fd).st_mode):
        raise LocalAffordancePreparationReadinessError("authority source is not a regular file")
    return file_fd, parent_identity, tuple(ancestors)


def _read_snapshot(root_fd: int, relative_path: str, stack: ExitStack) -> "PreparationSource":
    fd, parent_identity, ancestors = _open_source(root_fd, relative_path, stack)
    before = os.fstat(fd)
    chunks: list[bytes] = []
    while chunk := os.read(fd, 1024 * 1024):
        chunks.append(chunk)
    after = os.fstat(fd)
    content = b"".join(chunks)
    if _identity(before) != _identity(after) or before.st_size != len(content):
        raise LocalAffordancePreparationReadinessError("authority source changed while being read")
    os.lseek(fd, 0, os.SEEK_SET)
    return PreparationSource(
        relative_path=relative_path,
        content=content,
        sha256=_sha256(content),
        parent_identity=parent_identity,
        file_identity=_identity(before),
        ancestor_identities=ancestors,
    )


def _json_object(content: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise LocalAffordancePreparationReadinessError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise LocalAffordancePreparationReadinessError(f"{label} must be a JSON object")
    return value


def _ordered_ids_sha256(values: tuple[str, ...]) -> str:
    return _sha256(json.dumps(values, separators=(",", ":")).encode("utf-8"))


def _raw_destination(summary: Mapping[str, Any], repository: Path) -> Path:
    value = summary.get("destination")
    if not isinstance(value, str):
        raise LocalAffordancePreparationReadinessError("raw summary destination is missing")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or "." in pure.parts:
        raise LocalAffordancePreparationReadinessError("raw summary destination is unsafe")
    return repository.joinpath(*pure.parts)


def _expected_authority(sources: Mapping[str, "PreparationSource"]) -> ExpectedRawProbeAuthority:
    names = SOURCE_RELATIVE_PATHS[:4]
    try:
        return build_expected_raw_probe_authority(
            local_affordance_protocol_bytes=sources[names[0]].content,
            development_protocol_bytes=sources[names[1]].content,
            development_tasks_bytes=sources[names[2]].content,
            phase3_evidence_lock_bytes=sources[names[3]].content,
        )
    except (RawProbeAuthorityError, KeyError, TypeError, ValueError) as exc:
        raise LocalAffordancePreparationReadinessError(
            "raw authority source bytes are not the frozen development authority"
        ) from exc


def _validate_summary_and_plan(
    sources: Mapping[str, "PreparationSource"], repository: Path
) -> tuple[object, dict[str, Any]]:
    plan_lock = sources[PLAN_LOCK_RELATIVE_PATH].content
    summary_source = sources[RAW_CAPTURE_SUMMARY_RELATIVE_PATH]
    summary = _json_object(summary_source.content, "raw capture summary")
    if summary_source.sha256 != plan_module.FROZEN_SOURCE_SHA256["raw_capture_summary"]:
        raise LocalAffordancePreparationReadinessError("raw capture summary bytes are not frozen")
    try:
        plan = plan_module.validate_local_affordance_plan_lock_bytes(
            plan_lock, repository=repository
        )
    except (plan_module.LocalAffordancePlanError, OSError, TypeError, ValueError) as exc:
        raise LocalAffordancePreparationReadinessError(
            "local-affordance plan lock does not rebuild from frozen sources"
        ) from exc
    if summary.get("schema_version") != "milestone6.phase3.local-affordance-raw-capture-summary.v1":
        raise LocalAffordancePreparationReadinessError("raw capture summary schema drifted")
    if (
        summary.get("scope") != "known-development-only"
        or summary.get("final_family_access") is not False
        or summary.get("comparative_results_generated") is not False
        or summary.get("comparative_results_inspected") is not False
        or summary.get("status") not in {"complete", "complete-with-accounting-metadata-loss"}
    ):
        raise LocalAffordancePreparationReadinessError(
            "raw capture summary is incomplete or outside development scope"
        )
    activation_commit = summary.get("activation_git_commit")
    if not isinstance(activation_commit, str) or _COMMIT_RE.fullmatch(activation_commit) is None:
        raise LocalAffordancePreparationReadinessError("raw capture commit provenance is malformed")
    if summary.get("status") == "complete-with-accounting-metadata-loss" and not summary.get(
        "accounting_metadata_loss_reason"
    ):
        raise LocalAffordancePreparationReadinessError("raw accounting metadata loss is unexplained")
    for field, expected in (
        ("artifact_count", 240),
        ("key_count", 240),
        ("heldout_binding_count", 240),
        ("training_fold_count", 30),
        ("physical_probe_actions", 15_360),
        ("logical_consumer_equivalent_actions", 737_280),
        ("training_actions", 0),
        ("search_actions", 0),
        ("replay_actions", 0),
        ("evaluator_calls", 0),
        ("oracle_calls", 0),
    ):
        if summary.get(field) != expected:
            raise LocalAffordancePreparationReadinessError(f"raw summary {field} drifted")
    lock = _json_object(plan_lock, "plan lock")
    for field in ("raw_authority_manifest_id", "raw_authority_content_sha256", "raw_manifest_file_sha256"):
        if summary.get({
            "raw_authority_manifest_id": "raw_authority_manifest_id",
            "raw_authority_content_sha256": "raw_authority_content_sha256",
            "raw_manifest_file_sha256": "raw_manifest_file_sha256",
        }[field]) != lock.get(field):
            raise LocalAffordancePreparationReadinessError(f"plan lock and raw summary differ: {field}")
    return plan, summary


def _directory_chain(path: Path) -> tuple[tuple[str, tuple[int, int]], ...]:
    absolute = Path(os.path.abspath(path))
    chain = list(reversed(absolute.parents)) + [absolute]
    result: list[tuple[str, tuple[int, int]]] = []
    for item in chain:
        try:
            observed = item.lstat()
        except OSError as exc:
            raise LocalAffordancePreparationReadinessError("authority directory is unavailable") from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise LocalAffordancePreparationReadinessError("authority directory contains a symlink or non-directory")
        result.append((str(item), _identity(observed)))
    return tuple(result)


def _git_state(repository: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=repository, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=repository, check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LocalAffordancePreparationReadinessError("cannot capture repository provenance") from exc
    if _COMMIT_RE.fullmatch(commit) is None:
        raise LocalAffordancePreparationReadinessError("repository commit identity is malformed")
    return commit, dirty


@dataclass(frozen=True, slots=True)
class PreparationSource:
    relative_path: str
    content: bytes
    sha256: str
    parent_identity: tuple[int, int]
    file_identity: tuple[int, int]
    ancestor_identities: tuple[tuple[str, tuple[int, int]], ...]


@dataclass(frozen=True, slots=True)
class _SnapshotSeal:
    digest: str
    token: object


@dataclass(frozen=True, slots=True)
class RawAuthorityFileFingerprint:
    namespace: str
    name: str
    identity: tuple[int, int, int, int, int, int]
    sha256: str


@dataclass(frozen=True, slots=True)
class RawAuthorityFingerprint:
    """Raw-store integrity summary retaining no artifact payload bytes."""

    manifest_id: str
    authority_content_sha256: str
    manifest_file_sha256: str
    evidence_lock_file_sha256: str
    directory_identities: tuple[tuple[int, int], ...]
    files: tuple[RawAuthorityFileFingerprint, ...]


def _raw_fingerprint(snapshot: RawProbeAuthoritySnapshot) -> RawAuthorityFingerprint:
    records = (
        snapshot.manifest_file,
        *snapshot.artifact_files,
        *snapshot.key_files,
        *snapshot.training_fold_files,
        *snapshot.heldout_binding_files,
    )
    return RawAuthorityFingerprint(
        manifest_id=snapshot.manifest.manifest_id,
        authority_content_sha256=snapshot.authority_content_sha256,
        manifest_file_sha256=snapshot.manifest_file.snapshot.sha256,
        evidence_lock_file_sha256=snapshot.evidence_lock_file_sha256,
        directory_identities=tuple(
            (identity.device, identity.inode) for identity in snapshot.directory_identities
        ),
        files=tuple(
            RawAuthorityFileFingerprint(
                namespace=record.namespace,
                name=record.name,
                identity=(
                    record.snapshot.identity.device,
                    record.snapshot.identity.inode,
                    record.snapshot.identity.mode,
                    record.snapshot.identity.byte_length,
                    record.snapshot.identity.mtime_ns,
                    record.snapshot.identity.ctime_ns,
                ),
                sha256=record.snapshot.sha256,
            )
            for record in records
        ),
    )


def _validate_fingerprint(value: object) -> RawAuthorityFingerprint:
    expected_counts = {
        "root": 1,
        "artifacts": 240,
        "keys": 240,
        "training-folds": 30,
        "heldout-bindings": 240,
    }
    if type(value) is not RawAuthorityFingerprint:
        raise LocalAffordancePreparationReadinessError("raw authority fingerprint is malformed")
    if (
        len(value.directory_identities) != 5
        or len(value.files) != sum(expected_counts.values())
        or len({(item.namespace, item.name) for item in value.files}) != len(value.files)
        or not re.fullmatch(r"[0-9a-f]{64}", value.manifest_id)
        or not re.fullmatch(r"[0-9a-f]{64}", value.authority_content_sha256)
        or not re.fullmatch(r"[0-9a-f]{64}", value.manifest_file_sha256)
        or not re.fullmatch(r"[0-9a-f]{64}", value.evidence_lock_file_sha256)
    ):
        raise LocalAffordancePreparationReadinessError("raw authority fingerprint is incomplete")
    counts = {namespace: 0 for namespace in expected_counts}
    for item in value.files:
        if (
            type(item) is not RawAuthorityFileFingerprint
            or item.namespace not in counts
            or len(item.identity) != 6
            or any(type(field) is not int or field < 0 for field in item.identity)
            or not re.fullmatch(r"[0-9a-f]{64}", item.sha256)
        ):
            raise LocalAffordancePreparationReadinessError("raw file fingerprint is malformed")
        counts[item.namespace] += 1
    if counts != expected_counts:
        raise LocalAffordancePreparationReadinessError("raw file fingerprint matrix drifted")
    return value


def _fingerprint_json(value: RawAuthorityFingerprint) -> dict[str, object]:
    return {
        "manifest_id": value.manifest_id,
        "authority_content_sha256": value.authority_content_sha256,
        "manifest_file_sha256": value.manifest_file_sha256,
        "evidence_lock_file_sha256": value.evidence_lock_file_sha256,
        "directory_identities": value.directory_identities,
        "files": [
            {
                "namespace": item.namespace,
                "name": item.name,
                "identity": item.identity,
                "sha256": item.sha256,
            }
            for item in value.files
        ],
    }


def _snapshot_digest(snapshot: "LocalAffordancePreparationSnapshot") -> str:
    return _sha256(canonical_json_bytes({
        "repository": str(snapshot.repository),
        "repository_identity": snapshot.repository_identity,
        "sources": [
            (s.relative_path, s.content.hex(), s.sha256, s.parent_identity, s.file_identity, s.ancestor_identities)
            for s in snapshot.sources
        ],
        "raw_root": str(snapshot.raw_root),
        "raw_root_ancestors": snapshot.raw_root_ancestors,
        "plan_id": snapshot.plan.plan_id,
        "raw_authority_fingerprint": _fingerprint_json(snapshot.raw_authority_fingerprint),
        "git_commit_sha": snapshot.git_commit_sha,
        "git_dirty": snapshot.git_dirty,
    }))


@dataclass(frozen=True, slots=True, init=False)
class LocalAffordancePreparationSnapshot:
    repository: Path
    repository_identity: tuple[int, int]
    sources: tuple[PreparationSource, ...]
    plan: object
    _expected_authority: ExpectedRawProbeAuthority
    raw_authority_fingerprint: RawAuthorityFingerprint
    raw_root: Path
    raw_root_ancestors: tuple[tuple[str, tuple[int, int]], ...]
    git_commit_sha: str
    git_dirty: bool
    _seal: _SnapshotSeal
    _token: object

    def __init__(self, *, _token: object | None = None, **values: Any) -> None:
        if _token is not _SNAPSHOT_TOKEN:
            raise LocalAffordancePreparationReadinessError("preparation snapshot requires canonical capture")
        for key, value in values.items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, "_token", _SNAPSHOT_TOKEN)
        object.__setattr__(self, "_seal", _SnapshotSeal("", _SNAPSHOT_TOKEN))
        object.__setattr__(self, "_seal", _SnapshotSeal(_snapshot_digest(self), _SNAPSHOT_TOKEN))

    @property
    def source_by_path(self) -> Mapping[str, PreparationSource]:
        return MappingProxyType({item.relative_path: item for item in self.sources})

    @property
    def raw_capture_git_commit(self) -> str:
        self.require_sealed()
        summary = _json_object(
            self.source_by_path[RAW_CAPTURE_SUMMARY_RELATIVE_PATH].content,
            "raw capture summary",
        )
        value = summary.get("activation_git_commit")
        if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
            raise LocalAffordancePreparationReadinessError("raw capture commit provenance is malformed")
        return value

    def require_sealed(self) -> "LocalAffordancePreparationSnapshot":
        try:
            valid = (
                type(self) is LocalAffordancePreparationSnapshot
                and self._token is _SNAPSHOT_TOKEN
                and type(self._seal) is _SnapshotSeal
                and self._seal.token is _SNAPSHOT_TOKEN
                and self._seal.digest == _snapshot_digest(self)
                and tuple(item.relative_path for item in self.sources) == SOURCE_RELATIVE_PATHS
            )
        except (AttributeError, TypeError, ValueError):
            valid = False
        if not valid:
            raise LocalAffordancePreparationReadinessError("preparation snapshot is forged or rebound")
        try:
            require_expected_raw_probe_authority(self._expected_authority)
            _validate_fingerprint(self.raw_authority_fingerprint)
        except RawProbeAuthorityError as exc:
            raise LocalAffordancePreparationReadinessError("raw authority snapshot seal is invalid") from exc
        return self

    def recheck(self, *, expected_git_commit: str | None = None) -> None:
        self.require_sealed()
        _recheck_sources(self.repository, self.repository_identity, self.sources)
        _recheck_raw(
            self.raw_root,
            self._expected_authority,
            self.raw_authority_fingerprint,
            self.raw_root_ancestors,
        )
        commit, dirty = _git_state(self.repository)
        if commit != self.git_commit_sha or dirty != self.git_dirty:
            raise LocalAffordancePreparationReadinessError("repository provenance changed")
        if dirty:
            raise LocalAffordancePreparationReadinessError("preparation requires a clean repository")
        if expected_git_commit is not None and expected_git_commit != commit:
            raise LocalAffordancePreparationReadinessError("repository commit is not authorized")

    @contextmanager
    def activation(self, *, expected_git_commit: str) -> Iterator["LocalAffordancePreparationLease"]:
        self.recheck(expected_git_commit=expected_git_commit)
        with ExitStack() as stack:
            root_fd = secure_fs.open_directory_chain(self.repository)
            stack.callback(os.close, root_fd)
            if _identity(os.fstat(root_fd)) != self.repository_identity:
                raise LocalAffordancePreparationReadinessError("repository identity changed")
            held: dict[str, int] = {}
            held_sources: dict[str, PreparationSource] = {}
            for expected in self.sources:
                fd, parent_identity, ancestors = _open_source(root_fd, expected.relative_path, stack)
                content = _read_fd(fd)
                current = os.fstat(fd)
                if (
                    content != expected.content
                    or _identity(current) != expected.file_identity
                    or parent_identity != expected.parent_identity
                    or ancestors != expected.ancestor_identities
                ):
                    raise LocalAffordancePreparationReadinessError(
                        f"pinned source differs: {expected.relative_path}"
                    )
                held[expected.relative_path] = fd
                held_sources[expected.relative_path] = expected
            raw_context = stack.enter_context(open_existing_raw_probe_store(self.raw_root))
            raw_reader = raw_context
            raw_snapshot = validate_complete_raw_probe_authority(
                raw_reader, expected=self._expected_authority
            )
            if _raw_fingerprint(raw_snapshot) != self.raw_authority_fingerprint:
                raise LocalAffordancePreparationReadinessError("raw authority changed before activation")
            commit, dirty = _git_state(self.repository)
            if dirty or commit != expected_git_commit or commit != self.git_commit_sha:
                raise LocalAffordancePreparationReadinessError(
                    "repository authorization changed during activation"
                )
            lease = LocalAffordancePreparationLease(
                _snapshot=self,
                _raw_authority=raw_snapshot,
                _raw_reader=raw_reader,
                _source_fds=held,
                _held_sources=held_sources,
                _stack=stack,
                _token=_LEASE_TOKEN,
            )
            lease.require_active()
            yield lease
            lease._expire()


def _read_fd(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(fd, 1024 * 1024):
        chunks.append(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return b"".join(chunks)


def _recheck_sources(
    repository: Path,
    repository_identity: tuple[int, int],
    expected_sources: tuple[PreparationSource, ...],
) -> None:
    try:
        root_fd = secure_fs.open_directory_chain(repository)
        try:
            if _identity(os.fstat(root_fd)) != repository_identity:
                raise LocalAffordancePreparationReadinessError("repository identity changed")
            with ExitStack() as stack:
                for expected in expected_sources:
                    current = _read_snapshot(root_fd, expected.relative_path, stack)
                    if current != expected:
                        raise LocalAffordancePreparationReadinessError(
                            f"authority source changed: {expected.relative_path}"
                        )
        finally:
            os.close(root_fd)
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise LocalAffordancePreparationReadinessError("authority sources cannot be reopened safely") from exc


def _recheck_raw(
    raw_root: Path,
    expected: ExpectedRawProbeAuthority,
    fingerprint: RawAuthorityFingerprint,
    expected_ancestors: tuple[tuple[str, tuple[int, int]], ...],
) -> None:
    if _directory_chain(raw_root) != expected_ancestors:
        raise LocalAffordancePreparationReadinessError("raw-store root path identity changed")
    try:
        with open_existing_raw_probe_store(raw_root) as reader:
            current = validate_complete_raw_probe_authority(reader, expected=expected)
    except (RawProbeAuthorityError, RawProbeStoreError, OSError, ValueError) as exc:
        raise LocalAffordancePreparationReadinessError("complete raw-store authority failed revalidation") from exc
    if _raw_fingerprint(current) != fingerprint:
        raise LocalAffordancePreparationReadinessError("raw-store authority bytes or identities changed")


def _recheck_raw_file_identities(
    reader: PinnedRawProbeStoreReader,
    fingerprint: RawAuthorityFingerprint,
) -> None:
    """Cheaply recheck every validated file through its retained namespace fd."""
    namespaces = {
        "root": reader.root_fd,
        "artifacts": reader.artifacts_fd,
        "keys": reader.keys_fd,
        "training-folds": reader.training_folds_fd,
        "heldout-bindings": reader.heldout_bindings_fd,
    }
    for record in fingerprint.files:
        try:
            observed = os.stat(
                record.name,
                dir_fd=namespaces[record.namespace],
                follow_symlinks=False,
            )
        except OSError as exc:
            raise LocalAffordancePreparationReadinessError(
                f"validated raw file disappeared: {record.name}"
            ) from exc
        device, inode, mode, byte_length, mtime_ns, ctime_ns = record.identity
        if (
            not stat.S_ISREG(observed.st_mode)
            or int(observed.st_dev) != device
            or int(observed.st_ino) != inode
            or int(observed.st_mode) != mode
            or int(observed.st_size) != byte_length
            or int(observed.st_mtime_ns) != mtime_ns
            or int(observed.st_ctime_ns) != ctime_ns
        ):
            raise LocalAffordancePreparationReadinessError(
                f"validated raw file identity changed: {record.name}"
            )


@dataclass(slots=True, init=False, repr=False)
class LocalAffordancePreparationLease:
    _snapshot: LocalAffordancePreparationSnapshot
    _raw_reader: PinnedRawProbeStoreReader
    _raw_authority: RawProbeAuthoritySnapshot | None
    _source_fds: Mapping[str, int]
    _held_sources: Mapping[str, PreparationSource]
    _stack: ExitStack
    _token: object
    _active: bool

    def __init__(self, *, _token: object | None = None, **values: Any) -> None:
        if _token is not _LEASE_TOKEN:
            raise LocalAffordancePreparationReadinessError("preparation lease requires canonical activation")
        for key, value in values.items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, "_source_fds", MappingProxyType(dict(self._source_fds)))
        object.__setattr__(self, "_held_sources", MappingProxyType(dict(self._held_sources)))
        object.__setattr__(self, "_token", _LEASE_TOKEN)
        object.__setattr__(self, "_active", True)

    @property
    def plan(self) -> object:
        self.require_active()
        return self._snapshot.plan

    def issue_training_fold_probe_capability(
        self, *, fold_id: str, replicate: int
    ) -> _capabilities.TrainingFoldProbeCapability:
        """Issue one 40-task fold capability while this lease is active.

        The capability is a sealed, identity-free value copy.  It may be
        consumed after this context expires if the caller retained it; expiry
        prevents only subsequent issuance and does not revoke copied values.
        """
        self.require_active()
        authority = self._raw_authority
        if authority is None:
            raise LocalAffordancePreparationReadinessError("preparation lease is expired")
        capability = _capabilities.issue_training_fold_probe_capability(
            authority,
            fold_id=fold_id,
            replicate=replicate,
        )
        if type(capability) is not _capabilities.TrainingFoldProbeCapability:
            raise LocalAffordancePreparationReadinessError(
                "training capability issuer returned an unexpected capability"
            )
        return capability

    @property
    def provenance(self) -> Mapping[str, str]:
        self.require_active()
        return MappingProxyType({
            "git_commit_sha": self._snapshot.git_commit_sha,
            "raw_capture_git_commit": self._snapshot.raw_capture_git_commit,
            "plan_id": self._snapshot.plan.plan_id,
            "raw_authority_content_sha256": self._snapshot.raw_authority_fingerprint.authority_content_sha256,
        })

    def require_active(self) -> "LocalAffordancePreparationLease":
        if not self._active or self._token is not _LEASE_TOKEN:
            raise LocalAffordancePreparationReadinessError("preparation lease is expired")
        self._snapshot.require_sealed()
        self._raw_reader.recheck()
        _recheck_sources(
            self._snapshot.repository,
            self._snapshot.repository_identity,
            self._snapshot.sources,
        )
        if _directory_chain(self._snapshot.raw_root) != self._snapshot.raw_root_ancestors:
            raise LocalAffordancePreparationReadinessError("raw-store root path identity changed")
        for path, source in self._held_sources.items():
            fd = self._source_fds[path]
            if _read_fd(fd) != source.content or _identity(os.fstat(fd)) != source.file_identity:
                raise LocalAffordancePreparationReadinessError(f"held authority source changed: {path}")
        _recheck_raw_file_identities(
            self._raw_reader,
            self._snapshot.raw_authority_fingerprint,
        )
        commit, dirty = _git_state(self._snapshot.repository)
        if dirty or commit != self._snapshot.git_commit_sha:
            raise LocalAffordancePreparationReadinessError("preparation provenance changed")
        return self

    def _expire(self) -> None:
        object.__setattr__(self, "_active", False)
        object.__setattr__(self, "_raw_authority", None)


def capture_local_affordance_preparation_readiness(
    *, repository: str | Path = ROOT, raw_root: str | Path | None = None
) -> LocalAffordancePreparationSnapshot:
    """Capture exact plan/source/raw authority without training or execution."""
    repository_path = Path(os.path.abspath(repository))
    if _directory_chain(repository_path)[-1][1] != _identity(repository_path.stat()):
        raise LocalAffordancePreparationReadinessError("repository identity is unstable")
    try:
        root_fd = secure_fs.open_directory_chain(repository_path)
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise LocalAffordancePreparationReadinessError("repository is not safely openable") from exc
    try:
        repository_identity = _identity(os.fstat(root_fd))
        with ExitStack() as stack:
            sources = tuple(_read_snapshot(root_fd, path, stack) for path in SOURCE_RELATIVE_PATHS)
        by_path = MappingProxyType({source.relative_path: source for source in sources})
    finally:
        os.close(root_fd)

    plan, summary = _validate_summary_and_plan(by_path, repository_path)
    expected = _expected_authority(by_path)
    declared_raw_root = _raw_destination(summary, repository_path)
    if raw_root is not None and Path(os.path.abspath(raw_root)) != declared_raw_root:
        raise LocalAffordancePreparationReadinessError("raw root differs from committed capture summary")
    raw_path = declared_raw_root
    raw_ancestors = _directory_chain(raw_path)
    try:
        with open_existing_raw_probe_store(raw_path) as reader:
            raw_snapshot = validate_complete_raw_probe_authority(reader, expected=expected)
    except (RawProbeAuthorityError, RawProbeStoreError, OSError, ValueError) as exc:
        raise LocalAffordancePreparationReadinessError("complete raw development authority is invalid") from exc
    require_raw_probe_authority_snapshot(raw_snapshot)
    if (
        raw_snapshot.manifest.manifest_id != summary.get("raw_authority_manifest_id")
        or raw_snapshot.authority_content_sha256 != summary.get("raw_authority_content_sha256")
        or raw_snapshot.manifest_file.snapshot.sha256 != summary.get("raw_manifest_file_sha256")
    ):
        raise LocalAffordancePreparationReadinessError("raw-store content differs from capture summary")
    ordered_key_ids = tuple(key.key_id for key in expected.keys)
    artifact_id_by_key: dict[str, str] = {}
    try:
        for record in raw_snapshot.key_files:
            value = _json_object(record.snapshot.canonical_bytes, "raw key index")
            artifact_id_by_key[value["key_id"]] = value["artifact_id"]
        ordered_artifact_ids = tuple(artifact_id_by_key[key_id] for key_id in ordered_key_ids)
    except (KeyError, TypeError) as exc:
        raise LocalAffordancePreparationReadinessError("raw key/artifact order cannot be reconstructed") from exc
    if (
        summary.get("ordered_key_ids_sha256") != _ordered_ids_sha256(ordered_key_ids)
        or summary.get("ordered_artifact_ids_sha256") != _ordered_ids_sha256(ordered_artifact_ids)
        or len(artifact_id_by_key) != 240
    ):
        raise LocalAffordancePreparationReadinessError("raw ordered key/artifact identities differ from summary")
    _recheck_sources(repository_path, repository_identity, sources)
    commit, dirty = _git_state(repository_path)
    if dirty:
        raise LocalAffordancePreparationReadinessError("model preparation requires a clean repository")
    snapshot = LocalAffordancePreparationSnapshot(
        _token=_SNAPSHOT_TOKEN,
        repository=repository_path,
        repository_identity=repository_identity,
        sources=sources,
        plan=plan,
        _expected_authority=expected,
        raw_authority_fingerprint=_raw_fingerprint(raw_snapshot),
        raw_root=raw_path,
        raw_root_ancestors=raw_ancestors,
        git_commit_sha=commit,
        git_dirty=dirty,
    )
    return snapshot.require_sealed()


def require_local_affordance_preparation_snapshot(
    value: object,
) -> LocalAffordancePreparationSnapshot:
    if type(value) is not LocalAffordancePreparationSnapshot:
        raise LocalAffordancePreparationReadinessError("exact preparation snapshot required")
    return value.require_sealed()


__all__ = [
    "LocalAffordancePreparationLease",
    "LocalAffordancePreparationReadinessError",
    "LocalAffordancePreparationSnapshot",
    "PreparationSource",
    "RawAuthorityFileFingerprint",
    "RawAuthorityFingerprint",
    "SOURCE_RELATIVE_PATHS",
    "capture_local_affordance_preparation_readiness",
    "require_local_affordance_preparation_snapshot",
]
