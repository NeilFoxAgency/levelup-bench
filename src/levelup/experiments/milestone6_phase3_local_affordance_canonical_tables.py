"""Descriptor-pinned source of the pre-existing pooled affordance tables.

The local-affordance raw probe must reproduce the *same* pooled table that was
used by the Phase 2 development evidence.  Rebuilding that table and calling it
canonical would make the parity check circular.  This small capability instead
reads the five independent LOFO copies which contain a task, requires their
typed table bytes to agree, and exposes only a fresh ``AffordanceTableRecord``.

It deliberately has no API for traces, payloads, paths, run identifiers, model
artifacts, result namespaces, or outcomes.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from levelup.experiments.milestone6_phase3_local_affordance_evidence import RawProbeArtifactKey
from levelup.experiments.milestone6_phase3_local_affordance_readiness import (
    LocalAffordanceActivationLease,
    LocalAffordanceReadinessError,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import (
    AffordanceTableRecord,
    PinnedTrainingDataReader,
    TrainingDataEvidenceKey,
    TrainingDataEvidenceManifest,
    TrainingDataEvidencePayloadBundle,
    load_training_data_evidence_payload_bundle_from_at,
    open_training_data_reader,
)

_FAMILIES = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
_REPLICATES = (0, 1, 2, 3, 4)
_NAMESPACE_FIELDS = (
    "evidence_costs_fd",
    "view_costs_fd",
    "view_keys_fd",
    "evidence_root_fd",
    "artifact_root_fd",
)
_TOKEN = object()


class CanonicalPooledTableError(ValueError):
    """Raised if the pre-existing development evidence cannot be trusted."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity(fd: int) -> tuple[int, int]:
    observed = os.fstat(fd)
    if not stat.S_ISDIR(observed.st_mode):
        raise CanonicalPooledTableError("pinned evidence descriptor is not a directory")
    return int(observed.st_dev), int(observed.st_ino)


def _file_identity(fd: int) -> tuple[int, int]:
    observed = os.fstat(fd)
    if not stat.S_ISREG(observed.st_mode):
        raise CanonicalPooledTableError("pinned evidence entry is not a regular file")
    return int(observed.st_dev), int(observed.st_ino)


def _read_regular_fd(fd: int) -> bytes:
    before = os.fstat(fd)
    before_identity = _file_identity(fd)
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(fd, 1024 * 1024):
        chunks.append(chunk)
    after = os.fstat(fd)
    os.lseek(fd, 0, os.SEEK_SET)
    if before_identity != _file_identity(fd) or (
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise CanonicalPooledTableError("held evidence file changed during read")
    return b"".join(chunks)


def _safe_component(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise CanonicalPooledTableError(f"{label} is not a safe path component")
    if any(char in value for char in ("/", "\\", "\x00")):
        raise CanonicalPooledTableError(f"{label} is not a safe path component")
    return value


def _canonical_object(content: bytes, label: str) -> dict[str, Any]:
    if not isinstance(content, bytes) or not content:
        raise CanonicalPooledTableError(f"{label} bytes are missing")
    try:
        body = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise CanonicalPooledTableError(f"{label} bytes are not JSON") from exc
    if not isinstance(body, dict) or canonical_json_bytes(body) != content:
        raise CanonicalPooledTableError(f"{label} bytes are not canonical")
    return body


@dataclass(frozen=True, slots=True)
class _EvidenceExpectation:
    heldout_family: str
    replicate: int
    child_run_id: str
    evidence_id: str
    key: TrainingDataEvidenceKey
    manifest: TrainingDataEvidenceManifest
    manifest_sha256: str
    payload_sha256: str
    payload_bytes: int


def _expectations(lock_bytes: bytes) -> dict[tuple[str, int], _EvidenceExpectation]:
    body = _canonical_object(lock_bytes, "Phase 3 evidence lock")
    unsigned = dict(body)
    supplied = unsigned.pop("evidence_lock_sha256", None)
    if not isinstance(supplied, str) or _sha256(canonical_json_bytes(unsigned)) != supplied:
        # The lock's self hash is a hash of the canonical unsigned object.
        raise CanonicalPooledTableError("Phase 3 evidence lock self-hash drifted")
    if (
        body.get("schema_version") != "milestone6.phase3.evidence-lock.v1"
        or body.get("scope") != "known-development-only"
        or body.get("final_family_access") is not False
        or body.get("payloads_included") is not False
        or body.get("outcomes_included") is not False
        or body.get("aggregates") != []
        or body.get("final_results") != []
        or body.get("counts")
        != {"evidence_artifacts": 30, "families": 6, "replicates": 5}
    ):
        raise CanonicalPooledTableError("Phase 3 evidence lock is outside development-only scope")
    rows = body.get("evidence_artifacts")
    if not isinstance(rows, list) or len(rows) != 30:
        raise CanonicalPooledTableError("Phase 3 evidence lock must contain exactly 30 artifacts")
    result: dict[tuple[str, int], _EvidenceExpectation] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise CanonicalPooledTableError("Phase 3 evidence row is not an object")
        family = row.get("family_id")
        replicate = row.get("replicate")
        if family not in _FAMILIES or type(replicate) is not int or replicate not in _REPLICATES:
            raise CanonicalPooledTableError("Phase 3 evidence row family or replicate is invalid")
        try:
            key = TrainingDataEvidenceKey.model_validate(row["evidence_key"])
            manifest = TrainingDataEvidenceManifest.model_validate(row["evidence_manifest"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CanonicalPooledTableError("Phase 3 evidence row is not typed") from exc
        child = _safe_component(row.get("child_run_id"), "locked child run ID")
        evidence_id = str(row.get("evidence_id", ""))
        manifest_sha256 = row.get("canonical_manifest_bytes_sha256")
        payload_sha256 = row.get("payload_sha256")
        payload_bytes = row.get("payload_bytes")
        if (
            len(evidence_id) != 64
            or key.heldout_family_id != family
            or key.replicate != replicate
            or manifest.key != key
            or manifest.evidence_id != evidence_id
            or row.get("evidence_key_id") != key.key_id
            or row.get("evidence_manifest_key_id") != key.key_id
            or row.get("fold_id") != key.fold_id
            or tuple(row.get("ordered_training_task_ids", ())) != key.ordered_training_task_ids
            or manifest_sha256 is None
            or payload_sha256 != manifest.payload_sha256
            or payload_bytes != manifest.payload_bytes
        ):
            raise CanonicalPooledTableError("Phase 3 evidence row lineage differs from typed manifest")
        identity = (family, replicate)
        if identity in result:
            raise CanonicalPooledTableError("Phase 3 evidence lock duplicates a family/replicate")
        result[identity] = _EvidenceExpectation(
            heldout_family=family,
            replicate=replicate,
            child_run_id=child,
            evidence_id=evidence_id,
            key=key,
            manifest=manifest,
            manifest_sha256=str(manifest_sha256),
            payload_sha256=str(payload_sha256),
            payload_bytes=payload_bytes,
        )
    if set(result) != {(family, rep) for family in _FAMILIES for rep in _REPLICATES}:
        raise CanonicalPooledTableError("Phase 3 evidence lock coverage is incomplete or extra")
    if len({value.child_run_id for value in result.values()}) != len(_FAMILIES):
        raise CanonicalPooledTableError("Phase 3 evidence lock must bind exactly six child runs")
    for family in _FAMILIES:
        runs = {result[family, rep].child_run_id for rep in _REPLICATES}
        if len(runs) != 1:
            raise CanonicalPooledTableError("one LOFO family must use one pinned child run")
    return result


def _bundle_is_expected(bundle: TrainingDataEvidencePayloadBundle, expected: _EvidenceExpectation) -> None:
    if (
        bundle.manifest != expected.manifest
        or bundle.manifest.key != expected.key
        or bundle.manifest.evidence_id != expected.evidence_id
        or _sha256(bundle.manifest_bytes) != expected.manifest_sha256
        or _sha256(bundle.payload_bytes) != expected.payload_sha256
        or len(bundle.payload_bytes) != expected.payload_bytes
    ):
        raise CanonicalPooledTableError("descriptor-read evidence bundle differs from lock")


def _open_pinned_reader(stack: ExitStack, run_fd: int) -> PinnedTrainingDataReader:
    # Entering the public context manager retains all five namespace descriptors
    # until this capability closes.  The public bundle loader therefore never
    # reopens a run path.
    return stack.enter_context(open_training_data_reader(run_fd))


def _copy_table(value: AffordanceTableRecord) -> AffordanceTableRecord:
    return AffordanceTableRecord.model_validate(value.model_dump(mode="json"))


def _expectations_digest(
    expectations: Mapping[tuple[str, int], _EvidenceExpectation],
) -> str:
    return _sha256(
        canonical_json_bytes(
            tuple(
                {
                    "identity": identity,
                    "heldout_family": expected.heldout_family,
                    "replicate": expected.replicate,
                    "child_run_id": expected.child_run_id,
                    "evidence_id": expected.evidence_id,
                    "key": expected.key.model_dump(mode="json"),
                    "manifest": expected.manifest.model_dump(mode="json"),
                    "manifest_sha256": expected.manifest_sha256,
                    "payload_sha256": expected.payload_sha256,
                    "payload_bytes": expected.payload_bytes,
                }
                for identity, expected in sorted(expectations.items())
            )
        )
    )


@dataclass(frozen=True, slots=True)
class _SourceSeal:
    lease: LocalAffordanceActivationLease
    raw_root: Path
    root_identity: tuple[int, int]
    child_identities: tuple[tuple[str, tuple[int, int]], ...]
    namespace_identities: tuple[tuple[str, tuple[tuple[str, tuple[int, int]], ...]], ...]
    evidence_entry_identities: tuple[
        tuple[tuple[str, int], tuple[int, int], tuple[int, int], tuple[int, int]]
    ]
    bundle_bytes: tuple[tuple[tuple[str, int], bytes, bytes], ...]
    tables_digest: str
    lock_sha256: str
    key_universe: tuple[str, ...]
    expectations_digest: str
    token: object


class CanonicalPooledTableSource:
    """An active, opaque source that returns only canonical pooled tables."""

    __slots__ = (
        "_lease",
        "_raw_root",
        "_root_fd",
        "_child_fds",
        "_readers",
        "_expectations",
        "_tables",
        "_evidence_fds",
        "_lock_sha256",
        "_seal",
        "_active",
        "_token",
    )

    def __init__(self, *, _token: object | None = None, **kwargs: Any) -> None:
        if _token is not _TOKEN:
            raise CanonicalPooledTableError("canonical pooled tables require activation")
        bundle_bytes = kwargs.pop("_bundle_bytes")
        lock_bytes = kwargs.pop("_lock_bytes")
        for name, value in kwargs.items():
            setattr(self, name, value)
        table_bytes = tuple(
            (key_id, canonical_json_bytes(table.model_dump(mode="json")).hex())
            for key_id, table in sorted(self._tables.items())
        )
        self._child_fds = MappingProxyType(dict(self._child_fds))
        self._readers = MappingProxyType(dict(self._readers))
        self._expectations = MappingProxyType(dict(self._expectations))
        self._tables = MappingProxyType(dict(self._tables))
        self._evidence_fds = MappingProxyType(dict(self._evidence_fds))
        self._lock_sha256 = _sha256(lock_bytes)
        self._seal = _SourceSeal(
            lease=self._lease,
            raw_root=self._raw_root,
            root_identity=_identity(self._root_fd),
            child_identities=tuple(
                (name, _identity(fd)) for name, fd in sorted(self._child_fds.items())
            ),
            namespace_identities=tuple(
                (
                    name,
                    tuple((field, _identity(getattr(reader, field))) for field in _NAMESPACE_FIELDS),
                )
                for name, reader in sorted(self._readers.items())
            ),
            evidence_entry_identities=tuple(
                (
                    identity,
                    _identity(fds[0]),
                    _file_identity(fds[1]),
                    _file_identity(fds[2]),
                )
                for identity, fds in sorted(self._evidence_fds.items())
            ),
            bundle_bytes=tuple(
                (identity, manifest, payload)
                for identity, (manifest, payload) in sorted(bundle_bytes.items())
            ),
            tables_digest=_sha256(canonical_json_bytes(table_bytes)),
            lock_sha256=self._lock_sha256,
            key_universe=tuple(sorted(self._tables)),
            expectations_digest=_expectations_digest(self._expectations),
            token=_TOKEN,
        )
        self._active = True
        self._token = _TOKEN

    def __repr__(self) -> str:
        return "CanonicalPooledTableSource(active development-only tables)"

    def _require_active(self) -> None:
        try:
            if (
                type(self) is not CanonicalPooledTableSource
                or self._token is not _TOKEN
                or self._active is not True
                or type(self._seal) is not _SourceSeal
                or self._seal.token is not _TOKEN
                or self._lease is not self._seal.lease
                or self._raw_root != self._seal.raw_root
                or self._lock_sha256 != self._seal.lock_sha256
                or _identity(self._root_fd) != self._seal.root_identity
                or tuple((name, _identity(fd)) for name, fd in sorted(self._child_fds.items()))
                != self._seal.child_identities
                or tuple(
                    (
                        name,
                        tuple((field, _identity(getattr(reader, field))) for field in _NAMESPACE_FIELDS),
                    )
                    for name, reader in sorted(self._readers.items())
                )
                != self._seal.namespace_identities
                or tuple(
                    (
                        identity,
                        _identity(fds[0]),
                        _file_identity(fds[1]),
                        _file_identity(fds[2]),
                    )
                    for identity, fds in sorted(self._evidence_fds.items())
                )
                != self._seal.evidence_entry_identities
                or tuple(sorted(self._tables)) != self._seal.key_universe
                or _expectations_digest(self._expectations)
                != self._seal.expectations_digest
                or _sha256(
                    canonical_json_bytes(
                        tuple(
                            (
                                key_id,
                                canonical_json_bytes(table.model_dump(mode="json")).hex(),
                            )
                            for key_id, table in sorted(self._tables.items())
                        )
                    )
                )
                != self._seal.tables_digest
            ):
                raise CanonicalPooledTableError("canonical pooled table source is expired or forged")
            self._lease.require_active()
        except CanonicalPooledTableError:
            raise
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise CanonicalPooledTableError("canonical pooled table source is expired or forged") from exc

    def _check_current_evidence_entries(
        self,
        child_name: str,
        current_reader: PinnedTrainingDataReader,
    ) -> None:
        """Compare each current evidence dir/file identity to the held fd identity."""

        for identity, expected in self._expectations.items():
            if expected.child_run_id != child_name:
                continue
            held_dir, held_manifest, held_payload = self._evidence_fds[identity]
            evidence_fd = secure_fs.open_child_directory(current_reader.evidence_root_fd, expected.evidence_id)
            try:
                with secure_fs.open_regular_file_at(evidence_fd, "manifest.json") as manifest_fd:
                    with secure_fs.open_regular_file_at(evidence_fd, "samples.json") as payload_fd:
                        current = (
                            _identity(evidence_fd),
                            _file_identity(manifest_fd),
                            _file_identity(payload_fd),
                        )
                        held = (
                            _identity(held_dir),
                            _file_identity(held_manifest),
                            _file_identity(held_payload),
                        )
                if current != held:
                    raise CanonicalPooledTableError(
                        "Phase 2 evidence entry path identity changed"
                    )
            finally:
                os.close(evidence_fd)

    def _recheck(self) -> None:
        """Check current raw-root paths and every locked bundle without outcomes."""

        self._require_active()
        current_root = secure_fs.open_directory_chain(self._raw_root)
        try:
            if _identity(current_root) != self._seal.root_identity:
                raise CanonicalPooledTableError("Phase 2 raw-root path identity changed")
            for child_name, expected_identity in self._seal.child_identities:
                current_child = secure_fs.open_child_directory(current_root, child_name)
                try:
                    if _identity(current_child) != expected_identity:
                        raise CanonicalPooledTableError("Phase 2 child-run path identity changed")
                    current_reader_cm = open_training_data_reader(current_child)
                    with current_reader_cm as current_reader:
                        held_reader = self._readers[child_name]
                        for field in _NAMESPACE_FIELDS:
                            if _identity(getattr(current_reader, field)) != _identity(
                                getattr(held_reader, field)
                            ):
                                raise CanonicalPooledTableError(
                                    "Phase 2 evidence namespace path identity changed"
                                )
                        self._check_current_evidence_entries(child_name, current_reader)
                finally:
                    os.close(current_child)
        finally:
            os.close(current_root)
        expected_bytes = dict((identity, (manifest, payload)) for identity, manifest, payload in self._seal.bundle_bytes)
        for identity, expected in self._expectations.items():
            _held_dir, held_manifest, held_payload = self._evidence_fds[identity]
            if (
                _read_regular_fd(held_manifest),
                _read_regular_fd(held_payload),
            ) != expected_bytes[identity]:
                raise CanonicalPooledTableError("held evidence file bytes drifted")
            bundle = load_training_data_evidence_payload_bundle_from_at(
                self._readers[expected.child_run_id], expected.evidence_id, expected_key=expected.key
            )
            _bundle_is_expected(bundle, expected)
            if (bundle.manifest_bytes, bundle.payload_bytes) != expected_bytes[identity]:
                raise CanonicalPooledTableError("locked evidence manifest or payload bytes drifted")
        # Re-open the current paths after all held-descriptor reads: a same-byte
        # rename between the first check and loader call is still a hard drift.
        current_root = secure_fs.open_directory_chain(self._raw_root)
        try:
            for child_name, _expected_identity in self._seal.child_identities:
                current_child = secure_fs.open_child_directory(current_root, child_name)
                try:
                    with open_training_data_reader(current_child) as current_reader:
                        self._check_current_evidence_entries(child_name, current_reader)
                finally:
                    os.close(current_child)
        finally:
            os.close(current_root)
        self._require_active()

    def table_for(self, key: RawProbeArtifactKey) -> AffordanceTableRecord:
        """Return a fresh canonical table for one frozen development probe key."""

        if type(key) is not RawProbeArtifactKey:
            raise CanonicalPooledTableError("canonical table lookup requires a raw-probe key")
        self._require_active()
        try:
            value = self._tables[key.key_id]
        except KeyError as exc:
            raise CanonicalPooledTableError("raw-probe key is outside canonical development matrix") from exc
        return _copy_table(value)

    def require_active(self) -> "CanonicalPooledTableSource":
        """Perform the complete descriptor, lineage, and byte recheck.

        Capture calls this before starting probes and after all probes finish,
        immediately before the one raw-store publication.  Individual table
        reads intentionally avoid rescanning 30 large payloads.
        """

        try:
            self._recheck()
        except CanonicalPooledTableError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise CanonicalPooledTableError(
                "canonical pooled table source cannot be revalidated"
            ) from exc
        return self

    def _deactivate(self) -> None:
        self._active = False


@contextmanager
def activate_canonical_pooled_tables(
    lease: LocalAffordanceActivationLease,
    *,
    phase2_raw_root: str | os.PathLike[str],
) -> Iterator[CanonicalPooledTableSource]:
    """Pin and validate all 30 evidence bundles, then expose 240 tables only."""

    if type(lease) is not LocalAffordanceActivationLease:
        raise CanonicalPooledTableError("canonical pooled tables require an activation lease")
    try:
        lease.require_active()
    except LocalAffordanceReadinessError as exc:
        raise CanonicalPooledTableError("activation lease is not active") from exc
    try:
        phase3_evidence_lock_bytes = lease.phase3_evidence_lock_bytes()
    except LocalAffordanceReadinessError as exc:
        raise CanonicalPooledTableError("retained evidence lock cannot be revalidated") from exc
    if _sha256(phase3_evidence_lock_bytes) != lease.authority.evidence_lock_file_sha256:
        raise CanonicalPooledTableError("retained evidence lock digest differs from authority")
    expectations = _expectations(phase3_evidence_lock_bytes)
    root = Path(os.path.abspath(phase2_raw_root))
    stack = ExitStack()
    stack.__enter__()
    source: CanonicalPooledTableSource | None = None
    try:
        root_fd = secure_fs.open_directory_chain(root)
        stack.callback(os.close, root_fd)
        child_by_family = {
            family: expectations[family, 0].child_run_id for family in _FAMILIES
        }
        if len(set(child_by_family.values())) != len(_FAMILIES):
            raise CanonicalPooledTableError("locked LOFO families do not map to six child runs")
        child_fds: dict[str, int] = {}
        readers: dict[str, PinnedTrainingDataReader] = {}
        for child_name in child_by_family.values():
            child_fd = secure_fs.open_child_directory(root_fd, child_name)
            stack.callback(os.close, child_fd)
            child_fds[child_name] = child_fd
            readers[child_name] = _open_pinned_reader(stack, child_fd)
        bundle_bytes: dict[tuple[str, int], tuple[bytes, bytes]] = {}
        bundles: dict[tuple[str, int], TrainingDataEvidencePayloadBundle] = {}
        evidence_fds: dict[tuple[str, int], tuple[int, int, int]] = {}
        for identity, expected in expectations.items():
            evidence_fd = secure_fs.open_child_directory(
                readers[expected.child_run_id].evidence_root_fd, expected.evidence_id
            )
            stack.callback(os.close, evidence_fd)
            manifest_fd = os.open(
                "manifest.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=evidence_fd
            )
            stack.callback(os.close, manifest_fd)
            payload_fd = os.open("samples.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=evidence_fd)
            stack.callback(os.close, payload_fd)
            _file_identity(manifest_fd)
            _file_identity(payload_fd)
            evidence_fds[identity] = (evidence_fd, manifest_fd, payload_fd)
            bundle = load_training_data_evidence_payload_bundle_from_at(
                readers[expected.child_run_id], expected.evidence_id, expected_key=expected.key
            )
            _bundle_is_expected(bundle, expected)
            if (
                _read_regular_fd(manifest_fd),
                _read_regular_fd(payload_fd),
            ) != (bundle.manifest_bytes, bundle.payload_bytes):
                raise CanonicalPooledTableError(
                    "descriptor-read evidence bytes differ from held files"
                )
            bundles[identity] = bundle
            bundle_bytes[identity] = (bundle.manifest_bytes, bundle.payload_bytes)
        tables: dict[str, AffordanceTableRecord] = {}
        for key in lease.authority.keys:
            included = [
                bundles[heldout, key.replicate]
                for heldout in _FAMILIES
                if heldout != key.family_id
            ]
            if len(included) != 5:
                raise CanonicalPooledTableError("raw key does not have exactly five LOFO evidence copies")
            records: list[bytes] = []
            table: AffordanceTableRecord | None = None
            for bundle in included:
                matches = [sample for sample in bundle.payload.samples if sample.task_id == key.task_id]
                if len(matches) != 1:
                    raise CanonicalPooledTableError("LOFO evidence has missing or duplicate task table")
                candidate = matches[0].affordances
                encoded = canonical_json_bytes(candidate.model_dump(mode="json"))
                records.append(encoded)
                if table is None:
                    table = _copy_table(candidate)
            if table is None or len(set(records)) != 1:
                raise CanonicalPooledTableError("LOFO pooled affordance tables are not byte-identical")
            if key.key_id in tables:
                raise CanonicalPooledTableError("raw authority duplicates a canonical table key")
            tables[key.key_id] = table
        if len(tables) != len(lease.authority.keys):
            raise CanonicalPooledTableError("canonical pooled table matrix is incomplete")
        source = CanonicalPooledTableSource(
            _lease=lease,
            _raw_root=root,
            _root_fd=root_fd,
            _child_fds=child_fds,
            _readers=readers,
            _expectations=expectations,
            _tables=tables,
            _evidence_fds=evidence_fds,
            _bundle_bytes=bundle_bytes,
            _lock_bytes=phase3_evidence_lock_bytes,
            _token=_TOKEN,
        )
        source._recheck()
        yield source
        source._recheck()
    except CanonicalPooledTableError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, secure_fs.SecureFilesystemError) as exc:
        raise CanonicalPooledTableError("cannot pin canonical development pooled tables") from exc
    finally:
        if source is not None:
            source._deactivate()
        stack.close()


__all__ = [
    "CanonicalPooledTableError",
    "CanonicalPooledTableSource",
    "activate_canonical_pooled_tables",
]
