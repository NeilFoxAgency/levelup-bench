"""Administrative publication of a completed Phase 2 result-namespace identity.

This is deliberately not part of model preparation, execution, or selection.  It
is a one-time trusted-administrator capture after the frozen Phase 2 development
selection has been made.  It may re-read the already-validated result namespaces
to prove identity and completeness, but it does not aggregate, compare, or emit
outcome values.  The returned canonical bytes can be reviewed and committed by a
caller; this module never writes into the repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from levelup.experiments.milestone6_phase2_screening_provenance import (
    CANONICAL_READINESS_PATH,
)
from levelup.experiments.milestone6_phase2_screening_runtime import (
    DirectorySnapshot,
    ScreeningRuntime,
    ScreeningRuntimeFold,
    recheck_screening_runtime_readonly,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.storage import provenance_identity_sha256
from levelup.experiments.runner.training_data_artifacts import TrainingDataArtifactError

FAMILIES = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
EXPECTED_FOLD_UNITS = 1_520
EXPECTED_TOTAL_UNITS = 9_120
SELECTION_LOCK_PATH = Path("configs/milestone6/phase2_screening_selection.json")
SCHEMA_VERSION = "milestone6.phase2.result-namespace-identity.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _fail(message: str, exc: BaseException | None = None) -> None:
    if exc is None:
        raise TrainingDataArtifactError(message)
    raise TrainingDataArtifactError(message) from exc


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _identity(metadata: os.stat_result) -> dict[str, int]:
    return {
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "ctime_ns": int(metadata.st_ctime_ns),
        "mtime_ns": int(metadata.st_mtime_ns),
        "size": int(metadata.st_size),
        "file_type": int(stat.S_IFMT(metadata.st_mode)),
    }


def _read_stable_file(directory_fd: int, name: str) -> tuple[bytes, os.stat_result]:
    try:
        with secure_fs.open_regular_file_at(directory_fd, name) as descriptor:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                _fail(f"administrative snapshot entry is not a regular file: {name}")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if _identity(before) != _identity(after):
                _fail(f"administrative snapshot file changed while being read: {name}")
            return b"".join(chunks), after
    except secure_fs.SecureFilesystemError as exc:
        _fail(f"cannot securely read administrative snapshot entry: {name}", exc)
    raise AssertionError("unreachable")


def _fold_expected_ids(fold: ScreeningRuntimeFold) -> tuple[str, ...]:
    planned = tuple(fold.store.expected.units)
    if len(planned) != EXPECTED_FOLD_UNITS:
        _fail(f"Phase 2 fold {fold.family_id} does not contain exactly 1,520 planned units")
    if any(
        item.key.family_id != fold.family_id or item.key.phase != "validation"
        for item in planned
    ):
        _fail(f"Phase 2 fold {fold.family_id} contains a foreign family or non-validation unit")
    if len({item.unit_id for item in planned}) != EXPECTED_FOLD_UNITS:
        _fail(f"Phase 2 fold {fold.family_id} has duplicate planned unit identities")
    return tuple(item.unit_id for item in planned)


def _capture_fold_namespaces(
    fold: ScreeningRuntimeFold,
) -> tuple[tuple[tuple[str, DirectorySnapshot], ...], list[dict[str, Any]]]:
    expected_ids = set(_fold_expected_ids(fold))
    namespaces: list[tuple[str, DirectorySnapshot]] = []
    records: list[dict[str, Any]] = []
    for namespace in ("units", "attempts"):
        files: list[tuple[str, str]] = []
        file_rows: list[dict[str, Any]] = []
        try:
            with fold.store._open_result_namespace(namespace) as (_, namespace_fd):
                directory = os.fstat(namespace_fd)
                if not stat.S_ISDIR(directory.st_mode):
                    _fail(f"Phase 2 {fold.family_id}/{namespace} is not a directory")
                names = secure_fs.strict_regular_entries(namespace_fd)
                if namespace == "units":
                    observed_ids = {
                        name[:-5]
                        for name in names
                        if name.endswith(".json") and name[:-5] in expected_ids
                    }
                    if len(names) != len(expected_ids) or observed_ids != expected_ids:
                        _fail(f"Phase 2 {fold.family_id} completed-unit names are incomplete or foreign")
                else:
                    for name in names:
                        if not name.endswith(".json"):
                            _fail(f"Phase 2 {fold.family_id} has a malformed attempt name")
                        stem = name[:-5]
                        unit_id, sep, serial = stem.rpartition(".attempt-")
                        if (
                            not sep
                            or unit_id not in expected_ids
                            or len(serial) != 4
                            or not serial.isdigit()
                            or int(serial) < 1
                        ):
                            _fail(f"Phase 2 {fold.family_id} has a foreign or malformed attempt name")
                for name in names:
                    content, file_stat = _read_stable_file(namespace_fd, name)
                    digest = _sha256(content)
                    files.append((name, digest))
                    file_rows.append(
                        {"name": name, "payload_sha256": digest, "fstat": _identity(file_stat)}
                    )
                after_directory = os.fstat(namespace_fd)
                if _identity(directory) != _identity(after_directory):
                    _fail(f"Phase 2 {fold.family_id}/{namespace} changed during capture")
                snapshot: DirectorySnapshot = (
                    directory.st_dev,
                    directory.st_ino,
                    directory.st_ctime_ns,
                    directory.st_mtime_ns,
                    stat.S_IFMT(directory.st_mode),
                    tuple(files),
                )
                namespaces.append((namespace, snapshot))
                records.append(
                    {
                        "name": namespace,
                        "directory": {
                            "device": int(directory.st_dev),
                            "inode": int(directory.st_ino),
                            "ctime_ns": int(directory.st_ctime_ns),
                            "mtime_ns": int(directory.st_mtime_ns),
                            "file_type": int(stat.S_IFMT(directory.st_mode)),
                        },
                        "files": file_rows,
                    }
                )
        except TrainingDataArtifactError:
            raise
        except (OSError, RuntimeError, ValueError, secure_fs.SecureFilesystemError) as exc:
            _fail(f"cannot capture Phase 2 {fold.family_id}/{namespace}", exc)
    return tuple(namespaces), records


def _canonical_selection_lock(
    runtime: ScreeningRuntime,
    expected_sha256: str,
    *,
    expected_parent_identity: tuple[int, int] | None = None,
    expected_file_identity: dict[str, int] | None = None,
) -> tuple[dict[str, Any], tuple[int, int], dict[str, int]]:
    if not _SHA256.fullmatch(expected_sha256):
        _fail("selection-lock SHA256 must be lowercase hexadecimal")
    relative = SELECTION_LOCK_PATH
    authority_repository = runtime.authority_repository
    if authority_repository is None or runtime.authority_repository_identity is None:
        _fail("cannot resolve the pinned Phase 2 authority repository")
    authority_fd = -1
    try:
        authority_fd = secure_fs.open_directory_chain(authority_repository)
        if secure_fs.directory_identity(authority_fd) != runtime.authority_repository_identity:
            _fail("Phase 2 authority repository identity changed before selection-lock read")
        parent_fd = secure_fs.open_child_chain(authority_fd, *relative.parent.parts)
    except secure_fs.SecureFilesystemError as exc:
        if authority_fd >= 0:
            os.close(authority_fd)
        _fail("cannot securely open committed Phase 2 selection-lock directory", exc)
    except TrainingDataArtifactError:
        if authority_fd >= 0:
            os.close(authority_fd)
        raise
    try:
        parent_identity = secure_fs.directory_identity(parent_fd)
        if expected_parent_identity is not None and parent_identity != expected_parent_identity:
            _fail("committed Phase 2 selection-lock parent identity changed")
        content, metadata = _read_stable_file(parent_fd, relative.name)
    finally:
        os.close(parent_fd)
        os.close(authority_fd)
    file_identity = _identity(metadata)
    if expected_file_identity is not None and file_identity != expected_file_identity:
        _fail("committed Phase 2 selection-lock file identity changed")
    if _sha256(content) != expected_sha256:
        _fail("committed Phase 2 selection lock differs from the requested frozen SHA")
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        _fail("committed Phase 2 selection lock is invalid", exc)
    if (
        value.get("schema_version") != "milestone6.phase2.selection-lock.v1"
        or value.get("scientific_boundary")
        != {
            "development_only": True,
            "final_family_access": False,
            "final_method_selection": False,
            "claims_deferred": [
                "transition_information",
                "history_or_sequence",
                "frontier_optimum_pairing",
            ],
        }
        or value.get("matrix", {}).get("families") != 6
        or value.get("matrix", {}).get("units") != EXPECTED_TOTAL_UNITS
    ):
        _fail("selection lock does not bind the frozen Phase 2 development-only matrix")
    return value, parent_identity, file_identity


def publish_phase2_result_namespace_identity(
    runtime: ScreeningRuntime,
    *,
    selection_lock_sha256: str,
) -> bytes:
    """Return canonical, self-hashed identity JSON for completed Phase 2 results.

    This explicit administrative publication boundary is post-selection and
    development-only. It verifies the frozen lock and complete 9,120-unit
    inventory, then emits file names, payload digests, filesystem identities,
    and authority hashes only. No metric, label, or outcome value is serialized.
    """

    if type(runtime) is not ScreeningRuntime:
        _fail("administrative result publication requires an exact validated ScreeningRuntime")
    # Rechecking validates canonical readiness bytes, source/repository
    # provenance, prepared inventory, and the runtime's pre-existing result
    # snapshot. It does not run units or aggregate comparative metrics.
    recheck_screening_runtime_readonly(runtime)
    if (
        tuple(runtime.manifest.family_order) != FAMILIES
        or runtime.manifest.development_only is not True
        or runtime.manifest.final_family_access is not False
        or tuple(fold.family_id for fold in runtime.folds) != FAMILIES
        or runtime.manifest.expected_total_units != EXPECTED_TOTAL_UNITS
    ):
        _fail("runtime is not the exact six-family development-only Phase 2 plan")
    if runtime.provenance.git_dirty or runtime.authority_provenance is None:
        _fail("Phase 2 publication requires clean screening and authority provenance")
    if runtime.authority_provenance.git_dirty:
        _fail("Phase 2 authority repository is dirty")
    if runtime.authority_repository is None or runtime.authority_repository_identity is None:
        _fail("Phase 2 authority repository identity is missing")
    selection_lock, selection_parent_identity, selection_identity = _canonical_selection_lock(
        runtime, selection_lock_sha256
    )
    expected_frozen_snapshot_sha = selection_lock["analysis"][
        "result_namespace_snapshot_sha256"
    ]
    if not isinstance(expected_frozen_snapshot_sha, str) or not _SHA256.fullmatch(
        expected_frozen_snapshot_sha
    ):
        _fail("selection lock has no valid frozen result-namespace snapshot SHA")
    lock_authority = selection_lock.get("authority")
    if not isinstance(lock_authority, dict) or (
        lock_authority.get("readiness_manifest_bytes_sha256")
        != _sha256(runtime.manifest_bytes)
        or lock_authority.get("source_git_commit_sha")
        != runtime.provenance.git_commit_sha
        or lock_authority.get("source_provenance_sha256")
        != provenance_identity_sha256(runtime.provenance)
        or lock_authority.get("prepared_tree_sha256") != runtime.tree_sha256
    ):
        _fail("selection lock does not bind the runtime manifest, provenance, and prepared tree")
    expected_namespaces: list[tuple[str, tuple[tuple[str, DirectorySnapshot], ...]]] = []
    family_records: list[dict[str, Any]] = []
    for fold in runtime.folds:
        namespace_snapshot, namespace_records = _capture_fold_namespaces(fold)
        expected_namespaces.append((fold.store.run_id, namespace_snapshot))
        family_records.append(
            {
                "family_id": fold.family_id,
                "run_id": fold.store.run_id,
                "expected_units": len(fold.store.expected.units),
                "namespaces": namespace_records,
            }
        )
    actual_snapshot = tuple(expected_namespaces)
    if actual_snapshot != runtime.result_namespace_snapshot:
        _fail("Phase 2 result namespace changed since runtime validation")
    if sum(row["expected_units"] for row in family_records) != EXPECTED_TOTAL_UNITS:
        _fail("Phase 2 publication does not cover exactly 9,120 development units")
    result_snapshot_sha256 = _sha256(canonical_json_bytes(actual_snapshot))
    if result_snapshot_sha256 != expected_frozen_snapshot_sha:
        _fail("current result namespace differs from the frozen selection snapshot")
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "trusted_administrative_post_selection_identity_capture",
        "use_boundary": {
            "development_only": True,
            "final_family_access": False,
            "learner_input": False,
            "administrative_metadata_authority": True,
            "comparative_inspection_performed_here": False,
            "outcome_values_serialized": False,
        },
        "frozen_selection": {
            "selection_lock_sha256": selection_lock_sha256,
            "selection_lock_parent_identity": list(selection_parent_identity),
            "selection_lock_fstat": selection_identity,
            "frozen_result_namespace_snapshot_sha256": expected_frozen_snapshot_sha,
        },
        "readiness": {
            "manifest_relative_path": CANONICAL_READINESS_PATH,
            "manifest_bytes_sha256": _sha256(runtime.manifest_bytes),
            "manifest_bytes_length": len(runtime.manifest_bytes),
            "manifest_parent_identity": list(runtime.manifest_parent_identity),
            "manifest_file_identity": list(runtime.manifest_file_identity),
            "prepared_tree_sha256": runtime.tree_sha256,
        },
        "repositories": {
            "screening": {
                "path": str(runtime.repository),
                "git_commit_sha": runtime.provenance.git_commit_sha,
                "git_dirty": runtime.provenance.git_dirty,
                "provenance_sha256": provenance_identity_sha256(runtime.provenance),
            },
            "authority": {
                "path": str(runtime.authority_repository),
                "git_commit_sha": runtime.authority_provenance.git_commit_sha,
                "git_dirty": runtime.authority_provenance.git_dirty,
                "provenance_sha256": provenance_identity_sha256(
                    runtime.authority_provenance
                ),
                "directory_identity": list(runtime.authority_repository_identity),
                "source_sha256": [
                    {"label": label, "sha256": digest}
                    for label, digest in runtime.authority_digests
                ],
            },
        },
        "development_matrix": {
            "family_order": list(FAMILIES),
            "family_count": len(FAMILIES),
            "units_per_fold": EXPECTED_FOLD_UNITS,
            "total_units": EXPECTED_TOTAL_UNITS,
            "final_families": [],
        },
        "result_namespace_snapshot_sha256": result_snapshot_sha256,
        "runtime_result_namespace_snapshot": _json_value(actual_snapshot),
        "folds": family_records,
    }
    # Use the same canonical digest recipe used by adjacent scientific artifacts.
    body["identity_sha256"] = _sha256(canonical_json_bytes(body))
    payload = canonical_json_bytes(body) + b"\n"
    # Close the capture window: a mutation during serialization invalidates the
    # candidate bytes instead of publishing a stale identity statement.
    recheck_screening_runtime_readonly(runtime)
    final_snapshot = tuple(
        (fold.store.run_id, _capture_fold_namespaces(fold)[0])
        for fold in runtime.folds
    )
    if final_snapshot != actual_snapshot:
        _fail("Phase 2 result namespace changed while identity bytes were assembled")
    _canonical_selection_lock(
        runtime,
        selection_lock_sha256,
        expected_parent_identity=selection_parent_identity,
        expected_file_identity=selection_identity,
    )
    return payload


__all__ = ["publish_phase2_result_namespace_identity"]
