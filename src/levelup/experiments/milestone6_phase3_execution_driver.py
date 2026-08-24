"""Resumable, development-only Phase 3 execution orchestration.

This module is the narrow boundary between the frozen Phase 3 authorities and
the one-unit executor.  It validates the source and prepared result tree,
holds one readiness lease, activates all six stores transactionally, and then
walks the exact 11,520-unit matrix.  It intentionally contains no reducer,
aggregate, selection, or final-family path.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from levelup.experiments.milestone6_phase3_execution import (
    Phase3ExecutionContext,
    execute_phase3_unit,
)
from levelup.experiments.milestone6_phase3_execution_gate import (
    Phase3ActivationError,
    phase3_activation,
)
from levelup.experiments.milestone6_phase3_model_authority import (
    Phase3ModelArtifactAuthority,
    load_phase3_model_artifact_authority_bytes,
)
from levelup.experiments.milestone6_phase3_plan import (
    FAMILIES,
    Phase3PlannedUnit,
    ValidatedPhase3Plan,
    bind_validated_phase3_plan,
    validate_phase3_plan_lock_bytes,
)
from levelup.experiments.milestone6_phase3_protocol import ROOT
from levelup.experiments.milestone6_phase3_readiness import (
    PHASE3_MODEL_AUTHORITY_RELATIVE,
    Phase3ReadinessError,
    Phase3ReadinessSnapshot,
    capture_phase3_readiness,
)
from levelup.experiments.milestone6_phase3_result_store import (
    EXPECTED_FAMILY_UNIT_COUNT,
    EXPECTED_TOTAL_UNIT_COUNT,
    Phase3ExpectedPlan,
    Phase3ResultStore,
    Phase3ResultStoreError,
    build_phase3_expected_plan,
    load_phase3_result_stores,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.provenance import utc_now
from levelup.experiments.runner.records import (
    AttemptRecord,
    UnitPayload,
    UnitRecord,
)

RESULT_ROOT_MARKER = "phase3-activation.json"
EXPECTED_TOTAL_UNITS = EXPECTED_TOTAL_UNIT_COUNT
EXPECTED_UNITS_PER_FAMILY = EXPECTED_FAMILY_UNIT_COUNT
_METADATA_FILES = frozenset(("config.json", "expected-units.json", "run.json"))
_STAGE_EXECUTION = "execution"
_STAGE_PAYLOAD = "payload-validation"
_STAGE_RECORD = "record-construction"
_STAGE_PUBLICATION = "record-publication"


class Phase3ExecutionDriverError(RuntimeError):
    """Raised when the complete development matrix cannot be executed safely."""


def _fail(message: str, exc: BaseException | None = None) -> None:
    if exc is None:
        raise Phase3ExecutionDriverError(message)
    raise Phase3ExecutionDriverError(message) from exc


def _canonical_repository(value: str | os.PathLike[str], *, label: str) -> Path:
    """Resolve an existing directory only after rejecting lexical symlinks."""

    lexical = Path(value).absolute()
    for candidate in (lexical, *lexical.parents):
        try:
            if os.path.lexists(candidate) and candidate.is_symlink():
                _fail(f"Phase 3 {label} repository or ancestor is a symlink")
        except OSError as exc:
            _fail(f"cannot inspect Phase 3 {label} repository", exc)
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        _fail(f"Phase 3 {label} repository does not exist", exc)
    if not resolved.is_dir():
        _fail(f"Phase 3 {label} repository is not a directory")
    return resolved


def _canonical_result_root(value: str | os.PathLike[str]) -> Path:
    """Require an already-created, real result root (never mkdir here)."""

    lexical = Path(value).absolute()
    for candidate in (lexical, *lexical.parents):
        try:
            if os.path.lexists(candidate) and candidate.is_symlink():
                _fail("Phase 3 result root or ancestor is a symlink")
        except OSError as exc:
            _fail("cannot inspect Phase 3 result root", exc)
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        _fail("Phase 3 result root must already exist", exc)
    if not resolved.is_dir():
        _fail("Phase 3 result root is not a directory")
    return resolved


def _read_repository_bytes(repository: Path, relative: str) -> bytes:
    """Read one repository-relative regular file through pinned descriptors."""

    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail("Phase 3 authority path is unsafe")
    try:
        with ExitStack() as stack:
            root_fd = secure_fs.open_directory_chain(repository)
            stack.callback(os.close, root_fd)
            parent_fd = root_fd
            for component in pure.parts[:-1]:
                parent_fd = secure_fs.open_child_directory(parent_fd, component)
                stack.callback(os.close, parent_fd)
            with secure_fs.open_regular_file_at(parent_fd, pure.parts[-1]) as file_fd:
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(file_fd, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                return b"".join(chunks)
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        _fail(f"cannot read Phase 3 authority file: {relative}", exc)
    raise AssertionError("unreachable")


def _load_authorities(
    repository: Path,
) -> tuple[ValidatedPhase3Plan, Phase3ModelArtifactAuthority, Phase3ExpectedPlan]:
    """Load the compact committed plan and model authority exactly once."""

    try:
        plan_bytes = _read_repository_bytes(repository, "configs/milestone6/phase3_plan_lock.json")
        plan = validate_phase3_plan_lock_bytes(plan_bytes)
        validated = bind_validated_phase3_plan(plan)
        authority_bytes = _read_repository_bytes(repository, PHASE3_MODEL_AUTHORITY_RELATIVE)
        authority = load_phase3_model_artifact_authority_bytes(authority_bytes)
        expected = build_phase3_expected_plan(validated, authority)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, Phase3ExecutionDriverError):
            raise
        _fail("Phase 3 plan/model authority cannot be loaded", exc)
    if expected.final_family_access or tuple(expected.family_order) != tuple(FAMILIES):
        _fail("Phase 3 expected plan is not the frozen development matrix")
    if len(expected.stores) != len(FAMILIES) or len(expected.units) != EXPECTED_TOTAL_UNIT_COUNT:
        _fail("Phase 3 expected plan does not contain exactly 11,520 units")
    if any(len(store.units) != EXPECTED_FAMILY_UNIT_COUNT for store in expected.stores):
        _fail("Phase 3 expected plan family partition is incomplete")
    return validated, authority, expected


def _entry_kind(path: Path) -> str:
    try:
        value = path.lstat()
    except OSError as exc:
        _fail(f"cannot inspect prepared Phase 3 entry: {path.name}", exc)
    if stat.S_ISLNK(value.st_mode):
        _fail(f"prepared Phase 3 namespace contains a symlink: {path.name}")
    if stat.S_ISDIR(value.st_mode):
        return "directory"
    if stat.S_ISREG(value.st_mode):
        return "file"
    _fail(f"prepared Phase 3 entry is not regular: {path.name}")
    raise AssertionError("unreachable")


def _require_existing_store_tree(
    root: Path,
    expected: Phase3ExpectedPlan,
) -> bool:
    """Validate that all six stores pre-existed before idempotent loading."""

    try:
        root_names = {entry.name for entry in root.iterdir()}
    except OSError as exc:
        _fail("cannot enumerate the prepared Phase 3 result root", exc)
    marker_present = RESULT_ROOT_MARKER in root_names
    allowed = set(FAMILIES) | ({RESULT_ROOT_MARKER} if marker_present else set())
    if root_names != allowed:
        _fail("prepared Phase 3 result root has missing, foreign, or temporary entries")
    if marker_present and _entry_kind(root / RESULT_ROOT_MARKER) != "file":
        _fail("Phase 3 activation marker is not a regular file")
    for family, spec in zip(FAMILIES, expected.stores, strict=True):
        family_path = root / family
        if _entry_kind(family_path) != "directory":
            _fail(f"prepared Phase 3 family is not a directory: {family}")
        family_names = {entry.name for entry in family_path.iterdir()}
        if family_names != {spec.run_id}:
            _fail(f"prepared Phase 3 family namespace drifted: {family}")
        run_path = family_path / spec.run_id
        if _entry_kind(run_path) != "directory":
            _fail(f"prepared Phase 3 run directory is not a directory: {family}")
        names = {entry.name for entry in run_path.iterdir()}
        if names != set(_METADATA_FILES) | {"units", "attempts"}:
            _fail(f"prepared Phase 3 run namespace drifted: {family}")
        for name in _METADATA_FILES:
            if _entry_kind(run_path / name) != "file":
                _fail(f"prepared Phase 3 metadata is not a file: {family}/{name}")
        for name in ("units", "attempts"):
            directory = run_path / name
            if _entry_kind(directory) != "directory":
                _fail(f"prepared Phase 3 namespace is not a directory: {family}/{name}")
            if not marker_present:
                try:
                    if any(directory.iterdir()):
                        _fail("orphan Phase 3 records require an activation marker")
                except OSError as exc:
                    _fail("cannot inspect prepared Phase 3 records", exc)
    return marker_present


def _load_prepared_stores(
    root: Path,
    validated: ValidatedPhase3Plan,
    authority: Phase3ModelArtifactAuthority,
    expected: Phase3ExpectedPlan,
) -> tuple[Phase3ResultStore, ...]:
    _require_existing_store_tree(root, expected)
    try:
        stores = tuple(load_phase3_result_stores(root, validated, authority))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, Phase3ExecutionDriverError):
            raise
        _fail("prepared Phase 3 result stores cannot be loaded", exc)
    if tuple(store.family_id for store in stores) != tuple(FAMILIES):
        _fail("prepared Phase 3 stores are not in canonical family order")
    return stores


def _retryable(exception: BaseException) -> bool:
    """Only transient execution failures are retryable; authority drift is not."""

    return not isinstance(
        exception,
        (
            Phase3ActivationError,
            Phase3ReadinessError,
            Phase3ResultStoreError,
            ValidationError,
            ValueError,
            TypeError,
        ),
    )


def _attempt(
    family: Any,
    planned: Phase3PlannedUnit,
    exception: BaseException,
    *,
    attempt_number: int,
    stage: str,
    retryable: bool,
    started_at: datetime,
    elapsed: float,
    status: str = "failed",
) -> None:
    if status not in {"failed", "interrupted"}:
        _fail("invalid Phase 3 attempt status")
    try:
        record = AttemptRecord(
            run_id=family.run_id,
            config_sha256=family.config_sha256,
            unit_id=planned.unit.unit_id,
            attempt=attempt_number,
            key=planned.unit.key,
            seeds=planned.unit.seeds,
            status=status,  # type: ignore[arg-type]
            stage=stage,
            exception_type=type(exception).__name__,
            sanitized_message=f"{stage} raised {type(exception).__name__}",
            retryable=retryable,
            started_at_utc=started_at,
            finished_at_utc=utc_now(),
            elapsed_wall_seconds=max(0.0, elapsed),
        )
        family.write_attempt(record)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _fail("Phase 3 failure attempt could not be durably recorded", exc)


def _execute_loop(
    batch: Any,
    validated: ValidatedPhase3Plan,
    authority: Phase3ModelArtifactAuthority,
    expected: Phase3ExpectedPlan,
    model_root: Path,
) -> dict[str, int | bool]:
    stores = tuple(batch.stores)
    if tuple(item.family_id for item in stores) != tuple(FAMILIES):
        _fail("activated Phase 3 stores are not in canonical family order")
    family_by_id = {store.family_id: store for store in stores}
    context = Phase3ExecutionContext.canonical(authority, validated, model_root)
    completed_by_family = {
        family_id: set(family_by_id[family_id].completed_unit_ids())
        for family_id in FAMILIES
    }
    latest_attempt_by_unit: dict[str, AttemptRecord] = {}
    for family_id in FAMILIES:
        for attempt in family_by_id[family_id].attempt_records():
            previous = latest_attempt_by_unit.get(attempt.unit_id)
            if previous is None or attempt.attempt > previous.attempt:
                latest_attempt_by_unit[attempt.unit_id] = attempt
    next_attempt_by_unit = {
        unit_id: attempt.attempt + 1
        for unit_id, attempt in latest_attempt_by_unit.items()
    }
    counts = {"completed": 0, "skipped": 0, "failed": 0, "interrupted": 0}
    for planned in expected.units:
        family = family_by_id.get(planned.heldout_family)
        if family is None:
            _fail("Phase 3 planned unit has no activated family store")
        if planned.unit.unit_id in completed_by_family[planned.heldout_family]:
            counts["skipped"] += 1
            continue
        prior = latest_attempt_by_unit.get(planned.unit.unit_id)
        if prior is not None and not prior.retryable:
            _fail("non-retryable Phase 3 attempt leaves an incomplete unit")
        started_at = utc_now()
        started = time.perf_counter()
        stage = _STAGE_EXECUTION
        try:
            raw_payload = execute_phase3_unit(context, planned)
            stage = _STAGE_PAYLOAD
            payload_input = (
                raw_payload.model_dump(mode="json", warnings=False)
                if isinstance(raw_payload, UnitPayload)
                else raw_payload
            )
            payload = UnitPayload.model_validate(payload_input)
            stage = _STAGE_RECORD
            record = UnitRecord(
                run_id=family.run_id,
                config_sha256=family.config_sha256,
                unit_id=planned.unit.unit_id,
                key=planned.unit.key,
                seeds=planned.unit.seeds,
                exposure_manifest_sha256=planned.unit.exposure_manifest_sha256,
                started_at_utc=started_at,
                finished_at_utc=utc_now(),
                elapsed_wall_seconds=max(0.0, time.perf_counter() - started),
                outcome=payload.outcome,
                accounting=payload.accounting,
                shared_artifact=payload.shared_artifact,
                shared_artifacts=payload.shared_artifacts,
                candidate_generation_sha256=payload.candidate_generation_sha256,
                history_shuffle_permutation_map_sha256=payload.history_shuffle_permutation_map_sha256,
                diagnostics=payload.diagnostics,
            )
            stage = _STAGE_PUBLICATION
            if family.write_completed(record):
                counts["completed"] += 1
            else:
                counts["skipped"] += 1
            completed_by_family[planned.heldout_family].add(planned.unit.unit_id)
        except KeyboardInterrupt as exc:
            _attempt(
                family,
                planned,
                exc,
                attempt_number=next_attempt_by_unit.get(planned.unit.unit_id, 1),
                stage=stage,
                retryable=True,
                started_at=started_at,
                elapsed=time.perf_counter() - started,
                status="interrupted",
            )
            counts["interrupted"] += 1
            raise
        except Exception as exc:
            _attempt(
                family,
                planned,
                exc,
                attempt_number=next_attempt_by_unit.get(planned.unit.unit_id, 1),
                stage=stage,
                retryable=_retryable(exc),
                started_at=started_at,
                elapsed=time.perf_counter() - started,
            )
            counts["failed"] += 1
            raise
    expected_by_family = {
        store.family_id: {item.unit.unit_id for item in store.units}
        for store in expected.stores
    }
    missing = sum(
        len(expected_by_family[family_id] - completed_by_family[family_id])
        for family_id in FAMILIES
    )
    if missing:
        _fail(f"Phase 3 execution finished with {missing} missing units")
    counts["complete"] = True
    return counts


def run_phase3_development(
    authority_repository: str | os.PathLike[str],
    result_root: str | os.PathLike[str],
    *,
    expected_git_commit: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate or execute the complete frozen Phase 3 development matrix."""

    repository = _canonical_repository(authority_repository, label="authority")
    try:
        if repository != Path(ROOT).resolve(strict=True):
            _fail("Phase 3 authority repository is not the canonical source repository")
    except (OSError, RuntimeError) as exc:
        _fail("canonical Phase 3 source repository is unavailable", exc)
    root = _canonical_result_root(result_root)
    if not isinstance(expected_git_commit, str) or not expected_git_commit:
        _fail("Phase 3 execution requires an explicit expected git commit")
    validated, authority, expected = _load_authorities(repository)
    derived_model_root = repository / "runs" / "milestone6" / authority.artifact_store_id
    if not derived_model_root.is_dir() or derived_model_root.is_symlink():
        _fail("published Phase 3 model root is unavailable or symlinked")
    stores = _load_prepared_stores(root, validated, authority, expected)
    snapshot: Phase3ReadinessSnapshot
    try:
        snapshot = capture_phase3_readiness(
            repository,
            model_store_root=derived_model_root,
            execution_preflight=True,
            expected_git_commit=expected_git_commit,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, Phase3ExecutionDriverError):
            raise
        _fail("Phase 3 readiness preflight failed", exc)
    result: dict[str, Any] = {
        "dry_run": bool(dry_run),
        "plan_id": expected.plan_id,
        "protocol_sha256": expected.protocol_sha256,
        "model_authority_sha256": expected.model_authority_sha256,
        "expected_total": EXPECTED_TOTAL_UNIT_COUNT,
    }
    if dry_run:
        result.update({"completed": 0, "skipped": 0, "failed": 0, "interrupted": 0, "complete": False})
        return result
    try:
        with snapshot.hold_for_activation(expected_git_commit=expected_git_commit) as lease:
            with phase3_activation(stores, expected, lease, expected_git_commit=expected_git_commit) as batch:
                counts = _execute_loop(
                    batch,
                    validated,
                    authority,
                    expected,
                    derived_model_root,
                )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, Phase3ExecutionDriverError):
            raise
        _fail("Phase 3 development execution failed closed", exc)
    result.update(counts)
    return result


run_phase3_execution = run_phase3_development


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-repository", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = run_phase3_development(
        args.authority_repository,
        args.result_root,
        expected_git_commit=args.expected_git_commit,
        dry_run=args.validate_only,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_FAMILY_UNIT_COUNT",
    "EXPECTED_TOTAL_UNIT_COUNT",
    "EXPECTED_TOTAL_UNITS",
    "EXPECTED_UNITS_PER_FAMILY",
    "Phase3ExecutionDriverError",
    "main",
    "run_phase3_development",
    "run_phase3_execution",
]
