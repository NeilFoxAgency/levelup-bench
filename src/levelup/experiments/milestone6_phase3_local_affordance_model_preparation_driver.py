"""Prepare exactly one frozen development model owner, with no task execution.

This driver joins the committed Phase 2 publication, current Phase 3 authority,
and pinned raw-development lease, then loads one optimum-evidence bundle and
stores one trained model.  It has no search, evaluator, oracle, unit-runner, or
final-family path.  ``validate_only`` performs the same authority preflight but
does not issue a training capability, train, or create output directories.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

from levelup.experiments.milestone6_phase2_screening_provenance import (
    CANONICAL_READINESS_PATH,
)
from levelup.experiments.milestone6_phase2_screening_runtime import (
    ScreeningRuntime,
    load_screening_runtime_metadata_only,
    recheck_screening_runtime_metadata_only,
)
from levelup.experiments.milestone6_phase3_anchor import (
    PHASE3_ANCHOR_MANIFEST_PATH,
    load_committed_phase3_anchor_manifest_bytes,
    validate_committed_phase3_anchor_for_local_affordance_preparation,
)
from levelup.experiments.milestone6_phase3_evidence import (
    validate_phase3_evidence_lock_bytes,
)
from levelup.experiments.milestone6_phase3_local_affordance_model_records import (
    build_model_record,
)
from levelup.experiments.milestone6_phase3_local_affordance_model_store import (
    serialize_local_affordance_model,
    write_local_affordance_model,
)
from levelup.experiments.milestone6_phase3_local_affordance_models import (
    prepare_local_affordance_model,
)
from levelup.experiments.milestone6_phase3_local_affordance_plan import (
    EXPECTED_COUNTS,
    LocalAffordanceModelOwner,
    LocalAffordancePlan,
    validate_local_affordance_plan,
)
from levelup.experiments.milestone6_phase3_local_affordance_preparation_readiness import (
    LocalAffordancePreparationSnapshot,
    capture_local_affordance_preparation_readiness,
    require_local_affordance_preparation_snapshot,
)
from levelup.experiments.milestone6_phase3_local_affordance_training_source import (
    load_local_affordance_training_source,
)
from levelup.experiments.milestone6_phase3_local_affordance_training_views import (
    build_local_affordance_training_view,
)
from levelup.experiments.milestone6_phase3_plan import (
    bind_validated_phase3_plan,
    validate_phase3_plan_lock_bytes,
)
from levelup.experiments.milestone6_phase3_protocol import PHASE3_PROTOCOL_PATH
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import DevicePolicy
from levelup.experiments.runner.provenance import (
    apply_runtime_policy,
    capture_system_provenance,
)
from levelup.experiments.runner.storage import provenance_identity_sha256

EXPECTED_MODEL_OWNERS = EXPECTED_COUNTS["model_owners"]
MODEL_STORE_RELATIVE_PATH = Path("runs/milestone6/phase3-local-affordance-models")
RESULT_SNAPSHOT_SHA256 = "537039da0e4d3bf1442f3bc6ffe2e513b2d2ffe4f010f3426cc1c63c2448b730"
SELECTION_LOCK_SHA256 = "499910276af4359ce3c295933c4ae81293d8a23434f712e310921d05b626f06c"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_SOURCE_ROOT = PHASE3_PROTOCOL_PATH.parents[2]


class LocalAffordanceModelPreparationDriverError(RuntimeError):
    """Raised when exact one-owner development authority cannot be established."""


def _fail(message: str, exc: BaseException | None = None) -> None:
    error = LocalAffordanceModelPreparationDriverError(message)
    if exc is None:
        raise error
    raise error from exc


def _git_state(repository: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=repository, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            cwd=repository, check=True, capture_output=True, text=True,
        ).stdout)
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail("cannot establish clean repository provenance", exc)
    if _COMMIT.fullmatch(commit) is None or set(commit) == {"0"}:
        _fail("repository commit identity is malformed")
    return commit, dirty


def _repository(path: str | Path, label: str) -> Path:
    lexical = Path(os.path.abspath(path))
    for candidate in (lexical, *lexical.parents):
        try:
            observed = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(observed.st_mode):
            _fail(f"{label} repository or an ancestor is a symlink")
    try:
        result = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        _fail(f"{label} repository is unavailable", exc)
    if not result.is_dir():
        _fail(f"{label} repository is not a directory")
    return result


def _read_regular(path: Path) -> bytes:
    """Read one regular file with no-follow on its pinned parent descriptor."""
    try:
        parent_fd = secure_fs.open_directory_chain(path.parent)
        try:
            return secure_fs.read_bytes_at(parent_fd, path.name)
        finally:
            os.close(parent_fd)
    except (OSError, RuntimeError, ValueError) as exc:
        _fail(f"cannot safely read committed authority file: {path.name}", exc)


def _read_pinned_result_snapshot(path: str | Path) -> bytes:
    """Read only the exact-hash published administrative namespace artifact."""
    target = Path(os.path.abspath(path))
    for candidate in (target, *target.parents):
        try:
            observed = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(observed.st_mode):
            _fail("result snapshot or one of its ancestors is a symlink")
    content = _read_regular(target)
    if hashlib.sha256(content).hexdigest() != RESULT_SNAPSHOT_SHA256:
        _fail("administrative result snapshot bytes differ from the exact frozen SHA-256")
    return content


def _assert_output_root(
    output_root: str | Path,
    *,
    authority: Path,
    raw_root: Path,
    runtime: ScreeningRuntime | None,
) -> Path:
    output = Path(os.path.abspath(output_root))
    for candidate in (output, *output.parents):
        try:
            observed = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(observed.st_mode):
            _fail("model output root or an ancestor is a symlink")
        if candidate != output and not stat.S_ISDIR(observed.st_mode):
            _fail("model output ancestry contains a non-directory")
    resolved = output.resolve(strict=False)
    expected = (authority / MODEL_STORE_RELATIVE_PATH).resolve(strict=False)
    if resolved != expected:
        _fail("model output root must be the dedicated canonical owner store")
    if not resolved.parent.is_dir():
        _fail("dedicated model-store parent must already exist")
    forbidden = [raw_root.resolve(strict=True)]
    if runtime is not None:
        forbidden.extend(fold.store.run_dir.resolve(strict=True) for fold in runtime.folds)
    for root in forbidden:
        if resolved == root or resolved in root.parents or root in resolved.parents:
            _fail("model output overlaps raw evidence or a screening result namespace")
    return resolved


def _check_runtime_policy(repository: Path, expected_commit: str) -> tuple[Any, str]:
    policy = DevicePolicy(
        requested_device="cpu", torch_threads=1, torch_interop_threads=1
    )
    try:
        resolved = apply_runtime_policy(policy)
        provenance = capture_system_provenance(repository, policy)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _fail("cannot establish model preparation runtime provenance", exc)
    if (
        resolved != "cpu"
        or provenance.git_dirty
        or provenance.git_commit_sha != expected_commit
        or provenance.requested_device != "cpu"
        or provenance.resolved_device != "cpu"
        or provenance.requested_torch_threads != 1
        or provenance.actual_torch_threads != 1
        or provenance.requested_torch_interop_threads != 1
        or provenance.actual_torch_interop_threads != 1
        or provenance.processes != 1
    ):
        _fail("model preparation requires clean CPU one-thread one-process provenance")
    return provenance, provenance_identity_sha256(provenance)


def _validated_evidence_lock(authority: Path, runtime: ScreeningRuntime, snapshot: LocalAffordancePreparationSnapshot):
    try:
        old_plan_path = authority / "configs/milestone6/phase3_plan_lock.json"
        old_anchor_path = authority / "configs/milestone6" / PHASE3_ANCHOR_MANIFEST_PATH.name
        plan_bytes = _read_regular(old_plan_path)
        validated_old_plan = bind_validated_phase3_plan(
            validate_phase3_plan_lock_bytes(plan_bytes)
        )
        anchor_bytes = load_committed_phase3_anchor_manifest_bytes(old_anchor_path)
        evidence_source = snapshot.source_by_path[
            "configs/milestone6/phase3_evidence_lock.json"
        ]
        if hashlib.sha256(evidence_source.content).hexdigest() != evidence_source.sha256:
            _fail("captured evidence-lock bytes failed their source digest")
        anchor = validate_committed_phase3_anchor_for_local_affordance_preparation(
            anchor_bytes,
            runtime=runtime,
            evidence_lock_bytes=evidence_source.content,
        )
        return validate_phase3_evidence_lock_bytes(
            evidence_source.content,
            runtime=runtime,
            validated_plan=validated_old_plan,
            anchor_manifest=anchor,
            anchor_file_bytes=anchor_bytes,
            plan_lock_bytes=plan_bytes,
        )
    except LocalAffordanceModelPreparationDriverError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        _fail("current Phase 3 evidence lock cannot be revalidated against screening publication", exc)


def run_local_affordance_model_preparation(
    manifest_path: str | Path,
    manifest_bytes_sha256: str,
    raw_root: str | Path,
    screening_repository: str | Path,
    authority_repository: str | Path,
    output_root: str | Path,
    *,
    result_snapshot_path: str | Path,
    owner_id: str,
    validate_only: bool = False,
) -> dict[str, Any]:
    """Validate authority and optionally prepare exactly one development owner.

    ``owner_id`` is mandatory: there is intentionally no batch/limit/default
    owner selection.  A successful validate-only response is a read-only
    preflight, not authorization for comparative execution or a 480-owner run.
    """
    if type(validate_only) is not bool:
        _fail("validate_only must be a boolean")
    if not isinstance(owner_id, str) or _HEX64.fullmatch(owner_id) is None:
        _fail("exact lowercase frozen development owner_id is required")
    if not isinstance(manifest_bytes_sha256, str) or _HEX64.fullmatch(manifest_bytes_sha256) is None:
        _fail("Phase 2 publication manifest byte hash is required")

    screening = _repository(screening_repository, "screening")
    authority = _repository(authority_repository, "current authority")
    raw = Path(os.path.abspath(raw_root))
    if screening == authority:
        _fail("screening publication and current authority must be separate repositories")
    if authority != _SOURCE_ROOT.resolve(strict=True):
        _fail(
            "current authority must match the repository whose frozen Phase 3 validators are loaded"
        )
    canonical_manifest = screening / CANONICAL_READINESS_PATH
    if Path(os.path.abspath(manifest_path)) != canonical_manifest:
        _fail("driver requires the canonical committed Phase 2 publication manifest")
    committed_manifest_bytes = _read_regular(canonical_manifest)
    if hashlib.sha256(committed_manifest_bytes).hexdigest() != manifest_bytes_sha256:
        _fail("Phase 2 publication manifest bytes differ from the supplied exact hash")

    screening_commit, screening_dirty = _git_state(screening)
    authority_commit, authority_dirty = _git_state(authority)
    if screening_dirty or authority_dirty:
        _fail("both screening publication and current authority repositories must be clean")
    if _COMMIT.fullmatch(screening_commit) is None:
        _fail("screening publication commit identity is malformed")
    result_snapshot_bytes = _read_pinned_result_snapshot(result_snapshot_path)

    try:
        snapshot = require_local_affordance_preparation_snapshot(
            capture_local_affordance_preparation_readiness(
                repository=authority, raw_root=raw
            )
        )
        if snapshot.git_commit_sha != authority_commit:
            _fail("current authority snapshot commit differs from clean repository HEAD")
        plan = snapshot.plan
        if type(plan) is not LocalAffordancePlan:
            _fail("current development plan is not validator-issued")
        validate_local_affordance_plan(plan, repository=authority)
        if (
            plan.final_family_access
            or len(plan.model_owners) != EXPECTED_MODEL_OWNERS
            or len(plan.units) != EXPECTED_COUNTS["units"]
        ):
            _fail("current authority is not the exact development-only owner plan")
        matches = tuple(item for item in plan.model_owners if item.owner_id == owner_id)
        if len(matches) != 1 or type(matches[0]) is not LocalAffordanceModelOwner:
            _fail("selected owner is absent from the frozen development plan")
        owner = matches[0]
        store_root = _assert_output_root(
            output_root, authority=authority, raw_root=raw, runtime=None
        )
        runtime = load_screening_runtime_metadata_only(
            canonical_manifest,
            raw,
            screening,
            manifest_bytes_sha256=manifest_bytes_sha256,
            result_snapshot_bytes=result_snapshot_bytes,
            selection_lock_sha256=SELECTION_LOCK_SHA256,
            authority_repository=authority,
        )
        recheck_screening_runtime_metadata_only(runtime)
        store_root = _assert_output_root(
            store_root,
            authority=authority,
            raw_root=raw,
            runtime=runtime,
        )
        evidence_lock = _validated_evidence_lock(authority, runtime, snapshot)
        if validate_only:
            snapshot.recheck(expected_git_commit=authority_commit)
            recheck_screening_runtime_metadata_only(runtime)
            return {
                "schema_version": "milestone6.phase3.local-affordance-model-preparation-preflight.v1",
                "status": "validated-only",
                "plan_id": plan.plan_id,
                "owner_id": owner.owner_id,
                "fold_id": owner.fold_id,
                "replicate": owner.replicate,
                "output_root": str(store_root),
                "training_performed": False,
                "comparative_units_run": False,
                "final_family_access": False,
                "expected_owner_count": EXPECTED_MODEL_OWNERS,
                "full_matrix_authorized": False,
            }

        _, provenance_sha256 = _check_runtime_policy(authority, authority_commit)
        prepared = None
        entry = None
        with snapshot.activation(expected_git_commit=authority_commit) as lease:
            lease.require_active()
            fold = lease.training_fold_manifest(owner.fold_id, owner.replicate)
            capability = lease.issue_training_fold_probe_capability(
                fold_id=owner.fold_id, replicate=owner.replicate
            )
            source_bundle = load_local_affordance_training_source(
                runtime,
                evidence_lock,
                family_id=owner.heldout_family,
                replicate=owner.replicate,
            )
            view = next((item for item in plan.views if item.view_id == owner.view_id), None)
            if view is None:
                _fail("selected owner view is absent from the frozen plan")
            training_view = build_local_affordance_training_view(
                view=view,
                fold=fold,
                evidence_lock=evidence_lock,
                source_bundle=source_bundle,
                probe_capability=capability,
            )
            lease.require_active()
            prepared = prepare_local_affordance_model(
                plan=plan, view=view, owner=owner, training_view=training_view
            )
            model_bytes = serialize_local_affordance_model(plan, prepared)
            record = build_model_record(
                plan=plan,
                owner_id=owner.owner_id,
                training_examples=prepared.report.training_examples,
                training_examples_sha256=prepared.training_examples_sha256,
                model_state_sha256=prepared.model_state_sha256,
                model_identity_sha256=prepared.model_identity_sha256,
                artifact_bytes=len(model_bytes),
                artifact_sha256=hashlib.sha256(model_bytes).hexdigest(),
                preparation_git_commit_sha=authority_commit,
                preparation_provenance_sha256=provenance_sha256,
            )
            lease.require_active()
            entry = write_local_affordance_model(
                store_root,
                plan,
                prepared,
                record,
                preparation_git_commit_sha=authority_commit,
                preparation_provenance_sha256=provenance_sha256,
            )
            lease.require_active()
        recheck_screening_runtime_metadata_only(runtime)
        snapshot.recheck(expected_git_commit=authority_commit)
        final_provenance = capture_system_provenance(authority, DevicePolicy(
            requested_device="cpu", torch_threads=1, torch_interop_threads=1
        ))
        if (
            final_provenance.git_dirty
            or final_provenance.git_commit_sha != authority_commit
            or provenance_identity_sha256(final_provenance) != provenance_sha256
        ):
            _fail("repository/runtime provenance changed during owner preparation")
        return {
            "schema_version": "milestone6.phase3.local-affordance-model-preparation-result.v1",
            "status": "owner-prepared",
            "plan_id": plan.plan_id,
            "owner_id": owner.owner_id,
            "fold_id": owner.fold_id,
            "replicate": owner.replicate,
            "output_root": str(store_root),
            "record_sha256": entry.record_sha256,
            "model_identity_sha256": prepared.model_identity_sha256,
            "preparation_provenance_sha256": provenance_sha256,
            "training_performed": True,
            "comparative_units_run": False,
            "final_family_access": False,
            "expected_owner_count": EXPECTED_MODEL_OWNERS,
            "full_matrix_authorized": False,
        }
    except LocalAffordanceModelPreparationDriverError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        _fail("development-only one-owner model preparation failed closed", exc)


__all__ = [
    "EXPECTED_MODEL_OWNERS",
    "MODEL_STORE_RELATIVE_PATH",
    "LocalAffordanceModelPreparationDriverError",
    "run_local_affordance_model_preparation",
]
