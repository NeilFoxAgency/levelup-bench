"""Preparation-only orchestration for the frozen outcome diagnostic models.

The driver is intentionally narrower than an experiment runner.  It loads the
descriptor-pinned Phase 2 development runtime, reconstructs the committed
outcome plan, reads the exact thirty evidence payloads, and delegates only model
construction to the resumable batch.  No environment, evaluator, oracle,
search, result, or final-family interface is reachable from this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from levelup.experiments.milestone6_phase2_screening_provenance import (
    CANONICAL_READINESS_PATH,
)
from levelup.experiments.milestone6_phase2_screening_runtime import (
    load_screening_runtime,
    recheck_screening_runtime_readonly,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
    OUTCOME_ARTIFACT_STORE_PREFIX,
    PinnedOutcomeTrainingEvidence,
    outcome_artifact_store_id,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_batch import (
    prepare_outcome_diagnostic_model_batch,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    EXPECTED_MODEL_OWNERS,
    bind_validated_outcome_diagnostic_plan,
    build_outcome_group_diagnostic_plan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    load_outcome_group_diagnostic_protocol,
)
from levelup.experiments.runner.config import DevicePolicy, canonical_json_bytes
from levelup.experiments.runner.provenance import (
    apply_runtime_policy,
    capture_system_provenance,
)
from levelup.experiments.runner.storage import provenance_identity_sha256
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataEvidenceKey,
    TrainingDataEvidenceManifest,
    load_training_data_evidence_payload_bundle_from_at,
    open_training_data_reader,
)

EXPECTED_UNITS = 5_760
MODEL_OUTPUT_PREFIX = f"runs/milestone6/{OUTCOME_ARTIFACT_STORE_PREFIX}"


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _validate_digest(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} is not a lowercase SHA-256 digest")


def _validate_commit(value: str) -> None:
    if (
        not isinstance(value, str)
        or not 40 <= len(value) <= 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
        or set(value) == {"0"}
    ):
        _fail("expected preparation commit is not a nonzero lowercase git SHA")


def _validate_preparation_provenance(
    provenance: Any, *, expected_commit: str, expected_identity: str | None = None
) -> str:
    if provenance.git_dirty or provenance.git_commit_sha != expected_commit:
        _fail("authority repository is dirty or differs from expected preparation commit")
    if (
        provenance.requested_device != "cpu"
        or provenance.resolved_device != "cpu"
        or provenance.requested_torch_threads != 1
        or provenance.actual_torch_threads != 1
        or provenance.requested_torch_interop_threads != 1
        or provenance.actual_torch_interop_threads != 1
        or provenance.processes != 1
    ):
        _fail("model preparation provenance is not clean CPU one-thread one-process")
    observed = provenance_identity_sha256(provenance)
    if expected_identity is not None and observed != expected_identity:
        _fail("model preparation provenance changed during preparation")
    return observed


def _reject_repository(path: str | Path, label: str) -> Path:
    lexical = Path(path).absolute()
    for candidate in (lexical, *lexical.parents):
        if os.path.lexists(candidate) and candidate.is_symlink():
            _fail(f"Phase 3 {label} repository or an ancestor is a symlink")
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Phase 3 {label} repository does not exist") from exc
    if not resolved.is_dir():
        _fail(f"Phase 3 {label} repository is not a directory")
    return resolved


def _reject_output(output_root: str | Path, forbidden: Iterable[str | Path]) -> Path:
    output = Path(output_root).absolute()
    for candidate in (output, *output.parents):
        if os.path.lexists(candidate) and candidate.is_symlink():
            _fail("Phase 3 diagnostic model output root or an ancestor is a symlink")
    resolved = output.resolve(strict=False)
    for raw in forbidden:
        root = Path(raw).absolute().resolve(strict=False)
        if resolved == root or resolved in root.parents or root in resolved.parents:
            _fail("Phase 3 diagnostic model output overlaps raw evidence or result output")
    return resolved


def _read_evidence(runtime: Any, plan: Any) -> dict[tuple[str, int], PinnedOutcomeTrainingEvidence]:
    folds = {str(fold.family_id): fold for fold in runtime.folds}
    if len(folds) != 6:
        _fail("Phase 2 runtime does not contain the exact six development folds")
    result: dict[tuple[str, int], PinnedOutcomeTrainingEvidence] = {}
    for raw in plan.plan.evidence_lineage_rows:
        try:
            row = json.loads(raw)
            key = TrainingDataEvidenceKey.model_validate(row["evidence_key"])
            manifest = TrainingDataEvidenceManifest.model_validate(row["evidence_manifest"])
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError("outcome evidence lineage row is not typed canonical JSON") from exc
        if canonical_json_bytes(row) != raw:
            _fail("outcome evidence lineage row is not canonical")
        family = row.get("family_id")
        replicate = row.get("replicate")
        if (
            not isinstance(family, str)
            or not isinstance(replicate, int)
            or isinstance(replicate, bool)
            or key.heldout_family_id != family
            or key.replicate != replicate
            or manifest.key != key
            or row.get("evidence_id") != manifest.evidence_id
            or row.get("evidence_key_id") != key.key_id
        ):
            _fail("outcome evidence lineage identity differs from its typed row")
        fold = folds.get(family)
        if fold is None:
            _fail("outcome evidence row references a foreign fold")
        try:
            with fold.store._open_pinned_run() as root_fd:
                with open_training_data_reader(root_fd) as reader:
                    bundle = load_training_data_evidence_payload_bundle_from_at(
                        reader, manifest.evidence_id, expected_key=key
                    )
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            raise RuntimeError("descriptor-pinned outcome evidence read failed") from exc
        if (
            hashlib.sha256(bundle.manifest_bytes).hexdigest()
            != row.get("canonical_manifest_bytes_sha256")
            or bundle.manifest != manifest
            or hashlib.sha256(bundle.payload_bytes).hexdigest() != manifest.payload_sha256
            or len(bundle.payload_bytes) != manifest.payload_bytes
            or tuple(sample.task_id for sample in bundle.payload.samples)
            != tuple(key.ordered_training_task_ids)
        ):
            _fail("descriptor-read outcome evidence differs from frozen lineage")
        pair = (family, replicate)
        if pair in result:
            _fail("outcome evidence lineage contains duplicate sources")
        result[pair] = PinnedOutcomeTrainingEvidence(bundle.payload, bundle.payload_bytes)
    if len(result) != 30:
        _fail("outcome evidence coverage is not exactly thirty sources")
    return result


def run_outcome_diagnostic_model_preparation(
    manifest_path: str | Path,
    manifest_bytes_sha256: str,
    raw_root: str | Path,
    screening_repository: str | Path,
    authority_repository: str | Path,
    output_root: str | Path,
    *,
    expected_preparation_commit_sha: str,
    owner_ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Prepare bounded outcome model owners and return deterministic progress JSON."""

    screening = _reject_repository(screening_repository, "screening")
    authority = _reject_repository(authority_repository, "authority")
    canonical_manifest = screening / CANONICAL_READINESS_PATH
    if Path(manifest_path).resolve(strict=False) != canonical_manifest:
        _fail("outcome driver requires the canonical committed readiness manifest")
    _validate_digest(manifest_bytes_sha256, "manifest byte pin")
    _validate_commit(expected_preparation_commit_sha)
    if owner_ids is not None and limit is not None:
        _fail("owner_ids and limit are mutually exclusive")
    if limit is not None and (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 0 <= limit <= EXPECTED_MODEL_OWNERS
    ):
        _fail("limit is outside the frozen 240-owner universe")
    selected = None if owner_ids is None else tuple(owner_ids)
    if selected is not None:
        if len(set(selected)) != len(selected) or any(
            not isinstance(owner, str)
            or len(owner) != 64
            or owner != owner.lower()
            or any(character not in "0123456789abcdef" for character in owner)
            for owner in selected
        ):
            _fail("owner_ids must be unique lowercase SHA-256 identities")

    # Reject unsafe output ancestry and raw-root overlap before opening any
    # runtime descriptors.  A second check below adds every pinned fold result
    # namespace once the runtime is loaded.
    _reject_output(output_root, (raw_root,))

    runtime = load_screening_runtime(
        manifest_path,
        raw_root,
        screening,
        manifest_bytes_sha256=manifest_bytes_sha256,
        authority_repository=authority,
    )
    recheck_screening_runtime_readonly(runtime)
    snapshot = load_outcome_group_diagnostic_protocol()
    snapshot_repository = getattr(snapshot, "repository", None)
    if snapshot_repository is None:
        _fail("outcome protocol snapshot has no pinned authority repository")
    try:
        if Path(snapshot_repository).resolve(strict=True) != authority:
            _fail("outcome protocol authority repository differs from requested authority")
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("outcome protocol authority repository is unavailable") from exc
    plan = bind_validated_outcome_diagnostic_plan(
        build_outcome_group_diagnostic_plan(snapshot), snapshot=snapshot
    )
    if (
        len(plan.plan.model_owners) != EXPECTED_MODEL_OWNERS
        or len(plan.plan.units) != EXPECTED_UNITS
    ):
        _fail("outcome plan is not the exact 240-owner/5760-unit development matrix")
    expected_output = (
        authority / "runs" / "milestone6" / outcome_artifact_store_id(plan.plan.plan_id)
    )
    safe_output = _reject_output(
        output_root,
        (runtime.raw_root, *(fold.store.run_dir for fold in runtime.folds)),
    )
    if safe_output != expected_output.resolve(strict=False):
        _fail("output root must be the exact canonical outcome diagnostic model store path")

    policy = DevicePolicy(requested_device="cpu", torch_threads=1, torch_interop_threads=1)
    if apply_runtime_policy(policy) != "cpu":
        _fail("model preparation runtime policy did not resolve to CPU")
    provenance = capture_system_provenance(authority, policy)
    preparation_provenance_sha256 = _validate_preparation_provenance(
        provenance, expected_commit=expected_preparation_commit_sha
    )
    evidence = _read_evidence(runtime, plan)
    recheck_screening_runtime_readonly(runtime)
    result = prepare_outcome_diagnostic_model_batch(
        plan,
        snapshot,
        safe_output,
        evidence,
        preparation_git_commit_sha=expected_preparation_commit_sha,
        preparation_provenance_sha256=preparation_provenance_sha256,
        preparation_provenance=provenance,
        owner_ids=selected,
        limit=limit,
    )
    recheck_screening_runtime_readonly(runtime)
    fresh_snapshot = load_outcome_group_diagnostic_protocol()
    if fresh_snapshot != snapshot:
        _fail("outcome diagnostic protocol changed during preparation")
    fresh_repository = getattr(fresh_snapshot, "repository", None)
    if fresh_repository is None or Path(fresh_repository).resolve(strict=True) != authority:
        _fail("outcome protocol authority repository changed during preparation")
    final_provenance = capture_system_provenance(authority, policy)
    _validate_preparation_provenance(
        final_provenance,
        expected_commit=expected_preparation_commit_sha,
        expected_identity=preparation_provenance_sha256,
    )
    return {
        "schema_version": "milestone6.phase3.outcome-diagnostic-model-preparation-result.v1",
        "plan_id": plan.plan.plan_id,
        "protocol_sha256": plan.plan.protocol_sha256,
        "output_root": str(safe_output),
        "complete": result.complete,
        "requested_owner_count": len(selected)
        if selected is not None
        else (limit if limit is not None else EXPECTED_MODEL_OWNERS),
        "completed_owner_count": len(result.completed_owner_ids),
        "completed_owner_ids": list(result.completed_owner_ids),
        "expected_owner_count": EXPECTED_MODEL_OWNERS,
        "expected_unit_count": EXPECTED_UNITS,
        "progress_sha256": result.progress.progress_sha256,
    }


prepare_outcome_diagnostic_models = run_outcome_diagnostic_model_preparation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--screening-repository", type=Path, required=True)
    parser.add_argument("--authority-repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-preparation-commit", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=None)
    selection.add_argument("--owner-id", action="append", default=None)
    args = parser.parse_args(argv)
    result = run_outcome_diagnostic_model_preparation(
        args.manifest_path,
        args.manifest_sha256,
        args.raw_root,
        args.screening_repository,
        args.authority_repository,
        args.output_root,
        expected_preparation_commit_sha=args.expected_preparation_commit,
        owner_ids=args.owner_id,
        limit=args.limit,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_MODEL_OWNERS",
    "EXPECTED_UNITS",
    "MODEL_OUTPUT_PREFIX",
    "main",
    "prepare_outcome_diagnostic_models",
    "run_outcome_diagnostic_model_preparation",
]
