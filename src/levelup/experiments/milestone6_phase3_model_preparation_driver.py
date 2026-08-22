"""Run the preparation-only Phase 3 model construction boundary.

This module is deliberately a small orchestration boundary.  It accepts the
canonical Phase 2 readiness manifest and a caller-supplied output directory,
loads the immutable Phase 3 plan/anchor/evidence authorities, and delegates
model construction to :func:`prepare_phase3_model_batch`.  It has no path or
API for final families, outcomes, search, replay, evaluators, or oracles.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from levelup.experiments.milestone6_phase2_screening_provenance import (
    CANONICAL_READINESS_PATH,
)
from levelup.experiments.milestone6_phase2_screening_runtime import (
    load_screening_runtime,
)
from levelup.experiments.milestone6_phase3_anchor import (
    load_committed_phase3_anchor_manifest_bytes,
    validate_phase3_anchor_manifest_bytes,
)
from levelup.experiments.milestone6_phase3_evidence import (
    load_committed_phase3_evidence_lock_bytes,
    validate_phase3_evidence_lock_bytes,
)
from levelup.experiments.milestone6_phase3_model_preparation import (
    EXPECTED_MODELS,
    Phase3ModelPreparationResult,
    prepare_phase3_model_batch,
)
from levelup.experiments.milestone6_phase3_plan import (
    bind_validated_phase3_plan,
    load_committed_phase3_plan_lock_bytes,
    validate_phase3_plan_lock_bytes,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.provenance import capture_system_provenance

PHASE3_PLAN_LOCK_RELATIVE_PATH = Path("configs/milestone6/phase3_plan_lock.json")
PHASE3_ANCHOR_RELATIVE_PATH = Path("configs/milestone6/phase3_anchor_manifest.json")
PHASE3_EVIDENCE_RELATIVE_PATH = Path("configs/milestone6/phase3_evidence_lock.json")


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _validate_digest(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail("manifest byte pin is not a lowercase SHA-256 digest")


def _reject_unsafe_output_root(output_root: str | Path, raw_root: str | Path) -> Path:
    """Reject output locations that overlap raw evidence or use symlinks."""

    output = Path(output_root).absolute()
    raw = Path(raw_root).absolute().resolve(strict=False)
    # Check the lexical chain before resolving.  This catches existing symlink
    # ancestors (including a broken final symlink) without following them.
    for candidate in (output, *output.parents):
        if os.path.lexists(candidate) and candidate.is_symlink():
            _fail("Phase 3 model output root or an ancestor is a symlink")
    resolved_output = output.resolve(strict=False)
    if (
        resolved_output == raw
        or raw in resolved_output.parents
        or resolved_output in raw.parents
    ):
        _fail("Phase 3 model output root must not overlap the raw evidence root")
    return resolved_output


def _authority_path(repository: Path, relative: Path) -> Path:
    return repository / relative


def _repository_identity(repository: Path) -> tuple[int, int]:
    """Pin the exact repository directory handed to the batch boundary."""

    try:
        directory_fd = secure_fs.open_directory_chain(repository)
        try:
            return secure_fs.directory_identity(directory_fd)
        finally:
            os.close(directory_fd)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _fail("Phase 3 authority repository cannot be securely pinned")
        raise AssertionError("unreachable") from exc


def _reject_repository_symlink_chain(
    repository: str | Path,
    *,
    label: str,
) -> Path:
    """Resolve one repository only after rejecting every lexical symlink."""

    lexical = Path(repository).absolute()
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


def run_phase3_model_preparation(
    manifest_path: str | Path,
    manifest_bytes_sha256: str,
    raw_root: str | Path,
    repository: str | Path | None,
    output_root: str | Path,
    *,
    screening_repository: str | Path | None = None,
    authority_repository: str | Path | None = None,
    owner_ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Prepare the selected Phase 3 model owners and return deterministic JSON data."""

    if screening_repository is None:
        if repository is None:
            _fail("Phase 3 driver requires a screening repository")
        screening_repository = repository
    screening_repository_path = _reject_repository_symlink_chain(
        screening_repository,
        label="screening",
    )
    if authority_repository is None:
        _fail("Phase 3 driver requires an explicit authority repository")
    authority_repository_path = _reject_repository_symlink_chain(
        authority_repository,
        label="authority",
    )
    repository_path = screening_repository_path
    canonical_manifest_path = repository_path / CANONICAL_READINESS_PATH
    if Path(manifest_path).resolve(strict=False) != canonical_manifest_path:
        _fail("Phase 3 driver requires the canonical committed readiness manifest")
    _validate_digest(manifest_bytes_sha256)
    if owner_ids is not None and limit is not None:
        _fail("owner_ids and limit are mutually exclusive")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= EXPECTED_MODELS):
        _fail("limit is outside the frozen 480-owner universe")
    selected_owner_ids = None if owner_ids is None else tuple(owner_ids)
    if selected_owner_ids is not None:
        if len(set(selected_owner_ids)) != len(selected_owner_ids):
            _fail("owner_ids contain duplicates")
        if any(
            not isinstance(owner_id, str)
            or len(owner_id) != 64
            or owner_id != owner_id.lower()
            or any(character not in "0123456789abcdef" for character in owner_id)
            for owner_id in selected_owner_ids
        ):
            _fail("owner_ids must be lowercase SHA-256 identities")
    safe_output_root = _reject_unsafe_output_root(output_root, raw_root)
    authority_repository_identity = _repository_identity(authority_repository_path)

    # The runtime loader is the first authority gate.  The following bytes are
    # retained in local immutable variables and passed to every validator.  The
    # batch then reopens this exact pinned repository and byte-compares the same
    # files immediately before preparation.
    runtime = load_screening_runtime(
        manifest_path,
        raw_root,
        screening_repository,
        manifest_bytes_sha256=manifest_bytes_sha256,
        authority_repository=authority_repository_path,
    )
    plan_lock_bytes = load_committed_phase3_plan_lock_bytes(
        _authority_path(authority_repository_path, PHASE3_PLAN_LOCK_RELATIVE_PATH)
    )
    plan = validate_phase3_plan_lock_bytes(plan_lock_bytes)
    validated_plan = bind_validated_phase3_plan(plan)
    anchor_bytes = load_committed_phase3_anchor_manifest_bytes(
        _authority_path(authority_repository_path, PHASE3_ANCHOR_RELATIVE_PATH)
    )
    anchor_manifest = validate_phase3_anchor_manifest_bytes(
        anchor_bytes,
        runtime=runtime,
    )
    evidence_bytes = load_committed_phase3_evidence_lock_bytes(
        _authority_path(authority_repository_path, PHASE3_EVIDENCE_RELATIVE_PATH)
    )
    evidence_lock = validate_phase3_evidence_lock_bytes(
        evidence_bytes,
        runtime=runtime,
        validated_plan=validated_plan,
        anchor_manifest=anchor_manifest,
        anchor_file_bytes=anchor_bytes,
        plan_lock_bytes=plan_lock_bytes,
    )
    preparation_provenance = None
    if getattr(runtime, "device_policy", None) is not None:
        preparation_provenance = capture_system_provenance(
            authority_repository_path, runtime.device_policy
        )
        if preparation_provenance.git_dirty:
            _fail("Phase 3 authority repository must be clean for model preparation")
    batch_kwargs: dict[str, Any] = {
        "runtime": runtime,
        "validated_plan": validated_plan,
        "anchor_manifest": anchor_manifest,
        "evidence_lock": evidence_lock,
        "owner_ids": selected_owner_ids,
        "limit": limit,
        "authority_repository": authority_repository_path,
        "authority_repository_identity": authority_repository_identity,
        "plan_lock_bytes": plan_lock_bytes,
        "anchor_file_bytes": anchor_bytes,
        "evidence_lock_bytes": evidence_bytes,
    }
    if preparation_provenance is not None:
        batch_kwargs["preparation_provenance"] = preparation_provenance
    result: Phase3ModelPreparationResult = prepare_phase3_model_batch(
        safe_output_root,
        **batch_kwargs,
    )
    payload = result.model_dump(mode="json")
    payload["output_root"] = str(safe_output_root)
    return payload


# Descriptive aliases for callers using the preparation-boundary terminology.
prepare_phase3_models = run_phase3_model_preparation
run_model_preparation = run_phase3_model_preparation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--screening-repository", type=Path, required=True)
    parser.add_argument("--authority-repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=None)
    selection.add_argument("--owner-id", action="append", default=None)
    args = parser.parse_args(argv)
    result = run_phase3_model_preparation(
        args.manifest_path,
        args.manifest_sha256,
        args.raw_root,
        args.screening_repository,
        args.output_root,
        authority_repository=args.authority_repository,
        owner_ids=args.owner_id,
        limit=args.limit,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_READINESS_PATH",
    "EXPECTED_MODELS",
    "PHASE3_ANCHOR_RELATIVE_PATH",
    "PHASE3_EVIDENCE_RELATIVE_PATH",
    "PHASE3_PLAN_LOCK_RELATIVE_PATH",
    "main",
    "prepare_phase3_models",
    "run_model_preparation",
    "run_phase3_model_preparation",
]
