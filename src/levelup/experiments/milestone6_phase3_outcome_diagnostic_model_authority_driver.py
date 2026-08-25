"""Publish the complete development-only Phase 3 outcome-model authority.

This command is the narrow publication boundary after provenance-bound model
preparation.  It reloads the committed development runtime and protocol, reads
the same thirty descriptor-pinned evidence payloads used by preparation, and
maps them to the exact sixty condition/view identities.  It then asks the
descriptor-pinned authority builder to validate the complete model store.  No
candidate, search, evaluator, oracle, result, or final-family module is
reachable from this module.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from levelup.experiments.milestone6_phase2_screening_provenance import (
    CANONICAL_READINESS_PATH,
)
from levelup.experiments.milestone6_phase2_screening_runtime import (
    load_screening_runtime,
    recheck_screening_runtime_readonly,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
    canonical_outcome_model_artifact_authority_bytes,
    outcome_artifact_store_id,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_authority import (
    OutcomeDiagnosticModelAuthorityError,
    build_outcome_model_artifact_authority_from_store,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_preparation_driver import (
    EXPECTED_MODEL_OWNERS,
    _read_evidence,
    _reject_output,
    _reject_repository,
    _validate_commit,
    _validate_digest,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    EXPECTED_UNITS,
    EXPECTED_VIEWS,
    bind_validated_outcome_diagnostic_plan,
    build_outcome_group_diagnostic_plan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    load_outcome_group_diagnostic_protocol,
)
from levelup.experiments.runner import secure_fs

AUTHORITY_OUTPUT_PATH = "configs/milestone6/phase3_outcome_model_artifact_authority.json"
PROTOCOL_PATH = "configs/milestone6/phase3_outcome_group_diagnostic.json"


def _fail(message: str) -> None:
    raise OutcomeDiagnosticModelAuthorityError(message)


def _git(repository: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ("git", *args),
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OutcomeDiagnosticModelAuthorityError(
            "authority repository git state is unavailable"
        ) from exc


def _repo_state(repository: Path) -> tuple[str, bool]:
    commit = _git(repository, "rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    status = _git(repository, "status", "--porcelain=v1", "-z")
    if (
        not commit
        or not all(character in "0123456789abcdef" for character in commit)
        or not 40 <= len(commit) <= 64
    ):
        _fail("authority repository HEAD is not a valid git SHA")
    return commit, bool(status)


def _validate_generation_repository(repository: Path, expected_commit: str) -> None:
    commit, dirty = _repo_state(repository)
    if dirty or commit != expected_commit:
        _fail("authority repository is dirty or differs from expected generation commit")


def _canonical_output_path(repository: Path, output_path: str | Path) -> Path:
    target = Path(output_path).absolute()
    canonical = (repository / AUTHORITY_OUTPUT_PATH).absolute()
    for candidate in (target, *target.parents):
        if os.path.lexists(candidate) and candidate.is_symlink():
            _fail("authority output or an ancestor is a symlink")
    if target != canonical:
        _fail("authority output must be the exact canonical config path")
    if os.path.lexists(target):
        _fail("authority output already exists")
    return target


def _write_authority(path: Path, payload: bytes) -> None:
    parent_fd: int | None = None
    output_fd: int | None = None
    created = False
    try:
        parent_fd = secure_fs.open_directory_chain(path.parent)
        output_fd = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        created = True
        with os.fdopen(output_fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(output_fd)
        os.fsync(parent_fd)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if created and parent_fd is not None:
            try:
                os.unlink(path.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise OutcomeDiagnosticModelAuthorityError(
            "cannot exclusively publish outcome model authority"
        ) from exc
    finally:
        if output_fd is not None:
            os.close(output_fd)
        if parent_fd is not None:
            os.close(parent_fd)


def _load_inputs(
    manifest_path: str | Path,
    manifest_sha256: str,
    raw_root: str | Path,
    screening_repository: str | Path,
    authority_repository: str | Path,
):
    screening = _reject_repository(screening_repository, "screening")
    authority = _reject_repository(authority_repository, "authority")
    canonical_manifest = screening / CANONICAL_READINESS_PATH
    if Path(manifest_path).absolute() != canonical_manifest:
        _fail("outcome authority driver requires the canonical committed readiness manifest")
    _validate_digest(manifest_sha256, "manifest byte pin")
    runtime = load_screening_runtime(
        manifest_path,
        raw_root,
        screening,
        manifest_bytes_sha256=manifest_sha256,
        authority_repository=authority,
    )
    recheck_screening_runtime_readonly(runtime)
    snapshot = load_outcome_group_diagnostic_protocol(
        PROTOCOL_PATH,
        repository=authority,
    )
    if getattr(snapshot, "repository", None) is None:
        _fail("outcome protocol snapshot has no pinned authority repository")
    try:
        if Path(snapshot.repository).resolve(strict=True) != authority:
            _fail("outcome protocol authority repository differs from requested authority")
    except (OSError, RuntimeError) as exc:
        raise OutcomeDiagnosticModelAuthorityError(
            "outcome protocol authority repository is unavailable"
        ) from exc
    plan = bind_validated_outcome_diagnostic_plan(
        build_outcome_group_diagnostic_plan(snapshot), snapshot=snapshot
    )
    if (
        plan.plan.final_family_access
        or len(plan.plan.views) != EXPECTED_VIEWS
        or len(plan.plan.units) != EXPECTED_UNITS
        or len(plan.plan.model_owners) != EXPECTED_MODEL_OWNERS
    ):
        _fail("outcome authority plan is not the exact development-only matrix")
    return screening, authority, runtime, snapshot, plan


def run_outcome_model_authority_generation(
    manifest_path: str | Path,
    manifest_sha256: str,
    raw_root: str | Path,
    screening_repository: str | Path,
    authority_repository: str | Path,
    store_root: str | Path,
    *,
    expected_preparation_commit_sha: str,
    expected_preparation_provenance_sha256: str,
    expected_generation_commit_sha: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and optionally publish the complete development-only authority."""

    _validate_commit(expected_preparation_commit_sha)
    _validate_digest(expected_preparation_provenance_sha256, "preparation provenance pin")
    _validate_commit(expected_generation_commit_sha)
    screening, authority, runtime, snapshot, plan = _load_inputs(
        manifest_path,
        manifest_sha256,
        raw_root,
        screening_repository,
        authority_repository,
    )
    _validate_generation_repository(authority, expected_generation_commit_sha)
    expected_store = authority / "runs" / "milestone6" / outcome_artifact_store_id(
        plan.plan.plan_id
    )
    safe_store = _reject_output(store_root, (runtime.raw_root, *(fold.store.run_dir for fold in runtime.folds)))
    if safe_store != expected_store.resolve(strict=False):
        _fail("store root must be the exact canonical outcome diagnostic model store path")
    if not os.path.lexists(safe_store) or safe_store.is_symlink() or not safe_store.is_dir():
        _fail("canonical outcome diagnostic model store must already exist as a directory")

    evidence_by_source = _read_evidence(runtime, plan)
    try:
        evidence_by_view = {
            view.view_id: evidence_by_source[(view.heldout_family, view.replicate)]
            for view in plan.plan.views
        }
    except KeyError as exc:
        raise OutcomeDiagnosticModelAuthorityError(
            "outcome evidence mapping is missing a canonical view source"
        ) from exc
    if len(evidence_by_view) != EXPECTED_VIEWS:
        _fail("outcome evidence mapping does not cover the exact sixty views")
    recheck_screening_runtime_readonly(runtime)
    authority_value = build_outcome_model_artifact_authority_from_store(
        safe_store,
        plan,
        snapshot,
        evidence_by_view,
        preparation_git_commit_sha=expected_preparation_commit_sha,
        preparation_provenance_sha256=expected_preparation_provenance_sha256,
        generation_git_commit_sha=expected_generation_commit_sha,
    )
    payload = canonical_outcome_model_artifact_authority_bytes(authority_value)

    # Rebuild after all inputs have been read.  The store builder itself pins
    # and compares its complete identity snapshot; equality here additionally
    # proves that the compact bytes are deterministic across the final check.
    recheck_screening_runtime_readonly(runtime)
    _validate_generation_repository(authority, expected_generation_commit_sha)
    fresh_snapshot = load_outcome_group_diagnostic_protocol(
        PROTOCOL_PATH,
        repository=authority,
    )
    if fresh_snapshot != snapshot:
        _fail("outcome diagnostic protocol changed during authority generation")
    second = build_outcome_model_artifact_authority_from_store(
        safe_store,
        plan,
        snapshot,
        evidence_by_view,
        preparation_git_commit_sha=expected_preparation_commit_sha,
        preparation_provenance_sha256=expected_preparation_provenance_sha256,
        generation_git_commit_sha=expected_generation_commit_sha,
    )
    second_payload = canonical_outcome_model_artifact_authority_bytes(second)
    if second_payload != payload:
        _fail("store-derived outcome model authority is not deterministic")
    _validate_generation_repository(authority, expected_generation_commit_sha)
    published_path = None
    if output_path is not None:
        published_path = _canonical_output_path(authority, output_path)
        _write_authority(published_path, payload)
    return {
        "schema_version": "milestone6.phase3.outcome-diagnostic-model-authority-result.v1",
        "authority_sha256": authority_value.authority_sha256,
        "plan_id": plan.plan.plan_id,
        "protocol_sha256": plan.plan.protocol_sha256,
        "artifact_store_id": authority_value.artifact_store_id,
        "generation_git_commit_sha": authority_value.generation_git_commit_sha,
        "view_count": len(authority_value.views),
        "evidence_count": len(authority_value.evidence),
        "artifact_count": len(authority_value.artifacts),
        "output_path": str(published_path) if published_path is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--screening-repository", type=Path, required=True)
    parser.add_argument("--authority-repository", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--expected-preparation-commit", required=True)
    parser.add_argument("--expected-preparation-provenance", required=True)
    parser.add_argument("--expected-generation-commit", required=True)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args(argv)
    result = run_outcome_model_authority_generation(
        args.manifest_path,
        args.manifest_sha256,
        args.raw_root,
        args.screening_repository,
        args.authority_repository,
        args.store_root,
        expected_preparation_commit_sha=args.expected_preparation_commit,
        expected_preparation_provenance_sha256=args.expected_preparation_provenance,
        expected_generation_commit_sha=args.expected_generation_commit,
        output_path=args.output_path,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_OUTPUT_PATH",
    "PROTOCOL_PATH",
    "main",
    "run_outcome_model_authority_generation",
]
