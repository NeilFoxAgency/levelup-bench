"""Screening-only provenance rule for publishing the readiness manifest.

The preparation pass is expected to run from a clean checkout.  A later clean
checkout is accepted when it is either the same commit, or the direct child of
the preparation commit containing only the canonical readiness-manifest
artifact.  This exception is intentionally local to screening; :class:`RunStore`
continues to require exact provenance identity.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from levelup.experiments.runner.records import SystemProvenance
from levelup.experiments.runner.training_data_artifacts import TrainingDataArtifactError

CANONICAL_READINESS_PATH = "experiments/milestone6_phase2_screening_readiness.json"


def canonical_screening_repository(
    repository: str | Path,
    *,
    authority_root: str | Path,
) -> Path:
    """Require provenance and authority inputs to come from one checkout."""

    try:
        repository_path = Path(repository).resolve(strict=True)
        authority_path = Path(authority_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TrainingDataArtifactError(
            "cannot resolve the screening repository and authority checkout"
        ) from exc
    if not repository_path.is_dir() or repository_path != authority_path:
        raise TrainingDataArtifactError(
            "screening repository must be the canonical authority checkout"
        )
    return repository_path


def _stable_non_git_fields(value: SystemProvenance) -> tuple[object, ...]:
    """Return fields that must not change across an artifact-only publication."""

    payload = value.model_dump(mode="json")
    return tuple(
        (name, payload[name])
        for name in sorted(payload)
        if name
        not in {
            "git_commit_sha",
            "git_dirty",
            "git_diff_sha256",
            "captured_at_utc",
        }
    )


def _git(repository: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ("git", *args),
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TrainingDataArtifactError(
            "cannot verify screening publication provenance"
        ) from exc


def _direct_artifact_publication(
    repository: Path,
    preparation_sha: str,
    current_sha: str,
    manifest_bytes: bytes,
) -> bool:
    """Check the exact one-parent, one-file publication exception."""

    head = _git(repository, "rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    if head != current_sha:
        return False
    if _git(repository, "status", "--porcelain=v1", "-z"):
        return False
    parents = _git(repository, "rev-list", "--parents", "-n", "1", "HEAD").decode(
        "utf-8", errors="strict"
    ).strip().split()
    if len(parents) != 2 or parents[1] != preparation_sha:
        return False

    raw = _git(
        repository,
        "diff",
        "--raw",
        "--no-renames",
        "--no-ext-diff",
        "--format=",
        preparation_sha,
        "HEAD",
    )
    lines = [line for line in raw.splitlines() if line]
    if len(lines) != 1 or b"\t" not in lines[0]:
        return False
    metadata, encoded_path = lines[0].split(b"\t", 1)
    fields = metadata.decode("ascii", errors="strict").split()
    if len(fields) != 5:
        return False
    old_mode, new_mode, _old_sha, _new_sha, status = fields
    old_mode = old_mode.removeprefix(":")
    if (
        encoded_path.decode("utf-8", errors="surrogateescape")
        != CANONICAL_READINESS_PATH
        or status not in {"A", "M"}
        or new_mode != "100644"
        or old_mode not in {"000000", "100644"}
    ):
        return False
    try:
        published = _git(repository, "show", f"HEAD:{CANONICAL_READINESS_PATH}")
    except TrainingDataArtifactError:
        return False
    return published == manifest_bytes


def validate_screening_provenance(
    preparation: SystemProvenance,
    current: SystemProvenance,
    *,
    repository: str | Path,
    manifest_bytes: bytes,
) -> None:
    """Validate screening provenance, raising on any publication mismatch.

    Preparation must be clean.  Current provenance must also be clean and must
    match every non-git, non-timestamp field.  A changed git SHA is accepted
    only for the exact direct-child artifact publication described above.
    """

    if preparation.git_dirty or preparation.git_diff_sha256 is not None:
        raise TrainingDataArtifactError(
            "screening preparation provenance must be clean"
        )
    if current.git_dirty or current.git_diff_sha256 is not None:
        raise TrainingDataArtifactError(
            "current screening provenance changed: working tree is dirty"
        )
    if _stable_non_git_fields(preparation) != _stable_non_git_fields(current):
        raise TrainingDataArtifactError(
            "current captured screening provenance differs from preparation"
        )
    if current.git_commit_sha == preparation.git_commit_sha:
        # A clean exact-commit capture is normally sufficient.  When a real
        # repository is available, also bind the capture to the current HEAD
        # and status so a stale caller-supplied provenance cannot pass.
        try:
            head = _git(Path(repository), "rev-parse", "HEAD").decode(
                "ascii", errors="strict"
            ).strip()
            if head != current.git_commit_sha or _git(
                Path(repository), "status", "--porcelain=v1", "-z"
            ):
                raise TrainingDataArtifactError(
                    "current screening provenance changed: repository is not clean"
                )
        except TrainingDataArtifactError:
            raise
        return
    if not _direct_artifact_publication(
        Path(repository), preparation.git_commit_sha, current.git_commit_sha, manifest_bytes
    ):
        raise TrainingDataArtifactError(
            "current screening commit is not the permitted readiness publication child"
        )


def screening_provenance_matches(
    preparation: SystemProvenance,
    current: SystemProvenance,
    *,
    repository: str | Path,
    manifest_bytes: bytes,
) -> None:
    """Compatibility alias for callers phrased as a predicate/check."""

    validate_screening_provenance(
        preparation,
        current,
        repository=repository,
        manifest_bytes=manifest_bytes,
    )


__all__ = [
    "CANONICAL_READINESS_PATH",
    "canonical_screening_repository",
    "screening_provenance_matches",
    "validate_screening_provenance",
]
