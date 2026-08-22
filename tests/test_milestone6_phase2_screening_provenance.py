"""Adversarial tests for the screening-only readiness publication exception."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from levelup.experiments.milestone6_phase2_screening_provenance import (
    CANONICAL_READINESS_PATH,
    validate_screening_provenance,
)
from levelup.experiments.runner.records import SystemProvenance
from levelup.experiments.runner.training_data_artifacts import TrainingDataArtifactError

MANIFEST = b'{"schema_version":"test"}\n'


def _provenance(commit: str, *, dirty: bool = False) -> SystemProvenance:
    return SystemProvenance(
        git_commit_sha=commit,
        git_dirty=dirty,
        git_diff_sha256=None if not dirty else "a" * 64,
        python_version="test",
        packages={"levelup-bench": "test"},
        installed_packages_sha256="b" * 64,
        os="test",
        architecture="test",
        cpu="test",
        cpu_count=1,
        memory_bytes=1,
        requested_device="cpu",
        resolved_device="cpu",
        requested_torch_threads=1,
        actual_torch_threads=1,
        requested_torch_interop_threads=1,
        actual_torch_interop_threads=1,
        deterministic_algorithms_requested=True,
        deterministic_algorithms_actual=True,
        processes=1,
        captured_at_utc=datetime(2026, 8, 22, tzinfo=UTC),
    )


def _run(repo: Path, *args: str) -> str:
    return subprocess.run(("git", *args), cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init", "-q")
    _run(repo, "config", "user.email", "test@example.invalid")
    _run(repo, "config", "user.name", "Test")
    (repo / "README").write_text("base\n")
    _run(repo, "add", ".")
    _run(repo, "commit", "-qm", "preparation")
    return repo, _run(repo, "rev-parse", "HEAD")


def _publish(repo: Path, *paths: str) -> str:
    for path in paths:
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"publication\n")
    _run(repo, "add", ".")
    _run(repo, "commit", "-qm", "publication")
    return _run(repo, "rev-parse", "HEAD")


def test_direct_child_readiness_artifact_is_allowed(tmp_path: Path) -> None:
    repo, preparation = _git_repo(tmp_path)
    manifest = repo / CANONICAL_READINESS_PATH
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(MANIFEST)
    _run(repo, "add", CANONICAL_READINESS_PATH)
    _run(repo, "commit", "-qm", "publication")
    current = _run(repo, "rev-parse", "HEAD")
    validate_screening_provenance(
        _provenance(preparation),
        _provenance(current),
        repository=repo,
        manifest_bytes=MANIFEST,
    )


def test_exact_clean_preparation_commit_is_allowed(tmp_path: Path) -> None:
    repo, preparation = _git_repo(tmp_path)
    validate_screening_provenance(
        _provenance(preparation),
        _provenance(preparation),
        repository=repo,
        manifest_bytes=MANIFEST,
    )


def test_artifact_modify_case_is_allowed(tmp_path: Path) -> None:
    repo, _base = _git_repo(tmp_path)
    manifest = repo / CANONICAL_READINESS_PATH
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"old\n")
    _run(repo, "add", CANONICAL_READINESS_PATH)
    _run(repo, "commit", "-qm", "preparation")
    preparation = _run(repo, "rev-parse", "HEAD")
    manifest.write_bytes(MANIFEST)
    _run(repo, "add", CANONICAL_READINESS_PATH)
    _run(repo, "commit", "-qm", "publication")
    current = _run(repo, "rev-parse", "HEAD")
    validate_screening_provenance(
        _provenance(preparation),
        _provenance(current),
        repository=repo,
        manifest_bytes=MANIFEST,
    )


def test_later_descendant_is_rejected(tmp_path: Path) -> None:
    repo, preparation = _git_repo(tmp_path)
    _publish(repo, CANONICAL_READINESS_PATH)
    (repo / "README").write_text("later\n")
    _run(repo, "add", "README")
    _run(repo, "commit", "-qm", "later")
    with pytest.raises(TrainingDataArtifactError):
        validate_screening_provenance(
            _provenance(preparation),
            _provenance(_run(repo, "rev-parse", "HEAD")),
            repository=repo,
            manifest_bytes=MANIFEST,
        )


@pytest.mark.parametrize("dirty_name", ("README", "untracked.txt"))
def test_current_dirty_tracked_or_untracked_is_rejected(tmp_path: Path, dirty_name: str) -> None:
    repo, preparation = _git_repo(tmp_path)
    target = repo / dirty_name
    if dirty_name == "README":
        target.write_text("changed\n")
    else:
        target.write_text("untracked\n")
    with pytest.raises(TrainingDataArtifactError):
        validate_screening_provenance(
            _provenance(preparation),
            _provenance(preparation),
            repository=repo,
            manifest_bytes=MANIFEST,
        )


def test_missing_preparation_commit_is_rejected(tmp_path: Path) -> None:
    repo, _preparation = _git_repo(tmp_path)
    with pytest.raises(TrainingDataArtifactError):
        validate_screening_provenance(
            _provenance("f" * 40),
            _provenance(_run(repo, "rev-parse", "HEAD")),
            repository=repo,
            manifest_bytes=MANIFEST,
        )


def test_merge_commit_is_rejected(tmp_path: Path) -> None:
    repo, preparation = _git_repo(tmp_path)
    branch = _run(repo, "branch", "--show-current")
    _run(repo, "checkout", "-qb", "side")
    (repo / "side.txt").write_text("side\n")
    _run(repo, "add", "side.txt")
    _run(repo, "commit", "-qm", "side")
    _run(repo, "checkout", branch)
    _run(repo, "merge", "--no-ff", "-m", "merge", "side")
    with pytest.raises(TrainingDataArtifactError):
        validate_screening_provenance(
            _provenance(preparation),
            _provenance(_run(repo, "rev-parse", "HEAD")),
            repository=repo,
            manifest_bytes=MANIFEST,
        )


def test_non_git_environment_drift_is_rejected(tmp_path: Path) -> None:
    repo, preparation = _git_repo(tmp_path)
    current = _provenance(preparation).model_copy(update={"os": "different"})
    with pytest.raises(TrainingDataArtifactError):
        validate_screening_provenance(
            _provenance(preparation),
            current,
            repository=repo,
            manifest_bytes=MANIFEST,
        )


def test_supplied_current_sha_must_match_actual_head(tmp_path: Path) -> None:
    repo, preparation = _git_repo(tmp_path)
    _publish(repo, "docs/notes.md")
    with pytest.raises(TrainingDataArtifactError):
        validate_screening_provenance(
            _provenance(preparation),
            _provenance(preparation),
            repository=repo,
            manifest_bytes=MANIFEST,
        )


def test_wrong_manifest_blob_is_rejected(tmp_path: Path) -> None:
    repo, preparation = _git_repo(tmp_path)
    manifest = repo / CANONICAL_READINESS_PATH
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(MANIFEST + b" ")
    _run(repo, "add", CANONICAL_READINESS_PATH)
    _run(repo, "commit", "-qm", "publication")
    current = _run(repo, "rev-parse", "HEAD")
    with pytest.raises(TrainingDataArtifactError):
        validate_screening_provenance(
            _provenance(preparation),
            _provenance(current),
            repository=repo,
            manifest_bytes=MANIFEST,
        )


@pytest.mark.parametrize("extra", ("docs/notes.md", "experiments/other.json"))
def test_extra_path_or_wrong_blob_is_rejected(tmp_path: Path, extra: str) -> None:
    repo, preparation = _git_repo(tmp_path)
    _publish(repo, CANONICAL_READINESS_PATH, extra)
    with pytest.raises(TrainingDataArtifactError):
        validate_screening_provenance(
            _provenance(preparation),
            _provenance(_run(repo, "rev-parse", "HEAD")),
            repository=repo,
            manifest_bytes=MANIFEST,
        )


def test_dirty_preparation_and_later_descendant_are_rejected(tmp_path: Path) -> None:
    repo, preparation = _git_repo(tmp_path)
    _publish(repo, CANONICAL_READINESS_PATH)
    current = _run(repo, "rev-parse", "HEAD")
    with pytest.raises(TrainingDataArtifactError):
        validate_screening_provenance(
            _provenance(preparation, dirty=True),
            _provenance(current),
            repository=repo,
            manifest_bytes=MANIFEST,
        )
