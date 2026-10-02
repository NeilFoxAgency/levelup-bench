"""Fail-closed tests for the development-only model-preparation gate."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from levelup.experiments import (
    milestone6_phase3_local_affordance_preparation_readiness as readiness,
)
from levelup.experiments.milestone6_phase3_local_affordance_plan import ROOT

COMMIT = "a" * 40
RAW_ROOT = ROOT / "runs/milestone6/phase3-local-affordance-raw-development-59aac16"


@pytest.fixture(scope="session")
def authority_checkout(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    repository = tmp_path_factory.mktemp("local-affordance-preparation") / "checkout"
    for relative in readiness.SOURCE_RELATIVE_PATHS:
        source = ROOT / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    raw_root = repository / RAW_ROOT.relative_to(ROOT)
    shutil.copytree(RAW_ROOT, raw_root)
    return repository, raw_root


def _capture(
    monkeypatch: pytest.MonkeyPatch,
    authority_checkout: tuple[Path, Path],
) -> readiness.LocalAffordancePreparationSnapshot:
    monkeypatch.setattr(readiness, "_git_state", lambda _repository: (COMMIT, False))
    repository, raw_root = authority_checkout
    return readiness.capture_local_affordance_preparation_readiness(
        repository=repository,
        raw_root=raw_root,
    )


def test_capture_binds_complete_plan_and_raw_authority_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    authority_checkout: tuple[Path, Path],
) -> None:
    snapshot = _capture(monkeypatch, authority_checkout)

    assert readiness.require_local_affordance_preparation_snapshot(snapshot) is snapshot
    assert tuple(source.relative_path for source in snapshot.sources) == readiness.SOURCE_RELATIVE_PATHS
    assert all(source.sha256 for source in snapshot.sources)
    assert snapshot.plan.plan_id == "4e3390056a3a75fad36a033eecf4731dbd08ee946c354449f489c31f5a310718"
    assert snapshot.plan.final_family_access is False
    assert len(snapshot.plan.units) == 11_520
    assert not hasattr(snapshot, "raw_authority")
    assert not hasattr(snapshot, "raw_artifacts")
    assert not hasattr(snapshot, "_raw_authority")
    assert len(snapshot.raw_authority_fingerprint.files) == 751
    assert all(not hasattr(item, "canonical_bytes") for item in snapshot.raw_authority_fingerprint.files)
    assert all(not hasattr(item, "content") for item in snapshot.raw_authority_fingerprint.files)
    assert snapshot.git_commit_sha == COMMIT
    assert snapshot.git_dirty is False
    snapshot.recheck(expected_git_commit=COMMIT)

    for forbidden in ("RunStore", "evaluator", "oracle", "search", "execute", "train"):
        assert not hasattr(readiness, forbidden)


@pytest.mark.parametrize("same_bytes", [False, True])
def test_snapshot_rejects_changed_or_same_byte_replaced_source(
    monkeypatch: pytest.MonkeyPatch,
    authority_checkout: tuple[Path, Path],
    same_bytes: bool,
) -> None:
    snapshot = _capture(monkeypatch, authority_checkout)
    repository, _ = authority_checkout
    target = repository / readiness.PLAN_LOCK_RELATIVE_PATH
    original = target.read_bytes()
    replacement = target.with_name(".plan-lock-replacement.json")
    replacement.write_bytes(target.read_bytes() if same_bytes else b"{}\n")
    try:
        os.replace(replacement, target)
        with pytest.raises(readiness.LocalAffordancePreparationReadinessError, match="source"):
            snapshot.recheck(expected_git_commit=COMMIT)
    finally:
        target.write_bytes(original)


def test_snapshot_rejects_raw_store_artifact_replacement(
    monkeypatch: pytest.MonkeyPatch,
    authority_checkout: tuple[Path, Path],
) -> None:
    snapshot = _capture(monkeypatch, authority_checkout)
    _, raw_root = authority_checkout
    target = next((raw_root / "artifacts").glob("*.json"))
    original = target.read_bytes()
    replacement = target.with_name(".raw-artifact-replacement.json")
    replacement.write_bytes(original)
    try:
        os.replace(replacement, target)
        with pytest.raises(readiness.LocalAffordancePreparationReadinessError, match="raw"):
            snapshot.recheck(expected_git_commit=COMMIT)
    finally:
        target.write_bytes(original)


def test_snapshot_rejects_in_place_raw_store_artifact_mutation(
    monkeypatch: pytest.MonkeyPatch,
    authority_checkout: tuple[Path, Path],
) -> None:
    snapshot = _capture(monkeypatch, authority_checkout)
    _, raw_root = authority_checkout
    target = next((raw_root / "artifacts").glob("*.json"))
    original = target.read_bytes()
    changed = bytearray(original)
    digit = next(index for index, value in enumerate(changed) if 48 <= value <= 57)
    changed[digit] = ord("2") if changed[digit] != ord("2") else ord("3")
    try:
        with target.open("r+b") as stream:
            stream.write(changed)
            stream.flush()
            os.fsync(stream.fileno())
        with pytest.raises(readiness.LocalAffordancePreparationReadinessError, match="raw"):
            snapshot.recheck(expected_git_commit=COMMIT)
    finally:
        with target.open("r+b") as stream:
            stream.write(original)
            stream.truncate(len(original))


def test_snapshot_rejects_raw_store_symlink_substitution(
    monkeypatch: pytest.MonkeyPatch,
    authority_checkout: tuple[Path, Path],
) -> None:
    snapshot = _capture(monkeypatch, authority_checkout)
    _, raw_root = authority_checkout
    target = next((raw_root / "artifacts").glob("*.json"))
    replacement = raw_root / "artifacts" / ".raw-artifact-symlink-target.json"
    target.rename(replacement)
    try:
        target.symlink_to(replacement.name)
        with pytest.raises(readiness.LocalAffordancePreparationReadinessError, match="raw"):
            snapshot.recheck(expected_git_commit=COMMIT)
    finally:
        target.unlink(missing_ok=True)
        replacement.rename(target)


def test_snapshot_rejects_raw_store_namespace_replacement(
    monkeypatch: pytest.MonkeyPatch,
    authority_checkout: tuple[Path, Path],
) -> None:
    snapshot = _capture(monkeypatch, authority_checkout)
    _, raw_root = authority_checkout
    namespace = raw_root / "artifacts"
    retained = raw_root / ".artifacts-retained"
    namespace.rename(retained)
    try:
        namespace.symlink_to(retained.name, target_is_directory=True)
        with pytest.raises(readiness.LocalAffordancePreparationReadinessError, match="raw"):
            snapshot.recheck(expected_git_commit=COMMIT)
    finally:
        namespace.unlink(missing_ok=True)
        retained.rename(namespace)


def test_activation_rechecks_and_yields_only_pinned_development_data(
    monkeypatch: pytest.MonkeyPatch,
    authority_checkout: tuple[Path, Path],
) -> None:
    snapshot = _capture(monkeypatch, authority_checkout)
    with snapshot.activation(expected_git_commit=COMMIT) as lease:
        assert lease.require_active() is lease
        assert lease.plan is snapshot.plan
        assert not hasattr(lease, "raw_artifacts")
        assert not hasattr(lease, "raw_authority")
        assert not hasattr(lease, "issue_heldout_task_probe_capability")
        assert lease._raw_authority is not None
        view = next(
            item for item in lease.plan.views
            if item.fold_id == "plain" and item.replicate == 0
        )
        capability = lease.issue_training_fold_probe_capability(
            fold_id="plain", replicate=0
        )
        evidence = capability.consume_for(view.training_task_ids)
        assert len(evidence) == 40
        assert lease.provenance == {
            "git_commit_sha": COMMIT,
            "raw_capture_git_commit": "59aac16f3a87ea73946159a4bd811e0d400a5554",
            "plan_id": snapshot.plan.plan_id,
            "raw_authority_content_sha256": "909f2724a5723c53e57d965a82b4350fc3bc9a690c671310a964f7a3ecd561a3",
        }
        with pytest.raises(TypeError):
            lease.provenance["plan_id"] = "forged"

    with pytest.raises(readiness.LocalAffordancePreparationReadinessError, match="expired"):
        lease.require_active()
    with pytest.raises(readiness.LocalAffordancePreparationReadinessError, match="expired"):
        lease.issue_training_fold_probe_capability(fold_id="plain", replicate=0)
    assert lease._raw_authority is None
    # Issued fold evidence is a sealed value copy and remains consumable; it
    # cannot be revoked after leaving the preparation context.
    assert len(capability.consume_for(view.training_task_ids)) == 40


def test_activation_requires_exact_clean_commit(
    monkeypatch: pytest.MonkeyPatch,
    authority_checkout: tuple[Path, Path],
) -> None:
    snapshot = _capture(monkeypatch, authority_checkout)
    with pytest.raises(readiness.LocalAffordancePreparationReadinessError, match="not authorized"):
        with snapshot.activation(expected_git_commit="b" * 40):
            pytest.fail("an unrelated commit must not receive a preparation lease")


def test_direct_or_rebound_snapshots_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    authority_checkout: tuple[Path, Path],
) -> None:
    snapshot = _capture(monkeypatch, authority_checkout)
    object.__setattr__(snapshot, "raw_root", authority_checkout[0] / "runs")
    with pytest.raises(readiness.LocalAffordancePreparationReadinessError, match="forged or rebound"):
        snapshot.require_sealed()
    with pytest.raises((readiness.LocalAffordancePreparationReadinessError, TypeError)):
        readiness.LocalAffordancePreparationSnapshot()
