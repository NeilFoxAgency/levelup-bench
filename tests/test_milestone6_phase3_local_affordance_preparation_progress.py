from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from levelup.experiments.milestone6_phase3_local_affordance_model_store import (
    LocalAffordanceModelStoreError,
    open_local_affordance_model_store,
)
from levelup.experiments.milestone6_phase3_local_affordance_plan import (
    EXPECTED_COUNTS,
    FROZEN_SOURCE_SHA256,
    build_local_affordance_plan,
)
from levelup.experiments.milestone6_phase3_local_affordance_preparation_progress import (
    LocalAffordancePreparationProgressError,
    _snapshot_from_validated_records,
    inventory_local_affordance_model_progress,
)


def _record(owner_id: str, offset: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        key=SimpleNamespace(
            owner_id=owner_id,
            model_state_sha256=f"{offset + 1:064x}",
            model_identity_sha256=f"{offset + 2:064x}",
        ),
        record_sha256=f"{offset + 3:064x}",
        cost=SimpleNamespace(
            optimizer_steps=120,
            forward_passes=4_800,
            training_examples=40,
        ),
        artifact=SimpleNamespace(artifact_bytes=128 + offset),
    )


@pytest.fixture(scope="module")
def plan():
    return build_local_affordance_plan()


def test_frozen_owner_identity_universe_has_exactly_480_unique_ids(plan):
    owner_ids = tuple(owner.owner_id for owner in plan.model_owners)
    assert plan.plan_id == "4e3390056a3a75fad36a033eecf4731dbd08ee946c354449f489c31f5a310718"
    assert len(owner_ids) == EXPECTED_COUNTS["model_owners"] == 480
    assert len(set(owner_ids)) == 480
    assert tuple(plan.source_sha256) == tuple(FROZEN_SOURCE_SHA256.items())


def test_small_validated_subset_reports_missing_owners_and_cost_totals(plan):
    owner_ids = tuple(sorted(owner.owner_id for owner in plan.model_owners))
    rows = (_record(owner_ids[0], 0), _record(owner_ids[1], 1))

    progress = _snapshot_from_validated_records(
        plan=plan,
        preparation_git_commit_sha="b" * 40,
        preparation_provenance_sha256="c" * 64,
        store_manifest_sha256="d" * 64,
        records=rows,
    )

    assert progress.expected_owner_count == 480
    assert progress.completed_owner_ids == owner_ids[:2]
    assert len(progress.missing_owner_ids) == 478
    assert progress.total_optimizer_steps == 240
    assert progress.total_forward_passes == 9_600
    assert progress.total_training_examples == 80
    assert progress.total_serialized_artifact_bytes == 257
    assert progress.complete is False
    assert progress.canonical_bytes().endswith(b"\n")


def test_reducer_rejects_duplicate_or_foreign_owner_records(plan):
    owner_ids = tuple(sorted(owner.owner_id for owner in plan.model_owners))
    kwargs = {
        "plan": plan,
        "preparation_git_commit_sha": "b" * 40,
        "preparation_provenance_sha256": "c" * 64,
        "store_manifest_sha256": "d" * 64,
    }
    with pytest.raises(LocalAffordancePreparationProgressError, match="duplicate"):
        _snapshot_from_validated_records(
            **kwargs, records=(_record(owner_ids[0]), _record(owner_ids[0], 1))
        )
    with pytest.raises(LocalAffordancePreparationProgressError, match="foreign"):
        _snapshot_from_validated_records(
            **kwargs, records=(_record("f" * 64),)
        )


def test_read_only_inventory_does_not_create_missing_store(plan, tmp_path: Path):
    absent_store = tmp_path / "must-stay-absent"
    with pytest.raises(LocalAffordanceModelStoreError):
        inventory_local_affordance_model_progress(
            plan,
            absent_store,
            preparation_git_commit_sha="b" * 40,
            preparation_provenance_sha256="c" * 64,
        )
    assert not absent_store.exists()


def test_empty_existing_store_is_incomplete_not_complete(plan, tmp_path: Path):
    store_root = tmp_path / "empty-store"
    with open_local_affordance_model_store(store_root):
        pass

    progress = inventory_local_affordance_model_progress(
        plan,
        store_root,
        preparation_git_commit_sha="b" * 40,
        preparation_provenance_sha256="c" * 64,
    )

    assert progress.completed_owner_ids == ()
    assert len(progress.missing_owner_ids) == 480
    assert progress.complete is False
    assert not (store_root / "manifest.json").exists()


def test_inventory_rejects_invalid_expected_provenance_before_store_access(plan, tmp_path: Path):
    absent_store = tmp_path / "also-absent"
    with pytest.raises(LocalAffordancePreparationProgressError, match="provenance"):
        inventory_local_affordance_model_progress(
            plan,
            absent_store,
            preparation_git_commit_sha="b" * 40,
            preparation_provenance_sha256="0" * 64,
        )
    assert not absent_store.exists()
