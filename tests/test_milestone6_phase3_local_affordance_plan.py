from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from levelup.experiments.milestone6_phase3_local_affordance_plan import (
    CONDITIONS,
    FAMILIES,
    TUPLE_IDS,
    LocalAffordancePlanError,
    build_local_affordance_plan,
    canonical_local_affordance_plan_lock_bytes,
    load_committed_local_affordance_plan_lock,
    validate_local_affordance_plan,
    validate_local_affordance_plan_lock_bytes,
)


def test_complete_development_matrix_and_temperature_reuse() -> None:
    plan = build_local_affordance_plan()
    assert plan.final_family_access is False
    assert plan.family_order == FAMILIES
    assert plan.condition_ids == CONDITIONS
    assert plan.candidate_tuple_ids == TUPLE_IDS
    assert len(plan.views) == 120
    assert len(plan.model_owners) == 480
    assert len(plan.units) == 11_520
    assert dict(plan.source_sha256)["raw_capture_summary"] == "9e853e4c099c0d49d2ffe9243f1917522f40dc710fbae9a421c4d3dfdba385cb"
    assert len({owner.owner_id for owner in plan.model_owners}) == 480
    assert len({unit.unit.unit_id for unit in plan.units}) == 11_520
    assert all(len(owner.search_temperature_ids) == 3 for owner in plan.model_owners)
    assert all(unit.unit.key.phase == "validation" for unit in plan.units)


def test_seed_and_fold_identity_are_exact() -> None:
    plan = build_local_affordance_plan()
    first = plan.units[0].unit
    assert first.key.family_id == "plain"
    assert first.key.condition_id.startswith("B2-global-listwise-optimum--")
    assert first.key.task_index == 1
    assert first.seeds.model_seed == 6_100_000
    assert first.seeds.environment_seed == 0
    assert first.seeds.probe_seed == 6_200_001
    assert first.seeds.search_seed == 6_300_001
    assert first.seeds.data_order_seed == 6_400_000
    assert all(owner.trainable_parameters in (3601, 3841) for owner in plan.model_owners)
    assert sum(owner.trainable_parameters == 3601 for owner in plan.model_owners) == 120


def test_plan_validation_fails_closed_on_identity_drift() -> None:
    plan = build_local_affordance_plan()
    altered = dataclasses.replace(plan, final_family_access=True)
    with pytest.raises(LocalAffordancePlanError):
        validate_local_affordance_plan(altered)

    owner = plan.model_owners[0]
    altered_owner = dataclasses.replace(owner, trainable_parameters=3841)
    altered_owners = (altered_owner,) + plan.model_owners[1:]
    altered_plan = dataclasses.replace(plan, model_owners=altered_owners)
    with pytest.raises(LocalAffordancePlanError):
        validate_local_affordance_plan(altered_plan)


def test_canonical_lock_round_trip_and_tamper_rejection() -> None:
    plan = build_local_affordance_plan()
    lock = canonical_local_affordance_plan_lock_bytes(plan)
    assert validate_local_affordance_plan_lock_bytes(lock).plan_id == plan.plan_id
    tampered = lock.replace(plan.plan_id.encode(), b"0" * 64)
    with pytest.raises(LocalAffordancePlanError):
        validate_local_affordance_plan_lock_bytes(tampered)


def test_committed_plan_lock_is_exact_canonical_authority() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/milestone6/phase3_local_affordance_plan_lock.json"
    )
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == (
        "5b2b73e15607375ec83961c566f73a5d92b68ebf27de523fd98a9122992046e4"
    )
    assert load_committed_local_affordance_plan_lock(path) == (
        build_local_affordance_plan()
    )
