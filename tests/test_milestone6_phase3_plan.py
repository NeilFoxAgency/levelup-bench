"""Fail-closed tests for the Phase 3 logical development plan."""

from __future__ import annotations

from dataclasses import replace

import pytest

from levelup.experiments.milestone6_phase2_screening import (
    B2,
    screening_child_configs,
)
from levelup.experiments.milestone6_phase3_plan import (
    NEW_CONDITIONS,
    REPLICATES,
    TRAINING_TUPLE_IDS,
    ValidatedPhase3Plan,
    bind_validated_phase3_plan,
    build_phase3_plan,
    validate_phase3_plan,
)
from levelup.experiments.runner.storage import plan_expected_units


def test_phase3_plan_exact_matrix_and_scope() -> None:
    plan = build_phase3_plan()
    assert plan.final_family_access is False
    assert plan.condition_ids == NEW_CONDITIONS
    assert plan.family_order == ("plain", "battery", "cooldown", "heat", "momentum", "combo")
    assert plan.replicates == REPLICATES
    assert len(plan.candidate_tuple_ids) == 12
    assert len(plan.views) == 120
    assert len(plan.model_owners) == 480
    assert len(plan.units) == 11_520


def test_phase3_plan_temperature_reuse_and_training_tuple_owners() -> None:
    plan = build_phase3_plan()
    assert {owner.training_tuple_id for owner in plan.model_owners} == set(TRAINING_TUPLE_IDS)
    assert all(len(owner.search_temperature_ids) == 3 for owner in plan.model_owners)
    assert all(
        owner.search_temperature_ids == tuple(
            tuple_id
            for tuple_id in plan.candidate_tuple_ids
            if tuple_id.startswith(f"{owner.training_tuple_id}-t")
        )
        for owner in plan.model_owners
    )


def test_phase3_units_have_paired_phase2_seed_channels() -> None:
    plan = build_phase3_plan()
    configs = {
        config.parameters["heldout_family_id"]: config
        for config in screening_child_configs()
    }
    for family in plan.family_order:
        family_rows = [item for item in plan.units if item.heldout_family == family]
        assert len(family_rows) == 4 * 12 * 5 * 8
        phase2 = {
            (
                item.key.condition_id.removeprefix(f"{B2}--"),
                item.key.task_id,
                item.key.replicate,
            ): item
            for item in plan_expected_units(configs[family]).units
            if item.key.phase == "validation" and item.key.condition_id.startswith(f"{B2}--")
        }
        for task_id in {item.unit.key.task_id for item in family_rows}:
            for replicate in REPLICATES:
                for tuple_id in plan.candidate_tuple_ids:
                    rows = [
                        item
                        for item in family_rows
                        if item.unit.key.task_id == task_id
                        and item.unit.key.replicate == replicate
                        and item.tuple_id == tuple_id
                    ]
                    assert len(rows) == len(plan.condition_ids) == 4
                    anchor = phase2[(tuple_id, task_id, replicate)]
                    assert all(item.unit.seeds == anchor.seeds for item in rows)
                    assert all(item.unit.key.task_index == anchor.key.task_index for item in rows)


def test_phase3_plan_rejects_duplicate_unit() -> None:
    plan = build_phase3_plan()
    duplicate = replace(plan, units=plan.units[:-1] + (plan.units[0],))
    with pytest.raises(ValueError, match="unit matrix|unit keys"):
        validate_phase3_plan(duplicate)


def test_phase3_plan_rejects_missing_owner() -> None:
    plan = build_phase3_plan()
    bad = replace(plan, units=(replace(plan.units[0], model_owner_id="0" * 64),) + plan.units[1:])
    with pytest.raises(ValueError, match="missing owner"):
        validate_phase3_plan(bad)


def test_phase3_plan_rejects_rehashed_or_existing_identity_substitution() -> None:
    plan = build_phase3_plan()
    with pytest.raises(ValueError, match="complete frozen authority"):
        validate_phase3_plan(replace(plan, plan_id="0" * 64))
    substituted = replace(
        plan,
        units=(
            replace(plan.units[0], model_owner_id=plan.model_owners[-1].owner_id),
        )
        + plan.units[1:],
    )
    with pytest.raises(ValueError, match="complete frozen authority"):
        validate_phase3_plan(substituted)


def test_phase3_plan_rejects_self_consistent_child_config_substitution() -> None:
    configs = screening_child_configs()
    substituted = configs[0].model_copy(update={"replicates": 4})
    with pytest.raises(ValueError, match="differ from frozen authority"):
        build_phase3_plan(child_configs=(substituted, *configs[1:]))


def test_phase3_plan_is_deterministic() -> None:
    first = build_phase3_plan()
    second = build_phase3_plan()
    assert first.plan_id == second.plan_id
    assert first.views == second.views
    assert first.model_owners == second.model_owners
    assert first.units == second.units


def test_validated_plan_gate_requires_exact_unit_membership() -> None:
    plan = build_phase3_plan()
    with pytest.raises(ValueError, match="canonical plan gate"):
        ValidatedPhase3Plan(plan, {})
    authority = bind_validated_phase3_plan(plan)
    authority.require_unit(plan.units[0])
    changed = replace(
        plan.units[0],
        unit=plan.units[0].unit.model_copy(
            update={"exposure_manifest_sha256": "0" * 64}
        ),
    )
    with pytest.raises(ValueError, match="validated frozen plan"):
        authority.require_unit(changed)
