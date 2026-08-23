from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest

from levelup.experiments.milestone6_phase3_model_authority import (
    load_phase3_model_artifact_authority_bytes,
)
from levelup.experiments.milestone6_phase3_plan import (
    FAMILIES,
    ValidatedPhase3Plan,
    bind_validated_phase3_plan,
    build_phase3_plan,
)
from levelup.experiments.milestone6_phase3_result_store import (
    EXPECTED_FAMILY_UNIT_COUNT,
    EXPECTED_TOTAL_UNIT_COUNT,
    Phase3ResultStorePlanError,
    Phase3ResultStoreSpec,
    build_phase3_expected_plan,
    validate_phase3_expected_plan,
)

AUTHORITY_PATH = Path("configs/milestone6/phase3_model_artifact_authority.json")


def _authorities() -> tuple[ValidatedPhase3Plan, object]:
    plan = bind_validated_phase3_plan(build_phase3_plan())
    authority = load_phase3_model_artifact_authority_bytes(AUTHORITY_PATH.read_bytes())
    return plan, authority


def test_result_plan_is_exact_six_family_partition() -> None:
    validated, authority = _authorities()
    result = build_phase3_expected_plan(validated, authority)

    assert result.family_order == FAMILIES
    assert tuple(store.family_id for store in result.stores) == FAMILIES
    assert tuple(len(store.units) for store in result.stores) == (EXPECTED_FAMILY_UNIT_COUNT,) * 6
    assert len(result.units) == EXPECTED_TOTAL_UNIT_COUNT
    assert set(result.unit_ids) == set(validated.plan.unit_ids)
    assert result.plan_id == validated.plan.plan_id == authority.plan_id
    assert result.protocol_sha256 == validated.plan.protocol_sha256 == authority.protocol_sha256
    assert result.model_authority_sha256 == authority.authority_sha256
    assert all(store.final_family_access is False for store in result.stores)


def test_result_plan_copies_exact_units_and_uses_explicit_store_hashes() -> None:
    validated, authority = _authorities()
    result = build_phase3_expected_plan(validated, authority)

    by_family = {
        family: tuple(item for item in validated.plan.units if item.heldout_family == family)
        for family in FAMILIES
    }
    for store in result.stores:
        assert store.units == by_family[store.family_id]
        assert store.store_config_sha256 != store.run_id
        assert len(store.store_config_sha256) == len(store.run_id) == 64
    assert validate_phase3_expected_plan(result, validated, authority) is result


def test_result_store_specs_require_canonical_construction_and_recomputed_hashes() -> None:
    validated, authority = _authorities()
    store = build_phase3_expected_plan(validated, authority).stores[0]
    raw = {
        item.name: getattr(store, item.name)
        for item in fields(store)
        if item.name != "_construction_token"
    }
    with pytest.raises(Phase3ResultStorePlanError, match="construction"):
        Phase3ResultStoreSpec(**raw)
    with pytest.raises(Phase3ResultStorePlanError, match="config or run"):
        replace(store, run_id="0" * 64)


def test_generic_expected_unit_planning_drift_cannot_change_result_identities(monkeypatch) -> None:
    validated, authority = _authorities()
    first = build_phase3_expected_plan(validated, authority)

    # A result-plan build must not consult generic ExperimentConfig planning.
    import levelup.experiments.runner.storage as storage

    monkeypatch.setattr(
        storage,
        "plan_expected_units",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("generic planner used")),
    )
    second = build_phase3_expected_plan(validated, authority)
    assert second == first
    assert tuple(store.run_id for store in second.stores) == tuple(
        store.run_id for store in first.stores
    )


def test_result_plan_rejects_final_plan() -> None:
    validated, authority = _authorities()
    bad_plan = replace(validated.plan, final_family_access=True)
    forged = ValidatedPhase3Plan(
        bad_plan,
        {item.unit.unit_id: item for item in bad_plan.units},
        _construction_token=validated._construction_token,
    )
    with pytest.raises(Phase3ResultStorePlanError, match="final"):
        build_phase3_expected_plan(forged, authority)


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_result_plan_rejects_missing_or_duplicate_unit_material(mutation: str) -> None:
    validated, authority = _authorities()
    if mutation == "missing":
        units = validated.plan.units[:-1]
    else:
        units = validated.plan.units[:-1] + (validated.plan.units[0],)
    bad_plan = replace(validated.plan, units=units)
    forged = ValidatedPhase3Plan(
        bad_plan,
        {item.unit.unit_id: item for item in bad_plan.units},
        _construction_token=validated._construction_token,
    )
    with pytest.raises(Phase3ResultStorePlanError, match="unit"):
        build_phase3_expected_plan(forged, authority)


def test_result_plan_rejects_model_authority_lineage_drift() -> None:
    validated, authority = _authorities()
    drifted = authority.model_copy(update={"plan_id": "0" * 64})
    with pytest.raises(Phase3ResultStorePlanError, match="lineage"):
        build_phase3_expected_plan(validated, drifted)


def test_result_plan_rejects_forged_plan_body_with_reused_plan_id() -> None:
    validated, authority = _authorities()
    first = validated.plan.units[0]
    changed = replace(first, training_tuple_id="forged-training-tuple")
    body = replace(validated.plan, units=(changed, *validated.plan.units[1:]))
    forged = ValidatedPhase3Plan(
        body,
        {item.unit.unit_id: item for item in body.units},
        _construction_token=validated._construction_token,
    )
    with pytest.raises(Phase3ResultStorePlanError, match="plan body"):
        build_phase3_expected_plan(forged, authority)
