from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

from levelup.experiments.milestone6_phase3_model_authority import (
    load_phase3_model_artifact_authority_bytes,
)
from levelup.experiments.milestone6_phase3_models import (
    H4_CONDITION,
    H4_SHUFFLED_CONDITION,
    HISTORY_PARAMETERS,
    S_CONDITION,
    S_PARAMETERS,
)
from levelup.experiments.milestone6_phase3_plan import (
    bind_validated_phase3_plan,
    build_phase3_plan,
)
from levelup.experiments.milestone6_phase3_reducer import (
    EXPECTED_UNIT_COUNT,
    Phase3ReducerError,
    validate_phase3_matrix,
)
from levelup.experiments.milestone6_phase3_result_store import (
    build_phase3_expected_plan,
)
from levelup.experiments.runner.records import (
    PhaseAccounting,
    ResourceAccounting,
    SharedArtifactReference,
    UnitOutcome,
    UnitRecord,
)


@lru_cache(maxsize=1)
def _authorities():
    plan = bind_validated_phase3_plan(build_phase3_plan())
    authority = load_phase3_model_artifact_authority_bytes(
        Path("configs/milestone6/phase3_model_artifact_authority.json").read_bytes()
    )
    return plan, authority


@lru_cache(maxsize=1)
def _records() -> tuple[UnitRecord, ...]:
    plan, authority = _authorities()
    owners = {owner.owner_id: owner for owner in plan.plan.model_owners}
    authority_rows = {row.owner_id: row for row in authority.models}
    stores = {
        store.family_id: store
        for store in build_phase3_expected_plan(plan, authority).stores
    }
    records: list[UnitRecord] = []
    for planned in plan.plan.units:
        owner = owners[planned.model_owner_id]
        authority_row = authority_rows[owner.owner_id]
        store = stores[planned.heldout_family]
        shuffled = planned.base_condition_id == H4_SHUFFLED_CONDITION
        recurrent_steps = 1 if planned.base_condition_id in {
            H4_CONDITION,
            H4_SHUFFLED_CONDITION,
        } else 0
        parameters = (
            S_PARAMETERS if planned.base_condition_id == S_CONDITION else HISTORY_PARAMETERS
        )
        diagnostics = {
            "development_phase3": True,
            "model_trainable_parameters": parameters,
            "model_optimizer_steps": owner.training_epochs,
            "model_forward_passes": owner.training_epochs,
            "model_recurrent_steps": recurrent_steps,
            "model_training_examples": 1,
            "history_shuffle_claim_eligible": True if shuffled else None,
            "history_shuffle_eligible_windows": 10 if shuffled else 0,
            "history_shuffle_map_nonidentity_windows": 10 if shuffled else 0,
            "history_shuffle_effective_tensor_changed_windows": 8 if shuffled else 0,
            "history_shuffle_duplicate_vector_no_effect_windows": 2 if shuffled else 0,
            "history_shuffle_unchanged_short_windows": 3 if shuffled else 0,
        }
        shuffle_sha = (
            hashlib.sha256(f"shuffle:{planned.unit.unit_id}".encode()).hexdigest()
            if shuffled
            else None
        )
        records.append(
            UnitRecord(
                run_id=store.run_id,
                config_sha256=store.store_config_sha256,
                unit_id=planned.unit.unit_id,
                key=planned.unit.key,
                seeds=planned.unit.seeds,
                exposure_manifest_sha256=planned.unit.exposure_manifest_sha256,
                started_at_utc="2026-08-23T00:00:00+00:00",
                finished_at_utc="2026-08-23T00:00:01+00:00",
                elapsed_wall_seconds=1.0,
                outcome=UnitOutcome(
                    evaluator_ran=True,
                    valid=False,
                    completed=False,
                    success=False,
                    performance_metric_id="performance_value",
                    performance_value=None,
                    performance_direction="minimize",
                    censored=True,
                    censoring_budget=2048,
                    censoring_reason="fixed_endpoint",
                ),
                accounting=ResourceAccounting(
                    probes=PhaseAccounting(actions=64),
                    search=PhaseAccounting(episodes=150, actions=1984),
                ),
                shared_artifact=SharedArtifactReference(
                    key_id=authority_row.key_id,
                    artifact_id=authority_row.artifact_id,
                    cost_id=authority_row.cost_id,
                ),
                candidate_generation_sha256=planned.unit.unit_id,
                history_shuffle_permutation_map_sha256=shuffle_sha,
                diagnostics=diagnostics,
            )
        )
    return tuple(records)


def _replace_record(
    records: tuple[UnitRecord, ...],
    index: int,
    replacement: UnitRecord,
) -> tuple[UnitRecord, ...]:
    return (*records[:index], replacement, *records[index + 1 :])


def test_reducer_rejects_untyped_authorities_before_reading_records() -> None:
    with pytest.raises(Phase3ReducerError, match="canonical validated plan"):
        validate_phase3_matrix(SimpleNamespace(), SimpleNamespace(), ())


def test_complete_matrix_is_validated_before_metric_reduction() -> None:
    plan, authority = _authorities()
    result = validate_phase3_matrix(plan, authority, _records())

    assert result.unit_count == EXPECTED_UNIT_COUNT
    assert result.model_owner_count == 480
    assert result.cost.family_counts == {family: 1920 for family in plan.plan.family_order}
    assert result.cost.condition_counts == {
        condition: 2880 for condition in plan.plan.condition_ids
    }
    assert result.control.shuffled_unit_count == 2880
    assert result.control.effective_change_fraction == pytest.approx(0.8)
    assert result.control.heldout_search_claim_eligible is True
    assert tuple(record.unit_id for record in result.records) == plan.plan.unit_ids


def test_reducer_rejects_missing_shuffled_map_digest() -> None:
    plan, authority = _authorities()
    records = _records()
    index = next(
        index
        for index, planned in enumerate(plan.plan.units)
        if planned.base_condition_id == H4_SHUFFLED_CONDITION
    )
    changed = records[index].model_copy(
        update={"history_shuffle_permutation_map_sha256": None}
    )
    with pytest.raises(Phase3ReducerError, match="permutation-map"):
        validate_phase3_matrix(plan, authority, _replace_record(records, index, changed))


def test_reducer_rejects_variant_condition_identity_drift() -> None:
    plan, authority = _authorities()
    records = _records()
    changed_key = records[0].key.model_copy(update={"condition_id": "forged"})
    changed = records[0].model_copy(update={"key": changed_key})
    with pytest.raises(Phase3ReducerError, match="identity"):
        validate_phase3_matrix(plan, authority, _replace_record(records, 0, changed))


def test_reducer_rejects_wrong_store_or_model_authority_reference() -> None:
    plan, authority = _authorities()
    records = _records()
    wrong_store = records[0].model_copy(update={"run_id": "wrong-store"})
    with pytest.raises(Phase3ReducerError, match="run/spec"):
        validate_phase3_matrix(
            plan, authority, _replace_record(records, 0, wrong_store)
        )

    assert records[0].shared_artifact is not None
    wrong_reference = records[0].shared_artifact.model_copy(
        update={"cost_id": "f" * 64}
    )
    with pytest.raises(Phase3ReducerError, match="shared-model"):
        validate_phase3_matrix(
            plan,
            authority,
            _replace_record(
                records,
                0,
                records[0].model_copy(update={"shared_artifact": wrong_reference}),
            ),
        )


def test_reducer_rejects_bad_censoring_and_owner_diagnostics() -> None:
    plan, authority = _authorities()
    records = _records()
    bad_outcome = records[0].outcome.model_copy(update={"censoring_budget": 2047})
    with pytest.raises(Phase3ReducerError, match="fixed-endpoint"):
        validate_phase3_matrix(
            plan,
            authority,
            _replace_record(
                records,
                0,
                records[0].model_copy(update={"outcome": bad_outcome}),
            ),
        )

    bad_evaluator = records[0].outcome.model_copy(update={"evaluator_ran": False})
    with pytest.raises(Phase3ReducerError, match="evaluator"):
        validate_phase3_matrix(
            plan,
            authority,
            _replace_record(
                records,
                0,
                records[0].model_copy(update={"outcome": bad_evaluator}),
            ),
        )

    same_owner = next(
        index
        for index, planned in enumerate(plan.plan.units[1:], start=1)
        if planned.model_owner_id == plan.plan.units[0].model_owner_id
    )
    diagnostics = dict(records[same_owner].diagnostics)
    diagnostics["model_recurrent_steps"] = 1
    with pytest.raises(Phase3ReducerError, match="owner's consumers"):
        validate_phase3_matrix(
            plan,
            authority,
            _replace_record(
                records,
                same_owner,
                records[same_owner].model_copy(update={"diagnostics": diagnostics}),
            ),
        )
