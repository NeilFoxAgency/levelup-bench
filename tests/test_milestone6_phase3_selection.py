import hashlib
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import pytest

from levelup.experiments.milestone6_phase3_model_authority import (
    load_phase3_model_artifact_authority_bytes,
)
from levelup.experiments.milestone6_phase3_models import (
    H0_CONDITION,
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
from levelup.experiments.milestone6_phase3_reducer import validate_phase3_matrix
from levelup.experiments.milestone6_phase3_result_store import (
    build_phase3_expected_plan,
)
from levelup.experiments.milestone6_phase3_selection import (
    EXPECTED_TUPLES,
    Phase3ConditionSelection,
    Phase3FamilyMetric,
    Phase3SelectedMetric,
    Phase3SelectionError,
    Phase3SelectionResult,
    evaluate_phase3_claims,
    select_phase3_tuples,
)
from levelup.experiments.runner.records import (
    PhaseAccounting,
    ResourceAccounting,
    SharedArtifactReference,
    UnitOutcome,
    UnitRecord,
)

FAMILIES = ("plain", "battery", "cooldown", "heat", "momentum", "combo")


@lru_cache(maxsize=1)
def _authorities():
    plan = bind_validated_phase3_plan(build_phase3_plan())
    authority = load_phase3_model_artifact_authority_bytes(
        Path("configs/milestone6/phase3_model_artifact_authority.json").read_bytes()
    )
    return plan, authority


@lru_cache(maxsize=1)
def _complete_matrix():
    plan, authority = _authorities()
    owners = {owner.owner_id: owner for owner in plan.plan.model_owners}
    authority_rows = {row.owner_id: row for row in authority.models}
    stores = {
        store.family_id: store for store in build_phase3_expected_plan(plan, authority).stores
    }
    task_indices = {
        family: tuple(
            sorted(
                {
                    item.unit.key.task_index
                    for item in plan.plan.units
                    if item.heldout_family == family
                }
            )
        )
        for family in FAMILIES
    }
    records: list[UnitRecord] = []
    for planned in plan.plan.units:
        owner = owners[planned.model_owner_id]
        authority_row = authority_rows[owner.owner_id]
        store = stores[planned.heldout_family]
        shuffled = planned.base_condition_id == H4_SHUFFLED_CONDITION
        recurrent = planned.base_condition_id in {
            H0_CONDITION,
            H4_CONDITION,
            H4_SHUFFLED_CONDITION,
        }
        parameters = (
            S_PARAMETERS if planned.base_condition_id == S_CONDITION else HISTORY_PARAMETERS
        )
        task_rank = task_indices[planned.heldout_family].index(planned.unit.key.task_index)
        # Tuple 0 has 30/40 successes per family. Tuple 1 is exactly 0.05
        # lower at 28/40 but has a much better restricted-interaction median,
        # so the inclusive tolerance rule must retain and select tuple 1.
        if planned.tuple_id == EXPECTED_TUPLES[0]:
            success = task_rank < 6
            first_hit = 1_000 if success else None
        elif planned.tuple_id == EXPECTED_TUPLES[1]:
            success = task_rank < 5 or (task_rank == 5 and planned.unit.key.replicate < 3)
            first_hit = 100 if success else None
        else:
            success = False
            first_hit = None
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
                    valid=success,
                    completed=success,
                    success=success,
                    performance_metric_id="performance_value",
                    performance_value=1.0 if success else None,
                    performance_direction="minimize",
                    first_optimum_episode=1 if success else None,
                    first_optimum_adaptation_actions=first_hit,
                    censored=not success,
                    censoring_budget=None if success else 2048,
                    censoring_reason=None if success else "fixed_endpoint",
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
                history_shuffle_permutation_map_sha256=(
                    hashlib.sha256(f"shuffle:{planned.unit.unit_id}".encode()).hexdigest()
                    if shuffled
                    else None
                ),
                diagnostics={
                    "model_trainable_parameters": parameters,
                    "model_optimizer_steps": owner.training_epochs,
                    "model_forward_passes": owner.training_epochs,
                    "model_recurrent_steps": 1 if recurrent else 0,
                    "model_training_examples": 1,
                    "history_shuffle_claim_eligible": True if shuffled else None,
                    "history_shuffle_eligible_windows": 10 if shuffled else 0,
                    "history_shuffle_map_nonidentity_windows": 10 if shuffled else 0,
                    "history_shuffle_effective_tensor_changed_windows": 8 if shuffled else 0,
                    "history_shuffle_duplicate_vector_no_effect_windows": 2 if shuffled else 0,
                    "history_shuffle_unchanged_short_windows": 3 if shuffled else 0,
                },
            )
        )
    return validate_phase3_matrix(plan, authority, tuple(records))


def _metric(
    condition: str, minimum: Fraction, *, tuple_id: str = EXPECTED_TUPLES[0]
) -> Phase3SelectedMetric:
    per_family = tuple(
        Phase3FamilyMetric(
            family_id=family,
            units=40,
            successes=int(minimum * 40),
            success_rate=minimum,
            median_restricted_interactions=Fraction(100),
        )
        for family in FAMILIES
    )
    return Phase3SelectedMetric(
        condition_id=condition,
        tuple_id=tuple_id,
        training_tuple_id=tuple_id.rsplit("-t", 1)[0],
        family_metrics=per_family,
        minimum_family_success_rate=minimum,
        worst_family_median_restricted_interactions=Fraction(100),
        macro_average_family_median_restricted_interactions=Fraction(100),
        optimizer_steps=30,
        forward_passes=30,
        recurrent_steps=30,
    )


def _selection(*metrics: Phase3SelectedMetric) -> Phase3SelectionResult:
    return Phase3SelectionResult(
        tuple(
            Phase3ConditionSelection(
                condition_id=metric.condition_id,
                candidates=(metric,),
                best_minimum_family_success_rate=metric.minimum_family_success_rate,
                retained_tuple_ids=(metric.tuple_id,),
                selected=metric,
            )
            for metric in metrics
        )
    )


def test_claims_use_strict_five_percentage_thresholds_and_never_final_access() -> None:
    b2 = _metric("B2-global-listwise-optimum", Fraction(3, 4))
    t = _metric("T-markov-state-transition-listwise-optimum", Fraction(4, 5))
    selection = _selection(
        _metric(S_CONDITION, Fraction(3, 4)),
        _metric(H0_CONDITION, Fraction(7, 10)),
        _metric(H4_CONDITION, Fraction(9, 10)),
        replace(
            _metric(
                H4_SHUFFLED_CONDITION,
                Fraction(4, 5),
                tuple_id=EXPECTED_TUPLES[1],
            ),
            heldout_shuffle_claim_eligible=True,
        ),
    )
    # A 0.05 difference is not robust; the transition claim is deliberately
    # false while the strict H4 comparisons and both shuffle gates pass.
    result = evaluate_phase3_claims(
        selection,
        locked_b2=b2,
        locked_t=t,
        training_shuffle_claim_eligible=True,
    )
    assert result.transition_claim is False
    assert result.history_access_claim is True
    assert result.sequence_order_claim is True
    assert result.advancement_to_paired_objectives is True
    assert result.final_family_access is False


def test_malformed_anchor_is_rejected() -> None:
    malformed = _metric("B2-global-listwise-optimum", Fraction(3, 4))
    malformed = replace(malformed, minimum_family_success_rate=Fraction(1))
    with pytest.raises(Phase3SelectionError):
        evaluate_phase3_claims(
            _selection(
                *(
                    _metric(condition, Fraction(1, 2))
                    for condition in (
                        S_CONDITION,
                        H0_CONDITION,
                        H4_CONDITION,
                        H4_SHUFFLED_CONDITION,
                    )
                )
            ),
            locked_b2=malformed,
            locked_t=_metric("T-markov-state-transition-listwise-optimum", Fraction(1, 2)),
        )


def test_selection_result_requires_exact_new_condition_universe() -> None:
    with pytest.raises(Phase3SelectionError):
        evaluate_phase3_claims(
            {S_CONDITION: _metric(S_CONDITION, Fraction(1, 2))},
            locked_b2=_metric("B2-global-listwise-optimum", Fraction(1, 2)),
            locked_t=_metric("T-markov-state-transition-listwise-optimum", Fraction(1, 2)),
        )


def test_complete_selector_retains_inclusive_band_and_deduplicates_owner_costs() -> None:
    plan, authority = _authorities()
    result = select_phase3_tuples(
        plan,
        authority,
        _complete_matrix(),
        training_shuffle_claim_eligible=True,
    )

    assert tuple(item.condition_id for item in result.condition_selections) == (
        S_CONDITION,
        H0_CONDITION,
        H4_CONDITION,
        H4_SHUFFLED_CONDITION,
    )
    for item in result.condition_selections:
        assert len(item.candidates) == 12
        assert item.best_minimum_family_success_rate == Fraction(3, 4)
        assert item.retained_tuple_ids == EXPECTED_TUPLES[:2]
        assert item.selected.tuple_id == EXPECTED_TUPLES[1]
        assert item.selected.optimizer_steps == 30 * 120
        assert item.selected.forward_passes == 30 * 120
        assert item.selected.recurrent_steps == (0 if item.condition_id == S_CONDITION else 30)
    shuffled = result.by_condition()[H4_SHUFFLED_CONDITION]
    assert shuffled.heldout_shuffle_claim_eligible is True
    assert shuffled.training_shuffle_claim_eligible is True
    assert result.final_family_access is False


def test_selector_rejects_a_forged_validated_matrix_wrapper() -> None:
    plan, authority = _authorities()
    matrix = _complete_matrix()
    forged = matrix.model_copy(
        update={
            "control": matrix.control.model_copy(update={"heldout_search_claim_eligible": False})
        }
    )
    with pytest.raises(Phase3SelectionError, match="differs from canonical"):
        select_phase3_tuples(plan, authority, forged)
