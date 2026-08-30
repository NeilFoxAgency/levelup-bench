"""Synthetic tests for the pure outcome-diagnostic reducer boundary."""

from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace

import pytest

from levelup.experiments import milestone6_phase3_outcome_diagnostic_reducer as reducer
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    OutcomeModelOwner,
    OutcomePlannedUnit,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_reducer import (
    CONDITIONS,
    FAILURE_CENSORING_BUDGET,
    FAILURE_SENTINEL,
    MATCHED_S_TUPLE,
    OutcomeDiagnosticCandidateMetric,
    OutcomeDiagnosticConditionSelection,
    OutcomeDiagnosticFamilyMetric,
    OutcomeDiagnosticLockedFamilyMetric,
    OutcomeDiagnosticLockedMetric,
    OutcomeDiagnosticReducerError,
    OutcomeDiagnosticSelectionResult,
    _classify,
    _restricted,
    _tuple_numeric,
    evaluate_outcome_diagnostic_claims,
)
from levelup.experiments.runner.records import (
    PhaseAccounting,
    ResourceAccounting,
    UnitKey,
    UnitOutcome,
    UnitRecord,
    UnitSeeds,
)


def _record(*, success: bool, censored: bool = False, budget: int | None = None) -> UnitRecord:
    return UnitRecord(
        run_id="run",
        config_sha256="a" * 64,
        unit_id="b" * 64,
        key=UnitKey(
            phase="validation", condition_id="condition--tuple", family_id="plain", task_id="t", task_index=0, replicate=0
        ),
        seeds=UnitSeeds(model_seed=1, environment_seed=2, probe_seed=3, search_seed=4, data_order_seed=5),
        exposure_manifest_sha256="c" * 64,
        started_at_utc="2026-08-24T00:00:00+00:00",
        finished_at_utc="2026-08-24T00:00:01+00:00",
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
            first_optimum_adaptation_actions=100 if success else None,
            censored=censored,
            censoring_budget=budget,
            censoring_reason="fixed_endpoint" if censored else None,
        ),
        accounting=ResourceAccounting(
            probes=PhaseAccounting(actions=64), search=PhaseAccounting(actions=100, episodes=1)
        ),
        candidate_generation_sha256="d" * 64,
    )


@dataclass(frozen=True)
class _SyntheticFixture:
    plan: object
    authority: object
    records: tuple[UnitRecord, ...]
    expected: dict[str, OutcomePlannedUnit]
    owners: dict[str, OutcomeModelOwner]


@pytest.fixture(scope="module")
def complete_fixture() -> _SyntheticFixture:
    """A complete 5,760-unit matrix built entirely in memory (no runs/ access)."""
    units: list[OutcomePlannedUnit] = []
    owners: dict[str, OutcomeModelOwner] = {}
    counter = 0
    for condition in CONDITIONS:
        for family in ("plain", "battery", "cooldown", "heat", "momentum", "combo"):
            for replicate in range(5):
                for training in ("lr0p003-e120", "lr0p003-e180", "lr0p01-e120", "lr0p01-e180"):
                    owner_id = f"{len(owners) + 1:064x}"
                    owner = OutcomeModelOwner(owner_id, condition, f"fold-{family}", family, replicate, training, f"{counter + 1:064x}", 6100000 + counter, 0.003 if training.startswith("lr0p003") else 0.01, int(training.rsplit("e", 1)[1]), (f"{training}-t0p6", f"{training}-t0p9", f"{training}-t1p2"), 3841, "a" * 64, "b" * 64, f"{counter + 2:064x}")
                    owners[owner_id] = owner
                    counter += 1
                    for task_index in range(8):
                        task_id = f"task-{task_index}"
                        for temperature in ("0p6", "0p9", "1p2"):
                            tuple_id = f"{training}-t{temperature}"
                            unit_id = f"{len(units) + 1:064x}"
                            units.append(OutcomePlannedUnit(unit_id, condition, tuple_id, training, f"fold-{family}", family, task_id, task_index, replicate, owner_id, owner.view_id, owner.model_seed, 0, 6200000 + counter, 6300000 + counter, 6400000 + counter, "c" * 64, "a" * 64, "b" * 64, owner.model_identity_sha256, 150, 2048, 64, 64, False))
    expected = {unit.unit_id: unit for unit in units}
    plan = SimpleNamespace(plan=SimpleNamespace(final_family_access=False, family_order=("plain", "battery", "cooldown", "heat", "momentum", "combo"), protocol_sha256="b" * 64, plan_id="a" * 64, units=tuple(units)))
    family_ids = {family: tuple(unit.unit_id for unit in units if unit.heldout_family == family) for family in ("plain", "battery", "cooldown", "heat", "momentum", "combo")}
    records: list[UnitRecord] = []
    for unit in units:
        owner = owners[unit.model_owner_id]
        _config, run_id = reducer._store_hashes(unit.heldout_family, "a" * 64, "b" * 64, family_ids[unit.heldout_family])
        success = unit.tuple_id.endswith("t0p6") or unit.tuple_id.endswith("t1p2")
        records.append(UnitRecord(run_id=run_id, config_sha256=_config, unit_id=unit.unit_id, key=UnitKey(phase="validation", condition_id=f"{unit.condition_id}--{unit.tuple_id}", family_id=unit.heldout_family, task_id=unit.task_id, task_index=unit.task_index, replicate=unit.replicate), seeds=UnitSeeds(model_seed=unit.model_seed, environment_seed=unit.environment_seed, probe_seed=unit.probe_seed, search_seed=unit.search_seed, data_order_seed=unit.data_order_seed), exposure_manifest_sha256=unit.exposure_manifest_sha256, started_at_utc="2026-08-24T00:00:00+00:00", finished_at_utc="2026-08-24T00:00:01+00:00", elapsed_wall_seconds=1.0, outcome=UnitOutcome(evaluator_ran=True, valid=success, completed=success, success=success, performance_metric_id="performance_value", performance_value=1.0 if success else None, performance_direction="minimize", first_optimum_episode=1 if success else None, first_optimum_adaptation_actions=64 if success else None, censored=not success, censoring_budget=None if success else 2048, censoring_reason=None if success else "fixed_endpoint"), accounting=ResourceAccounting(probes=PhaseAccounting(actions=64), search=PhaseAccounting(actions=1 if success else 1984, episodes=1 if success else 31)), candidate_generation_sha256="d" * 64, diagnostics={"development_outcome_diagnostic": True, "model_trainable_parameters": 3841, "model_optimizer_steps": owner.training_epochs, "model_forward_passes": owner.training_epochs, "model_training_examples": 1, "model_serialization_calls": 1, "model_recurrent_steps": 0}))
    return _SyntheticFixture(plan, object(), tuple(records), expected, owners)


def _metric(condition: str, value: Fraction, tuple_id: str = MATCHED_S_TUPLE) -> OutcomeDiagnosticCandidateMetric:
    families = tuple(
        OutcomeDiagnosticFamilyMetric(family, 40, int(value * 40), value, Fraction(100))
        for family in ("plain", "battery", "cooldown", "heat", "momentum", "combo")
    )
    return OutcomeDiagnosticCandidateMetric(condition, tuple_id, tuple_id.rsplit("-t", 1)[0], families, value, Fraction(100), Fraction(100), 1, 1, 0)


def _candidate_set(condition: str, value: Fraction) -> tuple[OutcomeDiagnosticCandidateMetric, ...]:
    return tuple(_metric(condition, value, tuple_id) for tuple_id in reducer.EXPECTED_TUPLES)


def _locked(condition: str, value: Fraction) -> OutcomeDiagnosticLockedMetric:
    families = tuple(
        OutcomeDiagnosticLockedFamilyMetric(
            family, 40, int(value * 40), value, Fraction(100)
        )
        for family in ("plain", "battery", "cooldown", "heat", "momentum", "combo")
    )
    return OutcomeDiagnosticLockedMetric(
        condition,
        MATCHED_S_TUPLE if condition.startswith("S-") else "lr0p003-e120-t1p2",
        "lr0p01-e120" if condition.startswith("S-") else "lr0p003-e120",
        families,
        value,
        Fraction(100),
        Fraction(100),
        1,
        1,
        0,
    )


def test_restricted_interactions_use_typed_success_and_failure_sentinel() -> None:
    assert _restricted(_record(success=True)) == 100
    assert _restricted(_record(success=False, censored=True, budget=FAILURE_CENSORING_BUDGET)) == FAILURE_SENTINEL
    with pytest.raises(OutcomeDiagnosticReducerError):
        _restricted(_record(success=False))


def test_complete_synthetic_matrix_validates_and_selects(
    complete_fixture: _SyntheticFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reducer, "_authority_and_plan", lambda _plan, _authority: (complete_fixture.expected, complete_fixture.owners, {}))
    matrix = reducer.validate_outcome_diagnostic_matrix(complete_fixture.plan, complete_fixture.authority, complete_fixture.records)
    assert matrix.unit_count == 5760
    assert matrix.model_owner_count == 240
    assert matrix.cost.model_owner_consumer_count == 5760
    locked_s = _locked("S-state-availability-listwise-optimum", Fraction(1, 2))
    selection = reducer.select_outcome_diagnostic_tuples(complete_fixture.plan, complete_fixture.authority, matrix, locked_s=locked_s)
    assert len(selection.condition_selections) == 2
    assert all(len(row.candidates) == 12 for row in selection.condition_selections)


@pytest.mark.parametrize("tamper", ["missing", "duplicate", "extra", "key", "seed", "store", "censoring", "diagnostic"])
def test_complete_matrix_tamper_fails_closed(
    complete_fixture: _SyntheticFixture, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    monkeypatch.setattr(reducer, "_authority_and_plan", lambda _plan, _authority: (complete_fixture.expected, complete_fixture.owners, {}))
    records = list(complete_fixture.records)
    if tamper == "missing":
        records.pop()
    elif tamper == "duplicate":
        records[-1] = records[0]
    elif tamper == "extra":
        records[-1] = records[-1].model_copy(update={"unit_id": "f" * 64})
    elif tamper == "key":
        records[0] = records[0].model_copy(update={"key": records[0].key.model_copy(update={"task_index": 7})})
    elif tamper == "seed":
        records[0] = records[0].model_copy(update={"seeds": records[0].seeds.model_copy(update={"search_seed": 99})})
    elif tamper == "store":
        records[0] = records[0].model_copy(update={"run_id": "bad"})
    elif tamper == "censoring":
        outcome = records[0].outcome.model_copy(update={"first_optimum_adaptation_actions": 10})
        records[0] = records[0].model_copy(update={"outcome": outcome})
    else:
        diagnostics = dict(records[0].diagnostics)
        diagnostics["model_serialization_calls"] = 2
        records[0] = records[0].model_copy(update={"diagnostics": diagnostics})
    with pytest.raises(OutcomeDiagnosticReducerError):
        reducer.validate_outcome_diagnostic_matrix(complete_fixture.plan, complete_fixture.authority, records)


def test_tuple_numeric_and_inclusive_tolerance_boundary_are_exact() -> None:
    assert _tuple_numeric("lr0p003-e120-t0p6") < _tuple_numeric("lr0p01-e120-t0p6")
    assert _classify(Fraction(1, 20), Fraction(1, 20), [Fraction(0)], [Fraction(0)]) == "inconclusive"
    assert _classify(Fraction(1, 20) + Fraction(1, 100), Fraction(1, 20) + Fraction(1, 100), [Fraction(0)], [Fraction(0)]) == "robust_gain"


def test_claims_report_exact_deltas_and_possible_interaction_without_final_access() -> None:
    locked_s = _locked("S-state-availability-listwise-optimum", Fraction(1, 2))
    locked_t = _locked("T-markov-state-transition-listwise-optimum", Fraction(1, 4))
    selections = tuple(
        OutcomeDiagnosticConditionSelection(
            condition,
            _candidate_set(condition, Fraction(1, 2)),
            Fraction(1, 2),
            reducer.EXPECTED_TUPLES,
            _metric(condition, Fraction(1, 2), reducer.EXPECTED_TUPLES[0]),
        )
        for condition in CONDITIONS
    )
    result = evaluate_outcome_diagnostic_claims(OutcomeDiagnosticSelectionResult(selections), locked_s=locked_s, locked_t=locked_t)
    assert result.inconclusive
    assert result.t_delta_vs_s == Fraction(-1, 4)
    assert result.possible_interaction_hypothesis
    assert result.final_family_access is False


def test_claims_reject_duplicate_condition_rows_in_forged_selection_trace() -> None:
    """Convenience lookups must not collapse a forged duplicate condition row."""
    locked_s = _locked("S-state-availability-listwise-optimum", Fraction(1, 2))
    locked_t = _locked("T-markov-state-transition-listwise-optimum", Fraction(1, 4))
    rows = tuple(
        OutcomeDiagnosticConditionSelection(
            condition,
            _candidate_set(condition, Fraction(1, 2)),
            Fraction(1, 2),
            reducer.EXPECTED_TUPLES,
            _metric(condition, Fraction(1, 2), reducer.EXPECTED_TUPLES[0]),
        )
        for condition in CONDITIONS
    )
    forged = OutcomeDiagnosticSelectionResult((rows[0], rows[1], rows[0]))
    with pytest.raises(OutcomeDiagnosticReducerError, match="condition matrix"):
        evaluate_outcome_diagnostic_claims(
            forged, locked_s=locked_s, locked_t=locked_t
        )


def test_claims_recompute_retention_and_tie_break_from_candidates() -> None:
    """Stale trace fields cannot override the reducer's frozen selection rule."""
    locked_s = _locked("S-state-availability-listwise-optimum", Fraction(1, 2))
    locked_t = _locked("T-markov-state-transition-listwise-optimum", Fraction(1, 4))
    rows = []
    for condition in CONDITIONS:
        candidates = list(_candidate_set(condition, Fraction(1, 2)))
        # Make the final tuple the unique best candidate while retaining a
        # forged trace that claims the first tuple was selected.
        winner = candidates[-1]
        family_metrics = tuple(
            row.__class__(
                row.family_id,
                row.units,
                30,
                Fraction(3, 4),
                row.median_restricted_interactions,
            )
            for row in winner.family_metrics
        )
        candidates[-1] = winner.__class__(
            winner.condition_id,
            winner.tuple_id,
            winner.training_tuple_id,
            family_metrics,
            Fraction(3, 4),
            winner.worst_family_median_restricted_interactions,
            winner.macro_average_family_median_restricted_interactions,
            winner.optimizer_steps,
            winner.forward_passes,
            winner.recurrent_steps,
        )
        rows.append(
            OutcomeDiagnosticConditionSelection(
                condition,
                tuple(candidates),
                Fraction(1, 2),
                reducer.EXPECTED_TUPLES,
                candidates[0],
            )
        )
    with pytest.raises(OutcomeDiagnosticReducerError, match="selection trace"):
        evaluate_outcome_diagnostic_claims(
            OutcomeDiagnosticSelectionResult(tuple(rows)),
            locked_s=locked_s,
            locked_t=locked_t,
        )


def test_claims_recompute_tie_break_when_primary_rates_are_equal() -> None:
    locked_s = _locked("S-state-availability-listwise-optimum", Fraction(1, 2))
    locked_t = _locked("T-markov-state-transition-listwise-optimum", Fraction(1, 4))
    rows = []
    for condition in CONDITIONS:
        candidates = list(_candidate_set(condition, Fraction(1, 2)))
        winner = candidates[-1]
        families = tuple(
            row.__class__(
                row.family_id,
                row.units,
                row.successes,
                row.success_rate,
                Fraction(50),
            )
            for row in winner.family_metrics
        )
        candidates[-1] = winner.__class__(
            winner.condition_id,
            winner.tuple_id,
            winner.training_tuple_id,
            families,
            winner.minimum_family_success_rate,
            Fraction(50),
            Fraction(50),
            winner.optimizer_steps,
            winner.forward_passes,
            winner.recurrent_steps,
        )
        rows.append(
            OutcomeDiagnosticConditionSelection(
                condition,
                tuple(candidates),
                Fraction(1, 2),
                reducer.EXPECTED_TUPLES,
                candidates[0],
            )
        )
    with pytest.raises(OutcomeDiagnosticReducerError, match="selection trace"):
        evaluate_outcome_diagnostic_claims(
            OutcomeDiagnosticSelectionResult(tuple(rows)),
            locked_s=locked_s,
            locked_t=locked_t,
        )


def test_one_group_harm_does_not_trigger_interaction_flag() -> None:
    locked_s = _locked("S-state-availability-listwise-optimum", Fraction(1, 2))
    locked_t = _locked("T-markov-state-transition-listwise-optimum", Fraction(1, 4))
    selections = []
    for index, condition in enumerate(CONDITIONS):
        value = Fraction(1, 4) if index == 0 else Fraction(1, 2)
        candidates = _candidate_set(condition, value)
        selected = candidates[0]
        selections.append(OutcomeDiagnosticConditionSelection(condition, candidates, value, reducer.EXPECTED_TUPLES, selected))
    result = evaluate_outcome_diagnostic_claims(OutcomeDiagnosticSelectionResult(tuple(selections)), locked_s=locked_s, locked_t=locked_t)
    assert result.rp_robust_harm
    assert not result.pec_robust_harm
    assert not result.robust_group_harm
    assert not result.possible_interaction_hypothesis


def test_t_lower_than_s_is_sufficient_for_interaction_when_groups_not_harm() -> None:
    locked_s = _locked("S-state-availability-listwise-optimum", Fraction(1, 2))
    locked_t = _locked("T-markov-state-transition-listwise-optimum", Fraction(19, 40))
    selections = tuple(
        OutcomeDiagnosticConditionSelection(
            condition,
            _candidate_set(condition, Fraction(1, 2)),
            Fraction(1, 2),
            reducer.EXPECTED_TUPLES,
            _metric(condition, Fraction(1, 2), reducer.EXPECTED_TUPLES[0]),
        )
        for condition in CONDITIONS
    )
    result = evaluate_outcome_diagnostic_claims(OutcomeDiagnosticSelectionResult(selections), locked_s=locked_s, locked_t=locked_t)
    assert result.possible_interaction_hypothesis
