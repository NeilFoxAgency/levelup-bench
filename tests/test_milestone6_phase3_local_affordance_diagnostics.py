from __future__ import annotations

import pytest

from levelup.experiments.milestone6_phase3_local_affordance_diagnostics import (
    FAMILY_ORDER,
    POPULATION_ORDER,
    LocalAffordanceDiagnosticsError,
    LocalAffordanceQuery,
    aggregate_local_affordance_diagnostics,
    validate_local_affordance_diagnostic_report,
)
from levelup.learning.state_conditioned import (
    IndexedProbeRow,
    ObservableState,
    ObservedTransition,
    TaskProbeRows,
    bind_task_local_affordance_evidence,
    build_affordance_table,
)


def _evidence() -> object:
    rows = []
    for index in range(64):
        before = ObservableState(index / 64, 1 - index / 64, index / 64, 0.5, 0.25, ("a",))
        after = ObservableState(
            (index + 1) / 64,
            1 - (index + 1) / 64,
            (index + 1) / 64,
            0.5 + index / 128,
            0.25,
            ("a",),
        )
        rows.append(IndexedProbeRow(index, ObservedTransition(before, "a", after, False)))
    transitions = tuple(row.transition for row in rows)
    return bind_task_local_affordance_evidence(
        TaskProbeRows(tuple(rows)), build_affordance_table(transitions, target_samples_per_alias=8)
    )


def _queries() -> tuple[LocalAffordanceQuery, ...]:
    evidence = _evidence()
    state = ObservableState(0.5, 0.5, 0.5, 0.5, 0.25, ("a",))
    return tuple(
        LocalAffordanceQuery(population=population, family_id=family, evidence=evidence, states=(state,))
        for population in POPULATION_ORDER
        for family in FAMILY_ORDER
    )


def test_aggregation_is_complete_deterministic_and_fraction_backed() -> None:
    report = aggregate_local_affordance_diagnostics(_queries())
    assert tuple(item.population for item in report.populations) == POPULATION_ORDER
    training = report.for_population("training")
    assert tuple(item.family_id for item in training.family_summaries) == FAMILY_ORDER
    assert training.alias_counts[0].alias == "a"
    assert training.alias_counts[0].count == 6
    assert training.n == 384
    assert training.k_eff == 24
    assert training.eligible == 6
    assert training.local_vs_pooled_outcome_block_byte_difference == 6
    assert training.coverage_gate.fraction.numerator == 1
    assert training.coverage_gate.fraction.denominator == 1
    assert training.coverage_gate.passes is True
    assert report.model_dump(mode="json") == aggregate_local_affordance_diagnostics(
        tuple(reversed(_queries()))
    ).model_dump(mode="json")


def test_missing_population_or_family_fails_closed() -> None:
    queries = _queries()
    with pytest.raises(LocalAffordanceDiagnosticsError, match="coverage"):
        aggregate_local_affordance_diagnostics(queries[:-1])


def test_zero_eligible_fails_closed() -> None:
    # Every outcome vector is identical, so the fixed reducer correctly reports no eligible rows.
    rows = tuple(
        IndexedProbeRow(
            index,
            ObservedTransition(
                ObservableState(0.5, 0.5, 0.5, 0.5, 0.25, ("a",)),
                "a",
                ObservableState(0.5, 0.5, 0.5, 0.5, 0.25, ("a",)),
                False,
            ),
        )
        for index in range(64)
    )
    same = bind_task_local_affordance_evidence(
        TaskProbeRows(rows),
        build_affordance_table(tuple(row.transition for row in rows), target_samples_per_alias=8),
    )
    queries = tuple(item.model_copy(update={"evidence": same}) for item in _queries())
    report = aggregate_local_affordance_diagnostics(queries)
    assert report.for_population("training").eligible == 0
    assert report.for_population("training").coverage_gate.fraction is None
    assert report.for_population("training").coverage_gate.passes is False


def test_forged_summary_and_gate_are_rejected() -> None:
    report = aggregate_local_affordance_diagnostics(_queries())
    payload = report.model_dump(mode="json")
    payload["populations"][0]["family_summaries"] = payload["populations"][0]["family_summaries"][:-1]
    with pytest.raises(LocalAffordanceDiagnosticsError):
        validate_local_affordance_diagnostic_report(payload)

    payload = report.model_dump(mode="json")
    payload["populations"][0]["coverage_gate"]["difference"] = 0
    with pytest.raises(LocalAffordanceDiagnosticsError):
        validate_local_affordance_diagnostic_report(payload)

    payload = report.model_dump(mode="json")
    payload["populations"][0]["coverage_gate"]["threshold"] = {"numerator": 1, "denominator": 2}
    with pytest.raises(LocalAffordanceDiagnosticsError):
        validate_local_affordance_diagnostic_report(payload)

    payload = report.model_dump(mode="json")
    payload["populations"][0]["family_summaries"][0]["coverage_gate"]["threshold"] = {
        "numerator": 4,
        "denominator": 5,
    }
    with pytest.raises(LocalAffordanceDiagnosticsError):
        validate_local_affordance_diagnostic_report(payload)


def test_query_rejects_forged_or_unknown_family() -> None:
    with pytest.raises(ValueError):
        LocalAffordanceQuery(
            population="training", family_id="secret", evidence=_evidence(), states=(ObservableState(0, 1, 0, 0.5, 0.25, ("a",)),)
        )
