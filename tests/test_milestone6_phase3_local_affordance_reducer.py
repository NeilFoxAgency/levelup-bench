from __future__ import annotations

import pytest
import torch

from levelup.learning.state_conditioned import (
    STATE_AVAILABILITY_ZEROED_INDICES,
    STATE_FEATURE_COUNT,
    AffordanceTable,
    IndexedProbeRow,
    ObservableState,
    ObservedTransition,
    TaskLocalAffordanceEvidence,
    TaskProbeRows,
    apply_state_availability_mask,
    bind_task_local_affordance_evidence,
    build_affordance_table,
    candidate_tensor,
    local_affordance_candidate_view,
    p_candidate_tensor,
    s_candidate_tensor,
    transition_features,
)


def _state(progress: float, aliases: tuple[str, ...] = ("a", "b")) -> ObservableState:
    return ObservableState(
        progress_fraction=progress,
        remaining_fraction=1.0 - progress,
        elapsed_per_target=progress,
        resource_fraction=0.5,
        pressure_fraction=0.25,
        available_aliases=aliases,
    )


def _rows() -> TaskProbeRows:
    rows: list[IndexedProbeRow] = []
    for index in range(64):
        before = _state(index / 64)
        after = _state((index + 1) / 64)
        rows.append(
            IndexedProbeRow(
                index,
                ObservedTransition(before, "a", after, index == 63),
            )
        )
    return TaskProbeRows(tuple(rows))


def test_task_rows_require_exactly_one_complete_indexed_probe() -> None:
    rows = _rows().rows
    with pytest.raises(ValueError, match="exactly 64"):
        TaskProbeRows(rows[:-1])
    with pytest.raises(ValueError, match="canonical"):
        TaskProbeRows(rows[:-1] + (IndexedProbeRow(0, rows[-1].transition),))
    with pytest.raises(ValueError, match="canonical"):
        TaskProbeRows(tuple(reversed(rows)))
    with pytest.raises(ValueError, match="0..63"):
        IndexedProbeRow(64, rows[0].transition)


def test_local_reducer_uses_index_tie_break_and_fixed_k() -> None:
    rows = list(_rows().rows)
    # All rows are equidistant, but outcomes depend on the index.  The reducer must choose
    # canonical probe indices 0..3 rather than artifact insertion or action ordering.
    tied = tuple(
        IndexedProbeRow(
            row.probe_index,
            ObservedTransition(
                _state(0.5),
                "a",
                _state(0.5 + (row.probe_index + 1) / 1000),
                False,
            ),
        )
        for row in rows
    )
    task_rows = TaskProbeRows(tied)
    transitions = tuple(row.transition for row in task_rows.rows)
    table = build_affordance_table(transitions, target_samples_per_alias=8)
    evidence = bind_task_local_affordance_evidence(task_rows, table)
    view = local_affordance_candidate_view(_state(0.5), evidence)
    diagnostic = view.diagnostics[0]
    assert diagnostic.n == 64
    assert diagnostic.k_eff == 4
    assert diagnostic.selected_max_distance == pytest.approx(0.0)
    assert diagnostic.eligible is True
    assert diagnostic.local_used is True
    mean_progress_delta = view.features[0, STATE_FEATURE_COUNT + 4]
    assert mean_progress_delta == pytest.approx((0.001 + 0.002 + 0.003 + 0.004) / 4)
    assert diagnostic.local_vs_pooled_outcome_block_byte_difference is True


def test_local_changes_only_masked_outcome_slots_and_never_leaks_index() -> None:
    task_rows = _rows()
    table = build_affordance_table(tuple(row.transition for row in task_rows.rows), target_samples_per_alias=8)
    evidence = bind_task_local_affordance_evidence(task_rows, table)
    state = _state(0.25, ("a", "unknown"))
    pooled_aliases, pooled, unknown = candidate_tensor(state, table)
    view = local_affordance_candidate_view(state, evidence)
    assert view.aliases == pooled_aliases
    assert view.unknown == unknown == 1
    masked = apply_state_availability_mask(pooled)
    preserved = tuple(index for index in range(masked.shape[1]) if index not in {
        STATE_FEATURE_COUNT + block * 12 + outcome
        for block in range(4)
        for outcome in STATE_AVAILABILITY_ZEROED_INDICES
    })
    assert torch.equal(view.features[:, preserved], masked[:, preserved])
    assert torch.equal(view.features[1], masked[1])
    assert view.diagnostics[1].n == 0
    assert view.diagnostics[1].local_used is False


def test_local_reducer_uses_k_eff_for_aliases_with_fewer_than_four_rows() -> None:
    source = list(_rows().rows)
    limited = tuple(
        IndexedProbeRow(
            row.probe_index,
            ObservedTransition(row.transition.before, "b" if row.probe_index < 2 else "a", row.transition.after, False),
        )
        for row in source
    )
    task_rows = TaskProbeRows(limited)
    table = build_affordance_table(tuple(row.transition for row in task_rows.rows), target_samples_per_alias=8)
    evidence = bind_task_local_affordance_evidence(task_rows, table)
    view = local_affordance_candidate_view(_state(0.25), evidence)
    by_alias = {item.alias: item for item in view.diagnostics}
    assert by_alias["a"].n == 62
    assert by_alias["a"].k_eff == 4
    assert by_alias["b"].n == 2
    assert by_alias["b"].k_eff == 2
    assert by_alias["b"].local_used is False
    assert by_alias["b"].n_less_than_4 is True


def test_binding_rejects_a_pooled_table_from_different_rows() -> None:
    task_rows = _rows()
    table = build_affordance_table(tuple(row.transition for row in task_rows.rows), target_samples_per_alias=8)
    changed = list(table.features["a"])
    changed[4] += 0.25
    mismatched = AffordanceTable({"a": tuple(changed)}, dict(table.sample_counts))
    with pytest.raises(ValueError, match="bitwise-identical"):
        bind_task_local_affordance_evidence(task_rows, mismatched)


def test_binding_retains_an_immutable_canonical_table_copy() -> None:
    task_rows = _rows()
    table = build_affordance_table(
        tuple(row.transition for row in task_rows.rows), target_samples_per_alias=8
    )
    mutable_features = dict(table.features)
    mutable_counts = dict(table.sample_counts)
    supplied = AffordanceTable(mutable_features, mutable_counts)
    evidence = bind_task_local_affordance_evidence(task_rows, supplied)
    mutable_features["a"] = (999.0,) + mutable_features["a"][1:]
    mutable_counts["a"] = 1
    assert evidence.pooled_affordances.features["a"][0] != 999.0
    assert evidence.pooled_affordances.sample_counts["a"] == 64
    with pytest.raises(TypeError):
        evidence.pooled_affordances.features["a"] = mutable_features["a"]  # type: ignore[index]


def test_reducer_rejects_forced_replacement_of_sealed_evidence_fields() -> None:
    task_rows = _rows()
    table = build_affordance_table(
        tuple(row.transition for row in task_rows.rows), target_samples_per_alias=8
    )
    evidence = bind_task_local_affordance_evidence(task_rows, table)
    forged = AffordanceTable({"a": (999.0,) * 49}, {"a": 64})
    object.__setattr__(evidence, "pooled_affordances", forged)
    with pytest.raises(ValueError, match="sealed parity evidence"):
        p_candidate_tensor(_state(0.25), evidence)
    with pytest.raises(ValueError, match="sealed parity evidence"):
        local_affordance_candidate_view(_state(0.25), evidence)


def test_reducer_rejects_evidence_forged_without_a_seal() -> None:
    forged = object.__new__(TaskLocalAffordanceEvidence)
    object.__setattr__(forged, "task_rows", _rows())
    object.__setattr__(forged, "pooled_affordances", AffordanceTable({"a": (0.0,) * 49}, {"a": 64}))
    with pytest.raises(ValueError, match="sealed parity evidence"):
        local_affordance_candidate_view(_state(0.25), forged)


def test_local_reducer_rejects_forced_nested_row_mutation() -> None:
    task_rows = _rows()
    table = build_affordance_table(
        tuple(row.transition for row in task_rows.rows), target_samples_per_alias=8
    )
    evidence = bind_task_local_affordance_evidence(task_rows, table)
    object.__setattr__(evidence.task_rows.rows[0].transition, "action_alias", "forged")
    with pytest.raises(ValueError, match="sealed parity evidence"):
        local_affordance_candidate_view(_state(0.25), evidence)


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), -float("inf")])
def test_local_query_rejects_nonfinite_current_state_coordinates(coordinate: float) -> None:
    task_rows = _rows()
    table = build_affordance_table(tuple(row.transition for row in task_rows.rows), target_samples_per_alias=8)
    evidence = bind_task_local_affordance_evidence(task_rows, table)
    invalid = ObservableState(coordinate, 0.5, 0.0, 0.5, 0.25, ("a",))
    with pytest.raises(ValueError, match="finite numeric"):
        local_affordance_candidate_view(invalid, evidence)


def test_p_and_s_helpers_have_canonical_parity() -> None:
    state = _state(0.3)
    task_rows = _rows()
    table = build_affordance_table(
        tuple(row.transition for row in task_rows.rows), target_samples_per_alias=8
    )
    evidence = bind_task_local_affordance_evidence(task_rows, table)
    p_aliases, p_features, p_unknown = p_candidate_tensor(state, evidence)
    t_aliases, t_features, t_unknown = candidate_tensor(state, table)
    s_aliases, s_features, s_unknown = s_candidate_tensor(state, evidence)
    assert p_aliases == t_aliases
    assert p_unknown == t_unknown
    assert torch.equal(p_features.view(torch.int32), t_features.view(torch.int32))
    assert s_aliases == t_aliases
    assert s_unknown == t_unknown
    assert torch.equal(s_features.view(torch.int32), apply_state_availability_mask(t_features).view(torch.int32))


def test_binding_accepts_terminal_after_state_without_available_actions() -> None:
    source = list(_rows().rows)
    terminal = ObservableState(1.0, 0.0, 1.0, 0.5, 0.25, ())
    source[-1] = IndexedProbeRow(
        63,
        ObservedTransition(source[-1].transition.before, "a", terminal, True),
    )
    task_rows = TaskProbeRows(tuple(source))
    table = build_affordance_table(
        tuple(row.transition for row in task_rows.rows), target_samples_per_alias=8
    )
    evidence = bind_task_local_affordance_evidence(task_rows, table)
    assert evidence.rows[-1].transition.after.available_aliases == ()


def test_local_state_validation_rejects_out_of_bounds_coordinates() -> None:
    task_rows = _rows()
    table = build_affordance_table(
        tuple(row.transition for row in task_rows.rows), target_samples_per_alias=8
    )
    evidence = bind_task_local_affordance_evidence(task_rows, table)
    invalid = ObservableState(1.01, 0.0, 0.0, 0.5, 0.25, ("a",))
    with pytest.raises(ValueError, match="canonical bounds"):
        local_affordance_candidate_view(invalid, evidence)


def test_diagnostic_eligibility_requires_distinct_outcome_vectors() -> None:
    rows = list(_rows().rows)
    same = tuple(
        IndexedProbeRow(row.probe_index, ObservedTransition(_state(0.5), "a", _state(0.5), False))
        for row in rows
    )
    task_rows = TaskProbeRows(same)
    table = build_affordance_table(tuple(row.transition for row in task_rows.rows), target_samples_per_alias=8)
    evidence = bind_task_local_affordance_evidence(task_rows, table)
    diagnostic = local_affordance_candidate_view(_state(0.5), evidence).diagnostics[0]
    assert diagnostic.n == 64
    assert diagnostic.k_eff == 4
    assert diagnostic.eligible is False
    assert diagnostic.local_vs_pooled_outcome_block_byte_difference is False


def test_transition_features_do_not_include_probe_index() -> None:
    row = _rows().rows[3]
    assert len(transition_features(row.transition)) == 12
    # The row index is carried by the wrapper only; the learner-visible transition vector
    # remains the canonical 12-channel encoding.
    assert isinstance(row.probe_index, int)
