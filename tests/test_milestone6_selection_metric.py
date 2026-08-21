from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from levelup.experiments.runner.records import (
    PhaseAccounting,
    ResourceAccounting,
    SharedArtifactReference,
    UnitKey,
    UnitOutcome,
    UnitRecord,
    UnitSeeds,
)
from levelup.experiments.runner.selection_metric import (
    _SPEC_CONSTRUCTION_TOKEN,
    ExpectedSelectionUnit,
    SelectionMetricSpec,
    merge_selection_metric_specs,
    restricted_interactions,
    summarize_variant,
    within_parameter_tolerance,
)

_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _record(
    *,
    family_id: str = "family-a",
    index: int = 0,
    probe_actions: int = 7,
    search_actions: int = 19,
    first_hit: int | None = 26,
    success: bool = True,
    endpoint: int = 64,
    condition_id: str = "variant",
) -> UnitRecord:
    if success:
        outcome = UnitOutcome(
            evaluator_ran=True,
            valid=True,
            completed=True,
            success=True,
            performance_metric_id="score",
            performance_value=1.0,
            performance_direction="minimize",
            first_valid_completion_episode=1,
            first_optimum_episode=2,
            first_optimum_adaptation_actions=first_hit,
        )
    else:
        outcome = UnitOutcome(
            evaluator_ran=True,
            valid=True,
            completed=True,
            success=False,
            performance_metric_id="score",
            performance_value=2.0,
            performance_direction="minimize",
            first_valid_completion_episode=1,
            censored=True,
            censoring_budget=endpoint,
            censoring_reason="fixed_endpoint",
        )
    accounting = ResourceAccounting(
        probes=PhaseAccounting(actions=probe_actions),
        search=PhaseAccounting(actions=search_actions, episodes=2),
        # These phases are deliberately large: they must not enter the restricted metric.
        training=PhaseAccounting(actions=100, forward_passes=17, optimizer_steps=9),
        replay=PhaseAccounting(actions=200, environment_steps=200),
        evaluator=PhaseAccounting(calls=3, actions=300),
    )
    return UnitRecord(
        run_id="selection-metric-test-run",
        config_sha256="a" * 64,
        unit_id=f"{index + 1:064x}",
        key=UnitKey(
            phase="validation",
            condition_id=condition_id,
            family_id=family_id,
            task_id=f"{family_id}-task-{index}",
            task_index=index,
            replicate=0,
        ),
        seeds=UnitSeeds(
            model_seed=index,
            environment_seed=index + 10,
            probe_seed=index + 20,
            search_seed=index + 30,
            data_order_seed=index + 40,
        ),
        exposure_manifest_sha256="b" * 64,
        started_at_utc=_START,
        finished_at_utc=_START + timedelta(seconds=1),
        elapsed_wall_seconds=1.0,
        outcome=outcome,
        accounting=accounting,
    )


def _spec(
    records: tuple[UnitRecord, ...],
    *,
    endpoint: int = 64,
    require_shared_preparation: bool = False,
    family_universe: tuple[str, ...] | None = None,
) -> SelectionMetricSpec:
    if family_universe is None:
        family_universe = tuple(sorted({record.key.family_id for record in records}))
    return SelectionMetricSpec(
        condition_id="variant",
        phase="validation",
        endpoint=endpoint,
        failure_sentinel=endpoint + 1,
        protocol_sha256="c" * 64,
        screening_candidates_sha256="d" * 64,
        task_manifest_sha256="e" * 64,
        family_universe=family_universe,
        expected_units=tuple(
            ExpectedSelectionUnit(
                run_id=record.run_id,
                config_sha256=record.config_sha256,
                unit_id=record.unit_id,
                key=record.key,
                seeds=record.seeds,
                exposure_manifest_sha256=record.exposure_manifest_sha256,
                shared_key_ids=tuple(
                    sorted((item.kind, item.key_id) for item in record.shared_artifacts)
                ),
            )
            for record in records
        ),
        require_shared_preparation=require_shared_preparation,
        _construction_token=_SPEC_CONSTRUCTION_TOKEN,
    )


def _shared_record(record: UnitRecord) -> UnitRecord:
    references = tuple(
        SharedArtifactReference(
            kind=kind,
            key_id=key_character * 64,
            artifact_id=artifact_character * 64,
            cost_id=str(index) * 64,
        )
        for index, (kind, key_character, artifact_character) in enumerate(
            (
                ("training_data_evidence", "a", "d"),
                ("training_data_view", "b", "e"),
                ("training_artifact", "c", "f"),
            ),
            start=1,
        )
    )
    return record.model_copy(
        update={
            "accounting": record.accounting.model_copy(
                update={"training": PhaseAccounting()}
            ),
            "shared_artifacts": references,
            "candidate_generation_sha256": "e" * 64,
        }
    )


def test_restricted_interactions_uses_typed_first_hit_and_paid_adaptation_actions() -> None:
    record = _record(probe_actions=7, search_actions=19, first_hit=26)

    assert restricted_interactions(record, _spec((record,))) == 26


def test_failed_unit_reports_endpoint_plus_one_not_partial_work() -> None:
    record = _record(
        probe_actions=9,
        search_actions=13,
        first_hit=None,
        success=False,
        endpoint=64,
    )

    assert restricted_interactions(record, _spec((record,))) == 65


@pytest.mark.parametrize(
    ("record", "endpoint"),
    [
        pytest.param(
            _record(first_hit=None),
            64,
            id="missing-first-hit",
        ),
        pytest.param(
            _record(probe_actions=10, search_actions=25, first_hit=30),
            32,
            id="overbudget-executed-actions",
        ),
        pytest.param(
            _record(probe_actions=10, search_actions=20, first_hit=9),
            64,
            id="first-hit-before-paid-probes",
        ),
    ],
)
def test_invalid_success_first_hit_is_rejected(record: UnitRecord, endpoint: int) -> None:
    with pytest.raises(ValueError):
        restricted_interactions(record, _spec((record,), endpoint=endpoint))


def test_restricted_metric_excludes_replay_evaluator_and_training_phases() -> None:
    record = _record(probe_actions=5, search_actions=8, first_hit=13)

    assert restricted_interactions(record, _spec((record,))) == 13


def test_summarize_variant_aggregates_within_family_before_equal_family_weighting() -> None:
    records = (
        _record(family_id="family-a", index=0, probe_actions=4, search_actions=6, first_hit=10),
        _record(family_id="family-a", index=1, probe_actions=8, search_actions=12, first_hit=20),
        _record(family_id="family-b", index=2, probe_actions=10, search_actions=20, first_hit=30),
        _record(
            family_id="family-b",
            index=3,
            probe_actions=12,
            search_actions=18,
            first_hit=None,
            success=False,
        ),
    )

    summary = summarize_variant(records, _spec(records))

    assert summary.minimum_family_exact_optimum_success_rate == pytest.approx(0.5)
    assert summary.worst_family_median_restricted_interactions == pytest.approx(47.5)
    assert summary.macro_average_family_median_restricted_interactions == pytest.approx(31.25)
    assert tuple(family.family_id for family in summary.families) == ("family-a", "family-b")
    assert tuple(family.exact_optimum_success_rate for family in summary.families) == (
        1.0,
        0.5,
    )
    # The pooled median would be 25; the selected summary is the macro-average of family medians.
    assert summary.macro_average_family_median_restricted_interactions != pytest.approx(25.0)


def test_summarize_variant_rejects_duplicate_missing_and_extra_units() -> None:
    first = _record(index=0)
    second = _record(index=1)
    spec = _spec((first, second))

    for malformed in ((first, first), (first,), (first, second, _record(index=2))):
        with pytest.raises(ValueError):
            summarize_variant(malformed, spec)


def test_summarize_variant_rejects_incomplete_frozen_family_universe() -> None:
    record = _record(family_id="family-a")
    spec = _spec((record,), family_universe=("family-a", "family-b"))

    with pytest.raises(ValueError, match="complete frozen family universe"):
        summarize_variant((record,), spec)


def test_selection_spec_rejects_final_or_mixed_phase_expected_units() -> None:
    validation = _record(index=0)
    final = _record(index=1).model_copy(
        update={"key": _record(index=1).key.model_copy(update={"phase": "final"})}
    )

    with pytest.raises(ValueError, match="never final"):
        SelectionMetricSpec(
            condition_id="variant",
            phase="final",
            endpoint=64,
            failure_sentinel=65,
            protocol_sha256="c" * 64,
            screening_candidates_sha256="d" * 64,
            task_manifest_sha256="e" * 64,
            family_universe=("family-a",),
            expected_units=(),
            _construction_token=_SPEC_CONSTRUCTION_TOKEN,
        )
    with pytest.raises(ValueError, match="condition or phase"):
        SelectionMetricSpec(
            condition_id="variant",
            phase="validation",
            endpoint=64,
            failure_sentinel=65,
            protocol_sha256="c" * 64,
            screening_candidates_sha256="d" * 64,
            task_manifest_sha256="e" * 64,
            family_universe=("family-a",),
            expected_units=tuple(
                ExpectedSelectionUnit(
                    run_id=record.run_id,
                    config_sha256=record.config_sha256,
                    unit_id=record.unit_id,
                    key=record.key,
                    seeds=record.seeds,
                    exposure_manifest_sha256=record.exposure_manifest_sha256,
                )
                for record in (validation, final)
            ),
            require_shared_preparation=False,
            _construction_token=_SPEC_CONSTRUCTION_TOKEN,
        )


def test_failed_selection_unit_requires_independent_evaluator_evidence() -> None:
    record = _record(first_hit=None, success=False).model_copy(
        update={
            "outcome": _record(first_hit=None, success=False).outcome.model_copy(
                update={"evaluator_ran": False, "valid": False}
            )
        }
    )

    with pytest.raises(ValueError, match="independent evaluator"):
        restricted_interactions(record, _spec((record,)))


def test_direct_metric_rejects_wrong_run_or_config_identity() -> None:
    record = _record()
    spec = _spec((record,))

    for changed in (
        record.model_copy(update={"run_id": "different-run"}),
        record.model_copy(update={"config_sha256": "f" * 64}),
    ):
        with pytest.raises(ValueError, match="selection unit"):
            restricted_interactions(changed, spec)


def test_shared_selection_requires_planned_keys_candidate_hash_and_zero_local_training() -> None:
    record = _shared_record(_record())
    spec = _spec((record,), require_shared_preparation=True)
    assert restricted_interactions(record, spec) == 26

    malformed = (
        record.model_copy(update={"shared_artifacts": record.shared_artifacts[:-1]}),
        record.model_copy(update={"candidate_generation_sha256": None}),
        record.model_copy(
            update={
                "accounting": record.accounting.model_copy(
                    update={"training": PhaseAccounting(optimizer_steps=1)}
                )
            }
        ),
        record.model_copy(
            update={
                "shared_artifacts": (
                    record.shared_artifacts[0].model_copy(update={"key_id": "f" * 64}),
                    *record.shared_artifacts[1:],
                )
            }
        ),
    )
    for changed in malformed:
        with pytest.raises(ValueError):
            restricted_interactions(changed, spec)


def test_merge_selection_specs_requires_compatible_disjoint_families() -> None:
    family_a = _record(family_id="family-a", index=0)
    family_b = _record(family_id="family-b", index=1)
    universe = ("family-a", "family-b")
    left = _spec((family_a,), family_universe=universe)
    right = _spec((family_b,), family_universe=universe)

    merged = merge_selection_metric_specs((left, right))
    assert merged.family_ids == frozenset({"family-a", "family-b"})
    assert len(merged.expected_units) == 2

    with pytest.raises(ValueError, match="at least one"):
        merge_selection_metric_specs(())
    with pytest.raises(ValueError, match="overlapping"):
        merge_selection_metric_specs((left, left))
    with pytest.raises(ValueError, match="family universe"):
        merge_selection_metric_specs((left,))
    incompatible = _spec((family_b,), endpoint=32, family_universe=universe)
    with pytest.raises(ValueError, match="incompatible"):
        merge_selection_metric_specs((left, incompatible))


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (100, 110, True),
        (110, 100, True),
        (100, 90, True),
        (90, 100, True),
        (100, 111, True),
        (111, 100, True),
        (100, 112, False),
        (112, 100, False),
        (100, 89, False),
        (89, 100, False),
    ],
)
def test_within_parameter_tolerance_is_symmetric_at_ten_percent_boundary(
    left: int, right: int, expected: bool
) -> None:
    assert within_parameter_tolerance(left, right, tolerance=0.1) is expected
