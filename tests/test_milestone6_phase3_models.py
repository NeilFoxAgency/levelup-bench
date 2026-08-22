from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from levelup.experiments.milestone6_phase3_models import (
    H0_CONDITION,
    H4_CONDITION,
    H4_SHUFFLED_CONDITION,
    S_CONDITION,
    HistoryShuffleDiagnostics,
    Phase3ModelPreparationError,
    prepare_phase3_model,
    prepare_phase3_view,
)
from levelup.experiments.milestone6_phase3_plan import Phase3ModelOwner, Phase3View
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import (
    AffordanceTableRecord,
    ObservableStateRecord,
    ObservableTraceRecord,
    ObservedTransitionRecord,
    TrainingDataPayload,
    TrainingDataSample,
)


def _payload() -> tuple[TrainingDataPayload, bytes, dict[str, object]]:
    states = tuple(
        ObservableStateRecord(
            progress_fraction=index / 4,
            remaining_fraction=1 - index / 4,
            elapsed_per_target=index / 4,
            resource_fraction=0.5 + index / 20,
            pressure_fraction=0.25,
            available_aliases=("a", "b"),
        )
        for index in range(4)
    )
    trace = ObservableTraceRecord(
        transitions=tuple(
            ObservedTransitionRecord(
                before=states[index],
                action_alias="a",
                after=states[index + 1],
                completed=index == 2,
            )
            for index in range(3)
        )
    )
    payload = TrainingDataPayload(
        samples=(
            TrainingDataSample(
                task_id="task-0",
                trace=trace,
                affordances=AffordanceTableRecord(
                    features={"a": (0.1,) * 49, "b": (0.2,) * 49},
                    sample_counts={"a": 2, "b": 2},
                ),
            ),
        )
    )
    encoded = canonical_json_bytes(payload.model_dump(mode="json"))
    manifest = {
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "payload_bytes": len(encoded),
        "sample_task_ids": ("task-0",),
        "key": {
            "protocol_sha256": "1" * 64,
            "expected_unit_plan_sha256": "2" * 64,
            "ordered_training_task_ids": ("task-0",),
            "fold_id": "fold-0",
            "heldout_family_id": "combo",
            "replicate": 0,
            "data_order_seed": 17,
            "condition_id": "C-state-conditioned-listwise-optimum",
        },
    }
    return payload, encoded, manifest


def _view(condition: str) -> Phase3View:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "phase2_config_sha256": "1" * 64,
                "expected_units_sha256": "2" * 64,
                "training_task_ids": ("task-0",),
                "replicate": 0,
            }
        )
    ).hexdigest()
    return Phase3View(
        view_id="b" * 64,
        condition_id=condition,
        fold_id="fold-0",
        heldout_family="combo",
        replicate=0,
        training_task_ids=("task-0",),
        data_order_seed=17,
        evidence_lineage_sha256=digest,
        representation_sha256="c" * 64,
    )


def _owner(view: Phase3View) -> Phase3ModelOwner:
    return Phase3ModelOwner(
        owner_id="d" * 64,
        condition_id=view.condition_id,
        fold_id=view.fold_id,
        heldout_family=view.heldout_family,
        replicate=view.replicate,
        training_tuple_id="lr0p003-e120",
        view_id=view.view_id,
        model_seed=4,
        learning_rate=0.003,
        training_epochs=1,
        search_temperature_ids=(
            "lr0p003-e120-t0p6",
            "lr0p003-e120-t0p9",
            "lr0p003-e120-t1p2",
        ),
    )


@pytest.mark.parametrize("condition", (S_CONDITION, H0_CONDITION, H4_CONDITION, H4_SHUFFLED_CONDITION))
def test_view_and_model_use_one_evidence_payload(condition: str) -> None:
    payload, encoded, manifest = _payload()
    view = _view(condition)
    prepared = prepare_phase3_view(
        payload,
        manifest,
        view,
        payload_bytes=encoded,
        trace_or_episode_ids=("optimum:task-0:0",),
        _allow_test_identity=True,
    )
    assert prepared.sample_task_ids == ("task-0",)
    assert len(prepared.transition_examples) == len(prepared.examples) == 3
    if condition == H4_SHUFFLED_CONDITION:
        assert prepared.history_shuffle is not None
        assert prepared.history_shuffle.permutation_map_sha256
    model = prepare_phase3_model(
        prepared, _owner(view), _allow_test_identity=True
    )
    expected = 3841 if condition == S_CONDITION else 3889
    assert model.report.trainable_parameters == expected
    assert model.report.optimizer_steps == 1
    assert model.report.training_examples == 3
    assert model.search_temperature_ids == _owner(view).search_temperature_ids


def test_evidence_substitution_and_task_order_fail_closed() -> None:
    payload, encoded, manifest = _payload()
    view = _view(S_CONDITION)
    with pytest.raises(Phase3ModelPreparationError, match="bytes"):
        prepare_phase3_view(
            payload,
            manifest,
            view,
            payload_bytes=encoded + b"x",
            _allow_test_identity=True,
        )
    with pytest.raises(Phase3ModelPreparationError, match="task order"):
        changed = {**manifest, "sample_task_ids": ("other-task",)}
        prepare_phase3_view(
            payload,
            changed,
            view,
            payload_bytes=encoded,
            _allow_test_identity=True,
        )
    with pytest.raises(Phase3ModelPreparationError, match="evidence lineage"):
        prepare_phase3_view(
            payload,
            manifest,
            replace(view, evidence_lineage_sha256="a" * 64),
            payload_bytes=encoded,
            _allow_test_identity=True,
        )


def test_production_view_requires_frozen_plan_authority() -> None:
    payload, encoded, manifest = _payload()
    with pytest.raises(Phase3ModelPreparationError, match="validated Phase 3 plan"):
        prepare_phase3_view(
            payload,
            manifest,
            _view(S_CONDITION),
            payload_bytes=encoded,
        )


def test_empty_shuffle_coverage_is_not_claim_eligible() -> None:
    diagnostics = HistoryShuffleDiagnostics(0, 0, 0, 0, 1, "a" * 64)
    assert diagnostics.effective_change_fraction == 1.0
    assert diagnostics.claim_eligible is False


def test_model_owner_temperature_and_view_lineage_are_frozen() -> None:
    payload, encoded, manifest = _payload()
    view = _view(S_CONDITION)
    prepared = prepare_phase3_view(
        payload,
        manifest,
        view,
        payload_bytes=encoded,
        _allow_test_identity=True,
    )
    owner = _owner(view)
    with pytest.raises(Phase3ModelPreparationError, match="temperature"):
        prepare_phase3_model(
            prepared,
            replace(owner, search_temperature_ids=("bad",) * 3),
            _allow_test_identity=True,
        )
    with pytest.raises(Phase3ModelPreparationError, match="view identity"):
        prepare_phase3_model(
            prepared,
            replace(owner, view_id="e" * 64),
            _allow_test_identity=True,
        )
