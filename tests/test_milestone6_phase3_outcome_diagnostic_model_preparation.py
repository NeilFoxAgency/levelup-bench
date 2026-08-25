from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import levelup.experiments.milestone6_phase3_outcome_diagnostic_model_preparation as preparation
import levelup.experiments.milestone6_phase3_outcome_diagnostic_plan as plan_module
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    OutcomeModelOwner,
    OutcomePlan,
    OutcomePlannedUnit,
    OutcomeView,
    ValidatedOutcomePlan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    CONDITIONS,
    OutcomeDiagnosticProtocolSnapshot,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import TrainingDataPayload
from levelup.learning.state_conditioned import DecisionExample, StateConditionedScorer

PREPARATION_GIT_COMMIT = "1" * 40
PREPARATION_PROVENANCE = "2" * 64


def _snapshot() -> OutcomeDiagnosticProtocolSnapshot:
    body = {
        "scope": "known-development-only",
        "execution_boundary": {
            "final_family_access": False,
            "final_method_selection": False,
            "advancement_to_paired_objectives": False,
        },
    }
    content = canonical_json_bytes(body)
    return OutcomeDiagnosticProtocolSnapshot(
        repository=Path("."),
        path=Path("protocol.json"),
        content=content,
        sha256="a" * 64,
        payload=body,
        authority_bytes=(),
    )


def _validated_plan(
    *, consumer_count: int = 24, view_id: str = "view", owner_view_id: str | None = None
) -> ValidatedOutcomePlan:
    condition = CONDITIONS[0]
    owner = OutcomeModelOwner(
        owner_id="owner",
        condition_id=condition,
        fold_id="fold",
        heldout_family="plain",
        replicate=0,
        training_tuple_id="lr0p003-e120",
        view_id=owner_view_id or view_id,
        model_seed=7,
        learning_rate=0.003,
        training_epochs=1,
        search_temperature_ids=(
            "lr0p003-e120-t0p6",
            "lr0p003-e120-t0p9",
            "lr0p003-e120-t1p2",
        ),
        trainable_parameters=3841,
        feature_mask_sha256="b" * 64,
        transformation_sha256="c" * 64,
        model_identity_sha256="d" * 64,
    )
    view = OutcomeView(
        view_id=view_id,
        condition_id=condition,
        fold_id="fold",
        heldout_family="plain",
        replicate=0,
        training_task_ids=("train-0",),
        data_order_seed=1,
        evidence_lineage_sha256="e" * 64,
        feature_mask_sha256="b" * 64,
        transformation_sha256="c" * 64,
        representation_sha256="f" * 64,
    )
    units = tuple(
        OutcomePlannedUnit(
            unit_id=f"unit-{index}",
            condition_id=condition,
            tuple_id="lr0p003-e120-t0p6",
            training_tuple_id="lr0p003-e120",
            fold_id="fold",
            heldout_family="plain",
            task_id=f"task-{index}",
            task_index=index,
            replicate=0,
            model_owner_id="owner",
            view_id=view_id,
            model_seed=7,
            environment_seed=100 + index,
            probe_seed=200 + index,
            search_seed=300 + index,
            data_order_seed=1,
            exposure_manifest_sha256="1" * 64,
            feature_mask_sha256="b" * 64,
            transformation_sha256="c" * 64,
            model_identity_sha256="d" * 64,
            candidate_episodes_per_task=150,
            adaptation_actions_per_task=2048,
            probe_actions_per_task=64,
            maximum_actions_per_candidate_episode=64,
        )
        for index in range(consumer_count)
    )
    plan = OutcomePlan(
        schema_version="test",
        plan_id="plan",
        parent_commit_sha="parent",
        protocol_sha256="a" * 64,
        authority_hashes=(),
        family_order=("plain",),
        replicates=(0,),
        condition_ids=(condition,),
        candidate_tuple_ids=("lr0p003-e120-t0p6",),
        evidence_lineage_rows=(),
        views=(view,),
        model_owners=(owner,),
        units=units,
    )
    return ValidatedOutcomePlan(
        plan,
        {unit.unit_id: unit for unit in units},
        _construction_token=plan_module._TOKEN,
    )


@pytest.fixture
def prepared_inputs(monkeypatch):
    examples = (DecisionExample(torch.zeros((2, 54), dtype=torch.float32), 0),)
    evidence = preparation.PinnedOutcomeTrainingEvidence(
        TrainingDataPayload.model_construct(samples=()), b"evidence-a"
    )
    canonical_record = SimpleNamespace(record_id="record-a")
    forwarded = {}
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)

    class FakeAuthorization:
        def __init__(self, record):
            self.record = record

        def __eq__(self, other):
            return type(other) is type(self) and self.record == other.record

    monkeypatch.setattr(preparation, "AuthorizedOutcomeModelArtifact", FakeAuthorization)
    monkeypatch.setattr(preparation, "_reconstruct_examples", lambda *_: examples)
    monkeypatch.setattr(preparation, "outcome_group_training_examples", lambda *_: examples)
    monkeypatch.setattr(
        preparation,
        "build_outcome_model_artifact_key",
        lambda *args, **kwargs: forwarded.update(key=kwargs) or SimpleNamespace(key_id="key-a"),
    )
    monkeypatch.setattr(
        preparation, "build_outcome_model_artifact_record", lambda *_: canonical_record
    )

    def validate(record, _state, supplied_evidence, *args, **kwargs):
        forwarded["validate"] = {"args": args, "kwargs": kwargs}
        if record is not canonical_record:
            raise ValueError("record drift")
        if supplied_evidence.payload_bytes != b"evidence-a":
            raise ValueError("evidence drift")
        return FakeAuthorization(canonical_record)

    monkeypatch.setattr(preparation, "validate_outcome_model_artifact_against_plan", validate)
    plan = _validated_plan()
    snapshot = _snapshot()
    return plan, snapshot, evidence, canonical_record, forwarded


def _prepare(prepared_inputs):
    plan, snapshot, evidence, _record, _forwarded = prepared_inputs
    return preparation.prepare_outcome_diagnostic_model(
        plan,
        snapshot,
        owner_id="owner",
        training_evidence=evidence,
        preparation_git_commit_sha=PREPARATION_GIT_COMMIT,
        preparation_provenance_sha256=PREPARATION_PROVENANCE,
    )


def test_prepare_is_canonical_cpu_single_thread_and_frozen(prepared_inputs):
    prepared = _prepare(prepared_inputs)
    assert type(prepared.model) is StateConditionedScorer
    assert prepared.model.training is False
    assert all(parameter.device.type == "cpu" for parameter in prepared.model.parameters())
    assert all(parameter.requires_grad is False for parameter in prepared.model.parameters())
    assert torch.get_num_threads() == preparation.TORCH_THREADS
    assert prepared.report.trainable_parameters == preparation.FROZEN_PARAMETER_COUNT
    assert prepared.report.optimizer_steps == 1
    assert prepared.report.forward_passes == len(prepared.examples)
    assert prepared.report.training_examples == len(prepared.examples)
    assert prepared.report.recurrent_steps == 0
    assert tuple(row.name for row in prepared.state_payload.tensors) == tuple(
        sorted(row.name for row in prepared.state_payload.tensors)
    )
    assert all(row.data for row in prepared.state_payload.tensors)


def test_prepare_rejects_runtime_interop_thread_drift(monkeypatch, prepared_inputs):
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 2)
    with pytest.raises(
        preparation.OutcomeDiagnosticModelPreparationError,
        match="interop thread",
    ):
        _prepare(prepared_inputs)


@pytest.mark.parametrize("owner_id", ["foreign-owner", ""])
def test_prepare_rejects_foreign_owner(prepared_inputs, owner_id):
    plan, snapshot, evidence, _record, _forwarded = prepared_inputs
    with pytest.raises(preparation.OutcomeDiagnosticModelPreparationError, match="owner"):
        preparation.prepare_outcome_diagnostic_model(
            plan,
            snapshot,
            owner_id=owner_id,
            training_evidence=evidence,
            preparation_git_commit_sha=PREPARATION_GIT_COMMIT,
            preparation_provenance_sha256=PREPARATION_PROVENANCE,
        )


def test_prepare_rejects_consumer_matrix_drift(monkeypatch, prepared_inputs):
    plan, snapshot, evidence, _record, _forwarded = prepared_inputs
    drifted = _validated_plan(consumer_count=23)
    with pytest.raises(preparation.OutcomeDiagnosticModelPreparationError, match="consumer"):
        preparation.prepare_outcome_diagnostic_model(
            drifted,
            snapshot,
            owner_id="owner",
            training_evidence=evidence,
            preparation_git_commit_sha=PREPARATION_GIT_COMMIT,
            preparation_provenance_sha256=PREPARATION_PROVENANCE,
        )


def test_prepare_rejects_missing_owner_view(prepared_inputs):
    plan, snapshot, evidence, _record, _forwarded = prepared_inputs
    drifted = _validated_plan(owner_view_id="missing-view")
    with pytest.raises(preparation.OutcomeDiagnosticModelPreparationError, match="view"):
        preparation.prepare_outcome_diagnostic_model(
            drifted,
            snapshot,
            owner_id="owner",
            training_evidence=evidence,
            preparation_git_commit_sha=PREPARATION_GIT_COMMIT,
            preparation_provenance_sha256=PREPARATION_PROVENANCE,
        )


def test_validation_rejects_live_state_mutation(prepared_inputs):
    prepared = _prepare(prepared_inputs)
    plan, snapshot, _evidence, _record, _forwarded = prepared_inputs
    next(prepared.model.parameters()).data.add_(1.0)
    with pytest.raises(preparation.OutcomeDiagnosticModelPreparationError, match="state"):
        preparation.validate_prepared_outcome_diagnostic_model(
            prepared,
            plan=plan,
            snapshot=snapshot,
            preparation_git_commit_sha=PREPARATION_GIT_COMMIT,
            preparation_provenance_sha256=PREPARATION_PROVENANCE,
        )


def test_validation_rejects_report_evidence_record_and_authorization_drift(prepared_inputs):
    prepared = _prepare(prepared_inputs)
    plan, snapshot, _evidence, record, forwarded = prepared_inputs
    with pytest.raises(preparation.OutcomeDiagnosticModelPreparationError, match="accounting"):
        preparation.validate_prepared_outcome_diagnostic_model(
            replace(prepared, report=replace(prepared.report, forward_passes=999)),
            plan=plan,
            snapshot=snapshot,
            preparation_git_commit_sha=PREPARATION_GIT_COMMIT,
            preparation_provenance_sha256=PREPARATION_PROVENANCE,
        )
    with pytest.raises(preparation.OutcomeDiagnosticModelPreparationError, match="artifact"):
        preparation.validate_prepared_outcome_diagnostic_model(
            replace(prepared, record=SimpleNamespace(record_id="forged")),
            plan=plan,
            snapshot=snapshot,
            preparation_git_commit_sha=PREPARATION_GIT_COMMIT,
            preparation_provenance_sha256=PREPARATION_PROVENANCE,
        )
    altered_evidence = preparation.PinnedOutcomeTrainingEvidence(
        TrainingDataPayload.model_construct(samples=()), b"evidence-b"
    )
    with pytest.raises(preparation.OutcomeDiagnosticModelPreparationError, match="artifact"):
        preparation.validate_prepared_outcome_diagnostic_model(
            replace(prepared, training_evidence=altered_evidence),
            plan=plan,
            snapshot=snapshot,
            preparation_git_commit_sha=PREPARATION_GIT_COMMIT,
            preparation_provenance_sha256=PREPARATION_PROVENANCE,
        )
    with pytest.raises(preparation.OutcomeDiagnosticModelPreparationError, match="authorization"):
        preparation.validate_prepared_outcome_diagnostic_model(
            replace(
                prepared,
                authorization=SimpleNamespace(record=SimpleNamespace(record_id="forged")),
            ),
            plan=plan,
            snapshot=snapshot,
            preparation_git_commit_sha=PREPARATION_GIT_COMMIT,
            preparation_provenance_sha256=PREPARATION_PROVENANCE,
        )
    assert record.record_id == "record-a"
    assert forwarded["key"]["preparation_git_commit_sha"] == PREPARATION_GIT_COMMIT
    assert forwarded["key"]["preparation_provenance_sha256"] == PREPARATION_PROVENANCE
    assert forwarded["validate"]["kwargs"]["preparation_git_commit_sha"] == PREPARATION_GIT_COMMIT
    assert (
        forwarded["validate"]["kwargs"]["preparation_provenance_sha256"] == PREPARATION_PROVENANCE
    )


def test_source_has_no_environment_evaluator_oracle_access():
    source = Path(preparation.__file__).read_text(encoding="utf-8").lower()
    assert "observableenvironment" not in source
    assert "milestone6_baselines" not in source
    assert "result_store" not in source
