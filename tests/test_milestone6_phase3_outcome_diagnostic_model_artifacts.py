from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts as artifacts
import levelup.experiments.milestone6_phase3_outcome_diagnostic_plan as plan_module
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
    OutcomeDiagnosticModelArtifactAuthority,
    OutcomeDiagnosticModelArtifactError,
    OutcomeDiagnosticModelArtifactKey,
    OutcomeDiagnosticModelArtifactRecord,
    OutcomeStateTensorPayload,
    OutcomeTrainingAccounting,
    PinnedOutcomeModelState,
    PinnedOutcomeTrainingEvidence,
    build_outcome_model_artifact_authority,
    build_outcome_model_artifact_key,
    build_outcome_model_artifact_record,
    canonical_outcome_model_artifact_authority_bytes,
    canonical_outcome_model_artifact_record_bytes,
    load_outcome_model_artifact_authority_bytes,
    validate_outcome_model_artifact_against_plan,
    validate_outcome_model_artifact_authority,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    OutcomeModelOwner,
    OutcomePlan,
    OutcomePlannedUnit,
    OutcomeView,
    bind_validated_outcome_diagnostic_plan,
    build_outcome_group_diagnostic_plan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    CONDITIONS,
    OutcomeDiagnosticProtocolSnapshot,
    load_outcome_group_diagnostic_protocol,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import TrainingDataPayload

FAMILIES = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
TRAINING = (
    ("lr0p003-e120", 0.003, 120),
    ("lr0p003-e180", 0.003, 180),
    ("lr0p01-e120", 0.01, 120),
    ("lr0p01-e180", 0.01, 180),
)
TEMPERATURES = ("t0p6", "t0p9", "t1p2")


@pytest.fixture(scope="module")
def monkeypatch_module():
    patch = pytest.MonkeyPatch()
    yield patch
    patch.undo()


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _snapshot() -> OutcomeDiagnosticProtocolSnapshot:
    payload = {
        "schema_version": "test",
        "scope": "known-development-only",
        "execution_boundary": {
            "final_family_access": False,
            "final_method_selection": False,
            "advancement_to_paired_objectives": False,
        },
    }
    payload["diagnostic_protocol_sha256"] = _digest(payload)
    content = canonical_json_bytes(payload)
    return OutcomeDiagnosticProtocolSnapshot(
        repository=Path("."),
        path=Path("protocol.json"),
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        payload=payload,
        authority_bytes=(),
    )


def _state(byte: int = 0) -> PinnedOutcomeModelState:
    rows = []
    for name, shape, _dtype in artifacts.STATE_SCHEMA:
        count = 1
        for dimension in shape:
            count *= dimension
        rows.append(OutcomeStateTensorPayload(name, shape, bytes([byte]) * (4 * count)))
    return PinnedOutcomeModelState(tuple(rows))


def _training_evidence() -> PinnedOutcomeTrainingEvidence:
    return PinnedOutcomeTrainingEvidence(
        TrainingDataPayload.model_construct(samples=()), b"synthetic-test-evidence"
    )


def _full_plan(snapshot: OutcomeDiagnosticProtocolSnapshot) -> OutcomePlan:
    evidence = []
    views = []
    owners = []
    units = []
    for family_index, family in enumerate(FAMILIES):
        for replicate in range(5):
            training_tasks = tuple(f"train-{family}-{replicate}-{index}" for index in range(8))
            evidence.append(
                canonical_json_bytes(
                    {
                        "family_id": family,
                        "replicate": replicate,
                        "payload_sha256": _digest([family, replicate]),
                        "payload_bytes": 100 + family_index * 5 + replicate,
                        "ordered_training_task_ids": list(training_tasks),
                    }
                )
            )
            for condition_index, condition in enumerate(CONDITIONS):
                view_id = _digest(["view", condition, family, replicate])
                mask = _digest(["mask", condition])
                transform = _digest(["transform", condition])
                representation = _digest(["representation", condition])
                views.append(
                    OutcomeView(
                        view_id,
                        condition,
                        f"lofo-{family}",
                        family,
                        replicate,
                        training_tasks,
                        6400000 + family_index * 10 + replicate,
                        _digest(["evidence", family, replicate]),
                        mask,
                        transform,
                        representation,
                    )
                )
                for tuple_index, (training_id, lr, epochs) in enumerate(TRAINING):
                    owner_id = _digest(["owner", condition, family, replicate, training_id])
                    model_seed = (
                        6100000
                        + condition_index * 10000
                        + family_index * 100
                        + replicate * 10
                        + tuple_index
                    )
                    model_identity = _digest(["model", owner_id, model_seed])
                    owners.append(
                        OutcomeModelOwner(
                            owner_id,
                            condition,
                            f"lofo-{family}",
                            family,
                            replicate,
                            training_id,
                            view_id,
                            model_seed,
                            lr,
                            epochs,
                            tuple(f"{training_id}-{temperature}" for temperature in TEMPERATURES),
                            3841,
                            mask,
                            transform,
                            model_identity,
                        )
                    )
                    for task_index in range(8):
                        for temperature_index, temperature in enumerate(TEMPERATURES):
                            tuple_id = f"{training_id}-{temperature}"
                            unit_id = _digest(["unit", owner_id, task_index, temperature_index])
                            units.append(
                                OutcomePlannedUnit(
                                    unit_id,
                                    condition,
                                    tuple_id,
                                    training_id,
                                    f"lofo-{family}",
                                    family,
                                    f"heldout-{family}-{task_index}",
                                    task_index,
                                    replicate,
                                    owner_id,
                                    view_id,
                                    model_seed,
                                    6500000 + task_index,
                                    6200000 + task_index,
                                    6300000 + temperature_index,
                                    6400000 + family_index * 10 + replicate,
                                    _digest(["exposure", family, replicate]),
                                    mask,
                                    transform,
                                    model_identity,
                                    150,
                                    2048,
                                    64,
                                    64,
                                    False,
                                )
                            )
    return OutcomePlan(
        "test-plan",
        "a" * 64,
        "b" * 40,
        snapshot.sha256,
        (),
        FAMILIES,
        (0, 1, 2, 3, 4),
        CONDITIONS,
        tuple(
            f"{training_id}-{temperature}"
            for training_id, _, _ in TRAINING
            for temperature in TEMPERATURES
        ),
        tuple(evidence),
        tuple(views),
        tuple(owners),
        tuple(units),
        False,
    )


@pytest.fixture(scope="module")
def canonical_inputs(monkeypatch_module):
    snapshot = _snapshot()
    plan = _full_plan(snapshot)
    monkeypatch_module.setattr(
        plan_module, "validate_outcome_diagnostic_plan", lambda *args, **kwargs: None
    )
    validated = bind_validated_outcome_diagnostic_plan(plan, snapshot=snapshot)
    monkeypatch_module.setattr(
        artifacts, "validate_outcome_diagnostic_plan", lambda *args, **kwargs: None
    )
    monkeypatch_module.setattr(
        artifacts, "load_outcome_group_diagnostic_protocol", lambda: snapshot
    )
    monkeypatch_module.setattr(artifacts, "_derive_training_example_count", lambda *args: 11)
    return snapshot, plan, validated


@pytest.fixture(scope="module")
def complete_artifacts(canonical_inputs):
    snapshot, plan, validated = canonical_inputs
    payload = _state()
    payloads = {owner.owner_id: payload for owner in plan.model_owners}
    training_evidence = {view.view_id: _training_evidence() for view in plan.views}
    records = []
    for owner in plan.model_owners:
        epochs = owner.training_epochs
        key = build_outcome_model_artifact_key(
            validated,
            snapshot,
            owner_id=owner.owner_id,
            state_payload=payload,
            training_evidence=training_evidence[owner.view_id],
            device="cpu",
            training_accounting=OutcomeTrainingAccounting(
                optimizer_steps=epochs,
                forward_passes=epochs * 11,
                training_examples=11,
                serialization_calls=1,
            ),
        )
        records.append(build_outcome_model_artifact_record(key))
    authority = build_outcome_model_artifact_authority(
        records, payloads, training_evidence, validated, snapshot
    )
    return tuple(records), payloads, training_evidence, authority


def _accounting(epochs: int) -> OutcomeTrainingAccounting:
    return OutcomeTrainingAccounting(
        optimizer_steps=epochs,
        forward_passes=epochs * 11,
        training_examples=11,
        serialization_calls=1,
    )


def _rehashed_record(record, **changes):
    key_body = record.key.model_dump(mode="json")
    key_body.update(changes)
    key_body["key_id"] = _digest({key: value for key, value in key_body.items() if key != "key_id"})
    key = OutcomeDiagnosticModelArtifactKey.model_validate(key_body)
    record_body = {
        "schema_version": record.schema_version,
        "record_id": "0" * 64,
        "key": key.model_dump(mode="json"),
    }
    record_body["record_id"] = _digest(
        {key: value for key, value in record_body.items() if key != "record_id"}
    )
    return OutcomeDiagnosticModelArtifactRecord.model_validate(record_body)


def test_real_canonical_protocol_and_plan_pass_the_artifact_authority_gate() -> None:
    snapshot = load_outcome_group_diagnostic_protocol()
    raw_plan = build_outcome_group_diagnostic_plan(snapshot)
    validated = bind_validated_outcome_diagnostic_plan(raw_plan, snapshot=snapshot)
    canonical, fresh = artifacts._require_canonical_inputs(validated, snapshot)
    assert canonical == raw_plan
    assert fresh.content == snapshot.content
    assert canonical.protocol_sha256 == snapshot.sha256


def test_requires_public_validated_plan_and_fresh_snapshot(canonical_inputs) -> None:
    snapshot, raw_plan, validated = canonical_inputs
    owner = raw_plan.model_owners[0]
    with pytest.raises(OutcomeDiagnosticModelArtifactError, match="validated"):
        build_outcome_model_artifact_key(
            raw_plan,
            snapshot,
            owner_id=owner.owner_id,
            state_payload=_state(),
            training_evidence=_training_evidence(),
            device="cpu",
            training_accounting=_accounting(owner.training_epochs),
        )
    forged = replace(snapshot, sha256="f" * 64)
    with pytest.raises(OutcomeDiagnosticModelArtifactError, match="raw hash|fresh"):
        build_outcome_model_artifact_key(
            validated,
            forged,
            owner_id=owner.owner_id,
            state_payload=_state(),
            training_evidence=_training_evidence(),
            device="cpu",
            training_accounting=_accounting(owner.training_epochs),
        )


def test_state_bytes_schema_and_optimizer_steps_are_recomputed(canonical_inputs) -> None:
    snapshot, plan, validated = canonical_inputs
    owner = plan.model_owners[0]
    with pytest.raises(OutcomeDiagnosticModelArtifactError, match="optimizer"):
        build_outcome_model_artifact_key(
            validated,
            snapshot,
            owner_id=owner.owner_id,
            state_payload=_state(),
            training_evidence=_training_evidence(),
            device="cpu",
            training_accounting=_accounting(owner.training_epochs + 1),
        )


@pytest.mark.parametrize("device", ("mps", "cuda"))
def test_non_cpu_device_is_rejected(canonical_inputs, device) -> None:
    snapshot, plan, validated = canonical_inputs
    owner = plan.model_owners[0]
    with pytest.raises(OutcomeDiagnosticModelArtifactError, match="CPU"):
        build_outcome_model_artifact_key(
            validated,
            snapshot,
            owner_id=owner.owner_id,
            state_payload=_state(),
            training_evidence=_training_evidence(),
            device=device,
            training_accounting=_accounting(owner.training_epochs),
        )


def test_tiny_caller_asserted_example_count_is_rejected(canonical_inputs) -> None:
    snapshot, plan, validated = canonical_inputs
    owner = plan.model_owners[0]
    with pytest.raises(OutcomeDiagnosticModelArtifactError, match="example count"):
        build_outcome_model_artifact_key(
            validated,
            snapshot,
            owner_id=owner.owner_id,
            state_payload=_state(),
            training_evidence=_training_evidence(),
            device="cpu",
            training_accounting=OutcomeTrainingAccounting(
                optimizer_steps=owner.training_epochs,
                forward_passes=owner.training_epochs,
                training_examples=1,
                serialization_calls=1,
            ),
        )
    with pytest.raises(OutcomeDiagnosticModelArtifactError, match="schema"):
        build_outcome_model_artifact_key(
            validated,
            snapshot,
            owner_id=owner.owner_id,
            state_payload=PinnedOutcomeModelState(_state().tensors[:-1]),
            training_evidence=_training_evidence(),
            device="cpu",
            training_accounting=_accounting(owner.training_epochs),
        )


@pytest.mark.parametrize(
    "change",
    (
        {"payload_bytes": True},
        {"ordered_training_task_ids": [0, *[f"task-{index}" for index in range(7)]]},
    ),
)
def test_evidence_schema_types_fail_closed(canonical_inputs, change) -> None:
    snapshot, plan, _validated = canonical_inputs
    first = json.loads(plan.evidence_lineage_rows[0])
    first.update(change)
    mutated = replace(
        plan,
        evidence_lineage_rows=(canonical_json_bytes(first), *plan.evidence_lineage_rows[1:]),
    )
    rebound = bind_validated_outcome_diagnostic_plan(mutated, snapshot=snapshot)
    owner = mutated.model_owners[0]
    with pytest.raises(OutcomeDiagnosticModelArtifactError, match="evidence"):
        build_outcome_model_artifact_key(
            rebound,
            snapshot,
            owner_id=owner.owner_id,
            state_payload=_state(),
            training_evidence=_training_evidence(),
            device="cpu",
            training_accounting=_accounting(owner.training_epochs),
        )


def test_semantic_validation_rejects_substituted_state(
    canonical_inputs, complete_artifacts
) -> None:
    snapshot, _plan, validated = canonical_inputs
    records, _payloads, evidence, _authority = complete_artifacts
    with pytest.raises(OutcomeDiagnosticModelArtifactError, match="differs"):
        validate_outcome_model_artifact_against_plan(
            records[0], _state(1), evidence[records[0].key.view_id], validated, snapshot
        )


def test_semantic_validation_rejects_rehashed_seed_mask_owner_and_evidence(
    canonical_inputs, complete_artifacts
) -> None:
    snapshot, plan, validated = canonical_inputs
    records, payloads, evidence, _authority = complete_artifacts
    original = records[0]
    attacks = (
        {"model_seed": original.key.model_seed + 1},
        {"feature_mask_sha256": "f" * 64},
        {"evidence_payload_sha256": "e" * 64},
        {"owner_id": plan.model_owners[1].owner_id},
        {"condition_id": CONDITIONS[1]},
    )
    for changes in attacks:
        forged = _rehashed_record(original, **changes)
        payload = payloads.get(forged.key.owner_id, _state())
        with pytest.raises(OutcomeDiagnosticModelArtifactError):
            validate_outcome_model_artifact_against_plan(
                forged,
                payload,
                evidence[original.key.view_id],
                validated,
                snapshot,
            )


def test_capacity_and_state_schema_mutations_fail_even_when_rehashed(
    complete_artifacts,
) -> None:
    records, _payloads, _evidence, _authority = complete_artifacts
    original = records[0]
    with pytest.raises(ValueError, match="capacity"):
        _rehashed_record(original, trainable_parameters=3842)
    state = original.key.model_dump(mode="json")["state_schema"]
    state[0]["shape"] = [49]
    state[0]["byte_length"] = 196
    with pytest.raises(ValueError, match="schema"):
        _rehashed_record(original, state_schema=state)


def test_authority_reconstruction_rejects_rehashed_lineage_and_reordering(
    canonical_inputs, complete_artifacts
) -> None:
    snapshot, _plan, validated = canonical_inputs
    records, payloads, evidence, authority = complete_artifacts
    validate_outcome_model_artifact_authority(
        authority, records, payloads, evidence, validated, snapshot
    )
    body = authority.model_dump(mode="json")
    body["plan_parent_commit_sha"] = "c" * 40
    body["authority_sha256"] = _digest(
        {key: value for key, value in body.items() if key != "authority_sha256"}
    )
    mutated = OutcomeDiagnosticModelArtifactAuthority.model_validate(body)
    with pytest.raises(OutcomeDiagnosticModelArtifactError, match="differs"):
        validate_outcome_model_artifact_authority(
            mutated, records, payloads, evidence, validated, snapshot
        )
    reordered = authority.model_dump(mode="json")
    reordered["artifacts"] = list(reversed(reordered["artifacts"]))
    reordered["authority_sha256"] = _digest(
        {key: value for key, value in reordered.items() if key != "authority_sha256"}
    )
    with pytest.raises(ValueError, match="ordered"):
        OutcomeDiagnosticModelArtifactAuthority.model_validate(reordered)


def test_canonical_serializers_revalidate_constructed_objects(complete_artifacts) -> None:
    records, _payloads, _evidence, authority = complete_artifacts
    forged_record = records[0].model_construct(record_id="0" * 64, key=records[0].key)
    with pytest.raises(OutcomeDiagnosticModelArtifactError, match="canonical"):
        canonical_outcome_model_artifact_record_bytes(forged_record)
    forged_authority = authority.model_construct(
        **{**authority.__dict__, "authority_sha256": "0" * 64}
    )
    with pytest.raises(OutcomeDiagnosticModelArtifactError, match="canonical"):
        canonical_outcome_model_artifact_authority_bytes(forged_authority)
    raw = canonical_outcome_model_artifact_authority_bytes(authority)
    assert load_outcome_model_artifact_authority_bytes(raw) == authority


def test_partial_duplicate_extra_and_foreign_records_fail(
    canonical_inputs, complete_artifacts
) -> None:
    snapshot, _plan, validated = canonical_inputs
    records, payloads, evidence, _authority = complete_artifacts
    with pytest.raises(OutcomeDiagnosticModelArtifactError, match="240"):
        build_outcome_model_artifact_authority(
            records[:-1], payloads, evidence, validated, snapshot
        )
    with pytest.raises(OutcomeDiagnosticModelArtifactError, match="partial|foreign"):
        build_outcome_model_artifact_authority(
            (*records[:-1], records[0]), payloads, evidence, validated, snapshot
        )
