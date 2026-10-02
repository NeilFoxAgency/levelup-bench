from __future__ import annotations

import hashlib

import pytest

from levelup.experiments.milestone6_phase3_local_affordance_model_records import (
    LocalAffordanceModelRecord,
    LocalAffordanceModelRecordError,
    build_model_record,
    validate_model_record,
)
from levelup.experiments.milestone6_phase3_local_affordance_plan import (
    build_local_affordance_plan,
)
from levelup.experiments.runner.config import canonical_json_bytes


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@pytest.fixture(scope="module")
def plan():
    return build_local_affordance_plan()


@pytest.fixture(scope="module")
def record(plan):
    owner = plan.model_owners[0]
    return build_model_record(
        plan=plan,
        owner_id=owner.owner_id,
        training_examples=137,
        training_examples_sha256="1" * 64,
        model_state_sha256="2" * 64,
        model_identity_sha256="3" * 64,
        artifact_bytes=8192,
        artifact_sha256="a" * 64,
        preparation_git_commit_sha="b" * 40,
        preparation_provenance_sha256="c" * 64,
    )


def test_one_owner_record_serves_three_temperatures_and_exact_training_cost(plan, record):
    assert len(plan.model_owners) == 480
    assert record.key.temperature_consumer_ids == plan.model_owners[0].search_temperature_ids
    assert len(record.key.temperature_consumer_ids) == 3
    expected_capacity = 3601 if record.key.condition_id.startswith("B2-") else 3841
    assert record.key.trainable_parameters == expected_capacity
    assert record.cost.optimizer_steps == record.key.optimizer_epochs
    assert record.cost.forward_passes == record.cost.optimizer_steps * 137
    assert record.cost.recurrent_steps == 0
    assert (
        record.cost.probe_actions,
        record.cost.search_actions,
        record.cost.evaluator_calls,
        record.cost.oracle_calls,
    ) == (0, 0, 0, 0)
    validate_model_record(record, plan=plan)
    roundtrip = LocalAffordanceModelRecord.model_validate(record.model_dump(mode="json"))
    assert roundtrip == record
    assert roundtrip.key.training_examples_sha256 == "1" * 64
    assert roundtrip.key.model_state_sha256 == "2" * 64
    assert roundtrip.key.model_identity_sha256 == "3" * 64


def test_frozen_capacity_is_3601_for_b2_and_3841_for_state_conditions(plan):
    for condition_id in (
        "B2-global-listwise-optimum",
        "S-state-availability-listwise-optimum",
        "P-state-availability-alias-pooled-outcome-listwise-optimum",
        "L-state-availability-local-outcome-listwise-optimum",
    ):
        owner = next(item for item in plan.model_owners if item.condition_id == condition_id)
        record = build_model_record(
            plan=plan,
            owner_id=owner.owner_id,
            training_examples=137,
            training_examples_sha256="1" * 64,
            model_state_sha256="2" * 64,
            model_identity_sha256="3" * 64,
            artifact_bytes=8192,
            artifact_sha256="a" * 64,
            preparation_git_commit_sha="b" * 40,
            preparation_provenance_sha256="c" * 64,
        )
        assert record.key.trainable_parameters == (3601 if condition_id.startswith("B2-") else 3841)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("key", "owner_id"), "d" * 64),
        (("key", "trainable_parameters"), 3841),
        (("key", "training_examples_sha256"), "d" * 64),
        (("key", "model_state_sha256"), "d" * 64),
        (("key", "model_identity_sha256"), "d" * 64),
        (("cost", "optimizer_steps"), 999),
    ],
)
def test_tampered_identity_capacity_or_cost_is_rejected(record, path, value):
    payload = record.model_dump(mode="json")
    payload[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        LocalAffordanceModelRecord.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    ("training_examples_sha256", "model_state_sha256", "model_identity_sha256"),
)
def test_zero_model_identity_hash_is_rejected_even_when_hashes_are_recomputed(record, field):
    payload = record.model_dump(mode="json")
    payload["key"][field] = "0" * 64
    key_body = dict(payload["key"])
    key_body.pop("key_sha256")
    payload["key"]["key_sha256"] = _digest(key_body)
    record_body = dict(payload)
    record_body.pop("record_sha256")
    payload["record_sha256"] = _digest(record_body)
    with pytest.raises(ValueError, match="must be nonzero"):
        LocalAffordanceModelRecord.model_validate(payload)


def test_cross_fold_identity_is_rejected_even_with_recomputed_self_hashes(plan, record):
    payload = record.model_dump(mode="json")
    payload["key"]["heldout_family"] = "combo"
    payload["key"]["fold_id"] = "combo"
    key_body = dict(payload["key"])
    key_body.pop("key_sha256")
    payload["key"]["key_sha256"] = _digest(key_body)
    record_body = dict(payload)
    record_body.pop("record_sha256")
    payload["record_sha256"] = _digest(record_body)
    forged = LocalAffordanceModelRecord.model_validate(payload)
    with pytest.raises(LocalAffordanceModelRecordError, match="planned owner"):
        validate_model_record(forged, plan=plan)


def test_provenance_is_bound_to_raw_and_plan_lineage(record, plan):
    payload = record.model_dump(mode="json")
    payload["provenance"]["raw_manifest_file_sha256"] = "e" * 64
    provenance_body = dict(payload["provenance"])
    provenance_body.pop("provenance_sha256")
    payload["provenance"]["provenance_sha256"] = _digest(provenance_body)
    record_body = dict(payload)
    record_body.pop("record_sha256")
    payload["record_sha256"] = _digest(record_body)
    forged = LocalAffordanceModelRecord.model_validate(payload)
    with pytest.raises(LocalAffordanceModelRecordError, match="provenance lineage"):
        validate_model_record(forged, plan=plan)
