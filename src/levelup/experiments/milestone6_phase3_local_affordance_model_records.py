"""Typed, metadata-only records for Phase 3 local-affordance models.

The module defines the canonical identity, cost, artifact, and provenance
envelopes for the 480 frozen development model owners.  It deliberately does
not read or write files, train models, inspect outcomes, or access final data.
One owner is trained once and is shared by its three temperature consumers.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from levelup.experiments.milestone6_phase3_local_affordance_plan import (
    CONDITIONS,
    EXPECTED_COUNTS,
    FAMILIES,
    FROZEN_SOURCE_SHA256,
    TRAINING_TUPLE_IDS,
    LocalAffordancePlan,
    _as_json,
)
from levelup.experiments.milestone6_phase3_local_affordance_plan import (
    SCHEMA_VERSION as PLAN_SCHEMA_VERSION,
)
from levelup.experiments.runner.config import canonical_json_bytes

HEX64 = r"^[0-9a-f]{64}$"
COMMIT = r"^[0-9a-f]{40,64}$"
SCHEMA_VERSION = "milestone6.phase3.local-affordance-model-record.v1"
KEY_SCHEMA_VERSION = "milestone6.phase3.local-affordance-model-key.v1"
COST_SCHEMA_VERSION = "milestone6.phase3.local-affordance-model-cost.v1"
PROVENANCE_SCHEMA_VERSION = "milestone6.phase3.local-affordance-model-provenance.v1"
TEMPERATURES = ("t0p6", "t0p9", "t1p2")
FROZEN_PLAN_ID = "4e3390056a3a75fad36a033eecf4731dbd08ee946c354449f489c31f5a310718"
FROZEN_RAW_LINEAGE = (
    "8632bbfa4c2a57c4fe531bddc7f09e05d61d152477ceca0d039571b7aa1b07f8",
    "909f2724a5723c53e57d965a82b4350fc3bc9a690c671310a964f7a3ecd561a3",
    "7d178620426747d66722c81e4e9e2de047d652a30ad965212ce8990ab07e9cc1",
)
TRAINING = {
    "lr0p003-e120": (0.003, 120),
    "lr0p003-e180": (0.003, 180),
    "lr0p01-e120": (0.01, 120),
    "lr0p01-e180": (0.01, 180),
}


class LocalAffordanceModelRecordError(ValueError):
    """Raised when local-affordance model metadata is inconsistent."""


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _body(model: BaseModel, field: str) -> dict[str, Any]:
    value = model.model_dump(mode="json")
    value.pop(field)
    return value


def _require_hash(value: str, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(HEX64, value) is None:
        raise LocalAffordanceModelRecordError(f"{label} must be a lowercase SHA-256 digest")


def _require_plan_identity(plan: LocalAffordancePlan) -> None:
    """Check the complete in-memory plan seal and frozen source/raw lineage."""
    if (
        plan.schema_version != PLAN_SCHEMA_VERSION
        or plan.final_family_access
        or plan.family_order != FAMILIES
        or len(plan.views) != EXPECTED_COUNTS["views"]
        or len(plan.model_owners) != EXPECTED_COUNTS["model_owners"]
        or len(plan.units) != EXPECTED_COUNTS["units"]
        or len({owner.owner_id for owner in plan.model_owners}) != len(plan.model_owners)
    ):
        raise LocalAffordanceModelRecordError("plan is incomplete or outside development scope")
    if tuple(plan.source_sha256) != tuple(FROZEN_SOURCE_SHA256.items()):
        raise LocalAffordanceModelRecordError("plan source hashes differ from frozen authority")
    for value, label in (
        (plan.raw_authority_manifest_id, "raw manifest identity"),
        (plan.raw_authority_content_sha256, "raw authority content hash"),
        (plan.raw_manifest_file_sha256, "raw manifest file hash"),
    ):
        _require_hash(value, label)
    body = {
        "schema_version": plan.schema_version,
        "source_sha256": dict(plan.source_sha256),
        "raw_authority_manifest_id": plan.raw_authority_manifest_id,
        "raw_authority_content_sha256": plan.raw_authority_content_sha256,
        "raw_manifest_file_sha256": plan.raw_manifest_file_sha256,
        "family_order": plan.family_order,
        "replicates": plan.replicates,
        "condition_ids": plan.condition_ids,
        "candidate_tuple_ids": plan.candidate_tuple_ids,
        "views": [_as_json(view) for view in plan.views],
        "model_owners": [_as_json(owner) for owner in plan.model_owners],
        "units": [_as_json(unit) for unit in plan.units],
        "final_family_access": False,
    }
    if plan.plan_id != _digest(body):
        raise LocalAffordanceModelRecordError("plan self-hash mismatch")
    if plan.plan_id != FROZEN_PLAN_ID:
        raise LocalAffordanceModelRecordError("plan ID differs from frozen lock")
    if (
        plan.raw_authority_manifest_id,
        plan.raw_authority_content_sha256,
        plan.raw_manifest_file_sha256,
    ) != FROZEN_RAW_LINEAGE:
        raise LocalAffordanceModelRecordError("plan raw authority differs from frozen capture")


class LocalAffordanceModelKey(BaseModel):
    """Temperature-independent, frozen scientific identity for one trained owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["milestone6.phase3.local-affordance-model-key.v1"] = KEY_SCHEMA_VERSION
    plan_id: str = Field(pattern=HEX64)
    owner_id: str = Field(pattern=HEX64)
    view_id: str = Field(pattern=HEX64)
    condition_id: str = Field(min_length=1)
    fold_id: str = Field(min_length=1)
    heldout_family: str = Field(min_length=1)
    replicate: StrictInt = Field(ge=0, le=4)
    training_tuple_id: str = Field(min_length=1)
    model_seed: StrictInt
    learning_rate: StrictFloat = Field(gt=0)
    optimizer_epochs: StrictInt = Field(gt=0)
    architecture_sha256: str = Field(pattern=HEX64)
    trainable_parameters: StrictInt = Field(gt=0)
    training_examples_sha256: str = Field(pattern=HEX64)
    model_state_sha256: str = Field(pattern=HEX64)
    model_identity_sha256: str = Field(pattern=HEX64)
    temperature_consumer_ids: tuple[str, str, str]
    key_sha256: str = Field(pattern=HEX64)

    @model_validator(mode="after")
    def validate_frozen_identity(self) -> "LocalAffordanceModelKey":
        if self.condition_id not in CONDITIONS or self.heldout_family not in FAMILIES:
            raise ValueError("model key condition/family is outside development authority")
        if self.fold_id != self.heldout_family:
            raise ValueError("model key fold must equal its held-out development family")
        if self.training_tuple_id not in TRAINING_TUPLE_IDS:
            raise ValueError("model key training tuple is not frozen")
        if self.temperature_consumer_ids != tuple(
            f"{self.training_tuple_id}-{temperature}" for temperature in TEMPERATURES
        ):
            raise ValueError("owner must serve exactly the three frozen temperature consumers")
        rate, epochs = TRAINING[self.training_tuple_id]
        capacity = 3601 if self.condition_id.startswith("B2-") else 3841
        if self.learning_rate != rate or self.optimizer_epochs != epochs:
            raise ValueError("model key optimizer tuple differs from frozen authority")
        if self.trainable_parameters != capacity:
            raise ValueError("model capacity differs from frozen condition capacity")
        if any(
            set(value) == {"0"}
            for value in (
                self.training_examples_sha256,
                self.model_state_sha256,
                self.model_identity_sha256,
            )
        ):
            raise ValueError("training/model identity hashes must be nonzero")
        if self.key_sha256 != _digest(_body(self, "key_sha256")):
            raise ValueError("model key self-hash mismatch")
        return self


class LocalAffordanceModelCost(BaseModel):
    """Exact training cost plus explicit zero-cost local-preparation boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["milestone6.phase3.local-affordance-model-cost.v1"] = COST_SCHEMA_VERSION
    optimizer_steps: StrictInt = Field(gt=0)
    forward_passes: StrictInt = Field(gt=0)
    training_examples: StrictInt = Field(gt=0)
    recurrent_steps: StrictInt = Field(ge=0)
    probe_actions: StrictInt = Field(ge=0)
    search_actions: StrictInt = Field(ge=0)
    evaluator_calls: StrictInt = Field(ge=0)
    oracle_calls: StrictInt = Field(ge=0)
    cost_sha256: str = Field(pattern=HEX64)

    @model_validator(mode="after")
    def validate_exact_cost(self) -> "LocalAffordanceModelCost":
        if self.forward_passes != self.optimizer_steps * self.training_examples:
            raise ValueError("forward-pass count must equal epochs times examples")
        if self.recurrent_steps != 0:
            raise ValueError("local-affordance model preparation has no recurrent steps")
        if (self.probe_actions, self.search_actions, self.evaluator_calls, self.oracle_calls) != (0, 0, 0, 0):
            raise ValueError("local model preparation cannot consume probe/search/evaluator/oracle budget")
        if self.cost_sha256 != _digest(_body(self, "cost_sha256")):
            raise ValueError("model cost self-hash mismatch")
        return self


class LocalAffordanceModelProvenance(BaseModel):
    """Self-hashed pointer to the exact plan, raw capture, sources, and preparation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["milestone6.phase3.local-affordance-model-provenance.v1"] = PROVENANCE_SCHEMA_VERSION
    plan_id: str = Field(pattern=HEX64)
    plan_source_sha256: tuple[tuple[str, str], ...]
    raw_authority_manifest_id: str = Field(pattern=HEX64)
    raw_authority_content_sha256: str = Field(pattern=HEX64)
    raw_manifest_file_sha256: str = Field(pattern=HEX64)
    preparation_git_commit_sha: str = Field(pattern=COMMIT)
    preparation_provenance_sha256: str = Field(pattern=HEX64)
    provenance_sha256: str = Field(pattern=HEX64)

    @model_validator(mode="after")
    def validate_provenance(self) -> "LocalAffordanceModelProvenance":
        if set(self.preparation_git_commit_sha) == {"0"} or set(self.preparation_provenance_sha256) == {"0"}:
            raise ValueError("nonzero preparation commit and provenance are required")
        if not self.plan_source_sha256 or len(dict(self.plan_source_sha256)) != len(self.plan_source_sha256):
            raise ValueError("plan source hash lineage is incomplete or duplicated")
        if any(re.fullmatch(HEX64, digest) is None for _name, digest in self.plan_source_sha256):
            raise ValueError("plan source hash lineage contains malformed digests")
        if self.provenance_sha256 != _digest(_body(self, "provenance_sha256")):
            raise ValueError("preparation provenance self-hash mismatch")
        return self


class LocalAffordanceModelArtifactMetadata(BaseModel):
    """Metadata for one serialized model payload; contains no payload bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_name: str = Field(pattern=r"^[0-9a-f]{64}\.model$")
    artifact_bytes: StrictInt = Field(gt=0)
    artifact_sha256: str = Field(pattern=HEX64)


class LocalAffordanceModelRecord(BaseModel):
    """Complete immutable record for one owner (not one temperature/unit)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["milestone6.phase3.local-affordance-model-record.v1"] = SCHEMA_VERSION
    key: LocalAffordanceModelKey
    cost: LocalAffordanceModelCost
    artifact: LocalAffordanceModelArtifactMetadata
    provenance: LocalAffordanceModelProvenance
    record_sha256: str = Field(pattern=HEX64)

    @model_validator(mode="after")
    def validate_record(self) -> "LocalAffordanceModelRecord":
        if self.cost.optimizer_steps != self.key.optimizer_epochs:
            raise ValueError("cost optimizer steps must equal frozen training epochs")
        if self.artifact.artifact_name != f"{self.key.owner_id}.model":
            raise ValueError("artifact name must be derived from owner identity")
        if self.provenance.plan_id != self.key.plan_id:
            raise ValueError("record key and provenance plan lineage differ")
        if self.record_sha256 != _digest(_body(self, "record_sha256")):
            raise ValueError("model record self-hash mismatch")
        return self


def build_model_record(
    *,
    plan: LocalAffordancePlan,
    owner_id: str,
    training_examples: int,
    training_examples_sha256: str,
    model_state_sha256: str,
    model_identity_sha256: str,
    artifact_bytes: int,
    artifact_sha256: str,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
) -> LocalAffordanceModelRecord:
    """Build one record after confirming the owner is in the supplied frozen plan."""
    if type(plan) is not LocalAffordancePlan or plan.final_family_access:
        raise LocalAffordanceModelRecordError("development-only local-affordance plan is required")
    _require_plan_identity(plan)
    matches = tuple(owner for owner in plan.model_owners if owner.owner_id == owner_id)
    if len(matches) != 1:
        raise LocalAffordanceModelRecordError("owner ID is missing or duplicated in the canonical plan")
    owner = matches[0]
    source_hashes = tuple(plan.source_sha256)
    if len(plan.model_owners) != EXPECTED_COUNTS["model_owners"]:
        raise LocalAffordanceModelRecordError("plan does not contain the complete 480-owner matrix")
    _require_hash(artifact_sha256, "artifact hash")
    _require_hash(training_examples_sha256, "training examples hash")
    _require_hash(model_state_sha256, "model state hash")
    _require_hash(model_identity_sha256, "model identity hash")
    rate, epochs = TRAINING[owner.training_tuple_id]
    key_body = {
        "schema_version": KEY_SCHEMA_VERSION,
        "plan_id": plan.plan_id,
        "owner_id": owner.owner_id,
        "view_id": owner.view_id,
        "condition_id": owner.condition_id,
        "fold_id": owner.fold_id,
        "heldout_family": owner.heldout_family,
        "replicate": owner.replicate,
        "training_tuple_id": owner.training_tuple_id,
        "model_seed": owner.model_seed,
        "learning_rate": rate,
        "optimizer_epochs": epochs,
        "architecture_sha256": owner.architecture_sha256,
        "trainable_parameters": owner.trainable_parameters,
        "training_examples_sha256": training_examples_sha256,
        "model_state_sha256": model_state_sha256,
        "model_identity_sha256": model_identity_sha256,
        "temperature_consumer_ids": owner.search_temperature_ids,
    }
    key = LocalAffordanceModelKey(**key_body, key_sha256=_digest(key_body))
    cost_body = {
        "schema_version": COST_SCHEMA_VERSION,
        "optimizer_steps": epochs,
        "forward_passes": epochs * training_examples,
        "training_examples": training_examples,
        "recurrent_steps": 0,
        "probe_actions": 0,
        "search_actions": 0,
        "evaluator_calls": 0,
        "oracle_calls": 0,
    }
    cost = LocalAffordanceModelCost(**cost_body, cost_sha256=_digest(cost_body))
    provenance_body = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "plan_id": plan.plan_id,
        "plan_source_sha256": source_hashes,
        "raw_authority_manifest_id": plan.raw_authority_manifest_id,
        "raw_authority_content_sha256": plan.raw_authority_content_sha256,
        "raw_manifest_file_sha256": plan.raw_manifest_file_sha256,
        "preparation_git_commit_sha": preparation_git_commit_sha,
        "preparation_provenance_sha256": preparation_provenance_sha256,
    }
    provenance = LocalAffordanceModelProvenance(
        **provenance_body, provenance_sha256=_digest(provenance_body)
    )
    artifact = LocalAffordanceModelArtifactMetadata(
        artifact_name=f"{owner.owner_id}.model",
        artifact_bytes=artifact_bytes,
        artifact_sha256=artifact_sha256,
    )
    record_body = {
        "schema_version": SCHEMA_VERSION,
        "key": key.model_dump(mode="json"),
        "cost": cost.model_dump(mode="json"),
        "artifact": artifact.model_dump(mode="json"),
        "provenance": provenance.model_dump(mode="json"),
    }
    return LocalAffordanceModelRecord(**record_body, record_sha256=_digest(record_body))


def validate_model_record(record: LocalAffordanceModelRecord, *, plan: LocalAffordancePlan) -> None:
    """Rebind a record to the exact owner, plan, raw authority, and cost tuple."""
    if type(record) is not LocalAffordanceModelRecord or type(plan) is not LocalAffordancePlan:
        raise LocalAffordanceModelRecordError("typed model record and plan are required")
    _require_plan_identity(plan)
    # Reparse every nested model so object.__setattr__ tampering cannot bypass frozen models.
    try:
        checked = LocalAffordanceModelRecord.model_validate(record.model_dump(mode="json"))
    except (TypeError, ValueError) as exc:
        raise LocalAffordanceModelRecordError("model record failed schema/hash validation") from exc
    if checked != record or record.key.plan_id != plan.plan_id:
        raise LocalAffordanceModelRecordError("record does not match its canonical plan")
    owners = tuple(owner for owner in plan.model_owners if owner.owner_id == record.key.owner_id)
    if len(owners) != 1:
        raise LocalAffordanceModelRecordError("record owner is not unique in canonical plan")
    owner = owners[0]
    expected = {
        "view_id": owner.view_id,
        "condition_id": owner.condition_id,
        "fold_id": owner.fold_id,
        "heldout_family": owner.heldout_family,
        "replicate": owner.replicate,
        "training_tuple_id": owner.training_tuple_id,
        "model_seed": owner.model_seed,
        "learning_rate": owner.learning_rate,
        "optimizer_epochs": owner.training_epochs,
        "architecture_sha256": owner.architecture_sha256,
        "trainable_parameters": owner.trainable_parameters,
        "temperature_consumer_ids": owner.search_temperature_ids,
    }
    if any(getattr(record.key, field) != value for field, value in expected.items()):
        raise LocalAffordanceModelRecordError("record identity differs from its planned owner")
    provenance = record.provenance
    if (
        provenance.plan_source_sha256 != plan.source_sha256
        or provenance.raw_authority_manifest_id != plan.raw_authority_manifest_id
        or provenance.raw_authority_content_sha256 != plan.raw_authority_content_sha256
        or provenance.raw_manifest_file_sha256 != plan.raw_manifest_file_sha256
    ):
        raise LocalAffordanceModelRecordError("record plan/raw provenance lineage differs")


__all__ = [
    "LocalAffordanceModelArtifactMetadata",
    "LocalAffordanceModelCost",
    "LocalAffordanceModelKey",
    "LocalAffordanceModelProvenance",
    "LocalAffordanceModelRecord",
    "LocalAffordanceModelRecordError",
    "build_model_record",
    "validate_model_record",
]
