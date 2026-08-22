"""Path-free, externally anchored Phase 3 preparation identities.

This module is a schema and validation boundary only. It never reads a run
directory, trains a model, executes a task, or inspects a result.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from levelup.experiments.milestone6_phase3_anchor import (
    Phase3AnchorManifest,
    load_committed_phase3_anchor_manifest_bytes,
    require_phase3_anchor_manifest,
)
from levelup.experiments.milestone6_phase3_evidence import (
    Phase3EvidenceLock,
    load_committed_phase3_evidence_lock_bytes,
    require_phase3_evidence_lock,
)
from levelup.experiments.milestone6_phase3_plan import (
    ValidatedPhase3Plan,
    load_committed_phase3_plan_lock_bytes,
    validate_phase3_plan_lock_bytes,
)
from levelup.experiments.milestone6_phase3_protocol import FAMILIES, NEW_CONDITIONS
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import UnitKey, UnitSeeds
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataEvidenceKey,
    TrainingDataEvidenceManifest,
)

HEX = r"^[0-9a-f]{64}$"
HASH = Field(pattern=HEX)
SCHEMA_VERSION = "milestone6.phase3.artifacts.v2"
VIEW_SCHEMA = "milestone6.phase3.view-manifest.v2"
MODEL_SCHEMA = "milestone6.phase3.model-manifest.v2"
LINEAGE_SCHEMA = "milestone6.phase3.unit-lineage.v2"
TRAINING_TUPLES = ("lr0p003-e120", "lr0p003-e180", "lr0p01-e120", "lr0p01-e180")
TEMPERATURES = ("t0p6", "t0p9", "t1p2")
CAPACITY_BY_CONDITION = {
    "S-state-availability-listwise-optimum": 3_841,
    "H0-null-history-transition-listwise-optimum": 3_889,
    "H4-causal-history-transition-listwise-optimum": 3_889,
    "H4-shuffled-history-transition-listwise-optimum": 3_889,
}
ARCHITECTURE_BY_CONDITION = {
    "S-state-availability-listwise-optimum": "state-availability-mlp-v1",
    "H0-null-history-transition-listwise-optimum": "causal-history-gru-mlp-v1",
    "H4-causal-history-transition-listwise-optimum": "causal-history-gru-mlp-v1",
    "H4-shuffled-history-transition-listwise-optimum": "causal-history-gru-mlp-v1",
}
TUPLE_PARAMETERS = {
    "lr0p003-e120": (0.003, 120), "lr0p003-e180": (0.003, 180),
    "lr0p01-e120": (0.01, 120), "lr0p01-e180": (0.01, 180),
}


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unique(values: Sequence[str], label: str) -> None:
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be non-empty and unique")


def _base_condition(condition_id: str) -> str:
    for condition in NEW_CONDITIONS:
        if condition_id == condition or condition_id.startswith(condition + "--"):
            return condition
    raise ValueError("condition is outside the Phase 3 representation ladder")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceIdentity(_Frozen):
    """Identity copied from one real typed ``TrainingDataEvidenceManifest``."""

    manifest: TrainingDataEvidenceManifest
    evidence_key: TrainingDataEvidenceKey
    evidence_key_id: str = HASH
    evidence_id: str = HASH
    payload_sha256: str = HASH
    payload_bytes: StrictInt = Field(gt=0)
    sample_task_ids: tuple[str, ...]
    manifest_bytes_sha256: str = HASH

    @model_validator(mode="after")
    def typed_manifest_is_exact(self) -> "EvidenceIdentity":
        if self.evidence_key != self.manifest.key:
            raise ValueError("evidence key is not the typed manifest key")
        if self.evidence_key_id != self.manifest.evidence_key_id != self.evidence_key.key_id:
            raise ValueError("evidence key identity mismatch")
        if self.evidence_id != self.manifest.evidence_id:
            raise ValueError("evidence identity mismatch")
        if self.payload_sha256 != self.manifest.payload_sha256 or self.payload_bytes != self.manifest.payload_bytes:
            raise ValueError("evidence payload identity mismatch")
        if self.sample_task_ids != self.manifest.sample_task_ids:
            raise ValueError("evidence sample order differs from typed manifest")
        if self.manifest_bytes_sha256 != _sha_bytes(canonical_json_bytes(self.manifest.model_dump(mode="json"))):
            raise ValueError("canonical full evidence manifest bytes mismatch")
        return self


class Phase3ViewManifest(_Frozen):
    schema_version: Literal[VIEW_SCHEMA] = VIEW_SCHEMA
    manifest_sha256: str = HASH
    view_id: str = HASH
    condition_id: str = Field(min_length=1)
    fold_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    replicate: StrictInt = Field(ge=0, le=4)
    training_task_ids: tuple[str, ...]
    data_order_seed: StrictInt
    evidence: EvidenceIdentity
    representation_identity_sha256: str = HASH
    permutation_identity_sha256: str | None = Field(default=None, pattern=HEX)
    plan_sha256: str = HASH
    protocol_sha256: str = HASH

    @property
    def expected_manifest_sha256(self) -> str:
        return _digest(self.model_dump(mode="json", exclude={"manifest_sha256"}))

    @model_validator(mode="after")
    def self_identity_is_exact(self) -> "Phase3ViewManifest":
        if self.condition_id not in NEW_CONDITIONS or self.family_id not in FAMILIES:
            raise ValueError("view is outside the frozen development universe")
        _unique(self.training_task_ids, "training task IDs")
        key = self.evidence.evidence_key
        if (self.fold_id, self.family_id, self.replicate, self.data_order_seed) != (key.fold_id, key.heldout_family_id, key.replicate, key.data_order_seed):
            raise ValueError("view/evidence fold, family, replicate, or seed drifted")
        if self.training_task_ids != key.ordered_training_task_ids:
            raise ValueError("view task order differs from evidence key")
        shuffled = self.condition_id == "H4-shuffled-history-transition-listwise-optimum"
        if shuffled != (self.permutation_identity_sha256 is not None):
            raise ValueError("history permutation identity is inconsistent with condition")
        if self.manifest_sha256 != self.expected_manifest_sha256:
            raise ValueError("view manifest self-hash mismatch")
        return self


class TrainingSpecRecord(_Frozen):
    training_tuple_id: str = Field(min_length=1)
    epochs: StrictInt = Field(gt=0)
    learning_rate: float = Field(gt=0, allow_inf_nan=False)
    weight_decay: float = Field(ge=0, allow_inf_nan=False)
    spec_sha256: str = HASH

    @model_validator(mode="after")
    def exact_tuple(self) -> "TrainingSpecRecord":
        if self.training_tuple_id not in TUPLE_PARAMETERS:
            raise ValueError("unknown training tuple")
        lr, epochs = TUPLE_PARAMETERS[self.training_tuple_id]
        if self.learning_rate != lr or self.epochs != epochs or self.weight_decay != 0.0001:
            raise ValueError("training tuple parameters drifted")
        if self.spec_sha256 != _digest(self.model_dump(mode="json", exclude={"spec_sha256"})):
            raise ValueError("training specification identity mismatch")
        return self


class TrainingReportRecord(_Frozen):
    trainable_parameters: StrictInt = Field(ge=0)
    recurrent_steps: StrictInt = Field(ge=0)
    optimizer_steps: StrictInt = Field(ge=0)
    forward_passes: StrictInt = Field(ge=0)
    training_examples: StrictInt = Field(gt=0)
    report_sha256: str = HASH

    @model_validator(mode="after")
    def report_is_canonical(self) -> "TrainingReportRecord":
        if self.report_sha256 != _digest(self.model_dump(mode="json", exclude={"report_sha256"})):
            raise ValueError("training report identity mismatch")
        return self


class TemperatureConsumer(_Frozen):
    temperature_id: str = Field(min_length=1)
    consumer_unit_ids: tuple[str, ...]

    @model_validator(mode="after")
    def exact_task_consumers(self) -> "TemperatureConsumer":
        if len(self.consumer_unit_ids) != 8:
            raise ValueError("each temperature must have exactly eight task consumers")
        _unique(self.consumer_unit_ids, "temperature consumer unit IDs")
        return self


class Phase3ModelManifest(_Frozen):
    schema_version: Literal[MODEL_SCHEMA] = MODEL_SCHEMA
    manifest_sha256: str = HASH
    owner_id: str = HASH
    view_id: str = HASH
    condition_id: str = Field(min_length=1)
    fold_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    replicate: StrictInt = Field(ge=0, le=4)
    training_tuple_id: str = Field(min_length=1)
    training_spec: TrainingSpecRecord
    training_report: TrainingReportRecord
    architecture_id: str = Field(min_length=1)
    trainable_parameters: StrictInt = Field(ge=0)
    model_state_sha256: str = HASH
    ordered_tensor_sha256: tuple[str, ...]
    artifact_id: str = HASH
    artifact_manifest_sha256: str = HASH
    temperature_consumers: tuple[TemperatureConsumer, ...]

    @property
    def tensor_sha256(self) -> tuple[str, ...]:
        return self.ordered_tensor_sha256

    @property
    def expected_manifest_sha256(self) -> str:
        return _digest(self.model_dump(mode="json", exclude={"manifest_sha256"}))

    @model_validator(mode="after")
    def model_is_exact(self) -> "Phase3ModelManifest":
        if self.condition_id not in NEW_CONDITIONS or self.family_id not in FAMILIES:
            raise ValueError("model is outside the frozen development universe")
        if self.training_tuple_id != self.training_spec.training_tuple_id:
            raise ValueError("model training tuple differs from spec")
        capacity = CAPACITY_BY_CONDITION[self.condition_id]
        if self.trainable_parameters != capacity or self.training_report.trainable_parameters != capacity:
            raise ValueError("model capacity differs from condition")
        if self.architecture_id != ARCHITECTURE_BY_CONDITION[self.condition_id]:
            raise ValueError("model architecture differs from condition")
        if self.training_report.optimizer_steps != self.training_spec.epochs:
            raise ValueError("optimizer steps must equal epochs")
        if self.training_report.forward_passes != self.training_spec.epochs * self.training_report.training_examples:
            raise ValueError("forward-pass accounting differs from epochs times examples")
        _unique(self.ordered_tensor_sha256, "ordered tensor hashes")
        if len(self.temperature_consumers) != 3:
            raise ValueError("each model must have exactly three consumers")
        expected = tuple(f"{self.training_tuple_id}-{t}" for t in TEMPERATURES)
        if tuple(c.temperature_id for c in self.temperature_consumers) != expected:
            raise ValueError("full training-temperature tuple IDs are not exact")
        all_consumers = tuple(
            unit_id
            for consumer in self.temperature_consumers
            for unit_id in consumer.consumer_unit_ids
        )
        if len(all_consumers) != 24 or len(set(all_consumers)) != 24:
            raise ValueError("model must have exactly 24 unique unit consumers")
        if self.manifest_sha256 != self.expected_manifest_sha256:
            raise ValueError("model manifest self-hash mismatch")
        return self


class AnchorUnitIdentity(_Frozen):
    anchor_unit_id: str = HASH
    anchor_run_id: str = Field(min_length=1)
    anchor_result_bytes_sha256: str = HASH
    anchor_family_id: str = Field(min_length=1)
    anchor_task_id: str = Field(min_length=1)
    anchor_task_index: StrictInt = Field(ge=0)
    anchor_replicate: StrictInt = Field(ge=0, le=4)
    anchor_candidate_tuple_id: str = Field(min_length=1)
    anchor_condition_id: Literal["B2-global-listwise-optimum"] = "B2-global-listwise-optimum"


class Phase3UnitLineage(_Frozen):
    schema_version: Literal[LINEAGE_SCHEMA] = LINEAGE_SCHEMA
    lineage_sha256: str = HASH
    unit_id: str = HASH
    key: UnitKey
    seeds: UnitSeeds
    exposure_manifest_sha256: str = HASH
    plan_sha256: str = HASH
    view_id: str = HASH
    owner_id: str = HASH
    tuple_id: str = Field(min_length=1)
    training_tuple_id: str = Field(min_length=1)
    anchor: AnchorUnitIdentity

    @property
    def expected_lineage_sha256(self) -> str:
        return _digest(self.model_dump(mode="json", exclude={"lineage_sha256"}))

    @model_validator(mode="after")
    def lineage_is_canonical(self) -> "Phase3UnitLineage":
        if self.key.phase != "validation":
            raise ValueError("Phase 3 lineage must be validation-only")
        base = _base_condition(self.key.condition_id)
        if self.key.family_id not in FAMILIES or self.training_tuple_id not in TRAINING_TUPLES:
            raise ValueError("unit is outside the frozen development universe")
        if self.key.condition_id != f"{base}--{self.tuple_id}":
            raise ValueError("unit tuple/condition identity drifted")
        if self.anchor.anchor_candidate_tuple_id != self.tuple_id:
            raise ValueError("unit anchor candidate tuple differs")
        if self.lineage_sha256 != self.expected_lineage_sha256:
            raise ValueError("unit lineage self-hash mismatch")
        return self


class Phase3ReadinessEnvelope(_Frozen):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    envelope_sha256: str = HASH
    development_only: StrictBool = True
    final: StrictBool = False
    execution_authorized: Literal[False] = False
    family_order: tuple[str, ...]
    condition_ids: tuple[str, ...]
    training_tuple_ids: tuple[str, ...]
    evidence: tuple[EvidenceIdentity, ...]
    views: tuple[Phase3ViewManifest, ...]
    models: tuple[Phase3ModelManifest, ...]
    units: tuple[Phase3UnitLineage, ...]
    protocol_sha256: str = HASH
    plan_id: str = HASH
    anchor_manifest_sha256: str = HASH
    anchor_file_sha256: str = HASH
    evidence_lock_sha256: str = HASH
    evidence_lock_file_sha256: str = HASH

    @property
    def expected_envelope_sha256(self) -> str:
        return _digest(self.model_dump(mode="json", exclude={"envelope_sha256"}))

    @model_validator(mode="after")
    def envelope_self_scope(self) -> "Phase3ReadinessEnvelope":
        if not self.development_only or self.final or self.execution_authorized:
            raise ValueError(
                "Phase 3 artifact inventory must be development-only, final=false, "
                "and non-executable"
            )
        if self.family_order != FAMILIES or self.condition_ids != NEW_CONDITIONS:
            raise ValueError("readiness family/condition universe drifted")
        if self.training_tuple_ids != TRAINING_TUPLES:
            raise ValueError("readiness training tuple universe drifted")
        if (len(self.evidence), len(self.views), len(self.models), len(self.units)) != (30, 120, 480, 11_520):
            raise ValueError("readiness inventory counts are not exact")
        _unique(tuple(x.evidence_id for x in self.evidence), "evidence IDs")
        _unique(tuple(x.view_id for x in self.views), "view IDs")
        _unique(tuple(x.owner_id for x in self.models), "owner IDs")
        _unique(tuple(x.unit_id for x in self.units), "unit IDs")
        expected_views = {(c, f, r) for c in NEW_CONDITIONS for f in FAMILIES for r in range(5)}
        if {(x.condition_id, x.family_id, x.replicate) for x in self.views} != expected_views:
            raise ValueError("readiness view coverage drifted")
        return self


def _anchor_rows(anchor: Phase3AnchorManifest) -> dict[tuple[str, str, int, str], Mapping[str, Any]]:
    rows = anchor.body.get("unit_results")
    if not isinstance(rows, list):
        raise ValueError("validated anchor manifest has no unit rows")
    return {
        (row["family_id"], row["task_id"], row["replicate"], row["candidate_tuple_id"]): row
        for row in rows if row.get("base_condition_id") == "B2-global-listwise-optimum"
    }


def _evidence_map(
    authority: Sequence[TrainingDataEvidenceManifest],
) -> dict[str, TrainingDataEvidenceManifest]:
    values = authority
    result: dict[str, TrainingDataEvidenceManifest] = {}
    for item in values:
        if not isinstance(item, TrainingDataEvidenceManifest):
            raise ValueError("evidence authority must contain typed manifests")
        if item.evidence_id in result and result[item.evidence_id] != item:
            raise ValueError("duplicate evidence identity")
        result[item.evidence_id] = item
    return result


def _validate_phase3_readiness(
    value: Phase3ReadinessEnvelope | Mapping[str, Any],
    *,
    plan: ValidatedPhase3Plan,
    anchor_manifest: Phase3AnchorManifest,
    evidence_lock: Phase3EvidenceLock,
    require_committed: bool,
) -> Phase3ReadinessEnvelope:
    """Validate a non-executable inventory against external identity authorities.

    This schema does not validate stored model tensors, reports, or artifact bytes
    and therefore must never be used as an execution gate.
    """
    if not isinstance(plan, ValidatedPhase3Plan):
        raise ValueError("a ValidatedPhase3Plan is required")
    try:
        require_phase3_anchor_manifest(anchor_manifest)
    except ValueError as exc:
        raise ValueError("a validated Phase3AnchorManifest is required") from exc
    try:
        require_phase3_evidence_lock(evidence_lock)
    except ValueError as exc:
        raise ValueError("a validated Phase3EvidenceLock is required") from exc
    if require_committed:
        try:
            committed_plan = validate_phase3_plan_lock_bytes(
                load_committed_phase3_plan_lock_bytes()
            )
            committed_anchor = load_committed_phase3_anchor_manifest_bytes()
            committed_evidence = load_committed_phase3_evidence_lock_bytes()
        except ValueError as exc:
            raise ValueError(
                "committed Phase 3 authorities cannot be read safely"
            ) from exc
        if committed_plan != plan.plan:
            raise ValueError("validated plan differs from committed Phase 3 authority")
        if committed_anchor != anchor_manifest.canonical_bytes:
            raise ValueError("validated anchor differs from committed Phase 3 authority")
        if committed_evidence != evidence_lock.canonical_bytes:
            raise ValueError("validated evidence differs from committed Phase 3 authority")
    envelope = value if isinstance(value, Phase3ReadinessEnvelope) else Phase3ReadinessEnvelope.model_validate(value)
    if envelope.plan_id != plan.plan.plan_id:
        raise ValueError("readiness plan identity differs from validated plan")
    if envelope.anchor_manifest_sha256 != anchor_manifest.anchor_manifest_sha256:
        raise ValueError("readiness anchor identity differs from validated anchor")
    lineage = evidence_lock.body.get("lineage")
    if not isinstance(lineage, Mapping) or (
        lineage.get("phase3_protocol_sha256"),
        lineage.get("phase3_plan_id"),
        lineage.get("phase3_anchor_manifest_sha256"),
        lineage.get("phase3_anchor_file_sha256"),
    ) != (
        plan.plan.protocol_sha256,
        plan.plan.plan_id,
        anchor_manifest.anchor_manifest_sha256,
        _sha_bytes(anchor_manifest.canonical_bytes),
    ):
        raise ValueError("evidence lock lineage differs from plan or anchor authority")
    if envelope.protocol_sha256 != plan.plan.protocol_sha256:
        raise ValueError("readiness protocol identity differs from validated plan")
    if envelope.anchor_file_sha256 != _sha_bytes(anchor_manifest.canonical_bytes):
        raise ValueError("readiness anchor file identity differs from validated anchor")
    if (
        envelope.evidence_lock_sha256 != evidence_lock.evidence_lock_sha256
        or envelope.evidence_lock_file_sha256 != _sha_bytes(evidence_lock.canonical_bytes)
    ):
        raise ValueError("readiness evidence-lock identity differs from authority")
    evidence_rows = evidence_lock.body.get("evidence_artifacts")
    if not isinstance(evidence_rows, list):
        raise ValueError("evidence lock has no typed evidence inventory")
    try:
        evidence_manifests = tuple(
            TrainingDataEvidenceManifest.model_validate(row["evidence_manifest"])
            for row in evidence_rows
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("evidence lock typed manifest inventory is invalid") from exc
    authority = _evidence_map(evidence_manifests)
    if len(authority) != 30:
        raise ValueError("evidence authority must contain exactly 30 manifests")
    expected_evidence_keys = {
        (view.fold_id, view.heldout_family, view.replicate)
        for view in plan.plan.views
    }
    observed_evidence_keys = {
        (manifest.key.fold_id, manifest.key.heldout_family_id, manifest.key.replicate)
        for manifest in authority.values()
    }
    if observed_evidence_keys != expected_evidence_keys or len(expected_evidence_keys) != 30:
        raise ValueError("evidence fold/replicate coverage is not exact")
    if {x.evidence_id for x in envelope.evidence} != set(authority):
        raise ValueError("envelope evidence inventory differs from authority")
    if any(item.manifest != authority[item.evidence_id] for item in envelope.evidence):
        raise ValueError("envelope evidence object differs from typed authority")
    plan_views = {x.view_id: x for x in plan.plan.views}
    plan_owners = {x.owner_id: x for x in plan.plan.model_owners}
    plan_units = {x.unit.unit_id: x for x in plan.plan.units}
    if len(plan_views) != 120 or len(plan_owners) != 480 or len(plan_units) != 11_520:
        raise ValueError("validated plan inventory is not exact")
    for view in envelope.views:
        expected = plan_views.get(view.view_id)
        if expected is None or (
            expected.condition_id,
            expected.fold_id,
            expected.heldout_family,
            expected.replicate,
            expected.training_task_ids,
            expected.data_order_seed,
            expected.representation_sha256,
            plan.plan.plan_id,
            plan.plan.protocol_sha256,
        ) != (
            view.condition_id,
            view.fold_id,
            view.family_id,
            view.replicate,
            view.training_task_ids,
            view.data_order_seed,
            view.representation_identity_sha256,
            view.plan_sha256,
            view.protocol_sha256,
        ):
            raise ValueError("view differs from validated Phase3Plan")
        if view.evidence.evidence_id not in authority or view.evidence.manifest != authority[view.evidence.evidence_id]:
            raise ValueError("view evidence is not the exact typed authority")
    evidence_consumers: dict[str, list[Phase3ViewManifest]] = {
        evidence_id: [] for evidence_id in authority
    }
    for view in envelope.views:
        evidence_consumers[view.evidence.evidence_id].append(view)
    for evidence_id, consumers in evidence_consumers.items():
        manifest = authority[evidence_id]
        if len(consumers) != len(NEW_CONDITIONS) or {
            item.condition_id for item in consumers
        } != set(NEW_CONDITIONS):
            raise ValueError("each evidence artifact must feed exactly four condition views")
        if any(
            (
                item.fold_id,
                item.family_id,
                item.replicate,
            )
            != (
                manifest.key.fold_id,
                manifest.key.heldout_family_id,
                manifest.key.replicate,
            )
            for item in consumers
        ):
            raise ValueError("evidence consumers differ from their exact fold identity")
    if {x.owner_id for x in envelope.models} != set(plan_owners):
        raise ValueError("model owners differ from validated Phase3Plan")
    for model in envelope.models:
        owner = plan_owners[model.owner_id]
        if (model.view_id, model.condition_id, model.fold_id, model.family_id, model.replicate, model.training_tuple_id) != (owner.view_id, owner.condition_id, owner.fold_id, owner.heldout_family, owner.replicate, owner.training_tuple_id):
            raise ValueError("model owner fields differ from validated Phase3Plan")
        expected_consumers = []
        for temperature_id in owner.search_temperature_ids:
            consumer_ids = tuple(
                item.unit.unit_id
                for item in plan.plan.units
                if item.model_owner_id == owner.owner_id
                and item.tuple_id == temperature_id
            )
            expected_consumers.append(
                (temperature_id, consumer_ids)
            )
        observed_consumers = [
            (item.temperature_id, item.consumer_unit_ids)
            for item in model.temperature_consumers
        ]
        if observed_consumers != expected_consumers:
            raise ValueError("model consumers differ from validated Phase3Plan")
    reports_by_training_identity: dict[
        tuple[str, int, str], dict[str, TrainingReportRecord]
    ] = {}
    for model in envelope.models:
        reports_by_training_identity.setdefault(
            (model.fold_id, model.replicate, model.training_tuple_id), {}
        )[model.condition_id] = model.training_report
    for reports in reports_by_training_identity.values():
        if set(reports) != set(NEW_CONDITIONS):
            raise ValueError("matched condition training-report matrix is incomplete")
        state_report = reports["S-state-availability-listwise-optimum"]
        if state_report.recurrent_steps != 0:
            raise ValueError("state-only training must report zero recurrent steps")
        matched = [
            reports[condition]
            for condition in NEW_CONDITIONS
            if condition != "S-state-availability-listwise-optimum"
        ]
        if len({report.recurrent_steps for report in matched}) != 1:
            raise ValueError("history-control recurrent-step accounting is not matched")
        if len(
            {
                (
                    report.training_examples,
                    report.optimizer_steps,
                    report.forward_passes,
                )
                for report in reports.values()
            }
        ) != 1:
            raise ValueError("same-data training accounting is not matched")
    anchors = _anchor_rows(anchor_manifest)
    if len(anchors) != 12 * 6 * 8 * 5:
        raise ValueError("validated anchor inventory is incomplete")
    if {x.unit_id for x in envelope.units} != set(plan_units):
        raise ValueError("unit identities differ from validated Phase3Plan")
    for lineage in envelope.units:
        planned = plan_units[lineage.unit_id]
        if lineage.key != planned.unit.key or lineage.seeds != planned.unit.seeds or lineage.exposure_manifest_sha256 != planned.unit.exposure_manifest_sha256:
            raise ValueError("unit key, seeds, or exposure differs from validated Phase3Plan")
        if (lineage.plan_sha256, lineage.view_id, lineage.owner_id, lineage.tuple_id, lineage.training_tuple_id) != (plan.plan.plan_id, planned.view_id, planned.model_owner_id, planned.tuple_id, planned.training_tuple_id):
            raise ValueError("unit plan lineage differs from validated Phase3Plan")
        row = anchors.get((lineage.key.family_id, lineage.key.task_id, lineage.key.replicate, lineage.tuple_id))
        if row is None:
            raise ValueError("unit lacks exact B2 anchor row")
        if (lineage.anchor.anchor_unit_id, lineage.anchor.anchor_run_id, lineage.anchor.anchor_result_bytes_sha256) != (row["unit_id"], row["run_id"], row["result_bytes_sha256"]):
            raise ValueError("unit anchor differs from validated Phase3AnchorManifest")
        if (lineage.anchor.anchor_family_id, lineage.anchor.anchor_task_id, lineage.anchor.anchor_task_index, lineage.anchor.anchor_replicate, lineage.anchor.anchor_candidate_tuple_id) != (row["family_id"], row["task_id"], row["task_index"], row["replicate"], row["candidate_tuple_id"]):
            raise ValueError("unit anchor key differs from validated Phase3AnchorManifest")
    if envelope.envelope_sha256 != envelope.expected_envelope_sha256:
        raise ValueError("readiness envelope self-hash mismatch")
    return envelope


def validate_phase3_readiness(
    value: Phase3ReadinessEnvelope | Mapping[str, Any],
    *,
    plan: ValidatedPhase3Plan,
    anchor_manifest: Phase3AnchorManifest,
    evidence_lock: Phase3EvidenceLock,
) -> Phase3ReadinessEnvelope:
    """Validate the schema against descriptor-read committed authorities."""

    return _validate_phase3_readiness(
        value,
        plan=plan,
        anchor_manifest=anchor_manifest,
        evidence_lock=evidence_lock,
        require_committed=True,
    )


def _validate_phase3_readiness_for_test(
    value: Phase3ReadinessEnvelope | Mapping[str, Any],
    *,
    plan: ValidatedPhase3Plan,
    anchor_manifest: Phase3AnchorManifest,
    evidence_lock: Phase3EvidenceLock,
) -> Phase3ReadinessEnvelope:
    """Private adapter for synthetic schema fixtures with no file authority."""

    return _validate_phase3_readiness(
        value,
        plan=plan,
        anchor_manifest=anchor_manifest,
        evidence_lock=evidence_lock,
        require_committed=False,
    )


__all__ = [
    "AnchorUnitIdentity", "ARCHITECTURE_BY_CONDITION", "CAPACITY_BY_CONDITION",
    "EvidenceIdentity", "Phase3ModelManifest", "Phase3ReadinessEnvelope",
    "Phase3UnitLineage", "Phase3ViewManifest", "TemperatureConsumer",
    "TrainingReportRecord", "TrainingSpecRecord", "validate_phase3_readiness",
]
