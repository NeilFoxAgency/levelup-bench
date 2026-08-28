"""Typed, identity-only plan for the Phase 3 outcome-channel diagnostic.

The diagnostic is deliberately additive.  It reuses the already committed Phase 3
plan and evidence identities and creates no result store, model, environment, or
final-family access.  The plan is an in-memory authority until a later execution
slice chooses to persist a lock.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from levelup.experiments.milestone6_phase3_anchor_selection_metrics import (
    validate_pinned_phase3_anchor_selection_metrics_bytes,
)
from levelup.experiments.milestone6_phase3_model_authority import (
    load_phase3_model_artifact_authority_bytes,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    BASE_CONDITION,
    CONDITIONS,
    FAMILIES,
    PHASE3_OUTCOME_DIAGNOSTIC_PROTOCOL_PATH,
    OutcomeDiagnosticProtocolSnapshot,
    load_outcome_group_diagnostic_protocol,
)
from levelup.experiments.milestone6_phase3_plan import (
    REPLICATES,
    Phase3Plan,
    validate_phase3_plan_lock_bytes,
)
from levelup.experiments.runner.config import canonical_json_bytes

PARENT_COMMIT_SHA = "5c8cfb8c6bf4edb17e96ebb6c9c6b9128715f6aa"
SCHEMA_VERSION = "milestone6.phase3.outcome-group-diagnostic-plan.v1"
TRAINING_TUPLE_IDS = (
    "lr0p003-e120",
    "lr0p003-e180",
    "lr0p01-e120",
    "lr0p01-e180",
)
EXPECTED_PARAMETER_COUNT = 3_841
EXPECTED_VIEWS = 60
EXPECTED_MODEL_OWNERS = 240
EXPECTED_UNITS = 5_760
EXPECTED_TASKS_PER_FAMILY = 8
EXPECTED_TUPLES = (
    "lr0p003-e120-t0p6",
    "lr0p003-e120-t0p9",
    "lr0p003-e120-t1p2",
    "lr0p003-e180-t0p6",
    "lr0p003-e180-t0p9",
    "lr0p003-e180-t1p2",
    "lr0p01-e120-t0p6",
    "lr0p01-e120-t0p9",
    "lr0p01-e120-t1p2",
    "lr0p01-e180-t0p6",
    "lr0p01-e180-t0p9",
    "lr0p01-e180-t1p2",
)
_TOKEN = object()


class OutcomeDiagnosticPlanError(ValueError):
    """Raised when the frozen outcome diagnostic plan is incomplete or altered."""


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "__dataclass_fields__"):
        return {name: _jsonable(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _canonical_payload_bytes(value: Any) -> bytes:
    """Canonicalize a protocol payload without trusting its container types.

    Readiness snapshots recursively freeze payload mappings and lists into
    ``MappingProxyType`` and tuples.  Those wrappers are semantically the same
    JSON object/array as the freshly loaded payload, but direct equality is not
    reliable across the two representations.  Convert only JSON-compatible
    mapping/sequence containers here, preserving key types so malformed or
    injected payloads still fail closed.
    """

    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("protocol payload mappings require string keys")
            return {key: thaw(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [thaw(child) for child in item]
        if isinstance(item, (set, frozenset)):
            raise TypeError("protocol payload sets are not JSON-compatible")
        return item

    return canonical_json_bytes(thaw(value))


def _authority_body(
    snapshot: OutcomeDiagnosticProtocolSnapshot, name: str, *, canonical: bool = True
) -> dict[str, Any]:
    try:
        content = dict(snapshot.authority_bytes)[name]
        body = json.loads(content)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OutcomeDiagnosticPlanError(f"authority {name} is not valid JSON") from exc
    if not isinstance(body, dict) or (canonical and canonical_json_bytes(body) != content):
        raise OutcomeDiagnosticPlanError(f"authority {name} is not canonical")
    return body


def _require_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise OutcomeDiagnosticPlanError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _mask_specs(snapshot: OutcomeDiagnosticProtocolSnapshot) -> dict[str, dict[str, Any]]:
    rows = snapshot.payload.get("conditions")
    # Readiness pins the protocol payload recursively, replacing mutable JSON
    # lists with tuples and dicts with mapping proxies.  Accept both the
    # freshly-loaded JSON shape and that immutable pinned shape, while keeping
    # the exact condition/order checks below as the authority boundary.
    if not isinstance(rows, (list, tuple)):
        raise OutcomeDiagnosticPlanError("diagnostic condition matrix is missing")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise OutcomeDiagnosticPlanError("diagnostic condition row is malformed")
        condition = row.get("condition_id")
        representation = row.get("representation")
        if condition not in CONDITIONS or not isinstance(representation, Mapping):
            raise OutcomeDiagnosticPlanError("diagnostic mask identity is malformed")
        result[condition] = dict(representation)
    if tuple(result) != CONDITIONS:
        raise OutcomeDiagnosticPlanError("diagnostic condition order drifted")
    return result


def feature_mask_sha256(snapshot: OutcomeDiagnosticProtocolSnapshot, condition_id: str) -> str:
    """Return the deterministic digest of one frozen T-derived feature mask."""

    specs = _mask_specs(snapshot)
    if condition_id not in specs:
        raise OutcomeDiagnosticPlanError("unknown outcome diagnostic condition")
    representation = specs[condition_id]
    return _digest(
        {
            "source_condition_id": "T-markov-state-transition-listwise-optimum",
            "input_width": representation["input_width"],
            "retained_indices_per_summary_block": representation[
                "retained_indices_per_summary_block"
            ],
            "zeroed_indices_per_summary_block": representation["zeroed_indices_per_summary_block"],
            "operation": "retain listed T channels and zero the complement byte-for-byte",
        }
    )


def transformation_sha256(snapshot: OutcomeDiagnosticProtocolSnapshot, condition_id: str) -> str:
    """Return the deterministic digest of the frozen example transformation."""

    mask = feature_mask_sha256(snapshot, condition_id)
    return _digest(
        {
            "source_condition_id": "T-markov-state-transition-listwise-optimum",
            "input_width": 54,
            "operation": "mask_decision_examples",
            "feature_mask_sha256": mask,
            "same_source_examples": True,
            "bitwise_retained_values": True,
        }
    )


@dataclass(frozen=True, slots=True)
class OutcomeView:
    view_id: str
    condition_id: str
    fold_id: str
    heldout_family: str
    replicate: int
    training_task_ids: tuple[str, ...]
    data_order_seed: int
    evidence_lineage_sha256: str
    feature_mask_sha256: str
    transformation_sha256: str
    representation_sha256: str


@dataclass(frozen=True, slots=True)
class OutcomeModelOwner:
    owner_id: str
    condition_id: str
    fold_id: str
    heldout_family: str
    replicate: int
    training_tuple_id: str
    view_id: str
    model_seed: int
    learning_rate: float
    training_epochs: int
    search_temperature_ids: tuple[str, ...]
    trainable_parameters: int
    feature_mask_sha256: str
    transformation_sha256: str
    model_identity_sha256: str


@dataclass(frozen=True, slots=True)
class OutcomePlannedUnit:
    unit_id: str
    condition_id: str
    tuple_id: str
    training_tuple_id: str
    fold_id: str
    heldout_family: str
    task_id: str
    task_index: int
    replicate: int
    model_owner_id: str
    view_id: str
    model_seed: int
    environment_seed: int
    probe_seed: int
    search_seed: int
    data_order_seed: int
    exposure_manifest_sha256: str
    feature_mask_sha256: str
    transformation_sha256: str
    model_identity_sha256: str
    candidate_episodes_per_task: int
    adaptation_actions_per_task: int
    probe_actions_per_task: int
    maximum_actions_per_candidate_episode: int
    final_family_access: bool = False


@dataclass(frozen=True, slots=True)
class OutcomePlan:
    schema_version: str
    plan_id: str
    parent_commit_sha: str
    protocol_sha256: str
    authority_hashes: tuple[tuple[str, str], ...]
    family_order: tuple[str, ...]
    replicates: tuple[int, ...]
    condition_ids: tuple[str, ...]
    candidate_tuple_ids: tuple[str, ...]
    evidence_lineage_rows: tuple[bytes, ...]
    views: tuple[OutcomeView, ...]
    model_owners: tuple[OutcomeModelOwner, ...]
    units: tuple[OutcomePlannedUnit, ...]
    final_family_access: bool = False

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(item.unit_id for item in self.units)


@dataclass(frozen=True, slots=True, init=False)
class ValidatedOutcomePlan:
    plan: OutcomePlan
    _units_by_id: Mapping[str, OutcomePlannedUnit]
    _construction_token: object

    def __init__(
        self,
        plan: OutcomePlan,
        units_by_id: Mapping[str, OutcomePlannedUnit],
        *,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _TOKEN:
            raise OutcomeDiagnosticPlanError(
                "validated outcome plans require the canonical plan gate"
            )
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "_units_by_id", MappingProxyType(dict(units_by_id)))
        object.__setattr__(self, "_construction_token", _construction_token)

    def require_unit(self, unit: OutcomePlannedUnit) -> None:
        if self._construction_token is not _TOKEN or self._units_by_id.get(unit.unit_id) != unit:
            raise OutcomeDiagnosticPlanError("unit differs from the validated outcome plan")


def _plan_body(plan: OutcomePlan, *, include_id: bool = False) -> dict[str, Any]:
    body = {
        "schema_version": plan.schema_version,
        "parent_commit_sha": plan.parent_commit_sha,
        "protocol_sha256": plan.protocol_sha256,
        "authority_hashes": dict(plan.authority_hashes),
        "family_order": list(plan.family_order),
        "replicates": list(plan.replicates),
        "condition_ids": list(plan.condition_ids),
        "candidate_tuple_ids": list(plan.candidate_tuple_ids),
        "evidence_lineage_rows_sha256": [
            hashlib.sha256(row).hexdigest() for row in plan.evidence_lineage_rows
        ],
        "views": [_jsonable(item) for item in plan.views],
        "model_owners": [_jsonable(item) for item in plan.model_owners],
        "units": [_jsonable(item) for item in plan.units],
        "final_family_access": plan.final_family_access,
    }
    if include_id:
        body["plan_id"] = plan.plan_id
    return body


def outcome_plan_id(plan: OutcomePlan) -> str:
    return _digest(_plan_body(plan))


def canonical_outcome_plan_bytes(
    plan: OutcomePlan, *, snapshot: OutcomeDiagnosticProtocolSnapshot
) -> bytes:
    validate_outcome_diagnostic_plan(plan, snapshot=snapshot)
    if outcome_plan_id(plan) != plan.plan_id:
        raise OutcomeDiagnosticPlanError(
            "outcome plan self-hash identity differs from canonical body"
        )
    return canonical_json_bytes(_plan_body(plan, include_id=True))


def _load_authorities(
    snapshot: OutcomeDiagnosticProtocolSnapshot,
) -> tuple[Phase3Plan, dict[str, Any], Any, Any]:
    authority = dict(snapshot.authority_bytes)
    try:
        plan_bytes = authority["phase3_plan"]
        phase3_plan = validate_phase3_plan_lock_bytes(plan_bytes)
        model = load_phase3_model_artifact_authority_bytes(authority["phase3_model_authority"])
        anchor = validate_pinned_phase3_anchor_selection_metrics_bytes(
            authority["phase3_anchor_metrics"]
        )
        evidence = _authority_body(snapshot, "phase3_evidence")
        selection = _authority_body(snapshot, "phase3_development_selection", canonical=False)
    except (OSError, TypeError, ValueError) as exc:
        raise OutcomeDiagnosticPlanError(
            "committed Phase 3 authorities failed typed loading"
        ) from exc
    if model.plan_id != phase3_plan.plan_id:
        raise OutcomeDiagnosticPlanError("model authority is not bound to the Phase 3 plan")
    if (
        evidence.get("evidence_lock_sha256")
        != snapshot.payload["authority"]["phase3_evidence"]["evidence_lock_sha256"]
    ):
        raise OutcomeDiagnosticPlanError("evidence lock identity drifted")
    if (
        selection.get("selection_lock_sha256")
        != snapshot.payload["authority"]["phase3_development_selection"]["selection_lock_sha256"]
    ):
        raise OutcomeDiagnosticPlanError("selection lock identity drifted")
    if len(evidence.get("evidence_artifacts", [])) != 30:
        raise OutcomeDiagnosticPlanError("Phase 3 evidence lineage is not exactly 30 rows")
    if anchor.body.get("final_family_access") is not False:
        raise OutcomeDiagnosticPlanError("anchor authority permits final-family access")
    return phase3_plan, evidence, model, anchor


def _require_canonical_snapshot(
    snapshot: OutcomeDiagnosticProtocolSnapshot,
) -> OutcomeDiagnosticProtocolSnapshot:
    """Re-read and byte/field compare the frozen protocol; injected snapshots are not trust."""

    if not isinstance(snapshot, OutcomeDiagnosticProtocolSnapshot):
        raise OutcomeDiagnosticPlanError("outcome diagnostic protocol is not typed")
    try:
        fresh = load_outcome_group_diagnostic_protocol()
    except (OSError, TypeError, ValueError) as exc:
        raise OutcomeDiagnosticPlanError(
            "canonical outcome protocol cannot be revalidated"
        ) from exc
    try:
        payload_matches = _canonical_payload_bytes(snapshot.payload) == _canonical_payload_bytes(
            fresh.payload
        )
    except (TypeError, ValueError):
        payload_matches = False
    if (
        snapshot.repository != fresh.repository
        or snapshot.path != fresh.path
        or snapshot.content != fresh.content
        or snapshot.sha256 != fresh.sha256
        or not payload_matches
        or snapshot.authority_bytes != fresh.authority_bytes
    ):
        raise OutcomeDiagnosticPlanError(
            "outcome protocol snapshot differs from committed authority"
        )
    return fresh


def _construct_outcome_group_diagnostic_plan(
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    phase3_plan: Phase3Plan,
    evidence: dict[str, Any],
) -> OutcomePlan:
    if (
        PHASE3_OUTCOME_DIAGNOSTIC_PROTOCOL_PATH != snapshot.path
        and snapshot.path.name != PHASE3_OUTCOME_DIAGNOSTIC_PROTOCOL_PATH.name
    ):
        raise OutcomeDiagnosticPlanError("unexpected diagnostic protocol path")
    if phase3_plan.final_family_access or phase3_plan.family_order != FAMILIES:
        raise OutcomeDiagnosticPlanError("Phase 3 plan is not development-only")
    if phase3_plan.candidate_tuple_ids != EXPECTED_TUPLES:
        raise OutcomeDiagnosticPlanError("candidate tuple universe drifted")
    s_views = {
        (item.heldout_family, item.replicate): item
        for item in phase3_plan.views
        if item.condition_id == BASE_CONDITION
    }
    s_units = {
        (item.heldout_family, item.unit.key.replicate, item.tuple_id, item.unit.key.task_id): item
        for item in phase3_plan.units
        if item.base_condition_id == BASE_CONDITION
    }
    if len(s_views) != 30 or len(s_units) != 30 * 12 * 8:
        raise OutcomeDiagnosticPlanError("Phase 3 S seed matrix is incomplete")
    masks = _mask_specs(snapshot)
    authorities = dict(snapshot.payload["authority"])
    authority_hashes = tuple(
        (name, _require_digest(source["sha256"], f"authority {name}"))
        for name, source in authorities.items()
    )
    evidence_rows = tuple(canonical_json_bytes(row) for row in evidence["evidence_artifacts"])
    evidence_row_by_key = {
        (row["family_id"], int(row["replicate"])): canonical_json_bytes(row)
        for row in evidence["evidence_artifacts"]
    }
    if set(evidence_row_by_key) != {
        (family, replicate) for family in FAMILIES for replicate in REPLICATES
    }:
        raise OutcomeDiagnosticPlanError("Phase 3 evidence family/replicate coverage drifted")
    views: list[OutcomeView] = []
    owners: list[OutcomeModelOwner] = []
    units: list[OutcomePlannedUnit] = []
    owner_by_key: dict[tuple[str, str, int, str], OutcomeModelOwner] = {}
    tuple_rows = {row["tuple_id"]: row for row in snapshot.payload["candidate_tuples"]}
    for condition_id in CONDITIONS:
        mask = feature_mask_sha256(snapshot, condition_id)
        transformation = transformation_sha256(snapshot, condition_id)
        for family in FAMILIES:
            for replicate in REPLICATES:
                source_view = s_views[(family, replicate)]
                evidence_hash = hashlib.sha256(evidence_row_by_key[(family, replicate)]).hexdigest()
                representation = _digest(
                    {
                        "source": "T",
                        "condition_id": condition_id,
                        "mask": mask,
                        "transformation": transformation,
                        "input_width": masks[condition_id]["input_width"],
                    }
                )
                view_id = _digest(
                    {
                        "parent_commit_sha": PARENT_COMMIT_SHA,
                        "protocol_sha256": snapshot.sha256,
                        "condition_id": condition_id,
                        "fold_id": source_view.fold_id,
                        "family": family,
                        "replicate": replicate,
                        "data_order_seed": source_view.data_order_seed,
                        "evidence_lineage_sha256": evidence_hash,
                        "feature_mask_sha256": mask,
                        "transformation_sha256": transformation,
                    }
                )
                view = OutcomeView(
                    view_id,
                    condition_id,
                    source_view.fold_id,
                    family,
                    replicate,
                    source_view.training_task_ids,
                    source_view.data_order_seed,
                    evidence_hash,
                    mask,
                    transformation,
                    representation,
                )
                views.append(view)
                for training_tuple_id in TRAINING_TUPLE_IDS:
                    tuple_row = next(
                        row
                        for row in snapshot.payload["candidate_tuples"]
                        if row["training_tuple_id"] == training_tuple_id
                    )
                    seed_source = next(
                        item
                        for item in phase3_plan.units
                        if item.base_condition_id == BASE_CONDITION
                        and item.heldout_family == family
                        and item.unit.key.replicate == replicate
                        and item.training_tuple_id == training_tuple_id
                    )
                    model_identity = _digest(
                        {
                            "owner_view_id": view_id,
                            "condition_id": condition_id,
                            "training_tuple_id": training_tuple_id,
                            "model_seed": seed_source.unit.seeds.model_seed,
                            "feature_mask_sha256": mask,
                            "transformation_sha256": transformation,
                            "trainable_parameters": EXPECTED_PARAMETER_COUNT,
                        }
                    )
                    owner_id = _digest(
                        {
                            "parent_commit_sha": PARENT_COMMIT_SHA,
                            "view_id": view_id,
                            "training_tuple_id": training_tuple_id,
                            "model_identity_sha256": model_identity,
                        }
                    )
                    owner = OutcomeModelOwner(
                        owner_id,
                        condition_id,
                        source_view.fold_id,
                        family,
                        replicate,
                        training_tuple_id,
                        view_id,
                        seed_source.unit.seeds.model_seed,
                        float(tuple_row["learning_rate"]),
                        int(tuple_row["training_epochs"]),
                        tuple(
                            row["tuple_id"]
                            for row in snapshot.payload["candidate_tuples"]
                            if row["training_tuple_id"] == training_tuple_id
                        ),
                        EXPECTED_PARAMETER_COUNT,
                        mask,
                        transformation,
                        model_identity,
                    )
                    owners.append(owner)
                    owner_by_key[(condition_id, family, replicate, training_tuple_id)] = owner
                task_ids = tuple(
                    dict.fromkeys(
                        key[3] for key in s_units if key[0] == family and key[1] == replicate
                    )
                )
                for task_id in task_ids:
                    for tuple_id in EXPECTED_TUPLES:
                        source = s_units[(family, replicate, tuple_id, task_id)]
                        owner = owner_by_key[
                            (condition_id, family, replicate, source.training_tuple_id)
                        ]
                        row = tuple_rows[tuple_id]
                        seeds = source.unit.seeds
                        unit_id = _digest(
                            {
                                "parent_commit_sha": PARENT_COMMIT_SHA,
                                "condition_id": condition_id,
                                "tuple_id": tuple_id,
                                "family": family,
                                "task_id": task_id,
                                "task_index": source.unit.key.task_index,
                                "replicate": replicate,
                                "owner_id": owner.owner_id,
                                "view_id": view_id,
                                "seeds": seeds.model_dump(mode="json"),
                                "exposure_manifest_sha256": source.unit.exposure_manifest_sha256,
                                "feature_mask_sha256": mask,
                                "transformation_sha256": transformation,
                            }
                        )
                        units.append(
                            OutcomePlannedUnit(
                                unit_id,
                                condition_id,
                                tuple_id,
                                row["training_tuple_id"],
                                source.fold_id,
                                family,
                                task_id,
                                source.unit.key.task_index,
                                replicate,
                                owner.owner_id,
                                view_id,
                                seeds.model_seed,
                                seeds.environment_seed,
                                seeds.probe_seed,
                                seeds.search_seed,
                                seeds.data_order_seed,
                                source.unit.exposure_manifest_sha256,
                                mask,
                                transformation,
                                owner.model_identity_sha256,
                                150,
                                2048,
                                64,
                                64,
                                False,
                            )
                        )
    plan = OutcomePlan(
        SCHEMA_VERSION,
        "",
        PARENT_COMMIT_SHA,
        snapshot.sha256,
        authority_hashes,
        FAMILIES,
        REPLICATES,
        CONDITIONS,
        EXPECTED_TUPLES,
        evidence_rows,
        tuple(views),
        tuple(owners),
        tuple(units),
        False,
    )
    plan = OutcomePlan(
        plan.schema_version,
        outcome_plan_id(plan),
        plan.parent_commit_sha,
        plan.protocol_sha256,
        plan.authority_hashes,
        plan.family_order,
        plan.replicates,
        plan.condition_ids,
        plan.candidate_tuple_ids,
        plan.evidence_lineage_rows,
        plan.views,
        plan.model_owners,
        plan.units,
        False,
    )
    return plan


def build_outcome_group_diagnostic_plan(
    snapshot: OutcomeDiagnosticProtocolSnapshot | None = None,
) -> OutcomePlan:
    snapshot = load_outcome_group_diagnostic_protocol() if snapshot is None else snapshot
    snapshot = _require_canonical_snapshot(snapshot)
    return build_outcome_group_diagnostic_plan_from_pinned_snapshot(snapshot)


def build_outcome_group_diagnostic_plan_from_pinned_snapshot(
    snapshot: OutcomeDiagnosticProtocolSnapshot,
) -> OutcomePlan:
    """Build from immutable authority bytes held by a readiness capability."""

    if not isinstance(snapshot, OutcomeDiagnosticProtocolSnapshot):
        raise OutcomeDiagnosticPlanError("pinned outcome protocol snapshot is not typed")
    phase3_plan, evidence, _model_authority, _anchor = _load_authorities(snapshot)
    plan = _construct_outcome_group_diagnostic_plan(snapshot, phase3_plan, evidence)
    plan = OutcomePlan(
        plan.schema_version,
        outcome_plan_id(plan),
        plan.parent_commit_sha,
        plan.protocol_sha256,
        plan.authority_hashes,
        plan.family_order,
        plan.replicates,
        plan.condition_ids,
        plan.candidate_tuple_ids,
        plan.evidence_lineage_rows,
        plan.views,
        plan.model_owners,
        plan.units,
        False,
    )
    _validate_outcome_diagnostic_plan_from_pinned_snapshot(
        plan, snapshot=snapshot, phase3_plan=phase3_plan
    )
    return plan


def validate_outcome_diagnostic_plan(
    plan: OutcomePlan,
    *,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    phase3_plan: Phase3Plan | None = None,
) -> None:
    snapshot = _require_canonical_snapshot(snapshot)
    _validate_outcome_diagnostic_plan_from_pinned_snapshot(
        plan, snapshot=snapshot, phase3_plan=phase3_plan
    )


def _validate_outcome_diagnostic_plan_from_pinned_snapshot(
    plan: OutcomePlan,
    *,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    phase3_plan: Phase3Plan | None = None,
) -> None:
    if not isinstance(snapshot, OutcomeDiagnosticProtocolSnapshot):
        raise OutcomeDiagnosticPlanError("pinned outcome protocol snapshot is not typed")
    if (
        not isinstance(plan, OutcomePlan)
        or plan.schema_version != SCHEMA_VERSION
        or plan.final_family_access
    ):
        raise OutcomeDiagnosticPlanError("outcome plan is not development-only")
    if (
        plan.parent_commit_sha != PARENT_COMMIT_SHA
        or plan.family_order != FAMILIES
        or plan.replicates != REPLICATES
        or plan.condition_ids != CONDITIONS
        or plan.candidate_tuple_ids != EXPECTED_TUPLES
    ):
        raise OutcomeDiagnosticPlanError("outcome plan universe drifted")
    if len(plan.evidence_lineage_rows) != 30 or any(
        not isinstance(row, bytes) for row in plan.evidence_lineage_rows
    ):
        raise OutcomeDiagnosticPlanError("evidence lineage matrix is incomplete")
    if (
        len(plan.views) != EXPECTED_VIEWS
        or len({item.view_id for item in plan.views}) != EXPECTED_VIEWS
    ):
        raise OutcomeDiagnosticPlanError("view matrix is incomplete or duplicated")
    if (
        len(plan.model_owners) != EXPECTED_MODEL_OWNERS
        or len({item.owner_id for item in plan.model_owners}) != EXPECTED_MODEL_OWNERS
    ):
        raise OutcomeDiagnosticPlanError("model-owner matrix is incomplete or duplicated")
    if len(plan.units) != EXPECTED_UNITS or len(plan.unit_ids) != len(set(plan.unit_ids)):
        raise OutcomeDiagnosticPlanError("unit matrix is incomplete or duplicated")
    if any(item.trainable_parameters != EXPECTED_PARAMETER_COUNT for item in plan.model_owners):
        raise OutcomeDiagnosticPlanError("capacity matching drifted")
    view_ids = {item.view_id for item in plan.views}
    owner_ids = {item.owner_id for item in plan.model_owners}
    if any(
        item.view_id not in view_ids or item.model_owner_id not in owner_ids for item in plan.units
    ):
        raise OutcomeDiagnosticPlanError("unit references missing view or owner")
    if any(len(owner.search_temperature_ids) != 3 for owner in plan.model_owners):
        raise OutcomeDiagnosticPlanError("model-owner temperature fanout drifted")
    if any(
        sum(item.model_owner_id == owner.owner_id for item in plan.units) != 24
        for owner in plan.model_owners
    ):
        raise OutcomeDiagnosticPlanError("model-owner consumer fanout drifted")
    if outcome_plan_id(plan) != plan.plan_id:
        raise OutcomeDiagnosticPlanError("outcome plan self-hash mismatch")
    if snapshot is not None and plan.protocol_sha256 != snapshot.sha256:
        raise OutcomeDiagnosticPlanError("outcome plan protocol lineage drifted")
    canonical_phase3_plan, evidence, _model_authority, _anchor = _load_authorities(snapshot)
    if phase3_plan is not None and phase3_plan != canonical_phase3_plan:
        raise OutcomeDiagnosticPlanError("outcome plan Phase 3 plan authority drifted")
    expected = _construct_outcome_group_diagnostic_plan(snapshot, canonical_phase3_plan, evidence)
    expected = OutcomePlan(
        expected.schema_version,
        outcome_plan_id(expected),
        expected.parent_commit_sha,
        expected.protocol_sha256,
        expected.authority_hashes,
        expected.family_order,
        expected.replicates,
        expected.condition_ids,
        expected.candidate_tuple_ids,
        expected.evidence_lineage_rows,
        expected.views,
        expected.model_owners,
        expected.units,
        expected.final_family_access,
    )
    if plan != expected:
        raise OutcomeDiagnosticPlanError("outcome plan differs from canonical authority")


def bind_validated_outcome_diagnostic_plan(
    plan: OutcomePlan, *, snapshot: OutcomeDiagnosticProtocolSnapshot
) -> ValidatedOutcomePlan:
    validate_outcome_diagnostic_plan(plan, snapshot=snapshot)
    return ValidatedOutcomePlan(
        plan, {item.unit_id: item for item in plan.units}, _construction_token=_TOKEN
    )


def bind_pinned_outcome_diagnostic_plan(
    plan: OutcomePlan, *, snapshot: OutcomeDiagnosticProtocolSnapshot
) -> ValidatedOutcomePlan:
    """Bind a plan without reopening authority paths after readiness capture."""

    _validate_outcome_diagnostic_plan_from_pinned_snapshot(plan, snapshot=snapshot)
    return ValidatedOutcomePlan(
        plan, {item.unit_id: item for item in plan.units}, _construction_token=_TOKEN
    )


__all__ = [
    "OutcomeDiagnosticPlanError",
    "OutcomeView",
    "OutcomeModelOwner",
    "OutcomePlannedUnit",
    "OutcomePlan",
    "ValidatedOutcomePlan",
    "PARENT_COMMIT_SHA",
    "build_outcome_group_diagnostic_plan_from_pinned_snapshot",
    "build_outcome_group_diagnostic_plan",
    "validate_outcome_diagnostic_plan",
    "bind_validated_outcome_diagnostic_plan",
    "bind_pinned_outcome_diagnostic_plan",
    "canonical_outcome_plan_bytes",
    "outcome_plan_id",
    "feature_mask_sha256",
    "transformation_sha256",
]
