"""Logical, development-only Phase 3 representation-ladder plan.

This module freezes the *identity* of the next development tranche.  It deliberately
does not prepare data, train models, run search, call an evaluator/oracle, or read any
result artifacts.  In particular, the four new representation conditions are paired
to the already frozen Phase 2 B2 unit seeds and task identities.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

from levelup.experiments.milestone6_phase2_screening import (
    B2,
    screening_child_configs,
)
from levelup.experiments.milestone6_phase3_protocol import (
    FAMILIES,
    NEW_CONDITIONS,
    Phase3ProtocolSnapshot,
    load_phase3_protocol,
)
from levelup.experiments.runner.config import (
    ExperimentConfig,
    canonical_json_bytes,
)
from levelup.experiments.runner.records import PlannedUnit, UnitKey, unit_id_for
from levelup.experiments.runner.storage import (
    expected_units_sha256,
    plan_expected_units,
)

REPLICATES = (0, 1, 2, 3, 4)
TRAINING_TUPLE_IDS = (
    "lr0p003-e120",
    "lr0p003-e180",
    "lr0p01-e120",
    "lr0p01-e180",
)
PHASE = "validation"
SCHEMA_VERSION = "milestone6.phase3.logical-plan.v1"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _tuple_rows(snapshot: Phase3ProtocolSnapshot) -> tuple[dict[str, Any], ...]:
    rows = snapshot.payload.get("candidate_tuples")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("Phase 3 candidate tuple matrix is missing or incomplete")
    result = tuple(dict(row) for row in rows)
    ids = tuple(row.get("tuple_id") for row in result)
    if any(not isinstance(item, str) for item in ids) or len(set(ids)) != 12:
        raise ValueError("Phase 3 candidate tuple identities are not unique")
    expected_training = tuple(row.get("training_tuple_id") for row in result)
    if tuple(sorted(set(expected_training))) != tuple(sorted(TRAINING_TUPLE_IDS)):
        raise ValueError("Phase 3 training tuple universe drifted")
    if any(row.get("training_tuple_id") not in TRAINING_TUPLE_IDS for row in result):
        raise ValueError("Phase 3 candidate tuple has an unknown training tuple")
    return result


def _variant_id(condition_id: str, tuple_id: str) -> str:
    return f"{condition_id}--{tuple_id}"


def _authority_hashes(snapshot: Phase3ProtocolSnapshot) -> dict[str, str]:
    authority = snapshot.payload.get("authority")
    if not isinstance(authority, dict):
        raise ValueError("Phase 3 authority is missing")
    result = {
        "protocol_sha256": snapshot.sha256,
        "development_protocol_sha256": authority["development_protocol"]["sha256"],
        "development_tasks_sha256": authority["development_tasks"]["sha256"],
        "phase2_candidates_sha256": authority["phase2_candidates"]["sha256"],
        "phase2_selection_lock_sha256": authority["phase2_selection_lock"]["sha256"],
    }
    if any(not isinstance(value, str) or len(value) != 64 for value in result.values()):
        raise ValueError("Phase 3 authority hash is malformed")
    return result


def _new_condition_ids(snapshot: Phase3ProtocolSnapshot) -> tuple[str, ...]:
    rows = snapshot.payload.get("conditions")
    if not isinstance(rows, list):
        raise ValueError("Phase 3 condition matrix is missing")
    ids = tuple(row.get("condition_id") for row in rows if isinstance(row, dict))
    if ids[:2] != ("B2-global-listwise-optimum", "T-markov-state-transition-listwise-optimum"):
        raise ValueError("Phase 3 anchor condition order drifted")
    if ids[2:] != NEW_CONDITIONS:
        raise ValueError("Phase 3 new condition order drifted")
    return NEW_CONDITIONS


@dataclass(frozen=True, slots=True)
class Phase3View:
    """One temperature-independent representation view owner."""

    view_id: str
    condition_id: str
    fold_id: str
    heldout_family: str
    replicate: int
    training_task_ids: tuple[str, ...]
    data_order_seed: int
    evidence_lineage_sha256: str
    representation_sha256: str


@dataclass(frozen=True, slots=True)
class Phase3ModelOwner:
    """One temperature-independent model owner shared by three temperatures."""

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


@dataclass(frozen=True, slots=True)
class Phase3PlannedUnit:
    """One held-out development task and candidate tuple evaluation."""

    unit: PlannedUnit
    base_condition_id: str
    tuple_id: str
    training_tuple_id: str
    fold_id: str
    heldout_family: str
    model_owner_id: str
    view_id: str


@dataclass(frozen=True, slots=True)
class Phase3Plan:
    """Complete immutable logical plan; no outcomes or execution state."""

    schema_version: str
    plan_id: str
    protocol_sha256: str
    authority_hashes: tuple[tuple[str, str], ...]
    family_order: tuple[str, ...]
    replicates: tuple[int, ...]
    condition_ids: tuple[str, ...]
    candidate_tuple_ids: tuple[str, ...]
    views: tuple[Phase3View, ...]
    model_owners: tuple[Phase3ModelOwner, ...]
    units: tuple[Phase3PlannedUnit, ...]
    final_family_access: bool = False

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(item.unit.unit_id for item in self.units)

    @property
    def expected_units(self) -> tuple[PlannedUnit, ...]:
        return tuple(item.unit for item in self.units)


def _canonical_phase2_inputs(
    child_configs: Iterable[ExperimentConfig] | None,
    *,
    tuple_ids: tuple[str, ...],
) -> tuple[ExperimentConfig, ...]:
    canonical = tuple(screening_child_configs())
    configs = canonical if child_configs is None else tuple(child_configs)
    if configs != canonical:
        raise ValueError("supplied Phase 2 child configs differ from frozen authority")
    if len(configs) != len(FAMILIES):
        raise ValueError("Phase 3 requires exactly six canonical Phase 2 child configs")
    by_family = {config.parameters.get("heldout_family_id"): config for config in configs}
    if set(by_family) != set(FAMILIES) or tuple(by_family) != FAMILIES:
        raise ValueError("Phase 2 child family order or identity drifted")
    ordered = tuple(by_family[family] for family in FAMILIES)
    for family, config in zip(FAMILIES, ordered, strict=True):
        if config.split.final_tasks:
            raise ValueError("Phase 3 logical plan cannot include final tasks")
        if tuple(config.parameters.get("candidate_tuple_ids", ())) != tuple_ids:
            raise ValueError(f"Phase 2 candidate tuple authority drifted for {family}")
        if len(config.split.development_tasks) != 40 or len(config.split.validation_tasks) != 8:
            raise ValueError("Phase 2 child LOFO task matrix drifted")
        if any(task.family_id == family for task in config.split.development_tasks):
            raise ValueError("held-out family leaked into Phase 3 training tasks")
        if {task.family_id for task in config.split.validation_tasks} != {family}:
            raise ValueError("Phase 2 child validation family drifted")
    return ordered


def _phase2_seed_indices(
    config: ExperimentConfig, tuple_ids: tuple[str, ...]
) -> dict[str, dict[tuple[str, int], PlannedUnit]]:
    """Read the Phase 2 expected matrix once and index all candidate variants."""

    expected = plan_expected_units(config)
    indices = {
        tuple_id: {
            (item.key.task_id, item.key.replicate): item
            for item in expected.units
            if item.key.phase == PHASE and item.key.condition_id == _variant_id(B2, tuple_id)
        }
        for tuple_id in tuple_ids
    }
    if any(len(index) != 8 * len(REPLICATES) for index in indices.values()):
        raise ValueError("Phase 2 anchor seed matrix is incomplete")
    return indices


def _make_plan(
    snapshot: Phase3ProtocolSnapshot,
    configs: tuple[ExperimentConfig, ...],
) -> Phase3Plan:
    rows = _tuple_rows(snapshot)
    new_conditions = _new_condition_ids(snapshot)
    tuple_ids = tuple(row["tuple_id"] for row in rows)
    authority = _authority_hashes(snapshot)
    views: list[Phase3View] = []
    owners: list[Phase3ModelOwner] = []
    units: list[Phase3PlannedUnit] = []
    owner_by_key: dict[tuple[str, str, int, str], Phase3ModelOwner] = {}
    for family, config in zip(FAMILIES, configs, strict=True):
        fold_id = str(config.parameters.get("fold_id"))
        expected = plan_expected_units(config)
        expected_sha256 = expected_units_sha256(expected)
        seeds_by_tuple = _phase2_seed_indices(config, tuple_ids)
        training_ids = tuple(task.task_id for task in config.split.development_tasks)
        for condition_id in new_conditions:
            for replicate in REPLICATES:
                # View identity is independent of candidate temperature and training tuple.
                seed_anchor = seeds_by_tuple[tuple_ids[0]][
                    config.split.validation_tasks[0].task_id, replicate
                ]
                seed = seed_anchor.seeds
                representation_sha = _sha256_json(
                    {
                        "protocol_sha256": snapshot.sha256,
                        "condition_id": condition_id,
                        "schema_version": SCHEMA_VERSION,
                    }
                )
                evidence_lineage = _sha256_json(
                    {
                        "phase2_config_sha256": config.parameters.get(
                            "development_protocol_sha256"
                        ),
                        "expected_units_sha256": expected_sha256,
                        "training_task_ids": training_ids,
                        "replicate": replicate,
                    }
                )
                view_id = _sha256_json(
                    {
                        "condition_id": condition_id,
                        "fold_id": fold_id,
                        "replicate": replicate,
                        "data_order_seed": seed.data_order_seed,
                        "evidence_lineage_sha256": evidence_lineage,
                        "representation_sha256": representation_sha,
                        "authority": authority,
                    }
                )
                view = Phase3View(
                    view_id=view_id,
                    condition_id=condition_id,
                    fold_id=fold_id,
                    heldout_family=family,
                    replicate=replicate,
                    training_task_ids=training_ids,
                    data_order_seed=seed.data_order_seed,
                    evidence_lineage_sha256=evidence_lineage,
                    representation_sha256=representation_sha,
                )
                views.append(view)
                for training_tuple_id in TRAINING_TUPLE_IDS:
                    tuple_row = next(
                        row for row in rows if row["training_tuple_id"] == training_tuple_id
                    )
                    owner_id = _sha256_json(
                        {
                            "condition_id": condition_id,
                            "fold_id": fold_id,
                            "replicate": replicate,
                            "training_tuple_id": training_tuple_id,
                            "view_id": view_id,
                            "model_seed": seed.model_seed,
                            "authority": authority,
                        }
                    )
                    owner = Phase3ModelOwner(
                        owner_id=owner_id,
                        condition_id=condition_id,
                        fold_id=fold_id,
                        heldout_family=family,
                        replicate=replicate,
                        training_tuple_id=training_tuple_id,
                        view_id=view_id,
                        model_seed=seed.model_seed,
                        learning_rate=float(tuple_row["learning_rate"]),
                        training_epochs=int(tuple_row["training_epochs"]),
                        search_temperature_ids=tuple(
                            row["tuple_id"] for row in rows if row["training_tuple_id"] == training_tuple_id
                        ),
                    )
                    owners.append(owner)
                    owner_by_key[(condition_id, family, replicate, training_tuple_id)] = owner
                for task in config.split.validation_tasks:
                    for tuple_row in rows:
                        tuple_id = tuple_row["tuple_id"]
                        phase2_anchor = seeds_by_tuple[tuple_id][
                            (task.task_id, replicate)
                        ]
                        key = UnitKey(
                            phase=PHASE,
                            condition_id=_variant_id(condition_id, tuple_id),
                            family_id=family,
                            task_id=task.task_id,
                            task_index=phase2_anchor.key.task_index,
                            replicate=replicate,
                        )
                        exposure_hash = _sha256_json(
                            {
                                "protocol_sha256": snapshot.sha256,
                                "condition_id": condition_id,
                                "tuple_id": tuple_id,
                                "learner_visible": "optimum_only_development_training",
                            }
                        )
                        planned = PlannedUnit(
                            unit_id=unit_id_for(key),
                            key=key,
                            seeds=phase2_anchor.seeds,
                            exposure_manifest_sha256=exposure_hash,
                        )
                        owner = owner_by_key[(condition_id, family, replicate, tuple_row["training_tuple_id"])]
                        units.append(
                            Phase3PlannedUnit(
                                unit=planned,
                                base_condition_id=condition_id,
                                tuple_id=tuple_id,
                                training_tuple_id=tuple_row["training_tuple_id"],
                                fold_id=fold_id,
                                heldout_family=family,
                                model_owner_id=owner.owner_id,
                                view_id=view.view_id,
                            )
                        )
    plan_body = {
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": snapshot.sha256,
        "authority_hashes": authority,
        "family_order": FAMILIES,
        "replicates": REPLICATES,
        "condition_ids": new_conditions,
        "candidate_tuple_ids": tuple_ids,
        "views": [view.__dict__ if hasattr(view, "__dict__") else _dataclass_json(view) for view in views],
        "model_owners": [_dataclass_json(owner) for owner in owners],
        "units": [_dataclass_json(item) for item in units],
        "final_family_access": False,
    }
    plan_id = _sha256_json(plan_body)
    plan = Phase3Plan(
        schema_version=SCHEMA_VERSION,
        plan_id=plan_id,
        protocol_sha256=snapshot.sha256,
        authority_hashes=tuple(authority.items()),
        family_order=FAMILIES,
        replicates=REPLICATES,
        condition_ids=new_conditions,
        candidate_tuple_ids=tuple_ids,
        views=tuple(views),
        model_owners=tuple(owners),
        units=tuple(units),
    )
    return plan


def _dataclass_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _dataclass_json(value.model_dump(mode="json"))
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _dataclass_json(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, tuple):
        return [_dataclass_json(item) for item in value]
    if isinstance(value, list):
        return [_dataclass_json(item) for item in value]
    return value


def validate_phase3_plan(
    plan: Phase3Plan,
    *,
    protocol: Phase3ProtocolSnapshot | None = None,
    child_configs: Iterable[ExperimentConfig] | None = None,
) -> None:
    """Fail closed on matrix, identity, ownership, or final-scope drift."""

    if plan.schema_version != SCHEMA_VERSION or plan.final_family_access:
        raise ValueError("Phase 3 plan is not a development-only logical plan")
    if plan.family_order != FAMILIES or plan.replicates != REPLICATES:
        raise ValueError("Phase 3 plan family or replicate matrix drifted")
    if plan.condition_ids != NEW_CONDITIONS or len(plan.condition_ids) != 4:
        raise ValueError("Phase 3 new condition matrix drifted")
    if len(plan.candidate_tuple_ids) != 12 or len(set(plan.candidate_tuple_ids)) != 12:
        raise ValueError("Phase 3 candidate tuple matrix drifted")
    if len(plan.views) != 120 or len({view.view_id for view in plan.views}) != 120:
        raise ValueError("Phase 3 view matrix is incomplete or duplicated")
    if len(plan.model_owners) != 480 or len({owner.owner_id for owner in plan.model_owners}) != 480:
        raise ValueError("Phase 3 model-owner matrix is incomplete or duplicated")
    if len(plan.units) != 11520 or len(plan.unit_ids) != len(set(plan.unit_ids)):
        raise ValueError("Phase 3 unit matrix is incomplete or duplicated")
    if len({item.unit.key.model_dump_json() for item in plan.units}) != len(plan.units):
        raise ValueError("Phase 3 unit keys are duplicated")
    owner_ids = {owner.owner_id for owner in plan.model_owners}
    view_ids = {view.view_id for view in plan.views}
    if any(item.model_owner_id not in owner_ids or item.view_id not in view_ids for item in plan.units):
        raise ValueError("Phase 3 unit references a missing owner or view")
    if any(len(owner.search_temperature_ids) != 3 for owner in plan.model_owners):
        raise ValueError("Phase 3 model owner temperature reuse matrix drifted")
    if any(owner.condition_id not in NEW_CONDITIONS for owner in plan.model_owners):
        raise ValueError("Phase 3 model owner has an extra condition")
    if any(item.unit.key.phase != PHASE for item in plan.units):
        raise ValueError("Phase 3 logical plan contains a non-validation unit")
    if any(item.unit.key.family_id not in FAMILIES for item in plan.units):
        raise ValueError("Phase 3 logical plan contains an unknown family")
    snapshot = load_phase3_protocol() if protocol is None else protocol
    tuple_ids = tuple(row["tuple_id"] for row in _tuple_rows(snapshot))
    configs = _canonical_phase2_inputs(child_configs, tuple_ids=tuple_ids)
    canonical = _make_plan(snapshot, configs)
    if plan != canonical:
        raise ValueError("Phase 3 plan differs from the complete frozen authority")


def build_phase3_plan(
    *,
    protocol: Phase3ProtocolSnapshot | None = None,
    child_configs: Iterable[ExperimentConfig] | None = None,
) -> Phase3Plan:
    """Build the frozen Phase 3 development plan without touching outcomes."""

    snapshot = load_phase3_protocol() if protocol is None else protocol
    tuple_ids = tuple(row["tuple_id"] for row in _tuple_rows(snapshot))
    configs = _canonical_phase2_inputs(child_configs, tuple_ids=tuple_ids)
    plan = _make_plan(snapshot, configs)
    validate_phase3_plan(plan, protocol=snapshot, child_configs=configs)
    return plan


def load_phase3_plan() -> Phase3Plan:
    """Convenience alias used by execution code after the plan is committed."""

    return build_phase3_plan()
