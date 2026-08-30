"""Frozen logical plan for the Phase 3 local-affordance development matrix.

The plan in this module is an identity/authority object only.  Building it reads
the committed protocol, task manifest, and raw-capture summary, but does not read
raw evidence, prepare models, run an environment, search, replay, evaluate, or
write an artifact.  The complete matrix is deliberately materialised in memory
so later preparation/execution code cannot silently add a task, condition, or
temperature consumer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import PlannedUnit, UnitKey, UnitSeeds, unit_id_for

ROOT = Path(__file__).resolve().parents[3]
LOCAL_PROTOCOL_PATH = ROOT / "configs/milestone6/phase3_local_affordance_protocol.json"
DEVELOPMENT_PROTOCOL_PATH = ROOT / "configs/milestone6/development_protocol.json"
DEVELOPMENT_TASKS_PATH = ROOT / "configs/milestone6/development_tasks.json"
RAW_CAPTURE_SUMMARY_PATH = ROOT / "experiments/milestone6_phase3_local_affordance_raw_capture.json"
REPRESENTATION_LADDER_PATH = ROOT / "configs/milestone6/phase3_representation_ladder.json"
LOCAL_PLAN_LOCK_PATH = ROOT / "configs/milestone6/phase3_local_affordance_plan_lock.json"

SCHEMA_VERSION = "milestone6.phase3.local-affordance-logical-plan.v1"
FAMILIES = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
CONDITIONS = (
    "B2-global-listwise-optimum",
    "S-state-availability-listwise-optimum",
    "P-state-availability-alias-pooled-outcome-listwise-optimum",
    "L-state-availability-local-outcome-listwise-optimum",
)
REPLICATES = (0, 1, 2, 3, 4)
TUPLE_IDS = (
    "lr0p003-e120-t0p6", "lr0p003-e120-t0p9", "lr0p003-e120-t1p2",
    "lr0p003-e180-t0p6", "lr0p003-e180-t0p9", "lr0p003-e180-t1p2",
    "lr0p01-e120-t0p6", "lr0p01-e120-t0p9", "lr0p01-e120-t1p2",
    "lr0p01-e180-t0p6", "lr0p01-e180-t0p9", "lr0p01-e180-t1p2",
)
TRAINING_TUPLE_IDS = ("lr0p003-e120", "lr0p003-e180", "lr0p01-e120", "lr0p01-e180")
EXPECTED_COUNTS = {"views": 120, "model_owners": 480, "units": 11_520}
FROZEN_SOURCE_SHA256 = {
    "local_protocol": "a5b97f793cc72692943e44e7497f79e3e5528e65abd4badfa3b98c44e27896c2",
    "development_protocol": "7e6911c120db091e2b250f7a91520dd5f81a481cb4a19662eeae858c7da1c059",
    "development_tasks": "20f6606bd2150d808b18f011976bbf7c8298627e1cc01eeb67f653eacba9731f",
    "representation_ladder": "287b43ff8a3d7231162b6dfd9580af04073ad6d7b6eb030ac41fbf2121dd7afa",
    "raw_capture_summary": "9e853e4c099c0d49d2ffe9243f1917522f40dc710fbae9a421c4d3dfdba385cb",
}


class LocalAffordancePlanError(ValueError):
    """Raised when a frozen development authority is missing or has drifted."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _digest(value: Any) -> str:
    return _sha256(canonical_json_bytes(value))


def _json(content: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise LocalAffordancePlanError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise LocalAffordancePlanError(f"{label} must be a JSON object")
    return value


def _read(path: Path, label: str, expected_sha256: str | None = None) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise LocalAffordancePlanError(f"{label} must be a regular file")
    content = path.read_bytes()
    actual = _sha256(content)
    if expected_sha256 is not None and actual != expected_sha256:
        raise LocalAffordancePlanError(f"{label} source hash changed")
    return content, _json(content, label)


@dataclass(frozen=True, slots=True)
class LocalAffordanceView:
    view_id: str
    condition_id: str
    fold_id: str
    heldout_family: str
    replicate: int
    training_task_ids: tuple[str, ...]
    data_order_seed: int
    evidence_manifest_sha256: str
    representation_sha256: str


@dataclass(frozen=True, slots=True)
class LocalAffordanceModelOwner:
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
    architecture_sha256: str


@dataclass(frozen=True, slots=True)
class LocalAffordancePlannedUnit:
    unit: PlannedUnit
    condition_id: str
    tuple_id: str
    training_tuple_id: str
    fold_id: str
    heldout_family: str
    model_owner_id: str
    view_id: str


@dataclass(frozen=True, slots=True)
class LocalAffordancePlan:
    schema_version: str
    plan_id: str
    source_sha256: tuple[tuple[str, str], ...]
    raw_authority_manifest_id: str
    raw_authority_content_sha256: str
    raw_manifest_file_sha256: str
    family_order: tuple[str, ...]
    replicates: tuple[int, ...]
    condition_ids: tuple[str, ...]
    candidate_tuple_ids: tuple[str, ...]
    views: tuple[LocalAffordanceView, ...]
    model_owners: tuple[LocalAffordanceModelOwner, ...]
    units: tuple[LocalAffordancePlannedUnit, ...]
    final_family_access: bool = False

    @property
    def expected_units(self) -> tuple[PlannedUnit, ...]:
        return tuple(item.unit for item in self.units)

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(item.unit.unit_id for item in self.units)


def _tuple_rows(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = protocol.get("candidate_tuples")
    if isinstance(rows, list) and all(isinstance(row, str) for row in rows):
        if tuple(rows) != TUPLE_IDS:
            raise LocalAffordancePlanError("candidate tuple order drifted")
        # Numeric tuple values are read from the committed representation-ladder
        # source when available; the local protocol intentionally stores only IDs.
        return tuple({"tuple_id": item, "training_tuple_id": item.rsplit("-t", 1)[0]} for item in rows)
    if not isinstance(rows, list) or len(rows) != len(TUPLE_IDS):
        raise LocalAffordancePlanError("candidate tuple grid is incomplete")
    result = tuple(dict(row) for row in rows if isinstance(row, Mapping))
    if tuple(row.get("tuple_id") for row in result) != TUPLE_IDS:
        raise LocalAffordancePlanError("candidate tuple order drifted")
    if any(row.get("training_tuple_id") not in TRAINING_TUPLE_IDS for row in result):
        raise LocalAffordancePlanError("candidate training tuple identity drifted")
    return result


def _selected_tasks(tasks: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if tasks.get("schema_version") != "milestone6.development_tasks.v1":
        raise LocalAffordancePlanError("development task schema drifted")
    if tasks.get("family_order") != list(FAMILIES) or tasks.get("environment_reset_seed") != 0:
        raise LocalAffordancePlanError("development task authority drifted")
    rows = tasks.get("tasks")
    if not isinstance(rows, list):
        raise LocalAffordancePlanError("development task rows are missing")
    selected = []
    for row in rows:
        if not isinstance(row, dict) or "training_core" not in row.get("roles", ()):
            continue
        if row.get("family") not in FAMILIES or "known_development" not in row.get("roles", ()):
            raise LocalAffordancePlanError("development task scope drifted")
        selected.append(dict(row))
    ordered = tuple(sorted(selected, key=lambda r: (FAMILIES.index(r["family"]), r["task_index"])))
    if tuple(selected) != ordered or len(selected) != 48:
        raise LocalAffordancePlanError("development task order or count drifted")
    if any(sum(row["family"] == family for row in selected) != 8 for family in FAMILIES):
        raise LocalAffordancePlanError("development family task count drifted")
    return ordered


def _validate_local_protocol(protocol: Mapping[str, Any], rows: tuple[dict[str, Any], ...]) -> None:
    if (
        protocol.get("schema_version") != "milestone6.phase3.local-affordance-protocol.v2"
        or protocol.get("status") != "frozen-design-only"
        or protocol.get("scope") != "known-development-only"
        or protocol.get("execution") is not False
    ):
        raise LocalAffordancePlanError("local-affordance protocol is not frozen development-only")
    if protocol.get("execution_boundary", {}).get("final_family_access") is not False:
        raise LocalAffordancePlanError("local-affordance protocol permits final access")
    matrix = protocol.get("development_matrix")
    if not isinstance(matrix, dict) or matrix.get("family_order") != list(FAMILIES):
        raise LocalAffordancePlanError("local-affordance family matrix drifted")
    if matrix.get("units") != EXPECTED_COUNTS["units"] or matrix.get("model_owners") != EXPECTED_COUNTS["model_owners"]:
        raise LocalAffordancePlanError("local-affordance matrix counts drifted")
    conditions = protocol.get("conditions")
    if not isinstance(conditions, list) or tuple(c.get("condition_id") for c in conditions) != CONDITIONS:
        raise LocalAffordancePlanError("local-affordance condition order drifted")
    if tuple(r["tuple_id"] for r in _tuple_rows(protocol)) != tuple(r["tuple_id"] for r in rows):
        raise LocalAffordancePlanError("candidate tuple source drifted")
    capacity = protocol.get("capacity_matching", {})
    if capacity.get("counts") != {"B2": 3601, "S": 3841, "P": 3841, "L": 3841} or capacity.get("symmetric_parameter_tolerance_fraction") != 0.1:
        raise LocalAffordancePlanError("capacity authority drifted")
    fixed = protocol.get("shared_training_and_search", {})
    for key, expected in {"optimizer": "adam", "weight_decay": 0.0001, "device": "cpu", "probe_actions_per_task": 64, "candidate_episodes_per_task": 150, "adaptation_actions_per_task": 2048, "maximum_actions_per_candidate_episode": 64, "exact_optimum_affects_search_control_flow": False}.items():
        if fixed.get(key) != expected:
            raise LocalAffordancePlanError(f"fixed budget {key} drifted")


def _validate_raw_summary(raw: Mapping[str, Any]) -> None:
    if raw.get("schema_version") != "milestone6.phase3.local-affordance-raw-capture-summary.v1":
        raise LocalAffordancePlanError("raw capture summary schema drifted")
    if raw.get("scope") != "known-development-only" or raw.get("final_family_access") is not False:
        raise LocalAffordancePlanError("raw capture summary is not development-only")
    if raw.get("status") not in {"complete", "complete-with-accounting-metadata-loss"}:
        raise LocalAffordancePlanError("raw evidence capture is incomplete")
    for key, expected in {"artifact_count": 240, "key_count": 240, "heldout_binding_count": 240, "training_fold_count": 30, "physical_probe_actions": 15360, "logical_consumer_equivalent_actions": 737280}.items():
        if raw.get(key) != expected:
            raise LocalAffordancePlanError(f"raw summary {key} drifted")
    for key in ("raw_authority_manifest_id", "raw_authority_content_sha256", "raw_manifest_file_sha256"):
        value = raw.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise LocalAffordancePlanError(f"raw summary {key} is malformed")


def _build_local_affordance_plan(*, repository: str | Path = ROOT) -> LocalAffordancePlan:
    """Build the canonical plan body without invoking validation."""
    repo = Path(repository).resolve(strict=True)
    paths = {
        "local_protocol": repo / LOCAL_PROTOCOL_PATH.relative_to(ROOT),
        "development_protocol": repo / DEVELOPMENT_PROTOCOL_PATH.relative_to(ROOT),
        "development_tasks": repo / DEVELOPMENT_TASKS_PATH.relative_to(ROOT),
        "representation_ladder": repo / REPRESENTATION_LADDER_PATH.relative_to(ROOT),
    }
    payloads: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    for name, path in paths.items():
        content, payload = _read(path, name, FROZEN_SOURCE_SHA256[name])
        source_hashes[name] = _sha256(content)
        payloads[name] = payload
    raw_path = repo / RAW_CAPTURE_SUMMARY_PATH.relative_to(ROOT)
    raw_content, raw = _read(raw_path, "raw capture summary", FROZEN_SOURCE_SHA256["raw_capture_summary"])
    source_hashes["raw_capture_summary"] = _sha256(raw_content)
    rows = _tuple_rows(payloads["representation_ladder"])
    _validate_local_protocol(payloads["local_protocol"], rows)
    selected = _selected_tasks(payloads["development_tasks"])
    _validate_raw_summary(raw)
    tasks_by_family = {family: tuple(row for row in selected if row["family"] == family) for family in FAMILIES}
    views: list[LocalAffordanceView] = []
    owners: list[LocalAffordanceModelOwner] = []
    units: list[LocalAffordancePlannedUnit] = []
    owner_by_key: dict[tuple[str, str, int, str], LocalAffordanceModelOwner] = {}
    for family in FAMILIES:
        family_index = FAMILIES.index(family)
        # Raw-store training/heldout manifests use the held-out family itself as
        # their fold identity (e.g. ``plain.r0.json``), not a prefixed alias.
        fold_id = family
        training_tasks = tuple(row for other in FAMILIES if other != family for row in tasks_by_family[other])
        for replicate in REPLICATES:
            model_seed = 6_100_000 + family_index * 10_000 + replicate * 100_000
            data_seed = 6_400_000 + family_index * 10_000 + replicate * 100_000
            for condition in CONDITIONS:
                representation_sha = _digest({"protocol_sha256": source_hashes["local_protocol"], "condition_id": condition, "schema_version": SCHEMA_VERSION})
                evidence_sha = _digest({"raw_authority_manifest_id": raw["raw_authority_manifest_id"], "raw_authority_content_sha256": raw["raw_authority_content_sha256"], "fold_id": fold_id, "replicate": replicate, "training_task_ids": [r["task_id"] for r in training_tasks]})
                view_id = _digest({"condition_id": condition, "fold_id": fold_id, "replicate": replicate, "data_order_seed": data_seed, "evidence_sha256": evidence_sha, "representation_sha256": representation_sha})
                views.append(LocalAffordanceView(view_id, condition, fold_id, family, replicate, tuple(r["task_id"] for r in training_tasks), data_seed, evidence_sha, representation_sha))
                for training_tuple in TRAINING_TUPLE_IDS:
                    tuple_row = next(r for r in rows if r["training_tuple_id"] == training_tuple)
                    architecture_sha = _digest({"condition_id": condition, "input_width": 49 if condition.startswith("B2") else 54, "trainable_parameters": 3601 if condition.startswith("B2") else 3841, "hidden_widths": [48, 24], "optimizer": "adam"})
                    owner_id = _digest({"condition_id": condition, "fold_id": fold_id, "replicate": replicate, "training_tuple_id": training_tuple, "view_id": view_id, "model_seed": model_seed, "architecture_sha256": architecture_sha, "source_sha256": source_hashes})
                    owner = LocalAffordanceModelOwner(owner_id, condition, fold_id, family, replicate, training_tuple, view_id, model_seed, float(tuple_row["learning_rate"]), int(tuple_row["training_epochs"]), tuple(r["tuple_id"] for r in rows if r["training_tuple_id"] == training_tuple), 3601 if condition.startswith("B2") else 3841, architecture_sha)
                    owners.append(owner)
                    owner_by_key[(condition, family, replicate, training_tuple)] = owner
                for task in tasks_by_family[family]:
                    probe_seed = 6_200_000 + family_index * 10_000 + replicate * 100_000 + int(task["task_index"])
                    search_seed = 6_300_000 + family_index * 10_000 + replicate * 100_000 + int(task["task_index"])
                    for tuple_row in rows:
                        tuple_id = tuple_row["tuple_id"]
                        condition_variant = f"{condition}--{tuple_id}"
                        key = UnitKey(phase="validation", condition_id=condition_variant, family_id=family, task_id=task["task_id"], task_index=int(task["task_index"]), replicate=replicate)
                        exposure_hash = _digest({"protocol_sha256": source_hashes["local_protocol"], "condition_id": condition, "tuple_id": tuple_id, "learner_visible": "optimum_only_development_training"})
                        planned = PlannedUnit(unit_id=unit_id_for(key), key=key, seeds=UnitSeeds(model_seed=model_seed, environment_seed=0, probe_seed=probe_seed, search_seed=search_seed, data_order_seed=data_seed), exposure_manifest_sha256=exposure_hash)
                        owner = owner_by_key[(condition, family, replicate, tuple_row["training_tuple_id"])]
                        view = next(v for v in views if v.condition_id == condition and v.heldout_family == family and v.replicate == replicate)
                        units.append(LocalAffordancePlannedUnit(planned, condition, tuple_id, tuple_row["training_tuple_id"], fold_id, family, owner.owner_id, view.view_id))
    plan_body = {"schema_version": SCHEMA_VERSION, "source_sha256": source_hashes, "raw_authority_manifest_id": raw["raw_authority_manifest_id"], "raw_authority_content_sha256": raw["raw_authority_content_sha256"], "raw_manifest_file_sha256": raw["raw_manifest_file_sha256"], "family_order": FAMILIES, "replicates": REPLICATES, "condition_ids": CONDITIONS, "candidate_tuple_ids": TUPLE_IDS, "views": [_as_json(v) for v in views], "model_owners": [_as_json(o) for o in owners], "units": [_as_json(u) for u in units], "final_family_access": False}
    plan = LocalAffordancePlan(SCHEMA_VERSION, _digest(plan_body), tuple(source_hashes.items()), raw["raw_authority_manifest_id"], raw["raw_authority_content_sha256"], raw["raw_manifest_file_sha256"], FAMILIES, REPLICATES, CONDITIONS, TUPLE_IDS, tuple(views), tuple(owners), tuple(units))
    return plan


def build_local_affordance_plan(*, repository: str | Path = ROOT) -> LocalAffordancePlan:
    """Build and validate the complete 11,520-unit development-only plan."""
    plan = _build_local_affordance_plan(repository=repository)
    validate_local_affordance_plan(plan, repository=repository)
    return plan


def _as_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _as_json(value.model_dump(mode="json"))
    if hasattr(value, "__dataclass_fields__"):
        return {name: _as_json(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, tuple):
        return [_as_json(item) for item in value]
    if isinstance(value, list):
        return [_as_json(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _as_json(v) for k, v in value.items()}
    return value


def validate_local_affordance_plan(plan: LocalAffordancePlan, *, repository: str | Path = ROOT) -> None:
    if type(plan) is not LocalAffordancePlan or plan.final_family_access or plan.schema_version != SCHEMA_VERSION:
        raise LocalAffordancePlanError("plan is not a development-only local-affordance plan")
    if plan.family_order != FAMILIES or plan.replicates != REPLICATES or plan.condition_ids != CONDITIONS or plan.candidate_tuple_ids != TUPLE_IDS:
        raise LocalAffordancePlanError("plan identity matrix drifted")
    if len(plan.views) != EXPECTED_COUNTS["views"] or len(plan.model_owners) != EXPECTED_COUNTS["model_owners"] or len(plan.units) != EXPECTED_COUNTS["units"]:
        raise LocalAffordancePlanError("plan matrix is incomplete")
    if len({v.view_id for v in plan.views}) != len(plan.views) or len({o.owner_id for o in plan.model_owners}) != len(plan.model_owners) or len(plan.unit_ids) != len(set(plan.unit_ids)):
        raise LocalAffordancePlanError("plan identities are duplicated")
    if any(item.unit.key.phase != "validation" or item.unit.key.family_id not in FAMILIES for item in plan.units):
        raise LocalAffordancePlanError("plan contains a non-development unit")
    if any(o.search_temperature_ids != tuple(t for t in TUPLE_IDS if t.startswith(o.training_tuple_id + "-")) for o in plan.model_owners):
        raise LocalAffordancePlanError("temperature consumers are not reused canonically")
    if tuple(plan.source_sha256) != tuple((name, FROZEN_SOURCE_SHA256[name]) for name in ("local_protocol", "development_protocol", "development_tasks", "representation_ladder", "raw_capture_summary")):
        raise LocalAffordancePlanError("plan source hashes drifted")
    # Rebuilding is intentionally the final fail-closed identity check.  It also
    # catches tampering with seeds, exposure hashes, owners, or raw-capture digests.
    source_root = Path(repository).resolve(strict=True)
    for name, relative in (("local_protocol", LOCAL_PROTOCOL_PATH.relative_to(ROOT)), ("development_protocol", DEVELOPMENT_PROTOCOL_PATH.relative_to(ROOT)), ("development_tasks", DEVELOPMENT_TASKS_PATH.relative_to(ROOT)), ("representation_ladder", REPRESENTATION_LADDER_PATH.relative_to(ROOT)), ("raw_capture_summary", RAW_CAPTURE_SUMMARY_PATH.relative_to(ROOT))):
        content, _ = _read(source_root / relative, name, FROZEN_SOURCE_SHA256[name])
        if dict(plan.source_sha256).get(name) != _sha256(content):
            raise LocalAffordancePlanError("plan source bytes differ from its authority")
    _, raw = _read(source_root / RAW_CAPTURE_SUMMARY_PATH.relative_to(ROOT), "raw capture summary", FROZEN_SOURCE_SHA256["raw_capture_summary"])
    _validate_raw_summary(raw)
    if any(plan_value != raw_value for plan_value, raw_value in ((plan.raw_authority_manifest_id, raw["raw_authority_manifest_id"]), (plan.raw_authority_content_sha256, raw["raw_authority_content_sha256"]), (plan.raw_manifest_file_sha256, raw["raw_manifest_file_sha256"]))):
        raise LocalAffordancePlanError("raw authority digest differs from the committed capture")
    body = {"schema_version": SCHEMA_VERSION, "source_sha256": dict(plan.source_sha256), "raw_authority_manifest_id": plan.raw_authority_manifest_id, "raw_authority_content_sha256": plan.raw_authority_content_sha256, "raw_manifest_file_sha256": plan.raw_manifest_file_sha256, "family_order": plan.family_order, "replicates": plan.replicates, "condition_ids": plan.condition_ids, "candidate_tuple_ids": plan.candidate_tuple_ids, "views": [_as_json(v) for v in plan.views], "model_owners": [_as_json(o) for o in plan.model_owners], "units": [_as_json(u) for u in plan.units], "final_family_access": False}
    if plan.plan_id != _digest(body):
        raise LocalAffordancePlanError("plan self-hash mismatch")
    # Re-derive every semantic identity from the frozen sources.  This prevents
    # a caller from replacing a capacity, seed, owner, view, or association while
    # preserving the top-level plan self-hash.
    _, local_payload = _read(source_root / LOCAL_PROTOCOL_PATH.relative_to(ROOT), "local_protocol", FROZEN_SOURCE_SHA256["local_protocol"])
    _, ladder_payload = _read(source_root / REPRESENTATION_LADDER_PATH.relative_to(ROOT), "representation_ladder", FROZEN_SOURCE_SHA256["representation_ladder"])
    _, tasks_payload = _read(source_root / DEVELOPMENT_TASKS_PATH.relative_to(ROOT), "development_tasks", FROZEN_SOURCE_SHA256["development_tasks"])
    rows = _tuple_rows(ladder_payload)
    _validate_local_protocol(local_payload, rows)
    selected = _selected_tasks(tasks_payload)
    by_family = {family: tuple(row for row in selected if row["family"] == family) for family in FAMILIES}
    views_by_key = {(view.condition_id, view.heldout_family, view.replicate): view for view in plan.views}
    owners_by_key = {(owner.condition_id, owner.heldout_family, owner.replicate, owner.training_tuple_id): owner for owner in plan.model_owners}
    if len(views_by_key) != 120 or len(owners_by_key) != 480:
        raise LocalAffordancePlanError("plan view/owner identity matrix drifted")
    for family in FAMILIES:
        for replicate in REPLICATES:
            fi = FAMILIES.index(family)
            model_seed = 6_100_000 + fi * 10_000 + replicate * 100_000
            data_seed = 6_400_000 + fi * 10_000 + replicate * 100_000
            training_ids = tuple(row["task_id"] for other in FAMILIES if other != family for row in by_family[other])
            for condition in CONDITIONS:
                view = views_by_key[(condition, family, replicate)]
                representation_sha = _digest({"protocol_sha256": dict(plan.source_sha256)["local_protocol"], "condition_id": condition, "schema_version": SCHEMA_VERSION})
                evidence_sha = _digest({"raw_authority_manifest_id": plan.raw_authority_manifest_id, "raw_authority_content_sha256": plan.raw_authority_content_sha256, "fold_id": family, "replicate": replicate, "training_task_ids": list(training_ids)})
                view_id = _digest({"condition_id": condition, "fold_id": family, "replicate": replicate, "data_order_seed": data_seed, "evidence_sha256": evidence_sha, "representation_sha256": representation_sha})
                if view.view_id != view_id or view.fold_id != family or view.training_task_ids != training_ids or view.data_order_seed != data_seed or view.evidence_manifest_sha256 != evidence_sha or view.representation_sha256 != representation_sha:
                    raise LocalAffordancePlanError("view semantic identity drifted")
                for training_tuple in TRAINING_TUPLE_IDS:
                    owner = owners_by_key[(condition, family, replicate, training_tuple)]
                    count = 3601 if condition.startswith("B2") else 3841
                    arch = _digest({"condition_id": condition, "input_width": 49 if count == 3601 else 54, "trainable_parameters": count, "hidden_widths": [48, 24], "optimizer": "adam"})
                    owner_id = _digest({"condition_id": condition, "fold_id": family, "replicate": replicate, "training_tuple_id": training_tuple, "view_id": view_id, "model_seed": model_seed, "architecture_sha256": arch, "source_sha256": dict(plan.source_sha256)})
                    if owner.owner_id != owner_id or owner.model_seed != model_seed or owner.view_id != view_id or owner.trainable_parameters != count or owner.architecture_sha256 != arch:
                        raise LocalAffordancePlanError("model-owner semantic identity drifted")
    for item in plan.units:
        key = item.unit.key
        try:
            base, tuple_id = key.condition_id.rsplit("--", 1)
        except ValueError as exc:
            raise LocalAffordancePlanError("unit condition variant is malformed") from exc
        if base != item.condition_id or tuple_id != item.tuple_id or tuple_id not in TUPLE_IDS:
            raise LocalAffordancePlanError("unit condition/tuple association drifted")
        fi = FAMILIES.index(key.family_id)
        expected_seed = (6_100_000 + fi * 10_000 + key.replicate * 100_000, 6_200_000 + fi * 10_000 + key.replicate * 100_000 + key.task_index, 6_300_000 + fi * 10_000 + key.replicate * 100_000 + key.task_index, 6_400_000 + fi * 10_000 + key.replicate * 100_000)
        if (item.unit.seeds.model_seed, item.unit.seeds.probe_seed, item.unit.seeds.search_seed, item.unit.seeds.data_order_seed) != expected_seed:
            raise LocalAffordancePlanError("unit seed identity drifted")
        expected_exposure = _digest({"protocol_sha256": dict(plan.source_sha256)["local_protocol"], "condition_id": base, "tuple_id": tuple_id, "learner_visible": "optimum_only_development_training"})
        if item.unit.exposure_manifest_sha256 != expected_exposure:
            raise LocalAffordancePlanError("unit exposure identity drifted")
        owner = owners_by_key[(base, key.family_id, key.replicate, item.training_tuple_id)]
        view = views_by_key[(base, key.family_id, key.replicate)]
        if item.model_owner_id != owner.owner_id or item.view_id != view.view_id:
            raise LocalAffordancePlanError("unit owner/view association drifted")
    canonical = _build_local_affordance_plan(repository=source_root)
    if plan != canonical:
        raise LocalAffordancePlanError("plan differs from the canonical frozen authority")


def _plan_lock_body(plan: LocalAffordancePlan) -> dict[str, Any]:
    return {
        "schema_version": "milestone6.phase3.local-affordance-plan-lock.v1",
        "scope": "known-development-only",
        "final_family_access": False,
        "plan_id": plan.plan_id,
        "source_sha256": dict(plan.source_sha256),
        "raw_authority_manifest_id": plan.raw_authority_manifest_id,
        "raw_authority_content_sha256": plan.raw_authority_content_sha256,
        "raw_manifest_file_sha256": plan.raw_manifest_file_sha256,
        "family_order": list(plan.family_order),
        "replicates": list(plan.replicates),
        "condition_ids": list(plan.condition_ids),
        "candidate_tuple_ids": list(plan.candidate_tuple_ids),
        "counts": {key: len(getattr(plan, key)) for key in ("views", "model_owners", "units")},
        "view_ids_sha256": _digest([view.view_id for view in plan.views]),
        "model_owner_ids_sha256": _digest([owner.owner_id for owner in plan.model_owners]),
        "unit_ids_sha256": _digest(list(plan.unit_ids)),
    }


def canonical_local_affordance_plan_lock_bytes(plan: LocalAffordancePlan) -> bytes:
    """Serialize a compact, commit-ready authority lock (without writing it)."""
    validate_local_affordance_plan(plan)
    body = _plan_lock_body(plan)
    return canonical_json_bytes({**body, "plan_lock_sha256": _digest(body)}) + b"\n"


def validate_local_affordance_plan_lock_bytes(
    content: bytes, *, repository: str | Path = ROOT
) -> LocalAffordancePlan:
    """Validate lock bytes against the freshly rebuilt development plan."""
    if type(content) is not bytes or not content:
        raise LocalAffordancePlanError("plan lock bytes are missing")
    try:
        payload = json.loads(content)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise LocalAffordancePlanError("plan lock bytes are not valid JSON") from exc
    if not isinstance(payload, dict) or canonical_json_bytes(payload) + b"\n" != content:
        raise LocalAffordancePlanError("plan lock bytes are not canonical")
    supplied = payload.get("plan_lock_sha256")
    unsigned = dict(payload)
    unsigned.pop("plan_lock_sha256", None)
    if not isinstance(supplied, str) or _digest(unsigned) != supplied:
        raise LocalAffordancePlanError("plan lock self-hash mismatch")
    plan = build_local_affordance_plan(repository=repository)
    expected = json.loads(canonical_local_affordance_plan_lock_bytes(plan))
    if payload != expected:
        raise LocalAffordancePlanError("plan lock differs from the canonical authority")
    return plan


def load_committed_local_affordance_plan_lock(
    path: str | Path = LOCAL_PLAN_LOCK_PATH,
    *,
    repository: str | Path = ROOT,
) -> LocalAffordancePlan:
    """Load the exact committed development lock through the canonical validator."""

    lock_path = Path(path)
    if lock_path.is_symlink() or not lock_path.is_file():
        raise LocalAffordancePlanError("committed plan lock must be a regular file")
    return validate_local_affordance_plan_lock_bytes(
        lock_path.read_bytes(),
        repository=repository,
    )


__all__ = [
    "LOCAL_PLAN_LOCK_PATH",
    "LocalAffordancePlan",
    "LocalAffordancePlanError",
    "LocalAffordanceModelOwner",
    "LocalAffordancePlannedUnit",
    "LocalAffordanceView",
    "build_local_affordance_plan",
    "canonical_local_affordance_plan_lock_bytes",
    "load_committed_local_affordance_plan_lock",
    "validate_local_affordance_plan",
    "validate_local_affordance_plan_lock_bytes",
]
