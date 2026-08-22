"""Identity-only lineage gate for the frozen Phase 2 anchors used by Phase 3.

The Phase 3 protocol reuses the already completed B2 and C development runs.  This
module records that reuse without loading a model, evaluating an outcome, activating
any store, or writing an artifact.  It deliberately reads unit bytes through the
descriptor-pinned result namespace owned by :class:`RunStore`.  A custom byte reader
is accepted only behind an explicit private test gate.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol

from levelup.experiments.milestone6_phase3_protocol import (
    FAMILIES,
    PHASE3_PROTOCOL_PATH,
    Phase3ProtocolSnapshot,
    load_phase3_protocol,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import UnitRecord

SCHEMA_VERSION = "milestone6.phase3.anchor.v1"
ANCHOR_BASES = (
    "B2-global-listwise-optimum",
    "C-state-conditioned-listwise-optimum",
)
T_ALIAS = "T-markov-state-transition-listwise-optimum"
EXPECTED_OWNER_COUNT = 240
EXPECTED_UNIT_COUNT = 5_760
EXPECTED_OWNERS_PER_BASE = EXPECTED_OWNER_COUNT // len(ANCHOR_BASES)
EXPECTED_UNITS_PER_BASE = EXPECTED_UNIT_COUNT // len(ANCHOR_BASES)
EXPECTED_UNITS_PER_FOLD = EXPECTED_UNIT_COUNT // len(FAMILIES)
EXPECTED_PARAMETERS = {
    "B2-global-listwise-optimum": 3_601,
    "C-state-conditioned-listwise-optimum": 3_841,
}
TRAINING_TUPLE_IDS = (
    "lr0p003-e120",
    "lr0p003-e180",
    "lr0p01-e120",
    "lr0p01-e180",
)
CANDIDATE_TUPLE_IDS = tuple(
    f"{training_tuple_id}-{temperature}"
    for training_tuple_id in TRAINING_TUPLE_IDS
    for temperature in ("t0p6", "t0p9", "t1p2")
)
_HEX64 = frozenset("0123456789abcdef")


class ResultBytesReader(Protocol):
    """Read one completed unit result through a safe, already-pinned boundary."""

    def __call__(self, store: Any, unit_id: str) -> bytes: ...


class AnchorManifestError(ValueError):
    """Raised when an anchor runtime or manifest fails closed."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_dump(item) for item in value]
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _dump(item) for key, item in value.items()}
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {str(key): _dump(item) for key, item in attributes.items()}
    return value


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in _HEX64 for character in value
    ):
        raise AnchorManifestError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _base_condition(condition: Any) -> str | None:
    params = getattr(condition, "parameters", {})
    if isinstance(params, Mapping):
        base = params.get("base_condition_id")
        if isinstance(base, str):
            return base
    condition_id = getattr(condition, "condition_id", None)
    if not isinstance(condition_id, str):
        return None
    for base in ANCHOR_BASES:
        if condition_id == base or condition_id.startswith(base + "--"):
            return base
    return None


def _conditions_by_id(config: Any) -> dict[str, tuple[str, str]]:
    conditions = tuple(getattr(config, "conditions", ()))
    result: dict[str, tuple[str, str]] = {}
    for condition in conditions:
        condition_id = getattr(condition, "condition_id", None)
        base = _base_condition(condition)
        if not isinstance(condition_id, str) or base is None:
            continue
        parameters = getattr(condition, "parameters", {})
        candidate_tuple_id = (
            parameters.get("candidate_tuple_id")
            if isinstance(parameters, Mapping)
            else None
        )
        if candidate_tuple_id not in CANDIDATE_TUPLE_IDS:
            raise AnchorManifestError("anchor candidate-tuple identity is unknown")
        identity = (base, str(candidate_tuple_id))
        if condition_id in result and result[condition_id] != identity:
            raise AnchorManifestError("condition identity maps to multiple anchor bases")
        result[condition_id] = identity
    expected = {
        (base, candidate_tuple_id)
        for base in ANCHOR_BASES
        for candidate_tuple_id in CANDIDATE_TUPLE_IDS
    }
    if set(result.values()) != expected or len(result) != len(expected):
        raise AnchorManifestError("anchor condition grid is incomplete or extra")
    return result


def _require_development_runtime(runtime: Any) -> tuple[Any, ...]:
    """Check only immutable runtime facts; never activate or re-open by pathname."""

    manifest = getattr(runtime, "manifest", None)
    if manifest is None:
        raise AnchorManifestError("Phase 2 runtime has no readiness manifest")
    if tuple(getattr(manifest, "family_order", ())) != FAMILIES:
        raise AnchorManifestError("Phase 2 runtime family order is not frozen")
    for name in (
        "development_only",
        "final_family_access",
        "validation_executed",
        "search_executed",
        "selection_performed",
    ):
        if getattr(manifest, name, None) is not (True if name == "development_only" else False):
            raise AnchorManifestError(f"Phase 2 runtime {name} boundary is invalid")
    if tuple(getattr(runtime, "folds", ())) == ():
        raise AnchorManifestError("Phase 2 runtime has no development folds")
    folds = tuple(runtime.folds)
    if tuple(getattr(fold, "family_id", None) for fold in folds) != FAMILIES:
        raise AnchorManifestError("Phase 2 runtime fold order is not frozen")
    raw_root_identity = getattr(runtime, "raw_root_identity", None)
    if (
        not isinstance(raw_root_identity, tuple)
        or len(raw_root_identity) != 2
        or any(not isinstance(value, int) for value in raw_root_identity)
    ):
        raise AnchorManifestError("Phase 2 runtime lacks pinned filesystem identities")
    child_identities = tuple(getattr(runtime, "child_identities", ()))
    expected_child_ids = tuple(getattr(manifest, "child_run_ids", ()))
    if (
        len(child_identities) != len(FAMILIES)
        or tuple(row[0] for row in child_identities) != expected_child_ids
        or any(
            not isinstance(row[1], tuple)
            or len(row[1]) != 2
            or any(not isinstance(value, int) for value in row[1])
            for row in child_identities
        )
    ):
        raise AnchorManifestError("Phase 2 runtime child identities are not exact")
    if getattr(runtime, "result_namespace_snapshot", None) is None:
        raise AnchorManifestError("Phase 2 runtime lacks a result namespace snapshot")
    for name in (
        "manifest_parent_identity",
        "manifest_file_identity",
        "authority_sources",
        "tree_sha256",
        "provenance",
        "manifest_bytes",
    ):
        if getattr(runtime, name, None) in (None, (), b"", ""):
            raise AnchorManifestError(f"Phase 2 runtime lacks pinned {name}")
    for source in runtime.authority_sources:
        if (
            getattr(source, "parent_identity", None) is None
            or getattr(source, "file_identity", None) is None
            or not getattr(source, "content", b"")
            or not getattr(source, "sha256", "")
        ):
            raise AnchorManifestError("Phase 2 authority source is not identity-pinned")
    for fold in folds:
        config = getattr(fold, "config", None)
        if config is None or getattr(config.split, "final_tasks", ()):
            raise AnchorManifestError("Phase 2 anchor contains final tasks")
        if any(
            tuple(getattr(condition, "execution_phases", ())) != ("validation",)
            for condition in tuple(getattr(config, "conditions", ()))
        ):
            raise AnchorManifestError("Phase 2 anchor contains a non-development condition")
        store = getattr(fold, "store", None)
        if store is None:
            raise AnchorManifestError("Phase 2 fold has no locked RunStore")
        if bool(getattr(store, "_execution_ready", False)):
            raise AnchorManifestError("Phase 2 anchor store is execution-ready")
    return folds


@contextmanager
def _fold_result_bytes_reader(
    store: Any,
    test_reader: ResultBytesReader | None,
) -> Iterator[Callable[[str], bytes]]:
    if test_reader is not None:
        yield lambda unit_id: test_reader(store, unit_id)
        return
    opener = getattr(store, "_open_result_namespace", None)
    if not callable(opener):
        raise AnchorManifestError(
            "runtime store exposes no descriptor-pinned result-byte reader"
        )
    try:
        with opener("units") as (_, namespace_fd):
            yield lambda unit_id: secure_fs.read_bytes_at(
                namespace_fd, f"{unit_id}.json"
            )
    except (OSError, RuntimeError, TypeError, ValueError, secure_fs.SecureFilesystemError) as exc:
        raise AnchorManifestError("cannot read pinned Phase 2 unit results") from exc


def _validate_unit_bytes(raw: bytes, unit_id: str, planned: Any, store: Any) -> None:
    if not isinstance(raw, bytes) or not raw:
        raise AnchorManifestError("unit result bytes are missing or not immutable bytes")
    try:
        record = UnitRecord.model_validate_json(raw)
    except (TypeError, ValueError) as exc:
        raise AnchorManifestError("Phase 2 unit result bytes are not a typed UnitRecord") from exc
    if record.unit_id != unit_id or record.status != "completed":
        raise AnchorManifestError("Phase 2 unit result bytes do not match their unit ID")
    if (
        record.run_id != getattr(store, "run_id", None)
        or record.config_sha256 != getattr(store, "config_sha256", None)
        or record.key != planned.key
        or record.seeds != planned.seeds
        or record.exposure_manifest_sha256 != planned.exposure_manifest_sha256
    ):
        raise AnchorManifestError("Phase 2 unit result lineage differs from its plan")
    if raw != canonical_json_bytes(record.model_dump(mode="json")) + b"\n":
        raise AnchorManifestError("Phase 2 unit result bytes are not canonical")


def _model_owner_rows(folds: tuple[Any, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for fold in folds:
        family_id = str(fold.family_id)
        model_keys = getattr(getattr(fold, "model_keys", None), "models", None)
        manifests = getattr(getattr(fold, "models", None), "manifests", None)
        costs = getattr(getattr(fold, "models", None), "costs", None)
        computes = getattr(getattr(fold, "models", None), "compute", None)
        if not all(isinstance(value, Mapping) for value in (model_keys, manifests, costs, computes)):
            raise AnchorManifestError("Phase 2 model-owner inventory is unavailable")
        local: set[tuple[str, str, int]] = set()
        for identity, key in model_keys.items():
            if not isinstance(identity, tuple) or len(identity) != 3:
                raise AnchorManifestError("Phase 2 model-owner identity is malformed")
            base, training_tuple_id, replicate = identity
            if base not in ANCHOR_BASES:
                continue
            owner = (str(base), str(training_tuple_id), int(replicate))
            if owner in local:
                raise AnchorManifestError("duplicate Phase 2 model owner in one fold")
            local.add(owner)
            manifest = manifests.get(identity)
            cost = costs.get(identity)
            compute = computes.get(identity)
            if manifest is None or cost is None or compute is None:
                raise AnchorManifestError("Phase 2 model-owner lineage is incomplete")
            key_id = getattr(key, "key_id", None)
            artifact_id = getattr(manifest, "artifact_id", None)
            cost_id = getattr(cost, "cost_id", None)
            for value, label in (
                (key_id, "key_id"),
                (artifact_id, "artifact_id"),
                (cost_id, "cost_id"),
            ):
                _require_digest(value, label)
            if getattr(cost, "key_id", None) != key_id or getattr(cost, "artifact_id", None) != artifact_id:
                raise AnchorManifestError("Phase 2 model-owner cost lineage drifted")
            expected_cost_id = getattr(cost, "expected_cost_id", None)
            if not isinstance(expected_cost_id, str) or expected_cost_id != cost_id:
                raise AnchorManifestError("Phase 2 model-owner cost digest drifted")
            trainable_parameters = int(getattr(compute, "trainable_parameters"))
            if trainable_parameters != EXPECTED_PARAMETERS[owner[0]]:
                raise AnchorManifestError("Phase 2 model-owner capacity drifted")
            global_identity = (family_id, owner[0], owner[1], owner[2])
            if global_identity in seen:
                raise AnchorManifestError("duplicate Phase 2 model owner")
            seen.add(global_identity)
            rows.append(
                {
                    "family_id": family_id,
                    "base_condition_id": owner[0],
                    "training_tuple_id": owner[1],
                    "replicate": owner[2],
                    "key_id": key_id,
                    "artifact_id": artifact_id,
                    "cost_id": cost_id,
                    "model_manifest_sha256": _sha256(canonical_json_bytes(_dump(manifest))),
                    "trainable_parameters": trainable_parameters,
                    "optimizer_steps": int(getattr(compute, "optimizer_steps")),
                    "forward_passes": int(getattr(compute, "forward_passes")),
                }
            )
        if len(local) != 40:
            raise AnchorManifestError("each Phase 2 fold must contain exactly 40 B2/C model owners")
    if len(rows) != EXPECTED_OWNER_COUNT:
        raise AnchorManifestError("Phase 2 anchor does not contain exactly 240 model owners")
    if sum(row["base_condition_id"] == base for row in rows) != EXPECTED_OWNERS_PER_BASE:
        raise AnchorManifestError("Phase 2 model-owner base coverage is not exact")
    expected = {
        (family, base, training_tuple_id, replicate)
        for family in FAMILIES
        for base in ANCHOR_BASES
        for training_tuple_id in TRAINING_TUPLE_IDS
        for replicate in range(5)
    }
    observed = {
        (
            row["family_id"],
            row["base_condition_id"],
            row["training_tuple_id"],
            row["replicate"],
        )
        for row in rows
    }
    if observed != expected:
        raise AnchorManifestError("Phase 2 model-owner matrix is incomplete or extra")
    return sorted(rows, key=lambda row: tuple(str(row[key]) for key in (
        "family_id", "base_condition_id", "training_tuple_id", "replicate", "key_id"
    )))


def _unit_rows(
    folds: tuple[Any, ...],
    result_bytes_reader: ResultBytesReader | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold in folds:
        config = fold.config
        condition_map = _conditions_by_id(config)
        store = fold.store
        expected = getattr(getattr(store, "expected", None), "units", None)
        if not isinstance(expected, tuple):
            expected = tuple(expected or ())
        local = 0
        with _fold_result_bytes_reader(store, result_bytes_reader) as read_result:
            for planned in expected:
                unit_id = getattr(planned, "unit_id", None)
                condition_id = getattr(getattr(planned, "key", None), "condition_id", None)
                condition_identity = condition_map.get(condition_id)
                if condition_identity is None:
                    continue
                base, candidate_tuple_id = condition_identity
                if not isinstance(unit_id, str) or unit_id in seen:
                    raise AnchorManifestError("duplicate or malformed Phase 2 anchor unit ID")
                raw = read_result(unit_id)
                _validate_unit_bytes(raw, unit_id, planned, store)
                seen.add(unit_id)
                local += 1
                key = planned.key
                rows.append(
                    {
                        "unit_id": unit_id,
                        "result_id": unit_id,
                        "run_id": str(getattr(store, "run_id", "")),
                        "family_id": str(fold.family_id),
                        "base_condition_id": base,
                        "candidate_tuple_id": candidate_tuple_id,
                        "condition_id": str(condition_id),
                        "task_id": str(getattr(key, "task_id", "")),
                        "task_index": int(getattr(key, "task_index", -1)),
                        "replicate": int(getattr(key, "replicate", -1)),
                        "phase": str(getattr(key, "phase", "")),
                        "result_bytes": len(raw),
                        "result_bytes_sha256": _sha256(raw),
                    }
                )
        if local != EXPECTED_UNITS_PER_FOLD:
            raise AnchorManifestError("each Phase 2 fold must contain exactly 960 B2/C unit results")
        validation_tasks = tuple(getattr(config.split, "validation_tasks", ()))
        task_rows = tuple(
            (getattr(task, "task_id", None), getattr(task, "task_index", None))
            for task in validation_tasks
        )
        if (
            len(task_rows) != 8
            or len(set(task_rows)) != 8
            or any(
                not isinstance(task_id, str) or not isinstance(task_index, int)
                for task_id, task_index in task_rows
            )
        ):
            raise AnchorManifestError("Phase 2 fold validation-task matrix drifted")
        expected_local = {
            (base, candidate_tuple_id, task_id, task_index, replicate)
            for base in ANCHOR_BASES
            for candidate_tuple_id in CANDIDATE_TUPLE_IDS
            for task_id, task_index in task_rows
            for replicate in range(5)
        }
        observed_local = {
            (
                row["base_condition_id"],
                row["candidate_tuple_id"],
                row["task_id"],
                row["task_index"],
                row["replicate"],
            )
            for row in rows
            if row["family_id"] == fold.family_id
        }
        if observed_local != expected_local:
            raise AnchorManifestError("Phase 2 unit matrix is incomplete or extra")
        attempts = tuple(store.attempt_records())
        if attempts:
            raise AnchorManifestError("Phase 2 anchor contains execution attempts")
    if len(rows) != EXPECTED_UNIT_COUNT:
        raise AnchorManifestError("Phase 2 anchor does not contain exactly 5,760 unit results")
    if sum(row["base_condition_id"] == base for row in rows) != EXPECTED_UNITS_PER_BASE:
        raise AnchorManifestError("Phase 2 anchor unit base coverage is not exact")
    return sorted(rows, key=lambda row: tuple(str(row[key]) for key in (
        "family_id", "base_condition_id", "condition_id", "task_id", "task_index", "replicate", "unit_id"
    )))


@dataclass(frozen=True, slots=True)
class Phase3AnchorManifest:
    """Canonical identity-only manifest and its self-hash."""

    body: dict[str, Any]
    canonical_bytes: bytes
    anchor_manifest_sha256: str

    @property
    def sha256(self) -> str:
        return self.anchor_manifest_sha256

    def model_dump(self) -> dict[str, Any]:
        return dict(self.body)


def _lineage(runtime: Any, protocol: Phase3ProtocolSnapshot) -> dict[str, Any]:
    manifest = runtime.manifest
    authority = protocol.payload["authority"]
    selection = json.loads(dict(protocol.authority_bytes)["phase2_selection_lock"])
    return {
        "phase3_protocol_sha256": protocol.sha256,
        "development_protocol_sha256": str(authority["development_protocol"]["sha256"]),
        "development_tasks_sha256": str(authority["development_tasks"]["sha256"]),
        "phase2_candidates_sha256": str(authority["phase2_candidates"]["sha256"]),
        "phase2_selection_lock_sha256": str(authority["phase2_selection_lock"]["sha256"]),
        "phase2_selection_analysis_sha256": str(authority["phase2_selection_lock"]["analysis_sha256"]),
        "phase2_readiness_manifest_sha256": str(manifest.manifest_sha256),
        "phase2_readiness_manifest_bytes_sha256": _sha256(runtime.manifest_bytes),
        "phase2_result_namespace_snapshot_sha256": _sha256(
            canonical_json_bytes(_dump(runtime.result_namespace_snapshot))
        ),
        "phase2_tree_sha256": str(runtime.tree_sha256),
        "phase2_provenance_sha256": str(manifest.provenance_sha256),
        "selection_lock_schema_version": str(selection.get("schema_version", "")),
    }


def _canonical_tasks_by_family() -> dict[str, tuple[tuple[str, int], ...]]:
    from levelup.experiments.milestone6_phase2_screening import (
        screening_child_configs,
    )

    configs = screening_child_configs()
    rows: dict[str, tuple[tuple[str, int], ...]] = {}
    for config in configs:
        family = str(config.parameters["heldout_family_id"])
        if config.split.final_tasks:
            raise AnchorManifestError("canonical anchor config contains final tasks")
        tasks = tuple(
            (task.task_id, task.task_index) for task in config.split.validation_tasks
        )
        if len(tasks) != 8 or len(set(tasks)) != 8:
            raise AnchorManifestError("canonical anchor validation tasks drifted")
        rows[family] = tasks
    if tuple(rows) != FAMILIES:
        raise AnchorManifestError("canonical anchor family order drifted")
    return rows


def _validate_frozen_lineage(
    lineage: Mapping[str, Any],
    protocol: Phase3ProtocolSnapshot,
) -> None:
    gate = protocol.payload["canonical_evidence_reuse"]["anchor_lineage_gate"]
    selection = json.loads(dict(protocol.authority_bytes)["phase2_selection_lock"])
    selection_authority = selection["authority"]
    selection_analysis = selection["analysis"]
    expected = {
        "phase2_readiness_manifest_sha256": gate[
            "phase2_readiness_manifest_sha256"
        ],
        "phase2_readiness_manifest_bytes_sha256": gate[
            "phase2_readiness_manifest_bytes_sha256"
        ],
        "phase2_result_namespace_snapshot_sha256": gate[
            "phase2_result_namespace_snapshot_sha256"
        ],
        "phase2_tree_sha256": selection_authority["prepared_tree_sha256"],
        "phase2_provenance_sha256": selection_authority[
            "source_provenance_sha256"
        ],
    }
    if selection_authority["readiness_manifest_sha256"] != expected[
        "phase2_readiness_manifest_sha256"
    ]:
        raise AnchorManifestError("Phase 2 selection/readiness authority drifted")
    if selection_authority["readiness_manifest_bytes_sha256"] != expected[
        "phase2_readiness_manifest_bytes_sha256"
    ]:
        raise AnchorManifestError("Phase 2 selection/readiness bytes drifted")
    if selection_analysis["result_namespace_snapshot_sha256"] != expected[
        "phase2_result_namespace_snapshot_sha256"
    ]:
        raise AnchorManifestError("Phase 2 selection/result namespace drifted")
    for key, value in expected.items():
        if lineage.get(key) != value:
            raise AnchorManifestError(f"frozen anchor lineage changed: {key}")


def build_phase3_anchor_manifest(
    runtime: Any,
    *,
    protocol: Phase3ProtocolSnapshot | None = None,
    protocol_path: str | Path = PHASE3_PROTOCOL_PATH,
    result_bytes_reader: ResultBytesReader | None = None,
    _allow_test_reader: bool = False,
) -> Phase3AnchorManifest:
    """Build a deterministic identity-only B2/C anchor from a locked runtime.

    This function is intentionally read-only.  In particular it never calls
    ``recheck_before_execution`` or ``_activate_prepared_batch``.
    """

    folds = _require_development_runtime(runtime)
    snapshot = protocol or load_phase3_protocol(protocol_path)
    if snapshot.payload.get("final_family_access") is not False:
        raise AnchorManifestError("Phase 3 protocol permits final-family access")
    if result_bytes_reader is not None and not _allow_test_reader:
        raise AnchorManifestError("custom result-byte readers are test-only")
    owners = _model_owner_rows(folds)
    units = _unit_rows(folds, result_bytes_reader)
    lineage = _lineage(runtime, snapshot)
    for key, value in lineage.items():
        if key.endswith("sha256"):
            _require_digest(value, key)
    _validate_frozen_lineage(lineage, snapshot)
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scope": "known-development-only",
        "final_family_access": False,
        "new_execution": False,
        "aggregates": [],
        "final_results": [],
        "lineage": lineage,
        "t_alias": {
            "condition_id": T_ALIAS,
            "historical_condition_id": "C-state-conditioned-listwise-optimum",
            "source_base_condition_id": "C-state-conditioned-listwise-optimum",
            "analysis_only": True,
            "new_view": False,
            "new_model": False,
            "new_unit_results": False,
        },
        "counts": {
            "families": len(FAMILIES),
            "anchor_base_conditions": len(ANCHOR_BASES),
            "model_owners": len(owners),
            "unit_results": len(units),
        },
        "model_owners": owners,
        "unit_results": units,
    }
    digest = _sha256(canonical_json_bytes(body))
    body["anchor_manifest_sha256"] = digest
    payload = canonical_json_bytes(body)
    return Phase3AnchorManifest(body=body, canonical_bytes=payload, anchor_manifest_sha256=digest)


def validate_phase3_anchor_manifest(
    manifest: Phase3AnchorManifest | Mapping[str, Any],
    *,
    runtime: Any,
    protocol: Phase3ProtocolSnapshot | None = None,
    protocol_path: str | Path = PHASE3_PROTOCOL_PATH,
    result_bytes_reader: ResultBytesReader | None = None,
    _allow_test_reader: bool = False,
) -> Phase3AnchorManifest:
    """Validate exact inventories by rebuilding from the pinned Phase 2 runtime."""

    if isinstance(manifest, Phase3AnchorManifest):
        body = manifest.model_dump()
    elif isinstance(manifest, Mapping):
        body = dict(manifest)
    else:
        raise AnchorManifestError("anchor manifest must be a mapping")
    supplied = body.get("anchor_manifest_sha256")
    _require_digest(supplied, "anchor_manifest_sha256")
    unsigned = dict(body)
    unsigned.pop("anchor_manifest_sha256", None)
    expected = _sha256(canonical_json_bytes(unsigned))
    if supplied != expected:
        raise AnchorManifestError("anchor manifest self-hash mismatch")
    if body.get("schema_version") != SCHEMA_VERSION or body.get("scope") != "known-development-only":
        raise AnchorManifestError("anchor manifest schema or scope drifted")
    if body.get("final_family_access") is not False or body.get("new_execution") is not False:
        raise AnchorManifestError("anchor manifest permits final access or new execution")
    if body.get("aggregates") != [] or body.get("final_results") != []:
        raise AnchorManifestError("anchor manifest contains aggregate or final data")
    counts = body.get("counts")
    if counts != {"families": 6, "anchor_base_conditions": 2, "model_owners": EXPECTED_OWNER_COUNT, "unit_results": EXPECTED_UNIT_COUNT}:
        raise AnchorManifestError("anchor manifest counts drifted")
    if body.get("t_alias", {}).get("historical_condition_id") != ANCHOR_BASES[1] or body.get("t_alias", {}).get("analysis_only") is not True:
        raise AnchorManifestError("T alias is not analysis-only historical C")
    owners = body.get("model_owners")
    units = body.get("unit_results")
    if not isinstance(owners, list) or len(owners) != EXPECTED_OWNER_COUNT:
        raise AnchorManifestError("anchor model-owner inventory is incomplete")
    if not isinstance(units, list) or len(units) != EXPECTED_UNIT_COUNT:
        raise AnchorManifestError("anchor unit inventory is incomplete")
    owner_ids = [(row.get("family_id"), row.get("base_condition_id"), row.get("training_tuple_id"), row.get("replicate")) for row in owners]
    expected_owner_ids = {
        (family, base, training_tuple_id, replicate)
        for family in FAMILIES
        for base in ANCHOR_BASES
        for training_tuple_id in TRAINING_TUPLE_IDS
        for replicate in range(5)
    }
    if set(owner_ids) != expected_owner_ids or len(owner_ids) != len(set(owner_ids)):
        raise AnchorManifestError("anchor model-owner identities are duplicate or unknown")
    for row in owners:
        for key in ("key_id", "artifact_id", "cost_id", "model_manifest_sha256"):
            _require_digest(row.get(key), key)
        if row.get("trainable_parameters") != EXPECTED_PARAMETERS[row["base_condition_id"]]:
            raise AnchorManifestError("anchor model-owner capacity drifted")
    unit_ids = [row.get("unit_id") for row in units]
    if len(set(unit_ids)) != len(unit_ids) or any(not isinstance(value, str) or len(value) != 64 for value in unit_ids):
        raise AnchorManifestError("anchor unit identities are duplicate or malformed")
    for row in units:
        _require_digest(row.get("result_bytes_sha256"), "result_bytes_sha256")
        if (
            row.get("family_id") not in FAMILIES
            or row.get("base_condition_id") not in ANCHOR_BASES
            or row.get("candidate_tuple_id") not in CANDIDATE_TUPLE_IDS
            or row.get("phase") != "validation"
            or row.get("replicate") not in range(5)
        ):
            raise AnchorManifestError("anchor unit contains a non-B2/C or non-validation result")
        if (
            row.get("result_id") != row.get("unit_id")
            or not isinstance(row.get("run_id"), str)
            or not row["run_id"]
            or row.get("condition_id")
            != f"{row['base_condition_id']}--{row['candidate_tuple_id']}"
            or not isinstance(row.get("result_bytes"), int)
            or row["result_bytes"] < 1
        ):
            raise AnchorManifestError("anchor unit row identity is malformed")
    expected_tasks = _canonical_tasks_by_family()
    expected_unit_matrix = {
        (family, base, candidate_tuple_id, task_index, replicate)
        for family in FAMILIES
        for base in ANCHOR_BASES
        for candidate_tuple_id in CANDIDATE_TUPLE_IDS
        for _, task_index in expected_tasks[family]
        for replicate in range(5)
    }
    observed_unit_matrix = {
        (
            row["family_id"],
            row["base_condition_id"],
            row["candidate_tuple_id"],
            row["task_index"],
            row["replicate"],
        )
        for row in units
    }
    if observed_unit_matrix != expected_unit_matrix:
        raise AnchorManifestError("anchor unit matrix is incomplete or extra")
    expected_task_ids = {
        family: {task_index: task_id for task_id, task_index in tasks}
        for family, tasks in expected_tasks.items()
    }
    for row in units:
        if row["task_id"] != expected_task_ids[row["family_id"]].get(row["task_index"]):
            raise AnchorManifestError("anchor unit task identity drifted")
    rebuilt = build_phase3_anchor_manifest(
        runtime,
        protocol=protocol,
        protocol_path=protocol_path,
        result_bytes_reader=result_bytes_reader,
        _allow_test_reader=_allow_test_reader,
    )
    if rebuilt.canonical_bytes != canonical_json_bytes(body):
        raise AnchorManifestError("anchor manifest drifted from the locked runtime")
    canonical = canonical_json_bytes(body)
    return Phase3AnchorManifest(body=body, canonical_bytes=canonical, anchor_manifest_sha256=supplied)


# Descriptive aliases for callers that use “create” or “anchor” terminology.
create_phase3_anchor_manifest = build_phase3_anchor_manifest
validate_anchor_manifest = validate_phase3_anchor_manifest

__all__ = [
    "AnchorManifestError",
    "Phase3AnchorManifest",
    "ResultBytesReader",
    "build_phase3_anchor_manifest",
    "create_phase3_anchor_manifest",
    "validate_anchor_manifest",
    "validate_phase3_anchor_manifest",
]
