"""Canonical identity lock for the Phase 3 shared evidence artifacts.

This is a preparation-only boundary. It freshly rechecks an already
descriptor-validated Phase 2 runtime, reloads typed evidence costs through pinned
store descriptors, and records the exact condition-independent evidence identities
that every Phase 3 representation view must consume. It never runs an environment,
searches, asks an evaluator/oracle, or exposes payload samples to a learner.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from levelup.experiments.milestone6_phase2_screening_runtime import (
    ScreeningRuntime,
    recheck_screening_runtime_readonly,
)
from levelup.experiments.milestone6_phase3_anchor import (
    PHASE3_ANCHOR_MANIFEST_PATH,
    Phase3AnchorManifest,
    load_committed_phase3_anchor_manifest_bytes,
    require_phase3_anchor_manifest,
)
from levelup.experiments.milestone6_phase3_plan import (
    FAMILIES,
    REPLICATES,
    ValidatedPhase3Plan,
    load_committed_phase3_plan_lock_bytes,
    validate_phase3_plan_lock_bytes,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataEvidenceCostRecord,
    TrainingDataEvidenceKey,
    TrainingDataEvidenceManifest,
)

SCHEMA_VERSION = "milestone6.phase3.evidence-lock.v1"
PHASE3_EVIDENCE_LOCK_PATH = PHASE3_ANCHOR_MANIFEST_PATH.with_name(
    "phase3_evidence_lock.json"
)
EXPECTED_ARTIFACTS = len(FAMILIES) * len(REPLICATES)
_HEX = frozenset("0123456789abcdef")
_EVIDENCE_LOCK_TOKEN = object()


class EvidenceLockError(ValueError):
    """Raised when the frozen evidence lineage is incomplete or has drifted."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _sha256(canonical_json_bytes(value))


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX for char in value):
        raise EvidenceLockError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_dump(item) for item in value]
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _dump(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True, init=False)
class Phase3EvidenceLock:
    """Canonical identity-only evidence lock and its self-hash."""

    body: dict[str, Any]
    canonical_bytes: bytes
    evidence_lock_sha256: str
    _construction_token: object

    def __init__(
        self,
        *,
        body: dict[str, Any],
        canonical_bytes: bytes,
        evidence_lock_sha256: str,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _EVIDENCE_LOCK_TOKEN:
            raise EvidenceLockError(
                "evidence locks require the validated Phase 2 runtime gate"
            )
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "canonical_bytes", canonical_bytes)
        object.__setattr__(self, "evidence_lock_sha256", evidence_lock_sha256)
        object.__setattr__(self, "_construction_token", _construction_token)

    @property
    def sha256(self) -> str:
        return self.evidence_lock_sha256

    def model_dump(self) -> dict[str, Any]:
        return dict(self.body)


def require_phase3_evidence_lock(lock: Any) -> Phase3EvidenceLock:
    """Require a canonical lock produced by the validated evidence boundary."""

    unsigned = dict(getattr(lock, "body", {}))
    supplied = unsigned.pop("evidence_lock_sha256", None)
    if (
        not isinstance(lock, Phase3EvidenceLock)
        or lock._construction_token is not _EVIDENCE_LOCK_TOKEN
        or canonical_json_bytes(lock.body) != lock.canonical_bytes
        or supplied != lock.evidence_lock_sha256
        or _digest(unsigned) != supplied
    ):
        raise EvidenceLockError("Phase 3 evidence authority is not canonical")
    return lock


def load_committed_phase3_evidence_lock_bytes(
    path: str | os.PathLike[str] = PHASE3_EVIDENCE_LOCK_PATH,
) -> bytes:
    """Read and self-validate the committed evidence-lock authority safely."""

    target = Path(path).absolute()
    try:
        parent_fd = secure_fs.open_directory_chain(target.parent)
        try:
            content = secure_fs.read_bytes_at(parent_fd, target.name)
        finally:
            os.close(parent_fd)
        body = json.loads(content)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise EvidenceLockError(
            "committed Phase 3 evidence authority cannot be read safely"
        ) from exc
    if not isinstance(body, dict) or canonical_json_bytes(body) != content:
        raise EvidenceLockError("committed Phase 3 evidence authority is not canonical")
    supplied = body.get("evidence_lock_sha256")
    unsigned = dict(body)
    unsigned.pop("evidence_lock_sha256", None)
    if _digest(unsigned) != supplied:
        raise EvidenceLockError("committed Phase 3 evidence authority self-hash drifted")
    return content


def _require_gates(
    runtime: ScreeningRuntime,
    validated_plan: ValidatedPhase3Plan,
    anchor_manifest: Phase3AnchorManifest,
    anchor_file_bytes: bytes,
    plan_lock_bytes: bytes,
    *,
    _allow_test_runtime: bool,
) -> tuple[tuple[Any, ...], Any, Any]:
    if not _allow_test_runtime:
        try:
            recheck_screening_runtime_readonly(runtime)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise EvidenceLockError(
                "Phase 3 evidence requires a freshly revalidated ScreeningRuntime"
            ) from exc
    if not isinstance(validated_plan, ValidatedPhase3Plan):
        raise EvidenceLockError("Phase 3 evidence requires an opaque validated plan")
    plan = validated_plan.plan
    if getattr(plan, "final_family_access", True) is not False:
        raise EvidenceLockError("Phase 3 plan permits final-family access")
    if plan.family_order != FAMILIES or plan.replicates != REPLICATES:
        raise EvidenceLockError("Phase 3 plan family or replicate matrix drifted")
    if len(plan.views) != 120:
        raise EvidenceLockError("Phase 3 plan view matrix is incomplete")
    if len({view.view_id for view in plan.views}) != len(plan.views):
        raise EvidenceLockError("Phase 3 plan view identities are duplicated")
    try:
        require_phase3_anchor_manifest(anchor_manifest)
    except ValueError as exc:
        raise EvidenceLockError(
            "Phase 3 evidence requires a validated anchor manifest"
        ) from exc
    if not isinstance(anchor_file_bytes, bytes) or anchor_file_bytes != anchor_manifest.canonical_bytes:
        raise EvidenceLockError("anchor file bytes differ from canonical anchor authority")
    if not _allow_test_runtime:
        try:
            committed_anchor_bytes = load_committed_phase3_anchor_manifest_bytes()
        except ValueError as exc:
            raise EvidenceLockError(
                "committed Phase 3 anchor authority is unavailable"
            ) from exc
        if anchor_file_bytes != committed_anchor_bytes:
            raise EvidenceLockError(
                "anchor bytes differ from the committed Phase 3 authority"
            )
    try:
        locked_plan = validate_phase3_plan_lock_bytes(plan_lock_bytes)
    except (TypeError, ValueError) as exc:
        raise EvidenceLockError("Phase 3 plan lock bytes are invalid") from exc
    if locked_plan != plan:
        raise EvidenceLockError("Phase 3 plan lock differs from validated plan")
    if not _allow_test_runtime:
        try:
            committed_plan_bytes = load_committed_phase3_plan_lock_bytes()
        except ValueError as exc:
            raise EvidenceLockError(
                "committed Phase 3 plan authority is unavailable"
            ) from exc
        if plan_lock_bytes != committed_plan_bytes:
            raise EvidenceLockError(
                "plan lock bytes differ from the committed Phase 3 authority"
            )

    manifest = getattr(runtime, "manifest", None)
    folds = tuple(getattr(runtime, "folds", ()))
    if manifest is None or len(folds) != len(FAMILIES):
        raise EvidenceLockError("Phase 2 runtime is missing its development folds")
    if tuple(getattr(manifest, "family_order", ())) != FAMILIES:
        raise EvidenceLockError("Phase 2 runtime family order drifted")
    for name in (
        "development_only",
        "validation_executed",
        "search_executed",
        "outcomes_present",
        "selection_performed",
        "final_family_access",
    ):
        expected = True if name == "development_only" else False
        if getattr(manifest, name, None) is not expected:
            raise EvidenceLockError(f"Phase 2 runtime {name} boundary is invalid")
    if tuple(getattr(fold, "family_id", None) for fold in folds) != FAMILIES:
        raise EvidenceLockError("Phase 2 runtime fold order drifted")
    if any(getattr(fold.config.split, "final_tasks", ()) for fold in folds):
        raise EvidenceLockError("Phase 2 evidence runtime contains final tasks")
    _require_digest(plan.protocol_sha256, "phase3_protocol_sha256")
    _require_digest(plan.plan_id, "phase3_plan_id")
    anchor_lineage = anchor_manifest.body.get("lineage")
    if not isinstance(anchor_lineage, Mapping):
        raise EvidenceLockError("Phase 3 anchor lineage is missing")
    expected_anchor_lineage = {
        "phase3_protocol_sha256": plan.protocol_sha256,
        "phase2_readiness_manifest_sha256": getattr(manifest, "manifest_sha256", None),
        "phase2_readiness_manifest_bytes_sha256": _sha256(runtime.manifest_bytes),
        "phase2_result_namespace_snapshot_sha256": _digest(
            getattr(runtime, "result_namespace_snapshot", ())
        ),
        "phase2_tree_sha256": getattr(runtime, "tree_sha256", None),
        "phase2_provenance_sha256": getattr(manifest, "provenance_sha256", None),
    }
    if any(anchor_lineage.get(key) != value for key, value in expected_anchor_lineage.items()):
        raise EvidenceLockError("Phase 3 anchor lineage differs from the Phase 2 runtime")
    return folds, plan, manifest


def _lineage(
    runtime: Any,
    plan: Any,
    manifest: Any,
    anchor: Phase3AnchorManifest,
    anchor_file_bytes: bytes,
    plan_lock_bytes: bytes,
) -> dict[str, Any]:
    manifest_bytes = getattr(runtime, "manifest_bytes", None)
    if not isinstance(manifest_bytes, bytes) or not manifest_bytes:
        raise EvidenceLockError("Phase 2 runtime readiness bytes are missing")
    required = {
        "phase2_readiness_manifest_sha256": getattr(manifest, "manifest_sha256", None),
        "phase2_readiness_manifest_bytes_sha256": _sha256(manifest_bytes),
        "phase2_tree_sha256": getattr(runtime, "tree_sha256", None),
        "phase2_provenance_sha256": getattr(manifest, "provenance_sha256", None),
        "phase2_result_namespace_snapshot_sha256": _digest(
            getattr(runtime, "result_namespace_snapshot", ())
        ),
    }
    for key, value in required.items():
        _require_digest(value, key)
    result = {
        "phase3_protocol_sha256": plan.protocol_sha256,
        "phase3_plan_id": plan.plan_id,
        "phase2_readiness_manifest_sha256": required["phase2_readiness_manifest_sha256"],
        "phase2_readiness_manifest_bytes_sha256": required["phase2_readiness_manifest_bytes_sha256"],
        "phase2_tree_sha256": required["phase2_tree_sha256"],
        "phase2_provenance_sha256": required["phase2_provenance_sha256"],
        "phase2_result_namespace_snapshot_sha256": required[
            "phase2_result_namespace_snapshot_sha256"
        ],
        "phase3_anchor_manifest_sha256": anchor.anchor_manifest_sha256,
        "phase3_anchor_file_sha256": _sha256(anchor_file_bytes),
        "phase3_plan_lock_file_sha256": _sha256(plan_lock_bytes),
    }
    try:
        plan_lock = json.loads(plan_lock_bytes)
        result["phase3_plan_lock_sha256"] = plan_lock["plan_lock_sha256"]
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceLockError("Phase 3 plan lock identity is missing") from exc
    for key, value in result.items():
        if key.endswith("sha256") or key.endswith("_id"):
            if key.endswith("sha256"):
                _require_digest(value, key)
            elif not isinstance(value, str) or not value:
                raise EvidenceLockError(f"{key} is missing")
    return result


def _rows(
    folds: tuple[Any, ...],
    plan: Any,
    readiness: Any,
    *,
    provenance: Any,
    _allow_test_runtime: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observed: set[tuple[str, int]] = set()
    for fold in folds:
        family = str(fold.family_id)
        parameters = getattr(getattr(fold, "config", None), "parameters", {})
        fold_id_value = parameters.get("fold_id") if isinstance(parameters, Mapping) else None
        if not isinstance(fold_id_value, str) or not fold_id_value:
            raise EvidenceLockError("Phase 2 fold identity is missing")
        fold_id = fold_id_value
        if not _allow_test_runtime:
            from levelup.experiments.milestone6_phase2_screening_preparation import (
                build_screening_data_keys,
            )

            if fold.data_keys != build_screening_data_keys(fold.config, provenance):
                raise EvidenceLockError("Phase 2 evidence keys differ from canonical authority")
        views = tuple(view for view in plan.views if view.heldout_family == family)
        if len(views) != 20:
            raise EvidenceLockError(f"Phase 3 plan evidence views for {family} are incomplete")
        for replicate in REPLICATES:
            identity = (family, replicate)
            if identity in observed:
                raise EvidenceLockError("duplicate evidence fold/replicate identity")
            observed.add(identity)
            data_keys = getattr(fold, "data_keys", None)
            data = getattr(fold, "data", None)
            try:
                key = data_keys.evidence[replicate]
                manifest = data.manifests.evidence[replicate]
                cost_id = data.evidence_cost_ids[replicate]
            except (AttributeError, KeyError, TypeError) as exc:
                raise EvidenceLockError("Phase 2 evidence inventory is incomplete") from exc
            if not isinstance(key, TrainingDataEvidenceKey) or not isinstance(manifest, TrainingDataEvidenceManifest):
                raise EvidenceLockError("Phase 2 evidence inventory is not typed")
            if key.fold_id != fold_id or key.heldout_family_id != family or key.replicate != replicate:
                raise EvidenceLockError("evidence key fold or replicate identity drifted")
            if manifest.key != key or manifest.evidence_key_id != key.key_id:
                raise EvidenceLockError("evidence manifest key lineage drifted")
            development_tasks = tuple(getattr(fold.config.split, "development_tasks", ()))
            expected_training = tuple(getattr(task, "task_id", None) for task in development_tasks)
            heldout_tasks = tuple(getattr(fold.config.split, "validation_tasks", ()))
            expected_heldout = tuple(getattr(task, "task_id", None) for task in heldout_tasks)
            if (
                len(expected_training) != 40
                or any(not isinstance(task_id, str) for task_id in expected_training)
                or any(getattr(task, "family_id", None) == family for task in development_tasks)
                or key.ordered_training_task_ids != expected_training
                or len(expected_heldout) != 8
                or any(not isinstance(task_id, str) for task_id in expected_heldout)
                or key.ordered_heldout_task_ids != expected_heldout
            ):
                raise EvidenceLockError("evidence tasks differ from the exact development fold")
            if manifest.sample_task_ids != key.ordered_training_task_ids or len(key.ordered_training_task_ids) != 40:
                raise EvidenceLockError("evidence training task order is not the exact 40-task fold")
            try:
                cost = fold.store.load_shared_cost(
                    key.key_id,
                    "training_data_evidence",
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise EvidenceLockError("typed evidence cost cannot be reloaded") from exc
            if not isinstance(cost, TrainingDataEvidenceCostRecord):
                raise EvidenceLockError("evidence cost record is not typed")
            if (
                cost.cost_id != cost_id
                or cost.key != key
                or cost.key_id != key.key_id
                or cost.artifact_id != manifest.evidence_id
            ):
                raise EvidenceLockError("evidence cost lineage differs from typed storage")
            matching_views = tuple(view for view in views if view.replicate == replicate)
            if len(matching_views) != 4 or any(
                (
                    view.fold_id,
                    view.data_order_seed,
                    view.training_task_ids,
                )
                != (
                    key.fold_id,
                    key.data_order_seed,
                    key.ordered_training_task_ids,
                )
                for view in matching_views
            ):
                raise EvidenceLockError("Phase 3 view training task order differs from evidence")
            store = getattr(fold, "store", None)
            run_id = getattr(store, "run_id", None)
            if not isinstance(run_id, str) or not run_id:
                raise EvidenceLockError("Phase 2 child run identity is missing")
            children = tuple(getattr(readiness, "children", ()))
            child = next(
                (item for item in children if getattr(item, "heldout_family_id", None) == family),
                None,
            )
            if child is None or getattr(child, "run_id", None) != run_id:
                raise EvidenceLockError("Phase 2 child run identity differs from readiness")
            manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
            rows.append(
                {
                    "family_id": family,
                    "fold_id": fold_id,
                    "replicate": replicate,
                    "child_run_id": run_id,
                    "evidence_key_id": key.key_id,
                    "evidence_key": key.model_dump(mode="json"),
                    "evidence_id": manifest.evidence_id,
                    "evidence_manifest_key_id": manifest.evidence_key_id,
                    "evidence_manifest": manifest.model_dump(mode="json"),
                    "payload_sha256": manifest.payload_sha256,
                    "payload_bytes": manifest.payload_bytes,
                    "ordered_training_task_ids": list(key.ordered_training_task_ids),
                    "canonical_manifest_bytes_sha256": _sha256(manifest_bytes),
                    "evidence_cost_id": cost_id,
                    "evidence_cost": cost.model_dump(mode="json"),
                    "phase3_view_ids": [view.view_id for view in matching_views],
                }
            )
    if observed != {(family, rep) for family in FAMILIES for rep in REPLICATES} or len(rows) != EXPECTED_ARTIFACTS:
        raise EvidenceLockError("Phase 3 evidence artifact matrix is incomplete or extra")
    return rows


def _build_phase3_evidence_lock(
    runtime: ScreeningRuntime,
    validated_plan: ValidatedPhase3Plan,
    anchor_manifest: Phase3AnchorManifest,
    anchor_file_bytes: bytes,
    plan_lock_bytes: bytes,
    *,
    allow_test_runtime: bool,
) -> Phase3EvidenceLock:
    folds, plan, readiness = _require_gates(
        runtime,
        validated_plan,
        anchor_manifest,
        anchor_file_bytes,
        plan_lock_bytes,
        _allow_test_runtime=allow_test_runtime,
    )
    lineage = _lineage(
        runtime,
        plan,
        readiness,
        anchor_manifest,
        anchor_file_bytes,
        plan_lock_bytes,
    )
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scope": "known-development-only",
        "final_family_access": False,
        "payloads_included": False,
        "outcomes_included": False,
        "aggregates": [],
        "final_results": [],
        "lineage": lineage,
        "counts": {"families": len(FAMILIES), "replicates": len(REPLICATES), "evidence_artifacts": EXPECTED_ARTIFACTS},
        "evidence_artifacts": _rows(
            folds,
            plan,
            readiness,
            provenance=getattr(runtime, "provenance", None),
            _allow_test_runtime=allow_test_runtime,
        ),
    }
    digest = _digest(body)
    body["evidence_lock_sha256"] = digest
    payload = canonical_json_bytes(body)
    return Phase3EvidenceLock(
        body=body,
        canonical_bytes=payload,
        evidence_lock_sha256=digest,
        _construction_token=_EVIDENCE_LOCK_TOKEN,
    )


def build_phase3_evidence_lock(
    runtime: ScreeningRuntime,
    validated_plan: ValidatedPhase3Plan,
    anchor_manifest: Phase3AnchorManifest,
    anchor_file_bytes: bytes,
    plan_lock_bytes: bytes,
) -> Phase3EvidenceLock:
    """Build a lock only from a freshly revalidated real screening runtime."""

    return _build_phase3_evidence_lock(
        runtime,
        validated_plan,
        anchor_manifest,
        anchor_file_bytes,
        plan_lock_bytes,
        allow_test_runtime=False,
    )


def _build_phase3_evidence_lock_for_test(
    runtime: Any,
    validated_plan: ValidatedPhase3Plan,
    anchor_manifest: Phase3AnchorManifest,
    anchor_file_bytes: bytes,
    plan_lock_bytes: bytes,
) -> Phase3EvidenceLock:
    """Test-double adapter; unavailable through the public production API."""

    return _build_phase3_evidence_lock(
        runtime,
        validated_plan,
        anchor_manifest,
        anchor_file_bytes,
        plan_lock_bytes,
        allow_test_runtime=True,
    )


def _validate_phase3_evidence_lock(
    lock: Phase3EvidenceLock | Mapping[str, Any],
    *,
    runtime: Any,
    validated_plan: ValidatedPhase3Plan,
    anchor_manifest: Phase3AnchorManifest,
    anchor_file_bytes: bytes,
    plan_lock_bytes: bytes,
    allow_test_runtime: bool,
) -> Phase3EvidenceLock:
    body = lock.model_dump() if isinstance(lock, Phase3EvidenceLock) else dict(lock)
    supplied = body.get("evidence_lock_sha256")
    _require_digest(supplied, "evidence_lock_sha256")
    unsigned = dict(body)
    unsigned.pop("evidence_lock_sha256", None)
    if _digest(unsigned) != supplied:
        raise EvidenceLockError("evidence lock self-hash mismatch")
    if body.get("schema_version") != SCHEMA_VERSION or body.get("scope") != "known-development-only":
        raise EvidenceLockError("evidence lock schema or scope drifted")
    if body.get("final_family_access") is not False or body.get("payloads_included") is not False or body.get("outcomes_included") is not False:
        raise EvidenceLockError("evidence lock permits final access or learner payload/outcome data")
    if body.get("aggregates") != [] or body.get("final_results") != []:
        raise EvidenceLockError("evidence lock contains aggregate or final data")
    rebuilt = _build_phase3_evidence_lock(
        runtime,
        validated_plan,
        anchor_manifest,
        anchor_file_bytes,
        plan_lock_bytes,
        allow_test_runtime=allow_test_runtime,
    )
    if rebuilt.canonical_bytes != canonical_json_bytes(body):
        raise EvidenceLockError("evidence lock differs from the pinned runtime and plan")
    return Phase3EvidenceLock(
        body=body,
        canonical_bytes=canonical_json_bytes(body),
        evidence_lock_sha256=supplied,
        _construction_token=_EVIDENCE_LOCK_TOKEN,
    )


def validate_phase3_evidence_lock(
    lock: Phase3EvidenceLock | Mapping[str, Any],
    *,
    runtime: ScreeningRuntime,
    validated_plan: ValidatedPhase3Plan,
    anchor_manifest: Phase3AnchorManifest,
    anchor_file_bytes: bytes,
    plan_lock_bytes: bytes,
) -> Phase3EvidenceLock:
    return _validate_phase3_evidence_lock(
        lock,
        runtime=runtime,
        validated_plan=validated_plan,
        anchor_manifest=anchor_manifest,
        anchor_file_bytes=anchor_file_bytes,
        plan_lock_bytes=plan_lock_bytes,
        allow_test_runtime=False,
    )


def validate_phase3_evidence_lock_bytes(content: bytes, **kwargs: Any) -> Phase3EvidenceLock:
    if not isinstance(content, bytes) or not content:
        raise EvidenceLockError("evidence lock bytes are missing")
    try:
        body = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise EvidenceLockError("evidence lock bytes are not valid JSON") from exc
    if not isinstance(body, dict) or canonical_json_bytes(body) != content:
        raise EvidenceLockError("evidence lock bytes are not canonical")
    return validate_phase3_evidence_lock(body, **kwargs)


def _validate_phase3_evidence_lock_bytes_for_test(
    content: bytes,
    **kwargs: Any,
) -> Phase3EvidenceLock:
    if not isinstance(content, bytes) or not content:
        raise EvidenceLockError("evidence lock bytes are missing")
    try:
        body = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise EvidenceLockError("evidence lock bytes are not valid JSON") from exc
    if not isinstance(body, dict) or canonical_json_bytes(body) != content:
        raise EvidenceLockError("evidence lock bytes are not canonical")
    return _validate_phase3_evidence_lock(
        body,
        allow_test_runtime=True,
        **kwargs,
    )


__all__ = [
    "EvidenceLockError",
    "Phase3EvidenceLock",
    "build_phase3_evidence_lock",
    "load_committed_phase3_evidence_lock_bytes",
    "require_phase3_evidence_lock",
    "validate_phase3_evidence_lock",
    "validate_phase3_evidence_lock_bytes",
]
