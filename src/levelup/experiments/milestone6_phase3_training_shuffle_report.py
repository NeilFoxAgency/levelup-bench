"""Immutable development-only report for the Phase 3 shuffled-history control.

The report is a deliberately small, pre-execution sidecar.  It reconstructs the
H4-shuffled training views from the already frozen evidence authority and checks
the identity of every model owner that will consume a view.  No model tensor is
opened, and this module has no environment, search, replay, evaluator, oracle,
result-store, or final-family dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from levelup.experiments.milestone6_phase2_screening_runtime import (
    ScreeningRuntime,
    recheck_screening_runtime_readonly,
)
from levelup.experiments.milestone6_phase3_evidence import (
    Phase3EvidenceLock,
    require_phase3_evidence_lock,
)
from levelup.experiments.milestone6_phase3_model_artifacts import (
    ARTIFACTS_DIR,
    COSTS_DIR,
    KEYS_DIR,
    MANIFEST_NAME,
    Phase3ModelArtifactCost,
    Phase3ModelArtifactKey,
    Phase3ModelArtifactManifest,
    _fd_json,
    load_phase3_model_index_at,
    open_phase3_model_artifact_reader_at,
)
from levelup.experiments.milestone6_phase3_model_artifacts import (
    _digest as _model_digest,
)
from levelup.experiments.milestone6_phase3_model_authority import (
    Phase3ModelArtifactAuthority,
)
from levelup.experiments.milestone6_phase3_model_preparation import (
    _read_payload_bundle,
    _validate_bundle_lineage,
)
from levelup.experiments.milestone6_phase3_models import (
    H4_SHUFFLED_CONDITION,
    _model_identity_sha256,
    prepare_phase3_view,
)
from levelup.experiments.milestone6_phase3_plan import (
    FAMILIES,
    REPLICATES,
    ValidatedPhase3Plan,
    validate_phase3_plan,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataEvidenceKey,
    TrainingDataEvidenceManifest,
)

SCHEMA_VERSION = "milestone6.phase3.training-shuffle-report.v1"
EXPECTED_VIEWS = 30
EXPECTED_OWNERS = 120
OWNERS_PER_VIEW = 4
_HEX = frozenset("0123456789abcdef")
_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "scope",
        "development_only",
        "final_family_access",
        "outcomes_included",
        "search_included",
        "model_authority_sha256",
        "artifact_store_id",
        "counts",
        "views",
        "report_sha256",
    }
)
_VIEW_KEYS = frozenset(
    {
        "plan_id",
        "protocol_sha256",
        "evidence_lock_sha256",
        "view_id",
        "condition_id",
        "fold_id",
        "heldout_family",
        "replicate",
        "evidence_payload_sha256",
        "evidence_payload_bytes",
        "representation_identity_sha256",
        "model_owner_ids",
        "model_key_ids",
        "model_artifact_ids",
        "model_cost_ids",
        "model_manifest_sha256s",
        "model_identity_sha256s",
        "training_permutation_map_sha256",
        "eligible_windows",
        "map_nonidentity_windows",
        "effective_tensor_changed_windows",
        "duplicate_vector_no_effect_windows",
        "unchanged_short_windows",
        "effective_change_fraction",
        "claim_eligible",
    }
)


class Phase3TrainingShuffleReportError(ValueError):
    """Raised when the shuffled-history sidecar cannot be proven canonical."""


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX for c in value):
        raise Phase3TrainingShuffleReportError(f"{label} is not a SHA-256 digest")
    return value


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(k): _dump(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_dump(v) for v in value]
    return value


def _body_without_hash(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result.pop("report_sha256", None)
    return result


def _validate_body(body: Mapping[str, Any]) -> None:
    if set(body) != _REPORT_KEYS:
        raise Phase3TrainingShuffleReportError("training shuffle report schema differs")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("scope") != "known-development-only"
        or body.get("development_only") is not True
        or body.get("final_family_access") is not False
        or body.get("outcomes_included") is not False
        or body.get("search_included") is not False
    ):
        raise Phase3TrainingShuffleReportError("training shuffle report scope is invalid")
    _require_sha(body.get("model_authority_sha256"), "model authority")
    if (
        not isinstance(body.get("artifact_store_id"), str)
        or not body["artifact_store_id"]
        or "/" in body["artifact_store_id"]
        or "\\" in body["artifact_store_id"]
    ):
        raise Phase3TrainingShuffleReportError("model artifact store identity is invalid")
    supplied = body.get("report_sha256")
    _require_sha(supplied, "report self-hash")
    if _digest(_body_without_hash(body)) != supplied:
        raise Phase3TrainingShuffleReportError("training shuffle report self-hash drifted")
    counts = body.get("counts")
    if counts != {
        "families": 6,
        "replicates": 5,
        "views": EXPECTED_VIEWS,
        "owners": EXPECTED_OWNERS,
        "owners_per_view": OWNERS_PER_VIEW,
    }:
        raise Phase3TrainingShuffleReportError("training shuffle report coverage counts drifted")
    views = body.get("views")
    if not isinstance(views, list) or len(views) != EXPECTED_VIEWS:
        raise Phase3TrainingShuffleReportError("training shuffle report requires exactly 30 views")
    seen_views: set[str] = set()
    seen_owners: set[str] = set()
    seen_pairs: set[tuple[str, int]] = set()
    lineage: tuple[str, str, str] | None = None
    for view in views:
        if not isinstance(view, dict) or set(view) != _VIEW_KEYS:
            raise Phase3TrainingShuffleReportError("training shuffle view schema differs")
        view_id = view.get("view_id")
        if not isinstance(view_id, str) or view_id in seen_views:
            raise Phase3TrainingShuffleReportError("training shuffle view is duplicated")
        seen_views.add(view_id)
        if view.get("condition_id") != H4_SHUFFLED_CONDITION:
            raise Phase3TrainingShuffleReportError("report contains a non-shuffled view")
        if view.get("heldout_family") not in FAMILIES or view.get("replicate") not in REPLICATES:
            raise Phase3TrainingShuffleReportError("training shuffle view fold identity is invalid")
        pair = (view["heldout_family"], view["replicate"])
        if pair in seen_pairs:
            raise Phase3TrainingShuffleReportError("training shuffle family/replicate is duplicated")
        seen_pairs.add(pair)
        for field in (
            "plan_id",
            "protocol_sha256",
            "evidence_lock_sha256",
            "evidence_payload_sha256",
            "representation_identity_sha256",
            "training_permutation_map_sha256",
        ):
            _require_sha(view.get(field), field)
        current_lineage = (
            view["plan_id"],
            view["protocol_sha256"],
            view["evidence_lock_sha256"],
        )
        if lineage is None:
            lineage = current_lineage
        elif current_lineage != lineage:
            raise Phase3TrainingShuffleReportError("training shuffle authority lineage differs")
        if not isinstance(view.get("evidence_payload_bytes"), int) or view["evidence_payload_bytes"] < 1:
            raise Phase3TrainingShuffleReportError("evidence payload byte count is invalid")
        owners = view.get("model_owner_ids")
        key_ids = view.get("model_key_ids")
        artifact_ids = view.get("model_artifact_ids")
        cost_ids = view.get("model_cost_ids")
        manifest_sha256s = view.get("model_manifest_sha256s")
        identities = view.get("model_identity_sha256s")
        if (
            not isinstance(owners, list)
            or len(owners) != OWNERS_PER_VIEW
            or len(set(owners)) != OWNERS_PER_VIEW
            or not isinstance(identities, list)
            or len(identities) != OWNERS_PER_VIEW
            or not all(
                isinstance(values, list) and len(values) == OWNERS_PER_VIEW
                for values in (key_ids, artifact_ids, cost_ids, manifest_sha256s)
            )
        ):
            raise Phase3TrainingShuffleReportError("training shuffle owner coverage is invalid")
        for owner in owners:
            if (
                not isinstance(owner, str)
                or len(owner) != 64
                or any(character not in _HEX for character in owner)
            ):
                raise Phase3TrainingShuffleReportError("model owner identity is malformed")
            if owner in seen_owners:
                raise Phase3TrainingShuffleReportError("training shuffle model owner is duplicated")
            seen_owners.add(owner)
        for identity in identities:
            _require_sha(identity, "model identity")
        for label, values in (
            ("model key", key_ids),
            ("model artifact", artifact_ids),
            ("model cost", cost_ids),
            ("model manifest", manifest_sha256s),
        ):
            for value in values:
                _require_sha(value, label)
        if len(set(identities)) != OWNERS_PER_VIEW:
            raise Phase3TrainingShuffleReportError("model identities are duplicated within a view")
        counters = (
            "eligible_windows",
            "map_nonidentity_windows",
            "effective_tensor_changed_windows",
            "duplicate_vector_no_effect_windows",
            "unchanged_short_windows",
        )
        for field in counters:
            value = view.get(field)
            if not isinstance(value, int) or value < 0:
                raise Phase3TrainingShuffleReportError(f"{field} is invalid")
        eligible = view["eligible_windows"]
        changed = view["effective_tensor_changed_windows"]
        nonidentity = view["map_nonidentity_windows"]
        duplicate = view["duplicate_vector_no_effect_windows"]
        if (
            nonidentity > eligible
            or changed > nonidentity
            or changed + duplicate != eligible
        ):
            raise Phase3TrainingShuffleReportError(
                "training shuffle counters are internally inconsistent"
            )
        fraction = 1.0 if eligible == 0 else changed / eligible
        if view.get("effective_change_fraction") != fraction:
            raise Phase3TrainingShuffleReportError("effective-change fraction is not reproducible")
        if view.get("claim_eligible") is not (eligible > 0 and fraction >= 0.80):
            raise Phase3TrainingShuffleReportError("claim eligibility is not reproducible")
    if (
        len(seen_views) != EXPECTED_VIEWS
        or len(seen_owners) != EXPECTED_OWNERS
        or seen_pairs != {(family, replicate) for family in FAMILIES for replicate in REPLICATES}
    ):
        raise Phase3TrainingShuffleReportError("training shuffle report owner/view coverage is incomplete")


@dataclass(frozen=True, slots=True)
class Phase3TrainingShuffleReport:
    """Canonical immutable sidecar and its self-hash."""

    body: dict[str, Any]
    canonical_bytes: bytes
    report_sha256: str

    @property
    def views(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.body["views"])

    def model_dump(self) -> dict[str, Any]:
        return dict(self.body)


def _authority_field(authority: Any, field: str, default: Any = None) -> Any:
    if isinstance(authority, Mapping):
        return authority.get(field, default)
    return getattr(authority, field, default)


def _row_field(row: Any, field: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(field)
    return getattr(row, field, None)


def _model_authority_rows(authority: Any) -> dict[str, Any]:
    authority_rows = tuple(_authority_field(authority, "models", ()))
    rows: dict[str, Any] = {}
    for row in authority_rows:
        owner_id = _row_field(row, "owner_id")
        if not isinstance(owner_id, str) or owner_id in rows:
            raise Phase3TrainingShuffleReportError(
                "model authority owner identity is malformed or duplicated"
            )
        rows[owner_id] = row
    if (
        len(authority_rows) != 480
        or len(rows) != 480
        or set(rows) != set(_authority_field(authority, "owner_ids", ()))
    ):
        raise Phase3TrainingShuffleReportError("model authority owner universe is incomplete")
    return rows


def _plan_from_validated(value: Any) -> Any:
    if not isinstance(value, ValidatedPhase3Plan):
        raise Phase3TrainingShuffleReportError("report requires an opaque validated Phase 3 plan")
    plan = value.plan
    try:
        validate_phase3_plan(plan)
    except (TypeError, ValueError) as exc:
        raise Phase3TrainingShuffleReportError(
            "Phase 3 plan is not the complete frozen authority"
        ) from exc
    if (
        plan.final_family_access is not False
        or len(plan.views) != 120
        or len(plan.model_owners) != 480
    ):
        raise Phase3TrainingShuffleReportError("Phase 3 plan coverage or final-family scope is invalid")
    return plan


def _load_owner_keys(
    artifact_output_root: str | os.PathLike[str],
    owner_ids: tuple[str, ...],
    authority: Phase3ModelArtifactAuthority,
) -> tuple[dict[str, Phase3ModelArtifactKey], dict[str, Any]]:
    """Read only model-key JSON through pinned descriptors; never load tensors."""

    rows = _model_authority_rows(authority)
    root = Path(artifact_output_root).absolute()
    if root.name != _authority_field(authority, "artifact_store_id"):
        raise Phase3TrainingShuffleReportError("model artifact root differs from authority")
    try:
        root_fd = secure_fs.open_directory_chain(root)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Phase3TrainingShuffleReportError("model artifact root cannot be descriptor-pinned") from exc
    identities: dict[str, tuple[int, int]] = {
        "root": secure_fs.directory_identity(root_fd)
    }
    try:
        with open_phase3_model_artifact_reader_at(root_fd) as reader:
            identities.update(
                {
                    KEYS_DIR: secure_fs.directory_identity(reader.keys_fd),
                    COSTS_DIR: secure_fs.directory_identity(reader.costs_fd),
                    ARTIFACTS_DIR: secure_fs.directory_identity(reader.artifacts_fd),
                }
            )
            loaded: dict[str, Phase3ModelArtifactKey] = {}
            for owner_id in owner_ids:
                row = rows.get(owner_id)
                row_key_id = _row_field(row, "key_id")
                if not isinstance(row_key_id, str):
                    raise Phase3TrainingShuffleReportError("model authority key identity is missing")
                try:
                    index = load_phase3_model_index_at(reader, row_key_id)
                    key = index.key
                    cost = Phase3ModelArtifactCost.model_validate(
                        _fd_json(reader.costs_fd, f"{row_key_id}.json")
                    )
                    artifact_fd = secure_fs.open_child_directory(
                        reader.artifacts_fd, index.artifact_id
                    )
                    try:
                        manifest = Phase3ModelArtifactManifest.model_validate(
                            _fd_json(artifact_fd, MANIFEST_NAME)
                        )
                    finally:
                        os.close(artifact_fd)
                except (TypeError, ValueError) as exc:
                    raise Phase3TrainingShuffleReportError(
                        "model key/index/cost/manifest JSON is invalid"
                    ) from exc
                if (
                    key.owner_id != owner_id
                    or key.key_id != row_key_id
                    or index.artifact_id != _row_field(row, "artifact_id")
                    or index.manifest_sha256 != _row_field(row, "manifest_sha256")
                    or cost.key != key
                    or cost.key_id != row_key_id
                    or cost.artifact_id != index.artifact_id
                    or cost.cost_id != _row_field(row, "cost_id")
                    or manifest.key != key
                    or manifest.artifact_id != index.artifact_id
                    or _model_digest(manifest.model_dump(mode="json"))
                    != index.manifest_sha256
                ):
                    raise Phase3TrainingShuffleReportError(
                        "model key/index/cost/manifest lineage differs from authority"
                    )
                loaded[owner_id] = key
            return loaded, rows
    except Phase3TrainingShuffleReportError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Phase3TrainingShuffleReportError("model keys cannot be read through pinned descriptors") from exc
    finally:
        os.close(root_fd)
        try:
            current_root = secure_fs.open_directory_chain(root)
            current_children: list[int] = []
            try:
                current = {"root": secure_fs.directory_identity(current_root)}
                for name in (KEYS_DIR, COSTS_DIR, ARTIFACTS_DIR):
                    child = secure_fs.open_child_directory(current_root, name)
                    current_children.append(child)
                    current[name] = secure_fs.directory_identity(child)
            finally:
                for child in reversed(current_children):
                    os.close(child)
                os.close(current_root)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise Phase3TrainingShuffleReportError(
                "model artifact path changed during key reads"
            ) from exc
        if current != identities:
            raise Phase3TrainingShuffleReportError(
                "model artifact identity changed during key reads"
            )


def _prepared_view(
    runtime: Any,
    plan: Any,
    evidence_lock: Phase3EvidenceLock,
    view: Any,
) -> Any:
    rows = evidence_lock.body.get("evidence_artifacts", ())
    if not isinstance(rows, (tuple, list)) or any(
        not isinstance(item, Mapping) for item in rows
    ):
        raise Phase3TrainingShuffleReportError(
            "evidence authority rows are not typed mappings"
        )
    row = next(
        (item for item in rows if item.get("family_id") == view.heldout_family and item.get("replicate") == view.replicate),
        None,
    )
    if not isinstance(row, Mapping):
        raise Phase3TrainingShuffleReportError("H4-shuffled view has no exact evidence row")
    try:
        folds = {str(item.family_id): item for item in runtime.folds}
        fold = folds[view.heldout_family]
        typed_key = TrainingDataEvidenceKey.model_validate(row["evidence_key"])
        typed_manifest = TrainingDataEvidenceManifest.model_validate(row["evidence_manifest"])
        bundle = _read_payload_bundle(fold, typed_manifest.evidence_id, typed_key)
        _validate_bundle_lineage(bundle, {"manifest": typed_manifest, "row": row})
        # The canonical preparation function is the only permitted constructor.
        return prepare_phase3_view(
            bundle.payload,
            bundle.manifest,
            view,
            plan=plan,
            payload_bytes=bundle.payload_bytes,
        )
    except Phase3TrainingShuffleReportError:
        raise
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Phase3TrainingShuffleReportError("frozen H4-shuffled view cannot be reconstructed") from exc


def build_phase3_training_shuffle_report(
    runtime: Any,
    validated_plan: ValidatedPhase3Plan,
    evidence_lock: Phase3EvidenceLock,
    model_authority: Phase3ModelArtifactAuthority,
    artifact_output_root: str | os.PathLike[str],
) -> Phase3TrainingShuffleReport:
    """Reconstruct and validate the exact 30-view shuffled training sidecar."""

    if not isinstance(runtime, ScreeningRuntime):
        raise Phase3TrainingShuffleReportError(
            "training shuffle report requires a typed development runtime"
        )
    try:
        recheck_screening_runtime_readonly(runtime)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Phase3TrainingShuffleReportError(
            "development runtime is not freshly descriptor-validated"
        ) from exc
    plan = _plan_from_validated(validated_plan)
    try:
        lock = require_phase3_evidence_lock(evidence_lock)
    except (TypeError, ValueError) as exc:
        raise Phase3TrainingShuffleReportError("evidence lock is not canonical") from exc
    if not isinstance(model_authority, Phase3ModelArtifactAuthority):
        raise Phase3TrainingShuffleReportError(
            "training shuffle report requires a typed model authority"
        )
    try:
        revalidated_authority = Phase3ModelArtifactAuthority.model_validate(
            model_authority.model_dump(mode="json")
        )
    except (TypeError, ValueError) as exc:
        raise Phase3TrainingShuffleReportError("model authority is not canonical") from exc
    if (
        revalidated_authority != model_authority
        or model_authority.authority_sha256
        != model_authority.expected_authority_sha256
        or
        _authority_field(model_authority, "development_only") is not True
        or _authority_field(model_authority, "final") is not False
        or _authority_field(model_authority, "final_family_accessed") is not False
        or _authority_field(model_authority, "plan_id") != plan.plan_id
        or _authority_field(model_authority, "protocol_sha256") != plan.protocol_sha256
        or _authority_field(model_authority, "evidence_lock_sha256") != lock.evidence_lock_sha256
    ):
        raise Phase3TrainingShuffleReportError("model authority lineage differs from frozen plan/evidence")
    if tuple(_authority_field(model_authority, "family_order", ())) != FAMILIES:
        raise Phase3TrainingShuffleReportError("model authority family order drifted")
    evidence_rows = lock.body.get("evidence_artifacts", ())
    if (
        not isinstance(evidence_rows, (tuple, list))
        or len(evidence_rows) != EXPECTED_VIEWS
        or any(not isinstance(item, Mapping) for item in evidence_rows)
        or {
            (item.get("family_id"), item.get("replicate")) for item in evidence_rows
        }
        != {(family, replicate) for family in FAMILIES for replicate in REPLICATES}
    ):
        raise Phase3TrainingShuffleReportError(
            "evidence authority is not the exact 30-row development matrix"
        )
    views = tuple(view for view in plan.views if view.condition_id == H4_SHUFFLED_CONDITION)
    if len(views) != EXPECTED_VIEWS or {(v.heldout_family, v.replicate) for v in views} != {(f, r) for f in FAMILIES for r in REPLICATES}:
        raise Phase3TrainingShuffleReportError("H4-shuffled view matrix is not exactly 30 views")
    owners = tuple(owner for owner in plan.model_owners if owner.condition_id == H4_SHUFFLED_CONDITION)
    if len(owners) != EXPECTED_OWNERS:
        raise Phase3TrainingShuffleReportError("H4-shuffled owner matrix is not exactly 120 owners")
    keys, authority_rows = _load_owner_keys(
        artifact_output_root,
        tuple(owner.owner_id for owner in owners),
        model_authority,
    )
    rows: list[dict[str, Any]] = []
    for view in views:
        prepared = _prepared_view(runtime, plan, lock, view)
        shuffle = getattr(prepared, "history_shuffle", None)
        if shuffle is None or not isinstance(getattr(shuffle, "permutation_map_sha256", None), str):
            raise Phase3TrainingShuffleReportError("H4-shuffled preparation has no permutation map")
        view_owners = tuple(owner for owner in owners if owner.view_id == view.view_id)
        if len(view_owners) != OWNERS_PER_VIEW:
            raise Phase3TrainingShuffleReportError("H4-shuffled view does not have four owners")
        if any(
            (
                owner.condition_id,
                owner.view_id,
                owner.fold_id,
                owner.heldout_family,
                owner.replicate,
            )
            != (
                view.condition_id,
                view.view_id,
                view.fold_id,
                view.heldout_family,
                view.replicate,
            )
            for owner in view_owners
        ):
            raise Phase3TrainingShuffleReportError(
                "H4-shuffled owner-to-view lineage differs"
            )
        identity_values: list[str] = []
        key_ids: list[str] = []
        artifact_ids: list[str] = []
        cost_ids: list[str] = []
        manifest_sha256s: list[str] = []
        for owner in view_owners:
            key = keys.get(owner.owner_id)
            authority_row = authority_rows.get(owner.owner_id)
            if key is None:
                raise Phase3TrainingShuffleReportError("H4-shuffled owner key is missing")
            if authority_row is None:
                raise Phase3TrainingShuffleReportError(
                    "H4-shuffled owner authority row is missing"
                )
            expected_identity = _model_identity_sha256(
                owner,
                prepared,
                model_state_sha256=key.model_state_sha256,
            )
            if key.model_identity_sha256 != expected_identity:
                raise Phase3TrainingShuffleReportError(
                    "training permutation map is not folded into model identity"
                )
            if (
                key.plan_id != plan.plan_id
                or key.protocol_sha256 != plan.protocol_sha256
                or key.evidence_lock_sha256 != lock.evidence_lock_sha256
                or key.view_id != view.view_id
                or key.owner_id != owner.owner_id
                or key.condition_id != owner.condition_id
                or key.fold_id != owner.fold_id
                or key.heldout_family != owner.heldout_family
                or key.replicate != owner.replicate
                or key.training_tuple_id != owner.training_tuple_id
                or key.model_seed != owner.model_seed
                or key.evidence_payload_sha256 != prepared.evidence_payload_sha256
                or key.evidence_payload_bytes != prepared.evidence_payload_bytes
                or key.preparation_git_commit_sha
                != _authority_field(model_authority, "preparation_git_commit_sha")
                or key.preparation_provenance_sha256
                != _authority_field(model_authority, "preparation_provenance_sha256")
            ):
                raise Phase3TrainingShuffleReportError("model key authority lineage differs from prepared view")
            identity_values.append(key.model_identity_sha256)
            key_ids.append(key.key_id)
            artifact_ids.append(_row_field(authority_row, "artifact_id"))
            cost_ids.append(_row_field(authority_row, "cost_id"))
            manifest_sha256s.append(_row_field(authority_row, "manifest_sha256"))
            if key.key_id != _row_field(authority_row, "key_id"):
                raise Phase3TrainingShuffleReportError(
                    "model key identity differs from published authority"
                )
        rows.append(
            {
                "plan_id": plan.plan_id,
                "protocol_sha256": plan.protocol_sha256,
                "evidence_lock_sha256": lock.evidence_lock_sha256,
                "view_id": view.view_id,
                "condition_id": view.condition_id,
                "fold_id": view.fold_id,
                "heldout_family": view.heldout_family,
                "replicate": view.replicate,
                "evidence_payload_sha256": prepared.evidence_payload_sha256,
                "evidence_payload_bytes": prepared.evidence_payload_bytes,
                "representation_identity_sha256": prepared.representation_identity_sha256,
                "model_owner_ids": [owner.owner_id for owner in view_owners],
                "model_key_ids": key_ids,
                "model_artifact_ids": artifact_ids,
                "model_cost_ids": cost_ids,
                "model_manifest_sha256s": manifest_sha256s,
                "model_identity_sha256s": identity_values,
                "training_permutation_map_sha256": shuffle.permutation_map_sha256,
                "eligible_windows": int(shuffle.eligible_windows),
                "map_nonidentity_windows": int(shuffle.map_nonidentity_windows),
                "effective_tensor_changed_windows": int(shuffle.effective_tensor_changed_windows),
                "duplicate_vector_no_effect_windows": int(shuffle.duplicate_vector_no_effect_windows),
                "unchanged_short_windows": int(shuffle.unchanged_short_windows),
                "effective_change_fraction": float(shuffle.effective_change_fraction),
                "claim_eligible": bool(shuffle.claim_eligible),
            }
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "scope": "known-development-only",
        "development_only": True,
        "final_family_access": False,
        "outcomes_included": False,
        "search_included": False,
        "model_authority_sha256": _authority_field(
            model_authority, "authority_sha256"
        ),
        "artifact_store_id": _authority_field(model_authority, "artifact_store_id"),
        "counts": {"families": 6, "replicates": 5, "views": EXPECTED_VIEWS, "owners": EXPECTED_OWNERS, "owners_per_view": OWNERS_PER_VIEW},
        "views": rows,
    }
    body["report_sha256"] = _digest(body)
    canonical = canonical_json_bytes(body)
    _validate_body(body)
    return Phase3TrainingShuffleReport(body=body, canonical_bytes=canonical, report_sha256=body["report_sha256"])


def _open_parent(path: Path) -> int:
    try:
        return secure_fs.open_directory_chain(path.parent)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Phase3TrainingShuffleReportError("report parent cannot be descriptor-pinned") from exc


def _report_file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise Phase3TrainingShuffleReportError("report target is symlinked or non-regular")
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_stable_report_at(parent_fd: int, name: str) -> bytes:
    """Read one report fd and prove its directory entry stayed identical."""

    try:
        before_path = _report_file_identity(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
        with secure_fs.open_regular_file_at(parent_fd, name) as file_fd:
            before_fd = _report_file_identity(os.fstat(file_fd))
            chunks: list[bytes] = []
            while True:
                chunk = os.read(file_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            content = b"".join(chunks)
            after_fd = _report_file_identity(os.fstat(file_fd))
        after_path = _report_file_identity(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
        if (
            before_path != before_fd
            or before_fd != after_fd
            or after_fd != after_path
            or len(content) != before_fd[3]
        ):
            raise Phase3TrainingShuffleReportError("report target changed during read")
        return content
    except Phase3TrainingShuffleReportError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Phase3TrainingShuffleReportError("cannot read training shuffle report safely") from exc


def save_phase3_training_shuffle_report(path: str | os.PathLike[str], report: Phase3TrainingShuffleReport) -> None:
    """Publish once; identical repeats are idempotent and conflicts fail closed."""

    if not isinstance(report, Phase3TrainingShuffleReport):
        raise Phase3TrainingShuffleReportError("report is not typed")
    _validate_body(report.body)
    if report.canonical_bytes != canonical_json_bytes(report.body):
        raise Phase3TrainingShuffleReportError("report canonical bytes drifted")
    target = Path(path).absolute()
    parent_fd = _open_parent(target)
    parent_identity = secure_fs.directory_identity(parent_fd)
    temporary = f".{target.name}.{uuid.uuid4().hex}.tmp"
    fd: int | None = None
    try:
        try:
            os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
            observed = _read_stable_report_at(parent_fd, target.name)
            if observed != report.canonical_bytes:
                raise Phase3TrainingShuffleReportError("existing training shuffle report conflicts")
            return
        except FileNotFoundError:
            pass
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
        remaining = memoryview(report.canonical_bytes)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("short report write")
            remaining = remaining[written:]
        os.fsync(fd)
        os.close(fd)
        fd = None
        try:
            os.link(
                temporary,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            observed = _read_stable_report_at(parent_fd, target.name)
            if observed != report.canonical_bytes:
                raise Phase3TrainingShuffleReportError(
                    "racing training shuffle report conflicts"
                ) from None
        os.fsync(parent_fd)
    except Phase3TrainingShuffleReportError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Phase3TrainingShuffleReportError("cannot atomically save training shuffle report") from exc
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)
        current_parent = _open_parent(target)
        try:
            current_identity = secure_fs.directory_identity(current_parent)
        finally:
            os.close(current_parent)
        if current_identity != parent_identity:
            raise Phase3TrainingShuffleReportError(
                "report parent identity changed during publication"
            )


def load_phase3_training_shuffle_report(path: str | os.PathLike[str]) -> Phase3TrainingShuffleReport:
    """Load and fully validate a canonical report through a pinned parent fd."""

    target = Path(path).absolute()
    parent_fd = _open_parent(target)
    parent_identity = secure_fs.directory_identity(parent_fd)
    try:
        try:
            content = _read_stable_report_at(parent_fd, target.name)
        except Phase3TrainingShuffleReportError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise Phase3TrainingShuffleReportError("cannot read training shuffle report safely") from exc
        try:
            raw = json.loads(content)
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise Phase3TrainingShuffleReportError("training shuffle report JSON is invalid") from exc
        if not isinstance(raw, dict) or canonical_json_bytes(raw) != content:
            raise Phase3TrainingShuffleReportError("training shuffle report is not canonical JSON")
        _validate_body(raw)
        canonical = canonical_json_bytes(raw)
        return Phase3TrainingShuffleReport(body=raw, canonical_bytes=canonical, report_sha256=raw["report_sha256"])
    finally:
        os.close(parent_fd)
        current_parent = _open_parent(target)
        try:
            current_identity = secure_fs.directory_identity(current_parent)
        finally:
            os.close(current_parent)
        if current_identity != parent_identity:
            raise Phase3TrainingShuffleReportError(
                "report parent identity changed during read"
            )


__all__ = [
    "EXPECTED_OWNERS",
    "EXPECTED_VIEWS",
    "Phase3TrainingShuffleReport",
    "Phase3TrainingShuffleReportError",
    "build_phase3_training_shuffle_report",
    "load_phase3_training_shuffle_report",
    "save_phase3_training_shuffle_report",
]
