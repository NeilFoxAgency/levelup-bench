"""Load and validate shared model artifacts for one screening unit.

The screening executor calls this boundary once per atomic unit.  It is deliberately
small: A0/A1 have no learner-visible training artifacts, while a learned condition
receives exactly the evidence, view, and temperature-independent model that its
committed shared-artifact plan authorizes.  Model objects may be cached across the
three temperature variants, but never across a fold, tuple, or replicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from levelup.experiments.milestone6_phase2_screening import (
    FIXED_CONDITIONS,
    base_condition_id,
    candidate_for_condition,
)
from levelup.experiments.milestone6_phase2_screening_models import (
    _expected_model_id,
    _model_factory,
)
from levelup.experiments.milestone6_phase2_screening_runtime import ScreeningRuntimeFold
from levelup.experiments.runner.config import ConditionSpec
from levelup.experiments.runner.records import PlannedUnit, SharedArtifactReference
from levelup.experiments.runner.storage import ArtifactValidationError
from levelup.experiments.runner.training_artifacts import (
    TrainingArtifactManifest,
    TrainingReportMetadata,
    load_training_key_index,
    load_training_model,
)

ModelIdentity = tuple[str, str, str, int]
ModelSlotIdentity = tuple[str, str, int]


@dataclass(frozen=True, slots=True)
class PreparedUnitModel:
    """The model and exact shared-artifact lineage prepared for one unit."""

    model: torch.nn.Module
    report: TrainingReportMetadata
    references: tuple[SharedArtifactReference, ...]
    identity: ModelIdentity


class ScreeningModelCache:
    """In-memory cache scoped by the complete scientific model identity."""

    def __init__(self) -> None:
        self._entries: dict[
            ModelIdentity, tuple[torch.nn.Module, TrainingArtifactManifest]
        ] = {}

    def get(
        self, identity: ModelIdentity
    ) -> tuple[torch.nn.Module, TrainingArtifactManifest] | None:
        return self._entries.get(identity)

    def put(
        self,
        identity: ModelIdentity,
        model: torch.nn.Module,
        manifest: TrainingArtifactManifest,
    ) -> None:
        existing = self._entries.get(identity)
        if existing is not None and (
            existing[0] is not model or existing[1] != manifest
        ):
            raise ArtifactValidationError("screening model cache identity collision")
        self._entries[identity] = (model, manifest)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def _fail(message: str) -> None:
    raise ArtifactValidationError(message)


def _same_value(left: Any, right: Any) -> bool:
    """Compare JSON-valued candidate parameters without numeric coercion surprises."""

    return type(left) is type(right) and left == right


def _resolve_learned_condition(
    fold: ScreeningRuntimeFold,
    condition: ConditionSpec,
) -> tuple[str, dict[str, Any], str]:
    base = base_condition_id(condition.condition_id)
    if base is None:
        _fail("screening learned model requested for a non-learned condition")
    candidate = candidate_for_condition(condition.condition_id)
    if candidate is None:
        _fail("screening learned condition has no frozen candidate tuple")
    parameters = condition.parameters
    expected = {
        "base_condition_id": base,
        "candidate_tuple_id": candidate["tuple_id"],
        "training_tuple_id": candidate["training_tuple_id"],
        "learning_rate": candidate["learning_rate"],
        "training_epochs": candidate["training_epochs"],
        "search_temperature": candidate["search_temperature"],
    }
    if any(
        name not in parameters or not _same_value(parameters[name], value)
        for name, value in expected.items()
    ):
        _fail("screening candidate tuple and condition parameters disagree")
    training_tuple_id = str(candidate["training_tuple_id"])
    if condition.condition_id not in {
        item.condition_id for item in fold.config.conditions
    }:
        _fail("screening condition is not present in the fold configuration")
    return base, candidate, training_tuple_id


def _planned_shared(
    fold: ScreeningRuntimeFold,
    planned: PlannedUnit,
    condition: ConditionSpec,
    *,
    kind: str,
    key_id: str,
    owner_group_id: str,
) -> Any:
    matches = tuple(
        artifact
        for artifact in fold.shared_plan.artifacts
        if artifact.kind == kind
        and artifact.key_id == key_id
        and planned.unit_id in artifact.consumer_unit_ids
        and condition.condition_id in artifact.consumer_condition_ids
        and artifact.owner_family_id == fold.family_id
        and artifact.owner_replicate == planned.key.replicate
        and artifact.owner_fold_id == str(fold.config.parameters["fold_id"])
        and artifact.owner_group_id == owner_group_id
    )
    if len(matches) != 1:
        _fail("screening shared-artifact authorization is not exact")
    artifact = matches[0]
    if artifact.consumer_phase != planned.key.phase:
        _fail("screening shared-artifact consumer phase drifted")
    return artifact


def _reference(
    *,
    kind: str,
    key_id: str,
    artifact_id: str,
    cost_id: str,
) -> SharedArtifactReference:
    try:
        return SharedArtifactReference(
            kind=kind,
            key_id=key_id,
            artifact_id=artifact_id,
            cost_id=cost_id,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError("screening shared-artifact reference is invalid") from exc


def _training_identity(
    fold: ScreeningRuntimeFold,
    planned: PlannedUnit,
    base: str,
    training_tuple_id: str,
) -> tuple[ModelSlotIdentity, Any, TrainingArtifactManifest, Any]:
    identity_key = (base, training_tuple_id, planned.key.replicate)
    key = fold.model_keys.models.get(identity_key)
    manifest = fold.models.manifests.get(identity_key)
    cost = fold.models.costs.get(identity_key)
    if key is None or manifest is None or cost is None:
        _fail("screening model key/manifest/cost inventory is incomplete")
    if (
        key.condition_id != base
        or key.training_tuple_id != training_tuple_id
        or key.replicate != planned.key.replicate
        or manifest.key != key
        or manifest.artifact_id != cost.artifact_id
        or cost.key_id != key.key_id
        or cost.key != key.model_dump(mode="json")
    ):
        _fail("screening model key/manifest/cost identity drifted")
    return identity_key, key, manifest, cost


def prepare_unit_model(
    fold: ScreeningRuntimeFold,
    planned: PlannedUnit,
    condition: ConditionSpec,
    cache: ScreeningModelCache | None = None,
) -> PreparedUnitModel | None:
    """Prepare the exact shared model and lineage for one planned unit."""

    if condition.condition_id in FIXED_CONDITIONS:
        return None
    if condition.condition_id.startswith("A"):
        _fail("unknown fixed screening condition")
    if planned.key.condition_id != condition.condition_id:
        _fail("screening planned unit and condition disagree")
    if planned.key.family_id != fold.family_id:
        _fail("screening planned unit belongs to another fold")
    if planned.key.phase != "validation":
        _fail("screening model preparation requires a validation unit")

    base, candidate, training_tuple_id = _resolve_learned_condition(fold, condition)
    model_slot_identity, model_key, expected_manifest, model_cost = _training_identity(
        fold, planned, base, training_tuple_id
    )
    model_identity: ModelIdentity = (
        str(fold.store.run_id),
        model_slot_identity[0],
        model_slot_identity[1],
        model_slot_identity[2],
    )
    if model_key.fold_id != str(fold.config.parameters["fold_id"]):
        _fail("screening model key belongs to another fold")
    if model_key.heldout_family_id != fold.family_id:
        _fail("screening model key belongs to another family")

    evidence_key = fold.data_keys.evidence.get(planned.key.replicate)
    evidence_manifest = fold.data.manifests.evidence.get(planned.key.replicate)
    evidence_cost_id = fold.data.evidence_cost_ids.get(planned.key.replicate)
    view_identity = (base, planned.key.replicate)
    view_key = fold.data_keys.views.get(view_identity)
    view_manifest = fold.data.manifests.views.get(view_identity)
    view_cost_id = fold.data.view_cost_ids.get(view_identity)
    if (
        evidence_key is None
        or evidence_manifest is None
        or evidence_cost_id is None
        or view_key is None
        or view_manifest is None
        or view_cost_id is None
    ):
        _fail("screening data key/manifest/cost inventory is incomplete")
    fold_id = str(fold.config.parameters["fold_id"])
    if (
        evidence_manifest.key != evidence_key
        or evidence_key.fold_id != fold_id
        or evidence_key.heldout_family_id != fold.family_id
        or evidence_key.replicate != planned.key.replicate
    ):
        _fail("screening evidence key/manifest identity drifted")
    if (
        view_manifest.key != view_key
        or view_key.fold_id != fold_id
        or view_key.heldout_family_id != fold.family_id
        or view_key.condition_id != base
        or view_key.replicate != planned.key.replicate
    ):
        _fail("screening view key/manifest identity drifted")
    if expected_manifest.model_id != _expected_model_id(base):
        _fail("screening model ID differs from the canonical condition model")

    _planned_shared(
        fold,
        planned,
        condition,
        kind="training_data_evidence",
        key_id=evidence_key.key_id,
        owner_group_id="canonical-evidence",
    )
    _planned_shared(
        fold,
        planned,
        condition,
        kind="training_data_view",
        key_id=view_key.key_id,
        owner_group_id=base,
    )
    _planned_shared(
        fold,
        planned,
        condition,
        kind="training_artifact",
        key_id=model_key.key_id,
        owner_group_id=base,
    )

    # Resolve artifact IDs from the typed manifests and cost inventories.  The
    # RunStore call below independently reopens and validates all three files.
    evidence_reference = _reference(
        kind="training_data_evidence",
        key_id=evidence_key.key_id,
        artifact_id=evidence_manifest.evidence_id,
        cost_id=evidence_cost_id,
    )
    view_reference = _reference(
        kind="training_data_view",
        key_id=view_key.key_id,
        artifact_id=view_manifest.artifact_id,
        cost_id=view_cost_id,
    )
    model_reference = _reference(
        kind="training_artifact",
        key_id=model_key.key_id,
        artifact_id=expected_manifest.artifact_id,
        cost_id=model_cost.cost_id,
    )
    references = (evidence_reference, view_reference, model_reference)
    fold.store.validate_shared_reference_set(planned, references)

    cached = None if cache is None else cache.get(model_identity)
    if cached is None:
        index = load_training_key_index(fold.store.run_dir, model_key)
        if index.artifact_id != expected_manifest.artifact_id:
            _fail("screening model key index substituted an artifact")
        model, manifest = load_training_model(
            fold.store.run_dir,
            index.artifact_id,
            expected_key=model_key,
            model_factory=_model_factory,
        )
        if manifest != expected_manifest or model.training:
            _fail("screening model manifest or evaluation mode drifted")
        if cache is not None:
            cache.put(model_identity, model, manifest)
    else:
        model, cached_manifest = cached
        if model.training or cached_manifest != expected_manifest:
            _fail("screening cached model is not an exact evaluation artifact")

    return PreparedUnitModel(model, expected_manifest.report, references, model_identity)


__all__ = ["PreparedUnitModel", "ScreeningModelCache", "prepare_unit_model"]
