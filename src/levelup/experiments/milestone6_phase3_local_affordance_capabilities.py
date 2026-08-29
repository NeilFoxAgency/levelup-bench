"""Identity-stripping consumer capabilities for local-affordance raw evidence.

Authority snapshots contain the complete 240-artifact development universe and
must never enter learner code.  This module is the one-way boundary which
selects either an exact 40-task LOFO training view or one exact held-out planned
unit, validates the binding, and retains only sealed identity-free evidence.
It exposes no filesystem, lookup, enumeration, search, evaluator, or execution
surface.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from levelup.experiments.milestone6_phase3_local_affordance_evidence import (
    RawProbeTransitionRecord,
)
from levelup.experiments.milestone6_phase3_local_affordance_raw_authority import (
    FROZEN_LOCAL_AFFORDANCE_PROTOCOL_SHA256,
    PersistedRawProbeArtifact,
    RawProbeAuthoritySnapshot,
    require_raw_probe_authority_snapshot,
)
from levelup.experiments.milestone6_phase3_local_affordance_raw_store import (
    FAMILY_ORDER,
    HeldoutProbeBinding,
    TrainingFoldManifest,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import PlannedUnit, unit_id_for
from levelup.learning.state_conditioned import (
    AffordanceTable,
    IndexedProbeRow,
    ObservableState,
    ObservedTransition,
    TaskLocalAffordanceEvidence,
    TaskProbeRows,
    bind_task_local_affordance_evidence,
)

_CAPABILITY_TOKEN = object()
_LOCAL_CONDITION_IDS = (
    "B2-global-listwise-optimum",
    "S-state-availability-listwise-optimum",
    "P-state-availability-alias-pooled-outcome-listwise-optimum",
    "L-state-availability-local-outcome-listwise-optimum",
)
_LOCAL_TUPLE_IDS = (
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


class LocalAffordanceCapabilityError(ValueError):
    """Raised when an evidence view or consumer capability fails closed."""


def _state(record: Any) -> ObservableState:
    return ObservableState(
        progress_fraction=record.progress_fraction,
        remaining_fraction=record.remaining_fraction,
        elapsed_per_target=record.elapsed_per_target,
        resource_fraction=record.resource_fraction,
        pressure_fraction=record.pressure_fraction,
        available_aliases=record.available_aliases,
    )


def _transition(row: RawProbeTransitionRecord) -> ObservedTransition:
    return ObservedTransition(
        before=_state(row.before),
        action_alias=row.action_alias,
        after=_state(row.after),
        completed=row.completed,
    )


def _identity_free_evidence(
    artifact: PersistedRawProbeArtifact,
) -> TaskLocalAffordanceEvidence:
    try:
        validated = PersistedRawProbeArtifact.model_validate(artifact.model_dump(mode="json"))
        rows = TaskProbeRows(
            tuple(IndexedProbeRow(row.probe_index, _transition(row)) for row in validated.body.rows)
        )
        table = AffordanceTable(
            features={
                alias: tuple(values) for alias, values in validated.affordances.features.items()
            },
            sample_counts=dict(validated.affordances.sample_counts),
        )
        return bind_task_local_affordance_evidence(rows, table)
    except (TypeError, ValueError) as exc:
        raise LocalAffordanceCapabilityError(
            "persisted raw artifact cannot become identity-free evidence"
        ) from exc


def _parse_artifacts(
    snapshot: RawProbeAuthoritySnapshot,
) -> dict[str, PersistedRawProbeArtifact]:
    artifacts: dict[str, PersistedRawProbeArtifact] = {}
    try:
        for record in snapshot.artifact_files:
            artifact = PersistedRawProbeArtifact.model_validate_json(
                record.snapshot.canonical_bytes
            )
            if record.name != f"{artifact.manifest.artifact_id}.json":
                raise LocalAffordanceCapabilityError(
                    "snapshot artifact filename differs from its identity"
                )
            if artifact.manifest.artifact_id in artifacts:
                raise LocalAffordanceCapabilityError("snapshot artifact identities are duplicated")
            artifacts[artifact.manifest.artifact_id] = artifact
    except LocalAffordanceCapabilityError:
        raise
    except (TypeError, ValueError) as exc:
        raise LocalAffordanceCapabilityError("snapshot artifact is invalid") from exc
    if len(artifacts) != 240:
        raise LocalAffordanceCapabilityError("snapshot artifact matrix is incomplete")
    return artifacts


def _training_fold(
    snapshot: RawProbeAuthoritySnapshot,
    *,
    fold_id: str,
    replicate: int,
) -> TrainingFoldManifest:
    name = f"{fold_id}.r{replicate}.json"
    records = tuple(record for record in snapshot.training_fold_files if record.name == name)
    if len(records) != 1:
        raise LocalAffordanceCapabilityError("training fold is missing or duplicated")
    try:
        fold = TrainingFoldManifest.model_validate_json(records[0].snapshot.canonical_bytes)
    except (TypeError, ValueError) as exc:
        raise LocalAffordanceCapabilityError("training fold is invalid") from exc
    if fold.fold_id != fold_id or fold.replicate != replicate:
        raise LocalAffordanceCapabilityError("training fold identity differs from request")
    return fold


def _heldout_binding(
    snapshot: RawProbeAuthoritySnapshot,
    planned: PlannedUnit,
) -> HeldoutProbeBinding:
    key = planned.key
    name = f"{key.family_id}.r{key.replicate}.task-{key.task_index}.json"
    records = tuple(record for record in snapshot.heldout_binding_files if record.name == name)
    if len(records) != 1:
        raise LocalAffordanceCapabilityError("heldout binding is missing or duplicated")
    try:
        binding = HeldoutProbeBinding.model_validate_json(records[0].snapshot.canonical_bytes)
    except (TypeError, ValueError) as exc:
        raise LocalAffordanceCapabilityError("heldout binding is invalid") from exc
    return binding


def _require_local_planned_unit(
    planned: PlannedUnit,
    binding: HeldoutProbeBinding,
) -> None:
    key = planned.key
    try:
        condition_id, tuple_id = key.condition_id.rsplit("--", 1)
    except ValueError as exc:
        raise LocalAffordanceCapabilityError(
            "planned condition is not a local-affordance variant"
        ) from exc
    if condition_id not in _LOCAL_CONDITION_IDS or tuple_id not in _LOCAL_TUPLE_IDS:
        raise LocalAffordanceCapabilityError(
            "planned condition is outside the frozen local-affordance matrix"
        )
    if planned.unit_id != unit_id_for(key):
        raise LocalAffordanceCapabilityError("planned unit self-identity drifted")
    exposure = hashlib.sha256(
        canonical_json_bytes(
            {
                "protocol_sha256": FROZEN_LOCAL_AFFORDANCE_PROTOCOL_SHA256,
                "condition_id": condition_id,
                "tuple_id": tuple_id,
                "learner_visible": "optimum_only_development_training",
            }
        )
    ).hexdigest()
    if planned.exposure_manifest_sha256 != exposure:
        raise LocalAffordanceCapabilityError(
            "planned exposure differs from frozen local-affordance authority"
        )
    reference = binding.task_reference
    family_offset = FAMILY_ORDER.index(key.family_id) * 10_000
    replicate_offset = key.replicate * 100_000
    expected_seeds = (
        6_100_000 + family_offset + replicate_offset,
        reference.key.environment_seed,
        6_200_000 + family_offset + replicate_offset + key.task_index,
        6_300_000 + family_offset + replicate_offset + key.task_index,
        6_400_000 + family_offset + replicate_offset,
    )
    observed_seeds = (
        planned.seeds.model_seed,
        planned.seeds.environment_seed,
        planned.seeds.probe_seed,
        planned.seeds.search_seed,
        planned.seeds.data_order_seed,
    )
    if observed_seeds != expected_seeds:
        raise LocalAffordanceCapabilityError(
            "planned seeds differ from frozen local-affordance authority"
        )


@dataclass(frozen=True, slots=True)
class _TrainingCapabilitySeal:
    task_ids: tuple[str, ...]
    evidence: tuple[TaskLocalAffordanceEvidence, ...]
    authority_content_sha256: str
    token: object


@dataclass(frozen=True, slots=True, init=False, repr=False)
class TrainingFoldProbeCapability:
    """Opaque reusable authorization for exactly one 40-task LOFO view."""

    _task_ids: tuple[str, ...]
    _evidence: tuple[TaskLocalAffordanceEvidence, ...]
    _seal: _TrainingCapabilitySeal
    _token: object

    def __init__(
        self,
        *,
        task_ids: tuple[str, ...],
        evidence: tuple[TaskLocalAffordanceEvidence, ...],
        authority_content_sha256: str,
        _token: object | None = None,
    ) -> None:
        if _token is not _CAPABILITY_TOKEN:
            raise LocalAffordanceCapabilityError(
                "training capabilities require the authority issuer"
            )
        task_ids = tuple(task_ids)
        evidence = tuple(evidence)
        if len(task_ids) != 40 or len(set(task_ids)) != 40 or len(evidence) != 40:
            raise LocalAffordanceCapabilityError("training capability matrix is not exact")
        seal = _TrainingCapabilitySeal(
            task_ids=task_ids,
            evidence=evidence,
            authority_content_sha256=authority_content_sha256,
            token=_CAPABILITY_TOKEN,
        )
        object.__setattr__(self, "_task_ids", task_ids)
        object.__setattr__(self, "_evidence", evidence)
        object.__setattr__(self, "_seal", seal)
        object.__setattr__(self, "_token", _CAPABILITY_TOKEN)

    def _require_sealed(self) -> None:
        seal = getattr(self, "_seal", None)
        if (
            type(self) is not TrainingFoldProbeCapability
            or self._token is not _CAPABILITY_TOKEN
            or type(seal) is not _TrainingCapabilitySeal
            or seal.token is not _CAPABILITY_TOKEN
            or self._task_ids is not seal.task_ids
            or self._evidence is not seal.evidence
        ):
            raise LocalAffordanceCapabilityError("training capability is forged or rebound")

    def consume_for(
        self,
        ordered_training_task_ids: tuple[str, ...],
    ) -> tuple[TaskLocalAffordanceEvidence, ...]:
        """Return only identity-free evidence after exact private order matching."""

        self._require_sealed()
        if (
            type(ordered_training_task_ids) is not tuple
            or ordered_training_task_ids != self._task_ids
        ):
            raise LocalAffordanceCapabilityError("training task order differs from capability")
        return self._evidence

    def __copy__(self) -> "TrainingFoldProbeCapability":
        self._require_sealed()
        return self

    def __deepcopy__(self, _memo: dict[int, object]) -> "TrainingFoldProbeCapability":
        self._require_sealed()
        return self


@dataclass(frozen=True, slots=True)
class _HeldoutCapabilitySeal:
    planned: PlannedUnit
    evidence: TaskLocalAffordanceEvidence
    authority_content_sha256: str
    token: object


@dataclass(frozen=True, slots=True, init=False, repr=False)
class HeldoutTaskProbeCapability:
    """Opaque authorization for exactly one planned held-out development unit."""

    _planned: PlannedUnit
    _evidence: TaskLocalAffordanceEvidence
    _seal: _HeldoutCapabilitySeal
    _token: object

    def __init__(
        self,
        *,
        planned: PlannedUnit,
        evidence: TaskLocalAffordanceEvidence,
        authority_content_sha256: str,
        _token: object | None = None,
    ) -> None:
        if _token is not _CAPABILITY_TOKEN:
            raise LocalAffordanceCapabilityError(
                "heldout capabilities require the authority issuer"
            )
        seal = _HeldoutCapabilitySeal(
            planned=planned,
            evidence=evidence,
            authority_content_sha256=authority_content_sha256,
            token=_CAPABILITY_TOKEN,
        )
        object.__setattr__(self, "_planned", planned)
        object.__setattr__(self, "_evidence", evidence)
        object.__setattr__(self, "_seal", seal)
        object.__setattr__(self, "_token", _CAPABILITY_TOKEN)

    def _require_sealed(self) -> None:
        seal = getattr(self, "_seal", None)
        if (
            type(self) is not HeldoutTaskProbeCapability
            or self._token is not _CAPABILITY_TOKEN
            or type(seal) is not _HeldoutCapabilitySeal
            or seal.token is not _CAPABILITY_TOKEN
            or self._planned is not seal.planned
            or self._evidence is not seal.evidence
        ):
            raise LocalAffordanceCapabilityError("heldout capability is forged or rebound")

    def consume_for(self, planned: PlannedUnit) -> TaskLocalAffordanceEvidence:
        """Return one identity-free task evidence only for the exact planned unit."""

        self._require_sealed()
        if type(planned) is not PlannedUnit or planned != self._planned:
            raise LocalAffordanceCapabilityError("planned unit differs from capability")
        return self._evidence

    def __copy__(self) -> "HeldoutTaskProbeCapability":
        self._require_sealed()
        return self

    def __deepcopy__(self, _memo: dict[int, object]) -> "HeldoutTaskProbeCapability":
        self._require_sealed()
        return self


def issue_training_fold_probe_capability(
    snapshot: RawProbeAuthoritySnapshot,
    *,
    fold_id: str,
    replicate: int,
) -> TrainingFoldProbeCapability:
    """Select one exact LOFO fold and discard every artifact identity from output."""

    try:
        require_raw_probe_authority_snapshot(snapshot)
    except ValueError as exc:
        raise LocalAffordanceCapabilityError(
            "training capability requires a validator-issued authority"
        ) from exc
    if fold_id not in FAMILY_ORDER or type(replicate) is not int or replicate not in range(5):
        raise LocalAffordanceCapabilityError("training fold request is outside frozen matrix")
    if snapshot.manifest.execution_authorized is not False:
        raise LocalAffordanceCapabilityError("raw evidence authority cannot authorize execution")
    fold = _training_fold(snapshot, fold_id=fold_id, replicate=replicate)
    artifacts = _parse_artifacts(snapshot)
    selected: list[TaskLocalAffordanceEvidence] = []
    task_ids: list[str] = []
    for reference in fold.task_references:
        artifact = artifacts.get(reference.artifact_id)
        if (
            artifact is None
            or artifact.key != reference.key
            or artifact.manifest.key_id != reference.key_id
        ):
            raise LocalAffordanceCapabilityError(
                "training reference does not resolve to exact artifact"
            )
        selected.append(_identity_free_evidence(artifact))
        task_ids.append(reference.task_id)
    return TrainingFoldProbeCapability(
        task_ids=tuple(task_ids),
        evidence=tuple(selected),
        authority_content_sha256=snapshot.authority_content_sha256,
        _token=_CAPABILITY_TOKEN,
    )


def issue_heldout_task_probe_capability(
    snapshot: RawProbeAuthoritySnapshot,
    planned: PlannedUnit,
) -> HeldoutTaskProbeCapability:
    """Select one exact held-out task and discard its identity before consumption."""

    try:
        require_raw_probe_authority_snapshot(snapshot)
    except ValueError as exc:
        raise LocalAffordanceCapabilityError(
            "heldout capability requires a validator-issued authority"
        ) from exc
    if type(planned) is not PlannedUnit or planned.key.phase != "validation":
        raise LocalAffordanceCapabilityError(
            "heldout capabilities require an exact validation planned unit"
        )
    if planned.key.family_id not in FAMILY_ORDER or planned.key.replicate not in range(5):
        raise LocalAffordanceCapabilityError("planned unit is outside frozen matrix")
    binding = _heldout_binding(snapshot, planned)
    _require_local_planned_unit(planned, binding)
    reference = binding.task_reference
    if (
        binding.fold_id != planned.key.family_id
        or reference.task_id != planned.key.task_id
        or reference.task_index != planned.key.task_index
        or reference.replicate != planned.key.replicate
        or reference.key.environment_seed != planned.seeds.environment_seed
        or reference.key.probe_seed != planned.seeds.probe_seed
    ):
        raise LocalAffordanceCapabilityError(
            "planned unit differs from heldout raw-evidence binding"
        )
    artifacts = _parse_artifacts(snapshot)
    artifact = artifacts.get(reference.artifact_id)
    if (
        artifact is None
        or artifact.key != reference.key
        or artifact.manifest.key_id != reference.key_id
    ):
        raise LocalAffordanceCapabilityError("heldout binding does not resolve to exact artifact")
    return HeldoutTaskProbeCapability(
        planned=planned,
        evidence=_identity_free_evidence(artifact),
        authority_content_sha256=snapshot.authority_content_sha256,
        _token=_CAPABILITY_TOKEN,
    )


__all__ = [
    "HeldoutTaskProbeCapability",
    "LocalAffordanceCapabilityError",
    "TrainingFoldProbeCapability",
    "issue_heldout_task_probe_capability",
    "issue_training_fold_probe_capability",
]
