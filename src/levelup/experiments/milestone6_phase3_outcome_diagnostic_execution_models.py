"""Descriptor-pinned loading of outcome-diagnostic execution models.

This module is the narrow execution boundary for the development-only outcome
diagnostic.  Readiness has already checked the complete 240-owner store before
this code is called.  The cache retains those validated identities and the
typed payloads, while each load still reads the selected state through the
already-held model-store descriptors and rechecks the complete store lease.

No evidence, result, evaluator, search, oracle, or final-family resource is
opened here.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterator

import numpy as np
import torch

from levelup.experiments.milestone6_phase3_outcome_diagnostic_generation import (
    AuthorizedOutcomeGenerationModel,
    authorize_outcome_generation_model,
    model_state_sha256,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
    AuthorizedOutcomeModelArtifact,
    OutcomeDiagnosticArtifactRow,
    OutcomeDiagnosticModelArtifactAuthority,
    OutcomeDiagnosticModelArtifactError,
    OutcomeDiagnosticModelArtifactRecord,
    OutcomeTrainingAccounting,
    PinnedOutcomeModelState,
    authorize_outcome_model_artifact_from_compact_authority,
    inspect_outcome_model_state,
    prepare_outcome_compact_artifact_authorization_inputs,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_store import (
    OutcomeModelStateIndex,
    OutcomeModelStoreError,
    PinnedOutcomeModelStore,
    PinnedOutcomeModelStoreReader,
    load_outcome_model_artifact_payload_at,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    EXPECTED_MODEL_OWNERS,
    EXPECTED_UNITS,
    OutcomePlannedUnit,
    ValidatedOutcomePlan,
)
from levelup.learning.state_conditioned import StateConditionedScorer


class OutcomeDiagnosticExecutionModelError(ValueError):
    """Raised when a model cannot be proven to belong to the pinned authority."""


_CACHE_TOKEN = object()
_LINEAGE_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class OutcomeDiagnosticExecutionLineage:
    """Typed lineage carried with every loaded model for diagnostics/dedup."""

    owner_id: str
    record_id: str
    key_id: str
    model_state_sha256: str
    training_accounting: OutcomeTrainingAccounting
    _token: object

    def __init__(
        self,
        owner_id: str,
        record_id: str,
        key_id: str,
        model_state_sha256: str,
        training_accounting: OutcomeTrainingAccounting,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _LINEAGE_TOKEN:
            raise OutcomeDiagnosticExecutionModelError(
                "outcome execution lineage requires canonical validation"
            )
        if type(training_accounting) is not OutcomeTrainingAccounting:
            raise OutcomeDiagnosticExecutionModelError("training accounting is not typed")
        object.__setattr__(self, "owner_id", owner_id)
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "key_id", key_id)
        object.__setattr__(self, "model_state_sha256", model_state_sha256)
        object.__setattr__(self, "training_accounting", training_accounting)
        object.__setattr__(self, "_token", _LINEAGE_TOKEN)


@dataclass(frozen=True, slots=True, init=False)
class OutcomeDiagnosticExecutionArtifact:
    """One complete validated owner payload retained by the execution cache."""

    row: OutcomeDiagnosticArtifactRow
    record: OutcomeDiagnosticModelArtifactRecord
    index: OutcomeModelStateIndex
    state: PinnedOutcomeModelState
    authorization: AuthorizedOutcomeModelArtifact
    lineage: OutcomeDiagnosticExecutionLineage
    _token: object

    def __init__(
        self,
        row: OutcomeDiagnosticArtifactRow,
        record: OutcomeDiagnosticModelArtifactRecord,
        index: OutcomeModelStateIndex,
        state: PinnedOutcomeModelState,
        authorization: AuthorizedOutcomeModelArtifact,
        lineage: OutcomeDiagnosticExecutionLineage,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _CACHE_TOKEN:
            raise OutcomeDiagnosticExecutionModelError(
                "outcome execution artifacts require canonical cache construction"
            )
        object.__setattr__(self, "row", row)
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "authorization", authorization)
        object.__setattr__(self, "lineage", lineage)
        object.__setattr__(self, "_token", _CACHE_TOKEN)

    @property
    def key(self) -> Any:
        return self.record.key

    @property
    def state_index(self) -> OutcomeModelStateIndex:
        return self.index

    @property
    def training_accounting(self) -> OutcomeTrainingAccounting:
        return self.lineage.training_accounting


@dataclass(frozen=True, slots=True, init=False)
class OutcomeDiagnosticExecutionAuthorityCache:
    """Immutable owner/unit indexes built after complete readiness validation."""

    authority: OutcomeDiagnosticModelArtifactAuthority
    validated_plan: ValidatedOutcomePlan
    lease: Any
    _units_by_id: Mapping[str, OutcomePlannedUnit]
    _artifacts_by_owner_id: Mapping[str, OutcomeDiagnosticExecutionArtifact]
    _artifacts_by_unit_id: Mapping[str, OutcomeDiagnosticExecutionArtifact]
    _token: object

    def __init__(
        self,
        *,
        authority: OutcomeDiagnosticModelArtifactAuthority,
        validated_plan: ValidatedOutcomePlan,
        lease: Any,
        artifacts_by_owner_id: Mapping[str, OutcomeDiagnosticExecutionArtifact],
        _token: object | None = None,
    ) -> None:
        if _token is not _CACHE_TOKEN:
            raise OutcomeDiagnosticExecutionModelError(
                "outcome execution caches require canonical construction"
            )
        if type(authority) is not OutcomeDiagnosticModelArtifactAuthority:
            raise OutcomeDiagnosticExecutionModelError("outcome cache authority is not typed")
        if type(validated_plan) is not ValidatedOutcomePlan:
            raise OutcomeDiagnosticExecutionModelError("outcome cache plan is not typed")
        if authority.authority_sha256 != authority.expected_authority_sha256:
            raise OutcomeDiagnosticExecutionModelError("outcome authority self-hash differs")
        if (
            authority.plan_id != validated_plan.plan.plan_id
            or authority.protocol_sha256 != validated_plan.plan.protocol_sha256
            or validated_plan.plan.final_family_access
        ):
            raise OutcomeDiagnosticExecutionModelError("outcome authority and plan lineage differs")
        if not callable(getattr(lease, "require_active", None)):
            raise OutcomeDiagnosticExecutionModelError("outcome model readiness lease is required")
        owner_ids = tuple(sorted(artifacts_by_owner_id))
        if len(owner_ids) != EXPECTED_MODEL_OWNERS or len(set(owner_ids)) != EXPECTED_MODEL_OWNERS:
            raise OutcomeDiagnosticExecutionModelError("outcome cache requires exactly 240 owners")
        units = {unit.unit_id: unit for unit in validated_plan.plan.units}
        if len(units) != EXPECTED_UNITS:
            raise OutcomeDiagnosticExecutionModelError("outcome cache requires exactly 5760 units")
        if set(owner_ids) != {row.owner_id for row in authority.artifacts}:
            raise OutcomeDiagnosticExecutionModelError("outcome cache owner universe differs")
        by_unit: dict[str, OutcomeDiagnosticExecutionArtifact] = {}
        for unit in validated_plan.plan.units:
            artifact = artifacts_by_owner_id.get(unit.model_owner_id)
            if artifact is None:
                raise OutcomeDiagnosticExecutionModelError("outcome unit owner is absent")
            by_unit[unit.unit_id] = artifact
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "validated_plan", validated_plan)
        object.__setattr__(self, "lease", lease)
        object.__setattr__(self, "_units_by_id", MappingProxyType(units))
        object.__setattr__(self, "_artifacts_by_owner_id", MappingProxyType(dict(artifacts_by_owner_id)))
        object.__setattr__(self, "_artifacts_by_unit_id", MappingProxyType(by_unit))
        object.__setattr__(self, "_token", _CACHE_TOKEN)

    def require_active(self) -> None:
        if self._token is not _CACHE_TOKEN:
            raise OutcomeDiagnosticExecutionModelError("outcome execution cache authority is invalid")
        try:
            self.lease.require_active()
        except Exception as exc:
            if isinstance(exc, OutcomeDiagnosticExecutionModelError):
                raise
            raise OutcomeDiagnosticExecutionModelError(
                "prepared outcome model store lease is no longer active"
            ) from exc

    def resolve_unit(self, planned_unit: OutcomePlannedUnit) -> OutcomePlannedUnit:
        if type(planned_unit) is not OutcomePlannedUnit:
            raise OutcomeDiagnosticExecutionModelError("canonical outcome planned unit is required")
        expected = self._units_by_id.get(planned_unit.unit_id)
        if expected is None or expected != planned_unit:
            raise OutcomeDiagnosticExecutionModelError("planned outcome unit differs from cached plan")
        return expected

    def artifact_for_unit(self, planned_unit: OutcomePlannedUnit) -> OutcomeDiagnosticExecutionArtifact:
        planned = self.resolve_unit(planned_unit)
        try:
            return self._artifacts_by_unit_id[planned.unit_id]
        except KeyError as exc:
            raise OutcomeDiagnosticExecutionModelError("outcome unit payload is absent") from exc

    def artifact_for_owner(self, owner_id: str) -> OutcomeDiagnosticExecutionArtifact:
        try:
            return self._artifacts_by_owner_id[owner_id]
        except KeyError as exc:
            raise OutcomeDiagnosticExecutionModelError("outcome owner payload is absent") from exc


def _reader(store: PinnedOutcomeModelStore | PinnedOutcomeModelStoreReader) -> Any:
    if type(store) is PinnedOutcomeModelStore:
        return store.reader
    if type(store) is PinnedOutcomeModelStoreReader:
        return store
    raise OutcomeDiagnosticExecutionModelError("canonical pinned model store is required")


def _validate_payload(
    row: OutcomeDiagnosticArtifactRow,
    record: OutcomeDiagnosticModelArtifactRecord,
    index: OutcomeModelStateIndex,
    state: PinnedOutcomeModelState,
    authorization: AuthorizedOutcomeModelArtifact,
) -> OutcomeDiagnosticExecutionArtifact:
    key = record.key
    if (
        row.owner_id != key.owner_id
        or row.record_id != record.record_id
        or row.key_id != key.key_id
        or row.model_state_sha256 != key.model_state_sha256
        or index.owner_id != key.owner_id
        or index.record_id != record.record_id
        or index.model_state_sha256 != key.model_state_sha256
    ):
        raise OutcomeDiagnosticExecutionModelError("outcome payload lineage differs from authority")
    try:
        _schema, state_sha = inspect_outcome_model_state(state)
    except (TypeError, ValueError, OutcomeDiagnosticModelArtifactError) as exc:
        raise OutcomeDiagnosticExecutionModelError("outcome model state is not canonical") from exc
    if state_sha != key.model_state_sha256:
        raise OutcomeDiagnosticExecutionModelError("outcome model state hash differs from authority")
    lineage = OutcomeDiagnosticExecutionLineage(
        owner_id=key.owner_id,
        record_id=record.record_id,
        key_id=key.key_id,
        model_state_sha256=key.model_state_sha256,
        training_accounting=key.training_accounting,
        _token=_LINEAGE_TOKEN,
    )
    return OutcomeDiagnosticExecutionArtifact(
        row=row,
        record=record,
        index=index,
        state=state,
        authorization=authorization,
        lineage=lineage,
        _token=_CACHE_TOKEN,
    )


def _build_cache_from_components(
    authority: OutcomeDiagnosticModelArtifactAuthority,
    validated_plan: ValidatedOutcomePlan,
    lease: Any,
    *,
    protocol_snapshot: Any,
    payloads: Mapping[str, tuple[OutcomeDiagnosticModelArtifactRecord, OutcomeModelStateIndex, PinnedOutcomeModelState]] | None = None,
) -> OutcomeDiagnosticExecutionAuthorityCache:
    if type(authority) is not OutcomeDiagnosticModelArtifactAuthority:
        raise OutcomeDiagnosticExecutionModelError("typed outcome model authority is required")
    if type(validated_plan) is not ValidatedOutcomePlan:
        raise OutcomeDiagnosticExecutionModelError("validated outcome plan is required")
    try:
        lease.require_active()
    except Exception as exc:
        raise OutcomeDiagnosticExecutionModelError("active model readiness lease is required") from exc
    rows = {row.owner_id: row for row in authority.artifacts}
    if len(rows) != EXPECTED_MODEL_OWNERS:
        raise OutcomeDiagnosticExecutionModelError("outcome authority owner matrix is incomplete")
    reader = _reader(lease.store)
    authorization_inputs = prepare_outcome_compact_artifact_authorization_inputs(
        validated_plan, protocol_snapshot
    )
    fresh_snapshot = authorization_inputs.snapshot
    artifacts: dict[str, OutcomeDiagnosticExecutionArtifact] = {}
    for owner_id in sorted(rows):
        try:
            loaded = payloads[owner_id] if payloads is not None else load_outcome_model_artifact_payload_at(reader, owner_id)
            record, index, state = loaded
            authorization = authorize_outcome_model_artifact_from_compact_authority(
                record,
                state,
                authority,
                rows[owner_id],
                validated_plan,
                fresh_snapshot,
                _validated_inputs=authorization_inputs,
            )
            artifacts[owner_id] = _validate_payload(
                rows[owner_id], record, index, state, authorization
            )
        except (
            KeyError,
            OutcomeModelStoreError,
            OutcomeDiagnosticModelArtifactError,
            OutcomeDiagnosticExecutionModelError,
        ) as exc:
            raise OutcomeDiagnosticExecutionModelError(
                f"outcome owner payload {owner_id} failed cache validation"
            ) from exc
    owners = {owner.owner_id: owner for owner in validated_plan.plan.model_owners}
    views = {view.view_id: view for view in validated_plan.plan.views}
    for owner_id, artifact in artifacts.items():
        owner = owners.get(owner_id)
        view = views.get(artifact.row.view_id)
        if owner is None or view is None or (
            artifact.row.view_id != owner.view_id
            or artifact.row.condition_id != owner.condition_id
            or artifact.row.heldout_family != owner.heldout_family
            or artifact.row.fold_id != owner.fold_id
            or artifact.row.replicate != owner.replicate
            or artifact.row.training_tuple_id != owner.training_tuple_id
            or artifact.row.model_seed != owner.model_seed
            or artifact.row.data_order_seed != view.data_order_seed
            or artifact.row.feature_mask_sha256 != owner.feature_mask_sha256
            or artifact.row.transformation_sha256 != owner.transformation_sha256
            or artifact.row.representation_sha256 != view.representation_sha256
            or artifact.row.model_identity_sha256 != owner.model_identity_sha256
        ):
            raise OutcomeDiagnosticExecutionModelError(
                "outcome cached artifact row differs from the canonical owner"
            )
    cache = OutcomeDiagnosticExecutionAuthorityCache(
        authority=authority,
        validated_plan=validated_plan,
        lease=lease,
        artifacts_by_owner_id=artifacts,
        _token=_CACHE_TOKEN,
    )
    cache.require_active()
    return cache


def build_outcome_diagnostic_execution_authority_cache(
    authority: OutcomeDiagnosticModelArtifactAuthority | Any,
    validated_plan: ValidatedOutcomePlan | None = None,
    lease: Any | None = None,
    *,
    protocol_snapshot: Any | None = None,
    payloads: Mapping[str, tuple[OutcomeDiagnosticModelArtifactRecord, OutcomeModelStateIndex, PinnedOutcomeModelState]] | None = None,
) -> OutcomeDiagnosticExecutionAuthorityCache:
    """Build the immutable cache from the active readiness lease.

    A readiness snapshot may be supplied as the sole argument after capture;
    in that form its already-built cache is returned.  The explicit form is
    used by readiness itself so cache construction occurs before the snapshot
    is published.
    """

    if validated_plan is None and lease is None:
        snapshot = authority
        from levelup.experiments.milestone6_phase3_outcome_diagnostic_readiness import (
            OutcomeDiagnosticModelReadinessSnapshot,
        )

        if type(snapshot) is not OutcomeDiagnosticModelReadinessSnapshot:
            raise OutcomeDiagnosticExecutionModelError(
                "canonical outcome model readiness snapshot is required"
            )
        cache = getattr(snapshot, "execution_authority_cache", None)
        if type(cache) is not OutcomeDiagnosticExecutionAuthorityCache:
            raise OutcomeDiagnosticExecutionModelError("readiness snapshot has no execution cache")
        cache.require_active()
        return cache
    if validated_plan is None or lease is None:
        raise OutcomeDiagnosticExecutionModelError("validated plan and active lease are required")
    if protocol_snapshot is None:
        protocol_snapshot = getattr(validated_plan, "protocol_snapshot", None)
    if protocol_snapshot is None:
        raise OutcomeDiagnosticExecutionModelError("canonical outcome protocol snapshot is required")
    return _build_cache_from_components(
        authority,
        validated_plan,
        lease,
        protocol_snapshot=protocol_snapshot,
        payloads=payloads,
    )


def _reconstruct_state_model(state: PinnedOutcomeModelState) -> StateConditionedScorer:
    if sys.byteorder != "little":
        raise OutcomeDiagnosticExecutionModelError("little-endian float32 host is required")
    values: dict[str, torch.Tensor] = {}
    for tensor in state.tensors:
        try:
            array = np.frombuffer(tensor.data, dtype="<f4")
            expected = 1
            for dimension in tensor.shape:
                expected *= dimension
            if array.size != expected:
                raise ValueError("tensor byte length differs")
            array = array.reshape(tensor.shape).copy()
            values[tensor.name] = torch.from_numpy(array)
        except (TypeError, ValueError) as exc:
            raise OutcomeDiagnosticExecutionModelError("state tensor is not strict little-endian float32") from exc
    try:
        model = StateConditionedScorer()
        model.load_state_dict(values, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise OutcomeDiagnosticExecutionModelError("StateConditionedScorer state schema differs") from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _validate_loaded_against_cache(
    loaded: tuple[OutcomeDiagnosticModelArtifactRecord, OutcomeModelStateIndex, PinnedOutcomeModelState],
    expected: OutcomeDiagnosticExecutionArtifact,
) -> None:
    record, index, state = loaded
    if record != expected.record or index != expected.index or state != expected.state:
        raise OutcomeDiagnosticExecutionModelError("descriptor-read model payload differs from readiness cache")


@dataclass(frozen=True, slots=True, init=False)
class AuthorizedOutcomeExecutionModel:
    """Loaded generation capability plus outcome-specific lineage."""

    generation: AuthorizedOutcomeGenerationModel
    lineage: OutcomeDiagnosticExecutionLineage
    _active: bool
    _token: object

    def __init__(
        self,
        generation: AuthorizedOutcomeGenerationModel,
        lineage: OutcomeDiagnosticExecutionLineage,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _CACHE_TOKEN:
            raise OutcomeDiagnosticExecutionModelError("authorized execution models require canonical loading")
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "lineage", lineage)
        object.__setattr__(self, "_active", True)
        object.__setattr__(self, "_token", _CACHE_TOKEN)

    @property
    def model(self) -> StateConditionedScorer:
        return self.generation.model

    @property
    def authorized_model(self) -> AuthorizedOutcomeGenerationModel:
        """The public generation capability minted by outcome authorization."""

        return self.generation

    @property
    def generation_model(self) -> AuthorizedOutcomeGenerationModel:
        return self.generation

    @property
    def unit_id(self) -> str:
        return self.generation.unit_id

    @property
    def owner_id(self) -> str:
        return self.lineage.owner_id

    @property
    def record_id(self) -> str:
        return self.lineage.record_id

    @property
    def key_id(self) -> str:
        return self.lineage.key_id

    @property
    def model_state_sha256(self) -> str:
        return self.lineage.model_state_sha256

    @property
    def training_accounting(self) -> OutcomeTrainingAccounting:
        return self.lineage.training_accounting

    def require_active(self) -> "AuthorizedOutcomeExecutionModel":
        if self._token is not _CACHE_TOKEN or not self._active:
            raise OutcomeDiagnosticExecutionModelError(
                "authorized outcome execution model is no longer active"
            )
        return self


@contextmanager
def load_authorized_outcome_model_from_pinned_store(
    snapshot: Any,
    planned_unit: OutcomePlannedUnit,
) -> Iterator[AuthorizedOutcomeExecutionModel]:
    """Load one model using only an active readiness snapshot and held fds."""

    from levelup.experiments.milestone6_phase3_outcome_diagnostic_readiness import (
        OutcomeDiagnosticModelReadinessSnapshot,
    )

    if type(snapshot) is not OutcomeDiagnosticModelReadinessSnapshot:
        raise OutcomeDiagnosticExecutionModelError(
            "canonical outcome model readiness snapshot is required"
        )
    cache: OutcomeDiagnosticExecutionAuthorityCache | None = None
    result: AuthorizedOutcomeExecutionModel | None = None
    try:
        cache = build_outcome_diagnostic_execution_authority_cache(snapshot)
        cache.require_active()
        planned = cache.resolve_unit(planned_unit)
        expected = cache.artifact_for_unit(planned)
        reader = _reader(cache.lease.store)
        loaded = load_outcome_model_artifact_payload_at(reader, planned.model_owner_id)
        _validate_loaded_against_cache(loaded, expected)
        model = _reconstruct_state_model(loaded[2])
        generation = authorize_outcome_generation_model(
            model,
            expected.authorization,
            planned,
            cache.validated_plan,
            snapshot.protocol,
        )
        result = AuthorizedOutcomeExecutionModel(
            generation,
            expected.lineage,
            _token=_CACHE_TOKEN,
        )
        cache.require_active()
        result.require_active()
        yield result
    except OutcomeDiagnosticExecutionModelError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, OutcomeModelStoreError) as exc:
        raise OutcomeDiagnosticExecutionModelError(
            "authorized outcome model resolution failed"
        ) from exc
    finally:
        try:
            if cache is not None:
                cache.require_active()
            if result is not None:
                if model_state_sha256(result.model) != result.model_state_sha256:
                    raise OutcomeDiagnosticExecutionModelError(
                        "authorized outcome model state changed during execution"
                    )
                object.__setattr__(result, "_active", False)
        except OutcomeDiagnosticExecutionModelError:
            raise


__all__ = [
    "AuthorizedOutcomeExecutionModel",
    "OutcomeDiagnosticExecutionArtifact",
    "OutcomeDiagnosticExecutionAuthorityCache",
    "OutcomeDiagnosticExecutionLineage",
    "OutcomeDiagnosticExecutionModelError",
    "build_outcome_diagnostic_execution_authority_cache",
    "load_authorized_outcome_model_from_pinned_store",
]
