"""Typed model-artifact identities for the frozen outcome-group diagnostic.

This module is an identity boundary only.  It does not train a model, open an
environment, read a result store, or contain model weights.  The two diagnostic
conditions (RP and PEC) intentionally share one strict schema so that capacity,
data lineage, and accounting cannot drift between them.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)

from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    EXPECTED_MODEL_OWNERS,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_VIEWS,
    FAMILIES,
    TRAINING_TUPLE_IDS,
    OutcomeModelOwner,
    OutcomePlan,
    OutcomeView,
    ValidatedOutcomePlan,
    validate_outcome_diagnostic_plan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    CONDITIONS,
    OutcomeDiagnosticProtocolSnapshot,
    load_outcome_group_diagnostic_protocol,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataPayload,
    learner_samples,
)

HEX64 = r"^[0-9a-f]{64}$"
PREPARATION_COMMIT = r"^[0-9a-f]{40,64}$"
OUTCOME_ARTIFACT_STORE_PREFIX = "phase3-outcome-diagnostic-models-"
MODEL_SCHEMA_VERSION = "milestone6.phase3.outcome-diagnostic-model-artifact.v1"
AUTHORITY_SCHEMA_VERSION = "milestone6.phase3.outcome-diagnostic-model-authority.v1"
ARCHITECTURE_ID = "StateConditionedScorer"
INPUT_WIDTH = 54
EXPECTED_TRAINING_TASKS_PER_VIEW = 40
STATE_SCHEMA: tuple[tuple[str, tuple[int, ...], str], ...] = (
    ("network.0.bias", (48,), "float32"),
    ("network.0.weight", (48, 54), "float32"),
    ("network.2.bias", (24,), "float32"),
    ("network.2.weight", (24, 48), "float32"),
    ("network.4.bias", (1,), "float32"),
    ("network.4.weight", (1, 24), "float32"),
)


class OutcomeDiagnosticModelArtifactError(ValueError):
    """Raised when a diagnostic model identity is malformed or substituted."""


def _require_preparation_identity(
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
) -> tuple[str, str]:
    """Validate caller-supplied preparation provenance before deriving identities."""

    if (
        not isinstance(preparation_git_commit_sha, str)
        or not re.fullmatch(PREPARATION_COMMIT, preparation_git_commit_sha)
        or set(preparation_git_commit_sha) == {"0"}
        or not isinstance(preparation_provenance_sha256, str)
        or not re.fullmatch(HEX64, preparation_provenance_sha256)
        or set(preparation_provenance_sha256) == {"0"}
    ):
        raise OutcomeDiagnosticModelArtifactError(
            "nonzero preparation commit and provenance identities are required"
        )
    return preparation_git_commit_sha, preparation_provenance_sha256


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def outcome_artifact_store_id(plan_id: str) -> str:
    if not isinstance(plan_id, str) or not re.fullmatch(HEX64, plan_id):
        raise OutcomeDiagnosticModelArtifactError(
            "diagnostic artifact store derivation requires a canonical plan ID"
        )
    return f"{OUTCOME_ARTIFACT_STORE_PREFIX}{plan_id[:12]}"


def _require_snapshot(
    snapshot: OutcomeDiagnosticProtocolSnapshot,
) -> OutcomeDiagnosticProtocolSnapshot:
    if not isinstance(snapshot, OutcomeDiagnosticProtocolSnapshot):
        raise OutcomeDiagnosticModelArtifactError("diagnostic protocol snapshot is required")
    if not isinstance(snapshot.content, bytes) or not snapshot.content:
        raise OutcomeDiagnosticModelArtifactError("diagnostic protocol raw bytes are required")
    if _sha256(snapshot.content) != snapshot.sha256:
        raise OutcomeDiagnosticModelArtifactError("diagnostic protocol raw hash is inconsistent")
    supplied = snapshot.payload.get("diagnostic_protocol_sha256")
    if (
        not isinstance(supplied, str)
        or _digest(
            {
                key: value
                for key, value in snapshot.payload.items()
                if key != "diagnostic_protocol_sha256"
            }
        )
        != supplied
    ):
        raise OutcomeDiagnosticModelArtifactError("diagnostic protocol self-hash is inconsistent")
    execution_boundary = snapshot.payload.get("execution_boundary")
    if (
        snapshot.payload.get("scope") != "known-development-only"
        or not isinstance(execution_boundary, dict)
        or execution_boundary.get("final_family_access") is not False
        or execution_boundary.get("final_method_selection") is not False
        or execution_boundary.get("advancement_to_paired_objectives") is not False
    ):
        raise OutcomeDiagnosticModelArtifactError(
            "diagnostic protocol permits non-development authority"
        )
    try:
        fresh = load_outcome_group_diagnostic_protocol()
    except (OSError, TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelArtifactError(
            "canonical diagnostic protocol cannot be reloaded"
        ) from exc
    if snapshot != fresh:
        raise OutcomeDiagnosticModelArtifactError(
            "diagnostic protocol snapshot differs from fresh canonical authority"
        )
    return fresh


def _require_canonical_inputs(
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
) -> tuple[OutcomePlan, OutcomeDiagnosticProtocolSnapshot]:
    if not isinstance(plan, ValidatedOutcomePlan):
        raise OutcomeDiagnosticModelArtifactError("validated outcome plan is required")
    fresh = _require_snapshot(snapshot)
    try:
        validate_outcome_diagnostic_plan(plan.plan, snapshot=fresh)
    except (TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelArtifactError("outcome plan is not canonical") from exc
    if plan.plan.protocol_sha256 != fresh.sha256:
        raise OutcomeDiagnosticModelArtifactError("plan/protocol raw hash lineage differs")
    return plan.plan, fresh


class OutcomeTensorSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    shape: tuple[StrictInt, ...]
    dtype: Literal["float32"] = "float32"
    byte_length: StrictInt = Field(gt=0)
    sha256: str = Field(pattern=HEX64)

    @model_validator(mode="after")
    def tensor_shape_matches_bytes(self) -> "OutcomeTensorSchema":
        expected = 4
        for dimension in self.shape:
            if isinstance(dimension, bool) or dimension < 1:
                raise ValueError("tensor dimensions must be positive integers")
            expected *= dimension
        if not self.shape or self.byte_length != expected:
            raise ValueError("tensor shape and byte length differ")
        return self


@dataclass(frozen=True, slots=True)
class OutcomeStateTensorPayload:
    name: str
    shape: tuple[int, ...]
    data: bytes


@dataclass(frozen=True, slots=True)
class PinnedOutcomeModelState:
    """Typed in-memory state bytes held by the preparation boundary."""

    tensors: tuple[OutcomeStateTensorPayload, ...]


@dataclass(frozen=True, slots=True)
class PinnedOutcomeTrainingEvidence:
    """Exact canonical evidence payload bytes used to derive one training view."""

    payload: TrainingDataPayload
    payload_bytes: bytes


_MODEL_AUTHORIZATION_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class AuthorizedOutcomeModelArtifact:
    """Opaque result minted only after plan, evidence, and state validation."""

    record: OutcomeDiagnosticModelArtifactRecord
    owner_id: str
    view_id: str
    model_state_sha256: str
    training_examples: int
    _token: object

    def __init__(
        self,
        record: OutcomeDiagnosticModelArtifactRecord,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _MODEL_AUTHORIZATION_TOKEN:
            raise ValueError("outcome model authorization requires semantic validation")
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "owner_id", record.key.owner_id)
        object.__setattr__(self, "view_id", record.key.view_id)
        object.__setattr__(self, "model_state_sha256", record.key.model_state_sha256)
        object.__setattr__(
            self, "training_examples", record.key.training_accounting.training_examples
        )
        object.__setattr__(self, "_token", _MODEL_AUTHORIZATION_TOKEN)


class OutcomeTrainingAccounting(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    optimizer_steps: StrictInt = Field(gt=0)
    forward_passes: StrictInt = Field(gt=0)
    training_examples: StrictInt = Field(gt=0)
    serialization_calls: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def accounting_is_exact(self) -> "OutcomeTrainingAccounting":
        if self.forward_passes != self.optimizer_steps * self.training_examples:
            raise ValueError("diagnostic forward-pass accounting differs")
        if self.serialization_calls != 1:
            raise ValueError("diagnostic serialization accounting differs")
        return self


def inspect_outcome_model_state(
    payload: PinnedOutcomeModelState,
) -> tuple[tuple[OutcomeTensorSchema, ...], str]:
    """Recompute exact StateConditionedScorer tensor and whole-state identities."""

    if not isinstance(payload, PinnedOutcomeModelState):
        raise OutcomeDiagnosticModelArtifactError("typed pinned model state is required")
    observed = tuple((row.name, row.shape, "float32") for row in payload.tensors)
    if observed != STATE_SCHEMA:
        raise OutcomeDiagnosticModelArtifactError("StateConditionedScorer tensor schema drifted")
    digest = hashlib.sha256()
    metadata: list[OutcomeTensorSchema] = []
    for tensor in payload.tensors:
        if not isinstance(tensor.data, bytes):
            raise OutcomeDiagnosticModelArtifactError(
                "model tensor payload must be immutable bytes"
            )
        expected_bytes = 4
        for dimension in tensor.shape:
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
                raise OutcomeDiagnosticModelArtifactError("model tensor shape is invalid")
            expected_bytes *= dimension
        if len(tensor.data) != expected_bytes:
            raise OutcomeDiagnosticModelArtifactError("model tensor byte length differs")
        header = canonical_json_bytes(
            {"name": tensor.name, "dtype": "torch.float32", "shape": list(tensor.shape)}
        )
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(tensor.data).to_bytes(8, "big"))
        digest.update(tensor.data)
        metadata.append(
            OutcomeTensorSchema(
                name=tensor.name,
                shape=tensor.shape,
                byte_length=len(tensor.data),
                sha256=_sha256(tensor.data),
            )
        )
    return tuple(metadata), digest.hexdigest()


class OutcomeDiagnosticModelArtifactKey(BaseModel):
    """Canonical identity of one trained RP/PEC model owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[MODEL_SCHEMA_VERSION] = MODEL_SCHEMA_VERSION
    key_id: str = Field(pattern=HEX64)
    plan_id: str = Field(pattern=HEX64)
    plan_parent_commit_sha: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    protocol_sha256: str = Field(pattern=HEX64)
    protocol_self_sha256: str = Field(pattern=HEX64)
    protocol_file_sha256: str = Field(pattern=HEX64)
    condition_id: Literal[CONDITIONS[0], CONDITIONS[1]]
    view_id: str = Field(pattern=HEX64)
    owner_id: str = Field(pattern=HEX64)
    heldout_family: Literal[
        FAMILIES[0], FAMILIES[1], FAMILIES[2], FAMILIES[3], FAMILIES[4], FAMILIES[5]
    ]
    fold_id: str = Field(min_length=1)
    replicate: StrictInt = Field(ge=0, le=4)
    training_tuple_id: Literal[
        TRAINING_TUPLE_IDS[0], TRAINING_TUPLE_IDS[1], TRAINING_TUPLE_IDS[2], TRAINING_TUPLE_IDS[3]
    ]
    model_seed: StrictInt
    data_order_seed: StrictInt
    consumer_unit_ids_sha256: str = Field(pattern=HEX64)
    consumer_seed_lineage_sha256: str = Field(pattern=HEX64)
    consumer_count: StrictInt = 24
    candidate_episodes_per_task: StrictInt = 150
    adaptation_actions_per_task: StrictInt = 2048
    probe_actions_per_task: StrictInt = 64
    maximum_actions_per_candidate_episode: StrictInt = 64
    evidence_row_sha256: str = Field(pattern=HEX64)
    evidence_payload_sha256: str = Field(pattern=HEX64)
    evidence_payload_bytes: StrictInt = Field(gt=0)
    ordered_training_task_ids: tuple[str, ...]
    learning_rate: StrictFloat = Field(gt=0)
    training_epochs: StrictInt = Field(gt=0)
    optimizer_id: Literal["adam"] = "adam"
    weight_decay: StrictFloat = 0.0001
    device: Literal["cpu"]
    device_portable: StrictBool = True
    torch_threads: Literal[1] = 1
    processes: Literal[1] = 1
    feature_mask_sha256: str = Field(pattern=HEX64)
    transformation_sha256: str = Field(pattern=HEX64)
    representation_sha256: str = Field(pattern=HEX64)
    model_identity_sha256: str = Field(pattern=HEX64)
    architecture_id: Literal[ARCHITECTURE_ID] = ARCHITECTURE_ID
    input_width: StrictInt = INPUT_WIDTH
    trainable_parameters: StrictInt = EXPECTED_PARAMETER_COUNT
    state_schema: tuple[OutcomeTensorSchema, ...]
    model_state_sha256: str = Field(pattern=HEX64)
    training_accounting: OutcomeTrainingAccounting
    preparation_git_commit_sha: str = Field(pattern=PREPARATION_COMMIT)
    preparation_provenance_sha256: str = Field(pattern=HEX64)

    @property
    def expected_key_id(self) -> str:
        return _digest(self.model_dump(mode="json", exclude={"key_id"}))

    @model_validator(mode="after")
    def key_is_exact(self) -> "OutcomeDiagnosticModelArtifactKey":
        if set(self.preparation_git_commit_sha) == {"0"}:
            raise ValueError("preparation commit provenance is required")
        if set(self.preparation_provenance_sha256) == {"0"}:
            raise ValueError("preparation provenance identity is required")
        if self.input_width != INPUT_WIDTH or self.trainable_parameters != EXPECTED_PARAMETER_COUNT:
            raise ValueError("diagnostic architecture capacity drifted")
        if (
            self.consumer_count != 24
            or self.candidate_episodes_per_task != 150
            or self.adaptation_actions_per_task != 2048
            or self.probe_actions_per_task != 64
            or self.maximum_actions_per_candidate_episode != 64
        ):
            raise ValueError("diagnostic consumer or interaction budgets drifted")
        if (
            self.device != "cpu"
            or not self.device_portable
            or self.torch_threads != 1
            or self.processes != 1
            or self.weight_decay != 0.0001
        ):
            raise ValueError("diagnostic optimizer/device policy drifted")
        if (
            len(self.ordered_training_task_ids) != EXPECTED_TRAINING_TASKS_PER_VIEW
            or len(set(self.ordered_training_task_ids))
            != EXPECTED_TRAINING_TASKS_PER_VIEW
        ):
            raise ValueError("diagnostic training task order is incomplete")
        schema = tuple((row.name, tuple(row.shape), row.dtype) for row in self.state_schema)
        if schema != STATE_SCHEMA:
            raise ValueError("StateConditionedScorer tensor schema drifted")
        if self.key_id != self.expected_key_id:
            raise ValueError("diagnostic model key self-hash mismatch")
        return self


class OutcomeRPModelArtifactKey(OutcomeDiagnosticModelArtifactKey):
    condition_id: Literal[CONDITIONS[0]]


class OutcomePECModelArtifactKey(OutcomeDiagnosticModelArtifactKey):
    condition_id: Literal[CONDITIONS[1]]


class OutcomeDiagnosticModelArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[MODEL_SCHEMA_VERSION] = MODEL_SCHEMA_VERSION
    record_id: str = Field(pattern=HEX64)
    key: OutcomeDiagnosticModelArtifactKey

    @property
    def expected_record_id(self) -> str:
        return _digest(self.model_dump(mode="json", exclude={"record_id"}))

    @model_validator(mode="after")
    def record_is_exact(self) -> "OutcomeDiagnosticModelArtifactRecord":
        if self.record_id != self.expected_record_id:
            raise ValueError("diagnostic model record self-hash mismatch")
        return self


class OutcomeRPModelArtifactRecord(OutcomeDiagnosticModelArtifactRecord):
    key: OutcomeRPModelArtifactKey


class OutcomePECModelArtifactRecord(OutcomeDiagnosticModelArtifactRecord):
    key: OutcomePECModelArtifactKey


class OutcomeDiagnosticArtifactRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    owner_id: str = Field(pattern=HEX64)
    view_id: str = Field(pattern=HEX64)
    condition_id: Literal[CONDITIONS[0], CONDITIONS[1]]
    heldout_family: str = Field(min_length=1)
    fold_id: str = Field(min_length=1)
    replicate: StrictInt = Field(ge=0, le=4)
    training_tuple_id: str = Field(min_length=1)
    model_seed: StrictInt
    data_order_seed: StrictInt
    feature_mask_sha256: str = Field(pattern=HEX64)
    transformation_sha256: str = Field(pattern=HEX64)
    representation_sha256: str = Field(pattern=HEX64)
    model_identity_sha256: str = Field(pattern=HEX64)
    consumer_unit_ids_sha256: str = Field(pattern=HEX64)
    consumer_seed_lineage_sha256: str = Field(pattern=HEX64)
    record_id: str = Field(pattern=HEX64)
    key_id: str = Field(pattern=HEX64)
    model_state_sha256: str = Field(pattern=HEX64)


class OutcomeDiagnosticViewRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    view_id: str = Field(pattern=HEX64)
    condition_id: Literal[CONDITIONS[0], CONDITIONS[1]]
    heldout_family: str = Field(min_length=1)
    replicate: StrictInt = Field(ge=0, le=4)
    evidence_row_sha256: str = Field(pattern=HEX64)
    feature_mask_sha256: str = Field(pattern=HEX64)
    transformation_sha256: str = Field(pattern=HEX64)
    representation_sha256: str = Field(pattern=HEX64)


class OutcomeDiagnosticEvidenceRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    heldout_family: str = Field(min_length=1)
    replicate: StrictInt = Field(ge=0, le=4)
    evidence_row_sha256: str = Field(pattern=HEX64)
    evidence_payload_sha256: str = Field(pattern=HEX64)
    evidence_payload_bytes: StrictInt = Field(gt=0)
    ordered_training_task_ids: tuple[str, ...]


class OutcomeDiagnosticModelArtifactAuthority(BaseModel):
    """Compact authority summary; it contains identities, never weights or plans."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[AUTHORITY_SCHEMA_VERSION] = AUTHORITY_SCHEMA_VERSION
    authority_sha256: str = Field(pattern=HEX64)
    development_only: StrictBool = True
    final: StrictBool = False
    final_family_access: StrictBool = False
    plan_id: str = Field(pattern=HEX64)
    plan_parent_commit_sha: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    protocol_sha256: str = Field(pattern=HEX64)
    protocol_self_sha256: str = Field(pattern=HEX64)
    protocol_file_sha256: str = Field(pattern=HEX64)
    preparation_git_commit_sha: str = Field(pattern=PREPARATION_COMMIT)
    preparation_provenance_sha256: str = Field(pattern=HEX64)
    generation_git_commit_sha: str = Field(pattern=PREPARATION_COMMIT)
    artifact_store_id: str = Field(min_length=1)
    condition_ids: tuple[Literal[CONDITIONS[0]], Literal[CONDITIONS[1]]]
    views: tuple[OutcomeDiagnosticViewRow, ...]
    evidence: tuple[OutcomeDiagnosticEvidenceRow, ...]
    artifacts: tuple[OutcomeDiagnosticArtifactRow, ...]

    @property
    def expected_authority_sha256(self) -> str:
        return _digest(self.model_dump(mode="json", exclude={"authority_sha256"}))

    @model_validator(mode="after")
    def authority_is_exact(self) -> "OutcomeDiagnosticModelArtifactAuthority":
        if set(self.preparation_git_commit_sha) == {"0"}:
            raise ValueError("preparation commit provenance is required")
        if set(self.preparation_provenance_sha256) == {"0"}:
            raise ValueError("preparation provenance identity is required")
        if set(self.generation_git_commit_sha) == {"0"}:
            raise ValueError("authority generation commit provenance is required")
        if (
            self.artifact_store_id in {".", ".."}
            or "/" in self.artifact_store_id
            or "\\" in self.artifact_store_id
            or not self.artifact_store_id.startswith(OUTCOME_ARTIFACT_STORE_PREFIX)
        ):
            raise ValueError("diagnostic artifact store identity is unsafe")
        if not self.development_only or self.final or self.final_family_access:
            raise ValueError("diagnostic model authority permits final-family access")
        if self.condition_ids != CONDITIONS:
            raise ValueError("diagnostic condition universe drifted")
        if (
            any(row.heldout_family not in FAMILIES for row in self.views)
            or any(row.heldout_family not in FAMILIES for row in self.evidence)
            or any(row.heldout_family not in FAMILIES for row in self.artifacts)
        ):
            raise ValueError("diagnostic authority contains a foreign family")
        if (
            len(self.views) != EXPECTED_VIEWS
            or len({row.view_id for row in self.views}) != EXPECTED_VIEWS
        ):
            raise ValueError("diagnostic view universe is partial, extra, or duplicated")
        if {(row.condition_id, row.heldout_family, row.replicate) for row in self.views} != {
            (condition, family, replicate)
            for condition in CONDITIONS
            for family in FAMILIES
            for replicate in range(5)
        }:
            raise ValueError("diagnostic view matrix is partial or extra")
        if tuple(row.view_id for row in self.views) != tuple(
            sorted(row.view_id for row in self.views)
        ):
            raise ValueError("diagnostic view rows are not canonically ordered")
        if (
            len(self.evidence) != 30
            or len({(row.heldout_family, row.replicate) for row in self.evidence}) != 30
        ):
            raise ValueError("diagnostic evidence universe is partial, extra, or duplicated")
        if tuple((row.heldout_family, row.replicate) for row in self.evidence) != tuple(
            sorted((row.heldout_family, row.replicate) for row in self.evidence)
        ):
            raise ValueError("diagnostic evidence rows are not canonically ordered")
        if (
            len(self.artifacts) != EXPECTED_MODEL_OWNERS
            or len({row.owner_id for row in self.artifacts}) != EXPECTED_MODEL_OWNERS
        ):
            raise ValueError("diagnostic artifact universe is partial, extra, or duplicated")
        if {row.condition_id for row in self.artifacts} != set(CONDITIONS):
            raise ValueError("diagnostic artifact condition universe is incomplete")
        if any(
            sum(row.condition_id == condition for row in self.artifacts) != 120
            for condition in CONDITIONS
        ):
            raise ValueError("diagnostic artifact condition counts drifted")
        if tuple(row.owner_id for row in self.artifacts) != tuple(
            sorted(row.owner_id for row in self.artifacts)
        ):
            raise ValueError("diagnostic artifact rows are not canonically ordered")
        if self.authority_sha256 != self.expected_authority_sha256:
            raise ValueError("diagnostic authority self-hash mismatch")
        return self


def _canonical(value: BaseModel) -> bytes:
    return canonical_json_bytes(value.model_dump(mode="json"))


def canonical_outcome_model_artifact_key_bytes(key: OutcomeDiagnosticModelArtifactKey) -> bytes:
    if not isinstance(key, OutcomeDiagnosticModelArtifactKey):
        raise OutcomeDiagnosticModelArtifactError("model key is not typed")
    try:
        OutcomeDiagnosticModelArtifactKey.model_validate(key.model_dump(mode="json"))
    except (TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelArtifactError("model key is not canonical") from exc
    return _canonical(key)


def canonical_outcome_model_artifact_record_bytes(
    record: OutcomeDiagnosticModelArtifactRecord,
) -> bytes:
    if not isinstance(record, OutcomeDiagnosticModelArtifactRecord):
        raise OutcomeDiagnosticModelArtifactError("model record is not typed")
    try:
        OutcomeDiagnosticModelArtifactRecord.model_validate(record.model_dump(mode="json"))
    except (TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelArtifactError("model record is not canonical") from exc
    return _canonical(record)


def canonical_outcome_model_artifact_authority_bytes(
    authority: OutcomeDiagnosticModelArtifactAuthority,
) -> bytes:
    if not isinstance(authority, OutcomeDiagnosticModelArtifactAuthority):
        raise OutcomeDiagnosticModelArtifactError("model authority is not typed")
    try:
        OutcomeDiagnosticModelArtifactAuthority.model_validate(authority.model_dump(mode="json"))
    except (TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelArtifactError("model authority is not canonical") from exc
    return _canonical(authority)


def _load(content: bytes, model: type[BaseModel], label: str) -> BaseModel:
    if not isinstance(content, bytes) or not content:
        raise OutcomeDiagnosticModelArtifactError(f"{label} bytes are missing")
    try:
        raw = json.loads(content)
        if not isinstance(raw, dict) or canonical_json_bytes(raw) != content:
            raise ValueError
        return model.model_validate(raw)
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise OutcomeDiagnosticModelArtifactError(f"{label} bytes are not canonical") from exc


def load_outcome_model_artifact_key_bytes(content: bytes) -> OutcomeDiagnosticModelArtifactKey:
    """Parse intrinsic identity only; semantic use still requires plan/state validation."""

    return _load(content, OutcomeDiagnosticModelArtifactKey, "model key")  # type: ignore[return-value]


def load_outcome_model_artifact_record_bytes(
    content: bytes,
) -> OutcomeDiagnosticModelArtifactRecord:
    """Parse intrinsic identity only; semantic use still requires plan/state validation."""

    return _load(content, OutcomeDiagnosticModelArtifactRecord, "model record")  # type: ignore[return-value]


def load_outcome_model_artifact_authority_bytes(
    content: bytes,
) -> OutcomeDiagnosticModelArtifactAuthority:
    """Parse an untrusted summary; callers must run the public authority validator."""

    return _load(content, OutcomeDiagnosticModelArtifactAuthority, "model authority")  # type: ignore[return-value]


def _owner_view(plan: OutcomePlan, owner_id: str) -> tuple[OutcomeModelOwner, OutcomeView]:
    owner = next((row for row in plan.model_owners if row.owner_id == owner_id), None)
    if owner is None:
        raise OutcomeDiagnosticModelArtifactError("model owner is foreign to the canonical plan")
    view = next((row for row in plan.views if row.view_id == owner.view_id), None)
    if view is None:
        raise OutcomeDiagnosticModelArtifactError("model view is foreign to the canonical plan")
    return owner, view


def _evidence(
    plan: OutcomePlan, family: str, replicate: int
) -> tuple[str, str, int, tuple[str, ...]]:
    for raw in plan.evidence_lineage_rows:
        try:
            row = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OutcomeDiagnosticModelArtifactError("canonical evidence row is not JSON") from exc
        if not isinstance(row, dict) or canonical_json_bytes(row) != raw:
            raise OutcomeDiagnosticModelArtifactError("canonical evidence row bytes differ")
        if row.get("family_id") == family and row.get("replicate") == replicate:
            tasks = row.get("ordered_training_task_ids")
            if (
                not isinstance(tasks, list)
                or len(tasks) != EXPECTED_TRAINING_TASKS_PER_VIEW
                or len(set(tasks)) != EXPECTED_TRAINING_TASKS_PER_VIEW
                or any(not isinstance(task, str) or not task for task in tasks)
            ):
                raise OutcomeDiagnosticModelArtifactError(
                    "canonical evidence task order is invalid"
                )
            payload_sha = row.get("payload_sha256")
            payload_bytes = row.get("payload_bytes")
            if (
                not isinstance(payload_sha, str)
                or len(payload_sha) != 64
                or any(char not in "0123456789abcdef" for char in payload_sha)
                or isinstance(payload_bytes, bool)
                or not isinstance(payload_bytes, int)
                or payload_bytes < 1
            ):
                raise OutcomeDiagnosticModelArtifactError(
                    "canonical evidence payload identity is missing"
                )
            return _sha256(raw), payload_sha, payload_bytes, tuple(tasks)
    raise OutcomeDiagnosticModelArtifactError("canonical evidence row is missing")


def _derive_training_example_count(
    plan: OutcomePlan,
    view: OutcomeView,
    evidence: PinnedOutcomeTrainingEvidence,
) -> int:
    if not isinstance(evidence, PinnedOutcomeTrainingEvidence):
        raise OutcomeDiagnosticModelArtifactError("typed pinned training evidence is required")
    if not isinstance(evidence.payload, TrainingDataPayload) or not isinstance(
        evidence.payload_bytes, bytes
    ):
        raise OutcomeDiagnosticModelArtifactError("training evidence payload is not canonical")
    row_sha, expected_sha, expected_bytes, expected_tasks = _evidence(
        plan, view.heldout_family, view.replicate
    )
    del row_sha
    observed_bytes = canonical_json_bytes(evidence.payload.model_dump(mode="json"))
    sample_tasks = tuple(sample.task_id for sample in evidence.payload.samples)
    if (
        observed_bytes != evidence.payload_bytes
        or _sha256(observed_bytes) != expected_sha
        or len(observed_bytes) != expected_bytes
        or sample_tasks != expected_tasks
        or sample_tasks != view.training_task_ids
    ):
        raise OutcomeDiagnosticModelArtifactError(
            "training evidence content differs from canonical view authority"
        )
    # Late import avoids a module cycle: generation owns the exact shared-source
    # RP/PEC transformation, while this boundary independently derives its size.
    from levelup.experiments.milestone6_phase3_outcome_diagnostic_generation import (
        outcome_group_training_examples,
    )

    count = len(
        outcome_group_training_examples(learner_samples(evidence.payload), view.condition_id)
    )
    if count < 1:
        raise OutcomeDiagnosticModelArtifactError("canonical training example set is empty")
    return count


def _build_outcome_model_artifact_key_canonical(
    canonical_plan: OutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    *,
    owner_id: str,
    state_payload: PinnedOutcomeModelState,
    training_evidence: PinnedOutcomeTrainingEvidence,
    device: Literal["cpu"],
    training_accounting: OutcomeTrainingAccounting,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
) -> OutcomeDiagnosticModelArtifactKey:
    """Build after the public caller has validated immutable authorities."""

    preparation_git_commit_sha, preparation_provenance_sha256 = _require_preparation_identity(
        preparation_git_commit_sha, preparation_provenance_sha256
    )
    if device != "cpu":
        raise OutcomeDiagnosticModelArtifactError("diagnostic model preparation requires CPU")
    state_schema, model_state_sha256 = inspect_outcome_model_state(state_payload)
    owner, view = _owner_view(canonical_plan, owner_id)
    exact_training_examples = _derive_training_example_count(
        canonical_plan, view, training_evidence
    )
    evidence_row_sha, payload_sha, payload_bytes, task_ids = _evidence(
        canonical_plan, view.heldout_family, view.replicate
    )
    if tuple(view.training_task_ids) != task_ids:
        raise OutcomeDiagnosticModelArtifactError("evidence task order differs from canonical view")
    if owner.condition_id not in CONDITIONS:
        raise OutcomeDiagnosticModelArtifactError("owner condition is outside RP/PEC universe")
    consumers = tuple(
        unit for unit in canonical_plan.units if unit.model_owner_id == owner.owner_id
    )
    if len(consumers) != 24:
        raise OutcomeDiagnosticModelArtifactError("model owner must have exactly 24 consumers")
    budgets = {
        (
            unit.candidate_episodes_per_task,
            unit.adaptation_actions_per_task,
            unit.probe_actions_per_task,
            unit.maximum_actions_per_candidate_episode,
        )
        for unit in consumers
    }
    if budgets != {(150, 2048, 64, 64)}:
        raise OutcomeDiagnosticModelArtifactError("model owner consumer budgets drifted")
    consumer_unit_ids_sha256 = _digest([unit.unit_id for unit in consumers])
    consumer_seed_lineage_sha256 = _digest(
        [
            {
                "unit_id": unit.unit_id,
                "tuple_id": unit.tuple_id,
                "task_id": unit.task_id,
                "task_index": unit.task_index,
                "model_seed": unit.model_seed,
                "environment_seed": unit.environment_seed,
                "probe_seed": unit.probe_seed,
                "search_seed": unit.search_seed,
                "data_order_seed": unit.data_order_seed,
            }
            for unit in consumers
        ]
    )
    tuple_id = owner.training_tuple_id
    expected = {
        "lr0p003-e120": (0.003, 120),
        "lr0p003-e180": (0.003, 180),
        "lr0p01-e120": (0.01, 120),
        "lr0p01-e180": (0.01, 180),
    }[tuple_id]
    if (owner.learning_rate, owner.training_epochs) != expected:
        raise OutcomeDiagnosticModelArtifactError("owner training tuple differs")
    if training_accounting.optimizer_steps != owner.training_epochs:
        raise OutcomeDiagnosticModelArtifactError("optimizer steps differ from owner epochs")
    if training_accounting.training_examples != exact_training_examples:
        raise OutcomeDiagnosticModelArtifactError(
            "training example count differs from canonical evidence content"
        )
    key_data = dict(
        schema_version=MODEL_SCHEMA_VERSION,
        key_id="0" * 64,
        plan_id=canonical_plan.plan_id,
        plan_parent_commit_sha=canonical_plan.parent_commit_sha,
        protocol_sha256=canonical_plan.protocol_sha256,
        protocol_self_sha256=str(snapshot.payload["diagnostic_protocol_sha256"]),
        protocol_file_sha256=snapshot.sha256,
        condition_id=owner.condition_id,
        view_id=view.view_id,
        owner_id=owner.owner_id,
        heldout_family=view.heldout_family,
        fold_id=view.fold_id,
        replicate=view.replicate,
        training_tuple_id=tuple_id,
        model_seed=owner.model_seed,
        data_order_seed=view.data_order_seed,
        consumer_unit_ids_sha256=consumer_unit_ids_sha256,
        consumer_seed_lineage_sha256=consumer_seed_lineage_sha256,
        consumer_count=24,
        candidate_episodes_per_task=150,
        adaptation_actions_per_task=2048,
        probe_actions_per_task=64,
        maximum_actions_per_candidate_episode=64,
        evidence_row_sha256=evidence_row_sha,
        evidence_payload_sha256=payload_sha,
        evidence_payload_bytes=payload_bytes,
        ordered_training_task_ids=task_ids,
        learning_rate=owner.learning_rate,
        training_epochs=owner.training_epochs,
        optimizer_id="adam",
        weight_decay=0.0001,
        device=device,
        device_portable=True,
        torch_threads=1,
        processes=1,
        feature_mask_sha256=owner.feature_mask_sha256,
        transformation_sha256=owner.transformation_sha256,
        representation_sha256=view.representation_sha256,
        model_identity_sha256=owner.model_identity_sha256,
        architecture_id=ARCHITECTURE_ID,
        input_width=INPUT_WIDTH,
        trainable_parameters=EXPECTED_PARAMETER_COUNT,
        state_schema=[row.model_dump(mode="json") for row in state_schema],
        model_state_sha256=model_state_sha256,
        training_accounting=training_accounting.model_dump(mode="json"),
        preparation_git_commit_sha=preparation_git_commit_sha,
        preparation_provenance_sha256=preparation_provenance_sha256,
    )
    # The plan ID is already a canonical SHA identity; never derive a replacement.
    key_data["key_id"] = _digest({key: value for key, value in key_data.items() if key != "key_id"})
    try:
        return OutcomeDiagnosticModelArtifactKey.model_validate(key_data)
    except (TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelArtifactError("constructed model key is invalid") from exc


def build_outcome_model_artifact_key(
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    *,
    owner_id: str,
    state_payload: PinnedOutcomeModelState,
    training_evidence: PinnedOutcomeTrainingEvidence,
    device: Literal["cpu"],
    training_accounting: OutcomeTrainingAccounting,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
) -> OutcomeDiagnosticModelArtifactKey:
    """Construct one key after hashing canonical pinned state bytes."""

    canonical_plan, fresh = _require_canonical_inputs(plan, snapshot)
    return _build_outcome_model_artifact_key_canonical(
        canonical_plan,
        fresh,
        owner_id=owner_id,
        state_payload=state_payload,
        training_evidence=training_evidence,
        device=device,
        training_accounting=training_accounting,
        preparation_git_commit_sha=preparation_git_commit_sha,
        preparation_provenance_sha256=preparation_provenance_sha256,
    )


def build_outcome_model_artifact_record(
    key: OutcomeDiagnosticModelArtifactKey,
) -> OutcomeDiagnosticModelArtifactRecord:
    try:
        record_data = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "record_id": "0" * 64,
            "key": key.model_dump(mode="json"),
        }
        record_data["record_id"] = _digest(
            {key: value for key, value in record_data.items() if key != "record_id"}
        )
        return OutcomeDiagnosticModelArtifactRecord.model_validate(record_data)
    except (TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelArtifactError("constructed model record is invalid") from exc


def build_outcome_rp_model_artifact_key(
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    *,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
    **kwargs: Any,
) -> OutcomeRPModelArtifactKey:
    key = build_outcome_model_artifact_key(
        plan,
        snapshot,
        preparation_git_commit_sha=preparation_git_commit_sha,
        preparation_provenance_sha256=preparation_provenance_sha256,
        **kwargs,
    )
    try:
        return OutcomeRPModelArtifactKey.model_validate(key.model_dump(mode="json"))
    except (TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelArtifactError("RP model key is invalid") from exc


def build_outcome_pec_model_artifact_key(
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    *,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
    **kwargs: Any,
) -> OutcomePECModelArtifactKey:
    key = build_outcome_model_artifact_key(
        plan,
        snapshot,
        preparation_git_commit_sha=preparation_git_commit_sha,
        preparation_provenance_sha256=preparation_provenance_sha256,
        **kwargs,
    )
    try:
        return OutcomePECModelArtifactKey.model_validate(key.model_dump(mode="json"))
    except (TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelArtifactError("PEC model key is invalid") from exc


def build_outcome_rp_model_artifact_record(
    key: OutcomeRPModelArtifactKey,
) -> OutcomeRPModelArtifactRecord:
    generic = build_outcome_model_artifact_record(key)
    return OutcomeRPModelArtifactRecord.model_validate(generic.model_dump(mode="json"))


def build_outcome_pec_model_artifact_record(
    key: OutcomePECModelArtifactKey,
) -> OutcomePECModelArtifactRecord:
    generic = build_outcome_model_artifact_record(key)
    return OutcomePECModelArtifactRecord.model_validate(generic.model_dump(mode="json"))


def validate_outcome_model_artifact_against_plan(
    record: OutcomeDiagnosticModelArtifactRecord,
    state_payload: PinnedOutcomeModelState,
    training_evidence: PinnedOutcomeTrainingEvidence,
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    *,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
) -> AuthorizedOutcomeModelArtifact:
    canonical_plan, fresh = _require_canonical_inputs(plan, snapshot)
    preparation_git_commit_sha, preparation_provenance_sha256 = _require_preparation_identity(
        preparation_git_commit_sha, preparation_provenance_sha256
    )
    if record.key.plan_id != canonical_plan.plan_id:
        raise OutcomeDiagnosticModelArtifactError("model record plan lineage differs")
    if (
        record.key.preparation_git_commit_sha != preparation_git_commit_sha
        or record.key.preparation_provenance_sha256 != preparation_provenance_sha256
    ):
        raise OutcomeDiagnosticModelArtifactError("model record preparation provenance differs")
    expected = _build_outcome_model_artifact_key_canonical(
        canonical_plan,
        fresh,
        owner_id=record.key.owner_id,
        state_payload=state_payload,
        training_evidence=training_evidence,
        device=record.key.device,
        training_accounting=record.key.training_accounting,
        preparation_git_commit_sha=preparation_git_commit_sha,
        preparation_provenance_sha256=preparation_provenance_sha256,
    )
    if record.key != expected:
        raise OutcomeDiagnosticModelArtifactError("model record differs from canonical plan")
    return AuthorizedOutcomeModelArtifact(record, _token=_MODEL_AUTHORIZATION_TOKEN)


def build_outcome_model_artifact_authority(
    records: Sequence[OutcomeDiagnosticModelArtifactRecord],
    state_payloads: Mapping[str, PinnedOutcomeModelState],
    training_evidence_by_view: Mapping[str, PinnedOutcomeTrainingEvidence],
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    *,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
    generation_git_commit_sha: str,
) -> OutcomeDiagnosticModelArtifactAuthority:
    canonical_plan, snapshot = _require_canonical_inputs(plan, snapshot)
    preparation_git_commit_sha, preparation_provenance_sha256 = _require_preparation_identity(
        preparation_git_commit_sha, preparation_provenance_sha256
    )
    if (
        not isinstance(generation_git_commit_sha, str)
        or re.fullmatch(PREPARATION_COMMIT, generation_git_commit_sha) is None
        or set(generation_git_commit_sha) == {"0"}
    ):
        raise OutcomeDiagnosticModelArtifactError(
            "authority generation commit provenance is required"
        )
    if len(records) != EXPECTED_MODEL_OWNERS:
        raise OutcomeDiagnosticModelArtifactError("model authority requires exactly 240 owners")
    owner_ids = {record.key.owner_id for record in records}
    expected_owner_ids = {owner.owner_id for owner in canonical_plan.model_owners}
    if owner_ids != expected_owner_ids or len(owner_ids) != EXPECTED_MODEL_OWNERS:
        raise OutcomeDiagnosticModelArtifactError(
            "model authority owners are partial, extra, or foreign"
        )
    if set(state_payloads) != expected_owner_ids:
        raise OutcomeDiagnosticModelArtifactError("model state payload universe differs")
    expected_view_ids = {view.view_id for view in canonical_plan.views}
    if set(training_evidence_by_view) != expected_view_ids:
        raise OutcomeDiagnosticModelArtifactError("training evidence view universe differs")
    for record in records:
        expected_key = _build_outcome_model_artifact_key_canonical(
            canonical_plan,
            snapshot,
            owner_id=record.key.owner_id,
            state_payload=state_payloads[record.key.owner_id],
            training_evidence=training_evidence_by_view[record.key.view_id],
            device=record.key.device,
            training_accounting=record.key.training_accounting,
            preparation_git_commit_sha=preparation_git_commit_sha,
            preparation_provenance_sha256=preparation_provenance_sha256,
        )
        if (
            record.key != expected_key
            or record.record_id != record.expected_record_id
            or record.key.preparation_git_commit_sha != preparation_git_commit_sha
            or record.key.preparation_provenance_sha256 != preparation_provenance_sha256
        ):
            raise OutcomeDiagnosticModelArtifactError(
                "model record differs from canonical plan and state"
            )
    evidence_rows = []
    for raw in canonical_plan.evidence_lineage_rows:
        row = json.loads(raw)
        evidence_rows.append(
            OutcomeDiagnosticEvidenceRow(
                heldout_family=row["family_id"],
                replicate=row["replicate"],
                evidence_row_sha256=_sha256(raw),
                evidence_payload_sha256=row["payload_sha256"],
                evidence_payload_bytes=row["payload_bytes"],
                ordered_training_task_ids=tuple(row["ordered_training_task_ids"]),
            )
        )
    views = []
    for view in canonical_plan.views:
        row_sha, _payload_sha, _payload_bytes, _tasks = _evidence(
            canonical_plan, view.heldout_family, view.replicate
        )
        views.append(
            OutcomeDiagnosticViewRow(
                view_id=view.view_id,
                condition_id=view.condition_id,
                heldout_family=view.heldout_family,
                replicate=view.replicate,
                evidence_row_sha256=row_sha,
                feature_mask_sha256=view.feature_mask_sha256,
                transformation_sha256=view.transformation_sha256,
                representation_sha256=view.representation_sha256,
            )
        )
    artifact_rows = tuple(
        OutcomeDiagnosticArtifactRow(
            owner_id=record.key.owner_id,
            view_id=record.key.view_id,
            condition_id=record.key.condition_id,
            heldout_family=record.key.heldout_family,
            fold_id=record.key.fold_id,
            replicate=record.key.replicate,
            training_tuple_id=record.key.training_tuple_id,
            model_seed=record.key.model_seed,
            data_order_seed=record.key.data_order_seed,
            feature_mask_sha256=record.key.feature_mask_sha256,
            transformation_sha256=record.key.transformation_sha256,
            representation_sha256=record.key.representation_sha256,
            model_identity_sha256=record.key.model_identity_sha256,
            consumer_unit_ids_sha256=record.key.consumer_unit_ids_sha256,
            consumer_seed_lineage_sha256=record.key.consumer_seed_lineage_sha256,
            record_id=record.record_id,
            key_id=record.key.key_id,
            model_state_sha256=record.key.model_state_sha256,
        )
        for record in sorted(records, key=lambda item: item.key.owner_id)
    )
    data = dict(
        schema_version=AUTHORITY_SCHEMA_VERSION,
        authority_sha256="0" * 64,
        development_only=True,
        final=False,
        final_family_access=False,
        plan_id=canonical_plan.plan_id,
        plan_parent_commit_sha=canonical_plan.parent_commit_sha,
        protocol_sha256=canonical_plan.protocol_sha256,
        protocol_self_sha256=str(snapshot.payload["diagnostic_protocol_sha256"]),
        protocol_file_sha256=snapshot.sha256,
        preparation_git_commit_sha=preparation_git_commit_sha,
        preparation_provenance_sha256=preparation_provenance_sha256,
        generation_git_commit_sha=generation_git_commit_sha,
        artifact_store_id=outcome_artifact_store_id(canonical_plan.plan_id),
        condition_ids=CONDITIONS,
        views=tuple(sorted(views, key=lambda item: item.view_id)),
        evidence=tuple(
            sorted(evidence_rows, key=lambda item: (item.heldout_family, item.replicate))
        ),
        artifacts=artifact_rows,
    )
    data["authority_sha256"] = _digest(
        _jsonable({key: value for key, value in data.items() if key != "authority_sha256"})
    )
    try:
        return OutcomeDiagnosticModelArtifactAuthority.model_validate(data)
    except (TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelArtifactError("constructed model authority is invalid") from exc


def validate_outcome_model_artifact_authority(
    authority: OutcomeDiagnosticModelArtifactAuthority,
    records: Sequence[OutcomeDiagnosticModelArtifactRecord],
    state_payloads: Mapping[str, PinnedOutcomeModelState],
    training_evidence_by_view: Mapping[str, PinnedOutcomeTrainingEvidence],
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    *,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
    generation_git_commit_sha: str,
) -> None:
    """Revalidate an opaque summary against every typed artifact and the plan.

    A loaded summary alone is intentionally insufficient: callers must provide
    the complete typed records and the non-optional protocol snapshot.
    """

    if not isinstance(authority, OutcomeDiagnosticModelArtifactAuthority):
        raise OutcomeDiagnosticModelArtifactError("model authority is not typed")
    try:
        OutcomeDiagnosticModelArtifactAuthority.model_validate(authority.model_dump(mode="json"))
    except (TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelArtifactError("model authority is not canonical") from exc
    preparation_git_commit_sha, preparation_provenance_sha256 = _require_preparation_identity(
        preparation_git_commit_sha, preparation_provenance_sha256
    )
    if (
        authority.preparation_git_commit_sha != preparation_git_commit_sha
        or authority.preparation_provenance_sha256 != preparation_provenance_sha256
        or authority.generation_git_commit_sha != generation_git_commit_sha
    ):
        raise OutcomeDiagnosticModelArtifactError("model authority generation provenance differs")
    canonical_plan, _fresh = _require_canonical_inputs(plan, snapshot)
    if authority.artifact_store_id != outcome_artifact_store_id(canonical_plan.plan_id):
        raise OutcomeDiagnosticModelArtifactError("model authority artifact store identity differs")
    expected = build_outcome_model_artifact_authority(
        records,
        state_payloads,
        training_evidence_by_view,
        plan,
        snapshot,
        preparation_git_commit_sha=preparation_git_commit_sha,
        preparation_provenance_sha256=preparation_provenance_sha256,
        generation_git_commit_sha=generation_git_commit_sha,
    )
    if authority != expected:
        raise OutcomeDiagnosticModelArtifactError(
            "model authority differs from canonical plan, records, or state"
        )


__all__ = [
    "ARCHITECTURE_ID",
    "AUTHORITY_SCHEMA_VERSION",
    "OUTCOME_ARTIFACT_STORE_PREFIX",
    "MODEL_SCHEMA_VERSION",
    "OutcomeDiagnosticModelArtifactError",
    "OutcomeTensorSchema",
    "OutcomeStateTensorPayload",
    "PinnedOutcomeModelState",
    "PinnedOutcomeTrainingEvidence",
    "AuthorizedOutcomeModelArtifact",
    "OutcomeTrainingAccounting",
    "OutcomeDiagnosticModelArtifactKey",
    "OutcomeRPModelArtifactKey",
    "OutcomePECModelArtifactKey",
    "OutcomeDiagnosticModelArtifactRecord",
    "OutcomeRPModelArtifactRecord",
    "OutcomePECModelArtifactRecord",
    "OutcomeDiagnosticModelArtifactAuthority",
    "build_outcome_model_artifact_key",
    "build_outcome_model_artifact_record",
    "build_outcome_rp_model_artifact_key",
    "build_outcome_pec_model_artifact_key",
    "build_outcome_rp_model_artifact_record",
    "build_outcome_pec_model_artifact_record",
    "validate_outcome_model_artifact_against_plan",
    "build_outcome_model_artifact_authority",
    "validate_outcome_model_artifact_authority",
    "canonical_outcome_model_artifact_key_bytes",
    "canonical_outcome_model_artifact_record_bytes",
    "canonical_outcome_model_artifact_authority_bytes",
    "load_outcome_model_artifact_key_bytes",
    "load_outcome_model_artifact_record_bytes",
    "load_outcome_model_artifact_authority_bytes",
    "inspect_outcome_model_state",
    "outcome_artifact_store_id",
]
