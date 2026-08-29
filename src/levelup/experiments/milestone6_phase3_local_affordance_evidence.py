"""Immutable per-task raw-probe evidence for the local-affordance rung.

This is a serialization and sanitization boundary only.  It never opens an
environment or filesystem path.  Identity metadata is bound to the artifact,
then stripped before the reducer receives its 64 observable transitions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import (
    AffordanceTableRecord,
    ObservableStateRecord,
    ObservedTransitionRecord,
)
from levelup.learning.state_conditioned import (
    AffordanceTable,
    IndexedProbeRow,
    ObservableState,
    ObservedTransition,
    TaskLocalAffordanceEvidence,
    TaskProbeRows,
    bind_task_local_affordance_evidence,
    build_affordance_table,
)

HASH = Field(pattern=r"^[0-9a-f]{64}$")
ROWS_PER_ARTIFACT = 64
TARGET_SAMPLES_PER_ALIAS = 8
FAMILY_ORDER = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
_REDUCER_TOKEN = object()


class LocalAffordanceEvidenceError(ValueError):
    """Raised when evidence or its immutable identity fails closed."""


class RawProbeArtifactKey(BaseModel):
    """Every frozen input that can change one task's raw 64-row probe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["milestone6.phase3.raw-probe-artifact-key.v1"] = (
        "milestone6.phase3.raw-probe-artifact-key.v1"
    )
    phase: Literal["development"] = "development"
    local_affordance_protocol_sha256: str = HASH
    development_protocol_sha256: str = HASH
    development_tasks_sha256: str = HASH
    phase3_evidence_lock_sha256: str = HASH
    probe_policy_sha256: str = HASH
    family_id: str = Field(min_length=1)
    replicate: StrictInt = Field(ge=0, le=4)
    # This is the actual development-manifest task index used by the frozen seed formula,
    # not the task's ordinal position within the eight selected training-core tasks.
    task_index: StrictInt = Field(ge=0)
    task_id: str = Field(min_length=1)
    generator_seed: StrictInt = Field(ge=0)
    probe_seed: StrictInt = Field(ge=0)
    environment_seed: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def identity_is_development_safe(self) -> "RawProbeArtifactKey":
        if self.family_id not in FAMILY_ORDER:
            raise ValueError("raw probe family must be a known development family")
        if any(ord(char) < 32 for char in self.family_id + self.task_id):
            raise ValueError("raw probe identities cannot contain control characters")
        return self

    @property
    def key_id(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


class RawProbeTransitionRecord(BaseModel):
    """Exactly one observed transition plus its reducer-only canonical index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    probe_index: StrictInt = Field(ge=0, lt=ROWS_PER_ARTIFACT)
    before: ObservableStateRecord
    action_alias: str = Field(min_length=1)
    after: ObservableStateRecord
    completed: StrictBool

    @model_validator(mode="after")
    def action_is_learner_observable(self) -> "RawProbeTransitionRecord":
        if any(ord(char) < 32 for char in self.action_alias):
            raise ValueError("probe action alias contains a control character")
        if self.action_alias not in self.before.available_aliases:
            raise ValueError("probe action is unavailable in its pre-action observation")
        return self

    def transition_record(self) -> ObservedTransitionRecord:
        return ObservedTransitionRecord(
            before=self.before,
            action_alias=self.action_alias,
            after=self.after,
            completed=self.completed,
        )


RawProbeArtifactRow = RawProbeTransitionRecord
ProbeEvidenceRow = RawProbeTransitionRecord


class RawProbeArtifactBody(BaseModel):
    """Canonical task content with a self-checking SHA-256 digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["milestone6.phase3.raw-probe-artifact-body.v1"] = (
        "milestone6.phase3.raw-probe-artifact-body.v1"
    )
    rows: tuple[RawProbeTransitionRecord, ...]
    content_sha256: str = HASH

    @staticmethod
    def content_bytes(rows: Sequence[RawProbeTransitionRecord]) -> bytes:
        return canonical_json_bytes(
            {
                "schema_version": "milestone6.phase3.raw-probe-artifact-body.v1",
                "rows": [row.model_dump(mode="json") for row in rows],
            }
        )

    @model_validator(mode="after")
    def rows_and_digest_are_canonical(self) -> "RawProbeArtifactBody":
        if len(self.rows) != ROWS_PER_ARTIFACT:
            raise ValueError("raw probe artifacts require exactly 64 rows")
        if tuple(row.probe_index for row in self.rows) != tuple(range(ROWS_PER_ARTIFACT)):
            raise ValueError("probe indexes must be in canonical order 0..63")
        expected = hashlib.sha256(self.content_bytes(self.rows)).hexdigest()
        if self.content_sha256 != expected:
            raise ValueError("raw probe body content digest mismatch")
        return self

    @classmethod
    def from_rows(cls, rows: Sequence[RawProbeTransitionRecord]) -> "RawProbeArtifactBody":
        values = tuple(rows)
        digest = hashlib.sha256(cls.content_bytes(values)).hexdigest()
        return cls(rows=values, content_sha256=digest)


class RawProbeArtifactManifest(BaseModel):
    """Self-hashed key/body/pooled-table identity for one task artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["milestone6.phase3.raw-probe-artifact-manifest.v1"] = (
        "milestone6.phase3.raw-probe-artifact-manifest.v1"
    )
    artifact_id: str = HASH
    key_id: str = HASH
    key: RawProbeArtifactKey
    body_sha256: str = HASH
    pooled_affordance_sha256: str = HASH
    row_count: Literal[64] = 64

    @model_validator(mode="after")
    def identity_is_canonical(self) -> "RawProbeArtifactManifest":
        if self.key_id != self.key.key_id:
            raise ValueError("raw probe manifest key identity mismatch")
        expected = hashlib.sha256(
            canonical_json_bytes(self.model_dump(mode="json", exclude={"artifact_id"}))
        ).hexdigest()
        if self.artifact_id != expected:
            raise ValueError("raw probe manifest artifact identity mismatch")
        return self

    @classmethod
    def from_key_body(
        cls,
        key: RawProbeArtifactKey,
        body: RawProbeArtifactBody,
        *,
        pooled_affordance_sha256: str,
    ) -> "RawProbeArtifactManifest":
        unsigned = {
            "schema_version": "milestone6.phase3.raw-probe-artifact-manifest.v1",
            "key_id": key.key_id,
            "key": key.model_dump(mode="json"),
            "body_sha256": body.content_sha256,
            "pooled_affordance_sha256": pooled_affordance_sha256,
            "row_count": ROWS_PER_ARTIFACT,
        }
        artifact_id = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        return cls(artifact_id=artifact_id, **unsigned)


@dataclass(frozen=True, slots=True)
class _RawProbeReducerCapabilitySeal:
    artifact_id: str
    token: object


@dataclass(frozen=True, slots=True, init=False)
class RawProbeReducerCapability:
    artifact_id: str
    _seal: _RawProbeReducerCapabilitySeal
    _token: object

    def __init__(self, artifact_id: str, token: object) -> None:
        if token is not _REDUCER_TOKEN:
            raise ValueError("invalid raw probe reducer capability")
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(
            self,
            "_seal",
            _RawProbeReducerCapabilitySeal(artifact_id, _REDUCER_TOKEN),
        )
        object.__setattr__(self, "_token", _REDUCER_TOKEN)


def _capability_is_sealed(capability: object) -> bool:
    seal = getattr(capability, "_seal", None)
    return bool(
        type(capability) is RawProbeReducerCapability
        and getattr(capability, "_token", None) is _REDUCER_TOKEN
        and type(seal) is _RawProbeReducerCapabilitySeal
        and seal.token is _REDUCER_TOKEN
        and capability.artifact_id == seal.artifact_id
    )


def _affordance_bytes(record: AffordanceTableRecord) -> bytes:
    return canonical_json_bytes(record.model_dump(mode="json"))


def _validate_record(value: Any, model: type[BaseModel], label: str) -> BaseModel:
    if type(value) is not model:
        raise LocalAffordanceEvidenceError(f"{label} must have its exact typed model")
    try:
        return model.model_validate(value.model_dump(mode="json"))
    except (TypeError, ValueError) as exc:
        raise LocalAffordanceEvidenceError(f"{label} is invalid") from exc


@dataclass(frozen=True, slots=True)
class SanitizedRawProbeArtifact:
    key: RawProbeArtifactKey
    body: RawProbeArtifactBody
    manifest: RawProbeArtifactManifest
    affordances: AffordanceTableRecord
    _capability: RawProbeReducerCapability

    def __post_init__(self) -> None:
        key = _validate_record(self.key, RawProbeArtifactKey, "artifact key")
        body = _validate_record(self.body, RawProbeArtifactBody, "artifact body")
        manifest = _validate_record(self.manifest, RawProbeArtifactManifest, "artifact manifest")
        affordances = _validate_record(
            self.affordances, AffordanceTableRecord, "artifact pooled affordances"
        )
        if manifest.key != key or manifest.key_id != key.key_id:
            raise LocalAffordanceEvidenceError("artifact key and manifest identity mismatch")
        if manifest.body_sha256 != body.content_sha256:
            raise LocalAffordanceEvidenceError("artifact body and manifest identity mismatch")
        pooled_sha256 = hashlib.sha256(_affordance_bytes(affordances)).hexdigest()
        if manifest.pooled_affordance_sha256 != pooled_sha256:
            raise LocalAffordanceEvidenceError("artifact pooled table and manifest mismatch")
        if not _capability_is_sealed(self._capability):
            raise LocalAffordanceEvidenceError("artifact capability has the wrong type")
        if (
            self._capability.artifact_id != manifest.artifact_id
        ):
            raise LocalAffordanceEvidenceError("artifact capability identity mismatch")

    @property
    def reducer_capability(self) -> RawProbeReducerCapability:
        return self._capability


def _state_record(state: Any) -> ObservableStateRecord:
    return ObservableStateRecord(
        progress_fraction=state.progress_fraction,
        remaining_fraction=state.remaining_fraction,
        elapsed_per_target=state.elapsed_per_target,
        resource_fraction=state.resource_fraction,
        pressure_fraction=state.pressure_fraction,
        available_aliases=tuple(state.available_aliases),
    )


def _observable_state(record: ObservableStateRecord) -> ObservableState:
    return ObservableState(
        progress_fraction=record.progress_fraction,
        remaining_fraction=record.remaining_fraction,
        elapsed_per_target=record.elapsed_per_target,
        resource_fraction=record.resource_fraction,
        pressure_fraction=record.pressure_fraction,
        available_aliases=record.available_aliases,
    )


def _observed_transition(row: RawProbeTransitionRecord) -> ObservedTransition:
    return ObservedTransition(
        before=_observable_state(row.before),
        action_alias=row.action_alias,
        after=_observable_state(row.after),
        completed=row.completed,
    )


def _table_record(table: AffordanceTable) -> AffordanceTableRecord:
    return AffordanceTableRecord(
        features={alias: tuple(values) for alias, values in table.features.items()},
        sample_counts=dict(table.sample_counts),
    )


def _core_table(record: AffordanceTableRecord) -> AffordanceTable:
    return AffordanceTable(
        features={alias: tuple(values) for alias, values in record.features.items()},
        sample_counts=dict(record.sample_counts),
    )


def sanitize_probe_evidence(
    evidence: Any,
    *,
    local_affordance_protocol_sha256: str,
    development_protocol_sha256: str,
    development_tasks_sha256: str,
    phase3_evidence_lock_sha256: str,
    probe_policy_sha256: str,
    family_id: str,
    replicate: int,
    task_index: int,
    task_id: str,
    generator_seed: int,
    probe_seed: int,
    environment_seed: int,
    canonical_affordances: AffordanceTableRecord,
) -> SanitizedRawProbeArtifact:
    """Canonicalize one paid probe and require exact v1 pooled-table parity."""

    from levelup.experiments.milestone6_baselines import ProbeEvidence

    if type(evidence) is not ProbeEvidence:
        raise LocalAffordanceEvidenceError("sanitization requires a real ProbeEvidence")
    if type(canonical_affordances) is not AffordanceTableRecord:
        raise LocalAffordanceEvidenceError("canonical v1 pooled affordance table is required")
    if evidence.task_id != task_id:
        raise LocalAffordanceEvidenceError("probe evidence task identity mismatch")
    if type(evidence.accounting.actions) is not int or evidence.accounting.actions != 64:
        raise LocalAffordanceEvidenceError("probe accounting must charge exactly 64 actions")
    if len(evidence.transitions) != 64:
        raise LocalAffordanceEvidenceError("probe evidence must contain exactly 64 transitions")

    rows = tuple(
        RawProbeTransitionRecord(
            probe_index=index,
            before=_state_record(transition.before),
            action_alias=transition.action_alias,
            after=_state_record(transition.after),
            completed=transition.completed,
        )
        for index, transition in enumerate(evidence.transitions)
    )
    body = RawProbeArtifactBody.from_rows(rows)
    rebuilt = _table_record(
        build_affordance_table(
            tuple(_observed_transition(row) for row in rows),
            target_samples_per_alias=TARGET_SAMPLES_PER_ALIAS,
        )
    )
    rebuilt_bytes = _affordance_bytes(rebuilt)
    if rebuilt_bytes != _affordance_bytes(_table_record(evidence.affordances)):
        raise LocalAffordanceEvidenceError("ProbeEvidence pooled affordance table drift")
    if rebuilt_bytes != _affordance_bytes(canonical_affordances):
        raise LocalAffordanceEvidenceError("canonical v1 pooled affordance byte parity mismatch")
    expected_aliases = tuple(sorted(rebuilt.features))
    if evidence.accounting.discovered_aliases != expected_aliases:
        raise LocalAffordanceEvidenceError("probe discovered-alias accounting mismatch")

    key = RawProbeArtifactKey(
        local_affordance_protocol_sha256=local_affordance_protocol_sha256,
        development_protocol_sha256=development_protocol_sha256,
        development_tasks_sha256=development_tasks_sha256,
        phase3_evidence_lock_sha256=phase3_evidence_lock_sha256,
        probe_policy_sha256=probe_policy_sha256,
        family_id=family_id,
        replicate=replicate,
        task_index=task_index,
        task_id=task_id,
        generator_seed=generator_seed,
        probe_seed=probe_seed,
        environment_seed=environment_seed,
    )
    pooled_sha256 = hashlib.sha256(rebuilt_bytes).hexdigest()
    manifest = RawProbeArtifactManifest.from_key_body(
        key, body, pooled_affordance_sha256=pooled_sha256
    )
    capability = RawProbeReducerCapability(manifest.artifact_id, _REDUCER_TOKEN)
    return SanitizedRawProbeArtifact(key, body, manifest, rebuilt, capability)


def task_local_affordance_evidence_from_artifact(
    artifact: SanitizedRawProbeArtifact,
    capability: RawProbeReducerCapability,
) -> TaskLocalAffordanceEvidence:
    """Strip all identities and return the parity-bound core reducer capability."""

    if type(artifact) is not SanitizedRawProbeArtifact:
        raise LocalAffordanceEvidenceError("typed sanitized artifact required")
    # Re-run the aggregate invariant checks in case a forged dataclass bypassed init.
    artifact.__post_init__()
    if not _capability_is_sealed(capability):
        raise LocalAffordanceEvidenceError("sanitizer-issued reducer capability required")
    if (
        capability.artifact_id != artifact.manifest.artifact_id
        or capability != artifact.reducer_capability
    ):
        raise LocalAffordanceEvidenceError("reducer capability does not authorize this artifact")
    rows = TaskProbeRows(
        tuple(
            IndexedProbeRow(row.probe_index, _observed_transition(row))
            for row in artifact.body.rows
        )
    )
    return bind_task_local_affordance_evidence(rows, _core_table(artifact.affordances))


def reducer_payload_from_artifact(
    artifact: SanitizedRawProbeArtifact,
    capability: RawProbeReducerCapability,
) -> TaskProbeRows:
    """Compatibility helper returning only the core's identity-free task rows."""

    return task_local_affordance_evidence_from_artifact(artifact, capability).task_rows


sanitize_raw_probe_evidence = sanitize_probe_evidence
build_reducer_payload = reducer_payload_from_artifact


__all__ = [
    "FAMILY_ORDER",
    "LocalAffordanceEvidenceError",
    "ProbeEvidenceRow",
    "RawProbeArtifactBody",
    "RawProbeArtifactKey",
    "RawProbeArtifactManifest",
    "RawProbeArtifactRow",
    "RawProbeReducerCapability",
    "RawProbeTransitionRecord",
    "SanitizedRawProbeArtifact",
    "build_reducer_payload",
    "reducer_payload_from_artifact",
    "sanitize_probe_evidence",
    "sanitize_raw_probe_evidence",
    "task_local_affordance_evidence_from_artifact",
]
