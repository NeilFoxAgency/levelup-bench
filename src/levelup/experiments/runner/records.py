"""Validated run plans, raw unit records, and aggregate envelopes."""

from __future__ import annotations

import hashlib
import math
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from levelup.experiments.runner.config import canonical_json_bytes

SplitPhase = Literal["development", "validation", "final"]
DiagnosticValue = StrictBool | StrictInt | StrictFloat | None


class UnitKey(BaseModel):
    """Scientific identity of one condition/task/replicate evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: SplitPhase
    condition_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    task_index: int = Field(ge=0)
    replicate: int = Field(ge=0)


class UnitSeeds(BaseModel):
    """Every random channel resolved for one atomic unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_seed: int
    environment_seed: int
    probe_seed: int
    search_seed: int
    data_order_seed: int


class PlannedUnit(BaseModel):
    """One expected unit committed before execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    key: UnitKey
    seeds: UnitSeeds
    exposure_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExpectedUnits(BaseModel):
    """Complete deterministic matrix used to detect missing or extra outcomes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase1.expected.v1"] = "phase1.expected.v1"
    run_id: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    units: tuple[PlannedUnit, ...]

    @model_validator(mode="after")
    def units_are_unique(self) -> "ExpectedUnits":
        unit_ids = [unit.unit_id for unit in self.units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("expected unit_id values must be unique")
        keys = [unit.key.model_dump_json() for unit in self.units]
        if len(keys) != len(set(keys)):
            raise ValueError("expected unit keys must be unique")
        return self


class PlannedSharedArtifact(BaseModel):
    """One shared artifact and its exact atomic-unit consumers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["training_artifact", "training_data_evidence", "training_data_view"] = (
        "training_artifact"
    )
    key_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    owner_condition_id: str = Field(min_length=1)
    owner_group_id: str | None = Field(default=None, min_length=1)
    owner_family_id: str = Field(min_length=1)
    owner_fold_id: str = Field(min_length=1)
    owner_replicate: int = Field(ge=0)
    consumer_phase: SplitPhase
    consumer_condition_ids: tuple[str, ...]
    consumer_unit_ids: tuple[str, ...]

    @model_validator(mode="after")
    def consumers_are_unique(self) -> "PlannedSharedArtifact":
        if not self.consumer_unit_ids or len(set(self.consumer_unit_ids)) != len(
            self.consumer_unit_ids
        ):
            raise ValueError("shared artifact consumers must be unique and non-empty")
        if not self.consumer_condition_ids or len(set(self.consumer_condition_ids)) != len(
            self.consumer_condition_ids
        ):
            raise ValueError("shared artifact consumer conditions must be unique and non-empty")
        return self


class ExpectedSharedArtifacts(BaseModel):
    """Immutable shared-artifact plan bound to one run and config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runner.expected-shared.v1"] = "runner.expected-shared.v1"
    run_id: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[PlannedSharedArtifact, ...] = ()

    @model_validator(mode="after")
    def keys_are_unique(self) -> "ExpectedSharedArtifacts":
        keys = [(item.kind, item.key_id) for item in self.artifacts]
        if len(keys) != len(set(keys)):
            raise ValueError("shared artifact kind/key pairs must be unique")
        for kind in {item.kind for item in self.artifacts}:
            consumers = [
                unit_id
                for artifact in self.artifacts
                if artifact.kind == kind
                for unit_id in artifact.consumer_unit_ids
            ]
            if len(consumers) != len(set(consumers)):
                raise ValueError("one unit cannot consume multiple shared artifacts of one kind")
        return self


class SharedArtifactReference(BaseModel):
    """Unit-level reference to a validated shared artifact and its cost."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["training_artifact", "training_data_evidence", "training_data_view"] = (
        "training_artifact"
    )
    key_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    cost_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    consumer_accounting_scope: Literal["heldout_task_only"] = "heldout_task_only"


_SHARED_KIND_ORDER = {
    "training_data_evidence": 0,
    "training_data_view": 1,
    "training_artifact": 2,
}


def _validate_unit_shared_references(
    legacy: SharedArtifactReference | None,
    references: tuple[SharedArtifactReference, ...],
) -> None:
    if legacy is not None and references:
        raise ValueError("legacy shared_artifact cannot be combined with shared_artifacts")
    if legacy is not None and legacy.kind != "training_artifact":
        raise ValueError("legacy shared_artifact accepts only training artifacts")
    if len({item.kind for item in references}) != len(references):
        raise ValueError("a unit may have at most one shared artifact per kind")
    if references != tuple(sorted(references, key=lambda item: _SHARED_KIND_ORDER[item.kind])):
        raise ValueError("shared artifacts must use canonical kind order")
    kinds = {item.kind for item in references}
    if "training_data_view" in kinds and "training_data_evidence" not in kinds:
        raise ValueError("a training-data view requires its evidence reference")
    if "training_artifact" in kinds and "training_data_view" not in kinds:
        raise ValueError("a typed training artifact requires its training-data view")


class PhaseAccounting(BaseModel):
    """Multidimensional resource accounting for one execution phase."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    calls: int = Field(default=0, ge=0)
    episodes: int = Field(default=0, ge=0)
    actions: int = Field(default=0, ge=0)
    environment_steps: int = Field(default=0, ge=0)
    resets: int = Field(default=0, ge=0)
    forward_passes: int = Field(default=0, ge=0)
    optimizer_steps: int = Field(default=0, ge=0)
    nodes_expanded: int = Field(default=0, ge=0)
    wall_seconds: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class ResourceAccounting(BaseModel):
    """Keep setup, learning, search, replay, and evaluator costs visible."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    setup: PhaseAccounting = Field(default_factory=PhaseAccounting)
    probes: PhaseAccounting = Field(default_factory=PhaseAccounting)
    training: PhaseAccounting = Field(default_factory=PhaseAccounting)
    search: PhaseAccounting = Field(default_factory=PhaseAccounting)
    replay: PhaseAccounting = Field(default_factory=PhaseAccounting)
    evaluator: PhaseAccounting = Field(default_factory=PhaseAccounting)
    serialization: PhaseAccounting = Field(default_factory=PhaseAccounting)


class TrainingPreparationAccounting(BaseModel):
    """One-time model preparation costs, separate from held-out task execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    setup: PhaseAccounting = Field(default_factory=PhaseAccounting)
    training_probes: PhaseAccounting = Field(default_factory=PhaseAccounting)
    reference_replay: PhaseAccounting = Field(default_factory=PhaseAccounting)
    training: PhaseAccounting = Field(default_factory=PhaseAccounting)
    serialization: PhaseAccounting = Field(default_factory=PhaseAccounting)

    def as_resource_accounting(self) -> ResourceAccounting:
        return ResourceAccounting(
            setup=self.setup,
            probes=self.training_probes,
            training=self.training,
            replay=self.reference_replay,
            serialization=self.serialization,
        )


class TrainingArtifactCostRecord(BaseModel):
    """Torch-free first-writer cost record for a shared artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runner.training-artifact-cost.v1", "runner.training-artifact-cost.v2"]
    cost_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    key_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope: Literal["training_preparation"] | None = None
    key: dict[str, JsonValue]
    accounting: ResourceAccounting | TrainingPreparationAccounting

    @property
    def expected_cost_id(self) -> str:
        body = self.model_dump(mode="json", exclude={"cost_id"})
        if self.schema_version == "runner.training-artifact-cost.v1":
            body.pop("scope", None)
        return hashlib.sha256(canonical_json_bytes(body)).hexdigest()

    @model_validator(mode="after")
    def digest_is_valid(self) -> "TrainingArtifactCostRecord":
        if (
            self.schema_version == "runner.training-artifact-cost.v1"
            and (self.scope is not None or not isinstance(self.accounting, ResourceAccounting))
        ) or (
            self.schema_version == "runner.training-artifact-cost.v2"
            and (
                self.scope != "training_preparation"
                or not isinstance(self.accounting, TrainingPreparationAccounting)
            )
        ):
            raise ValueError("cost scope does not match schema version")
        if self.key_id != hashlib.sha256(canonical_json_bytes(self.key)).hexdigest():
            raise ValueError("cost key ID does not match canonical key")
        if self.cost_id != self.expected_cost_id:
            raise ValueError("cost ID does not match canonical cost body")
        return self


class UnitOutcome(BaseModel):
    """Validity, completion, performance, and discovery remain separate dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluator_ran: bool
    valid: bool
    completed: bool
    success: bool
    quality_value: float | None = Field(default=None, allow_inf_nan=False)
    performance_metric_id: str = Field(min_length=1)
    performance_value: float | None = Field(default=None, allow_inf_nan=False)
    performance_direction: Literal["minimize", "maximize"]
    first_valid_completion_episode: int | None = Field(default=None, ge=1)
    first_threshold_episode: int | None = Field(default=None, ge=1)
    first_optimum_episode: int | None = Field(default=None, ge=1)
    first_optimum_adaptation_actions: int | None = Field(default=None, ge=0)
    censored: bool = False
    censoring_budget: int | None = Field(default=None, ge=1)
    censoring_reason: Literal["fixed_endpoint"] | None = None

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> "UnitOutcome":
        if self.valid and not self.evaluator_ran:
            raise ValueError("validity requires an independent evaluator run")
        if self.success and not (self.evaluator_ran and self.valid and self.completed):
            raise ValueError("success requires evaluator_ran, valid, and completed")
        if self.valid and self.completed and self.performance_value is None:
            raise ValueError("valid completed outcomes require a performance value")
        if self.first_valid_completion_episode is not None and not (self.valid and self.completed):
            raise ValueError("first valid completion requires a valid completed outcome")
        if self.first_threshold_episode is not None and not self.success:
            raise ValueError("first threshold requires success")
        if self.first_optimum_episode is not None and not self.success:
            raise ValueError("first optimum requires success")
        if self.first_optimum_adaptation_actions is not None and not self.success:
            raise ValueError("first optimum adaptation actions require success")
        if self.censored != (self.censoring_budget is not None):
            raise ValueError("censored and censoring_budget must be set together")
        if self.censoring_reason is not None and not self.censored:
            raise ValueError("censoring reason requires a censored outcome")
        if self.censored and self.success:
            raise ValueError("a successful outcome cannot be censored")
        return self


class UnitPayload(BaseModel):
    """Experiment-specific executor output before runner provenance is attached."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: UnitOutcome
    accounting: ResourceAccounting
    shared_artifact: SharedArtifactReference | None = None
    shared_artifacts: tuple[SharedArtifactReference, ...] = ()
    candidate_generation_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    history_shuffle_permutation_map_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    diagnostics: dict[str, DiagnosticValue] = Field(default_factory=dict)

    @field_validator("diagnostics")
    @classmethod
    def diagnostic_floats_are_finite(
        cls, diagnostics: dict[str, DiagnosticValue]
    ) -> dict[str, DiagnosticValue]:
        if any(
            isinstance(value, float) and not math.isfinite(value) for value in diagnostics.values()
        ):
            raise ValueError("diagnostic float values must be finite")
        return diagnostics

    @model_validator(mode="after")
    def shared_kinds_are_unique(self) -> "UnitPayload":
        _validate_unit_shared_references(self.shared_artifact, self.shared_artifacts)
        return self


class UnitRecord(BaseModel):
    """Validated completed atomic result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase1.unit.v1"] = "phase1.unit.v1"
    run_id: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    unit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    key: UnitKey
    seeds: UnitSeeds
    exposure_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["completed"] = "completed"
    started_at_utc: AwareDatetime
    finished_at_utc: AwareDatetime
    elapsed_wall_seconds: float = Field(ge=0, allow_inf_nan=False)
    outcome: UnitOutcome
    accounting: ResourceAccounting
    shared_artifact: SharedArtifactReference | None = None
    shared_artifacts: tuple[SharedArtifactReference, ...] = ()
    candidate_generation_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    history_shuffle_permutation_map_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    diagnostics: dict[str, DiagnosticValue] = Field(default_factory=dict)

    @field_validator("diagnostics")
    @classmethod
    def diagnostic_floats_are_finite(
        cls, diagnostics: dict[str, DiagnosticValue]
    ) -> dict[str, DiagnosticValue]:
        if any(
            isinstance(value, float) and not math.isfinite(value) for value in diagnostics.values()
        ):
            raise ValueError("diagnostic float values must be finite")
        return diagnostics

    @model_validator(mode="after")
    def shared_kinds_are_unique(self) -> "UnitRecord":
        _validate_unit_shared_references(self.shared_artifact, self.shared_artifacts)
        return self

    @model_validator(mode="after")
    def timestamps_are_ordered(self) -> "UnitRecord":
        if self.finished_at_utc < self.started_at_utc:
            raise ValueError("unit finish timestamp precedes start timestamp")
        return self


class AttemptRecord(BaseModel):
    """Durable failure or interruption without unbounded traceback content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase1.attempt.v1"] = "phase1.attempt.v1"
    run_id: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    unit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt: int = Field(ge=1)
    key: UnitKey
    seeds: UnitSeeds
    status: Literal["failed", "interrupted"]
    stage: str = Field(min_length=1)
    exception_type: str = Field(min_length=1)
    sanitized_message: str = Field(max_length=500)
    retryable: bool
    started_at_utc: AwareDatetime
    finished_at_utc: AwareDatetime
    elapsed_wall_seconds: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def timestamps_are_ordered(self) -> "AttemptRecord":
        if self.finished_at_utc < self.started_at_utc:
            raise ValueError("attempt finish timestamp precedes start timestamp")
        return self


class SystemProvenance(BaseModel):
    """Reproducibility metadata with secret-bearing process state excluded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase1.provenance.v1"] = "phase1.provenance.v1"
    git_commit_sha: str = Field(min_length=1)
    git_dirty: bool
    git_diff_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    python_version: str = Field(min_length=1)
    packages: dict[str, str]
    installed_packages_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    os: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    cpu: str
    cpu_count: int | None = Field(default=None, ge=1)
    memory_bytes: int | None = Field(default=None, ge=1)
    requested_device: str = Field(min_length=1)
    resolved_device: str = Field(min_length=1)
    requested_torch_threads: int = Field(ge=1)
    actual_torch_threads: int | None = Field(default=None, ge=1)
    requested_torch_interop_threads: int = Field(ge=1)
    actual_torch_interop_threads: int | None = Field(default=None, ge=1)
    deterministic_algorithms_requested: bool
    deterministic_algorithms_actual: bool | None = None
    processes: int = Field(ge=1)
    captured_at_utc: AwareDatetime

    @model_validator(mode="after")
    def dirty_state_has_hash(self) -> "SystemProvenance":
        if self.git_dirty != (self.git_diff_sha256 is not None):
            raise ValueError("dirty provenance requires exactly one diff hash")
        return self


class Inventory(BaseModel):
    """Terminal and nonterminal unit inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected: int = Field(ge=0)
    completed: int = Field(ge=0)
    missing: int = Field(ge=0)
    units_with_failed_attempts: int = Field(ge=0)
    units_with_interrupted_attempts: int = Field(ge=0)
    failed_attempts: int = Field(ge=0)
    interrupted_attempts: int = Field(ge=0)


class SharedArtifactInventory(BaseModel):
    """Completion state for the frozen shared-artifact plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    planned: int = Field(default=0, ge=0)
    referenced: int = Field(default=0, ge=0)
    complete: bool = True


class AggregateSlice(BaseModel):
    """Visible outcome vectors rather than a universal scalar."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    completed_units: int = Field(ge=0)
    valid_units: int = Field(ge=0)
    successful_units: int = Field(ge=0)
    performance_values: tuple[float, ...]
    probe_actions: int = Field(ge=0)
    search_actions: int = Field(ge=0)
    replay_actions: int = Field(ge=0)
    forward_passes: int = Field(ge=0)
    optimizer_steps: int = Field(ge=0)
    wall_seconds: float = Field(ge=0, allow_inf_nan=False)


class AggregateArtifact(BaseModel):
    """Deterministic summary generated only from validated raw records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase1.aggregate.v1"] = "phase1.aggregate.v1"
    run_id: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_units_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_units_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_started_at_utc: AwareDatetime | None = None
    run_finished_at_utc: AwareDatetime | None = None
    observed_span_seconds: float = Field(ge=0, allow_inf_nan=False)
    complete: bool
    paired_seed_audit_passed: bool
    inventory: Inventory
    by_phase_condition: dict[SplitPhase, dict[str, AggregateSlice]]
    by_phase_family: dict[SplitPhase, dict[str, AggregateSlice]]
    shared_artifacts_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    shared_inventory: SharedArtifactInventory = Field(default_factory=SharedArtifactInventory)
    shared_accounting_by_owner_group: dict[str, ResourceAccounting] = Field(default_factory=dict)


def unit_id_for(key: UnitKey) -> str:
    return hashlib.sha256(canonical_json_bytes(key.model_dump(mode="json"))).hexdigest()
