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
    censored: bool = False
    censoring_budget: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> "UnitOutcome":
        if self.valid and not self.evaluator_ran:
            raise ValueError("validity requires an independent evaluator run")
        if self.success and not (self.evaluator_ran and self.valid and self.completed):
            raise ValueError("success requires evaluator_ran, valid, and completed")
        if self.valid and self.completed and self.performance_value is None:
            raise ValueError("valid completed outcomes require a performance value")
        if self.first_valid_completion_episode is not None and not (
            self.valid and self.completed
        ):
            raise ValueError("first valid completion requires a valid completed outcome")
        if self.first_threshold_episode is not None and not self.success:
            raise ValueError("first threshold requires success")
        if self.first_optimum_episode is not None and not self.success:
            raise ValueError("first optimum requires success")
        if self.censored != (self.censoring_budget is not None):
            raise ValueError("censored and censoring_budget must be set together")
        if self.censored and self.success:
            raise ValueError("a successful outcome cannot be censored")
        return self


class UnitPayload(BaseModel):
    """Experiment-specific executor output before runner provenance is attached."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: UnitOutcome
    accounting: ResourceAccounting
    diagnostics: dict[str, DiagnosticValue] = Field(default_factory=dict)

    @field_validator("diagnostics")
    @classmethod
    def diagnostic_floats_are_finite(
        cls, diagnostics: dict[str, DiagnosticValue]
    ) -> dict[str, DiagnosticValue]:
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in diagnostics.values()
        ):
            raise ValueError("diagnostic float values must be finite")
        return diagnostics


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
    diagnostics: dict[str, DiagnosticValue] = Field(default_factory=dict)

    @field_validator("diagnostics")
    @classmethod
    def diagnostic_floats_are_finite(
        cls, diagnostics: dict[str, DiagnosticValue]
    ) -> dict[str, DiagnosticValue]:
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in diagnostics.values()
        ):
            raise ValueError("diagnostic float values must be finite")
        return diagnostics

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


def unit_id_for(key: UnitKey) -> str:
    return hashlib.sha256(canonical_json_bytes(key.model_dump(mode="json"))).hexdigest()
