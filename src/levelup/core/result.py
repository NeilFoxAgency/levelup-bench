"""Benchmark result schema and validity-gated comparison semantics."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from levelup.core.task import TaskSpec

SCHEMA_VERSION = "0.1"


class ConstraintOutcome(BaseModel):
    """Verifier result for one hard task constraint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    constraint_id: str = Field(min_length=1)
    passed: bool
    evidence: str | None = None


class EfficiencyMetrics(BaseModel):
    """Resource use kept separate from correctness and performance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    environment_steps: int = Field(ge=0)
    agent_actions: int = Field(ge=0)
    wall_time_seconds: float = Field(ge=0, allow_inf_nan=False)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class BenchmarkResult(BaseModel):
    """Outcome of one evaluated trajectory.

    Validity is task-relative. A result is performance-eligible only when its
    task matches, every declared hard constraint has exactly one passing
    outcome, and the task completed. Efficiency never compensates for invalidity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    completed: bool
    constraint_outcomes: tuple[ConstraintOutcome, ...] = ()
    quality_value: float | None = Field(default=None, allow_inf_nan=False)
    performance_value: float | None = Field(default=None, allow_inf_nan=False)
    efficiency: EfficiencyMetrics
    final_state_hash: str | None = None

    @model_validator(mode="after")
    def constraint_ids_are_unique(self) -> "BenchmarkResult":
        ids = [outcome.constraint_id for outcome in self.constraint_outcomes]
        if len(ids) != len(set(ids)):
            raise ValueError("constraint outcomes must contain unique constraint_id values")
        return self

    def valid_for(self, task: TaskSpec) -> bool:
        """Return True only when the result completely satisfies this task."""

        if self.task_id != task.task_id:
            return False

        expected = {constraint.constraint_id for constraint in task.constraints}
        observed = {outcome.constraint_id for outcome in self.constraint_outcomes}
        if observed != expected:
            return False

        return all(outcome.passed for outcome in self.constraint_outcomes)

    def performance_eligible_for(self, task: TaskSpec) -> bool:
        """Whether the run may be compared on quality, performance, or efficiency."""

        return self.completed and self.valid_for(task)
