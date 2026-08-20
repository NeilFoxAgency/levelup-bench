"""Canonical task specification for LevelUp Bench."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

SCHEMA_VERSION = "0.1"


class EnvironmentSpec(BaseModel):
    """Identity and public configuration of the environment used by a task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter: str = Field(min_length=1)
    environment_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    seed: int | None = None
    configuration: dict[str, JsonValue] = Field(default_factory=dict)


class ConstraintSpec(BaseModel):
    """A hard natural-language rule paired with a machine verifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    constraint_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    verifier_id: str = Field(min_length=1)
    verifier_config: dict[str, JsonValue] = Field(default_factory=dict)
    hard: Literal[True] = True


class ObjectiveSpec(BaseModel):
    """Primary performance objective for valid completed runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_id: str = Field(min_length=1)
    direction: Literal["minimize", "maximize"]
    unit: str = Field(min_length=1)


class TaskSpec(BaseModel):
    """Single source of truth for a benchmark task.

    In v0.1, every declared constraint is hard. Performance is compared only
    after a run is valid and complete.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    environment: EnvironmentSpec
    instruction: str = Field(min_length=1)
    constraints: tuple[ConstraintSpec, ...] = ()
    objective: ObjectiveSpec
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def constraint_ids_are_unique(self) -> "TaskSpec":
        ids = [constraint.constraint_id for constraint in self.constraints]
        if len(ids) != len(set(ids)):
            raise ValueError("constraint_id values must be unique within a task")
        return self
