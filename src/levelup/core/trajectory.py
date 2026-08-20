"""Replayable action trajectories."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

SCHEMA_VERSION = "0.1"


class ActionRecord(BaseModel):
    """Environment-independent representation of one agent action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class TrajectoryStep(BaseModel):
    """One ordered action and optional hashes for audit and replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    action: ActionRecord
    observation_hash: str | None = None
    state_hash: str | None = None


class Trajectory(BaseModel):
    """Canonical action trace for an agent or reference run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    source: Literal["agent", "reference", "oracle"]
    environment_seed: int | None = None
    steps: tuple[TrajectoryStep, ...]
    final_state_hash: str | None = None

    @model_validator(mode="after")
    def steps_are_contiguous(self) -> "Trajectory":
        indices = [step.index for step in self.steps]
        if indices != list(range(len(indices))):
            raise ValueError("trajectory step indices must be contiguous and start at zero")
        return self
