"""Versioned contracts for transfer-learning experiments."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

SCHEMA_VERSION = "0.1"


class ImprovementStage(BaseModel):
    """One ordered demonstration in a synthetic or empirical improvement ladder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    label: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    performance_value: float = Field(allow_inf_nan=False)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)


class ImprovementLadder(BaseModel):
    """Strictly improving demonstrations without pretending synthetic stages are humans."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    direction: Literal["minimize", "maximize"]
    stages: tuple[ImprovementStage, ...]

    @model_validator(mode="after")
    def stages_are_ordered_and_improving(self) -> "ImprovementLadder":
        if not self.stages:
            raise ValueError("improvement ladder must contain at least one stage")
        ordinals = [stage.ordinal for stage in self.stages]
        if ordinals != list(range(len(self.stages))):
            raise ValueError("stage ordinals must be contiguous and start at zero")
        ids = [stage.stage_id for stage in self.stages]
        if len(ids) != len(set(ids)):
            raise ValueError("stage_id values must be unique within a ladder")
        trajectory_ids = [stage.trajectory_id for stage in self.stages]
        if len(trajectory_ids) != len(set(trajectory_ids)):
            raise ValueError("trajectory_id values must be unique within a ladder")

        values = [stage.performance_value for stage in self.stages]
        if self.direction == "minimize":
            improving = all(later < earlier for earlier, later in zip(values, values[1:]))
        else:
            improving = all(later > earlier for earlier, later in zip(values, values[1:]))
        if not improving:
            raise ValueError("stage performance must improve strictly with ordinal")
        return self

    def stage(self, label: str) -> ImprovementStage:
        matches = [stage for stage in self.stages if stage.label == label]
        if len(matches) != 1:
            raise KeyError(f"expected exactly one stage labelled {label!r}")
        return matches[0]


class ExposureManifest(BaseModel):
    """What a learning condition was permitted to see before held-out evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    condition_id: str = Field(min_length=1)
    train_task_ids: tuple[str, ...]
    held_out_task_ids: tuple[str, ...]
    exposed_trajectory_ids: tuple[str, ...] = ()
    exposed_stage_labels: tuple[str, ...] = ()
    privileged_state_access: bool = False
    structured_constraint_access: bool = True
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def train_and_test_are_disjoint(self) -> "ExposureManifest":
        overlap = set(self.train_task_ids) & set(self.held_out_task_ids)
        if overlap:
            raise ValueError(f"training and held-out tasks overlap: {sorted(overlap)}")
        if len(self.exposed_trajectory_ids) != len(set(self.exposed_trajectory_ids)):
            raise ValueError("exposed_trajectory_ids must be unique")
        return self


class DiscoveryPoint(BaseModel):
    """Best valid performance found by a fixed candidate-evaluation budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    budget: int = Field(ge=1)
    best_performance: float | None = Field(default=None, allow_inf_nan=False)
    optimum_found: bool


class DiscoveryRun(BaseModel):
    """One seeded held-out search run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    seed: int
    optimum_value: float = Field(allow_inf_nan=False)
    first_optimum_episode: int | None = Field(default=None, ge=1)
    points: tuple[DiscoveryPoint, ...]
