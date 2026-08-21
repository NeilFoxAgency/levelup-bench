"""Strict scientific experiment configuration and deterministic identities."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

SCHEMA_VERSION = "phase1.config.v1"
_SAFE_PREFIX = re.compile(r"[^a-z0-9]+")


class TrajectoryIdentity(BaseModel):
    """One reference trajectory whose task and stage identity are predeclared."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_label: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    source: Literal[
        "synthetic-reference",
        "human",
        "world-record",
        "tas",
        "oracle",
        "other",
    ]
    provenance: dict[str, JsonValue] = Field(default_factory=dict)


class TaskIdentity(BaseModel):
    """One exact task identity committed to a development/evaluation split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    task_index: int = Field(ge=0)
    generator_seed: int
    environment_reset_seed: int = 0
    trajectory_catalog: tuple[TrajectoryIdentity, ...] = ()

    @model_validator(mode="after")
    def trajectory_identities_are_unique(self) -> "TaskIdentity":
        labels = [item.stage_label for item in self.trajectory_catalog]
        trajectory_ids = [item.trajectory_id for item in self.trajectory_catalog]
        if len(labels) != len(set(labels)):
            raise ValueError("trajectory stage labels must be unique within a task")
        if len(trajectory_ids) != len(set(trajectory_ids)):
            raise ValueError("trajectory IDs must be unique within a task")
        return self


class ExposedTrajectory(BaseModel):
    """One exact task/stage/trajectory binding visible to a condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    stage_label: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)


class ExposureSpec(BaseModel):
    """Complete declared information exposure for one learning condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    train_task_ids: tuple[str, ...]
    exposed_trajectories: tuple[ExposedTrajectory, ...] = ()
    observable_state_access: Literal["none", "current", "history"]
    action_history_access: bool
    action_descriptors_access: bool
    probe_interaction_access: bool
    search_feedback_access: bool
    evaluator_output_access: bool
    optimum_threshold_access: bool
    privileged_state_access: bool
    structured_constraint_access: bool
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def identities_are_unique(self) -> "ExposureSpec":
        if len(self.train_task_ids) != len(set(self.train_task_ids)):
            raise ValueError("exposure train_task_ids must be unique")
        trajectory_ids = [item.trajectory_id for item in self.exposed_trajectories]
        if len(trajectory_ids) != len(set(trajectory_ids)):
            raise ValueError("exposed trajectory IDs must be unique")
        return self


class SplitSpec(BaseModel):
    """Exact task identities separated by experimental role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    development_tasks: tuple[TaskIdentity, ...]
    validation_tasks: tuple[TaskIdentity, ...] = ()
    final_tasks: tuple[TaskIdentity, ...] = ()

    @model_validator(mode="after")
    def task_ids_are_unique_and_disjoint(self) -> "SplitSpec":
        groups = {
            "development": self.development_tasks,
            "validation": self.validation_tasks,
            "final": self.final_tasks,
        }
        ids_by_group = {
            name: {task.task_id for task in tasks} for name, tasks in groups.items()
        }
        for name, tasks in groups.items():
            if len(tasks) != len(ids_by_group[name]):
                raise ValueError(f"duplicate task_id in {name} split")
        for left, right in (
            ("development", "validation"),
            ("development", "final"),
            ("validation", "final"),
        ):
            overlap = ids_by_group[left] & ids_by_group[right]
            if overlap:
                raise ValueError(f"{left} and {right} tasks overlap: {sorted(overlap)}")
        if not self.development_tasks:
            raise ValueError("at least one development task is required")
        if any(task.trajectory_catalog for task in self.validation_tasks):
            raise ValueError("validation task trajectory catalogs must be empty")
        if any(task.trajectory_catalog for task in self.final_tasks):
            raise ValueError("final task trajectory catalogs must be empty")
        return self


class ConditionSpec(BaseModel):
    """One condition and the complete information exposure it receives."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition_id: str = Field(min_length=1)
    learner_id: str = Field(min_length=1)
    execution_phases: tuple[
        Literal["development", "validation", "final"], ...
    ] = ("development", "validation", "final")
    exposure: ExposureSpec
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def execution_phases_are_unique(self) -> "ConditionSpec":
        if not self.execution_phases:
            raise ValueError("condition execution_phases cannot be empty")
        if len(self.execution_phases) != len(set(self.execution_phases)):
            raise ValueError("condition execution_phases must be unique")
        return self


class SeedPolicy(BaseModel):
    """Versioned deterministic seed derivation for atomic units."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    derivation_version: Literal["phase1.v1"] = "phase1.v1"
    model_seed_base: int
    environment_seed_offset: int = 0
    probe_seed_base: int
    search_seed_base: int
    data_order_seed_base: int
    replicate_stride: int = Field(default=100_000, ge=1)


class DevicePolicy(BaseModel):
    """Requested compute condition; resolved hardware is recorded as provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_device: Literal["cpu", "mps", "cuda"]
    torch_threads: int = Field(ge=1)
    torch_interop_threads: int = Field(default=1, ge=1)
    processes: Literal[1] = 1
    deterministic_algorithms: bool = False


class MetricSpec(BaseModel):
    """One visible metric; no universal weighted score is implied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_id: str = Field(min_length=1)
    direction: Literal["minimize", "maximize", "none"]
    unit: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SelectionSpec(BaseModel):
    """Development-only selection rule committed before model tuning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phases: tuple[Literal["development", "validation"], ...]
    primary_metric: str = Field(min_length=1)
    rule: str = Field(min_length=1)

    @model_validator(mode="after")
    def phases_are_unique(self) -> "SelectionSpec":
        if len(self.phases) != len(set(self.phases)):
            raise ValueError("selection phases must be unique")
        if not self.phases:
            raise ValueError("at least one non-final selection phase is required")
        return self


class ExperimentConfig(BaseModel):
    """All scientifically relevant choices used to derive a run identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    experiment_id: str = Field(min_length=1)
    method_revision: str = Field(min_length=1)
    split: SplitSpec
    conditions: tuple[ConditionSpec, ...]
    replicates: int = Field(ge=1)
    seed_policy: SeedPolicy
    device_policy: DevicePolicy
    metrics: tuple[MetricSpec, ...]
    selection: SelectionSpec
    diagnostic_fields: tuple[str, ...] = ()
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def identities_are_complete(self) -> "ExperimentConfig":
        condition_ids = [condition.condition_id for condition in self.conditions]
        if not condition_ids:
            raise ValueError("at least one condition is required")
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("condition_id values must be unique")

        metric_ids = [metric.metric_id for metric in self.metrics]
        if not metric_ids:
            raise ValueError("at least one metric is required")
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("metric_id values must be unique")
        if self.selection.primary_metric not in set(metric_ids):
            raise ValueError("selection primary_metric must be a declared metric")
        primary = next(
            metric for metric in self.metrics if metric.metric_id == self.selection.primary_metric
        )
        if primary.direction == "none":
            raise ValueError("selection primary_metric must have a minimize/maximize direction")
        if "validation" in self.selection.phases and not self.split.validation_tasks:
            raise ValueError("validation selection requires validation tasks")
        if len(self.diagnostic_fields) != len(set(self.diagnostic_fields)):
            raise ValueError("diagnostic_fields must be unique")
        if any(not field for field in self.diagnostic_fields):
            raise ValueError("diagnostic_fields cannot contain empty names")

        development = {task.task_id: task for task in self.split.development_tasks}
        for condition in self.conditions:
            train_ids = set(condition.exposure.train_task_ids)
            unknown_train = train_ids - set(development)
            if unknown_train:
                raise ValueError(
                    "condition exposure may train only on development tasks: "
                    f"{sorted(unknown_train)}"
                )
            catalogs = {
                task_id: {
                    (item.stage_label, item.trajectory_id)
                    for item in development[task_id].trajectory_catalog
                }
                for task_id in train_ids
            }
            for item in condition.exposure.exposed_trajectories:
                if item.task_id not in train_ids:
                    raise ValueError("exposed trajectory task must be a declared train task")
                if (item.stage_label, item.trajectory_id) not in catalogs[item.task_id]:
                    raise ValueError(
                        "exposed trajectory must match its development task catalog"
                    )
        return self


def _normalized_json(value: Any) -> JsonValue:
    """Normalize JSON values without destroying scientifically meaningful list order."""

    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scientific config cannot contain non-finite floats")
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list | tuple):
        return [_normalized_json(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("scientific config mappings require string keys")
        return {key: _normalized_json(item) for key, item in value.items()}
    raise TypeError(f"unsupported scientific config value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return one portable canonical JSON representation."""

    normalized = _normalized_json(value)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def scientific_exposure_value(exposure: ExposureSpec) -> dict[str, JsonValue]:
    """Canonicalize unordered exposure identities for hashing and snapshots."""

    value = exposure.model_dump(mode="json")
    value["train_task_ids"] = sorted(value["train_task_ids"])
    value["exposed_trajectories"] = sorted(
        value["exposed_trajectories"],
        key=lambda item: (
            item["task_id"],
            item["stage_label"],
            item["trajectory_id"],
        ),
    )
    return value


def scientific_config_value(config: ExperimentConfig) -> dict[str, JsonValue]:
    """Return a canonicalizable value with unordered identities sorted."""

    value = config.model_dump(mode="json")
    split = value["split"]
    if not isinstance(split, dict):
        raise TypeError("unexpected split serialization")
    for name in ("development_tasks", "validation_tasks", "final_tasks"):
        tasks = split[name]
        if not isinstance(tasks, list):
            raise TypeError("unexpected task serialization")
        for task in tasks:
            catalog = task["trajectory_catalog"]
            task["trajectory_catalog"] = sorted(
                catalog,
                key=lambda item: (item["stage_label"], item["trajectory_id"]),
            )
        split[name] = sorted(tasks, key=lambda task: task["task_id"])
    conditions = value["conditions"]
    metrics = value["metrics"]
    selection = value["selection"]
    if not isinstance(conditions, list) or not isinstance(metrics, list):
        raise TypeError("unexpected config serialization")
    if not isinstance(selection, dict) or not isinstance(selection["phases"], list):
        raise TypeError("unexpected selection serialization")
    exposures = {
        condition.condition_id: scientific_exposure_value(condition.exposure)
        for condition in config.conditions
    }
    for condition in conditions:
        condition["exposure"] = exposures[condition["condition_id"]]
    value["conditions"] = sorted(conditions, key=lambda row: row["condition_id"])
    value["metrics"] = sorted(metrics, key=lambda row: row["metric_id"])
    selection["phases"] = sorted(selection["phases"])
    return value


def scientific_config_sha256(config: ExperimentConfig) -> str:
    return hashlib.sha256(canonical_json_bytes(scientific_config_value(config))).hexdigest()


def run_id_for(config: ExperimentConfig) -> str:
    prefix = _SAFE_PREFIX.sub("-", config.experiment_id.lower()).strip("-")
    if not prefix:
        prefix = "experiment"
    return f"{prefix[:40]}-{scientific_config_sha256(config)[:12]}"


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load strict JSON or TOML without adding a third-party parser dependency."""

    config_path = Path(path)
    if config_path.suffix == ".json":
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    elif config_path.suffix == ".toml":
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    else:
        raise ValueError("experiment config must use .json or .toml")
    return ExperimentConfig.model_validate(raw)
