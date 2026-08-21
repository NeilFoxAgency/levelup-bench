"""Configuration-driven, resumable experiment infrastructure."""

from levelup.experiments.runner.aggregate import aggregate_run
from levelup.experiments.runner.config import (
    ExperimentConfig,
    load_experiment_config,
    run_id_for,
    scientific_config_sha256,
)
from levelup.experiments.runner.execution import ExperimentRunner
from levelup.experiments.runner.provenance import apply_runtime_policy
from levelup.experiments.runner.records import (
    ExpectedSharedArtifacts,
    PhaseAccounting,
    PlannedSharedArtifact,
    ResourceAccounting,
    SharedArtifactInventory,
    SharedArtifactReference,
    TrainingArtifactCostRecord,
    UnitOutcome,
    UnitPayload,
)
from levelup.experiments.runner.selection_metric import (
    ExpectedSelectionUnit,
    SelectionAuthority,
    SelectionMetricSpec,
    build_selection_metric_spec,
    load_selection_authority,
    merge_selection_metric_specs,
    restricted_interactions,
    summarize_variant,
    within_parameter_tolerance,
)
from levelup.experiments.runner.storage import RunStore

__all__ = [
    "ExperimentConfig",
    "ExperimentRunner",
    "ExpectedSharedArtifacts",
    "ExpectedSelectionUnit",
    "PhaseAccounting",
    "PlannedSharedArtifact",
    "ResourceAccounting",
    "SharedArtifactInventory",
    "SelectionMetricSpec",
    "SelectionAuthority",
    "RunStore",
    "UnitOutcome",
    "UnitPayload",
    "SharedArtifactReference",
    "TrainingArtifactCostRecord",
    "aggregate_run",
    "apply_runtime_policy",
    "build_selection_metric_spec",
    "load_experiment_config",
    "load_selection_authority",
    "merge_selection_metric_specs",
    "run_id_for",
    "scientific_config_sha256",
    "restricted_interactions",
    "summarize_variant",
    "within_parameter_tolerance",
]
