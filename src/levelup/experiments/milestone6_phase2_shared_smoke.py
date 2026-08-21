"""Development-only shared-artifact smoke for the frozen Phase 2 protocol.

This module validates preparation, lineage, temperature reuse, and resume mechanics. It is
explicitly not a comparative result or a method-selection surface, and it never loads final tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from levelup.envs.adaptive_track import adaptive_track_bundle
from levelup.envs.adaptive_track import optimal_path as adaptive_optimal_path
from levelup.envs.challenge_track import optimal_path as combo_optimal_path
from levelup.experiments.milestone6_baselines import (
    IndependentCandidateEvaluator,
    build_clean_optimum_training_sample,
    classify_exact_optimum,
    discover_affordances,
    evaluate_generated_search,
    generate_candidates_with_observable_policy,
    trajectory_content_sha256,
)
from levelup.experiments.milestone6_phase2 import (
    DEVELOPMENT_PROTOCOL_PATH,
    DEVELOPMENT_TASKS_PATH,
    ROOT,
    _environment,
    _forbidden_aliases,
    _task,
    _training_probe_seed,
    build_phase2_baseline_smoke_config,
    load_development_protocol,
)
from levelup.experiments.runner import (
    ExperimentRunner,
    PhaseAccounting,
    PlannedSharedArtifact,
    ResourceAccounting,
    SharedArtifactReference,
    UnitOutcome,
    UnitPayload,
    aggregate_run,
    apply_runtime_policy,
    within_parameter_tolerance,
)
from levelup.experiments.runner.config import (
    ConditionSpec,
    ExperimentConfig,
    canonical_json_bytes,
    run_id_for,
)
from levelup.experiments.runner.provenance import capture_system_provenance
from levelup.experiments.runner.records import (
    PlannedUnit,
    TrainingPreparationAccounting,
)
from levelup.experiments.runner.storage import (
    RunStore,
    expected_units_sha256,
    plan_expected_units,
    provenance_identity_sha256,
)
from levelup.experiments.runner.training_artifacts import (
    TrainingArtifactKey,
    TrainingArtifactManifest,
    TrainingReportMetadata,
    load_training_cost,
    load_training_key_index,
    load_training_model,
    write_training_artifact,
)
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataArtifactKey,
    TrainingDataArtifactManifest,
    TrainingDataEvidenceKey,
    evidence_key_for,
    learner_samples,
    load_training_data_artifact,
    load_training_data_evidence_cost,
    load_training_data_view_cost,
    sanitize_clean_optimum_samples,
    write_training_data_artifact,
)
from levelup.learning.state_conditioned import (
    AffordanceTable,
    GlobalAffordanceScorer,
    StateConditionedScorer,
    TrainingReport,
    TrainingSpec,
    global_frequency_optimum_examples,
    global_listwise_optimum_examples,
    optimum_imitation_examples,
    train_global_frequency_optimum_model,
    train_global_listwise_optimum_model,
    train_state_conditioned_optimum_model,
)

SCREENING_CANDIDATES_PATH = ROOT / "configs" / "milestone6" / "phase2_screening_candidates.json"
B1 = "B1-clean-global-optimum-frequency"
B2 = "B2-global-listwise-optimum"
C = "C-state-conditioned-listwise-optimum"
B2_VARIANTS = {
    "B2-global-listwise-optimum--t0p6": 0.6,
    "B2-global-listwise-optimum--t0p9": 0.9,
    "B2-global-listwise-optimum--t1p2": 1.2,
}
LEARNED_BASES = (B1, B2, C)
TRAINING_TUPLE_ID = "lr0p003-e120"
PreparationEvent = Callable[[str], None]
OracleProvider = Callable[[Any, str], float]


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _base_condition_id(condition_id: str) -> str:
    return B2 if condition_id in B2_VARIANTS else condition_id


def build_phase2_shared_smoke_config() -> ExperimentConfig:
    """Freeze the non-comparative one-fold shared-artifact smoke matrix."""

    base = build_phase2_baseline_smoke_config()
    conditions = {condition.condition_id: condition for condition in base.conditions}
    learned: list[ConditionSpec] = []
    for condition_id in (B1, C):
        source = conditions[condition_id]
        learned.append(
            source.model_copy(
                update={
                    "parameters": {
                        **source.parameters,
                        "base_condition_id": condition_id,
                        "training_tuple_id": TRAINING_TUPLE_ID,
                        "search_temperature": 0.9,
                    }
                }
            )
        )
    b2_source = conditions[B2]
    for variant_id, temperature in B2_VARIANTS.items():
        learned.append(
            b2_source.model_copy(
                update={
                    "condition_id": variant_id,
                    "parameters": {
                        **b2_source.parameters,
                        "base_condition_id": B2,
                        "training_tuple_id": TRAINING_TUPLE_ID,
                        "search_temperature": temperature,
                    },
                }
            )
        )
    parameters = {
        **base.parameters,
        "shared_artifact_training": True,
        "unit_local_training_repeated_and_counted": False,
        "development_protocol_sha256": _sha256_bytes(DEVELOPMENT_PROTOCOL_PATH),
        "screening_candidates_sha256": _sha256_bytes(SCREENING_CANDIDATES_PATH),
        "development_task_manifest_sha256": _sha256_bytes(DEVELOPMENT_TASKS_PATH),
        "selection_metric_id": "total_adaptation_actions_to_first_exact_optimum",
        "selection_metric_schema_version": "restricted-interactions.v1",
        "selection_metric_action_formula": "accounting.probes.actions + accounting.search.actions",
        "selection_metric_oracle_policy": "fixed_batch_then_independent_replay_then_reporting_only_oracle",
        "selection_metric_phase": "validation",
        "selection_metric_failure_sentinel": int(base.parameters["adaptation_action_cap"])
        + 1,
    }
    parameters.pop("search_temperature", None)
    return base.model_copy(
        update={
            "experiment_id": "milestone6-phase2-shared-artifact-smoke",
            "method_revision": "development-shared-artifact-boundary-v2",
            "conditions": (
                conditions["A0-no-probe-uniform"],
                conditions["A1-paid-probe-uniform"],
                *learned,
            ),
            "diagnostic_fields": (
                *base.diagnostic_fields,
                "shared_training_artifact",
                "search_temperature",
            ),
            "parameters": parameters,
        }
    )


def validate_phase2_shared_smoke_config(config: ExperimentConfig) -> None:
    protocol = load_development_protocol()
    canonical = build_phase2_baseline_smoke_config()
    screening = json.loads(SCREENING_CANDIDATES_PATH.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "milestone6.development_protocol.v2":
        raise RuntimeError("development protocol schema drifted")
    if screening.get("schema_version") != "milestone6.phase2_screening_candidates.v2":
        raise RuntimeError("screening candidate schema drifted")
    if screening.get("status") != "frozen-before-screening-results":
        raise RuntimeError("screening candidate protocol is not frozen")
    if screening.get("scope") != "known-development-only":
        raise RuntimeError("shared smoke is not development-only")
    if screening.get("final_family_access") is not False:
        raise RuntimeError("final-family access must remain locked")
    if screening["parent_protocol"]["sha256"] != _sha256_bytes(DEVELOPMENT_PROTOCOL_PATH):
        raise RuntimeError("screening parent protocol hash drifted")
    if screening["task_manifest"]["sha256"] != _sha256_bytes(DEVELOPMENT_TASKS_PATH):
        raise RuntimeError("screening task manifest hash drifted")
    if config.split.final_tasks:
        raise RuntimeError("shared smoke cannot contain final tasks")
    if (
        config.split.development_tasks != canonical.split.development_tasks
        or config.split.validation_tasks != canonical.split.validation_tasks
    ):
        raise RuntimeError("shared smoke development split drifted")
    if config.seed_policy != canonical.seed_policy:
        raise RuntimeError("shared smoke seed policy drifted")
    if config.device_policy != canonical.device_policy:
        raise RuntimeError("shared smoke device policy drifted")
    if config.parameters.get("fold_id") != "lofo-combo":
        raise RuntimeError("shared smoke fold drifted")
    if config.parameters.get("heldout_family_id") != "combo":
        raise RuntimeError("shared smoke held-out family drifted")
    if config.parameters.get("not_scientific_result") is not True:
        raise RuntimeError("shared smoke must be marked non-scientific")
    if config.parameters.get("shared_artifact_training") is not True:
        raise RuntimeError("shared-artifact marker is missing")
    if config.parameters.get("unit_local_training_repeated_and_counted") is not False:
        raise RuntimeError("unit-local training must be disabled")
    if config.selection != canonical.selection:
        raise RuntimeError("shared smoke selection declaration drifted")
    if config.method_revision != "development-shared-artifact-boundary-v2":
        raise RuntimeError("shared smoke method revision drifted")
    if config.parameters.get("development_protocol_sha256") != _sha256_bytes(
        DEVELOPMENT_PROTOCOL_PATH
    ):
        raise RuntimeError("shared smoke protocol identity drifted")
    if config.parameters.get("screening_candidates_sha256") != _sha256_bytes(
        SCREENING_CANDIDATES_PATH
    ):
        raise RuntimeError("shared smoke screening identity drifted")
    if config.parameters.get("development_task_manifest_sha256") != _sha256_bytes(
        DEVELOPMENT_TASKS_PATH
    ):
        raise RuntimeError("shared smoke task-manifest identity drifted")
    if (
        config.parameters.get("selection_metric_id")
        != "total_adaptation_actions_to_first_exact_optimum"
        or config.parameters.get("selection_metric_schema_version")
        != "restricted-interactions.v1"
        or config.parameters.get("selection_metric_action_formula")
        != "accounting.probes.actions + accounting.search.actions"
        or config.parameters.get("selection_metric_oracle_policy")
        != "fixed_batch_then_independent_replay_then_reporting_only_oracle"
        or config.parameters.get("selection_metric_phase") != "validation"
        or config.parameters.get("selection_metric_failure_sentinel")
        != int(config.parameters["adaptation_action_cap"]) + 1
    ):
        raise RuntimeError("shared smoke typed selection-metric identity drifted")
    metric = screening.get("screening_advancement_rule", {})
    if (
        metric.get("restricted_interactions_metric_id")
        != "total_adaptation_actions_to_first_exact_optimum"
        or metric.get("executed_action_formula")
        != "accounting.probes.actions + accounting.search.actions"
        or metric.get("failure_censoring_value")
        != int(metric.get("endpoint_adaptation_actions", 0)) + 1
        or "first_optimum_adaptation_actions typed field"
        not in str(metric.get("exact_hit_value", ""))
    ):
        raise RuntimeError("screening restricted-interaction metric drifted")
    capacity = screening.get("capacity_matching", {})
    if (
        capacity.get("cross_representation_parameter_tolerance_fraction") != 0.1
        or capacity.get("required_reporting")
        != [
            "trainable_parameters",
            "optimizer_steps",
            "forward_passes",
            "training_wall_seconds",
        ]
        or "objective-matched optimum-imitation baseline"
        not in str(capacity.get("optimum_imitation_compute_floor", ""))
    ):
        raise RuntimeError("screening capacity-matching contract drifted")
    expected_ids = {
        "A0-no-probe-uniform",
        "A1-paid-probe-uniform",
        B1,
        C,
        *B2_VARIANTS,
    }
    if {condition.condition_id for condition in config.conditions} != expected_ids:
        raise RuntimeError("shared smoke condition set drifted")
    if config.replicates != 1:
        raise RuntimeError("shared smoke requires one replicate")
    for name in (
        "probe_action_cap",
        "probe_coverage_target_samples_per_alias",
        "candidate_episodes",
        "adaptation_action_cap",
        "maximum_actions_per_candidate_episode",
        "learning_rate",
        "training_epochs",
        "weight_decay",
        "data_order",
    ):
        if config.parameters.get(name) != canonical.parameters.get(name):
            raise RuntimeError(f"shared smoke {name} drifted")
    if any(condition.execution_phases != ("validation",) for condition in config.conditions):
        raise RuntimeError("shared smoke conditions must remain validation-only")
    expected_temperatures = set(protocol["eligible_hyperparameters"]["search_temperature"])
    if set(B2_VARIANTS.values()) != expected_temperatures:
        raise RuntimeError("B2 temperature variants drifted")
    learned_exposures = {
        condition.exposure.model_dump_json()
        for condition in config.conditions
        if _base_condition_id(condition.condition_id) in LEARNED_BASES
    }
    if len(learned_exposures) != 1:
        raise RuntimeError("learned conditions do not have identical reference exposure")
    for condition in config.conditions:
        base_condition = _base_condition_id(condition.condition_id)
        if base_condition not in LEARNED_BASES:
            canonical_control = next(
                item for item in canonical.conditions if item.condition_id == condition.condition_id
            )
            if condition != canonical_control:
                raise RuntimeError("shared smoke fixed control drifted")
            continue
        canonical_condition = next(
            item for item in canonical.conditions if item.condition_id == base_condition
        )
        if (
            condition.learner_id != canonical_condition.learner_id
            or condition.exposure != canonical_condition.exposure
            or condition.parameters.get("objective")
            != canonical_condition.parameters.get("objective")
            or condition.parameters.get("base_condition_id") != base_condition
            or condition.parameters.get("training_tuple_id") != TRAINING_TUPLE_ID
        ):
            raise RuntimeError("shared smoke learned condition drifted")
        expected_temperature = B2_VARIANTS.get(condition.condition_id, 0.9)
        if condition.parameters.get("search_temperature") != expected_temperature:
            raise RuntimeError("shared smoke search temperature drifted")
        exposure = condition.exposure
        if (
            not exposure.probe_interaction_access
            or exposure.optimum_threshold_access
            or exposure.evaluator_output_access
            or exposure.search_feedback_access
            or exposure.privileged_state_access
        ):
            raise RuntimeError("learned exposure boundary drifted")
        if {item.stage_label for item in exposure.exposed_trajectories} != {"optimum"}:
            raise RuntimeError("shared smoke exposes a non-optimum reference")


@dataclass(frozen=True)
class SharedDataArtifacts:
    evidence_key: TrainingDataEvidenceKey
    evidence_id: str
    views: dict[str, tuple[TrainingDataArtifactKey, TrainingDataArtifactManifest]]


@dataclass(frozen=True)
class SharedModelArtifacts:
    keys: dict[str, TrainingArtifactKey]
    manifests: dict[str, TrainingArtifactManifest]


@dataclass(frozen=True)
class SharedSmokeRuntime:
    config: ExperimentConfig
    store: RunStore
    data: SharedDataArtifacts
    models: SharedModelArtifacts


def _representative_units(config: ExperimentConfig) -> dict[str, PlannedUnit]:
    expected = plan_expected_units(config)
    result: dict[str, PlannedUnit] = {}
    for unit in expected.units:
        base = _base_condition_id(unit.key.condition_id)
        if base not in LEARNED_BASES:
            continue
        if base == B2 and unit.key.condition_id != "B2-global-listwise-optimum--t0p9":
            continue
        result[base] = unit
    if set(result) != set(LEARNED_BASES):
        raise RuntimeError("shared smoke representative units are incomplete")
    return result


def build_training_data_keys(
    config: ExperimentConfig,
    *,
    provenance_sha256: str,
) -> dict[str, TrainingDataArtifactKey]:
    validate_phase2_shared_smoke_config(config)
    expected = plan_expected_units(config)
    representatives = _representative_units(config)
    protocol = load_development_protocol()
    training_tasks = config.split.development_tasks
    heldout_tasks = config.split.validation_tasks
    probe_seeds = tuple(
        _training_probe_seed(task, replicate=0, protocol=protocol) for task in training_tasks
    )
    environment_seeds = tuple(task.environment_reset_seed for task in training_tasks)
    probe_policy_sha256 = _digest(
        {
            "builder": "canonical-paid-probe-v1",
            "action_cap": config.parameters["probe_action_cap"],
            "coverage_target": config.parameters["probe_coverage_target_samples_per_alias"],
            "actions_per_attempt": protocol["fixed_defaults"]["probe_actions_per_attempt"],
        }
    )
    representations = {
        B1: "global-affordance-optimum-frequency-v1",
        B2: "global-affordance-listwise-optimum-v1",
        C: "state-conditioned-listwise-optimum-v1",
    }
    objectives = {B1: "optimum_frequency", B2: "listwise_optimum", C: "listwise_optimum"}
    keys: dict[str, TrainingDataArtifactKey] = {}
    for base in LEARNED_BASES:
        unit = representatives[base]
        keys[base] = TrainingDataArtifactKey(
            screening_candidates_sha256=_sha256_bytes(SCREENING_CANDIDATES_PATH),
            protocol_sha256=_sha256_bytes(DEVELOPMENT_PROTOCOL_PATH),
            task_manifest_sha256=_sha256_bytes(DEVELOPMENT_TASKS_PATH),
            expected_unit_plan_sha256=expected_units_sha256(expected),
            provenance_sha256=provenance_sha256,
            reference_exposure_sha256=unit.exposure_manifest_sha256,
            representation_sha256=_digest(
                {"representation_id": representations[base], "learner_visible_only": True}
            ),
            probe_policy_sha256=probe_policy_sha256,
            fold_id="lofo-combo",
            heldout_family_id="combo",
            ordered_training_task_ids=tuple(task.task_id for task in training_tasks),
            ordered_heldout_task_ids=tuple(task.task_id for task in heldout_tasks),
            condition_id=base,
            objective_id=objectives[base],
            replicate=0,
            data_order_seed=unit.seeds.data_order_seed,
            probe_seeds=probe_seeds,
            environment_seeds=environment_seeds,
        )
    if len({evidence_key_for(key).key_id for key in keys.values()}) != 1:
        raise RuntimeError("B1/B2/C training-data keys do not share evidence identity")
    return keys


def _build_canonical_data(
    config: ExperimentConfig,
) -> tuple[Any, TrainingPreparationAccounting]:
    protocol = load_development_protocol()
    representative = _representative_units(config)[B1]
    condition = next(item for item in config.conditions if item.condition_id == B1)
    exposure_by_task = {item.task_id: item for item in condition.exposure.exposed_trajectories}
    samples = []
    setup_wall = 0.0
    probe_calls = probe_actions = probe_resets = 0
    probe_wall = 0.0
    replay_calls = replay_actions = replay_resets = 0
    replay_wall = 0.0
    for task_id in condition.exposure.train_task_ids:
        task = _task(config, task_id)
        setup_started = time.perf_counter()
        bundle = adaptive_track_bundle(task.family_id, task.task_index, task.generator_seed)
        setup_wall += time.perf_counter() - setup_started
        exposure = exposure_by_task[task.task_id]
        sample = build_clean_optimum_training_sample(
            bundle.environment,
            bundle.trajectories[exposure.trajectory_id],
            task_identity=task,
            exposure=exposure,
            forbidden_aliases=_forbidden_aliases(bundle.environment),
            probe_seed=_training_probe_seed(
                task,
                replicate=representative.key.replicate,
                protocol=protocol,
            ),
            probe_action_cap=int(config.parameters["probe_action_cap"]),
            target_samples_per_alias=int(
                config.parameters["probe_coverage_target_samples_per_alias"]
            ),
            probe_actions_per_attempt=int(protocol["fixed_defaults"]["probe_actions_per_attempt"]),
        )
        samples.append(sample)
        probe_calls += sample.probe.accounting.attempts
        probe_actions += sample.probe.accounting.actions
        probe_resets += sample.probe.accounting.resets
        probe_wall += sample.probe.accounting.wall_seconds
        replay_calls += sample.reference.evaluator_calls
        replay_actions += (
            sample.reference.evaluator_replay_actions + sample.reference.observable_replay_actions
        )
        replay_resets += sample.reference.resets
        replay_wall += (
            sample.reference.evaluator_wall_seconds
            + sample.reference.observable_replay_wall_seconds
        )
    sanitized = sanitize_clean_optimum_samples(tuple(samples))
    return sanitized, TrainingPreparationAccounting(
        setup=PhaseAccounting(
            calls=len(samples),
            wall_seconds=setup_wall,
        ),
        training_probes=PhaseAccounting(
            calls=probe_calls,
            actions=probe_actions,
            environment_steps=probe_actions,
            resets=probe_resets,
            wall_seconds=probe_wall,
        ),
        reference_replay=PhaseAccounting(
            calls=replay_calls,
            actions=replay_actions,
            environment_steps=replay_actions,
            resets=replay_resets,
            wall_seconds=replay_wall,
        ),
        serialization=PhaseAccounting(calls=1),
    )


def prepare_shared_training_data(
    run_dir: Path,
    config: ExperimentConfig,
    keys: dict[str, TrainingDataArtifactKey],
    *,
    event: PreparationEvent | None = None,
) -> SharedDataArtifacts:
    """Build once or validate all three condition views without exposing outcomes."""

    loaded: dict[str, tuple[TrainingDataArtifactKey, TrainingDataArtifactManifest]] = {}
    missing: list[str] = []
    for base, key in keys.items():
        try:
            manifest, _ = load_training_data_artifact(run_dir, expected_key=key)
            load_training_data_view_cost(run_dir, key)
            loaded[base] = (key, manifest)
        except (OSError, RuntimeError, ValueError):
            missing.append(base)
    if loaded and missing:
        raise RuntimeError("partial shared training-data preparation fails closed")
    if not missing:
        evidence_ids = {manifest.evidence_id for _, manifest in loaded.values()}
        if len(evidence_ids) != 1:
            raise RuntimeError("persisted B1/B2/C views do not share one evidence artifact")
        evidence_key = evidence_key_for(keys[B1])
        load_training_data_evidence_cost(run_dir, evidence_key)
        if event is not None:
            event("training_data_loaded")
        return SharedDataArtifacts(evidence_key, evidence_ids.pop(), loaded)

    if event is not None:
        event("evidence_build")
    sanitized, evidence_accounting = _build_canonical_data(config)
    view_accounting = TrainingPreparationAccounting(serialization=PhaseAccounting(calls=1))
    for base in LEARNED_BASES:
        manifest = write_training_data_artifact(
            run_dir,
            keys[base],
            sanitized,
            evidence_accounting=evidence_accounting,
            view_accounting=view_accounting,
        )
        loaded[base] = (keys[base], manifest)
        if event is not None:
            event(f"view_materialized:{base}")
    evidence_ids = {manifest.evidence_id for _, manifest in loaded.values()}
    if (
        len(evidence_ids) != 1
        or len({manifest.artifact_id for _, manifest in loaded.values()}) != 3
    ):
        raise RuntimeError("shared training-data identity gate failed")
    evidence_key = evidence_key_for(keys[B1])
    load_training_data_evidence_cost(run_dir, evidence_key)
    return SharedDataArtifacts(evidence_key, evidence_ids.pop(), loaded)


def build_model_keys(
    config: ExperimentConfig,
    data: SharedDataArtifacts,
    *,
    provenance_sha256: str,
) -> dict[str, TrainingArtifactKey]:
    representatives = _representative_units(config)
    protocol = load_development_protocol()
    mappings = {
        B1: (
            "global-affordance-mlp-frequency-v1",
            "optimum_frequency",
            "global_affordance_mlp_frequency_v1",
        ),
        B2: (
            "global-affordance-mlp-listwise-v1",
            "listwise_optimum",
            "global_affordance_mlp_listwise_v1",
        ),
        C: (
            "state-affordance-mlp-listwise-v1",
            "listwise_optimum",
            "state_conditioned_mlp_listwise_v1",
        ),
    }
    training_config_sha256 = _digest(
        {
            "optimizer": "adam",
            "learning_rate": config.parameters["learning_rate"],
            "epochs": config.parameters["training_epochs"],
            "weight_decay": config.parameters["weight_decay"],
            "temperature_excluded": True,
        }
    )
    keys: dict[str, TrainingArtifactKey] = {}
    for base in LEARNED_BASES:
        learner_id, objective_id, backbone_id = mappings[base]
        unit = representatives[base]
        data_key, data_manifest = data.views[base]
        keys[base] = TrainingArtifactKey(
            screening_candidates_sha256=data_key.screening_candidates_sha256,
            protocol_sha256=data_key.protocol_sha256,
            task_manifest_sha256=data_key.task_manifest_sha256,
            expected_unit_plan_sha256=data_key.expected_unit_plan_sha256,
            exposure_sha256=unit.exposure_manifest_sha256,
            training_data_sha256=data_manifest.artifact_id,
            provenance_sha256=provenance_sha256,
            fold_id=data_key.fold_id,
            heldout_family_id=data_key.heldout_family_id,
            ordered_training_task_ids=data_key.ordered_training_task_ids,
            ordered_heldout_task_ids=data_key.ordered_heldout_task_ids,
            condition_id=base,
            learner_id=learner_id,
            objective_id=objective_id,
            backbone_id=backbone_id,
            training_tuple_id=TRAINING_TUPLE_ID,
            replicate=0,
            model_seed=unit.seeds.model_seed,
            data_order_seed=data_key.data_order_seed,
            probe_seeds=data_key.probe_seeds,
            environment_seeds=data_key.environment_seeds,
            probe_spec_sha256=data_key.probe_policy_sha256,
            training_config_sha256=training_config_sha256,
            capacity_spec_sha256=_digest(
                {
                    "hidden_widths": protocol["fixed_defaults"]["mlp_hidden_widths"],
                    "backbone_id": backbone_id,
                    "capacity_match_tolerance_fraction": 0.1,
                }
            ),
        )
    return keys


def _consumers(config: ExperimentConfig, base: str) -> tuple[PlannedUnit, ...]:
    return tuple(
        unit
        for unit in plan_expected_units(config).units
        if _base_condition_id(unit.key.condition_id) == base
    )


def build_shared_artifact_plan(
    config: ExperimentConfig,
    data: SharedDataArtifacts,
    model_keys: dict[str, TrainingArtifactKey],
) -> tuple[PlannedSharedArtifact, ...]:
    learned_units = tuple(
        unit
        for unit in plan_expected_units(config).units
        if _base_condition_id(unit.key.condition_id) in LEARNED_BASES
    )
    owner_by_base = {
        B1: B1,
        B2: "B2-global-listwise-optimum--t0p9",
        C: C,
    }
    common = {
        "owner_family_id": "combo",
        "owner_fold_id": "lofo-combo",
        "owner_replicate": 0,
        "consumer_phase": "validation",
    }
    plans: list[PlannedSharedArtifact] = [
        PlannedSharedArtifact(
            kind="training_data_evidence",
            key_id=data.evidence_key.key_id,
            owner_condition_id=B1,
            owner_group_id="canonical-evidence",
            consumer_condition_ids=tuple(sorted(unit.key.condition_id for unit in learned_units)),
            consumer_unit_ids=tuple(sorted(unit.unit_id for unit in learned_units)),
            **common,
        )
    ]
    for base in LEARNED_BASES:
        consumers = _consumers(config, base)
        consumer_ids = tuple(sorted(unit.key.condition_id for unit in consumers))
        unit_ids = tuple(sorted(unit.unit_id for unit in consumers))
        data_key, _ = data.views[base]
        plans.extend(
            (
                PlannedSharedArtifact(
                    kind="training_data_view",
                    key_id=data_key.key_id,
                    owner_condition_id=owner_by_base[base],
                    owner_group_id=base,
                    consumer_condition_ids=consumer_ids,
                    consumer_unit_ids=unit_ids,
                    **common,
                ),
                PlannedSharedArtifact(
                    kind="training_artifact",
                    key_id=model_keys[base].key_id,
                    owner_condition_id=owner_by_base[base],
                    owner_group_id=base,
                    consumer_condition_ids=consumer_ids,
                    consumer_unit_ids=unit_ids,
                    **common,
                ),
            )
        )
    return tuple(plans)


def _model_factory(model_id: str) -> nn.Module:
    if model_id in {
        "global_affordance_mlp_frequency_v1",
        "global_affordance_mlp_listwise_v1",
    }:
        return GlobalAffordanceScorer()
    if model_id == "state_conditioned_mlp_listwise_v1":
        return StateConditionedScorer()
    raise ValueError("unsupported shared-smoke model ID")


def _train_shared_model(
    base: str,
    payload: Any,
    key: TrainingArtifactKey,
    config: ExperimentConfig,
) -> tuple[nn.Module, TrainingReport, ResourceAccounting]:
    samples = learner_samples(payload)
    setup_started = time.perf_counter()
    training = TrainingSpec(
        epochs=int(config.parameters["training_epochs"]),
        learning_rate=float(config.parameters["learning_rate"]),
        weight_decay=float(config.parameters["weight_decay"]),
    )
    if base == B1:
        features, targets = global_frequency_optimum_examples(samples)
        example_count = int(features.shape[0])
        setup_wall = time.perf_counter() - setup_started
        train_started = time.perf_counter()
        model, report = train_global_frequency_optimum_model(
            features,
            targets,
            training=training,
            model_seed=key.model_seed,
        )
    elif base == B2:
        examples = global_listwise_optimum_examples(samples)
        example_count = len(examples)
        setup_wall = time.perf_counter() - setup_started
        train_started = time.perf_counter()
        model, report = train_global_listwise_optimum_model(
            examples,
            training=training,
            model_seed=key.model_seed,
        )
    elif base == C:
        examples = optimum_imitation_examples(samples)
        example_count = len(examples)
        setup_wall = time.perf_counter() - setup_started
        train_started = time.perf_counter()
        model, report = train_state_conditioned_optimum_model(
            examples,
            training=training,
            model_seed=key.model_seed,
        )
    else:
        raise RuntimeError("unsupported shared-smoke learner")
    training_wall = time.perf_counter() - train_started
    if report.training_examples != example_count:
        raise RuntimeError("training report example count drifted")
    return (
        model,
        report,
        ResourceAccounting(
            setup=PhaseAccounting(calls=1, wall_seconds=setup_wall),
            training=PhaseAccounting(
                calls=1,
                forward_passes=report.forward_passes,
                optimizer_steps=report.optimizer_steps,
                wall_seconds=training_wall,
            ),
            serialization=PhaseAccounting(calls=1),
        ),
    )


def prepare_shared_models(
    run_dir: Path,
    config: ExperimentConfig,
    data: SharedDataArtifacts,
    keys: dict[str, TrainingArtifactKey],
    *,
    event: PreparationEvent | None = None,
) -> SharedModelArtifacts:
    manifests: dict[str, TrainingArtifactManifest] = {}
    reports: dict[str, TrainingReportMetadata] = {}
    for base in LEARNED_BASES:
        key = keys[base]
        try:
            index = load_training_key_index(run_dir, key)
            cost = load_training_cost(run_dir, key)
            _, manifest = load_training_model(
                run_dir,
                index.artifact_id,
                expected_key=key,
                model_factory=_model_factory,
            )
            if cost.artifact_id != manifest.artifact_id:
                raise RuntimeError("shared model cost points to another artifact")
            if event is not None:
                event(f"model_loaded:{base}")
        except (OSError, RuntimeError, ValueError):
            if event is not None:
                event(f"model_train:{base}")
            _, payload = load_training_data_artifact(
                run_dir,
                expected_key=data.views[base][0],
            )
            model, report, accounting = _train_shared_model(base, payload, key, config)
            manifest = write_training_artifact(
                run_dir,
                key=key,
                model_id=key.backbone_id,
                model=model,
                accounting=accounting,
                report=TrainingReportMetadata(
                    trainable_parameters=report.trainable_parameters,
                    optimizer_steps=report.optimizer_steps,
                    forward_passes=report.forward_passes,
                    training_examples=report.training_examples,
                ),
            )
        manifests[base] = manifest
        reports[base] = manifest.report
    b2 = reports[B2]
    c_report = reports[C]
    _, b2_payload = load_training_data_artifact(run_dir, expected_key=data.views[B2][0])
    _, c_payload = load_training_data_artifact(run_dir, expected_key=data.views[C][0])
    b2_examples = global_listwise_optimum_examples(learner_samples(b2_payload))
    c_examples = optimum_imitation_examples(learner_samples(c_payload))
    b2_label_sha256 = _digest(
        [
            (example.selected_index, int(example.candidate_features.shape[0]))
            for example in b2_examples
        ]
    )
    c_label_sha256 = _digest(
        [
            (example.selected_index, int(example.candidate_features.shape[0]))
            for example in c_examples
        ]
    )
    same_affordance_rows = len(b2_examples) == len(c_examples) and all(
        b2_example.candidate_features.shape[0] == c_example.candidate_features.shape[0]
        and torch.equal(
            b2_example.candidate_features,
            c_example.candidate_features[:, 5:],
        )
        for b2_example, c_example in zip(b2_examples, c_examples)
    )
    if (
        b2.training_examples != c_report.training_examples
        or b2.optimizer_steps != c_report.optimizer_steps
        or b2.forward_passes != c_report.forward_passes
        or b2_label_sha256 != c_label_sha256
        or not same_affordance_rows
    ):
        raise RuntimeError("B2/C matched listwise training budget drifted")
    if not within_parameter_tolerance(
        b2.trainable_parameters,
        c_report.trainable_parameters,
        tolerance=0.1,
    ):
        raise RuntimeError("B2/C capacity matching tolerance failed")
    if reports[B1].optimizer_steps != b2.optimizer_steps:
        raise RuntimeError("optimum imitation received a smaller training budget")
    return SharedModelArtifacts(keys=keys, manifests=manifests)


def _shared_references(
    runtime: SharedSmokeRuntime,
    base: str,
) -> tuple[SharedArtifactReference, ...]:
    data_key, data_manifest = runtime.data.views[base]
    evidence_cost = load_training_data_evidence_cost(
        runtime.store.run_dir, runtime.data.evidence_key
    )
    view_cost = load_training_data_view_cost(runtime.store.run_dir, data_key)
    model_key = runtime.models.keys[base]
    model_cost = load_training_cost(runtime.store.run_dir, model_key)
    return (
        SharedArtifactReference(
            kind="training_data_evidence",
            key_id=runtime.data.evidence_key.key_id,
            artifact_id=runtime.data.evidence_id,
            cost_id=evidence_cost.cost_id,
        ),
        SharedArtifactReference(
            kind="training_data_view",
            key_id=data_key.key_id,
            artifact_id=data_manifest.artifact_id,
            cost_id=view_cost.cost_id,
        ),
        SharedArtifactReference(
            kind="training_artifact",
            key_id=model_key.key_id,
            artifact_id=runtime.models.manifests[base].artifact_id,
            cost_id=model_cost.cost_id,
        ),
    )


def _default_optimum_provider(environment: Any, family_id: str) -> float:
    if family_id == "combo":
        return float(combo_optimal_path(environment)[0])
    return float(adaptive_optimal_path(environment)[0])


def _generated_candidates_sha256(generated: Any) -> str:
    return _digest(
        [
            {
                "episode": item.episode,
                "adaptation_actions": item.adaptation_actions,
                "trajectory_sha256": trajectory_content_sha256(item.trajectory),
            }
            for item in generated.candidates
        ]
    )


def phase2_shared_smoke_executor(
    runtime: SharedSmokeRuntime,
    planned: PlannedUnit,
    *,
    optimum_provider: OracleProvider = _default_optimum_provider,
    event: PreparationEvent | None = None,
) -> UnitPayload:
    """Execute held-out interaction only; shared preparation is immutable and preloaded."""

    config = runtime.config
    if planned.key.phase != "validation":
        raise RuntimeError("shared smoke executes validation units only")
    condition = next(
        item for item in config.conditions if item.condition_id == planned.key.condition_id
    )
    base = _base_condition_id(condition.condition_id)
    setup_started = time.perf_counter()
    task = _task(config, planned.key.task_id)
    environment = _environment(task)
    forbidden = _forbidden_aliases(environment)
    setup_wall = time.perf_counter() - setup_started

    model: nn.Module | None = None
    report: TrainingReportMetadata | None = None
    references: tuple[SharedArtifactReference, ...] = ()
    if base in LEARNED_BASES:
        key = runtime.models.keys[base]
        model, manifest = load_training_model(
            runtime.store.run_dir,
            runtime.models.manifests[base].artifact_id,
            expected_key=key,
            model_factory=_model_factory,
        )
        report = manifest.report
        references = _shared_references(runtime, base)

    heldout_probe = None
    if condition.exposure.probe_interaction_access:
        heldout_probe = discover_affordances(
            environment,
            task_id=task.task_id,
            forbidden_aliases=forbidden,
            seed=planned.seeds.probe_seed,
            action_cap=int(config.parameters["probe_action_cap"]),
            target_samples_per_alias=int(
                config.parameters["probe_coverage_target_samples_per_alias"]
            ),
            actions_per_attempt=int(
                load_development_protocol()["fixed_defaults"]["probe_actions_per_attempt"]
            ),
        )
        affordances = heldout_probe.affordances
        prior_actions = heldout_probe.accounting.actions
    else:
        affordances = AffordanceTable(features={}, sample_counts={})
        prior_actions = 0

    temperature = float(condition.parameters.get("search_temperature", 0.9))
    generated = generate_candidates_with_observable_policy(
        environment,
        task_id=task.task_id,
        forbidden_aliases=forbidden,
        affordances=affordances,
        model=model,
        seed=planned.seeds.search_seed,
        temperature=temperature,
        max_episodes=int(config.parameters["candidate_episodes"]),
        max_actions_per_episode=int(config.parameters["maximum_actions_per_candidate_episode"]),
        total_adaptation_action_cap=int(config.parameters["adaptation_action_cap"]),
        prior_adaptation_actions=prior_actions,
        condition_id=condition.condition_id,
    )
    generated_sha256 = _generated_candidates_sha256(generated)
    if event is not None:
        event("generation_complete")
    search = evaluate_generated_search(generated, IndependentCandidateEvaluator(environment))
    if event is not None:
        event("candidate_evaluation_complete")
        event("optimum_oracle")
    oracle_started = time.perf_counter()
    optimum_performance = optimum_provider(environment, task.family_id)
    oracle_wall = time.perf_counter() - oracle_started
    exact = classify_exact_optimum(search, optimum_performance=optimum_performance)

    heldout_probe_attempts = heldout_probe.accounting.attempts if heldout_probe else 0
    heldout_probe_resets = heldout_probe.accounting.resets if heldout_probe else 0
    heldout_probe_actions = heldout_probe.accounting.actions if heldout_probe else 0
    heldout_probe_wall = heldout_probe.accounting.wall_seconds if heldout_probe else 0.0
    valid = bool(search.evaluated_candidates)
    completed = bool(generated.candidates)
    return UnitPayload(
        outcome=UnitOutcome(
            evaluator_ran=True,
            valid=valid,
            completed=completed,
            success=exact.success,
            performance_metric_id="performance_value",
            performance_value=search.best_performance,
            performance_direction="minimize",
            first_valid_completion_episode=search.first_valid_episode,
            first_optimum_episode=exact.first_episode,
            first_optimum_adaptation_actions=exact.first_adaptation_actions,
            censored=not exact.success,
            censoring_budget=(
                None if exact.success else int(config.parameters["adaptation_action_cap"])
            ),
            censoring_reason=None if exact.success else "fixed_endpoint",
        ),
        accounting=ResourceAccounting(
            setup=PhaseAccounting(calls=1, wall_seconds=setup_wall),
            probes=PhaseAccounting(
                calls=heldout_probe_attempts,
                actions=heldout_probe_actions,
                environment_steps=heldout_probe_actions,
                resets=heldout_probe_resets,
                wall_seconds=heldout_probe_wall,
            ),
            training=PhaseAccounting(),
            search=PhaseAccounting(
                calls=1,
                episodes=search.accounting.episodes,
                actions=search.accounting.actions,
                environment_steps=search.accounting.actions,
                resets=search.accounting.resets,
                forward_passes=search.accounting.forward_passes,
                wall_seconds=search.accounting.generation_wall_seconds,
            ),
            replay=PhaseAccounting(
                calls=search.accounting.evaluator_calls,
                actions=search.accounting.evaluator_replay_actions,
                environment_steps=search.accounting.evaluator_replay_actions,
                resets=search.accounting.evaluator_calls,
                wall_seconds=search.accounting.evaluator_wall_seconds,
            ),
            evaluator=PhaseAccounting(calls=1, wall_seconds=oracle_wall),
        ),
        shared_artifacts=references,
        candidate_generation_sha256=generated_sha256,
        diagnostics={
            "not_scientific_result": True,
            "first_optimum_adaptation_actions": exact.first_adaptation_actions,
            "unknown_affordance_decisions": search.accounting.unknown_affordance_decisions,
            "trainable_parameters": report.trainable_parameters if report else 0,
            "training_examples": report.training_examples if report else 0,
            "oracle_setup_calls": 1,
            "shared_training_artifact": base in LEARNED_BASES,
            "search_temperature": temperature,
        },
    )


def prepare_phase2_shared_smoke(
    output_root: Path,
    *,
    repository: Path = ROOT,
    event: PreparationEvent | None = None,
) -> SharedSmokeRuntime:
    """Open or prepare the frozen smoke without executing held-out task units."""

    config = build_phase2_shared_smoke_config()
    validate_phase2_shared_smoke_config(config)
    apply_runtime_policy(config.device_policy)
    provenance = capture_system_provenance(repository, config.device_policy)
    provenance_sha256 = provenance_identity_sha256(provenance)
    run_dir = output_root / run_id_for(config)
    data_keys = build_training_data_keys(
        config,
        provenance_sha256=provenance_sha256,
    )
    data = prepare_shared_training_data(
        run_dir,
        config,
        data_keys,
        event=event,
    )
    model_keys = build_model_keys(
        config,
        data,
        provenance_sha256=provenance_sha256,
    )
    plans = build_shared_artifact_plan(config, data, model_keys)
    store = RunStore(
        output_root,
        config,
        repository=repository,
        shared_artifacts=plans,
    )
    store.initialize()
    models = prepare_shared_models(
        store.run_dir,
        config,
        data,
        model_keys,
        event=event,
    )
    return SharedSmokeRuntime(config=config, store=store, data=data, models=models)


def run_phase2_shared_smoke(
    output_root: Path,
    *,
    repository: Path = ROOT,
    preparation_event: PreparationEvent | None = None,
) -> tuple[SharedSmokeRuntime, dict[str, int]]:
    runtime = prepare_phase2_shared_smoke(
        output_root,
        repository=repository,
        event=preparation_event,
    )
    counts = ExperimentRunner(runtime.store).execute(
        lambda planned: phase2_shared_smoke_executor(runtime, planned),
        phases=("validation",),
    )
    return runtime, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "runs" / "milestone6",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)

    runtime = prepare_phase2_shared_smoke(args.output_root)
    counts: dict[str, int] | None = None
    if not args.aggregate_only:
        counts = ExperimentRunner(runtime.store).execute(
            lambda planned: phase2_shared_smoke_executor(runtime, planned),
            phases=("validation",),
        )
    aggregate = aggregate_run(runtime.store, strict=not args.aggregate_only, write=True)
    print(
        json.dumps(
            {
                "run_id": runtime.store.run_id,
                "run_directory": str(runtime.store.run_dir),
                "execution": counts,
                "complete": aggregate.complete,
                "inventory": aggregate.inventory.model_dump(mode="json"),
                "shared_inventory": aggregate.shared_inventory.model_dump(mode="json"),
                "shared_artifacts_sha256": aggregate.shared_artifacts_sha256,
                "not_scientific_result": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
