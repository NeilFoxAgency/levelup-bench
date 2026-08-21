"""Protocol-derived Phase 2 development configs and baseline execution."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from levelup.core.trajectory import ActionRecord, Trajectory, TrajectoryStep
from levelup.envs.adaptive_track import adaptive_track_bundle, make_adaptive_track
from levelup.envs.adaptive_track import optimal_path as adaptive_optimal_path
from levelup.envs.challenge_track import frontier_path as combo_frontier_path
from levelup.envs.challenge_track import make_combo_track
from levelup.envs.challenge_track import optimal_path as combo_optimal_path
from levelup.experiments.milestone6_baselines import (
    IndependentCandidateEvaluator,
    build_clean_optimum_training_sample,
    classify_exact_optimum,
    discover_affordances,
    evaluate_generated_search,
    generate_candidates_with_observable_policy,
    optimum_only_training_samples,
    trajectory_content_sha256,
)
from levelup.experiments.runner import (
    ExperimentRunner,
    PhaseAccounting,
    ResourceAccounting,
    UnitOutcome,
    UnitPayload,
    aggregate_run,
)
from levelup.experiments.runner.config import (
    ConditionSpec,
    DevicePolicy,
    ExperimentConfig,
    ExposedTrajectory,
    ExposureSpec,
    MetricSpec,
    SeedPolicy,
    SelectionSpec,
    SplitSpec,
    TaskIdentity,
    TrajectoryIdentity,
)
from levelup.experiments.runner.records import PlannedUnit
from levelup.experiments.runner.storage import RunStore
from levelup.learning.state_conditioned import (
    AffordanceTable,
    TrainingReport,
    TrainingSpec,
    global_frequency_optimum_examples,
    global_listwise_optimum_examples,
    optimum_imitation_examples,
    train_global_frequency_optimum_model,
    train_global_listwise_optimum_model,
    train_state_conditioned_optimum_model,
)

ROOT = Path(__file__).resolve().parents[3]
DEVELOPMENT_TASKS_PATH = ROOT / "configs" / "milestone6" / "development_tasks.json"
DEVELOPMENT_PROTOCOL_PATH = ROOT / "configs" / "milestone6" / "development_protocol.json"
PHASE2_SMOKE_CONDITION_IDS = (
    "A0-no-probe-uniform",
    "A1-paid-probe-uniform",
    "B1-clean-global-optimum-frequency",
    "B2-global-listwise-optimum",
    "C-state-conditioned-listwise-optimum",
)


def load_development_protocol() -> dict[str, Any]:
    return json.loads(DEVELOPMENT_PROTOCOL_PATH.read_text(encoding="utf-8"))


def load_development_task_manifest() -> dict[str, Any]:
    return json.loads(DEVELOPMENT_TASKS_PATH.read_text(encoding="utf-8"))


def _manifest_tasks() -> tuple[dict[str, Any], ...]:
    payload = load_development_task_manifest()
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not all(isinstance(task, dict) for task in tasks):
        raise RuntimeError("invalid Milestone 6 development task manifest")
    return tuple(tasks)


def _training_identity(entry: dict[str, Any]) -> TaskIdentity:
    family = str(entry["family"])
    if family == "combo":
        environment = make_combo_track(
            int(entry["task_index"]),
            int(entry["generator_seed"]),
        )
        frontier_cost, frontier_actions = combo_frontier_path(environment)
        optimum_cost, optimum_actions = combo_optimal_path(environment)
        if frontier_cost <= optimum_cost or frontier_actions == optimum_actions:
            raise ValueError("generated Combo task has no strict frontier-to-optimum gap")
        if environment.task_spec.task_id != entry["task_id"]:
            raise RuntimeError("development manifest task reconstruction drift")
        catalog_items: list[TrajectoryIdentity] = []
        for label, actions in (
            ("frontier", frontier_actions),
            ("optimum", optimum_actions),
        ):
            trajectory_id = f"{environment.task_spec.task_id}.{label}"
            trajectory = Trajectory(
                trajectory_id=trajectory_id,
                task_id=environment.task_spec.task_id,
                source="reference",
                steps=tuple(
                    TrajectoryStep(index=index, action=ActionRecord(name=action))
                    for index, action in enumerate(actions)
                ),
            )
            catalog_items.append(
                TrajectoryIdentity(
                    stage_label=label,
                    trajectory_id=trajectory_id,
                    source="synthetic-reference",
                    provenance={
                        "content_sha256": trajectory_content_sha256(trajectory),
                        "kind": "synthetic_policy",
                        "generated_from_hidden_oracle": label == "optimum",
                    },
                )
            )
        catalog = tuple(catalog_items)
    else:
        bundle = adaptive_track_bundle(
            family,
            int(entry["task_index"]),
            int(entry["generator_seed"]),
        )
        if bundle.environment.task_spec.task_id != entry["task_id"]:
            raise RuntimeError("development manifest task reconstruction drift")
        catalog = tuple(
            TrajectoryIdentity(
                stage_label=stage.label,
                trajectory_id=stage.trajectory_id,
                source="synthetic-reference",
                provenance={
                    "content_sha256": trajectory_content_sha256(
                        bundle.trajectories[stage.trajectory_id]
                    ),
                    "kind": "synthetic_policy",
                    "generated_from_hidden_oracle": stage.label == "optimum",
                },
            )
            for stage in bundle.ladder.stages
        )
    return TaskIdentity(
        family_id=family,
        task_id=str(entry["task_id"]),
        task_index=int(entry["task_index"]),
        generator_seed=int(entry["generator_seed"]),
        environment_reset_seed=int(entry["environment_reset_seed"]),
        trajectory_catalog=catalog,
    )


def _heldout_identity(entry: dict[str, Any]) -> TaskIdentity:
    family = str(entry["family"])
    if family == "combo":
        environment = make_combo_track(
            int(entry["task_index"]),
            int(entry["generator_seed"]),
        )
    else:
        environment = make_adaptive_track(
            family,
            int(entry["task_index"]),
            int(entry["generator_seed"]),
        )
    if environment.task_spec.task_id != entry["task_id"]:
        raise RuntimeError("held-out development task reconstruction drift")
    return TaskIdentity(
        family_id=family,
        task_id=str(entry["task_id"]),
        task_index=int(entry["task_index"]),
        generator_seed=int(entry["generator_seed"]),
        environment_reset_seed=int(entry["environment_reset_seed"]),
        trajectory_catalog=(),
    )


def _optimum_exposure(tasks: tuple[TaskIdentity, ...]) -> tuple[ExposedTrajectory, ...]:
    exposed: list[ExposedTrajectory] = []
    for task in tasks:
        optimum = next(item for item in task.trajectory_catalog if item.stage_label == "optimum")
        exposed.append(
            ExposedTrajectory(
                task_id=task.task_id,
                stage_label="optimum",
                trajectory_id=optimum.trajectory_id,
            )
        )
    return tuple(exposed)


def _exposure(
    *,
    training_tasks: tuple[TaskIdentity, ...],
    exposed: tuple[ExposedTrajectory, ...],
    probe_access: bool,
) -> ExposureSpec:
    return ExposureSpec(
        train_task_ids=tuple(task.task_id for task in training_tasks),
        exposed_trajectories=exposed,
        observable_state_access="current",
        action_history_access=False,
        action_descriptors_access=False,
        probe_interaction_access=probe_access,
        search_feedback_access=False,
        evaluator_output_access=False,
        optimum_threshold_access=False,
        privileged_state_access=False,
        structured_constraint_access=True,
        metadata={
            "development_only": True,
            "exact_optimum_reporting_only": True,
        },
    )


def build_phase2_baseline_smoke_config() -> ExperimentConfig:
    """Build the frozen one-fold, one-task, one-replicate implementation smoke."""

    protocol = load_development_protocol()
    if protocol["status"] != "frozen-before-comparative-development-results":
        raise RuntimeError("Milestone 6 development protocol is not frozen")
    entries = _manifest_tasks()
    heldout_family = "combo"
    training_entries = tuple(
        entry
        for entry in entries
        if entry["family"] != heldout_family and "training_core" in entry["roles"]
    )
    if len(training_entries) != 40:
        raise RuntimeError("Phase 2 smoke requires eight core tasks from five families")
    heldout_entry = next(entry for entry in entries if entry["family"] == heldout_family)
    training_tasks = tuple(_training_identity(entry) for entry in training_entries)
    heldout_task = _heldout_identity(heldout_entry)
    optimum_exposure = _optimum_exposure(training_tasks)

    no_reference = _exposure(
        training_tasks=(),
        exposed=(),
        probe_access=False,
    )
    paid_probe = _exposure(
        training_tasks=(),
        exposed=(),
        probe_access=True,
    )
    optimum_only = _exposure(
        training_tasks=training_tasks,
        exposed=optimum_exposure,
        probe_access=True,
    )
    conditions = (
        ConditionSpec(
            condition_id="A0-no-probe-uniform",
            learner_id="uniform-visible-actions-v1",
            execution_phases=("validation",),
            exposure=no_reference,
            parameters={"probe_action_cap": 0},
        ),
        ConditionSpec(
            condition_id="A1-paid-probe-uniform",
            learner_id="uniform-visible-actions-v1",
            execution_phases=("validation",),
            exposure=paid_probe,
            parameters={"probe_action_cap": 16},
        ),
        ConditionSpec(
            condition_id="B1-clean-global-optimum-frequency",
            learner_id="global-affordance-mlp-frequency-v1",
            execution_phases=("validation",),
            exposure=optimum_only,
            parameters={"objective": "optimum_frequency"},
        ),
        ConditionSpec(
            condition_id="B2-global-listwise-optimum",
            learner_id="global-affordance-mlp-listwise-v1",
            execution_phases=("validation",),
            exposure=optimum_only,
            parameters={"objective": "listwise_optimum"},
        ),
        ConditionSpec(
            condition_id="C-state-conditioned-listwise-optimum",
            learner_id="state-affordance-mlp-listwise-v1",
            execution_phases=("validation",),
            exposure=optimum_only,
            parameters={"objective": "listwise_optimum"},
        ),
    )
    smoke = protocol["budgets"]["implementation_smoke"]
    combo_fold_offset = (
        protocol["family_order"].index("combo") * protocol["seed_policy"]["family_offset_stride"]
    )
    bases = protocol["seed_policy"]["bases"]
    return ExperimentConfig(
        experiment_id="milestone6-phase2-baseline-implementation-smoke",
        method_revision="development-baseline-boundary-v1",
        split=SplitSpec(
            development_tasks=training_tasks,
            validation_tasks=(heldout_task,),
            final_tasks=(),
        ),
        conditions=conditions,
        replicates=int(smoke["replicates"]),
        seed_policy=SeedPolicy(
            derivation_version="phase2.v1",
            model_seed_base=int(bases["model"]) + combo_fold_offset,
            probe_seed_base=int(bases["probe"]) + combo_fold_offset,
            search_seed_base=int(bases["search"]) + combo_fold_offset,
            data_order_seed_base=int(bases["data_order"]) + combo_fold_offset,
            replicate_stride=int(protocol["seed_policy"]["replicate_stride"]),
        ),
        device_policy=DevicePolicy(requested_device="cpu", torch_threads=1),
        metrics=(
            MetricSpec(
                metric_id="exact_optimum_success",
                direction="maximize",
                unit="boolean",
                description="Post-hoc exact-optimum success after fixed-budget search.",
            ),
            MetricSpec(
                metric_id="adaptation_actions_to_optimum",
                direction="minimize",
                unit="actions",
                description="Probe plus search actions to first post-hoc exact optimum.",
            ),
            MetricSpec(
                metric_id="performance_value",
                direction="minimize",
                unit="ticks",
                description="Best independently replayed valid performance.",
            ),
        ),
        selection=SelectionSpec(
            phases=("validation",),
            primary_metric="exact_optimum_success",
            rule="Implementation smoke only; no method selection or advancement decision.",
        ),
        diagnostic_fields=(
            "not_scientific_result",
            "first_optimum_adaptation_actions",
            "unknown_affordance_decisions",
            "trainable_parameters",
            "training_examples",
            "oracle_setup_calls",
        ),
        parameters={
            "not_scientific_result": True,
            "heldout_family": heldout_family,
            "fold_id": f"lofo-{heldout_family}",
            "heldout_family_id": heldout_family,
            "probe_action_cap": int(smoke["probe_actions_per_task"]),
            "probe_coverage_target_samples_per_alias": int(
                smoke["probe_coverage_target_samples_per_alias"]
            ),
            "candidate_episodes": int(smoke["candidate_episodes_per_task"]),
            "adaptation_action_cap": int(smoke["adaptation_actions_per_task"]),
            "maximum_actions_per_candidate_episode": int(
                protocol["budgets"]["maximum_actions_per_candidate_episode"]
            ),
            "learning_rate": 0.003,
            "training_epochs": 120,
            "weight_decay": float(protocol["fixed_defaults"]["weight_decay"]),
            "search_temperature": 0.9,
            "unit_local_training_repeated_and_counted": True,
            "data_order": "canonical_manifest_order_no_shuffle",
        },
    )


def _condition(config: ExperimentConfig, condition_id: str) -> ConditionSpec:
    return next(
        condition for condition in config.conditions if condition.condition_id == condition_id
    )


def _task(config: ExperimentConfig, task_id: str) -> TaskIdentity:
    return next(
        task
        for task in (
            *config.split.development_tasks,
            *config.split.validation_tasks,
        )
        if task.task_id == task_id
    )


def _environment(task: TaskIdentity) -> Any:
    if task.family_id == "combo":
        return make_combo_track(task.task_index, task.generator_seed)
    return make_adaptive_track(task.family_id, task.task_index, task.generator_seed)


def _forbidden_aliases(environment: Any) -> frozenset[str]:
    aliases = {
        constraint.verifier_config["action"]
        for constraint in environment.task_spec.constraints
        if constraint.verifier_id == "never_use_action"
        and isinstance(constraint.verifier_config.get("action"), str)
    }
    if len(aliases) != 1:
        raise RuntimeError("expected exactly one structured forbidden alias")
    return frozenset(aliases)


def _validate_smoke_parameters(
    config: ExperimentConfig,
    condition: ConditionSpec,
) -> TrainingSpec:
    _validate_smoke_structure(config)
    protocol = load_development_protocol()
    parameters = config.parameters
    smoke = protocol["budgets"]["implementation_smoke"]
    exact_parameters = {
        "probe_action_cap": int(smoke["probe_actions_per_task"]),
        "probe_coverage_target_samples_per_alias": int(
            smoke["probe_coverage_target_samples_per_alias"]
        ),
        "candidate_episodes": int(smoke["candidate_episodes_per_task"]),
        "adaptation_action_cap": int(smoke["adaptation_actions_per_task"]),
        "maximum_actions_per_candidate_episode": int(
            protocol["budgets"]["maximum_actions_per_candidate_episode"]
        ),
    }
    for name, expected in exact_parameters.items():
        if parameters.get(name) != expected:
            raise RuntimeError(f"{name} differs from the frozen smoke protocol")
    if config.replicates != int(smoke["replicates"]):
        raise RuntimeError("replicate count differs from the frozen smoke protocol")
    if config.split.final_tasks:
        raise RuntimeError("Phase 2 smoke cannot contain final tasks")
    if (
        config.parameters.get("fold_id") != "lofo-combo"
        or config.parameters.get("heldout_family_id") != "combo"
    ):
        raise RuntimeError("fold identity differs from the frozen smoke protocol")
    if condition.execution_phases != ("validation",):
        raise RuntimeError("Phase 2 smoke condition must be validation-only")
    if condition.condition_id == "A0-no-probe-uniform":
        if condition.exposure.probe_interaction_access:
            raise RuntimeError("A0 cannot receive probe interaction access")
        if condition.parameters.get("probe_action_cap") != 0:
            raise RuntimeError("A0 must declare a zero probe cap")
    elif not condition.exposure.probe_interaction_access:
        raise RuntimeError("A1/B1/B2/C require matched paid-probe access")
    if condition.condition_id == "A1-paid-probe-uniform" and (
        condition.parameters.get("probe_action_cap") != int(smoke["probe_actions_per_task"])
    ):
        raise RuntimeError("A1 must declare the frozen paid-probe cap")

    combo_fold_offset = protocol["family_order"].index("combo") * int(
        protocol["seed_policy"]["family_offset_stride"]
    )
    expected_bases = {
        "model_seed_base": int(protocol["seed_policy"]["bases"]["model"]) + combo_fold_offset,
        "probe_seed_base": int(protocol["seed_policy"]["bases"]["probe"]) + combo_fold_offset,
        "search_seed_base": int(protocol["seed_policy"]["bases"]["search"]) + combo_fold_offset,
        "data_order_seed_base": int(protocol["seed_policy"]["bases"]["data_order"])
        + combo_fold_offset,
        "replicate_stride": int(protocol["seed_policy"]["replicate_stride"]),
    }
    seed_values = config.seed_policy.model_dump(mode="json")
    if seed_values["derivation_version"] != "phase2.v1":
        raise RuntimeError("seed derivation differs from the frozen Phase 2 protocol")
    for name, expected in expected_bases.items():
        if seed_values[name] != expected:
            raise RuntimeError(f"{name} differs from the frozen seed protocol")
    if config.seed_policy.environment_seed_offset != 0:
        raise RuntimeError("environment seed offset differs from the frozen protocol")
    if config.device_policy != DevicePolicy(requested_device="cpu", torch_threads=1):
        raise RuntimeError("device policy differs from the frozen smoke protocol")

    learning_rate = float(parameters["learning_rate"])
    epochs = int(parameters["training_epochs"])
    temperature = float(parameters["search_temperature"])
    if learning_rate not in protocol["eligible_hyperparameters"]["learning_rate"]:
        raise RuntimeError("learning rate is outside the frozen protocol")
    if epochs not in protocol["eligible_hyperparameters"]["training_epochs"]:
        raise RuntimeError("training epochs are outside the frozen protocol")
    if temperature not in protocol["eligible_hyperparameters"]["search_temperature"]:
        raise RuntimeError("search temperature is outside the frozen protocol")
    weight_decay = float(parameters["weight_decay"])
    if weight_decay != float(protocol["fixed_defaults"]["weight_decay"]):
        raise RuntimeError("weight decay differs from the frozen protocol")
    return TrainingSpec(
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


def _task_manifest_identity(task: TaskIdentity) -> tuple[str, str, int, int, int]:
    return (
        task.family_id,
        task.task_id,
        task.task_index,
        task.generator_seed,
        task.environment_reset_seed,
    )


def _entry_manifest_identity(entry: dict[str, Any]) -> tuple[str, str, int, int, int]:
    return (
        str(entry["family"]),
        str(entry["task_id"]),
        int(entry["task_index"]),
        int(entry["generator_seed"]),
        int(entry["environment_reset_seed"]),
    )


def _validate_smoke_structure(config: ExperimentConfig) -> None:
    entries = _manifest_tasks()
    expected_training = tuple(
        _entry_manifest_identity(entry)
        for entry in entries
        if entry["family"] != "combo" and "training_core" in entry["roles"]
    )
    actual_training = tuple(
        _task_manifest_identity(task) for task in config.split.development_tasks
    )
    if actual_training != expected_training:
        raise RuntimeError("development split differs from the frozen smoke protocol")

    combo_entry = next(entry for entry in entries if entry["family"] == "combo")
    expected_validation = (_entry_manifest_identity(combo_entry),)
    actual_validation = tuple(
        _task_manifest_identity(task) for task in config.split.validation_tasks
    )
    if actual_validation != expected_validation:
        raise RuntimeError("validation split differs from the frozen smoke protocol")
    if config.split.validation_tasks[0].trajectory_catalog:
        raise RuntimeError("held-out task cannot expose a reference trajectory")

    actual_condition_ids = tuple(condition.condition_id for condition in config.conditions)
    if actual_condition_ids != PHASE2_SMOKE_CONDITION_IDS:
        raise RuntimeError("condition set differs from the frozen smoke protocol")

    expected_learners = {
        "A0-no-probe-uniform": "uniform-visible-actions-v1",
        "A1-paid-probe-uniform": "uniform-visible-actions-v1",
        "B1-clean-global-optimum-frequency": "global-affordance-mlp-frequency-v1",
        "B2-global-listwise-optimum": "global-affordance-mlp-listwise-v1",
        "C-state-conditioned-listwise-optimum": "state-affordance-mlp-listwise-v1",
    }
    expected_parameters = {
        "A0-no-probe-uniform": {"probe_action_cap": 0},
        "A1-paid-probe-uniform": {"probe_action_cap": 16},
        "B1-clean-global-optimum-frequency": {"objective": "optimum_frequency"},
        "B2-global-listwise-optimum": {"objective": "listwise_optimum"},
        "C-state-conditioned-listwise-optimum": {"objective": "listwise_optimum"},
    }
    training_task_ids = tuple(task.task_id for task in config.split.development_tasks)
    expected_optimum_exposure = _optimum_exposure(config.split.development_tasks)
    expected_exposures = {
        "A0-no-probe-uniform": _exposure(
            training_tasks=(),
            exposed=(),
            probe_access=False,
        ),
        "A1-paid-probe-uniform": _exposure(
            training_tasks=(),
            exposed=(),
            probe_access=True,
        ),
    }
    optimum_only = _exposure(
        training_tasks=config.split.development_tasks,
        exposed=expected_optimum_exposure,
        probe_access=True,
    )
    expected_exposures.update(
        {
            condition_id: optimum_only
            for condition_id in PHASE2_SMOKE_CONDITION_IDS
            if condition_id.startswith(("B", "C"))
        }
    )
    for candidate in config.conditions:
        if candidate.learner_id != expected_learners[candidate.condition_id]:
            raise RuntimeError("learner set differs from the frozen smoke protocol")
        if candidate.parameters != expected_parameters[candidate.condition_id]:
            raise RuntimeError("condition parameters differ from the frozen smoke protocol")
        if candidate.execution_phases != ("validation",):
            raise RuntimeError("Phase 2 smoke condition must be validation-only")
        exposure = candidate.exposure
        if exposure != expected_exposures[candidate.condition_id]:
            raise RuntimeError("condition exposure differs from the frozen smoke protocol")
        if candidate.condition_id.startswith("A"):
            continue
        if exposure.train_task_ids != training_task_ids:
            raise RuntimeError("training exposure differs from the frozen smoke protocol")
        if exposure.exposed_trajectories != expected_optimum_exposure:
            raise RuntimeError("optimum exposure differs from the frozen smoke protocol")


def _training_probe_seed(
    task: TaskIdentity,
    *,
    replicate: int,
    protocol: dict[str, Any],
) -> int:
    family_offset = protocol["family_order"].index(task.family_id) * int(
        protocol["seed_policy"]["family_offset_stride"]
    )
    return (
        int(protocol["seed_policy"]["bases"]["probe"])
        + family_offset
        + replicate * int(protocol["seed_policy"]["replicate_stride"])
        + task.task_index
    )


def _train_model(
    config: ExperimentConfig,
    planned: PlannedUnit,
    *,
    condition: ConditionSpec,
    training: TrainingSpec,
) -> tuple[Any | None, TrainingReport | None, dict[str, int | float], float, float]:
    if condition.condition_id.startswith("A"):
        return (
            None,
            None,
            {
                "probe_attempts": 0,
                "probe_resets": 0,
                "probe_actions": 0,
                "reference_evaluator_calls": 0,
                "reference_replay_actions": 0,
                "reference_observable_actions": 0,
                "reference_resets": 0,
                "probe_wall_seconds": 0.0,
                "reference_evaluator_wall_seconds": 0.0,
                "reference_observable_wall_seconds": 0.0,
            },
            0.0,
            0.0,
        )

    protocol = load_development_protocol()
    exposure_by_task = {item.task_id: item for item in condition.exposure.exposed_trajectories}
    samples = []
    accounting = {
        "probe_attempts": 0,
        "probe_resets": 0,
        "probe_actions": 0,
        "reference_evaluator_calls": 0,
        "reference_replay_actions": 0,
        "reference_observable_actions": 0,
        "reference_resets": 0,
        "probe_wall_seconds": 0.0,
        "reference_evaluator_wall_seconds": 0.0,
        "reference_observable_wall_seconds": 0.0,
    }
    setup_wall = 0.0
    for task_id in condition.exposure.train_task_ids:
        task = _task(config, task_id)
        setup_started = time.perf_counter()
        bundle = adaptive_track_bundle(
            task.family_id,
            task.task_index,
            task.generator_seed,
        )
        setup_wall += time.perf_counter() - setup_started
        exposure = exposure_by_task[task.task_id]
        trajectory = bundle.trajectories[exposure.trajectory_id]
        sample = build_clean_optimum_training_sample(
            bundle.environment,
            trajectory,
            task_identity=task,
            exposure=exposure,
            forbidden_aliases=_forbidden_aliases(bundle.environment),
            probe_seed=_training_probe_seed(
                task,
                replicate=planned.key.replicate,
                protocol=protocol,
            ),
            probe_action_cap=int(config.parameters["probe_action_cap"]),
            target_samples_per_alias=int(
                config.parameters["probe_coverage_target_samples_per_alias"]
            ),
            probe_actions_per_attempt=int(protocol["fixed_defaults"]["probe_actions_per_attempt"]),
        )
        samples.append(sample)
        accounting["probe_attempts"] += sample.probe.accounting.attempts
        accounting["probe_resets"] += sample.probe.accounting.resets
        accounting["probe_actions"] += sample.probe.accounting.actions
        accounting["reference_evaluator_calls"] += sample.reference.evaluator_calls
        accounting["reference_replay_actions"] += sample.reference.evaluator_replay_actions
        accounting["reference_observable_actions"] += sample.reference.observable_replay_actions
        accounting["reference_resets"] += sample.reference.resets
        accounting["probe_wall_seconds"] += sample.probe.accounting.wall_seconds
        accounting["reference_evaluator_wall_seconds"] += sample.reference.evaluator_wall_seconds
        accounting["reference_observable_wall_seconds"] += (
            sample.reference.observable_replay_wall_seconds
        )

    clean = optimum_only_training_samples(tuple(samples))
    started = time.perf_counter()
    if condition.condition_id == "B1-clean-global-optimum-frequency":
        features, targets = global_frequency_optimum_examples(clean)
        model, report = train_global_frequency_optimum_model(
            features,
            targets,
            training=training,
            model_seed=planned.seeds.model_seed,
        )
    elif condition.condition_id == "B2-global-listwise-optimum":
        model, report = train_global_listwise_optimum_model(
            global_listwise_optimum_examples(clean),
            training=training,
            model_seed=planned.seeds.model_seed,
        )
    elif condition.condition_id == "C-state-conditioned-listwise-optimum":
        model, report = train_state_conditioned_optimum_model(
            optimum_imitation_examples(clean),
            training=training,
            model_seed=planned.seeds.model_seed,
        )
    else:
        raise RuntimeError("unsupported Phase 2 baseline condition")
    return model, report, accounting, time.perf_counter() - started, setup_wall


def phase2_baseline_smoke_executor(
    config: ExperimentConfig,
    planned: PlannedUnit,
) -> UnitPayload:
    """Execute one fully counted, unit-local, non-comparative Phase 2 smoke unit."""

    if planned.key.phase != "validation":
        raise RuntimeError("Phase 2 baseline smoke executes validation units only")
    if not bool(config.parameters.get("not_scientific_result")):
        raise RuntimeError("Phase 2 implementation smoke must be non-scientific")
    condition = _condition(config, planned.key.condition_id)
    training = _validate_smoke_parameters(config, condition)
    task = _task(config, planned.key.task_id)
    setup_started = time.perf_counter()
    environment = _environment(task)
    setup_wall = time.perf_counter() - setup_started
    forbidden = _forbidden_aliases(environment)

    model, training_report, training_cost, training_wall, training_setup_wall = _train_model(
        config,
        planned,
        condition=condition,
        training=training,
    )
    setup_wall += training_setup_wall

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

    generated = generate_candidates_with_observable_policy(
        environment,
        task_id=task.task_id,
        forbidden_aliases=forbidden,
        affordances=affordances,
        model=model,
        seed=planned.seeds.search_seed,
        temperature=float(config.parameters["search_temperature"]),
        max_episodes=int(config.parameters["candidate_episodes"]),
        max_actions_per_episode=int(config.parameters["maximum_actions_per_candidate_episode"]),
        total_adaptation_action_cap=int(config.parameters["adaptation_action_cap"]),
        prior_adaptation_actions=prior_actions,
        condition_id=condition.condition_id,
    )
    search = evaluate_generated_search(
        generated,
        IndependentCandidateEvaluator(environment),
    )
    oracle_started = time.perf_counter()
    if task.family_id == "combo":
        optimum_performance = float(combo_optimal_path(environment)[0])
    else:
        optimum_performance = float(adaptive_optimal_path(environment)[0])
    oracle_wall = time.perf_counter() - oracle_started
    exact = classify_exact_optimum(
        search,
        optimum_performance=optimum_performance,
    )

    heldout_probe_attempts = heldout_probe.accounting.attempts if heldout_probe is not None else 0
    heldout_probe_resets = heldout_probe.accounting.resets if heldout_probe is not None else 0
    heldout_probe_actions = heldout_probe.accounting.actions if heldout_probe is not None else 0
    heldout_probe_wall = heldout_probe.accounting.wall_seconds if heldout_probe is not None else 0.0
    valid = bool(search.evaluated_candidates)
    completed = bool(generated.candidates)
    success = exact.success
    return UnitPayload(
        outcome=UnitOutcome(
            evaluator_ran=True,
            valid=valid,
            completed=completed,
            success=success,
            performance_metric_id="performance_value",
            performance_value=search.best_performance,
            performance_direction="minimize",
            first_valid_completion_episode=search.first_valid_episode,
            first_optimum_episode=exact.first_episode,
            first_optimum_adaptation_actions=exact.first_adaptation_actions,
            censored=not success,
            censoring_budget=(None if success else int(config.parameters["adaptation_action_cap"])),
            censoring_reason=None if success else "fixed_endpoint",
        ),
        accounting=ResourceAccounting(
            setup=PhaseAccounting(
                calls=1 + len(condition.exposure.train_task_ids),
                wall_seconds=setup_wall,
            ),
            probes=PhaseAccounting(
                calls=training_cost["probe_attempts"] + heldout_probe_attempts,
                actions=training_cost["probe_actions"] + heldout_probe_actions,
                environment_steps=training_cost["probe_actions"] + heldout_probe_actions,
                resets=training_cost["probe_resets"] + heldout_probe_resets,
                wall_seconds=training_cost["probe_wall_seconds"] + heldout_probe_wall,
            ),
            training=PhaseAccounting(
                calls=1 if training_report is not None else 0,
                forward_passes=(
                    training_report.forward_passes if training_report is not None else 0
                ),
                optimizer_steps=(
                    training_report.optimizer_steps if training_report is not None else 0
                ),
                wall_seconds=training_wall,
            ),
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
                calls=training_cost["reference_evaluator_calls"]
                + search.accounting.evaluator_calls,
                actions=training_cost["reference_replay_actions"]
                + training_cost["reference_observable_actions"]
                + search.accounting.evaluator_replay_actions,
                environment_steps=training_cost["reference_replay_actions"]
                + training_cost["reference_observable_actions"]
                + search.accounting.evaluator_replay_actions,
                resets=training_cost["reference_resets"] + search.accounting.evaluator_calls,
                wall_seconds=training_cost["reference_evaluator_wall_seconds"]
                + training_cost["reference_observable_wall_seconds"]
                + search.accounting.evaluator_wall_seconds,
            ),
            evaluator=PhaseAccounting(
                calls=1,
                wall_seconds=oracle_wall,
            ),
        ),
        diagnostics={
            "not_scientific_result": True,
            "first_optimum_adaptation_actions": exact.first_adaptation_actions,
            "unknown_affordance_decisions": (search.accounting.unknown_affordance_decisions),
            "trainable_parameters": (
                training_report.trainable_parameters if training_report is not None else 0
            ),
            "training_examples": (
                training_report.training_examples if training_report is not None else 0
            ),
            "oracle_setup_calls": 1,
        },
    )


def main(argv: list[str] | None = None) -> int:
    """Run or resume the non-scientific Phase 2 implementation smoke."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "runs" / "milestone6",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)

    config = build_phase2_baseline_smoke_config()
    store = RunStore(args.output_root, config, repository=ROOT)
    store.initialize(for_execution=not args.aggregate_only)
    counts: dict[str, int] | None = None
    if not args.aggregate_only:
        counts = ExperimentRunner(store).execute(
            lambda planned: phase2_baseline_smoke_executor(config, planned),
            phases=("validation",),
        )
    aggregate = aggregate_run(store, write=True)
    print(
        json.dumps(
            {
                "run_id": store.run_id,
                "run_directory": str(store.run_dir),
                "execution": counts,
                "complete": aggregate.complete,
                "inventory": aggregate.inventory.model_dump(mode="json"),
                "not_scientific_result": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
