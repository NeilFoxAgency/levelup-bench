"""Development-only smoke adapter for the Phase 1 experiment runner.

This command validates configuration, environment replay, atomic storage, resume,
and pure aggregation. Its numbers are not Milestone 6 scientific results.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from levelup.envs.adaptive_track import AdaptiveTrackBundle, collect_adaptive_bundles
from levelup.evaluation import evaluate_trajectory
from levelup.experiments.runner import (
    ExperimentRunner,
    PhaseAccounting,
    ResourceAccounting,
    RunStore,
    UnitOutcome,
    UnitPayload,
    aggregate_run,
    load_experiment_config,
)
from levelup.experiments.runner.config import ExperimentConfig
from levelup.experiments.runner.records import PlannedUnit


@lru_cache(maxsize=None)
def _bundles(family: str, generator_seed: int, count: int) -> tuple[AdaptiveTrackBundle, ...]:
    return collect_adaptive_bundles(family, count, generator_seed)


def _task_config(config: ExperimentConfig, task_id: str) -> Any:
    tasks = (
        config.split.development_tasks
        + config.split.validation_tasks
        + config.split.final_tasks
    )
    matches = [task for task in tasks if task.task_id == task_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one configured task {task_id!r}")
    return matches[0]


def _condition(config: ExperimentConfig, condition_id: str) -> Any:
    matches = [
        condition for condition in config.conditions if condition.condition_id == condition_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one configured condition {condition_id!r}")
    return matches[0]


def smoke_executor(config: ExperimentConfig, planned: PlannedUnit) -> UnitPayload:
    """Replay a declared development reference without model training or final data."""

    if planned.key.phase == "final":
        raise RuntimeError("Phase 1 smoke refuses final tasks")
    task = _task_config(config, planned.key.task_id)
    condition = _condition(config, planned.key.condition_id)
    stage = condition.parameters.get("stage")
    if stage not in ("frontier", "optimum"):
        raise RuntimeError("smoke condition stage must be frontier or optimum")
    if planned.seeds.environment_seed != task.generator_seed:
        raise RuntimeError("smoke environment seed must match the configured generator seed")

    candidates = _bundles(
        task.family_id,
        task.generator_seed,
        max(task.task_index + 1, 1),
    )
    matches = [bundle for bundle in candidates if bundle.ladder.task_id == task.task_id]
    if len(matches) != 1:
        raise RuntimeError(f"could not reconstruct configured task {task.task_id!r}")
    bundle = matches[0]
    trajectory = bundle.trajectory_for(stage)
    declared_exposure = {
        (item.task_id, item.stage_label, item.trajectory_id)
        for item in condition.exposure.exposed_trajectories
    }
    expected_exposure = (task.task_id, stage, trajectory.trajectory_id)
    if expected_exposure not in declared_exposure:
        raise RuntimeError("smoke trajectory is absent from the condition exposure manifest")
    started = time.perf_counter()
    result = evaluate_trajectory(bundle.environment.fresh(), trajectory)
    replay_seconds = time.perf_counter() - started
    eligible = result.performance_eligible_for(bundle.environment.task_spec)
    return UnitPayload(
        outcome=UnitOutcome(
            evaluator_ran=True,
            valid=result.valid_for(bundle.environment.task_spec),
            completed=result.completed,
            success=eligible,
            performance_metric_id="performance_value",
            performance_value=result.performance_value,
            performance_direction="minimize",
            first_valid_completion_episode=1 if eligible else None,
            first_optimum_episode=1 if stage == "optimum" and eligible else None,
        ),
        accounting=ResourceAccounting(
            replay=PhaseAccounting(
                calls=1,
                episodes=1,
                actions=len(trajectory.steps),
                environment_steps=len(trajectory.steps),
            ),
            evaluator=PhaseAccounting(
                calls=1,
                episodes=1,
                resets=1,
                wall_seconds=replay_seconds,
            ),
        ),
        diagnostics={
            "not_scientific_result": True,
            "smoke_stage_frontier": stage == "frontier",
            "smoke_stage_optimum": stage == "optimum",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-retry-failed", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    store = RunStore(args.output, config, repository=Path(args.repository))
    store.initialize(for_execution=not args.aggregate_only)
    execution: dict[str, int] | None = None
    if not args.aggregate_only:
        execution = ExperimentRunner(store).execute(
            lambda planned: smoke_executor(config, planned),
            resume=not args.no_resume,
            retry_failed=not args.no_retry_failed,
        )
    aggregate = aggregate_run(store, strict=not args.allow_incomplete, write=True)
    print(
        json.dumps(
            {
                "execution": execution,
                "aggregate": aggregate.model_dump(mode="json"),
                "run_directory": str(store.run_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
