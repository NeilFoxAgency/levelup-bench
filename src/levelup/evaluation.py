"""Deterministic trajectory replay and reference validation."""

from __future__ import annotations

from levelup.core.reference import ReferenceEntry
from levelup.core.result import BenchmarkResult, EfficiencyMetrics
from levelup.core.trajectory import Trajectory
from levelup.envs.base import BenchmarkEnvironment


class ReplayError(ValueError):
    """Raised when a trajectory cannot be faithfully replayed."""


class ReferenceValidationError(ValueError):
    """Raised when a claimed reference does not match replayed benchmark truth."""


def evaluate_trajectory(
    environment: BenchmarkEnvironment,
    trajectory: Trajectory,
    *,
    run_id: str | None = None,
    efficiency: EfficiencyMetrics | None = None,
) -> BenchmarkResult:
    """Replay a trajectory and compute benchmark truth from the environment.

    Performance is recorded even for invalid completed trajectories so that
    violations remain diagnostically visible. ``BenchmarkResult`` itself gates
    whether that performance is eligible for comparison.
    """

    task = environment.task_spec
    if trajectory.task_id != task.task_id:
        raise ReplayError(
            f"trajectory task_id {trajectory.task_id!r} does not match {task.task_id!r}"
        )

    seed = trajectory.environment_seed
    if seed is None:
        seed = task.environment.seed
    initial = environment.reset(seed=seed)

    if trajectory.steps and initial.completed:
        raise ReplayError("trajectory contains actions for a task already complete at reset")

    completed = initial.completed
    for offset, step in enumerate(trajectory.steps):
        if completed:
            raise ReplayError(f"trajectory contains action after completion at step {offset}")

        try:
            outcome = environment.step(step.action)
        except (TypeError, ValueError) as exc:
            raise ReplayError(f"environment rejected action at step {offset}: {exc}") from exc
        completed = outcome.completed

        if step.state_hash is not None and step.state_hash != outcome.state_hash:
            raise ReplayError(
                f"state hash mismatch at step {offset}: "
                f"expected {step.state_hash}, observed {outcome.state_hash}"
            )

    final_state_hash = environment.state_hash()
    if trajectory.final_state_hash is not None and trajectory.final_state_hash != final_state_hash:
        raise ReplayError(
            "final state hash mismatch: "
            f"expected {trajectory.final_state_hash}, observed {final_state_hash}"
        )

    outcomes = tuple(
        environment.verify_constraint(constraint) for constraint in task.constraints
    )

    if efficiency is None:
        efficiency = EfficiencyMetrics(
            environment_steps=len(trajectory.steps),
            agent_actions=len(trajectory.steps),
            wall_time_seconds=0.0,
        )

    return BenchmarkResult(
        run_id=run_id or f"replay:{trajectory.trajectory_id}",
        task_id=task.task_id,
        trajectory_id=trajectory.trajectory_id,
        completed=completed,
        constraint_outcomes=outcomes,
        performance_value=environment.objective_value() if completed else None,
        efficiency=efficiency,
        final_state_hash=final_state_hash,
    )


def validate_reference(
    environment: BenchmarkEnvironment,
    reference: ReferenceEntry,
    trajectory: Trajectory,
) -> BenchmarkResult:
    """Replay and validate a reference trajectory against its claimed measurement."""

    if reference.trajectory_id is None:
        raise ReferenceValidationError("reference has no trajectory_id")
    if reference.trajectory_id != trajectory.trajectory_id:
        raise ReferenceValidationError("reference trajectory_id does not match trajectory")

    result = evaluate_trajectory(environment, trajectory)
    task = environment.task_spec

    if not result.performance_eligible_for(task):
        raise ReferenceValidationError("reference trajectory is not a valid completed run")
    if result.performance_value != reference.performance_value:
        raise ReferenceValidationError(
            f"reference claims {reference.performance_value}, "
            f"but replay measured {result.performance_value}"
        )

    return result
