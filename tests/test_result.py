import pytest
from pydantic import ValidationError

from levelup.core import (
    BenchmarkResult,
    ConstraintOutcome,
    ConstraintSpec,
    EfficiencyMetrics,
    EnvironmentSpec,
    ObjectiveSpec,
    TaskSpec,
)


def task() -> TaskSpec:
    return TaskSpec(
        task_id="micro.route.001",
        environment=EnvironmentSpec(
            adapter="microgames",
            environment_id="route",
            version="1",
        ),
        instruction="Reach the goal without using the forbidden shortcut.",
        constraints=(
            ConstraintSpec(
                constraint_id="no_shortcut",
                description="Do not use the forbidden shortcut.",
                verifier_id="route.no_shortcut",
            ),
        ),
        objective=ObjectiveSpec(metric_id="steps", direction="minimize", unit="steps"),
    )


def efficiency() -> EfficiencyMetrics:
    return EfficiencyMetrics(
        environment_steps=100,
        agent_actions=40,
        wall_time_seconds=0.25,
    )


def test_valid_completed_run_is_performance_eligible() -> None:
    result = BenchmarkResult(
        run_id="run-valid",
        task_id="micro.route.001",
        trajectory_id="trajectory-valid",
        completed=True,
        constraint_outcomes=(ConstraintOutcome(constraint_id="no_shortcut", passed=True),),
        performance_value=10.0,
        efficiency=efficiency(),
    )

    assert result.valid_for(task()) is True
    assert result.performance_eligible_for(task()) is True


def test_faster_invalid_run_is_not_performance_eligible() -> None:
    result = BenchmarkResult(
        run_id="run-fast-invalid",
        task_id="micro.route.001",
        trajectory_id="trajectory-fast-invalid",
        completed=True,
        constraint_outcomes=(
            ConstraintOutcome(
                constraint_id="no_shortcut",
                passed=False,
                evidence="Forbidden shortcut used at step 38.",
            ),
        ),
        performance_value=5.0,
        efficiency=efficiency(),
    )

    assert result.valid_for(task()) is False
    assert result.performance_eligible_for(task()) is False
    assert result.performance_value == 5.0


def test_missing_constraint_outcome_is_invalid() -> None:
    result = BenchmarkResult(
        run_id="run-missing-check",
        task_id="micro.route.001",
        trajectory_id="trajectory-missing-check",
        completed=True,
        constraint_outcomes=(),
        performance_value=1.0,
        efficiency=efficiency(),
    )

    assert result.valid_for(task()) is False
    assert result.performance_eligible_for(task()) is False


def test_incomplete_valid_run_is_not_performance_eligible() -> None:
    result = BenchmarkResult(
        run_id="run-incomplete",
        task_id="micro.route.001",
        trajectory_id="trajectory-incomplete",
        completed=False,
        constraint_outcomes=(ConstraintOutcome(constraint_id="no_shortcut", passed=True),),
        efficiency=efficiency(),
    )

    assert result.valid_for(task()) is True
    assert result.performance_eligible_for(task()) is False


def test_duplicate_constraint_outcomes_are_rejected() -> None:
    outcome = ConstraintOutcome(constraint_id="same", passed=True)

    with pytest.raises(ValidationError, match="unique constraint_id"):
        BenchmarkResult(
            run_id="run",
            task_id="task",
            trajectory_id="trajectory",
            completed=True,
            constraint_outcomes=(outcome, outcome),
            efficiency=efficiency(),
        )
