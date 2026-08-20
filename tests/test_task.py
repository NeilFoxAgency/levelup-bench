import pytest
from pydantic import ValidationError

from levelup.core import ConstraintSpec, EnvironmentSpec, ObjectiveSpec, TaskSpec


def make_task(*constraints: ConstraintSpec) -> TaskSpec:
    return TaskSpec(
        task_id="micro.route.001",
        environment=EnvironmentSpec(
            adapter="microgames",
            environment_id="route",
            version="1",
            seed=7,
        ),
        instruction="Reach the goal while obeying every stated rule.",
        constraints=constraints,
        objective=ObjectiveSpec(metric_id="steps", direction="minimize", unit="steps"),
    )


def test_task_round_trip_is_stable() -> None:
    task = make_task(
        ConstraintSpec(
            constraint_id="no_red_tile",
            description="Do not enter a red tile.",
            verifier_id="route.no_red_tile",
        )
    )

    restored = TaskSpec.model_validate_json(task.model_dump_json())

    assert restored == task
    assert restored.schema_version == "0.1"


def test_duplicate_constraint_ids_are_rejected() -> None:
    constraint = ConstraintSpec(
        constraint_id="same",
        description="A rule.",
        verifier_id="test.rule",
    )

    with pytest.raises(ValidationError, match="constraint_id values must be unique"):
        make_task(constraint, constraint)


def test_constraint_cannot_be_soft_in_v01() -> None:
    with pytest.raises(ValidationError):
        ConstraintSpec(
            constraint_id="rule",
            description="A rule.",
            verifier_id="test.rule",
            hard=False,
        )
