import pytest
from pydantic import ValidationError

from levelup.core import ActionRecord, Trajectory, TrajectoryStep


def step(index: int, action: str = "move") -> TrajectoryStep:
    return TrajectoryStep(index=index, action=ActionRecord(name=action))


def test_contiguous_trajectory_round_trips() -> None:
    trajectory = Trajectory(
        trajectory_id="run-1",
        task_id="micro.route.001",
        source="agent",
        environment_seed=7,
        steps=(step(0), step(1)),
        final_state_hash="sha256:final",
    )

    restored = Trajectory.model_validate_json(trajectory.model_dump_json())

    assert restored == trajectory


def test_noncontiguous_steps_are_rejected() -> None:
    with pytest.raises(ValidationError, match="contiguous and start at zero"):
        Trajectory(
            trajectory_id="broken",
            task_id="micro.route.001",
            source="agent",
            steps=(step(0), step(2)),
        )
