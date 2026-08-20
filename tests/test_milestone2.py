from itertools import product

import pytest

from levelup.core.reference import PerformanceTier, ReferenceEntry
from levelup.core.trajectory import ActionRecord, Trajectory, TrajectoryStep
from levelup.envs.microgames import (
    DetourGrid,
    Switchboard,
    calibration_ladders,
    detour_invalid_shortcut,
    detour_oracle,
    switchboard_invalid_shortcut,
    switchboard_oracle,
)
from levelup.evaluation import (
    ReferenceValidationError,
    ReplayError,
    evaluate_trajectory,
    validate_reference,
)


def make_trajectory(task_id: str, trajectory_id: str, actions: tuple[str, ...]) -> Trajectory:
    return Trajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        source="agent",
        steps=tuple(
            TrajectoryStep(index=i, action=ActionRecord(name=action))
            for i, action in enumerate(actions)
        ),
    )


def test_detour_oracle_is_valid_and_slower_shortcut_is_not_rewarded() -> None:
    task = DetourGrid().task_spec
    oracle = evaluate_trajectory(DetourGrid(), detour_oracle())
    shortcut = evaluate_trajectory(DetourGrid(), detour_invalid_shortcut())

    assert oracle.completed is True
    assert oracle.performance_value == 6.0
    assert oracle.performance_eligible_for(task) is True
    assert shortcut.completed is True
    assert shortcut.performance_value == 4.0
    assert shortcut.performance_value < oracle.performance_value
    assert shortcut.valid_for(task) is False
    assert shortcut.performance_eligible_for(task) is False
    assert shortcut.constraint_outcomes[0].constraint_id == "avoid_forbidden_tile"
    assert shortcut.constraint_outcomes[0].passed is False


def test_switchboard_oracle_is_valid_and_one_press_shortcut_is_invalid() -> None:
    task = Switchboard().task_spec
    oracle = evaluate_trajectory(Switchboard(), switchboard_oracle())
    shortcut = evaluate_trajectory(Switchboard(), switchboard_invalid_shortcut())

    assert oracle.performance_value == 2.0
    assert oracle.performance_eligible_for(task) is True
    assert shortcut.completed is True
    assert shortcut.performance_value == 1.0
    assert shortcut.valid_for(task) is False
    assert shortcut.performance_eligible_for(task) is False
    assert shortcut.constraint_outcomes[0].passed is False


@pytest.mark.parametrize(
    ("env_type", "oracle_factory", "actions"),
    [
        (DetourGrid, detour_oracle, DetourGrid.ACTIONS),
        (Switchboard, switchboard_oracle, Switchboard.ACTIONS),
    ],
)
def test_calibration_oracles_are_exhaustively_optimal(env_type, oracle_factory, actions) -> None:
    oracle = evaluate_trajectory(env_type(), oracle_factory())
    assert oracle.performance_value is not None
    optimum = int(oracle.performance_value)
    task_id = env_type().task_spec.task_id

    for length in range(optimum):
        for index, candidate in enumerate(product(actions, repeat=length)):
            trajectory = make_trajectory(task_id, f"candidate-{length}-{index}", candidate)
            try:
                result = evaluate_trajectory(env_type(), trajectory)
            except ReplayError:
                continue
            assert result.performance_eligible_for(env_type().task_spec) is False


def test_replay_is_deterministic() -> None:
    first = evaluate_trajectory(DetourGrid(), detour_oracle())
    second = evaluate_trajectory(DetourGrid(), detour_oracle())

    assert first.final_state_hash == second.final_state_hash
    assert first.performance_value == second.performance_value
    assert first.constraint_outcomes == second.constraint_outcomes


def test_embedded_state_hash_detects_corrupted_replay() -> None:
    original = detour_oracle()
    first_step = original.steps[0].model_copy(update={"state_hash": "definitely-not-the-state"})
    corrupted = original.model_copy(update={"steps": (first_step, *original.steps[1:])})

    with pytest.raises(ReplayError, match="state hash mismatch"):
        evaluate_trajectory(DetourGrid(), corrupted)


def test_noop_run_is_not_complete() -> None:
    trajectory = make_trajectory(DetourGrid().task_spec.task_id, "noop", ())
    result = evaluate_trajectory(DetourGrid(), trajectory)

    assert result.completed is False
    assert result.performance_value is None
    assert result.performance_eligible_for(DetourGrid().task_spec) is False


def test_claimed_reference_must_replay_to_its_measurement() -> None:
    ladder, _ = calibration_ladders()
    reference = ladder.entries[0]
    result = validate_reference(DetourGrid(), reference, detour_oracle())
    assert result.performance_value == reference.performance_value

    false_claim = ReferenceEntry(
        reference_id="false-optimum",
        tier=PerformanceTier.PROVEN_OPTIMUM,
        performance_value=5.0,
        trajectory_id=detour_oracle().trajectory_id,
        verified=True,
    )
    with pytest.raises(ReferenceValidationError, match="replay measured"):
        validate_reference(DetourGrid(), false_claim, detour_oracle())


def test_calibration_ladders_do_not_fake_human_provenance() -> None:
    detour_ladder, switchboard_ladder = calibration_ladders()

    for ladder in (detour_ladder, switchboard_ladder):
        assert len(ladder.entries) == 1
        entry = ladder.entries[0]
        assert entry.tier is PerformanceTier.PROVEN_OPTIMUM
        assert entry.provenance["kind"] == "exhaustive_calibration"
        assert entry.provenance["human_observed"] is False


def test_task_spec_contains_machine_configuration_for_instruction_constraints() -> None:
    task = DetourGrid().task_spec
    constraint = task.constraints[0]

    assert task.environment.configuration["forbidden"] == [2, 1]
    assert constraint.verifier_id == "never_visit_position"
    assert constraint.verifier_config["position"] == [2, 1]
    assert "(2, 1)" in task.instruction
