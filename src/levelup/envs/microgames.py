"""Tiny deterministic calibration environments for LevelUp Bench v0.2.

These environments exist to test benchmark semantics, not to model realistic
video-game difficulty. They are deliberately small enough that their valid
optima can be exhaustively checked in unit tests.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from pydantic import JsonValue

from levelup.core.reference import PerformanceTier, ReferenceEntry, ReferenceLadder
from levelup.core.result import ConstraintOutcome
from levelup.core.task import ConstraintSpec, EnvironmentSpec, ObjectiveSpec, TaskSpec
from levelup.core.trajectory import ActionRecord, Trajectory, TrajectoryStep
from levelup.envs.base import BenchmarkEnvironment, StepOutcome


def _stable_hash(payload: dict[str, JsonValue]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trajectory(
    *,
    trajectory_id: str,
    task_id: str,
    actions: Iterable[str],
    source: str = "oracle",
) -> Trajectory:
    return Trajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        source=source,
        steps=tuple(
            TrajectoryStep(index=index, action=ActionRecord(name=action))
            for index, action in enumerate(actions)
        ),
    )


class DetourGrid(BenchmarkEnvironment):
    """Reach a goal while avoiding a tempting shorter forbidden tile."""

    ACTIONS = ("up", "down", "left", "right")
    START = (0, 1)
    GOAL = (4, 1)
    FORBIDDEN = (2, 1)
    WIDTH = 5
    HEIGHT = 3

    _TASK = TaskSpec(
        task_id="micro.detour_grid.v1",
        environment=EnvironmentSpec(
            adapter="microgame",
            environment_id="detour_grid",
            version="1",
            seed=0,
            configuration={
                "width": WIDTH,
                "height": HEIGHT,
                "start": list(START),
                "goal": list(GOAL),
                "forbidden": list(FORBIDDEN),
            },
        ),
        instruction=(
            "Reach the goal as quickly as possible. Do not step on the forbidden tile "
            "at coordinate (2, 1)."
        ),
        constraints=(
            ConstraintSpec(
                constraint_id="avoid_forbidden_tile",
                description="Do not step on the forbidden tile at coordinate (2, 1).",
                verifier_id="never_visit_position",
                verifier_config={"position": list(FORBIDDEN)},
            ),
        ),
        objective=ObjectiveSpec(metric_id="action_count", direction="minimize", unit="actions"),
        metadata={"suite": "microgames", "calibration": True},
    )

    def __init__(self) -> None:
        self._position = self.START
        self._visited: list[tuple[int, int]] = [self.START]
        self._action_count = 0

    @property
    def task_spec(self) -> TaskSpec:
        return self._TASK

    def reset(self, seed: int | None = None) -> StepOutcome:
        if seed not in (None, 0):
            raise ValueError("DetourGrid v1 supports only deterministic seed 0")
        self._position = self.START
        self._visited = [self.START]
        self._action_count = 0
        return self._outcome()

    def step(self, action: ActionRecord) -> StepOutcome:
        if action.arguments:
            raise ValueError("DetourGrid actions do not take arguments")
        if action.name not in self.ACTIONS:
            raise ValueError(f"unsupported DetourGrid action: {action.name!r}")

        dx, dy = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }[action.name]
        x, y = self._position
        next_position = (x + dx, y + dy)
        if 0 <= next_position[0] < self.WIDTH and 0 <= next_position[1] < self.HEIGHT:
            self._position = next_position

        self._action_count += 1
        self._visited.append(self._position)
        return self._outcome()

    def verify_constraint(self, constraint: ConstraintSpec) -> ConstraintOutcome:
        if constraint.verifier_id != "never_visit_position":
            raise ValueError(f"unsupported verifier: {constraint.verifier_id}")
        raw_position = constraint.verifier_config.get("position")
        if not isinstance(raw_position, list) or len(raw_position) != 2:
            raise ValueError("never_visit_position requires a two-element position")
        forbidden = (int(raw_position[0]), int(raw_position[1]))
        passed = forbidden not in self._visited
        evidence = (
            f"position {forbidden} was never visited"
            if passed
            else f"position {forbidden} was visited"
        )
        return ConstraintOutcome(
            constraint_id=constraint.constraint_id,
            passed=passed,
            evidence=evidence,
        )

    def objective_value(self) -> float:
        return float(self._action_count)

    def state_hash(self) -> str:
        return _stable_hash(
            {
                "position": list(self._position),
                "visited_forbidden": self.FORBIDDEN in self._visited,
                "action_count": self._action_count,
            }
        )

    def _outcome(self) -> StepOutcome:
        return StepOutcome(
            observation={
                "position": list(self._position),
                "goal": list(self.GOAL),
                "width": self.WIDTH,
                "height": self.HEIGHT,
            },
            completed=self._position == self.GOAL,
            state_hash=self.state_hash(),
        )


class Switchboard(BenchmarkEnvironment):
    """Turn on four lamps while refusing a one-step forbidden shortcut."""

    ACTIONS = ("blue", "green", "red")
    TARGET = 0b1111
    EFFECTS = {"blue": 0b0011, "green": 0b1100, "red": 0b1111}

    _TASK = TaskSpec(
        task_id="micro.switchboard.v1",
        environment=EnvironmentSpec(
            adapter="microgame",
            environment_id="switchboard",
            version="1",
            seed=0,
            configuration={
                "initial_mask": 0,
                "target_mask": TARGET,
                "effects": {name: effect for name, effect in EFFECTS.items()},
            },
        ),
        instruction=(
            "Turn on all four lamps using as few switch presses as possible. "
            "Do not press the red switch."
        ),
        constraints=(
            ConstraintSpec(
                constraint_id="never_press_red",
                description="Do not press the red switch.",
                verifier_id="never_use_action",
                verifier_config={"action": "red"},
            ),
        ),
        objective=ObjectiveSpec(metric_id="action_count", direction="minimize", unit="actions"),
        metadata={"suite": "microgames", "calibration": True},
    )

    def __init__(self) -> None:
        self._mask = 0
        self._actions: list[str] = []

    @property
    def task_spec(self) -> TaskSpec:
        return self._TASK

    def reset(self, seed: int | None = None) -> StepOutcome:
        if seed not in (None, 0):
            raise ValueError("Switchboard v1 supports only deterministic seed 0")
        self._mask = 0
        self._actions = []
        return self._outcome()

    def step(self, action: ActionRecord) -> StepOutcome:
        if action.arguments:
            raise ValueError("Switchboard actions do not take arguments")
        if action.name not in self.ACTIONS:
            raise ValueError(f"unsupported Switchboard action: {action.name!r}")

        self._mask ^= self.EFFECTS[action.name]
        self._actions.append(action.name)
        return self._outcome()

    def verify_constraint(self, constraint: ConstraintSpec) -> ConstraintOutcome:
        if constraint.verifier_id != "never_use_action":
            raise ValueError(f"unsupported verifier: {constraint.verifier_id}")
        forbidden = constraint.verifier_config.get("action")
        if not isinstance(forbidden, str):
            raise ValueError("never_use_action requires an action string")
        passed = forbidden not in self._actions
        evidence = (
            f"action {forbidden!r} was never used"
            if passed
            else f"action {forbidden!r} was used"
        )
        return ConstraintOutcome(
            constraint_id=constraint.constraint_id,
            passed=passed,
            evidence=evidence,
        )

    def objective_value(self) -> float:
        return float(len(self._actions))

    def state_hash(self) -> str:
        return _stable_hash(
            {
                "mask": self._mask,
                "used_red": "red" in self._actions,
                "action_count": len(self._actions),
            }
        )

    def _outcome(self) -> StepOutcome:
        return StepOutcome(
            observation={
                "lamps": [bool(self._mask & (1 << bit)) for bit in range(4)],
                "switches": list(self.ACTIONS),
            },
            completed=self._mask == self.TARGET,
            state_hash=self.state_hash(),
        )


def detour_oracle() -> Trajectory:
    return _trajectory(
        trajectory_id="micro.detour_grid.oracle.v1",
        task_id=DetourGrid._TASK.task_id,
        actions=("up", "right", "right", "right", "right", "down"),
    )


def detour_invalid_shortcut() -> Trajectory:
    return _trajectory(
        trajectory_id="micro.detour_grid.invalid_shortcut.v1",
        task_id=DetourGrid._TASK.task_id,
        actions=("right", "right", "right", "right"),
        source="reference",
    )


def switchboard_oracle() -> Trajectory:
    return _trajectory(
        trajectory_id="micro.switchboard.oracle.v1",
        task_id=Switchboard._TASK.task_id,
        actions=("blue", "green"),
    )


def switchboard_invalid_shortcut() -> Trajectory:
    return _trajectory(
        trajectory_id="micro.switchboard.invalid_shortcut.v1",
        task_id=Switchboard._TASK.task_id,
        actions=("red",),
        source="reference",
    )


def calibration_ladders() -> tuple[ReferenceLadder, ReferenceLadder]:
    """Return verified claims whose trajectories are checked by the test suite."""

    detour = ReferenceLadder(
        task_id=DetourGrid._TASK.task_id,
        entries=(
            ReferenceEntry(
                reference_id="micro.detour_grid.optimum.v1",
                tier=PerformanceTier.PROVEN_OPTIMUM,
                performance_value=6.0,
                trajectory_id=detour_oracle().trajectory_id,
                verified=True,
                provenance={"kind": "exhaustive_calibration", "human_observed": False},
            ),
        ),
    )
    switchboard = ReferenceLadder(
        task_id=Switchboard._TASK.task_id,
        entries=(
            ReferenceEntry(
                reference_id="micro.switchboard.optimum.v1",
                tier=PerformanceTier.PROVEN_OPTIMUM,
                performance_value=2.0,
                trajectory_id=switchboard_oracle().trajectory_id,
                verified=True,
                provenance={"kind": "exhaustive_calibration", "human_observed": False},
            ),
        ),
    )
    return detour, switchboard
