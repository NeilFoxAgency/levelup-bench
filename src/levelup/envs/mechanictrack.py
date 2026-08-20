"""Mechanically varied synthetic tracks for Milestone 4 neural transfer.

Action names are randomized per task. Learners receive only numeric action descriptors and
normal environment observations, not stable semantic labels such as ``burst`` or ``cool``.
The benchmark evaluator retains exact state and an exhaustive shortest-path oracle.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import random
from dataclasses import dataclass

from pydantic import JsonValue

from levelup.core.experiment import ImprovementLadder, ImprovementStage
from levelup.core.result import ConstraintOutcome
from levelup.core.task import ConstraintSpec, EnvironmentSpec, ObjectiveSpec, TaskSpec
from levelup.core.trajectory import ActionRecord, Trajectory, TrajectoryStep
from levelup.envs.base import BenchmarkEnvironment, StepOutcome

TRAIN_FAMILIES = ("plain", "battery", "cooldown")
HELD_OUT_FAMILY = "heat"
FEATURE_NAMES = (
    "progress",
    "tick_cost",
    "uses_resource",
    "restores_resource",
    "raises_pressure",
    "clears_pressure",
    "is_enabler",
    "target_scale",
)


@dataclass(frozen=True, slots=True)
class ActionMechanic:
    alias: str
    progress: int
    tick_cost: int
    resource_use: int = 0
    resource_gain: int = 0
    pressure_gain: int = 0
    pressure_clear: int = 0
    forbidden: bool = False

    def feature_vector(self, target: int) -> tuple[float, ...]:
        """Agent-visible descriptor with no stable action-name feature."""

        is_enabler = self.progress == 0 and bool(self.resource_gain or self.pressure_clear)
        return (
            self.progress / 3.0,
            self.tick_cost / 13.0,
            float(bool(self.resource_use)),
            float(bool(self.resource_gain)),
            float(bool(self.pressure_gain)),
            float(bool(self.pressure_clear)),
            float(is_enabler),
            target / 16.0,
        )


State = tuple[int, int, int]  # progress, resource, pressure


def _stable_hash(payload: dict[str, JsonValue]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _transition(
    *,
    family: str,
    target: int,
    capacity: int,
    pressure_cap: int,
    state: State,
    action: ActionMechanic,
) -> tuple[State, int] | None:
    progress, resource, pressure = state

    if action.forbidden:
        return (target, resource, pressure), action.tick_cost

    if family == "plain":
        next_progress = progress + action.progress
        if next_progress > target:
            next_progress = progress
        return (next_progress, resource, pressure), action.tick_cost

    if family == "battery":
        if action.resource_use > resource:
            return None
        next_resource = min(
            capacity,
            resource - action.resource_use + action.resource_gain,
        )
        next_progress = progress + action.progress
        if next_progress > target:
            next_progress = progress
        return (next_progress, next_resource, pressure), action.tick_cost

    if family == "cooldown":
        if action.pressure_gain and pressure > 0:
            return None
        next_pressure = pressure
        if action.pressure_clear:
            next_pressure = max(0, pressure - action.pressure_clear)
        elif action.pressure_gain:
            next_pressure = min(pressure_cap, pressure + action.pressure_gain)
        elif pressure > 0:
            # Ordinary progress consumes one cooldown unit.
            next_pressure = max(0, pressure - 1)
        next_progress = progress + action.progress
        if next_progress > target:
            next_progress = progress
        return (next_progress, resource, next_pressure), action.tick_cost

    if family == "heat":
        if action.pressure_clear:
            next_pressure = max(0, pressure - action.pressure_clear)
        else:
            if pressure + action.pressure_gain > pressure_cap:
                return None
            next_pressure = pressure + action.pressure_gain
        next_progress = progress + action.progress
        if next_progress > target:
            next_progress = progress
        return (next_progress, resource, next_pressure), action.tick_cost

    raise ValueError(f"unsupported mechanic family: {family!r}")


class MechanicTrack(BenchmarkEnvironment):
    """One exact-progress task drawn from a mechanic family."""

    def __init__(
        self,
        *,
        family: str,
        target: int,
        actions: tuple[ActionMechanic, ...],
        task_index: int,
        generator_seed: int,
        capacity: int = 0,
        start_resource: int = 0,
        pressure_cap: int = 0,
    ) -> None:
        if family not in (*TRAIN_FAMILIES, HELD_OUT_FAMILY):
            raise ValueError(f"unsupported family: {family!r}")
        if target < 3:
            raise ValueError("target must be at least 3")
        if len({action.alias for action in actions}) != len(actions):
            raise ValueError("action aliases must be unique within a task")
        forbidden = [action for action in actions if action.forbidden]
        if len(forbidden) != 1:
            raise ValueError("MechanicTrack requires exactly one forbidden override action")

        self.family = family
        self.target = target
        self.actions = actions
        self.task_index = task_index
        self.generator_seed = generator_seed
        self.capacity = capacity
        self.start_resource = start_resource
        self.pressure_cap = pressure_cap
        self._action_map = {action.alias: action for action in actions}
        self._forbidden_alias = forbidden[0].alias
        self._task = TaskSpec(
            task_id=(
                f"micro.mechanic_track.{family}.s{generator_seed}.i{task_index}.v1"
            ),
            environment=EnvironmentSpec(
                adapter="microgame",
                environment_id="mechanic_track",
                version="1",
                seed=0,
                configuration={
                    "family": family,
                    "target": target,
                    "capacity": capacity,
                    "start_resource": start_resource,
                    "pressure_cap": pressure_cap,
                    "feature_names": list(FEATURE_NAMES),
                    "actions": [
                        {
                            "alias": action.alias,
                            "features": list(action.feature_vector(target)),
                        }
                        for action in actions
                    ],
                },
            ),
            instruction=(
                f"Reach progress {target} in as few ticks as possible. "
                f"Do not use action {self._forbidden_alias}."
            ),
            constraints=(
                ConstraintSpec(
                    constraint_id="never_use_override",
                    description=f"Do not use action {self._forbidden_alias}.",
                    verifier_id="never_use_action",
                    verifier_config={"action": self._forbidden_alias},
                ),
            ),
            objective=ObjectiveSpec(
                metric_id="elapsed_ticks",
                direction="minimize",
                unit="ticks",
            ),
            metadata={
                "suite": "neural_microtransfer",
                "synthetic": True,
                "family": family,
                "randomized_action_aliases": True,
            },
        )
        self._progress = 0
        self._resource = start_resource
        self._pressure = 0
        self._elapsed_ticks = 0
        self._actions_taken: list[str] = []

    @property
    def task_spec(self) -> TaskSpec:
        return self._task

    @property
    def valid_actions(self) -> tuple[ActionMechanic, ...]:
        return tuple(action for action in self.actions if not action.forbidden)

    def fresh(self) -> "MechanicTrack":
        return MechanicTrack(
            family=self.family,
            target=self.target,
            actions=self.actions,
            task_index=self.task_index,
            generator_seed=self.generator_seed,
            capacity=self.capacity,
            start_resource=self.start_resource,
            pressure_cap=self.pressure_cap,
        )

    def reset(self, seed: int | None = None) -> StepOutcome:
        if seed not in (None, 0):
            raise ValueError("MechanicTrack v1 supports only deterministic environment seed 0")
        self._progress = 0
        self._resource = self.start_resource
        self._pressure = 0
        self._elapsed_ticks = 0
        self._actions_taken = []
        return self._outcome()

    def _state(self) -> State:
        return self._progress, self._resource, self._pressure

    def _transition(self, state: State, action: ActionMechanic) -> tuple[State, int] | None:
        return _transition(
            family=self.family,
            target=self.target,
            capacity=self.capacity,
            pressure_cap=self.pressure_cap,
            state=state,
            action=action,
        )

    def step(self, action: ActionRecord) -> StepOutcome:
        if action.arguments:
            raise ValueError("MechanicTrack actions do not take arguments")
        mechanic = self._action_map.get(action.name)
        if mechanic is None:
            raise ValueError(f"unsupported MechanicTrack action: {action.name!r}")
        transitioned = self._transition(self._state(), mechanic)
        if transitioned is None:
            raise ValueError(f"action {action.name!r} is unavailable in the current state")
        (self._progress, self._resource, self._pressure), tick_cost = transitioned
        self._elapsed_ticks += tick_cost
        self._actions_taken.append(action.name)
        return self._outcome()

    def available_actions(self) -> tuple[ActionMechanic, ...]:
        available: list[ActionMechanic] = []
        state = self._state()
        for action in self.actions:
            transitioned = self._transition(state, action)
            if transitioned is None:
                continue
            next_state, _ = transitioned
            # Overshooting progress actions are known to be unavailable, matching MacroTrack.
            if action.progress > 0 and not action.forbidden and next_state[0] == state[0]:
                continue
            available.append(action)
        return tuple(available)

    def verify_constraint(self, constraint: ConstraintSpec) -> ConstraintOutcome:
        if constraint.verifier_id != "never_use_action":
            raise ValueError(f"unsupported verifier: {constraint.verifier_id}")
        forbidden = constraint.verifier_config.get("action")
        if not isinstance(forbidden, str):
            raise ValueError("never_use_action requires an action string")
        passed = forbidden not in self._actions_taken
        return ConstraintOutcome(
            constraint_id=constraint.constraint_id,
            passed=passed,
            evidence=(
                f"action {forbidden!r} was never used"
                if passed
                else f"action {forbidden!r} was used"
            ),
        )

    def objective_value(self) -> float:
        return float(self._elapsed_ticks)

    def state_hash(self) -> str:
        return _stable_hash(
            {
                "family": self.family,
                "progress": self._progress,
                "resource": self._resource,
                "pressure": self._pressure,
                "elapsed_ticks": self._elapsed_ticks,
                "actions": self._actions_taken,
            }
        )

    def _outcome(self) -> StepOutcome:
        available = self.available_actions()
        return StepOutcome(
            observation={
                "progress": self._progress,
                "target": self.target,
                "elapsed_ticks": self._elapsed_ticks,
                "resource_fraction": (
                    self._resource / self.capacity if self.capacity else 0.0
                ),
                "pressure_fraction": (
                    self._pressure / self.pressure_cap if self.pressure_cap else 0.0
                ),
                "available_actions": [
                    {
                        "alias": action.alias,
                        "features": list(action.feature_vector(self.target)),
                    }
                    for action in available
                ],
            },
            completed=self._progress == self.target,
            state_hash=self.state_hash(),
        )


def _random_aliases(rng: random.Random, count: int) -> tuple[str, ...]:
    aliases: list[str] = []
    while len(aliases) < count:
        candidate = f"a{rng.randrange(16**6):06x}"
        if candidate not in aliases:
            aliases.append(candidate)
    return tuple(aliases)


def make_mechanic_track(family: str, task_index: int, generator_seed: int) -> MechanicTrack:
    """Generate a deterministic task with fresh opaque action aliases."""

    if family not in (*TRAIN_FAMILIES, HELD_OUT_FAMILY):
        raise ValueError(f"unsupported family: {family!r}")
    rng = random.Random(generator_seed * 10_000 + task_index)
    target = rng.randint(9, 16)
    alias_step, alias_stride, alias_burst, alias_utility, alias_override = _random_aliases(
        rng, 5
    )
    step = ActionMechanic(alias_step, progress=1, tick_cost=rng.randint(6, 8))
    stride = ActionMechanic(alias_stride, progress=2, tick_cost=rng.randint(9, 11))

    if family == "plain":
        burst = ActionMechanic(alias_burst, progress=3, tick_cost=rng.randint(11, 13))
        actions = (step, stride, burst)
        capacity = start_resource = pressure_cap = 0
    elif family == "battery":
        burst = ActionMechanic(
            alias_burst,
            progress=3,
            tick_cost=rng.randint(8, 10),
            resource_use=2,
        )
        utility = ActionMechanic(
            alias_utility,
            progress=0,
            tick_cost=rng.randint(3, 5),
            resource_gain=4,
        )
        actions = (step, stride, burst, utility)
        capacity, start_resource, pressure_cap = 4, 2, 0
    elif family == "cooldown":
        burst = ActionMechanic(
            alias_burst,
            progress=3,
            tick_cost=rng.randint(8, 10),
            pressure_gain=1,
        )
        utility = ActionMechanic(
            alias_utility,
            progress=0,
            tick_cost=rng.randint(3, 4),
            pressure_clear=1,
        )
        actions = (step, stride, burst, utility)
        capacity, start_resource, pressure_cap = 0, 0, 1
    else:  # heat, held out in the default Milestone 4 experiment
        burst = ActionMechanic(
            alias_burst,
            progress=3,
            tick_cost=rng.randint(8, 10),
            pressure_gain=2,
        )
        stride = ActionMechanic(
            alias_stride,
            progress=2,
            tick_cost=rng.randint(9, 11),
            pressure_gain=1,
        )
        utility = ActionMechanic(
            alias_utility,
            progress=0,
            tick_cost=rng.randint(3, 5),
            pressure_clear=3,
        )
        actions = (step, stride, burst, utility)
        capacity, start_resource, pressure_cap = 0, 0, 3

    override = ActionMechanic(
        alias_override,
        progress=10_000,
        tick_cost=1,
        forbidden=True,
    )
    return MechanicTrack(
        family=family,
        target=target,
        actions=(*actions, override),
        task_index=task_index,
        generator_seed=generator_seed,
        capacity=capacity,
        start_resource=start_resource,
        pressure_cap=pressure_cap,
    )


def _search_path(environment: MechanicTrack, *, penalize_complexity: bool) -> tuple[int, list[str]]:
    """Exact shortest path, optionally under a deliberately biased perceived cost."""

    start: State = (0, environment.start_resource, 0)
    queue: list[tuple[int, int, State, list[str]]] = [(0, 0, start, [])]
    best = {start: 0}
    counter = 0
    while queue:
        perceived_cost, _, state, path = heapq.heappop(queue)
        if perceived_cost != best[state]:
            continue
        if state[0] == environment.target:
            true_cost = sum(environment._action_map[name].tick_cost for name in path)
            return true_cost, path
        for action in environment.valid_actions:
            transitioned = environment._transition(state, action)
            if transitioned is None:
                continue
            next_state, tick_cost = transitioned
            if action.progress > 0 and next_state[0] == state[0]:
                continue
            candidate_tick_cost = tick_cost
            if penalize_complexity:
                multiplier = 100
                if (
                    action.resource_use
                    or action.resource_gain
                    or action.pressure_gain
                    or action.pressure_clear
                ):
                    multiplier += 45
                if action.progress >= 3:
                    multiplier += 25
                candidate_tick_cost *= multiplier
            next_cost = perceived_cost + candidate_tick_cost
            if next_cost < best.get(next_state, 10**18):
                best[next_state] = next_cost
                counter += 1
                heapq.heappush(queue, (next_cost, counter, next_state, path + [action.alias]))
    raise RuntimeError("MechanicTrack task has no valid completion")


def optimal_path(environment: MechanicTrack) -> tuple[int, list[str]]:
    return _search_path(environment, penalize_complexity=False)


def frontier_path(environment: MechanicTrack) -> tuple[int, list[str]]:
    return _search_path(environment, penalize_complexity=True)


def _trajectory(task_id: str, trajectory_id: str, actions: list[str]) -> Trajectory:
    return Trajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        source="reference",
        steps=tuple(
            TrajectoryStep(index=index, action=ActionRecord(name=name))
            for index, name in enumerate(actions)
        ),
    )


@dataclass(frozen=True, slots=True)
class MechanicTrackBundle:
    environment: MechanicTrack
    ladder: ImprovementLadder
    trajectories: dict[str, Trajectory]

    def trajectory_for(self, label: str) -> Trajectory:
        return self.trajectories[self.ladder.stage(label).trajectory_id]


def mechanic_track_bundle(
    family: str,
    task_index: int,
    generator_seed: int,
) -> MechanicTrackBundle:
    """Build a two-stage frontier -> optimum ladder.

    Tasks whose complexity-biased frontier accidentally equals the optimum are rejected so the
    experiment never fabricates an improvement transition that is not actually present.
    """

    environment = make_mechanic_track(family, task_index, generator_seed)
    frontier_cost, frontier_actions = frontier_path(environment)
    optimum_cost, optimum_actions = optimal_path(environment)
    if frontier_cost <= optimum_cost or frontier_actions == optimum_actions:
        raise ValueError("generated task has no strict frontier-to-optimum gap")

    trajectories: dict[str, Trajectory] = {}
    stages: list[ImprovementStage] = []
    for ordinal, (label, cost, actions) in enumerate(
        (
            ("frontier", frontier_cost, frontier_actions),
            ("optimum", optimum_cost, optimum_actions),
        )
    ):
        trajectory_id = f"{environment.task_spec.task_id}.{label}"
        trajectories[trajectory_id] = _trajectory(
            environment.task_spec.task_id,
            trajectory_id,
            actions,
        )
        stages.append(
            ImprovementStage(
                stage_id=f"{environment.task_spec.task_id}:{label}",
                ordinal=ordinal,
                label=label,
                trajectory_id=trajectory_id,
                performance_value=float(cost),
                provenance={
                    "kind": "synthetic_policy",
                    "human_observed": False,
                    "generated_from_hidden_oracle": label == "optimum",
                    "mechanic_family": family,
                },
            )
        )
    return MechanicTrackBundle(
        environment=environment,
        ladder=ImprovementLadder(
            task_id=environment.task_spec.task_id,
            direction="minimize",
            stages=tuple(stages),
        ),
        trajectories=trajectories,
    )


def collect_bundles(
    family: str,
    count: int,
    generator_seed: int,
) -> tuple[MechanicTrackBundle, ...]:
    bundles: list[MechanicTrackBundle] = []
    task_index = 0
    while len(bundles) < count:
        try:
            bundle = mechanic_track_bundle(family, task_index, generator_seed)
        except ValueError as exc:
            if "no strict frontier-to-optimum gap" not in str(exc):
                raise
        else:
            bundles.append(bundle)
        task_index += 1
    return tuple(bundles)


def held_out_tasks(
    count: int,
    generator_seed: int,
) -> tuple[tuple[MechanicTrack, float], ...]:
    """Return held-out environments and optimum values without exposing optimum trajectories."""

    tasks: list[tuple[MechanicTrack, float]] = []
    task_index = 0
    while len(tasks) < count:
        environment = make_mechanic_track(HELD_OUT_FAMILY, task_index, generator_seed)
        frontier_cost, frontier_actions = frontier_path(environment)
        optimum_cost, optimum_actions = optimal_path(environment)
        if frontier_cost > optimum_cost and frontier_actions != optimum_actions:
            tasks.append((environment, float(optimum_cost)))
        task_index += 1
    return tuple(tasks)
