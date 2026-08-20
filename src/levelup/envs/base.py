"""Minimal environment contract for deterministic LevelUp evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

from pydantic import JsonValue

from levelup.core.result import ConstraintOutcome
from levelup.core.task import ConstraintSpec, TaskSpec
from levelup.core.trajectory import ActionRecord

Observation = Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """Observable result after exactly one environment action."""

    observation: Observation
    completed: bool
    state_hash: str


class BenchmarkEnvironment(ABC):
    """Small contract required by the LevelUp replay evaluator.

    Environment implementations may keep privileged state internally. Only
    ``observation`` is agent-facing. Constraint verification and state hashing
    are evaluator-facing and may inspect privileged state.
    """

    @property
    @abstractmethod
    def task_spec(self) -> TaskSpec:
        """Return the canonical task specification for this environment instance."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> StepOutcome:
        """Reset to the task's initial state."""

    @abstractmethod
    def step(self, action: ActionRecord) -> StepOutcome:
        """Apply exactly one action."""

    @abstractmethod
    def verify_constraint(self, constraint: ConstraintSpec) -> ConstraintOutcome:
        """Evaluate one declared hard constraint from privileged state/history."""

    @abstractmethod
    def objective_value(self) -> float:
        """Return the task objective for the trajectory executed so far."""

    @abstractmethod
    def state_hash(self) -> str:
        """Return a deterministic hash of evaluator-relevant state."""
