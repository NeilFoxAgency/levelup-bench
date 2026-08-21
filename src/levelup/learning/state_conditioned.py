"""Observation-only representations for Milestone 6 development baselines.

This module is deliberately separated from environment construction and evaluation. Feature
builders accept sanitized observations and observed transitions, never benchmark environments,
task specifications, hidden action descriptors, state hashes, or optimum thresholds.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

OBSERVATION_KEYS = frozenset(
    {
        "progress",
        "target",
        "elapsed_ticks",
        "resource_fraction",
        "pressure_fraction",
        "available_actions",
    }
)
STATE_FEATURE_COUNT = 5
TRANSITION_FEATURE_COUNT = 12
PROBE_FEATURE_COUNT = TRANSITION_FEATURE_COUNT * 4 + 1
STATE_CONDITIONED_FEATURE_COUNT = STATE_FEATURE_COUNT + PROBE_FEATURE_COUNT


@dataclass(frozen=True, slots=True)
class ObservableState:
    """The complete numeric and action-list state permitted to a Milestone 6 learner."""

    progress_fraction: float
    remaining_fraction: float
    elapsed_per_target: float
    resource_fraction: float
    pressure_fraction: float
    available_aliases: tuple[str, ...]

    def features(self) -> tuple[float, ...]:
        return (
            self.progress_fraction,
            self.remaining_fraction,
            self.elapsed_per_target,
            self.resource_fraction,
            self.pressure_fraction,
        )


@dataclass(frozen=True, slots=True)
class ObservedTransition:
    """One agent-visible action consequence with no evaluator or task identity fields."""

    before: ObservableState
    action_alias: str
    after: ObservableState
    completed: bool


@dataclass(frozen=True, slots=True)
class ObservableTrace:
    """A sanitized reference replay consumed by state-conditioned training."""

    transitions: tuple[ObservedTransition, ...]

    def __post_init__(self) -> None:
        for index, transition in enumerate(self.transitions):
            if transition.action_alias not in transition.before.available_aliases:
                raise ValueError("trace action is unavailable in its pre-action observation")
            if index and self.transitions[index - 1].after != transition.before:
                raise ValueError("observable trace transitions do not form a contiguous chain")
            if index and self.transitions[index - 1].completed:
                raise ValueError("observable trace contains a transition after completion")


@dataclass(frozen=True, slots=True)
class AffordanceTable:
    """Fixed-width empirical action features learned from declared interactions."""

    features: Mapping[str, tuple[float, ...]]
    sample_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if set(self.features) != set(self.sample_counts):
            raise ValueError("affordance feature and sample-count aliases must match")
        if any(len(row) != PROBE_FEATURE_COUNT for row in self.features.values()):
            raise ValueError("unexpected affordance feature width")
        if any(count < 1 for count in self.sample_counts.values()):
            raise ValueError("affordance sample counts must be positive")

    def for_alias(self, alias: str) -> tuple[float, ...] | None:
        return self.features.get(alias)


@dataclass(frozen=True, slots=True)
class DecisionExample:
    """One listwise optimum-imitation decision with alias-free model inputs."""

    candidate_features: torch.Tensor
    selected_index: int

    def __post_init__(self) -> None:
        if self.candidate_features.ndim != 2:
            raise ValueError("candidate features must be a rank-two tensor")
        if self.candidate_features.shape[1] != STATE_CONDITIONED_FEATURE_COUNT:
            raise ValueError("unexpected state-conditioned feature width")
        if not 0 <= self.selected_index < self.candidate_features.shape[0]:
            raise ValueError("selected index is outside the candidate set")


class StateConditionedScorer(nn.Module):
    """Score observed action affordances in the current observable state."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_CONDITIONED_FEATURE_COUNT, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class GlobalAffordanceScorer(nn.Module):
    """Milestone 5-width action scorer with no current-state model input."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(PROBE_FEATURE_COUNT, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    epochs: int
    learning_rate: float
    weight_decay: float = 0.0001

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")


@dataclass(frozen=True, slots=True)
class TrainingReport:
    trainable_parameters: int
    optimizer_steps: int
    forward_passes: int
    training_examples: int


@dataclass(frozen=True, slots=True)
class GlobalDecisionExample:
    candidate_features: torch.Tensor
    selected_index: int

    def __post_init__(self) -> None:
        if self.candidate_features.ndim != 2:
            raise ValueError("candidate features must be a rank-two tensor")
        if self.candidate_features.shape[1] != PROBE_FEATURE_COUNT:
            raise ValueError("unexpected global-affordance feature width")
        if not 0 <= self.selected_index < self.candidate_features.shape[0]:
            raise ValueError("selected index is outside the candidate set")


def parse_observation(
    observation: Any,
    *,
    forbidden_aliases: frozenset[str] = frozenset(),
) -> ObservableState:
    """Validate and convert the complete opaque-action v1 observation surface."""

    if not isinstance(observation, dict):
        raise RuntimeError("observation must be a dictionary")
    unexpected = set(observation) - OBSERVATION_KEYS
    missing = OBSERVATION_KEYS - set(observation)
    if unexpected or missing:
        raise RuntimeError(
            f"observation schema mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    progress = observation["progress"]
    target = observation["target"]
    elapsed = observation["elapsed_ticks"]
    resource = observation["resource_fraction"]
    pressure = observation["pressure_fraction"]
    if isinstance(progress, bool) or not isinstance(progress, int) or progress < 0:
        raise RuntimeError("progress must be a nonnegative integer")
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise RuntimeError("target must be a positive integer")
    if progress > target:
        raise RuntimeError("progress cannot exceed target")
    if isinstance(elapsed, bool) or not isinstance(elapsed, int) or elapsed < 0:
        raise RuntimeError("elapsed_ticks must be a nonnegative integer")
    if (
        isinstance(resource, bool)
        or not isinstance(resource, (int, float))
        or not 0.0 <= float(resource) <= 1.0
    ):
        raise RuntimeError("resource_fraction must be numeric in [0, 1]")
    if (
        isinstance(pressure, bool)
        or not isinstance(pressure, (int, float))
        or not 0.0 <= float(pressure) <= 1.0
    ):
        raise RuntimeError("pressure_fraction must be numeric in [0, 1]")

    raw_actions = observation["available_actions"]
    if not isinstance(raw_actions, list):
        raise RuntimeError("available_actions must be a list")
    aliases: list[str] = []
    for item in raw_actions:
        if not isinstance(item, dict) or set(item) != {"alias"}:
            raise RuntimeError("available actions must contain opaque aliases only")
        alias = item["alias"]
        if not isinstance(alias, str) or not alias:
            raise RuntimeError("action alias must be a nonempty string")
        if alias not in forbidden_aliases:
            aliases.append(alias)
    if len(aliases) != len(set(aliases)):
        raise RuntimeError("available action aliases must be unique")

    return ObservableState(
        progress_fraction=progress / target,
        remaining_fraction=(target - progress) / target,
        elapsed_per_target=elapsed / target,
        resource_fraction=float(resource),
        pressure_fraction=float(pressure),
        available_aliases=tuple(aliases),
    )


def transition_features(transition: ObservedTransition) -> tuple[float, ...]:
    """Encode only measured before/after effects; aliases never enter the vector."""

    before = transition.before
    after = transition.after
    return (
        before.progress_fraction,
        before.remaining_fraction,
        before.resource_fraction,
        before.pressure_fraction,
        after.progress_fraction - before.progress_fraction,
        after.elapsed_per_target - before.elapsed_per_target,
        after.resource_fraction - before.resource_fraction,
        after.pressure_fraction - before.pressure_fraction,
        after.resource_fraction,
        after.pressure_fraction,
        float(transition.completed),
        len(before.available_aliases) / 5.0,
    )


def build_affordance_table(
    transitions: Sequence[ObservedTransition],
    *,
    target_samples_per_alias: int,
) -> AffordanceTable:
    """Summarize observed effects without assuming a hidden global action catalogue."""

    if target_samples_per_alias < 1:
        raise ValueError("target_samples_per_alias must be positive")
    grouped: dict[str, list[tuple[float, ...]]] = {}
    for transition in transitions:
        grouped.setdefault(transition.action_alias, []).append(
            transition_features(transition)
        )
    if not grouped:
        raise ValueError("at least one observed transition is required")

    features: dict[str, tuple[float, ...]] = {}
    counts: dict[str, int] = {}
    for alias in sorted(grouped):
        rows = torch.tensor(grouped[alias], dtype=torch.float32)
        summary = torch.cat(
            (
                rows.mean(dim=0),
                rows.std(dim=0, unbiased=False),
                rows.min(dim=0).values,
                rows.max(dim=0).values,
                torch.tensor(
                    [min(len(rows) / target_samples_per_alias, 1.0)],
                    dtype=torch.float32,
                ),
            )
        )
        features[alias] = tuple(float(value) for value in summary.tolist())
        counts[alias] = len(rows)
    return AffordanceTable(features=features, sample_counts=counts)


def candidate_tensor(
    state: ObservableState,
    affordances: AffordanceTable,
) -> tuple[tuple[str, ...], torch.Tensor, int]:
    """Build model inputs for current visible actions and count unknown affordances."""

    if not state.available_aliases:
        raise ValueError("cannot score a state without available actions")
    zero_affordance = (0.0,) * PROBE_FEATURE_COUNT
    rows: list[tuple[float, ...]] = []
    unknown = 0
    for alias in state.available_aliases:
        affordance = affordances.for_alias(alias)
        if affordance is None:
            affordance = zero_affordance
            unknown += 1
        rows.append((*state.features(), *affordance))
    return (
        state.available_aliases,
        torch.tensor(rows, dtype=torch.float32),
        unknown,
    )


def optimum_imitation_examples(
    samples: Sequence[tuple[ObservableTrace, AffordanceTable]],
) -> tuple[DecisionExample, ...]:
    """Build optimum-only listwise examples from already sanitized reference traces."""

    examples: list[DecisionExample] = []
    for trace, affordances in samples:
        for transition in trace.transitions:
            aliases, features, _ = candidate_tensor(transition.before, affordances)
            try:
                selected = aliases.index(transition.action_alias)
            except ValueError as exc:
                raise ValueError("reference selected an unavailable action") from exc
            examples.append(DecisionExample(features, selected))
    if not examples:
        raise ValueError("at least one optimum decision is required")
    return tuple(examples)


def global_listwise_optimum_examples(
    samples: Sequence[tuple[ObservableTrace, AffordanceTable]],
) -> tuple[GlobalDecisionExample, ...]:
    """Build the objective-matched action-only counterpart to state conditioning."""

    examples: list[GlobalDecisionExample] = []
    zero_affordance = (0.0,) * PROBE_FEATURE_COUNT
    for trace, affordances in samples:
        for transition in trace.transitions:
            aliases = transition.before.available_aliases
            rows = [affordances.for_alias(alias) or zero_affordance for alias in aliases]
            try:
                selected = aliases.index(transition.action_alias)
            except ValueError as exc:
                raise ValueError("reference selected an unavailable action") from exc
            examples.append(
                GlobalDecisionExample(
                    torch.tensor(rows, dtype=torch.float32),
                    selected,
                )
            )
    if not examples:
        raise ValueError("at least one optimum decision is required")
    return tuple(examples)


def global_frequency_optimum_examples(
    samples: Sequence[tuple[ObservableTrace, AffordanceTable]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean optimum-only Milestone 5 continuity examples with no frontier access."""

    rows: list[tuple[float, ...]] = []
    targets: list[float] = []
    zero_affordance = (0.0,) * PROBE_FEATURE_COUNT
    for trace, affordances in samples:
        selected = Counter(
            transition.action_alias for transition in trace.transitions
        )
        aliases = sorted(
            set(affordances.features).union(
                alias
                for transition in trace.transitions
                for alias in transition.before.available_aliases
            )
        )
        total = len(trace.transitions)
        for alias in aliases:
            rows.append(affordances.for_alias(alias) or zero_affordance)
            targets.append(selected.get(alias, 0) / total)
    if not rows:
        raise ValueError("at least one optimum action is required")
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(
        targets,
        dtype=torch.float32,
    )


def train_state_conditioned_optimum_model(
    examples: Sequence[DecisionExample],
    *,
    training: TrainingSpec,
    model_seed: int,
) -> tuple[StateConditionedScorer, TrainingReport]:
    """Train the minimal listwise state-conditioned optimum-imitation baseline."""

    if not examples:
        raise ValueError("at least one training example is required")
    torch.manual_seed(model_seed)
    model = StateConditionedScorer()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    for _ in range(training.epochs):
        optimizer.zero_grad()
        losses = [
            nn.functional.cross_entropy(
                model(example.candidate_features).unsqueeze(0),
                torch.tensor([example.selected_index], dtype=torch.long),
            )
            for example in examples
        ]
        torch.stack(losses).mean().backward()
        optimizer.step()
    model.eval()
    return model, TrainingReport(
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        optimizer_steps=training.epochs,
        forward_passes=training.epochs * len(examples),
        training_examples=len(examples),
    )


def train_global_listwise_optimum_model(
    examples: Sequence[GlobalDecisionExample],
    *,
    training: TrainingSpec,
    model_seed: int,
) -> tuple[GlobalAffordanceScorer, TrainingReport]:
    """Train the action-only, objective-matched state-conditioning control."""

    if not examples:
        raise ValueError("at least one training example is required")
    torch.manual_seed(model_seed)
    model = GlobalAffordanceScorer()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    for _ in range(training.epochs):
        optimizer.zero_grad()
        losses = [
            nn.functional.cross_entropy(
                model(example.candidate_features).unsqueeze(0),
                torch.tensor([example.selected_index], dtype=torch.long),
            )
            for example in examples
        ]
        torch.stack(losses).mean().backward()
        optimizer.step()
    model.eval()
    return model, TrainingReport(
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        optimizer_steps=training.epochs,
        forward_passes=training.epochs * len(examples),
        training_examples=len(examples),
    )


def train_global_frequency_optimum_model(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    training: TrainingSpec,
    model_seed: int,
) -> tuple[GlobalAffordanceScorer, TrainingReport]:
    """Train clean Milestone 5-style global optimum-frequency imitation."""

    if features.ndim != 2 or features.shape[1] != PROBE_FEATURE_COUNT:
        raise ValueError("unexpected global optimum feature tensor")
    if targets.ndim != 1 or len(targets) != len(features) or not len(targets):
        raise ValueError("global optimum targets do not match features")
    torch.manual_seed(model_seed)
    model = GlobalAffordanceScorer()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    for _ in range(training.epochs):
        optimizer.zero_grad()
        loss = torch.mean((model(features) - targets) ** 2)
        loss.backward()
        optimizer.step()
    model.eval()
    return model, TrainingReport(
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        optimizer_steps=training.epochs,
        forward_passes=training.epochs,
        training_examples=len(features),
    )


def visible_action_weights(
    model: StateConditionedScorer | None,
    state: ObservableState,
    affordances: AffordanceTable,
    *,
    temperature: float,
) -> tuple[dict[str, float], int]:
    """Return weights only for visible aliases; unknown aliases get uniform fallback mass."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    aliases, features, unknown = candidate_tensor(state, affordances)
    if model is None:
        return {alias: 1.0 for alias in aliases}, unknown
    with torch.no_grad():
        scores = model(features)
    if unknown:
        known_indices = [
            index for index, alias in enumerate(aliases) if affordances.for_alias(alias) is not None
        ]
        neutral_score = (
            scores[known_indices].mean()
            if known_indices
            else torch.tensor(0.0, dtype=scores.dtype, device=scores.device)
        )
        scores = scores.clone()
        for index, alias in enumerate(aliases):
            if affordances.for_alias(alias) is None:
                scores[index] = neutral_score
    probabilities = torch.softmax(scores / temperature, dim=0).tolist()
    return dict(zip(aliases, (float(value) for value in probabilities))), unknown


def global_visible_action_weights(
    model: GlobalAffordanceScorer,
    state: ObservableState,
    affordances: AffordanceTable,
    *,
    temperature: float,
) -> tuple[dict[str, float], int]:
    """Score current visible aliases without passing current state to the model."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    zero_affordance = (0.0,) * PROBE_FEATURE_COUNT
    rows: list[tuple[float, ...]] = []
    known: list[bool] = []
    for alias in state.available_aliases:
        row = affordances.for_alias(alias)
        known.append(row is not None)
        rows.append(row or zero_affordance)
    if not rows:
        raise ValueError("cannot score a state without available actions")
    with torch.no_grad():
        scores = model(torch.tensor(rows, dtype=torch.float32))
    if not all(known):
        known_indices = [index for index, present in enumerate(known) if present]
        neutral_score = (
            scores[known_indices].mean()
            if known_indices
            else torch.tensor(0.0, dtype=scores.dtype, device=scores.device)
        )
        scores = scores.clone()
        for index, present in enumerate(known):
            if not present:
                scores[index] = neutral_score
    probabilities = torch.softmax(scores / temperature, dim=0).tolist()
    return (
        dict(
            zip(
                state.available_aliases,
                (float(value) for value in probabilities),
            )
        ),
        known.count(False),
    )
