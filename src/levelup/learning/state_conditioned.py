"""Observation-only representations for Milestone 6 development baselines.

This module is deliberately separated from environment construction and evaluation. Feature
builders accept sanitized observations and observed transitions, never benchmark environments,
task specifications, hidden action descriptors, state hashes, or optimum thresholds.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

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
TRANSITION_SUMMARY_BLOCK_COUNT = 4
TRANSITION_SUMMARY_FEATURE_COUNT = 12
STATE_AVAILABILITY_INPUT_WIDTH = STATE_CONDITIONED_FEATURE_COUNT
STATE_AVAILABILITY_RETAINED_INDICES = (0, 1, 2, 3, 11)
STATE_AVAILABILITY_ZEROED_INDICES = (4, 5, 6, 7, 8, 9, 10)
RESOURCE_PRESSURE_RETAINED_INDICES = (0, 1, 2, 3, 6, 7, 8, 9, 11)
RESOURCE_PRESSURE_ZEROED_INDICES = (4, 5, 10)
PROGRESS_ELAPSED_COMPLETION_RETAINED_INDICES = (0, 1, 2, 3, 4, 5, 10, 11)
PROGRESS_ELAPSED_COMPLETION_ZEROED_INDICES = (6, 7, 8, 9)
HISTORY_FEATURE_COUNT = TRANSITION_FEATURE_COUNT
HISTORY_LENGTH = 4
HISTORY_HIDDEN_WIDTH = 8
HISTORY_HEAD_INPUT_WIDTH = STATE_CONDITIONED_FEATURE_COUNT + HISTORY_HIDDEN_WIDTH
HISTORY_MODEL_PARAMETER_COUNT = 3889


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


@dataclass(frozen=True, slots=True)
class OutcomeGroupExampleViews:
    """Four masks derived from one exact ordered tuple of T examples."""

    transition: tuple[DecisionExample, ...]
    state_availability: tuple[DecisionExample, ...]
    resource_pressure: tuple[DecisionExample, ...]
    progress_elapsed_completion: tuple[DecisionExample, ...]


@dataclass(frozen=True, slots=True)
class HistoryDecisionExample:
    """One listwise decision with a causal, learner-visible transition window.

    ``history_features`` contains only the preceding transitions for this decision,
    oldest first.  It is intentionally a variable-length rank-two tensor: decisions
    near the beginning of a trajectory have fewer than four preceding transitions.
    The recurrent model always starts from an all-zero hidden state for this tensor.
    """

    candidate_features: torch.Tensor
    history_features: torch.Tensor
    selected_index: int

    def __post_init__(self) -> None:
        if self.candidate_features.ndim != 2:
            raise ValueError("candidate features must be a rank-two tensor")
        if self.candidate_features.shape[1] != STATE_CONDITIONED_FEATURE_COUNT:
            raise ValueError("unexpected state-conditioned feature width")
        if self.history_features.ndim != 2:
            raise ValueError("history features must be a rank-two tensor")
        if self.history_features.shape[1] != HISTORY_FEATURE_COUNT:
            raise ValueError("unexpected history feature width")
        if self.history_features.shape[0] > HISTORY_LENGTH:
            raise ValueError("history window exceeds the frozen four-transition limit")
        if not 0 <= self.selected_index < self.candidate_features.shape[0]:
            raise ValueError("selected index is outside the candidate set")


# A descriptive alias makes the causal control's role explicit to callers while
# retaining one concrete example type for H0/H4/H4-shuffled.
CausalHistoryDecisionExample = HistoryDecisionExample


@dataclass(frozen=True, slots=True)
class HistoryPermutationIdentity:
    """Identity tuple used by the frozen, process-independent history shuffle."""

    fold_id: str
    replicate: int
    task_id: str
    phase: str
    trace_or_episode_id: str
    decision_index: int

    def as_list(self) -> list[str | int]:
        return [
            self.fold_id,
            self.replicate,
            self.task_id,
            self.phase,
            self.trace_or_episode_id,
            self.decision_index,
        ]

    def as_key(self) -> tuple[str | int, ...]:
        return tuple(self.as_list())


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


class HistoryConditionedScorer(nn.Module):
    """Architecture-matched H0/H4 scorer (exactly 3,889 trainable parameters).

    The GRU is intentionally evaluated from its implicit all-zero initial state for
    every ``forward`` call.  Callers therefore cannot accidentally leak hidden state
    from one decision, trajectory, or candidate episode into another decision.
    """

    def __init__(self) -> None:
        super().__init__()
        self.history_encoder = nn.GRU(
            input_size=HISTORY_FEATURE_COUNT,
            hidden_size=HISTORY_HIDDEN_WIDTH,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(HISTORY_HEAD_INPUT_WIDTH, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        if sum(parameter.numel() for parameter in self.parameters()) != HISTORY_MODEL_PARAMETER_COUNT:
            raise RuntimeError("history scorer architecture parameter count drifted")

    def encode_history(self, history_features: torch.Tensor) -> torch.Tensor:
        """Return one hidden vector, starting from zero, and perform no padding."""

        if history_features.ndim == 2:
            if history_features.shape[1] != HISTORY_FEATURE_COUNT:
                raise ValueError("unexpected history feature width")
            if history_features.shape[0] == 0:
                return torch.zeros(
                    (HISTORY_HIDDEN_WIDTH,),
                    dtype=history_features.dtype,
                    device=history_features.device,
                )
            _, hidden = self.history_encoder(history_features.unsqueeze(0))
            return hidden[-1, 0]
        if history_features.ndim == 3:
            if history_features.shape[2] != HISTORY_FEATURE_COUNT:
                raise ValueError("unexpected history feature width")
            if history_features.shape[1] == 0:
                return torch.zeros(
                    (history_features.shape[0], HISTORY_HIDDEN_WIDTH),
                    dtype=history_features.dtype,
                    device=history_features.device,
                )
            _, hidden = self.history_encoder(history_features)
            return hidden[-1]
        raise ValueError("history features must be rank two or rank three")

    def forward(
        self,
        candidate_features: torch.Tensor,
        history_features: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_features.ndim != 2 or candidate_features.shape[1] != STATE_CONDITIONED_FEATURE_COUNT:
            raise ValueError("unexpected state-conditioned candidate feature width")
        encoded = self.encode_history(history_features)
        if encoded.ndim == 1:
            if candidate_features.shape[0] < 1:
                raise ValueError("candidate set cannot be empty")
            encoded = encoded.expand(candidate_features.shape[0], -1)
        elif encoded.ndim == 2:
            raise ValueError("batched history requires one candidate set per batch item")
        return self.head(torch.cat((candidate_features, encoded), dim=1)).squeeze(-1)


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
    recurrent_steps: int = 0


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


def apply_transition_summary_mask(
    features: torch.Tensor,
    retained_indices: Sequence[int],
    *,
    zeroed_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Apply one deterministic outcome-group mask to T candidate features.

    ``retained_indices`` and ``zeroed_indices`` are relative to each 12-channel
    transition-summary block.  The five current-state channels and final
    coverage channel are always retained.  The transform is out-of-place so a
    single tuple of T examples can be reused byte-for-byte by S, RP, and PEC.
    """

    if features.ndim != 2 or features.shape[1] != STATE_AVAILABILITY_INPUT_WIDTH:
        raise ValueError("unexpected state-conditioned feature tensor")
    retained = tuple(retained_indices)
    if (
        not retained
        or len(set(retained)) != len(retained)
        or any(
            not isinstance(index, int) or not 0 <= index < TRANSITION_SUMMARY_FEATURE_COUNT
            for index in retained
        )
    ):
        raise ValueError("transition-summary retained indices are invalid")
    zeroed = (
        tuple(zeroed_indices)
        if zeroed_indices is not None
        else tuple(
            index for index in range(TRANSITION_SUMMARY_FEATURE_COUNT) if index not in retained
        )
    )
    if (
        len(set(zeroed)) != len(zeroed)
        or any(
            not isinstance(index, int) or not 0 <= index < TRANSITION_SUMMARY_FEATURE_COUNT
            for index in zeroed
        )
        or set(retained) | set(zeroed) != set(range(TRANSITION_SUMMARY_FEATURE_COUNT))
        or set(retained) & set(zeroed)
    ):
        raise ValueError("transition-summary retained and zeroed indices must partition one block")

    masked = features.clone()
    affordance = masked[:, STATE_FEATURE_COUNT:]
    for block in range(TRANSITION_SUMMARY_BLOCK_COUNT):
        start = block * TRANSITION_SUMMARY_FEATURE_COUNT
        keep = tuple(start + index for index in retained)
        zero = tuple(start + index for index in zeroed)
        affordance[:, list(zero)] = 0.0
        affordance[:, list(keep)] = features[:, [STATE_FEATURE_COUNT + index for index in keep]]
    masked[:, STATE_FEATURE_COUNT:] = affordance
    return masked


def apply_state_availability_mask(features: torch.Tensor) -> torch.Tensor:
    """Apply the frozen S-condition transform to T candidate features."""

    return apply_transition_summary_mask(
        features,
        STATE_AVAILABILITY_RETAINED_INDICES,
        zeroed_indices=STATE_AVAILABILITY_ZEROED_INDICES,
    )


def apply_resource_pressure_mask(features: torch.Tensor) -> torch.Tensor:
    """Add resource/pressure outcomes to the frozen S representation."""

    return apply_transition_summary_mask(
        features,
        RESOURCE_PRESSURE_RETAINED_INDICES,
        zeroed_indices=RESOURCE_PRESSURE_ZEROED_INDICES,
    )


def apply_progress_elapsed_completion_mask(features: torch.Tensor) -> torch.Tensor:
    """Add progress, elapsed, and completion outcomes to frozen S."""

    return apply_transition_summary_mask(
        features,
        PROGRESS_ELAPSED_COMPLETION_RETAINED_INDICES,
        zeroed_indices=PROGRESS_ELAPSED_COMPLETION_ZEROED_INDICES,
    )


def mask_decision_examples(
    examples: Sequence[DecisionExample],
    feature_mask: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[DecisionExample, ...]:
    """Apply an out-of-place feature mask while preserving labels and order."""

    source = tuple(examples)
    return tuple(
        DecisionExample(feature_mask(example.candidate_features), example.selected_index)
        for example in source
    )


def outcome_group_optimum_example_views(
    samples: Sequence[tuple[ObservableTrace, AffordanceTable]],
) -> OutcomeGroupExampleViews:
    """Derive T, S, resource/pressure, and progress/payoff views from one source tuple."""

    transition = tuple(optimum_imitation_examples(samples))
    return OutcomeGroupExampleViews(
        transition=transition,
        state_availability=mask_decision_examples(transition, apply_state_availability_mask),
        resource_pressure=mask_decision_examples(transition, apply_resource_pressure_mask),
        progress_elapsed_completion=mask_decision_examples(
            transition, apply_progress_elapsed_completion_mask
        ),
    )


# Short aliases used by experiment code and notebooks; all route through the one
# frozen implementation above.
state_availability_mask = apply_state_availability_mask
mask_state_availability_features = apply_state_availability_mask


def state_availability_optimum_examples(
    samples: Sequence[tuple[ObservableTrace, AffordanceTable]],
) -> tuple[DecisionExample, ...]:
    """Build S examples from the exact T examples with only the frozen mask applied."""

    return mask_decision_examples(
        optimum_imitation_examples(samples), apply_state_availability_mask
    )


def resource_pressure_optimum_examples(
    samples: Sequence[tuple[ObservableTrace, AffordanceTable]],
) -> tuple[DecisionExample, ...]:
    """Build resource/pressure examples from exact T examples with only the mask applied."""

    return mask_decision_examples(optimum_imitation_examples(samples), apply_resource_pressure_mask)


def progress_elapsed_completion_optimum_examples(
    samples: Sequence[tuple[ObservableTrace, AffordanceTable]],
) -> tuple[DecisionExample, ...]:
    """Build progress/elapsed/completion examples from exact T examples with only the mask."""

    return mask_decision_examples(
        optimum_imitation_examples(samples), apply_progress_elapsed_completion_mask
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


def lexicographic_derangements(length: int) -> tuple[tuple[int, ...], ...]:
    """Return all index derangements in lexicographic order.

    Phase 3 only permits windows of length two through four.  Keeping this helper
    general and validating the input makes accidental use of a non-derangeable
    one-element window explicit while allowing empty/one-element windows to remain
    unchanged in the history builder.
    """

    if length < 0:
        raise ValueError("derangement length cannot be negative")
    if length > HISTORY_LENGTH:
        raise ValueError("derangement length exceeds the frozen history limit")
    if length < 2:
        return (tuple(range(length)),)
    return tuple(
        permutation
        for permutation in itertools.permutations(range(length))
        if all(index != value for index, value in enumerate(permutation))
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def history_shuffle_digest(
    *,
    history_shuffle_base: int,
    identity: HistoryPermutationIdentity,
) -> bytes:
    """Return the SHA-256 digest used to select one deterministic derangement."""

    return hashlib.sha256(
        _canonical_json_bytes([history_shuffle_base, *identity.as_list()])
    ).digest()


def deterministic_history_derangement(
    length: int,
    *,
    history_shuffle_base: int = 6_700_000,
    identity: HistoryPermutationIdentity | None = None,
) -> tuple[int, ...]:
    """Select a deterministic index derangement using the frozen SHA-256 rule."""

    choices = lexicographic_derangements(length)
    if length < 2:
        return choices[0]
    if identity is None:
        raise ValueError("history identity is required for a shuffled window")
    digest = history_shuffle_digest(
        history_shuffle_base=history_shuffle_base,
        identity=identity,
    )
    return choices[int.from_bytes(digest[:8], "big", signed=False) % len(choices)]


# Concise names for callers that treat the permutation as a generic seeded map.
derangements = lexicographic_derangements
deterministic_derangement = deterministic_history_derangement


def canonical_permutation_map_bytes(
    records: Sequence[Mapping[str, Any]],
) -> bytes:
    """Serialize a permutation map using the protocol's canonical byte contract.

    Records are sorted by the six identity fields, not by the permutation itself.
    The function returns bytes without a trailing newline so the resulting digest is
    stable across platforms and Python versions.
    """

    required = (
        "fold_id",
        "replicate",
        "task_id",
        "phase",
        "trace_or_episode_id",
        "decision_index",
        "input_transition_indices",
        "permuted_transition_indices",
    )
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[str | int, ...]] = set()
    for record in records:
        if set(record) != set(required):
            raise ValueError("permutation record fields drifted")
        row = {key: record[key] for key in required}
        for key in ("fold_id", "task_id", "phase", "trace_or_episode_id"):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError("permutation identity strings must be nonempty")
        if not isinstance(row["replicate"], int) or not isinstance(
            row["decision_index"], int
        ):
            raise ValueError("permutation identity integers are invalid")
        if row["replicate"] < 0 or row["decision_index"] < 0:
            raise ValueError("permutation identity integers cannot be negative")
        identity = tuple(row[field] for field in required[:6])
        if identity in identities:
            raise ValueError("duplicate permutation identity")
        identities.add(identity)
        for key in ("input_transition_indices", "permuted_transition_indices"):
            values = row[key]
            if not isinstance(values, (list, tuple)) or any(
                not isinstance(item, int) for item in values
            ):
                raise ValueError("permutation indices must be integer lists")
            row[key] = list(values)
        input_indices = row["input_transition_indices"]
        output_indices = row["permuted_transition_indices"]
        if len(input_indices) > HISTORY_LENGTH or len(output_indices) != len(input_indices):
            raise ValueError("permutation window length is invalid")
        if input_indices != list(
            range(row["decision_index"] - len(input_indices), row["decision_index"])
        ):
            raise ValueError("permutation input is not the causal preceding window")
        if sorted(output_indices) != input_indices:
            raise ValueError("permutation output does not preserve the input multiset")
        if len(input_indices) < 2:
            if output_indices != input_indices:
                raise ValueError("short permutation window must remain unchanged")
        elif any(left == right for left, right in zip(input_indices, output_indices)):
            raise ValueError("permutation output is not a derangement")
        normalized.append(row)
    normalized.sort(
        key=lambda row: tuple(row[field] for field in required[:6])
    )
    return _canonical_json_bytes(normalized)


def permutation_map_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_permutation_map_bytes(records)).hexdigest()


def _history_identity(
    *,
    sample_index: int,
    decision_index: int,
    fold_id: str,
    replicate: int,
    task_ids: Sequence[str],
    phase: str,
    trace_or_episode_ids: Sequence[str] | None,
) -> HistoryPermutationIdentity:
    trace_id = (
        trace_or_episode_ids[sample_index]
        if trace_or_episode_ids is not None
        else f"trace-{sample_index}"
    )
    return HistoryPermutationIdentity(
        fold_id=fold_id,
        replicate=replicate,
        task_id=task_ids[sample_index],
        phase=phase,
        trace_or_episode_id=trace_id,
        decision_index=decision_index,
    )


def _history_rows(
    trace: ObservableTrace,
    decision_index: int,
    *,
    mode: str,
    history_shuffle_base: int,
    identity: HistoryPermutationIdentity,
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    if mode not in {"causal", "null", "shuffled"}:
        raise ValueError("history mode must be causal, null, or shuffled")
    start = max(0, decision_index - HISTORY_LENGTH)
    input_indices = tuple(range(start, decision_index))
    if mode == "shuffled":
        permutation = deterministic_history_derangement(
            len(input_indices),
            history_shuffle_base=history_shuffle_base,
            identity=identity,
        )
        output_indices = tuple(input_indices[index] for index in permutation)
    else:
        output_indices = input_indices
    if mode == "null":
        rows = torch.zeros(
            (len(output_indices), HISTORY_FEATURE_COUNT), dtype=torch.float32
        )
    else:
        rows = torch.tensor(
            [transition_features(trace.transitions[index]) for index in output_indices],
            dtype=torch.float32,
        )
    if not len(output_indices):
        rows = torch.zeros((0, HISTORY_FEATURE_COUNT), dtype=torch.float32)
    return rows, input_indices, output_indices


def history_optimum_examples(
    samples: Sequence[tuple[ObservableTrace, AffordanceTable]],
    *,
    mode: str = "causal",
    history_shuffle_base: int = 6_700_000,
    fold_id: str = "",
    replicate: int = 0,
    task_id: str = "",
    task_ids: Sequence[str] | None = None,
    phase: str = "train",
    trace_or_episode_ids: Sequence[str] | None = None,
    permutation_records: list[dict[str, Any]] | None = None,
) -> tuple[HistoryDecisionExample, ...]:
    """Build H0, H4, or H4-shuffled examples from identical optimum traces.

    The same sanitized samples and listwise labels are used for every mode.  For
    shuffled mode, optional ``permutation_records`` receives the complete identity
    map used for each decision, allowing callers to bind its canonical SHA-256 to a
    training view or search unit.
    """

    if task_id and task_ids is not None:
        raise ValueError("provide task_id or task_ids, not both")
    resolved_task_ids = tuple(task_ids) if task_ids is not None else (task_id,) * len(samples)
    if len(resolved_task_ids) != len(samples):
        raise ValueError("task identity count does not match samples")
    if trace_or_episode_ids is not None and len(trace_or_episode_ids) != len(samples):
        raise ValueError("trace identity count does not match samples")
    if mode == "shuffled":
        if permutation_records is None:
            raise ValueError("shuffled history requires captured permutation records")
        if (
            not fold_id
            or replicate < 0
            or not phase
            or trace_or_episode_ids is None
            or any(not value for value in resolved_task_ids)
            or any(not value for value in trace_or_episode_ids)
        ):
            raise ValueError("shuffled history requires complete nonempty identities")
        if len(set(zip(resolved_task_ids, trace_or_episode_ids))) != len(samples):
            raise ValueError("shuffled history trace identities must be unique per task")
    examples: list[HistoryDecisionExample] = []
    for sample_index, (trace, affordances) in enumerate(samples):
        for decision_index, transition in enumerate(trace.transitions):
            aliases, features, _ = candidate_tensor(transition.before, affordances)
            try:
                selected = aliases.index(transition.action_alias)
            except ValueError as exc:
                raise ValueError("reference selected an unavailable action") from exc
            identity = _history_identity(
                sample_index=sample_index,
                decision_index=decision_index,
                fold_id=fold_id,
                replicate=replicate,
                task_ids=resolved_task_ids,
                phase=phase,
                trace_or_episode_ids=trace_or_episode_ids,
            )
            history, input_indices, output_indices = _history_rows(
                trace,
                decision_index,
                mode=mode,
                history_shuffle_base=history_shuffle_base,
                identity=identity,
            )
            if permutation_records is not None:
                permutation_records.append(
                    {
                        "fold_id": identity.fold_id,
                        "replicate": identity.replicate,
                        "task_id": identity.task_id,
                        "phase": identity.phase,
                        "trace_or_episode_id": identity.trace_or_episode_id,
                        "decision_index": identity.decision_index,
                        "input_transition_indices": list(input_indices),
                        "permuted_transition_indices": list(output_indices),
                    }
                )
            examples.append(HistoryDecisionExample(features, history, selected))
    if not examples:
        raise ValueError("at least one optimum decision is required")
    return tuple(examples)


def causal_history_optimum_examples(
    samples: Sequence[tuple[ObservableTrace, AffordanceTable]],
    **kwargs: Any,
) -> tuple[HistoryDecisionExample, ...]:
    return history_optimum_examples(samples, mode="causal", **kwargs)


def null_history_optimum_examples(
    samples: Sequence[tuple[ObservableTrace, AffordanceTable]],
    **kwargs: Any,
) -> tuple[HistoryDecisionExample, ...]:
    return history_optimum_examples(samples, mode="null", **kwargs)


def shuffled_history_optimum_examples(
    samples: Sequence[tuple[ObservableTrace, AffordanceTable]],
    **kwargs: Any,
) -> tuple[HistoryDecisionExample, ...]:
    return history_optimum_examples(samples, mode="shuffled", **kwargs)


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


def train_history_optimum_model(
    examples: Sequence[HistoryDecisionExample],
    *,
    training: TrainingSpec,
    model_seed: int,
) -> tuple[HistoryConditionedScorer, TrainingReport]:
    """Train H0/H4 with identical listwise updates and explicit recurrent accounting."""

    if not examples:
        raise ValueError("at least one training example is required")
    torch.manual_seed(model_seed)
    model = HistoryConditionedScorer()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    recurrent_steps_per_epoch = sum(
        int(example.history_features.shape[0]) for example in examples
    )
    for _ in range(training.epochs):
        optimizer.zero_grad()
        losses = [
            nn.functional.cross_entropy(
                model(example.candidate_features, example.history_features).unsqueeze(0),
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
        recurrent_steps=training.epochs * recurrent_steps_per_epoch,
    )


train_causal_history_optimum_model = train_history_optimum_model


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


def state_availability_visible_action_weights(
    model: StateConditionedScorer,
    state: ObservableState,
    affordances: AffordanceTable,
    *,
    temperature: float,
) -> tuple[dict[str, float], int]:
    """Score visible actions with S's transition-outcome channels masked exactly."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    aliases, features, unknown = candidate_tensor(state, affordances)
    with torch.no_grad():
        scores = model(apply_state_availability_mask(features))
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


def history_visible_action_weights(
    model: HistoryConditionedScorer,
    state: ObservableState,
    affordances: AffordanceTable,
    history_features: torch.Tensor,
    *,
    temperature: float,
) -> tuple[dict[str, float], int]:
    """Score visible actions using a fresh zero-state recurrent pass for this decision."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    aliases, features, unknown = candidate_tensor(state, affordances)
    with torch.no_grad():
        scores = model(features, history_features)
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


def history_action_logits(
    model: HistoryConditionedScorer,
    candidate_features: torch.Tensor,
    history_features: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Return logits and the exact number of recurrent steps used for one decision."""

    if history_features.ndim != 2:
        raise ValueError("history features must be rank two for one decision")
    with torch.no_grad():
        logits = model(candidate_features, history_features)
    return logits, int(history_features.shape[0])
