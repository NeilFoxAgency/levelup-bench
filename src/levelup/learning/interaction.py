"""Infer action affordances from interaction traces and score them with a tiny neural model."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from levelup.core.trajectory import ActionRecord, Trajectory
from levelup.envs.adaptive_track import AdaptiveTrack, AdaptiveTrackBundle

TRANSITION_FEATURE_NAMES = (
    "pre_progress_fraction",
    "pre_remaining_fraction",
    "pre_resource_fraction",
    "pre_pressure_fraction",
    "delta_progress_scaled",
    "delta_ticks_scaled",
    "delta_resource_fraction",
    "delta_pressure_fraction",
    "post_resource_fraction",
    "post_pressure_fraction",
    "completed",
    "available_action_fraction",
)
PROBE_FEATURE_COUNT = len(TRANSITION_FEATURE_NAMES) * 4 + 1
TargetKind = Literal["delta", "shuffled", "pooled", "optimum"]


class InteractionScorer(nn.Module):
    """Score opaque actions using only empirically inferred effects."""

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
class ProbeResult:
    features: dict[str, tuple[float, ...]]
    interactions: int


def structured_forbidden_aliases(environment: AdaptiveTrack) -> set[str]:
    aliases: set[str] = set()
    for constraint in environment.task_spec.constraints:
        if constraint.verifier_id == "never_use_action":
            alias = constraint.verifier_config.get("action")
            if isinstance(alias, str):
                aliases.add(alias)
    return aliases


def available_aliases(observation: Any, forbidden: set[str]) -> list[str]:
    """Read only opaque aliases from the agent-facing action list."""

    if not isinstance(observation, dict):
        raise RuntimeError("AdaptiveTrack observation must be a dictionary")
    raw = observation.get("available_actions")
    if not isinstance(raw, list):
        raise RuntimeError("AdaptiveTrack observation is missing available_actions")
    aliases: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeError("available action must be an object")
        if set(item) != {"alias"}:
            raise RuntimeError("Milestone 5 forbids structured action descriptors")
        alias = item.get("alias")
        if isinstance(alias, str) and alias not in forbidden:
            aliases.append(alias)
    return aliases


def _state_fields(observation: Any) -> tuple[float, float, float, float, float]:
    if not isinstance(observation, dict):
        raise RuntimeError("AdaptiveTrack observation must be a dictionary")
    progress = observation.get("progress")
    target = observation.get("target")
    elapsed = observation.get("elapsed_ticks")
    resource = observation.get("resource_fraction")
    pressure = observation.get("pressure_fraction")
    if not isinstance(progress, int) or not isinstance(target, int) or target <= 0:
        raise RuntimeError("invalid progress observation")
    if not isinstance(elapsed, int):
        raise RuntimeError("invalid elapsed-tick observation")
    if not isinstance(resource, (int, float)) or not isinstance(pressure, (int, float)):
        raise RuntimeError("invalid resource or pressure observation")
    return float(progress), float(target), float(elapsed), float(resource), float(pressure)


def _transition_vector(
    before: Any,
    after: Any,
    *,
    completed: bool,
    available_count: int,
) -> tuple[float, ...]:
    pre_progress, target, pre_elapsed, pre_resource, pre_pressure = _state_fields(before)
    post_progress, _, post_elapsed, post_resource, post_pressure = _state_fields(after)
    return (
        pre_progress / target,
        (target - pre_progress) / target,
        pre_resource,
        pre_pressure,
        (post_progress - pre_progress) / 3.0,
        (post_elapsed - pre_elapsed) / 13.0,
        post_resource - pre_resource,
        post_pressure - pre_pressure,
        post_resource,
        post_pressure,
        float(completed),
        available_count / 5.0,
    )


def _summarize(rows: list[tuple[float, ...]], expected: int) -> tuple[float, ...]:
    if not rows:
        raise RuntimeError("probe produced no observations for an action")
    tensor = torch.tensor(rows, dtype=torch.float32)
    summary = torch.cat(
        (
            tensor.mean(dim=0),
            tensor.std(dim=0, unbiased=False),
            tensor.min(dim=0).values,
            tensor.max(dim=0).values,
            torch.tensor([len(rows) / expected], dtype=torch.float32),
        )
    )
    if len(summary) != PROBE_FEATURE_COUNT:
        raise RuntimeError("unexpected probe feature width")
    return tuple(float(value) for value in summary.tolist())


def probe_action_effects(
    environment: AdaptiveTrack,
    *,
    seed: int,
    probes_per_action: int,
) -> ProbeResult:
    """Infer action effects by trying opaque actions in varied reachable states."""

    if probes_per_action < 1:
        raise ValueError("probes_per_action must be positive")
    rng = random.Random(seed)
    forbidden = structured_forbidden_aliases(environment)
    target_aliases = tuple(
        alias for alias in environment.valid_action_aliases if alias not in forbidden
    )
    observations: dict[str, list[tuple[float, ...]]] = {
        alias: [] for alias in target_aliases
    }
    interactions = 0

    for target_alias in target_aliases:
        attempts = 0
        while len(observations[target_alias]) < probes_per_action and attempts < 400:
            attempts += 1
            probe_env = environment.fresh()
            outcome = probe_env.reset()
            prefix_length = rng.randrange(0, max(1, environment.target // 2 + 1))

            for _ in range(prefix_length):
                if outcome.completed:
                    break
                aliases = available_aliases(outcome.observation, forbidden)
                if not aliases:
                    break
                outcome = probe_env.step(ActionRecord(name=rng.choice(aliases)))
                interactions += 1

            if outcome.completed:
                continue
            aliases = available_aliases(outcome.observation, forbidden)
            if target_alias not in aliases:
                continue

            before = outcome.observation
            after = probe_env.step(ActionRecord(name=target_alias))
            interactions += 1
            observations[target_alias].append(
                _transition_vector(
                    before,
                    after.observation,
                    completed=after.completed,
                    available_count=len(aliases),
                )
            )

        if len(observations[target_alias]) != probes_per_action:
            raise RuntimeError(
                f"probe could not collect {probes_per_action} transitions for {target_alias!r}"
            )

    return ProbeResult(
        features={
            alias: _summarize(rows, probes_per_action)
            for alias, rows in observations.items()
        },
        interactions=interactions,
    )


def _frequency(trajectory: Trajectory) -> dict[str, float]:
    counts = Counter(step.action.name for step in trajectory.steps)
    total = len(trajectory.steps)
    return {alias: count / total for alias, count in counts.items()}


def training_examples(
    bundles: tuple[AdaptiveTrackBundle, ...],
    probe_cache: dict[str, ProbeResult],
    target_kind: TargetKind,
    *,
    label_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(label_seed)
    rows: list[tuple[float, ...]] = []
    targets: list[float] = []

    for bundle in bundles:
        frontier_frequency = _frequency(bundle.trajectory_for("frontier"))
        optimum_frequency = _frequency(bundle.trajectory_for("optimum"))
        reverse = target_kind == "shuffled" and rng.random() >= 0.5
        probe = probe_cache[bundle.environment.task_spec.task_id]
        for alias in bundle.environment.valid_action_aliases:
            before = frontier_frequency.get(alias, 0.0)
            after = optimum_frequency.get(alias, 0.0)
            delta = after - before
            if target_kind == "delta":
                target = delta
            elif target_kind == "shuffled":
                target = -delta if reverse else delta
            elif target_kind == "pooled":
                target = (before + after) / 2.0
            elif target_kind == "optimum":
                target = after
            else:
                raise ValueError(f"unsupported target kind: {target_kind!r}")
            rows.append(probe.features[alias])
            targets.append(target)

    return torch.tensor(rows, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


def train_interaction_model(
    bundles: tuple[AdaptiveTrackBundle, ...],
    probe_cache: dict[str, ProbeResult],
    target_kind: TargetKind,
    *,
    epochs: int,
    model_seed: int,
) -> InteractionScorer:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(model_seed)
    model = InteractionScorer()
    features, targets = training_examples(
        bundles,
        probe_cache,
        target_kind,
        label_seed=model_seed + 111,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.mean((model(features) - targets) ** 2)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _normalized_scores(model: InteractionScorer, features: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        scores = model(features)
    deviation = scores.std(unbiased=False)
    if len(scores) > 1 and float(deviation) > 1e-6:
        return (scores - scores.mean()) / deviation
    return torch.zeros_like(scores)


def action_weights(
    environment: AdaptiveTrack,
    probe: ProbeResult,
    models: dict[str, InteractionScorer] | None,
    mix: dict[str, float] | None,
    *,
    temperature: float,
) -> dict[str, float]:
    aliases = environment.valid_action_aliases
    if models is None or mix is None:
        return {alias: 1.0 for alias in aliases}

    features = torch.tensor(
        [probe.features[alias] for alias in aliases],
        dtype=torch.float32,
    )
    combined = torch.zeros(len(aliases), dtype=torch.float32)
    for target_kind, weight in mix.items():
        if weight <= 0:
            continue
        combined += weight * _normalized_scores(models[target_kind], features)
    probabilities = torch.softmax(combined / temperature, dim=0)
    return {
        alias: float(probability)
        for alias, probability in zip(aliases, probabilities.tolist())
    }
