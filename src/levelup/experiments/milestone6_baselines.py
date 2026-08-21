"""Boundary-clean Milestone 6 probing, reference replay, and development search."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

from levelup.core.trajectory import ActionRecord, Trajectory, TrajectoryStep
from levelup.evaluation import evaluate_trajectory
from levelup.experiments.runner.config import ExposedTrajectory, TaskIdentity
from levelup.learning.state_conditioned import (
    AffordanceTable,
    GlobalAffordanceScorer,
    ObservableTrace,
    ObservedTransition,
    StateConditionedScorer,
    build_affordance_table,
    global_visible_action_weights,
    parse_observation,
    visible_action_weights,
)

_CANONICAL_CLEAN_SAMPLE_TOKEN = object()


class ObservableOutcome(Protocol):
    """Only the agent-facing part of an environment outcome."""

    observation: Any
    completed: bool


class ObservableEnvironment(Protocol):
    """Minimal interaction surface available to probe/search controllers."""

    def fresh(self) -> ObservableEnvironment: ...

    def reset(self, seed: int | None = None) -> ObservableOutcome: ...

    def step(self, action: ActionRecord) -> ObservableOutcome: ...


class CandidateEvaluator(Protocol):
    """Reporting-only evaluator interface; verdicts never feed proposal generation."""

    def evaluate(self, trajectory: Trajectory) -> CandidateVerdict: ...


@dataclass(frozen=True, slots=True)
class ProbeAccounting:
    attempts: int
    resets: int
    actions: int
    discovered_aliases: tuple[str, ...]
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class ProbeEvidence:
    task_id: str
    affordances: AffordanceTable
    transitions: tuple[ObservedTransition, ...]
    accounting: ProbeAccounting


@dataclass(frozen=True, slots=True)
class ValidatedObservableTrace:
    task_id: str
    stage_label: str
    trajectory_id: str
    trace: ObservableTrace
    performance_value: float
    evaluator_calls: int
    evaluator_replay_actions: int
    observable_replay_actions: int
    resets: int
    evaluator_wall_seconds: float
    observable_replay_wall_seconds: float


@dataclass(frozen=True, slots=True)
class CandidateVerdict:
    valid: bool
    performance_value: float | None
    replay_actions: int


@dataclass(frozen=True, slots=True)
class SearchAccounting:
    episodes: int
    resets: int
    actions: int
    forward_passes: int
    evaluator_calls: int
    evaluator_replay_actions: int
    unknown_affordance_decisions: int
    generation_wall_seconds: float
    evaluator_wall_seconds: float


@dataclass(frozen=True, slots=True)
class GenerationAccounting:
    episodes: int
    resets: int
    actions: int
    forward_passes: int
    unknown_affordance_decisions: int
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class GeneratedCandidate:
    episode: int
    adaptation_actions: int
    trajectory: Trajectory


@dataclass(frozen=True, slots=True)
class GeneratedSearch:
    candidates: tuple[GeneratedCandidate, ...]
    accounting: GenerationAccounting


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    first_valid_episode: int | None
    best_performance: float | None
    evaluated_candidates: tuple[EvaluatedCandidate, ...]
    accounting: SearchAccounting


@dataclass(frozen=True, slots=True)
class EvaluatedCandidate:
    episode: int
    adaptation_actions: int
    performance_value: float


@dataclass(frozen=True, slots=True)
class ExactOptimumReport:
    first_episode: int | None
    first_adaptation_actions: int | None
    success: bool


@dataclass(frozen=True, slots=True, init=False)
class CleanOptimumTrainingSample:
    """Canonical optimum trace paired only with independently paid probe evidence."""

    reference: ValidatedObservableTrace
    probe: ProbeEvidence
    _construction_token: object

    def __init__(
        self,
        reference: ValidatedObservableTrace,
        probe: ProbeEvidence,
        *,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _CANONICAL_CLEAN_SAMPLE_TOKEN:
            raise ValueError("clean optimum samples require the canonical paid-probe builder")
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "probe", probe)
        object.__setattr__(self, "_construction_token", _construction_token)


class IndependentCandidateEvaluator:
    """Replay validity/performance without access to an optimum threshold."""

    def __init__(self, environment: Any) -> None:
        self._environment = environment

    def evaluate(self, trajectory: Trajectory) -> CandidateVerdict:
        result = evaluate_trajectory(self._environment.fresh(), trajectory)
        valid = result.performance_eligible_for(self._environment.task_spec)
        performance = result.performance_value if valid else None
        return CandidateVerdict(
            valid=valid,
            performance_value=performance,
            replay_actions=len(trajectory.steps),
        )


def trajectory_content_sha256(trajectory: Trajectory) -> str:
    """Bind a declared trajectory identity to its complete canonical serialized content."""

    encoded = json.dumps(
        trajectory.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def classify_exact_optimum(
    outcome: SearchOutcome,
    *,
    optimum_performance: float,
) -> ExactOptimumReport:
    """Classify exact success after fixed-budget search has completely finished."""

    for candidate in outcome.evaluated_candidates:
        if candidate.performance_value == optimum_performance:
            return ExactOptimumReport(
                first_episode=candidate.episode,
                first_adaptation_actions=candidate.adaptation_actions,
                success=True,
            )
    return ExactOptimumReport(None, None, False)


def discover_affordances(
    environment: ObservableEnvironment,
    *,
    task_id: str,
    forbidden_aliases: frozenset[str],
    seed: int,
    action_cap: int,
    target_samples_per_alias: int,
    actions_per_attempt: int = 16,
) -> ProbeEvidence:
    """Probe visible aliases under a fixed total action cap without hidden enumeration."""

    if action_cap < 1:
        raise ValueError("action_cap must be positive")
    if actions_per_attempt < 1:
        raise ValueError("actions_per_attempt must be positive")
    rng = random.Random(seed)
    started = time.perf_counter()
    transitions: list[ObservedTransition] = []
    sample_counts: dict[str, int] = {}
    attempts = 0
    resets = 0
    actions = 0

    while actions < action_cap:
        attempts += 1
        actions_before_attempt = actions
        probe_environment = environment.fresh()
        outcome = probe_environment.reset(seed=0)
        resets += 1
        for _ in range(min(actions_per_attempt, action_cap - actions)):
            if outcome.completed:
                break
            before = parse_observation(
                outcome.observation,
                forbidden_aliases=forbidden_aliases,
            )
            if not before.available_aliases:
                break
            minimum = min(sample_counts.get(alias, 0) for alias in before.available_aliases)
            least_sampled = [
                alias
                for alias in before.available_aliases
                if sample_counts.get(alias, 0) == minimum
            ]
            alias = rng.choice(least_sampled)
            outcome = probe_environment.step(ActionRecord(name=alias))
            after = parse_observation(
                outcome.observation,
                forbidden_aliases=forbidden_aliases,
            )
            transitions.append(
                ObservedTransition(
                    before=before,
                    action_alias=alias,
                    after=after,
                    completed=outcome.completed,
                )
            )
            sample_counts[alias] = sample_counts.get(alias, 0) + 1
            actions += 1
        if actions == actions_before_attempt and attempts >= max(4, action_cap):
            raise RuntimeError("probe cannot spend its action cap from observable states")

    table = build_affordance_table(
        transitions,
        target_samples_per_alias=target_samples_per_alias,
    )
    return ProbeEvidence(
        task_id=task_id,
        affordances=table,
        transitions=tuple(transitions),
        accounting=ProbeAccounting(
            attempts=attempts,
            resets=resets,
            actions=actions,
            discovered_aliases=tuple(sorted(table.features)),
            wall_seconds=time.perf_counter() - started,
        ),
    )


def validate_and_sanitize_reference(
    environment: Any,
    trajectory: Trajectory,
    *,
    task_identity: TaskIdentity,
    exposure: ExposedTrajectory,
    forbidden_aliases: frozenset[str],
) -> ValidatedObservableTrace:
    """Validate with evaluator truth, then expose observation/action consequences only."""

    if task_identity.task_id != trajectory.task_id:
        raise ValueError("task identity does not match reference trajectory")
    if environment.task_spec.task_id != task_identity.task_id:
        raise ValueError("task identity does not match reference environment")
    catalog = {item.trajectory_id: item for item in task_identity.trajectory_catalog}
    catalog_entry = catalog.get(trajectory.trajectory_id)
    if catalog_entry is None:
        raise ValueError("reference trajectory is absent from the canonical task catalog")
    if exposure.task_id != trajectory.task_id:
        raise ValueError("exposure task does not match reference trajectory")
    if exposure.trajectory_id != trajectory.trajectory_id:
        raise ValueError("exposure identity does not match reference trajectory")
    if exposure.stage_label not in {"frontier", "optimum"}:
        raise ValueError("unsupported development reference stage")
    if catalog_entry.stage_label != exposure.stage_label:
        raise ValueError("exposure stage does not match the canonical task catalog")
    expected_content_sha256 = catalog_entry.provenance.get("content_sha256")
    if not isinstance(expected_content_sha256, str) or len(expected_content_sha256) != 64:
        raise ValueError("canonical task catalog is missing a trajectory content hash")
    if trajectory_content_sha256(trajectory) != expected_content_sha256:
        raise ValueError("reference content does not match the canonical task catalog")

    evaluator_started = time.perf_counter()
    result = evaluate_trajectory(environment.fresh(), trajectory)
    evaluator_wall = time.perf_counter() - evaluator_started
    if not result.performance_eligible_for(environment.task_spec):
        raise ValueError("reference trajectory is not an independently valid completion")
    if result.performance_value is None:
        raise RuntimeError("valid reference has no performance value")

    observable_replay_started = time.perf_counter()
    replay_environment = environment.fresh()
    replay_seed = trajectory.environment_seed
    if replay_seed is None:
        replay_seed = environment.task_spec.environment.seed
    outcome = replay_environment.reset(seed=replay_seed)
    transitions: list[ObservedTransition] = []
    for step in trajectory.steps:
        before = parse_observation(
            outcome.observation,
            forbidden_aliases=forbidden_aliases,
        )
        if step.action.name not in before.available_aliases:
            raise ValueError("reference selected an unavailable or forbidden action")
        outcome = replay_environment.step(step.action)
        after = parse_observation(
            outcome.observation,
            forbidden_aliases=forbidden_aliases,
        )
        transitions.append(
            ObservedTransition(
                before=before,
                action_alias=step.action.name,
                after=after,
                completed=outcome.completed,
            )
        )
    if not outcome.completed:
        raise RuntimeError("sanitized replay did not reproduce reference completion")
    if replay_environment.objective_value() != result.performance_value:
        raise RuntimeError("sanitized replay performance disagrees with evaluator replay")
    observable_replay_wall = time.perf_counter() - observable_replay_started
    return ValidatedObservableTrace(
        task_id=task_identity.task_id,
        stage_label=exposure.stage_label,
        trajectory_id=exposure.trajectory_id,
        trace=ObservableTrace(tuple(transitions)),
        performance_value=float(result.performance_value),
        evaluator_calls=1,
        evaluator_replay_actions=len(trajectory.steps),
        observable_replay_actions=len(trajectory.steps),
        resets=2,
        evaluator_wall_seconds=evaluator_wall,
        observable_replay_wall_seconds=observable_replay_wall,
    )


def build_clean_optimum_training_sample(
    environment: Any,
    trajectory: Trajectory,
    *,
    task_identity: TaskIdentity,
    exposure: ExposedTrajectory,
    forbidden_aliases: frozenset[str],
    probe_seed: int,
    probe_action_cap: int,
    target_samples_per_alias: int,
    probe_actions_per_attempt: int = 16,
) -> CleanOptimumTrainingSample:
    """Build the only supported B/C sample path from optimum plus paid probes."""

    reference = validate_and_sanitize_reference(
        environment,
        trajectory,
        task_identity=task_identity,
        exposure=exposure,
        forbidden_aliases=forbidden_aliases,
    )
    if reference.stage_label != "optimum":
        raise ValueError("optimum imitation cannot consume non-optimum reference data")
    probe = discover_affordances(
        environment,
        task_id=task_identity.task_id,
        forbidden_aliases=forbidden_aliases,
        seed=probe_seed,
        action_cap=probe_action_cap,
        target_samples_per_alias=target_samples_per_alias,
        actions_per_attempt=probe_actions_per_attempt,
    )
    return CleanOptimumTrainingSample(
        reference=reference,
        probe=probe,
        _construction_token=_CANONICAL_CLEAN_SAMPLE_TOKEN,
    )


def optimum_only_training_samples(
    samples: tuple[CleanOptimumTrainingSample, ...],
) -> tuple[tuple[ObservableTrace, AffordanceTable], ...]:
    """Enforce the optimum-only exposure boundary before learner input construction."""

    if not samples:
        raise ValueError("at least one validated optimum sample is required")
    if any(sample._construction_token is not _CANONICAL_CLEAN_SAMPLE_TOKEN for sample in samples):
        raise ValueError("optimum imitation requires canonical paid-probe samples")
    if any(sample.reference.stage_label != "optimum" for sample in samples):
        raise ValueError("optimum imitation cannot consume non-optimum reference data")
    if any(sample.reference.task_id != sample.probe.task_id for sample in samples):
        raise ValueError("reference and paid-probe task identities do not match")
    return tuple((sample.reference.trace, sample.probe.affordances) for sample in samples)


def generate_candidates_with_observable_policy(
    environment: ObservableEnvironment,
    *,
    task_id: str,
    forbidden_aliases: frozenset[str],
    affordances: AffordanceTable,
    model: StateConditionedScorer | GlobalAffordanceScorer | None,
    seed: int,
    temperature: float,
    max_episodes: int,
    max_actions_per_episode: int,
    total_adaptation_action_cap: int,
    prior_adaptation_actions: int,
    condition_id: str,
) -> GeneratedSearch:
    """Generate a fixed-budget candidate batch with no evaluator object or feedback."""

    if max_episodes < 1 or max_actions_per_episode < 1:
        raise ValueError("episode budgets must be positive")
    if not 0 <= prior_adaptation_actions <= total_adaptation_action_cap:
        raise ValueError("prior adaptation actions exceed the total cap")
    rng = random.Random(seed)
    started = time.perf_counter()
    episodes = resets = actions = forward_passes = 0
    unknown_decisions = 0
    generated: list[GeneratedCandidate] = []

    for episode in range(1, max_episodes + 1):
        if prior_adaptation_actions + actions >= total_adaptation_action_cap:
            break
        episodes = episode
        candidate_environment = environment.fresh()
        outcome = candidate_environment.reset(seed=0)
        resets += 1
        aliases_taken: list[str] = []
        for _ in range(max_actions_per_episode):
            if outcome.completed:
                break
            if prior_adaptation_actions + actions >= total_adaptation_action_cap:
                break
            state = parse_observation(
                outcome.observation,
                forbidden_aliases=forbidden_aliases,
            )
            if not state.available_aliases:
                break
            if isinstance(model, GlobalAffordanceScorer):
                weights, unknown = global_visible_action_weights(
                    model,
                    state,
                    affordances,
                    temperature=temperature,
                )
            else:
                weights, unknown = visible_action_weights(
                    model,
                    state,
                    affordances,
                    temperature=temperature,
                )
            if model is not None:
                forward_passes += 1
            unknown_decisions += unknown
            alias = rng.choices(
                tuple(weights),
                weights=tuple(weights.values()),
                k=1,
            )[0]
            aliases_taken.append(alias)
            outcome = candidate_environment.step(ActionRecord(name=alias))
            actions += 1

        if not outcome.completed:
            continue
        trajectory = Trajectory(
            trajectory_id=f"search:{condition_id}:{task_id}:s{seed}:e{episode}",
            task_id=task_id,
            source="agent",
            steps=tuple(
                TrajectoryStep(index=index, action=ActionRecord(name=alias))
                for index, alias in enumerate(aliases_taken)
            ),
        )
        generated.append(
            GeneratedCandidate(
                episode=episode,
                adaptation_actions=prior_adaptation_actions + actions,
                trajectory=trajectory,
            )
        )

    return GeneratedSearch(
        candidates=tuple(generated),
        accounting=GenerationAccounting(
            episodes=episodes,
            resets=resets,
            actions=actions,
            forward_passes=forward_passes,
            unknown_affordance_decisions=unknown_decisions,
            wall_seconds=time.perf_counter() - started,
        ),
    )


def evaluate_generated_search(
    generated: GeneratedSearch,
    evaluator: CandidateEvaluator,
) -> SearchOutcome:
    """Evaluate a completed candidate batch without changing its generation history."""

    evaluated: list[EvaluatedCandidate] = []
    started = time.perf_counter()
    first_valid: int | None = None
    best: float | None = None
    evaluator_calls = 0
    replay_actions = 0
    for candidate in generated.candidates:
        verdict = evaluator.evaluate(candidate.trajectory)
        evaluator_calls += 1
        replay_actions += verdict.replay_actions
        if not verdict.valid or verdict.performance_value is None:
            continue
        if first_valid is None:
            first_valid = candidate.episode
        if best is None or verdict.performance_value < best:
            best = verdict.performance_value
        evaluated.append(
            EvaluatedCandidate(
                episode=candidate.episode,
                adaptation_actions=candidate.adaptation_actions,
                performance_value=verdict.performance_value,
            )
        )
    accounting = generated.accounting
    return SearchOutcome(
        first_valid_episode=first_valid,
        best_performance=best,
        evaluated_candidates=tuple(evaluated),
        accounting=SearchAccounting(
            episodes=accounting.episodes,
            resets=accounting.resets,
            actions=accounting.actions,
            forward_passes=accounting.forward_passes,
            evaluator_calls=evaluator_calls,
            evaluator_replay_actions=replay_actions,
            unknown_affordance_decisions=accounting.unknown_affordance_decisions,
            generation_wall_seconds=accounting.wall_seconds,
            evaluator_wall_seconds=time.perf_counter() - started,
        ),
    )
