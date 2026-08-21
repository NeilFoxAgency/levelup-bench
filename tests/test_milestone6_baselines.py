from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from levelup.core.trajectory import ActionRecord, Trajectory, TrajectoryStep
from levelup.envs.adaptive_track import adaptive_track_bundle
from levelup.experiments.milestone6_baselines import (
    CandidateVerdict,
    CleanOptimumTrainingSample,
    IndependentCandidateEvaluator,
    build_clean_optimum_training_sample,
    classify_exact_optimum,
    discover_affordances,
    evaluate_generated_search,
    generate_candidates_with_observable_policy,
    optimum_only_training_samples,
    trajectory_content_sha256,
    validate_and_sanitize_reference,
)
from levelup.experiments.runner.config import (
    ExposedTrajectory,
    TaskIdentity,
    TrajectoryIdentity,
)


def _forbidden_alias(bundle: object) -> frozenset[str]:
    environment = bundle.environment  # type: ignore[attr-defined]
    alias = environment.task_spec.constraints[0].verifier_config["action"]
    assert isinstance(alias, str)
    return frozenset({alias})


def _exposure(trajectory: Trajectory, stage: str) -> ExposedTrajectory:
    return ExposedTrajectory(
        task_id=trajectory.task_id,
        stage_label=stage,
        trajectory_id=trajectory.trajectory_id,
    )


def _task_identity(bundle: Any) -> TaskIdentity:
    environment = bundle.environment
    return TaskIdentity(
        family_id=environment.family,
        task_id=environment.task_spec.task_id,
        task_index=environment.task_index,
        generator_seed=environment.generator_seed,
        environment_reset_seed=0,
        trajectory_catalog=tuple(
            TrajectoryIdentity(
                stage_label=stage.label,
                trajectory_id=stage.trajectory_id,
                source="synthetic-reference",
                provenance={
                    "content_sha256": trajectory_content_sha256(
                        bundle.trajectories[stage.trajectory_id]
                    )
                },
            )
            for stage in bundle.ladder.stages
        ),
    )


def test_probe_discovers_aliases_from_observations_under_exact_action_cap() -> None:
    bundle = adaptive_track_bundle("battery", 0, 1000)
    evidence = discover_affordances(
        bundle.environment,
        task_id=bundle.environment.task_spec.task_id,
        forbidden_aliases=_forbidden_alias(bundle),
        seed=17,
        action_cap=32,
        target_samples_per_alias=4,
        actions_per_attempt=8,
    )
    assert evidence.accounting.actions == 32
    assert evidence.accounting.resets >= 4
    assert set(evidence.accounting.discovered_aliases) == set(
        evidence.affordances.features
    )
    assert all(
        transition.action_alias in transition.before.available_aliases
        for transition in evidence.transitions
    )


def test_probe_fails_boundedly_when_no_visible_action_can_be_taken() -> None:
    @dataclass
    class StalledOutcome:
        observation: dict[str, Any]
        completed: bool = False

    class StalledEnvironment:
        def fresh(self) -> StalledEnvironment:
            return StalledEnvironment()

        def reset(self, seed: int | None = None) -> StalledOutcome:
            return StalledOutcome(
                {
                    "progress": 0,
                    "target": 1,
                    "elapsed_ticks": 0,
                    "resource_fraction": 0.0,
                    "pressure_fraction": 0.0,
                    "available_actions": [],
                }
            )

        def step(self, action: ActionRecord) -> StalledOutcome:
            raise AssertionError("stalled environment must never be stepped")

    try:
        discover_affordances(
            StalledEnvironment(),
            task_id="stalled-task",
            forbidden_aliases=frozenset(),
            seed=1,
            action_cap=4,
            target_samples_per_alias=1,
        )
    except RuntimeError as exc:
        assert "cannot spend" in str(exc)
    else:
        raise AssertionError("stalled probe did not terminate")


def test_reference_is_independently_validated_then_sanitized() -> None:
    bundle = adaptive_track_bundle("plain", 1, 900)
    optimum = bundle.trajectory_for("optimum")
    validated = validate_and_sanitize_reference(
        bundle.environment,
        optimum,
        task_identity=_task_identity(bundle),
        exposure=_exposure(optimum, "optimum"),
        forbidden_aliases=_forbidden_alias(bundle),
    )
    assert validated.performance_value == bundle.ladder.stage("optimum").performance_value
    assert validated.evaluator_calls == 1
    assert validated.evaluator_replay_actions == len(optimum.steps)
    assert len(validated.trace.transitions) == len(optimum.steps)
    assert not hasattr(validated.trace, "task_id")
    assert not hasattr(validated.trace.transitions[0], "state_hash")


def test_invalid_reference_cannot_produce_observable_training_trace() -> None:
    bundle = adaptive_track_bundle("plain", 1, 900)
    optimum = bundle.trajectory_for("optimum")
    invalid = Trajectory(
        trajectory_id=optimum.trajectory_id,
        task_id=optimum.task_id,
        source="reference",
        steps=(TrajectoryStep(index=0, action=ActionRecord(name="not-an-alias")),),
    )
    try:
        validate_and_sanitize_reference(
            bundle.environment,
            invalid,
            task_identity=_task_identity(bundle),
            exposure=_exposure(invalid, "optimum"),
            forbidden_aliases=_forbidden_alias(bundle),
        )
    except ValueError as exc:
        assert "content does not match" in str(exc)
    else:
        raise AssertionError("invalid reference was accepted")


def test_uniform_search_obeys_hard_total_adaptation_cap() -> None:
    bundle = adaptive_track_bundle("plain", 1, 900)
    forbidden = _forbidden_alias(bundle)
    probe = discover_affordances(
        bundle.environment,
        task_id=bundle.environment.task_spec.task_id,
        forbidden_aliases=forbidden,
        seed=23,
        action_cap=16,
        target_samples_per_alias=2,
    )
    evaluator = IndependentCandidateEvaluator(bundle.environment)
    generated = generate_candidates_with_observable_policy(
        bundle.environment,
        task_id=bundle.environment.task_spec.task_id,
        forbidden_aliases=forbidden,
        affordances=probe.affordances,
        model=None,
        seed=29,
        temperature=0.9,
        max_episodes=100,
        max_actions_per_episode=64,
        total_adaptation_action_cap=32,
        prior_adaptation_actions=probe.accounting.actions,
        condition_id="paid-probe-uniform-test",
    )
    outcome = evaluate_generated_search(generated, evaluator)
    assert probe.accounting.actions + outcome.accounting.actions <= 32
    assert outcome.accounting.forward_passes == 0
    assert outcome.accounting.evaluator_calls >= 0


def test_exact_optimum_is_reporting_only_and_does_not_stop_search() -> None:
    @dataclass
    class Outcome:
        observation: dict[str, Any]
        completed: bool

    class OneStepEnvironment:
        def fresh(self) -> OneStepEnvironment:
            return OneStepEnvironment()

        def reset(self, seed: int | None = None) -> Outcome:
            assert seed == 0
            return Outcome(
                {
                    "progress": 0,
                    "target": 1,
                    "elapsed_ticks": 0,
                    "resource_fraction": 0.0,
                    "pressure_fraction": 0.0,
                    "available_actions": [{"alias": "finish"}],
                },
                False,
            )

        def step(self, action: ActionRecord) -> Outcome:
            assert action.name == "finish"
            return Outcome(
                {
                    "progress": 1,
                    "target": 1,
                    "elapsed_ticks": 1,
                    "resource_fraction": 0.0,
                    "pressure_fraction": 0.0,
                    "available_actions": [],
                },
                True,
            )

    class AlwaysExactEvaluator:
        def evaluate(self, trajectory: Trajectory) -> CandidateVerdict:
            return CandidateVerdict(True, 1.0, len(trajectory.steps))

    environment = OneStepEnvironment()
    probe = discover_affordances(
        environment,
        task_id="one-step-task",
        forbidden_aliases=frozenset(),
        seed=31,
        action_cap=1,
        target_samples_per_alias=1,
    )
    generated = generate_candidates_with_observable_policy(
        environment,
        task_id="one-step-task",
        forbidden_aliases=frozenset(),
        affordances=probe.affordances,
        model=None,
        seed=37,
        temperature=0.9,
        max_episodes=5,
        max_actions_per_episode=1,
        total_adaptation_action_cap=6,
        prior_adaptation_actions=probe.accounting.actions,
        condition_id="reporting-only-optimum-test",
    )
    outcome = evaluate_generated_search(generated, AlwaysExactEvaluator())
    assert outcome.accounting.episodes == 5
    assert outcome.accounting.actions == 5
    exact = classify_exact_optimum(outcome, optimum_performance=1.0)
    assert exact.first_episode == 1
    assert exact.success


def test_optimum_only_training_boundary_rejects_frontier_reference() -> None:
    bundle = adaptive_track_bundle("plain", 1, 900)
    forbidden = _forbidden_alias(bundle)
    optimum = bundle.trajectory_for("optimum")
    frontier = bundle.trajectory_for("frontier")
    sample = build_clean_optimum_training_sample(
        bundle.environment,
        optimum,
        task_identity=_task_identity(bundle),
        exposure=_exposure(optimum, "optimum"),
        forbidden_aliases=forbidden,
        probe_seed=41,
        probe_action_cap=8,
        target_samples_per_alias=1,
    )
    assert optimum_only_training_samples((sample,))

    try:
        build_clean_optimum_training_sample(
            bundle.environment,
            frontier,
            task_identity=_task_identity(bundle),
            exposure=_exposure(frontier, "frontier"),
            forbidden_aliases=forbidden,
            probe_seed=41,
            probe_action_cap=8,
            target_samples_per_alias=1,
        )
    except ValueError as exc:
        assert "non-optimum" in str(exc)
    else:
        raise AssertionError("frontier reference entered optimum-only training")


def test_clean_optimum_sample_cannot_be_constructed_with_arbitrary_probe() -> None:
    bundle = adaptive_track_bundle("plain", 1, 900)
    optimum = bundle.trajectory_for("optimum")
    canonical = build_clean_optimum_training_sample(
        bundle.environment,
        optimum,
        task_identity=_task_identity(bundle),
        exposure=_exposure(optimum, "optimum"),
        forbidden_aliases=_forbidden_alias(bundle),
        probe_seed=43,
        probe_action_cap=8,
        target_samples_per_alias=1,
    )
    try:
        CleanOptimumTrainingSample(canonical.reference, canonical.probe)
    except ValueError as exc:
        assert "canonical paid-probe builder" in str(exc)
    else:
        raise AssertionError("arbitrary clean optimum sample construction was accepted")


def test_canonical_catalog_rejects_relabeling_frontier_as_optimum() -> None:
    bundle = adaptive_track_bundle("plain", 1, 900)
    frontier = bundle.trajectory_for("frontier")
    try:
        validate_and_sanitize_reference(
            bundle.environment,
            frontier,
            task_identity=_task_identity(bundle),
            exposure=_exposure(frontier, "optimum"),
            forbidden_aliases=_forbidden_alias(bundle),
        )
    except ValueError as exc:
        assert "canonical task catalog" in str(exc)
    else:
        raise AssertionError("frontier reference was relabeled as optimum")


def test_canonical_catalog_rejects_different_actions_reusing_optimum_id() -> None:
    bundle = adaptive_track_bundle("plain", 1, 900)
    optimum = bundle.trajectory_for("optimum")
    frontier = bundle.trajectory_for("frontier")
    forged = Trajectory(
        trajectory_id=optimum.trajectory_id,
        task_id=optimum.task_id,
        source=optimum.source,
        steps=frontier.steps,
    )
    try:
        validate_and_sanitize_reference(
            bundle.environment,
            forged,
            task_identity=_task_identity(bundle),
            exposure=_exposure(optimum, "optimum"),
            forbidden_aliases=_forbidden_alias(bundle),
        )
    except ValueError as exc:
        assert "content does not match" in str(exc)
    else:
        raise AssertionError("forged optimum trajectory content was accepted")
