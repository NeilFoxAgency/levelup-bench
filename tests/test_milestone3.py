import pytest
from pydantic import ValidationError

from levelup.core.experiment import ExposureManifest, ImprovementLadder, ImprovementStage
from levelup.envs.macrotrack import MacroTrack, STAGE_LABELS, macro_track_bundle, optimum_value
from levelup.evaluation import evaluate_trajectory
from levelup.experiments.milestone3 import (
    HELD_OUT_DISTANCES,
    TRAIN_DISTANCES,
    build_conditions,
    discovery_run,
    fit_transition_delta_prior,
    run_experiment,
    validate_training_ladders,
)


def test_synthetic_ladder_is_strictly_improving_and_replayable() -> None:
    bundle = macro_track_bundle(10)
    assert tuple(stage.label for stage in bundle.ladder.stages) == STAGE_LABELS
    values = [stage.performance_value for stage in bundle.ladder.stages]
    assert values == sorted(values, reverse=True)
    for stage in bundle.ladder.stages:
        result = evaluate_trajectory(MacroTrack(10), bundle.trajectory_for(stage.label))
        assert result.performance_eligible_for(MacroTrack(10).task_spec)
        assert result.performance_value == stage.performance_value


def test_synthetic_ladder_does_not_claim_human_provenance() -> None:
    bundle = macro_track_bundle(9)
    for stage in bundle.ladder.stages:
        assert stage.provenance["human_observed"] is False


def test_non_improving_ladder_is_rejected() -> None:
    with pytest.raises(ValidationError, match="improve strictly"):
        ImprovementLadder(
            task_id="task",
            direction="minimize",
            stages=(
                ImprovementStage(
                    stage_id="a", ordinal=0, label="a", trajectory_id="a", performance_value=5
                ),
                ImprovementStage(
                    stage_id="b", ordinal=1, label="b", trajectory_id="b", performance_value=5
                ),
            ),
        )


def test_exposure_manifest_rejects_train_test_overlap() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        ExposureManifest(
            condition_id="bad",
            train_task_ids=("same",),
            held_out_task_ids=("same",),
        )


def test_all_training_ladders_validate_before_learning() -> None:
    bundles = tuple(macro_track_bundle(distance) for distance in TRAIN_DISTANCES)
    validate_training_ladders(bundles)


def test_held_out_optima_are_not_exposed_to_any_condition() -> None:
    _, held_out_ids, conditions = build_conditions()
    assert held_out_ids == tuple(MacroTrack(d).task_spec.task_id for d in HELD_OUT_DISTANCES)
    for _, manifest in conditions.values():
        assert set(manifest.train_task_ids).isdisjoint(manifest.held_out_task_ids)
        assert all(
            task_id not in trajectory_id
            for task_id in held_out_ids
            for trajectory_id in manifest.exposed_trajectory_ids
        )


def test_transition_learner_identifies_the_action_that_closes_training_gap() -> None:
    bundles = tuple(macro_track_bundle(distance) for distance in TRAIN_DISTANCES)
    prior = fit_transition_delta_prior(bundles)
    assert prior.weights["leap"] > prior.weights["dash"]
    assert prior.weights["leap"] > prior.weights["run"]
    assert prior.weights["leap"] > prior.weights["walk"]


def test_discovery_curve_is_deterministic_for_fixed_seed() -> None:
    bundles = tuple(macro_track_bundle(distance) for distance in TRAIN_DISTANCES)
    prior = fit_transition_delta_prior(bundles)
    a = discovery_run(13, "delta", prior, seed=1234, max_episodes=100, budgets=(1, 10, 100))
    b = discovery_run(13, "delta", prior, seed=1234, max_episodes=100, budgets=(1, 10, 100))
    assert a == b
    assert a.optimum_value == optimum_value(13)


def test_transition_signal_beats_frontier_imitation_in_toy_sanity_run() -> None:
    report = run_experiment(replicates=12, max_episodes=200, budgets=(1, 10, 100, 200))
    conditions = report["conditions"]
    delta = conditions["frontier_to_optimum_delta"]["median_total_episodes_across_held_out_tasks"]
    frontier = conditions["imitate_frontier"]["median_total_episodes_across_held_out_tasks"]
    pooled = conditions["pooled_frontier_optimum"]["median_total_episodes_across_held_out_tasks"]
    assert delta < frontier
    assert delta < pooled
