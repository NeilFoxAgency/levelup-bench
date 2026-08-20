import pytest

torch = pytest.importorskip("torch")

from levelup.envs.mechanictrack import (
    HELD_OUT_FAMILY,
    TRAIN_FAMILIES,
    ActionMechanic,
    collect_bundles,
    held_out_tasks,
    make_mechanic_track,
)
from levelup.evaluation import evaluate_trajectory
from levelup.experiments.milestone4 import (
    FeatureScorer,
    build_conditions,
    build_training_bundles,
    run_experiment,
    validate_training_ladders,
)


def test_action_alias_does_not_change_neural_features() -> None:
    first = ActionMechanic("opaque_one", progress=3, tick_cost=9, pressure_gain=1)
    second = ActionMechanic("completely_different", progress=3, tick_cost=9, pressure_gain=1)

    assert first.feature_vector(12) == second.feature_vector(12)
    assert FeatureScorer().network[0].in_features == len(first.feature_vector(12))


def test_default_split_holds_out_entire_heat_family() -> None:
    assert HELD_OUT_FAMILY not in TRAIN_FAMILIES
    train = collect_bundles("plain", 2, 900)
    held_out = held_out_tasks(2, 1337)

    assert {bundle.environment.family for bundle in train} == {"plain"}
    assert {environment.family for environment, _ in held_out} == {"heat"}
    assert set(bundle.ladder.task_id for bundle in train).isdisjoint(
        environment.task_spec.task_id for environment, _ in held_out
    )


def test_generated_aliases_are_opaque_and_task_specific() -> None:
    first = make_mechanic_track("battery", 0, 1000)
    second = make_mechanic_track("battery", 1, 1000)
    first_aliases = {action.alias for action in first.actions}
    second_aliases = {action.alias for action in second.actions}

    assert first_aliases.isdisjoint(second_aliases)
    for alias in first_aliases | second_aliases:
        assert alias.startswith("a")
        assert all(word not in alias for word in ("walk", "run", "burst", "recharge", "cool"))


def test_frontier_and_optimum_ladders_replay_strictly() -> None:
    bundles = (
        *collect_bundles("plain", 1, 900),
        *collect_bundles("battery", 1, 1000),
        *collect_bundles("cooldown", 1, 1100),
    )
    validate_training_ladders(bundles)

    for bundle in bundles:
        frontier = bundle.ladder.stage("frontier")
        optimum = bundle.ladder.stage("optimum")
        assert optimum.performance_value < frontier.performance_value
        result = evaluate_trajectory(
            bundle.environment.fresh(),
            bundle.trajectory_for("optimum"),
        )
        assert result.performance_eligible_for(bundle.environment.task_spec)
        assert result.performance_value == optimum.performance_value


def test_direction_and_pooled_controls_have_same_exposed_trajectories() -> None:
    bundles = build_training_bundles()
    held_out = held_out_tasks(2, 1337)
    held_out_ids = tuple(environment.task_spec.task_id for environment, _ in held_out)
    conditions = build_conditions(bundles, held_out_ids)

    directed = conditions["frontier_to_optimum_delta"].manifest
    shuffled = conditions["shuffled_transition_direction"].manifest
    pooled = conditions["pooled_frontier_optimum"].manifest

    assert directed.exposed_trajectory_ids == shuffled.exposed_trajectory_ids
    assert directed.exposed_trajectory_ids == pooled.exposed_trajectory_ids
    assert set(directed.train_task_ids).isdisjoint(directed.held_out_task_ids)


def test_small_experiment_smoke_run_is_reproducible() -> None:
    kwargs = dict(
        replicates=2,
        max_episodes=20,
        budgets=(1, 10, 20),
        base_seed=123_000,
        held_out_count=2,
    )
    first = run_experiment(**kwargs)
    second = run_experiment(**kwargs)

    assert first == second
    assert first["held_out_family"] == "heat"
    assert set(first["conditions"]) == {
        "uniform",
        "frontier_to_optimum_delta",
        "shuffled_transition_direction",
        "pooled_frontier_optimum",
        "imitate_optimum",
    }
