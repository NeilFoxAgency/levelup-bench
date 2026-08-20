import pytest

torch = pytest.importorskip("torch")

from levelup.envs.adaptive_track import (
    DEVELOPMENT_FAMILIES,
    FINAL_FAMILY,
    collect_adaptive_bundles,
    held_out_adaptive_tasks,
    make_adaptive_track,
)
from levelup.experiments.milestone5 import run_experiment
from levelup.learning.interaction import (
    PROBE_FEATURE_COUNT,
    InteractionScorer,
    probe_action_effects,
)


def test_final_family_is_not_a_development_family() -> None:
    assert FINAL_FAMILY == "overdrive"
    assert FINAL_FAMILY not in DEVELOPMENT_FAMILIES


def test_agent_observation_contains_aliases_but_no_action_descriptors() -> None:
    environment = make_adaptive_track("battery", 0, 1000)
    outcome = environment.reset()

    assert environment.task_spec.environment.configuration["action_descriptors_exposed"] is False
    actions = outcome.observation["available_actions"]
    assert actions
    for action in actions:
        assert set(action) == {"alias"}
        assert action["alias"].startswith("a")


def test_probe_representation_is_deterministic_and_neural_width_matches() -> None:
    environment = collect_adaptive_bundles("heat", 1, 1200)[0].environment
    first = probe_action_effects(environment, seed=12345, probes_per_action=3)
    second = probe_action_effects(environment, seed=12345, probes_per_action=3)

    assert first == second
    assert first.interactions > 0
    assert set(first.features) == set(environment.valid_action_aliases)
    assert all(len(features) == PROBE_FEATURE_COUNT for features in first.features.values())
    assert InteractionScorer().network[0].in_features == PROBE_FEATURE_COUNT


def test_overdrive_tasks_have_strict_hidden_frontier_gap() -> None:
    tasks = held_out_adaptive_tasks(FINAL_FAMILY, 2, 2026)
    assert len(tasks) == 2
    assert all(environment.family == FINAL_FAMILY for environment, _ in tasks)
    assert all(optimum > 0 for _, optimum in tasks)


def test_small_milestone5_run_is_reproducible_and_keeps_final_family_out_of_selection() -> None:
    kwargs = dict(
        development_tasks_per_family=1,
        final_task_count=1,
        replicates=1,
        max_episodes=5,
        probes_per_action=2,
        cv_validation_tasks=1,
        cv_replicates=1,
        cv_max_episodes=5,
        cv_model_epochs=4,
        final_model_epochs=4,
    )
    first = run_experiment(**kwargs)
    second = run_experiment(**kwargs)

    assert first == second
    assert first["final_family"] == FINAL_FAMILY
    assert first["method_selection"]["final_family_consulted"] is False
    assert set(first["development_families"]) == set(DEVELOPMENT_FAMILIES)
    assert set(first["conditions"]) == {
        "uniform",
        "frontier_to_optimum_delta",
        "shuffled_transition_direction",
        "pooled_frontier_optimum",
        "imitate_optimum",
        "robust_selected_mix",
    }
