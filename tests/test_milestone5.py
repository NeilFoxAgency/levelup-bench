import pytest

torch = pytest.importorskip("torch")

from levelup.envs.adaptive_track import (
    DEVELOPMENT_FAMILIES,
    collect_adaptive_bundles,
    make_adaptive_track,
)
from levelup.envs.challenge_track import (
    FINAL_CHALLENGE_FAMILY,
    held_out_combo_tasks,
    make_combo_track,
)
from levelup.experiments.milestone5 import FINAL_FAMILY, run_experiment
from levelup.learning.interaction import (
    PROBE_FEATURE_COUNT,
    InteractionScorer,
    probe_action_effects,
)


def test_final_family_is_not_a_development_family() -> None:
    assert FINAL_FAMILY == FINAL_CHALLENGE_FAMILY == "combo"
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


def test_combo_observation_is_opaque_and_effect_is_state_dependent() -> None:
    environment = make_combo_track(0, 2026)
    initial = environment.reset()
    for action in initial.observation["available_actions"]:
        assert set(action) == {"alias"}

    hidden_burst = next(
        action for action in environment.actions if action.pressure_clear and not action.forbidden
    )
    assert hidden_burst.alias not in environment.available_aliases()

    builder = next(
        action
        for action in environment.actions
        if action.pressure_gain and not action.forbidden
    )
    environment.step(type("Record", (), {"name": builder.alias, "arguments": {}})())
    assert hidden_burst.alias in environment.available_aliases()


def test_combo_tasks_have_strict_hidden_frontier_gap() -> None:
    tasks = held_out_combo_tasks(2, 2026)
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
