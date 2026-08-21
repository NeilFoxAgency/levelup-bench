from __future__ import annotations

import pytest
import torch

from levelup.learning.state_conditioned import (
    PROBE_FEATURE_COUNT,
    STATE_CONDITIONED_FEATURE_COUNT,
    AffordanceTable,
    GlobalAffordanceScorer,
    ObservableTrace,
    ObservedTransition,
    StateConditionedScorer,
    TrainingSpec,
    build_affordance_table,
    candidate_tensor,
    global_frequency_optimum_examples,
    global_listwise_optimum_examples,
    global_visible_action_weights,
    optimum_imitation_examples,
    parse_observation,
    train_global_listwise_optimum_model,
    train_state_conditioned_optimum_model,
    transition_features,
    visible_action_weights,
)


def _observation(
    *,
    aliases: tuple[str, ...] = ("opaque-a", "opaque-b"),
    progress: int = 2,
    elapsed: int = 7,
) -> dict[str, object]:
    return {
        "progress": progress,
        "target": 10,
        "elapsed_ticks": elapsed,
        "resource_fraction": 0.5,
        "pressure_fraction": 0.25,
        "available_actions": [{"alias": alias} for alias in aliases],
    }


def _transition(alias: str = "opaque-a") -> ObservedTransition:
    return ObservedTransition(
        before=parse_observation(_observation()),
        action_alias=alias,
        after=parse_observation(_observation(progress=3, elapsed=12)),
        completed=False,
    )


def test_observation_parser_rejects_identity_and_structured_action_leakage() -> None:
    with pytest.raises(RuntimeError, match="unexpected=\\['family_id'\\]"):
        parse_observation({**_observation(), "family_id": "plain"})

    structured = _observation()
    structured["available_actions"] = [{"alias": "opaque-a", "progress": 3}]
    with pytest.raises(RuntimeError, match="opaque aliases only"):
        parse_observation(structured)


def test_alias_renaming_preserves_numeric_state_and_transition_features() -> None:
    original = _transition("opaque-a")
    renamed = ObservedTransition(
        before=parse_observation(_observation(aliases=("renamed-x", "renamed-y"))),
        action_alias="renamed-x",
        after=parse_observation(
            _observation(aliases=("renamed-x", "renamed-y"), progress=3, elapsed=12)
        ),
        completed=False,
    )
    assert original.before.features() == renamed.before.features()
    assert transition_features(original) == transition_features(renamed)


def test_affordances_are_derived_only_from_observed_transitions() -> None:
    table = build_affordance_table((_transition(),), target_samples_per_alias=2)
    assert set(table.features) == {"opaque-a"}
    assert len(table.features["opaque-a"]) == PROBE_FEATURE_COUNT
    assert table.sample_counts == {"opaque-a": 1}


def test_candidate_tensor_scores_only_visible_actions_and_marks_unknowns() -> None:
    state = parse_observation(_observation(aliases=("opaque-a", "newly-visible")))
    table = build_affordance_table((_transition(),), target_samples_per_alias=1)
    aliases, features, unknown = candidate_tensor(state, table)
    assert aliases == ("opaque-a", "newly-visible")
    assert features.shape == (2, STATE_CONDITIONED_FEATURE_COUNT)
    assert unknown == 1
    assert torch.count_nonzero(features[1, 5:]) == 0


def test_optimum_examples_accept_only_sanitized_traces_and_have_no_alias_tensor() -> None:
    transition = _transition()
    table = build_affordance_table((transition,), target_samples_per_alias=1)
    examples = optimum_imitation_examples(
        ((ObservableTrace(transitions=(transition,)), table),),
    )
    assert len(examples) == 1
    assert examples[0].selected_index == 0
    assert examples[0].candidate_features.dtype == torch.float32
    assert examples[0].candidate_features.shape == (2, STATE_CONDITIONED_FEATURE_COUNT)


def test_uniform_weights_use_only_current_visible_aliases() -> None:
    state = parse_observation(_observation(aliases=("visible-a", "visible-b")))
    table = AffordanceTable(
        features={"not-visible": (0.0,) * PROBE_FEATURE_COUNT},
        sample_counts={"not-visible": 1},
    )
    weights, unknown = visible_action_weights(None, state, table, temperature=0.9)
    assert weights == {"visible-a": 1.0, "visible-b": 1.0}
    assert unknown == 2


def test_unknown_affordances_receive_explicit_neutral_mass() -> None:
    state = parse_observation(
        _observation(aliases=("known", "unknown-a", "unknown-b"))
    )
    table = AffordanceTable(
        features={"known": (1.0,) * PROBE_FEATURE_COUNT},
        sample_counts={"known": 1},
    )
    model = StateConditionedScorer().eval()
    weights, unknown = visible_action_weights(model, state, table, temperature=0.9)
    assert unknown == 2
    assert weights["known"] == pytest.approx(1 / 3)
    assert weights["unknown-a"] == pytest.approx(1 / 3)
    assert weights["unknown-b"] == pytest.approx(1 / 3)


def test_global_and_state_listwise_controls_match_objective_and_training_budget() -> None:
    transition = _transition()
    table = build_affordance_table((transition,), target_samples_per_alias=1)
    samples = ((ObservableTrace((transition,)), table),)
    global_examples = global_listwise_optimum_examples(samples)
    state_examples = optimum_imitation_examples(samples)
    training = TrainingSpec(epochs=2, learning_rate=0.003)

    global_model, global_report = train_global_listwise_optimum_model(
        global_examples,
        training=training,
        model_seed=43,
    )
    state_model, state_report = train_state_conditioned_optimum_model(
        state_examples,
        training=training,
        model_seed=43,
    )
    assert len(global_examples) == len(state_examples)
    assert global_report.optimizer_steps == state_report.optimizer_steps == 2
    assert global_report.forward_passes == state_report.forward_passes
    relative_gap = abs(
        global_report.trainable_parameters - state_report.trainable_parameters
    ) / state_report.trainable_parameters
    assert relative_gap < 0.1
    assert isinstance(global_model, GlobalAffordanceScorer)
    assert isinstance(state_model, StateConditionedScorer)


def test_clean_global_frequency_examples_use_optimum_actions_only() -> None:
    transition = _transition()
    table = build_affordance_table((transition,), target_samples_per_alias=1)
    features, targets = global_frequency_optimum_examples(
        ((ObservableTrace((transition,)), table),)
    )
    assert features.shape[1] == PROBE_FEATURE_COUNT
    assert float(targets.sum()) == pytest.approx(1.0)


def test_global_unknown_affordance_fallback_is_neutral() -> None:
    state = parse_observation(
        _observation(aliases=("known", "unknown-a", "unknown-b"))
    )
    table = AffordanceTable(
        features={"known": (1.0,) * PROBE_FEATURE_COUNT},
        sample_counts={"known": 1},
    )
    weights, unknown = global_visible_action_weights(
        GlobalAffordanceScorer().eval(),
        state,
        table,
        temperature=0.9,
    )
    assert unknown == 2
    assert weights == pytest.approx(
        {"known": 1 / 3, "unknown-a": 1 / 3, "unknown-b": 1 / 3}
    )
