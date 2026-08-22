from __future__ import annotations

import hashlib

import pytest
import torch

from levelup.learning.state_conditioned import (
    HISTORY_FEATURE_COUNT,
    HISTORY_MODEL_PARAMETER_COUNT,
    AffordanceTable,
    HistoryConditionedScorer,
    HistoryPermutationIdentity,
    ObservableTrace,
    ObservedTransition,
    TrainingSpec,
    apply_state_availability_mask,
    canonical_permutation_map_bytes,
    causal_history_optimum_examples,
    deterministic_history_derangement,
    history_optimum_examples,
    lexicographic_derangements,
    null_history_optimum_examples,
    parse_observation,
    permutation_map_sha256,
    shuffled_history_optimum_examples,
    state_availability_optimum_examples,
    train_history_optimum_model,
    transition_features,
)


def _state(progress: int, *, elapsed: int | None = None) -> dict[str, object]:
    return {
        "progress": progress,
        "target": 20,
        "elapsed_ticks": progress if elapsed is None else elapsed,
        "resource_fraction": 0.5,
        "pressure_fraction": 0.25,
        "available_actions": [{"alias": "a"}, {"alias": "b"}],
    }


def _samples(length: int = 6) -> tuple[tuple[ObservableTrace, AffordanceTable], ...]:
    transitions = tuple(
        ObservedTransition(
            before=parse_observation(_state(index)),
            action_alias="a",
            after=parse_observation(_state(index + 1)),
            completed=index == length - 1,
        )
        for index in range(length)
    )
    trace = ObservableTrace(transitions)
    table = AffordanceTable(
        features={"a": (0.0,) * 49, "b": (0.0,) * 49},
        sample_counts={"a": 1, "b": 1},
    )
    return ((trace, table),)


def test_state_availability_mask_keeps_state_and_support_but_zeroes_outcomes() -> None:
    features = torch.arange(54, dtype=torch.float32).reshape(1, 54)
    masked = apply_state_availability_mask(features)
    assert torch.equal(masked[:, :5], features[:, :5])
    assert masked[0, 53] == features[0, 53]
    for block in range(4):
        start = 5 + block * 12
        assert torch.equal(masked[0, start : start + 4], features[0, start : start + 4])
        assert masked[0, start + 11] == features[0, start + 11]
        assert torch.count_nonzero(masked[0, start + 4 : start + 11]) == 0

    with pytest.raises(ValueError, match="unexpected"):
        apply_state_availability_mask(torch.zeros(54))


def test_s_examples_are_exact_t_examples_with_only_mask_transform() -> None:
    samples = _samples(2)
    from levelup.learning.state_conditioned import optimum_imitation_examples

    t_examples = optimum_imitation_examples(samples)
    s_examples = state_availability_optimum_examples(samples)
    assert len(s_examples) == len(t_examples)
    for source, masked in zip(t_examples, s_examples):
        assert masked.selected_index == source.selected_index
        assert torch.equal(masked.candidate_features, apply_state_availability_mask(source.candidate_features))


def test_history_examples_share_labels_and_use_causal_windows() -> None:
    samples = _samples(6)
    causal = causal_history_optimum_examples(samples)
    null = null_history_optimum_examples(samples)
    assert len(causal) == len(null) == 6
    assert [item.selected_index for item in causal] == [item.selected_index for item in null]
    assert [item.history_features.shape[0] for item in causal] == [0, 1, 2, 3, 4, 4]
    assert torch.count_nonzero(null[-1].history_features) == 0
    expected = torch.tensor(
        [transition_features(samples[0][0].transitions[index]) for index in (1, 2, 3, 4)],
        dtype=torch.float32,
    )
    assert torch.equal(causal[-1].history_features, expected)
    for left, right in zip(causal, null):
        assert torch.equal(left.candidate_features, right.candidate_features)


def test_derangements_and_canonical_maps_are_process_independent() -> None:
    assert lexicographic_derangements(2) == ((1, 0),)
    assert all(all(index != value for index, value in enumerate(row)) for row in lexicographic_derangements(4))
    first = deterministic_history_derangement(
        4,
        identity=HistoryPermutationIdentity("fold", 0, "task", "train", "trace", 4),
    )
    second = deterministic_history_derangement(
        4,
        identity=HistoryPermutationIdentity("fold", 0, "task", "train", "trace", 4),
    )
    assert first == second
    records = [
        {
            "fold_id": "fold",
            "replicate": 0,
            "task_id": "task",
            "phase": "train",
            "trace_or_episode_id": "trace",
            "decision_index": 4,
            "input_transition_indices": [0, 1, 2, 3],
            "permuted_transition_indices": list(first),
        }
    ]
    encoded = canonical_permutation_map_bytes(records)
    assert b"\n" not in encoded
    assert permutation_map_sha256(records) == hashlib.sha256(encoded).hexdigest()


def test_shuffled_history_preserves_window_multiset_and_changes_only_order() -> None:
    records: list[dict[str, object]] = []
    causal = history_optimum_examples(_samples(6), mode="causal")
    shuffled = shuffled_history_optimum_examples(
        _samples(6),
        fold_id="fold",
        task_id="task",
        trace_or_episode_ids=("trace",),
        permutation_records=records,
    )
    assert len(records) == len(shuffled) == len(causal)
    for before, after in zip(causal, shuffled):
        assert before.history_features.shape == after.history_features.shape
        assert sorted(before.history_features.tolist()) == sorted(after.history_features.tolist())
    eligible = [record for record in records if len(record["input_transition_indices"]) >= 2]
    assert eligible
    assert all(
        all(left != right for left, right in zip(record["input_transition_indices"], record["permuted_transition_indices"]))
        for record in eligible
    )


def test_shuffled_history_requires_captured_per_sample_identity() -> None:
    samples = (_samples(2)[0], _samples(3)[0])
    with pytest.raises(ValueError, match="captured permutation records"):
        shuffled_history_optimum_examples(
            samples,
            fold_id="fold",
            task_ids=("task-a", "task-b"),
            trace_or_episode_ids=("trace", "trace"),
        )

    records: list[dict[str, object]] = []
    shuffled_history_optimum_examples(
        samples,
        fold_id="fold",
        task_ids=("task-a", "task-b"),
        trace_or_episode_ids=("trace", "trace"),
        permutation_records=records,
    )
    assert {record["task_id"] for record in records} == {"task-a", "task-b"}
    assert len(records) == 5


def test_canonical_permutation_map_rejects_malformed_or_duplicate_records() -> None:
    valid = {
        "fold_id": "fold",
        "replicate": 0,
        "task_id": "task",
        "phase": "train",
        "trace_or_episode_id": "trace",
        "decision_index": 2,
        "input_transition_indices": [0, 1],
        "permuted_transition_indices": [1, 0],
    }
    with pytest.raises(ValueError, match="duplicate permutation identity"):
        canonical_permutation_map_bytes([valid, valid])
    with pytest.raises(ValueError, match="not a derangement"):
        canonical_permutation_map_bytes(
            [{**valid, "permuted_transition_indices": [0, 1]}]
        )
    with pytest.raises(ValueError, match="causal preceding window"):
        canonical_permutation_map_bytes(
            [{**valid, "input_transition_indices": [1, 2], "permuted_transition_indices": [2, 1]}]
        )


def test_history_model_is_exactly_capacity_matched_and_resets_per_decision() -> None:
    model = HistoryConditionedScorer().eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == HISTORY_MODEL_PARAMETER_COUNT
    candidate = torch.randn(2, 54)
    history = torch.randn(4, HISTORY_FEATURE_COUNT)
    with torch.no_grad():
        first = model(candidate, history)
        second = model(candidate, history)
    assert torch.equal(first, second)
    assert model(candidate, torch.zeros((0, HISTORY_FEATURE_COUNT))).shape == (2,)


def test_history_training_reports_matched_recurrent_steps() -> None:
    examples = causal_history_optimum_examples(_samples(3))
    _, report = train_history_optimum_model(
        examples,
        training=TrainingSpec(epochs=2, learning_rate=0.003),
        model_seed=10,
    )
    assert report.trainable_parameters == HISTORY_MODEL_PARAMETER_COUNT
    assert report.optimizer_steps == 2
    assert report.forward_passes == 2 * len(examples)
    assert report.recurrent_steps == 2 * sum(item.history_features.shape[0] for item in examples)
