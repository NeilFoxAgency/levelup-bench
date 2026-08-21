from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from levelup.experiments.runner.training_data_artifacts import (
    AffordanceTableRecord,
    ObservableStateRecord,
    ObservableTraceRecord,
    ObservedTransitionRecord,
    SanitizedTrainingData,
    TrainingDataArtifactError,
    TrainingDataArtifactKey,
    TrainingDataSample,
    learner_samples,
    load_training_data_artifact,
    sanitize_clean_optimum_samples,
    write_training_data_artifact,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _key(condition: str = "B1") -> TrainingDataArtifactKey:
    return TrainingDataArtifactKey(
        screening_candidates_sha256=_hash("screening"),
        protocol_sha256=_hash("protocol"),
        task_manifest_sha256=_hash("tasks"),
        expected_unit_plan_sha256=_hash("units"),
        provenance_sha256=_hash("provenance"),
        reference_exposure_sha256=_hash("exposure"),
        representation_sha256=_hash("representation"),
        probe_policy_sha256=_hash("probe"),
        fold_id="fold-plain",
        heldout_family_id="combo",
        ordered_training_task_ids=("task-a",),
        ordered_heldout_task_ids=("task-b",),
        condition_id=condition,
        objective_id="optimum_frequency" if condition == "B1" else "listwise_optimum",
        replicate=0,
        data_order_seed=10,
        probe_seeds=(11,),
        environment_seeds=(12,),
    )


def _sample(task_id: str = "task-a") -> TrainingDataSample:
    state = ObservableStateRecord(
        progress_fraction=0.0,
        remaining_fraction=1.0,
        elapsed_per_target=0.0,
        resource_fraction=1.0,
        pressure_fraction=0.0,
        available_aliases=("wait",),
    )
    after = state.model_copy(update={"progress_fraction": 1.0, "remaining_fraction": 0.0})
    return TrainingDataSample(
        task_id=task_id,
        trace=ObservableTraceRecord(
            transitions=(
                ObservedTransitionRecord(
                    before=state, action_alias="wait", after=after, completed=True
                ),
            )
        ),
        affordances=AffordanceTableRecord(
            features={"wait": (1.0,) * 49}, sample_counts={"wait": 1}
        ),
    )


def _canonical_data(task_id: str = "task-a") -> SanitizedTrainingData:
    from levelup.experiments import milestone6_baselines as baselines
    from levelup.learning.state_conditioned import (
        AffordanceTable,
        ObservableState,
        ObservableTrace,
        ObservedTransition,
    )

    before = ObservableState(0.0, 1.0, 0.0, 1.0, 0.0, ("wait",))
    after = ObservableState(1.0, 0.0, 1.0, 1.0, 0.0, ())
    source = baselines.CleanOptimumTrainingSample(
        reference=baselines.ValidatedObservableTrace(
            task_id=task_id,
            stage_label="optimum",
            trajectory_id="audit-only",
            trace=ObservableTrace((ObservedTransition(before, "wait", after, True),)),
            performance_value=123.0,
            evaluator_calls=99,
            evaluator_replay_actions=1,
            observable_replay_actions=1,
            resets=2,
            evaluator_wall_seconds=1.0,
            observable_replay_wall_seconds=1.0,
        ),
        probe=baselines.ProbeEvidence(
            task_id=task_id,
            affordances=AffordanceTable(
                features={"wait": (1.0,) * 49}, sample_counts={"wait": 1}
            ),
            transitions=(),
            accounting=baselines.ProbeAccounting(1, 1, 1, ("wait",), 1.0),
        ),
        _construction_token=baselines._CANONICAL_CLEAN_SAMPLE_TOKEN,
    )
    return sanitize_clean_optimum_samples((source,))


def test_condition_views_share_content_evidence_without_sharing_view_identity(tmp_path: Path) -> None:
    data = _canonical_data()
    first = write_training_data_artifact(tmp_path, _key("B1"), data)
    second = write_training_data_artifact(tmp_path, _key("B2"), data)
    assert first.evidence_id == second.evidence_id
    assert first.artifact_id != second.artifact_id
    _, payload = load_training_data_artifact(tmp_path, expected_key=_key("B2"))
    assert payload.samples[0].task_id == "task-a"
    assert len(list((tmp_path / "training-data-evidence").iterdir())) == 1
    trace, affordances = learner_samples(payload)[0]
    assert trace.transitions[0].before.features() == (0.0, 1.0, 0.0, 1.0, 0.0)
    assert affordances.for_alias("wait") == (1.0,) * 49


def test_evidence_identity_binds_generation_metadata_not_only_payload(tmp_path: Path) -> None:
    first_key = _key("B1")
    changed_key = first_key.model_copy(update={"probe_policy_sha256": _hash("changed-probe")})
    data = _canonical_data()
    first = write_training_data_artifact(tmp_path, first_key, data)
    changed = write_training_data_artifact(tmp_path, changed_key, data)
    assert first.evidence_id != changed.evidence_id
    assert first.payload_sha256 == changed.payload_sha256


def test_tampered_evidence_is_rejected(tmp_path: Path) -> None:
    manifest = write_training_data_artifact(tmp_path, _key(), _canonical_data())
    payload = tmp_path / "training-data-evidence" / manifest.evidence_id / "samples.json"
    payload.write_text(payload.read_text().replace("wait", "tampered"), encoding="utf-8")
    with pytest.raises(TrainingDataArtifactError, match="integrity"):
        load_training_data_artifact(tmp_path, manifest.artifact_id)


def test_tampered_view_and_evidence_metadata_are_rejected(tmp_path: Path) -> None:
    manifest = write_training_data_artifact(tmp_path, _key(), _canonical_data())
    view_path = (
        tmp_path / "training-data-artifacts" / manifest.artifact_id / "manifest.json"
    )
    raw_view = json.loads(view_path.read_text(encoding="utf-8"))
    raw_view["payload_bytes"] += 1
    view_path.write_text(json.dumps(raw_view), encoding="utf-8")
    with pytest.raises(TrainingDataArtifactError, match="manifest"):
        load_training_data_artifact(tmp_path, manifest.artifact_id)

    other = tmp_path / "other"
    manifest = write_training_data_artifact(other, _key(), _canonical_data())
    evidence_path = (
        other / "training-data-evidence" / manifest.evidence_id / "manifest.json"
    )
    raw_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    raw_evidence["sample_task_ids"] = ["heldout-task"]
    evidence_path.write_text(json.dumps(raw_evidence), encoding="utf-8")
    with pytest.raises(TrainingDataArtifactError, match="evidence manifest"):
        load_training_data_artifact(other, manifest.artifact_id)


def test_unknown_learner_fields_are_rejected() -> None:
    with pytest.raises(ValueError):
        ObservableStateRecord.model_validate(
            {
                "progress_fraction": 0,
                "remaining_fraction": 1,
                "elapsed_per_target": 0,
                "resource_fraction": 1,
                "pressure_fraction": 0,
                "available_aliases": ["wait"],
                "state_hash": "hidden",
            }
        )


def test_symlinked_evidence_root_is_rejected(tmp_path: Path) -> None:
    manifest = write_training_data_artifact(tmp_path, _key(), _canonical_data())
    real = tmp_path / "training-data-evidence"
    moved = tmp_path / "evidence-real"
    real.rename(moved)
    os.symlink(moved, real)
    with pytest.raises(TrainingDataArtifactError):
        load_training_data_artifact(tmp_path, manifest.artifact_id)


def test_symlinked_manifest_leaf_is_rejected(tmp_path: Path) -> None:
    manifest = write_training_data_artifact(tmp_path, _key(), _canonical_data())
    path = tmp_path / "training-data-evidence" / manifest.evidence_id / "manifest.json"
    moved = tmp_path / "external-manifest.json"
    path.rename(moved)
    os.symlink(moved, path)
    with pytest.raises(TrainingDataArtifactError, match="symlink|unexpected files"):
        load_training_data_artifact(tmp_path, manifest.artifact_id)


def test_payload_must_exactly_match_ordered_training_fold(tmp_path: Path) -> None:
    with pytest.raises(TrainingDataArtifactError, match="ordered training fold"):
        write_training_data_artifact(tmp_path, _key(), _canonical_data("task-b"))


def test_affordance_width_and_finiteness_are_enforced() -> None:
    from levelup.learning.state_conditioned import PROBE_FEATURE_COUNT

    assert PROBE_FEATURE_COUNT == 49
    with pytest.raises(ValueError, match="feature width"):
        AffordanceTableRecord(features={"wait": (1.0,) * 48}, sample_counts={"wait": 1})
    with pytest.raises(ValueError, match="finite"):
        AffordanceTableRecord(
            features={"wait": (1.0,) * 48 + (float("nan"),)},
            sample_counts={"wait": 1},
        )


def test_sanitizer_supports_slotted_observable_types_and_drops_audit_fields() -> None:
    sample = _canonical_data().samples[0]
    rendered = sample.model_dump(mode="json")
    assert "performance_value" not in str(rendered)
    assert "evaluator" not in str(rendered)
    assert sample.trace.transitions[0].action_alias == "wait"


def test_sanitizer_rejects_forged_noncanonical_samples() -> None:
    with pytest.raises(ValueError, match="canonical clean samples"):
        sanitize_clean_optimum_samples((object(),))


def test_writer_rejects_hand_constructed_schema_valid_samples(tmp_path: Path) -> None:
    with pytest.raises(TrainingDataArtifactError, match="canonical sanitized batch"):
        write_training_data_artifact(tmp_path, _key(), (_sample(),))  # type: ignore[arg-type]
