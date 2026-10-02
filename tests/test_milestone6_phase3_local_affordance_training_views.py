"""Same-data and metadata-stripping checks for local-affordance training views."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from levelup.experiments import milestone6_phase3_local_affordance_capabilities as capabilities
from levelup.experiments.milestone6_phase3_evidence import (
    _EVIDENCE_LOCK_TOKEN,
    Phase3EvidenceLock,
)
from levelup.experiments.milestone6_phase3_local_affordance_capabilities import (
    TrainingFoldProbeCapability,
)
from levelup.experiments.milestone6_phase3_local_affordance_models import (
    LocalAffordanceTrainingView,
)
from levelup.experiments.milestone6_phase3_local_affordance_plan import (
    LocalAffordanceView,
    build_local_affordance_plan,
)
from levelup.experiments.milestone6_phase3_local_affordance_raw_store import (
    TrainingFoldManifest,
)
from levelup.experiments.milestone6_phase3_local_affordance_training_views import (
    CONDITION_B2,
    CONDITION_IDS,
    CONDITION_L,
    CONDITION_P,
    CONDITION_S,
    LocalAffordanceTrainingViewError,
    build_local_affordance_training_view,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import (
    AffordanceTableRecord,
    ObservableStateRecord,
    ObservableTraceRecord,
    ObservedTransitionRecord,
    TrainingDataEvidenceKey,
    TrainingDataEvidenceManifest,
    TrainingDataEvidencePayloadBundle,
    TrainingDataPayload,
    TrainingDataSample,
)
from levelup.learning.state_conditioned import (
    AffordanceTable,
    ObservableState,
    ObservableTrace,
    ObservedTransition,
    TaskLocalAffordanceEvidence,
    apply_state_availability_mask,
    candidate_tensor,
    global_listwise_optimum_examples,
    local_affordance_candidate_tensor,
    pooled_candidate_tensor,
    state_availability_candidate_tensor,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "milestone6"
LOCK_PATH = CONFIG / "phase3_evidence_lock.json"


@pytest.fixture(scope="session")
def expected_authority():
    from levelup.experiments import milestone6_phase3_local_affordance_raw_authority as authority

    return authority.build_expected_raw_probe_authority(
        local_affordance_protocol_bytes=(CONFIG / "phase3_local_affordance_protocol.json").read_bytes(),
        development_protocol_bytes=(CONFIG / "development_protocol.json").read_bytes(),
        development_tasks_bytes=(CONFIG / "development_tasks.json").read_bytes(),
        phase3_evidence_lock_bytes=(CONFIG / "phase3_evidence_lock.json").read_bytes(),
    )


@pytest.fixture(scope="session")
def authority_snapshot(tmp_path_factory, expected_authority):
    from test_milestone6_phase3_local_affordance_raw_authority import _artifact

    from levelup.experiments import milestone6_phase3_local_affordance_raw_authority as authority
    from levelup.experiments import (
        milestone6_phase3_local_affordance_raw_publication as publication,
    )

    artifacts = tuple(
        authority.PersistedRawProbeArtifact.model_validate(_artifact(key)[0])
        for key in expected_authority.keys
    )
    root = tmp_path_factory.mktemp("training-views-raw-authority") / "store"
    publication.publish_raw_probe_store(root, expected=expected_authority, artifacts=artifacts)
    from levelup.experiments.milestone6_phase3_local_affordance_raw_store import (
        open_existing_raw_probe_store,
    )

    with open_existing_raw_probe_store(root) as reader:
        return authority.validate_complete_raw_probe_authority(
            reader, expected=expected_authority
        )


def _fold_task_ids(snapshot, fold_id: str, replicate: int) -> tuple[str, ...]:
    name = f"{fold_id}.r{replicate}.json"
    record = next(item for item in snapshot.training_fold_files if item.name == name)
    fold = TrainingFoldManifest.model_validate_json(record.snapshot.canonical_bytes)
    return tuple(reference.task_id for reference in fold.task_references)


def _locked_source_bundle(task_ids, evidence):
    lock_body = json.loads(LOCK_PATH.read_bytes())
    row = next(
        item
        for item in lock_body["evidence_artifacts"]
        if item["family_id"] == "plain" and item["replicate"] == 0
    )
    key = TrainingDataEvidenceKey.model_validate(row["evidence_key"])
    samples = []
    for task_id, item in zip(task_ids, evidence, strict=True):
        state_before = ObservableStateRecord(
            progress_fraction=0.3,
            remaining_fraction=0.7,
            elapsed_per_target=0.1,
            resource_fraction=0.5,
            pressure_fraction=0.2,
            available_aliases=("a",),
        )
        state_after = ObservableStateRecord(
            progress_fraction=0.31,
            remaining_fraction=0.69,
            elapsed_per_target=0.1,
            resource_fraction=0.5,
            pressure_fraction=0.2,
            available_aliases=("a",),
        )
        trace = ObservableTraceRecord(
            transitions=(
                ObservedTransitionRecord(
                    before=state_before,
                    action_alias="a",
                    after=state_after,
                    completed=True,
                ),
            )
        )
        affordances = AffordanceTableRecord(
            features=dict(item.affordances.features),
            sample_counts=dict(item.affordances.sample_counts),
        )
        samples.append(TrainingDataSample(task_id=task_id, trace=trace, affordances=affordances))
    payload = TrainingDataPayload(samples=tuple(samples))
    payload_bytes = canonical_json_bytes(payload.model_dump(mode="json"))
    manifest_body = {
        "schema_version": "runner.training-data-evidence.v1",
        "evidence_key_id": key.key_id,
        "key": key.model_dump(mode="json"),
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "payload_bytes": len(payload_bytes),
        "sample_task_ids": list(task_ids),
    }
    manifest = TrainingDataEvidenceManifest(
        evidence_id=hashlib.sha256(canonical_json_bytes(manifest_body)).hexdigest(),
        **manifest_body,
    )
    manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
    row.update(
        {
            "evidence_key_id": key.key_id,
            "evidence_key": key.model_dump(mode="json"),
            "evidence_id": manifest.evidence_id,
            "evidence_manifest_key_id": key.key_id,
            "evidence_manifest": manifest.model_dump(mode="json"),
            "payload_sha256": manifest.payload_sha256,
            "payload_bytes": manifest.payload_bytes,
            "ordered_training_task_ids": list(task_ids),
            "canonical_manifest_bytes_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
    )
    unsigned = {key: value for key, value in lock_body.items() if key != "evidence_lock_sha256"}
    lock_body["evidence_lock_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    lock_bytes = canonical_json_bytes(lock_body)
    source_lock = Phase3EvidenceLock(
        body=lock_body,
        canonical_bytes=lock_bytes,
        evidence_lock_sha256=lock_body["evidence_lock_sha256"],
        _construction_token=_EVIDENCE_LOCK_TOKEN,
    )
    bundle = TrainingDataEvidencePayloadBundle(
        manifest=manifest,
        payload=payload,
        manifest_bytes=manifest_bytes,
        payload_bytes=payload_bytes,
    )
    return source_lock, bundle


def _state(progress: float, *, aliases: tuple[str, ...] = ("a",)) -> ObservableState:
    return ObservableState(
        progress_fraction=progress,
        remaining_fraction=1.0 - progress,
        elapsed_per_target=0.1,
        resource_fraction=0.5,
        pressure_fraction=0.2,
        available_aliases=aliases,
    )


@dataclass(frozen=True)
class Fixture:
    fold: TrainingFoldManifest
    views: tuple[LocalAffordanceView, ...]
    probe_capability: TrainingFoldProbeCapability
    evidence: tuple[TaskLocalAffordanceEvidence, ...]
    evidence_lock: Phase3EvidenceLock
    source_bundle: TrainingDataEvidencePayloadBundle


@pytest.fixture
def fixture(authority_snapshot) -> Fixture:
    heldout = "plain"
    replicate = 0
    name = f"{heldout}.r{replicate}.json"
    fold_record = next(item for item in authority_snapshot.training_fold_files if item.name == name)
    fold = TrainingFoldManifest.model_validate_json(fold_record.snapshot.canonical_bytes)
    task_ids = _fold_task_ids(authority_snapshot, heldout, replicate)
    capability = capabilities.issue_training_fold_probe_capability(
        authority_snapshot, fold_id=heldout, replicate=replicate
    )
    evidence = capability.consume_for(task_ids)
    evidence_lock, source_bundle = _locked_source_bundle(task_ids, evidence)
    plan = build_local_affordance_plan()
    views = tuple(
        view
        for view in plan.views
        if view.fold_id == heldout and view.replicate == replicate
    )
    assert {view.condition_id for view in views} == set(CONDITION_IDS)
    return Fixture(fold, views, capability, evidence, evidence_lock, source_bundle)


def _view(fixture: Fixture, condition_id: str) -> LocalAffordanceView:
    return next(view for view in fixture.views if view.condition_id == condition_id)


def _tensor_bytes(tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.int32).numpy().tobytes()


def _core_samples(bundle: TrainingDataEvidencePayloadBundle):
    samples = []
    for sample in bundle.payload.samples:
        trace = ObservableTrace(
            tuple(
                ObservedTransition(
                    ObservableState(
                        transition.before.progress_fraction,
                        transition.before.remaining_fraction,
                        transition.before.elapsed_per_target,
                        transition.before.resource_fraction,
                        transition.before.pressure_fraction,
                        transition.before.available_aliases,
                    ),
                    transition.action_alias,
                    ObservableState(
                        transition.after.progress_fraction,
                        transition.after.remaining_fraction,
                        transition.after.elapsed_per_target,
                        transition.after.resource_fraction,
                        transition.after.pressure_fraction,
                        transition.after.available_aliases,
                    ),
                    transition.completed,
                )
                for transition in sample.trace.transitions
            )
        )
        table = AffordanceTable(
            dict(sample.affordances.features), dict(sample.affordances.sample_counts)
        )
        samples.append((trace, table))
    return tuple(samples)


def test_all_views_keep_same_optimum_examples_order_aliases_and_labels(fixture: Fixture) -> None:
    outputs = {
        condition: build_local_affordance_training_view(
            view=_view(fixture, condition),
            fold=fixture.fold,
            evidence_lock=fixture.evidence_lock,
            source_bundle=fixture.source_bundle,
            probe_capability=fixture.probe_capability,
        )
        for condition in CONDITION_IDS
    }
    assert all(receipt.example_count == 40 for receipt in outputs.values())
    assert all(receipt.task_example_counts == (1,) * 40 for receipt in outputs.values())
    for condition in CONDITION_IDS[1:]:
        assert tuple(row.selected_index for row in outputs[condition].examples) == tuple(
            row.selected_index for row in outputs[CONDITION_B2].examples
        )
        assert all(row.candidate_features.shape == (1, 54) for row in outputs[condition].examples)
    assert all(row.candidate_features.shape == (1, 49) for row in outputs[CONDITION_B2].examples)
    assert all(row.selected_index == 0 for row in outputs[CONDITION_B2].examples)
    evidence = fixture.evidence[0]
    source_samples = _core_samples(fixture.source_bundle)
    state = source_samples[0][0].transitions[0].before
    aliases, baseline, _ = candidate_tensor(state, source_samples[0][1])
    assert aliases == ("a",)
    _, s_expected, _ = state_availability_candidate_tensor(state, evidence)
    _, p_expected, _ = pooled_candidate_tensor(state, evidence)
    _, l_expected, _ = local_affordance_candidate_tensor(state, evidence)
    b2_expected = global_listwise_optimum_examples(
        source_samples
    )
    assert _tensor_bytes(outputs[CONDITION_B2].examples[0].candidate_features) == _tensor_bytes(b2_expected[0].candidate_features)
    assert _tensor_bytes(outputs[CONDITION_S].examples[0].candidate_features) == _tensor_bytes(
        apply_state_availability_mask(baseline)
    )
    assert _tensor_bytes(outputs[CONDITION_S].examples[0].candidate_features) == _tensor_bytes(s_expected)
    assert _tensor_bytes(outputs[CONDITION_P].examples[0].candidate_features) == _tensor_bytes(p_expected)
    assert _tensor_bytes(outputs[CONDITION_L].examples[0].candidate_features) == _tensor_bytes(l_expected)


def test_unknown_condition_wrong_matrix_or_heldout_leak_fails_closed(
    fixture: Fixture, authority_snapshot
) -> None:
    with pytest.raises(LocalAffordanceTrainingViewError, match="unsupported"):
        build_local_affordance_training_view(
            view=replace(_view(fixture, CONDITION_L), condition_id="not-frozen"),
            fold=fixture.fold,
            evidence_lock=fixture.evidence_lock,
            source_bundle=fixture.source_bundle,
            probe_capability=fixture.probe_capability,
        )
    with pytest.raises(LocalAffordanceTrainingViewError, match="view identity"):
        build_local_affordance_training_view(
            view=replace(
                _view(fixture, CONDITION_L),
                training_task_ids=tuple(reversed(_view(fixture, CONDITION_L).training_task_ids)),
            ),
            fold=fixture.fold,
            evidence_lock=fixture.evidence_lock,
            source_bundle=fixture.source_bundle,
            probe_capability=fixture.probe_capability,
        )

    first_sample = fixture.source_bundle.payload.samples[0]
    first_transition = first_sample.trace.transitions[0]
    changed_after = first_transition.after.model_copy(
        update={"progress_fraction": first_transition.after.progress_fraction - 0.001}
    )
    changed_transition = first_transition.model_copy(update={"after": changed_after})
    changed_trace = first_sample.trace.model_copy(
        update={"transitions": (changed_transition,)}
    )
    changed_sample = first_sample.model_copy(update={"trace": changed_trace})
    changed_payload = fixture.source_bundle.payload.model_copy(
        update={"samples": (changed_sample, *fixture.source_bundle.payload.samples[1:])}
    )
    tampered_bundle = replace(fixture.source_bundle, payload=changed_payload)
    with pytest.raises(LocalAffordanceTrainingViewError, match="payload bytes/manifest"):
        build_local_affordance_training_view(
            view=_view(fixture, CONDITION_P),
            fold=fixture.fold,
            evidence_lock=fixture.evidence_lock,
            source_bundle=tampered_bundle,
            probe_capability=fixture.probe_capability,
        )
    foreign_fold_capability = capabilities.issue_training_fold_probe_capability(
        authority_snapshot,
        fold_id="battery",
        replicate=0,
    )
    with pytest.raises(LocalAffordanceTrainingViewError, match="exact ordered 40-task fold"):
        build_local_affordance_training_view(
            view=_view(fixture, CONDITION_S),
            fold=fixture.fold,
            source_bundle=fixture.source_bundle,
            evidence_lock=fixture.evidence_lock,
            probe_capability=foreign_fold_capability,
        )


def test_learner_examples_have_no_task_or_probe_identity(fixture: Fixture) -> None:
    training_view = build_local_affordance_training_view(
        view=_view(fixture, CONDITION_L),
        fold=fixture.fold,
        evidence_lock=fixture.evidence_lock,
        source_bundle=fixture.source_bundle,
        probe_capability=fixture.probe_capability,
    )
    assert type(training_view) is LocalAffordanceTrainingView
    examples = training_view.examples
    example = examples[0]
    assert set(example.__dataclass_fields__) == {"candidate_features", "selected_index"}
    assert type(example.candidate_features).__name__ == "Tensor"
    assert {name for name in dir(fixture.probe_capability) if not name.startswith("_")} == {
        "consume_for"
    }
    assert not any(
        hasattr(fixture.probe_capability, name)
        for name in ("task_ids", "artifact_ids", "heldout_ids", "lookup", "enumerate")
    )
    assert all(type(row) is TaskLocalAffordanceEvidence for row in fixture.evidence)
    assert not any(
        hasattr(row, name)
        for row in fixture.evidence
        for name in ("task_id", "family_id", "fold_id", "artifact_id", "key_id")
    )
