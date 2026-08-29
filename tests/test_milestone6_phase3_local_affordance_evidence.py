from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from levelup.experiments.milestone6_baselines import ProbeAccounting, ProbeEvidence
from levelup.experiments.milestone6_phase3_local_affordance_evidence import (
    LocalAffordanceEvidenceError,
    RawProbeArtifactKey,
    RawProbeReducerCapability,
    RawProbeTransitionRecord,
    SanitizedRawProbeArtifact,
    sanitize_probe_evidence,
    task_local_affordance_evidence_from_artifact,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import (
    AffordanceTableRecord,
    ObservableStateRecord,
)
from levelup.learning.state_conditioned import (
    AffordanceTable,
    ObservableState,
    ObservedTransition,
    local_affordance_candidate_tensor,
)

HASHES = {
    "local_affordance_protocol_sha256": "1" * 64,
    "development_protocol_sha256": "2" * 64,
    "development_tasks_sha256": "3" * 64,
    "phase3_evidence_lock_sha256": "4" * 64,
    "probe_policy_sha256": "5" * 64,
}


def _state(progress: float = 0.1) -> ObservableState:
    return ObservableState(progress, 1.0 - progress, progress, 0.5, 0.2, ("a",))


def _evidence(*, task_id: str = "plain-task-0") -> ProbeEvidence:
    transitions = tuple(
        ObservedTransition(_state(index / 100), "a", _state((index + 1) / 100), False)
        for index in range(64)
    )
    from levelup.learning.state_conditioned import build_affordance_table

    table = build_affordance_table(transitions, target_samples_per_alias=8)
    return ProbeEvidence(
        task_id=task_id,
        affordances=table,
        transitions=transitions,
        accounting=ProbeAccounting(4, 4, 64, ("a",), 0.0),
    )


def _canonical(evidence: ProbeEvidence) -> AffordanceTableRecord:
    return AffordanceTableRecord(
        features=dict(evidence.affordances.features),
        sample_counts=dict(evidence.affordances.sample_counts),
    )


def _sanitize(*, task_index: int = 0) -> SanitizedRawProbeArtifact:
    evidence = _evidence()
    return sanitize_probe_evidence(
        evidence,
        **HASHES,
        family_id="plain",
        replicate=0,
        task_index=task_index,
        task_id=evidence.task_id,
        generator_seed=10,
        probe_seed=11,
        environment_seed=12,
        canonical_affordances=_canonical(evidence),
    )


def test_key_requires_all_hashes_and_strict_nonnegative_integer_identity() -> None:
    complete = dict(
        **HASHES,
        family_id="plain",
        replicate=0,
        task_index=0,
        task_id="plain-task-0",
        generator_seed=10,
        probe_seed=11,
        environment_seed=12,
    )
    assert RawProbeArtifactKey(**complete).task_index == 0
    for field in HASHES:
        with pytest.raises(ValidationError):
            RawProbeArtifactKey(**{key: value for key, value in complete.items() if key != field})
    for field in ("replicate", "task_index", "generator_seed", "probe_seed", "environment_seed"):
        with pytest.raises(ValidationError):
            RawProbeArtifactKey(**{**complete, field: True})
        with pytest.raises(ValidationError):
            RawProbeArtifactKey(**{**complete, field: -1})
    # The frozen seed policy uses the source manifest index, which is not a 0..7 slot;
    # momentum development tasks, for example, have manifest indices above 100.
    assert RawProbeArtifactKey(**{**complete, "task_index": 126}).task_index == 126


def test_row_requires_available_action_and_forbids_extra_metadata() -> None:
    state = ObservableStateRecord(
        progress_fraction=0.1,
        remaining_fraction=0.9,
        elapsed_per_target=0.1,
        resource_fraction=0.5,
        pressure_fraction=0.2,
        available_aliases=("a",),
    )
    with pytest.raises(ValidationError, match="unavailable"):
        RawProbeTransitionRecord(
            probe_index=0, before=state, action_alias="b", after=state, completed=False
        )
    with pytest.raises(ValidationError):
        RawProbeTransitionRecord(
            probe_index=0,
            before=state,
            action_alias="a",
            after=state,
            completed=False,
            task_id="forbidden",
        )
    with pytest.raises(ValidationError):
        RawProbeTransitionRecord(
            probe_index=0,
            before=state,
            action_alias="a",
            after=state,
            completed=1,
        )


def test_sanitizer_requires_both_live_and_canonical_pooled_parity() -> None:
    evidence = _evidence()
    canonical = _canonical(evidence)
    changed_table = AffordanceTable(
        {"a": (999.0,) + evidence.affordances.features["a"][1:]}, {"a": 64}
    )
    with pytest.raises(LocalAffordanceEvidenceError, match="ProbeEvidence"):
        sanitize_probe_evidence(
            replace(evidence, affordances=changed_table),
            **HASHES,
            family_id="plain",
            replicate=0,
            task_index=0,
            task_id=evidence.task_id,
            generator_seed=10,
            probe_seed=11,
            environment_seed=12,
            canonical_affordances=canonical,
        )
    changed_canonical = canonical.model_copy(
        update={"features": {"a": (999.0,) + canonical.features["a"][1:]}}
    )
    with pytest.raises(LocalAffordanceEvidenceError, match="canonical v1"):
        sanitize_probe_evidence(
            evidence,
            **HASHES,
            family_id="plain",
            replicate=0,
            task_index=0,
            task_id=evidence.task_id,
            generator_seed=10,
            probe_seed=11,
            environment_seed=12,
            canonical_affordances=changed_canonical,
        )
    with pytest.raises(LocalAffordanceEvidenceError, match="required"):
        sanitize_probe_evidence(
            evidence,
            **HASHES,
            family_id="plain",
            replicate=0,
            task_index=0,
            task_id=evidence.task_id,
            generator_seed=10,
            probe_seed=11,
            environment_seed=12,
            canonical_affordances=None,  # type: ignore[arg-type]
        )


def test_forged_key_body_manifest_and_capability_combinations_fail_closed() -> None:
    first = _sanitize(task_index=0)
    second = _sanitize(task_index=1)
    with pytest.raises(LocalAffordanceEvidenceError, match="key and manifest"):
        SanitizedRawProbeArtifact(
            second.key, first.body, first.manifest, first.affordances, first.reducer_capability
        )
    forged_manifest = first.manifest.model_copy(update={"body_sha256": "0" * 64})
    with pytest.raises(LocalAffordanceEvidenceError, match="manifest"):
        SanitizedRawProbeArtifact(
            first.key, first.body, forged_manifest, first.affordances, first.reducer_capability
        )
    with pytest.raises(ValueError, match="invalid"):
        RawProbeReducerCapability(first.manifest.artifact_id, object())
    with pytest.raises(LocalAffordanceEvidenceError, match="authorize"):
        task_local_affordance_evidence_from_artifact(first, second.reducer_capability)
    object.__setattr__(first.reducer_capability, "artifact_id", second.manifest.artifact_id)
    with pytest.raises(LocalAffordanceEvidenceError, match="capability"):
        task_local_affordance_evidence_from_artifact(first, first.reducer_capability)


def test_reducer_conversion_strips_ids_hashes_and_index_from_actual_tensor() -> None:
    artifact = _sanitize()
    evidence = task_local_affordance_evidence_from_artifact(
        artifact, artifact.reducer_capability
    )
    assert len(evidence.task_rows.rows) == 64
    assert not hasattr(evidence, "task_id")
    assert not hasattr(evidence, "family_id")
    aliases, tensor, unknown = local_affordance_candidate_tensor(_state(0.3), evidence)
    assert aliases == ("a",)
    assert unknown == 0
    assert tensor.shape == (1, 54)
    raw = tensor.detach().cpu().numpy().tobytes()
    for forbidden in (
        artifact.key.task_id.encode(),
        artifact.key.family_id.encode(),
        artifact.manifest.artifact_id.encode(),
        artifact.body.content_sha256.encode(),
    ):
        assert forbidden not in raw
    # The reducer retains the index only in its typed wrapper, not in transitions/tensors.
    assert evidence.task_rows.rows[7].probe_index == 7
    assert not hasattr(evidence.task_rows.rows[7].transition, "probe_index")
    dumped = canonical_json_bytes(artifact.body.model_dump(mode="json"))
    assert b"family_id" not in dumped and b"task_id" not in dumped
