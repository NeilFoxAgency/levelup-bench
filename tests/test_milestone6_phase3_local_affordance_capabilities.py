"""Capability-bound views of the complete Phase 3 raw-evidence authority.

These tests deliberately exercise only the synthetic, known-development store.
They are a boundary test: the authority snapshot may contain identity-bearing
bytes, while a learner capability may release only identity-free reducer input.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from levelup.experiments.milestone6_phase3_local_affordance_raw_authority import (
    FROZEN_LOCAL_AFFORDANCE_PROTOCOL_SHA256,
)
from levelup.experiments.milestone6_phase3_local_affordance_raw_store import (
    TrainingFoldManifest,
    open_existing_raw_probe_store,
)
from levelup.experiments.milestone6_phase3_plan import build_phase3_plan
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import PlannedUnit
from levelup.learning.state_conditioned import TaskLocalAffordanceEvidence

capabilities = pytest.importorskip(
    "levelup.experiments.milestone6_phase3_local_affordance_capabilities"
)
CapabilityError = capabilities.LocalAffordanceCapabilityError

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "milestone6"


@pytest.fixture(scope="session")
def expected_authority():
    authority_module = pytest.importorskip(
        "levelup.experiments.milestone6_phase3_local_affordance_raw_authority"
    )
    return authority_module.build_expected_raw_probe_authority(
        local_affordance_protocol_bytes=(
            CONFIG / "phase3_local_affordance_protocol.json"
        ).read_bytes(),
        development_protocol_bytes=(CONFIG / "development_protocol.json").read_bytes(),
        development_tasks_bytes=(CONFIG / "development_tasks.json").read_bytes(),
        phase3_evidence_lock_bytes=(CONFIG / "phase3_evidence_lock.json").read_bytes(),
    )


@pytest.fixture(scope="session")
def authority_snapshot(tmp_path_factory, expected_authority):
    """Validate the complete synthetic authority once for this module."""

    authority_module = pytest.importorskip(
        "levelup.experiments.milestone6_phase3_local_affordance_raw_authority"
    )
    publication_module = pytest.importorskip(
        "levelup.experiments.milestone6_phase3_local_affordance_raw_publication"
    )
    from test_milestone6_phase3_local_affordance_raw_authority import _artifact

    artifacts = tuple(
        authority_module.PersistedRawProbeArtifact.model_validate(_artifact(key)[0])
        for key in expected_authority.keys
    )
    root = tmp_path_factory.mktemp("capability-raw-authority") / "store"
    publication_module.publish_raw_probe_store(
        root,
        expected=expected_authority,
        artifacts=artifacts,
    )
    with open_existing_raw_probe_store(root) as reader:
        return authority_module.validate_complete_raw_probe_authority(
            reader,
            expected=expected_authority,
        )


@pytest.fixture(scope="session")
def phase3_planned_unit() -> PlannedUnit:
    plan = build_phase3_plan()
    # Keep this fixture in the known-development validation matrix only.  The
    # plan itself is frozen and cannot contain final-family access.
    assert plan.final_family_access is False
    planned = plan.units[0].unit
    condition_id, tuple_id = planned.key.condition_id.rsplit("--", 1)
    exposure = hashlib.sha256(
        canonical_json_bytes(
            {
                "protocol_sha256": FROZEN_LOCAL_AFFORDANCE_PROTOCOL_SHA256,
                "condition_id": condition_id,
                "tuple_id": tuple_id,
                "learner_visible": "optimum_only_development_training",
            }
        )
    ).hexdigest()
    return planned.model_copy(update={"exposure_manifest_sha256": exposure})


def _fold_task_ids(snapshot, fold_id: str, replicate: int) -> tuple[str, ...]:
    name = f"{fold_id}.r{replicate}.json"
    record = next(item for item in snapshot.training_fold_files if item.name == name)
    fold = TrainingFoldManifest.model_validate(json.loads(record.snapshot.canonical_bytes))
    return tuple(reference.key.task_id for reference in fold.task_references)


def _assert_identity_free(evidence: TaskLocalAffordanceEvidence) -> None:
    assert type(evidence) is TaskLocalAffordanceEvidence
    assert len(evidence.rows) == 64
    assert len(evidence.affordances.features) >= 1
    for forbidden in (
        "task_id",
        "family_id",
        "fold_id",
        "replicate",
        "artifact_id",
        "key_id",
        "path",
        "snapshot",
    ):
        assert not hasattr(evidence, forbidden)
    for row in evidence.rows:
        assert not hasattr(row, "task_id")
        assert not hasattr(row.transition, "task_id")


def test_training_capability_releases_exact_forty_identity_free_rows(
    authority_snapshot,
):
    task_ids = _fold_task_ids(authority_snapshot, "plain", 0)
    capability = capabilities.issue_training_fold_probe_capability(
        authority_snapshot, fold_id="plain", replicate=0
    )
    output = capability.consume_for(task_ids)
    assert type(output) is tuple
    assert len(output) == 40
    assert all(isinstance(item, TaskLocalAffordanceEvidence) for item in output)
    for item in output:
        _assert_identity_free(item)

    # The held-out family is absent from the training view, and the output is
    # not an object-level index into the raw authority.
    assert all("plain" not in task_id for task_id in task_ids)
    public = {name for name in dir(capability) if not name.startswith("_")}
    assert public <= {"consume_for"}
    for forbidden in (
        "lookup",
        "enumerate",
        "list_artifacts",
        "artifact_ids",
        "task_ids",
        "paths",
        "snapshot",
        "fold_id",
        "replicate",
    ):
        assert not hasattr(capability, forbidden)
    representation = repr(capability)
    assert all(task_id not in representation for task_id in task_ids)


def test_training_capability_preserves_pooled_parity(authority_snapshot):
    ids = _fold_task_ids(authority_snapshot, "plain", 1)
    cap = capabilities.issue_training_fold_probe_capability(
        authority_snapshot, fold_id="plain", replicate=1
    )
    rows = cap.consume_for(ids)
    first = rows[0].affordances
    assert all(item.affordances == first for item in rows)


def test_heldout_capability_releases_one_exact_identity_free_evidence(
    authority_snapshot, phase3_planned_unit
):
    planned = phase3_planned_unit
    cap = capabilities.issue_heldout_task_probe_capability(authority_snapshot, planned)
    evidence = cap.consume_for(planned)
    _assert_identity_free(evidence)
    public = {name for name in dir(cap) if not name.startswith("_")}
    assert public <= {"consume_for"}
    for forbidden in (
        "lookup",
        "enumerate",
        "artifact_id",
        "key_id",
        "task_id",
        "path",
        "snapshot",
        "planned",
    ):
        assert not hasattr(cap, forbidden)
    representation = repr(cap)
    assert planned.key.task_id not in representation
    assert planned.key.family_id not in representation
    assert planned.unit_id not in representation


@pytest.mark.parametrize(
    ("fold_id", "replicate"),
    [("plain", 9), ("not-a-family", 0)],
)
def test_training_capability_rejects_wrong_fold_or_replicate(
    authority_snapshot, fold_id, replicate
):
    with pytest.raises(CapabilityError):
        capabilities.issue_training_fold_probe_capability(
            authority_snapshot, fold_id=fold_id, replicate=replicate
        )


def test_training_consume_rejects_wrong_order_duplicate_or_unknown_task(
    authority_snapshot,
):
    ids = _fold_task_ids(authority_snapshot, "plain", 0)
    cap = capabilities.issue_training_fold_probe_capability(
        authority_snapshot, fold_id="plain", replicate=0
    )
    for supplied in (tuple(reversed(ids)), ids[:-1], ids + (ids[0],), ids[:-1] + ("unknown",)):
        with pytest.raises(CapabilityError):
            cap.consume_for(supplied)


@pytest.mark.parametrize("field", ["task_id", "family_id", "replicate", "task_index"])
def test_heldout_issue_rejects_rebound_planned_identity(
    authority_snapshot, phase3_planned_unit, field
):
    key = phase3_planned_unit.key
    value = (
        "forged-task"
        if field == "task_id"
        else (
            "battery"
            if field == "family_id"
            else (1 if field == "replicate" else key.task_index + 1)
        )
    )
    forged_key = key.model_copy(update={field: value})
    forged = phase3_planned_unit.model_copy(update={"key": forged_key})
    with pytest.raises(CapabilityError):
        capabilities.issue_heldout_task_probe_capability(authority_snapshot, forged)


@pytest.mark.parametrize(
    "mutation",
    [
        "task",
        "phase",
        "condition",
        "unit_id",
        "exposure",
        "model_seed",
        "probe_seed",
        "environment_seed",
        "search_seed",
        "data_order_seed",
    ],
)
def test_heldout_issue_rejects_forged_task_seed_or_phase(
    authority_snapshot, phase3_planned_unit, mutation
):
    key = phase3_planned_unit.key
    if mutation == "task":
        forged_key = key.model_copy(update={"task_id": "forged-task"})
        forged_unit = phase3_planned_unit.model_copy(update={"key": forged_key})
    elif mutation == "phase":
        forged_key = key.model_copy(update={"phase": "development"})
        forged_unit = phase3_planned_unit.model_copy(update={"key": forged_key})
    elif mutation == "condition":
        forged_key = key.model_copy(update={"condition_id": "forged--lr0p003-e120-t0p6"})
        forged_unit = phase3_planned_unit.model_copy(update={"key": forged_key})
    elif mutation == "unit_id":
        forged_unit = phase3_planned_unit.model_copy(update={"unit_id": "0" * 64})
    elif mutation == "exposure":
        forged_unit = phase3_planned_unit.model_copy(update={"exposure_manifest_sha256": "0" * 64})
    else:
        forged_seeds = phase3_planned_unit.seeds.model_copy(
            update={mutation: getattr(phase3_planned_unit.seeds, mutation) + 1}
        )
        forged_unit = phase3_planned_unit.model_copy(update={"seeds": forged_seeds})
    with pytest.raises(CapabilityError):
        capabilities.issue_heldout_task_probe_capability(authority_snapshot, forged_unit)


def test_heldout_consume_rejects_forged_planned_unit(authority_snapshot, phase3_planned_unit):
    cap = capabilities.issue_heldout_task_probe_capability(authority_snapshot, phase3_planned_unit)
    forged_key = phase3_planned_unit.key.model_copy(update={"task_id": "forged-task"})
    forged_unit = phase3_planned_unit.model_copy(update={"key": forged_key})
    with pytest.raises(CapabilityError):
        cap.consume_for(forged_unit)


def test_direct_construction_copy_and_rebound_capabilities_fail_closed(
    authority_snapshot, phase3_planned_unit
):
    training = capabilities.issue_training_fold_probe_capability(
        authority_snapshot,
        fold_id="plain",
        replicate=0,
    )
    heldout = capabilities.issue_heldout_task_probe_capability(
        authority_snapshot, phase3_planned_unit
    )
    for cls in (type(training), type(heldout)):
        with pytest.raises((CapabilityError, TypeError)):
            cls()
    rebound_training = copy.copy(training)
    object.__setattr__(rebound_training, "_task_ids", ("rebound",))
    rebound_heldout = copy.copy(heldout)
    object.__setattr__(
        rebound_heldout,
        "_planned",
        phase3_planned_unit.model_copy(update={"unit_id": "0" * 64}),
    )
    for forged in (rebound_training, rebound_heldout):
        with pytest.raises(CapabilityError):
            forged.consume_for(())


def test_forged_snapshot_copy_cannot_mint_capability(authority_snapshot):
    forged = authority_snapshot.model_copy(update={"key_ids": ()})
    with pytest.raises(CapabilityError):
        capabilities.issue_training_fold_probe_capability(forged, fold_id="plain", replicate=0)
