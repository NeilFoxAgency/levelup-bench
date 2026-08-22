"""Fail-closed tests for the Phase 3 canonical evidence lock."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from levelup.experiments.milestone6_phase3_anchor import (
    _ANCHOR_MANIFEST_TOKEN,
    Phase3AnchorManifest,
)
from levelup.experiments.milestone6_phase3_evidence import (
    EvidenceLockError,
    Phase3EvidenceLock,
    _build_phase3_evidence_lock_for_test,
    _validate_phase3_evidence_lock_bytes_for_test,
    build_phase3_evidence_lock,
    load_committed_phase3_evidence_lock_bytes,
)
from levelup.experiments.milestone6_phase3_plan import (
    FAMILIES,
    bind_validated_phase3_plan,
    build_phase3_plan,
    canonical_phase3_plan_lock_bytes,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import TrainingPreparationAccounting
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataEvidenceCostRecord,
    TrainingDataEvidenceKey,
    TrainingDataEvidenceManifest,
)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def test_evidence_lock_cannot_be_constructed_without_validated_runtime_gate() -> None:
    with pytest.raises(EvidenceLockError, match="validated Phase 2 runtime gate"):
        Phase3EvidenceLock(
            body={},
            canonical_bytes=b"{}",
            evidence_lock_sha256="a" * 64,
        )


def test_committed_evidence_lock_is_canonical_and_exact() -> None:
    content = load_committed_phase3_evidence_lock_bytes()
    body = json.loads(content)
    assert hashlib.sha256(content).hexdigest() == (
        "82644954b94bd6ff495c425ffe921d18157a98ccbe230d922c24218a4faad875"
    )
    assert body["evidence_lock_sha256"] == (
        "7db4ad251f1e20ea14902c2643425ebfc2ef1064c4ea5ca5c90eb0629c2386b3"
    )
    assert body["counts"]["evidence_artifacts"] == 30


def _runtime(plan):
    folds = []
    for family in FAMILIES:
        views = tuple(view for view in plan.views if view.heldout_family == family)
        training = views[0].training_task_ids
        fold_id = views[0].fold_id
        config = SimpleNamespace(
            parameters={"fold_id": fold_id},
            split=SimpleNamespace(
                final_tasks=(),
                validation_tasks=tuple(
                    SimpleNamespace(
                        task_id=f"heldout-{family}-{i}", task_index=i
                    )
                    for i in range(8)
                ),
                development_tasks=tuple(SimpleNamespace(task_id=task_id, family_id="other") for task_id in training),
            ),
        )
        evidence = {}
        manifests = {}
        costs = {}
        cost_records = {}
        for rep in range(5):
            data_order_seed = next(
                view.data_order_seed for view in views if view.replicate == rep
            )
            key = TrainingDataEvidenceKey(
                screening_candidates_sha256="a" * 64,
                protocol_sha256="b" * 64,
                task_manifest_sha256="c" * 64,
                expected_unit_plan_sha256="d" * 64,
                provenance_sha256="e" * 64,
                reference_exposure_sha256="f" * 64,
                probe_policy_sha256="1" * 64,
                fold_id=fold_id,
                heldout_family_id=family,
                ordered_training_task_ids=training,
                ordered_heldout_task_ids=tuple(f"heldout-{family}-{i}" for i in range(8)),
                replicate=rep,
                data_order_seed=data_order_seed,
                probe_seeds=tuple(range(40)),
                environment_seeds=tuple(range(40, 80)),
            )
            manifest_body = {
                "schema_version": "runner.training-data-evidence.v1",
                "evidence_key_id": key.key_id,
                "key": key.model_dump(mode="json"),
                "payload_sha256": "2" * 64,
                "payload_bytes": 1,
                "sample_task_ids": list(training),
            }
            manifest_body["evidence_id"] = _digest({**manifest_body})
            manifest = TrainingDataEvidenceManifest.model_validate(manifest_body)
            cost_body = {
                "schema_version": "runner.training-data-evidence-cost.v1",
                "key_id": key.key_id,
                "artifact_id": manifest.evidence_id,
                "scope": "training_data_evidence_preparation",
                "key": key.model_dump(mode="json"),
                "accounting": TrainingPreparationAccounting().model_dump(mode="json"),
            }
            cost = TrainingDataEvidenceCostRecord(
                cost_id=_digest(cost_body),
                **cost_body,
            )
            evidence[rep] = key
            manifests[rep] = manifest
            costs[rep] = cost.cost_id
            cost_records[key.key_id] = cost
        data_keys = SimpleNamespace(evidence=evidence)
        data = SimpleNamespace(manifests=SimpleNamespace(evidence=manifests), evidence_cost_ids=costs)
        folds.append(
            SimpleNamespace(
                family_id=family,
                config=config,
                store=SimpleNamespace(
                    run_id=f"run-{family}",
                    load_shared_cost=lambda key_id, kind, records=cost_records: (
                        records[key_id]
                        if kind == "training_data_evidence"
                        else None
                    ),
                ),
                data_keys=data_keys,
                data=data,
            )
        )
    children = tuple(SimpleNamespace(heldout_family_id=family, run_id=f"run-{family}") for family in FAMILIES)
    manifest = SimpleNamespace(
        family_order=FAMILIES,
        development_only=True,
        validation_executed=False,
        search_executed=False,
        outcomes_present=False,
        selection_performed=False,
        final_family_access=False,
        children=children,
        manifest_sha256="4" * 64,
        provenance_sha256="5" * 64,
    )
    return SimpleNamespace(
        manifest=manifest,
        manifest_bytes=b"readiness",
        tree_sha256="6" * 64,
        provenance=SimpleNamespace(identity="provenance"),
        result_namespace_snapshot=(),
        folds=tuple(folds),
    )


def _anchor(runtime, plan) -> Phase3AnchorManifest:
    body = {
        "schema_version": "synthetic",
        "scope": "known-development-only",
        "lineage": {
            "phase3_protocol_sha256": plan.protocol_sha256,
            "phase2_readiness_manifest_sha256": runtime.manifest.manifest_sha256,
            "phase2_readiness_manifest_bytes_sha256": hashlib.sha256(
                runtime.manifest_bytes
            ).hexdigest(),
            "phase2_result_namespace_snapshot_sha256": _digest(
                runtime.result_namespace_snapshot
            ),
            "phase2_tree_sha256": runtime.tree_sha256,
            "phase2_provenance_sha256": runtime.manifest.provenance_sha256,
        },
    }
    body["anchor_manifest_sha256"] = _digest(body)
    return Phase3AnchorManifest(
        body=body,
        canonical_bytes=canonical_json_bytes(body),
        anchor_manifest_sha256=body["anchor_manifest_sha256"],
        _construction_token=_ANCHOR_MANIFEST_TOKEN,
    )


@pytest.fixture(scope="module")
def locked():
    plan = build_phase3_plan()
    validated = bind_validated_phase3_plan(plan)
    runtime = _runtime(plan)
    anchor = _anchor(runtime, plan)
    return (
        runtime,
        validated,
        anchor,
        anchor.canonical_bytes,
        canonical_phase3_plan_lock_bytes(plan),
    )


def test_phase3_evidence_lock_has_exact_30_rows_and_is_deterministic(locked):
    runtime, validated, anchor, anchor_file_bytes, plan_lock_bytes = locked
    first = _build_phase3_evidence_lock_for_test(
        runtime,
        validated,
        anchor,
        anchor_file_bytes,
        plan_lock_bytes,
    )
    second = _build_phase3_evidence_lock_for_test(
        runtime,
        validated,
        anchor,
        anchor_file_bytes,
        plan_lock_bytes,
    )
    assert len(first.body["evidence_artifacts"]) == 30
    assert first.canonical_bytes == second.canonical_bytes
    assert first.body["counts"]["evidence_artifacts"] == 30


def test_phase3_evidence_lock_rejects_rehashed_bytes(locked):
    runtime, validated, anchor, anchor_file_bytes, plan_lock_bytes = locked
    lock = _build_phase3_evidence_lock_for_test(
        runtime,
        validated,
        anchor,
        anchor_file_bytes,
        plan_lock_bytes,
    )
    body = json.loads(lock.canonical_bytes)
    body["evidence_artifacts"][0]["payload_bytes"] += 1
    body["evidence_lock_sha256"] = _digest({key: value for key, value in body.items() if key != "evidence_lock_sha256"})
    with pytest.raises(EvidenceLockError, match="differs|payload"):
        _validate_phase3_evidence_lock_bytes_for_test(
            canonical_json_bytes(body),
            runtime=runtime,
            validated_plan=validated,
            anchor_manifest=anchor,
            anchor_file_bytes=anchor_file_bytes,
            plan_lock_bytes=plan_lock_bytes,
        )
    with pytest.raises(EvidenceLockError, match="anchor file bytes"):
        _build_phase3_evidence_lock_for_test(
            runtime,
            validated,
            anchor,
            b"{}",
            plan_lock_bytes,
        )
    with pytest.raises(EvidenceLockError, match="plan lock bytes"):
        _build_phase3_evidence_lock_for_test(
            runtime,
            validated,
            anchor,
            anchor_file_bytes,
            plan_lock_bytes + b"\n",
        )


def test_phase3_evidence_lock_rejects_final_runtime(locked):
    runtime, validated, anchor, anchor_file_bytes, plan_lock_bytes = locked
    original = runtime.folds[0].config.split.final_tasks
    runtime.folds[0].config.split.final_tasks = ("final",)
    try:
        with pytest.raises(EvidenceLockError, match="final"):
            _build_phase3_evidence_lock_for_test(
                runtime,
                validated,
                anchor,
                anchor_file_bytes,
                plan_lock_bytes,
            )
    finally:
        runtime.folds[0].config.split.final_tasks = original


def test_phase3_evidence_lock_rejects_unvalidated_runtime_and_lineage_drift(locked):
    runtime, validated, anchor, anchor_file_bytes, plan_lock_bytes = locked
    with pytest.raises(EvidenceLockError, match="freshly revalidated"):
        build_phase3_evidence_lock(
            runtime,
            validated,
            anchor,
            anchor_file_bytes,
            plan_lock_bytes,
        )

    first_fold = runtime.folds[0]
    original_heldout = first_fold.config.split.validation_tasks
    first_fold.config.split.validation_tasks = (
        SimpleNamespace(task_id="foreign-heldout", task_index=0),
        *original_heldout[1:],
    )
    try:
        with pytest.raises(EvidenceLockError, match="tasks differ"):
            _build_phase3_evidence_lock_for_test(
                runtime,
                validated,
                anchor,
                anchor_file_bytes,
                plan_lock_bytes,
            )
    finally:
        first_fold.config.split.validation_tasks = original_heldout

    original_cost_id = first_fold.data.evidence_cost_ids[0]
    first_fold.data.evidence_cost_ids[0] = "7" * 64
    try:
        with pytest.raises(EvidenceLockError, match="cost lineage"):
            _build_phase3_evidence_lock_for_test(
                runtime,
                validated,
                anchor,
                anchor_file_bytes,
                plan_lock_bytes,
            )
    finally:
        first_fold.data.evidence_cost_ids[0] = original_cost_id
