from __future__ import annotations

import hashlib

import pytest

from levelup.experiments.milestone6_phase3_anchor import (
    _ANCHOR_MANIFEST_TOKEN,
    Phase3AnchorManifest,
)
from levelup.experiments.milestone6_phase3_artifacts import (
    ARCHITECTURE_BY_CONDITION,
    CAPACITY_BY_CONDITION,
    TRAINING_TUPLES,
    EvidenceIdentity,
    Phase3ModelManifest,
    Phase3ReadinessEnvelope,
    Phase3UnitLineage,
    Phase3ViewManifest,
    TemperatureConsumer,
    TrainingReportRecord,
    TrainingSpecRecord,
    _validate_phase3_readiness_for_test,
    validate_phase3_readiness,
)
from levelup.experiments.milestone6_phase3_evidence import (
    _EVIDENCE_LOCK_TOKEN,
    Phase3EvidenceLock,
)
from levelup.experiments.milestone6_phase3_plan import bind_validated_phase3_plan, build_phase3_plan
from levelup.experiments.milestone6_phase3_protocol import FAMILIES, NEW_CONDITIONS
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataEvidenceKey,
    TrainingDataEvidenceManifest,
)


def sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def evidence_manifest(view, replicate: int, protocol_sha256: str = "a" * 64) -> TrainingDataEvidenceManifest:
    key = TrainingDataEvidenceKey(
        screening_candidates_sha256="a" * 64,
        protocol_sha256=protocol_sha256,
        task_manifest_sha256="b" * 64,
        expected_unit_plan_sha256="c" * 64,
        provenance_sha256="d" * 64,
        reference_exposure_sha256="e" * 64,
        probe_policy_sha256="f" * 64,
        fold_id=view.fold_id,
        heldout_family_id=view.heldout_family,
        ordered_training_task_ids=view.training_task_ids,
        ordered_heldout_task_ids=tuple(f"heldout-{i}" for i in range(8)),
        replicate=replicate,
        data_order_seed=view.data_order_seed,
        probe_seeds=tuple(range(len(view.training_task_ids))),
        environment_seeds=tuple(range(100, 100 + len(view.training_task_ids))),
    )
    body = {
        "schema_version": "runner.training-data-evidence.v1",
        "evidence_key_id": key.key_id,
        "key": key.model_dump(mode="json"),
        "payload_sha256": "1" * 64,
        "payload_bytes": 1,
        "sample_task_ids": view.training_task_ids,
    }
    return TrainingDataEvidenceManifest(evidence_id=sha(body), **body)


def evidence_identity(manifest: TrainingDataEvidenceManifest) -> EvidenceIdentity:
    return EvidenceIdentity(
        manifest=manifest,
        evidence_key=manifest.key,
        evidence_key_id=manifest.evidence_key_id,
        evidence_id=manifest.evidence_id,
        payload_sha256=manifest.payload_sha256,
        payload_bytes=manifest.payload_bytes,
        sample_task_ids=manifest.sample_task_ids,
        manifest_bytes_sha256=sha(manifest.model_dump(mode="json")),
    )


def synthetic_envelope():
    plan = build_phase3_plan()
    validated = bind_validated_phase3_plan(plan)
    view_by_pair = {(v.fold_id, v.replicate): v for v in plan.views if v.condition_id == NEW_CONDITIONS[0]}
    # Keep one evidence object per exact fold/replicate pair.
    manifests = {(v.fold_id, v.replicate): evidence_manifest(v, v.replicate, plan.protocol_sha256) for v in view_by_pair.values()}
    identities = {pair: evidence_identity(manifest) for pair, manifest in manifests.items()}
    # Build view self hashes after assembling the exact plan identities.
    view_rows = []
    for item in plan.views:
        payload = {
            "schema_version": "milestone6.phase3.view-manifest.v2",
            "view_id": item.view_id, "condition_id": item.condition_id,
            "fold_id": item.fold_id, "family_id": item.heldout_family,
            "replicate": item.replicate, "training_task_ids": list(item.training_task_ids),
            "data_order_seed": item.data_order_seed,
            "evidence": identities[(item.fold_id, item.replicate)].model_dump(mode="json"),
            "representation_identity_sha256": item.representation_sha256,
            "permutation_identity_sha256": "9" * 64 if item.condition_id == NEW_CONDITIONS[-1] else None,
            "plan_sha256": plan.plan_id, "protocol_sha256": plan.protocol_sha256,
        }
        view_rows.append(Phase3ViewManifest(manifest_sha256=sha(payload), **payload))
    view_rows = tuple(view_rows)
    model_rows = []
    for owner in plan.model_owners:
        lr, epochs = {"lr0p003-e120": (0.003, 120), "lr0p003-e180": (0.003, 180), "lr0p01-e120": (0.01, 120), "lr0p01-e180": (0.01, 180)}[owner.training_tuple_id]
        spec_body = {"training_tuple_id": owner.training_tuple_id, "epochs": epochs, "learning_rate": lr, "weight_decay": 0.0001}
        spec = TrainingSpecRecord(spec_sha256=sha(spec_body), **spec_body)
        report_body = {"trainable_parameters": CAPACITY_BY_CONDITION[owner.condition_id], "recurrent_steps": (0 if owner.condition_id == NEW_CONDITIONS[0] else 1), "optimizer_steps": epochs, "forward_passes": epochs, "training_examples": 1}
        report = TrainingReportRecord(report_sha256=sha(report_body), **report_body)
        consumers = tuple(
            TemperatureConsumer(
                temperature_id=temperature_id,
                consumer_unit_ids=tuple(
                    item.unit.unit_id
                    for item in plan.units
                    if item.model_owner_id == owner.owner_id
                    and item.tuple_id == temperature_id
                ),
            )
            for temperature_id in owner.search_temperature_ids
        )
        payload = {"schema_version": "milestone6.phase3.model-manifest.v2", "owner_id": owner.owner_id, "view_id": owner.view_id, "condition_id": owner.condition_id, "fold_id": owner.fold_id, "family_id": owner.heldout_family, "replicate": owner.replicate, "training_tuple_id": owner.training_tuple_id, "training_spec": spec.model_dump(mode="json"), "training_report": report.model_dump(mode="json"), "architecture_id": ARCHITECTURE_BY_CONDITION[owner.condition_id], "trainable_parameters": CAPACITY_BY_CONDITION[owner.condition_id], "model_state_sha256": "2" * 64, "ordered_tensor_sha256": ("3" * 64,), "artifact_id": "4" * 64, "artifact_manifest_sha256": "5" * 64, "temperature_consumers": [c.model_dump(mode="json") for c in consumers]}
        model_rows.append(Phase3ModelManifest(manifest_sha256=sha(payload), **payload))
    anchor_rows = []
    for family in FAMILIES:
        tasks = [x for x in plan.units if x.heldout_family == family]
        for item in tasks[: 8 * 5 * 12]:
            candidate = item.tuple_id
            anchor_rows.append({"family_id": family, "task_id": item.unit.key.task_id, "task_index": item.unit.key.task_index, "replicate": item.unit.key.replicate, "candidate_tuple_id": candidate, "base_condition_id": "B2-global-listwise-optimum", "unit_id": sha((family, item.unit.key.task_id, item.unit.key.replicate, candidate)), "run_id": f"run-{family}", "result_bytes_sha256": "6" * 64})
    # Add the exact 5,760 B2 rows with no condition-dependent duplication.
    unique = {(r["family_id"], r["task_id"], r["replicate"], r["candidate_tuple_id"]): r for r in anchor_rows}
    anchor_body = {"unit_results": list(unique.values())}
    anchor_sha256 = sha(anchor_body)
    anchor_body["anchor_manifest_sha256"] = anchor_sha256
    anchor = Phase3AnchorManifest(
        body=anchor_body,
        canonical_bytes=canonical_json_bytes(anchor_body),
        anchor_manifest_sha256=anchor_sha256,
        _construction_token=_ANCHOR_MANIFEST_TOKEN,
    )
    lock_body = {
        "schema_version": "milestone6.phase3.evidence-lock.v1",
        "scope": "known-development-only",
        "lineage": {
            "phase3_protocol_sha256": plan.protocol_sha256,
            "phase3_plan_id": plan.plan_id,
            "phase3_anchor_manifest_sha256": anchor.anchor_manifest_sha256,
            "phase3_anchor_file_sha256": hashlib.sha256(
                anchor.canonical_bytes
            ).hexdigest(),
        },
        "evidence_artifacts": [
            {"evidence_manifest": manifest.model_dump(mode="json")}
            for manifest in manifests.values()
        ],
    }
    lock_sha256 = sha(lock_body)
    lock_body["evidence_lock_sha256"] = lock_sha256
    evidence_lock = Phase3EvidenceLock(
        body=lock_body,
        canonical_bytes=canonical_json_bytes(lock_body),
        evidence_lock_sha256=lock_sha256,
        _construction_token=_EVIDENCE_LOCK_TOKEN,
    )
    lineage_rows = []
    for item in plan.units:
        row = unique[(item.heldout_family, item.unit.key.task_id, item.unit.key.replicate, item.tuple_id)]
        anchor_identity = {"anchor_unit_id": row["unit_id"], "anchor_run_id": row["run_id"], "anchor_result_bytes_sha256": row["result_bytes_sha256"], "anchor_family_id": row["family_id"], "anchor_task_id": row["task_id"], "anchor_task_index": row["task_index"], "anchor_replicate": row["replicate"], "anchor_candidate_tuple_id": row["candidate_tuple_id"], "anchor_condition_id": "B2-global-listwise-optimum"}
        body = {"schema_version": "milestone6.phase3.unit-lineage.v2", "unit_id": item.unit.unit_id, "key": item.unit.key.model_dump(mode="json"), "seeds": item.unit.seeds.model_dump(mode="json"), "exposure_manifest_sha256": item.unit.exposure_manifest_sha256, "plan_sha256": plan.plan_id, "view_id": item.view_id, "owner_id": item.model_owner_id, "tuple_id": item.tuple_id, "training_tuple_id": item.training_tuple_id, "anchor": anchor_identity}
        lineage_rows.append(Phase3UnitLineage(lineage_sha256=sha(body), **body))
    env_body = {"schema_version": "milestone6.phase3.artifacts.v2", "development_only": True, "final": False, "execution_authorized": False, "family_order": FAMILIES, "condition_ids": NEW_CONDITIONS, "training_tuple_ids": TRAINING_TUPLES, "evidence": tuple(x.model_dump(mode="json") for x in identities.values()), "views": tuple(x.model_dump(mode="json") for x in view_rows), "models": tuple(x.model_dump(mode="json") for x in model_rows), "units": tuple(x.model_dump(mode="json") for x in lineage_rows), "protocol_sha256": plan.protocol_sha256, "plan_id": plan.plan_id, "anchor_manifest_sha256": anchor.anchor_manifest_sha256, "anchor_file_sha256": hashlib.sha256(anchor.canonical_bytes).hexdigest(), "evidence_lock_sha256": evidence_lock.evidence_lock_sha256, "evidence_lock_file_sha256": hashlib.sha256(evidence_lock.canonical_bytes).hexdigest()}
    return validated, anchor, evidence_lock, Phase3ReadinessEnvelope(envelope_sha256=sha(env_body), **env_body)


def test_typed_evidence_rejects_rehashed_payload_identity() -> None:
    plan = build_phase3_plan()
    manifest = evidence_manifest(plan.views[0], 0, plan.protocol_sha256)
    value = evidence_identity(manifest).model_dump(mode="json")
    value["payload_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="payload identity"):
        EvidenceIdentity.model_validate(value)


def test_readiness_requires_external_authorities() -> None:
    with pytest.raises(ValueError, match="ValidatedPhase3Plan"):
        validate_phase3_readiness(
            {},
            plan=None,  # type: ignore[arg-type]
            anchor_manifest=None,  # type: ignore[arg-type]
            evidence_lock=None,  # type: ignore[arg-type]
        )


def test_public_readiness_rejects_uncommitted_synthetic_authorities() -> None:
    validated, anchor, evidence_lock, envelope = synthetic_envelope()
    with pytest.raises(ValueError, match="committed Phase 3 authority"):
        validate_phase3_readiness(
            envelope,
            plan=validated,
            anchor_manifest=anchor,
            evidence_lock=evidence_lock,
        )


def test_full_synthetic_inventory_and_external_rehash_rejection() -> None:
    validated, anchor, evidence_lock, envelope = synthetic_envelope()
    assert (len(envelope.evidence), len(envelope.views), len(envelope.models), len(envelope.units)) == (30, 120, 480, 11_520)
    _validate_phase3_readiness_for_test(
        envelope,
        plan=validated,
        anchor_manifest=anchor,
        evidence_lock=evidence_lock,
    )
    protocol_value = envelope.model_dump(mode="json")
    protocol_value["protocol_sha256"] = "f" * 64
    protocol_value["envelope_sha256"] = sha(
        {
            key: item
            for key, item in protocol_value.items()
            if key != "envelope_sha256"
        }
    )
    protocol_changed = Phase3ReadinessEnvelope.model_validate(protocol_value)
    with pytest.raises(ValueError, match="protocol identity differs"):
        _validate_phase3_readiness_for_test(
            protocol_changed,
            plan=validated,
            anchor_manifest=anchor,
            evidence_lock=evidence_lock,
        )
    altered = envelope.model_copy(deep=True)
    altered.views[0].evidence  # ensure the nested object is materialized
    value = altered.model_dump(mode="json")
    value["views"][0]["view_id"] = "a" * 64
    value["views"][0]["manifest_sha256"] = sha({k: v for k, v in value["views"][0].items() if k != "manifest_sha256"})
    changed = Phase3ReadinessEnvelope.model_validate(value)
    with pytest.raises(ValueError, match="validated Phase3Plan"):
        _validate_phase3_readiness_for_test(
            changed,
            plan=validated,
            anchor_manifest=anchor,
            evidence_lock=evidence_lock,
        )

    plan_value = envelope.model_dump(mode="json")
    plan_value["views"][0]["plan_sha256"] = "f" * 64
    plan_view_body = {
        key: item
        for key, item in plan_value["views"][0].items()
        if key != "manifest_sha256"
    }
    plan_value["views"][0]["manifest_sha256"] = sha(plan_view_body)
    plan_envelope_body = {
        key: item for key, item in plan_value.items() if key != "envelope_sha256"
    }
    plan_value["envelope_sha256"] = sha(plan_envelope_body)
    plan_changed = Phase3ReadinessEnvelope.model_validate(plan_value)
    with pytest.raises(ValueError, match="validated Phase3Plan"):
        _validate_phase3_readiness_for_test(
            plan_changed,
            plan=validated,
            anchor_manifest=anchor,
            evidence_lock=evidence_lock,
        )

    seed_value = envelope.model_dump(mode="json")
    seed_value["units"][0]["seeds"]["probe_seed"] += 1
    seed_body = {
        key: item
        for key, item in seed_value["units"][0].items()
        if key != "lineage_sha256"
    }
    seed_value["units"][0]["lineage_sha256"] = sha(seed_body)
    envelope_body = {
        key: item for key, item in seed_value.items() if key != "envelope_sha256"
    }
    seed_value["envelope_sha256"] = sha(envelope_body)
    seed_changed = Phase3ReadinessEnvelope.model_validate(seed_value)
    with pytest.raises(ValueError, match="key, seeds, or exposure"):
        _validate_phase3_readiness_for_test(
            seed_changed,
            plan=validated,
            anchor_manifest=anchor,
            evidence_lock=evidence_lock,
        )

    anchor_value = envelope.model_dump(mode="json")
    anchor_value["units"][0]["anchor"]["anchor_unit_id"] = "f" * 64
    anchor_body = {
        key: item
        for key, item in anchor_value["units"][0].items()
        if key != "lineage_sha256"
    }
    anchor_value["units"][0]["lineage_sha256"] = sha(anchor_body)
    envelope_body = {
        key: item for key, item in anchor_value.items() if key != "envelope_sha256"
    }
    anchor_value["envelope_sha256"] = sha(envelope_body)
    anchor_changed = Phase3ReadinessEnvelope.model_validate(anchor_value)
    with pytest.raises(ValueError, match="anchor differs"):
        _validate_phase3_readiness_for_test(
            anchor_changed,
            plan=validated,
            anchor_manifest=anchor,
            evidence_lock=evidence_lock,
        )


def test_readiness_envelope_is_explicitly_non_executable() -> None:
    with pytest.raises(ValueError, match="execution_authorized"):
        Phase3ReadinessEnvelope.model_validate(
            {
                "schema_version": "milestone6.phase3.artifacts.v2",
                "envelope_sha256": "a" * 64,
                "development_only": True,
                "final": False,
                "execution_authorized": True,
            }
        )
