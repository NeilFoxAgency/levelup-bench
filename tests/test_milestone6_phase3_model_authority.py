from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from levelup.experiments.milestone6_phase3_model_authority import (
    Phase3ModelArtifactAuthority,
    Phase3ModelAuthorityCost,
    Phase3ModelAuthorityError,
    Phase3ModelAuthorityRow,
    _validate_phase3_authority_source_shapes,
    canonical_phase3_model_authority_bytes,
    load_phase3_model_artifact_authority_bytes,
    write_phase3_model_authority,
)
from levelup.experiments.milestone6_phase3_model_preparation import EXPECTED_MODELS
from levelup.experiments.milestone6_phase3_plan import (
    REPLICATES,
    TRAINING_TUPLE_IDS,
    Phase3Plan,
    validate_phase3_plan_lock_bytes,
)
from levelup.experiments.milestone6_phase3_protocol import FAMILIES, NEW_CONDITIONS
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import PhaseAccounting
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataEvidenceCostRecord,
    TrainingDataEvidenceKey,
    TrainingDataEvidenceManifest,
)


def _authority() -> Phase3ModelArtifactAuthority:
    zero = PhaseAccounting()
    cost = Phase3ModelAuthorityCost(
        setup=zero,
        training_probes=zero,
        reference_replay=zero,
        training=PhaseAccounting(optimizer_steps=1, forward_passes=1),
        serialization=PhaseAccounting(calls=EXPECTED_MODELS),
    )
    rows = tuple(
        Phase3ModelAuthorityRow(
            owner_id=f"{index:064x}",
            key_id=f"{index + 1000:064x}",
            artifact_id=f"{index + 2000:064x}",
            manifest_sha256=f"{index + 2500:064x}",
            cost_id=f"{index + 3000:064x}",
        )
        for index in range(EXPECTED_MODELS)
    )
    body = {
        "schema_version": "milestone6.phase3.model-artifact-authority.v1",
        "development_only": True,
        "final": False,
        "final_family_accessed": False,
        "execution_authorized": True,
        "artifact_store_id": "phase3-model-preparation-test",
        "plan_id": "b" * 64,
        "protocol_sha256": "c" * 64,
        "plan_file_sha256": "8" * 64,
        "anchor_manifest_sha256": "d" * 64,
        "anchor_file_sha256": "e" * 64,
        "evidence_lock_sha256": "f" * 64,
        "evidence_file_sha256": "1" * 64,
        "preparation_git_commit_sha": "2" * 40,
        "preparation_provenance_sha256": "3" * 64,
        "generation_git_commit_sha": "4" * 40,
        "progress_sha256": "5" * 64,
        "provenance_file_sha256": "6" * 64,
        "family_order": FAMILIES,
        "condition_ids": NEW_CONDITIONS,
        "replicates": REPLICATES,
        "training_tuple_ids": TRAINING_TUPLE_IDS,
        "owner_ids": tuple(row.owner_id for row in rows),
        "unit_owner_mapping_sha256": "7" * 64,
        "expected_evidence_count": 30,
        "expected_view_count": 120,
        "expected_model_count": EXPECTED_MODELS,
        "allowed_cost_accounting": cost.model_dump(mode="json"),
        "models": tuple(row.model_dump(mode="json") for row in rows),
    }
    body["authority_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return Phase3ModelArtifactAuthority.model_validate(body)


def _rehash(value: Phase3ModelArtifactAuthority, **updates: object) -> dict[str, object]:
    body = value.model_dump(mode="json")
    for key, update in updates.items():
        if hasattr(update, "model_dump"):
            body[key] = update.model_dump(mode="json")
        elif isinstance(update, tuple) and update and hasattr(update[0], "model_dump"):
            body[key] = tuple(item.model_dump(mode="json") for item in update)
        else:
            body[key] = update
    body.pop("authority_sha256")
    body["authority_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body


def _committed_authority_sources() -> tuple[Phase3Plan, dict[str, object], dict[str, object]]:
    config_root = Path(__file__).parents[1] / "configs" / "milestone6"
    return (
        validate_phase3_plan_lock_bytes((config_root / "phase3_plan_lock.json").read_bytes()),
        json.loads((config_root / "phase3_anchor_manifest.json").read_bytes()),
        json.loads((config_root / "phase3_evidence_lock.json").read_bytes()),
    )


def _reself_hash(body: dict[str, object], field: str) -> None:
    unsigned = {key: value for key, value in body.items() if key != field}
    body[field] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def _coherently_replace_first_evidence_key(
    evidence: dict[str, object], key_body: dict[str, object]
) -> None:
    rows = evidence["evidence_artifacts"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    row = rows[0]
    key = TrainingDataEvidenceKey.model_validate(key_body)
    tasks = list(key.ordered_training_task_ids)

    manifest_body = dict(row["evidence_manifest"])
    manifest_body.update(
        {
            "key": key.model_dump(mode="json"),
            "evidence_key_id": key.key_id,
            "sample_task_ids": tasks,
        }
    )
    manifest_body.pop("evidence_id")
    manifest_body["evidence_id"] = hashlib.sha256(canonical_json_bytes(manifest_body)).hexdigest()
    manifest = TrainingDataEvidenceManifest.model_validate(manifest_body)

    cost_body = dict(row["evidence_cost"])
    cost_body.update(
        {
            "key": key.model_dump(mode="json"),
            "key_id": key.key_id,
            "artifact_id": manifest.evidence_id,
        }
    )
    cost_body.pop("cost_id")
    cost_body["cost_id"] = hashlib.sha256(canonical_json_bytes(cost_body)).hexdigest()
    cost = TrainingDataEvidenceCostRecord.model_validate(cost_body)

    row.update(
        {
            "evidence_key": key.model_dump(mode="json"),
            "evidence_key_id": key.key_id,
            "evidence_manifest": manifest.model_dump(mode="json"),
            "evidence_manifest_key_id": key.key_id,
            "evidence_id": manifest.evidence_id,
            "evidence_cost": cost.model_dump(mode="json"),
            "evidence_cost_id": cost.cost_id,
            "ordered_training_task_ids": tasks,
            "canonical_manifest_bytes_sha256": hashlib.sha256(
                canonical_json_bytes(manifest.model_dump(mode="json"))
            ).hexdigest(),
        }
    )
    _reself_hash(evidence, "evidence_lock_sha256")


def test_authority_bytes_are_deterministic_and_self_hashed() -> None:
    value = _authority()
    first = canonical_phase3_model_authority_bytes(value)
    second = canonical_phase3_model_authority_bytes(
        Phase3ModelArtifactAuthority.model_validate(value.model_dump(mode="json"))
    )
    assert first == second
    assert (
        hashlib.sha256(
            canonical_json_bytes({**value.model_dump(mode="json"), "authority_sha256": None})
        ).hexdigest()
        != value.authority_sha256
    )


def test_committed_model_authority_is_exact_and_development_only() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "milestone6"
        / "phase3_model_artifact_authority.json"
    )
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == (
        "eecd68707e2cdfa34e9e9b30f787fd17b87ae767db63b659944e420cb7255388"
    )
    value = load_phase3_model_artifact_authority_bytes(content)
    assert value.authority_sha256 == (
        "8771eb52433faf15d6e5e935902a5c935526ec0e6b8e34621c3d6a922aea1a52"
    )
    assert value.generation_git_commit_sha == ("2758cdcefc1da0694573649a8b5cc4b726a38281")
    assert value.preparation_git_commit_sha == ("cc0820791427ac56acb8c50599446d99a7e06883")
    assert value.development_only is True
    assert value.execution_authorized is True
    assert value.final is False
    assert value.final_family_accessed is False
    assert len(value.models) == len(value.owner_ids) == value.expected_model_count == 480
    assert value.expected_evidence_count == 30
    assert value.expected_view_count == 120


def test_authority_byte_loader_requires_exact_canonical_bytes() -> None:
    value = _authority()
    canonical = canonical_phase3_model_authority_bytes(value)
    assert load_phase3_model_artifact_authority_bytes(canonical) == value
    with pytest.raises(Phase3ModelAuthorityError, match="not canonical"):
        load_phase3_model_artifact_authority_bytes(canonical + b"\n")
    with pytest.raises(Phase3ModelAuthorityError, match="not canonical"):
        load_phase3_model_artifact_authority_bytes(b"{}")


def test_authority_source_shapes_reject_reself_hashed_top_level_extra() -> None:
    plan, anchor, evidence = _committed_authority_sources()
    anchor["final_payload"] = []
    _reself_hash(anchor, "anchor_manifest_sha256")
    with pytest.raises(Phase3ModelAuthorityError, match="anchor schema"):
        _validate_phase3_authority_source_shapes(anchor, evidence, plan=plan)


def test_authority_source_shapes_reject_reself_hashed_evidence_row_extra() -> None:
    plan, anchor, evidence = _committed_authority_sources()
    rows = evidence["evidence_artifacts"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    rows[0]["outcomes"] = []
    _reself_hash(evidence, "evidence_lock_sha256")
    with pytest.raises(Phase3ModelAuthorityError, match="evidence artifact schema"):
        _validate_phase3_authority_source_shapes(anchor, evidence, plan=plan)


def test_authority_source_shapes_reject_reself_hashed_evidence_alias_drift() -> None:
    plan, anchor, evidence = _committed_authority_sources()
    rows = evidence["evidence_artifacts"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    rows[0]["payload_sha256"] = "0" * 64
    _reself_hash(evidence, "evidence_lock_sha256")
    with pytest.raises(Phase3ModelAuthorityError, match="row aliases"):
        _validate_phase3_authority_source_shapes(anchor, evidence, plan=plan)


def test_authority_source_shapes_reject_reself_hashed_phase2_lineage_drift() -> None:
    plan, anchor, evidence = _committed_authority_sources()
    lineage = evidence["lineage"]
    assert isinstance(lineage, dict)
    lineage["phase2_tree_sha256"] = "0" * 64
    _reself_hash(evidence, "evidence_lock_sha256")
    with pytest.raises(Phase3ModelAuthorityError, match="retained Phase 2 lineage"):
        _validate_phase3_authority_source_shapes(anchor, evidence, plan=plan)


def test_authority_source_shapes_reject_coherently_rehashed_task_set_drift() -> None:
    plan, anchor, evidence = _committed_authority_sources()
    rows = evidence["evidence_artifacts"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    key_body = dict(rows[0]["evidence_key"])
    tasks = list(key_body["ordered_training_task_ids"])
    tasks[0] = "final.hidden.task"
    key_body["ordered_training_task_ids"] = tasks
    _coherently_replace_first_evidence_key(evidence, key_body)
    with pytest.raises(Phase3ModelAuthorityError, match="acquisition authority"):
        _validate_phase3_authority_source_shapes(anchor, evidence, plan=plan)


def test_authority_source_shapes_reject_coherently_rehashed_probe_seed_drift() -> None:
    plan, anchor, evidence = _committed_authority_sources()
    rows = evidence["evidence_artifacts"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    key_body = dict(rows[0]["evidence_key"])
    seeds = list(key_body["probe_seeds"])
    seeds[0] += 1
    key_body["probe_seeds"] = seeds
    _coherently_replace_first_evidence_key(evidence, key_body)
    with pytest.raises(Phase3ModelAuthorityError, match="acquisition authority"):
        _validate_phase3_authority_source_shapes(anchor, evidence, plan=plan)


def test_authority_publication_is_exclusive(tmp_path) -> None:
    value = _authority()
    target = tmp_path / "authority.json"
    write_phase3_model_authority(target, value)
    assert target.read_bytes() == canonical_phase3_model_authority_bytes(value)
    with pytest.raises(Phase3ModelAuthorityError):
        write_phase3_model_authority(target, value)


@pytest.mark.parametrize(
    "update",
    [
        {"final": True},
        {"artifact_store_id": ".."},
        {
            "allowed_cost_accounting": Phase3ModelAuthorityCost(
                setup=PhaseAccounting(calls=1),
                training_probes=PhaseAccounting(),
                reference_replay=PhaseAccounting(),
                training=PhaseAccounting(optimizer_steps=1, forward_passes=1),
                serialization=PhaseAccounting(calls=EXPECTED_MODELS),
            )
        },
    ],
)
def test_authority_validator_rejects_scope_store_hash_and_cost_drift(update) -> None:
    value = _authority()
    with pytest.raises(ValueError):
        Phase3ModelArtifactAuthority.model_validate(_rehash(value, **update))


def test_authority_validator_rejects_stale_self_hash() -> None:
    value = _authority()
    body = value.model_dump(mode="json")
    body["authority_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="self-hash"):
        Phase3ModelArtifactAuthority.model_validate(body)


def test_authority_validator_rejects_owner_row_mismatch() -> None:
    value = _authority()
    rows = list(value.models)
    rows[0] = rows[0].model_copy(update={"owner_id": "f" * 64})
    with pytest.raises(ValueError):
        Phase3ModelArtifactAuthority.model_validate(_rehash(value, models=tuple(rows)))


def test_authority_validator_rejects_duplicate_manifest_identity() -> None:
    value = _authority()
    rows = list(value.models)
    rows[1] = rows[1].model_copy(update={"manifest_sha256": rows[0].manifest_sha256})
    with pytest.raises(ValueError, match="artifact identities"):
        Phase3ModelArtifactAuthority.model_validate(_rehash(value, models=tuple(rows)))


def test_authority_validator_rejects_zero_git_provenance() -> None:
    value = _authority()
    with pytest.raises(ValueError, match="nonzero git provenance"):
        Phase3ModelArtifactAuthority.model_validate(
            _rehash(value, generation_git_commit_sha="0" * 40)
        )
