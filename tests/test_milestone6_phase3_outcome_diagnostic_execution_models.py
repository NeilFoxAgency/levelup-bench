"""Tests for descriptor-pinned outcome model execution.

The preparation model store is intentionally ignored and is not available in
CI.  This fixture rebuilds its complete 240-owner authority in a temporary
store, using canonical protocol/plan inputs and deterministic zero weights.
"""

from __future__ import annotations

import hashlib
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from levelup.experiments import milestone6_phase3_outcome_diagnostic_execution_models as execution
from levelup.experiments import milestone6_phase3_outcome_diagnostic_model_artifacts as artifacts
from levelup.experiments import milestone6_phase3_outcome_diagnostic_model_store as model_store
from levelup.experiments import milestone6_phase3_outcome_diagnostic_readiness as readiness
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
    OutcomeDiagnosticModelArtifactAuthority,
    OutcomeDiagnosticModelArtifactKey,
    OutcomeDiagnosticModelArtifactRecord,
    OutcomeStateTensorPayload,
    PinnedOutcomeModelState,
    canonical_outcome_model_artifact_authority_bytes,
    load_outcome_model_artifact_authority_bytes,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    bind_validated_outcome_diagnostic_plan,
    build_outcome_group_diagnostic_plan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    load_outcome_group_diagnostic_protocol,
)
from levelup.experiments.runner.config import canonical_json_bytes

REPOSITORY = Path(__file__).parents[1]
AUTHORITY_PATH = REPOSITORY / "configs/milestone6/phase3_outcome_model_artifact_authority.json"


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _zero_state() -> PinnedOutcomeModelState:
    tensors = []
    for name, shape, _dtype in artifacts.STATE_SCHEMA:
        count = 1
        for dimension in shape:
            count *= dimension
        tensors.append(OutcomeStateTensorPayload(name, shape, bytes(4 * count)))
    return PinnedOutcomeModelState(tuple(tensors))


def _rehashed_key(
    owner,
    view,
    evidence,
    state,
    protocol,
    plan,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
) -> OutcomeDiagnosticModelArtifactKey:
    _schema, state_sha = artifacts.inspect_outcome_model_state(state)
    consumers = tuple(unit for unit in plan.units if unit.model_owner_id == owner.owner_id)
    body = {
        "schema_version": artifacts.MODEL_SCHEMA_VERSION,
        "key_id": "0" * 64,
        "plan_id": plan.plan_id,
        "plan_parent_commit_sha": plan.parent_commit_sha,
        "protocol_sha256": protocol.sha256,
        "protocol_self_sha256": protocol.payload["diagnostic_protocol_sha256"],
        "protocol_file_sha256": protocol.sha256,
        "condition_id": owner.condition_id,
        "view_id": owner.view_id,
        "owner_id": owner.owner_id,
        "heldout_family": owner.heldout_family,
        "fold_id": owner.fold_id,
        "replicate": owner.replicate,
        "training_tuple_id": owner.training_tuple_id,
        "model_seed": owner.model_seed,
        "data_order_seed": view.data_order_seed,
        "consumer_unit_ids_sha256": _digest([unit.unit_id for unit in consumers]),
        "consumer_seed_lineage_sha256": _digest(
            [
                {
                    "unit_id": unit.unit_id,
                    "tuple_id": unit.tuple_id,
                    "task_id": unit.task_id,
                    "task_index": unit.task_index,
                    "model_seed": unit.model_seed,
                    "environment_seed": unit.environment_seed,
                    "probe_seed": unit.probe_seed,
                    "search_seed": unit.search_seed,
                    "data_order_seed": unit.data_order_seed,
                }
                for unit in consumers
            ]
        ),
        "consumer_count": 24,
        "candidate_episodes_per_task": 150,
        "adaptation_actions_per_task": 2048,
        "probe_actions_per_task": 64,
        "maximum_actions_per_candidate_episode": 64,
        "evidence_row_sha256": evidence.evidence_row_sha256,
        "evidence_payload_sha256": evidence.evidence_payload_sha256,
        "evidence_payload_bytes": evidence.evidence_payload_bytes,
        "ordered_training_task_ids": list(view.training_task_ids),
        "learning_rate": owner.learning_rate,
        "training_epochs": owner.training_epochs,
        "optimizer_id": "adam",
        "weight_decay": 0.0001,
        "device": "cpu",
        "device_portable": True,
        "torch_threads": 1,
        "processes": 1,
        "feature_mask_sha256": owner.feature_mask_sha256,
        "transformation_sha256": owner.transformation_sha256,
        "representation_sha256": view.representation_sha256,
        "model_identity_sha256": owner.model_identity_sha256,
        "architecture_id": artifacts.ARCHITECTURE_ID,
        "input_width": artifacts.INPUT_WIDTH,
        "trainable_parameters": artifacts.EXPECTED_PARAMETER_COUNT,
        "state_schema": [item.model_dump(mode="json") for item in _schema],
        "model_state_sha256": state_sha,
        "training_accounting": {
            "optimizer_steps": owner.training_epochs,
            "forward_passes": owner.training_epochs * 11,
            "training_examples": 11,
            "serialization_calls": 1,
        },
        "preparation_git_commit_sha": preparation_git_commit_sha,
        "preparation_provenance_sha256": preparation_provenance_sha256,
    }
    # The two preparation identities are supplied by the canonical authority
    # below; retaining them here avoids any dependence on ignored records.
    body["key_id"] = _digest({key: value for key, value in body.items() if key != "key_id"})
    return OutcomeDiagnosticModelArtifactKey.model_validate(body)


def test_frozen_readiness_protocol_is_canonically_equivalent() -> None:
    protocol = load_outcome_group_diagnostic_protocol(repository=REPOSITORY)
    frozen = readiness._freeze_protocol(protocol)
    fresh = artifacts._require_snapshot(frozen)
    assert fresh.content == protocol.content
    assert fresh.sha256 == protocol.sha256


@pytest.fixture(scope="module")
def model_readiness(tmp_path_factory) -> readiness.OutcomeDiagnosticModelReadinessSnapshot:
    protocol = load_outcome_group_diagnostic_protocol(repository=REPOSITORY)
    validated = bind_validated_outcome_diagnostic_plan(
        build_outcome_group_diagnostic_plan(protocol), snapshot=protocol
    )
    plan = validated.plan
    source_authority = load_outcome_model_artifact_authority_bytes(AUTHORITY_PATH.read_bytes())
    state = _zero_state()
    _schema, state_sha = artifacts.inspect_outcome_model_state(state)
    evidence_by_key = {
        (row.heldout_family, row.replicate): row for row in source_authority.evidence
    }
    views = {view.view_id: view for view in plan.views}
    records: list[OutcomeDiagnosticModelArtifactRecord] = []
    rows = []
    for owner in plan.model_owners:
        source_row = next(row for row in source_authority.artifacts if row.owner_id == owner.owner_id)
        key = _rehashed_key(
            owner,
            views[owner.view_id],
            evidence_by_key[(owner.heldout_family, owner.replicate)],
            state,
            protocol,
            plan,
            source_authority.preparation_git_commit_sha,
            source_authority.preparation_provenance_sha256,
        )
        record_body = {
            "schema_version": artifacts.MODEL_SCHEMA_VERSION,
            "record_id": "0" * 64,
            "key": key.model_dump(mode="json"),
        }
        record_body["record_id"] = _digest(
            {item: value for item, value in record_body.items() if item != "record_id"}
        )
        record = OutcomeDiagnosticModelArtifactRecord.model_validate(record_body)
        records.append(record)
        rows.append(
            source_row.model_copy(
                update={
                    "record_id": record.record_id,
                    "key_id": record.key.key_id,
                    "model_state_sha256": state_sha,
                }
            )
        )

    authority_body = source_authority.model_dump(mode="json")
    authority_body["artifacts"] = [
        row.model_dump(mode="json") for row in sorted(rows, key=lambda item: item.owner_id)
    ]
    authority_body["authority_sha256"] = _digest(
        {item: value for item, value in authority_body.items() if item != "authority_sha256"}
    )
    authority = OutcomeDiagnosticModelArtifactAuthority.model_validate(authority_body)

    store_root = tmp_path_factory.mktemp("outcome-execution-model-store")
    with model_store.open_outcome_model_store(store_root) as pinned:
        for record in records:
            model_store.write_outcome_model_artifact(
                store_root, record, state, pinned_output=pinned
            )
        for name in model_store.ROOT_METADATA_FILES:
            model_store._write_new(pinned.reader.root_fd, name, b"{}\n", pinned.reader.staging_fd)

    owner_ids = tuple(sorted(record.key.owner_id for record in records))
    stack = ExitStack()
    stack.__enter__()
    store = stack.enter_context(model_store.open_existing_outcome_model_store(store_root))
    identities = model_store.snapshot_outcome_model_store_identities_at(store, owner_ids)
    lease = readiness.OutcomeDiagnosticModelReadinessLease(
        store, stack, owner_ids, identities, _token=readiness._LEASE_TOKEN
    )
    cache = execution.build_outcome_diagnostic_execution_authority_cache(
        authority, validated, lease, protocol_snapshot=protocol
    )
    snapshot = readiness.OutcomeDiagnosticModelReadinessSnapshot(
        SimpleNamespace(protocol=protocol),
        authority,
        SimpleNamespace(content=canonical_outcome_model_artifact_authority_bytes(authority)),
        store_root,
        owner_ids,
        lease,
        cache,
        _token=readiness._SNAPSHOT_TOKEN,
    )
    try:
        yield snapshot
    finally:
        snapshot.close()


def test_cache_covers_exact_owner_and_unit_universe(model_readiness) -> None:
    cache = model_readiness.execution_authority_cache
    assert isinstance(cache, execution.OutcomeDiagnosticExecutionAuthorityCache)
    assert len(cache._artifacts_by_owner_id) == 240
    assert len(cache._artifacts_by_unit_id) == 5_760
    first = cache.artifact_for_unit(cache.validated_plan.plan.units[0])
    assert first.lineage.owner_id == first.record.key.owner_id
    assert first.lineage.record_id == first.record.record_id
    assert first.lineage.key_id == first.record.key.key_id
    assert first.lineage.model_state_sha256 == first.record.key.model_state_sha256
    assert first.lineage.training_accounting == first.record.key.training_accounting


def test_loader_reconstructs_exact_frozen_model_and_lineage(model_readiness) -> None:
    planned = model_readiness.execution_authority_cache.validated_plan.plan.units[0]
    with execution.load_authorized_outcome_model_from_pinned_store(
        model_readiness, planned
    ) as loaded:
        assert loaded.require_active() is loaded
        assert type(loaded.model).__name__ == "StateConditionedScorer"
        assert sum(parameter.numel() for parameter in loaded.model.parameters()) == 3_841
        assert loaded.model.training is False
        assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
        expected = model_readiness.execution_authority_cache.artifact_for_unit(planned)
        assert loaded.owner_id == expected.lineage.owner_id
        assert loaded.record_id == expected.lineage.record_id
        assert loaded.key_id == expected.lineage.key_id
        assert loaded.model_state_sha256 == expected.lineage.model_state_sha256
        assert loaded.training_accounting == expected.lineage.training_accounting
    assert not loaded._active
    with pytest.raises(execution.OutcomeDiagnosticExecutionModelError, match="no longer active"):
        loaded.require_active()


def test_loader_rejects_forged_unit(model_readiness) -> None:
    planned = model_readiness.execution_authority_cache.validated_plan.plan.units[0]
    with pytest.raises(execution.OutcomeDiagnosticExecutionModelError, match="differs"):
        with execution.load_authorized_outcome_model_from_pinned_store(
            model_readiness, replace(planned, task_id="foreign-task")
        ):
            pass


def test_loader_rejects_descriptor_payload_substitution(model_readiness, monkeypatch) -> None:
    planned = model_readiness.execution_authority_cache.validated_plan.plan.units[0]
    expected = model_readiness.execution_authority_cache.artifact_for_unit(planned)
    original = execution.load_outcome_model_artifact_payload_at

    def substituted(reader, owner_id):
        record, index, state = original(reader, owner_id)
        return record, index, replace(state, tensors=state.tensors[:-1])

    monkeypatch.setattr(execution, "load_outcome_model_artifact_payload_at", substituted)
    with pytest.raises(execution.OutcomeDiagnosticExecutionModelError, match="differs"):
        with execution.load_authorized_outcome_model_from_pinned_store(model_readiness, planned):
            pass
    assert expected.lineage.model_state_sha256 == expected.record.key.model_state_sha256


def test_forged_cache_and_snapshot_are_rejected() -> None:
    with pytest.raises(execution.OutcomeDiagnosticExecutionModelError, match="snapshot"):
        execution.build_outcome_diagnostic_execution_authority_cache(SimpleNamespace())
    with pytest.raises(execution.OutcomeDiagnosticExecutionModelError, match="snapshot"):
        with execution.load_authorized_outcome_model_from_pinned_store(
            SimpleNamespace(), SimpleNamespace()
        ):
            pass


def test_loader_rejects_inactive_lease(model_readiness) -> None:
    planned = model_readiness.execution_authority_cache.validated_plan.plan.units[0]
    model_readiness.close()
    with pytest.raises(execution.OutcomeDiagnosticExecutionModelError, match="active"):
        with execution.load_authorized_outcome_model_from_pinned_store(model_readiness, planned):
            pass
