"""Focused tests for the descriptor-pinned outcome model execution boundary."""

from __future__ import annotations

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
    load_outcome_model_artifact_authority_bytes,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    bind_validated_outcome_diagnostic_plan,
    build_outcome_group_diagnostic_plan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    load_outcome_group_diagnostic_protocol,
)

REPOSITORY = Path(__file__).parents[1]
AUTHORITY_PATH = REPOSITORY / "configs/milestone6/phase3_outcome_model_artifact_authority.json"


def test_frozen_readiness_protocol_is_canonically_equivalent() -> None:
    protocol = load_outcome_group_diagnostic_protocol(repository=REPOSITORY)
    frozen = readiness._freeze_protocol(protocol)
    fresh = artifacts._require_snapshot(frozen)
    assert fresh.content == protocol.content
    assert fresh.sha256 == protocol.sha256


@pytest.fixture(scope="module")
def model_readiness() -> readiness.OutcomeDiagnosticModelReadinessSnapshot:
    protocol = load_outcome_group_diagnostic_protocol(repository=REPOSITORY)
    plan = bind_validated_outcome_diagnostic_plan(
        build_outcome_group_diagnostic_plan(protocol), snapshot=protocol
    )
    authority = load_outcome_model_artifact_authority_bytes(AUTHORITY_PATH.read_bytes())
    owner_ids = tuple(sorted(row.owner_id for row in authority.artifacts))
    store_root = REPOSITORY / "runs/milestone6" / authority.artifact_store_id
    stack = ExitStack()
    stack.__enter__()
    store = stack.enter_context(model_store.open_existing_outcome_model_store(store_root))
    identities = model_store.snapshot_outcome_model_store_identities_at(store, owner_ids)
    lease = readiness.OutcomeDiagnosticModelReadinessLease(
        store, stack, owner_ids, identities, _token=readiness._LEASE_TOKEN
    )
    cache = execution.build_outcome_diagnostic_execution_authority_cache(
        authority, plan, lease, protocol_snapshot=protocol
    )
    snapshot = readiness.OutcomeDiagnosticModelReadinessSnapshot(
        SimpleNamespace(protocol=protocol),
        authority,
        SimpleNamespace(content=AUTHORITY_PATH.read_bytes()),
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
    forged = replace(planned, task_id="foreign-task")
    with pytest.raises(execution.OutcomeDiagnosticExecutionModelError, match="differs"):
        with execution.load_authorized_outcome_model_from_pinned_store(
            model_readiness, forged
        ):
            pass


def test_loader_rejects_descriptor_payload_substitution(model_readiness, monkeypatch) -> None:
    # Reopen a fresh cache/lease is unnecessary: this test only substitutes the
    # descriptor-read tuple and verifies it cannot bypass the cached identities.
    cache = model_readiness.execution_authority_cache
    planned = cache.validated_plan.plan.units[0]
    expected = cache.artifact_for_unit(planned)
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
