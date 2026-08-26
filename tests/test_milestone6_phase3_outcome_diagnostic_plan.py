from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType

import pytest

from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    CONDITIONS,
    EXPECTED_TUPLES,
    FAMILIES,
    PARENT_COMMIT_SHA,
    OutcomeDiagnosticPlanError,
    _require_canonical_snapshot,
    bind_validated_outcome_diagnostic_plan,
    build_outcome_group_diagnostic_plan,
    canonical_outcome_plan_bytes,
    feature_mask_sha256,
    outcome_plan_id,
    transformation_sha256,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    load_outcome_group_diagnostic_protocol,
)


def _freeze_payload(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_payload(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_payload(item) for item in value)
    return value


def test_outcome_plan_exact_matrix_and_scope() -> None:
    plan = build_outcome_group_diagnostic_plan()
    assert plan.parent_commit_sha == PARENT_COMMIT_SHA
    assert plan.final_family_access is False
    assert plan.family_order == FAMILIES
    assert plan.replicates == (0, 1, 2, 3, 4)
    assert plan.condition_ids == CONDITIONS
    assert plan.candidate_tuple_ids == EXPECTED_TUPLES
    assert len(plan.evidence_lineage_rows) == 30
    assert len(plan.views) == 60
    assert len(plan.model_owners) == 240
    assert len(plan.units) == 5_760
    assert all(not unit.final_family_access for unit in plan.units)


def test_outcome_plan_owner_and_unit_fanout_is_exact() -> None:
    plan = build_outcome_group_diagnostic_plan()
    assert all(len(owner.search_temperature_ids) == 3 for owner in plan.model_owners)
    assert all(owner.trainable_parameters == 3_841 for owner in plan.model_owners)
    assert all(
        sum(unit.model_owner_id == owner.owner_id for unit in plan.units) == 24
        for owner in plan.model_owners
    )
    assert all(
        sum(unit.view_id == view.view_id for unit in plan.units) == 96 for view in plan.views
    )
    assert all(
        sum(unit.heldout_family == family for unit in plan.units) == 960 for family in FAMILIES
    )


def test_outcome_masks_and_transformations_are_deterministic() -> None:
    snapshot = load_outcome_group_diagnostic_protocol()
    first = tuple(
        (
            condition,
            feature_mask_sha256(snapshot, condition),
            transformation_sha256(snapshot, condition),
        )
        for condition in CONDITIONS
    )
    second = tuple(
        (
            condition,
            feature_mask_sha256(snapshot, condition),
            transformation_sha256(snapshot, condition),
        )
        for condition in CONDITIONS
    )
    assert first == second
    assert len({mask for _, mask, _ in first}) == 2
    assert len({transform for _, _, transform in first}) == 2


def test_outcome_plan_canonical_bytes_are_self_hashed() -> None:
    plan = build_outcome_group_diagnostic_plan()
    snapshot = load_outcome_group_diagnostic_protocol()
    content = canonical_outcome_plan_bytes(plan, snapshot=snapshot)
    assert json.loads(content)["plan_id"] == plan.plan_id
    assert outcome_plan_id(plan) == plan.plan_id
    assert canonical_outcome_plan_bytes(plan, snapshot=snapshot) == canonical_outcome_plan_bytes(
        plan, snapshot=snapshot
    )


def test_outcome_plan_validation_rejects_identity_mutation() -> None:
    plan = build_outcome_group_diagnostic_plan()
    snapshot = load_outcome_group_diagnostic_protocol()
    changed = replace(plan, plan_id="0" * 64)
    with pytest.raises(OutcomeDiagnosticPlanError, match="self-hash"):
        canonical_outcome_plan_bytes(changed, snapshot=snapshot)
    authority = bind_validated_outcome_diagnostic_plan(plan, snapshot=snapshot)
    authority.require_unit(plan.units[0])
    with pytest.raises(OutcomeDiagnosticPlanError, match="differs"):
        authority.require_unit(replace(plan.units[0], unit_id="0" * 64))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: replace(
            plan,
            units=(replace(plan.units[0], environment_seed=plan.units[0].environment_seed + 1),)
            + plan.units[1:],
        ),
        lambda plan: replace(
            plan,
            views=(replace(plan.views[0], heldout_family="combo"),) + plan.views[1:],
        ),
        lambda plan: replace(
            plan,
            model_owners=(replace(plan.model_owners[0], learning_rate=0.004),)
            + plan.model_owners[1:],
        ),
        lambda plan: replace(
            plan,
            units=(replace(plan.units[0], probe_actions_per_task=65),) + plan.units[1:],
        ),
        lambda plan: replace(
            plan,
            evidence_lineage_rows=(b"{}",) + plan.evidence_lineage_rows[1:],
        ),
        lambda plan: replace(
            plan,
            authority_hashes=((plan.authority_hashes[0][0], "0" * 64),) + plan.authority_hashes[1:],
        ),
        lambda plan: replace(plan, final_family_access=True),
    ],
)
def test_recomputed_self_hash_mutations_fail_closed(mutation) -> None:
    plan = build_outcome_group_diagnostic_plan()
    snapshot = load_outcome_group_diagnostic_protocol()
    mutated = mutation(plan)
    mutated = replace(mutated, plan_id=outcome_plan_id(mutated))
    with pytest.raises(OutcomeDiagnosticPlanError):
        bind_validated_outcome_diagnostic_plan(mutated, snapshot=snapshot)


def test_forged_snapshot_and_external_same_basename_fail_closed(tmp_path) -> None:
    snapshot = load_outcome_group_diagnostic_protocol()
    forged_payload = deepcopy(snapshot.payload)
    forged_payload["conditions"][0]["trainable_parameters"] = 1
    forged = replace(snapshot, payload=forged_payload)
    plan = build_outcome_group_diagnostic_plan()
    with pytest.raises(OutcomeDiagnosticPlanError, match="snapshot"):
        bind_validated_outcome_diagnostic_plan(plan, snapshot=forged)
    external = tmp_path / snapshot.path.name
    external.write_bytes(snapshot.content)
    external_snapshot = replace(snapshot, path=external)
    with pytest.raises(OutcomeDiagnosticPlanError, match="snapshot"):
        bind_validated_outcome_diagnostic_plan(plan, snapshot=external_snapshot)


def test_frozen_snapshot_payload_is_logically_equivalent() -> None:
    snapshot = load_outcome_group_diagnostic_protocol()
    frozen = replace(snapshot, payload=_freeze_payload(snapshot.payload))
    assert _require_canonical_snapshot(frozen) == snapshot


def test_frozen_snapshot_payload_drift_fails_closed() -> None:
    snapshot = load_outcome_group_diagnostic_protocol()
    changed = deepcopy(snapshot.payload)
    changed["conditions"][0]["trainable_parameters"] = 1
    forged = replace(snapshot, payload=_freeze_payload(changed))
    with pytest.raises(OutcomeDiagnosticPlanError, match="snapshot"):
        _require_canonical_snapshot(forged)
