"""Regression checks for the frozen development-only local-affordance design."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/milestone6/phase3_local_affordance_protocol.json"
FAMILIES = ["plain", "battery", "cooldown", "heat", "momentum", "combo"]
CONDITIONS = [
    "B2-global-listwise-optimum",
    "S-state-availability-listwise-optimum",
    "P-state-availability-alias-pooled-outcome-listwise-optimum",
    "L-state-availability-local-outcome-listwise-optimum",
]


def _load() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_bytes())


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_protocol_binds_exact_development_authority_and_no_final_scope() -> None:
    body = _load()
    assert body["schema_version"] == "milestone6.phase3.local-affordance-protocol.v2"
    assert body["status"] == "frozen-design-only"
    assert body["scope"] == "known-development-only"
    assert body["execution"] is False
    expected = {
        "development_protocol": "configs/milestone6/development_protocol.json",
        "development_tasks": "configs/milestone6/development_tasks.json",
        "phase2_selection_lock": "configs/milestone6/phase2_screening_selection.json",
        "phase3_representation_ladder": "configs/milestone6/phase3_representation_ladder.json",
        "phase3_plan_lock": "configs/milestone6/phase3_plan_lock.json",
        "phase3_outcome_group_diagnostic_result": "configs/milestone6/phase3_outcome_group_diagnostic_result.json",
    }
    assert set(body["authority"]) == set(expected)
    for key, path in expected.items():
        assert body["authority"][key] == {"path": path, "sha256": _sha256(path)}
    boundary = body["execution_boundary"]
    assert boundary["final_family_access"] is False
    assert boundary["comparative_result_inspection"] is False
    assert boundary["execution_authorized"] is False
    assert boundary["no_final_paths_or_consumers"] is True


def test_raw_probe_authority_is_new_immutable_and_learner_safe() -> None:
    raw = _load()["raw_probe_evidence_authority"]
    assert raw["schema_version"] == "milestone6.phase3.raw-probe-evidence-authority.v2"
    task_raw = raw["task_raw_artifacts"]
    assert task_raw["count"] == 240
    assert task_raw["rows_per_artifact"] == 64
    assert "one task raw artifact only" in task_raw["reducer_population"]
    assert "0..63 occur exactly once" in task_raw["probe_index_uniqueness"]
    training = raw["training_evidence_manifests"]
    assert training["count"] == 30
    assert training["task_references_per_manifest"] == 40
    assert training["raw_rows_referenced_per_manifest"] == 40 * 64
    assert "byte-identical" in training["pooled_table_parity"]
    heldout = raw["heldout_probe_bindings"]
    assert heldout["count"] == 240
    assert heldout["task_references_per_binding"] == 1
    policy = raw["probe_policy"]
    assert policy["probe_actions_per_task"] == 64
    assert policy["target_samples_per_alias"] == 8
    assert policy["probe_actions_per_attempt"] == 16
    assert policy["all_conditions_same_rows"] is True
    assert raw["immutability"]["regeneration"] == "forbidden after authority freeze"
    assert "40 task artifacts" in raw["immutability"]["fold_view_isolation"]
    assert "cannot enumerate or read" in raw["immutability"]["heldout_view_isolation"]
    assert "fail closed" in raw["immutability"]["readiness_check"]
    costs = raw["cost_ownership"]
    assert costs["physical_raw_probe_actions"] == 240 * 64 == 15360
    assert costs["whole_matrix_consumer_equivalent_probe_actions"] == 11520 * 64
    assert "never sum" in costs["reporting_rule"]
    visibility = raw["learner_visibility"]
    assert visibility["storage_row_payload"] == (
        "ObservedTransition plus canonical probe_index only"
    )
    assert "never serialize probe_index" in visibility["probe_index_use"]
    assert visibility["metadata_ids"] == "learner-invisible"
    for forbidden in ("pair_id", "alignment_id", "shared_record_key", "raw_artifact_digest"):
        assert forbidden in visibility["forbidden_fields"]


def test_matrix_conditions_capacity_and_full_grid_are_matched() -> None:
    body = _load()
    matrix = body["development_matrix"]
    assert matrix["family_order"] == FAMILIES
    assert matrix["replicates"] == [0, 1, 2, 3, 4]
    assert matrix["units"] == 4 * 12 * 6 * 5 * 8 == 11520
    assert matrix["model_owners"] == 480
    conditions = body["conditions"]
    assert [condition["condition_id"] for condition in conditions] == CONDITIONS
    assert conditions[0]["trainable_parameters"] == 3601
    assert all(condition["trainable_parameters"] == 3841 for condition in conditions[1:])
    assert "never restrict B2" in conditions[0]["capacity_policy"]
    assert "canonical S mask" in conditions[1]["bitwise_parity_requirement"]
    assert "canonical full T transform" in conditions[2]["bitwise_parity_requirement"]
    capacity = body["capacity_matching"]
    assert capacity["symmetric_parameter_tolerance_fraction"] == 0.1
    assert capacity["formula"] == "abs(left-right) / max(left,right)"
    assert capacity["counts"] == {"B2": 3601, "S": 3841, "P": 3841, "L": 3841}
    assert capacity["fixed_reducer_parameter_count"] == 0
    owners = body["model_owner_identity"]
    assert owners["search_temperature_in_identity"] is False
    assert owners["training_tuples_per_condition"] == 4
    assert owners["consumers_per_owner"] == 3 * 8 == 24
    assert "without retraining" in owners["temperature_reuse"]
    assert "30 unique model-owner reports" in owners["selected_tuple_cost_scope"]
    assert len(body["candidate_tuples"]) == 12
    assert body["shared_training_and_search"]["objective"] == (
        "optimum-only listwise next-action imitation"
    )
    assert body["shared_training_and_search"]["probe_actions_per_task"] == 64
    assert body["shared_training_and_search"]["adaptation_actions_per_task"] == 2048


def test_local_transform_and_pooled_control_are_byte_preserving() -> None:
    transform = _load()["local_affordance_transform"]
    assert transform["distance_coordinates"] == [
        "progress",
        "remaining",
        "resource",
        "pressure",
    ]
    assert transform["distance_definition"].startswith("squared Euclidean")
    assert "one task raw artifact" in transform["row_order"]
    assert "unique canonical probe_index" in transform["row_order"]
    assert transform["k"] == 4
    assert transform["k_selectable"] is False
    assert transform["k_eff"] == "min(4, n) where n is the same-alias row count"
    assert transform["statistics"]["std"] == "biased population standard deviation (unbiased=false)"
    assert transform["replace_only_indices"] == [4, 5, 6, 7, 8, 9, 10]
    assert transform["outcome_index_order"] == [
        "progress_delta",
        "elapsed_delta",
        "resource_delta",
        "pressure_delta",
        "after_resource",
        "after_pressure",
        "completed",
    ]
    assert transform["preserve_byte_identity_indices"] == [0, 1, 2, 3, 11]
    assert transform["coverage_index"] == 48
    assert transform["coverage_byte_identity"] is True
    assert "zero/neutral" in transform["unknown_alias"]
    assert "no radius" in transform["preprocessing"]
    assert "preserving every row and count" in transform["pooled_control"]


def test_selection_claim_gate_and_advancement_are_frozen() -> None:
    body = _load()
    selection = body["selection_rule"]
    assert selection["primary_metric"] == "minimum_family_exact_optimum_success_rate"
    assert selection["success_tolerance_absolute"] == 0.05
    assert selection["failure_sentinel"] == 2049
    assert selection["eligible_hyperparameters"] == [
        "learning_rate",
        "training_epochs",
        "search_temperature",
    ]
    assert "inherited exactly from the frozen Phase 3 selector" in selection["tie_breaking"]
    assert selection["ineligible_hyperparameters"]
    gate = body["alignment_claim_gate"]
    assert gate["aggregate_minimum_fraction"] == 0.8
    assert gate["per_family_minimum_fraction"] == 0.5
    assert gate["apply_separately"] == ["training", "heldout"]
    assert "n > 4" in gate["eligible_row"]
    queries = body["diagnostics"]["query_populations"]
    assert "optimum reference-decision pre-state" in queries["training"]
    assert "240 fixed heldout 64-action probe bindings" in queries["heldout"]
    assert "candidate-search trajectories" in queries["forbidden"]
    claims = body["claim_and_advancement_rule"]
    assert "strictly greater than 0.05" in claims["conditional_transition_claim"]
    assert "strictly greater than 0.05" in claims["local_alignment_claim"]
    assert "both" in claims["full_rung_claim"]
    assert "at or below 0.05" in claims["within_tolerance"]
    assert set(claims["forbidden_claims"]) >= {
        "pairing",
        "history",
        "sequence",
        "final-family access",
    }


def test_protocol_is_pretty_json_and_records_no_outcomes() -> None:
    raw = CONFIG_PATH.read_bytes()
    assert b"\n" in raw
    body = _load()
    assert body["freeze_record"]["comparative_results_inspected_before_freeze"] is False
    assert body["diagnostics"]["status"] == "pre-outcome and non-selection only"
    assert body["execution_boundary"]["comparative_result_inspection"] is False
