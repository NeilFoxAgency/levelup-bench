from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from levelup.experiments.milestone6_phase3_protocol import load_phase3_protocol

ROOT = Path(__file__).parents[1]
PROTOCOL_PATH = ROOT / "configs/milestone6/phase3_representation_ladder.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase3_protocol_binds_frozen_development_authority() -> None:
    protocol = _load(PROTOCOL_PATH)
    assert protocol["schema_version"] == "milestone6.phase3_representation_ladder.v1"
    assert protocol["status"] == "frozen-before-phase3-comparative-development-results"
    assert protocol["scope"] == "known-development-only"
    assert protocol["freeze_record"]["phase2_results_used_for_diagnosis"] is True
    assert protocol["freeze_record"]["phase3_comparative_results_inspected"] is False
    assert protocol["final_family_access"] is False

    for key in (
        "development_protocol",
        "development_tasks",
        "phase2_candidates",
        "phase2_selection_lock",
    ):
        authority = protocol["authority"][key]
        assert authority["sha256"] == _sha256(ROOT / authority["path"])

    selection = _load(ROOT / protocol["authority"]["phase2_selection_lock"]["path"])
    assert protocol["authority"]["phase2_selection_lock"]["analysis_sha256"] == (
        selection["analysis"]["analysis_sha256"]
    )
    assert selection["scientific_boundary"]["development_only"] is True
    assert selection["scientific_boundary"]["final_family_access"] is False


def test_phase3_conditions_operationalize_separate_ladder_questions() -> None:
    protocol = _load(PROTOCOL_PATH)
    conditions = {row["condition_id"]: row for row in protocol["conditions"]}
    assert tuple(conditions) == (
        "B2-global-listwise-optimum",
        "T-markov-state-transition-listwise-optimum",
        "S-state-availability-listwise-optimum",
        "H0-null-history-transition-listwise-optimum",
        "H4-causal-history-transition-listwise-optimum",
        "H4-shuffled-history-transition-listwise-optimum",
    )
    assert conditions["B2-global-listwise-optimum"]["role"] == (
        "strong_optimum_imitation_anchor"
    )
    assert conditions["T-markov-state-transition-listwise-optimum"][
        "historical_condition_id"
    ] == "C-state-conditioned-listwise-optimum"

    state_control = conditions["S-state-availability-listwise-optimum"]
    assert state_control["role"] == "transition_outcome_destroyed_control"
    representation = state_control["representation"]
    assert representation["retained_indices_per_summary_block"] == [0, 1, 2, 3, 11]
    assert representation["zeroed_indices_per_summary_block"] == [4, 5, 6, 7, 8, 9, 10]
    assert representation["coverage_index"] == 48
    assert "sampling-support" in representation["coverage_treatment"]
    assert state_control["representation"]["input_width"] == 54

    null_history = conditions["H0-null-history-transition-listwise-optimum"]
    ordered = conditions["H4-causal-history-transition-listwise-optimum"]
    shuffled = conditions["H4-shuffled-history-transition-listwise-optimum"]
    assert null_history["role"] == "architecture_matched_no_history_control"
    assert null_history["representation"]["learner_visible_prior_transitions"] == 0
    assert "min(4" in null_history["representation"]["history"]
    assert "never carry a persistent hidden state" in null_history["representation"][
        "reset"
    ]
    assert null_history["architecture"] == ordered["architecture"]
    assert ordered["architecture"]["history_length"] == 4
    assert ordered["representation"]["search_reset"].startswith(
        "each candidate episode starts"
    )
    assert "never carry a persistent hidden state" in ordered["representation"][
        "decision_evaluation"
    ]
    assert shuffled["architecture"] == "identical_to_H4"
    assert shuffled["control_semantics"]["same_multiset"] is True
    assert shuffled["control_semantics"]["future_transitions"] is False
    assert "derangement" in shuffled["control_semantics"]["permutation"]
    assert "0.80" in shuffled["control_semantics"]["sequence_claim_coverage_gate"]
    assert shuffled["control_semantics"]["training_and_search"].endswith(
        "both training and candidate generation"
    )

    questions = protocol["scientific_questions"]
    assert questions["transition_outcomes"]["comparison"].startswith(
        "S-state-availability"
    )
    assert questions["causal_history"]["comparison"].startswith(
        "T-markov-state-transition"
    )
    assert questions["sequence_order"]["comparison"].startswith(
        "H4-shuffled-history"
    )
    assert questions["explicit_pairing"] == {
        "status": "deferred",
        "claims_before_D1_F_gate": "forbidden",
    }


def test_phase3_same_data_capacity_budget_and_selection_contract() -> None:
    protocol = _load(PROTOCOL_PATH)
    phase2 = _load(ROOT / "configs/milestone6/phase2_screening_candidates.json")
    assert protocol["candidate_tuples"] == phase2["candidate_tuples"]

    matrix = protocol["development_matrix"]
    assert matrix["family_order"] == [
        "plain",
        "battery",
        "cooldown",
        "heat",
        "momentum",
        "combo",
    ]
    assert matrix["new_units"] == 4 * 12 * 6 * 5 * 8 == 11520
    assert matrix["existing_anchor_units"] == 2 * 12 * 6 * 5 * 8 == 5760
    assert matrix["combined_comparison_units"] == 17280
    assert protocol["canonical_evidence_reuse"]["regeneration"] == "forbidden"
    assert protocol["canonical_evidence_reuse"][
        "new_training_evidence_probe_or_reference_charge"
    ] == 0
    assert "no frontier trajectory" in protocol["canonical_evidence_reuse"][
        "trajectory_access"
    ]
    anchor = protocol["canonical_evidence_reuse"]["anchor_lineage_gate"]
    assert "240 B2/C model owners" in anchor["required_anchor_manifest"]
    assert "5,760 B2/C unit result IDs" in anchor["required_anchor_manifest"]
    assert anchor["T_alias_semantics"].startswith("T is an analysis-only alias")
    assert "any mismatch forbids" in anchor["parity_checks"]

    ownership = protocol["artifact_sharing_and_cost_ownership"]
    assert ownership["search_temperature_in_model_identity"] is False
    assert ownership["new_model_owners_per_condition"] == 4 * 6 * 5
    assert ownership["new_model_owners_total"] == 4 * 4 * 6 * 5
    assert ownership["unit_local_training"] == 0
    assert "exactly 30 unique model-owner reports" in ownership[
        "selected_tuple_cost_scope"
    ]

    counts = protocol["capacity_matching"]["counts"]
    assert counts == {
        "B2": 3601,
        "S": 3841,
        "T": 3841,
        "H0": 3889,
        "H4": 3889,
        "H4_shuffled": 3889,
    }
    for left in counts.values():
        for right in counts.values():
            assert abs(left - right) / max(left, right) <= 0.1
    assert "never capacity-" in protocol["capacity_matching"]["strong_B2_floor"]

    fixed = protocol["fixed_training_and_search"]
    assert fixed["probe_actions_per_task"] == 64
    assert fixed["candidate_episodes_per_task"] == 150
    assert fixed["adaptation_actions_per_task"] == 2048
    assert fixed["maximum_actions_per_candidate_episode"] == 64
    assert fixed["exact_optimum_affects_search_control_flow"] is False
    assert fixed["oracle_timing"] == (
        "after the complete fixed candidate batch and independent replay"
    )

    selection = protocol["selection_rule"]
    assert selection["primary_metric"] == "minimum_family_exact_optimum_success_rate"
    assert selection["success_tolerance_absolute"] == 0.05
    assert selection["failure_sentinel"] == 2049
    assert selection["changes_after_phase3_results"] == "forbidden"
    assert protocol["claim_and_advancement_rule"]["robust_primary_gain"].endswith(
        "more than 0.05 absolute"
    )


def test_phase3_shuffle_seed_and_diagnostic_channels_cannot_select() -> None:
    protocol = _load(PROTOCOL_PATH)
    seeds = protocol["seed_policy"]
    assert seeds["history_shuffle_base"] == 6700000
    assert "canonical JSON" in seeds["history_shuffle_derivation"]
    assert "hash()" in seeds["history_shuffle_derivation"]
    assert "sort_keys=true" in seeds["permutation_map_canonical_bytes"]
    assert "no trailing newline" in seeds["permutation_map_canonical_bytes"]
    assert "permutation-map SHA-256" in seeds["shuffle_identity"]
    diagnostics = protocol["diagnostics"]
    assert "Heat pressure-cap decision logits" in diagnostics["non_selection_only"]
    for forbidden_change in (
        "candidate eligibility",
        "selection rule",
        "capacity rule",
        "budgets",
        "seeds",
        "history length",
        "architecture",
        "claim thresholds",
    ):
        assert forbidden_change in diagnostics["cannot_change"]


def test_typed_phase3_loader_accepts_only_complete_frozen_contract() -> None:
    snapshot = load_phase3_protocol()
    assert snapshot.path == PROTOCOL_PATH.resolve()
    assert snapshot.sha256 == _sha256(PROTOCOL_PATH)
    assert tuple(key for key, _ in snapshot.authority_bytes) == (
        "development_protocol",
        "development_tasks",
        "phase2_candidates",
        "phase2_selection_lock",
    )
    assert snapshot.payload["development_matrix"]["new_units"] == 11520


def test_typed_phase3_loader_rejects_authority_and_final_drift(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    payload = _load(PROTOCOL_PATH)
    for row in payload["authority"].values():
        if not isinstance(row, dict) or "path" not in row:
            continue
        source = ROOT / row["path"]
        target = repository / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    protocol = repository / "phase3.json"
    protocol.write_text(json.dumps(payload))
    assert load_phase3_protocol(protocol, repository=repository).payload[
        "final_family_access"
    ] is False

    payload["final_family_access"] = True
    protocol.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="final-family access"):
        load_phase3_protocol(protocol, repository=repository)

    payload["final_family_access"] = False
    protocol.write_text(json.dumps(payload))
    authority_path = repository / payload["authority"]["development_tasks"]["path"]
    authority_path.write_bytes(authority_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="development_tasks changed"):
        load_phase3_protocol(protocol, repository=repository)
