"""Regression checks for the locked development-only outcome diagnostic result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from levelup.experiments.runner.config import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/milestone6/phase3_outcome_group_diagnostic_result.json"
REPORT_PATH = ROOT / "runs/milestone6/phase3-outcome-group-diagnostic-analysis-c971334.json"
FROZEN_CONFIG_PATH = ROOT / "configs/milestone6/phase3_outcome_group_diagnostic.json"
FAMILIES = ["plain", "battery", "cooldown", "heat", "momentum", "combo"]
CONDITIONS = [
    "S-PEC-state-progress-elapsed-completion-listwise-optimum",
    "S-RP-state-resource-pressure-outcome-listwise-optimum",
]


def _load() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_bytes())


def _self_hash(body: dict[str, object]) -> str:
    unsigned = dict(body)
    unsigned.pop("result_lock_sha256")
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def test_result_lock_is_pretty_and_self_hashed() -> None:
    raw = CONFIG_PATH.read_bytes()
    assert b"\n" in raw
    body = _load()
    assert body["result_lock_sha256"] == _self_hash(body)
    assert body["result_lock_sha256"] == (
        "439d2f19f0f7d717efcb498353e402ebaa446fe06131516a7af6367b7711cddf"
    )


def test_result_lock_records_exact_source_lineage_and_coverage() -> None:
    body = _load()
    source = body["source"]
    coverage = body["coverage"]
    assert source == {
        "report_path": "runs/milestone6/phase3-outcome-group-diagnostic-analysis-c971334.json",
        "source_file_sha256": "df096491a328023774c340f5bcd381630aba2675f2df82b2d8d7a9f633782bb7",
        "analysis_sha256": "c3eb878f4e1f099443c249e98c983127199a191c16ef75738a34f2f5fbc33a91",
        "report_commit_sha": "c971334dd7a540814afc7cb0f2ea0ab8fa3088a3",
        "ci_run_id": 33179541524,
        "frozen_config_path": "configs/milestone6/phase3_outcome_group_diagnostic.json",
        "frozen_config_sha256": "dda43928b46bbf6981d50fb9d03abc5c344cfe10612d5e16cc02bd76da7646c3",
    }
    assert coverage == {
        "unit_count": 5760,
        "condition_counts": {condition: 2880 for condition in CONDITIONS},
        "family_counts": {family: 960 for family in FAMILIES},
        "units_per_family_per_condition": 480,
        "units_per_family_per_tuple": 40,
        "tuples_per_condition": 12,
    }
    assert hashlib.sha256(FROZEN_CONFIG_PATH.read_bytes()).hexdigest() == source[
        "frozen_config_sha256"
    ]
    if REPORT_PATH.exists():
        assert hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest() == source[
            "source_file_sha256"
        ]
        report = json.loads(REPORT_PATH.read_bytes())
        assert report["analysis_sha256"] == source["analysis_sha256"]
        assert report["lineage"]["git_commit_sha"] == source["report_commit_sha"]


def test_result_lock_preserves_anchor_and_condition_facts() -> None:
    body = _load()
    anchors = body["locked_anchors"]
    assert anchors["S"]["tuple_id"] == "lr0p01-e120-t1p2"
    assert anchors["S"]["minimum_family_success_rate"] == {
        "numerator": 17,
        "denominator": 40,
    }
    assert anchors["S"]["family_order"] == FAMILIES
    assert anchors["S"]["family_success_counts"] == [37, 34, 35, 17, 21, 23]
    assert anchors["B2"]["tuple_id"] == "lr0p003-e120-t1p2"
    assert anchors["T"]["tuple_id"] == "lr0p003-e120-t1p2"

    expected = {
        CONDITIONS[0]: {
            "selected_tuple_id": "lr0p003-e120-t1p2",
            "retained": ["lr0p003-e120-t0p9", "lr0p003-e120-t1p2"],
            "selected_min": {"numerator": 1, "denominator": 5},
            "selected_counts": [39, 37, 38, 8, 28, 27],
            "matched_min": {"numerator": 1, "denominator": 20},
            "matched_counts": [39, 29, 26, 2, 19, 15],
            "selected_delta": {"numerator": -9, "denominator": 40},
            "matched_delta": {"numerator": -3, "denominator": 8},
        },
        CONDITIONS[1]: {
            "selected_tuple_id": "lr0p003-e120-t0p9",
            "retained": ["lr0p003-e120-t0p9", "lr0p003-e120-t1p2"],
            "selected_min": {"numerator": 11, "denominator": 40},
            "selected_counts": [40, 38, 39, 11, 24, 26],
            "matched_min": {"numerator": 1, "denominator": 20},
            "matched_counts": [40, 28, 38, 2, 6, 26],
            "selected_delta": {"numerator": -3, "denominator": 20},
            "matched_delta": {"numerator": -3, "denominator": 8},
        },
    }
    for condition in body["conditions"]:
        facts = expected[condition["condition_id"]]
        assert condition["classification"] == "robust_harm"
        assert condition["selected"]["tuple_id"] == facts["selected_tuple_id"]
        assert condition["selected"]["retained_tuple_ids"] == facts["retained"]
        assert condition["selected"]["minimum_family_success_rate"] == facts["selected_min"]
        assert condition["selected"]["family_success_counts"] == facts["selected_counts"]
        assert condition["matched_S"]["tuple_id"] == "lr0p01-e120-t1p2"
        assert condition["matched_S"]["minimum_family_success_rate"] == facts["matched_min"]
        assert condition["matched_S"]["family_success_counts"] == facts["matched_counts"]
        assert condition["delta_vs_locked_S"] == {
            "selected": facts["selected_delta"],
            "matched_S": facts["matched_delta"],
        }


def test_result_lock_claims_and_next_rung_are_scope_safe() -> None:
    body = _load()
    assert body["total_deduplicated_cost"] == {
        "unit_count": 5760,
        "model_owner_count": 240,
        "model_owner_consumer_count": 5760,
        "forward_passes": 9270000,
        "optimizer_steps": 36000,
        "recurrent_steps": 0,
        "training_examples": 61800,
    }
    assert body["claims"]["both_groups_robust_harm"] is True
    assert body["claims"]["possible_interaction"] is False
    assert body["claims"]["overall_inconclusive"] is False
    assert all(value is False for value in body["forbidden_claims"].values())
    assert body["development_only"] is True
    assert body["final_family_access"] is False
    assert body["final_method_selection"] is False
    next_work = body["next_authorized_work"]
    assert next_work["status"] == "design-freeze-only"
    assert next_work["scope"] == "development-only"
    assert next_work["execute"] is False
    assert next_work["inspect_comparative_results"] is False
    assert next_work["pairing_claim"] is False
    assert next_work["final_family_access"] is False
    assert "newly captured 64-action raw-probe evidence set" in next_work["representation"]
    assert "shared exactly across every next-rung condition" in next_work["representation"]
    assert "preserving every row and action count" in next_work["same_data_control"]
    assert "removing learner-visible local pre-state/effect alignment" in next_work[
        "same_data_control"
    ]
