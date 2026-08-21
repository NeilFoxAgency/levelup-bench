from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "milestone6" / "phase2_screening_candidates.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decimal_id(value: float) -> str:
    return format(value, "g").replace(".", "p")


def _tuple_id(learning_rate: float, epochs: int, temperature: float) -> str:
    return f"lr{_decimal_id(learning_rate)}-e{epochs}-t{_decimal_id(temperature)}"


def _training_tuple_id(learning_rate: float, epochs: int) -> str:
    return f"lr{_decimal_id(learning_rate)}-e{epochs}"


def test_phase2_screening_manifest_binds_frozen_inputs_and_stays_development_only() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    protocol = payload["parent_protocol"]
    tasks = payload["task_manifest"]
    assert payload["status"] == "frozen-before-screening-results"
    assert payload["scope"] == "known-development-only"
    assert payload["final_family_access"] is False
    assert _sha256(ROOT / protocol["path"]) == protocol["sha256"]
    assert _sha256(ROOT / tasks["path"]) == tasks["sha256"]
    assert payload["folds"]["replicates"] == list(range(5))
    assert payload["folds"]["family_order"] == [
        "plain",
        "battery",
        "cooldown",
        "heat",
        "momentum",
        "combo",
    ]


def test_phase2_screening_candidate_grid_is_complete_unique_and_symmetric() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tuples = payload["candidate_tuples"]
    assert len(tuples) == 12
    assert len({item["tuple_id"] for item in tuples}) == 12
    assert all(
        item["tuple_id"]
        == _tuple_id(
            item["learning_rate"],
            item["training_epochs"],
            item["search_temperature"],
        )
        for item in tuples
    )
    assert all(
        item["training_tuple_id"]
        == _training_tuple_id(item["learning_rate"], item["training_epochs"])
        for item in tuples
    )
    assert len({item["training_tuple_id"] for item in tuples}) == 4
    assert {
        (item["learning_rate"], item["training_epochs"], item["search_temperature"])
        for item in tuples
    } == {
        (learning_rate, epochs, temperature)
        for learning_rate in (0.003, 0.01)
        for epochs in (120, 180)
        for temperature in (0.6, 0.9, 1.2)
    }
    assert len(payload["fixed_controls"]) == 2
    learned = payload["learned_conditions"]
    assert [item["condition_id"].split("-", 1)[0] for item in learned] == [
        "B1",
        "B2",
        "C",
    ]
    assert all(item["candidate_tuple_ids"] == "all" for item in learned)


def test_phase2_screening_counts_and_artifact_sharing_are_frozen() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    counts = payload["expected_matrix"]
    assert counts == {
        "fixed_control_variants": 2,
        "learned_variants_per_condition": 12,
        "learned_conditions": 3,
        "total_variants": 38,
        "canonical_evidence_artifacts": 30,
        "training_data_artifacts": 90,
        "trained_model_artifacts": 360,
        "heldout_task_units": 9120,
    }
    sharing = payload["artifact_sharing"]
    assert "condition and temperature excluded" in sharing[
        "canonical_evidence_artifact_identity"
    ]
    assert sharing["cross_condition_evidence_reuse"].startswith("B1, B2 and C")
    assert sharing["search_temperature_in_model_artifact_identity"] is False
    assert sharing["heldout_probe_reuse_across_units"] is False
    assert sharing["task_unit_training_cost"] == 0
    advancement = payload["screening_advancement_rule"]
    assert advancement["cross_condition_elimination"] is False
    assert advancement["endpoint_adaptation_actions"] == 2048
    assert advancement["selection_rule_changes_after_screening_results"] == "forbidden"
    assert advancement["steps"][-1] == (
        "choose the ascending numeric tuple (learning_rate, training_epochs, search_temperature)"
    )
