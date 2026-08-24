import hashlib
import json
from fractions import Fraction
from pathlib import Path

from levelup.experiments.runner.config import canonical_json_bytes

LOCK = Path("configs/milestone6/phase3_development_selection.json")
FAMILIES = ("plain", "battery", "cooldown", "heat", "momentum", "combo")


def _fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def test_phase3_development_selection_lock_is_self_hashed_and_development_only() -> None:
    body = json.loads(LOCK.read_bytes())
    supplied = body.pop("selection_lock_sha256")
    assert supplied == hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    boundary = body["scientific_boundary"]
    assert boundary == {
        "development_only": True,
        "final_family_access": False,
        "final_method_selection": False,
        "final_families_locked": True,
        "advancement_to_paired_objectives": False,
        "permitted_next_step": "predeclared development-only diagnostics; preserve B2 as the strong reference and do not unlock final families",
    }
    assert body["matrix"]["family_order"] == list(FAMILIES)
    assert body["matrix"]["units"] == body["matrix"]["completed"] == 11_520
    assert body["matrix"]["failed"] == body["matrix"]["interrupted"] == 0


def test_phase3_claim_deltas_and_selected_family_counts_are_exact() -> None:
    body = json.loads(LOCK.read_bytes())
    selected = body["selected"]
    expected_conditions = {
        "B2-global-listwise-optimum",
        "T-markov-state-transition-listwise-optimum",
        "S-state-availability-listwise-optimum",
        "H0-null-history-transition-listwise-optimum",
        "H4-causal-history-transition-listwise-optimum",
        "H4-shuffled-history-transition-listwise-optimum",
    }
    assert set(selected) == expected_conditions
    for row in selected.values():
        assert len(row["family_success_counts"]) == len(FAMILIES)
        assert all(0 <= count <= 40 for count in row["family_success_counts"])
        assert _fraction(row["minimum_family_success_rate"]) == Fraction(
            min(row["family_success_counts"]), 40
        )

    claims = body["claims"]
    s = _fraction(selected["S-state-availability-listwise-optimum"]["minimum_family_success_rate"])
    t = _fraction(
        selected["T-markov-state-transition-listwise-optimum"]["minimum_family_success_rate"]
    )
    h0 = _fraction(
        selected["H0-null-history-transition-listwise-optimum"]["minimum_family_success_rate"]
    )
    h4 = _fraction(
        selected["H4-causal-history-transition-listwise-optimum"]["minimum_family_success_rate"]
    )
    shuffled = _fraction(
        selected["H4-shuffled-history-transition-listwise-optimum"]["minimum_family_success_rate"]
    )
    b2 = _fraction(selected["B2-global-listwise-optimum"]["minimum_family_success_rate"])
    assert _fraction(claims["transition_information_beyond_state"]["delta_T_minus_S"]) == t - s
    history = claims["history_access_beyond_transition_and_null_history"]
    assert _fraction(history["delta_H4_minus_T"]) == h4 - t
    assert _fraction(history["delta_H4_minus_H0"]) == h4 - h0
    assert _fraction(claims["sequence_order"]["delta_H4_minus_H4_shuffled"]) == h4 - shuffled
    assert _fraction(claims["S_vs_B2_diagnostic"]["delta_S_minus_B2"]) == s - b2
    assert claims["transition_information_beyond_state"]["claim"] is False
    assert claims["history_access_beyond_transition_and_null_history"]["claim"] is False
    assert claims["sequence_order"]["claim"] is False
    assert claims["H4_advancement_to_paired_objectives"]["claim"] is False
    assert claims["S_vs_B2_diagnostic"]["robust_superiority_claim"] is False
