"""Fail-closed tests for the committed Phase 3 anchor metric authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from levelup.experiments.milestone6_phase3_anchor_selection_metrics import (
    PHASE3_ANCHOR_SELECTION_METRICS_PATH,
    AnchorSelectionMetricsError,
    load_phase3_anchor_selection_metrics_bytes,
    phase3_anchor_selected_metrics,
    validate_phase3_anchor_selection_metrics_bytes,
)
from levelup.experiments.milestone6_phase3_models import (
    H0_CONDITION,
    H4_CONDITION,
    H4_SHUFFLED_CONDITION,
    S_CONDITION,
)
from levelup.experiments.milestone6_phase3_selection import evaluate_phase3_claims
from levelup.experiments.runner.config import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[1]


def _body() -> dict:
    return json.loads(PHASE3_ANCHOR_SELECTION_METRICS_PATH.read_bytes())


def _bytes(body: dict) -> bytes:
    unsigned = {
        key: value for key, value in body.items() if key != "anchor_selection_metrics_sha256"
    }
    body["anchor_selection_metrics_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return canonical_json_bytes(body)


def test_committed_authority_is_canonical_and_validates_lineage() -> None:
    content = load_phase3_anchor_selection_metrics_bytes()
    snapshot = validate_phase3_anchor_selection_metrics_bytes(content, repository=ROOT)
    assert snapshot.sha256 == json.loads(content)["anchor_selection_metrics_sha256"]
    assert (
        snapshot.body["conditions"]["T-markov-state-transition-listwise-optimum"][
            "historical_condition_id"
        ]
        == "C-state-conditioned-listwise-optimum"
    )
    b2, t = phase3_anchor_selected_metrics(snapshot)
    assert b2.minimum_family_success_rate.numerator == 2
    assert b2.minimum_family_success_rate.denominator == 5
    assert t.minimum_family_success_rate.numerator == 3
    assert t.minimum_family_success_rate.denominator == 40
    assert b2.macro_average_family_median_restricted_interactions.numerator == 1975
    assert b2.macro_average_family_median_restricted_interactions.denominator == 3
    assert t.macro_average_family_median_restricted_interactions.numerator == 7409
    assert t.macro_average_family_median_restricted_interactions.denominator == 12
    assert tuple(item.family_id for item in b2.family_metrics) == (
        "plain",
        "battery",
        "cooldown",
        "heat",
        "momentum",
        "combo",
    )
    evaluate_phase3_claims(
        {
            S_CONDITION: replace(b2, condition_id=S_CONDITION),
            H0_CONDITION: replace(b2, condition_id=H0_CONDITION),
            H4_CONDITION: replace(b2, condition_id=H4_CONDITION),
            H4_SHUFFLED_CONDITION: replace(
                b2,
                condition_id=H4_SHUFFLED_CONDITION,
                heldout_shuffle_claim_eligible=True,
                training_shuffle_claim_eligible=True,
            ),
        },
        locked_b2=b2,
        locked_t=t,
        training_shuffle_claim_eligible=True,
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("metric_contract", "endpoint"), 2047),
        (("metric_contract", "failure_sentinel"), 2048),
        (
            (
                "conditions",
                "B2-global-listwise-optimum",
                "families",
                3,
                "exact_optimum_success_count",
            ),
            17,
        ),
        (
            (
                "conditions",
                "B2-global-listwise-optimum",
                "families",
                3,
                "exact_optimum_success_rate",
            ),
            0.425,
        ),
        (
            ("conditions", "T-markov-state-transition-listwise-optimum", "historical_condition_id"),
            "not-C",
        ),
    ],
)
def test_metric_boundary_mutations_fail_closed(path: tuple[object, ...], value: object) -> None:
    body = _body()
    target = body
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(AnchorSelectionMetricsError):
        validate_phase3_anchor_selection_metrics_bytes(_bytes(body), repository=ROOT)


def test_linked_selection_hash_mutation_fails_closed() -> None:
    body = _body()
    body["source"]["selection_lock"]["sha256"] = "0" * 64
    with pytest.raises(AnchorSelectionMetricsError, match="selection lock source hash"):
        validate_phase3_anchor_selection_metrics_bytes(_bytes(body), repository=ROOT)


def test_anchor_manifest_self_hash_mutation_fails_closed() -> None:
    body = _body()
    body["source"]["anchor_manifest"]["anchor_manifest_sha256"] = "0" * 64
    with pytest.raises(AnchorSelectionMetricsError, match="anchor manifest self-hash"):
        validate_phase3_anchor_selection_metrics_bytes(_bytes(body), repository=ROOT)


def test_raw_analysis_is_not_required_at_runtime(tmp_path: Path) -> None:
    body = _body()
    body["source"]["analysis_file"]["path"] = "runs/milestone6/does-not-exist-analysis.json"
    validate_phase3_anchor_selection_metrics_bytes(_bytes(body), repository=ROOT)


def test_symlinked_authority_path_fails_closed(tmp_path: Path) -> None:
    link = tmp_path / "metrics.json"
    link.symlink_to(PHASE3_ANCHOR_SELECTION_METRICS_PATH)
    with pytest.raises(AnchorSelectionMetricsError):
        load_phase3_anchor_selection_metrics_bytes(link, repository=ROOT)
