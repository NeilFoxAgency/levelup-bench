from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from levelup.experiments.milestone6_phase3_training_shuffle_report import (
    Phase3TrainingShuffleReport,
    Phase3TrainingShuffleReportError,
    _digest,
    _validate_body,
    load_phase3_training_shuffle_report,
    save_phase3_training_shuffle_report,
)
from levelup.experiments.runner.config import canonical_json_bytes


def _sha(index: int) -> str:
    return hashlib.sha256(str(index).encode()).hexdigest()


def _body() -> dict[str, object]:
    views: list[dict[str, object]] = []
    for index in range(30):
        owner_ids = [_sha(index * 4 + offset) for offset in range(4)]
        views.append(
            {
                "plan_id": _sha(1000),
                "protocol_sha256": _sha(1001),
                "evidence_lock_sha256": _sha(1002),
                "view_id": _sha(2000 + index),
                "condition_id": "H4-shuffled-history-transition-listwise-optimum",
                "fold_id": f"fold-{index // 5}",
                "heldout_family": ("plain", "battery", "cooldown", "heat", "momentum", "combo")[index // 5],
                "replicate": index % 5,
                "evidence_payload_sha256": _sha(3000 + index),
                "evidence_payload_bytes": 100 + index,
                "representation_identity_sha256": _sha(4000 + index),
                "model_owner_ids": owner_ids,
                "model_key_ids": [
                    _sha(4500 + index * 4 + offset) for offset in range(4)
                ],
                "model_artifact_ids": [
                    _sha(4600 + index * 4 + offset) for offset in range(4)
                ],
                "model_cost_ids": [
                    _sha(4700 + index * 4 + offset) for offset in range(4)
                ],
                "model_manifest_sha256s": [
                    _sha(4800 + index * 4 + offset) for offset in range(4)
                ],
                "model_identity_sha256s": [_sha(5000 + index * 4 + offset) for offset in range(4)],
                "training_permutation_map_sha256": _sha(6000 + index),
                "eligible_windows": 10,
                "map_nonidentity_windows": 10,
                "effective_tensor_changed_windows": 7,
                "duplicate_vector_no_effect_windows": 3,
                "unchanged_short_windows": 0,
                "effective_change_fraction": 0.7,
                "claim_eligible": False,
            }
        )
    body: dict[str, object] = {
        "schema_version": "milestone6.phase3.training-shuffle-report.v1",
        "scope": "known-development-only",
        "development_only": True,
        "final_family_access": False,
        "outcomes_included": False,
        "search_included": False,
        "model_authority_sha256": _sha(999),
        "artifact_store_id": "phase3-model-preparation-test",
        "counts": {"families": 6, "replicates": 5, "views": 30, "owners": 120, "owners_per_view": 4},
        "views": views,
    }
    body["report_sha256"] = _digest(body)
    return body


def _report() -> Phase3TrainingShuffleReport:
    body = _body()
    return Phase3TrainingShuffleReport(
        body=body,
        canonical_bytes=canonical_json_bytes(body),
        report_sha256=body["report_sha256"],  # type: ignore[arg-type]
    )


def test_below_effective_change_gate_is_ineligible_without_data_mutation() -> None:
    body = _body()
    _validate_body(body)
    assert body["views"][0]["effective_change_fraction"] == 0.7  # type: ignore[index]
    assert body["views"][0]["claim_eligible"] is False  # type: ignore[index]


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_missing_or_duplicate_view_fails_closed(mutation: str) -> None:
    body = _body()
    views = body["views"]
    if mutation == "missing":
        views.pop()  # type: ignore[union-attr]
    else:
        views[-1] = dict(views[0])  # type: ignore[index]
    body["report_sha256"] = _digest({key: value for key, value in body.items() if key != "report_sha256"})
    with pytest.raises(Phase3TrainingShuffleReportError):
        _validate_body(body)


def test_missing_owner_fails_closed() -> None:
    body = _body()
    body["views"][0]["model_owner_ids"] = body["views"][0]["model_owner_ids"][:-1]  # type: ignore[index]
    with pytest.raises(Phase3TrainingShuffleReportError):
        _validate_body(body)


def test_duplicate_family_replicate_fails_closed() -> None:
    body = _body()
    body["views"][1]["heldout_family"] = body["views"][0]["heldout_family"]  # type: ignore[index]
    body["views"][1]["replicate"] = body["views"][0]["replicate"]  # type: ignore[index]
    body["report_sha256"] = _digest(
        {key: value for key, value in body.items() if key != "report_sha256"}
    )
    with pytest.raises(Phase3TrainingShuffleReportError, match="duplicated"):
        _validate_body(body)


def test_inconsistent_shuffle_counters_fail_closed() -> None:
    body = _body()
    body["views"][0]["duplicate_vector_no_effect_windows"] = 4  # type: ignore[index]
    body["report_sha256"] = _digest(
        {key: value for key, value in body.items() if key != "report_sha256"}
    )
    with pytest.raises(Phase3TrainingShuffleReportError, match="inconsistent"):
        _validate_body(body)


def test_self_hash_drift_fails_closed() -> None:
    body = _body()
    body["views"][0]["training_permutation_map_sha256"] = _sha(9999)  # type: ignore[index]
    with pytest.raises(Phase3TrainingShuffleReportError):
        _validate_body(body)


def test_atomic_save_and_load_reject_symlink_and_nonregular(tmp_path: Path) -> None:
    report = _report()
    target = tmp_path / "training-shuffle-report.json"
    save_phase3_training_shuffle_report(target, report)
    assert load_phase3_training_shuffle_report(target).report_sha256 == report.report_sha256

    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(Phase3TrainingShuffleReportError):
        save_phase3_training_shuffle_report(link, report)
    with pytest.raises(Phase3TrainingShuffleReportError):
        load_phase3_training_shuffle_report(link)

    directory = tmp_path / "directory.json"
    directory.mkdir()
    with pytest.raises(Phase3TrainingShuffleReportError):
        save_phase3_training_shuffle_report(directory, report)
    with pytest.raises(Phase3TrainingShuffleReportError):
        load_phase3_training_shuffle_report(directory)


def test_atomic_json_bytes_are_canonical(tmp_path: Path) -> None:
    report = _report()
    target = tmp_path / "report.json"
    save_phase3_training_shuffle_report(target, report)
    assert target.read_bytes() == canonical_json_bytes(report.body)
    assert not any(path.name.endswith(".tmp") for path in tmp_path.iterdir())


def test_existing_report_is_write_once_and_idempotent(tmp_path: Path) -> None:
    report = _report()
    target = tmp_path / "report.json"
    save_phase3_training_shuffle_report(target, report)
    save_phase3_training_shuffle_report(target, report)

    target.write_bytes(b"{}")
    with pytest.raises(Phase3TrainingShuffleReportError, match="conflicts"):
        save_phase3_training_shuffle_report(target, report)
