from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    PHASE3_OUTCOME_DIAGNOSTIC_PROTOCOL_PATH,
    OutcomeDiagnosticProtocolError,
    load_outcome_group_diagnostic_protocol,
)
from levelup.experiments.runner.config import canonical_json_bytes

ROOT = Path(__file__).parents[1]
AUTHORITY_PATHS = (
    "configs/milestone6/phase3_representation_ladder.json",
    "configs/milestone6/phase3_plan_lock.json",
    "configs/milestone6/phase3_evidence_lock.json",
    "configs/milestone6/phase3_model_artifact_authority.json",
    "configs/milestone6/phase3_anchor_selection_metrics.json",
    "configs/milestone6/phase3_development_selection.json",
)


def _sandbox(tmp_path: Path) -> tuple[Path, Path]:
    for relative in ("configs/milestone6/phase3_outcome_group_diagnostic.json", *AUTHORITY_PATHS):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path / "configs/milestone6/phase3_outcome_group_diagnostic.json", tmp_path


def _load_body(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_body(path: Path, body: dict[str, object]) -> None:
    unsigned = {key: value for key, value in body.items() if key != "diagnostic_protocol_sha256"}
    body["diagnostic_protocol_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def test_loader_accepts_complete_frozen_protocol() -> None:
    snapshot = load_outcome_group_diagnostic_protocol()
    assert snapshot.path == PHASE3_OUTCOME_DIAGNOSTIC_PROTOCOL_PATH.resolve()
    assert snapshot.payload["development_matrix"]["new_units"] == 5760
    assert len(snapshot.authority_bytes) == 6


def test_loader_rejects_protocol_self_hash_mutation(tmp_path: Path) -> None:
    path, repo = _sandbox(tmp_path)
    body = _load_body(path)
    body["scientific_question"]["claim_limit"] = "mutated"
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(OutcomeDiagnosticProtocolError, match="self-hash"):
        load_outcome_group_diagnostic_protocol(path, repository=repo)


def test_loader_rejects_same_semantics_with_different_protocol_bytes(tmp_path: Path) -> None:
    path, repo = _sandbox(tmp_path)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(OutcomeDiagnosticProtocolError, match="immutable identity"):
        load_outcome_group_diagnostic_protocol(path, repository=repo)


def test_loader_rejects_authority_symlink(tmp_path: Path) -> None:
    path, repo = _sandbox(tmp_path)
    source = repo / "configs/milestone6/phase3_plan_lock.json"
    source.unlink()
    source.symlink_to(ROOT / "configs/milestone6/phase3_plan_lock.json")
    with pytest.raises(OutcomeDiagnosticProtocolError, match="non-symlink"):
        load_outcome_group_diagnostic_protocol(path, repository=repo)


def test_loader_rejects_authority_path_escape_even_with_rehashed_protocol(tmp_path: Path) -> None:
    path, repo = _sandbox(tmp_path)
    body = _load_body(path)
    body["authority"]["phase3_protocol"]["path"] = "../README.md"
    _write_body(path, body)
    with pytest.raises(OutcomeDiagnosticProtocolError, match="escapes"):
        load_outcome_group_diagnostic_protocol(path, repository=repo)


def test_loader_rejects_protocol_outside_repository(tmp_path: Path) -> None:
    path, repo = _sandbox(tmp_path / "repository")
    outside = tmp_path / "outside.json"
    shutil.copyfile(path, outside)
    with pytest.raises(OutcomeDiagnosticProtocolError, match="protocol path escapes"):
        load_outcome_group_diagnostic_protocol(outside, repository=repo)


def test_loader_rejects_symlinked_repository_root(tmp_path: Path) -> None:
    path, repo = _sandbox(tmp_path / "repository")
    linked = tmp_path / "linked-repository"
    linked.symlink_to(repo, target_is_directory=True)
    with pytest.raises(OutcomeDiagnosticProtocolError, match="root cannot be a symlink"):
        load_outcome_group_diagnostic_protocol(path, repository=linked)


def test_loader_rejects_condition_mask_mutation(tmp_path: Path) -> None:
    path, repo = _sandbox(tmp_path)
    body = _load_body(path)
    body["conditions"][0]["representation"]["retained_indices_per_summary_block"] = [0, 1, 2, 3, 11]
    _write_body(path, body)
    with pytest.raises(
        OutcomeDiagnosticProtocolError,
        match="representation|mask is not a partition|S-RP retained|intersection",
    ):
        load_outcome_group_diagnostic_protocol(path, repository=repo)


def test_loader_rejects_rehashed_claim_or_channel_label_drift(tmp_path: Path) -> None:
    path, repo = _sandbox(tmp_path)
    body = _load_body(path)
    body["diagnostic_claim_rules"]["within_tolerance"] = "mutated"
    _write_body(path, body)
    with pytest.raises(OutcomeDiagnosticProtocolError, match="claim rules"):
        load_outcome_group_diagnostic_protocol(path, repository=repo)

    path, repo = _sandbox(tmp_path / "channel")
    body = _load_body(path)
    body["conditions"][0]["representation"]["added_channels"][0] = "hidden_reward"
    _write_body(path, body)
    with pytest.raises(OutcomeDiagnosticProtocolError, match="representation"):
        load_outcome_group_diagnostic_protocol(path, repository=repo)


def test_loader_rejects_authority_semantic_id_drift(tmp_path: Path) -> None:
    path, repo = _sandbox(tmp_path)
    protocol_path = repo / "configs/milestone6/phase3_representation_ladder.json"
    protocol = _load_body(protocol_path)
    protocol["conditions"][2]["condition_id"] = "S-renamed"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    body = _load_body(path)
    source = body["authority"]["phase3_protocol"]
    source["sha256"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    _write_body(path, body)
    with pytest.raises(OutcomeDiagnosticProtocolError, match="condition IDs"):
        load_outcome_group_diagnostic_protocol(path, repository=repo)


def test_loader_binds_matched_s_tuple_to_selection_authority(tmp_path: Path) -> None:
    path, repo = _sandbox(tmp_path)
    selection_path = repo / "configs/milestone6/phase3_development_selection.json"
    selection = _load_body(selection_path)
    selection["selected"]["S-state-availability-listwise-optimum"]["candidate_tuple_id"] = (
        "lr0p003-e120-t1p2"
    )
    unsigned = {key: value for key, value in selection.items() if key != "selection_lock_sha256"}
    selection["selection_lock_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    body = _load_body(path)
    body["authority"]["phase3_development_selection"]["sha256"] = hashlib.sha256(
        selection_path.read_bytes()
    ).hexdigest()
    body["authority"]["phase3_development_selection"]["selection_lock_sha256"] = selection[
        "selection_lock_sha256"
    ]
    _write_body(path, body)
    with pytest.raises(OutcomeDiagnosticProtocolError, match="selected .* tuple"):
        load_outcome_group_diagnostic_protocol(path, repository=repo)


def test_loader_rejects_matrix_or_boundary_drift(tmp_path: Path) -> None:
    path, repo = _sandbox(tmp_path)
    body = _load_body(path)
    body["development_matrix"]["new_units"] = 1
    _write_body(path, body)
    with pytest.raises(OutcomeDiagnosticProtocolError, match="diagnostic matrix"):
        load_outcome_group_diagnostic_protocol(path, repository=repo)

    path, repo = _sandbox(tmp_path / "boundary")
    body = _load_body(path)
    body["execution_boundary"]["final_family_access"] = True
    _write_body(path, body)
    with pytest.raises(OutcomeDiagnosticProtocolError, match="diagnostic execution boundary"):
        load_outcome_group_diagnostic_protocol(path, repository=repo)
