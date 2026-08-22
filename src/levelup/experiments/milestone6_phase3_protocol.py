"""Typed, fail-closed loader for the frozen Milestone 6 Phase 3 protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PHASE3_PROTOCOL_PATH = ROOT / "configs/milestone6/phase3_representation_ladder.json"
SCHEMA_VERSION = "milestone6.phase3_representation_ladder.v1"
STATUS = "frozen-before-phase3-comparative-development-results"
FAMILIES = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
CONDITIONS = (
    "B2-global-listwise-optimum",
    "T-markov-state-transition-listwise-optimum",
    "S-state-availability-listwise-optimum",
    "H0-null-history-transition-listwise-optimum",
    "H4-causal-history-transition-listwise-optimum",
    "H4-shuffled-history-transition-listwise-optimum",
)
NEW_CONDITIONS = CONDITIONS[2:]


@dataclass(frozen=True, slots=True)
class Phase3ProtocolSnapshot:
    repository: Path
    path: Path
    content: bytes
    sha256: str
    payload: dict[str, Any]
    authority_bytes: tuple[tuple[str, bytes], ...]


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _relative_source(repository: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Phase 3 authority path must be a nonempty string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("Phase 3 authority path must stay inside the repository")
    path = repository.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Phase 3 authority source must be a regular non-symlink file")
    return path


def _validate_authority(
    repository: Path,
    payload: dict[str, Any],
) -> tuple[tuple[str, bytes], ...]:
    authority = payload.get("authority")
    if not isinstance(authority, dict):
        raise ValueError("Phase 3 authority must be an object")
    rows: list[tuple[str, bytes]] = []
    for key in (
        "development_protocol",
        "development_tasks",
        "phase2_candidates",
        "phase2_selection_lock",
    ):
        source = authority.get(key)
        if not isinstance(source, dict) or set(source) < {"path", "sha256"}:
            raise ValueError(f"Phase 3 authority source {key} is incomplete")
        path = _relative_source(repository, source["path"])
        content = path.read_bytes()
        if _sha256(content) != source["sha256"]:
            raise ValueError(f"Phase 3 authority source {key} changed")
        rows.append((key, content))

    selection_source = authority["phase2_selection_lock"]
    selection = json.loads(dict(rows)["phase2_selection_lock"])
    if selection.get("analysis", {}).get("analysis_sha256") != selection_source.get(
        "analysis_sha256"
    ):
        raise ValueError("Phase 3 authority selection analysis identity changed")
    boundary = selection.get("scientific_boundary", {})
    if (
        boundary.get("development_only") is not True
        or boundary.get("final_family_access") is not False
        or boundary.get("final_method_selection") is not False
    ):
        raise ValueError("Phase 3 authority selection lock is not development-only")
    return tuple(rows)


def _validate_matrix(payload: dict[str, Any]) -> None:
    matrix = payload.get("development_matrix")
    if not isinstance(matrix, dict):
        raise ValueError("Phase 3 development matrix must be an object")
    expected = {
        "family_order": list(FAMILIES),
        "replicates": [0, 1, 2, 3, 4],
        "tasks_per_condition_tuple": 240,
        "phase2_anchor_conditions": 2,
        "new_conditions": 4,
        "candidate_tuples_per_condition": 12,
        "existing_anchor_units": 5760,
        "new_units": 11520,
        "combined_comparison_units": 17280,
        "canonical_evidence_artifacts": 30,
        "existing_anchor_views": 60,
        "new_views": 120,
        "combined_views": 180,
        "existing_anchor_models": 240,
        "new_models": 480,
        "combined_models": 720,
    }
    for key, value in expected.items():
        if matrix.get(key) != value:
            raise ValueError(f"Phase 3 matrix field {key} drifted")
    if matrix["new_units"] != len(NEW_CONDITIONS) * 12 * 6 * 5 * 8:
        raise ValueError("Phase 3 new-unit arithmetic drifted")


def _validate_conditions(payload: dict[str, Any]) -> None:
    rows = payload.get("conditions")
    if not isinstance(rows, list) or tuple(
        row.get("condition_id") if isinstance(row, dict) else None for row in rows
    ) != CONDITIONS:
        raise ValueError("Phase 3 condition order or identity drifted")
    by_id = {row["condition_id"]: row for row in rows}
    if by_id[CONDITIONS[0]].get("role") != "strong_optimum_imitation_anchor":
        raise ValueError("Phase 3 strong B2 anchor is missing")
    if by_id[CONDITIONS[1]].get("historical_condition_id") != (
        "C-state-conditioned-listwise-optimum"
    ):
        raise ValueError("Phase 3 T/C alias drifted")
    state = by_id[CONDITIONS[2]].get("representation", {})
    if (
        state.get("retained_indices_per_summary_block") != [0, 1, 2, 3, 11]
        or state.get("zeroed_indices_per_summary_block") != [4, 5, 6, 7, 8, 9, 10]
        or state.get("coverage_index") != 48
    ):
        raise ValueError("Phase 3 state/availability mask drifted")
    null = by_id[CONDITIONS[3]]
    ordered = by_id[CONDITIONS[4]]
    shuffled = by_id[CONDITIONS[5]]
    if null.get("architecture") != ordered.get("architecture"):
        raise ValueError("Phase 3 null and ordered histories are not architecture-matched")
    if shuffled.get("architecture") != "identical_to_H4":
        raise ValueError("Phase 3 shuffled history architecture drifted")
    controls = shuffled.get("control_semantics", {})
    if (
        controls.get("same_multiset") is not True
        or controls.get("future_transitions") is not False
        or "derangement" not in str(controls.get("permutation", ""))
        or "0.80" not in str(controls.get("sequence_claim_coverage_gate", ""))
    ):
        raise ValueError("Phase 3 shuffled-history control drifted")


def _validate_capacity_and_budget(payload: dict[str, Any]) -> None:
    capacity = payload.get("capacity_matching", {})
    counts = capacity.get("counts")
    expected = {
        "B2": 3601,
        "S": 3841,
        "T": 3841,
        "H0": 3889,
        "H4": 3889,
        "H4_shuffled": 3889,
    }
    if counts != expected:
        raise ValueError("Phase 3 parameter-count authority drifted")
    tolerance = capacity.get("symmetric_parameter_tolerance_fraction")
    if tolerance != 0.1:
        raise ValueError("Phase 3 parameter tolerance drifted")
    if any(
        abs(left - right) / max(left, right) > tolerance
        for left in counts.values()
        for right in counts.values()
    ):
        raise ValueError("Phase 3 parameter counts exceed tolerance")

    fixed = payload.get("fixed_training_and_search", {})
    expected_fixed = {
        "optimizer": "adam",
        "weight_decay": 0.0001,
        "device": "cpu",
        "torch_threads": 1,
        "processes": 1,
        "probe_actions_per_task": 64,
        "probe_coverage_target_samples_per_alias": 8,
        "probe_actions_per_attempt": 16,
        "candidate_episodes_per_task": 150,
        "adaptation_actions_per_task": 2048,
        "maximum_actions_per_candidate_episode": 64,
        "exact_optimum_affects_search_control_flow": False,
    }
    for key, value in expected_fixed.items():
        if fixed.get(key) != value:
            raise ValueError(f"Phase 3 fixed field {key} drifted")


def _validate_candidate_grid(
    payload: dict[str, Any],
    authority_bytes: tuple[tuple[str, bytes], ...],
) -> None:
    phase2 = json.loads(dict(authority_bytes)["phase2_candidates"])
    if payload.get("candidate_tuples") != phase2.get("candidate_tuples"):
        raise ValueError("Phase 3 candidate grid differs from Phase 2")
    if len(payload["candidate_tuples"]) != 12:
        raise ValueError("Phase 3 candidate grid must contain exactly 12 tuples")


def load_phase3_protocol(
    path: str | Path = PHASE3_PROTOCOL_PATH,
    *,
    repository: str | Path = ROOT,
) -> Phase3ProtocolSnapshot:
    """Load and validate the complete frozen development-only Phase 3 contract."""

    repo = Path(repository).resolve(strict=True)
    protocol_path = Path(path)
    if protocol_path.is_symlink() or not protocol_path.is_file():
        raise ValueError("Phase 3 protocol must be a regular non-symlink file")
    content = protocol_path.read_bytes()
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("Phase 3 protocol must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Phase 3 protocol schema drifted")
    if payload.get("status") != STATUS or payload.get("scope") != "known-development-only":
        raise ValueError("Phase 3 protocol is not frozen development-only authority")
    if payload.get("final_family_access") is not False:
        raise ValueError("Phase 3 protocol attempted final-family access")
    freeze = payload.get("freeze_record", {})
    if (
        freeze.get("phase2_results_used_for_diagnosis") is not True
        or freeze.get("phase3_comparative_results_inspected") is not False
    ):
        raise ValueError("Phase 3 freeze timing drifted")

    authority_bytes = _validate_authority(repo, payload)
    _validate_matrix(payload)
    _validate_conditions(payload)
    _validate_capacity_and_budget(payload)
    _validate_candidate_grid(payload, authority_bytes)
    return Phase3ProtocolSnapshot(
        repository=repo,
        path=protocol_path.resolve(strict=True),
        content=content,
        sha256=_sha256(content),
        payload=payload,
        authority_bytes=authority_bytes,
    )
