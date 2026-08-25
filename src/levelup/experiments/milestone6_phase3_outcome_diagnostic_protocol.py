"""Typed, fail-closed authority for the Phase 3 outcome-group diagnostic.

This loader validates only the small, predeclared development diagnostic contract.
It reads committed authority sources through descriptor-anchored paths, checks their
bytes and semantic identities, and never opens a result store or final-family data.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[3]
PHASE3_OUTCOME_DIAGNOSTIC_PROTOCOL_PATH = (
    ROOT / "configs/milestone6/phase3_outcome_group_diagnostic.json"
)
SCHEMA_VERSION = "milestone6.phase3.outcome-group-diagnostic.v1"
STATUS = "frozen-before-outcome-group-diagnostic-results"
SOURCE_RESULT_LOCK_COMMIT = "9ff9596bf64ef341d69759c0e0db680c51e768f9"
EXPECTED_PROTOCOL_SELF_HASH = "88b348600ba66494fc2d64af3c86f5a33989c2c8c2bb6d5aea3eda15809214c3"
EXPECTED_PROTOCOL_FILE_SHA256 = "dda43928b46bbf6981d50fb9d03abc5c344cfe10612d5e16cc02bd76da7646c3"
FAMILIES = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
REPLICATES = (0, 1, 2, 3, 4)
BASE_CONDITION = "S-state-availability-listwise-optimum"
CONDITIONS = (
    "S-RP-state-resource-pressure-outcome-listwise-optimum",
    "S-PEC-state-progress-elapsed-completion-listwise-optimum",
)
PHASE3_CONDITIONS = (
    "B2-global-listwise-optimum",
    "T-markov-state-transition-listwise-optimum",
    BASE_CONDITION,
    "H0-null-history-transition-listwise-optimum",
    "H4-causal-history-transition-listwise-optimum",
    "H4-shuffled-history-transition-listwise-optimum",
)
TUPLE_IDS = (
    "lr0p003-e120-t0p6",
    "lr0p003-e120-t0p9",
    "lr0p003-e120-t1p2",
    "lr0p003-e180-t0p6",
    "lr0p003-e180-t0p9",
    "lr0p003-e180-t1p2",
    "lr0p01-e120-t0p6",
    "lr0p01-e120-t0p9",
    "lr0p01-e120-t1p2",
    "lr0p01-e180-t0p6",
    "lr0p01-e180-t0p9",
    "lr0p01-e180-t1p2",
)
STATE_INDICES = [0, 1, 2, 3, 4]
SUMMARY_INDICES = set(range(12))
S_RETAINED = {0, 1, 2, 3, 11}
S_ZEROED = SUMMARY_INDICES - S_RETAINED
TIE_BREAK = [
    "worst_family_median_restricted_interactions",
    "macro_average_family_median_restricted_interactions",
    "summed_unique_model_owner_optimizer_steps",
    "summed_unique_model_owner_forward_passes",
    "summed_unique_model_owner_recurrent_steps",
    "numeric_learning_rate_epochs_temperature",
]
UNAVAILABLE_DIAGNOSTICS = {
    "Heat pressure-cap decision logits": "unavailable from the completed Phase 3 records; logits and a frozen decision scope were not persisted, so no reconstruction is permitted",
    "state-action coverage": "unavailable as a rate; the completed records do not persist an exact visible state-action denominator",
    "unknown-affordance rate": "unavailable as a rate; an unknown-alias count exists but the visible-action-slot denominator was not persisted",
    "label frequency": "not persisted and operationally under-specified; do not derive a post-hoc statistic for selection or claims",
    "history length actually available": "only aggregate recurrent-step totals are persisted, not the predeclared length distribution; do not invent a histogram",
    "shuffle realized-change rate": "available only from the typed H4 training and held-out eligible/effective counters with exact lineage; it remains non-selection evidence",
}
_HEX = frozenset("0123456789abcdef")


class OutcomeDiagnosticProtocolError(ValueError):
    """Raised when the diagnostic authority is malformed or has drifted."""


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticProtocolSnapshot:
    repository: Path
    path: Path
    content: bytes
    sha256: str
    payload: dict[str, Any]
    authority_bytes: tuple[tuple[str, bytes], ...]


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _digest(value: Any) -> str:
    return _sha256(canonical_json_bytes(value))


def _digest_field(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX for c in value):
        raise OutcomeDiagnosticProtocolError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OutcomeDiagnosticProtocolError(f"{label} must be an object")
    return value


def _relative_path(repository: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise OutcomeDiagnosticProtocolError(f"{label} path is missing")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise OutcomeDiagnosticProtocolError(f"{label} path escapes the repository")
    target = repository.joinpath(*pure.parts)
    if target.is_symlink() or not target.is_file():
        raise OutcomeDiagnosticProtocolError(f"{label} must be a regular non-symlink file")
    return target


def _read_source(repository: Path, source: Mapping[str, Any], label: str) -> bytes:
    declared = _digest_field(source.get("sha256"), f"{label} sha256")
    target = _relative_path(repository, source.get("path"), label)
    try:
        parent_fd = secure_fs.open_directory_chain(target.parent)
        try:
            content = secure_fs.read_bytes_at(parent_fd, target.name)
        finally:
            os.close(parent_fd)
    except (OSError, RuntimeError, TypeError, ValueError, secure_fs.SecureFilesystemError) as exc:
        raise OutcomeDiagnosticProtocolError(f"{label} cannot be read safely") from exc
    if _sha256(content) != declared:
        raise OutcomeDiagnosticProtocolError(f"{label} source hash changed")
    return content


def _json(content: bytes, label: str, *, canonical: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise OutcomeDiagnosticProtocolError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise OutcomeDiagnosticProtocolError(f"{label} must be an object")
    if canonical and canonical_json_bytes(value) != content:
        raise OutcomeDiagnosticProtocolError(f"{label} is not canonical JSON")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise OutcomeDiagnosticProtocolError(f"{label} drifted")


def _validate_self_hash(payload: Mapping[str, Any]) -> None:
    supplied = _digest_field(
        payload.get("diagnostic_protocol_sha256"), "diagnostic protocol self-hash"
    )
    unsigned = {key: value for key, value in payload.items() if key != "diagnostic_protocol_sha256"}
    if _digest(unsigned) != supplied:
        raise OutcomeDiagnosticProtocolError("diagnostic protocol self-hash mismatch")


def _validate_phase3_protocol(protocol: Mapping[str, Any]) -> None:
    _require_equal(
        protocol.get("schema_version"),
        "milestone6.phase3_representation_ladder.v1",
        "Phase 3 protocol schema",
    )
    _require_equal(protocol.get("scope"), "known-development-only", "Phase 3 protocol scope")
    _require_equal(protocol.get("final_family_access"), False, "Phase 3 final-family boundary")
    rows = protocol.get("conditions")
    if (
        not isinstance(rows, list)
        or tuple(_mapping(row, "Phase 3 condition").get("condition_id") for row in rows)
        != PHASE3_CONDITIONS
    ):
        raise OutcomeDiagnosticProtocolError("Phase 3 condition IDs drifted")
    _require_equal(
        rows[0],
        {
            "condition_id": PHASE3_CONDITIONS[0],
            "role": "strong_optimum_imitation_anchor",
            "source": "locked_phase2",
            "representation": "49-channel global empirical action-transition summary without current state",
            "trainable_parameters": 3601,
        },
        "Phase 3 B2 anchor",
    )
    _require_equal(
        rows[1],
        {
            "condition_id": PHASE3_CONDITIONS[1],
            "historical_condition_id": "C-state-conditioned-listwise-optimum",
            "role": "markov_transition_anchor",
            "source": "locked_phase2",
            "representation": "five current-state scalars plus the full 49-channel empirical action-transition summary",
            "trainable_parameters": 3841,
            "bitwise_parity_requirement": "the T representation, model, logits, generation, and selected Phase 2 outcomes must remain identical to historical C",
        },
        "Phase 3 T anchor",
    )
    _require_equal(
        protocol.get("scientific_questions", {}).get("explicit_pairing"),
        {"status": "deferred", "claims_before_D1_F_gate": "forbidden"},
        "Phase 3 pairing boundary",
    )
    tuples = protocol.get("candidate_tuples")
    if (
        not isinstance(tuples, list)
        or tuple(row.get("tuple_id") for row in tuples if isinstance(row, Mapping)) != TUPLE_IDS
    ):
        raise OutcomeDiagnosticProtocolError("Phase 3 candidate grid drifted")
    _require_equal(
        protocol.get("development_matrix", {}).get("family_order"),
        list(FAMILIES),
        "Phase 3 family order",
    )
    _require_equal(
        protocol.get("development_matrix", {}).get("replicates"),
        list(REPLICATES),
        "Phase 3 replicates",
    )
    _require_equal(
        protocol.get("fixed_training_and_search"),
        {
            "objective": "optimum-only listwise next-action imitation",
            "examples": "one causal decision example per optimum transition; identical identities, labels, count, order, and batches across S, T, H0, H4, and H4-shuffled",
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
            "oracle_timing": "after the complete fixed candidate batch and independent replay",
        },
        "Phase 3 fixed training/search",
    )
    _require_equal(
        protocol.get("capacity_matching", {}).get("strong_B2_floor"),
        "B2 retains the complete Phase 2 candidate grid, examples, optimizer updates, probes, candidate episodes, interaction budget, and selected result; it is never capacity-, training-, or search-restricted to favor H4",
        "Phase 3 strong B2 floor",
    )
    s = _mapping(rows[2], "Phase 3 S condition").get("representation")
    s = _mapping(s, "Phase 3 S representation")
    _require_equal(
        s.get("retained_indices_per_summary_block"), sorted(S_RETAINED), "Phase 3 S retained mask"
    )
    _require_equal(
        s.get("zeroed_indices_per_summary_block"), sorted(S_ZEROED), "Phase 3 S zeroed mask"
    )
    _require_equal(s.get("input_width"), 54, "Phase 3 S input width")


def _validate_authorities(
    body: Mapping[str, Any], repository: Path
) -> tuple[tuple[str, bytes], ...]:
    authority = _mapping(body.get("authority"), "authority")
    expected_names = (
        "phase3_protocol",
        "phase3_plan",
        "phase3_evidence",
        "phase3_model_authority",
        "phase3_anchor_metrics",
        "phase3_development_selection",
    )
    if set(authority) != set(expected_names):
        raise OutcomeDiagnosticProtocolError("diagnostic authority source set drifted")
    rows: list[tuple[str, bytes]] = []
    parsed: dict[str, dict[str, Any]] = {}
    source_keys = {
        "phase3_protocol": {"path", "sha256"},
        "phase3_plan": {"path", "sha256", "plan_id"},
        "phase3_evidence": {"path", "sha256", "evidence_lock_sha256"},
        "phase3_model_authority": {"path", "sha256", "authority_sha256"},
        "phase3_anchor_metrics": {"path", "sha256", "anchor_selection_metrics_sha256"},
        "phase3_development_selection": {
            "path",
            "sha256",
            "selection_lock_sha256",
            "analysis_sha256",
        },
    }
    for name in expected_names:
        source = _mapping(authority.get(name), f"authority {name}")
        if set(source) != source_keys[name]:
            raise OutcomeDiagnosticProtocolError(f"authority {name} field set drifted")
        content = _read_source(repository, source, f"authority {name}")
        rows.append((name, content))
        parsed[name] = _json(
            content,
            name,
            canonical=name in {"phase3_plan", "phase3_evidence", "phase3_anchor_metrics"},
        )
    protocol = parsed["phase3_protocol"]
    _validate_phase3_protocol(protocol)
    plan = parsed["phase3_plan"]
    _require_equal(
        plan.get("schema_version"), "milestone6.phase3.plan-lock.v1", "Phase 3 plan schema"
    )
    _require_equal(plan.get("scope"), "known-development-only", "Phase 3 plan scope")
    _require_equal(plan.get("final_family_access"), False, "Phase 3 plan final boundary")
    plan_self = _digest_field(plan.get("plan_lock_sha256"), "Phase 3 plan self-hash")
    if _digest({k: v for k, v in plan.items() if k != "plan_lock_sha256"}) != plan_self:
        raise OutcomeDiagnosticProtocolError("Phase 3 plan self-hash mismatch")
    _require_equal(
        plan.get("protocol_sha256"),
        authority["phase3_protocol"].get("sha256"),
        "Phase 3 plan protocol lineage",
    )
    _require_equal(
        plan.get("plan_id"), authority["phase3_plan"].get("plan_id"), "Phase 3 plan ID lineage"
    )
    _require_equal(
        plan.get("condition_ids"),
        [BASE_CONDITION, PHASE3_CONDITIONS[3], PHASE3_CONDITIONS[4], PHASE3_CONDITIONS[5]],
        "Phase 3 plan conditions",
    )
    _require_equal(plan.get("family_order"), list(FAMILIES), "Phase 3 plan families")
    _require_equal(plan.get("replicates"), list(REPLICATES), "Phase 3 plan replicates")
    _require_equal(plan.get("candidate_tuple_ids"), list(TUPLE_IDS), "Phase 3 plan candidate grid")
    _require_equal(
        plan.get("counts"),
        {"views": 120, "model_owners": 480, "units": 11520},
        "Phase 3 plan counts",
    )

    evidence = parsed["phase3_evidence"]
    for key, expected in (
        ("scope", "known-development-only"),
        ("final_family_access", False),
        ("outcomes_included", False),
        ("payloads_included", False),
        ("aggregates", []),
        ("final_results", []),
    ):
        _require_equal(evidence.get(key), expected, f"Phase 3 evidence {key}")
    _require_equal(
        evidence.get("counts"),
        {"evidence_artifacts": 30, "families": 6, "replicates": 5},
        "Phase 3 evidence counts",
    )
    _require_equal(
        evidence.get("evidence_lock_sha256"),
        authority["phase3_evidence"].get("evidence_lock_sha256"),
        "Phase 3 evidence self-hash lineage",
    )

    model = parsed["phase3_model_authority"]
    for key, expected in (
        ("development_only", True),
        ("execution_authorized", True),
        ("final", False),
        ("final_family_accessed", False),
        ("family_order", list(FAMILIES)),
        ("replicates", list(REPLICATES)),
        ("condition_ids", [BASE_CONDITION, *PHASE3_CONDITIONS[3:]]),
        ("expected_evidence_count", 30),
        ("expected_view_count", 120),
        ("expected_model_count", 480),
    ):
        _require_equal(model.get(key), expected, f"Phase 3 model authority {key}")
    _require_equal(model.get("plan_id"), plan.get("plan_id"), "Phase 3 model plan lineage")
    _require_equal(
        model.get("authority_sha256"),
        authority["phase3_model_authority"].get("authority_sha256"),
        "Phase 3 model authority ID lineage",
    )
    _require_equal(
        model.get("protocol_sha256"),
        authority["phase3_protocol"].get("sha256"),
        "Phase 3 model protocol lineage",
    )
    _require_equal(
        model.get("evidence_lock_sha256"),
        evidence.get("evidence_lock_sha256"),
        "Phase 3 model evidence lineage",
    )

    anchor = parsed["phase3_anchor_metrics"]
    for key, expected in (
        ("scope", "known-development-only"),
        ("development_only", True),
        ("final_family_access", False),
        ("final_method_selection", False),
    ):
        _require_equal(anchor.get(key), expected, f"Phase 3 anchor {key}")
    _require_equal(
        anchor.get("anchor_selection_metrics_sha256"),
        authority["phase3_anchor_metrics"].get("anchor_selection_metrics_sha256"),
        "Phase 3 anchor self-hash lineage",
    )
    anchor_conditions = _mapping(anchor.get("conditions"), "Phase 3 anchor conditions")
    if set(anchor_conditions) != {PHASE3_CONDITIONS[0], PHASE3_CONDITIONS[1]}:
        raise OutcomeDiagnosticProtocolError("Phase 3 anchor condition universe drifted")
    for condition_id, source_condition, historical in (
        (PHASE3_CONDITIONS[0], PHASE3_CONDITIONS[0], None),
        (
            PHASE3_CONDITIONS[1],
            "C-state-conditioned-listwise-optimum",
            "C-state-conditioned-listwise-optimum",
        ),
    ):
        row = _mapping(anchor_conditions[condition_id], f"Phase 3 anchor {condition_id}")
        _require_equal(row.get("selected_tuple_id"), "lr0p003-e120-t1p2", f"{condition_id} tuple")
        _require_equal(row.get("source_condition_id"), source_condition, f"{condition_id} source")
        _require_equal(row.get("historical_condition_id"), historical, f"{condition_id} history")
        _require_equal(
            row.get("cost", {}).get("unique_model_owner_artifacts"),
            30,
            f"{condition_id} model-owner coverage",
        )
        families = row.get("families")
        if not isinstance(families, list) or [item.get("family_id") for item in families] != list(
            FAMILIES
        ):
            raise OutcomeDiagnosticProtocolError(f"{condition_id} family order drifted")
        if any(item.get("units") != 40 for item in families):
            raise OutcomeDiagnosticProtocolError(f"{condition_id} family unit coverage drifted")
    _require_equal(
        anchor.get("metric_contract"),
        {
            "endpoint": 2048,
            "failure_sentinel": 2049,
            "family_aggregation": "within-family first, then equal family weight",
            "metric_id": "total_adaptation_actions_to_first_exact_optimum",
            "oracle_policy": "fixed batch and independent replay complete before reporting-only exact-optimum classification",
            "restricted_interactions": "accounting.probes.actions + accounting.search.actions",
            "success_tolerance_absolute": 0.05,
        },
        "Phase 3 anchor metric contract",
    )
    selection = parsed["phase3_development_selection"]
    _require_equal(
        selection.get("schema_version"),
        "milestone6.phase3.development-selection-lock.v1",
        "Phase 3 selection schema",
    )
    boundary = _mapping(selection.get("scientific_boundary"), "Phase 3 selection boundary")
    _require_equal(
        boundary,
        {
            "development_only": True,
            "final_family_access": False,
            "final_method_selection": False,
            "final_families_locked": True,
            "advancement_to_paired_objectives": False,
            "permitted_next_step": "predeclared development-only diagnostics; preserve B2 as the strong reference and do not unlock final families",
        },
        "Phase 3 selection boundary",
    )
    selection_self = _digest_field(
        selection.get("selection_lock_sha256"), "Phase 3 selection self-hash"
    )
    if (
        _digest({k: v for k, v in selection.items() if k != "selection_lock_sha256"})
        != selection_self
    ):
        raise OutcomeDiagnosticProtocolError("Phase 3 selection self-hash mismatch")
    _require_equal(
        selection_self,
        authority["phase3_development_selection"].get("selection_lock_sha256"),
        "Phase 3 selection lock lineage",
    )
    _require_equal(
        selection.get("analysis", {}).get("analysis_sha256"),
        authority["phase3_development_selection"].get("analysis_sha256"),
        "Phase 3 selection analysis lineage",
    )
    _require_equal(
        selection.get("matrix"),
        {
            "family_order": list(FAMILIES),
            "families": 6,
            "new_conditions": 4,
            "candidate_tuples_per_condition": 12,
            "units_per_candidate": 240,
            "units_per_family_candidate": 40,
            "units": 11520,
            "completed": 11520,
            "failed": 0,
            "interrupted": 0,
            "skipped": 0,
        },
        "Phase 3 selection matrix",
    )
    selected = _mapping(selection.get("selected"), "Phase 3 selected conditions")
    if set(selected) != set(PHASE3_CONDITIONS):
        raise OutcomeDiagnosticProtocolError("Phase 3 selected condition universe drifted")
    expected_tuples = {
        PHASE3_CONDITIONS[0]: "lr0p003-e120-t1p2",
        PHASE3_CONDITIONS[1]: "lr0p003-e120-t1p2",
        BASE_CONDITION: "lr0p01-e120-t1p2",
    }
    for condition_id, tuple_id in expected_tuples.items():
        row = _mapping(selected[condition_id], f"selected {condition_id}")
        _require_equal(row.get("candidate_tuple_id"), tuple_id, f"selected {condition_id} tuple")
        counts = row.get("family_success_counts")
        if (
            not isinstance(counts, list)
            or len(counts) != len(FAMILIES)
            or any(not isinstance(count, int) or not 0 <= count <= 40 for count in counts)
        ):
            raise OutcomeDiagnosticProtocolError(f"selected {condition_id} family coverage drifted")
        if condition_id in anchor_conditions:
            anchor_counts = [
                item["exact_optimum_success_count"]
                for item in anchor_conditions[condition_id]["families"]
            ]
            _require_equal(counts, anchor_counts, f"selected {condition_id} anchor parity")
    _require_equal(
        selection.get("authority", {}).get("protocol_sha256"),
        authority["phase3_protocol"].get("sha256"),
        "Phase 3 selection protocol lineage",
    )
    _require_equal(
        selection.get("authority", {}).get("plan_id"),
        authority["phase3_plan"].get("plan_id"),
        "Phase 3 selection plan lineage",
    )
    _require_equal(
        selection.get("authority", {}).get("model_authority_sha256"),
        authority["phase3_model_authority"].get("authority_sha256"),
        "Phase 3 selection model-authority lineage",
    )
    _require_equal(
        selection.get("authority", {}).get("anchor_selection_metrics_sha256"),
        authority["phase3_anchor_metrics"].get("anchor_selection_metrics_sha256"),
        "Phase 3 selection anchor lineage",
    )
    return tuple(rows)


def _validate_contract(
    body: Mapping[str, Any],
    protocol: Mapping[str, Any],
    authority_bytes: tuple[tuple[str, bytes], ...],
) -> None:
    rows = body.get("conditions")
    if (
        not isinstance(rows, list)
        or tuple(_mapping(row, "diagnostic condition").get("condition_id") for row in rows)
        != CONDITIONS
    ):
        raise OutcomeDiagnosticProtocolError("diagnostic condition IDs drifted")
    by_id = {row["condition_id"]: row for row in rows}
    expected_representations = {
        CONDITIONS[0]: {
            "input_width": 54,
            "retained_indices_per_summary_block": [0, 1, 2, 3, 6, 7, 8, 9, 11],
            "zeroed_indices_per_summary_block": [4, 5, 10],
            "added_to_S": [6, 7, 8, 9],
            "added_channels": [
                "resource_delta",
                "pressure_delta",
                "after_resource",
                "after_pressure",
            ],
            "state_indices": STATE_INDICES,
            "coverage_index": 53,
        },
        CONDITIONS[1]: {
            "input_width": 54,
            "retained_indices_per_summary_block": [0, 1, 2, 3, 4, 5, 10, 11],
            "zeroed_indices_per_summary_block": [6, 7, 8, 9],
            "added_to_S": [4, 5, 10],
            "added_channels": ["progress_delta", "elapsed_delta", "completed"],
            "state_indices": STATE_INDICES,
            "coverage_index": 53,
        },
    }
    masks: dict[str, set[int]] = {}
    for condition_id in CONDITIONS:
        row = _mapping(by_id[condition_id], condition_id)
        _require_equal(
            row.get("base_condition_id"), BASE_CONDITION, f"{condition_id} base condition"
        )
        _require_equal(row.get("trainable_parameters"), 3841, f"{condition_id} parameter count")
        rep = _mapping(row.get("representation"), f"{condition_id} representation")
        _require_equal(
            rep, expected_representations[condition_id], f"{condition_id} representation"
        )
        _require_equal(rep.get("input_width"), 54, f"{condition_id} input width")
        _require_equal(rep.get("state_indices"), STATE_INDICES, f"{condition_id} state indices")
        _require_equal(rep.get("coverage_index"), 53, f"{condition_id} coverage index")
        retained = rep.get("retained_indices_per_summary_block")
        zeroed = rep.get("zeroed_indices_per_summary_block")
        if (
            not isinstance(retained, list)
            or not isinstance(zeroed, list)
            or set(retained) & set(zeroed)
            or set(retained) | set(zeroed) != SUMMARY_INDICES
        ):
            raise OutcomeDiagnosticProtocolError(f"{condition_id} mask is not a partition")
        masks[condition_id] = set(retained)
    if (
        masks[CONDITIONS[0]] & masks[CONDITIONS[1]] != S_RETAINED
        or masks[CONDITIONS[0]] | masks[CONDITIONS[1]] != SUMMARY_INDICES
    ):
        raise OutcomeDiagnosticProtocolError(
            "diagnostic masks do not have the required S intersection/union"
        )
    _require_equal(masks[CONDITIONS[0]], S_RETAINED | {6, 7, 8, 9}, "S-RP retained channels")
    _require_equal(masks[CONDITIONS[1]], S_RETAINED | {4, 5, 10}, "S-PEC retained channels")
    _require_equal(
        body.get("scientific_question"),
        {
            "question": "Did the development Heat collapse arise from adding the resource/pressure outcome channels, the progress/elapsed/completion channels, or their interaction under the current pooled transition summary?",
            "claim_limit": "Only the effect of adding the named learner-visible measured channels to S under this exact representation, training, and search protocol may be discussed; this cannot retroactively establish the failed Phase 3 transition claim, justify pairing, or support final-family access.",
        },
        "diagnostic scientific question",
    )
    _require_equal(
        body.get("structural_controls"),
        {
            "same_source_examples": "derive S, S-RP, S-PEC, and T from one exact ordered tuple of optimum-imitation DecisionExample records; labels, candidate order, and example order are byte-identical before masking",
            "S_idempotence": "applying the S mask to an S tensor changes no byte",
            "mask_partition": "within every 12-channel summary block, S-RP intersection S-PEC equals S and S-RP union S-PEC equals the complete T block",
            "retained_byte_identity": "every retained current-state, summary, and coverage value equals the corresponding T value byte-for-byte; every excluded value is exactly zero",
            "architecture": "all new conditions use the unchanged StateConditionedScorer 54-to-48-to-24-to-1 MLP with exactly 3,841 trainable parameters",
            "strong_references": [
                "B2-global-listwise-optimum",
                BASE_CONDITION,
                "T-markov-state-transition-listwise-optimum",
            ],
            "reference_policy": "reuse locked metrics and exact lineage; do not retrain, rerun, restrict, or retune B2, S, or T",
        },
        "diagnostic structural controls",
    )
    _require_equal(
        body.get("candidate_tuples"), protocol.get("candidate_tuples"), "diagnostic candidate grid"
    )
    _require_equal(
        _mapping(body.get("development_matrix"), "development matrix"),
        {
            "family_order": list(FAMILIES),
            "fold_kind": "leave-one-family-out",
            "replicates": list(REPLICATES),
            "heldout_tasks_per_family": 8,
            "new_conditions": 2,
            "candidate_tuples_per_condition": 12,
            "units_per_condition_tuple": 240,
            "units_per_condition": 2880,
            "new_units": 5760,
            "canonical_evidence_artifacts_reused": 30,
            "new_views": 60,
            "training_tuples_per_view": 4,
            "new_model_owners": 240,
            "model_consumers_per_owner": 24,
        },
        "diagnostic matrix",
    )
    seeds = _mapping(body.get("data_and_seed_identity"), "seed identity")
    _require_equal(
        seeds.get("evidence"),
        "reuse the exact 30 Phase 3 evidence payloads and trajectory identities byte-for-byte; regeneration and new evidence probes are forbidden",
        "diagnostic evidence reuse",
    )
    _require_equal(
        seeds.get("examples"),
        "optimum-only listwise next-action examples with the exact S/T identities, labels, count, order, and batches",
        "diagnostic example identity",
    )
    _require_equal(
        seeds.get("condition_pairing"),
        "each new unit uses the exact S model, probe, search, data-order, environment, family, task, fold, replicate, and candidate-tuple seed identities for its matched unit",
        "diagnostic condition pairing",
    )
    _require_equal(
        seeds.get("seed_bases"),
        {"model": 6100000, "probe": 6200000, "search": 6300000, "data_order": 6400000},
        "diagnostic seed bases",
    )
    _require_equal(
        seeds.get("eligible_hyperparameters"),
        ["learning_rate", "training_epochs", "search_temperature"],
        "eligible hyperparameters",
    )
    _require_equal(
        seeds.get("ineligible_hyperparameters"),
        [
            "architecture",
            "hidden_widths",
            "weight_decay",
            "optimizer",
            "probe_budget",
            "search_budget",
            "interaction_budget",
            "seeds",
            "feature_mask",
            "training_objective",
        ],
        "ineligible hyperparameters",
    )
    fixed = _mapping(body.get("fixed_training_and_search"), "diagnostic fixed training/search")
    protocol_fixed = _mapping(
        protocol.get("fixed_training_and_search"), "Phase 3 fixed training/search"
    )
    for key, value in fixed.items():
        _require_equal(protocol_fixed.get(key), value, f"diagnostic fixed training/search {key}")
    _require_equal(
        fixed.get("objective"),
        "optimum-only listwise next-action imitation",
        "diagnostic objective",
    )
    _require_equal(
        fixed.get("oracle_timing"),
        "after the complete fixed candidate batch and independent replay",
        "diagnostic oracle timing",
    )
    selection = _mapping(body.get("selection_and_reporting"), "selection/reporting")
    if set(selection) != {
        "classification",
        "condition_selection",
        "primary_metric",
        "success_tolerance_absolute",
        "failure_sentinel",
        "tie_break_order",
        "matched_S_tuple",
        "required_reports",
        "changes_after_results",
    }:
        raise OutcomeDiagnosticProtocolError("diagnostic selection field set drifted")
    _require_equal(
        selection.get("classification"),
        "development-only exploratory diagnostic; not final method selection",
        "diagnostic selection classification",
    )
    _require_equal(
        selection.get("condition_selection"),
        "independently apply the unchanged Phase 3 rule to each new condition over the complete 12-tuple grid",
        "diagnostic condition selection",
    )
    _require_equal(
        selection.get("changes_after_results"), "forbidden", "diagnostic post-result changes"
    )
    _require_equal(selection.get("tie_break_order"), TIE_BREAK, "tie-break order")
    _require_equal(selection.get("matched_S_tuple"), "lr0p01-e120-t1p2", "matched S tuple")
    phase3_selection = _json(
        dict(authority_bytes)["phase3_development_selection"], "Phase 3 selection"
    )
    _require_equal(
        selection.get("matched_S_tuple"),
        phase3_selection["selected"][BASE_CONDITION]["candidate_tuple_id"],
        "matched S tuple authority",
    )
    _require_equal(
        selection.get("primary_metric"),
        "minimum_family_exact_optimum_success_rate",
        "diagnostic primary metric",
    )
    _require_equal(selection.get("failure_sentinel"), 2049, "diagnostic failure sentinel")
    _require_equal(selection.get("success_tolerance_absolute"), 0.05, "diagnostic tolerance")
    _require_equal(
        selection.get("required_reports"),
        [
            "all 12 tuple summaries",
            "independently selected tuple",
            "matched-S-tuple summary",
            "six family success counts",
            "exact rational deltas",
            "deduplicated model cost",
        ],
        "diagnostic required reports",
    )
    _require_equal(
        body.get("predeclared_diagnostic_availability"),
        UNAVAILABLE_DIAGNOSTICS,
        "diagnostic availability contract",
    )
    boundary = _mapping(body.get("execution_boundary"), "execution boundary")
    for key in (
        "final_family_access",
        "final_method_selection",
        "advancement_to_paired_objectives",
    ):
        _require_equal(boundary.get(key), False, f"diagnostic {key}")
    rules = _mapping(body.get("diagnostic_claim_rules"), "diagnostic claim rules")
    _require_equal(
        rules,
        {
            "robust_group_gain": "the independently selected condition and its matched-S tuple must each beat locked selected S minimum-family success by strictly more than 0.05, with no development-family success drop greater than 0.05 in either comparison",
            "robust_group_harm": "locked selected S must beat both the independently selected condition and its matched-S tuple by strictly more than 0.05",
            "within_tolerance": "absolute differences at or below 0.05 are inconclusive for a robust group effect",
            "interaction_diagnostic": "if both groups separately avoid robust harm but full T remains worse, record a possible group-interaction or optimization-burden hypothesis without a causal claim",
            "forbidden_conclusions": [
                "retroactive Phase 3 transition claim",
                "general causal feature importance",
                "advancement to paired objectives",
                "final method selection",
                "final family unlock",
            ],
        },
        "diagnostic claim rules",
    )


def load_outcome_group_diagnostic_protocol(
    path: str | os.PathLike[str] = PHASE3_OUTCOME_DIAGNOSTIC_PROTOCOL_PATH,
    *,
    repository: str | os.PathLike[str] = ROOT,
) -> OutcomeDiagnosticProtocolSnapshot:
    """Load and validate the complete, development-only diagnostic contract."""

    raw_repository = Path(repository)
    if raw_repository.is_symlink():
        raise OutcomeDiagnosticProtocolError("diagnostic repository root cannot be a symlink")
    repo = raw_repository.resolve(strict=True)
    target = Path(path)
    if not target.is_absolute():
        target = repo / target
    if target.is_symlink() or not target.is_file():
        raise OutcomeDiagnosticProtocolError(
            "diagnostic protocol must be a regular non-symlink file"
        )
    resolved_target = target.resolve(strict=True)
    try:
        resolved_target.relative_to(repo)
    except ValueError as exc:
        raise OutcomeDiagnosticProtocolError(
            "diagnostic protocol path escapes the repository"
        ) from exc
    try:
        parent_fd = secure_fs.open_directory_chain(target.parent)
        try:
            content = secure_fs.read_bytes_at(parent_fd, target.name)
        finally:
            os.close(parent_fd)
    except (OSError, RuntimeError, TypeError, ValueError, secure_fs.SecureFilesystemError) as exc:
        raise OutcomeDiagnosticProtocolError("diagnostic protocol cannot be read safely") from exc
    body = _json(content, "diagnostic protocol")
    _require_equal(body.get("schema_version"), SCHEMA_VERSION, "diagnostic schema")
    _require_equal(body.get("status"), STATUS, "diagnostic status")
    _require_equal(body.get("scope"), "known-development-only", "diagnostic scope")
    _validate_self_hash(body)
    freeze = _mapping(body.get("freeze_record"), "freeze record")
    _require_equal(
        freeze,
        {
            "frozen_at_local_date": "2026-08-24",
            "source_result_lock_commit_sha": SOURCE_RESULT_LOCK_COMMIT,
            "phase3_comparative_results_used_for_diagnosis": True,
            "outcome_group_results_inspected": False,
            "classification": "post-hoc exploratory development diagnostic",
            "reason": "S was competitive with B2 while T and H4 collapsed on development Heat; isolate which measured outcome-channel group changes the S representation without changing data, architecture, objective, or budget",
        },
        "diagnostic freeze record",
    )
    _require_equal(
        body.get("execution_boundary"),
        {
            "separate_from_phase3_store": True,
            "new_inert_result_namespace": True,
            "phase3_artifacts_immutable": True,
            "complete_matrix_before_comparative_read": True,
            "clean_exact_commit_readiness_required": True,
            "runtime_authority_requirement": "before store preparation or execution, bind an authorized post-freeze git commit and immutable protocol/authority bytes plus descriptor identities; reject a symlinked repository or output root and revalidate every byte and identity immediately before execution",
            "final_family_access": False,
            "final_method_selection": False,
            "advancement_to_paired_objectives": False,
        },
        "diagnostic execution boundary",
    )
    authority_bytes = _validate_authorities(body, repo)
    protocol = _json(dict(authority_bytes)["phase3_protocol"], "Phase 3 protocol")
    _validate_contract(body, protocol, authority_bytes)
    if (
        body.get("diagnostic_protocol_sha256") != EXPECTED_PROTOCOL_SELF_HASH
        or _sha256(content) != EXPECTED_PROTOCOL_FILE_SHA256
    ):
        raise OutcomeDiagnosticProtocolError("diagnostic protocol immutable identity changed")
    return OutcomeDiagnosticProtocolSnapshot(
        repo, resolved_target, content, _sha256(content), body, authority_bytes
    )


__all__ = [
    "BASE_CONDITION",
    "CONDITIONS",
    "OutcomeDiagnosticProtocolError",
    "OutcomeDiagnosticProtocolSnapshot",
    "PHASE3_OUTCOME_DIAGNOSTIC_PROTOCOL_PATH",
    "load_outcome_group_diagnostic_protocol",
]
