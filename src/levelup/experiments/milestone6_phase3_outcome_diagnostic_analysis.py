"""Descriptor-pinned publisher for the development-only outcome diagnostic.

This module is intentionally a terminal, read-only boundary.  It captures the
already activated diagnostic namespace, loads the complete matrix through held
descriptors, invokes only the pure reducer, and publishes one self-hashed JSON
artifact outside the result root.  No evaluator, oracle, final family, or
comparative result loader is reachable from here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import stat
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from levelup.experiments.milestone6_phase3_anchor_selection_metrics import (
    FAMILIES as ANCHOR_FAMILIES,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
    load_outcome_model_artifact_authority_bytes,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    bind_pinned_outcome_diagnostic_plan,
    build_outcome_group_diagnostic_plan_from_pinned_snapshot,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    CONDITIONS,
    TIE_BREAK,
    UNAVAILABLE_DIAGNOSTICS,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_readiness import (
    OutcomeDiagnosticAnalysisReadinessSnapshot,
    OutcomeDiagnosticModelReadinessSnapshot,
    OutcomeDiagnosticReadinessSnapshot,
    capture_outcome_group_diagnostic_analysis_readiness,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_reducer import (
    EXPECTED_TUPLES,
    EXPECTED_UNITS,
    MATCHED_S_TUPLE,
    OutcomeDiagnosticCandidateMetric,
    OutcomeDiagnosticConditionSelection,
    OutcomeDiagnosticFamilyMetric,
    OutcomeDiagnosticLockedFamilyMetric,
    OutcomeDiagnosticLockedMetric,
    OutcomeDiagnosticReducerError,
    OutcomeDiagnosticSelectionResult,
    evaluate_outcome_diagnostic_claims,
    reduce_outcome_group_diagnostic,
    validate_outcome_diagnostic_locked_metric,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_result_store import (
    OutcomeDiagnosticExpectedPlan,
    OutcomeDiagnosticResultStore,
    OutcomeDiagnosticResumeBaseline,
    activate_outcome_diagnostic_result_stores,
    build_outcome_diagnostic_expected_plan,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import UnitRecord

SCHEMA_VERSION = "milestone6.phase3.outcome-group-diagnostic-analysis.v1"
PHASE3_SELECTION_PATH = "configs/milestone6/phase3_development_selection.json"
ANCHOR_METRICS_PATH = "configs/milestone6/phase3_anchor_selection_metrics.json"
MODEL_AUTHORITY_PATH = "configs/milestone6/phase3_outcome_model_artifact_authority.json"


class OutcomeDiagnosticAnalysisError(ValueError):
    """Raised when the complete diagnostic cannot be safely reduced or published."""


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fraction(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _parse_fraction(value: object, label: str) -> Fraction:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"numerator", "denominator"}
        or type(value.get("numerator")) is not int
        or type(value.get("denominator")) is not int
        or value["denominator"] <= 0
    ):
        _fail(f"{label} is not an exact rational")
    parsed = Fraction(value["numerator"], value["denominator"])
    if _fraction(parsed) != value:
        _fail(f"{label} is not reduced")
    return parsed


def _summary_parts(
    value: object, *, condition_id: str, allow_missing_medians: bool
) -> tuple[
    str,
    str,
    tuple[OutcomeDiagnosticLockedFamilyMetric, ...],
    Fraction,
    Fraction,
    Fraction,
    int,
    int,
    int,
]:
    expected_keys = {
        "condition_id",
        "tuple_id",
        "training_tuple_id",
        "family_metrics",
        "minimum_family_success_rate",
        "worst_family_median_restricted_interactions",
        "macro_average_family_median_restricted_interactions",
        "deduplicated_model_cost",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        _fail("metric summary schema differs")
    tuple_id = value.get("tuple_id")
    training_tuple_id = value.get("training_tuple_id")
    families = value.get("family_metrics")
    cost = value.get("deduplicated_model_cost")
    if (
        value.get("condition_id") != condition_id
        or tuple_id not in EXPECTED_TUPLES
        or training_tuple_id != str(tuple_id).rsplit("-t", 1)[0]
        or not isinstance(families, list)
        or len(families) != len(ANCHOR_FAMILIES)
        or not isinstance(cost, Mapping)
        or set(cost) != {"optimizer_steps", "forward_passes", "recurrent_steps"}
        or any(type(cost.get(key)) is not int or cost[key] < 0 for key in cost)
    ):
        _fail("metric summary identity, family, or cost differs")
    parsed_families: list[OutcomeDiagnosticLockedFamilyMetric] = []
    for family_id, row in zip(ANCHOR_FAMILIES, families, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "family_id",
                "units",
                "successes",
                "success_rate",
                "median_restricted_interactions",
            }
            or row.get("family_id") != family_id
            or row.get("units") != 40
            or type(row.get("successes")) is not int
            or not 0 <= row["successes"] <= 40
        ):
            _fail("metric summary family row differs")
        success_rate = _parse_fraction(row.get("success_rate"), "family success rate")
        if success_rate != Fraction(row["successes"], 40):
            _fail("metric summary family success count/rate differs")
        raw_median = row.get("median_restricted_interactions")
        median = (
            None
            if raw_median is None and allow_missing_medians
            else _parse_fraction(raw_median, "family median")
        )
        parsed_families.append(
            OutcomeDiagnosticLockedFamilyMetric(
                family_id, 40, row["successes"], success_rate, median
            )
        )
    return (
        str(tuple_id),
        str(training_tuple_id),
        tuple(parsed_families),
        _parse_fraction(value.get("minimum_family_success_rate"), "minimum rate"),
        _parse_fraction(
            value.get("worst_family_median_restricted_interactions"), "worst median"
        ),
        _parse_fraction(
            value.get("macro_average_family_median_restricted_interactions"),
            "macro median",
        ),
        cost["optimizer_steps"],
        cost["forward_passes"],
        cost["recurrent_steps"],
    )


def _parse_candidate_summary(
    value: object, condition_id: str
) -> OutcomeDiagnosticCandidateMetric:
    parts = _summary_parts(
        value, condition_id=condition_id, allow_missing_medians=False
    )
    families = tuple(
        OutcomeDiagnosticFamilyMetric(
            row.family_id,
            row.units,
            row.successes,
            row.success_rate,
            row.median_restricted_interactions,  # type: ignore[arg-type]
        )
        for row in parts[2]
    )
    return OutcomeDiagnosticCandidateMetric(
        condition_id, parts[0], parts[1], families, *parts[3:]
    )


def _parse_locked_summary(
    value: object, *, condition_id: str, tuple_id: str, allow_missing_medians: bool
) -> OutcomeDiagnosticLockedMetric:
    parts = _summary_parts(
        value, condition_id=condition_id, allow_missing_medians=allow_missing_medians
    )
    metric = OutcomeDiagnosticLockedMetric(
        condition_id, parts[0], parts[1], parts[2], *parts[3:]
    )
    return _validate_locked_metric(
        metric, condition_id=condition_id, tuple_id=tuple_id
    )


def _validate_locked_metric(
    metric: OutcomeDiagnosticLockedMetric, *, condition_id: str, tuple_id: str
) -> OutcomeDiagnosticLockedMetric:
    """Keep reducer validation failures inside the publisher error boundary."""

    try:
        return validate_outcome_diagnostic_locked_metric(
            metric, condition_id=condition_id, tuple_id=tuple_id
        )
    except OutcomeDiagnosticReducerError as exc:
        raise OutcomeDiagnosticAnalysisError(str(exc)) from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, Fraction):
        return _fraction(value)
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "__dataclass_fields__"):
        return {name: _jsonable(getattr(value, name)) for name in value.__dataclass_fields__ if not name.startswith("_")}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _fail(message: str, exc: BaseException | None = None) -> None:
    if exc is None:
        raise OutcomeDiagnosticAnalysisError(message)
    raise OutcomeDiagnosticAnalysisError(message) from exc


def _base_snapshot(snapshot: Any) -> OutcomeDiagnosticReadinessSnapshot:
    return (
        snapshot.base
        if isinstance(
            snapshot,
            (
                OutcomeDiagnosticAnalysisReadinessSnapshot,
                OutcomeDiagnosticModelReadinessSnapshot,
            ),
        )
        else snapshot
    )


def _pinned_json(snapshot: Any, path: str, *, canonical: bool = True) -> dict[str, Any]:
    base = _base_snapshot(snapshot)
    source = base.files_by_path.get(path)
    if source is None:
        _fail(f"pinned authority is missing: {path}")
    try:
        body = json.loads(source.content)
    except (TypeError, ValueError, UnicodeError) as exc:
        _fail(f"pinned authority is invalid JSON: {path}", exc)
    if not isinstance(body, dict) or (canonical and source.content not in (
        canonical_json_bytes(body), canonical_json_bytes(body) + b"\n"
    )):
        _fail(f"pinned authority is not canonical JSON: {path}")
    return body


def _metric_from_row(condition: str, row: Mapping[str, Any]) -> OutcomeDiagnosticLockedMetric:
    families = row.get("families")
    if not isinstance(families, list) or tuple(item.get("family_id") for item in families if isinstance(item, Mapping)) != ANCHOR_FAMILIES:
        _fail(f"locked metric family order is invalid: {condition}")
    metrics = tuple(
        OutcomeDiagnosticLockedFamilyMetric(
            family_id=str(item["family_id"]),
            units=int(item["units"]),
            successes=int(item["exact_optimum_success_count"]),
            success_rate=Fraction(str(item["exact_optimum_success_rate"])),
            median_restricted_interactions=Fraction(str(item["median_restricted_interactions"])),
        )
        for item in families
    )
    cost = row.get("cost")
    if not isinstance(cost, Mapping):
        _fail(f"locked metric cost is missing: {condition}")
    metric = OutcomeDiagnosticLockedMetric(
        condition_id=condition,
        tuple_id=str(row["selected_tuple_id"]),
        training_tuple_id=str(row["training_tuple_id"]),
        family_metrics=metrics,
        minimum_family_success_rate=min(item.success_rate for item in metrics),
        worst_family_median_restricted_interactions=max(item.median_restricted_interactions for item in metrics),
        macro_average_family_median_restricted_interactions=sum((item.median_restricted_interactions for item in metrics), Fraction()) / len(metrics),
        optimizer_steps=int(cost["optimizer_steps"]),
        forward_passes=int(cost["forward_passes"]),
        recurrent_steps=int(cost.get("recurrent_steps", 0)),
    )
    if (
        Fraction(str(row.get("minimum_family_exact_optimum_success_rate")))
        != metric.minimum_family_success_rate
        or Fraction(str(row.get("worst_family_median_restricted_interactions")))
        != metric.worst_family_median_restricted_interactions
        or not math.isclose(
            float(row.get("macro_average_family_median_restricted_interactions")),
            float(metric.macro_average_family_median_restricted_interactions),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or cost.get("unique_model_owner_artifacts") != 30
    ):
        _fail(f"locked metric aggregate or owner count differs: {condition}")
    return metric


def _metric_from_selection_row(
    condition: str, row: Mapping[str, Any]
) -> OutcomeDiagnosticLockedMetric:
    """Decode the compact Phase 3 selection-lock row (which has no family objects)."""
    counts = row.get("family_success_counts")
    if not isinstance(counts, list) or len(counts) != len(ANCHOR_FAMILIES):
        _fail("locked selection family counts are incomplete")
    def frac(value: Any) -> Fraction:
        if isinstance(value, Mapping):
            return Fraction(int(value["numerator"]), int(value["denominator"]))
        return Fraction(str(value))
    minimum = frac(row.get("minimum_family_success_rate"))
    macro = frac(row.get("macro_family_median_restricted_interactions"))
    worst = frac(row.get("worst_family_median_restricted_interactions"))
    tuple_id = str(row.get("candidate_tuple_id"))
    training = tuple_id.rsplit("-t", 1)[0]
    cost = row.get("owner_cost")
    if not isinstance(cost, Mapping):
        _fail("locked selection owner cost is missing")
    families = tuple(
        OutcomeDiagnosticLockedFamilyMetric(
            family, 40, int(count), Fraction(int(count), 40), None
        )
        for family, count in zip(ANCHOR_FAMILIES, counts, strict=True)
    )
    return OutcomeDiagnosticLockedMetric(
        condition, tuple_id, training, families, minimum, worst, macro,
        int(cost["optimizer_steps"]), int(cost["forward_passes"]), int(cost.get("recurrent_steps", 0)),
    )


def _locked_references(snapshot: Any) -> tuple[
    OutcomeDiagnosticLockedMetric,
    OutcomeDiagnosticLockedMetric,
    OutcomeDiagnosticLockedMetric,
    dict[str, Any],
    dict[str, Any],
]:
    selection = _pinned_json(snapshot, PHASE3_SELECTION_PATH, canonical=False)
    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        _fail("locked Phase 3 selection is missing selected metrics")
    s = _metric_from_selection_row("S-state-availability-listwise-optimum", selected["S-state-availability-listwise-optimum"])
    anchor = _pinned_json(snapshot, ANCHOR_METRICS_PATH)
    conditions = anchor.get("conditions")
    if not isinstance(conditions, Mapping):
        _fail("locked anchor metrics are missing conditions")
    b2 = _metric_from_row("B2-global-listwise-optimum", conditions["B2-global-listwise-optimum"])
    t = _metric_from_row("T-markov-state-transition-listwise-optimum", conditions["T-markov-state-transition-listwise-optimum"])
    _validate_locked_metric(
        s,
        condition_id="S-state-availability-listwise-optimum",
        tuple_id=MATCHED_S_TUPLE,
    )
    _validate_locked_metric(
        b2,
        condition_id="B2-global-listwise-optimum",
        tuple_id="lr0p003-e120-t1p2",
    )
    _validate_locked_metric(
        t,
        condition_id="T-markov-state-transition-listwise-optimum",
        tuple_id="lr0p003-e120-t1p2",
    )
    return s, b2, t, selection, anchor


def _require_activated_resume_baseline(
    snapshot: Any, expected: OutcomeDiagnosticExpectedPlan
) -> OutcomeDiagnosticResumeBaseline:
    """Return only the descriptor-captured stores for an activated tree.

    Activated output already contains the durable runtime marker and completed
    records.  The inert result-store loader intentionally rejects that marker,
    so analysis must not reconstruct stores from paths.  Readiness has already
    inspected the tree through held descriptors and captured typed store
    objects in the resume baseline; this gate binds those objects to the exact
    current expected matrix before activation revalidates marker and record
    bytes under its own held descriptors.
    """

    base = _base_snapshot(snapshot)
    baseline = base.resume_baseline
    expected_baseline = base.resume_expected_plan
    if type(baseline) is not OutcomeDiagnosticResumeBaseline:
        _fail("activated output lacks a typed resume baseline")
    if type(expected_baseline) is not OutcomeDiagnosticExpectedPlan:
        _fail("activated output lacks a typed resume expected plan")
    if expected_baseline != expected:
        _fail("resume expected plan differs from the current frozen matrix")
    if (
        baseline.output_state != "activated"
        or baseline.output_state != base.output_state
        or baseline.output_root != base.output_root
        or baseline.output_root_identity != base.output_root_identity
    ):
        _fail("activated resume baseline state or root differs from readiness")
    stores = tuple(baseline.stores)
    expected_stores = tuple(expected.stores)
    if (
        len(stores) != len(ANCHOR_FAMILIES)
        or len(expected_stores) != len(ANCHOR_FAMILIES)
        or tuple(store.family_id for store in stores) != ANCHOR_FAMILIES
        or any(type(store) is not OutcomeDiagnosticResultStore for store in stores)
        or tuple(store.spec for store in stores) != expected_stores
    ):
        _fail("activated resume stores differ from the frozen expected matrix")
    return baseline


def _candidate_summary(candidate: Any) -> dict[str, Any]:
    return {
        "condition_id": candidate.condition_id,
        "tuple_id": candidate.tuple_id,
        "training_tuple_id": candidate.training_tuple_id,
        "family_metrics": [
            {
                "family_id": row.family_id,
                "units": row.units,
                "successes": row.successes,
                "success_rate": _fraction(row.success_rate),
                "median_restricted_interactions": (
                    None
                    if row.median_restricted_interactions is None
                    else _fraction(row.median_restricted_interactions)
                ),
            }
            for row in candidate.family_metrics
        ],
        "minimum_family_success_rate": _fraction(candidate.minimum_family_success_rate),
        "worst_family_median_restricted_interactions": _fraction(candidate.worst_family_median_restricted_interactions),
        "macro_average_family_median_restricted_interactions": _fraction(candidate.macro_average_family_median_restricted_interactions),
        "deduplicated_model_cost": {
            "optimizer_steps": candidate.optimizer_steps,
            "forward_passes": candidate.forward_passes,
            "recurrent_steps": candidate.recurrent_steps,
        },
    }


def _metric_summary(metric: OutcomeDiagnosticLockedMetric) -> dict[str, Any]:
    return _candidate_summary(metric)


def _build_report(snapshot: Any, batch: Any) -> dict[str, Any]:
    try:
        protocol = snapshot.protocol
        plan = bind_pinned_outcome_diagnostic_plan(
            build_outcome_group_diagnostic_plan_from_pinned_snapshot(protocol),
            snapshot=protocol,
        )
        base = _base_snapshot(snapshot)
        authority_source = getattr(snapshot, "authority_file", None)
        if authority_source is None:
            authority_source = base.files_by_path.get(MODEL_AUTHORITY_PATH)
        if authority_source is None:
            _fail("pinned model authority is missing")
        authority = load_outcome_model_artifact_authority_bytes(authority_source.content)
        expected = build_outcome_diagnostic_expected_plan(plan, protocol)
        s, b2, t, selection_lock, anchor = _locked_references(snapshot)
        records: list[UnitRecord] = []
        expected_ids = tuple(item.unit_id for item in expected.units)
        base = _base_snapshot(snapshot)
        batch.validate_existing_records_against_resume_baseline(
            base.resume_baseline
        )
        completed_ids = tuple(batch.completed_unit_ids())
        if len(completed_ids) != EXPECTED_UNITS or set(completed_ids) != set(expected_ids):
            _fail("diagnostic matrix is incomplete or contains foreign units")
        if batch.attempt_records():
            _fail("diagnostic matrix contains attempt records")
        for planned in expected.units:
            store = batch.store_for_family(planned.heldout_family)
            record = store.load_completed(planned.unit_id)
            if type(record) is not UnitRecord:
                _fail("diagnostic completed record is missing")
            records.append(record)
        batch._require_live(validate_records=True)
        matrix, result, claims = reduce_outcome_group_diagnostic(plan, authority, records, locked_s=s, locked_t=t)
        record_manifest = batch.records_manifest()
        runtime_lineage = batch.runtime_lineage()
        traces = {
            row.condition_id: {
                "best_minimum_family_success_rate": _fraction(row.best_minimum_family_success_rate),
                "retained_tuple_ids": list(row.retained_tuple_ids),
                "selected_tuple_id": row.selected.tuple_id,
                "candidates": [_candidate_summary(item) for item in row.candidates],
                "matched_S_tuple": _candidate_summary(next(item for item in row.candidates if item.tuple_id == MATCHED_S_TUPLE)),
            }
            for row in result.condition_selections
        }
        claim_body = _jsonable(claims)
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "scope": "known-development-only",
            "development_only": True,
            "final_family_access": False,
            "final_method_selection": False,
            "advancement_to_paired_objectives": False,
            "pairing_claim": False,
            "matrix": {"unit_count": matrix.unit_count, "family_counts": dict(matrix.cost.family_counts), "condition_counts": dict(matrix.cost.condition_counts)},
            "selection_traces": traces,
            "locked_references": {"S": _metric_summary(s), "B2": _metric_summary(b2), "T": _metric_summary(t)},
            "claims": claim_body,
            "forbidden_claims": {"transition": False, "history": False, "sequence": False, "pairing": False, "final_method_selection": False, "final_family_unlock": False},
            "cost": _jsonable(matrix.cost),
            "metric_contract": {"primary_metric": "minimum_family_exact_optimum_success_rate", "success_tolerance": _fraction(Fraction(1, 20)), "failure_censoring_budget": 2048, "failure_sentinel": 2049, "tie_break_order": list(protocol.payload["selection_and_reporting"]["tie_break_order"])},
            "unavailable_diagnostics": dict(protocol.payload.get("predeclared_diagnostic_availability", {})),
            "lineage": {
                "git_commit_sha": base.git_commit_sha,
                "protocol_sha256": protocol.sha256,
                "protocol_file_sha256": protocol.sha256,
                "protocol_self_sha256": protocol.payload.get("diagnostic_protocol_sha256"),
                "protocol_authority_file_hashes": {
                    name: source.get("sha256")
                    for name, source in protocol.payload.get("authority", {}).items()
                    if isinstance(source, Mapping)
                },
                "plan_id": plan.plan.plan_id,
                "model_authority_sha256": authority.authority_sha256,
                "model_authority_self_sha256": authority.expected_authority_sha256,
                "model_authority_file_sha256": _sha(authority_source.content),
                "model_preparation_git_commit_sha": authority.preparation_git_commit_sha,
                "model_preparation_provenance_sha256": authority.preparation_provenance_sha256,
                "source_result_lock_commit_sha": base.source_result_lock_commit_sha,
                "selection_lock_sha256": _sha(base.files_by_path[PHASE3_SELECTION_PATH].content),
                "selection_lock_self_sha256": selection_lock.get("selection_lock_sha256"),
                "selection_analysis_sha256": selection_lock.get("analysis", {}).get("analysis_sha256"),
                "anchor_selection_metrics_sha256": _sha(base.files_by_path[ANCHOR_METRICS_PATH].content),
                "anchor_selection_metrics_self_sha256": anchor.get("anchor_selection_metrics_sha256"),
                "activation_marker_sha256": runtime_lineage[
                    "activation_marker_sha256"
                ],
                "activation_marker_identity": runtime_lineage[
                    "activation_marker_identity"
                ],
                "root_identity": runtime_lineage["root_identity"],
                "stores": runtime_lineage["stores"],
                "records_manifest_sha256": _sha(
                    canonical_json_bytes(record_manifest)
                ),
                "records_manifest_count": len(record_manifest),
            },
            "selection_lock": {"sha256": _sha(base.files_by_path[PHASE3_SELECTION_PATH].content), "selected_S_tuple_id": s.tuple_id},
            "anchor_lock": {"sha256": _sha(base.files_by_path[ANCHOR_METRICS_PATH].content), "selected_B2_tuple_id": b2.tuple_id, "selected_T_tuple_id": t.tuple_id},
        }
        return report
    except OutcomeDiagnosticAnalysisError:
        raise
    except Exception as exc:
        _fail("diagnostic reduction failed closed", exc)


def build_outcome_group_diagnostic_analysis(
    snapshot: OutcomeDiagnosticReadinessSnapshot,
    *,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Reduce one complete activated namespace using pinned bytes and descriptors."""
    if not isinstance(
        snapshot,
        (
            OutcomeDiagnosticReadinessSnapshot,
            OutcomeDiagnosticAnalysisReadinessSnapshot,
            OutcomeDiagnosticModelReadinessSnapshot,
        ),
    ):
        _fail("canonical readiness snapshot is required")
    base = _base_snapshot(snapshot)
    if base.output_state != "activated":
        _fail("analysis requires an activated diagnostic namespace")
    snapshot.preflight(expected_git_commit=expected_git_commit)
    protocol = snapshot.protocol
    plan = bind_pinned_outcome_diagnostic_plan(
        build_outcome_group_diagnostic_plan_from_pinned_snapshot(protocol),
        snapshot=protocol,
    )
    expected = build_outcome_diagnostic_expected_plan(plan, protocol)
    with base.hold_for_activation(expected_git_commit=expected_git_commit) as lease:
        baseline = _require_activated_resume_baseline(snapshot, expected)
        with activate_outcome_diagnostic_result_stores(
            tuple(baseline.stores),
            expected,
            lease,
            expected_git_commit=expected_git_commit,
            validate_existing_records=False,
        ) as batch:
            report = _build_report(snapshot, batch)
            lease.require_active()
            batch._require_live(validate_records=True)
            return report


def _reject_output_target(target: Path, result_root: Path) -> None:
    for item in (target.parent, *target.parent.parents):
        try:
            if stat.S_ISLNK(item.lstat().st_mode):
                _fail("analysis output parent contains a symlink")
        except FileNotFoundError:
            continue
    if target == result_root or result_root in target.parents:
        _fail("analysis output must be outside result root")
    try:
        root_stat = result_root.stat()
        root_identity = (root_stat.st_dev, root_stat.st_ino)
        for item in (target.parent, *target.parent.parents):
            try:
                item_stat = item.stat()
            except FileNotFoundError:
                continue
            if (item_stat.st_dev, item_stat.st_ino) == root_identity:
                _fail("analysis output physically resolves inside result root")
    except OutcomeDiagnosticAnalysisError:
        raise
    except OSError as exc:
        _fail("analysis output containment cannot be verified", exc)
    if os.path.lexists(target):
        _fail("analysis output already exists")


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and bool(value)
        and set(value) != {"0"}
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_descriptor_identity(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(type(item) is int and item >= 0 for item in value)
        and value != [0, 0]
    )


def _validate_runtime_lineage(lineage: Mapping[str, Any]) -> None:
    expected_keys = {
        "git_commit_sha",
        "protocol_sha256",
        "protocol_file_sha256",
        "protocol_self_sha256",
        "protocol_authority_file_hashes",
        "plan_id",
        "model_authority_sha256",
        "model_authority_self_sha256",
        "model_authority_file_sha256",
        "model_preparation_git_commit_sha",
        "model_preparation_provenance_sha256",
        "source_result_lock_commit_sha",
        "selection_lock_sha256",
        "selection_lock_self_sha256",
        "selection_analysis_sha256",
        "anchor_selection_metrics_sha256",
        "anchor_selection_metrics_self_sha256",
        "activation_marker_sha256",
        "activation_marker_identity",
        "root_identity",
        "stores",
        "records_manifest_sha256",
        "records_manifest_count",
    }
    digest_keys = expected_keys - {
        "git_commit_sha",
        "protocol_authority_file_hashes",
        "model_preparation_git_commit_sha",
        "source_result_lock_commit_sha",
        "activation_marker_identity",
        "root_identity",
        "stores",
        "records_manifest_count",
    }
    commit_keys = {
        "git_commit_sha",
        "model_preparation_git_commit_sha",
        "source_result_lock_commit_sha",
    }
    authority_hashes = lineage.get("protocol_authority_file_hashes")
    expected_authority_keys = {
        "phase3_protocol",
        "phase3_plan",
        "phase3_evidence",
        "phase3_model_authority",
        "phase3_anchor_metrics",
        "phase3_development_selection",
    }
    stores = lineage.get("stores")
    root_identity = lineage.get("root_identity")
    if (
        set(lineage) != expected_keys
        or any(not _is_lower_hex(lineage.get(key), 64) for key in digest_keys)
        or any(not _is_lower_hex(lineage.get(key), 40) for key in commit_keys)
        or lineage.get("protocol_file_sha256") != lineage.get("protocol_sha256")
        or lineage.get("model_authority_self_sha256")
        != lineage.get("model_authority_sha256")
        or not isinstance(authority_hashes, Mapping)
        or set(authority_hashes) != expected_authority_keys
        or any(not _is_lower_hex(value, 64) for value in authority_hashes.values())
        or not _is_descriptor_identity(lineage.get("activation_marker_identity"))
        or not _is_descriptor_identity(root_identity)
        or not isinstance(stores, list)
        or len(stores) != len(ANCHOR_FAMILIES)
        or lineage.get("records_manifest_count") != EXPECTED_UNITS
    ):
        _fail("analysis runtime lineage schema or authority identities differ")

    expected_identity_keys = {"root", "family", "run", "namespaces", "records"}
    for family, store in zip(ANCHOR_FAMILIES, stores, strict=True):
        if not isinstance(store, Mapping) or set(store) != {
            "family_id",
            "run_id",
            "config_sha256",
            "identities",
        }:
            _fail("analysis runtime store lineage schema differs")
        identities = store.get("identities")
        if (
            store.get("family_id") != family
            or not _is_lower_hex(store.get("run_id"), 64)
            or not _is_lower_hex(store.get("config_sha256"), 64)
            or not isinstance(identities, Mapping)
            or set(identities) != expected_identity_keys
            or identities.get("root") != root_identity
            or any(
                not _is_descriptor_identity(identities.get(key))
                for key in ("root", "family", "run", "namespaces")
            )
        ):
            _fail("analysis runtime store authority or directory identity differs")
        records = identities.get("records")
        if (
            not isinstance(records, Mapping)
            or set(records) != set(CONDITIONS)
            or any(
                not _is_descriptor_identity(records.get(condition))
                for condition in CONDITIONS
            )
        ):
            _fail("analysis runtime record namespace identities differ")


def validate_outcome_group_diagnostic_analysis_bytes(content: bytes) -> dict[str, Any]:
    """Validate canonical, self-hashed, development-only analysis bytes."""

    try:
        body = json.loads(content)
    except (TypeError, ValueError, UnicodeError) as exc:
        _fail("analysis artifact is not valid JSON", exc)
    expected_top_level = {
        "schema_version",
        "scope",
        "development_only",
        "final_family_access",
        "final_method_selection",
        "advancement_to_paired_objectives",
        "pairing_claim",
        "matrix",
        "selection_traces",
        "locked_references",
        "claims",
        "forbidden_claims",
        "cost",
        "metric_contract",
        "unavailable_diagnostics",
        "lineage",
        "selection_lock",
        "anchor_lock",
        "analysis_sha256",
    }
    if (
        not isinstance(body, dict)
        or set(body) != expected_top_level
        or content != canonical_json_bytes(body) + b"\n"
        or body.get("schema_version") != SCHEMA_VERSION
    ):
        _fail("analysis artifact is not canonical or has the wrong schema")
    supplied = body.get("analysis_sha256")
    unsigned = {key: value for key, value in body.items() if key != "analysis_sha256"}
    if not isinstance(supplied, str) or supplied != _sha(canonical_json_bytes(unsigned)):
        _fail("analysis artifact self-hash differs")
    matrix = body.get("matrix")
    lineage = body.get("lineage")
    forbidden = body.get("forbidden_claims")
    traces = body.get("selection_traces")
    locked = body.get("locked_references")
    cost = body.get("cost")
    claims = body.get("claims")
    expected_families = {family: 960 for family in ANCHOR_FAMILIES}
    expected_conditions = {condition: 2_880 for condition in CONDITIONS}
    if (
        body.get("scope") != "known-development-only"
        or body.get("development_only") is not True
        or body.get("final_family_access") is not False
        or body.get("final_method_selection") is not False
        or body.get("advancement_to_paired_objectives") is not False
        or body.get("pairing_claim") is not False
        or not isinstance(matrix, Mapping)
        or matrix.get("unit_count") != EXPECTED_UNITS
        or matrix.get("family_counts") != expected_families
        or matrix.get("condition_counts") != expected_conditions
        or not isinstance(lineage, Mapping)
        or lineage.get("records_manifest_count") != EXPECTED_UNITS
        or not isinstance(traces, Mapping)
        or set(traces) != set(CONDITIONS)
        or not isinstance(locked, Mapping)
        or set(locked) != {"S", "B2", "T"}
        or not isinstance(cost, Mapping)
        or cost.get("unit_count") != EXPECTED_UNITS
        or cost.get("model_owner_count") != 240
        or cost.get("model_owner_consumer_count") != EXPECTED_UNITS
        or not isinstance(claims, Mapping)
        or claims.get("final_family_access") is not False
        or not isinstance(forbidden, Mapping)
        or set(forbidden)
        != {
            "transition",
            "history",
            "sequence",
            "pairing",
            "final_method_selection",
            "final_family_unlock",
        }
        or any(value is not False for value in forbidden.values())
    ):
        _fail("analysis artifact violates the development-only completeness boundary")
    _validate_runtime_lineage(lineage)
    s = _parse_locked_summary(
        locked["S"],
        condition_id="S-state-availability-listwise-optimum",
        tuple_id=MATCHED_S_TUPLE,
        allow_missing_medians=True,
    )
    b2 = _parse_locked_summary(
        locked["B2"],
        condition_id="B2-global-listwise-optimum",
        tuple_id="lr0p003-e120-t1p2",
        allow_missing_medians=False,
    )
    t = _parse_locked_summary(
        locked["T"],
        condition_id="T-markov-state-transition-listwise-optimum",
        tuple_id="lr0p003-e120-t1p2",
        allow_missing_medians=False,
    )
    selections: list[OutcomeDiagnosticConditionSelection] = []
    trace_keys = {
        "best_minimum_family_success_rate",
        "retained_tuple_ids",
        "selected_tuple_id",
        "candidates",
        "matched_S_tuple",
    }
    for condition in CONDITIONS:
        trace = traces[condition]
        if not isinstance(trace, Mapping) or set(trace) != trace_keys:
            _fail("selection trace schema differs")
        candidates_raw = trace.get("candidates")
        if not isinstance(candidates_raw, list) or len(candidates_raw) != 12:
            _fail("selection trace candidate count differs")
        candidates = tuple(
            _parse_candidate_summary(row, condition) for row in candidates_raw
        )
        if tuple(item.tuple_id for item in candidates) != EXPECTED_TUPLES:
            _fail("selection trace tuple order differs")
        selected_id = trace.get("selected_tuple_id")
        selected_rows = tuple(item for item in candidates if item.tuple_id == selected_id)
        retained = trace.get("retained_tuple_ids")
        if (
            len(selected_rows) != 1
            or not isinstance(retained, list)
            or any(item not in EXPECTED_TUPLES for item in retained)
            or trace.get("matched_S_tuple") != candidates_raw[EXPECTED_TUPLES.index(MATCHED_S_TUPLE)]
        ):
            _fail("selection trace selected, retained, or matched-S row differs")
        selections.append(
            OutcomeDiagnosticConditionSelection(
                condition,
                candidates,
                _parse_fraction(
                    trace.get("best_minimum_family_success_rate"), "best minimum rate"
                ),
                tuple(retained),
                selected_rows[0],
            )
        )
    expected_claims = _jsonable(
        evaluate_outcome_diagnostic_claims(
            OutcomeDiagnosticSelectionResult(tuple(selections)),
            locked_s=s,
            locked_t=t,
        )
    )
    metric_contract = body.get("metric_contract")
    selection_lock = body.get("selection_lock")
    anchor_lock = body.get("anchor_lock")
    if (
        claims != expected_claims
        or metric_contract
        != {
            "primary_metric": "minimum_family_exact_optimum_success_rate",
            "success_tolerance": {"numerator": 1, "denominator": 20},
            "failure_censoring_budget": 2048,
            "failure_sentinel": 2049,
            "tie_break_order": list(TIE_BREAK),
        }
        or body.get("unavailable_diagnostics") != UNAVAILABLE_DIAGNOSTICS
        or not isinstance(selection_lock, Mapping)
        or selection_lock
        != {
            "sha256": lineage["selection_lock_sha256"],
            "selected_S_tuple_id": MATCHED_S_TUPLE,
        }
        or not isinstance(anchor_lock, Mapping)
        or anchor_lock
        != {
            "sha256": lineage["anchor_selection_metrics_sha256"],
            "selected_B2_tuple_id": b2.tuple_id,
            "selected_T_tuple_id": t.tuple_id,
        }
    ):
        _fail("analysis artifact claims, contracts, or authority locks differ")
    return body


def publish_outcome_group_diagnostic_analysis(
    *,
    repository: str | os.PathLike[str],
    result_root: str | os.PathLike[str],
    expected_git_commit: str,
    output: str | os.PathLike[str],
) -> Path:
    """Capture, reduce, and atomically publish the development-only report."""
    target = Path(os.path.abspath(os.fspath(output)))
    root = Path(os.path.abspath(os.fspath(result_root)))
    _reject_output_target(target, root)
    snapshot = capture_outcome_group_diagnostic_analysis_readiness(
        repository=repository,
        output_root=result_root,
        expected_git_commit=expected_git_commit,
        output_state="activated",
    )
    try:
        report = build_outcome_group_diagnostic_analysis(snapshot, expected_git_commit=expected_git_commit)
    finally:
        snapshot.close()
    unsigned = dict(report)
    report["analysis_sha256"] = _sha(canonical_json_bytes(unsigned))
    rendered = canonical_json_bytes(report) + b"\n"
    parent_fd: int | None = None
    temporary_name: str | None = None
    target_linked = False
    publication_complete = False
    try:
        parent_fd = secure_fs.open_directory_chain(target.parent)
        parent_identity = secure_fs.directory_identity(parent_fd)
        temporary_name = f".{target.name}.tmp-{secrets.token_hex(12)}"
        fd = os.open(
            temporary_name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(fd, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_stat = os.stat(
            temporary_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if not stat.S_ISREG(temporary_stat.st_mode):
            _fail("analysis temporary output is not a regular file")
        if secure_fs.read_bytes_at(parent_fd, temporary_name) != rendered:
            _fail("analysis temporary output bytes differ")
        os.link(
            temporary_name,
            target.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        target_linked = True
        os.fsync(parent_fd)
        published = secure_fs.read_bytes_at(parent_fd, target.name)
        target_stat = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            (target_stat.st_dev, target_stat.st_ino)
            != (temporary_stat.st_dev, temporary_stat.st_ino)
            or published != rendered
        ):
            _fail("published analysis identity or bytes differ")
        validate_outcome_group_diagnostic_analysis_bytes(published)
        current_parent = secure_fs.open_directory_chain(target.parent)
        try:
            if secure_fs.directory_identity(current_parent) != parent_identity:
                _fail("analysis output parent changed during publication")
        finally:
            os.close(current_parent)
        os.unlink(temporary_name, dir_fd=parent_fd)
        temporary_name = None
        os.fsync(parent_fd)
        publication_complete = True
    except FileExistsError as exc:
        _fail("analysis output already exists", exc)
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        _fail("analysis output cannot be published safely", exc)
    finally:
        if parent_fd is not None:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except OSError:
                    pass
            if target_linked and not publication_complete:
                try:
                    os.unlink(target.name, dir_fd=parent_fd)
                except OSError:
                    pass
            try:
                os.fsync(parent_fd)
            except OSError:
                pass
            os.close(parent_fd)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    published = publish_outcome_group_diagnostic_analysis(
        repository=args.repository,
        result_root=args.result_root,
        expected_git_commit=args.expected_git_commit,
        output=args.output,
    )
    print(published)
    return 0


__all__ = [
    "OutcomeDiagnosticAnalysisError",
    "SCHEMA_VERSION",
    "build_outcome_group_diagnostic_analysis",
    "main",
    "publish_outcome_group_diagnostic_analysis",
    "validate_outcome_group_diagnostic_analysis_bytes",
]


if __name__ == "__main__":
    raise SystemExit(main())
