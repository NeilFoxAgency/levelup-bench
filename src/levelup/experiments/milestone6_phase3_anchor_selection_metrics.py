"""Locked development-only metrics for the Phase 3 B2 and T anchors.

The raw Phase 2 analysis is an ignored local artifact and is deliberately not
opened by this module.  This authority contains the small, already selected
metric surface needed by the Phase 3 reducer.  Its committed lineage binds the
selection lock, the identity-only anchor manifest, and the frozen ladder before
any Phase 3 comparative result exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from levelup.experiments.milestone6_phase3_selection import (
    Phase3FamilyMetric,
    Phase3SelectedMetric,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[3]
PHASE3_ANCHOR_SELECTION_METRICS_PATH = (
    ROOT / "configs/milestone6/phase3_anchor_selection_metrics.json"
)
SCHEMA_VERSION = "milestone6.phase3.anchor-selection-metrics.v1"
FAMILIES = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
B2 = "B2-global-listwise-optimum"
C = "C-state-conditioned-listwise-optimum"
T = "T-markov-state-transition-listwise-optimum"
_HEX = frozenset("0123456789abcdef")
_TOKEN = object()


class AnchorSelectionMetricsError(ValueError):
    """Raised when the locked anchor metric authority fails closed."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _digest(value: Any) -> str:
    return _sha256(canonical_json_bytes(value))


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX for char in value):
        raise AnchorSelectionMetricsError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnchorSelectionMetricsError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AnchorSelectionMetricsError(f"{label} is not finite")
    return result


def _require_relative_path(repository: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AnchorSelectionMetricsError(f"{label} path is missing")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise AnchorSelectionMetricsError(f"{label} path escapes the repository")
    return repository.joinpath(*pure.parts)


def _read_relative(repository: Path, value: Any, label: str) -> bytes:
    target = _require_relative_path(repository, value, label)
    try:
        parent_fd = secure_fs.open_directory_chain(target.parent)
        try:
            return secure_fs.read_bytes_at(parent_fd, target.name)
        finally:
            os.close(parent_fd)
    except (OSError, RuntimeError, TypeError, ValueError, secure_fs.SecureFilesystemError) as exc:
        raise AnchorSelectionMetricsError(
            f"{label} cannot be read through a pinned descriptor"
        ) from exc


def _require_source(body: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    source = body.get("source", {}).get(name) if isinstance(body.get("source"), Mapping) else None
    if not isinstance(source, Mapping):
        raise AnchorSelectionMetricsError(f"source {name} is missing")
    _require_digest(source.get("sha256"), f"source {name} sha256")
    return source


def _validate_canonical_json(content: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise AnchorSelectionMetricsError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != content:
        raise AnchorSelectionMetricsError(f"{label} is not canonical JSON")
    return value


def _validate_metric_rows(condition: Mapping[str, Any], label: str) -> None:
    rows = condition.get("families")
    if (
        not isinstance(rows, list)
        or tuple(row.get("family_id") for row in rows if isinstance(row, Mapping)) != FAMILIES
    ):
        raise AnchorSelectionMetricsError(f"{label} family order is incomplete")
    if len(rows) != len(FAMILIES):
        raise AnchorSelectionMetricsError(f"{label} family rows are incomplete")
    rates: list[Fraction] = []
    medians: list[Fraction] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("units") != 40:
            raise AnchorSelectionMetricsError(f"{label} must contain exactly 40 units per family")
        count = row.get("exact_optimum_success_count")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 40:
            raise AnchorSelectionMetricsError(f"{label} success count is invalid")
        rate = _require_finite_number(
            row.get("exact_optimum_success_rate"), f"{label} success rate"
        )
        if not 0 <= rate <= 1:
            raise AnchorSelectionMetricsError(f"{label} success rate is outside [0,1]")
        exact_rate = Fraction(str(row["exact_optimum_success_rate"]))
        if exact_rate != Fraction(count, 40):
            raise AnchorSelectionMetricsError(f"{label} success count/rate disagree")
        median = _require_finite_number(
            row.get("median_restricted_interactions"), f"{label} median"
        )
        if not 0 <= median <= 2049:
            raise AnchorSelectionMetricsError(
                f"{label} median is outside the endpoint/sentinel range"
            )
        rates.append(exact_rate)
        medians.append(Fraction(str(row["median_restricted_interactions"])))
    minimum = Fraction(str(condition.get("minimum_family_exact_optimum_success_rate")))
    worst = Fraction(str(condition.get("worst_family_median_restricted_interactions")))
    macro = Fraction(str(condition.get("macro_average_family_median_restricted_interactions")))
    expected_macro = sum(medians, Fraction(0)) / len(medians)
    if (
        minimum != min(rates)
        or worst != max(medians)
        or not math.isclose(float(macro), float(expected_macro), rel_tol=0.0, abs_tol=1e-12)
    ):
        raise AnchorSelectionMetricsError(f"{label} aggregate metrics do not match family rows")


def _validate_source_lineage(body: Mapping[str, Any], repository: Path) -> None:
    source_analysis = _require_source(body, "analysis_file")
    source_selection = _require_source(body, "selection_lock")
    source_anchor = _require_source(body, "anchor_manifest")
    source_protocol = _require_source(body, "protocol")
    # The raw analysis is intentionally not opened: only its locked digest is
    # checked against the committed selection lock's analysis fields below.
    analysis_digest = _require_digest(source_analysis.get("sha256"), "analysis file sha256")
    analysis_self_hash = _require_digest(
        source_analysis.get("analysis_sha256"), "analysis self-hash"
    )
    if (
        source_selection.get("analysis_file_sha256") != analysis_digest
        or source_selection.get("analysis_sha256") != analysis_self_hash
    ):
        raise AnchorSelectionMetricsError("selection lock does not bind the locked analysis hashes")
    selection_bytes = _read_relative(repository, source_selection.get("path"), "selection lock")
    if _sha256(selection_bytes) != source_selection["sha256"]:
        raise AnchorSelectionMetricsError("selection lock source hash changed")
    try:
        selection = json.loads(selection_bytes)
    except (TypeError, ValueError) as exc:
        raise AnchorSelectionMetricsError("selection lock is not valid JSON") from exc
    if not isinstance(selection, dict):
        raise AnchorSelectionMetricsError("selection lock is not an object")
    boundary = selection.get("scientific_boundary", {})
    if (
        boundary.get("development_only") is not True
        or boundary.get("final_family_access") is not False
        or boundary.get("final_method_selection") is not False
    ):
        raise AnchorSelectionMetricsError("selection lock is not development-only")
    analysis = selection.get("analysis", {})
    if (
        analysis.get("analysis_sha256") != analysis_self_hash
        or analysis.get("analysis_file_sha256") != analysis_digest
    ):
        raise AnchorSelectionMetricsError("selection lock analysis fields drifted")

    anchor_bytes = _read_relative(repository, source_anchor.get("path"), "anchor manifest")
    if _sha256(anchor_bytes) != source_anchor["sha256"]:
        raise AnchorSelectionMetricsError("anchor manifest source hash changed")
    anchor = _validate_canonical_json(anchor_bytes, "anchor manifest")
    anchor_self = _require_digest(anchor.get("anchor_manifest_sha256"), "anchor manifest self-hash")
    if (
        anchor_self != source_anchor.get("anchor_manifest_sha256")
        or _digest({k: v for k, v in anchor.items() if k != "anchor_manifest_sha256"})
        != anchor_self
    ):
        raise AnchorSelectionMetricsError("anchor manifest self-hash drifted")
    if (
        anchor.get("final_family_access") is not False
        or anchor.get("new_execution") is not False
        or anchor.get("final_results") != []
        or anchor.get("aggregates") != []
    ):
        raise AnchorSelectionMetricsError("anchor manifest contains final or aggregate data")
    t_alias = anchor.get("t_alias", {})
    if (
        t_alias.get("condition_id") != T
        or t_alias.get("historical_condition_id") != C
        or t_alias.get("source_base_condition_id") != C
        or t_alias.get("analysis_only") is not True
        or t_alias.get("new_model") is not False
        or t_alias.get("new_view") is not False
        or t_alias.get("new_unit_results") is not False
    ):
        raise AnchorSelectionMetricsError("anchor T alias lineage drifted")

    protocol_bytes = _read_relative(repository, source_protocol.get("path"), "Phase 3 protocol")
    if _sha256(protocol_bytes) != source_protocol["sha256"]:
        raise AnchorSelectionMetricsError("Phase 3 protocol source hash changed")
    try:
        protocol = json.loads(protocol_bytes)
    except (TypeError, ValueError) as exc:
        raise AnchorSelectionMetricsError("Phase 3 protocol is not valid JSON") from exc
    if (
        not isinstance(protocol, dict)
        or protocol.get("scope") != "known-development-only"
        or protocol.get("final_family_access") is not False
    ):
        raise AnchorSelectionMetricsError("Phase 3 protocol is not development-only")
    if (
        _require_digest(source_protocol["sha256"], "Phase 3 protocol sha256")
        != body["anchor_lineage"]["phase3_protocol_sha256"]
    ):
        raise AnchorSelectionMetricsError("Phase 3 protocol lineage changed")
    rule = protocol.get("selection_rule", {})
    contract = body["metric_contract"]
    if (
        rule.get("endpoint_adaptation_actions") != contract["endpoint"]
        or rule.get("failure_sentinel") != contract["failure_sentinel"]
        or rule.get("primary_metric") != "minimum_family_exact_optimum_success_rate"
    ):
        raise AnchorSelectionMetricsError("Phase 3 metric contract drifted")
    lineage = anchor.get("lineage", {})
    expected_lineage = body.get("anchor_lineage", {})
    for key in (
        "phase3_protocol_sha256",
        "phase2_selection_lock_sha256",
        "phase2_selection_analysis_sha256",
    ):
        if lineage.get(key) != expected_lineage.get(key):
            raise AnchorSelectionMetricsError(f"anchor lineage changed: {key}")
    if anchor_self != expected_lineage.get("phase3_anchor_manifest_sha256"):
        raise AnchorSelectionMetricsError("anchor self-hash lineage changed")

    selected = selection.get("selected", {})
    for label, condition in ((B2, body["conditions"][B2]), (T, body["conditions"][T])):
        source_condition = condition.get("source_condition_id")
        if source_condition not in (B2, C):
            raise AnchorSelectionMetricsError(f"{label} source condition is invalid")
        locked = selected.get(source_condition)
        if not isinstance(locked, Mapping) or locked.get("candidate_tuple_id") != condition.get(
            "selected_tuple_id"
        ):
            raise AnchorSelectionMetricsError(
                f"{label} selected tuple differs from the selection lock"
            )
        for key in (
            "minimum_family_exact_optimum_success_rate",
            "worst_family_median_restricted_interactions",
            "macro_average_family_median_restricted_interactions",
        ):
            if condition.get(key) != locked.get(key):
                raise AnchorSelectionMetricsError(
                    f"{label} selected metric {key} differs from the selection lock"
                )
        cost = condition.get("cost", {})
        if (
            cost.get("optimizer_steps") != locked.get("optimizer_steps")
            or cost.get("forward_passes") != locked.get("forward_passes")
            or cost.get("recurrent_steps") != 0
        ):
            raise AnchorSelectionMetricsError(
                f"{label} training cost differs from the selection lock"
            )


@dataclass(frozen=True, slots=True, init=False)
class Phase3AnchorSelectionMetrics:
    """Canonical immutable snapshot of selected B2 and historical C/T metrics."""

    body: dict[str, Any]
    canonical_bytes: bytes
    anchor_selection_metrics_sha256: str
    _construction_token: object

    def __init__(
        self,
        *,
        body: dict[str, Any],
        canonical_bytes: bytes,
        anchor_selection_metrics_sha256: str,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _TOKEN:
            raise AnchorSelectionMetricsError("anchor metric snapshots require validation")
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "canonical_bytes", canonical_bytes)
        object.__setattr__(self, "anchor_selection_metrics_sha256", anchor_selection_metrics_sha256)
        object.__setattr__(self, "_construction_token", _construction_token)

    @property
    def sha256(self) -> str:
        return self.anchor_selection_metrics_sha256

    def model_dump(self) -> dict[str, Any]:
        return dict(self.body)


def _validate_phase3_anchor_selection_metrics_bytes(
    content: bytes, *, repository: str | Path, verify_source_lineage: bool
) -> Phase3AnchorSelectionMetrics:
    if not isinstance(content, bytes) or not content:
        raise AnchorSelectionMetricsError("anchor selection metrics bytes are missing")
    body = _validate_canonical_json(content, "anchor selection metrics")
    supplied = _require_digest(
        body.get("anchor_selection_metrics_sha256"), "anchor selection metrics self-hash"
    )
    unsigned = {
        key: value for key, value in body.items() if key != "anchor_selection_metrics_sha256"
    }
    if _digest(unsigned) != supplied:
        raise AnchorSelectionMetricsError("anchor selection metrics self-hash drifted")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("scope") != "known-development-only"
    ):
        raise AnchorSelectionMetricsError("anchor selection metrics schema or scope drifted")
    if (
        body.get("development_only") is not True
        or body.get("final_family_access") is not False
        or body.get("final_method_selection") is not False
    ):
        raise AnchorSelectionMetricsError(
            "anchor selection metrics permits final access or selection"
        )
    contract = body.get("metric_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("endpoint") != 2048
        or contract.get("failure_sentinel") != 2049
    ):
        raise AnchorSelectionMetricsError("anchor selection metric endpoint contract drifted")
    conditions = body.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != {B2, T}:
        raise AnchorSelectionMetricsError("anchor selection metrics must contain exactly B2 and T")
    b2, t = conditions[B2], conditions[T]
    if not isinstance(b2, Mapping) or not isinstance(t, Mapping):
        raise AnchorSelectionMetricsError("anchor selection condition rows are malformed")
    if (
        t.get("historical_condition_id") != C
        or t.get("source_condition_id") != C
        or t.get("selected_tuple_id") != b2.get("selected_tuple_id")
    ):
        raise AnchorSelectionMetricsError("T is not the selected historical C alias")
    _validate_metric_rows(b2, B2)
    _validate_metric_rows(t, T)
    lineage = body.get("anchor_lineage")
    if not isinstance(lineage, Mapping):
        raise AnchorSelectionMetricsError("anchor selection lineage is missing")
    for key in (
        "phase2_analysis_file_sha256",
        "phase2_selection_analysis_sha256",
        "phase2_selection_lock_sha256",
        "phase3_anchor_manifest_sha256",
        "phase3_protocol_sha256",
    ):
        _require_digest(lineage.get(key), f"anchor lineage {key}")
    if verify_source_lineage:
        _validate_source_lineage(body, Path(repository).resolve(strict=True))
    return Phase3AnchorSelectionMetrics(
        body=body,
        canonical_bytes=content,
        anchor_selection_metrics_sha256=supplied,
        _construction_token=_TOKEN,
    )


def validate_phase3_anchor_selection_metrics_bytes(
    content: bytes, *, repository: str | Path = ROOT
) -> Phase3AnchorSelectionMetrics:
    """Validate canonical anchor bytes and their live committed source lineage."""

    return _validate_phase3_anchor_selection_metrics_bytes(
        content, repository=repository, verify_source_lineage=True
    )


def validate_pinned_phase3_anchor_selection_metrics_bytes(
    content: bytes,
) -> Phase3AnchorSelectionMetrics:
    """Validate anchor bytes already authenticated by a descriptor-pinned snapshot."""

    return _validate_phase3_anchor_selection_metrics_bytes(
        content, repository=ROOT, verify_source_lineage=False
    )


def require_phase3_anchor_selection_metrics(value: Any) -> Phase3AnchorSelectionMetrics:
    if (
        not isinstance(value, Phase3AnchorSelectionMetrics)
        or value._construction_token is not _TOKEN
        or canonical_json_bytes(value.body) != value.canonical_bytes
        or value.body.get("anchor_selection_metrics_sha256")
        != value.anchor_selection_metrics_sha256
        or _digest(
            {
                key: item
                for key, item in value.body.items()
                if key != "anchor_selection_metrics_sha256"
            }
        )
        != value.anchor_selection_metrics_sha256
    ):
        raise AnchorSelectionMetricsError("anchor selection metrics snapshot is not canonical")
    return value


def load_phase3_anchor_selection_metrics_bytes(
    path: str | os.PathLike[str] = PHASE3_ANCHOR_SELECTION_METRICS_PATH,
    *,
    repository: str | Path = ROOT,
) -> bytes:
    target = Path(path).absolute()
    try:
        parent_fd = secure_fs.open_directory_chain(target.parent)
        try:
            content = secure_fs.read_bytes_at(parent_fd, target.name)
        finally:
            os.close(parent_fd)
    except (OSError, RuntimeError, TypeError, ValueError, secure_fs.SecureFilesystemError) as exc:
        raise AnchorSelectionMetricsError(
            "committed anchor selection metrics cannot be read safely"
        ) from exc
    validate_phase3_anchor_selection_metrics_bytes(content, repository=repository)
    return content


def phase3_anchor_selected_metrics(
    value: Phase3AnchorSelectionMetrics,
) -> tuple[Phase3SelectedMetric, Phase3SelectedMetric]:
    """Convert validated authority rows into the selector's exact typed inputs."""

    snapshot = require_phase3_anchor_selection_metrics(value)
    selected: list[Phase3SelectedMetric] = []
    for condition_id in (B2, T):
        row = snapshot.body["conditions"][condition_id]
        family_metrics = tuple(
            Phase3FamilyMetric(
                family_id=family["family_id"],
                units=family["units"],
                successes=family["exact_optimum_success_count"],
                success_rate=Fraction(str(family["exact_optimum_success_rate"])),
                median_restricted_interactions=Fraction(
                    str(family["median_restricted_interactions"])
                ),
            )
            for family in row["families"]
        )
        cost = row["cost"]
        exact_minimum = min(item.success_rate for item in family_metrics)
        exact_worst_median = max(item.median_restricted_interactions for item in family_metrics)
        exact_macro_median = sum(
            (item.median_restricted_interactions for item in family_metrics),
            Fraction(0),
        ) / len(family_metrics)
        selected.append(
            Phase3SelectedMetric(
                condition_id=condition_id,
                tuple_id=row["selected_tuple_id"],
                training_tuple_id=row["training_tuple_id"],
                family_metrics=family_metrics,
                minimum_family_success_rate=exact_minimum,
                worst_family_median_restricted_interactions=exact_worst_median,
                macro_average_family_median_restricted_interactions=exact_macro_median,
                optimizer_steps=cost["optimizer_steps"],
                forward_passes=cost["forward_passes"],
                recurrent_steps=cost["recurrent_steps"],
            )
        )
    return selected[0], selected[1]


__all__ = [
    "AnchorSelectionMetricsError",
    "B2",
    "C",
    "FAMILIES",
    "PHASE3_ANCHOR_SELECTION_METRICS_PATH",
    "Phase3AnchorSelectionMetrics",
    "T",
    "load_phase3_anchor_selection_metrics_bytes",
    "phase3_anchor_selected_metrics",
    "require_phase3_anchor_selection_metrics",
    "validate_pinned_phase3_anchor_selection_metrics_bytes",
    "validate_phase3_anchor_selection_metrics_bytes",
]
