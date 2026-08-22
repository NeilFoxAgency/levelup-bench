#!/usr/bin/env python3
"""Deterministic, read-only reducer for completed Phase 2 development screening.

This module intentionally does not activate stores or execute units.  It delegates
record validation to the canonical runtime/extractor and only reduces the resulting
typed summaries plus the immutable training-cost inventory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from decimal import Decimal
from pathlib import Path
from typing import Any

from levelup.experiments.milestone6_phase2_screening_extraction import (
    extract_development_selection,
)
from levelup.experiments.milestone6_phase2_screening_provenance import (
    validate_screening_provenance,
)
from levelup.experiments.milestone6_phase2_screening_runtime import (
    _recheck_manifest_and_tree,
    load_screening_runtime,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.provenance import capture_system_provenance

FAMILIES = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
BASES = (
    "B1-clean-global-optimum-frequency",
    "B2-global-listwise-optimum",
    "C-state-conditioned-listwise-optimum",
)
CONTROLS = ("A0-no-probe-uniform", "A1-paid-probe-uniform")
EXPECTED_SUMMARIES = 38
EXPECTED_UNITS = 240
EXPECTED_TOTAL_UNITS = 9_120
SCHEMA_VERSION = "milestone6.phase2.postscreen-analysis.v1"


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_dump(item) for item in value]
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _dump(item) for key, item in value.items()}
    return value


def _summary_dict(summary: Any) -> dict[str, Any]:
    families = tuple(summary.families)
    if summary.endpoint != 2048 or summary.failure_sentinel != 2049 or len(families) != 6:
        raise ValueError("summary endpoint/sentinel or family coverage drifted")
    if tuple(item.family_id for item in families) != tuple(sorted(FAMILIES)):
        raise ValueError("summary family order drifted")
    if any(item.units != EXPECTED_UNITS // 6 for item in families):
        raise ValueError("summary family unit coverage drifted")
    return {
        "condition_id": summary.condition_id,
        "endpoint": summary.endpoint,
        "failure_sentinel": summary.failure_sentinel,
        "families": [
            {
                "family_id": item.family_id,
                "units": item.units,
                "exact_optimum_success_rate": item.exact_optimum_success_rate,
                "median_restricted_interactions": item.median_restricted_interactions,
            }
            for item in families
        ],
        "minimum_family_exact_optimum_success_rate": summary.minimum_family_exact_optimum_success_rate,
        "worst_family_median_restricted_interactions": summary.worst_family_median_restricted_interactions,
        "macro_average_family_median_restricted_interactions": summary.macro_average_family_median_restricted_interactions,
    }


def _condition_metadata(runtime: Any) -> dict[str, dict[str, Any]]:
    if len(runtime.folds) != len(FAMILIES) or tuple(f.family_id for f in runtime.folds) != FAMILIES:
        raise ValueError("runtime must contain the six frozen development folds")
    first = tuple(runtime.folds[0].config.conditions)
    if len(first) != EXPECTED_SUMMARIES:
        raise ValueError("runtime condition matrix is not the frozen 38-variant set")
    result: dict[str, dict[str, Any]] = {}
    for condition in first:
        params = dict(condition.parameters)
        cid = str(condition.condition_id)
        if cid in result:
            raise ValueError("duplicate condition metadata")
        if cid not in CONTROLS:
            required = (
                "base_condition_id",
                "candidate_tuple_id",
                "training_tuple_id",
                "learning_rate",
                "training_epochs",
                "search_temperature",
            )
            if any(name not in params for name in required):
                raise ValueError("learned condition metadata is incomplete")
            base = str(params["base_condition_id"])
            if base not in BASES:
                raise ValueError("unknown learned condition base")
        result[cid] = {"condition_id": cid, **params}
    if set(cid for cid in result if cid in CONTROLS) != set(CONTROLS):
        raise ValueError("retained controls are incomplete")
    for fold in runtime.folds[1:]:
        current = tuple(fold.config.conditions)
        if len(current) != len(first) or any(
            c.condition_id != f.condition_id or dict(c.parameters) != dict(f.parameters)
            for c, f in zip(current, first)
        ):
            raise ValueError("condition metadata differs across folds")
    learned = [meta for cid, meta in result.items() if cid not in CONTROLS]
    if {meta["base_condition_id"] for meta in learned} != set(BASES):
        raise ValueError("learned condition base coverage drifted")
    return result


def _training_costs(runtime: Any, metadata: dict[str, dict[str, Any]], cid: str) -> dict[str, Any]:
    meta = metadata[cid]
    base = str(meta["base_condition_id"])
    tuple_id = str(meta["training_tuple_id"])
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for fold in runtime.folds:
        matches = []
        for identity, cost in fold.models.costs.items():
            if not isinstance(identity, tuple) or len(identity) != 3:
                raise ValueError("malformed model cost identity")
            if identity[0] == base and identity[1] == tuple_id:
                matches.append((identity, cost))
        if len(matches) != 5:
            raise ValueError("each fold must contain exactly five model-owner costs")
        if {int(identity[2]) for identity, _ in matches} != {0, 1, 2, 3, 4}:
            raise ValueError("model-owner replicate coverage is not exactly zero through four")
        for identity, cost in matches:
            rep = int(identity[2])
            global_id = (fold.family_id, base, tuple_id, rep)
            if global_id in seen:
                raise ValueError("duplicate model-owner cost identity")
            seen.add(global_id)
            compute = fold.models.compute.get(identity)
            manifest = fold.models.manifests.get(identity)
            if compute is None or manifest is None or not hasattr(cost, "accounting"):
                raise ValueError("model cost inventory is incomplete")
            expected_key = fold.model_keys.models.get(identity)
            if expected_key is None:
                raise ValueError("model-owner expected key is absent")
            key_payload = getattr(cost, "key", None)
            expected_key_payload = expected_key.model_dump(mode="json")
            if not isinstance(key_payload, dict) or (
                key_payload != expected_key_payload
                or manifest.key.model_dump(mode="json") != expected_key_payload
                or cost.key_id != expected_key.key_id
                or cost.cost_id != cost.expected_cost_id
                or cost.schema_version != "runner.training-artifact-cost.v2"
                or cost.scope != "training_preparation"
                or cost.artifact_id != manifest.artifact_id
            ):
                raise ValueError("model-owner cost lineage drifted")
            if (
                key_payload.get("condition_id") != base
                or key_payload.get("training_tuple_id") != tuple_id
                or key_payload.get("replicate") != rep
            ):
                raise ValueError("model-owner cost key identity drifted")
            accounting = cost.accounting.training
            report = manifest.report
            if (
                accounting.optimizer_steps != compute.optimizer_steps
                or accounting.forward_passes != compute.forward_passes
                or accounting.optimizer_steps != report.optimizer_steps
                or accounting.forward_passes != report.forward_passes
            ):
                raise ValueError("training cost accounting disagrees with compute/report")
            rows.append({
                "family_id": fold.family_id,
                "base_condition_id": base,
                "training_tuple_id": tuple_id,
                "replicate": rep,
                "optimizer_steps": int(accounting.optimizer_steps),
                "forward_passes": int(accounting.forward_passes),
                "cost_id": str(cost.cost_id),
                "key_id": str(cost.key_id),
                "artifact_id": str(cost.artifact_id),
            })
    if (
        len(rows) != 30
        or len(seen) != 30
        or len({row["cost_id"] for row in rows}) != 30
        or len({row["key_id"] for row in rows}) != 30
        or len({row["artifact_id"] for row in rows}) != 30
    ):
        raise ValueError("learned variant requires exactly 30 unique model-owner artifacts")
    return {
        "artifact_count": len(rows),
        "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        "forward_passes": sum(row["forward_passes"] for row in rows),
        "artifacts": rows,
    }


def _numeric(meta: dict[str, Any]) -> tuple[Decimal, Decimal, Decimal]:
    numeric = (
        Decimal(str(meta["learning_rate"])),
        Decimal(str(meta["training_epochs"])),
        Decimal(str(meta["search_temperature"])),
    )
    if any(value <= 0 for value in numeric) or numeric[1] != numeric[1].to_integral_value():
        raise ValueError("candidate tuple numeric fields are invalid")
    return numeric


def _select(base: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    best_success = max(Decimal(str(item["summary"]["minimum_family_exact_optimum_success_rate"])) for item in candidates)
    retained = [item for item in candidates if Decimal(str(item["summary"]["minimum_family_exact_optimum_success_rate"])) >= best_success - Decimal("0.05")]
    def key(item: dict[str, Any]) -> tuple[Any, ...]:
        summary, cost, meta = item["summary"], item["cost"], item["metadata"]
        return (
            Decimal(str(summary["worst_family_median_restricted_interactions"])),
            Decimal(str(summary["macro_average_family_median_restricted_interactions"])),
            int(cost["optimizer_steps"]),
            int(cost["forward_passes"]),
            *_numeric(meta),
            str(meta["candidate_tuple_id"]),
            str(item["condition_id"]),
        )
    selected = min(retained, key=key)
    return {
        "base_condition_id": base,
        "selected_condition_id": selected["condition_id"],
        "selected_candidate_tuple_id": selected["metadata"]["candidate_tuple_id"],
        "selected_training_tuple_id": selected["metadata"]["training_tuple_id"],
        "rule_trace": {
            "best_minimum_family_success_rate": float(best_success),
            "inclusive_success_tolerance": 0.05,
            "retained_condition_ids": [item["condition_id"] for item in sorted(retained, key=lambda x: x["condition_id"])],
            "tie_break_order": ["worst_family_median", "macro_average_family_median", "summed_unique_optimizer_steps", "summed_unique_forward_passes", "numeric_learning_rate_epochs_temperature"],
        },
        "candidate_set": [
            {
                "condition_id": item["condition_id"],
                "training_tuple_id": item["metadata"]["training_tuple_id"],
                "candidate_tuple_id": item["metadata"]["candidate_tuple_id"],
                "learning_rate": item["metadata"]["learning_rate"],
                "training_epochs": item["metadata"]["training_epochs"],
                "search_temperature": item["metadata"]["search_temperature"],
                "minimum_family_exact_optimum_success_rate": item["summary"]["minimum_family_exact_optimum_success_rate"],
                "worst_family_median_restricted_interactions": item["summary"]["worst_family_median_restricted_interactions"],
                "macro_average_family_median_restricted_interactions": item["summary"]["macro_average_family_median_restricted_interactions"],
                "cost": {k: item["cost"][k] for k in ("artifact_count", "optimizer_steps", "forward_passes")},
            }
            for item in sorted(candidates, key=lambda x: x["condition_id"])
        ],
    }


def _read_only_recheck(runtime: Any) -> None:
    """Recheck the pinned screening boundary without activating stores."""

    _recheck_manifest_and_tree(
        runtime.manifest_path,
        runtime.raw_root,
        runtime.manifest_bytes,
        runtime.manifest,
        runtime.authority_sources,
        runtime.tree_sha256,
        runtime.raw_root_identity,
        runtime.child_identities,
        runtime.manifest_parent_identity,
        runtime.manifest_file_identity,
        runtime.folds,
        runtime.result_namespace_snapshot,
    )
    captured = capture_system_provenance(runtime.repository, runtime.device_policy)
    validate_screening_provenance(
        runtime.provenance,
        captured,
        repository=runtime.repository,
        manifest_bytes=runtime.manifest_bytes,
    )


def _publish_exclusive(parent: Path, name: str, payload: bytes) -> None:
    """Publish one file relative to a pinned parent directory descriptor."""

    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("analysis output name is not one local path component")
    parent_fd = secure_fs.open_directory_chain(parent)
    original_identity = secure_fs.directory_identity(parent_fd)
    temporary_name = f".{name}.{secrets.token_hex(16)}.tmp"
    file_fd = -1
    try:
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("analysis output appeared before publication")
        file_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        offset = 0
        while offset < len(payload):
            offset += os.write(file_fd, payload[offset:])
        os.fsync(file_fd)
        os.close(file_fd)
        file_fd = -1
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise ValueError("analysis output appeared during exclusive publication") from exc
        os.fsync(parent_fd)
        if secure_fs.read_bytes_at(parent_fd, name) != payload:
            raise ValueError("published analysis bytes failed verification")
        current_fd = secure_fs.open_directory_chain(parent)
        try:
            if secure_fs.directory_identity(current_fd) != original_identity:
                raise ValueError("analysis output parent changed during publication")
        finally:
            os.close(current_fd)
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


def reduce_screening(*, manifest_path: str | Path, manifest_sha256: str, raw_root: str | Path, repository: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(raw_root).resolve(strict=True)
    target = Path(output).absolute()
    if target.is_symlink() or os.path.lexists(target):
        raise ValueError("analysis output already exists or is a symlink")
    for parent_candidate in (target.parent, *target.parent.parents):
        if parent_candidate.exists() and parent_candidate.is_symlink():
            raise ValueError("analysis output parent chain contains a symlink")
    try:
        target.resolve().relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("analysis output must be outside raw_root")
    if not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    parent = target.parent.resolve(strict=True)
    if parent.is_symlink():
        raise ValueError("analysis output parent is a symlink")
    runtime = load_screening_runtime(manifest_path, raw_root, repository, manifest_bytes_sha256=manifest_sha256)
    manifest = runtime.manifest
    if tuple(manifest.family_order) != FAMILIES or manifest.development_only is not True or manifest.final_family_access is not False:
        raise ValueError("runtime is not development-only")
    if manifest.expected_total_units != EXPECTED_TOTAL_UNITS:
        raise ValueError("runtime total unit count drifted")
    if any(getattr(manifest, name) is not False for name in ("validation_executed", "search_executed", "outcomes_present", "selection_performed")):
        raise ValueError("runtime manifest has already performed screening")
    summaries = tuple(extract_development_selection(runtime))
    if len(summaries) != EXPECTED_SUMMARIES or len({s.condition_id for s in summaries}) != EXPECTED_SUMMARIES:
        raise ValueError("extractor did not return exact 38 summaries")
    metadata = _condition_metadata(runtime)
    summary_map = {s.condition_id: _summary_dict(s) for s in summaries}
    rows: list[dict[str, Any]] = []
    for cid, summary in summary_map.items():
        row = {"condition_id": cid, "summary": summary, "metadata": metadata[cid]}
        if cid not in CONTROLS:
            row["cost"] = _training_costs(runtime, metadata, cid)
        rows.append(row)
    selected = {_base: _select(_base, [r for r in rows if r["metadata"].get("base_condition_id") == _base]) for _base in BASES}
    if any(len(value["candidate_set"]) != 12 for value in selected.values()):
        raise ValueError("each learned base must contain exactly twelve candidates")
    _read_only_recheck(runtime)
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": {"repository": str(Path(repository).resolve()), "git_commit_sha": getattr(runtime.provenance, "git_commit_sha", None), "provenance": _dump(runtime.provenance)},
        "readiness": {
            "manifest_sha256": manifest.manifest_sha256,
            "manifest_bytes_sha256": manifest_sha256,
            "provenance_sha256": manifest.provenance_sha256,
            "protocol_sha256": manifest.protocol_sha256,
            "screening_candidates_sha256": manifest.screening_candidates_sha256,
            "task_manifest_sha256": manifest.task_manifest_sha256,
            "tree_sha256": runtime.tree_sha256,
            "children": [
                {
                    "heldout_family_id": child.heldout_family_id,
                    "run_id": child.run_id,
                    "config_sha256": child.config_sha256,
                    "expected_units_sha256": child.expected_units_sha256,
                    "provenance_sha256": child.provenance_sha256,
                }
                for child in manifest.children
            ],
        },
        "result_namespace_snapshot_sha256": hashlib.sha256(canonical_json_bytes(_dump(runtime.result_namespace_snapshot))).hexdigest(),
        "counts": {"families": 6, "units_per_variant": EXPECTED_UNITS, "total_units": EXPECTED_TOTAL_UNITS, "summaries": EXPECTED_SUMMARIES},
        "metric_contract": {
            "metric_id": "total_adaptation_actions_to_first_exact_optimum",
            "endpoint": 2048,
            "failure_sentinel": 2049,
            "executed_action_formula": "accounting.probes.actions + accounting.search.actions",
            "oracle_policy": "fixed batch and independent replay complete before reporting-only exact-optimum classification",
            "family_aggregation": "within-family first, then equal family weight",
            "success_tolerance_absolute": 0.05,
            "training_cost_aggregation": "sum each of 30 unique model-owner artifacts exactly once per variant",
        },
        "summaries": [summary_map[cid] for cid in sorted(summary_map)],
        "costs": {r["condition_id"]: r["cost"] for r in rows if r["condition_id"] not in CONTROLS},
        "retained_controls": [summary_map[cid] for cid in CONTROLS],
        "selected": selected,
        "cross_condition_elimination": False,
        "final_family_access": False,
    }
    body["analysis_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    payload = canonical_json_bytes(body) + b"\n"
    _publish_exclusive(parent, target.name, payload)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest-path", "manifest-sha256", "raw-root", "repository", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    result = reduce_screening(
        manifest_path=args.manifest_path,
        manifest_sha256=args.manifest_sha256,
        raw_root=args.raw_root,
        repository=args.repository,
        output=args.output,
    )
    print(json.dumps({"analysis_sha256": result["analysis_sha256"], "output": args.output}, sort_keys=True))


if __name__ == "__main__":
    main()
