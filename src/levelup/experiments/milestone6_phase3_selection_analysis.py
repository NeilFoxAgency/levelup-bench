"""Read-only Phase 3 development selection and claim publication.

This is deliberately a post-execution boundary.  It consumes only the
already-activated development result stores, the frozen authorities, and the
locked B2/T anchor metric file.  It never activates a store, writes beneath a
result root, imports an environment, or resolves a final-family path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any

from levelup.experiments.milestone6_phase3_anchor_selection_metrics import (
    phase3_anchor_selected_metrics,
    validate_phase3_anchor_selection_metrics_bytes,
)
from levelup.experiments.milestone6_phase3_execution_gate import (
    ACTIVATION_MARKER_NAME,
    open_activated_phase3_results,
)
from levelup.experiments.milestone6_phase3_model_authority import (
    load_phase3_model_artifact_authority_bytes,
)
from levelup.experiments.milestone6_phase3_plan import (
    FAMILIES,
    bind_validated_phase3_plan,
    validate_phase3_plan_lock_bytes,
)
from levelup.experiments.milestone6_phase3_protocol import ROOT
from levelup.experiments.milestone6_phase3_readiness import capture_phase3_readiness
from levelup.experiments.milestone6_phase3_reducer import validate_phase3_matrix
from levelup.experiments.milestone6_phase3_result_store import (
    EXPECTED_TOTAL_UNIT_COUNT,
    Phase3ResultStore,
    build_phase3_expected_plan,
    load_phase3_result_stores,
)
from levelup.experiments.milestone6_phase3_selection import (
    evaluate_phase3_claims,
    select_phase3_tuples,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes

SCHEMA_VERSION = "milestone6.phase3.selection-analysis.v1"


class Phase3SelectionAnalysisError(RuntimeError):
    """Raised when the complete development matrix cannot be published."""


def _fail(message: str, exc: BaseException | None = None) -> None:
    if exc is None:
        raise Phase3SelectionAnalysisError(message)
    raise Phase3SelectionAnalysisError(message) from exc


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _digest(value: object) -> str:
    return _sha(canonical_json_bytes(value))


def _json_bytes(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _exact_fraction(value: Any) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _canonical_repo(value: str | os.PathLike[str]) -> Path:
    path = Path(value).absolute()
    for item in (path, *path.parents):
        if os.path.lexists(item) and item.is_symlink():
            _fail("authority repository or ancestor is a symlink")
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        _fail("authority repository is unavailable", exc)
    if not path.is_dir():
        _fail("authority repository is not a directory")
    return path


def _read_source(repo: Path, relative: str) -> bytes:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail("unsafe authority path")
    try:
        with ExitStack() as stack:
            root_fd = secure_fs.open_directory_chain(repo)
            stack.callback(os.close, root_fd)
            parent = root_fd
            for component in pure.parts[:-1]:
                parent = secure_fs.open_child_directory(parent, component)
                stack.callback(os.close, parent)
            return secure_fs.read_bytes_at(parent, pure.parts[-1])
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        _fail(f"cannot read authority source: {relative}", exc)
    raise AssertionError("unreachable")


def _marker_snapshot(
    root: Path, stores: tuple[Phase3ResultStore, ...], readiness: Any, plan: Any, authority: Any
) -> tuple[dict[str, Any], str, tuple[int, int]]:
    """Read and validate the durable activation marker through a held fd."""
    try:
        with ExitStack() as stack:
            root_fd = secure_fs.open_directory_chain(root)
            stack.callback(os.close, root_fd)
            marker_fd = os.open(ACTIVATION_MARKER_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
            stack.callback(os.close, marker_fd)
            st = os.fstat(marker_fd)
            if not stat.S_ISREG(st.st_mode):
                _fail("activation marker is not a regular file")
            content = os.read(marker_fd, max(1, int(st.st_size) + 1))
            if len(content) != st.st_size:
                _fail("activation marker changed while being read")
            try:
                body = json.loads(content)
            except (TypeError, ValueError) as exc:
                _fail("activation marker is not valid JSON", exc)
            if not isinstance(body, dict) or canonical_json_bytes(body) + b"\n" != content:
                _fail("activation marker is not canonical")
            supplied = body.get("marker_sha256")
            unsigned = {key: value for key, value in body.items() if key != "marker_sha256"}
            if supplied != _digest(unsigned):
                _fail("activation marker self-hash drifted")
            root_identity = secure_fs.directory_identity(root_fd)
            if tuple(body.get("root_identity", ())) != root_identity:
                _fail("activation marker root identity differs")
            if (
                body.get("plan_id") != plan.plan.plan_id
                or body.get("model_authority_sha256") != authority.authority_sha256
            ):
                _fail("activation marker authority lineage differs")
            if body.get("protocol_sha256") != plan.plan.protocol_sha256:
                _fail("activation marker protocol lineage differs")
            readiness_body = body.get("readiness", {})
            if (
                readiness_body.get("git_commit_sha") != readiness.git_commit_sha
                or readiness_body.get("training_shuffle_report_sha256")
                != readiness.training_shuffle_report_sha256
            ):
                _fail("activation marker readiness lineage differs")
            rows = body.get("stores")
            if (
                not isinstance(rows, list)
                or tuple(row.get("family_id") for row in rows) != FAMILIES
            ):
                _fail("activation marker store identities are incomplete")
            for store, row in zip(stores, rows, strict=True):
                if (
                    row.get("run_id") != store.run_id
                    or row.get("config_sha256") != store.config_sha256
                ):
                    _fail("activation marker store lineage differs")
                expected = {
                    key: list(value)
                    for key, value in {
                        "root": store.root_identity,
                        "family": store.family_identity,
                        "run": store.run_identity,
                        "units": store.units_identity,
                        "attempts": store.attempts_identity,
                    }.items()
                }
                if row.get("identities") != expected:
                    _fail("activation marker store identity differs")
            return body, _sha(content), root_identity
    except Phase3SelectionAnalysisError:
        raise
    except (OSError, TypeError, ValueError, secure_fs.SecureFilesystemError) as exc:
        _fail("activation marker cannot be read safely", exc)
    raise AssertionError("unreachable")


def _metric(value: Any) -> dict[str, Any]:
    return {
        "condition_id": value.condition_id,
        "candidate_tuple_id": value.tuple_id,
        "training_tuple_id": value.training_tuple_id,
        "families": [
            {
                "family_id": row.family_id,
                "units": row.units,
                "successes": row.successes,
                "success_rate": float(row.success_rate),
                "success_rate_exact": _exact_fraction(row.success_rate),
                "median_restricted_interactions": float(row.median_restricted_interactions),
                "median_restricted_interactions_exact": _exact_fraction(
                    row.median_restricted_interactions
                ),
            }
            for row in value.family_metrics
        ],
        "minimum_family_success_rate": float(value.minimum_family_success_rate),
        "minimum_family_success_rate_exact": _exact_fraction(value.minimum_family_success_rate),
        "worst_family_median_restricted_interactions": float(
            value.worst_family_median_restricted_interactions
        ),
        "worst_family_median_restricted_interactions_exact": _exact_fraction(
            value.worst_family_median_restricted_interactions
        ),
        "macro_average_family_median_restricted_interactions": float(
            value.macro_average_family_median_restricted_interactions
        ),
        "macro_average_family_median_restricted_interactions_exact": _exact_fraction(
            value.macro_average_family_median_restricted_interactions
        ),
        "owner_cost": {
            "optimizer_steps": value.optimizer_steps,
            "forward_passes": value.forward_passes,
            "recurrent_steps": value.recurrent_steps,
        },
    }


def _selection_trace(value: Any) -> dict[str, Any]:
    return {
        "condition_id": value.condition_id,
        "best_primary_minimum_family_success_rate": float(value.best_minimum_family_success_rate),
        "best_primary_minimum_family_success_rate_exact": _exact_fraction(
            value.best_minimum_family_success_rate
        ),
        "retained_candidate_tuple_ids": list(value.retained_tuple_ids),
        "selected_candidate_tuple_id": value.selected.tuple_id,
    }


def _training_claim_eligible(readiness: Any, plan: Any, authority: Any) -> bool:
    try:
        report_path = "configs/milestone6/phase3_training_shuffle_report.json"
        content = readiness.files_by_path[report_path].content
        body = json.loads(content)
        views = body["views"]
        if (
            body.get("development_only") is not True
            or body.get("final_family_access") is not False
            or body.get("outcomes_included") is not False
            or body.get("search_included") is not False
        ):
            return False
        if (
            body.get("model_authority_sha256") != authority.authority_sha256
            or body.get("report_sha256") != readiness.training_shuffle_report_sha256
        ):
            return False
        return len(views) == 30 and all(
            view.get("claim_eligible") is True
            and view.get("plan_id") == plan.plan.plan_id
            and view.get("protocol_sha256") == plan.plan.protocol_sha256
            for view in views
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def build_phase3_selection_analysis(
    *,
    repository: str | os.PathLike[str] = ROOT,
    result_root: str | os.PathLike[str],
    expected_git_commit: str,
) -> dict[str, Any]:
    """Validate and reduce one complete, activated development matrix in memory."""
    repo = _canonical_repo(repository)
    root = Path(result_root).absolute()
    for item in (root, *root.parents):
        if os.path.lexists(item) and item.is_symlink():
            _fail("result root or ancestor is a symlink")
    try:
        root = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        _fail("result root is unavailable", exc)
    if not root.is_dir():
        _fail("result root is not a directory")
    try:
        readiness = capture_phase3_readiness(
            repo, execution_preflight=True, expected_git_commit=expected_git_commit
        )
        validated_plan = bind_validated_phase3_plan(
            validate_phase3_plan_lock_bytes(
                readiness.files_by_path["configs/milestone6/phase3_plan_lock.json"].content
            )
        )
        authority = load_phase3_model_artifact_authority_bytes(
            readiness.files_by_path[
                "configs/milestone6/phase3_model_artifact_authority.json"
            ].content
        )
        expected = build_phase3_expected_plan(validated_plan, authority)
        stores = tuple(load_phase3_result_stores(root, validated_plan, authority))
    except Exception as exc:
        if isinstance(exc, Phase3SelectionAnalysisError):
            raise
        _fail("Phase 3 authorities or result stores cannot be loaded", exc)
    if len(stores) != len(FAMILIES) or tuple(store.family_id for store in stores) != FAMILIES:
        _fail("Phase 3 result stores are not the complete six-family matrix")
    try:
        with readiness.hold_for_activation(expected_git_commit=expected_git_commit) as lease:
            with open_activated_phase3_results(
                stores,
                expected,
                lease,
                expected_git_commit=expected_git_commit,
            ) as batch:
                marker, marker_sha, root_identity = _marker_snapshot(
                    root, stores, readiness, validated_plan, authority
                )
                completed_ids = batch.completed_unit_ids()
                expected_ids = tuple(item.unit.unit_id for item in expected.units)
                if len(completed_ids) != EXPECTED_TOTAL_UNIT_COUNT or set(completed_ids) != set(
                    expected_ids
                ):
                    _fail("Phase 3 result matrix is incomplete or duplicated")
                readers = {store.family_id: store for store in batch.stores}
                all_records = []
                for planned in expected.units:
                    record = readers[planned.heldout_family].load_completed(planned.unit.unit_id)
                    if record is None:
                        _fail("Phase 3 result matrix lost a completed unit")
                    all_records.append(record)
                attempts = batch.attempt_records()
                if any(attempt.unit_id not in set(expected_ids) for attempt in attempts):
                    _fail("attempt record does not correspond to a completed unit")
                if any(not attempt.retryable for attempt in attempts):
                    _fail("non-retryable attempt remains in a completed Phase 3 matrix")
                matrix = validate_phase3_matrix(validated_plan, authority, tuple(all_records))
                anchor_path = "configs/milestone6/phase3_anchor_selection_metrics.json"
                anchor_content = readiness.files_by_path[anchor_path].content
                anchor = validate_phase3_anchor_selection_metrics_bytes(
                    anchor_content, repository=repo
                )
                locked_b2, locked_t = phase3_anchor_selected_metrics(anchor)
                training_eligible = _training_claim_eligible(readiness, validated_plan, authority)
                selected = select_phase3_tuples(
                    validated_plan,
                    authority,
                    matrix,
                    training_shuffle_claim_eligible=training_eligible,
                )
                claims = evaluate_phase3_claims(
                    selected,
                    locked_b2=locked_b2,
                    locked_t=locked_t,
                    training_shuffle_claim_eligible=training_eligible,
                )
                new_metrics = [_metric(item) for item in selected.selections]
                candidate_summaries = [
                    _metric(item)
                    for choice in selected.condition_selections
                    for item in choice.candidates
                ]
                selection_traces = [
                    _selection_trace(choice) for choice in selected.condition_selections
                ]
                claim_dict = {
                    "transition": {
                        "claim": claims.transition_claim,
                        "delta": float(claims.transition_gain),
                        "delta_exact": _exact_fraction(claims.transition_gain),
                        "gate": "strictly greater than 0.05",
                    },
                    "history_access": {
                        "claim": claims.history_access_claim,
                        "delta_over_t": float(claims.history_gain_over_t),
                        "delta_over_t_exact": _exact_fraction(claims.history_gain_over_t),
                        "delta_over_h0": float(claims.history_gain_over_h0),
                        "delta_over_h0_exact": _exact_fraction(claims.history_gain_over_h0),
                        "gate": "both strictly greater than 0.05",
                    },
                    "sequence_order": {
                        "claim": claims.sequence_order_claim,
                        "delta": float(claims.sequence_order_gain),
                        "delta_exact": _exact_fraction(claims.sequence_order_gain),
                        "training_shuffle_gate": claims.training_shuffle_claim_eligible,
                        "heldout_shuffle_gate": claims.heldout_shuffle_claim_eligible,
                        "gate": "delta > 0.05 and both eligibility gates",
                    },
                    "advancement": {
                        "claim": claims.advancement_to_paired_objectives,
                        "family_success_drops": [
                            {
                                "family_id": family,
                                "drop": float(drop),
                                "drop_exact": _exact_fraction(drop),
                            }
                            for family, drop in claims.b2_minus_h4_family_success_drops
                        ],
                        "minimum_family_drop": float(claims.b2_minus_h4_minimum_family_success),
                        "minimum_family_drop_exact": _exact_fraction(
                            claims.b2_minus_h4_minimum_family_success
                        ),
                        "gate": "history + sequence and no B2 family/minimum drop over 0.05",
                    },
                }
                report = {
                    "schema_version": SCHEMA_VERSION,
                    "scope": "known-development-only",
                    "development_only": True,
                    "final_method_selection": False,
                    "final_family_access": False,
                    "unit_count": len(all_records),
                    "candidate_summaries": candidate_summaries,
                    "selection_traces": selection_traces,
                    "selected_metrics": {
                        "B2": _metric(locked_b2),
                        "T": _metric(locked_t),
                        "new": new_metrics,
                    },
                    "claims": claim_dict,
                    "metric_contract": {
                        "primary": "minimum_family_exact_optimum_success_rate",
                        "restricted_interactions": "paid probes + candidate-generation actions through first post-hoc exact hit",
                        "failure_sentinel": 2049,
                        "excluded_from_primary_and_restricted_interaction_metrics": [
                            "replay",
                            "oracle",
                            "resets",
                            "model forward passes",
                            "model recurrent steps",
                            "wall",
                            "non-cost diagnostics",
                        ],
                        "cost_tie_break": "summed unique-owner optimizer steps, forward passes, then recurrent steps",
                    },
                    "sources": {
                        "git_commit_sha": readiness.git_commit_sha,
                        "plan_id": validated_plan.plan.plan_id,
                        "protocol_sha256": validated_plan.plan.protocol_sha256,
                        "model_authority_sha256": authority.authority_sha256,
                        "training_shuffle_report_sha256": readiness.training_shuffle_report_sha256,
                        "training_shuffle_report_file_sha256": readiness.training_shuffle_report_file_sha256,
                        "anchor_selection_metrics_sha256": anchor.sha256,
                        "anchor_selection_metrics_file_sha256": _sha(anchor_content),
                        "activation_marker_sha256": marker_sha,
                    },
                    "stores": [
                        {
                            "family_id": store.family_id,
                            "run_id": store.run_id,
                            "config_sha256": store.config_sha256,
                            "identities": {
                                "root": list(store.root_identity),
                                "family": list(store.family_identity),
                                "run": list(store.run_identity),
                                "units": list(store.units_identity),
                                "attempts": list(store.attempts_identity),
                            },
                        }
                        for store in stores
                    ],
                    "activation": {
                        "marker": marker,
                        "root_identity": list(root_identity),
                    },
                }
    except Phase3SelectionAnalysisError:
        raise
    except Exception as exc:
        _fail("Phase 3 result snapshot failed descriptor-pinned validation", exc)
    return report


def publish_phase3_selection_analysis(
    *,
    repository: str | os.PathLike[str] = ROOT,
    result_root: str | os.PathLike[str],
    expected_git_commit: str,
    output: str | os.PathLike[str],
) -> Path:
    """Build and exclusively publish the self-hashed analysis outside result_root."""
    report = build_phase3_selection_analysis(
        repository=repository, result_root=result_root, expected_git_commit=expected_git_commit
    )
    raw_target = Path(output).absolute()
    root = Path(result_root).absolute().resolve(strict=True)
    for item in (raw_target.parent, *raw_target.parent.parents):
        if os.path.lexists(item) and item.is_symlink():
            _fail("analysis output parent is a symlink")
    try:
        target_parent = raw_target.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        _fail("analysis output parent is unavailable", exc)
    target = target_parent / raw_target.name
    if target == root or root in target.parents:
        _fail("analysis output must be outside result_root")
    unsigned = dict(report)
    report["selection_analysis_sha256"] = _digest(unsigned)
    rendered = _json_bytes(report)
    try:
        parent_fd = secure_fs.open_directory_chain(target.parent)
        try:
            parent_identity = secure_fs.directory_identity(parent_fd)
            fd = os.open(
                target.name,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
            with os.fdopen(fd, "wb") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.fsync(parent_fd)
            if secure_fs.read_bytes_at(parent_fd, target.name) != rendered:
                _fail("published analysis bytes differ")
            current_parent = secure_fs.open_directory_chain(target.parent)
            try:
                if secure_fs.directory_identity(current_parent) != parent_identity:
                    _fail("analysis output parent changed during publication")
            finally:
                os.close(current_parent)
        finally:
            os.close(parent_fd)
    except FileExistsError as exc:
        _fail("analysis output already exists", exc)
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        _fail("analysis output cannot be published safely", exc)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=str(ROOT))
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = publish_phase3_selection_analysis(
        repository=args.repository,
        result_root=args.result_root,
        expected_git_commit=args.expected_git_commit,
        output=args.output,
    )
    report = json.loads(output.read_bytes())
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": report["selection_analysis_sha256"],
                "candidate_count": len(report["candidate_summaries"]),
                "unit_count": report["unit_count"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "Phase3SelectionAnalysisError",
    "build_phase3_selection_analysis",
    "publish_phase3_selection_analysis",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
