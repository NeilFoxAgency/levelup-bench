"""Synthetic safety tests for the outcome-diagnostic analysis publisher."""

import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from levelup.experiments import milestone6_phase3_outcome_diagnostic_analysis as analysis
from levelup.experiments import milestone6_phase3_outcome_diagnostic_result_store as result_store
from levelup.experiments.milestone6_phase3_outcome_diagnostic_analysis import (
    OutcomeDiagnosticAnalysisError,
    _candidate_summary,
    _fraction,
    _locked_references,
    _reject_output_target,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_reducer import (
    OutcomeDiagnosticCandidateMetric,
    OutcomeDiagnosticFamilyMetric,
)


def test_fraction_and_candidate_summary_keep_exact_rationals() -> None:
    metric = OutcomeDiagnosticCandidateMetric(
        "S-RP-state-resource-pressure-outcome-listwise-optimum",
        "lr0p01-e120-t1p2",
        "lr0p01-e120",
        (OutcomeDiagnosticFamilyMetric("plain", 40, 20, Fraction(1, 2), Fraction(2049)),),
        Fraction(1, 2),
        Fraction(2049),
        Fraction(2049),
        10,
        20,
        0,
    )
    assert _fraction(Fraction(2, 4)) == {"numerator": 1, "denominator": 2}
    assert _candidate_summary(metric)["minimum_family_success_rate"] == {
        "numerator": 1,
        "denominator": 2,
    }


def test_output_rejects_inside_root_and_existing_target(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    with pytest.raises(OutcomeDiagnosticAnalysisError):
        _reject_output_target(root / "report.json", root)
    traversal = Path(os.path.abspath(root.parent / "other" / ".." / root.name / "x.json"))
    with pytest.raises(OutcomeDiagnosticAnalysisError):
        _reject_output_target(traversal, root)
    target = tmp_path / "report.json"
    target.write_bytes(b"x")
    with pytest.raises(OutcomeDiagnosticAnalysisError):
        _reject_output_target(target, root)


def test_output_rejects_lexical_symlink_parent(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(OutcomeDiagnosticAnalysisError):
        _reject_output_target(link / "report.json", root)


def test_locked_references_decode_tracked_authority_bytes() -> None:
    class Source:
        def __init__(self, content: bytes) -> None:
            self.content = content
    paths = {
        "configs/milestone6/phase3_development_selection.json",
        "configs/milestone6/phase3_anchor_selection_metrics.json",
    }
    files = {path: Source(Path(path).read_bytes()) for path in paths}
    snapshot = SimpleNamespace(files_by_path=files)
    s, b2, t, _selection, _anchor = _locked_references(snapshot)
    assert s.tuple_id == "lr0p01-e120-t1p2"
    assert b2.tuple_id == "lr0p003-e120-t1p2"
    assert t.tuple_id == "lr0p003-e120-t1p2"
    assert s.minimum_family_success_rate == Fraction(17, 40)


def test_locked_S_keeps_unavailable_family_medians_and_exact_aggregates() -> None:
    class Source:
        def __init__(self, content: bytes) -> None:
            self.content = content

    paths = {
        "configs/milestone6/phase3_development_selection.json",
        "configs/milestone6/phase3_anchor_selection_metrics.json",
    }
    snapshot = SimpleNamespace(
        files_by_path={path: Source(Path(path).read_bytes()) for path in paths}
    )
    s, _b2, _t, _selection, _anchor = _locked_references(snapshot)
    assert all(item.median_restricted_interactions is None for item in s.family_metrics)
    assert s.macro_average_family_median_restricted_interactions == Fraction(7957, 12)
    assert s.worst_family_median_restricted_interactions == Fraction(2049)
    assert tuple(item.successes for item in s.family_metrics) == (37, 34, 35, 17, 21, 23)


def test_locked_references_reject_B2_count_rate_mismatch() -> None:
    class Source:
        def __init__(self, content: bytes) -> None:
            self.content = content

    anchor = json.loads(Path(analysis.ANCHOR_METRICS_PATH).read_bytes())
    anchor["conditions"]["B2-global-listwise-optimum"]["families"][0][
        "exact_optimum_success_count"
    ] = 0
    snapshot = SimpleNamespace(
        files_by_path={
            analysis.PHASE3_SELECTION_PATH: Source(
                Path(analysis.PHASE3_SELECTION_PATH).read_bytes()
            ),
            analysis.ANCHOR_METRICS_PATH: Source(
                analysis.canonical_json_bytes(anchor) + b"\n"
            ),
        }
    )
    with pytest.raises(OutcomeDiagnosticAnalysisError):
        _locked_references(snapshot)


def _unsigned_report() -> dict[str, object]:
    class Source:
        def __init__(self, content: bytes) -> None:
            self.content = content

    authority_paths = {
        analysis.PHASE3_SELECTION_PATH,
        analysis.ANCHOR_METRICS_PATH,
    }
    snapshot = SimpleNamespace(
        files_by_path={path: Source(Path(path).read_bytes()) for path in authority_paths}
    )
    s, b2, t, _selection, _anchor = analysis._locked_references(snapshot)
    selections = []
    traces = {}
    for condition in analysis.CONDITIONS:
        candidates = tuple(
            OutcomeDiagnosticCandidateMetric(
                condition,
                tuple_id,
                tuple_id.rsplit("-t", 1)[0],
                tuple(
                    OutcomeDiagnosticFamilyMetric(
                        family, 40, 20, Fraction(1, 2), Fraction(100)
                    )
                    for family in analysis.ANCHOR_FAMILIES
                ),
                Fraction(1, 2),
                Fraction(100),
                Fraction(100),
                1,
                1,
                0,
            )
            for tuple_id in analysis.EXPECTED_TUPLES
        )
        selection = analysis.OutcomeDiagnosticConditionSelection(
            condition,
            candidates,
            Fraction(1, 2),
            analysis.EXPECTED_TUPLES,
            candidates[0],
        )
        selections.append(selection)
        traces[condition] = {
            "best_minimum_family_success_rate": _fraction(Fraction(1, 2)),
            "retained_tuple_ids": list(analysis.EXPECTED_TUPLES),
            "selected_tuple_id": candidates[0].tuple_id,
            "candidates": [_candidate_summary(item) for item in candidates],
            "matched_S_tuple": _candidate_summary(
                candidates[analysis.EXPECTED_TUPLES.index(analysis.MATCHED_S_TUPLE)]
            ),
        }
    claims = analysis._jsonable(
        analysis.evaluate_outcome_diagnostic_claims(
            analysis.OutcomeDiagnosticSelectionResult(tuple(selections)),
            locked_s=s,
            locked_t=t,
        )
    )
    root_identity = [1, 100]
    lineage = {
        "git_commit_sha": "a" * 40,
        "protocol_sha256": "a" * 64,
        "protocol_file_sha256": "a" * 64,
        "protocol_self_sha256": "a" * 64,
        "protocol_authority_file_hashes": {
            key: "a" * 64
            for key in (
                "phase3_protocol",
                "phase3_plan",
                "phase3_evidence",
                "phase3_model_authority",
                "phase3_anchor_metrics",
                "phase3_development_selection",
            )
        },
        "plan_id": "a" * 64,
        "model_authority_sha256": "b" * 64,
        "model_authority_self_sha256": "b" * 64,
        "model_authority_file_sha256": "c" * 64,
        "model_preparation_git_commit_sha": "b" * 40,
        "model_preparation_provenance_sha256": "d" * 64,
        "source_result_lock_commit_sha": "c" * 40,
        "selection_lock_sha256": "e" * 64,
        "selection_lock_self_sha256": "f" * 64,
        "selection_analysis_sha256": "1" * 64,
        "anchor_selection_metrics_sha256": "2" * 64,
        "anchor_selection_metrics_self_sha256": "3" * 64,
        "activation_marker_sha256": "4" * 64,
        "activation_marker_identity": [1, 101],
        "root_identity": root_identity,
        "stores": [
            {
                "family_id": family,
                "run_id": f"{index + 7:x}" * 64,
                "config_sha256": f"{index + 1:x}" * 64,
                "identities": {
                    "root": root_identity,
                    "family": [1, 200 + index * 10],
                    "run": [1, 201 + index * 10],
                    "namespaces": [1, 202 + index * 10],
                    "records": {
                        condition: [1, 203 + index * 10 + condition_index]
                        for condition_index, condition in enumerate(analysis.CONDITIONS)
                    },
                },
            }
            for index, family in enumerate(analysis.ANCHOR_FAMILIES)
        ],
        "records_manifest_sha256": "5" * 64,
        "records_manifest_count": 5_760,
    }
    return {
        "schema_version": analysis.SCHEMA_VERSION,
        "scope": "known-development-only",
        "development_only": True,
        "final_family_access": False,
        "final_method_selection": False,
        "advancement_to_paired_objectives": False,
        "pairing_claim": False,
        "matrix": {
            "unit_count": 5_760,
            "family_counts": {family: 960 for family in analysis.ANCHOR_FAMILIES},
            "condition_counts": {
                condition: 2_880 for condition in analysis.CONDITIONS
            },
        },
        "selection_traces": traces,
        "locked_references": {
            "S": analysis._metric_summary(s),
            "B2": analysis._metric_summary(b2),
            "T": analysis._metric_summary(t),
        },
        "lineage": lineage,
        "cost": {
            "unit_count": 5_760,
            "model_owner_count": 240,
            "model_owner_consumer_count": 5_760,
        },
        "claims": claims,
        "forbidden_claims": {
            "transition": False,
            "history": False,
            "sequence": False,
            "pairing": False,
            "final_method_selection": False,
            "final_family_unlock": False,
        },
        "metric_contract": {
            "primary_metric": "minimum_family_exact_optimum_success_rate",
            "success_tolerance": {"numerator": 1, "denominator": 20},
            "failure_censoring_budget": 2048,
            "failure_sentinel": 2049,
            "tie_break_order": list(analysis.TIE_BREAK),
        },
        "unavailable_diagnostics": analysis.UNAVAILABLE_DIAGNOSTICS,
        "selection_lock": {
            "sha256": lineage["selection_lock_sha256"],
            "selected_S_tuple_id": analysis.MATCHED_S_TUPLE,
        },
        "anchor_lock": {
            "sha256": lineage["anchor_selection_metrics_sha256"],
            "selected_B2_tuple_id": b2.tuple_id,
            "selected_T_tuple_id": t.tuple_id,
        },
    }


def _signed_report() -> bytes:
    body = _unsigned_report()
    body["analysis_sha256"] = hashlib.sha256(
        analysis.canonical_json_bytes(body)
    ).hexdigest()
    return analysis.canonical_json_bytes(body) + b"\n"


def test_analysis_artifact_validator_requires_canonical_self_hashed_development_report() -> None:
    rendered = _signed_report()
    assert analysis.validate_outcome_group_diagnostic_analysis_bytes(rendered)[
        "analysis_sha256"
    ]

    parsed = json.loads(rendered)
    parsed["analysis_sha256"] = "0" * 64
    with pytest.raises(analysis.OutcomeDiagnosticAnalysisError, match="self-hash"):
        analysis.validate_outcome_group_diagnostic_analysis_bytes(
            analysis.canonical_json_bytes(parsed) + b"\n"
        )

    parsed = json.loads(rendered)
    parsed["development_only"] = False
    parsed.pop("analysis_sha256")
    parsed["analysis_sha256"] = hashlib.sha256(
        analysis.canonical_json_bytes(parsed)
    ).hexdigest()
    with pytest.raises(analysis.OutcomeDiagnosticAnalysisError, match="development-only"):
        analysis.validate_outcome_group_diagnostic_analysis_bytes(
            analysis.canonical_json_bytes(parsed) + b"\n"
        )

    with pytest.raises(analysis.OutcomeDiagnosticAnalysisError, match="canonical"):
        analysis.validate_outcome_group_diagnostic_analysis_bytes(rendered + b"\n")


@pytest.mark.parametrize(
    "tamper",
    (
        "bogus-selected",
        "extra-true-claim",
        "extra-forbidden",
        "missing-locks",
        "extra-lineage",
        "malformed-store-lineage",
        "inconsistent-root-lineage",
        "malformed-run-id",
    ),
)
def test_analysis_artifact_validator_rejects_rehashed_semantic_tamper(
    tamper: str,
) -> None:
    body = _unsigned_report()
    if tamper == "bogus-selected":
        body["selection_traces"][analysis.CONDITIONS[0]]["selected_tuple_id"] = "bogus"
    elif tamper == "extra-true-claim":
        body["claims"]["transition_claim"] = True
    elif tamper == "extra-forbidden":
        body["forbidden_claims"]["evil"] = False
    elif tamper == "missing-locks":
        body.pop("locked_references")
    elif tamper == "extra-lineage":
        body["lineage"]["unvalidated"] = "a" * 64
    elif tamper == "malformed-store-lineage":
        body["lineage"]["stores"][0]["identities"]["records"].pop(
            analysis.CONDITIONS[0]
        )
    elif tamper == "inconsistent-root-lineage":
        body["lineage"]["stores"][0]["identities"]["root"] = [9, 9]
    else:
        body["lineage"]["stores"][0]["run_id"] = "x"
    body["analysis_sha256"] = hashlib.sha256(
        analysis.canonical_json_bytes(body)
    ).hexdigest()
    with pytest.raises(OutcomeDiagnosticAnalysisError):
        analysis.validate_outcome_group_diagnostic_analysis_bytes(
            analysis.canonical_json_bytes(body) + b"\n"
        )


def test_publish_closes_snapshot_and_validates_successful_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_root = tmp_path / "result-root"
    result_root.mkdir()
    parent = tmp_path / "reports"
    parent.mkdir()
    target = parent / "analysis.json"
    closed = 0

    class Snapshot:
        def close(self) -> None:
            nonlocal closed
            closed += 1

    monkeypatch.setattr(
        analysis,
        "capture_outcome_group_diagnostic_analysis_readiness",
        lambda **_kwargs: Snapshot(),
    )
    monkeypatch.setattr(
        analysis,
        "build_outcome_group_diagnostic_analysis",
        lambda _snapshot, *, expected_git_commit: _unsigned_report(),
    )

    published = analysis.publish_outcome_group_diagnostic_analysis(
        repository=tmp_path,
        result_root=result_root,
        expected_git_commit="commit",
        output=target,
    )
    assert published == target.absolute()
    assert closed == 1
    assert analysis.validate_outcome_group_diagnostic_analysis_bytes(
        target.read_bytes()
    )["matrix"]["unit_count"] == 5_760
    assert not list(parent.glob(".analysis.json.tmp-*"))


def test_publish_closes_snapshot_when_build_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_root = tmp_path / "result-root"
    result_root.mkdir()
    closed = 0

    class Snapshot:
        def close(self) -> None:
            nonlocal closed
            closed += 1

    monkeypatch.setattr(
        analysis,
        "capture_outcome_group_diagnostic_analysis_readiness",
        lambda **_kwargs: Snapshot(),
    )

    def fail_build(_snapshot, *, expected_git_commit: str) -> dict[str, object]:
        raise analysis.OutcomeDiagnosticAnalysisError("synthetic reduction failure")

    monkeypatch.setattr(analysis, "build_outcome_group_diagnostic_analysis", fail_build)
    with pytest.raises(analysis.OutcomeDiagnosticAnalysisError, match="synthetic"):
        analysis.publish_outcome_group_diagnostic_analysis(
            repository=tmp_path,
            result_root=result_root,
            expected_git_commit="commit",
            output=tmp_path / "analysis.json",
        )
    assert closed == 1


def test_publish_rejects_existing_target_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_root = tmp_path / "result-root"
    result_root.mkdir()
    target = tmp_path / "analysis.json"
    target.write_bytes(b"sentinel")
    monkeypatch.setattr(
        analysis,
        "capture_outcome_group_diagnostic_analysis_readiness",
        lambda **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        analysis,
        "build_outcome_group_diagnostic_analysis",
        lambda _snapshot, *, expected_git_commit: _unsigned_report(),
    )
    with pytest.raises(analysis.OutcomeDiagnosticAnalysisError, match="already exists"):
        analysis.publish_outcome_group_diagnostic_analysis(
            repository=tmp_path,
            result_root=result_root,
            expected_git_commit="commit",
            output=target,
        )
    assert target.read_bytes() == b"sentinel"


def test_publish_failed_target_readback_leaves_no_target_or_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_root = tmp_path / "result-root"
    result_root.mkdir()
    parent = tmp_path / "reports"
    parent.mkdir()
    target = parent / "analysis.json"
    monkeypatch.setattr(
        analysis,
        "capture_outcome_group_diagnostic_analysis_readiness",
        lambda **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        analysis,
        "build_outcome_group_diagnostic_analysis",
        lambda _snapshot, *, expected_git_commit: _unsigned_report(),
    )
    original_read = analysis.secure_fs.read_bytes_at

    def corrupt_published_bytes(directory_fd: int, name: str) -> bytes:
        value = original_read(directory_fd, name)
        if name == target.name:
            return value + b"corrupted"
        return value

    monkeypatch.setattr(analysis.secure_fs, "read_bytes_at", corrupt_published_bytes)
    with pytest.raises(analysis.OutcomeDiagnosticAnalysisError, match="bytes differ"):
        analysis.publish_outcome_group_diagnostic_analysis(
            repository=tmp_path,
            result_root=result_root,
            expected_git_commit="commit",
            output=target,
        )
    assert not target.exists()
    assert not list(parent.glob(".analysis.json.tmp-*"))


def test_publish_cleanup_error_does_not_mask_failure_or_skip_target_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_root = tmp_path / "result-root"
    result_root.mkdir()
    parent = tmp_path / "reports"
    parent.mkdir()
    target = parent / "analysis.json"
    monkeypatch.setattr(
        analysis,
        "capture_outcome_group_diagnostic_analysis_readiness",
        lambda **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        analysis,
        "build_outcome_group_diagnostic_analysis",
        lambda _snapshot, *, expected_git_commit: _unsigned_report(),
    )
    original_read = analysis.secure_fs.read_bytes_at
    original_unlink = analysis.os.unlink

    def corrupt_published_bytes(directory_fd: int, name: str) -> bytes:
        value = original_read(directory_fd, name)
        if name == target.name:
            monkeypatch.setattr(analysis.os, "unlink", fail_temporary_cleanup)
            return value + b"corrupted"
        return value

    def fail_temporary_cleanup(
        path: str | bytes, *, dir_fd: int | None = None
    ) -> None:
        if isinstance(path, str) and path.startswith(".analysis.json.tmp-"):
            raise IsADirectoryError(path)
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(analysis.secure_fs, "read_bytes_at", corrupt_published_bytes)
    with pytest.raises(analysis.OutcomeDiagnosticAnalysisError, match="bytes differ"):
        analysis.publish_outcome_group_diagnostic_analysis(
            repository=tmp_path,
            result_root=result_root,
            expected_git_commit="commit",
            output=target,
        )
    assert not target.exists()


def test_resume_baseline_rejects_replacement_and_addition_before_record_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public baseline gate compares names/fingerprints without parsing JSON."""

    records_dir = tmp_path / "records"
    records_dir.mkdir()
    records_fd = os.open(records_dir, os.O_RDONLY | os.O_DIRECTORY)
    empty_dir = tmp_path / "empty-records"
    empty_dir.mkdir()
    empty_fd = os.open(empty_dir, os.O_RDONLY | os.O_DIRECTORY)
    unit_name = "unit-1.json"
    fingerprint = (1, 2, 0o100600, 3, 4, 5)
    digest = "a" * 64
    current_names = [unit_name]
    current_fingerprint = fingerprint
    current_digest = digest

    class Store:
        family_id = "plain"

    condition = result_store.CONDITIONS[0]
    other_condition = result_store.CONDITIONS[1]
    planned = SimpleNamespace(unit_id="unit-1", condition_id=condition)
    batch = result_store.OutcomeDiagnosticActivatedBatch(
        _stores=(Store(),),
        _expected=object(),
        _lease=object(),
        _root_fd=-1,
        _descriptors=({"records": {condition: records_fd, other_condition: empty_fd}},),
        _identities=({"root": (9, 10)},),
        _marker_fd=-1,
        _marker_identity=(0, 0),
        _marker_fingerprint=(0, 0, 0, 0, 0, ""),
        _marker_bytes=b"",
        _record_fingerprints={},
        _unit_maps=({"unit-1": planned},),
        _token=result_store._RUNTIME_TOKEN,
    )
    monkeypatch.setattr(
        result_store.OutcomeDiagnosticActivatedBatch,
        "_require_live",
        lambda self, **_kwargs: None,
    )
    monkeypatch.setattr(
        result_store,
        "_runtime_read_entries",
        lambda fd: tuple(current_names) if fd == records_fd and current_names else (),
    )
    monkeypatch.setattr(
        result_store,
        "_resume_file_snapshot",
        lambda _fd, _name: (current_fingerprint, current_digest),
    )
    baseline = result_store.OutcomeDiagnosticResumeBaseline(
        output_root=tmp_path,
        output_root_identity=(9, 10),
        output_state="activated",
        directory_identities=(),
        records=(
            result_store.OutcomeDiagnosticResumeRecordFingerprint(
                "plain", condition, unit_name, fingerprint, digest
            ),
        ),
        stores=(),
    )
    try:
        # A parser call would violate this gate; only schema-neutral fingerprints
        # are allowed before the caller opens result records.
        monkeypatch.setattr(
            result_store,
            "_runtime_parse_record",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("baseline gate parsed a record")
            ),
        )
        batch.require_complete_resume_baseline(baseline)

        current_digest = "b" * 64
        with pytest.raises( result_store.OutcomeDiagnosticResultStoreError, match="identities or bytes changed"):
            batch.require_complete_resume_baseline(baseline)

        current_digest = digest
        current_names.append("foreign.json")
        with pytest.raises(result_store.OutcomeDiagnosticResultStoreError, match="incomplete, foreign"):
            batch.require_complete_resume_baseline(baseline)

        current_names[:] = [unit_name]
        current_digest = digest

        def replace_after_schema_neutral_gate(_self, _baseline) -> None:
            nonlocal current_digest
            current_digest = "c" * 64

        monkeypatch.setattr(
            result_store.OutcomeDiagnosticActivatedBatch,
            "require_complete_resume_baseline",
            replace_after_schema_neutral_gate,
        )
        monkeypatch.setattr(
            result_store,
            "_runtime_record_snapshot",
            lambda _fd, _name: (
                b"{}",
                (
                    fingerprint[0],
                    fingerprint[1],
                    fingerprint[3],
                    fingerprint[4],
                    fingerprint[5],
                    current_digest,
                ),
            ),
        )
        with pytest.raises(
            result_store.OutcomeDiagnosticResultStoreError,
            match="changed between readiness and parsing",
        ):
            batch.validate_existing_records_against_resume_baseline(baseline)
    finally:
        os.close(records_fd)
        os.close(empty_fd)
