from __future__ import annotations

from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase3_local_affordance_diagnostic_capture as capture
from levelup.experiments.milestone6_phase3_local_affordance_diagnostics import (
    FAMILY_ORDER,
    POPULATION_ORDER,
    LocalAffordanceQuery,
    aggregate_local_affordance_diagnostics,
)
from levelup.learning.state_conditioned import (
    IndexedProbeRow,
    ObservableState,
    ObservedTransition,
    TaskProbeRows,
    bind_task_local_affordance_evidence,
    build_affordance_table,
)


def _queries() -> tuple[LocalAffordanceQuery, ...]:
    rows = tuple(
        IndexedProbeRow(
            index,
            ObservedTransition(
                ObservableState(index / 64, 1 - index / 64, index / 64, 0.5, 0.25, ("a",)),
                "a",
                ObservableState((index + 1) / 64, 1 - (index + 1) / 64, (index + 1) / 64, 0.5, 0.25, ("a",)),
                False,
            ),
        )
        for index in range(64)
    )
    evidence = bind_task_local_affordance_evidence(
        TaskProbeRows(rows),
        build_affordance_table(tuple(row.transition for row in rows), target_samples_per_alias=8),
    )
    state = rows[0].transition.before
    return tuple(
        LocalAffordanceQuery(
            population=population,
            family_id=family,
            evidence=evidence,
            states=(state,),
        )
        for population in POPULATION_ORDER
        for family in FAMILY_ORDER
        for _ in range(200 if population == "training" else 40)
    )


def test_resource_counters_are_zero_only() -> None:
    assert capture.DiagnosticResourceCounters().model_dump(mode="json") == {
        "training_actions": 0,
        "search_actions": 0,
        "replay_actions": 0,
        "evaluator_calls": 0,
        "oracle_calls": 0,
        "candidate_searches": 0,
        "model_training_runs": 0,
    }
    with pytest.raises(ValueError):
        capture.DiagnosticResourceCounters(training_actions=1)


def test_capture_recomputes_report_and_rejects_substitution(monkeypatch: pytest.MonkeyPatch) -> None:
    report = aggregate_local_affordance_diagnostics(_queries())
    authority = SimpleNamespace(
        queries=_queries(),
        require_complete=lambda: None,
        model_dump=lambda: {
            "query_matrix_sha256": "a" * 64,
            "training_state_query_count": 1200,
            "heldout_state_query_count": 240,
        },
    )
    monkeypatch.setattr(capture, "recheck_screening_runtime_readonly", lambda _runtime: None)
    monkeypatch.setattr(capture, "require_raw_probe_authority_snapshot", lambda _raw: None)
    monkeypatch.setattr(capture, "build_local_affordance_diagnostic_query_authority", lambda *args: authority)
    monkeypatch.setattr(capture, "aggregate_local_affordance_diagnostics", lambda _queries: report)
    runtime = SimpleNamespace(
        provenance=SimpleNamespace(git_commit_sha="1" * 40),
        manifest_bytes=b"manifest",
    )
    raw = SimpleNamespace(
        evidence_lock_file_sha256=capture._sha(b"{}"),
        authority_content_sha256="b" * 64,
        manifest=SimpleNamespace(manifest_id="a" * 64),
        manifest_file=SimpleNamespace(snapshot=SimpleNamespace(sha256="c" * 64)),
    )
    evidence = SimpleNamespace(evidence_lock_sha256="d" * 64)
    plan = SimpleNamespace(plan_id="e" * 64)
    source_values = {name: "f" * 64 for name in capture.SOURCE_DIGEST_NAMES}
    source_values.update(
        {
            "phase2_readiness_manifest": capture._sha(b"manifest"),
            "phase3_evidence_lock_file": capture._sha(b"{}"),
            "raw_authority_content": "b" * 64,
            "raw_authority_manifest_file": "c" * 64,
            "raw_authority_manifest_id": "a" * 64,
        }
    )
    source_digests = tuple(sorted(source_values.items()))
    result = capture._build_capture(
        runtime=runtime,
        evidence_lock=evidence,
        evidence_lock_bytes=b"{}",
        raw_snapshot=raw,
        plan=plan,
        source_digests=source_digests,
        plan_lock_file_sha256="f" * 64,
    )
    assert result.diagnostics_pass is all(
        population.coverage_gate.passes
        and all(family.coverage_gate.passes for family in population.family_summaries)
        for population in report.populations
    )
    assert result.model_training is False
    assert result.final_family_access is False

    calls = iter((report, report.model_copy(update={"populations": tuple(reversed(report.populations))})))
    monkeypatch.setattr(capture, "aggregate_local_affordance_diagnostics", lambda _queries: next(calls))
    with pytest.raises(capture.LocalAffordanceDiagnosticCaptureError, match="recomputation"):
        capture._build_capture(
            runtime=runtime,
            evidence_lock=evidence,
            evidence_lock_bytes=b"{}",
            raw_snapshot=raw,
            plan=plan,
            source_digests=source_digests,
            plan_lock_file_sha256="f" * 64,
        )

    payload = result.model_dump(mode="json")
    payload["final_family_access"] = True
    with pytest.raises(ValueError, match="prohibited"):
        capture.LocalAffordanceDiagnosticCapture.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["training_state_query_count"] += 1
    with pytest.raises(ValueError, match="state-query"):
        capture.LocalAffordanceDiagnosticCapture.model_validate(payload)

    payload = result.model_dump(mode="json")
    readiness_index = next(
        index
        for index, (name, _digest) in enumerate(payload["source_digests"])
        if name == "phase2_readiness_manifest"
    )
    payload["source_digests"][readiness_index][1] = "0" * 64
    with pytest.raises(ValueError, match="source digests differ"):
        capture.LocalAffordanceDiagnosticCapture.model_validate(payload)


def test_capture_path_rejects_noncanonical_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    manifest = repository / "experiments/milestone6/phase2_screening_readiness.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"{}")
    screening_raw = tmp_path / "screening-raw"
    raw_probe = tmp_path / "raw-probe"
    screening_raw.mkdir()
    raw_probe.mkdir()
    with pytest.raises(capture.LocalAffordanceDiagnosticCaptureError, match="canonical"):
        capture.run_local_affordance_diagnostic_capture(
            manifest,
            "a" * 64,
            screening_raw,
            raw_probe,
            repository,
            repository,
        )
