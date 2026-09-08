from __future__ import annotations

from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase3_local_affordance_diagnostic_queries as queries


def _fake_lock(rows):
    return SimpleNamespace(body={
        "schema_version": "milestone6.phase3.evidence-lock.v1",
        "scope": "known-development-only",
        "final_family_access": False,
        "payloads_included": False,
        "outcomes_included": False,
        "aggregates": [],
        "final_results": [],
        "counts": {"evidence_artifacts": 30, "families": 6, "replicates": 5},
        "evidence_artifacts": rows,
    })


def test_lock_rows_rejects_missing_or_duplicate_identity(monkeypatch):
    monkeypatch.setattr(queries, "require_phase3_evidence_lock", lambda _lock: None)
    rows = [
        {"family_id": family, "replicate": replicate}
        for family in queries.FAMILIES
        for replicate in range(5)
    ]
    assert len(queries._lock_rows(_fake_lock(rows))) == 30
    with pytest.raises(queries.LocalAffordanceDiagnosticQueryError):
        queries._lock_rows(_fake_lock(rows[:-1]))
    duplicate = rows[:-1] + [rows[0]]
    with pytest.raises(queries.LocalAffordanceDiagnosticQueryError):
        queries._lock_rows(_fake_lock(duplicate))


def test_runtime_task_order_and_coverage_are_exact():
    tasks = tuple(SimpleNamespace(task_id=f"task-{index}", family_id="plain") for index in range(40))
    fold = SimpleNamespace(config=SimpleNamespace(split=SimpleNamespace(development_tasks=tasks)))
    assert len(queries._runtime_task_families(fold, tuple(task.task_id for task in tasks))) == 40
    with pytest.raises(queries.LocalAffordanceDiagnosticQueryError):
        queries._runtime_task_families(fold, tuple(reversed(tuple(task.task_id for task in tasks))))


def test_heldout_matrix_rejects_missing_unit():
    plan = SimpleNamespace(
        candidate_tuple_ids=("lr0p003-e120-t0p6",),
        condition_ids=("B2-global-listwise-optimum",),
        units=(),
    )
    with pytest.raises(queries.LocalAffordanceDiagnosticQueryError):
        queries._heldout_queries(plan, object())


def test_authority_metadata_dump_contains_no_task_or_probe_ids(monkeypatch):
    fake_queries = tuple(
        SimpleNamespace(population="training", family_id=family, states=(object(),))
        for family in queries.FAMILIES
        for _ in range(200)
    ) + tuple(
        SimpleNamespace(population="heldout", family_id=family, states=(object(),))
        for family in queries.FAMILIES
        for _ in range(40)
    )
    monkeypatch.setattr(queries, "_query_matrix_sha256", lambda _queries: "e" * 64)
    monkeypatch.setattr(queries, "LocalAffordanceQuery", SimpleNamespace)
    authority = object.__new__(queries.LocalAffordanceDiagnosticQueryAuthority)
    object.__setattr__(authority, "queries", fake_queries)
    object.__setattr__(authority, "training_query_count", 1200)
    object.__setattr__(authority, "heldout_query_count", 240)
    object.__setattr__(authority, "training_fold_count", 30)
    object.__setattr__(authority, "heldout_binding_count", 240)
    object.__setattr__(authority, "state_query_count", 1440)
    object.__setattr__(authority, "training_state_query_count", 1200)
    object.__setattr__(authority, "heldout_state_query_count", 240)
    object.__setattr__(authority, "evidence_lock_sha256", "a" * 64)
    object.__setattr__(authority, "raw_authority_content_sha256", "b" * 64)
    object.__setattr__(authority, "plan_id", "c" * 64)
    object.__setattr__(authority, "runtime_manifest_sha256", "d" * 64)
    object.__setattr__(authority, "_token", queries._TOKEN)
    object.__setattr__(
        authority,
        "_seal",
        queries._QueryAuthoritySeal(
            query_matrix_sha256="e" * 64,
            training_query_count=1200,
            heldout_query_count=240,
            training_fold_count=30,
            heldout_binding_count=240,
            state_query_count=1440,
            training_state_query_count=1200,
            heldout_state_query_count=240,
            evidence_lock_sha256="a" * 64,
            raw_authority_content_sha256="b" * 64,
            plan_id="c" * 64,
            runtime_manifest_sha256="d" * 64,
            token=queries._TOKEN,
        ),
    )
    dumped = authority.model_dump()
    text = repr(dumped)
    assert "task_id" not in text
    assert "probe_index" not in text


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("training_query_count", 1199),
        ("heldout_query_count", 239),
        ("training_fold_count", 29),
        ("heldout_binding_count", 239),
        ("state_query_count", 1441),
        ("evidence_lock_sha256", "f" * 64),
        ("raw_authority_content_sha256", "f" * 64),
        ("plan_id", "f" * 64),
        ("runtime_manifest_sha256", "f" * 64),
    ),
)
def test_authority_recheck_rejects_mutated_counts_and_lineage(monkeypatch, field, value):
    fake_queries = tuple(
        SimpleNamespace(population="training", family_id=family, states=(object(),))
        for family in queries.FAMILIES
        for _ in range(200)
    ) + tuple(
        SimpleNamespace(population="heldout", family_id=family, states=(object(),))
        for family in queries.FAMILIES
        for _ in range(40)
    )
    monkeypatch.setattr(queries, "_query_matrix_sha256", lambda _queries: "e" * 64)
    monkeypatch.setattr(queries, "LocalAffordanceQuery", SimpleNamespace)
    authority = object.__new__(queries.LocalAffordanceDiagnosticQueryAuthority)
    for name, original in {
        "queries": fake_queries,
        "training_query_count": 1200,
        "heldout_query_count": 240,
        "training_fold_count": 30,
        "heldout_binding_count": 240,
        "state_query_count": 1440,
        "training_state_query_count": 1200,
        "heldout_state_query_count": 240,
        "evidence_lock_sha256": "a" * 64,
        "raw_authority_content_sha256": "b" * 64,
        "plan_id": "c" * 64,
        "runtime_manifest_sha256": "d" * 64,
        "_token": queries._TOKEN,
    }.items():
        object.__setattr__(authority, name, original)
    object.__setattr__(
        authority,
        "_seal",
        queries._QueryAuthoritySeal(
            query_matrix_sha256="e" * 64,
            training_query_count=1200,
            heldout_query_count=240,
            training_fold_count=30,
            heldout_binding_count=240,
            state_query_count=1440,
            training_state_query_count=1200,
            heldout_state_query_count=240,
            evidence_lock_sha256="a" * 64,
            raw_authority_content_sha256="b" * 64,
            plan_id="c" * 64,
            runtime_manifest_sha256="d" * 64,
            token=queries._TOKEN,
        ),
    )
    object.__setattr__(authority, field, value)
    with pytest.raises(queries.LocalAffordanceDiagnosticQueryError):
        authority.require_complete()
