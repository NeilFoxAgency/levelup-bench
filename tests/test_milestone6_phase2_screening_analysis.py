from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase2_screening_analysis as analysis

FAMILIES = analysis.FAMILIES
BASES = analysis.BASES


def _summary(cid: str, success: float, worst: float, macro: float):
    families = tuple(
        SimpleNamespace(
            family_id=f,
            units=40,
            exact_optimum_success_rate=success,
            median_restricted_interactions=worst if f == "plain" else macro,
        )
        for f in sorted(FAMILIES)
    )
    return SimpleNamespace(
        condition_id=cid,
        endpoint=2048,
        failure_sentinel=2049,
        families=families,
        minimum_family_exact_optimum_success_rate=success,
        worst_family_median_restricted_interactions=worst,
        macro_average_family_median_restricted_interactions=macro,
    )


def _runtime(tmp_path: Path):
    controls = [SimpleNamespace(condition_id=c, parameters={}) for c in analysis.CONTROLS]
    conditions = controls[:]
    for base in BASES:
        for i in range(12):
            conditions.append(
                SimpleNamespace(
                    condition_id=f"{base}--t{i}",
                    parameters={
                        "base_condition_id": base,
                        "candidate_tuple_id": f"t{i}-candidate",
                        "training_tuple_id": f"t{i}",
                        "learning_rate": 0.003 if i < 6 else 0.01,
                        "training_epochs": 120 if i % 2 == 0 else 180,
                        "search_temperature": (0.6, 0.9, 1.2)[i % 3],
                    },
                )
            )
    folds = []
    for family in FAMILIES:
        costs, computes, manifests, model_keys = {}, {}, {}, {}
        for base in BASES:
            for i in range(12):
                for rep in range(5):
                    key = (base, f"t{i}", rep)
                    report = SimpleNamespace(optimizer_steps=120 + i, forward_passes=240 + i)
                    key_payload = {
                        "condition_id": base,
                        "training_tuple_id": f"t{i}",
                        "replicate": rep,
                    }
                    key_id = hashlib.sha256(repr((family, key)).encode()).hexdigest()
                    expected_key = SimpleNamespace(
                        key_id=key_id,
                        model_dump=lambda mode=None, payload=key_payload: dict(payload),
                    )
                    model_keys[key] = expected_key
                    computes[key] = report
                    artifact_id = hashlib.sha256(repr(("artifact", family, key)).encode()).hexdigest()
                    manifests[key] = SimpleNamespace(
                        report=report,
                        artifact_id=artifact_id,
                        key=expected_key,
                    )
                    cost_id = hashlib.sha256(repr(("cost", family, key)).encode()).hexdigest()
                    costs[key] = SimpleNamespace(
                        artifact_id=artifact_id,
                        key_id=key_id,
                        cost_id=cost_id,
                        expected_cost_id=cost_id,
                        schema_version="runner.training-artifact-cost.v2",
                        scope="training_preparation",
                        key=key_payload,
                        accounting=SimpleNamespace(training=report),
                    )
        config = SimpleNamespace(conditions=tuple(conditions))
        models = SimpleNamespace(costs=costs, compute=computes, manifests=manifests)
        folds.append(
            SimpleNamespace(
                family_id=family,
                config=config,
                models=models,
                model_keys=SimpleNamespace(models=model_keys),
            )
        )
    manifest = SimpleNamespace(
        family_order=FAMILIES,
        development_only=True,
        final_family_access=False,
        validation_executed=False,
        search_executed=False,
        outcomes_present=False,
        selection_performed=False,
        manifest_sha256="a" * 64,
        provenance_sha256="b" * 64,
        protocol_sha256="c" * 64,
        screening_candidates_sha256="d" * 64,
        task_manifest_sha256="e" * 64,
        expected_total_units=9_120,
        children=tuple(
            SimpleNamespace(
                heldout_family_id=family,
                run_id=f"run-{family}",
                config_sha256="1" * 64,
                expected_units_sha256="2" * 64,
                provenance_sha256="3" * 64,
            )
            for family in FAMILIES
        ),
    )
    provenance = SimpleNamespace(git_commit_sha="f" * 40, model_dump=lambda mode=None: {"git_commit_sha": "f" * 40})
    runtime = SimpleNamespace(
        folds=tuple(folds),
        manifest=manifest,
        provenance=provenance,
        result_namespace_snapshot=(("plain", ()),),
        tree_sha256="4" * 64,
    )
    summaries = [_summary(c.condition_id, 0.5, 10.0, 8.0) for c in conditions]
    return runtime, summaries


def test_reducer_calls_extractor_once_and_selects_independently(tmp_path, monkeypatch):
    runtime, summaries = _runtime(tmp_path)
    calls = []
    monkeypatch.setattr(analysis, "load_screening_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(analysis, "extract_development_selection", lambda rt: calls.append(rt) or tuple(summaries))
    monkeypatch.setattr(analysis, "_read_only_recheck", lambda rt: None)
    # B1: two candidates within inclusive 5pp; lower worst wins. B2/C are independent.
    for idx, summary in enumerate(summaries):
        if summary.condition_id == f"{BASES[0]}--t0":
            summary.worst_family_median_restricted_interactions = 9.0
        if summary.condition_id == f"{BASES[0]}--t1":
            summary.minimum_family_exact_optimum_success_rate = 0.45
            summary.worst_family_median_restricted_interactions = 1.0
        if summary.condition_id == f"{BASES[1]}--t2":
            summary.worst_family_median_restricted_interactions = 1.0
        if summary.condition_id == f"{BASES[2]}--t3":
            summary.worst_family_median_restricted_interactions = 1.0
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "sibling" / "analysis.json"
    result = analysis.reduce_screening(
        manifest_path=tmp_path / "manifest.json",
        manifest_sha256="1" * 64,
        raw_root=raw,
        repository=tmp_path,
        output=out,
    )
    assert len(calls) == 1
    assert result["selected"][BASES[0]]["selected_condition_id"] == f"{BASES[0]}--t1"
    assert result["selected"][BASES[1]]["base_condition_id"] == BASES[1]
    assert result["costs"][f"{BASES[0]}--t1"]["artifact_count"] == 30
    assert result["costs"][f"{BASES[0]}--t1"]["optimizer_steps"] == 30 * 121
    assert result["costs"][f"{BASES[0]}--t1"]["forward_passes"] == 30 * 241
    assert result["cross_condition_elimination"] is False
    assert result["final_family_access"] is False
    stored = __import__("json").loads(out.read_text())
    digest_body = dict(stored)
    digest_body.pop("analysis_sha256")
    assert hashlib.sha256(analysis.canonical_json_bytes(digest_body)).hexdigest() == result["analysis_sha256"]


def test_output_safety_and_inclusive_tolerance(tmp_path, monkeypatch):
    runtime, summaries = _runtime(tmp_path)
    monkeypatch.setattr(analysis, "load_screening_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(analysis, "extract_development_selection", lambda rt: tuple(summaries))
    monkeypatch.setattr(analysis, "_read_only_recheck", lambda rt: None)
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(ValueError, match="outside raw_root"):
        analysis.reduce_screening(manifest_path="m", manifest_sha256="1" * 64, raw_root=raw, repository=tmp_path, output=raw / "x.json")
    nested = raw / "new" / "nested"
    with pytest.raises(ValueError, match="outside raw_root"):
        analysis.reduce_screening(
            manifest_path="m",
            manifest_sha256="1" * 64,
            raw_root=raw,
            repository=tmp_path,
            output=nested / "x.json",
        )
    assert not nested.exists()
    conflict = tmp_path / "conflict.json"
    conflict.write_text("old")
    with pytest.raises(ValueError, match="already exists"):
        analysis.reduce_screening(manifest_path="m", manifest_sha256="1" * 64, raw_root=raw, repository=tmp_path, output=conflict)
    # Exact 0.05 absolute band is retained by the Decimal comparison.
    candidates = [
        {"condition_id": "x", "metadata": {"candidate_tuple_id": "x-candidate", "training_tuple_id": "x", "learning_rate": 1, "training_epochs": 1, "search_temperature": 1}, "summary": {"minimum_family_exact_optimum_success_rate": 0.50, "worst_family_median_restricted_interactions": 2, "macro_average_family_median_restricted_interactions": 2}, "cost": {"artifact_count": 30, "optimizer_steps": 1, "forward_passes": 1}},
        {"condition_id": "y", "metadata": {"candidate_tuple_id": "y-candidate", "training_tuple_id": "y", "learning_rate": 2, "training_epochs": 2, "search_temperature": 2}, "summary": {"minimum_family_exact_optimum_success_rate": 0.45, "worst_family_median_restricted_interactions": 1, "macro_average_family_median_restricted_interactions": 1}, "cost": {"artifact_count": 30, "optimizer_steps": 1, "forward_passes": 1}},
    ]
    assert analysis._select("B1", candidates)["selected_condition_id"] == "y"


def test_numeric_tuple_final_tie():
    base = {"summary": {"minimum_family_exact_optimum_success_rate": 1, "worst_family_median_restricted_interactions": 1, "macro_average_family_median_restricted_interactions": 1}, "cost": {"artifact_count": 30, "optimizer_steps": 1, "forward_passes": 1}}
    candidates = [
        {**base, "condition_id": "late", "metadata": {"candidate_tuple_id": "late-candidate", "training_tuple_id": "late", "learning_rate": 0.01, "training_epochs": 120, "search_temperature": 0.9}},
        {**base, "condition_id": "early", "metadata": {"candidate_tuple_id": "early-candidate", "training_tuple_id": "early", "learning_rate": 0.003, "training_epochs": 120, "search_temperature": 0.9}},
    ]
    assert analysis._select("B1", candidates)["selected_condition_id"] == "early"


def test_read_only_recheck_uses_snapshot_boundary_without_activation(monkeypatch):
    values = tuple(object() for _ in range(12))
    runtime = SimpleNamespace(
        manifest_path=values[0],
        raw_root=values[1],
        manifest_bytes=values[2],
        manifest=values[3],
        authority_sources=values[4],
        tree_sha256=values[5],
        raw_root_identity=values[6],
        child_identities=values[7],
        manifest_parent_identity=values[8],
        manifest_file_identity=values[9],
        folds=values[10],
        result_namespace_snapshot=values[11],
        repository=Path("/repo"),
        device_policy=object(),
        provenance=object(),
    )
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        analysis,
        "_recheck_manifest_and_tree",
        lambda *args: seen.setdefault("recheck", args),
    )
    monkeypatch.setattr(
        analysis,
        "capture_system_provenance",
        lambda repository, policy: seen.setdefault("capture", (repository, policy)),
    )
    monkeypatch.setattr(
        analysis,
        "validate_screening_provenance",
        lambda *args, **kwargs: seen.setdefault("validate", (args, kwargs)),
    )
    analysis._read_only_recheck(runtime)
    assert seen["recheck"] == values
    assert seen["capture"] == (runtime.repository, runtime.device_policy)
    assert "validate" in seen
    assert not hasattr(runtime, "recheck_before_execution")


def test_descriptor_anchored_publication_rejects_parent_replacement(
    tmp_path, monkeypatch
):
    parent = tmp_path / "output"
    parent.mkdir()
    moved = tmp_path / "moved"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    original_write = analysis.os.write
    replaced = False

    def substitute_parent(fd, payload):
        nonlocal replaced
        if not replaced:
            replaced = True
            parent.rename(moved)
            parent.symlink_to(replacement, target_is_directory=True)
        return original_write(fd, payload)

    monkeypatch.setattr(analysis.os, "write", substitute_parent)
    with pytest.raises(Exception, match="cannot securely open|parent changed"):
        analysis._publish_exclusive(parent, "analysis.json", b"payload")
    assert not (replacement / "analysis.json").exists()


def test_selection_lock_binds_reviewed_reducer_and_frozen_authority() -> None:
    root = Path(__file__).parents[1]
    selection = json.loads(
        (root / "configs/milestone6/phase2_screening_selection.json").read_text()
    )
    assert selection["schema_version"] == "milestone6.phase2.selection-lock.v1"
    assert selection["scientific_boundary"] == {
        "development_only": True,
        "final_family_access": False,
        "final_method_selection": False,
        "claims_deferred": [
            "transition_information",
            "history_or_sequence",
            "frontier_optimum_pairing",
        ],
    }
    assert selection["analysis"]["integrated_source_sha256"] == hashlib.sha256(
        (root / "src/levelup/experiments/milestone6_phase2_screening_analysis.py").read_bytes()
    ).hexdigest()
    for key, path in (
        ("protocol_sha256", "configs/milestone6/development_protocol.json"),
        (
            "screening_candidates_sha256",
            "configs/milestone6/phase2_screening_candidates.json",
        ),
        ("task_manifest_sha256", "configs/milestone6/development_tasks.json"),
        (
            "readiness_manifest_bytes_sha256",
            "experiments/milestone6_phase2_screening_readiness.json",
        ),
    ):
        assert selection["authority"][key] == hashlib.sha256(
            (root / path).read_bytes()
        ).hexdigest()
    assert selection["matrix"] == {
        "family_order": ["plain", "battery", "cooldown", "heat", "momentum", "combo"],
        "families": 6,
        "variants": 38,
        "units": 9120,
        "units_per_variant": 240,
        "evidence_artifacts": 30,
        "representation_views": 90,
        "models": 360,
        "shared_artifact_slots": 480,
    }
    selected = selection["selected"]
    assert selected["B1-clean-global-optimum-frequency"]["candidate_tuple_id"] == (
        "lr0p003-e120-t0p6"
    )
    assert selected["B2-global-listwise-optimum"]["candidate_tuple_id"] == (
        "lr0p003-e120-t1p2"
    )
    assert selected["C-state-conditioned-listwise-optimum"]["candidate_tuple_id"] == (
        "lr0p003-e120-t1p2"
    )
    comparison = selection["comparison"]["b2_vs_c"]
    assert comparison["B2_min_family_success"] == 0.4
    assert comparison["C_min_family_success"] == 0.075
    assert comparison["tuples_C_loses_primary_min_success"] == 12
    assert comparison["tuples_C_improves_macro_median"] == 1
