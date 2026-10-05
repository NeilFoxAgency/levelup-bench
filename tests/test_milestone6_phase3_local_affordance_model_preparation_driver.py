from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from levelup.experiments import milestone6_phase2_screening_runtime as screening_runtime
from levelup.experiments import milestone6_phase3_anchor as anchor
from levelup.experiments import (
    milestone6_phase3_local_affordance_model_preparation_driver as driver,
)
from levelup.experiments.milestone6_phase3_anchor import (
    load_committed_phase3_anchor_manifest_bytes,
    validate_committed_phase3_anchor_for_local_affordance_preparation,
)
from levelup.experiments.milestone6_phase3_evidence import (
    load_committed_phase3_evidence_lock_bytes,
)
from levelup.experiments.milestone6_phase3_local_affordance_plan import (
    build_local_affordance_plan,
)
from levelup.experiments.runner.storage import RunStore


def _setup_driver(tmp_path, monkeypatch, *, validate_only=False):
    plan = build_local_affordance_plan()
    owner = plan.model_owners[0]
    screening = tmp_path / "screening"
    screening.mkdir()
    manifest_path = screening / driver.CANONICAL_READINESS_PATH
    manifest_path.parent.mkdir(parents=True)
    manifest_bytes = b"canonical phase2 publication fixture\n"
    manifest_path.write_bytes(manifest_bytes)
    screening_raw_root = tmp_path / "screening-raw"
    screening_raw_root.mkdir()
    raw_probe_root = tmp_path / "raw-probe"
    raw_probe_root.mkdir()
    output_relative = Path("tests")
    monkeypatch.setattr(driver, "MODEL_STORE_RELATIVE_PATH", output_relative)
    snapshot_bytes = b"canonical administrative result identity fixture"
    monkeypatch.setattr(
        driver, "RESULT_SNAPSHOT_SHA256", hashlib.sha256(snapshot_bytes).hexdigest()
    )
    runtime = SimpleNamespace(
        folds=(SimpleNamespace(store=SimpleNamespace(run_dir=tmp_path / "screening-result")),)
    )
    runtime.folds[0].store.run_dir.mkdir()
    calls = []

    class Lease:
        def require_active(self):
            calls.append("lease-recheck")
            return self

        def training_fold_manifest(self, fold_id, replicate):
            calls.append(("fold", fold_id, replicate))
            return object()

        def issue_training_fold_probe_capability(self, *, fold_id, replicate):
            calls.append(("capability", fold_id, replicate))
            return object()

    @contextmanager
    def activation(*, expected_git_commit):
        calls.append(("activation", expected_git_commit))
        yield Lease()

    snapshot = SimpleNamespace(
        plan=plan,
        git_commit_sha="a" * 40,
        recheck=lambda **kwargs: calls.append("snapshot-recheck"),
        activation=activation,
    )
    monkeypatch.setattr(driver, "_git_state", lambda repo: ("a" * 40, False))
    monkeypatch.setattr(
        driver,
        "capture_local_affordance_preparation_readiness",
        lambda **kwargs: calls.append(("readiness", kwargs.get("raw_root"))) or snapshot,
    )
    monkeypatch.setattr(driver, "require_local_affordance_preparation_snapshot", lambda value: value)

    def read_snapshot(path):
        assert Path(path) == kwargs_path
        calls.append("snapshot-read")
        return snapshot_bytes

    def load_metadata_runtime(*args, **kwargs):
        assert Path(args[1]) == screening_raw_root
        assert kwargs["result_snapshot_bytes"] == snapshot_bytes
        assert kwargs["selection_lock_sha256"] == driver.SELECTION_LOCK_SHA256
        calls.append("metadata-load")
        return runtime

    monkeypatch.setattr(driver, "_read_pinned_result_snapshot", read_snapshot)
    monkeypatch.setattr(driver, "load_screening_runtime_metadata_only", load_metadata_runtime)
    monkeypatch.setattr(
        driver,
        "recheck_screening_runtime_metadata_only",
        lambda value: calls.append("metadata-recheck"),
    )
    def fail_legacy(*_args, **_kwargs):
        raise AssertionError("legacy payload-reading Phase 2 runtime API was used")

    monkeypatch.setattr(driver, "load_screening_runtime", fail_legacy, raising=False)
    monkeypatch.setattr(driver, "recheck_screening_runtime_readonly", fail_legacy, raising=False)
    monkeypatch.setattr(screening_runtime, "load_screening_runtime", fail_legacy)
    monkeypatch.setattr(screening_runtime, "recheck_screening_runtime_readonly", fail_legacy)
    monkeypatch.setattr(RunStore, "completed_records", fail_legacy)
    monkeypatch.setattr(RunStore, "attempt_records", fail_legacy)
    monkeypatch.setattr(RunStore, "_open_result_namespace", fail_legacy)
    monkeypatch.setattr(driver, "_validated_evidence_lock", lambda *args: calls.append("evidence-lock") or object())
    monkeypatch.setattr(driver, "_check_runtime_policy", lambda *args: (object(), "b" * 64))
    monkeypatch.setattr(
        driver,
        "capture_system_provenance",
        lambda *args: SimpleNamespace(git_dirty=False, git_commit_sha="a" * 40),
    )
    monkeypatch.setattr(driver, "provenance_identity_sha256", lambda value: "b" * 64)

    def training_source(*args, **kwargs):
        calls.append(("source", kwargs["family_id"], kwargs["replicate"]))
        return object()

    def view_builder(**kwargs):
        calls.append(("view", kwargs["view"].view_id))
        return object()

    def prepare(**kwargs):
        calls.append(("train", kwargs["owner"].owner_id))
        return SimpleNamespace(
            owner=owner,
            report=SimpleNamespace(training_examples=37),
            training_examples_sha256="c" * 64,
            model_state_sha256="d" * 64,
            model_identity_sha256="e" * 64,
        )

    def serialize(plan_arg, prepared):
        assert plan_arg is plan
        calls.append("serialize")
        return b"model bytes"

    def make_record(**kwargs):
        calls.append("record")
        assert kwargs["preparation_git_commit_sha"] == "a" * 40
        assert kwargs["preparation_provenance_sha256"] == "b" * 64
        return object()

    def store(root, plan_arg, prepared, record, **kwargs):
        calls.append(("store", prepared.owner.owner_id))
        return SimpleNamespace(record_sha256="f" * 64)

    monkeypatch.setattr(driver, "load_local_affordance_training_source", training_source)
    monkeypatch.setattr(driver, "build_local_affordance_training_view", view_builder)
    monkeypatch.setattr(driver, "prepare_local_affordance_model", prepare)
    monkeypatch.setattr(driver, "serialize_local_affordance_model", serialize)
    monkeypatch.setattr(driver, "build_model_record", make_record)
    monkeypatch.setattr(driver, "write_local_affordance_model", store)
    kwargs = {
        "manifest_path": manifest_path,
        "manifest_bytes_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "screening_raw_root": screening_raw_root,
        "screening_repository": screening,
        "authority_repository": driver._SOURCE_ROOT,
        "output_root": driver._SOURCE_ROOT / output_relative,
        "raw_probe_root": raw_probe_root,
        "result_snapshot_path": driver._SOURCE_ROOT
        / "experiments/phase2-result-namespace-identity.json",
        "owner_id": owner.owner_id,
        "validate_only": validate_only,
    }
    kwargs_path = kwargs["result_snapshot_path"]
    return kwargs, calls, owner


def test_one_owner_driver_uses_only_selected_training_bundle_and_stores_once(tmp_path, monkeypatch):
    kwargs, calls, owner = _setup_driver(tmp_path, monkeypatch)

    result = driver.run_local_affordance_model_preparation(**kwargs)

    assert result["status"] == "owner-prepared"
    assert result["owner_id"] == owner.owner_id
    assert result["full_matrix_authorized"] is False
    assert result["comparative_units_run"] is False
    assert "snapshot-read" in calls
    assert ("readiness", kwargs["raw_probe_root"]) in calls
    assert "metadata-load" in calls
    assert calls.count("metadata-recheck") == 2
    assert sum(isinstance(call, tuple) and call[0] == "train" for call in calls) == 1
    assert calls.count("serialize") == 1
    assert calls.count("record") == 1
    assert sum(isinstance(call, tuple) and call[0] == "source" for call in calls) == 1
    assert sum(isinstance(call, tuple) and call[0] == "store" for call in calls) == 1
    assert next(call for call in calls if isinstance(call, tuple) and call[0] == "source") == (
        "source", owner.heldout_family, owner.replicate
    )
    assert calls.index("evidence-lock") < next(
        index for index, call in enumerate(calls)
        if isinstance(call, tuple) and call[0] == "activation"
    )
    assert calls.index("snapshot-read") < calls.index("metadata-load")
    assert calls.index("metadata-load") < calls.index("metadata-recheck")
    ordered_steps = ("fold", "capability", "source", "view", "train", "serialize", "record", "store")
    step_indices = []
    for step in ordered_steps:
        step_indices.append(
            next(
                index for index, call in enumerate(calls)
                if (call == step if step in {"serialize", "record"} else isinstance(call, tuple) and call[0] == step)
            )
        )
    assert step_indices == sorted(step_indices)


def test_validate_only_checks_exact_owner_without_issuing_training_or_writes(tmp_path, monkeypatch):
    kwargs, calls, owner = _setup_driver(tmp_path, monkeypatch, validate_only=True)

    result = driver.run_local_affordance_model_preparation(**kwargs)

    assert result["status"] == "validated-only"
    assert result["owner_id"] == owner.owner_id
    assert result["training_performed"] is False
    assert result["full_matrix_authorized"] is False
    assert "evidence-lock" in calls
    assert "metadata-load" in calls
    assert calls.count("metadata-recheck") == 2
    assert not any(
        call == "serialize"
        or call == "record"
        or (isinstance(call, tuple) and call[0] in {"activation", "source", "store"})
        for call in calls
    )


def test_validate_only_fails_on_invalid_evidence_lock_before_capability(tmp_path, monkeypatch):
    kwargs, calls, _owner = _setup_driver(tmp_path, monkeypatch, validate_only=True)
    monkeypatch.setattr(
        driver,
        "_validated_evidence_lock",
        lambda *args: (_ for _ in ()).throw(
            driver.LocalAffordanceModelPreparationDriverError("invalid evidence lock")
        ),
    )

    with pytest.raises(driver.LocalAffordanceModelPreparationDriverError, match="invalid evidence"):
        driver.run_local_affordance_model_preparation(**kwargs)

    assert not any(
        isinstance(call, tuple) and call[0] in {"activation", "capability", "source", "store"}
        for call in calls
    )


def test_outcome_free_anchor_validation_never_opens_phase2_result_namespaces(monkeypatch):
    content = load_committed_phase3_anchor_manifest_bytes()
    body = json.loads(content)
    rows_by_family = {family: [] for family in anchor.FAMILIES}
    for row in body["unit_results"]:
        rows_by_family[row["family_id"]].append(row)
    condition_map = {
        row["condition_id"]: (row["base_condition_id"], row["candidate_tuple_id"])
        for row in body["unit_results"]
    }
    folds = []
    for family, rows in rows_by_family.items():
        planned = tuple(
            SimpleNamespace(
                unit_id=row["unit_id"],
                key=SimpleNamespace(
                    condition_id=row["condition_id"],
                    task_id=row["task_id"],
                    task_index=row["task_index"],
                    replicate=row["replicate"],
                    phase=row["phase"],
                ),
            )
            for row in rows
        )
        store = SimpleNamespace(
            run_id=rows[0]["run_id"],
            expected=SimpleNamespace(units=planned),
            attempt_records=lambda: (),
        )
        folds.append(SimpleNamespace(family_id=family, config=object(), store=store))
    runtime = SimpleNamespace()
    monkeypatch.setattr(anchor, "_require_development_runtime", lambda _runtime: tuple(folds))
    monkeypatch.setattr(anchor, "_lineage", lambda *_args: body["lineage"])
    monkeypatch.setattr(anchor, "_model_owner_rows", lambda _folds: body["model_owners"])
    monkeypatch.setattr(anchor, "_validate_frozen_lineage", lambda *_args: None)
    monkeypatch.setattr(
        anchor,
        "_canonical_tasks_by_family",
        lambda: {
            family: tuple(sorted({(row["task_id"], row["task_index"]) for row in rows}))
            for family, rows in rows_by_family.items()
        },
    )
    monkeypatch.setattr(anchor, "_conditions_by_id", lambda _config: condition_map)
    monkeypatch.setattr(
        anchor,
        "_fold_result_bytes_reader",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preparation opened a Phase 2 result namespace")
        ),
    )

    validated = validate_committed_phase3_anchor_for_local_affordance_preparation(
        content,
        runtime=runtime,
        evidence_lock_bytes=load_committed_phase3_evidence_lock_bytes(),
    )

    assert validated.anchor_manifest_sha256 == body["anchor_manifest_sha256"]


def test_forged_owner_is_rejected_before_runtime_or_training(tmp_path, monkeypatch):
    kwargs, calls, _owner = _setup_driver(tmp_path, monkeypatch)
    kwargs["owner_id"] = "0" * 64

    with pytest.raises(driver.LocalAffordanceModelPreparationDriverError, match="absent"):
        driver.run_local_affordance_model_preparation(**kwargs)

    assert not any(call in {"metadata-load", "serialize"} for call in calls)


def test_noncanonical_or_final_output_path_is_rejected_before_runtime(tmp_path, monkeypatch):
    kwargs, calls, _owner = _setup_driver(tmp_path, monkeypatch)
    kwargs["output_root"] = kwargs["screening_raw_root"] / "final-family-output"

    with pytest.raises(driver.LocalAffordanceModelPreparationDriverError, match="canonical"):
        driver.run_local_affordance_model_preparation(**kwargs)

    assert not any(call in {"metadata-load", "serialize"} for call in calls)


def test_screening_and_probe_roots_must_be_distinct_before_any_runtime_work(tmp_path, monkeypatch):
    kwargs, calls, _owner = _setup_driver(tmp_path, monkeypatch)
    kwargs["raw_probe_root"] = kwargs["screening_raw_root"]

    with pytest.raises(driver.LocalAffordanceModelPreparationDriverError, match="must be distinct"):
        driver.run_local_affordance_model_preparation(**kwargs)

    assert "snapshot-read" not in calls
    assert "metadata-load" not in calls
    assert not any(
        isinstance(call, tuple) and call[0] == "readiness"
        for call in calls
    )


def test_output_root_rejects_both_raw_roots_and_all_screening_children(tmp_path, monkeypatch):
    kwargs, _calls, _owner = _setup_driver(tmp_path, monkeypatch)
    output = kwargs["output_root"]
    authority = driver._SOURCE_ROOT
    screening_raw = kwargs["screening_raw_root"]
    probe_raw = kwargs["raw_probe_root"]

    with pytest.raises(driver.LocalAffordanceModelPreparationDriverError, match="overlaps"):
        driver._assert_output_root(
            output,
            authority=authority,
            screening_raw_root=output,
            raw_probe_root=probe_raw,
            runtime=None,
        )
    with pytest.raises(driver.LocalAffordanceModelPreparationDriverError, match="overlaps"):
        driver._assert_output_root(
            output,
            authority=authority,
            screening_raw_root=screening_raw,
            raw_probe_root=output,
            runtime=None,
        )

    child_dirs = tuple(tmp_path / f"screening-child-{index}" for index in range(6))
    for child in child_dirs:
        child.mkdir()
    runtime = SimpleNamespace(
        folds=tuple(
            SimpleNamespace(store=SimpleNamespace(run_dir=child))
            for child in child_dirs[:5]
        )
        + (SimpleNamespace(store=SimpleNamespace(run_dir=output)),)
    )
    with pytest.raises(driver.LocalAffordanceModelPreparationDriverError, match="overlaps"):
        driver._assert_output_root(
            output,
            authority=authority,
            screening_raw_root=screening_raw,
            raw_probe_root=probe_raw,
            runtime=runtime,
        )


def test_symlinked_output_root_is_rejected_before_runtime(tmp_path, monkeypatch):
    kwargs, calls, _owner = _setup_driver(tmp_path, monkeypatch)
    output_link = tmp_path / "model-output-link"
    output_link.symlink_to(kwargs["output_root"], target_is_directory=True)
    kwargs["output_root"] = output_link

    with pytest.raises(driver.LocalAffordanceModelPreparationDriverError, match="symlink"):
        driver.run_local_affordance_model_preparation(**kwargs)

    assert not any(call in {"metadata-load", "serialize"} for call in calls)


def test_result_snapshot_hash_and_selection_lock_are_frozen():
    assert driver.RESULT_SNAPSHOT_SHA256 == (
        "537039da0e4d3bf1442f3bc6ffe2e513b2d2ffe4f010f3426cc1c63c2448b730"
    )
    assert driver.SELECTION_LOCK_SHA256 == (
        "499910276af4359ce3c295933c4ae81293d8a23434f712e310921d05b626f06c"
    )


def test_result_snapshot_rejects_wrong_bytes_and_symlink_paths(tmp_path, monkeypatch):
    authority = tmp_path / "authority"
    authority.mkdir()
    path = authority / "snapshot.json"
    path.write_bytes(b"wrong administrative snapshot bytes")

    with pytest.raises(driver.LocalAffordanceModelPreparationDriverError, match="exact frozen"):
        driver._read_pinned_result_snapshot(path)

    outside = tmp_path / "outside.json"
    outside.write_bytes(b"external file")
    link = authority / "snapshot-link.json"
    link.symlink_to(outside)
    with pytest.raises(driver.LocalAffordanceModelPreparationDriverError, match="symlink"):
        driver._read_pinned_result_snapshot(link)
