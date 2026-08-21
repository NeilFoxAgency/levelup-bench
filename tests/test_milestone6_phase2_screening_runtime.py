"""Contract tests for the development-only screening runtime boundary."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase2_screening_runtime as runtime
from levelup.experiments.runner.config import DevicePolicy, canonical_json_bytes
from levelup.experiments.runner.execution import ExperimentRunner
from levelup.experiments.runner.records import SystemProvenance
from levelup.experiments.runner.training_data_artifacts import TrainingDataArtifactError

PROVENANCE = SystemProvenance(
    git_commit_sha="0" * 40,
    git_dirty=False,
    python_version="test",
    packages={"levelup-bench": "test"},
    installed_packages_sha256="a" * 64,
    os="test",
    architecture="test",
    cpu="test",
    cpu_count=1,
    memory_bytes=1,
    requested_device="cpu",
    resolved_device="cpu",
    requested_torch_threads=1,
    actual_torch_threads=1,
    requested_torch_interop_threads=1,
    actual_torch_interop_threads=1,
    deterministic_algorithms_requested=True,
    deterministic_algorithms_actual=True,
    processes=1,
    captured_at_utc=datetime(2026, 8, 21, tzinfo=UTC),
)
_REAL_RECHECK = runtime._recheck_manifest_and_tree


class _Manifest:
    child_run_ids = ("child",)
    children = (SimpleNamespace(run_id="child"),)
    family_order = ("plain",)
    development_only = True
    final_family_access = False
    validation_executed = False
    search_executed = False
    outcomes_present = False
    selection_performed = False
    protocol_sha256 = "1" * 64
    screening_candidates_sha256 = "2" * 64
    task_manifest_sha256 = "3" * 64
    provenance_sha256 = runtime.provenance_identity_sha256(PROVENANCE)
    provenance = PROVENANCE

    def model_dump(self, **_kwargs):
        return {
            "schema_version": "test",
            "manifest_sha256": "4" * 64,
            "protocol_sha256": self.protocol_sha256,
            "screening_candidates_sha256": self.screening_candidates_sha256,
            "task_manifest_sha256": self.task_manifest_sha256,
        }


def _fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    manifest = _Manifest()
    body = canonical_json_bytes(manifest.model_dump()) + b"\n"
    committed = tmp_path / "committed.json"
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "child" / "units").mkdir(parents=True)
    (raw_root / "child" / "attempts").mkdir()
    (raw_root / "phase2-screening-readiness.json").write_bytes(body)
    committed.write_bytes(body)
    source = tmp_path / "protocol.json"
    source.write_bytes(b"authority")
    source_snapshot = runtime.AuthoritySourceSnapshot(
        "protocol", source, source.read_bytes(), hashlib.sha256(source.read_bytes()).hexdigest()
    )
    config = SimpleNamespace(
        device_policy=SimpleNamespace(requested_device="cpu"),
        parameters={"heldout_family_id": "plain"},
        split=SimpleNamespace(final_tasks=()),
        conditions=(),
    )
    monkeypatch.setattr(runtime, "load_screening_readiness_manifest", lambda _path: manifest)
    monkeypatch.setattr(runtime, "_authority_sources", lambda _manifest: (source_snapshot,))
    monkeypatch.setattr(runtime, "screening_child_configs", lambda: (config,))
    monkeypatch.setattr(runtime, "_assert_development_manifest", lambda *_args: None)
    monkeypatch.setattr(runtime, "build_screening_plan", lambda: SimpleNamespace())
    monkeypatch.setattr(runtime, "validate_screening_plan", lambda *_args: None)
    monkeypatch.setattr(runtime, "_assert_tree_shape", lambda *_args: None)
    monkeypatch.setattr(
        runtime,
        "_load_fold",
        lambda *_args: SimpleNamespace(store=SimpleNamespace(_execution_ready=True)),
    )
    monkeypatch.setattr(
        runtime,
        "_activate_prepared_batch",
        lambda stores, _provenance: [
            setattr(store, "_execution_ready", True) for store in stores
        ],
    )
    monkeypatch.setattr(runtime, "_assert_global_inventory", lambda *_args: None)
    monkeypatch.setattr(runtime, "_recheck_manifest_and_tree", lambda *_args: None)
    return committed, raw_root, source, body, config


def test_manifest_byte_pin_is_checked_before_loading(monkeypatch, tmp_path):
    committed, raw_root, _source, _body, _config = _fixture(monkeypatch, tmp_path)
    with pytest.raises(TrainingDataArtifactError, match="supplied pin"):
        runtime.load_screening_runtime(
            committed,
            raw_root,
            tmp_path,
            manifest_bytes_sha256="0" * 64,
            provenance=PROVENANCE,
        )


def test_supplied_provenance_still_applies_and_captures_once(monkeypatch, tmp_path):
    committed, raw_root, _source, body, _config = _fixture(monkeypatch, tmp_path)
    policy_calls: list[str] = []
    capture_calls: list[str] = []
    monkeypatch.setattr(runtime, "apply_runtime_policy", lambda *_args: policy_calls.append("apply"))
    monkeypatch.setattr(
        runtime,
        "capture_system_provenance",
        lambda *_args: capture_calls.append("capture") or PROVENANCE,
    )
    handle = runtime.load_screening_runtime(
        committed,
        raw_root,
        tmp_path,
        manifest_bytes_sha256=hashlib.sha256(body).hexdigest(),
        provenance=PROVENANCE,
    )
    assert handle.provenance == PROVENANCE
    assert policy_calls == ["apply"]
    assert capture_calls == ["capture"]


def test_missing_provenance_applies_policy_and_captures_once(monkeypatch, tmp_path):
    committed, raw_root, _source, body, config = _fixture(monkeypatch, tmp_path)
    policy_calls: list[str] = []
    capture_calls: list[str] = []
    monkeypatch.setattr(runtime, "apply_runtime_policy", lambda *_args: policy_calls.append("apply"))
    monkeypatch.setattr(
        runtime,
        "capture_system_provenance",
        lambda *_args: capture_calls.append("capture") or PROVENANCE,
    )
    handle = runtime.load_screening_runtime(
        committed,
        raw_root,
        tmp_path,
        manifest_bytes_sha256=hashlib.sha256(body).hexdigest(),
    )
    assert handle.provenance == PROVENANCE
    assert policy_calls == ["apply"]
    assert capture_calls == ["capture"]
    assert config.device_policy.requested_device == "cpu"


def test_load_returns_locked_stores_and_runner_cannot_execute(monkeypatch, tmp_path):
    committed, raw_root, _source, body, _config = _fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(runtime, "apply_runtime_policy", lambda *_args: None)
    monkeypatch.setattr(runtime, "capture_system_provenance", lambda *_args: PROVENANCE)
    handle = runtime.load_screening_runtime(
        committed,
        raw_root,
        tmp_path,
        manifest_bytes_sha256=hashlib.sha256(body).hexdigest(),
        provenance=PROVENANCE,
    )
    assert all(fold.store._execution_ready is False for fold in handle.folds)
    with pytest.raises(RuntimeError, match="for_execution=True"):
        ExperimentRunner(handle.folds[0].store).execute(lambda _unit: None)


def test_successful_required_recheck_opens_all_store_gates(monkeypatch, tmp_path):
    committed, raw_root, _source, body, _config = _fixture(monkeypatch, tmp_path)
    capture_calls: list[str] = []
    monkeypatch.setattr(runtime, "apply_runtime_policy", lambda *_args: None)
    monkeypatch.setattr(
        runtime,
        "capture_system_provenance",
        lambda *_args: capture_calls.append("capture") or PROVENANCE,
    )
    handle = runtime.load_screening_runtime(
        committed,
        raw_root,
        tmp_path,
        manifest_bytes_sha256=hashlib.sha256(body).hexdigest(),
        provenance=PROVENANCE,
    )
    handle.recheck_before_execution()
    assert all(fold.store._execution_ready is True for fold in handle.folds)
    assert capture_calls == ["capture", "capture"]


def test_post_activation_recheck_failure_clears_all_gates(monkeypatch, tmp_path):
    committed, raw_root, _source, body, _config = _fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(runtime, "apply_runtime_policy", lambda *_args: None)
    monkeypatch.setattr(runtime, "capture_system_provenance", lambda *_args: PROVENANCE)
    handle = runtime.load_screening_runtime(
        committed,
        raw_root,
        tmp_path,
        manifest_bytes_sha256=hashlib.sha256(body).hexdigest(),
        provenance=PROVENANCE,
    )
    rechecks: list[str] = []

    def recheck(*_args):
        rechecks.append("recheck")
        if len(rechecks) == 2:
            raise TrainingDataArtifactError("post-activation tamper")

    monkeypatch.setattr(runtime, "_recheck_manifest_and_tree", recheck)
    with pytest.raises(TrainingDataArtifactError, match="post-activation"):
        handle.recheck_before_execution()
    assert rechecks == ["recheck", "recheck"]
    assert all(fold.store._execution_ready is False for fold in handle.folds)


def test_supplied_provenance_rejects_current_capture_identity_mismatch(monkeypatch, tmp_path):
    committed, raw_root, _source, body, _config = _fixture(monkeypatch, tmp_path)
    captured = PROVENANCE.model_copy(update={"git_commit_sha": "1" * 40})
    calls: list[str] = []
    monkeypatch.setattr(runtime, "apply_runtime_policy", lambda *_args: calls.append("apply"))
    monkeypatch.setattr(
        runtime,
        "capture_system_provenance",
        lambda *_args: calls.append("capture") or captured,
    )
    with pytest.raises(TrainingDataArtifactError, match="current captured"):
        runtime.load_screening_runtime(
            committed,
            raw_root,
            tmp_path,
            manifest_bytes_sha256=hashlib.sha256(body).hexdigest(),
            provenance=PROVENANCE,
        )
    assert calls == ["apply", "capture"]


def test_plan_identity_tamper_is_rejected_before_child_loading():
    plan = SimpleNamespace(
        plan_id="a" * 64,
        family_order=("plain",),
        protocol_sha256="1" * 64,
        screening_candidates_sha256="2" * 64,
        task_manifest_sha256="3" * 64,
        replicates=(0, 1, 2, 3, 4),
        children=(),
    )
    manifest = SimpleNamespace(
        screening_plan_id="b" * 64,
        family_order=("plain",),
        protocol_sha256="1" * 64,
        screening_candidates_sha256="2" * 64,
        task_manifest_sha256="3" * 64,
        children=(),
        development_only=True,
        final_family_access=False,
        validation_executed=False,
        search_executed=False,
        outcomes_present=False,
        selection_performed=False,
    )
    with pytest.raises(TrainingDataArtifactError, match="screening plan"):
        runtime._assert_development_manifest(manifest, (), plan)


def test_global_inventory_union_tamper_is_rejected():
    manifest = SimpleNamespace(
        evidence_key_ids=("e",),
        view_key_ids=("v",),
        model_key_ids=("m",),
        shared_artifact_key_ids=("s",),
        model_artifact_ids=("a",),
    )
    fold = SimpleNamespace(
        data_keys=SimpleNamespace(
            evidence={0: SimpleNamespace(key_id="different-e")},
            views={("base", 0): SimpleNamespace(key_id="v")},
        ),
        model_keys=SimpleNamespace(models={("base", "tuple", 0): SimpleNamespace(key_id="m")}),
        shared_plan=SimpleNamespace(artifacts=(SimpleNamespace(key_id="s"),)),
        models=SimpleNamespace(manifests={"model": SimpleNamespace(artifact_id="a")}),
    )
    with pytest.raises(TrainingDataArtifactError, match="global evidence"):
        runtime._assert_global_inventory(manifest, (fold,))


def test_recheck_detects_post_load_tree_tampering(monkeypatch, tmp_path):
    committed, raw_root, source, body, _config = _fixture(monkeypatch, tmp_path)
    # Use the real tree/authority recheck for this test; the fixture's fake
    # inventory and activation boundary remain isolated.
    monkeypatch.undo()
    manifest = _Manifest()
    monkeypatch.setattr(runtime, "load_screening_readiness_manifest", lambda _path: manifest)
    monkeypatch.setattr(runtime, "_authority_sources", lambda _manifest: (
        runtime.AuthoritySourceSnapshot(
            "protocol", source, source.read_bytes(), hashlib.sha256(source.read_bytes()).hexdigest()
        ),
    ))
    monkeypatch.setattr(runtime, "screening_child_configs", lambda: ())
    monkeypatch.setattr(runtime, "_assert_development_manifest", lambda *_args: None)
    monkeypatch.setattr(runtime, "_validate_child_top_level", lambda *_args: None)
    monkeypatch.setattr(runtime, "_CHILD_TOP_LEVEL_NAMES", frozenset({"marker", "units", "attempts"}))
    monkeypatch.setattr(runtime, "_load_fold", lambda *_args: SimpleNamespace(store=object()))
    monkeypatch.setattr(runtime, "_activate_prepared_batch", lambda *_args: None)
    monkeypatch.setattr(runtime, "apply_runtime_policy", lambda *_args: None)
    monkeypatch.setattr(runtime, "capture_system_provenance", lambda *_args: PROVENANCE)
    marker = raw_root / "child" / "marker"
    marker.write_bytes(b"original")
    # Construct the handle directly so the stored tree digest represents the
    # pre-tamper state and no execution path is needed for the test.
    handle = runtime.ScreeningRuntime(
        manifest_path=committed,
        raw_root=raw_root,
        repository=tmp_path,
        device_policy=DevicePolicy(requested_device="cpu", torch_threads=1),
        manifest_bytes=body,
        manifest=manifest,
        authority_sources=(
            runtime.AuthoritySourceSnapshot(
                "protocol", source, source.read_bytes(), hashlib.sha256(source.read_bytes()).hexdigest()
            ),
        ),
        provenance=PROVENANCE,
        folds=(),
        tree_sha256=runtime._walk_tree_digest(raw_root),
    )
    marker.write_bytes(b"tampered")
    with pytest.raises(TrainingDataArtifactError, match="tree changed"):
        handle.recheck_before_execution()


def test_recheck_recaptures_repository_identity_once_without_reapplying_policy(
    monkeypatch,
    tmp_path,
):
    committed, raw_root, _source, body, _config = _fixture(monkeypatch, tmp_path)
    changed = PROVENANCE.model_copy(update={"git_dirty": True, "git_diff_sha256": "b" * 64})
    captures = [PROVENANCE, changed]
    capture_calls: list[str] = []
    policy_calls: list[str] = []
    monkeypatch.setattr(runtime, "apply_runtime_policy", lambda *_args: policy_calls.append("apply"))

    def capture(*_args):
        capture_calls.append("capture")
        return captures.pop(0)

    monkeypatch.setattr(runtime, "capture_system_provenance", capture)
    handle = runtime.load_screening_runtime(
        committed,
        raw_root,
        tmp_path,
        manifest_bytes_sha256=hashlib.sha256(body).hexdigest(),
        provenance=PROVENANCE,
    )
    with pytest.raises(TrainingDataArtifactError, match="provenance changed"):
        handle.recheck_before_execution()
    assert capture_calls == ["capture", "capture"]
    assert policy_calls == ["apply"]
    assert all(fold.store._execution_ready is False for fold in handle.folds)


def test_recheck_rejects_identical_byte_authority_symlink_substitution(
    monkeypatch,
    tmp_path,
):
    committed, raw_root, source, body, _config = _fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(runtime, "apply_runtime_policy", lambda *_args: None)
    monkeypatch.setattr(runtime, "capture_system_provenance", lambda *_args: PROVENANCE)
    handle = runtime.load_screening_runtime(
        committed,
        raw_root,
        tmp_path,
        manifest_bytes_sha256=hashlib.sha256(body).hexdigest(),
        provenance=PROVENANCE,
    )
    monkeypatch.setattr(runtime, "_recheck_manifest_and_tree", _REAL_RECHECK)
    replacement = tmp_path / "same-authority-bytes.json"
    replacement.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(replacement)
    with pytest.raises(TrainingDataArtifactError, match="contains a symlink"):
        handle.recheck_before_execution()
