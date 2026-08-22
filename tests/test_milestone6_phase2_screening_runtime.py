"""Contract tests for the development-only screening runtime boundary."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase2_screening_runtime as runtime
from levelup.experiments.runner.config import DevicePolicy, canonical_json_bytes
from levelup.experiments.runner.execution import ExperimentRunner
from levelup.experiments.runner.records import (
    ResourceAccounting,
    SystemProvenance,
    UnitOutcome,
    UnitRecord,
)
from levelup.experiments.runner.storage import RunStore
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


def _tree_snapshot(raw_root: Path, manifest):
    fd = runtime.secure_fs.open_directory_chain(raw_root)
    try:
        return runtime._tree_identities_at(fd, manifest)
    finally:
        os.close(fd)


def _tree_digest(raw_root: Path) -> str:
    fd = runtime.secure_fs.open_directory_chain(raw_root)
    try:
        return runtime._walk_tree_digest_at(fd, canonical_child_ids=("child",))
    finally:
        os.close(fd)


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
    def __init__(self, provenance: SystemProvenance = PROVENANCE) -> None:
        self.provenance = provenance
        self.provenance_sha256 = runtime.provenance_identity_sha256(provenance)

    def model_dump(self, **_kwargs):
        return {
            "schema_version": "test",
            "manifest_sha256": "4" * 64,
            "protocol_sha256": self.protocol_sha256,
            "screening_candidates_sha256": self.screening_candidates_sha256,
            "task_manifest_sha256": self.task_manifest_sha256,
        }


def _fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    provenance: SystemProvenance = PROVENANCE,
):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    manifest = _Manifest(provenance)
    body = canonical_json_bytes(manifest.model_dump()) + b"\n"
    committed = (
        tmp_path / "experiments" / "milestone6_phase2_screening_readiness.json"
    )
    committed.parent.mkdir()
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "child" / "units").mkdir(parents=True)
    (raw_root / "child" / "attempts").mkdir()
    (raw_root / "phase2-screening-readiness.json").write_bytes(body)
    committed.write_bytes(body)
    source = tmp_path / "protocol.json"
    source.write_bytes(b"authority")
    source_bytes, source_parent_identity, source_file_identity = runtime._read_pinned_file(
        source, label="test authority"
    )
    source_snapshot = runtime.AuthoritySourceSnapshot(
        "protocol",
        source,
        source_bytes,
        hashlib.sha256(source_bytes).hexdigest(),
        source_parent_identity,
        source_file_identity,
    )
    config = SimpleNamespace(
        device_policy=SimpleNamespace(requested_device="cpu"),
        parameters={"heldout_family_id": "plain"},
        split=SimpleNamespace(final_tasks=()),
        conditions=(),
    )
    committed_bytes, committed_parent_identity, committed_file_identity = runtime._read_pinned_file(
        committed, label="test manifest"
    )
    def manifest_bytes(_path, pin):
        if pin != hashlib.sha256(body).hexdigest():
            raise TrainingDataArtifactError("supplied pin does not match committed manifest")
        return body, manifest, committed_parent_identity, committed_file_identity

    monkeypatch.setattr(runtime, "_manifest_bytes", manifest_bytes)
    monkeypatch.setattr(runtime, "_authority_sources", lambda _manifest: (source_snapshot,))
    monkeypatch.setattr(runtime, "screening_child_configs", lambda: (config,))
    monkeypatch.setattr(runtime, "_assert_development_manifest", lambda *_args: None)
    monkeypatch.setattr(runtime, "build_screening_plan", lambda: SimpleNamespace())
    monkeypatch.setattr(runtime, "validate_screening_plan", lambda *_args: None)
    monkeypatch.setattr(runtime, "_assert_tree_shape_at", lambda *_args: None)
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

    def validate_for_fixture(preparation, current, **_kwargs):
        if current.git_dirty or current.git_diff_sha256 is not None:
            raise TrainingDataArtifactError("screening runtime provenance changed")
        if preparation.git_commit_sha != current.git_commit_sha:
            raise TrainingDataArtifactError("screening runtime provenance changed")
    monkeypatch.setattr(runtime, "validate_screening_provenance", validate_for_fixture)
    return committed, raw_root, source, body, config


def _readonly_fixture_handle(monkeypatch, tmp_path):
    """Load a small fixture handle with the typed fold fields recheck reads."""

    committed, raw_root, _source, body, config = _fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(runtime, "apply_runtime_policy", lambda *_args: None)
    monkeypatch.setattr(runtime, "capture_system_provenance", lambda *_args: PROVENANCE)
    data_keys = object()
    model_keys = object()
    shared_plan = object()
    monkeypatch.setattr(runtime, "build_screening_data_keys", lambda *_args: data_keys)
    monkeypatch.setattr(runtime, "build_screening_model_keys", lambda *_args: model_keys)
    monkeypatch.setattr(runtime, "build_screening_shared_plan", lambda *_args: shared_plan)
    monkeypatch.setattr(runtime, "_assert_global_inventory", lambda *_args: None)
    monkeypatch.setattr(runtime, "_recheck_manifest_and_tree", lambda *_args: None)
    handle = runtime.load_screening_runtime(
        committed,
        raw_root,
        tmp_path,
        manifest_bytes_sha256=hashlib.sha256(body).hexdigest(),
        provenance=PROVENANCE,
    )
    child = handle.manifest.children[0]
    monkeypatch.setattr(runtime, "_child_manifest", lambda *_args: child)
    fold = runtime.ScreeningRuntimeFold(
        family_id="plain",
        config=config,
        store=handle.folds[0].store,
        data_keys=data_keys,
        data=SimpleNamespace(manifests={}),
        model_keys=model_keys,
        models=object(),
        shared_plan=shared_plan,
    )
    return replace(handle, folds=(fold,)), body


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


def test_runtime_rejects_repository_distinct_from_authority_checkout(
    monkeypatch, tmp_path
):
    committed, raw_root, _source, body, _config = _fixture(monkeypatch, tmp_path)
    other_repository = tmp_path / "other-repository"
    other_repository.mkdir()
    with pytest.raises(TrainingDataArtifactError, match="canonical authority checkout"):
        runtime.load_screening_runtime(
            committed,
            raw_root,
            other_repository,
            manifest_bytes_sha256=hashlib.sha256(body).hexdigest(),
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
    monkeypatch.setattr(runtime, "validate_screening_provenance", lambda *_args, **_kwargs: None)
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
    assert capture_calls == ["capture", "capture", "capture"]


def test_raw_prepublication_manifest_loads_but_cannot_activate(monkeypatch, tmp_path):
    committed, raw_root, _source, body, _config = _fixture(monkeypatch, tmp_path)
    copied = tmp_path / "raw-prepublication-readiness.json"
    copied.write_bytes(committed.read_bytes())
    monkeypatch.setattr(runtime, "apply_runtime_policy", lambda *_args: None)
    monkeypatch.setattr(runtime, "capture_system_provenance", lambda *_args: PROVENANCE)
    monkeypatch.setattr(runtime, "validate_screening_provenance", lambda *_args, **_kwargs: None)
    handle = runtime.load_screening_runtime(
        copied,
        raw_root,
        tmp_path,
        manifest_bytes_sha256=hashlib.sha256(body).hexdigest(),
        provenance=PROVENANCE,
    )
    with pytest.raises(
        TrainingDataArtifactError,
        match="canonical committed readiness manifest",
    ):
        handle.recheck_before_execution()
    assert all(fold.store._execution_ready is False for fold in handle.folds)


def test_runtime_load_and_recheck_accept_exact_artifact_publication_child(
    monkeypatch,
    tmp_path,
):
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.invalid")
    _git(repository, "config", "user.name", "Test")
    (repository / "README").write_text("preparation\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "preparation")
    preparation_sha = _git(repository, "rev-parse", "HEAD")
    preparation = PROVENANCE.model_copy(update={"git_commit_sha": preparation_sha})

    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    committed, raw_root, _source, body, _config = _fixture(
        monkeypatch,
        fixture_root,
        provenance=preparation,
    )
    monkeypatch.setattr(runtime, "ROOT", repository)
    canonical = (
        repository / "experiments" / "milestone6_phase2_screening_readiness.json"
    )
    canonical.parent.mkdir()
    canonical.write_bytes(body)
    _git(repository, "add", "experiments/milestone6_phase2_screening_readiness.json")
    _git(repository, "commit", "-qm", "publish readiness")
    publication_sha = _git(repository, "rev-parse", "HEAD")
    current = PROVENANCE.model_copy(update={"git_commit_sha": publication_sha})
    captures: list[str] = []
    monkeypatch.setattr(runtime, "apply_runtime_policy", lambda *_args: None)
    monkeypatch.setattr(
        runtime,
        "capture_system_provenance",
        lambda *_args: captures.append("capture") or current,
    )
    from levelup.experiments.milestone6_phase2_screening_provenance import (
        validate_screening_provenance,
    )

    monkeypatch.setattr(runtime, "validate_screening_provenance", validate_screening_provenance)
    handle = runtime.load_screening_runtime(
        canonical,
        raw_root,
        repository,
        manifest_bytes_sha256=hashlib.sha256(body).hexdigest(),
    )
    handle.recheck_before_execution()
    assert captures == ["capture", "capture", "capture"]
    assert all(fold.store._execution_ready is True for fold in handle.folds)


def test_post_activation_provenance_change_clears_all_gates(monkeypatch, tmp_path):
    committed, raw_root, _source, body, _config = _fixture(monkeypatch, tmp_path)
    changed = PROVENANCE.model_copy(
        update={"git_dirty": True, "git_diff_sha256": "b" * 64}
    )
    captures = [PROVENANCE, PROVENANCE, changed]
    monkeypatch.setattr(runtime, "apply_runtime_policy", lambda *_args: None)
    monkeypatch.setattr(runtime, "capture_system_provenance", lambda *_args: captures.pop(0))
    handle = runtime.load_screening_runtime(
        committed,
        raw_root,
        tmp_path,
        manifest_bytes_sha256=hashlib.sha256(body).hexdigest(),
        provenance=PROVENANCE,
    )
    with pytest.raises(TrainingDataArtifactError, match="provenance changed"):
        handle.recheck_before_execution()
    assert captures == []
    assert all(fold.store._execution_ready is False for fold in handle.folds)


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
    monkeypatch.setattr(runtime, "validate_screening_provenance", lambda *_args, **_kwargs: None)
    manifest = _Manifest()
    monkeypatch.setattr(
        runtime,
        "_manifest_bytes",
        lambda _path, _pin: (
            body,
            manifest,
            runtime._read_pinned_file(committed, label="test manifest")[1],
            runtime._read_pinned_file(committed, label="test manifest")[2],
        ),
    )
    monkeypatch.setattr(runtime, "_authority_sources", lambda _manifest: (
        runtime.AuthoritySourceSnapshot(
            "protocol",
            source,
            source.read_bytes(),
            hashlib.sha256(source.read_bytes()).hexdigest(),
            runtime._read_pinned_file(source, label="test authority")[1],
            runtime._read_pinned_file(source, label="test authority")[2],
        ),
    ))
    monkeypatch.setattr(runtime, "screening_child_configs", lambda: ())
    monkeypatch.setattr(runtime, "_assert_development_manifest", lambda *_args: None)
    monkeypatch.setattr(runtime, "_CHILD_TOP_LEVEL_NAMES", frozenset({"marker", "units", "attempts"}))
    monkeypatch.setattr(runtime, "_load_fold", lambda *_args: SimpleNamespace(store=object()))
    monkeypatch.setattr(runtime, "_activate_prepared_batch", lambda *_args: None)
    monkeypatch.setattr(runtime, "apply_runtime_policy", lambda *_args: None)
    monkeypatch.setattr(runtime, "capture_system_provenance", lambda *_args: PROVENANCE)
    marker = raw_root / "child" / "marker"
    marker.write_bytes(b"original")
    # Construct the handle directly so the stored tree digest represents the
    # pre-tamper state and no execution path is needed for the test.
    raw_root_identity, child_identities = _tree_snapshot(raw_root, manifest)
    manifest_parent_identity = runtime._read_pinned_file(committed, label="test manifest")[1]
    manifest_file_identity = runtime._read_pinned_file(committed, label="test manifest")[2]
    handle = runtime.ScreeningRuntime(
        manifest_path=committed,
        raw_root=raw_root,
        repository=tmp_path,
        device_policy=DevicePolicy(requested_device="cpu", torch_threads=1),
        manifest_bytes=body,
        manifest=manifest,
        authority_sources=(
            runtime.AuthoritySourceSnapshot(
                "protocol",
                source,
                source.read_bytes(),
                hashlib.sha256(source.read_bytes()).hexdigest(),
                runtime._read_pinned_file(source, label="test authority")[1],
                runtime._read_pinned_file(source, label="test authority")[2],
            ),
        ),
        provenance=PROVENANCE,
        folds=(),
        tree_sha256=_tree_digest(raw_root),
        raw_root_identity=raw_root_identity,
        child_identities=child_identities,
        manifest_parent_identity=manifest_parent_identity,
        manifest_file_identity=manifest_file_identity,
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


def test_manually_constructed_runtime_without_identities_fails_closed(monkeypatch, tmp_path):
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
    broken = replace(handle, raw_root_identity=None)
    with pytest.raises(TrainingDataArtifactError, match="missing pinned filesystem identities"):
        broken.recheck_before_execution()


def test_recheck_rejects_raw_root_substitution_before_reading_replacement(
    monkeypatch, tmp_path
):
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
    monkeypatch.setattr(runtime, "_recheck_manifest_and_tree", _REAL_RECHECK)
    original = tmp_path / "raw-original"
    raw_root.rename(original)
    shutil.copytree(original, raw_root)
    with pytest.raises(TrainingDataArtifactError, match="raw root identity"):
        handle.recheck_before_execution()


def test_recheck_rejects_same_byte_committed_manifest_replacement(monkeypatch, tmp_path):
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
    monkeypatch.setattr(runtime, "_recheck_manifest_and_tree", _REAL_RECHECK)
    committed.unlink()
    committed.write_bytes(body)
    with pytest.raises(TrainingDataArtifactError, match="committed manifest identity changed"):
        handle.recheck_before_execution()


def test_readonly_recheck_keeps_execution_gates_closed(monkeypatch, tmp_path):
    handle, _body = _readonly_fixture_handle(monkeypatch, tmp_path)
    assert all(fold.store._execution_ready is False for fold in handle.folds)

    runtime.recheck_screening_runtime_readonly(handle)

    assert all(fold.store._execution_ready is False for fold in handle.folds)


def test_readonly_recheck_rejects_canonical_data_key_mutation(monkeypatch, tmp_path):
    handle, _body = _readonly_fixture_handle(monkeypatch, tmp_path)
    mutated_fold = replace(handle.folds[0], data_keys=object())
    mutated = replace(handle, folds=(mutated_fold,))

    with pytest.raises(TrainingDataArtifactError, match="data keys differ"):
        runtime.recheck_screening_runtime_readonly(mutated)
    assert mutated_fold.store._execution_ready is False


def test_readonly_recheck_rejects_child_inventory_mutation(monkeypatch, tmp_path):
    handle, _body = _readonly_fixture_handle(monkeypatch, tmp_path)
    mutated_manifest = _Manifest(handle.provenance)
    mutated_manifest.children = ()
    mutated = replace(handle, manifest=mutated_manifest)

    with pytest.raises(TrainingDataArtifactError, match="fold inventory is incomplete"):
        runtime.recheck_screening_runtime_readonly(mutated)
    assert all(fold.store._execution_ready is False for fold in handle.folds)


def test_readonly_recheck_rejects_manifest_canonical_drift(monkeypatch, tmp_path):
    handle, body = _readonly_fixture_handle(monkeypatch, tmp_path)
    mutated = replace(handle, manifest_bytes=body + b" ")

    with pytest.raises(TrainingDataArtifactError, match="manifest bytes are not canonical"):
        runtime.recheck_screening_runtime_readonly(mutated)
    assert all(fold.store._execution_ready is False for fold in handle.folds)


def test_mutable_result_entries_do_not_change_prepared_tree_digest(tmp_path):
    raw_root = tmp_path / "raw"
    (raw_root / "child" / "units").mkdir(parents=True)
    (raw_root / "child" / "attempts").mkdir()
    before = _tree_digest(raw_root)
    (raw_root / "child" / "units" / ("a" * 64 + ".json")).write_bytes(b"partial")
    (raw_root / "child" / "attempts" / ("a" * 64 + ".attempt-0001.json")).write_bytes(
        b"attempt"
    )
    assert _tree_digest(raw_root) == before


def test_result_namespace_identity_is_bound_by_prepared_tree_digest(tmp_path):
    raw_root = tmp_path / "raw"
    units = raw_root / "child" / "units"
    units.mkdir(parents=True)
    (raw_root / "child" / "attempts").mkdir()
    before = _tree_digest(raw_root)
    replacement = raw_root / "child" / "units-replacement"
    units.rename(replacement)
    units.mkdir()
    assert _tree_digest(raw_root) != before


def test_unexpected_result_entry_names_fail_closed(tmp_path):
    namespace = tmp_path / "units"
    namespace.mkdir()
    (namespace / "unexpected.json").write_bytes(b"{}")
    fd = runtime.secure_fs.open_directory_chain(namespace)
    try:
        with pytest.raises(TrainingDataArtifactError, match="unexpected unit result"):
            runtime._assert_result_namespace_shape_at(fd, "units", {"a" * 64})
    finally:
        os.close(fd)


def test_symlink_result_entry_fails_closed(tmp_path):
    namespace = tmp_path / "attempts"
    namespace.mkdir()
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    (namespace / ("a" * 64 + ".attempt-0001.json")).symlink_to(target)
    fd = runtime.secure_fs.open_directory_chain(namespace)
    try:
        with pytest.raises(TrainingDataArtifactError, match="namespace is unsafe"):
            runtime._assert_result_namespace_shape_at(fd, "attempts", {"a" * 64})
    finally:
        os.close(fd)


def test_partial_and_complete_result_state_is_validated_while_locked():
    calls: list[str] = []

    class Store:
        run_id = "fold"
        _execution_ready = False
        _result_directory_identities = None
        expected = SimpleNamespace(units=())

        def _capture_result_directory_identities(self):
            calls.append("identities")
            return {"run": (1, 1), "units": (1, 2), "attempts": (1, 3)}

        def completed_records(self):
            calls.append("completed")
            return ("complete",)

        def attempt_records(self):
            calls.append("attempts")
            return ("partial",)

    runtime._validate_result_namespaces((SimpleNamespace(store=Store()),))
    assert calls == ["identities", "completed", "attempts"]


def test_corrupt_result_record_fails_closed_through_store_api():
    class Store:
        run_id = "fold"
        _execution_ready = False
        _result_directory_identities = None

        def _capture_result_directory_identities(self):
            return {"run": (1, 1), "units": (1, 2), "attempts": (1, 3)}

        def completed_records(self):
            raise RuntimeError("invalid artifact")

        def attempt_records(self):
            return ()

    with pytest.raises(TrainingDataArtifactError, match="result state is invalid"):
        runtime._validate_result_namespaces((SimpleNamespace(store=Store()),))


def test_real_locked_store_resumes_valid_result_and_snapshot_detects_replacement(
    tmp_path: Path,
) -> None:
    from levelup.experiments.milestone6_phase2_screening import (
        screening_child_configs,
    )

    config = screening_child_configs()[0]
    store = RunStore(tmp_path, config, repository=tmp_path)
    store.units_dir.mkdir(parents=True)
    store.attempts_dir.mkdir()
    store._result_directory_identities = store._capture_result_directory_identities()
    planned = next(
        unit
        for unit in store.expected.units
        if unit.key.condition_id == "A0-no-probe-uniform"
    )
    record = UnitRecord(
        run_id=store.run_id,
        config_sha256=store.config_sha256,
        unit_id=planned.unit_id,
        key=planned.key,
        seeds=planned.seeds,
        exposure_manifest_sha256=planned.exposure_manifest_sha256,
        started_at_utc="2026-08-22T00:00:00+00:00",
        finished_at_utc="2026-08-22T00:00:01+00:00",
        elapsed_wall_seconds=1.0,
        outcome=UnitOutcome(
            evaluator_ran=True,
            valid=False,
            completed=False,
            success=False,
            performance_metric_id="performance_value",
            performance_direction="minimize",
            censored=True,
            censoring_budget=2048,
            censoring_reason="fixed_endpoint",
        ),
        accounting=ResourceAccounting(),
        candidate_generation_sha256="d" * 64,
    )
    assert store.write_completed(record) is True
    store._execution_ready = False
    fold = SimpleNamespace(store=store)

    runtime._validate_result_namespaces((fold,))
    assert store._execution_ready is False
    snapshot = runtime._result_namespace_snapshot((fold,))

    changed = record.model_copy(update={"elapsed_wall_seconds": 2.0})
    (store.units_dir / f"{planned.unit_id}.json").write_bytes(
        canonical_json_bytes(changed.model_dump(mode="json")) + b"\n"
    )
    with pytest.raises(TrainingDataArtifactError, match="snapshot changed"):
        runtime._validate_result_namespaces((fold,), snapshot)

    resumed = RunStore(tmp_path, config, repository=tmp_path)
    resumed._result_directory_identities = resumed._capture_result_directory_identities()
    resumed_fold = SimpleNamespace(store=resumed)
    runtime._validate_result_namespaces((resumed_fold,))
    assert resumed.completed_records() == (changed,)
    assert resumed._execution_ready is False
