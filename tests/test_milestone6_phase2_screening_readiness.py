"""Contract tests for the development-only Phase 2 screening readiness gate.

These tests deliberately replace the data/model builders with typed fakes.  Readiness
must prove the six-fold inventory and lineage contract without executing a runner unit,
search, replay, evaluator, oracle, or reducer operation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase2_screening_readiness as readiness
from levelup.experiments.milestone6_phase2_screening import (
    C,
    screening_child_configs,
)
from levelup.experiments.milestone6_phase2_screening_preparation import (
    ScreeningDataManifests,
    build_screening_data_keys,
    build_screening_model_keys,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import SystemProvenance
from levelup.experiments.runner.training_artifacts import (
    TrainingArtifactManifest,
    TrainingReportMetadata,
)
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataArtifactError,
    TrainingDataArtifactManifest,
    TrainingDataEvidenceManifest,
)

FAMILY_ORDER = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
PROVENANCE = SystemProvenance(
    git_commit_sha="0" * 40,
    git_dirty=False,
    python_version="test-python",
    packages={"levelup-bench": "test"},
    installed_packages_sha256="a" * 64,
    os="test-os",
    architecture="test-arch",
    cpu="test-cpu",
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
_REAL_BUILD_PLAN = readiness.build_screening_readiness_plan
_REAL_SCREENING_CONFIGS = screening_child_configs
_REAL_BUILD_DATA_KEYS = build_screening_data_keys
_REAL_BUILD_MODEL_KEYS = build_screening_model_keys
_REAL_BUILD_SHARED_PLAN = readiness.build_screening_shared_plan
_PLAN_CACHE = None
_CONFIG_CACHE = None
_DATA_KEY_CACHE = {}
_MODEL_KEY_CACHE = {}
_SHARED_PLAN_CACHE = {}


def _digest(*parts: object) -> str:
    value: object = parts[0] if len(parts) == 1 else parts
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fake_manifests(config, data_keys) -> ScreeningDataManifests:
    evidence = {}
    views = {}
    family = config.parameters["heldout_family_id"]
    for replicate, key in data_keys.evidence.items():
        body = {
            "schema_version": "runner.training-data-evidence.v1",
            "evidence_key_id": key.key_id,
            "key": key.model_dump(mode="json"),
            "payload_sha256": _digest("payload", family, replicate),
            "payload_bytes": 1,
            "sample_task_ids": key.ordered_training_task_ids,
        }
        evidence[replicate] = TrainingDataEvidenceManifest(
            evidence_id=_digest(body),
            **body,
        )
    for identity, key in data_keys.views.items():
        source = evidence[identity[1]]
        body = {
            "schema_version": "runner.training-data-manifest.v1",
            "evidence_id": source.evidence_id,
            "key_id": key.key_id,
            "key": key.model_dump(mode="json"),
            "payload_sha256": source.payload_sha256,
            "payload_bytes": source.payload_bytes,
            "sample_task_ids": source.sample_task_ids,
        }
        views[identity] = TrainingDataArtifactManifest(
            artifact_id=_digest(body),
            **body,
        )
    return ScreeningDataManifests(evidence=evidence, views=views)


def _fake_data(config, data_keys):
    manifests = _fake_manifests(config, data_keys)
    return SimpleNamespace(
        keys=data_keys,
        manifests=manifests,
        evidence_cost_ids={r: _digest("evidence-cost", r) for r in range(5)},
        view_cost_ids={identity: _digest("view-cost", *identity) for identity in data_keys.views},
    )


def _fake_models(config, data, keys):
    return SimpleNamespace(
        keys=keys,
        manifests={
            identity: TrainingArtifactManifest.model_construct(
                artifact_id=_digest("model", config.parameters["heldout_family_id"], *identity),
                key=key,
                model_id=key.backbone_id,
                tensors=(),
                report=TrainingReportMetadata(
                    trainable_parameters=3601 if identity[0] != C else 3841,
                    training_examples=40,
                    optimizer_steps=120,
                    forward_passes=120,
                ),
            )
            for identity, key in keys.models.items()
        },
        costs={identity: _digest("model-cost", *identity) for identity in keys.models},
        compute={
            identity: SimpleNamespace(
                model_id=key.backbone_id,
                objective_id=key.objective_id,
                trainable_parameters=3601 if identity[0] != C else 3841,
                training_examples=40,
                optimizer_steps=120,
                forward_passes=120,
                training_wall_seconds=0.001,
            )
            for identity, key in keys.models.items()
        },
    )


def _fake_namespace(root: Path, name: str, count: int, *, directories: bool = False) -> None:
    namespace = root / name
    namespace.mkdir(exist_ok=True)
    for index in range(count):
        entry = namespace / f"artifact-{index:03d}"
        if directories:
            entry.mkdir(exist_ok=True)
        else:
            entry.touch(exist_ok=True)


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Install a deterministic backend that cannot accidentally execute science."""

    calls: list[str] = []
    provenance_calls: list[str] = []
    policy_calls: list[str] = []
    provenance = PROVENANCE
    monkeypatch.setattr(readiness, "ROOT", tmp_path)

    def cached_plan():
        global _PLAN_CACHE
        if _PLAN_CACHE is None:
            _PLAN_CACHE = _REAL_BUILD_PLAN()
        return _PLAN_CACHE

    def cached_configs():
        global _CONFIG_CACHE
        if _CONFIG_CACHE is None:
            _CONFIG_CACHE = _REAL_SCREENING_CONFIGS()
        return _CONFIG_CACHE

    def cached_data_keys(config, captured):
        family = str(config.parameters["heldout_family_id"])
        if family not in _DATA_KEY_CACHE:
            _DATA_KEY_CACHE[family] = _REAL_BUILD_DATA_KEYS(config, captured)
        return _DATA_KEY_CACHE[family]

    def cached_model_keys(config, data_keys, manifests):
        family = str(config.parameters["heldout_family_id"])
        if family not in _MODEL_KEY_CACHE:
            _MODEL_KEY_CACHE[family] = _REAL_BUILD_MODEL_KEYS(config, data_keys, manifests)
        return _MODEL_KEY_CACHE[family]

    def cached_shared_plan(config, data_keys, manifests, model_keys):
        family = str(config.parameters["heldout_family_id"])
        if family not in _SHARED_PLAN_CACHE:
            _SHARED_PLAN_CACHE[family] = _REAL_BUILD_SHARED_PLAN(
                config, data_keys, manifests, model_keys
            )
        return _SHARED_PLAN_CACHE[family]

    def fake_prepare_data(config, data_keys, output_root, **kwargs):
        calls.append("data")
        root = Path(output_root)
        for name, count, directories in (
            ("screening-data-intents", 5, False),
            ("training-data-evidence-costs", 5, False),
            ("training-data-view-costs", 15, False),
            ("training-data-artifact-keys", 15, False),
            ("training-data-evidence", 5, True),
            ("training-data-artifacts", 15, True),
        ):
            _fake_namespace(root, name, count, directories=directories)
        return _fake_data(config, data_keys)

    def fake_prepare_models(config, data_keys, data, model_keys, output_root, **kwargs):
        calls.append("models")
        root = Path(output_root)
        for name, count, directories in (
            ("screening-model-intents", 60, False),
            ("training-artifact-costs", 60, False),
            ("training-artifact-keys", 60, False),
            ("training-artifacts", 60, True),
        ):
            _fake_namespace(root, name, count, directories=directories)
        return _fake_models(config, data, model_keys)

    def capture(*args, **kwargs):
        provenance_calls.append("capture")
        return provenance

    monkeypatch.setattr(readiness, "capture_system_provenance", capture)
    monkeypatch.setattr(
        readiness,
        "apply_runtime_policy",
        lambda *args, **kwargs: policy_calls.append("apply"),
    )
    monkeypatch.setattr(readiness, "materialize_screening_data", fake_prepare_data)
    monkeypatch.setattr(readiness, "materialize_screening_models", fake_prepare_models)
    monkeypatch.setattr(readiness, "build_screening_readiness_plan", cached_plan)
    monkeypatch.setattr(readiness, "screening_child_configs", cached_configs)
    monkeypatch.setattr(readiness, "build_screening_data_keys", cached_data_keys)
    monkeypatch.setattr(readiness, "build_screening_model_keys", cached_model_keys)
    monkeypatch.setattr(readiness, "build_screening_shared_plan", cached_shared_plan)

    forbidden = {
        "execute": "runner.execute",
        "search": "search",
        "evaluator": "evaluator",
        "oracle": "oracle",
        "aggregate": "aggregate",
    }
    for name, label in forbidden.items():
        monkeypatch.setattr(
            readiness,
            name,
            lambda *args, _label=label, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"readiness called forbidden {_label}")
            ),
            raising=False,
        )
    return tmp_path, calls, provenance, provenance_calls, policy_calls


def _run(fake_backend, tmp_path: Path):
    _root, calls, _provenance, _provenance_calls, _policy_calls = fake_backend
    return readiness.prepare_screening_readiness(
        tmp_path,
        repository=tmp_path,
    ), calls


def test_six_child_run_dirs_have_one_run_id_and_exact_readiness_inventory(
    fake_backend,
    tmp_path: Path,
) -> None:
    result, calls = _run(fake_backend, tmp_path)
    canonical_children = result.manifest.children
    assert len(canonical_children) == 6
    assert {child.heldout_family_id for child in canonical_children} == set(FAMILY_ORDER)
    assert len(calls) == 12
    assert fake_backend[3] == ["capture"]
    assert fake_backend[4] == ["apply"]
    for child in canonical_children:
        assert child.run_id in {path.name for path in tmp_path.iterdir()}
        assert child.run_id.count("--") >= 0
        assert child.expected_units == 1520
        assert child.expected_evidence_artifacts == 5
        assert child.expected_training_data_views == 15
        assert child.expected_model_artifacts == 60
        assert child.expected_shared_artifacts == 80
        assert len(child.evidence_key_ids) == 5
        assert len(child.view_key_ids) == 15
        assert len(child.model_key_ids) == 60
        assert len(child.shared_artifact_key_ids) == 80
        assert len(child.compute_reports) == 60
        assert not (tmp_path / child.run_id / "aggregate.json").exists()
    assert result.development_only is True
    assert result.final_family_access is False
    assert result.validation_executed is False
    assert result.search_executed is False
    assert result.outcomes_present is False
    assert result.selection_performed is False


def test_dirty_capture_is_rejected_before_any_materializer_call(
    fake_backend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, calls, _provenance, _captures, _policies = fake_backend
    dirty = PROVENANCE.model_copy(update={"git_dirty": True, "git_diff_sha256": "c" * 64})
    monkeypatch.setattr(readiness, "capture_system_provenance", lambda *_args: dirty)
    with pytest.raises(TrainingDataArtifactError, match="requires a clean repository"):
        readiness.prepare_screening_readiness(tmp_path, repository=tmp_path)
    assert calls == []


def test_preparation_rejects_repository_distinct_from_authority_checkout(
    fake_backend,
    tmp_path: Path,
) -> None:
    _root, calls, _provenance, captures, _policies = fake_backend
    other_repository = tmp_path / "other-repository"
    other_repository.mkdir()
    with pytest.raises(
        TrainingDataArtifactError,
        match="canonical authority checkout",
    ):
        readiness.prepare_screening_readiness(
            tmp_path / "output",
            repository=other_repository,
        )
    assert calls == []
    assert captures == []


def test_all_folds_share_exact_provenance_and_sorted_identity_order(
    fake_backend,
    tmp_path: Path,
) -> None:
    result, _ = _run(fake_backend, tmp_path)
    assert result.provenance == PROVENANCE
    assert tuple(child.heldout_family_id for child in result.manifest.children) == FAMILY_ORDER
    for child in result.manifest.children:
        assert child.evidence_key_ids == tuple(sorted(child.evidence_key_ids))
        assert child.view_key_ids == tuple(sorted(child.view_key_ids))
        assert child.model_key_ids == tuple(sorted(child.model_key_ids))
        assert child.shared_artifact_key_ids == tuple(sorted(child.shared_artifact_key_ids))
        assert child.model_artifact_ids == tuple(sorted(child.model_artifact_ids))
        assert child.compute_reports == tuple(
            sorted(
                child.compute_reports,
                key=lambda report: (
                    report.base_condition_id,
                    report.training_tuple_id,
                    report.replicate,
                ),
            )
        )


def test_shared_plan_and_typed_cost_lineage_are_bound_to_exact_artifacts(
    fake_backend,
    tmp_path: Path,
) -> None:
    result, _ = _run(fake_backend, tmp_path)
    for child in result.manifest.children:
        assert len(child.shared_artifact_key_ids) == 80
        assert len(set(child.shared_artifact_key_ids)) == 80
        assert len(child.model_key_ids) == len(child.model_artifact_ids) == 60
        assert len(child.model_manifest_sha256) == 64
        assert len(child.data_manifest_sha256) == 64
        assert len(child.shared_plan_sha256) == 64
        for report in child.compute_reports:
            assert len(report.model_key_id) == 64
            assert len(report.model_artifact_id) == 64
            assert report.training_examples > 0
            assert report.optimizer_steps > 0
            assert report.forward_passes > 0


def test_resume_is_atomic_and_does_not_rematerialize_completed_children(
    fake_backend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls = _run(fake_backend, tmp_path)
    assert len(calls) == 12
    calls.clear()
    resumed, resumed_calls = _run(fake_backend, tmp_path)
    assert resumed.manifest == result.manifest
    # The public preparation boundaries are re-entered so that their typed
    # loaders can revalidate every durable artifact; the forbidden execution
    # channels remain monkeypatched to fail if any science is repeated.
    assert resumed_calls == ["data", "models"] * 6
    assert all(not path.name.startswith(".") for path in tmp_path.rglob("*"))
    assert resumed.child_run_ids == tuple(child.run_id for child in resumed.manifest.children)


def test_symlinked_output_root_is_rejected_before_any_materialization(
    fake_backend,
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    _root, calls, _provenance, _captures, _policies = fake_backend
    with pytest.raises((TrainingDataArtifactError, ValueError)):
        readiness.prepare_screening_readiness(
            linked_root,
            repository=tmp_path,
        )
    assert calls == []


def test_symlinked_output_ancestor_is_rejected_before_any_materialization(
    fake_backend,
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    _root, calls, _provenance, _captures, _policies = fake_backend
    with pytest.raises(TrainingDataArtifactError):
        readiness.prepare_screening_readiness(
            linked_parent / "new-root",
            repository=tmp_path,
        )
    assert calls == []


def test_manifest_loader_rejects_symlinked_ancestor(
    fake_backend,
    tmp_path: Path,
) -> None:
    result, _ = _run(fake_backend, tmp_path)
    assert result.manifest is not None
    linked_parent = tmp_path.parent / f"{tmp_path.name}-linked"
    linked_parent.symlink_to(tmp_path, target_is_directory=True)
    try:
        with pytest.raises(RuntimeError, match="contains a symlink"):
            readiness.load_screening_readiness_manifest(
                linked_parent / "phase2-screening-readiness.json"
            )
    finally:
        linked_parent.unlink()


def test_child_run_path_escape_is_rejected_before_materialization(
    fake_backend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, calls, _provenance, _captures, _policies = fake_backend
    monkeypatch.setattr(readiness, "run_id_for", lambda config: "../escape")
    with pytest.raises((TrainingDataArtifactError, ValueError)):
        readiness.prepare_screening_readiness(
            tmp_path,
            repository=tmp_path,
        )
    assert calls == []


def test_authority_change_during_materialization_fails_closed(
    fake_backend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls = _run(fake_backend, tmp_path)
    assert result.children
    calls.clear()
    original = readiness._assert_source_snapshot
    checks = 0

    def changed_snapshot(*args, **kwargs):
        nonlocal checks
        checks += 1
        if checks >= 3:
            raise TrainingDataArtifactError("screening authority changed before materialization")
        return original(*args, **kwargs)

    monkeypatch.setattr(readiness, "_assert_source_snapshot", changed_snapshot)
    with pytest.raises((TrainingDataArtifactError, ValueError)):
        _run(fake_backend, tmp_path)
    assert calls == []


def test_resume_accepts_new_capture_timestamp_but_preserves_first_provenance(
    fake_backend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _run(fake_backend, tmp_path)
    later = PROVENANCE.model_copy(
        update={"captured_at_utc": PROVENANCE.captured_at_utc + timedelta(seconds=1)}
    )
    monkeypatch.setattr(
        readiness,
        "capture_system_provenance",
        lambda *args, **kwargs: later,
    )
    resumed, _ = _run(fake_backend, tmp_path)
    assert resumed.provenance == result.provenance == PROVENANCE
    assert resumed.provenance_sha256 == result.provenance_sha256


def test_crash_resume_without_parent_manifest_adopts_child_first_writer_provenance(
    fake_backend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global _PLAN_CACHE, _CONFIG_CACHE
    result, _ = _run(fake_backend, tmp_path)
    (tmp_path / "phase2-screening-readiness.json").unlink()
    _PLAN_CACHE = None
    _CONFIG_CACHE = None
    _DATA_KEY_CACHE.clear()
    _MODEL_KEY_CACHE.clear()
    _SHARED_PLAN_CACHE.clear()
    later = PROVENANCE.model_copy(
        update={"captured_at_utc": PROVENANCE.captured_at_utc + timedelta(seconds=1)}
    )
    monkeypatch.setattr(
        readiness,
        "capture_system_provenance",
        lambda *args, **kwargs: later,
    )
    resumed, _ = _run(fake_backend, tmp_path)
    assert resumed.provenance == result.provenance == PROVENANCE
    assert (tmp_path / "phase2-screening-readiness.json").is_file()


@pytest.mark.parametrize(
    "kind",
    ("conflict", "partial", "symlink", "known_type", "known_symlink", "toctou"),
)
def test_existing_or_racing_readiness_state_fails_closed_without_science(
    fake_backend,
    tmp_path: Path,
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls = _run(fake_backend, tmp_path)
    assert result.children
    calls.clear()
    child = result.children[-1]
    if kind == "conflict":
        marker = tmp_path / "phase2-screening-readiness.json"
        marker.write_text(json.dumps({"run_id": "foreign"}), encoding="utf-8")
    elif kind == "partial":
        (child.run_dir / "screening-readiness-staging").mkdir()
    elif kind == "symlink":
        marker = tmp_path / "phase2-screening-readiness.json"
        marker.unlink()
        marker.symlink_to(child.run_dir)
    elif kind == "known_type":
        marker = child.run_dir / "units"
        marker.rmdir()
        marker.write_text("not a directory", encoding="utf-8")
    elif kind == "known_symlink":
        marker = child.run_dir / "config.json"
        marker.unlink()
        marker.symlink_to(child.run_dir / "provenance.json")
    else:
        monkeypatch.setattr(
            readiness,
            "_assert_source_snapshot",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                TrainingDataArtifactError("TOCTOU conflict")
            ),
            raising=False,
        )
    with pytest.raises((TrainingDataArtifactError, ValueError)):
        _run(fake_backend, tmp_path)
    assert calls == []


def test_readiness_does_not_construct_final_tasks_or_invoke_runner_channels(
    fake_backend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        readiness,
        "screening_child_configs",
        lambda: tuple(screening_child_configs()),
    )
    result, _ = _run(fake_backend, tmp_path)
    assert result.development_only is True
    assert result.final_family_access is False
    assert result.outcomes_present is False
    assert all(not (child.run_dir / "aggregate.json").exists() for child in result.children)
