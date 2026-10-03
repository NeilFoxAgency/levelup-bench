from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from levelup.experiments.milestone6_phase2_screening_runtime import (
    ScreeningRuntime,
    ScreeningRuntimeFold,
)
from levelup.experiments.milestone6_phase3_evidence import (
    _EVIDENCE_LOCK_TOKEN,
    Phase3EvidenceLock,
)
from levelup.experiments.milestone6_phase3_local_affordance_training_source import (
    LocalAffordanceTrainingSourceError,
    load_local_affordance_training_source,
)
from levelup.experiments.milestone6_phase3_protocol import FAMILIES
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import TrainingPreparationAccounting
from levelup.experiments.runner.secure_fs import open_directory_chain
from levelup.experiments.runner.training_data_artifacts import (
    _CANONICAL_SANITIZED_DATA_TOKEN,
    AffordanceTableRecord,
    ObservableStateRecord,
    ObservableTraceRecord,
    ObservedTransitionRecord,
    SanitizedTrainingData,
    TrainingDataArtifactKey,
    TrainingDataEvidencePayloadBundle,
    TrainingDataSample,
    evidence_key_for,
    load_training_data_evidence,
    write_training_data_artifact,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _source(tmp_path: Path, *, swap_pinned_path: bool = False):
    family = "plain"
    replicate = 0
    task_ids = tuple(f"{family}-train-{index:02d}" for index in range(40))
    heldout_ids = tuple(f"{family}-heldout-{index:02d}" for index in range(8))
    key = TrainingDataArtifactKey(
        screening_candidates_sha256=_sha("screening"),
        protocol_sha256=_sha("protocol"),
        task_manifest_sha256=_sha("tasks"),
        expected_unit_plan_sha256=_sha("units"),
        provenance_sha256=_sha("provenance"),
        reference_exposure_sha256=_sha("exposure"),
        representation_sha256=_sha("representation"),
        probe_policy_sha256=_sha("probe"),
        fold_id=f"lofo-{family}",
        heldout_family_id=family,
        ordered_training_task_ids=task_ids,
        ordered_heldout_task_ids=heldout_ids,
        condition_id="B2",
        objective_id="listwise_optimum",
        replicate=replicate,
        data_order_seed=100,
        probe_seeds=tuple(range(1000, 1040)),
        environment_seeds=tuple(range(2000, 2040)),
    )
    before = ObservableStateRecord(
        progress_fraction=0.0,
        remaining_fraction=1.0,
        elapsed_per_target=0.0,
        resource_fraction=1.0,
        pressure_fraction=0.0,
        available_aliases=("wait",),
    )
    after = before.model_copy(update={"progress_fraction": 1.0, "remaining_fraction": 0.0})
    trace = ObservableTraceRecord(
        transitions=(
            ObservedTransitionRecord(
                before=before, action_alias="wait", after=after, completed=True
            ),
        )
    )
    samples = tuple(
        TrainingDataSample(
            task_id=task_id,
            trace=trace,
            affordances=AffordanceTableRecord(
                features={"wait": (1.0,) * 49}, sample_counts={"wait": 1}
            ),
        )
        for task_id in task_ids
    )
    data = SanitizedTrainingData(samples, _construction_token=_CANONICAL_SANITIZED_DATA_TOKEN)
    run_dir = tmp_path / "run"
    written = write_training_data_artifact(
        run_dir,
        key,
        data,
        evidence_accounting=TrainingPreparationAccounting(),
        view_accounting=TrainingPreparationAccounting(),
    )
    evidence_key = evidence_key_for(key)
    manifest, _payload = load_training_data_evidence(run_dir, written.evidence_id, expected_key=evidence_key)
    manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
    row = {
        "family_id": family,
        "fold_id": key.fold_id,
        "replicate": replicate,
        "child_run_id": "source-child-run",
        "evidence_key_id": evidence_key.key_id,
        "evidence_key": evidence_key.model_dump(mode="json"),
        "evidence_id": manifest.evidence_id,
        "evidence_manifest_key_id": manifest.evidence_key_id,
        "evidence_manifest": manifest.model_dump(mode="json"),
        "payload_sha256": manifest.payload_sha256,
        "payload_bytes": manifest.payload_bytes,
        "ordered_training_task_ids": list(task_ids),
        "canonical_manifest_bytes_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    rows = [
        {"family_id": known_family, "replicate": known_replicate}
        for known_family in FAMILIES
        for known_replicate in range(5)
    ]
    rows[rows.index({"family_id": family, "replicate": replicate})] = row
    body = {
        "schema_version": "milestone6.phase3.evidence-lock.v1",
        "scope": "known-development-only",
        "final_family_access": False,
        "payloads_included": False,
        "outcomes_included": False,
        "aggregates": [],
        "final_results": [],
        "counts": {"evidence_artifacts": 30, "families": 6, "replicates": 5},
        "evidence_artifacts": rows,
    }
    digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    body["evidence_lock_sha256"] = digest
    lock = Phase3EvidenceLock(
        body=body,
        canonical_bytes=canonical_json_bytes(body),
        evidence_lock_sha256=digest,
        _construction_token=_EVIDENCE_LOCK_TOKEN,
    )

    class Store:
        run_id = "source-child-run"

        @contextlib.contextmanager
        def _open_pinned_run(self):
            descriptor = open_directory_chain(run_dir)
            try:
                if swap_pinned_path:
                    detached = tmp_path / "detached"
                    run_dir.rename(detached)
                    run_dir.symlink_to(tmp_path / "decoy", target_is_directory=True)
                    (tmp_path / "decoy").mkdir()
                yield descriptor
            finally:
                os.close(descriptor)

    fold = ScreeningRuntimeFold(
        family_id=family,
        config=SimpleNamespace(
            parameters={"fold_id": key.fold_id},
            split=SimpleNamespace(
                development_tasks=tuple(
                    SimpleNamespace(task_id=task_id) for task_id in task_ids
                )
            ),
        ),
        store=Store(),
        data_keys=None,
        data=None,
        model_keys=None,
        models=None,
        shared_plan=None,
    )
    folds = (fold,) + tuple(
        ScreeningRuntimeFold(
            family_id=other,
            config=None,
            store=None,
            data_keys=None,
            data=None,
            model_keys=None,
            models=None,
            shared_plan=None,
        )
        for other in FAMILIES
        if other != family
    )
    runtime = ScreeningRuntime(
        manifest_path=tmp_path / "manifest.json",
        raw_root=tmp_path,
        repository=tmp_path,
        device_policy=None,
        manifest_bytes=b"{}",
        manifest=None,
        authority_sources=(),
        provenance=None,
        folds=folds,
        tree_sha256=_sha("tree"),
        raw_root_identity=(1, 1),
        child_identities=((family, (1, 1)),),
        manifest_parent_identity=(1, 1),
        manifest_file_identity=None,
        authority_repository=tmp_path,
        authority_repository_identity=(1, 1),
        authority_provenance=None,
    )
    return runtime, lock, row, run_dir


def test_loads_only_requested_locked_training_bundle(monkeypatch, tmp_path: Path) -> None:
    runtime, lock, row, _run_dir = _source(tmp_path)
    from levelup.experiments import milestone6_phase3_local_affordance_training_source as source

    monkeypatch.setattr(source, "recheck_screening_runtime_readonly", lambda _runtime: None)
    bundle = load_local_affordance_training_source(
        runtime, lock, family_id="plain", replicate=0
    )
    assert type(bundle) is TrainingDataEvidencePayloadBundle
    assert bundle.manifest.evidence_id == row["evidence_id"]
    assert tuple(sample.task_id for sample in bundle.payload.samples) == tuple(
        row["ordered_training_task_ids"]
    )


def test_source_read_stays_on_pinned_run_after_path_substitution(monkeypatch, tmp_path: Path) -> None:
    runtime, lock, row, _run_dir = _source(tmp_path, swap_pinned_path=True)
    from levelup.experiments import milestone6_phase3_local_affordance_training_source as source

    monkeypatch.setattr(source, "recheck_screening_runtime_readonly", lambda _runtime: None)
    bundle = load_local_affordance_training_source(
        runtime, lock, family_id="plain", replicate=0
    )
    assert bundle.manifest.evidence_id == row["evidence_id"]
    assert (tmp_path / "run").is_symlink()


@pytest.mark.parametrize(
    ("family_id", "replicate"),
    [("plain", True), ("not-a-family", 0), ("plain", 5)],
)
def test_rejects_invalid_development_identity(
    monkeypatch, tmp_path: Path, family_id: str, replicate: int
) -> None:
    runtime, lock, _row, _run_dir = _source(tmp_path)
    from levelup.experiments import milestone6_phase3_local_affordance_training_source as source

    monkeypatch.setattr(source, "recheck_screening_runtime_readonly", lambda _runtime: None)
    with pytest.raises(LocalAffordanceTrainingSourceError):
        load_local_affordance_training_source(
            runtime, lock, family_id=family_id, replicate=replicate
        )


def test_rejects_locked_payload_hash_drift(monkeypatch, tmp_path: Path) -> None:
    runtime, lock, row, _run_dir = _source(tmp_path)
    row["payload_sha256"] = _sha("wrong payload")
    from levelup.experiments import milestone6_phase3_local_affordance_training_source as source

    monkeypatch.setattr(source, "recheck_screening_runtime_readonly", lambda _runtime: None)
    with pytest.raises(LocalAffordanceTrainingSourceError):
        load_local_affordance_training_source(
            runtime, lock, family_id="plain", replicate=0
        )


def test_rejects_wrong_runtime_task_order(monkeypatch, tmp_path: Path) -> None:
    runtime, lock, _row, _run_dir = _source(tmp_path)
    runtime.folds[0].config.split.development_tasks = tuple(
        reversed(runtime.folds[0].config.split.development_tasks)
    )
    from levelup.experiments import milestone6_phase3_local_affordance_training_source as source

    monkeypatch.setattr(source, "recheck_screening_runtime_readonly", lambda _runtime: None)
    with pytest.raises(LocalAffordanceTrainingSourceError, match="runtime fold"):
        load_local_affordance_training_source(
            runtime, lock, family_id="plain", replicate=0
        )
