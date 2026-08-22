from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase3_model_preparation as preparation
from levelup.experiments.milestone6_phase3_model_artifacts import (
    Phase3ModelArtifactKey,
    Phase3OptimizerSpec,
    Phase3TrainingReport,
    open_phase3_model_output,
)
from levelup.experiments.milestone6_phase3_model_preparation import (
    EXPECTED_MODELS,
    Phase3ModelPreparationError,
    Phase3ModelPreparationProgress,
    _atomic_progress_at,
    _ensure_preparation_provenance_at,
    _evidence_rows,
    _model_identity_sha256,
    _read_progress_at,
    _validate_bundle_lineage,
    _validate_progress_preparation_provenance,
    _validate_resumed_model_accounting,
)
from levelup.experiments.runner.records import SystemProvenance
from levelup.experiments.runner.storage import provenance_identity_sha256


def _progress(**overrides: object) -> Phase3ModelPreparationProgress:
    body: dict[str, object] = {
        "plan_id": "a" * 64,
        "protocol_sha256": "b" * 64,
        "anchor_manifest_sha256": "c" * 64,
        "evidence_lock_sha256": "d" * 64,
        "expected_owner_ids": tuple(f"{index:064x}" for index in range(EXPECTED_MODELS)),
        "completed_owner_ids": (),
        "evidence_count": 30,
        "view_count": 120,
        "model_count": 0,
        "preparation_git_commit_sha": "e" * 40,
        "preparation_provenance_sha256": "f" * 64,
    }
    body.update(overrides)
    return Phase3ModelPreparationProgress.model_validate(body)


def test_progress_cannot_claim_complete_for_partial_owner_set() -> None:
    with pytest.raises(ValueError, match="cannot be complete"):
        _progress(status="complete", completed_owner_ids=())


def test_progress_rejects_extra_owner() -> None:
    with pytest.raises(ValueError, match="unexpected completed owner"):
        _progress(completed_owner_ids=("f" * 64,))


def test_progress_requires_matching_count_and_canonical_newline(tmp_path) -> None:
    progress = _progress()
    path = tmp_path / preparation.PROGRESS_NAME
    with open_phase3_model_output(tmp_path) as output:
        _atomic_progress_at(output, progress)
        assert path.read_bytes().endswith(b"\n")
        assert _read_progress_at(output) == progress
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
        with pytest.raises(Phase3ModelPreparationError, match="canonical"):
            _read_progress_at(output)

    with pytest.raises(ValueError, match="model count"):
        _progress(model_count=1)


def test_evidence_rows_require_exact_typed_lineage() -> None:
    lock = SimpleNamespace(
        body={
            "evidence_artifacts": [
                {
                    "family_id": "combo",
                    "fold_id": "fold-combo",
                    "replicate": 0,
                    "evidence_key": {"bad": "lineage"},
                    "evidence_manifest": {},
                }
            ]
        }
    )
    with pytest.raises(Phase3ModelPreparationError, match="exact 30-row"):
        _evidence_rows(lock)  # type: ignore[arg-type]


def test_descriptor_bundle_manifest_bytes_are_bound_to_evidence_row() -> None:
    import hashlib

    manifest = SimpleNamespace(payload_sha256=hashlib.sha256(b"abc").hexdigest(), payload_bytes=3)
    bundle = SimpleNamespace(
        manifest=manifest,
        manifest_bytes=b"manifest",
        payload_bytes=b"abc",
    )
    evidence = {
        "manifest": manifest,
        "row": {
            "canonical_manifest_bytes_sha256": hashlib.sha256(b"manifest").hexdigest(),
            "payload_sha256": hashlib.sha256(b"abc").hexdigest(),
            "payload_bytes": 3,
        },
    }
    _validate_bundle_lineage(bundle, evidence)
    evidence["row"]["canonical_manifest_bytes_sha256"] = "f" * 64
    with pytest.raises(Phase3ModelPreparationError, match="descriptor-read evidence"):
        _validate_bundle_lineage(bundle, evidence)


def _resume_fixture(*, history: bool = False) -> tuple[object, object, Phase3ModelArtifactKey]:
    condition = (
        "H4-causal-history-transition-listwise-optimum"
        if history
        else "S-state-availability-listwise-optimum"
    )
    owner = SimpleNamespace(
        owner_id="a" * 64,
        condition_id=condition,
        view_id="b" * 64,
        model_seed=11,
        training_tuple_id="lr0p003-e120",
        learning_rate=0.003,
        training_epochs=120,
    )
    view = SimpleNamespace(
        view_id=owner.view_id,
        condition_id=condition,
        evidence_payload_sha256="c" * 64,
        representation_identity_sha256="d" * 64,
        history_shuffle=None,
    )
    examples = (
        (SimpleNamespace(history_features=SimpleNamespace(shape=(2,))),)
        if history
        else (SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    )
    prepared_view = SimpleNamespace(
        examples=examples,
        view=view,
        evidence_payload_sha256=view.evidence_payload_sha256,
        representation_identity_sha256=view.representation_identity_sha256,
        history_shuffle=None,
    )
    state_sha = "e" * 64
    identity = _model_identity_sha256(owner, prepared_view, model_state_sha256=state_sha)
    expected_examples = len(examples)
    expected_recurrent = 120 * 2 if history else 0
    report = Phase3TrainingReport(
        trainable_parameters=3889 if history else 3841,
        optimizer_steps=120,
        forward_passes=120 * expected_examples,
        training_examples=expected_examples,
        recurrent_steps=expected_recurrent,
    )
    key = Phase3ModelArtifactKey(
        plan_id="f" * 64,
        protocol_sha256="1" * 64,
        evidence_lock_sha256="2" * 64,
        evidence_payload_sha256=view.evidence_payload_sha256,
        evidence_payload_bytes=12,
        view_id=owner.view_id,
        owner_id=owner.owner_id,
        condition_id=condition,
        fold_id="fold-combo",
        heldout_family="combo",
        replicate=0,
        training_tuple_id=owner.training_tuple_id,
        model_seed=owner.model_seed,
        architecture_id=("causal-history-gru-mlp-v1" if history else "state-availability-mlp-v1"),
        capacity_parameters=(3889 if history else 3841),
        optimizer=Phase3OptimizerSpec(learning_rate=0.003, weight_decay=0.0001),
        report=report,
        recurrent_steps=expected_recurrent,
        model_identity_sha256=identity,
        model_state_sha256=state_sha,
        preparation_git_commit_sha="3" * 40,
        preparation_provenance_sha256="4" * 64,
    )
    return owner, prepared_view, key


def test_resume_rejects_self_consistent_forged_report_counts() -> None:
    owner, prepared_view, key = _resume_fixture()
    forged = key.model_copy(
        update={
            "report": Phase3TrainingReport(
                trainable_parameters=3841,
                optimizer_steps=120,
                forward_passes=480,
                training_examples=4,
                recurrent_steps=0,
            )
        }
    )
    with pytest.raises(Phase3ModelPreparationError, match="report or identity"):
        _validate_resumed_model_accounting(forged, owner=owner, prepared_view=prepared_view)


def test_resume_rejects_self_consistent_forged_recurrent_and_identity_fields() -> None:
    owner, prepared_view, key = _resume_fixture(history=True)
    forged = key.model_copy(
        update={
            "recurrent_steps": 0,
            "report": Phase3TrainingReport(
                trainable_parameters=3889,
                optimizer_steps=120,
                forward_passes=120,
                training_examples=1,
                recurrent_steps=0,
            ),
            "model_identity_sha256": "0" * 64,
        }
    )
    with pytest.raises(Phase3ModelPreparationError, match="report or identity"):
        _validate_resumed_model_accounting(forged, owner=owner, prepared_view=prepared_view)


def test_explicit_authority_handoff_rejects_partial_inputs() -> None:
    with pytest.raises(Phase3ModelPreparationError, match="retained bytes, and provenance"):
        preparation.prepare_phase3_model_batch(
            "/tmp/phase3-model-preparation-test",
            runtime=object(),
            validated_plan=object(),
            anchor_manifest=object(),
            evidence_lock=object(),
            authority_repository="/tmp/repository",
        )


def _provenance(*, dirty: bool = False) -> SystemProvenance:
    return SystemProvenance.model_construct(
        git_commit_sha="a" * 40,
        git_dirty=dirty,
        git_diff_sha256=("b" * 64 if dirty else None),
        python_version="3.11",
        packages={"torch": "2"},
        installed_packages_sha256="c" * 64,
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
        captured_at_utc=datetime.now(UTC),
    )


def test_preparation_provenance_is_write_once_and_rejects_dirty(tmp_path) -> None:
    clean = _provenance()
    with open_phase3_model_output(tmp_path) as output:
        authority = _ensure_preparation_provenance_at(output, clean)
        assert authority.provenance_sha256 == provenance_identity_sha256(clean)
    later = clean.model_copy(update={"captured_at_utc": clean.captured_at_utc.replace(year=2030)})
    with open_phase3_model_output(tmp_path) as output:
        assert _ensure_preparation_provenance_at(output, later) == authority
    drifted = clean.model_copy(update={"git_commit_sha": "d" * 40})
    with open_phase3_model_output(tmp_path) as output:
        with pytest.raises(Phase3ModelPreparationError, match="changed"):
            _ensure_preparation_provenance_at(output, drifted)
    with open_phase3_model_output(tmp_path / "dirty") as output:
        with pytest.raises(ValueError):
            _ensure_preparation_provenance_at(output, _provenance(dirty=True))


def test_resume_rejects_preparation_provenance_drift() -> None:
    progress = _progress(
        preparation_git_commit_sha="a" * 40,
        preparation_provenance_sha256="b" * 64,
    )
    with pytest.raises(Phase3ModelPreparationError, match="provenance differs"):
        _validate_progress_preparation_provenance(
            progress,
            git_commit_sha="c" * 40,
            provenance_sha256="d" * 64,
        )


def test_explicit_authority_handoff_rejects_changed_repository_bytes_before_prep(tmp_path, monkeypatch) -> None:
    repository = tmp_path / "repository"
    authority_dir = repository / "configs" / "milestone6"
    authority_dir.mkdir(parents=True)
    (authority_dir / "phase3_plan_lock.json").write_bytes(b"plan-a")
    (authority_dir / "phase3_anchor_manifest.json").write_bytes(b"anchor-a")
    (authority_dir / "phase3_evidence_lock.json").write_bytes(b"evidence-changed")
    plan = object()
    class DummyValidatedPlan:
        def __init__(self) -> None:
            self.plan = plan

    validated_plan = DummyValidatedPlan()
    anchor = SimpleNamespace(canonical_bytes=b"anchor-a")
    evidence = SimpleNamespace(canonical_bytes=b"evidence-a", body={})
    monkeypatch.setattr(preparation, "require_phase3_anchor_manifest", lambda *_: None)
    monkeypatch.setattr(preparation, "require_phase3_evidence_lock", lambda *_: None)
    monkeypatch.setattr(preparation, "validate_phase3_plan_lock_bytes", lambda _content: plan)
    monkeypatch.setattr(preparation, "ValidatedPhase3Plan", DummyValidatedPlan)
    repository_fd = preparation.secure_fs.open_directory_chain(repository)
    try:
        repository_identity = preparation.secure_fs.directory_identity(repository_fd)
    finally:
        import os

        os.close(repository_fd)
    with pytest.raises(Phase3ModelPreparationError, match="bytes changed"):
        preparation.prepare_phase3_model_batch(
            tmp_path / "output",
            runtime=SimpleNamespace(repository=repository),
            validated_plan=validated_plan,
            anchor_manifest=anchor,
            evidence_lock=evidence,
            authority_repository=repository,
            authority_repository_identity=repository_identity,
            plan_lock_bytes=b"plan-a",
            anchor_file_bytes=b"anchor-a",
            evidence_lock_bytes=b"evidence-a",
            preparation_provenance=_provenance(),
            limit=0,
        )
