from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase3_outcome_diagnostic_model_preparation_driver as driver


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    manifest = repository / driver.CANONICAL_READINESS_PATH
    output = repository / f"{driver.MODEL_OUTPUT_PREFIX}{'a' * 12}"
    return manifest, repository, raw_root, output


def _call(manifest: Path, repository: Path, raw: Path, output: Path, **kwargs):
    return driver.run_outcome_diagnostic_model_preparation(
        manifest,
        "a" * 64,
        raw,
        repository,
        repository,
        output,
        expected_preparation_commit_sha="b" * 40,
        **kwargs,
    )


def test_owner_ids_and_limit_are_mutually_exclusive_before_loading_runtime(tmp_path):
    manifest, repository, raw, output = _paths(tmp_path)
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        _call(manifest, repository, raw, output, owner_ids=("c" * 64,), limit=0)


def test_rejects_symlinked_output_and_raw_overlap(tmp_path):
    manifest, repository, raw, output = _paths(tmp_path)
    output_link = tmp_path / "output-link"
    output_link.symlink_to(output, target_is_directory=True)
    with pytest.raises(RuntimeError, match="output root or an ancestor is a symlink"):
        _call(manifest, repository, raw, output_link)
    with pytest.raises(RuntimeError, match="overlaps raw evidence"):
        _call(manifest, repository, raw, raw / "raw-child")


def test_dirty_or_foreign_preparation_commit_fails_closed(monkeypatch, tmp_path):
    manifest, repository, raw, output = _paths(tmp_path)
    runtime = SimpleNamespace(raw_root=raw, folds=())
    plan = SimpleNamespace(
        plan=SimpleNamespace(
            plan_id="a" * 64,
            model_owners=tuple(range(240)),
            units=tuple(range(5760)),
            evidence_lineage_rows=(),
        )
    )
    monkeypatch.setattr(driver, "load_screening_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(driver, "recheck_screening_runtime_readonly", lambda _: None)
    monkeypatch.setattr(
        driver,
        "load_outcome_group_diagnostic_protocol",
        lambda: SimpleNamespace(repository=repository),
    )
    monkeypatch.setattr(driver, "build_outcome_group_diagnostic_plan", lambda _: object())
    monkeypatch.setattr(
        driver, "bind_validated_outcome_diagnostic_plan", lambda *_args, **_kwargs: plan
    )
    monkeypatch.setattr(
        driver,
        "capture_system_provenance",
        lambda *_args, **_kwargs: SimpleNamespace(
            git_dirty=True,
            git_commit_sha="c" * 40,
            requested_device="cpu",
            resolved_device="cpu",
            requested_torch_threads=1,
            actual_torch_threads=1,
            requested_torch_interop_threads=1,
            actual_torch_interop_threads=1,
            processes=1,
        ),
    )
    with pytest.raises(RuntimeError, match="dirty or differs"):
        _call(manifest, repository, raw, output)


def test_driver_has_no_execution_interfaces():
    source = Path(driver.__file__).read_text()
    assert "prepare_outcome_diagnostic_model_batch" in source
    assert "load_training_data_evidence_payload_bundle_from_at" in source
    assert "open_outcome_diagnostic_result_store" not in source
    assert "run_outcome_diagnostic_unit" not in source


def test_evidence_byte_substitution_is_rejected(monkeypatch):
    class FakeKey:
        def __init__(self, family, replicate):
            self.heldout_family_id = family
            self.replicate = replicate
            self.key_id = f"{family}-{replicate}"
            self.ordered_training_task_ids = (f"{family}-task",)

        @classmethod
        def model_validate(cls, value):
            return cls(value["family_id"], value["replicate"])

        def __eq__(self, other):
            return type(other) is type(self) and self.__dict__ == other.__dict__

    class FakeManifest:
        def __init__(self, key):
            self.key = key
            self.evidence_id = f"evidence-{key.heldout_family_id}-{key.replicate}"
            self.payload_sha256 = hashlib.sha256(b"payload").hexdigest()
            self.payload_bytes = len(b"payload")

        @classmethod
        def model_validate(cls, value):
            return cls(FakeKey(value["family_id"], value["replicate"]))

        def __eq__(self, other):
            return type(other) is type(self) and self.__dict__ == other.__dict__

    class FakePayload:
        samples = (SimpleNamespace(task_id="plain-task"),)

    class FakeBundle:
        manifest_bytes = b"manifest"
        payload_bytes = b"payload"
        payload = FakePayload()

        def __init__(self, manifest):
            self.manifest = manifest

    @contextmanager
    def pinned_run():
        yield 1

    class FakeStore:
        _open_pinned_run = staticmethod(pinned_run)

    families = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
    folds = tuple(SimpleNamespace(family_id=family, store=FakeStore()) for family in families)
    runtime = SimpleNamespace(folds=folds)
    rows = []
    for family in families:
        for replicate in range(5):
            key = {"family_id": family, "replicate": replicate}
            manifest = {"family_id": family, "replicate": replicate}
            rows.append(
                driver.canonical_json_bytes(
                    {
                        "family_id": family,
                        "replicate": replicate,
                        "evidence_key": key,
                        "evidence_manifest": manifest,
                        "evidence_id": f"evidence-{family}-{replicate}",
                        "evidence_key_id": f"{family}-{replicate}",
                        "canonical_manifest_bytes_sha256": hashlib.sha256(b"manifest").hexdigest(),
                    },
                )
            )
    plan = SimpleNamespace(plan=SimpleNamespace(evidence_lineage_rows=tuple(rows)))
    monkeypatch.setattr(driver, "TrainingDataEvidenceKey", FakeKey)
    monkeypatch.setattr(driver, "TrainingDataEvidenceManifest", FakeManifest)
    monkeypatch.setattr(driver, "open_training_data_reader", lambda _fd: pinned_run())
    monkeypatch.setattr(
        driver,
        "load_training_data_evidence_payload_bundle_from_at",
        lambda _reader, _id, expected_key: FakeBundle(FakeManifest(expected_key)),
    )
    monkeypatch.setattr(
        driver,
        "PinnedOutcomeTrainingEvidence",
        lambda payload, payload_bytes: (payload, payload_bytes),
    )
    # A payload byte substitution must fail against the frozen manifest hash.
    monkeypatch.setattr(
        driver,
        "load_training_data_evidence_payload_bundle_from_at",
        lambda _reader, _id, expected_key: SimpleNamespace(
            manifest=FakeManifest(expected_key),
            manifest_bytes=b"manifest",
            payload_bytes=b"substituted",
            payload=FakePayload(),
        ),
    )
    with pytest.raises(RuntimeError, match="differs from frozen lineage"):
        driver._read_evidence(runtime, plan)


def test_happy_path_forwards_exact_plan_evidence_and_provenance(monkeypatch, tmp_path):
    manifest, repository, raw, output = _paths(tmp_path)
    runtime = SimpleNamespace(raw_root=raw, folds=())
    plan_body = SimpleNamespace(
        plan_id="a" * 64,
        protocol_sha256="d" * 64,
        model_owners=tuple(range(240)),
        units=tuple(range(5760)),
    )
    plan = SimpleNamespace(plan=plan_body)
    snapshot = SimpleNamespace(repository=repository)
    provenance = SimpleNamespace(
        git_dirty=False,
        git_commit_sha="b" * 40,
        requested_device="cpu",
        resolved_device="cpu",
        requested_torch_threads=1,
        actual_torch_threads=1,
        requested_torch_interop_threads=1,
        actual_torch_interop_threads=1,
        processes=1,
    )
    batch_calls = {}
    progress = SimpleNamespace(progress_sha256="e" * 64)
    batch_result = SimpleNamespace(
        complete=False,
        completed_owner_ids=("c" * 64,),
        progress=progress,
    )
    evidence = {("plain", 0): object()}
    monkeypatch.setattr(driver, "load_screening_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(driver, "recheck_screening_runtime_readonly", lambda _: None)
    monkeypatch.setattr(driver, "load_outcome_group_diagnostic_protocol", lambda: snapshot)
    monkeypatch.setattr(driver, "build_outcome_group_diagnostic_plan", lambda _: object())
    monkeypatch.setattr(driver, "bind_validated_outcome_diagnostic_plan", lambda *_a, **_k: plan)
    monkeypatch.setattr(driver, "_read_evidence", lambda _runtime, _plan: evidence)
    captures = []
    monkeypatch.setattr(
        driver,
        "capture_system_provenance",
        lambda *_a, **_k: captures.append(provenance) or provenance,
    )
    monkeypatch.setattr(driver, "provenance_identity_sha256", lambda value: "f" * 64)

    def fake_batch(*args, **kwargs):
        batch_calls["args"] = args
        batch_calls["kwargs"] = kwargs
        return batch_result

    monkeypatch.setattr(driver, "prepare_outcome_diagnostic_model_batch", fake_batch)
    result = _call(manifest, repository, raw, output, limit=1)
    assert result["plan_id"] == "a" * 64
    assert result["completed_owner_count"] == 1
    assert batch_calls["args"][:3] == (plan, snapshot, output.resolve())
    assert batch_calls["args"][3] is evidence
    assert batch_calls["kwargs"]["preparation_git_commit_sha"] == "b" * 40
    assert batch_calls["kwargs"]["preparation_provenance_sha256"] == "f" * 64
    assert batch_calls["kwargs"]["limit"] == 1
    assert len(captures) == 2


def test_second_provenance_capture_drift_fails_closed(monkeypatch, tmp_path):
    manifest, repository, raw, output = _paths(tmp_path)
    runtime = SimpleNamespace(raw_root=raw, folds=())
    plan = SimpleNamespace(
        plan=SimpleNamespace(
            plan_id="a" * 64,
            protocol_sha256="d" * 64,
            model_owners=tuple(range(240)),
            units=tuple(range(5760)),
        )
    )
    snapshot = SimpleNamespace(repository=repository)
    first = SimpleNamespace(
        git_dirty=False,
        git_commit_sha="b" * 40,
        requested_device="cpu",
        resolved_device="cpu",
        requested_torch_threads=1,
        actual_torch_threads=1,
        requested_torch_interop_threads=1,
        actual_torch_interop_threads=1,
        processes=1,
    )
    second = SimpleNamespace(**{**first.__dict__, "git_commit_sha": "c" * 40})
    captures = iter((first, second))
    monkeypatch.setattr(driver, "load_screening_runtime", lambda *a, **k: runtime)
    monkeypatch.setattr(driver, "recheck_screening_runtime_readonly", lambda _: None)
    monkeypatch.setattr(driver, "load_outcome_group_diagnostic_protocol", lambda: snapshot)
    monkeypatch.setattr(driver, "build_outcome_group_diagnostic_plan", lambda _: object())
    monkeypatch.setattr(driver, "bind_validated_outcome_diagnostic_plan", lambda *_a, **_k: plan)
    monkeypatch.setattr(driver, "_read_evidence", lambda *_a: {("plain", 0): object()})
    monkeypatch.setattr(driver, "capture_system_provenance", lambda *_a, **_k: next(captures))
    monkeypatch.setattr(driver, "provenance_identity_sha256", lambda _: "f" * 64)
    monkeypatch.setattr(
        driver,
        "prepare_outcome_diagnostic_model_batch",
        lambda *a, **k: SimpleNamespace(
            complete=False,
            completed_owner_ids=(),
            progress=SimpleNamespace(progress_sha256="e" * 64),
        ),
    )
    with pytest.raises(RuntimeError, match="dirty or differs"):
        _call(manifest, repository, raw, output, limit=1)
