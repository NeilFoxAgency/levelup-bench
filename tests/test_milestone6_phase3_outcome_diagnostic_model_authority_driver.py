from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase3_outcome_diagnostic_model_authority_driver as driver

COMMIT = "a" * 40
PREP_PROVENANCE = "b" * 64
PLAN_ID = "c" * 64
STORE_ID = "phase3-outcome-diagnostic-models-cccccccccccc"


def _fake_inputs(tmp_path: Path):
    authority = tmp_path / "authority"
    authority.mkdir()
    (authority / "configs" / "milestone6").mkdir(parents=True)
    (authority / "runs" / "milestone6" / STORE_ID).mkdir(parents=True)
    raw_root = tmp_path / "raw"
    fold_root = tmp_path / "fold"
    raw_root.mkdir()
    fold_root.mkdir()
    views = tuple(
        SimpleNamespace(
            view_id=f"{index + 1:064x}",
            heldout_family=f"family-{index // 10}",
            replicate=index % 5,
        )
        for index in range(60)
    )
    plan = SimpleNamespace(
        plan=SimpleNamespace(
            final_family_access=False,
            views=views,
            units=tuple(range(5_760)),
            model_owners=tuple(range(240)),
            plan_id=PLAN_ID,
            protocol_sha256="d" * 64,
        )
    )
    runtime = SimpleNamespace(
        raw_root=raw_root,
        folds=(SimpleNamespace(store=SimpleNamespace(run_dir=fold_root)),),
    )
    source = {
        (f"family-{family}", replicate): object()
        for family in range(6)
        for replicate in range(5)
    }
    return authority, runtime, plan, source


def _patch_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    authority, runtime, plan, source = _fake_inputs(tmp_path)
    snapshot = object()
    monkeypatch.setattr(
        driver,
        "_load_inputs",
        lambda *args: (tmp_path / "screening", authority, runtime, snapshot, plan),
    )
    monkeypatch.setattr(
        driver,
        "load_outcome_group_diagnostic_protocol",
        lambda path, *, repository: snapshot,
    )
    monkeypatch.setattr(driver, "_read_evidence", lambda *args: source)
    monkeypatch.setattr(driver, "_reject_output", lambda path, forbidden: Path(path).absolute())
    monkeypatch.setattr(driver, "_validate_generation_repository", lambda *args: None)
    monkeypatch.setattr(driver, "recheck_screening_runtime_readonly", lambda runtime: None)
    monkeypatch.setattr(driver, "canonical_outcome_model_artifact_authority_bytes", lambda value: b'{"authority":1}')
    return authority, runtime, plan


def _authority(store_id: str = STORE_ID):
    return SimpleNamespace(
        authority_sha256="e" * 64,
        artifact_store_id=store_id,
        generation_git_commit_sha=COMMIT,
        views=tuple(range(60)),
        evidence=tuple(range(30)),
        artifacts=tuple(range(240)),
    )


def test_load_inputs_anchors_protocol_to_requested_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authority, runtime, plan, _source = _fake_inputs(tmp_path)
    screening = tmp_path / "screening"
    screening.mkdir()
    manifest = screening / driver.CANONICAL_READINESS_PATH
    manifest.parent.mkdir(parents=True)
    observed: list[tuple[str, Path]] = []
    snapshot = SimpleNamespace(repository=authority)

    monkeypatch.setattr(
        driver,
        "_reject_repository",
        lambda path, label: screening if label == "screening" else authority,
    )
    monkeypatch.setattr(driver, "load_screening_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(driver, "recheck_screening_runtime_readonly", lambda value: None)

    def load_protocol(path, *, repository):
        observed.append((path, repository))
        return snapshot

    monkeypatch.setattr(driver, "load_outcome_group_diagnostic_protocol", load_protocol)
    monkeypatch.setattr(driver, "build_outcome_group_diagnostic_plan", lambda value: plan)
    monkeypatch.setattr(
        driver,
        "bind_validated_outcome_diagnostic_plan",
        lambda value, *, snapshot: value,
    )
    loaded = driver._load_inputs(
        manifest,
        "f" * 64,
        tmp_path / "raw",
        screening,
        authority,
    )
    assert loaded == (screening, authority, runtime, snapshot, plan)
    assert observed == [(driver.PROTOCOL_PATH, authority)]


def test_clean_generation_builds_twice_and_forwards_exact_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authority_repo, _runtime, _plan = _patch_run(monkeypatch, tmp_path)
    calls: list[str] = []

    def build(*args, **kwargs):
        calls.append(kwargs["generation_git_commit_sha"])
        return _authority()

    monkeypatch.setattr(driver, "build_outcome_model_artifact_authority_from_store", build)
    result = driver.run_outcome_model_authority_generation(
        "manifest.json",
        "f" * 64,
        "raw",
        "screening",
        authority_repo,
        authority_repo / "runs" / "milestone6" / STORE_ID,
        expected_preparation_commit_sha=COMMIT,
        expected_preparation_provenance_sha256=PREP_PROVENANCE,
        expected_generation_commit_sha=COMMIT,
    )
    assert calls == [COMMIT, COMMIT]
    assert result["generation_git_commit_sha"] == COMMIT
    assert result["view_count"] == 60
    assert result["evidence_count"] == 30
    assert result["artifact_count"] == 240


def test_cli_emits_canonical_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {
        "authority_sha256": "e" * 64,
        "schema_version": "milestone6.phase3.outcome-diagnostic-model-authority-result.v1",
    }
    monkeypatch.setattr(
        driver,
        "run_outcome_model_authority_generation",
        lambda *args, **kwargs: expected,
    )
    assert (
        driver.main(
            [
                "--manifest-path",
                "manifest.json",
                "--manifest-sha256",
                "f" * 64,
                "--raw-root",
                "raw",
                "--screening-repository",
                "screening",
                "--authority-repository",
                "authority",
                "--store-root",
                "store",
                "--expected-preparation-commit",
                COMMIT,
                "--expected-preparation-provenance",
                PREP_PROVENANCE,
                "--expected-generation-commit",
                COMMIT,
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ) + "\n"


def test_generation_repository_rejects_wrong_commit_and_dirty_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    monkeypatch.setattr(driver, "_repo_state", lambda path: ("d" * 40, False))
    with pytest.raises(driver.OutcomeDiagnosticModelAuthorityError, match="generation commit"):
        driver._validate_generation_repository(repository, COMMIT)
    monkeypatch.setattr(driver, "_repo_state", lambda path: (COMMIT, True))
    with pytest.raises(driver.OutcomeDiagnosticModelAuthorityError, match="dirty"):
        driver._validate_generation_repository(repository, COMMIT)


def test_noncanonical_store_path_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authority_repo, _runtime, _plan = _patch_run(monkeypatch, tmp_path)
    with pytest.raises(driver.OutcomeDiagnosticModelAuthorityError, match="exact canonical"):
        driver.run_outcome_model_authority_generation(
            "manifest.json",
            "f" * 64,
            "raw",
            "screening",
            authority_repo,
            tmp_path / "other-store",
            expected_preparation_commit_sha=COMMIT,
            expected_preparation_provenance_sha256=PREP_PROVENANCE,
            expected_generation_commit_sha=COMMIT,
        )


def test_missing_canonical_store_is_rejected_before_builder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authority_repo, _runtime, _plan = _patch_run(monkeypatch, tmp_path)
    store = authority_repo / "runs" / "milestone6" / STORE_ID
    store.rmdir()
    called = False

    def build(*args, **kwargs):
        nonlocal called
        called = True
        return _authority()

    monkeypatch.setattr(driver, "build_outcome_model_artifact_authority_from_store", build)
    with pytest.raises(driver.OutcomeDiagnosticModelAuthorityError, match="must already exist"):
        driver.run_outcome_model_authority_generation(
            "manifest.json",
            "f" * 64,
            "raw",
            "screening",
            authority_repo,
            store,
            expected_preparation_commit_sha=COMMIT,
            expected_preparation_provenance_sha256=PREP_PROVENANCE,
            expected_generation_commit_sha=COMMIT,
        )
    assert not called


def test_output_path_must_be_new_canonical_path_and_rejects_symlinks(tmp_path: Path) -> None:
    authority = tmp_path / "authority"
    (authority / "configs" / "milestone6").mkdir(parents=True)
    canonical = authority / driver.AUTHORITY_OUTPUT_PATH
    canonical.write_bytes(b"old")
    with pytest.raises(driver.OutcomeDiagnosticModelAuthorityError, match="already exists"):
        driver._canonical_output_path(authority, canonical)

    canonical.unlink()
    link = tmp_path / "link"
    link.symlink_to(authority, target_is_directory=True)
    with pytest.raises(driver.OutcomeDiagnosticModelAuthorityError, match="symlink"):
        driver._canonical_output_path(authority, link / driver.AUTHORITY_OUTPUT_PATH)


def test_output_is_exclusive_and_fsync_published(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authority_repo, _runtime, _plan = _patch_run(monkeypatch, tmp_path)
    monkeypatch.setattr(driver, "build_outcome_model_artifact_authority_from_store", lambda *a, **k: _authority())
    output = authority_repo / driver.AUTHORITY_OUTPUT_PATH
    result = driver.run_outcome_model_authority_generation(
        "manifest.json",
        "f" * 64,
        "raw",
        "screening",
        authority_repo,
        authority_repo / "runs" / "milestone6" / STORE_ID,
        expected_preparation_commit_sha=COMMIT,
        expected_preparation_provenance_sha256=PREP_PROVENANCE,
        expected_generation_commit_sha=COMMIT,
        output_path=output,
    )
    assert output.read_bytes() == b'{"authority":1}'
    assert result["output_path"] == str(output)
    with pytest.raises(driver.OutcomeDiagnosticModelAuthorityError, match="already exists"):
        driver._canonical_output_path(authority_repo, output)


def test_runtime_recheck_drift_blocks_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authority_repo, runtime, _plan = _patch_run(monkeypatch, tmp_path)
    calls = iter([None, RuntimeError("runtime drift")])

    def recheck(_runtime):
        value = next(calls)
        if isinstance(value, Exception):
            raise value

    monkeypatch.setattr(driver, "recheck_screening_runtime_readonly", recheck)
    monkeypatch.setattr(driver, "build_outcome_model_artifact_authority_from_store", lambda *a, **k: _authority())
    with pytest.raises(RuntimeError, match="runtime drift"):
        driver.run_outcome_model_authority_generation(
            "manifest.json",
            "f" * 64,
            "raw",
            "screening",
            authority_repo,
            authority_repo / "runs" / "milestone6" / STORE_ID,
            expected_preparation_commit_sha=COMMIT,
            expected_preparation_provenance_sha256=PREP_PROVENANCE,
            expected_generation_commit_sha=COMMIT,
        )


def test_protocol_drift_blocks_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authority_repo, _runtime, _plan = _patch_run(monkeypatch, tmp_path)
    monkeypatch.setattr(driver, "build_outcome_model_artifact_authority_from_store", lambda *a, **k: _authority())
    monkeypatch.setattr(
        driver,
        "load_outcome_group_diagnostic_protocol",
        lambda path, *, repository: object(),
    )
    with pytest.raises(driver.OutcomeDiagnosticModelAuthorityError, match="protocol changed"):
        driver.run_outcome_model_authority_generation(
            "manifest.json",
            "f" * 64,
            "raw",
            "screening",
            authority_repo,
            authority_repo / "runs" / "milestone6" / STORE_ID,
            expected_preparation_commit_sha=COMMIT,
            expected_preparation_provenance_sha256=PREP_PROVENANCE,
            expected_generation_commit_sha=COMMIT,
        )
