from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase3_model_preparation_driver as driver


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    manifest = repository / driver.CANONICAL_READINESS_PATH
    return manifest, repository, raw_root, output_root


def _stub_authorities(monkeypatch, calls: list[tuple[object, ...]], result=None):
    runtime = object()
    plan = object()
    validated = object()
    anchor = object()
    evidence = object()
    monkeypatch.setattr(
        driver,
        "load_screening_runtime",
        lambda *args, **kwargs: calls.append(("runtime", args, kwargs)) or runtime,
    )
    monkeypatch.setattr(
        driver,
        "load_committed_phase3_plan_lock_bytes",
        lambda *args, **kwargs: calls.append(("plan-load", args, kwargs)) or b"plan",
    )
    monkeypatch.setattr(
        driver,
        "validate_phase3_plan_lock_bytes",
        lambda *args, **kwargs: calls.append(("plan-validate", args, kwargs)) or plan,
    )
    monkeypatch.setattr(
        driver,
        "bind_validated_phase3_plan",
        lambda *args, **kwargs: calls.append(("plan-bind", args, kwargs)) or validated,
    )
    monkeypatch.setattr(
        driver,
        "load_committed_phase3_anchor_manifest_bytes",
        lambda *args, **kwargs: calls.append(("anchor-load", args, kwargs)) or b"anchor",
    )
    monkeypatch.setattr(
        driver,
        "validate_phase3_anchor_manifest_bytes",
        lambda *args, **kwargs: calls.append(("anchor-validate", args, kwargs)) or anchor,
    )
    monkeypatch.setattr(
        driver,
        "load_committed_phase3_evidence_lock_bytes",
        lambda *args, **kwargs: calls.append(("evidence-load", args, kwargs)) or b"evidence",
    )
    monkeypatch.setattr(
        driver,
        "validate_phase3_evidence_lock_bytes",
        lambda *args, **kwargs: calls.append(("evidence-validate", args, kwargs)) or evidence,
    )
    monkeypatch.setattr(
        driver,
        "prepare_phase3_model_batch",
        lambda *args, **kwargs: calls.append(("prepare", args, kwargs))
        or (result or SimpleNamespace(model_dump=lambda **_: {"model_count": 0})),
    )
    return runtime, plan, validated, anchor, evidence


def test_requires_canonical_manifest_path_and_lowercase_digest(monkeypatch, tmp_path):
    manifest, repository, raw_root, output_root = _paths(tmp_path)
    loaded = False

    def load(*_args, **_kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("runtime must not load")

    monkeypatch.setattr(driver, "load_screening_runtime", load)
    with pytest.raises(RuntimeError, match="canonical committed readiness"):
        driver.run_phase3_model_preparation(
            tmp_path / "copied.json",
            "a" * 64,
            raw_root,
            repository,
            output_root,
            authority_repository=repository,
        )
    with pytest.raises(RuntimeError, match="lowercase SHA-256"):
        driver.run_phase3_model_preparation(
            manifest,
            "A" * 64,
            raw_root,
            repository,
            output_root,
            authority_repository=repository,
        )
    assert loaded is False


@pytest.mark.parametrize("limit", [-1, driver.EXPECTED_MODELS + 1])
def test_rejects_limit_and_owner_selection_conflicts_or_range(monkeypatch, tmp_path, limit):
    manifest, repository, raw_root, output_root = _paths(tmp_path)
    monkeypatch.setattr(driver, "load_screening_runtime", lambda *_a, **_k: object())
    with pytest.raises(RuntimeError, match="frozen 480-owner"):
        driver.run_phase3_model_preparation(
            manifest,
            "a" * 64,
            raw_root,
            repository,
            output_root,
            authority_repository=repository,
            limit=limit,
        )
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        driver.run_phase3_model_preparation(
            manifest,
            "a" * 64,
            raw_root,
            repository,
            output_root,
            authority_repository=repository,
            owner_ids=("b" * 64,),
            limit=0,
        )


def test_rejects_raw_overlap_and_symlinked_output_ancestry(tmp_path):
    manifest, repository, raw_root, output_root = _paths(tmp_path)
    raw_root.mkdir()
    with pytest.raises(RuntimeError, match="not overlap the raw evidence"):
        driver.run_phase3_model_preparation(
            manifest,
            "a" * 64,
            raw_root,
            repository,
            raw_root / "models",
            authority_repository=repository,
        )
    with pytest.raises(RuntimeError, match="not overlap the raw evidence"):
        driver.run_phase3_model_preparation(
            manifest,
            "a" * 64,
            raw_root,
            repository,
            raw_root.parent,
            authority_repository=repository,
        )
    link_parent = tmp_path / "link-parent"
    link_parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        driver.run_phase3_model_preparation(
            manifest,
            "a" * 64,
            raw_root,
            repository,
            link_parent / "models",
            authority_repository=repository,
        )


def test_rejects_symlinked_screening_and_authority_repositories(tmp_path):
    manifest, repository, raw_root, output_root = _paths(tmp_path)
    raw_root.mkdir()
    repository_link = tmp_path / "repository-link"
    repository_link.symlink_to(repository, target_is_directory=True)

    with pytest.raises(RuntimeError, match="screening repository.*symlink"):
        driver.run_phase3_model_preparation(
            manifest,
            "a" * 64,
            raw_root,
            repository_link,
            output_root,
            authority_repository=repository,
        )
    with pytest.raises(RuntimeError, match="authority repository.*symlink"):
        driver.run_phase3_model_preparation(
            manifest,
            "a" * 64,
            raw_root,
            repository,
            output_root,
            authority_repository=repository_link,
        )


def test_loader_sequence_and_arguments_are_frozen(monkeypatch, tmp_path):
    manifest, repository, raw_root, output_root = _paths(tmp_path)
    raw_root.mkdir()
    calls: list[tuple[object, ...]] = []
    runtime, plan, validated, anchor, evidence = _stub_authorities(monkeypatch, calls)
    result = driver.run_phase3_model_preparation(
        manifest,
        "a" * 64,
        raw_root,
        repository,
        output_root,
        authority_repository=repository,
        owner_ids=("b" * 64,),
    )
    assert [entry[0] for entry in calls] == [
        "runtime",
        "plan-load",
        "plan-validate",
        "plan-bind",
        "anchor-load",
        "anchor-validate",
        "evidence-load",
        "evidence-validate",
        "prepare",
    ]
    assert calls[0][1][:4] == (manifest, raw_root, repository,)
    assert calls[0][2] == {
        "manifest_bytes_sha256": "a" * 64,
        "authority_repository": repository.resolve(),
    }
    assert calls[2][1] == (b"plan",)
    assert calls[3][1] == (plan,)
    assert calls[5][1] == (b"anchor",)
    assert calls[5][2] == {"runtime": runtime}
    assert calls[7][1] == (b"evidence",)
    assert calls[7][2] == {
        "runtime": runtime,
        "validated_plan": validated,
        "anchor_manifest": anchor,
        "anchor_file_bytes": b"anchor",
        "plan_lock_bytes": b"plan",
    }
    assert calls[8][1] == (output_root.resolve(),)
    assert calls[8][2] == {
        "runtime": runtime,
        "validated_plan": validated,
        "anchor_manifest": anchor,
        "evidence_lock": evidence,
        "owner_ids": ("b" * 64,),
        "limit": None,
        "authority_repository": repository.resolve(),
        "authority_repository_identity": calls[8][2]["authority_repository_identity"],
        "plan_lock_bytes": b"plan",
        "anchor_file_bytes": b"anchor",
        "evidence_lock_bytes": b"evidence",
    }
    assert result["output_root"] == str(output_root.resolve())


def test_cli_prints_deterministic_json(monkeypatch, tmp_path, capsys):
    manifest, repository, raw_root, output_root = _paths(tmp_path)
    payload = {"output_root": str(output_root.resolve()), "model_count": 0}
    monkeypatch.setattr(driver, "run_phase3_model_preparation", lambda *a, **k: payload)
    argv = [
        "--manifest-path",
        str(manifest),
        "--manifest-sha256",
        "c" * 64,
        "--raw-root",
        str(raw_root),
        "--screening-repository",
        str(repository),
        "--authority-repository",
        str(repository),
        "--output-root",
        str(output_root),
        "--limit",
        "0",
    ]
    assert driver.main(argv) == 0
    first = capsys.readouterr().out
    assert driver.main(argv) == 0
    second = capsys.readouterr().out
    assert first == second == '{"model_count":0,"output_root":"' + str(output_root.resolve()) + '"}\n'
