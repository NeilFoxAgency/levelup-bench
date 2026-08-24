"""Synthetic, non-result tests for the Phase 3 analysis publication boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from levelup.experiments.milestone6_phase3_selection_analysis import (
    Phase3SelectionAnalysisError,
    build_phase3_selection_analysis,
    publish_phase3_selection_analysis,
)
from levelup.experiments.runner.config import canonical_json_bytes


def test_missing_marker_fails_before_any_analysis_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "result"
    root.mkdir()
    called = False

    def fake_readiness(*args, **kwargs):
        nonlocal called
        called = True
        raise RuntimeError("synthetic readiness stop")

    monkeypatch.setattr(
        "levelup.experiments.milestone6_phase3_selection_analysis.capture_phase3_readiness",
        fake_readiness,
    )
    with pytest.raises(Phase3SelectionAnalysisError):
        build_phase3_selection_analysis(
            repository=tmp_path, result_root=root, expected_git_commit="a" * 40
        )
    assert called
    assert not (tmp_path / "analysis.json").exists()


def test_publish_is_self_hashed_exclusive_and_outside_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "result"
    root.mkdir()
    output = tmp_path / "analysis.json"
    monkeypatch.setattr(
        "levelup.experiments.milestone6_phase3_selection_analysis.build_phase3_selection_analysis",
        lambda **_: {"schema_version": "synthetic", "unit_count": 0, "candidate_summaries": []},
    )
    published = publish_phase3_selection_analysis(
        repository=tmp_path, result_root=root, expected_git_commit="a" * 40, output=output
    )
    body = json.loads(published.read_bytes())
    supplied = body.pop("selection_analysis_sha256")
    assert supplied == hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    with pytest.raises(Phase3SelectionAnalysisError, match="already exists"):
        publish_phase3_selection_analysis(
            repository=tmp_path, result_root=root, expected_git_commit="a" * 40, output=output
        )


def test_inside_result_root_and_symlink_output_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "result"
    root.mkdir()
    monkeypatch.setattr(
        "levelup.experiments.milestone6_phase3_selection_analysis.build_phase3_selection_analysis",
        lambda **_: {"schema_version": "synthetic"},
    )
    with pytest.raises(Phase3SelectionAnalysisError, match="outside result_root"):
        publish_phase3_selection_analysis(
            repository=tmp_path,
            result_root=root,
            expected_git_commit="a" * 40,
            output=root / "analysis.json",
        )
    target = tmp_path / "target.json"
    target.write_text("x")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(Phase3SelectionAnalysisError):
        publish_phase3_selection_analysis(
            repository=tmp_path, result_root=root, expected_git_commit="a" * 40, output=link
        )


def test_normalized_output_cannot_escape_back_into_result_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "result"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(
        "levelup.experiments.milestone6_phase3_selection_analysis.build_phase3_selection_analysis",
        lambda **_: {"schema_version": "synthetic"},
    )
    disguised = outside / ".." / root.name / "analysis.json"
    with pytest.raises(Phase3SelectionAnalysisError, match="outside result_root"):
        publish_phase3_selection_analysis(
            repository=tmp_path,
            result_root=root,
            expected_git_commit="a" * 40,
            output=disguised,
        )
    assert not (root / "analysis.json").exists()
