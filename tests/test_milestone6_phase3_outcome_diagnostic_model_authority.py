from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase3_outcome_diagnostic_model_authority as authority
import levelup.experiments.milestone6_phase3_outcome_diagnostic_model_batch as model_batch
import levelup.experiments.milestone6_phase3_outcome_diagnostic_model_store as model_store
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
    PinnedOutcomeTrainingEvidence,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    ValidatedOutcomePlan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    OutcomeDiagnosticProtocolSnapshot,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import SystemProvenance
from levelup.experiments.runner.storage import provenance_identity_sha256
from levelup.experiments.runner.training_data_artifacts import TrainingDataPayload


@dataclass
class _FakeStore:
    root_fd: int = 0

    def recheck(self) -> None:
        return None


def _snapshot() -> OutcomeDiagnosticProtocolSnapshot:
    return OutcomeDiagnosticProtocolSnapshot(
        repository=Path("."),
        path=Path("protocol.json"),
        content=b"protocol",
        sha256="0" * 64,
        payload={},
        authority_bytes=(),
    )


def _plan(owner_count: int = 240) -> ValidatedOutcomePlan:
    raw = object.__new__(ValidatedOutcomePlan)
    object.__setattr__(
        raw,
        "plan",
        SimpleNamespace(
            plan_id="1" * 64,
            protocol_sha256="2" * 64,
            model_owners=tuple(
                SimpleNamespace(owner_id=f"{index + 1:064x}", view_id="3" * 64)
                for index in range(owner_count)
            ),
            views=(SimpleNamespace(view_id="3" * 64),),
        ),
    )
    object.__setattr__(raw, "_units_by_id", {})
    object.__setattr__(raw, "_construction_token", object())
    return raw


def _evidence() -> PinnedOutcomeTrainingEvidence:
    return PinnedOutcomeTrainingEvidence(
        TrainingDataPayload.model_construct(samples=()), b"evidence"
    )


def _provenance(commit: str) -> SystemProvenance:
    return SystemProvenance.model_validate(
        {
            "git_commit_sha": commit,
            "git_dirty": False,
            "git_diff_sha256": None,
            "python_version": "3.11",
            "packages": {},
            "installed_packages_sha256": "4" * 64,
            "os": "test",
            "architecture": "test",
            "cpu": "test",
            "cpu_count": 1,
            "memory_bytes": 1,
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "requested_torch_threads": 1,
            "actual_torch_threads": 1,
            "requested_torch_interop_threads": 1,
            "actual_torch_interop_threads": 1,
            "deterministic_algorithms_requested": False,
            "deterministic_algorithms_actual": False,
            "processes": 1,
            "captured_at_utc": datetime.now(timezone.utc),
        }
    )


def _patch_complete(monkeypatch: pytest.MonkeyPatch, snapshots: list[object]) -> None:
    fake_store = _FakeStore()

    @contextmanager
    def open_store(_root):
        yield fake_store

    monkeypatch.setattr(authority, "open_existing_outcome_model_store", open_store)
    monkeypatch.setattr(authority, "_reader", lambda value: value)
    monkeypatch.setattr(
        authority,
        "validate_outcome_model_preparation_metadata_at",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(authority, "scan_outcome_model_inventory_at", lambda *a, **k: None)
    monkeypatch.setattr(
        authority,
        "load_outcome_model_artifact_at",
        lambda *a, **k: (
            SimpleNamespace(key=SimpleNamespace(view_id="3" * 64)),
            object(),
            object(),
        ),
    )
    monkeypatch.setattr(
        authority,
        "build_outcome_model_artifact_authority",
        lambda *a, **k: "authority",
    )
    monkeypatch.setattr(
        model_store,
        "snapshot_outcome_model_store_identities_at",
        lambda *a, **k: snapshots.pop(0),
        raising=False,
    )


def test_authority_build_is_deterministic_and_reads_complete_owner_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = ["same", "same", "same", "same"]
    _patch_complete(monkeypatch, snapshots)
    plan = _plan()
    evidence = {"3" * 64: _evidence()}
    first = authority.build_outcome_model_artifact_authority_from_store(
        "models", plan, _snapshot(), evidence,
        preparation_git_commit_sha="a" * 40,
        preparation_provenance_sha256="b" * 64,
        generation_git_commit_sha="c" * 40,
    )
    second = authority.build_outcome_model_artifact_authority_from_store(
        "models", plan, _snapshot(), evidence,
        preparation_git_commit_sha="a" * 40,
        preparation_provenance_sha256="b" * 64,
        generation_git_commit_sha="c" * 40,
    )
    assert first == second == "authority"


def test_authority_rejects_partial_owner_plan_before_opening_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = False

    @contextmanager
    def open_store(_root):
        nonlocal opened
        opened = True
        yield _FakeStore()

    monkeypatch.setattr(authority, "open_existing_outcome_model_store", open_store)
    with pytest.raises(authority.OutcomeDiagnosticModelAuthorityError, match="240 owners"):
        authority.build_outcome_model_artifact_authority_from_store(
            "models", _plan(239), _snapshot(), {"3" * 64: _evidence()},
            preparation_git_commit_sha="a" * 40,
            preparation_provenance_sha256="b" * 64,
            generation_git_commit_sha="c" * 40,
        )
    assert not opened


def test_authority_rejects_persisted_provenance_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = ["same", "same"]
    _patch_complete(monkeypatch, snapshots)
    monkeypatch.setattr(
        authority,
        "validate_outcome_model_preparation_metadata_at",
        lambda *a, **k: (_ for _ in ()).throw(
            authority.OutcomeDiagnosticModelAuthorityError("provenance drift")
        ),
    )
    with pytest.raises(authority.OutcomeDiagnosticModelAuthorityError, match="provenance drift"):
        authority.build_outcome_model_artifact_authority_from_store(
            "models", _plan(), _snapshot(), {"3" * 64: _evidence()},
            preparation_git_commit_sha="a" * 40,
            preparation_provenance_sha256="b" * 64,
            generation_git_commit_sha="c" * 40,
        )


def test_authority_rejects_store_identity_drift_after_semantic_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = ["before", "after"]
    _patch_complete(monkeypatch, snapshots)
    with pytest.raises(authority.OutcomeDiagnosticModelAuthorityError, match="identities changed"):
        authority.build_outcome_model_artifact_authority_from_store(
            "models", _plan(), _snapshot(), {"3" * 64: _evidence()},
            preparation_git_commit_sha="a" * 40,
            preparation_provenance_sha256="b" * 64,
            generation_git_commit_sha="c" * 40,
        )


def test_persisted_progress_and_provenance_are_read_through_the_pinned_store(
    tmp_path: Path,
) -> None:
    plan = _plan()
    owner_ids = tuple(sorted(owner.owner_id for owner in plan.plan.model_owners))
    commit = "a" * 40
    provenance = _provenance(commit)
    provenance_sha = provenance_identity_sha256(provenance)
    progress = model_batch._make_progress(
        plan_id=plan.plan.plan_id,
        protocol_sha256=plan.plan.protocol_sha256,
        preparation_git_commit_sha=commit,
        preparation_provenance_sha256=provenance_sha,
        completed_owner_ids=owner_ids,
    )
    with model_store.open_outcome_model_store(tmp_path / "models") as pinned:
        model_store._write_new(
            pinned.reader.root_fd,
            model_batch.PROGRESS_NAME,
            canonical_json_bytes(progress.model_dump(mode="json")) + b"\n",
            pinned.reader.staging_fd,
        )
        model_store._write_new(
            pinned.reader.root_fd,
            model_batch.PREPARATION_PROVENANCE_NAME,
            canonical_json_bytes(provenance.model_dump(mode="json")) + b"\n",
            pinned.reader.staging_fd,
        )
        authority.validate_outcome_model_preparation_metadata_at(
            pinned.reader,
            plan,
            preparation_git_commit_sha=commit,
            preparation_provenance_sha256=provenance_sha,
            expected_owner_ids=owner_ids,
        )
        with pytest.raises(
            authority.OutcomeDiagnosticModelAuthorityError,
            match="another run",
        ):
            authority.validate_outcome_model_preparation_metadata_at(
                pinned.reader,
                plan,
                preparation_git_commit_sha="b" * 40,
                preparation_provenance_sha256=provenance_sha,
                expected_owner_ids=owner_ids,
            )
