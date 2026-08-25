from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase3_outcome_diagnostic_model_batch as batch
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_batch import (
    OutcomeDiagnosticModelBatchError,
    OutcomeModelPreparationProgress,
    _make_progress,
    _owner_ids,
    _validate_identity,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    _TOKEN,
    OutcomeModelOwner,
    OutcomePlan,
    ValidatedOutcomePlan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    OutcomeDiagnosticProtocolSnapshot,
)
from levelup.experiments.runner.records import SystemProvenance
from levelup.experiments.runner.storage import provenance_identity_sha256


def _plan() -> ValidatedOutcomePlan:
    owners = tuple(
        OutcomeModelOwner(
            owner_id=f"{index:064x}",
            condition_id="S-RP-state-resource-pressure-outcome-listwise-optimum",
            fold_id="fold",
            heldout_family="plain",
            replicate=0,
            training_tuple_id="lr0p003-e120",
            view_id=f"{index:064x}",
            model_seed=index,
            learning_rate=0.003,
            training_epochs=120,
            search_temperature_ids=(),
            trainable_parameters=3841,
            feature_mask_sha256="0" * 64,
            transformation_sha256="0" * 64,
            model_identity_sha256="0" * 64,
        )
        for index in range(240)
    )
    plan = OutcomePlan(
        schema_version="test",
        plan_id="a" * 64,
        parent_commit_sha="b" * 40,
        protocol_sha256="c" * 64,
        authority_hashes=(),
        family_order=(),
        replicates=(),
        condition_ids=(),
        candidate_tuple_ids=(),
        evidence_lineage_rows=(),
        views=(),
        model_owners=owners,
        units=(),
    )
    return ValidatedOutcomePlan(plan, {}, _construction_token=_TOKEN)


def test_progress_self_hash_and_owner_order_are_canonical() -> None:
    progress = _make_progress(
        plan_id="a" * 64,
        protocol_sha256="b" * 64,
        preparation_git_commit_sha="c" * 40,
        preparation_provenance_sha256="d" * 64,
        completed_owner_ids=("0" * 64, "1" * 64),
    )
    assert progress.expected_progress_sha256 == progress.progress_sha256
    with pytest.raises(ValueError, match="self-hash"):
        OutcomeModelPreparationProgress.model_validate(
            progress.model_copy(update={"progress_sha256": "e" * 64}).model_dump()
        )
    with pytest.raises(ValueError):
        OutcomeModelPreparationProgress.model_validate(
            progress.model_copy(update={"completed_owner_ids": ("1" * 64, "0" * 64)}).model_dump()
        )


@pytest.mark.parametrize("commit", [True, False])
def test_zero_or_malformed_preparation_identity_rejected(commit: bool) -> None:
    with pytest.raises(OutcomeDiagnosticModelBatchError):
        _validate_identity("0" * (40 if commit else 64), commit=commit)
    with pytest.raises(OutcomeDiagnosticModelBatchError):
        _validate_identity("not-a-digest", commit=commit)


def test_owner_limit_is_bounded_and_deterministic() -> None:
    plan = _plan()
    assert _owner_ids(plan, None, 3) == tuple(f"{index:064x}" for index in range(3))
    with pytest.raises(OutcomeDiagnosticModelBatchError, match="outside"):
        _owner_ids(plan, None, 241)
    with pytest.raises(OutcomeDiagnosticModelBatchError, match="mutually"):
        _owner_ids(plan, ("0" * 64,), 1)


def test_foreign_and_duplicate_owner_ids_rejected() -> None:
    plan = _plan()
    with pytest.raises(OutcomeDiagnosticModelBatchError, match="foreign"):
        _owner_ids(plan, ("f" * 64,), None)
    with pytest.raises(OutcomeDiagnosticModelBatchError, match="duplicated"):
        _owner_ids(plan, ("0" * 64, "0" * 64), None)


def _provenance(*, device: str = "cpu", dirty: bool = False) -> SystemProvenance:
    return SystemProvenance.model_validate(
        {
            "git_commit_sha": "c" * 40,
            "git_dirty": dirty,
            "git_diff_sha256": "d" * 64 if dirty else None,
            "python_version": "3.11",
            "packages": {},
            "installed_packages_sha256": "e" * 64,
            "os": "test",
            "architecture": "test",
            "cpu": "test",
            "cpu_count": 1,
            "memory_bytes": 1,
            "requested_device": device,
            "resolved_device": device,
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


def _snapshot() -> OutcomeDiagnosticProtocolSnapshot:
    return OutcomeDiagnosticProtocolSnapshot(
        Path("."), Path("protocol.json"), b"p", hashlib.sha256(b"p").hexdigest(), {}, ()
    )


def test_batch_partial_resume_without_retraining(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, provenance = _plan(), _provenance()
    provenance_sha = provenance_identity_sha256(provenance)
    records: dict[str, object] = {}
    progress: bytes | None = None
    calls: list[str] = []
    scan_calls: list[int] = []

    class Store:
        reader = SimpleNamespace(root_fd=1, staging_fd=2)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def recheck(self):
            return None

    @contextmanager
    def open_store(_root):
        yield Store()

    monkeypatch.setattr(batch, "open_outcome_model_store", open_store)
    monkeypatch.setattr(batch, "_persist_provenance", lambda *_: None)
    monkeypatch.setattr(
        batch,
        "_validate_evidence",
        lambda plan_arg, *_: {owner.view_id: object() for owner in plan_arg.plan.model_owners},
    )
    monkeypatch.setattr(
        batch, "_load_existing", lambda *_a, **_k: (dict(records), dict(records), dict(records))
    )

    def read(_fd, _name):
        if progress is None:
            raise batch.OutcomeDiagnosticModelBatchError("missing") from FileNotFoundError()
        return progress

    monkeypatch.setattr(batch, "_read_at", read)

    def write(_s, _n, content):
        nonlocal progress
        progress = content

    monkeypatch.setattr(batch, "_write_at", write)

    def prepare(*_a, owner_id, **_k):
        calls.append(owner_id)
        return SimpleNamespace(
            record=f"record-{owner_id}", state_payload=object(), authorization=object()
        )

    monkeypatch.setattr(batch, "prepare_outcome_diagnostic_model", prepare)
    monkeypatch.setattr(
        batch,
        "write_outcome_model_artifact",
        lambda _r, record, _s, **_k: records.__setitem__(
            str(record).removeprefix("record-"), record
        ),
    )
    monkeypatch.setattr(
        batch,
        "scan_outcome_model_inventory_at",
        lambda *_a, **_k: scan_calls.append(len(records)),
    )
    kwargs = dict(
        preparation_git_commit_sha="c" * 40,
        preparation_provenance_sha256=provenance_sha,
        preparation_provenance=provenance,
        limit=1,
    )
    first = batch.prepare_outcome_diagnostic_model_batch(plan, _snapshot(), tmp_path, {}, **kwargs)
    second = batch.prepare_outcome_diagnostic_model_batch(plan, _snapshot(), tmp_path, {}, **kwargs)
    assert not first.complete and not second.complete and len(calls) == 1
    complete = batch.prepare_outcome_diagnostic_model_batch(
        plan,
        _snapshot(),
        tmp_path,
        {},
        preparation_git_commit_sha="c" * 40,
        preparation_provenance_sha256=provenance_sha,
        preparation_provenance=provenance,
        owner_ids=tuple(owner.owner_id for owner in plan.plan.model_owners),
    )
    assert complete.complete and len(complete.completed_owner_ids) == 240 and len(calls) == 240
    assert scan_calls == [240]
    assert progress is not None
    value = json.loads(progress)
    value["plan_id"] = "f" * 64
    progress = json.dumps(value).encode()
    with pytest.raises(batch.OutcomeDiagnosticModelBatchError):
        batch.prepare_outcome_diagnostic_model_batch(plan, _snapshot(), tmp_path, {}, **kwargs)


def test_batch_repairs_progress_lag_after_artifact_publication_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, provenance = _plan(), _provenance()
    provenance_sha = provenance_identity_sha256(provenance)
    records: dict[str, object] = {}
    progress: bytes | None = None
    prepare_calls: list[str] = []
    progress_writes = 0

    class Store:
        reader = SimpleNamespace(root_fd=1, staging_fd=2)

    @contextmanager
    def open_store(_root):
        yield Store()

    monkeypatch.setattr(batch, "open_outcome_model_store", open_store)
    monkeypatch.setattr(batch, "_persist_provenance", lambda *_: None)
    monkeypatch.setattr(
        batch,
        "_validate_evidence",
        lambda plan_arg, *_: {owner.view_id: object() for owner in plan_arg.plan.model_owners},
    )
    monkeypatch.setattr(
        batch, "_load_existing", lambda *_a, **_k: (dict(records), dict(records), dict(records))
    )

    def read(_fd, _name):
        if progress is None:
            raise batch.OutcomeDiagnosticModelBatchError("missing") from FileNotFoundError()
        return progress

    def write(_store, _name, content):
        nonlocal progress, progress_writes
        progress_writes += 1
        if progress_writes == 2:
            raise batch.OutcomeDiagnosticModelBatchError("simulated progress publication crash")
        progress = content

    def prepare(*_args, owner_id, **_kwargs):
        prepare_calls.append(owner_id)
        return SimpleNamespace(
            record=f"record-{owner_id}", state_payload=object(), authorization=object()
        )

    monkeypatch.setattr(batch, "_read_at", read)
    monkeypatch.setattr(batch, "_write_at", write)
    monkeypatch.setattr(batch, "prepare_outcome_diagnostic_model", prepare)
    monkeypatch.setattr(
        batch,
        "write_outcome_model_artifact",
        lambda _root, record, _state, **_kwargs: records.__setitem__(
            str(record).removeprefix("record-"), record
        ),
    )
    kwargs = dict(
        preparation_git_commit_sha="c" * 40,
        preparation_provenance_sha256=provenance_sha,
        preparation_provenance=provenance,
        limit=1,
    )
    with pytest.raises(
        batch.OutcomeDiagnosticModelBatchError,
        match="simulated progress publication crash",
    ):
        batch.prepare_outcome_diagnostic_model_batch(plan, _snapshot(), tmp_path, {}, **kwargs)
    assert len(records) == 1
    assert len(prepare_calls) == 1

    resumed = batch.prepare_outcome_diagnostic_model_batch(
        plan, _snapshot(), tmp_path, {}, **kwargs
    )
    assert not resumed.complete
    assert tuple(records) == resumed.completed_owner_ids
    assert len(prepare_calls) == 1
    assert progress is not None
    repaired = batch.OutcomeModelPreparationProgress.model_validate_json(progress)
    assert repaired.completed_owner_ids == resumed.completed_owner_ids


@pytest.mark.parametrize("kwargs", [{"dirty": True}, {"device": "mps"}])
def test_batch_provenance_policy_rejects(kwargs: dict[str, object]) -> None:
    provenance = _provenance(**kwargs)
    with pytest.raises(OutcomeDiagnosticModelBatchError):
        batch._bind_provenance(
            provenance,
            preparation_git_commit_sha="c" * 40,
            preparation_provenance_sha256=provenance_identity_sha256(provenance),
        )


def test_provenance_persistence_ignores_capture_time_but_rejects_identity_drift(
    tmp_path: Path,
) -> None:
    first = _provenance()
    second = first.model_copy(update={"captured_at_utc": datetime.now(timezone.utc)})
    drifted = first.model_copy(update={"python_version": "different"})
    with batch.open_outcome_model_store(tmp_path) as store:
        batch._persist_provenance(store, first)
        batch._persist_provenance(store, second)
        with pytest.raises(OutcomeDiagnosticModelBatchError, match="differs"):
            batch._persist_provenance(store, drifted)
