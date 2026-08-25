"""Preparation-only, resumable batch for the frozen outcome diagnostic models.

The batch owns model construction and persistence only.  It accepts already
loaded canonical evidence bytes, never opens an environment or result store,
and can therefore be resumed before any outcome/search work is authorized.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
    AuthorizedOutcomeModelArtifact,
    OutcomeDiagnosticModelArtifactRecord,
    PinnedOutcomeModelState,
    PinnedOutcomeTrainingEvidence,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_preparation import (
    prepare_outcome_diagnostic_model,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_store import (
    OutcomeModelStoreError,
    PinnedOutcomeModelStore,
    load_outcome_model_artifact_at,
    load_outcome_model_manifest_at,
    open_outcome_model_store,
    scan_outcome_model_inventory_at,
    write_outcome_model_artifact,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    EXPECTED_MODEL_OWNERS,
    EXPECTED_VIEWS,
    ValidatedOutcomePlan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    OutcomeDiagnosticProtocolSnapshot,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import SystemProvenance
from levelup.experiments.runner.storage import provenance_identity_sha256
from levelup.experiments.runner.training_data_artifacts import TrainingDataPayload

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
PROGRESS_SCHEMA_VERSION = "milestone6.phase3.outcome-diagnostic-model-preparation-progress.v1"
PROGRESS_NAME = "preparation-progress.json"
PREPARATION_PROVENANCE_NAME = "outcome-model-preparation-provenance.json"


class OutcomeDiagnosticModelBatchError(ValueError):
    """Raised when batch inputs, progress, or persisted owners drift."""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read_at(directory_fd: int, name: str) -> bytes:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except OSError as exc:
        raise OutcomeDiagnosticModelBatchError(f"cannot read {name}") from exc
    try:
        before = os.fstat(fd)
        path_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(path_before.st_mode)
            or (before.st_dev, before.st_ino) != (path_before.st_dev, path_before.st_ino)
        ):
            raise OSError("progress entry is not a stable regular file")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 1024 * 1024):
            chunks.append(chunk)
        content = b"".join(chunks)
        after = os.fstat(fd)
        path_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
            or (after.st_dev, after.st_ino) != (path_after.st_dev, path_after.st_ino)
            or len(content) != after.st_size
        ):
            raise OSError("progress entry changed while being read")
        return content
    except OSError as exc:
        raise OutcomeDiagnosticModelBatchError(f"cannot read {name}") from exc
    finally:
        os.close(fd)


def _missing(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, FileNotFoundError):
            return True
        current = current.__cause__
    return False


def _write_at(store: PinnedOutcomeModelStore, name: str, content: bytes) -> None:
    """Atomically write one progress file relative to the held store fd."""

    reader = store.reader
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    fd: int | None = None
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=reader.staging_fd,
        )
        view = memoryview(content)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short progress write")
            view = view[written:]
        os.fsync(fd)
    except OSError as exc:
        raise OutcomeDiagnosticModelBatchError("cannot stage preparation progress") from exc
    finally:
        if fd is not None:
            os.close(fd)
    try:
        store.recheck()
        os.replace(temporary, name, src_dir_fd=reader.staging_fd, dst_dir_fd=reader.root_fd)
        os.fsync(reader.root_fd)
        store.recheck()
    except OSError as exc:
        raise OutcomeDiagnosticModelBatchError("cannot publish preparation progress") from exc
    finally:
        try:
            os.unlink(temporary, dir_fd=reader.staging_fd)
        except FileNotFoundError:
            pass


def _claim_at(store: PinnedOutcomeModelStore, name: str, content: bytes) -> bool:
    """Publish a write-once file using an exclusive descriptor-relative link."""

    reader = store.reader
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    fd: int | None = None
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=reader.staging_fd,
        )
        view = memoryview(content)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short provenance write")
            view = view[written:]
        os.fsync(fd)
        store.recheck()
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=reader.staging_fd,
                dst_dir_fd=reader.root_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            return False
        os.fsync(reader.root_fd)
        store.recheck()
        return True
    except OSError as exc:
        raise OutcomeDiagnosticModelBatchError("cannot claim preparation provenance") from exc
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary, dir_fd=reader.staging_fd)
        except FileNotFoundError:
            pass


class OutcomeModelPreparationProgress(BaseModel):
    """Canonical resumable cursor, bound to all preparation authorities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[PROGRESS_SCHEMA_VERSION] = PROGRESS_SCHEMA_VERSION
    progress_sha256: str = Field(pattern=HEX64)
    plan_id: str = Field(pattern=HEX64)
    protocol_sha256: str = Field(pattern=HEX64)
    preparation_git_commit_sha: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    preparation_provenance_sha256: str = Field(pattern=HEX64)
    completed_owner_ids: tuple[str, ...] = ()

    @property
    def expected_progress_sha256(self) -> str:
        return _sha(self.model_dump(mode="json", exclude={"progress_sha256"}))

    @model_validator(mode="after")
    def canonical(self) -> "OutcomeModelPreparationProgress":
        if self.progress_sha256 != self.expected_progress_sha256:
            raise ValueError("preparation progress self-hash mismatch")
        if set(self.preparation_git_commit_sha) == {"0"}:
            raise ValueError("preparation git identity is required")
        if set(self.preparation_provenance_sha256) == {"0"}:
            raise ValueError("preparation provenance identity is required")
        if tuple(self.completed_owner_ids) != tuple(sorted(self.completed_owner_ids)):
            raise ValueError("preparation progress owners are not canonical")
        if len(set(self.completed_owner_ids)) != len(self.completed_owner_ids):
            raise ValueError("preparation progress owners are duplicated")
        if any(not HEX64.fullmatch(owner) for owner in self.completed_owner_ids):
            raise ValueError("preparation progress contains a foreign owner identity")
        return self


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticModelBatchResult:
    """Exact typed preparation outputs; no outcome/search data is included."""

    complete: bool
    completed_owner_ids: tuple[str, ...]
    records: tuple[OutcomeDiagnosticModelArtifactRecord, ...]
    state_payloads: Mapping[str, PinnedOutcomeModelState]
    authorizations: Mapping[str, AuthorizedOutcomeModelArtifact]
    training_evidence_by_view: Mapping[str, PinnedOutcomeTrainingEvidence]
    progress: OutcomeModelPreparationProgress


def _validate_identity(value: str, *, commit: bool) -> None:
    pattern = HEX_COMMIT if commit else HEX64
    if not isinstance(value, str) or not pattern.fullmatch(value) or set(value) == {"0"}:
        label = "git commit" if commit else "provenance"
        raise OutcomeDiagnosticModelBatchError(f"nonzero preparation {label} identity required")


def _bind_provenance(
    provenance: SystemProvenance,
    *,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
) -> tuple[str, str]:
    if type(provenance) is not SystemProvenance or provenance.git_dirty:
        raise OutcomeDiagnosticModelBatchError(
            "model preparation provenance must be clean and typed"
        )
    try:
        if SystemProvenance.model_validate(provenance.model_dump(mode="json")) != provenance:
            raise ValueError("typed provenance differs from canonical schema")
    except (TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelBatchError(
            "model preparation provenance is not canonical"
        ) from exc
    if (
        provenance.git_commit_sha != preparation_git_commit_sha
        or provenance.requested_device != "cpu"
        or provenance.resolved_device != "cpu"
        or provenance.requested_torch_threads != 1
        or provenance.actual_torch_threads != 1
        or provenance.requested_torch_interop_threads != 1
        or provenance.actual_torch_interop_threads != 1
        or provenance.processes != 1
    ):
        raise OutcomeDiagnosticModelBatchError(
            "model preparation provenance does not describe CPU one-thread execution"
        )
    observed = provenance_identity_sha256(provenance)
    if observed != preparation_provenance_sha256:
        raise OutcomeDiagnosticModelBatchError("preparation provenance digest differs")
    return provenance.git_commit_sha, observed


def _persist_provenance(store: PinnedOutcomeModelStore, provenance: SystemProvenance) -> None:
    content = canonical_json_bytes(provenance.model_dump(mode="json")) + b"\n"
    if _claim_at(store, PREPARATION_PROVENANCE_NAME, content):
        return
    try:
        existing = _read_at(store.reader.root_fd, PREPARATION_PROVENANCE_NAME)
    except OutcomeDiagnosticModelBatchError as exc:
        raise OutcomeDiagnosticModelBatchError("provenance claim was lost") from exc
    try:
        value = json.loads(existing)
        if canonical_json_bytes(value) + b"\n" != existing:
            raise ValueError("non-canonical provenance bytes")
        prior = SystemProvenance.model_validate(value)
        _bind_provenance(
            prior,
            preparation_git_commit_sha=provenance.git_commit_sha,
            preparation_provenance_sha256=provenance_identity_sha256(provenance),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OutcomeDiagnosticModelBatchError("persisted preparation provenance differs") from exc
    if provenance_identity_sha256(prior) != provenance_identity_sha256(provenance):
        raise OutcomeDiagnosticModelBatchError("persisted preparation provenance differs")


def _owner_ids(
    plan: ValidatedOutcomePlan, owner_ids: tuple[str, ...] | None, limit: int | None
) -> tuple[str, ...]:
    expected = tuple(owner.owner_id for owner in plan.plan.model_owners)
    if len(expected) != EXPECTED_MODEL_OWNERS or len(set(expected)) != EXPECTED_MODEL_OWNERS:
        raise OutcomeDiagnosticModelBatchError("validated plan does not contain exact 240 owners")
    if owner_ids is not None and limit is not None:
        raise OutcomeDiagnosticModelBatchError("owner_ids and limit are mutually exclusive")
    if limit is not None and (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 0 <= limit <= EXPECTED_MODEL_OWNERS
    ):
        raise OutcomeDiagnosticModelBatchError("limit is outside the frozen 240-owner universe")
    if owner_ids is None:
        return expected if limit is None else expected[:limit]
    selected = tuple(owner_ids)
    if len(set(selected)) != len(selected) or any(item not in set(expected) for item in selected):
        raise OutcomeDiagnosticModelBatchError("owner_ids are foreign or duplicated")
    return selected


def _validate_evidence(
    plan: ValidatedOutcomePlan,
    evidence_by_family_replicate: Mapping[tuple[str, int], PinnedOutcomeTrainingEvidence],
) -> dict[str, PinnedOutcomeTrainingEvidence]:
    if not isinstance(evidence_by_family_replicate, Mapping):
        raise OutcomeDiagnosticModelBatchError("evidence must be a mapping")
    expected_keys: set[tuple[str, int]] = set()
    rows_by_key: dict[tuple[str, int], dict[str, object]] = {}
    for raw in plan.plan.evidence_lineage_rows:
        try:
            row = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OutcomeDiagnosticModelBatchError(
                "evidence lineage row is not canonical JSON"
            ) from exc
        if not isinstance(row, dict) or canonical_json_bytes(row) != raw:
            raise OutcomeDiagnosticModelBatchError("evidence lineage row bytes differ")
        key = (row.get("family_id"), row.get("replicate"))
        if not isinstance(key[0], str) or not isinstance(key[1], int) or isinstance(key[1], bool):
            raise OutcomeDiagnosticModelBatchError("evidence lineage key is malformed")
        expected_keys.add(key)
        if key in rows_by_key:
            raise OutcomeDiagnosticModelBatchError("duplicate evidence lineage source")
        rows_by_key[key] = row
    if len(expected_keys) != 30 or set(evidence_by_family_replicate) != expected_keys:
        raise OutcomeDiagnosticModelBatchError(
            "evidence mapping must contain exactly 30 frozen sources"
        )
    view_evidence: dict[str, PinnedOutcomeTrainingEvidence] = {}
    by_pair: dict[tuple[str, int], bytes] = {}
    for view in plan.plan.views:
        key = (view.heldout_family, view.replicate)
        evidence = evidence_by_family_replicate[key]
        if not isinstance(evidence, PinnedOutcomeTrainingEvidence) or not isinstance(
            evidence.payload, TrainingDataPayload
        ):
            raise OutcomeDiagnosticModelBatchError("evidence payload is not typed and canonical")
        payload_bytes = canonical_json_bytes(evidence.payload.model_dump(mode="json"))
        if evidence.payload_bytes != payload_bytes:
            raise OutcomeDiagnosticModelBatchError(
                "evidence payload bytes differ from parsed payload"
            )
        row = rows_by_key[key]
        if (
            hashlib.sha256(payload_bytes).hexdigest() != row.get("payload_sha256")
            or len(payload_bytes) != row.get("payload_bytes")
            or tuple(sample.task_id for sample in evidence.payload.samples)
            != tuple(view.training_task_ids)
        ):
            raise OutcomeDiagnosticModelBatchError(
                "evidence payload does not match frozen view lineage"
            )
        prior = by_pair.get(key)
        if prior is not None and prior != evidence.payload_bytes:
            raise OutcomeDiagnosticModelBatchError(
                "RP and PEC views do not share exact evidence bytes"
            )
        by_pair[key] = evidence.payload_bytes
        view_evidence[view.view_id] = evidence
    if len(view_evidence) != EXPECTED_VIEWS:
        raise OutcomeDiagnosticModelBatchError("evidence view coverage is incomplete")
    source_counts = {
        key: sum(1 for view in plan.plan.views if (view.heldout_family, view.replicate) == key)
        for key in expected_keys
    }
    if set(source_counts.values()) != {2}:
        raise OutcomeDiagnosticModelBatchError(
            "each evidence source must back exactly RP and PEC views"
        )
    return view_evidence


def _load_existing(
    store: PinnedOutcomeModelStore,
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    evidence_by_view: Mapping[str, PinnedOutcomeTrainingEvidence],
    *,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
) -> tuple[
    dict[str, OutcomeDiagnosticModelArtifactRecord],
    dict[str, PinnedOutcomeModelState],
    dict[str, AuthorizedOutcomeModelArtifact],
]:
    try:
        manifest = load_outcome_model_manifest_at(store.reader)
    except OutcomeModelStoreError as exc:
        raise OutcomeDiagnosticModelBatchError("model store manifest is invalid") from exc
    expected = {owner.owner_id for owner in plan.plan.model_owners}
    found = {entry.owner_id for entry in manifest.entries}
    if not found <= expected:
        raise OutcomeDiagnosticModelBatchError("model store contains a foreign owner")
    records: dict[str, OutcomeDiagnosticModelArtifactRecord] = {}
    states: dict[str, PinnedOutcomeModelState] = {}
    authorizations: dict[str, AuthorizedOutcomeModelArtifact] = {}
    for owner_id in sorted(found):
        owner = next(item for item in plan.plan.model_owners if item.owner_id == owner_id)
        try:
            record, state, authorization = load_outcome_model_artifact_at(
                store.reader,
                owner_id,
                evidence_by_view[owner.view_id],
                plan,
                snapshot,
                preparation_git_commit_sha=preparation_git_commit_sha,
                preparation_provenance_sha256=preparation_provenance_sha256,
            )
        except (OutcomeModelStoreError, KeyError) as exc:
            raise OutcomeDiagnosticModelBatchError(
                "persisted owner failed semantic validation"
            ) from exc
        if (
            getattr(record.key, "preparation_git_commit_sha", None) != preparation_git_commit_sha
            or getattr(record.key, "preparation_provenance_sha256", None)
            != preparation_provenance_sha256
        ):
            raise OutcomeDiagnosticModelBatchError("persisted owner preparation provenance differs")
        records[owner_id] = record
        states[owner_id] = state
        authorizations[owner_id] = authorization
    return records, states, authorizations


def _progress_bytes(progress: OutcomeModelPreparationProgress) -> bytes:
    return canonical_json_bytes(progress.model_dump(mode="json")) + b"\n"


def _make_progress(
    *,
    plan_id: str,
    protocol_sha256: str,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
    completed_owner_ids: tuple[str, ...],
) -> OutcomeModelPreparationProgress:
    body = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "progress_sha256": "0" * 64,
        "plan_id": plan_id,
        "protocol_sha256": protocol_sha256,
        "preparation_git_commit_sha": preparation_git_commit_sha,
        "preparation_provenance_sha256": preparation_provenance_sha256,
        "completed_owner_ids": completed_owner_ids,
    }
    body["progress_sha256"] = _sha(
        {key: value for key, value in body.items() if key != "progress_sha256"}
    )
    return OutcomeModelPreparationProgress.model_validate(body)


def prepare_outcome_diagnostic_model_batch(
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    output_root: str | Path,
    evidence_by_family_replicate: Mapping[tuple[str, int], PinnedOutcomeTrainingEvidence],
    *,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
    preparation_provenance: SystemProvenance,
    owner_ids: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> OutcomeDiagnosticModelBatchResult:
    """Prepare a bounded owner subset and resume from semantically checked state."""

    if type(plan) is not ValidatedOutcomePlan or not isinstance(
        snapshot, OutcomeDiagnosticProtocolSnapshot
    ):
        raise OutcomeDiagnosticModelBatchError(
            "validated outcome plan and protocol snapshot are required"
        )
    _validate_identity(preparation_git_commit_sha, commit=True)
    _validate_identity(preparation_provenance_sha256, commit=False)
    preparation_git_commit_sha, preparation_provenance_sha256 = _bind_provenance(
        preparation_provenance,
        preparation_git_commit_sha=preparation_git_commit_sha,
        preparation_provenance_sha256=preparation_provenance_sha256,
    )
    selected = _owner_ids(plan, owner_ids, limit)
    evidence_by_view = _validate_evidence(plan, evidence_by_family_replicate)
    with open_outcome_model_store(output_root) as store:
        _persist_provenance(store, preparation_provenance)
        records, states, authorizations = _load_existing(
            store,
            plan,
            snapshot,
            evidence_by_view,
            preparation_git_commit_sha=preparation_git_commit_sha,
            preparation_provenance_sha256=preparation_provenance_sha256,
        )
        existing_ids = tuple(sorted(records))
        progress_raw: bytes | None
        try:
            progress_raw = _read_at(store.reader.root_fd, PROGRESS_NAME)
        except OutcomeDiagnosticModelBatchError as exc:
            if not _missing(exc):
                raise
            progress_raw = None
        if progress_raw is None:
            progress = _make_progress(
                plan_id=plan.plan.plan_id,
                protocol_sha256=plan.plan.protocol_sha256,
                preparation_git_commit_sha=preparation_git_commit_sha,
                preparation_provenance_sha256=preparation_provenance_sha256,
                completed_owner_ids=existing_ids,
            )
            _write_at(store, PROGRESS_NAME, _progress_bytes(progress))
        else:
            try:
                value = json.loads(progress_raw)
                if canonical_json_bytes(value) + b"\n" != progress_raw:
                    raise ValueError("non-canonical preparation progress bytes")
                progress = OutcomeModelPreparationProgress.model_validate(value)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise OutcomeDiagnosticModelBatchError("preparation progress is invalid") from exc
            if (
                progress.plan_id != plan.plan.plan_id
                or progress.protocol_sha256 != plan.plan.protocol_sha256
                or progress.preparation_git_commit_sha != preparation_git_commit_sha
                or progress.preparation_provenance_sha256 != preparation_provenance_sha256
            ):
                raise OutcomeDiagnosticModelBatchError(
                    "preparation progress authority does not match this batch"
                )
            progress_ids = set(progress.completed_owner_ids)
            if not progress_ids <= set(existing_ids):
                raise OutcomeDiagnosticModelBatchError(
                    "preparation progress is ahead of semantic inventory"
                )
            if tuple(progress.completed_owner_ids) != existing_ids:
                progress = _make_progress(
                    plan_id=plan.plan.plan_id,
                    protocol_sha256=plan.plan.protocol_sha256,
                    preparation_git_commit_sha=preparation_git_commit_sha,
                    preparation_provenance_sha256=preparation_provenance_sha256,
                    completed_owner_ids=existing_ids,
                )
                _write_at(store, PROGRESS_NAME, _progress_bytes(progress))
        for owner_id in selected:
            if owner_id in records:
                continue
            owner = next(item for item in plan.plan.model_owners if item.owner_id == owner_id)
            prepared = prepare_outcome_diagnostic_model(
                plan,
                snapshot,
                owner_id=owner_id,
                training_evidence=evidence_by_view[owner.view_id],
                preparation_git_commit_sha=preparation_git_commit_sha,
                preparation_provenance_sha256=preparation_provenance_sha256,
            )
            write_outcome_model_artifact(
                output_root,
                prepared.record,
                prepared.state_payload,
                pinned_output=store,
            )
            records[owner_id] = prepared.record
            states[owner_id] = prepared.state_payload
            authorizations[owner_id] = prepared.authorization
            progress = _make_progress(
                plan_id=plan.plan.plan_id,
                protocol_sha256=plan.plan.protocol_sha256,
                preparation_git_commit_sha=preparation_git_commit_sha,
                preparation_provenance_sha256=preparation_provenance_sha256,
                completed_owner_ids=tuple(sorted(records)),
            )
            _write_at(store, PROGRESS_NAME, _progress_bytes(progress))
        complete = False
        if len(records) == EXPECTED_MODEL_OWNERS:
            scan_outcome_model_inventory_at(
                store.reader,
                tuple(sorted(records)),
                evidence_by_view,
                plan,
                snapshot,
                preparation_git_commit_sha=preparation_git_commit_sha,
                preparation_provenance_sha256=preparation_provenance_sha256,
            )
            complete = True
        return OutcomeDiagnosticModelBatchResult(
            complete=complete,
            completed_owner_ids=tuple(sorted(records)),
            records=tuple(records[key] for key in sorted(records)),
            state_payloads=dict(states),
            authorizations=dict(authorizations),
            training_evidence_by_view=dict(evidence_by_view),
            progress=progress,
        )


prepare_outcome_diagnostic_models = prepare_outcome_diagnostic_model_batch


__all__ = [
    "OutcomeDiagnosticModelBatchError",
    "OutcomeDiagnosticModelBatchResult",
    "OutcomeModelPreparationProgress",
    "PREPARATION_PROVENANCE_NAME",
    "PROGRESS_NAME",
    "prepare_outcome_diagnostic_model_batch",
    "prepare_outcome_diagnostic_models",
]
