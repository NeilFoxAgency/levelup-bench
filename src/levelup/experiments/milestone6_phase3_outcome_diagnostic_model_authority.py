"""Publish an opaque authority for the complete development model store.

This module is a read-only development boundary.  It consumes the frozen
outcome plan, typed training evidence, and a descriptor-pinned preparation
store, then returns the canonical compact authority identity.  It does not
open environments, result stores, evaluators, search, or any final-family
resource, and it never writes an authority file.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
    OutcomeDiagnosticModelArtifactAuthority,
    OutcomeDiagnosticModelArtifactError,
    PinnedOutcomeTrainingEvidence,
    build_outcome_model_artifact_authority,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_batch import (
    PREPARATION_PROVENANCE_NAME,
    PROGRESS_NAME,
    OutcomeModelPreparationProgress,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_store import (
    OutcomeModelStoreError,
    PinnedOutcomeModelStore,
    PinnedOutcomeModelStoreReader,
    load_outcome_model_artifact_at,
    open_outcome_model_store,
    scan_outcome_model_inventory_at,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    EXPECTED_MODEL_OWNERS,
    ValidatedOutcomePlan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    OutcomeDiagnosticProtocolSnapshot,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import SystemProvenance
from levelup.experiments.runner.storage import provenance_identity_sha256


class OutcomeDiagnosticModelAuthorityError(ValueError):
    """Raised when the complete development model store cannot be authorized."""


def _reader(
    value: PinnedOutcomeModelStore | PinnedOutcomeModelStoreReader,
) -> PinnedOutcomeModelStoreReader:
    """Return the descriptor-pinned reader while preserving store authority."""

    if type(value) is PinnedOutcomeModelStore:
        return value.reader
    if type(value) is PinnedOutcomeModelStoreReader:
        return value
    raise OutcomeDiagnosticModelAuthorityError("canonical pinned model store is required")


def _read_canonical_json(reader: Any, name: str) -> dict[str, Any]:
    try:
        raw = _read_store_file(reader, name)
        value = json.loads(raw)
        if not isinstance(value, dict) or canonical_json_bytes(value) + b"\n" != raw:
            raise ValueError("non-canonical JSON bytes")
        return value
    except (OSError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise OutcomeDiagnosticModelAuthorityError(
            f"model-store {name} is not canonical JSON"
        ) from exc


def _read_store_file(reader: Any, name: str) -> bytes:
    """Read one root file using the store's stable descriptor-relative primitive."""

    # Keep this indirection small so tests can exercise provenance failures
    # without constructing 240 tensor directories.  The production primitive
    # checks descriptor/path identity before and after the complete read.
    try:
        from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_store import (
            _read_stable,
        )

        return _read_stable(reader.root_fd, name)
    except (AttributeError, ImportError, OutcomeModelStoreError) as exc:
        raise OutcomeDiagnosticModelAuthorityError(
            f"cannot read model-store {name}"
        ) from exc


def _validate_persisted_progress_and_provenance(
    reader: Any,
    plan: ValidatedOutcomePlan,
    *,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
    expected_owner_ids: tuple[str, ...],
) -> None:
    progress_data = _read_canonical_json(reader, PROGRESS_NAME)
    try:
        progress = OutcomeModelPreparationProgress.model_validate(progress_data)
    except (TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelAuthorityError("model preparation progress is invalid") from exc
    if (
        progress.plan_id != plan.plan.plan_id
        or progress.protocol_sha256 != plan.plan.protocol_sha256
        or progress.preparation_git_commit_sha != preparation_git_commit_sha
        or progress.preparation_provenance_sha256 != preparation_provenance_sha256
        or tuple(progress.completed_owner_ids) != expected_owner_ids
    ):
        raise OutcomeDiagnosticModelAuthorityError(
            "model preparation progress is incomplete or provenance-bound to another run"
        )

    provenance_data = _read_canonical_json(reader, PREPARATION_PROVENANCE_NAME)
    try:
        provenance = SystemProvenance.model_validate(provenance_data)
    except (TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelAuthorityError(
            "model preparation provenance is invalid"
        ) from exc
    if (
        provenance.git_dirty
        or provenance.git_commit_sha != preparation_git_commit_sha
        or provenance_identity_sha256(provenance) != preparation_provenance_sha256
        or provenance.requested_device != "cpu"
        or provenance.resolved_device != "cpu"
        or provenance.requested_torch_threads != 1
        or provenance.actual_torch_threads != 1
        or provenance.requested_torch_interop_threads != 1
        or provenance.actual_torch_interop_threads != 1
        or provenance.processes != 1
    ):
        raise OutcomeDiagnosticModelAuthorityError(
            "model preparation provenance differs from the complete CPU preparation"
        )


def _store_identity_snapshot(reader: Any, expected_owner_ids: tuple[str, ...]) -> Any:
    """Obtain the store worker's complete identity snapshot, fail closed if absent."""

    try:
        from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_store import (
            snapshot_outcome_model_store_identities_at,
        )

        return snapshot_outcome_model_store_identities_at(reader, expected_owner_ids)
    except ImportError as exc:
        raise OutcomeDiagnosticModelAuthorityError(
            "model-store identity snapshot primitive is unavailable"
        ) from exc
    except (
        AttributeError,
        OutcomeModelStoreError,
        OSError,
        ValueError,
        secure_fs.SecureFilesystemError,
    ) as exc:
        raise OutcomeDiagnosticModelAuthorityError(
            "model-store identity snapshot failed"
        ) from exc


def build_outcome_model_artifact_authority_from_store(
    store_root: str | Path,
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    training_evidence_by_view: Mapping[str, PinnedOutcomeTrainingEvidence],
    *,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
    generation_git_commit_sha: str,
) -> OutcomeDiagnosticModelArtifactAuthority:
    """Build the canonical compact authority from one complete model store.

    The store is opened and held for the entire semantic operation.  Its
    complete identity snapshot is taken before inventory validation and again
    after authority construction; any drift invalidates the operation.
    """

    if type(plan) is not ValidatedOutcomePlan:
        raise OutcomeDiagnosticModelAuthorityError("validated outcome plan is required")
    if not isinstance(snapshot, OutcomeDiagnosticProtocolSnapshot):
        raise OutcomeDiagnosticModelAuthorityError(
            "canonical outcome protocol snapshot is required"
        )
    if not isinstance(training_evidence_by_view, Mapping):
        raise OutcomeDiagnosticModelAuthorityError("typed training evidence mapping is required")
    expected_owner_ids = tuple(sorted(owner.owner_id for owner in plan.plan.model_owners))
    if (
        len(expected_owner_ids) != EXPECTED_MODEL_OWNERS
        or len(set(expected_owner_ids)) != EXPECTED_MODEL_OWNERS
    ):
        raise OutcomeDiagnosticModelAuthorityError(
            "outcome plan does not contain exactly 240 owners"
        )
    expected_view_ids = {view.view_id for view in plan.plan.views}
    if set(training_evidence_by_view) != expected_view_ids:
        raise OutcomeDiagnosticModelAuthorityError("training evidence view universe is incomplete")
    if any(
        not isinstance(value, PinnedOutcomeTrainingEvidence)
        for value in training_evidence_by_view.values()
    ):
        raise OutcomeDiagnosticModelAuthorityError("training evidence values are not typed")

    try:
        with open_outcome_model_store(store_root) as store:
            reader = _reader(store)
            reader.recheck()
            initial_identities = _store_identity_snapshot(reader, expected_owner_ids)
            _validate_persisted_progress_and_provenance(
                reader,
                plan,
                preparation_git_commit_sha=preparation_git_commit_sha,
                preparation_provenance_sha256=preparation_provenance_sha256,
                expected_owner_ids=expected_owner_ids,
            )
            scan_outcome_model_inventory_at(
                reader,
                expected_owner_ids,
                training_evidence_by_view,
                plan,
                snapshot,
                preparation_git_commit_sha=preparation_git_commit_sha,
                preparation_provenance_sha256=preparation_provenance_sha256,
            )
            records = []
            state_payloads = {}
            owner_view_ids = {
                owner.owner_id: owner.view_id for owner in plan.plan.model_owners
            }
            for owner_id in expected_owner_ids:
                record, state, _authorization = load_outcome_model_artifact_at(
                    reader,
                    owner_id,
                    training_evidence_by_view[owner_view_ids[owner_id]],
                    plan,
                    snapshot,
                    preparation_git_commit_sha,
                    preparation_provenance_sha256,
                )
                records.append(record)
                state_payloads[owner_id] = state
            authority = build_outcome_model_artifact_authority(
                records,
                state_payloads,
                training_evidence_by_view,
                plan,
                snapshot,
                preparation_git_commit_sha=preparation_git_commit_sha,
                preparation_provenance_sha256=preparation_provenance_sha256,
                generation_git_commit_sha=generation_git_commit_sha,
            )
            final_identities = _store_identity_snapshot(reader, expected_owner_ids)
            if final_identities != initial_identities:
                raise OutcomeDiagnosticModelAuthorityError(
                    "model-store identities changed during authority construction"
                )
            reader.recheck()
            return authority
    except OutcomeDiagnosticModelAuthorityError:
        raise
    except (
        OSError,
        TypeError,
        ValueError,
        OutcomeModelStoreError,
        OutcomeDiagnosticModelArtifactError,
    ) as exc:
        raise OutcomeDiagnosticModelAuthorityError(
            "complete development model store failed authority validation"
        ) from exc


__all__ = [
    "OutcomeDiagnosticModelAuthorityError",
    "build_outcome_model_artifact_authority_from_store",
]
