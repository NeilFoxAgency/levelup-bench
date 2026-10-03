"""Read-only, descriptor-pinned inventory of prepared Phase 3 model owners.

This module is the canonical progress view for local-affordance preparation.
It accepts only the frozen development plan, reads the existing one-owner store
without creating paths, and reloads every committed model payload through the
store's plan/provenance/hash validation boundary.  It does not train, search,
open an environment, inspect outcomes, or access final-family data.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from levelup.experiments.milestone6_phase3_local_affordance_model_records import (
    LocalAffordanceModelRecord,
)
from levelup.experiments.milestone6_phase3_local_affordance_model_store import (
    MANIFEST_NAME,
    LocalAffordanceModelStoreError,
    _manifest,
    _stable_file_identity,
    _validate_shape,
    load_local_affordance_model_at,
    open_local_affordance_model_store,
)
from levelup.experiments.milestone6_phase3_local_affordance_plan import (
    EXPECTED_COUNTS,
    FROZEN_SOURCE_SHA256,
    LocalAffordancePlan,
    LocalAffordancePlanError,
    validate_local_affordance_plan,
)
from levelup.experiments.runner.config import canonical_json_bytes

SCHEMA_VERSION = "milestone6.phase3.local-affordance-preparation-progress.v1"
HEX64 = r"^[0-9a-f]{64}$"
COMMIT = r"^[0-9a-f]{40,64}$"
EXPECTED_MODEL_OWNERS = EXPECTED_COUNTS["model_owners"]


class LocalAffordancePreparationProgressError(ValueError):
    """Raised when a store cannot be proven to match its frozen plan."""


class LocalAffordanceOwnerProgress(BaseModel):
    """Verified metadata and training/serialization costs for one owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    owner_id: str = Field(pattern=HEX64)
    record_sha256: str = Field(pattern=HEX64)
    model_state_sha256: str = Field(pattern=HEX64)
    model_identity_sha256: str = Field(pattern=HEX64)
    optimizer_steps: StrictInt = Field(gt=0)
    forward_passes: StrictInt = Field(gt=0)
    training_examples: StrictInt = Field(gt=0)
    serialized_artifact_bytes: StrictInt = Field(gt=0)


class LocalAffordancePreparationProgress(BaseModel):
    """Canonical inventory snapshot; complete means 480 owners reloaded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    plan_id: str = Field(pattern=HEX64)
    raw_authority_manifest_id: str = Field(pattern=HEX64)
    raw_authority_content_sha256: str = Field(pattern=HEX64)
    raw_manifest_file_sha256: str = Field(pattern=HEX64)
    plan_source_sha256: tuple[tuple[str, str], ...]
    preparation_git_commit_sha: str = Field(pattern=COMMIT)
    preparation_provenance_sha256: str = Field(pattern=HEX64)
    store_manifest_sha256: str = Field(pattern=HEX64)
    expected_owner_ids_sha256: str = Field(pattern=HEX64)
    expected_owner_count: StrictInt = Field(gt=0)
    completed_owner_ids: tuple[str, ...]
    missing_owner_ids: tuple[str, ...]
    owners: tuple[LocalAffordanceOwnerProgress, ...]
    total_optimizer_steps: StrictInt = Field(ge=0)
    total_forward_passes: StrictInt = Field(ge=0)
    total_training_examples: StrictInt = Field(ge=0)
    total_serialized_artifact_bytes: StrictInt = Field(ge=0)
    complete: bool
    snapshot_sha256: str = Field(pattern=HEX64)

    @model_validator(mode="after")
    def validate_inventory(self) -> "LocalAffordancePreparationProgress":
        owner_ids = tuple(item.owner_id for item in self.owners)
        if (
            self.expected_owner_count != EXPECTED_MODEL_OWNERS
            or owner_ids != tuple(sorted(set(owner_ids)))
            or self.completed_owner_ids != owner_ids
            or self.missing_owner_ids != tuple(sorted(set(self.missing_owner_ids)))
            or set(self.completed_owner_ids).intersection(self.missing_owner_ids)
            or len(owner_ids) + len(self.missing_owner_ids) != self.expected_owner_count
            or _sha256(
                canonical_json_bytes(
                    tuple(sorted((*self.completed_owner_ids, *self.missing_owner_ids)))
                )
            )
            != self.expected_owner_ids_sha256
        ):
            raise ValueError("progress owner inventory is inconsistent")
        if (
            self.total_optimizer_steps != sum(item.optimizer_steps for item in self.owners)
            or self.total_forward_passes != sum(item.forward_passes for item in self.owners)
            or self.total_training_examples != sum(item.training_examples for item in self.owners)
            or self.total_serialized_artifact_bytes
            != sum(item.serialized_artifact_bytes for item in self.owners)
        ):
            raise ValueError("progress aggregate costs differ from owner rows")
        expected_complete = len(owner_ids) == EXPECTED_MODEL_OWNERS and not self.missing_owner_ids
        if self.complete is not expected_complete:
            raise ValueError("progress can be complete only after all 480 owners validate")
        body = self.model_dump(mode="json", exclude={"snapshot_sha256"})
        if self.snapshot_sha256 != _sha256(canonical_json_bytes(body)):
            raise ValueError("progress snapshot digest mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        """Return the deterministic persisted representation."""
        return canonical_json_bytes(self.model_dump(mode="json")) + b"\n"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _manifest_identity(reader) -> tuple[int, int, int, int, int, int] | None:
    try:
        os.stat(MANIFEST_NAME, dir_fd=reader.root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return _stable_file_identity(reader.root_fd, MANIFEST_NAME)


def _snapshot_from_validated_records(
    *,
    plan: LocalAffordancePlan,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
    store_manifest_sha256: str,
    records: tuple[LocalAffordanceModelRecord, ...],
) -> LocalAffordancePreparationProgress:
    """Reduce already reloaded and validated owner records to typed progress."""
    expected = tuple(sorted(owner.owner_id for owner in plan.model_owners))
    if len(expected) != EXPECTED_MODEL_OWNERS or len(set(expected)) != EXPECTED_MODEL_OWNERS:
        raise LocalAffordancePreparationProgressError("frozen owner universe is incomplete or duplicated")
    by_id: dict[str, LocalAffordanceModelRecord] = {}
    for record in records:
        owner_id = record.key.owner_id
        if owner_id not in set(expected):
            raise LocalAffordancePreparationProgressError("store contains a foreign model owner")
        if owner_id in by_id:
            raise LocalAffordancePreparationProgressError("store contains a duplicate model owner")
        by_id[owner_id] = record
    completed = tuple(sorted(by_id))
    missing = tuple(sorted(set(expected) - set(completed)))
    rows = tuple(
        LocalAffordanceOwnerProgress(
            owner_id=owner_id,
            record_sha256=record.record_sha256,
            model_state_sha256=record.key.model_state_sha256,
            model_identity_sha256=record.key.model_identity_sha256,
            optimizer_steps=record.cost.optimizer_steps,
            forward_passes=record.cost.forward_passes,
            training_examples=record.cost.training_examples,
            serialized_artifact_bytes=record.artifact.artifact_bytes,
        )
        for owner_id, record in sorted(by_id.items())
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan.plan_id,
        "raw_authority_manifest_id": plan.raw_authority_manifest_id,
        "raw_authority_content_sha256": plan.raw_authority_content_sha256,
        "raw_manifest_file_sha256": plan.raw_manifest_file_sha256,
        "plan_source_sha256": plan.source_sha256,
        "preparation_git_commit_sha": preparation_git_commit_sha,
        "preparation_provenance_sha256": preparation_provenance_sha256,
        "store_manifest_sha256": store_manifest_sha256,
        "expected_owner_ids_sha256": _sha256(canonical_json_bytes(expected)),
        "expected_owner_count": EXPECTED_MODEL_OWNERS,
        "completed_owner_ids": completed,
        "missing_owner_ids": missing,
        "owners": tuple(row.model_dump(mode="json") for row in rows),
        "total_optimizer_steps": sum(row.optimizer_steps for row in rows),
        "total_forward_passes": sum(row.forward_passes for row in rows),
        "total_training_examples": sum(row.training_examples for row in rows),
        "total_serialized_artifact_bytes": sum(row.serialized_artifact_bytes for row in rows),
        "complete": len(completed) == EXPECTED_MODEL_OWNERS and not missing,
    }
    return LocalAffordancePreparationProgress(
        **body, snapshot_sha256=_sha256(canonical_json_bytes(body))
    )


def inventory_local_affordance_model_progress(
    plan: LocalAffordancePlan,
    store_root: str | Path,
    *,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
    repository: str | Path | None = None,
) -> LocalAffordancePreparationProgress:
    """Read-only inventory; every committed model payload must reload cleanly.

    ``completed_owner_ids`` are derived from the validated store manifest and
    records, never accepted from the caller. The store root must already exist.
    """
    if (
        not isinstance(preparation_git_commit_sha, str)
        or re.fullmatch(COMMIT, preparation_git_commit_sha) is None
        or set(preparation_git_commit_sha) == {"0"}
    ):
        raise LocalAffordancePreparationProgressError("preparation commit identity is invalid")
    if (
        not isinstance(preparation_provenance_sha256, str)
        or re.fullmatch(HEX64, preparation_provenance_sha256) is None
        or set(preparation_provenance_sha256) == {"0"}
    ):
        raise LocalAffordancePreparationProgressError("preparation provenance identity is invalid")
    try:
        if repository is None:
            validate_local_affordance_plan(plan)
        else:
            validate_local_affordance_plan(plan, repository=repository)
    except (LocalAffordancePlanError, TypeError, ValueError) as exc:
        raise LocalAffordancePreparationProgressError(
            "plan differs from the frozen development authority"
        ) from exc
    if plan.plan_id != "4e3390056a3a75fad36a033eecf4731dbd08ee946c354449f489c31f5a310718":
        raise LocalAffordancePreparationProgressError("plan ID differs from frozen authority")
    if tuple(plan.source_sha256) != tuple(FROZEN_SOURCE_SHA256.items()):
        raise LocalAffordancePreparationProgressError("plan source provenance differs")
    with open_local_affordance_model_store(store_root, create=False) as reader:
        try:
            # Writers take LOCK_EX. The shared lock makes the manifest and all
            # owner reloads one coherent read-only inventory interval.
            fcntl.flock(reader.root_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise LocalAffordancePreparationProgressError(
                "model store is currently being written; retry inventory later"
            ) from exc
        try:
            reader.recheck()
            manifest_identity = _manifest_identity(reader)
            manifest = _manifest(reader)
            _validate_shape(reader, manifest)
            expected = {owner.owner_id for owner in plan.model_owners}
            manifest_ids = tuple(item.owner_id for item in manifest.entries)
            if len(expected) != EXPECTED_MODEL_OWNERS or not set(manifest_ids).issubset(expected):
                raise LocalAffordancePreparationProgressError(
                    "manifest has duplicate, foreign, or noncanonical owners"
                )
            records: list[LocalAffordanceModelRecord] = []
            for owner_id in manifest_ids:
                record, _tensors = load_local_affordance_model_at(
                    reader,
                    owner_id,
                    plan=plan,
                    preparation_git_commit_sha=preparation_git_commit_sha,
                    preparation_provenance_sha256=preparation_provenance_sha256,
                )
                records.append(record)
            if manifest_identity != _manifest_identity(reader):
                raise LocalAffordancePreparationProgressError(
                    "manifest changed during model inventory"
                )
            _validate_shape(reader, manifest)
            reader.recheck()
            return _snapshot_from_validated_records(
                plan=plan,
                preparation_git_commit_sha=preparation_git_commit_sha,
                preparation_provenance_sha256=preparation_provenance_sha256,
                store_manifest_sha256=manifest.manifest_sha256,
                records=tuple(records),
            )
        finally:
            fcntl.flock(reader.root_fd, fcntl.LOCK_UN)


__all__ = [
    "LocalAffordanceModelStoreError",
    "LocalAffordanceOwnerProgress",
    "LocalAffordancePreparationProgress",
    "LocalAffordancePreparationProgressError",
    "inventory_local_affordance_model_progress",
]
