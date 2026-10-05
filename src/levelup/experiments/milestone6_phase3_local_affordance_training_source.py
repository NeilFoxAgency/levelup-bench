"""Descriptor-pinned source for one Phase 3 optimum-training evidence bundle.

This narrow, development-only boundary joins the canonical Phase 3 evidence lock
to exactly one Phase 2 runtime fold/replicate.  It returns the typed payload
bundle consumed by ``local_affordance_training_views``; it does not access raw
held-out probe bindings, outcomes, other fold payloads, or any final-family data.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from levelup.experiments.milestone6_phase2_screening_runtime import (
    ScreeningRuntime,
    ScreeningRuntimeFold,
    recheck_screening_runtime_metadata_only,
    recheck_screening_runtime_readonly,
)
from levelup.experiments.milestone6_phase3_evidence import (
    EvidenceLockError,
    Phase3EvidenceLock,
    require_phase3_evidence_lock,
)
from levelup.experiments.milestone6_phase3_protocol import FAMILIES
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataEvidenceKey,
    TrainingDataEvidenceManifest,
    TrainingDataEvidencePayloadBundle,
    load_training_data_evidence_payload_bundle_from_at,
    open_training_data_reader,
)

REPLICATES = tuple(range(5))


class LocalAffordanceTrainingSourceError(ValueError):
    """Raised when canonical optimum evidence cannot be bound to its source."""


def _lock_row(
    evidence_lock: Phase3EvidenceLock,
    *,
    family_id: str,
    replicate: int,
) -> Mapping[str, Any]:
    try:
        lock = require_phase3_evidence_lock(evidence_lock)
    except (EvidenceLockError, TypeError, ValueError) as exc:
        raise LocalAffordanceTrainingSourceError(
            "training source requires a validator-issued Phase 3 evidence lock"
        ) from exc
    body = lock.body
    if (
        body.get("schema_version") != "milestone6.phase3.evidence-lock.v1"
        or body.get("scope") != "known-development-only"
        or body.get("final_family_access") is not False
        or body.get("payloads_included") is not False
        or body.get("outcomes_included") is not False
        or body.get("aggregates") != []
        or body.get("final_results") != []
        or body.get("counts") != {"evidence_artifacts": 30, "families": 6, "replicates": 5}
    ):
        raise LocalAffordanceTrainingSourceError(
            "evidence lock is not exact known-development-only authority"
        )
    rows = body.get("evidence_artifacts")
    if type(rows) is not list or len(rows) != len(FAMILIES) * len(REPLICATES):
        raise LocalAffordanceTrainingSourceError("Phase 3 evidence matrix is incomplete")
    identities: list[tuple[Any, Any]] = []
    matches: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise LocalAffordanceTrainingSourceError("Phase 3 evidence row is malformed")
        identity = (row.get("family_id"), row.get("replicate"))
        identities.append(identity)
        if identity == (family_id, replicate):
            matches.append(row)
    expected = {(family, seed) for family in FAMILIES for seed in REPLICATES}
    if len(set(identities)) != len(identities) or set(identities) != expected:
        raise LocalAffordanceTrainingSourceError(
            "Phase 3 evidence lock family/replicate matrix drifted"
        )
    if len(matches) != 1:
        raise LocalAffordanceTrainingSourceError("requested Phase 3 evidence row is not unique")
    return matches[0]


def _runtime_fold(runtime: ScreeningRuntime, family_id: str) -> ScreeningRuntimeFold:
    if type(runtime) is not ScreeningRuntime:
        raise LocalAffordanceTrainingSourceError(
            "training source requires an exact validated ScreeningRuntime"
        )
    folds = runtime.folds
    if type(folds) is not tuple or len(folds) != len(FAMILIES):
        raise LocalAffordanceTrainingSourceError("screening runtime fold inventory is incomplete")
    identities = tuple(getattr(fold, "family_id", None) for fold in folds)
    if len(set(identities)) != len(identities) or set(identities) != set(FAMILIES):
        raise LocalAffordanceTrainingSourceError("screening runtime family inventory drifted")
    matches = tuple(fold for fold in folds if fold.family_id == family_id)
    if len(matches) != 1 or type(matches[0]) is not ScreeningRuntimeFold:
        raise LocalAffordanceTrainingSourceError("requested screening runtime fold is unavailable")
    return matches[0]


def _typed_lineage(
    row: Mapping[str, Any],
    *,
    family_id: str,
    replicate: int,
) -> tuple[TrainingDataEvidenceKey, TrainingDataEvidenceManifest, tuple[str, ...]]:
    try:
        key = TrainingDataEvidenceKey.model_validate(row["evidence_key"])
        manifest = TrainingDataEvidenceManifest.model_validate(row["evidence_manifest"])
        task_ids = tuple(row["ordered_training_task_ids"])
        manifest_sha = row["canonical_manifest_bytes_sha256"]
        payload_sha = row["payload_sha256"]
        payload_size = row["payload_bytes"]
    except (KeyError, TypeError, ValueError) as exc:
        raise LocalAffordanceTrainingSourceError("Phase 3 evidence row is not typed") from exc
    if (
        type(replicate) is not int
        or family_id not in FAMILIES
        or replicate not in REPLICATES
        or key.heldout_family_id != family_id
        or key.fold_id != f"lofo-{family_id}"
        or key.replicate != replicate
        or manifest.key != key
        or manifest.evidence_key_id != key.key_id
        or manifest.evidence_id != row.get("evidence_id")
        or row.get("family_id") != family_id
        or row.get("fold_id") != key.fold_id
        or row.get("replicate") != replicate
        or row.get("evidence_key_id") != key.key_id
        or row.get("evidence_manifest_key_id") != key.key_id
        or row.get("evidence_id") != manifest.evidence_id
        or type(task_ids) is not tuple
        or task_ids != key.ordered_training_task_ids
        or len(task_ids) != 40
        or len(set(task_ids)) != len(task_ids)
        or manifest.sample_task_ids != task_ids
        or manifest.payload_sha256 != payload_sha
        or manifest.payload_bytes != payload_size
        or not isinstance(manifest_sha, str)
        or len(manifest_sha) != 64
        or not isinstance(payload_sha, str)
        or len(payload_sha) != 64
        or type(payload_size) is not int
        or payload_size <= 0
    ):
        raise LocalAffordanceTrainingSourceError(
            "requested evidence key, family, replicate, or task order differs from lock"
        )
    return key, manifest, task_ids


def load_local_affordance_training_source(
    runtime: ScreeningRuntime,
    evidence_lock: Phase3EvidenceLock,
    *,
    family_id: str,
    replicate: int,
) -> TrainingDataEvidencePayloadBundle:
    """Load only the requested fold/replicate's locked optimum payload.

    The returned exact ``TrainingDataEvidencePayloadBundle`` can be passed
    directly as ``source_bundle`` to ``build_local_affordance_training_view``.
    Runtime and lock validation happen before opening the single evidence
    namespace; all reads below are relative to retained run/data descriptors.
    """

    if type(family_id) is not str or family_id not in FAMILIES:
        raise LocalAffordanceTrainingSourceError("family is outside the frozen development scope")
    if type(replicate) is not int or replicate not in REPLICATES:
        raise LocalAffordanceTrainingSourceError("replicate is outside the frozen development scope")
    row = _lock_row(evidence_lock, family_id=family_id, replicate=replicate)
    fold = _runtime_fold(runtime, family_id)
    try:
        # Metadata-only runtimes deliberately cannot pass the historical
        # payload-inventory recheck. Dispatch only on the exact validated
        # runtime type and the presence of its frozen snapshot bytes.
        if type(runtime) is not ScreeningRuntime:
            raise TypeError("training source requires an exact ScreeningRuntime")
        if runtime.metadata_only_snapshot_bytes is not None:
            recheck_screening_runtime_metadata_only(runtime)
        else:
            recheck_screening_runtime_readonly(runtime)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise LocalAffordanceTrainingSourceError(
            "training source requires a freshly rechecked development runtime"
        ) from exc

    key, manifest, task_ids = _typed_lineage(
        row, family_id=family_id, replicate=replicate
    )
    config = fold.config
    store = fold.store
    parameters = getattr(config, "parameters", None)
    development_tasks = tuple(getattr(getattr(config, "split", None), "development_tasks", ()))
    runtime_task_ids = tuple(getattr(task, "task_id", None) for task in development_tasks)
    pinned_run = getattr(store, "_open_pinned_run", None)
    if (
        getattr(fold, "family_id", None) != family_id
        or not isinstance(parameters, Mapping)
        or parameters.get("fold_id") != key.fold_id
        or runtime_task_ids != task_ids
        or getattr(store, "run_id", None) != row.get("child_run_id")
        or not callable(pinned_run)
    ):
        raise LocalAffordanceTrainingSourceError(
            "requested evidence row is not owned by the matching pinned runtime fold"
        )

    try:
        with pinned_run() as run_fd:
            with open_training_data_reader(run_fd) as reader:
                bundle = load_training_data_evidence_payload_bundle_from_at(
                    reader,
                    manifest.evidence_id,
                    expected_key=key,
                )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise LocalAffordanceTrainingSourceError(
            "descriptor-pinned optimum evidence read failed"
        ) from exc

    manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
    payload_sha = hashlib.sha256(bundle.payload_bytes).hexdigest()
    if (
        type(bundle) is not TrainingDataEvidencePayloadBundle
        or bundle.manifest != manifest
        or bundle.manifest_bytes != manifest_bytes
        or hashlib.sha256(bundle.manifest_bytes).hexdigest()
        != row.get("canonical_manifest_bytes_sha256")
        or payload_sha != manifest.payload_sha256
        or payload_sha != row.get("payload_sha256")
        or len(bundle.payload_bytes) != manifest.payload_bytes
        or len(bundle.payload_bytes) != row.get("payload_bytes")
        or tuple(sample.task_id for sample in bundle.payload.samples) != task_ids
    ):
        raise LocalAffordanceTrainingSourceError(
            "descriptor-read optimum bytes or task order differ from frozen evidence lock"
        )
    return bundle


__all__ = [
    "LocalAffordanceTrainingSourceError",
    "load_local_affordance_training_source",
]
