"""Pure same-data optimum-imitation views for the Phase 3 local-affordance rung.

This module accepts already-materialized typed values only.  It performs no path
access, environment construction, training, search, replay, or evaluation. The
opaque training capability checks raw-artifact identities against the typed
40-task LOFO fold manifest and releases only identity-free probe evidence before
learner-facing :class:`DecisionExample`s are constructed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch

from levelup.experiments.milestone6_phase3_evidence import (
    EvidenceLockError,
    Phase3EvidenceLock,
    require_phase3_evidence_lock,
)
from levelup.experiments.milestone6_phase3_local_affordance_capabilities import (
    LocalAffordanceCapabilityError,
    TrainingFoldProbeCapability,
)
from levelup.experiments.milestone6_phase3_local_affordance_models import (
    LocalAffordanceTrainingView,
    _seal_local_affordance_training_view,
)
from levelup.experiments.milestone6_phase3_local_affordance_plan import (
    CONDITIONS,
    LocalAffordanceView,
)
from levelup.experiments.milestone6_phase3_local_affordance_raw_store import (
    FAMILY_ORDER,
    TrainingFoldManifest,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import (
    AffordanceTableRecord,
    ObservableTraceRecord,
    TrainingDataEvidenceKey,
    TrainingDataEvidenceManifest,
    TrainingDataEvidencePayloadBundle,
    TrainingDataPayload,
)
from levelup.learning.state_conditioned import (
    AffordanceTable,
    DecisionExample,
    ObservableState,
    ObservableTrace,
    ObservedTransition,
    TaskLocalAffordanceEvidence,
    candidate_tensor,
    global_listwise_optimum_examples,
    local_affordance_candidate_tensor,
    pooled_candidate_tensor,
    state_availability_candidate_tensor,
)

CONDITION_B2 = "B2-global-listwise-optimum"
CONDITION_S = "S-state-availability-listwise-optimum"
CONDITION_P = "P-state-availability-alias-pooled-outcome-listwise-optimum"
CONDITION_L = "L-state-availability-local-outcome-listwise-optimum"
CONDITION_IDS = (CONDITION_B2, CONDITION_S, CONDITION_P, CONDITION_L)
TASKS_PER_FOLD = 40
TASKS_PER_FAMILY = 8


class LocalAffordanceTrainingViewError(ValueError):
    """Raised when same-data training-view construction fails closed."""


def _float32_bytes(values: Sequence[float]) -> bytes:
    return torch.tensor(tuple(values), dtype=torch.float32).contiguous().view(torch.int32).numpy().tobytes()


def _require_affordance_parity(
    source: AffordanceTable,
    raw: AffordanceTable,
    *,
    task_id: str,
) -> None:
    if source.sample_counts != raw.sample_counts or set(source.features) != set(raw.features):
        raise LocalAffordanceTrainingViewError(
            f"raw probe pooled table differs from optimum training evidence for {task_id}"
        )
    for alias in sorted(source.features):
        if _float32_bytes(source.features[alias]) != _float32_bytes(raw.features[alias]):
            raise LocalAffordanceTrainingViewError(
                f"raw probe pooled table differs from optimum training evidence for {task_id}"
            )


def _observable_state(record) -> ObservableState:
    return ObservableState(
        progress_fraction=record.progress_fraction,
        remaining_fraction=record.remaining_fraction,
        elapsed_per_target=record.elapsed_per_target,
        resource_fraction=record.resource_fraction,
        pressure_fraction=record.pressure_fraction,
        available_aliases=record.available_aliases,
    )


def _validated_locked_training_samples(
    *,
    fold: TrainingFoldManifest,
    evidence_lock: Phase3EvidenceLock,
    source_bundle: TrainingDataEvidencePayloadBundle,
) -> tuple[tuple[tuple[str, ObservableTrace, AffordanceTable], ...], tuple[str, ...]]:
    try:
        lock = require_phase3_evidence_lock(evidence_lock)
    except (EvidenceLockError, TypeError, ValueError) as exc:
        raise LocalAffordanceTrainingViewError(
            "optimum source requires a validator-issued Phase 3 evidence lock"
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
        raise LocalAffordanceTrainingViewError("optimum source lock is not exact development-only authority")
    rows = body.get("evidence_artifacts")
    if type(rows) is not list or len(rows) != 30:
        raise LocalAffordanceTrainingViewError("optimum source lock has an incomplete evidence matrix")
    expected_lock_identities = {
        (family, replicate) for family in FAMILY_ORDER for replicate in range(5)
    }
    observed_lock_identities = tuple(
        (row.get("family_id"), row.get("replicate"))
        for row in rows
        if isinstance(row, dict)
    )
    if (
        len(observed_lock_identities) != 30
        or len(set(observed_lock_identities)) != 30
        or set(observed_lock_identities) != expected_lock_identities
    ):
        raise LocalAffordanceTrainingViewError("optimum source lock family/replicate matrix drifted")
    matches = tuple(
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("family_id") == fold.heldout_family
        and row.get("replicate") == fold.replicate
    )
    if len(matches) != 1:
        raise LocalAffordanceTrainingViewError("optimum source lock row is missing or duplicated")
    locked = matches[0]
    if type(source_bundle) is not TrainingDataEvidencePayloadBundle:
        raise LocalAffordanceTrainingViewError("typed training evidence payload bundle is required")
    try:
        manifest = TrainingDataEvidenceManifest.model_validate(
            source_bundle.manifest.model_dump(mode="json")
        )
        payload = TrainingDataPayload.model_validate_json(source_bundle.payload_bytes)
        bundle_payload = TrainingDataPayload.model_validate(
            source_bundle.payload.model_dump(mode="json")
        )
        locked_key = TrainingDataEvidenceKey.model_validate(locked["evidence_key"])
        locked_manifest = TrainingDataEvidenceManifest.model_validate(locked["evidence_manifest"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise LocalAffordanceTrainingViewError("typed optimum evidence bundle or lock row is invalid") from exc
    manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
    payload_bytes = canonical_json_bytes(payload.model_dump(mode="json"))
    payload_sha256 = hashlib.sha256(source_bundle.payload_bytes).hexdigest()
    if (
        manifest != source_bundle.manifest
        or payload != bundle_payload
        or source_bundle.manifest_bytes != manifest_bytes
        or source_bundle.payload_bytes != payload_bytes
        or manifest != locked_manifest
        or manifest.key != locked_key
        or locked.get("evidence_key_id") != locked_key.key_id
        or locked.get("evidence_manifest_key_id") != locked_key.key_id
        or locked.get("evidence_id") != manifest.evidence_id
        or locked.get("family_id") != locked_key.heldout_family_id
        or locked.get("fold_id") != locked_key.fold_id
        or locked.get("replicate") != locked_key.replicate
        or locked.get("payload_sha256") != manifest.payload_sha256
        or locked.get("payload_bytes") != manifest.payload_bytes
        or locked.get("canonical_manifest_bytes_sha256")
        != hashlib.sha256(source_bundle.manifest_bytes).hexdigest()
        or payload_sha256 != manifest.payload_sha256
        or len(source_bundle.payload_bytes) != manifest.payload_bytes
    ):
        raise LocalAffordanceTrainingViewError(
            "training payload bytes/manifest differ from the frozen Phase 3 evidence lock"
        )
    expected_task_ids = tuple(ref.task_id for ref in fold.task_references)
    key = manifest.key
    if (
        key.heldout_family_id != fold.heldout_family
        or key.fold_id != locked.get("fold_id")
        or key.replicate != fold.replicate
        or key.ordered_training_task_ids != expected_task_ids
        or manifest.sample_task_ids != expected_task_ids
        or tuple(sample.task_id for sample in payload.samples) != expected_task_ids
        or tuple(locked.get("ordered_training_task_ids", ())) != expected_task_ids
    ):
        raise LocalAffordanceTrainingViewError(
            "typed optimum traces are not the exact 40-task fold evidence"
        )
    samples: list[tuple[str, ObservableTrace, AffordanceTable]] = []
    for sample in payload.samples:
        try:
            trace_record = ObservableTraceRecord.model_validate(
                sample.trace.model_dump(mode="json")
            )
            affordance_record = AffordanceTableRecord.model_validate(
                sample.affordances.model_dump(mode="json")
            )
            trace = ObservableTrace(
                tuple(
                    ObservedTransition(
                        _observable_state(transition.before),
                        transition.action_alias,
                        _observable_state(transition.after),
                        transition.completed,
                    )
                    for transition in trace_record.transitions
                )
            )
            table = AffordanceTable(
                dict(affordance_record.features), dict(affordance_record.sample_counts)
            )
        except (TypeError, ValueError) as exc:
            raise LocalAffordanceTrainingViewError("optimum source sample is malformed") from exc
        samples.append((sample.task_id, trace, table))
    return tuple(samples), expected_task_ids


def _validate_inputs(
    *,
    view: LocalAffordanceView,
    fold: TrainingFoldManifest,
    evidence_lock: Phase3EvidenceLock,
    source_bundle: TrainingDataEvidencePayloadBundle,
    probe_capability: TrainingFoldProbeCapability,
) -> tuple[
    tuple[tuple[str, ObservableTrace, AffordanceTable], ...],
    tuple[TaskLocalAffordanceEvidence, ...],
]:
    if type(view) is not LocalAffordanceView or view.condition_id not in CONDITIONS:
        raise LocalAffordanceTrainingViewError("unsupported local-affordance condition")
    if type(fold) is not TrainingFoldManifest:
        raise LocalAffordanceTrainingViewError("typed training fold manifest is required")
    try:
        fold = TrainingFoldManifest.model_validate(fold.model_dump(mode="json"))
    except (TypeError, ValueError) as exc:
        raise LocalAffordanceTrainingViewError("training fold manifest is invalid") from exc
    if fold.fold_id != fold.heldout_family or fold.heldout_family not in FAMILY_ORDER:
        raise LocalAffordanceTrainingViewError("training fold is outside the frozen LOFO scope")
    if (
        view.fold_id != fold.fold_id
        or view.heldout_family != fold.heldout_family
        or view.replicate != fold.replicate
        or view.training_task_ids != tuple(ref.task_id for ref in fold.task_references)
    ):
        raise LocalAffordanceTrainingViewError(
            "typed view identity differs from its exact raw-evidence fold"
        )

    samples, expected_task_ids = _validated_locked_training_samples(
        fold=fold,
        evidence_lock=evidence_lock,
        source_bundle=source_bundle,
    )
    if len(samples) != TASKS_PER_FOLD:
        raise LocalAffordanceTrainingViewError("exactly 40 optimum training tasks are required")
    if any(type(row) is not tuple or len(row) != 3 for row in samples):
        raise LocalAffordanceTrainingViewError("training samples must be typed task/trace/table rows")
    if any(
        type(task_id) is not str
        or not task_id
        or any(ord(char) < 32 for char in task_id)
        or type(trace) is not ObservableTrace
        or not trace.transitions
        or type(table) is not AffordanceTable
        for task_id, trace, table in samples
    ):
        raise LocalAffordanceTrainingViewError("training samples must be typed task/trace/table rows")
    if type(probe_capability) is not TrainingFoldProbeCapability:
        raise LocalAffordanceTrainingViewError("opaque typed 40-task probe capability is required")

    references = fold.task_references
    sample_ids = tuple(row[0] for row in samples)
    expected_task_ids = tuple(ref.task_id for ref in references)
    if sample_ids != expected_task_ids:
        raise LocalAffordanceTrainingViewError("optimum task sample order differs from locked source")
    # Recheck the complement matrix locally as a guard against future manifest-schema drift.
    expected_families = set(FAMILY_ORDER) - {fold.heldout_family}
    counts = {family: 0 for family in expected_families}
    for ref in references:
        if ref.family_id == fold.heldout_family or ref.family_id not in counts:
            raise LocalAffordanceTrainingViewError("training evidence includes heldout/foreign family")
        if ref.replicate != fold.replicate:
            raise LocalAffordanceTrainingViewError("training evidence replicate differs from fold")
        counts[ref.family_id] += 1
    if set(counts) != expected_families or any(count != TASKS_PER_FAMILY for count in counts.values()):
        raise LocalAffordanceTrainingViewError("evidence is not the exact five-family LOFO complement")
    try:
        evidence_rows = probe_capability.consume_for(expected_task_ids)
    except LocalAffordanceCapabilityError as exc:
        raise LocalAffordanceTrainingViewError(
            "probe capability does not authorize this exact ordered 40-task fold"
        ) from exc
    if type(evidence_rows) is not tuple or len(evidence_rows) != TASKS_PER_FOLD:
        raise LocalAffordanceTrainingViewError("probe capability did not release exactly 40 task rows")
    if any(type(evidence) is not TaskLocalAffordanceEvidence for evidence in evidence_rows):
        raise LocalAffordanceTrainingViewError("probe capability released untyped task evidence")
    for (task_id, _trace, table), evidence in zip(samples, evidence_rows, strict=True):
        raw_table = evidence.pooled_affordances
        _require_affordance_parity(table, raw_table, task_id=task_id)
    return samples, evidence_rows


def build_local_affordance_training_view(
    *,
    view: LocalAffordanceView,
    fold: TrainingFoldManifest,
    evidence_lock: Phase3EvidenceLock,
    source_bundle: TrainingDataEvidencePayloadBundle,
    probe_capability: TrainingFoldProbeCapability,
) -> LocalAffordanceTrainingView:
    """Build one same-data B2/S/P/L optimum-imitation view.

    The source bundle must match a validator-issued evidence-lock row and be in
    the canonical task order declared by ``fold``. The capability releases only
    the matching identity-free 40-task probe evidence.
    The only returned values are action-alias-ordered feature tensors and optimum
    action indices; task and probe metadata never enter learner examples.
    """

    samples, evidence_rows = _validate_inputs(
        view=view,
        fold=fold,
        evidence_lock=evidence_lock,
        source_bundle=source_bundle,
        probe_capability=probe_capability,
    )
    condition_id = view.condition_id
    output: list[DecisionExample] = []

    # B2 is the frozen 49-input global listwise baseline, not the 54-input
    # state-conditioned scorer.  Its optimum examples and order are built by
    # the same implementation as the historical B2 condition.
    if condition_id == CONDITION_B2:
        examples = global_listwise_optimum_examples(
            tuple((trace, table) for _task_id, trace, table in samples)
        )
        return _seal_local_affordance_training_view(
            view=view,
            examples=examples,
            task_example_counts=tuple(
                len(trace.transitions) for _task_id, trace, _table in samples
            ),
        )

    # Build in the same task/trace/decision order as the canonical B2 examples.
    for (_task_id, trace, table), local_evidence in zip(samples, evidence_rows, strict=True):
        for transition in trace.transitions:
            aliases, _, _ = candidate_tensor(transition.before, table)
            try:
                selected = aliases.index(transition.action_alias)
            except ValueError as exc:
                raise LocalAffordanceTrainingViewError(
                    "optimum action is absent from its observable candidate list"
                ) from exc
            if condition_id == CONDITION_S:
                candidate_aliases, features, _ = state_availability_candidate_tensor(
                    transition.before, local_evidence
                )
            elif condition_id == CONDITION_P:
                candidate_aliases, features, _ = pooled_candidate_tensor(
                    transition.before, local_evidence
                )
            else:
                candidate_aliases, features, _ = local_affordance_candidate_tensor(
                    transition.before, local_evidence
                )
            if candidate_aliases != aliases:
                raise LocalAffordanceTrainingViewError(
                    "candidate alias order differs across same-data conditions"
                )
            output.append(DecisionExample(features, selected))

    if not output:
        raise LocalAffordanceTrainingViewError("optimum training view is empty or incomplete")
    return _seal_local_affordance_training_view(
        view=view,
        examples=tuple(output),
        task_example_counts=tuple(len(trace.transitions) for _task_id, trace, _table in samples),
    )


__all__ = [
    "CONDITION_B2",
    "CONDITION_IDS",
    "CONDITION_L",
    "CONDITION_P",
    "CONDITION_S",
    "LocalAffordanceTrainingViewError",
    "build_local_affordance_training_view",
]
