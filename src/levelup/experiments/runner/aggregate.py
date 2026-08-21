"""Pure aggregation from validated raw unit files."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import (
    AggregateArtifact,
    AggregateSlice,
    AttemptRecord,
    Inventory,
    ResourceAccounting,
    SplitPhase,
    UnitRecord,
)
from levelup.experiments.runner.storage import ArtifactValidationError, RunStore


class IncompleteRunError(RuntimeError):
    """Raised when strict aggregation sees missing or nonterminal units."""


def _records_sha256(records: tuple[UnitRecord, ...]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.unit_id):
        digest.update(canonical_json_bytes(record.model_dump(mode="json")))
        digest.update(b"\n")
    return digest.hexdigest()


def _expected_sha256(store: RunStore) -> str:
    return hashlib.sha256(
        canonical_json_bytes(store.expected.model_dump(mode="json"))
    ).hexdigest()


def _provenance_sha256(store: RunStore) -> str:
    return hashlib.sha256(
        canonical_json_bytes(store.load_provenance().model_dump(mode="json"))
    ).hexdigest()


def _phase_totals(accounting: ResourceAccounting) -> tuple[int, int, int, int, int]:
    phases = (
        accounting.setup,
        accounting.probes,
        accounting.training,
        accounting.search,
        accounting.replay,
        accounting.evaluator,
        accounting.serialization,
    )
    return (
        sum(phase.actions for phase in phases),
        sum(phase.forward_passes for phase in phases),
        sum(phase.optimizer_steps for phase in phases),
        accounting.probes.actions,
        accounting.search.actions,
    )


def _slice(records: list[UnitRecord]) -> AggregateSlice:
    ordered = sorted(records, key=lambda record: record.unit_id)
    probe_actions = 0
    search_actions = 0
    replay_actions = 0
    forward_passes = 0
    optimizer_steps = 0
    for record in ordered:
        _, forwards, steps, probes, search = _phase_totals(record.accounting)
        probe_actions += probes
        search_actions += search
        replay_actions += record.accounting.replay.actions
        forward_passes += forwards
        optimizer_steps += steps
    return AggregateSlice(
        completed_units=len(ordered),
        valid_units=sum(record.outcome.valid for record in ordered),
        successful_units=sum(record.outcome.success for record in ordered),
        performance_values=tuple(
            record.outcome.performance_value
            for record in ordered
            if record.outcome.valid
            and record.outcome.completed
            and record.outcome.performance_value is not None
        ),
        probe_actions=probe_actions,
        search_actions=search_actions,
        replay_actions=replay_actions,
        forward_passes=forward_passes,
        optimizer_steps=optimizer_steps,
        wall_seconds=sum(record.elapsed_wall_seconds for record in ordered),
    )


def _paired_seed_audit(store: RunStore) -> bool:
    grouped: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    seed_values: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    expected_conditions = {
        phase: {
            condition.condition_id
            for condition in store.config.conditions
            if phase in condition.execution_phases
        }
        for phase in ("development", "validation", "final")
    }
    for planned in store.expected.units:
        group = (planned.key.phase, planned.key.task_id, planned.key.replicate)
        grouped[group].add(planned.key.condition_id)
        seed_values[group].add(planned.seeds.model_dump_json())
    return all(
        conditions == expected_conditions[group[0]] and len(seed_values[group]) == 1
        for group, conditions in grouped.items()
    )


def _attempt_inventory(attempts: tuple[AttemptRecord, ...]) -> tuple[int, int, int, int]:
    failed = [attempt for attempt in attempts if attempt.status == "failed"]
    interrupted = [attempt for attempt in attempts if attempt.status == "interrupted"]
    return (
        len({attempt.unit_id for attempt in failed}),
        len({attempt.unit_id for attempt in interrupted}),
        len(failed),
        len(interrupted),
    )


def aggregate_run(
    store: RunStore,
    *,
    strict: bool = False,
    write: bool = False,
) -> AggregateArtifact:
    """Aggregate validated records without importing or executing an environment."""

    records = store.completed_records()
    attempts = store.attempt_records()
    completed_ids = {record.unit_id for record in records}
    missing = [unit for unit in store.expected.units if unit.unit_id not in completed_ids]
    paired = _paired_seed_audit(store)
    if not paired:
        raise ArtifactValidationError("expected-unit matrix failed paired seed audit")
    if strict and missing:
        raise IncompleteRunError(f"run has {len(missing)} incomplete units")

    failed_units, interrupted_units, failed_attempts, interrupted_attempts = (
        _attempt_inventory(attempts)
    )
    by_phase_condition: dict[tuple[SplitPhase, str], list[UnitRecord]] = defaultdict(list)
    by_phase_family: dict[tuple[SplitPhase, str], list[UnitRecord]] = defaultdict(list)
    for record in records:
        by_phase_condition[(record.key.phase, record.key.condition_id)].append(record)
        by_phase_family[(record.key.phase, record.key.family_id)].append(record)

    phases = tuple(
        phase
        for phase in ("development", "validation", "final")
        if any(unit.key.phase == phase for unit in store.expected.units)
    )
    family_ids_by_phase = {
        phase: sorted(
            {unit.key.family_id for unit in store.expected.units if unit.key.phase == phase}
        )
        for phase in phases
    }
    condition_ids_by_phase = {
        phase: sorted(
            condition.condition_id
            for condition in store.config.conditions
            if phase in condition.execution_phases
        )
        for phase in phases
    }
    started_at = min((record.started_at_utc for record in records), default=None)
    finished_at = max((record.finished_at_utc for record in records), default=None)
    observed_span = (
        (finished_at - started_at).total_seconds()
        if started_at is not None and finished_at is not None
        else 0.0
    )
    artifact = AggregateArtifact(
        run_id=store.run_id,
        config_sha256=store.config_sha256,
        expected_units_sha256=_expected_sha256(store),
        completed_units_sha256=_records_sha256(records),
        provenance_sha256=_provenance_sha256(store),
        run_started_at_utc=started_at,
        run_finished_at_utc=finished_at,
        observed_span_seconds=observed_span,
        complete=not missing,
        paired_seed_audit_passed=paired,
        inventory=Inventory(
            expected=len(store.expected.units),
            completed=len(records),
            missing=len(missing),
            units_with_failed_attempts=failed_units,
            units_with_interrupted_attempts=interrupted_units,
            failed_attempts=failed_attempts,
            interrupted_attempts=interrupted_attempts,
        ),
        by_phase_condition={
            phase: {
                condition_id: _slice(by_phase_condition[(phase, condition_id)])
                for condition_id in condition_ids_by_phase[phase]
            }
            for phase in phases
        },
        by_phase_family={
            phase: {
                family_id: _slice(by_phase_family[(phase, family_id)])
                for family_id in family_ids_by_phase[phase]
            }
            for phase in phases
        },
    )
    if write:
        store.write_aggregate(artifact)
    return artifact
