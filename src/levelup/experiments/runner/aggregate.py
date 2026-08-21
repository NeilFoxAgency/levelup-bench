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
    PhaseAccounting,
    ResourceAccounting,
    SharedArtifactInventory,
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


def _add_phase(left: PhaseAccounting, right: PhaseAccounting) -> PhaseAccounting:
    return PhaseAccounting(
        calls=left.calls + right.calls,
        episodes=left.episodes + right.episodes,
        actions=left.actions + right.actions,
        environment_steps=left.environment_steps + right.environment_steps,
        resets=left.resets + right.resets,
        forward_passes=left.forward_passes + right.forward_passes,
        optimizer_steps=left.optimizer_steps + right.optimizer_steps,
        nodes_expanded=left.nodes_expanded + right.nodes_expanded,
        wall_seconds=left.wall_seconds + right.wall_seconds,
    )


def _add_cost(left: ResourceAccounting, right: ResourceAccounting) -> ResourceAccounting:
    return ResourceAccounting(
        setup=_add_phase(left.setup, right.setup),
        probes=_add_phase(left.probes, right.probes),
        training=_add_phase(left.training, right.training),
        search=_add_phase(left.search, right.search),
        replay=_add_phase(left.replay, right.replay),
        evaluator=_add_phase(left.evaluator, right.evaluator),
        serialization=_add_phase(left.serialization, right.serialization),
    )


def _shared_summary(
    store: RunStore, records: tuple[UnitRecord, ...], *, strict: bool
) -> tuple[
    str | None,
    SharedArtifactInventory,
    dict[str, ResourceAccounting],
    bool,
]:
    plans = {item.key_id: item for item in store.expected_shared.artifacts}
    if not plans:
        return None, SharedArtifactInventory(), {}, True
    refs: dict[str, list[UnitRecord]] = defaultdict(list)
    for record in records:
        if record.shared_artifact is not None:
            refs[record.shared_artifact.key_id].append(record)
            store.validate_shared_reference(
                store.planned_unit(record.unit_id), record.shared_artifact
            )
    for key_id, consumers in refs.items():
        if key_id not in plans:
            raise ArtifactValidationError("unit references unplanned shared artifact")
        identities = {
            (record.shared_artifact.artifact_id, record.shared_artifact.cost_id)
            for record in consumers
            if record.shared_artifact is not None
        }
        if len(identities) != 1:
            raise ArtifactValidationError("shared artifact has conflicting references")
    cost_owner: dict[str, str] = {}
    artifact_owner: dict[str, str] = {}
    for key_id, consumers in refs.items():
        for record in consumers:
            if record.shared_artifact is None:
                continue
            previous = cost_owner.setdefault(record.shared_artifact.cost_id, key_id)
            if previous != key_id:
                raise ArtifactValidationError("one shared cost is claimed by multiple keys")
            previous_artifact = artifact_owner.setdefault(
                record.shared_artifact.artifact_id, key_id
            )
            if previous_artifact != key_id:
                raise ArtifactValidationError(
                    "one shared artifact is claimed by multiple keys"
                )
    missing_consumers = [
        item
        for item in store.expected_shared.artifacts
        if any(
            unit_id not in {record.unit_id for record in refs.get(item.key_id, ())}
            for unit_id in item.consumer_unit_ids
        )
    ]
    if strict and missing_consumers:
        raise IncompleteRunError("shared artifact consumers are incomplete")
    by_condition: dict[str, ResourceAccounting] = {}
    for key_id, consumers in refs.items():
        if not consumers:
            continue
        cost = store.load_shared_cost(key_id)
        owner = plans[key_id].owner_group_id or plans[key_id].owner_condition_id
        shared_cost = (
            cost.accounting.as_resource_accounting()
            if hasattr(cost.accounting, "as_resource_accounting")
            else cost.accounting
        )
        by_condition[owner] = _add_cost(
            by_condition.get(owner, ResourceAccounting()), shared_cost
        )
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "plan": store.expected_shared.model_dump(mode="json"),
                "refs": {
                    key: sorted(
                        (record.shared_artifact.artifact_id, record.shared_artifact.cost_id)
                        for record in values
                        if record.shared_artifact is not None
                    )
                    for key, values in sorted(refs.items())
                },
            }
        )
    ).hexdigest()
    inventory = SharedArtifactInventory(
        planned=len(plans),
        referenced=len(refs),
        complete=not missing_consumers,
    )
    return digest, inventory, by_condition, not missing_consumers


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
    shared_hash, shared_inventory, shared_costs, shared_complete = _shared_summary(
        store, records, strict=strict
    )

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
        complete=not missing and shared_complete,
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
        shared_artifacts_sha256=shared_hash,
        shared_inventory=shared_inventory,
        shared_accounting_by_owner_group=shared_costs,
    )
    if write:
        store.write_aggregate(artifact)
    return artifact
