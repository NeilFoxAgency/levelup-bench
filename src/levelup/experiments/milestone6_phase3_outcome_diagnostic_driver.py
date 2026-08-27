"""Resumable, development-only execution of the Phase 3 outcome diagnostic.

The driver is intentionally a bookkeeping boundary.  It is allowed to invoke
the one-unit executor, but it never reads or reduces outcome values and never
has a final-family path.  Result stores are activated only after the exact
development matrix and the readiness lease have been checked.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, TypedDict

import torch
from pydantic import ValidationError

from levelup.experiments.milestone6_phase3_outcome_diagnostic_execution import (
    OutcomeDiagnosticExecutionContext,
    execute_outcome_diagnostic_unit,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_execution_models import (
    OutcomeDiagnosticExecutionModelError,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    OutcomePlannedUnit,
    ValidatedOutcomePlan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    CONDITIONS,
    FAMILIES,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_readiness import (
    OutcomeDiagnosticModelReadinessSnapshot,
    OutcomeDiagnosticReadinessError,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_result_store import (
    EXPECTED_FAMILY_UNIT_COUNT,
    EXPECTED_NAMESPACE_UNIT_COUNT,
    EXPECTED_TOTAL_UNIT_COUNT,
    OutcomeDiagnosticExpectedPlan,
    OutcomeDiagnosticResultStore,
    OutcomeDiagnosticResultStoreError,
    OutcomeDiagnosticResumeBaseline,
    activate_outcome_diagnostic_result_stores,
    build_outcome_diagnostic_expected_plan,
    prepare_outcome_diagnostic_result_stores,
)
from levelup.experiments.runner.provenance import utc_now
from levelup.experiments.runner.records import (
    AttemptRecord,
    UnitKey,
    UnitPayload,
    UnitRecord,
    UnitSeeds,
)

_STAGE_EXECUTION = "execution"
_STAGE_PAYLOAD = "payload-validation"
_STAGE_RECORD = "record-construction"
_STAGE_PUBLICATION = "record-publication"


class OutcomeDiagnosticDriverError(RuntimeError):
    """Raised when the frozen outcome diagnostic cannot execute safely."""


class OutcomeDiagnosticRunSummary(TypedDict):
    """Noncomparative driver result; no performance or outcome values appear."""

    validate_only: bool
    plan_id: str
    protocol_sha256: str
    model_authority_sha256: str
    family_order: tuple[str, ...]
    condition_order: tuple[str, ...]
    expected_total: int
    completed: int
    skipped: int
    failed: int
    interrupted: int
    complete: bool


def _fail(message: str, exc: BaseException | None = None) -> None:
    if exc is None:
        raise OutcomeDiagnosticDriverError(message)
    raise OutcomeDiagnosticDriverError(message) from exc


def _authority_digest(authority: object) -> str:
    value = getattr(authority, "authority_sha256", None)
    if not isinstance(value, str):
        value = getattr(authority, "expected_authority_sha256", None)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail("outcome model authority has no canonical digest")
    if (
        getattr(authority, "expected_authority_sha256", value) != value
        or bool(getattr(authority, "final", False))
        or bool(getattr(authority, "final_family_access", False))
        or not bool(getattr(authority, "development_only", False))
    ):
        _fail("outcome model authority is not development-only")
    return value


def _validate_exact_matrix(
    context: OutcomeDiagnosticExecutionContext,
) -> OutcomeDiagnosticExpectedPlan:
    """Reassert the six-family/2-condition/5,760-unit universe at execution."""

    if type(context) is not OutcomeDiagnosticExecutionContext:
        _fail("canonical outcome execution context is required")
    try:
        context.authority_cache.require_active()
    except Exception as exc:
        if isinstance(exc, OutcomeDiagnosticDriverError):
            raise
        _fail("outcome execution authority cache is not active", exc)
    if type(context.plan) is not ValidatedOutcomePlan:
        _fail("validated outcome plan is required")
    plan = context.plan.plan
    if (
        plan.final_family_access
        or plan.family_order != FAMILIES
        or plan.condition_ids != CONDITIONS
        or len(plan.units) != EXPECTED_TOTAL_UNIT_COUNT
        or len({unit.unit_id for unit in plan.units}) != EXPECTED_TOTAL_UNIT_COUNT
    ):
        _fail("outcome diagnostic plan is not the exact development matrix")
    if any(
        unit.final_family_access
        or unit.heldout_family not in FAMILIES
        or unit.condition_id not in CONDITIONS
        for unit in plan.units
    ):
        _fail("outcome diagnostic plan contains a final or foreign unit")
    expected = build_outcome_diagnostic_expected_plan(context.plan, context.protocol)
    if (
        type(expected) is not OutcomeDiagnosticExpectedPlan
        or expected.final_family_access
        or expected.family_order != FAMILIES
        or expected.condition_order != CONDITIONS
        or len(expected.units) != EXPECTED_TOTAL_UNIT_COUNT
        or len({unit.unit_id for unit in expected.units}) != EXPECTED_TOTAL_UNIT_COUNT
    ):
        _fail("outcome expected plan is not the exact development matrix")
    if tuple(store.family_id for store in expected.stores) != FAMILIES:
        _fail("outcome expected stores are not in canonical family order")
    for store in expected.stores:
        if store.final_family_access or len(store.units) != EXPECTED_FAMILY_UNIT_COUNT:
            _fail("outcome family partition is incomplete")
        if tuple(namespace.condition_id for namespace in store.namespaces) != CONDITIONS:
            _fail("outcome condition namespace order drifted")
        if any(len(namespace.units) != EXPECTED_NAMESPACE_UNIT_COUNT for namespace in store.namespaces):
            _fail("outcome condition namespace cardinality drifted")
    return expected


def _expected_key(planned: OutcomePlannedUnit) -> UnitKey:
    return UnitKey(
        phase="validation",
        condition_id=f"{planned.condition_id}--{planned.tuple_id}",
        family_id=planned.heldout_family,
        task_id=planned.task_id,
        task_index=planned.task_index,
        replicate=planned.replicate,
    )


def _expected_seeds(planned: OutcomePlannedUnit) -> UnitSeeds:
    return UnitSeeds(
        model_seed=planned.model_seed,
        environment_seed=planned.environment_seed,
        probe_seed=planned.probe_seed,
        search_seed=planned.search_seed,
        data_order_seed=planned.data_order_seed,
    )


def _require_resume_baseline(
    snapshot: OutcomeDiagnosticModelReadinessSnapshot,
    expected: OutcomeDiagnosticExpectedPlan,
) -> OutcomeDiagnosticResumeBaseline:
    base = snapshot.base
    baseline = base.resume_baseline
    expected_baseline = base.resume_expected_plan
    if type(baseline) is not OutcomeDiagnosticResumeBaseline:
        _fail("prepared or activated output lacks a typed resume baseline")
    if type(expected_baseline) is not OutcomeDiagnosticExpectedPlan or expected_baseline != expected:
        _fail("resume expected plan differs from the current frozen matrix")
    if baseline.output_state != base.output_state:
        _fail("resume baseline output state differs from readiness")
    if baseline.output_root != snapshot.output_root:
        _fail("resume baseline output root differs from readiness")
    stores = tuple(baseline.stores)
    if (
        len(stores) != len(FAMILIES)
        or tuple(store.family_id for store in stores) != FAMILIES
        or any(type(store) is not OutcomeDiagnosticResultStore for store in stores)
        or tuple(store.spec for store in stores) != expected.stores
    ):
        _fail("resume baseline stores differ from the frozen expected plan")
    return baseline


def _validate_stores_for_readiness(
    snapshot: OutcomeDiagnosticModelReadinessSnapshot,
    expected: OutcomeDiagnosticExpectedPlan,
    lease: Any,
    *,
    validate_only: bool,
) -> tuple[OutcomeDiagnosticResultStore, ...] | None:
    state = snapshot.base.output_state
    if state == "empty":
        if snapshot.base.resume_baseline is not None or snapshot.base.resume_expected_plan is not None:
            _fail("empty output readiness unexpectedly contains a resume baseline")
        if validate_only:
            return None
        try:
            stores = prepare_outcome_diagnostic_result_stores(lease, snapshot.protocol, snapshot.execution_models.validated_plan)
        except (OutcomeDiagnosticResultStoreError, OSError, RuntimeError, TypeError, ValueError) as exc:
            _fail("outcome diagnostic result stores could not be prepared", exc)
        return tuple(stores)
    if state not in ("prepared", "activated"):
        _fail("outcome readiness has an invalid output state")
    baseline = _require_resume_baseline(snapshot, expected)
    stores = tuple(baseline.stores)
    # Prepared stores are inert and can be reloaded for a descriptor-relative
    # metadata check.  Activated stores necessarily contain records, so their
    # typed baseline (captured without parsing outcome values) is the only
    # admissible reconstruction.
    if state == "prepared":
        try:
            from levelup.experiments.milestone6_phase3_outcome_diagnostic_result_store import (
                load_outcome_diagnostic_result_stores,
            )

            loaded = tuple(
                load_outcome_diagnostic_result_stores(
                    lease,
                    snapshot.protocol,
                    snapshot.execution_models.validated_plan,
                )
            )
        except (OutcomeDiagnosticResultStoreError, OSError, RuntimeError, TypeError, ValueError) as exc:
            _fail("prepared outcome diagnostic stores could not be validated", exc)
        if tuple(store.spec for store in loaded) != tuple(store.spec for store in stores):
            _fail("validated resume stores differ from the readiness baseline")
    return stores


def _enforce_cpu_single_thread() -> None:
    try:
        torch.set_num_threads(1)
        if torch.get_num_threads() != 1:
            raise RuntimeError("torch CPU thread count is not one")
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError("torch CPU interop thread count is not one")
    except Exception as exc:
        _fail("outcome diagnostic requires one CPU torch thread", exc)


def _retryable(exception: BaseException) -> bool:
    """Authority/schema/model errors are permanent; unknown runtime errors retry."""

    return not isinstance(
        exception,
        (
            OutcomeDiagnosticReadinessError,
            OutcomeDiagnosticResultStoreError,
            OutcomeDiagnosticExecutionModelError,
            ValidationError,
            ValueError,
            TypeError,
        ),
    )


def _attempt(
    family: Any,
    planned: OutcomePlannedUnit,
    exception: BaseException,
    *,
    attempt_number: int,
    stage: str,
    retryable: bool,
    started_at: datetime,
    elapsed: float,
    status: str = "failed",
) -> None:
    if status not in {"failed", "interrupted"}:
        _fail("invalid outcome diagnostic attempt status")
    try:
        record = AttemptRecord(
            run_id=family.run_id,
            config_sha256=family.config_sha256,
            unit_id=planned.unit_id,
            attempt=attempt_number,
            key=_expected_key(planned),
            seeds=_expected_seeds(planned),
            status=status,  # type: ignore[arg-type]
            stage=stage,
            exception_type=type(exception).__name__,
            sanitized_message=f"{stage} raised {type(exception).__name__}",
            retryable=retryable,
            started_at_utc=started_at,
            finished_at_utc=utc_now(),
            elapsed_wall_seconds=max(0.0, elapsed),
        )
        published = family.write_attempt(record)
        if not isinstance(published, bool):
            _fail("outcome diagnostic attempt publication did not return a typed result")
        if published is False:
            # An exact duplicate is safe, but a runtime facade must still have
            # accepted the canonical record.  It will reject conflicts itself.
            return
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _fail("outcome diagnostic failure attempt could not be durably recorded", exc)


def _inventory(batch: Any, expected: OutcomeDiagnosticExpectedPlan) -> tuple[dict[str, Any], dict[str, Any]]:
    stores = tuple(batch.stores)
    if len(stores) != len(FAMILIES) or tuple(store.family_id for store in stores) != FAMILIES:
        _fail("activated diagnostic stores are not in canonical family order")
    completed: dict[str, set[str]] = {}
    attempts: dict[str, AttemptRecord] = {}
    expected_ids = {unit.unit_id for unit in expected.units}
    if len(expected_ids) != EXPECTED_TOTAL_UNIT_COUNT:
        _fail("expected diagnostic unit IDs are not unique")
    for store in stores:
        family_ids = {unit.unit_id for unit in expected.store_for_family(store.family_id).units}
        values = set(store.completed_unit_ids())
        if not values <= family_ids:
            _fail("completed diagnostic inventory contains a foreign unit")
        completed[store.family_id] = values
        for attempt in store.attempt_records():
            if attempt.unit_id not in family_ids:
                _fail("diagnostic attempt inventory contains a foreign unit")
            prior = attempts.get(attempt.unit_id)
            if prior is None or attempt.attempt > prior.attempt:
                attempts[attempt.unit_id] = attempt
    return completed, attempts


def _execute_loop(
    batch: Any,
    context: OutcomeDiagnosticExecutionContext,
    expected: OutcomeDiagnosticExpectedPlan,
) -> dict[str, int | bool]:
    completed_by_family, latest_attempt = _inventory(batch, expected)
    family_by_id = {store.family_id: store for store in tuple(batch.stores)}
    counts: dict[str, int | bool] = {
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "interrupted": 0,
    }
    _enforce_cpu_single_thread()
    for planned in expected.units:
        family = family_by_id.get(planned.heldout_family)
        if family is None:
            _fail("planned diagnostic unit has no activated family store")
        if planned.unit_id in completed_by_family[planned.heldout_family]:
            counts["skipped"] = int(counts["skipped"]) + 1
            continue
        prior = latest_attempt.get(planned.unit_id)
        if prior is not None and not prior.retryable:
            _fail("non-retryable diagnostic attempt leaves an incomplete unit")
        attempt_number = 1 if prior is None else prior.attempt + 1
        started_at = utc_now()
        started = time.perf_counter()
        stage = _STAGE_EXECUTION
        try:
            raw_payload = execute_outcome_diagnostic_unit(context, planned)
            stage = _STAGE_PAYLOAD
            payload_input = (
                raw_payload.model_dump(mode="json", warnings=False)
                if isinstance(raw_payload, UnitPayload)
                else raw_payload
            )
            payload = UnitPayload.model_validate(payload_input)
            stage = _STAGE_RECORD
            record = UnitRecord(
                run_id=family.run_id,
                config_sha256=family.config_sha256,
                unit_id=planned.unit_id,
                key=_expected_key(planned),
                seeds=_expected_seeds(planned),
                exposure_manifest_sha256=planned.exposure_manifest_sha256,
                started_at_utc=started_at,
                finished_at_utc=utc_now(),
                elapsed_wall_seconds=max(0.0, time.perf_counter() - started),
                outcome=payload.outcome,
                accounting=payload.accounting,
                shared_artifact=payload.shared_artifact,
                shared_artifacts=payload.shared_artifacts,
                candidate_generation_sha256=payload.candidate_generation_sha256,
                history_shuffle_permutation_map_sha256=payload.history_shuffle_permutation_map_sha256,
                diagnostics=payload.diagnostics,
            )
            stage = _STAGE_PUBLICATION
            published = family.write_completed(record)
            completed_by_family[planned.heldout_family].add(planned.unit_id)
            counts["completed"] = int(counts["completed"]) + (1 if published else 0)
            if not published:
                counts["skipped"] = int(counts["skipped"]) + 1
        except KeyboardInterrupt as exc:
            _attempt(
                family,
                planned,
                exc,
                attempt_number=attempt_number,
                stage=stage,
                retryable=True,
                started_at=started_at,
                elapsed=time.perf_counter() - started,
                status="interrupted",
            )
            counts["interrupted"] = int(counts["interrupted"]) + 1
            raise
        except Exception as exc:
            _attempt(
                family,
                planned,
                exc,
                attempt_number=attempt_number,
                stage=stage,
                retryable=_retryable(exc),
                started_at=started_at,
                elapsed=time.perf_counter() - started,
            )
            counts["failed"] = int(counts["failed"]) + 1
            raise
    completed_total = sum(len(values) for values in completed_by_family.values())
    if completed_total != EXPECTED_TOTAL_UNIT_COUNT:
        _fail("outcome diagnostic execution finished with missing units")
    counts["complete"] = True
    return counts


def _summary(
    *,
    context: OutcomeDiagnosticExecutionContext,
    validate_only: bool,
    completed: int = 0,
    skipped: int = 0,
    failed: int = 0,
    interrupted: int = 0,
    complete: bool = False,
) -> OutcomeDiagnosticRunSummary:
    return {
        "validate_only": bool(validate_only),
        "plan_id": context.plan.plan.plan_id,
        "protocol_sha256": context.protocol.sha256,
        "model_authority_sha256": _authority_digest(context.authority),
        "family_order": tuple(FAMILIES),
        "condition_order": tuple(CONDITIONS),
        "expected_total": EXPECTED_TOTAL_UNIT_COUNT,
        "completed": int(completed),
        "skipped": int(skipped),
        "failed": int(failed),
        "interrupted": int(interrupted),
        "complete": bool(complete),
    }


def run_outcome_diagnostic_development(
    model_readiness_snapshot: OutcomeDiagnosticModelReadinessSnapshot,
    *,
    expected_git_commit: str,
    validate_only: bool = False,
) -> OutcomeDiagnosticRunSummary:
    """Validate or execute all 5,760 frozen development units, resumably."""

    if type(model_readiness_snapshot) is not OutcomeDiagnosticModelReadinessSnapshot:
        _fail("canonical outcome model readiness snapshot is required")
    if not isinstance(expected_git_commit, str) or not expected_git_commit:
        _fail("outcome diagnostic execution requires an explicit git commit")
    try:
        model_readiness_snapshot.preflight(expected_git_commit=expected_git_commit)
        context = OutcomeDiagnosticExecutionContext.canonical(model_readiness_snapshot)
        expected = _validate_exact_matrix(context)
        _authority_digest(context.authority)
    except OutcomeDiagnosticDriverError:
        raise
    except (OutcomeDiagnosticReadinessError, OutcomeDiagnosticExecutionModelError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _fail("outcome diagnostic readiness cannot authorize execution", exc)

    if validate_only:
        try:
            with model_readiness_snapshot.base.hold_for_activation(expected_git_commit=expected_git_commit) as lease:
                _validate_stores_for_readiness(
                    model_readiness_snapshot,
                    expected,
                    lease,
                    validate_only=True,
                )
        except OutcomeDiagnosticDriverError:
            raise
        except (OutcomeDiagnosticReadinessError, OutcomeDiagnosticResultStoreError, OSError, RuntimeError, TypeError, ValueError) as exc:
            _fail("outcome diagnostic validation-only preflight failed", exc)
        return _summary(context=context, validate_only=True)

    try:
        with model_readiness_snapshot.base.hold_for_activation(expected_git_commit=expected_git_commit) as lease:
            stores = _validate_stores_for_readiness(
                model_readiness_snapshot,
                expected,
                lease,
                validate_only=False,
            )
            if stores is None:
                _fail("execution stores were not prepared")
            # Revalidate the model authority after result-store preparation and
            # immediately before granting the runtime write capability.
            model_readiness_snapshot.preflight(expected_git_commit=expected_git_commit)
            with activate_outcome_diagnostic_result_stores(
                stores,
                expected,
                lease,
                expected_git_commit=expected_git_commit,
            ) as batch:
                counts = _execute_loop(batch, context, expected)
    except OutcomeDiagnosticDriverError:
        raise
    except (OutcomeDiagnosticReadinessError, OutcomeDiagnosticResultStoreError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _fail("outcome diagnostic execution failed closed", exc)
    return _summary(context=context, validate_only=False, **counts)


__all__ = [
    "EXPECTED_TOTAL_UNIT_COUNT",
    "EXPECTED_TOTAL_UNITS",
    "OutcomeDiagnosticDriverError",
    "OutcomeDiagnosticRunSummary",
    "run_outcome_diagnostic_development",
]

EXPECTED_TOTAL_UNITS = EXPECTED_TOTAL_UNIT_COUNT
