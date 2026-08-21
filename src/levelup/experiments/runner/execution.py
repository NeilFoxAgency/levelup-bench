"""Sequential atomic-unit execution with explicit resume and retry semantics."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from typing import Literal

from pydantic import ValidationError

from levelup.experiments.runner.provenance import utc_now
from levelup.experiments.runner.records import (
    AttemptRecord,
    PlannedUnit,
    SplitPhase,
    UnitPayload,
    UnitRecord,
)
from levelup.experiments.runner.storage import (
    ArtifactValidationError,
    ConflictingResultError,
    RunStore,
)

UnitExecutor = Callable[[PlannedUnit], UnitPayload]


class ExperimentRunner:
    """Run missing atomic units without changing scientific semantics on resume."""

    def __init__(self, store: RunStore) -> None:
        self.store = store

    def execute(
        self,
        executor: UnitExecutor,
        *,
        resume: bool = True,
        retry_failed: bool = True,
        fail_fast: bool = True,
        phases: tuple[SplitPhase, ...] = ("development", "validation"),
        allow_final: bool = False,
    ) -> dict[str, int]:
        if not self.store._execution_ready:
            raise RuntimeError("RunStore.initialize(for_execution=True) is required")
        selected_phases = set(phases)
        if "final" in selected_phases and not allow_final:
            raise ValueError("final execution requires allow_final=True after method freeze")
        attempts_by_unit = {
            unit_id: [] for unit_id in (unit.unit_id for unit in self.store.expected.units)
        }
        for attempt in self.store.attempt_records():
            attempts_by_unit[attempt.unit_id].append(attempt)

        counts = {
            "completed": 0,
            "skipped": 0,
            "unselected": 0,
            "failed": 0,
            "interrupted": 0,
        }
        for planned in self.store.expected.units:
            if planned.key.phase not in selected_phases:
                counts["unselected"] += 1
                continue
            existing = self.store.load_completed(planned.unit_id)
            if existing is not None:
                if not resume:
                    raise ConflictingResultError(
                        f"resume is disabled and unit is already complete: {planned.unit_id}"
                    )
                counts["skipped"] += 1
                continue
            prior_attempts = attempts_by_unit[planned.unit_id]
            if prior_attempts:
                if not retry_failed or not prior_attempts[-1].retryable:
                    counts["skipped"] += 1
                    continue

            started_at = utc_now()
            started = time.perf_counter()
            stage = "executor"
            try:
                raw_payload = executor(planned)
                stage = "payload-validation"
                if isinstance(raw_payload, UnitPayload):
                    payload = UnitPayload.model_validate(
                        raw_payload.model_dump(mode="json", warnings=False)
                    )
                else:
                    payload = UnitPayload.model_validate(raw_payload)

                stage = "record-validation"
                record = UnitRecord(
                    run_id=self.store.run_id,
                    config_sha256=self.store.config_sha256,
                    unit_id=planned.unit_id,
                    key=planned.key,
                    seeds=planned.seeds,
                    exposure_manifest_sha256=planned.exposure_manifest_sha256,
                    started_at_utc=started_at,
                    finished_at_utc=utc_now(),
                    elapsed_wall_seconds=time.perf_counter() - started,
                    outcome=payload.outcome,
                    accounting=payload.accounting,
                    shared_artifact=payload.shared_artifact,
                    shared_artifacts=payload.shared_artifacts,
                    candidate_generation_sha256=payload.candidate_generation_sha256,
                    diagnostics=payload.diagnostics,
                )
                stage = "unit-publication"
                self.store.write_completed(record)
            except KeyboardInterrupt as exc:
                self._record_attempt(
                    planned,
                    status="interrupted",
                    stage=stage,
                    exception=exc,
                    retryable=True,
                    started_at=started_at,
                    elapsed=time.perf_counter() - started,
                )
                counts["interrupted"] += 1
                raise
            except Exception as exc:
                retryable = not isinstance(
                    exc,
                    (ArtifactValidationError, ConflictingResultError, ValidationError),
                )
                self._record_attempt(
                    planned,
                    status="failed",
                    stage=stage,
                    exception=exc,
                    retryable=retryable,
                    started_at=started_at,
                    elapsed=time.perf_counter() - started,
                )
                counts["failed"] += 1
                if fail_fast:
                    raise
                continue
            counts["completed"] += 1
        return counts

    def _record_attempt(
        self,
        planned: PlannedUnit,
        *,
        status: Literal["failed", "interrupted"],
        stage: str,
        exception: BaseException,
        retryable: bool,
        started_at: datetime,
        elapsed: float,
    ) -> None:
        exception_type = type(exception).__name__
        record = AttemptRecord(
            run_id=self.store.run_id,
            config_sha256=self.store.config_sha256,
            unit_id=planned.unit_id,
            attempt=self.store.next_attempt_number(planned.unit_id),
            key=planned.key,
            seeds=planned.seeds,
            status=status,
            stage=stage,
            exception_type=exception_type,
            sanitized_message=f"{stage} raised {exception_type}",
            retryable=retryable,
            started_at_utc=started_at,
            finished_at_utc=utc_now(),
            elapsed_wall_seconds=elapsed,
        )
        self.store.write_attempt(record)
