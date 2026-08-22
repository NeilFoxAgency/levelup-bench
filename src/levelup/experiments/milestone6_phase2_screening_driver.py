"""Run the frozen development-only Phase 2 screening matrix.

This module is intentionally an execution boundary, not an analysis boundary.  It
loads the pinned readiness inventory, rechecks the immutable runtime immediately
before execution, and asks :class:`ExperimentRunner` to execute missing validation
units.  It never reads completed records for aggregation or performs selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from levelup.experiments.milestone6_phase2_screening_execution import (
    ScreeningModelCache,
    execute_screening_unit,
)
from levelup.experiments.milestone6_phase2_screening_provenance import (
    CANONICAL_READINESS_PATH,
)
from levelup.experiments.milestone6_phase2_screening_runtime import (
    ScreeningRuntime,
    ScreeningRuntimeFold,
    load_screening_runtime,
)
from levelup.experiments.runner.execution import ExperimentRunner

# These are the only families authorized by the frozen development protocol.  The
# runtime manifest is checked against this ordered tuple rather than trusting a
# caller-provided subset or a newly constructed plan.
CANONICAL_FAMILY_ORDER = (
    "plain",
    "battery",
    "cooldown",
    "heat",
    "momentum",
    "combo",
)
EXPECTED_UNITS_PER_FOLD = 1_520
EXPECTED_TOTAL_UNITS = 9_120


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _validate_runtime_inventory(runtime: ScreeningRuntime) -> tuple[ScreeningRuntimeFold, ...]:
    """Validate the exact development matrix before opening any execution gate."""

    manifest = getattr(runtime, "manifest", None)
    family_order = tuple(getattr(manifest, "family_order", ()))
    if family_order != CANONICAL_FAMILY_ORDER:
        _fail("screening driver requires the exact six canonical development folds")
    folds = tuple(getattr(runtime, "folds", ()))
    if len(folds) != len(CANONICAL_FAMILY_ORDER):
        _fail("screening driver requires exactly six development folds")
    if tuple(getattr(fold, "family_id", "") for fold in folds) != family_order:
        _fail("screening fold order does not match the canonical development order")

    total = 0
    for fold in folds:
        store = getattr(fold, "store", None)
        if bool(getattr(store, "_execution_ready", False)):
            _fail("screening stores must be locked before the execution recheck")
        config = getattr(fold, "config", None)
        expected = getattr(store, "expected", None)
        units = tuple(getattr(expected, "units", ()))
        if len(units) != EXPECTED_UNITS_PER_FOLD:
            _fail("screening fold does not contain exactly 1,520 expected units")
        if any(getattr(getattr(unit, "key", None), "phase", None) != "validation" for unit in units):
            _fail("screening driver accepts validation units only")
        split = getattr(config, "split", None)
        if getattr(split, "final_tasks", ()):
            _fail("screening driver refuses configs containing final tasks")
        conditions = tuple(getattr(config, "conditions", ()))
        if any(tuple(getattr(condition, "execution_phases", ())) != ("validation",) for condition in conditions):
            _fail("screening driver refuses non-validation screening conditions")
        total += len(units)
    if total != EXPECTED_TOTAL_UNITS:
        _fail("screening matrix does not contain exactly 9,120 expected units")
    return folds


def run_development_screening(
    manifest_path: str | Path,
    manifest_bytes_sha256: str,
    raw_root: str | Path,
    repository: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute missing validation units from the pinned development inventory.

    ``dry_run`` loads and validates the complete inventory, but leaves stores locked
    and executes zero units.  Resume and retry behavior are fixed to the standard
    :class:`ExperimentRunner` semantics; scientific budgets and selection knobs are
    deliberately not exposed here.
    """

    repository_path = Path(repository).resolve(strict=False)
    canonical_manifest_path = repository_path / CANONICAL_READINESS_PATH
    if Path(manifest_path).resolve(strict=False) != canonical_manifest_path:
        _fail(
            "screening driver requires the canonical committed readiness manifest"
        )

    runtime = load_screening_runtime(
        manifest_path,
        raw_root,
        repository,
        manifest_bytes_sha256=manifest_bytes_sha256,
    )
    folds = _validate_runtime_inventory(runtime)
    # Validation-only deliberately leaves every store locked.  Execution has one
    # and only one transactional recheck, immediately before the fold loop.
    if dry_run:
        return {
            "dry_run": True,
            "manifest_path": str(Path(manifest_path)),
            "folds": [],
            "total": {"completed": 0, "skipped": 0, "unselected": 0, "failed": 0, "interrupted": 0},
        }
    runtime.recheck_before_execution()
    if any(not bool(getattr(fold.store, "_execution_ready", False)) for fold in folds):
        _fail("screening runtime recheck did not activate every fold store")

    fold_results: list[dict[str, Any]] = []
    total = {"completed": 0, "skipped": 0, "unselected": 0, "failed": 0, "interrupted": 0}
    expected_count_keys = frozenset(total)
    for fold in folds:
        # A cache is intentionally scoped to one held-out fold.  No model or
        # artifact object can leak across family boundaries.
        model_cache = ScreeningModelCache()
        counts = ExperimentRunner(fold.store).execute(
            lambda planned, *, _fold=fold, _cache=model_cache: execute_screening_unit(
                _fold,
                planned,
                model_cache=_cache,
            ),
            resume=True,
            retry_failed=True,
            fail_fast=True,
            phases=("validation",),
            allow_final=False,
        )
        if set(counts) != expected_count_keys:
            _fail("screening runner returned an unexpected bookkeeping count set")
        result: dict[str, int] = {}
        for key, value in counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail("screening runner returned an invalid bookkeeping count")
            result[str(key)] = value
        if any(result[key] != 0 for key in ("unselected", "failed", "interrupted")):
            _fail("screening validation matrix contains failed or unselected units")
        if sum(result.values()) != EXPECTED_UNITS_PER_FOLD:
            _fail("screening fold bookkeeping does not cover exactly 1,520 units")
        missing_units = getattr(fold.store, "missing_units", None)
        if not callable(missing_units) or tuple(missing_units()):
            _fail("screening fold completed counts but still has missing units")
        fold_results.append({"family_id": fold.family_id, "counts": result})
        for key in total:
            total[key] += result.get(key, 0)
    return {
        "dry_run": False,
        "manifest_path": str(Path(manifest_path)),
        "folds": fold_results,
        "total": total,
    }


# Short alias for callers that use the module's boundary name.
run_screening = run_development_screening


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="load the pinned inventory without activating stores or executing units",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="recheck and execute missing development validation units",
    )
    args = parser.parse_args(argv)
    result = run_development_screening(
        args.manifest_path,
        args.manifest_sha256,
        args.raw_root,
        args.repository,
        dry_run=args.validate_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_READINESS_PATH",
    "CANONICAL_FAMILY_ORDER",
    "EXPECTED_TOTAL_UNITS",
    "EXPECTED_UNITS_PER_FOLD",
    "main",
    "run_development_screening",
    "run_screening",
]
