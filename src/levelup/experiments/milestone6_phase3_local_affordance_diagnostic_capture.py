"""Development-only, pre-outcome local-affordance diagnostic capture.

The capture boundary binds the committed Phase 2 runtime, Phase 3 evidence
authorities, raw-probe store and local-affordance plan.  It computes diagnostics
from the sealed identity-free query matrix and emits metadata plus a typed
report.  No model training, search, replay, evaluator, oracle, comparative
result, or final-family path exists in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from levelup.experiments.milestone6_phase2_screening_provenance import CANONICAL_READINESS_PATH
from levelup.experiments.milestone6_phase2_screening_runtime import (
    ScreeningRuntime,
    load_screening_runtime,
    recheck_screening_runtime_readonly,
)
from levelup.experiments.milestone6_phase3_anchor import (
    validate_phase3_anchor_manifest_bytes,
)
from levelup.experiments.milestone6_phase3_evidence import (
    validate_phase3_evidence_lock_bytes,
)
from levelup.experiments.milestone6_phase3_local_affordance_diagnostic_queries import (
    build_local_affordance_diagnostic_query_authority,
)
from levelup.experiments.milestone6_phase3_local_affordance_diagnostics import (
    LocalAffordanceDiagnosticReport,
    aggregate_local_affordance_diagnostics,
    validate_local_affordance_diagnostic_report,
)
from levelup.experiments.milestone6_phase3_local_affordance_plan import (
    validate_local_affordance_plan,
    validate_local_affordance_plan_lock_bytes,
)
from levelup.experiments.milestone6_phase3_local_affordance_raw_authority import (
    build_expected_raw_probe_authority,
    require_raw_probe_authority_snapshot,
    validate_complete_raw_probe_authority,
)
from levelup.experiments.milestone6_phase3_local_affordance_raw_store import (
    open_existing_raw_probe_store,
)
from levelup.experiments.milestone6_phase3_plan import (
    bind_validated_phase3_plan,
    validate_phase3_plan_lock_bytes,
)
from levelup.experiments.runner import secure_fs

PHASE3_PLAN_LOCK_RELATIVE_PATH = Path("configs/milestone6/phase3_plan_lock.json")
PHASE3_ANCHOR_RELATIVE_PATH = Path("configs/milestone6/phase3_anchor_manifest.json")
PHASE3_EVIDENCE_RELATIVE_PATH = Path("configs/milestone6/phase3_evidence_lock.json")
LOCAL_PROTOCOL_RELATIVE_PATH = Path("configs/milestone6/phase3_local_affordance_protocol.json")
DEVELOPMENT_PROTOCOL_RELATIVE_PATH = Path("configs/milestone6/development_protocol.json")
DEVELOPMENT_TASKS_RELATIVE_PATH = Path("configs/milestone6/development_tasks.json")
REPRESENTATION_LADDER_RELATIVE_PATH = Path("configs/milestone6/phase3_representation_ladder.json")
RAW_CAPTURE_SUMMARY_RELATIVE_PATH = Path(
    "experiments/milestone6_phase3_local_affordance_raw_capture.json"
)
LOCAL_PLAN_LOCK_RELATIVE_PATH = Path("configs/milestone6/phase3_local_affordance_plan_lock.json")
HEX64 = r"^[0-9a-f]{64}$"
SOURCE_DIGEST_NAMES = (
    "development_protocol",
    "development_tasks",
    "local_affordance_plan_lock",
    "local_protocol",
    "phase2_readiness_manifest",
    "phase3_anchor_manifest",
    "phase3_evidence_lock_file",
    "phase3_plan_lock",
    "raw_authority_content",
    "raw_authority_manifest_file",
    "raw_authority_manifest_id",
    "raw_capture_summary",
    "representation_ladder",
)
_AUTHORITY_FILE_LOCATIONS = {
    "phase3_plan_lock": ("config", "phase3_plan_lock.json"),
    "phase3_anchor_manifest": ("config", "phase3_anchor_manifest.json"),
    "phase3_evidence_lock_file": ("config", "phase3_evidence_lock.json"),
    "local_affordance_plan_lock": ("config", "phase3_local_affordance_plan_lock.json"),
    "local_protocol": ("config", "phase3_local_affordance_protocol.json"),
    "development_protocol": ("config", "development_protocol.json"),
    "development_tasks": ("config", "development_tasks.json"),
    "representation_ladder": ("config", "phase3_representation_ladder.json"),
    "raw_capture_summary": ("authority_experiments", "milestone6_phase3_local_affordance_raw_capture.json"),
    "phase2_readiness_manifest": ("screening_experiments", "milestone6_phase2_screening_readiness.json"),
}


class LocalAffordanceDiagnosticCaptureError(RuntimeError):
    """Raised when the pre-outcome diagnostic boundary cannot prove integrity."""


class DiagnosticResourceCounters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    training_actions: StrictInt = 0
    search_actions: StrictInt = 0
    replay_actions: StrictInt = 0
    evaluator_calls: StrictInt = 0
    oracle_calls: StrictInt = 0
    candidate_searches: StrictInt = 0
    model_training_runs: StrictInt = 0

    @model_validator(mode="after")
    def all_zero(self) -> "DiagnosticResourceCounters":
        if any(value != 0 for value in self.model_dump(mode="python").values()):
            raise ValueError("diagnostic capture resource counters must all be zero")
        return self


class LocalAffordanceDiagnosticCapture(BaseModel):
    """Immutable metadata envelope for one complete development diagnostic."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["milestone6.phase3.local-affordance-diagnostic-capture.v1"] = (
        "milestone6.phase3.local-affordance-diagnostic-capture.v1"
    )
    scope: Literal["known-development-only"] = "known-development-only"
    status: Literal["complete"] = "complete"
    activation_git_commit_sha: str = Field(min_length=40, max_length=40, pattern=r"^[0-9a-f]+$")
    source_digests: tuple[tuple[str, str], ...]
    readiness_manifest_sha256: str = Field(pattern=HEX64)
    query_matrix_sha256: str = Field(pattern=HEX64)
    evidence_lock_sha256: str = Field(pattern=HEX64)
    evidence_lock_file_sha256: str = Field(pattern=HEX64)
    raw_authority_content_sha256: str = Field(pattern=HEX64)
    raw_authority_manifest_id: str = Field(pattern=HEX64)
    raw_manifest_file_sha256: str = Field(pattern=HEX64)
    plan_id: str = Field(pattern=HEX64)
    plan_lock_file_sha256: str = Field(pattern=HEX64)
    training_query_count: StrictInt = 1200
    heldout_query_count: StrictInt = 240
    training_fold_count: StrictInt = 30
    heldout_binding_count: StrictInt = 240
    training_state_query_count: StrictInt = Field(gt=0)
    heldout_state_query_count: StrictInt = Field(gt=0)
    report: LocalAffordanceDiagnosticReport
    diagnostics_pass: StrictBool
    resources: DiagnosticResourceCounters = DiagnosticResourceCounters()
    model_training: StrictBool = False
    candidate_search: StrictBool = False
    comparative_results_generated: StrictBool = False
    comparative_results_inspected: StrictBool = False
    final_family_access: StrictBool = False

    @model_validator(mode="after")
    def exact_scope_and_report(self) -> "LocalAffordanceDiagnosticCapture":
        if (
            self.training_query_count != 1200
            or self.heldout_query_count != 240
            or self.training_fold_count != 30
            or self.heldout_binding_count != 240
        ):
            raise ValueError("capture matrix counts differ from the frozen authority")
        if tuple(name for name, _digest in self.source_digests) != SOURCE_DIGEST_NAMES:
            raise ValueError("source digest names differ from the exact authority set")
        if any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for _name, digest in self.source_digests
        ):
            raise ValueError("source digest is malformed")
        sources = dict(self.source_digests)
        if (
            sources["phase2_readiness_manifest"] != self.readiness_manifest_sha256
            or sources["phase3_evidence_lock_file"] != self.evidence_lock_file_sha256
            or sources["local_affordance_plan_lock"] != self.plan_lock_file_sha256
            or sources["raw_authority_content"] != self.raw_authority_content_sha256
            or sources["raw_authority_manifest_id"] != self.raw_authority_manifest_id
            or sources["raw_authority_manifest_file"] != self.raw_manifest_file_sha256
        ):
            raise ValueError("source digests differ from the capture lineage fields")
        report = validate_local_affordance_diagnostic_report(self.report)
        if report != self.report:
            raise ValueError("report is not canonically validated")
        for population, expected in (("training", 1200), ("heldout", 240)):
            if report.for_population(population).evidence_query_count != expected:
                raise ValueError("report query count differs from capture envelope")
        if (
            report.for_population("training").state_query_count
            != self.training_state_query_count
            or report.for_population("heldout").state_query_count
            != self.heldout_state_query_count
        ):
            raise ValueError("report state-query counts differ from capture envelope")
        expected_pass = all(
            population.coverage_gate.passes
            and all(family.coverage_gate.passes for family in population.family_summaries)
            for population in report.populations
        )
        if self.diagnostics_pass is not expected_pass:
            raise ValueError("diagnostic pass flag differs from report gates")
        if self.activation_git_commit_sha == "0" * 40:
            raise ValueError("activation commit identity is missing")
        if any(
            (
                self.model_training,
                self.candidate_search,
                self.comparative_results_generated,
                self.comparative_results_inspected,
                self.final_family_access,
            )
        ):
            raise ValueError("diagnostic capture cannot include prohibited execution or access")
        return self


def _fail(message: str, exc: BaseException | None = None) -> None:
    if exc is None:
        raise LocalAffordanceDiagnosticCaptureError(message)
    raise LocalAffordanceDiagnosticCaptureError(message) from exc


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular(path: Path, label: str) -> bytes:
    absolute = Path(os.path.abspath(path))
    for candidate in (absolute, *absolute.parents):
        if os.path.lexists(candidate) and candidate.is_symlink():
            _fail(f"{label} path contains a symlink")
    if not absolute.is_file():
        _fail(f"{label} is not a regular file")
    try:
        content = absolute.read_bytes()
    except OSError as exc:
        _fail(f"cannot read {label}", exc)
    return content


def _canonical_directory(path: str | Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for candidate in (absolute, *absolute.parents):
        if os.path.lexists(candidate) and candidate.is_symlink():
            _fail(f"{label} path contains a symlink")
    if not absolute.is_dir():
        _fail(f"{label} is not a directory")
    return absolute


@dataclass(frozen=True, slots=True)
class _PinnedAuthorityFiles:
    initial: dict[str, bytes]
    descriptors: dict[str, int]

    def read_all(self) -> dict[str, bytes]:
        return {
            key: secure_fs.read_bytes_at(self.descriptors[location], name)
            for key, (location, name) in _AUTHORITY_FILE_LOCATIONS.items()
        }


@contextmanager
def _open_pinned_authority_files(
    screening_repository: Path,
    authority_repository: Path,
) -> Iterator[_PinnedAuthorityFiles]:
    """Hold repository roots and source directories through the whole diagnostic."""

    try:
        with ExitStack() as stack:
            screening_root_fd = secure_fs.open_directory_chain(screening_repository)
            stack.callback(os.close, screening_root_fd)
            authority_root_fd = secure_fs.open_directory_chain(authority_repository)
            stack.callback(os.close, authority_root_fd)
            screening_experiments_fd = secure_fs.open_child_directory(
                screening_root_fd, "experiments"
            )
            stack.callback(os.close, screening_experiments_fd)
            authority_experiments_fd = secure_fs.open_child_directory(
                authority_root_fd, "experiments"
            )
            stack.callback(os.close, authority_experiments_fd)
            authority_config_fd = secure_fs.open_child_chain(
                authority_root_fd, "configs", "milestone6"
            )
            stack.callback(os.close, authority_config_fd)
            descriptors = {
                "config": authority_config_fd,
                "authority_experiments": authority_experiments_fd,
                "screening_experiments": screening_experiments_fd,
            }
            pinned = _PinnedAuthorityFiles(initial={}, descriptors=descriptors)
            object.__setattr__(pinned, "initial", pinned.read_all())
            yield pinned
    except LocalAffordanceDiagnosticCaptureError:
        raise
    except Exception as exc:
        _fail("cannot pin committed diagnostic authority files", exc)


def _commit(runtime: ScreeningRuntime) -> str:
    value = getattr(runtime.provenance, "git_commit_sha", None)
    if not isinstance(value, str) or len(value) < 40:
        _fail("runtime provenance has no commit identity")
    return value


def _build_capture(
    *,
    runtime: ScreeningRuntime,
    evidence_lock: Any,
    evidence_lock_bytes: bytes,
    raw_snapshot: Any,
    plan: Any,
    source_digests: tuple[tuple[str, str], ...],
    plan_lock_file_sha256: str,
) -> LocalAffordanceDiagnosticCapture:
    try:
        recheck_screening_runtime_readonly(runtime)
        require_raw_probe_authority_snapshot(raw_snapshot)
        if raw_snapshot.evidence_lock_file_sha256 != _sha(evidence_lock_bytes):
            _fail("raw authority evidence-lock file digest differs")
        query_authority = build_local_affordance_diagnostic_query_authority(
            runtime, evidence_lock, raw_snapshot, plan
        )
        query_authority.require_complete()
        report = aggregate_local_affordance_diagnostics(query_authority.queries)
        # Recompute from the sealed matrix and compare, so a supplied report can
        # never become the source of truth.
        recomputed = aggregate_local_affordance_diagnostics(query_authority.queries)
        if recomputed != report:
            _fail("diagnostic report recomputation differs")
        query_authority.require_complete()
        recheck_screening_runtime_readonly(runtime)
        require_raw_probe_authority_snapshot(raw_snapshot)
    except LocalAffordanceDiagnosticCaptureError:
        raise
    except Exception as exc:
        _fail("pre-outcome diagnostic capture failed closed", exc)
    try:
        payload = query_authority.model_dump()
        expected_pass = all(
            pop.coverage_gate.passes
            and all(fam.coverage_gate.passes for fam in pop.family_summaries)
            for pop in report.populations
        )
        return LocalAffordanceDiagnosticCapture(
            activation_git_commit_sha=_commit(runtime),
            source_digests=source_digests,
            readiness_manifest_sha256=_sha(runtime.manifest_bytes),
            query_matrix_sha256=payload["query_matrix_sha256"],
            evidence_lock_sha256=evidence_lock.evidence_lock_sha256,
            evidence_lock_file_sha256=_sha(evidence_lock_bytes),
            raw_authority_content_sha256=raw_snapshot.authority_content_sha256,
            raw_authority_manifest_id=raw_snapshot.manifest.manifest_id,
            raw_manifest_file_sha256=raw_snapshot.manifest_file.snapshot.sha256,
            plan_id=plan.plan_id,
            plan_lock_file_sha256=plan_lock_file_sha256,
            training_state_query_count=payload["training_state_query_count"],
            heldout_state_query_count=payload["heldout_state_query_count"],
            report=report,
            diagnostics_pass=expected_pass,
        )
    except Exception as exc:
        _fail("diagnostic capture envelope construction failed", exc)


def run_local_affordance_diagnostic_capture(
    manifest_path: str | Path,
    manifest_bytes_sha256: str,
    screening_raw_root: str | Path,
    raw_probe_root: str | Path,
    screening_repository: str | Path,
    authority_repository: str | Path,
) -> LocalAffordanceDiagnosticCapture:
    """Load exact authorities and compute a development-only diagnostic report."""

    repository = _canonical_directory(screening_repository, "screening repository")
    authority = _canonical_directory(authority_repository, "authority repository")
    screening_raw = _canonical_directory(screening_raw_root, "screening raw root")
    raw_probe = _canonical_directory(raw_probe_root, "raw-probe root")
    manifest = Path(os.path.abspath(manifest_path))
    if manifest != repository / CANONICAL_READINESS_PATH:
        _fail("capture requires the canonical committed readiness manifest")
    initial_files = {
        "phase2_readiness_manifest": _read_regular(manifest, "Phase 2 readiness manifest"),
        "phase3_plan_lock": _read_regular(
            authority / PHASE3_PLAN_LOCK_RELATIVE_PATH, "Phase 3 plan lock"
        ),
        "phase3_anchor_manifest": _read_regular(
            authority / PHASE3_ANCHOR_RELATIVE_PATH, "Phase 3 anchor manifest"
        ),
        "phase3_evidence_lock_file": _read_regular(
            authority / PHASE3_EVIDENCE_RELATIVE_PATH, "Phase 3 evidence lock"
        ),
        "local_affordance_plan_lock": _read_regular(
            authority / LOCAL_PLAN_LOCK_RELATIVE_PATH, "local-affordance plan lock"
        ),
        "local_protocol": _read_regular(
            authority / LOCAL_PROTOCOL_RELATIVE_PATH, "local-affordance protocol"
        ),
        "development_protocol": _read_regular(
            authority / DEVELOPMENT_PROTOCOL_RELATIVE_PATH, "development protocol"
        ),
        "development_tasks": _read_regular(
            authority / DEVELOPMENT_TASKS_RELATIVE_PATH, "development tasks"
        ),
        "representation_ladder": _read_regular(
            authority / REPRESENTATION_LADDER_RELATIVE_PATH, "representation ladder"
        ),
        "raw_capture_summary": _read_regular(
            authority / RAW_CAPTURE_SUMMARY_RELATIVE_PATH, "raw capture summary"
        ),
    }
    if _sha(initial_files["phase2_readiness_manifest"]) != manifest_bytes_sha256:
        _fail("Phase 2 readiness manifest digest differs from the requested identity")
    try:
        runtime = load_screening_runtime(
            manifest,
            screening_raw,
            repository,
            manifest_bytes_sha256=manifest_bytes_sha256,
            authority_repository=authority,
        )
        plan_lock_bytes = initial_files["phase3_plan_lock"]
        validated_plan = bind_validated_phase3_plan(validate_phase3_plan_lock_bytes(plan_lock_bytes))
        anchor_bytes = initial_files["phase3_anchor_manifest"]
        anchor = validate_phase3_anchor_manifest_bytes(anchor_bytes, runtime=runtime)
        evidence_bytes = initial_files["phase3_evidence_lock_file"]
        evidence = validate_phase3_evidence_lock_bytes(
            evidence_bytes,
            runtime=runtime,
            validated_plan=validated_plan,
            anchor_manifest=anchor,
            anchor_file_bytes=anchor_bytes,
            plan_lock_bytes=plan_lock_bytes,
        )
        plan_lock_bytes_local = initial_files["local_affordance_plan_lock"]
        plan = validate_local_affordance_plan_lock_bytes(
            plan_lock_bytes_local, repository=authority
        )
        local_bytes = initial_files["local_protocol"]
        development_bytes = initial_files["development_protocol"]
        task_bytes = initial_files["development_tasks"]
        expected = build_expected_raw_probe_authority(
            local_affordance_protocol_bytes=local_bytes,
            development_protocol_bytes=development_bytes,
            development_tasks_bytes=task_bytes,
            phase3_evidence_lock_bytes=evidence_bytes,
        )
        with open_existing_raw_probe_store(raw_probe) as reader:
            raw_snapshot = validate_complete_raw_probe_authority(reader, expected=expected)
            plan_sources = tuple(plan.source_sha256)
            source_digests = tuple(
                sorted(
                    {
                        *plan_sources,
                        ("phase2_readiness_manifest", _sha(runtime.manifest_bytes)),
                        ("phase3_anchor_manifest", _sha(anchor_bytes)),
                        ("phase3_plan_lock", _sha(plan_lock_bytes)),
                        ("phase3_evidence_lock_file", _sha(evidence_bytes)),
                        ("local_affordance_plan_lock", _sha(plan_lock_bytes_local)),
                        ("raw_authority_manifest_id", raw_snapshot.manifest.manifest_id),
                        (
                            "raw_authority_manifest_file",
                            raw_snapshot.manifest_file.snapshot.sha256,
                        ),
                        ("raw_authority_content", raw_snapshot.authority_content_sha256),
                    }
                )
            )
            result = _build_capture(
                runtime=runtime,
                evidence_lock=evidence,
                evidence_lock_bytes=evidence_bytes,
                raw_snapshot=raw_snapshot,
                plan=plan,
                source_digests=source_digests,
                plan_lock_file_sha256=_sha(plan_lock_bytes_local),
            )
            # Reopen and fully validate the descriptor-pinned store after all
            # aggregation, closing the race between evidence and report.
            with open_existing_raw_probe_store(raw_probe) as final_reader:
                final_snapshot = validate_complete_raw_probe_authority(final_reader, expected=expected)
            if final_snapshot != raw_snapshot:
                _fail("raw authority changed during diagnostic aggregation")
            final_files = {
                "phase2_readiness_manifest": _read_regular(
                    manifest, "Phase 2 readiness manifest"
                ),
                "phase3_plan_lock": _read_regular(
                    authority / PHASE3_PLAN_LOCK_RELATIVE_PATH, "Phase 3 plan lock"
                ),
                "phase3_anchor_manifest": _read_regular(
                    authority / PHASE3_ANCHOR_RELATIVE_PATH, "Phase 3 anchor manifest"
                ),
                "phase3_evidence_lock_file": _read_regular(
                    authority / PHASE3_EVIDENCE_RELATIVE_PATH, "Phase 3 evidence lock"
                ),
                "local_affordance_plan_lock": _read_regular(
                    authority / LOCAL_PLAN_LOCK_RELATIVE_PATH, "local-affordance plan lock"
                ),
                "local_protocol": _read_regular(
                    authority / LOCAL_PROTOCOL_RELATIVE_PATH, "local-affordance protocol"
                ),
                "development_protocol": _read_regular(
                    authority / DEVELOPMENT_PROTOCOL_RELATIVE_PATH, "development protocol"
                ),
                "development_tasks": _read_regular(
                    authority / DEVELOPMENT_TASKS_RELATIVE_PATH, "development tasks"
                ),
                "representation_ladder": _read_regular(
                    authority / REPRESENTATION_LADDER_RELATIVE_PATH, "representation ladder"
                ),
                "raw_capture_summary": _read_regular(
                    authority / RAW_CAPTURE_SUMMARY_RELATIVE_PATH, "raw capture summary"
                ),
            }
            if final_files != initial_files:
                _fail("committed authority bytes changed during diagnostic aggregation")
            validate_local_affordance_plan(plan, repository=authority)
            recheck_screening_runtime_readonly(runtime)
            return result
    except LocalAffordanceDiagnosticCaptureError:
        raise
    except Exception as exc:
        _fail("diagnostic capture authority validation failed", exc)
    raise AssertionError("unreachable")


capture_local_affordance_diagnostics = run_local_affordance_diagnostic_capture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--screening-raw-root", type=Path, required=True)
    parser.add_argument("--raw-probe-root", type=Path, required=True)
    parser.add_argument("--screening-repository", type=Path, required=True)
    parser.add_argument("--authority-repository", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_local_affordance_diagnostic_capture(
        args.manifest_path,
        args.manifest_sha256,
        args.screening_raw_root,
        args.raw_probe_root,
        args.screening_repository,
        args.authority_repository,
    )
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DiagnosticResourceCounters",
    "LocalAffordanceDiagnosticCapture",
    "LocalAffordanceDiagnosticCaptureError",
    "capture_local_affordance_diagnostics",
    "main",
    "run_local_affordance_diagnostic_capture",
]
