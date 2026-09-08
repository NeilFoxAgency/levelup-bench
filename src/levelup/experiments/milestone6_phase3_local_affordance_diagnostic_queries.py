"""Descriptor-bound, pre-outcome query matrix for local-affordance diagnostics.

This module is the read-only bridge between the Phase 2 prepared evidence and the
pure reducer in :mod:`milestone6_phase3_local_affordance_diagnostics`.  It consumes
only already validated authority objects, rechecks their seals, and returns
identity-free query objects.  No path is written, and this module does not train,
search, replay, evaluate, or ask an optimum oracle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from levelup.experiments.milestone6_phase2_screening_runtime import (
    ScreeningRuntime,
    recheck_screening_runtime_readonly,
)
from levelup.experiments.milestone6_phase3_evidence import (
    EvidenceLockError,
    Phase3EvidenceLock,
    require_phase3_evidence_lock,
)
from levelup.experiments.milestone6_phase3_local_affordance_capabilities import (
    HeldoutTaskProbeCapability,
    LocalAffordanceCapabilityError,
    TrainingFoldProbeCapability,
    issue_heldout_task_probe_capability,
    issue_training_fold_probe_capability,
)
from levelup.experiments.milestone6_phase3_local_affordance_diagnostics import (
    LocalAffordanceQuery,
)
from levelup.experiments.milestone6_phase3_local_affordance_plan import (
    FAMILIES,
    LocalAffordancePlan,
    LocalAffordancePlanError,
    validate_local_affordance_plan,
)
from levelup.experiments.milestone6_phase3_local_affordance_raw_authority import (
    RawProbeAuthorityError,
    RawProbeAuthoritySnapshot,
    require_raw_probe_authority_snapshot,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataEvidenceKey,
    TrainingDataEvidenceManifest,
    load_training_data_evidence_payload_bundle_from_at,
    open_training_data_reader,
)
from levelup.learning.state_conditioned import ObservableState

TRAINING_QUERY_COUNT = 1_200
HELDOUT_QUERY_COUNT = 240
TRAINING_FOLD_COUNT = 30
HELDOUT_BINDING_COUNT = 240
_B2_CONDITION = "B2-global-listwise-optimum"
_TOKEN = object()


class LocalAffordanceDiagnosticQueryError(ValueError):
    """Raised when the complete pre-outcome query matrix cannot be built."""


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _state_json(value: ObservableState) -> dict[str, Any]:
    return {
        "progress_fraction": value.progress_fraction,
        "remaining_fraction": value.remaining_fraction,
        "elapsed_per_target": value.elapsed_per_target,
        "resource_fraction": value.resource_fraction,
        "pressure_fraction": value.pressure_fraction,
        "available_aliases": list(value.available_aliases),
    }


def _query_json(query: LocalAffordanceQuery) -> dict[str, Any]:
    evidence = query.evidence
    return {
        "population": query.population,
        "family_id": query.family_id,
        "evidence_rows": [
            {
                "probe_index": row.probe_index,
                "before": _state_json(row.transition.before),
                "action_alias": row.transition.action_alias,
                "after": _state_json(row.transition.after),
                "completed": row.transition.completed,
            }
            for row in evidence.rows
        ],
        "affordance_features": {
            alias: list(evidence.affordances.features[alias])
            for alias in sorted(evidence.affordances.features)
        },
        "sample_counts": {
            alias: evidence.affordances.sample_counts[alias]
            for alias in sorted(evidence.affordances.sample_counts)
        },
        "states": [_state_json(state) for state in query.states],
    }


def _query_matrix_sha256(queries: tuple[LocalAffordanceQuery, ...]) -> str:
    return _sha(tuple(_query_json(query) for query in queries))


@dataclass(frozen=True, slots=True)
class _QueryAuthoritySeal:
    query_matrix_sha256: str
    training_query_count: int
    heldout_query_count: int
    training_fold_count: int
    heldout_binding_count: int
    state_query_count: int
    training_state_query_count: int
    heldout_state_query_count: int
    evidence_lock_sha256: str
    raw_authority_content_sha256: str
    plan_id: str
    runtime_manifest_sha256: str
    token: object


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _state(value: Any) -> ObservableState:
    """Copy one persisted observable state into the learner-only representation."""

    try:
        return ObservableState(
            progress_fraction=value.progress_fraction,
            remaining_fraction=value.remaining_fraction,
            elapsed_per_target=value.elapsed_per_target,
            resource_fraction=value.resource_fraction,
            pressure_fraction=value.pressure_fraction,
            available_aliases=tuple(value.available_aliases),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise LocalAffordanceDiagnosticQueryError("payload contains an invalid observable state") from exc


def _lock_rows(lock: Phase3EvidenceLock) -> dict[tuple[str, int], Mapping[str, Any]]:
    try:
        require_phase3_evidence_lock(lock)
    except (EvidenceLockError, TypeError, ValueError) as exc:
        raise LocalAffordanceDiagnosticQueryError("evidence lock is not validator-issued") from exc
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
        raise LocalAffordanceDiagnosticQueryError("evidence lock scope or counts drifted")
    rows = body.get("evidence_artifacts")
    if not isinstance(rows, list) or len(rows) != TRAINING_FOLD_COUNT:
        raise LocalAffordanceDiagnosticQueryError("evidence lock rows are incomplete")
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise LocalAffordanceDiagnosticQueryError("evidence lock row is malformed")
        identity = (row.get("family_id"), row.get("replicate"))
        if identity[0] not in FAMILIES or type(identity[1]) is not int or identity[1] not in range(5):
            raise LocalAffordanceDiagnosticQueryError("evidence lock row identity is invalid")
        if identity in result:
            raise LocalAffordanceDiagnosticQueryError("evidence lock contains duplicate rows")
        result[identity] = row
    if set(result) != {(family, replicate) for family in FAMILIES for replicate in range(5)}:
        raise LocalAffordanceDiagnosticQueryError("evidence lock family/replicate matrix is incomplete")
    return result


def _read_training_payload(
    runtime_fold: Any,
    row: Mapping[str, Any],
    *,
    expected_family: str,
    expected_replicate: int,
) -> tuple[Any, tuple[str, ...]]:
    """Read one exact evidence bundle through the matching pinned run descriptor."""

    try:
        key = TrainingDataEvidenceKey.model_validate(row["evidence_key"])
        manifest = TrainingDataEvidenceManifest.model_validate(row["evidence_manifest"])
        expected_ids = tuple(key.ordered_training_task_ids)
        evidence_id = row["evidence_id"]
        manifest_sha = row["canonical_manifest_bytes_sha256"]
        payload_sha = row["payload_sha256"]
        payload_bytes = row["payload_bytes"]
    except (KeyError, TypeError, ValueError) as exc:
        raise LocalAffordanceDiagnosticQueryError("evidence lock row is not typed") from exc
    if (
        manifest.key != key
        or manifest.evidence_id != evidence_id
        or key.heldout_family_id != expected_family
        or key.fold_id != f"lofo-{expected_family}"
        or key.replicate != expected_replicate
        or manifest.key.heldout_family_id != expected_family
        or manifest.key.fold_id != f"lofo-{expected_family}"
        or manifest.key.replicate != expected_replicate
        or manifest.sample_task_ids != expected_ids
        or manifest.payload_sha256 != payload_sha
        or manifest.payload_bytes != payload_bytes
        or len(expected_ids) != 40
    ):
        raise LocalAffordanceDiagnosticQueryError("evidence manifest lineage differs from lock")
    store = getattr(runtime_fold, "store", None)
    if store is None or not hasattr(store, "_open_pinned_run"):
        raise LocalAffordanceDiagnosticQueryError("runtime fold lacks a pinned run descriptor")
    try:
        with store._open_pinned_run() as run_fd:
            with open_training_data_reader(run_fd) as reader:
                bundle = load_training_data_evidence_payload_bundle_from_at(
                    reader,
                    evidence_id,
                    expected_key=key,
                )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise LocalAffordanceDiagnosticQueryError("descriptor-read evidence payload failed") from exc
    if (
        bundle.manifest != manifest
        or hashlib.sha256(bundle.manifest_bytes).hexdigest() != manifest_sha
        or hashlib.sha256(bundle.payload_bytes).hexdigest() != payload_sha
        or len(bundle.payload_bytes) != payload_bytes
        or tuple(sample.task_id for sample in bundle.payload.samples) != expected_ids
    ):
        raise LocalAffordanceDiagnosticQueryError("descriptor payload bytes or task order differ from lock")
    return bundle.payload, expected_ids


def _runtime_task_families(runtime_fold: Any, task_ids: tuple[str, ...]) -> dict[str, str]:
    try:
        tasks = tuple(runtime_fold.config.split.development_tasks)
    except (AttributeError, TypeError) as exc:
        raise LocalAffordanceDiagnosticQueryError("runtime fold training task manifest is unavailable") from exc
    mapping = {task.task_id: task.family_id for task in tasks}
    if tuple(mapping) != task_ids or set(mapping) != set(task_ids) or len(mapping) != 40:
        raise LocalAffordanceDiagnosticQueryError("runtime fold task order or coverage drifted")
    if any(family not in FAMILIES for family in mapping.values()):
        raise LocalAffordanceDiagnosticQueryError("runtime fold contains a non-development family")
    return mapping


def _training_queries(
    runtime: ScreeningRuntime,
    lock_rows: Mapping[tuple[str, int], Mapping[str, Any]],
    raw_snapshot: RawProbeAuthoritySnapshot,
) -> tuple[LocalAffordanceQuery, ...]:
    folds = {fold.family_id: fold for fold in runtime.folds}
    if tuple(folds) != FAMILIES:
        raise LocalAffordanceDiagnosticQueryError("runtime fold order is not the frozen six-family order")
    queries: list[LocalAffordanceQuery] = []
    for heldout_family in FAMILIES:
        fold = folds[heldout_family]
        for replicate in range(5):
            row = lock_rows[(heldout_family, replicate)]
            payload, task_ids = _read_training_payload(
                fold,
                row,
                expected_family=heldout_family,
                expected_replicate=replicate,
            )
            task_families = _runtime_task_families(fold, task_ids)
            try:
                capability = issue_training_fold_probe_capability(
                    raw_snapshot,
                    fold_id=heldout_family,
                    replicate=replicate,
                )
                if type(capability) is not TrainingFoldProbeCapability:
                    raise LocalAffordanceDiagnosticQueryError("training capability is foreign")
                evidence = capability.consume_for(task_ids)
            except (LocalAffordanceCapabilityError, TypeError, ValueError) as exc:
                raise LocalAffordanceDiagnosticQueryError("training capability consumption failed") from exc
            evidence_by_task = dict(zip(task_ids, evidence, strict=True))
            for sample in payload.samples:
                sample_evidence = evidence_by_task.get(sample.task_id)
                if sample_evidence is None:
                    raise LocalAffordanceDiagnosticQueryError("training evidence task binding is incomplete")
                states = tuple(_state(transition.before) for transition in sample.trace.transitions)
                if not states:
                    raise LocalAffordanceDiagnosticQueryError("optimum trace has no query states")
                try:
                    queries.append(
                        LocalAffordanceQuery(
                            population="training",
                            family_id=task_families[sample.task_id],
                            evidence=sample_evidence,
                            states=states,
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise LocalAffordanceDiagnosticQueryError("training query construction failed") from exc
    if len(queries) != TRAINING_QUERY_COUNT:
        raise LocalAffordanceDiagnosticQueryError("training query count is not exactly 1,200")
    return tuple(queries)


def _heldout_queries(
    plan: LocalAffordancePlan,
    raw_snapshot: RawProbeAuthoritySnapshot,
) -> tuple[LocalAffordanceQuery, ...]:
    if not plan.candidate_tuple_ids or _B2_CONDITION not in getattr(plan, "condition_ids", ()):
        raise LocalAffordanceDiagnosticQueryError("local-affordance plan lacks B2 first tuple authority")
    first_tuple = plan.candidate_tuple_ids[0]
    selected = tuple(
        item
        for item in plan.units
        if item.condition_id == _B2_CONDITION and item.tuple_id == first_tuple
    )
    identities = {(item.heldout_family, item.unit.key.replicate, item.unit.key.task_id) for item in selected}
    expected_identity = {
        (family, replicate, item.unit.key.task_id)
        for family in FAMILIES
        for replicate in range(5)
        for item in plan.units
        if item.heldout_family == family
        and item.unit.key.replicate == replicate
        and item.condition_id == _B2_CONDITION
        and item.tuple_id == first_tuple
    }
    if (
        len(selected) != HELDOUT_BINDING_COUNT
        or len(identities) != HELDOUT_BINDING_COUNT
        or identities != expected_identity
    ):
        raise LocalAffordanceDiagnosticQueryError("heldout B2 first-tuple planned matrix is incomplete")
    queries: list[LocalAffordanceQuery] = []
    for item in selected:
        try:
            capability = issue_heldout_task_probe_capability(raw_snapshot, item.unit)
            if type(capability) is not HeldoutTaskProbeCapability:
                raise LocalAffordanceDiagnosticQueryError("heldout capability is foreign")
            evidence = capability.consume_for(item.unit)
        except (LocalAffordanceCapabilityError, TypeError, ValueError) as exc:
            raise LocalAffordanceDiagnosticQueryError("heldout capability consumption failed") from exc
        states = tuple(_state(row.transition.before) for row in evidence.rows)
        if len(states) != 64:
            raise LocalAffordanceDiagnosticQueryError("heldout probe evidence does not contain exactly 64 states")
        try:
            queries.append(
                LocalAffordanceQuery(
                    population="heldout",
                    family_id=item.heldout_family,
                    evidence=evidence,
                    states=states,
                )
            )
        except (TypeError, ValueError) as exc:
            raise LocalAffordanceDiagnosticQueryError("heldout query construction failed") from exc
    return tuple(queries)


@dataclass(frozen=True, slots=True)
class LocalAffordanceDiagnosticQueryAuthority:
    """Opaque complete training/held-out query matrix and binding digests."""

    queries: tuple[LocalAffordanceQuery, ...]
    training_query_count: int
    heldout_query_count: int
    training_fold_count: int
    heldout_binding_count: int
    state_query_count: int
    training_state_query_count: int
    heldout_state_query_count: int
    evidence_lock_sha256: str
    raw_authority_content_sha256: str
    plan_id: str
    runtime_manifest_sha256: str
    _token: object
    _seal: _QueryAuthoritySeal = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self._token is not _TOKEN:
            raise LocalAffordanceDiagnosticQueryError("query authority requires canonical construction")
        if (
            len(self.queries) != TRAINING_QUERY_COUNT + HELDOUT_QUERY_COUNT
            or self.training_query_count != TRAINING_QUERY_COUNT
            or self.heldout_query_count != HELDOUT_QUERY_COUNT
            or self.training_fold_count != TRAINING_FOLD_COUNT
            or self.heldout_binding_count != HELDOUT_BINDING_COUNT
            or self.state_query_count <= 0
            or self.training_state_query_count <= 0
            or self.heldout_state_query_count <= 0
            or self.state_query_count
            != self.training_state_query_count + self.heldout_state_query_count
        ):
            raise LocalAffordanceDiagnosticQueryError("query authority counts are incomplete")
        if any(
            type(value) is not int
            for value in (
                self.training_query_count,
                self.heldout_query_count,
                self.training_fold_count,
                self.heldout_binding_count,
                self.state_query_count,
                self.training_state_query_count,
                self.heldout_state_query_count,
            )
        ):
            raise LocalAffordanceDiagnosticQueryError("query authority counts must be exact integers")
        if not all(
            _is_sha256(value)
            for value in (
                self.evidence_lock_sha256,
                self.raw_authority_content_sha256,
                self.plan_id,
                self.runtime_manifest_sha256,
            )
        ):
            raise LocalAffordanceDiagnosticQueryError("query authority source digests are malformed")
        if any(type(query) is not LocalAffordanceQuery for query in self.queries):
            raise LocalAffordanceDiagnosticQueryError("query authority contains a foreign query object")
        training = tuple(query for query in self.queries if query.population == "training")
        heldout = tuple(query for query in self.queries if query.population == "heldout")
        if len(training) != TRAINING_QUERY_COUNT or len(heldout) != HELDOUT_QUERY_COUNT:
            raise LocalAffordanceDiagnosticQueryError("query authority population counts are incomplete")
        if any(sum(query.family_id == family for query in training) != 200 for family in FAMILIES):
            raise LocalAffordanceDiagnosticQueryError("training query family counts drifted")
        if any(sum(query.family_id == family for query in heldout) != 40 for family in FAMILIES):
            raise LocalAffordanceDiagnosticQueryError("heldout query family counts drifted")
        actual_training_states = sum(len(query.states) for query in training)
        actual_heldout_states = sum(len(query.states) for query in heldout)
        if (
            self.training_state_query_count != actual_training_states
            or self.heldout_state_query_count != actual_heldout_states
            or self.state_query_count != actual_training_states + actual_heldout_states
        ):
            raise LocalAffordanceDiagnosticQueryError("query authority state counts differ from queries")
        digest = _query_matrix_sha256(self.queries)
        object.__setattr__(
            self,
            "_seal",
            _QueryAuthoritySeal(
                query_matrix_sha256=digest,
                training_query_count=self.training_query_count,
                heldout_query_count=self.heldout_query_count,
                training_fold_count=self.training_fold_count,
                heldout_binding_count=self.heldout_binding_count,
                state_query_count=self.state_query_count,
                training_state_query_count=self.training_state_query_count,
                heldout_state_query_count=self.heldout_state_query_count,
                evidence_lock_sha256=self.evidence_lock_sha256,
                raw_authority_content_sha256=self.raw_authority_content_sha256,
                plan_id=self.plan_id,
                runtime_manifest_sha256=self.runtime_manifest_sha256,
                token=_TOKEN,
            ),
        )

    def require_complete(self) -> "LocalAffordanceDiagnosticQueryAuthority":
        if (
            type(self) is not LocalAffordanceDiagnosticQueryAuthority
            or self._token is not _TOKEN
            or type(getattr(self, "_seal", None)) is not _QueryAuthoritySeal
            or self._seal.token is not _TOKEN
        ):
            raise LocalAffordanceDiagnosticQueryError("query authority is forged")
        if (
            self.training_query_count != TRAINING_QUERY_COUNT
            or self.heldout_query_count != HELDOUT_QUERY_COUNT
            or self.training_fold_count != TRAINING_FOLD_COUNT
            or self.heldout_binding_count != HELDOUT_BINDING_COUNT
            or len(self.queries) != TRAINING_QUERY_COUNT + HELDOUT_QUERY_COUNT
            or len(self.queries) != self.training_query_count + self.heldout_query_count
        ):
            raise LocalAffordanceDiagnosticQueryError("query authority matrix changed")
        if _query_matrix_sha256(self.queries) != self._seal.query_matrix_sha256:
            raise LocalAffordanceDiagnosticQueryError("query authority query matrix changed")
        if any(type(query) is not LocalAffordanceQuery for query in self.queries):
            raise LocalAffordanceDiagnosticQueryError("query authority contains a foreign query object")
        if (
            self.state_query_count != self._seal.state_query_count
            or self.training_state_query_count != self._seal.training_state_query_count
            or self.heldout_state_query_count != self._seal.heldout_state_query_count
            or self.state_query_count
            != self.training_state_query_count + self.heldout_state_query_count
        ):
            raise LocalAffordanceDiagnosticQueryError("query authority state counts changed")
        if (
            self.training_query_count != self._seal.training_query_count
            or self.heldout_query_count != self._seal.heldout_query_count
            or self.training_fold_count != self._seal.training_fold_count
            or self.heldout_binding_count != self._seal.heldout_binding_count
            or self.evidence_lock_sha256 != self._seal.evidence_lock_sha256
            or self.raw_authority_content_sha256 != self._seal.raw_authority_content_sha256
            or self.plan_id != self._seal.plan_id
            or self.runtime_manifest_sha256 != self._seal.runtime_manifest_sha256
        ):
            raise LocalAffordanceDiagnosticQueryError("query authority metadata changed")
        training = tuple(query for query in self.queries if query.population == "training")
        heldout = tuple(query for query in self.queries if query.population == "heldout")
        actual_training_states = sum(len(query.states) for query in training)
        actual_heldout_states = sum(len(query.states) for query in heldout)
        if (
            len(training) != TRAINING_QUERY_COUNT
            or len(heldout) != HELDOUT_QUERY_COUNT
            or any(sum(query.family_id == family for query in training) != 200 for family in FAMILIES)
            or any(sum(query.family_id == family for query in heldout) != 40 for family in FAMILIES)
            or actual_training_states != self.training_state_query_count
            or actual_heldout_states != self.heldout_state_query_count
        ):
            raise LocalAffordanceDiagnosticQueryError("query authority family coverage changed")
        return self

    @property
    def query_count(self) -> int:
        return self.training_query_count + self.heldout_query_count

    def model_dump(self) -> dict[str, Any]:
        """Return authority metadata only; learner query identities are not exported."""

        self.require_complete()
        return {
            "schema_version": "milestone6.phase3.local-affordance-query-authority.v1",
            "training_query_count": self.training_query_count,
            "heldout_query_count": self.heldout_query_count,
            "training_fold_count": self.training_fold_count,
            "heldout_binding_count": self.heldout_binding_count,
            "state_query_count": self.state_query_count,
            "training_state_query_count": self.training_state_query_count,
            "heldout_state_query_count": self.heldout_state_query_count,
            "query_matrix_sha256": self._seal.query_matrix_sha256,
            "evidence_lock_sha256": self.evidence_lock_sha256,
            "raw_authority_content_sha256": self.raw_authority_content_sha256,
            "plan_id": self.plan_id,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
        }


def build_local_affordance_diagnostic_query_authority(
    runtime: ScreeningRuntime,
    evidence_lock: Phase3EvidenceLock,
    raw_snapshot: RawProbeAuthoritySnapshot,
    plan: LocalAffordancePlan,
) -> LocalAffordanceDiagnosticQueryAuthority:
    """Build the exact 1,440-query development-only matrix from sealed inputs."""

    if type(runtime) is not ScreeningRuntime:
        raise LocalAffordanceDiagnosticQueryError("query authority requires an exact ScreeningRuntime")
    if type(plan) is not LocalAffordancePlan:
        raise LocalAffordanceDiagnosticQueryError("query authority requires an exact LocalAffordancePlan")
    try:
        recheck_screening_runtime_readonly(runtime)
        validate_local_affordance_plan(plan)
        require_raw_probe_authority_snapshot(raw_snapshot)
        rows = _lock_rows(evidence_lock)
    except (OSError, RuntimeError, TypeError, ValueError, LocalAffordancePlanError, RawProbeAuthorityError) as exc:
        raise LocalAffordanceDiagnosticQueryError("query authority input seal recheck failed") from exc
    if (
        raw_snapshot.manifest.manifest_id != plan.raw_authority_manifest_id
        or raw_snapshot.authority_content_sha256 != plan.raw_authority_content_sha256
        or raw_snapshot.manifest_file.snapshot.sha256 != plan.raw_manifest_file_sha256
        or raw_snapshot.manifest.phase3_evidence_lock_sha256 != evidence_lock.evidence_lock_sha256
    ):
        raise LocalAffordanceDiagnosticQueryError("raw authority digests differ from the frozen plan or lock")
    training = _training_queries(runtime, rows, raw_snapshot)
    heldout = _heldout_queries(plan, raw_snapshot)
    queries = training + heldout
    training_state_count = sum(len(query.states) for query in training)
    heldout_state_count = sum(len(query.states) for query in heldout)
    state_count = training_state_count + heldout_state_count
    manifest_sha = getattr(runtime.manifest, "manifest_sha256", None)
    if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
        raise LocalAffordanceDiagnosticQueryError("runtime manifest digest is missing")
    authority_sha = getattr(raw_snapshot, "authority_content_sha256", None)
    if not isinstance(authority_sha, str) or len(authority_sha) != 64:
        raise LocalAffordanceDiagnosticQueryError("raw authority digest is missing")
    return LocalAffordanceDiagnosticQueryAuthority(
        queries=queries,
        training_query_count=len(training),
        heldout_query_count=len(heldout),
        training_fold_count=TRAINING_FOLD_COUNT,
        heldout_binding_count=HELDOUT_BINDING_COUNT,
        state_query_count=state_count,
        training_state_query_count=training_state_count,
        heldout_state_query_count=heldout_state_count,
        evidence_lock_sha256=evidence_lock.evidence_lock_sha256,
        raw_authority_content_sha256=authority_sha,
        plan_id=plan.plan_id,
        runtime_manifest_sha256=manifest_sha,
        _token=_TOKEN,
    )


# Short descriptive alias for callers that do not need the longer rung name.
build_diagnostic_query_authority = build_local_affordance_diagnostic_query_authority


__all__ = [
    "HELDOUT_BINDING_COUNT",
    "HELDOUT_QUERY_COUNT",
    "LocalAffordanceDiagnosticQueryAuthority",
    "LocalAffordanceDiagnosticQueryError",
    "TRAINING_FOLD_COUNT",
    "TRAINING_QUERY_COUNT",
    "build_local_affordance_diagnostic_query_authority",
    "build_diagnostic_query_authority",
]
