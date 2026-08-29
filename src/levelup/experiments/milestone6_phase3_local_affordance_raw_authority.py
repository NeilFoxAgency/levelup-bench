"""Complete, read-only authority gate for Phase 3 local-affordance raw evidence.

The gate validates the exact development-only 240/240/30/240 store through
descriptors pinned by :mod:`milestone6_phase3_local_affordance_raw_store`.
It returns immutable file snapshots only.  It never creates evidence, issues a
reducer capability, opens an environment, searches, evaluates, or authorizes
execution.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, StrictInt, model_validator

from levelup.experiments.milestone6_phase3_local_affordance_evidence import (
    FAMILY_ORDER,
    RawProbeArtifactBody,
    RawProbeArtifactKey,
    RawProbeArtifactManifest,
    RawProbeTransitionRecord,
)
from levelup.experiments.milestone6_phase3_local_affordance_raw_store import (
    ARTIFACTS_DIR,
    HELDOUT_BINDINGS_DIR,
    KEYS_DIR,
    TRAINING_FOLDS_DIR,
    HeldoutProbeBinding,
    PinnedRawProbeStoreReader,
    RawProbeStoreError,
    RawProbeStoreManifest,
    RawProbeTaskKeyIndex,
    RawProbeTaskReference,
    StableDirectoryIdentity,
    StableFileIdentity,
    StableFileSnapshot,
    TrainingFoldManifest,
    _file_identity,
    _stable_file_snapshot_at,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import AffordanceTableRecord
from levelup.learning.state_conditioned import (
    ObservableState,
    ObservedTransition,
    build_affordance_table,
)

PERSISTED_ARTIFACT_VERSION = "milestone6.phase3.local-affordance-persisted-artifact.v1"
AUTHORITY_SNAPSHOT_VERSION = "milestone6.phase3.local-affordance-raw-authority.v1"
HASH = Field(pattern=r"^[0-9a-f]{64}$")
REPLICATES = (0, 1, 2, 3, 4)
EXPECTED_TASKS_PER_FAMILY = 8
# These are the exact committed development-only sources present when the raw
# layout and authority gate were frozen, before raw capture or comparative
# local-affordance results.  A coherent but substituted set of source files is
# not an acceptable authority.
FROZEN_LOCAL_AFFORDANCE_PROTOCOL_SHA256 = (
    "a5b97f793cc72692943e44e7497f79e3e5528e65abd4badfa3b98c44e27896c2"
)
FROZEN_DEVELOPMENT_PROTOCOL_SHA256 = (
    "7e6911c120db091e2b250f7a91520dd5f81a481cb4a19662eeae858c7da1c059"
)
FROZEN_DEVELOPMENT_TASKS_SHA256 = "20f6606bd2150d808b18f011976bbf7c8298627e1cc01eeb67f653eacba9731f"
FROZEN_PHASE3_EVIDENCE_LOCK_FILE_SHA256 = (
    "82644954b94bd6ff495c425ffe921d18157a98ccbe230d922c24218a4faad875"
)
FROZEN_PHASE3_EVIDENCE_LOCK_SHA256 = (
    "7db4ad251f1e20ea14902c2643425ebfc2ef1064c4ea5ca5c90eb0629c2386b3"
)
FROZEN_PROBE_POLICY_SHA256 = "f44950c1d3317acc3d5518675488448c310a6bb15900644f681319677739db20"
_EXPECTED_TOKEN = object()
_SNAPSHOT_TOKEN = object()
_ALLOWED_DEVELOPMENT_ROLES = frozenset(
    {"known_development", "training_core", "historical_milestone5_development"}
)


class RawProbeAuthorityError(RawProbeStoreError):
    """Raised when the complete raw-evidence authority fails closed."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object) -> str:
    return _sha256(canonical_json_bytes(value))


def _json_object(content: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RawProbeAuthorityError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RawProbeAuthorityError(f"{label} must be a JSON object")
    return value


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise RawProbeAuthorityError(f"{label} must be an exact integer")
    return value


def _observable_state(record: Any) -> ObservableState:
    return ObservableState(
        progress_fraction=record.progress_fraction,
        remaining_fraction=record.remaining_fraction,
        elapsed_per_target=record.elapsed_per_target,
        resource_fraction=record.resource_fraction,
        pressure_fraction=record.pressure_fraction,
        available_aliases=record.available_aliases,
    )


def _observed_transition(row: RawProbeTransitionRecord) -> ObservedTransition:
    return ObservedTransition(
        before=_observable_state(row.before),
        action_alias=row.action_alias,
        after=_observable_state(row.after),
        completed=row.completed,
    )


def _rebuilt_affordances(body: RawProbeArtifactBody) -> AffordanceTableRecord:
    table = build_affordance_table(
        tuple(_observed_transition(row) for row in body.rows),
        target_samples_per_alias=8,
    )
    return AffordanceTableRecord(
        features={alias: tuple(values) for alias, values in table.features.items()},
        sample_counts=dict(table.sample_counts),
    )


class PersistedRawProbeArtifact(BaseModel):
    """Capability-free persisted envelope for one exact 64-row task probe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[PERSISTED_ARTIFACT_VERSION] = PERSISTED_ARTIFACT_VERSION
    key: RawProbeArtifactKey
    body: RawProbeArtifactBody
    manifest: RawProbeArtifactManifest
    affordances: AffordanceTableRecord

    @model_validator(mode="after")
    def identities_and_pooled_parity_are_exact(self) -> "PersistedRawProbeArtifact":
        if self.manifest.key != self.key or self.manifest.key_id != self.key.key_id:
            raise ValueError("persisted artifact key and manifest differ")
        if self.manifest.body_sha256 != self.body.content_sha256:
            raise ValueError("persisted artifact body and manifest differ")
        rebuilt = _rebuilt_affordances(self.body)
        if canonical_json_bytes(rebuilt.model_dump(mode="json")) != canonical_json_bytes(
            self.affordances.model_dump(mode="json")
        ):
            raise ValueError("persisted pooled affordances differ from raw rows")
        pooled_sha256 = _digest(self.affordances.model_dump(mode="json"))
        if self.manifest.pooled_affordance_sha256 != pooled_sha256:
            raise ValueError("persisted pooled affordance digest differs from manifest")
        return self


class ExpectedDevelopmentTask(BaseModel):
    """One selected task identity from the committed development manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family_id: str
    task_id: str
    task_index: StrictInt = Field(ge=0)
    generator_seed: StrictInt = Field(ge=0)
    environment_seed: StrictInt = Field(ge=0)


@dataclass(frozen=True, slots=True)
class _ExpectedRawProbeAuthoritySeal:
    content_sha256: str
    token: object


def _expected_authority_content(
    *,
    manifest: RawProbeStoreManifest,
    selected_tasks: tuple[ExpectedDevelopmentTask, ...],
    keys: tuple[RawProbeArtifactKey, ...],
    evidence_lock_file_sha256: str,
    key_filenames: tuple[str, ...],
    training_fold_filenames: tuple[str, ...],
    heldout_binding_filenames: tuple[str, ...],
) -> dict[str, object]:
    return {
        "manifest": manifest.model_dump(mode="json"),
        "selected_tasks": tuple(task.model_dump(mode="json") for task in selected_tasks),
        "keys": tuple(key.model_dump(mode="json") for key in keys),
        "evidence_lock_file_sha256": evidence_lock_file_sha256,
        "key_filenames": key_filenames,
        "training_fold_filenames": training_fold_filenames,
        "heldout_binding_filenames": heldout_binding_filenames,
    }


@dataclass(frozen=True, slots=True, init=False)
class ExpectedRawProbeAuthority:
    """Opaque deterministic expectation built from the four frozen source files."""

    manifest: RawProbeStoreManifest
    selected_tasks: tuple[ExpectedDevelopmentTask, ...]
    keys: tuple[RawProbeArtifactKey, ...]
    evidence_lock_file_sha256: str
    key_filenames: tuple[str, ...]
    training_fold_filenames: tuple[str, ...]
    heldout_binding_filenames: tuple[str, ...]
    _seal: _ExpectedRawProbeAuthoritySeal
    _token: object

    def __init__(
        self,
        *,
        manifest: RawProbeStoreManifest,
        selected_tasks: tuple[ExpectedDevelopmentTask, ...],
        keys: tuple[RawProbeArtifactKey, ...],
        evidence_lock_file_sha256: str,
        _token: object | None = None,
    ) -> None:
        if _token is not _EXPECTED_TOKEN:
            raise RawProbeAuthorityError(
                "expected raw authority requires validated frozen source bytes"
            )
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "selected_tasks", selected_tasks)
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "evidence_lock_file_sha256", evidence_lock_file_sha256)
        key_filenames = tuple(f"{key.key_id}.json" for key in keys)
        object.__setattr__(self, "key_filenames", key_filenames)
        training_fold_filenames = tuple(
            f"{family}.r{replicate}.json" for family in FAMILY_ORDER for replicate in REPLICATES
        )
        object.__setattr__(
            self,
            "training_fold_filenames",
            training_fold_filenames,
        )
        heldout_binding_filenames = tuple(
            f"{task.family_id}.r{replicate}.task-{task.task_index}.json"
            for task in selected_tasks
            for replicate in REPLICATES
        )
        object.__setattr__(
            self,
            "heldout_binding_filenames",
            heldout_binding_filenames,
        )
        object.__setattr__(
            self,
            "_seal",
            _ExpectedRawProbeAuthoritySeal(
                content_sha256=_digest(
                    _expected_authority_content(
                        manifest=manifest,
                        selected_tasks=selected_tasks,
                        keys=keys,
                        evidence_lock_file_sha256=evidence_lock_file_sha256,
                        key_filenames=key_filenames,
                        training_fold_filenames=training_fold_filenames,
                        heldout_binding_filenames=heldout_binding_filenames,
                    )
                ),
                token=_EXPECTED_TOKEN,
            ),
        )
        object.__setattr__(self, "_token", _EXPECTED_TOKEN)


class AuthorityFileRecord(BaseModel):
    """One immutable descriptor-read file in the complete authority snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    namespace: Literal["root", "artifacts", "keys", "training-folds", "heldout-bindings"]
    name: str = Field(min_length=1)
    snapshot: StableFileSnapshot


@dataclass(frozen=True, slots=True)
class _RawProbeAuthoritySnapshotSeal:
    snapshot_sha256: str
    token: object


class RawProbeAuthoritySnapshot(BaseModel):
    """Immutable authority output; intentionally contains no read capability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[AUTHORITY_SNAPSHOT_VERSION] = AUTHORITY_SNAPSHOT_VERSION
    manifest: RawProbeStoreManifest
    evidence_lock_file_sha256: str = HASH
    authority_content_sha256: str = HASH
    directory_identities: tuple[StableDirectoryIdentity, ...]
    manifest_file: AuthorityFileRecord
    artifact_files: tuple[AuthorityFileRecord, ...]
    key_files: tuple[AuthorityFileRecord, ...]
    training_fold_files: tuple[AuthorityFileRecord, ...]
    heldout_binding_files: tuple[AuthorityFileRecord, ...]
    key_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    _seal: _RawProbeAuthoritySnapshotSeal = PrivateAttr()

    def __init__(self, *, _token: object | None = None, **data: Any) -> None:
        if _token is not _SNAPSHOT_TOKEN:
            raise RawProbeAuthorityError("raw authority snapshots require the complete validator")
        super().__init__(**data)
        object.__setattr__(
            self,
            "_seal",
            _RawProbeAuthoritySnapshotSeal(
                snapshot_sha256=_digest(self.model_dump(mode="json")),
                token=_SNAPSHOT_TOKEN,
            ),
        )

    @model_validator(mode="after")
    def exact_counts(self) -> "RawProbeAuthoritySnapshot":
        if (
            len(self.artifact_files) != 240
            or len(self.key_files) != 240
            or len(self.training_fold_files) != 30
            or len(self.heldout_binding_files) != 240
            or len(self.key_ids) != 240
            or len(self.artifact_ids) != 240
        ):
            raise ValueError("raw authority snapshot matrix is incomplete")
        content_identity = {
            "manifest": (self.manifest_file.name, self.manifest_file.snapshot.sha256),
            "artifacts": tuple(
                (record.name, record.snapshot.sha256) for record in self.artifact_files
            ),
            "keys": tuple((record.name, record.snapshot.sha256) for record in self.key_files),
            "training_folds": tuple(
                (record.name, record.snapshot.sha256) for record in self.training_fold_files
            ),
            "heldout_bindings": tuple(
                (record.name, record.snapshot.sha256) for record in self.heldout_binding_files
            ),
        }
        if self.authority_content_sha256 != _digest(content_identity):
            raise ValueError("raw authority content digest mismatch")
        return self


def _require_canonical_source(content: bytes, label: str) -> dict[str, Any]:
    body = _json_object(content, label)
    canonical = canonical_json_bytes(body)
    if content not in (canonical, canonical + b"\n"):
        raise RawProbeAuthorityError(f"{label} bytes are not canonical JSON")
    return body


def _validate_development_sources(
    *,
    local_protocol: dict[str, Any],
    development_protocol: dict[str, Any],
    development_tasks: dict[str, Any],
    development_protocol_sha256: str,
    development_tasks_sha256: str,
) -> tuple[ExpectedDevelopmentTask, ...]:
    if (
        local_protocol.get("scope") != "known-development-only"
        or local_protocol.get("execution") is not False
        or local_protocol.get("status") != "frozen-design-only"
    ):
        raise RawProbeAuthorityError("local-affordance protocol is not frozen development-only")
    freeze = local_protocol.get("freeze_record")
    if (
        not isinstance(freeze, dict)
        or freeze.get("comparative_results_inspected_before_freeze") is not False
    ):
        raise RawProbeAuthorityError("local-affordance protocol was not frozen before results")
    authority = local_protocol.get("authority")
    if not isinstance(authority, dict):
        raise RawProbeAuthorityError("local-affordance authority block is missing")
    if (
        authority.get("development_protocol", {}).get("sha256") != development_protocol_sha256
        or authority.get("development_tasks", {}).get("sha256") != development_tasks_sha256
    ):
        raise RawProbeAuthorityError("local-affordance source hashes differ from supplied bytes")
    if (
        development_protocol.get("scope") != "known-development-only"
        or development_protocol.get("family_order") != list(FAMILY_ORDER)
        or development_protocol.get("final_family_access") != "forbidden_until_phase9_method_freeze"
    ):
        raise RawProbeAuthorityError("development protocol scope or family order drifted")
    seed_policy = development_protocol.get("seed_policy")
    if not isinstance(seed_policy, dict) or (
        seed_policy.get("screening_replicates") != list(REPLICATES)
        or seed_policy.get("family_offset_stride") != 10_000
        or seed_policy.get("replicate_stride") != 100_000
        or seed_policy.get("task_component") != "manifest task_index"
        or seed_policy.get("bases", {}).get("probe") != 6_200_000
    ):
        raise RawProbeAuthorityError("development probe seed policy drifted")
    if development_tasks.get("family_order") != list(FAMILY_ORDER):
        raise RawProbeAuthorityError("development task family order drifted")
    if development_tasks.get("environment_reset_seed") != 0:
        raise RawProbeAuthorityError("development environment reset seed drifted")
    generator_seeds = development_tasks.get("generator_seeds")
    tasks = development_tasks.get("tasks")
    if not isinstance(generator_seeds, dict) or not isinstance(tasks, list):
        raise RawProbeAuthorityError("development task manifest is malformed")
    selected: list[ExpectedDevelopmentTask] = []
    for entry in tasks:
        if not isinstance(entry, dict) or set(entry) != {
            "family",
            "task_id",
            "task_index",
            "generator_seed",
            "environment_reset_seed",
            "roles",
        }:
            raise RawProbeAuthorityError("development task row schema drifted")
        roles = entry["roles"]
        if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
            raise RawProbeAuthorityError("development task roles are malformed")
        if not set(roles) <= _ALLOWED_DEVELOPMENT_ROLES:
            raise RawProbeAuthorityError(
                "development task manifest contains a non-development role"
            )
        if "training_core" not in roles:
            continue
        if "known_development" not in roles:
            raise RawProbeAuthorityError("training-core task is not known development")
        family = entry["family"]
        if family not in FAMILY_ORDER:
            raise RawProbeAuthorityError("training-core task family is unknown")
        task_index = _exact_int(entry["task_index"], "task index")
        generator_seed = _exact_int(entry["generator_seed"], "generator seed")
        environment_seed = _exact_int(entry["environment_reset_seed"], "environment seed")
        if generator_seeds.get(family) != generator_seed or environment_seed != 0:
            raise RawProbeAuthorityError("selected task generator/environment seed drifted")
        task_id = entry["task_id"]
        if not isinstance(task_id, str) or not task_id:
            raise RawProbeAuthorityError("selected task id is malformed")
        selected.append(
            ExpectedDevelopmentTask(
                family_id=family,
                task_id=task_id,
                task_index=task_index,
                generator_seed=generator_seed,
                environment_seed=environment_seed,
            )
        )
    if len(selected) != len(FAMILY_ORDER) * EXPECTED_TASKS_PER_FAMILY:
        raise RawProbeAuthorityError("development training-core matrix is not 6 x 8")
    expected_order = sorted(
        selected,
        key=lambda task: (FAMILY_ORDER.index(task.family_id), task.task_index, task.task_id),
    )
    if selected != expected_order:
        raise RawProbeAuthorityError("development training-core rows are not canonical")
    by_family = {
        family: [task.task_index for task in selected if task.family_id == family]
        for family in FAMILY_ORDER
    }
    if any(len(values) != EXPECTED_TASKS_PER_FAMILY for values in by_family.values()):
        raise RawProbeAuthorityError("development families do not each have eight selected tasks")
    raw_authority = local_protocol.get("raw_probe_evidence_authority")
    layout = raw_authority.get("raw_store_layout") if isinstance(raw_authority, dict) else None
    if not isinstance(layout, dict) or layout.get("selected_task_indices_by_family") != by_family:
        raise RawProbeAuthorityError("local-affordance selected task indices drifted")
    if layout.get("root_exact_shape") != [
        "manifest.json",
        "artifacts/",
        "keys/",
        "training-folds/",
        "heldout-bindings/",
    ] or layout.get("namespace_entry_counts") != {
        "artifacts": 240,
        "keys": 240,
        "training-folds": 30,
        "heldout-bindings": 240,
    }:
        raise RawProbeAuthorityError("local-affordance raw-store layout drifted")
    return tuple(selected)


def build_expected_raw_probe_authority(
    *,
    local_affordance_protocol_bytes: bytes,
    development_protocol_bytes: bytes,
    development_tasks_bytes: bytes,
    phase3_evidence_lock_bytes: bytes,
) -> ExpectedRawProbeAuthority:
    """Build the exact authority expectation from frozen development-only bytes."""

    inputs = (
        local_affordance_protocol_bytes,
        development_protocol_bytes,
        development_tasks_bytes,
        phase3_evidence_lock_bytes,
    )
    if any(type(content) is not bytes for content in inputs):
        raise RawProbeAuthorityError("raw authority source inputs must be exact bytes")
    # The human-maintained protocol/manifest files are intentionally pretty
    # printed; their exact file bytes are hash-bound below.  The generated
    # evidence lock, by contrast, is a canonical machine authority.
    local_protocol = _json_object(local_affordance_protocol_bytes, "local-affordance protocol")
    development_protocol = _json_object(development_protocol_bytes, "development protocol")
    development_tasks = _json_object(development_tasks_bytes, "development task manifest")
    evidence_lock = _require_canonical_source(phase3_evidence_lock_bytes, "Phase 3 evidence lock")
    local_sha256 = _sha256(local_affordance_protocol_bytes)
    development_protocol_sha256 = _sha256(development_protocol_bytes)
    development_tasks_sha256 = _sha256(development_tasks_bytes)
    evidence_lock_file_sha256 = _sha256(phase3_evidence_lock_bytes)
    if (
        local_sha256 != FROZEN_LOCAL_AFFORDANCE_PROTOCOL_SHA256
        or development_protocol_sha256 != FROZEN_DEVELOPMENT_PROTOCOL_SHA256
        or development_tasks_sha256 != FROZEN_DEVELOPMENT_TASKS_SHA256
        or evidence_lock_file_sha256 != FROZEN_PHASE3_EVIDENCE_LOCK_FILE_SHA256
    ):
        raise RawProbeAuthorityError("raw authority source bytes differ from frozen commits")
    selected = _validate_development_sources(
        local_protocol=local_protocol,
        development_protocol=development_protocol,
        development_tasks=development_tasks,
        development_protocol_sha256=development_protocol_sha256,
        development_tasks_sha256=development_tasks_sha256,
    )
    raw_authority = local_protocol["raw_probe_evidence_authority"]
    probe = raw_authority.get("probe_policy")
    if not isinstance(probe, dict) or (
        probe.get("probe_actions_per_task") != 64
        or probe.get("target_samples_per_alias") != 8
        or probe.get("probe_actions_per_attempt") != 16
        or probe.get("all_conditions_same_rows") is not True
    ):
        raise RawProbeAuthorityError("local-affordance paid-probe policy drifted")
    probe_policy_sha256 = _digest(
        {
            "builder": "canonical-paid-probe-v1",
            "action_cap": 64,
            "coverage_target": 8,
            "actions_per_attempt": 16,
        }
    )
    if probe_policy_sha256 != FROZEN_PROBE_POLICY_SHA256:
        raise RawProbeAuthorityError("frozen paid-probe policy digest drifted")
    supplied_lock_sha256 = evidence_lock.get("evidence_lock_sha256")
    unsigned_lock = dict(evidence_lock)
    unsigned_lock.pop("evidence_lock_sha256", None)
    if (
        evidence_lock.get("schema_version") != "milestone6.phase3.evidence-lock.v1"
        or evidence_lock.get("scope") != "known-development-only"
        or evidence_lock.get("final_family_access") is not False
        or evidence_lock.get("aggregates") != []
        or evidence_lock.get("final_results") != []
        or evidence_lock.get("outcomes_included") is not False
        or evidence_lock.get("payloads_included") is not False
        or supplied_lock_sha256 != _digest(unsigned_lock)
        or supplied_lock_sha256 != FROZEN_PHASE3_EVIDENCE_LOCK_SHA256
    ):
        raise RawProbeAuthorityError("Phase 3 evidence lock is not development-only canonical")
    evidence_rows = evidence_lock.get("evidence_artifacts")
    if (
        evidence_lock.get("counts") != {"evidence_artifacts": 30, "families": 6, "replicates": 5}
        or not isinstance(evidence_rows, list)
        or len(evidence_rows) != 30
    ):
        raise RawProbeAuthorityError("Phase 3 evidence lock matrix is incomplete")
    observed_evidence_pairs: set[tuple[str, int]] = set()
    for row in evidence_rows:
        if not isinstance(row, dict):
            raise RawProbeAuthorityError("Phase 3 evidence lock row is malformed")
        key = row.get("evidence_key")
        family = row.get("family_id")
        replicate = row.get("replicate")
        if family not in FAMILY_ORDER or type(replicate) is not int or replicate not in REPLICATES:
            raise RawProbeAuthorityError("Phase 3 evidence fold identity drifted")
        observed_evidence_pairs.add((family, replicate))
        expected_training = tuple(task for task in selected if task.family_id != family)
        expected_heldout = tuple(task for task in selected if task.family_id == family)
        if not isinstance(key, dict) or (
            key.get("probe_policy_sha256") != probe_policy_sha256
            or key.get("protocol_sha256") != development_protocol_sha256
            or key.get("task_manifest_sha256") != development_tasks_sha256
            or row.get("fold_id") != f"lofo-{family}"
            or key.get("fold_id") != f"lofo-{family}"
            or key.get("heldout_family_id") != family
            or key.get("replicate") != replicate
            or key.get("ordered_training_task_ids") != [task.task_id for task in expected_training]
            or key.get("ordered_heldout_task_ids") != [task.task_id for task in expected_heldout]
            or key.get("environment_seeds") != [task.environment_seed for task in expected_training]
            or key.get("probe_seeds")
            != [
                6_200_000
                + FAMILY_ORDER.index(task.family_id) * 10_000
                + replicate * 100_000
                + task.task_index
                for task in expected_training
            ]
        ):
            raise RawProbeAuthorityError("Phase 3 evidence source identity drifted")
    if observed_evidence_pairs != {
        (family, replicate) for family in FAMILY_ORDER for replicate in REPLICATES
    }:
        raise RawProbeAuthorityError("Phase 3 evidence fold matrix is duplicated or incomplete")
    keys = tuple(
        RawProbeArtifactKey(
            local_affordance_protocol_sha256=local_sha256,
            development_protocol_sha256=development_protocol_sha256,
            development_tasks_sha256=development_tasks_sha256,
            phase3_evidence_lock_sha256=supplied_lock_sha256,
            probe_policy_sha256=probe_policy_sha256,
            family_id=task.family_id,
            replicate=replicate,
            task_index=task.task_index,
            task_id=task.task_id,
            generator_seed=task.generator_seed,
            probe_seed=(
                6_200_000
                + FAMILY_ORDER.index(task.family_id) * 10_000
                + replicate * 100_000
                + task.task_index
            ),
            environment_seed=task.environment_seed,
        )
        for family in FAMILY_ORDER
        for replicate in REPLICATES
        for task in selected
        if task.family_id == family
    )
    manifest = RawProbeStoreManifest.from_authority_hashes(
        local_affordance_protocol_sha256=local_sha256,
        development_protocol_sha256=development_protocol_sha256,
        development_tasks_sha256=development_tasks_sha256,
        phase3_evidence_lock_sha256=supplied_lock_sha256,
        probe_policy_sha256=probe_policy_sha256,
    )
    return ExpectedRawProbeAuthority(
        manifest=manifest,
        selected_tasks=selected,
        keys=keys,
        evidence_lock_file_sha256=_sha256(phase3_evidence_lock_bytes),
        _token=_EXPECTED_TOKEN,
    )


def require_expected_raw_probe_authority(
    expected: ExpectedRawProbeAuthority,
) -> ExpectedRawProbeAuthority:
    """Require an expectation built from the exact committed frozen sources."""

    try:
        seal = expected._seal
        content_sha256 = _digest(
            _expected_authority_content(
                manifest=expected.manifest,
                selected_tasks=expected.selected_tasks,
                keys=expected.keys,
                evidence_lock_file_sha256=expected.evidence_lock_file_sha256,
                key_filenames=expected.key_filenames,
                training_fold_filenames=expected.training_fold_filenames,
                heldout_binding_filenames=expected.heldout_binding_filenames,
            )
        )
    except (AttributeError, TypeError, ValueError):
        raise RawProbeAuthorityError("expected raw authority is not canonical") from None
    if (
        type(expected) is not ExpectedRawProbeAuthority
        or expected._token is not _EXPECTED_TOKEN
        or type(seal) is not _ExpectedRawProbeAuthoritySeal
        or seal.token is not _EXPECTED_TOKEN
        or seal.content_sha256 != content_sha256
        or len(expected.selected_tasks) != 48
        or len(expected.keys) != 240
        or len(set(expected.key_filenames)) != 240
    ):
        raise RawProbeAuthorityError("expected raw authority is not canonical")
    return expected


def require_raw_probe_authority_snapshot(
    snapshot: RawProbeAuthoritySnapshot,
) -> RawProbeAuthoritySnapshot:
    """Require an immutable snapshot issued by the complete authority validator."""

    seal = getattr(snapshot, "_seal", None)
    if (
        type(snapshot) is not RawProbeAuthoritySnapshot
        or type(seal) is not _RawProbeAuthoritySnapshotSeal
        or seal.token is not _SNAPSHOT_TOKEN
        or seal.snapshot_sha256 != _digest(snapshot.model_dump(mode="json"))
    ):
        raise RawProbeAuthorityError("raw authority snapshot is not validator-issued")
    return snapshot


def _root_shape(reader: PinnedRawProbeStoreReader) -> None:
    expected_directories = {ARTIFACTS_DIR, KEYS_DIR, TRAINING_FOLDS_DIR, HELDOUT_BINDINGS_DIR}
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    try:
        with os.scandir(reader.root_fd) as iterator:
            for entry in iterator:
                if entry.is_symlink():
                    raise RawProbeAuthorityError("raw-store root contains a symlink")
                if entry.is_file(follow_symlinks=False):
                    observed_files.add(entry.name)
                elif entry.is_dir(follow_symlinks=False):
                    observed_directories.add(entry.name)
                else:
                    raise RawProbeAuthorityError("raw-store root contains a special entry")
    except RawProbeAuthorityError:
        raise
    except (NotImplementedError, OSError) as exc:
        raise RawProbeAuthorityError("cannot enumerate raw-store root") from exc
    if observed_files != {"manifest.json"} or observed_directories != expected_directories:
        raise RawProbeAuthorityError("raw-store root shape differs from frozen authority")


def _namespace_names(
    directory_fd: int, expected_names: tuple[str, ...], label: str
) -> tuple[str, ...]:
    try:
        observed = secure_fs.strict_regular_entries(directory_fd)
    except secure_fs.SecureFilesystemError as exc:
        raise RawProbeAuthorityError(f"{label} namespace is unsafe") from exc
    if observed != tuple(sorted(expected_names)):
        raise RawProbeAuthorityError(f"{label} namespace inventory differs from authority")
    return observed


def _parse_snapshot(
    directory_fd: int,
    name: str,
    model: type[BaseModel],
    namespace: Literal["root", "artifacts", "keys", "training-folds", "heldout-bindings"],
) -> tuple[BaseModel, AuthorityFileRecord]:
    snapshot = _stable_file_snapshot_at(directory_fd, name)
    try:
        parsed = model.model_validate_json(snapshot.canonical_bytes)
    except (TypeError, ValueError) as exc:
        raise RawProbeAuthorityError(f"invalid {namespace} authority file: {name}") from exc
    return parsed, AuthorityFileRecord(namespace=namespace, name=name, snapshot=snapshot)


def _reference(index: RawProbeTaskKeyIndex) -> RawProbeTaskReference:
    return RawProbeTaskReference(
        artifact_id=index.artifact_id,
        key_id=index.key_id,
        key=index.key,
    )


def _identity_now(directory_fd: int, name: str) -> StableFileIdentity:
    try:
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise RawProbeAuthorityError(f"authority file disappeared: {name}") from exc
    return _file_identity(observed)


def _recheck_files(
    directory_fd: int,
    records: tuple[AuthorityFileRecord, ...],
) -> None:
    for record in records:
        if _identity_now(directory_fd, record.name) != record.snapshot.identity:
            raise RawProbeAuthorityError(f"authority file changed after validation: {record.name}")


def validate_complete_raw_probe_authority(
    reader: PinnedRawProbeStoreReader,
    *,
    expected: ExpectedRawProbeAuthority,
) -> RawProbeAuthoritySnapshot:
    """Validate and snapshot the exact completed development-only raw store."""

    if type(reader) is not PinnedRawProbeStoreReader:
        raise RawProbeAuthorityError("complete authority requires a pinned raw-store reader")
    require_expected_raw_probe_authority(expected)
    reader.recheck()
    _root_shape(reader)
    manifest_value, manifest_record = _parse_snapshot(
        reader.root_fd, "manifest.json", RawProbeStoreManifest, "root"
    )
    if manifest_value != expected.manifest:
        raise RawProbeAuthorityError("raw-store manifest differs from frozen authority")

    key_names = _namespace_names(reader.keys_fd, expected.key_filenames, KEYS_DIR)
    expected_keys = {key.key_id: key for key in expected.keys}
    indices: list[RawProbeTaskKeyIndex] = []
    key_records: list[AuthorityFileRecord] = []
    for name in key_names:
        parsed, record = _parse_snapshot(reader.keys_fd, name, RawProbeTaskKeyIndex, "keys")
        assert isinstance(parsed, RawProbeTaskKeyIndex)
        if name != f"{parsed.key_id}.json":
            raise RawProbeAuthorityError("key index filename differs from key identity")
        if expected_keys.get(parsed.key_id) != parsed.key:
            raise RawProbeAuthorityError("key index differs from exact task/seed authority")
        indices.append(parsed)
        key_records.append(record)
    if len({index.artifact_id for index in indices}) != 240:
        raise RawProbeAuthorityError("key indexes do not bind 240 unique artifacts")
    index_by_key = {index.key_id: index for index in indices}
    index_by_artifact = {index.artifact_id: index for index in indices}

    artifact_names = tuple(sorted(f"{index.artifact_id}.json" for index in indices))
    _namespace_names(reader.artifacts_fd, artifact_names, ARTIFACTS_DIR)
    artifact_records: list[AuthorityFileRecord] = []
    for name in artifact_names:
        parsed, record = _parse_snapshot(
            reader.artifacts_fd, name, PersistedRawProbeArtifact, "artifacts"
        )
        assert isinstance(parsed, PersistedRawProbeArtifact)
        if name != f"{parsed.manifest.artifact_id}.json":
            raise RawProbeAuthorityError("artifact filename differs from manifest identity")
        index = index_by_artifact.get(parsed.manifest.artifact_id)
        if index is None or index.key != parsed.key or index.key_id != parsed.key.key_id:
            raise RawProbeAuthorityError("artifact does not resolve to its exact key index")
        artifact_records.append(record)

    fold_names = _namespace_names(
        reader.training_folds_fd,
        expected.training_fold_filenames,
        TRAINING_FOLDS_DIR,
    )
    fold_records: list[AuthorityFileRecord] = []
    for name in fold_names:
        parsed, record = _parse_snapshot(
            reader.training_folds_fd, name, TrainingFoldManifest, "training-folds"
        )
        assert isinstance(parsed, TrainingFoldManifest)
        if name != f"{parsed.fold_id}.r{parsed.replicate}.json":
            raise RawProbeAuthorityError("training-fold filename differs from identity")
        expected_refs = tuple(
            _reference(index_by_key[key.key_id])
            for key in expected.keys
            if key.replicate == parsed.replicate and key.family_id != parsed.heldout_family
        )
        expected_fold = TrainingFoldManifest(
            fold_id=parsed.heldout_family,
            heldout_family=parsed.heldout_family,
            replicate=parsed.replicate,
            task_references=expected_refs,
        )
        if parsed != expected_fold:
            raise RawProbeAuthorityError(
                "training-fold references differ from exact LOFO authority"
            )
        fold_records.append(record)

    binding_names = _namespace_names(
        reader.heldout_bindings_fd,
        expected.heldout_binding_filenames,
        HELDOUT_BINDINGS_DIR,
    )
    binding_records: list[AuthorityFileRecord] = []
    for name in binding_names:
        parsed, record = _parse_snapshot(
            reader.heldout_bindings_fd, name, HeldoutProbeBinding, "heldout-bindings"
        )
        assert isinstance(parsed, HeldoutProbeBinding)
        expected_name = (
            f"{parsed.family_id}.r{parsed.replicate}.task-{parsed.task_reference.task_index}.json"
        )
        if name != expected_name:
            raise RawProbeAuthorityError("heldout-binding filename differs from identity")
        expected_key = expected_keys.get(parsed.task_reference.key_id)
        index = index_by_key.get(parsed.task_reference.key_id)
        if (
            expected_key != parsed.task_reference.key
            or index is None
            or _reference(index) != parsed.task_reference
        ):
            raise RawProbeAuthorityError("heldout binding differs from exact task authority")
        binding_records.append(record)

    manifest_records = (manifest_record,)
    artifact_record_tuple = tuple(artifact_records)
    key_record_tuple = tuple(key_records)
    fold_record_tuple = tuple(fold_records)
    binding_record_tuple = tuple(binding_records)
    _root_shape(reader)
    _namespace_names(reader.keys_fd, expected.key_filenames, KEYS_DIR)
    _namespace_names(reader.artifacts_fd, artifact_names, ARTIFACTS_DIR)
    _namespace_names(reader.training_folds_fd, expected.training_fold_filenames, TRAINING_FOLDS_DIR)
    _namespace_names(
        reader.heldout_bindings_fd, expected.heldout_binding_filenames, HELDOUT_BINDINGS_DIR
    )
    _recheck_files(reader.root_fd, manifest_records)
    _recheck_files(reader.artifacts_fd, artifact_record_tuple)
    _recheck_files(reader.keys_fd, key_record_tuple)
    _recheck_files(reader.training_folds_fd, fold_record_tuple)
    _recheck_files(reader.heldout_bindings_fd, binding_record_tuple)
    reader.recheck()
    content_identity = {
        "manifest": (manifest_record.name, manifest_record.snapshot.sha256),
        "artifacts": tuple(
            (record.name, record.snapshot.sha256) for record in artifact_record_tuple
        ),
        "keys": tuple((record.name, record.snapshot.sha256) for record in key_record_tuple),
        "training_folds": tuple(
            (record.name, record.snapshot.sha256) for record in fold_record_tuple
        ),
        "heldout_bindings": tuple(
            (record.name, record.snapshot.sha256) for record in binding_record_tuple
        ),
    }
    return RawProbeAuthoritySnapshot(
        _token=_SNAPSHOT_TOKEN,
        manifest=expected.manifest,
        evidence_lock_file_sha256=expected.evidence_lock_file_sha256,
        authority_content_sha256=_digest(content_identity),
        directory_identities=reader.identities,
        manifest_file=manifest_record,
        artifact_files=artifact_record_tuple,
        key_files=key_record_tuple,
        training_fold_files=fold_record_tuple,
        heldout_binding_files=binding_record_tuple,
        key_ids=tuple(sorted(index_by_key)),
        artifact_ids=tuple(sorted(index_by_artifact)),
    )


__all__ = [
    "AUTHORITY_SNAPSHOT_VERSION",
    "AuthorityFileRecord",
    "ExpectedDevelopmentTask",
    "ExpectedRawProbeAuthority",
    "FROZEN_DEVELOPMENT_PROTOCOL_SHA256",
    "FROZEN_DEVELOPMENT_TASKS_SHA256",
    "FROZEN_LOCAL_AFFORDANCE_PROTOCOL_SHA256",
    "FROZEN_PHASE3_EVIDENCE_LOCK_FILE_SHA256",
    "FROZEN_PHASE3_EVIDENCE_LOCK_SHA256",
    "FROZEN_PROBE_POLICY_SHA256",
    "PERSISTED_ARTIFACT_VERSION",
    "PersistedRawProbeArtifact",
    "RawProbeAuthorityError",
    "RawProbeAuthoritySnapshot",
    "build_expected_raw_probe_authority",
    "require_expected_raw_probe_authority",
    "require_raw_probe_authority_snapshot",
    "validate_complete_raw_probe_authority",
]
