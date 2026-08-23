"""Development-only authority for the completed Phase 3 model store.

This boundary is intentionally a read-only, descriptor-pinned gate.  It validates
the complete 480-owner preparation publication and emits a small opaque authority
which can be consumed by a later execution boundary.  No outcome, evaluator,
oracle, search, or final-family module is imported here.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from levelup.experiments.milestone6_phase2 import _training_probe_seed
from levelup.experiments.milestone6_phase2_screening import (
    LEARNED_BASES,
    _authority_snapshot,
    screening_child_configs,
)
from levelup.experiments.milestone6_phase2_screening_preparation import (
    _common_key_inputs,
    _learned_condition_ids,
    _representative_units,
)
from levelup.experiments.milestone6_phase3_model_artifacts import (
    ARTIFACTS_DIR,
    COSTS_DIR,
    KEYS_DIR,
    PREPARATION_PROVENANCE_NAME,
    STAGING_DIR,
    Phase3ModelArtifactCost,
    Phase3ModelArtifactIndex,
    Phase3PreparationProvenance,
    _digest,
    _fd_json,
    load_phase3_model_bundle_from_at,
    open_phase3_model_artifact_reader_at,
)
from levelup.experiments.milestone6_phase3_model_preparation import (
    EXPECTED_EVIDENCE,
    EXPECTED_MODELS,
    EXPECTED_VIEWS,
    PROGRESS_NAME,
    Phase3ModelPreparationProgress,
)
from levelup.experiments.milestone6_phase3_plan import (
    FAMILIES,
    REPLICATES,
    TRAINING_TUPLE_IDS,
    Phase3Plan,
    validate_phase3_plan_lock_bytes,
)
from levelup.experiments.milestone6_phase3_protocol import NEW_CONDITIONS, ROOT
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import PhaseAccounting, TrainingPreparationAccounting
from levelup.experiments.runner.storage import plan_expected_units
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataEvidenceCostRecord,
    TrainingDataEvidenceKey,
    TrainingDataEvidenceManifest,
)

SCHEMA_VERSION = "milestone6.phase3.model-artifact-authority.v1"
ANCHOR_SCHEMA_VERSION = "milestone6.phase3.anchor.v1"
EVIDENCE_SCHEMA_VERSION = "milestone6.phase3.evidence-lock.v1"
HEX64 = frozenset("0123456789abcdef")
ANCHOR_TOP_LEVEL_KEYS = frozenset(
    {
        "aggregates",
        "anchor_manifest_sha256",
        "counts",
        "final_family_access",
        "final_results",
        "lineage",
        "model_owners",
        "new_execution",
        "schema_version",
        "scope",
        "t_alias",
        "unit_results",
    }
)
ANCHOR_LINEAGE_KEYS = frozenset(
    {
        "development_protocol_sha256",
        "development_tasks_sha256",
        "phase2_candidates_sha256",
        "phase2_provenance_sha256",
        "phase2_readiness_manifest_bytes_sha256",
        "phase2_readiness_manifest_sha256",
        "phase2_result_namespace_snapshot_sha256",
        "phase2_selection_analysis_sha256",
        "phase2_selection_lock_sha256",
        "phase2_tree_sha256",
        "phase3_protocol_sha256",
        "selection_lock_schema_version",
    }
)
ANCHOR_T_ALIAS_KEYS = frozenset(
    {
        "analysis_only",
        "condition_id",
        "historical_condition_id",
        "new_model",
        "new_unit_results",
        "new_view",
        "source_base_condition_id",
    }
)
ANCHOR_MODEL_OWNER_KEYS = frozenset(
    {
        "artifact_id",
        "base_condition_id",
        "cost_id",
        "family_id",
        "forward_passes",
        "key_id",
        "model_manifest_sha256",
        "optimizer_steps",
        "replicate",
        "trainable_parameters",
        "training_tuple_id",
    }
)
ANCHOR_UNIT_RESULT_KEYS = frozenset(
    {
        "base_condition_id",
        "candidate_tuple_id",
        "condition_id",
        "family_id",
        "phase",
        "replicate",
        "result_bytes",
        "result_bytes_sha256",
        "result_id",
        "run_id",
        "task_id",
        "task_index",
        "unit_id",
    }
)
EVIDENCE_TOP_LEVEL_KEYS = frozenset(
    {
        "aggregates",
        "counts",
        "evidence_artifacts",
        "evidence_lock_sha256",
        "final_family_access",
        "final_results",
        "lineage",
        "outcomes_included",
        "payloads_included",
        "schema_version",
        "scope",
    }
)
EVIDENCE_LINEAGE_KEYS = frozenset(
    {
        "phase2_provenance_sha256",
        "phase2_readiness_manifest_bytes_sha256",
        "phase2_readiness_manifest_sha256",
        "phase2_result_namespace_snapshot_sha256",
        "phase2_tree_sha256",
        "phase3_anchor_file_sha256",
        "phase3_anchor_manifest_sha256",
        "phase3_plan_id",
        "phase3_plan_lock_file_sha256",
        "phase3_plan_lock_sha256",
        "phase3_protocol_sha256",
    }
)
EVIDENCE_ARTIFACT_KEYS = frozenset(
    {
        "canonical_manifest_bytes_sha256",
        "child_run_id",
        "evidence_cost",
        "evidence_cost_id",
        "evidence_id",
        "evidence_key",
        "evidence_key_id",
        "evidence_manifest",
        "evidence_manifest_key_id",
        "family_id",
        "fold_id",
        "ordered_training_task_ids",
        "payload_bytes",
        "payload_sha256",
        "phase3_view_ids",
        "replicate",
    }
)


class Phase3ModelAuthorityError(ValueError):
    """Raised when the prepared model store cannot be authorized."""


class Phase3ModelAuthorityCost(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    setup: PhaseAccounting
    training_probes: PhaseAccounting
    reference_replay: PhaseAccounting
    training: PhaseAccounting
    serialization: PhaseAccounting


class Phase3ModelAuthorityRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    owner_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    key_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cost_id: str = Field(pattern=r"^[0-9a-f]{64}$")


class Phase3ModelArtifactAuthority(BaseModel):
    """Opaque, deterministic authority for exactly one complete model store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_only: StrictBool = True
    final: StrictBool = False
    final_family_accessed: StrictBool = False
    execution_authorized: StrictBool = True
    artifact_store_id: str = Field(min_length=1)
    plan_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    anchor_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    anchor_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation_git_commit_sha: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    preparation_provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_git_commit_sha: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    progress_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    family_order: tuple[str, ...]
    condition_ids: tuple[str, ...]
    replicates: tuple[int, ...]
    training_tuple_ids: tuple[str, ...]
    owner_ids: tuple[str, ...]
    unit_owner_mapping_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_evidence_count: StrictInt
    expected_view_count: StrictInt
    expected_model_count: StrictInt
    allowed_cost_accounting: Phase3ModelAuthorityCost
    models: tuple[Phase3ModelAuthorityRow, ...]

    @property
    def expected_authority_sha256(self) -> str:
        return _digest(self.model_dump(mode="json", exclude={"authority_sha256"}))

    @model_validator(mode="after")
    def authority_is_canonical(self) -> "Phase3ModelArtifactAuthority":
        if not self.development_only or self.final or self.final_family_accessed:
            raise ValueError("Phase 3 model authority is not development-only")
        if not self.execution_authorized:
            raise ValueError("Phase 3 model authority must authorize execution")
        if (
            self.artifact_store_id in {".", ".."}
            or "/" in self.artifact_store_id
            or "\\" in self.artifact_store_id
        ):
            raise ValueError("artifact store identity must be a basename")
        if self.family_order != FAMILIES or self.condition_ids != NEW_CONDITIONS:
            raise ValueError("Phase 3 authority universe drifted")
        if self.replicates != REPLICATES or self.training_tuple_ids != TRAINING_TUPLE_IDS:
            raise ValueError("Phase 3 authority training matrix drifted")
        if len(self.owner_ids) != EXPECTED_MODELS or len(set(self.owner_ids)) != EXPECTED_MODELS:
            raise ValueError("Phase 3 authority owner universe is incomplete")
        if self.owner_ids != tuple(sorted(self.owner_ids)):
            raise ValueError("Phase 3 authority owner universe is not canonical")
        row_owner_ids = tuple(row.owner_id for row in self.models)
        if set(row_owner_ids) != set(self.owner_ids) or len(set(row_owner_ids)) != EXPECTED_MODELS:
            raise ValueError("Phase 3 authority rows do not match owner universe")
        if (
            len({row.key_id for row in self.models}) != EXPECTED_MODELS
            or len({row.artifact_id for row in self.models}) != EXPECTED_MODELS
            or len({row.manifest_sha256 for row in self.models}) != EXPECTED_MODELS
            or len({row.cost_id for row in self.models}) != EXPECTED_MODELS
        ):
            raise ValueError("Phase 3 authority artifact identities are duplicated")
        if set(self.preparation_git_commit_sha) == {"0"} or set(self.generation_git_commit_sha) == {
            "0"
        }:
            raise ValueError("Phase 3 authority requires nonzero git provenance")
        if (self.expected_evidence_count, self.expected_view_count, self.expected_model_count) != (
            EXPECTED_EVIDENCE,
            EXPECTED_VIEWS,
            EXPECTED_MODELS,
        ):
            raise ValueError("Phase 3 authority counts are not exact")
        if len(self.models) != EXPECTED_MODELS or tuple(
            row.owner_id for row in self.models
        ) != tuple(sorted(row.owner_id for row in self.models)):
            raise ValueError("Phase 3 authority rows are not sorted or complete")
        if self.authority_sha256 != self.expected_authority_sha256:
            raise ValueError("Phase 3 model authority self-hash mismatch")
        zero = PhaseAccounting()
        allowed = self.allowed_cost_accounting
        if (
            allowed.setup != zero
            or allowed.training_probes != zero
            or allowed.reference_replay != zero
        ):
            raise ValueError("Phase 3 authority includes forbidden preparation costs")
        if allowed.serialization != PhaseAccounting(calls=EXPECTED_MODELS):
            raise ValueError("Phase 3 serialization accounting differs from the owner count")
        if (
            any(
                getattr(allowed.training, field) != 0
                for field in (
                    "calls",
                    "episodes",
                    "actions",
                    "environment_steps",
                    "resets",
                    "nodes_expanded",
                    "wall_seconds",
                )
            )
            or allowed.training.optimizer_steps <= 0
            or allowed.training.forward_passes <= 0
        ):
            raise ValueError("Phase 3 training accounting contains forbidden costs")
        return self


def canonical_phase3_model_authority_bytes(value: Phase3ModelArtifactAuthority) -> bytes:
    """Return canonical bytes, validating the self-hash first."""

    if not isinstance(value, Phase3ModelArtifactAuthority):
        raise Phase3ModelAuthorityError("authority is not typed")
    try:
        Phase3ModelArtifactAuthority.model_validate(value.model_dump(mode="json"))
    except (TypeError, ValueError) as exc:
        raise Phase3ModelAuthorityError("authority is not canonical") from exc
    return canonical_json_bytes(value.model_dump(mode="json"))


def load_phase3_model_artifact_authority_bytes(
    content: bytes,
) -> Phase3ModelArtifactAuthority:
    """Load only canonical, self-hashed authority bytes."""

    if not isinstance(content, bytes) or not content:
        raise Phase3ModelAuthorityError("authority bytes are missing")
    try:
        raw = json.loads(content)
        if not isinstance(raw, dict) or canonical_json_bytes(raw) != content:
            raise ValueError
        return Phase3ModelArtifactAuthority.model_validate(raw)
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase3ModelAuthorityError("authority bytes are not canonical") from exc


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_exact_keys(value: object, expected: frozenset[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise Phase3ModelAuthorityError(f"{label} schema differs")
    return value


def _canonical_phase2_evidence_keys(
    provenance_sha256: str,
) -> dict[tuple[str, int], TrainingDataEvidenceKey]:
    """Rebuild exact Phase 2 evidence keys without reopening stored evidence."""

    snapshot = _authority_snapshot()
    result: dict[tuple[str, int], TrainingDataEvidenceKey] = {}
    for config in screening_child_configs():
        expected = plan_expected_units(config)
        family = str(config.parameters["heldout_family_id"])
        for replicate in REPLICATES:
            probe_seeds = tuple(
                _training_probe_seed(task, replicate=replicate, protocol=snapshot.protocol)
                for task in config.split.development_tasks
            )
            candidates: list[TrainingDataEvidenceKey] = []
            for base in LEARNED_BASES:
                condition_ids = _learned_condition_ids(config, base)
                units = _representative_units(
                    config,
                    expected,
                    condition_ids=condition_ids,
                    replicate=replicate,
                )
                candidates.append(
                    TrainingDataEvidenceKey(
                        **_common_key_inputs(
                            config,
                            expected,
                            provenance_sha256=provenance_sha256,
                            replicate=replicate,
                            exposure_sha256=units[0].exposure_manifest_sha256,
                            data_order_seed=units[0].seeds.data_order_seed,
                            probe_seeds=probe_seeds,
                        )
                    )
                )
            if len(candidates) != len(LEARNED_BASES) or len(set(candidates)) != 1:
                raise Phase3ModelAuthorityError(
                    "canonical Phase 2 evidence identities differ across views"
                )
            result[(family, replicate)] = candidates[0]
    if set(result) != {(family, replicate) for family in FAMILIES for replicate in REPLICATES}:
        raise Phase3ModelAuthorityError("canonical Phase 2 evidence matrix differs")
    return result


def _validate_phase3_authority_source_shapes(
    anchor: dict[str, object], evidence: dict[str, object], *, plan: Phase3Plan
) -> None:
    """Reject undeclared fields before trusting any retained authority source."""

    _require_exact_keys(anchor, ANCHOR_TOP_LEVEL_KEYS, "Phase 3 anchor")
    anchor_lineage = _require_exact_keys(
        anchor.get("lineage"), ANCHOR_LINEAGE_KEYS, "Phase 3 anchor lineage"
    )
    _require_exact_keys(anchor.get("t_alias"), ANCHOR_T_ALIAS_KEYS, "Phase 3 anchor alias")
    anchor_owners = anchor.get("model_owners")
    anchor_units = anchor.get("unit_results")
    if not isinstance(anchor_owners, list) or not isinstance(anchor_units, list):
        raise Phase3ModelAuthorityError("Phase 3 anchor inventories are malformed")
    for row in anchor_owners:
        _require_exact_keys(row, ANCHOR_MODEL_OWNER_KEYS, "Phase 3 anchor model owner")
    for row in anchor_units:
        _require_exact_keys(row, ANCHOR_UNIT_RESULT_KEYS, "Phase 3 anchor unit result")

    _require_exact_keys(evidence, EVIDENCE_TOP_LEVEL_KEYS, "Phase 3 evidence")
    evidence_lineage = _require_exact_keys(
        evidence.get("lineage"), EVIDENCE_LINEAGE_KEYS, "Phase 3 evidence lineage"
    )
    shared_phase2_lineage = {
        key for key in set(anchor_lineage) & set(evidence_lineage) if key.startswith("phase2_")
    }
    if not shared_phase2_lineage or any(
        anchor_lineage[key] != evidence_lineage[key] for key in shared_phase2_lineage
    ):
        raise Phase3ModelAuthorityError("Phase 3 retained Phase 2 lineage differs")
    phase2_evidence_keys = _canonical_phase2_evidence_keys(
        str(evidence_lineage["phase2_provenance_sha256"])
    )
    evidence_rows = evidence.get("evidence_artifacts")
    if not isinstance(evidence_rows, list):
        raise Phase3ModelAuthorityError("Phase 3 evidence inventory is malformed")
    for row in evidence_rows:
        typed_row = _require_exact_keys(row, EVIDENCE_ARTIFACT_KEYS, "Phase 3 evidence artifact")
        try:
            key = TrainingDataEvidenceKey.model_validate(typed_row["evidence_key"])
            manifest = TrainingDataEvidenceManifest.model_validate(typed_row["evidence_manifest"])
            cost = TrainingDataEvidenceCostRecord.model_validate(typed_row["evidence_cost"])
        except (KeyError, TypeError, ValueError) as exc:
            raise Phase3ModelAuthorityError("Phase 3 evidence nested authority is invalid") from exc
        if key != phase2_evidence_keys.get((key.heldout_family_id, key.replicate)):
            raise Phase3ModelAuthorityError("Phase 3 evidence acquisition authority differs")
        if (
            manifest.key != key
            or cost.key != key
            or typed_row["evidence_key_id"] != key.key_id
            or typed_row["evidence_manifest_key_id"] != manifest.evidence_key_id
            or typed_row["evidence_id"] != manifest.evidence_id
            or typed_row["evidence_cost_id"] != cost.cost_id
            or cost.artifact_id != manifest.evidence_id
        ):
            raise Phase3ModelAuthorityError("Phase 3 evidence nested lineage differs")
        matching_views = [
            view
            for view in plan.views
            if view.heldout_family == key.heldout_family_id and view.replicate == key.replicate
        ]
        expected_view_ids = [view.view_id for view in matching_views]
        expected_heldout_task_ids = tuple(
            dict.fromkeys(
                item.unit.key.task_id
                for item in plan.units
                if item.heldout_family == key.heldout_family_id
                and item.unit.key.replicate == key.replicate
            )
        )
        plan_authority = dict(plan.authority_hashes)
        ordered_training = typed_row["ordered_training_task_ids"]
        if (
            len(matching_views) != len(TRAINING_TUPLE_IDS)
            or any(
                view.training_task_ids != key.ordered_training_task_ids
                or view.fold_id != key.fold_id
                or view.data_order_seed != key.data_order_seed
                for view in matching_views
            )
            or len(expected_heldout_task_ids) != 8
            or key.ordered_heldout_task_ids != expected_heldout_task_ids
            or key.protocol_sha256 != plan_authority["development_protocol_sha256"]
            or key.task_manifest_sha256 != plan_authority["development_tasks_sha256"]
            or key.screening_candidates_sha256 != plan_authority["phase2_candidates_sha256"]
            or key.provenance_sha256 != evidence_lineage["phase2_provenance_sha256"]
            or typed_row["payload_sha256"] != manifest.payload_sha256
            or typed_row["payload_bytes"] != manifest.payload_bytes
            or typed_row["family_id"] != key.heldout_family_id
            or typed_row["fold_id"] != key.fold_id
            or isinstance(typed_row["replicate"], bool)
            or typed_row["replicate"] != key.replicate
            or not isinstance(ordered_training, list)
            or tuple(ordered_training) != key.ordered_training_task_ids
            or typed_row["canonical_manifest_bytes_sha256"]
            != _sha_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))
            or typed_row["phase3_view_ids"] != expected_view_ids
        ):
            raise Phase3ModelAuthorityError("Phase 3 evidence row aliases differ")


def _sum_phase(left: PhaseAccounting, right: PhaseAccounting) -> PhaseAccounting:
    values = {
        field: getattr(left, field) + getattr(right, field)
        for field in PhaseAccounting.model_fields
    }
    return PhaseAccounting(**values)


def _sum_cost(
    total: Phase3ModelAuthorityCost, cost: TrainingPreparationAccounting
) -> Phase3ModelAuthorityCost:
    return Phase3ModelAuthorityCost(
        **{
            field: _sum_phase(getattr(total, field), getattr(cost, field))
            for field in Phase3ModelAuthorityCost.model_fields
        }
    )


def _git(repository: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ("git", *args), cwd=repository, check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase3ModelAuthorityError("authority repository git state is unavailable") from exc


def _repo_state(repository: Path) -> tuple[str, bool]:
    commit = _git(repository, "rev-parse", "HEAD").decode().strip()
    status = _git(repository, "status", "--porcelain=v1", "-z")
    if not commit or any(char not in "0123456789abcdef" for char in commit):
        raise Phase3ModelAuthorityError("authority repository commit is malformed")
    return commit, bool(status)


def _safe_authority_repository(repository: str | Path) -> Path:
    lexical = Path(repository).absolute()
    for candidate in (lexical, *lexical.parents):
        if os.path.lexists(candidate) and candidate.is_symlink():
            raise Phase3ModelAuthorityError("authority repository or ancestor is a symlink")
    try:
        resolved = lexical.resolve(strict=True)
        root = ROOT.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Phase3ModelAuthorityError("authority repository cannot be resolved") from exc
    if resolved != root or not resolved.is_dir():
        raise Phase3ModelAuthorityError("authority repository must equal the current checkout")
    try:
        fd = secure_fs.open_directory_chain(resolved)
        try:
            identity = secure_fs.directory_identity(fd)
        finally:
            os.close(fd)
        root_fd = secure_fs.open_directory_chain(root)
        try:
            if identity != secure_fs.directory_identity(root_fd):
                raise Phase3ModelAuthorityError("authority repository identity drifted")
        finally:
            os.close(root_fd)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, Phase3ModelAuthorityError):
            raise
        raise Phase3ModelAuthorityError("authority repository cannot be pinned") from exc
    return resolved


def _repository_identity(repository: Path) -> tuple[int, int]:
    try:
        fd = secure_fs.open_directory_chain(repository)
        try:
            return secure_fs.directory_identity(fd)
        finally:
            os.close(fd)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Phase3ModelAuthorityError("authority repository identity cannot be read") from exc


@contextmanager
def _open_existing_output(
    output_root: str | Path,
) -> Iterator[tuple[int, tuple[tuple[int, int], ...]]]:
    lexical = Path(output_root).absolute()
    for candidate in (lexical, *lexical.parents):
        if os.path.lexists(candidate) and candidate.is_symlink():
            raise Phase3ModelAuthorityError("model output root or ancestor is a symlink")
    try:
        with ExitStack() as stack:
            root_fd = secure_fs.open_directory_chain(lexical)
            stack.callback(os.close, root_fd)
            children = []
            for name in (KEYS_DIR, COSTS_DIR, ARTIFACTS_DIR, STAGING_DIR):
                child_fd = secure_fs.open_child_directory(root_fd, name)
                stack.callback(os.close, child_fd)
                children.append(child_fd)
            identities = (secure_fs.directory_identity(root_fd),) + tuple(
                secure_fs.directory_identity(fd) for fd in children
            )
            yield root_fd, identities
    except Phase3ModelAuthorityError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Phase3ModelAuthorityError("model output namespaces cannot be pinned") from exc


def _recheck_output(output_root: Path, identities: tuple[tuple[int, int], ...]) -> None:
    try:
        fd = secure_fs.open_directory_chain(output_root)
        try:
            observed = [secure_fs.directory_identity(fd)]
            for name in (KEYS_DIR, COSTS_DIR, ARTIFACTS_DIR, STAGING_DIR):
                child = secure_fs.open_child_directory(fd, name)
                try:
                    observed.append(secure_fs.directory_identity(child))
                finally:
                    os.close(child)
        finally:
            os.close(fd)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Phase3ModelAuthorityError("model output root or namespace was replaced") from exc
    if tuple(observed) != identities:
        raise Phase3ModelAuthorityError("model output root or namespace was replaced")


def _root_entries(fd: int) -> dict[str, tuple[bool, bool, bool]]:
    try:
        with os.scandir(fd) as iterator:
            return {
                entry.name: (
                    entry.is_symlink(),
                    entry.is_file(follow_symlinks=False),
                    entry.is_dir(follow_symlinks=False),
                )
                for entry in iterator
            }
    except OSError as exc:
        raise Phase3ModelAuthorityError("model output inventory is unreadable") from exc


def _validate_empty_staging(fd: int) -> None:
    entries = _root_entries(fd)
    if entries:
        raise Phase3ModelAuthorityError("model output staging namespace is not empty")


def build_phase3_model_artifact_authority(
    output_root: str | Path,
    *,
    authority_repository: str | Path,
) -> Phase3ModelArtifactAuthority:
    """Validate and build the deterministic authority for a complete model store."""

    repository = _safe_authority_repository(authority_repository)
    repository_identity = _repository_identity(repository)
    generation_commit, dirty = _repo_state(repository)
    if dirty:
        raise Phase3ModelAuthorityError("authority repository must be clean")
    output_path = Path(output_root).absolute()
    store_id = output_path.name
    if not store_id or store_id in {".", ".."} or "/" in store_id or "\\" in store_id:
        raise Phase3ModelAuthorityError("model store identity is not a basename")
    # Read every committed authority source through one pinned repository
    # descriptor; path-based reloads would permit same-byte substitution races.
    try:
        authority_root_fd = secure_fs.open_directory_chain(repository)
        authority_cfg_fd = secure_fs.open_child_chain(authority_root_fd, "configs", "milestone6")
        try:
            plan_bytes = secure_fs.read_bytes_at(authority_cfg_fd, "phase3_plan_lock.json")
            anchor_bytes = secure_fs.read_bytes_at(authority_cfg_fd, "phase3_anchor_manifest.json")
            evidence_bytes = secure_fs.read_bytes_at(authority_cfg_fd, "phase3_evidence_lock.json")
        finally:
            os.close(authority_cfg_fd)
            os.close(authority_root_fd)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Phase3ModelAuthorityError(
            "Phase 3 committed authority sources are unavailable"
        ) from exc
    try:
        plan = validate_phase3_plan_lock_bytes(plan_bytes)
    except (TypeError, ValueError) as exc:
        raise Phase3ModelAuthorityError("Phase 3 plan authority is invalid") from exc
    if plan.final_family_access or len(plan.model_owners) != EXPECTED_MODELS:
        raise Phase3ModelAuthorityError("Phase 3 plan is not the complete development plan")
    expected_owners = tuple(owner.owner_id for owner in plan.model_owners)
    expected_set = set(expected_owners)
    try:
        anchor = json.loads(anchor_bytes)
        evidence = json.loads(evidence_bytes)
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase3ModelAuthorityError("Phase 3 anchor/evidence bytes are invalid") from exc
    if not isinstance(anchor, dict) or not isinstance(evidence, dict):
        raise Phase3ModelAuthorityError("Phase 3 anchor/evidence must be objects")
    _validate_phase3_authority_source_shapes(anchor, evidence, plan=plan)
    anchor_sha = anchor.get("anchor_manifest_sha256")
    evidence_sha = evidence.get("evidence_lock_sha256")
    if not isinstance(anchor_sha, str) or not isinstance(evidence_sha, str):
        raise Phase3ModelAuthorityError("Phase 3 anchor/evidence self-hashes are missing")
    if anchor.get("anchor_manifest_sha256") != _digest(
        {k: v for k, v in anchor.items() if k != "anchor_manifest_sha256"}
    ):
        raise Phase3ModelAuthorityError("Phase 3 anchor self-hash is invalid")
    if evidence.get("evidence_lock_sha256") != _digest(
        {k: v for k, v in evidence.items() if k != "evidence_lock_sha256"}
    ):
        raise Phase3ModelAuthorityError("Phase 3 evidence self-hash is invalid")
    if (
        canonical_json_bytes(anchor) != anchor_bytes
        or canonical_json_bytes(evidence) != evidence_bytes
    ):
        raise Phase3ModelAuthorityError("Phase 3 anchor/evidence bytes are not canonical")
    anchor_counts = {
        "families": len(FAMILIES),
        "anchor_base_conditions": 2,
        "model_owners": 240,
        "unit_results": 5760,
    }
    if (
        anchor.get("schema_version") != ANCHOR_SCHEMA_VERSION
        or anchor.get("scope") != "known-development-only"
        or anchor.get("final_family_access") is not False
        or anchor.get("new_execution") is not False
        or anchor.get("aggregates") != []
        or anchor.get("final_results") != []
        or anchor.get("counts") != anchor_counts
    ):
        raise Phase3ModelAuthorityError("Phase 3 anchor scope or coverage is invalid")
    anchor_lineage = anchor.get("lineage")
    plan_authority = dict(plan.authority_hashes)
    if not isinstance(anchor_lineage, dict) or any(
        anchor_lineage.get(anchor_key) != plan_authority.get(plan_key)
        for anchor_key, plan_key in (
            ("phase3_protocol_sha256", "protocol_sha256"),
            ("development_protocol_sha256", "development_protocol_sha256"),
            ("development_tasks_sha256", "development_tasks_sha256"),
            ("phase2_candidates_sha256", "phase2_candidates_sha256"),
            ("phase2_selection_lock_sha256", "phase2_selection_lock_sha256"),
        )
    ):
        raise Phase3ModelAuthorityError("Phase 3 anchor lineage differs from the plan")
    t_alias = anchor.get("t_alias")
    if (
        not isinstance(t_alias, dict)
        or t_alias.get("analysis_only") is not True
        or t_alias.get("new_view") is not False
        or t_alias.get("new_model") is not False
        or t_alias.get("new_unit_results") is not False
    ):
        raise Phase3ModelAuthorityError("Phase 3 anchor alias is not analysis-only")
    evidence_counts = {
        "families": len(FAMILIES),
        "replicates": len(REPLICATES),
        "evidence_artifacts": EXPECTED_EVIDENCE,
    }
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("scope") != "known-development-only"
        or evidence.get("final_family_access") is not False
        or evidence.get("payloads_included") is not False
        or evidence.get("outcomes_included") is not False
        or evidence.get("aggregates") != []
        or evidence.get("final_results") != []
        or evidence.get("counts") != evidence_counts
    ):
        raise Phase3ModelAuthorityError("Phase 3 evidence scope or coverage is invalid")
    evidence_lineage = evidence.get("lineage")
    if not isinstance(evidence_lineage, dict) or any(
        evidence_lineage.get(key) != expected
        for key, expected in (
            ("phase3_protocol_sha256", plan.protocol_sha256),
            ("phase3_plan_id", plan.plan_id),
            ("phase3_plan_lock_file_sha256", _sha_bytes(plan_bytes)),
            ("phase3_anchor_manifest_sha256", anchor_sha),
            ("phase3_anchor_file_sha256", _sha_bytes(anchor_bytes)),
        )
    ):
        raise Phase3ModelAuthorityError("Phase 3 evidence lineage differs")
    evidence_rows = evidence.get("evidence_artifacts")
    if not isinstance(evidence_rows, list) or len(evidence_rows) != EXPECTED_EVIDENCE:
        raise Phase3ModelAuthorityError("Phase 3 evidence coverage is incomplete")
    evidence_by_pair: dict[tuple[str, int], dict[str, object]] = {}
    for row in evidence_rows:
        if not isinstance(row, dict):
            raise Phase3ModelAuthorityError("Phase 3 evidence row is malformed")
        pair = (row.get("family_id"), row.get("replicate"))
        if pair in evidence_by_pair or pair[0] not in FAMILIES or pair[1] not in REPLICATES:
            raise Phase3ModelAuthorityError("Phase 3 evidence coverage is duplicated or extra")
        evidence_by_pair[pair] = row
    if set(evidence_by_pair) != {
        (family, replicate) for family in FAMILIES for replicate in REPLICATES
    }:
        raise Phase3ModelAuthorityError("Phase 3 evidence coverage differs from the frozen matrix")

    with _open_existing_output(output_path) as (root_fd, identities):
        entries = _root_entries(root_fd)
        expected_root = {
            KEYS_DIR,
            COSTS_DIR,
            ARTIFACTS_DIR,
            STAGING_DIR,
            PROGRESS_NAME,
            PREPARATION_PROVENANCE_NAME,
        }
        expected_types = {
            KEYS_DIR: (False, False, True),
            COSTS_DIR: (False, False, True),
            ARTIFACTS_DIR: (False, False, True),
            STAGING_DIR: (False, False, True),
            PROGRESS_NAME: (False, True, False),
            PREPARATION_PROVENANCE_NAME: (False, True, False),
        }
        if set(entries) != expected_root or entries != expected_types:
            raise Phase3ModelAuthorityError("model output root namespace differs")
        staging_fd = secure_fs.open_child_directory(root_fd, STAGING_DIR)
        try:
            _validate_empty_staging(staging_fd)
        finally:
            os.close(staging_fd)
        progress_bytes = secure_fs.read_bytes_at(root_fd, PROGRESS_NAME)
        provenance_bytes = secure_fs.read_bytes_at(root_fd, PREPARATION_PROVENANCE_NAME)
        try:
            progress_value = json.loads(progress_bytes)
            if canonical_json_bytes(progress_value) + b"\n" != progress_bytes:
                raise ValueError
            progress = Phase3ModelPreparationProgress.model_validate(progress_value)
            provenance_value = json.loads(provenance_bytes)
            if canonical_json_bytes(provenance_value) + b"\n" != provenance_bytes:
                raise ValueError
            provenance = Phase3PreparationProvenance.model_validate(provenance_value)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise Phase3ModelAuthorityError(
                "model preparation progress/provenance is invalid"
            ) from exc
        if (
            progress.status != "complete"
            or tuple(progress.expected_owner_ids) != expected_owners
            or tuple(progress.completed_owner_ids) != expected_owners
        ):
            raise Phase3ModelAuthorityError("model preparation progress is incomplete or reordered")
        if (
            progress.plan_id != plan.plan_id
            or progress.protocol_sha256 != plan.protocol_sha256
            or progress.anchor_manifest_sha256 != anchor_sha
            or progress.evidence_lock_sha256 != evidence_sha
        ):
            raise Phase3ModelAuthorityError("model preparation progress lineage differs")
        prep_commit = provenance.provenance.git_commit_sha
        prep_prov_sha = provenance.provenance_sha256
        if provenance.provenance.git_dirty:
            raise Phase3ModelAuthorityError("model preparation provenance is dirty")
        if (
            progress.preparation_git_commit_sha != prep_commit
            or progress.preparation_provenance_sha256 != prep_prov_sha
        ):
            raise Phase3ModelAuthorityError("model preparation provenance differs")
        rows: list[Phase3ModelAuthorityRow] = []
        total = Phase3ModelAuthorityCost(
            **{field: PhaseAccounting() for field in Phase3ModelAuthorityCost.model_fields}
        )
        with open_phase3_model_artifact_reader_at(root_fd) as reader:
            key_names = secure_fs.regular_entries_at(reader.keys_fd)
            cost_names = secure_fs.regular_entries_at(reader.costs_fd)
            artifact_inventory = _root_entries(reader.artifacts_fd)
            if any(value != (False, False, True) for value in artifact_inventory.values()):
                raise Phase3ModelAuthorityError("model artifact namespace contains unsafe entries")
            artifact_names = tuple(sorted(artifact_inventory))
            if (
                len(key_names) != EXPECTED_MODELS
                or len(cost_names) != EXPECTED_MODELS
                or len(artifact_names) != EXPECTED_MODELS
            ):
                raise Phase3ModelAuthorityError("model artifact namespace counts differ")
            by_owner: dict[str, tuple[Phase3ModelArtifactIndex, Phase3ModelArtifactCost, str]] = {}
            for name in key_names:
                try:
                    raw = _fd_json(reader.keys_fd, name)
                    index = Phase3ModelArtifactIndex.model_validate(raw)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise Phase3ModelAuthorityError("model key index is invalid") from exc
                if name != f"{index.key_id}.json":
                    raise Phase3ModelAuthorityError("model key filename differs")
                key = index.key
                if key.owner_id not in expected_set or key.owner_id in by_owner:
                    raise Phase3ModelAuthorityError("model owner coverage is duplicated or extra")
                expected_owner = next(
                    owner for owner in plan.model_owners if owner.owner_id == key.owner_id
                )
                if (
                    key.condition_id,
                    key.fold_id,
                    key.heldout_family,
                    key.replicate,
                    key.training_tuple_id,
                    key.view_id,
                ) != (
                    expected_owner.condition_id,
                    expected_owner.fold_id,
                    expected_owner.heldout_family,
                    expected_owner.replicate,
                    expected_owner.training_tuple_id,
                    expected_owner.view_id,
                ):
                    raise Phase3ModelAuthorityError("model key differs from frozen owner plan")
                if (
                    key.model_seed != expected_owner.model_seed
                    or key.optimizer.learning_rate != expected_owner.learning_rate
                    or key.report.optimizer_steps != expected_owner.training_epochs
                ):
                    raise Phase3ModelAuthorityError(
                        "model training identity differs from frozen owner plan"
                    )
                try:
                    cost = Phase3ModelArtifactCost.model_validate(
                        _fd_json(reader.costs_fd, f"{key.key_id}.json")
                    )
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise Phase3ModelAuthorityError("model cost is invalid") from exc
                if cost.key != key or cost.artifact_id != index.artifact_id:
                    raise Phase3ModelAuthorityError("model cost lineage differs")
                if (
                    key.plan_id != plan.plan_id
                    or key.protocol_sha256 != plan.protocol_sha256
                    or key.evidence_lock_sha256 != evidence_sha
                ):
                    raise Phase3ModelAuthorityError("model key authority lineage differs")
                evidence_row = evidence_by_pair[(key.heldout_family, key.replicate)]
                if (
                    key.evidence_payload_sha256 != evidence_row.get("payload_sha256")
                    or key.evidence_payload_bytes != evidence_row.get("payload_bytes")
                    or key.view_id not in tuple(evidence_row.get("phase3_view_ids", ()))
                ):
                    raise Phase3ModelAuthorityError("model key evidence lineage differs")
                if (
                    key.preparation_git_commit_sha != prep_commit
                    or key.preparation_provenance_sha256 != prep_prov_sha
                ):
                    raise Phase3ModelAuthorityError("model key preparation provenance differs")
                try:
                    _, _, manifest, _ = load_phase3_model_bundle_from_at(reader, key)
                except Exception as exc:
                    raise Phase3ModelAuthorityError(
                        "model bundle or tensor bytes are invalid"
                    ) from exc
                by_owner[key.owner_id] = (index, cost, _digest(manifest.model_dump(mode="json")))
                total = _sum_cost(total, cost.accounting)
            if (
                set(by_owner) != expected_set
                or set(cost_names) != {f"{value[1].key_id}.json" for value in by_owner.values()}
                or set(artifact_names) != {value[0].artifact_id for value in by_owner.values()}
            ):
                raise Phase3ModelAuthorityError(
                    "model namespaces do not exactly cover the owner universe"
                )
            for owner_id in expected_owners:
                index, cost, manifest_sha = by_owner[owner_id]
                rows.append(
                    Phase3ModelAuthorityRow(
                        owner_id=owner_id,
                        key_id=index.key_id,
                        artifact_id=index.artifact_id,
                        manifest_sha256=manifest_sha,
                        cost_id=cost.cost_id,
                    )
                )
        _recheck_output(output_path, identities)
    try:
        check_root_fd = secure_fs.open_directory_chain(repository)
        check_cfg_fd = secure_fs.open_child_chain(check_root_fd, "configs", "milestone6")
        try:
            if (
                secure_fs.read_bytes_at(check_cfg_fd, "phase3_plan_lock.json") != plan_bytes
                or secure_fs.read_bytes_at(check_cfg_fd, "phase3_anchor_manifest.json")
                != anchor_bytes
                or secure_fs.read_bytes_at(check_cfg_fd, "phase3_evidence_lock.json")
                != evidence_bytes
            ):
                raise Phase3ModelAuthorityError(
                    "Phase 3 committed authority sources changed during validation"
                )
        finally:
            os.close(check_cfg_fd)
            os.close(check_root_fd)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, Phase3ModelAuthorityError):
            raise
        raise Phase3ModelAuthorityError(
            "Phase 3 committed authority sources cannot be rechecked"
        ) from exc
    final_commit, final_dirty = _repo_state(repository)
    if (
        final_dirty
        or final_commit != generation_commit
        or _repository_identity(repository) != repository_identity
    ):
        raise Phase3ModelAuthorityError("authority repository changed during validation")
    body = {
        "schema_version": SCHEMA_VERSION,
        "development_only": True,
        "final": False,
        "final_family_accessed": False,
        "execution_authorized": True,
        "artifact_store_id": store_id,
        "plan_id": plan.plan_id,
        "protocol_sha256": plan.protocol_sha256,
        "plan_file_sha256": _sha_bytes(plan_bytes),
        "anchor_manifest_sha256": anchor_sha,
        "anchor_file_sha256": _sha_bytes(anchor_bytes),
        "evidence_lock_sha256": evidence_sha,
        "evidence_file_sha256": _sha_bytes(evidence_bytes),
        "preparation_git_commit_sha": prep_commit,
        "preparation_provenance_sha256": prep_prov_sha,
        "generation_git_commit_sha": generation_commit,
        "progress_sha256": _sha_bytes(progress_bytes),
        "provenance_file_sha256": _sha_bytes(provenance_bytes),
        "family_order": FAMILIES,
        "condition_ids": NEW_CONDITIONS,
        "replicates": REPLICATES,
        "training_tuple_ids": TRAINING_TUPLE_IDS,
        "expected_evidence_count": EXPECTED_EVIDENCE,
        "expected_view_count": EXPECTED_VIEWS,
        "expected_model_count": EXPECTED_MODELS,
        "allowed_cost_accounting": total.model_dump(mode="json"),
        "owner_ids": tuple(sorted(expected_owners)),
        "unit_owner_mapping_sha256": _digest(
            [(item.unit.unit_id, item.model_owner_id) for item in plan.units]
        ),
        "models": tuple(
            row.model_dump(mode="json") for row in sorted(rows, key=lambda row: row.owner_id)
        ),
    }
    body["authority_sha256"] = _digest(body)
    return Phase3ModelArtifactAuthority.model_validate(body)


def write_phase3_model_authority(path: str | Path, authority: Phase3ModelArtifactAuthority) -> None:
    """Exclusively publish canonical authority bytes after validation."""

    payload = canonical_phase3_model_authority_bytes(authority)
    target = Path(path).absolute()
    if target.exists() or target.is_symlink():
        raise Phase3ModelAuthorityError("authority output already exists")
    parent = target.parent
    fd: int | None = None
    parent_fd: int | None = None
    created = False
    try:
        parent_fd = secure_fs.open_directory_chain(parent)
        fd = os.open(
            target.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        created = True
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(fd)
        os.fsync(parent_fd)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if created and parent_fd is not None:
            try:
                os.unlink(target.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise Phase3ModelAuthorityError("cannot exclusively publish model authority") from exc
    finally:
        if fd is not None:
            os.close(fd)
        if parent_fd is not None:
            os.close(parent_fd)


validate_phase3_model_artifact_authority = Phase3ModelArtifactAuthority.model_validate
