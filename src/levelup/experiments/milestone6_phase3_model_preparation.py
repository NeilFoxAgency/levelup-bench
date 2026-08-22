"""Resumable, preparation-only Phase 3 model batch construction.

The batch driver is intentionally a thin orchestration boundary.  It accepts only
the opaque authorities produced by the Phase 3 plan/evidence gates, reads evidence
through a descriptor-pinned training-data reader, and delegates representation and
training semantics to :mod:`milestone6_phase3_models`.  It has no execution path:
there is no search, replay, evaluator, oracle, environment, result, or activation
operation in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from levelup.experiments.milestone6_phase2_screening_runtime import (
    ScreeningRuntime,
    recheck_screening_runtime_readonly,
)
from levelup.experiments.milestone6_phase3_anchor import (
    Phase3AnchorManifest,
    load_committed_phase3_anchor_manifest_bytes,
    require_phase3_anchor_manifest,
)
from levelup.experiments.milestone6_phase3_evidence import (
    Phase3EvidenceLock,
    load_committed_phase3_evidence_lock_bytes,
    require_phase3_evidence_lock,
)
from levelup.experiments.milestone6_phase3_model_artifacts import (
    KEYS_DIR,
    PREPARATION_PROVENANCE_NAME,
    Phase3ModelArtifactCost,
    Phase3ModelArtifactIndex,
    Phase3ModelArtifactKey,
    Phase3PreparationProvenance,
    PinnedPhase3ModelOutput,
    _digest,
    _exclusive_claim_at,
    _fd_json,
    _fd_model,
    _load_manifest_at,
    load_phase3_model_bundle_from_at,
    open_phase3_model_artifact_reader_at,
    open_phase3_model_output,
    write_phase3_model_artifact,
)
from levelup.experiments.milestone6_phase3_models import (
    HISTORY_PARAMETERS,
    S_CONDITION,
    S_PARAMETERS,
    Phase3ModelPreparation,
    Phase3ViewPreparation,
    _model_identity_sha256,
    prepare_phase3_model,
    prepare_phase3_view,
)
from levelup.experiments.milestone6_phase3_plan import (
    ValidatedPhase3Plan,
    load_committed_phase3_plan_lock_bytes,
    validate_phase3_plan_lock_bytes,
)
from levelup.experiments.milestone6_phase3_protocol import FAMILIES
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.provenance import capture_system_provenance
from levelup.experiments.runner.records import (
    PhaseAccounting,
    SystemProvenance,
    TrainingPreparationAccounting,
)
from levelup.experiments.runner.storage import ArtifactValidationError, provenance_identity_sha256
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataEvidenceKey,
    TrainingDataEvidenceManifest,
    load_training_data_evidence_payload_bundle_from_at,
    open_training_data_reader,
)

PROGRESS_NAME = "phase3-model-preparation-progress.json"
HEX64 = r"^[0-9a-f]{64}$"
EXPECTED_EVIDENCE = 30
EXPECTED_VIEWS = 120
EXPECTED_MODELS = 480
_SCHEMA = "milestone6.phase3.model-preparation-progress.v1"


class Phase3ModelPreparationError(ArtifactValidationError):
    """Raised when a preparation authority, artifact, or progress state drifts."""


class Phase3ModelPreparationProgress(BaseModel):
    """Durable progress record; ``complete`` is possible only for all 480 owners."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[_SCHEMA] = _SCHEMA
    plan_id: str = Field(pattern=HEX64)
    protocol_sha256: str = Field(pattern=HEX64)
    anchor_manifest_sha256: str = Field(pattern=HEX64)
    evidence_lock_sha256: str = Field(pattern=HEX64)
    expected_owner_ids: tuple[str, ...]
    completed_owner_ids: tuple[str, ...] = ()
    status: Literal["running", "failed", "complete"] = "running"
    evidence_count: int = 0
    view_count: int = 0
    model_count: int = 0
    error: str | None = None
    preparation_git_commit_sha: str = Field(
        pattern=r"^[0-9a-f]{40,64}$"
    )
    preparation_provenance_sha256: str = Field(pattern=HEX64)

    @model_validator(mode="after")
    def exact_progress(self) -> "Phase3ModelPreparationProgress":
        if set(self.preparation_git_commit_sha) == {"0"}:
            raise ValueError("preparation commit provenance is required")
        if set(self.preparation_provenance_sha256) == {"0"}:
            raise ValueError("preparation provenance identity is required")
        if len(self.expected_owner_ids) != EXPECTED_MODELS:
            raise ValueError("Phase 3 progress requires the exact 480-owner universe")
        if len(set(self.expected_owner_ids)) != EXPECTED_MODELS:
            raise ValueError("Phase 3 progress owner identities are duplicated")
        if any(owner not in set(self.expected_owner_ids) for owner in self.completed_owner_ids):
            raise ValueError("Phase 3 progress contains an unexpected completed owner")
        if len(set(self.completed_owner_ids)) != len(self.completed_owner_ids):
            raise ValueError("Phase 3 progress completed owners are duplicated")
        if self.model_count != len(self.completed_owner_ids):
            raise ValueError("Phase 3 progress model count differs from completed owners")
        if self.status == "failed" and not self.error:
            raise ValueError("failed Phase 3 progress requires an error")
        if self.status != "failed" and self.error is not None:
            raise ValueError("non-failed Phase 3 progress cannot carry an error")
        if self.status == "complete" and len(self.completed_owner_ids) != EXPECTED_MODELS:
            raise ValueError("Phase 3 progress cannot be complete before all owners finish")
        if self.status == "complete" and (self.evidence_count, self.view_count, self.model_count) != (
            EXPECTED_EVIDENCE,
            EXPECTED_VIEWS,
            EXPECTED_MODELS,
        ):
            raise ValueError("Phase 3 progress completion counts are incomplete")
        return self


class Phase3ModelPreparationResult(BaseModel):
    """Small deterministic return value; model tensors remain in artifact storage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(pattern=HEX64)
    evidence_count: int = Field(ge=0)
    view_count: int = Field(ge=0)
    model_count: int = Field(ge=0)
    completed_owner_ids: tuple[str, ...]
    complete: bool
    progress_path: str


def _atomic_progress_at(
    output: PinnedPhase3ModelOutput,
    progress: Phase3ModelPreparationProgress,
) -> None:
    """Atomically replace progress relative to the held output fd."""
    payload = canonical_json_bytes(progress.model_dump(mode="json")) + b"\n"
    temporary = f".{PROGRESS_NAME}.{os.getpid()}.{id(progress)}.tmp"
    fd: int | None = None
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=output.staging_fd,
        )
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        if fd is not None:
            os.close(fd)
    try:
        output.recheck()
        try:
            existing = os.stat(PROGRESS_NAME, dir_fd=output.root_fd, follow_symlinks=False)
            if not stat.S_ISREG(existing.st_mode):
                raise Phase3ModelPreparationError("refusing non-regular progress entry")
        except FileNotFoundError:
            pass
        os.replace(
            temporary,
            PROGRESS_NAME,
            src_dir_fd=output.staging_fd,
            dst_dir_fd=output.root_fd,
        )
        os.fsync(output.root_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=output.staging_fd)
        except FileNotFoundError:
            pass
        os.fsync(output.staging_fd)


def _read_progress_at(output: PinnedPhase3ModelOutput) -> Phase3ModelPreparationProgress | None:
    try:
        content = secure_fs.read_bytes_at(output.root_fd, PROGRESS_NAME)
    except secure_fs.SecureFilesystemError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            return None
        # secure_fs wraps ENOENT; distinguish via a direct no-follow stat.
        try:
            os.stat(PROGRESS_NAME, dir_fd=output.root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        raise Phase3ModelPreparationError("Phase 3 progress cannot be read") from exc
    try:
        value = json.loads(content)
        if canonical_json_bytes(value) + b"\n" != content:
            raise Phase3ModelPreparationError("Phase 3 progress bytes are not canonical")
        return Phase3ModelPreparationProgress.model_validate(value)
    except Phase3ModelPreparationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Phase3ModelPreparationError("Phase 3 progress record is invalid") from exc


def _validate_progress_preparation_provenance(
    progress: Phase3ModelPreparationProgress,
    *,
    git_commit_sha: str,
    provenance_sha256: str,
) -> None:
    if (
        progress.preparation_git_commit_sha != git_commit_sha
        or progress.preparation_provenance_sha256 != provenance_sha256
    ):
        raise Phase3ModelPreparationError(
            "existing Phase 3 progress preparation provenance differs"
        )


def _ensure_preparation_provenance_at(
    output: PinnedPhase3ModelOutput,
    provenance: SystemProvenance,
) -> Phase3PreparationProvenance:
    typed = SystemProvenance.model_validate(provenance.model_dump(mode="json"))
    authority = Phase3PreparationProvenance(
        provenance=typed,
        provenance_sha256=provenance_identity_sha256(typed),
    )
    payload = canonical_json_bytes(authority.model_dump(mode="json")) + b"\n"
    try:
        current = secure_fs.read_bytes_at(output.root_fd, PREPARATION_PROVENANCE_NAME)
    except secure_fs.SecureFilesystemError:
        try:
            _exclusive_claim_at(
                output.root_fd,
                PREPARATION_PROVENANCE_NAME,
                payload,
                staging_fd=output.staging_fd,
            )
        except (OSError, ArtifactValidationError) as exc:
            raise Phase3ModelPreparationError(
                "Phase 3 preparation provenance cannot be published"
            ) from exc
        # The path helper is used only for the initial claim; immediately bind
        # the result to the held root descriptor and recheck identity.
        output.recheck()
        current = secure_fs.read_bytes_at(output.root_fd, PREPARATION_PROVENANCE_NAME)
    try:
        parsed = Phase3PreparationProvenance.model_validate(json.loads(current))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Phase3ModelPreparationError("Phase 3 preparation provenance is invalid") from exc
    if parsed.provenance_sha256 != authority.provenance_sha256:
        raise Phase3ModelPreparationError(
            "Phase 3 preparation provenance changed after first publication"
        )
    return parsed


def _evidence_rows(lock: Phase3EvidenceLock) -> dict[tuple[str, int], dict[str, Any]]:
    rows = lock.body.get("evidence_artifacts")
    if not isinstance(rows, list) or len(rows) != EXPECTED_EVIDENCE:
        raise Phase3ModelPreparationError("Phase 3 evidence authority is not the exact 30-row matrix")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise Phase3ModelPreparationError("Phase 3 evidence row is not an object")
        family, replicate = row.get("family_id"), row.get("replicate")
        if not isinstance(family, str) or not isinstance(replicate, int):
            raise Phase3ModelPreparationError("Phase 3 evidence row identity is malformed")
        key = (family, replicate)
        if key in result:
            raise Phase3ModelPreparationError("Phase 3 evidence rows are duplicated")
        try:
            typed_key = TrainingDataEvidenceKey.model_validate(row["evidence_key"])
            typed_manifest = TrainingDataEvidenceManifest.model_validate(row["evidence_manifest"])
        except (KeyError, TypeError, ValueError) as exc:
            raise Phase3ModelPreparationError("Phase 3 evidence row is not typed") from exc
        if (
            typed_manifest.key != typed_key
            or row.get("evidence_id") != typed_manifest.evidence_id
            or row.get("evidence_key_id") != typed_key.key_id
            or typed_key.fold_id != row.get("fold_id")
            or typed_key.heldout_family_id != family
            or typed_key.replicate != replicate
        ):
            raise Phase3ModelPreparationError("Phase 3 evidence row lineage differs from its typed manifest")
        result[key] = {
            "row": row,
            "key": typed_key,
            "manifest": typed_manifest,
        }
    if set(result) != {(family, replicate) for family in FAMILIES for replicate in range(5)}:
        raise Phase3ModelPreparationError("Phase 3 evidence authority coverage is incomplete")
    return result


def _fold_by_family(runtime: ScreeningRuntime) -> dict[str, Any]:
    folds = tuple(getattr(runtime, "folds", ()))
    result = {str(fold.family_id): fold for fold in folds}
    if len(result) != len(folds) or len(result) != 6:
        raise Phase3ModelPreparationError("Phase 3 runtime fold matrix is incomplete")
    return result


def _read_payload_bundle(fold: Any, evidence_id: str, key: TrainingDataEvidenceKey) -> Any:
    store = getattr(fold, "store", None)
    try:
        pinned_run = getattr(store, "_open_pinned_run", None)
        if not callable(pinned_run):
            raise Phase3ModelPreparationError("Phase 3 runtime fold has no pinned RunStore context")
        with pinned_run() as root_fd:
            with open_training_data_reader(root_fd) as reader:
                return load_training_data_evidence_payload_bundle_from_at(
                    reader, evidence_id, expected_key=key
                )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Phase3ModelPreparationError("descriptor-pinned Phase 3 evidence read failed") from exc


def _validate_bundle_lineage(bundle: Any, evidence: Mapping[str, Any]) -> None:
    """Bind the exact descriptor-read manifest/payload bytes to the evidence row."""

    manifest = evidence["manifest"]
    row = evidence["row"]
    if (
        hashlib.sha256(bundle.manifest_bytes).hexdigest()
        != row.get("canonical_manifest_bytes_sha256")
        or bundle.manifest != manifest
        or hashlib.sha256(bundle.payload_bytes).hexdigest() != manifest.payload_sha256
        or len(bundle.payload_bytes) != manifest.payload_bytes
        or row.get("payload_sha256") != manifest.payload_sha256
        or row.get("payload_bytes") != manifest.payload_bytes
    ):
        raise Phase3ModelPreparationError("descriptor-read evidence bytes differ from frozen evidence authority")


def _prepared_view_for(
    view: Any,
    *,
    plan: Any,
    evidence: Mapping[str, Any],
    prepared_views: dict[str, Phase3ViewPreparation],
    payload_cache: dict[tuple[str, int], Any],
    folds: Mapping[str, Any],
) -> Phase3ViewPreparation:
    """Materialize a canonical view for resume-time accounting checks.

    This boundary intentionally prepares examples only; it never constructs or
    trains a model.  Recomputing the view is necessary because the stored key is
    not an authority for data-derived counts such as training examples and
    recurrent steps.
    """

    prepared = prepared_views.get(view.view_id)
    if prepared is not None:
        return prepared
    row = evidence["row"]
    view_ids = tuple(row.get("phase3_view_ids", ()))
    expected_view_ids = tuple(
        item.view_id
        for item in plan.views
        if item.heldout_family == view.heldout_family
        and item.replicate == view.replicate
    )
    if len(view_ids) != 4 or view_ids != expected_view_ids or view.view_id not in view_ids:
        raise Phase3ModelPreparationError("evidence-to-view mapping is not exactly four views")
    cache_key = (view.heldout_family, view.replicate)
    if cache_key not in payload_cache:
        try:
            payload_cache[cache_key] = _read_payload_bundle(
                folds[view.heldout_family], evidence["manifest"].evidence_id, evidence["key"]
            )
        except KeyError as exc:
            raise Phase3ModelPreparationError("model view has no exact evidence fold") from exc
    bundle = payload_cache[cache_key]
    _validate_bundle_lineage(bundle, evidence)
    try:
        prepared = prepare_phase3_view(
            bundle.payload,
            bundle.manifest,
            view,
            plan=plan,
            payload_bytes=bundle.payload_bytes,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Phase3ModelPreparationError("frozen Phase 3 view cannot be reconstructed") from exc
    prepared_views[view.view_id] = prepared
    return prepared


def _validate_resumed_model_accounting(
    key: Phase3ModelArtifactKey,
    *,
    owner: Any,
    prepared_view: Phase3ViewPreparation,
) -> None:
    """Bind stored report/identity fields to frozen data without retraining.

    Artifact schemas enforce internal consistency, but a forged key can make
    altered counts and identity values self-consistent.  The frozen owner and
    descriptor-read view are the independent authority for these fields.
    """

    expected_examples = len(prepared_view.examples)
    expected_forward_passes = owner.training_epochs * expected_examples
    expected_recurrent_steps = owner.training_epochs * sum(
        int(getattr(example, "history_features").shape[0])
        for example in prepared_view.examples
        if getattr(example, "history_features", None) is not None
    )
    expected_parameters = S_PARAMETERS if owner.condition_id == S_CONDITION else HISTORY_PARAMETERS
    expected_identity = _model_identity_sha256(
        owner,
        prepared_view,
        model_state_sha256=key.model_state_sha256,
    )
    if (
        key.report.trainable_parameters != expected_parameters
        or key.report.optimizer_steps != owner.training_epochs
        or key.report.training_examples != expected_examples
        or key.report.forward_passes != expected_forward_passes
        or key.report.recurrent_steps != expected_recurrent_steps
        or key.recurrent_steps != expected_recurrent_steps
        or key.model_identity_sha256 != expected_identity
    ):
        raise Phase3ModelPreparationError(
            "stored Phase 3 model report or identity differs from frozen view accounting"
        )


def _scan_existing(
    output_root: Path,
    expected_owner_ids: set[str],
    *,
    repairable_owner_ids: set[str] | None = None,
    preparation_git_commit_sha: str | None = None,
    preparation_provenance_sha256: str | None = None,
    pinned_output: PinnedPhase3ModelOutput | None = None,
) -> dict[str, Phase3ModelArtifactKey]:
    owns_root_fd = pinned_output is None
    if pinned_output is None:
        if not output_root.exists():
            return {}
        if output_root.is_symlink():
            raise Phase3ModelPreparationError("refusing symlinked Phase 3 model output root")
        keys_root = output_root / KEYS_DIR
        if not keys_root.exists():
            if (output_root / "phase3-model-artifacts").exists() or (output_root / "phase3-model-artifact-costs").exists():
                raise Phase3ModelPreparationError("Phase 3 model namespaces are incomplete")
            return {}
        if keys_root.is_symlink():
            raise Phase3ModelPreparationError("refusing symlinked Phase 3 model key namespace")
    result: dict[str, Phase3ModelArtifactKey] = {}
    try:
        if pinned_output is not None:
            pinned_output.recheck()
            root_fd = pinned_output.root_fd
        else:
            root_fd = secure_fs.open_directory_chain(output_root)
        try:
            with open_phase3_model_artifact_reader_at(root_fd) as reader:
                # The three namespaces form a small publication state machine:
                # artifact -> cost -> key index (the index is the commit
                # marker).  A crash may leave an artifact-only or
                # artifact+cost prefix, and those prefixes are repairable on a
                # subsequent deterministic write.  Any visible index must
                # point at an existing, fully validated artifact; an orphaned
                # cost/index or malformed entry is never silently adopted.
                manifests: dict[str, Any] = {}
                owners_by_artifact: dict[str, str] = {}
                try:
                    with os.scandir(reader.artifacts_fd) as iterator:
                        artifact_entries = tuple(iterator)
                except OSError as exc:
                    raise Phase3ModelPreparationError(
                        "Phase 3 model artifact inventory is unreadable"
                    ) from exc
                for entry in artifact_entries:
                    if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                        raise Phase3ModelPreparationError(
                            "Phase 3 artifact namespace contains an unsafe entry"
                        )
                    artifact_id = entry.name
                    if len(artifact_id) != 64 or any(
                        character not in "0123456789abcdef" for character in artifact_id
                    ):
                        raise Phase3ModelPreparationError(
                            "Phase 3 artifact namespace contains an unexpected entry"
                        )
                    try:
                        manifest = _load_manifest_at(reader, artifact_id)
                    except Exception as exc:
                        raise Phase3ModelPreparationError(
                            "stored Phase 3 model artifact is invalid"
                        ) from exc
                    owner_id = manifest.key.owner_id
                    if owner_id not in expected_owner_ids:
                        raise Phase3ModelPreparationError(
                            "stored Phase 3 model owner is outside the frozen universe"
                        )
                    if owner_id in owners_by_artifact.values():
                        raise Phase3ModelPreparationError(
                            "duplicate stored Phase 3 model owner"
                        )
                    manifests[artifact_id] = manifest
                    owners_by_artifact[artifact_id] = owner_id

                costs: dict[str, Phase3ModelArtifactCost] = {}
                try:
                    cost_names = tuple(secure_fs.regular_entries_at(reader.costs_fd))
                except (OSError, RuntimeError, ValueError) as exc:
                    raise Phase3ModelPreparationError(
                        "Phase 3 model cost inventory is unreadable"
                    ) from exc
                for name in cost_names:
                    if not name.endswith(".json"):
                        raise Phase3ModelPreparationError(
                            "unexpected Phase 3 model cost entry"
                        )
                    try:
                        raw = _fd_json(reader.costs_fd, name)
                        cost = _fd_model(Phase3ModelArtifactCost, raw, "model cost")
                        assert isinstance(cost, Phase3ModelArtifactCost)
                    except Exception as exc:
                        raise Phase3ModelPreparationError(
                            "stored Phase 3 model cost is invalid"
                        ) from exc
                    if name != f"{cost.key_id}.json" or cost.key.owner_id not in expected_owner_ids:
                        raise Phase3ModelPreparationError(
                            "stored Phase 3 model cost filename or owner differs"
                        )
                    if cost.key_id in costs:
                        raise Phase3ModelPreparationError(
                            "duplicate stored Phase 3 model cost"
                        )
                    costs[cost.key_id] = cost

                indexes: dict[str, Phase3ModelArtifactIndex] = {}
                for name in secure_fs.regular_entries_at(reader.keys_fd):
                    if not name.endswith(".json"):
                        raise Phase3ModelPreparationError("unexpected Phase 3 model key entry")
                    try:
                        raw = _fd_json(reader.keys_fd, name)
                        index = _fd_model(Phase3ModelArtifactIndex, raw, "model index")
                        assert isinstance(index, Phase3ModelArtifactIndex)
                    except Exception as exc:
                        raise Phase3ModelPreparationError("stored Phase 3 model index is invalid") from exc
                    key = index.key
                    if name != f"{index.key_id}.json":
                        raise Phase3ModelPreparationError("stored Phase 3 model index filename differs")
                    if key.owner_id not in expected_owner_ids:
                        raise Phase3ModelPreparationError("stored Phase 3 model owner is outside the frozen universe")
                    if preparation_git_commit_sha is not None and key.preparation_git_commit_sha != preparation_git_commit_sha:
                        raise Phase3ModelPreparationError("stored Phase 3 model preparation commit differs")
                    if preparation_provenance_sha256 is not None and key.preparation_provenance_sha256 != preparation_provenance_sha256:
                        raise Phase3ModelPreparationError("stored Phase 3 model preparation provenance differs")
                    if index.key_id in indexes:
                        raise Phase3ModelPreparationError("duplicate stored Phase 3 model index")
                    manifest = manifests.get(index.artifact_id)
                    if manifest is None:
                        raise Phase3ModelPreparationError(
                            "stored Phase 3 model index has no artifact"
                        )
                    if (
                        manifest.key != key
                        or _digest(manifest.model_dump(mode="json")) != index.manifest_sha256
                    ):
                        raise Phase3ModelPreparationError(
                            "stored Phase 3 model index does not match its artifact"
                        )
                    indexes[index.key_id] = index

                # Costs cannot exist without their artifact.  They may be
                # visible before the final key index because the index is the
                # last publication step.
                for key_id, cost in costs.items():
                    manifest = manifests.get(cost.artifact_id)
                    if manifest is None or manifest.key != cost.key:
                        raise Phase3ModelPreparationError(
                            "stored Phase 3 model cost has no matching artifact"
                        )

                # Only index+cost+artifact triples count as completed.  Partial
                # prefixes remain in place and are repaired by the deterministic
                # writer when their owner is selected on a subsequent run.
                for key_id, index in indexes.items():
                    cost = costs.get(key_id)
                    if cost is None:
                        continue
                    key = index.key
                    if key.owner_id in result:
                        raise Phase3ModelPreparationError("duplicate stored Phase 3 model owner")
                    load_phase3_model_bundle_from_at(reader, key)
                    result[key.owner_id] = key
                partial_owners = set(owners_by_artifact.values()) - set(result)
                if repairable_owner_ids is not None and not partial_owners.issubset(
                    repairable_owner_ids
                ):
                    raise Phase3ModelPreparationError(
                        "incomplete Phase 3 publication belongs to an unselected owner"
                    )
        finally:
            if owns_root_fd:
                os.close(root_fd)
    except Phase3ModelPreparationError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Phase3ModelPreparationError("stored Phase 3 model inventory cannot be revalidated") from exc
    return result



def _accounting(preparation: Phase3ModelPreparation) -> TrainingPreparationAccounting:
    return TrainingPreparationAccounting(
        training=PhaseAccounting(
            optimizer_steps=int(preparation.report.optimizer_steps),
            forward_passes=int(preparation.report.forward_passes),
        ),
        serialization=PhaseAccounting(calls=1),
    )


def _recheck_authority_before_success(
    runtime: ScreeningRuntime,
    *,
    authority_repository: str | Path | None,
    authority_repository_identity: tuple[int, int] | None,
    plan_lock_bytes: bytes | None,
    anchor_file_bytes: bytes | None,
    evidence_lock_bytes: bytes | None,
    preparation_provenance: Phase3PreparationProvenance | None,
) -> None:
    if authority_repository is None:
        return
    if (
        authority_repository_identity is None
        or plan_lock_bytes is None
        or anchor_file_bytes is None
        or evidence_lock_bytes is None
        or preparation_provenance is None
    ):
        raise Phase3ModelPreparationError("Phase 3 final authority handoff is incomplete")
    repository_fd = secure_fs.open_directory_chain(authority_repository)
    try:
        if secure_fs.directory_identity(repository_fd) != authority_repository_identity:
            raise Phase3ModelPreparationError("Phase 3 authority repository was replaced")
        milestone6_fd = secure_fs.open_child_chain(repository_fd, "configs", "milestone6")
        try:
            observed = (
                secure_fs.read_bytes_at(milestone6_fd, "phase3_plan_lock.json"),
                secure_fs.read_bytes_at(milestone6_fd, "phase3_anchor_manifest.json"),
                secure_fs.read_bytes_at(milestone6_fd, "phase3_evidence_lock.json"),
            )
        finally:
            os.close(milestone6_fd)
    finally:
        os.close(repository_fd)
    if observed != (plan_lock_bytes, anchor_file_bytes, evidence_lock_bytes):
        raise Phase3ModelPreparationError("Phase 3 authority bytes changed before success")
    device_policy = getattr(runtime, "device_policy", None)
    if device_policy is None:
        raise Phase3ModelPreparationError("Phase 3 runtime has no device policy")
    current = capture_system_provenance(authority_repository, device_policy)
    if (
        current.git_dirty
        or current.git_commit_sha != preparation_provenance.provenance.git_commit_sha
        or provenance_identity_sha256(current) != preparation_provenance.provenance_sha256
    ):
        raise Phase3ModelPreparationError("Phase 3 preparation provenance changed before success")


def _prepare_phase3_model_batch_pinned(
    output_root: str | Path,
    *,
    runtime: ScreeningRuntime,
    validated_plan: ValidatedPhase3Plan,
    anchor_manifest: Phase3AnchorManifest,
    evidence_lock: Phase3EvidenceLock,
    owner_ids: Iterable[str] | None = None,
    limit: int | None = None,
    authority_repository: str | Path | None = None,
    authority_repository_identity: tuple[int, int] | None = None,
    plan_lock_bytes: bytes | None = None,
    anchor_file_bytes: bytes | None = None,
    evidence_lock_bytes: bytes | None = None,
    preparation_provenance: SystemProvenance | None = None,
    pinned_output: PinnedPhase3ModelOutput,
) -> Phase3ModelPreparationResult:
    """Prepare and persist the frozen 480-owner model universe.

    ``owner_ids`` and ``limit`` are bounded development conveniences.  A bounded
    invocation remains partial and can never write a ``complete`` progress state.
    Existing artifacts are fully reloaded (key, index, cost, manifest, and tensors)
    before they are counted as resumed.
    """
    if owner_ids is not None and limit is not None:
        raise Phase3ModelPreparationError("owner_ids and limit cannot both be supplied")
    explicit_authority = (
        authority_repository,
        authority_repository_identity,
        plan_lock_bytes,
        anchor_file_bytes,
        evidence_lock_bytes,
    )
    if any(value is not None for value in explicit_authority) and not all(
        value is not None for value in explicit_authority
    ):
        raise Phase3ModelPreparationError(
            "Phase 3 authority handoff requires repository, identity, and all retained bytes"
        )
    if authority_repository_identity is not None and (
        not isinstance(authority_repository_identity, tuple)
        or len(authority_repository_identity) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in authority_repository_identity
        )
    ):
        raise Phase3ModelPreparationError("Phase 3 authority repository identity is invalid")
    typed_preparation_provenance: Phase3PreparationProvenance | None = None
    if preparation_provenance is not None:
        try:
            typed_preparation_provenance = Phase3PreparationProvenance(
                provenance=SystemProvenance.model_validate(
                    preparation_provenance.model_dump(mode="json")
                ),
                provenance_sha256=provenance_identity_sha256(preparation_provenance),
            )
        except (TypeError, ValueError) as exc:
            raise Phase3ModelPreparationError("Phase 3 preparation provenance is invalid") from exc
    try:
        require_phase3_anchor_manifest(anchor_manifest)
        require_phase3_evidence_lock(evidence_lock)
    except (TypeError, ValueError) as exc:
        raise Phase3ModelPreparationError("Phase 3 preparation authorities are not canonical") from exc
    if not isinstance(validated_plan, ValidatedPhase3Plan):
        raise Phase3ModelPreparationError("Phase 3 preparation requires an opaque validated plan")
    plan = validated_plan.plan
    try:
        if authority_repository is None:
            committed_plan_bytes = load_committed_phase3_plan_lock_bytes()
            committed_anchor_bytes = load_committed_phase3_anchor_manifest_bytes()
            committed_evidence_bytes = load_committed_phase3_evidence_lock_bytes()
        else:
            runtime_repository = getattr(
                runtime,
                "authority_repository",
                getattr(runtime, "repository", None),
            )
            if runtime_repository is None or Path(authority_repository).resolve(strict=False) != Path(
                runtime_repository
            ).resolve(strict=False):
                raise Phase3ModelPreparationError(
                    "Phase 3 authority repository differs from the runtime repository"
                )
            repository_fd = secure_fs.open_directory_chain(authority_repository)
            try:
                if secure_fs.directory_identity(repository_fd) != authority_repository_identity:
                    raise Phase3ModelPreparationError(
                        "Phase 3 authority repository identity differs from the retained handoff"
                    )
                milestone6_fd = secure_fs.open_child_chain(
                    repository_fd, "configs", "milestone6"
                )
                try:
                    committed_plan_bytes = secure_fs.read_bytes_at(
                        milestone6_fd, "phase3_plan_lock.json"
                    )
                    committed_anchor_bytes = secure_fs.read_bytes_at(
                        milestone6_fd, "phase3_anchor_manifest.json"
                    )
                    committed_evidence_bytes = secure_fs.read_bytes_at(
                        milestone6_fd, "phase3_evidence_lock.json"
                    )
                finally:
                    os.close(milestone6_fd)
            finally:
                os.close(repository_fd)
        if (
            committed_plan_bytes != plan_lock_bytes
                or committed_anchor_bytes != anchor_file_bytes
                or committed_evidence_bytes != evidence_lock_bytes
            ):
                raise Phase3ModelPreparationError(
                    "Phase 3 authority bytes changed after the driver handoff"
                )
        if preparation_provenance is None:
            raise Phase3ModelPreparationError(
                "Phase 3 preparation requires captured authority provenance"
            )
        if authority_repository is not None:
            device_policy = getattr(runtime, "device_policy", None)
            if device_policy is None:
                raise Phase3ModelPreparationError(
                    "Phase 3 preparation runtime has no device policy for provenance"
                )
            current_provenance = capture_system_provenance(
                authority_repository, device_policy
            )
            if current_provenance.git_dirty or provenance_identity_sha256(
                current_provenance
            ) != typed_preparation_provenance.provenance_sha256:
                raise Phase3ModelPreparationError(
                    "Phase 3 preparation authority provenance changed before execution"
                )
        retained_plan = validate_phase3_plan_lock_bytes(committed_plan_bytes)
        if retained_plan != plan:
            raise Phase3ModelPreparationError(
                "validated plan differs from the retained Phase 3 plan lock"
            )
        if committed_anchor_bytes != anchor_manifest.canonical_bytes:
            raise Phase3ModelPreparationError(
                "validated anchor differs from the retained Phase 3 anchor"
            )
        if committed_evidence_bytes != evidence_lock.canonical_bytes:
            raise Phase3ModelPreparationError(
                "validated evidence differs from the retained Phase 3 evidence lock"
            )
    except Phase3ModelPreparationError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Phase3ModelPreparationError("committed Phase 3 authorities cannot be read safely") from exc
    lineage = evidence_lock.body.get("lineage")
    if not isinstance(lineage, Mapping) or (
        lineage.get("phase3_plan_id"),
        lineage.get("phase3_protocol_sha256"),
        lineage.get("phase3_anchor_manifest_sha256"),
    ) != (
        plan.plan_id,
        plan.protocol_sha256,
        anchor_manifest.anchor_manifest_sha256,
    ):
        raise Phase3ModelPreparationError("Phase 3 evidence lineage differs from plan or anchor authority")
    owners = tuple(plan.model_owners)
    if len(owners) != EXPECTED_MODELS or len({owner.owner_id for owner in owners}) != EXPECTED_MODELS:
        raise Phase3ModelPreparationError("Phase 3 plan does not contain exactly 480 unique owners")
    try:
        recheck_screening_runtime_readonly(runtime)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Phase3ModelPreparationError("Phase 3 preparation requires a freshly rechecked runtime") from exc
    rows = _evidence_rows(evidence_lock)
    folds = _fold_by_family(runtime)
    views = {view.view_id: view for view in plan.views}
    if len(views) != EXPECTED_VIEWS:
        raise Phase3ModelPreparationError("Phase 3 plan does not contain exactly 120 views")
    selected_ids = tuple(owner.owner_id for owner in owners)
    if owner_ids is not None:
        selected_ids = tuple(owner_ids)
        if len(set(selected_ids)) != len(selected_ids) or any(item not in {o.owner_id for o in owners} for item in selected_ids):
            raise Phase3ModelPreparationError("owner_ids contain missing or extra owners")
    elif limit is not None:
        if not isinstance(limit, int) or limit < 0 or limit > EXPECTED_MODELS:
            raise Phase3ModelPreparationError("limit is outside the frozen owner universe")
        selected_ids = tuple(owner.owner_id for owner in owners[:limit])
    owner_map = {owner.owner_id: owner for owner in owners}
    out = Path(output_root)
    progress_path = out / PROGRESS_NAME
    pinned_output.recheck()
    prior = _read_progress_at(pinned_output)
    if typed_preparation_provenance is not None:
        typed_preparation_provenance = _ensure_preparation_provenance_at(
            pinned_output, typed_preparation_provenance.provenance
        )
    preparation_git_commit_sha = (
        typed_preparation_provenance.provenance.git_commit_sha
        if typed_preparation_provenance is not None
        else None
    )
    preparation_provenance_sha256 = (
        typed_preparation_provenance.provenance_sha256
        if typed_preparation_provenance is not None
        else None
    )
    expected_ids = tuple(owner.owner_id for owner in owners)
    if prior is not None and (
        prior.plan_id != plan.plan_id
        or prior.protocol_sha256 != plan.protocol_sha256
        or prior.anchor_manifest_sha256 != anchor_manifest.anchor_manifest_sha256
        or prior.evidence_lock_sha256 != evidence_lock.evidence_lock_sha256
        or prior.expected_owner_ids != expected_ids
    ):
        raise Phase3ModelPreparationError("existing Phase 3 progress authority differs")
    if prior is not None and preparation_git_commit_sha is not None:
        _validate_progress_preparation_provenance(
            prior,
            git_commit_sha=preparation_git_commit_sha,
            provenance_sha256=preparation_provenance_sha256 or "",
        )
    existing = _scan_existing(
        out,
        set(expected_ids),
        repairable_owner_ids=set(selected_ids),
        preparation_git_commit_sha=preparation_git_commit_sha,
        preparation_provenance_sha256=preparation_provenance_sha256,
        pinned_output=pinned_output,
    )
    # These caches are shared by resume validation and fresh preparation.  The
    # resume path must reconstruct the canonical view (without training) before
    # accepting any stored data-derived report fields.
    prepared_views: dict[str, Phase3ViewPreparation] = {}
    payload_cache: dict[tuple[str, int], Any] = {}
    for owner_id, key in existing.items():
        owner = owner_map[owner_id]
        view = views.get(owner.view_id)
        evidence = rows.get((owner.heldout_family, owner.replicate))
        if view is None or evidence is None:
            raise Phase3ModelPreparationError("stored Phase 3 model has no exact plan/evidence lineage")
        manifest = evidence["manifest"]
        prepared_view = _prepared_view_for(
            view,
            plan=plan,
            evidence=evidence,
            prepared_views=prepared_views,
            payload_cache=payload_cache,
            folds=folds,
        )
        _validate_resumed_model_accounting(
            key,
            owner=owner,
            prepared_view=prepared_view,
        )
        if (
            key.plan_id,
            key.protocol_sha256,
            key.evidence_lock_sha256,
            key.evidence_payload_sha256,
            key.evidence_payload_bytes,
            key.view_id,
            key.owner_id,
            key.condition_id,
            key.fold_id,
            key.heldout_family,
            key.replicate,
            key.training_tuple_id,
            key.model_seed,
            key.optimizer.learning_rate,
            key.report.optimizer_steps,
        ) != (
            plan.plan_id,
            plan.protocol_sha256,
            evidence_lock.evidence_lock_sha256,
            manifest.payload_sha256,
            manifest.payload_bytes,
            view.view_id,
            owner.owner_id,
            owner.condition_id,
            owner.fold_id,
            owner.heldout_family,
            owner.replicate,
            owner.training_tuple_id,
            owner.model_seed,
            owner.learning_rate,
            owner.training_epochs,
        ):
            raise Phase3ModelPreparationError("stored Phase 3 model key lineage differs from frozen authority")
    completed = set(prior.completed_owner_ids if prior is not None else ())
    if not set(completed).issubset(existing):
        raise Phase3ModelPreparationError("Phase 3 progress does not match revalidated stored artifacts")
    # A crash can publish the artifact/index/cost after the progress replace but
    # before the process records completion.  Rebuild from the fully validated
    # artifact inventory instead of discarding that durable work.
    completed = set(existing)
    progress = Phase3ModelPreparationProgress(
        plan_id=plan.plan_id,
        protocol_sha256=plan.protocol_sha256,
        anchor_manifest_sha256=anchor_manifest.anchor_manifest_sha256,
        evidence_lock_sha256=evidence_lock.evidence_lock_sha256,
        expected_owner_ids=expected_ids,
        completed_owner_ids=tuple(owner.owner_id for owner in owners if owner.owner_id in completed),
        status="running",
        evidence_count=EXPECTED_EVIDENCE,
        view_count=EXPECTED_VIEWS,
        model_count=len(completed),
        preparation_git_commit_sha=preparation_git_commit_sha or "0" * 64,
        preparation_provenance_sha256=preparation_provenance_sha256 or "0" * 64,
    )
    _atomic_progress_at(pinned_output, progress)
    try:
        for owner_id in selected_ids:
            if owner_id in completed:
                continue
            owner = owner_map[owner_id]
            view = views.get(owner.view_id)
            if view is None or (view.condition_id, view.fold_id, view.heldout_family, view.replicate) != (
                owner.condition_id,
                owner.fold_id,
                owner.heldout_family,
                owner.replicate,
            ):
                raise Phase3ModelPreparationError("model owner/view lineage differs from plan")
            evidence = rows.get((view.heldout_family, view.replicate))
            if evidence is None:
                raise Phase3ModelPreparationError("model view has no exact evidence row")
            prepared_view = _prepared_view_for(
                view,
                plan=plan,
                evidence=evidence,
                prepared_views=prepared_views,
                payload_cache=payload_cache,
                folds=folds,
            )
            preparation = prepare_phase3_model(prepared_view, owner, plan=plan)
            write_phase3_model_artifact(
                out,
                preparation=preparation,
                plan_id=plan.plan_id,
                protocol_sha256=plan.protocol_sha256,
                evidence_lock_sha256=evidence_lock.evidence_lock_sha256,
                preparation_git_commit_sha=preparation_git_commit_sha or "0" * 64,
                preparation_provenance_sha256=preparation_provenance_sha256 or "0" * 64,
                accounting=_accounting(preparation),
                pinned_output=pinned_output,
            )
            completed.add(owner_id)
            progress = progress.model_copy(
                update={
                    "completed_owner_ids": tuple(owner.owner_id for owner in owners if owner.owner_id in completed),
                    "model_count": len(completed),
                    "status": "running",
                    "error": None,
                }
            )
            _atomic_progress_at(pinned_output, progress)
            pinned_output.recheck()
        # A long batch must not finish on stale runtime/output authorities.
        pinned_output.recheck()
        try:
            recheck_screening_runtime_readonly(runtime)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise Phase3ModelPreparationError(
                "Phase 3 preparation runtime changed before final inventory"
            ) from exc
        _recheck_authority_before_success(
            runtime,
            authority_repository=authority_repository,
            authority_repository_identity=authority_repository_identity,
            plan_lock_bytes=plan_lock_bytes,
            anchor_file_bytes=anchor_file_bytes,
            evidence_lock_bytes=evidence_lock_bytes,
            preparation_provenance=typed_preparation_provenance,
        )
        strict_existing = _scan_existing(
            out,
            set(expected_ids),
            repairable_owner_ids=set(),
            preparation_git_commit_sha=preparation_git_commit_sha,
            preparation_provenance_sha256=preparation_provenance_sha256,
            pinned_output=pinned_output,
        )
        if set(strict_existing) != completed:
            raise Phase3ModelPreparationError(
                "successful Phase 3 preparation inventory differs from completed owners"
            )
    except Exception as exc:
        failed = progress.model_copy(update={"status": "failed", "error": str(exc)[:500]})
        _atomic_progress_at(pinned_output, failed)
        raise
    complete = len(completed) == EXPECTED_MODELS and set(selected_ids) == set(expected_ids)
    final = progress.model_copy(
        update={"status": "complete" if complete else "running", "error": None}
    )
    _atomic_progress_at(pinned_output, final)
    return Phase3ModelPreparationResult(
        plan_id=plan.plan_id,
        evidence_count=EXPECTED_EVIDENCE,
        view_count=EXPECTED_VIEWS,
        model_count=len(completed),
        completed_owner_ids=tuple(owner.owner_id for owner in owners if owner.owner_id in completed),
        complete=complete,
        progress_path=str(progress_path),
    )


def prepare_phase3_model_batch(
    output_root: str | Path,
    *,
    runtime: ScreeningRuntime,
    validated_plan: ValidatedPhase3Plan,
    anchor_manifest: Phase3AnchorManifest,
    evidence_lock: Phase3EvidenceLock,
    owner_ids: Iterable[str] | None = None,
    limit: int | None = None,
    authority_repository: str | Path | None = None,
    authority_repository_identity: tuple[int, int] | None = None,
    plan_lock_bytes: bytes | None = None,
    anchor_file_bytes: bytes | None = None,
    evidence_lock_bytes: bytes | None = None,
    preparation_provenance: SystemProvenance | None = None,
) -> Phase3ModelPreparationResult:
    """Run preparation while retaining one pinned output namespace set."""
    handoff = (
        authority_repository,
        authority_repository_identity,
        plan_lock_bytes,
        anchor_file_bytes,
        evidence_lock_bytes,
        preparation_provenance,
    )
    if not all(value is not None for value in handoff):
        raise Phase3ModelPreparationError(
            "Phase 3 authority handoff requires repository, identity, retained bytes, and provenance"
        )
    with open_phase3_model_output(output_root) as pinned_output:
        return _prepare_phase3_model_batch_pinned(
            output_root,
            runtime=runtime,
            validated_plan=validated_plan,
            anchor_manifest=anchor_manifest,
            evidence_lock=evidence_lock,
            owner_ids=owner_ids,
            limit=limit,
            authority_repository=authority_repository,
            authority_repository_identity=authority_repository_identity,
            plan_lock_bytes=plan_lock_bytes,
            anchor_file_bytes=anchor_file_bytes,
            evidence_lock_bytes=evidence_lock_bytes,
            preparation_provenance=preparation_provenance,
            pinned_output=pinned_output,
        )


prepare_phase3_models = prepare_phase3_model_batch


__all__ = [
    "EXPECTED_EVIDENCE",
    "EXPECTED_MODELS",
    "EXPECTED_VIEWS",
    "Phase3ModelPreparationError",
    "Phase3ModelPreparationProgress",
    "Phase3ModelPreparationResult",
    "prepare_phase3_model_batch",
    "prepare_phase3_models",
]
