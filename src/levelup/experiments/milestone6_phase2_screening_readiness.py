"""Prepare the complete, development-only Phase 2 screening inventory.

This is an explicit preparation boundary.  It captures and applies one runtime
provenance, materializes the frozen data and model artifacts for the six known
development folds, and writes only plans, provenance, and preparation artifacts.
It deliberately has no validation, search, outcome, selection, or aggregation
entry points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from levelup.experiments.milestone6_phase2 import (
    DEVELOPMENT_PROTOCOL_PATH,
    DEVELOPMENT_TASKS_PATH,
    ROOT,
)
from levelup.experiments.milestone6_phase2_screening import (
    SCREENING_CANDIDATES_PATH,
    build_screening_plan,
    screening_child_configs,
    validate_screening_plan,
)
from levelup.experiments.milestone6_phase2_screening_models import (
    MaterializedScreeningModels,
    materialize_screening_models,
)
from levelup.experiments.milestone6_phase2_screening_preparation import (
    MaterializedScreeningData,
    ScreeningDataKeys,
    ScreeningModelKeys,
    build_screening_data_keys,
    build_screening_model_keys,
    build_screening_shared_plan,
    materialize_screening_data,
)
from levelup.experiments.runner.config import (
    ExperimentConfig,
    canonical_json_bytes,
    run_id_for,
    scientific_config_sha256,
    scientific_config_value,
)
from levelup.experiments.runner.provenance import (
    apply_runtime_policy,
    capture_system_provenance,
)
from levelup.experiments.runner.records import ExpectedSharedArtifacts, SystemProvenance
from levelup.experiments.runner.storage import (
    RunStore,
    expected_units_sha256,
    plan_expected_units,
    provenance_identity_sha256,
)
from levelup.experiments.runner.training_data_artifacts import TrainingDataArtifactError

PreparationEvent = Any
_SOURCE_ROWS = (
    (DEVELOPMENT_PROTOCOL_PATH, "protocol_sha256"),
    (SCREENING_CANDIDATES_PATH, "screening_candidates_sha256"),
    (DEVELOPMENT_TASKS_PATH, "task_manifest_sha256"),
)
_SCHEMA_VERSION = "milestone6.phase2.screening-readiness.v1"


@dataclass(frozen=True, slots=True)
class _AuthoritySourceSnapshot:
    """Immutable bytes and digests for the three authority inputs."""

    bytes_by_path: dict[Path, bytes]
    digests: dict[str, str]


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _source_snapshot() -> _AuthoritySourceSnapshot:
    bytes_by_path = {path: path.read_bytes() for path, _ in _SOURCE_ROWS}
    digests = {
        field: hashlib.sha256(bytes_by_path[path]).hexdigest() for path, field in _SOURCE_ROWS
    }
    return _AuthoritySourceSnapshot(bytes_by_path=bytes_by_path, digests=digests)


def _authority_snapshot() -> _AuthoritySourceSnapshot:
    """Testable source snapshot hook used at every preparation boundary."""

    return _source_snapshot()


def _assert_source_snapshot(snapshot: _AuthoritySourceSnapshot) -> None:
    if any(path.read_bytes() != content for path, content in snapshot.bytes_by_path.items()):
        raise TrainingDataArtifactError(
            "Phase 2 authority source changed during readiness preparation"
        )


def _safe_dir(path: Path, *, create: bool = False) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if os.path.lexists(candidate) and candidate.is_symlink():
            raise TrainingDataArtifactError(
                f"readiness directory path contains a symlink: {candidate}"
            )
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise TrainingDataArtifactError(f"unsafe readiness directory: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise TrainingDataArtifactError(f"readiness directory is unavailable: {path}")


def _child_run_dir(root: Path, config: ExperimentConfig) -> Path:
    """Resolve one child path without allowing links or root escape."""

    if root.is_symlink():
        raise TrainingDataArtifactError("readiness output root cannot be a symlink")
    _safe_dir(root, create=True)
    child = root / run_id_for(config)
    if child.is_symlink():
        raise TrainingDataArtifactError("readiness child run directory cannot be a symlink")
    child_resolved = child.resolve()
    root_resolved = root.resolve()
    try:
        child_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise TrainingDataArtifactError(
            "readiness child run directory escapes output root"
        ) from exc
    if child_resolved.parent != root_resolved:
        raise TrainingDataArtifactError("readiness child run directory is not a direct child")
    return child


def _claim_readiness_intent(path: Path, body: dict[str, Any]) -> bool:
    """Exclusively claim one child; a prior matching claim is resumable."""

    payload = canonical_json_bytes(body) + b"\n"
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise TrainingDataArtifactError("screening readiness intent conflicts")
        return False
    _exclusive_json(path, body)
    return True


_CHILD_TOP_LEVEL_FILES = frozenset(
    {
        "screening-readiness-intent.json",
        "config.json",
        "expected-units.json",
        "expected-shared-artifacts.json",
        "provenance.json",
    }
)
_CHILD_TOP_LEVEL_DIRECTORIES = frozenset(
    {
        "units",
        "attempts",
        "screening-data-intents",
        "training-data-evidence-costs",
        "training-data-view-costs",
        "training-data-artifact-keys",
        "training-data-evidence",
        "training-data-artifacts",
        "screening-model-intents",
        "training-artifact-costs",
        "training-artifact-keys",
        "training-artifacts",
    }
)
_CHILD_TOP_LEVEL_NAMES = _CHILD_TOP_LEVEL_FILES | _CHILD_TOP_LEVEL_DIRECTORIES


def _validate_child_top_level(run_dir: Path) -> None:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise TrainingDataArtifactError("screening readiness child directory is unsafe")
    for item in run_dir.iterdir():
        if item.name not in _CHILD_TOP_LEVEL_NAMES or item.is_symlink():
            raise TrainingDataArtifactError(
                "screening readiness child has partial or unknown state"
            )
        if item.name in _CHILD_TOP_LEVEL_FILES and not item.is_file():
            raise TrainingDataArtifactError(
                "screening readiness child file namespace has the wrong type"
            )
        if item.name in _CHILD_TOP_LEVEL_DIRECTORIES and not item.is_dir():
            raise TrainingDataArtifactError(
                "screening readiness child directory namespace has the wrong type"
            )


def _canonical_file_matches(path: Path, value: Any, *, label: str) -> None:
    if not os.path.lexists(path):
        return
    expected = canonical_json_bytes(value) + b"\n"
    if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
        raise TrainingDataArtifactError(f"stored readiness {label} conflicts")


def _load_child_provenance(path: Path) -> SystemProvenance | None:
    if not os.path.lexists(path):
        return None
    if path.is_symlink() or not path.is_file():
        raise TrainingDataArtifactError("stored child provenance is unsafe")
    payload_bytes = path.read_bytes()
    try:
        value = SystemProvenance.model_validate(json.loads(payload_bytes))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TrainingDataArtifactError("stored child provenance is invalid") from exc
    if payload_bytes != canonical_json_bytes(value.model_dump(mode="json")) + b"\n":
        raise TrainingDataArtifactError("stored child provenance is not canonical")
    return value


def _validate_existing_child_files(
    run_dir: Path,
    config: ExperimentConfig,
    *,
    current_provenance_sha256: str,
) -> SystemProvenance | None:
    """Reject durable child conflicts before paid preparation can begin."""

    _canonical_file_matches(
        run_dir / "config.json",
        scientific_config_value(config),
        label="config",
    )
    _canonical_file_matches(
        run_dir / "expected-units.json",
        plan_expected_units(config).model_dump(mode="json"),
        label="expected-unit plan",
    )
    persisted = _load_child_provenance(run_dir / "provenance.json")
    if persisted is not None and provenance_identity_sha256(persisted) != current_provenance_sha256:
        raise TrainingDataArtifactError("stored child provenance differs from the current runtime")
    shared_path = run_dir / "expected-shared-artifacts.json"
    if os.path.lexists(shared_path):
        if shared_path.is_symlink() or not shared_path.is_file():
            raise TrainingDataArtifactError("stored shared plan is unsafe")
        payload_bytes = shared_path.read_bytes()
        try:
            shared = ExpectedSharedArtifacts.model_validate(json.loads(payload_bytes))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise TrainingDataArtifactError("stored shared plan is invalid") from exc
        if payload_bytes != canonical_json_bytes(shared.model_dump(mode="json")) + b"\n":
            raise TrainingDataArtifactError("stored shared plan is not canonical")
        if (
            shared.run_id != run_id_for(config)
            or shared.config_sha256 != scientific_config_sha256(config)
            or len(shared.artifacts) != 80
        ):
            raise TrainingDataArtifactError("stored shared plan authority conflicts")
        try:
            RunStore(
                run_dir.parent,
                config,
                repository=run_dir.parent,
                shared_artifacts=tuple(shared.artifacts),
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise TrainingDataArtifactError(
                "stored shared plan ownership or consumer authority conflicts"
            ) from exc
        # The shared plan is published only after all data and models exist. If
        # it is present on resume, require complete namespaces so the public
        # materializers can only reload and validate; they cannot fill gaps or
        # repeat paid preparation before the exact concrete plan comparison.
        for name, expected_count, entry_kind in (
            ("screening-data-intents", 5, "file"),
            ("training-data-evidence-costs", 5, "file"),
            ("training-data-view-costs", 15, "file"),
            ("training-data-artifact-keys", 15, "file"),
            ("training-data-evidence", 5, "directory"),
            ("training-data-artifacts", 15, "directory"),
            ("screening-model-intents", 60, "file"),
            ("training-artifact-costs", 60, "file"),
            ("training-artifact-keys", 60, "file"),
            ("training-artifacts", 60, "directory"),
        ):
            namespace = run_dir / name
            if namespace.is_symlink() or not namespace.is_dir():
                raise TrainingDataArtifactError(
                    "stored shared plan has an incomplete preparation namespace"
                )
            entries = tuple(namespace.iterdir())
            if len(entries) != expected_count or any(
                entry.is_symlink()
                or (entry_kind == "file" and not entry.is_file())
                or (entry_kind == "directory" and not entry.is_dir())
                for entry in entries
            ):
                raise TrainingDataArtifactError(
                    "stored shared plan has an unsafe preparation inventory"
                )
    return persisted


def _exclusive_json(path: Path, value: Any) -> None:
    """Publish one canonical JSON object without replacing a prior object."""

    payload = canonical_json_bytes(value) + b"\n"
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"readiness artifact conflicts: {path.name}")
        return
    _safe_dir(path.parent, create=True)
    if path.parent.is_symlink():
        raise RuntimeError("readiness artifact parent is a symlink")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError(f"readiness artifact has a conflicting winner: {path.name}")
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


class ReadinessComputeReport(BaseModel):
    """First-writer compute report for one prepared shared model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_condition_id: str = Field(min_length=1)
    training_tuple_id: str = Field(min_length=1)
    replicate: int = Field(ge=0)
    model_key_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_artifact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str = Field(min_length=1)
    objective_id: str = Field(min_length=1)
    trainable_parameters: int = Field(ge=0)
    training_examples: int = Field(ge=0)
    optimizer_steps: int = Field(ge=0)
    forward_passes: int = Field(ge=0)
    training_wall_seconds: float = Field(ge=0, allow_inf_nan=False)


class ScreeningReadinessChild(BaseModel):
    """Typed inventory and provenance for one held-out development fold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    heldout_family_id: str = Field(min_length=1)
    fold_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_units_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shared_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_key_ids: tuple[str, ...]
    evidence_artifact_ids: tuple[str, ...]
    view_key_ids: tuple[str, ...]
    view_artifact_ids: tuple[str, ...]
    model_key_ids: tuple[str, ...]
    model_artifact_ids: tuple[str, ...]
    shared_artifact_key_ids: tuple[str, ...]
    compute_reports: tuple[ReadinessComputeReport, ...]
    expected_units: Literal[1520] = 1520
    expected_evidence_artifacts: Literal[5] = 5
    expected_training_data_views: Literal[15] = 15
    expected_model_artifacts: Literal[60] = 60
    expected_shared_artifacts: Literal[80] = 80

    @model_validator(mode="after")
    def inventory_is_exact(self) -> "ScreeningReadinessChild":
        for name, values, expected in (
            ("evidence keys", self.evidence_key_ids, 5),
            ("evidence artifacts", self.evidence_artifact_ids, 5),
            ("view keys", self.view_key_ids, 15),
            ("view artifacts", self.view_artifact_ids, 15),
            ("model keys", self.model_key_ids, 60),
            ("model artifacts", self.model_artifact_ids, 60),
            ("shared keys", self.shared_artifact_key_ids, 80),
        ):
            if len(values) != expected or tuple(values) != tuple(sorted(values)):
                raise ValueError(f"{name} inventory is not sorted and exact")
            if len(set(values)) != expected:
                raise ValueError(f"{name} inventory contains duplicates")
        if len(self.compute_reports) != 60:
            raise ValueError("compute report inventory is not exact")
        return self


class ScreeningReadinessManifest(BaseModel):
    """Canonical, preparation-only manifest for all six development folds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[_SCHEMA_VERSION] = _SCHEMA_VERSION
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    screening_plan_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    screening_candidates_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: SystemProvenance
    family_order: tuple[str, ...]
    child_run_ids: tuple[str, ...]
    children: tuple[ScreeningReadinessChild, ...]
    evidence_key_ids: tuple[str, ...]
    view_key_ids: tuple[str, ...]
    model_key_ids: tuple[str, ...]
    shared_artifact_key_ids: tuple[str, ...]
    model_artifact_ids: tuple[str, ...]
    expected_total_units: Literal[9120] = 9120
    expected_total_evidence_artifacts: Literal[30] = 30
    expected_total_training_data_views: Literal[90] = 90
    expected_total_model_artifacts: Literal[360] = 360
    expected_total_shared_artifacts: Literal[480] = 480
    development_only: Literal[True] = True
    validation_executed: Literal[False] = False
    search_executed: Literal[False] = False
    outcomes_present: Literal[False] = False
    selection_performed: Literal[False] = False
    final_family_access: Literal[False] = False

    @property
    def expected_manifest_sha256(self) -> str:
        return _digest(self.model_dump(mode="json", exclude={"manifest_sha256"}))

    @model_validator(mode="after")
    def manifest_is_canonical(self) -> "ScreeningReadinessManifest":
        if self.manifest_sha256 != self.expected_manifest_sha256:
            raise ValueError("readiness manifest identity mismatch")
        if (
            len(self.children) != 6
            or tuple(child.heldout_family_id for child in self.children) != self.family_order
        ):
            raise ValueError("readiness child family order drifted")
        if tuple(child.run_id for child in self.children) != self.child_run_ids:
            raise ValueError("readiness child run order drifted")
        for name, values, expected in (
            ("evidence keys", self.evidence_key_ids, 30),
            ("view keys", self.view_key_ids, 90),
            ("model keys", self.model_key_ids, 360),
            ("shared keys", self.shared_artifact_key_ids, 480),
            ("model artifacts", self.model_artifact_ids, 360),
        ):
            if (
                len(values) != expected
                or tuple(values) != tuple(sorted(values))
                or len(set(values)) != expected
            ):
                raise ValueError(f"global {name} inventory drifted")
        if self.provenance_sha256 != provenance_identity_sha256(self.provenance):
            raise ValueError("readiness provenance hash drifted")
        return self


class PreparedScreeningChild:
    """In-process handle for one prepared child and its typed inventories."""

    def __init__(
        self,
        *,
        family_id: str,
        config: ExperimentConfig,
        run_dir: Path,
        provenance: SystemProvenance,
        data: MaterializedScreeningData,
        models: MaterializedScreeningModels,
        shared_plan: ExpectedSharedArtifacts,
    ) -> None:
        self.family_id = family_id
        self.config = config
        self.run_dir = run_dir
        self.provenance = provenance
        self.data = data
        self.models = models
        self.shared_plan = shared_plan


class PreparedScreeningReadiness:
    """Runtime handle; ``manifest`` is the durable canonical readout."""

    def __init__(
        self,
        children: tuple[PreparedScreeningChild, ...],
        manifest: ScreeningReadinessManifest | None,
    ) -> None:
        self.children = children
        self.manifest = manifest

    def __getattr__(self, name: str) -> Any:
        if self.manifest is not None:
            return getattr(self.manifest, name)
        raise AttributeError(name)


def build_screening_readiness_plan() -> Any:
    """Build and validate the canonical parent plan without executing anything."""

    plan = build_screening_plan()
    validate_screening_plan(plan)
    return plan


def _write_run_store_structure(store: RunStore, provenance: SystemProvenance) -> None:
    _safe_dir(store.run_dir, create=True)
    _safe_dir(store.units_dir, create=True)
    _safe_dir(store.attempts_dir, create=True)
    _exclusive_json(store.run_dir / "config.json", scientific_config_value(store.config))
    _exclusive_json(store.run_dir / "expected-units.json", store.expected.model_dump(mode="json"))
    _exclusive_json(store.run_dir / "provenance.json", provenance.model_dump(mode="json"))
    _exclusive_json(
        store.run_dir / "expected-shared-artifacts.json",
        store.expected_shared.model_dump(mode="json"),
    )


def _child_manifest(
    config: ExperimentConfig,
    data_keys: ScreeningDataKeys,
    data: MaterializedScreeningData,
    model_keys: ScreeningModelKeys,
    models: MaterializedScreeningModels,
    shared: ExpectedSharedArtifacts,
    provenance: SystemProvenance,
) -> ScreeningReadinessChild:
    data_manifest_body = {
        "evidence": tuple(
            data.manifests.evidence[replicate].model_dump(mode="json")
            for replicate in sorted(data.manifests.evidence)
        ),
        "views": tuple(
            data.manifests.views[identity].model_dump(mode="json")
            for identity in sorted(data.manifests.views)
        ),
    }
    model_manifest_body = tuple(
        models.manifests[identity].model_dump(mode="json") for identity in sorted(models.manifests)
    )
    shared_body = tuple(item.model_dump(mode="json") for item in shared.artifacts)
    reports = tuple(
        ReadinessComputeReport(
            base_condition_id=identity[0],
            training_tuple_id=identity[1],
            replicate=identity[2],
            model_key_id=model_keys.models[identity].key_id,
            model_artifact_id=models.manifests[identity].artifact_id,
            model_id=report.model_id,
            objective_id=report.objective_id,
            trainable_parameters=report.trainable_parameters,
            training_examples=report.training_examples,
            optimizer_steps=report.optimizer_steps,
            forward_passes=report.forward_passes,
            training_wall_seconds=report.training_wall_seconds,
        )
        for identity, report in sorted(models.compute.items())
    )
    return ScreeningReadinessChild(
        heldout_family_id=str(config.parameters["heldout_family_id"]),
        fold_id=str(config.parameters["fold_id"]),
        run_id=run_id_for(config),
        config_sha256=scientific_config_sha256(config),
        expected_units_sha256=expected_units_sha256(plan_expected_units(config)),
        provenance_sha256=provenance_identity_sha256(provenance),
        data_manifest_sha256=_digest(data_manifest_body),
        model_manifest_sha256=_digest(model_manifest_body),
        shared_plan_sha256=_digest(shared_body),
        evidence_key_ids=tuple(sorted(key.key_id for key in data_keys.evidence.values())),
        evidence_artifact_ids=tuple(
            sorted(item.evidence_id for item in data.manifests.evidence.values())
        ),
        view_key_ids=tuple(sorted(key.key_id for key in data_keys.views.values())),
        view_artifact_ids=tuple(sorted(item.artifact_id for item in data.manifests.views.values())),
        model_key_ids=tuple(sorted(key.key_id for key in model_keys.models.values())),
        model_artifact_ids=tuple(sorted(item.artifact_id for item in models.manifests.values())),
        shared_artifact_key_ids=tuple(sorted(item.key_id for item in shared.artifacts)),
        compute_reports=reports,
    )


def _build_manifest(
    plan: Any,
    configs: tuple[ExperimentConfig, ...],
    children: tuple[ScreeningReadinessChild, ...],
    provenance: SystemProvenance,
    snapshot: _AuthoritySourceSnapshot,
) -> ScreeningReadinessManifest:
    evidence_ids = tuple(sorted(item for child in children for item in child.evidence_key_ids))
    view_ids = tuple(sorted(item for child in children for item in child.view_key_ids))
    model_ids = tuple(sorted(item for child in children for item in child.model_key_ids))
    shared_ids = tuple(sorted(item for child in children for item in child.shared_artifact_key_ids))
    model_artifacts = tuple(sorted(item for child in children for item in child.model_artifact_ids))
    body = dict(
        schema_version=_SCHEMA_VERSION,
        screening_plan_id=plan.plan_id,
        protocol_sha256=snapshot.digests["protocol_sha256"],
        screening_candidates_sha256=snapshot.digests["screening_candidates_sha256"],
        task_manifest_sha256=snapshot.digests["task_manifest_sha256"],
        provenance_sha256=provenance_identity_sha256(provenance),
        provenance=provenance.model_dump(mode="json"),
        family_order=tuple(str(config.parameters["heldout_family_id"]) for config in configs),
        child_run_ids=tuple(child.run_id for child in children),
        children=tuple(child.model_dump(mode="json") for child in children),
        evidence_key_ids=evidence_ids,
        view_key_ids=view_ids,
        model_key_ids=model_ids,
        shared_artifact_key_ids=shared_ids,
        model_artifact_ids=model_artifacts,
        expected_total_units=9120,
        expected_total_evidence_artifacts=30,
        expected_total_training_data_views=90,
        expected_total_model_artifacts=360,
        expected_total_shared_artifacts=480,
        development_only=True,
        validation_executed=False,
        search_executed=False,
        outcomes_present=False,
        selection_performed=False,
        final_family_access=False,
    )
    return ScreeningReadinessManifest(
        manifest_sha256=_digest(body),
        **body,
    )


def prepare_screening_readiness(
    output_root: str | Path,
    *,
    repository: str | Path = ROOT,
    event: PreparationEvent | None = None,
    provenance: SystemProvenance | None = None,
) -> PreparedScreeningReadiness:
    """Materialize all six frozen development folds and publish one manifest."""

    snapshot = _authority_snapshot()
    plan = build_screening_readiness_plan()
    configs = screening_child_configs()
    _assert_source_snapshot(snapshot)
    if (
        plan.protocol_sha256 != snapshot.digests["protocol_sha256"]
        or plan.screening_candidates_sha256 != snapshot.digests["screening_candidates_sha256"]
        or plan.task_manifest_sha256 != snapshot.digests["task_manifest_sha256"]
    ):
        raise TrainingDataArtifactError(
            "screening plan was not built from the immutable authority snapshot"
        )
    if tuple(config.parameters["heldout_family_id"] for config in configs) != plan.family_order:
        raise TrainingDataArtifactError(
            "screening child configuration order differs from canonical plan"
        )
    plan_children = {child.heldout_family: child for child in plan.children}
    if len(plan_children) != 6:
        raise TrainingDataArtifactError("screening plan child authority is incomplete")
    for config in configs:
        family_id = str(config.parameters["heldout_family_id"])
        planned = plan_children.get(family_id)
        if (
            planned is None
            or planned.run_id != run_id_for(config)
            or planned.config_sha256 != scientific_config_sha256(config)
            or planned.expected_units_sha256 != expected_units_sha256(plan_expected_units(config))
            or config.parameters.get("development_protocol_sha256")
            != snapshot.digests["protocol_sha256"]
            or config.parameters.get("screening_candidates_sha256")
            != snapshot.digests["screening_candidates_sha256"]
            or config.parameters.get("development_task_manifest_sha256")
            != snapshot.digests["task_manifest_sha256"]
        ):
            raise TrainingDataArtifactError(
                "screening child differs from the immutable authority snapshot"
            )
    if any(config.split.final_tasks for config in configs):
        raise TrainingDataArtifactError("readiness preparation received final-family tasks")
    if any(config.device_policy != configs[0].device_policy for config in configs):
        raise TrainingDataArtifactError("screening child runtime policies are not identical")
    root = Path(output_root)
    if root.is_symlink():
        raise TrainingDataArtifactError("readiness output root cannot be a symlink")
    _safe_dir(root, create=True)
    apply_runtime_policy(configs[0].device_policy)
    captured_provenance = capture_system_provenance(repository, configs[0].device_policy)
    if provenance is not None and provenance_identity_sha256(
        captured_provenance
    ) != provenance_identity_sha256(provenance):
        raise TrainingDataArtifactError("supplied provenance differs from captured provenance")
    provenance = captured_provenance
    provenance_sha256 = provenance_identity_sha256(provenance)
    children: list[PreparedScreeningChild] = []
    canonical_children: list[ScreeningReadinessChild] = []
    parent_manifest_path = root / "phase2-screening-readiness.json"
    existing: ScreeningReadinessManifest | None = None
    if os.path.lexists(parent_manifest_path):
        try:
            existing = load_screening_readiness_manifest(parent_manifest_path)
        except RuntimeError as exc:
            raise TrainingDataArtifactError(
                "stored readiness manifest is invalid or unsafe"
            ) from exc
        if (
            existing.screening_plan_id != plan.plan_id
            or existing.protocol_sha256 != snapshot.digests["protocol_sha256"]
            or existing.screening_candidates_sha256
            != snapshot.digests["screening_candidates_sha256"]
            or existing.task_manifest_sha256 != snapshot.digests["task_manifest_sha256"]
            or existing.provenance_sha256 != provenance_sha256
        ):
            raise TrainingDataArtifactError(
                "stored readiness manifest differs from current authority or provenance"
            )
        # Preserve the first writer's timestamped provenance while the freshly
        # captured identity proves that the current runtime still matches it.
        provenance = existing.provenance
        provenance_sha256 = existing.provenance_sha256

    # Validate every child and adopt any first-writer provenance before a paid
    # probe, model training step, or new intent can run.
    child_dirs: list[Path] = []
    for config in configs:
        _assert_source_snapshot(snapshot)
        child_dir = _child_run_dir(root, config)
        _safe_dir(child_dir, create=True)
        _validate_child_top_level(child_dir)
        child_dirs.append(child_dir)

    persisted_provenance = tuple(
        item
        for config, child_dir in zip(configs, child_dirs, strict=True)
        if (
            item := _validate_existing_child_files(
                child_dir,
                config,
                current_provenance_sha256=provenance_sha256,
            )
        )
        is not None
    )
    if persisted_provenance:
        first_writer = persisted_provenance[0]
        if any(item != first_writer for item in persisted_provenance[1:]):
            raise TrainingDataArtifactError(
                "screening children contain different first-writer provenance"
            )
        if existing is not None and first_writer != existing.provenance:
            raise TrainingDataArtifactError(
                "child provenance differs from the parent readiness manifest"
            )
        provenance = first_writer
        provenance_sha256 = provenance_identity_sha256(provenance)

    # Claim every child before any materializer is entered, so a conflict in
    # the last fold cannot permit computation in an earlier fold.
    for config, child_dir in zip(configs, child_dirs, strict=True):
        intent = child_dir / "screening-readiness-intent.json"
        intent_body = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": run_id_for(config),
            "config_sha256": scientific_config_sha256(config),
            "provenance_sha256": provenance_sha256,
            "protocol_sha256": snapshot.digests["protocol_sha256"],
            "screening_candidates_sha256": snapshot.digests["screening_candidates_sha256"],
            "task_manifest_sha256": snapshot.digests["task_manifest_sha256"],
        }
        _claim_readiness_intent(intent, intent_body)

    for config, child_dir in zip(configs, child_dirs, strict=True):
        _assert_source_snapshot(snapshot)
        data_keys = build_screening_data_keys(config, provenance)
        _assert_source_snapshot(snapshot)
        data = materialize_screening_data(config, data_keys, child_dir, event=event)
        model_keys = build_screening_model_keys(config, data_keys, data.manifests)
        _assert_source_snapshot(snapshot)
        models = materialize_screening_models(
            config, data_keys, data, model_keys, child_dir, event=event
        )
        shared = build_screening_shared_plan(config, data_keys, data.manifests, model_keys)
        _assert_source_snapshot(snapshot)
        if len(data.manifests.evidence) != 5 or len(data.manifests.views) != 15:
            raise TrainingDataArtifactError("screening data readiness inventory is not exact")
        if len(models.manifests) != 60 or len(shared.artifacts) != 80:
            raise TrainingDataArtifactError(
                "screening model/shared readiness inventory is not exact"
            )
        store = RunStore(
            root,
            config,
            repository=repository,
            shared_artifacts=tuple(shared.artifacts),
        )
        _write_run_store_structure(store, provenance)
        if store.run_dir != child_dir or store.load_provenance() != provenance:
            raise TrainingDataArtifactError("RunStore readiness provenance or path drifted")
        children.append(
            PreparedScreeningChild(
                family_id=str(config.parameters["heldout_family_id"]),
                config=config,
                run_dir=child_dir,
                provenance=provenance,
                data=data,
                models=models,
                shared_plan=store.expected_shared,
            )
        )
        canonical_children.append(
            _child_manifest(config, data_keys, data, model_keys, models, shared, provenance)
        )
        _assert_source_snapshot(snapshot)
    if len(children) != 6:
        raise TrainingDataArtifactError("screening readiness child inventory is incomplete")
    _assert_source_snapshot(snapshot)
    manifest: ScreeningReadinessManifest | None = None
    if len(canonical_children) == 6:
        manifest = _build_manifest(plan, configs, tuple(canonical_children), provenance, snapshot)
        if provenance_sha256 != manifest.provenance_sha256:
            raise TrainingDataArtifactError("readiness provenance changed during preparation")
        _assert_source_snapshot(snapshot)
        for child_dir in child_dirs:
            _validate_child_top_level(child_dir)
        _exclusive_json(parent_manifest_path, manifest.model_dump(mode="json"))
        manifest = load_screening_readiness_manifest(parent_manifest_path)
    result = PreparedScreeningReadiness(tuple(children), manifest)
    return result


def load_screening_readiness_manifest(path: str | Path) -> ScreeningReadinessManifest:
    """Load a readiness manifest only when its bytes and self-hash are exact."""

    target = Path(path)
    for candidate in (target.absolute(), *target.absolute().parents):
        if os.path.lexists(candidate) and candidate.is_symlink():
            raise RuntimeError("readiness manifest path contains a symlink")
    if target.is_symlink() or not target.is_file():
        raise RuntimeError("readiness manifest must be a regular file")
    payload_bytes = target.read_bytes()
    try:
        payload = json.loads(payload_bytes)
        manifest = ScreeningReadinessManifest.model_validate(payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError("invalid screening readiness manifest") from exc
    expected_bytes = canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
    if payload_bytes != expected_bytes:
        raise RuntimeError("screening readiness manifest is not canonical byte content")
    return manifest


materialize_phase2_screening_readiness = prepare_screening_readiness
build_phase2_screening_readiness_plan = build_screening_readiness_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--prepare", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--repository", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.plan_only:
        print(json.dumps(build_screening_readiness_plan().model_dump(mode="json"), indent=2))
        return 0
    if args.output_root is None:
        parser.error("--prepare requires --output-root")
    manifest = prepare_screening_readiness(args.output_root, repository=args.repository)
    print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
