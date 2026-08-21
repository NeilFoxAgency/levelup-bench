"""Atomic experiment storage and deterministic expected-unit planning."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from levelup.experiments.runner.config import (
    ExperimentConfig,
    canonical_json_bytes,
    run_id_for,
    scientific_config_sha256,
    scientific_config_value,
    scientific_exposure_value,
)
from levelup.experiments.runner.provenance import apply_runtime_policy, capture_system_provenance
from levelup.experiments.runner.records import (
    AggregateArtifact,
    AttemptRecord,
    ExpectedSharedArtifacts,
    ExpectedUnits,
    PlannedSharedArtifact,
    PlannedUnit,
    ResourceAccounting,
    SharedArtifactReference,
    SystemProvenance,
    TrainingArtifactCostRecord,
    UnitKey,
    UnitRecord,
    UnitSeeds,
    unit_id_for,
)


def _shared_refs(record: UnitRecord) -> tuple[SharedArtifactReference, ...]:
    return (
        (record.shared_artifact,) if record.shared_artifact is not None else ()
    ) + record.shared_artifacts


ModelT = TypeVar("ModelT", bound=BaseModel)


class ArtifactValidationError(RuntimeError):
    """Raised when a stored result is partial, corrupt, unexpected, or mismatched."""


class ConflictingResultError(RuntimeError):
    """Raised when execution tries to replace a different completed atomic result."""


def _atomic_write_json(path: Path, value: Any) -> None:
    """Publish validated canonical JSON through a same-directory atomic replace."""

    if path.is_symlink():
        raise ArtifactValidationError(f"refusing to replace symlink: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = canonical_json_bytes(value) + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    if path.is_symlink():
        raise ArtifactValidationError(f"refusing to read symlink: {path.name}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return model_type.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"invalid artifact {path.name}: {type(exc).__name__}"
        ) from None


def _revalidate_instance(instance: ModelT, model_type: type[ModelT]) -> ModelT:
    try:
        return model_type.model_validate(instance.model_dump(mode="json", warnings=False))
    except (ValidationError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"invalid {model_type.__name__} instance: {type(exc).__name__}"
        ) from None


def _tasks_by_phase(config: ExperimentConfig) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    return (
        ("development", config.split.development_tasks),
        ("validation", config.split.validation_tasks),
        ("final", config.split.final_tasks),
    )


def _provenance_identity(provenance: SystemProvenance) -> dict[str, Any]:
    return provenance.model_dump(mode="json", exclude={"captured_at_utc"})


def provenance_identity_sha256(provenance: SystemProvenance) -> str:
    """Hash stable execution provenance while excluding its capture timestamp."""

    return hashlib.sha256(canonical_json_bytes(_provenance_identity(provenance))).hexdigest()


def expected_units_sha256(expected: ExpectedUnits) -> str:
    """Hash the immutable unit plan used by shared-artifact keys."""

    return hashlib.sha256(canonical_json_bytes(expected.model_dump(mode="json"))).hexdigest()


def _validate_provenance_policy(
    provenance: SystemProvenance,
    config: ExperimentConfig,
    resolved_device: str,
) -> None:
    policy = config.device_policy
    if (
        provenance.requested_device != policy.requested_device
        or provenance.resolved_device != resolved_device
        or provenance.requested_torch_threads != policy.torch_threads
        or provenance.actual_torch_threads != policy.torch_threads
        or provenance.requested_torch_interop_threads != policy.torch_interop_threads
        or provenance.actual_torch_interop_threads != policy.torch_interop_threads
        or provenance.deterministic_algorithms_requested != policy.deterministic_algorithms
        or provenance.deterministic_algorithms_actual != policy.deterministic_algorithms
        or provenance.processes != policy.processes
    ):
        raise ArtifactValidationError(
            "execution provenance does not match the configured runtime policy"
        )


def plan_expected_units(config: ExperimentConfig) -> ExpectedUnits:
    """Freeze the complete task/condition/replicate matrix and resolved seeds."""

    config_hash = scientific_config_sha256(config)
    run_id = run_id_for(config)
    policy = config.seed_policy
    model_replicate_step = (
        policy.replicate_stride if policy.derivation_version == "phase2.v1" else 1
    )
    data_replicate_step = policy.replicate_stride if policy.derivation_version == "phase2.v1" else 1
    units: list[PlannedUnit] = []
    conditions = sorted(config.conditions, key=lambda condition: condition.condition_id)
    for phase, tasks in _tasks_by_phase(config):
        for task in sorted(tasks, key=lambda item: item.task_id):
            for replicate in range(config.replicates):
                seeds = UnitSeeds(
                    model_seed=policy.model_seed_base + replicate * model_replicate_step,
                    environment_seed=(task.environment_reset_seed + policy.environment_seed_offset),
                    probe_seed=(
                        policy.probe_seed_base
                        + replicate * policy.replicate_stride
                        + task.task_index
                    ),
                    search_seed=(
                        policy.search_seed_base
                        + replicate * policy.replicate_stride
                        + task.task_index
                    ),
                    data_order_seed=policy.data_order_seed_base + replicate * data_replicate_step,
                )
                for condition in conditions:
                    if phase not in condition.execution_phases:
                        continue
                    key = UnitKey(
                        phase=phase,
                        condition_id=condition.condition_id,
                        family_id=task.family_id,
                        task_id=task.task_id,
                        task_index=task.task_index,
                        replicate=replicate,
                    )
                    exposure_hash = hashlib.sha256(
                        canonical_json_bytes(scientific_exposure_value(condition.exposure))
                    ).hexdigest()
                    units.append(
                        PlannedUnit(
                            unit_id=unit_id_for(key),
                            key=key,
                            seeds=seeds,
                            exposure_manifest_sha256=exposure_hash,
                        )
                    )
    return ExpectedUnits(
        run_id=run_id,
        config_sha256=config_hash,
        units=tuple(sorted(units, key=lambda unit: unit.unit_id)),
    )


class RunStore:
    """Validated file store for one deterministic experiment run."""

    def __init__(
        self,
        output_root: str | Path,
        config: ExperimentConfig,
        *,
        repository: str | Path,
        shared_artifacts: tuple[PlannedSharedArtifact, ...] = (),
    ) -> None:
        self.config = config
        self.config_sha256 = scientific_config_sha256(config)
        self.run_id = run_id_for(config)
        self.repository = Path(repository)
        self.run_dir = Path(output_root) / self.run_id
        self.units_dir = self.run_dir / "units"
        self.attempts_dir = self.run_dir / "attempts"
        self.aggregate_path = self.run_dir / "aggregate.json"
        self.expected = plan_expected_units(config)
        self.expected_shared = ExpectedSharedArtifacts(
            run_id=self.run_id,
            config_sha256=self.config_sha256,
            artifacts=shared_artifacts,
        )
        self._expected_by_id = {unit.unit_id: unit for unit in self.expected.units}
        self._validate_shared_plan()
        self._execution_ready = False

    def _validate_shared_plan(self) -> None:
        units = {unit.unit_id: unit for unit in self.expected.units}
        conditions = {condition.condition_id for condition in self.config.conditions}
        configured_fold = self.config.parameters.get("fold_id")
        configured_family = self.config.parameters.get("heldout_family_id")
        for artifact in self.expected_shared.artifacts:
            if (
                configured_fold != artifact.owner_fold_id
                or configured_family != artifact.owner_family_id
            ):
                raise ArtifactValidationError(
                    "shared artifact fold owner does not match scientific config"
                )
            if artifact.owner_condition_id not in conditions:
                raise ArtifactValidationError("shared artifact owner condition is unknown")
            consumers = [units.get(unit_id) for unit_id in artifact.consumer_unit_ids]
            if any(unit is None for unit in consumers):
                raise ArtifactValidationError("shared artifact references an unknown consumer unit")
            if not any(
                unit is not None and unit.key.condition_id == artifact.owner_condition_id
                for unit in consumers
            ):
                raise ArtifactValidationError(
                    "shared artifact owner condition is not a declared consumer"
                )
            observed_phases = {unit.key.phase for unit in consumers if unit is not None}
            observed_conditions = {unit.key.condition_id for unit in consumers if unit is not None}
            if observed_phases != {artifact.consumer_phase}:
                raise ArtifactValidationError(
                    "shared artifact consumer phase does not match its frozen plan"
                )
            if observed_conditions != set(artifact.consumer_condition_ids):
                raise ArtifactValidationError(
                    "shared artifact consumer conditions do not match their frozen plan"
                )
            if artifact.owner_condition_id not in artifact.consumer_condition_ids:
                raise ArtifactValidationError("shared artifact owner condition is not authorized")
            if any(
                unit is not None
                and (
                    unit.key.replicate != artifact.owner_replicate
                    or unit.key.family_id != artifact.owner_family_id
                )
                for unit in consumers
            ):
                raise ArtifactValidationError("shared artifact consumer owner mismatch")

    def initialize(
        self,
        *,
        for_execution: bool = True,
    ) -> None:
        """Create immutable config/unit plans and first-run provenance."""

        resolved_device = apply_runtime_policy(self.config.device_policy) if for_execution else None

        self.units_dir.mkdir(parents=True, exist_ok=True)
        self.attempts_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.run_dir / "config.json"
        expected_path = self.run_dir / "expected-units.json"
        shared_path = self.run_dir / "expected-shared-artifacts.json"
        provenance_path = self.run_dir / "provenance.json"

        if config_path.exists():
            stored_config = _load_model(config_path, ExperimentConfig)
            if scientific_config_sha256(stored_config) != self.config_sha256:
                raise ArtifactValidationError("stored config does not match requested run")
        else:
            _atomic_write_json(config_path, scientific_config_value(self.config))

        if expected_path.exists():
            stored_expected = _load_model(expected_path, ExpectedUnits)
            if stored_expected != self.expected:
                raise ArtifactValidationError("stored expected-unit plan does not match config")
        else:
            _atomic_write_json(expected_path, self.expected.model_dump(mode="json"))

        if shared_path.exists():
            stored_shared = _load_model(shared_path, ExpectedSharedArtifacts)
            if stored_shared != self.expected_shared:
                raise ArtifactValidationError("stored shared-artifact plan does not match config")
        else:
            _atomic_write_json(shared_path, self.expected_shared.model_dump(mode="json"))

        if provenance_path.exists():
            stored_provenance = _load_model(provenance_path, SystemProvenance)
            if for_execution:
                current_provenance = capture_system_provenance(
                    self.repository,
                    self.config.device_policy,
                )
                if resolved_device is None:
                    raise RuntimeError("execution device was not resolved")
                _validate_provenance_policy(
                    current_provenance,
                    self.config,
                    resolved_device,
                )
                if _provenance_identity(stored_provenance) != _provenance_identity(
                    current_provenance
                ):
                    raise ArtifactValidationError(
                        "stored provenance does not match the current execution environment"
                    )
        else:
            captured = capture_system_provenance(
                self.repository,
                self.config.device_policy,
            )
            if for_execution:
                if resolved_device is None:
                    raise RuntimeError("execution device was not resolved")
                _validate_provenance_policy(captured, self.config, resolved_device)
            _atomic_write_json(provenance_path, captured.model_dump(mode="json"))
        self._execution_ready = for_execution

    def planned_unit(self, unit_id: str) -> PlannedUnit:
        try:
            return self._expected_by_id[unit_id]
        except KeyError as exc:
            raise ArtifactValidationError(f"unexpected unit_id: {unit_id}") from exc

    def load_provenance(self) -> SystemProvenance:
        return _load_model(self.run_dir / "provenance.json", SystemProvenance)

    def planned_shared(self, key_id: str, kind: str = "training_artifact") -> PlannedSharedArtifact:
        matches = [
            item
            for item in self.expected_shared.artifacts
            if item.key_id == key_id and item.kind == kind
        ]
        if len(matches) != 1:
            raise ArtifactValidationError("unknown shared-artifact key")
        return matches[0]

    def load_shared_cost(self, key_id: str, kind: str = "training_artifact") -> Any:
        self.planned_shared(key_id, kind)
        if kind == "training_data_evidence":
            from levelup.experiments.runner.training_data_artifacts import (
                load_training_data_evidence_cost,
            )

            try:
                return load_training_data_evidence_cost(self.run_dir, key_id)
            except (OSError, RuntimeError, ValueError) as exc:
                raise ArtifactValidationError("invalid shared training-data evidence cost") from exc
        if kind == "training_data_view":
            from levelup.experiments.runner.training_data_artifacts import (
                load_training_data_view_cost,
            )

            try:
                return load_training_data_view_cost(self.run_dir, key_id)
            except (OSError, RuntimeError, ValueError) as exc:
                raise ArtifactValidationError("invalid shared training-data view cost") from exc
        if kind != "training_artifact":
            raise ArtifactValidationError("unknown shared-artifact kind")
        costs_root = self.run_dir / "training-artifact-costs"
        path = costs_root / f"{key_id}.json"
        if self.run_dir.is_symlink() or costs_root.is_symlink() or path.is_symlink():
            raise ArtifactValidationError("refusing symlink shared-artifact cost path")
        try:
            path.resolve().relative_to(costs_root.resolve())
        except ValueError:
            raise ArtifactValidationError("shared-artifact cost path escapes its root") from None
        record = _load_model(path, TrainingArtifactCostRecord)
        if record.key_id != key_id or record.expected_cost_id != record.cost_id:
            raise ArtifactValidationError("shared-artifact cost identity mismatch")
        return record

    def validate_shared_reference(
        self, unit: PlannedUnit, reference: SharedArtifactReference
    ) -> None:
        from levelup.experiments.runner.training_artifacts import (
            TrainingArtifactKey,
            load_training_cost,
            load_training_key_index,
            load_training_manifest,
        )

        planned = self.planned_shared(reference.key_id, reference.kind)
        if unit.unit_id not in planned.consumer_unit_ids:
            raise ArtifactValidationError("unit is not a declared shared-artifact consumer")
        if (
            unit.key.replicate != planned.owner_replicate
            or unit.key.family_id != planned.owner_family_id
        ):
            raise ArtifactValidationError("shared-artifact owner/replicate mismatch")
        expected_plan_sha256 = expected_units_sha256(self.expected)
        expected_provenance_sha256 = provenance_identity_sha256(self.load_provenance())
        cost = self.load_shared_cost(reference.key_id, reference.kind)
        if cost.artifact_id != reference.artifact_id or cost.cost_id != reference.cost_id:
            raise ArtifactValidationError("shared-artifact reference does not match cost record")
        expected_group = planned.owner_group_id or planned.owner_condition_id
        if reference.kind == "training_data_evidence":
            try:
                from levelup.experiments.runner.training_data_artifacts import (
                    TrainingDataEvidenceKey,
                    load_training_data_evidence,
                    load_training_data_evidence_cost,
                )

                evidence_key = TrainingDataEvidenceKey.model_validate(cost.key)
                if (
                    cost.scope != "training_data_evidence_preparation"
                    or evidence_key.key_id != reference.key_id
                    or evidence_key.fold_id != planned.owner_fold_id
                    or evidence_key.replicate != planned.owner_replicate
                    or evidence_key.heldout_family_id != planned.owner_family_id
                    or evidence_key.expected_unit_plan_sha256 != expected_plan_sha256
                    or evidence_key.provenance_sha256 != expected_provenance_sha256
                    or evidence_key.reference_exposure_sha256
                    != unit.exposure_manifest_sha256
                ):
                    raise ArtifactValidationError(
                        "shared evidence key does not match its frozen owner"
                    )
                validated_cost = load_training_data_evidence_cost(self.run_dir, evidence_key)
                manifest, _ = load_training_data_evidence(
                    self.run_dir, reference.artifact_id, expected_key=evidence_key
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise ArtifactValidationError("shared training-data evidence is invalid") from exc
            if validated_cost != cost or manifest.evidence_id != reference.artifact_id:
                raise ArtifactValidationError(
                    "shared training-data evidence files do not match the unit reference"
                )
            return
        if reference.kind == "training_data_view":
            try:
                from levelup.experiments.runner.training_data_artifacts import (
                    TrainingDataArtifactKey,
                    load_training_data_artifact,
                    load_training_data_view_cost,
                )

                data_key = TrainingDataArtifactKey.model_validate(cost.key)
                if (
                    cost.scope != "training_data_view_preparation"
                    or data_key.key_id != reference.key_id
                    or data_key.condition_id != expected_group
                    or data_key.fold_id != planned.owner_fold_id
                    or data_key.replicate != planned.owner_replicate
                    or data_key.heldout_family_id != planned.owner_family_id
                    or data_key.expected_unit_plan_sha256 != expected_plan_sha256
                    or data_key.provenance_sha256 != expected_provenance_sha256
                    or data_key.reference_exposure_sha256
                    != unit.exposure_manifest_sha256
                ):
                    raise ArtifactValidationError(
                        "shared training-data view key does not match its frozen owner"
                    )
                validated_cost = load_training_data_view_cost(self.run_dir, data_key)
                manifest, _ = load_training_data_artifact(
                    self.run_dir, reference.artifact_id, expected_key=data_key
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise ArtifactValidationError("shared training-data view is invalid") from exc
            if validated_cost != cost or manifest.artifact_id != reference.artifact_id:
                raise ArtifactValidationError(
                    "shared training-data view files do not match the unit reference"
                )
            return
        if cost.scope != "training_preparation":
            raise ArtifactValidationError("shared model cost has the wrong scope")
        try:
            training_key = TrainingArtifactKey.model_validate(cost.key)
        except (TypeError, ValueError):
            raise ArtifactValidationError(
                "shared-artifact cost does not contain a valid training key"
            ) from None
        if (
            training_key.key_id != reference.key_id
            or training_key.condition_id != expected_group
            or training_key.fold_id != planned.owner_fold_id
            or training_key.replicate != planned.owner_replicate
            or training_key.heldout_family_id != planned.owner_family_id
            or training_key.expected_unit_plan_sha256 != expected_plan_sha256
            or training_key.provenance_sha256 != expected_provenance_sha256
            or training_key.exposure_sha256 != unit.exposure_manifest_sha256
        ):
            raise ArtifactValidationError(
                "shared-artifact training key does not match its frozen owner"
            )
        validated_cost = load_training_cost(self.run_dir, training_key)
        index = load_training_key_index(self.run_dir, training_key)
        manifest = load_training_manifest(self.run_dir, index.artifact_id)
        if (
            validated_cost != cost
            or index.artifact_id != reference.artifact_id
            or manifest.artifact_id != reference.artifact_id
        ):
            raise ArtifactValidationError("shared-artifact files do not match the unit reference")

    def validate_shared_reference_set(
        self,
        unit: PlannedUnit,
        references: tuple[SharedArtifactReference, ...],
    ) -> None:
        """Validate cross-kind provenance after validating each typed reference."""

        for reference in references:
            self.validate_shared_reference(unit, reference)
        by_kind = {reference.kind: reference for reference in references}
        evidence = by_kind.get("training_data_evidence")
        view = by_kind.get("training_data_view")
        model = by_kind.get("training_artifact")
        if evidence is not None and view is not None:
            try:
                from levelup.experiments.runner.training_data_artifacts import (
                    evidence_key_for,
                    load_training_data_artifact,
                    load_training_data_evidence_cost,
                    load_training_data_view_cost,
                )

                evidence_cost = load_training_data_evidence_cost(
                    self.run_dir, evidence.key_id
                )
                view_cost = load_training_data_view_cost(self.run_dir, view.key_id)
                view_manifest, _ = load_training_data_artifact(
                    self.run_dir, view.artifact_id, expected_key=view_cost.key
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise ArtifactValidationError(
                    "shared training-data lineage is invalid"
                ) from exc
            if (
                evidence_key_for(view_cost.key) != evidence_cost.key
                or view_manifest.evidence_id != evidence.artifact_id
            ):
                raise ArtifactValidationError(
                    "shared training-data view does not derive from the referenced evidence"
                )
        if view is not None and model is not None:
            from levelup.experiments.runner.training_artifacts import TrainingArtifactKey

            model_cost = self.load_shared_cost(model.key_id, model.kind)
            try:
                model_key = TrainingArtifactKey.model_validate(model_cost.key)
            except (AttributeError, TypeError, ValueError):
                raise ArtifactValidationError(
                    "shared model cost does not contain a valid training key"
                ) from None
            if model_key.training_data_sha256 != view.artifact_id:
                raise ArtifactValidationError(
                    "shared model is not bound to the referenced training-data view"
                )

    def _unit_path(self, unit_id: str) -> Path:
        self.planned_unit(unit_id)
        return self.units_dir / f"{unit_id}.json"

    def load_completed(self, unit_id: str) -> UnitRecord | None:
        path = self._unit_path(unit_id)
        if not path.exists():
            return None
        record = _load_model(path, UnitRecord)
        expected = self.planned_unit(unit_id)
        if (
            record.run_id != self.run_id
            or record.config_sha256 != self.config_sha256
            or record.unit_id != unit_id
            or record.key != expected.key
            or record.seeds != expected.seeds
            or record.exposure_manifest_sha256 != expected.exposure_manifest_sha256
        ):
            raise ArtifactValidationError(f"completed unit identity mismatch: {unit_id}")
        self._validate_outcome_metric(record)
        refs = _shared_refs(record)
        if refs:
            # Other phases are deliberately task-local held-out costs. Shared
            # evidence records distinguish training_probes/reference_replay,
            # while a learned consumer must never repeat optimizer training.
            if record.accounting.training != ResourceAccounting().training:
                raise ArtifactValidationError(
                    "shared-artifact consumer cannot duplicate task-local training accounting"
                )
            self.validate_shared_reference_set(expected, refs)
        return record

    def _validate_outcome_metric(self, record: UnitRecord) -> None:
        metrics = {metric.metric_id: metric.direction for metric in self.config.metrics}
        metric_id = record.outcome.performance_metric_id
        if metric_id not in metrics:
            raise ArtifactValidationError(
                f"completed unit uses undeclared performance metric: {metric_id}"
            )
        if metrics[metric_id] != record.outcome.performance_direction:
            raise ArtifactValidationError(f"completed unit metric direction mismatch: {metric_id}")
        unknown_diagnostics = set(record.diagnostics) - set(self.config.diagnostic_fields)
        if unknown_diagnostics:
            raise ArtifactValidationError("completed unit uses undeclared diagnostic fields")

    def write_completed(self, record: UnitRecord) -> bool:
        """Write once; identical repeats are idempotent and conflicts are rejected."""

        record = _revalidate_instance(record, UnitRecord)
        path = self._unit_path(record.unit_id)
        expected = self.planned_unit(record.unit_id)
        if (
            record.run_id != self.run_id
            or record.config_sha256 != self.config_sha256
            or record.key != expected.key
            or record.seeds != expected.seeds
            or record.exposure_manifest_sha256 != expected.exposure_manifest_sha256
        ):
            raise ArtifactValidationError("completed record does not match expected unit")
        self._validate_outcome_metric(record)
        refs = _shared_refs(record)
        if refs:
            # Held-out probes, replay, search, setup, and serialization remain
            # visible here; only optimizer training belongs exclusively to the
            # shared preparation record.
            if record.accounting.training != ResourceAccounting().training:
                raise ArtifactValidationError(
                    "shared-artifact consumer cannot duplicate task-local training accounting"
                )
            self.validate_shared_reference_set(expected, refs)
        existing = self.load_completed(record.unit_id)
        if existing is not None:
            if existing == record:
                return False
            raise ConflictingResultError(f"completed unit already exists: {record.unit_id}")
        _atomic_write_json(path, record.model_dump(mode="json"))
        self.load_completed(record.unit_id)
        return True

    def completed_records(self) -> tuple[UnitRecord, ...]:
        expected_paths = {f"{unit.unit_id}.json" for unit in self.expected.units}
        observed_paths = {path.name for path in self.units_dir.glob("*.json")}
        unexpected = observed_paths - expected_paths
        if unexpected:
            raise ArtifactValidationError(f"unexpected completed unit files: {sorted(unexpected)}")
        records = [
            record
            for unit in self.expected.units
            if (record := self.load_completed(unit.unit_id)) is not None
        ]
        return tuple(records)

    def missing_units(self) -> tuple[PlannedUnit, ...]:
        return tuple(
            unit for unit in self.expected.units if self.load_completed(unit.unit_id) is None
        )

    def next_attempt_number(self, unit_id: str) -> int:
        self.planned_unit(unit_id)
        prefix = f"{unit_id}.attempt-"
        numbers: list[int] = []
        for path in self.attempts_dir.glob(f"{prefix}*.json"):
            suffix = path.stem.removeprefix(prefix)
            if suffix.isdigit():
                numbers.append(int(suffix))
        return max(numbers, default=0) + 1

    def write_attempt(self, record: AttemptRecord) -> None:
        record = _revalidate_instance(record, AttemptRecord)
        expected = self.planned_unit(record.unit_id)
        if (
            record.run_id != self.run_id
            or record.config_sha256 != self.config_sha256
            or record.key != expected.key
            or record.seeds != expected.seeds
        ):
            raise ArtifactValidationError("attempt record does not match expected unit")
        path = self.attempts_dir / (f"{record.unit_id}.attempt-{record.attempt:04d}.json")
        if path.exists():
            raise ConflictingResultError(f"attempt already exists: {path.name}")
        _atomic_write_json(path, record.model_dump(mode="json"))
        _load_model(path, AttemptRecord)

    def attempt_records(self) -> tuple[AttemptRecord, ...]:
        records: list[AttemptRecord] = []
        for path in sorted(self.attempts_dir.glob("*.json")):
            record = _load_model(path, AttemptRecord)
            expected = self.planned_unit(record.unit_id)
            expected_name = f"{record.unit_id}.attempt-{record.attempt:04d}.json"
            if (
                path.name != expected_name
                or record.run_id != self.run_id
                or record.config_sha256 != self.config_sha256
                or record.key != expected.key
                or record.seeds != expected.seeds
            ):
                raise ArtifactValidationError(f"attempt identity mismatch: {path.name}")
            records.append(record)
        return tuple(records)

    def write_aggregate(self, aggregate: AggregateArtifact) -> bool:
        from levelup.experiments.runner.aggregate import aggregate_run

        aggregate = _revalidate_instance(aggregate, AggregateArtifact)
        expected = aggregate_run(self, strict=aggregate.complete, write=False)
        if aggregate != expected:
            raise ArtifactValidationError("aggregate does not match validated raw records")
        if self.aggregate_path.exists():
            stored = _load_model(self.aggregate_path, AggregateArtifact)
            if stored == aggregate:
                return False
            monotonic_completion = (
                stored.run_id == aggregate.run_id
                and stored.config_sha256 == aggregate.config_sha256
                and stored.expected_units_sha256 == aggregate.expected_units_sha256
                and not stored.complete
                and aggregate.inventory.completed > stored.inventory.completed
                and aggregate.inventory.expected == stored.inventory.expected
            )
            if not monotonic_completion:
                raise ConflictingResultError("aggregate already exists with different content")
        _atomic_write_json(self.aggregate_path, aggregate.model_dump(mode="json"))
        return True
