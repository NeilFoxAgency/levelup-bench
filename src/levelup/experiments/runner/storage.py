"""Atomic experiment storage and deterministic expected-unit planning."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator, TypeVar

from pydantic import BaseModel, ValidationError

from levelup.experiments.runner import secure_fs
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


def _require_directory_fd_support() -> None:
    """Fail closed outside Unix-style directory-fd storage semantics."""

    required_dir_fd = (os.open, os.link, os.stat, os.unlink)
    if (
        os.name != "posix"
        or any(function not in os.supports_dir_fd for function in required_dir_fd)
        or os.scandir not in os.supports_fd
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
    ):
        raise ArtifactValidationError(
            "result storage requires Unix directory-fd and O_NOFOLLOW support"
        )


def _directory_flags() -> int:
    _require_directory_fd_support()
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_directory_chain(path: Path) -> int:
    """Open every absolute path component without following symlinks."""
    try:
        absolute = Path(os.path.abspath(path))
        for candidate in (absolute, *absolute.parents):
            if candidate.is_symlink():
                raise ArtifactValidationError(
                    f"cannot securely open symlink directory path: {candidate}"
                )
        return secure_fs.open_directory_chain(path)
    except ArtifactValidationError:
        raise
    except secure_fs.SecureFilesystemError as exc:
        raise ArtifactValidationError(f"cannot securely open directory: {path}") from exc


def _directory_identity(directory_fd: int) -> tuple[int, int]:
    try:
        return secure_fs.directory_identity(directory_fd)
    except secure_fs.SecureFilesystemError as exc:
        raise ArtifactValidationError("result namespace descriptor is not a directory") from exc


def _open_child_directory(parent_fd: int, name: str) -> int:
    try:
        return secure_fs.open_child_directory(parent_fd, name)
    except secure_fs.SecureFilesystemError as exc:
        raise ArtifactValidationError(
            f"cannot securely open result namespace or symlink: {name}"
        ) from exc


def _load_model_at(directory_fd: int, name: str, model_type: type[ModelT]) -> ModelT:
    """Read a final artifact entry without following it or re-resolving a path."""

    try:
        raw = secure_fs.read_json_at(directory_fd, name)
        return model_type.model_validate(raw)
    except ArtifactValidationError:
        raise
    except secure_fs.SecureFilesystemError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            raise ArtifactValidationError(f"missing prepared {name}") from None
        raise ArtifactValidationError(
            f"invalid artifact {name}: {type(exc).__name__}"
        ) from None
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"invalid artifact {name}: {type(exc).__name__}"
        ) from None


def _exclusive_write_json_at(
    run_directory_fd: int,
    namespace_directory_fd: int,
    name: str,
    value: Any,
) -> bool:
    """Exclusively publish canonical JSON through already-pinned directories.

    The private temporary entry lives in the run directory, not the strict
    result namespace.  Thus even an exceptional post-publication cleanup
    cannot poison unit/attempt enumeration.  Once the destination link exists,
    cleanup/fsync failures are best-effort and do not misreport the durable
    result as an execution failure.
    """

    rendered = canonical_json_bytes(value) + b"\n"
    temporary_name: str | None = None
    try:
        for _ in range(32):
            candidate = f".publication-{name}.{uuid.uuid4().hex}.tmp"
            try:
                temporary_fd = os.open(
                    candidate,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=run_directory_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        else:
            raise OSError("unable to allocate exclusive artifact temporary")
        try:
            with os.fdopen(temporary_fd, "wb") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            # The outer cleanup removes the private temporary entry.
            raise
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=run_directory_fd,
                dst_dir_fd=namespace_directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            return False
        try:
            os.fsync(namespace_directory_fd)
        except OSError:
            # The destination entry is already complete and visible.  Do not
            # turn a post-publication durability warning into a retry that
            # would be recorded as a wholly failed scientific unit.
            pass
        return True
    finally:
        if temporary_name is not None:
            for _ in range(3):
                try:
                    os.unlink(temporary_name, dir_fd=run_directory_fd)
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    continue
        # Persist cleanup when possible.  A successfully linked and fsynced
        # result remains success even if this best-effort directory sync fails.
        try:
            os.fsync(run_directory_fd)
        except OSError:
            pass


def _strict_namespace_entries(
    namespace_directory_fd: int,
    namespace: str,
    expected_units: set[str],
) -> tuple[str, ...]:
    try:
        entries = list(secure_fs.strict_regular_entries(namespace_directory_fd))
        for name in entries:
            if namespace == "units":
                if not name.endswith(".json") or name[:-5] not in expected_units:
                    raise ArtifactValidationError(f"unexpected completed unit files: {name}")
            else:
                stem = name.removesuffix(".json")
                unit_id, separator, number = stem.rpartition(".attempt-")
                if (
                    not name.endswith(".json")
                    or not separator
                    or unit_id not in expected_units
                    or len(number) != 4
                    or not number.isdigit()
                    or int(number) < 1
                ):
                    raise ArtifactValidationError(f"unexpected attempt file: {name}")
    except ArtifactValidationError:
        raise
    except secure_fs.SecureFilesystemError as exc:
        raise ArtifactValidationError(
            f"cannot enumerate {namespace} namespace"
        ) from exc
    return tuple(sorted(entries))


def _load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    if path.is_symlink():
        raise ArtifactValidationError(f"refusing to read symlink: {path.name}")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                fd = -1
                raw = json.load(handle)
        finally:
            if fd != -1:
                os.close(fd)
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
        self._result_directory_identities: dict[str, tuple[int, int]] | None = None

    def _capture_result_directory_identities(self) -> dict[str, tuple[int, int]]:
        """Pin the run directory and its two direct result namespaces."""

        run_fd = _open_directory_chain(self.run_dir)
        identities = {"run": _directory_identity(run_fd)}
        try:
            for namespace in ("units", "attempts"):
                namespace_fd = _open_child_directory(run_fd, namespace)
                try:
                    identities[namespace] = _directory_identity(namespace_fd)
                finally:
                    os.close(namespace_fd)
        finally:
            os.close(run_fd)
        return identities

    @contextmanager
    def _open_result_namespace(self, namespace: str) -> Iterator[tuple[int, int]]:
        """Yield pinned run/namespace fds after comparing their inode identities."""

        if namespace not in {"units", "attempts"}:
            raise ArtifactValidationError(f"unknown result namespace: {namespace}")
        expected = self._result_directory_identities
        if expected is None:
            raise ArtifactValidationError("result directories have not been initialized")
        run_fd = _open_directory_chain(self.run_dir)
        try:
            if _directory_identity(run_fd) != expected["run"]:
                raise ArtifactValidationError("run directory identity changed after activation")
            namespace_fd = _open_child_directory(run_fd, namespace)
            try:
                if _directory_identity(namespace_fd) != expected[namespace]:
                    raise ArtifactValidationError(
                        f"{namespace} directory identity changed after activation"
                    )
                yield run_fd, namespace_fd
            finally:
                os.close(namespace_fd)
        finally:
            os.close(run_fd)

    @contextmanager
    def _open_pinned_run(self) -> Iterator[int]:
        """Pin the run root for all shared-artifact lineage reads."""

        if self._result_directory_identities is None:
            raise ArtifactValidationError("result directories have not been initialized")
        run_fd = _open_directory_chain(self.run_dir)
        try:
            expected = self._result_directory_identities["run"]
            if _directory_identity(run_fd) != expected:
                raise ArtifactValidationError("run directory identity changed")
            yield run_fd
        finally:
            os.close(run_fd)
        current_fd = _open_directory_chain(self.run_dir)
        try:
            current_identity = _directory_identity(current_fd)
        finally:
            os.close(current_fd)
        if current_identity != self._result_directory_identities["run"]:
            raise ArtifactValidationError("run directory identity changed after shared-artifact read")

    def _assert_result_namespace_current(self, namespace: str) -> None:
        # Re-open the current textual path after each operation.  The work was
        # anchored to safe fds; this detects a concurrent rename/substitution
        # while guaranteeing no access occurred through the replacement.
        with self._open_result_namespace(namespace):
            pass

    @staticmethod
    def _require_canonical_bytes_at(
        directory_fd: int,
        name: str,
        expected: bytes,
        *,
        kind: str,
    ) -> bytes:
        try:
            observed = secure_fs.read_bytes_at(directory_fd, name)
        except secure_fs.SecureFilesystemError as exc:
            raise ArtifactValidationError(f"cannot read prepared {kind}: {name}") from exc
        if observed != expected:
            raise ArtifactValidationError(f"non-canonical prepared {kind}: {name}")
        return observed

    @classmethod
    def _activate_prepared_batch(
        cls,
        stores: tuple["RunStore", ...],
        provenance: SystemProvenance,
    ) -> None:
        """Internally activate already-prepared stores as one transaction.

        The internal caller must have just applied the runtime policy and
        captured ``provenance`` from the current execution environment.
        Preparation must have created all plans and stored provenance already;
        this read-only boundary only verifies those bytes and flips the
        in-memory execution gates after every store has passed validation.
        Public callers must continue to use ``initialize(for_execution=True)``
        so policy application and live provenance capture cannot be skipped.
        """

        try:
            provided = tuple(stores)
        except TypeError as exc:
            raise ArtifactValidationError("prepared stores must be iterable") from exc

        # Clear stale readiness before any validation.  A failed reactivation
        # must never leave a previously-ready store usable.
        for store in provided:
            if isinstance(store, cls):
                store._execution_ready = False
        if not provided:
            raise ArtifactValidationError("prepared store batch cannot be empty")
        if any(not isinstance(store, cls) for store in provided):
            raise ArtifactValidationError("prepared batch contains a non-RunStore")
        if not isinstance(provenance, SystemProvenance):
            raise ArtifactValidationError("prepared batch provenance is invalid")
        provenance = _revalidate_instance(provenance, SystemProvenance)

        run_ids = [store.run_id for store in provided]
        if len(run_ids) != len(set(run_ids)):
            raise ArtifactValidationError("prepared batch contains duplicate run IDs")

        stack = ExitStack()
        stack.__enter__()
        canonical_paths: list[Path] = []
        result_identities: list[dict[str, tuple[int, int]]] = []
        run_fds: list[int] = []
        try:
            for store in provided:
                run_fd = _open_directory_chain(store.run_dir)
                stack.callback(os.close, run_fd)
                units_fd = _open_child_directory(run_fd, "units")
                stack.callback(os.close, units_fd)
                attempts_fd = _open_child_directory(run_fd, "attempts")
                stack.callback(os.close, attempts_fd)
                run_fds.append(run_fd)
                canonical_paths.append(Path(os.path.abspath(store.run_dir)))
                result_identities.append(
                    {
                        "run": _directory_identity(run_fd),
                        "units": _directory_identity(units_fd),
                        "attempts": _directory_identity(attempts_fd),
                    }
                )
            if len(canonical_paths) != len(set(canonical_paths)):
                raise ArtifactValidationError("prepared batch contains duplicate run paths")

            for store, run_fd in zip(provided, run_fds, strict=True):
                cls._require_canonical_bytes_at(
                    run_fd,
                    "config.json",
                    canonical_json_bytes(scientific_config_value(store.config)) + b"\n",
                    kind="config",
                )
                cls._require_canonical_bytes_at(
                    run_fd,
                    "expected-units.json",
                    canonical_json_bytes(store.expected.model_dump(mode="json")) + b"\n",
                    kind="expected-units",
                )
                cls._require_canonical_bytes_at(
                    run_fd,
                    "expected-shared-artifacts.json",
                    canonical_json_bytes(store.expected_shared.model_dump(mode="json")) + b"\n",
                    kind="expected-shared-artifacts",
                )
                stored_provenance = _load_model_at(run_fd, "provenance.json", SystemProvenance)
                cls._require_canonical_bytes_at(
                    run_fd,
                    "provenance.json",
                    canonical_json_bytes(stored_provenance.model_dump(mode="json", warnings=False))
                    + b"\n",
                    kind="provenance",
                )
                if _provenance_identity(stored_provenance) != _provenance_identity(provenance):
                    raise ArtifactValidationError(
                        "prepared provenance identity does not match supplied provenance"
                    )
                _validate_provenance_policy(provenance, store.config, provenance.resolved_device)
        # No mutation, policy application, or provenance capture occurs above
        # this point.  The batch becomes usable only after every store passes.
            for store, identities in zip(provided, result_identities, strict=True):
                if store._capture_result_directory_identities() != identities:
                    raise ArtifactValidationError(
                        "prepared result directory identity changed during activation"
                    )
                if (
                    store._result_directory_identities is not None
                    and store._result_directory_identities != identities
                ):
                    raise ArtifactValidationError(
                        "prepared result directory identity changed after prior activation"
                    )
            for store, identities in zip(provided, result_identities, strict=True):
                store._result_directory_identities = identities
                store._execution_ready = True
        except BaseException:
            stack.close()
            raise
        stack.close()

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

        self._execution_ready = False
        resolved_device = apply_runtime_policy(self.config.device_policy) if for_execution else None

        self.units_dir.mkdir(parents=True, exist_ok=True)
        self.attempts_dir.mkdir(parents=True, exist_ok=True)
        prepared_identities = self._capture_result_directory_identities()
        if (
            self._result_directory_identities is not None
            and prepared_identities != self._result_directory_identities
        ):
            raise ArtifactValidationError("result directory identity changed after initialization")
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
        if self._capture_result_directory_identities() != prepared_identities:
            raise ArtifactValidationError("result directory identity changed during initialization")
        self._result_directory_identities = prepared_identities
        self._execution_ready = for_execution

    def initialize_prepared(self, provenance: SystemProvenance) -> None:
        """Publish and pin a preparation-only store from supplied provenance.

        Readiness captures provenance once for the complete six-fold
        transaction.  This boundary must therefore never resolve a device or
        recapture the host if a file disappears during publication.  All four
        canonical RunStore files are published and verified relative to one
        retained run descriptor, and the result namespaces remain disabled for
        execution until the later prepared-batch activation succeeds.
        """

        self._execution_ready = False
        if not isinstance(provenance, SystemProvenance):
            raise ArtifactValidationError("prepared provenance is invalid")
        provenance = _revalidate_instance(provenance, SystemProvenance)

        stack = ExitStack()
        stack.__enter__()
        try:
            run_fd = _open_directory_chain(self.run_dir)
            stack.callback(os.close, run_fd)
            units_fd = _open_child_directory(run_fd, "units")
            stack.callback(os.close, units_fd)
            attempts_fd = _open_child_directory(run_fd, "attempts")
            stack.callback(os.close, attempts_fd)
            identities = {
                "run": _directory_identity(run_fd),
                "units": _directory_identity(units_fd),
                "attempts": _directory_identity(attempts_fd),
            }

            canonical_values = (
                ("config.json", scientific_config_value(self.config)),
                ("expected-units.json", self.expected.model_dump(mode="json")),
                (
                    "expected-shared-artifacts.json",
                    self.expected_shared.model_dump(mode="json"),
                ),
                ("provenance.json", provenance.model_dump(mode="json")),
            )
            for name, value in canonical_values:
                _exclusive_write_json_at(run_fd, run_fd, name, value)
                self._require_canonical_bytes_at(
                    run_fd,
                    name,
                    canonical_json_bytes(value) + b"\n",
                    kind=name.removesuffix(".json"),
                )

            stored_provenance = _load_model_at(run_fd, "provenance.json", SystemProvenance)
            if _provenance_identity(stored_provenance) != _provenance_identity(provenance):
                raise ArtifactValidationError(
                    "prepared provenance identity does not match supplied provenance"
                )
            if self._capture_result_directory_identities() != identities:
                raise ArtifactValidationError(
                    "prepared result directory identity changed during initialization"
                )
            if (
                self._result_directory_identities is not None
                and self._result_directory_identities != identities
            ):
                raise ArtifactValidationError(
                    "prepared result directory identity changed after prior initialization"
                )
            self._result_directory_identities = identities
        except BaseException:
            stack.close()
            raise
        stack.close()

    def planned_unit(self, unit_id: str) -> PlannedUnit:
        try:
            return self._expected_by_id[unit_id]
        except KeyError as exc:
            raise ArtifactValidationError(f"unexpected unit_id: {unit_id}") from exc

    def load_provenance(self) -> SystemProvenance:
        with self._open_pinned_run() as run_fd:
            return _load_model_at(run_fd, "provenance.json", SystemProvenance)

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
        with self._open_pinned_run() as run_fd:
            return self._load_shared_cost_at(run_fd, key_id, kind)

    def _load_shared_cost_at(
        self,
        run_fd: int,
        key_id: str,
        kind: str = "training_artifact",
        *,
        data_reader: Any | None = None,
        model_reader: Any | None = None,
    ) -> Any:
        """Load one planned shared cost through the pinned run descriptor."""

        self.planned_shared(key_id, kind)
        if kind == "training_data_evidence":
            from levelup.experiments.runner.training_data_artifacts import (
                load_training_data_evidence_cost_from_at,
                open_training_data_reader,
            )

            if data_reader is None:
                with open_training_data_reader(run_fd) as opened:
                    return load_training_data_evidence_cost_from_at(opened, key_id)
            return load_training_data_evidence_cost_from_at(data_reader, key_id)
        if kind == "training_data_view":
            from levelup.experiments.runner.training_data_artifacts import (
                load_training_data_view_cost_from_at,
                open_training_data_reader,
            )

            if data_reader is None:
                with open_training_data_reader(run_fd) as opened:
                    return load_training_data_view_cost_from_at(opened, key_id)
            return load_training_data_view_cost_from_at(data_reader, key_id)
        if kind != "training_artifact":
            raise ArtifactValidationError("unknown shared-artifact kind")
        from levelup.experiments.runner.training_artifacts import (
            load_training_cost_by_id_from_at,
            open_training_artifact_reader,
        )

        if model_reader is None:
            with open_training_artifact_reader(run_fd) as opened:
                raw = load_training_cost_by_id_from_at(opened, key_id)
        else:
            raw = load_training_cost_by_id_from_at(model_reader, key_id)
        if raw.key_id != key_id or raw.expected_cost_id != raw.cost_id:
            raise ArtifactValidationError("shared-artifact cost identity mismatch")
        return raw

    def validate_shared_reference(
        self, unit: PlannedUnit, reference: SharedArtifactReference
    ) -> None:
        with self._open_pinned_run() as run_fd:
            self._validate_shared_reference_set_at(run_fd, unit, (reference,))

    def _validate_shared_reference_at(
        self,
        run_fd: int,
        unit: PlannedUnit,
        reference: SharedArtifactReference,
        *,
        data_reader: Any | None,
        model_reader: Any | None,
        model_lineage: tuple[Any, Any, Any] | None = None,
    ) -> tuple[Any, Any]:
        from levelup.experiments.runner.training_artifacts import (
            TrainingArtifactKey,
            load_training_lineage_from_at,
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
        expected_provenance_sha256 = provenance_identity_sha256(
            _load_model_at(run_fd, "provenance.json", SystemProvenance)
        )
        expected_group = planned.owner_group_id or planned.owner_condition_id
        if reference.kind == "training_data_evidence":
            if data_reader is None:
                raise ArtifactValidationError("training-data reader is absent")
            try:
                from levelup.experiments.runner.training_data_artifacts import (
                    TrainingDataEvidenceKey,
                    load_training_data_evidence_bundle_from_at,
                )

                cost, manifest, _ = load_training_data_evidence_bundle_from_at(
                    data_reader, reference.key_id
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
            except (OSError, RuntimeError, ValueError) as exc:
                raise ArtifactValidationError("shared training-data evidence is invalid") from exc
            if (
                cost.artifact_id != reference.artifact_id
                or cost.cost_id != reference.cost_id
                or manifest.evidence_id != reference.artifact_id
            ):
                raise ArtifactValidationError(
                    "shared training-data evidence reference does not match its cost record"
                )
            return cost, manifest
        if reference.kind == "training_data_view":
            if data_reader is None:
                raise ArtifactValidationError("training-data reader is absent")
            try:
                from levelup.experiments.runner.training_data_artifacts import (
                    TrainingDataArtifactKey,
                    load_training_data_view_bundle_from_at,
                )

                cost, manifest, _ = load_training_data_view_bundle_from_at(
                    data_reader, reference.key_id
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
            except (OSError, RuntimeError, ValueError) as exc:
                raise ArtifactValidationError("shared training-data view is invalid") from exc
            if (
                cost.artifact_id != reference.artifact_id
                or cost.cost_id != reference.cost_id
                or manifest.artifact_id != reference.artifact_id
            ):
                raise ArtifactValidationError(
                    "shared training-data view reference does not match its cost record"
                )
            return cost, manifest
        if model_reader is None:
            raise ArtifactValidationError("training-artifact reader is absent")
        try:
            if model_lineage is None:
                raw_cost = self._load_shared_cost_at(
                    run_fd,
                    reference.key_id,
                    reference.kind,
                    data_reader=data_reader,
                    model_reader=model_reader,
                )
                training_key = TrainingArtifactKey.model_validate(raw_cost.key)
                manifest, index, cost = load_training_lineage_from_at(
                    model_reader, training_key
                )
            else:
                manifest, index, cost = model_lineage
                raw_cost = cost
                training_key = TrainingArtifactKey.model_validate(cost.key)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ArtifactValidationError("shared training artifact is invalid") from exc
        if cost.scope != "training_preparation":
            raise ArtifactValidationError("shared model cost has the wrong scope")
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
        if (
            raw_cost != cost
            or cost.artifact_id != reference.artifact_id
            or cost.cost_id != reference.cost_id
            or index.artifact_id != reference.artifact_id
            or manifest.artifact_id != reference.artifact_id
        ):
            raise ArtifactValidationError("shared-artifact files do not match the unit reference")
        return cost, manifest

    def validate_shared_reference_set(
        self,
        unit: PlannedUnit,
        references: tuple[SharedArtifactReference, ...],
    ) -> None:
        with self._open_pinned_run() as run_fd:
            self._validate_shared_reference_set_at(run_fd, unit, references)

    def _validate_shared_reference_set_at(
        self,
        run_fd: int,
        unit: PlannedUnit,
        references: tuple[SharedArtifactReference, ...],
        *,
        data_reader: Any | None = None,
        model_reader: Any | None = None,
        model_lineage: tuple[Any, Any, Any] | None = None,
    ) -> None:
        """Validate cross-kind provenance after validating each typed reference."""

        if data_reader is None and model_reader is None:
            from levelup.experiments.runner.training_artifacts import (
                open_training_artifact_reader,
            )
            from levelup.experiments.runner.training_data_artifacts import (
                open_training_data_reader,
            )

            kinds = {reference.kind for reference in references}
            with ExitStack() as stack:
                opened_data = (
                    stack.enter_context(open_training_data_reader(run_fd))
                    if kinds & {"training_data_evidence", "training_data_view"}
                    else None
                )
                opened_model = (
                    stack.enter_context(open_training_artifact_reader(run_fd))
                    if "training_artifact" in kinds
                    else None
                )
                self._validate_shared_reference_set_at(
                    run_fd,
                    unit,
                    references,
                    data_reader=opened_data,
                    model_reader=opened_model,
                    model_lineage=model_lineage,
                )
                return
        kinds = {reference.kind for reference in references}
        if (
            kinds & {"training_data_evidence", "training_data_view"}
            and data_reader is None
        ) or ("training_artifact" in kinds and model_reader is None):
            raise ArtifactValidationError("required shared reference reader is absent")

        validated_by_kind = {
            reference.kind: self._validate_shared_reference_at(
                run_fd,
                unit,
                reference,
                data_reader=data_reader,
                model_reader=model_reader,
                model_lineage=(
                    model_lineage if reference.kind == "training_artifact" else None
                ),
            )
            for reference in references
        }
        by_kind = {reference.kind: reference for reference in references}
        evidence = by_kind.get("training_data_evidence")
        view = by_kind.get("training_data_view")
        model = by_kind.get("training_artifact")
        if evidence is not None and view is not None:
            from levelup.experiments.runner.training_data_artifacts import evidence_key_for

            evidence_cost, _ = validated_by_kind["training_data_evidence"]
            view_cost, view_manifest = validated_by_kind["training_data_view"]
            if (
                evidence_key_for(view_cost.key) != evidence_cost.key
                or view_manifest.evidence_id != evidence.artifact_id
            ):
                raise ArtifactValidationError(
                    "shared training-data view does not derive from the referenced evidence"
                )
        if view is not None and model is not None:
            from levelup.experiments.runner.training_artifacts import TrainingArtifactKey

            model_cost, _ = validated_by_kind["training_artifact"]
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

    def load_completed(self, unit_id: str) -> UnitRecord | None:
        self.planned_unit(unit_id)
        name = f"{unit_id}.json"
        with self._open_result_namespace("units") as (_, namespace_fd):
            entries = _strict_namespace_entries(
                namespace_fd, "units", set(self._expected_by_id)
            )
            record = (
                _load_model_at(namespace_fd, name, UnitRecord) if name in entries else None
            )
        self._assert_result_namespace_current("units")
        if record is None:
            return None
        self._validate_completed_record(record, unit_id)
        return record

    def _validate_completed_record(self, record: UnitRecord, unit_id: str) -> None:
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
        name = f"{record.unit_id}.json"
        with self._open_result_namespace("units") as (run_fd, namespace_fd):
            entries = _strict_namespace_entries(
                namespace_fd, "units", set(self._expected_by_id)
            )
            if name in entries:
                existing = _load_model_at(namespace_fd, name, UnitRecord)
                self._validate_completed_record(existing, record.unit_id)
                if existing == record:
                    published = False
                else:
                    raise ConflictingResultError(
                        f"completed unit already exists: {record.unit_id}"
                    )
            else:
                published = _exclusive_write_json_at(
                    run_fd,
                    namespace_fd,
                    name,
                    record.model_dump(mode="json"),
                )
                if not published:
                    existing = _load_model_at(namespace_fd, name, UnitRecord)
                    self._validate_completed_record(existing, record.unit_id)
                    if existing != record:
                        raise ConflictingResultError(
                            f"completed unit already exists: {record.unit_id}"
                        )
                else:
                    stored = _load_model_at(namespace_fd, name, UnitRecord)
                    self._validate_completed_record(stored, record.unit_id)
        self._assert_result_namespace_current("units")
        return published

    def completed_records(self) -> tuple[UnitRecord, ...]:
        with self._open_result_namespace("units") as (_, namespace_fd):
            entries = set(
                _strict_namespace_entries(namespace_fd, "units", set(self._expected_by_id))
            )
            records: list[UnitRecord] = []
            for unit in self.expected.units:
                name = f"{unit.unit_id}.json"
                if name in entries:
                    record = _load_model_at(namespace_fd, name, UnitRecord)
                    self._validate_completed_record(record, unit.unit_id)
                    records.append(record)
        self._assert_result_namespace_current("units")
        return tuple(records)

    def missing_units(self) -> tuple[PlannedUnit, ...]:
        return tuple(
            unit for unit in self.expected.units if self.load_completed(unit.unit_id) is None
        )

    def next_attempt_number(self, unit_id: str) -> int:
        self.planned_unit(unit_id)
        with self._open_result_namespace("attempts") as (_, namespace_fd):
            entries = _strict_namespace_entries(
                namespace_fd, "attempts", set(self._expected_by_id)
            )
        prefix = f"{unit_id}.attempt-"
        numbers: list[int] = []
        for name in entries:
            suffix = name.removesuffix(".json").removeprefix(prefix)
            if name.startswith(prefix):
                numbers.append(int(suffix))
        result = max(numbers, default=0) + 1
        self._assert_result_namespace_current("attempts")
        return result

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
        if record.attempt > 9999:
            raise ArtifactValidationError("attempt number does not fit the frozen NNNN filename")
        name = f"{record.unit_id}.attempt-{record.attempt:04d}.json"
        with self._open_result_namespace("attempts") as (run_fd, namespace_fd):
            _strict_namespace_entries(namespace_fd, "attempts", set(self._expected_by_id))
            if not _exclusive_write_json_at(
                run_fd,
                namespace_fd,
                name,
                record.model_dump(mode="json"),
            ):
                raise ConflictingResultError(f"attempt already exists: {name}")
            stored = _load_model_at(namespace_fd, name, AttemptRecord)
            self._validate_attempt_record(stored, name)
        self._assert_result_namespace_current("attempts")

    def _validate_attempt_record(self, record: AttemptRecord, name: str) -> None:
        expected = self.planned_unit(record.unit_id)
        expected_name = f"{record.unit_id}.attempt-{record.attempt:04d}.json"
        if (
            name != expected_name
            or record.run_id != self.run_id
            or record.config_sha256 != self.config_sha256
            or record.key != expected.key
            or record.seeds != expected.seeds
        ):
            raise ArtifactValidationError(f"attempt identity mismatch: {name}")

    def attempt_records(self) -> tuple[AttemptRecord, ...]:
        records: list[AttemptRecord] = []
        with self._open_result_namespace("attempts") as (_, namespace_fd):
            entries = _strict_namespace_entries(
                namespace_fd, "attempts", set(self._expected_by_id)
            )
            for name in entries:
                record = _load_model_at(namespace_fd, name, AttemptRecord)
                self._validate_attempt_record(record, name)
                records.append(record)
        self._assert_result_namespace_current("attempts")
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
