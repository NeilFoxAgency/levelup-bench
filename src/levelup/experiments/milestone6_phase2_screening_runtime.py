"""Load the prepared, development-only Phase 2 screening inventory.

This module is deliberately a narrow boundary between the paid preparation pass and
held-out execution.  It does not build candidates, run an evaluator, invoke search,
aggregate records, or perform selection.  Its job is to prove that the immutable
readiness manifest, the three authority files, and all six prepared child trees still
describe the same development-only experiment before stores are activated.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from levelup.experiments.milestone6_phase2 import ROOT
from levelup.experiments.milestone6_phase2_screening import (
    build_screening_plan,
    screening_child_configs,
    selection_authority,
    validate_screening_plan,
)
from levelup.experiments.milestone6_phase2_screening_models import MaterializedScreeningModels
from levelup.experiments.milestone6_phase2_screening_preparation import (
    MaterializedScreeningData,
    ScreeningDataKeys,
    ScreeningModelKeys,
    build_screening_data_keys,
    build_screening_model_keys,
    build_screening_shared_plan,
)
from levelup.experiments.milestone6_phase2_screening_provenance import (
    CANONICAL_READINESS_PATH,
    canonical_screening_repository,
    validate_screening_provenance,
)
from levelup.experiments.milestone6_phase2_screening_readiness import (
    _CHILD_TOP_LEVEL_NAMES,
    ScreeningReadinessChild,
    ScreeningReadinessManifest,
    _child_manifest,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import (
    DevicePolicy,
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
from levelup.experiments.runner.records import (
    ExpectedSharedArtifacts,
    SystemProvenance,
)
from levelup.experiments.runner.storage import (
    RunStore,
    expected_units_sha256,
    plan_expected_units,
    provenance_identity_sha256,  # noqa: F401 - retained compatibility export
)
from levelup.experiments.runner.training_artifacts import open_training_artifact_reader
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataArtifactError,
    open_training_data_reader,
)

FileIdentity = tuple[int, int, int, int, int]
DirectorySnapshot = tuple[int, int, int, int, int, tuple[tuple[str, str], ...]]


def load_screening_data_inventory(
    config: ExperimentConfig,
    data_keys: ScreeningDataKeys,
    output_root: str | Path,
) -> MaterializedScreeningData:
    """Late-bound adapter for the preparation module's reload-only inventory API."""

    from levelup.experiments.milestone6_phase2_screening_preparation import (
        load_screening_data_inventory as loader,
    )

    return loader(config, data_keys, output_root)


def load_screening_model_inventory(
    config: ExperimentConfig,
    data_keys: ScreeningDataKeys,
    data: MaterializedScreeningData,
    model_keys: ScreeningModelKeys,
    run_dir: str | Path,
) -> MaterializedScreeningModels:
    """Late-bound adapter for the preparation module's reload-only model API."""

    from levelup.experiments.milestone6_phase2_screening_models import (
        load_screening_model_inventory as loader,
    )

    return loader(config, data_keys, data, model_keys, run_dir)


def load_screening_data_inventory_at(
    config: ExperimentConfig,
    data_keys: ScreeningDataKeys,
    run_fd: int,
    **retained: Any,
) -> MaterializedScreeningData:
    """Late-bound adapter for the descriptor-relative data inventory API."""

    from levelup.experiments.milestone6_phase2_screening_preparation import (
        load_screening_data_inventory_at as loader,
    )

    return loader(config, data_keys, run_fd, **retained)


def load_screening_model_inventory_at(
    config: ExperimentConfig,
    data_keys: ScreeningDataKeys,
    data: MaterializedScreeningData,
    model_keys: ScreeningModelKeys,
    run_fd: int,
    **retained: Any,
) -> MaterializedScreeningModels:
    """Late-bound adapter for the descriptor-relative model inventory API."""

    from levelup.experiments.milestone6_phase2_screening_models import (
        load_screening_model_inventory_at as loader,
    )

    return loader(config, data_keys, data, model_keys, run_fd, **retained)


def _activate_prepared_batch(
    stores: tuple[RunStore, ...], provenance: SystemProvenance
) -> None:
    """Private adapter for the transactional RunStore execution gate."""

    RunStore._activate_prepared_batch(stores, provenance)


@dataclass(frozen=True, slots=True)
class AuthoritySourceSnapshot:
    """Immutable bytes and digest for one frozen development authority source."""

    label: str
    path: Path
    content: bytes
    sha256: str
    parent_identity: tuple[int, int]
    file_identity: FileIdentity


@dataclass(frozen=True, slots=True)
class ScreeningRuntimeFold:
    """Typed prepared inventory for one leave-one-family-out development fold."""

    family_id: str
    config: ExperimentConfig
    store: RunStore
    data_keys: ScreeningDataKeys
    data: MaterializedScreeningData
    model_keys: ScreeningModelKeys
    models: MaterializedScreeningModels
    shared_plan: ExpectedSharedArtifacts


@dataclass(frozen=True, slots=True)
class ScreeningRuntime:
    """Frozen execution handle; no experiment work happens during construction."""

    manifest_path: Path
    raw_root: Path
    repository: Path
    device_policy: DevicePolicy
    manifest_bytes: bytes
    manifest: ScreeningReadinessManifest
    authority_sources: tuple[AuthoritySourceSnapshot, ...]
    provenance: SystemProvenance
    folds: tuple[ScreeningRuntimeFold, ...]
    tree_sha256: str
    raw_root_identity: tuple[int, int]
    child_identities: tuple[tuple[str, tuple[int, int]], ...]
    manifest_parent_identity: tuple[int, int]
    manifest_file_identity: FileIdentity
    result_namespace_snapshot: tuple[tuple[str, tuple[tuple[str, DirectorySnapshot], ...]], ...] = ()

    @property
    def authority_bytes_by_path(self) -> tuple[tuple[Path, bytes], ...]:
        return tuple((source.path, source.content) for source in self.authority_sources)

    @property
    def authority_digests(self) -> tuple[tuple[str, str], ...]:
        return tuple((source.label, source.sha256) for source in self.authority_sources)

    @property
    def children(self) -> tuple[ScreeningRuntimeFold, ...]:
        return self.folds

    def recheck_before_execution(self) -> None:
        """Reconfirm all authority, then transactionally open execution gates."""

        stores = tuple(fold.store for fold in self.folds)
        for store in stores:
            store._execution_ready = False
        try:
            if self.manifest_path != self.repository / CANONICAL_READINESS_PATH:
                _fail(
                    "screening execution requires the canonical committed readiness manifest"
                )
            if (
                self.raw_root_identity is None
                or not self.child_identities
                or self.manifest_parent_identity is None
                or self.manifest_file_identity is None
                or any(
                    source.parent_identity is None or source.file_identity is None
                    for source in self.authority_sources
                )
            ):
                _fail("screening runtime is missing pinned filesystem identities")
            try:
                captured = capture_system_provenance(self.repository, self.device_policy)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                _fail("cannot recapture screening runtime provenance", exc)
            try:
                validate_screening_provenance(
                    self.provenance,
                    captured,
                    repository=self.repository,
                    manifest_bytes=self.manifest_bytes,
                )
            except TrainingDataArtifactError:
                raise
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                _fail("screening runtime provenance changed after runtime load", exc)

            _recheck_manifest_and_tree(
                self.manifest_path,
                self.raw_root,
                self.manifest_bytes,
                self.manifest,
                self.authority_sources,
                self.tree_sha256,
                self.raw_root_identity,
                self.child_identities,
                self.manifest_parent_identity,
                self.manifest_file_identity,
                self.folds,
                self.result_namespace_snapshot,
            )
            _activate_prepared_batch(stores, self.provenance)
            # Activation is read-only, but immediately recheck the immutable
            # path boundary before returning ready stores to the caller.
            _recheck_manifest_and_tree(
                self.manifest_path,
                self.raw_root,
                self.manifest_bytes,
                self.manifest,
                self.authority_sources,
                self.tree_sha256,
                self.raw_root_identity,
                self.child_identities,
                self.manifest_parent_identity,
                self.manifest_file_identity,
                self.folds,
                self.result_namespace_snapshot,
            )
            # Git/worktree state is path-based rather than part of the retained
            # prepared-tree descriptors.  Recapture it after activation and the
            # final tree check so a checkout or dirty edit during that interval
            # closes every gate instead of escaping the pre-activation check.
            try:
                captured_after_activation = capture_system_provenance(
                    self.repository, self.device_policy
                )
                validate_screening_provenance(
                    self.provenance,
                    captured_after_activation,
                    repository=self.repository,
                    manifest_bytes=self.manifest_bytes,
                )
            except TrainingDataArtifactError:
                raise
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                _fail("screening runtime provenance changed during activation", exc)
        except Exception:
            for store in stores:
                store._execution_ready = False
            raise


def recheck_screening_runtime_readonly(runtime: ScreeningRuntime) -> None:
    """Revalidate a loaded runtime without activating any execution gate."""

    if not isinstance(runtime, ScreeningRuntime):
        _fail("read-only screening recheck requires a loaded ScreeningRuntime")
    if runtime.manifest_path != runtime.repository / CANONICAL_READINESS_PATH:
        _fail("read-only screening reuse requires the canonical committed manifest")
    if (
        runtime.raw_root_identity is None
        or not runtime.child_identities
        or runtime.manifest_parent_identity is None
        or runtime.manifest_file_identity is None
        or any(
            source.parent_identity is None or source.file_identity is None
            for source in runtime.authority_sources
        )
    ):
        _fail("read-only screening reuse is missing pinned filesystem identities")
    current_authority_sources = _authority_sources(runtime.manifest)
    if current_authority_sources != runtime.authority_sources:
        _fail("read-only screening authority sources differ from the loaded runtime")
    if runtime.manifest_bytes != (
        canonical_json_bytes(runtime.manifest.model_dump(mode="json")) + b"\n"
    ):
        _fail("screening runtime manifest bytes are not canonical")
    if runtime.manifest.provenance != runtime.provenance:
        _fail("screening runtime provenance differs from its manifest")
    try:
        captured = capture_system_provenance(runtime.repository, runtime.device_policy)
        validate_screening_provenance(
            runtime.provenance,
            captured,
            repository=runtime.repository,
            manifest_bytes=runtime.manifest_bytes,
        )
    except TrainingDataArtifactError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _fail("screening runtime provenance changed before read-only reuse", exc)
    if len(runtime.folds) != len(runtime.manifest.children):
        _fail("screening runtime fold inventory is incomplete")
    for fold, child in zip(runtime.folds, runtime.manifest.children, strict=True):
        canonical_data_keys = build_screening_data_keys(fold.config, runtime.provenance)
        if fold.data_keys != canonical_data_keys:
            _fail("screening runtime data keys differ from canonical authority")
        canonical_model_keys = build_screening_model_keys(
            fold.config,
            fold.data_keys,
            fold.data.manifests,
        )
        if fold.model_keys != canonical_model_keys:
            _fail("screening runtime model keys differ from canonical authority")
        canonical_shared = build_screening_shared_plan(
            fold.config,
            fold.data_keys,
            fold.data.manifests,
            fold.model_keys,
        )
        if fold.shared_plan != canonical_shared:
            _fail("screening runtime shared plan differs from canonical authority")
        if (
            _child_manifest(
                fold.config,
                fold.data_keys,
                fold.data,
                fold.model_keys,
                fold.models,
                fold.shared_plan,
                runtime.provenance,
            )
            != child
        ):
            _fail("screening runtime child inventory differs from readiness authority")
    _assert_global_inventory(runtime.manifest, runtime.folds)
    _recheck_manifest_and_tree(
        runtime.manifest_path,
        runtime.raw_root,
        runtime.manifest_bytes,
        runtime.manifest,
        runtime.authority_sources,
        runtime.tree_sha256,
        runtime.raw_root_identity,
        runtime.child_identities,
        runtime.manifest_parent_identity,
        runtime.manifest_file_identity,
        (),
        (),
    )


def _fail(message: str, exc: BaseException | None = None) -> None:
    if exc is None:
        raise TrainingDataArtifactError(message)
    raise TrainingDataArtifactError(message) from exc


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _reject_symlink_chain(path: Path, *, require_exists: bool = True) -> Path:
    """Return an absolute path after rejecting symlinked ancestors."""

    target = path.absolute()
    for candidate in (target, *target.parents):
        if os.path.lexists(candidate) and candidate.is_symlink():
            _fail(f"screening runtime path contains a symlink: {candidate}")
    if require_exists and not os.path.lexists(target):
        _fail(f"screening runtime path does not exist: {target}")
    return target


def _safe_basename(value: str, *, label: str) -> None:
    if (
        not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or value != value.strip()
        or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        _fail(f"unsafe screening runtime {label}")


def _canonical_json_file(path: Path, *, label: str) -> tuple[bytes, Any]:
    try:
        content, _parent_identity, _file_identity = _read_pinned_file(path, label=label)
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        _fail(f"screening runtime {label} is invalid", exc)
    return content, value


def _canonical_json_file_at(directory_fd: int, name: str, *, label: str) -> tuple[bytes, Any]:
    try:
        content = secure_fs.read_bytes_at(directory_fd, name)
        return content, json.loads(content)
    except (secure_fs.SecureFilesystemError, json.JSONDecodeError, TypeError, ValueError) as exc:
        _fail(f"screening runtime {label} is invalid", exc)
    raise AssertionError("unreachable")


def _read_pinned_file(
    path: Path,
    *,
    label: str,
    expected_parent_identity: tuple[int, int] | None = None,
    expected_file_identity: FileIdentity | None = None,
) -> tuple[bytes, tuple[int, int], FileIdentity]:
    """Read a regular file relative to one pinned parent directory."""

    target = _reject_symlink_chain(path)
    try:
        parent_fd = secure_fs.open_directory_chain(target.parent)
    except secure_fs.SecureFilesystemError as exc:
        _fail(f"screening runtime {label} parent cannot be securely opened", exc)
    try:
        parent_identity = secure_fs.directory_identity(parent_fd)
        if (
            expected_parent_identity is not None
            and parent_identity != expected_parent_identity
        ):
            _fail(f"screening runtime {label} parent identity changed")
        with secure_fs.open_regular_file_at(parent_fd, target.name) as file_fd:
            observed = os.fstat(file_fd)
            if not stat.S_ISREG(observed.st_mode):
                _fail(f"screening runtime {label} must be a regular file")
            # Device/inode alone is insufficient because filesystems may
            # immediately reuse an inode after unlink/recreate.  Preserve
            # nanosecond metadata and size so same-byte replacement remains
            # observable without weakening the retained-descriptor read.
            file_identity = (
                observed.st_dev,
                observed.st_ino,
                observed.st_ctime_ns,
                observed.st_mtime_ns,
                observed.st_size,
            )
            if expected_file_identity is not None and file_identity != expected_file_identity:
                _fail(f"screening runtime {label} identity changed")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(file_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks), parent_identity, file_identity
    except secure_fs.SecureFilesystemError as exc:
        _fail(f"screening runtime {label} is invalid", exc)
    except OSError as exc:
        _fail(f"screening runtime {label} is invalid", exc)
    finally:
        os.close(parent_fd)
    raise AssertionError("unreachable")


def _manifest_bytes(
    path: Path, pin: str
) -> tuple[bytes, ScreeningReadinessManifest, tuple[int, int], FileIdentity]:
    content, parent_identity, file_identity = _read_pinned_file(path, label="committed manifest")
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        _fail("committed readiness manifest is not canonical", exc)
    if _sha256(content) != pin:
        _fail("committed readiness manifest bytes do not match the supplied pin")
    try:
        manifest = ScreeningReadinessManifest.model_validate(value)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _fail("committed readiness manifest is not canonical", exc)
    if content != canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n":
        _fail("committed readiness manifest bytes are not canonical")
    return content, manifest, parent_identity, file_identity


def _authority_sources(manifest: ScreeningReadinessManifest) -> tuple[AuthoritySourceSnapshot, ...]:
    try:
        authority = selection_authority()
        rows = (
            ("protocol", authority.protocol_path, manifest.protocol_sha256),
            (
                "screening_candidates",
                authority.screening_candidates_path,
                manifest.screening_candidates_sha256,
            ),
            ("task_manifest", authority.task_manifest_path, manifest.task_manifest_sha256),
        )
        snapshots = []
        for label, path, expected in rows:
            target = _reject_symlink_chain(Path(path))
            content, parent_identity, file_identity = _read_pinned_file(target, label=f"{label} authority")
            digest = _sha256(content)
            if digest != expected:
                _fail(f"current {label} authority does not match the readiness manifest")
            snapshots.append(
                AuthoritySourceSnapshot(
                    label, target, content, digest, parent_identity, file_identity
                )
            )
        return tuple(snapshots)
    except TrainingDataArtifactError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _fail("cannot load the frozen development authority sources", exc)
    raise AssertionError("unreachable")


def _assert_development_manifest(
    manifest: ScreeningReadinessManifest,
    configs: tuple[ExperimentConfig, ...],
    plan: Any | None = None,
) -> None:
    if plan is None:
        try:
            plan = build_screening_plan()
            validate_screening_plan(plan)
        except (RuntimeError, TypeError, ValueError) as exc:
            _fail("current screening plan is not canonical", exc)
    if (
        plan.plan_id != manifest.screening_plan_id
        or tuple(plan.family_order) != tuple(manifest.family_order)
        or plan.protocol_sha256 != manifest.protocol_sha256
        or plan.screening_candidates_sha256 != manifest.screening_candidates_sha256
        or plan.task_manifest_sha256 != manifest.task_manifest_sha256
        or any(config.replicates != len(plan.replicates) for config in configs)
        or tuple(plan.replicates) != tuple(range(configs[0].replicates))
        or plan.expected_total_units != manifest.expected_total_units
        or plan.expected_total_evidence_artifacts
        != manifest.expected_total_evidence_artifacts
        or plan.expected_total_training_data_views
        != manifest.expected_total_training_data_views
        or plan.expected_total_model_artifacts != manifest.expected_total_model_artifacts
        or plan.final_family_access is not False
    ):
        _fail("readiness manifest does not match the current canonical screening plan")
    plan_children = tuple(plan.children)
    if len(plan_children) != len(manifest.children):
        _fail("screening plan child inventory does not match the readiness manifest")
    for plan_child, manifest_child in zip(plan_children, manifest.children, strict=True):
        if any(
            left != right
            for left, right in (
                (plan_child.heldout_family, manifest_child.heldout_family_id),
                (plan_child.run_id, manifest_child.run_id),
                (plan_child.config_sha256, manifest_child.config_sha256),
                (plan_child.expected_units_sha256, manifest_child.expected_units_sha256),
                (plan_child.expected_units, manifest_child.expected_units),
                (
                    plan_child.expected_evidence_artifacts,
                    manifest_child.expected_evidence_artifacts,
                ),
                (
                    plan_child.expected_training_data_views,
                    manifest_child.expected_training_data_views,
                ),
                (plan_child.expected_model_artifacts, manifest_child.expected_model_artifacts),
            )
        ):
            _fail("screening plan child identity differs from the readiness manifest")
    if (
        manifest.development_only is not True
        or manifest.final_family_access is not False
        or manifest.validation_executed is not False
        or manifest.search_executed is not False
        or manifest.outcomes_present is not False
        or manifest.selection_performed is not False
    ):
        _fail("readiness manifest is not a preparation-only development manifest")
    if len(configs) != 6 or tuple(config.parameters["heldout_family_id"] for config in configs) != manifest.family_order:
        _fail("readiness manifest does not contain the exact six canonical folds")
    if tuple(config.parameters["heldout_family_id"] for config in configs) != (
        "plain", "battery", "cooldown", "heat", "momentum", "combo"
    ):
        _fail("screening family order is not the frozen development order")
    for config, child in zip(configs, manifest.children, strict=True):
        if (
            child.heldout_family_id != str(config.parameters["heldout_family_id"])
            or child.run_id != run_id_for(config)
            or child.config_sha256 != scientific_config_sha256(config)
            or child.expected_units_sha256 != expected_units_sha256(plan_expected_units(config))
            or config.split.final_tasks
            or any("final" in condition.execution_phases for condition in config.conditions)
        ):
            _fail("canonical screening child differs from readiness manifest authority")


def _walk_tree_digest_at(
    root_fd: int,
    expected_child_identities: tuple[tuple[str, tuple[int, int]], ...] = (),
    canonical_child_ids: tuple[str, ...] = (),
) -> str:
    """Hash one already-pinned runtime tree without resolving its path."""

    digest = hashlib.sha256()
    expected = dict(expected_child_identities)
    canonical = set(canonical_child_ids) or set(expected)

    def visit(directory_fd: int, relative: str) -> None:
        try:
            with os.scandir(directory_fd) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
                for entry in entries:
                    name = entry.name
                    child_relative = f"{relative}/{name}" if relative else name
                    if entry.is_symlink():
                        _fail(f"screening runtime tree contains a symlink: {child_relative}")
                    if entry.is_dir(follow_symlinks=False):
                        digest.update(b"D\0" + child_relative.encode() + b"\0")
                        child_fd = secure_fs.open_child_directory(directory_fd, name)
                        try:
                            if name in expected and secure_fs.directory_identity(child_fd) != expected[name]:
                                _fail(f"screening runtime child identity changed: {name}")
                            # Result namespaces are mutable write-once state.  Bind
                            # their directory type and identity, but deliberately
                            # exclude their entries from the immutable prepared-tree
                            # digest so resumable execution can publish records.
                            if relative in canonical and name in {"units", "attempts"}:
                                namespace_identity = secure_fs.directory_identity(child_fd)
                                digest.update(
                                    f"I\0{child_relative}\0{namespace_identity[0]}:{namespace_identity[1]}\0".encode()
                                )
                                continue
                            visit(child_fd, child_relative)
                        finally:
                            os.close(child_fd)
                    elif entry.is_file(follow_symlinks=False):
                        digest.update(b"F\0" + child_relative.encode() + b"\0")
                        digest.update(
                            _sha256(secure_fs.read_bytes_at(directory_fd, name)).encode() + b"\0"
                        )
                    else:
                        _fail(
                            f"screening runtime tree contains a non-regular entry: {child_relative}"
                        )
        except OSError as exc:
            _fail("cannot enumerate the screening runtime tree", exc)
    visit(root_fd, "")
    return digest.hexdigest()


def _assert_tree_shape_at(
    raw_fd: int,
    manifest: ScreeningReadinessManifest,
    expected_child_identities: tuple[tuple[str, tuple[int, int]], ...] = (),
    expected_units_by_run: dict[str, set[str]] | None = None,
) -> None:
    expected_names = {"phase2-screening-readiness.json", *manifest.child_run_ids}
    try:
        with os.scandir(raw_fd) as iterator:
            entries = tuple(iterator)
            observed_names = {entry.name for entry in entries}
            if observed_names != expected_names:
                _fail("screening runtime raw root has unexpected direct children")
            for entry in entries:
                if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    if entry.name != "phase2-screening-readiness.json" or entry.is_symlink():
                        _fail("screening runtime raw root contains an unsafe entry")
            expected = dict(expected_child_identities)
            for run_id in manifest.child_run_ids:
                _safe_basename(run_id, label="child run id")
                child_fd = secure_fs.open_child_directory(raw_fd, run_id)
                try:
                    if run_id in expected and secure_fs.directory_identity(child_fd) != expected[run_id]:
                        _fail(f"screening runtime child identity changed: {run_id}")
                    child_names: set[str] = set()
                    child_dirs: set[str] = set()
                    with os.scandir(child_fd) as child_entries:
                        for child_entry in child_entries:
                            if child_entry.is_symlink():
                                _fail("screening runtime child contains a symlink")
                            if child_entry.is_dir(follow_symlinks=False):
                                child_dirs.add(child_entry.name)
                            elif child_entry.is_file(follow_symlinks=False):
                                child_names.add(child_entry.name)
                            else:
                                _fail("screening runtime child contains a non-regular entry")
                    if child_names | child_dirs != _CHILD_TOP_LEVEL_NAMES:
                        _fail("screening runtime child top-level names are incomplete or extra")
                    for namespace in ("units", "attempts"):
                        namespace_fd = secure_fs.open_child_directory(child_fd, namespace)
                        try:
                            _assert_result_namespace_shape_at(
                                namespace_fd,
                                namespace,
                                None
                                if expected_units_by_run is None
                                else expected_units_by_run.get(run_id, set()),
                            )
                        finally:
                            os.close(namespace_fd)
                    if "aggregate.json" in child_names:
                        _fail("screening runtime contains forbidden aggregate state")
                finally:
                    os.close(child_fd)
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        _fail("screening runtime tree shape is unsafe", exc)


def _assert_result_namespace_shape_at(
    namespace_fd: int, namespace: str, expected_units: set[str] | None
) -> None:
    """Reject unsafe or unexpected result entry names without reading outcomes."""

    try:
        entries = secure_fs.strict_regular_entries(namespace_fd)
    except secure_fs.SecureFilesystemError as exc:
        _fail(f"screening runtime {namespace} namespace is unsafe", exc)
    for name in entries:
        if namespace == "units":
            unit_id = name[:-5] if name.endswith(".json") else ""
            if (
                not name.endswith(".json")
                or (expected_units is None and (len(unit_id) != 64 or any(c not in "0123456789abcdef" for c in unit_id)))
                or (expected_units is not None and unit_id not in expected_units)
            ):
                _fail(f"screening runtime contains unexpected unit result: {name}")
            continue
        stem = name.removesuffix(".json")
        unit_id, separator, number = stem.rpartition(".attempt-")
        if (
            not name.endswith(".json")
            or not separator
            or (
                expected_units is None
                and (len(unit_id) != 64 or any(c not in "0123456789abcdef" for c in unit_id))
            )
            or (expected_units is not None and unit_id not in expected_units)
            or len(number) != 4
            or not number.isdigit()
            or int(number) < 1
        ):
            _fail(f"screening runtime contains unexpected attempt result: {name}")


def _tree_identities_at(
    raw_fd: int, manifest: ScreeningReadinessManifest
) -> tuple[tuple[int, int], tuple[tuple[str, tuple[int, int]], ...]]:
    root_identity = secure_fs.directory_identity(raw_fd)
    children: list[tuple[str, tuple[int, int]]] = []
    for run_id in manifest.child_run_ids:
        child_fd = secure_fs.open_child_directory(raw_fd, run_id)
        try:
            children.append((run_id, secure_fs.directory_identity(child_fd)))
        finally:
            os.close(child_fd)
    return root_identity, tuple(children)


def _result_namespace_snapshot(
    folds: tuple[ScreeningRuntimeFold, ...],
) -> tuple[tuple[str, tuple[tuple[str, DirectorySnapshot], ...]], ...]:
    """Capture stable run/namespace identities and directory types.

    This snapshot is intentionally separate from the immutable prepared-tree
    digest: a fresh runtime load captures the current write-once inventory, and
    all validation/activation gates require it to remain byte-identical.
    """

    snapshots: list[tuple[str, tuple[tuple[str, DirectorySnapshot], ...]]] = []
    for fold in folds:
        store = fold.store
        # Test doubles for the loader boundary may not expose a filesystem
        # backed RunStore.  Real prepared folds always do, and are checked
        # strictly below.
        if not hasattr(store, "run_dir"):
            continue
        if store._result_directory_identities is None:
            _fail(f"screening runtime {store.run_id} result identities are not pinned")
        entries: list[tuple[str, DirectorySnapshot]] = []
        try:
            for namespace in ("units", "attempts"):
                # RunStore compares the textual path to its retained run and
                # namespace identities before yielding either descriptor.  Do
                # not read a replacement namespace before that comparison.
                with store._open_result_namespace(namespace) as (_, namespace_fd):
                    observed = os.fstat(namespace_fd)
                    if not stat.S_ISDIR(observed.st_mode):
                        _fail(f"screening runtime {store.run_id} {namespace} is not a directory")
                    names_and_digests = tuple(
                        (name, _sha256(secure_fs.read_bytes_at(namespace_fd, name)))
                        for name in secure_fs.strict_regular_entries(namespace_fd)
                    )
                    entries.append(
                        (
                            namespace,
                            (
                                observed.st_dev,
                                observed.st_ino,
                                observed.st_ctime_ns,
                                observed.st_mtime_ns,
                                stat.S_IFMT(observed.st_mode),
                                names_and_digests,
                            ),
                        )
                    )
            snapshots.append((store.run_id, tuple(entries)))
        except (OSError, RuntimeError, ValueError, secure_fs.SecureFilesystemError) as exc:
            _fail(f"screening runtime {store.run_id} result namespace is invalid", exc)
    return tuple(snapshots)


def _recheck_manifest_and_tree(
    manifest_path: Path,
    raw_root: Path,
    manifest_bytes: bytes,
    manifest: ScreeningReadinessManifest,
    authority_sources: tuple[AuthoritySourceSnapshot, ...],
    tree_sha256: str,
    raw_root_identity: tuple[int, int],
    child_identities: tuple[tuple[str, tuple[int, int]], ...],
    manifest_parent_identity: tuple[int, int],
    manifest_file_identity: FileIdentity,
    folds: tuple[ScreeningRuntimeFold, ...] = (),
    expected_result_namespace_snapshot: tuple[
        tuple[str, tuple[tuple[str, DirectorySnapshot], ...]], ...
    ] = (),
) -> None:
    current_manifest, _current_parent_identity, _current_file_identity = _read_pinned_file(
        manifest_path,
        label="committed manifest",
        expected_parent_identity=manifest_parent_identity,
        expected_file_identity=manifest_file_identity,
    )
    if current_manifest != manifest_bytes:
        _fail("committed readiness manifest changed after runtime load")
    for source in authority_sources:
        try:
            current, _parent_identity, _file_identity = _read_pinned_file(
                source.path,
                label=f"{source.label} authority",
                expected_parent_identity=source.parent_identity,
                expected_file_identity=source.file_identity,
            )
        except TrainingDataArtifactError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            _fail(f"cannot reread {source.label} authority", exc)
        if (
            current != source.content
            or _sha256(current) != source.sha256
        ):
            _fail(f"{source.label} authority changed after runtime load")
    try:
        raw_fd = secure_fs.open_directory_chain(raw_root)
    except secure_fs.SecureFilesystemError as exc:
        _fail("screening runtime raw root cannot be securely reopened", exc)
    try:
        observed_root_identity, observed_child_identities = _tree_identities_at(raw_fd, manifest)
        if observed_root_identity != raw_root_identity:
            _fail("screening runtime raw root identity changed after runtime load")
        if observed_child_identities != child_identities:
            _fail("screening runtime child identity changed after runtime load")
        raw_manifest, raw_value = _canonical_json_file_at(
            raw_fd, "phase2-screening-readiness.json", label="raw-root manifest"
        )
        if raw_manifest != manifest_bytes or raw_value != manifest.model_dump(mode="json"):
            _fail("raw-root readiness manifest changed after runtime load")
        expected_units_by_run = {
            fold.store.run_id: {unit.unit_id for unit in fold.store.expected.units}
            for fold in folds
        }
        _assert_tree_shape_at(
            raw_fd,
            manifest,
            child_identities,
            expected_units_by_run if folds else None,
        )
        _validate_result_namespaces(folds, expected_result_namespace_snapshot)
        if (
            _walk_tree_digest_at(
                raw_fd,
                child_identities,
                canonical_child_ids=manifest.child_run_ids,
            )
            != tree_sha256
        ):
            _fail("screening runtime tree changed after runtime load")
    finally:
        os.close(raw_fd)


def _validate_result_namespaces(
    folds: tuple[ScreeningRuntimeFold, ...],
    expected_snapshot: tuple[
        tuple[str, tuple[tuple[str, DirectorySnapshot], ...]], ...
    ] = (),
) -> None:
    """Validate resumable result state through each store's pinned APIs."""

    before_snapshot = _result_namespace_snapshot(folds)
    if expected_snapshot and before_snapshot != expected_snapshot:
        _fail("screening runtime result namespace snapshot changed")
    for fold in folds:
        store = fold.store
        try:
            identities = store._capture_result_directory_identities()
            if (
                store._result_directory_identities is not None
                and identities != store._result_directory_identities
            ):
                _fail(f"screening runtime {store.run_id} result directory identity changed")
            store._result_directory_identities = identities
            # These APIs securely enumerate, parse, and identity-check every
            # existing record.  They intentionally do not aggregate or select
            # and therefore do not inspect comparative outcome values.
            store.completed_records()
            store.attempt_records()
        except TrainingDataArtifactError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            _fail(f"screening runtime {store.run_id} result state is invalid", exc)
    if _result_namespace_snapshot(folds) != before_snapshot:
        _fail("screening runtime result state changed during validation")


def _assert_global_inventory(
    manifest: ScreeningReadinessManifest,
    folds: tuple[ScreeningRuntimeFold, ...],
) -> None:
    observed = {
        "evidence_key_ids": tuple(
            sorted(key.key_id for fold in folds for key in fold.data_keys.evidence.values())
        ),
        "view_key_ids": tuple(
            sorted(key.key_id for fold in folds for key in fold.data_keys.views.values())
        ),
        "model_key_ids": tuple(
            sorted(key.key_id for fold in folds for key in fold.model_keys.models.values())
        ),
        "shared_artifact_key_ids": tuple(
            sorted(item.key_id for fold in folds for item in fold.shared_plan.artifacts)
        ),
        "model_artifact_ids": tuple(
            sorted(
                item.artifact_id
                for fold in folds
                for item in fold.models.manifests.values()
            )
        ),
    }
    for name, value in observed.items():
        if value != tuple(getattr(manifest, name)):
            _fail(f"screening readiness global {name} union differs from child inventories")


def _load_fold(
    config: ExperimentConfig,
    child_manifest: ScreeningReadinessChild,
    raw_root: Path,
    raw_fd: int,
    expected_child_identity: tuple[int, int],
    repository: Path,
    provenance: SystemProvenance,
) -> ScreeningRuntimeFold:
    try:
        with ExitStack() as stack:
            child_fd = secure_fs.open_child_directory(raw_fd, child_manifest.run_id)
            stack.callback(os.close, child_fd)
            if secure_fs.directory_identity(child_fd) != expected_child_identity:
                _fail("screening runtime child identity changed before fold loading")
            child_metadata: dict[str, tuple[bytes, Any]] = {
                name: _canonical_json_file_at(child_fd, name, label=name)
                for name in (
                    "config.json",
                    "expected-units.json",
                    "expected-shared-artifacts.json",
                    "provenance.json",
                )
            }
            data_reader = stack.enter_context(open_training_data_reader(child_fd))
            model_reader = stack.enter_context(open_training_artifact_reader(child_fd))
            data_intent_fd = secure_fs.open_child_directory(
                child_fd, "screening-data-intents"
            )
            stack.callback(os.close, data_intent_fd)
            model_intent_fd = secure_fs.open_child_directory(
                child_fd, "screening-model-intents"
            )
            stack.callback(os.close, model_intent_fd)
            data_keys = build_screening_data_keys(config, provenance)
            data = load_screening_data_inventory_at(
                config,
                data_keys,
                child_fd,
                reader=data_reader,
                intent_fd=data_intent_fd,
            )
            model_keys = build_screening_model_keys(config, data_keys, data.manifests)
            models = load_screening_model_inventory_at(
                config,
                data_keys,
                data,
                model_keys,
                child_fd,
                data_reader=data_reader,
                data_intent_fd=data_intent_fd,
                model_reader=model_reader,
                model_intent_fd=model_intent_fd,
            )
            shared = build_screening_shared_plan(
                config, data_keys, data.manifests, model_keys
            )
            expected_child = _child_manifest(
                config,
                data_keys,
                data,
                model_keys,
                models,
                shared,
                provenance,
            )
            if expected_child != child_manifest:
                _fail(
                    "screening runtime child inventory does not match the readiness manifest"
                )
            store = RunStore(
                raw_root,
                config,
                repository=repository,
                shared_artifacts=tuple(shared.artifacts),
            )
            for name, expected in (
                ("config.json", scientific_config_value(config)),
                ("expected-units.json", store.expected.model_dump(mode="json")),
                (
                    "expected-shared-artifacts.json",
                    store.expected_shared.model_dump(mode="json"),
                ),
                ("provenance.json", provenance.model_dump(mode="json")),
            ):
                content, value = child_metadata[name]
                if content != canonical_json_bytes(expected) + b"\n" or value != expected:
                    _fail(f"screening runtime child {name} does not match its authority")

            units_fd = secure_fs.open_child_directory(child_fd, "units")
            stack.callback(os.close, units_fd)
            attempts_fd = secure_fs.open_child_directory(child_fd, "attempts")
            stack.callback(os.close, attempts_fd)
            anchored_identities = {
                "run": secure_fs.directory_identity(child_fd),
                "units": secure_fs.directory_identity(units_fd),
                "attempts": secure_fs.directory_identity(attempts_fd),
            }
            # Pin the result namespaces from the same retained child descriptor
            # used for every metadata and artifact read.  Reopening the textual
            # path is permitted only to compare identities before any result
            # bytes are read through RunStore.
            current_run_fd = secure_fs.open_directory_chain(store.run_dir)
            stack.callback(os.close, current_run_fd)
            if (
                secure_fs.directory_identity(current_run_fd)
                != anchored_identities["run"]
            ):
                _fail("screening runtime child result path changed before pinning")
            current_units_fd = secure_fs.open_child_directory(current_run_fd, "units")
            stack.callback(os.close, current_units_fd)
            current_attempts_fd = secure_fs.open_child_directory(
                current_run_fd, "attempts"
            )
            stack.callback(os.close, current_attempts_fd)
            if {
                "run": secure_fs.directory_identity(current_run_fd),
                "units": secure_fs.directory_identity(current_units_fd),
                "attempts": secure_fs.directory_identity(current_attempts_fd),
            } != anchored_identities:
                _fail("screening runtime child result path changed before pinning")
            store._result_directory_identities = anchored_identities
            fold = ScreeningRuntimeFold(
                family_id=child_manifest.heldout_family_id,
                config=config,
                store=store,
                data_keys=data_keys,
                data=data,
                model_keys=model_keys,
                models=models,
                shared_plan=shared,
            )
            # Existing partial/complete write-once state is valid input to a
            # resumable reload, but every entry must pass the fd-pinned RunStore
            # schema and identity validators before the child descriptor closes.
            _validate_result_namespaces((fold,))
            return fold
    except TrainingDataArtifactError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        _fail("screening runtime child inventory failed closed", exc)



def load_screening_runtime(
    manifest_path: str | Path,
    raw_root: str | Path,
    repository: str | Path,
    *,
    manifest_bytes_sha256: str,
    provenance: SystemProvenance | None = None,
) -> ScreeningRuntime:
    """Load and validate the exact six-fold development screening inventory."""

    if len(manifest_bytes_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_bytes_sha256
    ):
        _fail("manifest byte pin is not a lowercase SHA-256 digest")
    committed = _reject_symlink_chain(Path(manifest_path))
    root = _reject_symlink_chain(Path(raw_root))
    repository_path = _reject_symlink_chain(Path(repository)).resolve(strict=True)
    try:
        repository_path = canonical_screening_repository(
            repository_path, authority_root=ROOT
        )
    except TrainingDataArtifactError as exc:
        _fail(str(exc), exc)
    manifest_bytes, manifest, manifest_parent_identity, manifest_file_identity = _manifest_bytes(
        committed, manifest_bytes_sha256
    )
    try:
        root_fd = secure_fs.open_directory_chain(root)
    except secure_fs.SecureFilesystemError as exc:
        _fail("screening runtime raw root cannot be securely opened", exc)
    try:
        raw_manifest_bytes, raw_value = _canonical_json_file_at(
            root_fd, "phase2-screening-readiness.json", label="raw-root manifest"
        )
        if raw_manifest_bytes != manifest_bytes or raw_value != manifest.model_dump(mode="json"):
            _fail("raw-root readiness manifest differs from the pinned committed manifest")
        _assert_tree_shape_at(root_fd, manifest)
        tree_sha256 = _walk_tree_digest_at(
            root_fd,
            canonical_child_ids=manifest.child_run_ids,
        )
        raw_root_identity, child_identities = _tree_identities_at(root_fd, manifest)
    finally:
        os.close(root_fd)
    authority_sources = _authority_sources(manifest)
    try:
        plan = build_screening_plan()
        validate_screening_plan(plan)
    except (RuntimeError, TypeError, ValueError) as exc:
        _fail("current screening plan is not canonical", exc)
    configs = screening_child_configs()
    _assert_development_manifest(manifest, configs, plan)
    supplied_provenance = (
        None
        if provenance is None
        else SystemProvenance.model_validate(provenance.model_dump(mode="json"))
    )
    apply_runtime_policy(configs[0].device_policy)
    captured_provenance = capture_system_provenance(repository_path, configs[0].device_policy)
    try:
        validate_screening_provenance(
            manifest.provenance,
            captured_provenance,
            repository=repository_path,
            manifest_bytes=manifest_bytes,
        )
        if supplied_provenance is not None:
            validate_screening_provenance(
                supplied_provenance,
                captured_provenance,
                repository=repository_path,
                manifest_bytes=manifest_bytes,
            )
    except TrainingDataArtifactError as exc:
        _fail(f"current captured screening provenance rejected: {exc}", exc)
    # Preserve the first-writer timestamp only after current policy/provenance
    # identity has been independently re-established.
    provenance_value = manifest.provenance
    child_identity_map = dict(child_identities)
    try:
        fold_root_fd = secure_fs.open_directory_chain(root)
    except secure_fs.SecureFilesystemError as exc:
        _fail("screening runtime raw root cannot be pinned for fold loading", exc)
    try:
        if secure_fs.directory_identity(fold_root_fd) != raw_root_identity:
            _fail("screening runtime raw root changed before fold loading")
        folds = tuple(
            _load_fold(
                config,
                child,
                root,
                fold_root_fd,
                child_identity_map[child.run_id],
                repository_path,
                provenance_value,
            )
            for config, child in zip(configs, manifest.children, strict=True)
        )
    finally:
        os.close(fold_root_fd)
    _assert_global_inventory(manifest, folds)
    result_namespace_snapshot = _result_namespace_snapshot(folds)
    stores = tuple(fold.store for fold in folds)
    for store in stores:
        store._execution_ready = False
    # Loading proves the immutable inventory but deliberately leaves execution
    # locked.  The caller must perform a fresh required recheck immediately
    # before execution to open all gates transactionally.
    _recheck_manifest_and_tree(
        committed,
        root,
        manifest_bytes,
        manifest,
        authority_sources,
        tree_sha256,
        raw_root_identity,
        child_identities,
        manifest_parent_identity,
        manifest_file_identity,
        folds,
        result_namespace_snapshot,
    )
    return ScreeningRuntime(
        manifest_path=committed,
        raw_root=root,
        repository=repository_path,
        device_policy=configs[0].device_policy,
        manifest_bytes=manifest_bytes,
        manifest=manifest,
        authority_sources=authority_sources,
        provenance=provenance_value,
        folds=folds,
        tree_sha256=tree_sha256,
        raw_root_identity=raw_root_identity,
        child_identities=child_identities,
        manifest_parent_identity=manifest_parent_identity,
        manifest_file_identity=manifest_file_identity,
        result_namespace_snapshot=result_namespace_snapshot,
    )


__all__ = [
    "AuthoritySourceSnapshot",
    "ScreeningRuntime",
    "ScreeningRuntimeFold",
    "recheck_screening_runtime_readonly",
    "load_screening_runtime",
    "load_screening_data_inventory",
    "load_screening_model_inventory",
]
