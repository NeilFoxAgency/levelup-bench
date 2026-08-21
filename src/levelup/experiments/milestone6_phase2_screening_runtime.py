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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from levelup.experiments.milestone6_phase2_screening_readiness import (
    _CHILD_TOP_LEVEL_NAMES,
    ScreeningReadinessChild,
    ScreeningReadinessManifest,
    _child_manifest,
    _validate_child_top_level,
    load_screening_readiness_manifest,
)
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
    provenance_identity_sha256,
)
from levelup.experiments.runner.training_data_artifacts import TrainingDataArtifactError


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
            try:
                captured = capture_system_provenance(self.repository, self.device_policy)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                _fail("cannot recapture screening runtime provenance", exc)
            captured_identity = provenance_identity_sha256(captured)
            if (
                captured_identity != self.manifest.provenance_sha256
                or captured_identity != provenance_identity_sha256(self.provenance)
            ):
                _fail("screening runtime provenance changed after runtime load")

            _recheck_manifest_and_tree(
                self.manifest_path,
                self.raw_root,
                self.manifest_bytes,
                self.manifest,
                self.authority_sources,
                self.tree_sha256,
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
            )
        except Exception:
            for store in stores:
                store._execution_ready = False
            raise


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
    _reject_symlink_chain(path)
    if not path.is_file():
        _fail(f"screening runtime {label} must be a regular file")
    try:
        content = path.read_bytes()
        value = json.loads(content)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        _fail(f"screening runtime {label} is invalid", exc)
    return content, value


def _manifest_bytes(path: Path, pin: str) -> tuple[bytes, ScreeningReadinessManifest]:
    content, _ = _canonical_json_file(path, label="committed manifest")
    if _sha256(content) != pin:
        _fail("committed readiness manifest bytes do not match the supplied pin")
    try:
        manifest = load_screening_readiness_manifest(path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _fail("committed readiness manifest is not canonical", exc)
    if content != canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n":
        _fail("committed readiness manifest bytes are not canonical")
    return content, manifest


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
            content = target.read_bytes()
            digest = _sha256(content)
            if digest != expected:
                _fail(f"current {label} authority does not match the readiness manifest")
            snapshots.append(
                AuthoritySourceSnapshot(label, target, content, digest)
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


def _walk_tree_digest(root: Path) -> str:
    """Hash names, types, and bytes while rejecting every symlink."""

    digest = hashlib.sha256()

    def visit(directory: Path, relative: str) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
        except OSError as exc:
            _fail("cannot enumerate the screening runtime tree", exc)
        for entry in entries:
            if entry.is_symlink():
                _fail(f"screening runtime tree contains a symlink: {entry}")
            child_relative = f"{relative}/{entry.name}" if relative else entry.name
            if entry.is_dir():
                digest.update(b"D\0" + child_relative.encode() + b"\0")
                visit(entry, child_relative)
            elif entry.is_file():
                digest.update(b"F\0" + child_relative.encode() + b"\0")
                try:
                    content = entry.read_bytes()
                except OSError as exc:
                    _fail("cannot read the screening runtime tree", exc)
                digest.update(_sha256(content).encode() + b"\0")
            else:
                _fail(f"screening runtime tree contains a non-regular entry: {entry}")

    visit(root, "")
    return digest.hexdigest()


def _assert_tree_shape(raw_root: Path, manifest: ScreeningReadinessManifest) -> None:
    _reject_symlink_chain(raw_root)
    if not raw_root.is_dir():
        _fail("screening runtime raw root must be a directory")
    expected_names = {"phase2-screening-readiness.json", *manifest.child_run_ids}
    observed_names = {entry.name for entry in raw_root.iterdir()}
    if observed_names != expected_names:
        _fail("screening runtime raw root has unexpected direct children")
    for run_id in manifest.child_run_ids:
        _safe_basename(run_id, label="child run id")
        child = raw_root / run_id
        _reject_symlink_chain(child)
        if child.parent != raw_root or not child.is_dir():
            _fail("screening runtime child is not a safe direct directory")
        try:
            _validate_child_top_level(child)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            _fail("screening runtime child top-level inventory is invalid", exc)
        if {entry.name for entry in child.iterdir()} != _CHILD_TOP_LEVEL_NAMES:
            _fail("screening runtime child top-level names are incomplete or extra")
        # These namespaces must exist but contain no outcome or attempt state.
        for namespace in ("units", "attempts"):
            directory = child / namespace
            if directory.is_symlink() or not directory.is_dir() or tuple(directory.iterdir()):
                _fail(f"screening runtime {namespace} namespace is not empty")
        if (child / "aggregate.json").exists():
            _fail("screening runtime contains forbidden aggregate state")
    parent_manifest = raw_root / "phase2-screening-readiness.json"
    _reject_symlink_chain(parent_manifest)
    if not parent_manifest.is_file():
        _fail("screening runtime raw root is missing its readiness manifest")


def _recheck_manifest_and_tree(
    manifest_path: Path,
    raw_root: Path,
    manifest_bytes: bytes,
    manifest: ScreeningReadinessManifest,
    authority_sources: tuple[AuthoritySourceSnapshot, ...],
    tree_sha256: str,
) -> None:
    current_manifest = _manifest_bytes(manifest_path, _sha256(manifest_bytes))[0]
    if current_manifest != manifest_bytes:
        _fail("committed readiness manifest changed after runtime load")
    for source in authority_sources:
        _reject_symlink_chain(source.path)
        try:
            current = source.path.read_bytes()
        except OSError as exc:
            _fail(f"cannot reread {source.label} authority", exc)
        if current != source.content or _sha256(current) != source.sha256:
            _fail(f"{source.label} authority changed after runtime load")
    _assert_tree_shape(raw_root, manifest)
    if _walk_tree_digest(raw_root) != tree_sha256:
        _fail("screening runtime tree changed after runtime load")


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
    repository: Path,
    provenance: SystemProvenance,
) -> ScreeningRuntimeFold:
    child_dir = raw_root / child_manifest.run_id
    try:
        data_keys = build_screening_data_keys(config, provenance)
        data = load_screening_data_inventory(config, data_keys, child_dir)
        model_keys = build_screening_model_keys(config, data_keys, data.manifests)
        models = load_screening_model_inventory(
            config,
            data_keys,
            data,
            model_keys,
            child_dir,
        )
        shared = build_screening_shared_plan(config, data_keys, data.manifests, model_keys)
        expected_child = _child_manifest(
            config,
            data_keys,
            data,
            model_keys,
            models,
            shared,
            provenance,
        )
    except TrainingDataArtifactError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        _fail("screening runtime child inventory failed closed", exc)
    if expected_child != child_manifest:
        _fail("screening runtime child inventory does not match the readiness manifest")
    store = RunStore(
        raw_root,
        config,
        repository=repository,
        shared_artifacts=tuple(shared.artifacts),
    )
    for name, expected in (
        ("config.json", scientific_config_value(config)),
        ("expected-units.json", store.expected.model_dump(mode="json")),
        ("expected-shared-artifacts.json", store.expected_shared.model_dump(mode="json")),
        ("provenance.json", provenance.model_dump(mode="json")),
    ):
        content, value = _canonical_json_file(child_dir / name, label=name)
        if content != canonical_json_bytes(expected) + b"\n" or value != expected:
            _fail(f"screening runtime child {name} does not match its authority")
    return ScreeningRuntimeFold(
        family_id=child_manifest.heldout_family_id,
        config=config,
        store=store,
        data_keys=data_keys,
        data=data,
        model_keys=model_keys,
        models=models,
        shared_plan=shared,
    )


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
    if not repository_path.is_dir():
        _fail("screening runtime repository must be a directory")
    manifest_bytes, manifest = _manifest_bytes(committed, manifest_bytes_sha256)
    _assert_tree_shape(root, manifest)
    raw_manifest_bytes, _ = _canonical_json_file(
        root / "phase2-screening-readiness.json", label="raw-root manifest"
    )
    if raw_manifest_bytes != manifest_bytes:
        _fail("raw-root readiness manifest differs from the pinned committed manifest")
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
    if supplied_provenance is not None and provenance_identity_sha256(
        supplied_provenance
    ) != provenance_identity_sha256(captured_provenance):
        _fail("supplied provenance identity differs from the current captured provenance")
    if provenance_identity_sha256(captured_provenance) != manifest.provenance_sha256:
        _fail("screening runtime provenance identity differs from readiness authority")
    # Preserve the first-writer timestamp only after current policy/provenance
    # identity has been independently re-established.
    provenance_value = manifest.provenance
    folds = tuple(
        _load_fold(config, child, root, repository_path, provenance_value)
        for config, child in zip(configs, manifest.children, strict=True)
    )
    _assert_global_inventory(manifest, folds)
    stores = tuple(fold.store for fold in folds)
    for store in stores:
        store._execution_ready = False
    # Loading proves the immutable inventory but deliberately leaves execution
    # locked.  The caller must perform a fresh required recheck immediately
    # before execution to open all gates transactionally.
    tree_sha256 = _walk_tree_digest(root)
    _recheck_manifest_and_tree(
        committed,
        root,
        manifest_bytes,
        manifest,
        authority_sources,
        tree_sha256,
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
    )


__all__ = [
    "AuthoritySourceSnapshot",
    "ScreeningRuntime",
    "ScreeningRuntimeFold",
    "load_screening_runtime",
    "load_screening_data_inventory",
    "load_screening_model_inventory",
]
