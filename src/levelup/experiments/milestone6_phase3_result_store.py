"""Typed, development-only result-store partition for Milestone 6 Phase 3.

The Phase 3 logical plan is deliberately *not* represented by a synthetic
``ExperimentConfig``.  Its exposure hashes are part of the already-authorized
plan and were produced by the Phase 3 authority.  This module partitions an
opaque :class:`ValidatedPhase3Plan` into six result-store specifications and
can prepare their immutable, inert metadata namespaces.  It never calls
``plan_expected_units`` and cannot activate a store or publish a result.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from levelup.experiments.milestone6_phase3_model_authority import (
    Phase3ModelArtifactAuthority,
)
from levelup.experiments.milestone6_phase3_plan import (
    FAMILIES,
    PHASE,
    Phase3Plan,
    Phase3PlannedUnit,
    ValidatedPhase3Plan,
    _plan_body,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import AttemptRecord, UnitRecord

SCHEMA_VERSION = "milestone6.phase3.result-store-plan.v1"
EXPECTED_PHASE3_MODEL_AUTHORITY_SHA256 = (
    "8771eb52433faf15d6e5e935902a5c935526ec0e6b8e34621c3d6a922aea1a52"
)
EXPECTED_FAMILY_UNIT_COUNT = 1_920
EXPECTED_TOTAL_UNIT_COUNT = 11_520
_CONSTRUCTION_TOKEN = object()


class Phase3ResultStorePlanError(ValueError):
    """Raised when a result-store partition is not the frozen plan."""


class Phase3ResultStoreError(Phase3ResultStorePlanError):
    """Raised when a prepared result namespace is unsafe or inconsistent."""


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_hex(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise Phase3ResultStorePlanError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise Phase3ResultStorePlanError(f"{label} must be a SHA-256 hex digest") from exc
    return value


@dataclass(frozen=True, slots=True)
class Phase3ResultStoreSpec:
    """One family-owned, validation-only result namespace specification.

    ``units`` are copied directly from the validated logical plan.  They are
    not rebuilt from a config, and their exposure hashes are consequently
    independent of any later generic-config drift.
    """

    schema_version: str
    family_id: str
    phase: str
    plan_id: str
    protocol_sha256: str
    model_authority_sha256: str
    store_config_sha256: str
    run_id: str
    units: tuple[Phase3PlannedUnit, ...]
    unit_ids_sha256: str
    final_family_access: bool = False
    _construction_token: object | None = field(default=None, repr=False, compare=False)

    @property
    def expected_units(self) -> tuple[Phase3PlannedUnit, ...]:
        """Compatibility alias used by readiness/execution adapters."""

        return self.units

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(item.unit.unit_id for item in self.units)

    def __post_init__(self) -> None:
        if self._construction_token is not _CONSTRUCTION_TOKEN:
            raise Phase3ResultStorePlanError(
                "result-store specs require the canonical construction gate"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise Phase3ResultStorePlanError("result-store schema version drifted")
        if self.phase != PHASE:
            raise Phase3ResultStorePlanError("result stores must be validation-only")
        if self.family_id not in FAMILIES:
            raise Phase3ResultStorePlanError("result store contains an unknown family")
        if self.final_family_access:
            raise Phase3ResultStorePlanError("result store cannot include final-family access")
        _require_hex(self.plan_id, "plan_id")
        _require_hex(self.protocol_sha256, "protocol_sha256")
        _require_hex(self.model_authority_sha256, "model_authority_sha256")
        _require_hex(self.store_config_sha256, "store_config_sha256")
        _require_hex(self.run_id, "run_id")
        if len(self.units) != EXPECTED_FAMILY_UNIT_COUNT:
            raise Phase3ResultStorePlanError(
                f"family {self.family_id} must contain exactly {EXPECTED_FAMILY_UNIT_COUNT} units"
            )
        if any(type(item) is not Phase3PlannedUnit for item in self.units):
            raise Phase3ResultStorePlanError("result store contains untyped planned material")
        if any(
            item.heldout_family != self.family_id
            or item.unit.key.family_id != self.family_id
            or item.unit.key.phase != PHASE
            for item in self.units
        ):
            raise Phase3ResultStorePlanError("result store family or phase partition drifted")
        unit_ids = self.unit_ids
        if len(set(unit_ids)) != len(unit_ids):
            raise Phase3ResultStorePlanError("result store contains duplicate unit identities")
        key_ids = tuple(item.unit.key.model_dump_json() for item in self.units)
        if len(set(key_ids)) != len(key_ids):
            raise Phase3ResultStorePlanError("result store contains duplicate unit keys")
        if self.unit_ids_sha256 != _sha256_json(unit_ids):
            raise Phase3ResultStorePlanError("result store unit identity digest drifted")
        expected_config, expected_run = _store_hashes(
            family_id=self.family_id,
            plan_id=self.plan_id,
            protocol_sha256=self.protocol_sha256,
            model_authority_sha256=self.model_authority_sha256,
            unit_ids=unit_ids,
        )
        if (
            self.store_config_sha256 != expected_config
            or self.run_id != expected_run
        ):
            raise Phase3ResultStorePlanError("result store config or run digest drifted")


@dataclass(frozen=True, slots=True)
class Phase3ExpectedPlan:
    """Complete six-store expected matrix bound to frozen authorities."""

    schema_version: str
    plan_id: str
    protocol_sha256: str
    model_authority_sha256: str
    family_order: tuple[str, ...]
    stores: tuple[Phase3ResultStoreSpec, ...]
    final_family_access: bool = False
    _construction_token: object | None = field(default=None, repr=False, compare=False)

    @property
    def family_specs(self) -> tuple[Phase3ResultStoreSpec, ...]:
        return self.stores

    @property
    def units(self) -> tuple[Phase3PlannedUnit, ...]:
        return tuple(item for store in self.stores for item in store.units)

    @property
    def expected_units(self) -> tuple[Phase3PlannedUnit, ...]:
        return self.units

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(item.unit.unit_id for item in self.units)

    def store_for_family(self, family_id: str) -> Phase3ResultStoreSpec:
        for store in self.stores:
            if store.family_id == family_id:
                return store
        raise Phase3ResultStorePlanError(f"unknown Phase 3 result-store family: {family_id}")

    def __post_init__(self) -> None:
        if self._construction_token is not _CONSTRUCTION_TOKEN:
            raise Phase3ResultStorePlanError(
                "expected result plans require the canonical construction gate"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise Phase3ResultStorePlanError("result-store schema version drifted")
        if self.final_family_access:
            raise Phase3ResultStorePlanError("expected result plan cannot include final families")
        if self.family_order != FAMILIES:
            raise Phase3ResultStorePlanError("result-store family order drifted")
        _require_hex(self.plan_id, "plan_id")
        _require_hex(self.protocol_sha256, "protocol_sha256")
        _require_hex(self.model_authority_sha256, "model_authority_sha256")
        if len(self.stores) != len(FAMILIES):
            raise Phase3ResultStorePlanError("result plan must contain exactly six family stores")
        family_ids = tuple(store.family_id for store in self.stores)
        if family_ids != FAMILIES:
            raise Phase3ResultStorePlanError("result-store family partition is missing or extra")
        if any(
            store.plan_id != self.plan_id
            or store.protocol_sha256 != self.protocol_sha256
            or store.model_authority_sha256 != self.model_authority_sha256
            for store in self.stores
        ):
            raise Phase3ResultStorePlanError("result-store authority lineage drifted")
        unit_ids = self.unit_ids
        if len(unit_ids) != EXPECTED_TOTAL_UNIT_COUNT:
            raise Phase3ResultStorePlanError("result plan does not contain exactly 11,520 units")
        if len(set(unit_ids)) != EXPECTED_TOTAL_UNIT_COUNT:
            raise Phase3ResultStorePlanError("result plan contains duplicate or overlapping units")


def _validate_authorities(
    validated_plan: ValidatedPhase3Plan,
    authority: Phase3ModelArtifactAuthority,
) -> Phase3Plan:
    if type(validated_plan) is not ValidatedPhase3Plan:
        raise Phase3ResultStorePlanError("result plan requires the canonical validated Phase 3 plan")
    if type(authority) is not Phase3ModelArtifactAuthority:
        raise Phase3ResultStorePlanError("result plan requires the canonical model authority")
    plan = validated_plan.plan
    if type(plan) is not Phase3Plan:
        raise Phase3ResultStorePlanError("validated plan body is not canonical")
    try:
        # This also checks that the opaque construction token is still valid.
        validated_plan.require_unit(plan.units[0])
    except (IndexError, TypeError, ValueError) as exc:
        raise Phase3ResultStorePlanError("validated plan authority is not canonical") from exc
    if plan.final_family_access:
        raise Phase3ResultStorePlanError("Phase 3 result stores cannot include final families")
    if (
        plan.plan_id != authority.plan_id
        or plan.protocol_sha256 != authority.protocol_sha256
        or plan.family_order != FAMILIES
        or authority.family_order != FAMILIES
        or authority.authority_sha256 != EXPECTED_PHASE3_MODEL_AUTHORITY_SHA256
        or authority.expected_authority_sha256 != authority.authority_sha256
        or authority.development_only is not True
        or authority.final is not False
        or authority.final_family_accessed is not False
        or authority.execution_authorized is not True
    ):
        raise Phase3ResultStorePlanError("Phase 3 plan/model-authority lineage is not canonical")
    if len(plan.units) != EXPECTED_TOTAL_UNIT_COUNT:
        raise Phase3ResultStorePlanError("Phase 3 validated plan has the wrong unit count")
    mapping = [(item.unit.unit_id, item.model_owner_id) for item in plan.units]
    if (
        _sha256_json(_plan_body(plan)) != plan.plan_id
        or _sha256_json(mapping) != authority.unit_owner_mapping_sha256
        or tuple(sorted(owner.owner_id for owner in plan.model_owners))
        != authority.owner_ids
    ):
        raise Phase3ResultStorePlanError(
            "Phase 3 validated plan body or unit-owner mapping differs from the published authority"
        )
    if any(item.unit.key.phase != PHASE for item in plan.units):
        raise Phase3ResultStorePlanError("Phase 3 validated plan contains non-validation units")
    return plan


def _store_hashes(
    *,
    family_id: str,
    plan_id: str,
    protocol_sha256: str,
    model_authority_sha256: str,
    unit_ids: tuple[str, ...],
) -> tuple[str, str]:
    body = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "family_id": family_id,
        "plan_id": plan_id,
        "protocol_sha256": protocol_sha256,
        "model_authority_sha256": model_authority_sha256,
        "unit_ids": list(unit_ids),
    }
    return _sha256_json({"kind": "store-config", **body}), _sha256_json(
        {"kind": "run-id", **body}
    )


def build_phase3_expected_plan(
    validated_plan: ValidatedPhase3Plan,
    authority: Phase3ModelArtifactAuthority,
) -> Phase3ExpectedPlan:
    """Partition the canonical Phase 3 plan into six exact result stores.

    The function intentionally has no ``ExperimentConfig`` parameter and does
    not invoke generic expected-unit planning.  Every unit, including its
    exposure and seed channels, is copied from the opaque validated authority.
    """

    plan = _validate_authorities(validated_plan, authority)
    by_family: dict[str, list[Phase3PlannedUnit]] = {family: [] for family in FAMILIES}
    seen_ids: set[str] = set()
    for item in plan.units:
        family = item.heldout_family
        if family not in by_family:
            raise Phase3ResultStorePlanError("Phase 3 plan contains an extra family")
        if item.unit.unit_id in seen_ids:
            raise Phase3ResultStorePlanError("Phase 3 plan contains duplicate unit material")
        seen_ids.add(item.unit.unit_id)
        by_family[family].append(item)
    if set(by_family) != set(FAMILIES) or any(
        len(by_family[family]) != EXPECTED_FAMILY_UNIT_COUNT for family in FAMILIES
    ):
        raise Phase3ResultStorePlanError("Phase 3 family partition is incomplete")

    stores: list[Phase3ResultStoreSpec] = []
    for family in FAMILIES:
        units = tuple(by_family[family])
        unit_ids = tuple(item.unit.unit_id for item in units)
        config_sha256, run_id = _store_hashes(
            family_id=family,
            plan_id=plan.plan_id,
            protocol_sha256=plan.protocol_sha256,
            model_authority_sha256=authority.authority_sha256,
            unit_ids=unit_ids,
        )
        stores.append(
            Phase3ResultStoreSpec(
                schema_version=SCHEMA_VERSION,
                family_id=family,
                phase=PHASE,
                plan_id=plan.plan_id,
                protocol_sha256=plan.protocol_sha256,
                model_authority_sha256=authority.authority_sha256,
                store_config_sha256=config_sha256,
                run_id=run_id,
                units=units,
                unit_ids_sha256=_sha256_json(unit_ids),
                _construction_token=_CONSTRUCTION_TOKEN,
            )
        )
    return Phase3ExpectedPlan(
        schema_version=SCHEMA_VERSION,
        plan_id=plan.plan_id,
        protocol_sha256=plan.protocol_sha256,
        model_authority_sha256=authority.authority_sha256,
        family_order=FAMILIES,
        stores=tuple(stores),
        _construction_token=_CONSTRUCTION_TOKEN,
    )


def validate_phase3_expected_plan(
    value: Phase3ExpectedPlan,
    validated_plan: ValidatedPhase3Plan,
    authority: Phase3ModelArtifactAuthority,
) -> Phase3ExpectedPlan:
    """Require exact equality with a plan rebuilt from the two frozen authorities."""

    if type(value) is not Phase3ExpectedPlan:
        raise Phase3ResultStorePlanError("result plan is not the canonical typed value")
    canonical = build_phase3_expected_plan(validated_plan, authority)
    if value._construction_token is not _CONSTRUCTION_TOKEN or value != canonical:
        raise Phase3ResultStorePlanError("result plan differs from the canonical partition")
    return value


# ---------------------------------------------------------------------------
# Preparation-only persistence
# ---------------------------------------------------------------------------

_STORE_FILES = ("config.json", "expected-units.json", "run.json")
_ATTEMPT_RE = re.compile(r"^(?P<unit>[0-9a-f]{64})\.attempt-(?P<number>[0-9]{4})\.json$")


def _planned_unit_value(item: Phase3PlannedUnit) -> dict[str, Any]:
    """Render one opaque planned unit into canonical, JSON-compatible data."""

    return {
        "unit": item.unit.model_dump(mode="json"),
        "base_condition_id": item.base_condition_id,
        "tuple_id": item.tuple_id,
        "training_tuple_id": item.training_tuple_id,
        "fold_id": item.fold_id,
        "heldout_family": item.heldout_family,
        "model_owner_id": item.model_owner_id,
        "view_id": item.view_id,
    }


def _store_metadata(store: Phase3ResultStoreSpec) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = {
        "schema_version": SCHEMA_VERSION,
        "phase": store.phase,
        "family_id": store.family_id,
        "plan_id": store.plan_id,
        "protocol_sha256": store.protocol_sha256,
        "model_authority_sha256": store.model_authority_sha256,
        "config_sha256": store.store_config_sha256,
        "run_id": store.run_id,
        "unit_ids_sha256": store.unit_ids_sha256,
        "final_family_access": False,
    }
    expected = {
        "schema_version": SCHEMA_VERSION,
        "phase": store.phase,
        "family_id": store.family_id,
        "run_id": store.run_id,
        "config_sha256": store.store_config_sha256,
        "unit_ids_sha256": store.unit_ids_sha256,
        "units": [_planned_unit_value(item) for item in store.units],
    }
    run = {
        "schema_version": SCHEMA_VERSION,
        "phase": store.phase,
        "family_id": store.family_id,
        "run_id": store.run_id,
        "config_sha256": store.store_config_sha256,
        "plan_id": store.plan_id,
        "protocol_sha256": store.protocol_sha256,
        "model_authority_sha256": store.model_authority_sha256,
        "development_only": True,
        "final_family_access": False,
        "execution_ready": False,
    }
    return config, expected, run


def _canonical_bytes(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _stable_file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    if not stat.S_ISREG(value.st_mode):
        raise Phase3ResultStoreError("result record is not a regular file")
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_stable_json_at(directory_fd: int, name: str) -> Any:
    """Read once from one file descriptor and reject in-place mutation."""

    try:
        with secure_fs.open_regular_file_at(directory_fd, name) as file_fd:
            before = _stable_file_identity(os.fstat(file_fd))
            chunks: list[bytes] = []
            while True:
                chunk = os.read(file_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            content = b"".join(chunks)
            after = _stable_file_identity(os.fstat(file_fd))
        if before != after or len(content) != before[3]:
            raise Phase3ResultStoreError(f"result record changed during read: {name}")
        return json.loads(content)
    except Phase3ResultStoreError:
        raise
    except (OSError, secure_fs.SecureFilesystemError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase3ResultStoreError(f"cannot safely read result record: {name}") from exc


def _mkdir_child(parent_fd: int, name: str) -> int:
    try:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
        return secure_fs.open_child_directory(parent_fd, name)
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise Phase3ResultStoreError(f"cannot securely prepare directory: {name}") from exc


def _write_or_verify(fd: int, name: str, value: object) -> None:
    expected = _canonical_bytes(value)
    try:
        observed = secure_fs.read_bytes_at(fd, name)
    except secure_fs.SecureFilesystemError as exc:
        # Missing entries are the only entries that may be published.  The
        # secure primitive also rejects symlinks and non-regular files.
        cause = exc.__cause__
        if not isinstance(cause, FileNotFoundError):
            raise Phase3ResultStoreError(f"cannot read prepared metadata: {name}") from exc
        try:
            temp = f".{name}.{uuid.uuid4().hex}.tmp"
            temp_fd = os.open(
                temp,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600,
                dir_fd=fd,
            )
            try:
                with os.fdopen(temp_fd, "wb") as handle:
                    handle.write(expected)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.link(temp, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
                os.fsync(fd)
            finally:
                try:
                    os.unlink(temp, dir_fd=fd)
                except FileNotFoundError:
                    pass
            return
        except FileExistsError:
            # A concurrent publisher won; verify its complete canonical bytes.
            observed = secure_fs.read_bytes_at(fd, name)
        except (OSError, secure_fs.SecureFilesystemError) as write_exc:
            raise Phase3ResultStoreError(f"cannot publish prepared metadata: {name}") from write_exc
    if observed != expected:
        raise Phase3ResultStoreError(f"prepared metadata differs from canonical {name}")


def _validate_run_entries(run_fd: int) -> None:
    """Allow only canonical metadata files plus the two result directories."""

    allowed = set(_STORE_FILES) | {"units", "attempts"}
    try:
        with os.scandir(run_fd) as iterator:
            names: set[str] = set()
            for entry in iterator:
                if entry.name not in allowed or entry.is_symlink():
                    raise Phase3ResultStoreError("run directory contains extra or symlinked entries")
                if entry.name in {"units", "attempts"}:
                    if not entry.is_dir(follow_symlinks=False):
                        raise Phase3ResultStoreError("result namespace is not a directory")
                elif not entry.is_file(follow_symlinks=False):
                    raise Phase3ResultStoreError("run metadata is not a regular file")
                names.add(entry.name)
    except Phase3ResultStoreError:
        raise
    except OSError as exc:
        raise Phase3ResultStoreError("cannot enumerate prepared run directory") from exc
    if names != allowed:
        raise Phase3ResultStoreError("run directory contains extra or missing metadata")


def _verify_record_identity(
    record: UnitRecord | AttemptRecord,
    store: Phase3ResultStoreSpec,
    expected_by_id: dict[str, Phase3PlannedUnit],
    *,
    filename: str,
) -> None:
    expected_unit = expected_by_id.get(record.unit_id)
    if expected_unit is None:
        raise Phase3ResultStoreError(f"foreign result record: {filename}")
    expected_name = f"{record.unit_id}.json" if isinstance(record, UnitRecord) else (
        f"{record.unit_id}.attempt-{record.attempt:04d}.json"
    )
    if (
        filename != expected_name
        or record.run_id != store.run_id
        or record.config_sha256 != store.store_config_sha256
        or record.key != expected_unit.unit.key
        or record.seeds != expected_unit.unit.seeds
        or (
            isinstance(record, UnitRecord)
            and record.exposure_manifest_sha256 != expected_unit.unit.exposure_manifest_sha256
        )
    ):
        raise Phase3ResultStoreError(f"result record identity mismatch: {filename}")


@dataclass(frozen=True, slots=True)
class Phase3ResultStore:
    """One prepared, development-only family namespace.

    Preparation validates and publishes only immutable metadata.  The result
    namespace is deliberately inert: activation and all result writes belong
    to a later, private readiness transaction.
    """

    spec: Phase3ResultStoreSpec
    root: Path
    run_dir: Path
    root_identity: tuple[int, int]
    family_identity: tuple[int, int]
    run_identity: tuple[int, int]
    units_identity: tuple[int, int]
    attempts_identity: tuple[int, int]
    execution_ready: bool = False

    def __post_init__(self) -> None:
        if self.execution_ready is not False:
            raise Phase3ResultStoreError(
                "prepared Phase 3 result stores cannot be constructed execution-ready"
            )

    @property
    def family_id(self) -> str:
        return self.spec.family_id

    @property
    def run_id(self) -> str:
        return self.spec.run_id

    @property
    def config_sha256(self) -> str:
        return self.spec.store_config_sha256

    def _capture_current_identities(self) -> dict[str, tuple[int, int]]:
        stack = ExitStack()
        stack.__enter__()
        try:
            root_fd = secure_fs.open_directory_chain(self.root)
            stack.callback(os.close, root_fd)
            family_fd = secure_fs.open_child_directory(root_fd, self.family_id)
            stack.callback(os.close, family_fd)
            run_fd = secure_fs.open_child_directory(family_fd, self.run_id)
            stack.callback(os.close, run_fd)
            units_fd = secure_fs.open_child_directory(run_fd, "units")
            stack.callback(os.close, units_fd)
            attempts_fd = secure_fs.open_child_directory(run_fd, "attempts")
            stack.callback(os.close, attempts_fd)
            return {
                "root": secure_fs.directory_identity(root_fd),
                "family": secure_fs.directory_identity(family_fd),
                "run": secure_fs.directory_identity(run_fd),
                "units": secure_fs.directory_identity(units_fd),
                "attempts": secure_fs.directory_identity(attempts_fd),
            }
        except (OSError, secure_fs.SecureFilesystemError) as exc:
            raise Phase3ResultStoreError("cannot securely open prepared result store") from exc
        finally:
            stack.close()

    def _expected_identities(self) -> dict[str, tuple[int, int]]:
        return {
            "root": self.root_identity,
            "family": self.family_identity,
            "run": self.run_identity,
            "units": self.units_identity,
            "attempts": self.attempts_identity,
        }

    @contextmanager
    def _open_pinned(self) -> Iterator[dict[str, int]]:
        """Hold every result descriptor for one operation, then recheck paths."""

        expected = self._expected_identities()
        stack = ExitStack()
        stack.__enter__()
        descriptors: dict[str, int] = {}
        try:
            root_fd = secure_fs.open_directory_chain(self.root)
            stack.callback(os.close, root_fd)
            descriptors["root"] = root_fd
            family_fd = secure_fs.open_child_directory(root_fd, self.family_id)
            stack.callback(os.close, family_fd)
            descriptors["family"] = family_fd
            run_fd = secure_fs.open_child_directory(family_fd, self.run_id)
            stack.callback(os.close, run_fd)
            descriptors["run"] = run_fd
            units_fd = secure_fs.open_child_directory(run_fd, "units")
            stack.callback(os.close, units_fd)
            descriptors["units"] = units_fd
            attempts_fd = secure_fs.open_child_directory(run_fd, "attempts")
            stack.callback(os.close, attempts_fd)
            descriptors["attempts"] = attempts_fd
            observed = {
                name: secure_fs.directory_identity(fd) for name, fd in descriptors.items()
            }
            if observed != expected:
                raise Phase3ResultStoreError("prepared result directory identity changed")
            yield descriptors
        except Phase3ResultStoreError:
            raise
        except (OSError, secure_fs.SecureFilesystemError) as exc:
            raise Phase3ResultStoreError("cannot securely access prepared result store") from exc
        finally:
            stack.close()
            # All reads/writes above used the original pinned tree.  Reopening
            # only after those descriptors close detects a concurrent rename or
            # same-byte replacement without ever following the replacement.
            if self._capture_current_identities() != expected:
                raise Phase3ResultStoreError(
                    "prepared result directory identity changed during operation"
                )

    def _validated_records(
        self, units_fd: int, attempts_fd: int
    ) -> tuple[tuple[UnitRecord, ...], tuple[AttemptRecord, ...]]:
        """Open and validate every record exactly once, returning those objects."""

        expected_by_id = {item.unit.unit_id: item for item in self.spec.units}
        try:
            unit_entries = secure_fs.strict_regular_entries(units_fd)
            expected_names = {f"{unit_id}.json" for unit_id in expected_by_id}
            if not set(unit_entries) <= expected_names:
                raise Phase3ResultStoreError("units namespace contains extra or foreign records")
            units: list[UnitRecord] = []
            for name in unit_entries:
                try:
                    record = UnitRecord.model_validate(_read_stable_json_at(units_fd, name))
                except Exception as exc:
                    raise Phase3ResultStoreError(f"invalid unit record: {name}") from exc
                _verify_record_identity(record, self.spec, expected_by_id, filename=name)
                units.append(record)

            attempt_entries = secure_fs.strict_regular_entries(attempts_fd)
            attempts: list[AttemptRecord] = []
            for name in attempt_entries:
                match = _ATTEMPT_RE.fullmatch(name)
                if match is None or match.group("unit") not in expected_by_id:
                    raise Phase3ResultStoreError("attempts namespace contains extra or foreign records")
                try:
                    record = AttemptRecord.model_validate(
                        _read_stable_json_at(attempts_fd, name)
                    )
                except Exception as exc:
                    raise Phase3ResultStoreError(f"invalid attempt record: {name}") from exc
                _verify_record_identity(record, self.spec, expected_by_id, filename=name)
                attempts.append(record)
            by_unit_id = {record.unit_id: record for record in units}
            return (
                tuple(
                    by_unit_id[item.unit.unit_id]
                    for item in self.spec.units
                    if item.unit.unit_id in by_unit_id
                ),
                tuple(attempts),
            )
        except secure_fs.SecureFilesystemError as exc:
            raise Phase3ResultStoreError("unsafe result namespace") from exc

    def _validate_pinned(
        self, descriptors: dict[str, int]
    ) -> tuple[tuple[UnitRecord, ...], tuple[AttemptRecord, ...]]:
        config, expected, run = _store_metadata(self.spec)
        for name, value in zip(_STORE_FILES, (config, expected, run), strict=True):
            observed = secure_fs.read_bytes_at(descriptors["run"], name)
            if observed != _canonical_bytes(value):
                raise Phase3ResultStoreError(f"prepared metadata differs from canonical {name}")
        _validate_run_entries(descriptors["run"])
        return self._validated_records(descriptors["units"], descriptors["attempts"])

    def validate_resume(self) -> None:
        """Strictly validate an existing prepared namespace without writing."""

        with self._open_pinned() as descriptors:
            self._validate_pinned(descriptors)

    def completed_records(self) -> tuple[UnitRecord, ...]:
        with self._open_pinned() as descriptors:
            records, _ = self._validate_pinned(descriptors)
            return records

    def write_completed(self, *_args: Any, **_kwargs: Any) -> bool:
        raise Phase3ResultStoreError("prepared Phase 3 result stores are not execution-ready")

    def write_attempt(self, *_args: Any, **_kwargs: Any) -> None:
        raise Phase3ResultStoreError("prepared Phase 3 result stores are not execution-ready")


PreparedPhase3ResultStore = Phase3ResultStore


def _prepare_one_store(spec: Phase3ResultStoreSpec, output_root: Path) -> Phase3ResultStore:
    root = Path(os.path.abspath(output_root))
    # The caller must create the single output root explicitly.  Requiring an
    # existing root avoids a check-then-mkdir path race; every descendant is
    # created and opened relative to the pinned root descriptor below.
    try:
        with ExitStack() as stack:
            root_fd = secure_fs.open_directory_chain(root)
            stack.callback(os.close, root_fd)
            family_fd = _mkdir_child(root_fd, spec.family_id)
            stack.callback(os.close, family_fd)
            run_fd = _mkdir_child(family_fd, spec.run_id)
            stack.callback(os.close, run_fd)
            units_fd = _mkdir_child(run_fd, "units")
            stack.callback(os.close, units_fd)
            attempts_fd = _mkdir_child(run_fd, "attempts")
            stack.callback(os.close, attempts_fd)
            config, expected, run = _store_metadata(spec)
            for name, value in zip(_STORE_FILES, (config, expected, run), strict=True):
                _write_or_verify(run_fd, name, value)
            prepared = Phase3ResultStore(
                spec=spec,
                root=root,
                run_dir=root / spec.family_id / spec.run_id,
                root_identity=secure_fs.directory_identity(root_fd),
                family_identity=secure_fs.directory_identity(family_fd),
                run_identity=secure_fs.directory_identity(run_fd),
                units_identity=secure_fs.directory_identity(units_fd),
                attempts_identity=secure_fs.directory_identity(attempts_fd),
            )
            prepared._validate_pinned(
                {
                    "root": root_fd,
                    "family": family_fd,
                    "run": run_fd,
                    "units": units_fd,
                    "attempts": attempts_fd,
                }
            )
        prepared.validate_resume()
        return prepared
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise Phase3ResultStoreError("cannot prepare Phase 3 result namespace") from exc


def prepare_phase3_result_store(
    output_root: str | Path,
    validated_plan: ValidatedPhase3Plan,
    authority: Phase3ModelArtifactAuthority,
    *,
    family_id: str,
) -> Phase3ResultStore:
    expected = build_phase3_expected_plan(validated_plan, authority)
    try:
        spec = expected.store_for_family(family_id)
    except Phase3ResultStorePlanError:
        raise
    return _prepare_one_store(spec, Path(output_root))


def prepare_phase3_result_stores(
    output_root: str | Path,
    validated_plan: ValidatedPhase3Plan,
    authority: Phase3ModelArtifactAuthority,
) -> tuple[Phase3ResultStore, ...]:
    """Prepare all six family stores; never activates or executes them."""

    expected = build_phase3_expected_plan(validated_plan, authority)
    prepared: list[Phase3ResultStore] = []
    try:
        for spec in expected.stores:
            prepared.append(_prepare_one_store(spec, Path(output_root)))
        return tuple(prepared)
    except BaseException:
        # Preparation is idempotent and intentionally does not roll back a
        # previously prepared store; callers may inspect/resume the namespace.
        raise


def _load_one_store(spec: Phase3ResultStoreSpec, output_root: Path) -> Phase3ResultStore:
    """Load one already-published store without creating or modifying anything.

    Every descendant is opened relative to a pinned root descriptor with the
    secure, non-following primitives.  The metadata and record namespaces are
    validated while those descriptors are held, then the returned value is
    revalidated through its normal resume path after all descriptors close.
    Missing or substituted entries therefore fail closed and never trigger a
    mkdir/write fallback.
    """

    root = Path(os.path.abspath(output_root))
    try:
        with ExitStack() as stack:
            root_fd = secure_fs.open_directory_chain(root)
            stack.callback(os.close, root_fd)
            family_fd = secure_fs.open_child_directory(root_fd, spec.family_id)
            stack.callback(os.close, family_fd)
            run_fd = secure_fs.open_child_directory(family_fd, spec.run_id)
            stack.callback(os.close, run_fd)
            units_fd = secure_fs.open_child_directory(run_fd, "units")
            stack.callback(os.close, units_fd)
            attempts_fd = secure_fs.open_child_directory(run_fd, "attempts")
            stack.callback(os.close, attempts_fd)
            prepared = Phase3ResultStore(
                spec=spec,
                root=root,
                run_dir=root / spec.family_id / spec.run_id,
                root_identity=secure_fs.directory_identity(root_fd),
                family_identity=secure_fs.directory_identity(family_fd),
                run_identity=secure_fs.directory_identity(run_fd),
                units_identity=secure_fs.directory_identity(units_fd),
                attempts_identity=secure_fs.directory_identity(attempts_fd),
            )
            prepared._validate_pinned(
                {
                    "root": root_fd,
                    "family": family_fd,
                    "run": run_fd,
                    "units": units_fd,
                    "attempts": attempts_fd,
                }
            )
        prepared.validate_resume()
        return prepared
    except Phase3ResultStoreError:
        raise
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise Phase3ResultStoreError("cannot load prepared Phase 3 result namespace") from exc


def load_phase3_result_store(
    output_root: str | Path,
    validated_plan: ValidatedPhase3Plan,
    authority: Phase3ModelArtifactAuthority,
    *,
    family_id: str,
) -> Phase3ResultStore:
    """Load one existing, inert family store without filesystem mutation."""

    expected = build_phase3_expected_plan(validated_plan, authority)
    spec = expected.store_for_family(family_id)
    return _load_one_store(spec, Path(output_root))


def load_phase3_result_stores(
    output_root: str | Path,
    validated_plan: ValidatedPhase3Plan,
    authority: Phase3ModelArtifactAuthority,
) -> tuple[Phase3ResultStore, ...]:
    """Load all six existing, inert family stores without filesystem mutation.

    Unlike :func:`prepare_phase3_result_stores`, this function never creates
    directories or publishes metadata.  The complete canonical tree must
    already exist and pass strict descriptor-relative validation.
    """

    expected = build_phase3_expected_plan(validated_plan, authority)
    return tuple(
        _load_one_store(spec, Path(output_root)) for spec in expected.stores
    )


# A descriptive alias keeps call sites readable when they only need the six
# family store specifications.
build_phase3_result_store_plan = build_phase3_expected_plan


__all__ = [
    "EXPECTED_FAMILY_UNIT_COUNT",
    "EXPECTED_PHASE3_MODEL_AUTHORITY_SHA256",
    "EXPECTED_TOTAL_UNIT_COUNT",
    "Phase3ResultStore",
    "PreparedPhase3ResultStore",
    "Phase3ExpectedPlan",
    "Phase3ResultStoreError",
    "Phase3ResultStorePlanError",
    "Phase3ResultStoreSpec",
    "SCHEMA_VERSION",
    "build_phase3_expected_plan",
    "build_phase3_result_store_plan",
    "load_phase3_result_store",
    "load_phase3_result_stores",
    "prepare_phase3_result_store",
    "prepare_phase3_result_stores",
    "validate_phase3_expected_plan",
]
