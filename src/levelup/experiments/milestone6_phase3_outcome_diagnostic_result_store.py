"""Inert, descriptor-held result namespaces for the Phase 3 outcome diagnostic.

This module is deliberately a preparation boundary.  It partitions the opaque,
validated outcome-diagnostic plan into six family stores and publishes only
canonical metadata under the empty output descriptor held by readiness.  No
``UnitRecord`` (or any other outcome) is created here, and there is no path-only
loader.  A later executor can consume the returned identities and the same
readiness lease without reopening a path.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    OutcomePlannedUnit,
    ValidatedOutcomePlan,
    outcome_plan_id,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    CONDITIONS,
    FAMILIES,
    OutcomeDiagnosticProtocolSnapshot,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_readiness import (
    OutcomeDiagnosticActivationReadinessLease,
    OutcomeDiagnosticReadinessError,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes

SCHEMA_VERSION = "milestone6.phase3.outcome-group-diagnostic.result-store.v1"
ACTIVATION_INTENT_NAME = "outcome-diagnostic-activation-intent.json"
ROOT_METADATA_NAME = "outcome-diagnostic-root.json"
EXPECTED_FAMILY_UNIT_COUNT = 960
EXPECTED_NAMESPACE_UNIT_COUNT = 480
EXPECTED_TOTAL_UNIT_COUNT = 5760
_CONSTRUCTION_TOKEN = object()


class OutcomeDiagnosticResultStoreError(ValueError):
    """Raised when a prepared diagnostic namespace is unsafe or inconsistent."""


class OutcomeDiagnosticResultStorePlanError(OutcomeDiagnosticResultStoreError):
    """Raised when the typed diagnostic plan cannot form the exact store matrix."""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _canonical(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    if not stat.S_ISREG(value.st_mode):
        raise OutcomeDiagnosticResultStoreError("diagnostic metadata is not a regular file")
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_stable_bytes_at(directory_fd: int, name: str) -> bytes:
    """Read metadata through one descriptor and reject replacement races."""

    try:
        with secure_fs.open_regular_file_at(directory_fd, name) as file_fd:
            before = _file_identity(os.fstat(file_fd))
            path_before = _file_identity(os.stat(name, dir_fd=directory_fd, follow_symlinks=False))
            if before != path_before:
                raise OutcomeDiagnosticResultStoreError("diagnostic metadata identity changed")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(file_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            content = b"".join(chunks)
            after = _file_identity(os.fstat(file_fd))
            path_after = _file_identity(os.stat(name, dir_fd=directory_fd, follow_symlinks=False))
            if before != after or after != path_after or len(content) != after[3]:
                raise OutcomeDiagnosticResultStoreError(
                    "diagnostic metadata changed while being read"
                )
            return content
    except OutcomeDiagnosticResultStoreError:
        raise
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise OutcomeDiagnosticResultStoreError(f"cannot safely read metadata: {name}") from exc


def _is_missing(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, FileNotFoundError):
            return True
        current = current.__cause__
    return False


def _identity(fd: int) -> tuple[int, int]:
    try:
        value = os.fstat(fd)
    except OSError as exc:
        raise OutcomeDiagnosticResultStoreError("cannot stat held result descriptor") from exc
    if not stat.S_ISDIR(value.st_mode):
        raise OutcomeDiagnosticResultStoreError("result descriptor is not a directory")
    return int(value.st_dev), int(value.st_ino)


def _jsonable(value: object) -> object:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _jsonable(getattr(value, name))
            for name in value.__dataclass_fields__
            if not name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _unit_body(unit: OutcomePlannedUnit) -> dict[str, Any]:
    return _jsonable(unit)  # type: ignore[return-value]


def _unit_ids(units: tuple[OutcomePlannedUnit, ...]) -> tuple[str, ...]:
    return tuple(unit.unit_id for unit in units)


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise OutcomeDiagnosticResultStorePlanError(f"{label} is not a SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticNamespaceSpec:
    schema_version: str
    condition_id: str
    namespace_id: str
    units: tuple[OutcomePlannedUnit, ...]
    unit_ids_sha256: str
    _construction_token: object | None = field(default=None, repr=False, compare=False)

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return _unit_ids(self.units)

    def __post_init__(self) -> None:
        if self._construction_token is not _CONSTRUCTION_TOKEN:
            raise OutcomeDiagnosticResultStorePlanError(
                "namespace specs require the canonical construction gate"
            )
        if self.schema_version != SCHEMA_VERSION or self.condition_id not in CONDITIONS:
            raise OutcomeDiagnosticResultStorePlanError("diagnostic namespace schema drifted")
        _require_digest(self.namespace_id, "namespace_id")
        if len(self.units) != EXPECTED_NAMESPACE_UNIT_COUNT:
            raise OutcomeDiagnosticResultStorePlanError(
                "diagnostic namespace must contain 480 units"
            )
        if any(unit.condition_id != self.condition_id for unit in self.units):
            raise OutcomeDiagnosticResultStorePlanError("namespace condition partition drifted")
        if len(set(self.unit_ids)) != len(self.unit_ids):
            raise OutcomeDiagnosticResultStorePlanError("namespace contains duplicate units")
        if self.unit_ids_sha256 != _sha(list(self.unit_ids)):
            raise OutcomeDiagnosticResultStorePlanError("namespace unit digest drifted")


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticResultStoreSpec:
    schema_version: str
    family_id: str
    plan_id: str
    protocol_sha256: str
    run_id: str
    config_sha256: str
    unit_ids_sha256: str
    units: tuple[OutcomePlannedUnit, ...]
    namespaces: tuple[OutcomeDiagnosticNamespaceSpec, ...]
    final_family_access: bool = False
    _construction_token: object | None = field(default=None, repr=False, compare=False)

    @property
    def expected_units(self) -> tuple[OutcomePlannedUnit, ...]:
        return self.units

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return _unit_ids(self.units)

    def namespace_for_condition(self, condition_id: str) -> OutcomeDiagnosticNamespaceSpec:
        for namespace in self.namespaces:
            if namespace.condition_id == condition_id:
                return namespace
        raise OutcomeDiagnosticResultStorePlanError(f"unknown diagnostic condition: {condition_id}")

    def __post_init__(self) -> None:
        if self._construction_token is not _CONSTRUCTION_TOKEN:
            raise OutcomeDiagnosticResultStorePlanError(
                "result-store specs require the canonical construction gate"
            )
        if self.schema_version != SCHEMA_VERSION or self.family_id not in FAMILIES:
            raise OutcomeDiagnosticResultStorePlanError("diagnostic result-store schema drifted")
        if self.final_family_access:
            raise OutcomeDiagnosticResultStorePlanError(
                "diagnostic store cannot access final families"
            )
        for label, value in (
            ("plan_id", self.plan_id),
            ("protocol_sha256", self.protocol_sha256),
            ("run_id", self.run_id),
            ("config_sha256", self.config_sha256),
            ("unit_ids_sha256", self.unit_ids_sha256),
        ):
            _require_digest(value, label)
        if len(self.units) != EXPECTED_FAMILY_UNIT_COUNT:
            raise OutcomeDiagnosticResultStorePlanError("family store must contain 960 units")
        if len(set(self.unit_ids)) != len(self.unit_ids):
            raise OutcomeDiagnosticResultStorePlanError("family store contains duplicate units")
        if any(unit.heldout_family != self.family_id for unit in self.units):
            raise OutcomeDiagnosticResultStorePlanError("family store partition drifted")
        if tuple(namespace.condition_id for namespace in self.namespaces) != CONDITIONS:
            raise OutcomeDiagnosticResultStorePlanError("namespace condition order drifted")
        if tuple(namespace.unit_ids for namespace in self.namespaces) != tuple(
            tuple(unit.unit_id for unit in self.units if unit.condition_id == condition)
            for condition in CONDITIONS
        ):
            raise OutcomeDiagnosticResultStorePlanError(
                "namespace units do not partition family units"
            )
        if self.unit_ids_sha256 != _sha(list(self.unit_ids)):
            raise OutcomeDiagnosticResultStorePlanError("family unit digest drifted")


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticExpectedPlan:
    schema_version: str
    plan_id: str
    protocol_sha256: str
    family_order: tuple[str, ...]
    condition_order: tuple[str, ...]
    stores: tuple[OutcomeDiagnosticResultStoreSpec, ...]
    final_family_access: bool = False
    _construction_token: object | None = field(default=None, repr=False, compare=False)

    @property
    def family_specs(self) -> tuple[OutcomeDiagnosticResultStoreSpec, ...]:
        return self.stores

    @property
    def units(self) -> tuple[OutcomePlannedUnit, ...]:
        return tuple(unit for store in self.stores for unit in store.units)

    @property
    def expected_units(self) -> tuple[OutcomePlannedUnit, ...]:
        return self.units

    def store_for_family(self, family_id: str) -> OutcomeDiagnosticResultStoreSpec:
        for store in self.stores:
            if store.family_id == family_id:
                return store
        raise OutcomeDiagnosticResultStorePlanError(f"unknown diagnostic family: {family_id}")

    def __post_init__(self) -> None:
        if self._construction_token is not _CONSTRUCTION_TOKEN:
            raise OutcomeDiagnosticResultStorePlanError(
                "expected plan requires canonical construction"
            )
        if self.schema_version != SCHEMA_VERSION or self.family_order != FAMILIES:
            raise OutcomeDiagnosticResultStorePlanError("diagnostic family order drifted")
        if self.condition_order != CONDITIONS or self.final_family_access:
            raise OutcomeDiagnosticResultStorePlanError(
                "diagnostic expected plan is not development-only"
            )
        _require_digest(self.plan_id, "plan_id")
        _require_digest(self.protocol_sha256, "protocol_sha256")
        if tuple(store.family_id for store in self.stores) != FAMILIES:
            raise OutcomeDiagnosticResultStorePlanError(
                "diagnostic store family partition is incomplete"
            )
        if any(
            store.plan_id != self.plan_id or store.protocol_sha256 != self.protocol_sha256
            for store in self.stores
        ):
            raise OutcomeDiagnosticResultStorePlanError(
                "diagnostic store authority lineage drifted"
            )
        if (
            len(self.units) != EXPECTED_TOTAL_UNIT_COUNT
            or len(set(unit.unit_id for unit in self.units)) != EXPECTED_TOTAL_UNIT_COUNT
        ):
            raise OutcomeDiagnosticResultStorePlanError(
                "diagnostic expected plan must contain 5,760 unique units"
            )


def _store_hashes(
    family_id: str, plan_id: str, protocol_sha256: str, unit_ids: tuple[str, ...]
) -> tuple[str, str]:
    body = {
        "schema_version": SCHEMA_VERSION,
        "family_id": family_id,
        "plan_id": plan_id,
        "protocol_sha256": protocol_sha256,
        "unit_ids": list(unit_ids),
    }
    return _sha({"kind": "diagnostic-store-config", **body}), _sha(
        {"kind": "diagnostic-run-id", **body}
    )


def _namespace_id(
    condition_id: str, family_id: str, plan_id: str, unit_ids: tuple[str, ...]
) -> str:
    return _sha(
        {
            "kind": "diagnostic-namespace",
            "condition_id": condition_id,
            "family_id": family_id,
            "plan_id": plan_id,
            "unit_ids": list(unit_ids),
        }
    )


def build_outcome_diagnostic_expected_plan(
    validated_plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
) -> OutcomeDiagnosticExpectedPlan:
    """Build the exact six-family/2-condition development matrix from opaque authorities."""

    if type(validated_plan) is not ValidatedOutcomePlan:
        raise OutcomeDiagnosticResultStorePlanError(
            "expected plan requires the canonical validated outcome plan"
        )
    if type(snapshot) is not OutcomeDiagnosticProtocolSnapshot:
        raise OutcomeDiagnosticResultStorePlanError(
            "expected plan requires the typed protocol snapshot"
        )
    plan = validated_plan.plan
    try:
        parsed_protocol = json.loads(snapshot.content)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OutcomeDiagnosticResultStorePlanError("pinned protocol bytes are invalid") from exc
    if (
        _jsonable(snapshot.payload) != parsed_protocol
        or hashlib.sha256(snapshot.content).hexdigest() != snapshot.sha256
        or plan.protocol_sha256 != snapshot.sha256
        or outcome_plan_id(plan) != plan.plan_id
    ):
        raise OutcomeDiagnosticResultStorePlanError(
            "validated outcome plan differs from pinned authority"
        )
    try:
        for unit in plan.units:
            validated_plan.require_unit(unit)
    except (TypeError, ValueError) as exc:
        raise OutcomeDiagnosticResultStorePlanError(
            "validated outcome plan token or unit matrix drifted"
        ) from exc
    if (
        plan.final_family_access
        or plan.family_order != FAMILIES
        or plan.condition_ids != CONDITIONS
    ):
        raise OutcomeDiagnosticResultStorePlanError(
            "outcome diagnostic plan is not development-only"
        )
    stores: list[OutcomeDiagnosticResultStoreSpec] = []
    for family in FAMILIES:
        family_units = tuple(unit for unit in plan.units if unit.heldout_family == family)
        if len(family_units) != EXPECTED_FAMILY_UNIT_COUNT:
            raise OutcomeDiagnosticResultStorePlanError(
                f"family {family} does not contain 960 units"
            )
        namespaces: list[OutcomeDiagnosticNamespaceSpec] = []
        for condition in CONDITIONS:
            units = tuple(unit for unit in family_units if unit.condition_id == condition)
            ids = _unit_ids(units)
            namespaces.append(
                OutcomeDiagnosticNamespaceSpec(
                    SCHEMA_VERSION,
                    condition,
                    _namespace_id(condition, family, plan.plan_id, ids),
                    units,
                    _sha(list(ids)),
                    _construction_token=_CONSTRUCTION_TOKEN,
                )
            )
        ids = _unit_ids(family_units)
        config_sha, run_id = _store_hashes(family, plan.plan_id, plan.protocol_sha256, ids)
        stores.append(
            OutcomeDiagnosticResultStoreSpec(
                SCHEMA_VERSION,
                family,
                plan.plan_id,
                plan.protocol_sha256,
                run_id,
                config_sha,
                _sha(list(ids)),
                family_units,
                tuple(namespaces),
                False,
                _CONSTRUCTION_TOKEN,
            )
        )
    return OutcomeDiagnosticExpectedPlan(
        SCHEMA_VERSION,
        plan.plan_id,
        plan.protocol_sha256,
        FAMILIES,
        CONDITIONS,
        tuple(stores),
        False,
        _CONSTRUCTION_TOKEN,
    )


def validate_outcome_diagnostic_expected_plan(
    value: OutcomeDiagnosticExpectedPlan,
    validated_plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
) -> OutcomeDiagnosticExpectedPlan:
    if type(value) is not OutcomeDiagnosticExpectedPlan:
        raise OutcomeDiagnosticResultStorePlanError("expected plan is not canonical typed material")
    expected = build_outcome_diagnostic_expected_plan(validated_plan, snapshot)
    if value != expected:
        raise OutcomeDiagnosticResultStorePlanError(
            "expected diagnostic plan differs from authority"
        )
    return value


def _write_or_verify(directory_fd: int, name: str, value: object) -> None:
    expected = _canonical(value)
    try:
        observed = _read_stable_bytes_at(directory_fd, name)
    except OutcomeDiagnosticResultStoreError as exc:
        if not _is_missing(exc):
            raise
        temporary = f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            fd = os.open(
                temporary,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(expected)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(
                temporary,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            os.fsync(directory_fd)
            return
        except FileExistsError:
            observed = _read_stable_bytes_at(directory_fd, name)
        except (OSError, secure_fs.SecureFilesystemError) as write_exc:
            raise OutcomeDiagnosticResultStoreError(
                f"cannot publish metadata: {name}"
            ) from write_exc
        finally:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
    if observed != expected:
        raise OutcomeDiagnosticResultStoreError(f"metadata differs from canonical: {name}")


def _mkdir(parent_fd: int, name: str) -> int:
    try:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
        return secure_fs.open_child_directory(parent_fd, name)
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise OutcomeDiagnosticResultStoreError(f"cannot securely open directory: {name}") from exc


def _entry_kinds(fd: int) -> dict[str, str]:
    try:
        with os.scandir(fd) as iterator:
            result: dict[str, str] = {}
            for entry in iterator:
                if entry.is_symlink():
                    raise OutcomeDiagnosticResultStoreError(
                        "diagnostic namespace contains a symlink"
                    )
                if entry.is_dir(follow_symlinks=False):
                    result[entry.name] = "dir"
                elif entry.is_file(follow_symlinks=False):
                    result[entry.name] = "file"
                else:
                    raise OutcomeDiagnosticResultStoreError(
                        "diagnostic namespace contains a nonregular entry"
                    )
            return result
    except OutcomeDiagnosticResultStoreError:
        raise
    except OSError as exc:
        raise OutcomeDiagnosticResultStoreError("cannot enumerate diagnostic namespace") from exc


def _metadata(
    spec: OutcomeDiagnosticResultStoreSpec, identities: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = {
        "schema_version": SCHEMA_VERSION,
        "family_id": spec.family_id,
        "plan_id": spec.plan_id,
        "protocol_sha256": spec.protocol_sha256,
        "run_id": spec.run_id,
        "config_sha256": spec.config_sha256,
        "unit_ids_sha256": spec.unit_ids_sha256,
        "condition_ids": list(CONDITIONS),
        "final_family_access": False,
    }
    expected = {
        "schema_version": SCHEMA_VERSION,
        "family_id": spec.family_id,
        "run_id": spec.run_id,
        "plan_id": spec.plan_id,
        "unit_ids_sha256": spec.unit_ids_sha256,
        "units": [_unit_body(unit) for unit in spec.units],
    }
    run = {
        "schema_version": SCHEMA_VERSION,
        "family_id": spec.family_id,
        "run_id": spec.run_id,
        "plan_id": spec.plan_id,
        "protocol_sha256": spec.protocol_sha256,
        "config_sha256": spec.config_sha256,
        "development_only": True,
        "final_family_access": False,
        "execution_ready": False,
        "identities": _jsonable(identities),
    }
    intent = {
        "schema_version": f"{SCHEMA_VERSION}.activation-intent.v1",
        "marker": "prepared-inert-outcome-diagnostic",
        "family_id": spec.family_id,
        "run_id": spec.run_id,
        "plan_id": spec.plan_id,
        "protocol_sha256": spec.protocol_sha256,
        "unit_ids_sha256": spec.unit_ids_sha256,
        "expected_unit_count": EXPECTED_FAMILY_UNIT_COUNT,
        "condition_ids": list(CONDITIONS),
        "execution_ready": False,
        "final_family_access": False,
        "identities": _jsonable(identities),
    }
    namespace_meta = {
        condition.condition_id: {
            "schema_version": SCHEMA_VERSION,
            "condition_id": condition.condition_id,
            "namespace_id": condition.namespace_id,
            "family_id": spec.family_id,
            "run_id": spec.run_id,
            "unit_ids_sha256": condition.unit_ids_sha256,
            "expected_unit_count": EXPECTED_NAMESPACE_UNIT_COUNT,
            "record_namespace": "records",
            "execution_ready": False,
            "final_family_access": False,
        }
        for condition in spec.namespaces
    }
    return config, expected, run, {"intent": intent, "namespace": namespace_meta}


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticResultStore:
    spec: OutcomeDiagnosticResultStoreSpec
    root: Path
    root_identity: tuple[int, int]
    family_identity: tuple[int, int]
    run_identity: tuple[int, int]
    namespaces_parent_identity: tuple[int, int]
    namespace_identities: Mapping[str, tuple[int, int]]
    record_namespace_identities: Mapping[str, tuple[int, int]]
    execution_ready: bool = False
    _construction_token: object | None = field(default=None, repr=False, compare=False)

    @property
    def family_id(self) -> str:
        return self.spec.family_id

    @property
    def run_id(self) -> str:
        return self.spec.run_id

    @property
    def config_sha256(self) -> str:
        return self.spec.config_sha256

    def __post_init__(self) -> None:
        if self.execution_ready:
            raise OutcomeDiagnosticResultStoreError(
                "diagnostic store preparation cannot be execution-ready"
            )
        if self._construction_token is not _CONSTRUCTION_TOKEN:
            raise OutcomeDiagnosticResultStoreError(
                "diagnostic stores require canonical preparation"
            )
        if type(self.spec) is not OutcomeDiagnosticResultStoreSpec:
            raise OutcomeDiagnosticResultStoreError("diagnostic store spec is not canonical")
        identities = (
            self.root_identity,
            self.family_identity,
            self.run_identity,
            self.namespaces_parent_identity,
            *self.namespace_identities.values(),
            *self.record_namespace_identities.values(),
        )
        if any(
            type(identity) is not tuple
            or len(identity) != 2
            or any(type(part) is not int or part < 0 for part in identity)
            for identity in identities
        ):
            raise OutcomeDiagnosticResultStoreError("diagnostic store identities are malformed")
        if (
            tuple(self.namespace_identities) != CONDITIONS
            or tuple(self.record_namespace_identities) != CONDITIONS
        ):
            raise OutcomeDiagnosticResultStoreError(
                "diagnostic namespace identities are incomplete"
            )
        object.__setattr__(
            self, "namespace_identities", MappingProxyType(dict(self.namespace_identities))
        )
        object.__setattr__(
            self,
            "record_namespace_identities",
            MappingProxyType(dict(self.record_namespace_identities)),
        )

    def _verify_tree(self, root_fd: int) -> None:
        with ExitStack() as stack:
            if _identity(root_fd) != self.root_identity:
                raise OutcomeDiagnosticResultStoreError("diagnostic output root identity changed")
            family_fd = secure_fs.open_child_directory(root_fd, self.family_id)
            stack.callback(os.close, family_fd)
            if _identity(family_fd) != self.family_identity:
                raise OutcomeDiagnosticResultStoreError("diagnostic family identity changed")
            run_fd = secure_fs.open_child_directory(family_fd, self.run_id)
            stack.callback(os.close, run_fd)
            if _identity(run_fd) != self.run_identity:
                raise OutcomeDiagnosticResultStoreError("diagnostic run identity changed")
            namespaces_fd = secure_fs.open_child_directory(run_fd, "namespaces")
            stack.callback(os.close, namespaces_fd)
            if _identity(namespaces_fd) != self.namespaces_parent_identity:
                raise OutcomeDiagnosticResultStoreError(
                    "diagnostic namespaces parent identity changed"
                )
            for condition in CONDITIONS:
                namespace_fd = secure_fs.open_child_directory(namespaces_fd, condition)
                stack.callback(os.close, namespace_fd)
                if _identity(namespace_fd) != self.namespace_identities[condition]:
                    raise OutcomeDiagnosticResultStoreError("diagnostic namespace identity changed")
                records_fd = secure_fs.open_child_directory(namespace_fd, "records")
                stack.callback(os.close, records_fd)
                if _identity(records_fd) != self.record_namespace_identities[condition]:
                    raise OutcomeDiagnosticResultStoreError("diagnostic records identity changed")

    @contextmanager
    def open_pinned(self, lease: OutcomeDiagnosticActivationReadinessLease):
        """Yield fd-relative descriptors for later execution under a live lease."""

        active = _validate_lease(lease, lease.snapshot.protocol)
        _check_output_root_path(active)
        stack = ExitStack()
        stack.__enter__()
        try:
            root_fd = os.dup(active.output_root_fd)
            stack.callback(os.close, root_fd)
            if _identity(root_fd) != self.root_identity:
                raise OutcomeDiagnosticResultStoreError("diagnostic output root identity changed")
            family_fd = secure_fs.open_child_directory(root_fd, self.family_id)
            stack.callback(os.close, family_fd)
            if _identity(family_fd) != self.family_identity:
                raise OutcomeDiagnosticResultStoreError("diagnostic family identity changed")
            run_fd = secure_fs.open_child_directory(family_fd, self.run_id)
            stack.callback(os.close, run_fd)
            if _identity(run_fd) != self.run_identity:
                raise OutcomeDiagnosticResultStoreError("diagnostic run identity changed")
            namespaces_fd = secure_fs.open_child_directory(run_fd, "namespaces")
            stack.callback(os.close, namespaces_fd)
            if _identity(namespaces_fd) != self.namespaces_parent_identity:
                raise OutcomeDiagnosticResultStoreError(
                    "diagnostic namespaces parent identity changed"
                )
            descriptors: dict[str, int] = {"root": root_fd, "family": family_fd, "run": run_fd}
            for condition in CONDITIONS:
                namespace_fd = secure_fs.open_child_directory(namespaces_fd, condition)
                stack.callback(os.close, namespace_fd)
                if _identity(namespace_fd) != self.namespace_identities[condition]:
                    raise OutcomeDiagnosticResultStoreError("diagnostic namespace identity changed")
                records_fd = secure_fs.open_child_directory(namespace_fd, "records")
                stack.callback(os.close, records_fd)
                if _identity(records_fd) != self.record_namespace_identities[condition]:
                    raise OutcomeDiagnosticResultStoreError("diagnostic records identity changed")
                descriptors[f"namespace:{condition}"] = namespace_fd
                descriptors[f"records:{condition}"] = records_fd
            active.require_active()
            yield descriptors
        finally:
            try:
                if "root_fd" in locals():
                    self._verify_tree(root_fd)
            finally:
                stack.close()
            if not active.active:
                raise OutcomeDiagnosticResultStoreError(
                    "diagnostic readiness lease closed during pinned store use"
                )
            active.require_active()
            _check_output_root_path(active)


def _validate_lease(
    lease: object, snapshot: OutcomeDiagnosticProtocolSnapshot
) -> OutcomeDiagnosticActivationReadinessLease:
    if type(lease) is not OutcomeDiagnosticActivationReadinessLease:
        raise OutcomeDiagnosticResultStoreError(
            "diagnostic store operation requires the canonical readiness lease"
        )
    try:
        active = lease.require_active()
    except (OutcomeDiagnosticReadinessError, OSError, ValueError) as exc:
        raise OutcomeDiagnosticResultStoreError("diagnostic readiness lease is not active") from exc
    if active.snapshot.protocol is not snapshot:
        raise OutcomeDiagnosticResultStoreError(
            "diagnostic protocol snapshot is not the pinned lease snapshot"
        )
    return active


def _check_output_root_path(lease: OutcomeDiagnosticActivationReadinessLease) -> None:
    """Detect replacement of the canonical output path while its fd is held."""

    try:
        observed = os.stat(lease.snapshot.output_root, follow_symlinks=False)
    except OSError as exc:
        raise OutcomeDiagnosticResultStoreError("diagnostic output root path changed") from exc
    if not stat.S_ISDIR(observed.st_mode):
        raise OutcomeDiagnosticResultStoreError("diagnostic output root path is not a directory")
    if (int(observed.st_dev), int(observed.st_ino)) != _identity(lease.output_root_fd):
        raise OutcomeDiagnosticResultStoreError("diagnostic output root path identity changed")


def _root_metadata(
    expected: OutcomeDiagnosticExpectedPlan, root_identity: tuple[int, int]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "marker": "prepared-inert-outcome-diagnostic-root",
        "plan_id": expected.plan_id,
        "protocol_sha256": expected.protocol_sha256,
        "family_order": list(FAMILIES),
        "run_ids": [store.run_id for store in expected.stores],
        "expected_unit_count": EXPECTED_TOTAL_UNIT_COUNT,
        "final_family_access": False,
        "root_identity": list(root_identity),
    }


def _check_root(
    root_fd: int, expected: OutcomeDiagnosticExpectedPlan, *, require_root_meta: bool
) -> tuple[int, ...]:
    root_identity = _identity(root_fd)
    entries = _entry_kinds(root_fd)
    allowed = set(FAMILIES) | {ROOT_METADATA_NAME}
    if set(entries) != allowed:
        raise OutcomeDiagnosticResultStoreError(
            "diagnostic output root has partial, extra, or historical entries"
        )
    if (
        any(entries[family] != "dir" for family in FAMILIES)
        or entries[ROOT_METADATA_NAME] != "file"
    ):
        raise OutcomeDiagnosticResultStoreError("diagnostic output root layout is not canonical")
    if require_root_meta:
        observed = _read_stable_bytes_at(root_fd, ROOT_METADATA_NAME)
        if observed != _canonical(_root_metadata(expected, root_identity)):
            raise OutcomeDiagnosticResultStoreError(
                "diagnostic root metadata differs from canonical authority"
            )
    return root_identity


def _prepare_or_load_one(
    root_fd: int,
    expected: OutcomeDiagnosticResultStoreSpec,
    *,
    prepare: bool,
    root_path: Path,
) -> OutcomeDiagnosticResultStore:
    stack = ExitStack()
    try:
        family_fd = (
            _mkdir(root_fd, expected.family_id)
            if prepare
            else secure_fs.open_child_directory(root_fd, expected.family_id)
        )
        stack.callback(os.close, family_fd)
        family_identity = _identity(family_fd)
        family_entries = _entry_kinds(family_fd)
        if prepare and family_entries and set(family_entries) != {"family.json", expected.run_id}:
            raise OutcomeDiagnosticResultStoreError(
                "family namespace contains partial or extra entries"
            )
        if not prepare and set(family_entries) != {"family.json", expected.run_id}:
            raise OutcomeDiagnosticResultStoreError("family namespace is incomplete or foreign")
        run_fd = (
            _mkdir(family_fd, expected.run_id)
            if prepare
            else secure_fs.open_child_directory(family_fd, expected.run_id)
        )
        stack.callback(os.close, run_fd)
        run_identity = _identity(run_fd)
        namespaces_fd = (
            _mkdir(run_fd, "namespaces")
            if prepare
            else secure_fs.open_child_directory(run_fd, "namespaces")
        )
        stack.callback(os.close, namespaces_fd)
        namespaces_parent_identity = _identity(namespaces_fd)
        namespace_identities: dict[str, tuple[int, int]] = {}
        record_identities: dict[str, tuple[int, int]] = {}
        for namespace in expected.namespaces:
            namespace_fd = (
                _mkdir(namespaces_fd, namespace.condition_id)
                if prepare
                else secure_fs.open_child_directory(namespaces_fd, namespace.condition_id)
            )
            stack.callback(os.close, namespace_fd)
            namespace_identities[namespace.condition_id] = _identity(namespace_fd)
            ns_entries = _entry_kinds(namespace_fd)
            if prepare and ns_entries and set(ns_entries) != {"namespace.json", "records"}:
                raise OutcomeDiagnosticResultStoreError("namespace contains partial metadata")
            if set(ns_entries) - {"namespace.json", "records"}:
                raise OutcomeDiagnosticResultStoreError(
                    "namespace contains extra or foreign entries"
                )
            records_fd = (
                _mkdir(namespace_fd, "records")
                if prepare
                else secure_fs.open_child_directory(namespace_fd, "records")
            )
            stack.callback(os.close, records_fd)
            record_identities[namespace.condition_id] = _identity(records_fd)
            if _entry_kinds(records_fd):
                raise OutcomeDiagnosticResultStoreError(
                    "diagnostic records namespace must be empty before execution"
                )
        if prepare:
            names = {
                "config.json",
                "expected-units.json",
                "run.json",
                ACTIVATION_INTENT_NAME,
                "namespaces",
            }
            current = _entry_kinds(run_fd)
            if current and set(current) not in ({"namespaces"}, names):
                raise OutcomeDiagnosticResultStoreError("run namespace contains partial metadata")
            if set(current) - names:
                raise OutcomeDiagnosticResultStoreError(
                    "run namespace contains extra or historical entries"
                )
            identities = {
                "root": list(_identity(root_fd)),
                "family": list(family_identity),
                "run": list(run_identity),
                "namespaces_parent": list(namespaces_parent_identity),
                "namespaces": {key: list(value) for key, value in namespace_identities.items()},
                "records": {key: list(value) for key, value in record_identities.items()},
            }
            config, expected_units, run, extras = _metadata(expected, identities)
            _write_or_verify(run_fd, "config.json", config)
            _write_or_verify(run_fd, "expected-units.json", expected_units)
            _write_or_verify(run_fd, "run.json", run)
            _write_or_verify(
                family_fd,
                "family.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "family_id": expected.family_id,
                    "plan_id": expected.plan_id,
                    "protocol_sha256": expected.protocol_sha256,
                    "run_id": expected.run_id,
                    "expected_unit_count": EXPECTED_FAMILY_UNIT_COUNT,
                    "final_family_access": False,
                },
            )
            for namespace in expected.namespaces:
                namespace_fd = secure_fs.open_child_directory(namespaces_fd, namespace.condition_id)
                stack.callback(os.close, namespace_fd)
                if _identity(namespace_fd) != namespace_identities[namespace.condition_id]:
                    raise OutcomeDiagnosticResultStoreError(
                        "namespace identity changed before metadata write"
                    )
                _write_or_verify(
                    namespace_fd, "namespace.json", extras["namespace"][namespace.condition_id]
                )
            # Re-scan every child immediately before publishing the durable
            # activation intent.  Any concurrent extra/replacement is fatal.
            if _entry_kinds(run_fd) != {
                "config.json": "file",
                "expected-units.json": "file",
                "run.json": "file",
                "namespaces": "dir",
            }:
                raise OutcomeDiagnosticResultStoreError("run namespace raced during preparation")
            if _entry_kinds(family_fd) != {"family.json": "file", expected.run_id: "dir"}:
                raise OutcomeDiagnosticResultStoreError("family namespace raced during preparation")
            if _entry_kinds(namespaces_fd) != {
                condition.condition_id: "dir" for condition in expected.namespaces
            }:
                raise OutcomeDiagnosticResultStoreError("namespace matrix raced during preparation")
            for condition in expected.namespaces:
                check_fd = secure_fs.open_child_directory(namespaces_fd, condition.condition_id)
                try:
                    if _identity(check_fd) != namespace_identities[condition.condition_id]:
                        raise OutcomeDiagnosticResultStoreError(
                            "namespace identity changed during preparation"
                        )
                    namespace_entries = _entry_kinds(check_fd)
                    if namespace_entries != {"namespace.json": "file", "records": "dir"}:
                        if set(namespace_entries) - {"namespace.json", "records"}:
                            raise OutcomeDiagnosticResultStoreError(
                                "records namespace raced during preparation"
                            )
                        raise OutcomeDiagnosticResultStoreError(
                            "namespace raced during preparation"
                        )
                    records_check_fd = secure_fs.open_child_directory(check_fd, "records")
                    try:
                        if _identity(records_check_fd) != record_identities[condition.condition_id]:
                            raise OutcomeDiagnosticResultStoreError(
                                "records identity changed during preparation"
                            )
                        if _entry_kinds(records_check_fd):
                            raise OutcomeDiagnosticResultStoreError(
                                "records namespace changed during preparation"
                            )
                    finally:
                        os.close(records_check_fd)
                finally:
                    os.close(check_fd)
            _write_or_verify(run_fd, ACTIVATION_INTENT_NAME, extras["intent"])
        current_run = _entry_kinds(run_fd)
        expected_run_names = {
            "config.json",
            "expected-units.json",
            "run.json",
            ACTIVATION_INTENT_NAME,
            "namespaces",
        }
        if current_run != {
            name: ("dir" if name == "namespaces" else "file") for name in expected_run_names
        }:
            raise OutcomeDiagnosticResultStoreError(
                "run namespace is incomplete, extra, or historical"
            )
        identities = {
            "root": list(_identity(root_fd)),
            "family": list(family_identity),
            "run": list(run_identity),
            "namespaces_parent": list(namespaces_parent_identity),
            "namespaces": {key: list(value) for key, value in namespace_identities.items()},
            "records": {key: list(value) for key, value in record_identities.items()},
        }
        if not prepare:
            try:
                stored_run = json.loads(_read_stable_bytes_at(run_fd, "run.json"))
                stored_identities = stored_run["identities"]
                persisted = {
                    "root": tuple(stored_identities["root"]),
                    "family": tuple(stored_identities["family"]),
                    "run": tuple(stored_identities["run"]),
                    "namespaces_parent": tuple(stored_identities["namespaces_parent"]),
                    "namespaces": {
                        key: tuple(value) for key, value in stored_identities["namespaces"].items()
                    },
                    "records": {
                        key: tuple(value) for key, value in stored_identities["records"].items()
                    },
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise OutcomeDiagnosticResultStoreError(
                    "stored diagnostic identities are invalid"
                ) from exc
            current = {
                "root": tuple(identities["root"]),
                "family": tuple(identities["family"]),
                "run": tuple(identities["run"]),
                "namespaces_parent": tuple(identities["namespaces_parent"]),
                "namespaces": {
                    key: tuple(value) for key, value in identities["namespaces"].items()
                },
                "records": {key: tuple(value) for key, value in identities["records"].items()},
            }
            if persisted != current:
                raise OutcomeDiagnosticResultStoreError("diagnostic descriptor identity changed")
            identities = {
                "root": list(persisted["root"]),
                "family": list(persisted["family"]),
                "run": list(persisted["run"]),
                "namespaces_parent": list(persisted["namespaces_parent"]),
                "namespaces": {key: list(value) for key, value in persisted["namespaces"].items()},
                "records": {key: list(value) for key, value in persisted["records"].items()},
            }
        config, expected_units, run, extras = _metadata(expected, identities)
        for name, value in (
            ("config.json", config),
            ("expected-units.json", expected_units),
            ("run.json", run),
            (ACTIVATION_INTENT_NAME, extras["intent"]),
        ):
            if _read_stable_bytes_at(run_fd, name) != _canonical(value):
                raise OutcomeDiagnosticResultStoreError(
                    f"diagnostic metadata differs from canonical authority: {name}"
                )
        family_metadata = {
            "schema_version": SCHEMA_VERSION,
            "family_id": expected.family_id,
            "plan_id": expected.plan_id,
            "protocol_sha256": expected.protocol_sha256,
            "run_id": expected.run_id,
            "expected_unit_count": EXPECTED_FAMILY_UNIT_COUNT,
            "final_family_access": False,
        }
        if _read_stable_bytes_at(family_fd, "family.json") != _canonical(family_metadata):
            raise OutcomeDiagnosticResultStoreError(
                "family metadata differs from canonical authority"
            )
        for namespace in expected.namespaces:
            namespace_fd = secure_fs.open_child_directory(namespaces_fd, namespace.condition_id)
            stack.callback(os.close, namespace_fd)
            if _read_stable_bytes_at(namespace_fd, "namespace.json") != _canonical(
                extras["namespace"][namespace.condition_id]
            ):
                raise OutcomeDiagnosticResultStoreError(
                    "namespace metadata differs from canonical authority"
                )
        return OutcomeDiagnosticResultStore(
            expected,
            root_path,
            _identity(root_fd),
            family_identity,
            run_identity,
            namespaces_parent_identity,
            namespace_identities,
            record_identities,
            False,
            _CONSTRUCTION_TOKEN,
        )
    except OutcomeDiagnosticResultStoreError:
        raise
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise OutcomeDiagnosticResultStoreError(
            "cannot prepare/load diagnostic result store"
        ) from exc
    finally:
        stack.close()


def _prepare_all(
    lease: OutcomeDiagnosticActivationReadinessLease, expected: OutcomeDiagnosticExpectedPlan
) -> tuple[OutcomeDiagnosticResultStore, ...]:
    root_fd = lease.output_root_fd
    lease.require_active()
    _check_output_root_path(lease)
    entries = _entry_kinds(root_fd)
    if entries:
        # A complete canonical tree is repeatable; every partial or foreign tree is rejected.
        if set(entries) != set(FAMILIES) | {ROOT_METADATA_NAME}:
            raise OutcomeDiagnosticResultStoreError(
                "diagnostic output root is not empty or canonical"
            )
    else:
        pass
    if not entries:
        # Root marker is published only after all six stores have been prepared.
        stores = tuple(
            _prepare_or_load_one(
                root_fd,
                spec,
                prepare=True,
                root_path=lease.snapshot.output_root,
            )
            for spec in expected.stores
        )
        _write_or_verify(root_fd, ROOT_METADATA_NAME, _root_metadata(expected, _identity(root_fd)))
    else:
        _check_root(root_fd, expected, require_root_meta=True)
        stores = tuple(
            _prepare_or_load_one(
                root_fd,
                spec,
                prepare=False,
                root_path=lease.snapshot.output_root,
            )
            for spec in expected.stores
        )
    lease.require_active()
    _check_output_root_path(lease)
    return stores


def prepare_outcome_diagnostic_result_stores(
    lease: OutcomeDiagnosticActivationReadinessLease,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    validated_plan: ValidatedOutcomePlan,
) -> tuple[OutcomeDiagnosticResultStore, ...]:
    """Prepare all six inert stores under the lease's already-held output fd."""

    active = _validate_lease(lease, snapshot)
    _check_output_root_path(active)
    expected = build_outcome_diagnostic_expected_plan(validated_plan, snapshot)
    return _prepare_all(active, expected)


def load_outcome_diagnostic_result_stores(
    lease: OutcomeDiagnosticActivationReadinessLease,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    validated_plan: ValidatedOutcomePlan,
) -> tuple[OutcomeDiagnosticResultStore, ...]:
    """Load an existing complete inert tree; never creates or follows a path."""

    active = _validate_lease(lease, snapshot)
    expected = build_outcome_diagnostic_expected_plan(validated_plan, snapshot)
    root_fd = active.output_root_fd
    _check_root(root_fd, expected, require_root_meta=True)
    stores = tuple(
        _prepare_or_load_one(
            root_fd,
            spec,
            prepare=False,
            root_path=active.snapshot.output_root,
        )
        for spec in expected.stores
    )
    active.require_active()
    _check_output_root_path(active)
    return stores


__all__ = [
    "ACTIVATION_INTENT_NAME",
    "EXPECTED_FAMILY_UNIT_COUNT",
    "EXPECTED_NAMESPACE_UNIT_COUNT",
    "EXPECTED_TOTAL_UNIT_COUNT",
    "OutcomeDiagnosticExpectedPlan",
    "OutcomeDiagnosticNamespaceSpec",
    "OutcomeDiagnosticResultStore",
    "OutcomeDiagnosticResultStoreError",
    "OutcomeDiagnosticResultStorePlanError",
    "OutcomeDiagnosticResultStoreSpec",
    "ROOT_METADATA_NAME",
    "SCHEMA_VERSION",
    "build_outcome_diagnostic_expected_plan",
    "load_outcome_diagnostic_result_stores",
    "prepare_outcome_diagnostic_result_stores",
    "validate_outcome_diagnostic_expected_plan",
]
