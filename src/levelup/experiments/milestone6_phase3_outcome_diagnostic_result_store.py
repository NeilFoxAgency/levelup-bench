"""Inert, descriptor-held result namespaces for the Phase 3 outcome diagnostic.

This module partitions the opaque, validated outcome-diagnostic plan into six
family stores and publishes canonical metadata under a descriptor held by
readiness.  Preparation remains inert; the separate activation facade at the
bottom of this module is the only path which grants write capability.  Both
boundaries are descriptor-relative and never use a path-only result loader.
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
from typing import Any, Iterator, Literal, Mapping

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
from levelup.experiments.runner.records import AttemptRecord, UnitKey, UnitRecord, UnitSeeds

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
            try:
                active.require_active()
            except (OutcomeDiagnosticReadinessError, OSError, ValueError) as exc:
                raise OutcomeDiagnosticResultStoreError(
                    "diagnostic output root path identity changed"
                ) from exc
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


# ---------------------------------------------------------------------------
# Activated, descriptor-pinned runtime facade
# ---------------------------------------------------------------------------

# Preparation intentionally leaves the six stores inert.  The marker below is
# the sole durable commit which permits a driver to publish result records.
# It is distinct from the per-family activation intent written during
# preparation; the latter is descriptive metadata and never grants write
# authority.
RUNTIME_ACTIVATION_MARKER_NAME = "outcome-diagnostic-activation.json"
ACTIVATION_MARKER_NAME = RUNTIME_ACTIVATION_MARKER_NAME
RUNTIME_ACTIVATION_SCHEMA_VERSION = f"{SCHEMA_VERSION}.activation.v1"

_ATTEMPT_MARKER = ".attempt-"
_RUNTIME_TOKEN = object()


def _runtime_identity(fd: int) -> tuple[int, int]:
    try:
        value = os.fstat(fd)
    except OSError as exc:
        raise OutcomeDiagnosticResultStoreError("diagnostic runtime descriptor is closed") from exc
    if not stat.S_ISDIR(value.st_mode):
        raise OutcomeDiagnosticResultStoreError("diagnostic runtime descriptor is not a directory")
    return int(value.st_dev), int(value.st_ino)


def _runtime_child_directory_identity(parent_fd: int, name: str) -> tuple[int, int]:
    """Return a no-follow child-directory identity from its canonical parent."""

    try:
        observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise OutcomeDiagnosticResultStoreError(
            f"diagnostic runtime directory path changed: {name}"
        ) from exc
    if not stat.S_ISDIR(observed.st_mode):
        raise OutcomeDiagnosticResultStoreError(
            f"diagnostic runtime directory path is not a directory: {name}"
        )
    return int(observed.st_dev), int(observed.st_ino)


def _runtime_record_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, str]:
    if not stat.S_ISREG(value.st_mode):
        raise OutcomeDiagnosticResultStoreError("diagnostic result is not a regular file")
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        "",
    )


def _runtime_record_snapshot(fd: int, name: str) -> tuple[bytes, tuple[int, int, int, int, int, str]]:
    """Read one record through a held descriptor and reject races."""

    try:
        with secure_fs.open_regular_file_at(fd, name) as record_fd:
            before = os.fstat(record_fd)
            fingerprint_before = _runtime_record_fingerprint(before)
            path_before = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if _runtime_record_fingerprint(path_before) != fingerprint_before:
                raise OutcomeDiagnosticResultStoreError("diagnostic result identity changed while opening")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(record_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            rendered = b"".join(chunks)
            after = os.fstat(record_fd)
            fingerprint_after = _runtime_record_fingerprint(after)
            path_after = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if (
                fingerprint_after != fingerprint_before
                or _runtime_record_fingerprint(path_after) != fingerprint_after
                or len(rendered) != int(after.st_size)
            ):
                raise OutcomeDiagnosticResultStoreError("diagnostic result changed while being read")
    except OutcomeDiagnosticResultStoreError:
        raise
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise OutcomeDiagnosticResultStoreError(f"cannot read diagnostic result: {name}") from exc
    return rendered, (*fingerprint_after[:-1], hashlib.sha256(rendered).hexdigest())


def _runtime_parse_record(
    rendered: bytes,
    name: str,
    model: type[UnitRecord] | type[AttemptRecord],
) -> UnitRecord | AttemptRecord:
    try:
        value = json.loads(rendered)
        record = model.model_validate(value)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OutcomeDiagnosticResultStoreError(f"invalid diagnostic result record: {name}") from exc
    if rendered != _canonical(record.model_dump(mode="json")):
        raise OutcomeDiagnosticResultStoreError(f"diagnostic result record is not canonical: {name}")
    return record


def _runtime_attempt_name(unit_id: str, attempt: int) -> str:
    # Keep the historical four-digit presentation for ordinary attempts while
    # permitting arbitrarily large positive integers.  There is deliberately
    # no upper bound here: retry policy belongs to the driver, not storage.
    return f"{unit_id}{_ATTEMPT_MARKER}{attempt:04d}.json"


def _runtime_parse_name(name: str, expected_ids: set[str]) -> tuple[str, int, str]:
    if name.endswith(".json") and _ATTEMPT_MARKER not in name:
        unit_id = name[:-5]
        if unit_id in expected_ids:
            return unit_id, 0, "completed"
    if not name.endswith(".json"):
        raise OutcomeDiagnosticResultStoreError("diagnostic records namespace contains a non-json entry")
    stem = name[:-5]
    unit_id, separator, number = stem.rpartition(_ATTEMPT_MARKER)
    if not separator or unit_id not in expected_ids or not number.isdigit() or not number:
        raise OutcomeDiagnosticResultStoreError(f"foreign or malformed diagnostic record: {name}")
    # int accepts arbitrarily large values, and formatting enforces one
    # canonical filename (no alternate leading-zero spellings).
    try:
        attempt = int(number)
    except ValueError as exc:
        raise OutcomeDiagnosticResultStoreError(f"malformed diagnostic attempt: {name}") from exc
    if attempt < 1 or _runtime_attempt_name(unit_id, attempt) != name:
        raise OutcomeDiagnosticResultStoreError(f"non-canonical diagnostic attempt filename: {name}")
    return unit_id, attempt, "attempt"


def _runtime_expected_key(planned: OutcomePlannedUnit) -> UnitKey:
    return UnitKey(
        phase="validation",
        condition_id=f"{planned.condition_id}--{planned.tuple_id}",
        family_id=planned.heldout_family,
        task_id=planned.task_id,
        task_index=planned.task_index,
        replicate=planned.replicate,
    )


def _runtime_expected_seeds(planned: OutcomePlannedUnit) -> UnitSeeds:
    return UnitSeeds(
        model_seed=planned.model_seed,
        environment_seed=planned.environment_seed,
        probe_seed=planned.probe_seed,
        search_seed=planned.search_seed,
        data_order_seed=planned.data_order_seed,
    )


def _runtime_validate_record_identity(
    record: UnitRecord | AttemptRecord,
    planned: OutcomePlannedUnit,
    spec: OutcomeDiagnosticResultStoreSpec,
    *,
    filename: str,
    condition_id: str,
) -> None:
    expected_name = (
        f"{record.unit_id}.json"
        if isinstance(record, UnitRecord)
        else _runtime_attempt_name(record.unit_id, record.attempt)
    )
    if (
        filename != expected_name
        or record.unit_id != planned.unit_id
        or record.run_id != spec.run_id
        or record.config_sha256 != spec.config_sha256
        or record.key != _runtime_expected_key(planned)
        or record.seeds != _runtime_expected_seeds(planned)
        or planned.condition_id != condition_id
        or (isinstance(record, UnitRecord)
            and record.exposure_manifest_sha256 != planned.exposure_manifest_sha256)
    ):
        raise OutcomeDiagnosticResultStoreError(
            f"diagnostic {type(record).__name__} identity differs from planned unit: {filename}"
        )


def _runtime_read_entries(fd: int) -> tuple[str, ...]:
    try:
        # strict_regular_entries rejects symlinks and non-regular children.  A
        # separate name parser below rejects temporary/foreign regular files.
        return tuple(sorted(secure_fs.strict_regular_entries(fd)))
    except secure_fs.SecureFilesystemError as exc:
        raise OutcomeDiagnosticResultStoreError("diagnostic records namespace is unsafe") from exc


def _runtime_write_once(fd: int, name: str, rendered: bytes) -> bool:
    """Atomically publish a new regular file, accepting only exact retries."""

    try:
        existing, _fingerprint = _runtime_record_snapshot(fd, name)
    except OutcomeDiagnosticResultStoreError as exc:
        if not _is_missing(exc):
            raise
        temporary = f".{name}.{uuid.uuid4().hex}.tmp"
        temp_fd: int | None = None
        try:
            temp_fd = os.open(
                temporary,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600,
                dir_fd=fd,
            )
            with os.fdopen(temp_fd, "wb") as handle:
                temp_fd = None
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
            os.fsync(fd)
            return True
        except FileExistsError:
            existing, _fingerprint = _runtime_record_snapshot(fd, name)
        except (OSError, secure_fs.SecureFilesystemError) as write_exc:
            raise OutcomeDiagnosticResultStoreError(f"cannot publish diagnostic result: {name}") from write_exc
        finally:
            if temp_fd is not None:
                os.close(temp_fd)
            try:
                os.unlink(temporary, dir_fd=fd)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise OutcomeDiagnosticResultStoreError("cannot remove diagnostic result temporary") from exc
    if existing != rendered:
        raise OutcomeDiagnosticResultStoreError(f"conflicting diagnostic result: {name}")
    return False


@dataclass(slots=True)
class _RuntimeFamilyStore:
    _batch: "OutcomeDiagnosticActivatedBatch"
    _index: int

    @property
    def family_id(self) -> str:
        return self._batch._stores[self._index].family_id

    @property
    def run_id(self) -> str:
        return self._batch._stores[self._index].run_id

    @property
    def config_sha256(self) -> str:
        return self._batch._stores[self._index].config_sha256

    def write_completed(self, record: UnitRecord) -> bool:
        return self._batch._write_completed(self._index, record)

    def write_attempt(self, record: AttemptRecord) -> bool:
        return self._batch._write_attempt(self._index, record)

    def load_completed(self, unit_id: str) -> UnitRecord | None:
        return self._batch._load_completed(self._index, unit_id)

    def completed_unit_ids(self) -> tuple[str, ...]:
        return self._batch.completed_unit_ids(self.family_id)

    def attempt_records(self) -> tuple[AttemptRecord, ...]:
        return self._batch.attempt_records(self.family_id)

    def next_attempt_number(self, unit_id: str) -> int:
        return self._batch.next_attempt_number(unit_id, self.family_id)

    def planned_unit(self, unit_id: str) -> OutcomePlannedUnit:
        return self._batch.planned_unit(unit_id, self.family_id)

    def latest_attempt(self, unit_id: str) -> AttemptRecord | None:
        return self._batch.latest_attempt(unit_id, self.family_id)

    def last_attempt_retryable(self, unit_id: str) -> bool | None:
        latest = self.latest_attempt(unit_id)
        return None if latest is None else latest.retryable


@dataclass(slots=True)
class OutcomeDiagnosticActivatedBatch:
    """Capability-bearing six-family runtime facade.

    All descriptors are opened from the live readiness lease's held output
    descriptor and remain open for the context lifetime.  No method reopens a
    path, aggregates outcomes, or consults an evaluator.
    """

    _stores: tuple[OutcomeDiagnosticResultStore, ...]
    _expected: OutcomeDiagnosticExpectedPlan
    _lease: OutcomeDiagnosticActivationReadinessLease
    _root_fd: int
    _descriptors: tuple[dict[str, Any], ...]
    _identities: tuple[dict[str, Any], ...]
    _marker_fd: int
    _marker_identity: tuple[int, int]
    _marker_fingerprint: tuple[int, int, int, int, int, str]
    _marker_bytes: bytes
    _record_fingerprints: dict[tuple[str, str], tuple[int, int, int, int, int, str]]
    _unit_maps: tuple[dict[str, OutcomePlannedUnit], ...]
    _token: object = field(repr=False, compare=False)
    _active: bool = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def stores(self) -> tuple[_RuntimeFamilyStore, ...]:
        self._require_live()
        return tuple(_RuntimeFamilyStore(self, i) for i in range(len(self._stores)))

    def store_for_family(self, family_id: str) -> _RuntimeFamilyStore:
        self._require_live()
        try:
            return _RuntimeFamilyStore(self, FAMILIES.index(family_id))
        except ValueError as exc:
            raise OutcomeDiagnosticResultStoreError(f"unknown diagnostic family: {family_id}") from exc

    def _require_live(self, *, validate_records: bool = False) -> None:
        if self._token is not _RUNTIME_TOKEN or not self._active:
            raise OutcomeDiagnosticResultStoreError("diagnostic runtime capability has expired")
        try:
            self._lease.require_active()
        except (OutcomeDiagnosticReadinessError, OSError, ValueError) as exc:
            raise OutcomeDiagnosticResultStoreError("diagnostic readiness lease is not active") from exc
        try:
            _check_output_root_path(self._lease)
            held_marker_fingerprint = _runtime_record_fingerprint(os.fstat(self._marker_fd))
            if held_marker_fingerprint[:-1] != self._marker_fingerprint[:-1]:
                raise OutcomeDiagnosticResultStoreError("diagnostic activation marker identity changed")
            rendered, path_fingerprint = _runtime_record_snapshot(
                self._root_fd, RUNTIME_ACTIVATION_MARKER_NAME
            )
            if (
                path_fingerprint != self._marker_fingerprint
                or path_fingerprint[:2] != self._marker_identity
                or rendered != self._marker_bytes
            ):
                raise OutcomeDiagnosticResultStoreError("diagnostic activation marker changed")
            if _runtime_identity(self._root_fd) != self._identities[0]["root"]:
                raise OutcomeDiagnosticResultStoreError("diagnostic output root identity changed")
            for index, descriptor in enumerate(self._descriptors):
                store = self._stores[index]
                if (
                    _runtime_child_directory_identity(self._root_fd, store.family_id)
                    != store.family_identity
                    or _runtime_child_directory_identity(
                        descriptor["family"], store.run_id
                    )
                    != store.run_identity
                    or _runtime_child_directory_identity(
                        descriptor["run"], "namespaces"
                    )
                    != store.namespaces_parent_identity
                ):
                    raise OutcomeDiagnosticResultStoreError(
                        "diagnostic result directory path identity changed"
                    )
                for condition in CONDITIONS:
                    if (
                        _runtime_child_directory_identity(
                            descriptor["namespaces"], condition
                        )
                        != store.namespace_identities[condition]
                        or _runtime_child_directory_identity(
                            descriptor["namespace_fds"][condition], "records"
                        )
                        != store.record_namespace_identities[condition]
                    ):
                        raise OutcomeDiagnosticResultStoreError(
                            "diagnostic namespace path identity changed"
                        )
                for key, expected_identity in self._identities[index].items():
                    if key.startswith("records:"):
                        fd = descriptor["records"][key.split(":", 1)[1]]
                    else:
                        fd = descriptor[key]
                    if _runtime_identity(fd) != expected_identity:
                        raise OutcomeDiagnosticResultStoreError("diagnostic result directory identity changed")
                if validate_records:
                    self._validate_record_namespace(index, track_new=False)
        except OutcomeDiagnosticResultStoreError:
            raise
        except (OSError, secure_fs.SecureFilesystemError) as exc:
            raise OutcomeDiagnosticResultStoreError("cannot revalidate diagnostic runtime tree") from exc

    def _validate_record_namespace(self, index: int, *, track_new: bool) -> None:
        descriptor = self._descriptors[index]
        expected_ids = set(self._unit_maps[index])
        for condition in CONDITIONS:
            records_fd = descriptor["records"][condition]
            names = _runtime_read_entries(records_fd)
            for name in names:
                unit_id, _attempt, kind = _runtime_parse_name(name, expected_ids)
                planned = self._unit_maps[index][unit_id]
                rendered, fingerprint = _runtime_record_snapshot(records_fd, name)
                key = (f"{self._stores[index].family_id}:{condition}", name)
                previous = self._record_fingerprints.get(key)
                if previous is None:
                    if not track_new:
                        raise OutcomeDiagnosticResultStoreError("untracked diagnostic result appeared")
                    self._record_fingerprints[key] = fingerprint
                elif previous != fingerprint:
                    raise OutcomeDiagnosticResultStoreError("diagnostic result identity or content changed")
                model = UnitRecord if kind == "completed" else AttemptRecord
                record = _runtime_parse_record(rendered, name, model)
                _runtime_validate_record_identity(
                    record, planned, self._stores[index].spec, filename=name, condition_id=condition
                )
        known = {
            (f"{self._stores[index].family_id}:{condition}", name)
            for condition in CONDITIONS
            for name in _runtime_read_entries(descriptor["records"][condition])
        }
        if any(key[0].startswith(self._stores[index].family_id + ":") and key not in known for key in self._record_fingerprints):
            raise OutcomeDiagnosticResultStoreError("diagnostic result was removed during runtime")

    def _planned(self, index: int, unit_id: str) -> OutcomePlannedUnit:
        try:
            return self._unit_maps[index][unit_id]
        except KeyError as exc:
            raise OutcomeDiagnosticResultStoreError(f"foreign diagnostic unit: {unit_id}") from exc

    def planned_unit(
        self, unit_id: str, family_id: str | None = None
    ) -> OutcomePlannedUnit:
        self._require_live()
        if family_id is None:
            matches = [mapping[unit_id] for mapping in self._unit_maps if unit_id in mapping]
            if len(matches) != 1:
                raise OutcomeDiagnosticResultStoreError(f"foreign or duplicate diagnostic unit: {unit_id}")
            return matches[0]
        try:
            index = FAMILIES.index(family_id)
        except ValueError as exc:
            raise OutcomeDiagnosticResultStoreError(
                f"unknown diagnostic family: {family_id}"
            ) from exc
        return self._planned(index, unit_id)

    def _write_completed(self, index: int, record: UnitRecord) -> bool:
        self._require_live()
        if type(record) is not UnitRecord:
            raise OutcomeDiagnosticResultStoreError("completed diagnostic result must be UnitRecord")
        planned = self._planned(index, record.unit_id)
        condition = planned.condition_id
        name = f"{record.unit_id}.json"
        _runtime_validate_record_identity(record, planned, self._stores[index].spec, filename=name, condition_id=condition)
        fd = self._descriptors[index]["records"][condition]
        published = _runtime_write_once(fd, name, _canonical(record.model_dump(mode="json")))
        rendered, fingerprint = _runtime_record_snapshot(fd, name)
        if rendered != _canonical(record.model_dump(mode="json")):
            raise OutcomeDiagnosticResultStoreError("completed diagnostic result changed during publication")
        key = (f"{self._stores[index].family_id}:{condition}", name)
        if key in self._record_fingerprints and self._record_fingerprints[key] != fingerprint:
            raise OutcomeDiagnosticResultStoreError("completed diagnostic result identity changed")
        self._record_fingerprints[key] = fingerprint
        self._require_live()
        return published

    def _write_attempt(self, index: int, record: AttemptRecord) -> bool:
        self._require_live()
        if type(record) is not AttemptRecord:
            raise OutcomeDiagnosticResultStoreError("attempt diagnostic result must be AttemptRecord")
        planned = self._planned(index, record.unit_id)
        condition = planned.condition_id
        name = _runtime_attempt_name(record.unit_id, record.attempt)
        _runtime_validate_record_identity(record, planned, self._stores[index].spec, filename=name, condition_id=condition)
        fd = self._descriptors[index]["records"][condition]
        published = _runtime_write_once(fd, name, _canonical(record.model_dump(mode="json")))
        rendered, fingerprint = _runtime_record_snapshot(fd, name)
        if rendered != _canonical(record.model_dump(mode="json")):
            raise OutcomeDiagnosticResultStoreError("attempt diagnostic result changed during publication")
        key = (f"{self._stores[index].family_id}:{condition}", name)
        if key in self._record_fingerprints and self._record_fingerprints[key] != fingerprint:
            raise OutcomeDiagnosticResultStoreError("attempt diagnostic result identity changed")
        self._record_fingerprints[key] = fingerprint
        self._require_live()
        return published

    def _load_completed(self, index: int, unit_id: str) -> UnitRecord | None:
        self._require_live()
        planned = self._planned(index, unit_id)
        name = f"{unit_id}.json"
        fd = self._descriptors[index]["records"][planned.condition_id]
        if name not in _runtime_read_entries(fd):
            return None
        rendered, fingerprint = _runtime_record_snapshot(fd, name)
        key = (f"{self._stores[index].family_id}:{planned.condition_id}", name)
        if self._record_fingerprints.get(key) != fingerprint:
            raise OutcomeDiagnosticResultStoreError("completed diagnostic result identity changed")
        record = _runtime_parse_record(rendered, name, UnitRecord)
        _runtime_validate_record_identity(record, planned, self._stores[index].spec, filename=name, condition_id=planned.condition_id)
        self._require_live()
        return record

    def completed_unit_ids(self, family_id: str | None = None) -> tuple[str, ...]:
        self._require_live(validate_records=True)
        if family_id is None:
            indices = range(len(self._stores))
        else:
            try:
                indices = (FAMILIES.index(family_id),)
            except ValueError as exc:
                raise OutcomeDiagnosticResultStoreError(
                    f"unknown diagnostic family: {family_id}"
                ) from exc
        values: list[str] = []
        for index in indices:
            present: set[str] = set()
            expected_ids = set(self._unit_maps[index])
            for condition in CONDITIONS:
                fd = self._descriptors[index]["records"][condition]
                for name in _runtime_read_entries(fd):
                    unit_id, _attempt, kind = _runtime_parse_name(name, expected_ids)
                    if kind == "completed":
                        present.add(unit_id)
            for planned in self._stores[index].spec.units:
                if planned.unit_id in present:
                    values.append(planned.unit_id)
        self._require_live(validate_records=True)
        return tuple(values)

    def attempt_records(self, family_id: str | None = None) -> tuple[AttemptRecord, ...]:
        self._require_live(validate_records=True)
        if family_id is None:
            indices = range(len(self._stores))
        else:
            try:
                indices = (FAMILIES.index(family_id),)
            except ValueError as exc:
                raise OutcomeDiagnosticResultStoreError(
                    f"unknown diagnostic family: {family_id}"
                ) from exc
        values: list[AttemptRecord] = []
        for index in indices:
            expected_ids = set(self._unit_maps[index])
            for condition in CONDITIONS:
                fd = self._descriptors[index]["records"][condition]
                for name in _runtime_read_entries(fd):
                    unit_id, _attempt, kind = _runtime_parse_name(name, expected_ids)
                    if kind != "attempt":
                        continue
                    rendered, fingerprint = _runtime_record_snapshot(fd, name)
                    key = (f"{self._stores[index].family_id}:{condition}", name)
                    if self._record_fingerprints.get(key) != fingerprint:
                        raise OutcomeDiagnosticResultStoreError("attempt diagnostic result identity changed")
                    record = _runtime_parse_record(rendered, name, AttemptRecord)
                    _runtime_validate_record_identity(record, self._unit_maps[index][unit_id], self._stores[index].spec, filename=name, condition_id=condition)
                    values.append(record)
        self._require_live(validate_records=True)
        return tuple(values)

    def latest_attempt(self, unit_id: str, family_id: str | None = None) -> AttemptRecord | None:
        attempts = [item for item in self.attempt_records(family_id) if item.unit_id == unit_id]
        if not attempts:
            # Validate that the requested unit is not foreign even when no
            # attempts exist.
            if family_id is None:
                if not any(unit_id in mapping for mapping in self._unit_maps):
                    raise OutcomeDiagnosticResultStoreError(f"foreign diagnostic unit: {unit_id}")
            else:
                self._planned(FAMILIES.index(family_id), unit_id)
            return None
        return max(attempts, key=lambda item: item.attempt)

    def next_attempt_number(self, unit_id: str, family_id: str | None = None) -> int:
        latest = self.latest_attempt(unit_id, family_id)
        return 1 if latest is None else latest.attempt + 1

    def last_attempt_retryable(
        self, unit_id: str, family_id: str | None = None
    ) -> bool | None:
        latest = self.latest_attempt(unit_id, family_id)
        return None if latest is None else latest.retryable

    def close(self) -> None:
        if not self._active:
            return
        self._active = False
        self._record_fingerprints.clear()
        for mapping in self._unit_maps:
            mapping.clear()


def _runtime_validate_store_arguments(
    stores: tuple[OutcomeDiagnosticResultStore, ...], expected: OutcomeDiagnosticExpectedPlan,
) -> None:
    if type(expected) is not OutcomeDiagnosticExpectedPlan:
        raise OutcomeDiagnosticResultStoreError("activation requires the canonical expected diagnostic plan")
    if expected.final_family_access or expected.family_order != FAMILIES or expected.condition_order != CONDITIONS:
        raise OutcomeDiagnosticResultStoreError("activation expected plan is not development-only")
    if len(stores) != len(FAMILIES) or tuple(item.family_id for item in stores) != FAMILIES:
        raise OutcomeDiagnosticResultStoreError("activation requires exactly six stores in frozen family order")
    if any(type(item) is not OutcomeDiagnosticResultStore for item in stores):
        raise OutcomeDiagnosticResultStoreError("activation stores are not canonical prepared stores")
    for item, spec in zip(stores, expected.stores, strict=True):
        if item.spec != spec or item.execution_ready:
            raise OutcomeDiagnosticResultStoreError(f"store differs from expected diagnostic plan: {item.family_id}")
    roots = {item.root_identity for item in stores}
    if len(roots) != 1:
        raise OutcomeDiagnosticResultStoreError("diagnostic stores do not share one root identity")


def _runtime_validate_inert_tree(
    descriptor: dict[str, Any],
    store: OutcomeDiagnosticResultStore,
    expected: OutcomeDiagnosticExpectedPlan,
    *,
    root_identity: tuple[int, int],
    marker_present: bool,
) -> None:
    """Validate all prepared metadata while the runtime descriptors are held."""

    root_entries = _entry_kinds(descriptor["root"])
    allowed_root = set(FAMILIES) | {ROOT_METADATA_NAME}
    if marker_present:
        allowed_root.add(RUNTIME_ACTIVATION_MARKER_NAME)
    if set(root_entries) != allowed_root:
        raise OutcomeDiagnosticResultStoreError("diagnostic root contains foreign or missing entries")
    if _read_stable_bytes_at(descriptor["root"], ROOT_METADATA_NAME) != _canonical(
        _root_metadata(expected, root_identity)
    ):
        raise OutcomeDiagnosticResultStoreError("diagnostic root metadata differs from authority")

    family_entries = _entry_kinds(descriptor["family"])
    if family_entries != {"family.json": "file", store.run_id: "dir"}:
        raise OutcomeDiagnosticResultStoreError("diagnostic family namespace is incomplete or foreign")
    family_metadata = {
        "schema_version": SCHEMA_VERSION,
        "family_id": store.family_id,
        "plan_id": store.spec.plan_id,
        "protocol_sha256": store.spec.protocol_sha256,
        "run_id": store.run_id,
        "expected_unit_count": EXPECTED_FAMILY_UNIT_COUNT,
        "final_family_access": False,
    }
    if _read_stable_bytes_at(descriptor["family"], "family.json") != _canonical(family_metadata):
        raise OutcomeDiagnosticResultStoreError("diagnostic family metadata differs from authority")

    run_entries = _entry_kinds(descriptor["run"])
    required_run = {
        "config.json": "file",
        "expected-units.json": "file",
        "run.json": "file",
        ACTIVATION_INTENT_NAME: "file",
        "namespaces": "dir",
    }
    if run_entries != required_run:
        raise OutcomeDiagnosticResultStoreError("diagnostic run namespace is incomplete or foreign")
    identities = {
        "root": list(store.root_identity),
        "family": list(store.family_identity),
        "run": list(store.run_identity),
        "namespaces_parent": list(store.namespaces_parent_identity),
        "namespaces": {key: list(value) for key, value in store.namespace_identities.items()},
        "records": {key: list(value) for key, value in store.record_namespace_identities.items()},
    }
    config, expected_units, run, extras = _metadata(store.spec, identities)
    for name, value in (
        ("config.json", config),
        ("expected-units.json", expected_units),
        ("run.json", run),
        (ACTIVATION_INTENT_NAME, extras["intent"]),
    ):
        if _read_stable_bytes_at(descriptor["run"], name) != _canonical(value):
            raise OutcomeDiagnosticResultStoreError(f"diagnostic metadata differs from authority: {name}")
    if _entry_kinds(descriptor["namespaces"]) != {condition: "dir" for condition in CONDITIONS}:
        raise OutcomeDiagnosticResultStoreError("diagnostic condition namespace matrix is incomplete")
    for condition in CONDITIONS:
        namespace_fd = descriptor["namespace_fds"][condition]
        entries = _entry_kinds(namespace_fd)
        if entries != {"namespace.json": "file", "records": "dir"}:
            raise OutcomeDiagnosticResultStoreError("diagnostic condition namespace is incomplete")
        if _read_stable_bytes_at(namespace_fd, "namespace.json") != _canonical(
            extras["namespace"][condition]
        ):
            raise OutcomeDiagnosticResultStoreError("diagnostic condition metadata differs from authority")


def _runtime_marker_body(
    expected: OutcomeDiagnosticExpectedPlan,
    stores: tuple[OutcomeDiagnosticResultStore, ...],
    root_identity: tuple[int, int],
) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_ACTIVATION_SCHEMA_VERSION,
        "result_store_schema_version": SCHEMA_VERSION,
        "development_only": True,
        "final_family_access": False,
        "plan_id": expected.plan_id,
        "protocol_sha256": expected.protocol_sha256,
        "family_order": list(FAMILIES),
        "condition_order": list(CONDITIONS),
        "root_identity": list(root_identity),
        "stores": [
            {
                "family_id": store.family_id,
                "run_id": store.run_id,
                "config_sha256": store.config_sha256,
                "root_identity": list(store.root_identity),
                "family_identity": list(store.family_identity),
                "run_identity": list(store.run_identity),
                "namespaces_parent_identity": list(store.namespaces_parent_identity),
                "namespace_identities": {k: list(v) for k, v in store.namespace_identities.items()},
                "record_namespace_identities": {k: list(v) for k, v in store.record_namespace_identities.items()},
                "expected_unit_count": EXPECTED_FAMILY_UNIT_COUNT,
            }
            for store in stores
        ],
    }


def _runtime_marker_with_hash(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "activation_sha256": _sha(body)}


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticResumeRecordFingerprint:
    """Opaque stat/content fingerprint for one persisted runtime record."""

    family_id: str
    condition_id: str
    name: str
    stat: tuple[int, int, int, int, int, int]
    sha256: str


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticResumeBaseline:
    """Immutable descriptor-pinned baseline for a resumable diagnostic tree.

    The inspector deliberately does not parse record JSON.  Consumers may use
    these fingerprints to detect replacement or mutation before reopening a
    runtime capability; comparative values remain outside this boundary.
    """

    output_root: Path
    output_root_identity: tuple[int, int]
    output_state: Literal["prepared", "activated"]
    directory_identities: tuple[tuple[str, tuple[int, int]], ...]
    records: tuple[OutcomeDiagnosticResumeRecordFingerprint, ...]
    stores: tuple[OutcomeDiagnosticResultStore, ...]
    marker_stat: tuple[int, int, int, int, int, int] | None = None
    marker_sha256: str | None = None


def _resume_path_identity(parent_fd: int, name: str) -> tuple[int, int]:
    try:
        value = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise OutcomeDiagnosticResultStoreError(
            f"diagnostic resume entry disappeared: {name}"
        ) from exc
    if not stat.S_ISDIR(value.st_mode):
        raise OutcomeDiagnosticResultStoreError(
            f"diagnostic resume entry is not a directory: {name}"
        )
    return int(value.st_dev), int(value.st_ino)


def _resume_open_directory(
    parent_fd: int, name: str
) -> tuple[int, tuple[int, int]]:
    try:
        child_fd = secure_fs.open_child_directory(parent_fd, name)
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise OutcomeDiagnosticResultStoreError(
            f"cannot open diagnostic resume directory: {name}"
        ) from exc
    try:
        identity = _runtime_identity(child_fd)
        if _resume_path_identity(parent_fd, name) != identity:
            raise OutcomeDiagnosticResultStoreError(
                f"diagnostic resume directory identity changed: {name}"
            )
        return child_fd, identity
    except BaseException:
        os.close(child_fd)
        raise


def _resume_verify_directory(parent_fd: int, name: str, fd: int, identity: tuple[int, int]) -> None:
    if _runtime_identity(fd) != identity or _resume_path_identity(parent_fd, name) != identity:
        raise OutcomeDiagnosticResultStoreError(
            f"diagnostic resume directory identity changed: {name}"
        )


def _resume_file_snapshot(
    directory_fd: int, name: str
) -> tuple[tuple[int, int, int, int, int, int], str]:
    """Read one regular file through a held descriptor and pin stat+digest."""

    try:
        with secure_fs.open_regular_file_at(directory_fd, name) as file_fd:
            before = _file_identity(os.fstat(file_fd))
            path_before = _file_identity(
                os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            )
            if before != path_before:
                raise OutcomeDiagnosticResultStoreError(
                    f"diagnostic resume file identity changed: {name}"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(file_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            rendered = b"".join(chunks)
            after = _file_identity(os.fstat(file_fd))
            path_after = _file_identity(
                os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            )
            if before != after or after != path_after or len(rendered) != after[3]:
                raise OutcomeDiagnosticResultStoreError(
                    f"diagnostic resume file changed while being read: {name}"
                )
    except OutcomeDiagnosticResultStoreError:
        raise
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise OutcomeDiagnosticResultStoreError(
            f"cannot safely read diagnostic resume file: {name}"
        ) from exc
    return after, hashlib.sha256(rendered).hexdigest()


def inspect_outcome_diagnostic_resume_tree_at(
    output_root_fd: int,
    output_root_path: str | Path,
    expected: OutcomeDiagnosticExpectedPlan,
    *,
    output_state: Literal["prepared", "activated"],
) -> OutcomeDiagnosticResumeBaseline:
    """Inspect a prepared or activated tree using only held descriptors.

    This is intentionally a schema-neutral resume check: metadata is compared
    to canonical bytes, while record files are only fingerprinted and their
    JSON values are never parsed.  The caller retains ``output_root_fd``;
    every descendant descriptor opened by this function is closed exactly once
    before returning.
    """

    if type(expected) is not OutcomeDiagnosticExpectedPlan:
        raise OutcomeDiagnosticResultStoreError("resume inspection requires canonical expected plan")
    if output_state not in ("prepared", "activated"):
        raise OutcomeDiagnosticResultStoreError("resume inspection state is invalid")
    stack = ExitStack()
    stack.__enter__()
    records_seen: list[tuple[int, str, OutcomeDiagnosticResumeRecordFingerprint]] = []
    directories: list[tuple[str, tuple[int, int]]] = []
    try:
        root_identity = _runtime_identity(output_root_fd)
        try:
            path_value = os.stat(output_root_path, follow_symlinks=False)
        except OSError as exc:
            raise OutcomeDiagnosticResultStoreError("diagnostic output root path changed") from exc
        if not stat.S_ISDIR(path_value.st_mode) or (
            int(path_value.st_dev), int(path_value.st_ino)
        ) != root_identity:
            raise OutcomeDiagnosticResultStoreError("diagnostic output root path identity changed")
        root_entries = _entry_kinds(output_root_fd)
        allowed = set(FAMILIES) | {ROOT_METADATA_NAME}
        if output_state == "activated":
            allowed.add(RUNTIME_ACTIVATION_MARKER_NAME)
        if set(root_entries) != allowed:
            raise OutcomeDiagnosticResultStoreError("diagnostic resume root layout is not canonical")
        if _read_stable_bytes_at(output_root_fd, ROOT_METADATA_NAME) != _canonical(
            _root_metadata(expected, root_identity)
        ):
            raise OutcomeDiagnosticResultStoreError("diagnostic root metadata differs from authority")

        marker_stat: tuple[int, int, int, int, int, int] | None = None
        marker_sha256: str | None = None
        family_marker_stores: list[OutcomeDiagnosticResultStore] = []
        for spec in expected.stores:
            family_fd, family_identity = _resume_open_directory(output_root_fd, spec.family_id)
            stack.callback(os.close, family_fd)
            directories.append((f"{spec.family_id}:family", family_identity))
            if _entry_kinds(family_fd) != {"family.json": "file", spec.run_id: "dir"}:
                raise OutcomeDiagnosticResultStoreError("diagnostic family namespace is incomplete or foreign")
            family_meta = {
                "schema_version": SCHEMA_VERSION,
                "family_id": spec.family_id,
                "plan_id": spec.plan_id,
                "protocol_sha256": spec.protocol_sha256,
                "run_id": spec.run_id,
                "expected_unit_count": EXPECTED_FAMILY_UNIT_COUNT,
                "final_family_access": False,
            }
            if _read_stable_bytes_at(family_fd, "family.json") != _canonical(family_meta):
                raise OutcomeDiagnosticResultStoreError("diagnostic family metadata differs from authority")
            run_fd, run_identity = _resume_open_directory(family_fd, spec.run_id)
            stack.callback(os.close, run_fd)
            directories.append((f"{spec.family_id}:run", run_identity))
            required_run = {
                "config.json": "file",
                "expected-units.json": "file",
                "run.json": "file",
                ACTIVATION_INTENT_NAME: "file",
                "namespaces": "dir",
            }
            if _entry_kinds(run_fd) != required_run:
                raise OutcomeDiagnosticResultStoreError("diagnostic run namespace is incomplete or foreign")
            namespaces_fd, namespaces_identity = _resume_open_directory(run_fd, "namespaces")
            stack.callback(os.close, namespaces_fd)
            directories.append((f"{spec.family_id}:namespaces", namespaces_identity))
            identities = {
                "root": list(root_identity),
                "family": list(family_identity),
                "run": list(run_identity),
                "namespaces_parent": list(namespaces_identity),
                "namespaces": {},
                "records": {},
            }
            namespace_fds: dict[str, int] = {}
            record_fds: dict[str, int] = {}
            for namespace in spec.namespaces:
                condition = namespace.condition_id
                namespace_fd, namespace_identity = _resume_open_directory(namespaces_fd, condition)
                stack.callback(os.close, namespace_fd)
                namespace_fds[condition] = namespace_fd
                directories.append((f"{spec.family_id}:namespace:{condition}", namespace_identity))
                identities["namespaces"][condition] = list(namespace_identity)
                if _entry_kinds(namespace_fd) != {"namespace.json": "file", "records": "dir"}:
                    raise OutcomeDiagnosticResultStoreError("diagnostic condition namespace is incomplete or foreign")
                records_fd, records_identity = _resume_open_directory(namespace_fd, "records")
                stack.callback(os.close, records_fd)
                record_fds[condition] = records_fd
                directories.append((f"{spec.family_id}:records:{condition}", records_identity))
                identities["records"][condition] = list(records_identity)

            config, expected_units, run, extras = _metadata(spec, identities)
            for name, value in (
                ("config.json", config),
                ("expected-units.json", expected_units),
                ("run.json", run),
                (ACTIVATION_INTENT_NAME, extras["intent"]),
            ):
                if _read_stable_bytes_at(run_fd, name) != _canonical(value):
                    raise OutcomeDiagnosticResultStoreError(
                        f"diagnostic metadata differs from authority: {name}"
                    )
            for namespace in spec.namespaces:
                condition = namespace.condition_id
                if _read_stable_bytes_at(namespace_fds[condition], "namespace.json") != _canonical(
                    extras["namespace"][condition]
                ):
                    raise OutcomeDiagnosticResultStoreError("diagnostic condition metadata differs from authority")
                entries = _entry_kinds(record_fds[condition])
                if output_state == "prepared" and entries:
                    raise OutcomeDiagnosticResultStoreError("prepared diagnostic records namespace is not empty")
                expected_ids = {unit.unit_id for unit in namespace.units}
                for name in entries:
                    _unit_id, _attempt, _kind = _runtime_parse_name(name, expected_ids)
                    stat_value, digest = _resume_file_snapshot(record_fds[condition], name)
                    record = OutcomeDiagnosticResumeRecordFingerprint(
                        spec.family_id, condition, name, stat_value, digest
                    )
                    records_seen.append((record_fds[condition], name, record))
            _resume_verify_directory(output_root_fd, spec.family_id, family_fd, family_identity)
            _resume_verify_directory(family_fd, spec.run_id, run_fd, run_identity)
            _resume_verify_directory(run_fd, "namespaces", namespaces_fd, namespaces_identity)
            for condition in CONDITIONS:
                _resume_verify_directory(namespaces_fd, condition, namespace_fds[condition], tuple(identities["namespaces"][condition]))
                _resume_verify_directory(namespace_fds[condition], "records", record_fds[condition], tuple(identities["records"][condition]))
            family_marker_stores.append(
                OutcomeDiagnosticResultStore(
                    spec,
                    Path(output_root_path),
                    root_identity,
                    family_identity,
                    run_identity,
                    namespaces_identity,
                    {key: tuple(value) for key, value in identities["namespaces"].items()},
                    {key: tuple(value) for key, value in identities["records"].items()},
                    False,
                    _CONSTRUCTION_TOKEN,
                )
            )

        if output_state == "activated":
            marker_body = _runtime_marker_body(expected, tuple(family_marker_stores), root_identity)
            marker_bytes = _canonical(_runtime_marker_with_hash(marker_body))
            observed_marker = _read_stable_bytes_at(output_root_fd, RUNTIME_ACTIVATION_MARKER_NAME)
            if observed_marker != marker_bytes:
                raise OutcomeDiagnosticResultStoreError("diagnostic activation marker differs from authority")
            marker_stat, marker_sha256 = _resume_file_snapshot(
                output_root_fd, RUNTIME_ACTIVATION_MARKER_NAME
            )
            if marker_sha256 != hashlib.sha256(marker_bytes).hexdigest():
                raise OutcomeDiagnosticResultStoreError(
                    "diagnostic activation marker changed during inspection"
                )

        # A second descriptor-relative pass closes the race window between the
        # initial record snapshot and return, including same-byte inode swaps.
        for record_fd, name, baseline in records_seen:
            stat_value, digest = _resume_file_snapshot(record_fd, name)
            if stat_value != baseline.stat or digest != baseline.sha256:
                raise OutcomeDiagnosticResultStoreError("diagnostic resume record changed during inspection")
        if output_state == "activated":
            final_marker_stat, final_marker_sha256 = _resume_file_snapshot(
                output_root_fd, RUNTIME_ACTIVATION_MARKER_NAME
            )
            if final_marker_stat != marker_stat or final_marker_sha256 != marker_sha256:
                raise OutcomeDiagnosticResultStoreError(
                    "diagnostic activation marker changed during inspection"
                )
        if _entry_kinds(output_root_fd) != {name: ("dir" if name in FAMILIES else "file") for name in allowed}:
            raise OutcomeDiagnosticResultStoreError("diagnostic resume root changed during inspection")
        final_path = os.stat(output_root_path, follow_symlinks=False)
        if (
            not stat.S_ISDIR(final_path.st_mode)
            or (int(final_path.st_dev), int(final_path.st_ino)) != root_identity
            or _runtime_identity(output_root_fd) != root_identity
        ):
            raise OutcomeDiagnosticResultStoreError("diagnostic output root identity changed during inspection")
        return OutcomeDiagnosticResumeBaseline(
            Path(output_root_path),
            root_identity,
            output_state,
            tuple(directories),
            tuple(item[2] for item in records_seen),
            tuple(family_marker_stores),
            marker_stat,
            marker_sha256,
        )
    except OutcomeDiagnosticResultStoreError:
        raise
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise OutcomeDiagnosticResultStoreError("cannot inspect diagnostic resume tree") from exc
    finally:
        stack.close()


def _runtime_publish_marker(fd: int, rendered: bytes) -> None:
    try:
        existing, _fingerprint = _runtime_record_snapshot(fd, RUNTIME_ACTIVATION_MARKER_NAME)
    except OutcomeDiagnosticResultStoreError as exc:
        if not _is_missing(exc):
            raise
        temporary = f".{RUNTIME_ACTIVATION_MARKER_NAME}.{uuid.uuid4().hex}.tmp"
        temp_fd: int | None = None
        try:
            temp_fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600, dir_fd=fd)
            with os.fdopen(temp_fd, "wb") as handle:
                temp_fd = None
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, RUNTIME_ACTIVATION_MARKER_NAME, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
            os.fsync(fd)
        except FileExistsError:
            existing, _fingerprint = _runtime_record_snapshot(fd, RUNTIME_ACTIVATION_MARKER_NAME)
        except (OSError, secure_fs.SecureFilesystemError) as exc_write:
            raise OutcomeDiagnosticResultStoreError("cannot publish diagnostic activation marker") from exc_write
        finally:
            if temp_fd is not None:
                os.close(temp_fd)
            try:
                os.unlink(temporary, dir_fd=fd)
            except FileNotFoundError:
                pass
        if "existing" not in locals():
            return
    if existing != rendered:
        raise OutcomeDiagnosticResultStoreError("diagnostic activation marker conflicts with canonical authority")


@contextmanager
def activate_outcome_diagnostic_result_stores(
    stores: tuple[OutcomeDiagnosticResultStore, ...] | list[OutcomeDiagnosticResultStore],
    expected: OutcomeDiagnosticExpectedPlan,
    readiness_lease: OutcomeDiagnosticActivationReadinessLease,
    *,
    expected_git_commit: str,
) -> Iterator[OutcomeDiagnosticActivatedBatch]:
    """Atomically authorize and open the complete six-store runtime matrix."""

    typed_stores = tuple(stores)
    _runtime_validate_store_arguments(typed_stores, expected)
    if type(readiness_lease) is not OutcomeDiagnosticActivationReadinessLease:
        raise OutcomeDiagnosticResultStoreError("activation requires the canonical readiness lease")
    try:
        lease = readiness_lease.require_active()
    except (OutcomeDiagnosticReadinessError, OSError, ValueError) as exc:
        raise OutcomeDiagnosticResultStoreError("diagnostic readiness lease is not active") from exc
    if lease.snapshot.git_dirty or lease.snapshot.git_commit_sha != expected_git_commit:
        raise OutcomeDiagnosticResultStoreError("activation requires the exact authorized clean commit")
    if lease.snapshot.protocol.sha256 != expected.protocol_sha256:
        raise OutcomeDiagnosticResultStoreError("activation protocol differs from expected plan")

    stack = ExitStack()
    stack.__enter__()
    batch: OutcomeDiagnosticActivatedBatch | None = None
    try:
        root_fd = os.dup(lease.output_root_fd)
        stack.callback(os.close, root_fd)
        root_identity = _runtime_identity(root_fd)
        if root_identity != typed_stores[0].root_identity:
            raise OutcomeDiagnosticResultStoreError("diagnostic output root identity differs from prepared store")
        descriptors: list[dict[str, Any]] = []
        identities: list[dict[str, Any]] = []
        for store in typed_stores:
            family_fd = secure_fs.open_child_directory(root_fd, store.family_id)
            stack.callback(os.close, family_fd)
            run_fd = secure_fs.open_child_directory(family_fd, store.run_id)
            stack.callback(os.close, run_fd)
            namespaces_fd = secure_fs.open_child_directory(run_fd, "namespaces")
            stack.callback(os.close, namespaces_fd)
            descriptor: dict[str, Any] = {
                "root": root_fd,
                "family": family_fd,
                "run": run_fd,
                "namespaces": namespaces_fd,
                "namespace_fds": {},
                "records": {},
            }
            for condition in CONDITIONS:
                namespace_fd = secure_fs.open_child_directory(namespaces_fd, condition)
                stack.callback(os.close, namespace_fd)
                records_fd = secure_fs.open_child_directory(namespace_fd, "records")
                stack.callback(os.close, records_fd)
                descriptor["namespace_fds"][condition] = namespace_fd
                descriptor["records"][condition] = records_fd
            descriptors.append(descriptor)
            identities.append({
                "root": root_identity,
                "family": _runtime_identity(family_fd),
                "run": _runtime_identity(run_fd),
                "namespaces": _runtime_identity(namespaces_fd),
                **{f"records:{condition}": _runtime_identity(descriptor["records"][condition]) for condition in CONDITIONS},
            })
            observed_identities = identities[-1]
            expected_identities = {
                "root": store.root_identity,
                "family": store.family_identity,
                "run": store.run_identity,
                "namespaces": store.namespaces_parent_identity,
                **{
                    f"records:{condition}": store.record_namespace_identities[condition]
                    for condition in CONDITIONS
                },
            }
            if observed_identities != expected_identities:
                raise OutcomeDiagnosticResultStoreError(
                    f"prepared diagnostic descriptor identity changed: {store.family_id}"
                )
        # Ensure that the exact inert layout is what is being authorized and
        # that no orphan record exists before the marker commit.
        marker_existing = RUNTIME_ACTIVATION_MARKER_NAME in _entry_kinds(root_fd)
        for index, descriptor in enumerate(descriptors):
            _runtime_validate_inert_tree(
                {**descriptor, "root": root_fd},
                typed_stores[index],
                expected,
                root_identity=root_identity,
                marker_present=marker_existing,
            )
        if not marker_existing:
            for descriptor in descriptors:
                if any(_runtime_read_entries(descriptor["records"][condition]) for condition in CONDITIONS):
                    raise OutcomeDiagnosticResultStoreError("orphan diagnostic records require an activation marker")
        body = _runtime_marker_body(expected, typed_stores, root_identity)
        marker_bytes = _canonical(_runtime_marker_with_hash(body))
        _runtime_publish_marker(root_fd, marker_bytes)
        marker_fd = stack.enter_context(
            secure_fs.open_regular_file_at(root_fd, RUNTIME_ACTIVATION_MARKER_NAME)
        )
        marker_stat = os.fstat(marker_fd)
        marker_identity = (int(marker_stat.st_dev), int(marker_stat.st_ino))
        # Parse and enforce canonical marker bytes before granting the
        # capability; a raced or malformed marker never becomes usable.
        observed, marker_fingerprint = _runtime_record_snapshot(
            root_fd, RUNTIME_ACTIVATION_MARKER_NAME
        )
        if (
            marker_fingerprint[:2] != marker_identity
            or observed != marker_bytes
        ):
            raise OutcomeDiagnosticResultStoreError("diagnostic activation marker differs from authority")
        fingerprints: dict[tuple[str, str], tuple[int, int, int, int, int, str]] = {}
        units = tuple({item.unit_id: item for item in store.spec.units} for store in typed_stores)
        batch = OutcomeDiagnosticActivatedBatch(
            typed_stores, expected, lease, root_fd, tuple(descriptors), tuple(identities),
            marker_fd,
            marker_identity,
            marker_fingerprint,
            marker_bytes,
            fingerprints,
            units,
            _RUNTIME_TOKEN,
        )
        for index in range(len(typed_stores)):
            # Existing records are only valid when resuming an already marked
            # tree.  Track every identity before yielding to the driver.
            batch._validate_record_namespace(index, track_new=True)
        batch._require_live()
        yield batch
    finally:
        if batch is not None:
            try:
                batch._require_live(validate_records=True)
            finally:
                batch.close()
        stack.close()



__all__ = [
    "ACTIVATION_INTENT_NAME",
    "ACTIVATION_MARKER_NAME",
    "EXPECTED_FAMILY_UNIT_COUNT",
    "EXPECTED_NAMESPACE_UNIT_COUNT",
    "EXPECTED_TOTAL_UNIT_COUNT",
    "OutcomeDiagnosticExpectedPlan",
    "OutcomeDiagnosticNamespaceSpec",
    "OutcomeDiagnosticResultStore",
    "OutcomeDiagnosticResultStoreError",
    "OutcomeDiagnosticResultStorePlanError",
    "OutcomeDiagnosticResultStoreSpec",
    "OutcomeDiagnosticResumeBaseline",
    "OutcomeDiagnosticResumeRecordFingerprint",
    "OutcomeDiagnosticActivatedBatch",
    "RUNTIME_ACTIVATION_MARKER_NAME",
    "RUNTIME_ACTIVATION_SCHEMA_VERSION",
    "ROOT_METADATA_NAME",
    "SCHEMA_VERSION",
    "build_outcome_diagnostic_expected_plan",
    "load_outcome_diagnostic_result_stores",
    "prepare_outcome_diagnostic_result_stores",
    "activate_outcome_diagnostic_result_stores",
    "inspect_outcome_diagnostic_resume_tree_at",
    "validate_outcome_diagnostic_expected_plan",
]
