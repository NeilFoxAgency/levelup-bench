"""Typed, development-only result-store partition for Milestone 6 Phase 3.

The Phase 3 logical plan is deliberately *not* represented by a synthetic
``ExperimentConfig``.  Its exposure hashes are part of the already-authorized
plan and were produced by the Phase 3 authority.  This module therefore only
partitions an opaque :class:`ValidatedPhase3Plan` into six result-store
specifications; it never calls ``plan_expected_units`` and never opens or
writes a result namespace.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

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
from levelup.experiments.runner.config import canonical_json_bytes

SCHEMA_VERSION = "milestone6.phase3.result-store-plan.v1"
EXPECTED_PHASE3_MODEL_AUTHORITY_SHA256 = (
    "8771eb52433faf15d6e5e935902a5c935526ec0e6b8e34621c3d6a922aea1a52"
)
EXPECTED_FAMILY_UNIT_COUNT = 1_920
EXPECTED_TOTAL_UNIT_COUNT = 11_520
_CONSTRUCTION_TOKEN = object()


class Phase3ResultStorePlanError(ValueError):
    """Raised when a result-store partition is not the frozen plan."""


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


# A descriptive alias keeps call sites readable when they only need the six
# family store specifications.
build_phase3_result_store_plan = build_phase3_expected_plan


__all__ = [
    "EXPECTED_FAMILY_UNIT_COUNT",
    "EXPECTED_PHASE3_MODEL_AUTHORITY_SHA256",
    "EXPECTED_TOTAL_UNIT_COUNT",
    "Phase3ExpectedPlan",
    "Phase3ResultStorePlanError",
    "Phase3ResultStoreSpec",
    "SCHEMA_VERSION",
    "build_phase3_expected_plan",
    "build_phase3_result_store_plan",
    "validate_phase3_expected_plan",
]
