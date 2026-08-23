"""Structural validation and cost reduction for the frozen Phase 3 matrix.

This module is deliberately not a metric reducer.  It accepts completed unit
records only after execution and checks that they are exactly the units in the
opaque Phase 3 plan.  It validates the typed first-hit/censoring contract and
deduplicates model-preparation accounting by model owner (a model is consumed
by 24 task/temperature records).  No performance value is read for ranking or
selection and no result store is opened here.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Collection

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt

from levelup.experiments.milestone6_phase3_execution_models import (
    EXPECTED_PHASE3_MODEL_AUTHORITY_SHA256,
)
from levelup.experiments.milestone6_phase3_model_authority import Phase3ModelArtifactAuthority
from levelup.experiments.milestone6_phase3_models import (
    H0_CONDITION,
    H4_CONDITION,
    H4_SHUFFLED_CONDITION,
    HISTORY_PARAMETERS,
    S_CONDITION,
    S_PARAMETERS,
)
from levelup.experiments.milestone6_phase3_plan import (
    FAMILIES,
    REPLICATES,
    TRAINING_TUPLE_IDS,
    Phase3ModelOwner,
    Phase3PlannedUnit,
    ValidatedPhase3Plan,
    _plan_body,
)
from levelup.experiments.milestone6_phase3_result_store import (
    build_phase3_expected_plan,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import (
    PhaseAccounting,
    SharedArtifactReference,
    UnitRecord,
)

EXPECTED_UNIT_COUNT = 11_520
EXPECTED_MODEL_OWNER_COUNT = 480
EXPECTED_OWNER_CONSUMER_COUNT = 24
FAILURE_CENSORING_BUDGET = 2_048
REQUIRED_MODEL_DIAGNOSTICS = (
    "model_trainable_parameters",
    "model_optimizer_steps",
    "model_forward_passes",
    "model_recurrent_steps",
    "model_training_examples",
)
_CONDITIONS = (S_CONDITION, H0_CONDITION, H4_CONDITION, H4_SHUFFLED_CONDITION)


class Phase3ReducerError(ValueError):
    """Raised when a Phase 3 result matrix is incomplete or structurally invalid."""


class Phase3CostSummary(BaseModel):
    """Counts and preparation costs after deduplicating the 480 model owners."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_count: StrictInt = Field(ge=0)
    family_counts: dict[str, StrictInt]
    condition_counts: dict[str, StrictInt]
    model_owner_count: StrictInt = Field(ge=0)
    model_owner_consumer_count: StrictInt = Field(ge=0)
    deduplicated_model_trainable_parameters: StrictInt = Field(ge=0)
    deduplicated_model_optimizer_steps: StrictInt = Field(ge=0)
    deduplicated_model_forward_passes: StrictInt = Field(ge=0)
    deduplicated_model_training_examples: StrictInt = Field(ge=0)
    deduplicated_model_recurrent_steps: StrictInt = Field(ge=0)

    # Explicit names make it difficult for callers to mistake this for a
    # per-unit training total.
    @property
    def deduplicated_preparation_optimizer_steps(self) -> int:
        return self.deduplicated_model_optimizer_steps

    @property
    def deduplicated_preparation_forward_passes(self) -> int:
        return self.deduplicated_model_forward_passes


class Phase3ControlSummary(BaseModel):
    """Aggregate held-out search coverage for the sequence-order control."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shuffled_unit_count: StrictInt = Field(ge=0)
    eligible_windows: StrictInt = Field(ge=0)
    map_nonidentity_windows: StrictInt = Field(ge=0)
    effective_tensor_changed_windows: StrictInt = Field(ge=0)
    duplicate_vector_no_effect_windows: StrictInt = Field(ge=0)
    unchanged_short_windows: StrictInt = Field(ge=0)
    effective_change_fraction: StrictFloat = Field(ge=0.0, le=1.0)
    heldout_search_claim_eligible: StrictBool


class Phase3ValidatedMatrix(BaseModel):
    """Opaque structural matrix handed to a later, separately frozen reducer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    records: tuple[UnitRecord, ...]
    cost: Phase3CostSummary
    control: Phase3ControlSummary
    owner_diagnostics: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]

    @property
    def unit_count(self) -> int:
        return self.cost.unit_count

    @property
    def model_owner_count(self) -> int:
        return self.cost.model_owner_count


def _fail(message: str) -> None:
    raise Phase3ReducerError(message)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _authority_and_plan(
    plan: ValidatedPhase3Plan, authority: Phase3ModelArtifactAuthority
) -> tuple[
    dict[str, Phase3PlannedUnit],
    dict[str, Phase3ModelOwner],
    dict[str, object],
]:
    if type(plan) is not ValidatedPhase3Plan:
        _fail("Phase 3 reducer requires the canonical validated plan")
    if type(authority) is not Phase3ModelArtifactAuthority:
        _fail("Phase 3 reducer requires the canonical model authority")
    if (
        authority.authority_sha256 != authority.expected_authority_sha256
        or authority.authority_sha256 != EXPECTED_PHASE3_MODEL_AUTHORITY_SHA256
        or not authority.execution_authorized
        or not authority.development_only
        or authority.final
        or authority.final_family_accessed
    ):
        _fail("Phase 3 model authority is not the published development authority")
    body = plan.plan
    if body.final_family_access or body.family_order != FAMILIES or body.replicates != REPLICATES:
        _fail("Phase 3 plan permits final access or has a changed family matrix")
    if (
        body.condition_ids != _CONDITIONS
        or len(body.candidate_tuple_ids) != 12
        or len(set(body.candidate_tuple_ids)) != 12
        or any(
            tuple_id.rsplit("-t", 1)[0] not in TRAINING_TUPLE_IDS
            or tuple_id.rsplit("-t", 1)[-1] not in {"0p6", "0p9", "1p2"}
            for tuple_id in body.candidate_tuple_ids
        )
    ):
        _fail("Phase 3 plan condition or tuple universe differs")
    if len(body.units) != EXPECTED_UNIT_COUNT or len(body.model_owners) != EXPECTED_MODEL_OWNER_COUNT:
        _fail("Phase 3 plan counts differ from the frozen matrix")
    if (
        body.plan_id != authority.plan_id
        or body.protocol_sha256 != authority.protocol_sha256
        or {owner.owner_id for owner in body.model_owners} != set(authority.owner_ids)
    ):
        _fail("Phase 3 plan and model authority lineage differs")
    mapping = [(item.unit.unit_id, item.model_owner_id) for item in body.units]
    if (
        _sha256_json(_plan_body(body)) != body.plan_id
        or _sha256_json(mapping) != authority.unit_owner_mapping_sha256
    ):
        _fail("Phase 3 plan body or unit-owner mapping differs from authority")
    if len(authority.models) != EXPECTED_MODEL_OWNER_COUNT or {
        row.owner_id for row in authority.models
    } != set(authority.owner_ids):
        _fail("Phase 3 authority model-owner matrix is incomplete")
    units: dict[str, Phase3PlannedUnit] = {}
    for item in body.units:
        if item.unit.unit_id in units:
            _fail("Phase 3 plan contains duplicate unit IDs")
        units[item.unit.unit_id] = item
    owners: dict[str, Phase3ModelOwner] = {}
    for owner in body.model_owners:
        if owner.owner_id in owners:
            _fail("Phase 3 plan contains duplicate model owners")
        if owner.condition_id not in _CONDITIONS or owner.heldout_family not in FAMILIES:
            _fail("Phase 3 model owner is outside the frozen universe")
        if owner.training_tuple_id not in TRAINING_TUPLE_IDS:
            _fail("Phase 3 model owner has an unknown training tuple")
        owners[owner.owner_id] = owner
    if {item.model_owner_id for item in body.units} != set(owners):
        _fail("Phase 3 unit-to-owner references are incomplete")
    return units, owners, {row.owner_id: row for row in authority.models}


def _diagnostics(record: UnitRecord) -> tuple[int, int, int, int, int]:
    values: list[int] = []
    for name in REQUIRED_MODEL_DIAGNOSTICS:
        value = record.diagnostics.get(name)
        if type(value) is not int or value < 0:
            _fail(f"Phase 3 record is missing numeric diagnostic {name}")
        values.append(value)
    if values[0] < 1 or values[1] < 1 or values[2] < 1 or values[4] < 1:
        _fail("Phase 3 model diagnostics contain non-positive report fields")
    if values[2] != values[1] * values[4]:
        _fail("Phase 3 forward-pass diagnostic is inconsistent with training")
    return tuple(values)  # type: ignore[return-value]


def _nonnegative_diagnostic(record: UnitRecord, name: str) -> int:
    value = record.diagnostics.get(name)
    if type(value) is not int or value < 0:
        _fail(f"Phase 3 record is missing numeric diagnostic {name}")
    return value


def _check_outcome(record: UnitRecord, planned: Phase3PlannedUnit) -> tuple[int, ...] | None:
    outcome = record.outcome
    if record.candidate_generation_sha256 is None:
        _fail("every Phase 3 unit must carry a candidate-generation hash")
    if record.accounting.training != PhaseAccounting():
        _fail("Phase 3 units must not charge unit-local training")
    probe_actions = record.accounting.probes.actions
    search_actions = record.accounting.search.actions
    search_episodes = record.accounting.search.episodes
    if (
        outcome.evaluator_ran is not True
        or probe_actions != 64
        or search_actions < 1
        or search_actions > FAILURE_CENSORING_BUDGET - probe_actions
        or probe_actions + search_actions > FAILURE_CENSORING_BUDGET
        or search_episodes < 1
        or search_episodes > 150
    ):
        _fail("Phase 3 outcome evaluator or fixed interaction accounting differs")
    if outcome.success:
        if (
            outcome.censored
            or outcome.censoring_budget is not None
            or outcome.censoring_reason is not None
            or outcome.first_optimum_episode is None
            or outcome.first_optimum_adaptation_actions is None
            or outcome.first_optimum_episode < 1
            or outcome.first_optimum_episode > search_episodes
            or outcome.first_optimum_adaptation_actions < probe_actions
            or outcome.first_optimum_adaptation_actions > probe_actions + search_actions
            or outcome.first_optimum_adaptation_actions > FAILURE_CENSORING_BUDGET
        ):
            _fail("successful Phase 3 outcome lacks typed first-hit semantics")
    elif (
        not outcome.censored
        or outcome.censoring_budget != FAILURE_CENSORING_BUDGET
        or outcome.censoring_reason != "fixed_endpoint"
        or outcome.first_optimum_episode is not None
        or outcome.first_optimum_adaptation_actions is not None
    ):
        _fail("failed Phase 3 outcome does not use fixed-endpoint censoring")
    shuffle_names = (
        "history_shuffle_eligible_windows",
        "history_shuffle_map_nonidentity_windows",
        "history_shuffle_effective_tensor_changed_windows",
        "history_shuffle_duplicate_vector_no_effect_windows",
        "history_shuffle_unchanged_short_windows",
    )
    if planned.base_condition_id != H4_SHUFFLED_CONDITION:
        if record.history_shuffle_permutation_map_sha256 is not None:
            _fail("non-shuffled Phase 3 unit carries a shuffle-map digest")
        if record.diagnostics.get("history_shuffle_claim_eligible") is not None or any(
            _nonnegative_diagnostic(record, name) != 0 for name in shuffle_names
        ):
            _fail("non-shuffled Phase 3 unit carries shuffle coverage")
        return None
    if record.history_shuffle_permutation_map_sha256 is None:
        _fail("H4-shuffled unit is missing its search permutation-map digest")
    values = tuple(_nonnegative_diagnostic(record, name) for name in shuffle_names)
    eligible, map_nonidentity, effective, duplicate_no_effect, _short = values
    if (
        map_nonidentity != eligible
        or effective + duplicate_no_effect != eligible
        or effective > map_nonidentity
    ):
        _fail("H4-shuffled coverage counters are internally inconsistent")
    expected_claim = eligible > 0 and effective / eligible >= 0.80
    if record.diagnostics.get("history_shuffle_claim_eligible") is not expected_claim:
        _fail("H4-shuffled per-unit coverage claim flag differs from its counters")
    return values


def validate_phase3_matrix(
    plan: ValidatedPhase3Plan,
    authority: Phase3ModelArtifactAuthority,
    records: Collection[UnitRecord],
) -> Phase3ValidatedMatrix:
    """Validate the complete 11,520-record matrix and deduplicate model costs."""

    expected, owners, authority_rows = _authority_and_plan(plan, authority)
    store_plan = build_phase3_expected_plan(plan, authority)
    store_by_family = {store.family_id: store for store in store_plan.stores}
    materialized = tuple(records)
    if len(materialized) != EXPECTED_UNIT_COUNT:
        _fail("Phase 3 result collection must contain exactly 11,520 records")
    by_id: dict[str, UnitRecord] = {}
    owner_values: dict[str, tuple[int, int, int, int, int]] = {}
    shuffled_totals = [0, 0, 0, 0, 0]
    shuffled_units = 0
    family_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    for record in materialized:
        if type(record) is not UnitRecord:
            _fail("Phase 3 result collection contains a non-UnitRecord")
        if record.unit_id in by_id:
            _fail("Phase 3 result collection contains duplicate unit IDs")
        planned = expected.get(record.unit_id)
        if planned is None:
            _fail("Phase 3 result collection contains an extra unit")
        if (
            record.key != planned.unit.key
            or record.seeds != planned.unit.seeds
            or record.exposure_manifest_sha256 != planned.unit.exposure_manifest_sha256
            or record.key.phase != "validation"
            or record.key.family_id != planned.heldout_family
        ):
            _fail("Phase 3 record identity differs from its frozen planned unit")
        store_spec = store_by_family[planned.heldout_family]
        if (
            record.run_id != store_spec.run_id
            or record.config_sha256 != store_spec.store_config_sha256
        ):
            _fail("Phase 3 record run/spec identity differs from its family store")
        by_id[record.unit_id] = record
        shuffle_values = _check_outcome(record, planned)
        if shuffle_values is not None:
            shuffled_units += 1
            for index, value in enumerate(shuffle_values):
                shuffled_totals[index] += value
        diagnostics = _diagnostics(record)
        owner = owners.get(planned.model_owner_id)
        if owner is None:
            _fail("Phase 3 unit references an unknown model owner")
        authority_row = authority_rows[owner.owner_id]
        reference = record.shared_artifact
        if (
            type(reference) is not SharedArtifactReference
            or record.shared_artifacts
            or reference.key_id != authority_row.key_id
            or reference.artifact_id != authority_row.artifact_id
            or reference.cost_id != authority_row.cost_id
        ):
            _fail("Phase 3 record shared-model reference differs from authority")
        expected_parameters = S_PARAMETERS if owner.condition_id == S_CONDITION else HISTORY_PARAMETERS
        if diagnostics[0] != expected_parameters or diagnostics[1] != owner.training_epochs:
            _fail("Phase 3 model diagnostics differ from the frozen owner")
        previous = owner_values.get(owner.owner_id)
        if previous is not None and previous != diagnostics:
            _fail("model diagnostics differ across one owner's consumers")
        owner_values[owner.owner_id] = diagnostics
        family_counts[record.key.family_id] += 1
        condition_counts[planned.base_condition_id] += 1
    if set(by_id) != set(expected):
        _fail("Phase 3 result collection is missing one or more planned units")
    if set(owner_values) != set(owners):
        _fail("Phase 3 result collection does not cover every model owner")
    owner_consumers = Counter(expected[item_id].model_owner_id for item_id in by_id)
    if any(count != EXPECTED_OWNER_CONSUMER_COUNT for count in owner_consumers.values()):
        _fail("each Phase 3 model owner must have exactly 24 consumers")
    if any(family_counts[family] != 1_920 for family in FAMILIES) or len(family_counts) != len(FAMILIES):
        _fail("Phase 3 family coverage is not six families by 1,920 units")
    if any(condition_counts[condition] != 2_880 for condition in _CONDITIONS):
        _fail("Phase 3 condition coverage differs from the frozen matrix")
    cost = Phase3CostSummary(
        unit_count=len(by_id),
        family_counts=dict(family_counts),
        condition_counts=dict(condition_counts),
        model_owner_count=len(owner_values),
        model_owner_consumer_count=sum(owner_consumers.values()),
        deduplicated_model_trainable_parameters=sum(values[0] for values in owner_values.values()),
        deduplicated_model_optimizer_steps=sum(values[1] for values in owner_values.values()),
        deduplicated_model_forward_passes=sum(values[2] for values in owner_values.values()),
        deduplicated_model_training_examples=sum(values[4] for values in owner_values.values()),
        deduplicated_model_recurrent_steps=sum(values[3] for values in owner_values.values()),
    )
    owner_diagnostics = tuple(
        (owner_id, tuple(zip(REQUIRED_MODEL_DIAGNOSTICS, owner_values[owner_id], strict=True)))
        for owner_id in sorted(owner_values)
    )
    eligible, map_nonidentity, effective, duplicate_no_effect, short = shuffled_totals
    fraction = effective / eligible if eligible else 1.0
    control = Phase3ControlSummary(
        shuffled_unit_count=shuffled_units,
        eligible_windows=eligible,
        map_nonidentity_windows=map_nonidentity,
        effective_tensor_changed_windows=effective,
        duplicate_vector_no_effect_windows=duplicate_no_effect,
        unchanged_short_windows=short,
        effective_change_fraction=fraction,
        heldout_search_claim_eligible=eligible > 0 and fraction >= 0.80,
    )
    return Phase3ValidatedMatrix(
        records=tuple(by_id[item_id] for item_id in expected),
        cost=cost,
        control=control,
        owner_diagnostics=owner_diagnostics,
    )


def reduce_phase3_records(
    plan: ValidatedPhase3Plan,
    authority: Phase3ModelArtifactAuthority,
    records: Collection[UnitRecord],
) -> Phase3ValidatedMatrix:
    """Compatibility spelling for callers that treat this as a reducer."""

    return validate_phase3_matrix(plan, authority, records)


__all__ = [
    "EXPECTED_UNIT_COUNT",
    "FAILURE_CENSORING_BUDGET",
    "Phase3CostSummary",
    "Phase3ControlSummary",
    "Phase3ReducerError",
    "Phase3ValidatedMatrix",
    "REQUIRED_MODEL_DIAGNOSTICS",
    "reduce_phase3_records",
    "validate_phase3_matrix",
]
