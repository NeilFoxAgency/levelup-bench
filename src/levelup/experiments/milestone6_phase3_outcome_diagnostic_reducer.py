"""Typed, development-only reduction for the Phase 3 outcome-group diagnostic.

The reducer is deliberately pure: it accepts the already validated plan,
compact model authority, and completed unit records in memory.  It never opens
a result store, calls an evaluator/oracle, or resolves a final-family path.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Collection, Mapping, Sequence

from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
    OutcomeDiagnosticModelArtifactAuthority,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    CONDITIONS,
    EXPECTED_MODEL_OWNERS,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_TUPLES,
    FAMILIES,
    OutcomeModelOwner,
    OutcomePlannedUnit,
    ValidatedOutcomePlan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_result_store import (
    _store_hashes,
)
from levelup.experiments.runner.records import PhaseAccounting, UnitRecord

FAILURE_CENSORING_BUDGET = 2_048
FAILURE_SENTINEL = 2_049
SUCCESS_TOLERANCE = Fraction(1, 20)
EXPECTED_UNITS = 5_760
EXPECTED_UNITS_PER_CONDITION = 2_880
EXPECTED_UNITS_PER_TUPLE = 240
EXPECTED_UNITS_PER_FAMILY = 40
EXPECTED_OWNER_CONSUMERS = 24
MATCHED_S_TUPLE = "lr0p01-e120-t1p2"
MATCHED_S_TRAINING_TUPLE = "lr0p01-e120"
REQUIRED_DIAGNOSTICS = (
    "model_trainable_parameters",
    "model_optimizer_steps",
    "model_forward_passes",
    "model_training_examples",
    "model_recurrent_steps",
)


class OutcomeDiagnosticReducerError(ValueError):
    """Raised when the complete diagnostic matrix or reduction is invalid."""


def _fail(message: str) -> None:
    raise OutcomeDiagnosticReducerError(message)


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticFamilyMetric:
    family_id: str
    units: int
    successes: int
    success_rate: Fraction
    median_restricted_interactions: Fraction


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticLockedFamilyMetric:
    family_id: str
    units: int
    successes: int
    success_rate: Fraction
    median_restricted_interactions: Fraction | None


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticLockedMetric:
    """Exact committed reference metric, allowing unavailable family medians."""

    condition_id: str
    tuple_id: str
    training_tuple_id: str
    family_metrics: tuple[OutcomeDiagnosticLockedFamilyMetric, ...]
    minimum_family_success_rate: Fraction
    worst_family_median_restricted_interactions: Fraction
    macro_average_family_median_restricted_interactions: Fraction
    optimizer_steps: int
    forward_passes: int
    recurrent_steps: int

    @property
    def family_success_rates(self) -> Mapping[str, Fraction]:
        return {row.family_id: row.success_rate for row in self.family_metrics}


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticCandidateMetric:
    condition_id: str
    tuple_id: str
    training_tuple_id: str
    family_metrics: tuple[OutcomeDiagnosticFamilyMetric, ...]
    minimum_family_success_rate: Fraction
    worst_family_median_restricted_interactions: Fraction
    macro_average_family_median_restricted_interactions: Fraction
    optimizer_steps: int
    forward_passes: int
    recurrent_steps: int

    @property
    def selected_tuple_id(self) -> str:
        return self.tuple_id

    @property
    def family_success_rates(self) -> Mapping[str, Fraction]:
        return {row.family_id: row.success_rate for row in self.family_metrics}


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticConditionSelection:
    condition_id: str
    candidates: tuple[OutcomeDiagnosticCandidateMetric, ...]
    best_minimum_family_success_rate: Fraction
    retained_tuple_ids: tuple[str, ...]
    selected: OutcomeDiagnosticCandidateMetric

    @property
    def matched_s(self) -> OutcomeDiagnosticCandidateMetric:
        return next(row for row in self.candidates if row.tuple_id == MATCHED_S_TUPLE)


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticSelectionResult:
    condition_selections: tuple[OutcomeDiagnosticConditionSelection, ...]
    final_family_access: bool = False

    def by_condition(self) -> dict[str, OutcomeDiagnosticCandidateMetric]:
        return {row.condition_id: row.selected for row in self.condition_selections}

    def condition_by_id(self) -> dict[str, OutcomeDiagnosticConditionSelection]:
        return {row.condition_id: row for row in self.condition_selections}

    @property
    def selections(self) -> tuple[OutcomeDiagnosticCandidateMetric, ...]:
        return tuple(row.selected for row in self.condition_selections)


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticCostSummary:
    unit_count: int
    family_counts: Mapping[str, int]
    condition_counts: Mapping[str, int]
    model_owner_count: int
    model_owner_consumer_count: int
    deduplicated_optimizer_steps: int
    deduplicated_forward_passes: int
    deduplicated_training_examples: int
    deduplicated_recurrent_steps: int

    @property
    def deduplicated_model_optimizer_steps(self) -> int:
        return self.deduplicated_optimizer_steps

    @property
    def deduplicated_model_forward_passes(self) -> int:
        return self.deduplicated_forward_passes

    @property
    def deduplicated_model_recurrent_steps(self) -> int:
        return self.deduplicated_recurrent_steps

    @property
    def deduplicated_model_training_examples(self) -> int:
        return self.deduplicated_training_examples


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticValidatedMatrix:
    records: tuple[UnitRecord, ...]
    cost: OutcomeDiagnosticCostSummary
    owner_diagnostics: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    final_family_access: bool = False

    @property
    def unit_count(self) -> int:
        return self.cost.unit_count

    @property
    def model_owner_count(self) -> int:
        return self.cost.model_owner_count


@dataclass(frozen=True, slots=True)
class OutcomeDiagnosticClaimResult:
    rp_classification: str
    pec_classification: str
    both_groups_robust_gain: bool
    both_groups_robust_harm: bool
    overall_inconclusive: bool
    possible_interaction_hypothesis: bool
    rp_selected_delta_vs_s: Fraction
    rp_matched_delta_vs_s: Fraction
    pec_selected_delta_vs_s: Fraction
    pec_matched_delta_vs_s: Fraction
    t_delta_vs_s: Fraction
    rp_family_drops_selected: tuple[tuple[str, Fraction], ...]
    rp_family_drops_matched: tuple[tuple[str, Fraction], ...]
    pec_family_drops_selected: tuple[tuple[str, Fraction], ...]
    pec_family_drops_matched: tuple[tuple[str, Fraction], ...]
    rp_selected_no_drop_gate: bool
    rp_matched_no_drop_gate: bool
    pec_selected_no_drop_gate: bool
    pec_matched_no_drop_gate: bool
    rp_robust_gain: bool = False
    rp_robust_harm: bool = False
    rp_inconclusive: bool = True
    pec_robust_gain: bool = False
    pec_robust_harm: bool = False
    pec_inconclusive: bool = True
    final_family_access: bool = False

    @property
    def robust_group_gain(self) -> bool:
        return self.both_groups_robust_gain

    @property
    def robust_group_harm(self) -> bool:
        return self.both_groups_robust_harm

    @property
    def inconclusive(self) -> bool:
        return self.overall_inconclusive


def _median(values: Sequence[int]) -> Fraction:
    if not values:
        _fail("candidate family has no records")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return Fraction(ordered[middle])
    return Fraction(ordered[middle - 1] + ordered[middle], 2)


def _tuple_numeric(tuple_id: str) -> tuple[Fraction, int, Fraction]:
    parts = tuple_id.split("-")
    if len(parts) != 3:
        _fail("malformed candidate tuple")
    try:
        rate = Fraction(parts[0][2:].replace("p", "."))
        epochs = int(parts[1][1:])
        temperature = Fraction(parts[2][1:].replace("p", "."))
    except (ValueError, ZeroDivisionError):
        _fail("malformed candidate tuple")
    if tuple_id not in EXPECTED_TUPLES or epochs not in (120, 180):
        _fail("candidate tuple is outside the frozen grid")
    return rate, epochs, temperature


def _restricted(record: UnitRecord) -> int:
    outcome = record.outcome
    search_episodes = record.accounting.search.episodes
    probe_actions = record.accounting.probes.actions
    search_actions = record.accounting.search.actions
    if outcome.success:
        value = outcome.first_optimum_adaptation_actions
        if (
            outcome.censored
            or outcome.censoring_budget is not None
            or outcome.censoring_reason is not None
            or outcome.first_optimum_episode is None
            or outcome.first_optimum_episode < 1
            or outcome.first_optimum_episode > search_episodes
            or value is None
            or type(value) is not int
            or value < probe_actions
            or value > probe_actions + search_actions
            or value > FAILURE_CENSORING_BUDGET
        ):
            _fail("successful outcome lacks typed first-hit actions")
        return value
    if (
        outcome.censored is not True
        or outcome.censoring_budget != FAILURE_CENSORING_BUDGET
        or outcome.censoring_reason != "fixed_endpoint"
        or outcome.first_optimum_episode is not None
        or outcome.first_optimum_adaptation_actions is not None
    ):
        _fail("failed outcome lacks fixed-endpoint censoring")
    return FAILURE_SENTINEL


def _authority_and_plan(
    plan: ValidatedOutcomePlan, authority: OutcomeDiagnosticModelArtifactAuthority
) -> tuple[dict[str, OutcomePlannedUnit], dict[str, OutcomeModelOwner], dict[str, object]]:
    if type(plan) is not ValidatedOutcomePlan:
        _fail("validated outcome plan is required")
    if type(authority) is not OutcomeDiagnosticModelArtifactAuthority:
        _fail("typed outcome model authority is required")
    body = plan.plan
    if (
        body.final_family_access
        or body.family_order != FAMILIES
        or body.replicates != (0, 1, 2, 3, 4)
        or body.condition_ids != CONDITIONS
        or body.candidate_tuple_ids != EXPECTED_TUPLES
        or len(body.units) != EXPECTED_UNITS
        or len(body.model_owners) != EXPECTED_MODEL_OWNERS
    ):
        _fail("outcome plan is not the exact development matrix")
    if (
        authority.final
        or authority.final_family_access
        or not authority.development_only
        or authority.authority_sha256 != authority.expected_authority_sha256
        or authority.plan_id != body.plan_id
        or authority.protocol_sha256 != body.protocol_sha256
        or tuple(authority.condition_ids) != CONDITIONS
        or len(authority.artifacts) != EXPECTED_MODEL_OWNERS
    ):
        _fail("outcome model authority is not bound to the frozen plan")
    units = {item.unit_id: item for item in body.units}
    owners = {item.owner_id: item for item in body.model_owners}
    if len(units) != EXPECTED_UNITS or len(owners) != EXPECTED_MODEL_OWNERS:
        _fail("outcome plan contains duplicate identities")
    if set(units) != {item.unit_id for item in body.units}:
        _fail("outcome unit identity matrix is malformed")
    if set(owners) != {row.owner_id for row in authority.artifacts}:
        _fail("model authority owner universe differs")
    if any(owner.trainable_parameters != EXPECTED_PARAMETER_COUNT for owner in owners.values()):
        _fail("model capacity differs from frozen 3,841 parameter rule")
    authority_rows = {row.owner_id: row for row in authority.artifacts}
    for owner_id, owner in owners.items():
        row = authority_rows.get(owner_id)
        if row is None or (
            row.view_id != owner.view_id
            or row.condition_id != owner.condition_id
            or row.heldout_family != owner.heldout_family
            or row.fold_id != owner.fold_id
            or row.replicate != owner.replicate
            or row.training_tuple_id != owner.training_tuple_id
            or row.model_seed != owner.model_seed
            or row.data_order_seed != next(
                view.data_order_seed for view in body.views if view.view_id == owner.view_id
            )
            or row.feature_mask_sha256 != owner.feature_mask_sha256
            or row.transformation_sha256 != owner.transformation_sha256
            or row.model_identity_sha256 != owner.model_identity_sha256
        ):
            _fail("model authority lineage differs from outcome plan")
    if any(sum(item.model_owner_id == owner_id for item in body.units) != EXPECTED_OWNER_CONSUMERS for owner_id in owners):
        _fail("each model owner must have exactly 24 consumers")
    return units, owners, authority_rows


def _diagnostics(record: UnitRecord, owner: OutcomeModelOwner) -> tuple[int, int, int, int, int]:
    if record.diagnostics.get("development_outcome_diagnostic") is not True:
        _fail("record is not marked as outcome diagnostic")
    if record.diagnostics.get("model_serialization_calls") != 1:
        _fail("model serialization diagnostic is not exactly one")
    values: list[int] = []
    for name in REQUIRED_DIAGNOSTICS:
        value = record.diagnostics.get(name)
        if type(value) is not int or value < 0:
            _fail(f"missing numeric model diagnostic: {name}")
        values.append(value)
    if (
        values[0] != EXPECTED_PARAMETER_COUNT
        or values[1] != owner.training_epochs
        or values[1] < 1
        or values[2] != values[1] * values[3]
        or values[3] < 1
        or values[4] != 0
    ):
        _fail("model training or recurrent diagnostics differ from authority")
    return tuple(values)  # type: ignore[return-value]


def validate_outcome_diagnostic_matrix(
    plan: ValidatedOutcomePlan,
    authority: OutcomeDiagnosticModelArtifactAuthority,
    records: Collection[UnitRecord],
) -> OutcomeDiagnosticValidatedMatrix:
    """Validate exactly 5,760 completed development records and deduplicate costs."""
    expected, owners, authority_rows = _authority_and_plan(plan, authority)
    materialized = tuple(records)
    if len(materialized) != EXPECTED_UNITS:
        _fail("diagnostic result collection must contain exactly 5,760 records")
    by_id: dict[str, UnitRecord] = {}
    owner_values: dict[str, tuple[int, int, int, int, int]] = {}
    family_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    store_identities = {
        family: _store_hashes(
            family,
            plan.plan.plan_id,
            plan.plan.protocol_sha256,
            tuple(item.unit_id for item in plan.plan.units if item.heldout_family == family),
        )
        for family in FAMILIES
    }
    for record in materialized:
        if type(record) is not UnitRecord or record.unit_id in by_id:
            _fail("diagnostic result collection contains a non-record or duplicate unit")
        planned = expected.get(record.unit_id)
        if planned is None:
            _fail("diagnostic result collection contains an extra unit")
        tuple_condition = f"{planned.condition_id}--{planned.tuple_id}"
        if (
            record.key.phase != "validation"
            or record.key.condition_id != tuple_condition
            or record.key.family_id != planned.heldout_family
            or record.key.task_id != planned.task_id
            or record.key.task_index != planned.task_index
            or record.key.replicate != planned.replicate
            or record.seeds.model_seed != planned.model_seed
            or record.seeds.environment_seed != planned.environment_seed
            or record.seeds.probe_seed != planned.probe_seed
            or record.seeds.search_seed != planned.search_seed
            or record.seeds.data_order_seed != planned.data_order_seed
            or record.exposure_manifest_sha256 != planned.exposure_manifest_sha256
            or record.shared_artifact is not None
            or record.shared_artifacts
            or record.candidate_generation_sha256 is None
        ):
            _fail("diagnostic unit/key/seed/shared-artifact identity differs")
        expected_config, expected_run = store_identities[planned.heldout_family]
        if record.run_id != expected_run or record.config_sha256 != expected_config:
            _fail("diagnostic record store identity differs from frozen family store")
        accounting = record.accounting
        if (
            accounting.training != PhaseAccounting()
            or record.outcome.evaluator_ran is not True
            or accounting.probes.actions != planned.probe_actions_per_task
            or accounting.search.actions < 1
            or accounting.probes.actions + accounting.search.actions > FAILURE_CENSORING_BUDGET
            or accounting.search.episodes < 1
            or accounting.search.episodes > planned.candidate_episodes_per_task
            or accounting.search.actions
            > accounting.search.episodes * planned.maximum_actions_per_candidate_episode
        ):
            _fail("diagnostic interaction accounting or evaluator contract differs")
        _restricted(record)
        owner = owners.get(planned.model_owner_id)
        if owner is None or owner.condition_id != planned.condition_id:
            _fail("diagnostic model owner identity differs")
        diagnostics = _diagnostics(record, owner)
        prior = owner_values.get(owner.owner_id)
        if prior is not None and prior != diagnostics:
            _fail("one model owner has inconsistent consumer diagnostics")
        owner_values[owner.owner_id] = diagnostics
        by_id[record.unit_id] = record
        family_counts[planned.heldout_family] += 1
        condition_counts[planned.condition_id] += 1
    if set(by_id) != set(expected):
        _fail("diagnostic result collection is missing a planned unit")
    if set(owner_values) != set(owners):
        _fail("diagnostic result collection does not cover all model owners")
    if any(family_counts[family] != 960 for family in FAMILIES):
        _fail("diagnostic family coverage is not exactly 960 units")
    if any(condition_counts[condition] != EXPECTED_UNITS_PER_CONDITION for condition in CONDITIONS):
        _fail("diagnostic condition coverage is not exactly 2,880 units")
    cost = OutcomeDiagnosticCostSummary(
        EXPECTED_UNITS,
        dict(family_counts),
        dict(condition_counts),
        len(owner_values),
        EXPECTED_UNITS,
        sum(value[1] for value in owner_values.values()),
        sum(value[2] for value in owner_values.values()),
        sum(value[3] for value in owner_values.values()),
        sum(value[4] for value in owner_values.values()),
    )
    owner_diagnostics = tuple(
        (owner_id, tuple(zip(REQUIRED_DIAGNOSTICS, owner_values[owner_id], strict=True)))
        for owner_id in sorted(owner_values)
    )
    return OutcomeDiagnosticValidatedMatrix(
        tuple(by_id[item_id] for item_id in expected), cost, owner_diagnostics, False
    )


def _select_condition(
    condition: str,
    records: Sequence[UnitRecord],
    plan_units: Mapping[str, OutcomePlannedUnit],
    owner_diagnostics: Mapping[str, tuple[int, int, int, int, int]],
) -> OutcomeDiagnosticConditionSelection:
    by_tuple: dict[str, list[UnitRecord]] = defaultdict(list)
    owner_by_tuple: dict[str, set[str]] = defaultdict(set)
    for record in records:
        planned = plan_units[record.unit_id]
        by_tuple[planned.tuple_id].append(record)
        owner_by_tuple[planned.tuple_id].add(planned.model_owner_id)
    candidates: list[OutcomeDiagnosticCandidateMetric] = []
    for tuple_id in EXPECTED_TUPLES:
        rows = by_tuple.get(tuple_id, [])
        if len(rows) != EXPECTED_UNITS_PER_TUPLE:
            _fail("diagnostic candidate tuple does not contain 240 units")
        family_metrics: list[OutcomeDiagnosticFamilyMetric] = []
        for family in FAMILIES:
            family_rows = [row for row in rows if row.key.family_id == family]
            if len(family_rows) != EXPECTED_UNITS_PER_FAMILY:
                _fail("diagnostic candidate family coverage is incomplete")
            successes = sum(row.outcome.success for row in family_rows)
            interactions = [_restricted(row) for row in family_rows]
            family_metrics.append(
                OutcomeDiagnosticFamilyMetric(
                    family, len(family_rows), successes, Fraction(successes, len(family_rows)), _median(interactions)
                )
            )
        owners = owner_by_tuple[tuple_id]
        training_tuple = tuple_id.rsplit("-t", 1)[0]
        if len(owners) != 30 or any(plan_units[row.unit_id].condition_id != condition for row in rows):
            _fail("diagnostic candidate owner matrix is incomplete")
        owner_costs = [owner_diagnostics[owner_id] for owner_id in owners]
        candidates.append(
            OutcomeDiagnosticCandidateMetric(
                condition,
                tuple_id,
                training_tuple,
                tuple(family_metrics),
                min(item.success_rate for item in family_metrics),
                max(item.median_restricted_interactions for item in family_metrics),
                sum((item.median_restricted_interactions for item in family_metrics), Fraction()) / len(FAMILIES),
                sum(item[1] for item in owner_costs),
                sum(item[2] for item in owner_costs),
                sum(item[4] for item in owner_costs),
            )
        )
    best = max(item.minimum_family_success_rate for item in candidates)
    retained = [item for item in candidates if best - item.minimum_family_success_rate <= SUCCESS_TOLERANCE]
    selected = min(retained, key=_selection_key)
    return OutcomeDiagnosticConditionSelection(condition, tuple(candidates), best, tuple(item.tuple_id for item in retained), selected)


def _selection_key(item: OutcomeDiagnosticCandidateMetric) -> tuple[object, ...]:
    return (
        item.worst_family_median_restricted_interactions,
        item.macro_average_family_median_restricted_interactions,
        item.optimizer_steps,
        item.forward_passes,
        item.recurrent_steps,
        _tuple_numeric(item.tuple_id),
    )


def select_outcome_diagnostic_tuples(
    plan: ValidatedOutcomePlan,
    authority: OutcomeDiagnosticModelArtifactAuthority,
    matrix: OutcomeDiagnosticValidatedMatrix,
    *,
    locked_s: OutcomeDiagnosticLockedMetric,
) -> OutcomeDiagnosticSelectionResult:
    """Select one tuple independently for each of RP and PEC."""
    canonical = validate_outcome_diagnostic_matrix(plan, authority, matrix.records)
    if canonical != matrix:
        _fail("validated diagnostic matrix differs from canonical reduction")
    _require_locked_metric(
        locked_s, "S-state-availability-listwise-optimum", MATCHED_S_TUPLE
    )
    expected = {item.unit_id: item for item in plan.plan.units}
    owner_diag = {owner: dict(entries) for owner, entries in matrix.owner_diagnostics}
    diag_tuples = {
        owner: (
            values["model_trainable_parameters"],
            values["model_optimizer_steps"],
            values["model_forward_passes"],
            values["model_training_examples"],
            values["model_recurrent_steps"],
        )
        for owner, values in owner_diag.items()
    }
    selections = tuple(
        _select_condition(
            condition,
            [record for record in matrix.records if expected[record.unit_id].condition_id == condition],
            expected,
            diag_tuples,
        )
        for condition in CONDITIONS
    )
    return OutcomeDiagnosticSelectionResult(selections, False)


def _require_candidate(value: object, condition: str) -> OutcomeDiagnosticCandidateMetric:
    if not isinstance(value, OutcomeDiagnosticCandidateMetric) or value.condition_id != condition:
        _fail("diagnostic selected metric is malformed")
    if (
        value.tuple_id not in EXPECTED_TUPLES
        or value.training_tuple_id != value.tuple_id.rsplit("-t", 1)[0]
        or tuple(row.family_id for row in value.family_metrics) != FAMILIES
        or any(
            row.units != EXPECTED_UNITS_PER_FAMILY
            or row.successes < 0
            or row.successes > row.units
            or row.success_rate != Fraction(row.successes, row.units)
            or row.median_restricted_interactions < 0
            or row.median_restricted_interactions > FAILURE_SENTINEL
            for row in value.family_metrics
        )
    ):
        _fail("diagnostic selected metric family universe is malformed")
    if (
        value.minimum_family_success_rate != min(row.success_rate for row in value.family_metrics)
        or value.worst_family_median_restricted_interactions
        != max(row.median_restricted_interactions for row in value.family_metrics)
        or value.macro_average_family_median_restricted_interactions
        != sum((row.median_restricted_interactions for row in value.family_metrics), Fraction()) / len(FAMILIES)
        or min(value.optimizer_steps, value.forward_passes, value.recurrent_steps) < 0
    ):
        _fail("diagnostic selected metric primary is inconsistent")
    return value


def _require_locked_metric(
    value: object, condition: str, tuple_id: str
) -> OutcomeDiagnosticLockedMetric:
    if not isinstance(value, OutcomeDiagnosticLockedMetric) or value.condition_id != condition or value.tuple_id != tuple_id:
        _fail("locked baseline metric identity is malformed")
    if value.training_tuple_id != tuple_id.rsplit("-t", 1)[0] or tuple(row.family_id for row in value.family_metrics) != FAMILIES:
        _fail("locked baseline family universe is malformed")
    if any(
        row.units != EXPECTED_UNITS_PER_FAMILY
        or row.successes < 0
        or row.successes > row.units
        or row.success_rate != Fraction(row.successes, row.units)
        or (
            row.median_restricted_interactions is not None
            and not 0 <= row.median_restricted_interactions <= FAILURE_SENTINEL
        )
        for row in value.family_metrics
    ):
        _fail("locked baseline family coverage is malformed")
    medians = tuple(
        row.median_restricted_interactions for row in value.family_metrics
    )
    if (
        value.minimum_family_success_rate != min(row.success_rate for row in value.family_metrics)
        or not 0 <= value.worst_family_median_restricted_interactions <= FAILURE_SENTINEL
        or not 0 <= value.macro_average_family_median_restricted_interactions <= FAILURE_SENTINEL
        or (
            all(item is not None for item in medians)
            and (
                value.worst_family_median_restricted_interactions != max(medians)
                or value.macro_average_family_median_restricted_interactions
                != sum(medians, Fraction()) / len(FAMILIES)
            )
        )
        or min(value.optimizer_steps, value.forward_passes, value.recurrent_steps) < 0
    ):
        _fail("locked baseline aggregate metric is malformed")
    return value


def validate_outcome_diagnostic_locked_metric(
    value: object, *, condition_id: str, tuple_id: str
) -> OutcomeDiagnosticLockedMetric:
    """Validate one exact committed development reference for publication."""

    return _require_locked_metric(value, condition_id, tuple_id)


def _classify(
    selected_delta: Fraction,
    matched_delta: Fraction,
    selected_drops: Sequence[Fraction],
    matched_drops: Sequence[Fraction],
) -> str:
    gain = selected_delta > SUCCESS_TOLERANCE and matched_delta > SUCCESS_TOLERANCE and all(
        drop <= SUCCESS_TOLERANCE for drop in (*selected_drops, *matched_drops)
    )
    harm = selected_delta < -SUCCESS_TOLERANCE and matched_delta < -SUCCESS_TOLERANCE
    if gain:
        return "robust_gain"
    if harm:
        return "robust_harm"
    return "inconclusive"


def evaluate_outcome_diagnostic_claims(
    selection: OutcomeDiagnosticSelectionResult,
    *,
    locked_s: OutcomeDiagnosticLockedMetric,
    locked_t: OutcomeDiagnosticLockedMetric,
) -> OutcomeDiagnosticClaimResult:
    """Evaluate only the predeclared exploratory gain/harm/interaction flags."""
    if type(selection) is not OutcomeDiagnosticSelectionResult or selection.final_family_access:
        _fail("diagnostic selection is not development-only")
    _require_locked_metric(
        locked_s, "S-state-availability-listwise-optimum", MATCHED_S_TUPLE
    )
    _require_locked_metric(locked_t, "T-markov-state-transition-listwise-optimum", "lr0p003-e120-t1p2")
    by_condition = selection.by_condition()
    if set(by_condition) != set(CONDITIONS):
        _fail("diagnostic selection does not cover both conditions")
    for condition in CONDITIONS:
        condition_selection = selection.condition_by_id().get(condition)
        if condition_selection is None or len(condition_selection.candidates) != 12:
            _fail("diagnostic condition selection does not contain exactly 12 candidates")
        if tuple(item.tuple_id for item in condition_selection.candidates) != EXPECTED_TUPLES:
            _fail("diagnostic candidate tuple universe is malformed")
        for candidate in condition_selection.candidates:
            _require_candidate(candidate, condition)
        best = max(
            item.minimum_family_success_rate
            for item in condition_selection.candidates
        )
        retained = tuple(
            item
            for item in condition_selection.candidates
            if best - item.minimum_family_success_rate <= SUCCESS_TOLERANCE
        )
        if (
            condition_selection.best_minimum_family_success_rate != best
            or condition_selection.retained_tuple_ids
            != tuple(item.tuple_id for item in retained)
            or condition_selection.selected != min(retained, key=_selection_key)
        ):
            _fail("diagnostic selection trace differs from the frozen rule")
    selected = [_require_candidate(by_condition[c], c) for c in CONDITIONS]
    matched = []
    for condition in CONDITIONS:
        candidates = selection.condition_by_id()[condition].candidates
        try:
            candidate = next(row for row in candidates if row.tuple_id == MATCHED_S_TUPLE)
        except StopIteration:
            _fail("diagnostic selection is missing the matched-S tuple")
        matched.append(_require_candidate(candidate, condition))
    def drops(metric: OutcomeDiagnosticCandidateMetric) -> tuple[tuple[str, Fraction], ...]:
        return tuple((family, locked_s.family_success_rates[family] - metric.family_success_rates[family]) for family in FAMILIES)
    selected_drops = [drops(item) for item in selected]
    matched_drops = [drops(item) for item in matched]
    deltas_selected = [item.minimum_family_success_rate - locked_s.minimum_family_success_rate for item in selected]
    deltas_matched = [item.minimum_family_success_rate - locked_s.minimum_family_success_rate for item in matched]
    classifications = [
        _classify(deltas_selected[i], deltas_matched[i], [d for _, d in selected_drops[i]], [d for _, d in matched_drops[i]])
        for i in range(2)
    ]
    no_drop = [
        all(d <= SUCCESS_TOLERANCE for _, d in selected_drops[i])
        for i in range(2)
    ]
    matched_no_drop = [
        all(d <= SUCCESS_TOLERANCE for _, d in matched_drops[i])
        for i in range(2)
    ]
    robust_gain = all(item == "robust_gain" for item in classifications)
    robust_harm = all(item == "robust_harm" for item in classifications)
    inconclusive = not robust_gain and not robust_harm
    t_delta = locked_t.minimum_family_success_rate - locked_s.minimum_family_success_rate
    possible_interaction = (
        all(item != "robust_harm" for item in classifications)
        and locked_t.minimum_family_success_rate < locked_s.minimum_family_success_rate
    )
    return OutcomeDiagnosticClaimResult(
        classifications[0],
        classifications[1],
        robust_gain,
        robust_harm,
        inconclusive,
        possible_interaction,
        deltas_selected[0],
        deltas_matched[0],
        deltas_selected[1],
        deltas_matched[1],
        t_delta,
        selected_drops[0],
        matched_drops[0],
        selected_drops[1],
        matched_drops[1],
        no_drop[0],
        matched_no_drop[0],
        no_drop[1],
        matched_no_drop[1],
        classifications[0] == "robust_gain",
        classifications[0] == "robust_harm",
        classifications[0] == "inconclusive",
        classifications[1] == "robust_gain",
        classifications[1] == "robust_harm",
        classifications[1] == "inconclusive",
        False,
    )


def reduce_outcome_group_diagnostic(
    plan: ValidatedOutcomePlan,
    authority: OutcomeDiagnosticModelArtifactAuthority,
    records: Collection[UnitRecord],
    *,
    locked_s: OutcomeDiagnosticLockedMetric,
    locked_t: OutcomeDiagnosticLockedMetric,
) -> tuple[OutcomeDiagnosticValidatedMatrix, OutcomeDiagnosticSelectionResult, OutcomeDiagnosticClaimResult]:
    matrix = validate_outcome_diagnostic_matrix(plan, authority, records)
    selection = select_outcome_diagnostic_tuples(plan, authority, matrix, locked_s=locked_s)
    return matrix, selection, evaluate_outcome_diagnostic_claims(selection, locked_s=locked_s, locked_t=locked_t)


__all__ = [
    "FAILURE_CENSORING_BUDGET",
    "FAILURE_SENTINEL",
    "CONDITIONS",
    "FAMILIES",
    "EXPECTED_TUPLES",
    "MATCHED_S_TUPLE",
    "OutcomeDiagnosticCandidateMetric",
    "OutcomeDiagnosticClaimResult",
    "OutcomeDiagnosticConditionSelection",
    "OutcomeDiagnosticCostSummary",
    "OutcomeDiagnosticFamilyMetric",
    "OutcomeDiagnosticLockedFamilyMetric",
    "OutcomeDiagnosticLockedMetric",
    "OutcomeDiagnosticReducerError",
    "OutcomeDiagnosticSelectionResult",
    "OutcomeDiagnosticValidatedMatrix",
    "evaluate_outcome_diagnostic_claims",
    "reduce_outcome_group_diagnostic",
    "select_outcome_diagnostic_tuples",
    "validate_outcome_diagnostic_matrix",
    "validate_outcome_diagnostic_locked_metric",
]
