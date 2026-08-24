"""Pure Phase 3 development-matrix selection and claim reduction.

The execution and result-store boundaries deliberately stop at
``Phase3ValidatedMatrix``.  This module is the next, in-memory boundary: it
accepts only the canonical plan, model authority, and validated matrix and
reduces the four new representation conditions.  It does not open a path,
activate a store, import an environment, call an evaluator/oracle, or inspect
any final-family artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

from levelup.experiments.milestone6_phase3_execution_models import (
    EXPECTED_PHASE3_MODEL_AUTHORITY_SHA256,
)
from levelup.experiments.milestone6_phase3_model_authority import (
    Phase3ModelArtifactAuthority,
)
from levelup.experiments.milestone6_phase3_models import (
    H0_CONDITION,
    H4_CONDITION,
    H4_SHUFFLED_CONDITION,
    S_CONDITION,
)
from levelup.experiments.milestone6_phase3_plan import (
    FAMILIES,
    Phase3ModelOwner,
    ValidatedPhase3Plan,
)
from levelup.experiments.milestone6_phase3_reducer import (
    FAILURE_CENSORING_BUDGET,
    Phase3ValidatedMatrix,
    validate_phase3_matrix,
)

NEW_CONDITIONS = (S_CONDITION, H0_CONDITION, H4_CONDITION, H4_SHUFFLED_CONDITION)
EXPECTED_TUPLES = (
    "lr0p003-e120-t0p6",
    "lr0p003-e120-t0p9",
    "lr0p003-e120-t1p2",
    "lr0p003-e180-t0p6",
    "lr0p003-e180-t0p9",
    "lr0p003-e180-t1p2",
    "lr0p01-e120-t0p6",
    "lr0p01-e120-t0p9",
    "lr0p01-e120-t1p2",
    "lr0p01-e180-t0p6",
    "lr0p01-e180-t0p9",
    "lr0p01-e180-t1p2",
)
EXPECTED_TRAINING_TUPLES = (
    "lr0p003-e120",
    "lr0p003-e180",
    "lr0p01-e120",
    "lr0p01-e180",
)
EXPECTED_UNITS_PER_CONDITION = 2_880
EXPECTED_UNITS_PER_TUPLE = 240
EXPECTED_UNITS_PER_FAMILY_TUPLE = 40
EXPECTED_OWNERS_PER_TRAINING_TUPLE = 30
FAILURE_SENTINEL = 2_049
SUCCESS_TOLERANCE = Fraction(1, 20)
_BASE_CONDITION_IDS = {
    S_CONDITION,
    H0_CONDITION,
    H4_CONDITION,
    H4_SHUFFLED_CONDITION,
}


class Phase3SelectionError(ValueError):
    """Raised when a canonical matrix cannot be selected or reduced."""


def _fail(message: str) -> None:
    raise Phase3SelectionError(message)


def _tuple_numeric(tuple_id: str) -> tuple[Fraction, int, Fraction]:
    """Decode the frozen tuple identity without binary floating-point math."""

    parts = tuple_id.split("-")
    if len(parts) != 3 or parts[0] not in {"lr0p003", "lr0p01"}:
        _fail("unknown Phase 3 candidate tuple")
    try:
        learning_rate = Fraction(parts[0][2:].replace("p", "."))
        epochs = int(parts[1][1:])
        temperature = Fraction(parts[2][1:].replace("p", "."))
    except (ValueError, ZeroDivisionError):
        _fail("malformed Phase 3 candidate tuple")
    if tuple_id not in EXPECTED_TUPLES or epochs not in (120, 180):
        _fail("candidate tuple is outside the frozen Phase 3 universe")
    return learning_rate, epochs, temperature


@dataclass(frozen=True, slots=True)
class Phase3FamilyMetric:
    family_id: str
    units: int
    successes: int
    success_rate: Fraction
    median_restricted_interactions: Fraction


@dataclass(frozen=True, slots=True)
class Phase3SelectedMetric:
    """Selection quantities for one already-selected condition/tuple.

    The family rates are retained so the B2-vs-H4 advancement gate can be
    evaluated without re-reading records.  Anchor metrics use this exact type;
    callers cannot supply an untyped float summary accidentally.
    """

    condition_id: str
    tuple_id: str
    training_tuple_id: str
    family_metrics: tuple[Phase3FamilyMetric, ...]
    minimum_family_success_rate: Fraction
    worst_family_median_restricted_interactions: Fraction
    macro_average_family_median_restricted_interactions: Fraction
    optimizer_steps: int
    forward_passes: int
    recurrent_steps: int
    heldout_shuffle_claim_eligible: bool | None = None
    training_shuffle_claim_eligible: bool | None = None

    @property
    def family_success_rates(self) -> Mapping[str, Fraction]:
        return {item.family_id: item.success_rate for item in self.family_metrics}


@dataclass(frozen=True, slots=True)
class Phase3ConditionSelection:
    """All frozen candidates and the selection trace for one condition."""

    condition_id: str
    candidates: tuple[Phase3SelectedMetric, ...]
    best_minimum_family_success_rate: Fraction
    retained_tuple_ids: tuple[str, ...]
    selected: Phase3SelectedMetric


@dataclass(frozen=True, slots=True)
class Phase3SelectionResult:
    condition_selections: tuple[Phase3ConditionSelection, ...]
    final_family_access: bool = False

    @property
    def selections(self) -> tuple[Phase3SelectedMetric, ...]:
        return tuple(item.selected for item in self.condition_selections)

    def by_condition(self) -> dict[str, Phase3SelectedMetric]:
        return {item.condition_id: item for item in self.selections}

    def condition_by_id(self) -> dict[str, Phase3ConditionSelection]:
        return {item.condition_id: item for item in self.condition_selections}


@dataclass(frozen=True, slots=True)
class Phase3ClaimResult:
    transition_claim: bool
    history_access_claim: bool
    sequence_order_claim: bool
    advancement_to_paired_objectives: bool
    training_shuffle_claim_eligible: bool
    heldout_shuffle_claim_eligible: bool
    transition_gain: Fraction
    history_gain_over_t: Fraction
    history_gain_over_h0: Fraction
    sequence_order_gain: Fraction
    b2_minus_h4_family_success_drops: tuple[tuple[str, Fraction], ...]
    b2_minus_h4_minimum_family_success: Fraction
    final_family_access: bool = False


def _validate_inputs(
    plan: ValidatedPhase3Plan,
    authority: Phase3ModelArtifactAuthority,
    matrix: Phase3ValidatedMatrix,
) -> tuple[dict[str, Any], dict[str, Phase3ModelOwner], dict[str, tuple[str, int, int, int]]]:
    if type(plan) is not ValidatedPhase3Plan:
        _fail("Phase 3 selector requires the canonical validated plan")
    if type(authority) is not Phase3ModelArtifactAuthority:
        _fail("Phase 3 selector requires the canonical model authority")
    if type(matrix) is not Phase3ValidatedMatrix:
        _fail("Phase 3 selector requires the canonical validated matrix")
    body = plan.plan
    if body.final_family_access or body.family_order != FAMILIES:
        _fail("Phase 3 selection cannot access final families")
    if body.condition_ids != NEW_CONDITIONS:
        _fail("Phase 3 plan condition universe differs")
    if tuple(body.candidate_tuple_ids) != EXPECTED_TUPLES:
        _fail("Phase 3 plan candidate tuple universe differs")
    if (
        authority.final
        or authority.final_family_accessed
        or not authority.development_only
        or authority.authority_sha256 != authority.expected_authority_sha256
        or authority.authority_sha256 != EXPECTED_PHASE3_MODEL_AUTHORITY_SHA256
    ):
        _fail("Phase 3 authority is not development-only")
    if authority.plan_id != body.plan_id or authority.family_order != FAMILIES:
        _fail("Phase 3 authority and plan lineage differs")
    # Phase3ValidatedMatrix is publicly constructible, so its class alone is
    # not proof that the structural reducer produced it. Re-run the complete
    # validation and require byte-for-byte-equivalent typed content before any
    # performance field enters selection.
    try:
        canonical_matrix = validate_phase3_matrix(plan, authority, matrix.records)
    except (TypeError, ValueError) as exc:
        raise Phase3SelectionError(
            "Phase 3 selector input failed canonical structural validation"
        ) from exc
    if canonical_matrix != matrix:
        _fail("Phase 3 validated matrix differs from canonical reduction")
    matrix = canonical_matrix
    if matrix.unit_count != 11_520 or matrix.model_owner_count != 480:
        _fail("Phase 3 matrix is incomplete or contains extra units")
    expected_by_id = {item.unit.unit_id: item for item in body.units}
    records = tuple(matrix.records)
    if len(records) != 11_520 or len({item.unit_id for item in records}) != 11_520:
        _fail("Phase 3 matrix record identities are incomplete or duplicated")
    if set(item.unit_id for item in records) != set(expected_by_id):
        _fail("Phase 3 matrix contains missing or extra units")
    owners = {owner.owner_id: owner for owner in body.model_owners}
    if len(owners) != 480:
        _fail("Phase 3 model-owner universe is incomplete")
    diagnostics: dict[str, tuple[str, int, int, int]] = {}
    for owner_id, entries in matrix.owner_diagnostics:
        if owner_id in diagnostics or len(entries) != 5:
            _fail("Phase 3 owner diagnostics are duplicated or malformed")
        values = dict(entries)
        if set(values) != {
            "model_trainable_parameters",
            "model_optimizer_steps",
            "model_forward_passes",
            "model_recurrent_steps",
            "model_training_examples",
        } or any(type(values[name]) is not int or values[name] < 0 for name in values):
            _fail("Phase 3 owner diagnostics contain invalid fields")
        diagnostics[owner_id] = (
            owner_id,
            values["model_optimizer_steps"],
            values["model_forward_passes"],
            values["model_recurrent_steps"],
        )
    if set(diagnostics) != set(owners):
        _fail("Phase 3 owner diagnostics do not cover the exact owner universe")
    return expected_by_id, owners, diagnostics


def _restricted_interactions(record: object) -> int:
    outcome = record.outcome
    if outcome.success:
        value = outcome.first_optimum_adaptation_actions
        if value is None or type(value) is not int or value < 0 or value > FAILURE_CENSORING_BUDGET:
            _fail("successful Phase 3 outcome lacks typed first-hit actions")
        return value
    if (
        outcome.censored is not True
        or outcome.censoring_budget != FAILURE_CENSORING_BUDGET
        or outcome.censoring_reason != "fixed_endpoint"
        or outcome.first_optimum_adaptation_actions is not None
    ):
        _fail("failed Phase 3 outcome lacks fixed-endpoint censoring")
    return FAILURE_SENTINEL


def _median(values: Sequence[int]) -> Fraction:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return Fraction(ordered[middle])
    return Fraction(ordered[middle - 1] + ordered[middle], 2)


def _select(
    base_condition: str,
    variants: Mapping[str, tuple[Sequence[object], set[str]]],
    owners: Mapping[str, Phase3ModelOwner],
    diagnostics: Mapping[str, tuple[str, int, int, int]],
    heldout_shuffle_claim_eligible: bool | None,
    training_shuffle_claim_eligible: bool | None,
) -> Phase3ConditionSelection:
    candidates: list[Phase3SelectedMetric] = []
    for tuple_id in EXPECTED_TUPLES:
        try:
            records, owner_ids = variants[tuple_id]
        except KeyError:
            _fail("Phase 3 condition is missing a frozen candidate tuple")
        family_records: dict[str, list[object]] = {family: [] for family in FAMILIES}
        for record in records:
            family_records[record.key.family_id].append(record)
        family_metrics = tuple(
            Phase3FamilyMetric(
                family_id=family,
                units=len(family_records[family]),
                successes=sum(1 for record in family_records[family] if record.outcome.success),
                success_rate=Fraction(
                    sum(1 for record in family_records[family] if record.outcome.success),
                    len(family_records[family]),
                ),
                median_restricted_interactions=_median(
                    [_restricted_interactions(record) for record in family_records[family]]
                ),
            )
            for family in FAMILIES
        )
        if any(item.units != EXPECTED_UNITS_PER_FAMILY_TUPLE for item in family_metrics):
            _fail("Phase 3 candidate tuple does not have exact family coverage")
        selected_owner_ids = {
            owner_id
            for owner_id in owner_ids
            if owner_id in owners
            and owners[owner_id].condition_id == base_condition
            and owners[owner_id].training_tuple_id == tuple_id.rsplit("-t", 1)[0]
        }
        if len(selected_owner_ids) != EXPECTED_OWNERS_PER_TRAINING_TUPLE:
            _fail("selected tuple does not resolve exactly 30 unique model owners")
        candidates.append(
            Phase3SelectedMetric(
                condition_id=base_condition,
                tuple_id=tuple_id,
                training_tuple_id=tuple_id.rsplit("-t", 1)[0],
                family_metrics=family_metrics,
                minimum_family_success_rate=min(item.success_rate for item in family_metrics),
                worst_family_median_restricted_interactions=max(
                    item.median_restricted_interactions for item in family_metrics
                ),
                macro_average_family_median_restricted_interactions=sum(
                    (item.median_restricted_interactions for item in family_metrics), Fraction(0)
                )
                / len(family_metrics),
                optimizer_steps=sum(diagnostics[item][1] for item in selected_owner_ids),
                forward_passes=sum(diagnostics[item][2] for item in selected_owner_ids),
                recurrent_steps=sum(diagnostics[item][3] for item in selected_owner_ids),
                heldout_shuffle_claim_eligible=(
                    heldout_shuffle_claim_eligible
                    if base_condition == H4_SHUFFLED_CONDITION
                    else None
                ),
                training_shuffle_claim_eligible=(
                    training_shuffle_claim_eligible
                    if base_condition == H4_SHUFFLED_CONDITION
                    else None
                ),
            )
        )
    if len(candidates) != 12:
        _fail("Phase 3 condition does not contain exactly 12 candidate tuples")
    best_primary = max(item.minimum_family_success_rate for item in candidates)
    eligible = [
        item
        for item in candidates
        if best_primary - item.minimum_family_success_rate <= SUCCESS_TOLERANCE
    ]
    selected = min(
        eligible,
        key=lambda item: (
            item.worst_family_median_restricted_interactions,
            item.macro_average_family_median_restricted_interactions,
            item.optimizer_steps,
            item.forward_passes,
            item.recurrent_steps,
            _tuple_numeric(item.tuple_id),
        ),
    )
    return Phase3ConditionSelection(
        condition_id=base_condition,
        candidates=tuple(candidates),
        best_minimum_family_success_rate=best_primary,
        retained_tuple_ids=tuple(item.tuple_id for item in candidates if item in eligible),
        selected=selected,
    )


def select_phase3_tuples(
    plan: ValidatedPhase3Plan,
    authority: Phase3ModelArtifactAuthority,
    matrix: Phase3ValidatedMatrix,
    *,
    training_shuffle_claim_eligible: bool | None = None,
) -> Phase3SelectionResult:
    """Select one tuple independently for each new Phase 3 condition."""

    expected, owners, diagnostics_raw = _validate_inputs(plan, authority, matrix)
    diagnostics = {
        owner_id: (owner_id, values[1], values[2], values[3])
        for owner_id, values in diagnostics_raw.items()
    }
    records_by_variant: dict[str, tuple[list[object], set[str]]] = {}
    for record in matrix.records:
        planned = expected.get(record.unit_id)
        if planned is None:
            _fail("Phase 3 matrix contains an extra unit")
        if planned.base_condition_id not in NEW_CONDITIONS:
            _fail("Phase 3 matrix contains an anchor or final condition")
        if record.key.condition_id != f"{planned.base_condition_id}--{planned.tuple_id}":
            _fail("Phase 3 record condition identity differs from plan")
        key = f"{planned.base_condition_id}::{planned.tuple_id}"
        bucket = records_by_variant.setdefault(key, ([], set()))
        bucket[0].append(record)
        bucket[1].add(planned.model_owner_id)
    condition_selections = []
    for condition in NEW_CONDITIONS:
        variants = {
            tuple_id: records_by_variant[f"{condition}::{tuple_id}"]
            for tuple_id in EXPECTED_TUPLES
            if f"{condition}::{tuple_id}" in records_by_variant
        }
        if len(variants) != 12:
            _fail("Phase 3 matrix is missing or contains extra candidate tuples")
        condition_selections.append(
            _select(
                condition,
                variants,
                owners,
                diagnostics,
                matrix.control.heldout_search_claim_eligible
                if condition == H4_SHUFFLED_CONDITION
                else None,
                training_shuffle_claim_eligible,
            )
        )
    return Phase3SelectionResult(tuple(condition_selections), final_family_access=False)


def _require_metric(value: Phase3SelectedMetric, expected_condition: str) -> None:
    if type(value) is not Phase3SelectedMetric:
        _fail("selected conditions must use the typed Phase 3 metric")
    if value.condition_id != expected_condition or value.tuple_id not in EXPECTED_TUPLES:
        _fail("selected Phase 3 metric identity is malformed")
    if tuple(item.family_id for item in value.family_metrics) != FAMILIES:
        _fail("selected Phase 3 metric family matrix is malformed")
    if value.training_tuple_id != value.tuple_id.rsplit("-t", 1)[0]:
        _fail("selected Phase 3 metric training tuple is malformed")
    if value.tuple_id not in EXPECTED_TUPLES:
        _fail("selected Phase 3 metric tuple is malformed")
    if any(
        item.units != EXPECTED_UNITS_PER_FAMILY_TUPLE
        or item.successes < 0
        or item.successes > item.units
        or item.success_rate != Fraction(item.successes, item.units)
        or item.median_restricted_interactions < 0
        for item in value.family_metrics
    ):
        _fail("selected Phase 3 metric unit coverage is malformed")
    if (
        value.minimum_family_success_rate != min(item.success_rate for item in value.family_metrics)
        or value.worst_family_median_restricted_interactions
        != max(item.median_restricted_interactions for item in value.family_metrics)
        or value.macro_average_family_median_restricted_interactions
        != sum((item.median_restricted_interactions for item in value.family_metrics), Fraction(0))
        / len(value.family_metrics)
        or min(value.optimizer_steps, value.forward_passes, value.recurrent_steps) < 0
    ):
        _fail("selected Phase 3 aggregate metric is malformed")


def evaluate_phase3_claims(
    selection: Phase3SelectionResult | Mapping[str, Phase3SelectedMetric],
    *,
    locked_b2: Phase3SelectedMetric,
    locked_t: Phase3SelectedMetric,
    training_shuffle_claim_eligible: bool | None = None,
) -> Phase3ClaimResult:
    """Evaluate only the predeclared strict Phase 3 development claims."""

    _require_metric(locked_b2, "B2-global-listwise-optimum")
    _require_metric(locked_t, "T-markov-state-transition-listwise-optimum")
    values = (
        selection.by_condition()
        if isinstance(selection, Phase3SelectionResult)
        else dict(selection)
    )
    if set(values) != set(NEW_CONDITIONS) or any(
        type(value) is not Phase3SelectedMetric for value in values.values()
    ):
        _fail("Phase 3 selection result does not cover exactly the four new conditions")
    for condition, value in values.items():
        _require_metric(value, condition)
    s, h0, h4, shuffled = (values[condition] for condition in NEW_CONDITIONS)
    transition_gain = locked_t.minimum_family_success_rate - s.minimum_family_success_rate
    history_gain_over_t = h4.minimum_family_success_rate - locked_t.minimum_family_success_rate
    history_gain_over_h0 = h4.minimum_family_success_rate - h0.minimum_family_success_rate
    sequence_order_gain = h4.minimum_family_success_rate - shuffled.minimum_family_success_rate
    transition = transition_gain > SUCCESS_TOLERANCE
    history = history_gain_over_t > SUCCESS_TOLERANCE and history_gain_over_h0 > SUCCESS_TOLERANCE
    if (
        training_shuffle_claim_eligible is not None
        and shuffled.training_shuffle_claim_eligible is not None
        and training_shuffle_claim_eligible is not shuffled.training_shuffle_claim_eligible
    ):
        _fail("training shuffle eligibility differs between selection and claim inputs")
    training_gate = (
        training_shuffle_claim_eligible
        if training_shuffle_claim_eligible is not None
        else shuffled.training_shuffle_claim_eligible
    ) is True
    heldout_gate = shuffled.heldout_shuffle_claim_eligible is True
    sequence = sequence_order_gain > SUCCESS_TOLERANCE and training_gate and heldout_gate
    b2_rates = locked_b2.family_success_rates
    h4_rates = h4.family_success_rates
    if set(b2_rates) != set(FAMILIES) or set(h4_rates) != set(FAMILIES):
        _fail("B2/H4 claim metrics do not cover the frozen family universe")
    family_drops = tuple((family, b2_rates[family] - h4_rates[family]) for family in FAMILIES)
    minimum_drop = locked_b2.minimum_family_success_rate - h4.minimum_family_success_rate
    advancement = (
        history
        and sequence
        and all(drop <= SUCCESS_TOLERANCE for _, drop in family_drops)
        and minimum_drop <= SUCCESS_TOLERANCE
    )
    return Phase3ClaimResult(
        transition_claim=transition,
        history_access_claim=history,
        sequence_order_claim=sequence,
        advancement_to_paired_objectives=advancement,
        training_shuffle_claim_eligible=training_gate,
        heldout_shuffle_claim_eligible=heldout_gate,
        transition_gain=transition_gain,
        history_gain_over_t=history_gain_over_t,
        history_gain_over_h0=history_gain_over_h0,
        sequence_order_gain=sequence_order_gain,
        b2_minus_h4_family_success_drops=family_drops,
        b2_minus_h4_minimum_family_success=minimum_drop,
        final_family_access=False,
    )


reduce_phase3_selection = select_phase3_tuples
evaluate_claims = evaluate_phase3_claims

__all__ = [
    "EXPECTED_TRAINING_TUPLES",
    "EXPECTED_TUPLES",
    "FAILURE_SENTINEL",
    "NEW_CONDITIONS",
    "Phase3ClaimResult",
    "Phase3ConditionSelection",
    "Phase3FamilyMetric",
    "Phase3SelectedMetric",
    "Phase3SelectionError",
    "Phase3SelectionResult",
    "evaluate_claims",
    "evaluate_phase3_claims",
    "reduce_phase3_selection",
    "select_phase3_tuples",
]
