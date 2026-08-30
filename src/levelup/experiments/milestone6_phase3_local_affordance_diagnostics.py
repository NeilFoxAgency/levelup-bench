"""Pure, pre-outcome diagnostics for the Phase 3 local-affordance rung.

The reducer in :mod:`levelup.learning.state_conditioned` is the sole source of
row-level diagnostics.  This module only supplies identity-free evidence and
observable states to that reducer, then aggregates its typed output.  It never
opens a path, trains a model, searches, replays, evaluates, or reads final data.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from levelup.learning.state_conditioned import (
    LocalAffordanceDiagnostics,
    ObservableState,
    TaskLocalAffordanceEvidence,
    local_affordance_diagnostics,
)

FAMILY_ORDER = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
POPULATION_ORDER = ("training", "heldout")
DiagnosticPopulation = Literal["training", "heldout"]


class LocalAffordanceDiagnosticsError(ValueError):
    """Raised when diagnostics are incomplete, forged, or internally inconsistent."""


class FractionValue(BaseModel):
    """A non-negative, reduced rational suitable for canonical JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    numerator: StrictInt = Field(ge=0)
    denominator: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def is_reduced(self) -> "FractionValue":
        value = Fraction(self.numerator, self.denominator)
        if (value.numerator, value.denominator) != (self.numerator, self.denominator):
            raise ValueError("fraction must be reduced")
        return self

    @classmethod
    def from_fraction(cls, value: Fraction) -> "FractionValue":
        value = Fraction(value)
        return cls(numerator=value.numerator, denominator=value.denominator)

    def as_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


class LocalAffordanceQuery(BaseModel):
    """One identity-free evidence object and its observable query states."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    population: DiagnosticPopulation
    family_id: str = Field(min_length=1)
    evidence: TaskLocalAffordanceEvidence
    states: tuple[ObservableState, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def exact_types_and_family(self) -> "LocalAffordanceQuery":
        if self.family_id not in FAMILY_ORDER:
            raise ValueError("query family is not in the frozen development family order")
        if type(self.evidence) is not TaskLocalAffordanceEvidence:
            raise ValueError("query evidence must be exact identity-free TaskLocalAffordanceEvidence")
        if any(type(state) is not ObservableState for state in self.states):
            raise ValueError("query states must be exact observable states")
        return self


class AliasCount(BaseModel):
    """Number of visible rows for one opaque action alias."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: str = Field(min_length=1)
    count: StrictInt = Field(ge=1)


class AliasDiagnosticSummary(BaseModel):
    """Exact integer summary of all reducer rows for one alias."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: str = Field(min_length=1)
    alias_count: StrictInt = Field(ge=1)
    n: StrictInt = Field(ge=0)
    k_eff: StrictInt = Field(ge=0)
    kth_distance_sum: FractionValue | None = None
    kth_distance_count: StrictInt = Field(ge=0)
    kth_distance_min: FractionValue | None = None
    kth_distance_max: FractionValue | None = None
    kth_distance_mean: FractionValue | None = None
    kth_distance: FractionValue | None = None
    n_less_than_4: StrictInt = Field(ge=0)
    unknown: StrictInt = Field(ge=0)
    unknown_alias_count: StrictInt = Field(ge=0)
    local_used: StrictInt = Field(ge=0)
    eligible: StrictInt = Field(ge=0)
    local_vs_pooled_outcome_block_byte_difference: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def counts_are_bounded(self) -> "AliasDiagnosticSummary":
        for name in (
            "n_less_than_4",
            "unknown",
            "local_used",
            "eligible",
            "local_vs_pooled_outcome_block_byte_difference",
            "kth_distance_count",
        ):
            if getattr(self, name) > self.alias_count:
                raise ValueError(f"{name} exceeds alias row count")
        if self.unknown > 0 and self.n != 0:
            raise ValueError("unknown aliases must have zero same-alias support")
        if self.unknown_alias_count != self.unknown:
            raise ValueError("unknown-alias count differs from unknown count")
        if self.eligible > self.local_used:
            raise ValueError("eligible rows must use local support")
        if self.local_vs_pooled_outcome_block_byte_difference > self.eligible:
            raise ValueError("effective differences require eligible rows")
        if self.kth_distance_count == 0:
            if any(value is not None for value in (self.kth_distance_sum, self.kth_distance_min, self.kth_distance_max, self.kth_distance_mean, self.kth_distance)):
                raise ValueError("empty kth-distance distribution must be null")
        else:
            if any(value is None for value in (self.kth_distance_sum, self.kth_distance_min, self.kth_distance_max, self.kth_distance_mean, self.kth_distance)):
                raise ValueError("kth-distance distribution is incomplete")
            assert self.kth_distance_sum is not None and self.kth_distance_min is not None
            assert self.kth_distance_max is not None and self.kth_distance_mean is not None
            assert self.kth_distance is not None
            if self.kth_distance_min.as_fraction() > self.kth_distance_max.as_fraction():
                raise ValueError("kth-distance minimum exceeds maximum")
            expected_mean = FractionValue.from_fraction(self.kth_distance_sum.as_fraction() / self.kth_distance_count)
            if self.kth_distance_mean != expected_mean or self.kth_distance != expected_mean:
                raise ValueError("kth-distance mean is inconsistent with sum/count")
        return self

    @property
    def n_less_than_4_bool(self) -> bool:
        return self.n_less_than_4 > 0

class CoverageGate(BaseModel):
    """Frozen local-alignment coverage gate for one population/family scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    eligible: StrictInt = Field(ge=0)
    difference: StrictInt = Field(ge=0)
    fraction: FractionValue | None = None
    threshold: FractionValue
    passes: StrictBool

    @model_validator(mode="after")
    def gate_is_exact(self) -> "CoverageGate":
        if self.eligible == 0:
            if self.difference != 0 or self.fraction is not None or self.passes is not False:
                raise ValueError("zero-eligible coverage must be an explicit failed gate")
            return self
        expected = FractionValue.from_fraction(Fraction(self.difference, self.eligible))
        if self.fraction != expected:
            raise ValueError("coverage fraction does not match integer counts")
        if self.difference > self.eligible:
            raise ValueError("coverage differences exceed eligible rows")
        expected_pass = self.fraction.as_fraction() >= self.threshold.as_fraction()
        if self.passes is not expected_pass:
            raise ValueError("coverage pass flag does not match threshold")
        return self


class _DiagnosticScope(BaseModel):
    """Shared aggregate fields for a family or complete population."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alias_counts: tuple[AliasCount, ...]
    evidence_query_count: StrictInt = Field(ge=0)
    state_query_count: StrictInt = Field(ge=0)
    alias_rows: StrictInt = Field(ge=0)
    n: StrictInt = Field(ge=0)
    k_eff: StrictInt = Field(ge=0)
    kth_distance_sum: FractionValue | None = None
    kth_distance_count: StrictInt = Field(ge=0)
    kth_distance_min: FractionValue | None = None
    kth_distance_max: FractionValue | None = None
    kth_distance_mean: FractionValue | None = None
    kth_distance: FractionValue | None = None
    n_less_than_4: StrictInt = Field(ge=0)
    unknown: StrictInt = Field(ge=0)
    unknown_alias_count: StrictInt = Field(ge=0)
    local_used: StrictInt = Field(ge=0)
    eligible: StrictInt = Field(ge=0)
    local_vs_pooled_outcome_block_byte_difference: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def scope_counts_are_exact(self) -> "_DiagnosticScope":
        aliases = tuple(item.alias for item in self.alias_counts)
        if aliases != tuple(sorted(set(aliases))):
            raise ValueError("alias counts must be unique and canonically sorted")
        if sum(item.count for item in self.alias_counts) != self.alias_rows:
            raise ValueError("alias counts do not sum to alias rows")
        for name in ("n_less_than_4", "unknown", "local_used", "eligible", "local_vs_pooled_outcome_block_byte_difference", "kth_distance_count"):
            if getattr(self, name) > self.alias_rows:
                raise ValueError(f"{name} exceeds alias rows")
        if self.unknown > self.alias_rows or self.eligible > self.local_used:
            raise ValueError("scope diagnostic counts are inconsistent")
        if self.unknown_alias_count != self.unknown:
            raise ValueError("scope unknown-alias count differs from unknown count")
        if self.local_vs_pooled_outcome_block_byte_difference > self.eligible:
            raise ValueError("scope differences require eligible rows")
        if self.kth_distance_count and self.kth_distance_sum is None:
            raise ValueError("scope kth-distance count requires a sum")
        if self.kth_distance_count == 0:
            if any(value is not None for value in (self.kth_distance_sum, self.kth_distance_min, self.kth_distance_max, self.kth_distance_mean, self.kth_distance)):
                raise ValueError("empty kth-distance distribution must be null")
        else:
            if any(value is None for value in (self.kth_distance_sum, self.kth_distance_min, self.kth_distance_max, self.kth_distance_mean, self.kth_distance)):
                raise ValueError("kth-distance distribution is incomplete")
            assert self.kth_distance_sum is not None
            assert self.kth_distance_min is not None
            assert self.kth_distance_max is not None
            assert self.kth_distance_mean is not None
            assert self.kth_distance is not None
            if self.kth_distance_min.as_fraction() > self.kth_distance_max.as_fraction():
                raise ValueError("kth-distance minimum exceeds maximum")
            expected_mean = FractionValue.from_fraction(self.kth_distance_sum.as_fraction() / self.kth_distance_count)
            if self.kth_distance_mean != expected_mean or self.kth_distance != expected_mean:
                raise ValueError("kth-distance mean is inconsistent with sum/count")
        return self


class FamilyDiagnosticSummary(_DiagnosticScope):
    family_id: str
    coverage_gate: CoverageGate

    @model_validator(mode="after")
    def family_id_is_known(self) -> "FamilyDiagnosticSummary":
        if self.family_id not in FAMILY_ORDER:
            raise ValueError("unknown diagnostic family")
        if self.coverage_gate.threshold.as_fraction() != Fraction(1, 2):
            raise ValueError("family coverage threshold drifted from frozen 1/2")
        if self.coverage_gate.eligible != self.eligible:
            raise ValueError("family coverage denominator differs from eligible count")
        if self.coverage_gate.difference != self.local_vs_pooled_outcome_block_byte_difference:
            raise ValueError("family coverage numerator differs from difference count")
        return self


class PopulationDiagnosticSummary(_DiagnosticScope):
    population: DiagnosticPopulation
    family_summaries: tuple[FamilyDiagnosticSummary, ...]
    coverage_gate: CoverageGate

    @model_validator(mode="after")
    def complete_family_matrix(self) -> "PopulationDiagnosticSummary":
        if tuple(item.family_id for item in self.family_summaries) != FAMILY_ORDER:
            raise ValueError("population must contain exactly the frozen six families in order")
        if self.coverage_gate.eligible != self.eligible:
            raise ValueError("population coverage denominator differs from eligible count")
        if self.coverage_gate.difference != self.local_vs_pooled_outcome_block_byte_difference:
            raise ValueError("population coverage numerator differs from difference count")
        if self.coverage_gate.threshold.as_fraction() != Fraction(4, 5):
            raise ValueError("population coverage threshold drifted from frozen 4/5")
        for family in self.family_summaries:
            if family.coverage_gate.threshold.as_fraction() != Fraction(1, 2):
                raise ValueError("family coverage threshold drifted from frozen 1/2")
        for name in ("evidence_query_count", "state_query_count", "alias_rows", "n", "k_eff", "kth_distance_count", "n_less_than_4", "unknown", "unknown_alias_count", "local_used", "eligible", "local_vs_pooled_outcome_block_byte_difference"):
            if getattr(self, name) != sum(getattr(item, name) for item in self.family_summaries):
                raise ValueError(f"population {name} does not equal family sum")
        if self.alias_counts != _merge_alias_counts(self.family_summaries):
            raise ValueError("population alias counts do not equal family sums")
        expected_distance = _sum_fraction_values(item.kth_distance_sum for item in self.family_summaries)
        if self.kth_distance_sum != expected_distance:
            raise ValueError("population kth-distance sum does not equal family sums")
        distances_min = min(
            (item.kth_distance_min for item in self.family_summaries if item.kth_distance_min is not None),
            key=lambda value: value.as_fraction(),
            default=None,
        )
        distances_max = max(
            (item.kth_distance_max for item in self.family_summaries if item.kth_distance_max is not None),
            key=lambda value: value.as_fraction(),
            default=None,
        )
        if self.kth_distance_min != distances_min or self.kth_distance_max != distances_max:
            raise ValueError("population kth-distance extrema do not equal family extrema")
        expected_mean = (
            FractionValue.from_fraction(self.kth_distance_sum.as_fraction() / self.kth_distance_count)
            if self.kth_distance_count
            else None
        )
        if self.kth_distance_mean != expected_mean:
            raise ValueError("population kth-distance mean does not equal sum/count")
        if self.kth_distance != self.kth_distance_mean:
            raise ValueError("population kth-distance alias differs from its mean")
        return self


class LocalAffordanceDiagnosticReport(BaseModel):
    """Complete training/held-out pre-outcome diagnostic report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["milestone6.phase3.local-affordance-diagnostics.v1"] = "milestone6.phase3.local-affordance-diagnostics.v1"
    populations: tuple[PopulationDiagnosticSummary, ...]

    @model_validator(mode="after")
    def complete_populations(self) -> "LocalAffordanceDiagnosticReport":
        if tuple(item.population for item in self.populations) != POPULATION_ORDER:
            raise ValueError("report must contain exactly training and heldout populations in order")
        return self

    def for_population(self, population: DiagnosticPopulation) -> PopulationDiagnosticSummary:
        return next(item for item in self.populations if item.population == population)


def _sum_fraction_values(values: Sequence[FractionValue | None]) -> FractionValue | None:
    present = [value.as_fraction() for value in values if value is not None]
    return FractionValue.from_fraction(sum(present, Fraction(0))) if present else None


def _merge_alias_counts(families: Sequence[_DiagnosticScope]) -> tuple[AliasCount, ...]:
    counts: dict[str, int] = {}
    for family in families:
        for item in family.alias_counts:
            counts[item.alias] = counts.get(item.alias, 0) + item.count
    return tuple(AliasCount(alias=alias, count=counts[alias]) for alias in sorted(counts))


def _distance_distribution(distances: Sequence[Fraction]) -> tuple[FractionValue | None, FractionValue | None, FractionValue | None, FractionValue | None]:
    if not distances:
        return None, None, None, None
    total = sum(distances, Fraction(0))
    mean = total / len(distances)
    return (
        FractionValue.from_fraction(total),
        FractionValue.from_fraction(min(distances)),
        FractionValue.from_fraction(max(distances)),
        FractionValue.from_fraction(mean),
    )


def _scope_from_rows(
    rows: Sequence[LocalAffordanceDiagnostics],
    *,
    evidence_query_count: int,
    state_query_count: int,
) -> tuple[dict[str, AliasDiagnosticSummary], _DiagnosticScope]:
    grouped: dict[str, list[LocalAffordanceDiagnostics]] = {}
    for row in rows:
        if type(row) is not LocalAffordanceDiagnostics:
            raise LocalAffordanceDiagnosticsError("reducer returned a forged diagnostic row")
        if (
            type(row.alias) is not str
            or not row.alias
            or type(row.n) is not int
            or row.n < 0
            or type(row.k_eff) is not int
            or not 0 <= row.k_eff <= min(4, row.n)
            or (row.selected_max_distance is not None and (type(row.selected_max_distance) is not float or not math.isfinite(row.selected_max_distance) or row.selected_max_distance < 0))
            or (row.n == 0 and row.selected_max_distance is not None)
            or (row.n > 0 and row.selected_max_distance is None)
            or type(row.eligible) is not bool
            or type(row.local_used) is not bool
            or type(row.local_vs_pooled_outcome_block_byte_difference) is not bool
            or row.local_used is not (row.n > 4)
            or row.eligible and not row.local_used
            or row.local_vs_pooled_outcome_block_byte_difference and not row.eligible
        ):
            raise LocalAffordanceDiagnosticsError("fixed reducer returned malformed diagnostic fields")
        grouped.setdefault(row.alias, []).append(row)
    aliases: dict[str, AliasDiagnosticSummary] = {}
    for alias in sorted(grouped):
        values = grouped[alias]
        distances = [Fraction(str(row.kth_distance)) for row in values if row.kth_distance is not None]
        distance_sum, distance_min, distance_max, distance_mean = _distance_distribution(distances)
        aliases[alias] = AliasDiagnosticSummary(
            alias=alias,
            alias_count=len(values),
            n=sum(row.n for row in values),
            k_eff=sum(row.k_eff for row in values),
            kth_distance=distance_mean,
            kth_distance_sum=distance_sum,
            kth_distance_count=len(distances),
            kth_distance_min=distance_min,
            kth_distance_max=distance_max,
            kth_distance_mean=distance_mean,
            n_less_than_4=sum(row.n_less_than_4 for row in values),
            unknown=sum(1 for row in values if row.n == 0),
            unknown_alias_count=sum(1 for row in values if row.n == 0),
            local_used=sum(row.local_used for row in values),
            eligible=sum(row.eligible for row in values),
            local_vs_pooled_outcome_block_byte_difference=sum(row.local_vs_pooled_outcome_block_byte_difference for row in values),
        )
    distance_sum = _sum_fraction_values([value.kth_distance_sum for value in aliases.values()])
    unknown_count = sum(value.unknown for value in aliases.values())
    scope = _DiagnosticScope(
        alias_counts=tuple(AliasCount(alias=alias, count=value.alias_count) for alias, value in aliases.items()),
        evidence_query_count=evidence_query_count,
        state_query_count=state_query_count,
        alias_rows=len(rows),
        n=sum(value.n for value in aliases.values()),
        k_eff=sum(value.k_eff for value in aliases.values()),
        kth_distance=FractionValue.from_fraction(distance_sum.as_fraction() / sum(value.kth_distance_count for value in aliases.values())) if distance_sum is not None else None,
        kth_distance_sum=distance_sum,
        kth_distance_count=sum(value.kth_distance_count for value in aliases.values()),
        kth_distance_min=min((value.kth_distance_min for value in aliases.values() if value.kth_distance_min is not None), key=lambda value: value.as_fraction(), default=None),
        kth_distance_max=max((value.kth_distance_max for value in aliases.values() if value.kth_distance_max is not None), key=lambda value: value.as_fraction(), default=None),
        kth_distance_mean=FractionValue.from_fraction(distance_sum.as_fraction() / sum(value.kth_distance_count for value in aliases.values())) if distance_sum is not None else None,
        n_less_than_4=sum(value.n_less_than_4 for value in aliases.values()),
        unknown=unknown_count,
        unknown_alias_count=unknown_count,
        local_used=sum(value.local_used for value in aliases.values()),
        eligible=sum(value.eligible for value in aliases.values()),
        local_vs_pooled_outcome_block_byte_difference=sum(value.local_vs_pooled_outcome_block_byte_difference for value in aliases.values()),
    )
    return aliases, scope


def _gate(scope: _DiagnosticScope, *, threshold: Fraction) -> CoverageGate:
    if scope.eligible <= 0:
        return CoverageGate(
            eligible=0,
            difference=0,
            fraction=None,
            threshold=FractionValue.from_fraction(threshold),
            passes=False,
        )
    fraction = Fraction(scope.local_vs_pooled_outcome_block_byte_difference, scope.eligible)
    return CoverageGate(
        eligible=scope.eligible,
        difference=scope.local_vs_pooled_outcome_block_byte_difference,
        fraction=FractionValue.from_fraction(fraction),
        threshold=FractionValue.from_fraction(threshold),
        passes=fraction >= threshold,
    )


def _family_summary(
    family_id: str,
    rows: Sequence[LocalAffordanceDiagnostics],
    *,
    evidence_query_count: int,
    state_query_count: int,
) -> FamilyDiagnosticSummary:
    _aliases, scope = _scope_from_rows(
        rows,
        evidence_query_count=evidence_query_count,
        state_query_count=state_query_count,
    )
    return FamilyDiagnosticSummary(
        family_id=family_id,
        **scope.model_dump(),
        coverage_gate=_gate(scope, threshold=Fraction(1, 2)),
    )


def _population_summary(
    population: DiagnosticPopulation,
    family_rows: dict[str, list[LocalAffordanceDiagnostics]],
    family_evidence_counts: dict[str, int],
    family_state_counts: dict[str, int],
) -> PopulationDiagnosticSummary:
    families = tuple(
        _family_summary(
            family,
            family_rows[family],
            evidence_query_count=family_evidence_counts[family],
            state_query_count=family_state_counts[family],
        )
        for family in FAMILY_ORDER
    )
    _aliases, scope = _scope_from_rows(
        [row for family in FAMILY_ORDER for row in family_rows[family]],
        evidence_query_count=sum(family_evidence_counts.values()),
        state_query_count=sum(family_state_counts.values()),
    )
    return PopulationDiagnosticSummary(
        population=population,
        family_summaries=families,
        **scope.model_dump(),
        coverage_gate=_gate(scope, threshold=Fraction(4, 5)),
    )


def aggregate_local_affordance_diagnostics(
    queries: Sequence[LocalAffordanceQuery],
) -> LocalAffordanceDiagnosticReport:
    """Run the fixed reducer for every query state and aggregate fail-closed."""

    try:
        values = tuple(LocalAffordanceQuery.model_validate(query) for query in queries)
    except (TypeError, ValueError) as exc:
        raise LocalAffordanceDiagnosticsError("query matrix is malformed") from exc
    if not values:
        raise LocalAffordanceDiagnosticsError("diagnostic query matrix is empty")
    rows: dict[str, dict[str, list[LocalAffordanceDiagnostics]]] = {
        population: {family: [] for family in FAMILY_ORDER} for population in POPULATION_ORDER
    }
    evidence_counts: dict[str, dict[str, int]] = {
        population: {family: 0 for family in FAMILY_ORDER} for population in POPULATION_ORDER
    }
    state_counts: dict[str, dict[str, int]] = {
        population: {family: 0 for family in FAMILY_ORDER} for population in POPULATION_ORDER
    }
    for query in values:
        evidence_counts[query.population][query.family_id] += 1
        state_counts[query.population][query.family_id] += len(query.states)
        for state in query.states:
            try:
                diagnostics = local_affordance_diagnostics(state, query.evidence)
            except (TypeError, ValueError) as exc:
                raise LocalAffordanceDiagnosticsError("fixed reducer rejected query evidence") from exc
            rows[query.population][query.family_id].extend(diagnostics)
    if any(not rows[population][family] for population in POPULATION_ORDER for family in FAMILY_ORDER):
        raise LocalAffordanceDiagnosticsError("diagnostic query matrix is missing population/family coverage")
    try:
        return LocalAffordanceDiagnosticReport(
            populations=tuple(
                _population_summary(
                    population,
                    rows[population],
                    evidence_counts[population],
                    state_counts[population],
                )
                for population in POPULATION_ORDER
            )
        )
    except LocalAffordanceDiagnosticsError:
        raise
    except (TypeError, ValueError) as exc:
        raise LocalAffordanceDiagnosticsError("diagnostic aggregation failed closed") from exc


def validate_local_affordance_diagnostic_report(value: Any) -> LocalAffordanceDiagnosticReport:
    """Re-validate a report's exact schema and all count/gate invariants."""

    try:
        report = value if type(value) is LocalAffordanceDiagnosticReport else LocalAffordanceDiagnosticReport.model_validate(value)
        return LocalAffordanceDiagnosticReport.model_validate(report.model_dump(mode="json"))
    except (TypeError, ValueError) as exc:
        raise LocalAffordanceDiagnosticsError("diagnostic report is forged or incomplete") from exc


# Descriptive compatibility names for callers and notebooks.
build_local_affordance_diagnostic_report = aggregate_local_affordance_diagnostics
validate_diagnostic_report = validate_local_affordance_diagnostic_report

__all__ = [
    "AliasCount",
    "AliasDiagnosticSummary",
    "CoverageGate",
    "FAMILY_ORDER",
    "FractionValue",
    "FamilyDiagnosticSummary",
    "LocalAffordanceDiagnosticReport",
    "LocalAffordanceDiagnosticsError",
    "LocalAffordanceQuery",
    "POPULATION_ORDER",
    "PopulationDiagnosticSummary",
    "aggregate_local_affordance_diagnostics",
    "build_local_affordance_diagnostic_report",
    "validate_diagnostic_report",
    "validate_local_affordance_diagnostic_report",
]
