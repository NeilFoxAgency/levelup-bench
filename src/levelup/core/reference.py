"""Reference performance ladders used to study optimality transfer."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

SCHEMA_VERSION = "0.1"


class PerformanceTier(StrEnum):
    """Canonical skill tiers. Multiple entries may occupy the same tier."""

    NOVICE = "novice"
    HUMAN = "human"
    EXPERIENCED_HUMAN = "experienced_human"
    ELITE_HUMAN = "elite_human"
    WORLD_RECORD = "world_record"
    TAS = "tas"
    PROVEN_OPTIMUM = "proven_optimum"


_TIER_ORDER = {tier: index for index, tier in enumerate(PerformanceTier)}


class ReferenceEntry(BaseModel):
    """One measured reference point on a task's performance ladder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_id: str = Field(min_length=1)
    tier: PerformanceTier
    performance_value: float = Field(allow_inf_nan=False)
    trajectory_id: str | None = None
    verified: bool = False
    provenance: dict[str, JsonValue] = Field(default_factory=dict)


class ReferenceLadder(BaseModel):
    """Ordered collection of reference performances for one task.

    Entries follow canonical tier order. Multiple historical records may share
    a tier, so tier values are not required to be unique.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    entries: tuple[ReferenceEntry, ...] = ()

    @model_validator(mode="after")
    def entries_are_well_formed(self) -> "ReferenceLadder":
        ids = [entry.reference_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("reference_id values must be unique within a ladder")

        ranks = [_TIER_ORDER[entry.tier] for entry in self.entries]
        if ranks != sorted(ranks):
            raise ValueError("reference entries must follow canonical performance tier order")
        return self
