import pytest
from pydantic import ValidationError

from levelup.core import PerformanceTier, ReferenceEntry, ReferenceLadder


def test_ladder_allows_multiple_historical_entries_in_one_tier() -> None:
    ladder = ReferenceLadder(
        task_id="micro.route.001",
        entries=(
            ReferenceEntry(
                reference_id="wr-2025",
                tier=PerformanceTier.WORLD_RECORD,
                performance_value=100.0,
            ),
            ReferenceEntry(
                reference_id="wr-2026",
                tier=PerformanceTier.WORLD_RECORD,
                performance_value=95.0,
            ),
            ReferenceEntry(
                reference_id="tas-2026",
                tier=PerformanceTier.TAS,
                performance_value=80.0,
                verified=True,
            ),
        ),
    )

    assert len(ladder.entries) == 3
    assert ladder.entries[-1].tier is PerformanceTier.TAS


def test_duplicate_reference_ids_are_rejected() -> None:
    entry = ReferenceEntry(
        reference_id="same",
        tier=PerformanceTier.HUMAN,
        performance_value=120.0,
    )

    with pytest.raises(ValidationError, match="reference_id values must be unique"):
        ReferenceLadder(task_id="micro.route.001", entries=(entry, entry))


def test_ladder_rejects_reversed_tier_order() -> None:
    with pytest.raises(ValidationError, match="canonical performance tier order"):
        ReferenceLadder(
            task_id="micro.route.001",
            entries=(
                ReferenceEntry(
                    reference_id="tas",
                    tier=PerformanceTier.TAS,
                    performance_value=80.0,
                ),
                ReferenceEntry(
                    reference_id="human",
                    tier=PerformanceTier.HUMAN,
                    performance_value=120.0,
                ),
            ),
        )
