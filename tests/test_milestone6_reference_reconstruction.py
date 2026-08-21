from __future__ import annotations

import pytest

from levelup.experiments.milestone6_baselines import trajectory_content_sha256
from levelup.experiments.milestone6_phase2 import (
    _manifest_tasks,
    _training_identity,
    reconstruct_development_training_task,
)


def _training_entry(family: str) -> dict[str, object]:
    return next(
        entry
        for entry in _manifest_tasks()
        if entry["family"] == family and "training_core" in entry["roles"]
    )


@pytest.mark.parametrize(
    "family",
    ("plain", "battery", "cooldown", "heat", "momentum", "combo"),
)
def test_development_training_reconstruction_matches_identity_catalog(family: str) -> None:
    entry = _training_entry(family)
    reconstruction = reconstruct_development_training_task(
        family=family,
        task_index=int(entry["task_index"]),
        generator_seed=int(entry["generator_seed"]),
        expected_task_id=str(entry["task_id"]),
    )
    identity = _training_identity(entry)

    assert reconstruction.environment.task_spec.task_id == identity.task_id
    assert reconstruction.catalog == identity.trajectory_catalog
    assert set(reconstruction.trajectories) == {
        item.trajectory_id for item in reconstruction.catalog
    }
    for item in reconstruction.catalog:
        trajectory = reconstruction.trajectories[item.trajectory_id]
        assert trajectory.task_id == identity.task_id
        assert item.provenance["content_sha256"] == trajectory_content_sha256(trajectory)


def test_combo_reconstruction_rejects_manifest_task_identity_drift() -> None:
    entry = _training_entry("combo")

    with pytest.raises(RuntimeError, match="reconstruction drift"):
        reconstruct_development_training_task(
            family="combo",
            task_index=int(entry["task_index"]),
            generator_seed=int(entry["generator_seed"]),
            expected_task_id=f"{entry['task_id']}.tampered",
        )
