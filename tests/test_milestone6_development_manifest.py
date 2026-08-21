from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from levelup.envs.adaptive_track import collect_adaptive_bundles
from levelup.envs.challenge_track import held_out_combo_tasks

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "milestone6" / "development_tasks.json"
PROTOCOL = ROOT / "configs" / "milestone6" / "development_protocol.json"
ADAPTIVE_SEEDS = {
    "plain": 900,
    "battery": 1000,
    "cooldown": 1100,
    "heat": 1200,
    "momentum": 1300,
}


def _load() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_development_manifest_reconstructs_canonical_known_tasks() -> None:
    payload = _load()
    expected: list[tuple[str, str, int, int]] = []
    for family, generator_seed in ADAPTIVE_SEEDS.items():
        bundles = collect_adaptive_bundles(family, 30, generator_seed)
        expected.extend(
            (
                family,
                bundle.environment.task_spec.task_id,
                bundle.environment.task_index,
                generator_seed,
            )
            for bundle in bundles
        )
    expected.extend(
        (
            "combo",
            environment.task_spec.task_id,
            environment.task_index,
            2026,
        )
        for environment, _ in held_out_combo_tasks(8, 2026)
    )

    observed = [
        (
            task["family"],
            task["task_id"],
            task["task_index"],
            task["generator_seed"],
        )
        for task in payload["tasks"]
    ]
    assert observed == expected
    assert len(observed) == 158
    assert {task["environment_reset_seed"] for task in payload["tasks"]} == {0}


def test_manifest_roles_are_development_only_and_first_eight_are_training_core() -> None:
    payload = _load()
    for family in payload["family_order"]:
        tasks = [task for task in payload["tasks"] if task["family"] == family]
        assert all("known_development" in task["roles"] for task in tasks)
        assert ["training_core" in task["roles"] for task in tasks] == [
            index < 8 for index in range(len(tasks))
        ]
        assert all("final" not in role for task in tasks for role in task["roles"])
        assert all("trajectory" not in key for task in tasks for key in task)

    combo = [task for task in payload["tasks"] if task["family"] == "combo"]
    assert all("historical_milestone5_development" in task["roles"] for task in combo)


def test_development_protocol_freezes_seeds_selection_and_no_optimum_stopping() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["family_order"] == [
        "plain",
        "battery",
        "cooldown",
        "heat",
        "momentum",
        "combo",
    ]
    assert protocol["seed_policy"]["screening_replicates"] == list(range(5))
    assert protocol["seed_policy"]["selection_replicates"] == list(range(20))
    assert len(set(protocol["seed_policy"]["bases"].values())) == 6
    assert protocol["budgets"]["selection"]["adaptation_actions_per_task"] == 8192
    assert protocol["budgets"]["maximum_actions_per_candidate_episode"] == 64
    assert not protocol["budgets"]["exact_optimum_affects_search_control_flow"]
    assert protocol["selection"]["exact_optimum_is_reporting_only"]
    assert protocol["selection"]["primary_metric"] == (
        "minimum_family_exact_optimum_success_rate"
    )
    assert protocol["capacity_matching"]["strong_optimum_baseline"]
    assert not protocol["eligible_hyperparameters"]["transformer_eligible"]
    assert protocol["final_family_access"] == "forbidden_until_phase9_method_freeze"
