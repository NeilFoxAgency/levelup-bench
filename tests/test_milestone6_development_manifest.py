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
    assert protocol["schema_version"] == "milestone6.development_protocol.v2"
    assert protocol["freeze_record"] == {
        "amended_at_local_date": "2026-08-21",
        "amendment_timing": "before comparative development results",
        "comparative_results_inspected_before_amendment": False,
        "previous_sha256": "a0d6e5760591ce70f95df4a87b8166dc69defa5e3242ddadabe3393b3e82a488",
        "reason": "operationalize restricted interactions, bind clean causal controls, and isolate explicit pairing",
    }
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
    metric = protocol["selection"]["restricted_interactions_metric"]
    assert metric["metric_id"] == "total_adaptation_actions_to_first_exact_optimum"
    assert metric["executed_action_formula"] == (
        "accounting.probes.actions + accounting.search.actions"
    )
    assert "endpoint_adaptation_actions + 1" in metric["failure_value"]
    assert "oracle never changes search control flow" in metric["oracle_timing"]
    assert "within each family first" in metric["family_aggregation"]
    assert "typed field" in metric["typed_record_requirement"]
    assert protocol["capacity_matching"]["strong_optimum_baseline"]
    assert not protocol["eligible_hyperparameters"]["transformer_eligible"]
    assert protocol["final_family_access"] == "forbidden_until_phase9_method_freeze"


def test_representation_ladder_freezes_pairing_only_and_destroyed_structure_controls() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["baseline_ladder"] == [
        "A0_no_probe_uniform",
        "A1_paid_probe_uniform",
        "B1_clean_global_optimum_frequency",
        "B2_global_listwise_optimum",
        "C_state_conditioned_listwise_optimum",
        "D_state_conditioned_pooled",
        "D1_state_conditioned_unpaired_same_trajectories",
        "E1_independently_randomized_direction",
        "E2_randomized_pairing",
        "F_correctly_paired_improvement",
    ]

    controls = protocol["destroyed_structure_controls"]
    assert "multi-structure" in controls["pooled"]
    assert all(
        structure in controls["pooled"]
        for structure in ("pairing", "order", "better-stage")
    )
    assert all(
        phrase in controls["pairing_only_unpaired"]
        for phrase in (
            "same frontier and optimum trajectories",
            "sequence order",
            "stage labels",
            "example multiset as F",
            "only cross-trajectory pair membership",
            "capacity",
            "optimizer",
            "budgets",
        )
    )
    assert "independent fair-coin" in controls["randomized_direction"]
    assert "no intentionally correct subset" in controls["randomized_direction"]
    assert "within-stratum derangement" in controls["randomized_pairing"]
    assert "no self-pairs" in controls["randomized_pairing"]
    assert "same identical exclusions as F" in controls["randomized_pairing"]

    requirements = protocol["control_seed_identity_requirements"]
    assert requirements["pairing_only_unpaired"]["seed"] == (
        "no additional randomization; inherit F's non-label seeds"
    )
    assert requirements["pairing_only_unpaired"]["removed_structure"] == (
        "cross-trajectory pair membership only"
    )
    assert requirements["pairing_only_unpaired"]["capacity_optimizer_budgets"] == (
        "identical to F"
    )
    membership = requirements["pairing_only_unpaired"][
        "learner_visible_membership_metadata"
    ]
    assert membership.startswith("none")
    assert all(
        forbidden in membership
        for forbidden in ("trajectory-pair IDs", "alignment-pair IDs", "shared record keys")
    )
    assert requirements["randomized_direction"]["seed"] == (
        "seed_policy.bases.randomized_direction"
    )
    assert requirements["randomized_pairing"]["seed"] == (
        "seed_policy.bases.randomized_pairing"
    )
    assert requirements["randomized_pairing"]["self_pairs"] is False
    assert requirements["pooled"]["kind"] == "multi-structure"
    assert requirements["pooled"]["removed_structures"] == [
        "cross_trajectory_pair_membership",
        "sequence_order",
        "better_stage_labels",
    ]

    assert protocol["representation_ladder_questions"] == [
        "Does state conditioning help beyond global/action-only statistics?",
        "Does transition information help beyond current state?",
        "Does history/sequence help beyond transition information?",
        "Does explicit frontier-to-optimum pairing help beyond the exact same trajectories without pair membership?",
    ]
    stages = protocol["representation_ladder_stage_contract"]
    assert stages["state_conditioning"]["status"] == "eligible_in_phase2_screening"
    assert "B2_global" in stages["state_conditioning"]["matched_comparison"]
    assert stages["transition_information"]["claims_before_gate"] == "forbidden"
    assert "state-only" in stages["transition_information"]["required_match"]
    assert stages["history_sequence"]["claims_before_gate"] == "forbidden"
    assert "transition-only" in stages["history_sequence"]["required_match"]
    assert stages["explicit_pairing"]["claims_before_gate"] == "forbidden"
    assert stages["explicit_pairing"]["matched_comparison"].startswith(
        "D1_state_conditioned_unpaired_same_trajectories versus F_"
    )
