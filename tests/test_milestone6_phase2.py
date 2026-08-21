from __future__ import annotations

import pytest

from levelup.experiments.milestone6_phase2 import (
    build_phase2_baseline_smoke_config,
    phase2_baseline_smoke_executor,
)
from levelup.experiments.runner.config import ExperimentConfig
from levelup.experiments.runner.storage import plan_expected_units


def test_phase2_smoke_config_is_validation_only_and_has_exact_clean_exposures() -> None:
    config = build_phase2_baseline_smoke_config()
    assert len(config.split.development_tasks) == 40
    assert len(config.split.validation_tasks) == 1
    assert not config.split.final_tasks
    assert config.split.validation_tasks[0].family_id == "combo"
    assert all(len(task.trajectory_catalog) == 2 for task in config.split.development_tasks)
    assert all(
        len(item.provenance["content_sha256"]) == 64
        for task in config.split.development_tasks
        for item in task.trajectory_catalog
    )

    conditions = {condition.condition_id: condition for condition in config.conditions}
    assert not conditions["A0-no-probe-uniform"].exposure.train_task_ids
    assert not conditions["A1-paid-probe-uniform"].exposure.train_task_ids
    for condition_id in (
        "B1-clean-global-optimum-frequency",
        "B2-global-listwise-optimum",
        "C-state-conditioned-listwise-optimum",
    ):
        exposure = conditions[condition_id].exposure
        assert len(exposure.train_task_ids) == 40
        assert len(exposure.exposed_trajectories) == 40
        assert {item.stage_label for item in exposure.exposed_trajectories} == {"optimum"}
    assert all(condition.execution_phases == ("validation",) for condition in config.conditions)

    expected = plan_expected_units(config)
    assert len(expected.units) == 5
    assert {unit.key.phase for unit in expected.units} == {"validation"}
    assert len({unit.seeds.model_dump_json() for unit in expected.units}) == 1


def test_all_phase2_smoke_conditions_run_matched_fixed_budgets() -> None:
    config = build_phase2_baseline_smoke_config()
    payloads = {
        unit.key.condition_id: phase2_baseline_smoke_executor(config, unit)
        for unit in plan_expected_units(config).units
    }
    assert set(payloads) == {
        "A0-no-probe-uniform",
        "A1-paid-probe-uniform",
        "B1-clean-global-optimum-frequency",
        "B2-global-listwise-optimum",
        "C-state-conditioned-listwise-optimum",
    }
    for payload in payloads.values():
        assert payload.diagnostics["not_scientific_result"] is True
        assert payload.accounting.setup.wall_seconds > 0
        assert payload.accounting.search.episodes == 10
        assert payload.accounting.search.wall_seconds > 0
        assert payload.accounting.replay.wall_seconds > 0
        assert payload.accounting.evaluator.calls == 1
        assert payload.accounting.evaluator.wall_seconds > 0
        assert payload.diagnostics["oracle_setup_calls"] == 1
        assert payload.accounting.replay.calls >= 1
        assert payload.accounting.search.actions + min(payload.accounting.probes.actions, 16) <= 256

    assert payloads["A0-no-probe-uniform"].accounting.probes.actions == 0
    assert payloads["A0-no-probe-uniform"].accounting.probes.wall_seconds == 0
    assert payloads["A1-paid-probe-uniform"].accounting.probes.actions == 16
    assert payloads["A1-paid-probe-uniform"].accounting.probes.wall_seconds > 0
    learned = [
        payloads["B1-clean-global-optimum-frequency"],
        payloads["B2-global-listwise-optimum"],
        payloads["C-state-conditioned-listwise-optimum"],
    ]
    assert {payload.accounting.probes.actions for payload in learned} == {656}
    assert all(payload.accounting.probes.wall_seconds > 0 for payload in learned)
    assert {payload.accounting.training.optimizer_steps for payload in learned} == {120}
    assert (
        learned[1].diagnostics["training_examples"] == learned[2].diagnostics["training_examples"]
    )
    global_parameters = learned[1].diagnostics["trainable_parameters"]
    state_parameters = learned[2].diagnostics["trainable_parameters"]
    assert isinstance(global_parameters, int)
    assert isinstance(state_parameters, int)
    assert abs(global_parameters - state_parameters) / state_parameters < 0.1


def test_phase2_executor_rejects_budget_drift_from_frozen_protocol() -> None:
    raw = build_phase2_baseline_smoke_config().model_dump(mode="json")
    raw["parameters"]["adaptation_action_cap"] = 257
    config = ExperimentConfig.model_validate(raw)
    planned = plan_expected_units(config).units[0]
    with pytest.raises(RuntimeError, match="adaptation_action_cap differs"):
        phase2_baseline_smoke_executor(config, planned)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda raw: raw["conditions"].pop(),
            "condition set differs",
        ),
        (
            lambda raw: raw["split"]["validation_tasks"][0].__setitem__(
                "task_id", "noncanonical-heldout"
            ),
            "validation split differs",
        ),
        (
            lambda raw: raw["conditions"][0]["exposure"].__setitem__(
                "optimum_threshold_access", True
            ),
            "condition exposure differs",
        ),
        (
            lambda raw: raw["seed_policy"].__setitem__("environment_seed_offset", 1),
            "environment seed offset differs",
        ),
        (
            lambda raw: raw["device_policy"].__setitem__("requested_device", "mps"),
            "device policy differs",
        ),
    ),
)
def test_phase2_executor_rejects_noncanonical_structure(mutate, message: str) -> None:
    raw = build_phase2_baseline_smoke_config().model_dump(mode="json")
    mutate(raw)
    config = ExperimentConfig.model_validate(raw)
    planned = plan_expected_units(config).units[0]
    with pytest.raises(RuntimeError, match=message):
        phase2_baseline_smoke_executor(config, planned)
