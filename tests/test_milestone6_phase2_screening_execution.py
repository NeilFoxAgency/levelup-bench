from __future__ import annotations

from types import SimpleNamespace

import pytest

import levelup.experiments.milestone6_phase2_screening_execution as execution
from levelup.experiments.runner.config import (
    ConditionSpec,
    DevicePolicy,
    ExperimentConfig,
    ExposureSpec,
    MetricSpec,
    SeedPolicy,
    SelectionSpec,
    SplitSpec,
    TaskIdentity,
    scientific_config_sha256,
)
from levelup.experiments.runner.records import (
    ExpectedUnits,
    PhaseAccounting,
    PlannedUnit,
    SharedArtifactReference,
    UnitKey,
    UnitSeeds,
)


def _config(condition_id: str = "A0-no-probe-uniform") -> ExperimentConfig:
    task = TaskIdentity(
        family_id="plain",
        task_id="plain-validation-0",
        task_index=0,
        generator_seed=11,
        environment_reset_seed=12,
    )
    exposure = ExposureSpec(
        train_task_ids=(),
        observable_state_access="none",
        action_history_access=False,
        action_descriptors_access=True,
        probe_interaction_access=condition_id != "A0-no-probe-uniform",
        search_feedback_access=False,
        evaluator_output_access=False,
        optimum_threshold_access=False,
        privileged_state_access=False,
        structured_constraint_access=False,
    )
    condition = ConditionSpec(
        condition_id=condition_id,
        learner_id="uniform-visible-actions-v1",
        execution_phases=("validation",),
        exposure=exposure,
        parameters={
            "probe_action_cap": 0 if condition_id.startswith("A0") else 64,
        },
    )
    return ExperimentConfig(
        experiment_id="test-screening",
        method_revision="test",
        split=SplitSpec(development_tasks=(TaskIdentity(
            family_id="plain", task_id="plain-training-0", task_index=1, generator_seed=13
        ),), validation_tasks=(task,)),
        conditions=(condition,),
        replicates=1,
        seed_policy=SeedPolicy(
            derivation_version="phase2.v1",
            model_seed_base=1,
            probe_seed_base=2,
            search_seed_base=3,
            data_order_seed_base=4,
        ),
        device_policy=DevicePolicy(requested_device="cpu", torch_threads=1),
        metrics=(MetricSpec(
            metric_id="performance_value",
            direction="minimize",
            unit="ticks",
            description="test",
        ),),
        selection=SelectionSpec(
            phases=("validation",), primary_metric="performance_value", rule="test"
        ),
        parameters={
            "heldout_family_id": "plain",
            "probe_action_cap": 64,
            "candidate_episodes": 150,
            "adaptation_action_cap": 2048,
            "maximum_actions_per_candidate_episode": 64,
            "probe_actions_per_attempt": 16,
            "probe_coverage_target_samples_per_alias": 8,
        },
    )


def _fold_and_unit(config: ExperimentConfig) -> tuple[SimpleNamespace, PlannedUnit]:
    unit = PlannedUnit(
        unit_id="a" * 64,
        key=UnitKey(
            phase="validation",
            condition_id=config.conditions[0].condition_id,
            family_id="plain",
            task_id="plain-validation-0",
            task_index=0,
            replicate=0,
        ),
        seeds=UnitSeeds(
            model_seed=1,
            environment_seed=12,
            probe_seed=3,
            search_seed=4,
            data_order_seed=5,
        ),
        exposure_manifest_sha256="b" * 64,
    )
    expected = ExpectedUnits(
        run_id="test",
        config_sha256=scientific_config_sha256(config),
        units=(unit,),
    )
    store = SimpleNamespace(
        _execution_ready=True,
        config_sha256=scientific_config_sha256(config),
        expected=expected,
        planned_unit=lambda unit_id: unit if unit_id == unit.unit_id else None,
    )
    return SimpleNamespace(family_id="plain", config=config, store=store), unit


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, *, success: bool = False) -> None:
    monkeypatch.setattr(execution, "validate_screening_child_config", lambda config: None)
    monkeypatch.setattr(
        execution,
        "_task",
        lambda config, task_id: config.split.validation_tasks[0],
    )
    monkeypatch.setattr(execution, "_environment", lambda task: SimpleNamespace())
    monkeypatch.setattr(execution, "_forbidden_aliases", lambda environment: frozenset())
    monkeypatch.setattr(
        execution,
        "prepare_unit_model",
        lambda *args: None,
    )
    monkeypatch.setattr(
        execution,
        "generate_candidates_with_observable_policy",
        lambda environment, **kwargs: SimpleNamespace(
            candidates=(SimpleNamespace(episode=1, adaptation_actions=1, trajectory=object()),),
            accounting=SimpleNamespace(
                episodes=1,
                resets=1,
                actions=1,
                forward_passes=0,
                unknown_affordance_decisions=0,
                wall_seconds=0.0,
            ),
        ),
    )
    monkeypatch.setattr(execution, "trajectory_content_sha256", lambda trajectory: "c" * 64)
    monkeypatch.setattr(
        execution,
        "evaluate_generated_search",
        lambda generated, evaluator: SimpleNamespace(
            evaluated_candidates=(SimpleNamespace(),),
            first_valid_episode=1,
            best_performance=3.0,
            accounting=SimpleNamespace(
                episodes=1,
                resets=1,
                actions=1,
                forward_passes=0,
                evaluator_calls=1,
                evaluator_replay_actions=2,
                unknown_affordance_decisions=0,
                generation_wall_seconds=0.0,
                evaluator_wall_seconds=0.0,
            ),
        ),
    )
    monkeypatch.setattr(
        execution,
        "classify_exact_optimum",
        lambda search, *, optimum_performance: SimpleNamespace(
            first_episode=1 if success else None,
            first_adaptation_actions=1 if success else None,
            success=success,
        ),
    )


def test_a0_is_posthoc_oracle_only_and_failure_is_fixed_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    fold, planned = _fold_and_unit(config)
    _patch_runtime(monkeypatch)
    events: list[str] = []
    payload = execution.execute_screening_unit(
        fold,
        planned,
        optimum_provider=lambda environment, family: 3.0,
        event=events.append,
    )
    assert payload.accounting.training == PhaseAccounting()
    assert payload.shared_artifacts == ()
    assert payload.outcome.censored is True
    assert payload.outcome.censoring_budget == 2048
    assert payload.outcome.censoring_reason == "fixed_endpoint"
    assert payload.diagnostics["development_screening"] is True
    assert events == ["generation_complete", "candidate_evaluation_complete", "optimum_oracle"]


def test_execution_requires_open_store_and_rejects_final_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    fold, planned = _fold_and_unit(config)
    _patch_runtime(monkeypatch)
    fold.store._execution_ready = False
    with pytest.raises(RuntimeError, match="execution-ready"):
        execution.execute_screening_unit(fold, planned)
    fold.store._execution_ready = True
    final = planned.model_copy(update={"key": planned.key.model_copy(update={"phase": "final"})})
    with pytest.raises(RuntimeError, match="validation"):
        execution.execute_screening_unit(fold, final)


def test_a1_pays_probe_and_passes_only_observable_generation_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config("A1-paid-probe-uniform")
    fold, planned = _fold_and_unit(config)
    _patch_runtime(monkeypatch)

    probe_calls: list[dict[str, object]] = []
    affordances = SimpleNamespace(features={"visible": (0.0,)}, sample_counts={"visible": 1})
    monkeypatch.setattr(
        execution,
        "discover_affordances",
        lambda environment, **kwargs: (
            probe_calls.append(kwargs)
            or SimpleNamespace(
                affordances=affordances,
                accounting=SimpleNamespace(
                    attempts=4,
                    resets=4,
                    actions=64,
                    wall_seconds=0.25,
                ),
            )
        ),
    )
    generation_calls: list[dict[str, object]] = []

    def generate(environment, **kwargs):
        generation_calls.append(kwargs)
        return SimpleNamespace(
            candidates=(SimpleNamespace(episode=2, adaptation_actions=65, trajectory=object()),),
            accounting=SimpleNamespace(
                episodes=2,
                resets=2,
                actions=1,
                forward_passes=0,
                unknown_affordance_decisions=0,
                wall_seconds=0.1,
            ),
        )

    monkeypatch.setattr(execution, "generate_candidates_with_observable_policy", generate)
    payload = execution.execute_screening_unit(
        fold,
        planned,
        optimum_provider=lambda environment, family: 99.0,
    )

    assert len(probe_calls) == 1
    assert probe_calls[0] == {
        "task_id": "plain-validation-0",
        "forbidden_aliases": frozenset(),
        "seed": planned.seeds.probe_seed,
        "action_cap": 64,
        "target_samples_per_alias": 8,
        "actions_per_attempt": 16,
    }
    assert len(generation_calls) == 1
    generation = generation_calls[0]
    assert generation["task_id"] == "plain-validation-0"
    assert generation["forbidden_aliases"] == frozenset()
    assert generation["affordances"] is affordances
    assert generation["model"] is None
    assert generation["seed"] == planned.seeds.search_seed
    assert generation["temperature"] == pytest.approx(0.9)
    assert generation["max_episodes"] == execution.SCREENING_CANDIDATE_EPISODES
    assert generation["max_actions_per_episode"] == execution.SCREENING_MAX_ACTIONS_PER_EPISODE
    assert generation["total_adaptation_action_cap"] == execution.SCREENING_ADAPTATION_ACTION_CAP
    assert generation["prior_adaptation_actions"] == 64
    assert generation["condition_id"] == "A1-paid-probe-uniform"
    assert payload.accounting.probes == PhaseAccounting(
        calls=4,
        actions=64,
        environment_steps=64,
        resets=4,
        wall_seconds=0.25,
    )
    assert payload.accounting.training == PhaseAccounting()
    assert payload.shared_artifacts == ()


def test_learned_unit_passes_prepared_model_and_typed_references_without_local_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    condition_id = "B1-clean-global-optimum-frequency--lr0p003-e120-t0p6"
    config = _config(condition_id)
    condition = config.conditions[0].model_copy(
        update={
            "parameters": {
                "probe_action_cap": 64,
                "search_temperature": 0.6,
            }
        }
    )
    config = config.model_copy(update={"conditions": (condition,)})
    fold, planned = _fold_and_unit(config)
    _patch_runtime(monkeypatch)
    model = object()
    report = SimpleNamespace(trainable_parameters=123, training_examples=456)
    references = (
        SharedArtifactReference(
            kind="training_data_evidence",
            key_id="1" * 64,
            artifact_id="2" * 64,
            cost_id="3" * 64,
        ),
        SharedArtifactReference(
            kind="training_data_view",
            key_id="4" * 64,
            artifact_id="5" * 64,
            cost_id="6" * 64,
        ),
        SharedArtifactReference(
            kind="training_artifact",
            key_id="7" * 64,
            artifact_id="8" * 64,
            cost_id="9" * 64,
        ),
    )
    monkeypatch.setattr(
        execution,
        "prepare_unit_model",
        lambda *args: SimpleNamespace(model=model, report=report, references=references),
    )
    monkeypatch.setattr(
        execution,
        "discover_affordances",
        lambda environment, **kwargs: SimpleNamespace(
            affordances=SimpleNamespace(features={}, sample_counts={}),
            accounting=SimpleNamespace(
                attempts=1,
                resets=1,
                actions=64,
                wall_seconds=0.0,
            ),
        ),
    )
    generation_calls: list[dict[str, object]] = []

    def generate(environment, **kwargs):
        generation_calls.append(kwargs)
        return SimpleNamespace(
            candidates=(SimpleNamespace(episode=1, adaptation_actions=1, trajectory=object()),),
            accounting=SimpleNamespace(
                episodes=1,
                resets=1,
                actions=1,
                forward_passes=3,
                unknown_affordance_decisions=2,
                wall_seconds=0.0,
            ),
        )

    monkeypatch.setattr(execution, "generate_candidates_with_observable_policy", generate)
    payload = execution.execute_screening_unit(
        fold,
        planned,
        optimum_provider=lambda environment, family: 99.0,
    )

    assert payload.shared_artifacts == references
    assert payload.accounting.training == PhaseAccounting()
    assert generation_calls[0]["model"] is model
    assert generation_calls[0]["temperature"] == pytest.approx(0.6)
    assert payload.diagnostics["trainable_parameters"] == 123
    assert payload.diagnostics["training_examples"] == 456
    assert payload.diagnostics["shared_training_artifact"] is True


def test_success_records_typed_first_optimum_and_oracle_substitution_preserves_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    fold, planned = _fold_and_unit(config)
    _patch_runtime(monkeypatch)
    reports = {
        3.0: SimpleNamespace(first_episode=1, first_adaptation_actions=17, success=True),
        4.0: SimpleNamespace(first_episode=None, first_adaptation_actions=None, success=False),
    }
    monkeypatch.setattr(
        execution,
        "classify_exact_optimum",
        lambda search, *, optimum_performance: reports[optimum_performance],
    )
    first = execution.execute_screening_unit(
        fold,
        planned,
        optimum_provider=lambda environment, family: 3.0,
    )
    second = execution.execute_screening_unit(
        fold,
        planned,
        optimum_provider=lambda environment, family: 4.0,
    )

    assert first.candidate_generation_sha256 == second.candidate_generation_sha256
    assert first.outcome.success is True
    assert first.outcome.censored is False
    assert first.outcome.first_optimum_episode == 1
    assert first.outcome.first_optimum_adaptation_actions == 17
    assert first.outcome.censoring_budget is None
    assert second.outcome.success is False
    assert second.outcome.censored is True
    assert second.outcome.first_optimum_episode is None
    assert second.outcome.first_optimum_adaptation_actions is None
    assert second.outcome.censoring_budget == execution.SCREENING_ADAPTATION_ACTION_CAP
    assert second.outcome.censoring_reason == "fixed_endpoint"


def test_mutated_store_config_identity_is_rejected_before_unit_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    fold, planned = _fold_and_unit(config)
    _patch_runtime(monkeypatch)
    fold.store.config_sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="config identity"):
        execution.execute_screening_unit(fold, planned)
