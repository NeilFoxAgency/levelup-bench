from __future__ import annotations

import pytest

from levelup.experiments.milestone6_phase2_shared_smoke import (
    B1,
    B2,
    B2_VARIANTS,
    LEARNED_BASES,
    ROOT,
    C,
    _default_optimum_provider,
    _representative_units,
    build_phase2_shared_smoke_config,
    phase2_shared_smoke_executor,
    prepare_phase2_shared_smoke,
    validate_phase2_shared_smoke_config,
)
from levelup.experiments.runner import ExperimentRunner, aggregate_run
from levelup.experiments.runner.records import PhaseAccounting, ResourceAccounting
from levelup.experiments.runner.training_artifacts import load_training_cost
from levelup.experiments.runner.training_data_artifacts import (
    load_training_data_evidence_cost,
    load_training_data_view_cost,
)


@pytest.fixture(scope="module")
def runtime(tmp_path_factory: pytest.TempPathFactory):
    events: list[str] = []
    output_root = tmp_path_factory.mktemp("phase2-shared-smoke")
    runtime = prepare_phase2_shared_smoke(
        output_root,
        repository=ROOT,
        event=events.append,
    )
    return runtime, events, output_root


def test_shared_smoke_config_is_development_locked() -> None:
    config = build_phase2_shared_smoke_config()
    validate_phase2_shared_smoke_config(config)

    assert config.split.final_tasks == ()
    assert config.parameters["not_scientific_result"] is True
    assert config.parameters["shared_artifact_training"] is True
    assert config.parameters["unit_local_training_repeated_and_counted"] is False
    assert {condition.condition_id for condition in config.conditions} == {
        "A0-no-probe-uniform",
        "A1-paid-probe-uniform",
        B1,
        C,
        *B2_VARIANTS,
    }
    assert config.replicates == 1
    assert all(condition.execution_phases == ("validation",) for condition in config.conditions)
    unlocked = config.model_copy(
        update={
            "split": config.split.model_copy(update={"final_tasks": config.split.validation_tasks})
        }
    )
    with pytest.raises(RuntimeError, match="final tasks"):
        validate_phase2_shared_smoke_config(unlocked)
    b2_index = next(
        index
        for index, condition in enumerate(config.conditions)
        if condition.condition_id in B2_VARIANTS
    )
    wrong_temperature = config.model_copy(
        update={
            "conditions": tuple(
                condition.model_copy(
                    update={
                        "parameters": {
                            **condition.parameters,
                            "search_temperature": 999.0,
                        }
                    }
                )
                if index == b2_index
                else condition
                for index, condition in enumerate(config.conditions)
            )
        }
    )
    with pytest.raises(RuntimeError, match="search temperature"):
        validate_phase2_shared_smoke_config(wrong_temperature)


def test_shared_smoke_materializes_one_evidence_three_views_and_three_models(runtime) -> None:
    prepared, events, _ = runtime
    run_dir = prepared.store.run_dir

    assert len(list((run_dir / "training-data-evidence").iterdir())) == 1
    assert len(list((run_dir / "training-data-artifacts").iterdir())) == 3
    assert len(list((run_dir / "training-artifacts").iterdir())) == 3
    assert len(list((run_dir / "training-data-evidence-costs").iterdir())) == 1
    assert len(list((run_dir / "training-data-view-costs").iterdir())) == 3
    assert len(list((run_dir / "training-artifact-costs").iterdir())) == 3

    assert prepared.data.evidence_id
    assert len(prepared.data.views) == 3
    assert set(prepared.data.views) == set(LEARNED_BASES)
    assert len(prepared.models.keys) == 3
    assert set(prepared.models.keys) == set(LEARNED_BASES)
    assert len({manifest.artifact_id for _, manifest in prepared.data.views.values()}) == 3
    assert len({manifest.evidence_id for _, manifest in prepared.data.views.values()}) == 1
    assert {item for item in events if item == "evidence_build"} == {"evidence_build"}
    assert sum(item.startswith("view_materialized:") for item in events) == 3
    assert sum(item.startswith("model_train:") for item in events) == 3
    evidence_cost = load_training_data_evidence_cost(run_dir, prepared.data.evidence_key)
    assert evidence_cost.scope == "training_data_evidence_preparation"
    assert evidence_cost.accounting.training == PhaseAccounting()
    for base, (key, _) in prepared.data.views.items():
        view_cost = load_training_data_view_cost(run_dir, key)
        assert view_cost.scope == "training_data_view_preparation"
        assert view_cost.accounting.training_probes == PhaseAccounting()
        assert view_cost.accounting.reference_replay == PhaseAccounting()
        assert load_training_cost(run_dir, prepared.models.keys[base]).scope == (
            "training_preparation"
        )

    model_plans = [
        plan
        for plan in prepared.store.expected_shared.artifacts
        if plan.kind == "training_artifact"
    ]
    b2_plan = next(plan for plan in model_plans if plan.owner_group_id == B2)
    assert set(b2_plan.consumer_condition_ids) == set(B2_VARIANTS)
    assert len(b2_plan.consumer_condition_ids) == len(B2_VARIANTS) == 3
    assert {
        condition.parameters["search_temperature"]
        for condition in prepared.config.conditions
        if condition.condition_id in B2_VARIANTS
    } == set(B2_VARIANTS.values())
    assert "search_temperature" not in prepared.models.keys[B2].model_dump(mode="json")
    assert (
        prepared.models.manifests[B2].report.training_examples
        == prepared.models.manifests[C].report.training_examples
    )
    assert (
        prepared.models.manifests[B2].report.optimizer_steps
        == prepared.models.manifests[C].report.optimizer_steps
    )


def test_shared_smoke_resume_reuses_preparation_and_strict_aggregate(runtime) -> None:
    prepared, events, output_root = runtime
    runner = ExperimentRunner(prepared.store)
    calls = 0

    def interrupting_executor(planned):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return phase2_shared_smoke_executor(prepared, planned)

    with pytest.raises(KeyboardInterrupt):
        runner.execute(
            interrupting_executor,
            phases=("validation",),
        )
    assert calls == 2
    assert len(prepared.store.completed_records()) == 1
    assert events.count("evidence_build") == 1
    assert sum(item.startswith("model_train:") for item in events) == 3

    resume_events: list[str] = []
    resumed = prepare_phase2_shared_smoke(
        output_root,
        repository=ROOT,
        event=resume_events.append,
    )
    completed = ExperimentRunner(resumed.store).execute(
        lambda planned: phase2_shared_smoke_executor(resumed, planned),
        phases=("validation",),
    )
    assert completed["completed"] == len(resumed.store.expected.units) - 1
    assert completed["skipped"] == 1
    assert resume_events == [
        "training_data_loaded",
        f"model_loaded:{B1}",
        f"model_loaded:{B2}",
        f"model_loaded:{C}",
    ]
    assert all(
        record.accounting.training == PhaseAccounting()
        for record in resumed.store.completed_records()
    )
    assert events.count("evidence_build") == 1
    assert sum(item.startswith("model_train:") for item in events) == 3

    third = ExperimentRunner(resumed.store).execute(
        lambda planned: phase2_shared_smoke_executor(resumed, planned),
        phases=("validation",),
    )
    assert third["completed"] == 0
    assert third["skipped"] == len(resumed.store.expected.units)
    assert events.count("evidence_build") == 1
    assert sum(item.startswith("model_train:") for item in events) == 3

    aggregate = aggregate_run(resumed.store, strict=True)
    assert aggregate.complete is True
    assert aggregate.inventory.missing == 0
    assert aggregate.inventory.interrupted_attempts == 1
    assert aggregate.shared_inventory.planned == 7
    assert aggregate.shared_inventory.referenced == 7
    assert aggregate.shared_inventory.complete is True
    assert aggregate.shared_artifacts_sha256
    learned_records = [
        record
        for record in resumed.store.completed_records()
        if record.key.condition_id in {*LEARNED_BASES, *B2_VARIANTS}
    ]
    assert learned_records
    assert all(
        record.accounting.training == ResourceAccounting().training for record in learned_records
    )
    b2_model_references = {
        (
            reference.key_id,
            reference.artifact_id,
            reference.cost_id,
        )
        for record in resumed.store.completed_records()
        if record.key.condition_id in B2_VARIANTS
        for reference in record.shared_artifacts
        if reference.kind == "training_artifact"
    }
    assert len(b2_model_references) == 1


def test_shared_smoke_oracle_event_does_not_change_candidate_hash(runtime) -> None:
    prepared, _, _ = runtime
    planned = _representative_units(prepared.config)[B1]
    first_events: list[str] = []
    second_events: list[str] = []

    first = phase2_shared_smoke_executor(
        prepared,
        planned,
        optimum_provider=_default_optimum_provider,
        event=first_events.append,
    )

    def shifted_oracle(environment, family_id: str) -> float:
        return _default_optimum_provider(environment, family_id) + 0.25

    second = phase2_shared_smoke_executor(
        prepared,
        planned,
        optimum_provider=shifted_oracle,
        event=second_events.append,
    )

    assert first.candidate_generation_sha256 == second.candidate_generation_sha256
    assert first.shared_artifacts == second.shared_artifacts
    assert first.accounting.training == second.accounting.training == PhaseAccounting()
    assert first.accounting.probes.actions == second.accounting.probes.actions
    assert first.accounting.search.model_dump(exclude={"wall_seconds"}) == (
        second.accounting.search.model_dump(exclude={"wall_seconds"})
    )
    assert first.accounting.replay.model_dump(exclude={"wall_seconds"}) == (
        second.accounting.replay.model_dump(exclude={"wall_seconds"})
    )
    assert first_events == [
        "generation_complete",
        "candidate_evaluation_complete",
        "optimum_oracle",
    ]
    assert second_events == first_events
    assert first.diagnostics["oracle_setup_calls"] == 1
    assert second.diagnostics["oracle_setup_calls"] == 1
