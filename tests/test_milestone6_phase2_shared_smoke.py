from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

import levelup.experiments.milestone6_phase2 as phase2
import levelup.experiments.milestone6_phase2_shared_smoke as shared_smoke
from levelup.experiments.milestone6_baselines import ExactOptimumReport
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
from levelup.experiments.runner import (
    ExperimentRunner,
    aggregate_run,
    build_selection_metric_spec,
    load_selection_authority,
)
from levelup.experiments.runner.config import (
    TaskIdentity,
    run_id_for,
    scientific_config_sha256,
)
from levelup.experiments.runner.records import (
    ExpectedSharedArtifacts,
    PhaseAccounting,
    PlannedSharedArtifact,
    ResourceAccounting,
)
from levelup.experiments.runner.selection_metric import _AUTHORITY_CONSTRUCTION_TOKEN
from levelup.experiments.runner.storage import plan_expected_units
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
        model_cost = load_training_cost(run_dir, prepared.models.keys[base])
        assert model_cost.scope == "training_preparation"
        assert model_cost.accounting.training.wall_seconds > 0

    assert {
        manifest.evidence_id for _, manifest in prepared.data.views.values()
    } == {prepared.data.evidence_id}

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


def test_shared_smoke_typed_first_hit_and_fixed_endpoint_censoring(runtime, monkeypatch) -> None:
    prepared, _, _ = runtime
    planned = _representative_units(prepared.config)[B1]

    monkeypatch.setattr(
        shared_smoke,
        "classify_exact_optimum",
        lambda outcome, *, optimum_performance: ExactOptimumReport(
            first_episode=2,
            first_adaptation_actions=19,
            success=True,
        ),
    )
    successful = phase2_shared_smoke_executor(
        prepared,
        planned,
        optimum_provider=lambda environment, family_id: 123.0,
    )
    assert successful.outcome.success is True
    assert successful.outcome.censored is False
    assert successful.outcome.first_optimum_episode == 2
    assert successful.outcome.first_optimum_adaptation_actions == 19
    assert successful.diagnostics["first_optimum_adaptation_actions"] == 19

    monkeypatch.setattr(
        shared_smoke,
        "classify_exact_optimum",
        lambda outcome, *, optimum_performance: ExactOptimumReport(
            first_episode=None,
            first_adaptation_actions=None,
            success=False,
        ),
    )
    failed = phase2_shared_smoke_executor(
        prepared,
        planned,
        optimum_provider=lambda environment, family_id: 456.0,
    )
    assert failed.outcome.success is False
    assert failed.outcome.censored is True
    assert failed.outcome.first_optimum_episode is None
    assert failed.outcome.first_optimum_adaptation_actions is None
    assert failed.outcome.censoring_reason == "fixed_endpoint"
    assert failed.outcome.censoring_budget == prepared.config.parameters[
        "adaptation_action_cap"
    ]
    assert failed.diagnostics["first_optimum_adaptation_actions"] is None


def test_oracle_substitution_preserves_candidate_hash_and_typed_first_hit_semantics(
    runtime, monkeypatch
) -> None:
    prepared, _, _ = runtime
    planned = _representative_units(prepared.config)[B1]
    reports = {
        123.0: ExactOptimumReport(
            first_episode=1,
            first_adaptation_actions=17,
            success=True,
        ),
        456.0: ExactOptimumReport(
            first_episode=None,
            first_adaptation_actions=None,
            success=False,
        ),
    }
    monkeypatch.setattr(
        shared_smoke,
        "classify_exact_optimum",
        lambda outcome, *, optimum_performance: reports[optimum_performance],
    )
    first = phase2_shared_smoke_executor(
        prepared,
        planned,
        optimum_provider=lambda environment, family_id: 123.0,
    )
    second = phase2_shared_smoke_executor(
        prepared,
        planned,
        optimum_provider=lambda environment, family_id: 456.0,
    )

    assert first.candidate_generation_sha256 == second.candidate_generation_sha256
    assert first.outcome.first_optimum_adaptation_actions == 17
    assert first.diagnostics["first_optimum_adaptation_actions"] == 17
    assert second.outcome.first_optimum_adaptation_actions is None
    assert second.outcome.censoring_reason == "fixed_endpoint"


def test_shared_smoke_protocol_and_screening_hashes_bind_config_and_run_identity() -> None:
    config = build_phase2_shared_smoke_config()
    protocol_path = ROOT / "configs" / "milestone6" / "development_protocol.json"
    screening_path = ROOT / "configs" / "milestone6" / "phase2_screening_candidates.json"
    task_manifest_path = ROOT / "configs" / "milestone6" / "development_tasks.json"
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    screening_sha256 = hashlib.sha256(screening_path.read_bytes()).hexdigest()
    task_manifest_sha256 = hashlib.sha256(task_manifest_path.read_bytes()).hexdigest()

    assert config.parameters["development_protocol_sha256"] == protocol_sha256
    assert config.parameters["screening_candidates_sha256"] == screening_sha256
    assert config.parameters["development_task_manifest_sha256"] == task_manifest_sha256
    assert config.parameters["selection_metric_id"] == (
        "total_adaptation_actions_to_first_exact_optimum"
    )
    assert config.parameters["selection_metric_action_formula"] == (
        "accounting.probes.actions + accounting.search.actions"
    )
    assert config.parameters["selection_metric_phase"] == "validation"
    config_sha256 = scientific_config_sha256(config)
    assert run_id_for(config).endswith(config_sha256[:12])

    for field in (
        "development_protocol_sha256",
        "screening_candidates_sha256",
        "development_task_manifest_sha256",
    ):
        changed_parameters = {
            **config.parameters,
            field: "0" * 64,
        }
        changed = config.model_copy(update={"parameters": changed_parameters})
        assert scientific_config_sha256(changed) != config_sha256
        assert run_id_for(changed) != run_id_for(config)

    authority = load_selection_authority(protocol_path, screening_path, task_manifest_path)
    assert authority.protocol_sha256 == protocol_sha256
    assert authority.screening_candidates_sha256 == screening_sha256
    assert authority.task_manifest_sha256 == task_manifest_sha256
    assert authority.endpoint == 2048
    assert authority.failure_sentinel == 2049

    drifted = config.model_copy(
        update={
            "selection": config.selection.model_copy(
                update={"primary_metric": "drifted"}
            )
        }
    )
    with pytest.raises(RuntimeError, match="selection declaration"):
        validate_phase2_shared_smoke_config(drifted)


def test_selection_authority_rejects_consistent_cross_file_protocol_tampering(
    tmp_path,
) -> None:
    protocol = json.loads(
        (ROOT / "configs" / "milestone6" / "development_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    screening = json.loads(
        (ROOT / "configs" / "milestone6" / "phase2_screening_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    task_manifest = json.loads(
        (ROOT / "configs" / "milestone6" / "development_tasks.json").read_text(
            encoding="utf-8"
        )
    )

    cases = []

    changed_protocol = copy.deepcopy(protocol)
    changed_screening = copy.deepcopy(screening)
    changed_manifest = copy.deepcopy(task_manifest)
    reordered = [*changed_protocol["family_order"][1:], changed_protocol["family_order"][0]]
    changed_protocol["family_order"] = reordered
    changed_screening["folds"]["family_order"] = reordered
    changed_manifest["family_order"] = reordered
    cases.append(("family-order", changed_protocol, changed_screening, changed_manifest))

    changed_protocol = copy.deepcopy(protocol)
    changed_screening = copy.deepcopy(screening)
    changed_manifest = copy.deepcopy(task_manifest)
    changed_protocol["seed_policy"]["screening_replicates"] = [0, 1]
    changed_screening["folds"]["replicates"] = [0, 1]
    cases.append(("replicate-set", changed_protocol, changed_screening, changed_manifest))

    changed_protocol = copy.deepcopy(protocol)
    changed_screening = copy.deepcopy(screening)
    changed_manifest = copy.deepcopy(task_manifest)
    changed_manifest["tasks"][0]["roles"].append("final_evaluation")
    cases.append(("final-role", changed_protocol, changed_screening, changed_manifest))

    changed_protocol = copy.deepcopy(protocol)
    changed_screening = copy.deepcopy(screening)
    changed_manifest = copy.deepcopy(task_manifest)
    changed_screening["candidate_tuples"].pop()
    cases.append(("candidate-grid", changed_protocol, changed_screening, changed_manifest))

    changed_protocol = copy.deepcopy(protocol)
    changed_screening = copy.deepcopy(screening)
    changed_manifest = copy.deepcopy(task_manifest)
    changed_screening["fixed_controls"].pop()
    cases.append(("fixed-controls", changed_protocol, changed_screening, changed_manifest))

    changed_protocol = copy.deepcopy(protocol)
    changed_screening = copy.deepcopy(screening)
    changed_manifest = copy.deepcopy(task_manifest)
    changed_screening["learned_conditions"][0]["objective"] = "drifted"
    cases.append(("learned-condition", changed_protocol, changed_screening, changed_manifest))

    changed_protocol = copy.deepcopy(protocol)
    changed_screening = copy.deepcopy(screening)
    changed_manifest = copy.deepcopy(task_manifest)
    changed_protocol["freeze_record"]["comparative_results_inspected_before_amendment"] = True
    cases.append(("freeze-record", changed_protocol, changed_screening, changed_manifest))

    changed_protocol = copy.deepcopy(protocol)
    changed_screening = copy.deepcopy(screening)
    changed_manifest = copy.deepcopy(task_manifest)
    changed_protocol["representation_ladder_stage_contract"]["history_sequence"][
        "claims_before_gate"
    ] = "allowed"
    cases.append(("representation-gate", changed_protocol, changed_screening, changed_manifest))

    for name, changed_protocol, changed_screening, changed_manifest in cases:
        case_root = tmp_path / name
        case_root.mkdir()
        protocol_path = case_root / "development_protocol.json"
        screening_path = case_root / "phase2_screening_candidates.json"
        task_manifest_path = case_root / "development_tasks.json"
        task_manifest_bytes = json.dumps(changed_manifest, sort_keys=True).encode()
        task_manifest_path.write_bytes(task_manifest_bytes)
        changed_screening["task_manifest"]["sha256"] = hashlib.sha256(
            task_manifest_bytes
        ).hexdigest()
        protocol_bytes = json.dumps(changed_protocol, sort_keys=True).encode()
        protocol_path.write_bytes(protocol_bytes)
        changed_screening["parent_protocol"]["sha256"] = hashlib.sha256(
            protocol_bytes
        ).hexdigest()
        screening_path.write_text(json.dumps(changed_screening, sort_keys=True), encoding="utf-8")

        with pytest.raises(ValueError):
            load_selection_authority(protocol_path, screening_path, task_manifest_path)


def test_selection_spec_builder_binds_authority_endpoint_and_shared_key_plan() -> None:
    protocol_path = ROOT / "configs" / "milestone6" / "development_protocol.json"
    screening_path = ROOT / "configs" / "milestone6" / "phase2_screening_candidates.json"
    task_manifest_path = ROOT / "configs" / "milestone6" / "development_tasks.json"
    authority = load_selection_authority(
        protocol_path,
        screening_path,
        task_manifest_path,
    )
    base = build_phase2_shared_smoke_config()
    manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    heldout_tasks = tuple(
        TaskIdentity(
            family_id=item["family"],
            task_id=item["task_id"],
            task_index=item["task_index"],
            generator_seed=item["generator_seed"],
            environment_reset_seed=item["environment_reset_seed"],
        )
        for item in manifest["tasks"]
        if item["family"] == "combo" and "training_core" in item["roles"]
    )
    parameters = {
        **base.parameters,
        "adaptation_action_cap": authority.endpoint,
        "selection_metric_failure_sentinel": authority.failure_sentinel,
    }
    config = base.model_copy(
        update={
            "parameters": parameters,
            "replicates": len(authority.screening_replicates),
            "split": base.split.model_copy(
                update={"validation_tasks": heldout_tasks}
            ),
        }
    )
    expected = plan_expected_units(config)
    learned_units = tuple(
        unit for unit in expected.units if unit.key.condition_id == B1
    )
    config_sha256 = scientific_config_sha256(config)
    expected_shared = ExpectedSharedArtifacts(
        run_id=run_id_for(config),
        config_sha256=config_sha256,
        artifacts=tuple(
            PlannedSharedArtifact(
                kind=kind,
                key_id=f"{replicate * 3 + kind_index + 1:064x}",
                owner_condition_id=B1,
                owner_group_id=(
                    "canonical-evidence"
                    if kind == "training_data_evidence"
                    else B1
                ),
                owner_family_id="combo",
                owner_fold_id="lofo-combo",
                owner_replicate=replicate,
                consumer_phase="validation",
                consumer_condition_ids=(B1,),
                consumer_unit_ids=tuple(
                    unit.unit_id
                    for unit in learned_units
                    if unit.key.replicate == replicate
                ),
            )
            for replicate in authority.screening_replicates
            for kind_index, kind in enumerate(
                (
                    "training_data_evidence",
                    "training_data_view",
                    "training_artifact",
                )
            )
        ),
    )
    spec = build_selection_metric_spec(
        config,
        expected,
        expected_shared,
        authority,
        condition_id=B1,
    )
    assert spec.endpoint == 2048
    assert spec.protocol_sha256 == authority.protocol_sha256
    assert spec.screening_candidates_sha256 == authority.screening_candidates_sha256
    assert spec.task_manifest_sha256 == authority.task_manifest_sha256
    assert len(spec.expected_units) == 40
    assert spec.family_universe == authority.family_ids
    assert spec.has_complete_family_coverage is False

    forged_authority = replace(
        authority,
        family_ids=("combo",),
        _construction_token=_AUTHORITY_CONSTRUCTION_TOKEN,
    )
    with pytest.raises(ValueError, match="canonical frozen files"):
        build_selection_metric_spec(
            config,
            expected,
            expected_shared,
            forged_authority,
            condition_id=B1,
        )

    for field, value in (
        ("development_protocol_sha256", "0" * 64),
        ("screening_candidates_sha256", "0" * 64),
        ("development_task_manifest_sha256", "0" * 64),
        ("adaptation_action_cap", 1024),
    ):
        changed_parameters = {**parameters, field: value}
        if field == "adaptation_action_cap":
            changed_parameters["selection_metric_failure_sentinel"] = 1025
        changed = config.model_copy(update={"parameters": changed_parameters})
        changed_expected = plan_expected_units(changed)
        changed_shared = expected_shared.model_copy(
            update={
                "run_id": run_id_for(changed),
                "config_sha256": scientific_config_sha256(changed),
            }
        )
        with pytest.raises(ValueError, match="frozen selection metric"):
            build_selection_metric_spec(
                changed,
                changed_expected,
                changed_shared,
                authority,
                condition_id=B1,
            )

    final_task = heldout_tasks[0].model_copy(
        update={"task_id": f"{heldout_tasks[0].task_id}.forbidden-final"}
    )
    final_config = config.model_copy(
        update={
            "split": config.split.model_copy(update={"final_tasks": (final_task,)})
        }
    )
    final_expected = plan_expected_units(final_config)
    final_shared = expected_shared.model_copy(
        update={
            "run_id": run_id_for(final_config),
            "config_sha256": scientific_config_sha256(final_config),
        }
    )
    with pytest.raises(ValueError, match="forbidden final-family"):
        build_selection_metric_spec(
            final_config,
            final_expected,
            final_shared,
            authority,
            condition_id=B1,
        )

    tampered_training_task = config.split.development_tasks[0].model_copy(
        update={"generator_seed": config.split.development_tasks[0].generator_seed + 1}
    )
    tampered_training_config = config.model_copy(
        update={
            "split": config.split.model_copy(
                update={
                    "development_tasks": (
                        tampered_training_task,
                        *config.split.development_tasks[1:],
                    )
                }
            )
        }
    )
    tampered_training_expected = plan_expected_units(tampered_training_config)
    tampered_training_shared = expected_shared.model_copy(
        update={
            "run_id": run_id_for(tampered_training_config),
            "config_sha256": scientific_config_sha256(tampered_training_config),
        }
    )
    with pytest.raises(ValueError, match="frozen LOFO fold"):
        build_selection_metric_spec(
            tampered_training_config,
            tampered_training_expected,
            tampered_training_shared,
            authority,
            condition_id=B1,
        )

    wrong_owner_shared = expected_shared.model_copy(
        update={
            "artifacts": (
                expected_shared.artifacts[0].model_copy(
                    update={"owner_group_id": "wrong-owner"}
                ),
                *expected_shared.artifacts[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="owner lineage"):
        build_selection_metric_spec(
            config,
            expected,
            wrong_owner_shared,
            authority,
            condition_id=B1,
        )


@pytest.mark.parametrize(
    "heldout_family",
    ("plain", "battery", "cooldown", "heat", "momentum"),
)
def test_selection_spec_builder_supports_lofo_folds_that_train_on_combo(
    heldout_family: str,
) -> None:
    protocol_path = ROOT / "configs" / "milestone6" / "development_protocol.json"
    screening_path = ROOT / "configs" / "milestone6" / "phase2_screening_candidates.json"
    task_manifest_path = ROOT / "configs" / "milestone6" / "development_tasks.json"
    authority = load_selection_authority(
        protocol_path,
        screening_path,
        task_manifest_path,
    )
    entries = phase2._manifest_tasks()
    training_tasks = tuple(
        phase2._training_identity(entry)
        for entry in entries
        if entry["family"] != heldout_family and "training_core" in entry["roles"]
    )
    heldout_tasks = tuple(
        phase2._heldout_identity(entry)
        for entry in entries
        if entry["family"] == heldout_family and "training_core" in entry["roles"]
    )
    assert len(training_tasks) == 40
    assert len(heldout_tasks) == 8
    assert any(
        task.family_id == "combo" and len(task.trajectory_catalog) == 2
        for task in training_tasks
    )

    base = build_phase2_shared_smoke_config()
    optimum_exposure = phase2._optimum_exposure(training_tasks)
    conditions = tuple(
        condition.model_copy(
            update={
                "exposure": condition.exposure.model_copy(
                    update={
                        "train_task_ids": tuple(task.task_id for task in training_tasks),
                        "exposed_trajectories": optimum_exposure,
                    }
                )
            }
        )
        if condition.exposure.train_task_ids
        else condition
        for condition in base.conditions
    )
    family_offset = authority.family_ids.index(heldout_family) * 10_000
    parameters = {
        **base.parameters,
        "heldout_family": heldout_family,
        "heldout_family_id": heldout_family,
        "fold_id": f"lofo-{heldout_family}",
        "adaptation_action_cap": authority.endpoint,
        "selection_metric_failure_sentinel": authority.failure_sentinel,
    }
    config = base.model_copy(
        update={
            "parameters": parameters,
            "conditions": conditions,
            "replicates": len(authority.screening_replicates),
            "seed_policy": base.seed_policy.model_copy(
                update={
                    "model_seed_base": 6_100_000 + family_offset,
                    "probe_seed_base": 6_200_000 + family_offset,
                    "search_seed_base": 6_300_000 + family_offset,
                    "data_order_seed_base": 6_400_000 + family_offset,
                }
            ),
            "split": base.split.model_copy(
                update={
                    "development_tasks": training_tasks,
                    "validation_tasks": heldout_tasks,
                }
            ),
        }
    )
    expected = plan_expected_units(config)
    learned_units = tuple(
        unit for unit in expected.units if unit.key.condition_id == B1
    )
    expected_shared = ExpectedSharedArtifacts(
        run_id=run_id_for(config),
        config_sha256=scientific_config_sha256(config),
        artifacts=tuple(
            PlannedSharedArtifact(
                kind=kind,
                key_id=f"{replicate * 3 + kind_index + 1:064x}",
                owner_condition_id=B1,
                owner_group_id=(
                    "canonical-evidence"
                    if kind == "training_data_evidence"
                    else B1
                ),
                owner_family_id=heldout_family,
                owner_fold_id=f"lofo-{heldout_family}",
                owner_replicate=replicate,
                consumer_phase="validation",
                consumer_condition_ids=(B1,),
                consumer_unit_ids=tuple(
                    unit.unit_id
                    for unit in learned_units
                    if unit.key.replicate == replicate
                ),
            )
            for replicate in authority.screening_replicates
            for kind_index, kind in enumerate(
                (
                    "training_data_evidence",
                    "training_data_view",
                    "training_artifact",
                )
            )
        ),
    )

    spec = build_selection_metric_spec(
        config,
        expected,
        expected_shared,
        authority,
        condition_id=B1,
    )
    assert spec.family_ids == frozenset({heldout_family})
    assert len(spec.expected_units) == 40

    final_consumer_shared = expected_shared.model_copy(
        update={
            "artifacts": (
                expected_shared.artifacts[0].model_copy(
                    update={"consumer_phase": "final"}
                ),
                *expected_shared.artifacts[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="forbidden final-family"):
        build_selection_metric_spec(
            config,
            expected,
            final_consumer_shared,
            authority,
            condition_id=B1,
        )
