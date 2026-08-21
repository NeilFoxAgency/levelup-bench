from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from levelup.experiments.milestone6_phase2_screening import (
    B1,
    B2,
    C,
    ScreeningArtifactSlot,
    ScreeningPlan,
    base_condition_id,
    build_screening_artifact_slots,
    build_screening_child_config,
    build_screening_plan,
    candidate_for_condition,
    candidate_tuples,
    load_screening_plan,
    main,
    screening_child_configs,
    screening_condition_id,
    selection_authority,
    validate_screening_child_config,
    validate_screening_plan,
    validate_screening_plan_payload,
)
from levelup.experiments.runner.config import (
    canonical_json_bytes,
    run_id_for,
    scientific_config_sha256,
)
from levelup.experiments.runner.storage import expected_units_sha256, plan_expected_units


def test_screening_plan_freezes_six_children_and_exact_global_counts() -> None:
    authority = selection_authority()
    plan = build_screening_plan()

    assert plan.family_order == authority.family_ids
    assert plan.replicates == (0, 1, 2, 3, 4)
    assert len(plan.children) == 6
    assert plan.expected_total_units == 9_120
    assert plan.expected_total_evidence_artifacts == 30
    assert plan.expected_total_training_data_views == 90
    assert plan.expected_total_model_artifacts == 360
    assert plan.final_family_access is False
    assert plan == build_screening_plan()
    validate_screening_plan(plan)


@pytest.mark.parametrize(
    "heldout_family",
    ("plain", "battery", "cooldown", "heat", "momentum", "combo"),
)
def test_screening_child_config_has_exact_lofo_matrix(heldout_family: str) -> None:
    authority = selection_authority()
    config = build_screening_child_config(heldout_family)
    expected = plan_expected_units(config)

    assert config.split.final_tasks == ()
    assert len(config.split.development_tasks) == 40
    assert len(config.split.validation_tasks) == 8
    assert {task.family_id for task in config.split.development_tasks} == (
        set(authority.family_ids) - {heldout_family}
    )
    assert {task.family_id for task in config.split.validation_tasks} == {
        heldout_family
    }
    assert all(len(task.trajectory_catalog) == 2 for task in config.split.development_tasks)
    assert all(not task.trajectory_catalog for task in config.split.validation_tasks)
    assert len(config.conditions) == 38
    assert config.replicates == 5
    family_offset = authority.family_ids.index(heldout_family) * 10_000
    assert config.seed_policy.model_seed_base == 6_100_000 + family_offset
    assert config.seed_policy.probe_seed_base == 6_200_000 + family_offset
    assert config.seed_policy.search_seed_base == 6_300_000 + family_offset
    assert config.seed_policy.data_order_seed_base == 6_400_000 + family_offset
    assert config.seed_policy.replicate_stride == 100_000
    assert len(expected.units) == 1_520
    assert {unit.key.phase for unit in expected.units} == {"validation"}
    assert {unit.key.family_id for unit in expected.units} == {heldout_family}
    assert Counter(unit.key.replicate for unit in expected.units) == {
        replicate: 304 for replicate in range(5)
    }
    assert all(condition.execution_phases == ("validation",) for condition in config.conditions)
    assert config.parameters["adaptation_action_cap"] == 2_048
    assert config.parameters["selection_metric_failure_sentinel"] == 2_049
    assert config.parameters["final_family_access"] is False
    assert config.parameters["unit_local_training_repeated_and_counted"] is False
    assert config.parameters["optimizer"] == "adam"
    assert config.parameters["weight_decay"] == 0.0001
    assert config.parameters["mlp_hidden_widths"] == [48, 24]
    assert config.parameters["probe_actions_per_attempt"] == 16
    assert config.parameters["processes"] == 1
    assert config.parameters["unknown_affordance_policy"] == (
        "zero feature with eligible uniform fallback"
    )
    assert config.parameters["evaluator_feedback_to_policy"] == "none"
    assert config.parameters["capacity_matching"][
        "cross_representation_parameter_tolerance_fraction"
    ] == 0.1
    assert config.parameters["model_artifact_identity_excludes"] == [
        "search_temperature"
    ]
    validate_screening_child_config(config)


def test_screening_condition_grid_is_exact_and_temperature_reuses_training_tuple() -> None:
    rows = candidate_tuples()
    assert len(rows) == 12
    configs = screening_child_configs()
    assert len(configs) == 6

    condition_ids = tuple(condition.condition_id for condition in configs[0].conditions)
    assert condition_ids[:2] == (
        "A0-no-probe-uniform",
        "A1-paid-probe-uniform",
    )
    learned = condition_ids[2:]
    assert len(learned) == 36
    assert Counter(base_condition_id(condition_id) for condition_id in learned) == {
        B1: 12,
        B2: 12,
        C: 12,
    }
    for base in (B1, B2, C):
        variants = [
            candidate_for_condition(condition_id)
            for condition_id in learned
            if base_condition_id(condition_id) == base
        ]
        assert Counter(row["training_tuple_id"] for row in variants if row is not None) == {
            "lr0p003-e120": 3,
            "lr0p003-e180": 3,
            "lr0p01-e120": 3,
            "lr0p01-e180": 3,
        }
        assert {
            row["search_temperature"] for row in variants if row is not None
        } == {0.6, 0.9, 1.2}
        for condition_id in (
            item for item in learned if base_condition_id(item) == base
        ):
            condition = next(
                item
                for item in configs[0].conditions
                if item.condition_id == condition_id
            )
            row = candidate_for_condition(condition_id)
            assert row is not None
            assert condition.parameters["candidate_tuple_id"] == row["tuple_id"]
            assert condition.parameters["training_tuple_id"] == row["training_tuple_id"]
            assert condition.parameters["learning_rate"] == row["learning_rate"]
            assert condition.parameters["training_epochs"] == row["training_epochs"]
            assert condition.parameters["search_temperature"] == row["search_temperature"]


def test_screening_condition_identity_rejects_unknown_or_ambiguous_values() -> None:
    row = candidate_tuples()[0]
    condition_id = screening_condition_id(B1, row["tuple_id"])
    assert base_condition_id(condition_id) == B1
    assert candidate_for_condition(condition_id) == row
    assert base_condition_id("A0-no-probe-uniform") is None
    assert candidate_for_condition("A0-no-probe-uniform") is None

    with pytest.raises(ValueError):
        screening_condition_id("unknown", row["tuple_id"])
    with pytest.raises(ValueError):
        screening_condition_id(B1, "unknown")
    with pytest.raises(ValueError):
        candidate_for_condition(f"{B1}--unknown")


def test_screening_child_validation_rejects_hash_budget_seed_split_and_final_drift() -> None:
    config = build_screening_child_config("plain")
    final_task = config.split.validation_tasks[0].model_copy(
        update={"task_id": f"{config.split.validation_tasks[0].task_id}.final"}
    )
    tampered = (
        config.model_copy(
            update={
                "parameters": {
                    **config.parameters,
                    "development_protocol_sha256": "0" * 64,
                }
            }
        ),
        config.model_copy(
            update={
                "parameters": {
                    **config.parameters,
                    "adaptation_action_cap": 1_024,
                    "selection_metric_failure_sentinel": 1_025,
                }
            }
        ),
        config.model_copy(
            update={
                "seed_policy": config.seed_policy.model_copy(
                    update={"model_seed_base": config.seed_policy.model_seed_base + 1}
                )
            }
        ),
        config.model_copy(
            update={
                "split": config.split.model_copy(
                    update={"final_tasks": (final_task,)}
                )
            }
        ),
    )
    for changed in tampered:
        with pytest.raises(ValueError):
            validate_screening_child_config(changed)


def test_screening_plan_child_identities_match_canonical_configs() -> None:
    plan = build_screening_plan()
    configs = screening_child_configs()

    assert tuple(child.run_id for child in plan.children) == tuple(
        run_id_for(config) for config in configs
    )
    assert tuple(child.config_sha256 for child in plan.children) == tuple(
        scientific_config_sha256(config) for config in configs
    )


def test_screening_artifact_slots_derive_exact_reuse_and_consumer_counts() -> None:
    config = build_screening_child_config("plain")
    slots = build_screening_artifact_slots(config)

    assert Counter(slot.kind for slot in slots) == {
        "training_data_evidence": 5,
        "training_data_view": 15,
        "training_artifact": 60,
    }
    assert len({slot.lineage_slot_id for slot in slots}) == 80
    assert {slot.run_id for slot in slots} == {run_id_for(config)}
    assert {slot.config_sha256 for slot in slots} == {
        scientific_config_sha256(config)
    }
    assert {slot.expected_units_sha256 for slot in slots} == {
        expected_units_sha256(plan_expected_units(config))
    }
    assert all(slot.owner_condition_id in slot.consumer_condition_ids for slot in slots)
    assert all(
        len(slot.consumer_unit_ids)
        == {
            "training_data_evidence": 288,
            "training_data_view": 96,
            "training_artifact": 24,
        }[slot.kind]
        for slot in slots
    )
    for replicate in range(5):
        evidence = [
            slot
            for slot in slots
            if slot.kind == "training_data_evidence" and slot.replicate == replicate
        ]
        assert len(evidence) == 1
        assert len(evidence[0].consumer_condition_ids) == 36
        assert evidence[0].owner_group_id == "canonical-evidence"
        models = [
            slot
            for slot in slots
            if slot.kind == "training_artifact" and slot.replicate == replicate
        ]
        assert len(models) == 12
        assert {(slot.base_condition_id, slot.training_tuple_id) for slot in models} == {
            (base, training_tuple_id)
            for base in (B1, B2, C)
            for training_tuple_id in (
                "lr0p003-e120",
                "lr0p003-e180",
                "lr0p01-e120",
                "lr0p01-e180",
            )
        }
        assert all(len(slot.consumer_condition_ids) == 3 for slot in models)
        assert all(
            len(
                {
                    candidate_for_condition(condition_id)["search_temperature"]
                    for condition_id in slot.consumer_condition_ids
                }
            )
            == 3
            for slot in models
        )


def test_screening_artifact_validation_rejects_rehashed_cross_tuple_consumers() -> None:
    import levelup.experiments.milestone6_phase2_screening as screening_module

    config = build_screening_child_config("plain")
    slots = list(build_screening_artifact_slots(config))
    model_indexes = [
        index
        for index, slot in enumerate(slots)
        if slot.kind == "training_artifact"
        and slot.replicate == 0
        and slot.base_condition_id == B1
    ]
    left_index, right_index = model_indexes[:2]
    left = slots[left_index]
    right = slots[right_index]

    def with_consumers_from(
        source: ScreeningArtifactSlot,
        replacement: ScreeningArtifactSlot,
    ) -> ScreeningArtifactSlot:
        payload = source.model_dump(mode="json")
        payload["consumer_condition_ids"] = replacement.consumer_condition_ids
        payload["consumer_unit_ids"] = replacement.consumer_unit_ids
        payload["owner_condition_id"] = replacement.owner_condition_id
        body = {
            key: value for key, value in payload.items() if key != "lineage_slot_id"
        }
        payload["lineage_slot_id"] = hashlib.sha256(
            canonical_json_bytes(body)
        ).hexdigest()
        return screening_module.ScreeningArtifactSlot.model_validate(payload)

    slots[left_index] = with_consumers_from(left, right)
    slots[right_index] = with_consumers_from(right, left)
    with pytest.raises(ValueError, match="lineage or exact consumers drifted"):
        screening_module._validate_screening_artifact_slots(config, tuple(slots))


def _rehash_plan(payload: dict[str, object]) -> dict[str, object]:
    body = {key: value for key, value in payload.items() if key != "plan_id"}
    payload["plan_id"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return payload


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("protocol_sha256", "0" * 64),
        ("screening_candidates_sha256", "1" * 64),
        ("task_manifest_sha256", "2" * 64),
        ("family_order", ("combo", "momentum", "heat", "cooldown", "battery", "plain")),
        ("replicates", (0, 1, 2, 3, 9)),
        ("candidate_tuple_ids", ("forged",)),
    ),
)
def test_authoritative_plan_validation_rejects_rehashed_parent_drift(
    field: str,
    replacement: object,
) -> None:
    payload = build_screening_plan().model_dump(mode="json")
    payload[field] = replacement
    if field == "family_order":
        payload["children"] = list(reversed(payload["children"]))
    tampered = _rehash_plan(payload)

    with pytest.raises(ValueError, match="frozen canonical plan"):
        validate_screening_plan_payload(tampered)


def test_authoritative_plan_validation_rejects_rehashed_child_drift() -> None:
    payload = build_screening_plan().model_dump(mode="json")
    children = list(payload["children"])
    children[0] = {**children[0], "config_sha256": "f" * 64}
    payload["children"] = children
    tampered = _rehash_plan(payload)

    with pytest.raises(ValueError, match="frozen canonical plan"):
        validate_screening_plan_payload(tampered)


def test_screening_plan_type_rejects_direct_untrusted_construction() -> None:
    payload = build_screening_plan().model_dump(mode="json")
    with pytest.raises(ValueError, match="canonical authority construction"):
        ScreeningPlan.model_validate(payload)
    bypassed = ScreeningPlan.model_construct(**payload)
    with pytest.raises(ValueError, match="lacks canonical authority authorization"):
        validate_screening_plan(bypassed)


def test_persisted_plan_loader_requires_current_frozen_authority(
    tmp_path: Path,
) -> None:
    plan = build_screening_plan()
    path = tmp_path / "screening-plan.json"
    path.write_text(plan.model_dump_json(), encoding="utf-8")
    assert load_screening_plan(path) == plan

    payload = plan.model_dump(mode="json")
    children = list(payload["children"])
    children[0] = {**children[0], "artifact_slots_sha256": "e" * 64}
    payload["children"] = children
    tampered = _rehash_plan(payload)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen canonical plan"):
        load_screening_plan(path)


def test_authority_snapshot_rejects_a_change_between_validation_and_source_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import levelup.experiments.milestone6_phase2_screening as screening_module

    authority = selection_authority()
    original_read_bytes = Path.read_bytes

    def changed_protocol(path: Path) -> bytes:
        content = original_read_bytes(path)
        if path == authority.protocol_path:
            return content + b"\n"
        return content

    monkeypatch.setattr(screening_module, "selection_authority", lambda: authority)
    monkeypatch.setattr(Path, "read_bytes", changed_protocol)
    with pytest.raises(ValueError, match="changed while loading"):
        screening_module.candidate_tuples()


def test_screening_cli_remains_plan_only(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 2

    assert main(["--plan-only"]) == 0
    output = capsys.readouterr().out
    assert '"final_family_access": false' in output

    with pytest.raises(SystemExit) as exc_info:
        main(["--execute"])
    assert exc_info.value.code == 2
