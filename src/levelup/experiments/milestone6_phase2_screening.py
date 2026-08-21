"""Frozen development screening config and lineage plan; execution stays locked.

Concrete artifact-key construction, shared-plan conversion, preparation, execution, and
selection are separate implementation gates and are intentionally unavailable here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    model_validator,
)

from levelup.experiments.milestone6_phase2 import (
    DEVELOPMENT_PROTOCOL_PATH,
    DEVELOPMENT_TASKS_PATH,
    _exposure,
    _heldout_identity,
    _optimum_exposure,
    _training_identity,
)
from levelup.experiments.milestone6_phase2_shared_smoke import (
    B1,
    B2,
    SCREENING_CANDIDATES_PATH,
    C,
)
from levelup.experiments.runner.config import (
    ConditionSpec,
    DevicePolicy,
    ExperimentConfig,
    MetricSpec,
    SeedPolicy,
    SelectionSpec,
    SplitSpec,
    canonical_json_bytes,
    run_id_for,
    scientific_config_sha256,
)
from levelup.experiments.runner.selection_metric import (
    ACTION_FORMULA,
    METRIC_ID,
    METRIC_SCHEMA_VERSION,
    ORACLE_POLICY,
    SelectionAuthority,
    load_selection_authority,
)
from levelup.experiments.runner.storage import expected_units_sha256, plan_expected_units

SCHEMA_VERSION = "milestone6.phase2.screening-plan.v1"
METHOD_REVISION = "development-screening-boundary-v1"
FIXED_CONDITIONS = ("A0-no-probe-uniform", "A1-paid-probe-uniform")
LEARNED_BASES = (B1, B2, C)
_PLAN_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _AuthoritySnapshot:
    """One exact validated source snapshot used throughout a build."""

    authority: SelectionAuthority
    protocol: dict[str, Any]
    screening: dict[str, Any]
    task_manifest: dict[str, Any]


def _authority_snapshot() -> _AuthoritySnapshot:
    authority = selection_authority()
    source_rows = (
        (authority.protocol_path, authority.protocol_sha256, "protocol"),
        (
            authority.screening_candidates_path,
            authority.screening_candidates_sha256,
            "screening candidates",
        ),
        (authority.task_manifest_path, authority.task_manifest_sha256, "task manifest"),
    )
    payloads: dict[str, dict[str, Any]] = {}
    for path, expected_sha256, label in source_rows:
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError(f"{label} changed while loading screening authority")
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError(f"{label} must be a JSON object")
        payloads[label] = payload
    return _AuthoritySnapshot(
        authority=authority,
        protocol=payloads["protocol"],
        screening=payloads["screening candidates"],
        task_manifest=payloads["task manifest"],
    )


def selection_authority() -> SelectionAuthority:
    """Load the exact code-pinned development authority."""

    return load_selection_authority(
        DEVELOPMENT_PROTOCOL_PATH,
        SCREENING_CANDIDATES_PATH,
        DEVELOPMENT_TASKS_PATH,
    )


def _candidate_tuples(snapshot: _AuthoritySnapshot) -> tuple[dict[str, Any], ...]:
    rows = snapshot.screening["candidate_tuples"]
    return tuple(dict(row) for row in rows)


def candidate_tuples() -> tuple[dict[str, Any], ...]:
    """Return the reviewed numeric grid in its frozen ascending order."""

    return _candidate_tuples(_authority_snapshot())


def screening_condition_id(base_condition_id: str, tuple_id: str) -> str:
    if base_condition_id not in LEARNED_BASES:
        raise ValueError("screening variant requires a frozen learned base")
    if tuple_id not in {row["tuple_id"] for row in candidate_tuples()}:
        raise ValueError("screening variant requires a frozen candidate tuple")
    return f"{base_condition_id}--{tuple_id}"


def base_condition_id(condition_id: str) -> str | None:
    """Resolve an exact learned variant without substring ambiguity."""

    matches = [
        base
        for base in LEARNED_BASES
        if condition_id.startswith(f"{base}--")
    ]
    if len(matches) > 1:
        raise ValueError("ambiguous screening condition identity")
    return matches[0] if matches else None


def candidate_for_condition(condition_id: str) -> dict[str, Any] | None:
    base = base_condition_id(condition_id)
    if base is None:
        return None
    tuple_id = condition_id.removeprefix(f"{base}--")
    matches = [row for row in candidate_tuples() if row["tuple_id"] == tuple_id]
    if len(matches) != 1:
        raise ValueError("screening condition has an unknown numeric tuple")
    return matches[0]


def _screening_child_config(
    heldout_family: str,
    snapshot: _AuthoritySnapshot,
) -> ExperimentConfig:
    authority = snapshot.authority
    if heldout_family not in authority.family_ids:
        raise ValueError("held-out family is outside the frozen development universe")
    protocol = snapshot.protocol
    manifest_entries = tuple(snapshot.task_manifest["tasks"])
    training_tasks = tuple(
        _training_identity(entry)
        for entry in manifest_entries
        if entry["family"] != heldout_family and "training_core" in entry["roles"]
    )
    heldout_tasks = tuple(
        _heldout_identity(entry)
        for entry in manifest_entries
        if entry["family"] == heldout_family and "training_core" in entry["roles"]
    )
    if len(training_tasks) != 40 or len(heldout_tasks) != 8:
        raise RuntimeError("screening child does not contain an exact 40/8 LOFO split")

    no_reference = _exposure(training_tasks=(), exposed=(), probe_access=False)
    paid_reference = _exposure(training_tasks=(), exposed=(), probe_access=True)
    optimum_exposure = _optimum_exposure(training_tasks)
    optimum_only = _exposure(
        training_tasks=training_tasks,
        exposed=optimum_exposure,
        probe_access=True,
    )
    no_probe = ConditionSpec(
        condition_id=FIXED_CONDITIONS[0],
        learner_id="uniform-visible-actions-v1",
        execution_phases=("validation",),
        exposure=no_reference,
        parameters={"probe_action_cap": 0},
    )
    paid_probe = ConditionSpec(
        condition_id=FIXED_CONDITIONS[1],
        learner_id="uniform-visible-actions-v1",
        execution_phases=("validation",),
        exposure=paid_reference,
        parameters={"probe_action_cap": 64},
    )
    sources = {
        B1: ConditionSpec(
            condition_id=B1,
            learner_id="global-affordance-mlp-frequency-v1",
            execution_phases=("validation",),
            exposure=optimum_only,
            parameters={"objective": "optimum_frequency"},
        ),
        B2: ConditionSpec(
            condition_id=B2,
            learner_id="global-affordance-mlp-listwise-v1",
            execution_phases=("validation",),
            exposure=optimum_only,
            parameters={"objective": "listwise_optimum"},
        ),
        C: ConditionSpec(
            condition_id=C,
            learner_id="state-affordance-mlp-listwise-v1",
            execution_phases=("validation",),
            exposure=optimum_only,
            parameters={"objective": "listwise_optimum"},
        ),
    }
    rows = _candidate_tuples(snapshot)
    learned_conditions: list[ConditionSpec] = []
    for base_id in LEARNED_BASES:
        source = sources[base_id]
        for row in rows:
            learned_conditions.append(
                source.model_copy(
                    update={
                        "condition_id": f"{base_id}--{row['tuple_id']}",
                        "parameters": {
                            **source.parameters,
                            "base_condition_id": base_id,
                            "candidate_tuple_id": row["tuple_id"],
                            "training_tuple_id": row["training_tuple_id"],
                            "learning_rate": row["learning_rate"],
                            "training_epochs": row["training_epochs"],
                            "search_temperature": row["search_temperature"],
                        },
                    }
                )
            )

    screening = protocol["budgets"]["screening"]
    family_offset = authority.family_ids.index(heldout_family) * int(
        protocol["seed_policy"]["family_offset_stride"]
    )
    bases = protocol["seed_policy"]["bases"]
    parameters = {
        "development_only": True,
        "comparative_development_screening": True,
        "final_family_access": False,
        "fold_id": f"lofo-{heldout_family}",
        "heldout_family": heldout_family,
        "heldout_family_id": heldout_family,
        "probe_action_cap": int(screening["probe_actions_per_task"]),
        "probe_coverage_target_samples_per_alias": int(
            screening["probe_coverage_target_samples_per_alias"]
        ),
        "candidate_episodes": int(screening["candidate_episodes_per_task"]),
        "adaptation_action_cap": int(screening["adaptation_actions_per_task"]),
        "maximum_actions_per_candidate_episode": int(
            protocol["budgets"]["maximum_actions_per_candidate_episode"]
        ),
        "optimizer": str(protocol["fixed_defaults"]["optimizer"]),
        "weight_decay": float(protocol["fixed_defaults"]["weight_decay"]),
        "mlp_hidden_widths": list(protocol["fixed_defaults"]["mlp_hidden_widths"]),
        "probe_actions_per_attempt": int(
            protocol["fixed_defaults"]["probe_actions_per_attempt"]
        ),
        "processes": int(protocol["fixed_defaults"]["processes"]),
        "unknown_affordance_policy": protocol["fixed_defaults"][
            "unknown_affordance_policy"
        ],
        "evaluator_feedback_to_policy": protocol["fixed_defaults"][
            "evaluator_feedback_to_policy"
        ],
        "capacity_matching": dict(snapshot.screening["capacity_matching"]),
        "model_artifact_identity_excludes": ["search_temperature"],
        "data_order": "canonical_manifest_order_no_shuffle",
        "shared_artifact_training": True,
        "unit_local_training_repeated_and_counted": False,
        "development_protocol_sha256": authority.protocol_sha256,
        "screening_candidates_sha256": authority.screening_candidates_sha256,
        "development_task_manifest_sha256": authority.task_manifest_sha256,
        "selection_metric_id": METRIC_ID,
        "selection_metric_schema_version": METRIC_SCHEMA_VERSION,
        "selection_metric_action_formula": ACTION_FORMULA,
        "selection_metric_oracle_policy": ORACLE_POLICY,
        "selection_metric_phase": "validation",
        "selection_metric_failure_sentinel": int(screening["adaptation_actions_per_task"])
        + 1,
        "candidate_tuple_ids": [row["tuple_id"] for row in rows],
    }
    metrics = (
        MetricSpec(
            metric_id="exact_optimum_success",
            direction="maximize",
            unit="boolean",
            description="Post-hoc exact-optimum success after fixed-budget search.",
        ),
        MetricSpec(
            metric_id="adaptation_actions_to_optimum",
            direction="minimize",
            unit="actions",
            description="Probe plus search actions to first post-hoc exact optimum.",
        ),
        MetricSpec(
            metric_id="performance_value",
            direction="minimize",
            unit="ticks",
            description="Best independently replayed valid performance.",
        ),
        MetricSpec(
            metric_id=METRIC_ID,
            direction="minimize",
            unit="actions",
            description=(
                "Post-hoc paid probe plus candidate-generation actions to first exact optimum."
            ),
        ),
        MetricSpec(
            metric_id="minimum_family_exact_optimum_success_rate",
            direction="maximize",
            unit="fraction",
            description="Parent-only minimum exact-optimum success rate across families.",
        ),
    )
    return ExperimentConfig(
        experiment_id=f"milestone6-phase2-screening-lofo-{heldout_family}",
        method_revision=METHOD_REVISION,
        split=SplitSpec(
            development_tasks=training_tasks,
            validation_tasks=heldout_tasks,
            final_tasks=(),
        ),
        conditions=(no_probe, paid_probe, *learned_conditions),
        replicates=len(authority.screening_replicates),
        seed_policy=SeedPolicy(
            derivation_version="phase2.v1",
            model_seed_base=int(bases["model"]) + family_offset,
            probe_seed_base=int(bases["probe"]) + family_offset,
            search_seed_base=int(bases["search"]) + family_offset,
            data_order_seed_base=int(bases["data_order"]) + family_offset,
            replicate_stride=int(protocol["seed_policy"]["replicate_stride"]),
        ),
        device_policy=DevicePolicy(requested_device="cpu", torch_threads=1),
        metrics=metrics,
        selection=SelectionSpec(
            phases=("validation",),
            primary_metric="minimum_family_exact_optimum_success_rate",
            rule=(
                "Parent-only frozen Phase 2 advancement rule; no child-local ranking or "
                "cross-condition elimination."
            ),
        ),
        diagnostic_fields=(
            "development_screening",
            "first_optimum_adaptation_actions",
            "unknown_affordance_decisions",
            "trainable_parameters",
            "training_examples",
            "oracle_setup_calls",
            "shared_training_artifact",
            "search_temperature",
        ),
        parameters=parameters,
    )


def validate_screening_child_config(config: ExperimentConfig) -> None:
    """Require byte-equivalent scientific content to one frozen child config."""

    snapshot = _authority_snapshot()
    authority = snapshot.authority
    heldout_family = config.parameters.get("heldout_family_id")
    if not isinstance(heldout_family, str) or heldout_family not in authority.family_ids:
        raise ValueError("screening child has an invalid held-out family")
    canonical = _screening_child_config(heldout_family, snapshot)
    if config != canonical:
        raise ValueError("screening child config drifted from the frozen canonical config")
    expected = plan_expected_units(config)
    if len(config.conditions) != 38 or len(expected.units) != 1_520:
        raise ValueError("screening child matrix size drifted")
    if config.split.final_tasks or any(
        "final" in condition.execution_phases for condition in config.conditions
    ):
        raise ValueError("screening child contains forbidden final-family material")


def build_screening_child_config(heldout_family: str) -> ExperimentConfig:
    snapshot = _authority_snapshot()
    config = _screening_child_config(heldout_family, snapshot)
    _validate_screening_child_matrix(config)
    return config


def screening_child_configs() -> tuple[ExperimentConfig, ...]:
    snapshot = _authority_snapshot()
    configs = tuple(
        _screening_child_config(family, snapshot)
        for family in snapshot.authority.family_ids
    )
    for config in configs:
        _validate_screening_child_matrix(config)
    return configs


def _validate_screening_child_matrix(config: ExperimentConfig) -> None:
    expected = plan_expected_units(config)
    if len(config.conditions) != 38 or len(expected.units) != 1_520:
        raise ValueError("screening child matrix size drifted")
    if config.split.final_tasks or any(
        "final" in condition.execution_phases for condition in config.conditions
    ):
        raise ValueError("screening child contains forbidden final-family material")


ArtifactKind = Literal[
    "training_data_evidence",
    "training_data_view",
    "training_artifact",
]


class ScreeningArtifactSlot(BaseModel):
    """One logical reuse slot; concrete artifact keys are a later preparation gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ArtifactKind
    lineage_slot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_units_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_id: str = Field(min_length=1)
    heldout_family: str = Field(min_length=1)
    replicate: int = Field(ge=0)
    base_condition_id: str | None = None
    training_tuple_id: str | None = None
    owner_condition_id: str = Field(min_length=1)
    owner_group_id: str = Field(min_length=1)
    consumer_condition_ids: tuple[str, ...]
    consumer_unit_ids: tuple[str, ...]

    @model_validator(mode="after")
    def identity_and_shape_are_exact(self) -> "ScreeningArtifactSlot":
        body = self.model_dump(mode="json", exclude={"lineage_slot_id"})
        if self.lineage_slot_id != hashlib.sha256(canonical_json_bytes(body)).hexdigest():
            raise ValueError("screening artifact slot identity mismatch")
        if (
            not self.consumer_condition_ids
            or len(self.consumer_condition_ids) != len(set(self.consumer_condition_ids))
            or self.owner_condition_id not in self.consumer_condition_ids
            or not self.consumer_unit_ids
            or len(self.consumer_unit_ids) != len(set(self.consumer_unit_ids))
        ):
            raise ValueError("screening artifact owner and consumers are invalid")
        if self.kind == "training_data_evidence":
            if self.base_condition_id is not None or self.training_tuple_id is not None:
                raise ValueError("evidence slot cannot contain view or model identity")
        elif self.kind == "training_data_view":
            if self.base_condition_id not in LEARNED_BASES or self.training_tuple_id is not None:
                raise ValueError("data-view slot identity is invalid")
        elif (
            self.base_condition_id not in LEARNED_BASES
            or self.training_tuple_id is None
        ):
            raise ValueError("model slot identity is invalid")
        return self


def _artifact_slot(
    *,
    kind: ArtifactKind,
    config: ExperimentConfig,
    run_id: str,
    config_sha256: str,
    unit_plan_sha256: str,
    replicate: int,
    consumer_condition_ids: tuple[str, ...],
    consumer_unit_ids: tuple[str, ...],
    base_id: str | None = None,
    training_tuple_id: str | None = None,
) -> ScreeningArtifactSlot:
    body = {
        "kind": kind,
        "run_id": run_id,
        "config_sha256": config_sha256,
        "expected_units_sha256": unit_plan_sha256,
        "fold_id": str(config.parameters["fold_id"]),
        "heldout_family": str(config.parameters["heldout_family_id"]),
        "replicate": replicate,
        "base_condition_id": base_id,
        "training_tuple_id": training_tuple_id,
        "owner_condition_id": consumer_condition_ids[0],
        "owner_group_id": "canonical-evidence" if base_id is None else base_id,
        "consumer_condition_ids": consumer_condition_ids,
        "consumer_unit_ids": consumer_unit_ids,
    }
    return ScreeningArtifactSlot(
        lineage_slot_id=hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
        **body,
    )


def _screening_artifact_slots(
    config: ExperimentConfig,
    rows: tuple[dict[str, Any], ...],
) -> tuple[ScreeningArtifactSlot, ...]:
    learned_ids = tuple(condition.condition_id for condition in config.conditions[2:])
    training_tuple_ids = tuple(dict.fromkeys(row["training_tuple_id"] for row in rows))
    tuple_by_id = {row["tuple_id"]: row for row in rows}
    child_run_id = run_id_for(config)
    child_config_sha256 = scientific_config_sha256(config)
    expected_plan = plan_expected_units(config)
    child_unit_plan_sha256 = expected_units_sha256(expected_plan)

    def consumer_units(
        replicate: int,
        condition_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(
            unit.unit_id
            for unit in expected_plan.units
            if unit.key.replicate == replicate
            and unit.key.condition_id in condition_ids
        )
    slots: list[ScreeningArtifactSlot] = []
    for replicate in range(config.replicates):
        slots.append(
            _artifact_slot(
                kind="training_data_evidence",
                config=config,
                run_id=child_run_id,
                config_sha256=child_config_sha256,
                unit_plan_sha256=child_unit_plan_sha256,
                replicate=replicate,
                consumer_condition_ids=learned_ids,
                consumer_unit_ids=consumer_units(replicate, learned_ids),
            )
        )
        for base in LEARNED_BASES:
            base_consumers = tuple(
                condition_id
                for condition_id in learned_ids
                if condition_id.startswith(f"{base}--")
            )
            slots.append(
                _artifact_slot(
                    kind="training_data_view",
                    config=config,
                    run_id=child_run_id,
                    config_sha256=child_config_sha256,
                    unit_plan_sha256=child_unit_plan_sha256,
                    replicate=replicate,
                    base_id=base,
                    consumer_condition_ids=base_consumers,
                    consumer_unit_ids=consumer_units(replicate, base_consumers),
                )
            )
            for training_tuple_id in training_tuple_ids:
                model_consumers = tuple(
                    condition_id
                    for condition_id in base_consumers
                    if tuple_by_id[condition_id.removeprefix(f"{base}--")][
                        "training_tuple_id"
                    ]
                    == training_tuple_id
                )
                slots.append(
                    _artifact_slot(
                        kind="training_artifact",
                        config=config,
                        run_id=child_run_id,
                        config_sha256=child_config_sha256,
                        unit_plan_sha256=child_unit_plan_sha256,
                        replicate=replicate,
                        base_id=base,
                        training_tuple_id=training_tuple_id,
                        consumer_condition_ids=model_consumers,
                        consumer_unit_ids=consumer_units(replicate, model_consumers),
                    )
                )
    _validate_screening_artifact_slots(config, tuple(slots))
    return tuple(slots)


def _validate_screening_artifact_slots(
    config: ExperimentConfig,
    slots: tuple[ScreeningArtifactSlot, ...],
) -> None:
    learned_ids = tuple(condition.condition_id for condition in config.conditions[2:])
    training_tuple_ids = tuple(
        dict.fromkeys(
            str(condition.parameters["training_tuple_id"])
            for condition in config.conditions[2:]
        )
    )
    exact_slot_keys = {
        ("training_data_evidence", replicate, None, None)
        for replicate in range(config.replicates)
    } | {
        ("training_data_view", replicate, base, None)
        for replicate in range(config.replicates)
        for base in LEARNED_BASES
    } | {
        ("training_artifact", replicate, base, training_tuple_id)
        for replicate in range(config.replicates)
        for base in LEARNED_BASES
        for training_tuple_id in training_tuple_ids
    }
    actual_slot_keys = {
        (slot.kind, slot.replicate, slot.base_condition_id, slot.training_tuple_id)
        for slot in slots
    }
    if (
        len(slots) != 80
        or len({slot.lineage_slot_id for slot in slots}) != 80
        or actual_slot_keys != exact_slot_keys
    ):
        raise ValueError("screening artifact slot count or identity drifted")
    expected_plan = plan_expected_units(config)
    expected_run_id = run_id_for(config)
    expected_config_sha256 = scientific_config_sha256(config)
    expected_plan_sha256 = expected_units_sha256(expected_plan)
    expected_fold_id = str(config.parameters["fold_id"])
    expected_family = str(config.parameters["heldout_family_id"])
    conditions_by_id = {
        condition.condition_id: condition for condition in config.conditions[2:]
    }
    for slot in slots:
        if slot.kind == "training_data_evidence":
            exact_conditions = learned_ids
            exact_owner_group = "canonical-evidence"
        elif slot.kind == "training_data_view":
            exact_conditions = tuple(
                condition_id
                for condition_id in learned_ids
                if condition_id.startswith(f"{slot.base_condition_id}--")
            )
            exact_owner_group = slot.base_condition_id
        else:
            exact_conditions = tuple(
                condition_id
                for condition_id in learned_ids
                if condition_id.startswith(f"{slot.base_condition_id}--")
                and conditions_by_id[condition_id].parameters["training_tuple_id"]
                == slot.training_tuple_id
            )
            exact_owner_group = slot.base_condition_id
        exact_units = tuple(
            unit.unit_id
            for unit in expected_plan.units
            if unit.key.replicate == slot.replicate
            and unit.key.condition_id in exact_conditions
        )
        if (
            slot.run_id != expected_run_id
            or slot.config_sha256 != expected_config_sha256
            or slot.expected_units_sha256 != expected_plan_sha256
            or slot.fold_id != expected_fold_id
            or slot.heldout_family != expected_family
            or slot.consumer_condition_ids != exact_conditions
            or slot.consumer_unit_ids != exact_units
            or slot.owner_condition_id != exact_conditions[0]
            or slot.owner_group_id != exact_owner_group
        ):
            raise ValueError("screening artifact lineage or exact consumers drifted")
    expected_counts = {
        "training_data_evidence": 5,
        "training_data_view": 15,
        "training_artifact": 60,
    }
    for kind, expected_count in expected_counts.items():
        kind_slots = tuple(slot for slot in slots if slot.kind == kind)
        if len(kind_slots) != expected_count:
            raise ValueError("screening artifact kind count drifted")
        for replicate in range(config.replicates):
            consumers = [
                condition_id
                for slot in kind_slots
                if slot.replicate == replicate
                for condition_id in slot.consumer_condition_ids
            ]
            if sorted(consumers) != sorted(learned_ids):
                raise ValueError("screening artifact consumers do not exactly cover learners")
    if any(
        len(slot.consumer_condition_ids) != 3
        for slot in slots
        if slot.kind == "training_artifact"
    ):
        raise ValueError("each screening model must serve exactly three temperatures")
    expected_unit_counts = {
        "training_data_evidence": 288,
        "training_data_view": 96,
        "training_artifact": 24,
    }
    if any(
        len(slot.consumer_unit_ids) != expected_unit_counts[slot.kind]
        for slot in slots
    ):
        raise ValueError("screening artifact atomic-unit coverage drifted")


def build_screening_artifact_slots(
    config: ExperimentConfig,
) -> tuple[ScreeningArtifactSlot, ...]:
    """Build the exact per-fold logical reuse plan; no artifacts are materialized."""

    validate_screening_child_config(config)
    snapshot = _authority_snapshot()
    return _screening_artifact_slots(config, _candidate_tuples(snapshot))


class ScreeningChildPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    heldout_family: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_units_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_slots_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_units: Literal[1520] = 1520
    expected_evidence_artifacts: Literal[5] = 5
    expected_training_data_views: Literal[15] = 15
    expected_model_artifacts: Literal[60] = 60


class ScreeningPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    _authorization: object | None = PrivateAttr(default=None)

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    plan_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    screening_candidates_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    family_order: tuple[str, ...]
    replicates: tuple[int, ...]
    candidate_tuple_ids: tuple[str, ...]
    children: tuple[ScreeningChildPlan, ...]
    expected_total_units: Literal[9120] = 9120
    expected_total_evidence_artifacts: Literal[30] = 30
    expected_total_training_data_views: Literal[90] = 90
    expected_total_model_artifacts: Literal[360] = 360
    final_family_access: Literal[False] = False

    @model_validator(mode="before")
    @classmethod
    def construction_is_authorized(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> Any:
        if info.context is None or info.context.get("construction_token") is not (
            _PLAN_CONSTRUCTION_TOKEN
        ):
            raise ValueError("screening plans require canonical authority construction")
        return value

    @model_validator(mode="after")
    def identity_and_matrix_are_exact(self) -> "ScreeningPlan":
        body = self.model_dump(mode="json", exclude={"plan_id"})
        expected_id = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        if self.plan_id != expected_id:
            raise ValueError("screening parent plan identity mismatch")
        if (
            len(self.children) != 6
            or tuple(child.heldout_family for child in self.children) != self.family_order
            or sum(child.expected_units for child in self.children) != self.expected_total_units
            or sum(child.expected_evidence_artifacts for child in self.children)
            != self.expected_total_evidence_artifacts
            or sum(child.expected_training_data_views for child in self.children)
            != self.expected_total_training_data_views
            or sum(child.expected_model_artifacts for child in self.children)
            != self.expected_total_model_artifacts
        ):
            raise ValueError("screening parent plan matrix drifted")
        return self


def _build_screening_plan(snapshot: _AuthoritySnapshot) -> ScreeningPlan:
    authority = snapshot.authority
    configs = tuple(
        _screening_child_config(family, snapshot) for family in authority.family_ids
    )
    rows = _candidate_tuples(snapshot)
    child_rows: list[ScreeningChildPlan] = []
    for config in configs:
        _validate_screening_child_matrix(config)
        slots = _screening_artifact_slots(config, rows)
        kind_counts = {
            kind: sum(slot.kind == kind for slot in slots)
            for kind in (
                "training_data_evidence",
                "training_data_view",
                "training_artifact",
            )
        }
        child_rows.append(
            ScreeningChildPlan(
                heldout_family=str(config.parameters["heldout_family_id"]),
                run_id=run_id_for(config),
                config_sha256=scientific_config_sha256(config),
                expected_units_sha256=expected_units_sha256(plan_expected_units(config)),
                artifact_slots_sha256=hashlib.sha256(
                    canonical_json_bytes(
                        tuple(slot.model_dump(mode="json") for slot in slots)
                    )
                ).hexdigest(),
                expected_evidence_artifacts=kind_counts["training_data_evidence"],
                expected_training_data_views=kind_counts["training_data_view"],
                expected_model_artifacts=kind_counts["training_artifact"],
            )
        )
    children = tuple(child_rows)
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": authority.protocol_sha256,
        "screening_candidates_sha256": authority.screening_candidates_sha256,
        "task_manifest_sha256": authority.task_manifest_sha256,
        "family_order": authority.family_ids,
        "replicates": authority.screening_replicates,
        "candidate_tuple_ids": tuple(row["tuple_id"] for row in rows),
        "children": tuple(child.model_dump(mode="json") for child in children),
        "expected_total_units": sum(child.expected_units for child in children),
        "expected_total_evidence_artifacts": sum(
            child.expected_evidence_artifacts for child in children
        ),
        "expected_total_training_data_views": sum(
            child.expected_training_data_views for child in children
        ),
        "expected_total_model_artifacts": sum(
            child.expected_model_artifacts for child in children
        ),
        "final_family_access": False,
    }
    plan_id = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    plan = ScreeningPlan.model_validate(
        {"plan_id": plan_id, **body},
        context={"construction_token": _PLAN_CONSTRUCTION_TOKEN},
    )
    plan._authorization = _PLAN_CONSTRUCTION_TOKEN
    return plan


def validate_screening_plan(plan: ScreeningPlan) -> None:
    """Reject even self-consistent plans unless they equal current frozen authority."""

    if plan._authorization is not _PLAN_CONSTRUCTION_TOKEN:
        raise ValueError("screening plan lacks canonical authority authorization")
    snapshot = _authority_snapshot()
    if plan != _build_screening_plan(snapshot):
        raise ValueError("screening plan drifted from the frozen canonical plan")


def validate_screening_plan_payload(payload: dict[str, Any]) -> ScreeningPlan:
    """Authorize an untrusted serialized payload only when it is exactly canonical."""

    snapshot = _authority_snapshot()
    canonical = _build_screening_plan(snapshot)
    if payload != canonical.model_dump(mode="json"):
        raise ValueError("screening plan payload drifted from the frozen canonical plan")
    return canonical


def load_screening_plan(path: Path) -> ScreeningPlan:
    """Load a persisted parent plan through both structural and authority checks."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("screening plan payload must be a JSON object")
    return validate_screening_plan_payload(payload)


def build_screening_plan() -> ScreeningPlan:
    """Build the immutable path-independent parent plan without running any unit."""

    snapshot = _authority_snapshot()
    return _build_screening_plan(snapshot)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the immutable scientific plan; execution is intentionally unavailable.",
    )
    args = parser.parse_args(argv)
    if not args.plan_only:
        parser.error("screening execution is locked; use --plan-only")
    print(json.dumps(build_screening_plan().model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
