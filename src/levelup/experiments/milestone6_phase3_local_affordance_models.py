"""Pure in-memory model preparation for the frozen local-affordance ladder.

Callers must validate ``plan`` against its committed lock before entering this
module.  This boundary rechecks the complete in-memory owner/view/unit binding,
then trains exactly one owner.  It has no filesystem, raw-evidence, search,
replay, evaluator, oracle, or final-family access.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import torch

from levelup.experiments.milestone6_phase3_local_affordance_plan import (
    CONDITIONS,
    FAMILIES,
    FROZEN_SOURCE_SHA256,
    REPLICATES,
    TUPLE_IDS,
    LocalAffordanceModelOwner,
    LocalAffordancePlan,
    LocalAffordanceView,
    _as_json,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.learning.state_conditioned import (
    PROBE_FEATURE_COUNT,
    STATE_CONDITIONED_FEATURE_COUNT,
    DecisionExample,
    GlobalAffordanceScorer,
    GlobalDecisionExample,
    StateConditionedScorer,
    TrainingReport,
    TrainingSpec,
    train_global_listwise_optimum_model,
    train_state_conditioned_optimum_model,
)

CONDITION_B2 = "B2-global-listwise-optimum"
STATE_CONDITIONS = (
    "S-state-availability-listwise-optimum",
    "P-state-availability-alias-pooled-outcome-listwise-optimum",
    "L-state-availability-local-outcome-listwise-optimum",
)
WEIGHT_DECAY = 0.0001
PARAMETERS_BY_CONDITION = {CONDITION_B2: 3601, **dict.fromkeys(STATE_CONDITIONS, 3841)}
ARCHITECTURE_ID_BY_CONDITION = {
    CONDITION_B2: "global-listwise-mlp-49-48-24-1-v1",
    **dict.fromkeys(STATE_CONDITIONS, "state-conditioned-mlp-54-48-24-1-v1"),
}
_TRAINING_TUPLES = {
    "lr0p003-e120": (0.003, 120),
    "lr0p003-e180": (0.003, 180),
    "lr0p01-e120": (0.01, 120),
    "lr0p01-e180": (0.01, 180),
}
_HEX = frozenset("0123456789abcdef")
FROZEN_LOCAL_PLAN_ID = "4e3390056a3a75fad36a033eecf4731dbd08ee946c354449f489c31f5a310718"
_TRAINING_VIEW_TOKEN = object()
_MODEL_PREPARATION_TOKEN = object()


class LocalAffordanceModelPreparationError(ValueError):
    """Raised when a local-affordance model owner or its examples drift."""


@dataclass(frozen=True, slots=True, init=False)
class LocalAffordanceTrainingView:
    """Sealed receipt for the exact examples built from one frozen view."""

    view: LocalAffordanceView
    examples: tuple[DecisionExample, ...] | tuple[GlobalDecisionExample, ...]
    task_example_counts: tuple[int, ...]
    example_count: int
    examples_sha256: str
    _token: object

    def __init__(
        self,
        *,
        view: LocalAffordanceView,
        examples: tuple[DecisionExample, ...] | tuple[GlobalDecisionExample, ...],
        task_example_counts: tuple[int, ...],
        examples_sha256: str,
        _token: object | None = None,
    ) -> None:
        if _token is not _TRAINING_VIEW_TOKEN:
            raise LocalAffordanceModelPreparationError(
                "training-view receipts must come from the canonical view builder"
            )
        object.__setattr__(self, "view", view)
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "task_example_counts", task_example_counts)
        object.__setattr__(self, "example_count", len(examples))
        object.__setattr__(self, "examples_sha256", examples_sha256)
        object.__setattr__(self, "_token", _TRAINING_VIEW_TOKEN)


@dataclass(frozen=True, slots=True, init=False)
class LocalAffordanceModelPreparation:
    """In-memory trained model and exact identity/resource accounting."""

    plan_id: str
    owner: LocalAffordanceModelOwner
    view: LocalAffordanceView
    training_view: LocalAffordanceTrainingView
    model: torch.nn.Module
    report: TrainingReport
    training_spec: TrainingSpec
    training_examples_sha256: str
    model_state_sha256: str
    model_identity_sha256: str
    search_temperature_ids: tuple[str, ...]
    _token: object

    def __init__(
        self,
        *,
        plan_id: str,
        owner: LocalAffordanceModelOwner,
        view: LocalAffordanceView,
        training_view: LocalAffordanceTrainingView,
        model: torch.nn.Module,
        report: TrainingReport,
        training_spec: TrainingSpec,
        training_examples_sha256: str,
        model_state_sha256: str,
        model_identity_sha256: str,
        search_temperature_ids: tuple[str, ...],
        _token: object | None = None,
    ) -> None:
        if _token is not _MODEL_PREPARATION_TOKEN:
            raise LocalAffordanceModelPreparationError(
                "prepared models must come from the canonical in-memory trainer"
            )
        for name, value in locals().copy().items():
            if name in self.__dataclass_fields__ and name != "_token":
                object.__setattr__(self, name, value)
        object.__setattr__(self, "_token", _MODEL_PREPARATION_TOKEN)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        header = canonical_json_bytes(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        )
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _local_plan_body(plan: LocalAffordancePlan) -> dict[str, object]:
    """Rebuild the exact canonical self-hashed body without reading any files."""

    return {
        "schema_version": plan.schema_version,
        "source_sha256": dict(plan.source_sha256),
        "raw_authority_manifest_id": plan.raw_authority_manifest_id,
        "raw_authority_content_sha256": plan.raw_authority_content_sha256,
        "raw_manifest_file_sha256": plan.raw_manifest_file_sha256,
        "family_order": plan.family_order,
        "replicates": plan.replicates,
        "condition_ids": plan.condition_ids,
        "candidate_tuple_ids": plan.candidate_tuple_ids,
        "views": [_as_json(view) for view in plan.views],
        "model_owners": [_as_json(owner) for owner in plan.model_owners],
        "units": [_as_json(unit) for unit in plan.units],
        "final_family_access": False,
    }


def _validate_plan_owner_view(
    plan: LocalAffordancePlan,
    view: LocalAffordanceView,
    owner: LocalAffordanceModelOwner,
) -> None:
    if type(plan) is not LocalAffordancePlan or type(view) is not LocalAffordanceView:
        raise LocalAffordanceModelPreparationError(
            "a validated typed local-affordance plan and view are required"
        )
    if type(owner) is not LocalAffordanceModelOwner:
        raise LocalAffordanceModelPreparationError("a typed frozen model owner is required")
    if (
        not _is_sha256(plan.plan_id)
        or plan.schema_version != "milestone6.phase3.local-affordance-logical-plan.v1"
        or plan.final_family_access
        or plan.family_order != FAMILIES
        or plan.replicates != REPLICATES
        or plan.condition_ids != CONDITIONS
        or plan.candidate_tuple_ids != TUPLE_IDS
        or len(plan.views) != 120
        or len(plan.model_owners) != 480
        or len(plan.units) != 11_520
    ):
        raise LocalAffordanceModelPreparationError(
            "plan is incomplete, noncanonical, or permits final-family access"
        )
    if tuple(plan.source_sha256) != tuple(FROZEN_SOURCE_SHA256.items()):
        raise LocalAffordanceModelPreparationError("plan source identities are not frozen")
    if (
        plan.plan_id != _digest(_local_plan_body(plan))
        or plan.plan_id != FROZEN_LOCAL_PLAN_ID
    ):
        raise LocalAffordanceModelPreparationError(
            "plan self-hash differs from the frozen local-affordance plan identity"
        )

    owner_matches = tuple(item for item in plan.model_owners if item.owner_id == owner.owner_id)
    view_matches = tuple(item for item in plan.views if item.view_id == view.view_id)
    if owner_matches != (owner,) or view_matches != (view,):
        raise LocalAffordanceModelPreparationError("owner or view is not in the validated plan")
    if (
        owner.view_id != view.view_id
        or owner.condition_id != view.condition_id
        or (owner.fold_id, owner.heldout_family, owner.replicate)
        != (view.fold_id, view.heldout_family, view.replicate)
        or owner.heldout_family not in FAMILIES
        or owner.replicate not in REPLICATES
    ):
        raise LocalAffordanceModelPreparationError("owner/view lineage differs")
    if owner.condition_id not in PARAMETERS_BY_CONDITION:
        raise LocalAffordanceModelPreparationError("owner condition is outside the frozen ladder")

    expected_lr_epochs = _TRAINING_TUPLES.get(owner.training_tuple_id)
    if expected_lr_epochs != (owner.learning_rate, owner.training_epochs):
        raise LocalAffordanceModelPreparationError("owner training tuple is not frozen")
    expected_parameters = PARAMETERS_BY_CONDITION[owner.condition_id]
    expected_architecture_sha = _digest(
        {
            "condition_id": owner.condition_id,
            "input_width": 49 if owner.condition_id == CONDITION_B2 else 54,
            "trainable_parameters": expected_parameters,
            "hidden_widths": [48, 24],
            "optimizer": "adam",
        }
    )
    if (
        owner.trainable_parameters != expected_parameters
        or owner.architecture_sha256 != expected_architecture_sha
        or not _is_sha256(owner.owner_id)
    ):
        raise LocalAffordanceModelPreparationError("owner capacity or architecture identity differs")
    expected_temperatures = tuple(
        f"{owner.training_tuple_id}-{suffix}" for suffix in ("t0p6", "t0p9", "t1p2")
    )
    if owner.search_temperature_ids != expected_temperatures:
        raise LocalAffordanceModelPreparationError(
            "owner must share exactly its three frozen temperature consumers"
        )

    linked_units = tuple(unit for unit in plan.units if unit.model_owner_id == owner.owner_id)
    if len(linked_units) != 24 or tuple(
        unit.tuple_id for unit in linked_units
    ) != expected_temperatures * 8:
        raise LocalAffordanceModelPreparationError("owner consumer units are incomplete or reordered")
    task_ids = tuple(unit.unit.key.task_id for unit in linked_units)
    task_indices = tuple(unit.unit.key.task_index for unit in linked_units)
    if (
        any(task_ids[index : index + 3].count(task_ids[index]) != 3 for index in range(0, 24, 3))
        or len(set(task_ids[::3])) != 8
        or tuple(unit.tuple_id for unit in linked_units[:3]) != expected_temperatures
        or any(
            task_ids[index] != task_ids[index + 1]
            or task_ids[index] != task_ids[index + 2]
            or task_indices[index] != task_indices[index + 1]
            or task_indices[index] != task_indices[index + 2]
            for index in range(0, 24, 3)
        )
    ):
        raise LocalAffordanceModelPreparationError("owner consumer task order differs")
    if any(
        unit.condition_id != owner.condition_id
        or unit.training_tuple_id != owner.training_tuple_id
        or unit.fold_id != owner.fold_id
        or unit.heldout_family != owner.heldout_family
        or unit.view_id != view.view_id
        or unit.unit.key.phase != "validation"
        or unit.unit.key.family_id != owner.heldout_family
        or unit.unit.key.replicate != owner.replicate
        for unit in linked_units
    ):
        raise LocalAffordanceModelPreparationError("owner consumer-unit lineage differs")


def _validate_examples(
    condition_id: str,
    examples: Sequence[DecisionExample] | Sequence[GlobalDecisionExample],
) -> tuple[DecisionExample, ...] | tuple[GlobalDecisionExample, ...]:
    if type(examples) is not tuple or not examples:
        raise LocalAffordanceModelPreparationError("ordered nonempty training examples are required")
    expected_type = GlobalDecisionExample if condition_id == CONDITION_B2 else DecisionExample
    width = (
        PROBE_FEATURE_COUNT
        if condition_id == CONDITION_B2
        else STATE_CONDITIONED_FEATURE_COUNT
    )
    if any(type(example) is not expected_type for example in examples):
        raise LocalAffordanceModelPreparationError("example representation differs from owner condition")
    for example in examples:
        features = example.candidate_features
        if (
            type(features) is not torch.Tensor
            or features.device.type != "cpu"
            or features.dtype != torch.float32
            or features.ndim != 2
            or features.shape[0] < 1
            or features.shape[1] != width
            or not torch.isfinite(features).all()
            or isinstance(example.selected_index, bool)
            or not 0 <= example.selected_index < features.shape[0]
        ):
            raise LocalAffordanceModelPreparationError("training example tensor is invalid")
    return examples


def _seal_local_affordance_training_view(
    *,
    view: LocalAffordanceView,
    examples: Sequence[DecisionExample] | Sequence[GlobalDecisionExample],
    task_example_counts: Sequence[int],
) -> LocalAffordanceTrainingView:
    """Issue the builder-only receipt after all fold examples are materialized.

    The local-affordance training-view builder is the only production caller.
    Per-task counts preserve exact 40-task coverage without adding task identity
    to any learner-facing example.
    """

    if type(view) is not LocalAffordanceView:
        raise LocalAffordanceModelPreparationError("typed frozen view is required")
    frozen_examples = _validate_examples(view.condition_id, examples)
    counts = tuple(task_example_counts)
    if (
        len(view.training_task_ids) != 40
        or len(counts) != 40
        or any(type(count) is not int or count < 1 for count in counts)
        or sum(counts) != len(frozen_examples)
    ):
        raise LocalAffordanceModelPreparationError(
            "training view must cover each of its exact 40 ordered tasks"
        )
    digest = _digest(
        {
            "ordered_examples_sha256": _training_examples_sha256(frozen_examples),
            "task_example_counts": counts,
        }
    )
    return LocalAffordanceTrainingView(
        view=view,
        examples=frozen_examples,
        task_example_counts=counts,
        examples_sha256=digest,
        _token=_TRAINING_VIEW_TOKEN,
    )


def _validate_training_view(
    training_view: LocalAffordanceTrainingView,
    *,
    view: LocalAffordanceView,
    condition_id: str,
) -> tuple[DecisionExample, ...] | tuple[GlobalDecisionExample, ...]:
    if (
        type(training_view) is not LocalAffordanceTrainingView
        or training_view._token is not _TRAINING_VIEW_TOKEN
    ):
        raise LocalAffordanceModelPreparationError(
            "model training requires a sealed local-affordance training view"
        )
    if training_view.view != view or view.condition_id != condition_id:
        raise LocalAffordanceModelPreparationError(
            "sealed training view does not match the plan view identity"
        )
    examples = _validate_examples(condition_id, training_view.examples)
    if (
        training_view.example_count != len(examples)
        or len(training_view.task_example_counts) != len(view.training_task_ids)
        or any(type(count) is not int or count < 1 for count in training_view.task_example_counts)
        or sum(training_view.task_example_counts) != len(examples)
        or training_view.examples_sha256
        != _digest(
            {
                "ordered_examples_sha256": _training_examples_sha256(examples),
                "task_example_counts": training_view.task_example_counts,
            }
        )
    ):
        raise LocalAffordanceModelPreparationError(
            "sealed training-view count or content digest differs"
        )
    return examples


def _training_examples_sha256(
    examples: tuple[DecisionExample, ...] | tuple[GlobalDecisionExample, ...],
) -> str:
    digest = hashlib.sha256()
    for index, example in enumerate(examples):
        tensor = example.candidate_features.detach().contiguous()
        header = canonical_json_bytes(
            {
                "index": index,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "selected_index": example.selected_index,
            }
        )
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def prepare_local_affordance_model(
    *,
    plan: LocalAffordancePlan,
    view: LocalAffordanceView,
    owner: LocalAffordanceModelOwner,
    training_view: LocalAffordanceTrainingView,
) -> LocalAffordanceModelPreparation:
    """Train exactly one frozen owner from its same-data optimum examples.

    Plan source validation is deliberately performed by the readiness caller;
    this function performs no file access and rechecks exact in-memory lineage.
    The three temperature consumers refer to this single trained model.
    """

    _validate_plan_owner_view(plan, view, owner)
    validated_examples = _validate_training_view(
        training_view, view=view, condition_id=owner.condition_id
    )
    training = TrainingSpec(
        epochs=owner.training_epochs,
        learning_rate=owner.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    if owner.condition_id == CONDITION_B2:
        model, report = train_global_listwise_optimum_model(
            validated_examples, training=training, model_seed=owner.model_seed
        )
        expected_type: type[torch.nn.Module] = GlobalAffordanceScorer
    else:
        model, report = train_state_conditioned_optimum_model(
            validated_examples, training=training, model_seed=owner.model_seed
        )
        expected_type = StateConditionedScorer

    expected_parameters = PARAMETERS_BY_CONDITION[owner.condition_id]
    expected_report = TrainingReport(
        trainable_parameters=expected_parameters,
        optimizer_steps=owner.training_epochs,
        forward_passes=owner.training_epochs * len(validated_examples),
        training_examples=len(validated_examples),
    )
    if (
        type(model) is not expected_type
        or sum(parameter.numel() for parameter in model.parameters()) != expected_parameters
        or report != expected_report
    ):
        raise LocalAffordanceModelPreparationError("trained model capacity or cost accounting differs")
    model_state_sha256 = _model_state_sha256(model)
    examples_sha256 = training_view.examples_sha256
    identity = _digest(
        {
            "schema_version": "milestone6.phase3.local-affordance-model-preparation.v1",
            "plan_id": plan.plan_id,
            "owner_id": owner.owner_id,
            "view_id": view.view_id,
            "condition_id": owner.condition_id,
            "training_tuple_id": owner.training_tuple_id,
            "model_seed": owner.model_seed,
            "training_spec": {
                "epochs": training.epochs,
                "learning_rate": training.learning_rate,
                "weight_decay": training.weight_decay,
            },
            "architecture_id": ARCHITECTURE_ID_BY_CONDITION[owner.condition_id],
            "trainable_parameters": expected_parameters,
            "training_examples_sha256": examples_sha256,
            "training_report": {
                "optimizer_steps": report.optimizer_steps,
                "forward_passes": report.forward_passes,
                "training_examples": report.training_examples,
            },
            "model_state_sha256": model_state_sha256,
        }
    )
    return LocalAffordanceModelPreparation(
        plan_id=plan.plan_id,
        owner=owner,
        view=view,
        training_view=training_view,
        model=model,
        report=report,
        training_spec=training,
        training_examples_sha256=examples_sha256,
        model_state_sha256=model_state_sha256,
        model_identity_sha256=identity,
        search_temperature_ids=owner.search_temperature_ids,
        _token=_MODEL_PREPARATION_TOKEN,
    )


def require_local_affordance_model_preparation(
    prepared: LocalAffordanceModelPreparation,
    *,
    plan: LocalAffordancePlan,
) -> LocalAffordanceModelPreparation:
    """Validate an opaque prepared model immediately before artifact storage."""

    if (
        type(prepared) is not LocalAffordanceModelPreparation
        or prepared._token is not _MODEL_PREPARATION_TOKEN
    ):
        raise LocalAffordanceModelPreparationError(
            "model store requires a canonical sealed preparation"
        )
    if prepared.plan_id != plan.plan_id:
        raise LocalAffordanceModelPreparationError("preparation plan identity differs")
    _validate_plan_owner_view(plan, prepared.view, prepared.owner)
    examples = _validate_training_view(
        prepared.training_view,
        view=prepared.view,
        condition_id=prepared.owner.condition_id,
    )
    expected_parameters = PARAMETERS_BY_CONDITION[prepared.owner.condition_id]
    expected_type = (
        GlobalAffordanceScorer
        if prepared.owner.condition_id == CONDITION_B2
        else StateConditionedScorer
    )
    expected_report = TrainingReport(
        trainable_parameters=expected_parameters,
        optimizer_steps=prepared.owner.training_epochs,
        forward_passes=prepared.owner.training_epochs * len(examples),
        training_examples=len(examples),
    )
    expected_training = TrainingSpec(
        epochs=prepared.owner.training_epochs,
        learning_rate=prepared.owner.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    if (
        prepared.training_examples_sha256 != prepared.training_view.examples_sha256
        or type(prepared.model) is not expected_type
        or sum(parameter.numel() for parameter in prepared.model.parameters())
        != expected_parameters
        or prepared.report != expected_report
        or prepared.training_spec != expected_training
        or prepared.search_temperature_ids != prepared.owner.search_temperature_ids
        or prepared.model_state_sha256 != _model_state_sha256(prepared.model)
    ):
        raise LocalAffordanceModelPreparationError(
            "sealed preparation model, cost, or data digest differs"
        )
    expected_identity = _digest(
        {
            "schema_version": "milestone6.phase3.local-affordance-model-preparation.v1",
            "plan_id": prepared.plan_id,
            "owner_id": prepared.owner.owner_id,
            "view_id": prepared.view.view_id,
            "condition_id": prepared.owner.condition_id,
            "training_tuple_id": prepared.owner.training_tuple_id,
            "model_seed": prepared.owner.model_seed,
            "training_spec": {
                "epochs": prepared.training_spec.epochs,
                "learning_rate": prepared.training_spec.learning_rate,
                "weight_decay": prepared.training_spec.weight_decay,
            },
            "architecture_id": ARCHITECTURE_ID_BY_CONDITION[
                prepared.owner.condition_id
            ],
            "trainable_parameters": expected_parameters,
            "training_examples_sha256": prepared.training_examples_sha256,
            "training_report": {
                "optimizer_steps": prepared.report.optimizer_steps,
                "forward_passes": prepared.report.forward_passes,
                "training_examples": prepared.report.training_examples,
            },
            "model_state_sha256": prepared.model_state_sha256,
        }
    )
    if prepared.model_identity_sha256 != expected_identity:
        raise LocalAffordanceModelPreparationError("preparation identity hash differs")
    return prepared


__all__ = [
    "ARCHITECTURE_ID_BY_CONDITION",
    "CONDITION_B2",
    "FROZEN_LOCAL_PLAN_ID",
    "LocalAffordanceModelPreparation",
    "LocalAffordanceModelPreparationError",
    "LocalAffordanceTrainingView",
    "PARAMETERS_BY_CONDITION",
    "STATE_CONDITIONS",
    "WEIGHT_DECAY",
    "prepare_local_affordance_model",
    "require_local_affordance_model_preparation",
]
