from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import levelup.experiments.milestone6_phase3_local_affordance_models as models
from levelup.experiments.milestone6_phase3_local_affordance_plan import (
    build_local_affordance_plan,
)
from levelup.learning.state_conditioned import (
    DecisionExample,
    GlobalAffordanceScorer,
    GlobalDecisionExample,
    StateConditionedScorer,
    TrainingReport,
)


@pytest.fixture(scope="module")
def plan():
    return build_local_affordance_plan()


def _examples(condition: str):
    if condition == models.CONDITION_B2:
        source = GlobalDecisionExample(torch.tensor([[0.0] * 49, [1.0] * 49]), 1)
    else:
        source = DecisionExample(torch.tensor([[0.0] * 54, [1.0] * 54]), 1)
    return tuple(source for _ in range(40))


def _owner_view(plan, condition: str):
    owner = next(item for item in plan.model_owners if item.condition_id == condition)
    view = next(item for item in plan.views if item.view_id == owner.view_id)
    return owner, view


def _training_view(view, condition: str | None = None):
    return models._seal_local_affordance_training_view(
        view=view,
        examples=_examples(condition or view.condition_id),
        task_example_counts=(1,) * 40,
    )


def _fast_train(monkeypatch):
    def train_global(examples, *, training, model_seed):
        torch.manual_seed(model_seed)
        model = GlobalAffordanceScorer()
        return model, TrainingReport(
            3601, training.epochs, training.epochs * len(examples), len(examples)
        )

    def train_state(examples, *, training, model_seed):
        torch.manual_seed(model_seed)
        model = StateConditionedScorer()
        return model, TrainingReport(
            3841, training.epochs, training.epochs * len(examples), len(examples)
        )

    monkeypatch.setattr(models, "train_global_listwise_optimum_model", train_global)
    monkeypatch.setattr(models, "train_state_conditioned_optimum_model", train_state)


@pytest.mark.parametrize(
    ("condition", "model_type", "parameter_count"),
    [
        (models.CONDITION_B2, GlobalAffordanceScorer, 3601),
        ("S-state-availability-listwise-optimum", StateConditionedScorer, 3841),
        ("P-state-availability-alias-pooled-outcome-listwise-optimum", StateConditionedScorer, 3841),
        ("L-state-availability-local-outcome-listwise-optimum", StateConditionedScorer, 3841),
    ],
)
def test_prepares_each_ladder_condition_with_frozen_capacity_and_cost(
    plan, monkeypatch, condition, model_type, parameter_count
):
    _fast_train(monkeypatch)
    owner, view = _owner_view(plan, condition)
    training_view = _training_view(view)
    prepared = models.prepare_local_affordance_model(
        plan=plan, view=view, owner=owner, training_view=training_view
    )

    assert type(prepared.model) is model_type
    assert sum(parameter.numel() for parameter in prepared.model.parameters()) == parameter_count
    assert prepared.report == TrainingReport(
        parameter_count,
        owner.training_epochs,
        owner.training_epochs * 40,
        40,
    )
    assert prepared.training_spec.epochs == owner.training_epochs
    assert prepared.training_spec.learning_rate == owner.learning_rate
    assert prepared.training_spec.weight_decay == 0.0001
    assert prepared.search_temperature_ids == owner.search_temperature_ids
    assert len(prepared.search_temperature_ids) == 3
    assert prepared.training_examples_sha256 == training_view.examples_sha256
    assert len(prepared.model_state_sha256) == 64
    assert len(prepared.model_identity_sha256) == 64


def test_preparation_is_deterministic_for_same_owner_and_examples(plan, monkeypatch):
    _fast_train(monkeypatch)
    owner, view = _owner_view(plan, models.CONDITION_B2)
    training_view = _training_view(view)
    first = models.prepare_local_affordance_model(
        plan=plan, view=view, owner=owner, training_view=training_view
    )
    second = models.prepare_local_affordance_model(
        plan=plan, view=view, owner=owner, training_view=training_view
    )
    assert first.model_state_sha256 == second.model_state_sha256
    assert first.model_identity_sha256 == second.model_identity_sha256


def test_rejects_owner_or_view_not_exactly_bound_to_plan(plan, monkeypatch):
    _fast_train(monkeypatch)
    owner, view = _owner_view(plan, models.CONDITION_B2)
    training_view = _training_view(view)
    with pytest.raises(models.LocalAffordanceModelPreparationError, match="owner or view"):
        models.prepare_local_affordance_model(
            plan=plan,
            view=view,
            owner=replace(owner, trainable_parameters=3841),
            training_view=training_view,
        )
    other_view = next(item for item in plan.views if item.view_id != view.view_id)
    with pytest.raises(models.LocalAffordanceModelPreparationError, match="lineage"):
        models.prepare_local_affordance_model(
            plan=plan, view=other_view, owner=owner, training_view=training_view
        )


def test_rejects_plan_that_opens_final_family_scope(plan, monkeypatch):
    _fast_train(monkeypatch)
    owner, view = _owner_view(plan, models.CONDITION_B2)
    training_view = _training_view(view)
    unsafe = replace(plan, final_family_access=True)
    with pytest.raises(models.LocalAffordanceModelPreparationError, match="final-family"):
        models.prepare_local_affordance_model(
            plan=unsafe, view=view, owner=owner, training_view=training_view
        )


def test_rejects_forged_plan_even_when_it_retains_owner_and_view(plan, monkeypatch):
    _fast_train(monkeypatch)
    owner, view = _owner_view(plan, models.CONDITION_B2)
    training_view = _training_view(view)
    forged = replace(plan, raw_manifest_file_sha256="a" * 64)
    forged = replace(forged, plan_id=models._digest(models._local_plan_body(forged)))

    with pytest.raises(models.LocalAffordanceModelPreparationError, match="frozen.*identity"):
        models.prepare_local_affordance_model(
            plan=forged, view=view, owner=owner, training_view=training_view
        )

    unhashed = replace(plan, raw_manifest_file_sha256="b" * 64)
    with pytest.raises(models.LocalAffordanceModelPreparationError, match="self-hash"):
        models.prepare_local_affordance_model(
            plan=unhashed, view=view, owner=owner, training_view=training_view
        )


def test_rejects_wrong_representation_and_modified_consumer_order(plan, monkeypatch):
    _fast_train(monkeypatch)
    owner, view = _owner_view(plan, models.CONDITION_B2)
    training_view = _training_view(view)
    with pytest.raises(models.LocalAffordanceModelPreparationError, match="sealed training view"):
        models.prepare_local_affordance_model(
            plan=plan,
            view=view,
            owner=owner,
            training_view=_training_view(
                next(item for item in plan.views if item.condition_id == "S-state-availability-listwise-optimum" and item.fold_id == view.fold_id and item.replicate == view.replicate)
            ),
        )

    linked = [index for index, unit in enumerate(plan.units) if unit.model_owner_id == owner.owner_id]
    changed_units = list(plan.units)
    changed_units[linked[0]], changed_units[linked[1]] = (
        changed_units[linked[1]],
        changed_units[linked[0]],
    )
    malformed = replace(plan, units=tuple(changed_units))
    with pytest.raises(models.LocalAffordanceModelPreparationError, match="self-hash"):
        models.prepare_local_affordance_model(
            plan=malformed, view=view, owner=owner, training_view=training_view
        )


def test_rejects_invalid_example_tensor(plan, monkeypatch):
    _fast_train(monkeypatch)
    owner, view = _owner_view(plan, "S-state-availability-listwise-optimum")
    training_view = _training_view(view)
    training_view.examples[0].candidate_features[0, 0] = float("nan")
    with pytest.raises(models.LocalAffordanceModelPreparationError, match="tensor is invalid"):
        models.prepare_local_affordance_model(
            plan=plan, view=view, owner=owner, training_view=training_view
        )


def test_rejects_bare_example_sequences_and_tampered_receipt_digest(plan, monkeypatch):
    _fast_train(monkeypatch)
    owner, view = _owner_view(plan, models.CONDITION_B2)
    with pytest.raises(models.LocalAffordanceModelPreparationError, match="sealed"):
        models.prepare_local_affordance_model(
            plan=plan,
            view=view,
            owner=owner,
            training_view=_examples(owner.condition_id),  # type: ignore[arg-type]
        )

    training_view = _training_view(view)
    object.__setattr__(training_view, "examples_sha256", "0" * 64)
    with pytest.raises(models.LocalAffordanceModelPreparationError, match="digest"):
        models.prepare_local_affordance_model(
            plan=plan, view=view, owner=owner, training_view=training_view
        )


def test_preparation_is_opaque_and_store_validator_rejects_rebinding(plan, monkeypatch):
    _fast_train(monkeypatch)
    owner, view = _owner_view(plan, models.CONDITION_B2)
    training_view = _training_view(view)
    prepared = models.prepare_local_affordance_model(
        plan=plan, view=view, owner=owner, training_view=training_view
    )
    assert models.require_local_affordance_model_preparation(prepared, plan=plan) is prepared

    with pytest.raises(models.LocalAffordanceModelPreparationError, match="canonical in-memory"):
        models.LocalAffordanceModelPreparation(
            plan_id=prepared.plan_id,
            owner=prepared.owner,
            view=prepared.view,
            training_view=prepared.training_view,
            model=prepared.model,
            report=prepared.report,
            training_spec=prepared.training_spec,
            training_examples_sha256=prepared.training_examples_sha256,
            model_state_sha256=prepared.model_state_sha256,
            model_identity_sha256=prepared.model_identity_sha256,
            search_temperature_ids=prepared.search_temperature_ids,
        )

    object.__setattr__(prepared, "training_examples_sha256", "f" * 64)
    with pytest.raises(models.LocalAffordanceModelPreparationError, match="digest differs"):
        models.require_local_affordance_model_preparation(prepared, plan=plan)
