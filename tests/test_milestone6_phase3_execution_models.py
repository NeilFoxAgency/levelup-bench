from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import levelup.experiments.milestone6_phase3_execution_models as execution_models
from levelup.experiments.milestone6_phase3_model_artifacts import (
    Phase3ModelArtifactCost,
    Phase3ModelArtifactIndex,
    Phase3ModelArtifactKey,
    Phase3ModelArtifactManifest,
    open_phase3_model_artifact_reader_at,
)
from levelup.experiments.milestone6_phase3_model_authority import (
    load_phase3_model_artifact_authority_bytes,
)
from levelup.experiments.milestone6_phase3_plan import (
    ValidatedPhase3Plan,
    bind_validated_phase3_plan,
    build_phase3_plan,
)
from levelup.experiments.runner import secure_fs


@pytest.fixture(scope="module")
def authority_and_plan():
    authority = load_phase3_model_artifact_authority_bytes(
        Path("configs/milestone6/phase3_model_artifact_authority.json").read_bytes()
    )
    plan = bind_validated_phase3_plan(build_phase3_plan())
    return authority, plan


def _synthetic_namespaces(root: Path) -> None:
    root.mkdir()
    for name in (
        "phase3-model-artifact-keys",
        "phase3-model-artifact-costs",
        "phase3-model-artifacts",
    ):
        (root / name).mkdir()


def test_wrapper_cannot_be_forged_with_a_raw_model() -> None:
    with pytest.raises(execution_models.Phase3ExecutionModelError, match="construction"):
        execution_models.AuthorizedPhase3LoadedModel(
            model=torch.nn.Linear(1, 1),
            planned_unit=object(),  # type: ignore[arg-type]
            owner=object(),  # type: ignore[arg-type]
            key=object(),  # type: ignore[arg-type]
            index=object(),  # type: ignore[arg-type]
            cost=object(),  # type: ignore[arg-type]
            manifest=object(),  # type: ignore[arg-type]
        )


def test_authorized_wrapper_rejects_model_state_mutation(
    authority_and_plan, monkeypatch
) -> None:
    from levelup.experiments.milestone6_phase3_models import _model_state_sha256
    from levelup.learning.state_conditioned import StateConditionedScorer

    _, plan = authority_and_plan
    planned = next(
        item
        for item in plan.plan.units
        if item.base_condition_id == "S-state-availability-listwise-optimum"
    )
    owner = next(
        item for item in plan.plan.model_owners if item.owner_id == planned.model_owner_id
    )
    model = StateConditionedScorer().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loaded = execution_models.AuthorizedPhase3LoadedModel(
        model=model,
        planned_unit=planned,
        owner=owner,
        key=Phase3ModelArtifactKey.model_construct(),
        index=Phase3ModelArtifactIndex.model_construct(),
        cost=Phase3ModelArtifactCost.model_construct(),
        manifest=Phase3ModelArtifactManifest.model_construct(
            state_sha256=_model_state_sha256(model)
        ),
        _construction_token=execution_models._CONSTRUCTION_TOKEN,
    )
    monkeypatch.setattr(execution_models, "_validate_authority_and_plan", lambda *_: None)
    monkeypatch.setattr(execution_models, "_resolve_unit", lambda _plan, unit: unit)
    monkeypatch.setattr(
        execution_models,
        "_owner_for_unit",
        lambda *_: (loaded.owner, object()),
    )
    monkeypatch.setattr(execution_models, "_validate_loaded_lineage", lambda *_: None)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    with pytest.raises(execution_models.Phase3ExecutionModelError, match="state changed"):
        execution_models.validate_authorized_phase3_loaded_model(
            loaded,
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            loaded.planned_unit,
        )


def test_stolen_token_cannot_authorize_forged_metadata(authority_and_plan) -> None:
    from levelup.experiments.milestone6_phase3_models import _model_state_sha256

    authority, plan = authority_and_plan
    model = torch.nn.Linear(1, 1).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    forged = execution_models.AuthorizedPhase3LoadedModel(
        model=model,
        planned_unit=plan.plan.units[0],
        owner=object(),  # type: ignore[arg-type]
        key=object(),  # type: ignore[arg-type]
        index=object(),  # type: ignore[arg-type]
        cost=object(),  # type: ignore[arg-type]
        manifest=SimpleNamespace(state_sha256=_model_state_sha256(model)),  # type: ignore[arg-type]
        _construction_token=execution_models._CONSTRUCTION_TOKEN,
    )
    with pytest.raises(execution_models.Phase3ExecutionModelError, match="metadata"):
        execution_models.validate_authorized_phase3_loaded_model(
            forged,
            authority,
            plan,
            plan.plan.units[0],
        )


def test_loaded_model_lease_cannot_be_used_after_context_deactivation(
    authority_and_plan,
) -> None:
    authority, plan = authority_and_plan
    planned = next(
        item
        for item in plan.plan.units
        if item.base_condition_id == "S-state-availability-listwise-optimum"
    )
    owner = next(
        item for item in plan.plan.model_owners if item.owner_id == planned.model_owner_id
    )
    model = torch.nn.Linear(1, 1).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loaded = execution_models.AuthorizedPhase3LoadedModel(
        model=model,
        planned_unit=planned,
        owner=owner,
        key=Phase3ModelArtifactKey.model_construct(),
        index=Phase3ModelArtifactIndex.model_construct(),
        cost=Phase3ModelArtifactCost.model_construct(),
        manifest=Phase3ModelArtifactManifest.model_construct(),
        _construction_token=execution_models._CONSTRUCTION_TOKEN,
    )
    object.__setattr__(loaded, "_active", False)
    with pytest.raises(execution_models.Phase3ExecutionModelError, match="lease"):
        execution_models.validate_authorized_phase3_loaded_model(
            loaded,
            authority,
            plan,
            planned,
        )


def test_wrong_authority_type_fails_before_any_store_read(authority_and_plan, tmp_path) -> None:
    authority, plan = authority_and_plan
    unit = plan.plan.units[0]
    with pytest.raises(execution_models.Phase3ExecutionModelError, match="typed model authority"):
        with execution_models.open_authorized_phase3_model(
            object(),  # type: ignore[arg-type]
            plan,
            unit,
            tmp_path / authority.artifact_store_id,
        ):
            raise AssertionError("unreachable")


def test_forged_validated_plan_body_fails_before_store_read(
    authority_and_plan, tmp_path
) -> None:
    authority, plan = authority_and_plan
    first = plan.plan.units[0]
    changed_seeds = first.unit.seeds.model_copy(
        update={"search_seed": first.unit.seeds.search_seed + 1}
    )
    changed = replace(first, unit=first.unit.model_copy(update={"seeds": changed_seeds}))
    forged_body = replace(plan.plan, units=(changed, *plan.plan.units[1:]))
    forged = ValidatedPhase3Plan(
        forged_body,
        {item.unit.unit_id: item for item in forged_body.units},
        _construction_token=plan._construction_token,
    )
    with pytest.raises(execution_models.Phase3ExecutionModelError, match="plan body"):
        with execution_models.open_authorized_phase3_model(
            authority,
            forged,
            changed,
            tmp_path / authority.artifact_store_id,
        ):
            raise AssertionError("unreachable")


def test_self_consistent_unpublished_authority_fails_before_store_read(
    authority_and_plan, tmp_path
) -> None:
    authority, plan = authority_and_plan
    changed = authority.model_copy(update={"generation_git_commit_sha": "f" * 40})
    changed = changed.model_copy(update={"authority_sha256": changed.expected_authority_sha256})
    assert changed.authority_sha256 == changed.expected_authority_sha256
    with pytest.raises(execution_models.Phase3ExecutionModelError, match="published"):
        with execution_models.open_authorized_phase3_model(
            changed,
            plan,
            plan.plan.units[0],
            tmp_path / authority.artifact_store_id,
        ):
            raise AssertionError("unreachable")


def test_wrong_unit_owner_fails_closed(authority_and_plan, tmp_path) -> None:
    authority, plan = authority_and_plan
    unit = plan.plan.units[0].__class__(
        unit=plan.plan.units[0].unit,
        base_condition_id=plan.plan.units[0].base_condition_id,
        tuple_id=plan.plan.units[0].tuple_id,
        training_tuple_id=plan.plan.units[0].training_tuple_id,
        fold_id=plan.plan.units[0].fold_id,
        heldout_family=plan.plan.units[0].heldout_family,
        model_owner_id="0" * 64,
        view_id=plan.plan.units[0].view_id,
    )
    with pytest.raises(execution_models.Phase3ExecutionModelError, match="planned unit"):
        with execution_models.open_authorized_phase3_model(
            authority,
            plan,
            unit,
            tmp_path / authority.artifact_store_id,
        ):
            raise AssertionError("unreachable")


def test_store_basename_is_authority_bound(authority_and_plan, tmp_path) -> None:
    authority, plan = authority_and_plan
    root = tmp_path / "wrong-store-name"
    with pytest.raises(execution_models.Phase3ExecutionModelError, match="basename"):
        with execution_models.open_authorized_phase3_model(
            authority,
            plan,
            plan.plan.units[0],
            root,
        ):
            raise AssertionError("unreachable")


def test_symlinked_store_root_is_rejected(authority_and_plan, tmp_path) -> None:
    authority, plan = authority_and_plan
    real = tmp_path / "real-store"
    real.mkdir()
    link = tmp_path / authority.artifact_store_id
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(execution_models.Phase3ExecutionModelError, match="symlink"):
        with execution_models.open_authorized_phase3_model(
            authority,
            plan,
            plan.plan.units[0],
            link,
        ):
            raise AssertionError("unreachable")


def test_namespace_substitution_is_detected_while_reader_is_pinned(tmp_path) -> None:
    root = tmp_path / "synthetic-store"
    _synthetic_namespaces(root)
    root_fd = secure_fs.open_directory_chain(root)
    try:
        with open_phase3_model_artifact_reader_at(root_fd) as reader:
            identities = execution_models._namespace_identities(root_fd, reader)
            moved = root / "phase3-model-artifact-keys.old"
            (root / "phase3-model-artifact-keys").rename(moved)
            (root / "phase3-model-artifact-keys").mkdir()
            with pytest.raises(execution_models.Phase3ExecutionModelError, match="replaced"):
                execution_models._recheck_namespaces(root, root_fd, reader, identities)
    finally:
        os.close(root_fd)


def test_caller_value_error_is_not_wrapped_by_context_boundary(
    authority_and_plan, tmp_path, monkeypatch
) -> None:
    authority, plan = authority_and_plan
    root = tmp_path / authority.artifact_store_id
    _synthetic_namespaces(root)
    class FakeLoaded:
        _active = True

    monkeypatch.setattr(execution_models, "AuthorizedPhase3LoadedModel", FakeLoaded)
    monkeypatch.setattr(execution_models, "_load_one", lambda *_: FakeLoaded())
    monkeypatch.setattr(
        execution_models,
        "validate_authorized_phase3_loaded_model",
        lambda *_: None,
    )
    with pytest.raises(ValueError, match="caller failure"):
        with execution_models.open_authorized_phase3_model(
            authority,
            plan,
            plan.plan.units[0],
            root,
        ):
            raise ValueError("caller failure")


def test_body_error_and_namespace_substitution_are_both_preserved(
    authority_and_plan, tmp_path, monkeypatch
) -> None:
    authority, plan = authority_and_plan
    root = tmp_path / authority.artifact_store_id
    _synthetic_namespaces(root)
    class FakeLoaded:
        _active = True

    monkeypatch.setattr(execution_models, "AuthorizedPhase3LoadedModel", FakeLoaded)
    monkeypatch.setattr(execution_models, "_load_one", lambda *_: FakeLoaded())
    monkeypatch.setattr(
        execution_models,
        "validate_authorized_phase3_loaded_model",
        lambda *_: None,
    )
    with pytest.raises(BaseExceptionGroup) as caught:
        with execution_models.open_authorized_phase3_model(
            authority,
            plan,
            plan.plan.units[0],
            root,
        ):
            original = root / "phase3-model-artifact-keys"
            original.rename(root / "phase3-model-artifact-keys.old")
            original.mkdir()
            raise ValueError("caller failure")
    assert any(isinstance(item, ValueError) for item in caught.value.exceptions)
    assert any(
        isinstance(item, execution_models.Phase3ExecutionModelError)
        for item in caught.value.exceptions
    )


def test_body_error_and_reader_teardown_error_are_both_preserved(
    authority_and_plan, tmp_path, monkeypatch
) -> None:
    authority, plan = authority_and_plan
    root = tmp_path / authority.artifact_store_id
    root.mkdir()

    class FakeLoaded:
        _active = True

    @contextmanager
    def failing_reader(_root_fd):
        try:
            yield object()
        finally:
            raise RuntimeError("reader teardown failure")

    monkeypatch.setattr(
        execution_models,
        "open_phase3_model_artifact_reader_at",
        failing_reader,
    )
    monkeypatch.setattr(execution_models, "_namespace_identities", lambda *_: ())
    monkeypatch.setattr(execution_models, "_recheck_namespaces", lambda *_: None)
    monkeypatch.setattr(execution_models, "AuthorizedPhase3LoadedModel", FakeLoaded)
    monkeypatch.setattr(execution_models, "_load_one", lambda *_: FakeLoaded())
    monkeypatch.setattr(
        execution_models,
        "validate_authorized_phase3_loaded_model",
        lambda *_: None,
    )
    with pytest.raises(BaseExceptionGroup) as caught:
        with execution_models.open_authorized_phase3_model(
            authority,
            plan,
            plan.plan.units[0],
            root,
        ):
            raise ValueError("caller failure")
    assert any(isinstance(item, ValueError) for item in caught.value.exceptions)
    assert any(
        isinstance(item, RuntimeError) and "teardown" in str(item)
        for item in caught.value.exceptions
    )


def test_factory_has_only_the_two_frozen_architectures() -> None:
    from levelup.learning.state_conditioned import (
        HistoryConditionedScorer,
        StateConditionedScorer,
    )

    assert type(execution_models._model_factory("state-availability-mlp-v1")) is StateConditionedScorer
    assert type(execution_models._model_factory("causal-history-gru-mlp-v1")) is HistoryConditionedScorer
    with pytest.raises(execution_models.Phase3ExecutionModelError, match="architecture"):
        execution_models._model_factory("forged-architecture")


def test_authority_json_fixture_is_not_a_final_authority(authority_and_plan) -> None:
    authority, _ = authority_and_plan
    assert (
        authority.authority_sha256
        == execution_models.EXPECTED_PHASE3_MODEL_AUTHORITY_SHA256
    )
    assert authority.development_only is True
    assert authority.final is False
    assert authority.final_family_accessed is False
    assert json.loads(
        Path("configs/milestone6/phase3_model_artifact_authority.json").read_text()
    )["final"] is False


def test_execution_cache_rejects_forged_constructor_and_foreign_unit(authority_and_plan):
    authority, plan = authority_and_plan
    with pytest.raises(execution_models.Phase3ExecutionModelError, match="construction token"):
        execution_models.Phase3ExecutionAuthorityCache(
            authority=authority,
            validated_plan=plan,
            units_by_id={},
            owners_by_id={},
            views_by_id={},
            rows_by_owner_id={},
        )
    cache = execution_models.build_phase3_execution_authority_cache(authority, plan)
    planned = plan.plan.units[0]
    changed = replace(
        planned,
        unit=planned.unit.model_copy(
            update={"unit_id": "f" * 64}
        ),
    )
    with pytest.raises(execution_models.Phase3ExecutionModelError, match="cached plan"):
        cache.resolve_unit(changed)


def test_execution_cache_validates_full_authority_once_across_model_opens(
    authority_and_plan, tmp_path, monkeypatch
):
    authority, plan = authority_and_plan
    calls = 0
    original = execution_models._validate_authority_and_plan

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(execution_models, "_validate_authority_and_plan", counted)
    cache = execution_models.build_phase3_execution_authority_cache(authority, plan)
    assert calls == 1
    root = tmp_path / authority.artifact_store_id
    _synthetic_namespaces(root)

    class FakeLoaded:
        _active = True

    monkeypatch.setattr(execution_models, "AuthorizedPhase3LoadedModel", FakeLoaded)
    monkeypatch.setattr(execution_models, "_load_one", lambda *args: FakeLoaded())
    monkeypatch.setattr(
        execution_models, "validate_authorized_phase3_loaded_model", lambda *args: None
    )
    for planned in plan.plan.units[:2]:
        with execution_models.open_authorized_phase3_model(
            authority, plan, planned, root, authority_cache=cache
        ):
            pass
    assert calls == 1
