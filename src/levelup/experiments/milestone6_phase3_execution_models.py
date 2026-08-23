"""Fail-closed loading of one authorized Phase 3 model for execution.

This module is intentionally a small boundary between the frozen development
authority and a later executor.  Callers provide only the typed authority, the
opaque validated logical plan, one typed planned unit, and the
artifact output root.  Model, key, artifact, and owner selection are all
derived from those authorities; callers cannot override any of them.

No environment, replay, search, evaluator, oracle, outcome, or final-family
module is imported here.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from levelup.experiments.milestone6_phase3_model_artifacts import (
    ARTIFACTS_DIR,
    COSTS_DIR,
    KEYS_DIR,
    Phase3ModelArtifactCost,
    Phase3ModelArtifactError,
    Phase3ModelArtifactIndex,
    Phase3ModelArtifactKey,
    Phase3ModelArtifactManifest,
    PinnedPhase3ModelArtifactReader,
    load_phase3_model_from_at,
    load_phase3_model_index_at,
    open_phase3_model_artifact_reader_at,
)
from levelup.experiments.milestone6_phase3_model_authority import (
    Phase3ModelArtifactAuthority,
    Phase3ModelAuthorityRow,
)
from levelup.experiments.milestone6_phase3_model_preparation import (
    EXPECTED_EVIDENCE,
    EXPECTED_MODELS,
    EXPECTED_VIEWS,
)
from levelup.experiments.milestone6_phase3_models import (
    H0_CONDITION,
    H4_CONDITION,
    H4_SHUFFLED_CONDITION,
    S_CONDITION,
    _model_state_sha256,
)
from levelup.experiments.milestone6_phase3_plan import (
    FAMILIES,
    REPLICATES,
    TRAINING_TUPLE_IDS,
    Phase3ModelOwner,
    Phase3Plan,
    Phase3PlannedUnit,
    ValidatedPhase3Plan,
    _plan_body,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes

_CONSTRUCTION_TOKEN = object()
_CONDITION_ORDER = (S_CONDITION, H0_CONDITION, H4_CONDITION, H4_SHUFFLED_CONDITION)
EXPECTED_PHASE3_MODEL_AUTHORITY_SHA256 = (
    "8771eb52433faf15d6e5e935902a5c935526ec0e6b8e34621c3d6a922aea1a52"
)


class Phase3ExecutionModelError(ValueError):
    """Raised when a model cannot be proven to belong to the frozen authority."""


@dataclass(frozen=True, slots=True, init=False)
class AuthorizedPhase3LoadedModel:
    """One model loaded from an authority-bound artifact bundle.

    The constructor token is private to this module.  This prevents callers
    from manufacturing a wrapper that looks execution-authorized while carrying
    an arbitrary model or lineage.  The wrapper is yielded while the pinned
    descriptors remain open; the context boundary rechecks namespace identities
    before closing them.
    """

    model: torch.nn.Module
    planned_unit: Phase3PlannedUnit
    owner: Phase3ModelOwner
    key: Phase3ModelArtifactKey
    index: Phase3ModelArtifactIndex
    cost: Phase3ModelArtifactCost
    manifest: Phase3ModelArtifactManifest
    _construction_token: object
    _active: bool

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        planned_unit: Phase3PlannedUnit,
        owner: Phase3ModelOwner,
        key: Phase3ModelArtifactKey,
        index: Phase3ModelArtifactIndex,
        cost: Phase3ModelArtifactCost,
        manifest: Phase3ModelArtifactManifest,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _CONSTRUCTION_TOKEN:
            raise Phase3ExecutionModelError(
                "authorized Phase 3 models require the canonical resolver construction token"
            )
        if not isinstance(model, torch.nn.Module):
            raise Phase3ExecutionModelError("authorized model is not a torch module")
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise Phase3ExecutionModelError("authorized model is not eval/no-grad")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "planned_unit", planned_unit)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "_construction_token", _CONSTRUCTION_TOKEN)
        object.__setattr__(self, "_active", True)


def validate_authorized_phase3_loaded_model(
    value: object,
    authority: Phase3ModelArtifactAuthority,
    validated_plan: ValidatedPhase3Plan,
    planned_unit: Phase3PlannedUnit,
) -> None:
    """Revalidate one loaded wrapper against every frozen authority boundary."""

    if (
        type(value) is not AuthorizedPhase3LoadedModel
        or value._construction_token is not _CONSTRUCTION_TOKEN  # type: ignore[union-attr]
    ):
        raise Phase3ExecutionModelError(
            "value is not a canonical authorized Phase 3 loaded model"
        )
    if value._active is not True:  # type: ignore[union-attr]
        raise Phase3ExecutionModelError(
            "authorized Phase 3 model lease is no longer active"
        )
    _validate_authority_and_plan(authority, validated_plan)
    planned = _resolve_unit(validated_plan, planned_unit)
    owner, row = _owner_for_unit(authority, validated_plan, planned)
    if (
        value.planned_unit != planned  # type: ignore[union-attr]
        or value.owner != owner  # type: ignore[union-attr]
        or type(value.planned_unit) is not Phase3PlannedUnit  # type: ignore[union-attr]
        or type(value.owner) is not Phase3ModelOwner  # type: ignore[union-attr]
        or type(value.key) is not Phase3ModelArtifactKey  # type: ignore[union-attr]
        or type(value.index) is not Phase3ModelArtifactIndex  # type: ignore[union-attr]
        or type(value.cost) is not Phase3ModelArtifactCost  # type: ignore[union-attr]
        or type(value.manifest) is not Phase3ModelArtifactManifest  # type: ignore[union-attr]
    ):
        raise Phase3ExecutionModelError(
            "authorized Phase 3 wrapper metadata differs from frozen authority"
        )
    _validate_loaded_lineage(
        authority,
        planned,
        owner,
        row,
        value.key,  # type: ignore[union-attr]
        value.index,  # type: ignore[union-attr]
        value.cost,  # type: ignore[union-attr]
        value.manifest,  # type: ignore[union-attr]
    )
    if value.model.training or any(  # type: ignore[union-attr]
        parameter.requires_grad for parameter in value.model.parameters()  # type: ignore[union-attr]
    ):
        raise Phase3ExecutionModelError("authorized Phase 3 model is not eval/no-grad")
    from levelup.learning.state_conditioned import (
        HistoryConditionedScorer,
        StateConditionedScorer,
    )

    expected_type = (
        StateConditionedScorer if owner.condition_id == S_CONDITION else HistoryConditionedScorer
    )
    if type(value.model) is not expected_type:  # type: ignore[union-attr]
        raise Phase3ExecutionModelError(
            "authorized Phase 3 model class differs from frozen architecture"
        )
    if _model_state_sha256(value.model) != value.manifest.state_sha256:  # type: ignore[union-attr]
        raise Phase3ExecutionModelError("authorized Phase 3 model state changed after loading")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fail(message: str, exc: BaseException | None = None) -> None:
    if exc is None:
        raise Phase3ExecutionModelError(message)
    raise Phase3ExecutionModelError(message) from exc


def _validate_authority_and_plan(
    authority: Phase3ModelArtifactAuthority,
    validated_plan: ValidatedPhase3Plan,
) -> None:
    if type(authority) is not Phase3ModelArtifactAuthority:
        _fail("Phase 3 execution requires the canonical typed model authority")
    if type(validated_plan) is not ValidatedPhase3Plan:
        _fail("Phase 3 execution requires an opaque validated logical plan")
    try:
        if authority.authority_sha256 != authority.expected_authority_sha256:
            _fail("Phase 3 model authority self-hash differs")
        if authority.authority_sha256 != EXPECTED_PHASE3_MODEL_AUTHORITY_SHA256:
            _fail("Phase 3 model authority differs from the published execution authority")
        if (
            authority.development_only,
            authority.final,
            authority.final_family_accessed,
            authority.execution_authorized,
        ) != (True, False, False, True):
            _fail("Phase 3 execution authority is not development-only")
        if authority.family_order != FAMILIES or authority.replicates != REPLICATES:
            _fail("Phase 3 authority family or replicate universe differs")
        if authority.training_tuple_ids != TRAINING_TUPLE_IDS:
            _fail("Phase 3 authority training tuple universe differs")
        if authority.condition_ids != _CONDITION_ORDER:
            _fail("Phase 3 authority condition universe differs")
        plan = validated_plan.plan
        if type(plan) is not Phase3Plan:
            _fail("Phase 3 validated plan body is not the canonical immutable type")
        if plan.final_family_access:
            _fail("Phase 3 execution plan is not development-only")
        if (
            plan.plan_id != authority.plan_id
            or plan.protocol_sha256 != authority.protocol_sha256
            or plan.family_order != authority.family_order
            or plan.replicates != authority.replicates
            or set(plan.condition_ids) != set(authority.condition_ids)
        ):
            _fail("Phase 3 authority and validated plan lineage differs")
        if (
            authority.expected_evidence_count != EXPECTED_EVIDENCE
            or authority.expected_view_count != EXPECTED_VIEWS
            or authority.expected_model_count != EXPECTED_MODELS
            or len(plan.views) != authority.expected_view_count
            or len(plan.model_owners) != authority.expected_model_count
        ):
            _fail("Phase 3 authority and plan counts differ")
        owner_ids = tuple(sorted(owner.owner_id for owner in plan.model_owners))
        if owner_ids != authority.owner_ids:
            _fail("Phase 3 authority owner universe differs from the plan")
        mapping = [(item.unit.unit_id, item.model_owner_id) for item in plan.units]
        if _sha256_json(mapping) != authority.unit_owner_mapping_sha256:
            _fail("Phase 3 unit-to-owner mapping differs from authority")
        if _sha256_json(_plan_body(plan)) != authority.plan_id:
            _fail("Phase 3 validated plan body differs from published authority")
    except Phase3ExecutionModelError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        _fail("Phase 3 authority or validated plan is malformed", exc)


def _resolve_unit(
    validated_plan: ValidatedPhase3Plan,
    planned_unit: Phase3PlannedUnit,
) -> Phase3PlannedUnit:
    if type(planned_unit) is not Phase3PlannedUnit:
        _fail("Phase 3 planned unit must be the canonical typed unit")
    matches = [
        item
        for item in validated_plan.plan.units
        if item.unit.unit_id == planned_unit.unit.unit_id
    ]
    if len(matches) != 1 or matches[0] != planned_unit:
        _fail("Phase 3 planned unit differs from the canonical plan body")
    try:
        validated_plan.require_unit(planned_unit)
    except (AttributeError, TypeError, ValueError) as exc:
        _fail("Phase 3 planned unit is not in the validated frozen plan", exc)
    return planned_unit


def _owner_for_unit(
    authority: Phase3ModelArtifactAuthority,
    validated_plan: ValidatedPhase3Plan,
    planned: Phase3PlannedUnit,
) -> tuple[Phase3ModelOwner, Phase3ModelAuthorityRow]:
    owners = [
        owner
        for owner in validated_plan.plan.model_owners
        if owner.owner_id == planned.model_owner_id
    ]
    if len(owners) != 1:
        _fail("Phase 3 planned unit owner is absent or duplicated")
    owner = owners[0]
    views = [view for view in validated_plan.plan.views if view.view_id == owner.view_id]
    if len(views) != 1:
        _fail("Phase 3 planned owner view is absent or duplicated")
    view = views[0]
    rows = [row for row in authority.models if row.owner_id == owner.owner_id]
    if len(rows) != 1:
        _fail("Phase 3 authority row for planned owner is absent or duplicated")
    row = rows[0]
    if (
        owner.condition_id != planned.base_condition_id
        or owner.view_id != planned.view_id
        or owner.fold_id != planned.fold_id
        or owner.heldout_family != planned.heldout_family
        or owner.replicate != planned.unit.key.replicate
        or owner.training_tuple_id != planned.training_tuple_id
        or view.condition_id != owner.condition_id
        or view.fold_id != owner.fold_id
        or view.heldout_family != owner.heldout_family
        or view.replicate != owner.replicate
        or planned.unit.key.phase != "validation"
        or planned.unit.key.family_id != planned.heldout_family
        or planned.unit.key.condition_id != f"{planned.base_condition_id}--{planned.tuple_id}"
    ):
        _fail("Phase 3 planned unit and owner lineage differs")
    return owner, row


def _model_factory(architecture_id: str) -> torch.nn.Module:
    # The architecture is selected only from the canonical artifact key.  There
    # is deliberately no caller-supplied factory or architecture override.
    from levelup.learning.state_conditioned import (
        HistoryConditionedScorer,
        StateConditionedScorer,
    )

    if architecture_id == "state-availability-mlp-v1":
        return StateConditionedScorer()
    if architecture_id == "causal-history-gru-mlp-v1":
        return HistoryConditionedScorer()
    _fail("Phase 3 artifact architecture is not frozen")


def _validate_loaded_lineage(
    authority: Phase3ModelArtifactAuthority,
    planned: Phase3PlannedUnit,
    owner: Phase3ModelOwner,
    row: Phase3ModelAuthorityRow,
    key: Phase3ModelArtifactKey,
    index: Phase3ModelArtifactIndex,
    cost: Phase3ModelArtifactCost,
    manifest: Phase3ModelArtifactManifest,
) -> None:
    manifest_sha = hashlib.sha256(
        canonical_json_bytes(manifest.model_dump(mode="json"))
    ).hexdigest()
    if (
        index.key_id != row.key_id
        or index.artifact_id != row.artifact_id
        or cost.cost_id != row.cost_id
        or manifest_sha != row.manifest_sha256
        or key.key_id != row.key_id
        or manifest.artifact_id != row.artifact_id
        or cost.key_id != row.key_id
        or cost.artifact_id != row.artifact_id
        or index.key != key
        or cost.key != key
        or manifest.key != key
    ):
        _fail("Phase 3 artifact identities differ from authority")
    if (
        key.plan_id != authority.plan_id
        or key.protocol_sha256 != authority.protocol_sha256
        or key.evidence_lock_sha256 != authority.evidence_lock_sha256
        or key.owner_id != owner.owner_id
        or key.view_id != owner.view_id
        or key.condition_id != owner.condition_id
        or key.fold_id != owner.fold_id
        or key.heldout_family != owner.heldout_family
        or key.replicate != owner.replicate
        or key.training_tuple_id != owner.training_tuple_id
        or key.model_seed != owner.model_seed
        or key.optimizer.optimizer_id != "adam"
        or key.optimizer.learning_rate != owner.learning_rate
        or key.optimizer.weight_decay != 0.0001
        or key.report.optimizer_steps != owner.training_epochs
        or key.architecture_id not in {"state-availability-mlp-v1", "causal-history-gru-mlp-v1"}
    ):
        _fail("Phase 3 model owner/training identity differs from authority")
    if (
        planned.model_owner_id != owner.owner_id
        or planned.view_id != key.view_id
        or planned.fold_id != key.fold_id
        or planned.heldout_family != key.heldout_family
        or planned.training_tuple_id != key.training_tuple_id
    ):
        _fail("Phase 3 planned unit identity differs from loaded model")
    expected_architecture = (
        "state-availability-mlp-v1"
        if owner.condition_id == S_CONDITION
        else "causal-history-gru-mlp-v1"
    )
    if key.architecture_id != expected_architecture:
        _fail("Phase 3 model architecture differs from frozen condition")


def _check_root_name(output_root: str | Path, authority: Phase3ModelArtifactAuthority) -> Path:
    path = Path(os.path.abspath(output_root))
    if path.name != authority.artifact_store_id:
        _fail("Phase 3 artifact output root basename differs from authority")
    for candidate in (path, *path.parents):
        try:
            if os.path.lexists(candidate) and candidate.is_symlink():
                _fail("Phase 3 artifact output root or ancestor is a symlink")
        except OSError as exc:
            _fail("Phase 3 artifact output root cannot be inspected", exc)
    return path


def _namespace_identities(
    root_fd: int, reader: PinnedPhase3ModelArtifactReader
) -> tuple[tuple[int, int], ...]:
    try:
        return (
            secure_fs.directory_identity(root_fd),
            secure_fs.directory_identity(reader.keys_fd),
            secure_fs.directory_identity(reader.costs_fd),
            secure_fs.directory_identity(reader.artifacts_fd),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _fail("Phase 3 artifact namespace identity cannot be read", exc)


def _recheck_namespaces(
    root_path: Path,
    root_fd: int,
    reader: PinnedPhase3ModelArtifactReader,
    identities: tuple[tuple[int, int], ...],
) -> None:
    if _namespace_identities(root_fd, reader) != identities:
        _fail("Phase 3 artifact namespace descriptor changed")
    try:
        current_root = secure_fs.open_directory_chain(root_path)
        try:
            current = [secure_fs.directory_identity(current_root)]
            for name in (KEYS_DIR, COSTS_DIR, ARTIFACTS_DIR):
                child = secure_fs.open_child_directory(current_root, name)
                try:
                    current.append(secure_fs.directory_identity(child))
                finally:
                    os.close(child)
        finally:
            os.close(current_root)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _fail("Phase 3 artifact root or namespace was replaced", exc)
    if tuple(current) != identities:
        _fail("Phase 3 artifact root or namespace was replaced")


def _load_one(
    authority: Phase3ModelArtifactAuthority,
    validated_plan: ValidatedPhase3Plan,
    planned: Phase3PlannedUnit,
    reader: PinnedPhase3ModelArtifactReader,
) -> AuthorizedPhase3LoadedModel:
    owner, row = _owner_for_unit(authority, validated_plan, planned)
    try:
        committed_index = load_phase3_model_index_at(reader, row.key_id)
        key = committed_index.key
    except (Phase3ModelArtifactError, TypeError, ValueError) as exc:
        _fail("Phase 3 authority key index cannot be loaded", exc)
    if (
        committed_index.artifact_id != row.artifact_id
        or committed_index.manifest_sha256 != row.manifest_sha256
    ):
        _fail("Phase 3 authority key index differs from the published row")
    try:
        with torch.no_grad():
            model, index, cost, manifest = load_phase3_model_from_at(
                reader,
                key,
                model_factory=_model_factory,
            )
    except (Phase3ModelArtifactError, TypeError, ValueError, RuntimeError) as exc:
        _fail("Phase 3 authorized model bundle cannot be loaded", exc)
    _validate_loaded_lineage(authority, planned, owner, row, key, index, cost, manifest)
    from levelup.learning.state_conditioned import (
        HistoryConditionedScorer,
        StateConditionedScorer,
    )

    expected_type = (
        StateConditionedScorer
        if owner.condition_id == S_CONDITION
        else HistoryConditionedScorer
    )
    if type(model) is not expected_type:
        _fail("Phase 3 loaded model type differs from frozen architecture")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loaded = AuthorizedPhase3LoadedModel(
        model=model,
        planned_unit=planned,
        owner=owner,
        key=key,
        index=index,
        cost=cost,
        manifest=manifest,
        _construction_token=_CONSTRUCTION_TOKEN,
    )
    validate_authorized_phase3_loaded_model(
        loaded,
        authority,
        validated_plan,
        planned,
    )
    return loaded


def _raise_collected_errors(label: str, errors: list[BaseException]) -> None:
    if not errors:
        return
    if len(errors) == 1:
        error = errors[0]
        raise error.with_traceback(error.__traceback__)
    raise BaseExceptionGroup(label, errors)


@contextmanager
def open_authorized_phase3_model(
    authority: Phase3ModelArtifactAuthority,
    validated_plan: ValidatedPhase3Plan,
    planned_unit: Phase3PlannedUnit,
    artifact_output_root: str | Path,
) -> Iterator[AuthorizedPhase3LoadedModel]:
    """Pin and load one model selected solely by the frozen planned unit."""

    _validate_authority_and_plan(authority, validated_plan)
    planned = _resolve_unit(validated_plan, planned_unit)
    root_path = _check_root_name(artifact_output_root, authority)
    stack = ExitStack()
    setup_errors: list[BaseException] = []
    try:
        root_fd = secure_fs.open_directory_chain(root_path)
        stack.callback(os.close, root_fd)
        reader = stack.enter_context(open_phase3_model_artifact_reader_at(root_fd))
        identities = _namespace_identities(root_fd, reader)
        _recheck_namespaces(root_path, root_fd, reader, identities)
        loaded = _load_one(authority, validated_plan, planned, reader)
        _recheck_namespaces(root_path, root_fd, reader, identities)
    except BaseException as exc:
        if isinstance(exc, Phase3ExecutionModelError):
            setup_errors.append(exc)
        elif isinstance(
            exc,
            (
                Phase3ModelArtifactError,
                secure_fs.SecureFilesystemError,
                OSError,
                TypeError,
                ValueError,
            ),
        ):
            wrapped = Phase3ExecutionModelError(
                "Phase 3 authorized model resolution failed"
            )
            wrapped.__cause__ = exc
            setup_errors.append(wrapped)
        else:
            setup_errors.append(exc)
    if setup_errors:
        try:
            stack.close()
        except BaseException as exc:
            setup_errors.append(exc)
        _raise_collected_errors("Phase 3 model setup and teardown failed", setup_errors)

    errors: list[BaseException] = []
    try:
        yield loaded
    except BaseException as exc:
        errors.append(exc)
    try:
        validate_authorized_phase3_loaded_model(
            loaded,
            authority,
            validated_plan,
            planned,
        )
        _recheck_namespaces(root_path, root_fd, reader, identities)
    except BaseException as exc:
        errors.append(exc)
    if type(loaded) is AuthorizedPhase3LoadedModel:
        object.__setattr__(loaded, "_active", False)
    else:
        errors.append(
            Phase3ExecutionModelError(
                "Phase 3 resolver returned a noncanonical model lease"
            )
        )
    try:
        stack.close()
    except BaseException as exc:
        errors.append(exc)
    _raise_collected_errors(
        "Phase 3 execution body, authorization recheck, or teardown failed",
        errors,
    )


__all__ = [
    "AuthorizedPhase3LoadedModel",
    "EXPECTED_PHASE3_MODEL_AUTHORITY_SHA256",
    "Phase3ExecutionModelError",
    "open_authorized_phase3_model",
    "validate_authorized_phase3_loaded_model",
]
