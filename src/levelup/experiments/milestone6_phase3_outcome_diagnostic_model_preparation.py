"""In-memory model preparation for the frozen outcome-channel diagnostic.

This boundary consumes one exact, already-persisted Phase 3 evidence payload and
trains one of the 240 frozen RP/PEC model owners.  It never opens an environment,
generates candidates, evaluates outcomes, asks an optimum oracle, or touches a
result store.  Persistence and batch orchestration live in separate modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from levelup.experiments.milestone6_phase3_outcome_diagnostic_generation import (
    FROZEN_PARAMETER_COUNT,
    outcome_group_training_examples,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_model_artifacts import (
    AuthorizedOutcomeModelArtifact,
    OutcomeDiagnosticModelArtifactRecord,
    OutcomeStateTensorPayload,
    OutcomeTrainingAccounting,
    PinnedOutcomeModelState,
    PinnedOutcomeTrainingEvidence,
    build_outcome_model_artifact_key,
    build_outcome_model_artifact_record,
    inspect_outcome_model_state,
    validate_outcome_model_artifact_against_plan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_plan import (
    OutcomeModelOwner,
    OutcomeView,
    ValidatedOutcomePlan,
)
from levelup.experiments.milestone6_phase3_outcome_diagnostic_protocol import (
    OutcomeDiagnosticProtocolSnapshot,
)
from levelup.experiments.runner.training_data_artifacts import learner_samples
from levelup.learning.state_conditioned import (
    DecisionExample,
    StateConditionedScorer,
    TrainingReport,
    TrainingSpec,
    train_state_conditioned_optimum_model,
)

WEIGHT_DECAY = 0.0001
TORCH_THREADS = 1
_PREPARATION_TOKEN = object()


class OutcomeDiagnosticModelPreparationError(ValueError):
    """Raised when one prepared model differs from frozen data or authority."""


@dataclass(frozen=True, slots=True, init=False)
class PreparedOutcomeDiagnosticModel:
    """One fully validated model plus the exact bytes needed for persistence."""

    owner: OutcomeModelOwner
    view: OutcomeView
    examples: tuple[DecisionExample, ...]
    model: StateConditionedScorer
    report: TrainingReport
    state_payload: PinnedOutcomeModelState
    training_evidence: PinnedOutcomeTrainingEvidence
    record: OutcomeDiagnosticModelArtifactRecord
    authorization: AuthorizedOutcomeModelArtifact
    _token: object

    def __init__(
        self,
        *,
        owner: OutcomeModelOwner,
        view: OutcomeView,
        examples: tuple[DecisionExample, ...],
        model: StateConditionedScorer,
        report: TrainingReport,
        state_payload: PinnedOutcomeModelState,
        training_evidence: PinnedOutcomeTrainingEvidence,
        record: OutcomeDiagnosticModelArtifactRecord,
        authorization: AuthorizedOutcomeModelArtifact,
        _token: object | None = None,
    ) -> None:
        if _token is not _PREPARATION_TOKEN:
            raise OutcomeDiagnosticModelPreparationError(
                "prepared outcome models require canonical construction"
            )
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "view", view)
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "report", report)
        object.__setattr__(self, "state_payload", state_payload)
        object.__setattr__(self, "training_evidence", training_evidence)
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "authorization", authorization)
        object.__setattr__(self, "_token", _PREPARATION_TOKEN)


def outcome_model_state_payload(model: StateConditionedScorer) -> PinnedOutcomeModelState:
    """Capture exact finite float32 state bytes in canonical tensor-name order."""

    if type(model) is not StateConditionedScorer:
        raise OutcomeDiagnosticModelPreparationError(
            "outcome diagnostic preparation requires StateConditionedScorer"
        )
    rows: list[OutcomeStateTensorPayload] = []
    for name, tensor in sorted(model.state_dict().items()):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise OutcomeDiagnosticModelPreparationError(
                "outcome diagnostic state must contain float32 tensors"
            )
        value = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(value).all()):
            raise OutcomeDiagnosticModelPreparationError(
                "outcome diagnostic state contains non-finite values"
            )
        rows.append(
            OutcomeStateTensorPayload(
                name=name,
                shape=tuple(int(item) for item in value.shape),
                data=value.numpy().tobytes(order="C"),
            )
        )
    payload = PinnedOutcomeModelState(tuple(rows))
    inspect_outcome_model_state(payload)
    return payload


def _owner_and_view(
    plan: ValidatedOutcomePlan, owner_id: str
) -> tuple[OutcomeModelOwner, OutcomeView]:
    if type(plan) is not ValidatedOutcomePlan:
        raise OutcomeDiagnosticModelPreparationError(
            "outcome diagnostic preparation requires a validated plan"
        )
    owner = next((item for item in plan.plan.model_owners if item.owner_id == owner_id), None)
    if owner is None:
        raise OutcomeDiagnosticModelPreparationError(
            "model owner is foreign to the frozen outcome plan"
        )
    view = next((item for item in plan.plan.views if item.view_id == owner.view_id), None)
    if view is None:
        raise OutcomeDiagnosticModelPreparationError(
            "model owner view is missing from the frozen outcome plan"
        )
    consumers = tuple(item for item in plan.plan.units if item.model_owner_id == owner.owner_id)
    if (
        len(consumers) != 24
        or any(item.condition_id != owner.condition_id for item in consumers)
        or any(item.view_id != view.view_id for item in consumers)
        or tuple(owner.search_temperature_ids)
        != tuple(f"{owner.training_tuple_id}-{suffix}" for suffix in ("t0p6", "t0p9", "t1p2"))
    ):
        raise OutcomeDiagnosticModelPreparationError(
            "model owner consumer or temperature sharing matrix drifted"
        )
    return owner, view


def _validate_report(
    report: TrainingReport,
    owner: OutcomeModelOwner,
    examples: tuple[DecisionExample, ...],
) -> None:
    if type(report) is not TrainingReport or (
        report.trainable_parameters != FROZEN_PARAMETER_COUNT
        or report.optimizer_steps != owner.training_epochs
        or report.forward_passes != owner.training_epochs * len(examples)
        or report.training_examples != len(examples)
        or report.recurrent_steps != 0
    ):
        raise OutcomeDiagnosticModelPreparationError(
            "outcome diagnostic training accounting or capacity drifted"
        )


def _reconstruct_examples(
    evidence: PinnedOutcomeTrainingEvidence,
    condition_id: str,
) -> tuple[DecisionExample, ...]:
    if type(evidence) is not PinnedOutcomeTrainingEvidence:
        raise OutcomeDiagnosticModelPreparationError(
            "typed pinned outcome training evidence is required"
        )
    try:
        examples = outcome_group_training_examples(learner_samples(evidence.payload), condition_id)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelPreparationError(
            "outcome diagnostic training examples cannot be reconstructed"
        ) from exc
    if not examples or any(type(item) is not DecisionExample for item in examples):
        raise OutcomeDiagnosticModelPreparationError(
            "outcome diagnostic training examples are empty or foreign"
        )
    if any(
        item.candidate_features.dtype != torch.float32
        or not bool(torch.isfinite(item.candidate_features).all())
        for item in examples
    ):
        raise OutcomeDiagnosticModelPreparationError(
            "outcome diagnostic training examples are not finite float32 tensors"
        )
    return examples


def _examples_equal(left: tuple[DecisionExample, ...], right: tuple[DecisionExample, ...]) -> bool:
    return len(left) == len(right) and all(
        type(observed) is DecisionExample
        and observed.selected_index == expected.selected_index
        and torch.equal(observed.candidate_features, expected.candidate_features)
        for observed, expected in zip(left, right, strict=True)
    )


def prepare_outcome_diagnostic_model(
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    *,
    owner_id: str,
    training_evidence: PinnedOutcomeTrainingEvidence,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
) -> PreparedOutcomeDiagnosticModel:
    """Train and validate one exact RP/PEC model owner on CPU with one thread."""

    owner, view = _owner_and_view(plan, owner_id)
    examples = _reconstruct_examples(training_evidence, owner.condition_id)
    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS or torch.get_num_interop_threads() != 1:
        raise OutcomeDiagnosticModelPreparationError(
            "outcome diagnostic preparation requires one torch and interop thread"
        )
    training = TrainingSpec(
        epochs=owner.training_epochs,
        learning_rate=owner.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    try:
        model, report = train_state_conditioned_optimum_model(
            examples,
            training=training,
            model_seed=owner.model_seed,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelPreparationError(
            "outcome diagnostic owner training failed"
        ) from exc
    if type(model) is not StateConditionedScorer:
        raise OutcomeDiagnosticModelPreparationError(
            "outcome diagnostic trainer returned a foreign architecture"
        )
    _validate_report(report, owner, examples)
    state_payload = outcome_model_state_payload(model)
    accounting = OutcomeTrainingAccounting(
        optimizer_steps=report.optimizer_steps,
        forward_passes=report.forward_passes,
        training_examples=report.training_examples,
        serialization_calls=1,
    )
    try:
        key = build_outcome_model_artifact_key(
            plan,
            snapshot,
            owner_id=owner.owner_id,
            state_payload=state_payload,
            training_evidence=training_evidence,
            device="cpu",
            training_accounting=accounting,
            preparation_git_commit_sha=preparation_git_commit_sha,
            preparation_provenance_sha256=preparation_provenance_sha256,
        )
        record = build_outcome_model_artifact_record(key)
        authorization = validate_outcome_model_artifact_against_plan(
            record,
            state_payload,
            training_evidence,
            plan,
            snapshot,
            preparation_git_commit_sha=preparation_git_commit_sha,
            preparation_provenance_sha256=preparation_provenance_sha256,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelPreparationError(
            "trained outcome model differs from frozen artifact authority"
        ) from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prepared = PreparedOutcomeDiagnosticModel(
        owner=owner,
        view=view,
        examples=examples,
        model=model,
        report=report,
        state_payload=state_payload,
        training_evidence=training_evidence,
        record=record,
        authorization=authorization,
        _token=_PREPARATION_TOKEN,
    )
    return validate_prepared_outcome_diagnostic_model(
        prepared,
        plan=plan,
        snapshot=snapshot,
        preparation_git_commit_sha=preparation_git_commit_sha,
        preparation_provenance_sha256=preparation_provenance_sha256,
    )


def validate_prepared_outcome_diagnostic_model(
    prepared: PreparedOutcomeDiagnosticModel,
    *,
    plan: ValidatedOutcomePlan,
    snapshot: OutcomeDiagnosticProtocolSnapshot,
    preparation_git_commit_sha: str,
    preparation_provenance_sha256: str,
) -> PreparedOutcomeDiagnosticModel:
    """Recompute live state, report, and semantic artifact authorization."""

    if (
        type(prepared) is not PreparedOutcomeDiagnosticModel
        or prepared._token is not _PREPARATION_TOKEN
    ):
        raise OutcomeDiagnosticModelPreparationError("prepared outcome model is not canonical")
    owner, view = _owner_and_view(plan, prepared.owner.owner_id)
    if prepared.owner != owner or prepared.view != view:
        raise OutcomeDiagnosticModelPreparationError(
            "prepared outcome owner/view differs from the frozen plan"
        )
    expected_examples = _reconstruct_examples(prepared.training_evidence, owner.condition_id)
    if not _examples_equal(prepared.examples, expected_examples):
        raise OutcomeDiagnosticModelPreparationError(
            "prepared outcome examples differ from pinned training evidence"
        )
    _validate_report(prepared.report, owner, prepared.examples)
    if prepared.model.training or any(
        parameter.requires_grad for parameter in prepared.model.parameters()
    ):
        raise OutcomeDiagnosticModelPreparationError(
            "prepared outcome model must be frozen in evaluation mode"
        )
    live_state = outcome_model_state_payload(prepared.model)
    if live_state != prepared.state_payload:
        raise OutcomeDiagnosticModelPreparationError("prepared outcome model state changed")
    try:
        authorization = validate_outcome_model_artifact_against_plan(
            prepared.record,
            live_state,
            prepared.training_evidence,
            plan,
            snapshot,
            preparation_git_commit_sha=preparation_git_commit_sha,
            preparation_provenance_sha256=preparation_provenance_sha256,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise OutcomeDiagnosticModelPreparationError(
            "prepared outcome artifact validation failed"
        ) from exc
    if (
        type(prepared.authorization) is not AuthorizedOutcomeModelArtifact
        or authorization != prepared.authorization
    ):
        raise OutcomeDiagnosticModelPreparationError("prepared outcome authorization changed")
    return prepared


__all__ = [
    "TORCH_THREADS",
    "WEIGHT_DECAY",
    "OutcomeDiagnosticModelPreparationError",
    "PreparedOutcomeDiagnosticModel",
    "outcome_model_state_payload",
    "prepare_outcome_diagnostic_model",
    "validate_prepared_outcome_diagnostic_model",
]
