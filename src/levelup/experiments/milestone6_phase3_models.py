"""In-memory Phase 3 representation views and model preparation.

This is intentionally an additive preparation boundary.  It consumes an already
validated Phase 2 learner payload and never opens a store, regenerates probes, runs
an environment, searches, or asks an evaluator/oracle.  The resulting objects are
plain immutable values suitable for a later execution writer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from levelup.experiments.milestone6_phase3_plan import (
    Phase3ModelOwner,
    Phase3Plan,
    Phase3View,
    validate_phase3_plan,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataArtifactManifest,
    TrainingDataEvidenceManifest,
    TrainingDataPayload,
    learner_samples,
)
from levelup.learning.state_conditioned import (
    DecisionExample,
    HistoryConditionedScorer,
    HistoryDecisionExample,
    StateConditionedScorer,
    TrainingReport,
    TrainingSpec,
    causal_history_optimum_examples,
    null_history_optimum_examples,
    optimum_imitation_examples,
    permutation_map_sha256,
    shuffled_history_optimum_examples,
    state_availability_optimum_examples,
    train_history_optimum_model,
    train_state_conditioned_optimum_model,
)

S_CONDITION = "S-state-availability-listwise-optimum"
H0_CONDITION = "H0-null-history-transition-listwise-optimum"
H4_CONDITION = "H4-causal-history-transition-listwise-optimum"
H4_SHUFFLED_CONDITION = "H4-shuffled-history-transition-listwise-optimum"
HISTORY_CONDITIONS = (H0_CONDITION, H4_CONDITION, H4_SHUFFLED_CONDITION)
S_PARAMETERS = 3_841
HISTORY_PARAMETERS = 3_889
WEIGHT_DECAY = 0.0001
_HEX = frozenset("0123456789abcdef")
_PHASE3_VIEW_PREPARATION_TOKEN = object()
_PHASE3_MODEL_PREPARATION_TOKEN = object()


class Phase3ModelPreparationError(ValueError):
    """Raised when a view, evidence payload, or model owner is not identical."""


@dataclass(frozen=True, slots=True)
class HistoryShuffleDiagnostics:
    eligible_windows: int
    map_nonidentity_windows: int
    effective_tensor_changed_windows: int
    duplicate_vector_no_effect_windows: int
    unchanged_short_windows: int
    permutation_map_sha256: str | None

    @property
    def effective_change_fraction(self) -> float:
        return (
            self.effective_tensor_changed_windows / self.eligible_windows
            if self.eligible_windows
            else 1.0
        )

    @property
    def claim_eligible(self) -> bool:
        return self.eligible_windows > 0 and self.effective_change_fraction >= 0.80


@dataclass(frozen=True, slots=True, init=False)
class Phase3ViewPreparation:
    view: Phase3View
    evidence_payload_sha256: str
    evidence_payload_bytes: int
    sample_task_ids: tuple[str, ...]
    examples: tuple[DecisionExample, ...] | tuple[HistoryDecisionExample, ...]
    # T examples are retained for exact same-data comparisons and are never trained.
    transition_examples: tuple[DecisionExample, ...]
    history_shuffle: HistoryShuffleDiagnostics | None = None
    representation_identity_sha256: str = ""
    authority_validated: bool = False
    _construction_token: object | None = None

    def __init__(
        self,
        *,
        view: Phase3View,
        evidence_payload_sha256: str,
        evidence_payload_bytes: int,
        sample_task_ids: tuple[str, ...],
        examples: tuple[DecisionExample, ...] | tuple[HistoryDecisionExample, ...],
        transition_examples: tuple[DecisionExample, ...],
        authority_validated: bool,
        history_shuffle: HistoryShuffleDiagnostics | None = None,
        representation_identity_sha256: str = "",
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _PHASE3_VIEW_PREPARATION_TOKEN:
            raise Phase3ModelPreparationError(
                "prepared views require the canonical Phase 3 view builder"
            )
        object.__setattr__(self, "view", view)
        object.__setattr__(self, "evidence_payload_sha256", evidence_payload_sha256)
        object.__setattr__(self, "evidence_payload_bytes", evidence_payload_bytes)
        object.__setattr__(self, "sample_task_ids", sample_task_ids)
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "transition_examples", transition_examples)
        object.__setattr__(self, "history_shuffle", history_shuffle)
        object.__setattr__(
            self, "representation_identity_sha256", representation_identity_sha256
        )
        object.__setattr__(self, "authority_validated", authority_validated)
        object.__setattr__(self, "_construction_token", _construction_token)


def validate_phase3_view_preparation(
    prepared: Any, *, require_authority: bool = True
) -> None:
    """Reject view wrappers not produced by the canonical evidence boundary."""

    if (
        not isinstance(prepared, Phase3ViewPreparation)
        or prepared._construction_token is not _PHASE3_VIEW_PREPARATION_TOKEN
    ):
        raise Phase3ModelPreparationError(
            "model training requires a canonical prepared Phase 3 view"
        )
    if require_authority and not prepared.authority_validated:
        raise Phase3ModelPreparationError(
            "model training requires frozen view authority"
        )
    _require_sha(prepared.evidence_payload_sha256, "evidence payload identity")
    _require_sha(
        prepared.representation_identity_sha256, "representation identity"
    )
    if prepared.evidence_payload_bytes < 1:
        raise Phase3ModelPreparationError("evidence payload byte count is invalid")
    if prepared.sample_task_ids != prepared.view.training_task_ids:
        raise Phase3ModelPreparationError("prepared view task order differs")
    if len(prepared.examples) != len(prepared.transition_examples):
        raise Phase3ModelPreparationError("prepared view example count differs")


@dataclass(frozen=True, slots=True, init=False)
class Phase3ModelPreparation:
    owner: Phase3ModelOwner
    view: Phase3ViewPreparation
    model: torch.nn.Module
    report: TrainingReport
    training_spec: TrainingSpec
    model_state_sha256: str
    model_identity_sha256: str
    search_temperature_ids: tuple[str, ...]
    authority_validated: bool
    _construction_token: object

    def __init__(
        self,
        *,
        owner: Phase3ModelOwner,
        view: Phase3ViewPreparation,
        model: torch.nn.Module,
        report: TrainingReport,
        training_spec: TrainingSpec,
        model_state_sha256: str,
        model_identity_sha256: str,
        search_temperature_ids: tuple[str, ...],
        authority_validated: bool,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _PHASE3_MODEL_PREPARATION_TOKEN:
            raise Phase3ModelPreparationError(
                "prepared models require the canonical Phase 3 trainer"
            )
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "view", view)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "report", report)
        object.__setattr__(self, "training_spec", training_spec)
        object.__setattr__(self, "model_state_sha256", model_state_sha256)
        object.__setattr__(self, "model_identity_sha256", model_identity_sha256)
        object.__setattr__(self, "search_temperature_ids", search_temperature_ids)
        object.__setattr__(self, "authority_validated", authority_validated)
        object.__setattr__(self, "_construction_token", _construction_token)


def validate_phase3_model_preparation(
    prepared: Any, *, require_authority: bool = True
) -> None:
    """Reject wrappers not produced by the canonical in-memory trainer."""

    if (
        not isinstance(prepared, Phase3ModelPreparation)
        or prepared._construction_token is not _PHASE3_MODEL_PREPARATION_TOKEN
    ):
        raise Phase3ModelPreparationError(
            "production generation requires a canonical prepared Phase 3 model"
        )
    if require_authority and not prepared.authority_validated:
        raise Phase3ModelPreparationError(
            "production generation requires frozen plan authority"
        )
    owner = prepared.owner
    view = prepared.view
    expected_parameters = (
        S_PARAMETERS if owner.condition_id == S_CONDITION else HISTORY_PARAMETERS
    )
    expected_type = (
        StateConditionedScorer
        if owner.condition_id == S_CONDITION
        else HistoryConditionedScorer
    )
    if owner.condition_id not in {S_CONDITION, *HISTORY_CONDITIONS}:
        raise Phase3ModelPreparationError("prepared model condition is unknown")
    if type(prepared.model) is not expected_type:
        raise Phase3ModelPreparationError("prepared model architecture differs")
    if (
        owner.view_id != view.view.view_id
        or owner.condition_id != view.view.condition_id
        or (owner.fold_id, owner.heldout_family, owner.replicate)
        != (view.view.fold_id, view.view.heldout_family, view.view.replicate)
        or prepared.search_temperature_ids != owner.search_temperature_ids
    ):
        raise Phase3ModelPreparationError("prepared model owner/view lineage differs")
    if prepared.training_spec != TrainingSpec(
        epochs=owner.training_epochs,
        learning_rate=owner.learning_rate,
        weight_decay=WEIGHT_DECAY,
    ):
        raise Phase3ModelPreparationError("prepared model training spec differs")
    actual_parameters = sum(parameter.numel() for parameter in prepared.model.parameters())
    if (
        actual_parameters != expected_parameters
        or prepared.report.trainable_parameters != expected_parameters
        or prepared.report.training_examples != len(view.examples)
        or prepared.report.optimizer_steps != owner.training_epochs
        or prepared.report.forward_passes
        != owner.training_epochs * len(view.examples)
    ):
        raise Phase3ModelPreparationError("prepared model accounting differs")
    expected_recurrent_steps = owner.training_epochs * sum(
        int(example.history_features.shape[0])
        for example in view.examples
        if isinstance(example, HistoryDecisionExample)
    )
    if prepared.report.recurrent_steps != expected_recurrent_steps:
        raise Phase3ModelPreparationError("prepared model recurrent accounting differs")
    observed_state_sha256 = _model_state_sha256(prepared.model)
    if prepared.model_state_sha256 != observed_state_sha256:
        raise Phase3ModelPreparationError("prepared model state differs")
    if prepared.model_identity_sha256 != _model_identity_sha256(
        owner, view, model_state_sha256=observed_state_sha256
    ):
        raise Phase3ModelPreparationError("prepared model identity differs")


def _model_identity_sha256(
    owner: Phase3ModelOwner,
    prepared_view: Phase3ViewPreparation,
    *,
    model_state_sha256: str,
) -> str:
    expected_parameters = (
        S_PARAMETERS if owner.condition_id == S_CONDITION else HISTORY_PARAMETERS
    )
    architecture_id = (
        "state-availability-mlp-v1"
        if owner.condition_id == S_CONDITION
        else "causal-history-gru-mlp-v1"
    )
    return _digest(
        {
            "owner_id": owner.owner_id,
            "condition_id": owner.condition_id,
            "view_id": owner.view_id,
            "evidence_payload_sha256": prepared_view.evidence_payload_sha256,
            "representation_identity_sha256": (
                prepared_view.representation_identity_sha256
            ),
            "history_permutation_map_sha256": (
                prepared_view.history_shuffle.permutation_map_sha256
                if prepared_view.history_shuffle is not None
                else None
            ),
            "model_seed": owner.model_seed,
            "training_tuple_id": owner.training_tuple_id,
            "learning_rate": owner.learning_rate,
            "training_epochs": owner.training_epochs,
            "optimizer": "adam",
            "weight_decay": WEIGHT_DECAY,
            "architecture_id": architecture_id,
            "trainable_parameters": expected_parameters,
            "model_state_sha256": model_state_sha256,
        }
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _model_state_sha256(model: torch.nn.Module) -> str:
    """Hash exact ordered tensor metadata and bytes without serialization noise."""

    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        header = canonical_json_bytes(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        )
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX for c in value):
        raise Phase3ModelPreparationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _manifest_value(manifest: Any, field: str, default: Any = None) -> Any:
    if isinstance(manifest, Mapping):
        return manifest.get(field, default)
    return getattr(manifest, field, default)


def _manifest_key(manifest: Any) -> Any:
    return _manifest_value(manifest, "key", {})


def _key_value(manifest: Any, field: str, default: Any = None) -> Any:
    key = _manifest_key(manifest)
    if isinstance(key, Mapping):
        return key.get(field, default)
    return getattr(key, field, default)


def _validate_evidence(
    payload: TrainingDataPayload,
    manifest: TrainingDataArtifactManifest | Mapping[str, Any],
    payload_bytes: bytes | None,
) -> tuple[str, int, tuple[str, ...]]:
    if not isinstance(payload, TrainingDataPayload):
        raise Phase3ModelPreparationError("evidence payload must be TrainingDataPayload")
    canonical = payload.model_dump(mode="json")
    derived_bytes = canonical_json_bytes(canonical)
    observed_bytes = derived_bytes if payload_bytes is None else payload_bytes
    if not isinstance(observed_bytes, bytes) or not observed_bytes:
        raise Phase3ModelPreparationError("evidence payload bytes are missing")
    if observed_bytes != derived_bytes:
        raise Phase3ModelPreparationError("evidence payload bytes do not match payload")
    digest = hashlib.sha256(observed_bytes).hexdigest()
    manifest_digest = _manifest_value(manifest, "payload_sha256")
    if manifest_digest != digest:
        raise Phase3ModelPreparationError("evidence payload digest does not match manifest")
    if _manifest_value(manifest, "payload_bytes") != len(observed_bytes):
        raise Phase3ModelPreparationError("evidence payload byte count does not match manifest")
    sample_ids = tuple(sample.task_id for sample in payload.samples)
    manifest_ids = tuple(_manifest_value(manifest, "sample_task_ids", ()))
    key_ids = tuple(_key_value(manifest, "ordered_training_task_ids", ()))
    if sample_ids != manifest_ids or sample_ids != key_ids:
        raise Phase3ModelPreparationError("evidence sample task order does not match manifest")
    return digest, len(observed_bytes), sample_ids


def _validate_view_lineage(
    view: Phase3View,
    manifest: Any,
    sample_ids: tuple[str, ...],
) -> None:
    key_fields = {
        "fold_id": "fold_id",
        "heldout_family": "heldout_family_id",
        "replicate": "replicate",
        "data_order_seed": "data_order_seed",
    }
    for view_field, key_field in key_fields.items():
        expected = _key_value(manifest, key_field)
        if expected is not None and getattr(view, view_field) != expected:
            raise Phase3ModelPreparationError(f"view {view_field} differs from evidence key")
    if tuple(view.training_task_ids) != sample_ids:
        raise Phase3ModelPreparationError("view task order differs from evidence payload")
    _require_sha(view.view_id, "view identity")
    _require_sha(view.evidence_lineage_sha256, "view evidence lineage")
    _require_sha(view.representation_sha256, "view representation identity")
    condition = _key_value(manifest, "condition_id")
    # A Phase 3 view is allowed to be a new representation over the same evidence;
    # only an explicit condition key is checked, never silently substituted.
    if isinstance(condition, str) and condition not in {
        view.condition_id,
        "T-markov-state-transition-listwise-optimum",
        "C-state-conditioned-listwise-optimum",
    }:
        raise Phase3ModelPreparationError("view condition differs from evidence key")
    expected_evidence_lineage = _digest(
        {
            "phase2_config_sha256": _key_value(manifest, "protocol_sha256"),
            "expected_units_sha256": _key_value(
                manifest, "expected_unit_plan_sha256"
            ),
            "training_task_ids": sample_ids,
            "replicate": view.replicate,
        }
    )
    if view.evidence_lineage_sha256 != expected_evidence_lineage:
        raise Phase3ModelPreparationError("view evidence lineage digest differs")


def _require_plan_view(
    plan: Phase3Plan | None,
    view: Phase3View,
    *,
    allow_test_identity: bool,
) -> None:
    if allow_test_identity:
        return
    if plan is None:
        raise Phase3ModelPreparationError("a validated Phase 3 plan is required")
    validate_phase3_plan(plan)
    matches = tuple(item for item in plan.views if item.view_id == view.view_id)
    if len(matches) != 1 or matches[0] != view:
        raise Phase3ModelPreparationError("view differs from the frozen Phase 3 plan")


def _require_plan_owner(
    plan: Phase3Plan | None,
    owner: Phase3ModelOwner,
    *,
    allow_test_identity: bool,
) -> None:
    if allow_test_identity:
        return
    if plan is None:
        raise Phase3ModelPreparationError("a validated Phase 3 plan is required")
    validate_phase3_plan(plan)
    matches = tuple(item for item in plan.model_owners if item.owner_id == owner.owner_id)
    if len(matches) != 1 or matches[0] != owner:
        raise Phase3ModelPreparationError("model owner differs from the frozen Phase 3 plan")


def _shuffle_diagnostics(
    records: Sequence[Mapping[str, Any]],
    causal: Sequence[HistoryDecisionExample],
    shuffled: Sequence[HistoryDecisionExample],
) -> HistoryShuffleDiagnostics:
    eligible = map_nonidentity = changed = duplicate = short = 0
    for left, right, record in zip(causal, shuffled, records, strict=True):
        indices = tuple(record["input_transition_indices"])
        output = tuple(record["permuted_transition_indices"])
        if len(indices) < 2:
            short += 1
            continue
        eligible += 1
        if indices != output:
            map_nonidentity += 1
        if torch.equal(left.history_features, right.history_features):
            duplicate += 1
        else:
            changed += 1
    digest = permutation_map_sha256(records)
    return HistoryShuffleDiagnostics(eligible, map_nonidentity, changed, duplicate, short, digest)


def prepare_phase3_view(
    payload: TrainingDataPayload,
    manifest: TrainingDataEvidenceManifest
    | TrainingDataArtifactManifest
    | Mapping[str, Any],
    view: Phase3View,
    *,
    plan: Phase3Plan | None = None,
    payload_bytes: bytes | None = None,
    trace_or_episode_ids: Sequence[str] | None = None,
    _allow_test_identity: bool = False,
) -> Phase3ViewPreparation:
    """Build one temperature-independent Phase 3 view from existing evidence."""

    _require_plan_view(plan, view, allow_test_identity=_allow_test_identity)
    if not _allow_test_identity and not isinstance(
        manifest, (TrainingDataEvidenceManifest, TrainingDataArtifactManifest)
    ):
        raise Phase3ModelPreparationError(
            "production evidence manifest must be a typed validated manifest"
        )
    digest, byte_count, sample_ids = _validate_evidence(payload, manifest, payload_bytes)
    _validate_view_lineage(view, manifest, sample_ids)
    expected_trace_ids = tuple(f"optimum:{task_id}:0" for task_id in sample_ids)
    if trace_or_episode_ids is None:
        trace_or_episode_ids = expected_trace_ids
    elif not _allow_test_identity and tuple(trace_or_episode_ids) != expected_trace_ids:
        raise Phase3ModelPreparationError("history trace identities are not canonical")
    samples = learner_samples(payload)
    transition = tuple(optimum_imitation_examples(samples))
    if view.condition_id == S_CONDITION:
        examples: tuple[DecisionExample, ...] | tuple[HistoryDecisionExample, ...] = (
            state_availability_optimum_examples(samples)
        )
        shuffle = None
    elif view.condition_id in HISTORY_CONDITIONS:
        if trace_or_episode_ids is None or len(trace_or_episode_ids) != len(sample_ids):
            raise Phase3ModelPreparationError("history views require per-sample trace identities")
        if view.condition_id == H0_CONDITION:
            examples = null_history_optimum_examples(
                samples,
                task_ids=sample_ids,
                trace_or_episode_ids=trace_or_episode_ids,
            )
            shuffle = None
        elif view.condition_id == H4_CONDITION:
            examples = causal_history_optimum_examples(
                samples,
                task_ids=sample_ids,
                trace_or_episode_ids=trace_or_episode_ids,
            )
            shuffle = None
        else:
            records: list[dict[str, Any]] = []
            shuffled = shuffled_history_optimum_examples(
                samples,
                fold_id=view.fold_id,
                replicate=view.replicate,
                task_ids=sample_ids,
                phase="train",
                trace_or_episode_ids=trace_or_episode_ids,
                permutation_records=records,
            )
            causal = causal_history_optimum_examples(
                samples,
                task_ids=sample_ids,
                trace_or_episode_ids=trace_or_episode_ids,
            )
            examples = shuffled
            shuffle = _shuffle_diagnostics(records, causal, shuffled)
    else:
        raise Phase3ModelPreparationError(f"unknown Phase 3 condition: {view.condition_id}")
    return Phase3ViewPreparation(
        view=view,
        evidence_payload_sha256=digest,
        evidence_payload_bytes=byte_count,
        sample_task_ids=sample_ids,
        examples=examples,
        transition_examples=transition,
        history_shuffle=shuffle,
        representation_identity_sha256=view.representation_sha256,
        authority_validated=not _allow_test_identity,
        _construction_token=_PHASE3_VIEW_PREPARATION_TOKEN,
    )


def prepare_phase3_model(
    prepared_view: Phase3ViewPreparation,
    owner: Phase3ModelOwner,
    *,
    plan: Phase3Plan | None = None,
    _allow_test_identity: bool = False,
) -> Phase3ModelPreparation:
    """Train one frozen owner; its three search temperatures share this model."""

    validate_phase3_view_preparation(
        prepared_view, require_authority=not _allow_test_identity
    )
    _require_plan_owner(plan, owner, allow_test_identity=_allow_test_identity)
    if owner.view_id != prepared_view.view.view_id:
        raise Phase3ModelPreparationError("model owner view identity differs")
    _require_sha(owner.owner_id, "model owner identity")
    if owner.condition_id != prepared_view.view.condition_id:
        raise Phase3ModelPreparationError("model owner condition differs")
    if (owner.fold_id, owner.heldout_family, owner.replicate) != (
        prepared_view.view.fold_id,
        prepared_view.view.heldout_family,
        prepared_view.view.replicate,
    ):
        raise Phase3ModelPreparationError("model owner fold/replicate lineage differs")
    if len(owner.search_temperature_ids) != 3 or len(set(owner.search_temperature_ids)) != 3:
        raise Phase3ModelPreparationError("model owner must share exactly three temperatures")
    training_tuple_ids = {
        "lr0p003-e120",
        "lr0p003-e180",
        "lr0p01-e120",
        "lr0p01-e180",
    }
    if owner.training_tuple_id not in training_tuple_ids:
        raise Phase3ModelPreparationError("model owner training tuple is unknown")
    expected_temperatures = tuple(
        f"{owner.training_tuple_id}-{temperature}" for temperature in ("t0p6", "t0p9", "t1p2")
    )
    if tuple(owner.search_temperature_ids) != expected_temperatures:
        raise Phase3ModelPreparationError("model owner temperature reuse matrix drifted")
    training = TrainingSpec(
        epochs=owner.training_epochs,
        learning_rate=owner.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    if owner.condition_id == S_CONDITION:
        model, report = train_state_conditioned_optimum_model(
            prepared_view.examples, training=training, model_seed=owner.model_seed
        )
        expected = S_PARAMETERS
    elif owner.condition_id in HISTORY_CONDITIONS:
        model, report = train_history_optimum_model(
            prepared_view.examples, training=training, model_seed=owner.model_seed
        )
        expected = HISTORY_PARAMETERS
    else:
        raise Phase3ModelPreparationError("unknown Phase 3 model condition")
    if report.trainable_parameters != expected:
        raise Phase3ModelPreparationError("model capacity is not frozen")
    if report.training_examples != len(prepared_view.examples):
        raise Phase3ModelPreparationError("training example accounting drifted")
    expected_forward_passes = owner.training_epochs * len(prepared_view.examples)
    if (
        report.optimizer_steps != owner.training_epochs
        or report.forward_passes != expected_forward_passes
    ):
        raise Phase3ModelPreparationError("model optimizer/forward accounting drifted")
    expected_recurrent_steps = (
        owner.training_epochs
        * sum(
            int(example.history_features.shape[0])
            for example in prepared_view.examples
            if isinstance(example, HistoryDecisionExample)
        )
    )
    if report.recurrent_steps != expected_recurrent_steps:
        raise Phase3ModelPreparationError("model recurrent accounting drifted")
    model_state_sha256 = _model_state_sha256(model)
    model_identity = _model_identity_sha256(
        owner,
        prepared_view,
        model_state_sha256=model_state_sha256,
    )
    return Phase3ModelPreparation(
        owner=owner,
        view=prepared_view,
        model=model,
        report=report,
        training_spec=training,
        model_state_sha256=model_state_sha256,
        model_identity_sha256=model_identity,
        search_temperature_ids=owner.search_temperature_ids,
        authority_validated=(
            prepared_view.authority_validated and not _allow_test_identity
        ),
        _construction_token=_PHASE3_MODEL_PREPARATION_TOKEN,
    )


# Explicit aliases make the preparation boundary discoverable to execution code.
build_phase3_view = prepare_phase3_view
train_phase3_model = prepare_phase3_model


__all__ = [
    "H0_CONDITION",
    "H4_CONDITION",
    "H4_SHUFFLED_CONDITION",
    "S_CONDITION",
    "HistoryShuffleDiagnostics",
    "Phase3ModelPreparation",
    "Phase3ModelPreparationError",
    "Phase3ViewPreparation",
    "build_phase3_view",
    "prepare_phase3_model",
    "prepare_phase3_view",
    "train_phase3_model",
    "validate_phase3_model_preparation",
    "validate_phase3_view_preparation",
]
