"""All-or-nothing development-only local-affordance raw-probe capture.

This module is deliberately a narrow execution bridge.  It consumes two
already-active descriptor-pinned capabilities, spends the frozen observable
probe budget once for each of the 240 authorized keys, checks byte parity with
the pre-existing pooled development tables, and asks the immutable publisher to
activate the complete store exactly once.  It contains no training, search,
verification, outcome, or final-family behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from levelup.envs.adaptive_track import make_adaptive_track
from levelup.envs.challenge_track import make_combo_track
from levelup.experiments.milestone6_baselines import discover_affordances
from levelup.experiments.milestone6_phase3_local_affordance_canonical_tables import (
    CanonicalPooledTableError,
    CanonicalPooledTableSource,
)
from levelup.experiments.milestone6_phase3_local_affordance_evidence import (
    LocalAffordanceEvidenceError,
    RawProbeArtifactKey,
    SanitizedRawProbeArtifact,
    sanitize_probe_evidence,
)
from levelup.experiments.milestone6_phase3_local_affordance_raw_authority import (
    PersistedRawProbeArtifact,
    RawProbeAuthorityError,
    require_expected_raw_probe_authority,
)
from levelup.experiments.milestone6_phase3_local_affordance_raw_publication import (
    RawProbePublicationError,
    publish_raw_probe_store_from_readiness,
)
from levelup.experiments.milestone6_phase3_local_affordance_readiness import (
    LocalAffordanceActivationLease,
    LocalAffordanceReadinessError,
)

_RAW_ARTIFACT_COUNT = 240
_ACTIONS_PER_ARTIFACT = 64
_LOGICAL_CONSUMERS_PER_PHYSICAL_ACTION = 48
_PROBE_TARGET_SAMPLES_PER_ALIAS = 8
_ACTIONS_PER_ATTEMPT = 16


class RawProbeCaptureError(RuntimeError):
    """Raised when the one-shot raw capture cannot complete safely."""


@dataclass(frozen=True, slots=True)
class RawProbeCaptureSummary:
    """Immutable provenance and accounting for one complete publication.

    ``physical_probe_actions`` is the actual environment interaction count.
    ``logical_consumer_equivalent_actions`` is deliberately reported separately:
    the latter is the 48-consumer accounting view, not additional work.
    """

    manifest_id: str
    activation_git_commit: str
    ordered_key_ids: tuple[str, ...]
    ordered_artifact_ids: tuple[str, ...]
    physical_probe_actions: int
    logical_consumer_equivalent_actions: int
    probe_attempts: int
    probe_resets: int
    probe_wall_seconds: float
    training_actions: int = 0
    search_actions: int = 0
    replay_actions: int = 0
    evaluator_calls: int = 0
    oracle_calls: int = 0

    def __post_init__(self) -> None:
        if (
            len(self.ordered_key_ids) != _RAW_ARTIFACT_COUNT
            or len(self.ordered_artifact_ids) != _RAW_ARTIFACT_COUNT
            or len(set(self.ordered_key_ids)) != _RAW_ARTIFACT_COUNT
            or len(set(self.ordered_artifact_ids)) != _RAW_ARTIFACT_COUNT
        ):
            raise ValueError("raw capture summary must bind exactly 240 artifacts")
        if self.physical_probe_actions != _RAW_ARTIFACT_COUNT * _ACTIONS_PER_ARTIFACT:
            raise ValueError("raw capture physical probe action count drifted")
        if self.logical_consumer_equivalent_actions != (
            self.physical_probe_actions * _LOGICAL_CONSUMERS_PER_PHYSICAL_ACTION
        ):
            raise ValueError("raw capture logical action accounting drifted")
        if (
            type(self.probe_attempts) is not int
            or type(self.probe_resets) is not int
            or self.probe_attempts < 0
            or self.probe_resets < 0
            or not math.isfinite(self.probe_wall_seconds)
            or self.probe_wall_seconds < 0.0
        ):
            raise ValueError("raw capture probe accounting is invalid")
        if any(
            type(value) is not int or value != 0
            for value in (
                self.training_actions,
                self.search_actions,
                self.replay_actions,
                self.evaluator_calls,
                self.oracle_calls,
            )
        ):
            raise ValueError("raw capture may not report non-probe work")


def _require_capabilities(
    lease: object,
    canonical_tables: object,
) -> tuple[LocalAffordanceActivationLease, CanonicalPooledTableSource, tuple[RawProbeArtifactKey, ...]]:
    """Validate both unforgeable capabilities before opening any environment."""

    if type(lease) is not LocalAffordanceActivationLease:
        raise RawProbeCaptureError("raw capture requires an exact active readiness lease")
    if type(canonical_tables) is not CanonicalPooledTableSource:
        raise RawProbeCaptureError("raw capture requires exact canonical pooled tables")
    try:
        active_lease = lease.require_active()
        active_tables = canonical_tables.require_active()
        if active_tables is not canonical_tables or active_lease is not lease:
            raise RawProbeCaptureError("raw capture capabilities did not remain active")
        # A real source must be tied to this exact lease, not merely to an
        # independently active authority with coincidentally similar keys.
        if getattr(canonical_tables, "_lease", None) is not lease:
            raise RawProbeCaptureError("canonical pooled tables belong to another readiness lease")
        authority = require_expected_raw_probe_authority(lease.authority)
        keys = tuple(authority.keys)
    except (
        CanonicalPooledTableError,
        LocalAffordanceReadinessError,
        RawProbeAuthorityError,
        AttributeError,
        TypeError,
        ValueError,
    ) as exc:
        raise RawProbeCaptureError("raw capture capabilities are invalid or inactive") from exc
    if (
        len(keys) != _RAW_ARTIFACT_COUNT
        or any(type(key) is not RawProbeArtifactKey for key in keys)
        or len({key.key_id for key in keys}) != _RAW_ARTIFACT_COUNT
    ):
        raise RawProbeCaptureError("raw capture authority does not contain the exact 240 keys")
    return lease, canonical_tables, keys


def _environment_for(key: RawProbeArtifactKey) -> Any:
    if key.environment_seed != 0:
        raise RawProbeCaptureError("raw capture requires the frozen environment reset seed zero")
    if key.family_id == "combo":
        environment = make_combo_track(key.task_index, key.generator_seed)
    else:
        environment = make_adaptive_track(key.family_id, key.task_index, key.generator_seed)
    if environment.task_spec.task_id != key.task_id:
        raise RawProbeCaptureError("constructed environment task identity differs from authority")
    return environment


def _forbidden_aliases(environment: Any) -> frozenset[str]:
    """Derive exactly one public structured never-use constraint."""

    try:
        aliases = {
            constraint.verifier_config["action"]
            for constraint in environment.task_spec.constraints
            if constraint.verifier_id == "never_use_action"
            and isinstance(constraint.verifier_config, dict)
            and isinstance(constraint.verifier_config.get("action"), str)
        }
    except (AttributeError, TypeError, KeyError) as exc:
        raise RawProbeCaptureError("environment lacks a structured forbidden action") from exc
    if len(aliases) != 1:
        raise RawProbeCaptureError("raw capture requires exactly one forbidden action alias")
    return frozenset(aliases)


def _persist(artifact: SanitizedRawProbeArtifact) -> PersistedRawProbeArtifact:
    if type(artifact) is not SanitizedRawProbeArtifact:
        raise RawProbeCaptureError("sanitizer did not return an exact typed raw artifact")
    try:
        return PersistedRawProbeArtifact(
            key=artifact.key,
            body=artifact.body,
            manifest=artifact.manifest,
            affordances=artifact.affordances,
        )
    except (TypeError, ValueError) as exc:
        raise RawProbeCaptureError("sanitized raw artifact cannot be persisted") from exc


def capture_and_publish_raw_probe_store(
    lease: object,
    canonical_tables: object,
) -> RawProbeCaptureSummary:
    """Capture all authorized probes, then atomically publish the complete store.

    The publisher is intentionally unreachable until all 240 probes, parity
    checks, and a full post-capture capability recheck have succeeded.  No
    partial artifact can therefore become a raw-store authority.
    """

    active_lease, active_tables, keys = _require_capabilities(lease, canonical_tables)
    persisted: list[PersistedRawProbeArtifact] = []
    attempts = 0
    resets = 0
    wall_seconds = 0.0

    try:
        for key in keys:
            environment = _environment_for(key)
            evidence = discover_affordances(
                environment,
                task_id=key.task_id,
                forbidden_aliases=_forbidden_aliases(environment),
                seed=key.probe_seed,
                action_cap=_ACTIONS_PER_ARTIFACT,
                target_samples_per_alias=_PROBE_TARGET_SAMPLES_PER_ALIAS,
                actions_per_attempt=_ACTIONS_PER_ATTEMPT,
            )
            sanitized = sanitize_probe_evidence(
                evidence,
                local_affordance_protocol_sha256=key.local_affordance_protocol_sha256,
                development_protocol_sha256=key.development_protocol_sha256,
                development_tasks_sha256=key.development_tasks_sha256,
                phase3_evidence_lock_sha256=key.phase3_evidence_lock_sha256,
                probe_policy_sha256=key.probe_policy_sha256,
                family_id=key.family_id,
                replicate=key.replicate,
                task_index=key.task_index,
                task_id=key.task_id,
                generator_seed=key.generator_seed,
                probe_seed=key.probe_seed,
                environment_seed=key.environment_seed,
                canonical_affordances=active_tables.table_for(key),
            )
            persisted_artifact = _persist(sanitized)
            if persisted_artifact.key != key:
                raise RawProbeCaptureError("persisted raw artifact key differs from authority")
            accounting = evidence.accounting
            if (
                type(accounting.actions) is not int
                or accounting.actions != _ACTIONS_PER_ARTIFACT
                or type(accounting.attempts) is not int
                or type(accounting.resets) is not int
                or accounting.attempts < 0
                or accounting.resets < 0
                or not math.isfinite(accounting.wall_seconds)
                or accounting.wall_seconds < 0.0
            ):
                raise RawProbeCaptureError("probe accounting differs from the frozen budget")
            attempts += accounting.attempts
            resets += accounting.resets
            wall_seconds += accounting.wall_seconds
            persisted.append(persisted_artifact)
    except RawProbeCaptureError:
        raise
    except (LocalAffordanceEvidenceError, RuntimeError, TypeError, ValueError) as exc:
        raise RawProbeCaptureError("raw probe capture failed before publication") from exc

    if len(persisted) != _RAW_ARTIFACT_COUNT:
        raise RawProbeCaptureError("raw probe capture did not retain all 240 artifacts")
    if tuple(artifact.key for artifact in persisted) != keys:
        raise RawProbeCaptureError("raw probe artifact ordering differs from frozen authority")
    if len({artifact.manifest.artifact_id for artifact in persisted}) != _RAW_ARTIFACT_COUNT:
        raise RawProbeCaptureError("raw probe artifact identities are not unique")

    try:
        active_lease.require_active()
        active_tables.require_active()
        if getattr(active_tables, "_lease", None) is not active_lease:
            raise RawProbeCaptureError("canonical tables changed capability binding")
        publish_raw_probe_store_from_readiness(active_lease, artifacts=tuple(persisted))
    except RawProbeCaptureError:
        raise
    except (
        CanonicalPooledTableError,
        LocalAffordanceReadinessError,
        RawProbeAuthorityError,
        RawProbePublicationError,
        AttributeError,
        TypeError,
        ValueError,
    ) as exc:
        raise RawProbeCaptureError("raw probe publication readiness changed") from exc

    return RawProbeCaptureSummary(
        manifest_id=active_lease.authority.manifest.manifest_id,
        activation_git_commit=active_lease.git_commit_sha,
        ordered_key_ids=tuple(key.key_id for key in keys),
        ordered_artifact_ids=tuple(artifact.manifest.artifact_id for artifact in persisted),
        physical_probe_actions=_RAW_ARTIFACT_COUNT * _ACTIONS_PER_ARTIFACT,
        logical_consumer_equivalent_actions=(
            _RAW_ARTIFACT_COUNT
            * _ACTIONS_PER_ARTIFACT
            * _LOGICAL_CONSUMERS_PER_PHYSICAL_ACTION
        ),
        probe_attempts=attempts,
        probe_resets=resets,
        probe_wall_seconds=wall_seconds,
    )


__all__ = [
    "RawProbeCaptureError",
    "RawProbeCaptureSummary",
    "capture_and_publish_raw_probe_store",
]
