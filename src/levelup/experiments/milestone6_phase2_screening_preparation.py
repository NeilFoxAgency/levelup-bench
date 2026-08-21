"""Concrete development-screening keys, shared plans, and data materialization.

This module binds the frozen logical lineage plan to content-addressed evidence, view,
and model keys.  Its explicit data-materialization entry point performs paid probes and
reference replay on development training tasks only.  It performs no model training,
held-out execution, aggregation, or selection, and it never loads final-family material.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from levelup.experiments.milestone6_baselines import (
    build_clean_optimum_training_sample,
)
from levelup.experiments.milestone6_phase2 import (
    _forbidden_aliases,
    _training_probe_seed,
    reconstruct_development_training_task,
)
from levelup.experiments.milestone6_phase2_screening import (
    B1,
    B2,
    LEARNED_BASES,
    C,
    _authority_snapshot,
    build_screening_artifact_slots,
    validate_screening_child_config,
)
from levelup.experiments.runner.config import (
    ExperimentConfig,
    canonical_json_bytes,
    run_id_for,
    scientific_config_sha256,
)
from levelup.experiments.runner.records import (
    ExpectedSharedArtifacts,
    ExpectedUnits,
    PhaseAccounting,
    PlannedSharedArtifact,
    PlannedUnit,
    SystemProvenance,
    TrainingPreparationAccounting,
)
from levelup.experiments.runner.storage import (
    expected_units_sha256,
    plan_expected_units,
    provenance_identity_sha256,
)
from levelup.experiments.runner.training_artifacts import TrainingArtifactKey
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataArtifactError,
    TrainingDataArtifactKey,
    TrainingDataArtifactManifest,
    TrainingDataEvidenceKey,
    TrainingDataEvidenceManifest,
    evidence_key_for,
    load_training_data_artifact,
    load_training_data_evidence,
    load_training_data_evidence_cost,
    load_training_data_view_cost,
    sanitize_clean_optimum_samples,
    write_training_data_artifact,
)

_REPRESENTATIONS = {
    B1: "global-affordance-optimum-frequency-v1",
    B2: "global-affordance-listwise-optimum-v1",
    C: "state-conditioned-listwise-optimum-v1",
}
_OBJECTIVES = {
    B1: "optimum_frequency",
    B2: "listwise_optimum",
    C: "listwise_optimum",
}
_MODEL_IDENTITIES = {
    B1: (
        "global-affordance-mlp-frequency-v1",
        "global_affordance_mlp_frequency_v1",
    ),
    B2: (
        "global-affordance-mlp-listwise-v1",
        "global_affordance_mlp_listwise_v1",
    ),
    C: (
        "state-affordance-mlp-listwise-v1",
        "state_conditioned_mlp_listwise_v1",
    ),
}
PreparationEvent = Callable[[str], None]
LoadedScreeningReplicate = tuple[
    TrainingDataEvidenceManifest,
    dict[tuple[str, int], TrainingDataArtifactManifest],
    str,
    dict[tuple[str, int], str],
]


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class ScreeningDataKeys:
    """Five evidence keys and fifteen representation-view keys for one fold."""

    provenance: SystemProvenance
    evidence: dict[int, TrainingDataEvidenceKey]
    views: dict[tuple[str, int], TrainingDataArtifactKey]


@dataclass(frozen=True, slots=True)
class ScreeningDataManifests:
    """Materialized manifests supplied by the typed artifact loaders for one fold."""

    evidence: dict[int, TrainingDataEvidenceManifest]
    views: dict[tuple[str, int], TrainingDataArtifactManifest]


@dataclass(frozen=True, slots=True)
class ScreeningModelKeys:
    """Sixty temperature-independent model keys for one fold."""

    models: dict[tuple[str, str, int], TrainingArtifactKey]


@dataclass(frozen=True, slots=True)
class MaterializedScreeningData:
    """Fully reloaded 5+15 data inventory for one screening fold."""

    manifests: ScreeningDataManifests
    evidence_cost_ids: dict[int, str]
    view_cost_ids: dict[tuple[str, int], str]


def _learned_condition_ids(config: ExperimentConfig, base: str) -> tuple[str, ...]:
    return tuple(
        condition.condition_id
        for condition in config.conditions
        if condition.parameters.get("base_condition_id") == base
    )


def _representative_units(
    config: ExperimentConfig,
    expected: ExpectedUnits,
    *,
    condition_ids: tuple[str, ...],
    replicate: int,
) -> tuple[PlannedUnit, ...]:
    units = tuple(
        unit
        for unit in expected.units
        if unit.key.condition_id in condition_ids and unit.key.replicate == replicate
    )
    heldout_count = len(config.split.validation_tasks)
    if len(units) != heldout_count * len(condition_ids):
        raise ValueError("screening preparation representative matrix drifted")
    if (
        len({unit.exposure_manifest_sha256 for unit in units}) != 1
        or len({unit.seeds.model_seed for unit in units}) != 1
        or len({unit.seeds.data_order_seed for unit in units}) != 1
    ):
        raise ValueError("screening preparation exposure or paired seeds drifted")
    return units


def _common_key_inputs(
    config: ExperimentConfig,
    expected: ExpectedUnits,
    *,
    provenance_sha256: str,
    replicate: int,
    exposure_sha256: str,
    data_order_seed: int,
    probe_seeds: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "screening_candidates_sha256": str(
            config.parameters["screening_candidates_sha256"]
        ),
        "protocol_sha256": str(config.parameters["development_protocol_sha256"]),
        "task_manifest_sha256": str(
            config.parameters["development_task_manifest_sha256"]
        ),
        "expected_unit_plan_sha256": expected_units_sha256(expected),
        "provenance_sha256": provenance_sha256,
        "reference_exposure_sha256": exposure_sha256,
        "probe_policy_sha256": _digest(
            {
                "builder": "canonical-paid-probe-v1",
                "action_cap": config.parameters["probe_action_cap"],
                "coverage_target": config.parameters[
                    "probe_coverage_target_samples_per_alias"
                ],
                "actions_per_attempt": config.parameters["probe_actions_per_attempt"],
            }
        ),
        "fold_id": str(config.parameters["fold_id"]),
        "heldout_family_id": str(config.parameters["heldout_family_id"]),
        "ordered_training_task_ids": tuple(
            task.task_id for task in config.split.development_tasks
        ),
        "ordered_heldout_task_ids": tuple(
            task.task_id for task in config.split.validation_tasks
        ),
        "replicate": replicate,
        "data_order_seed": data_order_seed,
        "probe_seeds": probe_seeds,
        "environment_seeds": tuple(
            task.environment_reset_seed for task in config.split.development_tasks
        ),
    }


def build_screening_data_keys(
    config: ExperimentConfig,
    provenance: SystemProvenance,
) -> ScreeningDataKeys:
    """Build the exact evidence/view keys for one canonical screening child."""

    provenance = SystemProvenance.model_validate(provenance.model_dump(mode="json"))
    provenance_sha256 = provenance_identity_sha256(provenance)
    validate_screening_child_config(config)
    snapshot = _authority_snapshot()
    expected = plan_expected_units(config)
    protocol = snapshot.protocol
    probe_seeds_by_replicate = {
        replicate: tuple(
            _training_probe_seed(task, replicate=replicate, protocol=protocol)
            for task in config.split.development_tasks
        )
        for replicate in range(config.replicates)
    }
    evidence: dict[int, TrainingDataEvidenceKey] = {}
    views: dict[tuple[str, int], TrainingDataArtifactKey] = {}
    for replicate in range(config.replicates):
        replicate_evidence: TrainingDataEvidenceKey | None = None
        for base in LEARNED_BASES:
            condition_ids = _learned_condition_ids(config, base)
            if len(condition_ids) != 12:
                raise ValueError("screening data key requires twelve variants per base")
            units = _representative_units(
                config,
                expected,
                condition_ids=condition_ids,
                replicate=replicate,
            )
            common = _common_key_inputs(
                config,
                expected,
                provenance_sha256=provenance_sha256,
                replicate=replicate,
                exposure_sha256=units[0].exposure_manifest_sha256,
                data_order_seed=units[0].seeds.data_order_seed,
                probe_seeds=probe_seeds_by_replicate[replicate],
            )
            view = TrainingDataArtifactKey(
                **common,
                representation_sha256=_digest(
                    {
                        "representation_id": _REPRESENTATIONS[base],
                        "learner_visible_only": True,
                    }
                ),
                condition_id=base,
                objective_id=_OBJECTIVES[base],
            )
            view_evidence = evidence_key_for(view)
            if replicate_evidence is None:
                replicate_evidence = view_evidence
            elif view_evidence != replicate_evidence:
                raise ValueError("B1/B2/C views do not share exact evidence identity")
            views[(base, replicate)] = view
        if replicate_evidence is None:
            raise RuntimeError("screening evidence key construction is incomplete")
        evidence[replicate] = replicate_evidence
    if len(evidence) != 5 or len(views) != 15:
        raise ValueError("screening evidence/view inventory drifted")
    if len({key.key_id for key in (*evidence.values(), *views.values())}) != 20:
        raise ValueError("screening evidence/view keys collide")
    return ScreeningDataKeys(provenance=provenance, evidence=evidence, views=views)


def _validate_data_manifests(
    data_keys: ScreeningDataKeys,
    manifests: ScreeningDataManifests,
) -> None:
    if (
        set(manifests.evidence) != set(data_keys.evidence)
        or set(manifests.views) != set(data_keys.views)
    ):
        raise ValueError("screening data-manifest inventory is incomplete or extra")
    for replicate, evidence in manifests.evidence.items():
        evidence = TrainingDataEvidenceManifest.model_validate(
            evidence.model_dump(mode="json")
        )
        expected_key = data_keys.evidence[replicate]
        if evidence.key != expected_key or evidence.evidence_key_id != expected_key.key_id:
            raise ValueError("screening evidence manifest has a foreign key")
    for identity, view in manifests.views.items():
        view = TrainingDataArtifactManifest.model_validate(view.model_dump(mode="json"))
        expected_key = data_keys.views[identity]
        evidence = manifests.evidence[identity[1]]
        if (
            view.key != expected_key
            or view.key_id != expected_key.key_id
            or view.evidence_id != evidence.evidence_id
            or view.payload_sha256 != evidence.payload_sha256
            or view.payload_bytes != evidence.payload_bytes
            or view.sample_task_ids != evidence.sample_task_ids
        ):
            raise ValueError("screening view manifest breaks evidence or key lineage")
    evidence_ids = {item.evidence_id for item in manifests.evidence.values()}
    view_ids = {item.artifact_id for item in manifests.views.values()}
    if len(evidence_ids) != 5 or len(view_ids) != 15 or evidence_ids & view_ids:
        raise ValueError("screening data manifest identities collide")


def _screening_training_batch(
    config: ExperimentConfig,
    *,
    replicate: int,
) -> tuple[Any, TrainingPreparationAccounting]:
    """Build one canonical paid-probe/optimum batch across all 40 training tasks."""

    snapshot = _authority_snapshot()
    condition = next(
        item
        for item in config.conditions
        if item.parameters.get("base_condition_id") == B1
    )
    exposure_by_task = {
        item.task_id: item for item in condition.exposure.exposed_trajectories
    }
    if tuple(condition.exposure.train_task_ids) != tuple(
        task.task_id for task in config.split.development_tasks
    ):
        raise ValueError("screening materialization training order drifted")
    samples = []
    setup_wall = 0.0
    probe_calls = probe_actions = probe_resets = 0
    probe_wall = 0.0
    replay_calls = replay_actions = replay_resets = 0
    replay_wall = 0.0
    for task in config.split.development_tasks:
        setup_started = time.perf_counter()
        reconstruction = reconstruct_development_training_task(
            family=task.family_id,
            task_index=task.task_index,
            generator_seed=task.generator_seed,
            expected_task_id=task.task_id,
        )
        setup_wall += time.perf_counter() - setup_started
        exposure = exposure_by_task[task.task_id]
        trajectory = reconstruction.trajectories.get(exposure.trajectory_id)
        if trajectory is None:
            raise ValueError("screening optimum exposure is absent from reconstruction")
        sample = build_clean_optimum_training_sample(
            reconstruction.environment,
            trajectory,
            task_identity=task,
            exposure=exposure,
            forbidden_aliases=_forbidden_aliases(reconstruction.environment),
            probe_seed=_training_probe_seed(
                task,
                replicate=replicate,
                protocol=snapshot.protocol,
            ),
            probe_action_cap=int(config.parameters["probe_action_cap"]),
            target_samples_per_alias=int(
                config.parameters["probe_coverage_target_samples_per_alias"]
            ),
            probe_actions_per_attempt=int(config.parameters["probe_actions_per_attempt"]),
        )
        samples.append(sample)
        probe_calls += sample.probe.accounting.attempts
        probe_actions += sample.probe.accounting.actions
        probe_resets += sample.probe.accounting.resets
        probe_wall += sample.probe.accounting.wall_seconds
        replay_calls += sample.reference.evaluator_calls
        replay_actions += (
            sample.reference.evaluator_replay_actions
            + sample.reference.observable_replay_actions
        )
        replay_resets += sample.reference.resets
        replay_wall += (
            sample.reference.evaluator_wall_seconds
            + sample.reference.observable_replay_wall_seconds
        )
    sanitized = sanitize_clean_optimum_samples(tuple(samples))
    return sanitized, TrainingPreparationAccounting(
        setup=PhaseAccounting(calls=len(samples), wall_seconds=setup_wall),
        training_probes=PhaseAccounting(
            calls=probe_calls,
            actions=probe_actions,
            environment_steps=probe_actions,
            resets=probe_resets,
            wall_seconds=probe_wall,
        ),
        reference_replay=PhaseAccounting(
            calls=replay_calls,
            actions=replay_actions,
            environment_steps=replay_actions,
            resets=replay_resets,
            wall_seconds=replay_wall,
        ),
        serialization=PhaseAccounting(calls=1),
    )


def _intent_body(
    config: ExperimentConfig,
    data_keys: ScreeningDataKeys,
    replicate: int,
) -> dict[str, Any]:
    return {
        "schema_version": "milestone6.screening-data-intent.v1",
        "run_id": run_id_for(config),
        "config_sha256": scientific_config_sha256(config),
        "replicate": replicate,
        "evidence_key_id": data_keys.evidence[replicate].key_id,
        "view_key_ids": [
            data_keys.views[(base, replicate)].key_id for base in LEARNED_BASES
        ],
    }


def _intent_path(run_dir: Path, evidence_key_id: str) -> Path:
    return run_dir / "screening-data-intents" / f"{evidence_key_id}.json"


def _claim_materialization_intent(path: Path, body: dict[str, Any]) -> bool:
    """Publish one intent atomically and report whether this caller won the claim."""

    payload = canonical_json_bytes(body)
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file():
            raise TrainingDataArtifactError(
                "screening data intent must be a regular non-symlink file"
            )
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise TrainingDataArtifactError("cannot read screening data intent") from exc
        if observed != payload:
            raise TrainingDataArtifactError("screening data intent conflicts")
        return False
    if path.parent.is_symlink() or path.parent.parent.is_symlink():
        raise TrainingDataArtifactError("screening data intent path cannot be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise TrainingDataArtifactError("screening data intent directory is unsafe")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file():
                raise TrainingDataArtifactError(
                    "concurrent screening data intent claim is unsafe"
                )
            try:
                observed = path.read_bytes()
            except OSError as exc:
                raise TrainingDataArtifactError(
                    "cannot read concurrent screening data intent"
                ) from exc
            if observed != payload:
                raise TrainingDataArtifactError(
                    "concurrent screening data intent conflicts"
                )
            return False
        return True
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _load_screening_data_replicate(
    run_dir: Path,
    data_keys: ScreeningDataKeys,
    replicate: int,
) -> tuple[
    TrainingDataEvidenceManifest,
    dict[tuple[str, int], TrainingDataArtifactManifest],
    str,
    dict[tuple[str, int], str],
]:
    evidence_key = data_keys.evidence[replicate]
    evidence_cost = load_training_data_evidence_cost(run_dir, evidence_key)
    evidence_manifest, _ = load_training_data_evidence(
        run_dir,
        evidence_cost.artifact_id,
        expected_key=evidence_key,
    )
    views: dict[tuple[str, int], TrainingDataArtifactManifest] = {}
    view_cost_ids: dict[tuple[str, int], str] = {}
    for base in LEARNED_BASES:
        identity = (base, replicate)
        key = data_keys.views[identity]
        manifest, _ = load_training_data_artifact(run_dir, expected_key=key)
        cost = load_training_data_view_cost(run_dir, key)
        if (
            cost.artifact_id != manifest.artifact_id
            or manifest.evidence_id != evidence_manifest.evidence_id
            or manifest.payload_sha256 != evidence_manifest.payload_sha256
            or manifest.payload_bytes != evidence_manifest.payload_bytes
        ):
            raise TrainingDataArtifactError("screening data replicate lineage drifted")
        views[identity] = manifest
        view_cost_ids[identity] = cost.cost_id
    return evidence_manifest, views, evidence_cost.cost_id, view_cost_ids


def _require_namespace_inventory(
    run_dir: Path,
    expected: dict[str, tuple[set[str], str]],
) -> None:
    for name, (expected_names, entry_kind) in expected.items():
        directory = run_dir / name
        if not expected_names:
            if os.path.lexists(directory):
                raise TrainingDataArtifactError(
                    "screening data has an unclaimed or partial namespace"
                )
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise TrainingDataArtifactError("screening data namespace is missing or unsafe")
        entries = tuple(directory.iterdir())
        if {entry.name for entry in entries} != expected_names:
            raise TrainingDataArtifactError("screening data namespace inventory drifted")
        for entry in entries:
            if entry.is_symlink() or (
                entry_kind == "file" and not entry.is_file()
            ) or (entry_kind == "directory" and not entry.is_dir()):
                raise TrainingDataArtifactError(
                    "screening data namespace contains an unsafe entry"
                )


def _preflight_existing_data_namespace(
    run_dir: Path,
    config: ExperimentConfig,
    data_keys: ScreeningDataKeys,
) -> dict[int, LoadedScreeningReplicate]:
    """Load every claimed replicate and reject all other namespace state."""

    expected_intents = {
        f"{key.key_id}.json": replicate
        for replicate, key in data_keys.evidence.items()
    }
    intent_root = run_dir / "screening-data-intents"
    claimed: dict[int, Path] = {}
    if os.path.lexists(intent_root):
        if intent_root.is_symlink() or not intent_root.is_dir():
            raise TrainingDataArtifactError("screening data intent namespace is unsafe")
        for entry in intent_root.iterdir():
            replicate = expected_intents.get(entry.name)
            if (
                replicate is None
                or entry.is_symlink()
                or not entry.is_file()
            ):
                raise TrainingDataArtifactError(
                    "screening data intent namespace contains an unexpected entry"
                )
            claimed[replicate] = entry

    loaded: dict[int, LoadedScreeningReplicate] = {}
    for replicate in sorted(claimed):
        _claim_materialization_intent(
            claimed[replicate],
            _intent_body(config, data_keys, replicate),
        )
        loaded[replicate] = _load_screening_data_replicate(
            run_dir,
            data_keys,
            replicate,
        )

    expected_evidence_ids = {
        item[0].evidence_id for item in loaded.values()
    }
    expected_view_ids = {
        manifest.artifact_id
        for item in loaded.values()
        for manifest in item[1].values()
    }
    claimed_replicates = set(loaded)
    _require_namespace_inventory(
        run_dir,
        {
            "screening-data-intents": (
                {
                    f"{data_keys.evidence[replicate].key_id}.json"
                    for replicate in claimed_replicates
                },
                "file",
            ),
            "training-data-evidence-costs": (
                {
                    f"{data_keys.evidence[replicate].key_id}.json"
                    for replicate in claimed_replicates
                },
                "file",
            ),
            "training-data-view-costs": (
                {
                    f"{data_keys.views[(base, replicate)].key_id}.json"
                    for replicate in claimed_replicates
                    for base in LEARNED_BASES
                },
                "file",
            ),
            "training-data-artifact-keys": (
                {
                    f"{data_keys.views[(base, replicate)].key_id}.json"
                    for replicate in claimed_replicates
                    for base in LEARNED_BASES
                },
                "file",
            ),
            "training-data-evidence": (expected_evidence_ids, "directory"),
            "training-data-artifacts": (expected_view_ids, "directory"),
        },
    )
    return loaded


def _prepare_screening_data_replicate(
    run_dir: Path,
    config: ExperimentConfig,
    data_keys: ScreeningDataKeys,
    replicate: int,
    *,
    event: PreparationEvent | None = None,
) -> tuple[
    TrainingDataEvidenceManifest,
    dict[tuple[str, int], TrainingDataArtifactManifest],
    str,
    dict[tuple[str, int], str],
]:
    """Materialize one replicate once; any interrupted/corrupt state fails closed."""

    if replicate not in data_keys.evidence:
        raise ValueError("unknown screening data replicate")
    loaded_replicates = _preflight_existing_data_namespace(run_dir, config, data_keys)
    if replicate in loaded_replicates:
        if event is not None:
            event(f"replicate_loaded:{replicate}")
        return loaded_replicates[replicate]
    intent = _intent_path(run_dir, data_keys.evidence[replicate].key_id)
    if not _claim_materialization_intent(
        intent,
        _intent_body(config, data_keys, replicate),
    ):
        raise TrainingDataArtifactError(
            "screening data replicate was claimed concurrently; retry only after completion"
        )
    if event is not None:
        event(f"replicate_build:{replicate}")
    try:
        sanitized, evidence_accounting = _screening_training_batch(
            config,
            replicate=replicate,
        )
        view_accounting = TrainingPreparationAccounting(
            serialization=PhaseAccounting(calls=1)
        )
        for base in LEARNED_BASES:
            write_training_data_artifact(
                run_dir,
                data_keys.views[(base, replicate)],
                sanitized,
                evidence_accounting=evidence_accounting,
                view_accounting=view_accounting,
            )
    except Exception as exc:
        raise TrainingDataArtifactError(
            "screening data materialization interrupted; intent remains fail-closed"
        ) from exc
    completed = _preflight_existing_data_namespace(run_dir, config, data_keys)
    try:
        return completed[replicate]
    except KeyError as exc:
        raise TrainingDataArtifactError(
            "screening data materialization did not publish its claimed replicate"
        ) from exc


def _require_exact_data_namespace(
    run_dir: Path,
    config: ExperimentConfig,
    data_keys: ScreeningDataKeys,
    manifests: ScreeningDataManifests,
) -> None:
    loaded = _preflight_existing_data_namespace(run_dir, config, data_keys)
    if set(loaded) != set(data_keys.evidence):
        raise TrainingDataArtifactError("screening data namespace is incomplete")
    observed = ScreeningDataManifests(
        evidence={replicate: item[0] for replicate, item in loaded.items()},
        views={
            identity: manifest
            for item in loaded.values()
            for identity, manifest in item[1].items()
        },
    )
    if observed != manifests:
        raise TrainingDataArtifactError("screening data namespace reload drifted")


def materialize_screening_data(
    config: ExperimentConfig,
    data_keys: ScreeningDataKeys,
    output_root: str | Path,
    *,
    event: PreparationEvent | None = None,
) -> MaterializedScreeningData:
    """Materialize and fully reload the exact 5-evidence/15-view child inventory."""

    if data_keys != build_screening_data_keys(config, data_keys.provenance):
        raise ValueError("screening materialization received noncanonical data keys")
    run_dir = Path(output_root)
    if run_dir.is_symlink():
        raise TrainingDataArtifactError("screening materialization root cannot be a symlink")
    evidence: dict[int, TrainingDataEvidenceManifest] = {}
    views: dict[tuple[str, int], TrainingDataArtifactManifest] = {}
    evidence_cost_ids: dict[int, str] = {}
    view_cost_ids: dict[tuple[str, int], str] = {}
    for replicate in range(config.replicates):
        (
            evidence_manifest,
            replicate_views,
            evidence_cost_id,
            replicate_view_cost_ids,
        ) = _prepare_screening_data_replicate(
            run_dir,
            config,
            data_keys,
            replicate,
            event=event,
        )
        evidence[replicate] = evidence_manifest
        views.update(replicate_views)
        evidence_cost_ids[replicate] = evidence_cost_id
        view_cost_ids.update(replicate_view_cost_ids)
    manifests = ScreeningDataManifests(evidence=evidence, views=views)
    _validate_data_manifests(data_keys, manifests)
    _require_exact_data_namespace(run_dir, config, data_keys, manifests)
    return MaterializedScreeningData(
        manifests=manifests,
        evidence_cost_ids=evidence_cost_ids,
        view_cost_ids=view_cost_ids,
    )


def _training_tuple_conditions(
    config: ExperimentConfig,
    base: str,
    training_tuple_id: str,
) -> tuple[Any, ...]:
    conditions = tuple(
        condition
        for condition in config.conditions
        if condition.parameters.get("base_condition_id") == base
        and condition.parameters.get("training_tuple_id") == training_tuple_id
    )
    if (
        len(conditions) != 3
        or {condition.parameters.get("search_temperature") for condition in conditions}
        != {0.6, 0.9, 1.2}
        or len({condition.parameters.get("learning_rate") for condition in conditions}) != 1
        or len({condition.parameters.get("training_epochs") for condition in conditions}) != 1
    ):
        raise ValueError("screening training tuple variants drifted")
    return conditions


def build_screening_model_keys(
    config: ExperimentConfig,
    data_keys: ScreeningDataKeys,
    data_manifests: ScreeningDataManifests,
) -> ScreeningModelKeys:
    """Bind materialized view IDs to sixty temperature-independent model keys."""

    provenance_sha256 = provenance_identity_sha256(data_keys.provenance)
    canonical_data = build_screening_data_keys(config, data_keys.provenance)
    if data_keys != canonical_data:
        raise ValueError("screening data keys drifted from canonical construction")
    _validate_data_manifests(data_keys, data_manifests)

    expected = plan_expected_units(config)
    training_tuple_ids = tuple(
        dict.fromkeys(
            str(condition.parameters["training_tuple_id"])
            for condition in config.conditions[2:]
        )
    )
    if training_tuple_ids != (
        "lr0p003-e120",
        "lr0p003-e180",
        "lr0p01-e120",
        "lr0p01-e180",
    ):
        raise ValueError("screening training-tuple universe drifted")
    models: dict[tuple[str, str, int], TrainingArtifactKey] = {}
    for replicate in range(config.replicates):
        for base in LEARNED_BASES:
            data_key = data_keys.views[(base, replicate)]
            learner_id, backbone_id = _MODEL_IDENTITIES[base]
            for training_tuple_id in training_tuple_ids:
                conditions = _training_tuple_conditions(config, base, training_tuple_id)
                condition_ids = tuple(condition.condition_id for condition in conditions)
                units = _representative_units(
                    config,
                    expected,
                    condition_ids=condition_ids,
                    replicate=replicate,
                )
                learning_rate = conditions[0].parameters["learning_rate"]
                training_epochs = conditions[0].parameters["training_epochs"]
                training_config_sha256 = _digest(
                    {
                        "optimizer": config.parameters["optimizer"],
                        "learning_rate": learning_rate,
                        "epochs": training_epochs,
                        "weight_decay": config.parameters["weight_decay"],
                        "data_order": config.parameters["data_order"],
                        "temperature_excluded": True,
                    }
                )
                key = TrainingArtifactKey(
                    screening_candidates_sha256=data_key.screening_candidates_sha256,
                    protocol_sha256=data_key.protocol_sha256,
                    task_manifest_sha256=data_key.task_manifest_sha256,
                    expected_unit_plan_sha256=data_key.expected_unit_plan_sha256,
                    exposure_sha256=units[0].exposure_manifest_sha256,
                    training_data_sha256=data_manifests.views[
                        (base, replicate)
                    ].artifact_id,
                    provenance_sha256=provenance_sha256,
                    fold_id=data_key.fold_id,
                    heldout_family_id=data_key.heldout_family_id,
                    ordered_training_task_ids=data_key.ordered_training_task_ids,
                    ordered_heldout_task_ids=data_key.ordered_heldout_task_ids,
                    condition_id=base,
                    learner_id=learner_id,
                    objective_id=_OBJECTIVES[base],
                    backbone_id=backbone_id,
                    training_tuple_id=training_tuple_id,
                    replicate=replicate,
                    model_seed=units[0].seeds.model_seed,
                    data_order_seed=data_key.data_order_seed,
                    probe_seeds=data_key.probe_seeds,
                    environment_seeds=data_key.environment_seeds,
                    probe_spec_sha256=data_key.probe_policy_sha256,
                    training_config_sha256=training_config_sha256,
                    capacity_spec_sha256=_digest(
                        {
                            "hidden_widths": config.parameters["mlp_hidden_widths"],
                            "backbone_id": backbone_id,
                            "capacity_matching": config.parameters["capacity_matching"],
                        }
                    ),
                )
                if "search_temperature" in key.model_dump(mode="json"):
                    raise RuntimeError("search temperature entered model-key identity")
                models[(base, training_tuple_id, replicate)] = key
    if len(models) != 60 or len({key.key_id for key in models.values()}) != 60:
        raise ValueError("screening model-key inventory drifted or collided")
    return ScreeningModelKeys(models=models)


def build_screening_shared_plan(
    config: ExperimentConfig,
    data_keys: ScreeningDataKeys,
    data_manifests: ScreeningDataManifests,
    model_keys: ScreeningModelKeys,
) -> ExpectedSharedArtifacts:
    """Convert exact logical slots to concrete key IDs for one child RunStore."""

    if data_keys != build_screening_data_keys(config, data_keys.provenance):
        raise ValueError("screening shared plan received noncanonical data keys")
    _validate_data_manifests(data_keys, data_manifests)
    if model_keys != build_screening_model_keys(config, data_keys, data_manifests):
        raise ValueError("screening shared plan received noncanonical model keys")

    slots = build_screening_artifact_slots(config)
    artifacts: list[PlannedSharedArtifact] = []
    for slot in slots:
        if slot.kind == "training_data_evidence":
            key_id = data_keys.evidence[slot.replicate].key_id
        elif slot.kind == "training_data_view":
            if slot.base_condition_id is None:
                raise RuntimeError("screening view slot has no base")
            key_id = data_keys.views[(slot.base_condition_id, slot.replicate)].key_id
        else:
            if slot.base_condition_id is None or slot.training_tuple_id is None:
                raise RuntimeError("screening model slot identity is incomplete")
            key_id = model_keys.models[
                (slot.base_condition_id, slot.training_tuple_id, slot.replicate)
            ].key_id
        artifacts.append(
            PlannedSharedArtifact(
                kind=slot.kind,
                key_id=key_id,
                owner_condition_id=slot.owner_condition_id,
                owner_group_id=slot.owner_group_id,
                owner_family_id=slot.heldout_family,
                owner_fold_id=slot.fold_id,
                owner_replicate=slot.replicate,
                consumer_phase="validation",
                consumer_condition_ids=slot.consumer_condition_ids,
                consumer_unit_ids=slot.consumer_unit_ids,
            )
        )
    shared = ExpectedSharedArtifacts(
        run_id=run_id_for(config),
        config_sha256=scientific_config_sha256(config),
        artifacts=tuple(artifacts),
    )
    if len(shared.artifacts) != 80 or len({item.key_id for item in shared.artifacts}) != 80:
        raise ValueError("screening shared-plan inventory drifted or collided")
    return shared
