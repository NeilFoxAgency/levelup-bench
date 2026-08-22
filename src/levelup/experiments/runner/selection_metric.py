"""Pure, fail-closed development-selection metric helpers."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from levelup.experiments.runner.config import (
    ExperimentConfig,
    run_id_for,
    scientific_config_sha256,
)
from levelup.experiments.runner.records import (
    ExpectedSharedArtifacts,
    ExpectedUnits,
    PhaseAccounting,
    UnitKey,
    UnitRecord,
    UnitSeeds,
)

METRIC_ID = "total_adaptation_actions_to_first_exact_optimum"
METRIC_SCHEMA_VERSION = "restricted-interactions.v1"
ACTION_FORMULA = "accounting.probes.actions + accounting.search.actions"
ORACLE_POLICY = "fixed_batch_then_independent_replay_then_reporting_only_oracle"
_FROZEN_FAMILY_ORDER = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
_FROZEN_AUTHORITY_FAMILY_IDS_SHA256 = hashlib.sha256(
    json.dumps(_FROZEN_FAMILY_ORDER, separators=(",", ":"), ensure_ascii=True).encode()
).hexdigest()
_FROZEN_SCREENING_REPLICATES = (0, 1, 2, 3, 4)
_FROZEN_PROTOCOL_SHA256 = "7e6911c120db091e2b250f7a91520dd5f81a481cb4a19662eeae858c7da1c059"
_FROZEN_SCREENING_SHA256 = "f3c3b4c239df54de4ed5c675f21a846253102e54d822023d782c941542c19f69"
_FROZEN_TASK_MANIFEST_SHA256 = (
    "20f6606bd2150d808b18f011976bbf7c8298627e1cc01eeb67f653eacba9731f"
)
_AUTHORITY_CONSTRUCTION_TOKEN = object()
_SPEC_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class SelectionAuthority:
    """Metric authority loaded from the actual frozen protocol files."""

    protocol_path: Path
    screening_candidates_path: Path
    task_manifest_path: Path
    protocol_sha256: str
    screening_candidates_sha256: str
    task_manifest_sha256: str
    endpoint: int
    failure_sentinel: int
    family_ids: tuple[str, ...]
    screening_replicates: tuple[int, ...]
    heldout_task_ids: tuple[tuple[str, tuple[str, ...]], ...]
    training_core_task_identities: tuple[
        tuple[str, tuple[tuple[str, int, int, int], ...]], ...
    ]
    # Retained immutable bytes let a pinned runtime revalidate the authority
    # without reopening any source path after load.
    protocol_bytes: bytes = b""
    screening_candidates_bytes: bytes = b""
    task_manifest_bytes: bytes = b""
    _construction_token: InitVar[object | None] = None

    def __post_init__(self, _construction_token: object | None) -> None:
        if _construction_token is not _AUTHORITY_CONSTRUCTION_TOKEN:
            raise ValueError("selection authority must be loaded from frozen files")

    @property
    def source_bytes(self) -> dict[str, bytes]:
        """Return the immutable authority bytes retained at load time."""

        if not all(
            isinstance(value, bytes)
            for value in (
                self.protocol_bytes,
                self.screening_candidates_bytes,
                self.task_manifest_bytes,
            )
        ) or not all(
            (value for value in (
                self.protocol_bytes,
                self.screening_candidates_bytes,
                self.task_manifest_bytes,
            ))
        ):
            raise ValueError("selection authority does not retain all source bytes")
        return {
            "protocol": self.protocol_bytes,
            "screening_candidates": self.screening_candidates_bytes,
            "task_manifest": self.task_manifest_bytes,
        }


@dataclass(frozen=True, slots=True)
class ExpectedSelectionUnit:
    """One exact planned unit together with its child-run identity."""

    run_id: str
    config_sha256: str
    unit_id: str
    key: UnitKey
    seeds: UnitSeeds
    exposure_manifest_sha256: str
    shared_key_ids: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class SelectionMetricSpec:
    """Frozen metric semantics and complete expected matrix for one variant."""

    condition_id: str
    phase: str
    endpoint: int
    failure_sentinel: int
    expected_units: tuple[ExpectedSelectionUnit, ...]
    protocol_sha256: str
    screening_candidates_sha256: str
    task_manifest_sha256: str
    family_universe: tuple[str, ...]
    # Explicitly retain the authority universe copied by the frozen builder.
    # This prevents a merged child spec from silently substituting a private
    # family universe for the one loaded from the protocol files.
    authority_family_ids: tuple[str, ...] = ()
    authority_family_ids_sha256: str = ""
    metric_id: str = METRIC_ID
    schema_version: str = METRIC_SCHEMA_VERSION
    action_formula: str = ACTION_FORMULA
    oracle_policy: str = ORACLE_POLICY
    require_shared_preparation: bool = True
    require_candidate_generation_identity: bool = False
    require_zero_local_training: bool = False
    _construction_token: InitVar[object | None] = None

    def __post_init__(self, _construction_token: object | None) -> None:
        if _construction_token is not _SPEC_CONSTRUCTION_TOKEN:
            raise ValueError("selection metric specs require the authority-bound builder")
        if not self.condition_id:
            raise ValueError("selection condition ID cannot be empty")
        if self.phase not in {"development", "validation"}:
            raise ValueError("selection phase must be development or validation, never final")
        if self.endpoint < 1 or self.failure_sentinel != self.endpoint + 1:
            raise ValueError("selection sentinel must equal endpoint plus one")
        for label, digest in (
            ("protocol", self.protocol_sha256),
            ("screening candidates", self.screening_candidates_sha256),
            ("task manifest", self.task_manifest_sha256),
        ):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"selection {label} SHA-256 is invalid")
        if (
            self.metric_id != METRIC_ID
            or self.schema_version != METRIC_SCHEMA_VERSION
            or self.action_formula != ACTION_FORMULA
            or self.oracle_policy != ORACLE_POLICY
        ):
            raise ValueError("selection metric semantics drifted")
        if not self.expected_units:
            raise ValueError("selection metric requires a complete expected unit plan")
        if (
            not self.family_universe
            or len(self.family_universe) != len(set(self.family_universe))
        ):
            raise ValueError("selection family universe is invalid")
        if self.authority_family_ids:
            if (
                self.authority_family_ids != self.family_universe
                or self.authority_family_ids != _FROZEN_FAMILY_ORDER
                or len(self.authority_family_ids) != len(set(self.authority_family_ids))
                or self.authority_family_ids_sha256
                != _FROZEN_AUTHORITY_FAMILY_IDS_SHA256
            ):
                raise ValueError("selection authority family universe is inconsistent")
        identities = [(unit.run_id, unit.unit_id) for unit in self.expected_units]
        if len(identities) != len(set(identities)):
            raise ValueError("selection expected units contain duplicate identities")
        keys = [unit.key.model_dump_json() for unit in self.expected_units]
        if len(keys) != len(set(keys)):
            raise ValueError("selection expected units contain duplicate scientific keys")
        if any(
            unit.key.condition_id != self.condition_id or unit.key.phase != self.phase
            for unit in self.expected_units
        ):
            raise ValueError("selection expected units drift from condition or phase")
        if not self.family_ids <= frozenset(self.family_universe):
            raise ValueError("selection expected units exceed the frozen family universe")
        if self.require_shared_preparation and any(
            {kind for kind, _ in unit.shared_key_ids}
            != {"training_data_evidence", "training_data_view", "training_artifact"}
            for unit in self.expected_units
        ):
            raise ValueError("selection expected units require complete typed shared keys")

    @property
    def family_ids(self) -> frozenset[str]:
        return frozenset(unit.key.family_id for unit in self.expected_units)

    @property
    def has_complete_family_coverage(self) -> bool:
        return self.family_ids == frozenset(self.family_universe)


@dataclass(frozen=True, slots=True)
class FamilySelectionSummary:
    """Equal-weight input produced after aggregation within one family."""

    family_id: str
    units: int
    exact_optimum_success_rate: float
    median_restricted_interactions: float


@dataclass(frozen=True, slots=True)
class VariantSelectionSummary:
    """Frozen selection quantities for one numeric condition variant."""

    condition_id: str
    endpoint: int
    failure_sentinel: int
    families: tuple[FamilySelectionSummary, ...]
    minimum_family_exact_optimum_success_rate: float
    worst_family_median_restricted_interactions: float
    macro_average_family_median_restricted_interactions: float


def load_selection_authority(
    protocol_path: Path,
    screening_candidates_path: Path,
    task_manifest_path: Path,
    *,
    source_bytes: Mapping[str, bytes] | None = None,
) -> SelectionAuthority:
    """Load and cross-check the actual frozen development-selection files."""

    if source_bytes is not None and set(source_bytes) != {
        "protocol",
        "screening_candidates",
        "task_manifest",
    }:
        raise ValueError("authority source bytes must contain exactly the three frozen sources")
    protocol_bytes = (
        protocol_path.read_bytes()
        if source_bytes is None
        else bytes(source_bytes["protocol"])
    )
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes)
    screening_bytes = (
        screening_candidates_path.read_bytes()
        if source_bytes is None
        else bytes(source_bytes["screening_candidates"])
    )
    screening_sha256 = hashlib.sha256(screening_bytes).hexdigest()
    screening = json.loads(screening_bytes)
    task_manifest_bytes = (
        task_manifest_path.read_bytes()
        if source_bytes is None
        else bytes(source_bytes["task_manifest"])
    )
    task_manifest_sha256 = hashlib.sha256(task_manifest_bytes).hexdigest()
    task_manifest = json.loads(task_manifest_bytes)
    if not all(isinstance(payload, dict) for payload in (protocol, screening, task_manifest)):
        raise ValueError("frozen selection authority files must contain JSON objects")
    if (
        protocol_sha256 != _FROZEN_PROTOCOL_SHA256
        or screening_sha256 != _FROZEN_SCREENING_SHA256
        or task_manifest_sha256 != _FROZEN_TASK_MANIFEST_SHA256
        or protocol.get("schema_version") != "milestone6.development_protocol.v2"
        or protocol.get("status") != "frozen-before-comparative-development-results"
        or protocol.get("scope") != "known-development-only"
        or protocol.get("final_family_access") != "forbidden_until_phase9_method_freeze"
        or tuple(protocol.get("family_order", ())) != _FROZEN_FAMILY_ORDER
        or protocol.get("task_manifest") != "configs/milestone6/development_tasks.json"
        or screening.get("schema_version")
        != "milestone6.phase2_screening_candidates.v2"
        or screening.get("status") != "frozen-before-screening-results"
        or screening.get("scope") != "known-development-only"
        or screening.get("final_family_access") is not False
        or screening.get("parent_protocol", {}).get("sha256") != protocol_sha256
        or screening.get("parent_protocol", {}).get("path")
        != "configs/milestone6/development_protocol.json"
        or screening.get("task_manifest", {}).get("sha256") != task_manifest_sha256
        or screening.get("task_manifest", {}).get("path")
        != "configs/milestone6/development_tasks.json"
    ):
        raise ValueError("frozen selection authority files are inconsistent")

    protocol_folds = protocol.get("folds", {})
    protocol_seed_policy = protocol.get("seed_policy", {})
    screening_folds = screening.get("folds", {})
    screening_replicates = tuple(screening_folds.get("replicates", ()))
    protocol_family_order = tuple(protocol.get("family_order", ()))
    screening_family_order = tuple(screening_folds.get("family_order", ()))
    manifest_family_order = tuple(task_manifest.get("family_order", ()))
    if (
        protocol_folds.get("kind") != "leave-one-family-out"
        or protocol_folds.get("training_tasks_per_nonheld_family") != 8
        or protocol_folds.get("screening_heldout_tasks_per_family") != 8
        or tuple(protocol_seed_policy.get("screening_replicates", ()))
        != _FROZEN_SCREENING_REPLICATES
        or protocol_family_order != _FROZEN_FAMILY_ORDER
        or screening_family_order != protocol_family_order
        or manifest_family_order != protocol_family_order
        or screening_folds.get("kind") != "leave-one-family-out"
        or screening_replicates != _FROZEN_SCREENING_REPLICATES
        or len(set(screening_replicates)) != len(screening_replicates)
        or screening_folds.get("training_tasks")
        != "eight training_core tasks from each of the five non-held-out families"
        or screening_folds.get("heldout_tasks")
        != "eight training_core tasks from the held-out development family"
    ):
        raise ValueError("frozen leave-one-family-out authority drifted")

    allowed_roles = {
        "training_core",
        "known_development",
        "historical_milestone5_development",
    }
    tasks = task_manifest.get("tasks", ())
    if (
        task_manifest.get("schema_version") != "milestone6.development_tasks.v1"
        or task_manifest.get("purpose")
        != "Phase 2 known-family development manifest; contains no validation or final tasks."
        or tuple(task_manifest.get("family_order", ())) != _FROZEN_FAMILY_ORDER
        or set(task_manifest.get("role_metadata", {})) != allowed_roles
        or not isinstance(tasks, list)
        or not tasks
        or task_manifest.get("generator_seeds")
        != {
            "plain": 900,
            "battery": 1000,
            "cooldown": 1100,
            "heat": 1200,
            "momentum": 1300,
            "combo": 2026,
        }
        or task_manifest.get("environment_reset_seed") != 0
    ):
        raise ValueError("development task-manifest authority drifted")
    if any(not isinstance(task, dict) for task in tasks):
        raise ValueError("development task manifest contains a malformed task")
    task_ids = [task.get("task_id") for task in tasks]
    if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
        raise ValueError("development task manifest contains an invalid task ID")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("development task manifest contains duplicate task IDs")
    heldout_task_ids: list[tuple[str, tuple[str, ...]]] = []
    training_core_task_identities: list[
        tuple[str, tuple[tuple[str, int, int, int], ...]]
    ] = []
    for family_id in _FROZEN_FAMILY_ORDER:
        family_tasks = [task for task in tasks if task.get("family") == family_id]
        training_core = tuple(
            (
                task["task_id"],
                int(task["task_index"]),
                int(task["generator_seed"]),
                int(task["environment_reset_seed"]),
            )
            for task in family_tasks
            if "training_core" in task.get("roles", ())
        )
        if len(training_core) != 8 or len(set(training_core)) != 8:
            raise ValueError("development task manifest training-core coverage drifted")
        ordered_core = tuple(sorted(training_core))
        heldout_task_ids.append(
            (family_id, tuple(identity[0] for identity in ordered_core))
        )
        training_core_task_identities.append((family_id, ordered_core))
    for task in tasks:
        roles = task.get("roles")
        if (
            task.get("family") not in _FROZEN_FAMILY_ORDER
            or not isinstance(roles, list)
            or not roles
            or any(not isinstance(role, str) for role in roles)
            or not set(roles) <= allowed_roles
            or "known_development" not in roles
            or any(
                token in role.lower()
                for role in roles
                for token in ("final", "validation")
            )
            or not isinstance(task.get("task_index"), int)
            or not isinstance(task.get("generator_seed"), int)
            or not isinstance(task.get("environment_reset_seed"), int)
        ):
            raise ValueError("development task manifest contains a non-development task")

    expected_fixed_controls = [
        {
            "variant_id": "A0-fixed",
            "condition_id": "A0-no-probe-uniform",
            "learner_id": "uniform-visible-actions-v1",
            "probe_action_cap": 0,
            "search_policy": "uniform-visible-actions",
        },
        {
            "variant_id": "A1-fixed",
            "condition_id": "A1-paid-probe-uniform",
            "learner_id": "uniform-visible-actions-v1",
            "probe_action_cap": 64,
            "search_policy": "uniform-visible-actions",
        },
    ]
    expected_learned_conditions = [
        {
            "condition_id": "B1-clean-global-optimum-frequency",
            "learner_id": "global-affordance-mlp-frequency-v1",
            "objective": "optimum_frequency",
            "backbone": "global_affordance_mlp",
            "candidate_tuple_ids": "all",
        },
        {
            "condition_id": "B2-global-listwise-optimum",
            "learner_id": "global-affordance-mlp-listwise-v1",
            "objective": "listwise_optimum",
            "backbone": "global_affordance_mlp",
            "candidate_tuple_ids": "all",
        },
        {
            "condition_id": "C-state-conditioned-listwise-optimum",
            "learner_id": "state-affordance-mlp-listwise-v1",
            "objective": "listwise_optimum",
            "backbone": "state_affordance_mlp",
            "candidate_tuple_ids": "all",
        },
    ]
    candidates = screening.get("candidate_tuples", ())
    expected_candidates = [
        {
            "tuple_id": f"{learning_rate_id}-e{epochs}-t{temperature_id}",
            "training_tuple_id": f"{learning_rate_id}-e{epochs}",
            "learning_rate": learning_rate,
            "training_epochs": epochs,
            "search_temperature": temperature,
        }
        for learning_rate, learning_rate_id in ((0.003, "lr0p003"), (0.01, "lr0p01"))
        for epochs in (120, 180)
        for temperature, temperature_id in ((0.6, "0p6"), (0.9, "0p9"), (1.2, "1p2"))
    ]
    if (
        screening.get("fixed_controls") != expected_fixed_controls
        or screening.get("learned_conditions") != expected_learned_conditions
        or candidates != expected_candidates
    ):
        raise ValueError("frozen screening candidate matrix drifted")

    expected_matrix = screening.get("expected_matrix", {})
    frozen_expected_matrix = {
        "fixed_control_variants": 2,
        "learned_variants_per_condition": 12,
        "learned_conditions": 3,
        "total_variants": 38,
        "canonical_evidence_artifacts": 30,
        "training_data_artifacts": 90,
        "trained_model_artifacts": 360,
        "heldout_task_units": 9120,
    }
    if expected_matrix != frozen_expected_matrix:
        raise ValueError("frozen screening expected matrix drifted")
    authority_expected_matrix = {
        "fixed_control_variants": len(expected_fixed_controls),
        "learned_variants_per_condition": len(expected_candidates),
        "learned_conditions": len(expected_learned_conditions),
        "total_variants": len(expected_fixed_controls)
        + len(expected_learned_conditions) * len(expected_candidates),
        "canonical_evidence_artifacts": len(protocol_family_order)
        * len(screening_replicates),
        "training_data_artifacts": len(protocol_family_order)
        * len(screening_replicates)
        * 3,
        "trained_model_artifacts": len(protocol_family_order)
        * len(screening_replicates)
        * len(expected_learned_conditions)
        * 4,
        "heldout_task_units": len(protocol_family_order)
        * protocol_folds["screening_heldout_tasks_per_family"]
        * len(screening_replicates)
        * (
            len(expected_fixed_controls)
            + len(expected_learned_conditions) * len(expected_candidates)
        ),
    }
    if expected_matrix != authority_expected_matrix:
        raise ValueError("screening expected matrix is not bound to the frozen folds")

    expected_protocol_freeze = {
        "amended_at_local_date": "2026-08-21",
        "amendment_timing": "before comparative development results",
        "comparative_results_inspected_before_amendment": False,
        "previous_sha256": "a0d6e5760591ce70f95df4a87b8166dc69defa5e3242ddadabe3393b3e82a488",
        "reason": (
            "operationalize restricted interactions, bind clean causal controls, "
            "and isolate explicit pairing"
        ),
    }
    expected_screening_freeze = {
        "amended_at_local_date": "2026-08-21",
        "amendment_timing": "before comparative development results",
        "comparative_results_inspected_before_amendment": False,
        "previous_sha256": "9e1ba94a80324fcbfdf4ffdb7cc36d58b2a7bbca2713f293aca164cfdcf0e03c",
        "reason": "operationalize the interaction metric and bind capacity matching",
    }
    expected_stage_contract = {
        "state_conditioning": {
            "status": "eligible_in_phase2_screening",
            "matched_comparison": (
                "B2_global_listwise_optimum versus C_state_conditioned_listwise_optimum"
            ),
            "claim_scope": (
                "state-conditioned current-observation features beyond global affordance "
                "features under matched data, objective, updates, seeds, and search budget"
            ),
        },
        "transition_information": {
            "status": "deferred_until_named_phase3_matched_conditions_are_frozen",
            "claims_before_gate": "forbidden",
            "required_match": (
                "same trajectories, examples, objective, capacity band, seeds, optimizer, "
                "updates, inference budget, and search budget; transition-only must be "
                "compared with state-only"
            ),
        },
        "history_sequence": {
            "status": "deferred_until_named_phase3_phase6_matched_conditions_are_frozen",
            "claims_before_gate": "forbidden",
            "required_match": (
                "same trajectories, transitions, examples, objective, capacity band, seeds, "
                "optimizer, updates, inference budget, and search budget; history/sequence "
                "must be compared with transition-only"
            ),
        },
        "explicit_pairing": {
            "status": "deferred_until_D1_and_F_are_implemented_and_frozen",
            "claims_before_gate": "forbidden",
            "matched_comparison": (
                "D1_state_conditioned_unpaired_same_trajectories versus "
                "F_correctly_paired_improvement"
            ),
        },
    }
    if (
        protocol.get("freeze_record") != expected_protocol_freeze
        or screening.get("freeze_record") != expected_screening_freeze
        or protocol.get("representation_ladder_stage_contract")
        != expected_stage_contract
    ):
        raise ValueError("frozen protocol gate or freeze record drifted")

    rule = screening.get("screening_advancement_rule", {})
    endpoint = int(rule.get("endpoint_adaptation_actions", 0))
    sentinel = int(rule.get("failure_censoring_value", 0))
    if (
        rule.get("restricted_interactions_metric_id") != METRIC_ID
        or rule.get("executed_action_formula") != ACTION_FORMULA
        or sentinel != endpoint + 1
        or "reporting-only" not in str(rule.get("exact_optimum_search_control", ""))
        or rule.get("steps")
        != [
            "maximize the minimum family exact-optimum success rate",
            "retain tuples within five absolute percentage points of the best minimum-family success rate",
            "minimize worst-family median restricted interactions to exact optimum with failures at 2049",
            "minimize macro-average family median restricted interactions",
            "minimize one-time training optimizer steps, then forward passes",
            "choose the ascending numeric tuple (learning_rate, training_epochs, search_temperature)",
        ]
        or rule.get("cross_condition_elimination") is not False
        or rule.get("selection_rule_changes_after_screening_results") != "forbidden"
    ):
        raise ValueError("frozen selection metric authority drifted")
    return SelectionAuthority(
        protocol_path=protocol_path.resolve(),
        screening_candidates_path=screening_candidates_path.resolve(),
        task_manifest_path=task_manifest_path.resolve(),
        protocol_sha256=protocol_sha256,
        screening_candidates_sha256=screening_sha256,
        task_manifest_sha256=task_manifest_sha256,
        endpoint=endpoint,
        failure_sentinel=sentinel,
        family_ids=_FROZEN_FAMILY_ORDER,
        screening_replicates=screening_replicates,
        heldout_task_ids=tuple(heldout_task_ids),
        training_core_task_identities=tuple(training_core_task_identities),
        protocol_bytes=protocol_bytes,
        screening_candidates_bytes=screening_bytes,
        task_manifest_bytes=task_manifest_bytes,
        _construction_token=_AUTHORITY_CONSTRUCTION_TOKEN,
    )


def build_selection_metric_spec(
    config: ExperimentConfig,
    expected: ExpectedUnits,
    expected_shared: ExpectedSharedArtifacts,
    authority: SelectionAuthority,
    *,
    condition_id: str,
    phase: str = "validation",
) -> SelectionMetricSpec:
    """Build a metric spec only from a config that binds the frozen semantics."""

    from levelup.experiments.runner.storage import (
        ArtifactValidationError,
        RunStore,
        plan_expected_units,
    )

    canonical_authority = load_selection_authority(
        authority.protocol_path,
        authority.screening_candidates_path,
        authority.task_manifest_path,
        source_bytes=authority.source_bytes,
    )
    if authority != canonical_authority:
        raise ValueError("selection authority differs from the canonical frozen files")

    parameters = config.parameters
    endpoint = int(parameters.get("adaptation_action_cap", 0))
    if phase != "validation":
        raise ValueError("frozen screening selection uses validation units only")
    if (
        config.split.final_tasks
        or "final" in config.selection.phases
        or any("final" in condition.execution_phases for condition in config.conditions)
        or any(unit.key.phase != phase for unit in expected.units)
        or any(artifact.consumer_phase != phase for artifact in expected_shared.artifacts)
    ):
        raise ValueError("selection config or plan contains forbidden final-family material")
    if (
        parameters.get("selection_metric_id") != METRIC_ID
        or parameters.get("selection_metric_schema_version") != METRIC_SCHEMA_VERSION
        or parameters.get("selection_metric_action_formula") != ACTION_FORMULA
        or parameters.get("selection_metric_oracle_policy") != ORACLE_POLICY
        or parameters.get("selection_metric_phase") != phase
        or parameters.get("selection_metric_failure_sentinel") != endpoint + 1
        or parameters.get("development_protocol_sha256") != authority.protocol_sha256
        or parameters.get("screening_candidates_sha256")
        != authority.screening_candidates_sha256
        or parameters.get("development_task_manifest_sha256")
        != authority.task_manifest_sha256
        or endpoint != authority.endpoint
        or endpoint + 1 != authority.failure_sentinel
    ):
        raise ValueError("experiment config does not bind the frozen selection metric")
    if (
        parameters.get("shared_artifact_training") is not True
        or parameters.get("unit_local_training_repeated_and_counted") is not False
    ):
        raise ValueError("selection metric requires shared preparation and zero unit-local training")
    if phase not in config.selection.phases:
        raise ValueError("selection metric phase is absent from the config selection declaration")
    config_sha256 = scientific_config_sha256(config)
    run_id = run_id_for(config)
    canonical_expected = plan_expected_units(config)
    if (
        expected != canonical_expected
        or expected.config_sha256 != config_sha256
        or expected.run_id != run_id
        or expected_shared.config_sha256 != config_sha256
        or expected_shared.run_id != run_id
    ):
        raise ValueError("expected unit plan does not match the selection config identity")
    try:
        validated_store = RunStore(
            Path("."),
            config,
            repository=Path("."),
            shared_artifacts=expected_shared.artifacts,
        )
    except ArtifactValidationError as exc:
        raise ValueError("shared-artifact plan provenance is invalid") from exc
    if (
        validated_store.expected != expected
        or validated_store.expected_shared != expected_shared
    ):
        raise ValueError("selection plans differ from validated run-store plans")
    planned = tuple(
        unit
        for unit in expected.units
        if unit.key.condition_id == condition_id and unit.key.phase == phase
    )
    if not planned:
        raise ValueError("selection condition has no planned held-out units")
    planned_families = {unit.key.family_id for unit in planned}
    if len(planned_families) != 1:
        raise ValueError("each selection child must hold out exactly one development family")
    heldout_family = next(iter(planned_families))
    family_offset = authority.family_ids.index(heldout_family) * 10_000
    seed_policy = config.seed_policy
    if (
        seed_policy.derivation_version != "phase2.v1"
        or seed_policy.model_seed_base != 6_100_000 + family_offset
        or seed_policy.environment_seed_offset != 0
        or seed_policy.probe_seed_base != 6_200_000 + family_offset
        or seed_policy.search_seed_base != 6_300_000 + family_offset
        or seed_policy.data_order_seed_base != 6_400_000 + family_offset
        or seed_policy.replicate_stride != 100_000
    ):
        raise ValueError("selection child seed policy drifted from the frozen LOFO fold")
    expected_training = {
        task_id: (task_index, generator_seed, environment_reset_seed)
        for family_id, identities in authority.training_core_task_identities
        if family_id != heldout_family
        for task_id, task_index, generator_seed, environment_reset_seed in identities
    }
    expected_heldout = {
        task_id: (task_index, generator_seed, environment_reset_seed)
        for family_id, identities in authority.training_core_task_identities
        if family_id == heldout_family
        for task_id, task_index, generator_seed, environment_reset_seed in identities
    }
    actual_training = {
        task.task_id: (task.task_index, task.generator_seed, task.environment_reset_seed)
        for task in config.split.development_tasks
    }
    actual_heldout = {
        task.task_id: (task.task_index, task.generator_seed, task.environment_reset_seed)
        for task in config.split.validation_tasks
    }
    if actual_training != expected_training or actual_heldout != expected_heldout:
        raise ValueError("selection child task identities drift from the frozen LOFO fold")
    selected_conditions = [
        condition for condition in config.conditions if condition.condition_id == condition_id
    ]
    expected_exposure_tasks = set(expected_training) if condition_id not in {
        "A0-no-probe-uniform",
        "A1-paid-probe-uniform",
    } else set()
    if (
        len(selected_conditions) != 1
        or set(selected_conditions[0].exposure.train_task_ids)
        != expected_exposure_tasks
    ):
        raise ValueError("selection condition training exposure drifted from the LOFO fold")
    allowed_tasks = dict(authority.heldout_task_ids)
    if not planned_families <= set(authority.family_ids):
        raise ValueError("selection condition exceeds the frozen development families")
    for family_id in planned_families:
        family_units = [unit for unit in planned if unit.key.family_id == family_id]
        actual_task_ids = {unit.key.task_id for unit in family_units}
        if actual_task_ids != set(allowed_tasks[family_id]):
            raise ValueError("selection held-out task coverage drifted")
        for task_id in actual_task_ids:
            replicates = {
                unit.key.replicate for unit in family_units if unit.key.task_id == task_id
            }
            if replicates != set(authority.screening_replicates):
                raise ValueError("selection replicate coverage drifted")
    expected_unit_ids = {unit.unit_id for unit in expected.units}
    if any(
        artifact.consumer_phase != phase
        or not set(artifact.consumer_unit_ids) <= expected_unit_ids
        for artifact in expected_shared.artifacts
    ):
        raise ValueError("shared-artifact plan contains out-of-scope consumers")
    expected_keys_by_unit: dict[str, list[tuple[str, str]]] = {}
    for artifact in expected_shared.artifacts:
        for unit_id in artifact.consumer_unit_ids:
            expected_keys_by_unit.setdefault(unit_id, []).append(
                (artifact.kind, artifact.key_id)
            )
    fixed_conditions = {
        "A0-no-probe-uniform",
        "A1-paid-probe-uniform",
    }
    require_shared_preparation = condition_id not in fixed_conditions
    learned_bases = (
        "B1-clean-global-optimum-frequency",
        "B2-global-listwise-optimum",
        "C-state-conditioned-listwise-optimum",
    )
    matching_bases = [
        base
        for base in learned_bases
        if condition_id == base or condition_id.startswith(f"{base}--")
    ]
    if require_shared_preparation and len(matching_bases) != 1:
        raise ValueError("selection condition is not a frozen learned-condition variant")
    condition_base = matching_bases[0] if require_shared_preparation else None
    if not require_shared_preparation and any(
        expected_keys_by_unit.get(unit.unit_id)
        for unit in planned
    ):
        raise ValueError("fixed-control selection units cannot consume shared artifacts")
    artifacts_by_unit = {
        unit.unit_id: [
            artifact
            for artifact in expected_shared.artifacts
            if unit.unit_id in artifact.consumer_unit_ids
        ]
        for unit in planned
    }
    for unit in planned:
        artifacts = artifacts_by_unit[unit.unit_id]
        if require_shared_preparation and any(
            artifact.owner_family_id != heldout_family
            or artifact.owner_fold_id != f"lofo-{heldout_family}"
            or artifact.owner_replicate != unit.key.replicate
            or (
                artifact.kind == "training_data_evidence"
                and artifact.owner_group_id != "canonical-evidence"
            )
            or (
                artifact.kind in {"training_data_view", "training_artifact"}
                and artifact.owner_group_id != condition_base
            )
            for artifact in artifacts
        ):
            raise ValueError("selection shared-artifact owner lineage drifted")
    return SelectionMetricSpec(
        condition_id=condition_id,
        phase=phase,
        endpoint=endpoint,
        failure_sentinel=endpoint + 1,
        protocol_sha256=authority.protocol_sha256,
        screening_candidates_sha256=authority.screening_candidates_sha256,
        task_manifest_sha256=authority.task_manifest_sha256,
        family_universe=authority.family_ids,
        authority_family_ids=authority.family_ids,
        authority_family_ids_sha256=_FROZEN_AUTHORITY_FAMILY_IDS_SHA256,
        expected_units=tuple(
            ExpectedSelectionUnit(
                run_id=run_id,
                config_sha256=config_sha256,
                unit_id=unit.unit_id,
                key=unit.key,
                seeds=unit.seeds,
                exposure_manifest_sha256=unit.exposure_manifest_sha256,
                shared_key_ids=(
                    tuple(sorted(expected_keys_by_unit.get(unit.unit_id, ())))
                    if require_shared_preparation
                    else ()
                ),
            )
            for unit in planned
        ),
        require_shared_preparation=require_shared_preparation,
        require_candidate_generation_identity=True,
        require_zero_local_training=True,
        _construction_token=_SPEC_CONSTRUCTION_TOKEN,
    )


def merge_selection_metric_specs(
    specs: Iterable[SelectionMetricSpec],
) -> SelectionMetricSpec:
    """Combine disjoint child-fold plans without weakening their frozen semantics."""

    materialized = tuple(specs)
    if not materialized:
        raise ValueError("at least one selection metric spec is required")
    first = materialized[0]
    if (
        not first.authority_family_ids
        or first.authority_family_ids != _FROZEN_FAMILY_ORDER
        or first.authority_family_ids_sha256
        != _FROZEN_AUTHORITY_FAMILY_IDS_SHA256
        or any(spec.authority_family_ids != first.authority_family_ids for spec in materialized)
        or any(
            spec.authority_family_ids_sha256 != first.authority_family_ids_sha256
            for spec in materialized
        )
    ):
        raise ValueError("child selection metric specs lack the frozen authority family universe")
    comparable_fields = (
        "condition_id",
        "phase",
        "endpoint",
        "failure_sentinel",
        "metric_id",
        "schema_version",
        "action_formula",
        "oracle_policy",
        "require_shared_preparation",
        "require_candidate_generation_identity",
        "require_zero_local_training",
        "protocol_sha256",
        "screening_candidates_sha256",
        "task_manifest_sha256",
        "family_universe",
        "authority_family_ids",
        "authority_family_ids_sha256",
    )
    if any(
        any(getattr(spec, field) != getattr(first, field) for field in comparable_fields)
        for spec in materialized[1:]
    ):
        raise ValueError("child selection metric specs have incompatible semantics")
    family_sets = [spec.family_ids for spec in materialized]
    if any(
        left & right
        for index, left in enumerate(family_sets)
        for right in family_sets[index + 1 :]
    ):
        raise ValueError("child selection metric specs have overlapping held-out families")
    combined_families = frozenset().union(*family_sets)
    if combined_families != frozenset(first.authority_family_ids):
        raise ValueError("child selection metric specs do not cover the frozen family universe")
    return SelectionMetricSpec(
        condition_id=first.condition_id,
        phase=first.phase,
        endpoint=first.endpoint,
        failure_sentinel=first.failure_sentinel,
        protocol_sha256=first.protocol_sha256,
        screening_candidates_sha256=first.screening_candidates_sha256,
        task_manifest_sha256=first.task_manifest_sha256,
        family_universe=first.family_universe,
        authority_family_ids=first.authority_family_ids,
        authority_family_ids_sha256=first.authority_family_ids_sha256,
        expected_units=tuple(unit for spec in materialized for unit in spec.expected_units),
        require_shared_preparation=first.require_shared_preparation,
        require_candidate_generation_identity=first.require_candidate_generation_identity,
        require_zero_local_training=first.require_zero_local_training,
        _construction_token=_SPEC_CONSTRUCTION_TOKEN,
    )


def _validate_record_identity(record: UnitRecord, expected: ExpectedSelectionUnit) -> None:
    if (
        record.run_id != expected.run_id
        or record.config_sha256 != expected.config_sha256
        or record.unit_id != expected.unit_id
        or record.key != expected.key
        or record.seeds != expected.seeds
        or record.exposure_manifest_sha256 != expected.exposure_manifest_sha256
    ):
        raise ValueError("selection unit does not match its frozen expected identity")


def restricted_interactions(record: UnitRecord, spec: SelectionMetricSpec) -> int:
    """Return the typed post-hoc first-hit action count or frozen failure sentinel."""

    matches = [
        unit
        for unit in spec.expected_units
        if unit.run_id == record.run_id and unit.unit_id == record.unit_id
    ]
    if len(matches) != 1:
        raise ValueError("selection unit is absent from the frozen expected matrix")
    expected = matches[0]
    _validate_record_identity(record, expected)
    if not record.outcome.evaluator_ran:
        raise ValueError("selection units require independent evaluator evidence")
    if spec.require_shared_preparation:
        if record.accounting.training != PhaseAccounting():
            raise ValueError("selection units cannot contain unit-local training accounting")
        if record.shared_artifact is not None or {
            item.kind for item in record.shared_artifacts
        } != {"training_data_evidence", "training_data_view", "training_artifact"}:
            raise ValueError("learned selection units require complete typed shared lineage")
        actual_keys = tuple(sorted((item.kind, item.key_id) for item in record.shared_artifacts))
        if actual_keys != expected.shared_key_ids:
            raise ValueError("selection unit shared keys drift from the frozen artifact plan")
    elif spec.require_zero_local_training and record.accounting.training != PhaseAccounting():
        raise ValueError("fixed-control selection units cannot contain unit-local training")
    elif record.shared_artifact is not None or record.shared_artifacts:
        raise ValueError("fixed-control selection units cannot carry shared artifacts")
    if (
        (spec.require_shared_preparation or spec.require_candidate_generation_identity)
        and record.candidate_generation_sha256 is None
    ):
        raise ValueError("selection units require a candidate-generation identity")

    probe_actions = record.accounting.probes.actions
    search_actions = record.accounting.search.actions
    executed_actions = probe_actions + search_actions
    if executed_actions > spec.endpoint:
        raise ValueError("executed adaptation actions exceed the selection endpoint")

    outcome = record.outcome
    first_hit = outcome.first_optimum_adaptation_actions
    if outcome.success:
        if outcome.censored or outcome.censoring_budget is not None:
            raise ValueError("successful selection units cannot be censored")
        if outcome.first_optimum_episode is None:
            raise ValueError("successful selection units require a typed first optimum episode")
        if first_hit is None:
            raise ValueError("successful selection units require typed first optimum actions")
        if first_hit < probe_actions:
            raise ValueError("first optimum actions cannot precede paid probes")
        if first_hit > executed_actions:
            raise ValueError("first optimum actions exceed executed adaptation actions")
        if first_hit > spec.endpoint:
            raise ValueError("first optimum actions exceed the selection endpoint")
        return first_hit

    if first_hit is not None:
        raise ValueError("failed selection units cannot carry first optimum actions")
    if not outcome.censored or outcome.censoring_budget != spec.endpoint:
        raise ValueError("failed selection units require censoring at the fixed endpoint")
    if outcome.censoring_reason != "fixed_endpoint":
        raise ValueError("failed selection units require the fixed-endpoint censoring reason")
    return spec.failure_sentinel


def summarize_variant(
    records: Iterable[UnitRecord],
    spec: SelectionMetricSpec,
) -> VariantSelectionSummary:
    """Require the exact frozen matrix, aggregate within families, then equal-weight families."""

    if (
        spec.family_universe != _FROZEN_FAMILY_ORDER
        or spec.authority_family_ids != _FROZEN_FAMILY_ORDER
        or spec.authority_family_ids_sha256 != _FROZEN_AUTHORITY_FAMILY_IDS_SHA256
        or not spec.has_complete_family_coverage
    ):
        raise ValueError("selection summary requires the complete frozen family universe")
    materialized = tuple(records)
    actual_ids = [(record.run_id, record.unit_id) for record in materialized]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("selection records contain duplicate unit identities")
    expected_by_id = {(unit.run_id, unit.unit_id): unit for unit in spec.expected_units}
    if set(actual_ids) != set(expected_by_id):
        raise ValueError("selection records are missing or exceed the frozen expected matrix")

    by_family: dict[str, list[UnitRecord]] = {}
    for record in materialized:
        _validate_record_identity(record, expected_by_id[(record.run_id, record.unit_id)])
        by_family.setdefault(record.key.family_id, []).append(record)
    if set(by_family) != set(spec.family_ids):
        raise ValueError("selection family coverage drifted")

    families: list[FamilySelectionSummary] = []
    for family_id in sorted(by_family):
        family_records = by_family[family_id]
        interactions = [restricted_interactions(record, spec) for record in family_records]
        families.append(
            FamilySelectionSummary(
                family_id=family_id,
                units=len(family_records),
                exact_optimum_success_rate=sum(
                    1 for record in family_records if record.outcome.success
                )
                / len(family_records),
                median_restricted_interactions=float(statistics.median(interactions)),
            )
        )

    family_tuple = tuple(families)
    family_medians = [item.median_restricted_interactions for item in family_tuple]
    return VariantSelectionSummary(
        condition_id=spec.condition_id,
        endpoint=spec.endpoint,
        failure_sentinel=spec.failure_sentinel,
        families=family_tuple,
        minimum_family_exact_optimum_success_rate=min(
            item.exact_optimum_success_rate for item in family_tuple
        ),
        worst_family_median_restricted_interactions=max(family_medians),
        macro_average_family_median_restricted_interactions=sum(family_medians)
        / len(family_medians),
    )


def within_parameter_tolerance(left: int, right: int, tolerance: float = 0.1) -> bool:
    """Symmetric parameter-band check used by cross-representation comparisons."""

    if left < 1 or right < 1:
        raise ValueError("trainable parameter counts must be positive")
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError("parameter tolerance must be between zero and one")
    return abs(left - right) / max(left, right) <= tolerance
