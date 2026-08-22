"""Read-only extraction of the frozen development screening metric.

The extractor deliberately accepts only an already-loaded ``ScreeningRuntime``.
It never activates stores, rechecks filesystem paths, writes artifacts, or loads
anything from a final family.  All six child stores must contain the exact
1,520-unit validation matrix before any summary is produced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from levelup.experiments.runner.selection_metric import (
    VariantSelectionSummary,
    build_selection_metric_spec,
    load_selection_authority,
    merge_selection_metric_specs,
    summarize_variant,
)

if TYPE_CHECKING:
    from levelup.experiments.milestone6_phase2_screening_runtime import ScreeningRuntime


_FROZEN_FAMILIES = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
_FROZEN_FIXED = ("A0-no-probe-uniform", "A1-paid-probe-uniform")
_FROZEN_LEARNED_BASES = (
    "B1-clean-global-optimum-frequency",
    "B2-global-listwise-optimum",
    "C-state-conditioned-listwise-optimum",
)
_EXPECTED_UNITS_PER_CHILD = 1_520
_EXPECTED_TOTAL_UNITS = 9_120


def _authority_from_runtime(runtime: ScreeningRuntime):
    sources = tuple(runtime.authority_sources)
    if tuple(source.label for source in sources) != (
        "protocol",
        "screening_candidates",
        "task_manifest",
    ):
        raise ValueError("screening runtime authority source labels are not canonical")
    if any(not isinstance(source.content, bytes) for source in sources):
        raise ValueError("screening runtime authority source bytes are not immutable bytes")
    paths = {source.label: source.path for source in sources}
    contents = {source.label: source.content for source in sources}
    authority = load_selection_authority(
        paths["protocol"],
        paths["screening_candidates"],
        paths["task_manifest"],
        source_bytes=contents,
    )
    if tuple(authority.family_ids) != _FROZEN_FAMILIES:
        raise ValueError("screening runtime authority family universe drifted")
    if tuple(source.sha256 for source in sources) != (
        authority.protocol_sha256,
        authority.screening_candidates_sha256,
        authority.task_manifest_sha256,
    ):
        raise ValueError("screening runtime authority source digest drifted")
    return authority


def _condition_ids(runtime: ScreeningRuntime) -> tuple[str, ...]:
    configs = tuple(fold.config for fold in runtime.folds)
    if len(configs) != len(_FROZEN_FAMILIES):
        raise ValueError("screening runtime must contain exactly six development folds")
    ids = tuple(condition.condition_id for condition in configs[0].conditions)
    expected = set(_FROZEN_FIXED) | {
        f"{base}--{tuple_id}"
        for base in _FROZEN_LEARNED_BASES
        for tuple_id in (
            "lr0p003-e120-t0p6",
            "lr0p003-e120-t0p9",
            "lr0p003-e120-t1p2",
            "lr0p003-e180-t0p6",
            "lr0p003-e180-t0p9",
            "lr0p003-e180-t1p2",
            "lr0p01-e120-t0p6",
            "lr0p01-e120-t0p9",
            "lr0p01-e120-t1p2",
            "lr0p01-e180-t0p6",
            "lr0p01-e180-t0p9",
            "lr0p01-e180-t1p2",
        )
    }
    if len(ids) != 38 or set(ids) != expected or len(set(ids)) != len(ids):
        raise ValueError("screening runtime condition matrix is not the frozen 38-variant set")
    if any(tuple(condition.condition_id for condition in config.conditions) != ids for config in configs[1:]):
        raise ValueError("screening runtime folds disagree on condition identities")
    return ids


def extract_development_selection(runtime: ScreeningRuntime) -> tuple[VariantSelectionSummary, ...]:
    """Extract all 38 frozen variant summaries from completed development records."""

    manifest = runtime.manifest
    if (
        tuple(manifest.family_order) != _FROZEN_FAMILIES
        or manifest.development_only is not True
        or manifest.final_family_access is not False
        or manifest.validation_executed is not False
        or manifest.search_executed is not False
        or manifest.outcomes_present is not False
        or manifest.selection_performed is not False
    ):
        raise ValueError("screening runtime manifest is not an untouched development manifest")
    if tuple(fold.family_id for fold in runtime.folds) != _FROZEN_FAMILIES:
        raise ValueError("screening runtime fold order is not the frozen family order")
    authority = _authority_from_runtime(runtime)
    condition_ids = _condition_ids(runtime)
    records_by_family = {}
    specs_by_condition: dict[str, list] = {condition_id: [] for condition_id in condition_ids}
    for fold in runtime.folds:
        family_id = fold.family_id
        if family_id not in _FROZEN_FAMILIES or family_id in records_by_family:
            raise ValueError("screening runtime fold family coverage is invalid")
        if (
            fold.config.parameters.get("final_family_access") is not False
            or fold.config.split.final_tasks
            or any(
                tuple(condition.execution_phases) != ("validation",)
                for condition in fold.config.conditions
            )
        ):
            raise ValueError("screening runtime fold is not development-only")
        records = tuple(fold.store.completed_records())
        if len(records) != _EXPECTED_UNITS_PER_CHILD:
            raise ValueError("screening runtime child has incomplete or extra completed units")
        if any(record.key.phase != "validation" for record in records):
            raise ValueError("screening runtime contains a non-validation or final unit")
        if any(record.key.family_id != family_id for record in records):
            raise ValueError("screening runtime child contains a mismatched family record")
        records_by_family[family_id] = records
        for condition_id in condition_ids:
            specs_by_condition[condition_id].append(
                build_selection_metric_spec(
                    fold.config,
                    fold.store.expected,
                    fold.store.expected_shared,
                    authority,
                    condition_id=condition_id,
                )
            )
    if tuple(sorted(records_by_family)) != tuple(sorted(_FROZEN_FAMILIES)):
        raise ValueError("screening runtime does not cover all six frozen families")
    if sum(len(records) for records in records_by_family.values()) != _EXPECTED_TOTAL_UNITS:
        raise ValueError("screening runtime does not contain exactly 9,120 completed units")

    summaries: list[VariantSelectionSummary] = []
    for condition_id in condition_ids:
        merged = merge_selection_metric_specs(specs_by_condition[condition_id])
        records = tuple(
            record
            for family_id in _FROZEN_FAMILIES
            for record in records_by_family[family_id]
            if record.key.condition_id == condition_id
        )
        if len(records) != 240:
            raise ValueError("screening variant does not contain the exact 240-unit matrix")
        summaries.append(summarize_variant(records, merged))
    return tuple(summaries)


# Descriptive alias for callers that use the module's screening terminology.
extract_screening_selection = extract_development_selection


__all__ = ["extract_development_selection", "extract_screening_selection"]
