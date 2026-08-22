from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from levelup.experiments.milestone6_phase2_screening_extraction import (
    extract_development_selection,
)
from levelup.experiments.runner.selection_metric import VariantSelectionSummary

FAMILIES = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
CONDITIONS = ("A0-no-probe-uniform", "A1-paid-probe-uniform") + tuple(
    f"{base}--{tuple_id}"
    for base in (
        "B1-clean-global-optimum-frequency",
        "B2-global-listwise-optimum",
        "C-state-conditioned-listwise-optimum",
    )
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
)


def _record(family: str, condition: str, index: int, *, phase: str = "validation"):
    return SimpleNamespace(
        key=SimpleNamespace(
            family_id=family,
            condition_id=condition,
            phase=phase,
        ),
        identity=(family, condition, index),
    )


def _runtime(*, records_per_fold: int = 1520, duplicate_family: bool = False):
    folds = []
    per_condition = 41 if records_per_fold > 1520 else 40
    for family_index, family in enumerate(FAMILIES):
        rows = [
            _record(family, condition, index)
            for condition in CONDITIONS
            for index in range(per_condition)
        ][:records_per_fold]
        fold_family = FAMILIES[0] if duplicate_family and family_index == 1 else family
        folds.append(
            SimpleNamespace(
                family_id=fold_family,
                config=SimpleNamespace(
                    conditions=tuple(
                        SimpleNamespace(
                            condition_id=item,
                            execution_phases=("validation",),
                        )
                        for item in CONDITIONS
                    ),
                    parameters={"final_family_access": False},
                    split=SimpleNamespace(final_tasks=()),
                ),
                store=SimpleNamespace(
                    expected=object(),
                    expected_shared=object(),
                    completed_records=lambda rows=tuple(rows): tuple(rows),
                ),
            )
        )
    sources = tuple(
        SimpleNamespace(
            label=label,
            path=f"/does/not/exist/{label}.json",
            content=f"{label}-bytes".encode(),
            sha256=hashlib.sha256(f"{label}-bytes".encode()).hexdigest(),
        )
        for label in ("protocol", "screening_candidates", "task_manifest")
    )
    return SimpleNamespace(
        folds=tuple(folds),
        authority_sources=sources,
        manifest=SimpleNamespace(
            family_order=FAMILIES,
            development_only=True,
            final_family_access=False,
            validation_executed=False,
            search_executed=False,
            outcomes_present=False,
            selection_performed=False,
        ),
    )


def _patch_lightweight(monkeypatch):
    import levelup.experiments.milestone6_phase2_screening_extraction as extraction

    authority = SimpleNamespace(
        family_ids=FAMILIES,
        protocol_sha256=hashlib.sha256(b"protocol-bytes").hexdigest(),
        screening_candidates_sha256=hashlib.sha256(b"screening_candidates-bytes").hexdigest(),
        task_manifest_sha256=hashlib.sha256(b"task_manifest-bytes").hexdigest(),
    )
    monkeypatch.setattr(extraction, "load_selection_authority", lambda *args, **kwargs: authority)
    monkeypatch.setattr(extraction, "build_selection_metric_spec", lambda *args, **kwargs: object())
    monkeypatch.setattr(extraction, "merge_selection_metric_specs", lambda specs: object())
    monkeypatch.setattr(
        extraction,
        "summarize_variant",
        lambda records, spec: VariantSelectionSummary(
            condition_id=records[0].key.condition_id,
            endpoint=2048,
            failure_sentinel=2049,
            families=(),
            minimum_family_exact_optimum_success_rate=1.0,
            worst_family_median_restricted_interactions=1.0,
            macro_average_family_median_restricted_interactions=1.0,
        ),
    )


def test_extraction_emits_exact_38_typed_summaries_without_path_reads(monkeypatch) -> None:
    runtime = _runtime()
    _patch_lightweight(monkeypatch)
    summaries = extract_development_selection(runtime)
    assert len(summaries) == 38
    assert tuple(item.condition_id for item in summaries) == CONDITIONS


def test_extractor_condition_ids_match_real_frozen_screening_configs() -> None:
    import levelup.experiments.milestone6_phase2_screening_extraction as extraction
    from levelup.experiments.milestone6_phase2_screening import (
        screening_child_configs,
    )

    configs = screening_child_configs()
    actual = tuple(condition.condition_id for condition in configs[0].conditions)
    handle = SimpleNamespace(
        folds=tuple(SimpleNamespace(config=config) for config in configs)
    )
    assert extraction._condition_ids(handle) == actual


@pytest.mark.parametrize(
    "runtime",
    [
        _runtime(records_per_fold=1519),
        _runtime(records_per_fold=1521),
        _runtime(duplicate_family=True),
    ],
)
def test_extraction_rejects_incomplete_extra_or_duplicate_folds(monkeypatch, runtime) -> None:
    _patch_lightweight(monkeypatch)
    with pytest.raises(ValueError):
        extract_development_selection(runtime)


def test_extraction_rejects_final_unit(monkeypatch) -> None:
    runtime = _runtime()
    runtime.folds[0].store.completed_records = lambda: (
        _record("plain", CONDITIONS[0], 0, phase="final"),
        *_runtime(records_per_fold=1519).folds[0].store.completed_records(),
    )
    _patch_lightweight(monkeypatch)
    with pytest.raises(ValueError, match="non-validation|final"):
        extract_development_selection(runtime)


def test_extraction_rejects_reordered_folds_and_manifest_flags(monkeypatch) -> None:
    _patch_lightweight(monkeypatch)
    runtime = _runtime()
    runtime.folds = (runtime.folds[1], runtime.folds[0], *runtime.folds[2:])
    with pytest.raises(ValueError, match="fold order"):
        extract_development_selection(runtime)
    runtime = _runtime()
    runtime.manifest.development_only = False
    with pytest.raises(ValueError, match="manifest"):
        extract_development_selection(runtime)


def test_extraction_rejects_final_config_before_reading_records(monkeypatch) -> None:
    _patch_lightweight(monkeypatch)
    runtime = _runtime()
    reads: list[str] = []
    runtime.folds[0].config.split.final_tasks = (object(),)
    runtime.folds[0].store.completed_records = lambda: reads.append("read") or ()
    with pytest.raises(ValueError, match="development-only"):
        extract_development_selection(runtime)
    assert reads == []


def test_authority_bytes_are_supplied_immutably_and_drift_fails_closed(monkeypatch) -> None:
    import levelup.experiments.milestone6_phase2_screening_extraction as extraction

    runtime = _runtime()
    seen = {}
    authority = SimpleNamespace(
        family_ids=FAMILIES,
        protocol_sha256=hashlib.sha256(b"protocol-bytes").hexdigest(),
        screening_candidates_sha256=hashlib.sha256(b"screening_candidates-bytes").hexdigest(),
        task_manifest_sha256=hashlib.sha256(b"task_manifest-bytes").hexdigest(),
    )

    def loader(*args, **kwargs):
        seen.update(kwargs["source_bytes"])
        if any(
            hashlib.sha256(value).hexdigest()
            != hashlib.sha256(f"{label}-bytes".encode()).hexdigest()
            for label, value in kwargs["source_bytes"].items()
        ):
            raise ValueError("authority bytes drifted")
        return authority

    monkeypatch.setattr(extraction, "load_selection_authority", loader)
    assert extraction._authority_from_runtime(runtime) is authority
    assert seen == {label: f"{label}-bytes".encode() for label in ("protocol", "screening_candidates", "task_manifest")}

    runtime.authority_sources = (
        *runtime.authority_sources[:-1],
        SimpleNamespace(
            label="task_manifest",
            path=runtime.authority_sources[-1].path,
            content=b"drifted",
            sha256=runtime.authority_sources[-1].sha256,
        ),
    )
    with pytest.raises(ValueError):
        extraction._authority_from_runtime(runtime)


def test_authority_loader_bytes_mode_does_not_read_paths() -> None:
    from levelup.experiments.milestone6_phase2_screening import selection_authority
    from levelup.experiments.runner.selection_metric import load_selection_authority

    source = selection_authority()
    loaded = load_selection_authority(
        Path("/does/not/exist/protocol.json"),
        Path("/does/not/exist/screening.json"),
        Path("/does/not/exist/tasks.json"),
        source_bytes=source.source_bytes,
    )
    assert loaded.protocol_sha256 == source.protocol_sha256
    assert loaded.screening_candidates_sha256 == source.screening_candidates_sha256
    assert loaded.task_manifest_sha256 == source.task_manifest_sha256
    assert loaded.source_bytes == source.source_bytes
