from __future__ import annotations

import shutil
from dataclasses import fields, replace
from functools import lru_cache
from pathlib import Path

import pytest

from levelup.experiments.milestone6_phase3_model_authority import (
    load_phase3_model_artifact_authority_bytes,
)
from levelup.experiments.milestone6_phase3_plan import (
    FAMILIES,
    ValidatedPhase3Plan,
    bind_validated_phase3_plan,
    build_phase3_plan,
)
from levelup.experiments.milestone6_phase3_result_store import (
    EXPECTED_FAMILY_UNIT_COUNT,
    EXPECTED_TOTAL_UNIT_COUNT,
    Phase3ResultStoreError,
    Phase3ResultStorePlanError,
    Phase3ResultStoreSpec,
    build_phase3_expected_plan,
    load_phase3_result_store,
    load_phase3_result_stores,
    prepare_phase3_result_store,
    prepare_phase3_result_stores,
    validate_phase3_expected_plan,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import (
    AttemptRecord,
    ResourceAccounting,
    UnitOutcome,
    UnitRecord,
)

AUTHORITY_PATH = Path("configs/milestone6/phase3_model_artifact_authority.json")


@lru_cache(maxsize=1)
def _authorities() -> tuple[ValidatedPhase3Plan, object]:
    plan = bind_validated_phase3_plan(build_phase3_plan())
    authority = load_phase3_model_artifact_authority_bytes(AUTHORITY_PATH.read_bytes())
    return plan, authority


def _prepare(tmp_path: Path, family_id: str = FAMILIES[0]):
    plan, authority = _authorities()
    root = tmp_path / "phase3-results"
    root.mkdir()
    return root, prepare_phase3_result_store(root, plan, authority, family_id=family_id)


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _unit_record(store, planned, **updates) -> UnitRecord:
    value = UnitRecord(
        run_id=store.run_id,
        config_sha256=store.config_sha256,
        unit_id=planned.unit.unit_id,
        key=planned.unit.key,
        seeds=planned.unit.seeds,
        exposure_manifest_sha256=planned.unit.exposure_manifest_sha256,
        started_at_utc="2026-08-23T00:00:00+00:00",
        finished_at_utc="2026-08-23T00:00:01+00:00",
        elapsed_wall_seconds=1.0,
        outcome=UnitOutcome(
            evaluator_ran=False,
            valid=False,
            completed=False,
            success=False,
            performance_metric_id="performance_value",
            performance_direction="minimize",
        ),
        accounting=ResourceAccounting(),
    )
    return value.model_copy(update=updates)


def _attempt_record(store, planned, **updates) -> AttemptRecord:
    value = AttemptRecord(
        run_id=store.run_id,
        config_sha256=store.config_sha256,
        unit_id=planned.unit.unit_id,
        attempt=1,
        key=planned.unit.key,
        seeds=planned.unit.seeds,
        status="failed",
        stage="test",
        exception_type="RuntimeError",
        sanitized_message="test failure",
        retryable=False,
        started_at_utc="2026-08-23T00:00:00+00:00",
        finished_at_utc="2026-08-23T00:00:01+00:00",
        elapsed_wall_seconds=1.0,
    )
    return value.model_copy(update=updates)


def test_result_plan_is_exact_six_family_partition() -> None:
    validated, authority = _authorities()
    result = build_phase3_expected_plan(validated, authority)

    assert result.family_order == FAMILIES
    assert tuple(store.family_id for store in result.stores) == FAMILIES
    assert tuple(len(store.units) for store in result.stores) == (EXPECTED_FAMILY_UNIT_COUNT,) * 6
    assert len(result.units) == EXPECTED_TOTAL_UNIT_COUNT
    assert set(result.unit_ids) == set(validated.plan.unit_ids)
    assert result.plan_id == validated.plan.plan_id == authority.plan_id
    assert result.protocol_sha256 == validated.plan.protocol_sha256 == authority.protocol_sha256
    assert result.model_authority_sha256 == authority.authority_sha256
    assert all(store.final_family_access is False for store in result.stores)


def test_result_plan_copies_exact_units_and_uses_explicit_store_hashes() -> None:
    validated, authority = _authorities()
    result = build_phase3_expected_plan(validated, authority)

    by_family = {
        family: tuple(item for item in validated.plan.units if item.heldout_family == family)
        for family in FAMILIES
    }
    for store in result.stores:
        assert store.units == by_family[store.family_id]
        assert store.store_config_sha256 != store.run_id
        assert len(store.store_config_sha256) == len(store.run_id) == 64
    assert validate_phase3_expected_plan(result, validated, authority) is result


def test_result_store_specs_require_canonical_construction_and_recomputed_hashes() -> None:
    validated, authority = _authorities()
    store = build_phase3_expected_plan(validated, authority).stores[0]
    raw = {
        item.name: getattr(store, item.name)
        for item in fields(store)
        if item.name != "_construction_token"
    }
    with pytest.raises(Phase3ResultStorePlanError, match="construction"):
        Phase3ResultStoreSpec(**raw)
    with pytest.raises(Phase3ResultStorePlanError, match="config or run"):
        replace(store, run_id="0" * 64)


def test_generic_expected_unit_planning_drift_cannot_change_result_identities(monkeypatch) -> None:
    validated, authority = _authorities()
    first = build_phase3_expected_plan(validated, authority)

    # A result-plan build must not consult generic ExperimentConfig planning.
    import levelup.experiments.runner.storage as storage

    monkeypatch.setattr(
        storage,
        "plan_expected_units",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("generic planner used")),
    )
    second = build_phase3_expected_plan(validated, authority)
    assert second == first
    assert tuple(store.run_id for store in second.stores) == tuple(
        store.run_id for store in first.stores
    )


def test_result_plan_rejects_final_plan() -> None:
    validated, authority = _authorities()
    bad_plan = replace(validated.plan, final_family_access=True)
    forged = ValidatedPhase3Plan(
        bad_plan,
        {item.unit.unit_id: item for item in bad_plan.units},
        _construction_token=validated._construction_token,
    )
    with pytest.raises(Phase3ResultStorePlanError, match="final"):
        build_phase3_expected_plan(forged, authority)


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_result_plan_rejects_missing_or_duplicate_unit_material(mutation: str) -> None:
    validated, authority = _authorities()
    if mutation == "missing":
        units = validated.plan.units[:-1]
    else:
        units = validated.plan.units[:-1] + (validated.plan.units[0],)
    bad_plan = replace(validated.plan, units=units)
    forged = ValidatedPhase3Plan(
        bad_plan,
        {item.unit.unit_id: item for item in bad_plan.units},
        _construction_token=validated._construction_token,
    )
    with pytest.raises(Phase3ResultStorePlanError, match="unit"):
        build_phase3_expected_plan(forged, authority)


def test_result_plan_rejects_model_authority_lineage_drift() -> None:
    validated, authority = _authorities()
    drifted = authority.model_copy(update={"plan_id": "0" * 64})
    with pytest.raises(Phase3ResultStorePlanError, match="lineage"):
        build_phase3_expected_plan(validated, drifted)


def test_result_plan_rejects_forged_plan_body_with_reused_plan_id() -> None:
    validated, authority = _authorities()
    first = validated.plan.units[0]
    changed = replace(first, training_tuple_id="forged-training-tuple")
    body = replace(validated.plan, units=(changed, *validated.plan.units[1:]))
    forged = ValidatedPhase3Plan(
        body,
        {item.unit.unit_id: item for item in body.units},
        _construction_token=validated._construction_token,
    )
    with pytest.raises(Phase3ResultStorePlanError, match="plan body"):
        build_phase3_expected_plan(forged, authority)


def test_preparation_requires_an_existing_output_root(tmp_path: Path) -> None:
    plan, authority = _authorities()
    with pytest.raises(Phase3ResultStoreError):
        prepare_phase3_result_store(
            tmp_path / "does-not-exist",
            plan,
            authority,
            family_id=FAMILIES[0],
        )


def test_six_store_preparation_is_idempotent_and_inert(tmp_path: Path) -> None:
    plan, authority = _authorities()
    root = tmp_path / "phase3-results"
    root.mkdir()

    first = prepare_phase3_result_stores(root, plan, authority)
    second = prepare_phase3_result_stores(root, plan, authority)

    assert tuple(store.family_id for store in first) == FAMILIES
    assert tuple(store.family_id for store in second) == FAMILIES
    assert all(store.execution_ready is False for store in first + second)
    assert tuple(store.run_id for store in first) == tuple(store.run_id for store in second)
    for store in first:
        store.validate_resume()
        with pytest.raises(Phase3ResultStoreError, match="not execution-ready"):
            store.write_completed()
        with pytest.raises(Phase3ResultStoreError, match="not execution-ready"):
            store.write_attempt()


def test_read_only_loader_reuses_existing_tree_without_writes(tmp_path: Path) -> None:
    plan, authority = _authorities()
    root = tmp_path / "phase3-results"
    root.mkdir()
    prepared = prepare_phase3_result_stores(root, plan, authority)
    before = sorted(
        (path.relative_to(root), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    )

    loaded = load_phase3_result_stores(root, plan, authority)

    assert tuple(store.family_id for store in loaded) == FAMILIES
    assert tuple(store.run_id for store in loaded) == tuple(store.run_id for store in prepared)
    assert all(store.execution_ready is False for store in loaded)
    after = sorted(
        (path.relative_to(root), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    )
    assert after == before

    one = load_phase3_result_store(
        root, plan, authority, family_id=FAMILIES[0]
    )
    assert one.family_id == FAMILIES[0]
    assert one.run_id == prepared[0].run_id


def test_read_only_loader_missing_tree_does_not_create_anything(tmp_path: Path) -> None:
    plan, authority = _authorities()
    root = tmp_path / "missing-phase3-results"

    with pytest.raises(Phase3ResultStoreError):
        load_phase3_result_stores(root, plan, authority)

    assert not root.exists()
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == []


@pytest.mark.parametrize("component", ["family", "run", "units", "attempts"])
def test_read_only_loader_rejects_symlinked_descendant_without_repair(
    tmp_path: Path, component: str
) -> None:
    plan, authority = _authorities()
    root = tmp_path / "phase3-results"
    root.mkdir()
    spec = build_phase3_expected_plan(plan, authority).store_for_family(FAMILIES[0])
    family = root / spec.family_id
    family.mkdir()
    run = family / spec.run_id
    run.mkdir()
    if component == "family":
        run.rmdir()
        family.rmdir()
        target = tmp_path / "family-target"
        target.mkdir()
        family.symlink_to(target, target_is_directory=True)
    elif component == "run":
        run.rmdir()
        target = tmp_path / "run-target"
        target.mkdir()
        run.symlink_to(target, target_is_directory=True)
    else:
        target = run / f"{component}-target"
        target.mkdir()
        (run / component).symlink_to(target, target_is_directory=True)

    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    with pytest.raises(Phase3ResultStoreError):
        load_phase3_result_store(root, plan, authority, family_id=FAMILIES[0])
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before


@pytest.mark.parametrize("kind", ["symlink", "file"])
def test_preparation_rejects_symlinked_or_nonregular_output_root(
    tmp_path: Path, kind: str
) -> None:
    plan, authority = _authorities()
    real = tmp_path / "real-root"
    real.mkdir()
    root = tmp_path / "phase3-results"
    if kind == "symlink":
        root.symlink_to(real, target_is_directory=True)
    else:
        root.write_text("not a directory")

    with pytest.raises(Phase3ResultStoreError):
        prepare_phase3_result_store(root, plan, authority, family_id=FAMILIES[0])


@pytest.mark.parametrize("component", ["run", "units", "attempts"])
@pytest.mark.parametrize("kind", ["symlink", "file"])
def test_preparation_rejects_symlinked_or_nonregular_descendant(
    tmp_path: Path, component: str, kind: str
) -> None:
    plan, authority = _authorities()
    root = tmp_path / "phase3-results"
    root.mkdir()
    store_spec = build_phase3_expected_plan(plan, authority).store_for_family(FAMILIES[0])
    family = root / store_spec.family_id
    family.mkdir()
    run = family / store_spec.run_id
    if component == "run":
        if kind == "symlink":
            target = tmp_path / "run-target"
            target.mkdir()
            run.symlink_to(target, target_is_directory=True)
        else:
            run.write_text("not a directory")
    else:
        run.mkdir()
        target = run / f"{component}-target"
        if kind == "symlink":
            target.mkdir()
            (run / component).symlink_to(target, target_is_directory=True)
        else:
            (run / component).write_text("not a directory")

    with pytest.raises(Phase3ResultStoreError):
        prepare_phase3_result_store(root, plan, authority, family_id=FAMILIES[0])


@pytest.mark.parametrize("method", ["validate_resume", "completed_records"])
def test_same_byte_run_replacement_is_detected(
    tmp_path: Path, method: str
) -> None:
    _, store = _prepare(tmp_path)
    original = store.run_dir
    replacement = original.with_name(f"{original.name}.replacement")
    backup = original.with_name(f"{original.name}.original")
    shutil.copytree(original, replacement)
    original.rename(backup)
    replacement.rename(original)

    with pytest.raises(Phase3ResultStoreError, match="identity"):
        getattr(store, method)()


def test_foreign_unit_record_is_rejected(tmp_path: Path) -> None:
    _, store = _prepare(tmp_path)
    planned = store.spec.units[0]
    foreign_id = "f" * 64
    record = _unit_record(store, planned, unit_id=foreign_id)
    _write_json(store.run_dir / "units" / f"{foreign_id}.json", record.model_dump(mode="json"))

    with pytest.raises(Phase3ResultStoreError, match="foreign"):
        store.validate_resume()


def test_conflicting_unit_record_is_rejected(tmp_path: Path) -> None:
    _, store = _prepare(tmp_path)
    planned = store.spec.units[0]
    record = _unit_record(store, planned, run_id="conflicting-run")
    _write_json(
        store.run_dir / "units" / f"{planned.unit.unit_id}.json",
        record.model_dump(mode="json"),
    )

    with pytest.raises(Phase3ResultStoreError, match="identity mismatch"):
        store.validate_resume()


def test_foreign_attempt_record_is_rejected(tmp_path: Path) -> None:
    _, store = _prepare(tmp_path)
    planned = store.spec.units[0]
    foreign_id = "e" * 64
    record = _attempt_record(store, planned, unit_id=foreign_id)
    _write_json(
        store.run_dir / "attempts" / f"{foreign_id}.attempt-0001.json",
        record.model_dump(mode="json"),
    )

    with pytest.raises(Phase3ResultStoreError, match="foreign"):
        store.validate_resume()


def test_conflicting_attempt_record_is_rejected(tmp_path: Path) -> None:
    _, store = _prepare(tmp_path)
    planned = store.spec.units[0]
    record = _attempt_record(store, planned, config_sha256="f" * 64)
    _write_json(
        store.run_dir / "attempts" / f"{planned.unit.unit_id}.attempt-0001.json",
        record.model_dump(mode="json"),
    )

    with pytest.raises(Phase3ResultStoreError, match="identity mismatch"):
        store.validate_resume()


def test_phase3_preparation_does_not_touch_phase2_run_directories(tmp_path: Path) -> None:
    plan, authority = _authorities()
    runs = tmp_path / "runs"
    phase2 = runs / "milestone6" / "phase2"
    phase2.mkdir(parents=True)
    marker = phase2 / "sentinel.txt"
    marker.write_text("untouched")
    before = sorted(path.relative_to(phase2) for path in phase2.rglob("*"))

    phase3 = runs / "milestone6" / "phase3"
    phase3.mkdir()
    prepare_phase3_result_stores(phase3, plan, authority)

    assert marker.read_text() == "untouched"
    assert sorted(path.relative_to(phase2) for path in phase2.rglob("*")) == before
