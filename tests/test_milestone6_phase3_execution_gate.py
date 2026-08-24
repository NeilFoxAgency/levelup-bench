from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

import levelup.experiments.milestone6_phase3_readiness as readiness
from levelup.experiments.milestone6_phase3_execution_gate import (
    ACTIVATION_MARKER_NAME,
    Phase3ActivationError,
    phase3_activation,
)
from levelup.experiments.milestone6_phase3_model_authority import (
    load_phase3_model_artifact_authority_bytes,
)
from levelup.experiments.milestone6_phase3_plan import (
    ValidatedPhase3Plan,
    bind_validated_phase3_plan,
    build_phase3_plan,
)
from levelup.experiments.milestone6_phase3_readiness import (
    AuthorityFileSnapshot,
    Phase3ActivationReadinessLease,
    Phase3ReadinessSnapshot,
)
from levelup.experiments.milestone6_phase3_result_store import (
    FAMILIES,
    build_phase3_expected_plan,
    prepare_phase3_result_stores,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import (
    AttemptRecord,
    PhaseAccounting,
    ResourceAccounting,
    UnitOutcome,
    UnitRecord,
)

AUTHORITY_PATH = Path("configs/milestone6/phase3_model_artifact_authority.json")
MODEL_KEY_FIXTURE_PATH = Path("tests/fixtures/phase3_model_key_indices")
_MODEL_KEYS_FD: int | None = None


@pytest.fixture(scope="module", autouse=True)
def _held_model_key_fixture() -> Iterator[None]:
    global _MODEL_KEYS_FD
    _MODEL_KEYS_FD = os.open(MODEL_KEY_FIXTURE_PATH, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield
    finally:
        os.close(_MODEL_KEYS_FD)
        _MODEL_KEYS_FD = None


def _authorities() -> tuple[ValidatedPhase3Plan, object]:
    plan = bind_validated_phase3_plan(build_phase3_plan())
    authority = load_phase3_model_artifact_authority_bytes(AUTHORITY_PATH.read_bytes())
    return plan, authority


def _lease(expected, *, active: bool = True) -> Phase3ActivationReadinessLease:
    snapshot = object.__new__(Phase3ReadinessSnapshot)
    values = {
        "plan_id": expected.plan_id,
        "model_authority_sha256": expected.model_authority_sha256,
        "git_dirty": False,
        "git_commit_sha": "a" * 40,
        "training_shuffle_report_sha256": "b" * 64,
        "training_shuffle_report_file_sha256": "c" * 64,
    }
    for name, value in values.items():
        object.__setattr__(snapshot, name, value)
    files = tuple(
        AuthorityFileSnapshot(
            relative_path=relative,
            content=(content := Path(relative).read_bytes()),
            sha256=hashlib.sha256(content).hexdigest(),
            parent_identity=(1, 1),
            file_identity=(1, 1),
            ancestor_identities=(),
        )
        for relative in (
            "configs/milestone6/phase3_representation_ladder.json",
            "configs/milestone6/phase3_model_artifact_authority.json",
            "configs/milestone6/phase3_training_shuffle_report.json",
        )
    )
    object.__setattr__(snapshot, "files", files)
    assert _MODEL_KEYS_FD is not None
    authority = load_phase3_model_artifact_authority_bytes(
        Path(readiness.PHASE3_MODEL_AUTHORITY_RELATIVE).read_bytes()
    )
    keys_relative = (
        f"runs/milestone6/{authority.artifact_store_id}/"
        "phase3-model-artifact-keys"
    )
    lease = Phase3ActivationReadinessLease(
        snapshot,
        {},
        {keys_relative: _MODEL_KEYS_FD},
        _token=readiness._ACTIVATION_LEASE_TOKEN,
    )
    if not active:
        lease._deactivate()
    return lease


def _valid_record(store, planned, authority, *, success: bool = False, history_digest: str | None = None) -> UnitRecord:
    owner = next(item for item in _authorities()[0].plan.model_owners if item.owner_id == planned.model_owner_id)
    row = next(item for item in authority.models if item.owner_id == owner.owner_id)
    report = json.loads((MODEL_KEY_FIXTURE_PATH / f"{row.key_id}.json").read_bytes())[
        "key"
    ]["report"]
    history = planned.base_condition_id == "H4-shuffled-history-transition-listwise-optimum"
    diagnostics = {
        "model_trainable_parameters": report["trainable_parameters"],
        "model_optimizer_steps": report["optimizer_steps"],
        "model_forward_passes": report["forward_passes"],
        "model_recurrent_steps": report["recurrent_steps"],
        "model_training_examples": report["training_examples"],
        "history_shuffle_claim_eligible": False if history else None,
        "history_shuffle_eligible_windows": 0,
        "history_shuffle_map_nonidentity_windows": 0,
        "history_shuffle_effective_tensor_changed_windows": 0,
        "history_shuffle_duplicate_vector_no_effect_windows": 0,
        "history_shuffle_unchanged_short_windows": 0,
    }
    outcome = UnitOutcome(
        evaluator_ran=True,
        valid=True,
        completed=True,
        success=success,
        performance_metric_id="performance_value",
        performance_value=1.0,
        performance_direction="minimize",
        first_optimum_episode=1 if success else None,
        first_optimum_adaptation_actions=65 if success else None,
        censored=not success,
        censoring_budget=None if success else 2048,
        censoring_reason=None if success else "fixed_endpoint",
    )
    return UnitRecord(
        run_id=store.run_id,
        config_sha256=store.config_sha256,
        unit_id=planned.unit.unit_id,
        key=planned.unit.key,
        seeds=planned.unit.seeds,
        exposure_manifest_sha256=planned.unit.exposure_manifest_sha256,
        started_at_utc="2026-08-23T00:00:00+00:00",
        finished_at_utc="2026-08-23T00:00:01+00:00",
        elapsed_wall_seconds=1.0,
        outcome=outcome,
        accounting=ResourceAccounting(
            probes=PhaseAccounting(actions=64, environment_steps=64),
            search=PhaseAccounting(episodes=1, actions=1, environment_steps=1, forward_passes=1),
            replay=PhaseAccounting(actions=1, environment_steps=1),
        ),
        shared_artifact={"key_id": row.key_id, "artifact_id": row.artifact_id, "cost_id": row.cost_id},
        candidate_generation_sha256="d" * 64,
        history_shuffle_permutation_map_sha256=history_digest,
        diagnostics=diagnostics,
    )


@pytest.fixture()
def prepared(tmp_path: Path):
    plan, authority = _authorities()
    expected = build_phase3_expected_plan(plan, authority)
    root = tmp_path / "phase3-results"
    root.mkdir()
    stores = prepare_phase3_result_stores(root, plan, authority)
    return root, expected, stores, authority


def test_activation_publishes_one_canonical_marker_and_is_idempotent(prepared) -> None:
    root, expected, stores, authority = prepared
    lease = _lease(expected)
    with phase3_activation(stores, expected, lease, expected_git_commit="a" * 40) as batch:
        assert batch.active
        assert tuple(item.family_id for item in batch.stores) == FAMILIES
    marker = root / ACTIVATION_MARKER_NAME
    first = marker.read_bytes()
    with phase3_activation(stores, expected, _lease(expected), expected_git_commit="a" * 40) as resumed:
        assert resumed.store_for_family(FAMILIES[0]).family_id == FAMILIES[0]
    assert marker.read_bytes() == first


def test_forged_or_expired_lease_is_rejected(prepared) -> None:
    root, expected, stores, _authority = prepared
    del root
    with pytest.raises(Phase3ActivationError, match="readiness lease"):
        with phase3_activation(stores, expected, object(), expected_git_commit="a" * 40):
            pass
    with pytest.raises(Phase3ActivationError, match="readiness lease"):
        with phase3_activation(stores, expected, _lease(expected, active=False), expected_git_commit="a" * 40):
            pass


def test_orphan_records_without_marker_are_rejected(prepared) -> None:
    root, expected, stores, _authority = prepared
    unit = stores[0].spec.units[0].unit.unit_id
    (root / FAMILIES[0] / stores[0].run_id / "units" / f"{unit}.json").write_text("{}")
    with pytest.raises(Phase3ActivationError, match="orphan"):
        with phase3_activation(stores, expected, _lease(expected), expected_git_commit="a" * 40):
            pass


def test_marker_tamper_and_temporary_remnant_are_rejected(
    prepared, tmp_path: Path
) -> None:
    root, expected, stores, _authority = prepared
    with phase3_activation(stores, expected, _lease(expected), expected_git_commit="a" * 40):
        pass
    marker = root / ACTIVATION_MARKER_NAME
    marker.write_bytes(marker.read_bytes() + b" ")
    with pytest.raises(Phase3ActivationError, match="marker"):
        with phase3_activation(stores, expected, _lease(expected), expected_git_commit="a" * 40):
            pass

    # Restore the canonical marker using a fresh activation in an isolated tree
    # so the tamper test does not rely on a destructive repair.
    other_root = tmp_path / "isolated-results"
    other_root.mkdir()
    plan, authority = _authorities()
    other_expected = build_phase3_expected_plan(plan, authority)
    other_stores = prepare_phase3_result_stores(other_root, plan, authority)
    (other_root / ".phase3-activation.leftover.tmp").write_bytes(b"x")
    with pytest.raises(Phase3ActivationError, match="foreign.*temporary"):
        with phase3_activation(
            other_stores,
            other_expected,
            _lease(other_expected),
            expected_git_commit="a" * 40,
        ):
            pass


def test_exact_commit_is_mandatory(prepared) -> None:
    _root, expected, stores, _authority = prepared
    with pytest.raises(Phase3ActivationError, match="exact authorized"):
        with phase3_activation(stores, expected, _lease(expected), expected_git_commit=None):  # type: ignore[arg-type]
            pass
    with pytest.raises(Phase3ActivationError, match="exact authorized"):
        with phase3_activation(stores, expected, _lease(expected), expected_git_commit="b" * 40):
            pass


@pytest.mark.parametrize("kind", ["missing", "reordered", "duplicate"])
def test_store_matrix_shape_fails_before_marker(prepared, kind: str) -> None:
    root, expected, stores, _authority = prepared
    if kind == "missing":
        candidate = stores[:-1]
    elif kind == "reordered":
        candidate = (stores[1], stores[0], *stores[2:])
    else:
        candidate = (*stores[:-1], stores[0])
    with pytest.raises(Phase3ActivationError):
        with phase3_activation(candidate, expected, _lease(expected), expected_git_commit="a" * 40):
            pass
    assert not (root / ACTIVATION_MARKER_NAME).exists()


def test_later_store_failure_leaves_no_marker(prepared) -> None:
    root, expected, stores, _authority = prepared
    run_json = root / stores[1].family_id / stores[1].run_id / "run.json"
    run_json.write_bytes(run_json.read_bytes().replace(b'"execution_ready":false', b'"execution_ready":true'))
    with pytest.raises(Phase3ActivationError):
        with phase3_activation(stores, expected, _lease(expected), expected_git_commit="a" * 40):
            pass
    assert not (root / ACTIVATION_MARKER_NAME).exists()


def test_same_byte_marker_inode_replacement_is_rejected(prepared) -> None:
    root, expected, stores, _authority = prepared
    with pytest.raises(Phase3ActivationError, match="marker"):
        with phase3_activation(stores, expected, _lease(expected), expected_git_commit="a" * 40) as batch:
            marker = root / ACTIVATION_MARKER_NAME
            replacement = root / ".marker-replacement"
            replacement.write_bytes(marker.read_bytes())
            replacement.replace(marker)
            batch.store_for_family(FAMILIES[0]).load_completed(stores[0].spec.units[0].unit.unit_id)


def test_body_exception_runs_postcheck_and_deactivates(prepared) -> None:
    root, expected, stores, _authority = prepared
    saved = None
    with pytest.raises(RuntimeError):
        with phase3_activation(stores, expected, _lease(expected), expected_git_commit="a" * 40) as batch:
            saved = batch
            raise RuntimeError("body failure")
    assert saved is not None and not saved.active
    with pytest.raises(Phase3ActivationError, match="expired"):
        saved.stores
    assert (root / ACTIVATION_MARKER_NAME).exists()


def test_canonical_completed_write_load_resume_and_conflict(prepared) -> None:
    _root, expected, stores, authority = prepared
    planned = stores[0].spec.units[0]
    record = _valid_record(stores[0], planned, authority)
    with phase3_activation(stores, expected, _lease(expected), expected_git_commit="a" * 40) as batch:
        family = batch.store_for_family(FAMILIES[0])
        unit_map = batch._unit_maps[0]
        assert batch._unit(0, planned.unit.unit_id) is unit_map[planned.unit.unit_id]
        assert batch._unit_maps[0] is unit_map
        assert family.write_completed(record)
        assert family.completed_unit_ids() == (planned.unit.unit_id,)
        assert batch.completed_unit_ids(FAMILIES[0]) == (planned.unit.unit_id,)
        assert not family.write_completed(record)
        assert family.load_completed(planned.unit.unit_id) == record
        with pytest.raises(Phase3ActivationError, match="conflicting"):
            family.write_completed(record.model_copy(update={"candidate_generation_sha256": "e" * 64}))
        assert len(batch._scientific._indices) == 1
    assert not batch._scientific._indices


def test_replacing_completed_record_while_active_fails_closed(prepared) -> None:
    root, expected, stores, authority = prepared
    planned = stores[0].spec.units[0]
    record = _valid_record(stores[0], planned, authority)
    with pytest.raises(Phase3ActivationError, match="identity"):
        with phase3_activation(
            stores,
            expected,
            _lease(expected),
            expected_git_commit="a" * 40,
        ) as batch:
            family = batch.store_for_family(FAMILIES[0])
            assert family.write_completed(record)
            path = root / FAMILIES[0] / stores[0].run_id / "units" / f"{planned.unit.unit_id}.json"
            replacement = path.with_name(".replacement.json")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
            with pytest.raises(Phase3ActivationError, match="identity"):
                batch.completed_unit_ids(FAMILIES[0])


def test_same_inode_valid_record_mutation_fails_on_exit(prepared) -> None:
    root, expected, stores, authority = prepared
    planned = stores[0].spec.units[0]
    record = _valid_record(stores[0], planned, authority)
    path = root / FAMILIES[0] / stores[0].run_id / "units" / f"{planned.unit.unit_id}.json"
    with pytest.raises(Phase3ActivationError, match="fingerprint|identity"):
        with phase3_activation(
            stores, expected, _lease(expected), expected_git_commit="a" * 40
        ) as batch:
            assert batch.store_for_family(FAMILIES[0]).write_completed(record)
            mutated = record.model_copy(update={"candidate_generation_sha256": "e" * 64})
            rendered = canonical_json_bytes(mutated.model_dump(mode="json")) + b"\n"
            with path.open("r+b") as handle:
                handle.seek(0)
                handle.write(rendered)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())


def test_external_valid_canonical_publication_fails_on_exit(prepared) -> None:
    root, expected, stores, authority = prepared
    planned = stores[0].spec.units[1]
    record = _valid_record(stores[0], planned, authority)
    with pytest.raises(Phase3ActivationError, match="untracked|prepared family"):
        with phase3_activation(
            stores, expected, _lease(expected), expected_git_commit="a" * 40
        ):
            path = (
                root
                / FAMILIES[0]
                / stores[0].run_id
                / "units"
                / f"{planned.unit.unit_id}.json"
            )
            path.write_bytes(canonical_json_bytes(record.model_dump(mode="json")) + b"\n")


def test_completed_foreign_and_semantic_records_are_rejected(prepared) -> None:
    _root, expected, stores, authority = prepared
    planned = stores[0].spec.units[0]
    record = _valid_record(stores[0], planned, authority)
    with phase3_activation(stores, expected, _lease(expected), expected_git_commit="a" * 40) as batch:
        family = batch.store_for_family(FAMILIES[0])
        with pytest.raises(Phase3ActivationError, match="foreign"):
            family.write_completed(record.model_copy(update={"unit_id": "f" * 64}))
        with pytest.raises(Phase3ActivationError, match="accounting"):
            family.write_completed(record.model_copy(update={"outcome": record.outcome.model_copy(update={"performance_direction": "maximize"})}))
        forged_diagnostics = dict(record.diagnostics)
        forged_diagnostics["model_training_examples"] += 1
        forged_diagnostics["model_forward_passes"] = (
            forged_diagnostics["model_optimizer_steps"]
            * forged_diagnostics["model_training_examples"]
        )
        with pytest.raises(Phase3ActivationError, match="held model-key"):
            family.write_completed(
                record.model_copy(update={"diagnostics": forged_diagnostics})
            )
        success = _valid_record(stores[0], planned, authority, success=True)
        with pytest.raises(Phase3ActivationError, match="first-hit"):
            family.write_completed(
                success.model_copy(
                    update={
                        "outcome": success.outcome.model_copy(
                            update={"first_optimum_adaptation_actions": 66}
                        )
                    }
                )
            )


def test_h4_search_shuffle_digest_is_distinct_from_training_view_digest(prepared) -> None:
    _root, expected, stores, authority = prepared
    planned = next(
        item
        for item in stores[0].spec.units
        if item.base_condition_id
        == "H4-shuffled-history-transition-listwise-optimum"
    )
    record = _valid_record(
        stores[0],
        planned,
        authority,
        history_digest="e" * 64,
    )
    with phase3_activation(
        stores,
        expected,
        _lease(expected),
        expected_git_commit="a" * 40,
    ) as batch:
        assert batch.store_for_family(FAMILIES[0]).write_completed(record)


def test_foreign_record_added_while_active_fails_exit_validation(prepared) -> None:
    root, expected, stores, _authority = prepared
    with pytest.raises(Phase3ActivationError, match="prepared family store"):
        with phase3_activation(
            stores,
            expected,
            _lease(expected),
            expected_git_commit="a" * 40,
        ):
            units = root / FAMILIES[0] / stores[0].run_id / "units"
            (units / "foreign.json").write_text("{}")


@pytest.mark.parametrize("location", ["root", "family"])
def test_foreign_outer_namespace_added_while_active_fails_exit_validation(
    prepared,
    location: str,
) -> None:
    root, expected, stores, _authority = prepared
    with pytest.raises(Phase3ActivationError, match="namespace|output root"):
        with phase3_activation(
            stores,
            expected,
            _lease(expected),
            expected_git_commit="a" * 40,
        ):
            parent = root if location == "root" else root / FAMILIES[0]
            (parent / "foreign-entry").mkdir()


def test_attempt_write_list_and_next_number(prepared) -> None:
    _root, expected, stores, _authority = prepared
    planned = stores[0].spec.units[0]
    attempt = AttemptRecord(
        run_id=stores[0].run_id,
        config_sha256=stores[0].config_sha256,
        unit_id=planned.unit.unit_id,
        attempt=1,
        key=planned.unit.key,
        seeds=planned.unit.seeds,
        status="failed",
        stage="test",
        exception_type="RuntimeError",
        sanitized_message="failure",
        retryable=False,
        started_at_utc="2026-08-23T00:00:00+00:00",
        finished_at_utc="2026-08-23T00:00:01+00:00",
        elapsed_wall_seconds=1.0,
    )
    with phase3_activation(stores, expected, _lease(expected), expected_git_commit="a" * 40) as batch:
        family = batch.store_for_family(FAMILIES[0])
        assert family.write_attempt(attempt)
        assert batch.next_attempt_number(planned.unit.unit_id, FAMILIES[0]) == 2
        assert batch.attempt_records(FAMILIES[0]) == (attempt,)
        with pytest.raises(Phase3ActivationError, match="9999"):
            family.write_attempt(attempt.model_copy(update={"attempt": 10000}))
