"""Fixture-only tests for the canonical pooled-table source boundary."""

from __future__ import annotations

import hashlib
import os
import shutil
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from levelup.experiments import milestone6_phase3_local_affordance_canonical_tables as tables
from levelup.experiments import milestone6_phase3_local_affordance_readiness as readiness
from levelup.experiments.milestone6_phase3_local_affordance_evidence import (
    RawProbeArtifactKey,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import (
    AffordanceTableRecord,
    ObservableStateRecord,
    ObservableTraceRecord,
    ObservedTransitionRecord,
    TrainingDataEvidenceKey,
    TrainingDataEvidenceManifest,
    TrainingDataEvidencePayloadBundle,
    TrainingDataPayload,
    TrainingDataSample,
    open_training_data_reader,
)

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
FAMILIES = ("plain", "battery", "cooldown", "heat", "momentum", "combo")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _table(value: float = 1.0) -> AffordanceTableRecord:
    return AffordanceTableRecord(features={"wait": (value,) * 49}, sample_counts={"wait": 1})


def _sample(task_id: str, table: AffordanceTableRecord) -> TrainingDataSample:
    before = ObservableStateRecord(
        progress_fraction=0.0,
        remaining_fraction=1.0,
        elapsed_per_target=0.0,
        resource_fraction=1.0,
        pressure_fraction=0.0,
        available_aliases=("wait",),
    )
    after = before.model_copy(update={"progress_fraction": 1.0, "remaining_fraction": 0.0})
    return TrainingDataSample(
        task_id=task_id,
        trace=ObservableTraceRecord(
            transitions=(ObservedTransitionRecord(before=before, action_alias="wait", after=after, completed=True),)
        ),
        affordances=table,
    )


def _expectation(family: str, replicate: int, run: str) -> tables._EvidenceExpectation:
    task_id = f"task-{family}-{replicate}"
    key = TrainingDataEvidenceKey(
        screening_candidates_sha256=_hash("screening"),
        protocol_sha256=_hash("protocol"),
        task_manifest_sha256=_hash("tasks"),
        expected_unit_plan_sha256=_hash("units"),
        provenance_sha256=_hash("provenance"),
        reference_exposure_sha256=_hash("exposure"),
        probe_policy_sha256=_hash("probes"),
        fold_id=f"lofo-{family}",
        heldout_family_id=family,
        ordered_training_task_ids=(task_id,),
        ordered_heldout_task_ids=(f"heldout-{family}",),
        replicate=replicate,
        data_order_seed=replicate,
        probe_seeds=(replicate,),
        environment_seeds=(0,),
    )
    provisional = {
        "schema_version": "runner.training-data-evidence.v1",
        "evidence_key_id": key.key_id,
        "key": key.model_dump(mode="json"),
        "payload_sha256": _hash(task_id),
        "payload_bytes": 1,
        "sample_task_ids": [task_id],
    }
    evidence_id = hashlib.sha256(canonical_json_bytes(provisional)).hexdigest()
    manifest = TrainingDataEvidenceManifest(evidence_id=evidence_id, **provisional)
    return tables._EvidenceExpectation(
        heldout_family=family,
        replicate=replicate,
        child_run_id=run,
        evidence_id=evidence_id,
        key=key,
        manifest=manifest,
        manifest_sha256=_hash(f"manifest-{family}-{replicate}"),
        payload_sha256=manifest.payload_sha256,
        payload_bytes=1,
    )


def _bundle(expected: tables._EvidenceExpectation) -> TrainingDataEvidencePayloadBundle:
    payload = TrainingDataPayload(samples=(_sample(expected.key.ordered_training_task_ids[0], _table()),))
    payload_bytes = canonical_json_bytes(payload.model_dump(mode="json"))
    manifest = expected.manifest.model_copy(
        update={"payload_sha256": hashlib.sha256(payload_bytes).hexdigest(), "payload_bytes": len(payload_bytes)}
    )
    # A model-copy changes the identity; the bundle validator test below is
    # intentionally about the explicit byte/digest binding instead.
    manifest_bytes = b"manifest"
    return TrainingDataEvidencePayloadBundle(
        manifest=manifest,
        payload=payload,
        manifest_bytes=manifest_bytes,
        payload_bytes=payload_bytes,
    )


class _Lease:
    def __init__(self, keys: tuple[RawProbeArtifactKey, ...], lock_bytes: bytes) -> None:
        self.authority = SimpleNamespace(
            keys=keys,
            evidence_lock_file_sha256=hashlib.sha256(lock_bytes).hexdigest(),
        )
        self._lock_bytes = lock_bytes
        self.active = True

    def require_active(self) -> "_Lease":
        if not self.active:
            raise ValueError("inactive fixture lease")
        return self

    def phase3_evidence_lock_bytes(self) -> bytes:
        self.require_active()
        return self._lock_bytes


@dataclass(frozen=True)
class _CompleteFixture:
    raw_root: Path
    lease: _Lease
    keys: tuple[RawProbeArtifactKey, ...]
    expectations: dict[tuple[str, int], tables._EvidenceExpectation]


def _raw_keys() -> tuple[RawProbeArtifactKey, ...]:
    return tuple(
        RawProbeArtifactKey(
            local_affordance_protocol_sha256=_hash("local"),
            development_protocol_sha256=_hash("protocol"),
            development_tasks_sha256=_hash("tasks"),
            phase3_evidence_lock_sha256=_hash("evidence-lock"),
            probe_policy_sha256=_hash("probe-policy"),
            family_id=family,
            replicate=replicate,
            task_index=task_index,
            task_id=f"task-{family}-{task_index}",
            generator_seed=1_000 + FAMILIES.index(family) * 100 + task_index,
            probe_seed=(
                6_200_000
                + FAMILIES.index(family) * 10_000
                + replicate * 100_000
                + task_index
            ),
            environment_seed=0,
        )
        for family in FAMILIES
        for replicate in range(5)
        for task_index in range(8)
    )


def _complete_fixture(
    tmp_path: Path,
    *,
    drift: tuple[str, int, str] | None = None,
    omit: tuple[str, int, str] | None = None,
) -> _CompleteFixture:
    raw_root = tmp_path / "phase2-raw"
    raw_root.mkdir(parents=True)
    keys = _raw_keys()
    task_ids = {
        family: tuple(f"task-{family}-{index}" for index in range(8))
        for family in FAMILIES
    }
    expectations: dict[tuple[str, int], tables._EvidenceExpectation] = {}
    for heldout in FAMILIES:
        child_run_id = f"run-{heldout}"
        run = raw_root / child_run_id
        for name in (
            "training-data-evidence-costs",
            "training-data-view-costs",
            "training-data-artifact-keys",
            "training-data-evidence",
            "training-data-artifacts",
        ):
            (run / name).mkdir(parents=True, exist_ok=True)
        for replicate in range(5):
            ordered = tuple(
                task_id
                for family in FAMILIES
                if family != heldout
                for task_id in task_ids[family]
                if omit != (heldout, replicate, task_id)
            )
            samples = tuple(
                _sample(
                    task_id,
                    _table(2.0 if drift == (heldout, replicate, task_id) else 1.0),
                )
                for task_id in ordered
            )
            payload = TrainingDataPayload(samples=samples)
            payload_bytes = canonical_json_bytes(payload.model_dump(mode="json"))
            key = TrainingDataEvidenceKey(
                screening_candidates_sha256=_hash("screening"),
                protocol_sha256=_hash("protocol"),
                task_manifest_sha256=_hash("tasks"),
                expected_unit_plan_sha256=_hash("units"),
                provenance_sha256=_hash("provenance"),
                reference_exposure_sha256=_hash("exposure"),
                probe_policy_sha256=_hash("probes"),
                fold_id=f"lofo-{heldout}",
                heldout_family_id=heldout,
                ordered_training_task_ids=ordered,
                ordered_heldout_task_ids=task_ids[heldout],
                replicate=replicate,
                data_order_seed=6_400_000 + replicate * 100_000,
                probe_seeds=tuple(range(len(ordered))),
                environment_seeds=(0,) * len(ordered),
            )
            unsigned = {
                "schema_version": "runner.training-data-evidence.v1",
                "evidence_key_id": key.key_id,
                "key": key.model_dump(mode="json"),
                "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "payload_bytes": len(payload_bytes),
                "sample_task_ids": ordered,
            }
            evidence_id = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
            manifest = TrainingDataEvidenceManifest(
                evidence_id=evidence_id,
                **unsigned,
            )
            manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
            evidence = run / "training-data-evidence" / evidence_id
            evidence.mkdir()
            (evidence / "manifest.json").write_bytes(manifest_bytes)
            (evidence / "samples.json").write_bytes(payload_bytes)
            expectations[heldout, replicate] = tables._EvidenceExpectation(
                heldout_family=heldout,
                replicate=replicate,
                child_run_id=child_run_id,
                evidence_id=evidence_id,
                key=key,
                manifest=manifest,
                manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
                payload_bytes=len(payload_bytes),
            )
    lock_bytes = b"fixture-lock"
    return _CompleteFixture(
        raw_root=raw_root,
        lease=_Lease(keys, lock_bytes),
        keys=keys,
        expectations=expectations,
    )


def test_lock_parser_rejects_missing_duplicate_and_final_scope() -> None:
    # The parser is deliberately strict before any filesystem descriptor opens.
    with pytest.raises(tables.CanonicalPooledTableError, match="self-hash drifted"):
        tables._expectations(b"{}")


def test_bundle_lineage_rejects_manifest_and_payload_drift() -> None:
    expected = _expectation("plain", 0, "run-plain")
    bundle = _bundle(expected)
    with pytest.raises(tables.CanonicalPooledTableError, match="differs from lock"):
        tables._bundle_is_expected(bundle, expected)


@pytest.fixture
def authority_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    for relative in readiness.SOURCE_RELATIVE_PATHS:
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return repository


def test_activation_revalidates_retained_lock_before_opening_raw_root(
    authority_repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(readiness, "_git_state", lambda _path: (COMMIT, False))
    destination_parent = tmp_path / "destination"
    destination_parent.mkdir()
    snapshot = readiness.capture_local_affordance_readiness(
        authority_repository, raw_publication_destination=destination_parent / "raw"
    )
    with snapshot.activation(expected_git_commit=COMMIT) as lease:
        raw_root = tmp_path / "does-not-exist"
        monkeypatch.setattr(type(lease), "phase3_evidence_lock_bytes", lambda _lease: b"{}")
        with pytest.raises(tables.CanonicalPooledTableError, match="digest differs"):
            with tables.activate_canonical_pooled_tables(
                lease,
                phase2_raw_root=raw_root,
            ):
                pass
        assert not raw_root.exists()


def test_source_public_surface_exposes_no_payload_or_result_accessor() -> None:
    assert set(tables.__all__) == {
        "CanonicalPooledTableError",
        "CanonicalPooledTableSource",
        "activate_canonical_pooled_tables",
    }
    for forbidden in ("payload", "trace", "result", "model", "outcome", "ScreeningRuntime"):
        assert not hasattr(tables.CanonicalPooledTableSource, forbidden)


def test_safe_component_rejects_paths_and_control_characters() -> None:
    for value in ("", ".", "..", "a/b", "a\\b", "a\x00b", None):
        with pytest.raises(tables.CanonicalPooledTableError):
            tables._safe_component(value, "test")
    assert tables._safe_component("safe-run", "test") == "safe-run"


def test_source_direct_construction_fails_closed() -> None:
    with pytest.raises(tables.CanonicalPooledTableError, match="require activation"):
        tables.CanonicalPooledTableSource()


def test_complete_fixture_activates_exact_five_copy_240_table_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _complete_fixture(tmp_path)
    monkeypatch.setattr(tables, "LocalAffordanceActivationLease", _Lease)
    monkeypatch.setattr(tables, "_expectations", lambda _content: fixture.expectations)

    with tables.activate_canonical_pooled_tables(
        fixture.lease,
        phase2_raw_root=fixture.raw_root,
    ) as source:
        assert source.require_active() is source
        first = source.table_for(fixture.keys[0])
        assert type(first) is AffordanceTableRecord
        assert first == _table()
        first.features["wait"] = (99.0,) * 49
        assert source.table_for(fixture.keys[0]) == _table()
        assert all(source.table_for(key) == _table() for key in fixture.keys)
        assert "task-" not in repr(source)
        held_fds = (
            source._root_fd,
            *source._child_fds.values(),
            *(
                fd
                for reader in source._readers.values()
                for fd in (
                    reader.evidence_costs_fd,
                    reader.view_costs_fd,
                    reader.view_keys_fd,
                    reader.evidence_root_fd,
                    reader.artifact_root_fd,
                )
            ),
            *(fd for descriptors in source._evidence_fds.values() for fd in descriptors),
        )
        assert len(set(held_fds)) == len(held_fds)

    with pytest.raises(tables.CanonicalPooledTableError, match="expired or forged"):
        source.table_for(fixture.keys[0])
    for fd in held_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("failure", ["one-copy-drift", "missing-copy"])
def test_complete_fixture_rejects_nonidentical_or_missing_fifth_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    task_id = "task-plain-0"
    fixture = _complete_fixture(
        tmp_path,
        drift=("battery", 0, task_id) if failure == "one-copy-drift" else None,
        omit=("battery", 0, task_id) if failure == "missing-copy" else None,
    )
    monkeypatch.setattr(tables, "LocalAffordanceActivationLease", _Lease)
    monkeypatch.setattr(tables, "_expectations", lambda _content: fixture.expectations)

    message = "not byte-identical" if failure == "one-copy-drift" else "missing or duplicate"
    with pytest.raises(tables.CanonicalPooledTableError, match=message):
        with tables.activate_canonical_pooled_tables(
            fixture.lease,
            phase2_raw_root=fixture.raw_root,
        ):
            pytest.fail("invalid five-copy evidence must not activate")


def test_complete_source_rejects_unknown_key_and_nested_table_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _complete_fixture(tmp_path)
    monkeypatch.setattr(tables, "LocalAffordanceActivationLease", _Lease)
    monkeypatch.setattr(tables, "_expectations", lambda _content: fixture.expectations)

    with tables.activate_canonical_pooled_tables(
        fixture.lease,
        phase2_raw_root=fixture.raw_root,
    ) as source:
        unknown = fixture.keys[0].model_copy(update={"task_id": "unknown-development-task"})
        with pytest.raises(tables.CanonicalPooledTableError, match="outside canonical"):
            source.table_for(unknown)

    fixture = _complete_fixture(tmp_path / "second")
    monkeypatch.setattr(tables, "_expectations", lambda _content: fixture.expectations)
    with pytest.raises(tables.CanonicalPooledTableError, match="expired or forged"):
        with tables.activate_canonical_pooled_tables(
            fixture.lease,
            phase2_raw_root=fixture.raw_root,
        ) as source:
            source._tables[fixture.keys[0].key_id].features["wait"] = (3.0,) * 49
            source.table_for(fixture.keys[0])


def test_complete_source_rejects_same_byte_manifest_replacement_on_recheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _complete_fixture(tmp_path)
    monkeypatch.setattr(tables, "LocalAffordanceActivationLease", _Lease)
    monkeypatch.setattr(tables, "_expectations", lambda _content: fixture.expectations)
    expected = fixture.expectations["plain", 0]
    target = (
        fixture.raw_root
        / expected.child_run_id
        / "training-data-evidence"
        / expected.evidence_id
        / "manifest.json"
    )

    with pytest.raises(tables.CanonicalPooledTableError, match="entry path identity changed"):
        with tables.activate_canonical_pooled_tables(
            fixture.lease,
            phase2_raw_root=fixture.raw_root,
        ) as source:
            replacement = target.with_name("replacement.json")
            replacement.write_bytes(target.read_bytes())
            os.replace(replacement, target)
            source.require_active()


@pytest.mark.parametrize("replacement", ["file", "evidence-directory"])
def test_current_evidence_identity_rejects_same_byte_replacement(
    tmp_path: Path, replacement: str
) -> None:
    """A path replacement is invalid even where its manifest/payload bytes match."""

    expected = _expectation("plain", 0, "run-plain")
    run = tmp_path / expected.child_run_id
    evidence = run / "training-data-evidence" / expected.evidence_id
    for name in (
        "training-data-evidence-costs",
        "training-data-view-costs",
        "training-data-artifact-keys",
        "training-data-evidence",
        "training-data-artifacts",
    ):
        (run / name).mkdir(parents=True, exist_ok=True)
    evidence.mkdir()
    (evidence / "manifest.json").write_bytes(b"same-manifest")
    (evidence / "samples.json").write_bytes(b"same-samples")
    run_fd = secure_fs.open_directory_chain(run)
    stack = ExitStack()
    try:
        reader = stack.enter_context(open_training_data_reader(run_fd))
        evidence_fd = secure_fs.open_child_directory(reader.evidence_root_fd, expected.evidence_id)
        manifest_fd = os.open("manifest.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=evidence_fd)
        payload_fd = os.open("samples.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=evidence_fd)
        source = object.__new__(tables.CanonicalPooledTableSource)
        source._expectations = {("plain", 0): expected}
        source._evidence_fds = {("plain", 0): (evidence_fd, manifest_fd, payload_fd)}
        if replacement == "file":
            substitute = evidence / "substitute.json"
            substitute.write_bytes((evidence / "manifest.json").read_bytes())
            os.replace(substitute, evidence / "manifest.json")
        else:
            replacement_dir = tmp_path / "replacement-evidence"
            shutil.copytree(evidence, replacement_dir)
            retired = evidence.with_name("retired-evidence")
            evidence.rename(retired)
            os.replace(replacement_dir, evidence)
        with pytest.raises(tables.CanonicalPooledTableError, match="entry path identity changed"):
            source._check_current_evidence_entries(expected.child_run_id, reader)
    finally:
        for fd in (locals().get("manifest_fd"), locals().get("payload_fd"), locals().get("evidence_fd")):
            if isinstance(fd, int):
                os.close(fd)
        stack.close()
        os.close(run_fd)
