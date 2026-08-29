"""Complete-authority tests for the Phase 3 local-affordance raw store.

The fixture deliberately contains the complete frozen development universe (240
task artifacts, 30 LOFO manifests, and 240 held-out bindings).  It is entirely
synthetic: rows contain only the public observable transition schema and no
outcome/final-family data.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from levelup.experiments.milestone6_phase3_local_affordance_evidence import (
    RawProbeArtifactBody,
    RawProbeArtifactKey,
    RawProbeArtifactManifest,
    RawProbeTransitionRecord,
)
from levelup.experiments.milestone6_phase3_local_affordance_raw_store import (
    FAMILY_ORDER,
    HeldoutProbeBinding,
    RawProbeStoreManifest,
    RawProbeTaskKeyIndex,
    RawProbeTaskReference,
    TrainingFoldManifest,
    open_existing_raw_probe_store,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.training_data_artifacts import (
    AffordanceTableRecord,
    ObservableStateRecord,
)
from levelup.learning.state_conditioned import (
    ObservableState,
    ObservedTransition,
    build_affordance_table,
)

authority = pytest.importorskip(
    "levelup.experiments.milestone6_phase3_local_affordance_raw_authority"
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "milestone6"


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _canonical(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _selected_tasks() -> list[dict[str, object]]:
    body = json.loads((CONFIG / "development_tasks.json").read_bytes())
    return [row for row in body["tasks"] if "training_core" in row["roles"]]


def _state(index: int) -> ObservableStateRecord:
    progress = index / 64
    return ObservableStateRecord(
        progress_fraction=progress,
        remaining_fraction=1.0 - progress,
        elapsed_per_target=progress,
        resource_fraction=0.5,
        pressure_fraction=0.2,
        available_aliases=("a",),
    )


def _artifact(key: RawProbeArtifactKey):
    rows = tuple(
        RawProbeTransitionRecord(
            probe_index=index,
            before=_state(index),
            action_alias="a",
            after=_state(index + 1),
            completed=index == 63,
        )
        for index in range(64)
    )
    body = RawProbeArtifactBody.from_rows(rows)
    transitions = tuple(
        ObservedTransition(
            ObservableState(
                row.before.progress_fraction,
                row.before.remaining_fraction,
                row.before.elapsed_per_target,
                row.before.resource_fraction,
                row.before.pressure_fraction,
                row.before.available_aliases,
            ),
            row.action_alias,
            ObservableState(
                row.after.progress_fraction,
                row.after.remaining_fraction,
                row.after.elapsed_per_target,
                row.after.resource_fraction,
                row.after.pressure_fraction,
                row.after.available_aliases,
            ),
            row.completed,
        )
        for row in rows
    )
    table = build_affordance_table(transitions, target_samples_per_alias=8)
    affordances = AffordanceTableRecord(
        features={alias: tuple(values) for alias, values in table.features.items()},
        sample_counts=dict(table.sample_counts),
    )
    manifest = RawProbeArtifactManifest.from_key_body(
        key,
        body,
        pooled_affordance_sha256=_sha(affordances.model_dump(mode="json")),
    )
    return {
        "schema_version": "milestone6.phase3.local-affordance-persisted-artifact.v1",
        "key": key.model_dump(mode="json"),
        "body": body.model_dump(mode="json"),
        "manifest": manifest.model_dump(mode="json"),
        "affordances": affordances.model_dump(mode="json"),
    }, manifest.artifact_id


@pytest.fixture(scope="session")
def expected_authority():
    return authority.build_expected_raw_probe_authority(
        local_affordance_protocol_bytes=(
            CONFIG / "phase3_local_affordance_protocol.json"
        ).read_bytes(),
        development_protocol_bytes=(CONFIG / "development_protocol.json").read_bytes(),
        development_tasks_bytes=(CONFIG / "development_tasks.json").read_bytes(),
        phase3_evidence_lock_bytes=(CONFIG / "phase3_evidence_lock.json").read_bytes(),
    )


@pytest.fixture(scope="session")
def _complete_store_template(tmp_path_factory, expected_authority):
    root = tmp_path_factory.mktemp("raw-authority-template")
    for namespace in ("artifacts", "keys", "training-folds", "heldout-bindings"):
        (root / namespace).mkdir()

    hashes = {
        "local_affordance_protocol_sha256": hashlib.sha256(
            (CONFIG / "phase3_local_affordance_protocol.json").read_bytes()
        ).hexdigest(),
        "development_protocol_sha256": hashlib.sha256(
            (CONFIG / "development_protocol.json").read_bytes()
        ).hexdigest(),
        "development_tasks_sha256": hashlib.sha256(
            (CONFIG / "development_tasks.json").read_bytes()
        ).hexdigest(),
        "phase3_evidence_lock_sha256": json.loads(
            (CONFIG / "phase3_evidence_lock.json").read_bytes()
        )["evidence_lock_sha256"],
        "probe_policy_sha256": "f44950c1d3317acc3d5518675488448c310a6bb15900644f681319677739db20",
    }
    manifest = RawProbeStoreManifest.from_authority_hashes(**hashes)
    (root / "manifest.json").write_bytes(_canonical(manifest.model_dump(mode="json")))

    by_key: dict[tuple[str, int, int], tuple[RawProbeArtifactKey, str]] = {}
    selected = _selected_tasks()
    family_index = {family: i for i, family in enumerate(FAMILY_ORDER)}
    for replicate in range(5):
        for task in selected:
            family = str(task["family"])
            index = int(task["task_index"])
            key = RawProbeArtifactKey(
                **hashes,
                family_id=family,
                replicate=replicate,
                task_index=index,
                task_id=str(task["task_id"]),
                generator_seed=int(task["generator_seed"]),
                probe_seed=6_200_000 + family_index[family] * 10_000 + replicate * 100_000 + index,
                environment_seed=int(task["environment_reset_seed"]),
            )
            artifact, artifact_id = _artifact(key)
            (root / "artifacts" / f"{artifact_id}.json").write_bytes(_canonical(artifact))
            index_row = RawProbeTaskKeyIndex(key_id=key.key_id, artifact_id=artifact_id, key=key)
            (root / "keys" / f"{key.key_id}.json").write_bytes(
                _canonical(index_row.model_dump(mode="json"))
            )
            by_key[(family, replicate, index)] = (key, artifact_id)

    for replicate in range(5):
        for heldout in FAMILY_ORDER:
            refs = []
            for family in FAMILY_ORDER:
                if family == heldout:
                    continue
                for task in selected:
                    if task["family"] != family:
                        continue
                    key, artifact_id = by_key[(family, replicate, int(task["task_index"]))]
                    refs.append(
                        RawProbeTaskReference(artifact_id=artifact_id, key_id=key.key_id, key=key)
                    )
            refs.sort(
                key=lambda ref: (
                    FAMILY_ORDER.index(ref.family_id),
                    ref.task_index,
                    ref.task_id,
                    ref.key_id,
                )
            )
            fold = TrainingFoldManifest(
                fold_id=heldout,
                heldout_family=heldout,
                replicate=replicate,
                task_references=tuple(refs),
            )
            (root / "training-folds" / f"{heldout}.r{replicate}.json").write_bytes(
                _canonical(fold.model_dump(mode="json"))
            )
            for task in selected:
                if task["family"] != heldout:
                    continue
                key, artifact_id = by_key[(heldout, replicate, int(task["task_index"]))]
                binding = HeldoutProbeBinding(
                    fold_id=heldout,
                    family_id=heldout,
                    replicate=replicate,
                    task_reference=RawProbeTaskReference(
                        artifact_id=artifact_id, key_id=key.key_id, key=key
                    ),
                )
                filename = f"{heldout}.r{replicate}.task-{int(task['task_index'])}.json"
                (root / "heldout-bindings" / filename).write_bytes(
                    _canonical(binding.model_dump(mode="json"))
                )
    return root


@pytest.fixture
def complete_store(tmp_path, _complete_store_template):
    root = tmp_path / "raw-authority"
    shutil.copytree(_complete_store_template, root)
    return root


def _validate(root: Path, expected_authority):
    with open_existing_raw_probe_store(root) as reader:
        return authority.validate_complete_raw_probe_authority(reader, expected=expected_authority)


def test_expected_authority_rejects_coherent_source_byte_substitution() -> None:
    with pytest.raises(authority.RawProbeAuthorityError, match="frozen commits"):
        authority.build_expected_raw_probe_authority(
            local_affordance_protocol_bytes=(
                CONFIG / "phase3_local_affordance_protocol.json"
            ).read_bytes()
            + b"\n",
            development_protocol_bytes=(CONFIG / "development_protocol.json").read_bytes(),
            development_tasks_bytes=(CONFIG / "development_tasks.json").read_bytes(),
            phase3_evidence_lock_bytes=(CONFIG / "phase3_evidence_lock.json").read_bytes(),
        )


def test_complete_authority_validates_and_snapshot_has_no_capability(
    complete_store, expected_authority
):
    snapshot = _validate(complete_store, expected_authority)
    assert isinstance(snapshot, authority.RawProbeAuthoritySnapshot)
    assert authority.require_raw_probe_authority_snapshot(snapshot) is snapshot
    with pytest.raises(authority.RawProbeAuthorityError, match="complete validator"):
        authority.RawProbeAuthoritySnapshot(**snapshot.model_dump(mode="python"))
    copied = snapshot.model_copy(update={"evidence_lock_file_sha256": "0" * 64})
    with pytest.raises(authority.RawProbeAuthorityError, match="validator-issued"):
        authority.require_raw_probe_authority_snapshot(copied)
    assert not hasattr(snapshot, "reducer_capability")
    assert not hasattr(snapshot, "lookup")
    assert snapshot.manifest.raw_artifact_count == 240
    assert snapshot.manifest.execution_authorized is False
    assert (
        snapshot.evidence_lock_file_sha256
        == hashlib.sha256((CONFIG / "phase3_evidence_lock.json").read_bytes()).hexdigest()
    )
    assert (
        snapshot.manifest.phase3_evidence_lock_sha256
        == json.loads((CONFIG / "phase3_evidence_lock.json").read_bytes())["evidence_lock_sha256"]
    )
    assert snapshot.evidence_lock_file_sha256 != snapshot.manifest.phase3_evidence_lock_sha256
    assert len(snapshot.authority_content_sha256) == 64


@pytest.mark.parametrize("namespace", ["artifacts", "keys", "training-folds", "heldout-bindings"])
@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_exact_inventory_missing_and_extra_rejected(
    complete_store, expected_authority, namespace, operation
):
    if operation == "missing":
        entry = next((complete_store / namespace).iterdir())
        entry.unlink()
    else:
        (complete_store / namespace / "extra.json").write_bytes(_canonical({"extra": True}))
    with pytest.raises(authority.RawProbeAuthorityError):
        _validate(complete_store, expected_authority)


def test_hash_mixing_rejected(complete_store, expected_authority):
    target = next((complete_store / "keys").iterdir())
    value = json.loads(target.read_bytes())
    value["key"]["probe_policy_sha256"] = "0" * 64
    target.write_bytes(_canonical(value))
    with pytest.raises(authority.RawProbeAuthorityError):
        _validate(complete_store, expected_authority)


def test_cross_reference_permutation_rejected(complete_store, expected_authority):
    paths = sorted((complete_store / "keys").iterdir())[:2]
    left, right = (json.loads(path.read_bytes()) for path in paths)
    left["artifact_id"], right["artifact_id"] = right["artifact_id"], left["artifact_id"]
    paths[0].write_bytes(_canonical(left))
    paths[1].write_bytes(_canonical(right))
    with pytest.raises(authority.RawProbeAuthorityError):
        _validate(complete_store, expected_authority)


@pytest.mark.parametrize("part", ["body", "manifest", "affordances"])
def test_artifact_envelope_corruption_rejected(complete_store, expected_authority, part):
    path = next((complete_store / "artifacts").iterdir())
    value = json.loads(path.read_bytes())
    if part == "body":
        value[part]["rows"][0]["action_alias"] = "x"
    elif part == "manifest":
        value[part]["artifact_id"] = "f" * 64
    else:
        value[part]["sample_counts"]["a"] = 63
    path.write_bytes(_canonical(value))
    with pytest.raises(authority.RawProbeAuthorityError):
        _validate(complete_store, expected_authority)


def test_sparse_task_identity_and_seed_mismatch_rejected(complete_store, expected_authority):
    path = next((complete_store / "keys").iterdir())
    value = json.loads(path.read_bytes())
    value["key"]["task_index"] = 999
    path.write_bytes(_canonical(value))
    with pytest.raises(authority.RawProbeAuthorityError):
        _validate(complete_store, expected_authority)


def test_filename_mismatch_rejected(complete_store, expected_authority):
    path = next((complete_store / "heldout-bindings").iterdir())
    renamed = path.with_name(path.name.replace("task-", "task-999-", 1))
    path.rename(renamed)
    with pytest.raises(authority.RawProbeAuthorityError):
        _validate(complete_store, expected_authority)


def test_directory_substitution_rejected(complete_store, expected_authority):
    replacement = complete_store.parent / "replacement-ns"
    replacement.mkdir()
    for namespace in ("artifacts", "keys", "training-folds", "heldout-bindings"):
        (replacement / namespace).mkdir()
    old = complete_store.parent / "old-authority"
    complete_store.rename(old)
    replacement.rename(complete_store)
    with pytest.raises(authority.RawProbeAuthorityError):
        _validate(complete_store, expected_authority)
