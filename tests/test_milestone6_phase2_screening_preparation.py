from __future__ import annotations

import hashlib
from collections import Counter
from datetime import UTC, datetime

import pytest

from levelup.experiments.milestone6_phase2_screening import (
    B1,
    B2,
    C,
    build_screening_child_config,
    screening_child_configs,
)
from levelup.experiments.milestone6_phase2_screening_preparation import (
    ScreeningDataManifests,
    build_screening_data_keys,
    build_screening_model_keys,
    build_screening_shared_plan,
)
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import SystemProvenance
from levelup.experiments.runner.storage import (
    ArtifactValidationError,
    RunStore,
    provenance_identity_sha256,
)
from levelup.experiments.runner.training_data_artifacts import (
    TrainingDataArtifactManifest,
    TrainingDataEvidenceManifest,
)

PROVENANCE = SystemProvenance(
    git_commit_sha="0" * 40,
    git_dirty=False,
    python_version="test-python",
    packages={"levelup-bench": "test"},
    installed_packages_sha256="a" * 64,
    os="test-os",
    architecture="test-arch",
    cpu="test-cpu",
    cpu_count=1,
    memory_bytes=1,
    requested_device="cpu",
    resolved_device="cpu",
    requested_torch_threads=1,
    actual_torch_threads=1,
    requested_torch_interop_threads=1,
    actual_torch_interop_threads=1,
    deterministic_algorithms_requested=True,
    deterministic_algorithms_actual=True,
    processes=1,
    captured_at_utc=datetime(2026, 8, 21, tzinfo=UTC),
)
PROVENANCE_SHA256 = provenance_identity_sha256(PROVENANCE)
BASES = (B1, B2, C)
TRAINING_TUPLES = (
    "lr0p003-e120",
    "lr0p003-e180",
    "lr0p01-e120",
    "lr0p01-e180",
)


def _artifact_id(*parts: object) -> str:
    return hashlib.sha256(canonical_json_bytes(parts)).hexdigest()


def _data_manifests(config, data_keys) -> ScreeningDataManifests:
    evidence = {}
    views = {}
    family = config.parameters["heldout_family_id"]
    for replicate, key in data_keys.evidence.items():
        payload_sha256 = _artifact_id("payload", family, replicate)
        body = {
            "schema_version": "runner.training-data-evidence.v1",
            "evidence_key_id": key.key_id,
            "key": key.model_dump(mode="json"),
            "payload_sha256": payload_sha256,
            "payload_bytes": 1,
            "sample_task_ids": key.ordered_training_task_ids,
        }
        evidence_id = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        evidence[replicate] = TrainingDataEvidenceManifest(
            evidence_id=evidence_id,
            **body,
        )
    for identity, key in data_keys.views.items():
        evidence_manifest = evidence[identity[1]]
        body = {
            "schema_version": "runner.training-data-manifest.v1",
            "evidence_id": evidence_manifest.evidence_id,
            "key_id": key.key_id,
            "key": key.model_dump(mode="json"),
            "payload_sha256": evidence_manifest.payload_sha256,
            "payload_bytes": evidence_manifest.payload_bytes,
            "sample_task_ids": key.ordered_training_task_ids,
        }
        artifact_id = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        views[identity] = TrainingDataArtifactManifest(artifact_id=artifact_id, **body)
    return ScreeningDataManifests(evidence=evidence, views=views)


def _plain_preparation():
    config = build_screening_child_config("plain")
    data_keys = build_screening_data_keys(config, PROVENANCE)
    data_manifests = _data_manifests(config, data_keys)
    model_keys = build_screening_model_keys(
        config,
        data_keys,
        data_manifests,
    )
    shared = build_screening_shared_plan(
        config,
        data_keys,
        data_manifests,
        model_keys,
    )
    return config, data_keys, data_manifests, model_keys, shared


def test_plain_fold_converts_lineage_to_exact_keys_and_shared_plan() -> None:
    config, data_keys, data_manifests, model_keys, shared = _plain_preparation()

    assert set(data_keys.evidence) == set(range(5))
    assert set(data_keys.views) == {(base, replicate) for base in BASES for replicate in range(5)}
    assert set(model_keys.models) == {
        (base, training_tuple, replicate)
        for base in BASES
        for training_tuple in TRAINING_TUPLES
        for replicate in range(5)
    }
    assert len(shared.artifacts) == 80
    assert Counter(item.kind for item in shared.artifacts) == {
        "training_data_evidence": 5,
        "training_data_view": 15,
        "training_artifact": 60,
    }

    expected_fold = config.parameters["fold_id"]
    expected_family = config.parameters["heldout_family_id"]
    for replicate, key in data_keys.evidence.items():
        assert key.replicate == replicate
        assert key.fold_id == expected_fold
        assert key.heldout_family_id == expected_family
        assert key.provenance_sha256 == PROVENANCE_SHA256
    for (base, replicate), key in data_keys.views.items():
        assert key.condition_id == base
        assert key.replicate == replicate
        assert key.fold_id == expected_fold
        assert key.heldout_family_id == expected_family
        assert key.provenance_sha256 == PROVENANCE_SHA256
    for (base, training_tuple, replicate), key in model_keys.models.items():
        assert key.condition_id == base
        assert key.training_tuple_id == training_tuple
        assert key.replicate == replicate
        assert key.fold_id == expected_fold
        assert key.heldout_family_id == expected_family
        assert key.provenance_sha256 == PROVENANCE_SHA256
        assert key.training_data_sha256 == data_manifests.views[
            (base, replicate)
        ].artifact_id
        assert "search_temperature" not in key.model_dump(mode="json")

    for item in shared.artifacts:
        assert item.owner_family_id == expected_family
        assert item.owner_fold_id == expected_fold
        assert item.owner_replicate in range(5)
        assert item.owner_condition_id in item.consumer_condition_ids
        assert item.consumer_unit_ids
        assert len(item.consumer_unit_ids) == {
            "training_data_evidence": 288,
            "training_data_view": 96,
            "training_artifact": 24,
        }[item.kind]


def test_all_six_folds_have_exact_inventory_and_disjoint_key_ids() -> None:
    all_data_keys = []
    all_model_keys = []
    all_shared = []
    for config in screening_child_configs():
        data_keys = build_screening_data_keys(config, PROVENANCE)
        data_manifests = _data_manifests(config, data_keys)
        model_keys = build_screening_model_keys(
            config,
            data_keys,
            data_manifests,
        )
        shared = build_screening_shared_plan(
            config,
            data_keys,
            data_manifests,
            model_keys,
        )
        assert len(data_keys.evidence) == 5
        assert len(data_keys.views) == 15
        assert len(model_keys.models) == 60
        assert len(shared.artifacts) == 80
        assert Counter(item.kind for item in shared.artifacts) == {
            "training_data_evidence": 5,
            "training_data_view": 15,
            "training_artifact": 60,
        }
        all_data_keys.extend((*data_keys.evidence.values(), *data_keys.views.values()))
        all_model_keys.extend(model_keys.models.values())
        all_shared.extend(shared.artifacts)

    key_ids = [key.key_id for key in (*all_data_keys, *all_model_keys)]
    assert len(key_ids) == len(set(key_ids)) == 6 * (5 + 15 + 60)
    assert len(all_shared) == 6 * 80
    assert len({item.key_id for item in all_shared}) == len(all_shared)


def test_model_keys_reuse_across_three_temperatures_without_temperature_identity() -> None:
    config, data_keys, _, model_keys, _ = _plain_preparation()
    learned = config.conditions[2:]

    for base in BASES:
        for training_tuple in TRAINING_TUPLES:
            consumers = tuple(
                condition
                for condition in learned
                if condition.parameters["base_condition_id"] == base
                and condition.parameters["training_tuple_id"] == training_tuple
            )
            assert len(consumers) == 3
            assert {condition.parameters["search_temperature"] for condition in consumers} == {
                0.6,
                0.9,
                1.2,
            }
            keys = {
                model_keys.models[(base, training_tuple, replicate)].key_id
                for replicate in range(config.replicates)
            }
            assert len(keys) == 5
            assert all(
                "search_temperature" not in model_keys.models[
                    (base, training_tuple, replicate)
                ].model_dump(mode="json")
                for replicate in range(config.replicates)
            )


@pytest.mark.parametrize("tamper", ("foreign_key", "duplicate_id", "wrong_evidence"))
def test_model_key_builder_rejects_unbound_data_manifests(tamper: str) -> None:
    config = build_screening_child_config("plain")
    data_keys = build_screening_data_keys(config, PROVENANCE)
    manifests = _data_manifests(config, data_keys)
    views = dict(manifests.views)
    left = (B1, 0)
    right = (B2, 0)
    if tamper == "foreign_key":
        views[left] = views[left].model_copy(update={"key": data_keys.views[right]})
    elif tamper == "duplicate_id":
        views[left] = views[left].model_copy(
            update={"artifact_id": views[right].artifact_id}
        )
    else:
        views[left] = views[left].model_copy(
            update={"evidence_id": manifests.evidence[1].evidence_id}
        )
    changed = ScreeningDataManifests(
        evidence=manifests.evidence,
        views=views,
    )

    with pytest.raises(ValueError):
        build_screening_model_keys(config, data_keys, changed)


@pytest.mark.parametrize("tamper", ("unknown_unit", "wrong_replicate", "owner", "conditions"))
def test_run_store_rejects_shared_owner_consumer_tampering(tamper: str, tmp_path) -> None:
    config, _, _, _, shared = _plain_preparation()
    first = shared.artifacts[0]
    if tamper == "unknown_unit":
        changed = first.model_copy(
            update={"consumer_unit_ids": (*first.consumer_unit_ids[:-1], "f" * 64)}
        )
    elif tamper == "wrong_replicate":
        changed = first.model_copy(update={"owner_replicate": 1})
    elif tamper == "owner":
        changed = first.model_copy(update={"owner_condition_id": "A0-no-probe-uniform"})
    else:
        changed = first.model_copy(update={"consumer_condition_ids": (B1,)})
    artifacts = (changed, *shared.artifacts[1:])

    with pytest.raises((ArtifactValidationError, ValueError)):
        RunStore(
            tmp_path,
            config,
            repository=tmp_path,
            shared_artifacts=artifacts,
        )
