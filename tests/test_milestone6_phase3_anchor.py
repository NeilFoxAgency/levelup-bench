from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from levelup.experiments.milestone6_phase3_anchor import (
    CANDIDATE_TUPLE_IDS,
    AnchorManifestError,
    build_phase3_anchor_manifest,
    validate_phase3_anchor_manifest,
)
from levelup.experiments.milestone6_phase3_protocol import (
    Phase3ProtocolSnapshot,
    load_phase3_protocol,
)
from levelup.experiments.runner.config import canonical_json_bytes

FAMILIES = ("plain", "battery", "cooldown", "heat", "momentum", "combo")
BASES = (
    "B2-global-listwise-optimum",
    "C-state-conditioned-listwise-optimum",
)
TUPLES = ("lr0p003-e120", "lr0p003-e180", "lr0p01-e120", "lr0p01-e180")


def _skip_unit_validation(
    _raw: bytes, _unit_id: str, _planned: object, _store: object
) -> None:
    return None


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _patch_fake_task_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "levelup.experiments.milestone6_phase3_anchor._canonical_tasks_by_family",
        lambda: {
            family: tuple((f"{family}-task-{index}", index) for index in range(8))
            for family in FAMILIES
        },
    )


def _runtime() -> tuple[SimpleNamespace, dict[str, bytes]]:
    result_bytes: dict[str, bytes] = {}
    folds = []
    for family_index, family in enumerate(FAMILIES):
        conditions = []
        planned = []
        records = []
        for base in BASES:
            for candidate_tuple_id in CANDIDATE_TUPLE_IDS:
                condition_id = f"{base}--{candidate_tuple_id}"
                conditions.append(
                    SimpleNamespace(
                        condition_id=condition_id,
                        parameters={
                            "base_condition_id": base,
                            "candidate_tuple_id": candidate_tuple_id,
                        },
                        execution_phases=("validation",),
                    )
                )
                for task_index in range(8):
                    for replicate in range(5):
                        unit_id = _sha(
                            f"{family}:{base}:{candidate_tuple_id}:{task_index}:{replicate}"
                        )
                        key = SimpleNamespace(
                            condition_id=condition_id,
                            family_id=family,
                            task_id=f"{family}-task-{task_index}",
                            task_index=task_index,
                            replicate=replicate,
                            phase="validation",
                        )
                        planned_unit = SimpleNamespace(unit_id=unit_id, key=key)
                        planned.append(planned_unit)
                        records.append(SimpleNamespace(unit_id=unit_id, status="completed"))
                        result_bytes[unit_id] = f"raw:{unit_id}".encode()

        model_keys = {}
        manifests = {}
        costs = {}
        computes = {}
        for base in BASES:
            for tuple_id in TUPLES:
                for replicate in range(5):
                    identity = (base, tuple_id, replicate)
                    key_id = _sha(f"key:{family}:{base}:{tuple_id}:{replicate}")
                    artifact_id = _sha(f"artifact:{family}:{base}:{tuple_id}:{replicate}")
                    cost_id = _sha(f"cost:{family}:{base}:{tuple_id}:{replicate}")
                    model_keys[identity] = SimpleNamespace(key_id=key_id)
                    manifests[identity] = SimpleNamespace(artifact_id=artifact_id)
                    costs[identity] = SimpleNamespace(
                        key_id=key_id,
                        artifact_id=artifact_id,
                        cost_id=cost_id,
                        expected_cost_id=cost_id,
                    )
                    computes[identity] = SimpleNamespace(
                        trainable_parameters=(3601 if base == BASES[0] else 3841),
                        optimizer_steps=120,
                        forward_passes=240,
                    )

        store = SimpleNamespace(
            run_id=f"run-{family}",
            expected=SimpleNamespace(units=tuple(planned)),
            models=SimpleNamespace(),
            _execution_ready=False,
            completed_records=lambda records=tuple(records): records,
            attempt_records=lambda: (),
        )
        fold = SimpleNamespace(
            family_id=family,
            config=SimpleNamespace(
                conditions=tuple(conditions),
                split=SimpleNamespace(
                    final_tasks=(),
                    validation_tasks=tuple(
                        SimpleNamespace(
                            task_id=f"{family}-task-{index}", task_index=index
                        )
                        for index in range(8)
                    ),
                ),
            ),
            store=store,
            model_keys=SimpleNamespace(models=model_keys),
            models=SimpleNamespace(
                manifests=manifests,
                costs=costs,
                compute=computes,
            ),
        )
        folds.append(fold)

    manifest = SimpleNamespace(
        family_order=FAMILIES,
        development_only=True,
        final_family_access=False,
        validation_executed=False,
        search_executed=False,
        selection_performed=False,
        manifest_sha256=_sha("readiness"),
        provenance_sha256=_sha("provenance"),
        child_run_ids=tuple(f"run-{family}" for family in FAMILIES),
    )
    runtime = SimpleNamespace(
        manifest=manifest,
        folds=tuple(folds),
        raw_root_identity=(1, 2),
        child_identities=tuple(
            (f"run-{family}", (1, index + 10))
            for index, family in enumerate(FAMILIES)
        ),
        result_namespace_snapshot=(("run", ()),),
        manifest_bytes=b"readiness-bytes",
        tree_sha256=_sha("tree"),
        manifest_parent_identity=(1, 90),
        manifest_file_identity=(1, 91),
        authority_sources=(
            SimpleNamespace(
                parent_identity=(1, 92),
                file_identity=(1, 93),
                content=b"authority",
                sha256=_sha("authority"),
            ),
        ),
        provenance=SimpleNamespace(identity="provenance"),
    )
    return runtime, result_bytes


def _protocol_for_runtime(runtime: SimpleNamespace) -> Phase3ProtocolSnapshot:
    source = load_phase3_protocol()
    payload = copy.deepcopy(source.payload)
    gate = payload["canonical_evidence_reuse"]["anchor_lineage_gate"]
    gate["phase2_readiness_manifest_sha256"] = runtime.manifest.manifest_sha256
    gate["phase2_readiness_manifest_bytes_sha256"] = hashlib.sha256(
        runtime.manifest_bytes
    ).hexdigest()
    snapshot_sha = hashlib.sha256(
        canonical_json_bytes(runtime.result_namespace_snapshot)
    ).hexdigest()
    gate["phase2_result_namespace_snapshot_sha256"] = snapshot_sha
    authority_bytes = dict(source.authority_bytes)
    selection = json.loads(authority_bytes["phase2_selection_lock"])
    selection["authority"]["readiness_manifest_sha256"] = gate[
        "phase2_readiness_manifest_sha256"
    ]
    selection["authority"]["readiness_manifest_bytes_sha256"] = gate[
        "phase2_readiness_manifest_bytes_sha256"
    ]
    selection["authority"]["prepared_tree_sha256"] = runtime.tree_sha256
    selection["authority"]["source_provenance_sha256"] = (
        runtime.manifest.provenance_sha256
    )
    selection["analysis"]["result_namespace_snapshot_sha256"] = snapshot_sha
    authority_bytes["phase2_selection_lock"] = canonical_json_bytes(selection)
    content = canonical_json_bytes(payload)
    return replace(
        source,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        payload=payload,
        authority_bytes=tuple(authority_bytes.items()),
    )


def test_anchor_manifest_is_exact_and_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, result_bytes = _runtime()
    monkeypatch.setattr(
        "levelup.experiments.milestone6_phase3_anchor._validate_unit_bytes",
        _skip_unit_validation,
    )
    _patch_fake_task_ids(monkeypatch)
    protocol = _protocol_for_runtime(runtime)
    def reader(_store: object, unit_id: str) -> bytes:
        return result_bytes[unit_id]
    first = build_phase3_anchor_manifest(
        runtime,
        protocol=protocol,
        result_bytes_reader=reader,
        _allow_test_reader=True,
    )
    second = build_phase3_anchor_manifest(
        runtime,
        protocol=protocol,
        result_bytes_reader=reader,
        _allow_test_reader=True,
    )
    assert first.canonical_bytes == second.canonical_bytes
    assert first.anchor_manifest_sha256 == second.anchor_manifest_sha256
    assert first.body["counts"] == {
        "families": 6,
        "anchor_base_conditions": 2,
        "model_owners": 240,
        "unit_results": 5760,
    }
    assert first.body["t_alias"]["historical_condition_id"] == BASES[1]
    assert (
        validate_phase3_anchor_manifest(
            first,
            runtime=runtime,
            protocol=protocol,
            result_bytes_reader=reader,
            _allow_test_reader=True,
        ).canonical_bytes
        == first.canonical_bytes
    )


def test_anchor_manifest_rejects_drift_and_execution_ready_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, result_bytes = _runtime()
    protocol = _protocol_for_runtime(runtime)
    monkeypatch.setattr(
        "levelup.experiments.milestone6_phase3_anchor._validate_unit_bytes",
        _skip_unit_validation,
    )
    _patch_fake_task_ids(monkeypatch)
    def reader(_store: object, unit_id: str) -> bytes:
        return result_bytes[unit_id]
    manifest = build_phase3_anchor_manifest(
        runtime,
        protocol=protocol,
        result_bytes_reader=reader,
        _allow_test_reader=True,
    ).model_dump()
    mutated = dict(manifest)
    mutated["unit_results"] = list(manifest["unit_results"])
    mutated["unit_results"][0] = dict(mutated["unit_results"][0])
    mutated["unit_results"][0]["result_bytes_sha256"] = _sha("changed")
    with pytest.raises(AnchorManifestError, match="self-hash"):
        validate_phase3_anchor_manifest(mutated, runtime=runtime, protocol=protocol)

    rehashed = manifest.copy()
    rehashed["unit_results"] = list(manifest["unit_results"])
    rehashed["unit_results"][0] = dict(rehashed["unit_results"][0])
    rehashed["unit_results"][0]["run_id"] = "substituted-run"
    unsigned = dict(rehashed)
    unsigned.pop("anchor_manifest_sha256")
    rehashed["anchor_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(AnchorManifestError, match="locked runtime"):
        validate_phase3_anchor_manifest(
            rehashed,
            runtime=runtime,
            protocol=protocol,
            result_bytes_reader=reader,
            _allow_test_reader=True,
        )

    runtime.folds[0].store._execution_ready = True
    with pytest.raises(AnchorManifestError, match="execution-ready"):
        build_phase3_anchor_manifest(
            runtime,
            protocol=protocol,
            result_bytes_reader=reader,
            _allow_test_reader=True,
        )


def test_anchor_manifest_rejects_wrong_frozen_runtime_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, result_bytes = _runtime()
    monkeypatch.setattr(
        "levelup.experiments.milestone6_phase3_anchor._validate_unit_bytes",
        _skip_unit_validation,
    )

    def reader(_store: object, unit_id: str) -> bytes:
        return result_bytes[unit_id]

    with pytest.raises(AnchorManifestError, match="custom result-byte readers are test-only"):
        build_phase3_anchor_manifest(runtime, result_bytes_reader=reader)
    with pytest.raises(AnchorManifestError, match="frozen anchor lineage changed"):
        build_phase3_anchor_manifest(
            runtime,
            result_bytes_reader=reader,
            _allow_test_reader=True,
        )
