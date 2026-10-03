from __future__ import annotations

import copy
import hashlib
import json

import pytest
from test_milestone6_phase3_anchor import (
    _patch_fake_task_ids,
    _protocol_for_runtime,
    _runtime,
    _skip_unit_validation,
)

from levelup.experiments.milestone6_phase3_anchor import (
    AnchorManifestError,
    build_phase3_anchor_manifest,
    require_phase3_anchor_manifest,
    validate_committed_phase3_anchor_for_local_affordance_preparation,
)
from levelup.experiments.runner.config import canonical_json_bytes


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _anchor_for_runtime(monkeypatch: pytest.MonkeyPatch):
    runtime, result_bytes = _runtime()
    _patch_fake_task_ids(monkeypatch)
    monkeypatch.setattr(
        "levelup.experiments.milestone6_phase3_anchor._validate_unit_bytes",
        _skip_unit_validation,
    )
    protocol = _protocol_for_runtime(runtime)
    monkeypatch.setattr(
        "levelup.experiments.milestone6_phase3_anchor.load_phase3_protocol",
        lambda: protocol,
    )
    anchor = build_phase3_anchor_manifest(
        runtime,
        protocol=protocol,
        result_bytes_reader=lambda _store, unit_id: result_bytes[unit_id],
        _allow_test_reader=True,
    )
    monkeypatch.setattr(
        "levelup.experiments.milestone6_phase3_anchor.load_committed_phase3_anchor_manifest_bytes",
        lambda: anchor.canonical_bytes,
    )
    lock_body = {
        "schema_version": "milestone6.phase3.evidence-lock.v1",
        "scope": "known-development-only",
        "final_family_access": False,
        "lineage": {
            "phase3_protocol_sha256": protocol.sha256,
            "phase3_anchor_manifest_sha256": anchor.anchor_manifest_sha256,
            "phase3_anchor_file_sha256": _sha_bytes(anchor.canonical_bytes),
        },
    }
    lock_body["evidence_lock_sha256"] = _sha_bytes(canonical_json_bytes(lock_body))
    lock_bytes = canonical_json_bytes(lock_body)
    monkeypatch.setattr(
        "levelup.experiments.milestone6_phase3_evidence.load_committed_phase3_evidence_lock_bytes",
        lambda: lock_bytes,
    )
    return runtime, protocol, anchor, lock_bytes


def test_preparation_anchor_validation_never_reads_phase2_unit_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _protocol, anchor, lock_bytes = _anchor_for_runtime(monkeypatch)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Phase 2 unit-result payload reader was invoked")

    monkeypatch.setattr(
        "levelup.experiments.milestone6_phase3_anchor._fold_result_bytes_reader",
        forbidden,
    )
    for fold in runtime.folds:
        fold.store.attempt_records = forbidden
    validated = validate_committed_phase3_anchor_for_local_affordance_preparation(
        anchor.canonical_bytes,
        runtime=runtime,
        evidence_lock_bytes=lock_bytes,
    )

    assert validated.canonical_bytes == anchor.canonical_bytes
    assert validated.anchor_manifest_sha256 == anchor.anchor_manifest_sha256
    assert require_phase3_anchor_manifest(validated) is validated


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("anchor", "committed Phase 3 authority"),
        ("lineage", "in-memory Phase 2 authority"),
        ("final", "scope or schema"),
    ),
)
def test_preparation_anchor_validation_rejects_mutated_anchor_lineage_and_final_scope(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    runtime, _protocol, anchor, lock_bytes = _anchor_for_runtime(monkeypatch)
    body = copy.deepcopy(anchor.body)
    if mutation == "anchor":
        body["counts"]["unit_results"] -= 1
    elif mutation == "lineage":
        body["lineage"]["phase2_tree_sha256"] = "0" * 64
    else:
        body["final_family_access"] = True
    unsigned = dict(body)
    unsigned.pop("anchor_manifest_sha256", None)
    body["anchor_manifest_sha256"] = _sha_bytes(canonical_json_bytes(unsigned))
    altered = canonical_json_bytes(body)
    monkeypatch.setattr(
        "levelup.experiments.milestone6_phase3_anchor.load_committed_phase3_anchor_manifest_bytes",
        lambda: altered,
    )

    if mutation == "anchor":
        with pytest.raises(AnchorManifestError, match="counts drifted"):
            validate_committed_phase3_anchor_for_local_affordance_preparation(
                altered,
                runtime=runtime,
                evidence_lock_bytes=lock_bytes,
            )
    else:
        with pytest.raises(AnchorManifestError, match=message):
            validate_committed_phase3_anchor_for_local_affordance_preparation(
                altered,
                runtime=runtime,
                evidence_lock_bytes=lock_bytes,
            )


def test_evidence_lock_must_be_committed_and_bind_anchor_file_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _protocol, anchor, lock_bytes = _anchor_for_runtime(monkeypatch)
    validated = validate_committed_phase3_anchor_for_local_affordance_preparation(
        anchor.canonical_bytes,
        runtime=runtime,
        evidence_lock_bytes=lock_bytes,
    )
    assert validated.anchor_manifest_sha256 == anchor.anchor_manifest_sha256

    changed_body = json.loads(lock_bytes)
    changed_body["lineage"]["phase3_anchor_file_sha256"] = "f" * 64
    unsigned = dict(changed_body)
    unsigned.pop("evidence_lock_sha256")
    changed_body["evidence_lock_sha256"] = _sha_bytes(canonical_json_bytes(unsigned))
    changed = canonical_json_bytes(changed_body)
    monkeypatch.setattr(
        "levelup.experiments.milestone6_phase3_evidence.load_committed_phase3_evidence_lock_bytes",
        lambda: changed,
    )
    with pytest.raises(AnchorManifestError, match="bound to this committed anchor"):
        validate_committed_phase3_anchor_for_local_affordance_preparation(
            anchor.canonical_bytes,
            runtime=runtime,
            evidence_lock_bytes=changed,
        )


def test_preparation_anchor_validation_requires_evidence_lock_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _protocol, anchor, _lock_bytes = _anchor_for_runtime(monkeypatch)
    with pytest.raises(TypeError):
        validate_committed_phase3_anchor_for_local_affordance_preparation(
            anchor.canonical_bytes,
            runtime=runtime,
        )
