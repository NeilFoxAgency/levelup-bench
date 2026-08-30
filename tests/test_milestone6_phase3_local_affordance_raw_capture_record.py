from __future__ import annotations

import hashlib
import json
from pathlib import Path

from levelup.experiments.milestone6_phase3_local_affordance_raw_authority import (
    build_expected_raw_probe_authority,
)
from levelup.experiments.milestone6_phase3_local_affordance_readiness import (
    SOURCE_RELATIVE_PATHS,
)

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "experiments" / "milestone6_phase3_local_affordance_raw_capture.json"


def test_committed_raw_capture_record_is_canonical_development_only_provenance() -> None:
    encoded = RECORD.read_bytes()
    body = json.loads(encoded)
    assert encoded == json.dumps(body, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert body["schema_version"] == (
        "milestone6.phase3.local-affordance-raw-capture-summary.v1"
    )
    assert body["scope"] == "known-development-only"
    assert body["status"] == "complete-with-accounting-metadata-loss"
    assert body["activation_git_commit"] == "59aac16f3a87ea73946159a4bd811e0d400a5554"
    assert body["source_phase2_raw_root"] == (
        "runs/milestone6/phase2-screening-readiness-9daa444"
    )
    assert body["destination"] == (
        "runs/milestone6/phase3-local-affordance-raw-development-59aac16"
    )
    assert body["source_readiness_manifest_sha256"] == (
        "ee2cd37c0981b459237bc8691511ed6e048863cdcf5aa04bc7f0713726ef1109"
    )
    assert body["raw_authority_content_sha256"] == (
        "909f2724a5723c53e57d965a82b4350fc3bc9a690c671310a964f7a3ecd561a3"
    )
    assert body["raw_manifest_file_sha256"] == (
        "7d178620426747d66722c81e4e9e2de047d652a30ad965212ce8990ab07e9cc1"
    )
    assert body["ordered_artifact_ids_sha256"] == (
        "549ecd7db26cd646fd683ca48aca40b3a67566ab4f46bb7a39010ce0634c9e43"
    )
    assert (
        body["artifact_count"],
        body["key_count"],
        body["training_fold_count"],
        body["heldout_binding_count"],
    ) == (240, 240, 30, 240)
    assert body["physical_probe_actions"] == 240 * 64 == 15_360
    assert body["logical_consumer_equivalent_actions"] == 11_520 * 64 == 737_280
    assert body["probe_attempts"] is body["probe_resets"] is body["probe_wall_seconds"] is None
    assert body["accounting_metadata_loss_reason"]
    assert (
        body["training_actions"],
        body["search_actions"],
        body["replay_actions"],
        body["evaluator_calls"],
        body["oracle_calls"],
    ) == (0, 0, 0, 0, 0)
    assert body["comparative_results_generated"] is False
    assert body["comparative_results_inspected"] is False
    assert body["final_family_access"] is False

    sources = tuple((ROOT / relative).read_bytes() for relative in SOURCE_RELATIVE_PATHS)
    expected = build_expected_raw_probe_authority(
        local_affordance_protocol_bytes=sources[0],
        development_protocol_bytes=sources[1],
        development_tasks_bytes=sources[2],
        phase3_evidence_lock_bytes=sources[3],
    )
    assert body["raw_authority_manifest_id"] == expected.manifest.manifest_id
    ordered_key_ids = tuple(key.key_id for key in expected.keys)
    assert body["ordered_key_ids_sha256"] == hashlib.sha256(
        json.dumps(ordered_key_ids, separators=(",", ":")).encode()
    ).hexdigest()
