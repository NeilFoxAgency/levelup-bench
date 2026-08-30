"""Unit tests for the all-or-nothing raw local-affordance capture bridge.

These are deliberately fixture-only: they exercise capability and accounting
boundaries without opening a Phase 2 payload, a raw-store destination, or a
development environment implementation.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from levelup.experiments import milestone6_phase3_local_affordance_raw_capture as capture


@dataclass(frozen=True)
class _Key:
    family_id: str
    replicate: int
    task_index: int
    task_id: str
    generator_seed: int
    probe_seed: int
    environment_seed: int = 0
    local_affordance_protocol_sha256: str = "a" * 64
    development_protocol_sha256: str = "b" * 64
    development_tasks_sha256: str = "c" * 64
    phase3_evidence_lock_sha256: str = "d" * 64
    probe_policy_sha256: str = "e" * 64

    @property
    def key_id(self) -> str:
        return f"key-{self.family_id}-{self.replicate}-{self.task_index}"


@dataclass(frozen=True)
class _Manifest:
    artifact_id: str = ""
    manifest_id: str = "a" * 64


@dataclass(frozen=True)
class _Snapshot:
    manifest: _Manifest = _Manifest()
    authority_content_sha256: str = "b" * 64
    manifest_file: object = field(
        default_factory=lambda: SimpleNamespace(
            snapshot=SimpleNamespace(sha256="c" * 64)
        )
    )


@dataclass(frozen=True)
class _Sanitized:
    key: _Key
    body: object = object()
    affordances: object = object()

    @property
    def manifest(self) -> _Manifest:
        return _Manifest(f"artifact-{self.key.key_id}")


@dataclass(frozen=True)
class _Persisted:
    key: _Key
    body: object
    manifest: _Manifest
    affordances: object


class _Lease:
    def __init__(self, keys: tuple[_Key, ...]) -> None:
        self.authority = SimpleNamespace(manifest=_Manifest(), keys=keys)
        self.git_commit_sha = "f" * 40
        self.calls = 0

    def require_active(self) -> "_Lease":
        self.calls += 1
        return self


class _Tables:
    def __init__(self, lease: _Lease) -> None:
        self._lease = lease
        self.calls = 0
        self.requested: list[_Key] = []

    def require_active(self) -> "_Tables":
        self.calls += 1
        return self

    def table_for(self, key: _Key) -> object:
        self.requested.append(key)
        return object()


@dataclass(frozen=True)
class _ProbeAccounting:
    attempts: int = 4
    resets: int = 4
    actions: int = 64
    wall_seconds: float = 0.25


@dataclass(frozen=True)
class _Evidence:
    accounting: _ProbeAccounting = _ProbeAccounting()


def _keys() -> tuple[_Key, ...]:
    return tuple(
        _Key(
            family_id=family,
            replicate=replicate,
            task_index=task_index,
            task_id=f"{family}-{task_index}",
            generator_seed=1_000 + task_index,
            probe_seed=6_200_000 + family_index * 10_000 + replicate * 100_000 + task_index,
        )
        for family_index, family in enumerate(("plain", "battery", "cooldown", "heat", "momentum", "combo"))
        for replicate in range(5)
        for task_index in range(8)
    )


@pytest.fixture
def patched_capture(monkeypatch: pytest.MonkeyPatch) -> tuple[_Lease, _Tables, list[tuple[str, tuple[object, ...], dict[str, object]]], list[object]]:
    keys = _keys()
    lease = _Lease(keys)
    tables = _Tables(lease)
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    published: list[object] = []

    monkeypatch.setattr(capture, "LocalAffordanceActivationLease", _Lease)
    monkeypatch.setattr(capture, "CanonicalPooledTableSource", _Tables)
    monkeypatch.setattr(capture, "RawProbeArtifactKey", _Key)
    monkeypatch.setattr(capture, "SanitizedRawProbeArtifact", _Sanitized)
    monkeypatch.setattr(capture, "PersistedRawProbeArtifact", _Persisted)
    monkeypatch.setattr(capture, "require_expected_raw_probe_authority", lambda authority: authority)
    monkeypatch.setattr(capture, "require_raw_probe_authority_snapshot", lambda snapshot: snapshot)

    def environment(*args: object) -> object:
        calls.append(("combo", args, {}))
        index = int(args[0])
        return SimpleNamespace(
            task_spec=SimpleNamespace(
                task_id=f"combo-{index}",
                constraints=(
                    SimpleNamespace(
                        verifier_id="never_use_action", verifier_config={"action": "forbidden"}
                    ),
                ),
            )
        )

    def adaptive(*args: object) -> object:
        calls.append(("adaptive", args, {}))
        family, index, _seed = args
        return SimpleNamespace(
            task_spec=SimpleNamespace(
                task_id=f"{family}-{index}",
                constraints=(
                    SimpleNamespace(
                        verifier_id="never_use_action", verifier_config={"action": "forbidden"}
                    ),
                ),
            )
        )

    def discover(*args: object, **kwargs: object) -> _Evidence:
        calls.append(("probe", args, kwargs))
        return _Evidence()

    def sanitize(_evidence: object, **kwargs: object) -> _Sanitized:
        calls.append(("sanitize", (), kwargs))
        return _Sanitized(
            kwargs["canonical_affordances"]
            and next(
                key
                for key in keys
                if (
                    key.family_id == kwargs["family_id"]
                    and key.replicate == kwargs["replicate"]
                    and key.task_index == kwargs["task_index"]
                    and key.task_id == kwargs["task_id"]
                    and key.generator_seed == kwargs["generator_seed"]
                    and key.probe_seed == kwargs["probe_seed"]
                    and key.environment_seed == kwargs["environment_seed"]
                )
            )
        )

    def publish(_lease: object, *, artifacts: tuple[_Persisted, ...]) -> object:
        published.append((_lease, artifacts))
        return _Snapshot()

    monkeypatch.setattr(capture, "make_combo_track", environment)
    monkeypatch.setattr(capture, "make_adaptive_track", adaptive)
    monkeypatch.setattr(capture, "discover_affordances", discover)
    monkeypatch.setattr(capture, "sanitize_probe_evidence", sanitize)
    monkeypatch.setattr(capture, "publish_raw_probe_store_from_readiness", publish)
    return lease, tables, calls, published


def test_capture_uses_frozen_key_order_budgets_factories_and_single_publication(
    patched_capture: tuple[_Lease, _Tables, list[tuple[str, tuple[object, ...], dict[str, object]]], list[object]],
) -> None:
    lease, tables, calls, published = patched_capture

    summary = capture.capture_and_publish_raw_probe_store(lease, tables)

    probes = [item for item in calls if item[0] == "probe"]
    factories = [item for item in calls if item[0] in {"combo", "adaptive"}]
    assert len(probes) == len(factories) == len(tables.requested) == 240
    assert tuple(tables.requested) == lease.authority.keys
    assert all(
        probe[2]["action_cap"] == 64
        and probe[2]["target_samples_per_alias"] == 8
        and probe[2]["actions_per_attempt"] == 16
        and probe[2]["seed"] == key.probe_seed
        and probe[2]["task_id"] == key.task_id
        and probe[2]["forbidden_aliases"] == frozenset({"forbidden"})
        for probe, key in zip(probes, lease.authority.keys, strict=True)
    )
    assert all(
        (factory[0] == "combo" and factory[1] == (key.task_index, key.generator_seed))
        or (
            factory[0] == "adaptive"
            and factory[1] == (key.family_id, key.task_index, key.generator_seed)
        )
        for factory, key in zip(factories, lease.authority.keys, strict=True)
    )
    assert len(published) == 1
    published_lease, artifacts = published[0]
    assert published_lease is lease
    assert type(artifacts) is tuple
    assert tuple(artifact.key for artifact in artifacts) == lease.authority.keys
    assert lease.calls == tables.calls == 2
    assert summary.manifest_id == "a" * 64
    assert summary.authority_content_sha256 == "b" * 64
    assert summary.manifest_file_sha256 == "c" * 64
    assert summary.activation_git_commit == "f" * 40
    assert summary.physical_probe_actions == 15_360
    assert summary.logical_consumer_equivalent_actions == 737_280
    assert summary.probe_attempts == summary.probe_resets == 960
    assert summary.probe_wall_seconds == 60.0
    assert (
        summary.training_actions,
        summary.search_actions,
        summary.replay_actions,
        summary.evaluator_calls,
        summary.oracle_calls,
    ) == (0, 0, 0, 0, 0)


def test_sanitized_identity_mismatch_in_family_or_task_index_blocks_publication(
    patched_capture: tuple[_Lease, _Tables, list[tuple[str, tuple[object, ...], dict[str, object]]], list[object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, tables, calls, published = patched_capture
    original = capture.sanitize_probe_evidence

    def wrong_identity(evidence: object, **kwargs: object) -> _Sanitized:
        result = original(evidence, **kwargs)
        key = result.key
        if len([item for item in calls if item[0] == "sanitize"]) == 1:
            key = _Key(
                family_id="wrong-family",
                replicate=key.replicate,
                task_index=key.task_index + 100,
                task_id=key.task_id,
                generator_seed=key.generator_seed,
                probe_seed=key.probe_seed,
            )
        return _Sanitized(key=key)

    monkeypatch.setattr(capture, "sanitize_probe_evidence", wrong_identity)
    with pytest.raises(capture.RawProbeCaptureError, match="key differs"):
        capture.capture_and_publish_raw_probe_store(lease, tables)
    assert published == []


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra"])
def test_invalid_authority_is_rejected_before_environment_construction(
    patched_capture: tuple[_Lease, _Tables, list[tuple[str, tuple[object, ...], dict[str, object]]], list[object]],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    lease, tables, calls, published = patched_capture
    keys = list(lease.authority.keys)
    if mutation == "missing":
        keys = keys[:-1]
    elif mutation == "duplicate":
        keys[-1] = keys[-2]
    else:
        keys.append(keys[0])
    lease.authority = SimpleNamespace(
        manifest=_Manifest(), keys=tuple(keys)
    )

    def validating_authority(authority: object) -> object:
        candidate = tuple(getattr(authority, "keys"))
        if len(candidate) != 240 or len({key.key_id for key in candidate}) != 240:
            raise capture.RawProbeAuthorityError("fixture authority key universe is invalid")
        return authority

    monkeypatch.setattr(capture, "require_expected_raw_probe_authority", validating_authority)
    with pytest.raises(capture.RawProbeCaptureError, match="invalid or inactive"):
        capture.capture_and_publish_raw_probe_store(lease, tables)
    assert calls == []
    assert published == []


def test_accounting_drift_blocks_publication(
    patched_capture: tuple[_Lease, _Tables, list[tuple[str, tuple[object, ...], dict[str, object]]], list[object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, tables, _calls, published = patched_capture

    monkeypatch.setattr(capture, "discover_affordances", lambda *a, **k: _Evidence(_ProbeAccounting(actions=63)))
    with pytest.raises(capture.RawProbeCaptureError, match="accounting"):
        capture.capture_and_publish_raw_probe_store(lease, tables)
    assert published == []


def test_canonical_table_failure_on_last_key_blocks_publication(
    patched_capture: tuple[_Lease, _Tables, list[tuple[str, tuple[object, ...], dict[str, object]]], list[object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, tables, calls, published = patched_capture
    original = tables.table_for

    def fail_last(key: _Key) -> object:
        if len(tables.requested) == 239:
            raise capture.CanonicalPooledTableError("fixture final canonical table failure")
        return original(key)

    monkeypatch.setattr(tables, "table_for", fail_last)
    with pytest.raises(capture.RawProbeCaptureError, match="capture failed"):
        capture.capture_and_publish_raw_probe_store(lease, tables)
    assert len([item for item in calls if item[0] == "probe"]) == 240
    assert published == []


def test_publication_is_called_once_after_complete_ordered_batch(
    patched_capture: tuple[_Lease, _Tables, list[tuple[str, tuple[object, ...], dict[str, object]]], list[object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, tables, calls, _published = patched_capture
    invocations: list[tuple[int, tuple[_Key, ...]]] = []

    def publish_once(_lease: object, *, artifacts: tuple[_Persisted, ...]) -> object:
        invocations.append(
            (
                len([item for item in calls if item[0] == "probe"]),
                tuple(artifact.key for artifact in artifacts),
            )
        )
        return _Snapshot()

    monkeypatch.setattr(capture, "publish_raw_probe_store_from_readiness", publish_once)
    capture.capture_and_publish_raw_probe_store(lease, tables)
    assert invocations == [(240, lease.authority.keys)]


def test_capture_does_not_publish_if_last_probe_fails(
    patched_capture: tuple[_Lease, _Tables, list[tuple[str, tuple[object, ...], dict[str, object]]], list[object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, tables, calls, published = patched_capture

    def fail_last(*args: object, **kwargs: object) -> _Evidence:
        if len([item for item in calls if item[0] == "probe"]) == 239:
            raise RuntimeError("fixture failure at ordinal 239")
        calls.append(("probe", args, kwargs))
        return _Evidence()

    monkeypatch.setattr(capture, "discover_affordances", fail_last)

    with pytest.raises(capture.RawProbeCaptureError, match="capture failed") as raised:
        capture.capture_and_publish_raw_probe_store(lease, tables)
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "ordinal 239" in str(raised.value.__cause__)
    assert published == []


@pytest.mark.parametrize("lease_kind,tables_kind", [(object, _Tables), (_Lease, object)])
def test_forged_capabilities_are_rejected_before_environment_creation(
    patched_capture: tuple[_Lease, _Tables, list[tuple[str, tuple[object, ...], dict[str, object]]], list[object]],
    lease_kind: type[object],
    tables_kind: type[object],
) -> None:
    lease, tables, calls, published = patched_capture
    maybe_lease = object() if lease_kind is object else lease
    maybe_tables = object() if tables_kind is object else tables

    with pytest.raises(capture.RawProbeCaptureError):
        capture.capture_and_publish_raw_probe_store(maybe_lease, maybe_tables)
    assert calls == []
    assert published == []


def test_capture_module_has_no_forbidden_execution_imports() -> None:
    tree = ast.parse(inspect.getsource(capture))
    forbidden = (
        "optimum",
        "classifier",
        "evaluator",
        "candidate",
        "model",
        "runstore",
        "runtime",
        "result",
        "final",
    )
    modules = [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not [name for name in modules if any(word in name.lower() for word in forbidden)]
