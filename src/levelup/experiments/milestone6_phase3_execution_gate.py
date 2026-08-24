"""All-or-none activation for the frozen Milestone 6 Phase 3 result stores.

Preparation of a Phase 3 result tree is intentionally inert.  This module is
the only boundary which turns the six prepared stores into a writable batch.
The boundary consumes a *live* :class:`Phase3ActivationReadinessLease`, checks
the complete six-store matrix while all descriptors are held, and publishes a
single immutable marker in the result root as its durable commit.  A missing
marker therefore always means that the tree is not executable.

The implementation does not execute a unit, inspect an outcome, or consult an
oracle.  It only validates metadata and publishes/consumes typed result
records after activation.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from levelup.experiments.milestone6_phase3_model_artifacts import (
    KEYS_DIR,
    Phase3ModelArtifactIndex,
)
from levelup.experiments.milestone6_phase3_model_authority import (
    load_phase3_model_artifact_authority_bytes,
)
from levelup.experiments.milestone6_phase3_models import (
    HISTORY_PARAMETERS,
    S_CONDITION,
    S_PARAMETERS,
)
from levelup.experiments.milestone6_phase3_readiness import (
    PHASE3_MODEL_AUTHORITY_RELATIVE,
    PHASE3_TRAINING_SHUFFLE_REPORT_RELATIVE,
    Phase3ActivationReadinessLease,
    Phase3ReadinessError,
)
from levelup.experiments.milestone6_phase3_result_store import (
    FAMILIES,
    Phase3ExpectedPlan,
    Phase3ResultStore,
    Phase3ResultStoreError,
    _verify_record_identity,
)
from levelup.experiments.milestone6_phase3_result_store import (
    SCHEMA_VERSION as RESULT_STORE_SCHEMA_VERSION,
)
from levelup.experiments.runner import secure_fs
from levelup.experiments.runner.config import canonical_json_bytes
from levelup.experiments.runner.records import (
    AttemptRecord,
    ResourceAccounting,
    UnitRecord,
)

H4_SHUFFLED_CONDITION = "H4-shuffled-history-transition-listwise-optimum"

ACTIVATION_SCHEMA_VERSION = "milestone6.phase3.activation.v1"
ACTIVATION_MARKER_NAME = "phase3-activation.json"
_MARKER_TMP_PREFIX = ".phase3-activation."
_BATCH_TOKEN = object()


class Phase3ActivationError(RuntimeError):
    """Raised when the prepared Phase 3 namespace cannot be activated safely."""


def _identity(value: os.stat_result) -> tuple[int, int]:
    if not stat.S_ISDIR(value.st_mode):
        raise Phase3ActivationError("Phase 3 result entry is not a directory")
    return int(value.st_dev), int(value.st_ino)


def _record_identity(value: tuple[int, int]) -> list[int]:
    return [int(value[0]), int(value[1])]


RecordFingerprint = tuple[int, int, int, int, int, str]


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _canonical(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _sha(value: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read_file(fd: int, name: str) -> bytes:
    try:
        return secure_fs.read_bytes_at(fd, name)
    except secure_fs.SecureFilesystemError as exc:
        raise Phase3ActivationError(f"cannot read activation marker: {name}") from exc


def _open_marker(root_fd: int, stack: ExitStack) -> tuple[int, tuple[int, int]]:
    try:
        marker_fd = os.open(
            ACTIVATION_MARKER_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd
        )
        stack.callback(os.close, marker_fd)
        observed = os.fstat(marker_fd)
        if not stat.S_ISREG(observed.st_mode):
            raise Phase3ActivationError("activation marker is not a regular file")
        return marker_fd, (int(observed.st_dev), int(observed.st_ino))
    except Phase3ActivationError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise Phase3ActivationError("cannot pin activation marker") from exc


def _marker_bytes(marker_fd: int) -> bytes:
    try:
        os.lseek(marker_fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(marker_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise Phase3ActivationError("cannot read pinned activation marker") from exc


def _parse_canonical_record(
    rendered: bytes,
    name: str,
    model: type[UnitRecord] | type[AttemptRecord],
):
    try:
        value = json.loads(rendered)
        record = model.model_validate(value)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise Phase3ActivationError(f"invalid stored result record: {name}") from exc
    if rendered != _canonical(record.model_dump(mode="json")):
        raise Phase3ActivationError(f"stored result record is not canonical: {name}")
    return record


def _record_snapshot(fd: int, name: str) -> tuple[bytes, RecordFingerprint]:
    """Read one record through a pinned descriptor and return stable bytes/fingerprint."""

    try:
        with secure_fs.open_regular_file_at(fd, name) as record_fd:
            before = os.fstat(record_fd)
            if not stat.S_ISREG(before.st_mode):
                raise Phase3ActivationError("stored result is not a regular file")
            path_before = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if _stat_fingerprint(path_before) != _stat_fingerprint(before):
                raise Phase3ActivationError("stored result identity changed while opening")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(record_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            rendered = b"".join(chunks)
            after = os.fstat(record_fd)
            path_after = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if (
                _stat_fingerprint(before) != _stat_fingerprint(after)
                or len(rendered) != int(after.st_size)
                or _stat_fingerprint(path_after) != _stat_fingerprint(after)
            ):
                raise Phase3ActivationError("stored result changed while being read")
    except Phase3ActivationError:
        raise
    except (OSError, secure_fs.SecureFilesystemError, TypeError, ValueError) as exc:
        raise Phase3ActivationError("cannot fingerprint stored result") from exc
    return rendered, (*_stat_fingerprint(after), hashlib.sha256(rendered).hexdigest())


def _record_fingerprint(fd: int, name: str) -> RecordFingerprint:
    return _record_snapshot(fd, name)[1]


def _load_canonical_record(fd: int, name: str, model: type[UnitRecord] | type[AttemptRecord]):
    rendered, _fingerprint = _record_snapshot(fd, name)
    return _parse_canonical_record(rendered, name, model)


@dataclass(slots=True)
class _ScientificAuthorityCache:
    """Activation-scoped, descriptor-pinned scientific authority.

    The authority and shuffle report are immutable readiness bytes, so parse
    them once.  Model-key indices are loaded lazily through the held keys
    directory and cached by owner.  We retain the entry identity and stat
    fingerprint so a replacement or in-place mutation is detected on every
    subsequent validation without repeatedly reparsing the 480-owner store.
    """

    lease: Phase3ActivationReadinessLease
    authority: Any
    rows: dict[str, Any]
    shuffle_views: dict[str, Mapping[str, Any]]
    keys_fd: int
    _indices: dict[str, tuple[Any, tuple[int, int], tuple[int, int, int, int, int]]] = field(
        default_factory=dict
    )

    @classmethod
    def from_lease(cls, lease: Phase3ActivationReadinessLease) -> "_ScientificAuthorityCache":
        files = lease.snapshot.files_by_path
        try:
            authority = load_phase3_model_artifact_authority_bytes(
                files[PHASE3_MODEL_AUTHORITY_RELATIVE].content
            )
            report_content = files[PHASE3_TRAINING_SHUFFLE_REPORT_RELATIVE].content
            report = json.loads(report_content)
            if not isinstance(report, dict) or canonical_json_bytes(report) != report_content:
                raise ValueError("shuffle report is not canonical")
            views = report["views"]
            if not isinstance(views, list):
                raise ValueError("shuffle report views are not a list")
            shuffle_views = {str(view["view_id"]): view for view in views}
            if len(shuffle_views) != len(views):
                raise ValueError("shuffle report view IDs are duplicated")
            rows = {row.owner_id: row for row in authority.models}
            if len(rows) != len(authority.models):
                raise ValueError("model authority owner IDs are duplicated")
            keys_relative = f"runs/milestone6/{authority.artifact_store_id}/{KEYS_DIR}"
            keys_fd = lease.directory_descriptors.get(keys_relative)
            if type(keys_fd) is not int:
                raise ValueError("held model-key authority is unavailable")
            return cls(lease, authority, rows, shuffle_views, keys_fd)
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise Phase3ActivationError("activation scientific authority is invalid") from exc

    @staticmethod
    def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )

    def row_for(self, planned: Any) -> Any:
        try:
            return self.rows[planned.model_owner_id]
        except (AttributeError, KeyError) as exc:
            raise Phase3ActivationError(
                "completed record has no published model-owner authority row"
            ) from exc

    def _assert_index_stable(
        self,
        key_id: str,
        identity: tuple[int, int],
        fingerprint: tuple[int, int, int, int, int],
    ) -> None:
        try:
            observed = os.stat(key_id + ".json", dir_fd=self.keys_fd, follow_symlinks=False)
            if not stat.S_ISREG(observed.st_mode):
                raise Phase3ActivationError("held model-key authority is not a regular file")
            if (int(observed.st_dev), int(observed.st_ino)) != identity:
                raise Phase3ActivationError("held model-key authority identity changed")
            if self._fingerprint(observed) != fingerprint:
                raise Phase3ActivationError("held model-key authority changed")
        except Phase3ActivationError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise Phase3ActivationError("held model-key authority cannot be revalidated") from exc

    def index_for(self, planned: Any, row: Any) -> Any:
        cached = self._indices.get(row.key_id)
        if cached is not None:
            index, identity, fingerprint = cached
            self._assert_index_stable(row.key_id, identity, fingerprint)
            return index
        try:
            with secure_fs.open_regular_file_at(self.keys_fd, f"{row.key_id}.json") as index_fd:
                observed_before = os.fstat(index_fd)
                identity = (int(observed_before.st_dev), int(observed_before.st_ino))
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(index_fd, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                rendered_index = b"".join(chunks)
                observed_after = os.fstat(index_fd)
                if (
                    self._fingerprint(observed_before)
                    != self._fingerprint(observed_after)
                    or len(rendered_index) != int(observed_after.st_size)
                ):
                    raise Phase3ActivationError(
                        "held model-key authority changed while being read"
                    )
                index = Phase3ModelArtifactIndex.model_validate(json.loads(rendered_index))
                fingerprint = self._fingerprint(observed_after)
        except (
            OSError,
            secure_fs.SecureFilesystemError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            raise Phase3ActivationError("held model-key authority is invalid") from exc
        if rendered_index != _canonical(index.model_dump(mode="json")):
            raise Phase3ActivationError("held model-key authority is not canonical")
        self._indices[row.key_id] = (
            index,
            identity,
            fingerprint,
        )
        self._assert_index_stable(row.key_id, identity, fingerprint)
        return index

    def shuffle_view(self, view_id: str) -> Mapping[str, Any]:
        try:
            return self.shuffle_views[view_id]
        except KeyError as exc:
            raise Phase3ActivationError("H4-shuffled report view lineage is unavailable") from exc

    def revalidate(self) -> None:
        for key_id, (_index, identity, fingerprint) in self._indices.items():
            self._assert_index_stable(key_id, identity, fingerprint)

    def discard(self) -> None:
        """Release parsed authority and lazy key state after the capability closes."""

        self._indices.clear()
        self.rows.clear()
        self.shuffle_views.clear()
        self.authority = None
        self.keys_fd = -1


def _scientific_record_check(
    record: UnitRecord,
    planned: Any,
    scientific: _ScientificAuthorityCache,
) -> None:
    """Validate the typed Phase 3 contract before a record is published."""

    outcome = record.outcome
    accounting = record.accounting
    if (
        outcome.performance_metric_id != "performance_value"
        or outcome.performance_direction != "minimize"
        or outcome.evaluator_ran is not True
        or accounting.training != ResourceAccounting().training
        or accounting.probes.actions != 64
        or accounting.probes.environment_steps != accounting.probes.actions
        or accounting.search.actions < 1
        or accounting.search.actions > 1984
        or accounting.search.environment_steps != accounting.search.actions
        or accounting.probes.actions + accounting.search.actions > 2048
        or accounting.replay.environment_steps != accounting.replay.actions
        or accounting.search.episodes < 1
        or accounting.search.episodes > 150
        or record.candidate_generation_sha256 is None
    ):
        raise Phase3ActivationError("completed record violates the Phase 3 accounting contract")
    if outcome.success:
        if (
            outcome.censored
            or outcome.censoring_budget is not None
            or outcome.censoring_reason is not None
            or outcome.first_optimum_episode is None
            or outcome.first_optimum_adaptation_actions is None
            or outcome.first_optimum_episode < 1
            or outcome.first_optimum_episode > accounting.search.episodes
            or outcome.first_optimum_adaptation_actions < accounting.probes.actions
            or outcome.first_optimum_adaptation_actions
            > accounting.probes.actions + accounting.search.actions
            or outcome.first_optimum_adaptation_actions > 2048
        ):
            raise Phase3ActivationError("successful record lacks typed first-hit semantics")
    elif (
        not outcome.censored
        or outcome.censoring_budget != 2048
        or outcome.censoring_reason != "fixed_endpoint"
        or outcome.first_optimum_episode is not None
        or outcome.first_optimum_adaptation_actions is not None
    ):
        raise Phase3ActivationError("failed record violates fixed-endpoint censoring")

    authority = scientific.authority
    row = scientific.row_for(planned)
    reference = record.shared_artifact
    if (
        reference is None
        or record.shared_artifacts
        or reference.key_id != row.key_id
        or reference.artifact_id != row.artifact_id
        or reference.cost_id != row.cost_id
    ):
        raise Phase3ActivationError("completed record model reference differs from authority")
    index = scientific.index_for(planned, row)
    key = index.key
    if (
        index.key_id != row.key_id
        or index.artifact_id != row.artifact_id
        or index.manifest_sha256 != row.manifest_sha256
        or key.owner_id != planned.model_owner_id
        or key.plan_id != authority.plan_id
        or key.protocol_sha256 != authority.protocol_sha256
        or key.view_id != planned.view_id
        or key.condition_id != planned.base_condition_id
        or key.fold_id != planned.fold_id
        or key.heldout_family != planned.heldout_family
        or key.training_tuple_id != planned.training_tuple_id
    ):
        raise Phase3ActivationError("held model-key lineage differs from the frozen unit")
    required_names = (
        "model_trainable_parameters",
        "model_optimizer_steps",
        "model_forward_passes",
        "model_recurrent_steps",
        "model_training_examples",
    )
    diagnostics = record.diagnostics
    if any(
        type(diagnostics.get(name)) is not int or diagnostics[name] < 0
        for name in required_names
    ):
        raise Phase3ActivationError("completed record lacks required model diagnostics")
    if any(
        diagnostics[name] < 1
        for name in (
            "model_trainable_parameters",
            "model_optimizer_steps",
            "model_forward_passes",
            "model_training_examples",
        )
    ):
        raise Phase3ActivationError("completed record has non-positive model diagnostics")
    if diagnostics["model_forward_passes"] != (
        diagnostics["model_optimizer_steps"] * diagnostics["model_training_examples"]
    ):
        raise Phase3ActivationError("completed record model forward-pass diagnostic is inconsistent")
    expected_parameters = S_PARAMETERS if planned.base_condition_id == S_CONDITION else HISTORY_PARAMETERS
    try:
        expected_epochs = int(planned.training_tuple_id.rsplit("-e", 1)[1])
    except (IndexError, ValueError) as exc:
        raise Phase3ActivationError("planned training tuple has no frozen epoch count") from exc
    if (
        diagnostics["model_trainable_parameters"] != expected_parameters
        or diagnostics["model_optimizer_steps"] != expected_epochs
        or diagnostics["model_trainable_parameters"]
        != key.report.trainable_parameters
        or diagnostics["model_optimizer_steps"] != key.report.optimizer_steps
        or diagnostics["model_forward_passes"] != key.report.forward_passes
        or diagnostics["model_recurrent_steps"] != key.report.recurrent_steps
        or diagnostics["model_training_examples"] != key.report.training_examples
    ):
        raise Phase3ActivationError(
            "completed record model diagnostics differ from held model-key authority"
        )

    if planned.base_condition_id != H4_SHUFFLED_CONDITION:
        if record.history_shuffle_permutation_map_sha256 is not None:
            raise Phase3ActivationError("non-shuffled record carries a history permutation digest")
        shuffle_names = (
            "history_shuffle_eligible_windows",
            "history_shuffle_map_nonidentity_windows",
            "history_shuffle_effective_tensor_changed_windows",
            "history_shuffle_duplicate_vector_no_effect_windows",
            "history_shuffle_unchanged_short_windows",
        )
        if diagnostics.get("history_shuffle_claim_eligible") is not None or any(
            diagnostics.get(name) != 0 for name in shuffle_names
        ):
            raise Phase3ActivationError("non-shuffled record carries shuffle coverage")
        return
    if record.history_shuffle_permutation_map_sha256 is None:
        raise Phase3ActivationError("H4-shuffled record is missing its history permutation digest")
    try:
        view = scientific.shuffle_view(planned.view_id)
        eligible = diagnostics.get("history_shuffle_eligible_windows")
        map_nonidentity = diagnostics.get("history_shuffle_map_nonidentity_windows")
        effective = diagnostics.get("history_shuffle_effective_tensor_changed_windows")
        duplicate = diagnostics.get("history_shuffle_duplicate_vector_no_effect_windows")
        short = diagnostics.get("history_shuffle_unchanged_short_windows")
        claim = diagnostics.get("history_shuffle_claim_eligible")
        if (
            planned.model_owner_id not in view["model_owner_ids"]
            or any(
                type(value) is not int or value < 0
                for value in (eligible, map_nonidentity, effective, duplicate, short)
            )
            or map_nonidentity != eligible
            or effective + duplicate != eligible
            or effective > map_nonidentity
            or claim is not (eligible > 0 and effective / eligible >= 0.80)
        ):
            raise Phase3ActivationError("H4-shuffled record history digest differs from report lineage")
    except Phase3ActivationError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Phase3ActivationError("H4-shuffled report view lineage is unavailable") from exc


def _exclusive_publish(fd: int, name: str, rendered: bytes) -> None:
    """Publish one file without replacing an existing directory entry."""

    temporary = f"{_MARKER_TMP_PREFIX}{uuid.uuid4().hex}.tmp"
    temp_fd: int | None = None
    try:
        temp_fd = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
            dir_fd=fd,
        )
        with os.fdopen(temp_fd, "wb") as handle:
            temp_fd = None
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
        os.fsync(fd)
    except FileExistsError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise Phase3ActivationError("cannot publish activation marker") from exc
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        try:
            os.unlink(temporary, dir_fd=fd)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise Phase3ActivationError("cannot remove activation temporary") from exc


def _entries(fd: int) -> set[str]:
    try:
        with os.scandir(fd) as iterator:
            result: set[str] = set()
            for entry in iterator:
                if entry.is_symlink():
                    raise Phase3ActivationError("activation namespace contains a symlink")
                result.add(entry.name)
            return result
    except Phase3ActivationError:
        raise
    except OSError as exc:
        raise Phase3ActivationError("cannot enumerate activation namespace") from exc


def _validate_outer_namespaces(
    root_fd: int,
    stores: tuple[Phase3ResultStore, ...],
    descriptors: tuple[dict[str, int], ...],
    *,
    marker_exists: bool,
) -> None:
    expected_root = set(FAMILIES)
    if marker_exists:
        expected_root.add(ACTIVATION_MARKER_NAME)
    if _entries(root_fd) != expected_root:
        raise Phase3ActivationError(
            "result output root contains foreign, missing, or temporary entries"
        )
    for store, descriptor in zip(stores, descriptors, strict=True):
        if _entries(descriptor["family"]) != {store.run_id}:
            raise Phase3ActivationError(
                f"family namespace contains a foreign or missing run: {store.family_id}"
            )


def _marker_body(
    expected: Phase3ExpectedPlan,
    stores: tuple[Phase3ResultStore, ...],
    lease: Phase3ActivationReadinessLease,
    root_identity: tuple[int, int],
    identities: tuple[dict[str, tuple[int, int]], ...],
) -> dict[str, Any]:
    snapshot = lease.snapshot
    return {
        "schema_version": ACTIVATION_SCHEMA_VERSION,
        "result_store_schema_version": RESULT_STORE_SCHEMA_VERSION,
        "phase": "validation",
        "plan_id": expected.plan_id,
        "protocol_sha256": expected.protocol_sha256,
        "model_authority_sha256": expected.model_authority_sha256,
        "readiness": {
            "git_commit_sha": snapshot.git_commit_sha,
            "training_shuffle_report_sha256": snapshot.training_shuffle_report_sha256,
            "training_shuffle_report_file_sha256": snapshot.training_shuffle_report_file_sha256,
        },
        "root_identity": _record_identity(root_identity),
        "stores": [
            {
                "family_id": store.family_id,
                "run_id": store.run_id,
                "config_sha256": store.config_sha256,
                "identities": {
                    key: _record_identity(value) for key, value in identity.items()
                },
            }
            for store, identity in zip(stores, identities, strict=True)
        ],
    }


def _marker_with_self_hash(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "marker_sha256": _sha(body)}


def _validate_marker(value: object, expected: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise Phase3ActivationError("activation marker is not an object")
    supplied = value.get("marker_sha256")
    unsigned = dict(value)
    unsigned.pop("marker_sha256", None)
    if supplied != _sha(unsigned) or value != expected:
        raise Phase3ActivationError("activation marker differs from the canonical activation")


def _validate_lease(lease: object) -> Phase3ActivationReadinessLease:
    if type(lease) is not Phase3ActivationReadinessLease:
        raise Phase3ActivationError("activation requires the canonical readiness lease")
    try:
        return lease.require_active()
    except Phase3ReadinessError as exc:
        raise Phase3ActivationError("activation readiness lease is not active") from exc


def _validate_store_arguments(
    stores: tuple[Phase3ResultStore, ...], expected: Phase3ExpectedPlan
) -> Path:
    if type(expected) is not Phase3ExpectedPlan:
        raise Phase3ActivationError("activation requires the canonical expected plan")
    if expected.family_order != FAMILIES or expected.final_family_access:
        raise Phase3ActivationError("activation expected plan is not development-only")
    if len(stores) != len(FAMILIES):
        raise Phase3ActivationError("activation requires exactly six family stores")
    if tuple(store.family_id for store in stores) != FAMILIES:
        raise Phase3ActivationError("activation stores must use frozen family order")
    if any(type(store) is not Phase3ResultStore for store in stores):
        raise Phase3ActivationError("activation stores are not canonical prepared stores")
    if any(store.execution_ready for store in stores):
        raise Phase3ActivationError("prepared stores cannot already be execution-ready")
    roots = tuple(Path(os.path.abspath(store.root)) for store in stores)
    if len(set(roots)) != 1:
        raise Phase3ActivationError("activation stores do not share one output root")
    for store, spec in zip(stores, expected.stores, strict=True):
        if store.spec != spec:
            raise Phase3ActivationError(f"store differs from frozen family plan: {store.family_id}")
    return roots[0]


def _open_store_descriptors(
    root: Path,
    stores: tuple[Phase3ResultStore, ...],
    stack: ExitStack,
) -> tuple[int, tuple[dict[str, int], ...], tuple[dict[str, tuple[int, int]], ...]]:
    try:
        root_fd = secure_fs.open_directory_chain(root)
        stack.callback(os.close, root_fd)
        descriptors: list[dict[str, int]] = []
        identities: list[dict[str, tuple[int, int]]] = []
        for store in stores:
            family_fd = secure_fs.open_child_directory(root_fd, store.family_id)
            stack.callback(os.close, family_fd)
            run_fd = secure_fs.open_child_directory(family_fd, store.run_id)
            stack.callback(os.close, run_fd)
            units_fd = secure_fs.open_child_directory(run_fd, "units")
            stack.callback(os.close, units_fd)
            attempts_fd = secure_fs.open_child_directory(run_fd, "attempts")
            stack.callback(os.close, attempts_fd)
            value = {
                "root": root_fd,
                "family": family_fd,
                "run": run_fd,
                "units": units_fd,
                "attempts": attempts_fd,
            }
            descriptors.append(value)
            identities.append({key: secure_fs.directory_identity(fd) for key, fd in value.items()})
        return root_fd, tuple(descriptors), tuple(identities)
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise Phase3ActivationError("cannot pin all six prepared result stores") from exc


def _validate_store_descriptors(
    stores: tuple[Phase3ResultStore, ...],
    descriptors: tuple[dict[str, int], ...],
    scientific: _ScientificAuthorityCache,
    record_fingerprints: dict[tuple[str, str, str], RecordFingerprint],
    *,
    marker_exists: bool,
    initial_scan: bool,
) -> None:
    for store, descriptor in zip(stores, descriptors, strict=True):
        observed = {key: secure_fs.directory_identity(fd) for key, fd in descriptor.items()}
        expected = {
            "root": store.root_identity,
            "family": store.family_identity,
            "run": store.run_identity,
            "units": store.units_identity,
            "attempts": store.attempts_identity,
        }
        if observed != expected:
            raise Phase3ActivationError(
                f"prepared family store identity differs from its pinned identity: {store.family_id}"
            )
        try:
            units = secure_fs.strict_regular_entries(descriptor["units"])
            attempts = secure_fs.strict_regular_entries(descriptor["attempts"])
            if not marker_exists and (units or attempts):
                raise Phase3ActivationError("orphan result records require an activation marker")
            store._validate_pinned(descriptor)
            expected_by_id = {item.unit.unit_id: item for item in store.spec.units}
            for name in units:
                identity_key = (store.family_id, "units", name)
                rendered, fingerprint = _record_snapshot(descriptor["units"], name)
                prior_fingerprint = record_fingerprints.get(identity_key)
                if prior_fingerprint is None:
                    if not initial_scan:
                        raise Phase3ActivationError("untracked result appeared during activation")
                    record_fingerprints[identity_key] = fingerprint
                elif prior_fingerprint != fingerprint:
                    raise Phase3ActivationError(
                        "completed result identity or fingerprint changed"
                    )
                record = _parse_canonical_record(rendered, name, UnitRecord)
                _verify_record_identity(record, store.spec, expected_by_id, filename=name)
                _scientific_record_check(record, expected_by_id[record.unit_id], scientific)
            for name in attempts:
                identity_key = (store.family_id, "attempts", name)
                rendered, fingerprint = _record_snapshot(descriptor["attempts"], name)
                prior_fingerprint = record_fingerprints.get(identity_key)
                if prior_fingerprint is None:
                    if not initial_scan:
                        raise Phase3ActivationError("untracked result appeared during activation")
                    record_fingerprints[identity_key] = fingerprint
                elif prior_fingerprint != fingerprint:
                    raise Phase3ActivationError(
                        "attempt result identity or fingerprint changed"
                    )
                record = _parse_canonical_record(rendered, name, AttemptRecord)
                _verify_record_identity(record, store.spec, expected_by_id, filename=name)
            current_keys = {
                (store.family_id, "units", name) for name in units
            }
            current_keys.update(
                (store.family_id, "attempts", name) for name in attempts
            )
            for key in record_fingerprints:
                if key[0] == store.family_id and key not in current_keys:
                    raise Phase3ActivationError("stored result was removed during activation")
        except Phase3ActivationError:
            raise
        except (OSError, Phase3ResultStoreError, secure_fs.SecureFilesystemError) as exc:
            raise Phase3ActivationError(f"prepared family store is invalid: {store.family_id}") from exc
    scientific.revalidate()


def _reopen_identities(root: Path, stores: tuple[Phase3ResultStore, ...]) -> tuple[tuple[int, int], tuple[dict[str, tuple[int, int]], ...]]:
    try:
        with ExitStack() as stack:
            root_fd = secure_fs.open_directory_chain(root)
            stack.callback(os.close, root_fd)
            root_identity = secure_fs.directory_identity(root_fd)
            values: list[dict[str, tuple[int, int]]] = []
            for store in stores:
                family_fd = secure_fs.open_child_directory(root_fd, store.family_id)
                stack.callback(os.close, family_fd)
                run_fd = secure_fs.open_child_directory(family_fd, store.run_id)
                stack.callback(os.close, run_fd)
                units_fd = secure_fs.open_child_directory(run_fd, "units")
                stack.callback(os.close, units_fd)
                attempts_fd = secure_fs.open_child_directory(run_fd, "attempts")
                stack.callback(os.close, attempts_fd)
                values.append({
                    "root": root_identity,
                    "family": secure_fs.directory_identity(family_fd),
                    "run": secure_fs.directory_identity(run_fd),
                    "units": secure_fs.directory_identity(units_fd),
                    "attempts": secure_fs.directory_identity(attempts_fd),
                })
            return root_identity, tuple(values)
    except (OSError, secure_fs.SecureFilesystemError) as exc:
        raise Phase3ActivationError("result output root or store was substituted") from exc


def _write_record(fd: int, name: str, value: object) -> bool:
    rendered = _canonical(value)
    try:
        existing = secure_fs.read_bytes_at(fd, name)
    except secure_fs.SecureFilesystemError as exc:
        cause = exc.__cause__
        if not isinstance(cause, FileNotFoundError):
            raise Phase3ActivationError(f"cannot read existing result record: {name}") from exc
        temporary = f".{name}.{uuid.uuid4().hex}.tmp"
        temp_fd: int | None = None
        try:
            temp_fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600, dir_fd=fd)
            with os.fdopen(temp_fd, "wb") as handle:
                temp_fd = None
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
            os.fsync(fd)
            return True
        except FileExistsError:
            existing = secure_fs.read_bytes_at(fd, name)
        except (OSError, TypeError, ValueError) as write_exc:
            raise Phase3ActivationError(f"cannot publish result record: {name}") from write_exc
        finally:
            if temp_fd is not None:
                os.close(temp_fd)
            try:
                os.unlink(temporary, dir_fd=fd)
            except FileNotFoundError:
                pass
            except OSError as unlink_exc:
                raise Phase3ActivationError("cannot remove result temporary") from unlink_exc
    if existing != rendered:
        raise Phase3ActivationError(f"conflicting result record: {name}")
    return False


@dataclass(slots=True)
class _FamilyWritableStore:
    _batch: "Phase3ActivatedBatch"
    _index: int

    @property
    def family_id(self) -> str:
        return self._batch._stores[self._index].family_id

    @property
    def run_id(self) -> str:
        return self._batch._stores[self._index].run_id

    @property
    def config_sha256(self) -> str:
        return self._batch._stores[self._index].config_sha256

    def write_completed(self, record: UnitRecord) -> bool:
        return self._batch._write_completed(self._index, record)

    def write_attempt(self, record: AttemptRecord) -> bool:
        return self._batch._write_attempt(self._index, record)

    def load_completed(self, unit_id: str) -> UnitRecord | None:
        return self._batch._load_completed(self._index, unit_id)

    def completed_unit_ids(self) -> tuple[str, ...]:
        return self._batch.completed_unit_ids(self.family_id)

    def attempt_records(self) -> tuple[AttemptRecord, ...]:
        return self._batch.attempt_records(self.family_id)

    def next_attempt_number(self, unit_id: str) -> int:
        return self._batch.next_attempt_number(unit_id, self.family_id)


@dataclass(slots=True)
class Phase3ActivatedBatch:
    """Capability-bearing activated batch; usable only inside its context."""

    _root: Path
    _stores: tuple[Phase3ResultStore, ...]
    _expected: Phase3ExpectedPlan
    _lease: Phase3ActivationReadinessLease
    _root_fd: int
    _descriptors: tuple[dict[str, int], ...]
    _identities: tuple[dict[str, tuple[int, int]], ...]
    _marker: bytes
    _marker_fd: int
    _marker_identity: tuple[int, int]
    _scientific: _ScientificAuthorityCache
    _record_fingerprints: dict[tuple[str, str, str], RecordFingerprint]
    _unit_maps: tuple[dict[str, Any], ...]
    _token: object = field(repr=False, compare=False)
    _active: bool = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def stores(self) -> tuple[_FamilyWritableStore, ...]:
        self._require_live()
        return tuple(_FamilyWritableStore(self, index) for index in range(len(FAMILIES)))

    def store_for_family(self, family_id: str) -> _FamilyWritableStore:
        self._require_live()
        try:
            index = FAMILIES.index(family_id)
        except ValueError as exc:
            raise Phase3ActivationError(f"unknown activated family: {family_id}") from exc
        return _FamilyWritableStore(self, index)

    def _require_live(self) -> None:
        if self._token is not _BATCH_TOKEN or not self._active:
            raise Phase3ActivationError("activated Phase 3 batch capability has expired")
        _validate_lease(self._lease)
        try:
            marker_stat = os.fstat(self._marker_fd)
            if (int(marker_stat.st_dev), int(marker_stat.st_ino)) != self._marker_identity:
                raise Phase3ActivationError("pinned activation marker identity changed")
            if _marker_bytes(self._marker_fd) != self._marker:
                raise Phase3ActivationError("activation marker changed")
            try:
                with ExitStack() as marker_stack:
                    current_root_fd = secure_fs.open_directory_chain(self._root)
                    marker_stack.callback(os.close, current_root_fd)
                    with secure_fs.open_regular_file_at(current_root_fd, ACTIVATION_MARKER_NAME) as current_marker_fd:
                        current_marker = os.fstat(current_marker_fd)
                        current_identity = (int(current_marker.st_dev), int(current_marker.st_ino))
                        if current_identity != self._marker_identity:
                            raise Phase3ActivationError("activation marker identity changed")
            except Phase3ActivationError:
                raise
            except (OSError, secure_fs.SecureFilesystemError) as exc:
                raise Phase3ActivationError("cannot recheck activation marker") from exc
            root_identity, identities = _reopen_identities(self._root, self._stores)
            if root_identity != self._identities[0]["root"] or identities != self._identities:
                raise Phase3ActivationError("activated result directory identity changed")
            _validate_outer_namespaces(
                self._root_fd,
                self._stores,
                self._descriptors,
                marker_exists=True,
            )
        except Phase3ActivationError:
            raise

    def _assert_record_identity(self, index: int, kind: str, name: str) -> None:
        family_id = self._stores[index].family_id
        expected = self._record_fingerprints.get((family_id, kind, name))
        if expected is None:
            raise Phase3ActivationError("untracked result appeared during activation")
        observed = _record_fingerprint(self._descriptors[index][kind], name)
        if observed != expected:
            raise Phase3ActivationError("stored result identity or fingerprint changed")

    def _unit(self, index: int, unit_id: str):
        try:
            return self._unit_maps[index][unit_id]
        except KeyError as exc:
            raise Phase3ActivationError(f"foreign Phase 3 unit: {unit_id}") from exc

    def _write_completed(self, index: int, record: UnitRecord) -> bool:
        self._require_live()
        if type(record) is not UnitRecord:
            raise Phase3ActivationError("completed result must be a typed UnitRecord")
        store = self._stores[index]
        planned = self._unit(index, record.unit_id)
        try:
            _verify_record_identity(record, store.spec, {planned.unit.unit_id: planned}, filename=f"{record.unit_id}.json")
        except Phase3ResultStoreError as exc:
            raise Phase3ActivationError("completed record lineage differs from frozen unit") from exc
        _scientific_record_check(record, planned, self._scientific)
        name = f"{record.unit_id}.json"
        identity_key = (store.family_id, "units", name)
        previously_tracked = identity_key in self._record_fingerprints
        if previously_tracked:
            self._assert_record_identity(index, "units", name)
        published = _write_record(self._descriptors[index]["units"], name, record.model_dump(mode="json"))
        if not published and not previously_tracked:
            raise Phase3ActivationError("completed result raced with external publication")
        rendered, observed = _record_snapshot(self._descriptors[index]["units"], name)
        if rendered != _canonical(record.model_dump(mode="json")):
            raise Phase3ActivationError("completed result changed during publication")
        if previously_tracked and observed != self._record_fingerprints[identity_key]:
            raise Phase3ActivationError("completed result identity or fingerprint changed")
        self._record_fingerprints[identity_key] = observed
        self._require_live()
        return published

    def _write_attempt(self, index: int, record: AttemptRecord) -> bool:
        self._require_live()
        if type(record) is not AttemptRecord:
            raise Phase3ActivationError("attempt result must be a typed AttemptRecord")
        store = self._stores[index]
        planned = self._unit(index, record.unit_id)
        name = f"{record.unit_id}.attempt-{record.attempt:04d}.json"
        try:
            _verify_record_identity(record, store.spec, {planned.unit.unit_id: planned}, filename=name)
        except Phase3ResultStoreError as exc:
            raise Phase3ActivationError("attempt record lineage differs from frozen unit") from exc
        if not 1 <= record.attempt <= 9999:
            raise Phase3ActivationError("attempt number must be between 1 and 9999")
        identity_key = (store.family_id, "attempts", name)
        previously_tracked = identity_key in self._record_fingerprints
        if previously_tracked:
            self._assert_record_identity(index, "attempts", name)
        published = _write_record(self._descriptors[index]["attempts"], name, record.model_dump(mode="json"))
        if not published and not previously_tracked:
            raise Phase3ActivationError("attempt result raced with external publication")
        rendered, observed = _record_snapshot(self._descriptors[index]["attempts"], name)
        if rendered != _canonical(record.model_dump(mode="json")):
            raise Phase3ActivationError("attempt result changed during publication")
        if previously_tracked and observed != self._record_fingerprints[identity_key]:
            raise Phase3ActivationError("attempt result identity or fingerprint changed")
        self._record_fingerprints[identity_key] = observed
        self._require_live()
        return published

    def _load_completed(self, index: int, unit_id: str) -> UnitRecord | None:
        self._require_live()
        self._unit(index, unit_id)
        name = f"{unit_id}.json"
        try:
            entries = set(secure_fs.strict_regular_entries(self._descriptors[index]["units"]))
        except secure_fs.SecureFilesystemError as exc:
            raise Phase3ActivationError("cannot enumerate activated completed results") from exc
        if name not in entries:
            return None
        try:
            rendered, fingerprint = _record_snapshot(
                self._descriptors[index]["units"], name
            )
            expected_fingerprint = self._record_fingerprints.get(
                (self._stores[index].family_id, "units", name)
            )
            if expected_fingerprint is None or fingerprint != expected_fingerprint:
                raise Phase3ActivationError("stored result identity or fingerprint changed")
            record = _parse_canonical_record(rendered, name, UnitRecord)
            planned = self._unit(index, record.unit_id)
            _verify_record_identity(
                record,
                self._stores[index].spec,
                {planned.unit.unit_id: planned},
                filename=name,
            )
            self._scientific_check_for_unit(index, record)
        except Exception as exc:
            if isinstance(exc, Phase3ActivationError):
                raise
            raise Phase3ActivationError("stored completed record is invalid") from exc
        self._require_live()
        return record

    def _scientific_check_for_unit(self, index: int, record: UnitRecord) -> None:
        planned = self._unit(index, record.unit_id)
        _scientific_record_check(record, planned, self._scientific)

    def completed_unit_ids(self, family_id: str | None = None) -> tuple[str, ...]:
        """Return validated record presence without exposing outcome fields."""

        self._require_live()
        if family_id is None:
            indices = range(len(self._stores))
        else:
            try:
                indices = (FAMILIES.index(family_id),)
            except ValueError as exc:
                raise Phase3ActivationError(
                    f"unknown activated family: {family_id}"
                ) from exc
        values: list[str] = []
        for index in indices:
            try:
                names = set(
                    secure_fs.strict_regular_entries(
                        self._descriptors[index]["units"]
                    )
                )
            except secure_fs.SecureFilesystemError as exc:
                raise Phase3ActivationError(
                    "cannot enumerate activated completed results"
                ) from exc
            expected_ids = tuple(
                item.unit.unit_id for item in self._stores[index].spec.units
            )
            expected_names = {f"{unit_id}.json" for unit_id in expected_ids}
            if not names <= expected_names:
                raise Phase3ActivationError(
                    "completed namespace contains a foreign record"
                )
            for name in names:
                self._assert_record_identity(index, "units", name)
            values.extend(
                unit_id for unit_id in expected_ids if f"{unit_id}.json" in names
            )
        self._require_live()
        return tuple(values)

    def attempt_records(self, family_id: str | None = None) -> tuple[AttemptRecord, ...]:
        self._require_live()
        if family_id is None:
            indices = range(len(self._stores))
        else:
            try:
                indices = (FAMILIES.index(family_id),)
            except ValueError as exc:
                raise Phase3ActivationError(f"unknown activated family: {family_id}") from exc
        values: list[AttemptRecord] = []
        for index in indices:
            try:
                names = secure_fs.strict_regular_entries(self._descriptors[index]["attempts"])
            except (ValueError, IndexError, secure_fs.SecureFilesystemError) as exc:
                raise Phase3ActivationError("cannot enumerate activated attempts") from exc
            for name in names:
                if ".attempt-" not in name:
                    raise Phase3ActivationError("attempt namespace contains a foreign record")
                rendered, fingerprint = _record_snapshot(
                    self._descriptors[index]["attempts"], name
                )
                expected_fingerprint = self._record_fingerprints.get(
                    (self._stores[index].family_id, "attempts", name)
                )
                if expected_fingerprint is None or fingerprint != expected_fingerprint:
                    raise Phase3ActivationError("stored result identity or fingerprint changed")
                record = _parse_canonical_record(rendered, name, AttemptRecord)
                planned = self._unit(index, record.unit_id)
                try:
                    _verify_record_identity(
                        record,
                        self._stores[index].spec,
                        {planned.unit.unit_id: planned},
                        filename=name,
                    )
                except Phase3ResultStoreError as exc:
                    raise Phase3ActivationError("attempt record lineage differs from frozen unit") from exc
                if not 1 <= record.attempt <= 9999:
                    raise Phase3ActivationError("attempt number must be between 1 and 9999")
                values.append(record)
        self._require_live()
        return tuple(values)

    def next_attempt_number(self, unit_id: str, family_id: str | None = None) -> int:
        self._require_live()
        if family_id is None:
            family_id = next(
                (
                    self._stores[index].family_id
                    for index, unit_map in enumerate(self._unit_maps)
                    if unit_id in unit_map
                ),
                None,
            )
        if family_id not in FAMILIES:
            raise Phase3ActivationError(f"foreign Phase 3 unit: {unit_id}")
        index = FAMILIES.index(family_id)
        self._unit(index, unit_id)
        prefix = f"{unit_id}.attempt-"
        numbers: list[int] = []
        for name in secure_fs.strict_regular_entries(self._descriptors[index]["attempts"]):
            self._assert_record_identity(index, "attempts", name)
            if name.startswith(prefix) and name.endswith(".json"):
                try:
                    number = int(name[len(prefix) : -5])
                except ValueError as exc:
                    raise Phase3ActivationError("attempt namespace contains a malformed number") from exc
                if not 1 <= number <= 9999:
                    raise Phase3ActivationError("attempt number must be between 1 and 9999")
                numbers.append(number)
        result = max(numbers, default=0) + 1
        if result > 9999:
            raise Phase3ActivationError("attempt number space is exhausted")
        return result

    def close(self) -> None:
        self._active = False
        self._scientific.discard()
        self._record_fingerprints.clear()
        for values in self._unit_maps:
            values.clear()


@contextmanager
def activate_phase3_result_stores(
    stores: tuple[Phase3ResultStore, ...] | list[Phase3ResultStore],
    expected: Phase3ExpectedPlan,
    readiness_lease: Phase3ActivationReadinessLease,
    *,
    expected_git_commit: str,
) -> Iterator[Phase3ActivatedBatch]:
    """Validate and atomically activate the complete six-store result tree."""

    typed_stores = tuple(stores)
    root = _validate_store_arguments(typed_stores, expected)
    lease = _validate_lease(readiness_lease)
    snapshot = lease.snapshot
    if (
        not isinstance(expected_git_commit, str)
        or len(expected_git_commit) < 40
        or len(expected_git_commit) > 64
        or any(character not in "0123456789abcdef" for character in expected_git_commit.lower())
        or snapshot.git_commit_sha != expected_git_commit
    ):
        raise Phase3ActivationError("activation requires the exact authorized readiness commit")
    if snapshot.git_dirty:
        raise Phase3ActivationError("activation requires a clean readiness snapshot")
    if (
        snapshot.plan_id != expected.plan_id
        or snapshot.model_authority_sha256 != expected.model_authority_sha256
        or snapshot.files_by_path.get(
            "configs/milestone6/phase3_representation_ladder.json"
        ) is None
        or snapshot.files_by_path[
            "configs/milestone6/phase3_representation_ladder.json"
        ].sha256
        != expected.protocol_sha256
    ):
        raise Phase3ActivationError("readiness authority lineage differs from result plan")
    stack = ExitStack()
    stack.__enter__()
    marker_bytes: bytes | None = None
    marker_identity: tuple[int, int] | None = None
    root_identity: tuple[int, int] | None = None
    identities: tuple[dict[str, tuple[int, int]], ...] | None = None
    batch: Phase3ActivatedBatch | None = None
    record_fingerprints: dict[tuple[str, str, str], RecordFingerprint] = {}
    try:
        root_fd, descriptors, identities = _open_store_descriptors(root, typed_stores, stack)
        scientific = _ScientificAuthorityCache.from_lease(lease)
        root_identity = identities[0]["root"]
        if any(identity["root"] != root_identity for identity in identities):
            raise Phase3ActivationError("result stores do not share one root identity")
        root_entries = _entries(root_fd)
        marker_present = ACTIVATION_MARKER_NAME in root_entries
        _validate_outer_namespaces(
            root_fd,
            typed_stores,
            descriptors,
            marker_exists=marker_present,
        )
        _validate_store_descriptors(
            typed_stores,
            descriptors,
            scientific,
            record_fingerprints,
            marker_exists=marker_present,
            initial_scan=True,
        )
        body = _marker_body(expected, typed_stores, lease, root_identity, identities)
        marker_value = _marker_with_self_hash(body)
        marker_bytes = _canonical(marker_value)
        if marker_present:
            observed = _read_file(root_fd, ACTIVATION_MARKER_NAME)
            if observed != marker_bytes:
                raise Phase3ActivationError("activation marker is conflicting or tampered")
        else:
            # Every check above has completed before this sole durable commit.
            try:
                _exclusive_publish(root_fd, ACTIVATION_MARKER_NAME, marker_bytes)
            except FileExistsError:
                observed = _read_file(root_fd, ACTIVATION_MARKER_NAME)
                if observed != marker_bytes:
                    raise Phase3ActivationError("activation marker raced with a conflict")
        marker_fd, marker_identity = _open_marker(root_fd, stack)
        batch = Phase3ActivatedBatch(
            root,
            typed_stores,
            expected,
            lease,
            root_fd,
            descriptors,
            identities,
            marker_bytes,
            marker_fd,
            marker_identity,
            scientific,
            record_fingerprints,
            tuple(
                {item.unit.unit_id: item for item in store.spec.units}
                for store in typed_stores
            ),
            _BATCH_TOKEN,
        )
        try:
            yield batch
        finally:
            try:
                batch._require_live()
                _validate_outer_namespaces(
                    root_fd,
                    typed_stores,
                    descriptors,
                    marker_exists=True,
                )
                _validate_store_descriptors(
                    typed_stores,
                    descriptors,
                    scientific,
                    record_fingerprints,
                    marker_exists=True,
                    initial_scan=False,
                )
            finally:
                batch.close()
    finally:
        stack.close()
        # Reopen only after all descriptors close, so a root/store replacement
        # is detected without ever following the replacement during operation.
        if marker_bytes is not None and root_identity is not None and identities is not None:
            current_root, current_identities = _reopen_identities(root, typed_stores)
            if current_root != root_identity or current_identities != identities:
                raise Phase3ActivationError("result output root or store changed after activation")
            try:
                with ExitStack() as check_stack:
                    check_fd = secure_fs.open_directory_chain(root)
                    check_stack.callback(os.close, check_fd)
                    with secure_fs.open_regular_file_at(check_fd, ACTIVATION_MARKER_NAME) as check_marker_fd:
                        observed = os.fstat(check_marker_fd)
                        if (
                            marker_identity is None
                            or (int(observed.st_dev), int(observed.st_ino)) != marker_identity
                            or _read_file(check_fd, ACTIVATION_MARKER_NAME) != marker_bytes
                        ):
                            raise Phase3ActivationError("activation marker changed after activation")
            except Phase3ActivationError:
                raise


# The short spelling is convenient in execution drivers and retained as a
# compatibility alias for callers which call activation a ``batch``.
activate_phase3_batch = activate_phase3_result_stores
phase3_activation = activate_phase3_result_stores


__all__ = [
    "ACTIVATION_MARKER_NAME",
    "ACTIVATION_SCHEMA_VERSION",
    "Phase3ActivatedBatch",
    "Phase3ActivationError",
    "activate_phase3_batch",
    "activate_phase3_result_stores",
    "phase3_activation",
]
