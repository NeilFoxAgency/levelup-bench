"""Capture bounded, secret-conscious execution provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from levelup.experiments.runner.config import DevicePolicy
from levelup.experiments.runner.records import SystemProvenance


def utc_now() -> datetime:
    return datetime.now(UTC)


def _git(repository: Path, *args: str) -> bytes:
    return subprocess.run(
        ("git", *args),
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout


def _memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(page_size, int) or not isinstance(pages, int):
        return None
    return page_size * pages


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("levelup-bench", "numpy", "pydantic", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _installed_packages_sha256() -> str:
    rows = sorted(
        (
            (distribution.metadata.get("Name") or "unknown").casefold(),
            distribution.version,
        )
        for distribution in importlib.metadata.distributions()
    )
    digest = hashlib.sha256()
    for name, version in rows:
        digest.update(name.encode("utf-8"))
        digest.update(b"==")
        digest.update(version.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _dirty_tree_sha256(repository: Path, status: bytes) -> str:
    """Hash tracked diffs and untracked contents without persisting either."""

    digest = hashlib.sha256()
    digest.update(b"status\0")
    digest.update(status)
    digest.update(b"tracked\0")
    digest.update(_git(repository, "diff", "--binary", "HEAD"))
    untracked = _git(
        repository,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ).split(b"\0")
    for encoded_path in sorted(path for path in untracked if path):
        relative_path = encoded_path.decode("utf-8", errors="surrogateescape")
        path = repository / relative_path
        digest.update(b"untracked-path\0")
        digest.update(encoded_path)
        digest.update(b"\0untracked-content\0")
        if path.is_symlink():
            digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
        else:
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_device(requested: str) -> str:
    """Validate the requested device without silently selecting another one."""

    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(f"requested device {requested!r} requires PyTorch") from exc
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    raise RuntimeError(f"requested device {requested!r} is unavailable")


def apply_runtime_policy(device_policy: DevicePolicy) -> str:
    """Apply the declared PyTorch policy or fail instead of silently drifting."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("experiment runtime policy requires the 'ml' dependencies") from exc
    resolved_device = resolve_device(device_policy.requested_device)
    torch.set_num_threads(device_policy.torch_threads)
    current_interop = torch.get_num_interop_threads()
    if current_interop != device_policy.torch_interop_threads:
        try:
            torch.set_num_interop_threads(device_policy.torch_interop_threads)
        except RuntimeError as exc:
            raise RuntimeError(
                "PyTorch interop threads were initialized before experiment policy application"
            ) from exc
    torch.use_deterministic_algorithms(device_policy.deterministic_algorithms)
    return resolved_device


def capture_system_provenance(
    repository: str | Path,
    device_policy: DevicePolicy,
) -> SystemProvenance:
    """Record reproducibility fields without paths, environment variables, or host identity."""

    repository_path = Path(repository)
    commit = _git(repository_path, "rev-parse", "HEAD").decode().strip()
    status = _git(repository_path, "status", "--porcelain=v1", "-z")
    dirty = bool(status)
    diff_hash: str | None = None
    if dirty:
        diff_hash = _dirty_tree_sha256(repository_path, status)

    try:
        import torch
    except ImportError:
        actual_threads = None
        actual_interop_threads = None
        deterministic_algorithms = None
    else:
        actual_threads = torch.get_num_threads()
        actual_interop_threads = torch.get_num_interop_threads()
        deterministic_algorithms = torch.are_deterministic_algorithms_enabled()

    return SystemProvenance(
        git_commit_sha=commit,
        git_dirty=dirty,
        git_diff_sha256=diff_hash,
        python_version=sys.version,
        packages=_package_versions(),
        installed_packages_sha256=_installed_packages_sha256(),
        os=platform.platform(),
        architecture=platform.machine(),
        cpu=platform.processor(),
        cpu_count=os.cpu_count(),
        memory_bytes=_memory_bytes(),
        requested_device=device_policy.requested_device,
        resolved_device=resolve_device(device_policy.requested_device),
        requested_torch_threads=device_policy.torch_threads,
        actual_torch_threads=actual_threads,
        requested_torch_interop_threads=device_policy.torch_interop_threads,
        actual_torch_interop_threads=actual_interop_threads,
        deterministic_algorithms_requested=device_policy.deterministic_algorithms,
        deterministic_algorithms_actual=deterministic_algorithms,
        processes=device_policy.processes,
        captured_at_utc=utc_now(),
    )
