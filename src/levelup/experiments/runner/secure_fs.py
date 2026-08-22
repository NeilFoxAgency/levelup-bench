"""Small POSIX descriptor-relative filesystem primitives.

These helpers deliberately do not import runner storage or any artifact model.  A
caller pins its trusted root once, then resolves every descendant relative to
that descriptor with ``O_NOFOLLOW``.  This keeps a rename/substitution of the
root path from redirecting a later read through a replacement tree.
"""

from __future__ import annotations

import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class SecureFilesystemError(RuntimeError):
    """Raised when descriptor-relative storage cannot be opened safely."""


def _require_support() -> None:
    required_dir_fd = (os.open, os.stat, os.unlink)
    if (
        os.name != "posix"
        or any(function not in os.supports_dir_fd for function in required_dir_fd)
        or os.scandir not in os.supports_fd
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
    ):
        raise SecureFilesystemError(
            "secure artifact reads require POSIX directory-fd and O_NOFOLLOW support"
        )


def _directory_flags() -> int:
    _require_support()
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _component(name: str) -> str:
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise SecureFilesystemError("invalid descriptor-relative path component")
    if "/" in name or "\\" in name or "\x00" in name:
        raise SecureFilesystemError("descriptor-relative path must contain one component")
    return name


def open_directory_chain(path: str | Path) -> int:
    """Open an absolute directory path one component at a time, no symlinks."""

    absolute = Path(os.path.abspath(path))
    directory_fd: int | None = None
    try:
        directory_fd = os.open(absolute.anchor, _directory_flags())
        for component in absolute.parts[1:]:
            child_fd = os.open(_component(component), _directory_flags(), dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        assert directory_fd is not None
        result = directory_fd
        directory_fd = None
        return result
    except SecureFilesystemError:
        raise
    except (AttributeError, NotImplementedError, OSError, TypeError, ValueError) as exc:
        raise SecureFilesystemError(f"cannot securely open directory: {path}") from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def open_child_directory(parent_fd: int, name: str) -> int:
    """Open one directory below ``parent_fd`` without following a symlink."""

    try:
        return os.open(_component(name), _directory_flags(), dir_fd=parent_fd)
    except SecureFilesystemError:
        raise
    except (AttributeError, NotImplementedError, OSError, TypeError, ValueError) as exc:
        raise SecureFilesystemError(f"cannot securely open child directory: {name}") from exc


def open_child_chain(parent_fd: int, *components: str) -> int:
    """Open a non-empty child-directory chain and return its final descriptor."""

    if not components:
        raise SecureFilesystemError("descriptor-relative child chain cannot be empty")
    current_fd: int | None = None
    try:
        for component in components:
            child_fd = open_child_directory(parent_fd if current_fd is None else current_fd, component)
            if current_fd is not None:
                os.close(current_fd)
            current_fd = child_fd
        assert current_fd is not None
        result = current_fd
        current_fd = None
        return result
    finally:
        if current_fd is not None:
            os.close(current_fd)


def directory_identity(directory_fd: int) -> tuple[int, int]:
    """Return the device/inode identity of a directory descriptor."""

    try:
        observed = os.fstat(directory_fd)
    except OSError as exc:
        raise SecureFilesystemError("cannot stat directory descriptor") from exc
    if not stat.S_ISDIR(observed.st_mode):
        raise SecureFilesystemError("descriptor is not a directory")
    return observed.st_dev, observed.st_ino


@contextmanager
def open_regular_file_at(directory_fd: int, name: str) -> Iterator[int]:
    """Yield a non-symlink regular file opened relative to a directory fd."""

    fd: int | None = None
    try:
        fd = os.open(_component(name), os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode):
            raise SecureFilesystemError(f"descriptor entry is not a regular file: {name}")
        yield fd
    except SecureFilesystemError:
        raise
    except (AttributeError, NotImplementedError, OSError, TypeError, ValueError) as exc:
        raise SecureFilesystemError(f"cannot securely open file: {name}") from exc
    finally:
        if fd is not None:
            os.close(fd)


def read_bytes_at(directory_fd: int, name: str) -> bytes:
    """Read one regular file without resolving a path after the root is pinned."""

    try:
        with open_regular_file_at(directory_fd, name) as fd:
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
    except SecureFilesystemError:
        raise
    except OSError as exc:
        raise SecureFilesystemError(f"cannot read descriptor entry: {name}") from exc


def read_json_at(directory_fd: int, name: str) -> Any:
    """Parse JSON from one descriptor-relative regular file."""

    try:
        return json.loads(read_bytes_at(directory_fd, name))
    except SecureFilesystemError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SecureFilesystemError(f"invalid JSON descriptor entry: {name}") from exc


def strict_regular_entries(directory_fd: int) -> tuple[str, ...]:
    """Enumerate a directory, rejecting symlinks and every non-regular entry."""

    entries: list[str] = []
    try:
        with os.scandir(directory_fd) as iterator:
            for entry in iterator:
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise SecureFilesystemError(
                        f"directory contains a symlink or non-regular entry: {entry.name}"
                    )
                entries.append(entry.name)
    except SecureFilesystemError:
        raise
    except (NotImplementedError, OSError) as exc:
        raise SecureFilesystemError("cannot enumerate descriptor directory") from exc
    return tuple(sorted(entries))


# Explicit aliases make call sites read naturally while retaining one
# implementation of each primitive.
open_directory = open_directory_chain
open_child_directories = open_child_chain
read_regular_bytes_at = read_bytes_at
load_json_at = read_json_at
enumerate_regular_entries_at = strict_regular_entries
regular_entries_at = strict_regular_entries
