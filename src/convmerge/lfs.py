"""Small helpers for detecting Git LFS pointer files."""

from __future__ import annotations

from pathlib import Path

LFS_POINTER_VERSION = b"version https://git-lfs.github.com/spec/v1"


class LfsPointerError(ValueError):
    """Raised when a Git LFS pointer is used as dataset content."""


def is_lfs_pointer(data: bytes) -> bool:
    """Return whether *data* starts with a Git LFS pointer header."""
    # Only the head matters; avoid splitting a multi-GB download into lines.
    first_line = data[:256].split(b"\n", 1)[0].strip()
    return first_line == LFS_POINTER_VERSION


def ensure_not_lfs_pointer(path: str | Path) -> None:
    """Reject a local Git LFS pointer with an actionable recovery message."""
    source = Path(path)
    with source.open("rb") as handle:
        head = handle.read(256)
    if is_lfs_pointer(head):
        raise LfsPointerError(
            f"{source} is a Git LFS pointer, not the underlying dataset blob; "
            "fetch this source with mode: clone and lfs: true"
        )
