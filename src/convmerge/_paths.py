"""Internal preflight for protected files/trees and distinct output paths."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from convmerge.io import SamePathError


class _PathIndex:
    """Index lexical ancestry, resolved symlinks and existing hardlink identities."""

    def __init__(self) -> None:
        self.paths: dict[Path, Path] = {}
        self.parents: dict[Path, Path] = {}
        self.inodes: dict[tuple[int, int], Path] = {}

    def add(self, path: Path) -> None:
        resolved = path.resolve()
        self.paths.setdefault(resolved, path)
        for parent in resolved.parents:
            self.parents.setdefault(parent, path)
        try:
            info = path.stat()
        except FileNotFoundError:
            return
        self.inodes.setdefault((info.st_dev, info.st_ino), path)

    def overlap(self, path: Path) -> Path | None:
        resolved = path.resolve()
        for ancestor in (resolved, *resolved.parents):
            if ancestor in self.paths:
                return self.paths[ancestor]
        if resolved in self.parents:
            return self.parents[resolved]
        try:
            info = path.stat()
        except FileNotFoundError:
            return None
        return self.inodes.get((info.st_dev, info.st_ino))


def protect_paths(
    inputs: Iterable[str | Path | None], outputs: Iterable[str | Path | None]
) -> None:
    """Refuse aliases and containment before any output is opened.

    Directory inputs protect their complete trees, including hardlinked files.
    Output paths must also be distinct and non-overlapping. This is a preflight,
    not a lock against concurrent path mutation.
    """
    protected = _PathIndex()
    for value in inputs:
        if value is None:
            continue
        path = Path(value)
        protected.add(path)
        if path.is_dir():
            for child in path.rglob("*"):
                protected.add(child)
    written = _PathIndex()
    for value in outputs:
        if value is None:
            continue
        path = Path(value)
        conflict = protected.overlap(path)
        if conflict is not None:
            raise SamePathError(
                f"output {path} overlaps protected input {conflict}; write to a different path"
            )
        conflict = written.overlap(path)
        if conflict is not None:
            raise SamePathError(
                f"output {path} overlaps another output {conflict}; use separate paths"
            )
        written.add(path)
