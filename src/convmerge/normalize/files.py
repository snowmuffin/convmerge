"""Normalize a file or a whole directory tree into clean JSONL.

This is the logic behind ``convmerge normalize``; recipes call it directly.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from convmerge.io import SamePathError

NORMALIZE_EXTENSIONS: tuple[str, ...] = (".parquet", ".json", ".jsonl")

# Sidecars convmerge itself writes next to data files; never treat them as data.
SIDECAR_SUFFIXES: tuple[str, ...] = (".fetch.json", ".mix.json")


@dataclass
class NormalizeResult:
    """What :func:`normalize_path` wrote: ``(source, output, records)`` per file."""

    files: list[tuple[Path, Path, int]] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def records(self) -> int:
        return sum(n for _, _, n in self.files)


def normalize_file(
    src: Path, dst: Path, *, array_key: str = "conversation", sheet: str | None = None
) -> int:
    """Normalize one ``.parquet`` / ``.json`` / ``.jsonl`` file, or a ``.csv`` /
    ``.tsv`` table or ``.xlsx`` workbook sheet (one object per row, see
    :mod:`convmerge.normalize.tabular`; ``sheet`` picks the sheet, default the
    first); return records written. Directory walks (:func:`iter_data_files`)
    skip tables: pass a table file directly."""
    # Imported lazily so that convert works without the parquet extra.
    from convmerge.io import refuse_overwrite
    from convmerge.normalize.jsonl import normalize_to_jsonl
    from convmerge.normalize.tabular import (
        TABLE_EXTENSIONS,
        XLSX_EXTENSIONS,
        table_to_jsonl,
        xlsx_to_jsonl,
    )

    refuse_overwrite([src], [dst])
    suffix = src.suffix.lower()
    if suffix in XLSX_EXTENSIONS or suffix == ".xls":
        return xlsx_to_jsonl(src, dst, sheet=sheet)
    if sheet is not None:
        raise ValueError(f"{src}: sheet applies to .xlsx workbooks only")
    if suffix in TABLE_EXTENSIONS:
        return table_to_jsonl(src, dst)
    if src.suffix.lower() == ".parquet":
        from convmerge.normalize.parquet import parquet_to_jsonl

        dst.parent.mkdir(parents=True, exist_ok=True)
        return parquet_to_jsonl(src, dst)
    return normalize_to_jsonl(src, dst, array_key=array_key)


def iter_data_files(root: Path) -> Iterator[Path]:
    """Data files under ``root`` in sorted order, skipping sidecars and hidden paths."""
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in NORMALIZE_EXTENSIONS:
            continue
        if path.name.lower().endswith(SIDECAR_SUFFIXES):
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            # Hidden entries such as a cloned repo's .git directory.
            continue
        yield path


def normalize_path(
    src: Path,
    dst: Path,
    *,
    array_key: str = "conversation",
    on_file: Callable[[Path, Path, int | None, str | None], None] | None = None,
    sheet: str | None = None,
) -> NormalizeResult:
    """Normalize ``src`` (a file, written to ``dst``) or a directory (mirrored under ``dst``).

    In a directory, a file that fails is recorded in ``result.failed`` and the
    walk continues. ``on_file(src, dst, records, error)`` is called per file.
    ``sheet`` names the sheet of an ``.xlsx`` file given as ``src``.
    """
    result = NormalizeResult()
    if src.is_file():
        n = normalize_file(src, dst, array_key=array_key, sheet=sheet)
        result.files.append((src, dst, n))
        if on_file:
            on_file(src, dst, n, None)
        return result
    if not src.is_dir():
        raise FileNotFoundError(f"input not found: {src}")
    if sheet is not None:
        raise ValueError("sheet applies to one .xlsx file, not a directory")
    if dst.resolve().is_relative_to(src.resolve()):
        raise SamePathError(
            f"output directory {dst} is inside or is the input directory; "
            "write to a separate directory"
        )
    inputs = list(iter_data_files(src))
    outputs = [dst / p.relative_to(src).with_suffix(".jsonl") for p in inputs]
    _check_output_map(inputs, outputs)
    for in_path, out_path in zip(inputs, outputs):
        try:
            n = normalize_file(in_path, out_path, array_key=array_key)
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the walk
            msg = f"{type(e).__name__}: {e}"
            result.failed.append((in_path, msg))
            if on_file:
                on_file(in_path, out_path, None, msg)
            continue
        result.files.append((in_path, out_path, n))
        if on_file:
            on_file(in_path, out_path, n, None)
    return result


def _path_keys(path: Path) -> set[str | tuple[int, int]]:
    keys: set[str | tuple[int, int]] = {os.path.normcase(str(path.resolve()))}
    try:
        info = path.stat()
    except FileNotFoundError:
        pass
    else:
        keys.add((info.st_dev, info.st_ino))
    return keys


def _check_output_map(inputs: list[Path], outputs: list[Path]) -> None:
    """Linear-size preflight, including cross-file and existing hardlink aliases."""
    protected: set[str | tuple[int, int]] = set()
    for path in inputs:
        protected.update(_path_keys(path))
    seen: set[str | tuple[int, int]] = set()
    for path in outputs:
        keys = _path_keys(path)
        if keys & protected:
            raise SamePathError(f"output {path} is one of the input files; choose another path")
        if keys & seen:
            raise SamePathError(f"output {path} is given twice; use distinct source stems")
        seen.update(keys)
