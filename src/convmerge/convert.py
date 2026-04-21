"""Stream JSONL: adapter → TrainingExample → emitter."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from convmerge.adapters import get_adapter
from convmerge.emitters import get_emitter


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    adapter_name: str,
    output_format: str,
    encoding: str = "utf-8",
) -> tuple[int, int]:
    """
    Read JSONL lines, parse with adapter, write emitted JSONL.

    Returns (lines_read, lines_written).
    """
    adapter = get_adapter(adapter_name)
    emitter = get_emitter(output_format)

    lines_read = 0
    lines_written = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open(encoding=encoding) as fin,
        output_path.open("w", encoding=encoding) as fout,
    ):
        for raw in fin:
            lines_read += 1
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            for example in adapter(obj):
                row = emitter(example)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                lines_written += 1

    return lines_read, lines_written


def iter_converted_lines(
    lines: Iterator[str],
    *,
    adapter_name: str,
    output_format: str,
) -> Iterator[str]:
    """In-memory conversion (for tests)."""
    adapter = get_adapter(adapter_name)
    emitter = get_emitter(output_format)
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            continue
        for example in adapter(obj):
            yield json.dumps(emitter(example), ensure_ascii=False)


def convert_dir(
    src_dir: str | Path,
    dst_dir: str | Path,
    *,
    adapter_name: str,
    output_format: str,
    extensions: Iterable[str] = (".jsonl",),
    on_error: Literal["skip", "raise"] = "skip",
    encoding: str = "utf-8",
) -> tuple[int, int, int]:
    """Recursively run :func:`convert_file` over every JSONL under ``src_dir``.

    Relative paths are preserved under ``dst_dir``. Files whose suffix is not
    in ``extensions`` (case-insensitive) are ignored.

    Returns ``(n_files, total_read, total_written)``.
    """
    src_p = Path(src_dir)
    dst_p = Path(dst_dir)
    if not src_p.is_dir():
        raise NotADirectoryError(f"Not a directory: {src_p}")

    exts = tuple(e.lower() for e in extensions)
    n_files = 0
    total_read = 0
    total_written = 0

    for in_path in sorted(src_p.rglob("*")):
        if not in_path.is_file():
            continue
        if in_path.suffix.lower() not in exts:
            continue
        rel = in_path.relative_to(src_p)
        out_path = dst_p / rel
        try:
            n_in, n_out = convert_file(
                in_path,
                out_path,
                adapter_name=adapter_name,
                output_format=output_format,
                encoding=encoding,
            )
        except Exception as e:  # noqa: BLE001
            if on_error == "raise":
                raise
            print(f"[fail] {in_path}: {type(e).__name__}: {e}")
            continue
        n_files += 1
        total_read += n_in
        total_written += n_out

    return n_files, total_read, total_written
