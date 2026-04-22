"""Stream JSONL: adapter → TrainingExample → emitter."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from convmerge.adapter_resolve import resolve_adapter
from convmerge.config import AdapterOptions, ConvertConfig
from convmerge.emitters import get_emitter


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    adapter_name: str,
    output_format: str,
    encoding: str = "utf-8",
    adapter_options: AdapterOptions | None = None,
) -> tuple[int, int]:
    """
    Read JSONL lines, parse with adapter, write emitted JSONL.

    Returns (lines_read, lines_written).
    """
    adapter = resolve_adapter(adapter_name, adapter_options)
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


def convert_with_config(
    input_path: Path,
    output_path: Path,
    cfg: ConvertConfig,
) -> tuple[int, int]:
    """Run :func:`convert_file` using a resolved :class:`convmerge.config.ConvertConfig`."""
    return convert_file(
        input_path,
        output_path,
        adapter_name=cfg.adapter,
        output_format=cfg.output_format,
        encoding=cfg.encoding,
        adapter_options=cfg.adapter_options,
    )


def iter_converted_lines(
    lines: Iterator[str],
    *,
    adapter_name: str,
    output_format: str,
    adapter_options: AdapterOptions | None = None,
) -> Iterator[str]:
    """In-memory conversion (for tests)."""
    adapter = resolve_adapter(adapter_name, adapter_options)
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
