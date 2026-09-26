"""Stream JSONL: adapter → TrainingExample → emitter."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from convmerge.adapter_resolve import resolve_adapter
from convmerge.config import AdapterOptions, ConvertConfig
from convmerge.emitters import get_emitter


@dataclass
class ConvertStats:
    """Per-run counters filled by :func:`convert_file` when ``stats`` is passed.

    Every non-blank input line lands in exactly one of ``invalid_json``,
    ``non_object``, ``no_example``, or contributes to ``written`` (one record
    may yield several examples, e.g. ``pairwise_mode="both"``).
    """

    lines_read: int = 0
    written: int = 0
    blank: int = 0
    invalid_json: int = 0
    non_object: int = 0
    no_example: int = 0
    first_invalid_line: int | None = None

    @property
    def skipped(self) -> int:
        """Non-blank lines that produced no output."""
        return self.invalid_json + self.non_object + self.no_example


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    adapter_name: str,
    output_format: str,
    encoding: str = "utf-8",
    adapter_options: AdapterOptions | None = None,
    progress: bool = False,
    stats: ConvertStats | None = None,
) -> tuple[int, int]:
    """
    Read JSONL lines, parse with adapter, write emitted JSONL.

    Set ``progress=True`` to log periodic row counts to stderr (off by default;
    see :mod:`convmerge.progress`). Pass a :class:`ConvertStats` as ``stats``
    to learn how many lines were skipped and why (invalid JSON, non-object
    rows, or records the adapter could not map).

    Returns (lines_read, lines_written).
    """
    from convmerge.progress import ProgressReporter

    adapter = resolve_adapter(adapter_name, adapter_options)
    emitter = get_emitter(output_format)

    st = stats if stats is not None else ConvertStats()
    reporter = ProgressReporter(f"convert {input_path.name}", enabled=progress)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open(encoding=encoding) as fin,
        output_path.open("w", encoding=encoding) as fout,
    ):
        for raw in fin:
            st.lines_read += 1
            reporter.update()
            raw = raw.strip()
            if not raw:
                st.blank += 1
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                st.invalid_json += 1
                if st.first_invalid_line is None:
                    st.first_invalid_line = st.lines_read
                continue
            if not isinstance(obj, dict):
                st.non_object += 1
                continue
            produced = 0
            for example in adapter(obj):
                row = emitter(example)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                produced += 1
            if produced:
                st.written += produced
            else:
                st.no_example += 1

    reporter.done()
    return st.lines_read, st.written


def convert_with_config(
    input_path: Path,
    output_path: Path,
    cfg: ConvertConfig,
    *,
    progress: bool = False,
    stats: ConvertStats | None = None,
) -> tuple[int, int]:
    """Run :func:`convert_file` using a resolved :class:`convmerge.config.ConvertConfig`."""
    return convert_file(
        input_path,
        output_path,
        adapter_name=cfg.adapter,
        output_format=cfg.output_format,
        encoding=cfg.encoding,
        adapter_options=cfg.adapter_options,
        progress=progress,
        stats=stats,
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
