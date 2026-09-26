"""Stream JSONL: adapter → TrainingExample → emitter."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO

from convmerge.adapter_resolve import resolve_adapter
from convmerge.config import AdapterOptions, ConvertConfig
from convmerge.emitters import EmitOptions, EmitterFn, UnrepresentableExample, get_emitter
from convmerge.io import ReadStats, iter_jsonl
from convmerge.validate import validate_example

if TYPE_CHECKING:
    from convmerge.adapters import AdapterFn
    from convmerge.progress import ProgressReporter

OnInvalid = Literal["drop", "keep", "fail"]

# Line numbers remembered per drop reason, for reports.
_SAMPLE_LINES = 5


class InvalidExampleError(ValueError):
    """Raised by ``convert_file(on_invalid="fail")`` at the first invalid example."""

    def __init__(self, line_number: int, reasons: list[str]):
        super().__init__(f"line {line_number}: invalid example ({', '.join(reasons)})")
        self.line_number = line_number
        self.reasons = reasons

    def __reduce__(self):  # picklable across --workers processes
        return (InvalidExampleError, (self.line_number, self.reasons))


@dataclass
class ConvertStats:
    """Per-run counters filled by :func:`convert_file` when ``stats`` is passed.

    Every non-blank input line is either ``invalid_json``, ``non_object``,
    ``no_example`` (the adapter found nothing to map), or yields one or more
    examples (e.g. ``pairwise_mode="both"``). Each example is then either
    ``written`` or ``dropped``; ``drop_reasons`` counts why (validation reason
    codes, see :mod:`convmerge.validate`, or an output format that cannot
    represent the example) and ``drop_lines`` keeps the first few input line
    numbers per reason. With ``on_invalid="keep"``, invalid examples are
    written anyway and counted in ``kept_invalid``. ``lossy`` counts examples
    that were written but simplified by the output format (e.g.
    ``lossy_multiturn_flattened`` for ``alpaca``).
    """

    lines_read: int = 0
    written: int = 0
    blank: int = 0
    invalid_json: int = 0
    non_object: int = 0
    no_example: int = 0
    first_invalid_line: int | None = None
    dropped: int = 0
    kept_invalid: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)
    drop_lines: dict[str, list[int]] = field(default_factory=dict)
    lossy: dict[str, int] = field(default_factory=dict)

    @property
    def skipped(self) -> int:
        """Non-blank lines that produced no example at all."""
        return self.invalid_json + self.non_object + self.no_example

    def note(self, reasons: list[str], line_number: int) -> None:
        for r in reasons:
            self.drop_reasons[r] = self.drop_reasons.get(r, 0) + 1
            lines = self.drop_lines.setdefault(r, [])
            if len(lines) < _SAMPLE_LINES:
                lines.append(line_number)

    def merge(self, other: ConvertStats) -> None:
        """Add ``other`` (a later chunk of the same input) into these stats."""
        for name in (
            "lines_read", "written", "blank", "invalid_json", "non_object",
            "no_example", "dropped", "kept_invalid",
        ):  # fmt: skip
            setattr(self, name, getattr(self, name) + getattr(other, name))
        if self.first_invalid_line is None:
            self.first_invalid_line = other.first_invalid_line
        for r, n in other.drop_reasons.items():
            self.drop_reasons[r] = self.drop_reasons.get(r, 0) + n
        for r, lines in other.drop_lines.items():
            mine = self.drop_lines.setdefault(r, [])
            mine.extend(lines[: max(0, _SAMPLE_LINES - len(mine))])
        for r, n in other.lossy.items():
            self.lossy[r] = self.lossy.get(r, 0) + n

    def to_report(self) -> dict[str, object]:
        """JSON-ready summary (used by ``convert --report``)."""
        from convmerge.validate import REASONS

        report: dict[str, object] = asdict(self)
        report["skipped_lines"] = self.skipped
        report["reason_descriptions"] = {
            r: REASONS.get(r, _describe_extra(r)) for r in sorted(self.drop_reasons)
        }
        return report


def _describe_extra(reason: str) -> str:
    kind, _, media = reason.partition("_")
    if kind == "unresolved" and media:
        return f"a {media} placeholder has no matching reference in the record"
    if kind == "unused" and media:
        return f"the record lists more {media} references than placeholders"
    if reason.startswith("unrepresentable"):
        return "the output format cannot represent this example losslessly"
    return reason


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
    on_invalid: OnInvalid = "drop",
    emit_options: EmitOptions | None = None,
    workers: int = 1,
) -> tuple[int, int]:
    """
    Read JSONL lines, parse with adapter, validate, write emitted JSONL.

    ``workers > 1`` spreads parsing, conversion, and validation over that many
    processes. Output order, stats, and reports are identical to a
    single-process run. Adapters and formats are looked up by name in each
    worker, so custom ones must be registered through entry points (or at
    import time of a module the workers import).

    Each example is checked by :func:`convmerge.validate.validate_example`.
    ``on_invalid`` decides what happens to one that fails: ``"drop"``
    (default; counted in ``stats``), ``"keep"`` (written anyway), or
    ``"fail"`` (raise :class:`InvalidExampleError`).

    ``emit_options`` (:class:`convmerge.emitters.EmitOptions`) tunes the
    output format: tool-call argument encoding, ``--keep-meta``, and how
    ``alpaca`` handles multi-turn conversations. Lossy-but-kept conversions
    are counted in ``stats.lossy``.

    Set ``progress=True`` to log periodic row counts to stderr (off by default;
    see :mod:`convmerge.progress`). Pass a :class:`ConvertStats` as ``stats``
    to learn how many lines were skipped and why (invalid JSON, non-object
    rows, or records the adapter could not map).

    Returns (lines_read, lines_written).
    """
    from convmerge.progress import ProgressReporter

    if on_invalid not in ("drop", "keep", "fail"):
        raise ValueError(f"on_invalid must be 'drop', 'keep', or 'fail', got {on_invalid!r}")
    adapter = resolve_adapter(adapter_name, adapter_options)
    notes: list[str] = []
    emitter = get_emitter(output_format, options=emit_options, notes=notes)

    st = stats if stats is not None else ConvertStats()
    reporter = ProgressReporter(f"convert {input_path.name}", enabled=progress)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding=encoding) as fout:
        if workers > 1:
            _run_parallel(
                input_path,
                fout,
                st,
                reporter,
                encoding,
                workers,
                (adapter_name, adapter_options, output_format, emit_options, on_invalid),
            )
        else:
            _run(input_path, fout, adapter, emitter, st, reporter, encoding, on_invalid, notes)

    reporter.done()
    return st.lines_read, st.written


def validate_file(
    input_path: Path,
    *,
    adapter_name: str = "chat",
    adapter_options: AdapterOptions | None = None,
    encoding: str = "utf-8",
) -> ConvertStats:
    """Check every example in a JSONL file without writing anything.

    Records are read through ``adapter_name`` (default ``chat``, which
    understands the ``messages`` rows ``convert`` writes) and validated with
    :func:`convmerge.validate.validate_example`. Invalid examples are counted
    in ``dropped`` / ``drop_reasons`` / ``drop_lines`` of the returned stats.
    """
    st = ConvertStats()
    adapter = resolve_adapter(adapter_name, adapter_options)
    _run(input_path, None, adapter, None, st, None, encoding, "drop", [])
    return st


def _run(
    input_path: Path,
    fout: TextIO | None,
    adapter: AdapterFn,
    emitter: EmitterFn | None,
    st: ConvertStats,
    reporter: ProgressReporter | None,
    encoding: str,
    on_invalid: OnInvalid,
    notes: list[str],
) -> None:
    read = ReadStats()
    try:
        for line in iter_jsonl(input_path, encoding=encoding, stats=read):
            if reporter is not None:
                reporter.update()
            for row in _process(line.value, line.number, adapter, emitter, st, on_invalid, notes):
                if fout is not None:
                    fout.write(row)
    finally:
        st.lines_read = read.lines_read
        st.blank = read.blank
        st.invalid_json = read.invalid_json
        st.first_invalid_line = read.first_invalid_line


def _process(
    obj: object,
    number: int,
    adapter: AdapterFn,
    emitter: EmitterFn | None,
    st: ConvertStats,
    on_invalid: OnInvalid,
    notes: list[str],
) -> list[str]:
    """Convert one parsed record; return its output lines and update ``st``."""
    if not isinstance(obj, dict):
        st.non_object += 1
        return []
    rows: list[str] = []
    produced = 0
    for example in adapter(obj):
        produced += 1
        reasons = validate_example(example)
        if reasons:
            if on_invalid == "fail":
                raise InvalidExampleError(number, reasons)
            st.note(reasons, number)
            if on_invalid == "drop":
                st.dropped += 1
                continue
            st.kept_invalid += 1
        if emitter is None:
            st.written += 1
            continue
        try:
            row = emitter(example)
        except UnrepresentableExample as e:
            notes.clear()
            st.note([e.reason], number)
            st.dropped += 1
            continue
        rows.append(json.dumps(row, ensure_ascii=False) + "\n")
        st.written += 1
        for note in notes:
            st.lossy[note] = st.lossy.get(note, 0) + 1
        notes.clear()
    if not produced:
        st.no_example += 1
    return rows


# --- parallel convert -------------------------------------------------------

_CHUNK_LINES = 2_000
_WORKER: dict[str, object] = {}


def _worker_init(spec: tuple) -> None:
    adapter_name, adapter_options, output_format, emit_options, on_invalid = spec
    notes: list[str] = []
    _WORKER.update(
        adapter=resolve_adapter(adapter_name, adapter_options),
        emitter=get_emitter(output_format, options=emit_options, notes=notes),
        notes=notes,
        on_invalid=on_invalid,
    )


def _worker_chunk(chunk: list[tuple[int, str]]) -> tuple[str, ConvertStats]:
    st = ConvertStats()
    out: list[str] = []
    for number, raw in chunk:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            st.invalid_json += 1
            if st.first_invalid_line is None:
                st.first_invalid_line = number
            continue
        out.extend(
            _process(
                obj,
                number,
                _WORKER["adapter"],  # type: ignore[arg-type]
                _WORKER["emitter"],  # type: ignore[arg-type]
                st,
                _WORKER["on_invalid"],  # type: ignore[arg-type]
                _WORKER["notes"],  # type: ignore[arg-type]
            )
        )
    return "".join(out), st


def _raw_chunks(path: Path, encoding: str, st: ConvertStats):
    """Yield chunks of ``(line_number, text)``, counting lines and blanks in ``st``.

    Mirrors :func:`convmerge.io.iter_jsonl`; JSON is parsed in the workers.
    """
    chunk: list[tuple[int, str]] = []
    with path.open(encoding=encoding) as f:
        for number, line in enumerate(f, 1):
            st.lines_read += 1
            raw = line.strip()
            if number == 1:
                raw = raw.removeprefix("\ufeff").strip()
            if not raw:
                st.blank += 1
                continue
            chunk.append((number, raw))
            if len(chunk) >= _CHUNK_LINES:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def _run_parallel(
    input_path: Path,
    fout: TextIO,
    st: ConvertStats,
    reporter: ProgressReporter,
    encoding: str,
    workers: int,
    spec: tuple,
) -> None:
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor

    # A bounded window of in-flight chunks keeps memory flat on huge inputs
    # (Pool.imap would read the whole file ahead of the workers).
    window = workers * 4
    with ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(spec,)) as pool:
        pending: deque = deque()

        def drain_one() -> None:
            text, part = pending.popleft().result()
            fout.write(text)
            st.merge(part)

        for chunk in _raw_chunks(input_path, encoding, st):
            pending.append(pool.submit(_worker_chunk, chunk))
            reporter.update(len(chunk))
            if len(pending) >= window:
                drain_one()
        while pending:
            drain_one()


def convert_with_config(
    input_path: Path,
    output_path: Path,
    cfg: ConvertConfig,
    *,
    progress: bool = False,
    stats: ConvertStats | None = None,
    on_invalid: OnInvalid = "drop",
    emit_options: EmitOptions | None = None,
    workers: int = 1,
) -> tuple[int, int]:
    """Run :func:`convert_file` using a resolved :class:`convmerge.config.ConvertConfig`.

    ``emit_options`` defaults to ``cfg.emit_options`` (from a preset's
    ``output_options``).
    """
    return convert_file(
        input_path,
        output_path,
        adapter_name=cfg.adapter,
        output_format=cfg.output_format,
        encoding=cfg.encoding,
        adapter_options=cfg.adapter_options,
        progress=progress,
        stats=stats,
        on_invalid=on_invalid,
        emit_options=emit_options if emit_options is not None else cfg.emit_options,
        workers=workers,
    )


def iter_converted_lines(
    lines: Iterator[str],
    *,
    adapter_name: str,
    output_format: str,
    adapter_options: AdapterOptions | None = None,
) -> Iterator[str]:
    """In-memory conversion (for tests); no validation."""
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
