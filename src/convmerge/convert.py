"""Stream JSONL: adapter → TrainingExample → emitter."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TextIO

from convmerge import _parallel
from convmerge.adapter_resolve import resolve_adapter
from convmerge.adapters import oasst
from convmerge.config import AdapterOptions, ConvertConfig
from convmerge.emitters import (
    EmitOptions,
    EmitterFn,
    UnrepresentableExample,
    get_emitter,
    split_pair,
    wants_pairs,
)
from convmerge.io import ReadStats, iter_jsonl, refuse_overwrite
from convmerge.models import TrainingExample
from convmerge.reasoning import has_reasoning
from convmerge.transforms import TransformOptions, apply_transforms
from convmerge.validate import ISSUES, validate_example

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
    ``no_example`` (the adapter found nothing to map), ``grouped`` into an
    earlier line's record (OpenAssistant message rows), or yields one or more
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
    transforms: dict[str, int] = field(default_factory=dict)
    """How often each ``TransformOptions`` fix changed an example (see
    :data:`convmerge.transforms.TRANSFORM_COUNTERS`)."""
    reasoning: int = 0
    """Written examples with a reasoning trace (a field or an inline ``<think>``)."""
    grouped: int = 0
    """Input rows folded into the record of an earlier row: the message rows
    of one OpenAssistant tree become one conversation (see
    :mod:`convmerge.adapters.oasst`). They count in ``lines_read`` only."""

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
            "no_example", "dropped", "kept_invalid", "reasoning", "grouped",
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
        for r, n in other.transforms.items():
            self.transforms[r] = self.transforms.get(r, 0) + n

    def to_report(self) -> dict[str, object]:
        """JSON-ready summary (used by ``convert --report``)."""
        from convmerge.validate import REASONS

        report: dict[str, object] = asdict(self)
        report["skipped_lines"] = self.skipped
        report["reason_descriptions"] = {
            r: REASONS.get(r, _describe_extra(r)) for r in sorted(self.drop_reasons)
        }
        return report


# Schema version of the JSON written by ``convert --report`` and ``validate``;
# bumped only for incompatible changes (fields may be added without a bump).
REPORT_VERSION = 1


_UNREPRESENTABLE: dict[str, str] = {
    "unrepresentable_not_preference": (
        "not a chosen/rejected pair (the preference format needs one)"
    ),
    "unrepresentable_identical_pair": "the chosen and rejected conversations are identical",
    "unrepresentable_role_order": (
        "the turns do not alternate user/assistant as the format requires "
        "(e.g. two user turns in a row, or a system turn mid-conversation)"
    ),
    "unrepresentable_pair_continuation": (
        "an answer in the pair is not a single text reply (multi-turn or tool call)"
    ),
    "unrepresentable_incomplete_pair": (
        "the pair has no user prompt, or one side has no assistant answer after the prompt"
    ),
}


def _describe_extra(reason: str) -> str:
    if reason in ISSUES:
        return ISSUES[reason]
    if reason in _UNREPRESENTABLE:
        return _UNREPRESENTABLE[reason]
    kind, _, media = reason.partition("_")
    if kind == "unresolved" and media:
        return f"a {media} placeholder has no matching reference in the record"
    if kind == "unused" and media:
        return f"the record lists more {media} references than placeholders"
    if reason.startswith("unrepresentable"):
        return "the output format cannot represent this example losslessly"
    return reason


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
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
    transform_options: TransformOptions | None = None,
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

    ``transform_options`` (:class:`convmerge.transforms.TransformOptions`)
    fixes conversations for strict chat templates before validation (fold
    system turns, merge consecutive turns, split at user turns, keep only the
    last turn's reasoning); each fix is counted in ``stats.transforms``.

    Set ``progress=True`` to log periodic row counts to stderr (off by default;
    see :mod:`convmerge.progress`). Pass a :class:`ConvertStats` as ``stats``
    to learn how many lines were skipped and why (invalid JSON, non-object
    rows, or records the adapter could not map).

    Returns (lines_read, lines_written).
    """
    refuse_overwrite([input_path], [output_path])
    from convmerge.progress import ProgressReporter

    input_path, output_path = Path(input_path), Path(output_path)
    if on_invalid not in ("drop", "keep", "fail"):
        raise ValueError(f"on_invalid must be 'drop', 'keep', or 'fail', got {on_invalid!r}")
    notes: list[str] = []
    emitter = get_emitter(output_format, options=emit_options, notes=notes)
    pairs = wants_pairs(output_format)
    check_transforms(transform_options, pairs=pairs)
    adapter = resolve_adapter(adapter_name, adapter_options, pairs=pairs)
    transform = _transformer(transform_options)

    st = stats if stats is not None else ConvertStats()
    reporter = ProgressReporter(f"convert {input_path.name}", enabled=progress)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if workers > 1 and _groups_rows(input_path, encoding):
        workers = 1  # a tree's rows must reach one process together; reading is fast anyway

    with output_path.open("w", encoding="utf-8") as fout:
        if workers > 1:
            _run_parallel(
                input_path,
                fout,
                st,
                reporter,
                encoding,
                workers,
                (
                    adapter_name,
                    adapter_options,
                    output_format,
                    emit_options,
                    on_invalid,
                    transform_options,
                ),
            )
        else:
            _run(
                input_path,
                fout,
                adapter,
                emitter,
                st,
                reporter,
                encoding,
                on_invalid,
                notes,
                transform,
            )

    reporter.done()
    return st.lines_read, st.written


def check_transforms(options: TransformOptions | None, *, pairs: bool) -> None:
    """Reject option combinations that cannot work (``split_turns`` with pairs)."""
    if options is not None and options.split_turns and pairs:
        raise ValueError(
            "split_turns (--split-turns) cannot split preference pairs; use it with an SFT format"
        )


Transform = Callable[[TrainingExample, dict[str, int]], list[TrainingExample]]


def _transformer(options: TransformOptions | None) -> Transform | None:
    if options is None or not options.active:
        return None
    return partial(_apply, options=options)


def _apply(
    example: TrainingExample, counts: dict[str, int], *, options: TransformOptions
) -> list[TrainingExample]:
    return apply_transforms(example, options, counts)


def validate_file(
    input_path: str | Path,
    *,
    adapter_name: str = "chat",
    adapter_options: AdapterOptions | None = None,
    encoding: str = "utf-8",
) -> ConvertStats:
    """Check every example in a JSONL file without writing anything.

    Records are read through ``adapter_name`` (default ``chat``, which
    understands the ``messages`` and ``preference`` rows ``convert`` writes)
    and validated with :func:`convmerge.validate.validate_example`; preference
    rows must also form a usable chosen/rejected pair. Invalid examples are counted
    in ``dropped`` / ``drop_reasons`` / ``drop_lines`` of the returned stats.
    """
    input_path = Path(input_path)
    st = ConvertStats()
    # Preference rows (prompt / chosen / rejected) are read as pairs and must
    # also form a usable pair; other rows are checked as before.
    pairs = adapter_options is None or not adapter_options.preference
    adapter = resolve_adapter(adapter_name, adapter_options, pairs=pairs)
    _run(input_path, None, adapter, _check_pair, st, None, encoding, "drop", [])
    return st


def _check_pair(example: TrainingExample) -> dict[str, Any]:
    if example.rejected is not None:
        split_pair(example)
    return {}


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
    transform: Transform | None = None,
) -> None:
    read = ReadStats()
    trees = oasst.TreeBuffer()  # OpenAssistant message rows, one tree at a time

    def flush(done: tuple[int, Any, int] | None) -> None:
        if done is not None:
            st.grouped += done[2]
            for row in _process(done[1], done[0], adapter, emitter, st, on_invalid, notes,
                                transform):  # fmt: skip
                if fout is not None:
                    fout.write(row)

    try:
        for line in iter_jsonl(input_path, encoding=encoding, stats=read):
            if reporter is not None:
                reporter.update()
            value = line.value
            if type(value) is dict and "message_tree_id" in value and oasst.is_message(value):
                flush(trees.add(line.number, value))
                continue
            if trees.rows:
                flush(trees.flush())
            rows = _process(value, line.number, adapter, emitter, st, on_invalid, notes, transform)
            for row in rows:
                if fout is not None:
                    fout.write(row)
        flush(trees.flush())
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
    transform: Transform | None = None,
) -> list[str]:
    """Convert one parsed record; return its output lines and update ``st``."""
    rows = _process_rows(obj, number, adapter, emitter, st, on_invalid, notes, transform)
    return [json.dumps(row, ensure_ascii=False) + "\n" for row in rows]


def _process_rows(
    obj: object,
    number: int,
    adapter: AdapterFn,
    emitter: EmitterFn | None,
    st: ConvertStats,
    on_invalid: OnInvalid,
    notes: list[str],
    transform: Transform | None = None,
) -> list[dict[str, Any]]:
    """Convert one parsed record; return its output rows and update ``st``."""
    if not isinstance(obj, dict):
        st.non_object += 1
        return []
    rows: list[dict[str, Any]] = []
    produced = 0
    for example in _examples(adapter(obj), transform, st):
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
        rows.append(row)
        st.written += 1
        if _has_reasoning(example):
            st.reasoning += 1
        for note in notes:
            st.lossy[note] = st.lossy.get(note, 0) + 1
        notes.clear()
    if not produced:
        st.no_example += 1
    return rows


def convert_records(
    records: Iterable[Any],
    *,
    adapter_name: str = "auto",
    output_format: str = "messages",
    adapter_options: AdapterOptions | None = None,
    emit_options: EmitOptions | None = None,
    transform_options: TransformOptions | None = None,
    on_invalid: OnInvalid = "drop",
    stats: ConvertStats | None = None,
) -> Iterator[dict[str, Any]]:
    """Convert records in memory: the pipeline of :func:`convert_file` without files.

    ``records`` is any iterable of dicts, such as a list, a generator, or a
    Hugging Face ``datasets.Dataset``. Yields the output rows as dicts, lazily,
    in order; records that are not dicts or that fail validation are skipped
    and counted in ``stats`` as by ``convert_file`` (``lines_read`` counts the
    records). ``on_invalid="fail"`` raises :class:`InvalidExampleError` with
    the 1-based record number.

    >>> rows = list(convert_records([{"instruction": "Hi", "output": "Hello"}]))
    >>> rows[0]["messages"][1]
    {'role': 'assistant', 'content': 'Hello'}
    """
    if on_invalid not in ("drop", "keep", "fail"):
        raise ValueError(f"on_invalid must be 'drop', 'keep', or 'fail', got {on_invalid!r}")
    notes: list[str] = []
    emitter = get_emitter(output_format, options=emit_options, notes=notes)
    pairs = wants_pairs(output_format)
    check_transforms(transform_options, pairs=pairs)
    adapter = resolve_adapter(adapter_name, adapter_options, pairs=pairs)
    transform = _transformer(transform_options)
    st = stats if stats is not None else ConvertStats()

    def numbered() -> Iterator[tuple[int, Any]]:
        for number, record in enumerate(records, 1):
            st.lines_read += 1
            yield number, record

    for number, record, grouped in oasst.group_messages(numbered()):
        st.grouped += grouped
        yield from _process_rows(record, number, adapter, emitter, st, on_invalid, notes,
                                 transform)  # fmt: skip


def _groups_rows(path: Path, encoding: str) -> bool:
    """Whether the first record of ``path`` is an OpenAssistant message row."""
    try:
        for line in iter_jsonl(path, encoding=encoding):
            return oasst.is_message(line.value)
    except OSError:
        pass
    return False


def _examples(
    examples: Iterable[TrainingExample], transform: Transform | None, st: ConvertStats
) -> Iterator[TrainingExample]:
    if transform is None:
        yield from examples
        return
    for example in examples:
        yield from transform(example, st.transforms)


def _has_reasoning(example: TrainingExample) -> bool:
    sides = [example.messages, example.rejected or []]
    return any(m.role == "assistant" and has_reasoning(m) for side in sides for m in side)


# --- parallel convert -------------------------------------------------------

_WORKER: dict[str, object] = {}


def _worker_init(spec: tuple) -> None:
    (
        adapter_name, adapter_options, output_format, emit_options, on_invalid, transforms,
        encoding,
    ) = spec  # fmt: skip
    notes: list[str] = []
    emitter = get_emitter(output_format, options=emit_options, notes=notes)
    _WORKER.update(
        adapter=resolve_adapter(adapter_name, adapter_options, pairs=wants_pairs(output_format)),
        emitter=emitter,
        notes=notes,
        on_invalid=on_invalid,
        transform=_transformer(transforms),
        encoding=encoding,
    )


def _worker_chunk(chunk: _parallel.Chunk) -> tuple[str, ConvertStats]:
    st = ConvertStats()
    out: list[str] = []
    for number, _raw, obj in _parallel.parse_chunk(chunk, _WORKER["encoding"], st):  # type: ignore[arg-type]
        out.extend(
            _process(
                obj,
                number,
                _WORKER["adapter"],  # type: ignore[arg-type]
                _WORKER["emitter"],  # type: ignore[arg-type]
                st,
                _WORKER["on_invalid"],  # type: ignore[arg-type]
                _WORKER["notes"],  # type: ignore[arg-type]
                _WORKER["transform"],  # type: ignore[arg-type]
            )
        )
    return "".join(out), st


def _run_parallel(
    input_path: Path,
    fout: TextIO,
    st: ConvertStats,
    reporter: ProgressReporter,
    encoding: str,
    workers: int,
    spec: tuple,
) -> None:
    def chunks() -> Iterator[_parallel.Chunk]:
        for chunk in _parallel.raw_chunks(input_path, encoding, st):
            reporter.update(len(chunk[0]))
            yield chunk

    results = _parallel.ordered_map(
        chunks(), _worker_chunk, workers=workers, initializer=_worker_init,
        initargs=((*spec, encoding),),
    )  # fmt: skip
    for text, part in results:
        fout.write(text)
        st.merge(part)


def convert_with_config(
    input_path: str | Path,
    output_path: str | Path,
    cfg: ConvertConfig,
    *,
    progress: bool = False,
    stats: ConvertStats | None = None,
    on_invalid: OnInvalid = "drop",
    emit_options: EmitOptions | None = None,
    workers: int = 1,
    transform_options: TransformOptions | None = None,
) -> tuple[int, int]:
    """Run :func:`convert_file` using a resolved :class:`convmerge.config.ConvertConfig`.

    ``emit_options`` defaults to ``cfg.emit_options`` (from a preset's
    ``output_options``) and ``transform_options`` to ``cfg.transform_options``.
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
        transform_options=(
            transform_options if transform_options is not None else cfg.transform_options
        ),
    )
