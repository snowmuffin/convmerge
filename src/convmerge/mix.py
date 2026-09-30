"""Weighted mixing of multiple converted JSONL sources into one merged file."""

from __future__ import annotations

import json
import logging
import random
import tempfile
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Literal

from convmerge.io import iter_jsonl, iter_raw_lines

logger = logging.getLogger(__name__)

# A clipped source moves the mix off its weights; warn past this many points.
OFF_TARGET = 0.05


@dataclass(frozen=True)
class MixSource:
    """One source entry: a JSONL file and its sampling weight."""

    path: Path
    weight: float


@dataclass
class SourceStats:
    path: Path
    weight: float
    requested: int
    available: int
    written: int
    units: float | None = None
    """Characters or tokens in the source's valid rows (``by="chars"`` /
    ``"tokens"``); ``None`` when mixing by rows."""
    reasoning: int = 0
    """Valid rows whose answers carry a reasoning trace (a ``reasoning_content``
    / ``thinking`` / ``reasoning`` field or an inline ``<think>`` block)."""
    measured: int | None = None
    """Rows whose length was measured for ``units``: every valid row, or the
    random sample of ``by_sample`` rows (``units`` is then an estimate);
    ``None`` when mixing by rows."""

    @property
    def mean_units(self) -> float | None:
        """Mean characters or tokens per row (``None`` when mixing by rows)."""
        if self.units is None or not self.available:
            return None
        return self.units / self.available


@dataclass
class MixResult:
    total_written: int
    seed: int
    output: Path
    sources: list[SourceStats] = field(default_factory=list)
    sampler: str = "v2"
    by: str = "rows"
    """What the weights measured: ``rows``, ``chars`` or ``tokens``."""
    by_sample: int | None = None
    """Rows per source measured to estimate its length (``None``: all rows)."""


Sampler = Literal["v1", "v2"]
SAMPLERS: tuple[str, ...] = ("v1", "v2")
MixUnit = Literal["rows", "chars", "tokens"]
MIX_UNITS: tuple[str, ...] = ("rows", "chars", "tokens")

# v2 shuffles through temporary bucket files holding about this many lines
# each, so peak memory is bounded by one bucket rather than the whole output.
# A constant (not tuned to the machine) keeps results reproducible anywhere.
_BUCKET_LINES = 25_000


def mix_files(
    sources: list[MixSource],
    output_path: str | Path,
    *,
    total: int | None = None,
    seed: int = 42,
    oversample: bool = False,
    encoding: str = "utf-8",
    sampler: Sampler = "v2",
    by: MixUnit = "rows",
    tokenizer: Any = None,
    by_sample: int | None = None,
) -> MixResult:
    """Sample from each source at its weight and write a merged JSONL file.

    When *total* is None all records are merged (weights are ignored for
    sampling; the result is just a seeded shuffle of the concatenation).
    When a source has fewer records than requested and *oversample* is False
    the source is clipped to its full size; set *oversample=True* to repeat
    records instead.

    Weights are automatically normalized so they need not sum to 1.0.

    ``sampler="v2"`` (default since 0.7) streams: it reads each source twice
    (count, then pick the chosen lines) and shuffles through temporary files
    next to the output, so memory stays bounded by the sample size — or by one
    shuffle bucket when merging everything — instead of the input size.
    Oversampling repeats every record ``target // available`` times and fills
    the remainder without replacement. ``sampler="v1"`` is the 0.6 in-memory
    algorithm; use it to reproduce a mix made with an earlier version (the
    same seed selects different lines under v1 and v2).

    ``by`` says what the weights measure. ``"rows"`` (default) splits *total*
    rows by weight. ``"chars"`` and ``"tokens"`` split the training text
    instead: *total* is still a row count, allocated so that each source's
    share of the characters (every string in its rows but ``role`` / ``from``
    / ``type`` labels) or tokens is its
    weight, using each source's mean row length. ``"tokens"`` needs
    *tokenizer*: a Hugging Face tokenizer name or path (``transformers`` is
    then required) or an object with ``encode``. Both need *total* and the
    v2 sampler. *by_sample* measures only that many random rows of each
    source (the same ones for the same *seed*) and scales their mean length
    by the row count: much faster for ``"tokens"`` on large sources, at the
    cost of an estimate. ``None`` (default) measures every row.

    Returns a :class:`MixResult` with per-source statistics.
    """
    if not sources:
        raise ValueError("At least one source is required")
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {list(SAMPLERS)}, got {sampler!r}")
    if by not in MIX_UNITS:
        raise ValueError(f"by must be one of {list(MIX_UNITS)}, got {by!r}")
    if by != "rows":
        if total is None:
            raise ValueError(
                f"by={by!r} needs a total: without one every record is merged and "
                "weights are not used"
            )
        if sampler != "v2":
            raise ValueError(f"by={by!r} needs the v2 sampler")
        if by == "tokens" and tokenizer is None:
            raise ValueError("by='tokens' needs a tokenizer (name, path, or object)")
    if by_sample is not None:
        if by == "rows":
            raise ValueError("by_sample needs by='chars' or 'tokens'")
        if isinstance(by_sample, bool) or not isinstance(by_sample, int) or by_sample < 1:
            raise ValueError(f"by_sample must be a positive integer, got {by_sample!r}")

    total_weight = sum(s.weight for s in sources)
    if total_weight <= 0:
        raise ValueError("Weights must be positive")

    output_path = Path(output_path)
    normalized = [MixSource(Path(s.path), s.weight / total_weight) for s in sources]
    for src in normalized:
        if not src.path.is_file():
            raise FileNotFoundError(f"Source not found: {src.path}")

    if sampler == "v2":
        measure = _measure(by, tokenizer)
        result = _mix_v2(
            normalized, output_path, total, seed, oversample, encoding, by, measure, by_sample
        )
    else:
        result = _mix_v1(normalized, output_path, total, seed, oversample, encoding)
    _warn_off_target(result)
    return result


def written_shares(result: MixResult) -> list[float]:
    """Each source's share of what was written, in the unit the weights
    measured (rows, or the characters / tokens of its sample)."""
    sizes = [s.written * (s.mean_units or 0.0) if result.by != "rows" else float(s.written)
             for s in result.sources]  # fmt: skip
    total = sum(sizes)
    return [x / total if total else 0.0 for x in sizes]


def _warn_off_target(result: MixResult) -> None:
    clipped = [s for s in result.sources if s.written < s.requested]
    if not clipped:
        return
    shares = written_shares(result)
    off = [(s, share) for s, share in zip(result.sources, shares)
           if abs(share - s.weight) > OFF_TARGET]  # fmt: skip
    if not off:
        return
    unit = "rows" if result.by == "rows" else result.by
    detail = ", ".join(f"{s.path.name} {share:.0%} (weight {s.weight:.0%})" for s, share in off)
    short = ", ".join(f"{s.path.name} ({s.written:,} of {s.requested:,} rows)" for s in clipped)
    logger.warning(
        "the mix is off its weights: %s of the %s. Too few rows in %s; oversample them "
        "(--oversample), ask for fewer rows, or change the weights",
        detail, unit, short,
    )  # fmt: skip


def _measure(by: str, tokenizer: Any) -> Any:
    """Row → its length in ``by`` units (``None`` for rows)."""
    if by == "chars":
        return _chars
    if by == "tokens":
        tok = tokenizer
        if isinstance(tok, (str, Path)):
            from convmerge.tokens import load_tokenizer

            tok = load_tokenizer(str(tok))

        def tokens(row: Any) -> int:
            return len(tok.encode("\n".join(_strings(row)), add_special_tokens=False))

        return tokens
    return None


# Keys whose strings are labels, not training text.
_LABEL_KEYS = frozenset({"role", "from", "type"})


def _strings(value: Any) -> list[str]:
    """Every string in a row except labels (``role`` / ``from`` / ``type``)."""
    out: list[str] = []
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            stack.extend(reversed([x for k, x in v.items() if k not in _LABEL_KEYS]))
        elif isinstance(v, list):
            stack.extend(reversed(v))
    return out


def _chars(row: Any) -> int:
    return sum(len(s) for s in _strings(row))


_TRACE_KEYS = ("reasoning_content", "thinking", "reasoning")
_TRACE_SET = frozenset(_TRACE_KEYS)
_TURN_KEYS = ("messages", "conversations", "chosen")


def _may_have_trace(row: Any, raw: str) -> bool:
    """A cheap test run first: ``False`` means :func:`_has_trace` is ``False`` too.

    Without a ``<`` in the line (literal or escaped) no ``<think>`` tag can be
    there, so only the turns' keys need a look, which happens in C.
    """
    if not isinstance(row, dict):
        return False
    if "<" in raw or ("\\" in raw and "\\u003" in raw):
        return True
    try:
        for key in _TURN_KEYS:
            turns = row.get(key)
            if isinstance(turns, list) and not _TRACE_SET.isdisjoint(chain.from_iterable(turns)):
                return True
    except TypeError:  # a turn that is neither a dict nor a string
        return True
    return False


def _has_trace(row: Any) -> bool:
    """Whether a converted row's answers carry a reasoning trace."""
    if not isinstance(row, dict):
        return False
    for key in _TURN_KEYS:
        turns = row.get(key)
        if not isinstance(turns, list):
            continue
        for t in turns:
            if not isinstance(t, dict):
                continue
            for k in _TRACE_KEYS:
                v = t.get(k)
                if isinstance(v, str) and v.strip():
                    return True
            text = t.get("content", t.get("value"))
            if isinstance(text, str) and "<think>" in text:
                return True
    output = row.get("output")
    return isinstance(output, str) and "<think>" in output


def _mix_v1(
    normalized: list[MixSource],
    output_path: Path,
    total: int | None,
    seed: int,
    oversample: bool,
    encoding: str,
) -> MixResult:
    loaded: list[list[str]] = []
    traces: list[int] = []
    for src in normalized:
        lines, n_traces = _load_valid_lines(src.path, encoding)
        loaded.append(lines)
        traces.append(n_traces)

    if total is None:
        targets = [len(recs) for recs in loaded]
    else:
        targets = _allocate(normalized, total)

    rng = random.Random(seed)
    sampled: list[list[str]] = []
    stats: list[SourceStats] = []

    for src, recs, target, n_traces in zip(normalized, loaded, targets, traces):
        available = len(recs)
        if available == 0:
            sampled.append([])
            stats.append(SourceStats(src.path, src.weight, target, 0, 0))
            continue
        if target <= available:
            chosen = rng.sample(recs, target)
        elif oversample:
            chosen = rng.choices(recs, k=target)
        else:
            chosen = list(recs)
        sampled.append(chosen)
        stats.append(SourceStats(src.path, src.weight, target, available, len(chosen),
                                 reasoning=n_traces))  # fmt: skip

    all_records = [line for group in sampled for line in group]
    rng.shuffle(all_records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for line in all_records:
            f.write(line + "\n")

    return MixResult(
        total_written=len(all_records),
        seed=seed,
        output=output_path,
        sources=stats,
        sampler="v1",
    )


def _mix_v2(
    normalized: list[MixSource],
    output_path: Path,
    total: int | None,
    seed: int,
    oversample: bool,
    encoding: str,
    by: str = "rows",
    measure: Any = None,
    by_sample: int | None = None,
) -> MixResult:
    # Pass 1: count valid lines and remember where the invalid ones are, so
    # pass 2 can pick lines by ordinal without parsing JSON again.
    scans = [
        _scan(src.path, encoding, measure, by_sample, random.Random(f"{seed}:{i}"))
        for i, src in enumerate(normalized)
    ]
    available = [scan.rows for scan in scans]
    if total is None:
        targets = list(available)
    elif measure is None:
        targets = _allocate(normalized, total)
    else:
        # Rows per unit of weight: a source of long rows needs fewer rows.
        per_row = [s.weight * s_.rows / s_.units if s_.units else 0.0
                   for s, s_ in zip(normalized, scans)]  # fmt: skip
        norm = sum(per_row)
        if norm <= 0:
            raise ValueError(f"the sources have no {by} to mix")
        targets = _allocate([MixSource(s.path, w / norm) for s, w in zip(normalized, per_row)],
                            total)  # fmt: skip

    rng = random.Random(seed)
    plans: list[_Plan] = []
    stats: list[SourceStats] = []
    for src, scan, target in zip(normalized, scans, targets):
        plan = _plan(rng, scan.rows, target, oversample)
        plans.append(plan)
        units = None if measure is None else scan.units
        measured = None if measure is None else scan.measured
        stats.append(SourceStats(src.path, src.weight, target, scan.rows, plan.size,
                                 units=units, reasoning=scan.reasoning,
                                 measured=measured))  # fmt: skip

    n_out = sum(p.size for p in plans)
    n_buckets = max(1, -(-n_out // _BUCKET_LINES))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".convmerge-mix-", dir=output_path.parent) as tmp:
        buckets = [Path(tmp) / f"{i}.jsonl" for i in range(n_buckets)]
        handles = [b.open("w", encoding="utf-8") for b in buckets] if n_buckets > 1 else []
        in_memory: list[str] = []
        try:
            # Pass 2: stream each source, emit chosen lines to random buckets.
            for src, plan, scan in zip(normalized, plans, scans):
                if not plan.size:
                    continue
                for i, raw in enumerate(_valid_raw_lines(src.path, encoding, scan.invalid)):
                    for _ in range(plan.copies(i)):
                        if handles:
                            handles[rng.randrange(n_buckets)].write(raw + "\n")
                        else:
                            in_memory.append(raw)
        finally:
            for h in handles:
                h.close()

        with output_path.open("w", encoding="utf-8") as out:
            if not handles:
                rng.shuffle(in_memory)
                out.writelines(x + "\n" for x in in_memory)
            for bucket in buckets if handles else []:
                with bucket.open(encoding="utf-8") as f:
                    lines = f.readlines()
                rng.shuffle(lines)
                out.writelines(lines)
                del lines

    return MixResult(total_written=n_out, seed=seed, output=output_path, sources=stats,
                     sampler="v2", by=by, by_sample=by_sample)  # fmt: skip


@dataclass
class _Plan:
    """Which line ordinals of one source to emit, and how many times each."""

    base: int = 0
    """Every line is emitted this many times..."""
    extra: frozenset[int] = frozenset()
    """...plus once more for these ordinals..."""
    skip: frozenset[int] = frozenset()
    """...except these ordinals, which are emitted ``base - 1`` times."""
    size: int = 0

    def copies(self, i: int) -> int:
        if i in self.skip:
            return self.base - 1
        return self.base + (1 if i in self.extra else 0)


def _plan(rng: random.Random, available: int, target: int, oversample: bool) -> _Plan:
    if available == 0 or target <= 0:
        return _Plan()
    if target <= available:
        # Remember whichever set is smaller: the chosen lines or the rest.
        if target * 2 <= available:
            return _Plan(extra=frozenset(rng.sample(range(available), target)), size=target)
        skipped = frozenset(rng.sample(range(available), available - target))
        return _Plan(base=1, skip=skipped, size=target)
    if not oversample:
        return _Plan(base=1, size=available)
    base, rest = divmod(target, available)
    return _Plan(base=base, extra=frozenset(rng.sample(range(available), rest)), size=target)


@dataclass
class _Scan:
    rows: int
    invalid: frozenset[int]
    units: float
    reasoning: int
    measured: int = 0


def _scan(
    path: Path,
    encoding: str,
    measure: Any = None,
    sample: int | None = None,
    rng: random.Random | None = None,
) -> _Scan:
    """Count valid JSONL lines (and their length in ``measure`` units, and those
    with a reasoning trace); remember the invalid line numbers.

    With ``sample``, only a uniform random sample of that many rows (reservoir
    sampling with ``rng``) is measured, and ``units`` is their mean length
    times the row count.
    """
    invalid: list[int] = []
    rows = units = reasoning = 0
    reservoir: list[Any] = []
    for line in iter_jsonl(
        path, encoding=encoding, on_invalid=lambda e: invalid.append(e.line_number)
    ):
        rows += 1
        if _may_have_trace(line.value, line.raw) and _has_trace(line.value):
            reasoning += 1
        if measure is None:
            continue
        if sample is None:
            units += measure(line.value)
        elif len(reservoir) < sample:
            reservoir.append(line.value)
        else:
            j = (rng or random).randrange(rows)
            if j < sample:
                reservoir[j] = line.value
    measured = rows if measure is not None else 0
    if sample is not None and reservoir:
        measured = len(reservoir)
        units = sum(measure(v) for v in reservoir) * rows / measured
    return _Scan(rows, frozenset(invalid), units, reasoning, measured)


def _valid_raw_lines(path: Path, encoding: str, invalid: frozenset[int]):
    """Yield the same lines as :func:`iter_jsonl` (as ``raw``) without parsing.

    Mirrors its blank-line and BOM handling; ``invalid`` holds the line numbers
    :func:`_scan` found unparseable.
    """
    for number, raw in iter_raw_lines(path, encoding=encoding):
        if raw and number not in invalid:
            yield raw


def write_mix_recipe(result: MixResult, *, encoding: str = "utf-8") -> Path:
    """Write a sidecar <output>.mix.json recording the exact mix parameters.

    The file can be used to audit or replay the mix run.
    """
    import datetime

    recipe = {
        "version": 1,
        "sampler": result.sampler,
        "by": result.by,
        "by_sample": result.by_sample,
        "seed": result.seed,
        "total_written": result.total_written,
        "output": str(result.output),
        "sources": [
            {
                "path": str(s.path),
                "weight": s.weight,
                "requested": s.requested,
                "available": s.available,
                "written": s.written,
                "units": s.units,
                "reasoning": s.reasoning,
                "measured": s.measured,
                "share": round(share, 4),
            }
            for s, share in zip(result.sources, written_shares(result))
        ],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    sidecar = result.output.with_suffix(".mix.json")
    sidecar.write_text(json.dumps(recipe, indent=2, ensure_ascii=False), encoding=encoding)
    return sidecar


def load_mix_config(path: Path) -> tuple[list[MixSource], dict]:
    """Parse a YAML or JSON mix config file.

    Returns ``(sources, options)`` where *options* may contain
    ``total``, ``seed``, ``output``, ``oversample``, ``sampler``, ``by``,
    ``tokenizer``, and ``by_sample``.

    YAML support requires ``pyyaml`` (``pip install 'convmerge[preset]'``).
    """
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "pyyaml is required for YAML mix configs. "
                "Install with: pip install 'convmerge[preset]'"
            ) from e
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)

    if not isinstance(raw, dict):
        raise ValueError("Mix config root must be a mapping")

    entries_raw = raw.get("sources") or []
    if not isinstance(entries_raw, list) or not entries_raw:
        raise ValueError("Mix config must have a non-empty 'sources' list")

    sources: list[MixSource] = []
    for i, item in enumerate(entries_raw):
        if not isinstance(item, dict):
            raise ValueError(f"sources[{i}] must be a mapping")
        p = item.get("path")
        w = item.get("weight")
        if not p:
            raise ValueError(f"sources[{i}].path is required")
        if w is None:
            raise ValueError(f"sources[{i}].weight is required")
        sources.append(MixSource(Path(p), float(w)))

    options: dict = {}
    if "total" in raw:
        options["total"] = int(raw["total"])
    if "seed" in raw:
        options["seed"] = int(raw["seed"])
    if "output" in raw:
        options["output"] = Path(raw["output"])
    if "oversample" in raw:
        options["oversample"] = bool(raw["oversample"])
    if "sampler" in raw:
        if raw["sampler"] not in SAMPLERS:
            raise ValueError(f"sampler must be one of {list(SAMPLERS)}, got {raw['sampler']!r}")
        options["sampler"] = raw["sampler"]
    if "by" in raw:
        if raw["by"] not in MIX_UNITS:
            raise ValueError(f"by must be one of {list(MIX_UNITS)}, got {raw['by']!r}")
        options["by"] = raw["by"]
    if "tokenizer" in raw:
        options["tokenizer"] = str(raw["tokenizer"])
    if raw.get("by_sample") is not None:
        options["by_sample"] = int(raw["by_sample"])

    return sources, options


def _load_valid_lines(path: Path, encoding: str) -> tuple[list[str], int]:
    lines: list[str] = []
    traces = 0
    for line in iter_jsonl(path, encoding=encoding):
        lines.append(line.raw)
        traces += _may_have_trace(line.value, line.raw) and _has_trace(line.value)
    return lines, traces


def _allocate(sources: list[MixSource], total: int) -> list[int]:
    """Distribute *total* across sources by weight, correcting rounding error."""
    counts = [round(total * s.weight) for s in sources]
    diff = total - sum(counts)
    if diff != 0:
        idx = max(range(len(counts)), key=lambda i: counts[i])
        counts[idx] += diff
    return counts
