"""Weighted mixing of multiple converted JSONL sources into one merged file."""

from __future__ import annotations

import json
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from convmerge.io import iter_jsonl, iter_raw_lines


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


@dataclass
class MixResult:
    total_written: int
    seed: int
    output: Path
    sources: list[SourceStats] = field(default_factory=list)
    sampler: str = "v2"


Sampler = Literal["v1", "v2"]
SAMPLERS: tuple[str, ...] = ("v1", "v2")

# v2 shuffles through temporary bucket files holding about this many lines
# each, so peak memory is bounded by one bucket rather than the whole output.
# A constant (not tuned to the machine) keeps results reproducible anywhere.
_BUCKET_LINES = 25_000


def mix_files(
    sources: list[MixSource],
    output_path: Path,
    *,
    total: int | None = None,
    seed: int = 42,
    oversample: bool = False,
    encoding: str = "utf-8",
    sampler: Sampler = "v2",
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

    Returns a :class:`MixResult` with per-source statistics.
    """
    if not sources:
        raise ValueError("At least one source is required")
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {list(SAMPLERS)}, got {sampler!r}")

    total_weight = sum(s.weight for s in sources)
    if total_weight <= 0:
        raise ValueError("Weights must be positive")

    normalized = [MixSource(s.path, s.weight / total_weight) for s in sources]
    for src in normalized:
        if not src.path.is_file():
            raise FileNotFoundError(f"Source not found: {src.path}")

    if sampler == "v2":
        return _mix_v2(normalized, output_path, total, seed, oversample, encoding)
    return _mix_v1(normalized, output_path, total, seed, oversample, encoding)


def _mix_v1(
    normalized: list[MixSource],
    output_path: Path,
    total: int | None,
    seed: int,
    oversample: bool,
    encoding: str,
) -> MixResult:
    loaded: list[list[str]] = []
    for src in normalized:
        loaded.append(_load_valid_lines(src.path, encoding))

    if total is None:
        targets = [len(recs) for recs in loaded]
    else:
        targets = _allocate(normalized, total)

    rng = random.Random(seed)
    sampled: list[list[str]] = []
    stats: list[SourceStats] = []

    for src, recs, target in zip(normalized, loaded, targets):
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
        stats.append(SourceStats(src.path, src.weight, target, available, len(chosen)))

    all_records = [line for group in sampled for line in group]
    rng.shuffle(all_records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding=encoding) as f:
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
) -> MixResult:
    # Pass 1: count valid lines and remember where the invalid ones are, so
    # pass 2 can pick lines by ordinal without parsing JSON again.
    scans = [_scan(src.path, encoding) for src in normalized]
    available = [n for n, _ in scans]
    targets = list(available) if total is None else _allocate(normalized, total)

    rng = random.Random(seed)
    plans: list[_Plan] = []
    stats: list[SourceStats] = []
    for src, avail, target in zip(normalized, available, targets):
        plan = _plan(rng, avail, target, oversample)
        plans.append(plan)
        stats.append(SourceStats(src.path, src.weight, target, avail, plan.size))

    n_out = sum(p.size for p in plans)
    n_buckets = max(1, -(-n_out // _BUCKET_LINES))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".convmerge-mix-", dir=output_path.parent) as tmp:
        buckets = [Path(tmp) / f"{i}.jsonl" for i in range(n_buckets)]
        handles = [b.open("w", encoding=encoding) for b in buckets] if n_buckets > 1 else []
        in_memory: list[str] = []
        try:
            # Pass 2: stream each source, emit chosen lines to random buckets.
            for src, plan, (_, invalid) in zip(normalized, plans, scans):
                if not plan.size:
                    continue
                for i, raw in enumerate(_valid_raw_lines(src.path, encoding, invalid)):
                    for _ in range(plan.copies(i)):
                        if handles:
                            handles[rng.randrange(n_buckets)].write(raw + "\n")
                        else:
                            in_memory.append(raw)
        finally:
            for h in handles:
                h.close()

        with output_path.open("w", encoding=encoding) as out:
            if not handles:
                rng.shuffle(in_memory)
                out.writelines(x + "\n" for x in in_memory)
            for bucket in buckets if handles else []:
                with bucket.open(encoding=encoding) as f:
                    lines = f.readlines()
                rng.shuffle(lines)
                out.writelines(lines)
                del lines

    return MixResult(
        total_written=n_out, seed=seed, output=output_path, sources=stats, sampler="v2"
    )


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


def _scan(path: Path, encoding: str) -> tuple[int, frozenset[int]]:
    """Count valid JSONL lines; return the count and invalid line numbers."""
    invalid: list[int] = []
    n = sum(
        1
        for _ in iter_jsonl(
            path, encoding=encoding, on_invalid=lambda e: invalid.append(e.line_number)
        )
    )
    return n, frozenset(invalid)


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
            }
            for s in result.sources
        ],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    sidecar = result.output.with_suffix(".mix.json")
    sidecar.write_text(json.dumps(recipe, indent=2, ensure_ascii=False), encoding=encoding)
    return sidecar


def load_mix_config(path: Path) -> tuple[list[MixSource], dict]:
    """Parse a YAML or JSON mix config file.

    Returns ``(sources, options)`` where *options* may contain
    ``total``, ``seed``, ``output``, ``oversample``, and ``sampler``.

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

    return sources, options


def _load_valid_lines(path: Path, encoding: str) -> list[str]:
    return [line.raw for line in iter_jsonl(path, encoding=encoding)]


def _allocate(sources: list[MixSource], total: int) -> list[int]:
    """Distribute *total* across sources by weight, correcting rounding error."""
    counts = [round(total * s.weight) for s in sources]
    diff = total - sum(counts)
    if diff != 0:
        idx = max(range(len(counts)), key=lambda i: counts[i])
        counts[idx] += diff
    return counts
