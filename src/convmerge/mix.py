"""Weighted mixing of multiple converted JSONL sources into one merged file."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


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


def mix_files(
    sources: list[MixSource],
    output_path: Path,
    *,
    total: int | None = None,
    seed: int = 42,
    oversample: bool = False,
    encoding: str = "utf-8",
) -> MixResult:
    """Sample from each source at its weight and write a merged JSONL file.

    When *total* is None all records are merged (weights are ignored for
    sampling; the result is just a seeded shuffle of the concatenation).
    When a source has fewer records than requested and *oversample* is False
    the source is clipped to its full size; set *oversample=True* to allow
    sampling with replacement.

    Weights are automatically normalized so they need not sum to 1.0.

    Returns a :class:`MixResult` with per-source statistics.
    """
    if not sources:
        raise ValueError("At least one source is required")

    total_weight = sum(s.weight for s in sources)
    if total_weight <= 0:
        raise ValueError("Weights must be positive")

    normalized = [MixSource(s.path, s.weight / total_weight) for s in sources]

    loaded: list[list[str]] = []
    for src in normalized:
        if not src.path.is_file():
            raise FileNotFoundError(f"Source not found: {src.path}")
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
    )


def write_mix_recipe(result: MixResult, *, encoding: str = "utf-8") -> Path:
    """Write a sidecar <output>.mix.json recording the exact mix parameters.

    The file can be used to audit or replay the mix run.
    """
    import datetime

    recipe = {
        "version": 1,
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
    ``total``, ``seed``, ``output``, and ``oversample``.

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

    return sources, options


def _load_valid_lines(path: Path, encoding: str) -> list[str]:
    lines = []
    with path.open(encoding=encoding) as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
            except json.JSONDecodeError:
                continue
            lines.append(stripped)
    return lines


def _allocate(sources: list[MixSource], total: int) -> list[int]:
    """Distribute *total* across sources by weight, correcting rounding error."""
    counts = [round(total * s.weight) for s in sources]
    diff = total - sum(counts)
    if diff != 0:
        idx = max(range(len(counts)), key=lambda i: counts[i])
        counts[idx] += diff
    return counts
