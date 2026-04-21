"""Random-sampling utilities for JSONL files.

Two flavours are provided:

- :func:`sample_random` loads every non-blank line into memory and then calls
  :func:`random.sample`. Simple, deterministic, best when the file fits in RAM.
- :func:`reservoir_sample` streams the file once, keeping at most ``k`` lines
  resident. Use this when the source file is larger than available memory.
"""

from __future__ import annotations

import random
from pathlib import Path


def sample_random(
    src: str | Path,
    dst: str | Path,
    *,
    k: int,
    seed: int = 42,
) -> tuple[int, int]:
    """Write ``min(k, n_nonempty)`` random lines from ``src`` to ``dst``.

    Order of lines in ``dst`` follows :func:`random.sample` (pseudo-random).
    Returns ``(written, total_nonempty)``.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)

    with src_p.open(encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    total = len(lines)
    take = min(k, total)

    rng = random.Random(seed)
    picked = rng.sample(lines, take) if take else []
    with dst_p.open("w", encoding="utf-8") as out:
        for ln in picked:
            out.write(ln if ln.endswith("\n") else ln + "\n")
    return take, total


def reservoir_sample(
    src: str | Path,
    dst: str | Path,
    *,
    k: int,
    seed: int = 42,
) -> tuple[int, int]:
    """Streaming reservoir sampler (Algorithm R).

    Suitable for inputs that do not fit in memory: at most ``k`` lines are
    held resident at any time. Each line of the input is seen exactly once.

    Returns ``(written, total_nonempty)``.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    with src_p.open(encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            if seen < k:
                reservoir.append(raw)
            else:
                j = rng.randrange(seen + 1)
                if j < k:
                    reservoir[j] = raw
            seen += 1

    with dst_p.open("w", encoding="utf-8") as out:
        for ln in reservoir:
            out.write(ln if ln.endswith("\n") else ln + "\n")
    return len(reservoir), seen
