"""Turn-count analysis and single/multi-turn splitting on messages-style JSONL."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def count_turns(sample: dict[str, Any]) -> int:
    """Return the number of assistant turns in a messages-style sample."""
    msgs = sample.get("messages") or []
    return sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "assistant")


def is_single_turn(sample: dict[str, Any]) -> bool:
    """True when the sample contains exactly one assistant turn."""
    return count_turns(sample) == 1


def analyze_turn_distribution(path: str | Path) -> dict[str, Any]:
    """Compute single-turn vs multi-turn counts for a JSONL file.

    Returns a dict with keys ``total``, ``single``, ``multi``, and
    ``distribution`` (mapping turn-count -> number of samples).
    """
    p = Path(path)
    single = 0
    multi = 0
    dist: Counter[int] = Counter()
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if not isinstance(sample, dict):
                continue
            turns = count_turns(sample)
            dist[turns] += 1
            if turns == 1:
                single += 1
            else:
                multi += 1
    return {
        "total": single + multi,
        "single": single,
        "multi": multi,
        "distribution": dict(sorted(dist.items())),
    }


def split_by_turns(
    src: str | Path,
    *,
    single_out: str | Path,
    multi_out: str | Path,
) -> tuple[int, int]:
    """Split a messages-style JSONL file into single-turn and multi-turn files.

    Returns ``(single_written, multi_written)``.
    """
    src_p = Path(src)
    single_p = Path(single_out)
    multi_p = Path(multi_out)
    single_p.parent.mkdir(parents=True, exist_ok=True)
    multi_p.parent.mkdir(parents=True, exist_ok=True)

    s_count = 0
    m_count = 0
    with (
        src_p.open(encoding="utf-8") as rf,
        single_p.open("w", encoding="utf-8") as sf,
        multi_p.open("w", encoding="utf-8") as mf,
    ):
        for line in rf:
            raw = line.strip()
            if not raw:
                continue
            sample = json.loads(raw)
            if not isinstance(sample, dict):
                continue
            if is_single_turn(sample):
                sf.write(raw + "\n")
                s_count += 1
            else:
                mf.write(raw + "\n")
                m_count += 1
    return s_count, m_count
