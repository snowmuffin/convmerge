"""Pipeline A of the effect study: the conversions dataset cards and tutorials show.

Each function is written the way a user would write it for one dataset, on
purpose without convmerge's checks. See docs/design/effect-study.md.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

PER_SOURCE = 2000


def oasst(path: Path) -> list[dict]:
    # Pair each root prompt with its rank-0 reply (the common pandas snippet).
    df = pd.read_json(path, lines=True)
    prompts = df[df.parent_id.isna()]
    answers = df[(df.role == "assistant") & (df["rank"] == 0)]
    pairs = prompts.merge(
        answers, left_on="message_id", right_on="parent_id", suffixes=("_q", "_a")
    )
    return [
        {
            "messages": [
                {"role": "user", "content": r.text_q},
                {"role": "assistant", "content": r.text_a},
            ]
        }
        for r in pairs.itertuples()
    ]


def openhermes(path: Path) -> list[dict]:
    roles = {"system": "system", "human": "user", "gpt": "assistant"}
    out = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        out.append(
            {
                "messages": [
                    {"role": roles.get(t["from"], t["from"]), "content": t["value"]}
                    for t in row["conversations"]
                ]
            }
        )
    return out


def openr1(path: Path) -> list[dict]:
    return [{"messages": json.loads(line)["messages"]} for line in path.open(encoding="utf-8")]


def alpaca(path: Path) -> list[dict]:
    out = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        prompt = row["instruction"] + (f"\n\n{row['input']}" if row.get("input") else "")
        out.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": row["output"]},
                ]
            }
        )
    return out


BUILDERS = {"oasst": oasst, "openhermes": openhermes, "openr1": openr1, "alpaca": alpaca}


def build(raw: Path, out: Path, seed: int = 0) -> dict[str, int]:
    """Up to PER_SOURCE rows of each source, concatenated and shuffled."""
    rows, counts = [], {}
    for name, fn in BUILDERS.items():
        part = fn(raw / f"{name}.jsonl")[:PER_SOURCE]
        counts[name] = len(part)
        rows += [{**r, "meta": {"source": name}} for r in part]
    random.Random(seed).shuffle(rows)
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return counts
