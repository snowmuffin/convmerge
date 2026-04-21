"""Minimal end-to-end SFT dataset recipe using only convmerge primitives.

Run:

    python examples/recipes/build_sft_from_mixed_jsonl.py ./raw ./out

Expected layout under ``./raw/``:

- ``./raw/<any>.json``    (top-level array of alpaca or messages objects)
- ``./raw/<any>.jsonl``   (one alpaca / messages / sharegpt object per line)
- ``./raw/<any>.parquet`` (needs the ``convmerge[parquet]`` extra)

Produces:

- ``./out/final/train.jsonl`` / ``./out/final/test.jsonl`` — messages-shaped.
- Intermediate trees under ``./out/{jsonl,multi_turn,single_turn,final}``
  so you can inspect each stage.

This recipe does NOT upload anywhere, call any model, or run labeling. If
you need those steps, wrap ``build_sft_jsonl`` from your own script.
"""

from __future__ import annotations

import sys
from pathlib import Path

from convmerge.pipeline import build_sft_jsonl


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    raw_dir = Path(argv[1])
    out_dir = Path(argv[2])

    result = build_sft_jsonl(
        raw_dir=raw_dir,
        out_dir=out_dir,
        # Typical junk files that come out of HF dumps; tune for your dataset.
        prune_suffixes=("-res.jsonl", "config.jsonl", "_infos.jsonl"),
        train_ratio=0.98,
        seed=42,
        min_turns=1,
    )

    print("== counts ==")
    for k, v in result.counts.items():
        print(f"  {k:>22s}: {v:,}")
    print(f"train -> {result.train_path}")
    print(f"test  -> {result.test_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
