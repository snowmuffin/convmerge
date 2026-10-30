"""Time and peak memory of convmerge commands on synthetic data.

Usage::

    python benchmarks/bench.py                 # 200k rows per source
    python benchmarks/bench.py --rows 1000000  # bigger
    python benchmarks/bench.py --only mix-total mix-all
    python benchmarks/bench.py --only dedupe-near dedupe-near-sqlite  # needs [quality]

Each scenario runs in a fresh Python process that calls the CLI in-process,
so the reported peak RSS belongs to that command alone. Results print as a
table and, with ``--json PATH``, are saved for comparison between versions
together with the conditions they were measured under (Python, platform,
CPU count, input sizes).
Run it from a checkout (``PYTHONPATH=src``) or against an installed package.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path

_RUNNER = """
import resource, sys, time
from convmerge.cli import main
t = time.perf_counter()
main(sys.argv[1:])
elapsed = time.perf_counter() - t
rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"__BENCH__ {elapsed:.3f} {rss_kb}", file=sys.stderr)
"""


def _make_source(path: Path, rows: int, seed: int) -> None:
    rng = random.Random(seed)
    with path.open("w", encoding="utf-8") as f:
        for i in range(rows):
            msgs = [{"role": "system", "content": "You are helpful."}]
            for t in range(rng.choice((1, 2, 3))):
                msgs.append({"role": "user", "content": f"question {t} about item {i} {seed}"})
                msgs.append({"role": "assistant", "content": "answer " * rng.randint(5, 40)})
            f.write(json.dumps({"id": i, "messages": msgs}) + "\n")


def _make_unique_source(path: Path, rows: int, seed: int) -> None:
    """Rows of random words: almost no near-duplicates, so every row is indexed."""
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(50_000)]
    with path.open("w", encoding="utf-8") as f:
        for i in range(rows):
            answer = " ".join(rng.choices(vocab, k=rng.randint(20, 120)))
            msgs = [{"role": "user", "content": f"question {i}"},
                    {"role": "assistant", "content": answer}]  # fmt: skip
            f.write(json.dumps({"id": i, "messages": msgs}) + "\n")


def _scenarios(d: Path, rows: int) -> dict[str, list[str]]:
    a, b, c = (str(d / f"src_{k}.jsonl") for k in "abc")
    u = str(d / "src_unique.jsonl")
    out = str(d / "out.jsonl")
    return {
        "convert": ["convert", "-i", a, "-o", out, "--from", "chat", "-f", "messages"],
        "convert-w4": [
            "convert", "-i", a, "-o", out, "--from", "chat", "-f", "messages", "--workers", "4",
        ],
        "dedupe": ["dedupe", "-i", a, "-o", out],
        "dedupe-sqlite": ["dedupe", "-i", a, "-o", out, "--seen-store", "sqlite"],
        # High-uniqueness input: the near-duplicate index grows with every row.
        "dedupe-near": ["dedupe", "--near", "-i", u, "-o", out],
        "dedupe-near-sqlite": ["dedupe", "--near", "-i", u, "-o", out, "--seen-store", "sqlite"],
        "filter": ["filter", "-i", a, "-o", out],
        "filter-w4": ["filter", "-i", a, "-o", out, "--workers", "4"],
        "mix-total": [
            "mix", "-i", f"{a}:0.5", f"{b}:0.3", f"{c}:0.2", "-o", out,
            "--total", str(rows // 10), "--no-recipe",
        ],
        "mix-all": ["mix", "-i", f"{a}:1", f"{b}:1", f"{c}:1", "-o", out, "--no-recipe"],
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=200_000, help="rows per source (default 200k)")
    ap.add_argument("--only", nargs="+", default=None, help="scenario names to run")
    ap.add_argument("--json", type=Path, default=None, help="also write results here")
    args = ap.parse_args()

    results: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory(prefix="convmerge-bench-") as tmp:
        d = Path(tmp)
        for k, seed in zip("abc", (1, 2, 3)):
            _make_source(d / f"src_{k}.jsonl", args.rows, seed)
        _make_unique_source(d / "src_unique.jsonl", args.rows, 4)
        size_mb = (d / "src_a.jsonl").stat().st_size / 1e6
        unique_mb = (d / "src_unique.jsonl").stat().st_size / 1e6
        print(f"{args.rows:,} rows per source ({size_mb:.0f} MB each; "
              f"{unique_mb:.0f} MB for the near-dedupe input)")  # fmt: skip
        for name, argv in _scenarios(d, args.rows).items():
            if args.only and name not in args.only:
                continue
            proc = subprocess.run(
                [sys.executable, "-c", _RUNNER, *argv], capture_output=True, text=True
            )
            line = next((x for x in proc.stderr.splitlines() if x.startswith("__BENCH__")), None)
            if proc.returncode or line is None:
                print(f"{name:12} FAILED: {proc.stderr.strip().splitlines()[-1:]}")
                continue
            _, secs, rss_kb = line.split()
            results[name] = {"seconds": float(secs), "peak_rss_mb": int(rss_kb) / 1024}
            print(f"{name:12} {float(secs):7.2f}s  {int(rss_kb) / 1024:8.1f} MB peak RSS")
    if args.json:
        report = {
            "conditions": {
                "rows_per_source": args.rows,
                "source_mb": round(size_mb, 1),
                "near_dedupe_source_mb": round(unique_mb, 1),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpus": os.cpu_count(),
            },
            "results": results,
        }
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
