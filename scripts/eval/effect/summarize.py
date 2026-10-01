"""Effect study, step 3: aggregate the runs with the reading rule fixed in the design.

    python scripts/eval/effect/summarize.py --results DIR --stats stats.json [--summary PATH]

A metric shows a difference only when every seed points the same way and the
gap between the means is larger than the seed spread of both pipelines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = [
    ("eval_loss", "Eval loss (assistant tokens)", "lower"),
    ("stop_rate", "Stops on its own", "higher"),
    ("role_leak", "Starts another turn", "lower"),
    ("refusal", "Refusal phrases", "lower"),
    ("repetition", "Repeated 4-gram (x4)", "lower"),
    ("mean_words", "Mean words per answer", None),
    ("train_loss", "Train loss", None),
]


def verdict(a: list[float], b: list[float], better: str | None) -> str:
    pairs = list(zip(sorted(a), sorted(b)))
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    spread = max(max(a) - min(a), max(b) - min(b))
    by_seed = [y - x for x, y in zip(a, b)]
    same_way = all(d > 0 for d in by_seed) or all(d < 0 for d in by_seed)
    if len(pairs) < 2:
        return "needs two or more seeds"
    if not same_way or abs(mean_b - mean_a) <= spread:
        return "no clear difference"
    if better is None:
        return "B higher" if mean_b > mean_a else "B lower"
    b_better = (mean_b < mean_a) if better == "lower" else (mean_b > mean_a)
    return "B better" if b_better else "A better"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--stats", type=Path)
    p.add_argument("--summary", type=Path)
    args = p.parse_args()
    runs: dict[str, dict[int, dict]] = {"a": {}, "b": {}}
    for f in sorted(args.results.rglob("*.json")):
        r = json.loads(f.read_text())
        if r.get("label") in runs:
            runs[r["label"]][r["seed"]] = r
    seeds = sorted(set(runs["a"]) & set(runs["b"]))
    lines = [
        "## Effect study results",
        "",
        f"Seeds: {', '.join(map(str, seeds))}. A = common scripts, B = convmerge recipe.",
        "",
        "| Metric | A (mean, min–max) | B (mean, min–max) | Reading |",
        "|---|---|---|---|",
    ]
    table = {}
    for key, name, better in METRICS:
        a = [runs["a"][s][key] for s in seeds]
        b = [runs["b"][s][key] for s in seeds]
        if not a:
            continue
        v = verdict(a, b, better)
        table[key] = {"a": a, "b": b, "verdict": v}
        fmt = lambda xs: f"{sum(xs) / len(xs):.3f} ({min(xs):.3f}–{max(xs):.3f})"  # noqa: E731
        lines.append(f"| {name} | {fmt(a)} | {fmt(b)} | {v} |")
    if args.stats and args.stats.exists():
        st = json.loads(args.stats.read_text())
        lines += [
            "",
            "| Training data | A rows | A token share | B rows | B token share |",
            "|---|---|---|---|---|",
        ]
        names = sorted(set(st["a"]["sources"]) | set(st["b"]["sources"]))
        for n in names:
            sa, sb = st["a"]["sources"].get(n, {}), st["b"]["sources"].get(n, {})
            lines.append(
                f"| {n} | {sa.get('rows', 0)} | {sa.get('token_share', 0):.1%} | "
                f"{sb.get('rows', 0)} | {sb.get('token_share', 0):.1%} |"
            )
    text = "\n".join(lines) + "\n"
    print(text)
    (args.results / "summary.json").write_text(
        json.dumps({"seeds": seeds, "metrics": table}, indent=2)
    )
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
