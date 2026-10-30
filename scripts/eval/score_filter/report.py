"""Score-filter study, step 3: aggregate the runs and state the gate decision.

    python scripts/eval/score_filter/report.py --results DIR --stats stats.json [--summary PATH]

The reading rule is the effect study's, imported from its summarize.py so the
two cannot drift: a metric differs only when every seed points the same way
and the gap between the means is larger than the seed spread of both arms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "effect"))
from summarize import METRICS, verdict  # noqa: E402


def reading(r: list[float], s: list[float], better: str | None) -> str:
    # verdict() names its arms A and B; here A is R and B is S.
    v = verdict(r, s, better)
    return v.replace("B ", "S ").replace("A ", "R ")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--stats", type=Path)
    p.add_argument("--summary", type=Path)
    args = p.parse_args()
    runs: dict[str, dict[int, dict]] = {"s": {}, "r": {}}
    for f in sorted(args.results.rglob("*.json")):
        r = json.loads(f.read_text())
        if isinstance(r, dict) and r.get("label") in runs:
            runs[r["label"]][r["seed"]] = r
    seeds = sorted(set(runs["s"]) & set(runs["r"]))
    lines = [
        "## Score-filter study results",
        "",
        f"Seeds: {', '.join(map(str, seeds))}. S = best-scored rows, R = random rows.",
        "",
        "| Metric | R (mean, min–max) | S (mean, min–max) | Reading |",
        "|---|---|---|---|",
    ]
    table = {}
    fmt = lambda xs: f"{sum(xs) / len(xs):.3f} ({min(xs):.3f}–{max(xs):.3f})"  # noqa: E731
    for key, name, better in [*METRICS, ("seconds_per_step", "Seconds per step", None)]:
        r = [runs["r"][s].get(key) for s in seeds]
        s = [runs["s"][s].get(key) for s in seeds]
        if not r or None in r or None in s:
            continue
        v = reading(r, s, better)
        table[key] = {"r": r, "s": s, "verdict": v}
        lines.append(f"| {name} | {fmt(r)} | {fmt(s)} | {v} |")
    gate = table.get("eval_loss", {}).get("verdict")
    decision = "pass" if gate == "S better" else "fail"
    lines += ["", f"**Gate A: {decision}** (eval loss: {gate or 'no runs'})."]
    if len(seeds) < 5:
        lines.append(f"Only {len(seeds)} seed(s) finished; the design fixes five.")

    if args.stats and args.stats.exists():
        st = json.loads(args.stats.read_text())
        lines += [
            "",
            "| Training data | Pool | R | S |",
            "|---|---|---|---|",
        ]
        for key, name in (
            ("rows", "Rows"),
            ("tokens", "Tokens (training template)"),
            ("mean_response_tokens", "Mean response tokens"),
            ("distinct_prompts", "Distinct prompts"),
            ("helpfulness", "Helpfulness 0–4"),
            ("correctness", "Correctness 0–4"),
        ):
            cells = [st[arm].get(key) for arm in ("pool", "r", "s")]
            shown = [
                " / ".join(str(c.get(str(i), c.get(i, 0))) for i in range(5))
                if isinstance(c, dict)
                else str(c)
                for c in cells
            ]
            lines.append(f"| {name} | {' | '.join(shown)} |")
        cut = st["s"].get("cut", {})
        lines += [
            "",
            f"S ends at {', '.join(f'{k} {v}' for k, v in cut.items())}. "
            f"Rows in both arms: {st['overlap']['rows']}; "
            f"prompts in both: {st['overlap']['prompts']}.",
        ]
    text = "\n".join(lines) + "\n"
    print(text)
    (args.results / "summary.json").write_text(
        json.dumps({"seeds": seeds, "gate": decision, "metrics": table}, indent=2)
    )
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
