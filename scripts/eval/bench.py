"""Throughput and peak memory of each command on a synthetic 1M-row mix.

Generates ShareGPT rows (2-4 turns, ~500 bytes) with planted exact
duplicates, near-duplicates, and refusals, then times each command in a
fresh process and records its peak RSS. Baselines: a minimal hand-written
converter (no validation) and the common ``datasets.load_dataset("json") ->
map -> to_json`` approach. Evaluation harness, not part of the package.

    python scripts/eval/bench.py --out DIR [--rows 1000000]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

WORDS = [f"w{i}" for i in range(5000)] + ["the", "a", "of", "and", "to", "in", "is", "for"]

HAND_SCRIPT = r"""
import json, sys
ROLES = {"human": "user", "gpt": "assistant", "system": "system"}
with open(sys.argv[1]) as f, open(sys.argv[2], "w") as out:
    for line in f:
        row = json.loads(line)
        msgs = [{"role": ROLES.get(t["from"], t["from"]), "content": t["value"]}
                for t in row["conversations"]]
        out.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
"""

DATASETS_SCRIPT = r"""
import sys
from datasets import load_dataset
ROLES = {"human": "user", "gpt": "assistant", "system": "system"}
def to_messages(row):
    return {"messages": [{"role": ROLES.get(t["from"], t["from"]), "content": t["value"]}
                         for t in row["conversations"]]}
ds = load_dataset("json", data_files=sys.argv[1], split="train")
ds = ds.map(to_messages, remove_columns=ds.column_names, num_proc=int(sys.argv[3]))
ds.to_json(sys.argv[2], force_ascii=False)
"""


def generate(path: Path, n: int, seed: int = 0) -> None:
    rnd = random.Random(seed)
    base: list[str] = []
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            r = rnd.random()
            if base and r < 0.05:  # exact duplicate
                f.write(rnd.choice(base))
                continue
            turns = []
            for _ in range(rnd.randint(1, 2)):
                q = " ".join(rnd.choices(WORDS, k=rnd.randint(8, 20)))
                a = " ".join(rnd.choices(WORDS, k=rnd.randint(40, 90)))
                if r > 0.99:
                    a = "I'm sorry, but I can't help with that. " + a
                turns += [{"from": "human", "value": q}, {"from": "gpt", "value": a}]
            if base and 0.05 <= r < 0.07:  # near-duplicate of an earlier row
                src = json.loads(rnd.choice(base))
                v = src["conversations"][-1]["value"].split()
                v[rnd.randrange(len(v))] = "changed"
                src["conversations"][-1]["value"] = " ".join(v)
                turns = src["conversations"]
            line = json.dumps({"conversations": turns}) + "\n"
            if len(base) < 5000:
                base.append(line)
            f.write(line)


# Runs the command in a fresh wrapper so RUSAGE_CHILDREN covers only this command
# (and its worker processes).
WRAP = (
    "import resource, subprocess, sys, time\n"
    "s = time.perf_counter()\n"
    "p = subprocess.run(sys.argv[1:], capture_output=True, text=True)\n"
    "t = time.perf_counter() - s\n"
    "print(t, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, p.returncode)\n"
    "print(p.stderr.strip()[-200:], file=sys.stderr)\n"
)


def measure(label: str, cmd: list[str], rows: int, results: list[dict]) -> None:
    proc = subprocess.run([sys.executable, "-c", WRAP, *cmd], capture_output=True, text=True)
    secs_s, rss_s, code = proc.stdout.split()
    secs = float(secs_s)
    res = {"label": label, "seconds": round(secs, 1), "rows_per_s": round(rows / secs),
           "peak_rss_mb": round(int(rss_s) / 1024), "exit": int(code),
           "stderr": proc.stderr.strip()[-200:]}  # fmt: skip
    results.append(res)
    print(json.dumps(res), file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--near-rows", type=int, default=200_000)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    src = out / "bench.jsonl"
    if not src.exists():
        generate(src, args.rows)
    print(f"input {src.stat().st_size / 1e6:.0f} MB, {args.rows:,} rows", file=sys.stderr)
    py, cm = sys.executable, [sys.executable, "-m", "convmerge"]
    (out / "hand.py").write_text(HAND_SCRIPT)
    (out / "ds.py").write_text(DATASETS_SCRIPT)
    conv = out / "conv.jsonl"
    results: list[dict] = []
    n = args.rows
    measure("baseline: hand-written loop", [py, str(out / "hand.py"), str(src),
                                            str(out / "hand.jsonl")], n, results)  # fmt: skip
    for procs in ("1", "4"):
        cache = out / f"hf_cache_{procs}"  # a cold cache for each run
        shutil.rmtree(cache, ignore_errors=True)
        env_cmd = ["env", f"HF_DATASETS_CACHE={cache}", py, str(out / "ds.py"), str(src),
                   str(out / f"ds{procs}.jsonl"), procs]  # fmt: skip
        measure(f"baseline: datasets map ({procs} proc)", env_cmd, n, results)
        shutil.rmtree(cache, ignore_errors=True)
    measure("convert (1 worker)", [*cm, "convert", "-i", str(src), "-o", str(conv), "--from",
                                   "auto", "--format", "messages"], n, results)  # fmt: skip
    measure("convert (4 workers)", [*cm, "convert", "-i", str(src), "-o",
                                    str(out / "conv4.jsonl"), "--from", "auto", "--format",
                                    "messages", "--workers", "4"], n, results)  # fmt: skip
    measure("dedupe (exact)", [*cm, "dedupe", "-i", str(conv), "-o", str(out / "dd.jsonl")], n,
            results)  # fmt: skip
    measure("filter (defaults)", [*cm, "filter", "-i", str(conv), "-o", str(out / "f.jsonl")], n,
            results)  # fmt: skip
    ev = out / "eval.jsonl"
    rnd = random.Random(1)
    ev.write_text("".join(json.dumps({"question": " ".join(rnd.choices(WORDS, k=40))}) + "\n"
                          for _ in range(15000)))  # fmt: skip
    measure("decontam (15k eval passages)", [*cm, "decontam", "-i", str(conv), "--against",
                                             str(ev), "-o", str(out / "dc.jsonl")], n,
            results)  # fmt: skip
    measure("split (2% val)", [*cm, "split", "-i", str(conv), "-o", str(out / "tr.jsonl"),
                               "--val", "0.02"], n, results)  # fmt: skip
    measure("mix (2 sources, 500k)", [*cm, "mix", "--input", f"{conv}:0.5",
                                      f"{out / 'hand.jsonl'}:0.5",
                                      "--total", "500000", "-o", str(out / "mix.jsonl")], n,
            results)  # fmt: skip
    near_src = out / "near_in.jsonl"
    with conv.open() as f, near_src.open("w") as g:
        for i, line in enumerate(f):
            if i >= args.near_rows:
                break
            g.write(line)
    measure(f"dedupe --near ({args.near_rows:,} rows)", [*cm, "dedupe", "-i", str(near_src),
                                                        "-o", str(out / "near.jsonl"), "--near"],
            args.near_rows, results)  # fmt: skip
    (out / "bench.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
