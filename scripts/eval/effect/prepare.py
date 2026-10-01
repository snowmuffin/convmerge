"""Effect study, step 1: fetch the sources and build the training files of both pipelines.

    python scripts/eval/effect/prepare.py --out DIR --tokenizer NAME [--scale 1.0]
    python scripts/eval/effect/prepare.py --out DIR --tokenizer PATH --offline RAW_DIR

Writes DIR/a.jsonl (pipeline A, common scripts), DIR/b.jsonl (pipeline B,
convmerge recipe), DIR/eval.jsonl, and DIR/stats.json. See
docs/design/effect-study.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import naive  # noqa: E402

# (name, dataset, config, split, rows)
SOURCES = [
    ("oasst", "OpenAssistant/oasst2", None, "train", 20_000),
    ("openhermes", "teknium/OpenHermes-2.5", None, "train", 5_000),
    ("openr1", "open-r1/OpenR1-Math-220k", None, "train", 3_000),
    ("alpaca", "tatsu-lab/alpaca", None, "train", 5_000),
    ("eval", "HuggingFaceH4/no_robots", None, "test", 500),
]


def fetch(raw: Path, scale: float) -> None:
    from datasets import load_dataset

    raw.mkdir(parents=True, exist_ok=True)
    for name, repo, config, split, rows in SOURCES:
        n = rows if name == "eval" else max(1, int(rows * scale))
        ds = load_dataset(repo, config, split=split, streaming=True)
        with (raw / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in ds.take(n):
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(f"fetched {name}: {n} rows of {repo}", flush=True)


def stats(path: Path, tok) -> dict:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    per: dict[str, dict[str, int]] = {}
    for r in rows:
        s = per.setdefault(r.get("meta", {}).get("source", "?"), {"rows": 0, "tokens": 0})
        s["rows"] += 1
        text = "\n".join(
            m.get("content") or "" for m in r["messages"] if isinstance(m.get("content"), str)
        )
        s["tokens"] += len(tok(text, add_special_tokens=False)["input_ids"])
    total = sum(s["tokens"] for s in per.values()) or 1
    for s in per.values():
        s["token_share"] = round(s["tokens"] / total, 4)
    return {"rows": len(rows), "sources": per}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument(
        "--scale", type=float, default=1.0, help="fraction of the source rows (dry runs)"
    )
    p.add_argument("--offline", type=Path, help="directory with the raw files instead of the Hub")
    args = p.parse_args()
    out = args.out
    raw = out / "raw"
    if args.offline:
        shutil.copytree(args.offline, raw, dirs_exist_ok=True)
    else:
        fetch(raw, args.scale)
    shutil.copy(raw / "eval.jsonl", out / "eval.jsonl")

    counts_a = naive.build(raw, out / "a.jsonl")
    shutil.copy(HERE / "recipe.json", out / "recipe.json")
    run = subprocess.run(
        [sys.executable, "-m", "convmerge", "run", str(out / "recipe.json")],
        capture_output=True,
        text=True,
    )
    print(run.stderr[-3000:], flush=True)
    if run.returncode != 0:
        sys.exit(run.returncode)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    report_path = out / "build" / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    result = {
        "a": {**stats(out / "a.jsonl", tok), "taken": counts_a},
        "b": {**stats(out / "b.jsonl", tok), "report": report},
    }
    (out / "stats.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {k: {"rows": v["rows"], "sources": v["sources"]} for k, v in result.items()}, indent=1
        )
    )


if __name__ == "__main__":
    main()
