"""Score-filter study, step 1: build the pool and both arms.

    python scripts/eval/score_filter/prepare.py --out DIR --tokenizer NAME [--budget 1638400]
    python scripts/eval/score_filter/prepare.py --out DIR --tokenizer PATH --offline RAW_DIR

Writes DIR/s.jsonl (best-scored rows first), DIR/r.jsonl (the same pool
shuffled), DIR/eval.jsonl and DIR/stats.json. Each arm holds at least the
token budget plus 5%, counted under the training chat template. See
docs/design/score-filter-study.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

POOL = ("nvidia/HelpSteer2", "train")
EVAL = ("HuggingFaceH4/no_robots", "test", 500)
SCORES = ("helpfulness", "correctness")
MAX_TOKENS = 512
MARGIN = 1.05
SHUFFLE_SEED = 0  # arm R's order; independent of the training seeds


def fetch(raw: Path, scale: float) -> None:
    from datasets import load_dataset

    raw.mkdir(parents=True, exist_ok=True)
    pool = load_dataset(POOL[0], split=POOL[1])
    pool = pool.select(range(max(1, int(len(pool) * scale))))
    pool.to_json(raw / "pool.jsonl", lines=True, force_ascii=False)
    print(f"fetched pool: {len(pool)} rows of {POOL[0]}", flush=True)
    ev = load_dataset(EVAL[0], split=EVAL[1], streaming=True)
    with (raw / "eval.jsonl").open("w", encoding="utf-8") as f:
        for row in ev.take(EVAL[2]):
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"fetched eval: {EVAL[2]} rows of {EVAL[0]}", flush=True)


def convmerge(*argv: str) -> dict:
    """Run one convmerge command; return the JSON report it prints, if any."""
    run = subprocess.run([sys.executable, "-m", "convmerge", *argv], capture_output=True, text=True)
    print(run.stderr[-2000:], flush=True)
    if run.returncode != 0:
        sys.exit(run.returncode)
    return json.loads(run.stdout) if run.stdout.strip().startswith("{") else {}


def messages(row: dict) -> list[dict]:
    # What train_eval.py trains on: role and string content only.
    return [
        {"role": m["role"], "content": m.get("content") or ""}
        for m in row["messages"]
        if isinstance(m.get("content") or "", str)
    ]


def n_tokens(tok, msgs: list[dict]) -> int:
    ids = tok.apply_chat_template(msgs, tokenize=True)
    if hasattr(ids, "keys"):  # a BatchEncoding in some transformers releases
        ids = ids["input_ids"]
    return len(ids)


def take(order: list[dict], least: int, pool_tokens: int) -> list[dict]:
    picked, total = [], 0
    for r in order:
        if total >= least:
            break
        picked.append(r)
        total += r["tokens"]
    if total < least:
        sys.exit(f"the pool holds {pool_tokens} tokens, fewer than the {least} an arm needs")
    return picked


def describe(arm: list[dict]) -> dict:
    return {
        "rows": len(arm),
        "tokens": sum(r["tokens"] for r in arm),
        "mean_response_tokens": round(sum(r["response_tokens"] for r in arm) / len(arm), 1),
        "distinct_prompts": len({r["prompt"] for r in arm}),
        **{k: dict(sorted(Counter(r["scores"][k] for r in arm).items())) for k in SCORES},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument(
        "--budget", type=int, default=400 * 8 * MAX_TOKENS,
        help="training tokens per run: steps x 8 sequences x 512 (default: 400 steps)",
    )  # fmt: skip
    p.add_argument("--scale", type=float, default=1.0, help="fraction of the pool (dry runs)")
    p.add_argument("--offline", type=Path, help="directory with pool.jsonl and eval.jsonl")
    args = p.parse_args()
    out, raw, work = args.out, args.out / "raw", args.out / "work"
    work.mkdir(parents=True, exist_ok=True)
    if args.offline:
        shutil.copytree(args.offline, raw, dirs_exist_ok=True)
    else:
        fetch(raw, args.scale)
    shutil.copy(raw / "eval.jsonl", out / "eval.jsonl")

    # Preparation, the same for both arms.
    convmerge("convert", "-i", str(raw / "pool.jsonl"), "-o", str(work / "pool.jsonl"),
              "--from", "auto", "-f", "messages", "--keep-fields", ",".join(SCORES),
              "--report", str(work / "convert.json"))  # fmt: skip
    fit = convmerge("tokens", "-i", str(work / "pool.jsonl"), "--tokenizer", args.tokenizer,
                    "--max-tokens", str(MAX_TOKENS), "-o", str(work / "fit.jsonl"),
                    "--rejects", str(work / "long.jsonl"))  # fmt: skip
    clean = convmerge("decontam", "-i", str(work / "fit.jsonl"),
                      "--against", str(out / "eval.jsonl"), "-o", str(work / "clean.jsonl"),
                      "--rejects", str(work / "contaminated.jsonl"))  # fmt: skip

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pool = []
    for line in (work / "clean.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        msgs = messages(row)
        answers = [m["content"] for m in msgs if m["role"] == "assistant"]
        pool.append({
            "line": line,
            "scores": {k: row["meta"][k] for k in SCORES},
            "tokens": n_tokens(tok, msgs),
            "response_tokens": sum(len(tok(a, add_special_tokens=False)["input_ids"])
                                   for a in answers),
            "prompt": next((m["content"] for m in msgs if m["role"] == "user"), ""),
            "hash": hashlib.sha256(line.encode("utf-8")).hexdigest(),
        })  # fmt: skip
    pool_tokens = sum(r["tokens"] for r in pool)
    least = math.ceil(args.budget * MARGIN)

    by_score = sorted(pool, key=lambda r: (*(-r["scores"][k] for k in SCORES), r["hash"]))
    shuffled = list(pool)
    random.Random(SHUFFLE_SEED).shuffle(shuffled)
    arms = {"s": take(by_score, least, pool_tokens), "r": take(shuffled, least, pool_tokens)}
    for name, arm in arms.items():
        (out / f"{name}.jsonl").write_text("".join(r["line"] + "\n" for r in arm), encoding="utf-8")

    lines_s, lines_r = {r["line"] for r in arms["s"]}, {r["line"] for r in arms["r"]}
    prompts_s, prompts_r = {r["prompt"] for r in arms["s"]}, {r["prompt"] for r in arms["r"]}
    stats = {
        "budget_tokens": args.budget,
        "arm_least_tokens": least,
        "preparation": {
            "convert": json.loads((work / "convert.json").read_text()),
            "tokens": fit,
            "decontam": clean,
        },
        "pool": describe(pool),
        "s": {**describe(arms["s"]), "cut": arms["s"][-1]["scores"]},
        "r": describe(arms["r"]),
        "overlap": {"rows": len(lines_s & lines_r), "prompts": len(prompts_s & prompts_r)},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: stats[k] for k in ("pool", "s", "r", "overlap")}, indent=1))


if __name__ == "__main__":
    main()
