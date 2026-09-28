"""Robustness: run every file-reading command on random and malformed inputs.

A command passes a case when it exits 0, 1, or 2 (the documented codes) and
prints no Python traceback. Random records mix the keys convmerge looks for
with values of every JSON type, so adapters see near-misses of real layouts.
Evaluation harness, not part of the package.

    python scripts/eval/fuzz.py --out DIR [--random-files 60] [--tokenizer DIR]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

KEYS = ["messages", "conversations", "conversation", "role", "content", "from", "value",
        "instruction", "input", "output", "prompt", "chosen", "rejected", "response",
        "question", "answer", "text", "system", "tool_calls", "tools", "function",
        "arguments", "name", "reasoning_content", "thinking", "images", "type", "id"]  # fmt: skip
ROLES = ["user", "assistant", "system", "tool", "human", "gpt", "model", "", "USER", None, 3]


def value(rnd: random.Random, depth: int = 0):
    kinds = ["str", "int", "float", "bool", "null", "list", "dict", "role"]
    kind = rnd.choice(kinds if depth < 4 else kinds[:5])
    if kind == "str":
        return rnd.choice(["", "hi", "a" * rnd.randint(0, 300), "<think>x</think>y",
                           "<|im_start|>user\nq<|im_end|>", "[INST] q [/INST] a", "\x00\u202e",
                           "{\"a\": 1}", "🙂" * 5, "안녕하세요"])  # fmt: skip
    if kind == "int":
        return rnd.choice([0, -1, 2**63, 10**30])
    if kind == "float":
        return rnd.choice([0.5, 1e308, -0.0])
    if kind == "bool":
        return rnd.random() < 0.5
    if kind == "null":
        return None
    if kind == "role":
        return rnd.choice(ROLES)
    if kind == "list":
        return [value(rnd, depth + 1) for _ in range(rnd.randint(0, 4))]
    return {rnd.choice(KEYS): value(rnd, depth + 1) for _ in range(rnd.randint(0, 5))}


def random_record(rnd: random.Random) -> dict:
    rec = {rnd.choice(KEYS): value(rnd) for _ in range(rnd.randint(1, 5))}
    if rnd.random() < 0.5:  # near-miss of a real layout
        turns = [{rnd.choice(["role", "from"]): rnd.choice(ROLES),
                  rnd.choice(["content", "value"]): value(rnd, 2)}
                 for _ in range(rnd.randint(0, 5))]  # fmt: skip
        rec[rnd.choice(["messages", "conversations", "chosen"])] = turns
    return rec


def special_cases(out: Path) -> dict[str, Path]:
    cases: dict[str, bytes] = {
        "empty": b"",
        "whitespace": b"  \n\n\t\n",
        "bom_crlf": "\ufeff".encode() + b'{"instruction":"q","output":"a"}\r\n',
        "invalid_utf8": b'{"instruction":"\xff\xfe","output":"a"}\n',
        "nan_inf": b'{"instruction":"q","output":NaN,"x":Infinity}\n',
        "lone_surrogate": b'{"instruction":"\\ud800","output":"a"}\n',
        "null_bytes": b'{"instruction":"q\\u0000","output":"a"}\n\x00\x00\n',
        "json_array": b'[{"instruction":"q","output":"a"}]\n',
        "garbage": bytes(range(256)) * 20,
        "deep_nesting": b'{"messages":' + b"[" * 3000 + b"]" * 3000 + b"}\n",
        "no_trailing_newline": b'{"instruction":"q","output":"a"}',
        "huge_line": json.dumps({"instruction": "q", "output": "x" * 20_000_000}).encode(),
        "truncated": b'{"messages": [{"role": "user", "content": "q"}, {"role": "assis',
        "scalars": b'1\n"s"\nnull\ntrue\n[]\n',
        "dup_keys": b'{"output":"a","output":"b","instruction":"q"}\n',
    }
    paths = {}
    for name, data in cases.items():
        p = out / f"case_{name}.jsonl"
        p.write_bytes(data)
        paths[name] = p
    paths["directory"] = out  # a directory where a file is expected
    paths["missing"] = out / "does_not_exist.jsonl"
    return paths


def commands(src: Path, out: Path, tokenizer: Path | None) -> dict[str, list[str]]:
    o = str(out / "o.jsonl")
    cmds = {
        "convert": ["convert", "-i", str(src), "-o", o, "--from", "auto", "--format", "messages"],
        "convert-pref": ["convert", "-i", str(src), "-o", o, "--from", "auto", "--format",
                         "preference"],
        "convert-sharegpt": ["convert", "-i", str(src), "-o", o, "--from", "auto", "--format",
                             "sharegpt"],
        "validate": ["validate", "-i", str(src)],
        "inspect": ["inspect", str(src)],
        "dedupe": ["dedupe", "-i", str(src), "-o", o],
        "dedupe-near": ["dedupe", "-i", str(src), "-o", o, "--near"],
        "filter": ["filter", "-i", str(src), "-o", o, "--enable", "slop"],
        "decontam": ["decontam", "-i", str(src), "--against", str(src)],
        "split": ["split", "-i", str(src), "-o", o, "--val", "0.5"],
        "turns": ["turns", "-i", str(src)],
        "normalize": ["normalize", "-i", str(src), "-o", str(out / "n.jsonl")],
        "llamafactory-info": ["llamafactory-info", "-i", str(src), "--name", "x"],
        "axolotl-config": ["axolotl-config", "-i", str(src)],
    }  # fmt: skip
    if tokenizer is not None:
        cmds["tokens"] = ["tokens", "-i", str(src), "--tokenizer", str(tokenizer)]
    return cmds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-files", type=int, default=60)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(1234)
    inputs = special_cases(out)
    for i in range(args.random_files):
        p = out / f"random_{i}.jsonl"
        p.write_text("".join(json.dumps(random_record(rnd), ensure_ascii=rnd.random() < 0.5)
                             + "\n" for _ in range(rnd.randint(1, 40))))  # fmt: skip
        inputs[f"random_{i}"] = p
    runs = crashes = 0
    failures: list[dict] = []
    codes: dict[int, int] = {}
    for name, src in inputs.items():
        work = out / "work"
        work.mkdir(exist_ok=True)
        for cname, argv in commands(src, work, args.tokenizer).items():
            proc = subprocess.run([sys.executable, "-m", "convmerge", *argv], capture_output=True,
                                  text=True, errors="replace", timeout=300)  # fmt: skip
            runs += 1
            codes[proc.returncode] = codes.get(proc.returncode, 0) + 1
            if (
                proc.returncode not in (0, 1, 2)
                or "Traceback (most recent call last)" in proc.stderr
            ):
                crashes += 1
                failures.append({"case": name, "command": cname, "exit": proc.returncode,
                                 "stderr": proc.stderr[-700:]})  # fmt: skip
    summary = {"runs": runs, "crashes": crashes, "exit_codes": codes, "failures": failures}
    (out / "fuzz.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}))
    for f in failures[:40]:
        print(f"--- {f['case']} / {f['command']} (exit {f['exit']})\n{f['stderr'][-400:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
