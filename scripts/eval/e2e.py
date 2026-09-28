"""End to end: mixed raw sources -> one recipe -> TRL SFT with assistant-only loss.

Builds four raw sources in different layouts (Alpaca, ShareGPT, OpenAI
messages with tool calls, reasoning traces), plants problems (refusals, a
loop, exact and near duplicates, benchmark copies), runs a single recipe
(convert, mix, dedupe --near, filter, decontam, tokens, split), then trains a
tiny model with TRL's ``assistant_only_loss`` on the result and checks that
every row trains on some answer tokens. Evaluation harness, not part of the
package.

    python scripts/eval/e2e.py --out DIR --model DIR --template JINJA --trainer-python PY
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

GSM = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
    "clips in May. How many clips did Natalia sell altogether in April and May?"
)
WORDS = ["alpha", "beta", "gamma", "delta", "river", "stone", "light", "cloud", "green",
         "quick", "model", "data", "train", "value", "north", "south", "east", "west"]  # fmt: skip

TRAIN = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
model_dir, template, path, out = sys.argv[1:5]
tok = AutoTokenizer.from_pretrained(model_dir)
tok.chat_template = open(template).read()
ds = load_dataset("json", data_files={"train": path})["train"]
trainer = SFTTrainer(model=AutoModelForCausalLM.from_pretrained(model_dir), processing_class=tok,
    args=SFTConfig(output_dir=out, max_length=1024, max_steps=10, per_device_train_batch_size=4,
                   assistant_only_loss=True, report_to=[], use_cpu=True, logging_steps=5,
                   save_strategy="no"),
    train_dataset=ds)
def answer_tokens(r):
    if "assistant_masks" in r:
        return sum(r["assistant_masks"])
    return sum(1 for x in r["labels"] if x != -100)
masks = [answer_tokens(r) for r in trainer.train_dataset]
res = trainer.train()
print(json.dumps({"rows": len(masks), "rows_without_answer_tokens": sum(m == 0 for m in masks),
                  "min_answer_tokens": min(masks), "loss": round(res.training_loss, 4)}))
"""


def words(rnd: random.Random, n: int) -> str:
    return " ".join(rnd.choices(WORDS, k=n))


def sources(out: Path) -> None:
    rnd = random.Random(7)
    alpaca = [{"instruction": f"Describe {words(rnd, 3)}", "input": "",
               "output": words(rnd, 30)} for _ in range(120)]  # fmt: skip
    alpaca += [{"instruction": "Tell me something", "input": "",
                "output": "I'm sorry, but I can't help with that."}] * 3  # fmt: skip
    alpaca.append({"instruction": "Loop", "input": "", "output": " ".join(["stuck again"] * 80)})
    sharegpt = [{"conversations": [{"from": "human", "value": f"Why {words(rnd, 4)}?"},
                                   {"from": "gpt", "value": words(rnd, 40)}]}
                for _ in range(120)]  # fmt: skip
    near = dict(sharegpt[0])
    near["conversations"] = [dict(t) for t in sharegpt[0]["conversations"]]
    near["conversations"][1]["value"] = near["conversations"][1]["value"] + " indeed"
    sharegpt += [sharegpt[1], near]  # an exact and a near duplicate
    sharegpt.append({"conversations": [{"from": "human", "value": GSM},
                                       {"from": "gpt", "value": "72"}]})  # fmt: skip
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps({"q": "x"})},
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "d",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }
    ]
    tool_rows = [{"messages": [{"role": "user", "content": f"Look up {words(rnd, 2)}"},
                               {"role": "assistant", "content": None, "tool_calls": [call]},
                               {"role": "tool", "tool_call_id": "c1", "content": words(rnd, 5)},
                               {"role": "assistant", "content": words(rnd, 12)}],
                  "tools": tools} for _ in range(40)]  # fmt: skip
    reasoning = [{"messages": [{"role": "user", "content": f"Solve {words(rnd, 3)}"},
                               {"role": "assistant", "content": words(rnd, 8),
                                "reasoning_content": words(rnd, 30)}]}
                 for _ in range(40)]  # fmt: skip
    for name, rows in (
        ("alpaca", alpaca),
        ("sharegpt", sharegpt),
        ("tools", tool_rows),
        ("reasoning", reasoning),
    ):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "eval.jsonl").write_text(json.dumps({"question": GSM, "answer": "72"}) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--trainer-python", required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    sources(out)
    recipe = {
        "output": "train/sft.jsonl",
        "sources": {
            name: {"path": f"{name}.jsonl", "license": "mit",
                   "convert": {"from": "auto", "format": "messages",
                               **({"tool_arguments": "object"} if name == "tools" else {})}}
            for name in ("alpaca", "sharegpt", "tools", "reasoning")
        },
        "dedupe": {"near": True},
        "filter": True,
        "decontam": {"against": ["eval.jsonl"], "check": "all"},
        "tokens": {"tokenizer": str(args.model), "max_tokens": 1024,
                   "chat_template": str(args.template)},
        "split": {"val": 0.05},
    }  # fmt: skip
    (out / "recipe.json").write_text(json.dumps(recipe, indent=1))
    run = subprocess.run([sys.executable, "-m", "convmerge", "run", str(out / "recipe.json")],
                         capture_output=True, text=True)  # fmt: skip
    print(run.stderr[-1500:])
    report = json.loads((out / "build" / "report.json").read_text())
    steps = {k: v.get("stats", {}) for k, v in report["steps"].items()}
    summary = {
        "mix": steps.get("mix", {}).get("total") or steps.get("mix"),
        "dedupe": {k: steps["dedupe"].get(k) for k in ("total", "kept", "near_duplicates")},
        "filter": {k: steps["filter"].get(k) for k in ("rows", "kept", "rules")},
        "decontam": {k: steps["decontam"].get(k) for k in ("rows", "contaminated")},
        "tokens": {k: steps["tokens"].get(k) for k in ("rows", "kept", "template_errors",
                                                       "missing_eos", "generation_tags")},
        "output_records": report["output"]["records"],
    }  # fmt: skip
    print(json.dumps(summary, indent=1))
    (out / "train.py").write_text(TRAIN)
    trained = subprocess.run([args.trainer_python, str(out / "train.py"), str(args.model),
                              str(args.template), str(out / "train" / "sft.jsonl"),
                              str(out / "o")], capture_output=True, text=True)  # fmt: skip
    print(trained.stdout.strip().splitlines()[-1:] or trained.stderr[-2000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
