"""Effect study, step 2: train one pipeline's data with one seed, then measure.

    python scripts/eval/effect/train_eval.py --data a.jsonl --eval eval.jsonl \
        --model HuggingFaceTB/SmolLM2-135M --tokenizer HuggingFaceTB/SmolLM2-135M-Instruct \
        --seed 0 --steps 400 --out results.json

Training settings and metrics are fixed in docs/design/effect-study.md.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

END = "<|im_end|>"
LEAK = re.compile(r"<\|im_start\|>|^(user|human|assistant):|^### instruction", re.I | re.M)
REFUSAL = ("as an ai", "language model", "i'm sorry, but", "i cannot", "i can't assist")


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.open(encoding="utf-8"):
        msgs = [
            {"role": m["role"], "content": m.get("content") or ""}
            for m in json.loads(line)["messages"]
            if isinstance(m.get("content") or "", str)
        ]
        rows.append({"messages": msgs})
    return rows


def eval_loss(model, tok, convs: list[list[dict]], max_len: int) -> float:
    """Mean NLL of the assistant turns' tokens (their text and the end token)."""
    nll = n = 0.0
    model.eval()
    for msgs in convs:
        text = tok.apply_chat_template(msgs, tokenize=False)
        spans = []
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            start = len(
                tok.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True)
            )
            end = len(tok.apply_chat_template(msgs[: i + 1], tokenize=False))
            spans.append((start, end))
        enc = tok(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )
        ids = torch.tensor([enc["input_ids"]])
        keep = [any(s <= a < e for s, e in spans) for a, _ in enc["offset_mapping"]]
        if sum(keep[1:]) == 0:
            continue
        with torch.no_grad():
            logits = model(ids).logits[0, :-1]
        target = ids[0, 1:]
        lp = torch.log_softmax(logits.float(), dim=-1).gather(1, target[:, None])[:, 0]
        mask = torch.tensor(keep[1:])
        nll += float(-lp[mask].sum())
        n += int(mask.sum())
    return nll / n


def generate(model, tok, convs: list[list[dict]], n: int, max_new: int) -> list[tuple[str, bool]]:
    prompts = []
    for msgs in convs[:n]:
        first = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        prompts.append(
            tok.apply_chat_template(msgs[:first], tokenize=False, add_generation_prompt=True)
        )
    end_id = tok.convert_tokens_to_ids(END)
    tok.padding_side = "left"
    out: list[tuple[str, bool]] = []
    for i in range(0, len(prompts), 10):
        batch = tok(
            prompts[i : i + 10],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            truncation=True,
            max_length=768,
        )
        with torch.no_grad():
            gen = model.generate(
                **batch,
                max_new_tokens=max_new,
                do_sample=False,
                eos_token_id=end_id,
                pad_token_id=tok.pad_token_id,
            )
        for row in gen[:, batch["input_ids"].shape[1] :]:
            ids = row.tolist()
            stopped = end_id in ids
            if stopped:
                ids = ids[: ids.index(end_id)]
            ids = [t for t in ids if t != tok.pad_token_id]
            out.append((tok.decode(ids, skip_special_tokens=False), stopped))
    return out


def repeated_4gram(text: str) -> bool:
    words = text.lower().split()
    grams = Counter(tuple(words[i : i + 4]) for i in range(len(words) - 3))
    return any(c >= 4 for c in grams.values())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--gen", type=int, default=100, help="evaluation prompts to generate for")
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--label", default="")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok.eos_token = END
    if tok.pad_token is None or tok.pad_token == tok.eos_token:
        tok.pad_token = tok.unk_token or "<|endoftext|>"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    if model.get_input_embeddings().weight.shape[0] < len(tok):
        model.resize_token_embeddings(len(tok))
    model.config.eos_token_id = tok.convert_tokens_to_ids(END)

    train = Dataset.from_list(load_rows(args.data))
    config = SFTConfig(
        output_dir=str(args.out.parent / "trainer"),
        max_steps=args.steps,
        per_device_train_batch_size=8,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        max_length=1024,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        bf16=False,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=train, processing_class=tok)
    result = trainer.train()

    convs = [r["messages"] for r in load_rows(args.eval)]
    loss = eval_loss(model, tok, convs, 1024)
    gens = generate(model, tok, convs, args.gen, args.max_new)
    texts = [t for t, _ in gens]
    metrics = {
        "label": args.label,
        "seed": args.seed,
        "train_rows": len(train),
        "train_loss": round(result.training_loss, 4),
        "eval_loss": round(loss, 4),
        "eval_ppl": round(math.exp(loss), 2),
        "stop_rate": round(sum(s for _, s in gens) / len(gens), 4),
        "role_leak": round(sum(bool(LEAK.search(t)) for t in texts) / len(texts), 4),
        "refusal": round(sum(any(k in t.lower() for k in REFUSAL) for t in texts) / len(texts), 4),
        "repetition": round(sum(repeated_4gram(t) for t in texts) / len(texts), 4),
        "mean_words": round(sum(len(t.split()) for t in texts) / len(texts), 1),
        "samples": texts[:5],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in metrics.items() if k != "samples"}))


if __name__ == "__main__":
    main()
