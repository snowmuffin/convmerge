# Training with TRL

[TRL](https://huggingface.co/docs/trl) reads convmerge's `messages` rows for
SFT and its `preference` rows for DPO directly: no conversion code, no custom
collator. Files for this guide: [`examples/trl/`](../../examples/trl/).

> Verified end to end with TRL 1.14 and transformers 5.17 (`SFTTrainer` and
> `DPOTrainer`, including rows with tools and tool calls) on real data
> converted by convmerge; see the 0.11 changelog.

## SFT

```bash
pip install "convmerge[all]" trl
convmerge run examples/trl/recipe_sft.yaml      # -> train/sft.jsonl + train/sft.val.jsonl
python examples/trl/sft.py
```

The recipe ([recipe_sft.yaml](../../examples/trl/recipe_sft.yaml)) fetches
UltraChat and Glaive tool-calling data, converts both to `messages`, mixes
80/20, deduplicates, then:

- **`tokens`** renders every row with the model's chat template and drops
  rows the template rejects or that exceed `max_tokens` — so nothing fails or
  gets silently truncated once training has started;
- **`split`** holds out 2% for validation by content hash (duplicates never
  straddle train and validation).

`load_dataset("json", ...)` gives `messages` (and `tools` where present);
`SFTTrainer` applies the chat template itself.

### Tool calls

Hugging Face chat templates (Qwen, Llama 3.1, Mistral, ...) apply `tojson` to
tool-call arguments, so they expect **objects**. `--format messages` writes
arguments as JSON strings by default (the OpenAI convention); for TRL set
`tool_arguments: object` in the recipe (`--tool-arguments object` on the CLI),
as the example does. `convmerge tokens` catches the mismatch: it reports
`double_encoded_arguments` and warns when a file's string arguments would be
encoded twice by the template. Recent `datasets` versions keep the per-row
argument objects intact (checked with `datasets` 5.0).

## DPO

```bash
convmerge run examples/trl/recipe_dpo.yaml      # -> train/dpo.jsonl + train/dpo.val.jsonl
python examples/trl/dpo.py
```

`--format preference` writes TRL's conversational preference format —
`prompt`, `chosen`, `rejected` as message lists — from UltraFeedback,
HH-RLHF, Orca DPO pairs, LLaMA-Factory ranking data, Chatbot Arena, and more
([docs/format.md](../format.md#preference)). The `tokens` stage counts each
pair as its longer side. The same files work for ORPO and CPO trainers.

## Checking a file before a long run

```bash
convmerge tokens -i train/sft.jsonl --tokenizer Qwen/Qwen2.5-0.5B-Instruct --max-tokens 2048
convmerge validate -i train/dpo.jsonl
```

`tokens` exits 1 when any row fails the template, is too long, or has
double-encoded tool arguments; the JSON report lists the length percentiles
and each template error with line numbers.
