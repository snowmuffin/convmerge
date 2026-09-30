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

### Images (vision-language models)

TRL trains vision-language models on `messages` whose content is a list of
parts, with a bare `{"type": "image"}` part where each image goes, and an
`images` column holding the images in the same order. `--media placeholders`
writes exactly that from any image layout convmerge reads (LLaVA
`image` + `<image>` tokens, LLaMA-Factory `images`, OpenAI `image_url` parts):

```bash
convmerge convert -i llava.jsonl -o train/vlm.jsonl --from auto --media placeholders
```

```json
{"messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is this square?"}]},
              {"role": "assistant", "content": [{"type": "text", "text": "It is red."}]}],
 "images": ["imgs/0.png"]}
```

Every row gets an `images` column (empty for text-only rows) and list-typed
content, so the file loads as one Arrow schema. The images stay references
(paths or URLs, as the source gave them); turn them into images when loading:

```python
import datasets
from trl import SFTConfig, SFTTrainer

ds = datasets.load_dataset("json", data_files="train/vlm.jsonl", split="train")
ds = ds.cast_column("images", datasets.Sequence(datasets.Image()))  # paths / URLs -> PIL
trainer = SFTTrainer(model=model, args=SFTConfig(max_length=None, ...),
                     train_dataset=ds, processing_class=processor)
```

Relative paths resolve against the working directory, so run from the
dataset's folder or write absolute paths. `convmerge validate` drops rows
whose placeholders and `images` do not match in number.

> Verified with TRL 1.14, transformers 5.17 and `datasets` 5.0: a small
> Qwen2-VL trained with `SFTTrainer` on 29 converted rows (single-image,
> two-image, and text-only rows), with every row's image blocks matching its
> `images` in TRL's collator. Other TRL versions have changed this layout
> before; check the version you install.

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
and each template error with line numbers. It also warns (with the `convert`
flag that fixes it) about answers that start beyond `--max-tokens`, answers
not followed by a stop token, templates without `{% generation %}` markers
(needed by `assistant_only_loss`), and reasoning traces the template drops —
see [troubleshooting.md](troubleshooting.md).

## Reasoning models

Qwen3 templates read `reasoning_content` (or inline `<think>`), gpt-oss
templates read `thinking`; pick one with `--reasoning`, and use
`--split-turns` so every turn of a multi-turn conversation trains its trace:
[reasoning.md](reasoning.md).
