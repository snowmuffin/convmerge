# Training with LLaMA-Factory

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) reads datasets
registered in a `dataset_info.json`. convmerge writes its ShareGPT layout and
the registration entry. Files for this guide:
[`examples/llamafactory/`](../../examples/llamafactory/).

> Verified end to end with LLaMA-Factory 0.9.5 (`stage: sft` including tool
> calls, and `stage: dpo`) on real data converted by convmerge; the tool-call
> example rendered `<tool_call>` / `<tool_response>` turns with labels on the
> assistant turns only.

## SFT (with tool calls)

```bash
pip install "convmerge[all]"
convmerge run examples/llamafactory/recipe.yaml       # -> data/sft.jsonl + data/sft.val.jsonl
convmerge llamafactory-info -i data/sft.jsonl     --name my_sft     --info data/dataset_info.json
convmerge llamafactory-info -i data/sft.val.jsonl --name my_sft_val --info data/dataset_info.json
llamafactory-cli train examples/llamafactory/sft.yaml
```

`--format sharegpt` writes LLaMA-Factory's ShareGPT rows: `conversations`
with `human` / `gpt` / `function_call` / `observation` turns, the system
prompt in a `system` column, `tools` as a JSON string, media as `<image>`
tokens plus an `images` column. LLaMA-Factory requires strictly alternating
turns; conversations that cannot be written that way are dropped as
`unrepresentable_role_order` and counted in the report.

`llamafactory-info` inspects the file and adds the matching entry
(`formatting`, `columns`, `tags`, `ranking`) to `dataset_info.json`,
creating it if needed and keeping other entries. `file_name` is recorded
relative to the `dataset_info.json` directory, which is your `dataset_dir`.

To train on the last answer of each conversation only (datasets such as
Nemotron chat, whose earlier answers were not selected), set
`mask_history: true` in the training config; the `sharegpt` format has no
per-turn flag, so `--train-turns last` is for `--format messages` (axolotl).

## DPO (ranking data)

```bash
convmerge fetch hf://HuggingFaceH4/ultrafeedback_binarized --split train_prefs --max-rows 20000 -o raw
convmerge convert -i raw/HuggingFaceH4_ultrafeedback_binarized.jsonl -o data/dpo.jsonl \
  --from auto --format sharegpt-preference
convmerge llamafactory-info -i data/dpo.jsonl --name my_dpo --info data/dataset_info.json
llamafactory-cli train examples/llamafactory/dpo.yaml
```

`--format sharegpt-preference` writes LLaMA-Factory's ranking rows: the
prompt as `conversations`, and `chosen` / `rejected` as single `gpt` turns.
Pairs whose answers are multi-turn or tool calls cannot be written that way
(`unrepresentable_pair_continuation`); use `--format preference` with TRL for
those.

## Alpaca and OpenAI-style files

`llamafactory-info` also registers `--format alpaca` files and plain
`messages` files (OpenAI role tags). Files with OpenAI `tool_calls` are
refused — LLaMA-Factory cannot read them — with a hint to use
`--format sharegpt`, and so are files with `reasoning_content` / `thinking`
fields (LLaMA-Factory reads reasoning inline only; `--format sharegpt` and
`--reasoning inline` write it that way).

## Reasoning data

`--format sharegpt` writes traces inline as `<think>...</think>`, which the
`qwen3` / `deepseekr1` templates read. LLaMA-Factory trains every assistant
turn, including traces in the history that Qwen3 never shows at inference;
convert multi-turn data with `--reasoning-turns last` (LLaMA-Factory then
writes empty `<think>` blocks for earlier turns) or `--split-turns`. Details:
[reasoning.md](reasoning.md).
