# Training with axolotl

[axolotl](https://github.com/axolotl-ai-cloud/axolotl) reads each dataset
through a `type` whose field names must match the file. A mismatch is often
silent: axolotl then trains on empty turns. `convmerge axolotl-config` looks
at a converted file and prints the matching `datasets:` block.

> Verified with axolotl 0.19 (transformers 5.16, TRL 1.9) on convmerge 0.13
> output. Each run used a tiny model, the Qwen3 chat template, and the config
> this command printed:
>
> - SFT on `messages` rows with tool calls and reasoning traces. 19 of the
>   22 tool-call rows had their calls in the trained tokens, and 80 of 80
>   reasoning traces were trained.
> - SFT on `sharegpt` rows with a `system` column.
> - DPO on `preference` pairs.

## SFT

```bash
convmerge run examples/trl/recipe_sft.yaml       # -> train/sft.jsonl + train/sft.val.jsonl
convmerge axolotl-config -i train/sft.jsonl --val train/sft.val.jsonl --config-dir . -o datasets.yaml
```

```yaml
# datasets.yaml, appended to your axolotl config
datasets:
  - path: "train/sft.jsonl"
    ds_type: "json"
    type: "chat_template"
    field_messages: "messages"
    roles_to_train: ["assistant"]
test_datasets:
  - path: "train/sft.val.jsonl"
    ds_type: "json"
    type: "chat_template"
    field_messages: "messages"
    roles_to_train: ["assistant"]
    split: "train"
```

```yaml
# the rest of the config (excerpt)
base_model: Qwen/Qwen3-0.6B
chat_template: tokenizer_default
sequence_len: 2048
```

What the command adapts to:

- **Tool calls:** axolotl reads `tools` and `tool_calls` from `messages`
  rows as they are. It parses arguments stored as JSON strings itself.
- **Reasoning:** traces in `reasoning_content` are read by default. For
  `--reasoning thinking` files (gpt-oss), the block adds
  `field_thinking: thinking` and `template_thinking_key: thinking`.
- **ShareGPT:** `--format sharegpt` files get the `from` / `value` mappings,
  the role names, and `field_system: system`. Files with LLaMA-Factory
  `function_call` / `observation` turns are refused. Convert them with
  `--format messages` instead.
- **Paths:** with `--config-dir`, paths are written relative to the
  directory that holds your axolotl config.
- **Last turn only:** files converted with `--train-turns last` carry
  `"train": false` on earlier answers, and the entry gets
  `message_field_training: train` so axolotl trains only the final answer
  of each conversation (checked with `axolotl preprocess`).

## DPO

```bash
convmerge convert -i ultrafeedback.jsonl -o dpo.jsonl --from auto --format preference
convmerge axolotl-config -i dpo.jsonl
```

```yaml
rl: dpo
datasets:
  - path: "dpo.jsonl"
    ds_type: "json"
    type: "chat_template.default"
    field_messages: "prompt"
    field_chosen: "chosen"
    field_rejected: "rejected"
```

axolotl's `chat_template.default` reads only the last message of each side.
Pairs whose answers continue past one message (multi-turn or tool-call
continuations) are refused rather than silently cut.
`--format sharegpt-preference` files work too; they get the ShareGPT role
mappings.

## Before training

Check the file with the tokenizer axolotl will use:

```bash
convmerge tokens -i train/sft.jsonl --tokenizer <base_model> --max-tokens <sequence_len>
```

axolotl drops rows longer than `sequence_len`. In the SFT check above it
dropped 47 of 200 tool-calling rows, and `tokens --max-tokens 1024` had
reported exactly those 47 as `over_limit` beforehand. `tokens -o` writes the
rows that fit. `tokens` also reports template errors and missing stop
tokens; the [troubleshooting table](troubleshooting.md) lists the fixes.
