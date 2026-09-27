# Training with axolotl

[axolotl](https://github.com/axolotl-ai-cloud/axolotl) reads OpenAI-style
`messages` JSONL through its `chat_template` dataset type, so convmerge's
`--format messages` output needs no further conversion.

> Unlike the [TRL](trl.md) and [LLaMA-Factory](llamafactory.md) guides, this
> configuration has **not** been run by convmerge's own checks (axolotl could
> not be installed in the test environment). The keys below follow axolotl's
> documentation; check them against the axolotl version you use.

```bash
convmerge run examples/trl/recipe_sft.yaml        # same data as the TRL guide
```

```yaml
# axolotl config (excerpt)
base_model: Qwen/Qwen2.5-0.5B-Instruct
chat_template: tokenizer_default
datasets:
  - path: train/sft.jsonl
    ds_type: json
    type: chat_template
    field_messages: messages
    roles_to_train: ["assistant"]
test_datasets:
  - path: train/sft.val.jsonl
    ds_type: json
    split: train
    type: chat_template
    field_messages: messages
sequence_len: 2048
```

Run `convmerge tokens -i train/sft.jsonl --tokenizer <base_model> --max-tokens
<sequence_len>` first: rows the chat template rejects or that exceed
`sequence_len` are reported before axolotl tokenizes the dataset, and
`--tool-arguments object` applies here as in the [TRL guide](trl.md#tool-calls).
