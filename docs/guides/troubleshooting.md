# Troubleshooting: symptom → check → fix

Most training-data problems show up late: as a template error after the
first batch, or as a model that never stops or never reasons. Many of them
never raise at all. `convmerge tokens` renders every row with the target
model's chat template before training, and prints a `hint:` line for each
problem it finds. This table collects them.

```bash
convmerge tokens -i train.jsonl --tokenizer <model> --max-tokens <trainer max length>
```

| Symptom | `tokens` reports | Fix (`convmerge convert ...`) |
|---------|------------------|-------------------------------|
| `Conversation roles must alternate user/assistant/...` (Mistral, Gemma, Llama 2) | `template_errors` | `--merge-consecutive`; add `--system fold` when a system turn breaks the alternation |
| `System role not supported` (Gemma 2) | `template_errors` | `--system fold` (or `--system drop`) |
| `argument of type 'NoneType' is not iterable`, `can only concatenate str (not "NoneType")`, `unsupported operand ... 'NoneType'` on tool-call rows | `template_errors` | `--tool-content empty`, the default since 0.12; files converted with 0.11 or earlier have `"content": null` |
| Tool arguments show up as `"{\"city\": ...}"` in the prompt | `double_encoded_arguments` | `--tool-arguments object` |
| `can only concatenate str (not "dict") to str` on tool-call rows (DeepSeek-V3) | `template_errors` | `--tool-arguments string` |
| `'str object' has no attribute 'items'` on tool-call rows (GLM-4) | `template_errors` | `--tool-arguments object` |
| The model never stops generating | `missing_eos`, `stop_tokens` | Make the tokenizer's `eos_token` (or `generation_config.json` `eos_token_id`) the template's end-of-turn token. A classic case is a base model whose eos is `<\|endoftext\|>` trained with a ChatML template that ends turns with `<\|im_end\|>`. |
| Loss is 0 or does not move with TRL `assistant_only_loss` | `generation_tags: false` | Use a chat template with `{% generation %}` markers; see also `answer_beyond_limit` |
| Every label is -100 with Unsloth `train_on_responses_only` | `response_markers`, `markers_missing` | Pass the reported `instruction_part` / `response_part`; `markers_missing` counts rows whose tokens do not contain them |
| Rows train on nothing after truncation | `answer_beyond_limit` (with `--max-tokens`) | Filter with `tokens --max-tokens N -o fit.jsonl`, or raise the trainer's max length |
| Reasoning model forgets how to reason after fine-tuning | `reasoning_dropped_final` | `--reasoning reasoning_content` (Qwen3, DeepSeek), `--reasoning thinking` (gpt-oss), or `--reasoning inline`; see [reasoning.md](reasoning.md) |
| Multi-turn reasoning traces vanish, or history traces are trained on | `reasoning_dropped` | `--split-turns` (train every turn's trace) or `--reasoning-turns last` |
| LLaMA-Factory `KeyError` / `Unsupported cast from struct to utf8` | — | Register the file with `convmerge llamafactory-info` instead of writing `dataset_info.json` by hand |
| `ArrowInvalid: ... changed from string to object` when loading the file with `datasets` | `convmerge validate` reports `type_conflicts` | `--tool-arguments string` for tool-call arguments; keep text-only and multimodal rows in separate files |
| Only the last answer of each conversation should be trained (Nemotron chat) | — | `--train-turns last` with axolotl (`message_field_training: train`); `mask_history: true` with LLaMA-Factory |
| LLaMA-Factory drops or misreads rows with two user turns in a row | `convert` drop reason `unrepresentable_role_order` | `--merge-consecutive`, `--system fold` |
| A reasoning dataset converts with no traces | `convert` prints no "carry a reasoning trace" line | Name the key: `--adapter-kwargs '{"chat": {"reasoning_keys": [...]}}'` (turn key) or `record_reasoning_keys` (column) |

All `tokens` findings except template errors, rows over `--max-tokens`, and
double-encoded arguments are warnings. They do not change the exit status.
In a recipe, the `tokens` stage writes the same keys, `hints` included, to
`build/report.json`.
