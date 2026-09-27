# Reasoning data (`<think>`, `reasoning_content`, `thinking`)

Reasoning datasets store each answer's chain of thought in one of three
places, and every model family's chat template reads exactly one of them:

| Stored as | Datasets | Chat templates that render it |
|-----------|----------|-------------------------------|
| inline `<think>...</think>` before the answer | OpenR1-Math, OpenThoughts3, AM-Thinking, Llama-Nemotron, most ShareGPT sets | Qwen3 and DeepSeek-R1 (they split it off themselves), LLaMA-Factory's `qwen3` / `deepseekr1` templates |
| `reasoning_content` on the turn | DeepSeek API / vLLM exports | Qwen3 |
| `thinking` on the turn | gpt-oss data (HuggingFaceH4/Multilingual-Thinking) | gpt-oss |
| a column of its own | s1K (`deepseek_thinking_trajectory`) | — (map it with `record_reasoning_keys`) |

A trace in the wrong place is not an error anywhere. The template renders
the answer without it, and the model is fine-tuned without its reasoning.
convmerge reads all of these, writes the one you choose, and `convmerge
tokens` tells you when the template would drop a trace.

## Pick the output for your model

```bash
# Qwen3 (TRL / transformers): reasoning_content, or inline -- Qwen3 accepts both
convmerge convert -i raw.jsonl -o train.jsonl --from auto --format messages \
  --reasoning reasoning_content

# gpt-oss: the template reads `thinking`
convmerge convert -i raw.jsonl -o train.jsonl --from auto --format messages \
  --reasoning thinking --tool-arguments object

# LLaMA-Factory: inline only (sharegpt / alpaca always write traces inline)
convmerge convert -i raw.jsonl -o train.jsonl --from auto --format sharegpt

# A trace in its own column (s1K)
convmerge convert -i s1k.jsonl -o train.jsonl --from auto --format messages \
  --adapter-kwargs '{"chat": {"output_keys": ["deepseek_attempt"],
                              "record_reasoning_keys": ["deepseek_thinking_trajectory"]}}'
```

`--reasoning keep` (the default) leaves inline blocks byte for byte and
writes a separate trace as `reasoning_content`; `--reasoning drop` removes
traces (for a non-reasoning model). `convert` prints how many examples carry a
trace, and so does its report under `reasoning`. If that number is 0 on a
reasoning dataset, the trace sits under a key convmerge does not know: name
it with `reasoning_keys` (turn keys) or `record_reasoning_keys` (flat
columns).

## Multi-turn conversations

Qwen3, DeepSeek-R1, and gpt-oss templates render a trace **only after the
last user turn**. Earlier turns lose theirs, as they do at inference, when
the model never sees its own past reasoning. That shapes what each
trainer learns from a multi-turn reasoning conversation:

- **Trainers that apply the chat template** (TRL `SFTTrainer`): earlier
  traces are silently discarded. In our check on 40 two-turn conversations,
  the Qwen3 template put 40 of the 80 traces into the training text.
  **`--split-turns`** writes one example per user turn, each ending in its
  own answer, and all 80 traces were trained.
- **Trainers that train every assistant turn from the stored text**
  (LLaMA-Factory with ShareGPT data): every inline trace is trained,
  including history traces the model never sees at inference.
  **`--reasoning-turns last`** removes them; LLaMA-Factory's `qwen3`
  template then writes empty `<think>` blocks for those turns, the way
  Qwen3 renders its history.

`convmerge tokens` reports both situations: `reasoning_dropped` counts rows
where the template leaves out some trace; `reasoning_dropped_final` counts
rows where even the last answer's trace is missing, which means the file
stores it under a field the template does not read. Each comes with a hint
naming the fix.

gpt-oss additionally drops the reasoning of a tool-call turn once a final
answer follows it in the same turn. That is by design, and `tokens` counts
it under `reasoning_dropped`.

## Tool calls

Assistant turns that only call tools get `"content": ""`. With `null`, the
Qwen3, gpt-oss, DeepSeek-R1, GLM-4, Mistral, and Phi-4 templates raise an
error. Reasoning tool-call turns in gpt-oss data keep their `thinking`.

## Verified

With the 0.12 code, on converted reasoning data:

- **TRL 1.14 `SFTTrainer`** with the Qwen3 template, on three variants:
  - multi-turn `reasoning_content`: 40 of 80 traces rendered, as `tokens`
    predicted;
  - `--split-turns`: 80 of 80;
  - single-turn inline blocks moved to `reasoning_content`: 40 of 40.
- **TRL 1.14 `SFTTrainer`** with a gpt-oss template on `--reasoning thinking`
  data with tool calls: all final-answer traces rendered.
- **LLaMA-Factory 0.9.5** SFT with the `qwen3` template on
  `--format sharegpt` output, with and without `--reasoning-turns last`.
