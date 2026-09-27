# Output formats and adapters

## Design

1. **Adapters** parse one JSON object per input line (JSONL) into zero or more `TrainingExample` values (internal `messages` list).
2. **Emitters** turn each `TrainingExample` into one JSON object written as a single output line.

Invalid JSON lines are skipped. Empty adapter output yields no output lines.

## Output formats (`--format`)

### `messages`

OpenAI-style chat JSONL:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Roles are `system`, `user`, `assistant`, or `tool`. Plain-text turns look
exactly like the line above; richer data adds only what it needs:

- **Tool calling** — assistant `tool_calls` (`{"id"?, "type": "function",
  "function": {"name", "arguments"}}`), `tool` messages with `tool_call_id`
  when the source had ids, and a top-level `tools` list of schemas.
  `arguments` is a JSON string (OpenAI style); `--tool-arguments object`
  writes an object instead, which some Hugging Face chat templates expect.
  An assistant turn that only calls tools has `"content": ""`: `null` makes
  the Qwen3, gpt-oss, DeepSeek-R1, GLM-4, Mistral, and Phi-4 templates fail,
  while `""` renders everywhere. `--tool-content null` writes `null` (the
  0.11 output).
- **Reasoning** — see [Reasoning traces](#reasoning-traces-reasoning).
- **Multimodal** — `content` becomes a list of parts:
  `{"type": "text", "text"}`, `{"type": "image_url", "image_url": {"url"}}`,
  and `audio_url` / `video_url` in the same shape (vLLM / Qwen-VL
  convention). The URL is whatever reference the source held (URL or path);
  media is never downloaded. An unresolved placeholder stays as
  `{"type": "image"}` and fails validation.
- `name` on a message when the source had one.

```json
{"messages": [
  {"role": "user", "content": "Weather in Seoul?"},
  {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Seoul\"}"}}]},
  {"role": "tool", "content": "{\"temp_c\": 21}", "tool_call_id": "call_1"},
  {"role": "assistant", "content": "It is 21°C in Seoul."}],
 "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]}
```

### `alpaca`

Instruction tuning JSONL (LLaMA-Factory compatible):

```json
{"instruction": "...", "input": "", "output": "...", "system": "..."}
```

`system` is written only when the example has a system prompt. Anything other
than a single user→assistant pair follows `--alpaca-multiturn`:

| Value | Result |
|-------|--------|
| `flatten` *(default)* | User turns joined into `instruction`, last assistant turn as `output`. Earlier assistant turns are lost; each case is counted as `lossy_multiturn_flattened`. |
| `history` | Last pair as `instruction`/`output`, earlier pairs in `history: [[user, assistant], ...]`. Lossless for strictly alternating conversations; others are dropped (`unrepresentable_multiturn`). |
| `drop` | Multi-turn examples are dropped (`unrepresentable_multiturn`). |

Examples with tool calls, a `tools` schema list, or media cannot be represented
in this format and are always dropped (`unrepresentable_tool_calls` /
`unrepresentable_media`).

### `preference`

DPO-style preference pairs in TRL's conversational format — what
`trl.DPOTrainer` (and ORPO / CPO / KTO-from-pairs setups) read directly:

```json
{"prompt": [{"role": "user", "content": "Name a fruit."}],
 "chosen": [{"role": "assistant", "content": "Apple."}],
 "rejected": [{"role": "assistant", "content": "Carrot."}]}
```

The adapter reads the record twice — once with the chosen answer, once with
the rejected one ([shapes below](#preference-data)) — and the turns both
conversations share become `prompt` (system prompt and earlier turns
included); each side keeps its own continuation. `tools` is written when the
conversation has tool schemas, and `--keep-meta` / `--tool-arguments` apply
as for `messages`. Dropped, with a reason:

| Reason | When |
|--------|------|
| `unrepresentable_not_preference` | the record has no chosen/rejected pair |
| `unrepresentable_identical_pair` | chosen and rejected are the same |
| `unrepresentable_incomplete_pair` | no user turn in the prompt, or a side has no assistant answer |

`convmerge validate` checks `messages` rows, not preference rows.

### `sharegpt` output (LLaMA-Factory, Unsloth)

LLaMA-Factory / Unsloth ShareGPT rows (register them for LLaMA-Factory with
`convmerge llamafactory-info`, see [guides/llamafactory.md](guides/llamafactory.md)):

```json
{"conversations": [{"from": "human", "value": "<image>\nWhat is this?"},
                   {"from": "function_call", "value": "{\"name\": \"lookup\", \"arguments\": {\"q\": \"bus\"}}"},
                   {"from": "observation", "value": "{\"answer\": \"a bus\"}"},
                   {"from": "gpt", "value": "A bus."}],
 "system": "Be brief.", "tools": "[{\"name\": \"lookup\", ...}]", "images": ["bus.jpg"]}
```

- The system prompt goes to `system`; tool schemas to `tools` as a JSON
  string of function specs; media to `<image>` / `<video>` / `<audio>` tokens
  plus `images` / `videos` / `audios` lists.
- An assistant turn with tool calls becomes one `function_call` turn (a JSON
  list for parallel calls); consecutive tool results become one
  `observation` (joined by newlines). Text written next to a tool call has
  no place in this format and is counted as `lossy_tool_call_text`.
- LLaMA-Factory requires turns that alternate user side (`human`,
  `observation`) and model side (`gpt`, `function_call`), starting with
  `human` and ending with a model turn. Other conversations — two user turns
  in a row, a system turn in the middle — are dropped as
  `unrepresentable_role_order`.

### `sharegpt-preference` output (LLaMA-Factory ranking)

LLaMA-Factory ranking rows for DPO / ORPO / reward modeling: the prompt as
`conversations`, each answer as one `gpt` turn.

```json
{"conversations": [{"from": "human", "value": "Name a fruit."}],
 "chosen": {"from": "gpt", "value": "Apple."}, "rejected": {"from": "gpt", "value": "Carrot."}}
```

Pairs are built as for [`preference`](#preference). Answers that are
multi-turn or tool calls are dropped as `unrepresentable_pair_continuation`
(use `preference` with TRL for those).

### Reasoning traces (`--reasoning`)

An assistant turn's reasoning is stored one of two ways, and chat templates
disagree on which they read:

| Stored as | Written by | Read by |
|-----------|-----------|---------|
| inline `<think>...</think>` before the answer | DeepSeek-R1 distillations, OpenR1, OpenThoughts, most ShareGPT sets | Qwen3 and DeepSeek-R1 templates (they split it off), LLaMA-Factory, any template as plain text |
| a `reasoning_content` field on the turn | DeepSeek API, vLLM | Qwen3 templates |
| a `thinking` field on the turn | gpt-oss data (`HuggingFaceH4/Multilingual-Thinking`) | gpt-oss templates |

Adapters read `reasoning_content`, `thinking`, and `reasoning` turn keys
(`ChatMessage.reasoning`); inline blocks stay in the content. For flat
question/answer records, name the trace column with
`--adapter-kwargs '{"chat": {"record_reasoning_keys": ["trace_column"]}}'`
(it is off by default because a top-level `reasoning` column is often an
on/off flag, as in Llama-Nemotron). `--reasoning` decides where traces go:

| `--reasoning` | Effect |
|---------------|--------|
| `keep` (default) | Inline blocks stay as they are, byte for byte; a separate trace is written as `reasoning_content`. |
| `reasoning_content` / `thinking` | Inline blocks move into that field (use the one your target template reads). `messages` and `preference` only. |
| `inline` | Every trace is written as `<think>\n...\n</think>\n\n` before the answer. |
| `drop` | Traces are removed (fields and inline blocks). |

`alpaca`, `sharegpt`, and `sharegpt-preference` have no reasoning field and
always write traces inline. `convert` prints how many examples carry a trace
(`reasoning` in the report). Most reasoning templates render a trace only
after the last user turn; see `--reasoning-turns` and `--split-turns` below,
and `convmerge tokens`, which reports traces the template does not render.

### Fixes for strict chat templates

Off by default; each one is counted in the report's `transforms`:

| Flag | Effect | For |
|------|--------|-----|
| `--system fold` | Leading system turns are prepended to the first user turn (`--system drop` removes all system turns). | Templates without a system role (Gemma 2: "System role not supported"). |
| `--merge-consecutive` | Consecutive user turns, or consecutive assistant turns without tool calls, are joined with a blank line. Turns from different named speakers and tool turns are left alone. | "Conversation roles must alternate" (Mistral, Gemma, Llama 2); LLaMA-Factory's `unrepresentable_role_order`. |
| `--reasoning-turns last` | Removes the reasoning of assistant turns before the last user turn. | Qwen3 / gpt-oss / DeepSeek-R1 templates, which drop those traces at inference. |
| `--split-turns` | One example per user turn (the conversation up to the next user turn); earlier answers keep their text but lose their reasoning; `meta.turn` records the position. Not for preference formats. | Training every turn of a multi-turn reasoning conversation the way the model sees it. |

Order: system, merge, split, reasoning turns. The same options exist in
presets (`transforms:`), recipes (source `convert` keys), and the API
(`TransformOptions`).

### Provenance (`--keep-meta`)

`--keep-meta` adds the example's provenance under `meta` (`--meta-key` to
rename it): `source` (the adapter branch, e.g. `sharegpt`, `chat:pairwise`),
the record's own `id` when it has one, and the pairwise `branch`.
`--keep-meta source,id` keeps only those keys. Off by default.

## Source adapters (`--from`)

### `alpaca`

Expects keys such as `instruction`, optional `input`, and `output` (or `response`).

<details>
<summary>Sample input → output</summary>

Input (one JSONL line):

```json
{"instruction": "Say hello", "input": "", "output": "Hello!"}
```

Output with `--format messages`:

```json
{"messages": [{"role": "user", "content": "Say hello"}, {"role": "assistant", "content": "Hello!"}]}
```

Output with `--format alpaca`:

```json
{"instruction": "Say hello", "input": "", "output": "Hello!"}
```

</details>

### `sharegpt`

Expects `conversations`: list of `{"from": "human"|"gpt"|..., "value": "..."}`.
The `turn_mode` option controls how multi-turn conversations are emitted:

| `turn_mode` | Output |
|-------------|--------|
| `full` *(default since 0.6.0)* | One example with the whole conversation — system prompt and every turn, in order. Turns with an empty value are dropped. |
| `pairs` | One example per consecutive user→assistant pair (the 0.5.x default). System prompts, unpaired turns, tool turns, and earlier-turn context are dropped. |

Set it with `--adapter-kwargs '{"sharegpt": {"turn_mode": "pairs"}}'` or in a
preset under `adapter_options.sharegpt.turn_mode` (see
[custom_presets.md](custom_presets.md)).

In `full` mode the LLaMA-Factory ShareGPT extensions are understood:

- `function_call` turns (a JSON `{"name", "arguments"}` object, or a list for
  parallel calls) become assistant `tool_calls`; `observation` turns become
  `tool` messages.
- A `system` column becomes a system message (unless the conversation already
  has one); a `tools` column (JSON string or list) becomes top-level `tools`.
- `images` / `videos` / `audios` columns (and LLaVA's single `image`) bind in
  order to `<image>` / `<video>` / `<audio>` tokens, producing content parts.

<details>
<summary>Sample input → output</summary>

Input (one JSONL line):

```json
{"conversations": [{"from": "human", "value": "2+2?"}, {"from": "gpt", "value": "4"}]}
```

Output with `--format messages`:

```json
{"messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]}
```

Output with `--format alpaca`:

```json
{"instruction": "2+2?", "input": "", "output": "4"}
```

</details>

### `chat` (alias: `auto`)

A **heuristic** auto-detecting adapter for mixed or unknown chat schemas.
It does not read a schema descriptor — it inspects which keys are present on
each record and routes to the matching branch. Unusual or ambiguous shapes
may not be detected correctly; in that case either pin an explicit adapter
(`alpaca`, `sharegpt`) or call `iter_from_chat_line` directly with overridden
key lists (see below).

Tries, in order:

1. Pairwise preference rows (`conversation_a` / `conversation_b` with optional `winner`).
   - Default `pairwise_mode="winner"` emits only the winning branch; ties/unknown are skipped.
   - `pairwise_mode="both"` emits both branches; `"a"` / `"b"` always pick one side.
2. Chat-list containers named `messages`, `conversation`, or `conversations`.
   - Both `{role, content}` and ShareGPT-style `{from, value}` entries work,
     as do Capybara-style `{input, output}` turn pairs.
   - A default role map normalizes `human → user`, `gpt/bing/bot/model → assistant`,
     and `function/observation → tool`.
   - OpenAI-style content arrays are kept as content parts (text, and media by
     reference: `image_url`, `{"type": "image"}` placeholders, audio, video);
     text-only arrays collapse to a string. `tool_calls` (and the legacy
     `function_call`), `tool_call_id`, and `name` are kept.
   - Record-level `tools`, `system`, and media columns are handled as for the
     `sharegpt` adapter above.
   - Hermes-style tool calling is decoded ([below](#tool-calling-encodings)).
3. Tool-calling records in other layouts: Glaive `system` + `chat`
   transcripts and xLAM `query` / `answers` / `tools`
   ([below](#tool-calling-encodings)).
4. A `text` column rendered with a known chat template is split back into
   turns ([below](#template-rendered-text)). Any other `text` → emitted as a
   single assistant message (and then dropped as `no_user`) — **but only when the
   record does not carry strong Alpaca cues.** If both an instruction key
   and an output key (see step 6) are present, the record is routed to the
   Alpaca branch (step 6) instead, so a stray `text` field cannot silently
   discard the instruction/output pair. When `text` is taken while only a
   partial Alpaca key is present, a `logging` warning is emitted.
5. Llama-Nemotron rows: prompt turns in an `input` list and the answer in
   an `output` string (`system_prompt` becomes the system turn, as `system`
   does elsewhere).
6. Fallback: flat question/answer keys — `instruction` / `question` / `prompt` /
   `problem` / `query`, optional `input` / `context`, and the first of `output` /
   `response` / `completion` / `solution` / `answer` (so a full `solution` wins over
   a short final `answer`) — with the `alpaca` adapter's `system` / `history` handling.

You can override every part (`conversation_keys`, `role_keys`, `content_keys`,
`role_map`, `instruction_keys`, `input_keys`, `output_keys`, `pairwise_mode`,
`reasoning_keys`, `record_reasoning_keys`) with `--adapter-kwargs` or when
calling `iter_from_chat_line` programmatically.

<details>
<summary>Sample inputs → output (<code>--format messages</code>)</summary>

The heuristic routes each shape to the same `messages` output:

`messages` list:

```json
{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
```
```json
{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
```

ShareGPT-style `conversations`:

```json
{"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]}
```
```json
{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
```

Plain `text`:

```json
{"text": "some raw text"}
```
```json
{"messages": [{"role": "assistant", "content": "some raw text"}]}
```

</details>

### Tool-calling encodings

Tool calls always come out as OpenAI `tool_calls` on assistant turns, results
as `tool` turns (with the function `name` when known), and schemas as a
top-level `tools` list — whatever the source used:

| Source layout | Example dataset | How it is read |
|---------------|-----------------|----------------|
| OpenAI `tool_calls` / `tool_call_id` | — | as is |
| LLaMA-Factory `function_call` / `observation` turns + `tools` column | `llamafactory/glaive_toolcall_en` | calls decoded from the turn value |
| Hermes `<tool_call>` / `<tool_response>` tags | `NousResearch/hermes-function-calling-v1` | each `<tool_call>` block becomes a call (text around it is kept); each `<tool_response>` block becomes its own `tool` turn; schemas from the `tools` column or the system prompt's `<tools>` block |
| Glaive `system` + `chat` strings | `glaiveai/glaive-function-calling-v2` | `USER:` / `ASSISTANT:` / `FUNCTION RESPONSE:` turns; `<functioncall>` becomes a call; the function JSON in the system prompt moves to `tools` |
| xLAM `query` / `answers` / `tools` | `Salesforce/xlam-function-calling-60k` | user query + one assistant turn with the calls |

Hermes tags are decoded only in conversations that have a `tools` column, a
`tool` turn, or a `<tools>` block in the system prompt, so ordinary text that
mentions `<tool_call>` is left alone; a block that is not valid JSON stays in
the text. The system prompt itself is kept as written. Tool calls get no
invented ids (results pair with calls by order).

### Template-rendered `text`

Some datasets store each conversation as one string already rendered with a
chat template. These are split back into turns:

| Template | Marker | Example dataset |
|----------|--------|-----------------|
| ChatML | `<\|im_start\|>role ... <\|im_end\|>` | `OpenAssistant/oasst_top1_2023-08-25` |
| Llama 3 | `<\|start_header_id\|>role<\|end_header_id\|>` | |
| Gemma | `<start_of_turn>user` / `model` | |
| Llama 2 | `[INST] <<SYS>>...<</SYS>> ... [/INST]` | `mlabonne/guanaco-llama2-1k` |
| Guanaco | `### Human: ... ### Assistant: ...` | `timdettmers/openassistant-guanaco` |
| HH-RLHF | `Human: ...` / `Assistant: ...` separated by blank lines | |
| Alpaca prompt | `### Instruction:` / `### Input:` / `### Response:` | |

A trailing user turn without an answer is dropped (Guanaco often ends with
one). Records that also have Alpaca `instruction` / `output` keys use those
instead.

## Preference data

Preference datasets (DPO, reward models) keep two answers per prompt. Write
them as pairs with `--format preference` ([above](#preference)), or train SFT
on one side with `--preference chosen` (or `rejected`; also
`adapter_options.preference` in a preset, `{"preference": "chosen"}` in
`--adapter-kwargs`), which folds that answer into the conversation first.
Without either, an SFT conversion that cannot use such a record drops it as
`preference_record` and the CLI says which option to use. Both work with any
adapter and these shapes:

| Shape | Example | Result |
|-------|---------|--------|
| LLaMA-Factory ranking | `conversations` + `chosen: {"from": "gpt", "value"}` | chosen appended as the last assistant turn |
| LLaMA-Factory ranking (alpaca) | `instruction` + `chosen: "..."` | chosen used as `output` |
| HH-RLHF | `chosen: "\n\nHuman: ...\n\nAssistant: ..."` | transcript parsed into turns |
| UltraFeedback-binarized | `chosen: [user, assistant, ...]` | the list is the conversation |
| TRL prompt + continuation | `prompt` (string or messages) + `chosen: [assistant ...]` | prompt followed by the continuation |
| Orca DPO pairs | `system` + `question` + `chosen: "..."` | chosen used as the answer |
| Chatbot Arena (`--format preference` only) | `conversation_a` / `conversation_b` + `winner` | the winner is chosen, the other side rejected; ties are skipped |

## Validation

Every example `convert` produces is checked before it is written. Examples
that would train badly are **dropped by default** and counted by reason:

| Reason | Meaning |
|--------|---------|
| `no_messages` | the example has no messages |
| `unknown_role` | a role is not `system` / `user` / `assistant` / `tool` (e.g. an undecodable `function_call` turn) |
| `empty_message` | a message has neither content nor tool calls |
| `no_user` | there is no user message (e.g. a plain `text` record) |
| `no_assistant` | there is no assistant message with content or tool calls |
| `orphan_tool_message` | a `tool` message is not preceded by an assistant tool call |
| `tool_call_id_mismatch` | a `tool_call_id` matches no earlier tool call id |
| `unresolved_image` / `_video` / `_audio` | a media placeholder has no matching reference in the record |
| `unused_image` / `_video` / `_audio` | the record lists more media references than placeholders |
| `preference_record` | a chosen/rejected record in an SFT conversion (use `--format preference` or `--preference chosen`) |
| `unrepresentable_*` | the output format cannot hold the example losslessly (see the output formats above) |

Adapters skip blank turns (such as an empty system prompt) instead of
failing the whole example. Tool calls without ids (LLaMA-Factory) are paired
with tool results by order.

```bash
convmerge convert -i in.jsonl -o out.jsonl --from auto -f messages \
  --on-invalid drop \          # drop (default) | keep | fail
  --report out.report.json      # counts, reasons, first line numbers per reason

convmerge validate -i out.jsonl # same checks on an existing file; exit 1 if any fail
```

From Python: `convert_file(..., on_invalid="drop", stats=ConvertStats())`,
`convmerge.validate_example(example)`, and `convmerge.validate_file(path)`.

## Normalization utilities

These functions (all importable from `convmerge`) and the
`convmerge normalize / dedupe / turns` subcommands handle the pre-adapter
cleanup step:

- `normalize_to_jsonl(src, dst)` — rewrites parquet, JSON arrays, concatenated
  single-line JSON, or already-valid JSONL into clean newline-delimited JSONL.
  Parquet requires the `parquet` extra. Records that are themselves arrays —
  e.g. JSONL with one conversation per line stored as a list of turns — are
  wrapped as `{"conversation": [...]}` (`--array-key` / `array_key=` to
  rename), which the `chat` adapter reads directly; `inspect` profiles them
  the same way.
- `profile_schema(path)` / `convmerge inspect` — per-field types, presence,
  examples, nested `fields` / `items`, and for list fields `element_types` /
  `element_examples` (so a list of bare strings is visible).
- `deduplicate_jsonl(src, dst, keys=None, algorithm="md5")` — streaming dedup
  by an MD5/SHA256 hash of the whole record or a projected subset of keys.
  Seen hashes are tracked in memory by default; pass `seen_store="sqlite"`
  (CLI `--seen-store sqlite`, optional `--seen-db PATH`) to keep them in a
  disk-backed SQLite table for bounded memory on inputs with tens of millions
  of unique rows.
- `analyze_turn_distribution(path)` — reports single-turn vs multi-turn counts
  for messages-style JSONL, plus a per-turn-count histogram.
- `split_by_turns(src, single_out=..., multi_out=...)` — partitions a
  messages-style JSONL into single-turn and multi-turn files.
- `profile_schema(path_or_records, max_rows=None)` — infer a structural profile
  of a `.json` / `.jsonl` file: per-field value types, presence ratio, sample
  values, and nested `items` (list-of-object) / `fields` (object) so paths like
  `messages[].role` are distinguishable from a top-level `role`. Exposed on the
  CLI as `convmerge inspect -i FILE [--max-rows N] [--max-examples K]`, which is
  the recommended first step before choosing an adapter / writing key mappings
  for an unfamiliar dataset.

The other helpers `convmerge.normalize` used to re-export
(`single_turn_to_multi_turn_record`, `key_frequency`, `load_jsonl`, …) are
deprecated since 0.9; see [migration-1.0.md](migration-1.0.md).

### Progress reporting

`convert` and `dedupe` accept a `--progress` flag (or set the
`CONVMERGE_PROGRESS=1` environment variable) to log periodic row counts and
throughput to stderr for long-running jobs. Progress is **off by default**, so
normal output is unchanged. The `convert_file` and `deduplicate_jsonl` library
functions expose the same behavior via a `progress=True` keyword argument.

## Non-goals (current)

- **Model loading, inference, or training.** `convmerge` never imports
  PyTorch / Transformers / vLLM / etc. See the README's "Out of scope"
  section.
- **Automatic labeling or classification of samples.** No topic, quality,
  or safety classifier ships with this package.
- **Prompt-template rendering for specific model families.** Output JSONL
  uses the standard `messages` / `alpaca` shapes; downstream trainers apply
  their own chat template.
- **Tokenizer-aware length filtering or packing.** These live in the
  training stack.
- **HTML / raw web pages.** Full-site scraping and boilerplate removal are
  out of scope for core; a future optional extra may wrap libraries like
  `trafilatura`.
- **Binary / proprietary formats.** Not supported unless added as explicit
  adapters.
- **Guaranteed lossless round-trip** across all formats: not guaranteed;
  formats are views over the internal message list.

## Adding an adapter

1. Implement `iter_from_<name>_line(record: dict) -> Iterator[TrainingExample]` in `src/convmerge/adapters/`.
2. Register it in `convmerge.adapters.ADAPTERS`.
3. Add synthetic JSONL under `tests/fixtures/` and tests in `tests/test_adapters.py`.
