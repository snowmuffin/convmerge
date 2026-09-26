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
  {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Seoul\"}"}}]},
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

Examples with tool calls or media cannot be represented in this format and are
always dropped (`unrepresentable_tool_calls` / `unrepresentable_media`).

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
   - Both `{role, content}` and ShareGPT-style `{from, value}` entries work.
   - A default role map normalizes `human → user`, `gpt/bing/bot/model → assistant`,
     and `function/observation → tool`.
   - OpenAI-style content arrays are kept as content parts (text, and media by
     reference: `image_url`, `{"type": "image"}` placeholders, audio, video);
     text-only arrays collapse to a string. `tool_calls` (and the legacy
     `function_call`), `tool_call_id`, and `name` are kept.
   - Record-level `tools`, `system`, and media columns are handled as for the
     `sharegpt` adapter above.
3. Plain `text` → emitted as a single assistant message — **but only when the
   record does not carry strong Alpaca cues.** If both an instruction key
   (`instruction`/`question`/`prompt`) and an output key
   (`output`/`response`/`answer`) are present, the record is routed to the
   Alpaca branch (step 4) instead, so a stray `text` field cannot silently
   discard the instruction/output pair. When `text` is taken while only a
   partial Alpaca key is present, a `logging` warning is emitted.
4. Fallback: alpaca-like keys (`instruction`/`question`/`prompt` + `output`/`response`/`answer`),
   with the `alpaca` adapter's `system` / `history` handling.

You can override every part (`conversation_keys`, `role_keys`, `content_keys`,
`role_map`, `instruction_keys`, `input_keys`, `output_keys`, `pairwise_mode`)
when calling `iter_from_chat_line` programmatically.

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
| `unrepresentable_*` | the output format cannot hold the example losslessly (see `alpaca` below) |

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
`convmerge.validate.validate_example(example)`, and
`convmerge.convert.validate_file(path)`.

## Normalization utilities

`convmerge.normalize` and the `convmerge normalize / dedupe / turns`
subcommands handle the pre-adapter cleanup step:

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
- `single_turn_to_multi_turn_record` / `multi_turn_to_single_turn_record` —
  round-trip between `{instruction, input, output}` and `{messages: [...]}`.
- `profile_schema(path_or_records, max_rows=None)` — infer a structural profile
  of a `.json` / `.jsonl` file: per-field value types, presence ratio, sample
  values, and nested `items` (list-of-object) / `fields` (object) so paths like
  `messages[].role` are distinguishable from a top-level `role`. Exposed on the
  CLI as `convmerge inspect -i FILE [--max-rows N] [--max-examples K]`, which is
  the recommended first step before choosing an adapter / writing key mappings
  for an unfamiliar dataset. (`key_frequency` / `is_uniform_schema` remain as
  lighter-weight helpers.)

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
