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

Roles are normalized to `user`, `assistant`, or `system` where applicable.

### `alpaca`

Instruction tuning JSONL:

```json
{"instruction": "...", "input": "", "output": "..."}
```

For multi-turn internal examples, user contents are joined into `instruction` and the last assistant reply becomes `output` (MVP flattening).

## Source adapters (`--from`)

### `alpaca`

Expects keys such as `instruction`, optional `input`, and `output` (or `response`).

### `sharegpt`

Expects `conversations`: list of `{"from": "human"|"gpt"|..., "value": "..."}`.
Emits one example per consecutive user→assistant pair.

### `chat` (alias: `auto`)

Auto-detecting adapter for mixed / unknown chat schemas. Tries, in order:

1. Pairwise preference rows (`conversation_a` / `conversation_b` with optional `winner`).
   - Default `pairwise_mode="winner"` emits only the winning branch; ties/unknown are skipped.
   - `pairwise_mode="both"` emits both branches; `"a"` / `"b"` always pick one side.
2. Chat-list containers named `messages`, `conversation`, or `conversations`.
   - Both `{role, content}` and ShareGPT-style `{from, value}` entries work.
   - A default role map normalizes `human → user`, `gpt/bing/bot → assistant`.
3. Plain `text` → emitted as a single assistant message.
4. Fallback: alpaca-like keys (`instruction`/`question`/`prompt` + `output`/`response`/`answer`).

You can override every part (`conversation_keys`, `role_keys`, `content_keys`,
`role_map`, `instruction_keys`, `input_keys`, `output_keys`, `pairwise_mode`)
when calling `iter_from_chat_line` programmatically.

## Normalization utilities

`convmerge.normalize` and the `convmerge normalize / dedupe / turns`
subcommands handle the pre-adapter cleanup step:

- `normalize_to_jsonl(src, dst)` — rewrites parquet, JSON arrays, concatenated
  single-line JSON, or already-valid JSONL into clean newline-delimited JSONL.
  Parquet requires the `parquet` extra.
- `deduplicate_jsonl(src, dst, keys=None, algorithm="md5")` — streaming dedup
  by an MD5/SHA256 hash of the whole record or a projected subset of keys.
- `analyze_turn_distribution(path)` — reports single-turn vs multi-turn counts
  for messages-style JSONL, plus a per-turn-count histogram.
- `split_by_turns(src, single_out=..., multi_out=...)` — partitions a
  messages-style JSONL into single-turn and multi-turn files.
- `single_turn_to_multi_turn_record` / `multi_turn_to_single_turn_record` —
  round-trip between `{instruction, input, output}` and `{messages: [...]}`.

## Non-goals (current)

- **HTML / raw web pages**: full-site scraping and boilerplate removal are out of scope for core; a future optional extra may wrap libraries like `trafilatura`.
- **Binary / proprietary formats**: not supported unless added as explicit adapters.
- **Guaranteed lossless round-trip** across all formats: not guaranteed; formats are views over the internal message list.

## Adding an adapter

1. Implement `iter_from_<name>_line(record: dict) -> Iterator[TrainingExample]` in `src/convmerge/adapters/`.
2. Register it in `convmerge.adapters.ADAPTERS`.
3. Add synthetic JSONL under `tests/fixtures/` and tests in `tests/test_adapters.py`.
