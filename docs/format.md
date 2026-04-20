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

## Non-goals (current)

- **HTML / raw web pages**: full-site scraping and boilerplate removal are out of scope for core; a future optional extra may wrap libraries like `trafilatura`.
- **Binary / proprietary formats**: not supported unless added as explicit adapters.
- **Guaranteed lossless round-trip** across all formats: not guaranteed; formats are views over the internal message list.

## Adding an adapter

1. Implement `iter_from_<name>_line(record: dict) -> Iterator[TrainingExample]` in `src/convmerge/adapters/`.
2. Register it in `convmerge.adapters.ADAPTERS`.
3. Add synthetic JSONL under `tests/fixtures/` and tests in `tests/test_adapters.py`.
