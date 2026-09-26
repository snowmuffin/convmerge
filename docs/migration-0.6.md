# Migrating to 0.6

0.6 makes `convert` keep what 0.5 silently dropped (tool calls, media
references, system prompts, whole ShareGPT conversations) and stop writing
examples that would train badly. Most pipelines need no changes. Read this if
you diff outputs between versions or call the Python API.

## Output changes

| Change | Before (0.5.x) | Now (0.6) | To get the old behavior |
|--------|----------------|-----------|-------------------------|
| `--from sharegpt` | one example per user→assistant pair | one example per conversation (`turn_mode: full`) | `--adapter-kwargs '{"sharegpt": {"turn_mode": "pairs"}}'` |
| Invalid examples (no user turn, empty or orphan tool messages, ...) | written | dropped and counted | `--on-invalid keep` |
| OpenAI `tool_calls`, `tool` messages, `tools` | dropped (tool results kept without their calls) | kept | — (the old output was inconsistent) |
| OpenAI content arrays / vision samples | whole turn dropped (e.g. the user question) | kept as content parts, media by reference | — |
| LLaMA-Factory `function_call` / `observation` turns | written as unknown roles | assistant `tool_calls` / `tool` messages | — |
| `system`, `history`, `images` columns | ignored | system message / earlier turns / media parts | — |
| `--format alpaca` with a system prompt | system dropped | `"system"` field | drop the key downstream |
| `--format alpaca` with tool calls or media | reduced to text | dropped (`unrepresentable_*`) | — |
| Plain `text` records via `chat` | assistant-only example written | dropped (`no_user`) | `--on-invalid keep` |
| Blank turns (e.g. empty system prompt) | kept as empty messages | skipped | — |
| Flat Q&A with both `solution` and `answer` (LIMO, NuminaMath) | short `answer` used as target | full `solution` used | `adapter_options.chat.output_keys: [output, response, answer]` |
| `problem`/`solution`, `query`/`response`, `prompt`/`completion` records | dropped or assistant missing | converted | — |
| Diagnostics | some on stdout | stderr only | — |

Every drop is counted by reason on stderr; `--report PATH` writes the counts
and sample line numbers as JSON. `convmerge validate -i FILE` checks an
existing file with the same rules.

## New options

- `convert --on-invalid drop|keep|fail`, `--report PATH`
- `convert --tool-arguments string|object`
- `convert --alpaca-multiturn flatten|history|drop`
- `convert --keep-meta [KEYS]`, `--meta-key`
- `normalize --array-key` (JSONL whose lines are arrays)
- presets: `adapter_options.sharegpt.turn_mode`, `output_options.*`

## Python API

- `ChatMessage.content` may be `str`, a tuple of `ContentPart`, or `None`.
  Use `message.text` where you need a string. `ChatMessage(role, content)`
  still works, and new fields (`tool_calls`, `tool_call_id`, `name`) are
  optional.
- `TrainingExample` gains `tools` and `issues`.
- `convert_file(...)` gains `on_invalid=` and `emit_options=EmitOptions(...)`;
  it still returns `(lines_read, lines_written)`. `ConvertStats` gains
  `dropped`, `kept_invalid`, `drop_reasons`, `drop_lines`, and `lossy`.
- New: `convmerge.validate.validate_example`, `convmerge.convert.validate_file`,
  `convmerge.io.iter_jsonl`, `convmerge.emitters.EmitOptions`.
- `iter_json_records` yields array records as `{"conversation": [...]}`;
  pass `array_key=None` to skip them as before.
- `load_jsonl` and `run_manifest` no longer print to stdout (logger / stderr).
