# Migrating to 1.0

1.0 removes what 0.9 deprecated and otherwise changes nothing. Code that
runs on **0.9 without `DeprecationWarning`s** runs unchanged on 1.0, so the
simplest path is:

1. Upgrade to 0.9: `pip install "convmerge>=0.9,<1"`.
2. Run your code or tests with warnings visible:
   `python -W error::DeprecationWarning your_script.py` or
   `pytest -W error::DeprecationWarning`.
3. Fix each warning using the table below, then upgrade to 1.0.

The command line, output formats, and recipe / manifest / preset files have
**no** removals: a pipeline that only uses the CLI or `convmerge run` needs no
changes. What 1.x keeps stable is described in [stability.md](stability.md).

## Deprecated in 0.9, removed in 1.0

| Deprecated | Use instead |
|------------|-------------|
| `from convmerge.normalize import load_jsonl` | `convmerge.iter_jsonl(path)` — e.g. `[line.value for line in iter_jsonl(path)]`; it raises `JsonlDecodeError` on a bad line with `on_error="raise"`, or skips it (the default) |
| `load_jsonl(path)` returning `[]` when a line is malformed | as above: silently getting an empty list for a corrupt file goes away |
| `from convmerge.normalize import key_frequency, is_uniform_schema` | `convmerge.profile_schema(path)` (per-field presence and types) |
| `from convmerge.normalize import count_turns, is_single_turn` | `convmerge.analyze_turn_distribution(path)` / `split_by_turns(...)`; for one record, count its `assistant` messages (that is all `count_turns` did) |
| `from convmerge.normalize import detect_jsonl_shape, iter_json_records` | `convmerge.normalize_to_jsonl(src, dst)` then `convmerge.iter_jsonl(dst)` |
| `from convmerge.normalize import single_turn_to_multi_turn_record, multi_turn_to_single_turn_record` | `convmerge.convert_file(..., adapter_name="alpaca", output_format="messages")` (and `output_format="alpaca"` the other way) |
| `from convmerge.fetch import classify_entry, sanitize_name, redact_url, resolve_token` | internal helpers with no public replacement; `run_manifest` applies them for you |
| `convmerge.cli.FETCH_FILE_EXTENSIONS`, `convmerge.cli.SIDECAR_SUFFIXES` | none; `convmerge.cli` exports only `main()` |
| `convmerge.convert.iter_converted_lines` | `convmerge.convert_file`, or an adapter plus emitter for in-memory use |

`convmerge.normalize` keeps re-exporting the public functions it holds
(`normalize_to_jsonl`, `deduplicate_jsonl`, `DedupeStats`,
`analyze_turn_distribution`, `split_by_turns`), but import them from
`convmerge` in new code.

## Behaviour changes in 0.12

- `--format messages` / `preference`: an assistant turn that only calls
  tools is written with `"content": ""` instead of `null`. `null` breaks the
  Qwen3, gpt-oss, DeepSeek-R1, GLM-4, Mistral, and Phi-4 chat templates; `""`
  renders in all of them. `--tool-content null` (or `tool_content: null` in a
  preset / recipe, `EmitOptions(tool_content="null")`) restores the old
  output.
- Assistant turns with a `reasoning_content`, `thinking`, or `reasoning`
  string now keep it (written as `reasoning_content` by default; see
  `--reasoning`). Before 0.12 these fields were dropped.
- Llama-Nemotron rows (`input` turns + `output` answer) convert instead of
  being dropped, and a `system_prompt` column becomes the system turn like
  `system` does.
- `convert --report` and `tokens` JSON gain keys (`transforms`, `reasoning`;
  `generation_tags`, `answer_beyond_limit`, `stop_tokens`, `missing_eos`,
  `reasoning_dropped`, `reasoning_dropped_final`, `hints`); existing keys are
  unchanged.

## Behaviour changes already in 0.9

These are not deprecations; they shipped in 0.9 and stay:

- `--report`, `validate`, and recipe `report.json` output gain a top-level
  `"version": 1` field.
- Exit codes are consistent: an invalid or missing preset, manifest, mix
  config, or recipe, and an unsupported `fetch` URL, exit **2** (previously
  some exited 1 or crashed with a traceback on invalid YAML); `turns` and
  `dedupe` on a missing file exit 1 with an error message instead of a
  traceback.
