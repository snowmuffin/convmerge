# Python API

Everything below is importable from the top-level package
(`from convmerge import convert_file`), plus the `convmerge.recipe` and
`convmerge.fetch` modules, and is the **public API**. What exactly is
promised, and how anything is deprecated before it goes, is in
[stability.md](stability.md); from 1.0 this list only grows within a major
version. Modules and names not listed here, or that start with `_`, are
internal and may change without notice.

## Convert pipeline

| Name | Purpose |
|------|---------|
| `convert_file(input, output, *, adapter_name, output_format, ...)` | Read JSONL → adapter → validate → emit. Returns `(lines_read, lines_written)`. Options: `encoding`, `adapter_options`, `progress`, `stats`, `on_invalid` (`drop`/`keep`/`fail`), `emit_options`, `workers`, `transform_options`. |
| `convert_with_config(input, output, cfg, ...)` | Same, from a resolved `ConvertConfig`. |
| `build_convert_config(*, preset_path, adapter, output_format, ...)` | Merge a preset file, `--adapter-kwargs` JSON, and explicit overrides into a `ConvertConfig`. |
| `validate_file(input, *, adapter_name="chat")` | Run validation only; returns `ConvertStats`. |
| `ConvertStats` | Counters and drop reasons; pass one as `stats=` and read it afterwards. `to_report()` gives the `--report` JSON. |
| `InvalidExampleError` | Raised by `on_invalid="fail"`; has `line_number` and `reasons`. |
| `ConvertConfig`, `AdapterOptions`, `ChatAdapterOptions`, `SharegptAdapterOptions` | Adapter configuration (see [custom_presets.md](custom_presets.md)). |
| `EmitOptions` | Output options: `tool_arguments`, `keep_meta`, `meta_key`, `alpaca_multiturn`, `reasoning`, `tool_content`, `meta_values`. |
| `MapSpec` | The `--from map` field mapping: `MapSpec.from_mapping({"user": "q.text", "assistant": "a.text"})`, passed as `AdapterOptions(map=...)` (see [format.md](format.md#field-mapping---from-map)). |
| `TransformOptions` | Fixes for strict chat templates: `system`, `merge_consecutive`, `split_turns`, `reasoning_turns` (see [format.md](format.md#fixes-for-strict-chat-templates)). |
| `validate_example(example)` | Reason codes that make a `TrainingExample` unfit for SFT (empty list = valid). |

```python
from pathlib import Path
from convmerge import ConvertStats, EmitOptions, convert_file

stats = ConvertStats()
convert_file(
    Path("raw.jsonl"), Path("train.jsonl"),
    adapter_name="auto", output_format="messages",
    emit_options=EmitOptions(keep_meta=["id"]),
    stats=stats, workers=4,
)
print(stats.written, stats.drop_reasons)
```

## Data model

| Name | Purpose |
|------|---------|
| `TrainingExample(messages, meta={}, tools=None, issues=[], rejected=None)` | One example between adapter and emitter. For preference data `messages` is the chosen conversation and `rejected` the rejected one (read by the `preference` format). |
| `ChatMessage(role, content, tool_calls=(), tool_call_id=None, name=None, reasoning=None)` | `content` is a string, a sequence of `ContentPart`, or `None`; `.text` gives the text-only view, `.media` the media parts. `reasoning` is a trace stored apart from the answer (inline `<think>` blocks stay in `content`). |
| `ContentPart(type, text=None, url=None)` | `text`, or `image` / `audio` / `video` by reference (`url` is a URL or path). |
| `ToolCall(name, arguments="{}", id=None)` | `arguments` is a JSON string; `ToolCall.from_any(name, dict_or_str)` builds one from a dict. |

## Other commands

| Name | Purpose |
|------|---------|
| `mix_files(sources, output, *, total, seed, oversample, sampler)` with `MixSource(path, weight)` | Weighted mixing; `sampler="v2"` streams, `"v1"` reproduces pre-0.7 mixes. Returns a `MixResult` (`total_written`, `sources`, …). |
| `deduplicate_jsonl(src, dst, *, keys, algorithm, seen_store, stats)` with `DedupeStats` | Hash-based dedupe. |
| `normalize_to_jsonl(src, dst, *, array_key="conversation")` | Messy JSON / JSONL → clean JSONL. |
| `profile_schema(path_or_records, *, max_rows, max_examples)` | The structure report behind `convmerge inspect`. |
| `analyze_turn_distribution(path)`, `split_by_turns(src, *, single_out, multi_out)` | The report and split behind `convmerge turns`. |
| `split_jsonl(src, train_out, val_out, *, val, val_rows, seed, keys, stats)` with `SplitStats` | Content-hashed train/validation split (`convmerge split`); returns `(train, val)` counts. |
| `check_tokens(path, *, tokenizer, max_tokens, output, rejects, chat_template, stats)` with `TokenStats` | Token lengths and chat-template check (`convmerge tokens`, needs `convmerge[tokens]`); `tokenizer` is a name/path or a loaded `transformers` tokenizer; `stats.to_report()` gives the JSON report. |
| `iter_jsonl(path, *, encoding, on_error, stats, on_invalid)` | The shared JSONL reader (BOM, blank, and invalid lines handled consistently). Yields `JsonlLine(number, raw, value)`; fills a `ReadStats`; `on_error="raise"` raises `JsonlDecodeError` (a `ValueError` with `path`, `line_number`). |

## Recipes

`convmerge.recipe` is public as a module: `load_recipe(path)`,
`plan(recipe, force=None)` (list of `PlannedStep` with `action` / `reason`),
`run(recipe, force=None, hf_token=None, github_token=None)` (returns
`RunResult` with `ran`, `skipped`, `report`), `RecipeError` (invalid recipe;
the message names the key), and `RecipeRunError` (a step failed). See
[recipes.md](recipes.md).

## Fetch

`convmerge.fetch` is public as a module: `load_manifest(path)` parses a
manifest ([fetch.md](fetch.md)) into a `Manifest` (`version`, `auth`,
`defaults`, `datasets`: `AuthConfig`, `TokenSpec`, `Defaults`,
`DatasetEntry`), and `run_manifest(manifest, *, output_root, only, hf_token,
github_token, log, max_rows)` fetches it, returning a `FetchResult`
(`succeeded`, `skipped`, `failed`). Its submodules are internal.

## Extending convmerge

### Adapters

An adapter is a callable taking one raw record (`dict`) and yielding
`TrainingExample` objects (yield nothing to skip the record):

```python
from convmerge import ChatMessage, TrainingExample, register_adapter

def iter_from_qa(record):
    if "q" in record and "a" in record:
        yield TrainingExample(
            messages=[ChatMessage("user", record["q"]), ChatMessage("assistant", record["a"])],
            meta={"source": "qa", "id": record.get("id")},
        )

register_adapter("qa", iter_from_qa)   # now usable as --from qa / adapter_name="qa"
```

### Output formats

An output format takes a `TrainingExample` and returns the JSON object for one
line. It may accept an `options: EmitOptions` keyword, and may raise
`UnrepresentableExample("unrepresentable_<why>")` to drop an example (it is
counted in the drop report):

```python
from convmerge import UnrepresentableExample, register_emitter

def emit_pairs(example, options=None):
    if len(example.messages) != 2:
        raise UnrepresentableExample("unrepresentable_multiturn")
    return {"prompt": example.messages[0].text, "response": example.messages[1].text}

register_emitter("pairs", emit_pairs)   # --format pairs
```

### Plugins (entry points)

To ship an adapter or format as its own package — and have it available on
the CLI and in `convert --workers` processes without any import — declare an
entry point:

```toml
# pyproject.toml of your package
[project.entry-points."convmerge.adapters"]
qa = "my_pkg.convmerge_plugin:iter_from_qa"

[project.entry-points."convmerge.emitters"]
pairs = "my_pkg.convmerge_plugin:emit_pairs"
```

Entry points load on first use of an unknown name. Built-in names always win,
and a plugin that fails to import is skipped with a warning.
`convmerge formats` lists what is available, marking plugins.

`register_adapter()` / `register_emitter()` calls made in a script are
visible to `--workers` processes only when they are started by `fork` (the
Linux default before Python 3.14); use an entry point, or register at import
time of a module the workers import, to be portable.
