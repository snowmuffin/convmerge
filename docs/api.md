# Python API

Everything below is importable from the top-level package
(`from convmerge import convert_file`) and is the **public API**: these names
keep working across minor versions, and changes are announced in the
changelog before they happen. Modules and names that are not listed here, or
that start with `_`, are internal and may change without notice. 1.0 will
freeze this list.

## Convert pipeline

| Name | Purpose |
|------|---------|
| `convert_file(input, output, *, adapter_name, output_format, ...)` | Read JSONL → adapter → validate → emit. Returns `(lines_read, lines_written)`. Options: `encoding`, `adapter_options`, `progress`, `stats`, `on_invalid` (`drop`/`keep`/`fail`), `emit_options`, `workers`. |
| `convert_with_config(input, output, cfg, ...)` | Same, from a resolved `ConvertConfig`. |
| `build_convert_config(*, preset_path, adapter, output_format, ...)` | Merge a preset file, `--adapter-kwargs` JSON, and explicit overrides into a `ConvertConfig`. |
| `validate_file(input, *, adapter_name="chat")` | Run validation only; returns `ConvertStats`. |
| `ConvertStats` | Counters and drop reasons; pass one as `stats=` and read it afterwards. `to_report()` gives the `--report` JSON. |
| `InvalidExampleError` | Raised by `on_invalid="fail"`; has `line_number` and `reasons`. |
| `ConvertConfig`, `AdapterOptions`, `ChatAdapterOptions`, `SharegptAdapterOptions` | Adapter configuration (see [custom_presets.md](custom_presets.md)). |
| `EmitOptions` | Output options: `tool_arguments`, `keep_meta`, `meta_key`, `alpaca_multiturn`. |
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
| `TrainingExample(messages, meta={}, tools=None, issues=[])` | One example between adapter and emitter. |
| `ChatMessage(role, content, tool_calls=(), tool_call_id=None, name=None)` | `content` is a string, a sequence of `ContentPart`, or `None`; `.text` gives the text-only view, `.media` the media parts. |
| `ContentPart(type, text=None, url=None)` | `text`, or `image` / `audio` / `video` by reference (`url` is a URL or path). |
| `ToolCall(name, arguments="{}", id=None)` | `arguments` is a JSON string; `ToolCall.from_any(name, dict_or_str)` builds one from a dict. |

## Other commands

| Name | Purpose |
|------|---------|
| `mix_files(sources, output, *, total, seed, oversample, sampler)` with `MixSource(path, weight)` | Weighted mixing; `sampler="v2"` streams, `"v1"` reproduces pre-0.7 mixes. |
| `deduplicate_jsonl(src, dst, *, keys, algorithm, seen_store, stats)` with `DedupeStats` | Hash-based dedupe. |
| `normalize_to_jsonl(src, dst, *, array_key="conversation")` | Messy JSON / JSONL → clean JSONL. |
| `profile_schema(path_or_records, *, max_rows, max_examples)` | The structure report behind `convmerge inspect`. |
| `iter_jsonl(path, *, on_error, stats, on_invalid)` | The shared JSONL reader (BOM, blank, and invalid lines handled consistently). |

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
