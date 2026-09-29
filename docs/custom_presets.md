# Convert presets and team-specific rules

Presets are small YAML (or JSON) files that pin `adapter`, `output_format`, and optional `adapter_options.chat` tuning for the auto-detecting chat adapter.

## When to use what

1. **Only field names differ** (`from` / `value`, `conversation` vs `messages`): use `adapter_options.chat` in a preset (key lists, `role_map`, `pairwise_mode`).
2. **You need a repeatable team default**: commit a preset next to your manifest and call `convmerge convert --preset your_team.yaml`.
3. **YAML needs PyYAML**: `pip install "convmerge[preset]"` (or
   `pip install "convmerge[all]"`).
4. **Rules need custom code** (complex winner logic, non-JSON shapes): add a Python adapter in your project or upstream a patch; presets alone are declarative.

## v1 schema

```yaml
adapter: chat              # alpaca | sharegpt | chat | auto
output_format: messages    # messages | alpaca
encoding: utf-8            # optional

adapter_options:
  chat:
    pairwise_mode: winner    # winner | both | a | b
    # conversation_keys: [messages, conversation, conversations]
    # role_keys: [role, from]
    # content_keys: [content, value, text]
    # role_map: { human: user, gpt: assistant }
  map:                       # used when adapter: map (dotted paths, see format.md)
    user: question.text
    assistant: answer.text
  sharegpt:                  # used when adapter: sharegpt
    turn_mode: full          # full (default) | pairs (0.5.x behavior)

output_options:              # optional; CLI flags override each key
  tool_arguments: string     # string (OpenAI) | object
  keep_meta: false           # true, or a list such as [source, id]
  meta_key: meta
  alpaca_multiturn: flatten  # flatten | history | drop
  reasoning: keep            # keep | inline | reasoning_content | thinking | drop
  tool_content: empty        # empty ("") | null
  train_turns: all           # all | last ("train": false on earlier answers)
  meta:                      # constant fields under meta on every row
    dataset: my_dataset

transforms:                  # optional; fixes for strict chat templates
  system: keep               # keep | fold | drop
  merge_consecutive: false
  split_turns: false
  reasoning_turns: all       # all | last
  leading_assistant: keep    # keep | drop
```

CLI flags `--from`, `--format`, `--adapter-kwargs`, `--tool-arguments`, `--keep-meta`,
`--meta-key`, `--alpaca-multiturn`, `--reasoning`, `--tool-content`, `--train-turns`,
`--system`, `--merge-consecutive`, `--split-turns`, `--reasoning-turns`, and
`--leading-assistant` override the preset
when provided.

## Commands

```bash
convmerge preset init -o convert_preset.yaml
convmerge preset validate convert_preset.yaml
convmerge convert -i in.jsonl -o out.jsonl --preset convert_preset.yaml
```

## Python API

```python
from pathlib import Path
from convmerge.config import build_convert_config
from convmerge.convert import convert_with_config

cfg = build_convert_config(preset_path=Path("convert_preset.yaml"))
convert_with_config(Path("in.jsonl"), Path("out.jsonl"), cfg)
```
