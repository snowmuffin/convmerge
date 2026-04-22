# Convert presets and team-specific rules

Presets are small YAML (or JSON) files that pin `adapter`, `output_format`, and optional `adapter_options.chat` tuning for the auto-detecting chat adapter.

## When to use what

1. **Only field names differ** (`from` / `value`, `conversation` vs `messages`): use `adapter_options.chat` in a preset (key lists, `role_map`, `pairwise_mode`).
2. **You need a repeatable team default**: commit a preset next to your manifest and call `convmerge convert --preset your_team.yaml`.
3. **YAML needs PyYAML**: `pip install "convmerge[preset]"`.
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
```

CLI flags `--from`, `--format`, and `--adapter-kwargs` override the preset when provided.

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
