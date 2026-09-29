"""axolotl dataset configuration for convmerge output files.

axolotl reads each dataset through a ``type`` (prompt strategy) whose field
names must match the file; a mismatch is often silent (it trains on empty
turns). :func:`dataset_config` scans a JSONL file written by ``convert``
and returns the matching ``datasets:`` entry; :func:`render_config` turns
entries into a YAML snippet for an axolotl config.

- ``messages`` → ``type: chat_template`` (tools, and ``reasoning_content`` or
  ``thinking`` traces, are read by axolotl as they are; per-turn ``train``
  flags from ``--train-turns last`` add ``message_field_training: train``)
- ``sharegpt`` → ``chat_template`` with ``from`` / ``value`` mappings and the
  ``system`` column
- ``preference`` / ``sharegpt-preference`` → ``chat_template.default`` for
  ``rl: dpo`` (one answer per side: axolotl reads only the last message)
- ``alpaca`` → ``type: alpaca``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from convmerge.io import iter_jsonl

_SHAREGPT_ROLES = {
    "user": ["human", "user"],
    "assistant": ["gpt", "assistant"],
    "system": ["system"],
    "tool": ["observation", "tool"],
}


def dataset_config(path: str | Path, *, file_path: str | None = None) -> dict[str, Any]:
    """The axolotl ``datasets:`` entry for ``path`` (``rl: dpo`` pairs carry ``_rl``).

    ``file_path`` is the ``path`` to record (default: ``path`` as given).
    Raises ``ValueError`` for files axolotl cannot read as they are, saying
    which ``--format`` to convert to instead.
    """
    keys: set[str] = set()
    thinking = function_calls = multi_answer = train_flags = False
    rows = 0
    for line in iter_jsonl(path):
        value = line.value
        if not isinstance(value, dict):
            continue
        rows += 1
        keys.update(value)
        for turn in _turns(value):
            thinking = thinking or isinstance(turn.get("thinking"), str)
            train_flags = train_flags or isinstance(turn.get("train"), bool)
            function_calls = function_calls or turn.get("from") == "function_call"
        for side in ("chosen", "rejected"):
            answer = value.get(side)
            if isinstance(answer, list) and len(answer) != 1:
                multi_answer = True
    if not rows:
        raise ValueError(f"{path}: no JSON object rows")

    entry: dict[str, Any] = {"path": file_path or str(path), "ds_type": "json"}
    if "chosen" in keys and "rejected" in keys:
        if multi_answer:
            raise ValueError(
                f"{path}: some pairs continue with more than one message after the prompt; "
                "axolotl's chat_template.default reads only the last one"
            )
        prompt_field = "prompt" if "prompt" in keys else "conversations"
        entry.update(type="chat_template.default", field_messages=prompt_field,
                     field_chosen="chosen", field_rejected="rejected")  # fmt: skip
        if prompt_field == "conversations":
            entry["message_property_mappings"] = {"role": "from", "content": "value"}
            entry["roles"] = _SHAREGPT_ROLES
        if "system" in keys:
            entry["field_system"] = "system"
        entry["_rl"] = "dpo"
    elif "messages" in keys:
        entry.update(type="chat_template", field_messages="messages",
                     roles_to_train=["assistant"])  # fmt: skip
        if thinking:
            entry.update(field_thinking="thinking", template_thinking_key="thinking")
        if train_flags:
            # convert --train-turns last: "train": false on earlier answers.
            entry["message_field_training"] = "train"
    elif "conversations" in keys:
        if function_calls:
            raise ValueError(
                f"{path}: axolotl cannot read LLaMA-Factory function_call / observation "
                "turns; convert with --format messages instead"
            )
        entry.update(type="chat_template", field_messages="conversations",
                     message_property_mappings={"role": "from", "content": "value"},
                     roles=_SHAREGPT_ROLES, roles_to_train=["assistant"])  # fmt: skip
        if "system" in keys:
            entry["field_system"] = "system"
    elif "instruction" in keys and "output" in keys:
        entry["type"] = "alpaca"
    else:
        raise ValueError(f"{path}: not a messages, sharegpt, preference, or alpaca file")
    return entry


def _turns(value: dict[str, Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for key in ("messages", "conversations", "prompt", "chosen", "rejected"):
        v = value.get(key)
        if isinstance(v, list):
            turns.extend(t for t in v if isinstance(t, dict))
        elif isinstance(v, dict):
            turns.append(v)
    return turns


def render_config(
    train: dict[str, Any], val: dict[str, Any] | None = None, *, source: str = ""
) -> str:
    """A YAML snippet with ``datasets:`` (and ``test_datasets:``, ``rl:``) for axolotl."""
    rl = train.get("_rl")
    lines = [f"# axolotl datasets for {source} (written by convmerge axolotl-config)"]
    if rl:
        lines.append(f"rl: {rl}")
    lines.append("datasets:")
    lines += _entry(train)
    if val is not None:
        lines.append("test_datasets:")
        lines += _entry({**val, "split": "train"})
    return "\n".join(lines) + "\n"


def _entry(entry: dict[str, Any]) -> list[str]:
    out: list[str] = []
    items = [(k, v) for k, v in entry.items() if not k.startswith("_")]
    for i, (key, value) in enumerate(items):
        prefix = "  - " if i == 0 else "    "
        if isinstance(value, dict):
            out.append(f"{prefix}{key}:")
            out += [f"      {k}: {_scalar(v)}" for k, v in value.items()]
        else:
            out.append(f"{prefix}{key}: {_scalar(value)}")
    return out


def _scalar(value: Any) -> str:
    # JSON strings, lists, and booleans are valid YAML flow scalars.
    return json.dumps(value, ensure_ascii=False)
