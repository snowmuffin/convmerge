"""LLaMA-Factory ``dataset_info.json`` entries for convmerge output files.

LLaMA-Factory only reads a dataset it finds in ``dataset_info.json``, with
the column and role names spelled out. :func:`dataset_info_entry` scans a
JSONL file written by ``convert`` (``sharegpt``, ``sharegpt-preference``,
``alpaca``, or ``messages`` without tool calls) and returns that entry;
:func:`update_dataset_info` merges it into an existing file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from convmerge.io import iter_jsonl

_SHAREGPT_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
    "system_tag": "system",
}
_OPENAI_TAGS = {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system",
}
_OPTIONAL = ("system", "tools", "images", "videos", "audios")


def dataset_info_entry(path: str | Path, *, file_name: str | None = None) -> dict[str, Any]:
    """The ``dataset_info.json`` entry describing ``path``.

    ``file_name`` is what LLaMA-Factory resolves against its ``dataset_dir``
    (default: the file's name). Raises ``ValueError`` for files LLaMA-Factory
    cannot read as-is, saying which ``--format`` to convert to instead.
    """
    keys: set[str] = set()
    first: dict[str, Any] | None = None
    has_tool_calls = False
    for line in iter_jsonl(path):
        if not isinstance(line.value, dict):
            continue
        first = first or line.value
        keys.update(line.value)
        if not has_tool_calls and "messages" in line.value:
            has_tool_calls = any(
                isinstance(m, dict) and m.get("tool_calls") for m in line.value["messages"] or []
            )
    if first is None:
        raise ValueError(f"{path}: no JSON object rows")

    entry: dict[str, Any] = {"file_name": file_name or Path(path).name}
    if "conversations" in keys:
        entry["formatting"] = "sharegpt"
        columns: dict[str, str] = {"messages": "conversations"}
        if "chosen" in keys and "rejected" in keys:
            entry["ranking"] = True
            columns.update(chosen="chosen", rejected="rejected")
        columns.update({k: k for k in _OPTIONAL if k in keys})
        entry["columns"] = columns
        entry["tags"] = dict(_SHAREGPT_TAGS)
    elif "prompt" in keys and "chosen" in keys and "rejected" in keys:
        raise ValueError(
            f"{path}: TRL preference rows; LLaMA-Factory reads pairs written with "
            "--format sharegpt-preference"
        )
    elif "messages" in keys:
        if has_tool_calls or "tools" in keys:
            raise ValueError(
                f"{path}: LLaMA-Factory cannot read OpenAI tool_calls; convert with "
                "--format sharegpt instead"
            )
        entry["formatting"] = "sharegpt"
        entry["columns"] = {"messages": "messages"}
        entry["tags"] = dict(_OPENAI_TAGS)
    elif "instruction" in keys and "output" in keys:
        columns = {"prompt": "instruction", "query": "input", "response": "output"}
        columns.update({k: k for k in ("system", "history") if k in keys})
        entry["columns"] = columns  # formatting defaults to "alpaca"
    else:
        raise ValueError(f"{path}: not a sharegpt, alpaca, or messages file")
    return entry


def update_dataset_info(info_path: str | Path, name: str, entry: dict[str, Any]) -> bool:
    """Add or replace ``name`` in ``info_path`` (created if missing); True if it changed."""
    info = Path(info_path)
    data: dict[str, Any] = {}
    if info.is_file():
        data = json.loads(info.read_text(encoding="utf-8") or "{}")
        if not isinstance(data, dict):
            raise ValueError(f"{info}: expected a JSON object")
    changed = data.get(name) != entry
    data[name] = entry
    info.parent.mkdir(parents=True, exist_ok=True)
    info.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def relative_file_name(data_path: str | Path, info_path: str | Path) -> str:
    """``data_path`` as LLaMA-Factory resolves it from the directory of ``info_path``."""
    target, base = Path(data_path).resolve(), Path(info_path).resolve().parent
    return Path(os.path.relpath(target, base)).as_posix()
