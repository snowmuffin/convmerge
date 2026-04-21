"""Directory-level schema unification for chat / alpaca JSONL trees.

The row-level primitives live in :mod:`convmerge.adapters.chat` and
:mod:`convmerge.adapters.alpaca`. This module wraps them into
"walk a directory, rewrite every file in place (safely), preserve the tree"
operations that dataset preparation pipelines typically need, without
imposing any particular hard-coded schema.

All helpers support the "same source and target directory" pattern by
writing into a temp directory first and moving files back after all
sources are processed — so a crashed run cannot leave partially overwritten
files.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from convmerge.adapters.alpaca import (
    DEFAULT_INPUT_KEYS,
    DEFAULT_INSTRUCTION_KEYS,
    DEFAULT_OUTPUT_KEYS,
    remap_to_alpaca,
)
from convmerge.adapters.chat import iter_from_chat_line

# Common keys that carry moderation / provenance metadata on public chat
# datasets (LMSYS, WildChat, Chatbot Arena, …). Passed to
# :func:`unify_messages_dir` by default so users can override per project.
DEFAULT_SFT_STRIP_KEYS: frozenset[str] = frozenset(
    {
        "openai_moderation",
        "detoxify_moderation",
        "conversation_id",
        "model",
        "turn",
        "language",
        "redacted",
        "timestamp",
        "toxic",
        "model_a",
        "model_b",
        "winner",
        "question_id",
        "conversation_a",
        "conversation_b",
        "conversation",
        "conversations",
        "text",
        "user_id",
        "anonymized_user_id",
        "uid",
        "tstamp",
        "judge",
        "anony",
        "toxic_chat_tag",
    }
)

# Default mapping for tagged-text chat dumps (``<sys>...\n<usr>...\n<bot>...``).
DEFAULT_TAG_MAP: dict[str, str] = {
    "sys": "system",
    "usr": "user",
    "bot": "assistant",
}

RowShape = Literal["chat", "pairwise", "alpaca", "plain_text", "unknown"]


def classify_row_shape(
    row: dict[str, Any],
    *,
    instruction_keys: Iterable[str] = DEFAULT_INSTRUCTION_KEYS,
    output_keys: Iterable[str] = DEFAULT_OUTPUT_KEYS,
) -> RowShape:
    """Best-effort classification of a single JSON row into a schema family.

    The result is a loose signal, not a guarantee:

    - ``"pairwise"`` : ``conversation_a`` / ``conversation_b`` present.
    - ``"chat"``     : any of ``messages`` / ``conversation`` / ``conversations``
      is a non-empty list.
    - ``"plain_text"`` : a non-empty string on the ``text`` key.
    - ``"alpaca"``   : at least one instruction-key and one output-key hit.
    - ``"unknown"``  : none of the above.
    """
    if not isinstance(row, dict):
        return "unknown"
    if row.get("conversation_a") is not None and row.get("conversation_b") is not None:
        return "pairwise"
    for k in ("messages", "conversation", "conversations"):
        v = row.get(k)
        if isinstance(v, list) and v:
            return "chat"
    txt = row.get("text")
    if isinstance(txt, str) and txt.strip():
        return "plain_text"
    if any(
        isinstance(row.get(k), str) and row.get(k, "").strip() for k in instruction_keys
    ) and any(isinstance(row.get(k), str) and row.get(k, "").strip() for k in output_keys):
        return "alpaca"
    return "unknown"


def parse_tagged_text(
    text: str,
    *,
    tag_map: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Parse ``<tag>...<tag>...`` strings into a list of ``{role, content}``.

    ``tag_map`` maps the literal tag token (e.g. ``"usr"``) to the role that
    should end up in the returned dicts (e.g. ``"user"``). Tags are matched
    case-insensitively; unknown tags are ignored.
    """
    mapping = dict(tag_map or DEFAULT_TAG_MAP)
    if not mapping:
        return []
    alt = "|".join(re.escape(t) for t in mapping)
    pattern = re.compile(rf"<\s*({alt})\s*>", re.IGNORECASE)

    messages: list[dict[str, str]] = []
    current_role: str | None = None
    pos = 0
    for match in pattern.finditer(text):
        tag = match.group(1).lower()
        if current_role is not None:
            chunk = text[pos : match.start()].strip()
            if chunk:
                messages.append({"role": current_role, "content": chunk})
        current_role = mapping.get(tag)
        pos = match.end()
    if current_role is not None:
        chunk = text[pos:].strip()
        if chunk:
            messages.append({"role": current_role, "content": chunk})
    return messages


def unify_messages_dir(
    src_dir: str | Path,
    dst_dir: str | Path,
    *,
    adapter_kwargs: dict[str, Any] | None = None,
    sft_strip_metadata: bool = True,
    sft_strip_keys: Iterable[str] = DEFAULT_SFT_STRIP_KEYS,
    emit_meta: bool = False,
) -> tuple[int, int]:
    """Rewrite every JSONL under ``src_dir`` into a uniform ``{"messages": [...]}`` shape.

    Row-level logic is delegated to :func:`convmerge.adapters.chat.iter_from_chat_line`,
    so pairwise Arena rows, ``conversation`` / ``conversations`` / ``text``,
    and alpaca fall-back are all handled identically to the ``chat`` adapter.
    ``adapter_kwargs`` is forwarded to that function (e.g. ``pairwise_mode``,
    ``role_map``, custom ``conversation_keys``).

    When ``sft_strip_metadata=True``, any top-level key that is in
    ``sft_strip_keys`` is stripped on the way out (except ``messages``
    itself). Supply your own set via ``sft_strip_keys`` to customise.

    Supports in-place rewrites: when ``src_dir`` and ``dst_dir`` resolve to
    the same absolute path, writes go through a temp directory first and are
    moved back atomically at the end.

    Returns ``(n_files_written, n_rows_written)``.
    """
    adapter_kwargs = dict(adapter_kwargs or {})
    strip_set = set(sft_strip_keys)

    def _write_row(fout, raw_line: str) -> int:
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            return 0
        if not isinstance(obj, dict):
            return 0
        n = 0
        for example in iter_from_chat_line(obj, **adapter_kwargs):
            out: dict[str, Any] = {
                "messages": [{"role": m.role, "content": m.content} for m in example.messages]
            }
            if emit_meta and example.meta:
                out["meta"] = dict(example.meta)
            if sft_strip_metadata:
                # Propagate non-stripped top-level scalars (e.g. ``id``, ``source``)
                # that the caller hasn't listed for removal.
                for k, v in obj.items():
                    if k == "messages" or k in strip_set:
                        continue
                    if k in out:
                        continue
                    out[k] = v
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
        return n

    return _sweep_jsonl_tree(src_dir, dst_dir, _write_row)


def unify_message_entries(
    src_dir: str | Path,
    dst_dir: str | Path,
    *,
    list_key: str = "messages",
    source_keys: Iterable[str] = ("from",),
    target_key: str = "role",
    drop_keys: Iterable[str] = (),
    require_nonempty_str: bool = True,
) -> tuple[int, int]:
    """Normalize the fields inside each entry of a chat-list.

    For every dict inside ``row[list_key]`` (default ``messages``):

    - Remove any key in ``drop_keys``.
    - Take the value of the first present key in ``source_keys`` and store it
      under ``target_key`` (e.g. ``from`` -> ``role`` or ``value`` -> ``content``).
    - Entries where the chosen value is missing or empty are dropped.

    Rows without the ``list_key`` pass through unchanged. Relative paths are
    preserved and in-place rewrites are supported (temp-dir swap).

    Returns ``(n_files_written, n_rows_written)``.
    """
    src_keys_t = tuple(source_keys)
    drop_set = set(drop_keys)

    def _write_row(fout, raw_line: str) -> int:
        try:
            data: dict[str, Any] = json.loads(raw_line)
        except json.JSONDecodeError:
            return 0
        if not isinstance(data, dict):
            return 0

        items = data.get(list_key)
        if not isinstance(items, list):
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            return 1

        new_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            msg = dict(item)
            for k in drop_set:
                msg.pop(k, None)
            chosen: Any = None
            for sk in src_keys_t:
                if sk in msg:
                    chosen = msg.pop(sk)
                    break
            if chosen is None:
                continue
            if require_nonempty_str and isinstance(chosen, str) and not chosen.strip():
                continue
            msg[target_key] = chosen
            new_items.append(msg)
        if new_items:
            data[list_key] = new_items
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
        return 1

    return _sweep_jsonl_tree(src_dir, dst_dir, _write_row)


def unify_alpaca_dir(
    src_dir: str | Path,
    dst_dir: str | Path,
    *,
    instruction_keys: Iterable[str] = DEFAULT_INSTRUCTION_KEYS,
    output_keys: Iterable[str] = DEFAULT_OUTPUT_KEYS,
    input_keys: Iterable[str] = DEFAULT_INPUT_KEYS,
    drop_keys: Iterable[str] = (),
) -> tuple[int, int]:
    """Reshape single-turn JSONL trees into the alpaca ``{instruction, input, output}`` form.

    Row-level logic is delegated to
    :func:`convmerge.adapters.alpaca.remap_to_alpaca`; this function provides
    the same in-place-safe directory sweep as :func:`unify_messages_dir`.
    Rows that cannot produce both instruction and output are dropped.

    Returns ``(n_files_written, n_rows_written)``.
    """
    instr_t = tuple(instruction_keys)
    out_t = tuple(output_keys)
    inp_t = tuple(input_keys)
    drop_t = tuple(drop_keys)

    def _write_row(fout, raw_line: str) -> int:
        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError:
            return 0
        if not isinstance(data, dict):
            return 0
        norm = remap_to_alpaca(
            data,
            instruction_keys=instr_t,
            output_keys=out_t,
            input_keys=inp_t,
            drop_keys=drop_t,
        )
        if norm is None:
            return 0
        fout.write(json.dumps(norm, ensure_ascii=False) + "\n")
        return 1

    return _sweep_jsonl_tree(src_dir, dst_dir, _write_row)


def _sweep_jsonl_tree(
    src_dir: str | Path,
    dst_dir: str | Path,
    write_row,
) -> tuple[int, int]:
    """Shared driver for the ``unify_*_dir`` helpers.

    Walks ``src_dir`` recursively, opens each ``.jsonl`` file for reading,
    and passes every raw line to ``write_row(fout, raw_line) -> int``. The
    callback is expected to write zero or more output rows into ``fout`` and
    return how many were written (used only for stats).

    Handles the "src == dst" case by writing to a temp directory first.
    """
    src_p = Path(src_dir).resolve()
    dst_p = Path(dst_dir).resolve()
    if not src_p.is_dir():
        raise NotADirectoryError(f"Not a directory: {src_p}")

    in_place = src_p == dst_p
    if in_place:
        temp_root = Path(tempfile.mkdtemp(prefix="convmerge_reshape_"))
        actual_dst = temp_root
    else:
        temp_root = None
        actual_dst = dst_p
        actual_dst.mkdir(parents=True, exist_ok=True)

    try:
        n_files = 0
        n_rows = 0
        for in_path in sorted(src_p.rglob("*.jsonl")):
            if not in_path.is_file():
                continue
            rel = in_path.relative_to(src_p)
            out_path = actual_dst / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            rows_here = 0
            with in_path.open(encoding="utf-8") as rf, out_path.open("w", encoding="utf-8") as wf:
                for raw in rf:
                    line = raw.strip()
                    if not line:
                        continue
                    rows_here += write_row(wf, line)
            if rows_here:
                n_files += 1
                n_rows += rows_here
            else:
                # Drop empty output files so the target tree stays clean.
                try:
                    out_path.unlink()
                except OSError:
                    pass

        if in_place and temp_root is not None:
            for p in temp_root.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(temp_root)
                final_path = dst_p / rel
                final_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(final_path))
        return n_files, n_rows
    finally:
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)
