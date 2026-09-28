"""ShareGPT-style conversations (from/value) → TrainingExample."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from convmerge.adapters._common import build_example, coerce_messages, source_meta
from convmerge.models import ChatMessage, TrainingExample

# Common ShareGPT role labels
_FROM_TO_ROLE: dict[str, str] = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "bing": "assistant",
}

TURN_MODES: tuple[str, ...] = ("pairs", "full")

# Full-conversation role map: the classic labels plus LLaMA-Factory tool turns
# (``function_call`` turns are decoded into assistant tool calls separately).
_FULL_ROLE_MAP: dict[str, str] = {
    **_FROM_TO_ROLE, "observation": "tool", "tool": "tool", "function-response": "tool",
}  # fmt: skip


def _normalize_role(from_key: str) -> str:
    return _FROM_TO_ROLE.get(from_key.lower().strip(), from_key.lower().strip())


def _text(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def iter_from_sharegpt_line(
    record: dict[str, Any],
    *,
    turn_mode: str | None = None,
) -> Iterator[TrainingExample]:
    """
    One JSON object with ``conversations``: list of ``{"from": ..., "value": ...}``.

    ``turn_mode`` selects how a multi-turn conversation is emitted:

    - ``"full"`` (default; ``None`` means ``"full"``): one example holding the
      whole conversation in order. Turns with an empty value are dropped.
      LLaMA-Factory extensions are understood: ``function_call`` turns become
      assistant tool calls, ``observation`` turns become ``tool`` messages,
      and the ``system`` / ``tools`` / ``images`` (LLaVA ``image``) columns
      are attached.
    - ``"pairs"``: one example per consecutive human→assistant pair (the
      0.5.x default). System prompts, unpaired turns, tool turns, and the
      context of earlier turns are dropped.
    """
    if turn_mode not in (None, *TURN_MODES):
        raise ValueError(f"Unknown sharegpt turn_mode {turn_mode!r}. Use 'pairs' or 'full'.")

    convs = record.get("conversations")
    if not isinstance(convs, list):
        return

    if turn_mode == "pairs":
        meta = source_meta(record, {"source": "sharegpt"})
        for messages in _pair_messages(convs):
            yield TrainingExample(messages=messages, meta=dict(meta))
        return

    messages = coerce_messages(
        convs,
        role_keys=("from",),
        content_keys=("value",),
        role_map=_FULL_ROLE_MAP,
        strip=True,
    )
    if messages:
        yield build_example(messages, record, meta={"source": "sharegpt"})


def _pair_messages(convs: list[Any]) -> list[list[ChatMessage]]:
    out: list[list[ChatMessage]] = []
    i = 0
    while i + 1 < len(convs):
        a, b = convs[i], convs[i + 1]
        if not isinstance(a, dict) or not isinstance(b, dict):
            i += 1
            continue
        ra = _normalize_role(str(a.get("from", "")))
        rb = _normalize_role(str(b.get("from", "")))
        if ra == "user" and rb == "assistant":
            out.append(
                [
                    ChatMessage(role="user", content=_text(a.get("value"))),
                    ChatMessage(role="assistant", content=_text(b.get("value"))),
                ]
            )
            i += 2
        else:
            i += 1
    return out
