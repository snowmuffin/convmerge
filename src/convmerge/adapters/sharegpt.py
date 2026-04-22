"""ShareGPT-style conversations (from/value) → TrainingExample."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

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


def _normalize_role(from_key: str) -> str:
    return _FROM_TO_ROLE.get(from_key.lower().strip(), from_key.lower().strip())


def iter_from_sharegpt_line(record: dict[str, Any]) -> Iterator[TrainingExample]:
    """
    One JSON object with ``conversations``: list of ``{"from": ..., "value": ...}``.

    Emits one TrainingExample per consecutive human→assistant pair (typical SFT).
    """
    convs = record.get("conversations")
    if not isinstance(convs, list) or len(convs) < 2:
        return

    i = 0
    while i + 1 < len(convs):
        a, b = convs[i], convs[i + 1]
        if not isinstance(a, dict) or not isinstance(b, dict):
            i += 1
            continue
        ra = _normalize_role(str(a.get("from", "")))
        rb = _normalize_role(str(b.get("from", "")))
        va = (a.get("value") or "").strip()
        vb = (b.get("value") or "").strip()
        if ra == "user" and rb == "assistant":
            yield TrainingExample(
                messages=[
                    ChatMessage(role="user", content=va),
                    ChatMessage(role="assistant", content=vb),
                ],
                meta={"source": "sharegpt"},
            )
            i += 2
        else:
            i += 1
