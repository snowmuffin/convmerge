"""Auto-detecting chat adapter.

Routes a raw record to the right internal shape by looking at which keys are
present. Handles the common messy shapes seen across SFT datasets:

- ``messages`` / ``conversation`` / ``conversations`` lists with
  ``{role, content}`` or ``{from, value}`` entries.
- Pairwise preference rows (``conversation_a`` / ``conversation_b``), with an
  optional ``winner`` field; emits only the winner branch by default.
- Plain ``text`` strings (yielded as a single assistant message).
- Alpaca-style ``instruction`` / ``input`` / ``output`` (delegates to the
  existing alpaca adapter).

Users can override the key lists and role map to teach it about bespoke schemas
without writing a new adapter from scratch.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from convmerge.adapters.alpaca import iter_from_alpaca_line
from convmerge.models import ChatMessage, TrainingExample

# Default mapping from common ShareGPT-style ``from`` values onto standard roles.
DEFAULT_ROLE_MAP: dict[str, str] = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bing": "assistant",
    "bot": "assistant",
    "system": "system",
}

# Keys searched for the chat-list container, in priority order.
DEFAULT_CONVERSATION_KEYS: tuple[str, ...] = ("messages", "conversation", "conversations")

# Keys treated as role labels inside a chat-list entry.
DEFAULT_ROLE_KEYS: tuple[str, ...] = ("role", "from")

# Keys treated as message content inside a chat-list entry.
DEFAULT_CONTENT_KEYS: tuple[str, ...] = ("content", "value", "text")


def iter_from_chat_line(
    record: dict[str, Any],
    *,
    conversation_keys: tuple[str, ...] = DEFAULT_CONVERSATION_KEYS,
    role_keys: tuple[str, ...] = DEFAULT_ROLE_KEYS,
    content_keys: tuple[str, ...] = DEFAULT_CONTENT_KEYS,
    role_map: dict[str, str] | None = None,
    pairwise_mode: str = "winner",
    instruction_keys: tuple[str, ...] = ("instruction", "question", "prompt"),
    output_keys: tuple[str, ...] = ("output", "response", "answer"),
    input_keys: tuple[str, ...] = ("input", "context"),
) -> Iterator[TrainingExample]:
    """Yield zero or more :class:`TrainingExample` from a single raw record.

    ``pairwise_mode`` controls how ``conversation_a`` / ``conversation_b`` rows
    are handled:

    - ``"winner"`` (default): emit only the branch named by the ``winner`` field;
      emit nothing when ``winner`` is absent or unrecognised.
    - ``"both"``: emit both branches as independent examples.
    - ``"a"`` / ``"b"``: always emit the chosen branch.
    """
    role_map = role_map or DEFAULT_ROLE_MAP

    if "conversation_a" in record and "conversation_b" in record:
        yield from _iter_pairwise(
            record,
            role_keys=role_keys,
            content_keys=content_keys,
            role_map=role_map,
            pairwise_mode=pairwise_mode,
        )
        return

    for key in conversation_keys:
        convs = record.get(key)
        if isinstance(convs, list) and convs:
            msgs = _coerce_messages(
                convs, role_keys=role_keys, content_keys=content_keys, role_map=role_map
            )
            if msgs:
                yield TrainingExample(messages=msgs, meta={"source": "chat"})
            return

    txt = record.get("text")
    if isinstance(txt, str) and txt.strip():
        yield TrainingExample(
            messages=[ChatMessage(role="assistant", content=txt.strip())],
            meta={"source": "chat:text"},
        )
        return

    # Fall back to the alpaca adapter, but let callers override the key priority.
    remapped = _remap_for_alpaca(record, instruction_keys, input_keys, output_keys)
    if remapped is not None:
        yield from iter_from_alpaca_line(remapped)


def _iter_pairwise(
    record: dict[str, Any],
    *,
    role_keys: tuple[str, ...],
    content_keys: tuple[str, ...],
    role_map: dict[str, str],
    pairwise_mode: str,
) -> Iterator[TrainingExample]:
    a = record.get("conversation_a")
    b = record.get("conversation_b")
    winner = str(record.get("winner") or "").lower().strip()

    branches: list[tuple[str, Any]] = []
    if pairwise_mode == "both":
        branches = [("a", a), ("b", b)]
    elif pairwise_mode == "a":
        branches = [("a", a)]
    elif pairwise_mode == "b":
        branches = [("b", b)]
    elif pairwise_mode == "winner":
        if winner in ("model_a", "a"):
            branches = [("a", a)]
        elif winner in ("model_b", "b"):
            branches = [("b", b)]
        # Tie / unknown: emit nothing.
    else:
        raise ValueError(
            f"Unknown pairwise_mode {pairwise_mode!r}. Use 'winner', 'both', 'a', or 'b'."
        )

    for label, convs in branches:
        if not isinstance(convs, list) or not convs:
            continue
        msgs = _coerce_messages(
            convs, role_keys=role_keys, content_keys=content_keys, role_map=role_map
        )
        if msgs:
            yield TrainingExample(
                messages=msgs,
                meta={"source": "chat:pairwise", "branch": label},
            )


def _coerce_messages(
    convs: list[Any],
    *,
    role_keys: tuple[str, ...],
    content_keys: tuple[str, ...],
    role_map: dict[str, str],
) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for item in convs:
        if not isinstance(item, dict):
            continue
        role_raw: str | None = None
        for rk in role_keys:
            v = item.get(rk)
            if isinstance(v, str) and v.strip():
                role_raw = v.strip().lower()
                break
        if role_raw is None:
            continue
        role = role_map.get(role_raw, role_raw)

        content: str | None = None
        for ck in content_keys:
            v = item.get(ck)
            if isinstance(v, str):
                content = v
                break
        if content is None:
            continue
        out.append(ChatMessage(role=role, content=content))
    return out


def _remap_for_alpaca(
    record: dict[str, Any],
    instruction_keys: tuple[str, ...],
    input_keys: tuple[str, ...],
    output_keys: tuple[str, ...],
) -> dict[str, Any] | None:
    """Pick the first matching key for each slot and return a standard alpaca row."""
    instr = _first_string(record, instruction_keys)
    out = _first_string(record, output_keys)
    if instr is None and out is None:
        return None
    inp = _first_string(record, input_keys) or ""
    return {
        "instruction": instr or "",
        "input": inp,
        "output": out or "",
    }


def _first_string(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None
