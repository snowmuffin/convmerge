"""Structural validation of training examples.

``convert`` runs every example through :func:`validate_example` before it is
emitted (``on_invalid="drop"`` by default), and ``convmerge validate`` applies
the same rules to an existing JSONL file. Each problem is reported as a short
reason code so drops can be counted and explained.
"""

from __future__ import annotations

from convmerge.models import ChatMessage, TrainingExample

ALLOWED_ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool"})

# Reason codes, in the order they are checked. Adapter-reported issues
# (``TrainingExample.issues``, e.g. ``unresolved_image``) are appended as-is.
REASONS: dict[str, str] = {
    "no_messages": "the example has no messages",
    "unknown_role": "a message role is not one of system/user/assistant/tool",
    "empty_message": "a message has neither content nor tool calls",
    "no_user": "there is no user message",
    "no_assistant": "there is no assistant message with content or tool calls",
    "orphan_tool_message": "a tool message is not preceded by an assistant tool call",
    "tool_call_id_mismatch": "a tool message's tool_call_id matches no earlier tool call",
}


def validate_example(example: TrainingExample) -> list[str]:
    """Return the reason codes that make ``example`` unfit for SFT (empty = valid)."""
    msgs = example.messages
    if not msgs:
        return ["no_messages", *example.issues]

    reasons: list[str] = []
    if any(m.role not in ALLOWED_ROLES for m in msgs):
        reasons.append("unknown_role")
    if any(_is_empty(m) for m in msgs):
        reasons.append("empty_message")
    if not any(m.role == "user" for m in msgs):
        reasons.append("no_user")
    if not any(m.role == "assistant" and not _is_empty(m) for m in msgs):
        reasons.append("no_assistant")
    reasons.extend(_tool_pairing(msgs))
    reasons.extend(example.issues)
    return _dedupe(reasons)


def _is_empty(m: ChatMessage) -> bool:
    if m.tool_calls:
        return False
    if m.content is None:
        return True
    if isinstance(m.content, str):
        return not m.content.strip()
    return not (m.text.strip() or m.media)


def _tool_pairing(msgs: list[ChatMessage]) -> list[str]:
    reasons: list[str] = []
    seen_call = False
    call_ids: set[str] = set()
    for m in msgs:
        if m.role == "assistant" and m.tool_calls:
            seen_call = True
            call_ids.update(tc.id for tc in m.tool_calls if tc.id)
        elif m.role == "tool":
            if not seen_call:
                reasons.append("orphan_tool_message")
            elif m.tool_call_id is not None and call_ids and m.tool_call_id not in call_ids:
                reasons.append("tool_call_id_mismatch")
    return reasons


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
