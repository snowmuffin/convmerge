"""Conversation fixes applied by ``convert`` before validation.

Chat templates reject or silently mangle conversations that are perfectly
valid data: Gemma-style templates have no system role, Mistral / Gemma /
Llama 2 templates require strictly alternating user and assistant turns, and
Qwen3 / gpt-oss / DeepSeek-R1 templates render a turn's reasoning only after
the last user message, so a multi-turn reasoning conversation trains on
something the model never sees at inference. :class:`TransformOptions`
selects the fixes; every fix is off by default and counted when it changes an
example.

Order: ``leading_assistant`` → ``system`` → ``merge_consecutive`` → ``split_turns`` →
``reasoning_turns``. Preference pairs get the same fixes on both sides
(``split_turns`` is not available for pairs).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from convmerge.models import ChatMessage, ContentPart, TrainingExample
from convmerge.reasoning import has_reasoning, strip_reasoning

SystemMode = Literal["keep", "fold", "drop"]
LeadingAssistant = Literal["keep", "drop"]
ReasoningTurns = Literal["all", "last"]

# Counter names in ``ConvertStats.transforms`` (and the ``convert --report`` JSON).
TRANSFORM_COUNTERS: dict[str, str] = {
    "system_folded": "system turns folded into the first user turn",
    "system_dropped": "system turns removed",
    "leading_assistant_dropped": "assistant and tool turns before the first user turn removed",
    "turns_merged": "consecutive same-role turns merged into the one before",
    "reasoning_stripped": "reasoning traces removed from turns before the last user message",
    "split_examples": "examples written by splitting conversations at user turns",
}


@dataclass(frozen=True)
class TransformOptions:
    """Fixes ``convert`` applies to each example (all off by default).

    - ``system``: ``"fold"`` moves the leading system turns into the first
      user turn (for templates without a system role, e.g. Gemma 2);
      ``"drop"`` removes every system turn; ``"keep"`` (default) does neither.
    - ``merge_consecutive``: join consecutive user turns, or consecutive
      assistant turns without tool calls, with a blank line (templates that
      require alternating roles reject them otherwise). Turns from different
      named speakers are left apart; tool turns are never merged.
    - ``split_turns``: one example per user turn, holding the conversation up
      to the next user turn; earlier assistant turns keep their answers but
      lose their reasoning (how Qwen recommends training multi-turn
      reasoning conversations). The split examples record their position as
      ``meta["turn"]``.
    - ``reasoning_turns``: ``"last"`` removes the reasoning of every assistant
      turn before the last user message (what Qwen3 / gpt-oss templates
      render); ``"all"`` (default) keeps it.
    - ``leading_assistant``: ``"drop"`` removes assistant (and tool) turns
      before the first user turn (datasets that withhold the first prompt,
      such as Nemotron chat, whose examples otherwise fail validation as
      ``withheld_prompt``; or agent data that opens with a greeting);
      ``"keep"`` (default) leaves them.
    """

    system: SystemMode = "keep"
    merge_consecutive: bool = False
    split_turns: bool = False
    reasoning_turns: ReasoningTurns = "all"
    leading_assistant: LeadingAssistant = "keep"

    def __post_init__(self) -> None:
        if self.system not in ("keep", "fold", "drop"):
            raise ValueError(f"system must be 'keep', 'fold', or 'drop', got {self.system!r}")
        if self.reasoning_turns not in ("all", "last"):
            raise ValueError(
                f"reasoning_turns must be 'all' or 'last', got {self.reasoning_turns!r}"
            )
        if self.leading_assistant not in ("keep", "drop"):
            raise ValueError(
                f"leading_assistant must be 'keep' or 'drop', got {self.leading_assistant!r}"
            )

    @property
    def active(self) -> bool:
        return self != TransformOptions()


def apply_transforms(
    example: TrainingExample,
    options: TransformOptions,
    counts: dict[str, int] | None = None,
) -> list[TrainingExample]:
    """``example`` with ``options`` applied (a list: ``split_turns`` may yield several)."""
    tally: dict[str, int] = counts if counts is not None else {}

    def fix(msgs: list[ChatMessage]) -> list[ChatMessage]:
        if options.leading_assistant == "drop":
            msgs = _drop_leading_assistant(msgs, tally)
        if options.system != "keep":
            msgs = _system(msgs, options.system, tally)
        if options.merge_consecutive:
            msgs = _merge(msgs, tally)
        return msgs

    messages = fix(example.messages)
    issues = example.issues
    if "withheld_prompt" in issues and messages is not example.messages:
        # The answer to the withheld prompt is gone; the rest is a whole conversation.
        issues = [i for i in issues if i != "withheld_prompt"]
    example = replace(
        example,
        messages=messages,
        rejected=fix(example.rejected) if example.rejected is not None else None,
        issues=issues,
    )
    if options.split_turns and example.rejected is None:
        out = _split(example)
        if len(out) > 1:
            _add(tally, "split_examples", len(out))
    else:
        out = [example]
    if options.reasoning_turns == "last":
        out = [_last_turn_reasoning(ex, tally) for ex in out]
    return out


def _drop_leading_assistant(msgs: list[ChatMessage], tally: dict[str, int]) -> list[ChatMessage]:
    first_user = next((i for i, m in enumerate(msgs) if m.role == "user"), None)
    if first_user is None:
        return msgs
    kept = [m for m in msgs[:first_user] if m.role == "system"]
    if len(kept) == first_user:
        return msgs
    _add(tally, "leading_assistant_dropped")
    return [*kept, *msgs[first_user:]]


def _add(tally: dict[str, int], key: str, n: int = 1) -> None:
    tally[key] = tally.get(key, 0) + n


def _system(msgs: list[ChatMessage], mode: SystemMode, tally: dict[str, int]) -> list[ChatMessage]:
    if not any(m.role == "system" for m in msgs):
        return msgs
    if mode == "drop":
        kept = [m for m in msgs if m.role != "system"]
        _add(tally, "system_dropped", len(msgs) - len(kept))
        return kept
    lead = 0
    while lead < len(msgs) and msgs[lead].role == "system":
        lead += 1
    rest = msgs[lead:]
    user = next((i for i, m in enumerate(rest) if m.role == "user"), None)
    text = "\n\n".join(m.text for m in msgs[:lead] if m.text.strip())
    if not lead or user is None:
        return msgs
    if text:
        rest = [*rest]
        rest[user] = _prepend_text(rest[user], text)
    _add(tally, "system_folded", lead)
    return rest


def _prepend_text(m: ChatMessage, text: str) -> ChatMessage:
    if isinstance(m.content, tuple):
        return replace(m, content=(ContentPart("text", text=text), *m.content))
    body = m.content or ""
    return replace(m, content=f"{text}\n\n{body}" if body else text)


def _merge(msgs: list[ChatMessage], tally: dict[str, int]) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for m in msgs:
        prev = out[-1] if out else None
        if prev is not None and _mergeable(prev, m):
            out[-1] = _joined(prev, m)
            _add(tally, "turns_merged")
        else:
            out.append(m)
    return out


def _mergeable(a: ChatMessage, b: ChatMessage) -> bool:
    return (
        a.role == b.role
        and a.role in ("user", "assistant", "system")
        and not a.tool_calls
        and not b.tool_calls
        and a.name == b.name
    )


def _joined(a: ChatMessage, b: ChatMessage) -> ChatMessage:
    content: str | tuple[ContentPart, ...]
    if isinstance(a.content, str | None) and isinstance(b.content, str | None):
        content = "\n\n".join(t for t in (a.content, b.content) if t)
    else:
        content = (*_parts(a), *_parts(b))
    reasoning = "\n\n".join(r for r in (a.reasoning, b.reasoning) if r) or None
    return replace(a, content=content, reasoning=reasoning)


def _parts(m: ChatMessage) -> tuple[ContentPart, ...]:
    if m.content is None or isinstance(m.content, str):
        return (ContentPart("text", text=m.content),) if m.content else ()
    return tuple(m.content)


def _split(example: TrainingExample) -> list[TrainingExample]:
    msgs = example.messages
    users = [i for i, m in enumerate(msgs) if m.role == "user"]
    if len(users) < 2:
        return [example]
    out: list[TrainingExample] = []
    for n, start in enumerate(users):
        end = users[n + 1] if n + 1 < len(users) else len(msgs)
        turn = msgs[start:end]
        if not any(m.role == "assistant" for m in turn):
            continue
        history = [strip_reasoning(m) for m in msgs[:start]]
        meta = {**example.meta, "turn": n}
        out.append(replace(example, messages=[*history, *turn], meta=meta))
    return out or [example]


def _last_turn_reasoning(example: TrainingExample, tally: dict[str, int]) -> TrainingExample:
    def strip(msgs: Sequence[ChatMessage]) -> list[ChatMessage]:
        last_user = max((i for i, m in enumerate(msgs) if m.role == "user"), default=-1)
        out: list[ChatMessage] = []
        for i, m in enumerate(msgs):
            if i < last_user and m.role == "assistant" and has_reasoning(m):
                m = strip_reasoning(m)
                _add(tally, "reasoning_stripped")
            out.append(m)
        return out

    return replace(
        example,
        messages=strip(example.messages),
        rejected=strip(example.rejected) if example.rejected is not None else None,
    )
