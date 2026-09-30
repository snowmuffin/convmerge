"""Auto-detecting chat adapter.

Routes a raw record to the right internal shape by looking at which keys are
present. Handles the common messy shapes seen across SFT datasets:

- ``messages`` / ``conversation`` / ``conversations`` lists with
  ``{role, content}`` or ``{from, value}`` entries, or Capybara-style
  ``{input, output}`` turn pairs.
- Pairwise preference rows (``conversation_a`` / ``conversation_b``), with an
  optional ``winner`` field; emits only the winner branch by default.
- A ``text`` string rendered with a known chat template (ChatML, Llama 2/3,
  Gemma, Guanaco, HH-RLHF, the Alpaca prompt) is split back into turns;
  any other ``text`` is yielded as a single assistant message.
- Alpaca-style ``instruction`` / ``input`` / ``output`` (delegates to the
  existing alpaca adapter), and ``input`` turns + an ``output`` answer
  (Llama-Nemotron post-training data).
- OpenAssistant message trees (a ``prompt`` with nested ``replies``): the
  conversation along the best-ranked reply (see :mod:`convmerge.adapters.oasst`).
- Tool-calling encodings other than OpenAI's: Hermes tags, Glaive
  ``system`` + ``chat`` transcripts, and xLAM ``query`` / ``answers`` (see
  :mod:`convmerge.adapters.tool_formats`).

Around the turns it also keeps OpenAI content parts (text + media by
reference), ``tool_calls`` / ``tool_call_id`` / ``name``, LLaMA-Factory
``function_call`` / ``observation`` turns, the ``tools`` / ``system`` /
``images`` (``videos``, ``audios``, LLaVA ``image``) columns, and reasoning
traces kept apart from the answer (turn keys ``reasoning_content`` /
``thinking`` / ``reasoning``; for flat records the columns named in
``record_reasoning_keys``).

Users can override the key lists and role map to teach it about bespoke schemas
without writing a new adapter from scratch.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from convmerge.adapters import oasst
from convmerge.adapters._common import (
    DEFAULT_REASONING_KEYS,
    build_example,
    coerce_messages,
    first_text,
    parse_tool_calls,
    source_meta,
)
from convmerge.adapters.alpaca import iter_from_alpaca_line
from convmerge.adapters.text_chat import parse_text_chat
from convmerge.adapters.tool_formats import glaive_messages, is_glaive, is_xlam, xlam_messages
from convmerge.models import ChatMessage, TrainingExample

logger = logging.getLogger(__name__)

# Default mapping from common ShareGPT-style ``from`` values onto standard roles.
DEFAULT_ROLE_MAP: dict[str, str] = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bing": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
    "input": "system",  # heegyu/open-korean-instructions-v20231020: the system prompt
    "tool": "tool",
    "function": "tool",
    "function-response": "tool",
    "observation": "tool",
}

# Keys searched for the chat-list container, in priority order.
DEFAULT_CONVERSATION_KEYS: tuple[str, ...] = ("messages", "conversation", "conversations")

# Keys treated as role labels inside a chat-list entry.
DEFAULT_ROLE_KEYS: tuple[str, ...] = ("role", "from")

# Keys treated as message content inside a chat-list entry.
DEFAULT_CONTENT_KEYS: tuple[str, ...] = ("content", "value", "text")

# Flat question/answer records (Alpaca, MATH/NuminaMath ``problem``, MetaMathQA
# ``query``, prompt/completion, distilabel instruction/generation). Output keys
# are in priority order: a full ``solution`` beats a short final ``answer`` when
# a record has both.
DEFAULT_INSTRUCTION_KEYS: tuple[str, ...] = (
    "instruction",
    "question",
    "prompt",
    "problem",
    "query",
    "inputs",  # FLAN / P3 / Aya inputs/targets
)
DEFAULT_OUTPUT_KEYS: tuple[str, ...] = (
    "output",
    "response",
    "completion",
    "solution",
    "answer",
    "generated_solution",  # nvidia OpenMathInstruct
    "generation",  # distilabel text generation
    "targets",  # FLAN / P3 / Aya inputs/targets
)
DEFAULT_INPUT_KEYS: tuple[str, ...] = ("input", "context")


def iter_from_chat_line(
    record: dict[str, Any],
    *,
    conversation_keys: tuple[str, ...] = DEFAULT_CONVERSATION_KEYS,
    role_keys: tuple[str, ...] = DEFAULT_ROLE_KEYS,
    content_keys: tuple[str, ...] = DEFAULT_CONTENT_KEYS,
    role_map: dict[str, str] | None = None,
    pairwise_mode: str = "winner",
    instruction_keys: tuple[str, ...] = DEFAULT_INSTRUCTION_KEYS,
    output_keys: tuple[str, ...] = DEFAULT_OUTPUT_KEYS,
    input_keys: tuple[str, ...] = DEFAULT_INPUT_KEYS,
    reasoning_keys: tuple[str, ...] = DEFAULT_REASONING_KEYS,
    record_reasoning_keys: tuple[str, ...] = (),
) -> Iterator[TrainingExample]:
    """Yield zero or more :class:`TrainingExample` from a single raw record.

    ``reasoning_keys`` name the turn keys holding an assistant's reasoning
    trace; ``record_reasoning_keys`` the columns holding it in flat
    question/answer records (off by default: a top-level ``reasoning`` column
    is often an on/off flag, as in Llama-Nemotron).

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
            reasoning_keys=reasoning_keys,
        )
        return

    for key in conversation_keys:
        convs = record.get(key)
        if convs is None and f"{key}_json" in record:
            # ``messages_json`` / ``tools_json``: the columns as JSON strings.
            record = _json_columns(record, (key, "tools"))
            convs = record.get(key)
        if isinstance(convs, str):
            text = convs
            convs = _json_list(text)  # orca-agentinstruct: the turns as a JSON string
            if convs is None and text.strip():
                # A rendered conversation (Gemma, ChatML, ...) under a
                # conversation key instead of ``text``.
                turns = parse_text_chat(text)
                if turns:
                    yield build_example(turns, record, meta={"source": "chat:text"})
                    return
        if isinstance(convs, list) and convs:
            if "metadata" in record:
                convs = _with_turn_flags(convs, record["metadata"])
            msgs = coerce_messages(
                convs,
                role_keys=role_keys,
                content_keys=content_keys,
                role_map=role_map,
                reasoning_keys=reasoning_keys,
            ) or _input_output_turns(convs)
            if msgs and msgs[-1].role != "assistant" and not _has_answer(msgs):
                # The prompt turns, with the answer in its own column.
                answer = _answer_turn(record, (*output_keys, "target", "target_json"))
                if answer is not None:
                    msgs.append(answer)
            if msgs and not any(m.role == "user" for m in msgs):
                # The answer turns, with the question in its own column (smolagents).
                question = _first_string(record, _QUESTION_KEYS)
                if question is not None:
                    at = next((i for i, m in enumerate(msgs) if m.role != "system"), len(msgs))
                    msgs.insert(at, ChatMessage("user", question))
            if msgs:
                example = build_example(msgs, record, meta={"source": "chat"})
                if _starts_with_answer(msgs) and _null_user_turn(convs, role_keys, content_keys):
                    example.issues.append("withheld_prompt")
                yield example
            return

    prompt_turns, answer = record.get("input"), record.get("output")
    if isinstance(prompt_turns, list) and prompt_turns and isinstance(answer, str):
        # Llama-Nemotron: the prompt turns in ``input``, the answer in ``output``.
        msgs = coerce_messages(
            prompt_turns,
            role_keys=role_keys,
            content_keys=content_keys,
            role_map=role_map,
            reasoning_keys=reasoning_keys,
        )
        if msgs and answer.strip():
            reasoning = first_text(record, record_reasoning_keys)
            msgs.append(ChatMessage("assistant", answer, reasoning=reasoning))
            yield build_example(msgs, record, meta={"source": "chat"})
            return

    if is_glaive(record):
        msgs, tools = glaive_messages(record)
        if msgs:
            yield build_example(msgs, {**record, "tools": tools}, meta={"source": "chat:glaive"})
        return
    if is_xlam(record):
        yield build_example(xlam_messages(record), record, meta={"source": "chat:xlam"})
        return
    if oasst.is_tree(record):
        if record["prompt"].get("missing_root") is True:
            yield TrainingExample(meta={"source": "chat:oasst"}, issues=["missing_root"])
            return
        msgs = [ChatMessage(t["role"], t["content"]) for t in oasst.best_path(record)]
        tree = {"id": record.get("message_tree_id")}
        yield build_example(msgs, tree, meta={"source": "chat:oasst"})
        return

    # Resolve Alpaca cues up front so a stray ``text`` field can't silently
    # shadow a well-formed instruction/output record (see issue #17).
    instr = _first_string(record, instruction_keys)
    out = _first_string(record, output_keys)
    has_strong_alpaca = instr is not None and out is not None

    txt = record.get("text")
    if isinstance(txt, str) and txt.strip() and not has_strong_alpaca:
        turns = parse_text_chat(txt)
        if turns:
            yield build_example(turns, record, meta={"source": "chat:text"})
            return
        if instr is not None or out is not None:
            logger.warning(
                "chat adapter: routing record to the 'text' branch even though "
                "partial Alpaca keys are present; instruction/output content "
                "will be dropped. Pin --from alpaca for these records if that "
                "is wrong."
            )
        yield TrainingExample(
            messages=[ChatMessage(role="assistant", content=txt.strip())],
            meta=source_meta(record, {"source": "chat:text"}),
        )
        return

    # Fall back to the alpaca adapter, but let callers override the key priority.
    remapped = _remap_for_alpaca(record, instruction_keys, input_keys, output_keys)
    if remapped is not None:
        reasoning = first_text(record, record_reasoning_keys)
        yield from iter_from_alpaca_line(remapped, reasoning=reasoning)
        return
    lowered = _lower_case_keys(
        record, (*conversation_keys, *instruction_keys, *output_keys, "text", "chat")
    )
    if lowered is not record:
        # Only capitalised keys (``Instruction`` / ``Response``): read them again.
        yield from iter_from_chat_line(
            lowered,
            conversation_keys=conversation_keys,
            role_keys=role_keys,
            content_keys=content_keys,
            role_map=role_map,
            pairwise_mode=pairwise_mode,
            instruction_keys=instruction_keys,
            output_keys=output_keys,
            input_keys=input_keys,
            reasoning_keys=reasoning_keys,
            record_reasoning_keys=record_reasoning_keys,
        )


def _iter_pairwise(
    record: dict[str, Any],
    *,
    role_keys: tuple[str, ...],
    content_keys: tuple[str, ...],
    role_map: dict[str, str],
    pairwise_mode: str,
    reasoning_keys: tuple[str, ...] = DEFAULT_REASONING_KEYS,
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
        msgs = coerce_messages(
            convs,
            role_keys=role_keys,
            content_keys=content_keys,
            role_map=role_map,
            reasoning_keys=reasoning_keys,
        )
        if msgs:
            yield build_example(msgs, record, meta={"source": "chat:pairwise", "branch": label})


def _input_output_turns(convs: list[Any]) -> list[ChatMessage]:
    """Capybara-style turns: ``[{"input": user, "output": assistant}, ...]``."""
    msgs: list[ChatMessage] = []
    for item in convs:
        if not isinstance(item, dict):
            return []
        user, answer = item.get("input"), item.get("output")
        if not isinstance(user, str) or not isinstance(answer, str):
            return []
        if user.strip():
            msgs.append(ChatMessage("user", user))
        if answer.strip():
            msgs.append(ChatMessage("assistant", answer))
    return msgs


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
    # Keep the other columns (system, history, media) for the alpaca adapter.
    return {
        **record,
        "instruction": instr or "",
        "input": inp,
        "output": out or "",
        "response": "",
    }


def _lower_case_keys(record: dict[str, Any], known: tuple[str, ...]) -> dict[str, Any]:
    """``record`` with lower-case copies of its keys when only a capitalised
    form of a known key is present (``Instruction`` / ``Response``).

    A record that already has a known key is returned unchanged, so records
    that converted before are read exactly as before.
    """
    if any(k in record for k in known):
        return record
    extra = {k.lower(): v for k, v in record.items() if k.lower() != k and k.lower() in known}
    if not extra or any(k in record for k in extra):
        return record
    return {**record, **extra}


def _starts_with_answer(msgs: list[ChatMessage]) -> bool:
    for m in msgs:
        if m.role != "system":
            return m.role == "assistant"
    return False


def _null_user_turn(
    convs: list[Any], role_keys: tuple[str, ...], content_keys: tuple[str, ...]
) -> bool:
    """A user turn whose content is ``null`` (Nemotron chat withholds prompts that way)."""
    for item in convs:
        if not isinstance(item, dict):
            continue
        role = next((item[k] for k in role_keys if isinstance(item.get(k), str)), None)
        if role not in ("user", "human"):
            continue
        if all(item.get(k) is None for k in content_keys):
            return True
    return False


def _has_answer(msgs: list[ChatMessage]) -> bool:
    for m in msgs:
        if m.role == "assistant":
            return True
    return False


# Columns holding the question when the turns have no user turn at all.
_QUESTION_KEYS = ("prompt", "question", "original_question", "instruction", "query")


def _with_turn_flags(convs: list[Any], metadata: Any) -> list[Any]:
    """Turns with ``train`` set from a row-level flag list: Nemotron's
    ``metadata.train_turns``, one bool per turn (only when the lengths match)."""
    flags = metadata.get("train_turns") if isinstance(metadata, dict) else None
    if not isinstance(flags, list) or len(flags) != len(convs):
        return convs
    if not all(type(f) is bool for f in flags):
        return convs
    return [
        {**t, "train": f} if isinstance(t, dict) and "train" not in t else t
        for t, f in zip(convs, flags)
    ]


def _json_columns(record: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Read ``messages_json`` / ``tools_json`` (JSON strings) as ``messages`` /
    ``tools`` when the record has no column of that name."""
    extra = {
        k: record[f"{k}_json"]
        for k in keys
        if k not in record and isinstance(record.get(f"{k}_json"), str)
    }
    return {**record, **extra} if extra else record


def _answer_turn(record: dict[str, Any], keys: tuple[str, ...]) -> ChatMessage | None:
    """The assistant turn stored apart from the prompt turns: a string, or an
    object with ``content`` and/or ``tool_calls`` (``target_json``)."""
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and key.endswith("_json"):
            try:
                value = json.loads(value)
            except ValueError:
                continue
        if isinstance(value, str) and value.strip():
            return ChatMessage("assistant", value)
        if isinstance(value, dict):
            calls = tuple(parse_tool_calls(value))
            content = value.get("content")
            text = content if isinstance(content, str) and content.strip() else None
            if calls or text:
                return ChatMessage("assistant", text, tool_calls=calls)
    return None


def _first_string(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _json_list(text: str) -> list[Any] | None:
    stripped = text.lstrip()
    if not stripped.startswith("["):
        return None
    try:
        value = json.loads(stripped)
    except ValueError:
        return None
    return value if isinstance(value, list) else None
