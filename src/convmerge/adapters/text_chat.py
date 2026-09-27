"""Conversations serialized into one ``text`` string with a chat template.

Many SFT datasets ship already-rendered prompts (``timdettmers/
openassistant-guanaco``, ``OpenAssistant/oasst_top1``, HH-RLHF style
transcripts, Alpaca-template ``text`` columns). :func:`parse_text_chat`
recognizes the common templates and splits the string back into turns:

- ChatML: ``<|im_start|>role\\n...<|im_end|>``
- Llama 3: ``<|start_header_id|>role<|end_header_id|>\\n\\n...<|eot_id|>``
- Gemma: ``<start_of_turn>user\\n...<end_of_turn>`` (``model`` = assistant)
- Llama 2: ``[INST] <<SYS>>...<</SYS>> ... [/INST] ...`` with ``<s>`` / ``</s>``
- ``### Human: ... ### Assistant: ...`` (Guanaco)
- ``Human: ... Assistant: ...`` turns separated by blank lines (HH-RLHF)
- The Alpaca prompt: ``### Instruction:`` / ``### Input:`` / ``### Response:``
- ``<usr>`` / ``<bot>`` / ``<sys>`` lines (heegyu/open-korean-instructions):
  a leading ``<sys>`` is the system prompt (or a document to talk about); one
  right after a user turn is that turn's input and is joined to it

A trailing user turn with no answer is dropped (it carries nothing to learn
from). Text in none of these shapes returns ``None``.
"""

from __future__ import annotations

import re

from convmerge.models import ChatMessage

_ROLES = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "tool": "tool",
    "ipython": "tool",
}

_CHATML = re.compile(
    r"<\|im_start\|>\s*(\w+)[ \t]*\n?(.*?)(?:<\|im_end\|>|(?=<\|im_start\|>)|\Z)", re.S
)
_LLAMA3 = re.compile(
    r"<\|start_header_id\|>\s*(\w+)\s*<\|end_header_id\|>\s*(.*?)"
    r"(?:<\|eot_id\|>|(?=<\|start_header_id\|>)|\Z)",
    re.S,
)
_GEMMA = re.compile(
    r"<start_of_turn>\s*(\w+)[ \t]*\n?(.*?)(?:<end_of_turn>|(?=<start_of_turn>)|\Z)", re.S
)
_INST = re.compile(r"\[INST\](.*?)\[/INST\](.*?)(?=\[INST\]|<s>|\Z)", re.S)
_SYS = re.compile(r"<<SYS>>(.*?)<</SYS>>", re.S)
_GUANACO = re.compile(r"###\s*(Human|Assistant)\s*:[ \t]*")
_HH = re.compile(r"(?:^|\n\n)(Human|Assistant):[ \t]*")
_ALPACA = re.compile(r"###\s*(Instruction|Input|Response)\s*:[ \t]*\n?")
_USR_BOT = re.compile(r"(?:^|\n)[ \t]*<(usr|bot|sys)>[ \t]?")


def parse_text_chat(text: str) -> list[ChatMessage] | None:
    """Turns of a template-rendered conversation, or ``None`` if unrecognized."""
    if "<|im_start|>" in text:
        turns = _tagged(_CHATML, text)
    elif "<|start_header_id|>" in text:
        turns = _tagged(_LLAMA3, text)
    elif "<start_of_turn>" in text:
        turns = _tagged(_GEMMA, text)
    elif "<usr>" in text and "<bot>" in text:
        turns = _usr_bot(text)
    elif "[INST]" in text and "[/INST]" in text:
        turns = _llama2(text)
    elif _GUANACO.search(text):
        turns = _split(_GUANACO, text)
    elif "### Response:" in text and "### Instruction:" in text:
        turns = _alpaca(text)
    elif _HH.search(text) and "Assistant:" in text:
        turns = _split(_HH, text)
    else:
        return None
    while turns and turns[-1].role == "user":
        turns.pop()
    return turns or None


def _tagged(pattern: re.Pattern[str], text: str) -> list[ChatMessage]:
    turns: list[ChatMessage] = []
    for role, body in pattern.findall(text):
        mapped = _ROLES.get(role.lower())
        content = body.strip()
        if mapped and content:
            turns.append(ChatMessage(mapped, content))
    return turns


def _usr_bot(text: str) -> list[ChatMessage]:
    parts = _USR_BOT.split(text.strip())
    turns: list[ChatMessage] = []
    for i in range(1, len(parts) - 1, 2):
        tag, content = parts[i], parts[i + 1].strip()
        if not content:
            continue
        if tag == "sys" and turns and turns[-1].role == "user":
            turns[-1] = ChatMessage("user", f"{turns[-1].text}\n{content}")
        else:
            role = {"usr": "user", "bot": "assistant", "sys": "system"}[tag]
            turns.append(ChatMessage(role, content))
    return turns


def _llama2(text: str) -> list[ChatMessage]:
    turns: list[ChatMessage] = []
    for user, answer in _INST.findall(text):
        system = _SYS.search(user)
        if system:
            if system.group(1).strip() and not turns:
                turns.append(ChatMessage("system", system.group(1).strip()))
            user = _SYS.sub("", user)
        user = user.strip()
        answer = answer.replace("</s>", "").strip()
        if user:
            turns.append(ChatMessage("user", user))
        if answer:
            turns.append(ChatMessage("assistant", answer))
    return turns


def _split(pattern: re.Pattern[str], text: str) -> list[ChatMessage]:
    parts = pattern.split(text)
    # parts: [preamble, speaker, text, speaker, text, ...]
    turns: list[ChatMessage] = []
    for i in range(1, len(parts) - 1, 2):
        content = parts[i + 1].strip()
        if content:
            turns.append(ChatMessage("user" if parts[i] == "Human" else "assistant", content))
    return turns


def _alpaca(text: str) -> list[ChatMessage]:
    parts = _ALPACA.split(text)
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    user = "\n".join(s for s in (sections.get("Instruction"), sections.get("Input")) if s)
    answer = sections.get("Response", "")
    return [ChatMessage("user", user), ChatMessage("assistant", answer)] if user and answer else []
