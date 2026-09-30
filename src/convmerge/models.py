"""Internal training-example model (adapter in → emit out)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

MEDIA_PART_TYPES: tuple[str, ...] = ("image", "audio", "video")


@dataclass(frozen=True)
class ContentPart:
    """One piece of a multi-part message.

    ``type`` is ``"text"`` (with ``text``) or a media type from
    :data:`MEDIA_PART_TYPES` (with ``url``). Media is kept **by reference
    only** — a URL or a path exactly as the source dataset gave it; convmerge
    never downloads or decodes media. A media part whose ``url`` is ``None``
    is an unresolved placeholder (e.g. ``{"type": "image"}`` without a
    matching ``images`` entry).
    """

    type: str
    text: str | None = None
    url: str | None = None

    @property
    def is_media(self) -> bool:
        return self.type in MEDIA_PART_TYPES


@dataclass(frozen=True)
class ToolCall:
    """A function call requested by an assistant turn.

    ``arguments`` is a JSON-encoded string, following the OpenAI convention;
    use :meth:`from_any` to accept a dict (as many datasets store it).
    """

    name: str
    arguments: str = "{}"
    id: str | None = None

    @classmethod
    def from_any(cls, name: str, arguments: Any, id: str | None = None) -> ToolCall:
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return cls(name=name, arguments=arguments, id=id)

    def arguments_object(self) -> Any:
        """Arguments decoded from JSON; the raw string if it is not valid JSON."""
        try:
            return json.loads(self.arguments)
        except (TypeError, ValueError):
            return self.arguments


@dataclass(frozen=True)
class ChatMessage:
    """One turn in a conversation.

    ``content`` is a plain string, a sequence of :class:`ContentPart` (stored
    as a tuple), or ``None`` (e.g. an assistant turn that only calls tools).
    ``reasoning`` is an assistant turn's reasoning trace when the source keeps
    it apart from the answer (a ``reasoning_content`` / ``thinking`` field or
    column); reasoning written inline as ``<think>...</think>`` stays in
    ``content`` unless ``convert --reasoning`` asks for it to be split out.
    """

    role: str
    content: str | Sequence[ContentPart] | None
    tool_calls: Sequence[ToolCall] = ()
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str | None = None
    train: bool | None = None
    """For an assistant turn: whether the dataset marks it to be trained on
    (a per-turn ``train`` / ``loss`` / ``weight`` key, or a row-level list
    such as Nemotron's ``metadata.train_turns``); ``None`` when it says
    nothing. Written only with ``convert --train-turns data``."""

    def __post_init__(self) -> None:
        content = self.content
        if content is not None and type(content) is not str and type(content) is not tuple:
            object.__setattr__(self, "content", tuple(content))
        if type(self.tool_calls) is not tuple:
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))

    @property
    def text(self) -> str:
        """Text content only: the string itself, or text parts joined by newlines."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(p.text for p in self.content if p.type == "text" and p.text)

    @property
    def media(self) -> tuple[ContentPart, ...]:
        if self.content is None or isinstance(self.content, str):
            return ()
        return tuple(p for p in self.content if p.is_media)


@dataclass
class TrainingExample:
    """Normalized example for emitters (multi-turn capable).

    ``tools`` holds function/tool schemas available to the conversation, in
    OpenAI form (``{"type": "function", "function": {...}}``).
    """

    messages: list[ChatMessage] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)
    tools: list[dict[str, Any]] | None = None
    issues: list[str] = field(default_factory=list)
    """Problems an adapter noticed while mapping the record (e.g.
    ``unresolved_image``); reported by validation, never emitted."""
    rejected: list[ChatMessage] | None = None
    """For preference data: the whole *rejected* conversation, while
    ``messages`` holds the chosen one. The shared leading turns are the prompt
    (see the ``preference`` output format); other formats ignore it."""
