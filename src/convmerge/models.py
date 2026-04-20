"""Internal training-example model (adapter in → emit out)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChatMessage:
    """One turn in a conversation."""

    role: str
    content: str


@dataclass
class TrainingExample:
    """Normalized example for emitters (multi-turn capable)."""

    messages: list[ChatMessage] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)
