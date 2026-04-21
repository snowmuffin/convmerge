"""Source-format adapters: raw records → TrainingExample."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from convmerge.adapters.alpaca import iter_from_alpaca_line, remap_to_alpaca
from convmerge.adapters.chat import iter_from_chat_line
from convmerge.adapters.sharegpt import iter_from_sharegpt_line
from convmerge.models import TrainingExample

AdapterFn = Callable[[dict[str, Any]], Iterator[TrainingExample]]

ADAPTERS: dict[str, AdapterFn] = {
    "alpaca": iter_from_alpaca_line,
    "sharegpt": iter_from_sharegpt_line,
    "chat": iter_from_chat_line,
    # ``auto`` is an alias for ``chat`` since the chat adapter is already auto-detecting.
    "auto": iter_from_chat_line,
}


def get_adapter(name: str) -> AdapterFn:
    if name not in ADAPTERS:
        known = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"Unknown adapter {name!r}. Choose one of: {known}")
    return ADAPTERS[name]


__all__ = [
    "ADAPTERS",
    "AdapterFn",
    "get_adapter",
    "iter_from_alpaca_line",
    "iter_from_chat_line",
    "iter_from_sharegpt_line",
    "remap_to_alpaca",
]
