"""Source-format adapters: raw records → TrainingExample."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from convmerge.adapters.alpaca import iter_from_alpaca_line
from convmerge.adapters.sharegpt import iter_from_sharegpt_line
from convmerge.models import TrainingExample

AdapterFn = Callable[[dict[str, Any]], Iterator[TrainingExample]]

ADAPTERS: dict[str, AdapterFn] = {
    "alpaca": iter_from_alpaca_line,
    "sharegpt": iter_from_sharegpt_line,
}


def get_adapter(name: str) -> AdapterFn:
    if name not in ADAPTERS:
        known = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"Unknown adapter {name!r}. Choose one of: {known}")
    return ADAPTERS[name]
