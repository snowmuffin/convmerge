"""Source-format adapters: raw records → TrainingExample."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from convmerge.adapters.alpaca import iter_from_alpaca_line
from convmerge.adapters.chat import iter_from_chat_line
from convmerge.adapters.sharegpt import iter_from_sharegpt_line
from convmerge.models import TrainingExample
from convmerge.plugins import ADAPTER_GROUP, load_entry_points

AdapterFn = Callable[[dict[str, Any]], Iterator[TrainingExample]]

ADAPTERS: dict[str, AdapterFn] = {
    "alpaca": iter_from_alpaca_line,
    "sharegpt": iter_from_sharegpt_line,
    "chat": iter_from_chat_line,
    # ``auto`` is an alias for ``chat`` since the chat adapter is already auto-detecting.
    "auto": iter_from_chat_line,
}


def _map_needs_spec(record: dict[str, Any]) -> Iterator[TrainingExample]:
    raise ValueError(
        "the map adapter needs a field mapping: pass --adapter-kwargs "
        "'{\"map\": {...}}', or use convmerge.adapters.mapped.iter_from_mapped_line"
    )


# ``map`` is bound to its field mapping by convmerge.adapter_resolve.resolve_adapter.
ADAPTERS["map"] = _map_needs_spec


BUILTIN_ADAPTERS = frozenset(ADAPTERS)


def register_adapter(name: str, fn: AdapterFn, *, replace: bool = False) -> None:
    """Make ``fn`` available as ``--from name``.

    ``fn`` takes one raw record (``dict``) and yields :class:`TrainingExample`
    objects. Registering an existing name raises unless ``replace=True``.
    For ``convert --workers``, register at import time of your module or use
    the ``convmerge.adapters`` entry point (see :mod:`convmerge.plugins`).
    """
    if not replace and name in ADAPTERS:
        raise ValueError(f"adapter {name!r} is already registered (pass replace=True)")
    ADAPTERS[name] = fn


def available_adapters() -> list[str]:
    """Built-in, registered, and entry-point adapter names."""
    load_entry_points(ADAPTER_GROUP, ADAPTERS)
    return sorted(ADAPTERS)


def get_adapter(name: str) -> AdapterFn:
    if name not in ADAPTERS:
        load_entry_points(ADAPTER_GROUP, ADAPTERS)
    if name not in ADAPTERS:
        known = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"Unknown adapter {name!r}. Choose one of: {known}")
    return ADAPTERS[name]
