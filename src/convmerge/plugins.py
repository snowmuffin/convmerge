"""Discovery of third-party adapters and output formats via entry points.

A package can add an adapter or an output format without changing convmerge
by declaring an entry point::

    # pyproject.toml of the plugin package
    [project.entry-points."convmerge.adapters"]
    my_schema = "my_pkg.convmerge_plugin:iter_from_my_schema"

    [project.entry-points."convmerge.emitters"]
    my_format = "my_pkg.convmerge_plugin:emit_my_format"

Entry points are loaded lazily, the first time an unknown name is looked up
(or when names are listed). Built-in names always win over entry points; a
plugin that fails to import is skipped with a warning.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from importlib.metadata import entry_points
from typing import Any

ADAPTER_GROUP = "convmerge.adapters"
EMITTER_GROUP = "convmerge.emitters"

logger = logging.getLogger(__name__)
_loaded: set[str] = set()


def load_entry_points(group: str, registry: MutableMapping[str, Any]) -> None:
    """Add every entry point of ``group`` missing from ``registry`` (once per process)."""
    if group in _loaded:
        return
    _loaded.add(group)
    for ep in entry_points(group=group):
        if ep.name in registry:
            continue
        try:
            registry[ep.name] = ep.load()
        except Exception as e:  # noqa: BLE001 - a broken plugin must not break convmerge
            logger.warning("skipping %s plugin %r (%s): %s", group, ep.name, ep.value, e)


def reset_for_tests() -> None:
    _loaded.clear()


def unknown_name_message(kind: str, name: str, known: list[str]) -> str:
    """``Unknown <kind> 'x'. Did you mean 'y'? Choose one of: ...``"""
    import difflib

    close = difflib.get_close_matches(name, known, n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    return f"Unknown {kind} {name!r}.{hint} Choose one of: {', '.join(sorted(known))}"
