"""Deprecation helpers (see ``docs/stability.md``).

A deprecated name keeps working for at least the rest of its major version
and warns with :class:`DeprecationWarning` on use; the changelog lists it
under "Deprecated" in the release that starts warning.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable, Mapping
from typing import Any

REMOVED_IN = "2.0"


def warn_deprecated(
    what: str, *, instead: str | None = None, removed_in: str = REMOVED_IN, stacklevel: int = 3
) -> None:
    """Warn that ``what`` goes away in ``removed_in``.

    The default ``stacklevel`` points at the caller of the function that calls
    this helper.
    """
    message = f"{what} is deprecated and will be removed in convmerge {removed_in}"
    if instead:
        message += f"; {instead}"
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def deprecated_names(module: str, names: Mapping[str, tuple[str, str]]) -> Callable[[str], Any]:
    """Build a module ``__getattr__`` serving deprecated names.

    ``names`` maps each name to ``(target, instead)``: ``target`` is
    ``"package.module:attr"`` where the object now lives, ``instead`` tells
    the user what to use.
    """

    def __getattr__(name: str) -> Any:
        entry = names.get(name)
        if entry is None:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        target, instead = entry
        warn_deprecated(f"{module}.{name}", instead=instead)
        mod, _, attr = target.partition(":")
        return getattr(importlib.import_module(mod), attr)

    return __getattr__
