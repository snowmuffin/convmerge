"""Render the covered public surface as stable text (see docs/stability.md).

Every source module uses ``from __future__ import annotations``, so
annotations are the strings written in the source and render the same on
every supported Python version.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import importlib
import inspect
from typing import Any

PUBLIC_MODULES = ("convmerge", "convmerge.recipe", "convmerge.fetch")
# Values that change every release rather than by design.
_VALUE_NOT_PINNED = {"__version__"}


def _signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):  # builtins without introspectable signatures
        return "(...)"


def _describe_class(name: str, cls: type) -> list[str]:
    bases = ", ".join(b.__name__ for b in cls.__bases__ if b is not object)
    head = f"class {name}({bases})" if bases else f"class {name}"
    if issubclass(cls, enum.Enum):
        return [head, *(f"    {m.name} = {m.value!r}" for m in cls)]
    # A constructor inherited from a builtin (e.g. an Exception) has no
    # portable signature; show it only when defined in Python.
    init_owner = next(k for k in cls.__mro__ if "__init__" in vars(k) or k is object)
    if dataclasses.is_dataclass(cls) or inspect.isfunction(vars(init_owner).get("__init__")):
        head += _signature(cls)
    lines = [head]
    for attr, value in sorted(vars(cls).items()):
        if attr.startswith("_"):
            continue
        if isinstance(value, property):
            ret = inspect.signature(value.fget).return_annotation if value.fget else None
            lines.append(f"    {attr}: property -> {ret}")
        elif isinstance(value, (staticmethod, classmethod)):
            kind = type(value).__name__
            lines.append(f"    @{kind} {attr}{_signature(value.__func__)}")
        elif inspect.isfunction(value):
            lines.append(f"    def {attr}{_signature(value)}")
    return lines


def describe_module(module_name: str) -> list[str]:
    module = importlib.import_module(module_name)
    lines = [f"[{module_name}]"]
    for name in sorted(module.__all__):
        obj = getattr(module, name)
        if name in _VALUE_NOT_PINNED:
            lines.append(f"{name}: {type(obj).__name__}")
        elif inspect.isclass(obj):
            lines.extend(_describe_class(name, obj))
        elif callable(obj):
            lines.append(f"def {name}{_signature(obj)}")
        else:
            lines.append(f"{name}: {type(obj).__name__}")
    return lines


def describe_api() -> str:
    out: list[str] = []
    for module_name in PUBLIC_MODULES:
        out.extend(describe_module(module_name))
        out.append("")
    return "\n".join(out)


def _describe_parser(prog: str, parser: argparse.ArgumentParser) -> list[str]:
    lines = [f"[{prog}]"]
    nested: list[tuple[str, argparse.ArgumentParser]] = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                nested.append((f"{prog} {name}", sub))
            continue
        flags = ", ".join(action.option_strings) or action.dest
        parts = [flags]
        if action.option_strings and action.nargs == 0:
            parts.append("(switch)")
        elif action.nargs is not None:
            parts.append(f"nargs={action.nargs}")
        if action.choices is not None:
            parts.append("choices=" + "|".join(str(c) for c in action.choices))
        if action.required and action.option_strings:
            parts.append("required")
        if action.default not in (None, argparse.SUPPRESS, False):
            parts.append(f"default={action.default!r}")
        lines.append("  " + " ".join(parts))
    for sub_prog, sub in nested:
        lines.extend(_describe_parser(sub_prog, sub))
    return lines


def describe_cli() -> str:
    from convmerge.cli import _build_parser

    return "\n".join(_describe_parser("convmerge", _build_parser())) + "\n"
