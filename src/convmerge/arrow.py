"""Fields whose JSON type changes between rows (``convmerge validate``).

``datasets.load_dataset("json")`` infers one Arrow type per field; a field
that is a string in one row and an object or list in another fails with
``ArrowInvalid: JSON parse error: Column(...) changed from string to object``
(``datasets`` 4.0, which LLaMA-Factory installs, fails on both kinds of
conflict below; 4.8 and later read them).
:func:`type_conflicts` walks every row of a JSONL file and reports the
paths that hold more than one kind of value, with the first line of each.
Integers and floats count as one kind (Arrow widens them), and ``null`` fits
any type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from convmerge.io import iter_jsonl

_KINDS: dict[type, str] = {
    str: "string",
    bool: "bool",
    int: "number",
    float: "number",
    dict: "object",
    list: "list",
}


@dataclass
class TypeConflict:
    path: str
    """Where the field is: ``messages[].tool_calls[].function.arguments.data``."""
    first_line: dict[str, int] = field(default_factory=dict)
    """Each kind seen (``string``, ``number``, ``bool``, ``object``, ``list``) and
    the first line with it."""
    values: dict[str, int] = field(default_factory=dict)
    """How many values of each kind."""

    def hint(self) -> str:
        if ".arguments" in self.path:
            return "tool-call arguments differ between calls: convert with --tool-arguments string"
        if self.path.endswith("content") and {"string", "list"} <= set(self.first_line):
            return "text and multimodal (list) content are mixed: split the file by modality"
        return "give this field one type in every row"

    def to_report(self) -> dict[str, Any]:
        return {"path": self.path, "first_line": self.first_line, "values": self.values,
                "hint": self.hint()}  # fmt: skip


def type_conflicts(path: str | Path, *, encoding: str = "utf-8") -> list[TypeConflict]:
    """Paths of ``path`` whose values are of more than one kind (see the module doc)."""
    seen: dict[str, dict[str, list[int]]] = {}
    for line in iter_jsonl(path, encoding=encoding):
        try:
            _walk(line.value, "", line.number, seen)
        except RecursionError:  # nested too deeply to walk; no training file looks like this
            continue
    out: list[TypeConflict] = []
    for p, kinds in seen.items():
        if len(kinds) > 1:
            out.append(TypeConflict(
                path=p,
                first_line={k: v[0] for k, v in sorted(kinds.items())},
                values={k: v[1] for k, v in sorted(kinds.items())},
            ))  # fmt: skip
    return sorted(out, key=lambda c: c.path)


def _walk(value: Any, path: str, number: int, seen: dict[str, dict[str, list[int]]]) -> None:
    kind = _KINDS.get(type(value))
    if kind is None:  # null (or anything json never yields)
        return
    kinds = seen.get(path)
    if kinds is None:
        kinds = seen[path] = {}
    entry = kinds.get(kind)
    if entry is None:
        kinds[kind] = [number, 1]
    else:
        entry[1] += 1
    if kind == "object":
        prefix = f"{path}." if path else ""
        for k, v in value.items():
            _walk(v, prefix + k, number, seen)
    elif kind == "list":
        item = f"{path}[]"
        for v in value:
            _walk(v, item, number, seen)
