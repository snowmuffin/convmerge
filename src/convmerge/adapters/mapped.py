"""Declarative field mapping: ``--from map``.

For records whose layout no built-in adapter knows (AI Hub exports, in-house
logs, nested API dumps), a :class:`MapSpec` says where the conversation is,
with dotted paths instead of code::

    # flat records: one question and one answer somewhere in the record
    {"system": "meta.instruction", "user": "question.text", "assistant": "answer.text"}

    # a list of turns
    {"turns": "dialogue[].utterances[]", "role": "speaker", "content": "text",
     "role_map": {"A": "user", "B": "assistant"}}

Path syntax: ``a.b`` walks into keys, ``a[0]`` picks a list item, ``a[]``
walks every item of a list (so ``x[].y[]`` flattens nested lists; ``a[-1]``
is the last item), and
``a."b.c"`` quotes a key that contains dots. In ``turns`` mode, ``role``,
``content``, ``name``, and ``reasoning`` are paths inside each turn; the
others are paths from the record. A record where a required path is missing
yields an example carrying the ``map_path_missing`` issue, so ``convert``
drops and counts it.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from convmerge.adapters._common import (
    MISSING,
    build_example,
    collapse_text,
    normalize_tools,
    parse_content,
)
from convmerge.models import ChatMessage, TrainingExample

# Default role labels understood in ``turns`` mode (extend with ``role_map``).
_ROLES: dict[str, str] = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "tool": "tool",
}

_FLAT_KEYS = ("system", "user", "assistant", "reasoning", "chosen", "rejected", "tools")
_TURN_KEYS = ("turns", "role", "content", "name", "reasoning", "role_map", "system", "tools")
MAP_KEYS: tuple[str, ...] = tuple(dict.fromkeys((*_FLAT_KEYS, *_TURN_KEYS)))

_STEP = re.compile(r'"((?:[^"\\]|\\.)*)"|([^.\[\]"]+)|\[(-?\d+|)\]')


@dataclass(frozen=True)
class Path:
    """A compiled dotted path (see the module docstring)."""

    text: str
    steps: tuple[tuple[str, Any], ...] = field(compare=False)

    @classmethod
    def parse(cls, text: str) -> Path:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"map: a path must be a non-empty string, got {text!r}")
        steps: list[tuple[str, Any]] = []
        pos = 0
        s = text.strip()
        while pos < len(s):
            if s[pos] == "." and steps:
                pos += 1
            m = _STEP.match(s, pos)
            if m is None:
                raise ValueError(f"map: cannot parse path {text!r} at {s[pos:]!r}")
            quoted, key, index = m.groups()
            if quoted is not None:
                steps.append(("key", quoted.replace('\\"', '"')))
            elif key is not None:
                steps.append(("key", key))
            else:
                steps.append(("each", None) if index == "" else ("index", int(index)))
            pos = m.end()
        return cls(text, tuple(steps))

    @property
    def many(self) -> bool:
        return any(kind == "each" for kind, _ in self.steps)

    def values(self, obj: Any) -> list[Any]:
        """Every value the path reaches (empty if it reaches none)."""
        current = [obj]
        for kind, arg in self.steps:
            nxt: list[Any] = []
            for v in current:
                if kind == "key":
                    if isinstance(v, Mapping) and arg in v:
                        nxt.append(v[arg])
                elif kind == "index":
                    if isinstance(v, list) and -len(v) <= arg < len(v):
                        nxt.append(v[arg])
                elif isinstance(v, list):
                    nxt.extend(v)
            current = nxt
        return current

    def get(self, obj: Any) -> Any:
        """The single value at the path, or :data:`MISSING`."""
        found = self.values(obj)
        return found[0] if found else MISSING


@dataclass(frozen=True)
class MapSpec:
    """Where a record keeps its conversation (``--adapter-kwargs '{"map": {...}}'``).

    Give either ``turns`` (a path to the list of turns; ``role`` / ``content``
    / ``name`` / ``reasoning`` are then paths inside a turn and ``role_map``
    renames roles) or ``user`` and ``assistant`` (paths to a single exchange;
    ``reasoning`` is then the answer's trace). ``system`` and ``tools`` are
    paths from the record in both modes. ``chosen`` / ``rejected`` (flat mode)
    make a preference pair: ``chosen`` replaces ``assistant``.
    """

    turns: Path | None = None
    role: Path | None = None
    content: Path | None = None
    name: Path | None = None
    role_map: Mapping[str, str] | None = None
    system: Path | None = None
    user: Path | None = None
    assistant: Path | None = None
    reasoning: Path | None = None
    chosen: Path | None = None
    rejected: Path | None = None
    tools: Path | None = None

    @classmethod
    def from_mapping(cls, data: Any) -> MapSpec:
        if not isinstance(data, Mapping):
            raise ValueError("map: expected a mapping of field names to paths")
        unknown = set(data) - set(MAP_KEYS)
        if unknown:
            raise ValueError(
                f"map: unknown key(s) {sorted(unknown)}; supported: {', '.join(MAP_KEYS)}"
            )
        role_map = data.get("role_map")
        if role_map is not None and (
            not isinstance(role_map, Mapping)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in role_map.items())
        ):
            raise ValueError("map.role_map: expected a mapping of labels to roles")
        paths = {k: Path.parse(v) for k, v in data.items() if k != "role_map"}
        spec = cls(**paths, role_map=dict(role_map) if role_map else None)
        spec._check()
        return spec

    def _check(self) -> None:
        if self.turns is not None:
            flat = [k for k in ("user", "assistant", "chosen", "rejected") if getattr(self, k)]
            if flat:
                raise ValueError(f"map: 'turns' cannot be combined with {flat}")
        else:
            if self.user is None or (self.assistant is None and self.chosen is None):
                raise ValueError(
                    "map: give 'turns' (a list of turns) or 'user' and 'assistant' "
                    "(or 'chosen') paths"
                )
            if (self.chosen is None) != (self.rejected is None):
                raise ValueError("map: 'chosen' and 'rejected' go together")
            for key in ("role", "content", "name", "role_map"):
                if getattr(self, key) is not None:
                    raise ValueError(f"map: {key!r} only applies with 'turns'")

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in MAP_KEYS:
            value = getattr(self, key)
            if isinstance(value, Path):
                out[key] = value.text
            elif value is not None:
                out[key] = dict(value)
        return out


def iter_from_mapped_line(
    record: dict[str, Any], *, spec: MapSpec, preference: str | None = None
) -> Iterator[TrainingExample]:
    """Yield the example ``spec`` finds in ``record``.

    ``preference`` (``"chosen"`` / ``"rejected"``) turns a pair into an SFT
    example on that answer; otherwise a pair keeps both sides (``rejected``).
    """
    try:
        if spec.turns is not None:
            msgs = _turns(record, spec)
            rejected = None
        else:
            msgs, rejected = _flat(record, spec, preference)
    except _Missing as e:
        meta: dict[str, object] = {"source": "map", "missing": e.path}
        yield TrainingExample(meta=meta, issues=["map_path_missing"])
        return
    extras: dict[str, Any] = {"id": record.get("id")}
    if spec.tools is not None:
        extras["tools"] = normalize_tools(spec.tools.get(record))
    example = build_example(msgs, extras, meta={"source": "map"})
    if rejected is not None:
        example.rejected = [*example.messages[: len(example.messages) - 1], rejected]
    yield example


class _Missing(Exception):
    def __init__(self, path: str):
        super().__init__(path)
        self.path = path


def _text(path: Path, obj: Any, *, required: bool) -> Any:
    value = path.get(obj)
    if value is MISSING or value is None:
        if required:
            raise _Missing(path.text)
        return None
    parsed = parse_content(value)
    if parsed is MISSING:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if required:
            raise _Missing(path.text)
        return None
    return collapse_text(parsed)


def _system(record: dict[str, Any], spec: MapSpec) -> list[ChatMessage]:
    if spec.system is None:
        return []
    text = _text(spec.system, record, required=False)
    return [ChatMessage("system", text)] if isinstance(text, str) and text.strip() else []


def _flat(
    record: dict[str, Any], spec: MapSpec, preference: str | None
) -> tuple[list[ChatMessage], ChatMessage | None]:
    assert spec.user is not None
    user = _text(spec.user, record, required=True)
    reasoning = _reasoning(spec.reasoning, record)
    msgs = [*_system(record, spec), ChatMessage("user", user)]
    if spec.chosen is None:
        assert spec.assistant is not None
        answer = _text(spec.assistant, record, required=True)
        return [*msgs, ChatMessage("assistant", answer, reasoning=reasoning)], None
    assert spec.rejected is not None
    chosen_text = _text(spec.chosen, record, required=True)
    chosen = ChatMessage("assistant", chosen_text, reasoning=reasoning)
    rejected = ChatMessage("assistant", _text(spec.rejected, record, required=True))
    if preference == "rejected":
        return [*msgs, rejected], None
    if preference == "chosen":
        return [*msgs, chosen], None
    return [*msgs, chosen], rejected


def _reasoning(path: Path | None, obj: Any) -> str | None:
    if path is None:
        return None
    value = path.get(obj)
    return value if isinstance(value, str) and value.strip() else None


def _turns(record: dict[str, Any], spec: MapSpec) -> list[ChatMessage]:
    assert spec.turns is not None
    items = spec.turns.values(record)
    if len(items) == 1 and isinstance(items[0], list) and not spec.turns.many:
        items = items[0]
    if not items:
        raise _Missing(spec.turns.text)
    role_path = spec.role or Path.parse("role")
    content_path = spec.content or Path.parse("content")
    roles = {**_ROLES, **{k.lower(): v for k, v in (spec.role_map or {}).items()}}
    msgs = _system(record, spec)
    for turn in items:
        raw_role = role_path.get(turn)
        if not isinstance(raw_role, str) or not raw_role.strip():
            raise _Missing(role_path.text)
        role = roles.get(raw_role.strip().lower(), raw_role.strip())
        content = _text(content_path, turn, required=False)
        if content is None or (isinstance(content, str) and not content.strip()):
            continue
        name = spec.name.get(turn) if spec.name is not None else None
        msgs.append(
            ChatMessage(
                role,
                content,
                name=name if isinstance(name, str) and name else None,
                reasoning=_reasoning(spec.reasoning, turn) if role == "assistant" else None,
            )
        )
    return msgs
