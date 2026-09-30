"""OpenAssistant message trees (oasst1 / oasst2).

OpenAssistant publishes each conversation as a tree: a prompt, several ranked
assistant replies, prompter follow-ups under each reply, and so on. Two
layouts exist:

- ``*.trees.jsonl``: one tree per row, ``{"message_tree_id", "prompt":
  {"text", "role", "replies": [...]}}`` with replies nested.
- ``*.messages.jsonl`` and the Hub's parquet splits: one message per row
  (``message_id``, ``parent_id``, ``message_tree_id``, ``role``, ``text``,
  ``rank``), the rows of a tree next to each other in depth-first order.

:func:`group_messages` folds consecutive message rows of one tree into a
tree record, so both layouts reach the adapter the same way.
:func:`best_path` follows the best-ranked reply at every step (``rank`` 0,
as in OpenAssistant's own top-1 exports), skipping deleted replies and
replies that failed review; :func:`ranked_pair` turns the deepest step with
two ranked replies into a chosen / rejected pair.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

_ROLES = {"prompter": "user", "assistant": "assistant"}
_ID_TYPES = (str, int)


def is_tree(record: dict[str, Any]) -> bool:
    """A ``*.trees.jsonl`` row: the root message under ``prompt``, replies nested."""
    root: Any = record.get("prompt")
    return type(root) is dict and "replies" in root and "message_tree_id" in record


def is_message(record: Any) -> bool:
    """One row of a flat OpenAssistant message table."""
    return (
        type(record) is dict
        and "message_tree_id" in record
        and type(record.get("message_id")) in _ID_TYPES
        and type(record.get("parent_id", False)) in (*_ID_TYPES, type(None))
        and type(record.get("role")) is str
        and record["role"] in _ROLES
    )


class TreeBuffer:
    """Collects consecutive message rows of one tree (see :func:`group_messages`)."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.first = 0

    def add(self, number: int, row: dict[str, Any]) -> tuple[int, Any, int] | None:
        """Add a message row; returns the previous tree when this row starts a new one."""
        rows = self.rows
        if rows and row["message_tree_id"] == rows[0]["message_tree_id"]:
            rows.append(row)
            return None
        done = self.flush()
        self.rows, self.first = [row], number
        return done

    def flush(self) -> tuple[int, Any, int] | None:
        """The collected tree as ``(first line number, tree record, rows folded)``."""
        if not self.rows:
            return None
        done = (self.first, build_tree(self.rows), len(self.rows) - 1)
        self.rows = []
        return done


def group_messages(
    items: Iterable[tuple[int, Any]],
) -> Iterator[tuple[int, Any, int]]:
    """Fold consecutive message rows of one tree into one tree record.

    ``items`` are ``(line number, parsed row)``; yields ``(line number of the
    tree's first row, record, rows folded into it beyond the first)``. Rows
    that are not OpenAssistant messages pass through unchanged (folded 0).
    """
    buffer = TreeBuffer()
    for number, obj in items:
        if type(obj) is dict and "message_tree_id" in obj and is_message(obj):
            done = buffer.add(number, obj)
            if done is not None:
                yield done
            continue
        done = buffer.flush()
        if done is not None:
            yield done
        yield number, obj, 0
    done = buffer.flush()
    if done is not None:
        yield done


def build_tree(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A tree record (the ``*.trees.jsonl`` layout) from the message rows of one tree."""
    nodes = {r["message_id"]: {**r, "replies": []} for r in rows if r.get("message_id")}
    root = None
    for node in nodes.values():
        parent = nodes.get(node.get("parent_id"))
        if parent is not None:
            parent["replies"].append(node)
        elif node.get("parent_id") is None and root is None:
            root = node
    record: dict[str, Any] = {"message_tree_id": rows[0]["message_tree_id"], "prompt": root}
    if root is None:
        record["prompt"] = {"replies": [], "missing_root": True}
    return record


def _usable(node: Any) -> bool:
    return (
        type(node) is dict
        and type(node.get("role")) is str
        and node["role"] in _ROLES
        and isinstance(node.get("text"), str)
        and bool(node["text"].strip())
        and node.get("deleted") is not True
        and node.get("review_result") is not False
    )


def _replies(node: dict[str, Any]) -> list[dict[str, Any]]:
    replies = node.get("replies")
    if not isinstance(replies, list):
        return []
    return [r for r in replies if _usable(r)]


def _rank(node: dict[str, Any]) -> float:
    rank = node.get("rank")
    if isinstance(rank, (int, float)) and not isinstance(rank, bool):
        return float(rank)
    return float("inf")  # unranked replies come after ranked ones


def _best(replies: list[dict[str, Any]]) -> dict[str, Any]:
    return min(enumerate(replies), key=lambda p: (_rank(p[1]), p[0]))[1]


def best_path(record: dict[str, Any]) -> list[dict[str, str]]:
    """The conversation along the best-ranked reply at every step.

    A trailing prompter turn with no usable reply is left out, so the path
    ends on an answer when there is one; a tree that is only a prompt keeps
    it (and is then dropped as ``no_assistant``).
    """
    root: Any = record.get("prompt")
    if not _usable(root):
        return []
    path: list[dict[str, Any]] = [root]
    while True:
        replies = _replies(path[-1])
        if not replies:
            break
        path.append(_best(replies))
    if len(path) > 1 and path[-1].get("role") == "prompter":
        path.pop()  # a prompt with no answer yet (a lone prompt stays: no_assistant)
    return [_turn(n) for n in path]


def ranked_pair(record: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]] | None:
    """(chosen, rejected) conversations from the deepest step of the best path
    where at least two assistant replies carry different ranks: the
    best-ranked reply against the worst-ranked one. ``None`` if no step has two.
    """
    root: Any = record.get("prompt")
    if not _usable(root):
        return None
    path: list[dict[str, Any]] = [root]
    found = None
    while True:
        replies = _replies(path[-1])
        if not replies:
            break
        best = _best(replies)
        ranked = [r for r in replies if r.get("role") == "assistant" and _rank(r) != float("inf")]
        if best.get("role") == "assistant" and len(ranked) >= 2:
            worst = max(enumerate(ranked), key=lambda p: (_rank(p[1]), p[0]))[1]
            if _rank(worst) > _rank(best):
                found = (list(path), best, worst)
        path.append(best)
    if found is None:
        return None
    prefix, best, worst = found
    turns = [_turn(n) for n in prefix]
    return [*turns, _turn(best)], [*turns, _turn(worst)]


def _turn(node: dict[str, Any]) -> dict[str, str]:
    return {"role": _ROLES[node["role"]], "content": node["text"]}
