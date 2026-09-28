"""Format normalization utilities.

Turn messy chat/instruct dataset files (parquet, JSON arrays, single-line JSONL,
mixed schemas) into clean newline-delimited JSONL, split files by turn count,
and deduplicate.

The public entry points are re-exported from :mod:`convmerge` (see
``docs/api.md``); the submodules are internal.
"""

from __future__ import annotations

from convmerge.normalize.dedup import DedupeStats, deduplicate_jsonl
from convmerge.normalize.jsonl import normalize_to_jsonl
from convmerge.normalize.turns import analyze_turn_distribution, split_by_turns

__all__ = [
    "DedupeStats",
    "analyze_turn_distribution",
    "deduplicate_jsonl",
    "normalize_to_jsonl",
    "split_by_turns",
]
