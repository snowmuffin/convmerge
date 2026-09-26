"""Format normalization utilities.

Turn messy chat/instruct dataset files (parquet, JSON arrays, single-line JSONL,
mixed schemas) into clean newline-delimited JSONL, split files by turn count,
and deduplicate.

The public entry points are re-exported from :mod:`convmerge` (see
``docs/api.md``); the submodules are internal. The other helpers this package
used to re-export are deprecated here and leave this namespace in 1.0.
"""

from __future__ import annotations

from convmerge._deprecation import deprecated_names
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

_INTERNAL = "it is an internal helper; copy it if you depend on it"

__getattr__ = deprecated_names(
    __name__,
    {
        "count_turns": ("convmerge.normalize.turns:count_turns", _INTERNAL),
        "is_single_turn": ("convmerge.normalize.turns:is_single_turn", _INTERNAL),
        "detect_jsonl_shape": ("convmerge.normalize.jsonl:detect_jsonl_shape", _INTERNAL),
        "iter_json_records": ("convmerge.normalize.jsonl:iter_json_records", _INTERNAL),
        "load_jsonl": (
            "convmerge.normalize.jsonl:load_jsonl",
            "use convmerge.iter_jsonl() instead",
        ),
        "is_uniform_schema": ("convmerge.normalize.schema:is_uniform_schema", _INTERNAL),
        "key_frequency": (
            "convmerge.normalize.schema:key_frequency",
            "use convmerge.profile_schema() instead",
        ),
        "multi_turn_to_single_turn_record": (
            "convmerge.normalize.convert_turns:multi_turn_to_single_turn_record",
            _INTERNAL,
        ),
        "single_turn_to_multi_turn_record": (
            "convmerge.normalize.convert_turns:single_turn_to_multi_turn_record",
            _INTERNAL,
        ),
    },
)
