"""Format normalization utilities.

Turn messy chat/instruct dataset files (parquet, JSON arrays, single-line JSONL,
mixed schemas) into clean newline-delimited JSONL, and provide building blocks
for turn-count analysis, deduplication, and single-turn <-> multi-turn conversion.
"""

from __future__ import annotations

from convmerge.normalize.convert_turns import (
    multi_turn_to_single_turn_record,
    single_turn_to_multi_turn_record,
)
from convmerge.normalize.dedup import deduplicate_jsonl
from convmerge.normalize.jsonl import (
    detect_jsonl_shape,
    iter_json_records,
    load_jsonl,
    normalize_to_jsonl,
)
from convmerge.normalize.schema import is_uniform_schema, key_frequency
from convmerge.normalize.turns import (
    analyze_turn_distribution,
    count_turns,
    is_single_turn,
    split_by_turns,
)

__all__ = [
    "analyze_turn_distribution",
    "count_turns",
    "deduplicate_jsonl",
    "detect_jsonl_shape",
    "is_single_turn",
    "is_uniform_schema",
    "iter_json_records",
    "key_frequency",
    "load_jsonl",
    "multi_turn_to_single_turn_record",
    "normalize_to_jsonl",
    "single_turn_to_multi_turn_record",
    "split_by_turns",
]
