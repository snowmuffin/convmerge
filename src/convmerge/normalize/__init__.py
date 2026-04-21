"""Format normalization utilities.

Turn messy chat/instruct dataset files (parquet, JSON arrays, single-line JSONL,
mixed schemas) into clean newline-delimited JSONL, and provide building blocks
for turn-count analysis, deduplication, sampling, merging, splitting, and
schema reshaping.
"""

from __future__ import annotations

from convmerge.normalize.convert_turns import (
    multi_turn_to_single_turn_record,
    single_turn_to_multi_turn_record,
)
from convmerge.normalize.dedup import deduplicate_jsonl
from convmerge.normalize.jsonl import (
    detect_jsonl_shape,
    head_preview,
    head_preview_dir,
    iter_json_records,
    load_jsonl,
    normalize_dir,
    normalize_to_jsonl,
    prune_by_suffix,
    scan_shapes,
)
from convmerge.normalize.merge import collect_jsonl_tree, merge_jsonl
from convmerge.normalize.reshape import (
    DEFAULT_SFT_STRIP_KEYS,
    DEFAULT_TAG_MAP,
    classify_row_shape,
    parse_tagged_text,
    unify_alpaca_dir,
    unify_message_entries,
    unify_messages_dir,
)
from convmerge.normalize.resume import (
    count_lines,
    trim_corrupt_tail,
    truncate_to_n_lines,
)
from convmerge.normalize.sample import reservoir_sample, sample_random
from convmerge.normalize.schema import is_uniform_schema, key_frequency
from convmerge.normalize.split import train_test_split
from convmerge.normalize.turns import (
    analyze_turn_distribution,
    count_turns,
    filter_by_min_turns,
    is_single_turn,
    split_by_turns,
)

__all__ = [
    "DEFAULT_SFT_STRIP_KEYS",
    "DEFAULT_TAG_MAP",
    "analyze_turn_distribution",
    "classify_row_shape",
    "collect_jsonl_tree",
    "count_lines",
    "count_turns",
    "deduplicate_jsonl",
    "detect_jsonl_shape",
    "filter_by_min_turns",
    "head_preview",
    "head_preview_dir",
    "is_single_turn",
    "is_uniform_schema",
    "iter_json_records",
    "key_frequency",
    "load_jsonl",
    "merge_jsonl",
    "multi_turn_to_single_turn_record",
    "normalize_dir",
    "normalize_to_jsonl",
    "parse_tagged_text",
    "prune_by_suffix",
    "reservoir_sample",
    "sample_random",
    "scan_shapes",
    "single_turn_to_multi_turn_record",
    "split_by_turns",
    "train_test_split",
    "trim_corrupt_tail",
    "truncate_to_n_lines",
    "unify_alpaca_dir",
    "unify_message_entries",
    "unify_messages_dir",
]
