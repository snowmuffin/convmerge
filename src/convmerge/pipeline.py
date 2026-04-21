"""End-to-end dataset-preparation pipeline.

This is a thin orchestrator over the individual primitives in
:mod:`convmerge.normalize` and :mod:`convmerge.convert`. It intentionally
has no external dependencies beyond those primitives: no HuggingFace Hub,
no model inference, no classification / labeling calls. It produces clean
train/test JSONL on the local filesystem; everything after that (upload,
labeling, training) lives in your own pipeline.

Typical usage::

    from convmerge.pipeline import build_sft_jsonl

    result = build_sft_jsonl(
        raw_dir="./raw",
        out_dir="./out",
        multi_container_keys=("conversation", "conversations", "text"),
        alpaca_instruction_keys=("instruction", "question", "prompt"),
        alpaca_output_keys=("output", "answer", "solution", "completion"),
        train_ratio=0.98,
        seed=42,
    )
    print(result.train_path, result.test_path)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from convmerge.adapters.chat import DEFAULT_CONVERSATION_KEYS
from convmerge.convert import convert_dir
from convmerge.normalize.dedup import deduplicate_jsonl
from convmerge.normalize.jsonl import normalize_dir, prune_by_suffix
from convmerge.normalize.merge import collect_jsonl_tree
from convmerge.normalize.reshape import (
    DEFAULT_SFT_STRIP_KEYS,
    unify_alpaca_dir,
    unify_message_entries,
    unify_messages_dir,
)
from convmerge.normalize.split import train_test_split
from convmerge.normalize.turns import filter_by_min_turns


@dataclass(frozen=True)
class BuildResult:
    """Summary returned by :func:`build_sft_jsonl`."""

    jsonl_dir: Path
    multi_turn_dir: Path
    single_turn_dir: Path
    final_dir: Path
    all_path: Path
    dedup_path: Path
    filtered_path: Path
    train_path: Path
    test_path: Path
    counts: dict[str, int] = field(default_factory=dict)


def build_sft_jsonl(
    raw_dir: str | Path,
    out_dir: str | Path,
    *,
    # Intermediate layout (relative to out_dir).
    jsonl_subdir: str = "jsonl",
    multi_turn_subdir: str = "multi_turn",
    single_turn_subdir: str = "single_turn",
    final_subdir: str = "final",
    # Junk-file filters applied right after the raw -> jsonl normalize pass.
    prune_suffixes: Iterable[str] = (),
    # Schema-hoist keys for chat-style files (conversation -> messages, etc.).
    multi_container_keys: Iterable[str] = DEFAULT_CONVERSATION_KEYS,
    # ``from`` -> ``role`` and ``value`` -> ``content`` normalization for ShareGPT-style entries.
    multi_entry_source_keys: Iterable[str] = ("from", "value"),
    multi_entry_target_map: dict[str, str] | None = None,
    # Alpaca-shape sweep (single-turn files).
    alpaca_instruction_keys: Iterable[str] = ("instruction", "question", "prompt"),
    alpaca_output_keys: Iterable[str] = ("output", "answer", "solution", "completion"),
    alpaca_input_keys: Iterable[str] = ("input", "context"),
    alpaca_drop_keys: Iterable[str] = ("id", "url", "source"),
    sft_strip_metadata: bool = True,
    sft_strip_keys: Iterable[str] = DEFAULT_SFT_STRIP_KEYS,
    # Final-stage knobs.
    dedupe_algorithm: str = "md5",
    min_turns: int = 1,
    train_ratio: float = 0.98,
    seed: int = 42,
    max_samples: int | None = None,
) -> BuildResult:
    """Run the default SFT prep chain.

    Stages:
    normalize -> reshape -> convert -> merge -> dedupe -> filter -> split.

    All intermediate directories live under ``out_dir``; no network I/O or
    model calls are made. The return value points at the final ``train.jsonl``
    and ``test.jsonl`` plus every intermediate path, so callers can layer
    their own upload / labeling / training steps on top.
    """
    raw_p = Path(raw_dir)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    jsonl_dir = out_p / jsonl_subdir
    multi_turn_dir = out_p / multi_turn_subdir
    single_turn_dir = out_p / single_turn_subdir
    final_dir = out_p / final_subdir
    final_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    # 1) Raw -> clean JSONL tree.
    n_files, n_rows = normalize_dir(raw_p, jsonl_dir)
    counts["normalize_files"] = n_files
    counts["normalize_rows"] = n_rows

    # 2) Optional: prune junk files that the source dump ships.
    if tuple(prune_suffixes):
        removed = prune_by_suffix(jsonl_dir, prune_suffixes)
        counts["pruned_files"] = len(removed)

    # 3) Hoist chat-style rows under a single ``messages`` key (keeps pairwise
    #    winner branch, strips moderation metadata by default).
    multi_turn_dir.mkdir(parents=True, exist_ok=True)
    n_mt_files, n_mt_rows = unify_messages_dir(
        jsonl_dir,
        multi_turn_dir,
        adapter_kwargs={
            "conversation_keys": tuple(multi_container_keys),
        },
        sft_strip_metadata=sft_strip_metadata,
        sft_strip_keys=sft_strip_keys,
    )
    counts["multi_turn_files"] = n_mt_files
    counts["multi_turn_rows"] = n_mt_rows

    # 4) Normalize message entries (from -> role, value -> content, etc.).
    if multi_entry_target_map:
        # Apply once per source->target pair; each call rewrites the same tree.
        for sk, tk in multi_entry_target_map.items():
            unify_message_entries(
                multi_turn_dir,
                multi_turn_dir,
                source_keys=(sk,),
                target_key=tk,
            )
    else:
        # Reasonable default: try ``from`` -> ``role`` and ``value`` -> ``content``.
        unify_message_entries(
            multi_turn_dir, multi_turn_dir, source_keys=("from",), target_key="role"
        )
        unify_message_entries(
            multi_turn_dir, multi_turn_dir, source_keys=("value",), target_key="content"
        )

    # 5) Single-turn sweep (alpaca shape) — dropped into its own tree first
    #    so re-running does not clobber the unified multi_turn tree.
    single_turn_dir.mkdir(parents=True, exist_ok=True)
    n_st_files, n_st_rows = unify_alpaca_dir(
        jsonl_dir,
        single_turn_dir,
        instruction_keys=alpaca_instruction_keys,
        output_keys=alpaca_output_keys,
        input_keys=alpaca_input_keys,
        drop_keys=alpaca_drop_keys,
    )
    counts["single_turn_files"] = n_st_files
    counts["single_turn_rows"] = n_st_rows

    # 6) Convert single-turn -> multi-turn ``messages`` shape.
    n_conv_files, _, n_conv_written = convert_dir(
        single_turn_dir,
        multi_turn_dir,
        adapter_name="alpaca",
        output_format="messages",
    )
    counts["single_to_multi_files"] = n_conv_files
    counts["single_to_multi_rows"] = n_conv_written

    # 7) Collect everything into a single ``all.jsonl``.
    all_path = final_dir / "all.jsonl"
    n_all, n_bad = collect_jsonl_tree([multi_turn_dir], all_path)
    counts["all_rows"] = n_all
    counts["all_skipped_bad_json"] = n_bad

    # 8) Dedupe by full-row hash.
    dedup_path = final_dir / "all_unique.jsonl"
    n_total, n_kept = deduplicate_jsonl(all_path, dedup_path, algorithm=dedupe_algorithm)
    counts["dedupe_total"] = n_total
    counts["dedupe_kept"] = n_kept

    # 9) Filter out rows that don't carry at least ``min_turns`` assistant turns.
    filtered_path = final_dir / "all_filtered.jsonl"
    n_total_f, n_kept_f = filter_by_min_turns(dedup_path, filtered_path, min_turns=min_turns)
    counts["filter_total"] = n_total_f
    counts["filter_kept"] = n_kept_f

    # 10) Train / test split.
    train_path, test_path, n_tr, n_te = train_test_split(
        filtered_path,
        final_dir,
        train_ratio=train_ratio,
        seed=seed,
        max_samples=max_samples,
    )
    counts["train_rows"] = n_tr
    counts["test_rows"] = n_te

    return BuildResult(
        jsonl_dir=jsonl_dir,
        multi_turn_dir=multi_turn_dir,
        single_turn_dir=single_turn_dir,
        final_dir=final_dir,
        all_path=all_path,
        dedup_path=dedup_path,
        filtered_path=filtered_path,
        train_path=train_path,
        test_path=test_path,
        counts=counts,
    )
