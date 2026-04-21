# Migrating notebook-resident SFT prep code to convmerge 0.3

`convmerge 0.3` promotes a number of directory-level preprocessing helpers
that previously lived inside analyst notebooks (hand-rolled `sync_*`,
`merge_*`, `split_*`, and resume helpers). The library API is now rich
enough that those notebook cells can be deleted in favour of a handful
of imports.

This document is for notebook maintainers opening a **follow-up PR** that
replaces the duplicated utility cells. It is purposely mechanical; each
row below says "delete cell X, replace with this import + these two
lines".

## 1. Replace file-system utility cells

| Delete cell                            | Replace with                                                |
| -------------------------------------- | ----------------------------------------------------------- |
| `remove_endwith(suffixes, dir_path)`   | `from convmerge.normalize import prune_by_suffix`           |
| `print_first_line_each_file(...)`      | `from convmerge.normalize import head_preview, head_preview_dir` |
| `find_single_line_jsonl(...)`          | `from convmerge.normalize import scan_shapes`               |
| `filter_data(dir_path, out_dir)`       | `from convmerge.normalize import normalize_dir`             |

Call site examples:

```python
from convmerge.normalize import normalize_dir, prune_by_suffix, scan_shapes

normalize_dir(RAW_DIR, JSONL_DIR)
prune_by_suffix(JSONL_DIR, ["-res.jsonl", "config.jsonl", "_infos.jsonl"])
shapes = scan_shapes(JSONL_DIR)
```

## 2. Replace multi-turn schema unification cells

| Delete                                                 | Replace                                                    |
| ------------------------------------------------------ | ---------------------------------------------------------- |
| `sync_multi_turn_files_for_list(...)` + `transform_line` | `convmerge.normalize.reshape.unify_messages_dir`           |
| `sync_message_format(...)`                             | `convmerge.normalize.reshape.unify_message_entries`        |
| `parse_tagged_text_to_messages(...)`                   | `convmerge.normalize.reshape.parse_tagged_text`            |
| `check_multi_turn`, `_looks_like_chat_row`             | `convmerge.normalize.reshape.classify_row_shape`           |

Drop the hard-coded `_SFT_STRIP_TOP_LEVEL_KEYS` frozenset; pass
`sft_strip_keys=` to `unify_messages_dir` (or rely on its default, which
covers LMSYS / WildChat / Arena metadata fields).

```python
from convmerge.normalize.reshape import (
    unify_messages_dir, unify_message_entries,
)

unify_messages_dir(
    JSONL_DIR, MULT_TURN_DIR,
    adapter_kwargs={"conversation_keys": ("messages", "conversation", "conversations", "text")},
)
unify_message_entries(MULT_TURN_DIR, MULT_TURN_DIR, source_keys=("from",),  target_key="role")
unify_message_entries(MULT_TURN_DIR, MULT_TURN_DIR, source_keys=("value",), target_key="content")
```

## 3. Replace single-turn schema unification cells

| Delete                                | Replace                                                    |
| ------------------------------------- | ---------------------------------------------------------- |
| `normalize_single_turn_record(...)`   | `convmerge.adapters.alpaca.remap_to_alpaca`                |
| `sync_single_turn_files(...)`         | `convmerge.normalize.reshape.unify_alpaca_dir`             |
| `single_turn_to_multi_turn_record`    | (covered by `convert_file` with the `alpaca` adapter)      |
| `sync_single_turn_to_multi_turn(...)` | `convmerge.convert.convert_dir(adapter_name="alpaca", output_format="messages")` |

## 4. Replace merge / split / dedupe / turn-filter cells

| Delete                          | Replace                                        |
| ------------------------------- | ---------------------------------------------- |
| `merge_jsonl(file_paths, out)`  | `convmerge.normalize.merge.merge_jsonl`        |
| `collect_all_samples(dirs, out)`| `convmerge.normalize.merge.collect_jsonl_tree` |
| `filter_empty_turns(src, dst)`  | `convmerge.normalize.filter_by_min_turns`      |
| `duplication_filter(...)`       | `convmerge.normalize.deduplicate_jsonl`        |
| `split_dataset(...)` (numpy)    | `convmerge.normalize.split.train_test_split` (pure Python; numpy is **not** required) |

## 5. Replace sampling / resume helpers in `refine.ipynb`

| Delete                                      | Replace                                           |
| ------------------------------------------- | ------------------------------------------------- |
| `sample_jsonl_random_lines(...)`            | `convmerge.normalize.sample.sample_random` (or `reservoir_sample` for streaming) |
| `_jsonl_line_count(...)`                    | `convmerge.normalize.resume.count_lines`          |
| `_truncate_jsonl_to_n_lines(...)`           | `convmerge.normalize.resume.truncate_to_n_lines`  |
| `_trim_corrupt_bundle_tail_and_splits(...)` | `convmerge.normalize.resume.trim_corrupt_tail(path, companions=(...))` |

## 6. Optional: collapse the whole pipeline into one call

If your notebook's cell 15 is exactly the `normalize → reshape → convert
→ merge → dedupe → filter_by_min_turns → train_test_split` chain, replace
the whole cell with:

```python
from convmerge.pipeline import build_sft_jsonl

result = build_sft_jsonl(
    raw_dir=RAW_DIR,
    out_dir=OUT_DIR,
    prune_suffixes=("-res.jsonl", "config.jsonl", "_infos.jsonl"),
    train_ratio=0.98,
    seed=42,
    min_turns=1,
)
TRAIN_PATH = result.train_path
TEST_PATH  = result.test_path
```

## 7. What stays in the notebooks

These pieces are **product-specific** and intentionally out of scope for
`convmerge`. Keep them in the notebook, but import the generic primitives
above instead of re-implementing them:

- RunPod worker-pool orchestration / GraphQL polling / batch classify
  (filter.ipynb cells 10, 18).
- HuggingFace Hub upload, `prepare_hf_dataset_repo`, README / card
  generation (filter.ipynb cells 12, 13, 16, 17).
- Anthropic identity / safety batch evaluation, MD / HTML reports
  (refine.ipynb cells 3–7).
- Neura rewrite / translate client, Neura system prompt, Qwen-specific
  tokenizer counting (refine.ipynb cells 2, 8, 9, 10).

See the [`Out of scope`](../README.md#out-of-scope) section of the README
for the reasoning.
