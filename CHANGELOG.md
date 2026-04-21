# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-21

### Added

- **Directory-level normalize primitives** in `convmerge.normalize`:
  `normalize_dir`, `prune_by_suffix`, `head_preview`, `head_preview_dir`,
  `scan_shapes`. These hoist the directory-walking logic that previously
  lived only in the `convmerge normalize` CLI into module-level APIs so
  notebooks and scripts can reuse them.
- **Turn filtering**: `convmerge.normalize.filter_by_min_turns(src, dst,
  min_turns=...)` — stream a JSONL file while dropping rows whose assistant
  turn count is below a threshold.
- **New submodules** under `convmerge.normalize`:
  - `merge.py` — `merge_jsonl(sources, dst)` and
    `collect_jsonl_tree(source_dirs, dst)` for concatenating files and
    whole trees with optional JSON validation.
  - `split.py` — `train_test_split(src, out_dir, train_ratio=..., seed=...,
    max_samples=...)`. Pure-Python implementation, no numpy dependency.
  - `sample.py` — `sample_random` (in-memory) and `reservoir_sample`
    (streaming, memory-bounded) random samplers.
  - `resume.py` — `count_lines`, `truncate_to_n_lines`, and
    `trim_corrupt_tail` for append-only JSONL pipelines that need to
    resume after a crash.
  - `reshape.py` — directory-level schema unification:
    `unify_messages_dir`, `unify_message_entries`, `unify_alpaca_dir`,
    plus row-level helpers `parse_tagged_text` and `classify_row_shape`.
    In-place rewrites are supported via a temp-directory swap so a crashed
    run cannot leave partially overwritten files.
- **Alpaca remap helper**: `convmerge.adapters.alpaca.remap_to_alpaca` is
  now public (previously `_remap_for_alpaca` in the chat adapter). Use it
  to pull an `{instruction, input, output}` triple out of a messy
  single-turn row with configurable key-priority lists.
- **`convert_dir`**: `convmerge.convert.convert_dir(src_dir, dst_dir,
  adapter_name=..., output_format=...)` — a thin wrapper that runs
  `convert_file` over every JSONL under a directory tree while preserving
  the relative-path layout.
- **End-to-end pipeline**: `convmerge.pipeline.build_sft_jsonl(raw_dir,
  out_dir, ...)` runs normalize → reshape → convert → merge → dedupe →
  filter → split in a single call and returns a `BuildResult` with every
  intermediate path plus row counts. Intentionally local-filesystem-only:
  no HuggingFace Hub upload, no labeling, no model inference.
- **New CLI subcommands**: `convmerge merge`, `convmerge split`,
  `convmerge sample`, `convmerge build`.

### Changed

- `convmerge.normalize.__init__` re-exports the new primitives so most
  workflows can import from `convmerge.normalize` directly.
- `convmerge.adapters.__init__` now exports `remap_to_alpaca` alongside
  the existing adapter functions.

### Notes on scope

- No new required dependencies were added. `train_test_split` uses the
  standard-library `random` module; numpy is **not** a dependency of
  core or of the `pipeline` module.
- The library's "Out of scope" contract is unchanged — the new modules
  do not add model loading, inference, training, classification /
  labeling, tokenizer-aware filtering, or remote-service orchestration.

[0.3.0]: https://pypi.org/project/convmerge/0.3.0/

## [0.2.1] - 2026-04-20

### Added

- `CODE_OF_CONDUCT.md` based on Contributor Covenant 2.1.
- `examples/` directory with a README and ready-to-run `fetch` manifest
  skeletons for the Alpaca-style, ShareGPT-style, and mixed HF + GitHub
  patterns. Manifests use `<HF_ORG>/<DATASET>` and `ORG/REPO`
  placeholders rather than pinning specific third-party datasets.
- New issue templates: `new_adapter.yml` (adapter / emitter request) and
  `fetch_issue.yml` (fetch manifest problems). Issue config now links to
  the contributing guide and docs.
- `py.typed` marker in the distributed wheel, so downstream projects
  pick up inline type hints via PEP 561.

### Changed

- Expanded `CONTRIBUTING.md`: scope expectations (what is / isn't
  accepted), review SLA, end-to-end walkthrough for adding a new
  adapter, and updated install with the `[dev,fetch-all,parquet]`
  extras. "Good fits" examples are now described by pattern rather than
  by naming specific third-party projects.
- Richer `pyproject.toml` metadata: more `keywords` and `classifiers`
  (topic, audience, typed), additional `project.urls` entries for
  `Changelog` and `Documentation`, and `Development Status` bumped from
  pre-alpha to alpha.
- `README.md`: added PyPI / Python / CI / downloads / CoC badges, linked
  the Code of Conduct, and pointed to `good first issue` for
  contributors.

[0.2.1]: https://pypi.org/project/convmerge/0.2.1/

## [0.2.0] - 2026-04-20

### Added

- `convmerge fetch`: YAML-manifest driven downloader for HuggingFace and GitHub
  sources, with single-URL / `hf://` shortcut mode. See `docs/fetch.md`.
  - GitHub: raw URL download, Trees API recursive fetch with extension filter,
    `git clone` with optional `git lfs pull`.
  - HuggingFace: thin wrapper over `datasets.load_dataset(...).to_json(...)`.
  - Token resolution order: CLI flag → file → env var. URLs are redacted in logs.
- `convmerge normalize`: parquet / JSON array / single-line concatenated JSON
  → clean newline-delimited JSONL, batch over directories.
- `convmerge dedupe`: streaming MD5/SHA256-based deduplication, optional key
  projection.
- `convmerge turns`: single-turn vs multi-turn distribution report and
  deterministic file split.
- `convmerge.adapters.chat` / `auto`: auto-detecting adapter for
  `messages` / `conversation` / `conversations` / `text` / pairwise preference
  rows with overridable role map.
- Optional extras: `[fetch]` (pyyaml), `[fetch-hf]` (datasets),
  `[fetch-all]`, `[parquet]` (pyarrow).

### Changed

- PyPI publish workflow now authenticates with the `PYPI_API_TOKEN` GitHub
  Actions secret instead of OIDC trusted publishing.

[0.2.0]: https://pypi.org/project/convmerge/0.2.0/

## [0.1.0] - 2026-04-17

### Added

- `convmerge convert` CLI: `--input`, `--output`, `--from ADAPTER`, `--format FORMAT`.
- Adapters: `alpaca`, `sharegpt`.
- Output formats: `messages`, `alpaca`.
- Documentation: `docs/format.md`.
- CI workflow: Ruff + pytest on Python 3.10–3.12.
- Publish workflow: build and upload to PyPI on `v*` tags (trusted publishing).

[0.1.0]: https://pypi.org/project/convmerge/0.1.0/
