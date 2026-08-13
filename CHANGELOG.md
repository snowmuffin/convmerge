# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `inspect` command + `profile_schema()`: profile a `.json` / `.jsonl` file's
  structure — per-field value types, presence ratio, sample values, and
  preserved nesting (`items` for list-of-object fields, `fields` for object
  fields) so `messages[].role` is distinguishable from a top-level `role`.
  Intended as the first step for designing input → output key mappings on
  unfamiliar datasets.

### Fixed

- `fetch` resume now writes and validates completion sidecars, so interrupted
  or modified outputs are fetched again instead of being skipped as complete
  (#25).

## [0.5.0] - 2026-06-02

### Added

- `dedupe`: optional `--seen-store sqlite` (with `--seen-db PATH`) keeps the
  seen-hash set in a disk-backed SQLite table for bounded memory on inputs with
  tens of millions of unique rows. Default `memory` is unchanged. Exposed on
  `deduplicate_jsonl` via `seen_store` / `seen_db` (#14).
- `convert` / `dedupe`: optional `--progress` flag (or `CONVMERGE_PROGRESS=1`)
  logs periodic row counts and throughput to stderr for long-running jobs; off
  by default. Exposed on `convert_file` / `deduplicate_jsonl` via
  `progress=True` (#18).

### Changed

- docs: `docs/format.md` now shows concrete input → output sample blocks for
  the `alpaca`, `sharegpt`, and `chat`/`auto` adapters (#10).

## [0.4.2] - 2026-06-02

### Fixed

- `load_jsonl`: added `on_error="fail" | "skip"`. The default `"fail"` keeps the
  existing behavior (one bad line discards the whole file); `"skip"` logs and
  skips only the offending line, keeping every row that parsed (#15).
- `detect_jsonl_shape`: pretty-printed top-level JSON arrays (`[` followed by
  objects on subsequent lines) are no longer misclassified as `jsonl`, so they
  normalize correctly (#16).
- `chat` adapter: a stray `text` field no longer shadows a well-formed
  instruction/output record. When both an instruction and an output key are
  present the record is routed to the Alpaca branch; a partial-key `text`
  fallback now logs a warning. Resolution order documented in `docs/format.md`
  (#17).

## [0.4.1] - 2026-05-28

### Changed

- README: added search-friendly tagline and expanded opening paragraph with
  Alpaca / ShareGPT / messages-format keywords for better discoverability.
- `pyproject.toml`: updated `description` to problem-oriented wording; added
  `messages-format`, `llm-training`, `data-pipeline`, `chat-dataset` keywords.

## [0.4.0] - 2026-05-07

### Added

- `convmerge mix`: weighted sampling and merging of multiple converted JSONL
  sources into a single training file. Supports inline `FILE:WEIGHT` pairs or
  a YAML/JSON config file. Fixed seed guarantees reproducibility; a sidecar
  `.mix.json` recipe records exact parameters for auditing and replay.
  Optional `--oversample` allows sampling with replacement when a source is
  smaller than its allocation. YAML configs require `convmerge[preset]`.

### Fixed

- Normalize JSONL inputs with a leading UTF-8 BOM, CRLF line endings, and
  trailing whitespace; report trailing-comma JSONL lines with file and line
  context.
- Leading whitespace on JSONL lines is now stripped (regression introduced in
  the BOM/CRLF fix — `rstrip` was used instead of `strip`).

## [0.3.3] - 2026-04-23

### Added

- Optional extra ``all``: installs PyYAML, ``datasets``, and PyArrow (full runtime
  feature set: fetch with HF, parquet normalize, YAML presets).
- CLI: ``--help`` epilog lists extras; normalize / fetch / preset short help
  mentions required extras.

### Changed

- Documented that ``fetch-hf`` and ``fetch-all`` pull in the same packages;
  both names remain for backward compatibility.
- README install section: ``[all]`` one-liner, granular extras, and a
  command-to-extra table.

## [0.3.2] - 2026-04-22

### Changed

- Tests and `load_jsonl` documentation use generic wording throughout.

## [0.3.1] - 2026-04-22

### Added

- Convert **presets**: YAML/JSON files with `adapter`, `output_format`, optional
  `adapter_options.chat` (tuning for `iter_from_chat_line`). Install with
  `pip install "convmerge[preset]"` (adds PyYAML).
- CLI: `convmerge convert --preset PATH` (with optional `--from` / `--format` /
  `--adapter-kwargs` overrides), `convmerge preset init`, and
  `convmerge preset validate`.
- Library: `convmerge.config` (`ConvertConfig`, `ChatAdapterOptions`,
  `build_convert_config`), `convmerge.convert.convert_with_config`,
  `convmerge.adapter_resolve.resolve_adapter`, and `convmerge.preset` loaders.
- Documentation: [docs/custom_presets.md](docs/custom_presets.md).

### Reverted

- **`0.3.0` has been reverted.** The `pipeline`, `reshape`, `resume`,
  `sample`, `merge`, and `split` primitives, along with `convert_dir`
  and the expanded CLI (`merge`, `split`, `sample`, `build`), were
  removed. They may come back in a later release after more design
  iteration. The `0.3.0` release on PyPI has been yanked; `pip install
  convmerge` resolves to `0.2.1`.

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
