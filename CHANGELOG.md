# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0rc1] - 2026-09-26

Release candidate: `pip install convmerge==0.6.0rc1`. Output changes are
listed in [docs/migration-0.6.md](docs/migration-0.6.md).

### Added

- Tool calling: OpenAI `tool_calls` (and legacy `function_call`),
  `tool_call_id`, `name`, and top-level `tools` are kept by the `chat`
  adapter; LLaMA-Factory `function_call` / `observation` turns and `tools`
  columns are decoded by `chat` and `sharegpt`. The `messages` output follows
  the OpenAI schema; `--tool-arguments object` writes arguments as objects.
- Multimodal by reference: OpenAI content arrays (`image_url`, audio, video,
  `{"type": "image"}` placeholders) and `images` / `videos` / `audios` / LLaVA
  `image` columns bound to `<image>`-style tokens become content parts.
  Media is never downloaded or decoded.
- LLaMA-Factory `system` and alpaca `history` columns become system messages
  and earlier turns.
- Validation: `convert --on-invalid drop|keep|fail` (default `drop`) with
  per-reason counts, `--report PATH`, and a new `convmerge validate` command
  (`convmerge.validate.validate_example`, `convert.validate_file`).
- `alpaca` output: `system` field, `--alpaca-multiturn flatten|history|drop`,
  and lossy-conversion counts (#23).
- `--keep-meta [KEYS]` / `--meta-key` write provenance (`source`, record `id`,
  pairwise `branch`) (#22).
- Presets: `adapter_options.sharegpt` and an `output_options` block.
- `normalize` / `inspect` handle JSONL whose lines are arrays, wrapping them
  as `{"conversation": [...]}` (`--array-key`) (#26); `inspect` reports
  `element_types` / `element_examples` for list fields (#27).
- `convmerge.io.iter_jsonl`: one shared JSONL reader (BOM, blank and invalid
  lines handled and counted the same way in every command).
- Golden regression suite over realistic dataset shapes
  (`tests/golden/`, `CONVMERGE_UPDATE_GOLDEN=1`).

### Changed

- **Breaking:** `sharegpt` `turn_mode` defaults to `full` (whole
  conversation); `pairs` restores the 0.5 split (#23).
- **Breaking:** examples failing validation are dropped by default, e.g.
  assistant-only rows from plain `text` records.
- **Breaking:** `alpaca` output drops examples with tool calls or media
  instead of reducing them to text.
- Adapters skip blank turns (such as an empty system prompt).
- Diagnostics go to stderr only (`load_jsonl` logs via the `convmerge`
  logger; `fetch` progress lines go to stderr).
- `convert` is about 1.5x slower on plain chat data than 0.5.1 because every
  example is now validated (≈38k rows/s in our benchmark).

### Fixed

- Vision samples no longer lose their user turn, and tool-call conversations
  no longer keep tool results while dropping the calls.
- A UTF-8 BOM on the first line no longer turns it into an invalid row in
  `convert` / `dedupe` / `mix` / `turns`.

## [0.5.1] - 2026-09-26

### Security

- `fetch` (`mode: clone`): the token is no longer embedded in the clone URL.
  It is passed to `git` / `git lfs` as a host-scoped `Authorization` header via
  git's environment config (git ≥ 2.31), so it no longer lands in
  `.git/config`, the process list, or git error messages (which the runner
  logged). Existing clones are now pulled with the token, and a token that
  0.5.0 or earlier stored in `origin` is scrubbed on the next fetch.
  **If you cloned private repos with an earlier version, rotate that token.**
- `fetch` (raw URL / Trees API): the GitHub token is only sent to
  `github.com`, `api.github.com`, and `raw.githubusercontent.com`, and is no
  longer forwarded when a request is redirected to another host. Raw URLs on
  other hosts are fetched anonymously.
- Runner failure logs and `fetch` shortcut output are passed through
  `redact_url`.

### Added

- `inspect` command + `profile_schema()`: profile a `.json` / `.jsonl` file's
  structure — per-field value types, presence ratio, sample values, and
  preserved nesting (`items` for list-of-object fields, `fields` for object
  fields) so `messages[].role` is distinguishable from a top-level `role`.
  Intended as the first step for designing input → output key mappings on
  unfamiliar datasets.
- `sharegpt` adapter: `turn_mode` option — `full` emits the whole
  conversation (system prompt and all turns), `pairs` keeps the previous
  one-example-per-user/assistant-pair behavior. Set via
  `--adapter-kwargs '{"sharegpt": {"turn_mode": "full"}}'`, a preset's
  `adapter_options.sharegpt`, or `AdapterOptions(sharegpt=...)` (#23).
- `convert_file(..., stats=ConvertStats())` and
  `deduplicate_jsonl(..., stats=DedupeStats())` report why rows were dropped
  (invalid JSON, non-object rows, records the adapter could not map, true
  duplicates). Return values are unchanged.

### Deprecated

- `sharegpt` adapter: leaving `turn_mode` unset keeps `pairs` for now but
  emits a `FutureWarning` for each record whose output would change; the
  default becomes `full` in 0.6.0. Set `turn_mode` explicitly to pin either
  behavior. The CLI prints one summary line with the affected-record count.

### Fixed

- GitHub raw/tree fetch and JSON/JSONL normalization now reject Git LFS pointer
  files with guidance to use clone mode with LFS enabled (#28).
- `fetch` resume now writes and validates completion sidecars, so interrupted
  or modified outputs are fetched again instead of being skipped as complete
  (#25). Outputs fetched by earlier versions have no sidecar and are fetched
  once more on upgrade. Directory snapshots ignore `.git`.
- `normalize` on a directory skips convmerge sidecars (`*.fetch.json`,
  `*.mix.json`) and hidden paths such as a cloned repo's `.git`, instead of
  converting them as data.
- `convert` / `dedupe` no longer drop malformed rows silently: the CLI warns
  with counts and the first invalid line number, and `dedupe` reports invalid
  lines separately from duplicates.
- `iter_json_records` (and so `inspect`): a `.json` file that actually holds
  JSONL is re-read as JSONL instead of failing with `FileNotFoundError`.
- `mix`: an explicit `--seed` now always overrides the config file's `seed`
  (previously `--seed 42` was ignored when the config set one).
- `convert_with_config` now forwards `progress`.
- `sharegpt` adapter no longer crashes on a non-string `value`.

### Changed

- CI installs `.[dev,all]` so fetch tests run against the full extra set.
- CLI help, error hints, and docs consistently mention the umbrella `[all]`
  extra next to each narrow extra.

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
