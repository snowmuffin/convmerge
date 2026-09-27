# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Datasets` workflow: converts real rows of every catalog dataset from the
  Hub on catalog / adapter pull requests, weekly, and on demand, with a
  per-dataset summary table. `scripts/datasets.py check` gains `--summary`
  and `--min-ok`, skips gated datasets without `HF_TOKEN`, and fails a
  reasoning dataset whose converted rows carry no reasoning trace.

### Changed

- mypy checks against each CI job's own Python instead of a pinned 3.10, so
  stubs that newer dependencies ship only for newer Pythons (numpy 2.5) no
  longer break the check.

## [0.12.0] - 2026-09-27

What the chat template actually sees: reasoning traces in the field each
model reads, tool-call turns every template accepts, fixes for strict
templates, and `tokens` checks for the failures that otherwise stay silent
(no loss mask, no stop token, answers truncated away, reasoning dropped).

### Added

- Reasoning traces: `ChatMessage.reasoning`, read from `reasoning_content` /
  `thinking` / `reasoning` turn keys (`ChatAdapterOptions.reasoning_keys`)
  and, opt-in, from flat-record columns (`record_reasoning_keys`, e.g. s1K).
  `convert --reasoning keep|inline|reasoning_content|thinking|drop`
  (`EmitOptions.reasoning`) writes them where the target template reads them
  (Qwen3 / DeepSeek: `reasoning_content`, gpt-oss: `thinking`); formats
  without a field write them inline. Inline `<think>` text is untouched by
  default. `convert` counts examples with a trace (`reasoning`).
- Fixes for strict chat templates (`TransformOptions`, presets `transforms:`,
  recipe source keys): `--system fold|drop`, `--merge-consecutive`,
  `--split-turns` (one example per user turn), `--reasoning-turns last`;
  counted in the report under `transforms`.
- `convmerge tokens` checks: `generation_tags` (TRL `assistant_only_loss`
  needs `{% generation %}`), `answer_beyond_limit` (rows whose answer starts
  beyond `--max-tokens`), `stop_tokens` / `missing_eos` (tokenizer eos plus
  `generation_config.json`), `reasoning_dropped` / `reasoning_dropped_final`,
  and `hints` naming the `convert` flag that fixes each finding. They are
  warnings; the exit status is unchanged.
- Llama-Nemotron rows (`input` turns + `output`) and `system_prompt` columns.
- Catalog: reasoning datasets (OpenR1-Math, OpenThoughts3, s1K-1.1,
  Llama-Nemotron, Multilingual-Thinking, AM-Thinking); entries can carry
  `--adapter-kwargs` and emit flags.
- Guides: [reasoning data](docs/guides/reasoning.md) and a
  [symptom → fix table](docs/guides/troubleshooting.md).

### Changed

- **Tool-call-only assistant turns are written with `"content": ""`**
  instead of `null` (`--tool-content null` restores it): `null` fails in the
  Qwen3, gpt-oss, DeepSeek-R1, GLM-4, Mistral, and Phi-4 templates, `""`
  renders in all of them.
- `llamafactory-info` refuses files with `reasoning_content` / `thinking`
  fields, which LLaMA-Factory cannot read.
- `tokens` renders tool-call content and reasoning fields exactly as the file
  stores them.

### Verified

- TRL 1.14 `SFTTrainer` with the Qwen3 template: multi-turn
  `reasoning_content` data trains 40 of 80 traces (as `tokens` reports) and
  `--split-turns` data trains 80 of 80. gpt-oss template on `--reasoning
  thinking` data with tool calls. LLaMA-Factory 0.9.5 with the `qwen3`
  template on `--format sharegpt` reasoning data.

## [0.11.0] - 2026-09-27

From converted data to a training run: filter for the target model, split
off validation, and write what each trainer reads — checked by actually
training with TRL and LLaMA-Factory on convmerge output.

### Added

- `convmerge tokens` (`check_tokens`, new `[tokens]` extra with
  `transformers`, no PyTorch): renders every row with the model's chat
  template, reports the token-length distribution, rows over
  `--max-tokens`, rows the template rejects (grouped by the template's error,
  with line numbers), and rows whose tool-call arguments the template would
  encode twice; `-o` keeps the rows that render and fit.
- `convmerge split` (`split_jsonl`): train/validation split by seeded content
  hash — reproducible, independent of row order, duplicates on one side;
  `--val` (fraction, streaming) or `--val-rows` (exact); `--keys` to group.
- Recipe stages `tokens` and `split` after `mix` / `dedupe` (split writes
  `<output>.val.jsonl` by default); both are tracked by the lock file.
- `--format sharegpt` and `--format sharegpt-preference`: LLaMA-Factory /
  Unsloth ShareGPT rows (system, tools, media, function_call / observation
  turns) and LLaMA-Factory ranking pairs.
- `convmerge llamafactory-info`: the `dataset_info.json` entry for a
  converted file, printed or merged into an existing file.
- `convmerge validate` checks preference rows as pairs.
- Trainer guides (`docs/guides/`: TRL, LLaMA-Factory, axolotl) with the
  recipes and configs in `examples/`. The TRL (SFT, DPO) and LLaMA-Factory
  (SFT with tool calls, DPO) paths were run end to end on convmerge output.
- `tokens` JSON output carries `"version": 1` like the other reports.

### Changed

- `[all]` now includes `transformers` (for `tokens`).

## [0.10.0] - 2026-09-26

Reads the datasets people actually train on. A catalog of 33 popular SFT,
tool-calling, and preference datasets is now converted in the test suite
(see "Tested datasets" in the README), and preference data can be written as
DPO pairs.

### Added

- `--format preference`: chosen/rejected pairs in TRL's conversational
  format (`{prompt, chosen, rejected}`, shared turns as the prompt) from
  UltraFeedback-binarized / TRL, HH-RLHF, Orca DPO pairs, LLaMA-Factory
  ranking, Capybara DPO, and Chatbot Arena (`winner`) records.
  `TrainingExample` gains an optional `rejected` conversation.
- Tool-calling layouts beyond OpenAI's are decoded into standard
  `tool_calls` / `tool` turns / `tools`: Hermes `<tool_call>` /
  `<tool_response>` / `<tools>` tags, Glaive v2 `system` + `chat`
  transcripts, and xLAM `query` / `answers` / `tools`.
- Template-rendered `text` columns are split back into turns: ChatML, Llama 2,
  Llama 3, Gemma, Guanaco `### Human:`, HH `Human:` / `Assistant:`, and the
  Alpaca prompt template.
- Capybara-style `{input, output}` turn lists.
- `tests/datasets/catalog.json` + `tests/test_datasets.py`: 33 datasets with
  their real layouts, pinned outputs, and the README table generated from
  them; `scripts/datasets.py check` converts real rows from the Hub.

### Changed

- A preference record that an SFT conversion cannot use is dropped with the
  reason `preference_record` (and a CLI hint) instead of `no_assistant` or a
  silent skip; the drop report counts it under `dropped`, no longer under
  `no_example`.
- `validate_example()` on an example with no messages returns the adapter's
  issues when there are any, otherwise `no_messages` as before.

### Changed output

- Conversations with Hermes tool tags (a `tools` column, `tool` turns, or a
  `<tools>` system block) now get structured `tool_calls` and one `tool` turn
  per `<tool_response>`; previously the tags stayed in the text, or the
  conversation was dropped as `orphan_tool_message`.

## [0.9.0] - 2026-09-26

The last 0.x minor release: it defines what 1.0 will keep stable and
deprecates what 1.0 will drop. Nothing is removed and converted data is
byte-for-byte unchanged; code that runs on 0.9 without `DeprecationWarning`s
will run on 1.0
([migration-1.0.md](docs/migration-1.0.md)).

### Added

- `docs/stability.md`: what 1.x keeps compatible (public API, CLI flags and
  exit codes, file formats, reproducibility), the deprecation policy, and the
  Python version policy.
- `convmerge.fetch` is public API as a module: `load_manifest`,
  `run_manifest`, `Manifest`, `DatasetEntry`, `Defaults`, `AuthConfig`,
  `TokenSpec`, `FetchResult`.
- Top-level exports: `split_by_turns` and `analyze_turn_distribution` (the
  `turns` command), and the types public functions return or raise:
  `MixResult`, `JsonlLine`, `JsonlDecodeError`, `ReadStats`.
- `"version": 1` in the JSON of `convert --report`, `validate`, and the
  recipe `report.json` (fields are only ever added within a version).
- Contract tests (`tests/contract/`): snapshots of public signatures and CLI
  flags, files written by 0.8.0 that must keep working (the recipe lock is
  honoured and reproduces 0.8's output byte for byte), and exit codes.

### Changed

- Exit codes follow one rule — `0` success, `1` the work failed, `2` invalid
  invocation or configuration: a missing `fetch` manifest or `mix` config, an
  unsupported `fetch` URL, and an invalid fetch manifest now exit 2
  (previously 1 or a traceback).
- Development status is now Beta.

### Deprecated

Each warns with `DeprecationWarning` and is removed in 1.0:

- Re-exports from `convmerge.normalize` that are not public API:
  `load_jsonl`, `count_turns`, `is_single_turn`, `detect_jsonl_shape`,
  `iter_json_records`, `key_frequency`, `is_uniform_schema`,
  `single_turn_to_multi_turn_record`, `multi_turn_to_single_turn_record`.
- `load_jsonl()` returning `[]` when a line is malformed; use
  `convmerge.iter_jsonl()`.
- `convmerge.fetch.classify_entry`, `sanitize_name`, `redact_url`,
  `resolve_token` (internal helpers).
- `convmerge.cli.FETCH_FILE_EXTENSIONS` and `SIDECAR_SUFFIXES`.
- `convmerge.convert.iter_converted_lines`.

### Fixed

- An invalid YAML recipe, mix config, preset, or manifest crashed with a
  traceback; it is now reported as an error (exit 2), and `load_recipe()`
  raises `RecipeError` for it.
- `turns` and `dedupe` on a missing input file print an error (exit 1)
  instead of a traceback; `turns --single-out` without `--multi-out` fails
  before printing a report.

## [0.8.0] - 2026-09-26

Reproducible pipelines. No breaking changes: existing commands behave
exactly as in 0.7.

### Added

- `convmerge run recipe.yaml`: a declarative recipe chains fetch (manifest
  entry fields, `{manifest, name}` to reuse an existing fetch manifest, or a
  local `path`) → normalize → convert per source, then mix → dedupe into the
  output. Each step calls the same function as the CLI, so a recipe produces
  byte-for-byte the output of the hand-typed commands (verified on Stanford
  Alpaca, KoAlpaca, LLaMA-Factory tool-calling and DPO data).
- `recipe.lock.json`: per-step options, convmerge version, and input/output
  SHA-256 digests (cached by size and mtime). Re-runs repeat only steps whose
  options, inputs, or version changed or whose output was modified; a re-run
  that reproduces its output does not propagate downstream.
- `run --plan` (what would run and why), `--frozen` (exit 1 unless up to
  date, for CI), `--force [STEP|source|kind ...]`, and `--init` (commented
  template). `build/report.json` collects per-step stats and drop reasons.
- Steps write through hidden `.part` paths, so a failed run keeps previous
  outputs and the lock file intact. Recipe errors name the exact key.
- `convmerge.normalize.files.normalize_path()`: directory normalization as a
  library function (used by the CLI and recipes).
- Docs: `docs/recipes.md`; `docs/api.md` covers `convmerge.recipe`.

### Changed

- CI type-checks with mypy (`mypy` is part of the `dev` extra); fixed the
  handful of issues it found.

## [0.7.0] - 2026-09-26

Scale and extension. See [docs/migration-0.7.md](docs/migration-0.7.md);
`convert` output is byte-identical to 0.6.0 (checked on 17 real-dataset
cases, with and without `--workers`).

### Added

- `convert --workers N`: parallel conversion over a bounded window of chunks;
  output order, stats, drop reports, and `--on-invalid fail` errors are
  identical to a single process (200k rows on 4 cores: 6.4 s → 1.7 s).
- `convert --preference chosen|rejected` (preset
  `adapter_options.preference`): fold the chosen or rejected answer of
  preference data into the conversation — LLaMA-Factory ranking, HH-RLHF
  transcripts, UltraFeedback-binarized lists, TRL prompt + continuation.
- `fetch` sampling (#29): `max_rows` per manifest entry or `--max-rows N`.
  HF entries stream only N rows; raw and tree line files stop after N lines.
  Completion markers record `max_rows`, so a sample never satisfies a full
  fetch.
- `fetch` resolves Git LFS pointers from `raw.githubusercontent.com` (and
  tree files) through the Git LFS batch API — no clone; the token is sent to
  `github.com` only, never to the object store.
- Plugins: `convmerge.adapters` / `convmerge.emitters` entry points,
  `register_adapter()` / `register_emitter()`, `available_adapters()` /
  `available_formats()`, and a `convmerge formats` command.
- Public Python API: `from convmerge import ...` for everything in
  `convmerge.__all__` (lazy imports), documented in `docs/api.md`.
- `benchmarks/bench.py` (time and peak memory per command) and a design
  proposal for declarative recipes (`docs/design/recipe.md`, targeted at 0.8).

### Changed

- **Breaking:** `mix` streams sources (sampler `v2`, #24): memory is bounded
  by the sample (or one ~25k-line shuffle bucket when merging everything)
  instead of the inputs — three 119 MB sources: 405 MB → 30 MB peak. The same
  seed selects different lines than 0.6, and `--oversample` repeats every
  record evenly. `--sampler v1` reproduces 0.6 mixes exactly; `.mix.json`
  records the sampler.
- `fetch` raw downloads stream to disk instead of being held in memory.
- `convmerge.cli` is split into per-command modules (entry point and help
  text unchanged).

### Fixed

- `fetch` URL shortcuts no longer name files `x.jsonl.jsonl`.
- `--max-rows 0` / `--workers 0` are rejected instead of writing an empty
  file / silently running single-process.

## [0.6.0] - 2026-09-26

Output changes and how to get the 0.5 behavior back are listed in
[docs/migration-0.6.md](docs/migration-0.6.md). Validated as 0.6.0rc1 on
real datasets (Stanford Alpaca, KoAlpaca, LLaMA-Factory tool-calling and
multimodal demos, FastChat, OpenAI Cookbook samples) before release.

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
- `chat` / `auto` flat Q&A records: `problem` and `query` count as the
  question, and `completion` / `solution` as the answer. Output keys are tried
  in the order `output`, `response`, `completion`, `solution`, `answer`, so a
  full `solution` is kept instead of a short final `answer` (e.g. LIMO,
  NuminaMath), and `problem`/`solution` (MATH) and `query`/`response`
  (MetaMathQA) records are no longer dropped. Found while validating
  0.6.0rc1 on real datasets.
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
