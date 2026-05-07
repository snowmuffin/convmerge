# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Normalize JSONL inputs with a leading UTF-8 BOM, CRLF line endings, and trailing whitespace; report trailing-comma JSONL lines with file and line context.

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
