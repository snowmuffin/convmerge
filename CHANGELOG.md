# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `CODE_OF_CONDUCT.md` based on Contributor Covenant 2.1.
- `examples/` directory with READMEs and ready-to-run `fetch` manifests
  for Alpaca, ShareGPT, and mixed HF + GitHub sources.
- New issue templates: `new_adapter.yml` (adapter / emitter request) and
  `fetch_issue.yml` (fetch manifest problems). Existing config now links
  to the contributing guide and docs.
- `py.typed` marker in the distributed wheel, so downstream projects
  pick up inline type hints via PEP 561.

### Changed

- Expanded `CONTRIBUTING.md`: scope expectations (what is / isn't
  accepted), review SLA, end-to-end walkthrough for adding a new adapter,
  clearer install / check steps with the `[fetch-all,parquet]` extras.
- Richer `pyproject.toml` metadata: more `keywords` and `classifiers`
  (topic, audience, typed), additional `project.urls` entries for
  `Changelog` and `Documentation`, and `Development Status` bumped from
  pre-alpha to alpha.
- README: added PyPI / Python / CI / downloads / CoC badges, linked the
  Code of Conduct, and pointed to `good first issue` for contributors.

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
