# Contributing to convmerge

Thanks for your interest in improving `convmerge`. This project is small and
focused on **one thing**: turning heterogeneous chat / instruct datasets into
clean JSONL for supervised fine-tuning. Contributions that keep it sharp in
that role are very welcome.

Upstream: [github.com/snowmuffin/convmerge](https://github.com/snowmuffin/convmerge)
— fork it and open a PR from a feature branch.

## What we want / don't want

**Good fits for a PR (no issue required):**

- New **adapter** for a public chat / instruct schema with a clear reference
  (ShareGPT variants, OpenOrca, Dolly, OpenAssistant, Firefly, WizardLM, …).
- New **emitter** that reshapes the internal message list into another
  standard JSONL layout (e.g. ChatML-like raw record).
- New **fetch** backend for a public dataset host that can be implemented
  with stdlib `urllib` or a thin optional dependency (GitLab, Zenodo,
  Kaggle, …), behind an extras marker.
- Docs / examples / recipes for real public datasets.
- Bug fixes with a failing-then-passing test.
- Performance improvements with a benchmark / rationale.

**Please open an issue first for:**

- New runtime dependency in the core package.
- Changes to public API shapes (`TrainingExample`, `ChatMessage`, CLI flag
  names).
- New CLI subcommand or removal of an existing one.
- Re-introducing anything listed in the
  [Out of scope](README.md#out-of-scope) section of the README.

Out-of-scope PRs are closed quickly — not because the idea is bad, but
because that functionality belongs in a different layer of your pipeline.

## Setup

```bash
git clone https://github.com/<your-username>/convmerge.git
cd convmerge
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev,fetch-all,parquet]"
```

The `[dev,fetch-all,parquet]` extras install everything needed to run the
full test suite, including the optional `pyarrow` / `datasets` paths.

## Checks

Run these locally before opening a PR — CI runs the same three commands on
Python 3.10, 3.11, and 3.12:

```bash
ruff check src tests
ruff format --check src tests
pytest -q
```

Autofix style issues with:

```bash
ruff check --fix src tests
ruff format src tests
```

## Code conventions

- **All code comments, docstrings, CLI help strings, and error messages are
  in English.** Non-English strings inside test fixtures (e.g. Korean sample
  text) are fine when the test is specifically exercising multi-language
  handling.
- Follow the existing 100-character line length (`tool.ruff.line-length`).
- Keep the **core package dependency-free**. New runtime deps must go
  behind a named extras marker in `pyproject.toml`.
- Prefer small, composable functions. Public entry points live in
  `src/convmerge/__init__.py`; keep them stable.
- Don't add comments that only narrate what the code does (`# loop over
  lines`, `# return result`). Comments should explain intent or non-obvious
  trade-offs.

## Adding a new adapter (walkthrough)

Adapters convert one input record shape into a stream of `TrainingExample`.
A minimal adapter is ~30 lines.

1. Create `src/convmerge/adapters/<name>.py` exporting
   `iter_from_<name>_line(record: dict) -> Iterator[TrainingExample]`.
   Look at `alpaca.py` or `sharegpt.py` for the shape.
2. Register the adapter in `src/convmerge/adapters/__init__.py`:

   ```python
   from .your_adapter import iter_from_<name>_line
   ADAPTERS["<name>"] = iter_from_<name>_line
   ```

3. Add a fixture under `tests/fixtures/<name>.jsonl` with 3–5 realistic
   lines (redact anything sensitive).
4. Add tests in `tests/test_adapters.py` (or a new
   `tests/test_<name>_adapter.py`) that cover happy path + at least one
   edge case (empty content, missing optional fields, multi-turn).
5. Document the adapter in `docs/format.md` under the "Adapters" section.
6. Add a one-line entry to `CHANGELOG.md` under `## [Unreleased]`.

Emitters follow the same pattern under `src/convmerge/emitters/`.

## Pull requests

- Keep changes **focused** — one feature or fix per PR when possible.
- Add or update **tests** for any behaviour change.
- Update **docs** if you touch public surface (`README.md`,
  `docs/format.md`, `docs/fetch.md`).
- Fill out the PR template checklist. It's short on purpose.
- PRs with a red CI badge usually won't get reviewed. Fix lint / format /
  tests first.

### Review expectations

Maintainers aim to leave a first response within **3 working days**. If a
week goes by with no reply, it is fine to ping the PR with a short comment
— we may have missed the notification.

## Issues

Use one of the templates under `.github/ISSUE_TEMPLATE/`:

- **Bug report** — incorrect conversion, crash, or unexpected output.
- **New adapter / emitter** — a public chat / instruct schema that is not
  yet supported. Include a minimal redacted sample and the expected
  output.
- **Fetch issue** — problems with the YAML manifest or a specific
  HuggingFace / GitHub source.
- **Feature request** — anything else.

Blank issues are also enabled, but templates make triage an order of
magnitude faster.

## Labels

The repo uses labels such as `good first issue`, `help wanted`, `bug`,
`enhancement`, `adapter`, `emitter`, `fetch`, and `docs`. Pick one when
filing an issue; maintainers will relabel as needed.

## Code of Conduct

Participation in this project is governed by the
[Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you are
expected to uphold it.
