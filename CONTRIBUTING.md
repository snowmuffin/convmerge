# Contributing to convmerge

Upstream: [github.com/snowmuffin/convmerge](https://github.com/snowmuffin/convmerge) — fork it, open a PR from a feature branch.

## Setup

```bash
git clone https://github.com/<your-username>/convmerge.git
cd convmerge
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev]"
```

(Add your fork URL as `origin` if you cloned via SSH or a different host.)

## Checks

Run before opening a PR:

```bash
ruff check src tests
ruff format src tests
pytest
```

CI runs the same commands on Python 3.10–3.12.

## Pull requests

- Keep changes focused (one feature or fix per PR when possible).
- Add or update tests for behavior changes.
- Update [docs/format.md](docs/format.md) if you add adapters or output formats.

## Issues

Use the templates under `.github/ISSUE_TEMPLATE/`. For a **new source format**, include:

- A **minimal** redacted JSONL sample (input).
- Expected output line(s) after conversion.

## GitHub labels

The repo uses labels such as `good first issue`, `bug`, `enhancement`, and `adapter` (new or changed source adapter). Pick one when filing an issue.
