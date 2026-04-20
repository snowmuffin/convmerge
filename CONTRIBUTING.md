# Contributing to convmerge

## Setup

```bash
git clone <your-fork-url>
cd convmerge
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev]"
```

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

## GitHub labels (optional)

Repository maintainers can create labels such as:

| Label | Use |
|-------|-----|
| `good first issue` | Small docs/tests |
| `adapter` | New or changed source adapter |
| `bug` | Incorrect output or crash |

If you use GitHub CLI: `gh label create "good first issue" --color "0e8a16"`.

## Connect this repo to GitHub

If you created the project only on disk:

```bash
cd convmerge
git remote add origin https://github.com/<org>/convmerge.git
git push -u origin main
```

Then enable **Actions** and, for releases, register **PyPI trusted publishing** for the repository (see [README](README.md)).
