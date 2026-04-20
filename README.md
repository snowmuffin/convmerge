# convmerge

Merge heterogeneous chat-oriented JSONL sources into a **single LLM training format** (JSONL).

**Repository:** [github.com/snowmuffin/convmerge](https://github.com/snowmuffin/convmerge)  
**Status:** early development; APIs and CLI may change before 1.0.

## Install

```bash
pip install convmerge
```

(After the first release on PyPI. Until then, from a clone:)

```bash
git clone https://github.com/snowmuffin/convmerge.git
cd convmerge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
convmerge convert \
  --input samples/alpaca.jsonl \
  --output out/messages.jsonl \
  --from alpaca \
  --format messages
```

- **`--from`**: source adapter (`alpaca`, `sharegpt`, …).
- **`--format`**: output shape (`messages`, `alpaca`, …).

See [docs/format.md](docs/format.md) for schemas and how to add adapters.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). CI runs on every push to `main` (Ruff + pytest on Python 3.10–3.12).

## PyPI release (maintainers)

1. On [pypi.org](https://pypi.org), create the `convmerge` project (or claim the name) and add **trusted publishing** for this GitHub repo (`snowmuffin/convmerge`, workflow `publish.yml`, environment optional per your PyPI settings).
2. Tag and push: `git tag v0.1.0 && git push origin v0.1.0` — this runs [`.github/workflows/publish.yml`](.github/workflows/publish.yml).

## Changelog

[CHANGELOG.md](CHANGELOG.md)

## License

MIT
