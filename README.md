# convmerge

Merge heterogeneous chat-oriented JSONL sources into a **single LLM training format** (JSONL).

**Status:** early development; APIs and CLI may change before 1.0.

## Install

```bash
pip install convmerge
```

(After the first release on PyPI. Until then, from a clone:)

```bash
git clone https://github.com/<your-org>/convmerge.git
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

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Connect to GitHub

```bash
git remote add origin https://github.com/<org>/convmerge.git
git push -u origin main
```

Enable **GitHub Actions** in the repo settings. For PyPI releases, register this repository as a **trusted publisher** on [pypi.org](https://pypi.org) for the `convmerge` project, then push a tag `v0.1.0` to trigger [`.github/workflows/publish.yml`](.github/workflows/publish.yml).

## Changelog

[CHANGELOG.md](CHANGELOG.md)

## License

MIT
