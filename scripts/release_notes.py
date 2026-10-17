"""Helpers for the GitHub Releases workflow (``.github/workflows/releases.yml``).

``python scripts/release_notes.py notes VERSION`` prints the CHANGELOG section
for VERSION as release notes; ``assets VERSION DIR [--wait SECONDS]`` downloads
the files PyPI serves for VERSION into DIR, checking each against PyPI's
sha256; ``order TAG ...`` prints tags oldest first; ``latest TAG ...`` prints
the newest stable tag; ``is-prerelease VERSION`` exits 0 for a release
candidate. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYPI_JSON = "https://pypi.org/pypi/convmerge/{version}/json"
BLOB = "https://github.com/snowmuffin/convmerge/blob/v{version}/"
_VERSION = re.compile(r"v?(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?")
_LINK_DEFINITION = re.compile(r"\[[^\]]+\]: \S+")
# A relative Markdown link target, e.g. ``](docs/fetch.md)``.
_RELATIVE_LINK = re.compile(r"\]\((?![a-z][a-z0-9+.-]*:|#|/)([^)\s]+)\)")


def version_key(tag: str) -> tuple[int, int, int, int, int]:
    match = _VERSION.fullmatch(tag)
    if match is None:
        raise ValueError(f"not a release version: {tag!r}")
    major, minor, patch, rc = match.groups()
    # A release candidate sorts before its final release.
    return (int(major), int(minor), int(patch), 0 if rc else 1, int(rc or 0))


def is_prerelease(version: str) -> bool:
    return version_key(version)[3] == 0


def order(tags: list[str]) -> list[str]:
    return sorted(tags, key=version_key)


def latest(tags: list[str]) -> str | None:
    stable = [tag for tag in tags if not is_prerelease(tag)]
    return max(stable, key=version_key) if stable else None


def notes(version: str, changelog: str) -> str:
    """Return the CHANGELOG section for ``version``, without its heading.

    A release candidate without its own section gets its final release's notes.
    Relative links point at the files as of the release's tag.
    """
    try:
        body, date = _section(version, changelog)
    except LookupError:
        if not is_prerelease(version):
            raise
        final = version.split("rc")[0]
        final_body, _ = _section(final, changelog)
        body = [f"Release candidate for convmerge {final}; the notes are those of {final}.", ""]
        body += final_body
        date = None
    blob = BLOB.format(version=version)
    text = _RELATIVE_LINK.sub(lambda m: f"]({blob}{m.group(1)})", "\n".join(body))
    footer = (
        f"PyPI: https://pypi.org/project/convmerge/{version}/ · `pip install convmerge=={version}`"
    )
    if date:
        footer = f"Published to PyPI on {date}. " + footer
    return f"{text}\n\n---\n\n{footer}\n"


def _section(version: str, changelog: str) -> tuple[list[str], str | None]:
    lines = changelog.splitlines()
    heading = re.compile(rf"## \[{re.escape(version)}\](?: - (\S+))?\s*")
    for start, line in enumerate(lines):
        match = heading.fullmatch(line)
        if match:
            break
    else:
        raise LookupError(f"CHANGELOG.md has no section for {version}")
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## [")),
        len(lines),
    )
    body = lines[start + 1 : end]
    while body and (not body[-1].strip() or _LINK_DEFINITION.fullmatch(body[-1])):
        body.pop()
    while body and not body[0].strip():
        body.pop(0)
    return body, match.group(1)


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def pypi_files(version: str, wait: float = 0, fetch: Any = _fetch) -> list[dict[str, Any]]:
    """Return PyPI's file records for ``version``, waiting up to ``wait`` seconds."""
    deadline = time.monotonic() + wait
    while True:
        try:
            files = json.loads(fetch(PYPI_JSON.format(version=version)))["urls"]
        except Exception as exc:  # noqa: BLE001 - a fresh upload can 404 for a while
            files, error = [], exc
        else:
            error = None
        if files or time.monotonic() >= deadline:
            break
        time.sleep(15)
    if not files:
        raise LookupError(f"PyPI serves no files for convmerge {version}: {error or 'empty'}")
    return files


def download_assets(
    version: str, directory: Path, wait: float = 0, fetch: Any = _fetch
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for record in pypi_files(version, wait, fetch):
        data = fetch(record["url"])
        digest = hashlib.sha256(data).hexdigest()
        if digest != record["digests"]["sha256"]:
            raise ValueError(f"{record['filename']}: sha256 {digest} does not match PyPI")
        path = directory / Path(record["filename"]).name
        path.write_bytes(data)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("notes")
    p.add_argument("version")
    p.add_argument("--changelog", type=Path, default=ROOT / "CHANGELOG.md")
    p = sub.add_parser("assets")
    p.add_argument("version")
    p.add_argument("directory", type=Path)
    p.add_argument("--wait", type=float, default=0)
    for name in ("order", "latest"):
        sub.add_parser(name).add_argument("tags", nargs="*")
    sub.add_parser("is-prerelease").add_argument("version")
    args = parser.parse_args(argv)

    try:
        if args.command == "notes":
            sys.stdout.write(notes(args.version, args.changelog.read_text(encoding="utf-8")))
        elif args.command == "assets":
            for path in download_assets(args.version, args.directory, args.wait):
                print(path)
        elif args.command == "order":
            print("\n".join(order(args.tags)))
        elif args.command == "latest":
            print(latest(args.tags) or "")
        else:
            return 0 if is_prerelease(args.version) else 1
    except (LookupError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
