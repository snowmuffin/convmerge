"""scripts/release_notes.py builds GitHub Release notes and assets (offline here)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_release_notes_script", ROOT / "scripts" / "release_notes.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rn = _load()

CHANGELOG = """# Changelog

## [Unreleased]

## [1.1.0] - 2026-09-29

Intro line.

### Added

- `thing`: does it.

[1.1.0]: https://pypi.org/project/convmerge/1.1.0/

## [1.0.0rc1] - 2026-09-28

### Fixed

- rc fix.
"""


def test_notes_take_only_the_section_without_heading_or_link_definition() -> None:
    text = rn.notes("1.1.0", CHANGELOG)
    assert text.startswith("Intro line.\n\n### Added\n\n- `thing`: does it.\n\n---\n")
    assert "## [" not in text and "[1.1.0]:" not in text and "rc fix" not in text
    assert "Published to PyPI on 2026-09-29." in text
    assert "`pip install convmerge==1.1.0`" in text
    assert rn.notes("1.0.0rc1", CHANGELOG).startswith("### Fixed\n\n- rc fix.\n")


def test_a_release_candidate_without_a_section_gets_the_final_notes() -> None:
    text = rn.notes("1.1.0rc2", CHANGELOG)
    assert text.startswith("Release candidate for convmerge 1.1.0; the notes are those of 1.1.0.")
    assert "- `thing`: does it." in text and "Published to PyPI" not in text


def test_relative_links_point_at_the_tag() -> None:
    changelog = (
        "## [2.0.0] - 2026-10-20\n\nSee [docs](docs/a.md), [web](https://x.y/z), [top](#t).\n"
    )
    text = rn.notes("2.0.0", changelog)
    assert "[docs](https://github.com/snowmuffin/convmerge/blob/v2.0.0/docs/a.md)" in text
    assert "[web](https://x.y/z)" in text and "[top](#t)" in text


def test_notes_for_a_missing_version_fail() -> None:
    with pytest.raises(LookupError):
        rn.notes("1.0.0", CHANGELOG)
    assert rn.main(["notes", "9.9.9", "--changelog", str(ROOT / "CHANGELOG.md")]) == 2


def test_every_tagged_version_in_the_real_changelog_has_notes() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for version in ("1.6.5", "1.0.0rc1", "0.6.0rc1", "0.3.3"):
        assert rn.notes(version, changelog).strip()


def test_order_latest_and_prerelease() -> None:
    tags = ["v1.0.0", "v0.10.0", "v1.0.0rc2", "v0.9.0", "v1.0.0rc1", "v1.1.0rc1"]
    assert rn.order(tags) == ["v0.9.0", "v0.10.0", "v1.0.0rc1", "v1.0.0rc2", "v1.0.0", "v1.1.0rc1"]
    assert rn.latest(tags) == "v1.0.0"
    assert rn.latest(["v1.0.0rc1"]) is None
    assert rn.is_prerelease("1.0.0rc1") and not rn.is_prerelease("1.0.0")
    with pytest.raises(ValueError):
        rn.order(["v1.0"])


def test_assets_match_pypi_digests(tmp_path: Path) -> None:
    files = {"https://f/convmerge-1.0.0.tar.gz": b"sdist", "https://f/convmerge-1.0.0.whl": b"w"}
    index = {
        "urls": [
            {
                "url": url,
                "filename": url.rsplit("/", 1)[1],
                "digests": {"sha256": hashlib.sha256(data).hexdigest()},
            }
            for url, data in files.items()
        ]
    }

    def fetch(url: str) -> bytes:
        return json.dumps(index).encode() if url.endswith("/json") else files[url]

    written = rn.download_assets("1.0.0", tmp_path / "a", fetch=fetch)
    assert sorted(p.read_bytes() for p in written) == [b"sdist", b"w"]

    files["https://f/convmerge-1.0.0.whl"] = b"tampered"
    with pytest.raises(ValueError, match="does not match PyPI"):
        rn.download_assets("1.0.0", tmp_path / "b", fetch=fetch)


def test_assets_for_an_unpublished_version_fail() -> None:
    def fetch(url: str) -> bytes:
        raise OSError("404")

    with pytest.raises(LookupError, match="no files"):
        rn.pypi_files("9.9.9", fetch=fetch)
