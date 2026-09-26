"""The package version is declared twice; keep the copies in sync."""

from __future__ import annotations

import re
from pathlib import Path

import convmerge

ROOT = Path(__file__).parents[1]


def test_version_matches_pyproject() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version = "([^"]+)"', text, flags=re.M)
    assert m is not None
    assert convmerge.__version__ == m.group(1)


def test_changelog_has_section_for_version() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{convmerge.__version__}]" in changelog
