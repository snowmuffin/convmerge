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


def test_public_api_resolves() -> None:
    import importlib

    assert sorted(convmerge.__all__) == sorted(["__version__", *convmerge._EXPORTS])
    for name in convmerge.__all__:
        assert getattr(convmerge, name) is not None, name
    with __import__("pytest").raises(AttributeError):
        convmerge.not_a_real_name  # noqa: B018
    # `import convmerge` stays light: no submodule loaded until first use.
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", "import convmerge, sys; print('convmerge.convert' in sys.modules)"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert out == "False"
    assert importlib.import_module("convmerge").convert_file.__module__ == "convmerge.convert"
