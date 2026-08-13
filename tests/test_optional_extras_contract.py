"""Keep the documented optional extras aligned with the package metadata."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from convmerge.cli import main

ROOT = Path(__file__).parents[1]
RUNTIME_EXTRAS = {"fetch", "fetch-hf", "fetch-all", "parquet", "preset", "all"}


def _optional_dependencies() -> dict[str, list[str]]:
    lines = (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    in_section = False
    result: dict[str, list[str]] = {}
    for line in lines:
        stripped = line.strip()
        if stripped == "[project.optional-dependencies]":
            in_section = True
            continue
        if in_section and stripped.startswith("["):
            break
        if in_section and "=" in line:
            name, raw = line.split("=", 1)
            value = ast.literal_eval(raw.strip())
            if isinstance(value, list):
                result[name.strip()] = value
    return result


def _help(argv: list[str], capsys) -> str:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0
    return capsys.readouterr().out


def test_runtime_extras_match_locked_set() -> None:
    assert set(_optional_dependencies()) - {"dev"} == RUNTIME_EXTRAS


def test_readme_lists_every_runtime_extra() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"convmerge\[([^\]]+)\]", readme))
    assert RUNTIME_EXTRAS <= documented


def test_top_level_help_lists_every_runtime_extra(capsys) -> None:
    output = _help(["--help"], capsys)
    for extra in RUNTIME_EXTRAS:
        assert f"[{extra}]" in output


def test_subcommand_help_mentions_its_runtime_extra(capsys) -> None:
    expected = {
        "convert": "[preset]",
        "normalize": "[parquet]",
        "fetch": "[fetch]",
        "preset": "[preset]",
    }
    for command, extra in expected.items():
        output = _help([command, "--help"], capsys)
        assert extra in output
    fetch_help = _help(["fetch", "--help"], capsys)
    for extra in ("[fetch-all]", "[fetch-hf]", "[all]"):
        assert extra in fetch_help


def test_error_hints_include_narrow_and_umbrella_extras() -> None:
    expected = {
        "src/convmerge/normalize/parquet.py": "[parquet]",
        "src/convmerge/preset.py": "[preset]",
        "src/convmerge/fetch/manifest.py": "[fetch]",
        "src/convmerge/fetch/hf.py": "[fetch-all]",
    }
    for relative, narrow in expected.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert narrow in text
        assert "[all]" in text


def test_fetch_issue_template_covers_fetch_runtime_options() -> None:
    template = (ROOT / ".github/ISSUE_TEMPLATE/fetch_issue.yml").read_text(encoding="utf-8")
    for marker in ("[fetch]", "[fetch-hf]", "[fetch-all]", "[all]", "[parquet]"):
        assert marker in template
