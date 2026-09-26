"""The covered public surface only changes on purpose (docs/stability.md).

A failure here means a public signature or CLI flag changed. If that is
intended, regenerate with ``CONVMERGE_UPDATE_SNAPSHOTS=1 pytest tests/contract``
and describe the change in CHANGELOG.md (a removal needs a deprecation first).
"""

from __future__ import annotations

import difflib
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.contract._describe import describe_api, describe_cli

SNAPSHOTS = Path(__file__).parent / "snapshots"
UPDATE = os.environ.get("CONVMERGE_UPDATE_SNAPSHOTS") == "1"


@pytest.mark.parametrize(
    ("name", "render"), [("public_api.txt", describe_api), ("cli.txt", describe_cli)]
)
def test_surface_matches_snapshot(name: str, render: Callable[[], str]) -> None:
    actual = render()
    path = SNAPSHOTS / name
    if UPDATE:
        path.write_text(actual, encoding="utf-8")
        return
    expected = path.read_text(encoding="utf-8")
    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                f"snapshots/{name}",
                "current",
            )
        )
        pytest.fail(f"public surface changed:\n{diff}\n{__doc__}")


def test_every_public_source_module_uses_future_annotations() -> None:
    # Keeps rendered annotations identical across Python versions.
    src = Path(__file__).parents[2] / "src" / "convmerge"
    missing = [
        str(p.relative_to(src))
        for p in sorted(src.rglob("*.py"))
        if p.name != "__main__.py"
        and "from __future__ import annotations" not in p.read_text(encoding="utf-8")
    ]
    assert missing == []
