"""The recipes under examples/ stay valid as the schema evolves."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from convmerge.recipe import build_steps, load_recipe  # noqa: E402

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").rglob("recipe*.yaml"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: str(p.relative_to(p.parents[2])))
def test_example_recipe_is_valid(path: Path) -> None:
    recipe = load_recipe(path)
    kinds = [s.kind for s in build_steps(recipe)]
    assert "convert" in kinds and kinds[-1] in ("split", "output", "dedupe", "tokens")


def test_there_are_examples() -> None:
    assert len(EXAMPLES) >= 3
