"""Smoke-test an installed wheel, run with the isolated wheel environment's Python."""

from __future__ import annotations

import importlib.metadata
import json
import tempfile
from pathlib import Path

import convmerge
from convmerge import convert_file, normalize_to_jsonl


def main() -> None:
    version = importlib.metadata.version("convmerge")
    assert convmerge.__version__ == version
    assert "site-packages" in str(Path(convmerge.__file__).resolve())
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, normalized, output = (
            root / "in.json",
            root / "normalized.jsonl",
            root / "train.jsonl",
        )
        source.write_text('[{"instruction":"q","output":"a"}]', encoding="utf-8")
        assert normalize_to_jsonl(source, normalized) == 1
        assert convert_file(normalized, output, adapter_name="auto", output_format="messages") == (
            1,
            1,
        )
        assert json.loads(output.read_text(encoding="utf-8"))["messages"][-1]["content"] == "a"
        before = normalized.read_bytes()
        source.write_text('{"x":1}\n{broken\n', encoding="utf-8")
        try:
            normalize_to_jsonl(source, normalized)
        except ValueError:
            pass
        else:
            raise AssertionError("broken input was not rejected")
        assert normalized.read_bytes() == before
        long_record = {"instruction": "long", "output": "x" * 70000}
        source.write_text("\n" * 70000 + json.dumps(long_record) + "\n", encoding="utf-8")
        assert normalize_to_jsonl(source, normalized) == 1
        assert json.loads(normalized.read_text(encoding="utf-8")) == long_record

        from convmerge.io import SamePathError
        from convmerge.recipe import RecipeError, load_lock, parse_recipe, run

        recipe = parse_recipe(
            {
                "output": str(source),
                "workdir": str(root / "build"),
                "sources": {
                    "a": {"path": str(source), "normalize": False, "convert": {"from": "auto"}}
                },
            },
            path=root / "recipe.json",
        )
        before = source.read_bytes()
        try:
            run(recipe)
        except SamePathError:
            pass
        else:
            raise AssertionError("recipe source/output collision was accepted")
        assert source.read_bytes() == before and not recipe.workdir.exists()
        lock = root / "bad.lock.json"
        lock.write_text("[]", encoding="utf-8")
        try:
            load_lock(lock)
        except RecipeError:
            pass
        else:
            raise AssertionError("malformed lock was accepted")
        assert lock.read_text(encoding="utf-8") == "[]"
    print(json.dumps({"version": version, "import_path": convmerge.__file__, "smoke": "passed"}))


if __name__ == "__main__":
    main()
