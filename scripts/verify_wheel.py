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
    print(json.dumps({"version": version, "import_path": convmerge.__file__, "smoke": "passed"}))


if __name__ == "__main__":
    main()
