"""Small file-I/O smoke with exact input bytes and checked-out source identities."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path

import convmerge
from convmerge import normalize_to_jsonl
from convmerge.normalize.jsonl import detect_jsonl_shape


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    identity = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "version": convmerge.__version__,
        "import_path": convmerge.__file__,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "sources": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in [
                "src/convmerge/_output.py",
                "tests/test_normalize_jsonl.py",
                "tests/test_tables.py",
            ]
        },
    }
    print(json.dumps(identity), flush=True)
    cases = [
        (b'{"x":1}\n\n{"x":2}\n', "jsonl", [{"x": 1}, {"x": 2}]),
        (b'\xef\xbb\xbf{"x":1}  \r\n{"x":2}\t\r\n', "jsonl", [{"x": 1}, {"x": 2}]),
        (b'\xef\xbb\xbf{"x":1}\r\n', "single_line", [{"x": 1}]),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for repeat in range(3):
            for index, (raw, shape, expected) in enumerate(cases):
                src, dst = root / f"in-{index}.jsonl", root / f"out-{index}.jsonl"
                src.write_bytes(raw)
                before = src.read_bytes()
                observed = detect_jsonl_shape(src)
                count = normalize_to_jsonl(src, dst)
                after = dst.read_bytes()
                rows = [json.loads(s) for s in dst.read_text(encoding="utf-8").splitlines()]
                record = {
                    "repeat": repeat,
                    "case": index,
                    "input_hex": before.hex(),
                    "shape": observed,
                    "count": count,
                    "output_hex": after.hex(),
                }
                print(json.dumps(record), flush=True)
                assert observed == shape, record
                assert count == len(expected) and rows == expected, record
                assert src.read_bytes() == raw, record
    print("portable output smoke passed", flush=True)


if __name__ == "__main__":
    main()
