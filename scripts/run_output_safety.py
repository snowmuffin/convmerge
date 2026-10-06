"""Run the cross-platform safety suite and preserve exact failure/source evidence."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "safety-results"
SUITE = [
    "tests/test_atomic_output.py",
    "tests/test_output_safety.py",
    "tests/test_output_integrations.py",
    "tests/test_normalize_jsonl.py",
    "tests/test_tables.py",
    "tests/test_same_path.py",
    "tests/test_remaining_issues.py",
    "tests/test_correctness_boundaries.py",
]


class Evidence:
    def pytest_collection_finish(self, session):
        (RESULTS / "collection.json").write_text(
            json.dumps([item.nodeid for item in session.items], indent=2), encoding="utf-8"
        )

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        report = (yield).get_result()
        if not report.failed:
            return
        entry = {"nodeid": item.nodeid, "phase": report.when, "traceback": str(report.longrepr)}
        path = Path(str(item.path))
        entry["source_path"] = str(path)
        entry["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        tmp = item.funcargs.get("tmp_path")
        entry["fixtures"] = []
        if tmp is not None:
            for candidate in sorted(Path(tmp).rglob("*")):
                if candidate.is_file() and not candidate.is_symlink():
                    data = candidate.read_bytes()
                    entry["fixtures"].append(
                        {
                            "path": str(candidate.relative_to(tmp)),
                            "size": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "head_hex": data[:4096].hex(),
                        }
                    )
        with (RESULTS / "failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        print("\nSAFETY_FAILURE " + json.dumps(entry), flush=True)


def main() -> int:
    os.chdir(REPO)
    RESULTS.mkdir(exist_ok=True)
    identity = {
        "platform": platform.platform(),
        "python": sys.version,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "files": {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in SUITE},
    }
    (RESULTS / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    print("SAFETY_IDENTITY " + json.dumps(identity), flush=True)
    return int(
        pytest.main(
            [
                *SUITE,
                "-q",
                "--tb=short",
                "-p",
                "no:cacheprovider",
                "--junitxml",
                str(RESULTS / "junit.xml"),
            ],
            plugins=[Evidence()],
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
