"""Third-party adapters / output formats via entry points and register_*()."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from convmerge import plugins
from convmerge.adapters import ADAPTERS, available_adapters, get_adapter, register_adapter
from convmerge.emitters import EMITTERS, available_formats, get_emitter, register_emitter

PLUGIN_SRC = """
from convmerge.emitters import UnrepresentableExample
from convmerge.models import ChatMessage, TrainingExample


def iter_from_qa(record):
    if "q" in record and "a" in record:
        yield TrainingExample(
            messages=[ChatMessage("user", record["q"]), ChatMessage("assistant", record["a"])],
            meta={"source": "qa"},
        )


def emit_pairs(example, options=None):
    if len(example.messages) != 2:
        raise UnrepresentableExample("unrepresentable_long")
    meta = bool(options and options.keep_meta)
    return {"p": example.messages[0].text, "r": example.messages[1].text, "meta": meta}
"""


@pytest.fixture
def plugin_path(tmp_path: Path, monkeypatch) -> Path:
    site = tmp_path / "site"
    site.mkdir()
    (site / "cm_plugin.py").write_text(PLUGIN_SRC, encoding="utf-8")
    dist = site / "cm_plugin-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: cm-plugin\nVersion: 0.1\n")
    (dist / "entry_points.txt").write_text(
        "[convmerge.adapters]\nqa = cm_plugin:iter_from_qa\nbroken = cm_plugin:missing\n"
        "alpaca = cm_plugin:iter_from_qa\n\n"
        "[convmerge.emitters]\npairs = cm_plugin:emit_pairs\n"
    )
    monkeypatch.syspath_prepend(str(site))
    plugins.reset_for_tests()
    yield site
    for name in ("qa", "broken"):
        ADAPTERS.pop(name, None)
    EMITTERS.pop("pairs", None)
    plugins.reset_for_tests()


def test_entry_points_are_discovered_lazily(plugin_path, caplog) -> None:
    assert "qa" not in ADAPTERS
    ex = next(get_adapter("qa")({"q": "hi", "a": "yo"}))
    assert ex.messages[1].content == "yo"
    assert "qa" in available_adapters() and "pairs" in available_formats()
    # Built-ins always win over an entry point with the same name.
    from convmerge.adapters.alpaca import iter_from_alpaca_line

    assert get_adapter("alpaca") is iter_from_alpaca_line
    # A plugin that fails to import is skipped with a warning.
    assert "broken" not in ADAPTERS
    assert any("broken" in r.getMessage() for r in caplog.records)


def test_plugin_emitter_gets_options_when_it_asks(plugin_path) -> None:
    from convmerge.emitters import EmitOptions

    ex = next(get_adapter("qa")({"q": "hi", "a": "yo"}))
    assert get_emitter("pairs")(ex) == {"p": "hi", "r": "yo", "meta": False}
    assert get_emitter("pairs", options=EmitOptions(keep_meta=True))(ex)["meta"] is True


def test_cli_uses_plugins_including_workers(plugin_path, tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text(
        "".join(json.dumps({"q": f"q{i}", "a": f"a{i}"}) + "\n" for i in range(5)), encoding="utf-8"
    )
    out = tmp_path / "out.jsonl"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(plugin_path), *sys.path])}
    for workers in ("1", "2"):
        proc = subprocess.run(
            [sys.executable, "-m", "convmerge", "convert", "-i", str(src), "-o", str(out),
             "--from", "qa", "-f", "pairs", "--workers", workers],
            env=env, capture_output=True, text=True,
        )  # fmt: skip
        assert proc.returncode == 0, proc.stderr
        assert [json.loads(x)["p"] for x in out.read_text().splitlines()] == [
            f"q{i}" for i in range(5)
        ]
    listing = subprocess.run(
        [sys.executable, "-m", "convmerge", "formats"], env=env, capture_output=True, text=True
    ).stdout
    assert "qa (plugin)" in listing and "pairs (plugin)" in listing


def test_register_adapter_and_emitter() -> None:
    def fn(record):
        yield from ()

    register_adapter("tmp_adapter", fn)
    try:
        assert get_adapter("tmp_adapter") is fn
        with pytest.raises(ValueError, match="already registered"):
            register_adapter("tmp_adapter", fn)
        register_adapter("tmp_adapter", fn, replace=True)
    finally:
        ADAPTERS.pop("tmp_adapter", None)
    with pytest.raises(ValueError, match="already registered"):
        register_emitter("messages", lambda ex: {})
