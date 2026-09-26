"""Golden regression tests: real-world-shaped inputs → exact convert output.

Each case converts one fixture under ``tests/golden/inputs`` and compares the
result byte-for-byte with ``tests/golden/expected/<case>.jsonl``, plus the drop
counters in ``<case>.stats.json``. A behavior change therefore shows up as a
diff of the expected files, which is what reviewers should read.

Regenerate after an intentional change with::

    CONVMERGE_UPDATE_GOLDEN=1 pytest tests/test_golden.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import warnings
from pathlib import Path

import pytest

from convmerge.config import build_convert_config
from convmerge.convert import ConvertStats, convert_with_config

GOLDEN = Path(__file__).parent / "golden"
UPDATE = os.environ.get("CONVMERGE_UPDATE_GOLDEN", "").strip().lower() in {"1", "true", "yes"}

# case id -> (input fixture, adapter, output format, options or None). Options
# hold --adapter-kwargs; an "emit" entry holds EmitOptions overrides.
CASES: dict[str, tuple[str, str, str, dict | None]] = {
    "openai_chat.chat.messages": ("openai_chat", "chat", "messages", None),
    "openai_chat.chat.alpaca": ("openai_chat", "chat", "alpaca", None),
    "openai_multimodal.chat.messages": ("openai_multimodal", "chat", "messages", None),
    "sharegpt_basic.sharegpt.messages": ("sharegpt_basic", "sharegpt", "messages", None),
    "sharegpt_basic.sharegpt-full.messages": (
        "sharegpt_basic",
        "sharegpt",
        "messages",
        {"sharegpt": {"turn_mode": "full"}},
    ),
    "sharegpt_basic.sharegpt-full.alpaca": (
        "sharegpt_basic",
        "sharegpt",
        "alpaca",
        {"sharegpt": {"turn_mode": "full"}},
    ),
    "sharegpt_basic.chat.messages": ("sharegpt_basic", "chat", "messages", None),
    "sharegpt_tools.sharegpt-full.messages": (
        "sharegpt_tools",
        "sharegpt",
        "messages",
        {"sharegpt": {"turn_mode": "full"}},
    ),
    "sharegpt_tools.chat.messages": ("sharegpt_tools", "chat", "messages", None),
    "sharegpt_multimodal.sharegpt-full.messages": (
        "sharegpt_multimodal",
        "sharegpt",
        "messages",
        {"sharegpt": {"turn_mode": "full"}},
    ),
    "sharegpt_multimodal.chat.messages": ("sharegpt_multimodal", "chat", "messages", None),
    "alpaca.alpaca.messages": ("alpaca", "alpaca", "messages", None),
    "alpaca.alpaca.alpaca": ("alpaca", "alpaca", "alpaca", None),
    "alpaca.chat.messages": ("alpaca", "chat", "messages", None),
    "chat_variants.chat.messages": ("chat_variants", "chat", "messages", None),
    "chat_variants.chat-pairwise-both.messages": (
        "chat_variants",
        "chat",
        "messages",
        {"chat": {"pairwise_mode": "both"}},
    ),
    "messy.chat.messages": ("messy", "chat", "messages", None),
    "sharegpt_basic.sharegpt-full.alpaca-history": (
        "sharegpt_basic",
        "sharegpt",
        "alpaca",
        {"emit": {"alpaca_multiturn": "history"}},
    ),
    "openai_chat.chat.alpaca-drop": (
        "openai_chat",
        "chat",
        "alpaca",
        {"emit": {"alpaca_multiturn": "drop"}},
    ),
    "sharegpt_multimodal.sharegpt-full.messages-meta": (
        "sharegpt_multimodal",
        "sharegpt",
        "messages",
        {"emit": {"keep_meta": True}},
    ),
    "sharegpt_tools.sharegpt-full.messages-objectargs": (
        "sharegpt_tools",
        "sharegpt",
        "messages",
        {"emit": {"tool_arguments": "object"}},
    ),
}


def _run(case: str, tmp_path: Path) -> tuple[str, dict]:
    fixture, adapter, fmt, options = CASES[case]
    kwargs = dict(options or {})
    emit = kwargs.pop("emit", None)
    cfg = build_convert_config(
        adapter=adapter,
        output_format=fmt,
        adapter_kwargs_json=json.dumps(kwargs) if kwargs else None,
        emit_overrides=emit,
    )
    out = tmp_path / "out.jsonl"
    stats = ConvertStats()
    with warnings.catch_warnings():
        # Deprecation notices are covered by their own tests; golden files
        # capture output only.
        warnings.simplefilter("ignore", FutureWarning)
        convert_with_config(GOLDEN / "inputs" / f"{fixture}.jsonl", out, cfg, stats=stats)
    return out.read_text(encoding="utf-8"), dataclasses.asdict(stats)


@pytest.mark.parametrize("case", sorted(CASES))
def test_golden(case: str, tmp_path: Path) -> None:
    output, stats = _run(case, tmp_path)
    expected_out = GOLDEN / "expected" / f"{case}.jsonl"
    expected_stats = GOLDEN / "expected" / f"{case}.stats.json"
    stats_text = json.dumps(stats, indent=2, sort_keys=True) + "\n"

    if UPDATE:
        expected_out.parent.mkdir(parents=True, exist_ok=True)
        expected_out.write_text(output, encoding="utf-8")
        expected_stats.write_text(stats_text, encoding="utf-8")
        return

    assert expected_out.is_file(), (
        f"missing golden file; run with CONVMERGE_UPDATE_GOLDEN=1: {case}"
    )
    assert output == expected_out.read_text(encoding="utf-8")
    assert stats_text == expected_stats.read_text(encoding="utf-8")


def test_every_fixture_is_covered() -> None:
    fixtures = {p.stem for p in (GOLDEN / "inputs").glob("*.jsonl")}
    assert fixtures == {fixture for fixture, *_ in CASES.values()}


def test_no_stale_expected_files() -> None:
    known = set(CASES)
    for p in (GOLDEN / "expected").iterdir():
        case = p.name.removesuffix(".stats.json").removesuffix(".jsonl")
        assert case in known, f"stale golden file: {p.name}"
