"""Tests for the auto-detecting chat adapter."""

from __future__ import annotations

from convmerge.adapters import ADAPTERS
from convmerge.adapters.chat import iter_from_chat_line


def test_messages_shape() -> None:
    record = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }
    out = list(iter_from_chat_line(record))
    assert len(out) == 1
    msgs = out[0].messages
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[0].content == "hi"


def test_sharegpt_style_conversations_from_value() -> None:
    record = {
        "conversations": [
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": "A"},
        ]
    }
    out = list(iter_from_chat_line(record))
    assert len(out) == 1
    assert [m.role for m in out[0].messages] == ["user", "assistant"]
    assert out[0].messages[1].content == "A"


def test_conversation_singular_key() -> None:
    record = {
        "conversation": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
    }
    out = list(iter_from_chat_line(record))
    assert len(out) == 1


def test_plain_text_becomes_single_assistant_turn() -> None:
    out = list(iter_from_chat_line({"text": "just some pretraining text"}))
    assert len(out) == 1
    assert out[0].messages[0].role == "assistant"
    assert out[0].messages[0].content.startswith("just some")


def test_alpaca_fallback() -> None:
    out = list(
        iter_from_chat_line(
            {"instruction": "Q", "input": "ctx", "output": "A"},
        )
    )
    assert len(out) == 1
    msgs = out[0].messages
    assert msgs[-1].role == "assistant"
    assert msgs[-1].content == "A"


def _pairwise_record(winner: str | None = None) -> dict:
    out: dict = {
        "conversation_a": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A1"},
        ],
        "conversation_b": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A2"},
        ],
    }
    if winner is not None:
        out["winner"] = winner
    return out


def test_pairwise_winner_a() -> None:
    out = list(iter_from_chat_line(_pairwise_record("model_a")))
    assert len(out) == 1
    assert out[0].messages[-1].content == "A1"


def test_pairwise_both() -> None:
    out = list(iter_from_chat_line(_pairwise_record(), pairwise_mode="both"))
    assert len(out) == 2
    contents = {ex.messages[-1].content for ex in out}
    assert contents == {"A1", "A2"}


def test_pairwise_tie_emits_nothing() -> None:
    assert list(iter_from_chat_line(_pairwise_record("tie"))) == []


def test_adapter_registered() -> None:
    assert "chat" in ADAPTERS
    assert "auto" in ADAPTERS


def test_role_map_override() -> None:
    record = {"messages": [{"role": "foo", "content": "x"}]}
    out = list(iter_from_chat_line(record, role_map={"foo": "assistant"}))
    assert out[0].messages[0].role == "assistant"
