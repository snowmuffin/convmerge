"""Adapters on tool-calling, multimodal, and LLaMA-Factory shaped records."""

from __future__ import annotations

from convmerge.adapters.alpaca import iter_from_alpaca_line
from convmerge.adapters.chat import iter_from_chat_line
from convmerge.adapters.sharegpt import iter_from_sharegpt_line
from convmerge.models import ContentPart, ToolCall


def _one(it):
    out = list(it)
    assert len(out) == 1
    return out[0]


def test_image_token_left_alone_without_media_column() -> None:
    ex = _one(
        iter_from_chat_line(
            {"messages": [{"role": "user", "content": "What does <image> mean in HTML?"}]}
        )
    )
    assert ex.messages[0].content == "What does <image> mean in HTML?"
    assert ex.issues == []


def test_media_count_mismatch_is_reported() -> None:
    more_tokens = {
        "conversations": [
            {"from": "human", "value": "<image><image>compare"},
            {"from": "gpt", "value": "same"},
        ],
        "images": ["a.png"],
    }
    ex = _one(iter_from_sharegpt_line(more_tokens))
    assert [p.url for p in ex.messages[0].media] == ["a.png", None]
    assert ex.issues == ["unresolved_image"]

    more_images = {**more_tokens, "images": ["a.png", "b.png", "c.png"]}
    assert _one(iter_from_sharegpt_line(more_images)).issues == ["unused_image"]


def test_video_and_audio_columns() -> None:
    ex = _one(
        iter_from_sharegpt_line(
            {
                "conversations": [
                    {"from": "human", "value": "<video>What happens? <audio>"},
                    {"from": "gpt", "value": "A dog barks."},
                ],
                "videos": ["v.mp4"],
                "audios": ["a.wav"],
            }
        )
    )
    assert ex.messages[0].content == (
        ContentPart("video", url="v.mp4"),
        ContentPart("text", text="What happens?"),
        ContentPart("audio", url="a.wav"),
    )


def test_parallel_function_calls_and_bad_function_call_value() -> None:
    ex = _one(
        iter_from_sharegpt_line(
            {
                "conversations": [
                    {"from": "human", "value": "Seoul and Busan?"},
                    {
                        "from": "function_call",
                        "value": '[{"name": "w", "arguments": {"city": "Seoul"}},'
                        ' {"name": "w", "arguments": {"city": "Busan"}}]',
                    },
                    {"from": "observation", "value": "[21, 24]"},
                    {"from": "function_call", "value": "not json"},
                    {"from": "gpt", "value": "21 and 24."},
                ]
            }
        )
    )
    calls = ex.messages[1].tool_calls
    assert [c.arguments_object() for c in calls] == [{"city": "Seoul"}, {"city": "Busan"}]
    assert ex.messages[2].role == "tool"
    # An undecodable function_call keeps its raw role so validation can flag it.
    assert (ex.messages[3].role, ex.messages[3].content) == ("function_call", "not json")


def test_legacy_openai_function_call_and_function_role() -> None:
    ex = _one(
        iter_from_chat_line(
            {
                "messages": [
                    {"role": "user", "content": "time?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "function_call": {"name": "now", "arguments": "{}"},
                    },
                    {"role": "function", "name": "now", "content": "12:00"},
                    {"role": "assistant", "content": "Noon."},
                ]
            }
        )
    )
    assert ex.messages[1].tool_calls == (ToolCall("now", "{}"),)
    assert (ex.messages[2].role, ex.messages[2].name) == ("tool", "now")


def test_null_content_falls_back_to_next_key() -> None:
    ex = _one(
        iter_from_chat_line(
            {
                "messages": [
                    {"role": "user", "content": None, "text": "hi"},
                    {"role": "assistant", "content": "yo"},
                ]
            }
        )
    )
    assert ex.messages[0].content == "hi"


def test_tools_column_forms() -> None:
    bare = {"name": "f", "parameters": {}}
    wrapped = {"type": "function", "function": bare}
    base = {"messages": [{"role": "user", "content": "x"}]}
    assert _one(iter_from_chat_line({**base, "tools": [bare]})).tools == [wrapped]
    assert _one(
        iter_from_chat_line({**base, "tools": '[{"name": "f", "parameters": {}}]'})
    ).tools == [wrapped]
    assert _one(iter_from_chat_line({**base, "tools": "not json"})).tools is None


def test_existing_system_turn_wins_over_column() -> None:
    ex = _one(
        iter_from_chat_line(
            {
                "messages": [
                    {"role": "system", "content": "inline"},
                    {"role": "user", "content": "x"},
                ],
                "system": "column",
            }
        )
    )
    assert [m.content for m in ex.messages if m.role == "system"] == ["inline"]


def test_alpaca_history_skips_malformed_pairs() -> None:
    ex = _one(
        iter_from_alpaca_line(
            {
                "instruction": "q3",
                "output": "a3",
                "history": [["q1", "a1"], ["only one"], "bad", ["q2", ""], ["q2b", "a2b"]],
            }
        )
    )
    assert [m.content for m in ex.messages] == ["q1", "a1", "q2b", "a2b", "q3", "a3"]


def test_media_without_placeholders_leads_first_user_turn() -> None:
    ex = _one(
        iter_from_chat_line(
            {
                "messages": [
                    {"role": "system", "content": "Describe images."},
                    {"role": "user", "content": "What is this?"},
                    {"role": "assistant", "content": "A bus."},
                ],
                "images": [{"path": "bus.jpg", "bytes": None}, {"bytes": "AAAA"}],
            }
        )
    )
    assert ex.messages[1].content == (
        ContentPart("image", url="bus.jpg"),
        ContentPart("text", text="What is this?"),
    )
    assert ex.issues == []
