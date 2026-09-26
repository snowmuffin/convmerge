"""Data model v2: content parts, tool calls, tools, and messages emission."""

from __future__ import annotations

import pytest

from convmerge.emitters import emit_alpaca, emit_messages, get_emitter
from convmerge.models import ChatMessage, ContentPart, ToolCall, TrainingExample

WEATHER = {"type": "function", "function": {"name": "get_weather", "parameters": {}}}


def test_plain_text_message_serializes_as_before() -> None:
    ex = TrainingExample(messages=[ChatMessage("user", "hi"), ChatMessage("assistant", "yo")])
    assert emit_messages(ex) == {
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    }


def test_content_parts_are_tuples_and_text_property() -> None:
    m = ChatMessage(
        "user",
        [
            ContentPart("text", text="a"),
            ContentPart("image", url="x.png"),
            ContentPart("text", text="b"),
        ],
    )
    assert isinstance(m.content, tuple)
    assert m.text == "a\nb"
    assert [p.url for p in m.media] == ["x.png"]
    assert hash(m)  # frozen dataclass stays hashable
    assert ChatMessage("assistant", None).text == ""


def test_tool_call_from_any_normalizes_arguments() -> None:
    tc = ToolCall.from_any("f", {"city": "서울"}, id="c1")
    assert tc.arguments == '{"city": "서울"}'
    assert tc.arguments_object() == {"city": "서울"}
    assert ToolCall.from_any("f", None).arguments == "{}"
    assert ToolCall("f", "not json").arguments_object() == "not json"


def _tool_example() -> TrainingExample:
    return TrainingExample(
        messages=[
            ChatMessage("user", "Weather?", name="alice"),
            ChatMessage(
                "assistant",
                None,
                tool_calls=[ToolCall.from_any("get_weather", {"city": "Seoul"}, id="c1")],
            ),
            ChatMessage("tool", '{"temp_c": 21}', tool_call_id="c1"),
            ChatMessage(
                "user",
                [
                    ContentPart("image", url="https://e.x/a.png"),
                    ContentPart("audio", url="a.wav"),
                    ContentPart("video", url="v.mp4"),
                    ContentPart("image"),
                    ContentPart("text", text="and this?"),
                ],
            ),
        ],
        tools=[WEATHER],
    )


def test_emit_messages_openai_schema() -> None:
    row = emit_messages(_tool_example())
    assert row["tools"] == [WEATHER]
    user, call, tool, media = row["messages"]
    assert user == {"role": "user", "name": "alice", "content": "Weather?"}
    assert call == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Seoul"}'},
            }
        ],
    }
    assert tool == {"role": "tool", "content": '{"temp_c": 21}', "tool_call_id": "c1"}
    assert media["content"] == [
        {"type": "image_url", "image_url": {"url": "https://e.x/a.png"}},
        {"type": "audio_url", "audio_url": {"url": "a.wav"}},
        {"type": "video_url", "video_url": {"url": "v.mp4"}},
        {"type": "image"},
        {"type": "text", "text": "and this?"},
    ]


def test_emit_messages_tool_arguments_object() -> None:
    row = get_emitter("messages", tool_arguments="object")(_tool_example())
    assert row["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"city": "Seoul"}


def test_get_emitter_rejects_bad_tool_arguments() -> None:
    with pytest.raises(ValueError, match="tool_arguments"):
        get_emitter("messages", tool_arguments="dict")  # type: ignore[arg-type]


def test_emit_alpaca_uses_text_of_parts() -> None:
    ex = TrainingExample(
        messages=[
            ChatMessage("user", [ContentPart("text", text="q")]),
            ChatMessage("assistant", [ContentPart("text", text="a")]),
        ]
    )
    assert emit_alpaca(ex) == {"instruction": "q", "input": "", "output": "a"}
