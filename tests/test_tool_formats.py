"""Tool-calling encodings other than OpenAI's: Hermes tags, Glaive, xLAM.

Records mirror the real layouts (whitespace, separators, quoting) of
NousResearch/hermes-function-calling-v1, glaiveai/glaive-function-calling-v2
and Salesforce/xlam-function-calling-60k, with shortened content.
"""

from __future__ import annotations

import json
from pathlib import Path

from convmerge import ConvertStats, convert_file

GET_NEWS = {
    "name": "get_news_headlines",
    "description": "Get the latest news headlines",
    "parameters": {
        "type": "object",
        "properties": {"country": {"type": "string"}},
        "required": ["country"],
    },
}
PASSWORD = {"name": "generate_password", "description": "Make one", "parameters": {}}


def _convert(
    tmp_path: Path, record: dict, fmt: str = "messages"
) -> tuple[list[dict], ConvertStats]:
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(record) + "\n", encoding="utf-8")
    stats = ConvertStats()
    convert_file(src, tmp_path / "out.jsonl", adapter_name="auto", output_format=fmt, stats=stats)
    return [json.loads(x) for x in (tmp_path / "out.jsonl").read_text().splitlines()], stats


def _call(name: str, args: dict) -> dict:
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


# --- Glaive ------------------------------------------------------------------

GLAIVE = {
    "system": "SYSTEM: You are a helpful assistant with access to the following functions. "
    "Use them if required -\n" + json.dumps(GET_NEWS, indent=4) + "\n\n"
    + json.dumps(PASSWORD, indent=4) + "\n\n",
    "chat": "USER: Can you tell me the latest news headlines for the United States?\n\n\n"
    'ASSISTANT: <functioncall> {"name": "get_news_headlines", "arguments": '
    "'{\"country\": \"United States\"}'} <|endoftext|>\n\n\n"
    'FUNCTION RESPONSE: {"headlines": ["A", "B"]}\n\n\n'
    "ASSISTANT: Here are the headlines: A, B <|endoftext|>\n\n\n"
    "USER: Thanks!\n\n\n"
    "ASSISTANT: You're welcome! <|endoftext|>\n\n\n",
}  # fmt: skip


def test_glaive_transcript_becomes_tool_calls(tmp_path: Path) -> None:
    rows, stats = _convert(tmp_path, GLAIVE)
    assert stats.dropped == 0
    row = rows[0]
    assert row["tools"] == [
        {"type": "function", "function": GET_NEWS},
        {"type": "function", "function": PASSWORD},
    ]
    assert row["messages"] == [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to the following functions. "
            "Use them if required",
        },
        {"role": "user",
         "content": "Can you tell me the latest news headlines for the United States?"},
        {"role": "assistant", "content": "",
         "tool_calls": [_call("get_news_headlines", {"country": "United States"})]},
        {"role": "tool", "name": "get_news_headlines", "content": '{"headlines": ["A", "B"]}'},
        {"role": "assistant", "content": "Here are the headlines: A, B"},
        {"role": "user", "content": "Thanks!"},
        {"role": "assistant", "content": "You're welcome!"},
    ]  # fmt: skip


def test_glaive_without_functions(tmp_path: Path) -> None:
    record = {
        "system": "SYSTEM: You are a helpful assistant, with no access to external functions.\n\n",
        "chat": "USER: Hi\n\n\nASSISTANT: Hello! <|endoftext|>\n\n\n",
    }
    rows, _ = _convert(tmp_path, record)
    assert "tools" not in rows[0]
    assert [m["content"] for m in rows[0]["messages"]] == [
        "You are a helpful assistant, with no access to external functions.",
        "Hi",
        "Hello!",
    ]


# --- Hermes ------------------------------------------------------------------

HERMES_SYSTEM = (
    "You are a function calling AI model. You are provided with function signatures within "
    "<tools> </tools> XML tags.\n<tools>\n"
    + json.dumps([{"type": "function", "function": GET_NEWS}])
    + "\n</tools>\nFor each function call return a json object with function name and "
    "arguments within <tool_call> </tool_call> tags with the following schema:\n"
    '<tool_call>\n{"name": <function-name>, "arguments": <args-dict>}\n</tool_call>\n'
)


def _hermes(turns: list[tuple[str, str]], with_tools_column: bool = True) -> dict:
    record: dict = {
        "id": "h1",
        "conversations": [{"from": "system", "value": HERMES_SYSTEM}]
        + [{"from": f, "value": v} for f, v in turns],
    }
    if with_tools_column:
        record["tools"] = json.dumps([{"type": "function", "function": GET_NEWS}])
    return record


def test_hermes_parallel_calls_and_responses(tmp_path: Path) -> None:
    record = _hermes(
        [
            ("human", "News for US and France?"),
            ("gpt", '<tool_call>\n{"name": "get_news_headlines", "arguments": {"country": "US"}}\n'
                    '</tool_call>\n<tool_call>\n{"name": "get_news_headlines", "arguments": '
                    '{"country": "France"}}\n</tool_call>\n'),
            ("tool", '<tool_response>\n{"name": "get_news_headlines", "content": {"h": ["a"]}}\n'
                     '</tool_response>\n<tool_response>\n{"name": "get_news_headlines", '
                     '"content": {"h": ["b"]}}\n</tool_response>\n'),
            ("gpt", "US: a. France: b."),
        ]
    )  # fmt: skip
    rows, stats = _convert(tmp_path, record)
    assert stats.dropped == 0
    msgs = rows[0]["messages"]
    assert msgs[0]["content"] == HERMES_SYSTEM  # the system prompt is kept as written
    assert msgs[2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            _call("get_news_headlines", {"country": "US"}),
            _call("get_news_headlines", {"country": "France"}),
        ],
    }
    assert msgs[3:5] == [
        {"role": "tool", "name": "get_news_headlines", "content": '{"h": ["a"]}'},
        {"role": "tool", "name": "get_news_headlines", "content": '{"h": ["b"]}'},
    ]
    assert msgs[5] == {"role": "assistant", "content": "US: a. France: b."}
    assert rows[0]["tools"] == [{"type": "function", "function": GET_NEWS}]


def test_hermes_single_turn_and_tools_from_system_prompt(tmp_path: Path) -> None:
    record = _hermes(
        [
            ("human", "News?"),
            ("gpt", '<tool_call>\n{"name": "get_news_headlines", "arguments": {"country": "KR"}}\n'
                    "</tool_call>\n"),
        ],
        with_tools_column=False,
    )  # fmt: skip
    rows, _ = _convert(tmp_path, record)
    assert rows[0]["tools"] == [{"type": "function", "function": GET_NEWS}]
    assert rows[0]["messages"][-1]["tool_calls"] == [_call("get_news_headlines", {"country": "KR"})]


def test_text_around_calls_is_kept_and_bad_blocks_are_left_alone(tmp_path: Path) -> None:
    record = _hermes(
        [
            ("human", "News?"),
            ("gpt", 'Let me check.\n<tool_call>\n{"name": "get_news_headlines", "arguments": {}}\n'
                    "</tool_call>\n<tool_call>not json</tool_call>"),
        ]
    )  # fmt: skip
    rows, _ = _convert(tmp_path, record)
    last = rows[0]["messages"][-1]
    assert last["content"] == "Let me check.\n\n<tool_call>not json</tool_call>"
    assert last["tool_calls"] == [_call("get_news_headlines", {})]


def test_plain_conversations_are_untouched(tmp_path: Path) -> None:
    record = {"conversations": [{"from": "human", "value": "use <tool_call> tags?"},
                                {"from": "gpt", "value": "<tool_call> is a tag"}]}  # fmt: skip
    rows, _ = _convert(tmp_path, record)
    assert rows[0]["messages"][1]["content"] == "<tool_call> is a tag"


# --- xLAM --------------------------------------------------------------------


def test_xlam_record(tmp_path: Path) -> None:
    record = {
        "id": 7,
        "query": "Headlines for Japan and a 12-char password.",
        "answers": json.dumps(
            [
                {"name": "get_news_headlines", "arguments": {"country": "Japan"}},
                {"name": "generate_password", "arguments": {"length": 12}},
            ]
        ),
        "tools": json.dumps([GET_NEWS, PASSWORD]),
    }
    rows, stats = _convert(tmp_path, record)
    assert stats.dropped == 0
    assert rows[0]["messages"] == [
        {"role": "user", "content": "Headlines for Japan and a 12-char password."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _call("get_news_headlines", {"country": "Japan"}),
                _call("generate_password", {"length": 12}),
            ],
        },
    ]
    assert [t["function"]["name"] for t in rows[0]["tools"]] == [
        "get_news_headlines",
        "generate_password",
    ]


def test_tool_calls_survive_alpaca_as_unrepresentable(tmp_path: Path) -> None:
    rows, stats = _convert(tmp_path, GLAIVE, fmt="alpaca")
    assert rows == [] and stats.drop_reasons == {"unrepresentable_tool_calls": 1}
