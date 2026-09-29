"""Tool-calling encodings other than OpenAI's: Hermes tags, Glaive, xLAM.

Records mirror the real layouts (whitespace, separators, quoting) of
NousResearch/hermes-function-calling-v1, glaiveai/glaive-function-calling-v2
and Salesforce/xlam-function-calling-60k, with shortened content.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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


# --- Bracket calls (ToolACE) --------------------------------------------------

TOOLACE_SYSTEM = (
    "You are an expert in composing functions. Here is a list of functions in JSON format "
    "that you can invoke: " + json.dumps([
        {"name": "Market Trends API", "description": "d",
         "parameters": {"type": "dict", "properties": {"trend_type": {"type": "string"}}}},
        {"name": "SEC Filings", "description": "d", "parameters": {"type": "dict"}},
        {"name": "User Feed (Video Posts) V2", "description": "d"},
        {"name": "User Feed", "description": "d"},
    ]) + ".  Put it in the format of [func1(params_name=params_value), func2(params)]"
)  # fmt: skip


def test_toolace_bracket_calls(tmp_path: Path) -> None:
    record = {
        "system": TOOLACE_SYSTEM,
        "conversations": [
            {"from": "user", "value": "trends?"},
            {"from": "assistant", "value": '[Market Trends API(trend_type="GAINERS", n=2)]'},
            {"from": "tool", "value": '[{"name": "Market Trends API", "results": {}}]'},
            {"from": "assistant", "value": "Here."},
            {"from": "user", "value": "Both?"},
            {"from": "assistant",
             "value": '[SEC Filings(identifier="AAPL", ok=true, xs=[1, "a,b)"]), '
                      'Market Trends API(trend_type="LOSERS")]'},
        ],
    }  # fmt: skip
    (row,), stats = _convert(tmp_path, record)
    assert stats.written == 1
    msgs = row["messages"]
    assert msgs[2]["tool_calls"] == [_call("Market Trends API", {"trend_type": "GAINERS", "n": 2})]
    assert msgs[3]["role"] == "tool"
    assert msgs[-1]["tool_calls"] == [
        _call("SEC Filings", {"identifier": "AAPL", "ok": True, "xs": [1, "a,b)"]}),
        _call("Market Trends API", {"trend_type": "LOSERS"}),
    ]
    assert [t["function"]["name"] for t in row["tools"]][:2] == ["Market Trends API", "SEC Filings"]
    assert msgs[0]["content"] == TOOLACE_SYSTEM


@pytest.mark.parametrize(
    ("answer", "calls"),
    [
        # Argument names that are Python keywords or OData options (ToolACE).
        ('[SEC Filings(shareuid=6789, from="2025-01-01", to="2025-12-31", $top=10)]',
         [_call("SEC Filings", {"shareuid": 6789, "from": "2025-01-01", "to": "2025-12-31",
                                "$top": 10})]),
        # Names with parentheses; the longest listed name wins.
        ('[User Feed (Video Posts) V2(username="sunny"), User Feed (id=1)]',
         [_call("User Feed (Video Posts) V2", {"username": "sunny"}),
          _call("User Feed", {"id": 1})]),
        # JSON values, negative numbers, a trailing comma, no arguments.
        ('[SEC Filings(q={"a": true, "b": null}, lon=-93.2,), User Feed()]',
         [_call("SEC Filings", {"q": {"a": True, "b": None}, "lon": -93.2}),
          _call("User Feed", {})]),
    ],
)  # fmt: skip
def test_toolace_bracket_call_variants(tmp_path: Path, answer: str, calls: list) -> None:
    record = {"system": TOOLACE_SYSTEM, "conversations": [
        {"from": "user", "value": "q"}, {"from": "assistant", "value": answer}]}  # fmt: skip
    (row,), _ = _convert(tmp_path, record)
    assert row["messages"][-1]["tool_calls"] == calls


@pytest.mark.parametrize(
    "answer",
    [
        "[Unknown Tool(a=1)]",  # not a listed function
        "[SEC Filings(1, 2)]",  # positional arguments
        "[SEC Filings(a=open('x'))]",  # not a literal
        "[SEC Filings(a=1]",  # unbalanced
        "[SEC Filings(a==1)]",  # not an assignment
        "[SEC Filings(a=1 2)]",  # not one value
    ],
)
def test_bracket_text_that_is_not_a_call_is_left_alone(tmp_path: Path, answer: str) -> None:
    record = {"system": TOOLACE_SYSTEM, "conversations": [
        {"from": "user", "value": "q"}, {"from": "assistant", "value": answer}]}  # fmt: skip
    (row,), _ = _convert(tmp_path, record)
    assert row["messages"][-1]["content"] == answer
    assert "tool_calls" not in row["messages"][-1]


def test_function_call_turns(tmp_path: Path) -> None:
    record = {
        "conversations": [
            {"from": "system", "value": "You are a helpful assistant with access to the "
             "following functions. Use them if required -" + json.dumps(GET_NEWS, indent=4)},
            {"from": "human", "value": "News?"},
            {"from": "function-call",
             "value": '{"name": "get_news_headlines", "arguments": \'{"country": "US"}\'}'},
            {"from": "function-response", "value": '{"headlines": ["A"]}'},
            {"from": "gpt", "value": "A"},
        ]
    }  # fmt: skip
    (row,), stats = _convert(tmp_path, record)
    assert stats.written == 1
    msgs = row["messages"]
    assert msgs[0]["content"].endswith("Use them if required")
    assert msgs[2]["tool_calls"] == [_call("get_news_headlines", {"country": "US"})]
    assert msgs[3]["role"] == "tool"
    assert row["tools"][0]["function"]["name"] == "get_news_headlines"
