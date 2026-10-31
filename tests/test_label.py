"""convmerge label: parsing, retries, cache, existing labels, truncation, and the CLI."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from convmerge import LabelStats, label_jsonl
from convmerge.cli import main
from convmerge.llm import BudgetExceeded, LLMError
from tests.test_llm import FakeServer


@pytest.fixture()
def fake(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[FakeServer, str]]:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    state = FakeServer()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), state.handler())
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
    yield state, f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def _row(user: str, answer: str = "ok", **extra: object) -> dict:
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": answer}], **extra}  # fmt: skip


def _write(path: Path, rows: list) -> Path:
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows))
    return path


def _read(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _user(body: dict) -> str:
    return body["messages"][-1]["content"]


def _topic(body: dict) -> str:
    text = _user(body).split("Conversation:", 1)[1]
    return next((w for w in ("code", "math") if w in text), "other")


TOPICS = {"classes": ["code", "math", "other"], "question": "What is this mainly about?"}


def test_classes_labels_every_readable_row(fake: tuple[FakeServer, str], tmp_path: Path) -> None:
    server, url = fake
    server.reply = _topic
    src = _write(tmp_path / "in.jsonl", [
        _row("fix my code", meta={"source": "a"}),
        _row("solve this math"),
        {"prompt": "a math pair", "chosen": "good", "rejected": "bad"},
        "{not json",
        {"no": "conversation"},
        _row("hello", meta=None),
    ])  # fmt: skip
    st = label_jsonl(src, tmp_path / "out.jsonl", endpoint=url, model="tiny", name="topic",
                     **TOPICS)  # fmt: skip
    lines = (tmp_path / "out.jsonl").read_text().splitlines()
    assert lines[3] == "{not json"  # written as it was
    rows = [json.loads(line) for i, line in enumerate(lines) if i != 3]
    assert [r.get("meta") for r in rows] == [
        {"source": "a", "topic": "code"}, {"topic": "math"}, {"topic": "math"}, None,
        {"topic": "other"},
    ]  # fmt: skip
    assert rows[0]["messages"][0] == {"role": "user", "content": "fix my code"}
    assert (st.rows, st.labeled, st.unreadable, st.parse_failures) == (6, 4, 2, 0)
    assert st.values == {"code": 1, "math": 2, "other": 1}
    assert (st.calls, st.host, st.served_models) == (4, url.split("/")[2], ["tiny-served"])
    body = next(b for b in server.requests if "fix my code" in _user(b))  # any order
    assert body["messages"][0]["role"] == "system" and len(body["messages"]) == 2
    assert "Classes: code, math, other" in _user(body) and "What is this mainly" in _user(body)
    assert "user: fix my code\n\nassistant: ok" in _user(body)
    pair = next(_user(b) for b in server.requests if "a math pair" in _user(b))
    assert "assistant: good" in pair and "bad" not in pair  # prompt and chosen answer only


def test_parse_rules_and_one_retry(fake: tuple[FakeServer, str], tmp_path: Path) -> None:
    server, url = fake
    first = {"a": '"Math."', "b": "**code**", "c": "OTHER\nbecause it is", "d": "I think math",
             "e": "no idea"}  # fmt: skip

    def reply(body: dict) -> str:
        text = _user(body)
        key = text.split("user: ", 1)[1][0]
        if "Your previous answer was" in text:
            return "math" if key == "d" else "still no idea"
        return first[key]

    server.reply = reply
    src = _write(tmp_path / "in.jsonl", [_row(k) for k in "abcde"])
    st = label_jsonl(src, tmp_path / "out.jsonl", endpoint=url, model="tiny", name="topic",
                     **TOPICS)  # fmt: skip
    assert [r["meta"]["topic"] for r in _read(tmp_path / "out.jsonl")] == [
        "math", "code", "other", "math", None]  # fmt: skip
    assert (st.labeled, st.parse_failures, st.failed_samples) == (4, 1, ["still no idea"])
    retry = next(_user(b) for b in server.requests if "Your previous answer" in _user(b))
    assert "<<<\nI think math\n>>>" in retry or "<<<\nno idea\n>>>" in retry
    assert retry.endswith("Answer with exactly one of these class names and nothing else: "
                          "code, math, other")  # fmt: skip


def test_scale_with_rubric(fake: tuple[FakeServer, str], tmp_path: Path) -> None:
    server, url = fake
    answers = {"0": "4", "1": "9", "2": " 2 ", "3": "+5"}  # row 1 is out of range twice
    server.reply = lambda body: answers[_user(body).split("user: ", 1)[1][0]]
    src = _write(tmp_path / "in.jsonl", [_row(str(i)) for i in range(4)])
    st = label_jsonl(src, tmp_path / "out.jsonl", endpoint=url, model="tiny", name="difficulty",
                     scale="1-5", rubric="5 means expert-level.", workers=1)  # fmt: skip
    assert [r["meta"]["difficulty"] for r in _read(tmp_path / "out.jsonl")] == [4, None, 2, 5]
    assert st.values == {"1": 0, "2": 1, "3": 0, "4": 1, "5": 1}
    assert st.task == "scale" and st.parse_failures == 1
    assert "on a scale of 1-5" in _user(server.requests[0])
    assert "5 means expert-level." in _user(server.requests[0])


def test_existing_labels_need_overwrite(fake: tuple[FakeServer, str], tmp_path: Path) -> None:
    server, url = fake
    server.reply = lambda body: "code"
    src = _write(tmp_path / "in.jsonl", [_row("x"), _row("y", meta={"topic": "old", "k": 1})])
    with pytest.raises(ValueError, match=r"1 row\(s\) already have meta.topic \(first on line 2\)"):
        label_jsonl(src, tmp_path / "out.jsonl", endpoint=url, model="tiny", name="topic",
                    **TOPICS)  # fmt: skip
    assert server.requests == [] and not (tmp_path / "out.jsonl").exists()
    label_jsonl(src, tmp_path / "out.jsonl", endpoint=url, model="tiny", name="topic",
                overwrite=True, **TOPICS)  # fmt: skip
    assert _read(tmp_path / "out.jsonl")[1]["meta"] == {"topic": "code", "k": 1}
    label_jsonl(src, tmp_path / "other.jsonl", endpoint=url, model="tiny", name="topic",
                meta_key="labels", **TOPICS)  # fmt: skip
    assert _read(tmp_path / "other.jsonl")[1]["labels"] == {"topic": "code"}


def test_cache_reuse_and_cache_only(fake: tuple[FakeServer, str], tmp_path: Path) -> None:
    server, url = fake
    server.reply = _topic
    cache = tmp_path / "cache" / "llm.sqlite"
    src = _write(tmp_path / "in.jsonl", [_row("code a"), _row("math b")])
    label_jsonl(src, tmp_path / "o1.jsonl", endpoint=url, model="tiny", name="topic",
                cache=cache, **TOPICS)  # fmt: skip
    # The same conversation in another file (other row bytes) reuses the answer.
    other = _write(tmp_path / "other.jsonl", [_row("math b", meta={"id": 7}), _row("new")])
    st = label_jsonl(other, tmp_path / "o2.jsonl", endpoint=url, model="tiny", name="topic",
                     cache=cache, **TOPICS)  # fmt: skip
    assert (st.calls, st.cache_hits, len(server.requests)) == (1, 1, 3)
    offline = label_jsonl(src, tmp_path / "o3.jsonl", endpoint="https://api.example.com/v1",
                          model="tiny", name="topic", cache=tmp_path / "empty.sqlite",
                          cache_only=True, **TOPICS)  # fmt: skip
    assert (offline.calls, offline.call_failures, offline.labeled) == (0, 2, 0)
    assert [r["meta"]["topic"] for r in _read(tmp_path / "o3.jsonl")] == [None, None]
    again = label_jsonl(src, tmp_path / "o4.jsonl", endpoint=url, model="tiny", name="topic",
                        cache=cache, cache_only=True, **TOPICS)  # fmt: skip
    assert (again.calls, again.labeled) == (0, 2)
    assert (tmp_path / "o1.jsonl").read_bytes() == (tmp_path / "o4.jsonl").read_bytes()


def test_truncation_tools_and_media(fake: tuple[FakeServer, str], tmp_path: Path) -> None:
    server, url = fake
    server.reply = lambda body: "other"
    long = "START " + "x" * 5000 + " END"
    tool = {"messages": [
        {"role": "user", "content": [{"type": "image", "url": "a.png"},
                                     {"type": "text", "text": "what is this?"}]},
        {"role": "assistant", "content": None,
         "tool_calls": [{"type": "function", "id": "c1",
                         "function": {"name": "lookup", "arguments": "{\"q\": 1}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "found"},
        {"role": "assistant", "content": "a cat"}]}  # fmt: skip
    src = _write(tmp_path / "in.jsonl", [_row(long), tool])
    st = label_jsonl(src, tmp_path / "out.jsonl", endpoint=url, model="tiny", name="topic",
                     max_chars=1000, workers=1, **TOPICS)  # fmt: skip
    assert st.truncated == 1
    sent = _user(server.requests[0]).split("<<<\n", 1)[1].rsplit("\n>>>", 1)[0]
    assert len(sent) <= 1000 and "user: START" in sent and sent.endswith("assistant: ok")
    assert "characters cut ...]" in sent
    shown = _user(server.requests[1])
    assert "user: [image]\nwhat is this?" in shown
    assert 'assistant: [tool call] lookup({"q": 1})' in shown and "tool: found" in shown


def test_custom_prompt_and_option_errors(fake: tuple[FakeServer, str], tmp_path: Path) -> None:
    server, url = fake
    server.reply = lambda body: "math"
    src = _write(tmp_path / "in.jsonl", [_row("q")])
    out = tmp_path / "out.jsonl"
    st = label_jsonl(src, out, endpoint=url, model="tiny", name="topic", classes=["code", "math"],
                     prompt="Pick one of $classes, costs $$5.\n$conversation")  # fmt: skip
    expected = "Pick one of code, math, costs $5.\nuser: q\n\nassistant: ok"
    assert _user(server.requests[0]) == expected
    assert len(st.prompt_sha256) == 64
    bad = [
        ({"classes": ["a", "b"], "prompt": "no placeholder"}, "no \\$conversation"),
        ({"classes": ["a", "b"], "prompt": "$conversation $nope"}, "unknown or malformed"),
        ({"classes": ["a"], "question": "q"}, "at least two"),
        ({"classes": ["a", "A"], "question": "q"}, "repeat"),
        ({"classes": ["a", "b"]}, "needs a question"),
        ({"scale": "1-5"}, "needs a rubric"),
        ({"scale": "5-1", "question": "q"}, "LO < HI"),
        ({"classes": ["a", "b"], "scale": "1-5", "question": "q"}, "exactly one"),
    ]
    for kwargs, message in bad:
        with pytest.raises(ValueError, match=message):
            label_jsonl(src, tmp_path / "x.jsonl", endpoint=url, model="tiny", name="t",
                        **kwargs)  # type: ignore[arg-type]  # fmt: skip
    assert len(server.requests) == 1


def test_stops_keep_cache_and_write_nothing(fake: tuple[FakeServer, str], tmp_path: Path) -> None:
    server, url = fake
    server.reply = _topic
    src = _write(tmp_path / "in.jsonl", [_row(f"code {i}") for i in range(5)])
    out, cache = tmp_path / "out.jsonl", tmp_path / "llm.sqlite"
    stats = LabelStats()
    with pytest.raises(BudgetExceeded):
        label_jsonl(src, out, endpoint=url, model="tiny", name="topic", cache=cache,
                    max_calls=2, workers=1, stats=stats, **TOPICS)  # fmt: skip
    assert not out.exists() and stats.calls == 2
    st = label_jsonl(src, out, endpoint=url, model="tiny", name="topic", cache=cache,
                     **TOPICS)  # fmt: skip
    assert (st.cache_hits, st.calls, st.labeled) == (2, 3, 5)
    server.status = [401]
    with pytest.raises(LLMError, match="HTTP 401"):
        label_jsonl(src, tmp_path / "o2.jsonl", endpoint=url, model="other", name="topic",
                    **TOPICS)  # fmt: skip


def test_cli(fake: tuple[FakeServer, str], tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    server, url = fake
    server.reply = _topic
    src = _write(tmp_path / "in.jsonl", [_row("code"), _row("math")])
    out = tmp_path / "data" / "out.jsonl"
    rubric = tmp_path / "rubric.md"
    rubric.write_text("Higher is harder.")
    base = ["label", "-i", str(src), "--endpoint", url, "--model", "tiny"]
    main([*base, "-o", str(out), "--name", "topic", "--classes", "code, math,other",
          "--question", "Topic?"])  # fmt: skip
    report = json.loads(capsys.readouterr().out)
    assert (report["labeled"], report["values"]) == (2, {"code": 1, "math": 1, "other": 0})
    assert report["version"] >= 1 and "api" not in json.dumps(report).lower()
    assert (out.parent / ".convmerge-cache" / "llm.sqlite").is_file()
    server.reply = lambda body: "3"
    main([*base, "-i", str(out), "-o", str(tmp_path / "o2.jsonl"), "--name", "difficulty",
          "--scale", "1-5", "--rubric", str(rubric)])  # fmt: skip
    rows = _read(tmp_path / "o2.jsonl")
    assert rows[0]["meta"] == {"topic": "code", "difficulty": 3}
    assert "Higher is harder." in _user(server.requests[-1])

    def code(argv: list[str]) -> int:
        try:
            main(argv)
        except SystemExit as e:
            return int(e.code or 0)
        return 0

    topic = ["--name", "topic", "--classes", "a,b", "--question", "q"]
    assert code([*base, "-i", str(out), "-o", str(tmp_path / "o3.jsonl"), *topic]) == 2
    assert "--overwrite" in capsys.readouterr().err
    ext = ["label", "-i", str(src), "-o", str(tmp_path / "o4.jsonl"), "--endpoint",
           "https://api.example.com/v1", "--model", "m", *topic]  # fmt: skip
    assert code(ext) == 2 and "--allow-external" in capsys.readouterr().err
    server.status = [404]
    assert code([*base, "-i", str(src), "-o", str(tmp_path / "o5.jsonl"), "--model", "nope",
                 "--cache", str(tmp_path / "c.sqlite"), *topic]) == 1  # fmt: skip
    err = capsys.readouterr().err
    assert "HTTP 404" in err and "was not written" in err
    assert code([*base, "-o", str(tmp_path / "o6.jsonl"), "--name", "t", "--scale", "3"]) == 2
    assert code([*base, "-o", str(tmp_path / "o6.jsonl"), "--name", "t"]) == 2
