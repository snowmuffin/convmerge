"""convmerge.llm: the OpenAI-compatible call layer, against a local fake server."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from convmerge.llm import (
    BudgetExceeded,
    CallStats,
    Client,
    Endpoint,
    ExternalHostError,
    LLMError,
    ResponseCache,
    cache_key,
    host_allowed,
)


class FakeServer:
    """An OpenAI-compatible ``/v1/chat/completions`` whose answers a test scripts."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self.status: list[int] = []  # popped per request; empty means 200
        self.reply = lambda body: "ok"
        self.lock = threading.Lock()

    def handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with server.lock:
                    server.requests.append(body)
                    server.headers.append(dict(self.headers))
                    status = server.status.pop(0) if server.status else 200
                if self.path != "/v1/chat/completions":
                    status = 404
                if status != 200:
                    self.send_response(status)
                    self.end_headers()
                    return
                payload = {"model": body["model"] + "-served", "system_fingerprint": "fp1",
                           "choices": [{"message": {"role": "assistant",
                                                    "content": server.reply(body)}}]}  # fmt: skip
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        return Handler


@pytest.fixture()
def fake(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[FakeServer, str]]:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    state = FakeServer()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), state.handler())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield state, f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def _msgs(text: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": "Answer with one word."},
            {"role": "user", "content": text}]  # fmt: skip


def test_host_policy() -> None:
    for url in ("http://localhost:8000/v1", "http://127.0.0.1/v1", "http://[::1]:1/v1",
                "http://10.0.0.5/v1", "http://192.168.1.2/v1", "http://172.20.0.1/v1",
                "http://gpu-box.local:8000/v1", "http://169.254.1.1/v1"):  # fmt: skip
        assert host_allowed(url), url
    for url in ("https://api.openai.com/v1", "http://8.8.8.8/v1", "http://example.com/v1"):
        assert not host_allowed(url), url
    with pytest.raises(ExternalHostError, match="api.example.com"):
        Client(Endpoint("https://api.example.com/v1", "m"))
    Client(Endpoint("https://api.example.com/v1", "m", allow_external=True))
    Client(Endpoint("https://api.example.com/v1", "m"), cache_only=True)  # sends nothing
    with pytest.raises(ValueError, match="http"):
        Endpoint("ftp://host/v1", "m")


def test_request_shape_answers_and_order(fake: tuple[FakeServer, str]) -> None:
    server, url = fake
    server.reply = lambda body: body["messages"][-1]["content"].upper()
    stats = CallStats()
    client = Client(Endpoint(url, "tiny", max_tokens=8), workers=3, stats=stats)
    answers = client.complete_many([_msgs(w) for w in ("a", "b", "c", "d", "e")])
    assert [a.text for a in answers if a] == ["A", "B", "C", "D", "E"]
    body = server.requests[0]
    assert (body["model"], body["temperature"], body["max_tokens"], body["seed"]) == (
        "tiny", 0, 8, 0)  # fmt: skip
    assert stats.calls == 5 and stats.models == {"tiny-served"} and stats.fingerprints == {"fp1"}
    Client(Endpoint(url, "tiny", seed=None)).complete(_msgs("x"))
    assert "seed" not in server.requests[-1]


def test_cache_hits_and_cache_only(fake: tuple[FakeServer, str], tmp_path: Path) -> None:
    server, url = fake
    cache = ResponseCache(tmp_path / "llm.sqlite")
    endpoint = Endpoint(url, "tiny")
    Client(endpoint, cache=cache).complete_many([_msgs("a"), _msgs("b")])
    assert len(server.requests) == 2 and len(cache) == 2
    stats = CallStats()
    again = Client(endpoint, cache=cache, stats=stats).complete_many([_msgs("a"), _msgs("c")])
    assert len(server.requests) == 3 and (stats.cache_hits, stats.calls) == (1, 1)
    assert again[0] is not None and again[0].cached and again[0].model == "tiny-served"
    # Same conversation, other model or parameters: a different key.
    assert cache_key(endpoint, _msgs("a")) != cache_key(Endpoint(url, "other"), _msgs("a"))
    assert cache_key(endpoint, _msgs("a")) != cache_key(Endpoint(url, "tiny", seed=1), _msgs("a"))
    offline = CallStats()
    got = Client(endpoint, cache=ResponseCache(tmp_path / "llm.sqlite"), cache_only=True,
                 stats=offline).complete_many([_msgs("a"), _msgs("zzz")])  # fmt: skip
    assert got[0] is not None and got[1] is None and (offline.calls, offline.failures) == (0, 1)
    assert len(server.requests) == 3


def test_retries_then_success(fake: tuple[FakeServer, str]) -> None:
    server, url = fake
    server.status = [500, 429]
    stats = CallStats()
    answer = Client(Endpoint(url, "tiny"), backoff=0, stats=stats).complete(_msgs("a"))
    assert answer is not None and answer.text == "ok"
    assert (stats.calls, stats.retries, stats.failures) == (1, 2, 0)
    server.status = [503, 503, 503]
    stats = CallStats()
    assert Client(Endpoint(url, "tiny"), backoff=0, stats=stats).complete(_msgs("b")) is None
    assert (stats.retries, stats.failures) == (2, 1)


def test_fatal_status_stops_at_once(fake: tuple[FakeServer, str]) -> None:
    server, url = fake
    for status in (401, 403, 404):
        server.status = [status]
        with pytest.raises(LLMError, match=f"HTTP {status}"):
            Client(Endpoint(url, "tiny"), backoff=0).complete(_msgs("a"))
    with pytest.raises(LLMError, match="HTTP 404"):
        Client(Endpoint(url.replace("/v1", "/v2"), "tiny"), backoff=0).complete(_msgs("a"))


def test_early_stop_and_budget(fake: tuple[FakeServer, str], tmp_path: Path) -> None:
    server, url = fake
    server.status = [500] * 30
    with pytest.raises(LLMError, match="first 5 calls"):
        Client(Endpoint(url, "tiny"), backoff=0, retries=1, early_stop=5,
               workers=1).complete_many([_msgs(str(i)) for i in range(10)])  # fmt: skip
    server.status = []
    cache = ResponseCache(tmp_path / "llm.sqlite")
    with pytest.raises(BudgetExceeded, match="max_calls=3"):
        Client(Endpoint(url, "tiny"), cache=cache, max_calls=3,
               workers=1).complete_many([_msgs(str(i)) for i in range(6)])  # fmt: skip
    assert len(cache) == 3  # the answers before the cap are kept
    stats = CallStats()
    Client(Endpoint(url, "tiny"), cache=cache, max_calls=3, workers=1,
           stats=stats).complete_many([_msgs(str(i)) for i in range(6)])  # fmt: skip
    assert (stats.cache_hits, stats.calls) == (3, 3)


def test_api_key_from_environment_only(fake: tuple[FakeServer, str], tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    server, url = fake
    endpoint = Endpoint(url, "tiny", api_key_env="FAKE_LLM_KEY")
    with pytest.raises(LLMError, match="FAKE_LLM_KEY is not set"):
        Client(endpoint)
    monkeypatch.setenv("FAKE_LLM_KEY", "sk-secret-123")
    cache = ResponseCache(tmp_path / "llm.sqlite")
    Client(endpoint, cache=cache).complete(_msgs("a"))
    assert server.headers[-1]["Authorization"] == "Bearer sk-secret-123"
    cache.close()
    assert b"sk-secret-123" not in (tmp_path / "llm.sqlite").read_bytes()
    assert "sk-secret-123" not in repr(endpoint)
