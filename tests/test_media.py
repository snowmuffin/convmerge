"""convmerge media: local files, URLs (against a local server), placeholders, CLI."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from convmerge.cli import main
from convmerge.media import MediaStats, check_media


class _Handler(BaseHTTPRequestHandler):
    hits: list[str] = []

    def _answer(self) -> None:
        type(self).hits.append(f"{self.command} {self.path}")
        if self.path == "/ok.png" or (self.path == "/nohead.png" and self.command == "GET"):
            self.send_response(200 if self.command == "HEAD" else 206)
        elif self.path == "/nohead.png":
            self.send_response(405)
        else:
            self.send_response(404)
        self.end_headers()

    do_HEAD = do_GET = _answer

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture()
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")  # never route the local server via a proxy
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    _Handler.hits = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _row(*refs: str | None, text: str = "q") -> dict:
    parts: list[dict] = [{"type": "image", **({"url": r} if r else {})} for r in refs]
    return {"messages": [{"role": "user", "content": [*parts, {"type": "text", "text": text}]},
                         {"role": "assistant", "content": "a"}]}  # fmt: skip


def _write(path: Path, rows: list) -> Path:
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows),
                    encoding="utf-8")  # fmt: skip
    return path


def test_local_files_placeholders_and_layouts(tmp_path: Path) -> None:
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "a.png").write_bytes(b"x")
    rows = [
        _row("img/a.png"),
        _row("img/missing.png"),
        _row(None),
        {"images": ["img/a.png"], "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "q"}]},
            {"role": "assistant", "content": "a"}]},
        _row("data:image/png;base64,AAAA", "s3://bucket/x.png", "https://example.invalid/x.png"),
        {"messages": [{"role": "user", "content": "text only"},
                      {"role": "assistant", "content": "a"}]},
        {"foo": 1},
    ]  # fmt: skip
    src = _write(tmp_path / "data.jsonl", rows)
    st = check_media(src, output=tmp_path / "ok.jsonl", rejects=tmp_path / "bad.jsonl")
    assert (st.rows, st.rows_with_media, st.kept, st.rejected, st.unreadable) == (7, 5, 4, 3, 1)
    assert st.problems == {"missing_file": 1, "unresolved": 1}
    assert st.not_checked == {"url": 1, "s3": 1}
    assert st.references["image"] == 7
    assert st.samples["missing_file"][0]["line"] == 2
    kept = (tmp_path / "ok.jsonl").read_text().splitlines()
    assert [json.loads(k) for k in kept] == [rows[0], rows[3], rows[4], rows[5]]
    assert len((tmp_path / "bad.jsonl").read_text().splitlines()) == 3

    # Relative paths resolve against --root; absolute and file:// paths as given.
    other = tmp_path / "elsewhere"
    other.mkdir()
    moved = _write(other / "data.jsonl", [_row("img/a.png"), _row(f"file://{tmp_path}/img/a.png"),
                                          _row(str(tmp_path / "img" / "a.png"))])  # fmt: skip
    assert check_media(moved).problems == {"missing_file": 1}
    assert check_media(moved, root=tmp_path).problems.total() == 0


def test_urls_are_checked_once_and_only_when_asked(tmp_path: Path, server: str) -> None:
    rows = [_row(f"{server}/ok.png"), _row(f"{server}/ok.png"), _row(f"{server}/gone.png"),
            _row(f"{server}/nohead.png"), _row("http://127.0.0.1:9/closed.png")]  # fmt: skip
    src = _write(tmp_path / "urls.jsonl", rows)
    assert check_media(src).problems.total() == 0 and _Handler.hits == []  # no network
    st = check_media(src, check_urls=True, timeout=2, workers=2)
    assert st.problems == {"bad_url": 1, "unreachable": 1}
    assert st.samples["bad_url"][0]["detail"] == "HTTP 404"
    assert sorted(_Handler.hits) == ["GET /nohead.png", "HEAD /gone.png", "HEAD /nohead.png",
                                     "HEAD /ok.png"]  # fmt: skip
    with pytest.raises(ValueError, match="workers"):
        check_media(src, workers=0)


def test_preference_pairs_count_shared_media_once(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    prompt = [{"role": "user", "content": [{"type": "image", "url": "a.png"},
                                           {"type": "text", "text": "q"}]}]  # fmt: skip
    pair = {"prompt": prompt, "chosen": [{"role": "assistant", "content": "good"}],
            "rejected": [{"role": "assistant", "content": "bad"}]}  # fmt: skip
    st = MediaStats()
    check_media(_write(tmp_path / "p.jsonl", [pair]), stats=st)
    assert (st.rows_with_media, st.references["image"], st.kept) == (1, 1, 1)


def test_cli(tmp_path: Path, capsys) -> None:
    src = _write(tmp_path / "d.jsonl", [_row("missing.png"), _row("https://example.invalid/x")])
    with pytest.raises(SystemExit) as e:
        main(["media", "-i", str(src)])
    assert e.value.code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["problems"]["missing_file"] == 1
    assert "1 URL not checked; add --check-urls" in captured.err
    main(["media", "-i", str(src), "-o", str(tmp_path / "ok.jsonl")])  # -o: exit 0
    assert len((tmp_path / "ok.jsonl").read_text().splitlines()) == 1
    with pytest.raises(SystemExit) as e:
        main(["media", "-i", str(tmp_path / "missing.jsonl")])
    assert e.value.code == 1
