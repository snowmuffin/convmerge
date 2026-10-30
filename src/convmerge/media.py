"""Check the media a training file refers to (``convmerge media``).

convmerge keeps images, audio and video by reference only: a path or URL
exactly as the source gave it. This reads every row the way ``convert``
reads it and checks each reference before training finds a broken one:

- a local path must name an existing file, resolved against ``root`` (by
  default the directory of the file being checked);
- an ``http(s)`` URL is checked only with ``check_urls``: a ``HEAD`` request
  (falling back to a one-byte ``GET`` for servers that refuse ``HEAD``) must
  answer below 400. Each URL is requested once, however often it appears;
- ``data:`` URIs are inline and pass; other schemes (``s3://``, ``gs://``)
  are counted as not checked;
- a media placeholder with no reference (``{"type": "image"}`` without a
  matching ``images`` entry) is ``unresolved``.

With ``output``, rows whose references all pass are written there and the
others to ``rejects``, both byte-for-byte as read. Nothing is downloaded.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from convmerge.io import ReadStats, iter_jsonl, refuse_overwrite

__all__ = ["MediaStats", "check_media"]

_SAMPLES = 3
PROBLEMS = ("missing_file", "unresolved", "bad_url", "unreachable")


@dataclass
class MediaStats:
    """Counts for one :func:`check_media` call."""

    rows: int = 0
    rows_with_media: int = 0
    kept: int = 0
    rejected: int = 0
    unreadable: int = 0
    invalid_json: int = 0
    references: Counter[str] = field(default_factory=Counter)
    """Media references by part type (``image``, ``audio``, ``video``)."""
    problems: Counter[str] = field(default_factory=Counter)
    """References by problem: ``missing_file``, ``unresolved``, ``bad_url``, ``unreachable``."""
    not_checked: Counter[str] = field(default_factory=Counter)
    """References left unchecked: ``url`` (without ``check_urls``) or another scheme."""
    samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "rows_with_media": self.rows_with_media,
            "kept": self.kept,
            "rejected": self.rejected,
            "unreadable": self.unreadable,
            "invalid_json": self.invalid_json,
            "references": dict(sorted(self.references.items())),
            "problems": {p: self.problems.get(p, 0) for p in PROBLEMS},
            "not_checked": dict(sorted(self.not_checked.items())),
            "samples": self.samples,
        }


def check_media(
    path: str | Path,
    *,
    root: str | Path | None = None,
    check_urls: bool = False,
    timeout: float = 10.0,
    workers: int = 8,
    output: str | Path | None = None,
    rejects: str | Path | None = None,
    encoding: str = "utf-8",
    stats: MediaStats | None = None,
) -> MediaStats:
    """Check every media reference in ``path``; see the module docstring."""
    from contextlib import ExitStack

    from convmerge._output import atomic_text_writer
    from convmerge.adapter_resolve import resolve_adapter

    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    refuse_overwrite([path], [output, rejects])
    base = Path(root) if root is not None else Path(path).resolve().parent
    st = stats if stats is not None else MediaStats()
    adapter = resolve_adapter("auto", None, pairs=True)

    def references(value: Any) -> list[tuple[str, str | None]] | None:
        examples = list(adapter(value)) if isinstance(value, dict) else []
        if not examples:
            return None
        refs = [(p.type, p.url) for ex in examples
                for msg in [*ex.messages, *(ex.rejected or [])] for p in msg.media]  # fmt: skip
        return list(dict.fromkeys(refs))  # a pair's shared prompt counts once

    # Each distinct URL is requested once, before the rows are judged.
    answers: dict[str, str | None] = {}
    if check_urls:
        urls = sorted({u for line in iter_jsonl(path, encoding=encoding)
                       for _, u in references(line.value) or () if u and _web(u)})  # fmt: skip
        with ThreadPoolExecutor(max_workers=workers) as pool:
            answers = dict(zip(urls, pool.map(lambda u: _probe(u, timeout), urls)))

    read = ReadStats()
    with ExitStack() as stack:
        fout = stack.enter_context(atomic_text_writer(output)) if output is not None else None
        frej = stack.enter_context(atomic_text_writer(rejects)) if rejects is not None else None
        for line in iter_jsonl(path, encoding=encoding, stats=read):
            st.rows += 1
            refs = references(line.value)
            bad = refs is None
            if refs is None:
                st.unreadable += 1
            elif refs:
                st.rows_with_media += 1
                st.references.update(kind for kind, _ in refs)
            for kind, ref in refs or ():
                problem, detail = _check(ref, base, check_urls, answers, st)
                if problem is None:
                    continue
                bad = True
                st.problems[problem] += 1
                samples = st.samples.setdefault(problem, [])
                if len(samples) < _SAMPLES:
                    samples.append({"line": line.number, "type": kind, "ref": ref,
                                    "detail": detail})  # fmt: skip
            target = frej if bad else fout
            st.rejected += bad
            st.kept += not bad
            if target is not None:
                target.write(line.raw + "\n")
    st.invalid_json = read.invalid_json
    return st


def _web(ref: str) -> bool:
    return _scheme(ref) in ("http", "https")


def _scheme(ref: str) -> str:
    head, sep, _ = ref.partition("://")
    if sep and head.isalnum():
        return head.lower()
    return "data" if ref.startswith("data:") else ""


def _check(
    ref: str | None,
    base: Path,
    check_urls: bool,
    answers: dict[str, str | None],
    st: MediaStats,
) -> tuple[str | None, str]:
    """``(problem, detail)`` for one reference; ``problem`` is ``None`` when it passes."""
    if not ref:
        return "unresolved", "a media placeholder with no path or URL"
    scheme = _scheme(ref)
    if scheme == "data":
        return None, ""
    if scheme in ("http", "https"):
        if not check_urls:
            st.not_checked["url"] += 1
            return None, ""
        answer = answers.get(ref)
        if answer is None:
            return None, ""
        return ("unreachable" if answer.startswith("unreachable") else "bad_url"), answer
    if scheme == "file":
        ref = ref[len("file://") :]
    elif scheme:
        st.not_checked[scheme] += 1
        return None, ""
    target = Path(ref).expanduser()
    if not target.is_absolute():
        target = base / target
    if target.is_file():
        return None, ""
    return "missing_file", f"no file at {target}"


def _probe(url: str, timeout: float) -> str | None:
    """``None`` when ``url`` answers below 400, else what went wrong."""
    for method, headers in (("HEAD", {}), ("GET", {"Range": "bytes=0-0"})):
        req = urllib.request.Request(url, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - http(s) only
                status = resp.status
        except urllib.error.HTTPError as e:
            status = e.code
        except (urllib.error.URLError, OSError, ValueError) as e:
            reason = getattr(e, "reason", e)
            return f"unreachable: {type(reason).__name__}: {reason}"[:160]
        if status < 400:
            return None
        if method == "HEAD" and status in (403, 405, 501):
            continue  # some servers refuse HEAD but serve GET
        return f"HTTP {status}"
    return f"HTTP {status}"


def summary_line(st: MediaStats) -> str:
    bad = st.problems.total()
    return (
        f"{st.rows_with_media:,} of {st.rows:,} rows refer to media "
        f"({st.references.total():,} references); {bad:,} broken, "
        f"{st.rejected:,} rows rejected"
    )
