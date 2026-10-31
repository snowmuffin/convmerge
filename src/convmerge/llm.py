"""Calls to an OpenAI-compatible chat endpoint, with a cache, a host policy, retries and a budget.

This is the one place convmerge talks to a model; ``convmerge label`` is
built on it. Design: ``docs/design/labeling.md``. Internal: the names here
may change until a second command uses them.

- **Protocol**: ``POST {url}/chat/completions`` with ``model``, ``messages``,
  ``temperature`` 0, ``max_tokens`` and (unless ``seed`` is ``None``) ``seed``.
- **Hosts**: loopback, private, link-local and ``*.local`` hosts are allowed;
  any other host needs ``allow_external=True``, checked before anything is
  sent.
- **Keys**: read from the environment variable ``api_key_env`` names; never
  stored, logged or put in an error message.
- **Cache**: SQLite, keyed by host, model, messages and generation
  parameters. A cached request is never sent again; ``cache_only`` turns a
  miss into a failed answer instead of a call.
- **Errors**: connection errors, timeouts, HTTP 429 and 5xx are retried with
  exponential backoff; HTTP 401, 403 and 404 stop at once (every request
  would fail the same way), and so does a run whose first ``early_stop``
  calls all fail.
- **Budget**: ``max_calls`` caps network calls (cache hits are free);
  reaching it raises :class:`BudgetExceeded` after the answers so far are
  cached.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import sqlite3
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

__all__ = [
    "Answer",
    "BudgetExceeded",
    "CallStats",
    "Client",
    "Endpoint",
    "ExternalHostError",
    "LLMError",
    "ResponseCache",
    "host_allowed",
]

Messages = list[dict[str, str]]
_FATAL_STATUS = {401, 403, 404}


class LLMError(RuntimeError):
    """A failure that stops the whole run (bad key, model or URL; budget; policy)."""


class ExternalHostError(LLMError):
    """The endpoint is not a local or private host and external calls are not allowed."""


class BudgetExceeded(LLMError):
    """``max_calls`` network calls were made; the answers so far are cached."""


@dataclass(frozen=True)
class Endpoint:
    """Where and how to call: an OpenAI-compatible base URL (``.../v1``) and a model."""

    url: str
    model: str
    api_key_env: str | None = None
    allow_external: bool = False
    timeout: float = 60.0
    max_tokens: int = 16
    seed: int | None = 0

    def __post_init__(self) -> None:
        parts = urlsplit(self.url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"endpoint: expected an http(s) URL, got {self.url!r}")
        if not self.model:
            raise ValueError("endpoint: a model name is required")
        if self.timeout <= 0 or self.max_tokens < 1:
            raise ValueError("endpoint: timeout and max_tokens must be positive")

    @property
    def host(self) -> str:
        parts = urlsplit(self.url)
        return f"{parts.hostname}:{parts.port}" if parts.port else str(parts.hostname)

    def params(self) -> dict[str, Any]:
        """The generation parameters sent with every request (part of the cache key)."""
        out: dict[str, Any] = {"temperature": 0, "max_tokens": self.max_tokens}
        if self.seed is not None:
            out["seed"] = self.seed
        return out


@dataclass(frozen=True)
class Answer:
    text: str
    cached: bool
    model: str | None = None
    """The ``model`` the server reported, which can be more specific than the one asked for."""
    fingerprint: str | None = None


@dataclass
class CallStats:
    requests: int = 0
    calls: int = 0
    """Requests sent over the network (retries not counted)."""
    cache_hits: int = 0
    retries: int = 0
    failures: int = 0
    """Requests with no answer after every retry (or not in the cache with ``cache_only``)."""
    models: set[str] = field(default_factory=set)
    fingerprints: set[str] = field(default_factory=set)

    def to_report(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
            "failures": self.failures,
            "models": sorted(self.models),
            "fingerprints": sorted(self.fingerprints),
        }


def host_allowed(url: str) -> bool:
    """Whether ``url`` names a loopback, private, link-local or ``*.local`` host."""
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    if host == "localhost" or host.endswith((".localhost", ".local")):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def cache_key(endpoint: Endpoint, messages: Messages) -> str:
    payload = {"host": endpoint.host, "model": endpoint.model, "messages": messages,
               "params": endpoint.params()}  # fmt: skip
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """Answers by request key, in one SQLite file. Safe to share between threads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, text TEXT NOT NULL,"
            " model TEXT, fingerprint TEXT, created REAL NOT NULL)"
        )
        self._db.commit()

    def get(self, key: str) -> Answer | None:
        with self._lock:
            row = self._db.execute(
                "SELECT text, model, fingerprint FROM answers WHERE key = ?", (key,)
            ).fetchone()
        return Answer(row[0], True, row[1], row[2]) if row else None

    def put_many(self, items: Sequence[tuple[str, Answer]]) -> None:
        if not items:
            return
        now = time.time()
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?)",
                [(k, a.text, a.model, a.fingerprint, now) for k, a in items],
            )
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM answers").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()


class Client:
    """Send chat requests to one endpoint; see the module docstring for the rules."""

    def __init__(
        self,
        endpoint: Endpoint,
        *,
        cache: ResponseCache | None = None,
        cache_only: bool = False,
        max_calls: int | None = None,
        workers: int = 4,
        retries: int = 3,
        backoff: float = 1.0,
        early_stop: int = 20,
        stats: CallStats | None = None,
    ) -> None:
        if workers < 1 or retries < 1:
            raise ValueError("workers and retries must be at least 1")
        if max_calls is not None and max_calls < 0:
            raise ValueError("max_calls must not be negative")
        if not cache_only and not endpoint.allow_external and not host_allowed(endpoint.url):
            raise ExternalHostError(
                f"{endpoint.host} is not a local or private host; pass --allow-external "
                "(recipe: allow_external: true) to send rows to it"
            )
        self._key: str | None = None
        if endpoint.api_key_env:
            key = os.environ.get(endpoint.api_key_env)
            if not key and not cache_only:
                raise LLMError(f"environment variable {endpoint.api_key_env} is not set")
            self._key = key
        self.endpoint = endpoint
        self.cache = cache
        self.cache_only = cache_only
        self.max_calls = max_calls
        self.workers = workers
        self.retries = retries
        self.backoff = backoff
        self.early_stop = early_stop
        self.stats = stats if stats is not None else CallStats()
        self._lock = threading.Lock()
        self._answered = 0

    def complete_many(self, requests: Sequence[Messages]) -> list[Answer | None]:
        """Answers in request order; ``None`` where no answer came (counted in stats)."""
        out: list[Answer | None] = [None] * len(requests)
        todo: list[tuple[int, str]] = []
        for i, messages in enumerate(requests):
            self.stats.requests += 1
            key = cache_key(self.endpoint, messages)
            hit = self.cache.get(key) if self.cache is not None else None
            if hit is not None:
                self.stats.cache_hits += 1
                out[i] = hit
                self._note(hit)
            elif self.cache_only:
                self.stats.failures += 1
            else:
                todo.append((i, key))
        if not todo:
            return out
        fresh: list[tuple[str, Answer]] = []
        error: BaseException | None = None
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                results = pool.map(lambda job: self._call(requests[job[0]]), todo)
                for (i, key), answer in zip(todo, results):
                    if answer is None:
                        continue
                    out[i] = answer
                    fresh.append((key, answer))
        except LLMError as e:
            error = e
        finally:
            if self.cache is not None:
                self.cache.put_many(fresh)
        if error is not None:
            raise error
        return out

    def complete(self, messages: Messages) -> Answer | None:
        return self.complete_many([messages])[0]

    def _note(self, answer: Answer) -> None:
        if answer.model:
            self.stats.models.add(answer.model)
        if answer.fingerprint:
            self.stats.fingerprints.add(answer.fingerprint)

    def _call(self, messages: Messages) -> Answer | None:
        with self._lock:
            if self.max_calls is not None and self.stats.calls >= self.max_calls:
                raise BudgetExceeded(
                    f"reached max_calls={self.max_calls}; the answers so far are cached, "
                    "so the next run continues from there"
                )
            self.stats.calls += 1
        for attempt in range(self.retries):
            try:
                answer = self._post(messages)
            except _Retryable:
                if attempt + 1 < self.retries:
                    with self._lock:
                        self.stats.retries += 1
                    time.sleep(self.backoff * (2**attempt))
                continue
            with self._lock:
                self._answered += 1
                self._note(answer)
            return answer
        with self._lock:
            self.stats.failures += 1
            if self._answered == 0 and self.stats.failures >= self.early_stop:
                raise LLMError(
                    f"the first {self.stats.failures} calls to {self.endpoint.host} all failed; "
                    "check that the server is up and serves this model"
                )
        return None

    def _post(self, messages: Messages) -> Answer:
        import requests

        body = {"model": self.endpoint.model, "messages": messages, **self.endpoint.params()}
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        url = self.endpoint.url.rstrip("/") + "/chat/completions"
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=self.endpoint.timeout)
        except requests.RequestException as e:
            raise _Retryable(type(e).__name__) from None
        if resp.status_code in _FATAL_STATUS:
            raise LLMError(
                f"{self.endpoint.host} answered HTTP {resp.status_code} "
                f"({resp.reason}); check the URL, the model name and the API key"
            )
        if resp.status_code == 429 or resp.status_code >= 500:
            raise _Retryable(f"HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise LLMError(f"{self.endpoint.host} answered HTTP {resp.status_code}: "
                           f"{resp.text[:200]}")  # fmt: skip
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise _Retryable("malformed response") from None
        if not isinstance(text, str):
            raise _Retryable("malformed response")
        return Answer(text, False, data.get("model"), data.get("system_fingerprint"))


class _Retryable(Exception):
    pass
