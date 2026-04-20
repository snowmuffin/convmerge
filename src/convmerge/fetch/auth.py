"""Token resolution and redaction helpers.

Tokens may be supplied three ways, in priority order:

1. An explicit value passed to ``resolve_token(explicit=...)`` (e.g. from CLI).
2. The contents of a file at ``TokenSpec.file``.
3. The environment variable named by ``TokenSpec.env``.

Tokens never appear in plan output, URLs, or logs; use :func:`redact_url` when
echoing any URL that might have embedded credentials.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenSpec:
    """Where to look for a single token."""

    env: str | None = None
    file: str | None = None


@dataclass(frozen=True)
class AuthConfig:
    """Auth block parsed from the YAML manifest's ``auth`` section."""

    hf: TokenSpec = TokenSpec()
    github: TokenSpec = TokenSpec()


def resolve_token(spec: TokenSpec, *, explicit: str | None = None) -> str | None:
    """Return the first non-empty token from (explicit, file, env), or ``None``."""
    if explicit is not None:
        val = explicit.strip()
        if val:
            return val
    if spec.file:
        p = Path(spec.file).expanduser()
        if p.is_file():
            val = p.read_text(encoding="utf-8").strip()
            if val:
                return val
    if spec.env:
        val = (os.environ.get(spec.env) or "").strip()
        if val:
            return val
    return None


# Pattern matches a userinfo section in an HTTPS URL (``user:token@host``).
_USERINFO_RE = re.compile(r"(https?://)[^/@\s]+@")


def redact_url(url: str) -> str:
    """Remove any ``user:token@`` userinfo from ``url`` for safe logging."""
    return _USERINFO_RE.sub(r"\1", url)
