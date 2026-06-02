"""Lightweight, dependency-free progress reporting to stderr.

Progress is **opt-in**: every public helper here is a no-op unless explicitly
enabled, so default command output is unchanged. Enable it with the
``--progress`` CLI flag or by setting the ``CONVMERGE_PROGRESS`` environment
variable to a truthy value (``1``/``true``/``yes``/``on``).
"""

from __future__ import annotations

import os
import sys
import time
from typing import TextIO

# Rows between periodic progress lines for high-volume loops.
DEFAULT_INTERVAL = 50_000

_TRUTHY = {"1", "true", "yes", "on"}


def progress_enabled(flag: bool | None = None) -> bool:
    """Return whether progress reporting should be on.

    A truthy ``flag`` (typically the ``--progress`` CLI option) wins. Otherwise
    fall back to the ``CONVMERGE_PROGRESS`` environment variable.
    """
    if flag:
        return True
    return os.environ.get("CONVMERGE_PROGRESS", "").strip().lower() in _TRUTHY


class ProgressReporter:
    """Emit a progress line to ``stream`` every ``every`` updates.

    When ``enabled`` is False every method is a cheap no-op, so callers can wire
    it into hot loops unconditionally without branching at each row.
    """

    def __init__(
        self,
        label: str,
        *,
        every: int = DEFAULT_INTERVAL,
        enabled: bool = True,
        stream: TextIO | None = None,
    ) -> None:
        self.label = label
        self.every = max(1, every)
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.count = 0
        self._last_emitted = 0
        self._start = time.monotonic()

    def update(self, n: int = 1) -> None:
        """Record ``n`` processed items, emitting a line at each interval."""
        if not self.enabled:
            return
        self.count += n
        if self.count - self._last_emitted >= self.every:
            self._last_emitted = self.count
            self._emit()

    def done(self) -> None:
        """Emit a final line unless the last ``update`` already did."""
        if not self.enabled:
            return
        if self.count != self._last_emitted or self.count == 0:
            self._emit(final=True)

    def _emit(self, *, final: bool = False) -> None:
        elapsed = time.monotonic() - self._start
        rate = self.count / elapsed if elapsed > 0 else 0.0
        tag = "done" if final else "progress"
        print(
            f"[{tag}] {self.label}: {self.count:,} rows ({rate:,.0f}/s)",
            file=self.stream,
        )
