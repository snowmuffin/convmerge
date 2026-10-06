"""Internal, per-file publication. This is not a multi-file transaction or a lock."""

from __future__ import annotations

import errno
import logging
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

logger = logging.getLogger(__name__)


def _warn_cleanup(path: Path, error: BaseException) -> None:
    # A custom logging handler must not replace the original processing error.
    try:
        logger.warning("could not clean temporary output %s: %s", path, error)
    except Exception:
        pass


def _new_file(target: Path) -> tuple[Path, int]:
    for _ in range(100):
        path = target.with_name(f".convmerge-{secrets.token_hex(12)}.tmp")
        try:
            # Unlike changing os.umask(), this honors the process's umask without
            # a process-global race. O_EXCL never opens another writer's file.
            # Python's text wrapper handles newlines; never add CRT translation.
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            fd = os.open(path, flags, 0o666)
        except FileExistsError:
            continue
        return path, fd
    raise FileExistsError("could not allocate a unique temporary output")


@contextmanager
def atomic_text_writer(path: str | Path, *, encoding: str = "utf-8") -> Iterator[TextIO]:
    """Keep the old file (or no file) visible until a complete new file is ready.

    Resolve output symlinks as ordinary open() does; replace their target, not
    the link. Preserve existing basic permission bits; new files honor umask.
    Not intended for devices, FIFOs, concurrent writers, or crash recovery.
    """
    target = Path(path).resolve()
    mode: int | None = None
    try:
        info = target.stat()
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISREG(info.st_mode):
            raise OSError(errno.EINVAL, "output is not a regular file", str(path))
        mode = stat.S_IMODE(info.st_mode) & 0o777
        if not mode & 0o222:
            raise PermissionError(errno.EACCES, "output is read-only", str(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    fd: int | None = None
    stream: TextIO | None = None
    try:
        temporary, fd = _new_file(target)
        if mode is not None:
            os.chmod(temporary, mode)
        stream = os.fdopen(fd, "w", encoding=encoding)
        fd = None  # ownership transferred to stream
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        os.replace(temporary, target)
    except BaseException:
        if stream is not None:
            try:
                stream.close()
            except BaseException as cleanup_error:
                _warn_cleanup(temporary or target, cleanup_error)
        elif fd is not None:
            try:
                os.close(fd)
            except OSError as cleanup_error:
                _warn_cleanup(temporary or target, cleanup_error)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                _warn_cleanup(temporary, cleanup_error)
        raise
