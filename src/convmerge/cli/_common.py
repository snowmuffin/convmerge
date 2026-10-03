"""Helpers shared by the command modules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_progress_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--progress",
        action="store_true",
        help="Log periodic row counts to stderr (or set CONVMERGE_PROGRESS=1)",
    )


def positive_int(value: str) -> int:
    """argparse type for options that must be >= 1."""
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return n


def refuse_existing(path: Path) -> None:
    """Exit with 2 when a template would replace a file that is already there."""
    if path.exists():
        print(
            f"error: {path} already exists; write the template to a new path, "
            "or delete the file first",
            file=sys.stderr,
        )
        sys.exit(2)


def config_errors() -> tuple[type[Exception], ...]:
    """Exceptions that mean a config file (preset, manifest, mix config) is invalid."""
    errors: tuple[type[Exception], ...] = (ValueError, ImportError, OSError)
    try:
        import yaml
    except ImportError:
        return errors
    return (*errors, yaml.YAMLError)


def encoding_advice(bad_bytes_before: int, encoding: str, *, has_encoding_flag: bool) -> str | None:
    """Advice when skipped lines had bytes that are not valid in ``encoding``.

    ``bad_bytes_before`` is :func:`convmerge.io.bad_bytes_seen` taken before the
    command read its input; ``None`` means no such bytes were read (the lines
    were skipped for another reason, such as broken JSON).
    """
    from convmerge.io import bad_bytes_seen

    if bad_bytes_seen() == bad_bytes_before:
        return None
    if has_encoding_flag:
        return (
            f"some lines are not valid {encoding}; if the file was saved in another "
            "encoding (cp949 on Korean Windows, for example), pass --encoding NAME "
            "(output is always UTF-8)"
        )
    return (
        f"some lines are not valid {encoding}; re-save the file as UTF-8 first "
        "(`convmerge convert --encoding NAME` reads other encodings)"
    )


def skipped_line_reason(path: Path, number: int, encoding: str = "utf-8") -> tuple[str, str | None]:
    """Why line ``number`` of ``path`` was skipped, and advice when there is a fix.

    Reads the file again up to that line, so only call it for a skipped line.
    The advice names ``convmerge normalize`` only for the file shapes it
    rewrites (a JSON array, or objects run together on one line); a broken,
    too deeply nested, or unpaired-surrogate line cannot be repaired by it.
    """
    from convmerge.io import _parse_line, iter_raw_lines
    from convmerge.normalize.jsonl import detect_jsonl_shape

    why: object = "unreadable"
    try:
        for n, raw in iter_raw_lines(path, encoding=encoding):
            if n == number:
                _, why = _parse_line(raw, check_bytes=True, encoding=encoding)
                break
    except OSError:
        pass
    if isinstance(why, json.JSONDecodeError):
        reason = f"not valid JSON: {why.msg}"
        try:
            shape = detect_jsonl_shape(path)
        except (OSError, ValueError, RecursionError):
            shape = "invalid"
        if shape in ("json_array", "single_line"):
            return reason, (
                f"the file is not one JSON object per line (looks like {shape}); "
                f"run `convmerge normalize -i {path} -o <out.jsonl>` first"
            )
        return reason, None
    if why == "nested too deeply":
        from convmerge.io import MAX_DEPTH

        return f"nested more than {MAX_DEPTH} levels deep", None
    return str(why or "unreadable"), None
