"""Helpers shared by the command modules."""

from __future__ import annotations

import argparse


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
