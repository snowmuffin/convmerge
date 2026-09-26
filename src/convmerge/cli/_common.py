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
