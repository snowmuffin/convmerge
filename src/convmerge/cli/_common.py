"""Helpers shared by the command modules."""

from __future__ import annotations

import argparse


def add_progress_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--progress",
        action="store_true",
        help="Log periodic row counts to stderr (or set CONVMERGE_PROGRESS=1)",
    )
