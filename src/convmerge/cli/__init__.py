"""CLI entrypoint for convmerge.

Each command group lives in its own module (``convert``, ``data``, ``fetch``,
``mix``) exposing ``_add_<command>`` parser builders and ``_cmd_<command>``
handlers; this module wires them together.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable

from convmerge import __version__
from convmerge._deprecation import deprecated_names
from convmerge.cli import convert as _convert
from convmerge.cli import data as _data
from convmerge.cli import fetch as _fetch
from convmerge.cli import mix as _mix
from convmerge.cli import run as _run

__all__ = ["main"]

__getattr__ = deprecated_names(
    __name__,
    {
        "FETCH_FILE_EXTENSIONS": (
            "convmerge.normalize.files:NORMALIZE_EXTENSIONS",
            "the CLI module exports only main()",
        ),
        "SIDECAR_SUFFIXES": (
            "convmerge.normalize.files:SIDECAR_SUFFIXES",
            "the CLI module exports only main()",
        ),
    },
)

_INSTALL_EXTRAS_EPILOG = """
optional dependencies (pip install "convmerge[EXTRA]"):
  (none)     convert, dedupe, turns on JSONL; normalize on .json/.jsonl only
  [fetch]      YAML manifests and GitHub sources (PyYAML)
  [fetch-all]  above + HuggingFace (datasets); same packages as fetch-hf
  [parquet]    .parquet input for normalize
  [preset]     YAML presets (convert --preset, preset validate)
  [tokens]     token lengths and chat-template checks (transformers + jinja2, no PyTorch)
  [all]        fetch-all + parquet + preset + tokens (full CLI feature set)
""".strip()


# Subcommands in help order: (parser builder, handler).
_COMMANDS: dict[str, tuple[Callable, Callable]] = {
    "convert": (_convert._add_convert, _convert._cmd_convert),
    "validate": (_convert._add_validate, _convert._cmd_validate),
    "formats": (_convert._add_formats, _convert._cmd_formats),
    "inspect": (_data._add_inspect, _data._cmd_inspect),
    "normalize": (_data._add_normalize, _data._cmd_normalize),
    "dedupe": (_data._add_dedupe, _data._cmd_dedupe),
    "turns": (_data._add_turns, _data._cmd_turns),
    "split": (_data._add_split, _data._cmd_split),
    "tokens": (_data._add_tokens, _data._cmd_tokens),
    "llamafactory-info": (_data._add_llamafactory_info, _data._cmd_llamafactory_info),
    "fetch": (_fetch._add_fetch, _fetch._cmd_fetch),
    "preset": (_convert._add_preset, _convert._cmd_preset),
    "mix": (_mix._add_mix, _mix._cmd_mix),
    "run": (_run._add_run, _run._cmd_run),
}


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging()
    _COMMANDS[args.command][1](args)


class _StderrHandler(logging.Handler):
    """Write ``warning: ...`` lines to whatever ``sys.stderr`` is at emit time."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            print(f"{record.levelname.lower()}: {record.getMessage()}", file=sys.stderr)
        except Exception:  # noqa: BLE001 - logging must never break the command
            self.handleError(record)


def _configure_logging() -> None:
    """Route library log records to stderr unless the host app configured logging."""
    log = logging.getLogger("convmerge")
    if logging.getLogger().handlers or any(isinstance(h, _StderrHandler) for h in log.handlers):
        return
    log.addHandler(_StderrHandler())
    log.setLevel(logging.INFO)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convmerge",
        description=(
            "Fetch, normalize, and convert heterogeneous chat/instruct datasets "
            "into a single LLM training format."
        ),
        epilog=_INSTALL_EXTRAS_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for add, _ in _COMMANDS.values():
        add(subparsers)
    return parser
