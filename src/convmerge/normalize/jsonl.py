"""Load, detect shape of, and normalize JSON / JSONL files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

JSONLShape = Literal["jsonl", "single_line", "json_array", "invalid", "empty"]

# Bytes scanned from the head of a file to decide its shape without loading everything.
_HEAD_PEEK_BYTES = 65536

# File extensions recognised by the directory-level sweeps below.
NORMALIZE_EXTENSIONS: tuple[str, ...] = (".parquet", ".json", ".jsonl")

# Max bytes read from the head of a file by :func:`head_preview` when no newline is found.
_HEAD_PREVIEW_DEFAULT_BYTES = 8192


def load_jsonl(path: str | Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    """Load a well-formed JSONL file into a list of dicts.

    Lines that are empty are skipped. Any line that fails to parse aborts the
    load and returns an empty list, mirroring the permissive loader used in the
    Neura_Data base notebook.
    """
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if max_rows is not None and len(out) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                # Caller gets an empty list; the failing location is still printed
                # so the user can fix the source file.
                print(f"[JSONL ERROR] {path} line {i}: {e} :: {line[:80]!r}")
                return []
            if isinstance(obj, dict):
                out.append(obj)
    return out


def iter_json_records(path: str | Path, *, max_rows: int | None = None) -> Iterator[dict[str, Any]]:
    """Iterate dict records from a ``.json`` or ``.jsonl`` file.

    For ``.json`` the file is expected to contain a top-level array of objects
    (or a single object, which is yielded as one record). For ``.jsonl`` one
    JSON object per line is expected.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        yielded = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
                    yielded += 1
                    if max_rows is not None and yielded >= max_rows:
                        return
        return

    if suffix == ".json":
        with p.open(encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # Fallback: treat it as JSONL (some datasets ship .json that is really JSONL).
                yield from iter_json_records(p.with_suffix(".jsonl"), max_rows=max_rows)
                return
        if isinstance(data, dict):
            yield data
            return
        if isinstance(data, list):
            for i, item in enumerate(data):
                if max_rows is not None and i >= max_rows:
                    return
                if isinstance(item, dict):
                    yield item
        return

    raise ValueError(f"Unsupported file extension for iter_json_records: {p.suffix!r}")


def detect_jsonl_shape(path: str | Path) -> JSONLShape:
    """Classify the layout of a ``.json`` / ``.jsonl`` file without loading it all.

    - ``jsonl``       : multiple non-empty lines, each a JSON object.
    - ``single_line`` : exactly one line containing one or more JSON objects
      (common for dumps that forgot to add newlines; e.g. ``{...}{...}``).
    - ``json_array``  : one line that parses as a JSON array.
    - ``empty``       : file has no non-empty content.
    - ``invalid``     : none of the above.
    """
    p = Path(path)
    with p.open("rb") as f:
        head = f.read(_HEAD_PEEK_BYTES)
    if not head.strip():
        return "empty"
    text = head.decode("utf-8", errors="ignore")

    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    if len(non_empty_lines) >= 2:
        return "jsonl"

    first = non_empty_lines[0].strip() if non_empty_lines else ""
    if not first:
        return "empty"

    try:
        parsed = json.loads(first)
    except json.JSONDecodeError:
        # Concatenated objects like ``{...}{...}{...}`` are not valid JSON but
        # can be recovered by splitting on ``}{``.
        if first.count("}{") > 0:
            return "single_line"
        return "invalid"

    if isinstance(parsed, list):
        return "json_array"
    if isinstance(parsed, dict):
        return "single_line"
    return "invalid"


def normalize_to_jsonl(src: str | Path, dst: str | Path) -> int:
    """Rewrite ``src`` into a well-formed JSONL file at ``dst``.

    Returns the number of records written. Handles:

    - Already-valid JSONL (copies while skipping empty lines).
    - Top-level JSON arrays (``[ {...}, {...}, ... ]``).
    - Single-line concatenated objects (``{...}{...}{...}``).
    """
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)

    shape = detect_jsonl_shape(src_p)
    if shape == "empty":
        dst_p.write_text("", encoding="utf-8")
        return 0

    if shape == "jsonl":
        return _rewrite_jsonl(src_p, dst_p)
    if shape == "json_array":
        return _rewrite_json_array(src_p, dst_p)
    if shape == "single_line":
        return _rewrite_single_line(src_p, dst_p)
    raise ValueError(f"Cannot normalize {src_p}: shape detected as {shape!r}")


def _rewrite_jsonl(src: Path, dst: Path) -> int:
    n = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            # Validate each line so the output is guaranteed to round-trip.
            json.loads(line)
            fout.write(line + "\n")
            n += 1
    return n


def _rewrite_json_array(src: Path, dst: Path) -> int:
    with src.open(encoding="utf-8") as fin:
        data = json.load(fin)
    if not isinstance(data, list):
        raise ValueError(f"{src} is not a JSON array")
    n = 0
    with dst.open("w", encoding="utf-8") as fout:
        for obj in data:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def _rewrite_single_line(src: Path, dst: Path) -> int:
    text = src.read_text(encoding="utf-8").strip()
    records = _split_concatenated_objects(text)
    n = 0
    with dst.open("w", encoding="utf-8") as fout:
        for obj in records:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def _split_concatenated_objects(text: str) -> list[Any]:
    """Parse ``{...}{...}{...}`` by walking the string with a brace counter.

    Respects strings and escapes so characters inside JSON strings are not
    counted as braces.
    """
    out: list[Any] = []
    decoder = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, end = decoder.raw_decode(text, i)
        out.append(obj)
        i = end
    return out


def normalize_dir(
    src_dir: str | Path,
    dst_dir: str | Path,
    *,
    extensions: Iterable[str] = NORMALIZE_EXTENSIONS,
    on_error: Literal["skip", "raise"] = "skip",
    overwrite: bool = True,
) -> tuple[int, int]:
    """Recursively normalize parquet/json/jsonl files under a directory tree.

    Each matching file under ``src_dir`` is converted into a ``.jsonl`` file at
    the mirrored path under ``dst_dir``. Parquet files are streamed via
    :func:`convmerge.normalize.parquet.parquet_to_jsonl` (requires the
    ``parquet`` extra); json / jsonl files are funnelled through
    :func:`normalize_to_jsonl`.

    Returns ``(n_files, n_records)`` — how many files were written and the
    total number of records across them.

    ``on_error='skip'`` (the default) prints a failure line and continues;
    ``on_error='raise'`` re-raises the first exception.
    """
    src_p = Path(src_dir)
    dst_p = Path(dst_dir)
    if not src_p.is_dir():
        raise NotADirectoryError(f"Not a directory: {src_p}")

    exts_lower = tuple(e.lower() for e in extensions)
    n_files = 0
    n_records = 0

    for in_path in sorted(src_p.rglob("*")):
        if not in_path.is_file():
            continue
        suffix = in_path.suffix.lower()
        if suffix not in exts_lower:
            continue
        rel = in_path.relative_to(src_p).with_suffix(".jsonl")
        out_path = dst_p / rel
        if out_path.exists():
            if overwrite:
                try:
                    out_path.unlink()
                except OSError:
                    pass
            else:
                continue
        try:
            n = _normalize_any(in_path, out_path)
        except Exception as e:  # noqa: BLE001
            if on_error == "raise":
                raise
            print(f"[fail] {in_path}: {type(e).__name__}: {e}")
            continue
        n_files += 1
        n_records += n

    return n_files, n_records


def _normalize_any(src: Path, dst: Path) -> int:
    """Dispatch a single file to the right normalizer based on extension."""
    suffix = src.suffix.lower()
    if suffix == ".parquet":
        # Imported lazily so callers that never touch parquet do not need
        # the ``parquet`` extra.
        from convmerge.normalize.parquet import parquet_to_jsonl

        dst.parent.mkdir(parents=True, exist_ok=True)
        return parquet_to_jsonl(src, dst)
    return normalize_to_jsonl(src, dst)


def prune_by_suffix(
    dir_path: str | Path,
    suffixes: Iterable[str],
    *,
    dry_run: bool = False,
) -> list[Path]:
    """Recursively delete files whose filename ends with any of ``suffixes``.

    Intended for removing dataset-level junk files like ``*-res.jsonl`` or
    ``config.jsonl`` that slip through a bulk download. With ``dry_run=True``,
    returns the list of matching paths without deleting them.

    Returns the list of paths that were removed (or would be removed).
    """
    root = Path(dir_path)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    tail: tuple[str, ...] = tuple(suffixes)
    if not tail:
        return []

    removed: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if not any(name.endswith(t) for t in tail):
            continue
        removed.append(p)
        if not dry_run:
            try:
                p.unlink()
            except OSError as e:
                print(f"[prune_by_suffix] cannot remove {p}: {e}")
    return removed


def head_preview(path: str | Path, *, max_bytes: int = _HEAD_PREVIEW_DEFAULT_BYTES) -> str:
    """Return the first line (or the first ``max_bytes`` bytes) of a file as text.

    Useful for "what does this file look like?" sanity checks on large raw
    dumps without loading them. Binary bytes are decoded leniently.
    """
    p = Path(path)
    with p.open("rb") as f:
        chunk = f.read(max_bytes)
    text = chunk.decode("utf-8", errors="ignore")
    if "\n" in text:
        return text.split("\n", 1)[0]
    return text if len(chunk) < max_bytes else text + "..."


def head_preview_dir(
    dir_path: str | Path,
    *,
    extensions: Iterable[str] | None = None,
    max_bytes: int = _HEAD_PREVIEW_DEFAULT_BYTES,
) -> dict[Path, str]:
    """Run :func:`head_preview` on every file under a directory tree.

    If ``extensions`` is given, only files whose suffix matches are previewed
    (case-insensitive). The result is keyed by the original file path.
    """
    root = Path(dir_path)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    exts = tuple(e.lower() for e in extensions) if extensions else None

    out: dict[Path, str] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if exts is not None and p.suffix.lower() not in exts:
            continue
        out[p] = head_preview(p, max_bytes=max_bytes)
    return out


def scan_shapes(
    dir_path: str | Path,
    *,
    extensions: Iterable[str] = (".json", ".jsonl"),
) -> dict[Path, JSONLShape]:
    """Classify every ``.json`` / ``.jsonl`` file under a directory by shape.

    Returns a mapping of path -> :data:`JSONLShape`. Use alongside
    :func:`normalize_dir` to audit what a raw dump actually contains before
    committing to a pipeline.
    """
    root = Path(dir_path)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    exts = tuple(e.lower() for e in extensions)

    out: dict[Path, JSONLShape] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        out[p] = detect_jsonl_shape(p)
    return out
