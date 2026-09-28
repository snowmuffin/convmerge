"""Load, detect shape of, and normalize JSON / JSONL files."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from convmerge.io import JsonlDecodeError, iter_jsonl
from convmerge.lfs import ensure_not_lfs_pointer

logger = logging.getLogger(__name__)

JSONLShape = Literal["jsonl", "jsonl_of_arrays", "single_line", "json_array", "invalid", "empty"]

# Key a top-level JSON array record is wrapped under, so that a "one
# conversation (list of turns) per line" file becomes object records the
# chat adapter reads (its default conversation keys include this one).
DEFAULT_ARRAY_KEY = "conversation"

# Bytes scanned from the head of a file to decide its shape without loading everything.
_HEAD_PEEK_BYTES = 65536


def iter_json_records(
    path: str | Path,
    *,
    max_rows: int | None = None,
    array_key: str | None = DEFAULT_ARRAY_KEY,
) -> Iterator[dict[str, Any]]:
    """Iterate dict records from a ``.json`` or ``.jsonl`` file.

    For ``.json`` the file is expected to contain a top-level array of objects
    (or a single object, which is yielded as one record). For ``.jsonl`` one
    JSON object per line is expected. Records that are themselves arrays are
    yielded as ``{array_key: [...]}`` — the shape ``normalize`` writes — or
    skipped when ``array_key`` is ``None``.
    """
    p = Path(path)
    ensure_not_lfs_pointer(p)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl_records(p, max_rows=max_rows, array_key=array_key)
        return

    if suffix == ".json":
        with p.open(encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # Fallback: treat the same file as JSONL (some datasets ship
                # ``.json`` files that are really line-delimited).
                yield from _iter_jsonl_records(p, max_rows=max_rows, array_key=array_key)
                return
        if isinstance(data, dict):
            yield data
            return
        if isinstance(data, list):
            for i, item in enumerate(data):
                if max_rows is not None and i >= max_rows:
                    return
                record = _as_record(item, array_key)
                if record is not None:
                    yield record
        return

    raise ValueError(f"Unsupported file extension for iter_json_records: {p.suffix!r}")


def _as_record(value: Any, array_key: str | None) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and array_key is not None:
        return {array_key: value}
    return None


def _iter_jsonl_records(
    p: Path, *, max_rows: int | None, array_key: str | None
) -> Iterator[dict[str, Any]]:
    yielded = 0
    for line in iter_jsonl(p, on_error="raise"):
        record = _as_record(line.value, array_key)
        if record is not None:
            yield record
            yielded += 1
            if max_rows is not None and yielded >= max_rows:
                return


def detect_jsonl_shape(path: str | Path) -> JSONLShape:
    """Classify the layout of a ``.json`` / ``.jsonl`` file without loading it all.

    - ``jsonl``       : multiple non-empty lines, each a JSON object.
    - ``jsonl_of_arrays`` : multiple non-empty lines, each a JSON array (e.g.
      one conversation per line as a list of turns).
    - ``single_line`` : exactly one line containing one or more JSON objects
      (common for dumps that forgot to add newlines; e.g. ``{...}{...}``).
    - ``json_array``  : one line that parses as a JSON array.
    - ``empty``       : file has no non-empty content.
    - ``invalid``     : none of the above.
    """
    p = Path(path)
    ensure_not_lfs_pointer(p)
    with p.open("rb") as f:
        head = f.read(_HEAD_PEEK_BYTES)
    if not head.strip():
        return "empty"
    text = head.decode("utf-8", errors="ignore").removeprefix("\ufeff")

    # A leading '[' means a top-level JSON array. Pretty-printed arrays span
    # many lines, so detect this before the multi-line ``jsonl`` heuristic
    # below — otherwise an array like ``[\n  {...},\n  {...}\n]`` is mistaken
    # for line-delimited JSON and breaks normalization.
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    if text.lstrip()[:1] == "[":
        # ...unless the first line is a complete array and more lines follow:
        # that is JSONL whose records are arrays.
        if len(non_empty_lines) >= 2 and _is_json_array(non_empty_lines[0]):
            return "jsonl_of_arrays"
        return "json_array"

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


def _is_json_array(line: str) -> bool:
    try:
        return isinstance(json.loads(line.strip().removeprefix("\ufeff")), list)
    except json.JSONDecodeError:
        return False


def normalize_to_jsonl(
    src: str | Path, dst: str | Path, *, array_key: str = DEFAULT_ARRAY_KEY
) -> int:
    """Rewrite ``src`` into a well-formed JSONL file at ``dst``.

    Returns the number of records written. Handles:

    - Already-valid JSONL (copies while skipping empty lines).
    - JSONL whose lines are arrays: each array becomes ``{array_key: [...]}``.
    - Top-level JSON arrays (``[ {...}, {...}, ... ]``); array elements that
      are themselves arrays are wrapped the same way.
    - Single-line concatenated objects (``{...}{...}{...}``).
    """
    src_p = Path(src)
    ensure_not_lfs_pointer(src_p)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        shape = detect_jsonl_shape(src_p)
        if shape == "empty":
            dst_p.write_text("", encoding="utf-8")
            return 0
        if shape == "jsonl":
            return _rewrite_jsonl(src_p, dst_p)
        if shape == "jsonl_of_arrays":
            return _rewrite_jsonl_of_arrays(src_p, dst_p, array_key)
        if shape == "json_array":
            return _rewrite_json_array(src_p, dst_p, array_key)
        if shape == "single_line":
            return _rewrite_single_line(src_p, dst_p)
    except (UnicodeDecodeError, json.JSONDecodeError, JsonlDecodeError) as e:
        raise ValueError(f"Cannot normalize {src_p}: {e}") from None
    except RecursionError:
        raise ValueError(f"Cannot normalize {src_p}: nested too deeply") from None
    raise ValueError(
        f"Cannot normalize {src_p}: not JSON, JSONL, or a JSON array "
        "(check that the file is complete and UTF-8)"
    )


def _rewrite_jsonl(src: Path, dst: Path) -> int:
    n = 0
    with dst.open("w", encoding="utf-8") as fout:
        try:
            # Raise on the first bad line so the output is guaranteed to round-trip.
            for line in iter_jsonl(src, on_error="raise"):
                fout.write(line.raw + "\n")
                n += 1
        except JsonlDecodeError as err:
            if err.raw.endswith(","):
                raise ValueError(
                    f"Cannot normalize {src}: trailing comma at line {err.line_number}; "
                    "remove the comma or provide valid JSONL"
                ) from None
            raise ValueError(f"Cannot normalize {src}: {err}") from None
    return n


def _wrap_array(value: Any, array_key: str) -> Any:
    return {array_key: value} if isinstance(value, list) else value


def _rewrite_jsonl_of_arrays(src: Path, dst: Path, array_key: str) -> int:
    n = 0
    with dst.open("w", encoding="utf-8") as fout:
        for line in iter_jsonl(src, on_error="raise"):
            fout.write(_dumps(_wrap_array(line.value, array_key)) + "\n")
            n += 1
    return n


def _dumps(value: Any) -> str:
    """One JSONL line; unpaired surrogates are kept as ``\\uXXXX`` escapes."""
    text = json.dumps(value, ensure_ascii=False)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return json.dumps(value)
    return text


def _rewrite_json_array(src: Path, dst: Path, array_key: str = DEFAULT_ARRAY_KEY) -> int:
    with src.open(encoding="utf-8-sig") as fin:
        data = json.load(fin)
    if not isinstance(data, list):
        raise ValueError(f"{src} is not a JSON array")
    n = 0
    with dst.open("w", encoding="utf-8") as fout:
        for obj in data:
            fout.write(_dumps(_wrap_array(obj, array_key)) + "\n")
            n += 1
    return n


def _rewrite_single_line(src: Path, dst: Path) -> int:
    text = src.read_text(encoding="utf-8-sig").strip()
    records = _split_concatenated_objects(text)
    n = 0
    with dst.open("w", encoding="utf-8") as fout:
        for obj in records:
            fout.write(_dumps(obj) + "\n")
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
