"""Parse and validate the YAML manifest that drives ``convmerge fetch``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from convmerge.fetch.auth import AuthConfig, TokenSpec

EntryKind = Literal["hf", "url_raw", "url_github_tree", "url_github_clone"]


@dataclass(frozen=True)
class DatasetEntry:
    """One ``datasets[*]`` entry from the manifest."""

    name: str
    # Either ``hf`` or ``url`` is set. Others are optional modifiers.
    hf: str | None = None
    url: str | None = None

    # HuggingFace only
    config: str | None = None
    split: str | None = None

    # GitHub repo only
    ext: tuple[str, ...] = ()
    mode: Literal["tree", "clone"] | None = None
    lfs: bool = False

    # Optional output override (directory relative to manifest or absolute).
    output: str | None = None


@dataclass(frozen=True)
class Defaults:
    """Manifest-wide defaults applied when an entry does not override them."""

    output_root: str = "./raw"
    on_error: Literal["continue", "fail"] = "continue"
    resume: bool = True


@dataclass(frozen=True)
class Manifest:
    """Parsed top-level manifest."""

    version: int = 1
    auth: AuthConfig = field(default_factory=AuthConfig)
    defaults: Defaults = field(default_factory=Defaults)
    datasets: tuple[DatasetEntry, ...] = ()


_SANITIZE_BAD_CHARS = re.compile(r'[<>:"/\\|?*]')
_SANITIZE_WHITESPACE = re.compile(r"\s+")


def sanitize_name(name: str) -> str:
    """Turn an arbitrary dataset name into a safe file/directory name."""
    s = _SANITIZE_BAD_CHARS.sub("_", name)
    s = _SANITIZE_WHITESPACE.sub("_", s).strip("_")
    return s or "unnamed"


def load_manifest(path: str | Path) -> Manifest:
    """Parse a YAML manifest file into a :class:`Manifest`.

    Raises ``ImportError`` with an install hint if PyYAML is not available.
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "pyyaml is required for fetch manifests. Install with: pip install 'convmerge[fetch]'"
        ) from e

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Manifest root must be a mapping, got {type(raw).__name__}")
    return _from_dict(raw)


def _from_dict(raw: dict[str, Any]) -> Manifest:
    version = int(raw.get("version", 1))
    if version != 1:
        raise ValueError(
            f"Unsupported manifest version: {version}. This build understands version 1."
        )

    auth_raw = raw.get("auth") or {}
    auth = AuthConfig(
        hf=TokenSpec(env=auth_raw.get("hf_token_env"), file=auth_raw.get("hf_token_file")),
        github=TokenSpec(
            env=auth_raw.get("github_token_env"),
            file=auth_raw.get("github_token_file"),
        ),
    )

    d_raw = raw.get("defaults") or {}
    on_error = d_raw.get("on_error", "continue")
    if on_error not in ("continue", "fail"):
        raise ValueError(f"defaults.on_error must be 'continue' or 'fail', got {on_error!r}")
    defaults = Defaults(
        output_root=str(d_raw.get("output_root", "./raw")),
        on_error=on_error,
        resume=bool(d_raw.get("resume", True)),
    )

    entries_raw = raw.get("datasets") or []
    if not isinstance(entries_raw, list):
        raise ValueError("'datasets' must be a list")
    entries: list[DatasetEntry] = []
    for i, item in enumerate(entries_raw):
        if not isinstance(item, dict):
            raise ValueError(f"datasets[{i}] must be a mapping")
        entries.append(_entry_from_dict(item, index=i))

    return Manifest(version=version, auth=auth, defaults=defaults, datasets=tuple(entries))


def _entry_from_dict(item: dict[str, Any], *, index: int) -> DatasetEntry:
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"datasets[{index}].name is required and must be a non-empty string")

    hf = item.get("hf")
    url = item.get("url")
    if bool(hf) == bool(url):
        raise ValueError(f"datasets[{index}] ({name!r}) must set exactly one of 'hf' or 'url'")

    ext_raw = item.get("ext") or ()
    if isinstance(ext_raw, str):
        ext_raw = (ext_raw,)
    ext = tuple(str(e).lower() for e in ext_raw)

    mode = item.get("mode")
    if mode not in (None, "tree", "clone"):
        raise ValueError(
            f"datasets[{index}] ({name!r}) mode must be 'tree' or 'clone', got {mode!r}"
        )

    return DatasetEntry(
        name=name.strip(),
        hf=str(hf).strip() if hf else None,
        url=str(url).strip() if url else None,
        config=item.get("config"),
        split=item.get("split"),
        ext=ext,
        mode=mode,
        lfs=bool(item.get("lfs", False)),
        output=item.get("output"),
    )


_RAW_GITHUB_HOSTS = ("raw.githubusercontent.com",)
_RAW_SUFFIXES = (".json", ".jsonl", ".json.gz")


def classify_entry(entry: DatasetEntry) -> EntryKind:
    """Decide which fetch backend handles this entry."""
    if entry.hf:
        return "hf"
    url = (entry.url or "").lower()
    if not url:
        raise ValueError(f"Entry {entry.name!r} has neither 'hf' nor 'url'")
    if any(host in url for host in _RAW_GITHUB_HOSTS):
        return "url_raw"
    if url.endswith(_RAW_SUFFIXES):
        return "url_raw"
    if "github.com" in url:
        return "url_github_clone" if entry.mode == "clone" else "url_github_tree"
    raise ValueError(
        f"Entry {entry.name!r} has URL {entry.url!r}; only GitHub and "
        "raw GitHub URLs are supported. For HuggingFace use the 'hf' field."
    )
