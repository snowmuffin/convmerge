"""Recipe file format (version 1): parsing and validation.

A recipe declares, per source, how to get raw data (``fetch`` or a local
``path``), whether to ``normalize`` it, and how to ``convert`` it; then how to
``mix`` the sources and whether to ``dedupe`` the result. Paths are relative
to the recipe file. Every error names the offending key, e.g.
``sources.tools.convert.from: unknown adapter 'x'``.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from convmerge.convert import OnInvalid
from convmerge.fetch.auth import AuthConfig, TokenSpec
from convmerge.fetch.manifest import DatasetEntry

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CONVERT_KEYS = {
    "from",
    "format",
    "preset",
    "adapter_kwargs",
    "preference",
    "on_invalid",
    "workers",
    "tool_arguments",
    "keep_meta",
    "meta_key",
    "alpaca_multiturn",
    "reasoning",
    "tool_content",
    "system",
    "merge_consecutive",
    "split_turns",
    "reasoning_turns",
}
_FETCH_ENTRY_KEYS = {"hf", "url", "config", "split", "ext", "mode", "lfs", "max_rows"}


class RecipeError(ValueError):
    """A recipe file is invalid; the message starts with the offending key."""


@dataclass(frozen=True)
class ConvertSpec:
    adapter: str | None
    output_format: str | None
    preset: Path | None = None
    adapter_kwargs: dict[str, Any] | None = None
    preference: str | None = None
    on_invalid: OnInvalid = "drop"
    workers: int = 1
    emit: dict[str, Any] = field(default_factory=dict)
    transforms: dict[str, Any] = field(default_factory=dict)

    def options(self, base: Path) -> dict[str, Any]:
        """JSON-able options for step keys (preset as a path relative to ``base``)."""
        options = {
            "from": self.adapter,
            "format": self.output_format,
            "preset": _rel(self.preset, base) if self.preset else None,
            "adapter_kwargs": self.adapter_kwargs,
            "preference": self.preference,
            "on_invalid": self.on_invalid,
            "emit": {k: list(v) if isinstance(v, tuple) else v for k, v in self.emit.items()},
        }
        if self.transforms:  # only when set, so earlier lock files stay valid
            options["transforms"] = dict(self.transforms)
        return options


@dataclass(frozen=True)
class SourceSpec:
    name: str
    fetch: DatasetEntry | None
    path: Path | None
    normalize: bool
    array_key: str
    convert: ConvertSpec
    fetch_auth: AuthConfig | None = None


@dataclass(frozen=True)
class MixSpec:
    weights: dict[str, float]
    total: int | None = None
    seed: int = 42
    oversample: bool = False
    sampler: str = "v2"


@dataclass(frozen=True)
class DedupeSpec:
    keys: tuple[str, ...] | None = None
    algorithm: str = "md5"


@dataclass(frozen=True)
class TokensSpec:
    tokenizer: str
    """A Hub model name, or a local directory (then a path relative to the recipe)."""
    max_tokens: int | None = None
    revision: str | None = None
    chat_template: Path | None = None
    local: Path | None = None
    """The tokenizer directory when ``tokenizer`` names one on disk."""


@dataclass(frozen=True)
class SplitSpec:
    val_output: Path
    val: float | None = None
    val_rows: int | None = None
    seed: int = 42
    keys: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Recipe:
    path: Path
    base_dir: Path
    workdir: Path
    output: Path
    lock_path: Path
    report_path: Path
    sources: dict[str, SourceSpec]
    mix: MixSpec | None
    dedupe: DedupeSpec | None
    auth: AuthConfig
    split: SplitSpec | None = None
    tokens: TokensSpec | None = None


def load_recipe(path: str | Path) -> Recipe:
    """Parse and validate a recipe file (YAML needs PyYAML; JSON does not)."""
    p = Path(path).resolve()
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "YAML recipes need PyYAML: pip install 'convmerge[preset]' (or [all]), "
                "or write the recipe as JSON"
            ) from e
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise RecipeError(f"invalid YAML: {e}") from e
    return parse_recipe(raw, path=p)


def parse_recipe(raw: Any, *, path: Path) -> Recipe:
    base = path.parent
    top = _mapping(raw, "recipe")
    _only(
        top,
        {
            "version",
            "workdir",
            "output",
            "lock",
            "report",
            "auth",
            "sources",
            "mix",
            "dedupe",
            "tokens",
            "split",
        },  # fmt: skip
        "",
    )
    if top.get("version", 1) != 1:
        raise RecipeError(f"version: unsupported recipe version {top.get('version')!r} (use 1)")
    if "output" not in top:
        raise RecipeError("output: required (path of the final JSONL file)")

    workdir = base / _str(top.get("workdir", "build"), "workdir")
    auth = _auth(top.get("auth"), "auth")
    sources_raw = _mapping(top.get("sources"), "sources")
    if not sources_raw:
        raise RecipeError("sources: at least one source is required")
    sources: dict[str, SourceSpec] = {}
    for name, spec in sources_raw.items():
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise RecipeError(
                f"sources.{name}: source names use letters, digits, '_', '-', '.' "
                "(they become directory names)"
            )
        sources[name] = _source(name, spec, base)

    output = base / _str(top["output"], "output")
    return Recipe(
        path=path,
        base_dir=base,
        workdir=workdir,
        output=output,
        lock_path=base / _str(top.get("lock", f"{path.stem}.lock.json"), "lock"),
        report_path=base / _str(top["report"], "report")
        if "report" in top
        else workdir / "report.json",
        sources=sources,
        mix=_mix(top.get("mix"), sources),
        dedupe=_dedupe(top.get("dedupe")),
        auth=auth,
        split=_split(top.get("split"), base, output),
        tokens=_tokens(top.get("tokens"), base),
    )


def _source(name: str, raw: Any, base: Path) -> SourceSpec:
    where = f"sources.{name}"
    spec = _mapping(raw, where)
    _only(spec, {"fetch", "path", "normalize", "convert"}, where)
    if ("fetch" in spec) == ("path" in spec):
        raise RecipeError(f"{where}: set exactly one of 'fetch' or 'path'")

    fetch: DatasetEntry | None = None
    fetch_auth: AuthConfig | None = None
    local: Path | None = None
    if "fetch" in spec:
        fetch, fetch_auth = _fetch(name, spec["fetch"], base, f"{where}.fetch")
    else:
        local = base / _str(spec["path"], f"{where}.path")

    norm = spec.get("normalize", True)
    array_key = "conversation"
    if isinstance(norm, dict):
        _only(norm, {"array_key"}, f"{where}.normalize")
        array_key = _str(norm.get("array_key", array_key), f"{where}.normalize.array_key")
        norm = True
    elif not isinstance(norm, bool):
        raise RecipeError(f"{where}.normalize: expected true, false, or a mapping")

    if "convert" not in spec:
        raise RecipeError(f"{where}.convert: required (at least 'from', or a 'preset')")
    return SourceSpec(
        name=name,
        fetch=fetch,
        path=local,
        normalize=norm,
        array_key=array_key,
        convert=_convert(spec["convert"], base, f"{where}.convert"),
        fetch_auth=fetch_auth,
    )


def _fetch(name: str, raw: Any, base: Path, where: str) -> tuple[DatasetEntry, AuthConfig | None]:
    from convmerge.fetch.manifest import load_manifest

    spec = _mapping(raw, where)
    if "manifest" in spec:
        _only(spec, {"manifest", "name", "max_rows"}, where)
        manifest_path = base / _str(spec["manifest"], f"{where}.manifest")
        entry_name = _str(spec.get("name", name), f"{where}.name")
        try:
            manifest = load_manifest(manifest_path)
        except (OSError, ValueError) as e:
            raise RecipeError(f"{where}.manifest: {e}") from e
        found = [d for d in manifest.datasets if d.name == entry_name]
        if not found:
            names = ", ".join(d.name for d in manifest.datasets)
            raise RecipeError(f"{where}.name: no entry {entry_name!r} in {manifest_path} ({names})")
        entry = replace(found[0], name=name, output=None)
        if "max_rows" in spec:
            # Re-validate the entry with the override applied.
            fields = {k: v for k, v in dataclasses.asdict(entry).items() if v not in (None, ())}
            fields["ext"] = list(entry.ext)
            entry = _entry(name, {**fields, "max_rows": spec["max_rows"]}, where)
        return entry, manifest.auth
    _only(spec, _FETCH_ENTRY_KEYS, where)
    return _entry(name, spec, where), None


def _entry(name: str, spec: dict[str, Any], where: str) -> DatasetEntry:
    from convmerge.fetch.manifest import _entry_from_dict

    try:
        return _entry_from_dict({**spec, "name": name}, index=0)
    except ValueError as e:
        msg = re.sub(r"^datasets\[0\] \([^)]*\) ", "", str(e))
        raise RecipeError(f"{where}: {msg}") from e


def _convert(raw: Any, base: Path, where: str) -> ConvertSpec:
    from convmerge.adapters.preference import PREFERENCES
    from convmerge.config import build_convert_config

    spec = _mapping(raw, where)
    _only(spec, _CONVERT_KEYS, where)
    preset = base / _str(spec["preset"], f"{where}.preset") if "preset" in spec else None
    kwargs = spec.get("adapter_kwargs")
    if kwargs is not None and not isinstance(kwargs, dict):
        raise RecipeError(f"{where}.adapter_kwargs: expected a mapping")
    preference = spec.get("preference")
    if preference is not None and preference not in PREFERENCES:
        raise RecipeError(f"{where}.preference: expected one of {list(PREFERENCES)}")
    on_invalid = spec.get("on_invalid", "drop")
    if on_invalid not in ("drop", "keep", "fail"):
        raise RecipeError(f"{where}.on_invalid: expected drop, keep, or fail")
    workers = spec.get("workers", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise RecipeError(f"{where}.workers: expected a positive integer")
    emit: dict[str, Any] = {}
    for key in ("tool_arguments", "meta_key", "alpaca_multiturn", "reasoning", "tool_content"):
        if key in spec:
            emit[key] = _str(spec[key], f"{where}.{key}")
    transforms: dict[str, Any] = {}
    for key in ("system", "reasoning_turns"):
        if key in spec:
            transforms[key] = _str(spec[key], f"{where}.{key}")
    for key in ("merge_consecutive", "split_turns"):
        if key in spec:
            if not isinstance(spec[key], bool):
                raise RecipeError(f"{where}.{key}: expected true or false")
            transforms[key] = spec[key]
    if "keep_meta" in spec:
        km = spec["keep_meta"]
        if isinstance(km, bool):
            emit["keep_meta"] = km
        elif isinstance(km, list) and all(isinstance(k, str) for k in km):
            emit["keep_meta"] = tuple(km)
        else:
            raise RecipeError(f"{where}.keep_meta: expected true/false or a list of keys")

    convert = ConvertSpec(
        adapter=spec.get("from"),
        output_format=spec.get("format", None if preset else "messages"),
        preset=preset,
        adapter_kwargs=kwargs,
        preference=preference,
        on_invalid=on_invalid,
        workers=workers,
        emit=emit,
        transforms=transforms,
    )
    # Resolve once now so bad names, presets, or options fail before any work.
    try:
        cfg = build_convert_config(**convert_config_kwargs(convert))
    except (ValueError, OSError, ImportError) as e:
        raise RecipeError(f"{where}: {e}") from e
    _check_names(cfg.adapter, cfg.output_format, where)
    return convert


def convert_config_kwargs(spec: ConvertSpec) -> dict[str, Any]:
    return {
        "preset_path": spec.preset,
        "adapter": spec.adapter,
        "output_format": spec.output_format,
        "adapter_kwargs_json": json.dumps(spec.adapter_kwargs) if spec.adapter_kwargs else None,
        "emit_overrides": spec.emit or None,
        "preference": spec.preference,
        "transform_overrides": spec.transforms or None,
    }


def _check_names(adapter: str, output_format: str, where: str) -> None:
    from convmerge.adapters import available_adapters
    from convmerge.emitters import available_formats

    if adapter not in available_adapters():
        raise RecipeError(
            f"{where}.from: unknown adapter {adapter!r} ({', '.join(available_adapters())})"
        )
    if output_format not in available_formats():
        raise RecipeError(
            f"{where}.format: unknown format {output_format!r} ({', '.join(available_formats())})"
        )


def _mix(raw: Any, sources: dict[str, SourceSpec]) -> MixSpec | None:
    if raw is None:
        return None
    spec = _mapping(raw, "mix")
    _only(spec, {"weights", "total", "seed", "oversample", "sampler"}, "mix")
    weights_raw = spec.get("weights")
    if weights_raw is None:
        weights = {name: 1.0 for name in sources}
    else:
        wmap = _mapping(weights_raw, "mix.weights")
        weights = {}
        for name, w in wmap.items():
            if name not in sources:
                raise RecipeError(f"mix.weights.{name}: no such source")
            if isinstance(w, bool) or not isinstance(w, (int, float)) or w <= 0:
                raise RecipeError(f"mix.weights.{name}: expected a positive number")
            weights[name] = float(w)
        missing = [n for n in sources if n not in weights]
        if missing:
            raise RecipeError(f"mix.weights: missing weight for {', '.join(missing)}")
    total = spec.get("total")
    if total is not None and (isinstance(total, bool) or not isinstance(total, int) or total < 1):
        raise RecipeError("mix.total: expected a positive integer")
    seed = spec.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RecipeError("mix.seed: expected an integer")
    sampler = spec.get("sampler", "v2")
    if sampler not in ("v1", "v2"):
        raise RecipeError("mix.sampler: expected v1 or v2")
    return MixSpec(
        weights=weights,
        total=total,
        seed=seed,
        oversample=bool(spec.get("oversample", False)),
        sampler=sampler,
    )


def _dedupe(raw: Any) -> DedupeSpec | None:
    if raw is None or raw is False:
        return None
    if raw is True:
        return DedupeSpec()
    spec = _mapping(raw, "dedupe")
    _only(spec, {"keys", "algorithm"}, "dedupe")
    keys = spec.get("keys")
    if keys is not None and not (isinstance(keys, list) and all(isinstance(k, str) for k in keys)):
        raise RecipeError("dedupe.keys: expected a list of top-level keys")
    algorithm = spec.get("algorithm", "md5")
    if algorithm not in ("md5", "sha256"):
        raise RecipeError("dedupe.algorithm: expected md5 or sha256")
    return DedupeSpec(keys=tuple(keys) if keys else None, algorithm=algorithm)


def _tokens(raw: Any, base: Path) -> TokensSpec | None:
    if raw is None or raw is False:
        return None
    spec = _mapping(raw, "tokens")
    _only(spec, {"tokenizer", "max_tokens", "revision", "chat_template"}, "tokens")
    if "tokenizer" not in spec:
        raise RecipeError("tokens.tokenizer: required (a Hub model name or a local directory)")
    tokenizer = _str(spec["tokenizer"], "tokens.tokenizer")
    max_tokens = spec.get("max_tokens")
    if max_tokens is not None and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1
    ):
        raise RecipeError("tokens.max_tokens: expected a positive integer")
    revision = spec.get("revision")
    if revision is not None:
        revision = _str(revision, "tokens.revision")
    template = spec.get("chat_template")
    local = base / tokenizer
    return TokensSpec(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        revision=revision,
        chat_template=base / _str(template, "tokens.chat_template") if template else None,
        local=local if local.is_dir() else None,
    )


def _split(raw: Any, base: Path, output: Path) -> SplitSpec | None:
    from convmerge.split import default_val_path

    if raw is None or raw is False:
        return None
    spec = _mapping(raw, "split")
    _only(spec, {"val", "val_rows", "seed", "keys", "val_output"}, "split")
    val, val_rows = spec.get("val"), spec.get("val_rows")
    if (val is None) == (val_rows is None):
        raise RecipeError("split: give exactly one of 'val' (a fraction) or 'val_rows' (a count)")
    if val is not None and (
        isinstance(val, bool) or not isinstance(val, (int, float)) or not 0 < val < 1
    ):
        raise RecipeError("split.val: expected a fraction between 0 and 1")
    if val_rows is not None and (
        isinstance(val_rows, bool) or not isinstance(val_rows, int) or val_rows < 0
    ):
        raise RecipeError("split.val_rows: expected a non-negative integer")
    seed = spec.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RecipeError("split.seed: expected an integer")
    keys = spec.get("keys")
    if keys is not None and not (isinstance(keys, list) and all(isinstance(k, str) for k in keys)):
        raise RecipeError("split.keys: expected a list of top-level keys")
    val_output = (
        base / _str(spec["val_output"], "split.val_output")
        if "val_output" in spec
        else default_val_path(output)
    )
    if val_output == output:
        raise RecipeError("split.val_output: must differ from output")
    return SplitSpec(
        val_output=val_output,
        val=float(val) if val is not None else None,
        val_rows=val_rows,
        seed=seed,
        keys=tuple(keys) if keys else None,
    )


def _auth(raw: Any, where: str) -> AuthConfig:
    if raw is None:
        return AuthConfig()
    spec = _mapping(raw, where)
    _only(spec, {"hf_token_env", "hf_token_file", "github_token_env", "github_token_file"}, where)
    return AuthConfig(
        hf=TokenSpec(env=spec.get("hf_token_env"), file=spec.get("hf_token_file")),
        github=TokenSpec(env=spec.get("github_token_env"), file=spec.get("github_token_file")),
    )


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        kind = type(value).__name__ if value is not None else "nothing"
        raise RecipeError(f"{where or 'recipe'}: expected a mapping, got {kind}")
    return value


def _only(spec: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(spec) - allowed)
    if unknown:
        prefix = f"{where}." if where else ""
        raise RecipeError(
            f"{prefix}{unknown[0]}: unknown key (allowed: {', '.join(sorted(allowed))})"
        )


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecipeError(f"{where}: expected a non-empty string")
    return value


def _rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
