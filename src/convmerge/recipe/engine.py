"""Plan and run a recipe: ordered steps, a lock file, incremental re-runs.

Each step calls an existing convmerge function with the recipe's options and
writes its output atomically (a hidden ``.part`` path, then a rename). A step
is skipped when the lock file shows it already ran with the same options,
convmerge version, and input digests, and its outputs are unchanged on disk.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from convmerge import __version__
from convmerge.recipe.schema import Recipe, SourceSpec, convert_config_kwargs

LOCK_VERSION = 1
REPORT_VERSION = 1

LogFn = Callable[[str], None]


class RecipeRunError(RuntimeError):
    """A step failed; the message names the step."""


@dataclass
class Step:
    name: str
    kind: str
    inputs: list[Path]
    output: Path
    options: dict[str, Any]
    run: Callable[[Path], dict[str, Any]]
    """Called with the staging path to write to; returns stats for the report."""
    source: str | None = None


@dataclass
class PlannedStep:
    step: Step
    action: str  # "run" | "skip"
    reason: str


@dataclass
class RunResult:
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


# --- steps ------------------------------------------------------------------


def build_steps(
    recipe: Recipe, *, hf_token: str | None = None, github_token: str | None = None
) -> list[Step]:
    """The recipe as an ordered list of steps (sources first, then mix/dedupe/output)."""
    steps: list[Step] = []
    converted: dict[str, Path] = {}
    for src in recipe.sources.values():
        base = recipe.workdir / src.name
        if src.fetch is not None:
            raw = base / "raw"
            steps.append(_fetch_step(recipe, src, raw, hf_token, github_token))
        else:
            assert src.path is not None
            raw = src.path
        data = raw
        if src.normalize:
            data = base / "jsonl"
            steps.append(_normalize_step(src, raw, data))
        converted[src.name] = base / "converted.jsonl"
        steps.append(_convert_step(recipe, src, data, converted[src.name]))

    # Whole-dataset stages, in order; the last one writes the recipe output
    # (split writes it together with the validation file).
    names = list(recipe.sources)
    stages: list[tuple[str, Callable[[Path, Path], Step]]] = []
    if recipe.mix is not None or len(names) > 1:
        stages.append(("mixed", lambda src, out: _mix_step(recipe, converted, out)))
    if recipe.dedupe is not None:
        stages.append(("deduped", lambda src, out: _dedupe_step(recipe, src, out)))
    if recipe.filter is not None:
        stages.append(("quality", lambda src, out: _filter_step(recipe, src, out)))
    if recipe.decontam is not None:
        stages.append(("decontaminated", lambda src, out: _decontam_step(recipe, src, out)))
    if recipe.tokens is not None:
        stages.append(("filtered", lambda src, out: _tokens_step(recipe, src, out)))
    last = converted[names[0]]
    for i, (label, make) in enumerate(stages):
        final = i == len(stages) - 1 and recipe.split is None
        out = recipe.output if final else recipe.workdir / f"{label}.jsonl"
        steps.append(make(last, out))
        last = out
    if recipe.split is not None:
        steps.extend(_split_steps(recipe, last))
    elif last != recipe.output:
        steps.append(_copy_step(last, recipe.output))
    return steps


def _fetch_step(
    recipe: Recipe, src: SourceSpec, raw: Path, hf_token: str | None, github_token: str | None
) -> Step:
    from convmerge.fetch.manifest import Defaults, Manifest
    from convmerge.fetch.runner import run_manifest

    entry = src.fetch
    assert entry is not None
    auth = src.fetch_auth or recipe.auth

    def run(stage: Path) -> dict[str, Any]:
        stage.mkdir(parents=True)
        manifest = Manifest(
            auth=auth,
            defaults=Defaults(output_root=str(stage), on_error="fail", resume=False),
            datasets=(dataclasses.replace(entry, output=str(stage / "data")),),
        )
        run_manifest(manifest, hf_token=hf_token, github_token=github_token, log=lambda _m: None)
        stats: dict[str, Any] = {
            "files": sorted(p.relative_to(stage).as_posix() for p in _files(stage))
        }
        if entry.hf:
            from convmerge.licenses import detect_hf_license

            stats["license"] = detect_hf_license(entry.hf, token=hf_token)
        return stats

    options = {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in dataclasses.asdict(entry).items()
        # An unset revision stays out so steps recorded before it existed stay fresh.
        if k not in ("name", "output") and not (k == "revision" and v is None)
    }
    return Step(f"{src.name}.fetch", "fetch", [], raw, options, run, src.name)


def _normalize_step(src: SourceSpec, raw: Path, out: Path) -> Step:
    from convmerge.normalize.files import normalize_path

    def run(stage: Path) -> dict[str, Any]:
        if raw.is_file():
            stage.mkdir(parents=True)
            dst = stage / raw.with_suffix(".jsonl").name
            result = normalize_path(raw, dst, array_key=key, sheet=src.sheet)
        else:
            result = normalize_path(raw, stage, array_key=key, sheet=src.sheet)
            stage.mkdir(parents=True, exist_ok=True)
        if result.failed:
            detail = "; ".join(f"{p.name}: {e}" for p, e in result.failed[:3])
            raise RecipeRunError(f"{len(result.failed)} file(s) failed to normalize: {detail}")
        return {"files": len(result.files), "records": result.records}

    key = src.array_key
    options: dict[str, Any] = {"array_key": key}
    if src.sheet is not None:  # only then, so recipes written before 1.4 keep their fingerprints
        options["sheet"] = src.sheet
    return Step(f"{src.name}.normalize", "normalize", [raw], out, options, run, src.name)


def _convert_step(recipe: Recipe, src: SourceSpec, data: Path, out: Path) -> Step:
    from convmerge.config import build_convert_config
    from convmerge.convert import ConvertStats, convert_with_config
    from convmerge.normalize.files import iter_data_files

    spec = src.convert

    def run(stage: Path) -> dict[str, Any]:
        cfg = build_convert_config(**convert_config_kwargs(spec))
        files = [data] if data.is_file() else [p for p in iter_data_files(data)]
        if not files:
            raise RecipeRunError(f"no data files in {data}")
        total = ConvertStats()
        per_file = []
        with stage.open("wb") as out_f:
            for f in files:
                part = stage.with_name(stage.name + ".one")
                st = ConvertStats()
                convert_with_config(
                    f, part, cfg, stats=st, on_invalid=spec.on_invalid, workers=spec.workers
                )
                with part.open("rb") as pf:
                    shutil.copyfileobj(pf, out_f)
                part.unlink()
                total.merge(st)
                per_file.append(
                    {
                        "file": _display(f, recipe.base_dir),
                        "written": st.written,
                        "dropped": st.dropped,
                        "drop_reasons": st.drop_reasons,
                    }
                )
        report = total.to_report()
        if len(files) > 1:
            report["files"] = per_file
        return report

    inputs = [data] + ([spec.preset] if spec.preset else [])
    options = {**spec.options(recipe.base_dir), "workers": None}  # workers never change output
    return Step(f"{src.name}.convert", "convert", inputs, out, options, run, src.name)


def _mix_step(recipe: Recipe, converted: dict[str, Path], out: Path) -> Step:
    from convmerge.mix import MixSource, mix_files

    mix = recipe.mix
    weights = mix.weights if mix else {name: 1.0 for name in converted}
    total = mix.total if mix else None
    seed = mix.seed if mix else 42
    oversample = mix.oversample if mix else False
    sampler = mix.sampler if mix else "v2"
    by = mix.by if mix else "rows"
    tokenizer = mix.tokenizer if mix else None
    by_sample = mix.by_sample if mix else None
    max_tokens = mix.max_tokens if mix else None

    def run(stage: Path) -> dict[str, Any]:
        result = mix_files(
            [MixSource(converted[n], w) for n, w in weights.items()],
            stage,
            total=total,
            seed=seed,
            oversample=oversample,
            sampler=sampler,  # type: ignore[arg-type]
            by=by,  # type: ignore[arg-type]
            tokenizer=tokenizer,
            by_sample=by_sample,
            max_tokens=max_tokens,
        )
        return {
            "sampler": result.sampler,
            "seed": result.seed,
            "total_written": result.total_written,
            "sources": {
                n: {
                    "weight": s.weight,
                    "requested": s.requested,
                    "available": s.available,
                    "written": s.written,
                }  # fmt: skip
                for n, s in zip(weights, result.sources)
            },
        }

    options = {"weights": weights, "total": total, "seed": seed, "oversample": oversample,
               "sampler": sampler}  # fmt: skip
    if by != "rows":  # only then, so recipes written before 1.3 keep their fingerprints
        options.update(by=by, tokenizer=tokenizer)
    if by_sample is not None:  # likewise for recipes written before 1.4
        options["by_sample"] = by_sample
    if max_tokens is not None:  # likewise for recipes written before 1.6
        options["max_tokens"] = max_tokens
    return Step("mix", "mix", [converted[n] for n in weights], out, options, run)


def _dedupe_step(recipe: Recipe, src: Path, out: Path) -> Step:
    from convmerge.normalize.dedup import DedupeStats, deduplicate_jsonl

    spec = recipe.dedupe
    assert spec is not None

    def run(stage: Path) -> dict[str, Any]:
        if spec.near:
            from convmerge.normalize.near_dedup import NearDedupeStats, deduplicate_near_jsonl

            near = NearDedupeStats()
            deduplicate_near_jsonl(src, stage, threshold=spec.threshold, num_perm=spec.num_perm,
                                   keys=spec.keys, stats=near, workers=spec.workers)  # fmt: skip
            return dataclasses.asdict(near)
        st = DedupeStats()
        deduplicate_jsonl(src, stage, keys=spec.keys, algorithm=spec.algorithm, stats=st)
        return dataclasses.asdict(st)

    options: dict[str, Any] = {
        "keys": list(spec.keys) if spec.keys else None,
        "algorithm": spec.algorithm,
    }
    if spec.near:  # workers never change the output, so they are not an option here
        options.update(near=True, threshold=spec.threshold, num_perm=spec.num_perm)
    return Step("dedupe", "dedupe", [src], out, options, run)


def _filter_step(recipe: Recipe, src: Path, out: Path) -> Step:
    from convmerge.quality import FilterSpec, FilterStats, filter_jsonl

    spec = recipe.filter
    assert spec is not None
    inputs = [src] if spec.rules_file is None else [src, spec.rules_file]

    def run(stage: Path) -> dict[str, Any]:
        options = FilterSpec.from_options(**spec.options, rules_file=spec.rules_file)
        st = FilterStats()
        filter_jsonl(src, spec=options, output=stage, stats=st, workers=spec.workers)
        return st.to_report()

    options: dict[str, Any] = dict(spec.options)  # workers never change output
    if spec.rules_file is not None:
        options["rules_file"] = _display(spec.rules_file, recipe.base_dir)
    return Step("filter", "filter", inputs, out, options, run)


def _decontam_step(recipe: Recipe, src: Path, out: Path) -> Step:
    from convmerge.decontam import DecontamStats, EvalSource, build_index, decontaminate_jsonl
    from convmerge.fetch.auth import resolve_token

    spec = recipe.decontam
    assert spec is not None
    local = [Path(a) for a in spec.against if not a.startswith("hf:")]

    def run(stage: Path) -> dict[str, Any]:
        sources = [EvalSource(a, spec.fields) for a in spec.against]
        token = resolve_token(recipe.auth.hf)
        index = build_index(sources, ngram=spec.ngram, min_tokens=spec.min_tokens, token=token,
                            cache_dir=stage.parent)  # fmt: skip
        st = DecontamStats()
        decontaminate_jsonl(src, index, check=spec.check, output=stage,  # type: ignore[arg-type]
                            stats=st)  # fmt: skip
        report = st.to_report()
        report["eval_sets"] = {_eval_name(k, recipe.base_dir): v
                               for k, v in report["eval_sets"].items()}  # fmt: skip
        report["samples"] = [{**s, "eval": _eval_name(s["eval"], recipe.base_dir)}
                             for s in report["samples"]]  # fmt: skip
        return report

    options = {
        "against": [_eval_name(a, recipe.base_dir) for a in spec.against],
        "ngram": spec.ngram,
        "min_tokens": spec.min_tokens,
        "check": spec.check,
        "fields": list(spec.fields) if spec.fields else None,
    }
    return Step("decontam", "decontam", [src, *local], out, options, run)


def _eval_name(spec: str, base: Path) -> str:
    return spec if spec.startswith("hf:") else _display(Path(spec), base)


def _tokens_step(recipe: Recipe, src: Path, out: Path) -> Step:
    from convmerge.fetch.auth import resolve_token
    from convmerge.tokens import TokenStats, check_tokens, load_tokenizer

    spec = recipe.tokens
    assert spec is not None
    inputs = [src]
    if spec.local is not None:
        inputs.append(spec.local)
    if spec.chat_template is not None:
        inputs.append(spec.chat_template)

    def run(stage: Path) -> dict[str, Any]:
        name = str(spec.local) if spec.local is not None else spec.tokenizer
        token = resolve_token(recipe.auth.hf)
        tok = load_tokenizer(name, revision=spec.revision, token=token)
        template = spec.chat_template.read_text(encoding="utf-8") if spec.chat_template else None
        st = TokenStats()
        check_tokens(src, tokenizer=tok, max_tokens=spec.max_tokens, output=stage,
                     chat_template=template, stats=st)  # fmt: skip
        report = st.to_report()
        report["tokenizer"] = spec.tokenizer
        return report

    options = {
        "tokenizer": spec.tokenizer,
        "revision": spec.revision,
        "max_tokens": spec.max_tokens,
        "chat_template": _display(spec.chat_template, recipe.base_dir)
        if spec.chat_template
        else None,
    }
    return Step("tokens", "tokens", inputs, out, options, run)


def _split_steps(recipe: Recipe, src: Path) -> list[Step]:
    """Two steps over the same input: the train part (the recipe output) and the
    validation part. Assignment is a pure function of each row and the seed, so
    the two always agree, and each re-runs only when its own file is stale."""
    from convmerge.split import SplitStats, split_jsonl

    spec = recipe.split
    assert spec is not None
    options = {
        "val": spec.val,
        "val_rows": spec.val_rows,
        "seed": spec.seed,
        "keys": list(spec.keys) if spec.keys else None,
    }

    def part(which: str, out: Path) -> Step:
        def run(stage: Path) -> dict[str, Any]:
            st = SplitStats()
            with tempfile.TemporaryDirectory(dir=stage.parent) as tmp:
                other = Path(tmp) / "other.jsonl"
                train, val = (stage, other) if which == "train" else (other, stage)
                split_jsonl(
                    src, train, val, val=spec.val, val_rows=spec.val_rows, seed=spec.seed,
                    keys=spec.keys, stats=st,
                )  # fmt: skip
            return dataclasses.asdict(st)

        return Step(f"split.{which}", "split", [src], out, {**options, "part": which}, run)

    return [part("train", recipe.output), part("val", spec.val_output)]


def _copy_step(src: Path, out: Path) -> Step:
    def run(stage: Path) -> dict[str, Any]:
        shutil.copyfile(src, stage)
        return {}

    return Step("output", "output", [src], out, {}, run)


# --- digests and the lock file ----------------------------------------------


class _Digests:
    """SHA-256 of files and directories, reusing cached values when size and mtime match."""

    def __init__(self, cache: dict[str, list[Any]], base: Path):
        self.cache = cache
        self.base = base

    def of(self, path: Path) -> str | None:
        if path.is_file():
            return self._file(path)
        if path.is_dir():
            h = hashlib.sha256()
            for f in _files(path):
                h.update(f.relative_to(path).as_posix().encode() + b"\0")
                h.update((self._file(f) or "").encode() + b"\0")
            return "dir:" + h.hexdigest()
        return None

    def _file(self, path: Path) -> str:
        st = path.stat()
        key = _display(path, self.base)
        hit = self.cache.get(key)
        if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
            return str(hit[2])
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        self.cache[key] = [st.st_size, st.st_mtime_ns, digest]
        return digest


def _files(root: Path) -> list[Path]:
    """Files under ``root`` (sorted), ignoring hidden paths such as staging dirs or .git."""
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )


def load_lock(path: Path) -> dict[str, Any]:
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": LOCK_VERSION, "steps": {}, "files": {}}
    if lock.get("version") != LOCK_VERSION:
        return {"version": LOCK_VERSION, "steps": {}, "files": {}}
    lock.setdefault("steps", {})
    lock.setdefault("files", {})
    return lock


def _save_json(path: Path, data: dict[str, Any], *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    text = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=sort_keys)
    tmp.write_text(text + "\n", "utf-8")
    os.replace(tmp, path)


# --- planning and running ---------------------------------------------------


def _key(step: Step, input_digests: dict[str, str | None]) -> str:
    payload = {"kind": step.kind, "options": step.options, "inputs": input_digests,
               "convmerge": __version__}  # fmt: skip
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _decide(
    step: Step,
    recipe: Recipe,
    lock: dict[str, Any],
    digests: _Digests,
    forced: bool,
) -> tuple[str, str, dict[str, str | None]]:
    inputs = {_display(p, recipe.base_dir): digests.of(p) for p in step.inputs}
    missing = [k for k, v in inputs.items() if v is None]
    if missing:
        return "error", f"input not found: {missing[0]}", inputs
    prev = lock["steps"].get(step.name)
    if forced:
        return "run", "forced", inputs
    if prev is None:
        return "run", "not run before", inputs
    if prev.get("convmerge") != __version__:
        return "run", f"convmerge {prev.get('convmerge')} -> {__version__}", inputs
    if prev.get("options") != step.options:
        return "run", "options changed", inputs
    changed = [k for k, v in inputs.items() if prev.get("inputs", {}).get(k) != v]
    if changed:
        return "run", f"input changed: {changed[0]}", inputs
    out_digest = digests.of(step.output)
    if out_digest is None:
        return "run", "output missing", inputs
    if out_digest != prev.get("output"):
        return "run", "output modified", inputs
    return "skip", "up to date", inputs


def _matches(step: Step, patterns: list[str] | None) -> bool:
    if patterns is None:
        return False
    if not patterns:
        return True
    return any(p in (step.name, step.kind, step.source) for p in patterns)


def plan(recipe: Recipe, *, force: list[str] | None = None, steps: list[Step] | None = None
         ) -> list[PlannedStep]:  # fmt: skip
    """What ``run`` would do, without doing it.

    A step downstream of one that will run is reported as ``run`` ("after
    <step>"); at run time it is skipped if its inputs turn out unchanged.
    """
    steps = steps if steps is not None else build_steps(recipe)
    lock = load_lock(recipe.lock_path)
    digests = _Digests(dict(lock["files"]), recipe.base_dir)
    will_run: dict[Path, str] = {}
    planned: list[PlannedStep] = []
    for step in steps:
        upstream = next((will_run[p] for p in step.inputs if p in will_run), None)
        if upstream is not None and not _matches(step, force):
            action, reason = "run", f"after {upstream}"
        else:
            action, reason, _ = _decide(step, recipe, lock, digests, _matches(step, force))
        if action == "run":
            will_run[step.output] = step.name
        planned.append(PlannedStep(step, action, reason))
    return planned


def run(
    recipe: Recipe,
    *,
    force: list[str] | None = None,
    hf_token: str | None = None,
    github_token: str | None = None,
    log: LogFn | None = None,
) -> RunResult:
    """Run every step that is not up to date; update the lock file and write the report."""
    log = log or (lambda m: print(m, file=sys.stderr))
    steps = build_steps(recipe, hf_token=hf_token, github_token=github_token)
    lock = load_lock(recipe.lock_path)
    lock["convmerge"] = __version__
    lock["recipe"] = _display(recipe.path, recipe.base_dir)
    digests = _Digests(lock["files"], recipe.base_dir)
    result = RunResult()
    report: dict[str, Any] = {"version": REPORT_VERSION, "convmerge": __version__, "steps": {}}

    for step in steps:
        action, reason, inputs = _decide(step, recipe, lock, digests, _matches(step, force))
        if action == "error":
            raise RecipeRunError(f"{step.name}: {reason}")
        if action == "skip":
            log(f"[skip] {step.name} ({reason})")
            result.skipped.append(step.name)
            report["steps"][step.name] = {"action": "skipped",
                                          **lock["steps"][step.name].get("report", {})}  # fmt: skip
            continue
        log(f"[run]  {step.name} ({reason})")
        started = time.perf_counter()
        stats = _execute(step)
        seconds = round(time.perf_counter() - started, 3)
        entry = {
            "kind": step.kind,
            "options": step.options,
            "inputs": inputs,
            "output": digests.of(step.output),
            "convmerge": __version__,
            "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "report": {"seconds": seconds, "stats": stats},
        }
        lock["steps"][step.name] = entry
        _save_json(recipe.lock_path, lock)
        result.ran.append(step.name)
        report["steps"][step.name] = {"action": "ran", "seconds": seconds, "stats": stats}

    known = {s.name for s in steps}
    lock["steps"] = {k: v for k, v in lock["steps"].items() if k in known}
    # Forget cached digests of files that no longer exist.
    lock["files"] = {k: v for k, v in lock["files"].items() if (recipe.base_dir / k).is_file()}
    _save_json(recipe.lock_path, lock)
    report["output"] = {
        "path": _display(recipe.output, recipe.base_dir),
        "sha256": digests.of(recipe.output),
        "records": _count_lines(recipe.output),
    }
    report["licenses"], warnings = _licenses(recipe, report["steps"])
    report["license_warnings"] = warnings
    for line in warnings:
        log(f"[license] {line}")
    _save_json(recipe.report_path, report, sort_keys=False)  # keep step order
    result.report = report
    return result


def _licenses(recipe: Recipe, steps: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    from convmerge.licenses import license_summary

    info: dict[str, dict[str, Any]] = {}
    for name, src in recipe.sources.items():
        fetched = steps.get(f"{name}.fetch", {}).get("stats", {})
        converted = steps.get(f"{name}.convert", {}).get("stats", {})
        info[name] = {
            "declared": src.license,
            "detected": fetched.get("license"),
            "rows": converted.get("written"),
        }
    return license_summary(info)


def _execute(step: Step) -> dict[str, Any]:
    step.output.parent.mkdir(parents=True, exist_ok=True)
    stage = step.output.with_name(f".{step.output.name}.part")
    _remove(stage)
    try:
        stats = step.run(stage)
    except Exception as e:
        _remove(stage)
        if isinstance(e, RecipeRunError):
            raise RecipeRunError(f"{step.name}: {e}") from e
        raise RecipeRunError(f"{step.name}: {type(e).__name__}: {e}") from e
    old = step.output.with_name(f".{step.output.name}.old")
    _remove(old)
    if step.output.exists():
        os.replace(step.output, old)
    os.replace(stage, step.output)
    _remove(old)
    return stats


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _display(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
