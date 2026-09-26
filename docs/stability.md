# Stability and deprecation policy

This page says what convmerge promises to keep working, and how anything
that has to change is phased out. From **1.0** the package follows
[semantic versioning](https://semver.org): what is listed here changes
incompatibly only in a new major version. **0.9** is the last 0.x release
and already follows the deprecation rules below, so code that runs on 0.9
without `DeprecationWarning`s runs on 1.0.

## What is covered

### Python API

- The names in `convmerge.__all__` ([api.md](api.md)).
- The names in `convmerge.recipe.__all__` and `convmerge.fetch.__all__`
  (these two modules are public as modules).
- For each of those: the import path, the parameter names and kinds
  (positional vs keyword-only), defaults, return types, and dataclass fields.
  New optional keyword parameters and new fields may be added in minor
  versions.

Everything else is internal and may change in any release: other modules
(`convmerge.convert`, `convmerge.normalize.jsonl`, `convmerge.cli.*`, …),
anything whose name starts with `_`, and the exact text of log messages and
exception messages (the exception *types* are covered).

### Command line

- Command names, flag names, flag values (`choices`), and what they mean.
  New commands, flags, and values may be added in minor versions.
- Exit codes:
  - `0` success;
  - `1` the work failed: a missing input file, invalid examples found by
    `validate` (or an invalid preset found by `preset validate`),
    `convert --on-invalid fail` hitting an invalid example, a failed fetch
    entry with `on_error: fail`, a failed recipe step, `run --frozen` with
    steps out of date, or any unexpected error;
  - `2` the invocation is invalid: bad flags or flag combinations, an
    unsupported fetch URL, or an invalid preset, manifest, mix config, or
    recipe passed to a command that uses it.
- Output streams: data goes to files or stdout; progress, warnings, and
  errors go to stderr only. `inspect`, `validate`, `formats`, and
  `run --plan` print to stdout.

The wording of `--help` and of stderr messages is not covered.

### Files

| File | Versioned by | Covered |
|------|--------------|---------|
| Output formats (`messages`, `alpaca`, `sharegpt`, `openai`, …) | — | Keys and their meaning, as in [format.md](format.md) |
| Drop / issue reason codes (`no_user`, `unresolved_image`, …) | — | Codes are never renamed or reused |
| `convert --report`, `validate` JSON | `"version": 1` | Keys and meaning |
| Recipe (`recipe.yaml`) | `version: 1` | Schema ([recipes.md](recipes.md)) |
| `recipe.lock.json`, `build/report.json` | `"version": 1` | Readable by every 1.x release |
| Fetch manifest | `version: 1` | Schema ([fetch.md](fetch.md)) |
| Presets | — | Schema ([custom_presets.md](custom_presets.md)) |
| `.fetch.json`, `.mix.json` sidecars | `"version": 1` | Readable by every 1.x release |

Within a version number, fields are only ever **added**. Removing or
changing the meaning of a field means a new version number, and readers keep
accepting the previous one for the rest of the major version.

### Reproducibility

- The same input, options, and convmerge version give byte-identical output,
  including `convert --workers N` for any `N`.
- `mix` with the same inputs, weights, seed, and `sampler` picks the same
  rows in every 1.x release. `sampler: v1` reproduces pre-0.7 mixes and is
  kept frozen.
- `convert` output may change in a minor or patch release **only** to fix a
  bug (for example a field that was read wrongly); the changelog says so
  under "Changed output". A recipe lock notices the new convmerge version
  and re-runs those steps.

## How things are deprecated

1. The release that deprecates something keeps it working, makes it emit a
   `DeprecationWarning` naming the replacement, and lists it under
   **Deprecated** in the [changelog](../CHANGELOG.md).
2. It is removed no earlier than the next **major** version (for 0.9
   deprecations: 1.0). Removals are listed under **Removed** with a
   migration note ([migration-1.0.md](migration-1.0.md)).
3. Behaviour that the CLI user would notice (a changed default) is announced
   the same way, with a `FutureWarning` on stderr when the old default is
   relied on.

Python hides `DeprecationWarning` outside `__main__` and test runners; run
your test suite, or `python -W error::DeprecationWarning`, to see them.

## Python versions

convmerge supports every CPython version that is not past its upstream
end-of-life when a minor release is cut, and drops a version only in a
**minor** release after that end-of-life (never in a patch release).
`requires-python` in the package metadata is always accurate, so pip picks
the last compatible release for older interpreters.

## For contributors

The covered surface is pinned by snapshot tests in `tests/contract/`: the
public API signatures, the CLI flags, and files written by earlier releases
that must still be read. A change that alters a snapshot must be
intentional: regenerate with `CONVMERGE_UPDATE_SNAPSHOTS=1 pytest
tests/contract` and explain the change in the changelog. See
[CONTRIBUTING.md](../CONTRIBUTING.md).
