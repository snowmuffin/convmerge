# Migrating to 0.7

0.7 is about scale and extension. Existing `convert`, `dedupe`, `normalize`,
and `fetch` commands produce the same output as 0.6 — `convert` output was
checked byte-for-byte against 0.6.0 on 17 real-dataset cases, with and
without `--workers`. Two things can change what you get:

| Change | Before (0.6) | Now (0.7) | To get the old behavior |
|--------|--------------|-----------|-------------------------|
| `mix` sampler | in-memory; same seed → same lines as 0.6 | streaming `v2`: bounded memory, **different lines for the same seed** | `--sampler v1` (config `sampler: v1`, API `sampler="v1"`) reproduces 0.6 mixes exactly |
| `mix --oversample` | independent draws with replacement (some records may never appear) | every record repeated `target // available` times, the rest drawn without replacement | `--sampler v1` |
| raw URL returning a Git LFS pointer | `LfsPointerError` ("use mode: clone") | object fetched through the Git LFS batch API | — |
| `convmerge.cli` | one module | a package (`convmerge.cli.convert`, `.data`, `.fetch`, `.mix`) | `convmerge.cli:main` is unchanged; internal `_cmd_*` helpers moved |

The `.mix.json` recipe now records `sampler`, so replaying an old recipe
means passing `--sampler v1`.

## New

- `mix`: streaming with bounded memory (`--sampler v2`, default).
- `fetch --max-rows N` / manifest `max_rows`: sample HF (streaming), raw, and
  tree files without downloading everything; Git LFS files without cloning.
- `convert --workers N`: parallel conversion with identical output.
- `convert --preference chosen|rejected`: reuse DPO / reward datasets for SFT.
- Plugins: `convmerge.adapters` / `convmerge.emitters` entry points,
  `register_adapter()` / `register_emitter()`, and `convmerge formats`.
- Public Python API: everything in `convmerge.__all__`, documented in
  [api.md](api.md).
