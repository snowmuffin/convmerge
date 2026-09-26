# Benchmarks

`bench.py` generates synthetic chat data and reports wall time and peak RSS
for `convert`, `convert --workers 4`, `dedupe`, `mix --total`, and `mix`
(merge all), each in its own process.

```bash
PYTHONPATH=src python benchmarks/bench.py --rows 200000 --json before.json
```

Reference numbers (200k rows per 119 MB source, 3 sources for `mix`; one
Linux container, so compare runs on the same machine only):

| Scenario | 0.6.0 | 0.7.0 |
|----------|-------|-------|
| `convert` | 6.2 s / 14 MB | 6.4 s / 14 MB |
| `convert --workers 4` | n/a | 1.7 s / 51 MB* |
| `dedupe` | 3.1 s / 45 MB | 3.1 s / 45 MB |
| `mix --total 20000` | 4.8 s / 405 MB | 3.1 s / 30 MB |
| `mix` (merge all) | 5.8 s / 415 MB | 7.0 s / 31 MB |

\* peak RSS of the main process; each of the 4 workers adds roughly the
single-process footprint. Measured on a 4-core container.
