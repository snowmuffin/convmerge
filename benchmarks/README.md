# Benchmarks

`bench.py` generates synthetic chat data and reports wall time and peak RSS
for `convert`, `convert --workers 4`, `dedupe`, `filter`, `filter --workers 4`,
`mix --total`, and `mix` (merge all), each in its own process.

```bash
PYTHONPATH=src python benchmarks/bench.py --rows 200000 --json before.json
```

Reference numbers (200k rows per 119 MB source, 3 sources for `mix`; one
Linux container, so compare runs on the same machine only):

| Scenario | 0.6.0 | 0.7.0 |
|----------|-------|-------|
| `convert` | 6.2 s / 14 MB | 6.5 s / 17 MB |
| `convert --workers 4` | n/a | 1.7 s / 51 MB* |
| `dedupe` | 3.1 s / 45 MB | 3.1 s / 47 MB |
| `mix --total 20000` | 4.8 s / 405 MB | 2.9 s / 32 MB |
| `mix` (merge all) | 5.8 s / 415 MB | 6.7 s / 33 MB |

\* peak RSS of the main process; each of the 4 workers adds roughly the
single-process footprint. Measured on a 4-core container.

1.0.0 and 1.1.0, measured together on another 4-core container (200k rows;
`dedupe --near` numbers on real rows are in [docs/quality.md](../docs/quality.md)):

| Scenario | 1.0.0 | 1.1.0 |
|----------|-------|-------|
| `convert` | 7.2 s / 19 MB | 7.2 s / 20 MB |
| `convert --workers 4` | 2.0 s / 53 MB* | 2.0 s / 55 MB* |
| `dedupe` | 2.5 s / 48 MB | 2.8 s / 49 MB |
| `filter` | 7.6 s / 18 MB | 8.0 s / 20 MB |
| `filter --workers 4` | n/a | 2.2 s / 55 MB* |
| `mix --total 20000` | 3.0 s / 33 MB | 2.5 s / 34 MB |
| `mix` (merge all) | 6.3 s / 35 MB | 6.3 s / 36 MB |
