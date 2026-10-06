# Output safety

Since 1.6.4, `normalize`, `convert`, both `mix` samplers and the mix sidecar
publish regular files through the same internal writer. These are per-file
rules, not a general filesystem transaction or a concurrent-writer lock.

## What is guaranteed

A unique temporary file is created in the destination directory. The command
writes the complete result there, flushes Python buffers, calls `fsync`, closes
the file and only then replaces the destination with `os.replace`.

Before that replacement, an existing destination remains unchanged; a new
final path does not exist yet. Read, parse, serialization, write, flush, sync,
close and replace failures propagate without truncating the previous output.
Cleanup removes only the temporary file owned by that invocation. If cleanup
also fails, a warning names the leftover path and preserves the original error.
Normal empty input is a successful empty result, not a failure.

Temporary output needs disk space for the new file in addition to the previous
file. `mix` can additionally use its existing shuffle buckets. Streaming is
preserved: safety staging does not buffer the dataset in Python memory.
Final paths are completed artifacts, not a way to tail progress.

The design follows the standard library's [replace and fsync contracts](https://docs.python.org/3/library/os.html).

## Paths, permissions and links

Input/output aliases, including symlinks and hardlinks to inputs, are refused.
Direct `normalize_to_jsonl`, Parquet and table APIs apply the same checks as the
CLI. Folder normalization checks the complete source-to-output mapping before
writing, so `sample.json` and `sample.parquet` cannot both silently replace
`sample.jsonl`. Use distinct source stems or separate output directories.
An output directory inside the source tree is refused, even when empty, so a
second run cannot ingest results from the first one.

An output symlink retains the link and replaces its resolved target. Existing
basic permission bits are preserved; new files use normal `0666 & ~umask`
creation permissions. Read-only and non-regular destinations (directories,
FIFOs, devices) are refused. Replacing a file changes its inode: other hardlinks
to an old *output* still reference the old file. Owner, extended attributes and
ACL preservation are not promised. Inputs remain untouched.

## Folder and recipe behavior

Folder normalization is deliberately per-file. It continues after a bad input,
keeps that file's last successful output and commits unrelated successes. The
Python result lists successful and failed files; the CLI exits 1 if any input
failed. This does not imply a folder-wide rollback or a consistent snapshot of
a directory being modified concurrently.

Recipe steps use unique staging directories. A regular-file step replaces the
previous result directly. Replacing a non-empty directory needs moving its old
version aside first; on a handled publication error the previous directory is
restored. If the restore itself fails, the backup is retained and the exception
names its recovery location. Do not delete that location before recovering it.
Fixed `.part`/`.old` filenames not created by this invocation are not removed.
Lock/report files are published individually. Successful earlier steps remain
committed if a later step fails; output and lock updates are not one transaction.

## Explicit exclusions

- SIGKILL or process termination that bypasses Python cleanup can leave hidden
  temporary files. Before the replace, the final file is still old or absent.
  There is no automatic sweep of other executions' leftovers.
- Power-loss durability, directory metadata sync, hardware failure and network
  filesystem behavior are not guaranteed by these tests.
- Multiple processes writing one destination are not coordinated; do not rely
  on this mechanism for conflict detection or last-writer policy.
- `train`/`validation`, kept/rejected outputs, `convert --report`, data/sidecar
  pairs and a SQLite seen database are not multi-file transactions.
- Standalone `dedupe` (exact/near), `filter`, `tokens`, `decontam`, `split`,
  `turns`, fetch paths and trainer-config updates have not all migrated to the
  per-file writer. Existing source/alias guards still apply where documented,
  but do not generalize the new failure-preservation guarantee to those paths.

The release's tests inject failures at the commit boundary and compare old
bytes, not just exit codes. Cross-platform safety tests run on Linux, macOS and
Windows; signal and POSIX-permission cases are skipped where not applicable.
