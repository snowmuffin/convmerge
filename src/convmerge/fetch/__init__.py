"""YAML-manifest driven fetcher for HuggingFace + GitHub training data.

Requires the ``fetch`` extra (``pip install 'convmerge[fetch]'`` or
``convmerge[all]``) for manifests and GitHub-only flows. HuggingFace manifest
entries need ``datasets`` — use ``fetch-hf``, ``fetch-all`` (same
dependencies), or the umbrella ``all`` extra.

The names in ``__all__`` are public API (``docs/api.md``): parse a manifest
with :func:`load_manifest` and run it with :func:`run_manifest`; each entry
writes files under ``output_root/<sanitised_name>/``. Submodules are
internal. Single-dataset wrappers are not provided on purpose: if you only
need one HF dataset, call ``datasets.load_dataset`` directly.
"""

from __future__ import annotations

from convmerge.fetch.auth import AuthConfig, TokenSpec
from convmerge.fetch.manifest import DatasetEntry, Defaults, Manifest, load_manifest
from convmerge.fetch.runner import FetchResult, run_manifest

__all__ = [
    "AuthConfig",
    "DatasetEntry",
    "Defaults",
    "FetchResult",
    "Manifest",
    "TokenSpec",
    "load_manifest",
    "run_manifest",
]
