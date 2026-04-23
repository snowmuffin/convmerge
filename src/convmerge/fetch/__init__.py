"""YAML-manifest driven fetcher for HuggingFace + GitHub training data.

Requires the ``fetch`` extra (``pip install 'convmerge[fetch]'``) for manifests
and GitHub-only flows. HuggingFace manifest entries need ``datasets`` — use
``fetch-hf``, ``fetch-all`` (same dependencies), or the umbrella ``all`` extra.

Public API is intentionally small: parse a manifest, iterate entries, each
writes files under ``output_root/<sanitised_name>/``. Single-dataset wrappers
are not provided on purpose: if you only need one HF dataset, call
``datasets.load_dataset`` directly.
"""

from __future__ import annotations

from convmerge.fetch.auth import AuthConfig, TokenSpec, redact_url, resolve_token
from convmerge.fetch.manifest import (
    DatasetEntry,
    Defaults,
    Manifest,
    classify_entry,
    load_manifest,
    sanitize_name,
)
from convmerge.fetch.runner import FetchResult, run_manifest

__all__ = [
    "AuthConfig",
    "DatasetEntry",
    "Defaults",
    "FetchResult",
    "Manifest",
    "TokenSpec",
    "classify_entry",
    "load_manifest",
    "redact_url",
    "resolve_token",
    "run_manifest",
    "sanitize_name",
]
