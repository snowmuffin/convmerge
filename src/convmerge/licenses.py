"""Dataset licenses: detect them from Hugging Face dataset cards, flag risky mixes.

A recipe source may declare ``license:``; for Hugging Face sources without
one, ``fetch`` reads the ``license`` field of the dataset card. The build
report lists every source's license with how many rows it contributed, and
warns about sources whose terms restrict use (non-commercial or research-only
licenses, custom ``other`` terms) or that have no known license. convmerge
does not judge compatibility; it makes the mix visible.
"""

from __future__ import annotations

import re
from typing import Any

_RESTRICTED = re.compile(
    r"(?:^|[^a-z])nc(?:[^a-z]|$)|non-?commercial|research[- ]only|research use", re.I
)


def detect_hf_license(dataset_id: str, *, token: str | None = None) -> str | None:
    """The ``license`` of a Hub dataset's card, or ``None`` (no card, offline, gated...)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(dataset_id, token=token)
    except Exception:  # noqa: BLE001 - detection is best effort
        return None
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    value: Any = None
    if card is not None:
        value = card.get("license") if hasattr(card, "get") else getattr(card, "license", None)
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value if v)
    text = str(value).strip() if value else ""
    return text or None


def license_warning(license: str | None) -> str | None:
    """Why a license needs a look before the data is used, or ``None``."""
    if not license or license.strip().lower() in ("unknown", "none"):
        return "no known license"
    if _RESTRICTED.search(license):
        return "non-commercial or research-only terms"
    if license.strip().lower() == "other":
        return "custom terms (license: other); read the dataset card"
    return None


def license_summary(
    sources: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Per-source license entries and warning lines.

    ``sources`` maps a source name to ``{"declared", "detected", "rows"}``;
    the declared license wins over the detected one.
    """
    summary: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for name, info in sources.items():
        license = info.get("declared") or info.get("detected")
        entry: dict[str, Any] = {
            "license": license,
            "source": "declared" if info.get("declared") else "card" if license else None,
            "rows": info.get("rows"),
        }
        problem = license_warning(license)
        if problem:
            entry["warning"] = problem
            shown = f" ({license})" if license else ""
            warnings.append(f"{name}{shown}: {problem}")
        summary[name] = entry
    return summary, warnings
