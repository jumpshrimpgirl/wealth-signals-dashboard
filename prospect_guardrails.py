"""
Hard string / token guardrails before client scoring (complements AI validation).
"""

from __future__ import annotations

import re
from typing import Any

# Substrings in *name* — reject (case-insensitive)
_NAME_BANNED_SUBSTRINGS = (
    "list",
    "case",
    "region",
    "department",
    "technology",
    "model",
)

# Company values that are generic / junk as standalone org labels
_BAD_GENERIC_COMPANIES = frozenset(
    {
        "santa",
        "delta",
        "technology",
        "business",
        "unknown",
        "n/a",
        "none",
    }
)


def hard_guardrails_pass(
    name: str,
    company: str,
    role: str,
    article_text: str,
) -> bool:
    """
    Return True if row may proceed after AI validation.
    Rejects obvious junk names and generic company tokens.
    """
    n = (name or "").strip()
    co = (company or "").strip()
    rl = (role or "").strip().lower()
    art = (article_text or "").lower()

    if not n:
        return False

    low = n.lower()
    for frag in _NAME_BANNED_SUBSTRINGS:
        if frag in low:
            return False

    col = co.lower().strip()
    if col and col in _BAD_GENERIC_COMPANIES:
        return False
    if col and len(col.split()) == 1 and col in _BAD_GENERIC_COMPANIES:
        return False

    return True


__all__ = ["hard_guardrails_pass"]
