"""
Final display gates: Home / Top prospects vs Explore (broad recall).

Home must never show non-person junk, commentary-only FA misfires, or mislabeled signal types.
"""

from __future__ import annotations

import re
from typing import Any

# --- Structured extraction normalization ---------------------------------------------------------

_ENTITY_TYPES = frozenset(
    {"person", "company", "product", "organization", "region", "event", "unknown"}
)


def normalize_extracted_candidate(c: dict[str, Any]) -> dict[str, Any]:
    """Merge v2/v3 cache fields into a single schema for downstream code."""
    if not isinstance(c, dict):
        return {}
    out = dict(c)
    if "is_valid_prospect_person" not in out:
        out["is_valid_prospect_person"] = bool(out.get("is_real_person", True))
    et = str(out.get("entity_type") or "").strip().lower()
    if not et or et not in _ENTITY_TYPES:
        out["entity_type"] = "person" if out["is_valid_prospect_person"] else "unknown"
    else:
        out["entity_type"] = et
    if "is_real_person" not in out:
        out["is_real_person"] = out["is_valid_prospect_person"]
    return out


# --- Forbidden names (hard block) ---------------------------------------------------------------

_FRAT_PATTERN = re.compile(
    r"\b(alpha|beta|gamma|delta|sigma|omega|kappa|zeta|theta|phi|psi)\s+"
    r"(alpha|beta|gamma|delta|sigma|omega|kappa|zeta|theta|phi|psi)\s+"
    r"(alpha|beta|gamma|delta|sigma|omega|kappa|zeta|theta|phi|psi)\b",
    re.I,
)
_PRODUCT_MODEL = re.compile(
    r"\b(iphone|ipad|macbook|pixel|galaxy|model\s+[a-z0-9]+|series\s+[a-z0-9]+|"
    r"tesla\s+model|ford\s+f-\d+|bmw\s+\w+)\b",
    re.I,
)


def is_forbidden_display_name(name: str, article_text: str = "") -> bool:
    """
    True when the string must never be shown as a person (org/product/list/case/fraternity).
    Explore rows that fail this should be skipped entirely.
    """
    n = re.sub(r"\s+", " ", (name or "").strip())
    if not n:
        return True
    low = n.lower()
    art = (article_text or "").lower()

    if _FRAT_PATTERN.search(n):
        return True
    if _PRODUCT_MODEL.search(n):
        return True
    # Repeated token (Delta Delta Delta, etc.)
    parts = low.split()
    if len(parts) >= 2 and len(set(parts)) == 1:
        return True

    block_sub = (
        "list of",
        "murder",
        "timeline",
        "department of",
        "agency",
        "ministry",
        "university",
        "technology business",
        "middle east",
        "european union",
        "the white house",
        "murder case",
        "search:",
    )
    for s in block_sub:
        if s in low:
            return True
    if "case" in low and any(x in low for x in ("murder", "naina", "trial", "lawsuit")):
        return True
    if low in ("santa", "list of punjabi people"):
        return True
    if low.startswith("the ") and "grass roots" in low:
        return True

    # Strong list / index titles
    if re.search(r"\blist of\b", low):
        return True

    return False


def is_commentary_only(candidate: dict[str, Any], article_text: str) -> bool:
    """
    Quote / pundit context without founder-owner-operating relevance.
    Used to cap scores and block Home.
    """
    t = (article_text or "").lower()
    ct = str(candidate.get("context_type") or "").lower()
    er = str(candidate.get("economic_role") or "").lower()

    if ct == "commentary" and er == "commentator":
        return True

    quote_heavy = sum(
        1
        for k in (
            "told cnbc",
            "said in an interview",
            "according to",
            "analyst at",
            "commentator",
            "pundit",
            "weighed in",
            "speaking to",
        )
        if k in t
    )
    if quote_heavy >= 1 and er in ("commentator", "other", "investor"):
        if not any(
            k in t
            for k in (
                "founder",
                "co-founder",
                "chief executive",
                "ceo of",
                "owner of",
                "sold",
                "acquisition",
                "raised",
                "funding",
            )
        ):
            return True
    return False


def lawyer_only_without_business_event(candidate: dict[str, Any], article_text: str) -> bool:
    er = str(candidate.get("economic_role") or "").lower()
    if er != "lawyer":
        return False
    t = (article_text or "").lower()
    return not any(
        k in t
        for k in (
            "founder",
            "co-founder",
            "owner",
            "stake",
            "acquisition",
            "merger",
            "ipo",
            "sold",
            "business",
            "company",
            "firm he founded",
            "her company",
        )
    )


def sanitize_signal_type(article_text: str, candidate: dict[str, Any], article_signal_type: str) -> str:
    """
    Row-level signal label: do not infer Founder Exit / Funding / Promotion without support.
    """
    if is_commentary_only(candidate, article_text):
        return "Other"

    t = (article_text or "").lower()
    er = str(candidate.get("economic_role") or "").lower()
    ct = str(candidate.get("context_type") or "").lower()

    funding_ok = any(
        k in t
        for k in (
            "raised",
            "raising",
            "funding",
            "series a",
            "series b",
            "series c",
            "venture round",
            "investment round",
            "seed round",
        )
    )
    exit_ok = any(
        k in t
        for k in (
            "acquisition",
            "acquired",
            "merger",
            "sold his",
            "sold her",
            "stake sale",
            "buyout",
            "liquidity event",
            "exit",
            "divest",
        )
    )
    founder_owner = any(
        k in t for k in ("founder", "co-founder", "cofounder", "owner", "chief executive", "ceo of")
    ) or er in ("founder", "owner", "ceo")
    promo_ok = any(
        k in t
        for k in (
            "appointed",
            "appointment",
            "named ceo",
            "named chief",
            "succeeds",
            "succession",
            "promoted to",
            "new role",
            "will become",
        )
    )

    if funding_ok and not is_commentary_only(candidate, article_text):
        return "Funding"
    if exit_ok and founder_owner and not is_commentary_only(candidate, article_text):
        if "ipo" in t or "going public" in t:
            return "M&A"
        return "Founder Exit"
    if promo_ok and er in ("executive", "ceo", "founder", "other"):
        return "Promotion"
    if article_signal_type in ("Founder Wealth Creation", "Revenue") and founder_owner:
        if is_commentary_only(candidate, article_text) or er == "commentator":
            return "Other"
        return "Founder Wealth Creation"
    if article_signal_type in ("M&A", "Funding", "Revenue"):
        return article_signal_type
    return "Other"


def _cell_na(row: Any, key: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        v = row.get(key, default)
    else:
        try:
            v = row[key]
        except (KeyError, TypeError, IndexError):
            v = default
    if v is None:
        return default
    try:
        import pandas as pd

        if isinstance(v, float) and pd.isna(v):
            return default
    except Exception:
        pass
    return v


def can_render_on_home(row: Any) -> bool:
    """
    Final gate for Home / Top prospect cards. Explore can still show rows that fail this.
    """
    name = str(_cell_na(row, "name") or _cell_na(row, "person_name") or "").strip()
    if not name:
        return False

    summary = str(_cell_na(row, "summary") or "")

    raw_et = _cell_na(row, "entity_type", None)
    et = str(raw_et).strip().lower() if raw_et is not None and str(raw_et).strip() else ""
    if et in ("company", "product", "organization", "region", "event", "unknown"):
        return False
    if et not in ("person", ""):
        return False
    if not et:
        from prospect_hardening import is_valid_person_name

        if not is_valid_person_name(name, summary):
            return False

    ivp = _cell_na(row, "is_valid_prospect_person", None)
    if ivp is None:
        from prospect_hardening import is_valid_person_name

        ivp = is_valid_person_name(name, summary)
    if not bool(ivp):
        return False

    if is_forbidden_display_name(name, summary):
        return False

    ctx = str(_cell_na(row, "context_type") or "").lower()
    if ctx not in ("primary", "secondary"):
        return False

    try:
        ps = int(_cell_na(row, "priority_score") or _cell_na(row, "score") or 0)
    except (TypeError, ValueError):
        ps = 0
    if ps < 55:
        return False

    if bool(_cell_na(row, "candidate_historical_dead")):
        return False

    if bool(_cell_na(row, "commentary_only_row")):
        return False

    cand = {
        "economic_role": _cell_na(row, "economic_role"),
        "context_type": _cell_na(row, "context_type"),
    }
    if lawyer_only_without_business_event(cand, summary):
        return False

    return True


__all__ = [
    "can_render_on_home",
    "is_commentary_only",
    "is_forbidden_display_name",
    "lawyer_only_without_business_event",
    "normalize_extracted_candidate",
    "sanitize_signal_type",
]
