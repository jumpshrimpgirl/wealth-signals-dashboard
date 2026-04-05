"""
Sanity-checked wealth display: separate article money from personal net worth.

Used by the hybrid pipeline and final UI validation — never invent precision.
"""

from __future__ import annotations

import re
from typing import Any

# Rough ceiling for any numeric display without Forbes/Bloomberg-style list proof
_MAX_UNVERIFIED_DISPLAY_USD = 100_000_000  # $100M
_MAX_PLAUSIBLE_ANY_DISPLAY_USD = 300_000_000_000  # $300B — above this without list proof is junk

_MONEY_NUM = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|bn|m\b|b\b)?",
    re.I,
)

_NET_WORTH_NEAR = re.compile(
    r"(?:net\s+worth|worth\s+an\s+estimated|fortune\s+of)[^\$\n]{0,120}\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|bn|m\b)?",
    re.I,
)

_REVENUE_VALUATION = re.compile(
    r"\b(revenue|sales|valuation|funding|raised|series|round|market\s+cap|deal|acquisition|ipo)\b",
    re.I,
)


def _parse_usd_from_match(m: re.Match[str] | None) -> float | None:
    if not m:
        return None
    try:
        val = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    u = (m.group(2) or "").lower()
    if u.startswith("b") or u == "bn":
        return val * 1e9
    if u.startswith("m") or u == "m":
        return val * 1e6
    if val >= 1e6:
        return val
    return val


def parse_money_string_to_usd(s: str) -> float | None:
    """Best-effort parse of a single $X M/B style fragment."""
    if not s or "$" not in s:
        return None
    m = _MONEY_NUM.search(s)
    return _parse_usd_from_match(m)


def _format_usd_short(usd: float) -> str:
    if usd >= 1e9:
        return f"${usd / 1e9:.1f}B".replace(".0B", "B")
    if usd >= 1e6:
        return f"${usd / 1e6:.0f}M"
    if usd >= 1e3:
        return f"${usd / 1e3:.0f}K"
    return f"${usd:,.0f}"


def article_states_explicit_person_net_worth(article_text: str, person_name: str) -> bool:
    """True if article clearly attributes a net worth / fortune figure to a person (not company revenue)."""
    t = article_text or ""
    if not t.strip():
        return False
    low = t.lower()
    if _REVENUE_VALUATION.search(t) and "net worth" not in low and "fortune" not in low:
        # Heuristic: require explicit personal wealth language
        pass
    for m in _NET_WORTH_NEAR.finditer(t):
        span = t[max(0, m.start() - 200) : m.end() + 80].lower()
        pn = (person_name or "").strip().lower()
        if pn and any(w in span for w in pn.split() if len(w) > 2):
            return True
        if "his " in span or "her " in span or "whose " in span:
            return True
    if "net worth" in low and "$" in t:
        pn = (person_name or "").strip().lower()
        idx = low.find("net worth")
        window = low[max(0, idx - 250) : idx + 180]
        if pn and any(w in window for w in pn.split() if len(w) > 2):
            return True
    return False


def _externally_verified_wealth_number(
    cross_check_result: dict[str, Any],
    article_text: str,
    candidate: dict[str, Any],
) -> tuple[float | None, str]:
    """
    Returns (usd, source_tag) only when a specific number is justified.
    Does NOT use article-wide largest $ from revenue/funding.
    """
    wl = cross_check_result.get("_wealth_list") or {}
    we = str(cross_check_result.get("wealth_evidence") or "")
    agree = int(cross_check_result.get("agree_sources") or 0)
    est = str(cross_check_result.get("est_wealth") or "").strip()

    # Forbes / identity DB list (explicit list match)
    if wl.get("list_match") and "$" in est:
        v = parse_money_string_to_usd(est)
        if v is not None:
            return v, "wealth_list"

    # Explicit personal net worth in article copy (not company revenue headline)
    if article_states_explicit_person_net_worth(article_text, str(candidate.get("name") or "")):
        m = _NET_WORTH_NEAR.search(article_text or "")
        v = _parse_usd_from_match(m)
        if v is not None:
            return v, "article_explicit_nw"

    # Enrichment "direct" only with strong multi-source agreement (avoid wiki noise)
    if we == "direct" and "$" in est and agree >= 2:
        v = parse_money_string_to_usd(est)
        if v is not None and not _REVENUE_VALUATION.search(est):
            return v, "cross_check_direct"

    return None, ""


def estimate_wealth_safely(
    candidate: dict[str, Any],
    article_text: str,
    cross_check_result: dict[str, Any],
    *,
    wealth_status_hint: str | None = None,
) -> dict[str, Any]:
    """
    Returns:
      est_wealth: display string (never absurd precision without evidence)
      wealth_status: verified_wealth | likely_wealth | emerging_founder | unclear
      wealth_confidence: 0-100
      wealth_numeric_verified: bool (True only when a dollar figure is justified)
    """
    from ai_prospect_pipeline import classify_wealth_status

    ws = wealth_status_hint or classify_wealth_status(article_text, candidate, cross_check_result)

    num_usd, src = _externally_verified_wealth_number(cross_check_result, article_text, candidate)
    wl = cross_check_result.get("_wealth_list") or {}
    forbes_style = bool(wl.get("list_match")) and cross_check_result.get("prominence_tag") == "forbes_billionaires"

    wealth_numeric_verified = False
    est_wealth = "Data pending"
    conf = 32
    agree = int(cross_check_result.get("agree_sources") or 0)

    if num_usd is not None and src:
        if num_usd > _MAX_PLAUSIBLE_ANY_DISPLAY_USD and not forbes_style:
            est_wealth = "Likely high, unverified"
            ws = "likely_wealth"
            conf = 42
        elif num_usd > _MAX_UNVERIFIED_DISPLAY_USD and not (forbes_style or src in ("article_explicit_nw", "wealth_list")):
            est_wealth = "Likely high, unverified"
            ws = "likely_wealth"
            conf = 48
        else:
            est_wealth = _format_usd_short(num_usd)
            wealth_numeric_verified = True
            ws = "verified_wealth"
            conf = min(95, 58 + agree * 5)
    else:
        if ws == "verified_wealth":
            ws = "likely_wealth"
            est_wealth = "Likely high, unverified"
            conf = 52
        elif ws == "likely_wealth":
            est_wealth = "Likely high, unverified"
            conf = 48
        elif ws == "emerging_founder":
            est_wealth = "Emerging"
            conf = 44
        else:
            est_wealth = "Data pending"
            conf = 30

    return {
        "est_wealth": est_wealth,
        "wealth_status": ws,
        "wealth_confidence": max(0, min(100, conf)),
        "wealth_numeric_verified": wealth_numeric_verified,
    }


def validate_display_wealth(candidate_row: dict[str, Any]) -> dict[str, Any]:
    """
    Final gate before UI: strip absurd or unverified giant numbers.
    Mutates nothing — returns display fields to merge.
    """
    est = str(candidate_row.get("est_wealth") or candidate_row.get("est_wealth_display") or "").strip()
    verified = bool(candidate_row.get("wealth_numeric_verified"))

    out: dict[str, Any] = {"est_wealth_display": est, "wealth_numeric_verified": verified}

    v = parse_money_string_to_usd(est)
    if v is None:
        return out

    if not verified and v > _MAX_UNVERIFIED_DISPLAY_USD:
        out["est_wealth_display"] = "Likely high, unverified"
        out["wealth_numeric_verified"] = False
        return out

    if v > _MAX_PLAUSIBLE_ANY_DISPLAY_USD and not verified:
        out["est_wealth_display"] = "Likely high, unverified"
        out["wealth_numeric_verified"] = False
        return out

    return out
