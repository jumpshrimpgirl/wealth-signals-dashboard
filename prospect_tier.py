"""
Prospect tier: Tier A = actionable FA targets; Tier B = saturated / famous; Tier C = noise.

Rows are never dropped — tiers only affect scoring and Home / Top 5 selection.
"""

from __future__ import annotations

import re
from typing import Any

# Globally famous / saturated — realistic FA outreach is rarely the marginal use case
_KNOWN_SATURATED_SUBSTR = (
    "elon musk",
    "jeff bezos",
    "larry ellison",
    "warren buffett",
    "bill gates",
    "bernard arnault",
    "mark zuckerberg",
    "tim cook",
    "sundar pichai",
    "satya nadella",
    "jamie dimon",
    "ray dalio",
    "steve ballmer",
    "michael bloomberg",
    "oprah winfrey",
    "rihanna",
    "taylor swift",
    "beyonc",
)


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _parse_net_worth_billions_hint(est: str) -> float:
    """Rough parse for 'direct' wealth strings; returns 0 if unknown."""
    s = (est or "").strip().lower()
    if not s or "$" not in s and "billion" not in s and "million" not in s:
        return 0.0
    m = re.search(
        r"\$?\s*([\d,.]+)\s*(billion|million|bn|m\b)",
        s,
        re.I,
    )
    if not m:
        return 0.0
    try:
        v = float(m.group(1).replace(",", ""))
    except ValueError:
        return 0.0
    u = m.group(2).lower()
    if u.startswith("b") or u == "bn":
        return v
    if u.startswith("m") or u == "m":
        return v / 1000.0
    return 0.0


def is_high_profile(name: str, cross_check_result: dict[str, Any]) -> bool:
    """
    True → treat as Tier B (saturated / list-famous) unless overridden by tier rules.
    """
    nl = _norm_name(name)
    if not nl:
        return False
    for sub in _KNOWN_SATURATED_SUBSTR:
        if sub in nl:
            return True

    wl = cross_check_result.get("_wealth_list") or {}
    if wl.get("list_match") and cross_check_result.get("prominence_tag") == "forbes_billionaires":
        return True

    conf = float(cross_check_result.get("identity_confidence") or 0)
    we = str(cross_check_result.get("wealth_evidence") or "")
    nw_b = _parse_net_worth_billions_hint(str(cross_check_result.get("est_wealth") or ""))
    if nw_b < 0.01:
        nw_b = _parse_net_worth_billions_hint(str(cross_check_result.get("net_worth") or ""))

    if conf >= 0.88 and we == "direct" and nw_b >= 1.0:
        return True

    en = cross_check_result.get("_enrichment") or {}
    ex = (en.get("_wikipedia_extract") or "")[:900].lower() if isinstance(en, dict) else ""
    if ex and "billionaire" in ex and ("net worth" in ex or "billion" in ex):
        return True

    return False


def _tier_c_politician_ok(article: str) -> bool:
    t = (article or "").lower()
    return any(
        x in t
        for x in (
            "founder",
            "owner",
            "business",
            "company",
            "stake",
            "ipo",
            "acquisition",
        )
    )


def classify_prospect_tier(candidate: dict[str, Any], cross_check_result: dict[str, Any]) -> str:
    """
    tier_a | tier_b | tier_c

    Optional keys on ``cross_check_result`` (set by pipeline):
    - ``_tier_article_summary``: article text for politician/mention heuristics
    - ``_tier_wealth_status``: verified_wealth | likely_wealth | emerging_founder | unclear
    - ``_tier_founder_wealth_score``: int 0–40
    """
    article = str(cross_check_result.get("_tier_article_summary") or "")
    wstat = str(cross_check_result.get("_tier_wealth_status") or "unclear")
    fws = int(cross_check_result.get("_tier_founder_wealth_score") or 0)
    own = str(cross_check_result.get("ownership_inference") or "low")

    ctx = str(candidate.get("context_type") or "").lower()
    er = str(candidate.get("economic_role") or "").lower()
    if candidate.get("is_real_person") is False:
        return "tier_c"

    if er in ("commentator", "lawyer"):
        return "tier_c"
    if er == "politician" and not _tier_c_politician_ok(article):
        return "tier_c"
    if ctx in ("mention", "commentary") and er not in ("founder", "ceo", "owner"):
        return "tier_c"
    if ctx == "historical":
        return "tier_c"

    name = str(candidate.get("name") or "")
    if is_high_profile(name, cross_check_result):
        return "tier_b"

    wl = cross_check_result.get("_wealth_list") or {}
    if wl.get("list_match") and cross_check_result.get("prominence_tag") == "forbes_billionaires":
        return "tier_b"

    founder_like = er in ("founder", "ceo", "owner") or "founder" in er
    actionable_wealth = wstat in ("likely_wealth", "emerging_founder", "verified_wealth")
    strong_ops = fws >= 15 or own == "high"

    if founder_like and (actionable_wealth or strong_ops) and ctx in ("primary", "secondary"):
        return "tier_a"
    if founder_like and ctx == "primary":
        return "tier_a"
    if ctx == "secondary" and founder_like and wstat in ("likely_wealth", "verified_wealth"):
        return "tier_a"

    # Established but not mega-famous executives / investors — still outreach-relevant vs saturated
    if er in ("executive", "investor", "other") and ctx == "primary":
        return "tier_a"

    return "tier_b"


def apply_tier_priority_adjustment(priority_score: int, prospect_tier: str) -> int:
    """Post-pass-1 adjustment; clamp 0–100."""
    ps = int(priority_score)
    t = (prospect_tier or "tier_a").lower()
    if t == "tier_a":
        ps += 15
    elif t == "tier_b":
        ps -= 25
    elif t == "tier_c":
        ps = min(ps, 25)
    return max(0, min(100, ps))
