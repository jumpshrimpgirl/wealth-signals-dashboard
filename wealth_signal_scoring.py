"""
Advanced wealth-signal scoring from article text (event types, money magnitude, growth, institutional).
Used for high-value article detection instead of naive keyword counts.
"""

from __future__ import annotations

import re
from typing import Any

# Matches $1.5 billion, $250 million, $50M, $1.2bn, $100 (no unit)
_MONEY_RE = re.compile(
    r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|bn|m\b|b\b)?",
    re.I,
)


def _normalize_money_unit(unit: str | None) -> str:
    if not unit:
        return ""
    u = unit.strip().lower()
    if u in ("billion", "bn", "b"):
        return "billion"
    if u in ("million", "m"):
        return "million"
    return ""


def evaluate_signal_strength(article_text: str) -> dict[str, Any]:
    """
    Score financial signal strength from real patterns (events, money scale, growth, institutional).

    Returns dict: score (int), level (strong|moderate|weak), reasons (list of short strings).
    """
    text = (article_text or "").lower()
    score = 0
    reasons: list[str] = []

    event_patterns = {
        "m&a": ["acquisition", "merger", "deal", "buyout"],
        "funding": ["raised", "funding", "series", "investment"],
        "growth": ["revenue", "growth", "profit", "earnings"],
        "liquidity": ["ipo", "exit", "sold", "stake", "dividend"],
        "institutional": ["asset manager", "sovereign wealth", "fund"],
    }

    for event, keywords in event_patterns.items():
        if any(k in text for k in keywords):
            score += 15
            reasons.append(f"event:{event}")

    for m in _MONEY_RE.finditer(text):
        raw_amt = m.group(1)
        unit_raw = m.group(2)
        try:
            val = float(str(raw_amt).replace(",", ""))
        except (ValueError, TypeError):
            continue
        unit = _normalize_money_unit(unit_raw)

        if unit == "billion":
            score += 30
            reasons.append("billion_scale")
        elif unit == "million" and val > 100:
            score += 20
            reasons.append("large_million")
        elif val > 10:
            score += 10
            if "money_10plus" not in reasons:
                reasons.append("money_10plus")

    if "founded" in text or "ceo" in text or "startup" in text:
        score += 10
        reasons.append("company_context")

    if "revenue" in text and ("growth" in text or "increase" in text):
        score += 20
        reasons.append("growth_signal")

    if "projected" in text or "expected" in text:
        score += 10
        reasons.append("forward_growth")

    if "merger" in text and "asset manager" in text:
        score += 25
        reasons.append("institutional_restructuring")

    if score >= 60:
        level = "strong"
    elif score >= 30:
        level = "moderate"
    else:
        level = "weak"

    return {
        "score": score,
        "level": level,
        "reasons": reasons,
    }
