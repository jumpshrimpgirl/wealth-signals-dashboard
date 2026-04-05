"""
Smart 0–100 prospect scores from entity cues, role, company, story text, money, and recency.
Does not filter rows — only sets ``confidence`` for prioritization.
"""

from __future__ import annotations

import re
from typing import Any

_MONEY_SCORE = re.compile(
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


def score_prospect(p: dict[str, Any]) -> int:
    name = (p.get("name") or p.get("person_name") or "").lower()
    role = (p.get("role") or "").lower()
    company = (p.get("company") or p.get("company_name") or "").lower()
    text = (p.get("summary") or "").lower()
    if not text.strip():
        text = " ".join(
            str(p.get(k) or "")
            for k in ("raw_title", "full_explanation", "ai_summary")
        ).lower()

    score = 0

    if any(
        x in name
        for x in [
            "department",
            "agency",
            "licensing",
            "government",
            "middle east",
            "region",
            "committee",
        ]
    ):
        score -= 50

    if len(name.split()) < 2:
        score -= 30

    if "bot" in text or "named after" in text:
        score -= 50

    if any(x in text for x in ["died", "historian", "former secretary"]):
        score -= 40

    if "founder" in role:
        score += 40
    elif "ceo" in role:
        score += 35
    elif "partner" in role:
        score += 25
    elif "director" in role:
        score += 15
    else:
        score -= 20

    if not company or company == "unknown":
        score -= 30
    else:
        score += 10

    if any(x in text for x in ["raised", "funding", "series"]):
        score += 30

    if any(x in text for x in ["revenue", "growth", "profit"]):
        score += 25

    if any(x in text for x in ["acquisition", "merger", "deal"]):
        score += 30

    for m in _MONEY_SCORE.finditer(text):
        raw_amt = m.group(1)
        unit_raw = m.group(2)
        try:
            val = float(str(raw_amt).replace(",", ""))
        except (ValueError, TypeError):
            continue
        unit = _normalize_money_unit(unit_raw)
        if unit == "billion":
            score += 40
        elif unit == "million" and val > 100:
            score += 25

    if "today" in text or "hours ago" in text or "just" in text:
        score += 20

    return max(0, min(100, score))


def apply_prospect_scores(prospects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Set ``confidence`` from :func:`score_prospect` and sort highest first (does not drop rows)."""
    out: list[dict[str, Any]] = []
    for p in prospects:
        row = dict(p)
        row["confidence"] = score_prospect(row)
        out.append(row)
    out.sort(key=lambda x: int(x.get("confidence") or 0), reverse=True)
    return out
