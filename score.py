"""
Scoring logic for wealth signals.

High score = actionable signal: core event type plus solid extraction (person,
company, role) is more useful for outreach and prioritization.

Low score = weak or noisy signal: thin metadata, generic “Other” bucket, or
missing fields means less confidence the row is worth acting on.
"""

from __future__ import annotations

from typing import Any

# Base points by event type (before extraction bonuses / Other penalty).
EVENT_BASE_SCORES: dict[str, int] = {
    "Founder Exit": 90,
    "Funding": 80,
    "Promotion": 70,
    "Board Appointment": 65,
    "Other": 50,
}


def compute_signal_score(
    event_type: str,
    person_name: str = "",
    company_name: str = "",
    role: str = "",
) -> int:
    """
    Combined score: event-type base + data-quality adjustments, capped at 100.

    Bonuses: +20 person, +15 real company, +10 role.

    Penalties (volume-friendly; rows are kept, not dropped): -30 when type is Other;
    -25 when person_name is empty; -15 when company is missing or Unknown.
    """
    et = (event_type or "").strip()
    base = EVENT_BASE_SCORES.get(et, 50)

    score = base
    pn = str(person_name or "").strip()
    if pn:
        score += 20
    cn = str(company_name or "").strip()
    if cn and cn.lower() != "unknown":
        score += 15
    if str(role or "").strip():
        score += 10
    if et == "Other":
        score -= 30
    if not pn:
        score -= 25
    if not cn or cn.lower() == "unknown":
        score -= 15

    return max(0, min(100, score))


def score_for_event_type(event_type: str) -> int:
    """
    Type-only score (no person/company/role). Same bases as compute_signal_score
    with no extraction bonuses—useful when row fields are not available yet.
    """
    return compute_signal_score(event_type, "", "", "")


def apply_scores(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Attach or overwrite 'score' using event_type plus extraction fields when present.
    """
    out = []
    for row in signals:
        row = dict(row)
        row["score"] = compute_signal_score(
            str(row.get("event_type", "") or ""),
            str(row.get("person_name", "") or ""),
            str(row.get("company_name", "") or ""),
            str(row.get("role", "") or ""),
        )
        out.append(row)
    return out
