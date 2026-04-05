"""
Scoring logic for wealth signals.

Higher scores mean a stronger signal that someone may have had a meaningful
financial or career event worth paying attention to.
"""

from __future__ import annotations

from typing import Any

# Base scores by event type (used when building or refreshing signal rows).
EVENT_SCORES = {
    "Founder Exit": 92,  # 90+ range per product spec
    "Funding": 80,
    "Promotion": 70,
    "Board Appointment": 65,
    "Other": 55,  # Broad finance/career headlines that did not match core categories
}


def score_for_event_type(event_type: str) -> int:
    """
    Return the canonical score for a given event type.

    Unknown types default to a moderate score so the dashboard still works
    if you add new categories later.
    """
    if event_type not in EVENT_SCORES:
        return 50
    return EVENT_SCORES[event_type]


def apply_scores(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Given a list of signal dicts (each must include 'event_type'),
    attach or overwrite 'score' using the scoring rules.
    """
    out = []
    for row in signals:
        row = dict(row)
        row["score"] = score_for_event_type(row["event_type"])
        out.append(row)
    return out
