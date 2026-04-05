"""
Scoring logic for wealth signals.

Additive model starting at 0; clamp to 0–100 only at the end of the pipeline (see data.py).
"""

from __future__ import annotations

import re
from typing import Any

# --- Deal-size token (aligned with data.py headline extraction) ---
_DEAL_VALUE_RE = re.compile(
    r"\$([0-9]+(?:\.[0-9]+)?)\s*(M|B|million|billion)\b",
    re.I,
)


def _usd_from_deal_match(m: re.Match[str]) -> float:
    n = float(m.group(1))
    u = (m.group(2) or "").lower()
    if u in ("m", "million"):
        return n * 1_000_000
    if u in ("b", "billion"):
        return n * 1_000_000_000
    return 0.0


def _max_deal_value_usd(text: str) -> float:
    best = 0.0
    for m in _DEAL_VALUE_RE.finditer(text or ""):
        v = _usd_from_deal_match(m)
        if v > best:
            best = v
    return best


_FUNDING_TOKEN_RE = re.compile(
    r"^\$\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?\s*$",
    re.I,
)


def _parse_funding_token_usd(token: str) -> float | None:
    s = (token or "").strip().replace(",", "")
    m = _FUNDING_TOKEN_RE.match(s)
    if not m:
        return None
    try:
        n = float(m.group(1))
    except ValueError:
        return None
    suf = (m.group(2) or "").upper()
    mult = 1.0
    if suf == "K":
        mult = 1_000
    elif suf == "M":
        mult = 1_000_000
    elif suf == "B":
        mult = 1_000_000_000
    return n * mult


def _funding_detected(event_type: str, funding_amount: str, funding_stage: str) -> bool:
    if str(event_type or "").strip() == "Funding":
        return True
    fa = str(funding_amount or "").strip()
    v = _parse_funding_token_usd(fa)
    if v is not None and v > 0:
        return True
    return bool(str(funding_stage or "").strip())


def _deal_detected(event_type: str, raw_title: str, full_explanation: str) -> bool:
    if str(event_type or "").strip() == "Founder Exit":
        return True
    blob = f"{raw_title} {full_explanation}".lower()
    if _max_deal_value_usd(f"{raw_title} {full_explanation}") > 0:
        return True
    return bool(
        re.search(r"\b(acquisition|acquired|acquires|buyout|merger)\b", blob)
    )


_EXEC_ROLE_RE = re.compile(r"\b(ceo|cfo|coo|cto)\b", re.I)


def _executive_role(role: str) -> bool:
    return bool(_EXEC_ROLE_RE.search(role or ""))


def compute_signal_score(
    event_type: str,
    person_name: str = "",
    company_name: str = "",
    role: str = "",
    *,
    raw_title: str = "",
    full_explanation: str = "",
    funding_amount: str = "",
    funding_stage: str = "",
) -> int:
    """
    Additive score from 0 — no clamp here.

    +30 funding detected, +40 deal detected, +20 executive role, +15 person present,
    +10 company present; -25 no person, -30 Other event type.
    """
    score = 0

    if _funding_detected(event_type, funding_amount, funding_stage):
        score += 30
    if _deal_detected(event_type, raw_title, full_explanation):
        score += 40
    if _executive_role(role):
        score += 20

    pn = str(person_name or "").strip()
    if pn:
        score += 15
    else:
        score -= 25

    cn = str(company_name or "").strip()
    if cn and cn.lower() != "unknown":
        score += 10

    if (event_type or "").strip() == "Other":
        score -= 30

    return score


def score_for_event_type(event_type: str) -> int:
    """Type-only row (no body text): still applies Other penalty; no funding/deal boosts."""
    return compute_signal_score(event_type, "", "", "")


def apply_scores(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach ``score`` using the additive model when row fields are present."""
    out = []
    for row in signals:
        row = dict(row)
        row["score"] = compute_signal_score(
            str(row.get("event_type", "") or ""),
            str(row.get("person_name", "") or ""),
            str(row.get("company_name", "") or ""),
            str(row.get("role", "") or ""),
            raw_title=str(row.get("raw_title", "") or ""),
            full_explanation=str(row.get("full_explanation", "") or ""),
            funding_amount=str(row.get("funding_amount", "") or ""),
            funding_stage=str(row.get("funding_stage", "") or ""),
        )
        out.append(row)
    return out


def clamp_score_0_100(score: int | float) -> int:
    """Final-step clamp for pipeline use."""
    try:
        s = int(round(float(score)))
    except (TypeError, ValueError):
        s = 0
    return max(0, min(100, s))
