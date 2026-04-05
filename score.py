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
        if row.pop("_use_engine_score", False):
            row["score"] = clamp_score_0_100(int(row.pop("_engine_score", 0) or 0))
        else:
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


def compute_wealth_prospect_breakdown(
    *,
    seniority_pts: int,
    wealth_pts: int,
    signal_pts: int,
    confidence_pts: int,
) -> int:
    """Explicit 0–100 model: seniority + wealth likelihood + signal + data confidence (see wealth_prospect_engine)."""
    return clamp_score_0_100(int(seniority_pts) + int(wealth_pts) + int(signal_pts) + int(confidence_pts))


def clamp_score_0_100(score: int | float) -> int:
    """Final-step clamp for pipeline use."""
    try:
        s = int(round(float(score)))
    except (TypeError, ValueError):
        s = 0
    return max(0, min(100, s))


# --- Wealth-signal prioritization (HNWI / liquidity for financial advisors) ---

_WEALTH_SIGNAL_LABELS = frozenset({"Strong", "Moderate", "Weak", "None"})
_LIQUIDITY_LABELS = frozenset({"Yes", "No", "Potential"})

_RE_HAS_MONEY_CONTEXT = re.compile(
    r"\b("
    r"net worth|billionaire|millionaire|multi[- ]?million|fortune\s*500|ultra[- ]?high|uhnw|"
    r"estate sale|trust fund|heir|inherit|bequest|dividend|payout|liquidity|stake|equity|"
    r"record contract|signing bonus|comp package|compensation package|golden parachute"
    r")\b",
    re.I,
)
_RE_JUST_MADE_MONEY = re.compile(
    r"\b("
    r"acqui|acquired|acquires|buyout|merger|ipo\b|listing|going public|spac|"
    r"sold for|sale to|divest|exit|raised\s+\$|funding round|series [a-e]|seed round|"
    r"valuation|unicorn|payout|bonus|earnout|earn[- ]?out|stock sale|sell(?:s|ing)?\s+\d|"
    r"compensation|pay package|record deal"
    r")\b",
    re.I,
)
_RE_ABOUT_TO_MAKE = re.compile(
    r"\b("
    r"planning ipo|file(?:s|d)?\s+(?:for|s-?1)|going public|expected to raise|"
    r"seeking buyers|exploring (?:a )?sale|auction|take[- ]?private|"
    r"upcoming (?:offering|round)"
    r")\b",
    re.I,
)
_RE_DOLLAR_OR_DEAL = re.compile(
    r"(\$\s*[\d,]+(?:\.\d+)?\s*[kKmMbB]?(?:illion|illion)?|\b\d+(?:\.\d+)?\s*(?:million|billion|m|b)\b)",
    re.I,
)
_RE_MACRO_NOISE = re.compile(
    r"\b("
    r"election|politic|parliament|congress(?!\s+approval)|senate race|white house(?!\s+official)|"
    r"war in|military strike|hurricane|tornado|weather alert|heatwave|"
    r"murder trial|sentenced to prison|crime scene|shoplifting"
    r")\b",
    re.I,
)
_RE_REAL_ESTATE_WEALTH = re.compile(
    r"\b("
    r"mansion|penthouse|estate sale|record price|most expensive home|luxury property|"
    r"\$\s*\d+\s*(?:million|billion).{0,40}\b(?:home|house|property|estate)\b"
    r")\b",
    re.I,
)
_RE_SPORTS_ENT = re.compile(
    r"\b("
    r"nfl|nba|mlb|nhl|olymp|super bowl|contract extension|record deal|endorsement|"
    r"box office|streaming deal|royalties"
    r")\b",
    re.I,
)
_RE_OBITUARY_WEALTH = re.compile(
    r"\b(obituary|dies at|passed away|death of)\b",
    re.I,
)


def passes_wealth_high_priority_gate(
    *,
    raw_title: str,
    full_explanation: str,
    event_type: str,
    role: str,
    estimated_wealth: float,
    aggregated_estimated_wealth: float,
    funding_amount: str,
    funding_stage: str,
    is_billionaire: bool,
) -> bool:
    """
    True only if (a) has money, (b) just made money, or (c) is about to make money.

    Used to cap priority: if False, the row cannot be HIGH.
    """
    if is_billionaire:
        return True
    ew = float(estimated_wealth or 0)
    agg = float(aggregated_estimated_wealth or 0)
    if agg >= 500_000 or ew >= 250_000:
        return True

    blob = f"{raw_title} {full_explanation}".strip()
    if not blob:
        return False

    if _RE_HAS_MONEY_CONTEXT.search(blob):
        return True
    if _funding_detected(event_type, funding_amount, funding_stage):
        return True
    if _deal_detected(event_type, raw_title, full_explanation):
        return True
    if _RE_JUST_MADE_MONEY.search(blob) and _RE_DOLLAR_OR_DEAL.search(blob):
        return True
    if _RE_ABOUT_TO_MAKE.search(blob):
        return True
    if _RE_REAL_ESTATE_WEALTH.search(blob) and _RE_DOLLAR_OR_DEAL.search(blob):
        return True
    if _RE_SPORTS_ENT.search(blob) and _RE_DOLLAR_OR_DEAL.search(blob):
        return True
    if _RE_OBITUARY_WEALTH.search(blob) and _RE_HAS_MONEY_CONTEXT.search(blob):
        return True

    if _executive_role(role) and (
        _RE_JUST_MADE_MONEY.search(blob) or _deal_detected(event_type, raw_title, full_explanation)
    ):
        return True

    return False


def is_macro_noise_without_wealth_hook(
    blob: str,
    estimated_wealth: float,
    is_billionaire: bool,
    passes_gate: bool,
) -> bool:
    """General news / macro / crime / weather with no identifiable wealth hook."""
    if is_billionaire or float(estimated_wealth or 0) >= 100_000 or passes_gate:
        return False
    b = (blob or "").strip()
    if not b:
        return False
    if not _RE_MACRO_NOISE.search(b):
        return False
    if _RE_DOLLAR_OR_DEAL.search(b) or _RE_HAS_MONEY_CONTEXT.search(b):
        return False
    return True


def classify_liquidity_event(
    event_type: str,
    raw_title: str,
    full_explanation: str,
    funding_amount: str,
    funding_stage: str,
) -> str:
    if _deal_detected(event_type, raw_title, full_explanation):
        return "Yes"
    if _funding_detected(event_type, funding_amount, funding_stage):
        return "Yes"
    blob = f"{raw_title} {full_explanation}"
    if _RE_ABOUT_TO_MAKE.search(blob):
        return "Potential"
    if _RE_JUST_MADE_MONEY.search(blob) and not _RE_DOLLAR_OR_DEAL.search(blob):
        return "Potential"
    if "compensation" in blob.lower() or "pay package" in blob.lower():
        return "Potential"
    return "No"


def classify_wealth_signal_strength(
    *,
    raw_title: str,
    full_explanation: str,
    event_type: str,
    role: str,
    estimated_wealth: float,
    aggregated_estimated_wealth: float,
    is_billionaire: bool,
    linked_wealth_signal: bool,
    funding_amount: str,
    funding_stage: str,
    weak_signal: bool,
) -> str:
    blob = f"{raw_title} {full_explanation}"
    ew = float(estimated_wealth or 0)
    agg = float(aggregated_estimated_wealth or 0)

    if weak_signal and ew < 1 and agg < 1 and not is_billionaire:
        return "Weak" if _finance_career_broad(raw_title) else "None"

    if is_billionaire or agg >= 10_000_000 or ew >= 5_000_000:
        return "Strong"
    if (
        _deal_detected(event_type, raw_title, full_explanation)
        and (_RE_DOLLAR_OR_DEAL.search(blob) or event_type == "Founder Exit")
    ) or (linked_wealth_signal and _funding_detected(event_type, funding_amount, funding_stage)):
        return "Strong"
    if _funding_detected(event_type, funding_amount, funding_stage) and _RE_DOLLAR_OR_DEAL.search(blob):
        return "Strong"

    if (
        event_type in ("Founder Exit", "Funding")
        or _deal_detected(event_type, raw_title, full_explanation)
        or _funding_detected(event_type, funding_amount, funding_stage)
        or (ew >= 500_000 or agg >= 1_000_000)
    ):
        return "Moderate"

    if (
        event_type in ("Promotion", "Board Appointment")
        or _executive_role(role)
        or _RE_HAS_MONEY_CONTEXT.search(blob)
    ):
        return "Weak"

    if event_type == "Other" and _finance_career_broad(blob):
        return "Weak"

    return "None"


def _finance_career_broad(title_or_blob: str) -> bool:
    """Lightweight check aligned with data.py ingest (avoid importing data)."""
    t = (title_or_blob or "").lower()
    needles = (
        "ceo", "cfo", "founder", "funding", "million", "billion", "acquisition", "ipo",
        "investor", "venture", "board", "compensation", "equity", "valuation",
    )
    return any(n in t for n in needles)


def classify_client_type(
    role: str,
    event_type: str,
    raw_title: str,
) -> str:
    blob = f"{role} {raw_title}".lower()
    if any(x in blob for x in ("heir", "family office", "inherit", "trust fund", "estate of")):
        return "Heir / Family wealth"
    if any(
        x in blob
        for x in (
            "venture capitalist",
            " vc ",
            " pe ",
            "private equity",
            "hedge fund",
            "portfolio manager",
            "investor at",
        )
    ):
        return "Investor (PE/VC/HF)"
    if any(x in blob for x in ("nfl", "nba", "mlb", "olymp", "athlete", "actor", "celebrity", "musician")):
        return "Athlete / Celebrity"
    if "founder" in blob or "co-founder" in blob or "entrepreneur" in blob or event_type == "Funding":
        return "Founder / Entrepreneur"
    if any(x in blob for x in ("ceo", "cfo", "coo", "cto", "chief", "president", "chairman", "executive")):
        return "Executive"
    if event_type == "Board Appointment":
        return "Executive"
    return "Executive" if _executive_role(role) else "Founder / Entrepreneur"


def infer_source_of_wealth(
    event_type: str,
    raw_title: str,
    full_explanation: str,
) -> str:
    blob = f"{raw_title} {full_explanation}".lower()
    if event_type == "Founder Exit" or "acqui" in blob or "acquired" in blob or "sale" in blob:
        return "Company sale / M&A"
    if "ipo" in blob or "going public" in blob or "listing" in blob:
        return "IPO / public listing"
    if event_type == "Funding" or "raised" in blob or "round" in blob:
        return "Funding / equity round"
    if "compensation" in blob or "bonus" in blob or "salary" in blob:
        return "Compensation / bonus"
    if "real estate" in blob or "property" in blob or "mansion" in blob:
        return "Real estate"
    if "estate" in blob or "inherit" in blob or "heir" in blob:
        return "Inheritance / estate"
    if "contract" in blob and ("sport" in blob or "team" in blob):
        return "Contract / entertainment deal"
    if "equity" in blob or "stake" in blob or "shares" in blob:
        return "Equity / stake"
    return ""


def wealth_signal_rank(strength: str) -> int:
    """Lower is stronger (for sorting)."""
    s = (strength or "None").strip()
    return {"Strong": 0, "Moderate": 1, "Weak": 2, "None": 3}.get(s, 3)


def derive_wealth_priority_level(
    *,
    score: int,
    passes_gate: bool,
    strength: str,
    liquidity: str,
    macro_noise: bool,
    weak_signal: bool,
    is_billionaire: bool,
) -> str:
    """
    High / Medium / Low from wealth rules (not raw score alone).

    HIGH only when ``passes_gate`` and strength/liquidity justify it.
    """
    st = (strength or "None").strip()
    if st not in _WEALTH_SIGNAL_LABELS:
        st = "None"
    liq = (liquidity or "No").strip()
    if liq not in _LIQUIDITY_LABELS:
        liq = "No"

    if macro_noise and st in ("None", "Weak") and not is_billionaire:
        return "Low"
    if weak_signal and st == "None" and not passes_gate and not is_billionaire:
        return "Low"

    if is_billionaire and st != "None":
        return "High" if passes_gate or liq != "No" else "Medium"

    if not passes_gate:
        if st == "Moderate":
            return "Medium"
        if st == "Weak" and int(score) >= 68:
            return "Medium"
        return "Low"

    if st == "Strong":
        return "High"
    if st == "Moderate":
        return "High" if liq in ("Yes", "Potential") else "Medium"
    if st == "Weak":
        return "Medium" if liq != "No" else "Low"
    return "Low"
