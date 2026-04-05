"""
Hard gating for financial-advisor prospect priority (not general-news importance).

Priority is driven by identifiable prospects, wealth/liquidity substance, and FA relevance —
not outlet prestige or serious-sounding tone.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from person_validation import is_valid_person

# --- Negative / generic story shapes (suppress unless tied to a named wealthy prospect) ---

_RE_NEGATIVE_GENERIC = re.compile(
    r"\b("
    r"triple\s*lock|state\s+pension|pension\s+(?:rules|age|credit)|pensions?\s+(?:bill|reform)|"
    r"explainer|what\s+(?:is|are|does)|how\s+to\s+(?:understand|claim)|"
    r"consumer\s+(?:price|prices|goods)|price\s+hike|price\s+rise|"
    r"ps5|playstation\s*5|xbox\s+series|nintendo\s+switch\s+2|"
    r"social\s+media\s+(?:trial|ban|law)|trial\s+(?:begins|opens|verdict)|class\s+action\s+trend|"
    r"uk\s+(?:budget|government|westminster)|whitehall|parliament\s+debate|"
    r"product\s+(?:recall|update|launch)\s+(?:for|of)\s+(?:the\s+)?(?:ps|iphone)|"
    r"shipping\s+delay|black\s+friday\s+deal|"
    r"macro\s+(?:outlook|forecast)|gdp\s+(?:print|data)|inflation\s+print(?!\s+for\s+\w+\s+\w+)|"
    r"election\s+poll|campaign\s+trail(?!\s+.*\b(?:ceo|founder|chair))\b"
    r")\b",
    re.I,
)

# Title-focused only — “analysis” in body text is too common on serious business pieces.
_RE_NEGATIVE_OPINION_ANALYSIS = re.compile(
    r"\b("
    r"opinion|editorial|explainer|overview|five\s+things|what\s+we\s+learned|the\s+big\s+picture"
    r")\b",
    re.I,
)

_RE_NAMED_EXEC_IN_TITLE = re.compile(
    r"^[^(]+\s+[-–—]\s*(?:CEO|CFO|COO|CTO|CIO|founder|co-founder|chair|president)\b",
    re.I,
)

_RE_STRONG_DEAL = re.compile(
    r"\b("
    r"acqui(?:red|res|tion)|merger|buyout|ipo\b|listing|raised\s+\$|series\s+[a-e]\b|"
    r"seed\s+round|unicorn|valuation|exit|sold\s+(?:for|to|stake)|billionaire|millionaire|"
    r"net\s+worth|estate\s+of|inherit|bequest"
    r")\b",
    re.I,
)


def _parse_additional_people(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if str(x).strip()]
            except json.JSONDecodeError:
                pass
        return [s]
    return []


def prospect_identified_from_row(
    person_name: str,
    additional_people: Any,
    raw_title: str,
) -> tuple[bool, str]:
    """
    True if we have a plausible named individual (or small set) as a prospect anchor.

    Returns (identified, display_name).
    """
    pn = str(person_name or "").strip()
    if pn and is_valid_person(pn):
        return True, pn
    for x in _parse_additional_people(additional_people):
        if x and is_valid_person(x):
            return True, x
    # Title pattern: "Jane Doe - CEO ..." (common in press headlines)
    t = str(raw_title or "").strip()
    if t and _RE_NAMED_EXEC_IN_TITLE.search(t):
        lead = t.split(" - ")[0].split(" – ")[0].split(" — ")[0].strip()
        if lead and is_valid_person(lead):
            return True, lead
    return False, ""


def classify_suppression(
    raw_title: str,
    full_blob: str,
    prospect_identified: bool,
    is_billionaire: bool,
) -> tuple[str, str]:
    """
    none: no generic-news penalty
    soft: downrank FA relevance; never High unless UHNW rescue
    hard: cap priority at Low unless billionaire + named
    """
    blob = f"{raw_title} {full_blob}"
    reasons: list[str] = []

    if _RE_NEGATIVE_GENERIC.search(blob):
        reasons.append("generic_or_consumer_or_policy")

    if _RE_NEGATIVE_OPINION_ANALYSIS.search(raw_title):
        reasons.append("opinion_analysis_explainer")

    if not reasons:
        return "none", ""

    joined = "|".join(reasons)

    if is_billionaire and prospect_identified:
        return "soft", joined

    if prospect_identified and _RE_STRONG_DEAL.search(blob):
        return "soft", joined

    return "hard", joined


def wealth_liquidity_substance_ok(
    *,
    passes_wealth_gate: bool,
    strength: str,
    liquidity: str,
    is_billionaire: bool,
    estimated_wealth: float,
    aggregated_estimated_wealth: float,
) -> bool:
    """FA wealth/liquidity gate: at least one credible money / liquidity / WM hook."""
    st = (strength or "None").strip()
    liq = (liquidity or "No").strip()
    ew = float(estimated_wealth or 0)
    agg = float(aggregated_estimated_wealth or 0)

    if is_billionaire:
        return True
    if passes_wealth_gate:
        return True
    if st in ("Strong", "Moderate"):
        return True
    if liq in ("Yes", "Potential"):
        return True
    if ew >= 500_000 or agg >= 1_000_000:
        return True
    return False


def fa_worthiness_flags(
    *,
    prospect_identified: bool,
    passes_wealth_gate: bool,
    strength: str,
    liquidity: str,
    client_type: str,
    role: str,
    event_type: str,
    raw_title: str,
) -> tuple[bool, bool, bool, bool]:
    """
    Who is the prospect? Why wealthy/liquid? Targetable? Actionable recently?

    Used for FA relevance and debug logging (not outlet prestige).
    """
    st = (strength or "None").strip()
    liq = (liquidity or "No").strip()
    ct = str(client_type or "").strip().lower()
    et = str(event_type or "").strip()
    title_l = str(raw_title or "").lower()
    rl = str(role or "").lower()

    who_ok = prospect_identified

    why_money_ok = passes_wealth_gate or st in ("Strong", "Moderate", "Weak")

    targetable_ok = prospect_identified and (
        ct not in ("", "unknown")
        or any(
            k in rl
            for k in (
                "ceo",
                "cfo",
                "coo",
                "cto",
                "founder",
                "chair",
                "president",
                "chief",
                "partner",
                "investor",
            )
        )
        or et in ("Founder Exit", "Funding", "Promotion", "Board Appointment")
    )

    actionable_ok = (
        liq in ("Yes", "Potential")
        or et in ("Founder Exit", "Funding")
        or any(
            k in title_l
            for k in (
                "acqui",
                "merger",
                "ipo",
                "raised",
                "funding",
                "appointed",
                "named ceo",
                "promoted",
                "stepping down",
                "exit",
            )
        )
        or passes_wealth_gate
    )

    return who_ok, why_money_ok, targetable_ok, actionable_ok


def infer_fa_relevance(
    *,
    suppression: str,
    who_ok: bool,
    why_money_ok: bool,
    targetable_ok: bool,
    actionable_ok: bool,
    wealth_substance_ok: bool,
    macro_noise: bool,
) -> str:
    if suppression == "hard":
        return "Low"
    if macro_noise and not why_money_ok:
        return "Low"
    if not wealth_substance_ok:
        return "Low"
    if all((who_ok, why_money_ok, targetable_ok, actionable_ok)):
        return "High" if suppression == "none" else "Medium"
    if who_ok and why_money_ok and (targetable_ok or actionable_ok):
        return "Medium"
    if who_ok:
        return "Medium"
    return "Low"


def derive_fa_priority_level(
    *,
    score: int,
    prospect_identified: bool,
    strength: str,
    liquidity: str,
    passes_wealth_gate: bool,
    macro_noise: bool,
    weak_signal: bool,
    is_billionaire: bool,
    suppression: str,
    fa_relevance: str,
    wealth_substance_ok: bool,
    estimated_wealth: float,
    aggregated_estimated_wealth: float,
) -> str:
    """
    Hard rules:
    - High only with named prospect, Strong/Moderate signal, substance, FA High relevance, no hard suppression.
    - Medium: named person + plausible relevance, not hard-suppressed dead stories.
    - Low: everything else (including no person, no wealth hook, generic news).
    """
    st = (strength or "None").strip()
    liq = (liquidity or "No").strip()
    ew = float(estimated_wealth or 0)
    agg = float(aggregated_estimated_wealth or 0)

    # Absolute caps
    if not prospect_identified:
        return "Low"
    if st == "None":
        return "Low"
    if not wealth_substance_ok:
        return "Low"
    if suppression == "hard" and not is_billionaire:
        return "Low"
    if fa_relevance == "Low":
        return "Low"

    liquidity_or_uhnw_ok = (
        liq in ("Yes", "Potential")
        or (is_billionaire and st in ("Strong", "Moderate"))
        or ew >= 2_000_000
        or agg >= 5_000_000
        or (passes_wealth_gate and st in ("Strong", "Moderate") and liq != "No")
    )

    high_bar = (
        st in ("Strong", "Moderate")
        and fa_relevance == "High"
        and suppression == "none"
        and not macro_noise
        and not weak_signal
        and liquidity_or_uhnw_ok
        and passes_wealth_gate
    )

    if high_bar:
        return "High"

    # Medium: named + some substance; allow soft suppression or weaker signals
    if prospect_identified and st in ("Strong", "Moderate", "Weak") and wealth_substance_ok:
        if suppression == "hard" and not is_billionaire:
            return "Low"
        if fa_relevance in ("High", "Medium"):
            return "Medium"
        if st == "Weak" and liq != "No" and int(score) >= 55:
            return "Medium"
        if st in ("Strong", "Moderate") and fa_relevance == "Low" and liq in ("Yes", "Potential"):
            return "Medium"

    return "Low"


def build_fa_why_sentence(
    *,
    prospect_identified: bool,
    structured_name: str,
    strength: str,
    liquidity: str,
    passes_gate: bool,
    suppression: str,
) -> str:
    if not prospect_identified:
        return "No identifiable individual to prospect; treat as background news."
    nm = structured_name or "Named prospect"
    parts = [f"{nm} appears in a wealth-relevant story."]
    if passes_gate:
        parts.append("Money, deal, or liquidity context is present.")
    else:
        parts.append("Limited explicit money or deal hook in the text.")
    parts.append(f"Wealth signal: {strength}; liquidity: {liquidity}.")
    if suppression != "none":
        parts.append("Headline shape looks like generic news or policy/consumer coverage—verify before outreach.")
    return " ".join(parts)


def format_priority_debug(
    *,
    gate_person: bool,
    gate_wealth_substance: bool,
    gate_liquidity_bar: bool,
    gate_fa_relevance: bool,
    suppression: str,
    fa_relevance: str,
    final: str,
) -> str:
    return (
        f"person={int(gate_person)};wealth_substance={int(gate_wealth_substance)};"
        f"liquidity_or_uhnw={int(gate_liquidity_bar)};fa_relevance={fa_relevance}({int(gate_fa_relevance)});"
        f"suppression={suppression};priority={final}"
    )


@dataclass
class FAGatingResult:
    fa_prospect_identified: str
    fa_structured_name: str
    fa_relevance: str
    fa_why_one_sentence: str
    fa_pass_gate_person: bool
    fa_pass_gate_wealth_substance: bool
    fa_pass_gate_liquidity_or_uhnw: bool
    fa_pass_gate_fa_relevance: bool
    suppression_level: str
    suppression_reason: str
    priority_level: str
    fa_priority_debug: str


def compute_fa_gating_row(
    *,
    raw_title: str,
    full_explanation: str,
    person_name: str,
    additional_people: Any,
    event_type: str,
    role: str,
    client_type: str,
    strength: str,
    liquidity: str,
    passes_wealth_gate: bool,
    macro_noise: bool,
    weak_signal: bool,
    is_billionaire: bool,
    score: int,
    estimated_wealth: float,
    aggregated_estimated_wealth: float,
) -> FAGatingResult:
    blob = f"{raw_title} {full_explanation}".strip()

    pid, sname = prospect_identified_from_row(person_name, additional_people, raw_title)
    sup_lvl, sup_reason = classify_suppression(raw_title, blob, pid, is_billionaire)

    w_sub = wealth_liquidity_substance_ok(
        passes_wealth_gate=passes_wealth_gate,
        strength=strength,
        liquidity=liquidity,
        is_billionaire=is_billionaire,
        estimated_wealth=estimated_wealth,
        aggregated_estimated_wealth=aggregated_estimated_wealth,
    )

    who_ok, why_ok, tgt_ok, act_ok = fa_worthiness_flags(
        prospect_identified=pid,
        passes_wealth_gate=passes_wealth_gate,
        strength=strength,
        liquidity=liquidity,
        client_type=client_type,
        role=role,
        event_type=event_type,
        raw_title=raw_title,
    )

    fa_rel = infer_fa_relevance(
        suppression=sup_lvl,
        who_ok=who_ok,
        why_money_ok=why_ok,
        targetable_ok=tgt_ok,
        actionable_ok=act_ok,
        wealth_substance_ok=w_sub,
        macro_noise=macro_noise,
    )

    st = (strength or "None").strip()
    liq = (liquidity or "No").strip()
    ew = float(estimated_wealth or 0)
    agg = float(aggregated_estimated_wealth or 0)
    liq_bar = (
        liq in ("Yes", "Potential")
        or (is_billionaire and st in ("Strong", "Moderate"))
        or ew >= 2_000_000
        or agg >= 5_000_000
        or (passes_wealth_gate and st in ("Strong", "Moderate") and liq != "No")
    )

    gate_fa_rel = fa_rel == "High"

    pl = derive_fa_priority_level(
        score=score,
        prospect_identified=pid,
        strength=strength,
        liquidity=liquidity,
        passes_wealth_gate=passes_wealth_gate,
        macro_noise=macro_noise,
        weak_signal=weak_signal,
        is_billionaire=is_billionaire,
        suppression=sup_lvl,
        fa_relevance=fa_rel,
        wealth_substance_ok=w_sub,
        estimated_wealth=estimated_wealth,
        aggregated_estimated_wealth=aggregated_estimated_wealth,
    )

    why1 = build_fa_why_sentence(
        prospect_identified=pid,
        structured_name=sname,
        strength=strength,
        liquidity=liquidity,
        passes_gate=passes_wealth_gate,
        suppression=sup_lvl,
    )

    dbg = format_priority_debug(
        gate_person=pid,
        gate_wealth_substance=w_sub,
        gate_liquidity_bar=liq_bar,
        gate_fa_relevance=gate_fa_rel,
        suppression=sup_lvl,
        fa_relevance=fa_rel,
        final=pl,
    )

    return FAGatingResult(
        fa_prospect_identified="Yes" if pid else "No",
        fa_structured_name=sname or ("Not identified" if not pid else ""),
        fa_relevance=fa_rel,
        fa_why_one_sentence=why1,
        fa_pass_gate_person=pid,
        fa_pass_gate_wealth_substance=w_sub,
        fa_pass_gate_liquidity_or_uhnw=liq_bar,
        fa_pass_gate_fa_relevance=gate_fa_rel,
        suppression_level=sup_lvl,
        suppression_reason=sup_reason,
        priority_level=pl,
        fa_priority_debug=dbg,
    )
