"""
Match quality × signal quality ranking for prospect rows (broad recall, score separation).

Output fields: name, role, company, signal_type, signal_score, match_score,
priority_score, priority_label, est_wealth, source_title, source_url, summary,
plus optional debug_signal_reasons / debug_match_reasons (not shown in UI by default).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd

# --- Money parsing (values in dollars for comparison) ---
_MONEY_ALL = re.compile(
    r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|bn|m\b|b\b|k\b)?",
    re.I,
)

_APPOSITIVE_CO = re.compile(
    r"(?:founder|co-founder|ceo|c\.?e\.?o\.?|owner|president|chair(?:man)?)\s+(?:of|at)\s+([^.,;\n]+?)(?:\s*[,.;]|\s+who|\s+said|\s+announced|$)",
    re.I,
)
_AT_COMPANY = re.compile(r"\bat\s+([A-Z][A-Za-z0-9&\-\s]{2,60}?)(?:\s*[,.;]|\s+and|\s+who|\s+said|$)", re.I)

_BAD_NAME_TERMS = (
    "department",
    "agency",
    "licensing",
    "ministry",
    "committee",
    "region",
    "middle east",
    "technology business",
    "gangmasters",
)

_JOURNALIST_TERMS = (
    "attorney",
    "lawyer",
    "journalist",
    "reporter",
    "spokesperson",
    "spokeswoman",
    "spokesman",
)

_GENERIC_COMPANY = re.compile(
    r"^(the\s+)?(white house|fed|federal reserve|congress|senate|middle east|europe|asia|government)\b",
    re.I,
)


def _text_blob(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("raw_title") or ""),
        str(row.get("full_explanation") or ""),
        str(row.get("ai_summary") or ""),
        str(row.get("summary") or ""),
    ]
    return " ".join(p for p in parts if str(p).strip()).strip()


def _usd_value(raw_amt: str, unit: str | None) -> float:
    try:
        val = float(str(raw_amt).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0
    u = (unit or "").strip().lower()
    if u in ("billion", "bn", "b"):
        return val * 1e9
    if u in ("million", "m"):
        return val * 1e6
    if u == "k":
        return val * 1e3
    if val >= 1e6:
        return val
    return val


def _largest_money_usd(text: str) -> tuple[float, str | None]:
    """Return (usd_value, display_string) for largest parsed amount."""
    best_v = 0.0
    best_s: str | None = None
    for m in _MONEY_ALL.finditer(text):
        v = _usd_value(m.group(1), m.group(2))
        if v > best_v:
            best_v = v
            amt = m.group(1).replace(",", "")
            u = (m.group(2) or "").lower().strip()
            if u in ("billion", "bn", "b") or (not u and best_v >= 1e9):
                best_s = f"${float(amt):,.1f}B".replace(",", "") if "." in amt else f"${amt}B"
            elif u in ("million", "m") or best_v >= 1e6:
                best_s = f"${float(amt.replace(',', '')):,.0f}M"
            else:
                best_s = f"${amt}"
    return best_v, best_s


def score_article_signal(text: str, detected_at: Any) -> tuple[int, list[str]]:
    """Signal strength 0–60 (article/event), capped at 60."""
    t = text.lower()
    reasons: list[str] = []
    pts = 0

    if any(
        k in t
        for k in (
            "raised",
            "raising",
            "funding",
            "series a",
            "series b",
            "series c",
            "series d",
            "venture",
            "round",
            "investment",
        )
    ):
        pts += 25
        reasons.append("+25 fundraising")

    if any(
        k in t
        for k in (
            "acquisition",
            "merger",
            "acquire",
            "acquired",
            "sale of",
            "stake sale",
            "ipo",
            "go public",
            "buyout",
            "exit",
        )
    ):
        pts += 25
        reasons.append("+25 m&a/exit/ipo")

    if any(k in t for k in ("revenue", "growth", "profit", "valuation", "earnings")):
        pts += 20
        reasons.append("+20 revenue/growth/valuation")

    max_usd, _ = _largest_money_usd(text)
    if max_usd >= 1e9:
        pts += 30
        reasons.append("+30 $1B+")
    elif max_usd >= 1e8:
        pts += 20
        reasons.append("+20 $100M+")

    if any(k in t for k in ("today", "hours ago", "just announced", "breaking")):
        pts += 10
        reasons.append("+10 recency_keywords")
    else:
        try:
            dt = pd.Timestamp(detected_at)
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
            age_h = (pd.Timestamp.now(tz=timezone.utc) - dt).total_seconds() / 3600
            if age_h <= 48:
                pts += 10
                reasons.append("+10 fresh_rss")
        except Exception:
            pass

    capped = min(60, pts)
    if capped < pts:
        reasons.append(f"cap 60 (was {pts})")
    return capped, reasons


def _resolve_company(row: dict[str, Any], summary: str) -> tuple[str, bool]:
    """
    Return (company, looks_garbage).
    Chain: structured field → appositive → 'at X' → Unknown.
    """
    co = str(row.get("company_name") or row.get("company") or "").strip()
    if co and co.lower() not in ("unknown", "data pending", "not identified", ""):
        if _GENERIC_COMPANY.search(co):
            return co, True
        return co, False

    s = summary
    m = _APPOSITIVE_CO.search(s)
    if m:
        cand = m.group(1).strip()
        if len(cand) > 2 and not _GENERIC_COMPANY.search(cand):
            return cand.strip(), False

    m2 = _AT_COMPANY.search(s)
    if m2:
        cand = m2.group(1).strip()
        if len(cand) > 2:
            return cand, False

    return "Unknown", False


def _signal_type_from_row(row: dict[str, Any], summary: str) -> str:
    et = str(row.get("event_type") or "").strip()
    if et and et != "Other":
        return et
    sl = str(row.get("wealth_signal_label") or "").strip()
    if sl:
        return sl
    t = summary.lower()
    if any(k in t for k in ("acquisition", "merger", "buyout")):
        return "M&A"
    if any(k in t for k in ("funding", "raised", "series")):
        return "Funding"
    return "Other"


def estimate_wealth(summary: str, role: str) -> str:
    max_usd, _ = _largest_money_usd(summary)
    if max_usd >= 1e9:
        return f"${max_usd / 1e9:.1f}B"
    if max_usd >= 1e6:
        return f"${max_usd / 1e6:.0f}M"

    rl = role.lower()
    if "founder" in rl or "ceo" in rl or "owner" in rl:
        return "$10M–$100M"
    if "partner" in rl or "managing director" in rl or "director" in rl:
        return "$1M–$10M"
    return "Data pending"


def score_match_quality(
    name: str,
    role: str,
    company: str,
    summary: str,
    company_garbage: bool,
) -> tuple[int, list[str]]:
    """Match quality 0–40, clamped."""
    reasons: list[str] = []
    score = 0
    n = name.lower().strip()
    rl = role.lower()
    t = summary.lower()
    toks = name.split()

    for term in _BAD_NAME_TERMS:
        if term in n:
            score -= 40
            reasons.append(f"-40 bad_name:{term}")
            break

    if len(toks) < 2:
        score -= 25
        reasons.append("-25 single_token_name")

    for bad in ("bot", "named after", "ai bot", "fictional"):
        if bad in t:
            score -= 35
            reasons.append(f"-35 fictional/bot:{bad}")
            break

    for bad in ("died", "historian", "former secretary", "appointed by"):
        if bad in t:
            score -= 35
            reasons.append(f"-35 historical/dead:{bad}")
            break

    if "james schlesinger" in n or ("schlesinger" in n and "died" in t):
        score -= 30
        reasons.append("-30 historical_figure_context")

    if "gaskell" in n or "elizabeth gaskell" in n:
        score -= 25
        reasons.append("-25 literary/historical_name")

    # Role quality
    if "founder" in rl or "co-founder" in rl:
        score += 25
        reasons.append("+25 founder")
    elif "ceo" in rl or "owner" in rl:
        score += 22
        reasons.append("+22 ceo/owner")
    elif "managing partner" in rl or "managing director" in rl or "partner" in rl:
        score += 18
        reasons.append("+18 partner/md")
    elif "director" in rl or "vp" in rl or "vice president" in rl:
        score += 12
        reasons.append("+12 director/vp")
    else:
        hit_j = any(j in rl for j in _JOURNALIST_TERMS)
        liquidity = any(k in t for k in ("ipo", "acquisition", "sale", "stake", "founder", "raised", "exit"))
        if hit_j and not liquidity:
            score -= 15
            reasons.append("-15 journalist/legal_no_liquidity")

    if not company or company.lower() == "unknown":
        score -= 15
        reasons.append("-15 no_company")
    else:
        score += 10
        reasons.append("+10 company_present")
        if company_garbage or _GENERIC_COMPANY.search(company):
            score -= 10
            reasons.append("-10 generic_company")

    # Context fit (heuristic)
    quote_weak = False
    if toks:
        quote_weak = bool(
            re.search(r"\bsaid\b|\baccording to\b|\btold\s+(?:reporters|the)\b", t)
            and re.search(rf"{re.escape(toks[0])}.*\bsaid\b", t, re.I)
        )
    if quote_weak and not any(k in rl for k in ("founder", "ceo", "owner", "chair")):
        score -= 20
        reasons.append("-20 quote_commentary")

    if any(k in t for k in ("historian wrote", "novelist", "19th century", "victorian")):
        score -= 25
        reasons.append("-25 historical_tangential")

    main_event = any(
        k in t
        for k in (
            "founder",
            "co-founder",
            "chief executive",
            "ceo",
            "acquisition",
            "merger",
            "raised",
            "funding",
            "ipo",
        )
    )
    if main_event and any(k in rl for k in ("founder", "ceo", "owner", "president", "chair")):
        score += 15
        reasons.append("+15 context_main_event")

    raw_m = score
    out = max(0, min(40, score))
    if out != raw_m:
        reasons.append(f"clamp_0_40 (raw {raw_m})")
    return out, reasons


def priority_label_from_score(priority_score: int) -> str:
    if priority_score >= 90:
        return "Elite"
    if priority_score >= 75:
        return "High"
    if priority_score >= 55:
        return "Medium"
    return "Low"


def _process_one_row(row: dict[str, Any]) -> dict[str, Any]:
    name = str(row.get("person_name") or row.get("name") or "").strip()
    role = str(row.get("role") or "").strip()
    source_title = str(row.get("raw_title") or row.get("source_title") or "").strip()
    source_url = str(row.get("source_url") or "").strip()
    summary = _text_blob(row)
    detected_at = row.get("detected_at")

    company, co_garbage = _resolve_company(row, summary)
    signal_type = _signal_type_from_row(row, summary)

    sig, sig_r = score_article_signal(summary, detected_at)
    msc, m_r = score_match_quality(name, role, company, summary, co_garbage)
    priority_score = max(0, min(100, sig + msc))
    plab = priority_label_from_score(priority_score)
    ew = estimate_wealth(summary, role)

    return {
        "name": name,
        "role": role,
        "company": company,
        "signal_type": signal_type,
        "signal_score": sig,
        "match_score": msc,
        "priority_score": priority_score,
        "priority_label": plab,
        "est_wealth": ew,
        "source_title": source_title,
        "source_url": source_url,
        "summary": summary,
        "debug_signal_reasons": "; ".join(sig_r),
        "debug_match_reasons": "; ".join(m_r),
    }


def process_and_rank_prospects(raw_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Rank all rows by priority_score descending. Merges processor fields onto each row
    and syncs person_name, raw_title, company_name, score, priority_level for legacy UI.
    """
    if raw_rows is None or raw_rows.empty:
        return raw_rows

    out_rows: list[dict[str, Any]] = []
    for _, row in raw_rows.iterrows():
        base = row.to_dict()
        proc = _process_one_row(base)
        merged = {**base, **proc}
        merged["person_name"] = proc["name"]
        merged["company_name"] = proc["company"]
        merged["raw_title"] = proc["source_title"]
        merged["score"] = int(proc["priority_score"])
        merged["priority_level"] = proc["priority_label"]
        merged["est_wealth_display"] = proc["est_wealth"]
        out_rows.append(merged)

    out = pd.DataFrame(out_rows)
    out = out.sort_values(by="priority_score", ascending=False, na_position="last").reset_index(drop=True)
    return out
