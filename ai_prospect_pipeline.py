"""
AI-first prospect extraction: article signal gate, structured LLM candidates,
primary-actor selection, multi-source cross-check, and match scoring (0–40).

Designed so the LLM extractor can later be swapped for batch/async workers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

_CACHE_ROOT = Path(__file__).resolve().parent / ".cache" / "wealth_pipeline"

# -----------------------------------------------------------------------------
# Money helpers (aligned with hybrid_pipeline)
# -----------------------------------------------------------------------------
_MONEY_ALL = re.compile(
    r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|bn|m\b|b\b|k\b)?",
    re.I,
)


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


def _largest_money_usd(text: str) -> float:
    best_v = 0.0
    for m in _MONEY_ALL.finditer(text or ""):
        v = _usd_value(m.group(1), m.group(2))
        if v > best_v:
            best_v = v
    return best_v


# =============================================================================
# 1) Article-level signal (rule-based, lightweight)
# =============================================================================
def score_article_signal(
    article_text: str,
    source_title: str = "",
    published_at: Any = None,
) -> dict[str, Any]:
    """
    Returns:
      signal_score: 0–60
      signal_type: coarse label
      economic_relevance: bool (deal / money / business story)
    """
    t = f"{source_title or ''} {article_text or ''}".lower()
    reasons: list[str] = []
    pts = 0
    stype = "Other"

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
        pts += 30
        stype = "Funding"
        reasons.append("+30 funding")

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
        pts += 30
        if stype == "Other":
            stype = "M&A"
        reasons.append("+30 m&a")

    if any(k in t for k in ("revenue", "growth", "profit", "valuation", "earnings")):
        pts += 20
        if stype == "Other":
            stype = "Revenue"
        reasons.append("+20 revenue/growth")

    max_usd = _largest_money_usd(t)
    if max_usd >= 1e9:
        pts += 30
        reasons.append("+30 $1B+")
    elif max_usd >= 1e8:
        pts += 20
        reasons.append("+20 $100M+")
    elif max_usd >= 1e6:
        pts += 10
        reasons.append("+10 $1M+")

    # Freshness
    try:
        dt = pd.Timestamp(published_at)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        from datetime import timezone

        age_h = (pd.Timestamp.now(tz=timezone.utc) - dt).total_seconds() / 3600
        if age_h <= 72:
            pts += 8
            reasons.append("+8 fresh")
    except Exception:
        if any(k in t for k in ("today", "hours ago", "just announced", "breaking")):
            pts += 8
            reasons.append("+8 recency_keywords")

    capped = min(60, pts)
    economic_relevance = bool(
        capped >= 15
        or max_usd >= 1e6
        or any(
            k in t
            for k in (
                "billion",
                "million",
                "funding",
                "acquisition",
                "merger",
                "ipo",
                "valuation",
                "investment",
                "startup",
                "founder",
                "ceo",
            )
        )
    )

    return {
        "signal_score": capped,
        "signal_type": stype,
        "economic_relevance": economic_relevance,
        "_debug_reasons": reasons,
        "_raw_pts_before_cap": pts,
    }


# =============================================================================
# 2) AI extraction — structured JSON (OpenAI)
# =============================================================================
_CANDIDATE_SCHEMA: dict[str, Any] = {
    "name": "wealth_article_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "article_topic": {"type": "string"},
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                        "company": {"type": "string"},
                        "context_type": {
                            "type": "string",
                            "enum": ["primary", "secondary", "mention"],
                        },
                        "economic_role": {
                            "type": "string",
                            "enum": [
                                "founder",
                                "ceo",
                                "owner",
                                "partner",
                                "investor",
                                "executive",
                                "commentator",
                                "lawyer",
                                "politician",
                                "other",
                            ],
                        },
                        "wealth_relevance": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "article_relevance_reason": {"type": "string"},
                        "is_real_person": {"type": "boolean"},
                    },
                    "required": [
                        "name",
                        "role",
                        "company",
                        "context_type",
                        "economic_role",
                        "wealth_relevance",
                        "article_relevance_reason",
                        "is_real_person",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["article_topic", "candidates"],
        "additionalProperties": False,
    },
}


def _openai_extract_candidates(article_text: str, source_title: str) -> dict[str, Any] | None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    client = OpenAI(api_key=api_key)
    body = (article_text or "")[:24000]
    title = (source_title or "")[:500]
    prompt = f"""You extract structured data from a news article for wealth / business development intelligence.

Article title: {title}

Article body:
---
{body}
---

Task:
- List ALL named people in the article.
- For each: name, role, company ONLY when explicitly supported — never invent a company.
- context_type: primary / secondary / mention (donor lists, quotes = mention).
- economic_role: founder, ceo, owner, partner, investor, executive, commentator, lawyer, politician, other.
- wealth_relevance: high = clear wealth creation, liquidity, ownership, major funding; medium = some business relevance; low = commentary, politics, legal counsel, list-only.
- article_relevance_reason: one short phrase explaining classification.
- is_real_person: false for regions, agencies, companies-as-names, "Middle East", "Technology Business".

Return strict JSON only (schema enforced)."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_schema", "json_schema": _CANDIDATE_SCHEMA},
        )
    except Exception:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
        except Exception:
            return None

    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if "candidates" not in data:
        return None
    # Normalize v1 cache → v2 fields
    for c in data.get("candidates") or []:
        if isinstance(c, dict):
            if "wealth_relevance" not in c:
                c["wealth_relevance"] = "medium"
            if "article_relevance_reason" not in c:
                c["article_relevance_reason"] = str(c.get("reason") or "")
    return data


def extract_candidates_with_ai(article_text: str, source_title: str) -> dict[str, Any] | None:
    """Structured candidates + article_topic. None if API missing or failure."""
    return _openai_extract_candidates(article_text, source_title)


def _cache_key_article(source_url: str, source_title: str) -> str:
    base = f"{source_url or ''}|{source_title or ''}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:28]


_AI_MEM: dict[str, dict[str, Any]] = {}


def _ai_disk_path(key: str) -> Path:
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return _CACHE_ROOT / f"ai_extract_v2_{key}.json"


def extract_candidates_with_ai_cached(
    article_text: str,
    source_title: str,
    *,
    source_url: str = "",
) -> dict[str, Any] | None:
    key = _cache_key_article(source_url, source_title)
    if key in _AI_MEM:
        return _AI_MEM[key]
    p = _ai_disk_path(key)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "candidates" in data:
                _AI_MEM[key] = data
                return data
        except (json.JSONDecodeError, OSError):
            pass
    out = extract_candidates_with_ai(article_text, source_title)
    if out is not None:
        _AI_MEM[key] = out
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=0)
        except OSError:
            pass
    return out


def heuristic_candidates_from_row(row: dict[str, Any], summary: str) -> dict[str, Any]:
    """Fallback when AI unavailable; mark as low-trust."""
    name = str(row.get("person_name") or row.get("name") or "").strip()
    if not name:
        return {"article_topic": "", "candidates": [], "_heuristic": True}
    return {
        "article_topic": "",
        "candidates": [
            {
                "name": name,
                "role": str(row.get("role") or "").strip(),
                "company": str(row.get("company_name") or row.get("company") or "").strip(),
                "context_type": "primary",
                "economic_role": "other",
                "wealth_relevance": "low",
                "article_relevance_reason": "heuristic_fallback_no_ai",
                "is_real_person": True,
            }
        ],
        "_heuristic": True,
    }


# =============================================================================
# 3) Primary actor selection
# =============================================================================
_GOOD_ROLES = frozenset({"founder", "ceo", "owner", "partner", "investor"})
_BAD_ROLES = frozenset({"commentator", "lawyer", "politician"})


def _actor_selection_score(c: dict[str, Any]) -> float:
    ct = str(c.get("context_type") or "mention").lower()
    er = str(c.get("economic_role") or "other").lower()
    s = 0.0
    if ct == "primary":
        s += 100
    elif ct == "secondary":
        s += 50
    else:
        s += 10
    if er in _GOOD_ROLES:
        s += 80
    elif er == "executive":
        s += 40
    elif er in _BAD_ROLES:
        s -= 120
    elif er == "other":
        s += 5
    wr = str(c.get("wealth_relevance") or "medium").lower()
    if wr == "high":
        s += 25
    elif wr == "medium":
        s += 8
    else:
        s -= 15
    if not c.get("is_real_person", True):
        s -= 500
    return s


def select_primary_actor(
    ai_candidates: list[dict[str, Any]],
    article_signal: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    """
    Pick the best primary-like candidate. Returns (candidate, weak_primary).
    weak_primary=True → do not label High/Elite automatically (cap downstream).
    """
    del article_signal  # reserved for future use (e.g. signal strength weighting)
    people = [c for c in ai_candidates if isinstance(c, dict) and str(c.get("name") or "").strip()]
    if not people:
        return None, True
    scored = sorted(people, key=_actor_selection_score, reverse=True)
    best = scored[0]
    best_score = _actor_selection_score(best)
    weak = best_score < 80 or str(best.get("context_type")) == "mention"
    er = str(best.get("economic_role") or "").lower()
    if er in _BAD_ROLES and str(best.get("context_type")) != "primary":
        weak = True
    if str(best.get("context_type")) == "mention" and er not in _GOOD_ROLES:
        weak = True
    return best, weak


# =============================================================================
# 4–5) Cross-check identity + wealth lists
# =============================================================================
def cross_check_wealth_lists(name: str, company: str) -> dict[str, Any]:
    """Forbes-style lists: prominence, est wealth hint, modest confidence bump."""
    out: dict[str, Any] = {
        "prominence_tag": "",
        "est_wealth_from_list": "",
        "list_match": False,
        "identity_confidence_bump": 0.0,
    }
    try:
        from wealth_identity import build_identity_db
    except ImportError:
        return out

    db = build_identity_db()

    def _key(n: str) -> str:
        s = (n or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s*&\s*family\s*$", "", s, flags=re.I)
        return s.strip()

    k = _key(name)
    if k not in db:
        return out
    rec = db[k]
    out["list_match"] = True
    out["est_wealth_from_list"] = str(rec.get("net_worth") or "").strip()
    tag = rec.get("tag")
    if tag == "30_under_30":
        out["prominence_tag"] = "30_under_30"
    else:
        out["prominence_tag"] = "forbes_billionaires"
    out["identity_confidence_bump"] = 0.08
    co = str(rec.get("company") or "").strip()
    if co and (not company or SequenceMatcher(None, company.lower(), co.lower()).ratio() < 0.3):
        out["canonical_company_hint"] = co
    return out


def cross_check_identity(name: str, role: str, company: str) -> dict[str, Any]:
    """
    Merge Wikipedia / OpenCorporates / profile APIs via hybrid enrich_entity,
    then add agreement / mismatch signals for scoring.
    """
    from hybrid_pipeline import enrich_entity

    entity = {"name": name, "role": role, "company": company}
    en = enrich_entity(entity)
    wl = cross_check_wealth_lists(name, company)

    sources = list(en.get("sources") or [])
    wiki = bool(en.get("wikipedia_url"))
    oc = "opencorporates" in sources
    pdl = any(s for s in sources if s not in ("wikipedia", "opencorporates"))

    canon_co = str(en.get("company") or "").strip()
    art_co = str(company or "").strip()
    mismatch = False
    if art_co and canon_co and SequenceMatcher(None, art_co.lower(), canon_co.lower()).ratio() < 0.35:
        mismatch = True

    agree_sources = sum([wiki, oc, pdl])
    if wl.get("list_match"):
        agree_sources += 1

    conf = 0.35
    if wiki:
        conf += 0.2
    if oc or pdl:
        conf += 0.15
    if agree_sources >= 2 and not mismatch:
        conf += 0.15
    if mismatch:
        conf -= 0.25
    conf = max(0.0, min(1.0, conf + float(wl.get("identity_confidence_bump") or 0)))
    if wl.get("canonical_company_hint") and not mismatch:
        canon_co = wl["canonical_company_hint"] or canon_co

    est = str(en.get("est_net_worth") or "").strip()
    if wl.get("est_wealth_from_list"):
        est = wl["est_wealth_from_list"]

    return {
        "canonical_name": str(en.get("canonical_name") or name).strip(),
        "canonical_role": str(en.get("role") or role).strip(),
        "canonical_company": canon_co or art_co,
        "industry": str(en.get("industry") or "").strip(),
        "est_wealth": est,
        "verification_sources_used": sources + (["forbes_list"] if wl.get("list_match") else []),
        "identity_confidence": conf,
        "company_mismatch": mismatch,
        "agree_sources": agree_sources,
        "prominence_tag": wl.get("prominence_tag") or "",
        "_enrichment": en,
        "_wealth_list": wl,
    }


def cross_check_identity_and_wealth(
    name: str,
    company: str,
    role: str,
    *,
    article_summary: str = "",
) -> dict[str, Any]:
    """
    Multi-source identity + wealth hints (extends ``cross_check_identity``).
    ``wealth_evidence``: direct | indirect | none. Never invents net worth.
    """
    base = cross_check_identity(name, role, company)
    notes: list[str] = [str(x) for x in (base.get("verification_sources_used") or [])]
    wl = base.get("_wealth_list") or {}

    wealth_evidence = "none"
    nw = base.get("est_wealth") or None
    nw_s = str(nw or "")
    if wl.get("list_match") or (nw_s and "$" in nw_s):
        wealth_evidence = "direct"
        notes.append("list_or_explicit_net_worth")
    elif base.get("prominence_tag") == "forbes_billionaires":
        wealth_evidence = "direct"
    elif any(
        k in (article_summary or "").lower()
        for k in ("raised", "series", "valuation", "funding", "ipo", "acquisition")
    ) and str(base.get("canonical_role", "")).lower() in ("founder", "ceo", "chief executive", "ceo"):
        wealth_evidence = "indirect"

    sow = ""
    if wl.get("list_match"):
        sow = "Forbes-style list / public prominence"
    elif wealth_evidence == "indirect":
        sow = "Funding / growth / executive role (inference)"

    vc = int(base.get("agree_sources") or 0)

    return {
        **base,
        "wealth_evidence": wealth_evidence,
        "net_worth": nw if wealth_evidence == "direct" else None,
        "source_of_wealth_hint": sow,
        "verification_count": vc,
        "notes": notes[:12],
    }


def compute_wealth_status(
    cross: dict[str, Any],
    article_summary: str,
    economic_role: str,
) -> str:
    """verified_wealth | likely_wealth | emerging_founder | unclear"""
    we = str(cross.get("wealth_evidence") or "none")
    er = (economic_role or "").lower()
    t = (article_summary or "").lower()
    if we == "direct" or cross.get("_wealth_list", {}).get("list_match"):
        return "verified_wealth"
    if we == "indirect" and er in ("founder", "ceo", "owner"):
        return "likely_wealth"
    if any(k in t for k in ("raised", "series", "funding", "valuation", "billion", "million")) and er in (
        "founder",
        "ceo",
        "owner",
        "investor",
    ):
        return "likely_wealth"
    if er == "founder" or "founder" in str(cross.get("canonical_role", "")).lower():
        return "emerging_founder"
    return "unclear"


# =============================================================================
# 6) Match score 0–40
# =============================================================================
_BAD_COMPANY_FRAGMENTS = frozenset(
    {
        "middle east",
        "technology business",
        "european union",
        "white house",
        "the region",
    }
)


def compute_match_score(
    candidate: dict[str, Any],
    cross_check_result: dict[str, Any],
) -> tuple[int, list[str]]:
    reasons: list[str] = []
    s = 0

    ct = str(candidate.get("context_type") or "mention").lower()
    if ct == "primary":
        s += 18
        reasons.append("+18 primary")
    elif ct == "secondary":
        s += 8
        reasons.append("+8 secondary")
    else:
        reasons.append("+0 mention")

    er = str(candidate.get("economic_role") or "other").lower()
    if er in ("founder",):
        s += 12
        reasons.append("+12 founder")
    elif er in ("ceo", "owner"):
        s += 10
        reasons.append("+10 ceo/owner")
    elif er in ("partner", "investor"):
        s += 8
        reasons.append("+8 partner/investor")
    elif er == "executive":
        s += 4
        reasons.append("+4 executive")
    elif er == "commentator":
        s -= 10
        reasons.append("-10 commentator")
    elif er == "lawyer":
        rl = str(candidate.get("role") or "").lower()
        if any(x in rl for x in ("founder", "owner", "co-founder")):
            s += 4
            reasons.append("+4 lawyer+founder context")
        else:
            s -= 12
            reasons.append("-12 lawyer")
    elif er == "politician":
        s -= 12
        reasons.append("-12 politician")

    wr = str(candidate.get("wealth_relevance") or "medium").lower()
    if wr == "high":
        s += 4
        reasons.append("+4 wealth_relevance_high")
    elif wr == "low":
        s -= 5
        reasons.append("-5 wealth_relevance_low")

    agree = int(cross_check_result.get("agree_sources") or 0)
    mismatch = bool(cross_check_result.get("company_mismatch"))
    conf = float(cross_check_result.get("identity_confidence") or 0)

    if agree >= 2 and not mismatch:
        s += 10
        reasons.append("+10 verified (2+ sources)")
    elif agree >= 1 and not mismatch:
        s += 5
        reasons.append("+5 partial verify")
    if mismatch:
        s -= 10
        reasons.append("-10 company mismatch")
    if conf < 0.35 and agree == 0:
        s -= 8
        reasons.append("-8 low identity confidence")
    if mismatch and conf < 0.5:
        s -= 5
        reasons.append("-5 likely wrong identity")

    co = str(cross_check_result.get("canonical_company") or candidate.get("company") or "").strip()
    col = co.lower()
    vu_raw = cross_check_result.get("verification_sources_used") or []
    vu = [str(x).lower() for x in vu_raw] if isinstance(vu_raw, list) else []
    if not co or col in ("unknown", "data pending"):
        s -= 8
        reasons.append("-8 unknown company")
    elif any(b in col for b in _BAD_COMPANY_FRAGMENTS):
        s -= 10
        reasons.append("-10 bad company fragment")
    elif co and not any(b in col for b in _BAD_COMPANY_FRAGMENTS):
        if "wikipedia" in vu or "peopledatalabs" in str(vu) or "fullcontact" in str(vu):
            s += 5
            reasons.append("+5 real company / profile path")

    # Modest list boost (does not override weak article context)
    if cross_check_result.get("_wealth_list", {}).get("list_match"):
        s += 3
        reasons.append("+3 list prominence (modest)")

    s = max(0, min(40, s))
    return s, reasons


def build_processed_row_core(
    *,
    name: str,
    role: str,
    company: str,
    signal_type: str,
    signal_score: int,
    match_score: int,
    priority_label: str,
    est_wealth: str,
    source_title: str,
    source_url: str,
    summary: str,
    context_type: str,
    economic_role: str,
    identity_confidence: float,
    verification_sources_used: list[str],
    priority_score: int | None = None,
) -> dict[str, Any]:
    if priority_score is None:
        ps = max(0, min(100, int(signal_score) + int(match_score)))
    else:
        ps = max(0, min(100, int(priority_score)))
    return {
        "name": name,
        "role": role,
        "company": company,
        "signal_type": signal_type,
        "signal_score": int(signal_score),
        "match_score": int(match_score),
        "priority_score": ps,
        "priority_label": priority_label,
        "est_wealth": est_wealth,
        "source_title": source_title,
        "source_url": source_url,
        "summary": summary,
        "context_type": context_type,
        "economic_role": economic_role,
        "identity_confidence": identity_confidence,
        "verification_sources_used": verification_sources_used,
    }


def priority_label_from_priority_score(priority_score: int) -> str:
    if priority_score >= 90:
        return "Elite"
    if priority_score >= 75:
        return "High"
    if priority_score >= 55:
        return "Medium"
    return "Low"
