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

from two_pass_pipeline import compute_recency_score

from prospect_display_gates import normalize_extracted_candidate

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
      signal_score: 0–60 (recency is separate: ``compute_recency_score`` / Pass 1 adjustment)
      signal_type: coarse label
      economic_relevance: bool — False mainly for lawsuit/regulation/politics/commentary
        without a clear business/liquidity/fundraising event
    """
    del published_at  # freshness not part of signal_score per spec
    t = f"{source_title or ''} {article_text or ''}"
    tl = t.lower()
    reasons: list[str] = []
    pts = 0
    stype = "Other"

    if any(
        k in tl
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
        k in tl
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

    if any(k in tl for k in ("revenue", "growth", "profit", "valuation", "earnings")):
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

    capped = min(60, pts)

    # Founder-led operational wealth (private scale, revenue explosions) — parity with funding/M&A as FA triggers
    founder_exec = any(
        k in tl
        for k in (
            "founder",
            "co-founder",
            "cofounder",
            "chief executive",
            "chief executive officer",
        )
    ) or re.search(r"\bceo\b", tl) is not None
    revenue_ops = any(
        k in tl for k in ("revenue", "sales", "profit", "grew", "growth", "scaling", "earnings")
    )
    if founder_exec and revenue_ops:
        if max_usd >= 1e9:
            pts = max(pts, 55)
            stype = "Founder Wealth Creation"
            reasons.append("+founder_wealth_$1B+_scale")
        elif max_usd >= 1e8:
            pts = max(pts, 45)
            if stype in ("Other", "Revenue"):
                stype = "Founder Wealth Creation"
            reasons.append("+founder_wealth_$100M+_scale")
        elif max_usd >= 1e7:
            pts = max(pts, 35)
            if stype == "Other":
                stype = "Founder Wealth Creation"
            reasons.append("+founder_wealth_strong_scale")
    capped = min(60, pts)

    # Mainly non-economic stories: lawsuits, pure legal procedure, regulation noise,
    # political analysis, commentary — unless a clear business / liquidity / funding event
    deal_or_liquidity = any(
        k in tl
        for k in (
            "raised",
            "funding",
            "series ",
            "valuation",
            "acquisition",
            "merger",
            "ipo",
            "stake",
            "buyout",
            "ownership",
            "liquidity",
            "business expansion",
            "investment",
            "founder",
            "co-founder",
            "chief executive",
            "ceo of",
        )
    ) or max_usd >= 1e6 or (
        founder_exec
        and revenue_ops
        and max_usd >= 1e7
    )

    mainly_non_economic = any(
        k in tl
        for k in (
            "lawsuit",
            "litigation",
            "plaintiff",
            "defendant",
            "legal procedure",
            "court filing",
            "regulatory fine",
            "political analysis",
            "opinion:",
            "commentary:",
            "editorial:",
            "election odds",
            "campaign ad",
        )
    ) or (
        "regulation" in tl
        and not any(x in tl for x in ("startup", "funding", "ipo", "merger", "acquisition"))
    )

    economic_relevance = bool(not (mainly_non_economic and not deal_or_liquidity))

    if not economic_relevance:
        capped = min(15, capped)
        reasons.append("cap15_non_economic")

    return {
        "signal_score": capped,
        "signal_type": stype,
        "economic_relevance": economic_relevance,
        "_debug_reasons": reasons,
        "_raw_pts_before_cap": pts,
    }


# =============================================================================
# Founder wealth creation (operational scale, private ownership) — FA triggers
# =============================================================================
def _actor_is_founder_operator(candidate: dict[str, Any], cross: dict[str, Any]) -> bool:
    er = str(candidate.get("economic_role") or "").lower()
    if er in ("founder", "ceo", "owner") or "founder" in er:
        return True
    rl = (
        str(candidate.get("role") or "")
        + " "
        + str(cross.get("canonical_role") or "")
    ).lower()
    return any(
        x in rl
        for x in (
            "founder",
            "co-founder",
            "cofounder",
            "chief executive",
            "ceo",
            "owner",
        )
    )


def score_founder_wealth_creation(
    article_text: str,
    extracted_actor: dict[str, Any],
    cross_check_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Subscore 0–40: founder/CEO/owner + large revenue/profit scale without requiring funding/M&A.
    """
    t = (article_text or "").lower()
    reasons: list[str] = []
    s = 0

    if not _actor_is_founder_operator(extracted_actor, cross_check_result):
        return {"subscore": 0, "reasons": ["not_founder_operator"], "raw_before_cap": 0}

    er = str(extracted_actor.get("economic_role") or "").lower()
    if er not in ("founder", "ceo", "owner", "executive", "investor", "other") and "founder" not in er:
        if not any(
            x
            in (
                str(extracted_actor.get("role") or "")
                + str(cross_check_result.get("canonical_role") or "")
            ).lower()
            for x in ("founder", "ceo", "owner", "chief executive")
        ):
            return {"subscore": 0, "reasons": ["role_not_founder_class"], "raw_before_cap": 0}

    if any(
        k in t
        for k in (
            "revenue",
            "sales",
            "profit",
            "profitable",
            "grew",
            "growth",
            "earnings",
            "valuation",
            "scale",
            "scaling",
        )
    ):
        s += 25
        reasons.append("+25 revenue_profit_growth")

    max_usd = _largest_money_usd(article_text or "")
    if max_usd >= 1e9:
        s += 30
        reasons.append("+30 $1B+_mentioned")
    elif max_usd >= 1e8:
        s += 20
        reasons.append("+20 $100M+_mentioned")

    conc = (
        "bootstrapped",
        "self-funded",
        "no outside funding",
        "without raising",
        "no venture",
        "concentrated",
        "controlling stake",
        "majority stake",
        "owns ",
        "ownership",
        "his company",
        "her company",
    )
    if any(k in t for k in conc):
        s += 15
        reasons.append("+15 ownership_clues")

    private_led = (
        "privately held",
        "private company",
        "private business",
        "founder-led",
        "family-owned",
        "no institutional",
        "never raised",
    )
    if any(k in t for k in private_led):
        s += 10
        reasons.append("+10 private_founder_led")

    raw = s
    return {"subscore": max(0, min(40, raw)), "reasons": reasons, "raw_before_cap": raw}


def score_private_company_context(
    article_text: str,
    candidate: dict[str, Any],
    cross: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """0–15: private / bootstrap / thin-team scale; folded into founder cap with score_founder_wealth_creation."""
    cr = cross if isinstance(cross, dict) else {}
    t = (article_text or "").lower()
    s = 0
    rs: list[str] = []
    if not _actor_is_founder_operator(candidate, cr):
        return {"subscore": 0, "reasons": []}

    if any(
        k in t
        for k in (
            "privately held",
            "private company",
            "private ",
            "not publicly traded",
            "bootstrapped",
            "self-funded",
        )
    ):
        s += 6
        rs.append("+6 private_bootstrap")

    if re.search(r"\b\d{1,3}[-–]\s*(person|people|employee|team members)\b", t) and (
        "revenue" in t or "sales" in t or _largest_money_usd(article_text or "") >= 1e7
    ):
        s += 5
        rs.append("+5 small_team_scale")

    if "founder" in t and "ceo" in t and str(candidate.get("context_type") or "") == "primary":
        s += 4
        rs.append("+4 primary_founder_framing")

    return {"subscore": min(15, s), "reasons": rs}


def infer_ownership_strength(
    article_text: str,
    candidate: dict[str, Any],
    cross_check_result: dict[str, Any],
) -> str:
    """high | medium | low — concentrated equity / founder control likelihood."""
    t = (article_text or "").lower()
    en = cross_check_result.get("_enrichment") or {}
    wiki = str(en.get("_wikipedia_extract") or "").lower()[:600] if isinstance(en, dict) else ""

    score = 0
    if any(
        k in t
        for k in (
            "bootstrapped",
            "self-funded",
            "no outside funding",
            "never raised",
            "without taking venture",
            "concentrated ownership",
            "majority stake",
            "controlling interest",
        )
    ):
        score += 3
    if any(k in t for k in ("founded", "started the company", "launched", "built the business")):
        score += 2
    if str(candidate.get("context_type") or "") == "primary":
        score += 2
    if "private" in t and "company" in t:
        score += 1
    if "ipo" in t or "went public" in t or "spac" in t:
        score -= 2
    if "privately held" in wiki or ("private" in wiki and "company" in wiki):
        score += 1

    if score >= 5:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def classify_wealth_status(
    article_text: str,
    candidate: dict[str, Any],
    cross_check_result: dict[str, Any],
) -> str:
    """verified_wealth | likely_wealth | emerging_founder | unclear — Medvi-style → likely_wealth."""
    we = str(cross_check_result.get("wealth_evidence") or "none")
    er = str(candidate.get("economic_role") or "").lower()
    t = (article_text or "").lower()
    fwc = int(score_founder_wealth_creation(article_text, candidate, cross_check_result).get("subscore") or 0)
    own = infer_ownership_strength(article_text, candidate, cross_check_result)
    big_money = _largest_money_usd(article_text or "") >= 1e8

    if we == "direct" or cross_check_result.get("_wealth_list", {}).get("list_match"):
        return "verified_wealth"

    strong_ops = fwc >= 20 or (
        big_money
        and any(k in t for k in ("revenue", "sales", "profit", "grew", "growth"))
        and _actor_is_founder_operator(candidate, cross_check_result)
    )

    if strong_ops and (
        er in ("founder", "ceo", "owner", "investor")
        or "founder" in er
        or _actor_is_founder_operator(candidate, cross_check_result)
    ):
        return "likely_wealth"

    if we == "indirect" and er in ("founder", "ceo", "owner"):
        return "likely_wealth"

    if any(
        k in t
        for k in (
            "raised",
            "series ",
            "funding",
            "valuation",
            "billion",
            "million",
            "revenue",
            "sales",
        )
    ) and er in ("founder", "ceo", "owner", "investor"):
        return "likely_wealth"

    if own == "high" and fwc >= 15 and er in ("founder", "ceo", "owner"):
        return "likely_wealth"

    if er == "founder" or "founder" in str(candidate.get("role") or "").lower():
        return "emerging_founder"

    return "unclear"


# =============================================================================
# 2) AI extraction — structured JSON (OpenAI)
# =============================================================================
_CANDIDATE_SCHEMA: dict[str, Any] = {
    "name": "wealth_article_v5",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "article_topic": {"type": "string"},
            "candidate_people": {"type": "array", "items": {"type": "string"}},
            "candidate_companies": {"type": "array", "items": {"type": "string"}},
            "candidate_roles": {"type": "array", "items": {"type": "string"}},
            "event_type": {"type": "string"},
            "money_mentions": {"type": "array", "items": {"type": "string"}},
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "entity_type": {
                            "type": "string",
                            "enum": [
                                "person",
                                "company",
                                "product",
                                "organization",
                                "region",
                                "event",
                                "unknown",
                            ],
                        },
                        "role": {"type": "string"},
                        "company": {"type": "string"},
                        "context_type": {
                            "type": "string",
                            "enum": [
                                "primary",
                                "secondary",
                                "mention",
                                "historical",
                                "commentary",
                            ],
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
                                "historical",
                                "other",
                            ],
                        },
                        "wealth_relevance": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "is_valid_prospect_person": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "name",
                        "entity_type",
                        "role",
                        "company",
                        "context_type",
                        "economic_role",
                        "wealth_relevance",
                        "is_valid_prospect_person",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "article_topic",
            "candidate_people",
            "candidate_companies",
            "candidate_roles",
            "event_type",
            "money_mentions",
            "candidates",
        ],
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

The input may include labeled sections: TITLE, SUMMARY, FIRST_PARAGRAPHS, ARTICLE, and MONEY_MENTIONS.
Use the full ARTICLE and FIRST_PARAGRAPHS for facts. MONEY_MENTIONS are contextual (deals/revenue) — not personal net worth.

Article title: {title}

Article body:
---
{body}
---

Also at the top level, fill:
- candidate_people: distinct person names (strings only).
- candidate_companies: organization names mentioned.
- candidate_roles: role phrases (CEO, founder, etc.) as they appear.
- event_type: one label for the main story (e.g. Funding, M&A, Executive change, Other).
- money_mentions: short strings for major $ figures (deal/funding/revenue — not personal NW).

Task:
- For each notable entity, set entity_type: person | company | product | organization | region | event | unknown.
- ONLY set is_valid_prospect_person=true for real individual humans you would cold-email as a prospect. Sororities/fraternities (e.g. Delta Delta Delta), car/product models (Tesla Model X), brands, governments, lists, cases, regions are NOT people — entity_type must reflect that and is_valid_prospect_person must be false.
- name + role + company: fill company ONLY when explicitly stated; never invent or guess a company.
- Do NOT infer founder exit, funding, or promotion unless the article clearly supports it (funding round, acquisition/sale/stake, appointment).
- context_type: primary / secondary / mention / historical / commentary — quoted pundits / "told CNBC" without operating relevance = commentary.
- economic_role: founder, ceo, owner, partner, investor, executive, commentator, lawyer, politician, historical, other.
- wealth_relevance: high = clear ownership/liquidity/funding; medium = business relevance; low = commentary/legal/macro.

Explicitly NEVER emit as a person (use entity_type=organization/product/region/event and is_valid_prospect_person=false):
- Fraternity/sorority triple names, vehicle trim levels, iPhone/Tesla Model, list pages, court cases, regions.

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
    # v5 extraction lists — backfill from candidates if older cache
    if not data.get("candidate_people"):
        data["candidate_people"] = [
            str(c.get("name") or "").strip()
            for c in (data.get("candidates") or [])
            if isinstance(c, dict)
            and str(c.get("entity_type") or "").lower() == "person"
            and c.get("is_valid_prospect_person")
        ]
    if not data.get("candidate_companies"):
        data["candidate_companies"] = list(
            dict.fromkeys(
                str(c.get("company") or "").strip()
                for c in (data.get("candidates") or [])
                if isinstance(c, dict) and str(c.get("company") or "").strip()
            )
        )[:24]
    if not data.get("candidate_roles"):
        data["candidate_roles"] = list(
            dict.fromkeys(
                str(c.get("role") or "").strip()
                for c in (data.get("candidates") or [])
                if isinstance(c, dict) and str(c.get("role") or "").strip()
            )
        )[:24]
    if not data.get("event_type"):
        data["event_type"] = "Other"
    if not data.get("money_mentions"):
        data["money_mentions"] = []
    # Normalize cache / older payloads → v4 fields
    out_c: list[dict[str, Any]] = []
    for c in data.get("candidates") or []:
        if isinstance(c, dict):
            if "wealth_relevance" not in c:
                c["wealth_relevance"] = "medium"
            if "reason" not in c:
                c["reason"] = str(c.get("article_relevance_reason") or c.get("reason") or "")
            c["article_relevance_reason"] = str(c.get("reason") or c.get("article_relevance_reason") or "")
            if "is_valid_prospect_person" not in c:
                c["is_valid_prospect_person"] = bool(c.get("is_real_person", True))
            if "entity_type" not in c:
                c["entity_type"] = "person" if c.get("is_valid_prospect_person") else "unknown"
            if "is_real_person" not in c:
                c["is_real_person"] = c.get("is_valid_prospect_person", True)
            out_c.append(normalize_extracted_candidate(c))
    data["candidates"] = out_c
    return data


def extract_candidates_with_ai(article_text: str, source_title: str) -> dict[str, Any] | None:
    """Structured candidates + article_topic. None if API missing or failure."""
    return _openai_extract_candidates(article_text, source_title)


def _blended_article_for_ai(normalized_article: dict[str, Any]) -> str:
    """Blend metadata + body for the LLM (not og:title alone)."""
    na = normalized_article or {}
    chunks: list[str] = []
    t = str(na.get("title") or "").strip()
    if t:
        chunks.append(f"TITLE: {t}")
    s = str(na.get("summary") or "").strip()
    if s:
        chunks.append(f"SUMMARY: {s}")
    fp = str(na.get("first_paragraphs") or "").strip()
    if fp:
        chunks.append(f"FIRST_PARAGRAPHS:\n{fp}")
    body = str(na.get("article_text") or "").strip()
    if body:
        chunks.append(f"ARTICLE:\n{body}")
    mm = na.get("money_mentions") or []
    if isinstance(mm, list) and mm:
        chunks.append(
            "MONEY_MENTIONS (deal/company context — not personal net worth): "
            + ", ".join(str(x) for x in mm[:25])
        )
    return "\n\n".join(chunks)[:28000]


def extract_prospect_candidates(normalized_article: dict[str, Any]) -> dict[str, Any] | None:
    """
    Main prospect extractor: structured AI over a **normalized** article (parse layer output).
    """
    text = _blended_article_for_ai(normalized_article)
    title = str(normalized_article.get("title") or normalized_article.get("og_title") or "").strip()
    url = str(
        normalized_article.get("canonical_url") or normalized_article.get("url") or ""
    ).strip()
    if not text.strip():
        return None
    return extract_candidates_with_ai_cached(text, title, source_url=url)


def _cache_key_article(source_url: str, source_title: str) -> str:
    base = f"{source_url or ''}|{source_title or ''}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:28]


_AI_MEM: dict[str, dict[str, Any]] = {}


def _ai_disk_path(key: str) -> Path:
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return _CACHE_ROOT / f"ai_extract_v5_{key}.json"


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
    from prospect_hardening import is_valid_person_name

    name = str(row.get("person_name") or row.get("name") or "").strip()
    if not name or not is_valid_person_name(name, summary):
        return {"article_topic": "", "candidates": [], "_heuristic": True}
    return {
        "article_topic": "",
        "candidates": [
            {
                "name": name,
                "entity_type": "person",
                "role": str(row.get("role") or "").strip(),
                "company": str(row.get("company_name") or row.get("company") or "").strip(),
                "context_type": "primary",
                "economic_role": "other",
                "wealth_relevance": "low",
                "reason": "heuristic_fallback_no_ai",
                "article_relevance_reason": "heuristic_fallback_no_ai",
                "is_valid_prospect_person": True,
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
    elif ct in ("historical", "commentary"):
        s -= 80
    else:
        s += 10
    if er in _GOOD_ROLES:
        s += 80
    elif er == "executive":
        s += 40
    elif er in _BAD_ROLES:
        s -= 120
    elif er == "historical":
        s -= 100
    elif er == "other":
        s += 5
    wr = str(c.get("wealth_relevance") or "medium").lower()
    if wr == "high":
        s += 25
    elif wr == "medium":
        s += 8
    else:
        s -= 15
    if not c.get("is_real_person", True) or not c.get("is_valid_prospect_person", True):
        s -= 500
    return s


def select_primary_actor(
    ai_candidates: list[dict[str, Any]],
    article_signal: dict[str, Any],
    *,
    article_text: str = "",
) -> tuple[dict[str, Any] | None, bool]:
    """
    Pick the best primary-like candidate. Returns (candidate, weak_primary).
    weak_primary=True → do not label High/Elite automatically (cap downstream).
    """
    del article_signal  # reserved for future use (e.g. signal strength weighting)
    from prospect_display_gates import is_forbidden_display_name
    from prospect_hardening import is_valid_person_name

    people = [
        c
        for c in ai_candidates
        if isinstance(c, dict)
        and str(c.get("name") or "").strip()
        and c.get("is_valid_prospect_person", c.get("is_real_person", True))
        and str(c.get("entity_type") or "person").lower() in ("person", "unknown")
        and not is_forbidden_display_name(str(c.get("name") or ""), article_text)
        and is_valid_person_name(str(c.get("name")), article_text)
    ]
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
    context_type: str = "",
    economic_role: str = "",
) -> dict[str, Any]:
    """
    Multi-source identity + wealth hints (extends ``cross_check_identity``).
    ``wealth_evidence``: direct | indirect | none. Never invents net worth.
    """
    base = cross_check_identity(name, role, company)
    notes: list[str] = [str(x) for x in (base.get("verification_sources_used") or [])]
    wl = base.get("_wealth_list") or {}
    en = base.get("_enrichment") or {}
    wiki_ex = str(en.get("_wikipedia_extract") or "") if isinstance(en, dict) else ""

    deceased = bool(en.get("wiki_bio_deceased")) if isinstance(en, dict) else False
    if not deceased and wiki_ex:
        wl_ex = wiki_ex.lower()[:2000]
        deceased = ("born" in wl_ex and "died" in wl_ex) or bool(
            re.search(r"\bdied\s+\d{4}\b", wl_ex)
        )

    historical_only = (
        str(context_type or "").lower() == "historical"
        or str(economic_role or "").lower() == "historical"
        or ("historian" in wiki_ex.lower()[:1200] if wiki_ex else False)
    )

    wealth_evidence = "none"
    nw = base.get("est_wealth") or None
    nw_s = str(nw or "")
    if wl.get("list_match") or (nw_s and "$" in nw_s):
        wealth_evidence = "direct"
        notes.append("list_or_explicit_net_worth")
    elif base.get("prominence_tag") == "forbes_billionaires":
        wealth_evidence = "direct"
    elif (
        _largest_money_usd(article_summary or "") >= 1e8
        and any(
            k in (article_summary or "").lower()
            for k in ("revenue", "sales", "profit", "grew", "growth", "scaling", "earnings")
        )
        and any(
            k in (article_summary or "").lower()
            for k in ("founder", "co-founder", "ceo", "chief executive", "owner")
        )
    ):
        wealth_evidence = "indirect"
        notes.append("operational_scale_founder")
    elif any(
        k in (article_summary or "").lower()
        for k in ("raised", "series", "valuation", "funding", "ipo", "acquisition")
    ) and str(base.get("canonical_role", "")).lower() in ("founder", "ceo", "chief executive", "ceo"):
        wealth_evidence = "indirect"

    sow = ""
    if wl.get("list_match"):
        sow = "Forbes-style list / public prominence"
    elif wealth_evidence == "indirect" and any("operational_scale_founder" in str(n) for n in notes):
        sow = "Large operational scale / founder-led business (inference)"
    elif wealth_evidence == "indirect":
        sow = "Funding / growth / executive role (inference)"

    vc = int(base.get("agree_sources") or 0)

    return {
        **base,
        "canonical_name": str(base.get("canonical_name") or name).strip(),
        "canonical_company": str(base.get("canonical_company") or company).strip(),
        "canonical_role": str(base.get("canonical_role") or role).strip(),
        "wealth_evidence": wealth_evidence,
        "net_worth": nw if wealth_evidence == "direct" else None,
        "source_of_wealth_hint": sow,
        "verification_count": vc,
        "deceased": deceased,
        "historical_only": historical_only,
        "notes": notes[:12],
    }


def compute_wealth_status(
    cross: dict[str, Any],
    article_summary: str,
    economic_role: str,
) -> str:
    """Backward-compatible wrapper; prefer :func:`classify_wealth_status` with full candidate."""
    stub: dict[str, Any] = {
        "economic_role": economic_role,
        "role": str(cross.get("canonical_role") or ""),
        "context_type": "primary",
    }
    return classify_wealth_status(article_summary, stub, cross)


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
    article_summary: str = "",
) -> tuple[int, list[str]]:
    reasons: list[str] = []
    s = 0
    art = (article_summary or "").lower()

    ct = str(candidate.get("context_type") or "mention").lower()
    if ct == "primary":
        s += 18
        reasons.append("+18 primary")
    elif ct == "secondary":
        s += 8
        reasons.append("+8 secondary")
    elif ct in ("historical", "commentary"):
        s -= 18
        reasons.append("-18 historical/commentary context")
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
    elif er == "historical":
        s -= 14
        reasons.append("-14 historical role")
    elif er == "commentator":
        s -= 10
        reasons.append("-10 commentator")
    elif er == "lawyer":
        rl = str(candidate.get("role") or "").lower()
        if any(x in rl for x in ("founder", "owner", "co-founder")) or any(
            x in art for x in ("founded", "co-founder", "owner of")
        ):
            s += 4
            reasons.append("+4 lawyer+ownership context")
        else:
            s -= 12
            reasons.append("-12 lawyer")
    elif er == "politician":
        if any(
            x in art
            for x in (
                "founder",
                "owner",
                "stake",
                "business",
                "company",
                "invested",
                "ipo",
            )
        ):
            s -= 4
            reasons.append("-4 politician+business")
        else:
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
        s -= 15
        reasons.append("-15 likely wrong identity")

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
