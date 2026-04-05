"""
AI-first quality pipeline: primary-actor extraction, hard name filters, enrichment consistency,
wealth signal (0–10), combined confidence, and priority_score = wealth*6 + confidence*40 (0–100).

Heuristic extractors are bypassed when ``WEALTH_QUALITY_PIPELINE`` is enabled (default: on with OpenAI).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# --- Config -----------------------------------------------------------------

def quality_pipeline_enabled() -> bool:
    if os.environ.get("OPENAI_API_KEY", "").strip() == "":
        return False
    return os.environ.get("WEALTH_QUALITY_PIPELINE", "1").lower() not in ("0", "false", "no")


def quality_pipeline_strict() -> bool:
    """When True, failed quality extraction does not fall back to legacy multi-candidate path."""
    return os.environ.get("WEALTH_QUALITY_PIPELINE_STRICT", "1").lower() not in ("0", "false", "no")


# Exact / substring blocks on normalized name (Step 2 + Step 8)
_NAME_SUBSTRING_REJECT = frozenset(
    {
        "list of",
        "list ",
        " case",
        "case ",
        "department",
        "region",
        "middle east",
        "technology business",
        "list of punjabi",
    }
)

_NAME_EXACT_BLACKLIST = frozenset(
    {
        "middle east",
        "technology business",
        "list of punjabi people",
    }
)

_BAD_COMPANY_SINGLE_TOKENS = frozenset({"santa", "unknown", "inc", "llc", "corp"})


def _openai_json(prompt: str, *, temperature: float = 0.12) -> dict[str, Any] | None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
    except Exception:
        return None
    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def hard_filter_extracted_name(name: str) -> bool:
    """
    Hard filter: 2–3 word human-style names; reject org-like / list / region strings.
    """
    n = re.sub(r"\s+", " ", (name or "").strip())
    if not n:
        return False
    low = n.lower()
    if low in _NAME_EXACT_BLACKLIST:
        return False
    for ph in _NAME_SUBSTRING_REJECT:
        if ph in low:
            return False
    parts = n.split()
    if len(parts) < 2 or len(parts) > 3:
        return False
    if any(p.lower() in ("list", "case", "department", "region") for p in parts):
        return False
    return True


def extract_primary_business_person_llm(article_text: str) -> dict[str, Any] | None:
    """
    Step 1 — single structured extraction. Returns dict with name, role, company,
    is_primary_actor, context, extraction_confidence (0–1), or None on API failure.
    """
    t = (article_text or "").strip()
    if len(t) < 40:
        return None
    body = t[:24000]
    prompt = f"""Extract the primary business-relevant person from this article.

Return a single JSON object with exactly these keys:
- "name": string — full name of one real individual (empty string if none)
- "role": string — their role in the story
- "company": string — primary company tied to them (empty if unclear)
- "is_primary_actor": boolean — true only if this person is the main subject of the article for a business/wealth-relevant event (not a passing quote or list mention)
- "context": string — one short sentence describing why they matter here
- "extraction_confidence": number from 0 to 1 — your confidence in name/role/company correctness

Ignore and do NOT output as the primary person:
- historical figures unless the article is clearly about them in a current business context
- random mentions, journalists, lawyers in passing
- organizations or regions presented as if they were people
- fictional or non-business entities

Article:
{body}
"""
    data = _openai_json(prompt, temperature=0.1)
    if not data:
        return None
    try:
        data["is_primary_actor"] = bool(data.get("is_primary_actor"))
    except (TypeError, ValueError):
        data["is_primary_actor"] = False
    ec = data.get("extraction_confidence")
    try:
        data["extraction_confidence"] = max(0.0, min(1.0, float(ec)))
    except (TypeError, ValueError):
        data["extraction_confidence"] = 0.5
    return data


def enrich_entity_consistency_llm(
    *,
    name: str,
    company: str,
    role: str,
    article_text: str,
    wiki_snippet: str = "",
) -> dict[str, Any]:
    """
    Step 3 — lightweight cross-check (simulates search consistency). Returns:
    consistency (0–1), role_matches_article (bool), company_plausible (bool), passed (bool).
    """
    a = (article_text or "")[:12000]
    w = (wiki_snippet or "")[:2000]
    prompt = f"""You verify whether an extracted person/role/company is consistent with the article and optional Wikipedia intro.

Person: {name}
Role: {role}
Company: {company}

Article excerpt:
{a}

Wikipedia search snippet (may be empty or about a different person):
{w}

Return JSON only:
- "consistency": number 0-1 (how well name+role+company fit the article; penalize wrong company or wrong role)
- "role_matches_article": boolean
- "company_plausible": boolean — true if the company string is a real full company name or empty/unknown is justified; false for obvious fragments like a single generic word
- "notes": short string

Do not invent facts beyond the text."""
    data = _openai_json(prompt, temperature=0.08)
    if not data:
        return {
            "consistency": 0.35,
            "role_matches_article": False,
            "company_plausible": bool((company or "").strip()),
            "passed": False,
            "notes": "enrichment_api_fail",
        }
    try:
        c = max(0.0, min(1.0, float(data.get("consistency", 0))))
    except (TypeError, ValueError):
        c = 0.35
    rm = bool(data.get("role_matches_article"))
    cp = bool(data.get("company_plausible", True))
    passed = c >= 0.45 and rm
    return {
        "consistency": c,
        "role_matches_article": rm,
        "company_plausible": cp,
        "passed": passed,
        "notes": str(data.get("notes") or "")[:500],
    }


def wealth_signal_ai_0_10(article_text: str, name: str, company: str) -> int:
    """Step 4 — wealth relevance 0–10 from AI (not keyword rules)."""
    t = (article_text or "")[:14000]
    prompt = f"""Does this article indicate the person has significant or increasing personal or economic stake (wealth-relevant), not merely being mentioned?

Person: {name}
Company: {company}

Score 0-10 integer based on:
- ownership, equity, funding rounds tied to them
- revenue scale / business value creation
- assets, transactions, liquidity events
- NOT score high for mere name-drops or political commentary without economic stake

Article:
{t}

Return JSON: {{"wealth_signal_0_10": <0-10 integer>, "reason": "<short>"}}"""
    data = _openai_json(prompt, temperature=0.1)
    if not data:
        return 0
    try:
        v = int(data.get("wealth_signal_0_10", data.get("score", 0)))
    except (TypeError, ValueError):
        v = 0
    return max(0, min(10, v))


def article_clarity_0_1(article_text: str) -> float:
    """How clear/useful the article text is for extraction (0–1)."""
    t = (article_text or "").strip()
    if len(t) < 80:
        return 0.35
    prompt = f"""Rate how clear and concrete this news text is for identifying one business person and a wealth-related event (0=garbled/opinion only, 1=clear facts).

Return JSON: {{"clarity": <0-1 number>}}

Text:
{t[:8000]}"""
    data = _openai_json(prompt, temperature=0.05)
    if not data:
        return 0.5
    try:
        return max(0.0, min(1.0, float(data.get("clarity", 0.5))))
    except (TypeError, ValueError):
        return 0.5


def compute_combined_confidence(
    extraction_confidence: float,
    enrichment_consistency: float,
    article_clarity: float,
) -> float:
    """Step 5 — average of three components (0–1)."""
    a = max(0.0, min(1.0, float(extraction_confidence)))
    b = max(0.0, min(1.0, float(enrichment_consistency)))
    c = max(0.0, min(1.0, float(article_clarity)))
    return (a + b + c) / 3.0


def compute_priority_score_v1(wealth_signal_0_10: int, pipeline_confidence_0_1: float) -> int:
    """
    Step 6 — priority 0–100: wealth_signal * 6 + confidence * 40 (confidence on 0–1 scale).
    """
    w = max(0, min(10, int(wealth_signal_0_10)))
    p = max(0.0, min(1.0, float(pipeline_confidence_0_1)))
    raw = w * 6.0 + p * 40.0
    return max(0, min(100, int(round(raw))))


def _company_is_real(company: str) -> bool:
    c = re.sub(r"\s+", " ", (company or "").strip())
    if len(c) < 2:
        return False
    low = c.lower()
    if low in ("unknown", "n/a", "none"):
        return False
    if low in _BAD_COMPANY_SINGLE_TOKENS and " " not in c:
        return False
    return True


def eligible_for_home_top_v1(
    pipeline_confidence: float,
    company: str,
    enrichment: dict[str, Any],
) -> bool:
    """Step 7 — Top 5 cards: high confidence, real company, enrichment not failed."""
    if pipeline_confidence < 0.7:
        return False
    if not _company_is_real(company):
        return False
    if not enrichment.get("passed", False) and float(enrichment.get("consistency", 0)) < 0.55:
        return False
    return True


def run_quality_pipeline_v1(
    *,
    summary: str,
    source_title: str,
    source_url: str,
    published_at: Any,
    row: dict[str, Any],
    na: dict[str, Any],
    sig: dict[str, Any],
) -> tuple[list[dict[str, Any]], str] | None:
    """
    Build 0–1 prospect rows using quality v1 pipeline. Returns None to fall back to legacy path.

    Second return value is reason code: "ok", "disabled", "fallback", "empty".
    """
    if not quality_pipeline_enabled():
        return None

    ext = extract_primary_business_person_llm(summary)
    if not ext:
        if quality_pipeline_strict():
            return [], "empty"
        return None

    if not ext.get("is_primary_actor"):
        return [], "empty"

    name = str(ext.get("name") or "").strip()
    role = str(ext.get("role") or "").strip()
    company = str(ext.get("company") or "").strip()

    if not name or not hard_filter_extracted_name(name):
        return [], "empty"

    # Lazy imports — avoid circular import at module load
    from person_validation import enrich_with_search
    from prospect_hardening import (
        coerce_display_person_name,
        is_historical_or_dead,
        is_valid_person_name,
        sanitize_role_and_company,
    )
    from ai_prospect_pipeline import (
        build_processed_row_core,
        classify_wealth_status,
        compute_match_score,
        cross_check_identity_and_wealth,
        infer_ownership_strength,
        priority_label_from_priority_score,
        score_founder_wealth_creation,
        score_private_company_context,
    )
    from prospect_resolution import derive_wealth_fields, infer_signal_type
    from prospect_tier import apply_tier_priority_adjustment, classify_prospect_tier
    from prospect_display_gates import is_commentary_only
    from two_pass_pipeline import compute_recency_score, pass1_recency_adjustment
    from settings import SHOW_DEBUG

    wiki = enrich_with_search(name)
    wiki_snip = str(wiki.get("snippet") or wiki.get("extract") or "")[:1500]

    dead_hist = is_historical_or_dead(name, summary, {"_wikipedia_extract": wiki_snip})
    if dead_hist:
        return [], "empty"

    enrich = enrich_entity_consistency_llm(
        name=name,
        company=company,
        role=role,
        article_text=summary,
        wiki_snippet=wiki_snip,
    )

    if not _company_is_real(company):
        enrich = {**enrich, "consistency": min(float(enrich.get("consistency", 0.5)), 0.45)}

    w10 = wealth_signal_ai_0_10(summary, name, company)
    clarity = article_clarity_0_1(summary)
    ext_c = float(ext.get("extraction_confidence") or 0.65)
    conf = compute_combined_confidence(ext_c, float(enrich.get("consistency") or 0), clarity)

    prio = compute_priority_score_v1(w10, conf)
    eligible_home = eligible_for_home_top_v1(conf, company, enrich)

    c = {
        "name": name,
        "role": role,
        "company": company,
        "entity_type": "person",
        "context_type": "primary",
        "economic_role": "founder" if re.search(r"founder", role, re.I) else "other",
        "is_valid_prospect_person": True,
        "is_real_person": True,
        "wealth_relevance": "high" if w10 >= 6 else "medium",
    }

    cc = cross_check_identity_and_wealth(
        name,
        company,
        role,
        article_summary=summary,
        context_type="primary",
        economic_role=str(c.get("economic_role") or "other"),
    )
    role_san, co_san, _ = sanitize_role_and_company(c, summary, cc)
    cc = {**cc, "canonical_company": co_san, "canonical_role": role_san}

    msc, m_r = compute_match_score(c, cc, summary)
    fwc = score_founder_wealth_creation(summary, c, cc)
    priv = score_private_company_context(summary, c, cc)
    founder_wealth_score = min(40, int(fwc.get("subscore") or 0) + int(priv.get("subscore") or 0))
    own_inf = infer_ownership_strength(summary, c, cc)
    commentary_only_row = is_commentary_only(c, summary)

    wstat_pre = classify_wealth_status(summary, c, cc)
    dw = derive_wealth_fields(c, na, cc, wealth_status_hint=wstat_pre)
    wstat = str(dw.get("wealth_status") or wstat_pre)
    est_display = str(dw.get("est_wealth") or "Data pending")

    prospect_tier = classify_prospect_tier(
        c,
        {
            **cc,
            "_tier_article_summary": summary,
            "_tier_wealth_status": wstat,
            "_tier_founder_wealth_score": founder_wealth_score,
        },
    )
    prio2 = apply_tier_priority_adjustment(prio, prospect_tier)
    prio2 = max(0, min(100, int(prio2)))

    label = priority_label_from_priority_score(prio2)
    if conf < 0.6:
        label = "Low" if prio2 < 55 else label

    row_signal_type = infer_signal_type(na, c, str(sig.get("signal_type") or "Other"))
    if row_signal_type == "Other" and founder_wealth_score >= 12:
        row_signal_type = "Founder Wealth Creation"

    display_name = coerce_display_person_name(name, str(cc.get("canonical_name") or ""), summary)
    if not display_name.strip() or not is_valid_person_name(display_name, summary):
        return [], "empty"

    core = build_processed_row_core(
        name=display_name.strip(),
        role=str(cc.get("canonical_role") or role_san).strip(),
        company=str(cc.get("canonical_company") or co_san).strip(),
        signal_type=row_signal_type,
        signal_score=int(sig.get("signal_score") or 0),
        match_score=msc,
        priority_label=label,
        est_wealth=est_display,
        source_title=source_title,
        source_url=source_url,
        summary=summary,
        context_type="primary",
        economic_role=str(c.get("economic_role") or "other"),
        identity_confidence=float(conf),
        verification_sources_used=list(cc.get("verification_sources_used") or []),
        priority_score=prio2,
    )

    published_at = na.get("published_at") or row.get("detected_at") or row.get("event_date")
    pr_adj = pass1_recency_adjustment(published_at)
    # Recency nudge small vs quality score — keep subordinate to v1 priority
    prio_final = max(0, min(100, prio2 + max(-5, min(5, pr_adj // 3))))

    legacy = {
        **row,
        **core,
        "person_name": core["name"],
        "company_name": core["company"],
        "raw_title": source_title,
        "score": prio_final,
        "priority_level": core["priority_label"],
        "priority_score": prio_final,
        "est_wealth_display": est_display,
        "wealth_confidence": int(round(conf * 100)),
        "confidence_score": int(round(conf * 100)),
        "wealth_numeric_verified": bool(dw.get("wealth_numeric_verified")),
        "signal_type": core["signal_type"],
        "source_title": source_title,
        "source_url": source_url,
        "published_at": published_at,
        "recency_score": compute_recency_score(published_at),
        "wealth_status": wstat,
        "article_economic_relevance": bool(sig.get("economic_relevance")),
        "candidate_historical_dead": False,
        "founder_wealth_score": founder_wealth_score,
        "ownership_inference": own_inf,
        "wealth_evidence": str(cc.get("wealth_evidence") or "none"),
        "prospect_tier": prospect_tier,
        "entity_type": "person",
        "commentary_only_row": commentary_only_row,
        "normalized_article": na,
        "quality_pipeline_v1": True,
        "pipeline_confidence": float(conf),
        "wealth_signal_0_10": int(w10),
        "enrichment_consistency": float(enrich.get("consistency") or 0),
        "enrichment_passed": bool(enrich.get("passed")),
        "eligible_for_home_top": eligible_home,
        "primary_actor_context": str(ext.get("context") or "")[:500],
        "client_enrichment": enrich,
    }
    if SHOW_DEBUG:
        legacy["debug_match_reasons"] = "; ".join(str(x) for x in (m_r or []))
        legacy["_debug_quality_v1"] = json.dumps(
            {"enrich_notes": enrich.get("notes"), "wiki_found": bool(wiki.get("found"))}
        )

    return [legacy], "ok"
