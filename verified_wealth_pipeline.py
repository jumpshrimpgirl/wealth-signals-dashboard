"""
Verified entity + wealth-event pipeline (accuracy > volume).

Steps: 0 relevance gate → 1 LLM extraction → 2 enrichment scores → 3 wealth/actionability
→ 4 rejection rules → 5 final score. Heuristic name extraction is not used on this path.

Configurable via env: see ``PipelineConfig`` below.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Config (tune without code changes)
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Thresholds and weights — override with env vars where noted."""

    # Step 5 — overall_confidence = w_ext * extraction + w_id * identity + w_art * article_clarity
    weight_extraction_in_overall: float = 0.25
    weight_identity_in_overall: float = 0.50
    weight_article_in_overall: float = 0.25

    # identity_confidence = mean of five enrichment scores (equal weight per spec)
    enrichment_component_weights: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)

    # final_score_0_10 = w_w * wealth + w_a * actionability + w_c * (overall_confidence * 10)
    weight_wealth_in_final: float = 0.40
    weight_actionability_in_final: float = 0.35
    weight_overall_conf_in_final: float = 0.25

    # Top 5 (verified)
    top5_min_overall_confidence: float = 0.75
    top5_min_identity_confidence: float = 0.80
    top5_min_wealth_signal: int = 6
    top5_min_actionability: int = 5
    top5_min_company_exists: float = 0.55

    # Reject if identity below
    min_identity_to_accept: float = 0.35
    min_person_exists_to_accept: float = 0.40


def _cfg() -> PipelineConfig:
    c = PipelineConfig()
    def _f(key: str, default: float) -> float:
        v = os.environ.get(key, "").strip()
        if not v:
            return default
        try:
            return float(v)
        except ValueError:
            return default

    c.weight_extraction_in_overall = _f("WEALTH_PW_WEIGHT_EXT", c.weight_extraction_in_overall)
    c.weight_identity_in_overall = _f("WEALTH_PW_WEIGHT_ID", c.weight_identity_in_overall)
    c.weight_article_in_overall = _f("WEALTH_PW_WEIGHT_ART", c.weight_article_in_overall)
    c.weight_wealth_in_final = _f("WEALTH_PW_WEIGHT_W", c.weight_wealth_in_final)
    c.weight_actionability_in_final = _f("WEALTH_PW_WEIGHT_ACT", c.weight_actionability_in_final)
    c.weight_overall_conf_in_final = _f("WEALTH_PW_WEIGHT_OC", c.weight_overall_conf_in_final)
    c.top5_min_overall_confidence = _f("WEALTH_TOP5_MIN_OVERALL", c.top5_min_overall_confidence)
    c.top5_min_identity_confidence = _f("WEALTH_TOP5_MIN_IDENTITY", c.top5_min_identity_confidence)
    c.top5_min_wealth_signal = int(_f("WEALTH_TOP5_MIN_WEALTH", float(c.top5_min_wealth_signal)))
    c.top5_min_actionability = int(_f("WEALTH_TOP5_MIN_ACTION", float(c.top5_min_actionability)))
    return c


def verified_pipeline_enabled() -> bool:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return False
    # Supersedes WEALTH_QUALITY_PIPELINE when set (default on)
    if os.environ.get("WEALTH_VERIFIED_PIPELINE", "1").lower() in ("0", "false", "no"):
        return False
    return True


def verified_pipeline_fallback_legacy() -> bool:
    """If True, LLM/API failure falls back to legacy hybrid path."""
    return os.environ.get("WEALTH_VERIFIED_PIPELINE_FALLBACK", "0").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# OpenAI JSON helper
# ---------------------------------------------------------------------------


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


def _clamp01(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _clamp_int(x: Any, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(round(float(x)))))
    except (TypeError, ValueError):
        return lo


# ---------------------------------------------------------------------------
# STEP 0 — Article relevance
# ---------------------------------------------------------------------------


def step0_article_relevance_gate(article_text: str) -> dict[str, Any] | None:
    t = (article_text or "").strip()
    if len(t) < 60:
        return {"relevant": False, "reason": "too_short", "article_clarity_score": 0.1, "event_type_guess": None}
    body = t[:28000]
    prompt = f"""Is this article about a current, real, business-relevant person experiencing a potentially meaningful wealth-related event?

Return JSON only:
{{
  "relevant": boolean,
  "reason": string,
  "article_clarity_score": number from 0 to 1,
  "event_type_guess": string or null
}}

Relevant examples: founder exit, acquisition, IPO, funding round, secondary sale, major equity compensation, divestiture, asset sale, business ownership event, executive appointment with major economic significance.

NOT relevant: historical retrospectives; crime without wealth angle; general market commentary; no clear person; orgs with no primary human actor; dead historical figures as subject.

Article:
{body}
"""
    data = _openai_json(prompt, temperature=0.1)
    if not data:
        return None
    data["relevant"] = bool(data.get("relevant"))
    data["article_clarity_score"] = _clamp01(data.get("article_clarity_score", 0.5))
    data["reason"] = str(data.get("reason") or "")[:500]
    et = data.get("event_type_guess")
    data["event_type_guess"] = str(et).strip() if et else None
    return data


# ---------------------------------------------------------------------------
# STEP 1 — Extraction
# ---------------------------------------------------------------------------


def step1_extract_primary_actor(article_text: str) -> dict[str, Any] | None:
    t = (article_text or "").strip()
    if len(t) < 40:
        return None
    body = t[:28000]
    prompt = f"""Extract the PRIMARY business-relevant human actor from this article.

Return JSON only:
{{
  "name": string or null,
  "role": string or null,
  "company": string or null,
  "is_real_human": boolean,
  "is_primary_actor": boolean,
  "event_type": string or null,
  "context": string,
  "extraction_confidence": number 0-1
}}

Rules:
- Main human actor only, not an organization.
- Ignore historical figures, fictional characters, random mentions unless clearly main actor.
- Prefer person most tied to the economic/wealth event.
- If no clear business-relevant primary human, set is_primary_actor false.
- If not clearly a real human, set is_real_human false.

Article:
{body}
"""
    data = _openai_json(prompt, temperature=0.08)
    if not data:
        return None
    data["is_real_human"] = bool(data.get("is_real_human", False))
    data["is_primary_actor"] = bool(data.get("is_primary_actor", False))
    data["extraction_confidence"] = _clamp01(data.get("extraction_confidence", 0.5))
    for k in ("name", "role", "company", "event_type", "context"):
        v = data.get(k)
        data[k] = (str(v).strip() if v is not None and str(v).strip() else None) if k != "context" else str(v or "")[:1200]
    return data


# ---------------------------------------------------------------------------
# STEP 2 — Enrichment (LLM + Wikipedia snippet; no naive rules as primary gate)
# ---------------------------------------------------------------------------


def step2_enrichment_scores(
    *,
    name: str,
    role: str,
    company: str,
    article_text: str,
    wiki_title: str,
    wiki_extract: str,
) -> dict[str, Any]:
    a = (article_text or "")[:14000]
    wex = (wiki_extract or "")[:4000]
    wt = (wiki_title or "")[:200]
    prompt = f"""You verify whether an extracted person/company/role is real and consistent with the article. You may use the Wikipedia snippet if it refers to the same person; if wrong person, say so.

Person: {name}
Role: {role}
Company: {company}

Wikipedia title (may be wrong entity): {wt}
Wikipedia intro:
{wex}

Article excerpt:
{a}

Return JSON only:
{{
  "person_exists_score": 0-1,
  "company_exists_score": 0-1,
  "role_match_score": 0-1,
  "person_company_match_score": 0-1,
  "enrichment_consistency_score": 0-1,
  "notes": string
}}

Score conservatively. If company is a generic fragment (e.g. single meaningless token) or clearly not a real company name, company_exists_score should be very low.
If you cannot confirm the person, person_exists_score should be low."""
    data = _openai_json(prompt, temperature=0.06)
    if not data:
        return {
            "person_exists_score": 0.3,
            "company_exists_score": 0.3,
            "role_match_score": 0.3,
            "person_company_match_score": 0.3,
            "enrichment_consistency_score": 0.2,
            "notes": "enrichment_llm_unavailable",
        }
    out = {
        "person_exists_score": _clamp01(data.get("person_exists_score")),
        "company_exists_score": _clamp01(data.get("company_exists_score")),
        "role_match_score": _clamp01(data.get("role_match_score")),
        "person_company_match_score": _clamp01(data.get("person_company_match_score")),
        "enrichment_consistency_score": _clamp01(data.get("enrichment_consistency_score")),
        "notes": str(data.get("notes") or "")[:800],
    }
    return out


def compute_identity_confidence(enrich: dict[str, Any], cfg: PipelineConfig) -> float:
    scores = (
        enrich["person_exists_score"],
        enrich["company_exists_score"],
        enrich["role_match_score"],
        enrich["person_company_match_score"],
        enrich["enrichment_consistency_score"],
    )
    w = cfg.enrichment_component_weights
    if len(w) != len(scores):
        return sum(scores) / len(scores)
    return float(sum(s * w[i] for i, s in enumerate(scores)))


# Secondary sanity: noisy company tokens (not primary gate)
_NOISE_COMPANY = frozenset({"santa", "technology business", "unknown", "n/a"})


def secondary_company_sanity(company: str | None) -> bool:
    c = re.sub(r"\s+", " ", (company or "").strip())
    if len(c) < 2:
        return False
    low = c.lower()
    if low in _NOISE_COMPANY:
        return False
    if len(c.split()) == 1 and low in _NOISE_COMPANY:
        return False
    return True


# ---------------------------------------------------------------------------
# STEP 3 — Wealth + actionability
# ---------------------------------------------------------------------------


def step3_wealth_and_actionability(
    article_text: str,
    *,
    name: str,
    role: str,
    company: str,
    event_type: str | None,
    identity_context: str,
) -> dict[str, Any]:
    t = (article_text or "")[:16000]
    prompt = f"""Given this article and entity context, score wealth relevance and financial-advisor actionability.

Person: {name}
Role: {role}
Company: {company}
Event type hint: {event_type or "unknown"}
Verification notes: {identity_context[:1200]}

Return JSON only:
{{
  "wealth_signal_score": 0-10 integer,
  "actionability_score": 0-10 integer,
  "liquidity_timing": "immediate" | "near_term" | "unclear" | "long_term" | null,
  "rationale": string,
  "wealth_signal_type": string or null
}}

wealth_signal_type examples: founder_exit, acquisition, funding_round, ipo, secondary_sale, executive_equity_event, asset_sale, ownership_growth, family_wealth_transition, compensation_event, other

Do NOT score high for mere mentions or prestige. Focus on economic significance and near-term advisor relevance.

Article:
{t}
"""
    data = _openai_json(prompt, temperature=0.1)
    if not data:
        return {
            "wealth_signal_score": 0,
            "actionability_score": 0,
            "liquidity_timing": None,
            "rationale": "scoring_unavailable",
            "wealth_signal_type": None,
        }
    lt = data.get("liquidity_timing")
    if lt is not None and str(lt).lower() not in (
        "immediate", "near_term", "unclear", "long_term", "none",
    ):
        lt = "unclear"
    return {
        "wealth_signal_score": _clamp_int(data.get("wealth_signal_score"), 0, 10),
        "actionability_score": _clamp_int(data.get("actionability_score"), 0, 10),
        "liquidity_timing": lt,
        "rationale": str(data.get("rationale") or "")[:1500],
        "wealth_signal_type": (str(data.get("wealth_signal_type")).strip() or None) if data.get("wealth_signal_type") else None,
    }


# ---------------------------------------------------------------------------
# STEP 4 — Rejection rules
# ---------------------------------------------------------------------------

REASON_IRRELEVANT_ARTICLE = "irrelevant_article"
REASON_NO_PRIMARY_ACTOR = "no_primary_actor"
REASON_NON_HUMAN_ENTITY = "non_human_entity"
REASON_PERSON_UNVERIFIED = "person_unverified"
REASON_COMPANY_UNVERIFIED = "company_unverified"
REASON_ROLE_MISMATCH = "role_mismatch"
REASON_AMBIGUOUS_IDENTITY = "ambiguous_identity"
REASON_HISTORICAL_FIGURE = "historical_figure"
REASON_DECEASED_FIGURE = "deceased_figure"
REASON_WEAK_WEALTH_SIGNAL = "weak_wealth_signal"
REASON_EXTRACTION_FAILED = "extraction_failed"
REASON_GENERIC_NOISE = "generic_noise"


def step4_rejection_rules(
    *,
    relevance: dict[str, Any] | None,
    extraction: dict[str, Any] | None,
    enrich: dict[str, Any],
    identity_confidence: float,
    wealth_block: dict[str, Any],
    dead_or_historical: bool,
    cfg: PipelineConfig,
) -> tuple[bool, str | None]:
    """Returns (accept, reason_code or None)."""
    if relevance is not None and not relevance.get("relevant", False):
        return False, REASON_IRRELEVANT_ARTICLE
    if extraction is None:
        return False, REASON_EXTRACTION_FAILED
    if not extraction.get("is_primary_actor"):
        return False, REASON_NO_PRIMARY_ACTOR
    if not extraction.get("is_real_human"):
        return False, REASON_NON_HUMAN_ENTITY
    if not extraction.get("name"):
        return False, REASON_NO_PRIMARY_ACTOR
    if dead_or_historical:
        return False, REASON_DECEASED_FIGURE
    if enrich["person_exists_score"] < cfg.min_person_exists_to_accept:
        return False, REASON_PERSON_UNVERIFIED
    if not secondary_company_sanity(extraction.get("company")):
        return False, REASON_GENERIC_NOISE
    if enrich["company_exists_score"] < 0.25 and (extraction.get("company") or "").strip():
        return False, REASON_COMPANY_UNVERIFIED
    if enrich["role_match_score"] < 0.2 and (extraction.get("role") or "").strip():
        return False, REASON_ROLE_MISMATCH
    if identity_confidence < cfg.min_identity_to_accept:
        return False, REASON_AMBIGUOUS_IDENTITY
    if wealth_block["wealth_signal_score"] < 2 and wealth_block["actionability_score"] < 2:
        return False, REASON_WEAK_WEALTH_SIGNAL
    return True, None


# ---------------------------------------------------------------------------
# STEP 5 — Final scores
# ---------------------------------------------------------------------------


def compute_overall_confidence(
    extraction_confidence: float,
    identity_confidence: float,
    article_clarity_score: float,
    cfg: PipelineConfig,
) -> float:
    return (
        cfg.weight_extraction_in_overall * _clamp01(extraction_confidence)
        + cfg.weight_identity_in_overall * _clamp01(identity_confidence)
        + cfg.weight_article_in_overall * _clamp01(article_clarity_score)
    )


def compute_final_score_0_100(
    wealth_signal_0_10: int,
    actionability_0_10: int,
    overall_confidence_0_1: float,
    cfg: PipelineConfig,
) -> float:
    inner = (
        cfg.weight_wealth_in_final * float(wealth_signal_0_10)
        + cfg.weight_actionability_in_final * float(actionability_0_10)
        + cfg.weight_overall_conf_in_final * (overall_confidence_0_1 * 10.0)
    )
    # inner is 0..10 approximately
    return max(0.0, min(100.0, inner * 10.0))


def qualifies_verified_top5(
    *,
    overall_confidence: float,
    identity_confidence: float,
    company_exists_score: float,
    wealth_signal_score: int,
    actionability_score: int,
    rejection_reason: str | None,
    event_type: str | None,
    cfg: PipelineConfig,
) -> bool:
    if rejection_reason:
        return False
    if overall_confidence <= cfg.top5_min_overall_confidence:
        return False
    if identity_confidence <= cfg.top5_min_identity_confidence:
        return False
    if company_exists_score < cfg.top5_min_company_exists:
        return False
    if wealth_signal_score < cfg.top5_min_wealth_signal:
        return False
    if actionability_score < cfg.top5_min_actionability:
        return False
    if not (event_type and str(event_type).strip()):
        return False
    return True


# ---------------------------------------------------------------------------
# Normalization + dedupe keys
# ---------------------------------------------------------------------------


def normalized_entity_key(name: str, company: str, event_type: str | None) -> str:
    def norm(s: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "", (s or "").lower())
        return s[:80]

    h = hashlib.sha256(
        f"{norm(name)}|{norm(company)}|{norm(event_type or '')}".encode()
    ).hexdigest()[:20]
    return h


def run_verified_wealth_pipeline(
    *,
    summary: str,
    source_title: str,
    source_url: str,
    published_at: Any,
    row: dict[str, Any],
    na: dict[str, Any],
    sig: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None] | None:
    """
    Full verified pipeline for one article row.

    Returns:
      None — pipeline disabled or caller should use legacy (fallback)
      ([], "reason") — discarded
      ([row_dict], None) — one prospect row
    """
    if not verified_pipeline_enabled():
        return None

    cfg = _cfg()
    debug: dict[str, Any] = {"pipeline": "verified_wealth_v2"}

    # --- Step 0
    rel = step0_article_relevance_gate(summary)
    if rel is None:
        if verified_pipeline_fallback_legacy():
            return None
        return [], REASON_IRRELEVANT_ARTICLE
    debug["relevance_gate"] = rel
    if not rel.get("relevant"):
        return [], REASON_IRRELEVANT_ARTICLE

    # --- Step 1
    ext = step1_extract_primary_actor(summary)
    if ext is None:
        if verified_pipeline_fallback_legacy():
            return None
        return [], REASON_EXTRACTION_FAILED
    debug["extraction"] = ext

    name = ext.get("name") or ""
    role = ext.get("role") or ""
    company = ext.get("company") or ""

    if not ext.get("is_primary_actor") or not ext.get("is_real_human") or not name:
        reason = REASON_NO_PRIMARY_ACTOR if not ext.get("is_primary_actor") else REASON_NON_HUMAN_ENTITY
        return [], reason

    # Wikipedia (public)
    from person_validation import enrich_with_search
    from prospect_hardening import is_historical_or_dead

    wiki = enrich_with_search(name)
    wiki_title = str(wiki.get("title") or "")
    wiki_ex = str(wiki.get("extract") or wiki.get("snippet") or "")
    debug["wikipedia"] = {"found": bool(wiki.get("found")), "title": wiki_title[:200]}

    dead_hist = is_historical_or_dead(name, summary, {"_wikipedia_extract": wiki_ex})

    # --- Step 2
    enrich = step2_enrichment_scores(
        name=name,
        role=role or "",
        company=company or "",
        article_text=summary,
        wiki_title=wiki_title,
        wiki_extract=wiki_ex,
    )
    debug["enrichment"] = enrich
    identity_confidence = compute_identity_confidence(enrich, cfg)

    # --- Step 3
    wa = step3_wealth_and_actionability(
        summary,
        name=name,
        role=role or "",
        company=company or "",
        event_type=ext.get("event_type") or rel.get("event_type_guess"),
        identity_context=json.dumps(enrich)[:2000],
    )
    debug["wealth_actionability"] = wa

    article_clarity = float(rel.get("article_clarity_score") or 0.5)
    ext_conf = float(ext.get("extraction_confidence") or 0.5)

    overall_confidence = compute_overall_confidence(ext_conf, identity_confidence, article_clarity, cfg)
    final_score = compute_final_score_0_100(
        wa["wealth_signal_score"],
        wa["actionability_score"],
        overall_confidence,
        cfg,
    )
    priority_int = int(round(final_score))
    priority_int = max(0, min(100, priority_int))

    debug["identity_confidence"] = identity_confidence
    debug["overall_confidence"] = overall_confidence
    debug["final_score_0_100"] = final_score

    rej_ok, rej_reason = step4_rejection_rules(
        relevance=rel,
        extraction=ext,
        enrich=enrich,
        identity_confidence=identity_confidence,
        wealth_block=wa,
        dead_or_historical=dead_hist,
        cfg=cfg,
    )
    debug["rejection"] = {"accept": rej_ok, "reason": rej_reason}

    if not rej_ok:
        return [], rej_reason or "rejected"

    event_type_final = (ext.get("event_type") or rel.get("event_type_guess") or wa.get("wealth_signal_type") or "Other")

    qualifies_top5 = qualifies_verified_top5(
        overall_confidence=overall_confidence,
        identity_confidence=identity_confidence,
        company_exists_score=enrich["company_exists_score"],
        wealth_signal_score=wa["wealth_signal_score"],
        actionability_score=wa["actionability_score"],
        rejection_reason=None,
        event_type=str(event_type_final) if event_type_final else None,
        cfg=cfg,
    )
    debug["qualifies_verified_top5"] = qualifies_top5

    # Build row: reuse hybrid's pattern via late import
    from ai_prospect_pipeline import (
        build_processed_row_core,
        priority_label_from_priority_score,
    )
    from two_pass_pipeline import compute_recency_score, pass1_recency_adjustment

    label = priority_label_from_priority_score(priority_int)
    if overall_confidence < 0.5:
        label = "Low"

    est_display = "Data pending"
    if wa.get("wealth_signal_score", 0) >= 6:
        est_display = "Wealth signal detected"
    verification_status = "Verified" if identity_confidence >= 0.65 else "Partially verified"

    core = build_processed_row_core(
        name=name.strip(),
        role=(role or "").strip(),
        company=(company or "").strip() or "Unknown",
        signal_type=str(event_type_final)[:80],
        signal_score=int(sig.get("signal_score") or 0),
        match_score=int(round(identity_confidence * 40)),
        priority_label=label,
        est_wealth=est_display,
        source_title=source_title,
        source_url=source_url,
        summary=summary,
        context_type="primary",
        economic_role="founder" if re.search(r"founder", role or "", re.I) else "other",
        identity_confidence=float(identity_confidence),
        verification_sources_used=["llm_enrichment", "wikipedia"] if wiki.get("found") else ["llm_enrichment"],
        priority_score=priority_int,
    )

    published_at = na.get("published_at") or row.get("detected_at") or row.get("event_date")
    pr_adj = pass1_recency_adjustment(published_at)
    priority_final = max(0, min(100, priority_int + max(-3, min(3, pr_adj // 4))))

    why_matters = (wa.get("rationale") or ext.get("context") or "")[:1200]
    norm_name = re.sub(r"\s+", " ", name.strip().lower())
    norm_co = re.sub(r"\s+", " ", (company or "").strip().lower())
    norm_et = re.sub(r"\s+", " ", str(event_type_final).lower())

    legacy = {
        **row,
        **core,
        "person_name": core["name"],
        "company_name": core["company"],
        "raw_title": source_title,
        "score": priority_final,
        "priority_score": priority_final,
        "confidence_score": int(round(overall_confidence * 100)),
        "est_wealth_display": est_display,
        "est_wealth": est_display,
        "wealth_numeric_verified": False,
        "published_at": published_at,
        "recency_score": compute_recency_score(published_at),
        "wealth_status": "likely_wealth" if wa["wealth_signal_score"] >= 5 else "unclear",
        "article_economic_relevance": True,
        "candidate_historical_dead": False,
        "signal_type": core["signal_type"],
        "event_type": str(event_type_final)[:80],
        "source_title": source_title,
        "source_url": source_url,
        "summary": summary,
        "normalized_article": na,
        "verified_pipeline_v2": True,
        "article_clarity_score": article_clarity,
        "identity_confidence": float(identity_confidence),
        "overall_confidence": float(overall_confidence),
        "extraction_confidence": float(ext_conf),
        "wealth_signal_0_10": int(wa["wealth_signal_score"]),
        "actionability_score": int(wa["actionability_score"]),
        "liquidity_timing": wa.get("liquidity_timing"),
        "wealth_signal_type": wa.get("wealth_signal_type"),
        "final_score_display": round(final_score, 1),
        "qualifies_verified_top5": qualifies_top5,
        "eligible_for_home_top": qualifies_top5,
        "rejection_reason": None,
        "verification_status": verification_status,
        "source_count": 1,
        "why_this_matters_fa": why_matters,
        "why_it_matters": why_matters[:500],
        "ai_summary": why_matters[:800] or "Summary pending — see article.",
        "enrichment_scores": enrich,
        "normalized_name_key": norm_name[:120],
        "normalized_company_key": norm_co[:120],
        "normalized_event_key": norm_et[:80],
        "dedupe_fingerprint": normalized_entity_key(name, company or "", str(event_type_final)),
        "pipeline_debug_json": json.dumps(debug, ensure_ascii=False)[:25000],
    }
    return [legacy], None
