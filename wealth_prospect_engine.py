"""
Wealth prospect identification engine for RSS ingest.

Identifies real, economically relevant people tied to strong financial signals — not general news
summaries. Uses structured LLM extraction, optional **person enrichment** (People Data Labs /
FullContact when API keys are set), optional Whitepages-style secondary check, Wikipedia for
fiction/deceased flags, and confidence gating before rows reach the dashboard.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from company_resolution import build_resolution_data, resolve_company
from high_signal_article import (
    assign_signal_priority,
    company_first_result_to_candidates,
    extract_main_company,
    forced_actors_to_candidates,
    force_extract_key_entities,
    is_high_value_article,
    process_high_value_article,
)
from wealth_signal_scoring import evaluate_signal_strength
from person_enrichment import (
    compute_enrichment_confidence,
    enrich_person,
    enrichment_layer_enabled,
    is_valid_prospect,
    lookup_whitepages,
)
from person_validation import (
    enrich_with_search,
    is_likely_alive,
    is_valid_person,
    is_valid_person_entity,
    is_valid_role,
)
from prospect_hardening import is_valid_person_name as is_valid_prospect_person_name

# --- Config -----------------------------------------------------------------

_SESSION = requests.Session()
_SESSION.headers.update(
    {"User-Agent": "WealthSignalsDashboard/1.0 (prospect-engine; educational)"}
)

_RE_OPINION_DROP = re.compile(
    r"\b(opinion|editorial|column|commentary|my take|letter to the editor|guest essay)\b",
    re.I,
)
_RE_MACRO_ONLY = re.compile(
    r"\b(gdp|inflation data|federal reserve minutes|weather forecast|earthquake|"
    r"election poll|campaign rally)\b",
    re.I,
)
_RE_JOURNALIST = re.compile(
    r"\b(journalist|reporter|correspondent|columnist|anchor|commentator|writes for)\b",
    re.I,
)
_RE_LAWYER_OK = re.compile(
    r"\b(founding partner|managing partner|name partner|owner of|chair(?:man)?\s+of\s+.*\s+firm)\b",
    re.I,
)
_RE_LAWYER_BAD = re.compile(r"\b(attorney|lawyer|counsel|esq\.)\b", re.I)

_RE_SENIORITY_FOUNDER = re.compile(
    r"\b(founder|co-?founder|owner|proprietor)\b",
    re.I,
)
_RE_SENIORITY_C = re.compile(
    r"\b(ceo|chief executive|cfo|chief financial|coo|chief operating|cto|chief technology|cmo|chief marketing|ciso|"
    r"president and|president of|president,)\b",
    re.I,
)
_RE_SENIORITY_MD = re.compile(
    r"\b(managing director|general partner|partner(?!\s+ship)|principal)\b",
    re.I,
)
_RE_SENIORITY_VP = re.compile(
    r"\b(svp|evp|vp\b|vice president|director)\b",
    re.I,
)
_RE_INVESTOR = re.compile(
    r"\b(investor|venture capitalist|angel investor|limited partner|private equity|vc\b|pe\b)\b",
    re.I,
)
_RE_BOARD = re.compile(r"\b(board member|chairman|chairwoman|chair|non-executive director|independent director)\b", re.I)
_RE_RE_DEV = re.compile(r"\b(real estate developer|property developer)\b", re.I)

_RE_HIGH_VALUE_INDUSTRY = re.compile(
    r"\b(tech|software|saas|fintech|finance|banking|private equity|venture|energy|oil|gas|"
    r"real estate|reit|pharma|biotech|healthcare)\b",
    re.I,
)

_RE_HARD_SIGNAL = re.compile(
    r"\b(acqui(?:red|re|tion)|merger|buyout|ipo\b|going public|listing|raised\s+\$|"
    r"series\s+[a-e]\b|seed round|unicorn|valuation|sold (?:for|to|stake)|exit|divest|"
    r"earn-?out|take\s*private|spac)\b",
    re.I,
)
_RE_STRONG_TREND = re.compile(
    r"\b(hiring surge|expands|expansion|major contract|record deal|opens new|"
    r"facility|plant|headcount|milestone)\b",
    re.I,
)

_RE_DECEASED = re.compile(
    r"\b(died|death|obituary|passed away)\b",
    re.I,
)
_RE_FICTION = re.compile(
    r"\b(fictional|character in|novel|tv series|film series|video game|anime)\b",
    re.I,
)


def _engine_enabled() -> bool:
    return os.environ.get("WEALTH_SIGNALS_PROSPECT_ENGINE", "1").lower() not in ("0", "false", "no")


def ingest_via_prospect_engine() -> bool:
    """True when the prospect engine should run on RSS items (OpenAI + engine flag)."""
    return _engine_enabled() and bool(os.environ.get("OPENAI_API_KEY", "").strip())


# Set by ``process_article_to_rows`` so ingest can fall back to legacy heuristics only on API failure.
LAST_ENGINE_API_ERROR: bool = False


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


def step1_drop_article(title: str, summary: str, body: str) -> tuple[bool, str]:
    """
    Drop articles that are pure opinion, lack business relevance, or lack any plausible person+event.
    """
    blob = f"{title} {summary} {body}".strip()
    if len(blob) < 40:
        return True, "too_short"
    if _RE_OPINION_DROP.search(title) and not _RE_HARD_SIGNAL.search(blob):
        return True, "opinion_commentary"
    if _RE_MACRO_ONLY.search(title) and not _RE_HARD_SIGNAL.search(blob):
        return True, "macro_no_person_hook"
    # Must have at least one capitalized name-like pair or a strong deal token
    if not _RE_HARD_SIGNAL.search(blob) and not re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", blob):
        if not _finance_keyword_present(blob):
            return True, "no_business_anchor"
    return False, ""


def _finance_keyword_present(blob: str) -> bool:
    return bool(
        re.search(
            r"\b(ceo|cfo|founder|funding|million|billion|acquisition|ipo|investor|board|"
            r"executive|company|startup|venture|earnings|revenue|deal|merger)\b",
            blob,
            re.I,
        )
    )


def _sanitize_company_field(name: str) -> str:
    from data import normalize_company_name_field, sanitize_company_name

    s = (name or "").strip()
    if not s or len(s) > 90:
        return ""
    low = s.lower()
    if any(
        x in low
        for x in (
            "middle east",
            "eastern europe",
            "according to",
            "said in",
            "the white house",
            "the fed",
            "sources say",
        )
    ):
        return ""
    if re.search(r"\b(said|according|reported|analysts)\b", low):
        return ""
    cleaned = sanitize_company_name(s)
    if not cleaned:
        return ""
    return normalize_company_name_field(cleaned)


def wikipedia_identity_check(name: str, *, preloaded: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Wikipedia lookup: search + extract intro. Used to flag deceased/fictional entries.

    Pass ``preloaded`` from :func:`person_validation.enrich_with_search` to avoid a duplicate request
    when entity validation already fetched the same page.
    """
    out: dict[str, Any] = {
        "found": False,
        "title": "",
        "extract": "",
        "deceased_hint": False,
        "fiction_hint": False,
    }
    q = (name or "").strip()
    if len(q) < 4 or not is_valid_person(q):
        return out

    if preloaded is not None and preloaded.get("found"):
        excerpt = (preloaded.get("extract") or "")[:900]
        out["found"] = True
        out["title"] = str(preloaded.get("title") or "")
        out["extract"] = excerpt
    else:
        data = enrich_with_search(q) if preloaded is None else preloaded
        if not data.get("found"):
            return out
        excerpt = (data.get("extract") or "")[:900]
        out["found"] = True
        out["title"] = str(data.get("title") or "")
        out["extract"] = excerpt

    el = excerpt.lower()
    if _RE_FICTION.search(excerpt) or "fictional character" in el:
        out["fiction_hint"] = True
    if _RE_DECEASED.search(excerpt):
        out["deceased_hint"] = True
    if re.search(r"\b\d{4}\s*[–-]\s*\d{4}\b", excerpt[:500]):
        out["deceased_hint"] = True
    return out


def allowed_role_bucket(role: str, blob: str) -> tuple[bool, str]:
    """Returns (ok, bucket) for seniority scoring."""
    rl = (role or "").strip()
    b = f"{rl} {blob}".lower()
    if _RE_JOURNALIST.search(rl) or _RE_JOURNALIST.search(b[:800]):
        return False, "journalist"
    if _RE_LAWYER_BAD.search(rl):
        if _RE_LAWYER_OK.search(rl) or _RE_LAWYER_OK.search(b[:1200]):
            pass
        else:
            return False, "lawyer_non_owner"
    if _RE_SENIORITY_FOUNDER.search(rl):
        return True, "founder_owner"
    if _RE_SENIORITY_C.search(rl):
        return True, "c_suite"
    if _RE_INVESTOR.search(rl):
        return True, "investor"
    if _RE_SENIORITY_MD.search(rl):
        return True, "md_partner"
    if _RE_BOARD.search(rl):
        return True, "board"
    if _RE_RE_DEV.search(rl):
        return True, "re_dev"
    if _RE_SENIORITY_VP.search(rl):
        return True, "vp_director"
    return False, "low_tier"


def seniority_points(bucket: str) -> int:
    return {
        "founder_owner": 25,
        "c_suite": 20,
        "investor": 18,
        "md_partner": 18,
        "board": 16,
        "re_dev": 18,
        "vp_director": 12,
    }.get(bucket, 0)


def wealth_likelihood_points(company: str, industry_hint: str, blob: str) -> int:
    pts = 8
    bl = f"{blob} {company} {industry_hint}".lower()
    if _RE_HIGH_VALUE_INDUSTRY.search(bl):
        pts += 10
    if re.search(r"\b(\$[\d,]+|\d+\s*(million|billion)|unicorn|valuation)\b", bl, re.I):
        pts += 7
    return min(25, pts)


def signal_strength_points(blob: str, signal_label: str) -> int:
    s = (signal_label or "").strip().lower()
    bl = blob or ""
    if s == "hard" or _RE_HARD_SIGNAL.search(bl):
        return 30
    if s in ("strong", "strong_trend") or _RE_STRONG_TREND.search(bl):
        return 20
    if s in ("weak", "moderate"):
        return 5
    if _RE_HARD_SIGNAL.search(bl):
        return 30
    if _RE_STRONG_TREND.search(bl):
        return 20
    return 5


def data_confidence_points(
    wiki: dict[str, Any],
    person: str,
    company: str,
    *,
    anchor_in_text: bool,
    blob_lower: str,
) -> tuple[int, str]:
    """
    Returns (0-20 points, tier: verified|partial|unknown).
    unknown → caller drops row unless strong deal context rescues (handled in caller).
    """
    if wiki.get("fiction_hint") or wiki.get("deceased_hint"):
        return 0, "rejected"
    if wiki.get("found"):
        ex = (wiki.get("extract") or "").lower()
        pn = person.lower().split()
        company_l = (company or "").lower()
        name_hit = any(len(p) > 2 and p in ex for p in pn)
        co_hit = bool(company_l and company_l[:4] in ex.replace(" ", ""))
        if name_hit and (co_hit or not company):
            return 20, "verified"
        if name_hit:
            return 10, "partial"
        return 10, "partial"
    # No Wikipedia hit: require strong in-text anchoring
    mentions = blob_lower.count(person.lower())
    if anchor_in_text and is_valid_person(person) and (mentions >= 2 or _RE_HARD_SIGNAL.search(blob_lower)):
        return 10, "partial"
    if anchor_in_text and is_valid_person(person):
        return 10, "partial"
    return 0, "unknown"


def compute_engine_score(
    *,
    seniority: int,
    wealth_pts: int,
    signal_pts: int,
    conf_pts: int,
) -> int:
    total = int(seniority) + int(wealth_pts) + int(signal_pts) + int(conf_pts)
    return max(0, min(100, total))


def _low_confidence(
    *,
    conf_tier: str,
    conf_pts: int,
    enrichment_confidence: int | None,
    high_value_article: bool = False,
) -> bool:
    """True when identity/signal confidence is weak enough to downgrade score and labels."""
    if high_value_article:
        return False
    if conf_tier in ("partial", "unknown"):
        return True
    if int(conf_pts) <= 10:
        return True
    try:
        thresh = int(os.environ.get("WEALTH_SIGNALS_LOW_CONFIDENCE_ENRICH_MAX", "45"))
    except ValueError:
        thresh = 45
    if enrichment_confidence is not None and int(enrichment_confidence) < thresh:
        return True
    return False


def _downgrade_wealth_label(s: str) -> str:
    return {"Strong": "Moderate", "Moderate": "Weak", "Weak": "Weak"}.get(s, "Weak")


def _downgrade_liquidity_label(s: str) -> str:
    if s == "Yes":
        return "Potential"
    return s


def _apply_low_confidence_downgrade(
    score: int,
    wealth_signal_hint: str,
    liquidity_event_hint: str,
) -> tuple[int, str, str]:
    """Reduce pipeline score and step wealth/liquidity labels down one tier."""
    score_out = max(38, int(score) - 12)
    return (
        score_out,
        _downgrade_wealth_label(wealth_signal_hint),
        _downgrade_liquidity_label(liquidity_event_hint),
    )


def _extraction_prompt(title: str, summary: str, body: str) -> str:
    text = f"TITLE:\n{title}\n\nSUMMARY:\n{summary}\n\nARTICLE:\n{body}".strip()
    return f"""You are a strict prospecting analyst for financial advisors. Your job is NOT to summarize news.

Extract ONLY real human beings who plausibly have wealth or institutional decision-making power AND are tied to a concrete business/finance event in this article.

Return ONE JSON object with this exact shape:
{{
  "drop_article": true or false,
  "drop_reason": "short string if drop_article",
  "company_mentions": ["real company names mentioned, max 8"],
  "candidates": [
    {{
      "person_name": "Full name as in article",
      "company": "Primary employer/deal company or empty",
      "role": "Title/role as stated or inferred from article only",
      "is_journalist_or_commentator": true or false,
      "is_primary_subject": true if this person is central to the story,
      "seniority_bucket": "founder_owner|c_suite|investor|md_partner|board|vp_director|re_dev|other",
      "ownership_hint": true or false,
      "proximity_rank": 1-5 (5 = closest to the money event),
      "financial_signal": "hard|strong_trend|weak|none",
      "event_type": "Founder Exit|Funding|Promotion|Board Appointment|Other",
      "one_sentence_bio": "1-2 sentences: who they are professionally (facts from article only)",
      "what_happened": "one sentence: the event",
      "why_financial": "one sentence: why an FA should care (money/liquidity/control)"
    }}
  ]
}}

Rules:
- Include ALL qualifying candidates (up to 6), ranked by proximity to the money event. Do NOT pick only the first name — if a fictional or minor name appears with real founders, omit the fictional one entirely.
- Exclude: journalists, pundits, unnamed officials, generic regions (e.g. "Middle East"), companies mistaken for people, purely historical figures not driving today's event.
- If the article has no identifiable individual tied to a business/financial event, set drop_article true.
- If financial_signal is "none" for a person, omit them from candidates.
- Never invent people not present in the text.
- Company must be a real organization name, not a sentence fragment.

Text:
{text[:28000]}
"""


def _crunchbase_company_from_enriched(enriched: dict[str, Any] | None) -> str:
    """Reserved for Crunchbase API payloads nested under enrichment ``raw``."""
    if not enriched or not isinstance(enriched, dict):
        return ""
    raw = enriched.get("raw")
    if not isinstance(raw, dict):
        return ""
    for key in ("crunchbase_company", "crunchbase_org_name"):
        v = raw.get(key)
        if v:
            return str(v).strip()
    cbd = raw.get("crunchbase")
    if isinstance(cbd, dict):
        c = cbd.get("company")
        if c:
            return str(c).strip()
    return ""


def merge_ai_summary_parts(parts: dict[str, str]) -> str:
    bio = (parts.get("bio") or "").strip()
    what = (parts.get("what") or "").strip()
    why = (parts.get("why") or "").strip()
    lines = []
    if bio:
        lines.append(f"1) Who: {bio}")
    if what:
        lines.append(f"2) What happened: {what}")
    if why:
        lines.append(f"3) Why it matters financially: {why}")
    return "\n".join(lines) if lines else ""


def extract_candidates_via_llm(title: str, summary: str, body: str) -> dict[str, Any] | None:
    return _openai_json(_extraction_prompt(title, summary, body), temperature=0.15)


def final_drop_row(
    *,
    financial_signal: str,
    conf_tier: str,
    seniority_pts: int,
    ownership_hint: bool,
    wiki_rejected: bool,
    hard_event: bool,
) -> bool:
    if wiki_rejected:
        return True
    if financial_signal == "none":
        return True
    if conf_tier == "unknown" and not hard_event:
        return True
    if seniority_pts < 12 and not ownership_hint:
        return True
    if conf_tier == "partial" and financial_signal == "weak" and seniority_pts < 16 and not hard_event:
        return True
    return False


def _structural_opportunity_row(
    *,
    title: str,
    summary: str,
    body: str,
    full_explanation: str,
    link: str,
    entry_date_iso: str,
    detected_at,
    high_value_article: bool,
    signal_eval: dict[str, Any],
) -> dict[str, Any]:
    """
    No named person passed extraction/validation, but the article still looks like a major
    financial event — emit a single institutional follow-up row.
    """
    co_raw = extract_main_company(title, body)
    company_name_out = resolve_company(co_raw, {}) if co_raw else "Data pending"
    score = 55
    fin_sig = "hard"
    ws_hint = "Strong"
    liq_hint = "Yes"
    person = "Institutional Contact"
    role = "Decision maker (verify)"
    why = (
        "High-value financial event — no named individual passed validation; "
        "manual follow-up recommended to identify the right contact."
    )
    ai_sum = merge_ai_summary_parts(
        {
            "bio": f"{person} — {role} linked to {company_name_out}.",
            "what": "Story involves a significant capital, M&A, IPO, funding, or markets event.",
            "why": why,
        }
    )
    conf_sc = min(100, score + 5)
    audit: dict[str, Any] = {
        "engine": "v1",
        "structural_opportunity": True,
        "high_value_article": True,
        "conf_tier": "partial",
        "financial_signal": fin_sig,
        "company_resolved": company_name_out,
        "low_confidence_downgrade": False,
        "signal_strength_score": int(signal_eval.get("score") or 0),
        "signal_priority": assign_signal_priority(
            signal_level=str(signal_eval.get("level") or ""),
            high_value_article=high_value_article,
            confidence_score=conf_sc,
        ),
    }
    return {
        "person_name": person,
        "additional_people": [],
        "company_name": company_name_out,
        "event_type": "Other",
        "raw_title": title,
        "role": role,
        "event_date": entry_date_iso,
        "detected_at": detected_at,
        "why_it_matters": why[:500],
        "source_url": link,
        "full_explanation": (full_explanation or "")[:4000],
        "quality_score": 0,
        "confidence_score": conf_sc,
        "is_relevant": True,
        "weak_signal": False,
        "client_type_hint": "",
        "source_of_wealth_hint": "",
        "extraction_audit_json": json.dumps(audit)[:20000],
        "liquidity_event_hint": liq_hint,
        "wealth_signal_hint": ws_hint,
        "wealth_signal_raw_hint": fin_sig,
        "ingest_overall_extraction_confidence": "partial",
        "ai_summary": ai_sum,
        "ai_why_it_matters": why,
        "ai_client_who": f"{person} — {role}",
        "ai_why_money": why,
        "extracted_person_name": person,
        "extracted_role": role,
        "extracted_company": company_name_out if company_name_out != "Data pending" else "",
        "prospect_bio": "Institutional / unnamed decision-maker opportunity (verify in source).",
        "_use_engine_score": True,
        "_engine_score": score,
        "engine_pipeline_score": score,
    }


def process_article_to_rows(
    *,
    title: str,
    summary: str,
    article_paragraphs: str,
    full_explanation: str,
    link: str,
    entry_date_iso: str,
    detected_at,
) -> list[dict[str, Any]]:
    """
    Run the full prospect engine for one RSS item. Returns zero or more row dicts compatible with
    ``data._rss_items_to_signals`` output.
    """
    global LAST_ENGINE_API_ERROR
    LAST_ENGINE_API_ERROR = False

    if not _engine_enabled():
        return []

    body = (article_paragraphs or "").strip()
    blob_for_filter = f"{title} {summary} {body}".strip()
    signal_eval = evaluate_signal_strength(blob_for_filter)
    signal_level = str(signal_eval.get("level") or "")
    high_value_article = is_high_value_article(blob_for_filter)

    drop, _reason = step1_drop_article(title, summary, body)
    if drop:
        return []

    result = None
    if high_value_article:
        result = process_high_value_article(blob_for_filter)
    hv_candidates = company_first_result_to_candidates(result) if result else []

    parsed = extract_candidates_via_llm(title, summary, body)

    candidates: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    def _merge_candidates(lst: list[dict[str, Any]]) -> None:
        for c in lst:
            if not isinstance(c, dict):
                continue
            pn = str(c.get("person_name") or "").strip().lower()
            if pn and pn not in seen_names:
                candidates.append(c)
                seen_names.add(pn)

    _merge_candidates(hv_candidates)

    if parsed and not parsed.get("drop_article"):
        pc = parsed.get("candidates")
        if isinstance(pc, list):
            _merge_candidates(pc)

    if parsed and parsed.get("drop_article") and not high_value_article:
        return []

    if not parsed:
        if high_value_article:
            if not candidates:
                forced_list = force_extract_key_entities(blob_for_filter)
                if forced_list:
                    _merge_candidates(forced_actors_to_candidates(forced_list))
                    LAST_ENGINE_API_ERROR = False
                else:
                    LAST_ENGINE_API_ERROR = True
            else:
                LAST_ENGINE_API_ERROR = False
        else:
            LAST_ENGINE_API_ERROR = True
    else:
        LAST_ENGINE_API_ERROR = False
        if parsed.get("drop_article") and high_value_article and not candidates:
            forced_list = force_extract_key_entities(blob_for_filter)
            if forced_list:
                _merge_candidates(forced_actors_to_candidates(forced_list))

    if high_value_article and len(candidates) < 2:
        forced_list = force_extract_key_entities(blob_for_filter)
        if forced_list:
            _merge_candidates(forced_actors_to_candidates(forced_list))

    if not candidates:
        if high_value_article:
            return [
                _structural_opportunity_row(
                    title=title,
                    summary=summary,
                    body=body,
                    full_explanation=full_explanation,
                    link=link,
                    entry_date_iso=entry_date_iso,
                    detected_at=detected_at,
                    high_value_article=high_value_article,
                    signal_eval=signal_eval,
                )
            ]
        return []

    companies_raw = parsed.get("company_mentions") if parsed else None
    company_hints: list[str] = []
    if isinstance(companies_raw, list):
        company_hints = [str(x).strip() for x in companies_raw if str(x).strip()][:8]

    rows_out: list[dict[str, Any]] = []

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        person = str(cand.get("person_name") or "").strip()
        if not is_valid_prospect_person_name(person, blob_for_filter):
            continue
        company_article = _sanitize_company_field(str(cand.get("company") or ""))
        company_from_search = ""
        company_from_crunchbase = ""
        role = str(cand.get("role") or "").strip()
        if not person:
            continue
        # Entity-type validation + historical/deceased filter (before enrichment / scoring)
        if not is_valid_person_entity(person, blob_for_filter):
            continue
        wiki_early = enrich_with_search(person)
        if not is_likely_alive(person, wiki_early):
            continue
        if str(cand.get("is_journalist_or_commentator") or "").lower() == "true":
            continue

        fin_sig = str(cand.get("financial_signal") or "none").strip().lower()
        if fin_sig == "none":
            continue

        enrichment_audit: dict[str, Any] | None = None
        enrichment_confidence_value: int | None = None
        if enrichment_layer_enabled():
            enriched = enrich_person(person, company_hint=company_article, role_hint=role)
            if enriched is None:
                continue
            role = (str(enriched.get("job_title") or "").strip() or role).strip()
            co_en = str(enriched.get("company") or "").strip()
            if co_en:
                company_from_search = _sanitize_company_field(co_en) or co_en.strip()
            company_from_crunchbase = _crunchbase_company_from_enriched(enriched)
            seniority_e = str(enriched.get("seniority") or "").strip()
            if not is_valid_prospect(role, seniority_e if seniority_e else None):
                continue
            wp_data = lookup_whitepages(
                person, (company_from_search or company_article or "").strip()
            )
            econf = compute_enrichment_confidence(enriched, wp_data)
            enrichment_confidence_value = int(econf)
            if econf < 30:
                continue
            enrichment_audit = {
                "source": str(enriched.get("source") or ""),
                "confidence": econf,
                "whitepages": bool(wp_data),
                "api_verified": bool(enriched.get("api_verified")),
            }

        resolution_data = build_resolution_data(
            company_article=company_article,
            company_search=company_from_search,
            company_crunchbase=company_from_crunchbase,
        )
        company_name_out = resolve_company(company_article, resolution_data)
        company = "" if company_name_out == "Data pending" else company_name_out

        if not is_valid_role(role):
            continue

        ok_role, bucket = allowed_role_bucket(role, blob_for_filter)
        sb = str(cand.get("seniority_bucket") or "").strip().lower()
        if not ok_role and sb in {
            "founder_owner",
            "c_suite",
            "investor",
            "md_partner",
            "board",
            "vp_director",
            "re_dev",
        }:
            ok_role = True
            bucket = sb

        if not ok_role:
            continue

        s_pts = seniority_points(bucket)
        if s_pts == 0 and sb == "other":
            continue

        # Wikipedia fiction/deceased hints (reuse single fetch from entity step when possible)
        wiki = wikipedia_identity_check(person, preloaded=wiki_early)
        if wiki.get("fiction_hint") or wiki.get("deceased_hint"):
            continue

        blob_lower = blob_for_filter.lower()
        anchor = person.lower() in blob_lower
        w_pts = wealth_likelihood_points(company, "", blob_for_filter)
        sig_pts = signal_strength_points(blob_for_filter, fin_sig)
        conf_pts, conf_tier = data_confidence_points(
            wiki, person, company, anchor_in_text=anchor, blob_lower=blob_lower
        )

        ownership_hint = bool(cand.get("ownership_hint"))
        wiki_rej = conf_tier == "rejected"

        hard_ev = bool(_RE_HARD_SIGNAL.search(blob_for_filter))
        if final_drop_row(
            financial_signal=fin_sig,
            conf_tier=conf_tier,
            seniority_pts=s_pts,
            ownership_hint=ownership_hint,
            wiki_rejected=wiki_rej,
            hard_event=hard_ev,
        ):
            continue

        score = compute_engine_score(
            seniority=s_pts,
            wealth_pts=w_pts,
            signal_pts=sig_pts,
            conf_pts=conf_pts,
        )
        if score < 38:
            continue

        ws_hint = "Strong" if fin_sig == "hard" else ("Moderate" if fin_sig == "strong_trend" else "Weak")
        liq_hint = "Yes" if fin_sig == "hard" else "Potential"
        low_confidence = _low_confidence(
            conf_tier=conf_tier,
            conf_pts=conf_pts,
            enrichment_confidence=enrichment_confidence_value,
            high_value_article=high_value_article,
        )
        if low_confidence:
            score, ws_hint, liq_hint = _apply_low_confidence_downgrade(score, ws_hint, liq_hint)

        et = str(cand.get("event_type") or "Other").strip()
        if et not in ("Founder Exit", "Funding", "Promotion", "Board Appointment", "Other"):
            et = "Other"

        bio = str(cand.get("one_sentence_bio") or "").strip()
        what = str(cand.get("what_happened") or "").strip()
        why = str(cand.get("why_financial") or "").strip()
        ai_sum = merge_ai_summary_parts({"bio": bio, "what": what, "why": why})

        _co_key = (
            company_name_out if company_name_out != "Data pending" else company_article
        ).lower()
        others = [p for p in company_hints if p and p.lower() != _co_key][:6]
        extra_people: list[str] = []
        for c in candidates:
            if not isinstance(c, dict):
                continue
            n = str(c.get("person_name") or "").strip()
            if (
                n
                and n.lower() != person.lower()
                and is_valid_person_entity(n, blob_for_filter)
                and is_valid_prospect_person_name(n, blob_for_filter)
            ):
                extra_people.append(n)
        extra_people = list(dict.fromkeys(extra_people))[:8]
        extra_people = [x for x in extra_people if x.lower() != person.lower()]

        why_matters = why or (what + " " + why).strip() or "Wealth-relevant corporate event with a named decision-maker."

        audit = {
            "engine": "v1",
            "wiki_found": wiki.get("found"),
            "conf_tier": conf_tier,
            "financial_signal": fin_sig,
            "bucket": bucket,
            "company_resolved": company_name_out,
            "company_article": company_article,
            "company_search": company_from_search,
            "company_crunchbase": company_from_crunchbase,
        }
        if enrichment_audit is not None:
            audit["person_enrichment"] = enrichment_audit
        audit["low_confidence_downgrade"] = bool(low_confidence)
        audit["high_value_article"] = bool(high_value_article)
        audit["signal_strength_score"] = int(signal_eval.get("score") or 0)
        audit["signal_priority"] = assign_signal_priority(
            signal_level=signal_level,
            high_value_article=high_value_article,
            confidence_score=min(100, int(score) + 5),
        )
        if cand.get("_company_first"):
            audit["company_first_pipeline"] = True

        row: dict[str, Any] = {
            "person_name": person,
            "additional_people": extra_people,
            "company_name": company_name_out,
            "event_type": et,
            "raw_title": title,
            "role": role,
            "event_date": entry_date_iso,
            "detected_at": detected_at,
            "why_it_matters": why_matters[:500],
            "source_url": link,
            "full_explanation": (full_explanation or "")[:4000],
            "quality_score": 0,
            "confidence_score": min(100, score + 5),
            "is_relevant": True,
            "weak_signal": bool(fin_sig == "weak" or low_confidence),
            "client_type_hint": "",
            "source_of_wealth_hint": "",
            "extraction_audit_json": json.dumps(audit)[:20000],
            "liquidity_event_hint": liq_hint,
            "wealth_signal_hint": ws_hint,
            "wealth_signal_raw_hint": fin_sig,
            "ingest_overall_extraction_confidence": conf_tier,
            "ai_summary": ai_sum,
            "ai_why_it_matters": why,
            "ai_client_who": f"{person} — {role}" if role else person,
            "ai_why_money": why,
            "extracted_person_name": person,
            "extracted_role": role,
            "extracted_company": company_name_out if company_name_out != "Data pending" else "",
            "prospect_bio": bio,
            "_use_engine_score": True,
            "_engine_score": score,
            "engine_pipeline_score": score,
        }
        rows_out.append(row)

    # Sort by proximity / score
    rows_out.sort(
        key=lambda r: (-int(r.get("_engine_score") or 0), str(r.get("person_name") or ""))
    )
    return rows_out
