"""
Prospect pipeline orchestration: article processing, enrichment helpers, and DataFrame I/O.

Structured LLM extraction, signal gating, match scoring, and identity cross-check live in
``ai_prospect_pipeline``. This module supplies ``enrich_entity`` (Wikipedia / OpenCorporates / APIs),
``process_article_row`` / ``process_articles``, and compatibility helpers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Paths & caches (disk-backed; survives Streamlit reruns)
# -----------------------------------------------------------------------------
_CACHE_ROOT = Path(__file__).resolve().parent / ".cache" / "wealth_pipeline"
_ENTITY_FILE = _CACHE_ROOT / "entity_enrichment.json"


def _ensure_cache() -> None:
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _hash_key(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:24]


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json_file(path: Path, data: dict[str, Any]) -> None:
    _ensure_cache()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=0)
    except OSError:
        pass


_entity_cache: dict[str, Any] | None = None


def _get_entity_cache() -> dict[str, Any]:
    global _entity_cache
    if _entity_cache is None:
        _entity_cache = _load_json_file(_ENTITY_FILE)
    return _entity_cache


def _put_entity_cache(key: str, value: Any) -> None:
    global _entity_cache
    c = _get_entity_cache()
    c[key] = value
    _entity_cache = c
    _save_json_file(_ENTITY_FILE, c)


_SESSION = requests.Session()
_SESSION.headers.update(
    {"User-Agent": "WealthSignalsDashboard/1.1 (hybrid-pipeline; educational)"}
)

# --- Money (reuse-style for est wealth display) ---
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


def _largest_money_usd(text: str) -> tuple[float, str | None]:
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


# -----------------------------------------------------------------------------
# Hard filter (non-humans / garbage) — broad recall; AI marks junk via roles/scores
# -----------------------------------------------------------------------------
_BAD_NAME_SUBSTR = frozenset(
    {
        "department",
        "agency",
        "licensing",
        "government",
        "middle east",
        "technology business",
        "region",
        "committee",
        "ministry",
        "the white house",
        "european union",
    }
)


def hard_filter_entity(e: dict[str, Any]) -> bool:
    """Return True to KEEP."""
    name = (e.get("name") or "").strip()
    if not name:
        return False
    low = name.lower()
    if len(name.split()) < 2:
        return False
    if any(x in low for x in _BAD_NAME_SUBSTR):
        return False
    # Obvious org-style tokens in a "person" name
    if any(
        t in low
        for t in (
            " inc",
            " llc",
            " ltd",
            " corp",
            " corporation",
            " university",
            " college",
        )
    ):
        return False
    return True


# -----------------------------------------------------------------------------
# STEP 3 — Enrichment (Wikipedia → OpenCorporates → PDL/FC)
# -----------------------------------------------------------------------------
def _wikipedia_search_first_title(query: str) -> str | None:
    q = (query or "").strip()
    if len(q) < 3:
        return None
    try:
        r = _SESSION.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": q,
                "format": "json",
                "srlimit": 5,
            },
            timeout=12.0,
        )
    except (requests.RequestException, OSError):
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    hits = (data.get("query") or {}).get("search") or []
    if not isinstance(hits, list) or not hits:
        return None
    t0 = hits[0].get("title")
    return str(t0).strip() if t0 else None


def _wikipedia_extract_and_url(title: str) -> tuple[str, str, str]:
    """Returns (plain_extract, page_url, title)."""
    try:
        r = _SESSION.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "exintro": 1,
                "titles": title,
                "format": "json",
            },
            timeout=12.0,
        )
    except (requests.RequestException, OSError):
        return "", "", title
    if r.status_code != 200:
        return "", "", title
    try:
        data = r.json()
    except ValueError:
        return "", "", title
    pages = (data.get("query") or {}).get("pages") or {}
    if not isinstance(pages, dict):
        return "", "", title
    first = next(iter(pages.values()), {})
    extract = str((first or {}).get("extract") or "")
    title_norm = str((first or {}).get("title") or title)
    safe = title_norm.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{quote(safe, safe='/()_:')}"
    return extract.strip(), url, title_norm


def _opencorporates_company_hint(company: str) -> dict[str, Any]:
    c = (company or "").strip()
    if len(c) < 2:
        return {}
    key = os.environ.get("OPENCORPORATES_API_KEY", "").strip()
    params: dict[str, Any] = {"q": c, "format": "json"}
    if key:
        params["api_token"] = key
    try:
        r = _SESSION.get(
            "https://api.opencorporates.com/v0.4/companies/search",
            params=params,
            timeout=12.0,
        )
    except (requests.RequestException, OSError):
        return {}
    if r.status_code != 200:
        return {}
    try:
        data = r.json()
    except ValueError:
        return {}
    companies = (((data or {}).get("results") or {}).get("companies") or [])
    if not companies:
        return {}
    first = companies[0].get("company") or {}
    if not isinstance(first, dict):
        return {}
    return {
        "company_name": str(first.get("name") or "").strip(),
        "jurisdiction": str(first.get("jurisdiction_code") or "").strip(),
        "opencorporates_url": str(first.get("opencorporates_url") or "").strip(),
    }


def _parse_wikipedia_role_company(extract: str) -> tuple[str, str]:
    """Very light heuristic on first ~600 chars (not regex-NER: substring patterns only)."""
    snippet = (extract or "")[:800]
    role = ""
    company = ""
    low = snippet.lower()
    if "chief executive" in low or " ceo " in low:
        role = "CEO"
    elif "founder" in low:
        role = "Founder"
    # "of OpenAI" / "at Google"
    m = re.search(
        r"(?:CEO|founder|co-founder|president)\s+(?:of|at)\s+([A-Z][A-Za-z0-9&\-. ]{1,80}?)(?:[,.;]|\s+who|\s+and|\Z)",
        snippet,
        re.I,
    )
    if m:
        company = m.group(1).strip()
    return role, company


def enrich_entity(entity: dict[str, Any]) -> dict[str, Any]:
    """
    Merge Wikipedia + OpenCorporates (company) + optional enrich_person (PDL/FC).
    Cached by normalized name + company hint.
    """
    name = str(entity.get("name") or "").strip()
    co_hint = str(entity.get("company") or "").strip()
    cache_key = _hash_key(f"{name.lower()}|{co_hint.lower()}")
    cached = _get_entity_cache().get(cache_key)
    if isinstance(cached, dict) and cached.get("_version") == 4:
        return cached

    out: dict[str, Any] = {
        "_version": 4,
        "canonical_name": name,
        "role": "",
        "company": "",
        "industry": "",
        "est_net_worth": "",
        "wikipedia_url": "",
        "linkedin_url": "",
        "sources": [],
        "wiki_bio_deceased": False,
        "_wikipedia_extract": "",
    }

    title = _wikipedia_search_first_title(name)
    extract, wiki_url, canon_title = ("", "", name)
    if title:
        extract, wiki_url, canon_title = _wikipedia_extract_and_url(title)
        out["sources"].append("wikipedia")
        out["wikipedia_url"] = wiki_url
        wiki_display_title = canon_title.replace("_", " ")
        out["canonical_name"] = wiki_display_title
        out["_wikipedia_extract"] = extract or ""
        # Never adopt list pages / case titles / band disambiguation as a person's display name
        from prospect_hardening import is_valid_person_name

        if not is_valid_person_name(wiki_display_title, extract or ""):
            out["canonical_name"] = name
        el = (extract or "").lower()[:2000]
        out["wiki_bio_deceased"] = (
            ("born" in el and "died" in el)
            or bool(re.search(r"\bdied\s+\d{4}\b", el))
            or ("obituary" in el)
        )
        wr, wc = _parse_wikipedia_role_company(extract)
        if wr:
            out["role"] = wr
        if wc:
            out["company"] = wc

    if co_hint:
        oc = _opencorporates_company_hint(co_hint)
        if oc.get("company_name"):
            out["sources"].append("opencorporates")
            if not out["company"]:
                out["company"] = oc["company_name"]

    try:
        from person_enrichment import enrich_person as _pdl_fc_enrich
    except ImportError:
        _pdl_fc_enrich = None  # type: ignore[assignment]

    if _pdl_fc_enrich:
        api_hit = _pdl_fc_enrich(name, company_hint=co_hint or out["company"], role_hint=str(entity.get("role") or ""))
        if isinstance(api_hit, dict) and api_hit.get("source") not in (None, "extracted_fallback"):
            out["sources"].append(str(api_hit.get("source") or "profile_api"))
            jt = str(api_hit.get("job_title") or "").strip()
            co = str(api_hit.get("company") or "").strip()
            if jt:
                out["role"] = jt
            if co:
                out["company"] = co

    if not out["company"] and co_hint:
        out["company"] = co_hint

    if not out["role"]:
        out["role"] = str(entity.get("role") or "").strip()

    # Industry / net worth: optional from extract
    if "billionaire" in (extract or "").lower() or "net worth" in (extract or "").lower():
        out["est_net_worth"] = "See Wikipedia / public estimates"
    if m_li := re.search(r"linkedin\.com/in/([a-z0-9\-]+)", extract or "", re.I):
        out["linkedin_url"] = f"https://www.linkedin.com/in/{m_li.group(1)}"

    _put_entity_cache(cache_key, out)
    return out


# -----------------------------------------------------------------------------
# STEP 4 — Entity resolution
# -----------------------------------------------------------------------------
def _sim(a: str, b: str) -> float:
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def resolve_identity(entity: dict[str, Any], enriched_data: dict[str, Any]) -> dict[str, Any]:
    """
    Merge article entity with enrichment; flag mismatches for downstream match scoring.
    """
    name = str(entity.get("name") or "").strip()
    role = str(entity.get("role") or "").strip()
    company = str(entity.get("company") or "").strip()
    ed = enriched_data or {}

    canon = str(ed.get("canonical_name") or "").strip()
    if canon and _sim(name, canon) >= 0.55:
        name = canon

    ero = str(ed.get("role") or "").strip()
    eco = str(ed.get("company") or "").strip()

    if ero and (not role or _sim(role, ero) < 0.35):
        role = ero
    elif ero and role:
        role = role  # keep article if present

    company_conflict = False
    if eco:
        if company and _sim(company, eco) < 0.4 and len(company) > 2 and len(eco) > 2:
            company_conflict = True
        company = eco

    identity_confirmed = bool(ed.get("wikipedia_url")) and _sim(str(entity.get("name") or ""), canon) >= 0.5
    identity_mismatch = company_conflict

    return {
        "name": name,
        "role": role,
        "company": company,
        "context_type": str(entity.get("context_type") or "mention"),
        "confidence": float(entity.get("confidence") or 0),
        "identity_confirmed": identity_confirmed,
        "identity_mismatch": identity_mismatch,
    }


# -----------------------------------------------------------------------------
# STEP 5 — Match score (rules; optional LLM)
# -----------------------------------------------------------------------------
_MATCH_LLM_PROMPT = """Score how strong a wealth prospect this person is for THIS article alone.
Return JSON: {{"score": <integer 0-40>, "reason": "<short>"}}

Person: {name} | {role} at {company}
Story role: {context_type}
Article excerpt:
---
{snippet}
---
"""


def _optional_llm_match_boost(resolved: dict[str, Any], article_text: str) -> int | None:
    if os.environ.get("WEALTH_SIGNALS_LLM_MATCH", "0").lower() not in ("1", "true", "yes"):
        return None
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    snippet = (article_text or "")[:3500]
    prompt = _MATCH_LLM_PROMPT.format(
        name=resolved.get("name"),
        role=resolved.get("role"),
        company=resolved.get("company"),
        context_type=resolved.get("context_type"),
        snippet=snippet,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw = (response.choices[0].message.content or "").strip()
        data = json.loads(raw)
        sc = int(data.get("score", 0))
        return max(0, min(40, sc))
    except Exception:
        return None


def score_match(
    resolved: dict[str, Any],
    article_text: str,
    enriched_data: dict[str, Any],
) -> tuple[int, list[str]]:
    """
    Contextual match 0–40: AI context tier + verb context + role + company + enrichment alignment.
    """
    reasons: list[str] = []
    t = (article_text or "").lower()
    ct = str(resolved.get("context_type") or "mention").lower()

    s = 0
    if ct == "primary":
        s += 22
        reasons.append("+22 primary")
    elif ct == "secondary":
        s += 11
        reasons.append("+11 secondary")
    else:
        reasons.append("+0 mention")

    pos_v = ("founded", "raised", "launched", "ceo of", "chief executive officer of", "chief executive of")
    neg_v = ("said", "told", "according to")
    list_v = ("including", "among", "affiliated")

    if any(v in t for v in pos_v):
        s += 12
        reasons.append("+12 deal/exec context")
    if any(v in t for v in neg_v):
        s -= 10
        reasons.append("-10 quote/attribution")
    if any(v in t for v in list_v):
        s -= 14
        reasons.append("-14 list/weak tie")

    role = str(resolved.get("role") or "").lower()
    if "founder" in role:
        s += 10
        reasons.append("+10 founder role")
    elif "ceo" in role or "chief executive" in role:
        s += 8
        reasons.append("+8 ceo role")
    elif "partner" in role:
        s += 6
        reasons.append("+6 partner")
    elif "director" in role:
        s += 4
        reasons.append("+4 director")

    co = str(resolved.get("company") or "").strip()
    if co and co.lower() not in ("unknown", "data pending"):
        known = bool(enriched_data.get("wikipedia_url")) or "opencorporates" in (enriched_data.get("sources") or [])
        if known:
            s += 8
            reasons.append("+8 company verified")
        else:
            s += 4
            reasons.append("+4 company present")
    else:
        s -= 10
        reasons.append("-10 unknown company")

    if resolved.get("identity_confirmed"):
        s += 10
        reasons.append("+10 identity confirmed")
    if resolved.get("identity_mismatch"):
        s -= 14
        reasons.append("-14 identity mismatch")

    s = max(0, min(40, s))
    llm = _optional_llm_match_boost(resolved, article_text)
    if llm is not None:
        blended = int(round((s + llm) / 2))
        s = max(0, min(40, blended))
        reasons.append(f"llm_blend->{s}")

    return s, reasons


def estimate_wealth_from_context(summary: str, role: str) -> str:
    max_usd, disp = _largest_money_usd(summary or "")
    if disp:
        return disp
    if max_usd >= 1e9:
        return f"${max_usd / 1e9:.1f}B"
    if max_usd >= 1e6:
        return f"${max_usd / 1e6:.0f}M"
    rl = (role or "").lower()
    if "founder" in rl or "ceo" in rl:
        return "$10M–$100M (est.)"
    if "partner" in rl or "director" in rl:
        return "$1M–$10M (est.)"
    return "Data pending"


def normalize_priority_score(signal_score: int, match_score: int) -> int:
    """Combined 0–100 (signal capped 60, match capped 40)."""
    return max(0, min(100, int(signal_score) + int(match_score)))


def priority_label_from_score(priority_score: int) -> str:
    if priority_score >= 90:
        return "Elite"
    if priority_score >= 75:
        return "High"
    if priority_score >= 55:
        return "Medium"
    return "Low"


# -----------------------------------------------------------------------------
# STEP 8–9 — Core processors
# -----------------------------------------------------------------------------
def _text_blob(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("raw_title") or ""),
        str(row.get("full_explanation") or ""),
        str(row.get("ai_summary") or ""),
        str(row.get("summary") or ""),
    ]
    return " ".join(p for p in parts if str(p).strip()).strip()


def _fallback_entities_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    name = str(row.get("person_name") or row.get("name") or "").strip()
    if not name:
        return []
    return [
        {
            "name": name,
            "role": str(row.get("role") or "").strip(),
            "company": str(row.get("company_name") or row.get("company") or "").strip(),
            "confidence": 0.45,
            "context_type": "primary",
        }
    ]


def _recency_ts(row: dict[str, Any]) -> pd.Timestamp:
    v = row.get("detected_at") or row.get("event_date")
    try:
        dt = pd.Timestamp(v)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return dt
    except Exception:
        return pd.Timestamp(0, tz="UTC")


def _dedupe_within_article(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        k = re.sub(r"\s+", " ", str(r.get("name") or "").lower().strip())
        if not k:
            continue
        prev = best.get(k)
        if prev is None or int(r.get("priority_score") or 0) > int(prev.get("priority_score") or 0):
            best[k] = r
    return list(best.values())


def process_article_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """
    AI-first pipeline: article signal → (optional) structured LLM candidates →
    cross-check identity → match score → priority. Broad recall: one row per candidate.
    """
    from ai_prospect_pipeline import (
        build_processed_row_core,
        classify_wealth_status,
        compute_match_score,
        cross_check_identity_and_wealth,
        extract_candidates_with_ai_cached,
        heuristic_candidates_from_row,
        infer_ownership_strength,
        priority_label_from_priority_score,
        score_article_signal,
        score_founder_wealth_creation,
        score_private_company_context,
        select_primary_actor,
    )
    from prospect_hardening import (
        coerce_display_person_name,
        is_historical_or_dead,
        is_valid_person_name,
        sanitize_role_and_company,
    )
    from prospect_tier import apply_tier_priority_adjustment, classify_prospect_tier
    from wealth_display import estimate_wealth_safely, validate_display_wealth
    from settings import SHOW_DEBUG
    from two_pass_pipeline import compute_recency_score, pass1_recency_adjustment

    summary = _text_blob(row)
    source_title = str(row.get("raw_title") or row.get("source_title") or "").strip()
    source_url = str(row.get("source_url") or "").strip()
    published_at = row.get("detected_at") or row.get("event_date")
    recency = _recency_ts(row)

    sig = score_article_signal(summary, source_title, published_at)
    sig_r = sig.get("_debug_reasons") or []
    article_economic = bool(sig.get("economic_relevance"))
    # Broad recall: when OpenAI is configured, always attempt structured extraction (no hard-drop on weak signal).
    gate = bool(os.environ.get("OPENAI_API_KEY", "").strip()) or int(sig.get("signal_score") or 0) >= 25 or article_economic

    ai_payload: dict[str, Any] | None = None
    if gate:
        ai_payload = extract_candidates_with_ai_cached(
            summary, source_title, source_url=source_url
        )
    if not ai_payload or not ai_payload.get("candidates"):
        ai_payload = heuristic_candidates_from_row(row, summary)

    candidates = [c for c in (ai_payload.get("candidates") or []) if isinstance(c, dict)]
    if not candidates:
        return []

    if not any(
        is_valid_person_name(str(c.get("name") or "").strip(), summary) for c in candidates
    ) and not ai_payload.get("_heuristic"):
        fb = heuristic_candidates_from_row(row, summary)
        candidates = [c for c in (fb.get("candidates") or []) if isinstance(c, dict)]
        ai_payload = fb

    if not candidates:
        return []

    _primary, weak_primary = select_primary_actor(candidates, sig, article_text=summary)
    if not article_economic:
        weak_primary = True

    out: list[dict[str, Any]] = []
    for c in candidates:
        name = str(c.get("name") or "").strip()
        if not name:
            continue
        if not is_valid_person_name(name, summary):
            continue
        if not hard_filter_entity(
            {
                "name": name,
                "role": c.get("role"),
                "company": c.get("company"),
            }
        ):
            continue
        if c.get("is_real_person") is False:
            continue

        role_a = str(c.get("role") or "").strip()
        co_a = str(c.get("company") or "").strip()
        ctx_t = str(c.get("context_type") or "mention")
        eco_r = str(c.get("economic_role") or "other")

        cc = cross_check_identity_and_wealth(
            name,
            co_a,
            role_a,
            article_summary=summary,
            context_type=ctx_t,
            economic_role=eco_r,
        )

        role_san, co_san, _ = sanitize_role_and_company(c, summary, cc)
        cc = {**cc, "canonical_company": co_san, "canonical_role": role_san}

        fwc = score_founder_wealth_creation(summary, c, cc)
        priv = score_private_company_context(summary, c, cc)
        founder_wealth_score = min(40, int(fwc.get("subscore") or 0) + int(priv.get("subscore") or 0))
        own_inf = infer_ownership_strength(summary, c, cc)
        cc = {**cc, "ownership_inference": own_inf}

        dead_hist = is_historical_or_dead(name, summary, cc)
        msc, m_r = compute_match_score(c, cc, summary)
        if dead_hist:
            msc = 0
            m_r = ["dead_or_historical->match_0"]

        pr_adj = pass1_recency_adjustment(published_at)
        combined_match = min(40, int(msc) + int(founder_wealth_score))
        prio = int(sig.get("signal_score") or 0) + combined_match + pr_adj
        strong_founder_ops = founder_wealth_score >= 22
        if weak_primary and not strong_founder_ops:
            prio = min(prio, 74)
        if dead_hist:
            prio = min(prio, 20)
        if not article_economic and not strong_founder_ops:
            prio = min(prio, 74)
        prio = max(0, min(100, prio))

        wstat_pre = classify_wealth_status(summary, c, cc)
        safe_w = estimate_wealth_safely(c, summary, cc, wealth_status_hint=wstat_pre)
        wstat = str(safe_w.get("wealth_status") or wstat_pre)
        est = str(safe_w.get("est_wealth") or "Data pending")
        wealth_confidence = int(safe_w.get("wealth_confidence") or 0)
        wealth_numeric_verified = bool(safe_w.get("wealth_numeric_verified"))
        vd = validate_display_wealth(
            {
                "est_wealth": est,
                "est_wealth_display": est,
                "wealth_numeric_verified": wealth_numeric_verified,
            }
        )
        est_display = str(vd.get("est_wealth_display") or est)
        wealth_numeric_verified = bool(vd.get("wealth_numeric_verified"))

        prospect_tier = classify_prospect_tier(
            c,
            {
                **cc,
                "_tier_article_summary": summary,
                "_tier_wealth_status": wstat,
                "_tier_founder_wealth_score": founder_wealth_score,
            },
        )
        prio = apply_tier_priority_adjustment(prio, prospect_tier)
        prio = max(0, min(100, prio))

        label = priority_label_from_priority_score(prio)
        if weak_primary and label in ("Elite", "High") and not strong_founder_ops:
            label = "Medium"
        if dead_hist:
            label = "Low"

        row_signal_type = str(sig.get("signal_type") or "Other")
        if int(fwc.get("subscore") or 0) >= 12 or founder_wealth_score >= 18:
            row_signal_type = "Founder Wealth Creation"

        display_name = coerce_display_person_name(name, str(cc.get("canonical_name") or ""), summary)
        if not display_name.strip() or not is_valid_person_name(display_name, summary):
            continue
        core = build_processed_row_core(
            name=display_name.strip(),
            role=str(cc.get("canonical_role") or role_a).strip(),
            company=str(cc.get("canonical_company") or co_san or co_a).strip(),
            signal_type=row_signal_type,
            signal_score=int(sig.get("signal_score") or 0),
            match_score=msc,
            priority_label=label,
            est_wealth=est_display,
            source_title=source_title,
            source_url=source_url,
            summary=summary,
            context_type=ctx_t,
            economic_role=eco_r,
            identity_confidence=float(cc.get("identity_confidence") or 0.0),
            verification_sources_used=list(cc.get("verification_sources_used") or []),
            priority_score=prio,
        )

        clean = {
            **core,
            "source": source_url or source_title,
            "_recency_ts": recency,
        }
        if SHOW_DEBUG:
            clean["_debug_signal_reasons"] = "; ".join(str(x) for x in sig_r)
            clean["_debug_match_reasons"] = "; ".join(m_r)

        legacy = {
            **row,
            **clean,
            "person_name": core["name"],
            "company_name": core["company"],
            "raw_title": source_title,
            "score": core["priority_score"],
            "priority_level": core["priority_label"],
            "est_wealth_display": est_display,
            "wealth_confidence": wealth_confidence,
            "wealth_numeric_verified": wealth_numeric_verified,
            "signal_type": core["signal_type"],
            "source_title": source_title,
            "source_url": source_url,
            "published_at": published_at,
            "recency_score": compute_recency_score(published_at),
            "wealth_status": wstat,
            "wealth_relevance": str(c.get("wealth_relevance") or "medium"),
            "article_relevance_reason": str(
                c.get("article_relevance_reason") or c.get("reason") or ""
            ),
            "article_economic_relevance": article_economic,
            "candidate_historical_dead": dead_hist,
            "founder_wealth_score": founder_wealth_score,
            "ownership_inference": own_inf,
            "wealth_evidence": str(cc.get("wealth_evidence") or "none"),
            "prospect_tier": prospect_tier,
        }
        if SHOW_DEBUG:
            legacy["debug_signal_reasons"] = clean.get("_debug_signal_reasons", "")
            legacy["debug_match_reasons"] = clean.get("_debug_match_reasons", "")
        for drop_k in (
            "ai_fa_usefulness_score",
            "prospect_quality",
            "fa_priority_debug",
        ):
            if drop_k in legacy:
                try:
                    del legacy[drop_k]
                except KeyError:
                    pass

        out.append(legacy)

    return _dedupe_within_article(out)


def process_articles(raw_articles: pd.DataFrame | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Full batch: raw article rows → ranked prospect rows (multiple per article possible).
    Sort: priority_score desc, then recency desc.
    """
    if raw_articles is None:
        return []
    if isinstance(raw_articles, pd.DataFrame):
        if raw_articles.empty:
            return []
        iterrows = raw_articles.to_dict("records")
    else:
        iterrows = list(raw_articles)

    expanded: list[dict[str, Any]] = []
    for row in iterrows:
        if not isinstance(row, dict):
            continue
        expanded.extend(process_article_row(row))

    expanded.sort(
        key=lambda r: (
            -int(r.get("priority_score") or 0),
            -float((r.get("_recency_ts") or pd.Timestamp(0, tz="UTC")).timestamp()),
        )
    )
    for r in expanded:
        r.pop("_recency_ts", None)
    return expanded


def to_clean_dataframe(processed: list[dict[str, Any]]) -> pd.DataFrame:
    """Narrow columns for export / inspection (aligned with AI pipeline output)."""
    from settings import SHOW_DEBUG

    cols = (
        "name",
        "role",
        "company",
        "signal_type",
        "signal_score",
        "match_score",
        "priority_score",
        "priority_label",
        "est_wealth",
        "wealth_status",
        "wealth_confidence",
        "wealth_numeric_verified",
        "published_at",
        "source_title",
        "source_url",
        "summary",
    )
    rows = []
    for p in processed:
        row = {c: p.get(c) for c in cols}
        if SHOW_DEBUG:
            row["debug_signal_reasons"] = p.get("debug_signal_reasons")
            row["debug_match_reasons"] = p.get("debug_match_reasons")
        rows.append(row)
    return pd.DataFrame(rows)
