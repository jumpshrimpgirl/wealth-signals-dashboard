"""
High-value article detection and forced entity extraction when the primary pass is thin.

Uses :func:`wealth_signal_scoring.evaluate_signal_strength` (event types, money magnitude,
growth, institutional patterns) instead of naive keyword counts. Threshold tunable via
``WEALTH_SIGNALS_HIGH_ARTICLE_MIN_SCORE`` (minimum evaluated score; default 30 ≈ moderate+).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from wealth_signal_scoring import evaluate_signal_strength

_COMPANY_LEAD_RE = re.compile(
    r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3})\s+(?:raises|raised|acquires|acquired|merger|ipo|funding|announces)\b"
)


def is_high_value_article(article_text: str) -> bool:
    """
    True when evaluated financial signal score meets the configured floor (default: moderate band or better).
    """
    ev = evaluate_signal_strength(article_text or "")
    try:
        need = int(os.environ.get("WEALTH_SIGNALS_HIGH_ARTICLE_MIN_SCORE", "30"))
    except ValueError:
        need = 30
    return int(ev.get("score", 0)) >= need


def _openai_json_dict(prompt: str, *, temperature: float = 0.2) -> dict[str, Any] | None:
    """Single JSON object from the chat API (used for company-first steps)."""
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


def _openai_json_array(prompt: str, *, temperature: float = 0.2) -> list[dict[str, Any]] | None:
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
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        arr = data.get("actors") or data.get("entities") or data.get("people") or data.get("items")
        if isinstance(arr, list):
            return [x for x in arr if isinstance(x, dict)]
    return []


def force_extract_key_entities(article_text: str) -> list[dict[str, Any]]:
    """
    Second-pass LLM: focus on founders, executives, investors, decision-makers, deal companies.
    Returns list of {name, role, company, relevance} or [].
    """
    t = (article_text or "").strip()
    if len(t) > 32000:
        t = t[:32000] + "\n…"
    prompt = f"""Extract the MOST IMPORTANT real-world actors from this article.

Focus on:
- founders, executives, investors, decision-makers
- companies involved in deals or capital events

IGNORE:
- journalists, commentators, spokespeople (unless they are the deal principal)
- clearly historical or fictional names
- generic regions or governments unless a named executive is tied to a deal

Return ONE JSON object with a single key "actors" whose value is an array of objects.
Each object must have: "name", "role", "company", "relevance" (short string — why they matter for this story).

If no one qualifies, return {{"actors": []}}.

Article:
{t}
"""
    out = _openai_json_array(prompt, temperature=0.22)
    return out if out is not None else []


def forced_actors_to_candidates(
    actors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map forced-extraction JSON into prospect-engine candidate shape."""
    candidates: list[dict[str, Any]] = []
    for item in actors:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        candidates.append(
            {
                "person_name": name,
                "company": str(item.get("company") or "").strip(),
                "role": str(item.get("role") or "").strip(),
                "is_journalist_or_commentator": False,
                "is_primary_subject": True,
                "seniority_bucket": "c_suite",
                "ownership_hint": False,
                "proximity_rank": 4,
                "financial_signal": "hard",
                "event_type": "Other",
                "one_sentence_bio": str(item.get("relevance") or "").strip(),
                "what_happened": "",
                "why_financial": str(item.get("relevance") or "").strip(),
            }
        )
    return candidates


def _truncate_article(article_text: str, *, limit: int = 32000) -> str:
    t = (article_text or "").strip()
    if len(t) > limit:
        return t[:limit] + "\n…"
    return t


def extract_company_first(article_text: str) -> dict[str, Any] | None:
    """
    Identify the most important company in the article (company-first extraction, step 1).
    Returns dict with company, why_important, financial_signal or None.
    """
    t = _truncate_article(article_text)
    if len(t) < 40:
        return None
    prompt = f"""Identify the MOST IMPORTANT company in this article (the main subject of the business or deal story).

Return a single JSON object with exactly these keys:
- "company": string — canonical company name (not the journalist's outlet)
- "why_important": string — one sentence on why this company is central to the story
- "financial_signal": string — short phrase on the money/deal angle (funding, M&A, IPO, etc.)

If no real company is central, use empty string for "company".

Article:
{t}
"""
    return _openai_json_dict(prompt, temperature=0.2)


def extract_key_person(article_text: str, company: str) -> dict[str, Any] | None:
    """
    Given the anchor company, extract the most important person (step 2).
    """
    t = _truncate_article(article_text)
    co = (company or "").strip()
    if len(t) < 40 or not co:
        return None
    prompt = f"""The primary company in the story is: {co}

Extract the MOST IMPORTANT PERSON associated with this company in this article.

PRIORITIZE: founder, CEO, owner, or deal principal tied to that company.
IGNORE: journalists, commentators, analysts, unrelated politicians, generic "sources".

Return a single JSON object with exactly these keys:
- "name": string — full name (empty if no one qualifies)
- "role": string — title or role
- "why_relevant": string — one sentence on why they matter for this story

Article:
{t}
"""
    return _openai_json_dict(prompt, temperature=0.18)


def extract_financial_signal(article_text: str) -> dict[str, Any] | None:
    """
    Pull headline financial metrics / strength (step 3).
    """
    t = _truncate_article(article_text)
    if len(t) < 40:
        return None
    prompt = f"""Extract key financial metrics and signal strength from this article.

Return a single JSON object with exactly these keys (use empty string when unknown):
- "revenue": string
- "growth": string
- "valuation": string
- "signal_strength": string — exactly one of: strong, moderate, weak (how strong the financial story is)

Article:
{t}
"""
    return _openai_json_dict(prompt, temperature=0.15)


def process_high_value_article(article_text: str) -> dict[str, Any] | None:
    """
    Company → person → financials. Used for high-value articles when the primary extraction pass is thin.

    Returns a dict with name, role, company, signal (financial dict), priority HIGH, and supporting fields;
    None when company-first cannot anchor.
    """
    if os.environ.get("WEALTH_SIGNALS_COMPANY_FIRST_PIPELINE", "1").lower() in (
        "0",
        "false",
        "no",
    ):
        return None
    t = _truncate_article(article_text)
    if len(t) < 40:
        return None

    company_data = extract_company_first(t)
    if not company_data:
        return None
    company = str(company_data.get("company") or "").strip()
    if not company:
        return None

    person = extract_key_person(t, company)
    if not person:
        return None

    name = str(person.get("name") or "").strip()
    if not name:
        return None

    financials = extract_financial_signal(t)
    if not isinstance(financials, dict):
        financials = {}

    return {
        "name": name,
        "role": str(person.get("role") or "").strip(),
        "company": company,
        "company_why_important": str(company_data.get("why_important") or "").strip(),
        "company_financial_signal": str(company_data.get("financial_signal") or "").strip(),
        "person_why_relevant": str(person.get("why_relevant") or "").strip(),
        "signal": financials,
        "priority": "HIGH",
    }


def _map_signal_strength_to_financial_signal(signal_strength: str) -> str:
    s = (signal_strength or "").strip().lower()
    if s == "strong":
        return "hard"
    if s == "moderate":
        return "strong_trend"
    if s == "weak":
        return "weak"
    return "hard"


def company_first_result_to_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Map ``process_high_value_article`` output into prospect-engine candidate dicts (at most one)."""
    name = str(result.get("name") or "").strip()
    if not name:
        return []
    company = str(result.get("company") or "").strip()
    role = str(result.get("role") or "").strip()
    sig = result.get("signal") or {}
    if not isinstance(sig, dict):
        sig = {}
    ss = str(sig.get("signal_strength") or "").strip()
    fin = _map_signal_strength_to_financial_signal(ss)

    parts: list[str] = []
    cw = str(result.get("company_why_important") or "").strip()
    cf = str(result.get("company_financial_signal") or "").strip()
    pr = str(result.get("person_why_relevant") or "").strip()
    if cw:
        parts.append(cw)
    if cf:
        parts.append(cf)
    if pr:
        parts.append(pr)
    for label, key in (("Revenue", "revenue"), ("Valuation", "valuation"), ("Growth", "growth")):
        v = str(sig.get(key) or "").strip()
        if v:
            parts.append(f"{label}: {v}")
    rel = " — ".join(parts) if parts else "Company-first extraction (high-value article)."

    return [
        {
            "person_name": name,
            "company": company,
            "role": role,
            "is_journalist_or_commentator": False,
            "is_primary_subject": True,
            "seniority_bucket": "c_suite",
            "ownership_hint": False,
            "proximity_rank": 4,
            "financial_signal": fin,
            "event_type": "Other",
            "one_sentence_bio": rel[:500],
            "what_happened": "",
            "why_financial": rel[:500],
            "_company_first": True,
        }
    ]


def run_company_first_pipeline(article_text: str) -> list[dict[str, Any]]:
    """Run ``process_high_value_article`` and map to engine candidates (for tests / callers)."""
    r = process_high_value_article(article_text)
    if not r:
        return []
    return company_first_result_to_candidates(r)


def extract_main_company(title: str, body: str) -> str:
    """Best-effort company anchor from headline/body for structural-opportunity rows."""
    blob = f"{title} {body}".strip()[:4000]
    m = _COMPANY_LEAD_RE.search(blob)
    if m:
        return m.group(1).strip()
    m2 = re.search(
        r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,2})\s+(?:Inc\.|LLC|Ltd\.|Corp\.|Corporation|Group)\b",
        blob,
    )
    if m2:
        return m2.group(1).strip()
    return ""


def assign_signal_priority(
    *,
    signal_level: str | None = None,
    high_value_article: bool = False,
    confidence_score: int = 0,
) -> str:
    """
    Map ``evaluate_signal_strength`` level to dashboard priority: strong→HIGH, moderate→MEDIUM, weak→LOW.

    When ``signal_level`` is missing or unknown, falls back to high-value flag and confidence score.
    """
    lv = (signal_level or "").strip().lower()
    if lv == "strong":
        return "HIGH"
    if lv == "moderate":
        return "MEDIUM"
    if lv == "weak":
        return "LOW"
    if high_value_article:
        return "HIGH"
    if int(confidence_score) > 60:
        return "MEDIUM"
    return "LOW"
