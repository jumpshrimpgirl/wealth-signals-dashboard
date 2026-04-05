"""
OpenAI-based structured extraction when regex / heuristics miss a valid person name.

Requires ``OPENAI_API_KEY``. Optional ``OPENAI_MODEL`` (default ``gpt-4o-mini``).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# Canonical labels used across the app / data layer
_CANONICAL_EVENT_TYPES = frozenset(
    {
        "Founder Exit",
        "Funding",
        "Promotion",
        "Board Appointment",
        "Other",
    }
)


def _normalize_ai_event_type(raw: str) -> str | None:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    if s in _CANONICAL_EVENT_TYPES:
        return s
    key = s.lower()
    aliases = {
        "founder exit": "Founder Exit",
        "exit": "Founder Exit",
        "m&a": "Founder Exit",
        "acquisition": "Founder Exit",
        "funding": "Funding",
        "fundraise": "Funding",
        "promotion": "Promotion",
        "promoted": "Promotion",
        "board": "Board Appointment",
        "board appointment": "Board Appointment",
        "director": "Board Appointment",
        "other": "Other",
    }
    if key in aliases:
        return aliases[key]
    for prefix, canonical in (
        ("founder", "Founder Exit"),
        ("fund", "Funding"),
        ("promot", "Promotion"),
        ("board", "Board Appointment"),
    ):
        if key.startswith(prefix):
            return canonical
    return None


def extract_signal_with_ai(text: str) -> dict[str, Any]:
    """
    Call OpenAI to extract person_name, company_name, role, event_type from news text.

    Returns a dict (possibly empty on error / missing API key). Strings are stripped;
    unknown fields may be omitted or empty.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or not (text or "").strip():
        return {}

    try:
        from openai import OpenAI
    except ImportError:
        return {}

    truncated = (text or "").strip()
    try:
        max_chars = int(os.environ.get("OPENAI_EXTRACTION_MAX_CHARS", "28000"))
    except ValueError:
        max_chars = 28000
    max_chars = max(4000, min(max_chars, 100_000))
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars] + "\n…"

    prompt = f"""
Extract structured information from this news text for a financial-advisor wealth-signals feed.

Return JSON with:
- person_name (real human only; use empty string if none)
- company_name
- role
- event_type (one of: Founder Exit, Funding, Promotion, Board Appointment, Other)
- client_type (one of: Founder / Entrepreneur | Executive | Investor (PE/VC/HF) | Athlete / Celebrity | Heir / Family wealth | unknown)
- source_of_wealth (short phrase if inferable, e.g. "company sale", "IPO", "funding round", "compensation"; else empty string)

Rules:
- Only return real people (not companies, not cities)
- Ignore journalists
- Ignore vague phrases
- If unsure, leave the field blank (empty string)
- Prefer wealth-relevant roles (founder, CEO, investor) when multiple people appear

Text:
{truncated}
"""

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
    except Exception:
        return {}

    raw_content = (response.choices[0].message.content or "").strip()
    if not raw_content:
        return {}

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw_content)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}

    if not isinstance(data, dict):
        return {}

    out: dict[str, Any] = {}
    pn = str(data.get("person_name", "") or "").strip()
    out["person_name"] = pn
    out["company_name"] = str(data.get("company_name", "") or "").strip()
    out["role"] = str(data.get("role", "") or "").strip()
    et_raw = str(data.get("event_type", "") or "").strip()
    et = _normalize_ai_event_type(et_raw)
    out["event_type"] = et if et else (et_raw if et_raw in _CANONICAL_EVENT_TYPES else "Other")

    ct = str(data.get("client_type", "") or "").strip()
    out["client_type"] = ct if ct else ""
    sow = str(data.get("source_of_wealth", "") or "").strip()
    out["source_of_wealth"] = sow if sow else ""

    return out
