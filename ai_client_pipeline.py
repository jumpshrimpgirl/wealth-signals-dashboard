"""
Stages 2–3: AI entity validation + client enrichment (financial advisory discovery).

Runs after structured extraction; scoring happens only after these stages succeed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(__file__).resolve().parent / ".cache" / "client_pipeline"

_VALIDATE_SCHEMA: dict[str, Any] = {
    "name": "entity_validate_v1",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "is_real_person": {"type": "boolean"},
            "confidence": {"type": "integer"},
            "clean_name": {"type": "string"},
            "role": {"type": "string"},
            "company": {"type": "string"},
            "is_primary_subject": {"type": "boolean"},
        },
        "required": [
            "is_real_person",
            "confidence",
            "clean_name",
            "role",
            "company",
            "is_primary_subject",
        ],
        "additionalProperties": False,
    },
}

_ENRICH_SCHEMA: dict[str, Any] = {
    "name": "client_enrich_v1",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "estimated_wealth_range": {
                "type": "string",
                "enum": ["low", "medium", "high", "ultra"],
            },
            "wealth_confidence": {"type": "integer"},
            "client_relevance": {"type": "string", "enum": ["high", "medium", "low"]},
            "reason": {"type": "string"},
        },
        "required": ["estimated_wealth_range", "wealth_confidence", "client_relevance", "reason"],
        "additionalProperties": False,
    },
}


def _client_openai() -> Any | None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=api_key)
    except ImportError:
        return None


def _model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"


def validate_person_with_ai(article_text: str, extracted_name: str) -> dict[str, Any] | None:
    """
    Stage 2: determine if name is a real individual central to the story; clean role/company.
    Returns dict or None on failure. Caller applies confidence >= 70 and primary_subject filters.
    """
    client = _client_openai()
    if not client:
        return None
    body = (article_text or "")[:20000]
    name = (extracted_name or "").strip()
    if not name:
        return None

    prompt = f"""You are validating whether a string is a REAL individual human for financial-advisory prospecting.

Article text:
---
{body}
---

Extracted name to evaluate: "{name}"

Answer:
1. Is this a REAL individual (not a region, list title, product model, organization name, concept, or event)?
2. Most likely role for THIS person in THIS article (CEO, founder, investor, etc.)?
3. What company (if any) they are ACTUALLY associated with in THIS article — do not invent; use Unknown if unclear.
4. Is this person CENTRAL to the story (primary subject) or only mentioned in passing / quote / background?

Return JSON only with: is_real_person, confidence (0-100), clean_name, role, company, is_primary_subject."""

    try:
        response = client.chat.completions.create(
            model=_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_schema", "json_schema": _VALIDATE_SCHEMA},
        )
    except Exception:
        try:
            response = client.chat.completions.create(
                model=_model(),
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
    return data


def enrich_client_with_ai(
    article_text: str,
    *,
    clean_name: str,
    role: str,
    company: str,
    validation_confidence: int,
) -> dict[str, Any] | None:
    """
    Stage 3: wealth band + client relevance for FA discovery. No exact $ unless in article text.
    """
    client = _client_openai()
    if not client:
        return None
    body = (article_text or "")[:18000]

    prompt = f"""Estimate this person's wealth band and relevance as a **financial advisory** prospect (private wealth / liquidity / planning needs).

Article text:
---
{body}
---

Person: {clean_name}
Role: {role}
Company: {company}
Validation confidence (from prior step): {validation_confidence}

Rules:
- NEVER output exact dollar net worth unless the article explicitly states the person's net worth.
- Use estimated_wealth_range: low | medium | high | ultra as bands only.
- client_relevance: high = strong reason an FA would engage now; medium; low.
- reason: one concise sentence (e.g. "Founder raised funding → liquidity and planning needs likely").

Return JSON only."""

    try:
        response = client.chat.completions.create(
            model=_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
            response_format={"type": "json_schema", "json_schema": _ENRICH_SCHEMA},
        )
    except Exception:
        try:
            response = client.chat.completions.create(
                model=_model(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.15,
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
    return data if isinstance(data, dict) else None


def _cache_path(kind: str, key: str) -> Path:
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return _CACHE_ROOT / f"{kind}_{key}.json"


def validate_person_with_ai_cached(article_text: str, extracted_name: str, source_url: str) -> dict[str, Any] | None:
    key = hashlib.sha256(f"{source_url}|{extracted_name}|v1".encode()).hexdigest()[:24]
    p = _cache_path("validate", key)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict) and "is_real_person" in d:
                return d
        except (json.JSONDecodeError, OSError):
            pass
    out = validate_person_with_ai(article_text, extracted_name)
    if out is not None:
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=0)
        except OSError:
            pass
    return out


def enrich_client_with_ai_cached(
    article_text: str,
    *,
    clean_name: str,
    role: str,
    company: str,
    validation_confidence: int,
    source_url: str,
) -> dict[str, Any] | None:
    key = hashlib.sha256(
        f"{source_url}|{clean_name}|{role}|{company}|v1".encode()
    ).hexdigest()[:24]
    p = _cache_path("enrich", key)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict) and "client_relevance" in d:
                return d
        except (json.JSONDecodeError, OSError):
            pass
    out = enrich_client_with_ai(
        article_text,
        clean_name=clean_name,
        role=role,
        company=company,
        validation_confidence=validation_confidence,
    )
    if out is not None:
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=0)
        except OSError:
            pass
    return out


def client_relevance_to_priority_boost(client_relevance: str) -> int:
    cr = (client_relevance or "").lower().strip()
    if cr == "high":
        return 12
    if cr == "medium":
        return 6
    if cr == "low":
        return 0
    return 3


def wealth_range_to_display_label(rng: str) -> str:
    m = {
        "low": "Range: low",
        "medium": "Range: medium",
        "high": "Range: high",
        "ultra": "Range: ultra-high",
    }
    return m.get((rng or "").lower().strip(), "Range: see notes")


__all__ = [
    "client_relevance_to_priority_boost",
    "enrich_client_with_ai",
    "enrich_client_with_ai_cached",
    "validate_person_with_ai",
    "validate_person_with_ai_cached",
    "wealth_range_to_display_label",
]
