"""
OpenAI-based structured extraction for ingest (multi-pass, field-level).

Requires ``OPENAI_API_KEY``. Optional ``OPENAI_MODEL`` (default ``gpt-4o-mini``).
Uses ``structured_extraction`` for pass 1 + optional pass 2.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from structured_extraction import apply_display_fallbacks, merge_regex_and_structured, run_structured_extraction

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


def extract_signal_with_ai(
    text: str,
    *,
    regex_person: str = "",
    regex_company: str = "",
    regex_role: str = "",
    regex_event_type: str = "",
) -> dict[str, Any]:
    """
    Multi-pass structured extraction with per-field provenance and optional gap-fill pass 2.

    Returns a dict including legacy keys (``person_name``, ``company_name``, ``role``, …)
    plus ``extraction_audit_json``, ``why_it_matters``, wealth/liquidity hints, and display fallbacks.
    """
    if not (text or "").strip():
        return {}

    struct = run_structured_extraction(
        text,
        hints={
            "regex_person_guess": regex_person or "",
            "regex_company_guess": regex_company or "",
            "regex_role_guess": regex_role or "",
            "regex_event_type_guess": regex_event_type or "",
        },
    )
    merged = merge_regex_and_structured(regex_person, regex_company, regex_role, struct)
    fb = apply_display_fallbacks(struct)

    pn = str(merged.get("person_name_merged") or struct.get("person_name") or "").strip()
    cn = str(merged.get("company_name_merged") or struct.get("company_name") or "").strip()
    rl = str(merged.get("role_merged") or struct.get("role") or "").strip()

    et_raw = str(struct.get("event_type", "") or "").strip()
    et = _normalize_ai_event_type(et_raw)
    event_type = et if et else (et_raw if et_raw in _CANONICAL_EVENT_TYPES else "")

    ct = str(struct.get("client_type", "") or "").strip()
    sow = str(struct.get("source_of_wealth", "") or "").strip()
    why = str(struct.get("why_it_matters", "") or "").strip()

    audit = struct.get("_audit") if isinstance(struct.get("_audit"), dict) else {}
    audit_full = {
        "ingest": audit,
        "display": {
            "person_name": fb.get("person_name_display"),
            "role": fb.get("role_display"),
            "company_name": fb.get("company_name_display"),
            "wealth_signal": fb.get("wealth_signal_display"),
            "liquidity_event": fb.get("liquidity_event_display"),
            "source_of_wealth": fb.get("source_of_wealth_display"),
            "client_type": fb.get("client_type_display"),
            "estimated_wealth_note": fb.get("estimated_wealth_display"),
        },
        "liquidity_normalized": fb.get("liquidity_normalized"),
        "wealth_signal_for_rules": fb.get("wealth_signal_for_rules"),
        "wealth_signal_raw": fb.get("wealth_signal_raw"),
    }

    out: dict[str, Any] = {
        "person_name": pn,
        "company_name": cn,
        "role": rl,
        "event_type": event_type,
        "client_type": ct,
        "source_of_wealth": sow,
        "why_it_matters": why,
        "wealth_signal_hint": str(fb.get("wealth_signal_for_rules") or ""),
        "wealth_signal_raw_hint": str(fb.get("wealth_signal_raw") or ""),
        "liquidity_hint": str(fb.get("liquidity_normalized") or ""),
        "estimated_wealth_note": str(struct.get("estimated_wealth_note") or "").strip(),
        "overall_extraction_confidence": str(struct.get("overall_confidence") or "low"),
        "extraction_audit_json": json.dumps(audit_full, ensure_ascii=False)[:16000],
    }

    return out
