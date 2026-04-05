"""
Validate and standardize prospect dicts (name/role/company, priority, confidence, wealth estimate).
Designed for dashboard rows from ``DataFrame.to_dict('records')`` with column aliases.
"""

from __future__ import annotations

import json
import re
from typing import Any

from prospect_scoring import apply_prospect_scores

_MONEY_EST = re.compile(
    r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|bn|m\b|b\b)?",
    re.I,
)


def _prospect_text_blob(p: dict[str, Any]) -> str:
    parts = [
        str(p.get("summary") or ""),
        str(p.get("raw_title") or ""),
        str(p.get("full_explanation") or ""),
        str(p.get("ai_summary") or ""),
    ]
    return " ".join(x for x in parts if str(x).strip()).strip()


def _priority_from_row(p: dict[str, Any]) -> str:
    pr = str(p.get("priority") or "").strip().upper()
    if pr in ("HIGH", "MEDIUM", "LOW"):
        return pr
    sp = str(p.get("signal_priority") or "").strip().upper()
    if sp in ("HIGH", "MEDIUM", "LOW"):
        return sp
    ej = p.get("extraction_audit_json")
    if ej:
        try:
            audit = json.loads(ej) if isinstance(ej, str) else ej
            if isinstance(audit, dict):
                sp = str(audit.get("signal_priority") or "").strip().upper()
                if sp in ("HIGH", "MEDIUM", "LOW"):
                    return sp
        except (json.JSONDecodeError, TypeError):
            pass
    pl = str(p.get("priority_level") or "").strip().lower()
    if pl == "high":
        return "HIGH"
    if pl == "medium":
        return "MEDIUM"
    if pl == "low":
        return "LOW"
    return "LOW"


def _normalize_unit(unit: str | None) -> str:
    if not unit:
        return ""
    u = unit.strip().lower()
    if u in ("billion", "bn", "b"):
        return "billion"
    if u in ("million", "m"):
        return "million"
    return ""


def estimate_wealth(p: dict[str, Any]) -> str:
    text = _prospect_text_blob(p).lower()

    for m in _MONEY_EST.finditer(text):
        raw = m.group(1)
        unit_raw = m.group(2)
        try:
            val = float(str(raw).replace(",", ""))
        except (ValueError, TypeError):
            continue
        unit = _normalize_unit(unit_raw)
        if unit == "billion":
            return f"${val:.1f}B"
        if unit == "million":
            return f"${val:.0f}M"

    role = (p.get("role") or "").lower()
    if "founder" in role or "ceo" in role:
        return "$10M–$100M"
    if "partner" in role or "director" in role:
        return "$1M–$10M"

    return "Data pending"


def is_valid_prospect_row(p: dict[str, Any]) -> bool:
    name = (p.get("name") or p.get("person_name") or "").strip()
    role = (p.get("role") or "").lower()
    company = (p.get("company") or p.get("company_name") or "").strip()
    text = _prospect_text_blob(p).lower()

    bad_terms = [
        "department",
        "agency",
        "licensing",
        "government",
        "ministry",
        "committee",
        "region",
        "middle east",
        "technology business",
    ]
    if any(x in name.lower() for x in bad_terms):
        return False

    if len(name.split()) < 2:
        return False

    if "bot" in text or "named after" in text:
        return False

    if any(x in text for x in ["died", "historian", "former secretary"]):
        return False

    valid_roles = [
        "founder",
        "ceo",
        "co-founder",
        "partner",
        "director",
        "investor",
    ]
    if not any(r in role for r in valid_roles):
        return False

    if not company or company.lower() in ("unknown", "data pending", "not identified"):
        return False

    return True


def clean_and_standardize(prospects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return validated, schema-aligned prospect dicts (empty list if none pass)."""
    cleaned: list[dict[str, Any]] = []

    for p in prospects:
        canon = {
            "name": (p.get("name") or p.get("person_name") or "").strip(),
            "role": (p.get("role") or "").strip(),
            "company": (p.get("company") or p.get("company_name") or "").strip(),
            "summary": _prospect_text_blob(p),
            "signal_type": str(
                p.get("signal_type")
                or p.get("event_type")
                or p.get("wealth_signal_label")
                or "Other"
            ).strip()
            or "Other",
            "priority": _priority_from_row(p),
            "confidence": p.get("confidence", p.get("confidence_score", 0)),
        }
        if not is_valid_prospect_row(canon):
            continue
        cleaned.append(
            {
                "name": canon["name"],
                "role": canon["role"],
                "company": canon["company"],
                "signal_type": canon["signal_type"],
                "priority": canon["priority"],
                "summary": canon["summary"],
                "est_wealth": estimate_wealth(canon),
            }
        )

    return apply_prospect_scores(cleaned)
