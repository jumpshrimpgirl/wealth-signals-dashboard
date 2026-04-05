"""
Multi-pass structured prospect extraction for RSS ingest.

Optimized for **maximum credible recall**: fill every field you can support from the text,
including reasonable inferences—then label uncertainty with provenance + confidence, not silence.
Pass 2 tightens gaps when the first pass left important fields weak or empty.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

_CRITICAL_FIELDS = (
    "person_name",
    "role",
    "company_name",
    "event_type",
    "wealth_signal",
    "liquidity_event",
    "source_of_wealth",
    "client_type",
    "estimated_wealth_note",
    "why_it_matters",
)

_MISSING_REASONS = frozenset(
    {
        "not_mentioned",
        "not_stated_in_article",
        "title_only_reference",
        "ambiguous_entity",
        "insufficient_evidence",
        "extraction_failed",
        "unknown",
    }
)


def _openai_json(prompt: str, *, temperature: float = 0.15) -> dict[str, Any] | None:
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


def _norm_conf(v: str) -> str:
    s = (v or "").strip().lower()
    if s in ("high", "medium", "low"):
        return s
    return "low"


def _norm_prov(v: str) -> str:
    s = (v or "").strip().lower()
    if s in ("explicit", "inferred", "unknown"):
        return s
    return "unknown"


def _norm_missing(v: str) -> str:
    s = (v or "").strip().lower().replace(" ", "_")
    if s in _MISSING_REASONS:
        return s
    return "unknown"


def _empty_field_meta() -> dict[str, Any]:
    return {
        "provenance": "unknown",
        "confidence": "low",
        "missing_reason": "unknown",
    }


def _pass1_prompt(text: str, hints: dict[str, str]) -> str:
    h = "\n".join(f"- {k}: {v}" for k, v in hints.items() if str(v).strip())
    return f"""You extract wealth-prospecting fields from news text for financial advisors.

Philosophy: **extract the maximum credible structured information.** Do not stay silent to be "safe."
- Fill fields using **explicit statements** when possible; use **reasonable inference** when the text strongly suggests identity, role, company, deal type, or money context (set provenance to "inferred" and confidence to "low" or "medium").
- Use empty string **only** when there is no defensible anchor in the text—not merely because a detail is implied rather than quoted.
- **Never fabricate** precise dollar net worth; for wealth you may give **qualitative** notes (e.g. "Likely significant equity from Series C context — inference only") with provenance "inferred" and confidence "low".

Return ONE JSON object with this EXACT shape (all keys required):

{{
  "person_name": "full name of a real individual OR empty string",
  "role": "job title or office OR empty",
  "company_name": "primary company/employer OR empty (not a person's name)",
  "wealth_signal": "Strong" | "Moderate" | "Weak" | "None" | "Potential",
    (use "Potential" when money relevance is plausible but not explicit),
  "liquidity_event": "Yes" | "No" | "Potential" | "Unclear",
  "source_of_wealth": "short phrase OR empty",
  "client_type": "Founder / Entrepreneur" | "Executive" | "Investor (PE/VC/HF)" | "Athlete / Celebrity" | "Heir / Family wealth" | "unknown",
  "event_type": "Founder Exit" | "Funding" | "Promotion" | "Board Appointment" | "Other",
  "estimated_wealth_note": "Prefer a short credible note: explicit amounts if stated; else qualitative inference (e.g. 'Substantial equity likely — founder at post-unicorn company — inference only') or 'Not stated in article' / 'Data pending' only when no wealth clues exist",
  "why_it_matters": "ONE plain-English sentence: why this may or may not matter for FA prospecting (never empty — if useless, say so)",
  "overall_confidence": "high" | "medium" | "low",

  "person_name_provenance": "explicit" | "inferred" | "unknown",
  "person_name_confidence": "high" | "medium" | "low",
  "person_name_missing_reason": "not_mentioned" | "title_only_reference" | "ambiguous_entity" | "insufficient_evidence" | "not_stated_in_article" | "extraction_failed" | "unknown",

  "role_provenance": "explicit" | "inferred" | "unknown",
  "role_confidence": "high" | "medium" | "low",
  "role_missing_reason": "(same enum as person_name_missing_reason)",

  "company_name_provenance": "explicit" | "inferred" | "unknown",
  "company_name_confidence": "high" | "medium" | "low",
  "company_name_missing_reason": "(same enum)",

  "wealth_signal_provenance": "explicit" | "inferred" | "unknown",
  "wealth_signal_confidence": "high" | "medium" | "low",
  "wealth_signal_missing_reason": "(same enum)",

  "liquidity_event_provenance": "explicit" | "inferred" | "unknown",
  "liquidity_event_confidence": "high" | "medium" | "low",
  "liquidity_event_missing_reason": "(same enum)",

  "source_of_wealth_provenance": "explicit" | "inferred" | "unknown",
  "source_of_wealth_confidence": "high" | "medium" | "low",
  "source_of_wealth_missing_reason": "(same enum)",

  "client_type_provenance": "explicit" | "inferred" | "unknown",
  "client_type_confidence": "high" | "medium" | "low",
  "client_type_missing_reason": "(same enum)",

  "event_type_provenance": "explicit" | "inferred" | "unknown",
  "event_type_confidence": "high" | "medium" | "low",
  "event_type_missing_reason": "(same enum)",

  "estimated_wealth_note_provenance": "explicit" | "inferred" | "unknown",
  "estimated_wealth_note_confidence": "high" | "medium" | "low",
  "estimated_wealth_note_missing_reason": "(same enum)",

  "why_it_matters_provenance": "explicit" | "inferred" | "unknown",
  "why_it_matters_confidence": "high" | "medium" | "low",
  "why_it_matters_missing_reason": "(same enum)"
}}

Rules:
- Never put job titles alone in person_name (CEO alone is wrong; "CEO Jane Doe" → person_name "Jane Doe").
- If only a role/office is named with no person, leave person_name empty and describe in missing_reason.
- why_it_matters must always be a non-empty sentence.
- When money is **plausible but not explicit**, prefer wealth_signal "Potential" or "Weak" and liquidity_event "Potential" or "Unclear" over "None" / "No".
- Infer client_type from role when clear (e.g. founder → Founder / Entrepreneur; PE partner → Investor).

Pipeline hints (may be wrong):
{h}

Text:
{text}
"""


def _pass2_prompt(text: str, partial: dict[str, Any], missing: list[str]) -> str:
    miss = ", ".join(missing)
    partial_json = json.dumps(partial, indent=2)[:12000]
    return f"""Pass 1 left gaps. Your job is to **aggressively** fill: {miss}

Re-read the full article. Infer defensible values from context (company stage, titles, deal verbs, funding language, sports/entertainment contracts).
- Use provenance "inferred" and confidence "low" or "medium" when you are not quoting directly.
- Do not invent specific dollar net worth; qualitative estimates are OK with "inference only" wording.
- Return ONE complete JSON object with the SAME keys as pass 1. You may ONLY **change** keys related to the gaps above; copy all other keys unchanged from the partial JSON.

Partial JSON:
{partial_json}

Full text:
{text}
"""


def _count_gaps(data: dict[str, Any]) -> int:
    n = 0
    for k in (
        "person_name",
        "role",
        "company_name",
        "event_type",
        "source_of_wealth",
        "client_type",
    ):
        v = str(data.get(k, "") or "").strip()
        if not v or v.lower() in ("unknown", "n/a", "none"):
            n += 1
    ws = str(data.get("wealth_signal", "") or "").strip()
    if not ws or ws.lower() in ("none", "unknown"):
        n += 1
    liq = str(data.get("liquidity_event", "") or "").strip()
    if not liq or liq.lower() == "unclear":
        n += 1
    why = str(data.get("why_it_matters", "") or "").strip()
    if len(why) < 12:
        n += 1
    return n


def _pass2_enabled() -> bool:
    return os.environ.get("WEALTH_SIGNALS_EXTRACTION_PASS2", "1").lower() not in ("0", "false", "no")


def _pass2_threshold() -> int:
    """Run pass 2 when gap count >= this (default 1 = almost always tighten if any gap)."""
    try:
        return int(os.environ.get("WEALTH_SIGNALS_EXTRACTION_PASS2_GAP_THRESHOLD", "1"))
    except ValueError:
        return 1


def run_structured_extraction(
    text: str,
    *,
    hints: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Run pass 1 (+ optional pass 2). Returns flat dict with values + ``_audit`` nested dict.
    On API failure returns minimal defaults with empty ``_audit``.
    """
    hints = hints or {}
    t = (text or "").strip()
    if not t:
        return _fallback_empty()

    try:
        max_c = int(os.environ.get("OPENAI_EXTRACTION_MAX_CHARS", "28000"))
    except ValueError:
        max_c = 28000
    max_c = max(4000, min(max_c, 100_000))
    if len(t) > max_c:
        t = t[:max_c] + "\n…"

    data = _openai_json(_pass1_prompt(t, hints), temperature=0.22)
    if not data:
        return _fallback_empty()

    data = _normalize_payload(data)
    pass2_used = False

    if _pass2_enabled() and _count_gaps(data) >= _pass2_threshold():
        missing = []
        for k in _CRITICAL_FIELDS:
            if k == "estimated_wealth_note":
                v = str(data.get(k, "") or "").strip()
                if not v or v.lower() == "data pending":
                    missing.append(k)
            elif k == "why_it_matters":
                if len(str(data.get(k, "") or "").strip()) < 12:
                    missing.append(k)
            else:
                v = str(data.get(k, "") or "").strip()
                if not v:
                    missing.append(k)
        if missing:
            p2 = _openai_json(_pass2_prompt(t, data, missing), temperature=0.28)
            if p2:
                p2n = _normalize_payload(p2)
                for key in data:
                    if key.endswith("_provenance") or key.endswith("_confidence") or key.endswith("_missing_reason"):
                        continue
                    if key in p2n and str(p2n.get(key, "")).strip():
                        data[key] = p2n[key]
                for key in p2n:
                    if key.endswith("_provenance") or key.endswith("_confidence") or key.endswith("_missing_reason"):
                        data[key] = p2n[key]
                pass2_used = True

    audit = _build_audit(data, pass2_used)
    data["_audit"] = audit
    return data


def _normalize_payload(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out["person_name"] = str(out.get("person_name", "") or "").strip()
    out["role"] = str(out.get("role", "") or "").strip()
    out["company_name"] = str(out.get("company_name", "") or "").strip()
    out["wealth_signal"] = str(out.get("wealth_signal", "") or "").strip()
    out["liquidity_event"] = str(out.get("liquidity_event", "") or "").strip()
    out["source_of_wealth"] = str(out.get("source_of_wealth", "") or "").strip()
    out["client_type"] = str(out.get("client_type", "") or "").strip()
    out["event_type"] = str(out.get("event_type", "") or "").strip()
    out["estimated_wealth_note"] = str(out.get("estimated_wealth_note", "") or "").strip()
    out["why_it_matters"] = str(out.get("why_it_matters", "") or "").strip()
    out["overall_confidence"] = _norm_conf(str(out.get("overall_confidence", "low")))

    for base in (
        "person_name",
        "role",
        "company_name",
        "wealth_signal",
        "liquidity_event",
        "source_of_wealth",
        "client_type",
        "event_type",
        "estimated_wealth_note",
        "why_it_matters",
    ):
        out[f"{base}_provenance"] = _norm_prov(str(out.get(f"{base}_provenance", "unknown")))
        out[f"{base}_confidence"] = _norm_conf(str(out.get(f"{base}_confidence", "low")))
        out[f"{base}_missing_reason"] = _norm_missing(str(out.get(f"{base}_missing_reason", "unknown")))

    if not out["why_it_matters"]:
        out["why_it_matters"] = (
            "Insufficient detail in the text to assess FA prospecting value; treat as low signal until confirmed."
        )
        out["why_it_matters_provenance"] = "inferred"
        out["why_it_matters_confidence"] = "low"
        out["why_it_matters_missing_reason"] = "insufficient_evidence"

    return out


def _build_audit(data: dict[str, Any], pass2_used: bool) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for base in (
        "person_name",
        "role",
        "company_name",
        "wealth_signal",
        "liquidity_event",
        "source_of_wealth",
        "client_type",
        "event_type",
        "estimated_wealth_note",
        "why_it_matters",
    ):
        fields[base] = {
            "value": str(data.get(base, "") or ""),
            "provenance": data.get(f"{base}_provenance", "unknown"),
            "confidence": data.get(f"{base}_confidence", "low"),
            "missing_reason": data.get(f"{base}_missing_reason", "unknown"),
        }
    return {
        "fields": fields,
        "pass2_used": pass2_used,
        "overall_confidence": data.get("overall_confidence", "low"),
    }


def _fallback_empty() -> dict[str, Any]:
    data: dict[str, Any] = {
        "person_name": "",
        "role": "",
        "company_name": "",
        "event_type": "",
        "wealth_signal": "",
        "liquidity_event": "",
        "source_of_wealth": "",
        "client_type": "",
        "estimated_wealth_note": "Data pending",
        "why_it_matters": "Extraction did not run or failed; no structured assessment available.",
        "overall_confidence": "low",
    }
    for base in (
        "person_name",
        "role",
        "company_name",
        "wealth_signal",
        "liquidity_event",
        "source_of_wealth",
        "client_type",
        "event_type",
        "estimated_wealth_note",
        "why_it_matters",
    ):
        data[f"{base}_provenance"] = "unknown"
        data[f"{base}_confidence"] = "low"
        data[f"{base}_missing_reason"] = "extraction_failed"
    data["_audit"] = {"fields": {}, "pass2_used": False, "overall_confidence": "low", "error": "extraction_failed"}
    return data


def apply_display_fallbacks(struct: dict[str, Any]) -> dict[str, Any]:
    """
    Map model output to display-safe strings (no silent blanks for key columns).
    Returns keys compatible with ``extract_signal_with_ai`` consumers plus extras.
    """
    d = dict(struct)
    pn = str(d.get("person_name", "") or "").strip()
    rl = str(d.get("role", "") or "").strip()
    co = str(d.get("company_name", "") or "").strip()
    ws = str(d.get("wealth_signal", "") or "").strip()
    liq = str(d.get("liquidity_event", "") or "").strip()
    sow = str(d.get("source_of_wealth", "") or "").strip()
    ct = str(d.get("client_type", "") or "").strip()
    ewn = str(d.get("estimated_wealth_note", "") or "").strip()
    why = str(d.get("why_it_matters", "") or "").strip()

    d["person_name_display"] = pn if pn else "Not identified"
    d["role_display"] = rl if rl else "Not identified"
    d["company_name_display"] = co if co else "Not identified"
    d["wealth_signal_display"] = ws if ws else "Data pending"
    d["liquidity_event_display"] = liq if liq else "No clear liquidity event"
    d["source_of_wealth_display"] = sow if sow else "Not stated in article"
    d["client_type_display"] = ct if ct and ct.lower() != "unknown" else "Data pending"
    d["estimated_wealth_display"] = ewn if ewn else "Data pending"
    d["why_it_matters_display"] = why if why else "Not stated in article"

    # Normalize liquidity for downstream (score.py expects Yes/No/Potential)
    liq_l = liq.lower()
    if liq_l in ("unclear", "potential"):
        d["liquidity_normalized"] = "Potential"
    elif liq_l == "yes":
        d["liquidity_normalized"] = "Yes"
    elif liq_l == "no":
        d["liquidity_normalized"] = "No"
    else:
        d["liquidity_normalized"] = "Potential" if not liq else "Potential"

    # wealth_signal: Potential -> Weak for rule columns unless None
    ws_u = ws.strip()
    if ws_u.lower() == "potential":
        d["wealth_signal_for_rules"] = "Weak"
        d["wealth_signal_raw"] = "Potential"
    elif ws_u in ("Strong", "Moderate", "Weak", "None"):
        d["wealth_signal_for_rules"] = ws_u
        d["wealth_signal_raw"] = ws_u
    else:
        d["wealth_signal_for_rules"] = "Weak" if ws_u else "None"
        d["wealth_signal_raw"] = ws_u or "Data pending"

    return d


def merge_regex_and_structured(
    regex_person: str,
    regex_company: str,
    regex_role: str,
    structured: dict[str, Any],
) -> dict[str, Any]:
    """Prefer structured values when present and non-placeholder; else regex."""
    fb = apply_display_fallbacks(structured)
    out = dict(fb)

    def use(primary: str, fallback: str, placeholders: frozenset[str]) -> str:
        p = str(primary or "").strip()
        if p and p.lower() not in placeholders:
            return p
        return str(fallback or "").strip()

    ph_name = frozenset(("", "not identified", "n/a"))
    out["person_name_merged"] = use(structured.get("person_name", ""), regex_person, ph_name)
    out["company_name_merged"] = use(structured.get("company_name", ""), regex_company, ph_name | frozenset({"unknown"}))
    out["role_merged"] = use(structured.get("role", ""), regex_role, ph_name)

    if not out["person_name_merged"] and regex_person:
        out["person_name_merged"] = regex_person
    if not out["company_name_merged"] and regex_company:
        out["company_name_merged"] = regex_company
    if not out["role_merged"] and regex_role:
        out["role_merged"] = regex_role

    return out
