"""
Post-scoring AI layer: adds advisor-facing copy only (does not change scores).

Runs on rows with ``score`` >= ``AI_INTERPRETATION_MIN_SCORE`` (default 60).
Requires ``OPENAI_API_KEY``. Set ``WEALTH_SIGNALS_AI_INTERPRETATION=0`` to disable.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

AI_INTERPRETATION_MIN_SCORE = 60


def _ai_interpretation_enabled() -> bool:
    return os.environ.get("WEALTH_SIGNALS_AI_INTERPRETATION", "1").lower() not in ("0", "false", "no")


def interpret_signal_with_ai(row: dict[str, Any]) -> dict[str, str]:
    """
    Return ``ai_summary``, ``ai_why_it_matters``, ``ai_outreach`` for one signal.

    On missing key / API error / import failure, returns three empty strings.
    """
    out = {"ai_summary": "", "ai_why_it_matters": "", "ai_outreach": ""}
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or not _ai_interpretation_enabled():
        return out

    try:
        from openai import OpenAI
    except ImportError:
        return out

    title = str(row.get("raw_title", "") or "").strip()
    body = str(row.get("full_explanation", "") or "").strip()
    blob = f"Title: {title}\n\nStory / notes:\n{body}".strip()
    if not blob:
        return out

    try:
        max_chars = int(os.environ.get("OPENAI_INTERPRETATION_MAX_CHARS", "12000"))
    except ValueError:
        max_chars = 12000
    max_chars = max(2000, min(max_chars, 100_000))
    if len(blob) > max_chars:
        blob = blob[:max_chars] + "\n…"

    context = f"""
Person: {row.get("person_name", "")}
Company: {row.get("company_name", "")}
Role: {row.get("role", "")}
Event type: {row.get("event_type", "")}
Score (informational, do not change): {row.get("score", "")}
"""
    prompt = f"""You are helping a financial advisor who focuses on high-net-worth clients (roughly $5M+ investable assets).

Read the signal below. Explain why this event is relevant for a financial advisor targeting $5M+ clients. Be specific.

Return JSON with exactly these keys (each a single string, plain text, no markdown):
- ai_summary: One concise sentence (max ~200 chars) capturing the opportunity for outreach.
- ai_why_it_matters: 2-4 sentences on why this matters for wealth/advisory context (liquidity, concentration, planning triggers, etc.).
- ai_outreach: One concrete suggested line or angle for a respectful first touch (not generic platitudes).

Signal context:
{context}

News content:
{blob}
"""

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.35,
        )
    except Exception:
        return out

    raw_content = (response.choices[0].message.content or "").strip()
    if not raw_content:
        return out

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw_content)
        if not m:
            return out
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return out

    if not isinstance(data, dict):
        return out

    for key in ("ai_summary", "ai_why_it_matters", "ai_outreach"):
        v = data.get(key, "")
        out[key] = str(v).strip() if v is not None else ""

    return out


def enrich_dataframe_with_ai_interpretation(df):  # pd.DataFrame
    """Add AI interpretation columns for rows with score >= AI_INTERPRETATION_MIN_SCORE."""
    import pandas as pd

    if df is None or df.empty:
        return df

    out = df.copy()
    for c in ("ai_summary", "ai_why_it_matters", "ai_outreach"):
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str)

    if not _ai_interpretation_enabled() or not os.environ.get("OPENAI_API_KEY", "").strip():
        return out

    try:
        mask = pd.to_numeric(out["score"], errors="coerce").fillna(0) >= AI_INTERPRETATION_MIN_SCORE
    except Exception:
        return out

    if not mask.any():
        return out

    for idx in out.loc[mask].index:
        row = out.loc[idx]
        try:
            d = interpret_signal_with_ai(row.to_dict())
        except Exception:
            continue
        for k in ("ai_summary", "ai_why_it_matters", "ai_outreach"):
            if k in d and d[k]:
                out.at[idx, k] = d[k]

    for c in ("ai_summary", "ai_why_it_matters", "ai_outreach"):
        out[c] = out[c].fillna("").astype(str).str.strip()

    return out
