"""
Post-scoring AI layer: advisor-facing wealth-signal copy (does not change core scores).

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
    Return advisor-facing strings for one signal, focused on wealth / liquidity / client identity.

    On missing key / API error / import failure, returns empty strings for all keys.
    """
    out = {
        "ai_summary": "",
        "ai_why_it_matters": "",
        "ai_outreach": "",
        "ai_wealth_signal": "",
        "ai_liquidity_label": "",
        "ai_client_who": "",
        "ai_why_money": "",
    }
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
Rule-based wealth signal (hint): {row.get("wealth_signal_label", "")}
Liquidity (hint): {row.get("liquidity_event", "")}
Client type (hint): {row.get("client_type", "")}
Score (informational): {row.get("score", "")}
"""
    prompt = f"""You are helping a private banker / financial advisor find high-net-worth client opportunities.

This is NOT general news ranking. Judge only whether the story signals money: existing wealth, new liquidity, or imminent monetization for an identifiable person.

Return JSON with exactly these keys (each a single plain-text string, no markdown):
- ai_summary: One sentence: the wealth opportunity in advisor language (max ~220 chars).
- ai_why_it_matters: 2-3 sentences: liquidity, concentration, planning triggers, tax, or succession — money-focused.
- ai_outreach: One respectful first-touch line (not generic congratulations).
- ai_wealth_signal: Exactly one of: Strong | Moderate | Weak | None
- ai_liquidity_label: Exactly one of: Yes | No | Potential
- ai_client_who: Who is the prospective client — "Name, Role" or best available; if unclear say "Unknown individual".
- ai_why_money: Exactly one sentence: why this matters for money (not politics, not outlet prestige).

Do NOT rank or praise the news outlet. BBC/Reuters/etc. credibility does not increase client value.

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

    for key in (
        "ai_summary",
        "ai_why_it_matters",
        "ai_outreach",
        "ai_wealth_signal",
        "ai_liquidity_label",
        "ai_client_who",
        "ai_why_money",
    ):
        v = data.get(key, "")
        out[key] = str(v).strip() if v is not None else ""

    return out


def enrich_dataframe_with_ai_interpretation(df):  # pd.DataFrame
    """Add AI interpretation columns for rows with score >= AI_INTERPRETATION_MIN_SCORE."""
    import pandas as pd

    if df is None or df.empty:
        return df

    out = df.copy()
    for c in (
        "ai_summary",
        "ai_why_it_matters",
        "ai_outreach",
        "ai_wealth_signal",
        "ai_liquidity_label",
        "ai_client_who",
        "ai_why_money",
    ):
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
        for k in (
            "ai_summary",
            "ai_why_it_matters",
            "ai_outreach",
            "ai_wealth_signal",
            "ai_liquidity_label",
            "ai_client_who",
            "ai_why_money",
        ):
            if k in d and d[k]:
                out.at[idx, k] = d[k]

    for c in (
        "ai_summary",
        "ai_why_it_matters",
        "ai_outreach",
        "ai_wealth_signal",
        "ai_liquidity_label",
        "ai_client_who",
        "ai_why_money",
    ):
        out[c] = out[c].fillna("").astype(str).str.strip()

    return out
