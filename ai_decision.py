"""
AI decision layer for wealth prospecting: structured extraction, classification, usefulness, rerank, clustering.

Requires OPENAI_API_KEY. Disable with WEALTH_SIGNALS_AI_DECISION=0.
Runs after rule-based scoring (see fetch_signals). Complements rule-based columns; can refine labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

AI_DECISION_MIN_SCORE = 35
AI_RERANK_MAX_ROWS = 40


def _enabled() -> bool:
    return os.environ.get("WEALTH_SIGNALS_AI_DECISION", "1").lower() not in ("0", "false", "no")


def _rerank_enabled() -> bool:
    return os.environ.get("WEALTH_SIGNALS_AI_RERANK", "1").lower() not in ("0", "false", "no")


def _min_score() -> int:
    try:
        return int(os.environ.get("AI_DECISION_MIN_SCORE", str(AI_DECISION_MIN_SCORE)))
    except ValueError:
        return AI_DECISION_MIN_SCORE


def _empty_decision_row() -> dict[str, Any]:
    return {
        "extracted_person_name": "",
        "extracted_role": "",
        "extracted_company": "",
        "client_type": "",
        "wealth_signal": "",
        "liquidity_event": "",
        "source_of_wealth": "",
        "estimated_wealth_usd": None,
        "wealth_estimate_confidence": "none",
        "extraction_confidence_score": 0,
        "prospect_quality": "",
        "fa_usefulness_score": 0,
        "why_flagged": "",
        "why_matters_for_advisor": "",
        "ai_summary": "",
        "ai_why_it_matters": "",
        "ai_outreach": "",
        "ai_client_who": "",
        "ai_why_money": "",
        "cluster_fingerprint": "",
    }


def _copy_narrative_from_data(data: dict[str, Any], out: dict[str, Any]) -> None:
    for k in ("ai_summary", "ai_why_it_matters", "ai_outreach", "ai_client_who", "ai_why_money"):
        v = data.get(k)
        if v is not None and str(v).strip():
            out[k] = str(v).strip()


def analyze_prospect_article(row: dict[str, Any]) -> dict[str, Any]:
    """
    Single structured analysis for one article: extraction, wealth class, prospect quality, explainability.

    Fills the same narrative keys as ai_interpretation plus decision-layer fields.
    """
    out = _empty_decision_row()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or not _enabled():
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
        max_chars = int(os.environ.get("AI_DECISION_MAX_CHARS", "14000"))
    except ValueError:
        max_chars = 14000
    max_chars = max(2000, min(max_chars, 100_000))
    if len(blob) > max_chars:
        blob = blob[:max_chars] + "\n…"

    hints = f"""
Existing pipeline hints (may be wrong — verify from text):
- person_name: {row.get("person_name", "")}
- role: {row.get("role", "")}
- company_name: {row.get("company_name", "")}
- event_type: {row.get("event_type", "")}
- rule wealth_signal: {row.get("wealth_signal_label", "")}
- rule liquidity: {row.get("liquidity_event", "")}
- rule client_type: {row.get("client_type", "")}
- score: {row.get("score", "")}
"""

    prompt = f"""You are an intelligent wealth-prospecting assistant for financial advisors.

**Goal:** Extract the **maximum credible** structured value from the article. Prefer filling fields with defensible inference over leaving them empty—then label uncertainty (confidence, provenance in your wording).

Read the article and return ONE JSON object with these keys (strict types):

STRING fields (use "" only when no defensible anchor exists):
- extracted_person_name: primary individual prospect full name if identifiable (infer from bylines/titles when strong), else ""
- extracted_role: their role/title if stated or clearly inferable, else ""
- extracted_company: primary company if any or strongly implied, else ""
- client_type: one of: Founder / Entrepreneur | Executive | Investor (PE/VC/HF) | Athlete / Celebrity | Heir / Family wealth | Other | Unknown
- wealth_signal: exactly one of: Strong | Moderate | Weak | None
  (use Moderate/Weak when money relevance is plausible from role, deal, or context even if not quantified)
- liquidity_event: exactly one of: Yes | No | Potential
- source_of_wealth: short phrase; infer likely channel when cues exist (e.g. "likely equity from funding context — inference")
- why_flagged: one short sentence: why this appeared as a wealth-relevant prospecting lead
- why_matters_for_advisor: one short sentence: money or relationship value for an FA (not outlet prestige)
- ai_summary: one sentence opportunity summary for an FA (max ~240 chars)
- ai_why_it_matters: 2-3 sentences, money-focused (liquidity, tax, planning)
- ai_outreach: one respectful first-touch line
- ai_client_who: "Name — Role" or best available; describe partial identity if that is all the text supports
- ai_why_money: one sentence on why money matters here
- cluster_fingerprint: stable lowercase slug: normalize primary person + company + event gist, e.g. "jane-doe|acme|series-b-2026" or "" if no person

NUMBER or null:
- estimated_wealth_usd: USD number when the text supports an **order-of-magnitude** (explicit deal size, funding round, contract value, or strong contextual bracket). Use null if only vague richness with no scale.
- wealth_estimate_confidence: exactly one of: none | low | medium | high
  (use "low" or "medium" when inferring from deal/role context; "none" only when no scale exists at all)

INTEGERS 0-100:
- extraction_confidence_score: how much credible structured information you extracted (can be higher when inference is solid)
- fa_usefulness_score: how useful this row is for an FA seeking wealthy prospects (0-100)

STRING prospect_quality — exactly one of:
- Excellent prospect | Possible prospect | Low-value prospect | Not actionable

Rules:
- Do **not** fabricate precise net worth with no textual basis; **do** infer band/scale when funding, M&A, compensation, or contract amounts are stated or clearly implied.
- General politics, war, macro without any targetable person or money hook → wealth_signal "None", prospect_quality "Not actionable", fa_usefulness_score under 25.
- News outlet fame does not increase scores.

{hints}

Article:
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

    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        return out
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return out
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return out

    if not isinstance(data, dict):
        return out

    for k in (
        "extracted_person_name",
        "extracted_role",
        "extracted_company",
        "client_type",
        "wealth_signal",
        "liquidity_event",
        "source_of_wealth",
        "why_flagged",
        "why_matters_for_advisor",
        "ai_summary",
        "ai_why_it_matters",
        "ai_outreach",
        "ai_client_who",
        "ai_why_money",
        "cluster_fingerprint",
        "wealth_estimate_confidence",
        "prospect_quality",
    ):
        v = data.get(k)
        out[k] = str(v).strip() if v is not None else ("" if k != "client_type" else "")

    wconf = str(data.get("wealth_estimate_confidence") or "none").strip().lower()
    if wconf not in ("none", "low", "medium", "high"):
        wconf = "none"
    out["wealth_estimate_confidence"] = wconf

    # Keep numeric estimates whenever the model supplies them with low/medium/high confidence;
    # uncertainty is shown via wealth_estimate_confidence, not by dropping the field.
    est = data.get("estimated_wealth_usd")
    try:
        if est is None or wconf == "none":
            out["estimated_wealth_usd"] = None
        else:
            f = float(est)
            out["estimated_wealth_usd"] = f if f > 0 else None
    except (TypeError, ValueError):
        out["estimated_wealth_usd"] = None

    for k in ("extraction_confidence_score", "fa_usefulness_score"):
        try:
            out[k] = int(max(0, min(100, int(data.get(k, 0)))))
        except (TypeError, ValueError):
            out[k] = 0

    pq = str(data.get("prospect_quality") or "").strip()
    allowed_q = (
        "Excellent prospect",
        "Possible prospect",
        "Low-value prospect",
        "Not actionable",
    )
    if pq not in allowed_q:
        for a in allowed_q:
            if a.lower() in pq.lower():
                pq = a
                break
        else:
            pq = ""
    out["prospect_quality"] = pq

    _copy_narrative_from_data(data, out)

    return out


def batch_rerank_prospects(rows: list[dict[str, Any]]) -> dict[int, int]:
    """
    Second pass: assign rank (0 = best) within this batch.

    ``rows`` are in display order; field ``i`` is the 0-based index in this batch (not the dataframe index).
    Returns mapping position -> rank.
    """
    if not rows or len(rows) < 2 or not _rerank_enabled():
        return {}
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or not _enabled():
        return {}

    try:
        from openai import OpenAI
    except ImportError:
        return {}

    slim = []
    for i, r in enumerate(rows[:AI_RERANK_MAX_ROWS]):
        slim.append(
            {
                "i": i,
                "title": (str(r.get("raw_title", "") or ""))[:200],
                "person": (str(r.get("person_name", "") or ""))[:120],
                "quality": str(r.get("prospect_quality", "") or ""),
                "wealth": str(r.get("wealth_signal", "") or ""),
                "use": int(r.get("fa_usefulness_score", 0) or 0),
                "url": (str(r.get("source_url", "") or ""))[:120],
            }
        )
    payload = json.dumps(slim, indent=2)
    prompt = f"""You reorder wealth-prospecting leads for a financial advisor.

Input is a JSON array. Each object has "i" = its zero-based position in the array (0, 1, 2, ...).

Return ONE JSON object with key "order": an array of those same integers, sorted from BEST prospect for an FA to WORST.
Each integer 0..n-1 must appear exactly once.

Prioritize: identifiable wealthy or liquidity-rich individuals, strong wealth signals, actionable outreach.
Deprioritize: vague stories, no person, macro-only, not actionable.

Input:
{payload}
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

    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    order = data.get("order")
    if not isinstance(order, list):
        return {}

    out_map: dict[int, int] = {}
    for rank, pos in enumerate(order):
        try:
            ii = int(pos)
        except (TypeError, ValueError):
            continue
        if 0 <= ii < len(slim):
            out_map[ii] = rank
    return out_map


def enrich_dataframe_with_ai_decision(df):  # pd.DataFrame
    """
    Run structured prospect analysis per row (score >= min), then optional batch rerank on top rows.
    Populates ai_* decision columns and merges narrative fields used by the app.
    """
    import pandas as pd

    if df is None or df.empty:
        return df

    out = df.copy()

    decision_cols = (
        "ai_decision_client_type",
        "ai_decision_wealth_signal",
        "ai_decision_liquidity",
        "ai_decision_source_of_wealth",
        "ai_net_worth_inferred",
        "ai_wealth_estimate_confidence",
        "ai_extraction_confidence",
        "prospect_quality",
        "ai_fa_usefulness_score",
        "ai_why_flagged",
        "ai_why_matters_fa",
        "ai_cluster_fingerprint",
        "ai_rerank_priority",
        "cluster_group_id",
    )
    for c in ("extracted_person_name", "extracted_role", "extracted_company"):
        if c not in out.columns:
            out[c] = ""
    for c in decision_cols:
        if c not in out.columns:
            if c in ("ai_net_worth_inferred",):
                out[c] = pd.NA
            elif c in ("ai_rerank_priority", "cluster_group_id"):
                out[c] = pd.NA
            else:
                out[c] = ""

    narrative = (
        "ai_summary",
        "ai_why_it_matters",
        "ai_outreach",
        "ai_wealth_signal",
        "ai_liquidity_label",
        "ai_client_who",
        "ai_why_money",
    )
    for c in narrative:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str)

    if not _enabled() or not os.environ.get("OPENAI_API_KEY", "").strip():
        return out

    try:
        mask = pd.to_numeric(out["score"], errors="coerce").fillna(0) >= _min_score()
    except Exception:
        return out

    if not mask.any():
        return out

    idx_list = list(out.loc[mask].index)
    for idx in idx_list:
        row = out.loc[idx]
        try:
            d = analyze_prospect_article(row.to_dict())
        except Exception:
            continue

        exn = d.get("extracted_person_name") or ""
        exr = d.get("extracted_role") or ""
        exc = d.get("extracted_company") or ""
        exconf = int(d.get("extraction_confidence_score") or 0)

        out.at[idx, "extracted_person_name"] = str(exn).strip()[:200]
        out.at[idx, "extracted_role"] = str(exr).strip()[:200]
        out.at[idx, "extracted_company"] = str(exc).strip()[:200]

        if exn and exconf >= 60:
            out.at[idx, "person_name"] = str(exn).strip()[:200]
        if exr and exconf >= 55:
            out.at[idx, "role"] = str(exr).strip()[:200]
        if exc and exconf >= 55:
            out.at[idx, "company_name"] = str(exc).strip()[:200]

        ct = d.get("client_type") or ""
        if ct:
            out.at[idx, "ai_decision_client_type"] = str(ct).strip()
            if exconf >= 50:
                out.at[idx, "client_type"] = str(ct).strip()

        ws = d.get("wealth_signal") or ""
        if ws:
            out.at[idx, "ai_decision_wealth_signal"] = ws
            out.at[idx, "ai_wealth_signal"] = str(ws).strip()
            out.at[idx, "wealth_signal_label"] = str(ws).strip()

        liq = d.get("liquidity_event") or ""
        if liq:
            out.at[idx, "ai_decision_liquidity"] = liq
            out.at[idx, "ai_liquidity_label"] = str(liq).strip()
            out.at[idx, "liquidity_event"] = str(liq).strip()

        sow = d.get("source_of_wealth") or ""
        if sow:
            out.at[idx, "ai_decision_source_of_wealth"] = sow
            if exconf >= 45:
                out.at[idx, "source_of_wealth"] = str(sow).strip()

        out.at[idx, "ai_extraction_confidence"] = exconf
        out.at[idx, "prospect_quality"] = str(d.get("prospect_quality") or "").strip()
        out.at[idx, "ai_fa_usefulness_score"] = int(d.get("fa_usefulness_score") or 0)
        out.at[idx, "ai_why_flagged"] = str(d.get("why_flagged") or "").strip()
        out.at[idx, "ai_why_matters_fa"] = str(d.get("why_matters_for_advisor") or "").strip()

        fp = str(d.get("cluster_fingerprint") or "").strip().lower()
        fp = re.sub(r"[^a-z0-9|.\-]+", "-", fp).strip("-")[:180]
        out.at[idx, "ai_cluster_fingerprint"] = fp

        est = d.get("estimated_wealth_usd")
        wconf = str(d.get("wealth_estimate_confidence") or "none").strip().lower()
        out.at[idx, "ai_wealth_estimate_confidence"] = wconf
        out.at[idx, "ai_net_worth_inferred"] = pd.NA
        if est is not None and wconf in ("low", "medium", "high"):
            try:
                ev = float(est)
                if ev > 0:
                    out.at[idx, "ai_net_worth_inferred"] = ev
                    cur = float(pd.to_numeric(out.at[idx, "estimated_wealth"], errors="coerce") or 0)
                    if cur <= 0:
                        out.at[idx, "estimated_wealth"] = ev
            except (TypeError, ValueError):
                pass

        for nk in ("ai_summary", "ai_why_it_matters", "ai_outreach", "ai_client_who", "ai_why_money"):
            if d.get(nk):
                out.at[idx, nk] = str(d[nk]).strip()

    # Cluster group ids (deterministic hash buckets)
    fps = out["ai_cluster_fingerprint"].fillna("").astype(str).str.strip()
    uniq = [x for x in fps.unique() if x]
    mp = {u: hashlib.md5(u.encode()).hexdigest()[:12] for u in uniq}
    out["cluster_group_id"] = fps.map(lambda x: mp.get(x, "") if x else "")

    # Second pass rerank (top by score + usefulness)
    try:
        work = out.loc[mask].copy()
        work["_use"] = pd.to_numeric(work["ai_fa_usefulness_score"], errors="coerce").fillna(0)
        work["_sc"] = pd.to_numeric(work["score"], errors="coerce").fillna(0)
        top = work.sort_values(["_use", "_sc"], ascending=[False, False]).head(AI_RERANK_MAX_ROWS)
        batch_rows: list[dict[str, Any]] = []
        pos_to_idx = list(top.index)
        for idx in top.index:
            r = out.loc[idx]
            batch_rows.append(
                {
                    "raw_title": r.get("raw_title", ""),
                    "person_name": r.get("person_name", ""),
                    "prospect_quality": r.get("prospect_quality", ""),
                    "wealth_signal": r.get("ai_decision_wealth_signal") or r.get("wealth_signal_label", ""),
                    "fa_usefulness_score": r.get("ai_fa_usefulness_score", 0),
                    "source_url": r.get("source_url", ""),
                }
            )
        rmap = batch_rerank_prospects(batch_rows)
        out["ai_rerank_priority"] = pd.NA
        for pos, rank in rmap.items():
            if pos < len(pos_to_idx):
                orig_idx = pos_to_idx[pos]
                out.at[orig_idx, "ai_rerank_priority"] = int(rank)
    except Exception:
        pass

    for c in decision_cols:
        if c in out.columns and c not in ("ai_net_worth_inferred", "ai_rerank_priority"):
            if out[c].dtype == object:
                out[c] = out[c].fillna("").astype(str).str.strip()

    for c in narrative:
        out[c] = out[c].fillna("").astype(str).str.strip()

    return out
