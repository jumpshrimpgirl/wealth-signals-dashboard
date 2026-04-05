"""
Pass 2: strict Home / Top-5 re-ranking with recency + verification bonuses.

Pass 1 (broad) is produced in ``hybrid_pipeline.process_article_row``; this module
pools the top N rows by ``priority_score`` and runs a second LLM review only for Home.
"""

from __future__ import annotations

import json
import os
import re
from datetime import timezone
from typing import Any

import pandas as pd


def compute_recency_score(published_at: Any) -> int:
    """
    Article age vs now. Returns roughly -40 … +20 for use in Home ranking.
    """
    try:
        dt = pd.Timestamp(published_at)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        now = pd.Timestamp.now(tz=timezone.utc)
        days = max(0.0, (now - dt).total_seconds() / 86400.0)
    except Exception:
        return 0

    if days <= 2:
        return 20
    if days <= 7:
        return 12
    if days <= 30:
        return 5
    if days <= 90:
        return -10
    if days <= 180:
        return -25
    return -40


PASS1_RECENCY_WEIGHT = 0.25  # scales full recency into Pass-1 priority (small bump/penalty)


def pass1_recency_adjustment(published_at: Any) -> int:
    """Small adjustment for Explore / Pass-1 only: fraction of ``compute_recency_score``."""
    r = compute_recency_score(published_at)
    adj = int(round(r * PASS1_RECENCY_WEIGHT))
    return max(-10, min(10, adj))


def _verification_bonus(identity_confidence: float) -> int:
    return int(min(15, max(0, round(float(identity_confidence or 0) * 15))))


_RERANK_SCHEMA: dict[str, Any] = {
    "name": "home_rerank_v1",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "timeliness_now": {"type": "integer"},
                        "primary_actor_quality": {"type": "integer"},
                        "wealth_likelihood_now": {"type": "integer"},
                        "verification_strength": {"type": "integer"},
                        "top5_reason": {"type": "string"},
                        "keep_for_home": {"type": "boolean"},
                    },
                    "required": [
                        "index",
                        "timeliness_now",
                        "primary_actor_quality",
                        "wealth_likelihood_now",
                        "verification_strength",
                        "top5_reason",
                        "keep_for_home",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    },
}


def rerank_top_candidates_with_ai(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Strict second pass for Home. One batched LLM call; returns one result dict per input row (order preserved).
    Each result includes: ai_subscore (0-100 from 4×25 dims), top5_reason, keep_for_home,
    plus computed top5_score (ai + recency + verification bonus).
    """
    n = len(candidate_rows)
    if n == 0:
        return []

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _rerank_fallback_heuristic(candidate_rows)

    lines: list[str] = []
    for i, r in enumerate(candidate_rows):
        lines.append(
            f"[{i}] name={r.get('person_name') or r.get('name')} | role={r.get('role')} | "
            f"company={r.get('company_name') or r.get('company')} | "
            f"signal={r.get('signal_type')} pass1_priority={r.get('priority_score')} | "
            f"wealth_status={r.get('wealth_status')} | economic_role={r.get('economic_role')} | "
            f"context={r.get('context_type')} | identity_conf={r.get('identity_confidence')} | "
            f"summary_snip={(str(r.get('summary') or ''))[:400]}"
        )
    batch_text = "\n".join(lines)

    prompt = f"""You re-rank wealth-prospecting candidates for a **Home page Top 5** (highest precision).

For EACH row index 0..{n-1}, score independently (each dimension 0-25):
1) timeliness_now — is the story fresh enough to matter *now* for outreach?
2) primary_actor_quality — is this person clearly the primary economic actor (founder/CEO/owner) vs commentator/lawyer/list mention?
3) wealth_likelihood_now — ownership, liquidity, funding scale, or verified wealth evidence?
4) verification_strength — do role/company/identity look credible (use fields above; penalize commentators, donor-only mentions, stale mega-events)?

Rules:
- Fresh founder/CEO with real company context scores high on 2–3.
- Commentary-only, lawyers (non-founder), politician talk, donor-list mentions: low primary_actor_quality and usually keep_for_home=false.
- **Stale stories** (old leadership moves unless still market-moving): crush timeliness_now.
- Do not invent facts; use only the row text.

Rows:
{batch_text}

Return JSON: results array with one object per index, fields: index, timeliness_now, primary_actor_quality, wealth_likelihood_now, verification_strength (integers 0-25), top5_reason (short), keep_for_home (boolean)."""

    try:
        from openai import OpenAI
    except ImportError:
        return _rerank_fallback_heuristic(candidate_rows)

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
            response_format={"type": "json_schema", "json_schema": _RERANK_SCHEMA},
        )
    except Exception:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.15,
                response_format={"type": "json_object"},
            )
        except Exception:
            return _rerank_fallback_heuristic(candidate_rows)

    raw = (response.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _rerank_fallback_heuristic(candidate_rows)

    by_idx: dict[int, dict[str, Any]] = {}
    for item in (data.get("results") or []) if isinstance(data, dict) else []:
        if not isinstance(item, dict):
            continue
        idx = int(item.get("index", -1))
        if 0 <= idx < n:
            by_idx[idx] = item

    out: list[dict[str, Any]] = []
    for i in range(n):
        row = candidate_rows[i]
        pub = row.get("published_at") or row.get("detected_at") or row.get("event_date")
        rec = compute_recency_score(pub)
        vbonus = _verification_bonus(float(row.get("identity_confidence") or 0))
        it = by_idx.get(i, {})
        t1 = max(0, min(25, int(it.get("timeliness_now", 12))))
        t2 = max(0, min(25, int(it.get("primary_actor_quality", 12))))
        t3 = max(0, min(25, int(it.get("wealth_likelihood_now", 12))))
        t4 = max(0, min(25, int(it.get("verification_strength", 12))))
        ai_sum = t1 + t2 + t3 + t4
        top5 = max(0, min(100, ai_sum + rec + vbonus))
        reason = str(it.get("top5_reason") or "model_default")
        keep = bool(it.get("keep_for_home", True))
        out.append(
            {
                "index": i,
                "timeliness_now": t1,
                "primary_actor_quality": t2,
                "wealth_likelihood_now": t3,
                "verification_strength": t4,
                "ai_subscore": ai_sum,
                "top5_score": top5,
                "top5_reason": reason,
                "keep_for_home": keep,
                "recency_component": rec,
                "verification_bonus": vbonus,
            }
        )
    return out


def _rerank_fallback_heuristic(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """No API: rank by pass1 priority + recency only."""
    out: list[dict[str, Any]] = []
    for i, row in enumerate(candidate_rows):
        pub = row.get("published_at") or row.get("detected_at")
        rec = compute_recency_score(pub)
        ps = int(pd.to_numeric(row.get("priority_score") or row.get("score"), errors="coerce") or 0)
        vbonus = _verification_bonus(float(row.get("identity_confidence") or 0))
        ai_sum = min(60, ps * 3 // 5)
        top5 = max(0, min(100, ai_sum + rec + vbonus))
        er = str(row.get("economic_role") or "").lower()
        keep = er not in ("commentator", "lawyer", "politician") and str(row.get("context_type")) != "mention"
        out.append(
            {
                "index": i,
                "timeliness_now": 15,
                "primary_actor_quality": 15,
                "wealth_likelihood_now": 15,
                "verification_strength": 15,
                "ai_subscore": ai_sum,
                "top5_score": top5,
                "top5_reason": "heuristic_fallback_no_openai",
                "keep_for_home": keep,
                "recency_component": rec,
                "verification_bonus": vbonus,
            }
        )
    return out


def apply_pass2_home_rerank(df: pd.DataFrame, *, pool_size: int = 30) -> pd.DataFrame:
    """
    After Pass-1 dataframe: take top ``pool_size`` by ``priority_score``, run ``rerank_top_candidates_with_ai``,
    merge ``top5_score``, ``top5_reason``, ``keep_for_home``, ``home_priority_label``.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for col, default in (
        ("top5_score", pd.NA),
        ("top5_reason", ""),
        ("keep_for_home", pd.NA),
        ("home_priority_label", ""),
        ("pass2_ai_subscore", pd.NA),
    ):
        if col not in out.columns:
            out[col] = default

    sort_col = "priority_score" if "priority_score" in out.columns else "score"
    pool = out.sort_values(sort_col, ascending=False, na_position="last").head(pool_size)
    if pool.empty:
        return out

    rows = pool.to_dict("records")
    ranked = rerank_top_candidates_with_ai(rows)

    for res, idx in zip(ranked, pool.index):
        out.at[idx, "top5_score"] = res.get("top5_score")
        out.at[idx, "top5_reason"] = res.get("top5_reason", "")
        out.at[idx, "keep_for_home"] = res.get("keep_for_home", True)
        out.at[idx, "pass2_ai_subscore"] = res.get("ai_subscore")
        ts = int(res.get("top5_score") or 0)
        if ts >= 90:
            lab = "Elite"
        elif ts >= 75:
            lab = "High"
        elif ts >= 55:
            lab = "Medium"
        else:
            lab = "Low"
        out.at[idx, "home_priority_label"] = lab

    return out


def build_home_top_view(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Pick Home cards: prefer ``keep_for_home`` and ``top5_score``; fallback to ``priority_score``.
    """
    if df is None or df.empty:
        return df
    sub = df[df["top5_score"].notna()].copy()
    if sub.empty:
        return df.sort_values(
            "priority_score" if "priority_score" in df.columns else "score", ascending=False
        ).head(n)

    sub = sub.sort_values("top5_score", ascending=False, na_position="last")
    kh = sub["keep_for_home"].fillna(False)
    if kh.dtype == object:
        kh = kh.astype(str).str.lower().isin(("true", "1", "yes"))
    preferred = sub[kh.astype(bool)]
    if len(preferred) >= n:
        return preferred.head(n)
    rest = sub[~sub.index.isin(preferred.index)]
    merged = pd.concat([preferred, rest], axis=0).drop_duplicates()
    return merged.head(n)
