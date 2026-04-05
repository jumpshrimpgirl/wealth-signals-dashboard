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


def _safe_int_cell(val, default: int = 0) -> int:
    n = pd.to_numeric(val, errors="coerce")
    try:
        if pd.isna(n):
            return default
    except (TypeError, ValueError):
        return default
    try:
        return int(n)
    except (TypeError, ValueError, OverflowError):
        return default


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


def _normalize_rerank_item(it: dict[str, Any]) -> dict[str, Any]:
    """Map v1 LLM keys to FA v2 schema when needed."""
    if not it:
        return it
    out = dict(it)
    if "founder_operator_centrality" not in out and "primary_actor_quality" in out:
        out["founder_operator_centrality"] = out["primary_actor_quality"]
    if "likely_wealth_creation" not in out and "wealth_likelihood_now" in out:
        out["likely_wealth_creation"] = out["wealth_likelihood_now"]
    if "ownership_concentration_likelihood" not in out:
        lw = int(out.get("likely_wealth_creation", 12))
        out["ownership_concentration_likelihood"] = min(20, max(4, int(lw * 0.75)))
    if "external_verification_strength" not in out and "verification_strength" in out:
        vs = int(out["verification_strength"])
        out["external_verification_strength"] = max(0, min(15, int(vs * 15 / 25)))
    return out


def _pass2_home_score_cap(row: dict[str, Any]) -> int | None:
    """PASS 2 Home: dead/historical / weak context / non-economic article ceilings."""
    if row.get("candidate_historical_dead"):
        return 25
    fw = int(row.get("founder_wealth_score") or 0)
    ct = str(row.get("context_type") or "").lower()
    er = str(row.get("economic_role") or "").lower()
    if ct in ("mention", "commentary") or er == "commentator":
        return 45
    if row.get("article_economic_relevance") is False:
        # Strong founder operational scale can still be a core FA trigger even if headline is not "deal" news
        if fw >= 25 and str(row.get("wealth_status") or "") == "likely_wealth":
            return None
        return 25
    return None


_RERANK_SCHEMA: dict[str, Any] = {
    "name": "home_rerank_fa_v2",
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
                        "founder_operator_centrality": {"type": "integer"},
                        "likely_wealth_creation": {"type": "integer"},
                        "ownership_concentration_likelihood": {"type": "integer"},
                        "external_verification_strength": {"type": "integer"},
                        "fa_urgency_reason": {"type": "string"},
                        "keep_for_home": {"type": "boolean"},
                    },
                    "required": [
                        "index",
                        "timeliness_now",
                        "founder_operator_centrality",
                        "likely_wealth_creation",
                        "ownership_concentration_likelihood",
                        "external_verification_strength",
                        "fa_urgency_reason",
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
    Dimensions (0–100): timeliness 20 + founder centrality 20 + wealth creation 25 + ownership 20 + verification 15.
    ``top5_score`` uses model sum plus small recency/verification nudges.
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
            f"article_economic={r.get('article_economic_relevance')} | dead_hist={r.get('candidate_historical_dead')} | "
            f"founder_wealth={r.get('founder_wealth_score')} | ownership={r.get('ownership_inference')} | "
            f"wealth_evidence={r.get('wealth_evidence')} | "
            f"prospect_tier={r.get('prospect_tier')} | "
            f"summary_snip={(str(r.get('summary') or ''))[:400]}"
        )
    batch_text = "\n".join(lines)

    prompt = f"""You re-rank candidates for a **Home page Top 5** for **financial advisor outreach** (highest precision).

Core question for EACH row index 0..{n-1}:
**Would a financial advisor feel urgency to contact this person *now* based on likely wealth creation, concentrated equity ownership, liquidity/tax planning needs, or sudden private business scale?**

Score independently (must sum to meaningful 0-100 raw before normalization — use these maxima):
1) timeliness_now (0-20): Does this story matter *this week* for outreach?
2) founder_operator_centrality (0-20): Is this person clearly the founder/CEO/operator driving the business vs commentator/lawyer/list mention?
3) likely_wealth_creation (0-25): Private-company revenue/profit/valuation scale, trajectory, or liquidity event — **published net worth NOT required**.
4) ownership_concentration_likelihood (0-20): Bootstrapped, founder-led private co., concentrated equity, thin team + huge scale → high.
5) external_verification_strength (0-15): Credible identity/company from row fields; penalize junk entities.

Rules:
- **Massive private revenue growth** by a founder/operator should score extremely high on 3–4 even with **no** fundraising, M&A, or Forbes net worth.
- Commentary, legal procedurals, donor-list mentions: low scores; usually keep_for_home=false.
- Stale stories: crush timeliness_now.
- **Prospect tier (field ``prospect_tier``):** ``tier_a`` = realistic FA outreach targets (actionable, under-covered). ``tier_b`` = globally famous / saturated public figures — **down-rank** them: they likely already have extensive FA coverage; only score high if the *news event* is unusually actionable for outreach. ``tier_c`` = not suitable — set keep_for_home=false.
- Prioritize **accessibility**, **novelty of wealth event**, and **who an FA could plausibly contact now** — not fame or net worth alone.
- Do not invent facts; use only the row text and scores provided.

Rows:
{batch_text}

Return JSON: results array, one object per index: index, timeliness_now (0-20), founder_operator_centrality (0-20), likely_wealth_creation (0-25), ownership_concentration_likelihood (0-20), external_verification_strength (0-15), fa_urgency_reason (short), keep_for_home (boolean)."""

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
        it = _normalize_rerank_item(by_idx.get(i, {}))
        t1 = max(0, min(20, int(it.get("timeliness_now", 10))))
        t2 = max(0, min(20, int(it.get("founder_operator_centrality", 10))))
        t3 = max(0, min(25, int(it.get("likely_wealth_creation", 12))))
        t4 = max(0, min(20, int(it.get("ownership_concentration_likelihood", 10))))
        t5 = max(0, min(15, int(it.get("external_verification_strength", 8))))
        ai_sum = t1 + t2 + t3 + t4 + t5
        # Small nudge: recency and identity help without double-counting timeliness in the model
        top5 = max(0, min(100, ai_sum + max(-5, min(5, int(round(rec * 0.12)))) + min(3, vbonus // 5)))
        cap = _pass2_home_score_cap(row)
        if cap is not None:
            top5 = min(top5, cap)
        tier = str(row.get("prospect_tier") or "tier_a").lower()
        if tier == "tier_b":
            top5 = max(0, int(top5) - 12)
        reason = str(it.get("fa_urgency_reason") or it.get("top5_reason") or "model_default")
        keep = bool(it.get("keep_for_home", True))
        if tier == "tier_c":
            keep = False
        if row.get("candidate_historical_dead"):
            keep = False
        if row.get("article_economic_relevance") is False and int(row.get("founder_wealth_score") or 0) < 22:
            keep = False
        out.append(
            {
                "index": i,
                "timeliness_now": t1,
                "founder_operator_centrality": t2,
                "likely_wealth_creation": t3,
                "ownership_concentration_likelihood": t4,
                "external_verification_strength": t5,
                "ai_subscore": ai_sum,
                "top5_score": top5,
                "top5_reason": reason,
                "fa_urgency_reason": reason,
                "keep_for_home": keep,
                "recency_component": rec,
                "verification_bonus": vbonus,
            }
        )
    return out


def _rerank_fallback_heuristic(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """No API: FA-style 5-dim heuristic using founder_wealth_score, ownership, recency."""
    out: list[dict[str, Any]] = []
    for i, row in enumerate(candidate_rows):
        pub = row.get("published_at") or row.get("detected_at")
        rec = compute_recency_score(pub)
        vbonus = _verification_bonus(float(row.get("identity_confidence") or 0))
        fw = int(row.get("founder_wealth_score") or 0)
        own = str(row.get("ownership_inference") or "low")
        er = str(row.get("economic_role") or "").lower()
        ct = str(row.get("context_type") or "").lower()
        t1 = max(0, min(20, 6 + min(14, max(0, rec + 40) // 5)))
        t2 = (
            18
            if er in ("founder", "ceo", "owner") and ct == "primary"
            else (14 if er in ("founder", "ceo", "owner") else 6)
        )
        t3 = max(0, min(25, 6 + min(19, fw * 25 // 40)))
        t4 = 18 if own == "high" else (12 if own == "medium" else 5)
        t5 = max(0, min(15, int(float(row.get("identity_confidence") or 0) * 14)))
        ai_sum = t1 + t2 + t3 + t4 + t5
        top5 = max(0, min(100, ai_sum + max(-5, min(5, int(round(rec * 0.12)))) + min(3, vbonus // 5)))
        cap = _pass2_home_score_cap(row)
        if cap is not None:
            top5 = min(top5, cap)
        tier = str(row.get("prospect_tier") or "tier_a").lower()
        if tier == "tier_b":
            top5 = max(0, int(top5) - 12)
        keep = er not in ("commentator", "lawyer", "politician") and ct not in ("mention", "commentary")
        if tier == "tier_c":
            keep = False
        if row.get("candidate_historical_dead"):
            keep = False
        if row.get("article_economic_relevance") is False and fw < 22:
            keep = False
        reason = "heuristic_fallback_no_openai"
        out.append(
            {
                "index": i,
                "timeliness_now": t1,
                "founder_operator_centrality": t2,
                "likely_wealth_creation": t3,
                "ownership_concentration_likelihood": t4,
                "external_verification_strength": t5,
                "ai_subscore": ai_sum,
                "top5_score": top5,
                "top5_reason": reason,
                "fa_urgency_reason": reason,
                "keep_for_home": keep,
                "recency_component": rec,
                "verification_bonus": vbonus,
            }
        )
    return out


def _select_home_with_tier_policy(sub_ok: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Prefer Tier A; allow at most one Tier B if top5_score is exceptional; Tier C excluded by caller.
    """
    if sub_ok is None or sub_ok.empty:
        return sub_ok
    work = sub_ok.copy()
    work["_pt"] = work["prospect_tier"].fillna("tier_a").astype(str).str.lower()
    tier_a = work[work["_pt"] == "tier_a"].sort_values("top5_score", ascending=False, na_position="last")
    tier_b = work[work["_pt"] == "tier_b"].sort_values("top5_score", ascending=False, na_position="last")
    picked: list[Any] = []
    for idx in tier_a.index:
        if len(picked) >= n:
            break
        picked.append(idx)
    if len(picked) < n and not tier_b.empty:
        top_b = tier_b.iloc[0]
        try:
            ts = int(top_b["top5_score"])
        except (TypeError, ValueError):
            ts = 0
        if ts >= 82:
            picked.append(tier_b.index[0])
    if len(picked) < n:
        for idx in tier_a.index:
            if len(picked) >= n:
                break
            if idx not in picked:
                picked.append(idx)
    out = work.loc[picked[:n]].drop(columns=["_pt"], errors="ignore")
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
        ("fa_urgency_reason", ""),
        ("keep_for_home", pd.NA),
        ("home_priority_label", ""),
        ("pass2_ai_subscore", pd.NA),
    ):
        if col not in out.columns:
            out[col] = default
    if "prospect_tier" not in out.columns:
        out["prospect_tier"] = "tier_a"

    sort_col = "priority_score" if "priority_score" in out.columns else "score"
    pool = out.sort_values(sort_col, ascending=False, na_position="last").head(pool_size)
    if pool.empty:
        return out

    rows = pool.to_dict("records")
    ranked = rerank_top_candidates_with_ai(rows)

    for res, idx in zip(ranked, pool.index):
        out.at[idx, "top5_score"] = res.get("top5_score")
        fa_r = str(res.get("fa_urgency_reason") or res.get("top5_reason") or "")
        out.at[idx, "top5_reason"] = fa_r
        out.at[idx, "fa_urgency_reason"] = fa_r
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
    Pick Home cards: ``keep_for_home`` + ``top5_score``, then **prospect_tier** policy
    (majority Tier A; at most one exceptional Tier B; never Tier C).
    """
    if df is None or df.empty:
        return df
    sub = df[df["top5_score"].notna()].copy()
    if sub.empty:
        return df.sort_values(
            "priority_score" if "priority_score" in df.columns else "score", ascending=False
        ).head(n)

    if "prospect_tier" not in sub.columns:
        sub["prospect_tier"] = "tier_a"

    sub = sub.sort_values("top5_score", ascending=False, na_position="last")
    kh = sub["keep_for_home"].fillna(False)
    if kh.dtype == object:
        kh = kh.astype(str).str.lower().isin(("true", "1", "yes"))
    pool = sub[kh.astype(bool)].copy()
    pool = pool[pool["prospect_tier"].fillna("tier_a").astype(str).str.lower() != "tier_c"]

    if pool.empty:
        pool = sub[
            sub["prospect_tier"].fillna("tier_a").astype(str).str.lower() != "tier_c"
        ].copy()

    return _select_home_with_tier_policy(pool, n)
