"""
Hybrid AI extraction + enrichment + entity resolution + scoring.

Ranking is implemented in ``hybrid_pipeline``; this module exposes the Streamlit-facing
``process_and_rank_prospects`` entrypoint and stable aliases for tooling.
"""

from __future__ import annotations

import pandas as pd

from ai_prospect_pipeline import score_article_signal
from hybrid_pipeline import (
    estimate_wealth_from_context as estimate_wealth,
    priority_label_from_score,
    process_articles,
    score_match,
    to_clean_dataframe,
)
from two_pass_pipeline import apply_pass2_home_rerank
from prospect_dedupe import dedupe_prospect_dataframe
from wealth_display import validate_display_wealth


def _apply_wealth_display_gate(df: pd.DataFrame) -> pd.DataFrame:
    """Final wealth sanity before Pass-2 Home rerank and Explore table (idempotent)."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for idx in out.index:
        row = out.loc[idx]
        vd = validate_display_wealth(
            {
                "est_wealth": row.get("est_wealth"),
                "est_wealth_display": row.get("est_wealth_display"),
                "wealth_numeric_verified": row.get("wealth_numeric_verified"),
            }
        )
        disp = str(vd.get("est_wealth_display") or row.get("est_wealth_display") or row.get("est_wealth") or "")
        out.at[idx, "est_wealth_display"] = disp
        out.at[idx, "wealth_numeric_verified"] = bool(vd.get("wealth_numeric_verified"))
        if "est_wealth" in out.columns:
            out.at[idx, "est_wealth"] = disp
    return out


def process_and_rank_prospects(raw_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Pass 1: broad recall (``process_articles``). Pass 2: strict Home re-rank on top ~30 rows.

    Explore / table: ``priority_score`` (Pass 1). Home: ``top5_score`` / ``keep_for_home``.
    """
    if raw_rows is None or raw_rows.empty:
        return raw_rows

    processed = process_articles(raw_rows)
    df = pd.DataFrame(processed)
    df = dedupe_prospect_dataframe(df)
    df = _apply_wealth_display_gate(df)
    return apply_pass2_home_rerank(df, pool_size=30)


__all__ = [
    "estimate_wealth",
    "priority_label_from_score",
    "process_and_rank_prospects",
    "process_articles",
    "score_article_signal",
    "score_match",
    "to_clean_dataframe",
]
