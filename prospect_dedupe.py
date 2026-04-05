"""
Canonical entity/event deduplication for prospect rows (verified pipeline fingerprints).

Keeps the highest ``priority_score`` row per ``dedupe_fingerprint``; optional merge of source URLs later.
"""

from __future__ import annotations

import pandas as pd


def dedupe_prospect_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "dedupe_fingerprint" not in df.columns:
        return df
    out = df.copy()
    score_col = "priority_score" if "priority_score" in out.columns else "score"
    out["_dedupe_rank"] = pd.to_numeric(out[score_col], errors="coerce").fillna(0)
    out = out.sort_values("_dedupe_rank", ascending=False, na_position="last")
    out = out.drop_duplicates(subset=["dedupe_fingerprint"], keep="first")
    return out.drop(columns=["_dedupe_rank"], errors="ignore")
