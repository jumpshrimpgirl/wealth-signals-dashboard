"""
Hybrid AI extraction + enrichment + entity resolution + scoring.

Ranking is implemented in ``hybrid_pipeline``; this module exposes the Streamlit-facing
``process_and_rank_prospects`` entrypoint and stable aliases for tooling.
"""

from __future__ import annotations

import pandas as pd

from hybrid_pipeline import (
    estimate_wealth_from_context as estimate_wealth,
    priority_label_from_score,
    process_articles,
    score_article_signal,
    score_match,
    to_clean_dataframe,
)


def process_and_rank_prospects(raw_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw article rows into ranked prospect rows (multiple prospects per article possible).

    Output preserves legacy dashboard columns via per-row merge; core scores are
    ``signal_score``, ``match_score``, and ``priority_score`` (0–100).
    """
    if raw_rows is None or raw_rows.empty:
        return raw_rows

    processed = process_articles(raw_rows)
    return pd.DataFrame(processed)


__all__ = [
    "estimate_wealth",
    "priority_label_from_score",
    "process_and_rank_prospects",
    "process_articles",
    "score_article_signal",
    "score_match",
    "to_clean_dataframe",
]
