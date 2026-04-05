

"""
Wealth Signals Dashboard - Streamlit UI.

Run: streamlit run app.py
"""

import html
import json
import os
from datetime import date, datetime, timezone
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from data import fetch_signals
from person_validation import is_valid_person

# How long a signal counts as "NEW" in the feed (hours)
NEW_WINDOW_HOURS = 48

# Curated hero blocks: score floor (full table uses sidebar minimum score only)
TOP_CURATED_MIN_SCORE = 40
# Homepage tab: max signals to show (ranked)
HOME_TOP_SIGNALS = 5

# Explore view: drop weak / noisy rows (after sidebar filters)
EXPLORE_MIN_SCORE = 30

# Company field: clear when it looks like a news outlet (substring match on normalized name)
KNOWN_OUTLETS_TOKENS = frozenset(
    {"bbc", "cnn", "nyt", "economist", "forbes", "reuters", "techcrunch", "bloomberg", "msnbc", "nbc"}
)

# Person field: drop obvious non-name tokens
BAD_PERSON_TOKENS = frozenset({"the", "a", "in", "on", "air force", "central alabama"})


def _source_outlet_from_url(url: str) -> str:
    """Best-effort hostname for display (replaces nonexistent ``source`` column)."""
    try:
        u = urlparse(str(url or "").strip())
        h = (u.netloc or "").lower()
        if h.startswith("www."):
            h = h[4:]
        return h.split(":")[0] or ""
    except Exception:
        return ""


def _company_clear_if_known_outlet(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        return s
    sl = s.lower()
    for tok in KNOWN_OUTLETS_TOKENS:
        if tok in sl:
            return ""
    return s


def _person_clean_token(x) -> str:
    s = str(x or "").strip()
    if not s:
        return s
    if s.lower() in BAD_PERSON_TOKENS:
        return ""
    return s


def prepare_explore_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final pass for Explore / Home / Details: drop noisy rows, clarify fields, sort best → worst.

    Does not change underlying ``signals_df``; operates on the filtered view only.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    out = out[out["event_type"].astype(str).str.strip() != "Other"]
    sc = pd.to_numeric(out["score"], errors="coerce").fillna(0)
    out = out[sc >= EXPLORE_MIN_SCORE]
    if out.empty:
        return out
    if "source_url" in out.columns:
        out["source_outlet"] = out["source_url"].apply(_source_outlet_from_url)
    else:
        out["source_outlet"] = ""
    out["company_name"] = out["company_name"].fillna("").astype(str).map(_company_clear_if_known_outlet)
    out["person_name"] = out["person_name"].fillna("").astype(str).map(_person_clean_token)
    if "ai_summary" not in out.columns:
        out["ai_summary"] = ""
    out["ai_summary"] = out.apply(
        lambda r: (str(r.get("ai_summary") or "").strip() or str(r.get("raw_title") or "")),
        axis=1,
    )
    out["target_client"] = out.apply(
        lambda r: "YES" if int(pd.to_numeric(r.get("score"), errors="coerce") or 0) >= 70 else "NO",
        axis=1,
    )
    for q in ("quality_score", "confidence_score"):
        if q not in out.columns:
            out[q] = 0
        else:
            out[q] = pd.to_numeric(out[q], errors="coerce").fillna(0)
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0)
    out = out.sort_values(
        by=["score", "quality_score", "confidence_score"],
        ascending=[False, False, False],
        na_position="last",
    )
    return out


# Hero sections: core event types rank above "Other" (do not use categorical sort — it reversed order).
EVENT_TYPE_RANK = {
    "Founder Exit": 5,
    "Funding": 4,
    "Promotion": 3,
    "Board Appointment": 2,
    "Other": 1,
}


def _company_name_for_header(row) -> str:
    """Return company label for headers, or empty when unknown / placeholder."""
    cn = str(row.get("company_name", "") or "").strip()
    if not cn or cn.lower() == "unknown":
        return ""
    return cn


def billionaire_badge_html(row) -> str:
    """Small HTML badge when the person matched the billionaire / wealth list."""
    v = row.get("is_billionaire")
    if v is True or str(v).lower() == "true":
        return (
            '<span class="ws-badge" title="Matched billionaire list (net worth on file)" '
            'style="background:#fef9c3;border:1px solid #eab308;">💰</span> '
        )
    return ""


def target_client_badge_html(row) -> str:
    """Badge for primary target (high wealth / deal) or mid-tier ($1M–$5M est.)."""
    try:
        agg = float(row.get("aggregated_estimated_wealth") or 0)
    except (TypeError, ValueError):
        agg = 0.0
    hot = agg >= 10_000_000
    v = row.get("target_client")
    if v is True or str(v).lower() == "true" or str(v).strip().upper() == "YES":
        fire = (
            '<span class="ws-badge" title="Multi-signal wealth: $10M+ estimated across your feed for this person" '
            'style="background:#fef2f2;border:1px solid #ef4444;">🔥</span> '
            if hot
            else ""
        )
        return (
            fire
            + '<span class="ws-badge" title="Target client: strong wealth or $5M+ estimated personal stake" '
            'style="background:#dcfce7;border:1px solid #22c55e;">★</span> '
        )
    if v == "mid" or str(v).lower() == "mid":
        return (
            '<span class="ws-badge" title="Mid target: ~$1M–$5M estimated wealth from deal size" '
            'style="background:#ffedd5;border:1px solid #f97316;">◆</span> '
        )
    return ""


def _format_target_client_cell(v) -> str:
    if v is True or str(v).lower() == "true" or str(v).strip().upper() == "YES":
        return "yes"
    if v == "mid" or str(v).lower() == "mid":
        return "mid"
    return "no"


def format_signal_header_line(row) -> str:
    """
    Primary one-line signal title for cards and expanders.

    - Valid person: ``Name — EventType @ Company`` (drops ``@ Company`` when company unknown).
    - Otherwise: ``Executive move @ Company (low confidence)`` or event-only / generic fallback.
    Never emits blank names; invalid / place-like ``person_name`` values use the low-confidence template.
    """
    pn = str(row.get("person_name", "") or "").strip()
    et = str(row.get("event_type", "") or "").strip() or "Signal"
    co = _company_name_for_header(row)

    if pn and is_valid_person(pn):
        if co:
            return f"{pn} — {et} @ {co}"
        return f"{pn} — {et}"

    if co:
        return f"Executive move @ {co} (low confidence)"
    if et and et != "Signal":
        return f"Executive move — {et} (low confidence)"
    return "Executive move (low confidence)"


def _format_additional_people(val) -> str:
    """Pretty list of other people named in the same story (JSON array or empty)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if isinstance(val, list):
        return ", ".join(str(x) for x in val if str(x).strip())
    s = str(val).strip()
    if not s or s == "[]":
        return ""
    try:
        names = json.loads(s)
        if isinstance(names, list) and names:
            return ", ".join(str(x) for x in names)
    except (json.JSONDecodeError, TypeError):
        return s
    return ""


def ensure_required_signal_columns(df: pd.DataFrame) -> None:
    """Guarantee core columns exist so filters and hero sections never KeyError."""
    for col in ["person_name", "additional_people", "company_name", "role", "event_type", "score"]:
        if col not in df.columns:
            df[col] = 0 if col == "score" else ("[]" if col == "additional_people" else "")


def ensure_columns_present(df: pd.DataFrame, columns: list[str]) -> None:
    """Add missing columns with safe defaults (strings empty, numeric scores 0, event_date NaT)."""
    for col in columns:
        if col not in df.columns:
            if col in ("score", "quality_score", "confidence_score"):
                df[col] = 0
            elif col in ("event_date", "detected_at"):
                df[col] = pd.NaT
            else:
                df[col] = ""


def rank_for_hero_sections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Curated ordering for hero blocks: core types, strong extractions, then score.

    Used only as a fallback when AI ranking is unavailable (Home tab).
    """
    if df is None or df.empty:
        return df
    d = df.copy()
    ensure_required_signal_columns(d)
    if "confidence_score" not in d.columns:
        d["confidence_score"] = 0
    if "quality_score" not in d.columns:
        d["quality_score"] = 0
    d["_ev"] = d["event_type"].map(EVENT_TYPE_RANK).fillna(0).astype(int)
    d["_pn"] = (d["person_name"].fillna("").str.strip() != "").astype(int)
    d["_cn"] = ((d["company_name"].fillna("").str.strip() != "") & (d["company_name"] != "Unknown")).astype(int)
    d = d.sort_values(
        by=["_ev", "confidence_score", "quality_score", "score", "_pn", "_cn"],
        ascending=[False, False, False, False, False, False],
    )
    return d.drop(columns=["_ev", "_pn", "_cn"], errors="ignore")


def inject_styles() -> None:
    """
    Global look: soft canvas, typography, cards, metrics, table shell, sidebar.
    """
    st.markdown(
"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

  .stApp {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #e4e7ec !important;
    color: #0f172a;
  }

  section.main > div.block-container {
    max-width: 1100px !important;
    padding: 2rem 1.35rem 0.5rem 1.35rem !important;
  }

  /* ----- Sidebar: scan-friendly filters ----- */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #dce1e8 0%, #d5dae3 100%) !important;
    border-right: 1px solid #94a3b8 !important;
    box-shadow: inset -1px 0 0 rgba(15, 23, 42, 0.06);
  }
  section[data-testid="stSidebar"] .block-container {
    padding-top: 0.85rem !important;
    padding-left: 0.9rem !important;
    padding-right: 0.65rem !important;
  }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3,
  .ws-sidebar-filters-title {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #0f172a !important;
    margin: 0 0 0.65rem 0 !important;
    padding-bottom: 0.35rem !important;
    border-bottom: 2px solid #64748b !important;
  }
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
  section[data-testid="stSidebar"] .stMarkdown p {
    font-weight: 600 !important;
    color: #1e293b !important;
    font-size: 0.8125rem !important;
  }
  section[data-testid="stSidebar"] .stSelectbox,
  section[data-testid="stSidebar"] .stMultiselect,
  section[data-testid="stSidebar"] .stSlider,
  section[data-testid="stSidebar"] .stTextInput,
  section[data-testid="stSidebar"] .stCheckbox,
  section[data-testid="stSidebar"] .stRadio,
  section[data-testid="stSidebar"] .stDateInput {
    margin-bottom: 0.45rem !important;
  }
  section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    border-color: #94a3b8 !important;
  }
  section[data-testid="stSidebar"] .stExpander {
    border: 1px solid #94a3b8 !important;
    border-radius: 8px !important;
    background: rgba(255,255,255,0.45) !important;
  }

  /* ----- Hero / top bar (compact) ----- */
  .st-key-ws_hero_bar[data-testid="stVerticalBlockBorderWrapper"],
  [data-testid="stVerticalBlockBorderWrapper"].st-key-ws_hero_bar {
    background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%) !important;
    border: 1px solid #94a3b8 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08), 0 1px 2px rgba(15, 23, 42, 0.06) !important;
    padding: 0.55rem 0.85rem !important;
    margin-bottom: 0.5rem !important;
  }
  .ws-hero-title {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: #020617;
    margin: 0 0 0.1rem 0;
    line-height: 1.15;
  }
  .ws-hero-sub {
    font-size: 0.8rem;
    color: #475569;
    margin: 0 0 0.25rem 0;
    line-height: 1.35;
    max-width: 48rem;
    font-weight: 500;
  }
  .ws-last-updated {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.75rem;
    color: #334155;
    background: #fff;
    border: 1px solid #94a3b8;
    border-radius: 999px;
    padding: 0.28rem 0.75rem;
    margin-top: 0.15rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    font-weight: 500;
  }
  .ws-last-updated strong {
    color: #020617;
    font-weight: 700;
  }
  .ws-last-updated em {
    color: #64748b;
    font-style: normal;
    font-weight: 500;
  }

  /* ----- Section headers (strong hierarchy) ----- */
  .ws-section-head {
    margin: 0 0 0.45rem 0;
    padding: 0 0 0.35rem 0;
    border-bottom: 1px solid #cbd5e1;
  }
  .ws-h2 {
    font-size: 0.6875rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #0f172a;
    margin: 0 0 0.2rem 0;
    line-height: 1.2;
  }
  .ws-section-sub {
    font-size: 0.78rem;
    color: #475569;
    margin: 0;
    line-height: 1.35;
    font-weight: 500;
  }
  .ws-section-sub strong {
    color: #1e293b;
    font-weight: 700;
  }

  .ws-rule {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #94a3b8 12%, #94a3b8 88%, transparent);
    margin: 0.45rem 0;
  }

  /* ----- Section shell cards (priority / week / metrics) ----- */
  .st-key-ws_card_priority[data-testid="stVerticalBlockBorderWrapper"],
  .st-key-ws_card_week[data-testid="stVerticalBlockBorderWrapper"],
  .st-key-ws_card_metrics[data-testid="stVerticalBlockBorderWrapper"],
  [data-testid="stVerticalBlockBorderWrapper"].st-key-ws_card_priority,
  [data-testid="stVerticalBlockBorderWrapper"].st-key-ws_card_week,
  [data-testid="stVerticalBlockBorderWrapper"].st-key-ws_card_metrics {
    background: #ffffff !important;
    border: 1px solid #94a3b8 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.07), 0 1px 3px rgba(15, 23, 42, 0.06) !important;
    padding: 0.55rem 0.75rem 0.65rem 0.75rem !important;
    margin-bottom: 0.45rem !important;
  }

  /* Metrics row inside card */
  .st-key-ws_card_metrics [data-testid="stMetric"] {
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    padding: 0.55rem 0.65rem !important;
    box-shadow: none !important;
  }
  .st-key-ws_card_metrics [data-testid="stMetric"] label {
    color: #475569 !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
  }
  .st-key-ws_card_metrics [data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #020617 !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
  }

  /* Nested signal row cards (inside priority / week sections) */
  .st-key-ws_card_priority [data-testid="stVerticalBlockBorderWrapper"]:not(.st-key-ws_card_priority),
  .st-key-ws_card_week [data-testid="stVerticalBlockBorderWrapper"]:not(.st-key-ws_card_week) {
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06) !important;
    padding: 0.45rem 0.6rem !important;
    margin-bottom: 0.3rem !important;
  }

  /* Any other bordered container (fallback) */
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05) !important;
  }

  /* Primary button */
  .stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: 1px solid #334155 !important;
    background: #0f172a !important;
    color: #fff !important;
    padding: 0.4rem 0.75rem !important;
    font-size: 0.8125rem !important;
  }
  .stButton > button:hover {
    border-color: #020617 !important;
    background: #020617 !important;
  }

  /* Dataframe */
  [data-testid="stDataFrame"] {
    border: 1px solid #94a3b8 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06) !important;
  }

  /* Expanders (Details + global) */
  [data-testid="stExpander"] {
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    margin-bottom: 0.28rem !important;
    background: #ffffff !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05) !important;
  }
  [data-testid="stExpander"] details {
    border: none !important;
  }
  [data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 0.8125rem !important;
    color: #0f172a !important;
    padding: 0.35rem 0.5rem !important;
  }
  [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 0.35rem 0.65rem 0.65rem 0.65rem !important;
    border-top: 1px solid #e2e8f0 !important;
    background: #fafbfc !important;
  }
  [data-testid="stExpander"] .stMarkdown p {
    margin: 0.2rem 0 !important;
    font-size: 0.8125rem !important;
    line-height: 1.45 !important;
    color: #334155 !important;
  }
  [data-testid="stExpander"] .stMarkdown strong {
    color: #0f172a !important;
    font-weight: 700 !important;
  }

  .stAlert {
    border-radius: 8px !important;
    border: 1px solid #94a3b8 !important;
    font-size: 0.8125rem !important;
  }

  div[data-testid="stRadio"] > div { gap: 0.35rem; }

  /* Badges & pills */
  .ws-badge {
    display: inline-block;
    font-size: 0.625rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    vertical-align: middle;
  }
  .ws-badge--high {
    background: linear-gradient(180deg, #3d1219 0%, #2a0c10 100%);
    color: #fecaca;
    border: 1px solid #7f1d1d;
  }
  .ws-badge--medium {
    background: linear-gradient(180deg, #fffbeb 0%, #fef3c7 100%);
    color: #92400e;
    border: 1px solid #fcd34d;
  }
  .ws-badge--low {
    background: #e2e8f0;
    color: #334155;
    border: 1px solid #94a3b8;
  }
  .ws-badge--other {
    background: #ede9fe;
    color: #5b21b6;
    border: 1px solid #a78bfa;
  }
  .ws-pill {
    display: inline-block;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #0f766e;
    background: #ccfbf1;
    border: 1px solid #2dd4bf;
    border-radius: 999px;
    padding: 0.12rem 0.45rem;
    margin-left: 0.3rem;
    vertical-align: middle;
  }
  .ws-score-pill {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: #020617;
  }
  .ws-card-line {
    font-size: 0.8125rem;
    color: #1e293b;
    line-height: 1.35;
    margin: 0.05rem 0 0 0;
    font-weight: 500;
  }
  .ws-card-line strong { color: #020617; font-weight: 700; }
  .ws-card-meta {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 0.15rem;
    font-weight: 500;
  }
  .ws-link a {
    color: #0f172a !important;
    font-weight: 600 !important;
    text-decoration: none !important;
    border-bottom: 1px solid #64748b;
  }

  /* Details Explorer card */
  [data-testid="stVerticalBlockBorderWrapper"].st-key-ws_details_explorer,
  .st-key-ws_details_explorer[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg, #f1f5f9 0%, #ffffff 55%) !important;
    border: 1px solid #64748b !important;
    box-shadow: 0 6px 22px rgba(15, 23, 42, 0.1), 0 1px 3px rgba(15, 23, 42, 0.08) !important;
    padding: 0.55rem 0.75rem 0.65rem 0.75rem !important;
  }
  .st-key-ws_details_explorer label,
  .st-key-ws_details_explorer [data-testid="stWidgetLabel"] p {
    font-weight: 700 !important;
    color: #0f172a !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
  }
  [data-testid="stVerticalBlockBorderWrapper"].st-key-ws_details_scroll,
  .st-key-ws_details_scroll[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    background: #fafbfc !important;
  }

  /* --- Global button + layout (theme alignment) --- */
  div.stButton > button {
    background-color: #6366F1;
    color: white !important;
    border-radius: 10px;
    border: none;
    padding: 10px 16px;
    font-weight: 600;
  }
  div.stButton > button:hover {
    background-color: #4F46E5;
    color: white !important;
  }
  button {
    color: white !important;
  }
  [data-testid="stDataFrame"] {
    font-size: 14px;
  }
</style>
    """,
        unsafe_allow_html=True,
    )


def human_time_ago(ts) -> str:
    """Turn a timestamp into a short, human phrase (e.g. '2 hours ago', '1 day ago')."""
    if ts is None or pd.isna(ts):
        return "-"
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    now = pd.Timestamp.now(tz=timezone.utc)
    secs = max(0, int((now - t).total_seconds()))
    if secs < 45:
        return "just now"
    mins = secs // 60
    if secs < 3600:
        return f"{mins} min ago" if mins != 1 else "1 min ago"
    hours = secs // 3600
    if secs < 86400:
        return f"{hours} hours ago" if hours != 1 else "1 hour ago"
    days = secs // 86400
    return f"{days} days ago" if days != 1 else "1 day ago"


def is_signal_new(ts, hours: int = NEW_WINDOW_HOURS) -> bool:
    """True if the signal was detected within the last `hours` hours."""
    if ts is None or pd.isna(ts):
        return False
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    now = pd.Timestamp.now(tz=timezone.utc)
    return (now - t).total_seconds() < hours * 3600


def format_detected_utc(ts) -> str:
    """Clock time in UTC for the 'Last updated' line."""
    if ts is None or pd.isna(ts):
        return "-"
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.strftime("%Y-%m-%d %H:%M UTC")


def priority_badge_html(level: str) -> str:
    """HTML badge for priority level (styled via CSS)."""
    lv = (level or "").strip()
    if lv == "High":
        cls = "ws-badge ws-badge--high"
    elif lv == "Medium":
        cls = "ws-badge ws-badge--medium"
    else:
        cls = "ws-badge ws-badge--low"
    return f'<span class="{cls}">{html.escape(lv)}</span>'


def new_pill_html() -> str:
    return '<span class="ws-pill">NEW</span>'


def safe_href(url: str) -> str:
    """Escape a URL for use in HTML attributes."""
    return html.escape(str(url), quote=True)


def event_type_badge_html(event_type: str) -> str:
    """Small label for event_type; highlights 'Other' for debugging."""
    et = (event_type or "").strip()
    if et == "Other":
        return '<span class="ws-badge ws-badge--other">OTHER</span>'
    return f'<span class="ws-badge ws-badge--low">{html.escape(et)}</span>'


def _ai_ranking_enabled() -> bool:
    return os.environ.get("WEALTH_SIGNALS_AI_RANKING", "1").lower() not in ("0", "false", "no")


def rank_signals_with_ai(df: pd.DataFrame) -> list[dict]:
    """
    Advisor-style ordering: pick the best HOME_TOP_SIGNALS opportunities from the top 20 by score.

    Returns a list of dicts (best → worst). Does **not** read or write pipeline scores — ranking only.
    On failure or missing API, returns [] so the caller falls back to score sort.
    """
    if df is None or df.empty:
        return []
    if not _ai_ranking_enabled() or not os.environ.get("OPENAI_API_KEY", "").strip():
        return []
    try:
        from openai import OpenAI
    except ImportError:
        return []
    if "score" not in df.columns:
        return []

    candidates = df.sort_values("score", ascending=False).head(20).reset_index(drop=True)
    if candidates.empty:
        return []

    signals: list[dict] = []
    for _, row in candidates.iterrows():
        try:
            ew = float(row.get("estimated_wealth") or 0)
        except (TypeError, ValueError):
            ew = 0.0
        try:
            agg = float(row.get("aggregated_estimated_wealth") or 0)
        except (TypeError, ValueError):
            agg = 0.0
        nw = agg if agg > 0 else ew
        signals.append(
            {
                "person": str(row.get("person_name", "") or ""),
                "company": str(row.get("company_name", "") or ""),
                "event": str(row.get("event_type", "") or ""),
                "title": str(row.get("raw_title", "") or ""),
                "score": int(row.get("score") or 0),
                "net_worth": round(nw, 0),
                "source_url": str(row.get("source_url", "") or ""),
            }
        )

    k = min(HOME_TOP_SIGNALS, len(signals))
    if k < 1:
        return []
    if k == 1:
        return [signals[0]]

    payload = json.dumps(signals, indent=2)
    prompt = f"""You are a top financial advisor targeting $5M+ clients.

From this list of signals, select the BEST {k} outreach opportunities.

Rules:
- Prioritize liquidity events (exits, funding, promotions)
- Prioritize wealthy individuals (use net_worth as a hint)
- Ignore irrelevant or macro-only news
- Ignore journalists, locations, governments as the primary "person"
- Prefer real individuals with plausible money movement

Return a JSON object with a single key "ranked" whose value is an array of exactly {k} objects.
Each object must include these keys copied exactly from the chosen rows in the list below:
person, company, event, title, score, net_worth, source_url
Order the array best opportunity first, worst last. Do not change numeric scores.

Signals:
{payload}
"""

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "").strip())
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
    except Exception as e:
        print("AI ranking failed:", e)
        return []

    raw_content = (response.choices[0].message.content or "").strip()
    if not raw_content:
        return []

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        return []

    ranked = data.get("ranked") if isinstance(data, dict) else None
    if not isinstance(ranked, list):
        return []

    allowed = {str(s.get("source_url") or "").strip() for s in signals if s.get("source_url")}
    out: list[dict] = []
    seen: set[str] = set()
    for obj in ranked:
        if not isinstance(obj, dict):
            continue
        u = str(obj.get("source_url") or "").strip()
        if u and u in allowed and u not in seen:
            out.append(
                {
                    "person": str(obj.get("person", "") or ""),
                    "company": str(obj.get("company", "") or ""),
                    "event": str(obj.get("event", "") or ""),
                    "title": str(obj.get("title", "") or ""),
                    "score": obj.get("score", 0),
                    "net_worth": obj.get("net_worth", 0),
                    "source_url": u,
                }
            )
            seen.add(u)
        if len(out) >= k:
            break

    if len(out) < k:
        for s in signals:
            if len(out) >= k:
                break
            u = str(s.get("source_url") or "").strip()
            if u and u not in seen:
                out.append(s)
                seen.add(u)

    return out[:k]


def lookup_home_row_for_ai(df: pd.DataFrame, item: dict) -> pd.Series | None:
    """Map an AI-ranked dict back to a full dataframe row when source_url / title match."""
    if df is None or df.empty or not item:
        return None
    url = str(item.get("source_url") or "").strip()
    if url and "source_url" in df.columns:
        m = df[df["source_url"].astype(str).str.strip() == url]
        if len(m):
            return m.iloc[0]
    person = str(item.get("person") or "").strip()
    title = str(item.get("title") or "").strip()
    if person and title and "person_name" in df.columns and "raw_title" in df.columns:
        m = df[
            (df["person_name"].astype(str).str.strip() == person)
            & (df["raw_title"].astype(str).str.strip() == title)
        ]
        if len(m):
            return m.iloc[0]
    return None


def render_home_signal_card(row: pd.Series) -> None:
    """Rich card for Home tab (badges, links, AI lines when present)."""
    header_line = format_signal_header_line(row)
    hl = html.escape(header_line)
    _line = str(row.get("ai_summary", "") or "").strip() or str(row.get("outreach_angle", "") or "")
    out_e = html.escape(_line)
    new_html = new_pill_html() if is_signal_new(row.get("detected_at")) else ""
    ago = human_time_ago(row.get("detected_at"))
    href = safe_href(str(row.get("source_url", "")))
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(
                f"""<p class="ws-card-line"><strong>{hl}</strong>{target_client_badge_html(row)}{billionaire_badge_html(row)}{new_html}</p>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""<p class="ws-card-line">{event_type_badge_html(row.get("event_type", ""))} {priority_badge_html(row.get("priority_level", ""))} | Score: {int(row.get("score", 0) or 0)} | {out_e}</p>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""<p class="ws-card-meta">Detected {html.escape(ago)}</p>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<p style="text-align:right;margin:0;font-size:1.05rem;font-weight:700;">{int(row.get("score", 0) or 0)}</p>""",
                unsafe_allow_html=True,
            )
        _cap = str(row.get("ai_outreach", "") or "").strip() or str(row.get("suggested_next_step", "") or "")
        st.caption(_cap)
        st.markdown(
            f"""<p class="ws-link"><a href="{href}" target="_blank" rel="noopener noreferrer">Open source -&gt;</a></p>""",
            unsafe_allow_html=True,
        )


def render_home_signal_card_simple(item: dict) -> None:
    """Fallback when a ranked dict cannot be joined to the dataframe."""
    with st.container(border=True):
        st.write(f"{item.get('person', '')} @ {item.get('company', '')}")
        st.write(item.get("title", ""))
        st.write(f"Score: {item.get('score', '')}")
        st.markdown("---")


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wealth Signals",
    page_icon="📊",
    layout="wide",
)

inject_styles()

# Session: cached dataframe so "Refresh" can reload without restarting the app
if "signals_df" not in st.session_state:
    st.session_state.signals_df = fetch_signals()

signals_df = st.session_state.signals_df
ensure_required_signal_columns(signals_df)
ensure_columns_present(
    signals_df,
    [
        "raw_title",
        "event_date",
        "detected_at",
        "priority_level",
        "quality_score",
        "confidence_score",
        "outreach_angle",
        "suggested_next_step",
        "why_it_matters",
        "full_explanation",
        "ai_summary",
        "ai_why_it_matters",
        "ai_outreach",
    ],
)

# -----------------------------------------------------------------------------
# Header row: title + refresh
# -----------------------------------------------------------------------------
with st.container(border=True, key="ws_hero_bar"):
    st.markdown(
        """
<h2 style='margin-bottom:0;'>Wealth Signals</h2>
<p style='color:#9CA3AF; margin-top:0;'>High-probability client opportunities</p>
""",
        unsafe_allow_html=True,
    )
    st.caption("Public career & finance signals from RSS (sample if live fetch fails). Demo only — not investment advice.")
    _df = signals_df
    if len(_df) > 0 and _df["detected_at"].notna().any():
        lu = _df["detected_at"].max()
        st.markdown(
            f"""<div class="ws-last-updated"><strong>{html.escape(format_detected_utc(lu))}</strong>"""
            f"""<span>|</span><em>{html.escape(human_time_ago(lu))}</em></div>""",
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Sidebar: filters first (so "Pipeline & debug" can use the filtered dataframe)
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """<p class="ws-sidebar-filters-title">Filters</p>""",
    unsafe_allow_html=True,
)

_event_opts = sorted(signals_df.get("event_type", pd.Series(dtype=object)).replace("", pd.NA).dropna().unique().tolist())
if not _event_opts:
    _event_opts = ["Founder Exit", "Funding", "Promotion", "Board Appointment", "Other"]

selected_types = st.sidebar.multiselect(
    "Event type",
    options=_event_opts,
    default=_event_opts,
    help="Include every type by default. Narrow to debug specific buckets.",
)

min_score = st.sidebar.slider(
    "Minimum score",
    min_value=0,
    max_value=100,
    value=TOP_CURATED_MIN_SCORE,
    help=f"Applies to Home top {HOME_TOP_SIGNALS}, All signals, and Details. Home ranks from the filtered list (AI re-orders among top 20 by score when enabled).",
)

search_query = st.sidebar.text_input(
    "Search (person, company, title, role)",
    placeholder="Filter the feed…",
    help="Applies everywhere: Home, All signals, and Details (partial match, case-insensitive).",
).strip()

use_date_filter = st.sidebar.checkbox(
    "Filter by event date",
    value=False,
    help="Off by default so undated or wide date ranges do not hide the feed.",
)

has_dates = (
    "event_date" in signals_df.columns
    and signals_df["event_date"].notna().any()
)
date_start: date | None = None
date_end: date | None = None

if has_dates and use_date_filter:
    valid = signals_df["event_date"].dropna()
    dmin = valid.min()
    dmax = valid.max()
    dmin_d = dmin.date() if hasattr(dmin, "date") else dmin
    dmax_d = dmax.date() if hasattr(dmax, "date") else dmax
    dr = st.sidebar.date_input(
        "Event date range",
        value=(dmin_d, dmax_d),
        min_value=dmin_d,
        max_value=dmax_d,
        help='Only applied when "Filter by event date" is on. Rows with no date stay visible.',
    )
    if isinstance(dr, tuple) and len(dr) == 2:
        date_start, date_end = dr[0], dr[1]
    elif isinstance(dr, (date, datetime)):
        date_start = date_end = dr.date() if isinstance(dr, datetime) else dr

# --- Apply filters ---
filtered = signals_df.copy()
ensure_required_signal_columns(filtered)
ensure_columns_present(
    filtered,
    [
        "raw_title",
        "event_date",
        "detected_at",
        "priority_level",
        "quality_score",
        "confidence_score",
        "outreach_angle",
        "suggested_next_step",
        "why_it_matters",
        "full_explanation",
        "ai_summary",
        "ai_why_it_matters",
        "ai_outreach",
    ],
)

if selected_types:
    filtered = filtered[filtered["event_type"].isin(selected_types)]
else:
    filtered = filtered.iloc[0:0]

filtered = filtered[filtered["score"] >= min_score]

if search_query:
    sq = search_query
    pn = filtered.get("person_name", pd.Series([""] * len(filtered))).fillna("").astype(str)
    cn = filtered.get("company_name", pd.Series([""] * len(filtered))).fillna("").astype(str)
    rt = filtered.get("raw_title", pd.Series([""] * len(filtered))).fillna("").astype(str)
    rl = filtered.get("role", pd.Series([""] * len(filtered))).fillna("").astype(str)
    mask = (
        pn.str.contains(sq, case=False, na=False, regex=False)
        | cn.str.contains(sq, case=False, na=False, regex=False)
        | rt.str.contains(sq, case=False, na=False, regex=False)
        | rl.str.contains(sq, case=False, na=False, regex=False)
    )
    filtered = filtered[mask]

if use_date_filter and date_start is not None and date_end is not None and has_dates:
    day = filtered["event_date"].dt.date
    in_range = filtered["event_date"].isna() | ((day >= date_start) & (day <= date_end))
    filtered = filtered[in_range]

# Final view: drop noise, enrich, sort best → worst (Home, table, Details)
explore_view = prepare_explore_view(filtered)

# --- Pipeline debug (sidebar) ---
_ingest = getattr(signals_df, "attrs", {}).get("ingest_debug", {})
with st.sidebar.expander("Pipeline & debug", expanded=False):
    st.markdown("**Ingestion (last fetch)**")
    st.caption("If raw >> parsed, classification is strict or headlines are off-topic.")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Raw RSS items", _ingest.get("raw_rss_entries", "-"))
    with c2:
        st.metric("Parsed signals", _ingest.get("parsed_signal_rows", "-"))
    st.metric("Rows after dedupe", _ingest.get("rows_after_finalize", len(signals_df)))
    st.caption(f"Source: `{_ingest.get('data_source', 'unknown')}`")
    st.divider()
    st.markdown("**Current view (after filters + explore rules)**")
    st.metric("Rows shown", len(explore_view))
    if len(explore_view) > 0:
        miss_p = int((explore_view.get("person_name", pd.Series([""] * len(explore_view))).fillna("") == "").sum())
        miss_r = int((explore_view.get("role", pd.Series([""] * len(explore_view))).fillna("") == "").sum())
        st.caption(f"Missing person_name: **{miss_p}** | Missing role: **{miss_r}**")
        st.markdown("**Counts by event_type**")
        _vc = explore_view.get("event_type", pd.Series(dtype=object)).value_counts().rename_axis("event_type").reset_index(name="count")
        st.dataframe(_vc, hide_index=True, use_container_width=True)
    else:
        st.caption("No rows match filters — widen event types or lower the minimum score.")

# -----------------------------------------------------------------------------
# Main layout: Home (top 5) vs Explore (table, details, metrics)
# -----------------------------------------------------------------------------
n_total = len(signals_df)

tab_home, tab_explore = st.tabs(["Home", "Explore & data"])

with tab_home:
    with st.container(border=True, key="ws_card_priority"):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Top Signals")
        with col2:
            st.markdown('<div style="height:0.35rem"></div>', unsafe_allow_html=True)
            if st.button(
                "Refresh",
                use_container_width=True,
                help="Fetch latest RSS / sample data again",
                key="home_refresh_signals",
            ):
                st.session_state.signals_df = fetch_signals()
                st.rerun()
        st.caption(
            f"Up to **{HOME_TOP_SIGNALS}** from your current filters (sidebar). "
            f"AI picks the best {HOME_TOP_SIGNALS} among the top 20 by score when enabled; otherwise score order."
        )

        if len(signals_df) == 0:
            st.info("No signals loaded yet - try **Refresh data**.")
        elif len(filtered) == 0:
            st.info("No rows match your filters - widen event types or lower the minimum score, or open **Explore & data**.")
        elif len(explore_view) == 0:
            st.info(
                "No signals pass the explore view (score ≥ **30**, event type not **Other**). "
                "Lower the minimum score or widen event types."
            )
        else:
            ranked_signals = rank_signals_with_ai(explore_view)
            if not ranked_signals:
                top_home = explore_view.head(HOME_TOP_SIGNALS)
                for _, row in top_home.iterrows():
                    render_home_signal_card(row)
            else:
                for item in ranked_signals:
                    row = lookup_home_row_for_ai(explore_view, item)
                    if row is not None:
                        render_home_signal_card(row)
                    else:
                        render_home_signal_card_simple(item)
    st.caption(f"Full feed, search, and row details are on the **Explore & data** tab.")

with tab_explore:
    # Metrics (full loaded dataset)
    with st.container(border=True, key="ws_card_metrics"):
        st.markdown(
            """<div class="ws-section-head" style="border-bottom:none;padding-bottom:0;margin-bottom:0.35rem;"><h2 class="ws-h2">Performance snapshot</h2></div>""",
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total signals", f"{n_total:,}" if n_total else "0")
        with m2:
            if n_total and signals_df["score"].notna().any():
                hi = int(signals_df["score"].max())
            else:
                hi = None
            st.metric("Highest score", hi if hi is not None else "-")
        with m3:
            if n_total:
                modes = signals_df["event_type"].replace("", pd.NA).dropna().mode()
                common = str(modes.iloc[0]) if len(modes) else "-"
            else:
                common = "-"
            st.metric("Most common event type", common)

    st.markdown("""<hr class="ws-rule"/>""", unsafe_allow_html=True)
    
    # -----------------------------------------------------------------------------
    # Main table (narrow columns — same data as Details / Home)
    # -----------------------------------------------------------------------------
    _DISPLAY_RENAME = {
        "person_name": "Person",
        "company_name": "Company",
        "event_type": "Event",
        "score": "Score",
        "target_client": "Target client",
        "ai_summary": "AI summary (or headline)",
    }
    _NARROW_COLS = ["person_name", "company_name", "event_type", "score", "target_client", "ai_summary"]

    st.markdown(
        """
    <div class="ws-section-head">
      <h2 class="ws-h2">All signals</h2>
      <p class="ws-section-sub">Sorted by score, then quality &amp; confidence. Same search as the sidebar.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.write(f"Showing **{len(explore_view)}** signal(s) in the explore view (after score ≥ {EXPLORE_MIN_SCORE} and non-Other).")

    if explore_view.empty:
        st.info(
            "No rows in this view. Try clearing search, widening event types, lowering the minimum score, "
            f"or note that explore hides **Other** events and scores below **{EXPLORE_MIN_SCORE}**."
        )
    else:
        display_narrow = explore_view[_NARROW_COLS].copy()
        display_narrow = display_narrow.rename(columns=_DISPLAY_RENAME)
        st.dataframe(
            display_narrow,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Person": st.column_config.TextColumn("Person", width="medium"),
                "Company": st.column_config.TextColumn("Company", width="medium"),
                "Event": st.column_config.TextColumn("Event", width="small"),
                "Score": st.column_config.NumberColumn("Score", format="%d", width="small"),
                "Target client": st.column_config.TextColumn("Target client", width="small"),
                "AI summary (or headline)": st.column_config.TextColumn("AI summary (or headline)", width="large"),
            },
        )
    
    # -----------------------------------------------------------------------------
    # Details: compact scrollable panel (expandable rows)
    # -----------------------------------------------------------------------------
    with st.container(border=True, key="ws_details_explorer"):
        st.markdown(
            """
    <div class="ws-section-head">
      <h2 class="ws-h2">Details Explorer</h2>
      <p class="ws-section-sub">Uses the <strong>same filtered list</strong> as the table (sidebar search). Sort: best scores first.</p>
    </div>
    """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#0f172a;margin:0 0 0.4rem 0;">Filter</p>""",
            unsafe_allow_html=True,
        )

        def _details_priority_label(v: str) -> str:
            return {
                "all": "All",
                "high": "High priority",
                "medium": "Medium priority",
                "low": "Low priority",
            }[v]

        priority_filter = st.radio(
            "Priority",
            ["all", "high", "medium", "low"],
            horizontal=True,
            format_func=_details_priority_label,
            key="details_priority_filter",
            help="Optional: narrow by outreach priority (same rows as sidebar search).",
        )

        details_filtered = explore_view.copy()
        if priority_filter != "all" and "priority_level" in details_filtered.columns:
            _pl = {"high": "High", "medium": "Medium", "low": "Low"}[priority_filter]
            details_filtered = details_filtered[details_filtered["priority_level"] == _pl]
    
        with st.container(height=580, border=False, key="ws_details_scroll"):
            if details_filtered.empty:
                st.caption("No rows match this priority. Set Priority to All.")
            else:
                for _, row in details_filtered.iterrows():
                    also = _format_additional_people(row.get("additional_people"))
                    new_html = new_pill_html() if is_signal_new(row.get("detected_at")) else ""
                    title_plain = format_signal_header_line(row)
                    ed = row.get("event_date")
                    if pd.isna(ed):
                        date_str = "-"
                    else:
                        ts = pd.Timestamp(ed)
                        date_str = ts.strftime("%Y-%m-%d")
                    det = human_time_ago(row.get("detected_at"))
                    with st.expander(title_plain):
                        st.markdown(
                            f"""<p style="margin:0 0 0.75rem 0;">{target_client_badge_html(row)}{billionaire_badge_html(row)}{new_html} {priority_badge_html(row.get("priority_level", ""))}</p>""",
                            unsafe_allow_html=True,
                        )
                        if row.get("is_billionaire"):
                            st.caption(
                                f"💰 Billionaire list: net worth {row.get('net_worth') or '—'} "
                                f"({row.get('billionaire_company') or 'source —'})"
                            )
                        if also:
                            st.caption(f"Also named in story: {also}")
                        _so = str(row.get("source_outlet") or "").strip()
                        if _so:
                            st.caption(f"Source outlet: {_so}")
                        st.markdown(f"**Raw title:** {row.get('raw_title', '-')}")
                        st.markdown(f"**Priority:** {row.get('priority_level', '')}")
                        st.markdown(f"**Detected:** {det}")
                        _ai_s = str(row.get("ai_summary", "") or "").strip()
                        _ai_w = str(row.get("ai_why_it_matters", "") or "").strip()
                        _ai_o = str(row.get("ai_outreach", "") or "").strip()
                        if _ai_s or _ai_w or _ai_o:
                            st.markdown("**AI (advisor context)**")
                            if _ai_s:
                                st.markdown(f"**Summary:** {_ai_s}")
                            if _ai_w:
                                st.markdown(f"**Why it matters:** {_ai_w}")
                            if _ai_o:
                                st.markdown(f"**Outreach angle:** {_ai_o}")
                            st.markdown("---")
                        st.markdown(
                            f"**Outreach (template):** {row.get('outreach_angle', '')}"
                        )
                        st.markdown(f"**Suggested next step:** {row.get('suggested_next_step', '')}")
                        st.markdown("---")
                        st.markdown(f"**Role:** {row.get('role') or '-'}")
                        st.markdown(f"**Date:** {date_str}")
                        st.markdown(f"**Score:** {int(row.get('score', 0) or 0)}")
                        _tc = row.get("target_client")
                        _tc_label = (
                            "yes"
                            if _tc is True
                            or str(_tc).lower() == "true"
                            or str(_tc).strip().upper() == "YES"
                            else ("mid" if _tc == "mid" or str(_tc).lower() == "mid" else "no")
                        )
                        _ew = float(row.get("estimated_wealth") or 0)
                        _agg = float(row.get("aggregated_estimated_wealth") or 0)
                        st.markdown(
                            f"**Wealth score:** {int(row.get('wealth_score', 0) or 0)} | "
                            f"**Est. wealth (this row):** ${_ew:,.0f} | "
                            f"**Agg. est. (person):** ${_agg:,.0f} | "
                            f"**Target client:** {_tc_label}"
                        )
                        st.caption(
                            f"Cross-article: repeat person = {bool(row.get('repeat_person'))} | "
                            f"linked wealth signal = {bool(row.get('linked_wealth_signal'))} | "
                            f"repeat company = {bool(row.get('repeat_company'))}"
                        )
                        st.markdown("**Why it matters (baseline)**")
                        st.write(row.get("why_it_matters", ""))
                        st.markdown("**Full explanation**")
                        st.write(row.get("full_explanation") or "-")
                        st.markdown("**Source**")
                        st.markdown(f"[Open public source]({row.get('source_url', '')})")
