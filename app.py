

"""
Wealth Signals Dashboard - Streamlit UI.

Run: streamlit run app.py
"""

import html
import json
from datetime import date, datetime, timezone

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
    if v is True or str(v).lower() == "true":
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
    if v is True or str(v).lower() == "true":
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
    Curated ordering for top-of-page blocks: core types, strong extractions, then score.

    Full table / Details use raw `filtered` — this only affects hero row order.
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
    padding: 0.85rem 1.35rem 0.5rem 1.35rem !important;
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


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wealth Signals Dashboard",
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
    ],
)

# -----------------------------------------------------------------------------
# Header row: title + refresh
# -----------------------------------------------------------------------------
with st.container(border=True, key="ws_hero_bar"):
    header_left, header_right = st.columns([4, 1])
    with header_left:
        st.markdown(
            """<p class="ws-hero-title">Wealth Signals Dashboard</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<p class="ws-hero-sub">Public career & finance signals from RSS (sample data if live fetch fails). """
            """Demo only - not investment advice.</p>""",
            unsafe_allow_html=True,
        )
        _df = signals_df
        if len(_df) > 0 and _df["detected_at"].notna().any():
            lu = _df["detected_at"].max()
            st.markdown(
                f"""<div class="ws-last-updated"><strong>{html.escape(format_detected_utc(lu))}</strong>"""
                f"""<span>|</span><em>{html.escape(human_time_ago(lu))}</em></div>""",
                unsafe_allow_html=True,
            )
    with header_right:
        st.markdown('<div style="height:0.15rem"></div>', unsafe_allow_html=True)
        if st.button("Refresh data", use_container_width=True, help="Fetch latest RSS / sample data again"):
            st.session_state.signals_df = fetch_signals()
            st.rerun()

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
    help=f"Applies to Home top {HOME_TOP_SIGNALS}, All signals, and Details. Home also prefers score ≥ {TOP_CURATED_MIN_SCORE} when picking highlights.",
)

sort_by = st.sidebar.radio(
    "Sort list by",
    ("Newest", "Highest score"),
    horizontal=True,
    help="Order for the All signals table and Details section.",
)

search_q = st.sidebar.text_input(
    "Search",
    placeholder="Person or company...",
    help="Matches person or company name (partial, case-insensitive).",
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
    ],
)

if selected_types:
    filtered = filtered[filtered["event_type"].isin(selected_types)]
else:
    filtered = filtered.iloc[0:0]

filtered = filtered[filtered["score"] >= min_score]

if search_q:
    pn = filtered.get("person_name", pd.Series([""] * len(filtered))).fillna("").str.lower()
    cn = filtered.get("company_name", pd.Series([""] * len(filtered))).fillna("").str.lower()
    rt = (
        filtered.get("raw_title", pd.Series([""] * len(filtered))).fillna("").str.lower()
    )
    ap = filtered.get("additional_people", pd.Series([""] * len(filtered))).map(
        lambda x: _format_additional_people(x).lower()
    )
    mask = (
        pn.str.contains(search_q, regex=False, na=False)
        | cn.str.contains(search_q, regex=False, na=False)
        | rt.str.contains(search_q, regex=False, na=False)
        | ap.str.contains(search_q, regex=False, na=False)
    )
    filtered = filtered[mask]

if use_date_filter and date_start is not None and date_end is not None and has_dates:
    day = filtered["event_date"].dt.date
    in_range = filtered["event_date"].isna() | ((day >= date_start) & (day <= date_end))
    filtered = filtered[in_range]

if len(filtered) > 0:
    if sort_by == "Newest":
        filtered = filtered.sort_values("detected_at", ascending=False, na_position="last")
    else:
        filtered = filtered.sort_values("score", ascending=False, na_position="last")

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
    st.markdown("**Current view (after filters)**")
    st.metric("Rows shown", len(filtered))
    if len(filtered) > 0:
        miss_p = int((filtered.get("person_name", pd.Series([""] * len(filtered))).fillna("") == "").sum())
        miss_r = int((filtered.get("role", pd.Series([""] * len(filtered))).fillna("") == "").sum())
        st.caption(f"Missing person_name: **{miss_p}** | Missing role: **{miss_r}**")
        st.markdown("**Counts by event_type**")
        _vc = filtered.get("event_type", pd.Series(dtype=object)).value_counts().rename_axis("event_type").reset_index(name="count")
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
        st.markdown(
            f"""
<div class="ws-section-head">
  <h2 class="ws-h2">Top signals</h2>
  <p class="ws-section-sub">Up to <strong>{HOME_TOP_SIGNALS}</strong> highlights: score ≥ {TOP_CURATED_MIN_SCORE}, ranked for outreach (uses sidebar filters).</p>
</div>
""",
            unsafe_allow_html=True,
        )

        if len(signals_df) == 0:
            st.info("No signals loaded yet - try **Refresh data**.")
            top_home = signals_df.iloc[:0]
        elif len(filtered) == 0:
            st.info("No rows match your filters - widen event types or lower the minimum score, or open **Explore & data**.")
            top_home = filtered.iloc[:0]
        else:
            high_only = filtered[filtered["score"] >= TOP_CURATED_MIN_SCORE].copy()
            top_home = rank_for_hero_sections(high_only).head(HOME_TOP_SIGNALS)

        if len(signals_df) > 0 and len(filtered) > 0 and len(top_home) == 0:
            st.info(
                f"No signals at **score ≥ {TOP_CURATED_MIN_SCORE}** with current filters. Lower the sidebar minimum score or clear search."
            )
        elif len(top_home) > 0:
            for i, (_, row) in enumerate(top_home.iterrows()):
                header_line = format_signal_header_line(row)
                hl = html.escape(header_line)
                out_e = html.escape(str(row.get("outreach_angle", "")))
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
                    st.caption(row.get("suggested_next_step", ""))
                    st.markdown(
                        f"""<p class="ws-link"><a href="{href}" target="_blank" rel="noopener noreferrer">Open source -&gt;</a></p>""",
                        unsafe_allow_html=True,
                    )
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
    # Main table (formatted for readability)
    # -----------------------------------------------------------------------------
    table_columns = [
        "person_name",
        "additional_people",
        "company_name",
        "event_type",
        "raw_title",
        "role",
        "event_date",
        "score",
        "wealth_score",
        "estimated_wealth",
        "aggregated_estimated_wealth",
        "target_client",
        "priority_level",
        "outreach_angle",
        "why_it_matters",
        "source_url",
        "quality_score",
        "confidence_score",
        "is_billionaire",
        "net_worth",
        "billionaire_company",
        "priority",
        "repeat_person",
        "linked_wealth_signal",
        "repeat_company",
    ]
    
    ensure_columns_present(
        filtered,
        table_columns
        + [
            "detected_at",
            "wealth_score",
            "estimated_wealth",
            "aggregated_estimated_wealth",
            "target_client",
            "is_billionaire",
            "net_worth",
            "billionaire_company",
            "priority",
            "repeat_person",
            "linked_wealth_signal",
            "repeat_company",
        ],
    )
    display_df = filtered[table_columns].copy()
    if not display_df.empty and "additional_people" in display_df.columns:
        display_df["additional_people"] = display_df["additional_people"].map(_format_additional_people)
    if not display_df.empty:
        _det = filtered.get("detected_at", pd.Series([pd.NaT] * len(filtered)))
        display_df.insert(0, "Label", _det.apply(lambda x: "NEW" if is_signal_new(x) else ""))
        display_df["Detected"] = _det.apply(human_time_ago)
    if not display_df.empty and "event_date" in display_df.columns:
        display_df["event_date"] = pd.to_datetime(display_df["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        display_df["event_date"] = display_df["event_date"].fillna("-")
    if not display_df.empty and "is_billionaire" in display_df.columns:
        display_df["is_billionaire"] = display_df["is_billionaire"].fillna(False).astype(bool)
    if not display_df.empty and "target_client" in display_df.columns:
        display_df["target_client"] = display_df["target_client"].map(_format_target_client_cell)
    if not display_df.empty and "wealth_score" in display_df.columns:
        display_df["wealth_score"] = pd.to_numeric(display_df["wealth_score"], errors="coerce").fillna(0).astype(int)
    if not display_df.empty and "estimated_wealth" in display_df.columns:
        display_df["estimated_wealth"] = pd.to_numeric(
            display_df["estimated_wealth"], errors="coerce"
        ).fillna(0.0)
    if not display_df.empty and "aggregated_estimated_wealth" in display_df.columns:
        display_df["aggregated_estimated_wealth"] = pd.to_numeric(
            display_df["aggregated_estimated_wealth"], errors="coerce"
        ).fillna(0.0)
    if not display_df.empty and "repeat_person" in display_df.columns:
        display_df["repeat_person"] = display_df["repeat_person"].fillna(False).astype(bool)
    if not display_df.empty and "linked_wealth_signal" in display_df.columns:
        display_df["linked_wealth_signal"] = display_df["linked_wealth_signal"].fillna(False).astype(bool)
    if not display_df.empty and "repeat_company" in display_df.columns:
        display_df["repeat_company"] = display_df["repeat_company"].fillna(False).astype(bool)
    
    st.markdown(
        """
    <div class="ws-section-head">
      <h2 class="ws-h2">All signals</h2>
      <p class="ws-section-sub">Full feed with filters - <strong>NEW</strong> = detected in the last 48 hours.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.write(f"Showing **{len(filtered)}** signal(s) with current filters.")
    
    if filtered.empty:
        st.info("No rows match your filters. Try clearing search, widening the date range, or lowering the score.")
    else:
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Label": st.column_config.TextColumn(" ", width="small"),
                "person_name": st.column_config.TextColumn("Person", width="medium"),
                "additional_people": st.column_config.TextColumn("Also named", width="medium"),
                "company_name": st.column_config.TextColumn("Company", width="medium"),
                "event_type": st.column_config.TextColumn("Event", width="small"),
                "raw_title": st.column_config.TextColumn("Raw title (debug)", width="large"),
                "role": st.column_config.TextColumn("Role", width="small"),
                "event_date": st.column_config.TextColumn("Date", width="small"),
                "score": st.column_config.NumberColumn("Score", format="%d", width="small"),
                "wealth_score": st.column_config.NumberColumn("Wealth", format="%d", width="small"),
                "estimated_wealth": st.column_config.NumberColumn("Est. $", format="$%d", width="small"),
                "aggregated_estimated_wealth": st.column_config.NumberColumn("Agg. $", format="$%d", width="small"),
                "target_client": st.column_config.TextColumn("Target", width="small"),
                "priority_level": st.column_config.TextColumn("Priority", width="small"),
                "outreach_angle": st.column_config.TextColumn("Outreach angle", width="large"),
                "why_it_matters": st.column_config.TextColumn("Why it matters", width="large"),
                "Detected": st.column_config.TextColumn("Detected", width="small"),
                "source_url": st.column_config.LinkColumn("Source", width="medium"),
                "quality_score": st.column_config.NumberColumn("Quality", format="%d", width="small"),
                "confidence_score": st.column_config.NumberColumn("Conf.", format="%d", width="small"),
                "is_billionaire": st.column_config.CheckboxColumn("Billionaire", width="small"),
                "net_worth": st.column_config.TextColumn("Net worth (list)", width="small"),
                "billionaire_company": st.column_config.TextColumn("List source co.", width="medium"),
                "priority": st.column_config.TextColumn("Value tag", width="medium"),
                "repeat_person": st.column_config.CheckboxColumn("Repeat", width="small"),
                "linked_wealth_signal": st.column_config.CheckboxColumn("Linked $", width="small"),
                "repeat_company": st.column_config.CheckboxColumn("Co. repeat", width="small"),
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
      <p class="ws-section-sub">Search and filter, then expand a row. The list scrolls inside the panel below.</p>
    </div>
    """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#0f172a;margin:0 0 0.4rem 0;">Search &amp; filter</p>""",
            unsafe_allow_html=True,
        )
        search_col, prio_col = st.columns([5, 4], gap="medium")
        with search_col:
            details_search = st.text_input(
                "Search details",
                placeholder="Person, company, raw title, or role...",
                key="details_search",
                label_visibility="visible",
                help="Filters rows in this panel only (partial match, case-insensitive).",
            )
        with prio_col:
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
                help="Show all priorities or narrow to one level.",
            )
    
        details_filtered = filtered.copy()
        if details_search.strip():
            search_lower = details_search.strip().lower()
            pn = details_filtered.get("person_name", pd.Series([""] * len(details_filtered))).fillna("").str.lower()
            cn = details_filtered.get("company_name", pd.Series([""] * len(details_filtered))).fillna("").str.lower()
            rt = details_filtered.get("raw_title", pd.Series([""] * len(details_filtered))).fillna("").str.lower()
            rl = details_filtered.get("role", pd.Series([""] * len(details_filtered))).fillna("").str.lower()
            ap = details_filtered.get("additional_people", pd.Series([""] * len(details_filtered))).map(
                lambda x: _format_additional_people(x).lower()
            )
            mask = (
                pn.str.contains(search_lower, na=False, regex=False)
                | cn.str.contains(search_lower, na=False, regex=False)
                | rt.str.contains(search_lower, na=False, regex=False)
                | rl.str.contains(search_lower, na=False, regex=False)
                | ap.str.contains(search_lower, na=False, regex=False)
            )
            details_filtered = details_filtered[mask]
    
        if priority_filter != "all":
            _pl = {"high": "High", "medium": "Medium", "low": "Low"}[priority_filter]
            details_filtered = details_filtered[details_filtered["priority_level"] == _pl]
    
        with st.container(height=580, border=False, key="ws_details_scroll"):
            if details_filtered.empty:
                st.caption("No rows match these filters. Clear the search or set Priority to All.")
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
                        st.markdown(f"**Raw title:** {row.get('raw_title', '-')}")
                        st.markdown(f"**Priority:** {row.get('priority_level', '')}")
                        st.markdown(f"**Detected:** {det}")
                        st.markdown(f"**Outreach suggestion:** {row.get('outreach_angle', '')}")
                        st.markdown(f"**Suggested next step:** {row.get('suggested_next_step', '')}")
                        st.markdown("---")
                        st.markdown(f"**Role:** {row.get('role') or '-'}")
                        st.markdown(f"**Date:** {date_str}")
                        st.markdown(f"**Score:** {int(row.get('score', 0) or 0)}")
                        _tc = row.get("target_client")
                        _tc_label = (
                            "yes"
                            if _tc is True or str(_tc).lower() == "true"
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
                        st.markdown("**Why it matters**")
                        st.write(row.get("why_it_matters", ""))
                        st.markdown("**Full explanation**")
                        st.write(row.get("full_explanation") or "-")
                        st.markdown("**Source**")
                        st.markdown(f"[Open public source]({row.get('source_url', '')})")
