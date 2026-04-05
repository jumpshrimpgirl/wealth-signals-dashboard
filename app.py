"""
Wealth Signals Dashboard — Streamlit UI.

Run: streamlit run app.py
"""

import html
from datetime import date, datetime, timezone

import pandas as pd
import streamlit as st

from data import fetch_signals

# How long a signal counts as “NEW” in the feed (hours)
NEW_WINDOW_HOURS = 48


def inject_styles() -> None:
    """Global look: soft canvas, typography, cards, metrics, table shell, sidebar."""
    st.markdown(
        """
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

  .stApp {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: linear-gradient(180deg, #f4f5f7 0%, #f0f1f4 100%) !important;
  }

  section.main > div {
    max-width: 1120px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 2rem !important;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #e8ecf0 !important;
    border-right: 1px solid #d1d5db !important;
  }
  section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
  }

  /* Hero / title area */
  .ws-hero-title {
    font-size: 1.85rem;
    font-weight: 600;
    letter-spacing: -0.035em;
    color: #000;
    margin: 0 0 0.35rem 0;
    line-height: 1.2;
  }
  .ws-hero-sub {
    font-size: 0.95rem;
    color: #475569;
    margin: 0 0 0.5rem 0;
    line-height: 1.45;
    max-width: 52rem;
  }
  .ws-last-updated {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8125rem;
    color: #374151;
    background: #fff;
    border: 1px solid #d1d5db;
    border-radius: 999px;
    padding: 0.35rem 0.85rem;
    margin-top: 0.35rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
  }
  .ws-last-updated strong {
    color: #000;
    font-weight: 600;
  }
  .ws-last-updated em {
    color: #6b7280;
    font-style: normal;
  }

  /* Section headers */
  .ws-section-head {
    margin: 1rem 0 0.5rem 0;
  }
  .ws-section-head:first-of-type { margin-top: 0.5rem; }
  .ws-h2 {
    font-size: 1.125rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #000;
    margin: 0 0 0.35rem 0;
  }
  .ws-section-sub {
    font-size: 0.875rem;
    color: #475569;
    margin: 0;
    line-height: 1.5;
  }

  .ws-rule {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #d1d5db 15%, #d1d5db 85%, transparent);
    margin: 1.5rem 0;
  }

  /* Metric tiles */
  [data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.25rem !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
  }
  [data-testid="stMetric"] label {
    color: #374151 !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  [data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #000 !important;
    font-weight: 600 !important;
  }

  /* Bordered blocks (cards) */
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    padding: 1.5rem 1.75rem !important;
    margin-bottom: 0.75rem !important;
  }

  /* Primary button */
  .stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    border: 1px solid #9ca3af !important;
    background: #000 !important;
    color: #fff !important;
    padding: 0.5rem 1rem !important;
  }
  .stButton > button:hover {
    border-color: #000 !important;
    opacity: 0.92;
  }

  /* Dataframe shell */
  [data-testid="stDataFrame"] {
    border: 1px solid #d1d5db !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
  }

  /* Expanders */
  [data-testid="stExpander"] {
    border: 1px solid #d1d5db !important;
    border-radius: 12px !important;
    margin-bottom: 0.5rem !important;
    background: #fff !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
  }
  [data-testid="stExpander"] summary {
    font-weight: 500 !important;
    color: #000 !important;
  }

  /* Info / caption polish */
  .stAlert {
    border-radius: 12px !important;
    border: 1px solid #d1d5db !important;
  }

  /* Horizontal radio */
  div[data-testid="stRadio"] > div {
    gap: 0.5rem;
  }
</style>

<style>
  /* Badges & pills (used inside markdown) */
  .ws-badge {
    display: inline-block;
    font-size: 0.6875rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
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
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
  }
  .ws-badge--other {
    background: #ede9fe;
    color: #5b21b6;
    border: 1px solid #c4b5fd;
  }
  .ws-pill {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #0f766e;
    background: #ccfbf1;
    border: 1px solid #5eead4;
    border-radius: 999px;
    padding: 0.15rem 0.55rem;
    margin-left: 0.35rem;
    vertical-align: middle;
  }
  .ws-score-pill {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #000;
  }
  .ws-card-line {
    font-size: 0.9rem;
    color: #1f2937;
    line-height: 1.4;
    margin: 0.1rem 0 0 0;
  }
  .ws-card-meta {
    font-size: 0.78rem;
    color: #6b7280;
    margin-top: 0.25rem;
  }
  .ws-link a {
    color: #000 !important;
    font-weight: 500;
    text-decoration: none !important;
    border-bottom: 1px solid #9ca3af;
  }
  .ws-link a:hover { border-bottom-color: #000; }
</style>
        """,
        unsafe_allow_html=True,
    )


def human_time_ago(ts) -> str:
    """
    Turn a timestamp into a short, human phrase (e.g. '2 hours ago', '1 day ago').
    """
    if ts is None or pd.isna(ts):
        return "—"
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
        return "—"
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

# -----------------------------------------------------------------------------
# Header row: title + refresh
# -----------------------------------------------------------------------------
header_left, header_right = st.columns([4, 1])
with header_left:
    st.markdown(
        '<p class="ws-hero-title">Wealth Signals Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="ws-hero-sub">Public career & finance signals from RSS (sample data if live fetch fails). '
        "Demo only — not investment advice.</p>",
        unsafe_allow_html=True,
    )
    _df = st.session_state.signals_df
    if len(_df) > 0 and _df["detected_at"].notna().any():
        lu = _df["detected_at"].max()
        st.markdown(
            f'<div class="ws-last-updated"><strong>{html.escape(format_detected_utc(lu))}</strong>'
            f"<span>·</span><em>{html.escape(human_time_ago(lu))}</em></div>",
            unsafe_allow_html=True,
        )
with header_right:
    st.write("")
    if st.button("Refresh data", use_container_width=True, help="Fetch latest RSS / sample data again"):
        st.session_state.signals_df = fetch_signals()
        st.rerun()

signals_df = st.session_state.signals_df

# -----------------------------------------------------------------------------
# Sidebar: filters first (so “Pipeline & debug” can use the filtered dataframe)
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

_event_opts = sorted(signals_df["event_type"].replace("", pd.NA).dropna().unique().tolist())
if not _event_opts:
    _event_opts = ["Founder Exit", "Funding", "Promotion", "Board Appointment", "Other"]

selected_types = st.sidebar.multiselect(
    "Event type",
    options=_event_opts,
    default=_event_opts,
    help="Include every type by default. Narrow to debug specific buckets.",
)

show_other = st.sidebar.checkbox(
    "Show low-confidence / Other signals",
    value=True,
    help="When off, hides rows with event type “Other” (broad finance/career match).",
)

min_score = st.sidebar.slider(
    "Minimum score",
    min_value=0,
    max_value=100,
    value=0,
    help="Lower this if the table looks empty — ‘Other’ scores 55 by default.",
)

sort_by = st.sidebar.radio(
    "Sort list by",
    ("Newest", "Highest score"),
    horizontal=True,
    help="Order for the All signals table and Details section.",
)

search_q = st.sidebar.text_input(
    "Search",
    placeholder="Person or company…",
    help="Matches person or company name (partial, case-insensitive).",
).strip()

use_date_filter = st.sidebar.checkbox(
    "Filter by event date",
    value=False,
    help="Off by default so undated or wide date ranges do not hide the feed.",
)

has_dates = signals_df["event_date"].notna().any()
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
        help="Only applied when “Filter by event date” is on. Rows with no date stay visible.",
    )
    if isinstance(dr, tuple) and len(dr) == 2:
        date_start, date_end = dr[0], dr[1]
    elif isinstance(dr, (date, datetime)):
        date_start = date_end = dr.date() if isinstance(dr, datetime) else dr

# --- Apply filters ---
filtered = signals_df.copy()

if selected_types:
    filtered = filtered[filtered["event_type"].isin(selected_types)]
else:
    filtered = filtered.iloc[0:0]

if not show_other:
    filtered = filtered[filtered["event_type"] != "Other"]

filtered = filtered[filtered["score"] >= min_score]

if search_q:
    pn = filtered["person_name"].fillna("").str.lower()
    cn = filtered["company_name"].fillna("").str.lower()
    rt = filtered["raw_title"].fillna("").str.lower() if "raw_title" in filtered.columns else pd.Series([""] * len(filtered))
    mask = (
        pn.str.contains(search_q, regex=False, na=False)
        | cn.str.contains(search_q, regex=False, na=False)
        | rt.str.contains(search_q, regex=False, na=False)
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
    st.caption("If raw ≫ parsed, classification is strict or headlines are off-topic.")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Raw RSS items", _ingest.get("raw_rss_entries", "—"))
    with c2:
        st.metric("Parsed signals", _ingest.get("parsed_signal_rows", "—"))
    st.metric("Rows after dedupe", _ingest.get("rows_after_finalize", len(signals_df)))
    st.caption(f"Source: `{_ingest.get('data_source', 'unknown')}`")
    st.divider()
    st.markdown("**Current view (after filters)**")
    st.metric("Rows shown", len(filtered))
    if len(filtered) > 0:
        miss_p = int((filtered["person_name"].fillna("") == "").sum())
        miss_r = int((filtered["role"].fillna("") == "").sum())
        st.caption(f"Missing person_name: **{miss_p}** · Missing role: **{miss_r}**")
        st.markdown("**Counts by event_type**")
        _vc = filtered["event_type"].value_counts().rename_axis("event_type").reset_index(name="count")
        st.dataframe(_vc, hide_index=True, use_container_width=True)
    else:
        st.caption("No rows match filters — widen event types, raise score ceiling, or enable Other.")

# -----------------------------------------------------------------------------
# Top high priority opportunities (action layer — who to act on first)
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="ws-section-head">
  <h2 class="ws-h2">Top high priority opportunities</h2>
  <p class="ws-section-sub">Highest-scoring <strong>High</strong> priority signals — good candidates to engage this week.</p>
</div>
""",
    unsafe_allow_html=True,
)

if len(signals_df) == 0:
    st.info("No signals loaded yet — try **Refresh data**.")
    top_high = signals_df.iloc[:0]
else:
    high_only = signals_df[signals_df["priority_level"] == "High"]
    # Filter for high quality: quality_score >= 2, exclude low-confidence "Other", avoid missing person + weak company
    high_only = high_only[high_only["quality_score"] >= 2]
    high_only = high_only[~((high_only["event_type"] == "Other") & (high_only["quality_score"] < 3))]
    high_only = high_only[~((high_only["person_name"] == "") & (high_only["company_name"] == "Unknown"))]
    # Sort by quality first, then score
    top_high = high_only.sort_values(["quality_score", "score"], ascending=[False, False]).head(5)

if len(signals_df) > 0 and len(top_high) == 0:
    st.info("No **High** priority signals right now (score ≥ 85). Lower the minimum score filter below or check back after refresh.")
elif len(top_high) > 0:
    for i, (_, row) in enumerate(top_high.iterrows()):
        person = row["person_name"] or "—"
        company = row["company_name"] or "Unknown"
        pe, ce = html.escape(str(person)), html.escape(str(company))
        out_e = html.escape(str(row["outreach_angle"]))
        new_html = new_pill_html() if is_signal_new(row.get("detected_at")) else ""
        ago = human_time_ago(row.get("detected_at"))
        href = safe_href(row["source_url"])
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(
                    f'<p class="ws-card-line"><strong>{pe}</strong> · <em>{ce}</em>{new_html}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p class="ws-card-line">{event_type_badge_html(row["event_type"])} {priority_badge_html("High")} · Score: {int(row["score"])} · {out_e}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<p class="ws-card-meta">Detected {html.escape(ago)}</p>', unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f'<p style="text-align:right;margin:0;font-size:1.2rem;font-weight:600;">{int(row["score"])}</p>',
                    unsafe_allow_html=True,
                )
            st.caption(row["suggested_next_step"])
            st.markdown(
                f'<p class="ws-link"><a href="{href}" target="_blank" rel="noopener noreferrer">Open source →</a></p>',
                unsafe_allow_html=True,
            )

st.markdown('<hr class="ws-rule"/>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Metrics (based on the full loaded dataset — before search / date slicing)
# -----------------------------------------------------------------------------
m1, m2, m3 = st.columns(3)
n_total = len(signals_df)
with m1:
    st.metric("Total signals", f"{n_total:,}" if n_total else "0")
with m2:
    if n_total and signals_df["score"].notna().any():
        hi = int(signals_df["score"].max())
    else:
        hi = None
    st.metric("Highest score", hi if hi is not None else "—")
with m3:
    if n_total:
        modes = signals_df["event_type"].replace("", pd.NA).dropna().mode()
        common = str(modes.iloc[0]) if len(modes) else "—"
    else:
        common = "—"
    st.metric("Most common event type", common)

st.markdown(
    """
<div class="ws-section-head">
  <h2 class="ws-h2">Top signals this week</h2>
  <p class="ws-section-sub">Highest-scoring items from the last 7 days (falls back to top overall if none fall in that window).</p>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Top signals this week (last 7 days, or top overall if none in window)
# -----------------------------------------------------------------------------
if n_total == 0:
    top_week = signals_df.iloc[:0]
    used_week_fallback = False
else:
    now = pd.Timestamp.now()
    week_ago = now - pd.Timedelta(days=7)
    dated = signals_df["event_date"].notna()
    in_week = signals_df.loc[dated & (signals_df["event_date"] >= week_ago)]
    if len(in_week) > 0:
        # Filter for quality
        in_week = in_week[in_week["quality_score"] >= 2]
        in_week = in_week[~((in_week["event_type"] == "Other") & (in_week["quality_score"] < 3))]
        in_week = in_week[~((in_week["person_name"] == "") & (in_week["company_name"] == "Unknown"))]
        top_week = in_week.sort_values(["quality_score", "score"], ascending=[False, False]).head(5)
        used_week_fallback = False
    else:
        # Fallback to overall top, with quality filters
        overall = signals_df.copy()
        overall = overall[overall["quality_score"] >= 2]
        overall = overall[~((overall["event_type"] == "Other") & (overall["quality_score"] < 3))]
        overall = overall[~((overall["person_name"] == "") & (overall["company_name"] == "Unknown"))]
        top_week = overall.sort_values(["quality_score", "score"], ascending=[False, False]).head(5)
        used_week_fallback = True

if used_week_fallback and n_total > 0:
    st.caption("No dated signals in the last 7 days — showing the top 5 by score overall.")

if len(top_week) == 0 and n_total > 0:
    st.info("No rows to highlight.")
elif len(top_week) > 0:
    for i, (_, row) in enumerate(top_week.iterrows()):
        person = row["person_name"] or "—"
        company = row["company_name"] or "Unknown"
        pe, ce = html.escape(str(person)), html.escape(str(company))
        out_e = html.escape(str(row["outreach_angle"]))
        new_html = new_pill_html() if is_signal_new(row.get("detected_at")) else ""
        ago = human_time_ago(row.get("detected_at"))
        href = safe_href(row["source_url"])
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(
                    f'<p class="ws-card-line"><strong>{pe}</strong> · <em>{ce}</em>{new_html}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p class="ws-card-line">{event_type_badge_html(row["event_type"])} {priority_badge_html(row["priority_level"])} · Score: {int(row["score"])}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p class="ws-card-line" style="font-size:0.85rem;">{out_e}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<p class="ws-card-meta">Detected {html.escape(ago)}</p>', unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f'<p class="ws-score-pill" style="text-align:right;margin:0;">{int(row["score"])}</p>',
                    unsafe_allow_html=True,
                )
            st.write(row["why_it_matters"])
            st.markdown(
                f'<p class="ws-link"><a href="{href}" target="_blank" rel="noopener noreferrer">Open source →</a></p>',
                unsafe_allow_html=True,
            )

st.markdown('<hr class="ws-rule"/>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main table (formatted for readability)
# -----------------------------------------------------------------------------
table_columns = [
    "person_name",
    "company_name",
    "event_type",
    "raw_title",
    "role",
    "event_date",
    "score",
    "priority_level",
    "outreach_angle",
    "why_it_matters",
    "source_url",
    "quality_score",
]

display_df = filtered[table_columns].copy()
if not display_df.empty:
    display_df.insert(0, "Label", filtered["detected_at"].apply(lambda x: "NEW" if is_signal_new(x) else ""))
    display_df["Detected"] = filtered["detected_at"].apply(human_time_ago)
if not display_df.empty and "event_date" in display_df.columns:
    display_df["event_date"] = pd.to_datetime(display_df["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    display_df["event_date"] = display_df["event_date"].fillna("—")

st.markdown(
    """
<div class="ws-section-head">
  <h2 class="ws-h2">All signals</h2>
  <p class="ws-section-sub">Full feed with filters — <strong>NEW</strong> = detected in the last 48 hours.</p>
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
            "company_name": st.column_config.TextColumn("Company", width="medium"),
            "event_type": st.column_config.TextColumn("Event", width="small"),
            "raw_title": st.column_config.TextColumn("Raw title (debug)", width="large"),
            "role": st.column_config.TextColumn("Role", width="small"),
            "event_date": st.column_config.TextColumn("Date", width="small"),
            "score": st.column_config.NumberColumn("Score", format="%d", width="small"),
            "priority_level": st.column_config.TextColumn("Priority", width="small"),
            "outreach_angle": st.column_config.TextColumn("Outreach angle", width="large"),
            "why_it_matters": st.column_config.TextColumn("Why it matters", width="large"),
            "Detected": st.column_config.TextColumn("Detected", width="small"),
            "source_url": st.column_config.LinkColumn("Source", width="medium"),
            "quality_score": st.column_config.NumberColumn("Quality", format="%d", width="small"),
        },
    )

# -----------------------------------------------------------------------------
# Details: expandable rows (full explanation + source)
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="ws-section-head">
  <h2 class="ws-h2">Details</h2>
  <p class="ws-section-sub">Expand a row for actions, full story, and source link.</p>
</div>
""",
    unsafe_allow_html=True,
)

for _, row in filtered.iterrows():
    person = row["person_name"] or "—"
    company = row["company_name"] or "Unknown"
    new_html = new_pill_html() if is_signal_new(row.get("detected_at")) else ""
    title_plain = f"{person} — {row['event_type']} @ {company}"
    # Expander title must stay plain text for accessibility; show badges inside panel
    ed = row["event_date"]
    if pd.isna(ed):
        date_str = "—"
    else:
        ts = pd.Timestamp(ed)
        date_str = ts.strftime("%Y-%m-%d")
    det = human_time_ago(row.get("detected_at"))
    with st.expander(title_plain):
        st.markdown(
            f'<p style="margin:0 0 0.75rem 0;">{new_html} {priority_badge_html(row["priority_level"])}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Raw title:** {row.get('raw_title', '—')}")
        st.markdown(f"**Priority:** {row['priority_level']}")
        st.markdown(f"**Detected:** {det}")
        st.markdown(f"**Outreach suggestion:** {row['outreach_angle']}")
        st.markdown(f"**Suggested next step:** {row['suggested_next_step']}")
        st.markdown("---")
        st.markdown(f"**Role:** {row['role'] or '—'}")
        st.markdown(f"**Date:** {date_str}")
        st.markdown(f"**Score:** {int(row['score'])}")
        st.markdown("**Why it matters**")
        st.write(row["why_it_matters"])
        st.markdown("**Full explanation**")
        st.write(row["full_explanation"] or "—")
        st.markdown("**Source**")
        st.markdown(f"[Open public source]({row['source_url']})")
