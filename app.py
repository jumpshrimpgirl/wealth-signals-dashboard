

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

from data import fetch_signals, format_wealth
from person_validation import is_valid_person
from prospect_display_gates import can_render_on_home
from prospect_processor import process_and_rank_prospects
from two_pass_pipeline import build_home_top_view


def _load_signals_df() -> pd.DataFrame:
    """Load signals, then rank by match × signal quality (``process_and_rank_prospects``)."""
    df = fetch_signals()
    df = process_and_rank_prospects(df)
    return df


def _safe_int_from_cell(val, default: int = 0) -> int:
    """Coerce a table cell to int. NaN/None/invalid → default (``int(nan)`` raises otherwise)."""
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


# How long a signal counts as "NEW" in the feed (hours)
NEW_WINDOW_HOURS = 48

# Curated hero blocks: score floor (full table uses sidebar minimum score only)
TOP_CURATED_MIN_SCORE = 40
# Homepage tab: max signals to show (ranked)
HOME_TOP_SIGNALS = 5

# Explore view: drop weak / noisy rows (after sidebar filters)
EXPLORE_MIN_SCORE = 30
# Columns searched by the sidebar text box (must stay aligned with Details / table labels).
SEARCHABLE_COLUMNS = (
    "person_name",
    "company_name",
    "raw_title",
    "role",
    "summary",
    "wealth_status",
    "priority_score",
    "priority_label",
    "signal_type",
    "wealth_signal_label",
    "liquidity_event",
    "client_type",
    "source_of_wealth",
    "why_it_matters",
    "ai_summary",
    "full_explanation",
    "est_wealth_display",
    "net_worth",
    "billionaire_company",
    "priority_level",
    "prospect_bio",
    "engine_pipeline_score",
)

# Company field: clear when it looks like a news outlet (substring match on normalized name)
KNOWN_OUTLETS_TOKENS = frozenset(
    {"bbc", "cnn", "nyt", "economist", "forbes", "reuters", "techcrunch", "bloomberg", "msnbc", "nbc"}
)

# Person field: drop obvious non-name tokens
BAD_PERSON_TOKENS = frozenset({"the", "a", "in", "on", "air force", "central alabama"})


def _fa_nonempty(val, *, fallback: str) -> str:
    t = str(val or "").strip()
    return t if t else fallback


def _cell_str(val, *, default: str = "") -> str:
    """Stringify a dataframe cell without ``pd.NA`` boolean ambiguity."""
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        return default
    s = str(val).strip()
    return s if s else default


def _boolish(val) -> str:
    """Yes/No for gate booleans that may be NA."""
    if val is True:
        return "yes"
    if val is False:
        return "no"
    try:
        if pd.isna(val):
            return "no"
    except (TypeError, ValueError):
        pass
    return "yes" if val else "no"


def wealth_signal_for_display(r: pd.Series) -> str:
    """AI refines rules-based label when present; same concept as sidebar / table."""
    a = str(r.get("ai_wealth_signal") or "").strip()
    if a:
        return a
    return _fa_nonempty(r.get("wealth_signal_label"), fallback="Data pending")


def liquidity_for_display(r: pd.Series) -> str:
    a = str(r.get("ai_liquidity_label") or "").strip()
    if a:
        return a
    w = str(r.get("liquidity_event") or "").strip()
    return w if w else "No clear liquidity event"


def client_type_for_display(r: pd.Series) -> str:
    return _fa_nonempty(r.get("client_type"), fallback="Unknown")


def source_of_wealth_for_display(r: pd.Series) -> str:
    return _fa_nonempty(r.get("source_of_wealth"), fallback="Data pending")


def prospect_who_for_display(r: pd.Series) -> str:
    a = str(r.get("ai_client_who") or "").strip()
    if a:
        return a
    pn = str(r.get("person_name") or "").strip()
    rl = str(r.get("role") or "").strip()
    if pn and rl:
        return f"{pn} — {rl}"
    if pn:
        return pn
    return "Not identified"


def why_money_one_liner(r: pd.Series) -> str:
    w = str(r.get("ai_why_money") or "").strip()
    if w:
        return w
    return _fa_nonempty(r.get("why_it_matters"), fallback="Data pending")


def apply_prospect_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Subset rows where ``query`` appears in any searchable prospect field (same scope as Details)."""
    q = (query or "").strip()
    if not q or df is None or df.empty:
        return df
    mask = pd.Series(False, index=df.index)
    for c in SEARCHABLE_COLUMNS:
        if c in df.columns:
            mask = mask | df[c].fillna("").astype(str).str.contains(q, case=False, na=False, regex=False)
    return df[mask]


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
    Explore / Home / Details: keep broad recall, floor by ``priority_score`` (same as ``score``),
    sort by match × signal ranking only.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    pn = out["person_name"].fillna("").astype(str).str.strip()
    out = out[pn != ""]
    if out.empty:
        return out
    ps = pd.to_numeric(out.get("priority_score", out["score"]), errors="coerce").fillna(0)
    out = out[ps >= EXPLORE_MIN_SCORE]
    if out.empty:
        return out
    if "source_url" in out.columns:
        out["source_outlet"] = out["source_url"].apply(_source_outlet_from_url)
    else:
        out["source_outlet"] = ""
    out["company_name"] = out["company_name"].fillna("").astype(str).map(_company_clear_if_known_outlet)
    out["person_name"] = out["person_name"].fillna("").astype(str).map(_person_clean_token)
    out = out[out["person_name"].str.strip() != ""]
    if "ai_summary" not in out.columns:
        out["ai_summary"] = ""
    out["ai_summary"] = out.apply(
        lambda r: (str(r.get("ai_summary") or "").strip() or str(r.get("raw_title") or "")),
        axis=1,
    )
    out["target_client"] = out.apply(
        lambda r: "YES" if _safe_int_from_cell(r.get("priority_score", r.get("score")), 0) >= 70 else "NO",
        axis=1,
    )
    for q in ("quality_score", "confidence_score"):
        if q not in out.columns:
            out[q] = 0
        else:
            out[q] = pd.to_numeric(out[q], errors="coerce").fillna(0)
    out["score"] = pd.to_numeric(out.get("priority_score", out["score"]), errors="coerce").fillna(0)
    if "event_date" not in out.columns:
        out["event_date"] = pd.NaT
    out["wealth_signal_display"] = out.apply(wealth_signal_for_display, axis=1)
    out["liquidity_display"] = out.apply(liquidity_for_display, axis=1)
    sort_key = "priority_score" if "priority_score" in out.columns else "score"
    out = out.sort_values(by=sort_key, ascending=False, na_position="last")
    return out.reset_index(drop=True)


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
    if not cn or cn.lower() in ("unknown", "data pending"):
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
    for col in [
        "person_name",
        "additional_people",
        "company_name",
        "role",
        "event_type",
        "score",
        "quality_score",
        "confidence_score",
    ]:
        if col not in df.columns:
            if col == "score" or col in ("quality_score", "confidence_score"):
                df[col] = 0
            elif col == "additional_people":
                df[col] = "[]"
            else:
                df[col] = ""


def ensure_columns_present(df: pd.DataFrame, columns: list[str]) -> None:
    """Add missing columns with safe defaults (strings empty, numeric scores 0, event_date NaT)."""
    for col in columns:
        if col not in df.columns:
            if col == "wealth_rank":
                df[col] = 3
            elif col in ("score", "quality_score", "confidence_score", "ai_fa_usefulness_score", "ai_extraction_confidence"):
                df[col] = 0
            elif col == "ai_rerank_priority":
                df[col] = pd.NA
            elif col == "engine_pipeline_score":
                df[col] = pd.NA
            elif col in ("event_date", "detected_at"):
                df[col] = pd.NaT
            elif col.startswith("fa_pass_gate_") or col.startswith("pv_gate_"):
                df[col] = False
            else:
                df[col] = ""


def rank_for_hero_sections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback ordering for Home: wealth-signal strength first, then event-type tier, then clarity.

    When AI re-ranking / usefulness scores exist, they take precedence (same keys as Explore).
    """
    if df is None or df.empty:
        return df
    d = df.copy()
    ensure_required_signal_columns(d)
    if "confidence_score" not in d.columns:
        d["confidence_score"] = 0
    if "quality_score" not in d.columns:
        d["quality_score"] = 0
    if "wealth_rank" not in d.columns:
        d["wealth_rank"] = 3
    d["wealth_rank"] = pd.to_numeric(d["wealth_rank"], errors="coerce").fillna(3).astype(int)
    if "event_date" not in d.columns:
        d["event_date"] = pd.NaT
    d["_ev"] = d["event_type"].map(EVENT_TYPE_RANK).fillna(0).astype(int)
    d["_pn"] = (d["person_name"].fillna("").str.strip() != "").astype(int)
    d["_cn"] = ((d["company_name"].fillna("").str.strip() != "") & (d["company_name"] != "Unknown")).astype(int)
    sort_cols: list[str] = []
    sort_asc: list[bool] = []
    if "ai_rerank_priority" in d.columns:
        d["ai_rerank_priority"] = pd.to_numeric(d["ai_rerank_priority"], errors="coerce")
        if bool(d["ai_rerank_priority"].notna().any()):
            sort_cols.append("ai_rerank_priority")
            sort_asc.append(True)
    if "ai_fa_usefulness_score" in d.columns:
        d["ai_fa_usefulness_score"] = pd.to_numeric(d["ai_fa_usefulness_score"], errors="coerce").fillna(0)
        sort_cols.append("ai_fa_usefulness_score")
        sort_asc.append(False)
    sort_cols.extend(["wealth_rank", "_pn", "event_date", "_ev", "quality_score", "score", "_cn"])
    sort_asc.extend([True, False, False, False, False, False, False])
    d = d.sort_values(by=sort_cols, ascending=sort_asc, na_position="last")
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

  /* ----- Main tabs: Home / Explore & data — black (tablist only; not panel buttons) ----- */
  [data-testid="stTabs"] [role="tablist"] {
    gap: 0.25rem !important;
    border-bottom-color: #000000 !important;
  }
  [data-testid="stTabs"] [role="tablist"] [role="tab"],
  [data-testid="stTabs"] [role="tablist"] button[data-baseweb="tab"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    border-color: #262626 !important;
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important;
  }
  [data-testid="stTabs"] [role="tablist"] [role="tab"][aria-selected="false"],
  [data-testid="stTabs"] [role="tablist"] button[aria-selected="false"] {
    background-color: #171717 !important;
    color: #e5e5e5 !important;
  }
  [data-testid="stTabs"] [role="tablist"] [role="tab"][aria-selected="true"],
  [data-testid="stTabs"] [role="tablist"] button[aria-selected="true"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    border-bottom: 2px solid #ffffff !important;
  }
  [data-testid="stTabs"] [role="tablist"] [role="tab"]:hover,
  [data-testid="stTabs"] [role="tablist"] button:hover {
    background-color: #262626 !important;
    color: #ffffff !important;
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
  .ws-enrichment-label {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    color: #9a3412;
    background: #ffedd5;
    border: 1px solid #fb923c;
    border-radius: 999px;
    padding: 0.15rem 0.5rem;
    margin-left: 0.35rem;
    vertical-align: middle;
    white-space: nowrap;
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
    if lv == "Elite":
        cls = "ws-badge ws-badge--high"
    elif lv == "High":
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


def rank_signals_with_ai(df: pd.DataFrame) -> list[dict]:
    """
    Legacy AI rerank — **disabled**. Ranking is exclusively from ``process_and_rank_prospects``.
    """
    return []


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
    """Rich card for Home tab (Pass-2 Home score when available)."""
    if not can_render_on_home(row):
        return
    header_line = format_signal_header_line(row)
    hl = html.escape(header_line)
    _line = str(row.get("ai_summary", "") or "").strip() or str(row.get("summary", "") or "")[:220]
    out_e = html.escape(_line)
    new_html = new_pill_html() if is_signal_new(row.get("detected_at")) else ""
    ago = human_time_ago(row.get("detected_at"))
    href = safe_href(str(row.get("source_url", "")))
    _sig_t = str(row.get("signal_type") or row.get("event_type") or "").strip()
    _co_disp = str(row.get("company_name") or row.get("company") or "").strip()
    _ew = str(row.get("est_wealth_display") or row.get("est_wealth") or "").strip()
    _cl = str(row.get("client_likelihood") or "Medium").strip()
    _why_fa = str(row.get("why_this_matters_fa") or row.get("article_relevance_reason") or "").strip()
    _tplab = row.get("top5_score")
    try:
        _has_top5 = _tplab is not None and not pd.isna(_tplab)
    except Exception:
        _has_top5 = bool(_tplab)
    if _has_top5:
        _plab = str(row.get("home_priority_label") or row.get("priority_label") or "").strip()
    else:
        _plab = str(row.get("priority_label") or row.get("priority_level") or "").strip()
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(
                f"""<p class="ws-card-line"><strong>{hl}</strong>{target_client_badge_html(row)}{billionaire_badge_html(row)}{new_html}</p>""",
                unsafe_allow_html=True,
            )
            _co_html = html.escape(_co_disp) if _co_disp else "—"
            _ew_html = html.escape(_ew) if _ew else "—"
            _cl_html = html.escape(_cl) if _cl else "—"
            _why_html = html.escape(_why_fa) if _why_fa else ""
            st.markdown(
                f"""<p class="ws-card-line">{event_type_badge_html(_sig_t)} {priority_badge_html(_plab)} | """
                f"""Client likelihood: {_cl_html} | {_co_html} | Wealth: {_ew_html}</p>""",
                unsafe_allow_html=True,
            )
            if _why_html:
                st.markdown(
                    f"""<p class="ws-card-line" style="font-size:0.92rem;color:#E5E7EB;">"""
                    f"""<strong>Why this matters:</strong> {_why_html}</p>""",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"""<p class="ws-card-line" style="font-size:0.9rem;color:#9CA3AF;">{out_e}</p>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""<p class="ws-card-meta">Detected {html.escape(ago)}</p>""",
                unsafe_allow_html=True,
            )
        with c2:
            _ps = _safe_int_from_cell(
                row.get("top5_score") if _has_top5 else row.get("priority_score", row.get("score")),
                0,
            )
            st.markdown(
                f"""<p style="text-align:right;margin:0;font-size:0.85rem;color:#9CA3AF;">Priority</p>"""
                f"""<p style="text-align:right;margin:0;font-size:1.05rem;font-weight:700;">{_ps}</p>""",
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
    st.session_state.signals_df = _load_signals_df()

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
        "est_wealth_display",
        "wealth_rank",
        "wealth_signal_label",
        "liquidity_event",
        "client_type",
        "source_of_wealth",
        "ai_wealth_signal",
        "ai_liquidity_label",
        "ai_client_who",
        "ai_why_money",
        "extracted_person_name",
        "extracted_role",
        "extracted_company",
        "summary",
        "signal_score",
        "match_score",
        "priority_score",
        "priority_label",
        "est_wealth",
        "signal_type",
        "name",
        "company",
        "source_title",
        "ai_rerank_priority",
        "ai_why_flagged",
        "ai_why_matters_fa",
        "ai_cluster_fingerprint",
        "cluster_group_id",
        "ai_net_worth_inferred",
        "ai_wealth_estimate_confidence",
        "ai_extraction_confidence",
        "fa_prospect_identified",
        "fa_structured_name",
        "fa_relevance",
        "fa_why_one_sentence",
        "fa_pass_gate_person",
        "fa_pass_gate_wealth_substance",
        "fa_pass_gate_liquidity_or_uhnw",
        "fa_pass_gate_fa_relevance",
        "fa_suppression_level",
        "fa_suppression_reason",
        "fa_priority_debug",
        "pv_prospect_identified",
        "pv_display_name",
        "pv_role_title",
        "pv_wealth_signal",
        "pv_liquidity_event",
        "pv_fa_relevance",
        "pv_why_it_matters",
        "pv_gate_prospect_pass",
        "pv_gate_wealth_pass",
        "pv_gate_fa_relevance_pass",
        "pv_validation_debug",
        "pv_estimated_wealth_display",
        "person_name_validation",
        "extraction_audit_json",
        "liquidity_event_hint",
        "wealth_signal_hint",
        "wealth_signal_raw_hint",
        "ingest_overall_extraction_confidence",
        "prospect_bio",
        "engine_pipeline_score",
        "published_at",
        "recency_score",
        "wealth_status",
        "wealth_relevance",
        "article_relevance_reason",
        "top5_score",
        "top5_reason",
        "keep_for_home",
        "home_priority_label",
        "pass2_ai_subscore",
    ],
)

# -----------------------------------------------------------------------------
# Header row: title + refresh
# -----------------------------------------------------------------------------
with st.container(border=True, key="ws_hero_bar"):
    st.markdown(
        """
<h2 style='margin-bottom:0;'>Wealth Signals</h2>
<p style='color:#9CA3AF; margin-top:0;'>Wealth prospecting for financial advisors</p>
""",
        unsafe_allow_html=True,
    )
    st.caption(
        "Prospect feed from public RSS (sample if live fetch fails). Built for financial advisors scanning for "
        "wealth signals—not a general news dashboard. Demo only — not investment advice."
    )
    _df = signals_df
    if len(_df) > 0 and _df["detected_at"].notna().any():
        lu = _df["detected_at"].max()
        st.markdown(
            f"""<div class="ws-last-updated"><strong>{html.escape(format_detected_utc(lu))}</strong>"""
            f"""<span>|</span><em>{html.escape(human_time_ago(lu))}</em></div>""",
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Sidebar: same prospect schema as the table & Details (wealth-focused, not generic news)
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """<p class="ws-sidebar-filters-title">Find prospects</p>""",
    unsafe_allow_html=True,
)
st.sidebar.caption(
    "Filter by the **same fields** shown in Details: names, roles, companies, wealth signal, "
    "liquidity, client type, source-of-wealth hints, story text, and AI notes. "
    "Use this to find founders, executives, investors, athletes, heirs, liquidity events, and estimated-wealth cues."
)

search_query = st.sidebar.text_input(
    "Search prospects",
    placeholder="Name, company, role, wealth signal, liquidity, client type, source of wealth…",
    help=(
        "Case-insensitive partial match across prospect fields: Name, Role / title, Company, Wealth signal, "
        "Liquidity event, Client type, Source of wealth, Why it matters, AI summary, estimated wealth display, "
        "article text, and priority. Same scope as the Details panel."
    ),
).strip()

_event_opts = sorted(signals_df.get("event_type", pd.Series(dtype=object)).replace("", pd.NA).dropna().unique().tolist())
if not _event_opts:
    _event_opts = ["Founder Exit", "Funding", "Promotion", "Board Appointment", "Other"]

selected_types = st.sidebar.multiselect(
    "Story category",
    options=_event_opts,
    default=_event_opts,
    help="Pipeline event bucket (same as in row details). Leave all selected to include every category.",
)

_ws_present = set(signals_df.get("wealth_signal_label", pd.Series(dtype=object)).replace("", pd.NA).dropna().unique().tolist())
_ws_order = ("Strong", "Moderate", "Weak", "None")
_wealth_sig_options = [x for x in _ws_order if x in _ws_present] or list(_ws_order)
selected_wealth_signals = st.sidebar.multiselect(
    "Wealth signal",
    options=_wealth_sig_options,
    default=_wealth_sig_options,
    help=(
        "How strong the **money-relevant** signal is (rule-based labels; AI may restate in Details). "
        "Strong = clear liquidity, large capital, or known ultra-wealth context."
    ),
)

_liq_present = set(signals_df.get("liquidity_event", pd.Series(dtype=object)).replace("", pd.NA).dropna().unique().tolist())
_liq_order = ("Yes", "Potential", "No")
_liq_options = [x for x in _liq_order if x in _liq_present] or list(_liq_order)
selected_liquidity = st.sidebar.multiselect(
    "Liquidity event",
    options=_liq_options,
    default=_liq_options,
    help="Whether the story describes cash/stock/realization **now or soon** (Yes / Potential / No).",
)

_ct_series = signals_df.get("client_type", pd.Series(dtype=object)).fillna("").astype(str).str.strip()
_ct_opts = sorted({_fa_nonempty(x, fallback="Unknown") for x in _ct_series.tolist()})
if "Unknown" not in _ct_opts:
    _ct_opts.append("Unknown")
    _ct_opts = sorted(_ct_opts)
if not _ct_opts:
    _ct_opts = ["Unknown", "Executive", "Founder / Entrepreneur"]
selected_client_types = st.sidebar.multiselect(
    "Client type",
    options=_ct_opts,
    default=_ct_opts,
    help="Prospect archetype (same **Client type** field in Details): founder, executive, investor, athlete, heir, etc.",
)

min_score = st.sidebar.slider(
    "Minimum priority score",
    min_value=0,
    max_value=100,
    value=TOP_CURATED_MIN_SCORE,
    help=(
        "Floor for **priority_score** (signal strength + match quality, 0–100). "
        f"Applies to Home top {HOME_TOP_SIGNALS}, Explore table, and Details."
    ),
)

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
        "est_wealth_display",
        "wealth_rank",
        "wealth_signal_label",
        "liquidity_event",
        "client_type",
        "source_of_wealth",
        "ai_wealth_signal",
        "ai_liquidity_label",
        "ai_client_who",
        "ai_why_money",
        "extracted_person_name",
        "extracted_role",
        "extracted_company",
        "summary",
        "signal_score",
        "match_score",
        "priority_score",
        "priority_label",
        "est_wealth",
        "signal_type",
        "name",
        "company",
        "source_title",
        "ai_rerank_priority",
        "ai_why_flagged",
        "ai_why_matters_fa",
        "ai_cluster_fingerprint",
        "cluster_group_id",
        "ai_net_worth_inferred",
        "ai_wealth_estimate_confidence",
        "ai_extraction_confidence",
        "fa_prospect_identified",
        "fa_structured_name",
        "fa_relevance",
        "fa_why_one_sentence",
        "fa_pass_gate_person",
        "fa_pass_gate_wealth_substance",
        "fa_pass_gate_liquidity_or_uhnw",
        "fa_pass_gate_fa_relevance",
        "fa_suppression_level",
        "fa_suppression_reason",
        "fa_priority_debug",
        "pv_prospect_identified",
        "pv_display_name",
        "pv_role_title",
        "pv_wealth_signal",
        "pv_liquidity_event",
        "pv_fa_relevance",
        "pv_why_it_matters",
        "pv_gate_prospect_pass",
        "pv_gate_wealth_pass",
        "pv_gate_fa_relevance_pass",
        "pv_validation_debug",
        "pv_estimated_wealth_display",
        "person_name_validation",
        "extraction_audit_json",
        "liquidity_event_hint",
        "wealth_signal_hint",
        "wealth_signal_raw_hint",
        "ingest_overall_extraction_confidence",
        "prospect_bio",
        "engine_pipeline_score",
        "published_at",
        "recency_score",
        "wealth_status",
        "wealth_relevance",
        "article_relevance_reason",
        "top5_score",
        "top5_reason",
        "keep_for_home",
        "home_priority_label",
        "pass2_ai_subscore",
    ],
)

if selected_types:
    filtered = filtered[filtered["event_type"].isin(selected_types)]
else:
    filtered = filtered.iloc[0:0]

filtered = filtered[filtered["score"] >= min_score]

if selected_wealth_signals and "wealth_signal_label" in filtered.columns:
    filtered = filtered[filtered["wealth_signal_label"].isin(selected_wealth_signals)]

if selected_liquidity and "liquidity_event" in filtered.columns:
    filtered = filtered[filtered["liquidity_event"].isin(selected_liquidity)]

if selected_client_types and "client_type" in filtered.columns:
    fc = filtered["client_type"].fillna("").astype(str).str.strip().replace("", "Unknown")
    filtered = filtered[fc.isin(selected_client_types)]

if search_query:
    filtered = apply_prospect_search(filtered, search_query)

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
    if "social_rss_feeds_configured" in _ingest:
        st.caption(
            f"Social RSS feeds in config: **{_ingest['social_rss_feeds_configured']}** "
            "(Mastodon tags + optional `WEALTH_SOCIAL_RSS_URLS`)."
        )
    st.divider()
    st.markdown("**Current view (after filters + explore rules)**")
    st.metric("Rows shown", len(explore_view))
    if len(explore_view) > 0:
        miss_p = int((explore_view.get("person_name", pd.Series([""] * len(explore_view))).fillna("") == "").sum())
        miss_r = int((explore_view.get("role", pd.Series([""] * len(explore_view))).fillna("") == "").sum())
        st.caption(f"Missing person_name: **{miss_p}** | Missing role: **{miss_r}**")
        st.markdown("**Counts by event_type**")
        _vc = explore_view.get("event_type", pd.Series(dtype=object)).value_counts().rename_axis("event_type").reset_index(name="count")
        st.dataframe(_vc, hide_index=True, width="stretch")
        _ea = explore_view.iloc[0].get("extraction_audit_json", "")
        if str(_ea or "").strip():
            st.divider()
            st.markdown("**Structured extraction audit (first visible row)**")
            st.caption("Field-level provenance, confidence, and missingness (ingest pass 1 + optional pass 2).")
            try:
                st.json(json.loads(str(_ea)))
            except (json.JSONDecodeError, TypeError):
                st.code(str(_ea)[:2000], language=None)
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
            st.subheader("Top prospects")
        with col2:
            st.markdown('<div style="height:0.35rem"></div>', unsafe_allow_html=True)
            if st.button(
                "Refresh",
                width="stretch",
                help="Fetch latest RSS / sample data again",
                key="home_refresh_signals",
            ):
                st.session_state.signals_df = _load_signals_df()
                st.rerun()
        st.caption(
            f"Top **{HOME_TOP_SIGNALS}** use **Pass-2 Home ranking** (strict AI + recency + verification) on the top pool "
            "from Pass-1 scores — not the same ordering as the full table."
        )

        if len(signals_df) == 0:
            st.info("No signals loaded yet - try **Refresh data**.")
        elif len(filtered) == 0:
            st.info("No rows match your filters - widen event types or lower the minimum score, or open **Explore & data**.")
        elif len(explore_view) == 0:
            st.info(
                f"No signals pass the current floor (priority score ≥ **{EXPLORE_MIN_SCORE}**). "
                "Lower the minimum score slider or widen filters."
            )
        else:
            top_home = build_home_top_view(explore_view, HOME_TOP_SIGNALS)
            for _, row in top_home.iterrows():
                render_home_signal_card(row)
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
                et = signals_df["event_type"].replace("", pd.NA).dropna()
                et = et[~et.astype(str).str.strip().str.lower().eq("other")]
                if et.empty:
                    common = "-"
                else:
                    modes = et.mode()
                    common = str(modes.iloc[0]).strip() if len(modes) else "-"
            else:
                common = "-"
            st.metric("Most common event type", common)

    st.markdown("""<hr class="ws-rule"/>""", unsafe_allow_html=True)
    
    # -----------------------------------------------------------------------------
    # Main table (narrow columns — same data as Details / Home)
    # -----------------------------------------------------------------------------
    _PROC_COLS = [
        "name",
        "role",
        "company",
        "signal_type",
        "signal_score",
        "match_score",
        "priority_score",
        "priority_label",
        "est_wealth",
        "source_title",
        "source_url",
        "summary",
    ]
    _PROC_RENAME = {
        "name": "Name",
        "role": "Role / title",
        "company": "Company",
        "signal_type": "Signal type",
        "signal_score": "Signal",
        "match_score": "Match",
        "priority_score": "Priority",
        "priority_label": "Tier",
        "est_wealth": "Est. wealth",
        "source_title": "Headline",
        "source_url": "URL",
        "summary": "Summary",
    }

    st.markdown(
        """
    <div class="ws-section-head">
      <h2 class="ws-h2">Prospect table</h2>
      <p class="ws-section-sub">Signal (article) + match (person) → <strong>priority_score</strong>. Same ranked rows as Home.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.write(
        f"Showing **{len(explore_view)}** row(s) (priority score ≥ **{EXPLORE_MIN_SCORE}**, sidebar filters applied)."
    )

    if explore_view.empty:
        st.info(
            "No rows in this view. Try clearing search, widening filters, or lowering the minimum priority score."
        )
    else:
        st.caption(
            "**Priority** = signal_score (0–60, article/event) + match_score (0–40, person fit). "
            "Tier: Elite / High / Medium / Low."
        )
        _show = explore_view.copy()
        for c in _PROC_COLS:
            if c not in _show.columns:
                _show[c] = ""
        disp = _show[_PROC_COLS].rename(columns=_PROC_RENAME)
        if os.environ.get("WEALTH_SIGNALS_DEV_DEBUG", "").lower() in ("1", "true", "yes"):
            if "debug_signal_reasons" in _show.columns:
                disp["debug_signal"] = _show["debug_signal_reasons"].fillna("")
            if "debug_match_reasons" in _show.columns:
                disp["debug_match"] = _show["debug_match_reasons"].fillna("")
        st.dataframe(
            disp,
            width="stretch",
            hide_index=True,
            column_config={
                "Name": st.column_config.TextColumn("Name", width="medium"),
                "Role / title": st.column_config.TextColumn("Role / title", width="medium"),
                "Company": st.column_config.TextColumn("Company", width="medium"),
                "Signal type": st.column_config.TextColumn("Signal type", width="small"),
                "Signal": st.column_config.NumberColumn("Signal", width="small", format="%d"),
                "Match": st.column_config.NumberColumn("Match", width="small", format="%d"),
                "Priority": st.column_config.NumberColumn("Priority", width="small", format="%d"),
                "Tier": st.column_config.TextColumn("Tier", width="small"),
                "Est. wealth": st.column_config.TextColumn("Est. wealth", width="small"),
                "Headline": st.column_config.TextColumn("Headline", width="large"),
                "URL": st.column_config.LinkColumn("URL", width="small"),
                "Summary": st.column_config.TextColumn("Summary", width="large"),
            },
        )
    
    # -----------------------------------------------------------------------------
    # Details: compact scrollable panel (expandable rows)
    # -----------------------------------------------------------------------------
    with st.container(border=True, key="ws_details_explorer"):
        st.markdown(
            """
    <div class="ws-section-head">
      <h2 class="ws-h2">Details</h2>
      <p class="ws-section-sub">Same rows as the <strong>Prospect table</strong> and the same filters as the sidebar—one schema everywhere.</p>
    </div>
    """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Fields below match **Find prospects**: Name, Role / title, Company, Wealth signal, Liquidity event, "
            "estimated wealth, Source of wealth, Client type, and AI copy. Empty values show **Data pending** or **Not identified**."
        )
        st.markdown(
            """<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#0f172a;margin:0 0 0.4rem 0;">Filter</p>""",
            unsafe_allow_html=True,
        )

        def _details_priority_label(v: str) -> str:
            return {
                "all": "All tiers",
                "elite": "Elite",
                "high": "High",
                "medium": "Medium",
                "low": "Low",
            }[v]

        priority_filter = st.radio(
            "Tier",
            ["all", "elite", "high", "medium", "low"],
            horizontal=True,
            format_func=_details_priority_label,
            key="details_priority_filter",
            help="Filter by match×signal tier (Elite 90+, High 75+, Medium 55+, Low under 55).",
        )

        details_filtered = explore_view.copy()
        if priority_filter != "all" and "priority_label" in details_filtered.columns:
            _pl = {
                "elite": "Elite",
                "high": "High",
                "medium": "Medium",
                "low": "Low",
            }[priority_filter]
            details_filtered = details_filtered[
                details_filtered["priority_label"].astype(str).str.strip() == _pl
            ]
        elif priority_filter != "all" and "priority_level" in details_filtered.columns:
            _pl = {"elite": "Elite", "high": "High", "medium": "Medium", "low": "Low"}[priority_filter]
            details_filtered = details_filtered[
                details_filtered["priority_level"].astype(str).str.strip() == _pl
            ]
    
        with st.container(height=580, border=False, key="ws_details_scroll"):
            if details_filtered.empty:
                st.caption("No rows match this tier. Set Tier to All tiers.")
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
                        _tier = str(row.get("priority_label") or row.get("priority_level") or "").strip()
                        st.markdown(
                            f"""<p style="margin:0 0 0.75rem 0;">{target_client_badge_html(row)}{billionaire_badge_html(row)}{new_html} {priority_badge_html(_tier)}</p>""",
                            unsafe_allow_html=True,
                        )
                        if row.get("is_billionaire"):
                            st.caption(
                                f"💰 Billionaire list match: net worth {_fa_nonempty(row.get('net_worth'), fallback='Data pending')} "
                                f"({_fa_nonempty(row.get('billionaire_company'), fallback='Not identified')})"
                            )
                        if also:
                            st.caption(f"Also named in story: {also}")

                        st.markdown("##### Prospect")
                        _pn_disp = _cell_str(row.get("person_name"))
                        _rn = _cell_str(row.get("role"))
                        if not _pn_disp:
                            _pn_disp = "Not identified"
                        if _rn and _pn_disp != "Not identified" and _rn.lower() == _pn_disp.lower():
                            _rn = ""
                        st.markdown(f"**Name:** {_pn_disp}")
                        st.markdown(f"**Role / title:** {_rn if _rn else 'Not identified'}")
                        _pbio = _cell_str(row.get("prospect_bio"))
                        if _pbio:
                            st.markdown(f"**Bio:** {_pbio}")
                        _pval = _cell_str(row.get("person_name_validation"))
                        if _pval and _pval != "ok":
                            st.caption(f"Name validation: {_pval}")
                        st.markdown(f"**Company:** {_fa_nonempty(row.get('company_name'), fallback='Not identified')}")

                        st.markdown("##### Match × signal scores")
                        _ps = _safe_int_from_cell(row.get("priority_score", row.get("score")), 0)
                        _ss = _safe_int_from_cell(row.get("signal_score"), 0)
                        _ms = _safe_int_from_cell(row.get("match_score"), 0)
                        st.markdown(
                            f"**Priority:** {_ps} (signal {_ss} + match {_ms}) · **Tier:** {_cell_str(row.get('priority_label'))}"
                        )
                        st.markdown(f"**Est. wealth (processor):** {_cell_str(row.get('est_wealth'))}")

                        st.markdown("##### Wealth & liquidity")
                        st.markdown(f"**Wealth signal:** {wealth_signal_for_display(row)}")
                        st.caption("Strong / Moderate / Weak / None — how clearly the story implies money, liquidity, or ultra-wealth.")
                        st.markdown(f"**Liquidity event:** {liquidity_for_display(row)}")
                        st.caption("Yes / Potential / No — cash, stock sale, IPO, round, compensation event, etc.")

                        _ew_disp = str(row.get("est_wealth_display") or "").strip() or format_wealth(
                            row.get("estimated_wealth")
                        )
                        _agg_disp = format_wealth(row.get("aggregated_estimated_wealth"))
                        st.markdown(f"**Estimated wealth (this row):** {_ew_disp}")
                        st.markdown(f"**Net worth / aggregated (same person in feed):** {_agg_disp}")
                        st.caption("“Data pending” when no reliable estimate; aggregate sums per person in this feed.")

                        st.markdown(f"**Source of wealth:** {source_of_wealth_for_display(row)}")
                        st.caption("Channel for the wealth (e.g. M&A, IPO, equity round, compensation, inheritance).")

                        st.markdown(f"**Client type:** {client_type_for_display(row)}")
                        st.caption("Advisor archetype: founder, executive, investor, athlete, heir, or unknown.")

                        st.markdown(f"**Why it matters:** {_fa_nonempty(row.get('why_it_matters'), fallback='Data pending')}")

                        st.markdown("##### AI summary")
                        _ai_s = str(row.get("ai_summary", "") or "").strip()
                        _ai_w = str(row.get("ai_why_it_matters", "") or "").strip()
                        _ai_o = str(row.get("ai_outreach", "") or "").strip()
                        _ai_wm = str(row.get("ai_why_money") or "").strip()
                        _who = str(row.get("ai_client_who") or "").strip() or prospect_who_for_display(row)
                        st.markdown(f"**Who is the prospect:** {_who}")
                        if _ai_wm:
                            st.markdown(f"**Why it matters (money / prospect value):** {_ai_wm}")
                        if _ai_s:
                            st.markdown(f"**Summary:** {_ai_s}")
                        if _ai_w:
                            st.markdown(f"**Advisor context:** {_ai_w}")
                        if _ai_o:
                            st.markdown(f"**Outreach angle:** {_ai_o}")

                        st.markdown("##### Article & source")
                        st.markdown(f"**Headline:** {row.get('raw_title') or '—'}")
                        _url = str(row.get("source_url") or "").strip()
                        if _url:
                            st.markdown(f"**Article:** [Open public source]({safe_href(_url)})")
                        _so = str(row.get("source_outlet") or "").strip()
                        if _so:
                            st.caption(f"Publisher / domain: {_so}")

                        _tc = row.get("target_client")
                        _tc_label = (
                            "yes"
                            if _tc is True
                            or str(_tc).lower() == "true"
                            or str(_tc).strip().upper() == "YES"
                            else ("mid" if _tc == "mid" or str(_tc).lower() == "mid" else "no")
                        )
                        st.markdown("##### Secondary (pipeline)")
                        st.markdown(f"**Story category:** {row.get('event_type') or '—'}")
                        st.markdown(
                            f"**Pipeline score:** {int(row.get('score', 0) or 0)} · "
                            f"**Priority:** {row.get('priority_level') or '—'} · "
                            f"**Wealth score (rules):** {int(row.get('wealth_score', 0) or 0)} · "
                            f"**Target client flag:** {_tc_label}"
                        )
                        st.caption(f"Detected {det} · Event date: {date_str}")
                        st.markdown(f"**Outreach template:** {row.get('outreach_angle', '')}")
                        st.markdown(f"**Suggested next step:** {row.get('suggested_next_step', '')}")
                        st.caption(
                            f"Repeat person = {bool(row.get('repeat_person'))} · "
                            f"Linked wealth signal = {bool(row.get('linked_wealth_signal'))} · "
                            f"Repeat company = {bool(row.get('repeat_company'))}"
                        )
                        st.markdown("**Full text (snippet)**")
                        st.write(_fa_nonempty(row.get("full_explanation"), fallback="Data pending"))
