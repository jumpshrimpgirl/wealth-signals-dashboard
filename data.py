"""
Data layer: sample signals (fallback) and live RSS-based signals.

Uses public RSS feeds only - no LinkedIn or private sources.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import feedparser
import pandas as pd
import requests

from ai_extraction import extract_signal_with_ai
from ai_interpretation import enrich_dataframe_with_ai_interpretation
from person_validation import is_valid_person
from score import (
    apply_scores,
    clamp_score_0_100,
    classify_client_type,
    classify_liquidity_event,
    classify_wealth_signal_strength,
    compute_signal_score,
    derive_wealth_priority_level,
    infer_source_of_wealth,
    is_macro_noise_without_wealth_hook,
    passes_wealth_high_priority_gate,
    wealth_signal_rank,
)

# When True, ``finalize_dataframe`` emits one row per detected person (shared headline / URL).
SPLIT_SIGNAL_ROWS_PER_PERSON = False

# -----------------------------------------------------------------------------
# Column contract: every row returned to the app must have these keys.
# -----------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "person_name",
    "additional_people",
    "company_name",
    "industry",
    "stage",
    "company_description",
    "company_location",
    "funding_amount",
    "funding_stage",
    "event_type",
    "raw_title",
    "role",
    "event_date",
    "detected_at",
    "score",
    "why_it_matters",
    "outreach_angle",
    "priority_level",
    "suggested_next_step",
    "ai_summary",
    "ai_why_it_matters",
    "ai_outreach",
    "source_url",
    "full_explanation",
    "quality_score",
    "confidence_score",
    "is_relevant",
    "weak_signal",
    "wealth_score",
    "estimated_wealth",
    "est_wealth_display",
    "aggregated_estimated_wealth",
    "target_client",
    "is_billionaire",
    "net_worth",
    "billionaire_company",
    "priority",
    "repeat_person",
    "linked_wealth_signal",
    "repeat_company",
    "wealth_passes_gate",
    "wealth_signal_label",
    "liquidity_event",
    "client_type",
    "source_of_wealth",
    "wealth_rank",
    "ai_wealth_signal",
    "ai_liquidity_label",
    "ai_client_who",
    "ai_why_money",
]

# -----------------------------------------------------------------------------
# Kaggle-style billionaire list (``datasets/billionaires.csv``)
# -----------------------------------------------------------------------------
_BILLIONAIRE_LOOKUP_CACHE: dict[str, dict[str, Any]] | None = None


def load_billionaire_data() -> pd.DataFrame:
    """Load the local billionaire CSV (Kaggle-style export). Returns empty frame if missing."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "billionaires.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="latin-1")
    except (OSError, UnicodeDecodeError):
        try:
            return pd.read_csv(path, encoding="utf-8", errors="replace")
        except OSError:
            return pd.DataFrame()


def clean_billionaire_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep latest year, usable ages, and drop estate / family rows."""
    if df is None or df.empty:
        return df
    df = df.copy()
    if "year" not in df.columns and "Year" in df.columns:
        df["year"] = pd.to_numeric(df["Year"], errors="coerce")
    if "year" in df.columns:
        ymax = df["year"].max()
        if pd.notna(ymax):
            df = df[df["year"] == ymax]
    if "personName" in df.columns:
        df["name"] = df["personName"].astype(str)
    elif "full_name" in df.columns:
        df["name"] = df["full_name"].astype(str)
    elif "Name" in df.columns:
        df["name"] = df["Name"].astype(str)
    elif "name" in df.columns:
        df["name"] = df["name"].astype(str)
    else:
        df["name"] = ""
    df["name"] = df["name"].str.replace(r"\s+", " ", regex=True).str.strip()
    acol = "age" if "age" in df.columns else ("Age" if "Age" in df.columns else None)
    if acol:
        df[acol] = pd.to_numeric(df[acol], errors="coerce")
        df = df[df[acol].notnull()]
        df = df[(df[acol] >= 18) & (df[acol] <= 85)]
    df = df[~df["name"].str.contains(r"Estate|Heirs|Family", case=False, na=False, regex=True)]
    df["name"] = df["name"].str.strip()
    nw = None
    for cand in ("finalWorth", "net_worth"):
        if cand in df.columns:
            nw = df[cand]
            break
    if nw is None:
        for c in df.columns:
            if "net" in str(c).lower() and "worth" in str(c).lower():
                nw = df[c]
                break
    df["finalWorth"] = nw if nw is not None else ""
    org = None
    for cand in ("organization", "source", "Source(s) of wealth"):
        if cand in df.columns:
            org = df[cand]
            break
    df["organization"] = org if org is not None else ""
    cat = None
    for cand in ("category", "Nationality"):
        if cand in df.columns:
            cat = df[cand]
            break
    df["category"] = cat if cat is not None else None
    return df


def build_billionaire_lookup(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if df is None or df.empty:
        return lookup
    for _, row in df.iterrows():
        name = str(row.get("name", "") or "").strip().lower()
        if not name:
            continue
        nw = row.get("finalWorth", row.get("net_worth", None))
        if pd.isna(nw):
            nw = None
        else:
            nw = str(nw).strip()
        oc = row.get("organization", row.get("source", None))
        if pd.isna(oc):
            oc = None
        else:
            oc = str(oc).strip()
        ic = row.get("category", None)
        if pd.isna(ic):
            ic = None
        else:
            ic = str(ic).strip()
        lookup[name] = {"net_worth": nw, "company": oc, "industry": ic}
    return lookup


def normalize_name(name: str | None) -> str:
    return str(name or "").lower().strip()


def match_billionaire(
    person_name: str | None,
    lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Resolve ``person_name`` to billionaire row data: exact normalized match, else last-name match.

    Returns the value dict ``{"net_worth", "company", "industry"}`` or None.
    """
    if not lookup:
        return None
    name = normalize_name(person_name)
    if not name:
        return None
    if name in lookup:
        return lookup[name]
    parts = name.split()
    if len(parts) >= 2:
        last = parts[-1]
        for k in sorted(lookup.keys()):
            if last in k:
                return lookup[k]
    return None


def get_billionaire_lookup() -> dict[str, dict[str, Any]]:
    """Load, clean, and cache a name → {net_worth, company, industry} map."""
    global _BILLIONAIRE_LOOKUP_CACHE
    if _BILLIONAIRE_LOOKUP_CACHE is None:
        try:
            bdf = clean_billionaire_data(load_billionaire_data())
            _BILLIONAIRE_LOOKUP_CACHE = build_billionaire_lookup(bdf)
        except Exception:
            _BILLIONAIRE_LOOKUP_CACHE = {}
    return _BILLIONAIRE_LOOKUP_CACHE


def _enrich_signals_with_billionaire_list(out: pd.DataFrame) -> None:
    """Match ``person_name`` to the billionaire table; set wealth fields and a small score bonus."""
    if out.empty:
        return
    lookup = get_billionaire_lookup()
    for idx in out.index:
        out.at[idx, "is_billionaire"] = False
        out.at[idx, "net_worth"] = ""
        out.at[idx, "billionaire_company"] = ""
        if not lookup:
            continue
        match = match_billionaire(str(out.at[idx, "person_name"] or ""), lookup)
        if match:
            out.at[idx, "is_billionaire"] = True
            nw = match.get("net_worth")
            out.at[idx, "net_worth"] = "" if nw is None else str(nw)
            bc = match.get("company")
            out.at[idx, "billionaire_company"] = "" if bc is None else str(bc)
            sc = int(out.at[idx, "score"])
            out.at[idx, "score"] = sc + 20
            print("Matched billionaire:", out.at[idx, "person_name"])
    out["is_billionaire"] = out["is_billionaire"].fillna(False).astype(bool)


def _target_client_is_high_value(tc: object) -> bool:
    """True for primary target clients (not ``\"mid\"``)."""
    if isinstance(tc, str):
        return False
    try:
        return bool(tc)
    except Exception:
        return False


def _apply_value_priority_tags(out: pd.DataFrame) -> None:
    """Derive ``priority`` from ``target_client`` (wealth signals) and billionaire list match."""
    if out.empty or "priority" not in out.columns:
        return
    for idx in out.index:
        tc = out.at[idx, "target_client"]
        ib = bool(out.at[idx, "is_billionaire"])
        parts: list[str] = []
        if _target_client_is_high_value(tc):
            parts.append("TARGET CLIENT")
        elif isinstance(tc, str) and tc.lower() == "mid":
            parts.append("MID TARGET")
        if ib:
            parts.append("BILLIONAIRE LIST")
        if "aggregated_estimated_wealth" in out.columns:
            try:
                agg = float(out.at[idx, "aggregated_estimated_wealth"] or 0)
            except (TypeError, ValueError):
                agg = 0.0
            if agg >= 10_000_000:
                parts.append("AGGREGATE $10M+")
        out.at[idx, "priority"] = " + ".join(parts) if parts else ""


# -----------------------------------------------------------------------------
# Public RSS feeds (business / tech news). Swap or extend as needed.
# -----------------------------------------------------------------------------
RSS_FEEDS = [
    # BBC
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "http://feeds.bbci.co.uk/news/technology/rss.xml",
    # Reuters
    "http://feeds.reuters.com/reuters/businessNews",
    # Financial Times
    "https://www.ft.com/rss/home",
    # Economist
    "https://www.economist.com/business/rss.xml",
    # Bloomberg
    "https://feeds.bloomberg.com/markets/news.rss",
]

# Browser-like User-Agent: some feeds block generic Python clients.
REQUEST_HEADERS = {
    "User-Agent": "WealthSignalsDashboard/1.0 (+https://example.local; educational project)",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

# Article pages (HTML) — many sites require browser-like Accept.
HTML_REQUEST_HEADERS = {
    "User-Agent": "WealthSignalsDashboard/1.0 (+https://example.local; educational project)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT_SEC = 20
# Cap stored combined paragraph text to keep regex/LLM inputs bounded.
MAX_ARTICLE_TEXT_CHARS = 200_000


def fetch_article_paragraph_text(url: str) -> str:
    """
    Fetch ``url`` and return all ``<p>`` paragraph text joined with spaces.

    On network/parse errors or non-HTML, returns ``""`` so callers can fall back to the RSS title.
    """
    if not url or not str(url).strip():
        return ""
    try:
        resp = requests.get(
            str(url).strip(),
            headers=HTML_REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT_SEC,
            allow_redirects=True,
        )
        resp.raise_for_status()
    except (requests.RequestException, OSError, ValueError):
        return ""

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    try:
        soup = BeautifulSoup(resp.content, "html.parser")
        paragraphs = soup.find_all("p")
        parts = [p.get_text(separator=" ", strip=True) for p in paragraphs]
        parts = [p for p in parts if p]
        full_text = " ".join(parts)
    except (AttributeError, TypeError, ValueError):
        return ""

    if len(full_text) > MAX_ARTICLE_TEXT_CHARS:
        full_text = full_text[:MAX_ARTICLE_TEXT_CHARS]
    return full_text


def _extraction_text(full_text: str, raw_title: str) -> str:
    """Prefer article body; fall back to RSS title when fetch failed or no paragraphs."""
    ft = (full_text or "").strip()
    if ft:
        return ft
    return (raw_title or "").strip()


class _OutreachCtx:
    """Normalized fields for outreach one-liners (deterministic templates)."""

    __slots__ = ("person", "company", "role", "raw_title", "headline", "event_type", "source_url")

    def __init__(
        self,
        person: str,
        company: str,
        role: str,
        raw_title: str,
        event_type: str,
        source_url: str,
    ) -> None:
        self.person = (person or "").strip()
        self.company = (company or "").strip()
        self.role = (role or "").strip()
        self.raw_title = (raw_title or "").strip()
        self.headline = _short_headline(self.raw_title)
        self.event_type = (event_type or "").strip()
        self.source_url = (source_url or "").strip()

    @property
    def has_person(self) -> bool:
        return bool(self.person)

    @property
    def has_company(self) -> bool:
        return bool(self.company) and self.company != "Unknown"

    @property
    def has_role(self) -> bool:
        return bool(self.role)

    @property
    def has_headline(self) -> bool:
        return bool(self.headline)


def _short_headline(raw: str, max_len: int = 90) -> str:
    t = " ".join((raw or "").split())
    if not t:
        return ""
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "..."


def _stable_template_index(n: int, *parts: str) -> int:
    """Pick a template slot from stable row content (same inputs -> same index across runs)."""
    if n <= 0:
        return 0
    digest = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % n


def _funding_stage_hint(title: str) -> str:
    """Lightweight, deterministic label from the headline (no ML)."""
    t = (title or "").lower()
    if "seed" in t:
        return "seed"
    if "series a" in t or "series-a" in t or "series a," in t:
        return "Series A"
    if "series b" in t or "series-b" in t:
        return "Series B"
    if "series c" in t or "series-c" in t:
        return "Series C"
    if "series d" in t or "series-d" in t:
        return "later-stage"
    return ""


# --- Per-event templates: 6 variants each; deterministic pick via _stable_template_index ---


def _fe_0(c: _OutreachCtx) -> str:
    if c.has_person and c.has_company and c.has_role:
        return (
            f"Offer {c.person} ({c.role} at {c.company}) a working session on liquidity timing, "
            f"QSBS and other basis questions, and how to diversify without forcing a product pitch."
        )
    if c.has_person and c.has_company:
        return (
            f"{c.person} at {c.company} — the months after an exit usually set tax and cash trajectories; "
            f"propose a dated checklist instead of a generic milestone message."
        )
    if c.has_company:
        return (
            f"Exit-related news at {c.company}: ask whether founders or executives want help sequencing proceeds, "
            f"estimates, and reinvestment—not a congratulatory opener."
        )
    return (
        "Liquidity event in the headline—keep the note practical: taxes, cash, equity, and timeline, "
        "without assuming details you do not have."
    )


def _fe_1(c: _OutreachCtx) -> str:
    if c.has_person and c.has_company and c.has_role:
        return (
            f"Ask {c.person} how their new scope as {c.role} at {c.company} intersects with earn-outs, "
            f"lockups, or secondary windows you can help map."
        )
    if c.has_person and c.has_company:
        return (
            f"Use {c.person} + {c.company} to anchor a note on post-exit cash and governance: "
            f"what they optimize for in the next two quarters."
        )
    if c.has_company:
        return f"If leadership at {c.company} is in transition, offer a concise read on proceeds, risk, and personal balance sheet—not hype."
    return "Founder or executive exit—lead with questions about liquidity mechanics and tax years, not celebration."


def _fe_2(c: _OutreachCtx) -> str:
    if c.has_headline and (c.has_person or c.has_company):
        subj = c.person if c.has_person else c.company
        return f'Use the story ("{c.headline}") to ask {subj} what decision points they are facing next on the exit—not generic praise.'
    if c.has_company:
        return f"Tie your note to {c.company}'s exit headline: offer perspective on diversification and tax sequencing for whoever owns the outcome."
    return "Exit context is often public before details are—stay neutral, ask what they are trying to solve this year."


def _fe_3(c: _OutreachCtx) -> str:
    if c.has_person and c.has_role:
        rest = f" at {c.company}" if c.has_company else ""
        return f"{c.person} moved to {c.role}{rest}; ask how the exit changes their personal risk budget and liquidity needs."
    if c.has_person:
        return f"{c.person} — after a liquidity event, people often want a second opinion on concentration and tax; offer a structured follow-up."
    if c.has_company:
        return f"Reference {c.company} and ask who owns the economic outcome—then tailor wealth planning to that person's facts."
    return "Liquidity event—avoid inventing deal terms; ask what they are weighing on tax and cash this quarter."


def _fe_4(c: _OutreachCtx) -> str:
    if c.has_person and c.has_company:
        return (
            f"Write to {c.person} at {c.company} about balancing reinvestment, charitable, and family goals after liquidity—skip the congratulatory tone."
        )
    if c.has_company:
        return f"Frame {c.company}'s exit as a planning window: run through scenarios, not slogans."
    return "Post-exit planning—name concrete deliverables (tax projection, diversification outline) if you reach out."


def _fe_5(c: _OutreachCtx) -> str:
    if c.has_person and c.has_company and c.has_role:
        return (
            f"{c.person}, {c.role} ({c.company}): anchor on how long they expect to be tied to the asset and what \"done\" looks like financially."
        )
    if c.has_company:
        return f"Weak name data but a clear company ({c.company})—stay high-level: liquidity, governance, and what changed for insiders."
    return "Sparse exit details—send a short, neutral note offering a planning lens without claiming inside knowledge."


def _fu_0(c: _OutreachCtx) -> str:
    stage = _funding_stage_hint(c.raw_title)
    if c.has_person and c.has_company and c.has_role:
        core = (
            f"Ask {c.person} ({c.role}, {c.company}) how the round resets equity benchmarks and "
            f"409A expectations for the team."
        )
        if stage:
            return f"{core} The headline reads like {stage} financing—open with a specific question, not a generic congrats on the raise."
        return f"{core} Open with a specific question, not a generic congrats on the raise."
    if c.has_person and c.has_company:
        return (
            f"{c.person} at {c.company} — new capital usually shifts dilution and hiring grants; "
            f"offer a tight question on what changed for insiders, not a celebration."
        )
    if c.has_company:
        return (
            f"Funding at {c.company}: lead with cap table, runway, and employee equity—not generic praise for the press release."
        )
    return "Funding headline—ask what moved for insiders on equity and dilution before offering advice."


def _fu_1(c: _OutreachCtx) -> str:
    if c.has_person and c.has_company and c.has_role:
        return (
            f"Connect with {c.person} on how {c.role} responsibilities at {c.company} interact with refresh grants and retention after the raise."
        )
    if c.has_person and c.has_company:
        return f"{c.person} ({c.company}) — ask how they are sizing secondary liquidity or employee pools post-round."
    if c.has_company:
        return f"Use {c.company} to discuss valuation step-ups and whether comp bands need a refresh after new money."
    return "Raise announced—stay specific: runway, dilution, and who got diluted, or stay neutral."


def _fu_2(c: _OutreachCtx) -> str:
    if c.has_headline:
        return (
            f'Start from the headline ("{c.headline}") and ask what milestone the financing unlocks next—product, geo, or team.'
        )
    if c.has_company:
        return f"{c.company}'s financing—ask what metric or deadline the board is optimizing for before pitching planning."
    return "Funding story without rich metadata—open with one precise question about their round, then listen."


def _fu_3(c: _OutreachCtx) -> str:
    if c.has_person and c.has_company:
        return (
            f"Offer {c.person} a concise compare on cash vs. equity emphasis at {c.company} after the new round—skip cheerleading."
        )
    if c.has_company:
        return f"Reference {c.company} and ask whether founders want help communicating equity changes to employees."
    return "New capital—neutral line: offer perspective on incentive design and tax timing if it fits their stage."


def _fu_4(c: _OutreachCtx) -> str:
    st = _funding_stage_hint(c.raw_title)
    if st and c.has_company:
        return f"{st} dynamics at {c.company}: worth asking how option pools and refresh timing were negotiated."
    if c.has_role and c.has_company:
        return f"As {c.role} at {c.company}, the contact may care about budget, headcount, and equity tradeoffs—meet them there."
    return "Funding item—if facts are thin, acknowledge the signal and ask what problem they are solving with the raise."


def _fu_5(c: _OutreachCtx) -> str:
    return (
        "Weak or noisy funding headline—keep the note short: confirm stage, round size if public, "
        "and whether they want a planning touchpoint or just information."
    )


def _pr_0(c: _OutreachCtx) -> str:
    if c.has_person and c.has_company and c.has_role:
        return (
            f"{c.person}'s move to {c.role} at {c.company} likely shifts bonus, equity, and visibility—"
            f"offer an executive-comp sanity check, not a congratulations template."
        )
    if c.has_person and c.has_company:
        return (
            f"{c.person} at {c.company} — promotions often reset deferrals and equity vesting; ask what changed in remit and pay mix."
        )
    if c.has_company:
        return f"Leadership change at {c.company}: ask how scope and compensation moved for the person in question."
    return "Senior promotion—anchor on remit, pay structure, and equity, not generic praise."


def _pr_1(c: _OutreachCtx) -> str:
    if c.has_person and c.has_role:
        co = f" at {c.company}" if c.has_company else ""
        return f"Ask {c.person} how the {c.role} remit{co} changes equity, cash, and time allocation they care about."
    if c.has_person and c.has_company:
        return f"{c.person} ({c.company}) — worth probing title change vs. material comp change before giving advice."
    if c.has_company:
        return f"Promotion context at {c.company}: stay factual; ask who moved and what decision rights shifted."
    return "Career step—use a neutral hook: scope, compensation, and what success looks like in the new role."


def _pr_2(c: _OutreachCtx) -> str:
    if c.has_headline:
        return (
            f'Reference the announcement ("{c.headline}") and ask what success metrics matter in the next 12 months—then tie planning to that.'
        )
    if c.has_person:
        return f"{c.person} — new level often triggers deferred comp and benefit elections; offer timing help without assuming numbers."
    return "Promotion signal with thin fields—keep the outreach humble and question-led."


def _pr_3(c: _OutreachCtx) -> str:
    if c.has_person and c.has_company and c.has_role:
        return (
            f"Discuss with {c.person} ({c.role}, {c.company}) whether public visibility or board exposure changes their personal risk profile."
        )
    if c.has_company:
        return f"At {c.company}, senior moves can affect clawbacks and good-leaver terms—ask before prescribing."
    return "Executive move—offer a second opinion on contracts and benefits if they engage; otherwise stay light."


def _pr_4(c: _OutreachCtx) -> str:
    if c.has_person and c.has_company:
        return (
            f"Write to {c.person} about balancing visibility and family liquidity given their role at {c.company}—avoid a canned congrats line."
        )
    if c.has_role:
        return f"New {c.role} title—ask what resources and constraints come with it before discussing wealth tactics."
    return "Promotion headline—neutral outreach: confirm the move, then ask what they want to optimize."


def _pr_5(c: _OutreachCtx) -> str:
    return (
        "Sparse promotion details—use a short note offering perspective on comp, equity, and tax elections without inventing titles."
    )


def _bd_0(c: _OutreachCtx) -> str:
    if c.has_person and c.has_company and c.has_role:
        return (
            f"Ask {c.person} ({c.role}, {c.company} board) how cash, equity, and committee workload compare to expectations—"
            f"directors care about clarity, not flattery."
        )
    if c.has_person and c.has_company:
        return (
            f"{c.person} at {c.company} — board roles bring D&O, time, and comp tradeoffs; open with governance questions."
        )
    if c.has_company:
        return f"Board news at {c.company}: verify independent vs. insider status before discussing fees or equity."
    return "Board appointment—lead with fiduciary duties and schedule, not congratulations."


def _bd_1(c: _OutreachCtx) -> str:
    if c.has_person and c.has_company:
        return (
            f"Offer {c.person} a concise read on conflicts, equity grants, and personal liability at {c.company}—skip generic prestige talk."
        )
    if c.has_company:
        return f"{c.company}'s board refresh—ask what committees and risk areas the new director will own."
    return "Director role—neutral: fees, equity, insurance, and time—only what applies once you know their facts."


def _bd_2(c: _OutreachCtx) -> str:
    if c.has_headline:
        return (
            f'Use ("{c.headline}") to ask whether the appointment is independent, observer, or executive-linked before tailoring advice.'
        )
    if c.has_role:
        return f"With {c.role} in the story, ask how board duties overlap with operating responsibilities—overlap drives planning."
    return "Board headline without details—stay curious about role type and time commitment."


def _bd_3(c: _OutreachCtx) -> str:
    if c.has_person:
        return (
            f"{c.person} — new boards often trigger personal concentration in one stock; offer diversification framing if appropriate."
        )
    if c.has_company:
        return f"Tie to {c.company}: director comp and equity may differ from management—clarify which hat they wear."
    return "Governance move—ask what they need from a professional (legal, tax, wealth) before pitching."


def _bd_4(c: _OutreachCtx) -> str:
    if c.has_person and c.has_company:
        return f"Reach out to {c.person} about schedule and reputation risk on {c.company}'s board—not generic 'thrilled for you' copy."
    return "Board-related item—keep tone professional; ask what they are trying to protect or optimize."


def _bd_5(c: _OutreachCtx) -> str:
    return (
        "Thin board story—confirm the role, public vs. private issuer, and geography before giving specific wealth guidance."
    )


def _ot_0(c: _OutreachCtx) -> str:
    if c.has_person and c.has_company:
        return (
            f"{c.person} at {c.company} — connect the headline to their professional context; ask what they are prioritizing this quarter."
        )
    if c.has_company:
        return f"Use {c.company} as the anchor and ask what changed operationally or financially—avoid guessing personal details."
    return "Broad finance or career item—open with curiosity about their goals, not a canned product pitch."


def _ot_1(c: _OutreachCtx) -> str:
    if c.has_headline:
        return (
            f'Start from ("{c.headline}") and ask one concrete question about relevance to their work—skip generic congratulations.'
        )
    if c.has_person:
        return f"{c.person} — weak structured data; keep the note short and invite them to share what matters."
    return "Loose match—acknowledge uncertainty and offer help only where your expertise clearly fits."


def _ot_2(c: _OutreachCtx) -> str:
    if c.has_company:
        return (
            f"Sector or company mention ({c.company})—tie planning to public facts; do not invent insider narrative."
        )
    return "Other bucket—stay neutral: offer a relevant lens (tax, liquidity, career) and let them correct you."


def _ot_3(c: _OutreachCtx) -> str:
    if c.has_person and c.has_role:
        co = f" ({c.company})" if c.has_company else ""
        return f"Ask {c.person}{co} how {c.role} intersects with the headline you saw—verify before advising."
    if c.has_person:
        return f"{c.person} — ask what problem they are solving; broad headlines rarely support deep specificity."
    return "Weak signal—one sentence of context, one question, no assumptions."


def _ot_4(c: _OutreachCtx) -> str:
    return (
        "Catch-all story—lead with why the item might matter to wealth or career planning, "
        "and invite them to steer the conversation."
    )


def _ot_5(c: _OutreachCtx) -> str:
    return (
        "Noisy classification—prefer a humble check-in: offer perspective if the story touches liquidity, tax, or governance they care about."
    )


_OUTREACH_TEMPLATES: dict[str, list[Callable[[_OutreachCtx], str]]] = {
    "Founder Exit": [_fe_0, _fe_1, _fe_2, _fe_3, _fe_4, _fe_5],
    "Funding": [_fu_0, _fu_1, _fu_2, _fu_3, _fu_4, _fu_5],
    "Promotion": [_pr_0, _pr_1, _pr_2, _pr_3, _pr_4, _pr_5],
    "Board Appointment": [_bd_0, _bd_1, _bd_2, _bd_3, _bd_4, _bd_5],
    "Other": [_ot_0, _ot_1, _ot_2, _ot_3, _ot_4, _ot_5],
}


def generate_outreach_angle(row) -> str:
    """
    One-line outreach angle: varies by event_type and by deterministic template slot.

    Same row content always picks the same template (stable md5 index). Uses person, company,
    role, and raw headline when present; falls back to neutral copy when data is thin.
    """
    et = (row.get("event_type") or "").strip()
    if et not in _OUTREACH_TEMPLATES:
        et = "Other"

    c = _OutreachCtx(
        str(row.get("person_name") or ""),
        str(row.get("company_name") or ""),
        str(row.get("role") or ""),
        str(row.get("raw_title") or ""),
        et,
        str(row.get("source_url") or ""),
    )
    fns = _OUTREACH_TEMPLATES[et]
    idx = _stable_template_index(
        len(fns),
        c.person,
        c.company,
        c.role,
        c.raw_title,
        c.event_type,
        c.source_url,
    )
    return fns[idx](c)


def priority_level_from_score(score: Any) -> str:
    """
    High / Medium / Low from numeric score (same bands as the product spec).
    """
    try:
        s = int(score)
    except (TypeError, ValueError):
        return "Low"
    if s >= 85:
        return "High"
    if s >= 70:
        return "Medium"
    return "Low"


def suggested_next_step_from_priority(priority_level: str) -> str:
    """Concrete follow-up tied to priority (drives the “what do I do?” feeling)."""
    p = (priority_level or "").strip()
    if p == "High":
        return "Reach out within 7 days"
    if p == "Medium":
        return "Add to watchlist"
    return "Monitor for future updates"


def compute_extraction_quality(row: pd.Series) -> int:
    """
    0-8 extraction strength after normalization (used for hero ranking and filtering).

    Weights person/company/role, prefers core event types over Other, and rewards
    substantive headlines.
    """
    q = 0
    if str(row.get("person_name", "")).strip():
        q += 2
    cn = str(row.get("company_name", "")).strip()
    if cn and cn != "Unknown":
        q += 2
    if str(row.get("role", "")).strip():
        q += 1
    if str(row.get("event_type", "")).strip() != "Other":
        q += 2
    if len(str(row.get("raw_title", "")).strip()) >= 28:
        q += 1
    ap_raw = row.get("additional_people")
    extra = False
    if isinstance(ap_raw, list):
        extra = len(ap_raw) > 0
    elif isinstance(ap_raw, str) and ap_raw.strip() not in ("", "[]"):
        try:
            parsed = json.loads(ap_raw)
            extra = isinstance(parsed, list) and len(parsed) > 0
        except (json.JSONDecodeError, TypeError):
            extra = True
    if extra:
        q += 1
    return min(8, q)


def compute_confidence_score(row: pd.Series) -> int:
    """
    0-100: blends extraction quality + signal bump with structured pattern confidence.

    Pattern score (funding / M&A / promotion / board regexes, minus negative hits)
    is merged with the legacy quality×11 formula, then clamped.
    """
    q = int(row.get("quality_score", 0) or 0)
    s = int(row.get("score", 0) or 0)
    bump = min(15, max(0, s - 45) // 3)
    legacy = int(min(100, q * 11 + bump))
    blob = f"{str(row.get('raw_title', '') or '')} {str(row.get('full_explanation', '') or '')}".strip()
    leg_et = str(row.get("event_type", "") or "").strip() or None
    _, pattern_raw = structured_pattern_confidence_and_type(blob, legacy_event_type=leg_et)
    merged = legacy + pattern_raw
    return int(max(0, min(100, merged)))


def why_it_matters_for_event_type(event_type: str) -> str:
    """
    Default one-line explanation for an event type.

    Call this when a row has no `why_it_matters` text yet, or it is blank.
    """
    blurbs = {
        "Founder Exit": "Exits and acquisitions often mean liquidity or major outcomes for founders.",
        "Funding": "New funding can reshape equity, hiring, and future payout potential.",
        "Promotion": "Senior moves usually reflect expanded scope and compensation upside.",
        "Board Appointment": "Board roles can bring fees, equity, and strategic influence.",
        "Other": "Only pursue if a clear money, liquidity, or concentration angle appears — not general headlines.",
    }
    et = (event_type or "").strip()
    return blurbs.get(et, "Public career and finance news may signal changing wealth dynamics.")


# Strip trailing headline / filler tokens so "at Acme Corp today" → "Acme Corp"
_COMPANY_TAIL_STOPWORDS = frozenset(
    {
        "today",
        "yesterday",
        "tomorrow",
        "tonight",
        "happens",
        "amid",
        "after",
        "before",
        "during",
        "when",
        "while",
        "said",
        "says",
        "reports",
        "according",
        "this",
        "that",
        "these",
        "those",
        "here",
        "there",
        "just",
        "only",
        "also",
        "still",
        "now",
        "new",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "will",
        "can",
        "may",
        "might",
        "could",
        "should",
        "would",
        "on",
        "in",
        "at",
        "for",
        "and",
        "or",
        "as",
        "to",
        "of",
        "with",
        "from",
        "by",
    }
)


def _trim_company_phrase_tail(phrase: str) -> str:
    """Drop trailing stopwords from a captured company phrase."""
    toks = phrase.split()
    while len(toks) > 1 and toks[-1].lower().rstrip(",.;:()") in _COMPANY_TAIL_STOPWORDS:
        toks.pop()
    return " ".join(toks).strip()


# After "at/from/of [Company]", headline often continues with one of these
_HEADLINE_STOP_AFTER_COMPANY = (
    r"(?:today|yesterday|tomorrow|tonight|happens|said|says|reports|according|amid|"
    r"after|before|during|while|when|where|for|and|as|to|on|in|with|from|the|a|an|is|are|"
    r"was|were|has|have|had|will|can|may|might|could|should|would|just|also|still|new|here|"
    r"there|this|that|these|those)\b"
)


def _company_from_title_at_from_of(title: str) -> str:
    """
    When person_name is missing, try company-like spans after at / from / of.
    """
    if not title or not str(title).strip():
        return ""
    t = str(title).strip()
    for kw in ("at", "from", "of"):
        m = re.search(
            rf"\s+{kw}\s+(.+?)(?=\s+{_HEADLINE_STOP_AFTER_COMPANY}|$|[,.\)\|;])",
            t,
            re.I,
        )
        if m:
            s = _trim_company_phrase_tail(m.group(1).strip())
            if s and len(s) <= 120 and re.match(r"^[A-Za-z0-9]", s):
                return s
    return ""


def _company_label_from_url(url: str) -> str:
    """
    Human-readable label from source URL hostname (e.g. techcrunch.com -> Techcrunch).
    """
    if not url or not str(url).strip():
        return ""
    raw = str(url).strip()
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    try:
        host = (urlparse(raw).hostname or "").lower()
    except ValueError:
        return ""
    if not host:
        return ""
    parts = [p for p in host.split(".") if p]
    if parts and parts[0] == "www":
        parts = parts[1:]
    if not parts:
        return ""
    common_tlds = {
        "com",
        "org",
        "net",
        "io",
        "co",
        "gov",
        "edu",
        "uk",
        "us",
        "au",
        "de",
        "fr",
        "eu",
        "ca",
        "in",
        "jp",
        "cn",
        "info",
        "biz",
    }
    if len(parts) >= 3 and parts[-2] == "co" and parts[-1] in ("uk", "au", "jp", "nz", "in"):
        label = parts[-3]
    elif len(parts) >= 2 and parts[-1] in common_tlds:
        label = parts[-2]
        if label == "co" and len(parts) >= 3:
            label = parts[-3]
    else:
        label = parts[0]
    label = label.replace("-", " ").strip()
    if not label:
        return ""
    return label[:1].upper() + label[1:].lower()


def _title_as_company_anchor(title: str, max_len: int = 56) -> str:
    """Last-resort company label from headline (trimmed, single line)."""
    t = re.sub(r"\s+", " ", str(title).strip())
    if not t:
        return ""
    if len(t) > max_len:
        t = t[: max_len - 1].rstrip() + "…"
    return t


def ensure_signal_anchor(out: pd.DataFrame) -> None:
    """
    Ensure each row has at least one meaningful anchor: person_name or company_name.

    When person_name is missing and company is empty or ``Unknown``, try (in order):
    ``at`` / ``from`` / ``of`` spans in ``raw_title``, then hostname from ``source_url``,
    then a shortened headline. Each candidate is validated; invalid names become ``Unknown``.
    """
    if out.empty:
        return
    pn = out["person_name"].fillna("").astype(str).str.strip()
    cn = out["company_name"].fillna("").astype(str).str.strip()
    weak = cn.eq("") | cn.str.lower().eq("unknown")
    need = pn.eq("") & weak
    if not need.any():
        return
    for idx in out.loc[need].index:
        t = str(out.at[idx, "raw_title"] or "")
        u = str(out.at[idx, "source_url"] or "")
        chosen = ""
        for cand in (_company_from_title_at_from_of(t), _company_label_from_url(u), _title_as_company_anchor(t) if t else ""):
            if not cand:
                continue
            sc = sanitize_company_name(cand)
            if sc:
                chosen = sc
                break
        out.at[idx, "company_name"] = chosen if chosen else "Unknown"


def dedupe_signals_cross_source(out: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate signals that appear across feeds/sources.

    - Non-empty ``person_name``: key = person_name + company_name + event_type
    - Empty ``person_name``: key = ``raw_title``; if title is empty, fall back to ``source_url``
      so unnamed rows without a title do not collapse into one bucket.

    Keeps the row with the strongest wealth signal (then recency, extraction clarity, score).
    Source outlet is not a ranking factor.
    """
    if out.empty:
        return out
    o = out.copy()
    pn = o["person_name"].fillna("").astype(str).str.strip()
    cn = o["company_name"].fillna("").astype(str).str.strip()
    et = o["event_type"].fillna("").astype(str).str.strip()
    rt = o["raw_title"].fillna("").astype(str).str.strip()
    su = o["source_url"].fillna("").astype(str).str.strip()

    named = pn != ""
    key = pd.Series(index=o.index, dtype=object)
    key.loc[named] = "n\x01" + pn[named] + "\x01" + cn[named] + "\x01" + et[named]
    no_name = ~named
    has_title = no_name & (rt != "")
    no_title = no_name & (rt == "")
    key.loc[has_title] = "t\x01" + rt[has_title]
    key.loc[no_title] = "u\x01" + su[no_title]

    o["_dedupe_key"] = key
    if "quality_score" not in o.columns:
        o["quality_score"] = 0
    o["quality_score"] = pd.to_numeric(o["quality_score"], errors="coerce").fillna(0).astype(int)
    if "wealth_rank" not in o.columns:
        o["wealth_rank"] = 3
    o["wealth_rank"] = pd.to_numeric(o["wealth_rank"], errors="coerce").fillna(3).astype(int)
    o["_person_id"] = (pn != "").astype(int)
    o = o.sort_values(
        by=["wealth_rank", "_person_id", "event_date", "quality_score", "score"],
        ascending=[True, False, False, False, False],
        na_position="last",
    )
    o = o.drop(columns=["_person_id"], errors="ignore")
    o = o.drop_duplicates(subset=["_dedupe_key"], keep="first")
    return o.drop(columns=["_dedupe_key"], errors="ignore").reset_index(drop=True)


# --- Mock “Crunchbase-style” profiles (extend or replace with a real API later) ---
_MOCK_COMPANY_PROFILES: dict[str, dict[str, str]] = {
    "openai": {
        "industry": "AI",
        "stage": "Series D",
        "description": "AI research and deployment company.",
        "location": "San Francisco, CA",
    },
    "stripe": {
        "industry": "Fintech",
        "stage": "Series H",
        "description": "Payments and financial infrastructure.",
        "location": "San Francisco, CA",
    },
    "moderna": {
        "industry": "Biotech",
        "stage": "Public",
        "description": "Biotechnology / mRNA medicines.",
        "location": "Cambridge, MA",
    },
}


def _normalize_company_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


# --- Company name validation (reject garbage, locations, generic words) ---
_COMPANY_EXACT_REJECT = frozenset(
    {
        "the",
        "a",
        "an",
        "least",
        "maybe",
        "news",
        "source",
        "news source",
    }
)

# Common English / headline words that are not company names (match whole string, case-insensitive).
_COMPANY_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "all",
        "also",
        "an",
        "and",
        "another",
        "any",
        "are",
        "around",
        "as",
        "ask",
        "at",
        "back",
        "be",
        "because",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "day",
        "did",
        "do",
        "does",
        "down",
        "each",
        "even",
        "ever",
        "every",
        "few",
        "first",
        "for",
        "four",
        "from",
        "get",
        "give",
        "go",
        "good",
        "had",
        "has",
        "have",
        "having",
        "her",
        "here",
        "him",
        "his",
        "how",
        "if",
        "into",
        "its",
        "just",
        "know",
        "last",
        "least",
        "like",
        "long",
        "made",
        "make",
        "many",
        "may",
        "maybe",
        "me",
        "might",
        "more",
        "most",
        "much",
        "must",
        "my",
        "never",
        "new",
        "next",
        "no",
        "not",
        "now",
        "off",
        "old",
        "on",
        "once",
        "one",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "own",
        "part",
        "people",
        "place",
        "put",
        "said",
        "same",
        "say",
        "says",
        "see",
        "several",
        "shall",
        "she",
        "should",
        "show",
        "side",
        "since",
        "so",
        "some",
        "still",
        "such",
        "take",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "thing",
        "this",
        "those",
        "three",
        "through",
        "too",
        "two",
        "under",
        "until",
        "up",
        "upon",
        "us",
        "use",
        "used",
        "very",
        "want",
        "was",
        "way",
        "we",
        "well",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "year",
        "you",
        "your",
    }
)

# Single-token place names (lowercase) — US states, countries, major cities; ambiguous tokens omitted.
_PLACE_SINGLE_WORD_LOWER = frozenset(
    {
        "afghanistan",
        "alabama",
        "alaska",
        "albania",
        "algeria",
        "anchorage",
        "andorra",
        "angola",
        "argentina",
        "arizona",
        "arkansas",
        "armenia",
        "australia",
        "austria",
        "azerbaijan",
        "bahrain",
        "bangladesh",
        "barcelona",
        "belarus",
        "belgium",
        "bentonville",
        "birmingham",
        "bolivia",
        "boston",
        "brazil",
        "brisbane",
        "bulgaria",
        "calcutta",
        "calgary",
        "california",
        "cambodia",
        "cameroon",
        "canada",
        "chicago",
        "chile",
        "china",
        "cincinnati",
        "cleveland",
        "colombia",
        "colorado",
        "columbus",
        "connecticut",
        "copenhagen",
        "costa",
        "croatia",
        "cuba",
        "cyprus",
        "czechia",
        "dallas",
        "delaware",
        "denmark",
        "denver",
        "detroit",
        "dublin",
        "ecuador",
        "edmonton",
        "egypt",
        "estonia",
        "ethiopia",
        "finland",
        "florida",
        "france",
        "frankfurt",
        "geneva",
        "georgia",
        "germany",
        "ghana",
        "glasgow",
        "greece",
        "greenville",
        "guatemala",
        "hamburg",
        "hawaii",
        "helsinki",
        "honduras",
        "houston",
        "hungary",
        "iceland",
        "illinois",
        "india",
        "indiana",
        "indianapolis",
        "indonesia",
        "iowa",
        "iran",
        "iraq",
        "ireland",
        "israel",
        "italy",
        "jacksonville",
        "jakarta",
        "japan",
        "jordan",
        "kansas",
        "kentucky",
        "kenya",
        "korea",
        "kuwait",
        "kyrgyzstan",
        "latvia",
        "lebanon",
        "lisbon",
        "lithuania",
        "london",
        "louisiana",
        "louisville",
        "luxembourg",
        "macau",
        "madrid",
        "maine",
        "malaysia",
        "malta",
        "manila",
        "maryland",
        "massachusetts",
        "melbourne",
        "memphis",
        "mexico",
        "miami",
        "michigan",
        "milwaukee",
        "minneapolis",
        "minnesota",
        "mississippi",
        "missouri",
        "moldova",
        "monaco",
        "mongolia",
        "montana",
        "montreal",
        "morocco",
        "moscow",
        "mumbai",
        "munich",
        "nashville",
        "nebraska",
        "nepal",
        "netherlands",
        "nevada",
        "newark",
        "nicaragua",
        "nigeria",
        "norway",
        "nottingham",
        "ohio",
        "oklahoma",
        "oman",
        "oregon",
        "orlando",
        "osaka",
        "ottawa",
        "pakistan",
        "panama",
        "paris",
        "pennsylvania",
        "perth",
        "peru",
        "philadelphia",
        "phoenix",
        "pittsburgh",
        "poland",
        "portland",
        "portugal",
        "prague",
        "qatar",
        "quebec",
        "raleigh",
        "richmond",
        "romania",
        "rome",
        "russia",
        "rwanda",
        "sacramento",
        "santiago",
        "saskatchewan",
        "saudi",
        "scotland",
        "seattle",
        "serbia",
        "shanghai",
        "singapore",
        "slovakia",
        "slovenia",
        "somalia",
        "spain",
        "stockholm",
        "sudan",
        "sweden",
        "switzerland",
        "sydney",
        "taiwan",
        "tampa",
        "tennessee",
        "texas",
        "thailand",
        "tokyo",
        "toronto",
        "tucson",
        "tulsa",
        "tunisia",
        "turkey",
        "turkmenistan",
        "uae",
        "uganda",
        "ukraine",
        "uruguay",
        "utah",
        "uzbekistan",
        "vancouver",
        "venezuela",
        "vienna",
        "vietnam",
        "virginia",
        "warsaw",
        "wisconsin",
        "yemen",
        "zambia",
        "zimbabwe",
        "zurich",
    }
)

# Multi-word locations (normalized: single spaces, lower case).
_PLACE_PHRASE_LOWER = frozenset(
    {
        "costa rica",
        "czech republic",
        "district of columbia",
        "el salvador",
        "hong kong",
        "los angeles",
        "new hampshire",
        "new jersey",
        "new mexico",
        "new orleans",
        "new york",
        "new zealand",
        "north carolina",
        "north dakota",
        "north korea",
        "puerto rico",
        "san antonio",
        "san diego",
        "san francisco",
        "san jose",
        "saudi arabia",
        "south africa",
        "south carolina",
        "south dakota",
        "south korea",
        "sri lanka",
        "united arab emirates",
        "united kingdom",
        "united states",
        "west virginia",
    }
)

# Single-word lowercase tokens that are valid company names (skip the all-lowercase rejection).
_LOWERCASE_COMPANY_ALLOWLIST_LOWER = frozenset(
    {
        "amazon",
        "apple",
        "microsoft",
        "oracle",
        "dell",
        "cisco",
        "adobe",
        "salesforce",
        "palantir",
        "stripe",
        "square",
        "visa",
        "mastercard",
        "intel",
        "nvidia",
        "samsung",
        "sony",
        "toyota",
        "honda",
        "nissan",
        "uber",
        "lyft",
        "spotify",
        "netflix",
        "twitter",
        "linkedin",
        "facebook",
        "google",
        "yahoo",
        "oracle",
    }
)

# News publishers / outlets — not operating companies for ``company_name``.
MEDIA_OUTLETS = frozenset(
    {
        "nytimes",
        "new york times",
        "forbes",
        "bloomberg",
        "reuters",
        "cnn",
        "bbc",
        "msnbc",
        "nbc",
        "guardian",
        "wsj",
        "financial times",
    }
)

# Article noise tokens sometimes mistaken for a company name.
_COMPANY_GENERIC_WORD_REJECT = frozenset({"the", "a", "an", "this", "that"})


def _normalize_company_place_key(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").lower().strip())


def sanitize_company_name(name: str) -> str:
    """
    Return a cleaned company name, or "" if the value is not a plausible company
    (too short, stopword, place name, etc.).
    """
    raw = (name or "").strip()
    if not raw:
        return ""
    low = _normalize_company_place_key(raw)
    if raw.lower() == "unknown":
        return ""
    if low in MEDIA_OUTLETS:
        return ""
    if low in _COMPANY_GENERIC_WORD_REJECT:
        return ""
    if len(raw) < 3:
        return ""
    if low in _COMPANY_EXACT_REJECT:
        return ""
    if low in _COMPANY_STOPWORDS:
        return ""
    # Single token, all ASCII letters lowercase — usually noise; allow known brands.
    if re.fullmatch(r"[a-z]+", raw.strip()) and low not in _LOWERCASE_COMPANY_ALLOWLIST_LOWER:
        return ""
    if low in _PLACE_PHRASE_LOWER:
        return ""
    if low in _PLACE_SINGLE_WORD_LOWER and low not in _LOWERCASE_COMPANY_ALLOWLIST_LOWER:
        return ""
    return raw


def normalize_company_name_field(name: str) -> str:
    """Public normalization for storage: valid company or ``Unknown``."""
    s = sanitize_company_name(name)
    return s if s else "Unknown"


def _company_qualifies_for_score_boost(company_name: str) -> bool:
    """True when company is known-valid (not garbage/Unknown) — used for enrichment/funding bonuses."""
    return bool(sanitize_company_name(company_name or ""))


def _extract_stage_and_industry_from_text(text: str) -> tuple[str, str]:
    """Best-effort stage / industry from article body when mock data misses."""
    t = (text or "").lower()
    stage = ""
    if re.search(r"\bseries\s+d\b", t):
        stage = "Series D"
    elif re.search(r"\bseries\s+c\b", t):
        stage = "Series C"
    elif re.search(r"\bseries\s+b\b", t):
        stage = "Series B"
    elif re.search(r"\bseries\s+a\b", t):
        stage = "Series A"
    elif "seed round" in t or re.search(r"\bseed\s+funding\b", t):
        stage = "Seed"

    industry = ""
    if re.search(r"\b(ai|artificial intelligence|machine learning)\b", t):
        industry = "AI"
    elif "fintech" in t or "financial technology" in t:
        industry = "Fintech"
    elif "biotech" in t or "biopharma" in t or "life sciences" in t:
        industry = "Biotech"
    return stage, industry


def enrich_company_data(company_name: str, article_text: str = "") -> dict[str, str]:
    """
    Enrich a company with industry, funding stage, description, and optional location.

    Tries a small mock lookup (Crunchbase-style), then fills gaps from ``article_text``.
    Never raises; returns empty strings when nothing is found.
    """
    out: dict[str, str] = {"industry": "", "stage": "", "description": "", "location": ""}
    try:
        cn = (company_name or "").strip()
        if not cn or cn.lower() == "unknown" or not sanitize_company_name(cn):
            return out

        norm = _normalize_company_key(cn)
        for key, profile in _MOCK_COMPANY_PROFILES.items():
            kn = _normalize_company_key(key)
            if norm == kn or kn in norm or norm in kn:
                for k in out:
                    if profile.get(k):
                        out[k] = str(profile[k]).strip()
                break

        st, ind = _extract_stage_and_industry_from_text(article_text)
        if not out["stage"] and st:
            out["stage"] = st
        if not out["industry"] and ind:
            out["industry"] = ind

        if not out["description"] and article_text:
            snippet = " ".join(article_text.split())
            if len(snippet) > 280:
                snippet = snippet[:279].rstrip() + "…"
            if snippet:
                out["description"] = snippet

        return out
    except Exception:
        return {"industry": "", "stage": "", "description": "", "location": ""}


def _enrichment_score_bonus(stage: str, industry: str, company_name: str = "") -> int:
    """Extra score points from structured company enrichment; combined with base score (clip 0–100)."""
    if not _company_qualifies_for_score_boost(company_name):
        return 0
    bonus = 0
    st = (stage or "").strip()
    if any(x in st for x in ("Series B", "Series C", "Series D")):
        bonus += 20
    ind = (industry or "").strip()
    if any(seg in ind for seg in ("AI", "Biotech", "Fintech")):
        bonus += 15
    return bonus


# --- Funding / deal extraction from article text (dollar amounts + round labels) ---
# Baseline pattern \$[0-9]+(M|B)? plus commas/decimals and optional K.
_FUNDING_DOLLAR_RE = re.compile(r"\$[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?\s*[KkMmBb]?", re.I)
_FUNDING_DOLLAR_COMPACT = re.compile(r"\$[0-9]+[MmBb]?", re.I)


def _parse_single_funding_token_to_usd(token: str) -> float | None:
    """Parse a single amount like ``$120M``, ``$1.5B``, ``$500K``, or plain ``$50000000`` USD."""
    s = (token or "").strip().replace(",", "")
    m = re.match(r"^\$\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?\s*$", s)
    if not m:
        return None
    try:
        n = float(m.group(1))
    except ValueError:
        return None
    suf = (m.group(2) or "").upper()
    mult = 1.0
    if suf == "K":
        mult = 1_000
    elif suf == "M":
        mult = 1_000_000
    elif suf == "B":
        mult = 1_000_000_000
    return n * mult


def _extract_largest_funding_amount_string(text: str) -> str:
    """
    Find dollar amounts matching ``\\$[0-9]+...`` with optional ``M``/``B``/``K`` suffixes.
    Return the token with the largest USD value.
    """
    if not (text or "").strip():
        return ""
    best_val = -1.0
    best_raw = ""
    for pat in (_FUNDING_DOLLAR_RE, _FUNDING_DOLLAR_COMPACT):
        for m in pat.finditer(text):
            raw = m.group(0).strip()
            v = _parse_single_funding_token_to_usd(raw)
            if v is not None and v > best_val:
                best_val = v
                best_raw = raw
    return best_raw


def _extract_funding_stage_from_article(text: str) -> str:
    """Seed / Series A–C only (explicit round phrasing)."""
    t = text or ""
    if re.search(r"\bseries\s+c\b", t, re.I):
        return "Series C"
    if re.search(r"\bseries\s+b\b", t, re.I):
        return "Series B"
    if re.search(r"\bseries\s+a\b", t, re.I):
        return "Series A"
    if re.search(r"\bseed\b", t, re.I):
        return "Seed"
    return ""


def extract_funding_fields_from_text(text: str) -> dict[str, str]:
    """
    Extract ``funding_amount`` (largest ``$…`` token in text) and ``funding_stage`` (Seed / Series A–C).

    Safe on empty input; never raises.
    """
    try:
        blob = (text or "").strip()
        return {
            "funding_amount": _extract_largest_funding_amount_string(blob),
            "funding_stage": _extract_funding_stage_from_article(blob),
        }
    except Exception:
        return {"funding_amount": "", "funding_stage": ""}


def _funding_deal_score_bonus(funding_amount_str: str) -> int:
    """+30 for $100M+ deals, +20 for $10M+ (from parsed article amount)."""
    v = _parse_single_funding_token_to_usd((funding_amount_str or "").strip())
    if v is None:
        return 0
    if v >= 100_000_000:
        return 30
    if v >= 10_000_000:
        return 20
    return 0


# Deal size → individual wealth (rough ownership × headline $ amount)
_DEAL_VALUE_RE = re.compile(
    r"\$([0-9]+(?:\.[0-9]+)?)\s*(M|B|million|billion)\b",
    re.I,
)


def _usd_from_deal_match(m: re.Match[str]) -> float:
    n = float(m.group(1))
    u = (m.group(2) or "").lower()
    if u in ("m", "million"):
        return n * 1_000_000
    if u in ("b", "billion"):
        return n * 1_000_000_000
    return 0.0


def _max_deal_value_in_text(text: str) -> float:
    best = 0.0
    for m in _DEAL_VALUE_RE.finditer(text or ""):
        v = _usd_from_deal_match(m)
        if v > best:
            best = v
    return best


def extract_deal_value_usd(raw_title: str, full_explanation: str) -> float:
    """
    Largest ``$…M/B/million/billion`` from title + body.

    When the word ``valuation`` appears, prefer the largest amount in a window around it
    (company valuation as proxy for deal scale).
    """
    blob = f"{raw_title} {full_explanation}"
    blob_l = blob.lower()
    if "valuation" in blob_l:
        best = 0.0
        start = 0
        while True:
            i = blob_l.find("valuation", start)
            if i == -1:
                break
            win = blob[max(0, i - 120) : min(len(blob), i + 120)]
            v = _max_deal_value_in_text(win)
            if v > best:
                best = v
            start = i + 1
        if best > 0:
            return best
    return _max_deal_value_in_text(blob)


def _ownership_fraction_from_role(role: str) -> float:
    r = (role or "").lower()
    if re.search(r"co[- ]?founder", r):
        return 0.10
    if re.search(r"\bfounder\b", r):
        return 0.15
    if re.search(r"\bceo\b", r):
        return 0.05
    return 0.02


def estimate_wealth_from_deal(
    raw_title: str,
    full_explanation: str,
    role: str,
) -> tuple[float, float]:
    """``(estimated_wealth_usd, deal_value_usd)`` from deal size × implied ownership."""
    deal = extract_deal_value_usd(raw_title, full_explanation)
    own = _ownership_fraction_from_role(role)
    return (deal * own, deal)


def merge_target_client_row(
    wealth_score: int,
    estimated_wealth: float,
    aggregated_estimated_wealth: float = 0.0,
) -> bool | str:
    """
    Combine rule-based ``wealth_score`` with deal-based ``estimated_wealth``.

    When the same person has multiple rows, ``aggregated_estimated_wealth`` is the sum of
    per-row estimates (e.g. exit + separate funding) — ``>= $10M`` → primary target.

    Deal path: ``>= $5M`` personal estimate → primary target; ``$1M–$5M`` → ``\"mid\"``;
    ``wealth_score >= 50`` still marks a primary target.
    """
    agg = float(aggregated_estimated_wealth)
    if agg >= 10_000_000:
        return True
    w = int(wealth_score) >= 50
    ew = float(estimated_wealth)
    if ew >= 5_000_000:
        return True
    if w:
        return True
    if ew >= 1_000_000:
        return "mid"
    return False


def format_wealth(value) -> str:
    """Format estimated USD for display; None / NaN / zero → ``Data pending``."""
    if value is None:
        return "Data pending"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "Data pending"
    if pd.isna(v) or v == 0:
        return "Data pending"
    if v >= 1_000_000_000:
        return f"${v / 1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    return f"${v:,.0f}"


def normalize_name(name: str) -> str:
    """Stable key for cross-article person matching (lowercase, collapsed whitespace)."""
    s = (name or "").strip().lower()
    if not s:
        return ""
    return re.sub(r"\s+", " ", s)


def infer_wealth(row: pd.Series) -> float:
    """Coarse headline-based USD hint when deal parsing yields little or nothing."""
    title = str(row.get("raw_title", "") or "").lower()

    if "sold" in title or "acquired" in title:
        return 5_000_000.0
    if "raised" in title:
        return 1_000_000.0
    if "ceo" in title or "chief" in title:
        return 2_000_000.0
    return 0.0


KNOWN_BILLIONAIRES: dict[str, float] = {
    "larry ellison": 225_000_000_000,
    "elon musk": 700_000_000_000,
    "jeff bezos": 250_000_000_000,
    "mark zuckerberg": 220_000_000_000,
}


def enrich_known_wealth(row: pd.Series) -> float:
    """Override deal-derived estimate when ``person_name`` matches a hand-curated net worth."""
    name = normalize_name(str(row.get("person_name", "") or ""))
    if name in KNOWN_BILLIONAIRES:
        return float(KNOWN_BILLIONAIRES[name])
    try:
        return float(row.get("estimated_wealth", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def assign_aggregated_estimated_wealth(out: pd.DataFrame) -> None:
    """Set ``aggregated_estimated_wealth`` to the sum of ``estimated_wealth`` per normalized person."""
    if out.empty:
        return
    out["aggregated_estimated_wealth"] = 0.0
    person_index: dict[str, list[Any]] = {}
    for idx in out.index:
        nm = normalize_name(str(out.at[idx, "person_name"] or ""))
        if not nm:
            continue
        person_index.setdefault(nm, []).append(idx)
    for _nm, idxs in person_index.items():
        total = sum(float(out.at[i, "estimated_wealth"] or 0) for i in idxs)
        for idx in idxs:
            out.at[idx, "aggregated_estimated_wealth"] = total


def _row_has_funding_signal(row: pd.Series) -> bool:
    if str(row.get("event_type", "") or "").strip() == "Funding":
        return True
    fa = str(row.get("funding_amount", "") or "").strip()
    v = _parse_single_funding_token_to_usd(fa)
    if v is not None and v > 0:
        return True
    if str(row.get("funding_stage", "") or "").strip():
        return True
    return False


def _row_has_acquisition_signal(row: pd.Series) -> bool:
    if str(row.get("event_type", "") or "").strip() == "Founder Exit":
        return True
    blob = f"{row.get('raw_title', '')} {row.get('full_explanation', '')}".lower()
    return bool(re.search(r"\b(acquisition|acquired|acquires|buyout|merger)\b", blob))


def _row_has_deal_value_signal(row: pd.Series) -> bool:
    try:
        ew = float(row.get("estimated_wealth") or 0)
    except (TypeError, ValueError):
        ew = 0.0
    if ew > 0:
        return True
    dv = extract_deal_value_usd(
        str(row.get("raw_title", "")),
        str(row.get("full_explanation", "")),
    )
    return dv > 0


def _row_has_linked_wealth_signal(row: pd.Series) -> bool:
    """Funding, M&A-style acquisition, or parsed headline deal value."""
    return (
        _row_has_funding_signal(row)
        or _row_has_acquisition_signal(row)
        or _row_has_deal_value_signal(row)
    )


def apply_cross_article_enrichment(out: pd.DataFrame) -> None:
    """
    Index rows by ``normalize_name(person_name)``; boost score when the same person
    appears in multiple signals or when any row carries funding / acquisition / deal value.
    """
    if out.empty:
        return
    out["repeat_person"] = False
    out["linked_wealth_signal"] = False

    person_index: dict[str, list[Any]] = {}
    for idx in out.index:
        nm = normalize_name(str(out.at[idx, "person_name"] or ""))
        if not nm:
            continue
        person_index.setdefault(nm, []).append(idx)

    for _name, idxs in person_index.items():
        if len(idxs) > 1:
            for idx in idxs:
                out.at[idx, "repeat_person"] = True
                out.at[idx, "score"] = int(out.at[idx, "score"]) + 10

        any_link = any(_row_has_linked_wealth_signal(out.loc[i]) for i in idxs)
        if any_link:
            for idx in idxs:
                out.at[idx, "linked_wealth_signal"] = True
                out.at[idx, "score"] = int(out.at[idx, "score"]) + 20


def _company_index_key(company_name: str) -> str:
    """Stable key for cross-row company matching; empty if unknown or invalid."""
    cn = str(company_name or "").strip()
    if not cn or cn.lower() == "unknown":
        return ""
    if not sanitize_company_name(cn):
        return ""
    return _normalize_company_key(cn)


def _peer_funding_amount_usd(funding_amount_str: str) -> float:
    v = _parse_single_funding_token_to_usd((funding_amount_str or "").strip())
    return v if v is not None else 0.0


def propagate_company_funding_to_people(out: pd.DataFrame) -> None:
    """
    If funding is detected for a company on any row, copy the strongest peer
    ``funding_amount`` / ``funding_stage`` onto sibling rows for that company so
    person-level scores and wealth estimates reflect the shared deal.
    """
    if out.empty:
        return
    company_index: dict[str, list[Any]] = {}
    for idx in out.index:
        key = _company_index_key(str(out.at[idx, "company_name"] or ""))
        if not key:
            continue
        company_index.setdefault(key, []).append(idx)

    for _key, idxs in company_index.items():
        donors = [i for i in idxs if _row_has_funding_signal(out.loc[i])]
        if not donors:
            continue
        best = max(donors, key=lambda i: _peer_funding_amount_usd(str(out.at[i, "funding_amount"])))
        best_amt = str(out.at[best, "funding_amount"] or "").strip()
        best_stage = str(out.at[best, "funding_stage"] or "").strip()

        for idx in idxs:
            if _row_has_funding_signal(out.loc[idx]):
                continue
            if best_amt:
                out.at[idx, "funding_amount"] = best_amt
            if best_stage and not str(out.at[idx, "funding_stage"] or "").strip():
                out.at[idx, "funding_stage"] = best_stage


def apply_cross_company_enrichment(out: pd.DataFrame) -> None:
    """When the same company appears on multiple rows, treat as higher importance (+10 score each)."""
    if out.empty:
        return
    out["repeat_company"] = False
    company_index: dict[str, list[Any]] = {}
    for idx in out.index:
        key = _company_index_key(str(out.at[idx, "company_name"] or ""))
        if not key:
            continue
        company_index.setdefault(key, []).append(idx)

    for _key, idxs in company_index.items():
        if len(idxs) <= 1:
            continue
        for idx in idxs:
            out.at[idx, "repeat_company"] = True
            out.at[idx, "score"] = int(out.at[idx, "score"]) + 10


def compute_wealth_score(
    raw_title: str,
    full_explanation: str,
    funding_amount: str,
    event_type: str,
    role: str,
) -> int:
    """
    Internal wealth-signal score (not the same as ``score`` / outreach rank).

    Used to flag FA-relevant leads (~$10M+ potential) via ``target_client``.
    """
    ws = 0
    blob = f"{raw_title} {full_explanation}".lower()
    r = (role or "").lower()
    et = (event_type or "").strip()

    fa = (funding_amount or "").strip()
    v_col = _parse_single_funding_token_to_usd(fa)
    ext_tok = _extract_largest_funding_amount_string(f"{raw_title} {full_explanation}")
    v_ext = _parse_single_funding_token_to_usd(ext_tok) if ext_tok else None
    vals = [x for x in (v_col, v_ext) if x is not None and x > 0]
    v = max(vals) if vals else 0.0

    if v >= 100_000_000:
        ws += 60
    elif v >= 10_000_000:
        ws += 40
    elif v >= 1_000_000:
        ws += 20

    if et == "Founder Exit":
        ws += 50
    elif re.search(r"\b(acquisition|acquired|acquires|buyout|merger)\b", blob):
        ws += 50

    if re.search(r"\bceo\b", r):
        ws += 30
    elif re.search(r"\b(cfo|coo)\b", r):
        ws += 20

    if re.search(r"co[- ]?founder|\bfounder\b", r):
        ws += 40

    has_phrase = (
        "sold for" in blob
        or re.search(r"\bexit\b", blob) is not None
        or "acquired for" in blob
        or "valuation" in blob
        or "stake worth" in blob
        or re.search(r"\bequity\b", blob) is not None
    )
    if has_phrase:
        ws += 40

    return ws


# --- Drop only clearly off-topic stories (ingest + finalize) ---
_CLEARLY_IRRELEVANT_INGEST_SUBSTRINGS = (
    "weather",
    "sports",
    "northern lights",
    "celebrity gossip",
)

# Strict relevance: at least one of (A) financial (B) career (C) transaction — checked before extraction.
# A: $, million|billion, funding|raised, valuation
# B: appointed|named|joins|CEO|CFO|CTO (also appoints|names: common headline verb forms)
# C: acquired|sold|deal|merger
_STRICT_FINANCIAL_RE = re.compile(
    r"\$|\b(?:million|billion)\b|\b(?:funding|raised)\b|\bvaluation\b",
    re.I,
)
_STRICT_CAREER_RE = re.compile(
    r"\b(?:appointed|appoints|named|names|joins)\b|\b(?:CEO|CFO|CTO)\b",
    re.I,
)
_STRICT_TRANSACTION_RE = re.compile(
    r"\b(?:acquired|sold|deal|merger)\b",
    re.I,
)


def _contains_clearly_irrelevant_topics(text: str) -> bool:
    """True when the blob is clearly not business/finance (hard drop)."""
    t = (text or "").lower()
    return any(s in t for s in _CLEARLY_IRRELEVANT_INGEST_SUBSTRINGS)


def _strict_relevance_patterns_match(text: str) -> bool:
    """At least one financial, career, or transaction signal (see product spec)."""
    t = text or ""
    if not t.strip():
        return False
    return bool(
        _STRICT_FINANCIAL_RE.search(t)
        or _STRICT_CAREER_RE.search(t)
        or _STRICT_TRANSACTION_RE.search(t)
    )


_SOFT_ACTION_RE = re.compile(
    r"\b(?:"
    r"funding|funded|raises?|raised|financing|seed|venture|round|"
    r"invest(?:ment|or)?|ipo|unicorn|valuation|series\s+[a-e]|"
    r"appoints?|appointed|joins?|joining|hired|hiring|named|promoted|promotion|"
    r"ceo|cfo|cto|coo|chief|executive|board|director|leadership|founder|co-founder|"
    r"acquisition|acquired|acquires|merger|layoff|resigns|succession|partner"
    r")\b",
    re.I,
)

# Classify-style keywords not fully covered by _SOFT_ACTION_RE (same sentence as actor).
_EXTRA_EVENT_KEYWORD_RE = re.compile(
    r"\b(?:buyout|takeover|sold|exit|divests?|buys)\b"
    r"|boardroom"
    r"|stepping\s+down"
    r"|new\s+role"
    r"|executive\s+shuffle",
    re.I,
)


def _split_into_sentences(text: str) -> list[str]:
    """Split on sentence boundaries; preserves single block when no delimiter."""
    blob = (text or "").strip()
    if not blob:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", blob)
    out = [p.strip() for p in parts if p.strip()]
    return out if out else [blob]


def _sentence_has_event_keyword(sentence: str) -> bool:
    """True if this sentence contains an event-type keyword (classify / career / soft action)."""
    s = (sentence or "").strip()
    if not s:
        return False
    if _SOFT_ACTION_RE.search(s):
        return True
    if _EXTRA_EVENT_KEYWORD_RE.search(s):
        return True
    return _finance_career_broad(s)


def _person_name_in_sentence(sentence_lower: str, person_name: str) -> bool:
    """True if the sentence references the person (full name or distinctive token)."""
    p = (person_name or "").strip().lower()
    if len(p) < 2:
        return False
    if p in sentence_lower:
        return True
    parts = [re.sub(r"[^a-z0-9]", "", x) for x in p.split() if x]
    for w in parts:
        if len(w) >= 2 and w in sentence_lower:
            return True
    return False


def _company_name_in_sentence(sentence_lower: str, company_name: str) -> bool:
    """True if the sentence references the company (full name or major token, e.g. Helio from Helio Robotics)."""
    c = (company_name or "").strip().lower()
    if not c or c == "unknown":
        return False
    if c in sentence_lower:
        return True
    for w in c.split():
        w = re.sub(r"[^a-z0-9]", "", w)
        if len(w) >= 2 and w in sentence_lower:
            return True
    return False


def _sentence_has_actor(sentence: str, person_name: str, company_name: str) -> bool:
    """True if sentence mentions the extracted person and/or a valid company name."""
    sl = (sentence or "").lower()
    pn = (person_name or "").strip()
    cn = (company_name or "").strip()
    cn_ok = bool(cn) and cn.lower() != "unknown"
    pn_ok = len(pn) >= 2
    if not pn_ok and not cn_ok:
        return False
    if pn_ok and _person_name_in_sentence(sl, pn):
        return True
    if cn_ok and _company_name_in_sentence(sl, cn):
        return True
    return False


def _opportunity_sentence_gate(text_blob: str, person_name: str, company_name: str) -> bool:
    """
    Keep only if at least one sentence contains both an event keyword and (person OR company).

    Otherwise the row is not treated as an actionable opportunity.
    """
    sents = _split_into_sentences(text_blob or "")
    if not sents and (text_blob or "").strip():
        sents = [(text_blob or "").strip()]
    for sent in sents:
        if _sentence_has_event_keyword(sent) and _sentence_has_actor(sent, person_name, company_name):
            return True
    return False


# --- Structured financial / career pattern detection (regex + sentence structure) ---
_RE_STRUCT_FUNDING_DOLLAR = re.compile(
    r"\$[0-9]+(?:\.[0-9]+)?\s?(?:M|B|million|billion)\b",
    re.I,
)
_RE_STRUCT_FUNDING_SERIES = re.compile(r"Series\s+[A-C]\b", re.I)
_RE_STRUCT_VALUATION = re.compile(r"\bvaluation\b", re.I)

_RE_STRUCT_PROMOTION_1 = re.compile(
    r"[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:joins|joined|appointed|named|takes\s+over)\s+.*?(?:CEO|CFO|CTO)\b",
    re.I | re.DOTALL,
)
_RE_STRUCT_PROMOTION_2 = re.compile(
    r"(?:CEO|CFO|CTO)\s+of\s+[A-Z][a-zA-Z]+",
    re.I,
)

_RE_STRUCT_MA_VERB = re.compile(
    r"\b(?:acquire|acquires|acquired|buy|buys|bought)\b",
    re.I,
)
_RE_STRUCT_MA_DEAL = re.compile(r"deal\s+(?:worth|valued)\s+\$", re.I)

_RE_STRUCT_BOARD_ROLE = re.compile(r"\b(?:board|director|chairman)\b", re.I)
_RE_STRUCT_BOARD_VERB = re.compile(r"\b(?:joins|appointed|named)\b", re.I)

_STRUCTURED_NEGATIVE_SUBSTRINGS = (
    "cyberattack",
    "breach",
    "war",
    "lawsuit",
    "trial",
    "killed",
    "investigation",
)


def structured_pattern_confidence_and_type(
    text: str,
    legacy_event_type: str | None = None,
) -> tuple[str | None, int]:
    """
    Score structured financial/career signals from patterns (not keywords alone).

    Returns (event_type or None, raw confidence). Drop rows when raw confidence < 20.

    When no structured bucket matches but ``legacy_event_type`` is set (e.g. headline
    classifier), adds +25 so keyword-classified items can still pass the gate.
    """
    raw = (text or "").strip()
    if not raw:
        return None, 0

    conf = 0

    negative_hit = False
    for s in _split_into_sentences(raw) + [raw]:
        sl = s.lower()
        if any(bad in sl for bad in _STRUCTURED_NEGATIVE_SUBSTRINGS):
            negative_hit = True
            break

    if negative_hit:
        # Reject structured bonuses; only legacy headline path can partially offset.
        leg = (legacy_event_type or "").strip()
        conf = -50 + (25 if leg else 0)
        return (leg or None, conf)

    funding = bool(
        _RE_STRUCT_FUNDING_DOLLAR.search(raw)
        or _RE_STRUCT_FUNDING_SERIES.search(raw)
        or _RE_STRUCT_VALUATION.search(raw)
    )
    promotion = bool(_RE_STRUCT_PROMOTION_1.search(raw) or _RE_STRUCT_PROMOTION_2.search(raw))
    ma = bool(_RE_STRUCT_MA_VERB.search(raw) or _RE_STRUCT_MA_DEAL.search(raw))
    board = False
    for sent in _split_into_sentences(raw) or [raw]:
        if _RE_STRUCT_BOARD_ROLE.search(sent) and _RE_STRUCT_BOARD_VERB.search(sent):
            board = True
            break

    if funding:
        conf += 40
    if promotion:
        conf += 30
    if ma:
        conf += 40
    if board:
        conf += 25

    et: str | None = None
    if ma:
        et = "Founder Exit"
    elif funding:
        et = "Funding"
    elif promotion:
        et = "Promotion"
    elif board:
        et = "Board Appointment"

    leg = (legacy_event_type or "").strip()
    if et is None and leg:
        conf += 25
        et = leg

    return et, conf


def _is_relevant_signal(
    text_blob: str,
    company_name: str = "",
    person_name: str = "",
    legacy_event_type: str | None = None,
    source_url: str = "",
) -> bool:
    """
    Row is dropped only when clearly irrelevant (off-topic needles) or when the blob has
    no financial keywords and no person and no company.

    ``source_url`` / ``legacy_event_type`` are ignored for gating (volume-friendly).
    """
    del legacy_event_type, source_url
    return _article_row_should_keep(text_blob, person_name, company_name)


def _apply_wealth_signal_metadata(out: pd.DataFrame) -> None:
    """
    Populate wealth-signal fields and ``priority_level`` from HNWI / liquidity rules.

    HIGH priority requires ``passes_wealth_high_priority_gate`` (has / made / about to make money).
    """
    if out.empty:
        return
    for idx in out.index:
        r = out.loc[idx]
        blob = f"{r.get('raw_title', '')} {r.get('full_explanation', '')}"
        gate = passes_wealth_high_priority_gate(
            raw_title=str(r.get("raw_title", "")),
            full_explanation=str(r.get("full_explanation", "")),
            event_type=str(r.get("event_type", "")),
            role=str(r.get("role", "")),
            estimated_wealth=float(r.get("estimated_wealth") or 0),
            aggregated_estimated_wealth=float(r.get("aggregated_estimated_wealth") or 0),
            funding_amount=str(r.get("funding_amount", "")),
            funding_stage=str(r.get("funding_stage", "")),
            is_billionaire=bool(r.get("is_billionaire", False)),
        )
        macro = is_macro_noise_without_wealth_hook(
            blob,
            float(r.get("estimated_wealth") or 0),
            bool(r.get("is_billionaire", False)),
            gate,
        )
        strength = classify_wealth_signal_strength(
            raw_title=str(r.get("raw_title", "")),
            full_explanation=str(r.get("full_explanation", "")),
            event_type=str(r.get("event_type", "")),
            role=str(r.get("role", "")),
            estimated_wealth=float(r.get("estimated_wealth") or 0),
            aggregated_estimated_wealth=float(r.get("aggregated_estimated_wealth") or 0),
            is_billionaire=bool(r.get("is_billionaire", False)),
            linked_wealth_signal=bool(r.get("linked_wealth_signal", False)),
            funding_amount=str(r.get("funding_amount", "")),
            funding_stage=str(r.get("funding_stage", "")),
            weak_signal=bool(r.get("weak_signal", False)),
        )
        liq = classify_liquidity_event(
            str(r.get("event_type", "")),
            str(r.get("raw_title", "")),
            str(r.get("full_explanation", "")),
            str(r.get("funding_amount", "")),
            str(r.get("funding_stage", "")),
        )
        ctype = classify_client_type(
            str(r.get("role", "")),
            str(r.get("event_type", "")),
            str(r.get("raw_title", "")),
        )
        ch = str(r.get("client_type_hint", "") or "").strip()
        if ch and "unknown" not in ch.lower():
            ctype = ch
        sow = infer_source_of_wealth(
            str(r.get("event_type", "")),
            str(r.get("raw_title", "")),
            str(r.get("full_explanation", "")),
        )
        sh = str(r.get("source_of_wealth_hint", "") or "").strip()
        if sh:
            sow = sh
        wr = wealth_signal_rank(strength)
        sc = int(pd.to_numeric(r.get("score"), errors="coerce") or 0)
        pl = derive_wealth_priority_level(
            score=sc,
            passes_gate=gate,
            strength=strength,
            liquidity=liq,
            macro_noise=macro,
            weak_signal=bool(r.get("weak_signal", False)),
            is_billionaire=bool(r.get("is_billionaire", False)),
        )
        out.at[idx, "wealth_passes_gate"] = bool(gate)
        out.at[idx, "wealth_signal_label"] = strength
        out.at[idx, "liquidity_event"] = liq
        out.at[idx, "client_type"] = ctype
        out.at[idx, "source_of_wealth"] = sow
        out.at[idx, "wealth_rank"] = int(wr)
        out.at[idx, "priority_level"] = pl


def finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a signals table so the UI always gets safe, consistent data.

    - Ensures all REQUIRED_COLUMNS exist
    - Coerces ``additional_people`` to a list, promotes extras to ``person_name`` when needed,
      optionally splits one RSS row into several rows (``SPLIT_SIGNAL_ROWS_PER_PERSON``)
    - Fills missing text fields; company_name defaults to 'Unknown' (with anchor fallbacks when
      person_name is missing — at/from/of in title, then URL domain, then headline snippet)
    - Serializes ``additional_people`` to a JSON array string for the UI
    - Parses event_date to datetime (invalid -> NaT)
    - Sets detected_at (when the row was ingested); missing values become “now”
    - Fills empty why_it_matters using why_it_matters_for_event_type
    - Re-applies additive ``score`` from ``score.py`` (funding / deal / executive / person / company
      weights and penalties; uncapped), then billionaire (+20), cross-article (+10/+20), repeat-company (+10),
      then **single** clamp to 0–100
    - Enriches ``industry``, ``stage``, ``company_description``, ``company_location`` via
      ``enrich_company_data`` (mock + article fallback)
    - Sets ``funding_amount`` / ``funding_stage`` from article text via ``extract_funding_fields_from_text``
    - Computes ``wealth_score``, ``estimated_wealth`` (deal size × ownership), merges ``target_client``
      (rule-based + deal tiers); billionaire-list +20 to ``score`` (before final clamp) and ``priority`` tags
    - Cross-article enrichment: same normalized ``person_name`` → ``repeat_person`` (+10 when 2+ rows);
      shared ``linked_wealth_signal`` (+20 on all rows for that person when any row has funding,
      acquisition, or deal value); same normalized ``company_name`` on 2+ rows → ``repeat_company`` (+10);
      company-level funding is copied to sibling rows before scoring so each person inherits the deal;
      ``aggregated_estimated_wealth`` sums per-row deal estimates by person — at **$10M+** marks target client
    - Fills outreach_angle, priority_level, suggested_next_step (action layer)
    - Recomputes quality_score (0-8) and confidence_score (0-100) from final fields
    - Dedupes across sources by person+company+event (or by raw_title when person is missing),
      keeping the highest score
    - Drops duplicate source_url rows, keeping the highest confidence / score first
    - Drops rows only when the story is clearly off-topic (small blocklist) or has no finance keywords
      and no person and no company (see ``_article_row_should_keep``)
    - Rows flagged ``weak_signal`` (no strict finance/career/transaction regex on the ingest blob) get
      ``event_type`` = Other, ``why_it_matters`` for Other, and a final score in 30–50 (stable from URL)
    """
    if df is None or df.empty:
        out = pd.DataFrame(columns=REQUIRED_COLUMNS)
    else:
        out = df.copy()

    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    # --- Multiple people: list form, promote primary, optional one-row-per-person ---
    if not out.empty:
        out["additional_people"] = out["additional_people"].apply(coerce_additional_people_list)
        normalize_primary_and_additional_people(out)
        out = expand_dataframe_rows_per_person(out)

    # --- Strings: never NaN in the UI; trim whitespace ---
    for col in ("person_name", "role", "source_url", "full_explanation", "raw_title"):
        out[col] = out[col].fillna("").astype(str).str.strip()

    out["company_name"] = out["company_name"].fillna("").astype(str).str.strip()
    if not out.empty:
        ensure_signal_anchor(out)
    out["company_name"] = out["company_name"].map(normalize_company_name_field)

    out["event_type"] = out["event_type"].fillna("").astype(str).str.strip()
    out["why_it_matters"] = out["why_it_matters"].fillna("").astype(str).str.strip()

    # --- Company enrichment (mock lookup + article fallback; failures are no-ops) ---
    if not out.empty:
        for idx in out.index:
            try:
                blob = f"{out.at[idx, 'raw_title']} {out.at[idx, 'full_explanation']}"
                en = enrich_company_data(str(out.at[idx, 'company_name'] or ''), blob)
                out.at[idx, "industry"] = (en.get("industry") or "").strip()
                out.at[idx, "stage"] = (en.get("stage") or "").strip()
                out.at[idx, "company_description"] = (en.get("description") or "").strip()
                out.at[idx, "company_location"] = (en.get("location") or "").strip()
                fd = extract_funding_fields_from_text(blob)
                out.at[idx, "funding_amount"] = (fd.get("funding_amount") or "").strip()
                out.at[idx, "funding_stage"] = (fd.get("funding_stage") or "").strip()
            except Exception:
                pass
        for col in (
            "industry",
            "stage",
            "company_description",
            "company_location",
            "funding_amount",
            "funding_stage",
        ):
            if col in out.columns:
                out[col] = out[col].fillna("").astype(str).str.strip()

        out["is_relevant"] = out.apply(
            lambda r: _is_relevant_signal(
                f"{r.get('raw_title', '')} {r.get('full_explanation', '')}",
                str(r.get("company_name", "")),
                str(r.get("person_name", "")),
                legacy_event_type=str(r.get("event_type", "") or "").strip() or None,
                source_url=str(r.get("source_url", "") or ""),
            ),
            axis=1,
        )
        out = out[out["is_relevant"]].copy()

    # Fill why_it_matters when missing
    missing_blurb = out["why_it_matters"] == ""
    out.loc[missing_blurb, "why_it_matters"] = out.loc[missing_blurb, "event_type"].map(why_it_matters_for_event_type)

    # --- Dates: parse safely; bad values become NaT ---
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")

    # --- Freshness: when this row entered our pipeline (for “live feed” UI) ---
    out["detected_at"] = pd.to_datetime(out["detected_at"], errors="coerce", utc=True)
    missing_detected = out["detected_at"].isna()
    _now = pd.Timestamp.now(tz=timezone.utc)
    try:
        _now = _now.as_unit("s")
    except (AttributeError, ValueError):
        pass
    out.loc[missing_detected, "detected_at"] = _now

    if not out.empty:
        propagate_company_funding_to_people(out)

    # --- Scores: additive model (score.py); uncapped until final clamp below ---
    out["score"] = out.apply(
        lambda row: compute_signal_score(
            str(row.get("event_type", "")),
            str(row.get("person_name", "")),
            str(row.get("company_name", "")),
            str(row.get("role", "")),
            raw_title=str(row.get("raw_title", "")),
            full_explanation=str(row.get("full_explanation", "")),
            funding_amount=str(row.get("funding_amount", "")),
            funding_stage=str(row.get("funding_stage", "")),
        ),
        axis=1,
    )
    if not out.empty:

        out["wealth_score"] = out.apply(
            lambda row: compute_wealth_score(
                str(row.get("raw_title", "")),
                str(row.get("full_explanation", "")),
                str(row.get("funding_amount", "")),
                str(row.get("event_type", "")),
                str(row.get("role", "")),
            ),
            axis=1,
        ).astype(int)

        _deal_tuples = out.apply(
            lambda row: estimate_wealth_from_deal(
                str(row.get("raw_title", "")),
                str(row.get("full_explanation", "")),
                str(row.get("role", "")),
            ),
            axis=1,
        )
        out["estimated_wealth"] = _deal_tuples.map(lambda t: float(t[0])).astype(float)
        out["estimated_wealth"] = out.apply(
            lambda row: row["estimated_wealth"]
            if float(row["estimated_wealth"] or 0) > 0
            else infer_wealth(row),
            axis=1,
        ).astype(float)
        out["estimated_wealth"] = out.apply(enrich_known_wealth, axis=1).astype(float)
        assign_aggregated_estimated_wealth(out)
        out["target_client"] = out.apply(
            lambda row: merge_target_client_row(
                int(row.get("wealth_score", 0) or 0),
                float(row.get("estimated_wealth", 0) or 0),
                float(row.get("aggregated_estimated_wealth", 0) or 0),
            ),
            axis=1,
        )
        _enrich_signals_with_billionaire_list(out)
        _apply_value_priority_tags(out)
        apply_cross_article_enrichment(out)
        apply_cross_company_enrichment(out)

        out["score"] = out["score"].map(clamp_score_0_100).astype(int)

        weak = out["weak_signal"].fillna(False).astype(bool)
        if weak.any():
            out.loc[weak, "event_type"] = "Other"
            _wm_other = why_it_matters_for_event_type("Other")
            out.loc[weak, "why_it_matters"] = _wm_other
            out.loc[weak, "score"] = out.loc[weak, "source_url"].map(
                lambda u: _weak_signal_score_from_url(str(u or ""))
            ).astype(int)
    else:
        out["wealth_score"] = 0
        out["estimated_wealth"] = 0.0
        out["aggregated_estimated_wealth"] = 0.0
        out["target_client"] = False
        out["repeat_person"] = False
        out["linked_wealth_signal"] = False
        out["repeat_company"] = False
        out["wealth_passes_gate"] = False
        out["wealth_signal_label"] = "None"
        out["liquidity_event"] = "No"
        out["client_type"] = ""
        out["source_of_wealth"] = ""
        out["wealth_rank"] = 3
        out["priority_level"] = "Low"
        out["ai_wealth_signal"] = ""
        out["ai_liquidity_label"] = ""
        out["ai_client_who"] = ""
        out["ai_why_money"] = ""

    # --- Wealth-signal priority (HNWI / liquidity; not general news importance) ---
    if not out.empty:
        _apply_wealth_signal_metadata(out)

    # --- Action layer: outreach copy (priority comes from ``_apply_wealth_signal_metadata``) ---
    out["outreach_angle"] = out.apply(generate_outreach_angle, axis=1)
    out["suggested_next_step"] = out["priority_level"].apply(suggested_next_step_from_priority)

    for col in (
        "outreach_angle",
        "priority_level",
        "suggested_next_step",
        "priority",
        "ai_summary",
        "ai_why_it_matters",
        "ai_outreach",
        "wealth_signal_label",
        "liquidity_event",
        "client_type",
        "source_of_wealth",
        "ai_wealth_signal",
        "ai_liquidity_label",
        "ai_client_who",
        "ai_why_money",
    ):
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str).str.strip()

    # --- Extraction quality + confidence (single source of truth for hero sections) ---
    if not out.empty:
        out["quality_score"] = out.apply(compute_extraction_quality, axis=1).astype(int)
        out["confidence_score"] = out.apply(compute_confidence_score, axis=1).astype(int)

    # --- Dedupe across sources (same person/event or same headline without a person) ---
    if not out.empty:
        out = dedupe_signals_cross_source(out)

    # --- Dedupe URLs: keep strongest row per URL ---
    if not out.empty and "source_url" in out.columns:
        if "wealth_rank" not in out.columns:
            out["wealth_rank"] = 3
        out["wealth_rank"] = pd.to_numeric(out["wealth_rank"], errors="coerce").fillna(3).astype(int)
        _pn = out["person_name"].fillna("").astype(str).str.strip()
        out["_person_id"] = (_pn != "").astype(int)
        out = out.sort_values(
            ["wealth_rank", "_person_id", "event_date", "quality_score", "score"],
            ascending=[True, False, False, False, False],
            na_position="last",
        )
        out = out.drop(columns=["_person_id"], errors="ignore")
        before = len(out)
        if SPLIT_SIGNAL_ROWS_PER_PERSON:
            out = out.drop_duplicates(subset=["source_url", "person_name"], keep="first")
        else:
            out = out.drop_duplicates(subset=["source_url"], keep="first")
        if before != len(out):
            pass  # duplicates removed quietly; app does not need to know

    # --- Serialize additional_people (list in memory → JSON for stable column dtype) ---
    if not out.empty and "additional_people" in out.columns:
        out["additional_people"] = out["additional_people"].apply(
            lambda x: json.dumps(coerce_additional_people_list(x))
        )

    if not out.empty and "weak_signal" in out.columns:
        out["weak_signal"] = out["weak_signal"].fillna(False).astype(bool)

    if "estimated_wealth" in out.columns:
        out["est_wealth_display"] = out["estimated_wealth"].map(format_wealth)
    else:
        out["est_wealth_display"] = "Data pending"

    return out[REQUIRED_COLUMNS].reset_index(drop=True)


def _strip_html(text: str) -> str:
    """Remove simple HTML tags from RSS summaries for display."""
    if not text:
        return ""
    no_tags = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", no_tags).strip()


def _finance_career_broad(title: str) -> bool:
    """
    Loose filter: likely finance, business, tech, or career-adjacent content.
    Used to keep headlines as 'Other' when they do not match the four core buckets.
    """
    t = title.lower()
    needles = (
        "stock",
        "market",
        "investor",
        "invest",
        "investing",
        "startup",
        "venture",
        "ceo",
        "cfo",
        "cto",
        "coo",
        "chief",
        "executive",
        "earnings",
        "revenue",
        "billion",
        "million",
        "ipo",
        "economy",
        "economic",
        "bank",
        "banking",
        "equity",
        "crypto",
        "bitcoin",
        "tech",
        "technology",
        "software",
        "hardware",
        "company",
        "companies",
        "corporate",
        "layoff",
        "layoffs",
        "hiring",
        "hire",
        "merger",
        "deal",
        "deals",
        "quarter",
        "profit",
        "loss",
        "nasdaq",
        "s&p",
        "fund",
        "funding",
        "funded",
        "unicorn",
        "valuation",
        "round",
        "seed",
        "google",
        "microsoft",
        "amazon",
        "apple",
        "meta",
        "tesla",
        "nvidia",
        "openai",
        "anthropic",
        "ai ",
        " ai",
        "finance",
        "financial",
        "business",
        "sales",
        "growth",
        "shareholder",
        "share price",
        "wall street",
        "trading",
        "sec ",
        "regulator",
        "acquisition",
        "acquires",
        "lawsuit",
        "spinoff",
        "dividend",
        "bond",
        "yield",
        "inflation",
        "tariff",
        "trade",
        "silicon",
        "cloud",
        "data center",
        "founder",
        "president",
        "chair",
        "chairman",
        "partner",
        "director",
        "board",
        "workforce",
        "job cut",
    )
    return any(n in t for n in needles)


def _has_financial_keywords_for_ingest(blob: str) -> bool:
    """Strict money/career/transaction match OR broad finance/business keywords (higher recall)."""
    return _strict_relevance_patterns_match(blob) or _finance_career_broad(blob)


def _article_row_should_keep(blob: str, person_name: str, company_name: str) -> bool:
    """
    Keep unless the story is clearly off-topic, or it has no finance keywords and no person and no company.

    ``company_name`` should be the normalized/extracted value (``Unknown`` means no company).
    """
    if _contains_clearly_irrelevant_topics(blob):
        return False
    pn = str(person_name or "").strip()
    cn = str(company_name or "").strip()
    has_person = bool(pn)
    has_company = bool(cn and cn.lower() != "unknown")
    has_fin = _has_financial_keywords_for_ingest(blob)
    return bool(has_fin or has_person or has_company)


def _weak_signal_score_from_url(source_url: str) -> int:
    """Stable score in [30, 50] from URL (weak-signal band)."""
    h = hashlib.sha256(str(source_url or "").encode()).hexdigest()
    return 30 + (int(h[:12], 16) % 21)


def _classify_event_type(title: str) -> str | None:
    """
    Map headline → one of the four core types, 'Other', or None (skip).

    Order: specific deal types first, then board vs promotion, then broad 'Other'.
    """
    t = title.lower()

    for kw in (
        "acquired",
        "acquires",
        "acquisition",
        "buys",
        "buyout",
        "merger",
        "takeover",
        "sold",
        "exit",
        "divest",
        "divests",
    ):
        if kw in t:
            return "Founder Exit"

    for kw in (
        "raised",
        "raises",
        "funding",
        "series a",
        "series b",
        "series c",
        "series d",
        "series e",
        "seed round",
        "seed funding",
        "venture round",
        "unicorn",
        "valuation",
        "funding round",
    ):
        if kw in t:
            return "Funding"

    for kw in ("board", "director", "boardroom"):
        if kw in t:
            return "Board Appointment"

    for kw in (
        "promoted",
        "appointed",
        "named partner",
        "named ceo",
        "named cfo",
        "joins as",
        "new role",
        "executive shuffle",
        "succession",
        "stepping down",
        "resigns",
        "resignation",
    ):
        if kw in t:
            return "Promotion"

    if _finance_career_broad(title):
        return "Other"

    return None


def _parse_entry_date(entry: Any) -> str:
    """Turn an RSS entry's date into YYYY-MM-DD (best effort)."""
    try:
        if getattr(entry, "published_parsed", None):
            tt = entry.published_parsed
            return datetime(*tt[:6]).date().isoformat()
        if getattr(entry, "updated_parsed", None):
            tt = entry.updated_parsed
            return datetime(*tt[:6]).date().isoformat()
    except (TypeError, ValueError):
        pass
    return datetime.now(timezone.utc).date().isoformat()


# --- Person name extraction (conservative; empty string unless confident) ---
_PERSON_NAME_PAIR = re.compile(r"\b([A-Z][a-z]+) ([A-Z][a-z]+)\b")
_NAME_CONTEXT_KEYWORDS = re.compile(
    r"\b(?:appointed|appoints|joins|joining|named|names|promoted|promotes|promotion|ceo|cfo|cto|coo|founder|co-founder)\b",
    re.IGNORECASE,
)
_BLOCKED_NAME_TOKENS = frozenset(
    {
        "london",
        "paris",
        "berlin",
        "tokyo",
        "beijing",
        "shanghai",
        "mumbai",
        "sydney",
        "dublin",
        "moscow",
        "china",
        "india",
        "japan",
        "france",
        "germany",
        "canada",
        "australia",
        "inc",
        "corp",
        "llc",
        "ltd",
        "plc",
        "tech",
        "ai",
        "news",
        "media",
        "markets",
        "street",
        "journal",
        "review",
        "us",
        "uk",
        "eu",
        "nyse",
        "nasdaq",
        "new",
        "san",
        "los",
        "las",
        "hong",
    }
)
_BLOCKED_NAME_PHRASES = frozenset(
    {
        "new york",
        "los angeles",
        "san francisco",
        "las vegas",
        "hong kong",
        "tel aviv",
        "washington dc",
    }
)

# Second word looks like a company suffix, not a surname (e.g. "Pinwheel Labs")
_COMPANY_LIKE_NAME_SECONDS = frozenset(
    {
        "labs",
        "inc",
        "corp",
        "llc",
        "ltd",
        "plc",
        "group",
        "holdings",
        "partners",
        "capital",
        "ventures",
        "networks",
        "systems",
        "solutions",
        "technologies",
        "analytics",
        "robotics",
        "health",
        "bank",
        "security",
        "retail",
        "grid",
        "wire",
        "global",
        "international",
        "media",
    }
)


def _guess_all_person_names(title: str) -> tuple[str, list[str]]:
    """
    Extract every plausible "Firstname Lastname" span the headline supports.

    Returns (primary_name, additional_names). Primary is the first valid match; the rest
    are ordered, de-duplicated (case-insensitive). Empty title → ("", []).
    """
    if not title or not str(title).strip():
        return "", []

    t = str(title).strip()
    letters = [c for c in t if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.72:
        return "", []

    names: list[str] = []
    seen: set[str] = set()
    for m in _PERSON_NAME_PAIR.finditer(t):
        first, last = m.group(1), m.group(2)
        a, b = first.lower(), last.lower()
        if a in _BLOCKED_NAME_TOKENS or b in _BLOCKED_NAME_TOKENS:
            continue
        if b in _COMPANY_LIKE_NAME_SECONDS:
            continue
        if f"{a} {b}" in _BLOCKED_NAME_PHRASES:
            continue

        start, end = m.span()
        lo = max(0, start - 72)
        hi = min(len(t), end + 72)
        window = t[lo:hi]
        if not _NAME_CONTEXT_KEYWORDS.search(window):
            continue

        full = f"{first} {last}"
        key = full.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(full)

    if not names:
        return "", []
    return names[0], names[1:]


def _guess_person_name(title: str) -> str:
    """Backward-compatible: first detected name only."""
    primary, _ = _guess_all_person_names(title)
    return primary


def coerce_additional_people_list(val: Any) -> list[str]:
    """Normalize ``additional_people`` from list, JSON string, or missing → list of non-empty strings."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        s = val.strip()
        if not s or s == "[]":
            return []
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except json.JSONDecodeError:
                return []
        return []
    return []


def normalize_primary_and_additional_people(df: pd.DataFrame) -> None:
    """
    If ``person_name`` is empty but ``additional_people`` has values, promote the first
    extra name to ``person_name`` so the row always follows primary + extras semantics.
    Mutates ``df`` in place (list values in ``additional_people``).
    """
    if df.empty or "additional_people" not in df.columns:
        return
    for idx in df.index:
        pn = str(df.at[idx, "person_name"] or "").strip() if "person_name" in df.columns else ""
        ap = coerce_additional_people_list(df.at[idx, "additional_people"])
        if not pn and ap:
            df.at[idx, "person_name"] = ap[0]
            df.at[idx, "additional_people"] = ap[1:]


def expand_dataframe_rows_per_person(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optionally duplicate rows so each detected person gets its own row (same story metadata).

    ``person_name`` becomes that person; ``additional_people`` lists every other name (JSON
    list string applied later in finalize).
    """
    if df.empty or not SPLIT_SIGNAL_ROWS_PER_PERSON:
        return df
    out_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        pn = str(row.get("person_name") or "").strip()
        ap = coerce_additional_people_list(row.get("additional_people"))
        ordered: list[str] = []
        seen: set[str] = set()
        if pn:
            kl = pn.lower()
            if kl not in seen:
                seen.add(kl)
                ordered.append(pn)
        for n in ap:
            kl = n.lower()
            if kl not in seen:
                seen.add(kl)
                ordered.append(n)
        if len(ordered) <= 1:
            out_rows.append(row.to_dict())
            continue
        for name in ordered:
            d = row.to_dict()
            d["person_name"] = name
            d["additional_people"] = [x for x in ordered if x != name]
            out_rows.append(d)
    return pd.DataFrame(out_rows).reset_index(drop=True)


def _guess_company_name(title: str) -> str:
    """
    Very rough company guess; default is 'Unknown' per requirements.
    """
    t = title
    lower = t.lower()
    candidate = ""

    # "... acquires/buys Something ..."
    for needle in (" acquires ", " acquired ", " buys "):
        idx = lower.find(needle)
        if idx != -1:
            rest = t[idx + len(needle) :].strip()
            # Stop at common delimiters
            rest = re.split(r" for |,|\.|;|\||–|-", rest, maxsplit=1)[0].strip()
            if rest and len(rest) < 120:
                candidate = rest
                break

    if not candidate:
        m = re.search(r"\s+at\s+([A-Za-z0-9][A-Za-z0-9 &\-\.]{1,60}?)(?:\s|$|,|\.)", t)
        if m:
            candidate = m.group(1).strip()

    if not candidate:
        m = re.search(r"\s+from\s+([A-Za-z0-9][A-Za-z0-9 &\-\.]{1,60}?)(?:\s|$|,|\.)", t)
        if m:
            candidate = m.group(1).strip()

    if not candidate:
        m = re.search(r"\s+of\s+([A-Za-z0-9][A-Za-z0-9 &\-\.]{1,60}?)(?:\s|$|,|\.)", t)
        if m:
            candidate = m.group(1).strip()

    if not candidate:
        m = re.search(r"\s+joins\s+([A-Za-z0-9][A-Za-z0-9 &\-\.]{1,60}?)(?:\s|$|,|\.)", t)
        if m:
            candidate = m.group(1).strip()

    if not candidate:
        m = re.search(r"appointed\s+to\s+([A-Za-z0-9][A-Za-z0-9 &\-\.]{1,60}?)\s+board", t, re.I)
        if m:
            candidate = m.group(1).strip()

    if not candidate:
        m = re.match(r"^([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)?)\s+raises\b", t)
        if m:
            candidate = m.group(1).strip()

    return normalize_company_name_field(candidate)


def _guess_role(title: str) -> str:
    """Optional role hint from title; empty if we can't tell."""
    t = title.lower()
    role_map = {
        "ceo": "CEO",
        "cfo": "CFO",
        "cto": "CTO",
        "coo": "COO",
        "chief": "Chief",
        "president": "President",
        "partner": "Partner",
        "director": "Director",
        "chair": "Chair",
        "founder": "Founder",
        "co-founder": "Co-Founder",
        "board member": "Board Member",
        "executive director": "Executive Director",
    }
    for phrase, label in role_map.items():
        if phrase in t:
            return label
    return ""


def _person_name_fails_validation(person: str) -> bool:
    """True when regex gave no name or a name that fails ``is_valid_person`` (needs AI fallback)."""
    p = str(person or "").strip()
    return not p or not is_valid_person(p)


def _ai_extraction_enabled() -> bool:
    return os.environ.get("WEALTH_SIGNALS_AI_EXTRACTION", "1").lower() not in ("0", "false", "no")


def _merge_ai_extraction_for_row(
    title: str,
    article_text: str,
    summary: str,
    person: str,
    extra_people: list[str],
    company: str,
    role: str,
    event_type: str,
) -> tuple[str, list[str], str, str, str, str, str]:
    """
    Hybrid extraction: regex/heuristics first; if ``person_name`` fails validation, call AI
    and **replace** ``person_name``, ``company_name``, ``role``, and ``event_type`` with the
    model output (``additional_people`` cleared when AI runs). If the API returns nothing
    useful, keep regex fields unchanged.
    """
    if not _ai_extraction_enabled() or not os.environ.get("OPENAI_API_KEY", "").strip():
        return person, extra_people, company, role, event_type, "", ""
    if not _person_name_fails_validation(person):
        return person, extra_people, company, role, event_type, "", ""
    # Prefer article body first so downstream truncation keeps real paragraphs over the title.
    chunks = [article_text, title, summary]
    combined = "\n\n".join(c for c in chunks if (c or "").strip()).strip()
    if not combined:
        combined = title
    parsed = extract_signal_with_ai(combined)
    if not parsed:
        return person, extra_people, company, role, event_type, "", ""

    # Full overwrite from AI for structured fields (regex extras dropped — single AI snapshot)
    person = str(parsed.get("person_name", "") or "").strip()
    extra_people = []
    company = normalize_company_name_field(str(parsed.get("company_name", "") or "").strip())
    role = str(parsed.get("role", "") or "").strip()
    et_ai = str(parsed.get("event_type", "") or "").strip()
    if et_ai:
        event_type = et_ai
    ct = str(parsed.get("client_type", "") or "").strip()
    sow = str(parsed.get("source_of_wealth", "") or "").strip()
    return person, extra_people, company, role, event_type, ct, sow


def _fetch_one_feed(url: str) -> list[Any]:
    """Download and parse a single RSS URL; returns entries or []."""
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
        return list(getattr(parsed, "entries", []) or [])
    except (requests.RequestException, OSError) as e:
        # Network / HTTP errors: caller aggregates and may fall back to sample data
        raise RuntimeError(f"RSS fetch failed for {url}: {e}") from e


def _rss_items_to_signals(entries: list[Any]) -> list[dict[str, Any]]:
    """
    Turn RSS entries into row dicts.

    Pipeline per row: fetch article HTML and join ``<p>`` text when possible; run classification
    and regex on that body (fallback: RSS title). Then AI if ``person_name`` invalid; build row.
    """
    rows: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for entry in entries:
        title = (getattr(entry, "title", None) or "").strip()
        if not title:
            continue

        link = (getattr(entry, "link", None) or "").strip()
        if not link:
            continue
        if link in seen_urls:
            continue
        seen_urls.add(link)

        summary = _strip_html(getattr(entry, "summary", None) or getattr(entry, "description", None) or "")
        base_summary = summary or f"(No summary in feed.) Headline: {title}"

        article_paragraphs = fetch_article_paragraph_text(link)
        body_snip = (article_paragraphs or "").strip()[:3000]
        full_explanation = base_summary
        if body_snip:
            full_explanation = (base_summary + "\n\n" + body_snip).strip()[:4000]

        extraction_text = _extraction_text(article_paragraphs, title)

        blob_for_filter = f"{title} {summary} {article_paragraphs}".strip()

        legacy_et = _classify_event_type(extraction_text) or "Other"

        # 1) Fast regex / heuristics on article body (or title if fetch failed)
        person, extra_people = _guess_all_person_names(extraction_text)
        company = _guess_company_name(extraction_text)
        role = _guess_role(extraction_text)

        # 2) If person_name missing or invalid → AI fallback; 3) overwrite with AI output when used
        person, extra_people, company, role, event_type_merged, client_type_hint, source_of_wealth_hint = (
            _merge_ai_extraction_for_row(
                title, article_paragraphs, summary, person, extra_people, company, role, legacy_et
            )
        )

        if not _article_row_should_keep(blob_for_filter, person, company):
            continue

        weak_signal = not _strict_relevance_patterns_match(blob_for_filter)

        scan_text = f"{title} {summary} {extraction_text}".strip()
        structured_et, _ = structured_pattern_confidence_and_type(scan_text, legacy_event_type=legacy_et)
        event_type = structured_et if structured_et is not None else event_type_merged

        why = why_it_matters_for_event_type(event_type)

        rows.append(
            {
                "person_name": person,
                "additional_people": extra_people,
                "company_name": company,
                "event_type": event_type,
                "raw_title": title,
                "role": role,
                "event_date": _parse_entry_date(entry),
                "detected_at": datetime.now(timezone.utc),
                "why_it_matters": why,
                "source_url": link,
                "full_explanation": full_explanation[:4000],
                "quality_score": 0,
                "confidence_score": 0,
                "is_relevant": True,
                "weak_signal": weak_signal,
                "client_type_hint": client_type_hint,
                "source_of_wealth_hint": source_of_wealth_hint,
            }
        )

    return rows


def _raw_sample_signals() -> list[dict]:
    """
    Hardcoded demo signals - used when RSS is unavailable or errors out.
    Each dict matches the columns expected by the Streamlit app.
    """
    return [
        {
            "person_name": "Alex Rivera",
            "additional_people": [],
            "company_name": "Northbeam Analytics",
            "event_type": "Founder Exit",
            "raw_title": "Northbeam Analytics acquired in nine-figure strategic exit - Alex Rivera, Co-founder & CEO",
            "role": "Co-founder & CEO",
            "event_date": "2026-03-12",
            "why_it_matters": "Company acquisition often triggers liquidity for founders and early equity holders.",
            "source_url": "https://www.reuters.com/technology/",
            "full_explanation": (
                "Public M&A announcements typically disclose the buyer and sometimes the deal structure. "
                "Founders often reinvest or take time off after an exit; this is a classic 'wealth signal' "
                "for advisors and peers tracking career moves."
            ),
            "quality_score": 4,
        },
        {
            "person_name": "Jordan Lee",
            "additional_people": [],
            "company_name": "Helio Robotics",
            "event_type": "Funding",
            "raw_title": "Helio Robotics raises $120M Series B led by top-tier VCs",
            "role": "CTO",
            "event_date": "2026-02-28",
            "why_it_matters": "Large funding rounds can mean option refreshes, bonuses, or future liquidity events.",
            "source_url": "https://techcrunch.com/",
            "full_explanation": (
                "Venture funding is public when companies issue press releases. "
                "Senior leaders may receive new equity grants tied to milestones after a round."
            ),
            "quality_score": 4,
        },
        {
            "person_name": "Sam Patel",
            "additional_people": [],
            "company_name": "Crescent Health",
            "event_type": "Promotion",
            "raw_title": "Crescent Health names Sam Patel SVP of Product in executive promotion",
            "role": "VP of Product → SVP",
            "event_date": "2026-03-01",
            "why_it_matters": "Senior promotions often coincide with compensation step-ups and equity band changes.",
            "source_url": "https://www.businesswire.com/",
            "full_explanation": (
                "Companies frequently announce executive promotions via press releases. "
                "These roles usually carry higher base, bonus, and long-term incentive weight."
            ),
            "quality_score": 4,
        },
        {
            "person_name": "Taylor Kim",
            "additional_people": [],
            "company_name": "Meridian Bank",
            "event_type": "Board Appointment",
            "raw_title": "Meridian Bank named Taylor Kim independent director",
            "role": "Independent Director",
            "event_date": "2026-01-20",
            "why_it_matters": "Board roles can include cash retainers, equity, and visibility into major decisions.",
            "source_url": "https://www.sec.gov/edgar/search/",
            "full_explanation": (
                "Public companies file board changes with regulators (e.g., 8-K). "
                "Director compensation is disclosed in proxy statements for listed firms."
            ),
            "quality_score": 4,
        },
        {
            "person_name": "Morgan Chen",
            "additional_people": [],
            "company_name": "Lumen Grid",
            "event_type": "Founder Exit",
            "raw_title": "Lumen Grid founder Morgan Chen exits after strategic deal",
            "role": "Founder",
            "event_date": "2025-12-05",
            "why_it_matters": "Second-time founders often recycle capital into new ventures or angel investing.",
            "source_url": "https://www.bloomberg.com/news/",
            "full_explanation": (
                "Exit events are among the strongest signals because they can unlock significant personal liquidity, "
                "subject to earn-outs and vesting."
            ),
            "quality_score": 4,
        },
        {
            "person_name": "Riley Brooks",
            "additional_people": [],
            "company_name": "Atlas Security",
            "event_type": "Funding",
            "raw_title": "Atlas Security closes $85M funding round to scale enterprise sales",
            "role": "Chief Revenue Officer",
            "event_date": "2026-03-18",
            "why_it_matters": "GTM leaders are often rewarded when the company raises capital to scale sales.",
            "source_url": "https://www.prnewswire.com/",
            "full_explanation": (
                "Funding news is often paired with hiring and expansion plans. "
                "Commission and equity structures may change after a new round closes."
            ),
            "quality_score": 4,
        },
        {
            "person_name": "Casey Nguyen",
            "additional_people": [],
            "company_name": "Harbor Freight AI",
            "event_type": "Promotion",
            "raw_title": "Harbor Freight AI named Casey Nguyen VP of Engineering",
            "role": "Director of Engineering → VP Engineering",
            "event_date": "2026-02-10",
            "why_it_matters": "VP-level promotions in tech usually reflect expanded scope and pay band.",
            "source_url": "https://www.globenewswire.com/",
            "full_explanation": (
                "Engineering leadership promotions are sometimes announced alongside product launches or reorgs."
            ),
            "quality_score": 4,
        },
        {
            "person_name": "Jamie Foster",
            "additional_people": [],
            "company_name": "Silverline Retail",
            "event_type": "Board Appointment",
            "raw_title": "Silverline Retail appoints investor board observer Jamie Foster",
            "role": "Board Observer (investor seat)",
            "event_date": "2026-03-04",
            "why_it_matters": "Investor board observers gain strategic insight and network leverage.",
            "source_url": "https://www.ft.com/",
            "full_explanation": (
                "Board-related news can appear in specialty press or company blogs; always verify on primary sources."
            ),
            "quality_score": 4,
        },
        {
            "person_name": "Avery Smith",
            "additional_people": ["Blake Jones"],
            "company_name": "Pinwheel Labs",
            "event_type": "Board Appointment",
            "raw_title": (
                "Pinwheel Labs appoints co-founders Avery Smith and Blake Jones as board observers "
                "after Series B"
            ),
            "role": "Board Observer",
            "event_date": "2026-03-22",
            "why_it_matters": "Multiple named leaders in one story can mean several outreach threads from a single item.",
            "source_url": "https://www.pinwheel-labs.example/press/board-observers",
            "full_explanation": (
                "When several executives are named, capture each as a separate contact angle or split rows per person."
            ),
            "quality_score": 4,
        },
    ]


def load_sample_signals() -> pd.DataFrame:
    """Load demo signals, score them, then normalize for the app."""
    raw_list = _raw_sample_signals()
    rows = apply_scores(raw_list)
    df = pd.DataFrame(rows)
    out = finalize_dataframe(df)
    out = enrich_dataframe_with_ai_interpretation(out)
    out.attrs["ingest_debug"] = {
        "raw_rss_entries": len(raw_list),
        "parsed_signal_rows": len(raw_list),
        "rows_after_finalize": len(out),
        "data_source": "sample_fallback",
    }
    return out


def fetch_signals() -> pd.DataFrame:
    """
    Load signals from public RSS feeds, then score and clean them.

    - If every feed fails or returns no entries: return load_sample_signals().
    - If feeds return items but nothing classifies: empty dataframe with ingest_debug
      (so you can see raw count vs parsed count - not hidden behind sample data).
    - On success: rows from RSS with attrs['ingest_debug'] for pipeline transparency.
    """
    column_order = REQUIRED_COLUMNS

    all_entries: list[Any] = []

    for url in RSS_FEEDS:
        try:
            all_entries.extend(_fetch_one_feed(url))
        except RuntimeError:
            # One feed can fail; others may still work. If all fail, we fall back below.
            continue

    if not all_entries:
        # Every feed failed or returned no entries - use sample data
        return load_sample_signals()

    raw_rows = _rss_items_to_signals(all_entries)
    ingest_debug: dict[str, Any] = {
        "raw_rss_entries": len(all_entries),
        "parsed_signal_rows": len(raw_rows),
        "data_source": "rss",
    }

    if not raw_rows:
        out = finalize_dataframe(pd.DataFrame(columns=column_order))
        out = enrich_dataframe_with_ai_interpretation(out)
        ingest_debug["rows_after_finalize"] = len(out)
        out.attrs["ingest_debug"] = ingest_debug
        return out

    scored = apply_scores(raw_rows)
    df = pd.DataFrame(scored)
    out = finalize_dataframe(df)
    out = enrich_dataframe_with_ai_interpretation(out)
    ingest_debug["rows_after_finalize"] = len(out)
    out.attrs["ingest_debug"] = ingest_debug
    return out
