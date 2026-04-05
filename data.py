"""
Data layer: sample signals (fallback) and live RSS-based signals.

Uses public RSS feeds only - no LinkedIn or private sources.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import feedparser
import pandas as pd
import requests

from score import apply_scores, score_for_event_type

# -----------------------------------------------------------------------------
# Column contract: every row returned to the app must have these keys.
# -----------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "person_name",
    "company_name",
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
    "source_url",
    "full_explanation",
    "quality_score",
    "confidence_score",
]

# -----------------------------------------------------------------------------
# Public RSS feeds (business / tech news). Swap or extend as needed.
# -----------------------------------------------------------------------------
RSS_FEEDS = [
    # Tech / VC / startups
    "https://techcrunch.com/feed/",
    "https://venturebeat.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://www.techmeme.com/feed.xml",
    # Business & markets
    "https://feeds.arstechnica.com/arstechnica/business",
    "https://rss.cnn.com/rss/money_rss.xml",
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    # Executive / leadership / funding (Google News search RSS - public HTML)
    "https://news.google.com/rss/search?q=startup+funding+OR+CEO+appointment&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=board+of+directors+OR+executive+hire&hl=en-US&gl=US&ceid=US:en",
]

# Browser-like User-Agent: some feeds block generic Python clients.
REQUEST_HEADERS = {
    "User-Agent": "WealthSignalsDashboard/1.0 (+https://example.local; educational project)",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

REQUEST_TIMEOUT_SEC = 20


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
    return q


def compute_confidence_score(row: pd.Series) -> int:
    """
    0-100: combines extraction quality with signal score for curated top sections.

    Higher quality dominates; score adds a modest bump so High-priority rows surface.
    """
    q = int(row.get("quality_score", 0) or 0)
    s = int(row.get("score", 0) or 0)
    bump = min(15, max(0, s - 45) // 3)
    return int(min(100, q * 11 + bump))


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
        "Other": "General business or markets news - may still be a timely reason to reach out.",
    }
    et = (event_type or "").strip()
    return blurbs.get(et, "Public career and finance news may signal changing wealth dynamics.")


def finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a signals table so the UI always gets safe, consistent data.

    - Ensures all REQUIRED_COLUMNS exist
    - Fills missing text fields; company_name defaults to 'Unknown'
    - Parses event_date to datetime (invalid -> NaT)
    - Sets detected_at (when the row was ingested); missing values become “now”
    - Fills empty why_it_matters using why_it_matters_for_event_type
    - Re-applies scores from score.py (single source of truth)
    - Fills outreach_angle, priority_level, suggested_next_step (action layer)
    - Recomputes quality_score (0-8) and confidence_score (0-100) from final fields
    - Drops duplicate source_url rows, keeping the highest confidence / score first
    """
    if df is None or df.empty:
        out = pd.DataFrame(columns=REQUIRED_COLUMNS)
    else:
        out = df.copy()

    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    # --- Strings: never NaN in the UI; trim whitespace ---
    for col in ("person_name", "role", "source_url", "full_explanation", "raw_title"):
        out[col] = out[col].fillna("").astype(str).str.strip()

    out["company_name"] = out["company_name"].fillna("").astype(str).str.strip()
    out.loc[out["company_name"] == "", "company_name"] = "Unknown"

    out["event_type"] = out["event_type"].fillna("").astype(str).str.strip()
    out["why_it_matters"] = out["why_it_matters"].fillna("").astype(str).str.strip()

    # Fill why_it_matters when missing
    missing_blurb = out["why_it_matters"] == ""
    out.loc[missing_blurb, "why_it_matters"] = out.loc[missing_blurb, "event_type"].map(why_it_matters_for_event_type)

    # --- Dates: parse safely; bad values become NaT ---
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")

    # --- Freshness: when this row entered our pipeline (for “live feed” UI) ---
    out["detected_at"] = pd.to_datetime(out["detected_at"], errors="coerce", utc=True)
    missing_detected = out["detected_at"].isna()
    out.loc[missing_detected, "detected_at"] = pd.Timestamp.now(tz=timezone.utc)

    # --- Scores always follow score.py rules for the current event_type ---
    out["score"] = out["event_type"].apply(lambda et: score_for_event_type(str(et)))

    # --- Action layer: who to prioritize and what to say (derived from score + event_type) ---
    out["outreach_angle"] = out.apply(generate_outreach_angle, axis=1)
    out["priority_level"] = out["score"].apply(priority_level_from_score)
    out["suggested_next_step"] = out["priority_level"].apply(suggested_next_step_from_priority)

    for col in ("outreach_angle", "priority_level", "suggested_next_step"):
        out[col] = out[col].fillna("").astype(str).str.strip()

    # --- Extraction quality + confidence (single source of truth for hero sections) ---
    if not out.empty:
        out["quality_score"] = out.apply(compute_extraction_quality, axis=1).astype(int)
        out["confidence_score"] = out.apply(compute_confidence_score, axis=1).astype(int)

    # --- Dedupe URLs: keep strongest row per URL ---
    if not out.empty and "source_url" in out.columns:
        out = out.sort_values(
            ["confidence_score", "score", "event_date"],
            ascending=[False, False, False],
            na_position="last",
        )
        before = len(out)
        out = out.drop_duplicates(subset=["source_url"], keep="first")
        if before != len(out):
            pass  # duplicates removed quietly; app does not need to know

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


def _guess_person_name(title: str) -> str:
    """
    Very light extraction - many headlines won't match; that's OK (empty string).
    Looks for patterns like "Jane Doe joins ..." or "... names Jane Doe CEO".
    Conservative: require exactly two capitalized words for names.
    """
    # "First Last joins|appointed|named|promoted"
    m = re.search(
        r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:joins|appointed|named|promoted)\b",
        title,
    )
    if m:
        return m.group(1).strip()

    # "Company names Jane Doe as ..."
    m = re.search(
        r"\b(?:names|names\s+as)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b",
        title,
        re.I,
    )
    if m:
        return m.group(1).strip()

    # Two capitalized words at the very start (often a person in press titles)
    m = re.match(r"^([A-Z][a-z]+\s+[A-Z][a-z]+)\b", title.strip())
    if m and not title.lower().startswith(("the ", "a ", "an ")):
        return m.group(1).strip()

    return ""


def _guess_company_name(title: str) -> str:
    """
    Very rough company guess; default is 'Unknown' per requirements.
    """
    t = title
    lower = t.lower()

    # "... acquires/buys Something ..."
    for needle in (" acquires ", " acquired ", " buys "):
        idx = lower.find(needle)
        if idx != -1:
            rest = t[idx + len(needle) :].strip()
            # Stop at common delimiters
            rest = re.split(r" for |,|\.|;|\||–|-", rest, maxsplit=1)[0].strip()
            if rest and len(rest) < 120:
                return rest

    # "... at CompanyName ..."
    m = re.search(r"\s+at\s+([A-Za-z0-9][A-Za-z0-9 &\-\.]{1,60}?)(?:\s|$|,|\.)", t)
    if m:
        return m.group(1).strip()

    # "... of CompanyName ..."
    m = re.search(r"\s+of\s+([A-Za-z0-9][A-Za-z0-9 &\-\.]{1,60}?)(?:\s|$|,|\.)", t)
    if m:
        return m.group(1).strip()

    # "... joins CompanyName ..."
    m = re.search(r"\s+joins\s+([A-Za-z0-9][A-Za-z0-9 &\-\.]{1,60}?)(?:\s|$|,|\.)", t)
    if m:
        return m.group(1).strip()

    # "... appointed to CompanyName board ..."
    m = re.search(r"appointed\s+to\s+([A-Za-z0-9][A-Za-z0-9 &\-\.]{1,60}?)\s+board", t, re.I)
    if m:
        return m.group(1).strip()

    # "CompanyName raises ..."
    m = re.match(r"^([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)?)\s+raises\b", t)
    if m:
        return m.group(1).strip()

    return "Unknown"


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

    Keeps items that classify to a core type or 'Other'. Missing person_name / role
    does not drop a row - those fields may stay empty.
    """
    rows: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for entry in entries:
        title = (getattr(entry, "title", None) or "").strip()
        if not title:
            continue

        event_type = _classify_event_type(title)
        if not event_type:
            continue

        link = (getattr(entry, "link", None) or "").strip()
        if not link:
            continue
        if link in seen_urls:
            continue
        seen_urls.add(link)

        summary = _strip_html(getattr(entry, "summary", None) or getattr(entry, "description", None) or "")
        full_explanation = summary or f"(No summary in feed.) Headline: {title}"

        person = _guess_person_name(title)
        company = _guess_company_name(title)
        role = _guess_role(title)

        rows.append(
            {
                "person_name": person,
                "company_name": company,
                "event_type": event_type,
                "raw_title": title,
                "role": role,
                "event_date": _parse_entry_date(entry),
                "detected_at": datetime.now(timezone.utc),
                "why_it_matters": why_it_matters_for_event_type(event_type),
                "source_url": link,
                "full_explanation": full_explanation[:4000],
                "quality_score": 0,
                "confidence_score": 0,
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
            "company_name": "Meridian Bank",
            "event_type": "Board Appointment",
            "raw_title": "Meridian Bank adds Taylor Kim as independent director to board",
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
            "company_name": "Lumen Grid",
            "event_type": "Founder Exit",
            "raw_title": "Lumen Grid founder Morgan Chen exits after strategic acquisition",
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
            "company_name": "Harbor Freight AI",
            "event_type": "Promotion",
            "raw_title": "Harbor Freight AI promotes Casey Nguyen to VP of Engineering",
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
    ]


def load_sample_signals() -> pd.DataFrame:
    """Load demo signals, score them, then normalize for the app."""
    raw_list = _raw_sample_signals()
    rows = apply_scores(raw_list)
    df = pd.DataFrame(rows)
    out = finalize_dataframe(df)
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
        ingest_debug["rows_after_finalize"] = len(out)
        out.attrs["ingest_debug"] = ingest_debug
        return out

    scored = apply_scores(raw_rows)
    df = pd.DataFrame(scored)
    out = finalize_dataframe(df)
    ingest_debug["rows_after_finalize"] = len(out)
    out.attrs["ingest_debug"] = ingest_debug
    return out
