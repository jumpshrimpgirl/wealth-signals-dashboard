"""
Data layer: sample signals (fallback) and live RSS-based signals.

Uses public RSS feeds only - no LinkedIn or private sources.
"""

from __future__ import annotations

import re
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


def outreach_angle_for_event_type(event_type: str) -> str:
    """
    One-line conversation starter tailored to the event type (for demos / outreach planning).
    """
    et = (event_type or "").strip()
    angles = {
        "Founder Exit": "Congrats on exit - discuss liquidity and tax strategy",
        "Funding": "Congrats on raise - planning for future liquidity",
        "Promotion": "New role -> comp and tax optimization",
        "Board Appointment": "New board role -> expanding financial complexity",
        "Other": "Broad finance or career headline - lead with context and curiosity",
    }
    return angles.get(et, "Acknowledge the news - offer relevant planning context.")


def generate_outreach_angle(row) -> str:
    """
    Generate context-aware outreach angle based on available fields.
    Varies by event_type and uses person, company, role when available.
    """
    et = (row["event_type"] or "").strip()
    person = (row["person_name"] or "").strip()
    company = (row["company_name"] or "").strip()
    role = (row["role"] or "").strip()

    has_person = bool(person)
    has_company = bool(company) and company != "Unknown"
    has_role = bool(role)

    if et == "Founder Exit":
        if has_person and has_company and has_role:
            return f"Discuss liquidity options with {person}, {role} at {company}, following their recent exit."
        elif has_person and has_company:
            return f"Reach out to {person} from {company} about post-exit financial planning."
        elif has_company:
            return f"Explore tax strategies for {company}'s recent acquisition or exit."
        else:
            return "Discuss founder exit opportunities and wealth management."

    elif et == "Funding":
        if has_person and has_company and has_role:
            return f"Connect with {person}, {role} at {company}, on their funding success and future growth."
        elif has_person and has_company:
            return f"Congratulate {person} from {company} on the raise and discuss equity planning."
        elif has_company:
            return f"Discuss funding implications for {company}'s valuation and team equity."
        else:
            return "Explore startup funding strategies and liquidity events."

    elif et == "Promotion":
        if has_person and has_company and has_role:
            return f"Reach out to {person} on their {role} promotion at {company}."
        elif has_person and has_company:
            return f"Congratulate {person} from {company} on their career advancement."
        elif has_company:
            return f"Discuss compensation changes following promotions at {company}."
        else:
            return "Talk about career progression and executive compensation."

    elif et == "Board Appointment":
        if has_person and has_company and has_role:
            return f"Connect with {person} on their {role} appointment to {company}'s board."
        elif has_person and has_company:
            return f"Discuss board opportunities with {person} joining {company}."
        elif has_company:
            return f"Explore board compensation and governance at {company}."
        else:
            return "Discuss board roles and director compensation."

    else:  # Other or unknown
        if has_person and has_company:
            return f"Follow up with {person} from {company} on recent developments."
        elif has_company:
            return f"Discuss market updates related to {company}."
        else:
            return "Acknowledge the news - offer relevant planning context."


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
    - Drops duplicate source_url rows, keeping the highest score first
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

    # --- Quality score: ensure it's an int ---
    out["quality_score"] = out["quality_score"].fillna(0).astype(int)

    # --- Dedupe URLs: keep the row with the highest score, then first occurrence ---
    if not out.empty and "source_url" in out.columns:
        out = out.sort_values(["score", "event_date"], ascending=[False, False], na_position="last")
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

        quality_score = 0
        if person:
            quality_score += 1
        if company != "Unknown":
            quality_score += 1
        if role:
            quality_score += 1
        if event_type != "Other":
            quality_score += 1

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
                "quality_score": quality_score,
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
