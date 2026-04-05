"""
Multi-layer source parsing: metadata + visible content + normalization.

Produces ``normalized_article`` for prospect extraction — the parser does not guess people;
downstream AI + resolution layers assign entities.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

# Match ``data`` fetch settings without importing ``data`` (avoids circular imports).
HTML_REQUEST_HEADERS = {
    "User-Agent": "WealthSignalsDashboard/1.0 (+https://example.local; educational project)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
REQUEST_TIMEOUT_SEC = 20
MAX_ARTICLE_TEXT_CHARS = 200_000

_MONEY = re.compile(
    r"\$\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|bn|m\b|b\b|k\b)?|\b\d+(?:\.\d+)?\s*(?:billion|million)\b",
    re.I,
)
_DATE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b",
    re.I,
)
_ORG_HINT = re.compile(
    r"\b([A-Z][A-Za-z0-9&\-.]+(?:\s+[A-Z][A-Za-z0-9&\-.]+){0,4})\s+(?:CEO|founder|Inc\.|LLC|Ltd\.|Corp\.)\b",
)


def _canonical_url(resp_url: str, soup: Any) -> str:
    u = (resp_url or "").strip()
    try:
        from bs4 import BeautifulSoup

        if soup:
            link = soup.find("link", rel=lambda x: x and "canonical" in str(x).lower())
            if link and link.get("href"):
                href = str(link["href"]).strip()
                if href.startswith("http"):
                    return href
    except Exception:
        pass
    return u


def _meta_content(soup: Any, prop: str, *, name: str | None = None) -> str:
    try:
        from bs4 import BeautifulSoup

        if name:
            tag = soup.find("meta", attrs={"name": name})
        else:
            tag = soup.find("meta", property=prop)
        if tag and tag.get("content"):
            return str(tag["content"]).strip()
    except Exception:
        pass
    return ""


def _json_ld_blocks(soup: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        for script in soup.find_all("script", type="application/ld+json"):
            raw = script.string or script.get_text() or ""
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                out.append(data)
            elif isinstance(data, list):
                for x in data:
                    if isinstance(x, dict):
                        out.append(x)
    except Exception:
        pass
    return out


def _ld_article_fields(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Pull headline, datePublished, author from schema.org NewsArticle / Article."""
    headline = ""
    date_pub = ""
    authors: list[str] = []
    section = ""

    for b in blocks:
        typ = b.get("@type")
        types = typ if isinstance(typ, list) else [typ] if typ else []
        types_l = [str(t).lower() for t in types if t]
        if not any(t in ("newsarticle", "article", "blogposting") for t in types_l):
            if "headline" not in b and "datePublished" not in b:
                continue
        headline = headline or str(b.get("headline") or "").strip()
        date_pub = date_pub or str(b.get("datePublished") or b.get("dateModified") or "").strip()
        sec = b.get("articleSection")
        if sec:
            section = str(sec).strip()
        auth = b.get("author")
        if isinstance(auth, dict):
            authors.append(str(auth.get("name") or "").strip())
        elif isinstance(auth, list):
            for a in auth:
                if isinstance(a, dict):
                    n = str(a.get("name") or "").strip()
                    if n:
                        authors.append(n)
                elif isinstance(a, str) and a.strip():
                    authors.append(a.strip())
        elif isinstance(auth, str) and auth.strip():
            authors.append(auth.strip())
    return {
        "headline": headline,
        "datePublished": date_pub,
        "authors": [a for a in authors if a],
        "section": section,
    }


def _clean_paragraphs(soup: Any) -> list[str]:
    from bs4 import BeautifulSoup

    root = soup.find("article") or soup.find("main") or soup
    paragraphs = root.find_all("p")
    parts: list[str] = []
    for p in paragraphs:
        t = p.get_text(separator=" ", strip=True)
        if t and len(t) > 40:
            parts.append(t)
    if not parts:
        for p in soup.find_all("p"):
            t = p.get_text(separator=" ", strip=True)
            if t:
                parts.append(t)
    return parts


def _money_mentions(text: str, limit: int = 40) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _MONEY.finditer(text or ""):
        s = m.group(0).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= limit:
            break
    return out


def _date_mentions(text: str, limit: int = 20) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _DATE.finditer(text or ""):
        s = m.group(0).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= limit:
            break
    return out


def _org_mentions(text: str, limit: int = 30) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _ORG_HINT.finditer(text or ""):
        s = m.group(1).strip()
        if len(s) > 2 and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= limit:
            break
    return out


def _byline_people(text: str) -> list[str]:
    """Lightweight 'By First Last' / 'By A. B. Lastname' hints."""
    out: list[str] = []
    for m in re.finditer(
        r"(?im)^\s*(?:by|written by)\s+([A-Z][a-zA-Z\.\-]+(?:\s+[A-Z][a-zA-Z\.\-]+){1,3})\s*$",
        (text or "")[:4000],
    ):
        out.append(m.group(1).strip())
    return list(dict.fromkeys(out))[:8]


def parse_article_source(
    feed_row: dict[str, Any],
    *,
    session: Any | None = None,
) -> dict[str, Any]:
    """
    LAYER A–C: Fetch URL, extract metadata + body + normalize.

    ``feed_row`` must include ``url`` (or ``link``). Optional: ``feed_title``, ``feed_summary``,
    ``published_at`` from RSS.

    Returns ``normalized_article`` dict (always; on failure, minimal fields from feed only).
    """
    url = str(feed_row.get("url") or feed_row.get("link") or "").strip()
    feed_title = str(feed_row.get("feed_title") or feed_row.get("raw_title") or "").strip()
    feed_summary = str(feed_row.get("feed_summary") or feed_row.get("summary") or "").strip()
    rss_date = feed_row.get("published_at") or feed_row.get("event_date") or feed_row.get("detected_at")

    empty: dict[str, Any] = {
        "url": url,
        "canonical_url": url,
        "source_domain": urlparse(url).netloc.lower() if url else "",
        "title": feed_title,
        "summary": feed_summary,
        "article_text": "",
        "first_paragraphs": "",
        "published_at": _iso_or_empty(rss_date),
        "section": "",
        "authors": [],
        "money_mentions": [],
        "date_mentions": [],
        "org_mentions": [],
        "person_mentions": [],
        "og_title": "",
        "og_description": "",
        "_parse_ok": False,
    }

    if not url:
        return empty

    import requests

    sess = session or requests
    try:
        resp = sess.get(
            str(url).strip(),
            headers=HTML_REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT_SEC,
            allow_redirects=True,
        )
        resp.raise_for_status()
    except Exception:
        merged = dict(empty)
        merged["article_text"] = _blend_text(feed_title, feed_summary, "")
        merged["summary"] = merged["summary"] or feed_summary
        return merged

    final_url = str(resp.url or url).strip()
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(resp.content, "html.parser")
    except Exception:
        merged = dict(empty)
        merged["canonical_url"] = final_url
        merged["article_text"] = _blend_text(feed_title, feed_summary, "")
        return merged

    canonical = _canonical_url(final_url, soup)
    og_title = _meta_content(soup, "og:title") or _meta_content(soup, "", name="twitter:title")
    og_desc = _meta_content(soup, "og:description") or _meta_content(soup, "", name="description")
    art_time = _meta_content(soup, "article:published_time") or _meta_content(
        soup, "", name="article:published_time"
    )
    author_meta = _meta_content(soup, "", name="author") or _meta_content(soup, "", name="sailthru.author")

    ld_blocks = _json_ld_blocks(soup)
    ld = _ld_article_fields(ld_blocks)

    title_tag = ""
    if soup.title and soup.title.string:
        title_tag = str(soup.title.string).strip()

    title = og_title or ld.get("headline") or title_tag or feed_title
    paragraphs = _clean_paragraphs(soup)
    body = " ".join(paragraphs).strip()
    if len(body) > MAX_ARTICLE_TEXT_CHARS:
        body = body[:MAX_ARTICLE_TEXT_CHARS]

    first_three = "\n\n".join(paragraphs[:3]) if paragraphs else ""

    published = art_time or ld.get("datePublished") or ""
    if not published:
        published = _iso_or_empty(rss_date)
    else:
        published = _normalize_iso_date(published) or _iso_or_empty(rss_date)

    authors: list[str] = []
    if author_meta:
        authors.append(author_meta)
    authors.extend(ld.get("authors") or [])
    authors = list(dict.fromkeys([a for a in authors if a]))[:12]

    blob = f"{title}\n{og_desc}\n{feed_summary}\n{body}"
    money = _money_mentions(blob)
    dates = _date_mentions(blob)
    orgs = _org_mentions(body[:20000])
    people = _byline_people(body[:8000])
    # Light capitalized sequence near role words (not a person guess — hints only)
    for m in re.finditer(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s+(?:CEO|founder|CFO|president)\b",
        body[:15000],
    ):
        people.append(m.group(1).strip())
    people = list(dict.fromkeys([p for p in people if p]))[:20]

    summary = _blend_summary(feed_summary, og_desc, first_three, title)

    domain = urlparse(canonical or final_url).netloc.lower()

    return {
        "url": url,
        "canonical_url": canonical or final_url,
        "source_domain": domain,
        "title": title or feed_title,
        "summary": summary,
        "article_text": body,
        "first_paragraphs": first_three,
        "published_at": published,
        "section": ld.get("section") or "",
        "authors": authors,
        "money_mentions": money,
        "date_mentions": dates,
        "org_mentions": orgs,
        "person_mentions": people,
        "og_title": og_title,
        "og_description": og_desc,
        "_parse_ok": bool(body),
    }


def _blend_text(title: str, summary: str, body: str) -> str:
    parts = [p for p in (title, summary, body) if str(p).strip()]
    return " ".join(parts).strip()


def _blend_summary(feed_summary: str, og_desc: str, first_paragraphs: str, title: str) -> str:
    if og_desc and len(og_desc) > 80:
        base = og_desc
    elif feed_summary and len(feed_summary) > 40:
        base = feed_summary
    elif first_paragraphs:
        base = first_paragraphs[:500]
    else:
        base = title
    return re.sub(r"\s+", " ", str(base)).strip()[:2000]


def _iso_or_empty(v: Any) -> str:
    if v is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    if not s:
        return ""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return s


def _normalize_iso_date(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return s


def minimal_normalized_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """When RSS/HTML parse was not run (sample rows), build a minimal normalized_article."""
    title = str(row.get("raw_title") or row.get("source_title") or "").strip()
    summary = str(row.get("summary") or "").strip()
    url = str(row.get("source_url") or "").strip()
    body = str(row.get("full_explanation") or row.get("article_text") or "").strip()
    if not body:
        body = str(row.get("ai_summary") or "").strip()
    blob = _blend_text(title, summary, body)
    domain = urlparse(url).netloc.lower() if url else ""
    return {
        "url": url,
        "canonical_url": url,
        "source_domain": domain,
        "title": title,
        "summary": summary or blob[:1200],
        "article_text": body or blob,
        "first_paragraphs": "\n\n".join(body.split("\n\n")[:3]) if body else "",
        "published_at": _iso_or_empty(row.get("published_at") or row.get("event_date") or row.get("detected_at")),
        "section": "",
        "authors": [],
        "money_mentions": _money_mentions(blob),
        "date_mentions": _date_mentions(blob),
        "org_mentions": [],
        "person_mentions": [],
        "og_title": "",
        "og_description": "",
        "_parse_ok": bool(body),
    }


__all__ = ["minimal_normalized_from_row", "parse_article_source"]
