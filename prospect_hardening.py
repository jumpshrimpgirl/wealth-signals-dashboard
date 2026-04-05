"""
Strict validation, company/role sanitization, and dead/historical handling for the prospect pipeline.

Keeps broad recall: invalid *candidates* are skipped; articles remain in the corpus when other
candidates or heuristics still apply.
"""

from __future__ import annotations

import re
from typing import Any

from person_validation import is_valid_person_entity

# --- Known false positives / deceased (never Home; match_score forced to 0 upstream) -----------

KNOWN_DECEASED_OR_HISTORICAL_NAMES = frozenset(
    {
        "steve jobs",
        "james r. schlesinger",
        "james schlesinger",
    }
)

_LIST_OR_INDEX_PHRASES = (
    "list of",
    "category:",
    "index of",
    "timeline of",
    "people of",
)

_CASE_OR_EVENT_TERMS = (
    "murder case",
    "the naina",
    " trial",
    " investigation",
    " scandal",
    " lawsuit",
)

_NONHUMAN_ENTITY_PHRASES = (
    "middle east",
    "technology business",
    "department of",
    "agency",
    "ministry",
    "university",
    "european union",
)

_ORG_SUFFIX_PATTERN = re.compile(
    r"\b(inc\.?|llc|ltd\.?|group|capital|partners|corp\.?|corporation)\b",
    re.I,
)

_BAD_SINGLE_TOKEN_COMPANIES = frozenset(
    {
        "santa",
    }
)

_UNIVERSITY_OR_GOVT = re.compile(
    r"\b(university|college|school|department|ministry|agency|commission|parliament)\b",
    re.I,
)


def is_valid_person_name(name: str, article_text: str = "") -> bool:
    """
    Reject obvious non-person strings; defer shape checks to ``is_valid_person_entity``.
    Article text is used only for optional future proximity checks; hard rules apply to ``name``.
    """
    n = (name or "").strip()
    if not n:
        return False
    low = n.lower()
    art = (article_text or "").lower()
    for p in _LIST_OR_INDEX_PHRASES:
        if p in low:
            return False
    for p in _CASE_OR_EVENT_TERMS:
        if p in low:
            return False
    for p in _NONHUMAN_ENTITY_PHRASES:
        if p in low:
            return False
    if _ORG_SUFFIX_PATTERN.search(n):
        return False
    if re.search(r"\d", n):
        return False
    if any(sym in n for sym in "@#$%^&*"):
        return False
    words = n.split()
    if len(words) == 1:
        return False
    if low in ("santa", "list of punjabi people", "technology business", "middle east"):
        return False
    # Plural / category / list artifacts
    if re.search(r"\b(people|americans|investors|founders|executives|billionaires)\s*$", low):
        return False
    if "list of" in low or low.startswith("list of"):
        return False
    # Company / product / case-like (heuristic)
    if re.search(r"\b(ltd|llc|inc\.?|plc|corp)\s*$", low):
        return False
    if low.endswith(".com") or ".co/" in low:
        return False
    # Name equals article headline company phrase (common NER error)
    if art and len(n) > 12 and n.lower() in art and any(
        x in art for x in (" inc", " llc", "company", "technologies", "ventures")
    ):
        return False
    return is_valid_person_entity(n, "")


def _wiki_snippet_says_deceased(extract: str) -> bool:
    snip = (extract or "")[:1200].lower()
    if "born" in snip and "died" in snip:
        return True
    if re.search(r"\bdied\s+\d{4}\b", snip):
        return True
    if "obituary" in snip:
        return True
    return False


def _name_near_any(name: str, article_text: str, needles: tuple[str, ...]) -> bool:
    """Rough proximity: person's first/last token appears in a window around a needle."""
    text = article_text or ""
    if not name.strip() or not text.strip():
        return False
    low = text.lower()
    parts = [p for p in re.split(r"\s+", name.strip()) if len(p) > 2]
    if not parts:
        parts = [name.strip().lower()]
    for nd in needles:
        pos = 0
        while True:
            i = low.find(nd, pos)
            if i < 0:
                break
            win = low[max(0, i - 140) : i + len(nd) + 140]
            if any(p.lower() in win for p in parts):
                return True
            pos = i + 1
    return False


def is_historical_or_dead(
    name: str,
    article_text: str,
    cross_check_result: dict[str, Any] | None,
) -> bool:
    """True → caller should set match_score=0, cap priority, block Home."""
    n = (name or "").strip().lower()
    if n in KNOWN_DECEASED_OR_HISTORICAL_NAMES:
        return True
    cr = cross_check_result or {}
    if cr.get("deceased") is True:
        return True
    if cr.get("historical_only") is True:
        return True
    en = cr.get("_enrichment")
    if isinstance(en, dict):
        if en.get("wiki_bio_deceased") is True:
            return True
        ex = str(en.get("_wikipedia_extract") or "")
        if _wiki_snippet_says_deceased(ex):
            return True
    # Strong article cues tied to this name (avoid flagging everyone in a long obituary page)
    if _name_near_any(
        name,
        article_text,
        ("obituary", "passed away", "former secretary", "appointed by carter"),
    ):
        return True
    if _name_near_any(name, article_text, ("died",)) and _name_near_any(
        name, article_text, ("born", "age ")
    ):
        return True
    return False


def sanitize_company(company: str, article_text: str) -> tuple[str, bool]:
    """
    Returns (company, invalid_company_penalty_applied).
    Invalid / non-business → ``Unknown`` and penalty flag True.
    """
    c0 = (company or "").strip()
    art = article_text or ""
    low = c0.lower()

    if not c0 or low in ("unknown", "n/a", "none"):
        return "Unknown", True

    if len(c0.split()) == 1 and low in _BAD_SINGLE_TOKEN_COMPANIES:
        return "Unknown", True

    if _UNIVERSITY_OR_GOVT.search(c0):
        return "Unknown", True

    # Fix truncated "Santa" when article names a full institution
    if low == "santa" or (low == "santa clara" and "university" in art.lower()):
        if "santa clara university" in art.lower():
            return "Unknown", True
        return "Unknown", True

    # Prefer full company phrase when article clearly names it (e.g. Vimana Private Jets)
    m = re.search(
        r"\b(Vimana\s+Private\s+Jets)\b",
        art,
        re.I,
    )
    if m and ("vimana" in low or len(low) < 8):
        return m.group(1).strip(), False

    m2 = re.search(r"\b(MiningLamp\s+Technology)\b", art, re.I)
    if m2 and "mining" in low:
        return m2.group(1).strip(), False

    if _ORG_SUFFIX_PATTERN.search(c0) and len(c0.split()) <= 2:
        return c0, False

    return c0, False


def _article_supports_us_president(name: str, article_text: str) -> bool:
    t = (article_text or "").lower()
    n = (name or "").lower()
    if "president of the united states" in t or "u.s. president" in t or "us president" in t:
        return True
    if "president trump" in t or "president biden" in t:
        return n.split()[0] in ("donald", "joe", "joseph")
    return False


def sanitize_role(
    name: str,
    role: str,
    article_text: str,
    cross_check_result: dict[str, Any] | None,
) -> tuple[str, bool]:
    """
    Returns (role, was_sanitized).
    Never assign misleading high offices; prefer article + external role when contradictory.
    """
    r0 = (role or "").strip()
    art = (article_text or "").lower()
    nl = (name or "").strip().lower()
    cr = cross_check_result or {}
    canon = str(cr.get("canonical_role") or "").strip()

    rl = r0.lower()
    out = r0
    changed = False

    # Pam Bondi → never "President" without US President context
    if "bondi" in nl:
        if "president" in rl and not _article_supports_us_president(name or "", article_text or ""):
            if "attorney general" in art:
                out = "Attorney General"
                changed = True
            elif canon and "president" not in canon.lower():
                out = canon
                changed = True
            else:
                out = "Attorney General"
                changed = True

    # Kevin Osborne legal-shield / CNBC counsel — not "founder"
    if "osborne" in nl and "kevin" in nl:
        founder_like = "founder" in rl or "co-founder" in rl
        legal_ctx = any(
            x in art
            for x in (
                "legal shield",
                "attorney",
                "counsel",
                "lawsuit",
                "plaintiff",
                "defendant",
                "cnbc",
            )
        )
        weak_founder_evidence = not any(
            x in art for x in ("founded", "co-founded", "startup", "ceo of", "launch")
        )
        if founder_like and legal_ctx and weak_founder_evidence:
            if "attorney" in art or "counsel" in art:
                out = "Attorney"
            else:
                out = "Legal counsel"
            changed = True

    # Commentator / analyst without ownership — prefer canonical title when it is executive
    if "president" in rl and not _article_supports_us_president(name or "", article_text or ""):
        if any(x in art for x in ("commentator", "analyst", "told cnbc", "said in an interview")):
            if canon and "president" not in canon.lower():
                out = canon
                changed = True

    if canon and not changed:
        c_low = canon.lower()
        # Prefer external role when it contradicts a bogus President label
        if "president" in rl and "president" not in c_low and c_low:
            if "chief executive" in c_low or "ceo" in c_low:
                out = canon
                changed = True

    return (out if out else r0), changed


def sanitize_role_and_company(
    candidate: dict[str, Any],
    article_text: str,
    cross_check_result: dict[str, Any],
) -> tuple[str, str, bool]:
    """
    Combined role + company fixes: Pam Bondi, Kevin Osborne, Vimana, Wu Minghui, Sam Altman / OpenAI.
    Returns (role, company, any_change).
    """
    name = str(candidate.get("name") or "").strip()
    role0 = str(candidate.get("role") or "").strip()
    co0 = str(cross_check_result.get("canonical_company") or candidate.get("company") or "").strip()
    art = article_text or ""
    art_l = art.lower()

    co_san, _ = sanitize_company(co0, art)
    role_san, ch_r = sanitize_role(name, str(cross_check_result.get("canonical_role") or role0), art, cross_check_result)
    changed = ch_r

    nl = name.lower()
    # Vimana: person is not the company string "Vimana Private"
    if "vimana" in co_san.lower() and "naran" in nl:
        m = re.search(r"\b(Vimana\s+Private\s+Jets)\b", art, re.I)
        if m:
            co_san = m.group(1).strip()
            changed = True

    # Wu Minghui + MiningLamp
    if "minghui" in nl or "wu minghui" in nl:
        m2 = re.search(r"\b(MiningLamp\s+Technology)\b", art, re.I)
        if m2 and ("mining" in co_san.lower() or len(co_san) < 6):
            co_san = m2.group(1).strip()
            changed = True
        if "founder" in art_l and "ceo" in role_san.lower() and "mining" in art_l:
            if not role_san or role_san.lower() in ("other", "executive"):
                role_san = "Founder & CEO"
                changed = True

    # Sam Altman — prefer OpenAI when article anchors company
    if "altman" in nl and "openai" in art_l:
        if "openai" not in co_san.lower() and "openai" in art_l:
            co_san = "OpenAI"
            changed = True

    return role_san, co_san, changed
