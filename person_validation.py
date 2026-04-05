"""
Shared person-name validation for data extraction and Streamlit UI.

Kept in one module so ``data`` and ``app`` can import without circular dependencies.
"""

from __future__ import annotations

import json
import re

# Any token (case-insensitive) → reject (places, media, org cues, etc.)
PERSON_NAME_WORD_BLACKLIST = frozenset(
    {
        "force",
        "city",
        "county",
        "alabama",
        "boston",
        "air",
        "news",
        "daily",
        "startup",
        "company",
    }
)

NAME_CONNECTOR_WORDS = frozenset({"de", "van", "von", "da"})

COMMON_LOC_ORG_WORDS = frozenset(
    {
        "central",
        "north",
        "south",
        "east",
        "west",
        "northeast",
        "northwest",
        "southeast",
        "southwest",
        "metro",
        "national",
        "regional",
        "international",
        "university",
        "college",
        "school",
        "association",
        "foundation",
        "department",
        "group",
        "partners",
        "capital",
        "ventures",
        "global",
        "world",
        "area",
        "valley",
        "bay",
        "coast",
        "district",
        "state",
        "federal",
        "california",
        "texas",
        "florida",
        "york",
        "angeles",
        "francisco",
        "chicago",
        "atlanta",
        "philadelphia",
        "phoenix",
        "houston",
        "dallas",
        "seattle",
        "denver",
        "miami",
        "virginia",
        "carolina",
        "jersey",
        "island",
        "beach",
        "springs",
        "creek",
        "lake",
        "river",
        "mountain",
        "park",
        "center",
        "centre",
        "plaza",
        "systems",
        "services",
        "technologies",
        "solutions",
        "industries",
        "holdings",
        "management",
        "committee",
        "council",
        "government",
        "public",
        "affairs",
        "defense",
        "justice",
        "health",
        "education",
        "energy",
        "commerce",
        "security",
        "treasury",
        "san",
        "diego",
        "jose",
        "antonio",
        "oakland",
        "louis",
        "vegas",
        "memphis",
        "nashville",
        "columbus",
        "indianapolis",
        "milwaukee",
        "kansas",
        "tampa",
        "orlando",
        "charlotte",
        "detroit",
        "minneapolis",
        "portland",
        "baltimore",
        "medical",
        "general",
        "electric",
        "office",
        "director",
        "president",
        "minister",
        "senator",
        "representative",
        "private",
        "labor",
        "transportation",
        "agriculture",
        "interior",
        "terminal",
        "station",
        "airport",
        "harbor",
        "harbour",
        "square",
        "heights",
        "grove",
        "field",
        "plains",
        "desert",
        "forest",
        "hills",
    }
)


def _name_token_chars_ok(w: str) -> bool:
    """Letters (including accents), optional hyphens — no digits or other punctuation."""
    return bool(w) and all(c.isalpha() or c == "-" for c in w)


def is_valid_person(name: str) -> bool:
    """
    Hero-section filter: keep likely human full names; drop places, headlines, org phrases.

    Rules: 2–4 whitespace-separated tokens; hyphens and Unicode letters allowed; lowercase
    particles (de, van, von, da); blacklist; no leading 'The'; reject ALL-CAPS tokens; bonus
    loc/org phrase filter when every token matches COMMON_LOC_ORG_WORDS.
    """
    if not isinstance(name, str):
        return False

    words = name.strip().split()
    if len(words) < 2 or len(words) > 4:
        return False

    if words[0].lower() == "the":
        return False

    lowered = [w.lower() for w in words]

    if all(w in COMMON_LOC_ORG_WORDS for w in lowered):
        return False

    for w in words:
        wl = w.lower()
        if wl in PERSON_NAME_WORD_BLACKLIST:
            return False
        if w.isupper():
            return False
        if not _name_token_chars_ok(w):
            return False

        if wl in NAME_CONNECTOR_WORDS:
            if not w.isalpha() or w.isupper():
                return False
            continue

        if not w[0].isupper():
            return False
        if len(w) > 40:
            return False

    return True


# --- Reject generic offices / roles mistaken for names (whole-string match, normalized) ---

_ROLE_ONLY_PHRASES_LOWER = frozenset(
    {
        "attorney general",
        "prime minister",
        "vice president",
        "chief executive",
        "chief executive officer",
        "chief financial officer",
        "chief operating officer",
        "chief technology officer",
        "secretary of state",
        "secretary of the treasury",
        "house speaker",
        "senate majority leader",
        "cabinet minister",
        "police chief",
        "the president",
        "president",
        "billionaire investor",
        "pop star",
        "footballer",
        "soccer player",
        "chief justice",
        "foreign minister",
        "defense minister",
        "finance minister",
        "health secretary",
        "home secretary",
        "white house",
        "federal reserve chair",
        "fed chair",
    }
)

_ROLE_ONLY_SINGLE_WORDS_LOWER = frozenset(
    {
        "ceo",
        "cfo",
        "coo",
        "cto",
        "cio",
        "ciso",
        "founder",
        "co-founder",
        "cofounder",
        "chairman",
        "chairwoman",
        "chair",
        "president",
        "governor",
        "senator",
        "mayor",
        "judge",
        "director",
        "minister",
        "billionaire",
        "millionaire",
        "investor",
        "executive",
        "partner",
        "athlete",
        "celebrity",
        "footballer",
        "football",
        "basketball",
        "entrepreneur",
    }
)


def _strip_leading_article(s: str) -> str:
    t = str(s or "").strip()
    if t.lower().startswith("the "):
        return t[4:].strip()
    return t


def is_role_or_office_only_label(text: str) -> bool:
    """
    True when the entire string is a job title, office, or generic role — not a person's name.

    Used to reject values like "Attorney General" or "CEO" from the Name field.
    """
    raw = _strip_leading_article(str(text or "").strip())
    if not raw:
        return False
    lower = raw.lower()
    if lower in _ROLE_ONLY_PHRASES_LOWER:
        return True
    words = lower.split()
    if len(words) == 1 and words[0] in _ROLE_ONLY_SINGLE_WORDS_LOWER:
        return True
    if len(words) == 2:
        pair = " ".join(words)
        if pair in _ROLE_ONLY_PHRASES_LOWER:
            return True
        # "General Attorney" unlikely; "Attorney General" already in phrases
    return False


# Longest-first title prefixes for "Title Firstname Lastname" at start of headline or candidate string
# Title keywords are case-insensitive; **name tokens must be strict Title Case** so verbs like
# "speaks" are not swallowed under ``re.IGNORECASE`` matching ``[A-Z]`` to ``s``.
_TITLE_NAME_SPLIT = [
    re.compile(
        r"(?is)^(?:\s*(?:the\s+)?)?"
        r"(attorney\s+general|prime\s+minister|vice\s+president|"
        r"chief\s+executive(?:\s+officer)?|chief\s+financial(?:\s+officer)?|chief\s+operating(?:\s+officer)?|"
        r"secretary\s+of\s+state|secretary\s+of\s+the\s+treasury|house\s+speaker|"
        r"cabinet\s+minister|police\s+chief|federal\s+reserve\s+chair|fed\s+chair)\s+"
        r"((?-i:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}))(?=\s|$|[,.\-|–—:])"
    ),
    re.compile(
        r"(?is)^(?:\s*(?:the\s+)?)?(CEO|CFO|COO|CTO|CIO|CISO)\s+"
        r"((?-i:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}))(?=\s|$|[,.\-|–—:])"
    ),
    re.compile(
        r"(?is)^(?:\s*(?:the\s+)?)?(President|Governor|Senator|Mayor|Judge|Founder|Co-Founder|Chairman|Chairwoman)\s+"
        r"((?-i:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}))(?=\s|$|[,.\-|–—:])"
    ),
]


def split_title_and_person_name(text: str) -> tuple[str, str]:
    """
    If ``text`` begins with a known title/office + person name, return (person_name, title).

    Otherwise ("", "").
    """
    s = str(text or "").strip()
    if not s:
        return "", ""
    for rx in _TITLE_NAME_SPLIT:
        m = rx.match(s)
        if m:
            title_part = m.group(1).strip()
            name_part = m.group(2).strip()
            if name_part and is_valid_person(name_part):
                # Normalize title casing lightly
                tl = title_part
                if tl.isupper() and len(tl) <= 5:
                    pass
                else:
                    tl = title_part.title() if title_part.islower() else title_part
                return name_part, tl
    return "", ""


def _merge_roles(a: str, b: str) -> str:
    a, b = (a or "").strip(), (b or "").strip()
    if not a:
        return b
    if not b:
        return a
    if a.lower() == b.lower():
        return a
    return f"{a}; {b}"


def sanitize_person_name_and_role(
    person_name: str,
    role: str,
    raw_title: str,
) -> tuple[str, str, str]:
    """
    Clean entity extraction: never leave offices/titles in the Name field.

    Returns ``(person_name_out, role_out, validation_note)`` where ``validation_note`` is one of:
    ``ok``, ``empty``, ``rejected_role_only``, ``fixed_split_title``, ``rejected_invalid``.
    """
    pn = str(person_name or "").strip()
    rl = str(role or "").strip()
    title = str(raw_title or "").strip()

    # 1) Headline first — often has "Attorney General Pam Bondi …" even when the person field is wrong
    if title:
        name_from_title, title_from_title = split_title_and_person_name(title)
        if name_from_title and is_valid_person(name_from_title):
            if not pn or is_role_or_office_only_label(pn) or not is_valid_person(pn):
                return name_from_title, _merge_roles(title_from_title, rl), "fixed_split_title"

    # 2) Combined "Title Name" stored in person field
    if pn:
        name_from_pn, title_from_pn = split_title_and_person_name(pn)
        if name_from_pn:
            return name_from_pn, _merge_roles(title_from_pn, rl), "fixed_split_title"

    # 3) Whole field is a role/office only → clear name, move to role
    if pn and is_role_or_office_only_label(pn):
        return "", _merge_roles(pn, rl), "rejected_role_only"

    # 4) Already a valid human name
    if pn and is_valid_person(pn):
        return pn, rl, "ok"

    # 5) Leftover non-empty junk
    if pn:
        return "", rl, "rejected_invalid"

    return "", rl, "empty"


def filter_valid_additional_people(names: list[str]) -> list[str]:
    """Drop role-only strings and invalid shapes from ``additional_people``."""
    out: list[str] = []
    seen: set[str] = set()
    for x in names:
        s = str(x or "").strip()
        if not s or is_role_or_office_only_label(s):
            continue
        if not is_valid_person(s):
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def prospect_anchor_name_for_fa(person_name: str, additional_people: object, raw_title: str) -> tuple[bool, str]:
    """
    True only when we have a **validated** real-person name for prospecting (not a title).

    Returns (identified, display_name_or_empty).
    """
    pn, _, _tag = sanitize_person_name_and_role(str(person_name or ""), "", str(raw_title or ""))
    if pn and is_valid_person(pn):
        return True, pn

    ap_raw = additional_people
    if isinstance(ap_raw, str) and ap_raw.strip().startswith("["):
        try:
            ap_raw = json.loads(ap_raw)
        except json.JSONDecodeError:
            ap_raw = []
    if isinstance(ap_raw, list):
        for x in ap_raw:
            s = str(x or "").strip()
            if s and not is_role_or_office_only_label(s) and is_valid_person(s):
                return True, s

    t = str(raw_title or "").strip()
    if t:
        n2, _ = split_title_and_person_name(t)
        if n2 and is_valid_person(n2):
            return True, n2
        # "Jane Doe - CEO …"
        if " - " in t or " – " in t or " — " in t:
            lead = t.split(" - ")[0].split(" – ")[0].split(" — ")[0].strip()
            if lead and not is_role_or_office_only_label(lead) and is_valid_person(lead):
                return True, lead

    return False, ""
