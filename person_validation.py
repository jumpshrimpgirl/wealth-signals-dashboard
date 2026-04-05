"""
Shared person-name validation for data extraction and Streamlit UI.

Kept in one module so ``data`` and ``app`` can import without circular dependencies.
"""

from __future__ import annotations

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
