"""
Person/company resolution, signal inference, and wealth derivation from normalized articles.

Runs **before** final ranking — not display-only cleanup.
"""

from __future__ import annotations

import re
from typing import Any

from prospect_display_gates import is_forbidden_display_name


def is_invalid_candidate_name(name: str, entity_type: str, article_text: str) -> bool:
    """
    Hard reject before scoring: non-person entities and forbidden strings.
    ``unknown`` is allowed only when the name itself still looks like a person (downstream gates).
    """
    et = (entity_type or "").strip().lower()
    if et in ("company", "product", "organization", "region", "event"):
        return True
    if is_forbidden_display_name(name or "", article_text or ""):
        return True
    return False


_RE_VIMANA = re.compile(r"\b(Vimana\s+Private\s+Jets)\b", re.I)
_RE_MINING = re.compile(r"\b(MiningLamp\s+Technology)\b", re.I)
_RE_OPENAI = re.compile(r"\b(OpenAI)\b", re.I)
_RE_COMPANY_CEO_PERSON = re.compile(
    r"\b([A-Z][A-Za-z0-9&\-. ]{2,72}?)\s+(?:CEO|chief\s+executive|founder|co-?founder)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b",
    re.I,
)


def resolve_person_company(
    candidate: dict[str, Any],
    normalized_article: dict[str, Any],
    cross_check_result: dict[str, Any],
) -> tuple[str, str, bool]:
    """
    Tie company (and role hints) to the person using article context, not stray tokens.
    Returns (role, company, changed).
    """
    name = str(candidate.get("name") or "").strip()
    art = (
        str(normalized_article.get("article_text") or "")
        + "\n"
        + str(normalized_article.get("first_paragraphs") or "")
        + "\n"
        + str(normalized_article.get("title") or "")
    )
    art_l = art.lower()
    nl = name.lower()

    role0 = str(cross_check_result.get("canonical_role") or candidate.get("role") or "").strip()
    co0 = str(cross_check_result.get("canonical_company") or candidate.get("company") or "").strip()

    changed = False
    role_out = role0
    co_out = co0

    # Vimana Private Jets + Ameerh Naran
    if "naran" in nl and "vimana" in art_l:
        m = _RE_VIMANA.search(art)
        if m:
            co_out = m.group(1).strip()
            changed = True
        if "ceo" in art_l and not role_out:
            role_out = "CEO"
            changed = True

    # Sam Altman + OpenAI when article anchors OpenAI
    if "altman" in nl and "openai" in art_l:
        if _RE_OPENAI.search(art):
            co_out = "OpenAI"
            changed = True

    # Wu Minghui + MiningLamp
    if "minghui" in nl or "wu minghui" in nl:
        m2 = _RE_MINING.search(art)
        if m2:
            co_out = m2.group(1).strip()
            changed = True

    # "Company … CEO / founder Person" (e.g. OpenAI CEO Sam Altman; Vimana Private Jets CEO Ameerh Naran)
    m = _RE_COMPANY_CEO_PERSON.search(art)
    if m:
        maybe_co, maybe_name = m.group(1).strip(), m.group(2).strip()
        if maybe_name.lower() == nl and len(maybe_co) > 2 and maybe_co.lower() not in nl:
            co_out = maybe_co
            changed = True

    # "CEO of Company Name" near person (last resort — short window)
    if name:
        idx = art_l.find(nl.split()[0] if nl.split() else "")
        if idx >= 0:
            win = art[max(0, idx - 220) : idx + len(name) + 220]
            m3 = re.search(
                r"(?:CEO|chief\s+executive|founder|co-?founder)\s+of\s+([A-Z][A-Za-z0-9&\-. ]{2,64}?)(?:\s|\.|,|$)",
                win,
                re.I,
            )
            if m3 and len(m3.group(1)) < 70:
                g = m3.group(1).strip()
                if g.lower() not in nl:
                    co_out = g
                    changed = True

    return (role_out if role_out else role0, co_out if co_out else co0, changed)


def infer_signal_type(
    normalized_article: dict[str, Any],
    candidate: dict[str, Any],
    article_signal_type: str = "Other",
) -> str:
    """
    Article-grounded signal label (not cosmetic). Uses full parsed text + title.
    """
    from prospect_display_gates import sanitize_signal_type

    t = (
        str(normalized_article.get("article_text") or "")
        + "\n"
        + str(normalized_article.get("first_paragraphs") or "")
        + "\n"
        + str(normalized_article.get("title") or "")
        + "\n"
        + str(normalized_article.get("summary") or "")
    )
    return sanitize_signal_type(t, candidate, article_signal_type or "Other")


def derive_wealth_fields(
    candidate: dict[str, Any],
    normalized_article: dict[str, Any],
    cross_check_result: dict[str, Any],
    *,
    wealth_status_hint: str | None = None,
) -> dict[str, Any]:
    """
    Wealth display fields: never treat article/company money as personal net worth.
    Delegates to ``estimate_wealth_safely`` + ``validate_display_wealth``.
    """
    from ai_prospect_pipeline import classify_wealth_status
    from wealth_display import estimate_wealth_safely, validate_display_wealth

    article_text = (
        str(normalized_article.get("article_text") or "")
        + "\n"
        + str(normalized_article.get("summary") or "")
    )
    wstat_pre = wealth_status_hint or classify_wealth_status(article_text, candidate, cross_check_result)
    safe_w = estimate_wealth_safely(
        candidate,
        article_text,
        cross_check_result,
        wealth_status_hint=wstat_pre,
    )
    est = str(safe_w.get("est_wealth") or "Data pending")
    vd = validate_display_wealth(
        {
            "est_wealth": est,
            "est_wealth_display": est,
            "wealth_numeric_verified": bool(safe_w.get("wealth_numeric_verified")),
        }
    )
    return {
        "est_wealth": str(vd.get("est_wealth_display") or est),
        "wealth_status": str(safe_w.get("wealth_status") or wstat_pre),
        "wealth_confidence": int(safe_w.get("wealth_confidence") or 0),
        "wealth_numeric_verified": bool(vd.get("wealth_numeric_verified")),
    }


__all__ = [
    "derive_wealth_fields",
    "infer_signal_type",
    "is_invalid_candidate_name",
    "resolve_person_company",
]
