"""
Resolve canonical company name for a prospect from multiple sources (priority order).

Typical ``data`` shape::

    {
        "crunchbase": {"company": "Acme Inc"},
        "search": {"company": "Acme Corporation"},
    }

``person_company`` is the article/extraction fallback (equivalent to ``person.company`` in product models).
"""

from __future__ import annotations

from typing import Any, Mapping


def resolve_company(person_company: str, data: Mapping[str, Any] | None = None) -> str:
    """
    Priority: Crunchbase → search (enrichment APIs) → article/extracted company → ``Data pending``.
    """
    d: Mapping[str, Any] = data or {}

    cb = d.get("crunchbase")
    if isinstance(cb, Mapping):
        c = str(cb.get("company") or "").strip()
        if c:
            return c

    sr = d.get("search")
    if isinstance(sr, Mapping):
        c = str(sr.get("company") or "").strip()
        if c:
            return c

    pc = str(person_company or "").strip()
    if pc:
        return pc

    return "Data pending"


def build_resolution_data(
    *,
    company_article: str,
    company_search: str = "",
    company_crunchbase: str = "",
) -> dict[str, Any]:
    """Build the ``data`` mapping for :func:`resolve_company` from string fields."""
    out: dict[str, Any] = {}
    cc = str(company_crunchbase or "").strip()
    if cc:
        out["crunchbase"] = {"company": cc}
    cs = str(company_search or "").strip()
    if cs:
        out["search"] = {"company": cs}
    return out
