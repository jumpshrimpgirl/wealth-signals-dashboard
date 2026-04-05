"""
Person enrichment layer: validate extracted names against external profile APIs before dashboard ingest.

Primary sources (when API keys are set): People Data Labs, Clearbit, FullContact.
Secondary: optional Whitepages-style lookup — existence / contact hints only, never role/company.

When no provider keys are configured, enrichment falls back to article-extracted role/company
(``source=extracted_fallback``) so the feed still works; set WEALTH_SIGNALS_ENRICHMENT_STRICT=1 to
require a successful API match instead.
"""

from __future__ import annotations

import os
from typing import Any

import requests

_SESSION = requests.Session()
_SESSION.headers.update(
    {"User-Agent": "WealthSignalsDashboard/1.0 (person-enrichment; educational)"}
)

VALID_PROSPECT_KEYWORDS = (
    "founder",
    "co-founder",
    "cofounder",
    "ceo",
    "cfo",
    "coo",
    "cto",
    "cmo",
    "ciso",
    "chief",
    "managing director",
    "general partner",
    "partner",
    "vp",
    "vice president",
    "svp",
    "evp",
    "director",
    "owner",
    "investor",
    "president",
    "chairman",
    "chairwoman",
    "chair",
    "board",
    "principal",
    "angel",
    "venture",
    "private equity",
)

VALID_SENIORITY_TOKENS = (
    "cxo",
    "vp",
    "director",
    "owner",
    "partner",
    "founder",
    "board",
    "executive",
    "investor",
    "manager",
)


def enrichment_layer_enabled() -> bool:
    return os.environ.get("WEALTH_SIGNALS_PERSON_ENRICHMENT", "1").lower() not in ("0", "false", "no")


def enrichment_strict() -> bool:
    return os.environ.get("WEALTH_SIGNALS_ENRICHMENT_STRICT", "0").lower() in ("1", "true", "yes")


def is_valid_prospect(role: str | None, seniority: str | None) -> bool:
    """
    True when role (required) matches wealth-advisor-relevant keywords, or seniority hints align.
    """
    rl = (role or "").strip()
    if not rl:
        return False
    low = rl.lower()
    if any(k in low for k in VALID_PROSPECT_KEYWORDS):
        return True
    sen = (seniority or "").strip().lower()
    if sen and any(t in sen for t in VALID_SENIORITY_TOKENS):
        return True
    return False


def compute_enrichment_confidence(enriched: dict[str, Any] | None, wp_data: dict[str, Any] | None) -> int:
    """
    Match the pipeline spec: +20 if enriched record, +20 if role and company, +10 if Whitepages.
    """
    confidence = 0
    if enriched:
        confidence += 20
    role = str((enriched or {}).get("job_title") or "").strip()
    company = str((enriched or {}).get("company") or "").strip()
    if role and company:
        confidence += 20
    if wp_data:
        confidence += 10
    return confidence


def _pdl_enrich(name: str, company: str) -> dict[str, Any] | None:
    key = (os.environ.get("PEOPLE_DATA_LABS_API_KEY") or os.environ.get("PDL_API_KEY") or "").strip()
    if not key:
        return None
    url = "https://api.peopledatalabs.com/v5/person/enrich"
    payload: dict[str, Any] = {"name": name.strip()}
    if company:
        payload["company"] = company.strip()
    try:
        r = _SESSION.post(
            url,
            json=payload,
            headers={"X-Api-Key": key, "Content-Type": "application/json"},
            timeout=12.0,
        )
    except (requests.RequestException, OSError):
        return None
    if r.status_code != 200:
        return None
    try:
        body = r.json()
    except ValueError:
        return None
    status = body.get("status")
    if status is not None and int(status) != 200:
        return None
    data = body.get("data")
    if not isinstance(data, dict):
        return None
    job_title = str(data.get("job_title") or "").strip()
    job_company = str(data.get("job_company_name") or "").strip()
    levels = data.get("job_title_levels")
    seniority = ""
    if isinstance(levels, list) and levels:
        seniority = str(levels[0] or "").strip()
    return {
        "job_title": job_title,
        "company": job_company,
        "seniority": seniority,
        "source": "peopledatalabs",
        "api_verified": True,
        "raw": data,
    }


def _clearbit_enrich(name: str, company: str) -> dict[str, Any] | None:
    """
    Clearbit Person API is email-centric. Optional: CLEARBIT_API_KEY + person lookup by domain later.
    """
    del name, company
    return None


def _fullcontact_enrich(name: str, company: str) -> dict[str, Any] | None:
    key = (os.environ.get("FULLCONTACT_API_KEY") or "").strip()
    if not key:
        return None
    url = "https://api.fullcontact.com/v3/person.enrich"
    payload: dict[str, Any] = {"name": name.strip()}
    if company:
        payload["company"] = company.strip()
    try:
        r = _SESSION.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=12.0,
        )
    except (requests.RequestException, OSError):
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    title = ""
    org = ""
    details = data.get("details") or data
    if isinstance(details, dict):
        employment = details.get("employment") or details.get("employmentHistory")
        if isinstance(employment, list) and employment:
            e0 = employment[0]
            if isinstance(e0, dict):
                title = str(e0.get("title") or "").strip()
                org = str(e0.get("name") or e0.get("company") or "").strip()
        title = title or str(details.get("title") or "").strip()
        org = org or str(details.get("organization") or "").strip()
    return {
        "job_title": title,
        "company": org,
        "seniority": "",
        "source": "fullcontact",
        "api_verified": True,
        "raw": data,
    }


def enrich_person(
    name: str,
    *,
    company_hint: str = "",
    role_hint: str = "",
) -> dict[str, Any] | None:
    """
    Resolve a person's professional identity. Tries PDL, then FullContact. Clearbit is reserved
    for email-based flows. Returns None only when enrichment_strict() and all APIs fail or return
    no match. Otherwise returns extracted_fallback dict using hints from the article.
    """
    name = (name or "").strip()
    if not name:
        return None
    company_hint = (company_hint or "").strip()
    role_hint = (role_hint or "").strip()

    for fn in (_pdl_enrich, _fullcontact_enrich):
        out = fn(name, company_hint)
        if not out:
            continue
        if not out.get("job_title") and role_hint:
            out["job_title"] = role_hint
        if not out.get("company") and company_hint:
            out["company"] = company_hint
        out.setdefault("api_verified", True)
        return out

    if enrichment_strict():
        return None

    return {
        "job_title": role_hint,
        "company": company_hint,
        "seniority": "",
        "source": "extracted_fallback",
        "api_verified": False,
        "raw": {},
    }


def lookup_whitepages(name: str, company: str = "") -> dict[str, Any] | None:
    """
    Optional secondary check: confirm a real record may exist. Does not set role/company.

    Set WHITEPAGES_API_KEY and WHITEPAGES_API_URL (full URL template with {name} optional) to enable.
    Default: disabled (returns None).
    """
    key = (os.environ.get("WHITEPAGES_API_KEY") or "").strip()
    base = (os.environ.get("WHITEPAGES_API_URL") or "").strip()
    if not key or not base:
        return None
    try:
        url = base.format(name=name, company=company or "")
    except (KeyError, ValueError):
        url = base
    try:
        r = _SESSION.get(
            url,
            headers={"Authorization": f"Bearer {key}"},
            params={"name": name, "company": company},
            timeout=8.0,
        )
    except (requests.RequestException, OSError):
        return None
    if r.status_code != 200:
        return None
    return {"matched": True, "source": "whitepages", "status_code": r.status_code}
