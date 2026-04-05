"""
Wealth list ingestion (Forbes-style) + cross-check against news-derived prospects.

Loads local datasets when present, builds a normalized identity map, and applies
optional priority boosts when a person matches a curated list.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

_DATASETS = Path(__file__).resolve().parent / "datasets"
_BILLIONAIRES_CSV = _DATASETS / "billionaires.csv"
_THIRTY_JSON = _DATASETS / "forbes_30_under_30.json"


def _norm_identity_key(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*&\s*family\s*$", "", s, flags=re.I)
    return s.strip()


def load_forbes_billionaires() -> list[dict[str, Any]]:
    """
    Load billionaire rows from ``datasets/billionaires.csv`` (Kaggle/Forbes-style export).
    Returns canonical dicts: name, net_worth, company, source.
    """
    if not _BILLIONAIRES_CSV.is_file():
        return []
    try:
        df = pd.read_csv(_BILLIONAIRES_CSV, encoding="latin-1")
    except (OSError, UnicodeDecodeError):
        try:
            df = pd.read_csv(_BILLIONAIRES_CSV, encoding="utf-8", errors="replace")
        except OSError:
            return []

    if df.empty:
        return []

    name_col = "Name" if "Name" in df.columns else None
    if not name_col:
        for c in df.columns:
            if str(c).strip().lower() == "name":
                name_col = c
                break
    if not name_col:
        return []

    nw_col = None
    for c in df.columns:
        cl = str(c).lower().replace("\n", " ")
        if "net worth" in cl:
            nw_col = c
            break

    src_col = None
    for c in df.columns:
        cl = str(c).lower()
        if "source" in cl and "wealth" in cl:
            src_col = c
            break
    if not src_col:
        for c in df.columns:
            if "wealth" in str(c).lower() or "source" in str(c).lower():
                src_col = c
                break

    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_name = str(row.get(name_col) or "").strip()
        if not raw_name or raw_name.lower() == "nan":
            continue
        nw = ""
        if nw_col:
            nw = str(row.get(nw_col) or "").strip()
            nw = re.sub(r"\s+", " ", nw.replace("\xa0", " "))
        co = ""
        if src_col:
            co = str(row.get(src_col) or "").strip()
            co = re.sub(r"\s+", " ", co.replace("\xa0", " "))
        out.append(
            {
                "name": raw_name,
                "net_worth": nw if nw else "See Forbes list",
                "company": co if co else "",
                "source": "Forbes Billionaires (dataset)",
                "tag": None,
            }
        )
    return out


def load_forbes_30_under_30() -> list[dict[str, Any]]:
    """
    Optional JSON list at ``datasets/forbes_30_under_30.json``::

        [{"name": "...", "company": "...", "tag": "30_under_30"}]

    Returns ``[]`` if the file is missing.
    """
    if not _THIRTY_JSON.is_file():
        return []
    try:
        with open(_THIRTY_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        nm = str(item.get("name") or "").strip()
        if not nm:
            continue
        rows.append(
            {
                "name": nm,
                "company": str(item.get("company") or "").strip(),
                "net_worth": str(item.get("net_worth") or "").strip(),
                "source": str(item.get("source") or "Forbes 30 Under 30"),
                "tag": str(item.get("tag") or "30_under_30"),
            }
        )
    return rows


@lru_cache(maxsize=1)
def build_identity_db() -> dict[str, dict[str, Any]]:
    """
    Map normalized name → richest record first (billionaires), then 30 under 30 for gaps.
    """
    db: dict[str, dict[str, Any]] = {}

    for p in load_forbes_billionaires():
        key = _norm_identity_key(p["name"])
        if not key:
            continue
        db[key] = dict(p)

    for p in load_forbes_30_under_30():
        key = _norm_identity_key(p["name"])
        if not key:
            continue
        if key not in db:
            db[key] = dict(p)

    return db


def identity_db_clear_cache() -> None:
    """Tests / hot reload."""
    build_identity_db.cache_clear()


def enrich_from_identity_db(entity: dict[str, Any]) -> dict[str, Any]:
    """
    If name matches the identity DB, override company / est_wealth and set ``verified``.
    """
    db = build_identity_db()
    name = str(entity.get("name") or "")
    key = _norm_identity_key(name)
    if key in db:
        enriched = db[key]
        if enriched.get("company"):
            entity["company"] = enriched["company"]
        nw = enriched.get("net_worth") or ""
        if nw:
            entity["est_wealth"] = nw
        if enriched.get("tag"):
            entity["wealth_list_tag"] = enriched["tag"]
        entity["wealth_list_source"] = enriched.get("source", "")
        entity["verified"] = True
    else:
        entity.setdefault("verified", False)

    return entity


def apply_identity_boost(entity: dict[str, Any]) -> int:
    """
    Add list-based boosts to ``priority_score`` (cap 100). Mutates ``entity``.
    """
    score = int(entity.get("priority_score") or 0)
    if entity.get("verified"):
        score += 25
    if entity.get("wealth_list_tag") == "30_under_30":
        score += 15
    score = min(100, score)
    entity["priority_score"] = score
    return score


def apply_wealth_list_to_prospect(row: dict[str, Any]) -> dict[str, Any]:
    """
    Run identity enrichment + priority boost on one pipeline row (mutates dict).
    Syncs legacy score / label / display fields.
    """
    enrich_from_identity_db(row)
    apply_identity_boost(row)

    try:
        from hybrid_pipeline import priority_label_from_score
    except ImportError:
        priority_label_from_score = None  # type: ignore[assignment]

    ps = int(row.get("priority_score") or 0)
    row["score"] = ps
    if priority_label_from_score:
        lab = priority_label_from_score(ps)
        row["priority_level"] = lab
        row["priority_label"] = lab
    if row.get("est_wealth"):
        row["est_wealth_display"] = row["est_wealth"]

    return row
