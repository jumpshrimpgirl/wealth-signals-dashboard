"""
Post high-priority prospect summaries to Slack via incoming webhook (optional).
Deduplicates with a persistent JSON file so refreshes do not re-notify.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from settings import SLACK_WEBHOOK_URL

# Stored next to this module so cwd does not change the path when using Streamlit.
SENT_FILE = Path(__file__).resolve().parent / "sent_prospects.json"


def load_sent() -> set[str]:
    if SENT_FILE.exists():
        try:
            with open(SENT_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return set(str(x) for x in data)
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    return set()


def save_sent(sent_set: set[str]) -> None:
    try:
        with open(SENT_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(sent_set), f, indent=0)
    except OSError:
        pass


def _is_high_priority(p: dict[str, Any]) -> bool:
    pl = str(p.get("priority_label") or "").strip()
    if pl in ("Elite", "High"):
        return True
    try:
        ps = int(float(p.get("priority_score", p.get("score", 0)) or 0))
        if ps >= 75:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _dedupe_key(p: dict[str, Any]) -> str:
    name = str(p.get("name") or p.get("person_name") or "").strip().lower()
    company = str(p.get("company") or p.get("company_name") or "").strip().lower()
    return f"{name}_{company}"


def send_slack_alert(prospects: list[dict[str, Any]]) -> None:
    if not SLACK_WEBHOOK_URL:
        return

    sent_cache = load_sent()

    high_priority = [p for p in prospects if _is_high_priority(p)]

    new_prospects: list[dict[str, Any]] = []
    new_keys: list[str] = []
    for p in high_priority:
        name = str(p.get("name") or p.get("person_name") or "").strip()
        company = str(p.get("company") or p.get("company_name") or "").strip()
        if not name and not company:
            continue
        key = _dedupe_key(p)
        if key not in sent_cache:
            new_prospects.append(p)
            new_keys.append(key)

    if not new_prospects:
        return

    message = "🔥 *New High-Value Wealth Signals*\n\n"

    for p in new_prospects[:5]:
        name = str(p.get("name") or p.get("person_name") or "Unknown").strip()
        role = str(p.get("role") or "N/A").strip()
        company = str(p.get("company") or p.get("company_name") or "N/A").strip()
        try:
            ps = int(float(p.get("priority_score", p.get("score", 0)) or 0))
        except (TypeError, ValueError):
            ps = 0
        message += f"""• *{name}*
{role} @ {company}
Priority: {ps} ({str(p.get("priority_label") or "").strip() or "—"})

"""

    try:
        r = requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=15)
        r.raise_for_status()
    except requests.RequestException:
        return

    sent_cache.update(new_keys)
    save_sent(sent_cache)
