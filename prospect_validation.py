"""
Final Prospect Validation Layer — runs after extraction + FA gating, before display.

Conservative priority (private-banker style): only High when a real named prospect,
Strong/Moderate wealth signal, High FA relevance, and a clear money angle.
Uncertainty is explicit via ``pv_*`` fields and ``pv_validation_debug``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from fa_gating import compute_fa_gating_row, prospect_identified_from_row
from person_validation import is_role_or_office_only_label, is_valid_person
from score import wealth_signal_rank

logger = logging.getLogger(__name__)

_WEALTH_ORDER = {"Strong": 4, "Moderate": 3, "Weak": 2, "None": 1, "": 0}
_LIQ_ORDER = {"Yes": 3, "Potential": 2, "No": 1, "": 0}

_RE_MONEY_ANGLE = re.compile(
    r"\b("
    r"money|wealth|liquid|liquidity|deal|sale|sold|ipo|spac|funding|round|million|billion|"
    r"exit|equity|compensation|payout|acqui|valuation|worth|raised|invest|stock|grant|"
    r"acquired|net\s*worth|pay|dividend|stake|buyout|merger|contract|earnout|bonus|"
    r"portfolio|assets|fortune|proceeds|cash"
    r")\b",
    re.I,
)


def _format_est_wealth_display(value: Any) -> str:
    """Mirror ``data.format_wealth`` without importing ``data`` (avoid cycles)."""
    if value is None:
        return "Data pending"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "Data pending"
    if pd.isna(v) or v == 0:
        return "Data pending"
    if v >= 1_000_000_000:
        return f"${v / 1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    return f"${v:,.0f}"


def merge_wealth_signals(rule: str, ai: str) -> str:
    """Prefer the stronger of rules-based and AI decision labels."""
    r = (rule or "None").strip()
    a = (ai or "").strip()
    if a in _WEALTH_ORDER and _WEALTH_ORDER.get(a, 0) > _WEALTH_ORDER.get(r, 0):
        return a
    return r if r in _WEALTH_ORDER else "None"


def merge_liquidity(rule: str, ai: str) -> str:
    r = (rule or "No").strip()
    a = (ai or "").strip()
    if a in _LIQ_ORDER and _LIQ_ORDER.get(a, 0) > _LIQ_ORDER.get(r, 0):
        return a
    return r if r in _LIQ_ORDER else "No"


def resolve_prospect_anchor(row: dict[str, Any]) -> tuple[bool, str]:
    """
    Identifiable individual (not role-only). Prefer structured extraction when valid.
    """
    ex = str(row.get("extracted_person_name") or "").strip()
    if ex and is_valid_person(ex) and not is_role_or_office_only_label(ex):
        return True, ex
    pn = str(row.get("person_name") or "").strip()
    ap = row.get("additional_people")
    rt = str(row.get("raw_title") or "")
    return prospect_identified_from_row(pn, ap, rt)


def why_explains_wealth_angle(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 14:
        return False
    return bool(_RE_MONEY_ANGLE.search(t))


def build_pv_why_one_sentence(row: dict[str, Any], fr_why: str) -> str:
    candidates = [
        row.get("ai_why_money"),
        row.get("ai_why_matters_fa"),
        row.get("why_it_matters"),
        fr_why,
        row.get("fa_why_one_sentence"),
    ]
    for c in candidates:
        s = str(c or "").strip()
        if len(s) >= 14:
            return s[:500]
    return (
        "No clear wealth angle stated in available text; verify in source before outreach."
    )


def liquidity_for_pv_display(machine: str) -> str:
    m = (machine or "No").strip()
    if m == "No":
        return "No clear liquidity event"
    return m


def _row_weak_signal_bool(v: Any) -> bool:
    if v is True:
        return True
    if v is False:
        return False
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    return str(v).lower() in ("true", "1", "yes")


def apply_why_downgrade(base: str, explains: bool) -> str:
    """If the why-line does not support a money angle, step down one tier."""
    if explains:
        return base
    if base == "High":
        return "Medium"
    if base == "Medium":
        return "Low"
    return "Low"


def derive_strict_priority(
    *,
    prospect_ok: bool,
    merged_wealth: str,
    fa_relevance: str,
    suppression: str,
    macro_noise: bool,
    weak_signal: bool,
    is_billionaire: bool,
) -> str:
    """
    Non-negotiable High bar + conservative default.
    High only when: named prospect, Strong|Moderate wealth, FA High, no hard suppression,
    not macro/noise weak-ingest junk.
    """
    if not prospect_ok:
        return "Low"
    if merged_wealth == "None":
        return "Low"
    if suppression == "hard" and not (is_billionaire and prospect_ok):
        return "Low"
    rel = (fa_relevance or "Low").strip()
    if rel == "Low":
        return "Low"

    if (
        merged_wealth in ("Strong", "Moderate")
        and rel == "High"
        and suppression == "none"
        and not macro_noise
        and not weak_signal
    ):
        return "High"

    if merged_wealth in ("Strong", "Moderate", "Weak") and rel in ("High", "Medium"):
        return "Medium"

    return "Low"


def _cap_score_for_priority(score: int, priority: str) -> int:
    if priority == "Low":
        return min(score, 48)
    if priority == "Medium":
        return min(score, 78)
    return min(score, 100)


def validate_one_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return column updates for a single signals row."""
    merged_ws = merge_wealth_signals(
        str(row.get("wealth_signal_label") or ""),
        str(row.get("ai_decision_wealth_signal") or ""),
    )
    merged_liq = merge_liquidity(
        str(row.get("liquidity_event") or ""),
        str(row.get("ai_decision_liquidity") or ""),
    )

    pid, pname = resolve_prospect_anchor(row)
    display_name = pname.strip() if pid and pname else "Not identified"

    pn_for_gate = str(row.get("person_name", "")).strip()
    exn = str(row.get("extracted_person_name") or "").strip()
    if exn and is_valid_person(exn) and not is_role_or_office_only_label(exn):
        pn_for_gate = exn

    sc = int(pd.to_numeric(row.get("score"), errors="coerce") or 0)
    macro = bool(row.get("_macro_noise_for_pv", False))
    fr = compute_fa_gating_row(
        raw_title=str(row.get("raw_title", "")),
        full_explanation=str(row.get("full_explanation", "")),
        person_name=pn_for_gate or str(row.get("person_name", "")),
        additional_people=row.get("additional_people"),
        event_type=str(row.get("event_type", "")),
        role=str(row.get("role", "")),
        client_type=str(row.get("client_type", "")),
        strength=merged_ws,
        liquidity=merged_liq,
        passes_wealth_gate=bool(row.get("wealth_passes_gate", False)),
        macro_noise=macro,
        weak_signal=_row_weak_signal_bool(row.get("weak_signal", False)),
        is_billionaire=bool(row.get("is_billionaire", False)),
        score=sc,
        estimated_wealth=float(row.get("estimated_wealth") or 0),
        aggregated_estimated_wealth=float(row.get("aggregated_estimated_wealth") or 0),
    )

    weak = _row_weak_signal_bool(row.get("weak_signal", False))
    is_b = bool(row.get("is_billionaire", False))

    pv_why = build_pv_why_one_sentence(row, fr.fa_why_one_sentence)
    explains = why_explains_wealth_angle(pv_why)

    base = derive_strict_priority(
        prospect_ok=pid,
        merged_wealth=merged_ws,
        fa_relevance=fr.fa_relevance,
        suppression=fr.suppression_level,
        macro_noise=macro,
        weak_signal=weak,
        is_billionaire=is_b,
    )
    final_p = apply_why_downgrade(base, explains)

    fa_rel_s = str(fr.fa_relevance or "Low").strip() or "Low"

    dbg_parts: list[str] = [
        f"merged_ws={merged_ws}",
        f"merged_liq={merged_liq}",
        f"prospect={int(pid)}",
        f"fa_rel={fa_rel_s}",
        f"suppression={fr.suppression_level}",
        f"base={base}",
        f"why_explains={int(explains)}",
        f"final={final_p}",
        f"gate_prospect={int(pid)}",
        f"gate_wealth={int(merged_ws != 'None')}",
        f"gate_fa_not_low={int(fa_rel_s != 'Low')}",
    ]
    fa_dbg = str(fr.fa_priority_debug or "") + f"|pv_strict={final_p}"

    new_score = _cap_score_for_priority(sc, final_p)

    wr = int(wealth_signal_rank(merged_ws))

    out: dict[str, Any] = {
        "wealth_signal_label": merged_ws,
        "liquidity_event": merged_liq,
        "wealth_rank": wr,
        "priority_level": final_p,
        "score": new_score,
        "ai_wealth_signal": merged_ws,
        "ai_liquidity_label": merged_liq,
        "fa_prospect_identified": "Yes" if pid else "No",
        "fa_structured_name": display_name if pid else "Not identified",
        "fa_relevance": fa_rel_s,
        "fa_why_one_sentence": pv_why[:800],
        "fa_pass_gate_person": bool(pid),
        "fa_pass_gate_wealth_substance": bool(fr.fa_pass_gate_wealth_substance),
        "fa_pass_gate_liquidity_or_uhnw": bool(fr.fa_pass_gate_liquidity_or_uhnw),
        "fa_pass_gate_fa_relevance": bool(fr.fa_pass_gate_fa_relevance),
        "fa_suppression_level": fr.suppression_level,
        "fa_suppression_reason": fr.suppression_reason,
        "fa_priority_debug": fa_dbg,
        "pv_prospect_identified": "Yes" if pid else "No",
        "pv_display_name": display_name,
        "pv_role_title": str(row.get("role") or "").strip() or "—",
        "pv_wealth_signal": merged_ws,
        "pv_liquidity_event": liquidity_for_pv_display(merged_liq),
        "pv_fa_relevance": fa_rel_s,
        "pv_why_it_matters": pv_why,
        "pv_gate_prospect_pass": bool(pid),
        "pv_gate_wealth_pass": merged_ws != "None",
        "pv_gate_fa_relevance_pass": fa_rel_s != "Low",
        "pv_validation_debug": "|".join(dbg_parts),
        "pv_estimated_wealth_display": _format_est_wealth_display(row.get("estimated_wealth")),
    }

    return out


def apply_prospect_validation_layer(df: pd.DataFrame) -> None:
    """
    Mutates ``df`` in place: merged wealth/liquidity (rules + AI), strict ``priority_level``,
    capped ``score``, populated ``pv_*`` audit fields, refreshed ``est_wealth_display``.
    """
    if df is None or df.empty:
        return

    if "_macro_noise_for_pv" not in df.columns:
        # Recompute macro flag the same way as ``data._apply_wealth_signal_metadata`` uses
        from score import is_macro_noise_without_wealth_hook, passes_wealth_high_priority_gate

        for idx in df.index:
            r = df.loc[idx]
            blob = f"{r.get('raw_title', '')} {r.get('full_explanation', '')}"
            gate = passes_wealth_high_priority_gate(
                raw_title=str(r.get("raw_title", "")),
                full_explanation=str(r.get("full_explanation", "")),
                event_type=str(r.get("event_type", "")),
                role=str(r.get("role", "")),
                estimated_wealth=float(r.get("estimated_wealth") or 0),
                aggregated_estimated_wealth=float(r.get("aggregated_estimated_wealth") or 0),
                funding_amount=str(r.get("funding_amount", "")),
                funding_stage=str(r.get("funding_stage", "")),
                is_billionaire=bool(r.get("is_billionaire", False)),
            )
            macro = is_macro_noise_without_wealth_hook(
                blob,
                float(r.get("estimated_wealth") or 0),
                bool(r.get("is_billionaire", False)),
                gate,
            )
            df.at[idx, "_macro_noise_for_pv"] = bool(macro)

    for idx in df.index:
        row = df.loc[idx].to_dict()
        try:
            upd = validate_one_row(row)
        except Exception as ex:
            logger.warning("prospect_validation row %s failed: %s", idx, ex)
            continue
        for k, v in upd.items():
            df.at[idx, k] = v
        # Promote a validated extracted name into the primary field when empty
        pid = upd.get("pv_prospect_identified") == "Yes"
        dname = str(upd.get("pv_display_name") or "").strip()
        cur_pn = str(df.at[idx, "person_name"] or "").strip()
        if pid and dname and dname != "Not identified" and not cur_pn:
            df.at[idx, "person_name"] = dname
        df.at[idx, "est_wealth_display"] = upd.get(
            "pv_estimated_wealth_display", _format_est_wealth_display(df.at[idx, "estimated_wealth"])
        )
        logger.debug("pv_validation %s: %s", idx, upd.get("pv_validation_debug", ""))

    if "_macro_noise_for_pv" in df.columns:
        df.drop(columns=["_macro_noise_for_pv"], inplace=True, errors="ignore")

