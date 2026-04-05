"""Regression: founder-led private revenue scale (Medvi-style) without funding/M&A/Forbes."""

from ai_prospect_pipeline import (
    classify_wealth_status,
    score_article_signal,
    score_founder_wealth_creation,
)


MEDVI_STYLE_ARTICLE = """
Matthew Gallagher, founder and CEO of Medvi, a privately held telehealth company,
said the firm grew sales to about $401 million last year and is on a path toward
$1.8 billion in annual revenue. The business was bootstrapped with no outside funding
and remains founder-led with concentrated ownership. Gallagher started the company
personally and continues to lead operations.
"""


def test_score_article_signal_founder_wealth_creation_type():
    sig = score_article_signal(MEDVI_STYLE_ARTICLE, "Medvi CEO on growth", None)
    assert sig["economic_relevance"] is True
    assert sig["signal_type"] == "Founder Wealth Creation"
    assert sig["signal_score"] >= 50


def test_score_founder_wealth_creation_medvi_candidate():
    candidate = {
        "name": "Matthew Gallagher",
        "role": "CEO",
        "economic_role": "founder",
        "context_type": "primary",
    }
    cross = {"canonical_role": "CEO", "canonical_company": "Medvi"}
    out = score_founder_wealth_creation(MEDVI_STYLE_ARTICLE, candidate, cross)
    assert out["subscore"] >= 35


def test_classify_wealth_status_likely_not_unclear():
    candidate = {
        "name": "Matthew Gallagher",
        "role": "founder and CEO",
        "economic_role": "founder",
        "context_type": "primary",
    }
    cross = {
        "wealth_evidence": "indirect",
        "canonical_role": "CEO",
        "canonical_company": "Medvi",
        "_wealth_list": {},
        "_enrichment": {"_wikipedia_extract": ""},
    }
    ws = classify_wealth_status(MEDVI_STYLE_ARTICLE, candidate, cross)
    assert ws == "likely_wealth"
