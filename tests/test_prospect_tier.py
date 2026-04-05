"""Prospect tier: actionable (A) vs saturated (B) vs noise (C)."""

from prospect_tier import (
    apply_tier_priority_adjustment,
    classify_prospect_tier,
    is_high_profile,
)


def test_larry_ellison_high_profile_tier_b():
    cross = {"wealth_evidence": "none", "_wealth_list": {}, "_enrichment": {}}
    assert is_high_profile("Larry Ellison", cross) is True
    c = {"name": "Larry Ellison", "economic_role": "ceo", "context_type": "primary", "role": "CEO"}
    tier = classify_prospect_tier(
        c,
        {**cross, "_tier_article_summary": "Oracle earnings", "_tier_wealth_status": "likely_wealth", "_tier_founder_wealth_score": 10},
    )
    assert tier == "tier_b"


def test_medvi_style_tier_a():
    cross = {"wealth_evidence": "indirect", "_wealth_list": {}, "_enrichment": {}, "ownership_inference": "high"}
    c = {
        "name": "Matthew Gallagher",
        "economic_role": "founder",
        "context_type": "primary",
        "role": "CEO",
        "is_real_person": True,
    }
    tier = classify_prospect_tier(
        c,
        {
            **cross,
            "_tier_article_summary": "revenue growth private company",
            "_tier_wealth_status": "likely_wealth",
            "_tier_founder_wealth_score": 35,
        },
    )
    assert tier == "tier_a"


def test_commentator_tier_c():
    c = {"economic_role": "commentator", "context_type": "commentary", "name": "Jane Doe", "is_real_person": True}
    tier = classify_prospect_tier(c, {"_tier_article_summary": "", "_tier_wealth_status": "unclear", "_tier_founder_wealth_score": 0})
    assert tier == "tier_c"


def test_tier_priority_adjustment():
    assert apply_tier_priority_adjustment(70, "tier_a") == 85
    assert apply_tier_priority_adjustment(70, "tier_b") == 45
    assert apply_tier_priority_adjustment(80, "tier_c") == 25
