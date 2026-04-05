"""Regression tests for source parsing, invalid names, and person/company resolution."""

from __future__ import annotations

from article_parser import minimal_normalized_from_row, parse_article_source
from prospect_resolution import (
    is_invalid_candidate_name,
    resolve_person_company,
)


def test_is_invalid_fraternity_and_product():
    art = "Sorority members discussed philanthropy."
    assert is_invalid_candidate_name("Delta Delta Delta", "person", art)
    assert is_invalid_candidate_name("Tesla Model S", "product", art)
    assert is_invalid_candidate_name("Acme Corp", "company", art)
    assert not is_invalid_candidate_name("Jane Smith", "person", "Jane Smith founded Acme.")


def test_resolve_vimana_naran():
    na = {
        "article_text": "Vimana Private Jets CEO Ameerh Naran announced expansion in Dubai.",
        "first_paragraphs": "",
        "title": "Private aviation",
        "summary": "",
    }
    cand = {"name": "Ameerh Naran", "role": "CEO", "company": "Vimana"}
    cc = {"canonical_company": "Vimana", "canonical_role": "CEO"}
    rrole, rco, ch = resolve_person_company(cand, na, cc)
    assert ch
    assert "Vimana" in rco and "Private" in rco


def test_resolve_sam_altman_openai():
    na = {
        "article_text": "OpenAI CEO Sam Altman testified about AI policy.",
        "first_paragraphs": "",
        "title": "",
        "summary": "",
    }
    cand = {"name": "Sam Altman", "role": "CEO", "company": "Unknown"}
    cc = {"canonical_company": "Unknown", "canonical_role": "CEO"}
    _, rco, ch = resolve_person_company(cand, na, cc)
    assert ch
    assert rco == "OpenAI"


def test_minimal_normalized_from_row():
    row = {
        "raw_title": "Test headline",
        "source_url": "https://example.com/a",
        "full_explanation": "First para.\n\nSecond para.",
        "summary": "",
    }
    na = minimal_normalized_from_row(row)
    assert na["title"] == "Test headline"
    assert "example.com" in na["source_domain"]


def test_parse_article_source_feed_only_no_network():
    out = parse_article_source({"url": "", "feed_title": "T", "feed_summary": "S"})
    assert out["title"] == "T"
    assert not out.get("_parse_ok")
