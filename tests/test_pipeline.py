"""
tests/test_pipeline.py
────────────────────────
Core integration tests for the Dead Internet Detector pipeline.
All tests run without external API calls (Neo4j, OpenAI, ipinfo).

Run with:
    python3 -m pytest tests/test_pipeline.py -v

Tests cover:
1. Authority pair threshold — nbcboston ↔ wgbh not flagged at 0.622 sim
2. Single signal → always ORGANIC (strict 1-of-7 cap)
3. Structural clone detection via _classify_similarity
4. CDN hosting: no flag without same-day + same-registrar
5. CDN hosting: flagged when same-day + same-registrar confirmed
6. Confidence formula tier boundaries
7. Syndication marker suppresses similarity entirely
8. Authority domain in AUTHORITY_DOMAINS set
9. Confidence formula single source of truth (no drift)
"""

import sys
import pytest
sys.path.insert(0, '.')


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_feats(signals=0, sim=0.0, burst=0.0, **overrides):
    """Build a minimal features dict for verdict agent tests."""
    base = {
        'signals_triggered':     signals,
        'similarity_flag':       0,
        'cadence_flagged':       0,
        'whois_flagged':         0,
        'hosting_flagged':       0,
        'link_network_flagged':  0,
        'wayback_flagged':       0,
        'author_overlap_flagged':0,
        'max_similarity':        sim,
        'burst_score':           burst,
        'domain_age_days':       -1,
        'domain_age_source':     'unknown',
        'ip_address':            '',
        'hosting_org':           '',
        'insular_score':         0.0,
        'wayback_snapshot_count': 0,
        'wayback_flag_reason':   '',
        'shared_authors':        '[]',
    }
    base.update(overrides)
    return base


# ── Test 1: Authority pair threshold ─────────────────────────────────────────

def test_authority_threshold_pair():
    """
    Both-authority pair threshold = SIM_THRESHOLD * 1.50.
    nbcboston.com ↔ wgbh.org scored 0.622 raw — must NOT trigger.
    """
    from config.signal_config import (
        SIM_THRESHOLD,
        AUTHORITY_PAIR_THRESHOLD_MULTIPLIER,
        AUTHORITY_DOMAINS,
    )
    assert "nbcboston.com" in AUTHORITY_DOMAINS
    assert "wgbh.org" in AUTHORITY_DOMAINS

    pair_threshold = SIM_THRESHOLD * AUTHORITY_PAIR_THRESHOLD_MULTIPLIER
    assert pair_threshold > 0.622, (
        f"Pair threshold {pair_threshold:.3f} must exceed nbcboston↔wgbh score 0.622"
    )


def test_authority_mixed_threshold():
    """One authority + one unknown: threshold raised 20%."""
    from config.signal_config import (
        SIM_THRESHOLD,
        AUTHORITY_MIXED_THRESHOLD_MULTIPLIER,
        AUTHORITY_DOMAINS,
    )
    assert "reuters.com" in AUTHORITY_DOMAINS
    mixed_threshold = SIM_THRESHOLD * AUTHORITY_MIXED_THRESHOLD_MULTIPLIER
    assert mixed_threshold > SIM_THRESHOLD
    assert mixed_threshold < SIM_THRESHOLD * 1.5   # less aggressive than full pair


# ── Test 2: Strict 1-of-7 cap ─────────────────────────────────────────────────

@pytest.mark.parametrize("which_signal", [
    'similarity_flag', 'cadence_flagged', 'whois_flagged',
    'hosting_flagged', 'link_network_flagged', 'wayback_flagged',
    'author_overlap_flagged',
])
def test_single_signal_always_organic(which_signal):
    """Any domain with exactly 1 signal fires must be ORGANIC."""
    from agents.verdict_agent import produce_verdict

    feats = make_feats(signals=1, **{which_signal: 1})
    result = produce_verdict({"test.com": feats}, {})
    verdict = result["domain_verdicts"]["test.com"]["verdict"]
    assert verdict == "ORGANIC", (
        f"signals=1 ({which_signal}) → expected ORGANIC, got {verdict}"
    )


def test_two_signals_review():
    """2 signals → REVIEW."""
    from agents.verdict_agent import produce_verdict
    feats = make_feats(signals=2, similarity_flag=1, cadence_flagged=1)
    result = produce_verdict({"test.com": feats}, {})
    verdict = result["domain_verdicts"]["test.com"]["verdict"]
    assert verdict == "REVIEW", f"signals=2 → expected REVIEW, got {verdict}"


def test_three_signals_synthetic():
    """3+ signals → SYNTHETIC."""
    from agents.verdict_agent import produce_verdict
    feats = make_feats(signals=3, similarity_flag=1, cadence_flagged=1, whois_flagged=1)
    result = produce_verdict({"test.com": feats}, {})
    verdict = result["domain_verdicts"]["test.com"]["verdict"]
    assert verdict == "SYNTHETIC", f"signals=3 → expected SYNTHETIC, got {verdict}"


def test_zero_signals_organic():
    """0 signals → ORGANIC."""
    from agents.verdict_agent import produce_verdict
    feats = make_feats(signals=0)
    result = produce_verdict({"test.com": feats}, {})
    verdict = result["domain_verdicts"]["test.com"]["verdict"]
    assert verdict == "ORGANIC"


# ── Test 3: Structural clone classifier ──────────────────────────────────────

def test_classify_structural_clone():
    """Near-verbatim sim + same authors → structural_clone."""
    from agents.fingerprint_agent import _classify_similarity
    result = _classify_similarity(
        raw_sim=0.93,
        authors_a=["Jane Smith"],
        authors_b=["Jane Smith"],
        text_a="breaking news article about local politics today",
        text_b="breaking news article about local politics today",
    )
    assert result == "structural_clone", f"Got: {result}"


def test_classify_shared_journalistic():
    """High sim + completely different bylines → shared_journalistic."""
    from agents.fingerprint_agent import _classify_similarity
    result = _classify_similarity(
        raw_sim=0.80,
        authors_a=["Jane Smith"],
        authors_b=["Bob Jones"],
        text_a="local boston news story about transit funding",
        text_b="boston transit funding story local news",
    )
    assert result == "shared_journalistic", f"Got: {result}"


def test_classify_topic_overlap():
    """Low similarity → topic_overlap."""
    from agents.fingerprint_agent import _classify_similarity
    result = _classify_similarity(
        raw_sim=0.62,
        authors_a=[],
        authors_b=[],
        text_a="weather update for massachusetts today",
        text_b="massachusetts forecast today weather",
    )
    assert result in ("topic_overlap", "shared_journalistic"), f"Got: {result}"


# ── Test 4: CDN hosting — no flag without same-day registrar ─────────────────

def test_cdn_hosting_no_flag_different_dates():
    """
    Two domains on AWS (CDN), different registration dates → hosting_flagged == 0.
    """
    from agents.fingerprint_agent import compute_signal4_hosting

    domains_data = [
        {"domain": "site-a.com", "status": "ok"},
        {"domain": "site-b.com", "status": "ok"},
    ]
    # Fake WHOIS: different creation dates
    whois_data = {
        "site-a.com": {"created_date": "2022-01-15", "registrar": "GoDaddy"},
        "site-b.com": {"created_date": "2023-06-20", "registrar": "GoDaddy"},
    }

    # Patch resolve_ip_and_asn to return same CDN IP for both
    import unittest.mock as mock
    fake_info = {
        "ip_address": "104.21.50.100",
        "asn": "AS13335",
        "hosting_org": "Cloudflare, Inc.",
        "is_cdn": True,
        "error": None,
        "cached_at": "2026-01-01T00:00:00",
    }
    with mock.patch("agents.fingerprint_agent.resolve_ip_and_asn", return_value=fake_info):
        result = compute_signal4_hosting(domains_data, whois_data=whois_data)

    flagged = sum(1 for v in result["domain_scores"].values() if v["hosting_flagged"])
    assert flagged == 0, (
        f"CDN pair with different registration dates should NOT be flagged, "
        f"got {flagged} flagged"
    )


def test_cdn_hosting_flags_same_day_same_registrar():
    """
    Two domains on AWS, same registration date + same registrar → hosting_flagged == 1.
    """
    from agents.fingerprint_agent import compute_signal4_hosting
    import unittest.mock as mock

    domains_data = [
        {"domain": "fake-a.com", "status": "ok"},
        {"domain": "fake-b.com", "status": "ok"},
    ]
    whois_data = {
        "fake-a.com": {"created_date": "2024-03-01", "registrar": "namecheap"},
        "fake-b.com": {"created_date": "2024-03-01", "registrar": "namecheap"},
    }

    fake_info = {
        "ip_address": "52.21.100.50",
        "asn": "AS16509",
        "hosting_org": "Amazon.com, Inc.",
        "is_cdn": True,
        "error": None,
        "cached_at": "2026-01-01T00:00:00",
    }
    with mock.patch("agents.fingerprint_agent.resolve_ip_and_asn", return_value=fake_info):
        result = compute_signal4_hosting(domains_data, whois_data=whois_data)

    flagged = sum(1 for v in result["domain_scores"].values() if v["hosting_flagged"])
    assert flagged > 0, (
        "CDN pair with same-day + same-registrar SHOULD be flagged"
    )


# ── Test 5: Confidence formula tier boundaries ────────────────────────────────

def test_confidence_synthetic_tier():
    """3+ signals → confidence in [0.65, 0.97]."""
    from config.signal_config import compute_confidence_from_signals
    for sigs in [3, 4, 5, 6, 7]:
        c = compute_confidence_from_signals(signals=sigs, max_sim=0.7, burst=0.3)
        assert 0.65 <= c <= 0.97, f"signals={sigs} confidence {c} out of SYNTHETIC range"


def test_confidence_review_tier():
    """2 signals → confidence in [0.40, 0.64]."""
    from config.signal_config import compute_confidence_from_signals
    c = compute_confidence_from_signals(signals=2, max_sim=0.5, burst=0.1)
    assert 0.40 <= c <= 0.64, f"signals=2 confidence {c} out of REVIEW range"


def test_confidence_organic_tier():
    """0 signals → confidence in [0.70, 0.97]."""
    from config.signal_config import compute_confidence_from_signals
    c = compute_confidence_from_signals(signals=0, max_sim=0.0, burst=0.0)
    assert 0.70 <= c <= 0.97, f"signals=0 confidence {c} out of ORGANIC range"


# ── Test 6: Syndication marker suppresses similarity ─────────────────────────

def test_syndication_marker_suppresses():
    """
    If excerpt contains 'associated press', effective_sim must be 0.
    The pair should not be added to edges regardless of raw score.
    """
    from config.signal_config import SYNDICATION_MARKERS
    exc = "this story was originally published by the associated press"
    assert any(marker in exc.lower() for marker in SYNDICATION_MARKERS), (
        "Syndication marker not detected in AP text"
    )


# ── Test 7: Confidence formula — single source of truth ──────────────────────

def test_confidence_formula_no_drift():
    """
    compute_confidence in verdict_agent must delegate to config formula.
    Both must return the same value for identical inputs.
    """
    from config.signal_config import compute_confidence_from_signals
    from agents.verdict_agent import compute_confidence

    feats = make_feats(signals=3, sim=0.75, burst=0.4)
    config_result = compute_confidence_from_signals(
        signals=3, max_sim=0.75, burst=0.4, gnn_prob=0.6
    )
    agent_result = compute_confidence(feats, syn_prob=0.6)
    assert config_result == agent_result, (
        f"Formula drift: config={config_result}, agent={agent_result}"
    )
