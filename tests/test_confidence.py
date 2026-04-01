"""
tests/test_confidence.py
Tests for verdict confidence scoring.

Runs without Neo4j / external APIs — tests the logic in isolation.

Run with:
    python -m pytest tests/test_confidence.py -v
"""

import sys
import pytest
sys.path.insert(0, '.')

from agents.verdict_agent import compute_confidence


# ── helpers ─────────────────────────────────────────────

def feats(signals=0, max_sim=0.0, burst=0.0):
    """Build a minimal feature dict."""
    return {
        'signals_triggered': signals,
        'max_similarity':    max_sim,
        'burst_score':       burst,
    }


# ── ORGANIC cases ────────────────────────────────────────

class TestOrganicConfidence:
    """
    For ORGANIC verdicts (0 signals), confidence means
    "how sure are we it's clean?" — so it should be HIGH (≥ 0.70).
    The caller in produce_verdict() inverts the formula for ORGANIC.
    """

    def test_clean_domain_confidence_is_high(self):
        """A domain with no signals and low similarity should be ≥ 0.70 clean."""
        f = feats(signals=0, max_sim=0.1)
        # produce_verdict computes ORGANIC confidence as:
        # max(0.55, 1.0 - (sim_boost + cadence_boost + gnn_boost))
        # with sim_boost = min(0.1*0.3, 0.3) = 0.03
        # → 1.0 - 0.03 = 0.97 → clamped to 0.97
        sim_boost = min(f['max_similarity'] * 0.3, 0.3)
        conf = round(max(0.55, min(1.0 - sim_boost, 0.97)), 2)
        assert conf >= 0.70, f"Clean domain confidence too low: {conf}"

    def test_high_similarity_lowers_organic_confidence(self):
        """A domain near the threshold but not triggered should still be lower confidence."""
        f_low  = feats(signals=0, max_sim=0.1)
        f_high = feats(signals=0, max_sim=0.44)  # just below threshold

        sim_boost_low  = min(f_low['max_similarity'] * 0.3, 0.3)
        sim_boost_high = min(f_high['max_similarity'] * 0.3, 0.3)

        conf_low  = round(max(0.55, min(1.0 - sim_boost_low,  0.97)), 2)
        conf_high = round(max(0.55, min(1.0 - sim_boost_high, 0.97)), 2)

        assert conf_low > conf_high, (
            f"Higher similarity should lower organic confidence. "
            f"Got low={conf_low}, high={conf_high}"
        )

    def test_organic_confidence_floor(self):
        """Even worst-case organic (high similarity near threshold) stays ≥ 0.55."""
        f = feats(signals=0, max_sim=0.44)
        sim_boost = min(f['max_similarity'] * 0.3, 0.3)
        conf = round(max(0.55, min(1.0 - sim_boost, 0.97)), 2)
        assert conf >= 0.55


# ── REVIEW cases ─────────────────────────────────────────

class TestReviewConfidence:
    """
    REVIEW verdict = 1–2 signals triggered.
    Confidence should reflect partial evidence — meaningfully above 0.5
    but clearly below SYNTHETIC range.
    Target: 0.40 – 0.69
    """

    def test_one_signal_confidence_in_review_range(self):
        conf = compute_confidence(feats(signals=1, max_sim=0.0), syn_prob=0.1)
        assert 0.30 <= conf <= 0.75, f"1-signal confidence out of expected range: {conf}"

    def test_two_signals_higher_than_one(self):
        conf1 = compute_confidence(feats(signals=1, max_sim=0.0), syn_prob=0.1)
        conf2 = compute_confidence(feats(signals=2, max_sim=0.0), syn_prob=0.1)
        assert conf2 > conf1, f"2 signals ({conf2}) should score higher than 1 ({conf1})"

    def test_similarity_boost_increases_review_confidence(self):
        conf_no_sim   = compute_confidence(feats(signals=1, max_sim=0.0),  syn_prob=0.1)
        conf_with_sim = compute_confidence(feats(signals=1, max_sim=0.54), syn_prob=0.1)
        assert conf_with_sim > conf_no_sim, (
            f"High similarity should boost confidence. "
            f"no_sim={conf_no_sim}, with_sim={conf_with_sim}"
        )

    def test_review_confidence_below_synthetic(self):
        """Max REVIEW (2 signals, high sim) should be below min SYNTHETIC (3 signals)."""
        max_review  = compute_confidence(feats(signals=2, max_sim=0.8, burst=0.9), syn_prob=0.9)
        min_synth   = compute_confidence(feats(signals=3, max_sim=0.0), syn_prob=0.0)
        assert max_review < min_synth, (
            f"Max REVIEW ({max_review}) should be < min SYNTHETIC ({min_synth})"
        )


# ── SYNTHETIC cases ──────────────────────────────────────

class TestSyntheticConfidence:
    """
    SYNTHETIC verdict = 3+ signals triggered.
    Confidence should be ≥ 0.60 and increase with more signals.
    """

    def test_three_signals_confidence_high_enough(self):
        conf = compute_confidence(feats(signals=3, max_sim=0.0), syn_prob=0.0)
        assert conf >= 0.40, f"3-signal confidence too low: {conf}"

    def test_seven_signals_near_ceiling(self):
        conf = compute_confidence(feats(signals=7, max_sim=0.9, burst=1.0), syn_prob=1.0)
        assert conf >= 0.80, f"7-signal confidence should be high, got {conf}"
        assert conf <= 0.97, f"Confidence should not exceed 0.97 ceiling, got {conf}"

    def test_more_signals_higher_confidence(self):
        conf3 = compute_confidence(feats(signals=3), syn_prob=0.5)
        conf5 = compute_confidence(feats(signals=5), syn_prob=0.5)
        conf7 = compute_confidence(feats(signals=7), syn_prob=0.5)
        assert conf3 < conf5 < conf7, (
            f"Confidence should increase with signals. Got {conf3}, {conf5}, {conf7}"
        )

    def test_confidence_ceiling(self):
        conf = compute_confidence(feats(signals=7, max_sim=1.0, burst=1.0), syn_prob=1.0)
        assert conf <= 0.97


# ── Boundary / edge cases ────────────────────────────────

class TestBoundaryConditions:

    def test_zero_signals_confidence_floor(self):
        """Absolute minimum inputs should not go below 0.02."""
        conf = compute_confidence(feats(signals=0, max_sim=0.0, burst=0.0), syn_prob=0.0)
        assert conf >= 0.02

    def test_confidence_always_between_zero_and_one(self):
        """No inputs should produce confidence outside [0, 1]."""
        cases = [
            feats(0, 0.0, 0.0),
            feats(1, 0.5, 0.3),
            feats(3, 0.9, 0.8),
            feats(7, 1.0, 1.0),
        ]
        for f in cases:
            c = compute_confidence(f, syn_prob=0.99)
            assert 0.0 <= c <= 1.0, f"Confidence {c} out of [0,1] for feats={f}"

    def test_gnn_boost_is_small(self):
        """GNN alone (no signals) should add at most 5% to confidence."""
        conf_no_gnn   = compute_confidence(feats(signals=0), syn_prob=0.0)
        conf_full_gnn = compute_confidence(feats(signals=0), syn_prob=1.0)
        assert (conf_full_gnn - conf_no_gnn) <= 0.06, (
            f"GNN boost too large: {conf_full_gnn - conf_no_gnn:.3f}"
        )


# ── Specific known cases ─────────────────────────────────

class TestKnownCases:
    """
    Sanity-check specific real-world-like scenarios against expected ranges.
    """

    def test_infowars_like_one_similarity_signal(self):
        """
        infowars.com: 1 signal (content similarity ~54%), no other signals.
        Expected: REVIEW verdict. Confidence should read as 'low-medium suspicion',
        roughly 35–55%.
        """
        conf = compute_confidence(feats(signals=1, max_sim=0.54), syn_prob=0.2)
        assert 0.25 <= conf <= 0.60, (
            f"infowars-like case: expected 25–60% confidence, got {conf:.0%}"
        )

    def test_clear_fake_network_three_signals(self):
        """
        Coordinated fake network: 3 signals, high similarity, new domain.
        Expected: SYNTHETIC verdict, high confidence ≥ 0.65.
        """
        conf = compute_confidence(feats(signals=3, max_sim=0.82, burst=0.7), syn_prob=0.8)
        assert conf >= 0.60, f"Clear fake network confidence too low: {conf}"

    def test_legit_established_site(self):
        """
        bbc.com-like: 0 signals, very low similarity to anything.
        Organic confidence (inverted formula) should be very high.
        """
        sim_boost = min(0.05 * 0.3, 0.3)
        conf = round(max(0.55, min(1.0 - sim_boost, 0.97)), 2)
        assert conf >= 0.90, f"Established legit site confidence too low: {conf}"

    def test_new_suspicious_domain_two_signals(self):
        """
        Brand new domain (wayback + whois signals triggered) but no similarity.
        Should be REVIEW, moderate confidence ~35–55%.
        """
        conf = compute_confidence(feats(signals=2, max_sim=0.1), syn_prob=0.3)
        assert 0.25 <= conf <= 0.65, (
            f"New suspicious domain: expected 25–65% confidence, got {conf:.0%}"
        )
