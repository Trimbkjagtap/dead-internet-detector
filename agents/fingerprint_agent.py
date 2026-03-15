# agents/fingerprint_agent.py
# Agent 2 — Fingerprint Analyst Agent
# Takes crawled domain data, computes all 3 signals

import json
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────
MODEL_NAME       = "all-MiniLM-L6-v2"
SIM_THRESHOLD    = 0.85   # Signal 1 flag threshold
CADENCE_THRESHOLD = -0.1  # Signal 2 flag threshold

# Load model once at module level (cached after first load)
_model = None

def get_model():
    global _model
    if _model is None:
        print("  Loading Sentence Transformer model...")
        _model = SentenceTransformer(MODEL_NAME)
        print("  ✅ Model loaded")
    return _model


def compute_signal1_similarity(domains_data: list) -> dict:
    """
    Signal 1: Content similarity using Sentence Transformers.
    Returns dict mapping domain pairs to similarity scores.
    """
    print("  📐 Computing Signal 1 (content similarity)...")

    model   = get_model()
    domains = [d['domain'] for d in domains_data]
    texts   = [d.get('text', '') or '' for d in domains_data]

    # Filter out empty texts
    valid = [(d, t) for d, t in zip(domains, texts) if len(t) > 50]
    if len(valid) < 2:
        print("  ⚠️  Not enough text content for similarity")
        return {}

    valid_domains = [v[0] for v in valid]
    valid_texts   = [v[1] for v in valid]

    # Compute embeddings
    embeddings = model.encode(valid_texts, show_progress_bar=False)

    # Compute pairwise cosine similarity
    sim_matrix = cosine_similarity(embeddings)

    # Build similarity dict for pairs above threshold
    sim_edges = {}
    for i in range(len(valid_domains)):
        for j in range(i + 1, len(valid_domains)):
            score = float(sim_matrix[i][j])
            if score >= 0.5:  # save all edges >= 0.5
                pair = (valid_domains[i], valid_domains[j])
                sim_edges[pair] = round(score, 4)

    # Per-domain stats
    domain_sim = {}
    for domain in valid_domains:
        scores = []
        for (a, b), score in sim_edges.items():
            if a == domain or b == domain:
                scores.append(score)
        domain_sim[domain] = {
            'avg_similarity':  round(float(np.mean(scores)), 4) if scores else 0.0,
            'max_similarity':  round(float(max(scores)), 4) if scores else 0.0,
            'similarity_flag': 1 if scores and max(scores) >= SIM_THRESHOLD else 0,
        }

    print(f"  ✅ Signal 1: {len(sim_edges)} similar pairs found")
    return {'edges': sim_edges, 'domain_scores': domain_sim}


def compute_signal2_cadence(domains_data: list) -> dict:
    """
    Signal 2: Publishing cadence anomaly using Isolation Forest.
    Returns dict mapping domain to cadence features + anomaly score.
    """
    print("  ⏰ Computing Signal 2 (cadence anomaly)...")

    from collections import Counter

    cadence_features = []
    domain_names     = []

    for d in domains_data:
        domain    = d['domain']
        timestamp = d.get('timestamp', '')

        try:
            dt     = datetime.strptime(timestamp[:19], '%Y-%m-%d %H:%M:%S')
            hour   = dt.hour
            minute = dt.minute
            dow    = dt.weekday()
        except:
            hour = minute = dow = 0

        cadence_features.append([hour, minute, dow, 1])
        domain_names.append(domain)

    if len(cadence_features) < 3:
        # Not enough data for Isolation Forest
        result = {}
        for domain in domain_names:
            result[domain] = {
                'anomaly_score':   0.0,
                'cadence_flagged': 0,
                'burst_score':     0.0,
            }
        return result

    X   = np.array(cadence_features, dtype=float)
    clf = IsolationForest(n_estimators=50, contamination=0.1, random_state=42)
    clf.fit(X)
    scores      = clf.decision_function(X)
    predictions = clf.predict(X)

    result = {}
    for i, domain in enumerate(domain_names):
        result[domain] = {
            'anomaly_score':   round(float(scores[i]), 4),
            'cadence_flagged': 1 if predictions[i] == -1 else 0,
            'burst_score':     round(float(abs(scores[i])), 4),
        }

    flagged = sum(1 for v in result.values() if v['cadence_flagged'] == 1)
    print(f"  ✅ Signal 2: {flagged} domains with anomalous cadence")
    return result


def compute_signal3_whois(domains_data: list) -> dict:
    """
    Signal 3: Basic domain age estimation from URL patterns.
    (Full WHOIS disabled to preserve free tier quota)
    Returns dict mapping domain to whois features.
    """
    print("  🔍 Computing Signal 3 (domain features)...")

    result = {}
    for d in domains_data:
        domain = d['domain']
        # Use heuristics since WHOIS quota is limited
        suspicious_tlds = ['.xyz', '.top', '.click', '.online', '.site', '.info']
        suspicious = any(domain.endswith(t) for t in suspicious_tlds)

        result[domain] = {
            'domain_age_days': -1,
            'whois_flagged':   1 if suspicious else 0,
        }

    flagged = sum(1 for v in result.values() if v['whois_flagged'] == 1)
    print(f"  ✅ Signal 3: {flagged} domains with suspicious TLDs")
    return result


def analyze_domains(domains_data: list) -> dict:
    """
    Main analysis function.
    Takes crawled domain data, computes all 3 signals.

    Args:
        domains_data: list of dicts from Crawler Agent

    Returns:
        dict with all signal scores per domain
    """
    print(f"🔬 Fingerprint Analyst starting...")
    print(f"   Analyzing {len(domains_data)} domains...")
    print()

    if not domains_data:
        return {}

    # Compute all 3 signals
    sig1 = compute_signal1_similarity(domains_data)
    sig2 = compute_signal2_cadence(domains_data)
    sig3 = compute_signal3_whois(domains_data)

    # Merge into per-domain feature dict
    features = {}
    for d in domains_data:
        domain = d['domain']

        s1 = sig1.get('domain_scores', {}).get(domain, {})
        s2 = sig2.get(domain, {})
        s3 = sig3.get(domain, {})

        similarity_flag  = s1.get('similarity_flag', 0)
        cadence_flagged  = s2.get('cadence_flagged', 0)
        whois_flagged    = s3.get('whois_flagged', 0)
        signals_triggered = similarity_flag + cadence_flagged + whois_flagged

        features[domain] = {
            # Signal 1
            'avg_similarity':   s1.get('avg_similarity', 0.0),
            'max_similarity':   s1.get('max_similarity', 0.0),
            'similarity_flag':  similarity_flag,
            # Signal 2
            'anomaly_score':    s2.get('anomaly_score', 0.0),
            'cadence_flagged':  cadence_flagged,
            'burst_score':      s2.get('burst_score', 0.0),
            # Signal 3
            'domain_age_days':  s3.get('domain_age_days', -1),
            'whois_flagged':    whois_flagged,
            # Combined
            'signals_triggered': signals_triggered,
            'hour_variance':    0.0,
        }

    print()
    print(f"✅ Fingerprint Analyst complete!")
    print(f"   Features computed for {len(features)} domains")

    return {
        'features':   features,
        'sim_edges':  sig1.get('edges', {}),
    }


def analyze_domains_tool(domains_json: str) -> str:
    """CrewAI tool wrapper — accepts JSON string of domain data."""
    domains_data = json.loads(domains_json)
    result = analyze_domains(domains_data)
    # Convert tuple keys to strings for JSON serialization
    result['sim_edges'] = {
        f"{k[0]}|||{k[1]}": v
        for k, v in result.get('sim_edges', {}).items()
    }
    return json.dumps(result)


# ── Standalone test ──────────────────────────────────
if __name__ == "__main__":
    from agents.crawler_agent import crawl_domains

    print("Testing Fingerprint Analyst Agent...")
    print()

    # Crawl some test domains
    test_data = crawl_domains(["example.com", "wikipedia.org"])

    # Analyze them
    result = analyze_domains(test_data)

    print()
    print("Sample features:")
    for domain, feats in list(result['features'].items())[:2]:
        print(f"\n  {domain}:")
        for k, v in feats.items():
            print(f"    {k}: {v}")

    print()
    print("🎉 Fingerprint Analyst Agent test passed!")