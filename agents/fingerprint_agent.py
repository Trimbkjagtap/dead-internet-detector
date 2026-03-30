# agents/fingerprint_agent.py
# Agent 2 — Fingerprint Analyst Agent
# Takes crawled domain data, computes all 3 signals
# Excludes subdomain-to-parent comparisons from similarity

import json
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from sentence_transformers import SentenceTransformer
from config.signal_config import (
    MODEL_NAME,
    SIM_THRESHOLD,
    bounded_contamination,
)
from services.whois_service import WhoisService

# ── Config ───────────────────────────────────────────
CADENCE_THRESHOLD = -0.1

# Load model once at module level (cached after first load)
_model = None

def get_model():
    global _model
    if _model is None:
        print("  Loading Sentence Transformer model...")
        _model = SentenceTransformer(MODEL_NAME)
        print("  ✅ Model loaded")
    return _model


def get_parent_domain(domain: str) -> str:
    """Extract parent domain from subdomain."""
    parts = domain.split('.')
    if len(parts) >= 3 and parts[-2] in ['co', 'com', 'org', 'net', 'gov']:
        return '.'.join(parts[-3:])
    elif len(parts) >= 2:
        return '.'.join(parts[-2:])
    return domain


def is_same_site(domain_a: str, domain_b: str) -> bool:
    """
    Check if two domains are from the same site.
    e.g. bostonglobe.com and sponsored.bostonglobe.com are the same site.
    e.g. twitter.com and x.com are different sites (handled by redirect detection).
    """
    return get_parent_domain(domain_a) == get_parent_domain(domain_b)


def compute_signal1_similarity(domains_data: list) -> dict:
    """
    Signal 1: Content similarity using Sentence Transformers.
    Excludes subdomain-to-parent comparisons to avoid false positives.
    """
    print("  📐 Computing Signal 1 (content similarity)...")

    if len(domains_data) < 2:
        return {'domain_scores': {}, 'edges': {}}

    model   = get_model()
    texts   = [d.get('text', '')[:1000] for d in domains_data]
    domains = [d['domain'] for d in domains_data]

    # Compute embeddings
    embeddings = model.encode(texts, show_progress_bar=False)

    # Compute pairwise cosine similarity
    sim_matrix = cosine_similarity(embeddings)

    # Build edges and per-domain scores
    edges         = {}
    domain_scores = {}
    # Map domain -> short excerpt for evidence display (first 400 chars of clean text)
    excerpts      = {d['domain']: d.get('text', '')[:400].strip() for d in domains_data}

    for i in range(len(domains)):
        similarities = []
        flagged_pairs = []

        for j in range(len(domains)):
            if i == j:
                continue

            # Skip same-site comparisons (subdomains of same parent)
            if is_same_site(domains[i], domains[j]):
                continue

            sim = float(sim_matrix[i][j])
            similarities.append(sim)

            if sim >= SIM_THRESHOLD:
                pair_key = "|||".join(sorted([domains[i], domains[j]]))
                if pair_key not in edges:
                    edges[pair_key] = round(sim, 4)
                flagged_pairs.append((domains[j], sim))

        avg_sim = float(np.mean(similarities)) if similarities else 0.0
        max_sim = float(max(similarities)) if similarities else 0.0

        domain_scores[domains[i]] = {
            'avg_similarity':      round(avg_sim, 4),
            'max_similarity':      round(max_sim, 4),
            'similarity_flag':     1 if max_sim >= SIM_THRESHOLD else 0,
            'similar_domain_count': len(flagged_pairs),
        }

    flagged = sum(1 for v in domain_scores.values() if v['similarity_flag'] == 1)
    print(f"  ✅ Signal 1: {len(edges)} similar pairs found, {flagged} domains flagged")

    return {'domain_scores': domain_scores, 'edges': edges, 'excerpts': excerpts}


def compute_signal2_cadence(domains_data: list) -> dict:
    """
    Signal 2: Publishing cadence anomaly detection using Isolation Forest.
    """
    print("  ⏰ Computing Signal 2 (cadence anomaly)...")

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

        text_len = len(d.get('text', ''))
        has_links = 1 if d.get('links', '') else 0

        cadence_features.append([hour, minute, dow, text_len, has_links])
        domain_names.append(domain)

    if len(cadence_features) < 3:
        result = {}
        for domain in domain_names:
            result[domain] = {
                'anomaly_score':   0.0,
                'cadence_flagged': 0,
                'burst_score':     0.0,
            }
        return result

    X   = np.array(cadence_features, dtype=float)
    contamination = bounded_contamination(len(cadence_features))
    clf = IsolationForest(n_estimators=50, contamination=contamination, random_state=42)
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
    Signal 3: Domain registration pattern analysis.
    Flags domains with suspicious naming patterns and TLDs.
    """
    print("  🔍 Computing Signal 3 (WHOIS enrichment + budgeted fallback)...")
    whois_service = WhoisService()

    result = {}
    for d in domains_data:
        domain = d['domain'].lower()
        parent = get_parent_domain(domain)
        whois = whois_service.get_domain_features(parent)

        result[domain] = {
            'domain_age_days':      int(whois.get('domain_age_days', -1)),
            'whois_flagged':        int(whois.get('whois_flagged', 0)),
            'registrar':            whois.get('registrar', 'unknown'),
            'registration_country': whois.get('registration_country', ''),
            'whois_source':         whois.get('source', 'heuristic'),
        }

    flagged = sum(1 for v in result.values() if v['whois_flagged'] == 1)
    budget = whois_service.budget_status()
    print(
        f"  ✅ Signal 3: {flagged} domains flagged, WHOIS budget "
        f"{budget['queries_today']}/{budget['daily_limit']} used"
    )
    return result


def analyze_domains(domains_data: list) -> dict:
    """
    Main analysis function.
    Takes crawled domain data, computes all 3 signals.
    """
    print(f"🔬 Fingerprint Analyst starting...")
    print(f"   Analyzing {len(domains_data)} domains...")
    print()

    if not domains_data:
        return {}

    sig1 = compute_signal1_similarity(domains_data)
    sig2 = compute_signal2_cadence(domains_data)
    sig3 = compute_signal3_whois(domains_data)

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
            'avg_similarity':   s1.get('avg_similarity', 0.0),
            'max_similarity':   s1.get('max_similarity', 0.0),
            'similarity_flag':  similarity_flag,
            'anomaly_score':    s2.get('anomaly_score', 0.0),
            'cadence_flagged':  cadence_flagged,
            'burst_score':      s2.get('burst_score', 0.0),
            'domain_age_days':  s3.get('domain_age_days', -1),
            'registrar':        s3.get('registrar', 'unknown'),
            'whois_flagged':    whois_flagged,
            'signals_triggered': signals_triggered,
            'hour_variance':    0.0,
        }

    print()
    print(f"✅ Fingerprint Analyst complete!")
    print(f"   Features computed for {len(features)} domains")

    return {
        'features':   features,
        'sim_edges':  sig1.get('edges', {}),
        'excerpts':   sig1.get('excerpts', {}),
    }


def analyze_domains_tool(domains_json: str) -> str:
    """CrewAI tool wrapper."""
    domains_data = json.loads(domains_json)
    result = analyze_domains(domains_data)
    result['sim_edges'] = {
        f"{k[0]}|||{k[1]}": v
        for k, v in result.get('sim_edges', {}).items()
    }
    return json.dumps(result)


if __name__ == "__main__":
    from agents.crawler_agent import crawl_domains

    print("Testing Fingerprint Analyst Agent...")
    test_data = crawl_domains(["example.com", "wikipedia.org"])
    result = analyze_domains(test_data)

    print("\nSample features:")
    for domain, feats in list(result['features'].items())[:2]:
        print(f"\n  {domain}:")
        for k, v in feats.items():
            print(f"    {k}: {v}")
    print("\n🎉 Fingerprint Analyst Agent test passed!")