# agents/crew.py
# Wires all 4 agents into a complete analysis pipeline
# Run with: python3 agents/crew.py

import sys
import json
sys.path.insert(0, '.')

from agents.crawler_agent      import crawl_domains, get_failed_domains
from agents.fingerprint_agent  import analyze_domains
from agents.graph_builder_agent import build_graph
from agents.verdict_agent      import produce_verdict


def run_analysis(seed_domains: list) -> dict:
    """
    Run the complete 4-agent Dead Internet Detector pipeline.

    Args:
        seed_domains: list of domain strings to investigate
                      e.g. ['suspicious-news.com', 'fake-updates.net']

    Returns:
        dict with:
        - cluster_verdict: 'SYNTHETIC' | 'REVIEW' | 'ORGANIC'
        - max_confidence: float 0–1
        - domain_verdicts: per-domain breakdown
        - graph_stats: Neo4j graph statistics
        - summary: plain-English summary
        - failed_domains: list of domains that couldn't be crawled
    """
    print("=" * 55)
    print("🕸️  Dead Internet Detector — Full Pipeline")
    print("=" * 55)
    print(f"Seed domains: {seed_domains}")
    print()

    # ── AGENT 1: Crawl domains ──────────────────────
    print("━" * 40)
    print("STEP 1/4 — Crawler Agent")
    print("━" * 40)
    crawled_data = crawl_domains(seed_domains)

    # Get failed domains for reporting
    failed = get_failed_domains()

    if not crawled_data:
        return {
            'cluster_verdict': 'ERROR',
            'error': 'Crawler returned no data — check internet connection',
            'seed_domains': seed_domains,
            'failed_domains': failed,
        }

    print(f"  Crawled {len(crawled_data)} domains successfully")
    if failed:
        print(f"  ⚠️ Failed to crawl: {', '.join(failed)}")
    print()

    # ── AGENT 2: Compute signals ────────────────────
    print("━" * 40)
    print("STEP 2/4 — Fingerprint Analyst Agent")
    print("━" * 40)
    analysis = analyze_domains(crawled_data)

    features     = analysis.get('features', {})
    sim_edges    = analysis.get('sim_edges', {})
    excerpts     = analysis.get('excerpts', {})
    host_edges   = analysis.get('host_edges', {})
    author_edges = analysis.get('author_edges', {})

    print(f"  Features computed for {len(features)} domains")
    print()

    # ── AGENT 3: Build/update graph ─────────────────
    print("━" * 40)
    print("STEP 3/4 — Graph Builder Agent")
    print("━" * 40)
    graph_result = build_graph(analysis)
    graph_stats  = graph_result.get('graph_stats', {})

    print(f"  Graph: {graph_stats.get('total_nodes',0)} nodes, "
          f"{graph_stats.get('total_edges',0)} edges")
    print()

    # ── AGENT 4: Produce verdict ────────────────────
    print("━" * 40)
    print("STEP 4/4 — Verdict Agent")
    print("━" * 40)
    verdict = produce_verdict(features, sim_edges, excerpts=excerpts)
    print()

    # ── Build final result ──────────────────────────
    cluster = verdict.get('cluster_verdict', 'UNKNOWN')
    conf    = verdict.get('max_confidence', 0.0)

    # Generate human-readable summary
    if cluster == 'SYNTHETIC':
        summary = (
            f"High Risk - Likely part of a fake network. "
            f"{verdict.get('synthetic_domains',0)} domain(s) were flagged with converging signals. "
            f"Confidence: {conf:.0%}."
        )
    elif cluster == 'REVIEW':
        summary = (
            f"Suspicious - Some warning signs detected. "
            f"{verdict.get('review_domains',0)} domain(s) need a closer look. "
            f"Only one major signal fired, so this is not a confirmed synthetic network yet."
        )
    else:
        summary = (
            f"Looks Legitimate - No warning signs were detected. "
            f"All {len(features)} domains currently appear organic."
        )

    # Add failed domain info to summary
    if failed:
        summary += f" Note: {len(failed)} domain(s) could not be crawled: {', '.join(failed)}."

    # Build content similarity evidence pairs
    evidence_pairs = []
    for key, score in sim_edges.items():
        if "|||" in str(key):
            a, b = str(key).split("|||")
        elif isinstance(key, (list, tuple)) and len(key) == 2:
            a, b = key
        else:
            continue
        # score may be a float (legacy) or a dict with raw/effective/metadata
        if isinstance(score, dict):
            display_sim          = score.get('effective', score.get('score', 0))
            raw_sim              = score.get('score', display_sim)
            authority_discounted = score.get('authority_discounted', False)
            syndicated           = score.get('syndicated', False)
            similarity_type      = score.get('similarity_type', 'unknown')
        else:
            display_sim          = float(score)
            raw_sim              = display_sim
            authority_discounted = False
            syndicated           = False
            similarity_type      = 'unknown'
        evidence_pairs.append({
            'domain_a':             a,
            'domain_b':             b,
            'similarity':           round(float(display_sim), 4),
            'raw_similarity':       round(float(raw_sim), 4),
            'authority_discounted': authority_discounted,
            'syndicated':           syndicated,
            'similarity_type':      similarity_type,
            'excerpt_a':            excerpts.get(a, ''),
            'excerpt_b':            excerpts.get(b, ''),
        })
    evidence_pairs.sort(key=lambda x: x['similarity'], reverse=True)

    # Build hosting evidence (shared IP/ASN)
    hosting_evidence = []
    for key, edge_type in host_edges.items():
        a, b = key.split("|||")
        hosting_evidence.append({'domain_a': a, 'domain_b': b, 'edge_type': edge_type})

    # Build author evidence (shared bylines)
    author_evidence = []
    seen_author_keys = set()
    for key, author in author_edges.items():
        if key not in seen_author_keys:
            seen_author_keys.add(key)
            a, b = key.split("|||")
            author_evidence.append({'domain_a': a, 'domain_b': b, 'author': author})

    if cluster == 'SYNTHETIC':
        headline  = "High Risk - Likely part of a fake network"
        risk_level = "HIGH_RISK"
    elif cluster == 'REVIEW':
        headline  = "Suspicious - Some warning signs detected"
        risk_level = "SUSPICIOUS"
    else:
        headline  = "Looks Legitimate - No warning signs"
        risk_level = "LOW_RISK"

    final_result = {
        'cluster_verdict':   cluster,
        'max_confidence':    conf,
        'risk_level':        risk_level,
        'headline':          headline,
        'synthetic_domains': verdict.get('synthetic_domains', 0),
        'review_domains':    verdict.get('review_domains', 0),
        'organic_domains':   verdict.get('organic_domains', 0),
        'domain_verdicts':   verdict.get('domain_verdicts', {}),
        'graph_stats':       graph_stats,
        'summary':           summary,
        'seed_domains':      seed_domains,
        'domains_analyzed':  len(features),
        'failed_domains':    failed,
        'evidence_pairs':    evidence_pairs,
        'hosting_evidence':  hosting_evidence,
        'author_evidence':   author_evidence,
    }

    # ── Print final summary ─────────────────────────
    print("=" * 55)
    print("🎯 FINAL RESULT")
    print("=" * 55)
    print(f"Verdict:    {cluster}")
    print(f"Confidence: {conf:.3f}")
    print(f"Summary:    {summary}")
    if failed:
        print(f"Failed:     {', '.join(failed)}")
    print()
    print("Per-domain breakdown:")
    for domain, v in verdict.get('domain_verdicts', {}).items():
        icon = "🔴" if v['verdict'] == 'SYNTHETIC' else \
               "🟡" if v['verdict'] == 'REVIEW' else "🟢"
        print(f"  {icon} {domain:<40} {v['verdict']} "
              f"(conf={v['confidence']:.3f}, signals={v['signals_triggered']})")
    print("=" * 55)

    return final_result


# ── Standalone test ──────────────────────────────────
if __name__ == "__main__":
    test_domains = ["example.com", "bbc.com", "wikipedia.org"]

    print("Running full pipeline test...")
    print()

    result = run_analysis(test_domains)

    print()
    print("Full result JSON:")
    summary_result = {k: v for k, v in result.items()
                      if k != 'domain_verdicts'}
    print(json.dumps(summary_result, indent=2))
    print()
    print("🎉 Full pipeline test complete!")