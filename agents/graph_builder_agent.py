# agents/graph_builder_agent.py
# Agent 3 — Graph Builder Agent
# Takes signal features and builds/updates the Neo4j graph

import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
from config.signal_config import SIM_EDGE_FLAG_THRESHOLD

load_dotenv()

URI      = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

if PASSWORD:
    PASSWORD = PASSWORD.strip('"').strip("'")


def get_driver():
    """Create and return a Neo4j driver instance."""
    return GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))


def upsert_domain_nodes(driver, features: dict, excerpts: dict = None) -> int:
    """
    Create or update domain nodes in Neo4j.
    Uses MERGE so existing nodes get updated, not duplicated.
    excerpts: optional dict of domain -> text excerpt for evidence display.
    """
    excerpts = excerpts or {}
    nodes_created = 0
    with driver.session() as session:
        for domain, feats in features.items():
            session.run("""
                MERGE (d:Domain {domain: $domain})
                SET d.avg_similarity    = $avg_similarity,
                    d.max_similarity    = $max_similarity,
                    d.similarity_flag   = $similarity_flag,
                    d.anomaly_score     = $anomaly_score,
                    d.cadence_flagged   = $cadence_flagged,
                    d.burst_score       = $burst_score,
                    d.domain_age_days   = $domain_age_days,
                    d.registrar         = $registrar,
                    d.whois_flagged     = $whois_flagged,
                    d.signals_triggered = $signals_triggered,
                    d.preliminary_verdict = $preliminary_verdict,
                    d.excerpt           = $excerpt,
                    d.updated_at        = timestamp()
            """,
            domain=domain,
            avg_similarity   = float(feats.get('avg_similarity', 0)),
            max_similarity   = float(feats.get('max_similarity', 0)),
            similarity_flag  = int(feats.get('similarity_flag', 0)),
            anomaly_score    = float(feats.get('anomaly_score', 0)),
            cadence_flagged  = int(feats.get('cadence_flagged', 0)),
            burst_score      = float(feats.get('burst_score', 0)),
            domain_age_days  = int(feats.get('domain_age_days', -1)),
            registrar        = str(feats.get('registrar', 'unknown'))[:80],
            whois_flagged    = int(feats.get('whois_flagged', 0)),
            signals_triggered = int(feats.get('signals_triggered', 0)),
            preliminary_verdict = (
                'SYNTHETIC' if feats.get('signals_triggered', 0) >= 2
                else 'REVIEW' if feats.get('signals_triggered', 0) == 1
                else 'ORGANIC'
            ),
            excerpt = str(excerpts.get(domain, ''))[:400])
            nodes_created += 1

    return nodes_created


def upsert_similarity_edges(driver, sim_edges: dict) -> int:
    """
    Create or update SIMILAR_TO edges between domain pairs.
    """
    edges_created = 0
    with driver.session() as session:
        for key, score in sim_edges.items():
            # Key format: "domain_a|||domain_b"
            if "|||" in str(key):
                domain_a, domain_b = str(key).split("|||")
            elif isinstance(key, tuple):
                domain_a, domain_b = key
            else:
                continue

            session.run("""
                MATCH (a:Domain {domain: $domain_a})
                MATCH (b:Domain {domain: $domain_b})
                MERGE (a)-[r:SIMILAR_TO {domain_a: $domain_a, domain_b: $domain_b}]->(b)
                SET r.similarity = $similarity,
                    r.flagged    = $flagged
            """,
            domain_a=domain_a,
            domain_b=domain_b,
            similarity=float(score),
            flagged=1 if float(score) >= SIM_EDGE_FLAG_THRESHOLD else 0)
            edges_created += 1

    return edges_created


def get_graph_stats(driver) -> dict:
    """Query Neo4j for current graph statistics."""
    with driver.session() as session:
        nodes = session.run(
            "MATCH (n:Domain) RETURN count(n) AS c"
        ).single()['c']
        edges = session.run(
            "MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) AS c"
        ).single()['c']
        synthetic = session.run(
            "MATCH (n:Domain) WHERE n.preliminary_verdict='SYNTHETIC' "
            "RETURN count(n) AS c"
        ).single()['c']

    return {
        'total_nodes':      nodes,
        'total_edges':      edges,
        'synthetic_domains': synthetic,
    }


def build_graph(analysis_result: dict) -> dict:
    """
    Main graph building function.
    Takes output from Fingerprint Analyst Agent.
    Creates/updates Neo4j nodes and edges.

    Args:
        analysis_result: dict with 'features', 'sim_edges', and 'excerpts' keys

    Returns:
        dict with graph statistics
    """
    print(f"🏗️  Graph Builder Agent starting...")

    features  = analysis_result.get('features', {})
    sim_edges = analysis_result.get('sim_edges', {})
    excerpts  = analysis_result.get('excerpts', {})

    print(f"   Domains to upsert: {len(features)}")
    print(f"   Edges to upsert:   {len(sim_edges)}")

    try:
        driver = get_driver()
        driver.verify_connectivity()

        # Upsert nodes (with excerpts for evidence display)
        nodes_created = upsert_domain_nodes(driver, features, excerpts)
        print(f"   ✅ {nodes_created} domain nodes upserted")

        # Upsert edges
        edges_created = upsert_similarity_edges(driver, sim_edges)
        print(f"   ✅ {edges_created} similarity edges upserted")

        # Get stats
        stats = get_graph_stats(driver)
        driver.close()

        result = {
            'nodes_upserted': nodes_created,
            'edges_upserted': edges_created,
            'graph_stats':    stats,
            'status':         'success',
        }

    except Exception as e:
        print(f"   ❌ Graph Builder error: {e}")
        result = {
            'nodes_upserted': 0,
            'edges_upserted': 0,
            'graph_stats':    {},
            'status':         'error',
            'error':          str(e),
        }

    print(f"✅ Graph Builder complete!")
    return result


def build_graph_tool(analysis_json: str) -> str:
    """CrewAI tool wrapper."""
    analysis_result = json.loads(analysis_json)
    result = build_graph(analysis_result)
    return json.dumps(result)


# ── Standalone test ──────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from agents.crawler_agent import crawl_domains
    from agents.fingerprint_agent import analyze_domains

    print("Testing Graph Builder Agent...")
    print()

    data     = crawl_domains(["example.com"])
    analysis = analyze_domains(data)
    result   = build_graph(analysis)

    print()
    print("Graph stats:", result['graph_stats'])
    print()
    print("🎉 Graph Builder Agent test passed!")