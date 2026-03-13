# database/neo4j_client.py
# Builds the domain graph in Neo4j AuraDB
# Input:  data/domain_features.csv, data/similarity_edges.csv
# Run with: python3 database/neo4j_client.py

import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI      = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

# Strip quotes if present
if PASSWORD:
    PASSWORD = PASSWORD.strip('"').strip("'")

FEATURES_FILE   = "data/domain_features.csv"
SIMILARITY_FILE = "data/similarity_edges.csv"


class GraphBuilder:
    def __init__(self):
        print("Connecting to Neo4j...")
        self.driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
        self.driver.verify_connectivity()
        print("✅ Connected!")
        print()

    def close(self):
        self.driver.close()

    def clear_graph(self):
        """Delete all existing nodes and edges."""
        print("🗑️  Clearing old data...")
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        print("✅ Graph cleared")
        print()

    def create_domain_nodes(self, df):
        """Create one node per domain with all feature properties."""
        print(f"📍 Creating {len(df)} domain nodes...")

        # Process in batches of 100
        batch_size = 100
        created = 0

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            nodes = []

            for _, row in batch.iterrows():
                nodes.append({
                    "domain":              str(row['domain']),
                    "avg_similarity":      float(row.get('avg_similarity', 0)),
                    "max_similarity":      float(row.get('max_similarity', 0)),
                    "similarity_flag":     int(row.get('similarity_flag', 0)),
                    "anomaly_score":       float(row.get('anomaly_score', 0)),
                    "burst_score":         float(row.get('burst_score', 0)),
                    "cadence_flagged":     int(row.get('cadence_flagged', 0)),
                    "domain_age_days":     int(row.get('domain_age_days', -1)),
                    "whois_flagged":       int(row.get('whois_flagged', 0)),
                    "signals_triggered":   int(row.get('signals_triggered', 0)),
                    "preliminary_verdict": str(row.get('preliminary_verdict', 'ORGANIC')),
                    "label":               int(row.get('label', -1)),
                })

            with self.driver.session() as s:
                s.run("""
                    UNWIND $nodes AS node
                    CREATE (d:Domain {
                        domain:              node.domain,
                        avg_similarity:      node.avg_similarity,
                        max_similarity:      node.max_similarity,
                        similarity_flag:     node.similarity_flag,
                        anomaly_score:       node.anomaly_score,
                        burst_score:         node.burst_score,
                        cadence_flagged:     node.cadence_flagged,
                        domain_age_days:     node.domain_age_days,
                        whois_flagged:       node.whois_flagged,
                        signals_triggered:   node.signals_triggered,
                        preliminary_verdict: node.preliminary_verdict,
                        label:               node.label
                    })
                """, nodes=nodes)

            created += len(batch)
            print(f"  Created {created}/{len(df)} nodes...")

        print(f"✅ Created {len(df)} domain nodes")
        print()

    def create_similarity_edges(self, df_sim):
        """Create SIMILAR_TO edges between similar domain pairs."""
        print(f"🔗 Creating {len(df_sim)} similarity edges...")

        batch_size = 100
        created = 0

        for i in range(0, len(df_sim), batch_size):
            batch = df_sim.iloc[i:i+batch_size]
            edges = []

            for _, row in batch.iterrows():
                edges.append({
                    "domain_a":   str(row['domain_a']),
                    "domain_b":   str(row['domain_b']),
                    "similarity": float(row['similarity']),
                    "flagged":    int(row.get('flagged', 0)),
                })

            with self.driver.session() as s:
                s.run("""
                    UNWIND $edges AS edge
                    MATCH (a:Domain {domain: edge.domain_a})
                    MATCH (b:Domain {domain: edge.domain_b})
                    CREATE (a)-[:SIMILAR_TO {
                        similarity: edge.similarity,
                        flagged:    edge.flagged
                    }]->(b)
                """, edges=edges)

            created += len(batch)

        print(f"✅ Created {len(df_sim)} SIMILAR_TO edges")
        print()

    def create_link_edges(self, df_clean):
        """Create LINKS_TO edges from the links column."""
        print("🔗 Creating LINKS_TO edges from hyperlinks...")
        created = 0

        for _, row in df_clean.iterrows():
            links_str = str(row.get('links', ''))
            if not links_str or links_str == 'nan':
                continue

            linked_domains = [l.strip() for l in links_str.split('|') if l.strip()]
            domain_a = str(row['domain'])

            for domain_b in linked_domains[:5]:  # max 5 links per domain
                try:
                    with self.driver.session() as s:
                        s.run("""
                            MATCH (a:Domain {domain: $domain_a})
                            MATCH (b:Domain {domain: $domain_b})
                            MERGE (a)-[:LINKS_TO]->(b)
                        """, domain_a=domain_a, domain_b=domain_b)
                    created += 1
                except:
                    continue

        print(f"✅ Created {created} LINKS_TO edges")
        print()

    def get_graph_stats(self):
        """Print summary stats of the graph."""
        with self.driver.session() as s:
            node_count = s.run("MATCH (n:Domain) RETURN count(n) AS c").single()['c']
            edge_count = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()['c']
            synthetic  = s.run(
                "MATCH (n:Domain) WHERE n.preliminary_verdict='SYNTHETIC' RETURN count(n) AS c"
            ).single()['c']
            review = s.run(
                "MATCH (n:Domain) WHERE n.preliminary_verdict='REVIEW' RETURN count(n) AS c"
            ).single()['c']

        print("=" * 55)
        print("📊 Graph Statistics:")
        print(f"   Nodes (domains):    {node_count}")
        print(f"   Edges (total):      {edge_count}")
        print(f"   🔴 SYNTHETIC:       {synthetic}")
        print(f"   🟡 REVIEW:          {review}")
        print(f"   🟢 ORGANIC:         {node_count - synthetic - review}")
        print("=" * 55)


def build_graph():
    print("=" * 55)
    print("Dead Internet Detector — Day 10")
    print("Building Neo4j Domain Graph...")
    print("=" * 55)
    print()

    # ── Load data ──
    print("📂 Loading feature data...")
    df_features = pd.read_csv(FEATURES_FILE)
    df_sim      = pd.read_csv(SIMILARITY_FILE)
    df_clean    = pd.read_csv("data/domains_clean.csv")
    print(f"  ✅ {len(df_features)} domains to load")
    print(f"  ✅ {len(df_sim)} similarity edges to load")
    print()

    # ── Build graph ──
    builder = GraphBuilder()

    # Clear old data first
    builder.clear_graph()

    # Create nodes
    builder.create_domain_nodes(df_features)

    # Create similarity edges
    builder.create_similarity_edges(df_sim)

    # Create hyperlink edges
    builder.create_link_edges(df_clean)

    # Print stats
    builder.get_graph_stats()

    builder.close()

    print()
    print("🎉 Graph built successfully!")
    print()
    print("Next steps:")
    print("  1. Open Neo4j Browser at console.neo4j.io")
    print("  2. Run: MATCH (n:Domain) RETURN n LIMIT 25")
    print("  3. You should see domain nodes as a visual graph!")
    print()
    print("=" * 55)


if __name__ == "__main__":
    build_graph()