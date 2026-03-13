# pipeline/merge_features.py
# Merges all 3 signals into one master feature table
# Input:  data/domains_clean.csv, similarity_edges.csv,
#         cadence_scores.csv, whois_features.csv, ground_truth.csv
# Output: data/domain_features.csv
# Run with: python3 pipeline/merge_features.py

import os
import numpy as np
import pandas as pd

# ── File paths ──────────────────────────────────────
CLEAN_FILE      = "data/domains_clean.csv"
SIMILARITY_FILE = "data/similarity_edges.csv"
CADENCE_FILE    = "data/cadence_scores.csv"
WHOIS_FILE      = "data/whois_features.csv"
TRUTH_FILE      = "data/ground_truth.csv"
OUTPUT_FILE     = "data/domain_features.csv"

# ── Thresholds ──────────────────────────────────────
SIM_THRESHOLD     = 0.85   # Signal 1 flag threshold
CADENCE_THRESHOLD = -0.1   # Signal 2 flag threshold


def load_all_files():
    """Load all input files and report counts."""
    print("📂 Loading all signal files...")

    df_clean = pd.read_csv(CLEAN_FILE)
    print(f"  ✅ domains_clean.csv:    {len(df_clean)} domains")

    df_sim = pd.read_csv(SIMILARITY_FILE)
    print(f"  ✅ similarity_edges.csv: {len(df_sim)} pairs")

    df_cad = pd.read_csv(CADENCE_FILE)
    print(f"  ✅ cadence_scores.csv:   {len(df_cad)} domains")

    df_whois = pd.read_csv(WHOIS_FILE)
    print(f"  ✅ whois_features.csv:   {len(df_whois)} domains")

    df_truth = pd.read_csv(TRUTH_FILE)
    print(f"  ✅ ground_truth.csv:     {len(df_truth)} labeled domains")

    return df_clean, df_sim, df_cad, df_whois, df_truth


def compute_similarity_features(domains, df_sim):
    """
    For each domain compute:
    - avg_similarity: average similarity score with other domains
    - max_similarity: highest similarity score with any other domain
    - similarity_flag: 1 if max_similarity > threshold
    - similar_domain_count: how many domains it's similar to
    """
    sim_features = {}

    for domain in domains:
        # Find all edges involving this domain
        mask = (df_sim['domain_a'] == domain) | (df_sim['domain_b'] == domain)
        edges = df_sim[mask]

        if len(edges) == 0:
            sim_features[domain] = {
                'avg_similarity':      0.0,
                'max_similarity':      0.0,
                'similarity_flag':     0,
                'similar_domain_count': 0,
            }
        else:
            scores = edges['similarity'].tolist()
            max_sim = max(scores)
            sim_features[domain] = {
                'avg_similarity':      round(float(np.mean(scores)), 4),
                'max_similarity':      round(float(max_sim), 4),
                'similarity_flag':     1 if max_sim >= SIM_THRESHOLD else 0,
                'similar_domain_count': len(edges),
            }

    return pd.DataFrame.from_dict(sim_features, orient='index').reset_index().\
        rename(columns={'index': 'domain'})


def merge_all():
    print("=" * 55)
    print("Dead Internet Detector — Day 9")
    print("Merging all signals into master feature table...")
    print("=" * 55)
    print()

    # ── Load all data ──
    df_clean, df_sim, df_cad, df_whois, df_truth = load_all_files()
    print()

    # ── Start with clean domains as base ──
    print("🔧 Merging signals...")
    domains = df_clean['domain'].dropna().unique().tolist()
    base_df = pd.DataFrame({'domain': domains})

    # ── Signal 1: Similarity features ──
    print("  Adding Signal 1 (content similarity)...")
    sim_features = compute_similarity_features(domains, df_sim)
    base_df = base_df.merge(sim_features, on='domain', how='left')

    # ── Signal 2: Cadence features ──
    print("  Adding Signal 2 (cadence anomaly)...")
    cad_cols = ['domain', 'anomaly_score', 'burst_score',
                'hour_variance', 'cadence_flagged']
    # Only keep columns that exist
    cad_cols = [c for c in cad_cols if c in df_cad.columns]
    base_df = base_df.merge(df_cad[cad_cols], on='domain', how='left')

    # ── Signal 3: WHOIS features ──
    print("  Adding Signal 3 (WHOIS registration)...")
    whois_cols = ['domain', 'domain_age_days', 'registrar', 'whois_flagged']
    whois_cols = [c for c in whois_cols if c in df_whois.columns]
    base_df = base_df.merge(df_whois[whois_cols], on='domain', how='left')

    # ── Ground truth labels ──
    print("  Adding ground truth labels...")
    base_df = base_df.merge(
        df_truth[['domain', 'label']],
        on='domain', how='left'
    )

    # ── Fill missing values ──
    base_df['avg_similarity']       = base_df['avg_similarity'].fillna(0.0)
    base_df['max_similarity']       = base_df['max_similarity'].fillna(0.0)
    base_df['similarity_flag']      = base_df['similarity_flag'].fillna(0).astype(int)
    base_df['similar_domain_count'] = base_df['similar_domain_count'].fillna(0).astype(int)
    base_df['anomaly_score']        = base_df['anomaly_score'].fillna(0.0)
    base_df['burst_score']          = base_df['burst_score'].fillna(0.0)
    base_df['hour_variance']        = base_df['hour_variance'].fillna(0.0)
    base_df['cadence_flagged']      = base_df['cadence_flagged'].fillna(0).astype(int)
    base_df['domain_age_days']      = base_df['domain_age_days'].fillna(-1).astype(int)
    base_df['registrar']            = base_df['registrar'].fillna('unknown')
    base_df['whois_flagged']        = base_df['whois_flagged'].fillna(0).astype(int)
    base_df['label']                = base_df['label'].fillna(-1).astype(int)

    # ── Compute total signals triggered per domain ──
    base_df['signals_triggered'] = (
        base_df['similarity_flag'] +
        base_df['cadence_flagged'] +
        base_df['whois_flagged']
    )

    # ── Compute verdict ──
    # SYNTHETIC = 2+ signals triggered
    # REVIEW    = 1 signal triggered
    # ORGANIC   = 0 signals triggered
    def get_verdict(row):
        s = row['signals_triggered']
        if s >= 2:
            return 'SYNTHETIC'
        elif s == 1:
            return 'REVIEW'
        else:
            return 'ORGANIC'

    base_df['preliminary_verdict'] = base_df.apply(get_verdict, axis=1)

    # ── Save ──
    os.makedirs("data", exist_ok=True)
    base_df.to_csv(OUTPUT_FILE, index=False)

    # ── Summary ──
    labeled   = base_df[base_df['label'] != -1]
    synthetic = base_df[base_df['preliminary_verdict'] == 'SYNTHETIC']
    review    = base_df[base_df['preliminary_verdict'] == 'REVIEW']
    organic   = base_df[base_df['preliminary_verdict'] == 'ORGANIC']

    print()
    print("=" * 55)
    print(f"✅ Master feature table: {len(base_df)} rows × {len(base_df.columns)} columns")
    print(f"✅ Labeled domains:      {len(labeled)}")
    print()
    print("Preliminary verdicts:")
    print(f"  🔴 SYNTHETIC: {len(synthetic)} domains (2+ signals)")
    print(f"  🟡 REVIEW:    {len(review)} domains (1 signal)")
    print(f"  🟢 ORGANIC:   {len(organic)} domains (0 signals)")
    print()
    print("Columns:", list(base_df.columns))
    print()
    print("Sample rows:")
    print(base_df[['domain','avg_similarity','anomaly_score',
                   'signals_triggered','preliminary_verdict']
                 ].head(5).to_string(index=False))
    print()
    print(f"✅ Saved to: {OUTPUT_FILE}")
    print()
    print("=" * 55)
    print("🎉 Day 9 complete! domain_features.csv is ready.")
    print("    Next: Build Neo4j graph database (Day 10)")
    print("=" * 55)


if __name__ == "__main__":
    merge_all()