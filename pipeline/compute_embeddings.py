# pipeline/compute_embeddings.py
# Computes content embeddings and similarity scores (Signal 1)
# Input:  data/domains_clean.csv
# Output: data/similarity_edges.csv
# Run with: python3 pipeline/compute_embeddings.py

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_FILE   = "data/domains_clean.csv"
OUTPUT_FILE  = "data/similarity_edges.csv"
MODEL_NAME   = "all-MiniLM-L6-v2"   # fast, lightweight, runs on CPU
THRESHOLD    = 0.85                  # pairs above this are flagged suspicious
BATCH_SIZE   = 64                    # process embeddings in batches


def compute_embeddings():
    print("=" * 55)
    print("Dead Internet Detector — Day 6")
    print("Computing content embeddings (Signal 1)...")
    print("=" * 55)
    print()

    # ── Load clean data ──
    print(f"📂 Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # Use domains_clean if available, else domains_raw
    if len(df) == 0:
        print("❌ No data found. Run pipeline/preprocess.py first.")
        return

    # Fill missing text
    df['text'] = df['text'].fillna("").astype(str)

    # Keep only rows with meaningful text
    df = df[df['text'].str.len() > 50].reset_index(drop=True)
    print(f"✅ Loaded {len(df)} domains")
    print()

    # ── Load embedding model ──
    print(f"🤖 Loading embedding model: {MODEL_NAME}")
    print("   (Downloads ~80MB on first run — please wait...)")
    model = SentenceTransformer(MODEL_NAME)
    print("✅ Model loaded!")
    print()

    # ── Compute embeddings ──
    print(f"🔢 Computing embeddings for {len(df)} domains...")
    print("   This may take 5–10 minutes on CPU...")

    texts = df['text'].tolist()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"✅ Embeddings computed! Shape: {embeddings.shape}")
    print()

    # ── Compute pairwise cosine similarity ──
    print("📐 Computing pairwise cosine similarity...")
    print("   (Comparing all domain pairs...)")

    # For large datasets compute in chunks to save memory
    n = len(df)
    domains = df['domain'].tolist()
    rows = []

    # Process in chunks of 100 to avoid memory issues
    chunk_size = 100
    total_pairs = 0

    for i in range(0, n, chunk_size):
        chunk_emb = embeddings[i:i+chunk_size]
        # Compare this chunk against ALL embeddings
        sim_matrix = cosine_similarity(chunk_emb, embeddings)

        for local_idx, global_idx in enumerate(range(i, min(i+chunk_size, n))):
            for j in range(global_idx + 1, n):  # avoid duplicates
                score = float(sim_matrix[local_idx][j])
                if score >= THRESHOLD:
                    rows.append({
                        "domain_a":   domains[global_idx],
                        "domain_b":   domains[j],
                        "similarity": round(score, 4),
                        "flagged":    1
                    })
                total_pairs += 1

        # Show progress every 5 chunks
        if (i // chunk_size) % 5 == 0:
            print(f"   Processed {min(i+chunk_size, n)}/{n} domains...")

    print()
    print(f"✅ Compared {total_pairs:,} domain pairs")
    print(f"✅ Found {len(rows)} suspicious pairs (similarity > {THRESHOLD})")
    print()

    # ── Also save ALL pairs above 0.5 for graph building ──
    # (We'll need non-suspicious pairs too for the graph)
    all_rows = []
    for i in range(0, n, chunk_size):
        chunk_emb = embeddings[i:i+chunk_size]
        sim_matrix = cosine_similarity(chunk_emb, embeddings)
        for local_idx, global_idx in enumerate(range(i, min(i+chunk_size, n))):
            for j in range(global_idx + 1, n):
                score = float(sim_matrix[local_idx][j])
                if score >= 0.5:  # save edges above 0.5
                    all_rows.append({
                        "domain_a":   domains[global_idx],
                        "domain_b":   domains[j],
                        "similarity": round(score, 4),
                        "flagged":    1 if score >= THRESHOLD else 0
                    })

    # ── Save results ──
    os.makedirs("data", exist_ok=True)
    result_df = pd.DataFrame(all_rows)

    if len(result_df) == 0:
        # If no pairs found above 0.5, save top 100 most similar pairs
        print("⚠️  No pairs above 0.5 found — saving top 100 most similar pairs instead")
        all_pairs = []
        for i in range(0, n, chunk_size):
            chunk_emb = embeddings[i:i+chunk_size]
            sim_matrix = cosine_similarity(chunk_emb, embeddings)
            for local_idx, global_idx in enumerate(range(i, min(i+chunk_size, n))):
                for j in range(global_idx + 1, n):
                    score = float(sim_matrix[local_idx][j])
                    all_pairs.append({
                        "domain_a":   domains[global_idx],
                        "domain_b":   domains[j],
                        "similarity": round(score, 4),
                        "flagged":    1 if score >= THRESHOLD else 0
                    })
        result_df = pd.DataFrame(all_pairs).nlargest(100, 'similarity')

    result_df = result_df.sort_values('similarity', ascending=False).reset_index(drop=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    # ── Summary ──
    suspicious = result_df[result_df['flagged'] == 1]
    print("=" * 55)
    print(f"✅ Total edges saved:      {len(result_df)}")
    print(f"✅ Suspicious pairs (>{THRESHOLD}): {len(suspicious)}")
    print(f"✅ Saved to:               {OUTPUT_FILE}")
    print()
    print("Top 5 most similar domain pairs:")
    print(result_df[['domain_a','domain_b','similarity']].head(5).to_string(index=False))
    print()
    print("=" * 55)
    print("🎉 Signal 1 complete! similarity_edges.csv is ready.")
    print("=" * 55)


if __name__ == "__main__":
    compute_embeddings()