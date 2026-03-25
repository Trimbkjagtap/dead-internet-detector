# retrain_pipeline.py
# Rebuilds domain_features.csv with proper labels and retrains the GNN
# Run with: python3 retrain_pipeline.py
# Takes ~10-15 minutes (crawling 100 domains)

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timezone

sys.path.insert(0, '.')

from agents.crawler_agent import crawl_domains, fetch_page, extract_domain
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config.signal_config import SIM_THRESHOLD


# ══════════════════════════════════════════════════════
# STEP 1: CRAWL ALL GROUND TRUTH DOMAINS
# ══════════════════════════════════════════════════════

def crawl_ground_truth():
    """Crawl all ground truth domains and synthetic domains."""
    print("=" * 55)
    print("STEP 1: Crawling ground truth domains")
    print("=" * 55)
    
    gt = pd.read_csv('data/ground_truth.csv')
    syn = pd.read_csv('data/synthetic_ecosystem.csv')
    
    all_crawled = []
    
    # Crawl ground truth domains (real websites)
    for _, row in gt.iterrows():
        domain = row['domain']
        label = int(row['label'])
        
        print(f"  Crawling {domain} (label={label})...")
        data = fetch_page(f"https://{domain}")
        
        if data and data.get('text') and len(data.get('text', '')) > 50:
            data['label'] = label
            all_crawled.append(data)
            print(f"    ✅ {len(data['text'])} chars")
        else:
            # Try www version
            data = fetch_page(f"https://www.{domain}")
            if data and data.get('text') and len(data.get('text', '')) > 50:
                data['domain'] = domain  # Keep original domain name
                data['label'] = label
                all_crawled.append(data)
                print(f"    ✅ {len(data['text'])} chars (www)")
            else:
                print(f"    ⚠️ Skipped (no content)")
        
        import time
        time.sleep(0.5)
    
    # Add synthetic ecosystem domains
    for _, row in syn.iterrows():
        domain = str(row['domain']).strip().lower()
        text = str(row.get('text', ''))
        if text and len(text) > 50:
            all_crawled.append({
                'domain': domain,
                'url': f"https://{domain}",
                'text': text[:2000],
                'links': '',
                'timestamp': '2024-12-15 03:47:00',
                'status': 'synthetic',
                'label': 1,  # synthetic = fake
            })
    
    print(f"\n✅ Total crawled: {len(all_crawled)} domains")
    
    # Count labels
    labels = [d['label'] for d in all_crawled]
    print(f"   Real (0): {labels.count(0)}")
    print(f"   Fake (1): {labels.count(1)}")
    
    return all_crawled


# ══════════════════════════════════════════════════════
# STEP 2: COMPUTE ALL 3 SIGNALS
# ══════════════════════════════════════════════════════

def compute_features(crawled_data):
    """Compute all 3 signals for crawled domains."""
    print()
    print("=" * 55)
    print("STEP 2: Computing features (3 signals)")
    print("=" * 55)
    
    domains = [d['domain'] for d in crawled_data]
    texts = [d['text'][:1000] for d in crawled_data]
    labels = [d['label'] for d in crawled_data]
    
    # ── Signal 1: Content Similarity ──
    print("\n  Computing Signal 1 (content similarity)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    sim_matrix = cosine_similarity(embeddings)
    
    sim_features = {}
    sim_edges = {}
    
    for i in range(len(domains)):
        sims = []
        for j in range(len(domains)):
            if i != j:
                sims.append(float(sim_matrix[i][j]))
                if sim_matrix[i][j] >= SIM_THRESHOLD:  # Store edges for graph
                    pair = tuple(sorted([domains[i], domains[j]]))
                    if pair not in sim_edges:
                        sim_edges[pair] = round(float(sim_matrix[i][j]), 4)
        
        avg_sim = np.mean(sims) if sims else 0
        max_sim = max(sims) if sims else 0
        
        sim_features[domains[i]] = {
            'avg_similarity': round(float(avg_sim), 4),
            'max_similarity': round(float(max_sim), 4),
            'similarity_flag': 1 if max_sim >= SIM_THRESHOLD else 0,
            'similar_domain_count': sum(1 for s in sims if s >= SIM_THRESHOLD),
        }
    
    flagged_sim = sum(1 for v in sim_features.values() if v['similarity_flag'] == 1)
    print(f"  ✅ Signal 1: {flagged_sim} domains with high similarity")
    
    # ── Signal 2: Cadence Anomaly ──
    print("\n  Computing Signal 2 (cadence anomaly)...")
    cadence_features_list = []
    for d in crawled_data:
        ts = d.get('timestamp', '')
        try:
            dt = datetime.strptime(ts[:19], '%Y-%m-%d %H:%M:%S')
            hour, minute, dow = dt.hour, dt.minute, dt.weekday()
        except:
            hour = minute = dow = 0
        
        text_len = len(d.get('text', ''))
        has_links = 1 if d.get('links', '') else 0
        cadence_features_list.append([hour, minute, dow, text_len, has_links])
    
    X_cad = np.array(cadence_features_list, dtype=float)
    clf = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
    clf.fit(X_cad)
    cad_scores = clf.decision_function(X_cad)
    cad_preds = clf.predict(X_cad)
    
    cadence_features = {}
    for i, domain in enumerate(domains):
        cadence_features[domain] = {
            'anomaly_score': round(float(cad_scores[i]), 4),
            'cadence_flagged': 1 if cad_preds[i] == -1 else 0,
            'burst_score': round(float(abs(cad_scores[i])), 4),
        }
    
    flagged_cad = sum(1 for v in cadence_features.values() if v['cadence_flagged'] == 1)
    print(f"  ✅ Signal 2: {flagged_cad} domains with anomalous cadence")
    
    # ── Signal 3: WHOIS / Domain Patterns ──
    print("\n  Computing Signal 3 (domain patterns)...")
    suspicious_tlds = ['.xyz', '.top', '.click', '.online', '.site', '.info', '.buzz', '.icu']
    suspicious_keywords = [
        'truth', 'patriot', 'freedom', 'liberty', 'alert', 'insider',
        'expose', 'breaking', 'real-news', 'updates-now', 'daily-truth',
        'peoples-voice', 'wire', 'first-news', 'national-alert',
        'empire', 'activist', 'conspiracy', 'infowars', 'prison',
        'beforeitsnews', 'naturaln', 'globalresearch'
    ]
    
    whois_features = {}
    for d in crawled_data:
        domain = d['domain'].lower()
        tld_flag = any(domain.endswith(t) for t in suspicious_tlds)
        keyword_flag = any(kw in domain for kw in suspicious_keywords)
        hyphen_flag = domain.count('-') >= 2
        
        flagged = tld_flag or keyword_flag or hyphen_flag
        
        whois_features[domain] = {
            'domain_age_days': -1,
            'whois_flagged': 1 if flagged else 0,
        }
    
    flagged_whois = sum(1 for v in whois_features.values() if v['whois_flagged'] == 1)
    print(f"  ✅ Signal 3: {flagged_whois} domains with suspicious patterns")
    
    # ── Merge all features ──
    print("\n  Merging all features...")
    rows = []
    for i, domain in enumerate(domains):
        s1 = sim_features.get(domain, {})
        s2 = cadence_features.get(domain, {})
        s3 = whois_features.get(domain, {})
        
        sim_flag = s1.get('similarity_flag', 0)
        cad_flag = s2.get('cadence_flagged', 0)
        who_flag = s3.get('whois_flagged', 0)
        signals = sim_flag + cad_flag + who_flag
        
        if signals >= 2:
            verdict = 'SYNTHETIC'
        elif signals == 1:
            verdict = 'REVIEW'
        else:
            verdict = 'ORGANIC'
        
        rows.append({
            'domain': domain,
            'avg_similarity': s1.get('avg_similarity', 0),
            'max_similarity': s1.get('max_similarity', 0),
            'similarity_flag': sim_flag,
            'similar_domain_count': s1.get('similar_domain_count', 0),
            'anomaly_score': s2.get('anomaly_score', 0),
            'burst_score': s2.get('burst_score', 0),
            'hour_variance': 0.0,
            'cadence_flagged': cad_flag,
            'domain_age_days': -1,
            'registrar': 'unknown',
            'whois_flagged': who_flag,
            'label': labels[i],
            'signals_triggered': signals,
            'preliminary_verdict': verdict,
        })
    
    df = pd.DataFrame(rows)
    
    print(f"\n  Feature summary:")
    print(f"    Total domains: {len(df)}")
    print(f"    Signal distribution: {df['signals_triggered'].value_counts().to_dict()}")
    print(f"    Verdict distribution: {df['preliminary_verdict'].value_counts().to_dict()}")
    print(f"    Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df, sim_edges


# ══════════════════════════════════════════════════════
# STEP 3: SAVE FEATURES AND EDGES
# ══════════════════════════════════════════════════════

def save_features(df, sim_edges):
    """Save the new feature dataset and similarity edges."""
    print()
    print("=" * 55)
    print("STEP 3: Saving features")
    print("=" * 55)
    
    # Backup old files
    for f in ['data/domain_features.csv', 'data/similarity_edges.csv']:
        if os.path.exists(f):
            backup = f.replace('.csv', '_backup.csv')
            os.rename(f, backup)
            print(f"  📦 Backed up {f} → {backup}")
    
    # Save new features
    df.to_csv('data/domain_features.csv', index=False)
    print(f"  ✅ Saved data/domain_features.csv ({len(df)} rows)")
    
    # Save similarity edges
    edge_rows = []
    for (d1, d2), sim in sim_edges.items():
        edge_rows.append({'domain_a': d1, 'domain_b': d2, 'similarity': sim, 'flagged': 1 if sim >= SIM_THRESHOLD else 0})
    
    df_edges = pd.DataFrame(edge_rows)
    df_edges.to_csv('data/similarity_edges.csv', index=False)
    print(f"  ✅ Saved data/similarity_edges.csv ({len(df_edges)} rows)")
    
    return df


# ══════════════════════════════════════════════════════
# STEP 4: RETRAIN GNN
# ══════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)

    def forward(self, x, edge_index, n_nodes):
        row, col = edge_index
        agg = torch.zeros(n_nodes, x.shape[1], device=x.device)
        agg.scatter_add_(0, col.unsqueeze(1).expand(-1, x.shape[1]), x[row])
        deg = torch.bincount(col, minlength=n_nodes).float().clamp(min=1)
        return self.linear(agg / deg.unsqueeze(1))


class GCN(nn.Module):
    def __init__(self, in_f, hidden, n_classes, dropout=0.3):
        super().__init__()
        self.conv1 = GCNLayer(in_f, hidden)
        self.conv2 = GCNLayer(hidden, hidden // 2)
        self.linear = nn.Linear(hidden // 2, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, n_nodes):
        x = F.relu(self.conv1(x, edge_index, n_nodes))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, n_nodes))
        x = self.dropout(x)
        return self.linear(x)


def retrain_gnn(df, sim_edges):
    """Retrain the GNN with properly labeled data."""
    print()
    print("=" * 55)
    print("STEP 4: Retraining GNN")
    print("=" * 55)
    
    feature_cols = [
        'avg_similarity', 'max_similarity', 'similarity_flag',
        'similar_domain_count', 'anomaly_score', 'burst_score',
        'hour_variance', 'cadence_flagged', 'whois_flagged',
        'signals_triggered'
    ]
    
    domain_list = df['domain'].tolist()
    domain_to_idx = {d: i for i, d in enumerate(domain_list)}
    n_nodes = len(domain_list)
    
    # Build feature matrix
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Build labels
    labels = df['label'].values.astype(int)
    # Convert: 0=real(organic), 1=fake(synthetic)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Build edge index
    edge_src, edge_dst = [], []
    for (d1, d2), sim in sim_edges.items():
        if d1 in domain_to_idx and d2 in domain_to_idx:
            i, j = domain_to_idx[d1], domain_to_idx[d2]
            edge_src.extend([i, j])
            edge_dst.extend([j, i])
    
    # Self-loops
    for i in range(n_nodes):
        edge_src.append(i)
        edge_dst.append(i)
    
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Labels: 0={sum(labels==0)}, 1={sum(labels==1)}")
    
    # Create model
    model = GCN(in_f=len(feature_cols), hidden=64, n_classes=2, dropout=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Class weights (handle imbalance)
    n_real = sum(labels == 0)
    n_fake = sum(labels == 1)
    weight = torch.tensor([1.0, n_real / max(n_fake, 1)], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    # Train
    print(f"\n  Training GNN...")
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(X_tensor, edge_index, n_nodes)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor, edge_index, n_nodes).argmax(dim=1)
                acc = (pred == y_tensor).float().mean()
                print(f"    Epoch {epoch+1}/200 — Loss: {loss:.4f}, Accuracy: {acc:.2%}")
            model.train()
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        out = model(X_tensor, edge_index, n_nodes)
        probs = F.softmax(out, dim=1)
        pred = out.argmax(dim=1)
        acc = (pred == y_tensor).float().mean()
        
        syn_probs = probs[:, 1].numpy()
        
        print(f"\n  Final accuracy: {acc:.2%}")
        print(f"  Synthetic probability stats:")
        print(f"    Real domains (label=0): mean={syn_probs[labels==0].mean():.3f}, max={syn_probs[labels==0].max():.3f}")
        print(f"    Fake domains (label=1): mean={syn_probs[labels==1].mean():.3f}, min={syn_probs[labels==1].min():.3f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'domain_list': domain_list,
        'domain_to_idx': domain_to_idx,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'accuracy': float(acc),
        'n_domains': n_nodes,
        'trained_at': datetime.now(timezone.utc).isoformat(),
    }
    
    # Backup old model
    if os.path.exists('models/gnn_model.pt'):
        os.rename('models/gnn_model.pt', 'models/gnn_model_backup.pt')
        print(f"  📦 Backed up old model")
    
    torch.save(checkpoint, 'models/gnn_model.pt')
    print(f"  ✅ Saved models/gnn_model.pt")
    
    return model, acc


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("🕸️  Dead Internet Detector — Full Retraining Pipeline")
    print("=" * 55)
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Step 1: Crawl
    crawled = crawl_ground_truth()
    
    if len(crawled) < 20:
        print("❌ Not enough domains crawled. Check internet connection.")
        sys.exit(1)
    
    # Step 2: Compute features
    df, sim_edges = compute_features(crawled)
    
    # Step 3: Save
    save_features(df, sim_edges)
    
    # Step 4: Retrain GNN
    model, accuracy = retrain_gnn(df, sim_edges)
    
    print()
    print("=" * 55)
    print("🎉 RETRAINING COMPLETE!")
    print("=" * 55)
    print(f"  Domains:  {len(df)}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Model:    models/gnn_model.pt")
    print(f"  Features: data/domain_features.csv")
    print(f"  Edges:    data/similarity_edges.csv")
    print()
    print("Now restart the backend and test:")
    print("  uvicorn main:app --reload --port 8000")
    print()
    print(f"Finished at: {datetime.now().strftime('%H:%M:%S')}")