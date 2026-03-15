# agents/verdict_agent.py
# Agent 4 — Verdict Agent
# Runs GNN inference and produces final verdict with explanation

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH  = "models/gnn_model.pt"
THRESHOLD   = 0.5   # confidence threshold


# ── Re-define GNN architecture ───────────────────────
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
        self.conv1   = GCNLayer(in_f, hidden)
        self.conv2   = GCNLayer(hidden, hidden // 2)
        self.linear  = nn.Linear(hidden // 2, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, n_nodes):
        x = F.relu(self.conv1(x, edge_index, n_nodes))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, n_nodes))
        x = self.dropout(x)
        return self.linear(x)


def load_gnn_model():
    """Load the trained GNN model from disk."""
    checkpoint    = torch.load(MODEL_PATH, map_location='cpu')
    feature_cols  = checkpoint['feature_cols']
    n_features    = checkpoint['n_features']
    scaler_mean   = np.array(checkpoint['scaler_mean'])
    scaler_scale  = np.array(checkpoint['scaler_scale'])

    model = GCN(in_f=n_features, hidden=64, n_classes=2, dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, feature_cols, scaler_mean, scaler_scale


def build_graph_tensors(features: dict, sim_edges: dict):
    """Convert feature dict into PyTorch tensors for GNN inference."""
    domain_list   = list(features.keys())
    domain_to_idx = {d: i for i, d in enumerate(domain_list)}
    n_nodes       = len(domain_list)

    # Load model to get feature column order
    _, feature_cols, scaler_mean, scaler_scale = load_gnn_model()

    # Build feature matrix
    X = []
    for domain in domain_list:
        feats = features[domain]
        row   = [float(feats.get(c, 0)) for c in feature_cols]
        X.append(row)

    X = np.array(X, dtype=np.float32)
    X = (X - scaler_mean) / scaler_scale
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Build edge index
    edge_src, edge_dst = [], []
    for key in sim_edges:
        if "|||" in str(key):
            a, b = str(key).split("|||")
        elif isinstance(key, (list, tuple)) and len(key) == 2:
            a, b = key
        else:
            continue
        if a in domain_to_idx and b in domain_to_idx:
            i, j = domain_to_idx[a], domain_to_idx[b]
            edge_src.extend([i, j])
            edge_dst.extend([j, i])

    # Self-loops
    for i in range(n_nodes):
        edge_src.append(i)
        edge_dst.append(i)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    return X_tensor, edge_index, n_nodes, domain_list


def generate_explanation(domain: str, feats: dict, prob: float) -> str:
    """Generate plain-English explanation for the verdict."""
    reasons = []

    if feats.get('similarity_flag', 0):
        reasons.append(
            f"content similarity score {feats.get('max_similarity',0):.2f} "
            f"exceeds threshold (Signal 1 triggered)"
        )
    if feats.get('cadence_flagged', 0):
        reasons.append(
            f"publishing cadence anomaly detected — "
            f"burst score {feats.get('burst_score',0):.2f} (Signal 2 triggered)"
        )
    if feats.get('whois_flagged', 0):
        reasons.append("suspicious domain registration pattern (Signal 3 triggered)")

    if not reasons:
        return (f"{domain} shows no suspicious patterns across all 3 signals. "
                f"Content similarity, publishing cadence, and registration data "
                f"all appear organic.")

    signals = feats.get('signals_triggered', 0)
    return (
        f"{domain} triggered {signals}/3 signals: "
        + "; ".join(reasons) + ". "
        + (f"With {signals} signals converging and GNN confidence {prob:.0%}, "
           f"the 2-of-3 rule requires verdict escalation."
           if signals >= 2 else
           f"Single signal — flagged for human review only.")
    )


def produce_verdict(features: dict, sim_edges: dict) -> dict:
    """
    Main verdict function.
    Runs GNN inference and produces final verdict for all domains.

    Args:
        features: dict of domain → feature scores from Fingerprint Analyst
        sim_edges: dict of domain pair → similarity score

    Returns:
        dict with verdict, confidence, signals, explanation per domain
    """
    print(f"⚖️  Verdict Agent starting...")
    print(f"   Domains to evaluate: {len(features)}")

    if not features:
        return {'error': 'No domain features provided', 'verdicts': {}}

    try:
        # Load model
        model, feature_cols, scaler_mean, scaler_scale = load_gnn_model()
        print(f"   ✅ GNN model loaded")

        # Build tensors
        X_tensor, edge_index, n_nodes, domain_list = build_graph_tensors(
            features, sim_edges
        )

        # Run inference
        with torch.no_grad():
            out  = model(X_tensor, edge_index, n_nodes)
            prob = F.softmax(out, dim=1)
            synthetic_probs = prob[:, 1].numpy()

        print(f"   ✅ GNN inference complete")

        # Build verdict per domain
        verdicts = {}
        for i, domain in enumerate(domain_list):
            feats        = features[domain]
            syn_prob     = float(synthetic_probs[i])
            signals      = int(feats.get('signals_triggered', 0))

            # 2-of-3 rule
            if syn_prob >= THRESHOLD and signals >= 2:
                verdict = 'SYNTHETIC'
            elif syn_prob >= THRESHOLD and signals == 1:
                verdict = 'REVIEW'
            elif signals >= 2:
                verdict = 'REVIEW'
            else:
                verdict = 'ORGANIC'

            explanation = generate_explanation(domain, feats, syn_prob)

            verdicts[domain] = {
                'verdict':           verdict,
                'confidence':        round(syn_prob, 4),
                'signals_triggered': signals,
                'signal_1_similarity': int(feats.get('similarity_flag', 0)),
                'signal_2_cadence':    int(feats.get('cadence_flagged', 0)),
                'signal_3_whois':      int(feats.get('whois_flagged', 0)),
                'explanation':       explanation,
            }

        # Summary verdict for the whole cluster
        synthetic_count = sum(1 for v in verdicts.values() if v['verdict'] == 'SYNTHETIC')
        review_count    = sum(1 for v in verdicts.values() if v['verdict'] == 'REVIEW')
        max_confidence  = max(v['confidence'] for v in verdicts.values())

        if synthetic_count > 0:
            cluster_verdict = 'SYNTHETIC'
        elif review_count > 0:
            cluster_verdict = 'REVIEW'
        else:
            cluster_verdict = 'ORGANIC'

        result = {
            'cluster_verdict':  cluster_verdict,
            'max_confidence':   round(max_confidence, 4),
            'synthetic_domains': synthetic_count,
            'review_domains':   review_count,
            'organic_domains':  len(verdicts) - synthetic_count - review_count,
            'domain_verdicts':  verdicts,
        }

        print(f"✅ Verdict Agent complete!")
        print(f"   Cluster verdict: {cluster_verdict}")
        print(f"   Max confidence:  {max_confidence:.3f}")
        return result

    except Exception as e:
        print(f"   ❌ Verdict Agent error: {e}")
        return {'error': str(e), 'verdicts': {}}


def produce_verdict_tool(input_json: str) -> str:
    """CrewAI tool wrapper."""
    data      = json.loads(input_json)
    features  = data.get('features', {})
    sim_edges = data.get('sim_edges', {})
    result    = produce_verdict(features, sim_edges)
    return json.dumps(result)


# ── Standalone test ──────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from agents.crawler_agent import crawl_domains
    from agents.fingerprint_agent import analyze_domains

    print("Testing Verdict Agent...")
    print()

    data     = crawl_domains(["example.com", "bbc.com"])
    analysis = analyze_domains(data)
    verdict  = produce_verdict(
        analysis['features'],
        analysis.get('sim_edges', {})
    )

    print()
    print("Cluster verdict:", verdict['cluster_verdict'])
    print("Max confidence: ", verdict['max_confidence'])
    print()
    for domain, v in verdict['domain_verdicts'].items():
        print(f"  {domain}: {v['verdict']} (conf={v['confidence']:.3f})")
    print()
    print("🎉 Verdict Agent test passed!")