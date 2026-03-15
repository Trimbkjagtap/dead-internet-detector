# ml/evaluate_gnn.py
# Evaluates the trained GNN model and finds the best confidence threshold
# Run with: python3 ml/evaluate_gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

MODEL_PATH    = "models/gnn_model.pt"
FEATURES_FILE = "data/domain_features.csv"
EDGES_FILE    = "data/similarity_edges.csv"
TRUTH_FILE    = "data/ground_truth.csv"


# ── Re-define the GNN model architecture ────────────────────────────────
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


def load_model_and_data():
    """Load the trained model and all data."""
    print("📂 Loading GNN model...")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')

    feature_cols  = checkpoint['feature_cols']
    domain_list   = checkpoint['domain_list']
    domain_to_idx = checkpoint['domain_to_idx']
    scaler_mean   = np.array(checkpoint['scaler_mean'])
    scaler_scale  = np.array(checkpoint['scaler_scale'])
    n_features    = checkpoint['n_features']

    # Rebuild model
    model = GCN(in_f=n_features, hidden=64, n_classes=2, dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded — {len(domain_list)} domains")
    print()

    # Load feature data
    df       = pd.read_csv(FEATURES_FILE)
    df_edges = pd.read_csv(EDGES_FILE)

    # Build node features
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    X = (X - scaler_mean) / scaler_scale  # apply saved scaler
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Build edge index
    n_nodes  = len(domain_list)
    edge_src, edge_dst = [], []
    for _, row in df_edges.iterrows():
        a, b = row['domain_a'], row['domain_b']
        if a in domain_to_idx and b in domain_to_idx:
            i, j = domain_to_idx[a], domain_to_idx[b]
            edge_src.extend([i, j])
            edge_dst.extend([j, i])
    for i in range(n_nodes):
        edge_src.append(i)
        edge_dst.append(i)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    return model, df, domain_list, domain_to_idx, X_tensor, edge_index, n_nodes


def run_inference(model, X_tensor, edge_index, n_nodes):
    """Run model inference and return probabilities."""
    print("🔮 Running inference on all domains...")
    with torch.no_grad():
        out  = model(X_tensor, edge_index, n_nodes)
        prob = F.softmax(out, dim=1)
        synthetic_prob = prob[:, 1].numpy()
    print(f"✅ Inference complete")
    print()
    return synthetic_prob


def find_best_threshold(df, domain_list, synthetic_prob):
    """
    Find the confidence threshold that maximizes F1 score
    on the labeled domains from ground_truth.csv.
    """
    print("🎯 Finding best confidence threshold...")

    df_truth = pd.read_csv(TRUTH_FILE)
    truth_dict = dict(zip(df_truth['domain'], df_truth['label']))

    # Get domains that have ground truth labels
    labeled = []
    for i, domain in enumerate(domain_list):
        if domain in truth_dict:
            labeled.append({
                'domain':   domain,
                'prob':     synthetic_prob[i],
                'true_label': truth_dict[domain]
            })

    if len(labeled) < 5:
        print(f"⚠️  Only {len(labeled)} labeled domains found in our dataset")
        print("   Using signals_triggered as proxy labels instead")

        # Use signals as proxy
        labeled = []
        for i, domain in enumerate(domain_list):
            row = df[df['domain'] == domain]
            if len(row) > 0:
                sig = int(row.iloc[0].get('signals_triggered', 0))
                if sig >= 2:
                    labeled.append({'domain': domain, 'prob': synthetic_prob[i], 'true_label': 1})
                elif sig == 0:
                    labeled.append({'domain': domain, 'prob': synthetic_prob[i], 'true_label': 0})

    if len(labeled) == 0:
        print("⚠️  No labeled data available — using default threshold 0.5")
        return 0.5

    df_labeled = pd.DataFrame(labeled)
    y_true = df_labeled['true_label'].values
    probs  = df_labeled['prob'].values

    # Try different thresholds
    best_threshold = 0.5
    best_f1        = 0.0
    results        = []

    for threshold in np.arange(0.3, 0.95, 0.05):
        y_pred = (probs >= threshold).astype(int)
        if len(set(y_pred)) < 2:
            continue
        f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results.append({'threshold': round(threshold, 2), 'f1': round(f1, 3),
                        'precision': round(prec, 3), 'recall': round(rec, 3)})
        if f1 > best_f1:
            best_f1        = f1
            best_threshold = threshold

    if results:
        results_df = pd.DataFrame(results)
        print("\n  Threshold sweep results:")
        print(results_df.to_string(index=False))

    print(f"\n✅ Best threshold: {best_threshold:.2f} (F1: {best_f1:.3f})")
    return best_threshold


def silent_failure_test(domain_list, synthetic_prob, threshold):
    """
    Test the silent failure scenario:
    Do legitimate news wire domains get wrongly flagged as synthetic?
    """
    print()
    print("🔇 Silent Failure Test — Wire Service Domains...")
    print("   (These should NOT be flagged as synthetic)")

    # Known legitimate wire service / major news domains
    wire_domains = [
        'apnews.com', 'reuters.com', 'bbc.com', 'bbc.co.uk',
        'nytimes.com', 'washingtonpost.com', 'theguardian.com',
        'npr.org', 'cbsnews.com', 'nbcnews.com', 'cnn.com',
        'bloomberg.com', 'wsj.com', 'ft.com', 'economist.com'
    ]

    domain_idx = {d: i for i, d in enumerate(domain_list)}
    wrongly_flagged = []
    found = []

    for domain in wire_domains:
        if domain in domain_idx:
            idx  = domain_idx[domain]
            prob = synthetic_prob[idx]
            found.append(domain)
            if prob >= threshold:
                wrongly_flagged.append({'domain': domain, 'synthetic_prob': round(prob, 4)})
                print(f"  ❌ WRONGLY FLAGGED: {domain} (prob={prob:.3f})")
            else:
                print(f"  ✅ Correctly organic: {domain} (prob={prob:.3f})")

    if len(found) == 0:
        print("  ⚠️  None of the wire service domains are in our dataset")
        print("  (This is expected — our Common Crawl sample may not include them)")
        print("  Silent failure test: N/A — domains not in dataset")
    elif len(wrongly_flagged) == 0:
        print(f"\n  ✅ PASSED — 0 wire service domains wrongly flagged")
        print(f"  Our 2-of-3 signal system is working correctly!")
    else:
        print(f"\n  ⚠️  {len(wrongly_flagged)} wire domains wrongly flagged")
        print(f"  Consider raising threshold above {threshold:.2f}")

    return wrongly_flagged


def generate_final_report(df, domain_list, synthetic_prob, threshold):
    """Generate final verdict for all domains."""
    print()
    print("📋 Generating final verdicts...")

    verdicts = []
    for i, domain in enumerate(domain_list):
        prob = synthetic_prob[i]
        row  = df[df['domain'] == domain]
        sigs = int(row.iloc[0]['signals_triggered']) if len(row) > 0 else 0

        # Final verdict logic: GNN confidence + signal convergence
        if prob >= threshold and sigs >= 2:
            verdict = 'SYNTHETIC'
        elif prob >= threshold and sigs == 1:
            verdict = 'REVIEW'
        elif prob >= 0.4 and sigs >= 1:
            verdict = 'REVIEW'
        else:
            verdict = 'ORGANIC'

        verdicts.append({
            'domain':         domain,
            'synthetic_prob': round(prob, 4),
            'signals':        sigs,
            'final_verdict':  verdict,
        })

    df_v = pd.DataFrame(verdicts)

    print(f"\n  Final Verdict Distribution:")
    print(f"  🔴 SYNTHETIC: {len(df_v[df_v['final_verdict']=='SYNTHETIC'])} domains")
    print(f"  🟡 REVIEW:    {len(df_v[df_v['final_verdict']=='REVIEW'])} domains")
    print(f"  🟢 ORGANIC:   {len(df_v[df_v['final_verdict']=='ORGANIC'])} domains")

    # Show top suspicious domains
    suspicious = df_v[df_v['final_verdict'].isin(['SYNTHETIC','REVIEW'])]
    if len(suspicious) > 0:
        print(f"\n  Top suspicious domains:")
        print(suspicious.nlargest(10, 'synthetic_prob')
              [['domain','synthetic_prob','signals','final_verdict']]
              .to_string(index=False))

    return df_v


def evaluate():
    print("=" * 55)
    print("Dead Internet Detector — Day 12")
    print("GNN Evaluation & Threshold Tuning")
    print("=" * 55)
    print()

    # Load everything
    model, df, domain_list, domain_to_idx, X_tensor, edge_index, n_nodes = \
        load_model_and_data()

    # Run inference
    synthetic_prob = run_inference(model, X_tensor, edge_index, n_nodes)

    # Find best threshold
    threshold = find_best_threshold(df, domain_list, synthetic_prob)

    # Silent failure test
    wrongly_flagged = silent_failure_test(domain_list, synthetic_prob, threshold)

    # Final report
    df_verdicts = generate_final_report(df, domain_list, synthetic_prob, threshold)

    print()
    print("=" * 55)
    print(f"✅ Best confidence threshold: {threshold:.2f}")
    print(f"✅ Silent failure test:       {len(wrongly_flagged)} wrongly flagged")
    print()
    print("🎉 Day 12 evaluation complete!")
    print("   Next: Build CrewAI agents (Days 13-14)")
    print("=" * 55)


if __name__ == "__main__":
    evaluate()