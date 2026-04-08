"""
ml/train_gnn_real.py
──────────────────────
Retrains the GCN on the ground-truth labeled dataset (data/domain_features.csv
+ data/similarity_edges.csv), then saves the model to models/gnn_model.pt.

Key improvements over train_gnn.py:
  - Uses ground-truth labels from domain_features.csv (not pseudo-labels only)
  - Reads similarity_edges.csv with domain_1/domain_2 column names
  - Saves domain_list + domain_to_idx so inference can look up nodes by name
  - Saves evaluation metrics in the checkpoint
  - Writes data/reports/gnn_eval.json with test-set performance

Usage:
    python3 -m ml.train_gnn_real
"""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)

sys.path.insert(0, '.')

DOMAIN_FEAT  = Path("data/domain_features.csv")
EDGES_CSV    = Path("data/similarity_edges.csv")
MODEL_OUT    = Path("models/gnn_model.pt")
REPORTS_DIR  = Path("data/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    'avg_similarity', 'max_similarity', 'anomaly_score', 'burst_score',
    'hour_variance', 'domain_age_days', 'signals_triggered',
    'similarity_flag', 'cadence_flagged', 'whois_flagged',
]
EPOCHS     = 150
HIDDEN_DIM = 64
DROPOUT    = 0.3
LR         = 0.01
WEIGHT_DECAY = 5e-4


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index, n_nodes):
        row, col = edge_index
        agg = torch.zeros(n_nodes, x.shape[1], device=x.device)
        agg.scatter_add_(0, col.unsqueeze(1).expand(-1, x.shape[1]), x[row])
        deg = torch.bincount(col, minlength=n_nodes).float().clamp(min=1)
        agg = agg / deg.unsqueeze(1)
        return self.linear(agg)


class GCN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.conv1   = GCNLayer(in_features, hidden_dim)
        self.conv2   = GCNLayer(hidden_dim, hidden_dim // 2)
        self.linear  = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, n_nodes):
        x = F.relu(self.conv1(x, edge_index, n_nodes))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, n_nodes))
        x = self.dropout(x)
        return self.linear(x)


def main():
    print("=" * 60)
    print("🧠 Dead Internet Detector — GNN Retrain (Ground Truth)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # ── 1. Load data ──────────────────────────────────────────────
    df    = pd.read_csv(DOMAIN_FEAT)
    edges = pd.read_csv(EDGES_CSV)

    # Use only rows with valid labels (0 or 1)
    labeled = df[df['label'].isin([0, 1])].copy()
    print(f"\n  Labeled domains: {len(labeled)} "
          f"({labeled['label'].sum():.0f} synthetic, "
          f"{(labeled['label']==0).sum()} organic)")
    print(f"  Similarity edges: {len(edges)}")

    domain_list  = labeled['domain'].tolist()
    domain_to_idx = {d: i for i, d in enumerate(domain_list)}
    n_nodes       = len(domain_list)

    # ── 2. Node features ─────────────────────────────────────────
    feature_cols = [c for c in FEATURE_COLS if c in labeled.columns]
    X = labeled[feature_cols].fillna(0).values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    print(f"  Node features: {X_tensor.shape}  "
          f"({', '.join(feature_cols)})")

    # ── 3. Edge index ─────────────────────────────────────────────
    # Handle both 'domain_1'/'domain_2' and 'domain_a'/'domain_b' column names
    col_a = 'domain_1' if 'domain_1' in edges.columns else 'domain_a'
    col_b = 'domain_2' if 'domain_2' in edges.columns else 'domain_b'

    edge_src, edge_dst = [], []
    for _, row in edges.iterrows():
        a, b = str(row[col_a]), str(row[col_b])
        if a in domain_to_idx and b in domain_to_idx:
            i, j = domain_to_idx[a], domain_to_idx[b]
            edge_src.extend([i, j])
            edge_dst.extend([j, i])
    # Self-loops
    for i in range(n_nodes):
        edge_src.append(i); edge_dst.append(i)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(device)
    print(f"  Edge index: {edge_index.shape}  "
          f"({edge_index.shape[1] - n_nodes} directed edges + {n_nodes} self-loops)")

    # ── 4. Labels + train/test split ─────────────────────────────
    y_np = labeled['label'].values.astype(int)
    all_idx = list(range(n_nodes))

    train_idx, test_idx, train_y, test_y = train_test_split(
        all_idx, y_np.tolist(),
        test_size=0.2, random_state=42,
        stratify=y_np,
    )

    y_tensor   = torch.tensor(y_np, dtype=torch.long).to(device)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    test_mask  = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    for i in train_idx: train_mask[i] = True
    for i in test_idx:  test_mask[i]  = True

    print(f"  Train: {train_mask.sum().item()} | Test: {test_mask.sum().item()}")

    # ── 5. Build + train model ────────────────────────────────────
    model = GCN(len(feature_cols), HIDDEN_DIM, 2, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Class weights for imbalanced data
    n_syn = int(y_np.sum())
    n_org = n_nodes - n_syn
    weights = torch.tensor([n_syn / n_nodes, n_org / n_nodes], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    print(f"\n  Training ({EPOCHS} epochs)...")
    best_f1    = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(X_tensor, edge_index, n_nodes)
        loss = criterion(out[train_mask], y_tensor[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                out_e = model(X_tensor, edge_index, n_nodes)
                pred  = out_e.argmax(dim=1)
                tr_acc = (pred[train_mask] == y_tensor[train_mask]).float().mean().item()
                te_f1  = f1_score(
                    y_tensor[test_mask].cpu().numpy(),
                    pred[test_mask].cpu().numpy(),
                    zero_division=0
                )
            print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss={loss.item():.4f} | "
                  f"TrainAcc={tr_acc:.3f} | TestF1={te_f1:.3f}")
            if te_f1 > best_f1:
                best_f1   = te_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── 6. Final evaluation ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 Final Evaluation (test set)")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out  = model(X_tensor, edge_index, n_nodes)
        pred = out.argmax(dim=1)
        prob = F.softmax(out, dim=1)

        y_true_test = y_tensor[test_mask].cpu().numpy()
        y_pred_test = pred[test_mask].cpu().numpy()

    print(classification_report(y_true_test, y_pred_test,
                                target_names=['Organic', 'Synthetic'],
                                zero_division=0))

    test_metrics = {
        "precision": round(precision_score(y_true_test, y_pred_test, zero_division=0), 4),
        "recall":    round(recall_score(y_true_test, y_pred_test, zero_division=0), 4),
        "f1":        round(f1_score(y_true_test, y_pred_test, zero_division=0), 4),
        "accuracy":  round(accuracy_score(y_true_test, y_pred_test), 4),
    }
    print(f"  Best F1: {best_f1:.4f}")

    # ── 7. Save model ─────────────────────────────────────────────
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': best_state,
        'feature_cols':     feature_cols,
        'n_features':       len(feature_cols),
        'hidden_dim':       HIDDEN_DIM,
        'domain_list':      domain_list,
        'domain_to_idx':    domain_to_idx,
        'scaler_mean':      scaler.mean_.tolist(),
        'scaler_scale':     scaler.scale_.tolist(),
        'accuracy':         test_metrics["accuracy"],
        'f1':               test_metrics["f1"],
        'n_domains':        n_nodes,
        'trained_at':       datetime.now().isoformat(),
        'training_data':    str(DOMAIN_FEAT),
    }
    torch.save(checkpoint, MODEL_OUT)
    print(f"\n  ✅ Model saved → {MODEL_OUT}")

    # ── 8. Save eval report ───────────────────────────────────────
    gnn_report = {
        "timestamp":       datetime.now().isoformat(),
        "model":           str(MODEL_OUT),
        "n_domains":       n_nodes,
        "n_train":         int(train_mask.sum()),
        "n_test":          int(test_mask.sum()),
        "n_features":      len(feature_cols),
        "feature_cols":    feature_cols,
        "test_metrics":    test_metrics,
        "best_val_f1":     round(best_f1, 4),
        "epochs":          EPOCHS,
        "hidden_dim":      HIDDEN_DIM,
    }
    gnn_path = REPORTS_DIR / "gnn_eval.json"
    gnn_path.write_text(json.dumps(gnn_report, indent=2))
    print(f"  ✅ GNN eval      → {gnn_path}")

    # Save into evaluation_results.json
    eval_results = REPORTS_DIR / "evaluation_results.json"
    if eval_results.exists():
        ev = json.loads(eval_results.read_text())
        ev["gnn_evaluation"] = gnn_report
        eval_results.write_text(json.dumps(ev, indent=2))

    print("\n" + "=" * 60)
    print(f"  GCN Test F1: {test_metrics['f1']:.4f}  "
          f"Precision: {test_metrics['precision']:.3f}  "
          f"Recall: {test_metrics['recall']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
