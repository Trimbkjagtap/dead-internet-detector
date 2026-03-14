# ml/train_gnn.py
# Trains a Graph Convolutional Network (GCN) on domain graph data
# Run this on Google Colab with T4 GPU for best performance
# Can also run locally on CPU (slower but works)

# ── Install required libraries (run this first in Colab) ──────────────
# !pip install torch torch-geometric scikit-learn pandas numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

print("=" * 55)
print("Dead Internet Detector — Day 11")
print("Training Graph Neural Network...")
print("=" * 55)
print()

# ── Check GPU availability ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# ══════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════

print("📂 Loading data...")

# Load feature table
df = pd.read_csv('domain_features.csv')  # Colab path
df_edges = pd.read_csv('similarity_edges.csv')

print(f"✅ Domains loaded:        {len(df)}")
print(f"✅ Similarity edges:      {len(df_edges)}")
print()

# ── Build domain index ──
# Each domain needs a unique integer ID for the GNN
domain_list = df['domain'].tolist()
domain_to_idx = {d: i for i, d in enumerate(domain_list)}
n_nodes = len(domain_list)

# ══════════════════════════════════════════════════════
# 2. BUILD NODE FEATURES
# ══════════════════════════════════════════════════════

print("🔢 Building node features...")

# Select numeric feature columns for the GNN
feature_cols = [
    'avg_similarity',
    'max_similarity',
    'anomaly_score',
    'burst_score',
    'hour_variance',
    'domain_age_days',
    'signals_triggered',
    'similarity_flag',
    'cadence_flagged',
    'whois_flagged',
]

# Only use columns that exist
feature_cols = [c for c in feature_cols if c in df.columns]
print(f"  Feature columns used: {feature_cols}")

X = df[feature_cols].fillna(0).values.astype(np.float32)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to tensor
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
print(f"✅ Node feature matrix: {X_tensor.shape}")
print()

# ══════════════════════════════════════════════════════
# 3. BUILD EDGE INDEX
# ══════════════════════════════════════════════════════

print("🔗 Building edge index...")

edge_src = []
edge_dst = []

for _, row in df_edges.iterrows():
    a = row['domain_a']
    b = row['domain_b']
    if a in domain_to_idx and b in domain_to_idx:
        i = domain_to_idx[a]
        j = domain_to_idx[b]
        # Add both directions (undirected graph)
        edge_src.extend([i, j])
        edge_dst.extend([j, i])

# Add self-loops (each node connects to itself)
for i in range(n_nodes):
    edge_src.append(i)
    edge_dst.append(i)

edge_index = torch.tensor(
    [edge_src, edge_dst],
    dtype=torch.long
).to(device)

print(f"✅ Edge index shape: {edge_index.shape}")
print(f"✅ Total edges (incl. self-loops): {edge_index.shape[1]}")
print()

# ══════════════════════════════════════════════════════
# 4. BUILD LABELS
# ══════════════════════════════════════════════════════

print("🏷️  Building labels...")

# Use ground truth labels where available
# label=1 (fake/synthetic), label=0 (real/organic), label=-1 (unknown)
labels = df['label'].fillna(-1).astype(int).tolist()

# Also use signals_triggered as pseudo-labels for unlabeled nodes
# 2+ signals = likely synthetic (pseudo label 1)
# 0 signals = likely organic (pseudo label 0)
# 1 signal = uncertain (skip)
pseudo_labels = []
for i, row in df.iterrows():
    true_label = labels[i]
    if true_label != -1:
        pseudo_labels.append(true_label)
    else:
        sig = int(row.get('signals_triggered', 0))
        if sig >= 2:
            pseudo_labels.append(1)
        elif sig == 0:
            pseudo_labels.append(0)
        else:
            pseudo_labels.append(-1)  # uncertain — skip in training

# Get indices with valid labels
labeled_idx = [i for i, l in enumerate(pseudo_labels) if l != -1]
labeled_y   = [pseudo_labels[i] for i in labeled_idx]

print(f"✅ Total labeled nodes:    {len(labeled_idx)}")
print(f"   Synthetic (label=1):   {sum(1 for l in labeled_y if l == 1)}")
print(f"   Organic   (label=0):   {sum(1 for l in labeled_y if l == 0)}")
print()

# Split into train/val/test
if len(labeled_idx) < 10:
    print("⚠️  Very few labeled nodes — using all for training")
    train_idx = labeled_idx
    test_idx  = labeled_idx
else:
    train_idx, test_idx, train_y, test_y = train_test_split(
        labeled_idx, labeled_y,
        test_size=0.2,
        random_state=42,
        stratify=labeled_y if len(set(labeled_y)) > 1 else None
    )

# Convert labels to tensors
y_all = torch.tensor(pseudo_labels, dtype=torch.long).to(device)
train_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
test_mask  = torch.zeros(n_nodes, dtype=torch.bool).to(device)
for i in train_idx:
    train_mask[i] = True
for i in test_idx:
    test_mask[i] = True

print(f"✅ Train nodes: {train_mask.sum().item()}")
print(f"✅ Test nodes:  {test_mask.sum().item()}")
print()

# ══════════════════════════════════════════════════════
# 5. DEFINE GNN MODEL
# ══════════════════════════════════════════════════════

print("🧠 Defining GNN model...")


class GCNLayer(nn.Module):
    """Simple Graph Convolutional Network layer."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index, n_nodes):
        # Aggregate neighbor features
        row, col = edge_index
        # Sum neighbor features for each node
        agg = torch.zeros(n_nodes, x.shape[1], device=x.device)
        agg.scatter_add_(0, col.unsqueeze(1).expand(-1, x.shape[1]), x[row])
        # Normalize by degree
        deg = torch.bincount(col, minlength=n_nodes).float().clamp(min=1)
        agg = agg / deg.unsqueeze(1)
        return self.linear(agg)


class GCN(nn.Module):
    """2-layer Graph Convolutional Network."""

    def __init__(self, in_features, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.conv1   = GCNLayer(in_features, hidden_dim)
        self.conv2   = GCNLayer(hidden_dim, hidden_dim // 2)
        self.linear  = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, n_nodes):
        # Layer 1
        x = self.conv1(x, edge_index, n_nodes)
        x = F.relu(x)
        x = self.dropout(x)
        # Layer 2
        x = self.conv2(x, edge_index, n_nodes)
        x = F.relu(x)
        x = self.dropout(x)
        # Output
        x = self.linear(x)
        return x


in_features = X_tensor.shape[1]
model = GCN(
    in_features=in_features,
    hidden_dim=64,
    num_classes=2,
    dropout=0.3
).to(device)

print(f"✅ Model created:")
print(f"   Input features:  {in_features}")
print(f"   Hidden dim:      64")
print(f"   Output classes:  2 (organic / synthetic)")
total_params = sum(p.numel() for p in model.parameters())
print(f"   Total params:    {total_params}")
print()

# ══════════════════════════════════════════════════════
# 6. TRAIN
# ══════════════════════════════════════════════════════

print("🚀 Training...")
print()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

EPOCHS = 100
best_acc = 0.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    out  = model(X_tensor, edge_index, n_nodes)
    loss = criterion(out[train_mask], y_all[train_mask])

    loss.backward()
    optimizer.step()

    # Evaluate every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            out_eval = model(X_tensor, edge_index, n_nodes)
            pred     = out_eval.argmax(dim=1)

            train_acc = (pred[train_mask] == y_all[train_mask]).float().mean().item()
            test_acc  = (pred[test_mask]  == y_all[test_mask]).float().mean().item() \
                        if test_mask.sum() > 0 else 0.0

        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Loss: {loss.item():.4f} | "
              f"Train Acc: {train_acc:.3f} | "
              f"Test Acc: {test_acc:.3f}")

        if train_acc > best_acc:
            best_acc   = train_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

print()

# ══════════════════════════════════════════════════════
# 7. EVALUATE
# ══════════════════════════════════════════════════════

print("📊 Final Evaluation...")
model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    out  = model(X_tensor, edge_index, n_nodes)
    pred = out.argmax(dim=1)
    prob = F.softmax(out, dim=1)

    if test_mask.sum() > 0:
        y_true = y_all[test_mask].cpu().numpy()
        y_pred = pred[test_mask].cpu().numpy()
        print(classification_report(y_true, y_pred,
              target_names=['Organic', 'Synthetic'], zero_division=0))
    else:
        print("⚠️  No test samples — evaluating on training set")
        y_true = y_all[train_mask].cpu().numpy()
        y_pred = pred[train_mask].cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        print(f"Train Accuracy: {acc:.4f}")
        print(f"Train F1 Score: {f1:.4f}")

# ══════════════════════════════════════════════════════
# 8. SAVE MODEL
# ══════════════════════════════════════════════════════

print()
print("💾 Saving model...")

save_dict = {
    'model_state_dict': best_state,
    'feature_cols':     feature_cols,
    'n_features':       in_features,
    'domain_list':      domain_list,
    'domain_to_idx':    domain_to_idx,
    'scaler_mean':      scaler.mean_.tolist(),
    'scaler_scale':     scaler.scale_.tolist(),
}

torch.save(save_dict, 'gnn_model.pt')
print("✅ Model saved: gnn_model.pt")
print()

# Show predictions for all domains
all_preds  = pred.cpu().numpy()
all_probs  = prob[:, 1].cpu().numpy()  # probability of being synthetic

df['gnn_prediction'] = all_preds
df['synthetic_prob'] = np.round(all_probs, 4)
df['gnn_verdict']    = df['gnn_prediction'].map({0: 'ORGANIC', 1: 'SYNTHETIC'})

print("Sample predictions:")
print(df[['domain', 'synthetic_prob', 'gnn_verdict',
          'signals_triggered']].head(10).to_string(index=False))
print()

synthetic_found = (df['gnn_prediction'] == 1).sum()
print(f"✅ Domains classified as SYNTHETIC: {synthetic_found}")
print(f"✅ Domains classified as ORGANIC:   {(df['gnn_prediction'] == 0).sum()}")
print()
print("=" * 55)
print("🎉 GNN training complete!")
print("   Download gnn_model.pt and save to your models/ folder")
print("=" * 55)