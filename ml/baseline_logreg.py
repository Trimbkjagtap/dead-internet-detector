"""
ml/baseline_logreg.py
──────────────────────
Trains a Logistic Regression on the 7 binary signal flags from the
evaluation log and compares it against the 3-of-7 rule and GCN.

Reads:  data/reports/evaluation_log.csv   (produced by evaluate_ground_truth.py)
Writes: data/reports/model_comparison.csv
        data/reports/roc_curve.png

Usage:
    python3 -m ml.baseline_logreg
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, '.')

REPORTS_DIR  = Path("data/reports")
LOG_CSV      = REPORTS_DIR / "evaluation_log.csv"
RESULTS_JSON = REPORTS_DIR / "evaluation_results.json"

SIGNAL_COLS = [
    "signal_1_similarity",
    "signal_2_cadence",
    "signal_3_whois",
    "signal_4_hosting",
    "signal_5_link_network",
    "signal_6_wayback",
    "signal_7_authors",
]


def compute_metrics(y_true, y_pred, y_prob=None):
    from sklearn.metrics import (precision_score, recall_score,
                                 f1_score, accuracy_score, roc_auc_score)
    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    return {
        "precision": round(p, 4),
        "recall":    round(r, 4),
        "f1":        round(f1, 4),
        "accuracy":  round(acc, 4),
        "auc_roc":   round(auc, 4) if auc is not None else "N/A",
    }


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_curve

    print("=" * 60)
    print("📈 Dead Internet Detector — Baseline Model Comparison")
    print("=" * 60)

    if not LOG_CSV.exists():
        print(f"❌ {LOG_CSV} not found. Run evaluate_ground_truth.py first.")
        sys.exit(1)

    df = pd.read_csv(LOG_CSV)
    for col in SIGNAL_COLS:
        if col not in df.columns:
            df[col] = 0

    X = df[SIGNAL_COLS].values.astype(float)
    y = df["label"].values.astype(int)

    print(f"  Dataset: {len(df)} domains  "
          f"({y.sum()} synthetic, {(y==0).sum()} organic)\n")

    results = {}

    # ── Baseline 1: 3-of-7 Rule (already evaluated) ──────────────
    y_pred_rule = df["predicted_label"].tolist()
    # 3-of-7 has no probability, use signal count / 7 as proxy score
    y_prob_rule = (df[SIGNAL_COLS].sum(axis=1) / 7).tolist()
    m_rule = compute_metrics(y, y_pred_rule, y_prob_rule)
    results["3-of-7 Rule"] = m_rule
    print(f"  3-of-7 Rule      F1={m_rule['f1']:.4f}  "
          f"P={m_rule['precision']:.3f}  R={m_rule['recall']:.3f}  "
          f"AUC={m_rule['auc_roc']}")

    # ── Baseline 2: Logistic Regression (5-fold CV) ────────────────
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=42, solver="lbfgs")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_prob_lr   = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred_lr   = (y_prob_lr >= 0.5).astype(int)
    m_lr = compute_metrics(y, y_pred_lr, y_prob_lr)
    results["Logistic Regression (5-fold CV)"] = m_lr
    print(f"  Logistic Reg.    F1={m_lr['f1']:.4f}  "
          f"P={m_lr['precision']:.3f}  R={m_lr['recall']:.3f}  "
          f"AUC={m_lr['auc_roc']}")

    # Train final LR on full data for coefficient inspection
    lr.fit(X, y)
    coef_df = pd.DataFrame({
        "signal": SIGNAL_COLS,
        "coefficient": lr.coef_[0].round(4),
    }).sort_values("coefficient", ascending=False)
    print("\n  Logistic Regression coefficients (higher = more predictive):")
    for _, row in coef_df.iterrows():
        bar = "█" * int(abs(row["coefficient"]) * 5)
        print(f"    {row['signal']:<30} {row['coefficient']:>+7.4f}  {bar}")

    # ── Baseline 3: GCN (from evaluation log gnn scores if available) ──
    if "gnn_raw_score" in df.columns:
        y_prob_gnn = df["gnn_raw_score"].fillna(0.5).tolist()
        y_pred_gnn = [1 if p >= 0.5 else 0 for p in y_prob_gnn]
        m_gnn = compute_metrics(y, y_pred_gnn, y_prob_gnn)
        results["GCN (current)"] = m_gnn
        print(f"\n  GCN              F1={m_gnn['f1']:.4f}  "
              f"P={m_gnn['precision']:.3f}  R={m_gnn['recall']:.3f}  "
              f"AUC={m_gnn['auc_roc']}")
    else:
        print("\n  GCN: gnn_raw_score not in log — skipped")

    # ── Save comparison table ────────────────────────────────────
    rows = [{"model": name, **m} for name, m in results.items()]
    comp_df = pd.DataFrame(rows)
    comp_path = REPORTS_DIR / "model_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"\n  ✅ Comparison table → {comp_path}")

    # Save into evaluation_results.json
    if RESULTS_JSON.exists():
        ev = json.loads(RESULTS_JSON.read_text())
        ev["model_comparison"] = {name: m for name, m in results.items()}
        ev["logreg_coefficients"] = coef_df.to_dict("records")
        RESULTS_JSON.write_text(json.dumps(ev, indent=2))

    # ── ROC curve ────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_facecolor("#0d0d1a")
        fig.patch.set_facecolor("#0d0d1a")

        # 3-of-7 rule ROC
        fpr_r, tpr_r, _ = roc_curve(y, y_prob_rule)
        ax.plot(fpr_r, tpr_r, color="#ffaa00", lw=2,
                label=f"3-of-7 Rule (AUC={m_rule['auc_roc']})")

        # Logistic Regression ROC
        fpr_l, tpr_l, _ = roc_curve(y, y_prob_lr)
        ax.plot(fpr_l, tpr_l, color="#7fc3ff", lw=2,
                label=f"Logistic Regression (AUC={m_lr['auc_roc']})")

        if "GCN (current)" in results:
            fpr_g, tpr_g, _ = roc_curve(y, y_prob_gnn)
            ax.plot(fpr_g, tpr_g, color="#ff7fc3", lw=2,
                    label=f"GCN (AUC={results['GCN (current)']['auc_roc']})")

        ax.plot([0, 1], [0, 1], color="#555", lw=1, linestyle="--")
        ax.set_xlabel("False Positive Rate", color="white", fontsize=11)
        ax.set_ylabel("True Positive Rate", color="white", fontsize=11)
        ax.set_title("ROC Curves — Model Comparison", color="white", fontsize=13)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        leg = ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#444")
        for text in leg.get_texts():
            text.set_color("white")

        roc_path = REPORTS_DIR / "roc_curve.png"
        plt.tight_layout()
        plt.savefig(roc_path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
        plt.close()
        print(f"  ✅ ROC curve      → {roc_path}")
    except Exception as e:
        print(f"  ⚠️  ROC chart failed: {e}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    print(f"  {'Model':<35} {'F1':>6} {'AUC-ROC':>8} {'Acc':>6}")
    print(f"  {'─'*35} {'─'*6} {'─'*8} {'─'*6}")
    for name, m in results.items():
        print(f"  {name:<35} {m['f1']:>6.4f} "
              f"{str(m['auc_roc']):>8} {m['accuracy']:>6.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
