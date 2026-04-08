"""
tests/ablation_study.py
────────────────────────
Leave-one-signal-out ablation study.

For each of the 7 signals, zero it out and re-evaluate the full pipeline
on the ground truth log produced by evaluate_ground_truth.py.

Because re-crawling takes too long, this script reads the cached
evaluation_log.csv (which contains per-domain per-signal flags) and
re-runs ONLY the verdict logic with one signal masked.

Usage:
    python3 -m tests.ablation_study

Outputs:
    data/reports/ablation_table.csv     ← signal × metric table
    data/reports/ablation_chart.png     ← horizontal Delta-F1 bar chart
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')

REPORTS_DIR = Path("data/reports")
LOG_CSV     = REPORTS_DIR / "evaluation_log.csv"
RESULTS_JSON = REPORTS_DIR / "evaluation_results.json"

SIGNALS = [
    ("signal_1_similarity",   "Content Similarity"),
    ("signal_2_cadence",      "Cadence Anomaly"),
    ("signal_3_whois",        "WHOIS Registration"),
    ("signal_4_hosting",      "Shared Hosting"),
    ("signal_5_link_network", "Insular Link Network"),
    ("signal_6_wayback",      "Wayback History"),
    ("signal_7_authors",      "Author Overlap"),
]

SIGNAL_COLS = [s[0] for s in SIGNALS]

VERDICT_THRESHOLD = 3   # signals needed for SYNTHETIC
REVIEW_THRESHOLD  = 2   # signals needed for REVIEW (≥2, <3)


def signals_to_verdict(signal_vals: list) -> int:
    """Re-apply 3-of-7 rule. Returns 1 (flagged) or 0 (organic)."""
    count = sum(signal_vals)
    if count >= VERDICT_THRESHOLD:
        return 1   # SYNTHETIC
    elif count >= REVIEW_THRESHOLD:
        return 1   # REVIEW → still flagged (predicted positive)
    else:
        return 0   # ORGANIC


def compute_metrics(y_true, y_pred) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "accuracy":  round(accuracy, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main():
    print("=" * 60)
    print("🔬 Dead Internet Detector — Ablation Study")
    print("=" * 60)

    if not LOG_CSV.exists():
        print(f"❌ {LOG_CSV} not found.")
        print("   Run tests/evaluate_ground_truth.py first.")
        sys.exit(1)

    df = pd.read_csv(LOG_CSV)
    print(f"  Loaded {len(df)} domain results from {LOG_CSV}")

    # Make sure all signal columns are present
    for col, _ in SIGNALS:
        if col not in df.columns:
            df[col] = 0

    y_true = df["label"].tolist()

    # ── Baseline: full model (as evaluated) ─────────────────────
    y_pred_full = df["predicted_label"].tolist()
    base = compute_metrics(y_true, y_pred_full)
    print(f"\n  Full model F1: {base['f1']:.4f}  "
          f"(P={base['precision']:.3f} R={base['recall']:.3f})")

    rows = [{"signal_removed": "None (full model)",
             "signal_key": "full",
             **base, "delta_f1": 0.0}]

    print(f"\n  {'Signal Removed':<30} {'F1':>6} {'Delta-F1':>10} "
          f"{'Prec':>6} {'Rec':>6}")
    print(f"  {'─'*30} {'─'*6} {'─'*10} {'─'*6} {'─'*6}")

    for col, name in SIGNALS:
        # Zero out this signal for every domain, re-apply verdict rule
        y_pred_masked = []
        for _, row in df.iterrows():
            vals = [int(row.get(c, 0)) for c in SIGNAL_COLS]
            # Mask the ablated signal
            idx = SIGNAL_COLS.index(col)
            vals[idx] = 0
            y_pred_masked.append(signals_to_verdict(vals))

        m = compute_metrics(y_true, y_pred_masked)
        delta = round(m["f1"] - base["f1"], 4)

        print(f"  - {name:<28} {m['f1']:>6.4f} {delta:>+10.4f} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f}")

        rows.append({"signal_removed": name, "signal_key": col,
                     **m, "delta_f1": delta})

    # ── Save table ───────────────────────────────────────────────
    ablation_df = pd.DataFrame(rows)
    out_csv = REPORTS_DIR / "ablation_table.csv"
    ablation_df.to_csv(out_csv, index=False)
    print(f"\n  ✅ Ablation table → {out_csv}")

    # ── Save to evaluation_results.json ─────────────────────────
    if RESULTS_JSON.exists():
        eval_data = json.loads(RESULTS_JSON.read_text())
        eval_data["ablation"] = rows
        RESULTS_JSON.write_text(json.dumps(eval_data, indent=2))

    # ── Draw chart ───────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Filter to ablated rows only (not "full model")
        abl_rows = [r for r in rows if r["signal_key"] != "full"]
        names    = [r["signal_removed"] for r in abl_rows]
        deltas   = [r["delta_f1"] for r in abl_rows]

        # Sort by delta (most negative = most important)
        paired = sorted(zip(deltas, names))
        deltas, names = zip(*paired)

        colors = ["#ff4444" if d < -0.03 else
                  "#ffaa00" if d < 0 else
                  "#44bb44" for d in deltas]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(names, deltas, color=colors, edgecolor="none", height=0.55)

        ax.axvline(0, color="#888", linewidth=1, linestyle="--")
        ax.set_xlabel("Delta-F1 (drop when signal removed)", fontsize=11)
        ax.set_title("Ablation Study — Marginal Contribution of Each Signal",
                     fontsize=13, pad=12)
        ax.set_facecolor("#0d0d1a")
        fig.patch.set_facecolor("#0d0d1a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        for bar, delta in zip(bars, deltas):
            ax.text(delta - 0.002 if delta < 0 else delta + 0.002,
                    bar.get_y() + bar.get_height() / 2,
                    f"{delta:+.3f}", va="center",
                    ha="right" if delta < 0 else "left",
                    color="white", fontsize=9)

        plt.tight_layout()
        chart_path = REPORTS_DIR / "ablation_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight",
                    facecolor="#0d0d1a")
        plt.close()
        print(f"  ✅ Ablation chart → {chart_path}")
    except Exception as e:
        print(f"  ⚠️  Chart failed: {e}")

    print("\n  Most important signals (largest negative delta = most critical):")
    for r in sorted(rows[1:], key=lambda x: x["delta_f1"]):
        print(f"    {r['signal_removed']:<30} delta={r['delta_f1']:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
