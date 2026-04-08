"""
tests/evaluate_ground_truth.py
──────────────────────────────
Runs the full 7-signal pipeline on every domain in data/ground_truth.csv
and computes:
  - Precision, Recall, F1, Accuracy (overall)
  - Confusion matrix
  - Per-signal precision, recall, F1, false-positive rate
  - Per-domain verdict log

Outputs saved to data/reports/:
  evaluation_results.json   ← main metrics
  per_signal_metrics.json   ← per-signal breakdown
  confusion_matrix.png      ← visual confusion matrix
  evaluation_log.csv        ← per-domain predicted vs actual
  cluster_evaluation.json   ← cluster-mode results (--cluster)

Usage:
  python3 -m tests.evaluate_ground_truth [--limit N] [--skip-crawl] [--cluster]

  --limit N      only evaluate first N domains (for quick smoke tests)
  --skip-crawl   use cached WHOIS/enrichment only, skip live crawl
                 (faster, uses existing data/enrichment_cache.json)
  --cluster      also run cluster-mode eval: all synthetic domains as one batch
                 and all organic as another — tests cross-domain signals (1, 5, 7)

NOTE on evaluation methodology:
  The Dead Internet Detector is designed for NETWORK-LEVEL detection.
  Signals 1 (content similarity), 5 (link network), and 7 (author overlap)
  are cross-domain signals that cannot fire in single-domain mode.
  Single-domain eval measures only signals 2–4 and 6.
  Use --cluster for a realistic assessment of network detection capability.
"""

import sys
import os
import json
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')

# ── Paths ─────────────────────────────────────────────────────────────────────
GT_FILE     = Path("data/ground_truth.csv")
REPORTS_DIR = Path("data/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Label mapping ──────────────────────────────────────────────────────────────
# ground_truth label: 1 = synthetic/fake, 0 = organic/legitimate
# verdict mapping to binary:
VERDICT_TO_BINARY = {
    "SYNTHETIC": 1,
    "REVIEW":    1,   # flagged = predicted positive
    "ORGANIC":   0,
}


def crawl_single(domain: str, skip_crawl: bool) -> dict:
    """Return a crawled record for one domain."""
    from agents.crawler_agent import fetch_page
    if skip_crawl:
        return {
            "domain": domain, "url": f"https://{domain}",
            "text": "", "body_text": "", "links": "",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "fallback", "article_authors": [],
        }
    try:
        result = fetch_page(f"https://{domain}")
        if result is None:
            return {
                "domain": domain, "url": f"https://{domain}",
                "text": "", "body_text": "", "links": "",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "fallback", "article_authors": [],
            }
        return result
    except Exception as e:
        print(f"    ⚠️  Crawl failed for {domain}: {e}")
        return {
            "domain": domain, "url": f"https://{domain}",
            "text": "", "body_text": "", "links": "",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "fallback", "article_authors": [],
        }


def run_pipeline_single(domain: str, skip_crawl: bool) -> dict:
    """
    Run the full pipeline on ONE domain (treated as a single-domain cluster).
    Returns the per-domain feature dict and verdict dict.
    """
    from agents.fingerprint_agent import analyze_domains
    from agents.verdict_agent import produce_verdict

    crawled = crawl_single(domain, skip_crawl)
    domains_data = [crawled]

    try:
        analysis = analyze_domains(domains_data)
        features  = analysis.get("features", {})
        sim_edges = analysis.get("sim_edges", {})
        excerpts  = analysis.get("excerpts", {})

        if not features:
            return {"verdict": "ORGANIC", "signals_triggered": 0, "confidence": 0.90,
                    "features": {}, "error": "no features"}

        verdict_result = produce_verdict(features, sim_edges, excerpts=excerpts)
        domain_verdict = verdict_result.get("domain_verdicts", {}).get(domain, {})
        feats          = features.get(domain, {})

        return {
            "verdict":    domain_verdict.get("verdict", "ORGANIC"),
            "confidence": domain_verdict.get("confidence", 0.0),
            "signals_triggered": domain_verdict.get("signals_triggered", 0),
            "signal_1_similarity":   int(feats.get("similarity_flag", 0)),
            "signal_2_cadence":      int(feats.get("cadence_flagged", 0)),
            "signal_3_whois":        int(feats.get("whois_flagged", 0)),
            "signal_4_hosting":      int(feats.get("hosting_flagged", 0)),
            "signal_5_link_network": int(feats.get("link_network_flagged", 0)),
            "signal_6_wayback":      int(feats.get("wayback_flagged", 0)),
            "signal_7_authors":      int(feats.get("author_overlap_flagged", 0)),
            "features": feats,
        }
    except Exception as e:
        print(f"    ❌ Pipeline error for {domain}: {e}")
        return {"verdict": "ORGANIC", "signals_triggered": 0, "confidence": 0.90,
                "signal_1_similarity": 0, "signal_2_cadence": 0, "signal_3_whois": 0,
                "signal_4_hosting": 0, "signal_5_link_network": 0,
                "signal_6_wayback": 0, "signal_7_authors": 0,
                "features": {}, "error": str(e)}


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Compute precision, recall, F1, accuracy from binary lists."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(y_true) if y_true else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "accuracy":  round(accuracy, 4),
        "fpr":       round(fpr, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def per_signal_metrics(log_df: pd.DataFrame) -> dict:
    """Compute per-signal precision/recall/F1/FPR against ground truth."""
    SIGNALS = {
        "signal_1_similarity":   "Content Similarity",
        "signal_2_cadence":      "Cadence Anomaly",
        "signal_3_whois":        "WHOIS Registration",
        "signal_4_hosting":      "Shared Hosting",
        "signal_5_link_network": "Insular Link Network",
        "signal_6_wayback":      "Wayback History",
        "signal_7_authors":      "Author Overlap",
    }
    results = {}
    for col, name in SIGNALS.items():
        if col not in log_df.columns:
            continue
        y_true = log_df["label"].tolist()
        y_pred = log_df[col].tolist()
        m = compute_metrics(y_true, y_pred)
        results[col] = {"signal_name": name, **m}
    return results


def save_confusion_matrix_png(tp, tn, fp, fn, path: Path):
    """Save a simple confusion matrix as PNG using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(5, 4))
        matrix = np.array([[tn, fp], [fn, tp]])
        labels = [["TN\n(Organic → Organic)", "FP\n(Organic → Flagged)"],
                  ["FN\n(Fake → Missed)",     "TP\n(Fake → Detected)"]]
        colors = [["#1a5c38", "#8b0000"], ["#8b4500", "#1a5c38"]]

        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                             facecolor=colors[i][j], alpha=0.8, edgecolor="white", lw=2))
                ax.text(j + 0.5, 1.5 - i, f"{matrix[i, j]}\n{labels[i][j]}",
                        ha="center", va="center", fontsize=11,
                        color="white", fontweight="bold")

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["Predicted\nOrganic", "Predicted\nSynthetic"], fontsize=10)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["Actual\nSynthetic", "Actual\nOrganic"], fontsize=10)
        ax.set_title("Dead Internet Detector — Confusion Matrix", fontsize=13, pad=12)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="#0d0d1a", edgecolor="none")
        plt.close()
        print(f"  ✅ Confusion matrix saved → {path}")
    except Exception as e:
        print(f"  ⚠️  Could not save confusion matrix PNG: {e}")


def run_cluster_evaluation(gt: pd.DataFrame, skip_crawl: bool) -> dict:
    """
    Cluster-mode evaluation: run all synthetic domains together as one batch,
    then all organic as another. This exercises cross-domain signals (1, 5, 7).

    Returns dict with metrics for synthetic_cluster and organic_cluster.
    """
    from agents.fingerprint_agent import analyze_domains
    from agents.verdict_agent import produce_verdict

    print()
    print("=" * 60)
    print("🔗 CLUSTER-MODE EVALUATION")
    print("  (cross-domain signals: similarity, link network, author overlap)")
    print("=" * 60)

    synthetic_domains = gt[gt["label"] == 1]["domain"].tolist()
    organic_domains   = gt[gt["label"] == 0]["domain"].tolist()

    cluster_results = {}

    for cluster_name, domains, expected_label in [
        ("synthetic_cluster", synthetic_domains, 1),
        ("organic_cluster",   organic_domains,   0),
    ]:
        print(f"\n  Cluster: {cluster_name} ({len(domains)} domains)")
        crawled = [crawl_single(d, skip_crawl) for d in domains]
        crawled = [c for c in crawled if c]

        try:
            analysis = analyze_domains(crawled)
            features  = analysis.get("features", {})
            sim_edges = analysis.get("sim_edges", {})
            excerpts  = analysis.get("excerpts", {})
            verdict_result = produce_verdict(features, sim_edges, excerpts=excerpts)
            domain_verdicts = verdict_result.get("domain_verdicts", {})

            per_domain = {}
            for d in domains:
                dv = domain_verdicts.get(d, {})
                per_domain[d] = {
                    "verdict":           dv.get("verdict", "ORGANIC"),
                    "signals_triggered": dv.get("signals_triggered", 0),
                    "confidence":        dv.get("confidence", 0.0),
                    "predicted_label":   VERDICT_TO_BINARY.get(dv.get("verdict", "ORGANIC"), 0),
                }
                correct = per_domain[d]["predicted_label"] == expected_label
                print(f"    {'✅' if correct else '❌'} {d}: "
                      f"{per_domain[d]['verdict']} "
                      f"(signals={per_domain[d]['signals_triggered']})")

            # Aggregate: fraction flagged
            flagged = sum(1 for v in per_domain.values() if v["predicted_label"] == 1)
            cluster_results[cluster_name] = {
                "n_domains": len(domains),
                "n_flagged": flagged,
                "flagged_rate": round(flagged / len(domains), 4) if domains else 0.0,
                "expected_label": expected_label,
                "per_domain": per_domain,
                # Signals summary from features
                "signals_fired": {
                    col: sum(1 for d in domains
                             if features.get(d, {}).get(col.replace("signal_", "").replace("_", "_") + "_flagged"
                                                        if "signal_" in col else col, 0))
                    for col in ["similarity_flag", "cadence_flagged", "whois_flagged",
                                "hosting_flagged", "link_network_flagged",
                                "wayback_flagged", "author_overlap_flagged"]
                },
            }
        except Exception as e:
            print(f"    ❌ Cluster pipeline error: {e}")
            cluster_results[cluster_name] = {"error": str(e), "n_domains": len(domains)}

    # Compute cluster-level TP/FP metrics
    syn = cluster_results.get("synthetic_cluster", {})
    org = cluster_results.get("organic_cluster", {})

    if syn and org and "n_flagged" in syn and "n_flagged" in org:
        tp = syn["n_flagged"]
        fn = syn["n_domains"] - tp
        fp = org["n_flagged"]
        tn = org["n_domains"] - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        print(f"\n  Cluster-mode overall:")
        print(f"    TP={tp}  FN={fn}  FP={fp}  TN={tn}")
        print(f"    Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

        cluster_results["overall"] = {
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return cluster_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only first N domains")
    parser.add_argument("--skip-crawl", action="store_true",
                        help="Skip live crawl (use cached data only)")
    parser.add_argument("--cluster", action="store_true",
                        help="Also run cluster-mode evaluation (tests cross-domain signals)")
    args = parser.parse_args()

    print("=" * 60)
    print("🧪 Dead Internet Detector — Ground Truth Evaluation")
    print("=" * 60)

    gt = pd.read_csv(GT_FILE)
    if args.limit:
        gt = gt.head(args.limit)
        print(f"  ⚠️  Limited to first {args.limit} domains")

    print(f"  Dataset: {len(gt)} domains "
          f"({gt['label'].sum()} synthetic, {(gt['label']==0).sum()} organic)")
    print()

    log_rows   = []
    y_true_all = []
    y_pred_all = []

    for idx, row in gt.iterrows():
        domain = row["domain"].strip().lower()
        label  = int(row["label"])
        print(f"  [{idx+1:3d}/{len(gt)}] {domain} (true={'SYNTHETIC' if label else 'ORGANIC'})")

        t0     = time.time()
        result = run_pipeline_single(domain, skip_crawl=args.skip_crawl)
        elapsed = time.time() - t0

        predicted_label = VERDICT_TO_BINARY.get(result["verdict"], 0)
        y_true_all.append(label)
        y_pred_all.append(predicted_label)

        correct = "✅" if predicted_label == label else "❌"
        print(f"         → {result['verdict']} "
              f"(conf={result['confidence']:.0%}, "
              f"signals={result['signals_triggered']}/7) "
              f"{correct}  [{elapsed:.1f}s]")

        log_rows.append({
            "domain":              domain,
            "label":               label,
            "predicted_verdict":   result["verdict"],
            "predicted_label":     predicted_label,
            "correct":             predicted_label == label,
            "confidence":          result["confidence"],
            "signals_triggered":   result["signals_triggered"],
            "signal_1_similarity":   result.get("signal_1_similarity", 0),
            "signal_2_cadence":      result.get("signal_2_cadence", 0),
            "signal_3_whois":        result.get("signal_3_whois", 0),
            "signal_4_hosting":      result.get("signal_4_hosting", 0),
            "signal_5_link_network": result.get("signal_5_link_network", 0),
            "signal_6_wayback":      result.get("signal_6_wayback", 0),
            "signal_7_authors":      result.get("signal_7_authors", 0),
            "error":               result.get("error", ""),
        })

    print()
    print("=" * 60)
    print("📊 RESULTS")
    print("=" * 60)

    # ── Overall metrics ──────────────────────────────────────────
    overall = compute_metrics(y_true_all, y_pred_all)
    print(f"  Accuracy:  {overall['accuracy']:.4f}")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall:    {overall['recall']:.4f}")
    print(f"  F1 Score:  {overall['f1']:.4f}")
    print(f"  FP Rate:   {overall['fpr']:.4f}")
    print()
    print(f"  TP={overall['tp']}  TN={overall['tn']}  "
          f"FP={overall['fp']}  FN={overall['fn']}")
    print()

    # ── Per-signal metrics ───────────────────────────────────────
    log_df = pd.DataFrame(log_rows)
    sig_metrics = per_signal_metrics(log_df)

    print("  Per-signal breakdown:")
    print(f"  {'Signal':<30} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FPR':>6}")
    print(f"  {'─'*30} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for col, m in sig_metrics.items():
        print(f"  {m['signal_name']:<30} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['fpr']:>6.3f}")

    # ── Save outputs ─────────────────────────────────────────────
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # evaluation_results.json
    eval_out = {
        "timestamp":      ts,
        "dataset_size":   len(gt),
        "n_synthetic":    int(gt["label"].sum()),
        "n_organic":      int((gt["label"] == 0).sum()),
        "overall":        overall,
        "per_signal":     sig_metrics,
        "skip_crawl":     args.skip_crawl,
    }
    eval_path = REPORTS_DIR / "evaluation_results.json"
    eval_path.write_text(json.dumps(eval_out, indent=2))
    print(f"\n  ✅ Results saved  → {eval_path}")

    # per_signal_metrics.json
    sig_path = REPORTS_DIR / "per_signal_metrics.json"
    sig_path.write_text(json.dumps(sig_metrics, indent=2))
    print(f"  ✅ Signal metrics → {sig_path}")

    # evaluation_log.csv
    log_path = REPORTS_DIR / "evaluation_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"  ✅ Domain log     → {log_path}")

    # confusion_matrix.png
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    save_confusion_matrix_png(
        overall["tp"], overall["tn"], overall["fp"], overall["fn"], cm_path
    )

    print()
    print(f"  🎯 F1 = {overall['f1']:.4f} (single-domain mode)")

    # ── Cluster-mode evaluation ──────────────────────────────────
    if args.cluster:
        cluster_res = run_cluster_evaluation(gt, skip_crawl=args.skip_crawl)
        cluster_path = REPORTS_DIR / "cluster_evaluation.json"
        cluster_path.write_text(json.dumps(cluster_res, indent=2))
        print(f"\n  ✅ Cluster results  → {cluster_path}")
        if "overall" in cluster_res:
            co = cluster_res["overall"]
            print(f"  🎯 Cluster F1 = {co['f1']:.4f}  "
                  f"(Precision={co['precision']:.3f}, Recall={co['recall']:.3f})")

    print("=" * 60)


if __name__ == "__main__":
    main()
