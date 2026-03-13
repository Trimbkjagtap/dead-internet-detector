# pipeline/cadence_analysis.py
# Detects publishing cadence anomalies using Isolation Forest (Signal 2)
# Input:  data/domains_clean.csv
# Output: data/cadence_scores.csv
# Run with: python3 pipeline/cadence_analysis.py

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest

INPUT_FILE  = "data/domains_clean.csv"
OUTPUT_FILE = "data/cadence_scores.csv"


def parse_hour(timestamp):
    """Extract hour of day from timestamp string."""
    try:
        ts = str(timestamp)[:19]
        dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
        return dt.hour
    except:
        try:
            dt = datetime.strptime(ts[:19], '%Y-%m-%dT%H:%M:%S')
            return dt.hour
        except:
            return -1


def parse_minute(timestamp):
    """Extract minute from timestamp."""
    try:
        ts = str(timestamp)[:19]
        dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
        return dt.minute
    except:
        return -1


def parse_day_of_week(timestamp):
    """Extract day of week (0=Monday, 6=Sunday)."""
    try:
        ts = str(timestamp)[:19]
        dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
        return dt.weekday()
    except:
        return -1


def extract_cadence_features(df):
    """
    Extract publishing cadence features from timestamps.
    Each domain gets one row of features describing its publishing pattern.
    """
    print("  Extracting timestamp features...")

    # Extract time components
    df['hour']        = df['timestamp'].apply(parse_hour)
    df['minute']      = df['timestamp'].apply(parse_minute)
    df['day_of_week'] = df['timestamp'].apply(parse_day_of_week)

    # Remove rows where timestamp parsing failed
    df = df[df['hour'] >= 0].copy()

    print(f"  ✅ Valid timestamps: {len(df)}")

    # Group by domain and compute cadence features
    features = []

    for domain, group in df.groupby('domain'):
        hours   = group['hour'].tolist()
        minutes = group['minute'].tolist()
        days    = group['day_of_week'].tolist()

        # Feature 1: How many pages from this domain
        page_count = len(group)

        # Feature 2: Average publishing hour
        avg_hour = np.mean(hours)

        # Feature 3: Variance in publishing hour
        # Low variance = always posts at same time = suspicious
        hour_variance = np.var(hours) if len(hours) > 1 else 0.0

        # Feature 4: Variance in publishing minute
        # Very low minute variance = robot-like precision = suspicious
        minute_variance = np.var(minutes) if len(minutes) > 1 else 0.0

        # Feature 5: Unique hours used
        # Few unique hours = rigid schedule = suspicious
        unique_hours = len(set(hours))

        # Feature 6: Day of week spread
        # Only posts on weekdays or only weekends can be suspicious
        unique_days = len(set(days))

        # Feature 7: Burst score
        # How many pages posted within the same hour
        from collections import Counter
        hour_counts = Counter(hours)
        max_in_one_hour = max(hour_counts.values())
        burst_score = max_in_one_hour / page_count  # ratio of max burst

        features.append({
            'domain':           domain,
            'page_count':       page_count,
            'avg_hour':         round(avg_hour, 2),
            'hour_variance':    round(hour_variance, 4),
            'minute_variance':  round(minute_variance, 4),
            'unique_hours':     unique_hours,
            'unique_days':      unique_days,
            'burst_score':      round(burst_score, 4),
        })

    return pd.DataFrame(features)


def run_isolation_forest(features_df):
    """
    Run Isolation Forest on cadence features.
    Returns anomaly scores — more negative = more anomalous.
    """
    print("  Running Isolation Forest...")

    # Select only numeric feature columns
    feature_cols = [
        'page_count', 'avg_hour', 'hour_variance',
        'minute_variance', 'unique_hours', 'unique_days', 'burst_score'
    ]

    X = features_df[feature_cols].fillna(0).values

    # Train Isolation Forest
    # contamination=0.1 means we expect ~10% of domains to be anomalous
    clf = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42
    )
    clf.fit(X)

    # Get anomaly scores (more negative = more anomalous)
    scores = clf.decision_function(X)
    predictions = clf.predict(X)  # -1 = anomaly, 1 = normal

    features_df['anomaly_score']   = np.round(scores, 4)
    features_df['cadence_flagged'] = (predictions == -1).astype(int)

    return features_df


def analyze_cadence():
    print("=" * 55)
    print("Dead Internet Detector — Day 7")
    print("Publishing Cadence Anomaly Detection (Signal 2)...")
    print("=" * 55)
    print()

    # ── Load data ──
    print(f"📂 Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Loaded {len(df)} rows")
    print()

    # ── Extract features ──
    print("🔧 Extracting cadence features...")
    features_df = extract_cadence_features(df)
    print(f"✅ Features extracted for {len(features_df)} domains")
    print()

    # ── Run anomaly detection ──
    print("🤖 Running anomaly detection...")
    if len(features_df) < 5:
        print("⚠️  Too few domains for Isolation Forest.")
        print("   Assigning default scores...")
        features_df['anomaly_score']   = 0.0
        features_df['cadence_flagged'] = 0
    else:
        features_df = run_isolation_forest(features_df)

    flagged = features_df['cadence_flagged'].sum()
    print(f"✅ Anomaly detection complete!")
    print(f"✅ Flagged {flagged} domains as suspicious cadence")
    print()

    # ── Save results ──
    os.makedirs("data", exist_ok=True)
    features_df.to_csv(OUTPUT_FILE, index=False)

    # ── Summary ──
    print("=" * 55)
    print(f"✅ Total domains scored: {len(features_df)}")
    print(f"✅ Suspicious (flagged): {flagged}")
    print(f"✅ Saved to:             {OUTPUT_FILE}")
    print()
    print("Columns:", list(features_df.columns))
    print()
    print("Most anomalous domains:")
    top = features_df.nsmallest(5, 'anomaly_score')
    print(top[['domain','anomaly_score','burst_score','cadence_flagged']].to_string(index=False))
    print()
    print("=" * 55)
    print("🎉 Signal 2 complete! cadence_scores.csv is ready.")
    print("=" * 55)


if __name__ == "__main__":
    analyze_cadence()