# pipeline/whois_features.py
# Fetches WHOIS registration data for domains (Signal 3)
# Input:  data/domains_clean.csv
# Output: data/whois_features.csv
# Run with: python3 pipeline/whois_features.py

import os
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

INPUT_FILE   = "data/domains_clean.csv"
OUTPUT_FILE  = "data/whois_features.csv"
MAX_DOMAINS  = 100       # stay within free tier limit
SLEEP_SEC    = 0.5       # pause between API calls to avoid rate limiting
API_KEY      = os.getenv("WHOIS_API_KEY")


def query_whois(domain):
    """Call WHOIS XML API for one domain. Returns dict of features."""
    url = "https://www.whoisxmlapi.com/whoisserver/WhoisService"
    params = {
        "apiKey":       API_KEY,
        "domainName":   domain,
        "outputFormat": "JSON",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        record = data.get("WhoisRecord", {})

        # Extract registration date
        created_date = (
            record.get("createdDate") or
            record.get("registryData", {}).get("createdDate") or
            ""
        )

        # Extract registrar name
        registrar = (
            record.get("registrarName") or
            record.get("registryData", {}).get("registrarName") or
            "unknown"
        )

        # Extract nameservers
        ns_list = []
        ns_data = record.get("nameServers", {})
        if isinstance(ns_data, dict):
            ns_list = ns_data.get("hostNames", [])
        elif isinstance(ns_data, list):
            ns_list = ns_data

        nameservers = "|".join(ns_list[:3]) if ns_list else ""

        # Calculate domain age in days
        domain_age_days = -1
        if created_date:
            try:
                created = datetime.strptime(
                    created_date[:10], "%Y-%m-%d"
                )
                domain_age_days = (datetime.now() - created).days
            except:
                pass

        return {
            "domain":          domain,
            "created_date":    created_date[:10] if created_date else "",
            "registrar":       str(registrar)[:80],
            "nameservers":     nameservers,
            "domain_age_days": domain_age_days,
        }

    except Exception as e:
        return None


def detect_clustering(df):
    """
    Flag domains that share registrars suspiciously.
    If a registrar appears for many domains, those domains get flagged.
    """
    # Count how many domains use each registrar
    registrar_counts = Counter(df['registrar'].tolist())

    # Flag domains using very common cheap registrars
    # (used by many fake domain factories)
    suspicious_registrars = {
        r for r, count in registrar_counts.items()
        if count >= 3 and r not in [
            'MarkMonitor', 'CSC Corporate Domains',
            'Network Solutions', 'GoDaddy', 'Namecheap',
            'Google Domains', 'Cloudflare', 'Amazon',
            'unknown', ''
        ]
    }

    # Also flag very young domains (registered < 365 days ago)
    def flag_domain(row):
        age   = row['domain_age_days']
        reg   = row['registrar']
        young = (age >= 0 and age < 365)
        clust = (reg in suspicious_registrars)
        return 1 if (young or clust) else 0

    df['whois_flagged'] = df.apply(flag_domain, axis=1)
    return df


def fetch_whois_features():
    print("=" * 55)
    print("Dead Internet Detector — Day 8")
    print("Fetching WHOIS features (Signal 3)...")
    print("=" * 55)
    print()

    # ── Check API key ──
    if not API_KEY or API_KEY == "your_whois_key_here":
        print("❌ WHOIS_API_KEY not set in .env file!")
        print("   Go to whoisxmlapi.com → My Products → API Access")
        return

    print(f"✅ WHOIS API key loaded: {API_KEY[:8]}...")
    print()

    # ── Load domains ──
    print(f"📂 Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    domains = df['domain'].dropna().unique().tolist()

    # Limit to MAX_DOMAINS to stay within free tier
    domains = domains[:MAX_DOMAINS]
    print(f"✅ Querying WHOIS for {len(domains)} domains")
    print()

    # ── Query WHOIS API ──
    results = []
    failed  = 0

    for i, domain in enumerate(domains):
        result = query_whois(domain)

        if result:
            results.append(result)
            age = result['domain_age_days']
            reg = result['registrar'][:30] if result['registrar'] else 'unknown'
            print(f"  [{i+1}/{len(domains)}] {domain[:35]:<35} "
                  f"age: {age:>5}d  reg: {reg}")
        else:
            # API failed — add row with defaults
            results.append({
                "domain":          domain,
                "created_date":    "",
                "registrar":       "unknown",
                "nameservers":     "",
                "domain_age_days": -1,
            })
            failed += 1
            print(f"  [{i+1}/{len(domains)}] {domain[:35]:<35} ⚠️  API call failed")

        # Pause between calls
        time.sleep(SLEEP_SEC)

    print()
    print(f"✅ Queries complete — {len(results) - failed} succeeded, {failed} failed")
    print()

    # ── Build DataFrame ──
    result_df = pd.DataFrame(results)

    # ── Detect clustering ──
    print("🔍 Detecting registrar clustering...")
    result_df = detect_clustering(result_df)

    # ── Save ──
    os.makedirs("data", exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    # ── Summary ──
    flagged = result_df['whois_flagged'].sum()
    young   = len(result_df[result_df['domain_age_days'] < 365])

    print("=" * 55)
    print(f"✅ Total domains processed: {len(result_df)}")
    print(f"✅ Young domains (<1 year):  {young}")
    print(f"✅ Flagged suspicious:       {flagged}")
    print(f"✅ Saved to:                 {OUTPUT_FILE}")
    print()
    print("Sample results:")
    print(result_df[['domain','domain_age_days','registrar','whois_flagged']]
          .head(5).to_string(index=False))
    print()
    print("=" * 55)
    print("🎉 Signal 3 complete! whois_features.csv is ready.")
    print("=" * 55)


if __name__ == "__main__":
    fetch_whois_features()