import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import requests

WHOISDS_URL = os.getenv("WHOISDS_DAILY_URL", "")
SUSPICIOUS_TLDS = tuple(os.getenv("WHOISDS_SUSPICIOUS_TLDS", ".xyz,.top,.click,.online,.site,.info,.buzz,.icu").split(","))
MAX_DOMAINS = int(os.getenv("WHOISDS_MAX_DOMAINS", "500"))
OUTPUT_DIR = Path("data/incoming")
OUTPUT_FILE = OUTPUT_DIR / "whoisds_domains.json"


def _is_suspicious(domain: str) -> bool:
    d = domain.lower()
    return d.endswith(SUSPICIOUS_TLDS) or d.count("-") >= 2


def fetch_whoisds_domains() -> Dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not WHOISDS_URL:
        payload = {
            "source": "whoisds",
            "fetched_at": datetime.utcnow().isoformat(),
            "error": "WHOISDS_DAILY_URL not configured",
            "domain_count": 0,
            "domains": [],
        }
        OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    resp = requests.get(WHOISDS_URL, timeout=30)
    resp.raise_for_status()

    rows = csv.reader(resp.text.splitlines())
    domains: Set[str] = set()

    for row in rows:
        if not row:
            continue
        domain = row[0].strip().lower()
        if "." not in domain:
            continue
        if _is_suspicious(domain):
            domains.add(domain)
        if len(domains) >= MAX_DOMAINS:
            break

    payload = {
        "source": "whoisds",
        "fetched_at": datetime.utcnow().isoformat(),
        "domain_count": len(domains),
        "domains": sorted(domains),
    }
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = fetch_whoisds_domains()
    print(f"WHOISDS ingestion complete: {result['domain_count']} domains")
