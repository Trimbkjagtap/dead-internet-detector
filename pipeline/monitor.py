import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from agents.crew import run_analysis
from pipeline.ingest_reddit import fetch_reddit_domains
from pipeline.ingest_whoisds import fetch_whoisds_domains

load_dotenv()

BATCH_SIZE = int(os.getenv("MONITOR_BATCH_SIZE", "10"))
MAX_ANALYZE_PER_RUN = int(os.getenv("MONITOR_MAX_ANALYZE_PER_RUN", "40"))
LOG_FILE = Path("data/monitor_runs.jsonl")


def _chunks(items: List[str], size: int) -> List[List[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _read_domains(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("domains", [])
    except Exception:
        return []


def run_monitor_cycle() -> Dict:
    whois_payload = fetch_whoisds_domains()
    reddit_payload = fetch_reddit_domains()

    whois_domains = set(_read_domains(Path("data/incoming/whoisds_domains.json")))
    reddit_domains = set(_read_domains(Path("data/incoming/reddit_domains.json")))

    merged = sorted((whois_domains | reddit_domains))[:MAX_ANALYZE_PER_RUN]
    batches = _chunks(merged, BATCH_SIZE)

    results = []
    for batch in batches:
        if not batch:
            continue
        results.append(run_analysis(batch))

    summary = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "whoisds_count": whois_payload.get("domain_count", 0),
        "reddit_count": reddit_payload.get("domain_count", 0),
        "queued_unique": len(merged),
        "batches": len(batches),
        "results": len(results),
        "synthetic_batches": sum(1 for r in results if r.get("cluster_verdict") == "SYNTHETIC"),
    }

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(summary) + "\n")

    return summary


if __name__ == "__main__":
    report = run_monitor_cycle()
    print("Monitor cycle complete")
    print(json.dumps(report, indent=2))
