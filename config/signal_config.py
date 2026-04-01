import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.45"))
SIM_EDGE_FLAG_THRESHOLD = float(os.getenv("SIM_EDGE_FLAG_THRESHOLD", str(SIM_THRESHOLD)))
CADENCE_THRESHOLD = float(os.getenv("CADENCE_THRESHOLD", "-0.1"))
CADENCE_CONTAMINATION_MIN = float(os.getenv("CADENCE_CONTAMINATION_MIN", "0.1"))
CADENCE_CONTAMINATION_MAX = float(os.getenv("CADENCE_CONTAMINATION_MAX", "0.3"))
WHOIS_YOUNG_DOMAIN_DAYS = int(os.getenv("WHOIS_YOUNG_DOMAIN_DAYS", "90"))

# Signal 4 — Hosting overlap
IPINFO_TOKEN = os.getenv("IPINFO_TOKEN", "")

# Signal 5 — Link network
INSULAR_SCORE_THRESHOLD = float(os.getenv("INSULAR_SCORE_THRESHOLD", "0.5"))
MIN_LINKS_FOR_INSULAR   = int(os.getenv("MIN_LINKS_FOR_INSULAR", "3"))
MIN_DOMAINS_FOR_INSULAR = int(os.getenv("MIN_DOMAINS_FOR_INSULAR", "5"))

# Signal 6 — Wayback Machine
WAYBACK_MIN_SNAPSHOTS = int(os.getenv("WAYBACK_MIN_SNAPSHOTS", "5"))
WAYBACK_RECENT_DAYS   = int(os.getenv("WAYBACK_RECENT_DAYS", "30"))
WAYBACK_SPIKE_RATIO   = float(os.getenv("WAYBACK_SPIKE_RATIO", "0.5"))

# Verdict thresholds (now 7 signals total)
SIGNALS_TOTAL             = 7
SIGNALS_SYNTHETIC_THRESHOLD = int(os.getenv("SIGNALS_SYNTHETIC_THRESHOLD", "3"))


def bounded_contamination(sample_size: int) -> float:
    if sample_size <= 0:
        return CADENCE_CONTAMINATION_MIN
    return min(CADENCE_CONTAMINATION_MAX, max(CADENCE_CONTAMINATION_MIN, 1.0 / sample_size))
