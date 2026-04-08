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
SIGNALS_TOTAL               = 7
SIGNALS_SYNTHETIC_THRESHOLD = int(os.getenv("SIGNALS_SYNTHETIC_THRESHOLD", "3"))
# GNN training pseudo-label threshold — must match verdict threshold to avoid bias
PSEUDO_LABEL_THRESHOLD      = SIGNALS_SYNTHETIC_THRESHOLD

# Canonical noise-author set — shared by crawler_agent and fingerprint_agent
NOISE_AUTHORS = {
    "reuters", "ap", "associated press", "staff writer", "news desk",
    "wire service", "staff reporter", "the editors", "editorial board",
    "admin", "editor", "contributor", "guest writer", "press release",
}


def bounded_contamination(sample_size: int) -> float:
    if sample_size <= 0:
        return CADENCE_CONTAMINATION_MIN
    return min(CADENCE_CONTAMINATION_MAX, max(CADENCE_CONTAMINATION_MIN, 1.0 / sample_size))


def compute_confidence_from_signals(signals: int, max_sim: float = 0.0,
                                    burst: float = 0.0, gnn_prob: float = 0.0) -> float:
    """
    Single canonical confidence formula used across verdict_agent, main.py, and app.py.
    Returns a float in [0.02, 0.97].

    Tier anchoring:
      SYNTHETIC (>=3 signals): base 0.65, max 0.97
      REVIEW    (1-2 signals): base 0.40, max 0.64
      ORGANIC   (0 signals):   inverted — base 0.70, max 0.97
    """
    signals  = int(signals)
    max_sim  = float(max_sim)
    burst    = float(burst)
    gnn_prob = float(gnn_prob)

    sim_boost     = min(max_sim * 0.20, 0.10)
    cadence_boost = min(burst   * 0.15, 0.06)
    gnn_boost     = gnn_prob * 0.05

    if signals >= 3:
        base         = 0.65
        signal_boost = min((signals - 3) / 4.0, 1.0) * 0.20
        confidence   = base + signal_boost + sim_boost + cadence_boost + gnn_boost
        confidence   = max(0.65, min(confidence, 0.97))
    elif signals >= 1:
        base         = 0.40
        signal_boost = (signals - 1) * 0.10
        confidence   = base + signal_boost + sim_boost + cadence_boost + gnn_boost
        confidence   = max(0.02, min(confidence, 0.64))
    else:
        # ORGANIC — invert: lower suspicion scores = higher organic confidence
        confidence = max(0.70, min(1.0 - (sim_boost + cadence_boost + gnn_boost), 0.97))

    return round(confidence, 2)
