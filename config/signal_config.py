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


def bounded_contamination(sample_size: int) -> float:
    if sample_size <= 0:
        return CADENCE_CONTAMINATION_MIN
    return min(CADENCE_CONTAMINATION_MAX, max(CADENCE_CONTAMINATION_MIN, 1.0 / sample_size))
