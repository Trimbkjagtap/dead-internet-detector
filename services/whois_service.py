import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import requests
from dotenv import load_dotenv

from config.signal_config import WHOIS_YOUNG_DOMAIN_DAYS

load_dotenv()

WHOIS_API_KEY = os.getenv("WHOIS_API_KEY", "")
WHOIS_STRICT_BUDGET_MODE = os.getenv("WHOIS_STRICT_BUDGET_MODE", "true").lower() == "true"
WHOIS_DAILY_QUERY_LIMIT = int(os.getenv("WHOIS_DAILY_QUERY_LIMIT", "15"))
WHOIS_CACHE_FILE = Path(os.getenv("WHOIS_CACHE_FILE", "data/whois_cache.json"))
WHOIS_TIMEOUT_SEC = int(os.getenv("WHOIS_TIMEOUT_SEC", "10"))

SUSPICIOUS_TLDS = [".xyz", ".top", ".click", ".online", ".site", ".info", ".buzz", ".icu"]
SUSPICIOUS_KEYWORDS = [
    "truth", "patriot", "freedom", "liberty", "alert", "insider",
    "expose", "breaking", "real-news", "updates-now", "daily-truth",
    "peoples-voice", "wire", "report", "first-news", "national-alert",
]


class WhoisService:
    def __init__(self) -> None:
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        if not WHOIS_CACHE_FILE.exists():
            return {"meta": {"day": self._today(), "queries_today": 0}, "domains": {}}
        try:
            data = json.loads(WHOIS_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {"meta": {"day": self._today(), "queries_today": 0}, "domains": {}}

        meta = data.get("meta", {})
        if meta.get("day") != self._today():
            meta = {"day": self._today(), "queries_today": 0}
        data["meta"] = meta
        data.setdefault("domains", {})
        return data

    def _save_cache(self) -> None:
        WHOIS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        WHOIS_CACHE_FILE.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _budget_available(self) -> bool:
        return int(self.cache["meta"].get("queries_today", 0)) < WHOIS_DAILY_QUERY_LIMIT

    def _increment_budget(self) -> None:
        self.cache["meta"]["queries_today"] = int(self.cache["meta"].get("queries_today", 0)) + 1

    def _should_query_domain(self, domain: str) -> bool:
        if not WHOIS_STRICT_BUDGET_MODE:
            return True
        return any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS) or any(k in domain for k in SUSPICIOUS_KEYWORDS) or domain.count("-") >= 2

    def _parse_created_date(self, created_date: str) -> Tuple[str, int]:
        if not created_date:
            return "", -1
        raw = created_date.strip()
        normalized = raw[:10]
        age_days = -1
        candidates = [raw, raw[:19], raw[:10]]
        for candidate in candidates:
            try:
                dt = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - dt.replace(tzinfo=None)).days
                normalized = dt.strftime("%Y-%m-%d")
                break
            except Exception:
                try:
                    dt = datetime.strptime(candidate, "%Y-%m-%d")
                    age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - dt).days
                    normalized = dt.strftime("%Y-%m-%d")
                    break
                except Exception:
                    continue
        return normalized, age_days

    def _query_whois_api(self, domain: str) -> Dict:
        if not WHOIS_API_KEY:
            return {}

        url = "https://www.whoisxmlapi.com/whoisserver/WhoisService"
        params = {
            "apiKey": WHOIS_API_KEY,
            "domainName": domain,
            "outputFormat": "JSON",
        }

        try:
            response = requests.get(url, params=params, timeout=WHOIS_TIMEOUT_SEC)
            if response.status_code != 200:
                return {}
            data = response.json()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("WHOIS API HTTP/parse error for %s: %s", domain, exc)
            return {}

        record = data.get("WhoisRecord", {})

        created_date = record.get("createdDate") or record.get("registryData", {}).get("createdDate") or ""
        registrar = record.get("registrarName") or record.get("registryData", {}).get("registrarName") or "unknown"
        country = record.get("registrant", {}).get("country") or record.get("registryData", {}).get("registrant", {}).get("country") or ""

        created_norm, age_days = self._parse_created_date(created_date)
        return {
            "created_date": created_norm,
            "domain_age_days": age_days,
            "registrar": str(registrar)[:80],
            "registration_country": str(country)[:50],
            "source": "whoisxml",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_domain_features(self, domain: str) -> Dict:
        domain = domain.strip().lower()
        cached = self.cache["domains"].get(domain)
        if cached:
            return cached

        result = {}
        if self._budget_available() and self._should_query_domain(domain):
            try:
                result = self._query_whois_api(domain)
                if result:
                    self._increment_budget()
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("WHOIS lookup failed for %s: %s", domain, exc)
                result = {}

        if not result:
            age_guess = -1
            whois_flagged = int(self._should_query_domain(domain))
            result = {
                "created_date": "",
                "domain_age_days": age_guess,
                "registrar": "unknown",
                "registration_country": "",
                "source": "heuristic",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "whois_flagged": whois_flagged,
            }
        else:
            result["whois_flagged"] = 1 if (result.get("domain_age_days", -1) >= 0 and result.get("domain_age_days", -1) <= WHOIS_YOUNG_DOMAIN_DAYS) else 0

        self.cache["domains"][domain] = result
        self._save_cache()
        return result

    def budget_status(self) -> Dict:
        return {
            "strict_mode": WHOIS_STRICT_BUDGET_MODE,
            "queries_today": int(self.cache["meta"].get("queries_today", 0)),
            "daily_limit": WHOIS_DAILY_QUERY_LIMIT,
            "remaining": max(0, WHOIS_DAILY_QUERY_LIMIT - int(self.cache["meta"].get("queries_today", 0))),
        }
