# services/enrichment_service.py
# IP/ASN resolution (Signal 4) + Wayback Machine history (Signal 6)
# Results cached to data/enrichment_cache.json for 24 hours

import json
import socket
import os
import requests
from datetime import datetime, timezone, timedelta

CACHE_FILE = "data/enrichment_cache.json"
CACHE_TTL_HOURS = 24

# ASNs belonging to large CDNs — shared IP on these is NOT suspicious
_CDN_ASNS = {
    "AS13335",  # Cloudflare
    "AS54113",  # Fastly
    "AS16509",  # Amazon AWS
    "AS15169",  # Google
    "AS20940",  # Akamai
    "AS209242", # Cloudflare (secondary)
    "AS22120",  # Cloudflare (secondary)
    "AS14618",  # Amazon AWS
    "AS8075",   # Microsoft Azure
}

# Cloudflare IPv4 ranges (shared IPs here are never suspicious)
_CDN_IP_PREFIXES = (
    "104.16.", "104.17.", "104.18.", "104.19.", "104.20.", "104.21.",
    "172.64.", "172.65.", "172.66.", "172.67.", "162.158.", "198.41.",
    "1.1.1.", "1.0.0.",
)


def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"ip_asn": {}, "wayback": {}}


def _save_cache(cache: dict):
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def _is_fresh(cached_at: str) -> bool:
    try:
        t = datetime.fromisoformat(cached_at)
        return (datetime.now(timezone.utc) - t) < timedelta(hours=CACHE_TTL_HOURS)
    except Exception:
        return False


def resolve_ip_and_asn(domain: str, token: str = "") -> dict:
    """
    Resolve a domain to its IP address and hosting organisation/ASN.

    Returns:
        {
            "ip_address": str,
            "asn": str,          e.g. "AS13335"
            "hosting_org": str,  e.g. "Cloudflare, Inc."
            "is_cdn": bool,      True if hosted on a major CDN
            "error": str | None
        }
    """
    cache = _load_cache()
    entry = cache.get("ip_asn", {}).get(domain)
    if entry and _is_fresh(entry.get("cached_at", "")):
        return entry

    result = {
        "ip_address": "",
        "asn": "",
        "hosting_org": "",
        "is_cdn": False,
        "error": None,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }

    # Step 1: DNS resolution
    try:
        addr_info = socket.getaddrinfo(domain, None, socket.AF_INET)
        if addr_info:
            result["ip_address"] = addr_info[0][4][0]
    except Exception as e:
        result["error"] = f"DNS failed: {e}"
        return result

    ip = result["ip_address"]

    # Step 2: ipinfo.io for ASN/org
    try:
        url = f"https://ipinfo.io/{ip}/json"
        params = {}
        if token:
            params["token"] = token
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            org = data.get("org", "")  # e.g. "AS13335 Cloudflare, Inc."
            if org:
                parts = org.split(" ", 1)
                result["asn"] = parts[0] if parts[0].startswith("AS") else ""
                result["hosting_org"] = parts[1] if len(parts) > 1 else org
    except Exception as e:
        result["error"] = f"ipinfo failed: {e}"

    # Step 3: flag CDN
    result["is_cdn"] = (
        result["asn"] in _CDN_ASNS
        or any(ip.startswith(p) for p in _CDN_IP_PREFIXES)
    )

    cache.setdefault("ip_asn", {})[domain] = result
    _save_cache(cache)
    return result


def get_wayback_data(domain: str) -> dict:
    """
    Query Wayback Machine CDX API for snapshot history.

    Returns:
        {
            "wayback_snapshot_count": int,  total deduplicated snapshots
            "wayback_first_seen": str,      "YYYYMMDD" or ""
            "wayback_recent_count": int,    snapshots in last 30 days
            "wayback_age_days": int,        days since first snapshot (-1 if unknown)
            "wayback_error": bool
        }
    """
    cache = _load_cache()
    entry = cache.get("wayback", {}).get(domain)
    if entry and _is_fresh(entry.get("cached_at", "")):
        return entry

    result = {
        "wayback_snapshot_count": 0,
        "wayback_first_seen": "",
        "wayback_recent_count": 0,
        "wayback_age_days": -1,
        "wayback_error": False,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }

    base = "https://web.archive.org/cdx/search/cdx"

    try:
        # Total deduplicated monthly snapshots
        # matchType=domain covers all URLs under the domain (not just exact homepage)
        resp = requests.get(base, params={
            "url": domain,
            "matchType": "domain",
            "output": "json",
            "fl": "timestamp",
            "collapse": "timestamp:6",  # deduplicate by month
            "limit": "500",
        }, timeout=15)

        if resp.status_code == 200 and resp.text.strip():
            rows = resp.json()
            # First row is the header ["timestamp"]
            data_rows = rows[1:] if rows and rows[0] == ["timestamp"] else rows
            result["wayback_snapshot_count"] = len(data_rows)

            if data_rows:
                first_ts = data_rows[0][0]  # oldest first
                result["wayback_first_seen"] = first_ts[:8]
                try:
                    first_dt = datetime.strptime(first_ts[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
                    result["wayback_age_days"] = (datetime.now(timezone.utc) - first_dt).days
                except Exception:
                    pass

                # Count recent snapshots (last 30 days)
                cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y%m%d")
                result["wayback_recent_count"] = sum(
                    1 for row in data_rows if row[0][:8] >= cutoff
                )

    except Exception as e:
        result["wayback_error"] = True
        result["cached_at"] = datetime.now(timezone.utc).isoformat()

    cache.setdefault("wayback", {})[domain] = result
    _save_cache(cache)
    return result
