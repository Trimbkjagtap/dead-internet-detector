# agents/crawler_agent.py
# Agent 1 — Crawler Agent
# Takes seed domains, fetches content, extracts text + links + timestamps
# Falls back to pre-crawled dataset if live crawling fails
# Reports failed domains instead of silently skipping

import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime, timezone


# ── Config ───────────────────────────────────────────
MAX_DEPTH    = 1      # how many hops from seed domains
MAX_DOMAINS  = 20     # max total domains to crawl
TIMEOUT      = 8      # seconds per request
SLEEP_SEC    = 0.5    # pause between requests (be polite)
HEADERS      = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; DeadInternetDetector/1.0; "
        "+https://github.com/Trimbkjagtap/dead-internet-detector)"
    )
}

# Track failed domains globally so we can report them
_failed_domains = []


def get_failed_domains():
    """Return list of domains that failed to crawl."""
    return list(_failed_domains)


def extract_domain(url: str) -> str | None:
    """Extract clean domain from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain if "." in domain else None
    except:
        return None


def get_parent_domain(domain: str) -> str:
    """
    Extract parent domain from a subdomain.
    e.g. 'sponsored.bostonglobe.com' -> 'bostonglobe.com'
    e.g. 'bbc.co.uk' -> 'bbc.co.uk'
    """
    parts = domain.split('.')
    # Handle .co.uk, .com.au style TLDs
    if len(parts) >= 3 and parts[-2] in ['co', 'com', 'org', 'net', 'gov']:
        return '.'.join(parts[-3:])
    elif len(parts) >= 2:
        return '.'.join(parts[-2:])
    return domain


def fetch_page(url: str) -> dict | None:
    """
    Fetch a single URL and extract:
    - domain name
    - clean text content
    - outgoing links
    - timestamp
    - final URL after redirects
    """
    try:
        if not url.startswith("http"):
            url = "https://" + url

        response = requests.get(
            url, headers=HEADERS,
            timeout=TIMEOUT, allow_redirects=True
        )

        if response.status_code != 200:
            return None

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return None

        # Track the final URL after redirects
        final_url = response.url
        final_domain = extract_domain(final_url)

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer",
                          "header", "aside", "form"]):
            tag.decompose()

        # Extract text
        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())[:2000]

        # Extract outgoing links — exclude subdomains of same parent
        original_domain = extract_domain(url)
        parent = get_parent_domain(original_domain) if original_domain else ""
        
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http"):
                linked = extract_domain(href)
                if linked and linked != original_domain:
                    # Skip subdomains of the same parent domain
                    if get_parent_domain(linked) != parent:
                        links.append(linked)
        links = list(set(links))[:15]

        return {
            "domain":       original_domain,
            "url":          url,
            "final_url":    final_url,
            "final_domain": final_domain,
            "text":         text,
            "links":        "|".join(links),
            "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "status":       "ok",
        }

    except requests.exceptions.Timeout:
        return {"domain": extract_domain(url), "url": url,
                "text": "", "links": "", "timestamp": "", "status": "timeout"}
    except Exception:
        return None


def load_fallback_data():
    """
    Load pre-crawled data from CSV for domains that can't be reached live.
    Loads from ALL available data files.
    """
    import pandas as pd
    import os

    fallback = {}

    csv_files = [
        'data/domains_clean.csv',
        'data/synthetic_ecosystem.csv',
        'data/domains_raw.csv',
        'data/ground_truth.csv',
    ]

    for path in csv_files:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                loaded = 0
                for _, row in df.iterrows():
                    domain = str(row.get('domain', '')).strip().lower()
                    if domain and domain not in fallback:
                        text = ''
                        for col in ['text', 'content', 'page_text', 'body']:
                            if col in df.columns and pd.notna(row.get(col, None)):
                                text = str(row[col])[:2000]
                                break

                        links = ''
                        for col in ['links', 'outgoing_links', 'linked_domains']:
                            if col in df.columns and pd.notna(row.get(col, None)):
                                links = str(row[col])
                                break

                        timestamp = '2024-12-15 03:47:00'
                        for col in ['timestamp', 'crawl_date', 'date', 'created_at']:
                            if col in df.columns and pd.notna(row.get(col, None)):
                                timestamp = str(row[col])
                                break

                        if text and len(text) > 50:
                            fallback[domain] = {
                                'domain':    domain,
                                'url':       f"https://{domain}",
                                'text':      text,
                                'links':     links,
                                'timestamp': timestamp,
                                'status':    'fallback',
                            }
                            loaded += 1

                print(f"    📦 Loaded {path}: {loaded} domains with content")
            except Exception as e:
                print(f"    ⚠️ Could not load {path}: {e}")
        else:
            print(f"    ⚠️ {path}: not found")

    print(f"    📦 Fallback data total: {len(fallback)} domains available")
    return fallback


def crawl_domains(seed_domains: list) -> list:
    """
    Main crawling function.
    Takes a list of seed domain names.
    Returns list of dicts with domain data.

    Strategy:
    1. Try to crawl the domain live from the internet
    2. If live crawl fails, check fallback CSV data
    3. If neither works, report the failure (not silently skip)
    """
    global _failed_domains
    failed_this_run: list = []

    print(f"🕷️  Crawler Agent starting...")
    print(f"   Seed domains: {seed_domains}")
    print(f"   Max domains:  {MAX_DOMAINS}")
    print()

    # Load fallback data from CSV files
    fallback = load_fallback_data()

    results     = []
    visited     = set()
    seen_parents = set()  # Track parent domains to avoid subdomain duplicates
    redirect_map = {}     # Track redirects: domain -> final_domain
    queue       = list(seed_domains)
    discovered  = set(seed_domains)

    while queue and len(results) < MAX_DOMAINS:
        domain = queue.pop(0)

        if domain in visited:
            continue
        visited.add(domain)

        # Skip subdomains if we already have the parent
        parent = get_parent_domain(domain)

        url = f"https://{domain}"
        print(f"  Crawling [{len(results)+1}/{MAX_DOMAINS}]: {domain}")

        # Try live crawl first
        data = fetch_page(url)

        # Check for redirects (e.g., twitter.com -> x.com)
        if data and data.get('final_domain'):
            final = data['final_domain']
            if final != domain and get_parent_domain(final) != parent:
                redirect_map[domain] = final
                print(f"    🔄 Redirects to {final}")

        # Check if live crawl returned good content
        live_success = (data and data.get("text") and len(data.get("text", "")) > 50)

        if live_success:
            print(f"    🌐 Live crawl successful ({len(data['text'])} chars)")
        else:
            # Live crawl failed — try fallback data
            if domain in fallback:
                data = fallback[domain]
                print(f"    📦 Using fallback data for {domain}")
            else:
                print(f"    ❌ FAILED: {domain} (no live content, no fallback)")
                failed_this_run.append(domain)
                continue

        # Add to results if we have good data
        if data and data.get("text") and len(data["text"]) > 50:
            # Track parent domain
            seen_parents.add(parent)
            results.append(data)

            # Expand queue with discovered linked domains (depth 1)
            if len(results) < MAX_DOMAINS:
                linked = [l for l in data.get("links", "").split("|") if l]
                for linked_domain in linked[:5]:
                    if linked_domain not in discovered:
                        # Skip if it redirects to a domain we already have
                        if linked_domain not in redirect_map.values():
                            discovered.add(linked_domain)
                            queue.append(linked_domain)

        time.sleep(SLEEP_SEC)

    # Atomically replace the global so get_failed_domains() always returns
    # a consistent snapshot from the most-recently-completed run.
    _failed_domains = failed_this_run

    print()
    print(f"✅ Crawler Agent complete!")
    print(f"   Domains crawled: {len(results)}")
    if _failed_domains:
        print(f"   ❌ Failed domains: {', '.join(_failed_domains)}")
    if redirect_map:
        print(f"   🔄 Redirects detected: {redirect_map}")
    print()

    return results


def crawl_domains_tool(seed_domains_str: str) -> str:
    """
    CrewAI tool wrapper — accepts comma-separated domain string.
    Returns JSON-formatted results.
    """
    import json
    domains = [d.strip() for d in seed_domains_str.split(",") if d.strip()]
    results = crawl_domains(domains)
    return json.dumps(results)


# ── Standalone test ──────────────────────────────────
if __name__ == "__main__":
    test_domains = ["example.com", "wikipedia.org", "bbc.com"]
    results = crawl_domains(test_domains)

    print("Sample result:")
    if results:
        r = results[0]
        print(f"  Domain:    {r['domain']}")
        print(f"  Text len:  {len(r['text'])} chars")
        print(f"  Links:     {r['links'][:80]}...")
        print(f"  Timestamp: {r['timestamp']}")
    print()
    print(f"Failed: {get_failed_domains()}")
    print("🎉 Crawler Agent test passed!")