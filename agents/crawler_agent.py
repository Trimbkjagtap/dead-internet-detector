# agents/crawler_agent.py
# Agent 1 — Crawler Agent
# Takes seed domains, fetches content, extracts text + links + timestamps

import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime


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


def fetch_page(url: str) -> dict | None:
    """
    Fetch a single URL and extract:
    - domain name
    - clean text content
    - outgoing links
    - timestamp
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

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer",
                          "header", "aside", "form"]):
            tag.decompose()

        # Extract text
        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())[:2000]

        # Extract outgoing links
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http"):
                linked = extract_domain(href)
                if linked and linked != extract_domain(url):
                    links.append(linked)
        links = list(set(links))[:15]

        return {
            "domain":    extract_domain(url),
            "url":       url,
            "text":      text,
            "links":     "|".join(links),
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "status":    "ok",
        }

    except requests.exceptions.Timeout:
        return {"domain": extract_domain(url), "url": url,
                "text": "", "links": "", "timestamp": "", "status": "timeout"}
    except Exception:
        return None


def crawl_domains(seed_domains: list) -> list:
    """
    Main crawling function.
    Takes a list of seed domain names.
    Returns list of dicts with domain data.

    Args:
        seed_domains: list of domain strings e.g. ['example.com', 'news.com']

    Returns:
        list of dicts with keys: domain, url, text, links, timestamp
    """
    print(f"🕷️  Crawler Agent starting...")
    print(f"   Seed domains: {seed_domains}")
    print(f"   Max domains:  {MAX_DOMAINS}")
    print()

    results     = []
    visited     = set()
    queue       = list(seed_domains)
    discovered  = set(seed_domains)

    while queue and len(results) < MAX_DOMAINS:
        domain = queue.pop(0)

        if domain in visited:
            continue
        visited.add(domain)

        url = f"https://{domain}"
        print(f"  Crawling [{len(results)+1}/{MAX_DOMAINS}]: {domain}")

        data = fetch_page(url)

        if data and data.get("text") and len(data["text"]) > 50:
            results.append(data)

            # Expand queue with discovered linked domains (depth 1)
            if len(results) < MAX_DOMAINS:
                linked = [l for l in data.get("links","").split("|") if l]
                for linked_domain in linked[:5]:
                    if linked_domain not in discovered:
                        discovered.add(linked_domain)
                        queue.append(linked_domain)
        else:
            print(f"    ⚠️  Skipped (no content or error)")

        time.sleep(SLEEP_SEC)

    print()
    print(f"✅ Crawler Agent complete!")
    print(f"   Domains crawled: {len(results)}")
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
    print("🎉 Crawler Agent test passed!")