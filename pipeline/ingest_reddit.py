import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import urlparse

import requests

SUBREDDITS = ["worldnews", "conspiracy", "politics", "news"]
USER_AGENT = "dead-internet-detector/2.0 (+monitor)"
MAX_POSTS_PER_SUBREDDIT = 100
OUTPUT_DIR = Path("data/incoming")
OUTPUT_FILE = OUTPUT_DIR / "reddit_domains.json"


def _extract_domain(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    host = parsed.netloc.lower().replace("www.", "")
    return host


def _extract_urls_from_text(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"https?://[^\s)\]]+", text)


def fetch_reddit_domains() -> Dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    domains: Set[str] = set()
    scanned_posts = 0

    headers = {"User-Agent": USER_AGENT}

    for subreddit in SUBREDDITS:
        url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={MAX_POSTS_PER_SUBREDDIT}"
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code != 200:
                continue
            children = resp.json().get("data", {}).get("children", [])
        except Exception:
            continue

        for child in children:
            data = child.get("data", {})
            scanned_posts += 1
            post_url = data.get("url_overridden_by_dest") or data.get("url") or ""
            body = data.get("selftext", "")

            for raw_url in [post_url] + _extract_urls_from_text(body):
                d = _extract_domain(raw_url)
                if d and "." in d and "reddit.com" not in d:
                    domains.add(d)

    payload = {
        "source": "reddit",
        "fetched_at": datetime.utcnow().isoformat(),
        "subreddits": SUBREDDITS,
        "scanned_posts": scanned_posts,
        "domain_count": len(domains),
        "domains": sorted(domains),
    }
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = fetch_reddit_domains()
    print(f"Reddit ingestion complete: {result['domain_count']} domains from {result['scanned_posts']} posts")
