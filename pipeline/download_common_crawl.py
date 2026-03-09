# pipeline/download_common_crawl.py
# Downloads one Common Crawl WARC file and extracts domain data
# Run with: python3 pipeline/download_common_crawl.py

import os
import gzip
import requests
import pandas as pd
from io import BytesIO
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────
# One small WARC file from Common Crawl (~300MB)
WARC_URL = (
    "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/"
    "segments/1733066035857.0/warc/"
    "CC-MAIN-20241201162023-20241201192023-00000.warc.gz"
)
MAX_RECORDS   = 2000   # stop after 2000 pages (fast for MVP)
OUTPUT_FILE   = "data/domains_raw.csv"
MIN_TEXT_LEN  = 100    # skip pages with less than 100 characters
# ────────────────────────────────────────────────────


def extract_domain(url):
    """Get just the domain from a full URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except:
        return None


def extract_links(soup, base_domain):
    """Get all outgoing links from a page."""
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http"):
            linked_domain = extract_domain(href)
            if linked_domain and linked_domain != base_domain:
                links.append(linked_domain)
    # Return unique links as a pipe-separated string
    return "|".join(list(set(links))[:20])  # max 20 links per page


def extract_text(soup):
    """Get clean text from HTML — remove scripts and styles."""
    # Remove script and style tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # Collapse whitespace
    text = " ".join(text.split())
    return text[:2000]  # keep first 2000 chars only


def download_and_parse():
    print("=" * 55)
    print("Dead Internet Detector — Day 3")
    print("Downloading Common Crawl sample...")
    print("=" * 55)
    print()

    # ── Make sure data folder exists ──
    os.makedirs("data", exist_ok=True)

    # ── Download the WARC file into memory ──
    print(f"📥 Downloading WARC file...")
    print(f"   URL: {WARC_URL}")
    print(f"   This may take 5–10 minutes (~300MB)...")
    print()

    response = requests.get(WARC_URL, stream=True, timeout=120)
    if response.status_code != 200:
        print(f"❌ Download failed. Status code: {response.status_code}")
        return

    # Download with progress bar
    total = int(response.headers.get("content-length", 0))
    buf = BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
        for chunk in response.iter_content(chunk_size=65536):
            buf.write(chunk)
            pbar.update(len(chunk))

    buf.seek(0)
    print()
    print("✅ Download complete!")
    print()

    # ── Parse the WARC file ──
    print(f"🔍 Parsing web pages (max {MAX_RECORDS} records)...")
    print()

    records = []
    count   = 0
    skipped = 0

    with tqdm(total=MAX_RECORDS, desc="Processing pages") as pbar:
        for record in ArchiveIterator(buf):

            # Only process HTML response records
            if record.rec_type != "response":
                continue
            if "text/html" not in (record.http_headers.get_header("Content-Type") or ""):
                continue

            try:
                url = record.rec_headers.get_header("WARC-Target-URI")
                if not url or not url.startswith("http"):
                    continue

                domain = extract_domain(url)
                if not domain:
                    continue

                # Get the timestamp
                timestamp = record.rec_headers.get_header("WARC-Date") or ""

                # Read and parse HTML
                content = record.content_stream().read()
                if len(content) < MIN_TEXT_LEN:
                    skipped += 1
                    continue

                soup = BeautifulSoup(content, "html.parser")
                text = extract_text(soup)

                if len(text) < MIN_TEXT_LEN:
                    skipped += 1
                    continue

                links = extract_links(soup, domain)

                records.append({
                    "domain":    domain,
                    "url":       url,
                    "text":      text,
                    "links":     links,
                    "timestamp": timestamp,
                })

                count += 1
                pbar.update(1)

                if count >= MAX_RECORDS:
                    break

            except Exception:
                skipped += 1
                continue

    # ── Save to CSV ──
    print()
    print(f"💾 Saving {len(records)} records to {OUTPUT_FILE}...")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)

    # ── Summary ──
    print()
    print("=" * 55)
    print(f"✅ Processed:  {count} pages")
    print(f"⏭️  Skipped:    {skipped} pages (too short / no HTML)")
    print(f"✅ Saved:      {len(df)} domains to {OUTPUT_FILE}")
    print()
    print("📊 Sample of what was saved:")
    print(df[["domain", "timestamp"]].head(5).to_string(index=False))
    print()
    print("Unique domains found:", df["domain"].nunique())
    print()
    print("=" * 55)
    print("🎉 Day 3 complete! Run next:")
    print("   python3 -c \"import pandas as pd; df=pd.read_csv('data/domains_raw.csv'); print(df.head())\"")
    print("=" * 55)


if __name__ == "__main__":
    download_and_parse()