# pipeline/preprocess.py
# Cleans raw Common Crawl domain data
# Input:  data/domains_raw.csv
# Output: data/domains_clean.csv
# Run with: python3 pipeline/preprocess.py

import re
import os
import pandas as pd
from datetime import datetime

INPUT_FILE  = "data/domains_raw.csv"
OUTPUT_FILE = "data/domains_clean.csv"
MIN_TEXT_LEN = 100   # minimum characters of text to keep a row


# ── Helper Functions ─────────────────────────────────────────────────────

def clean_text(text):
    """Remove special characters, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    # Remove HTML entities
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    # Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
    # Collapse whitespace
    text = ' '.join(text.split())
    return text.strip()


def is_english(text):
    """Simple check if text is mostly English using common words."""
    if not isinstance(text, str) or len(text) < 50:
        return False
    # Check for common English words
    english_words = [
        'the', 'and', 'for', 'that', 'this', 'with',
        'are', 'from', 'have', 'not', 'but', 'they',
        'was', 'his', 'her', 'you', 'can', 'will',
        'been', 'more', 'also', 'than', 'its', 'were'
    ]
    text_lower = text.lower()
    word_count = sum(1 for w in english_words if f' {w} ' in text_lower)
    return word_count >= 3  # at least 3 common English words


def normalize_domain(domain):
    """Standardize domain format."""
    if not isinstance(domain, str):
        return None
    domain = domain.lower().strip()
    # Remove www. prefix
    if domain.startswith('www.'):
        domain = domain[4:]
    # Remove trailing dots or slashes
    domain = domain.rstrip('./').strip()
    # Must contain at least one dot
    if '.' not in domain:
        return None
    # Must be reasonable length
    if len(domain) < 4 or len(domain) > 100:
        return None
    return domain


def clean_links(links):
    """Clean the pipe-separated links field."""
    if not isinstance(links, str) or links == 'nan':
        return ""
    # Split, clean, rejoin
    parts = [l.strip() for l in links.split('|') if l.strip()]
    # Filter out invalid links
    parts = [l for l in parts if '.' in l and len(l) > 4]
    return '|'.join(parts[:15])  # max 15 links


def parse_timestamp(ts):
    """Standardize timestamp format."""
    if not isinstance(ts, str):
        return ""
    try:
        # Common Crawl format: 2024-12-01T17:20:07Z
        dt = datetime.strptime(ts[:19], '%Y-%m-%dT%H:%M:%S')
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(ts)[:19]


# ── Main Cleaning Function ────────────────────────────────────────────────

def clean_data():
    print("=" * 55)
    print("Dead Internet Detector — Day 5")
    print("Cleaning raw domain data...")
    print("=" * 55)
    print()

    # ── Load raw data ──
    print(f"📂 Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Raw rows loaded: {len(df)}")
    print()

    # ── Step 1: Normalize domain names ──
    print("🔧 Step 1: Normalizing domain names...")
    df['domain'] = df['domain'].apply(normalize_domain)
    df = df.dropna(subset=['domain'])
    print(f"   Rows after domain normalization: {len(df)}")

    # ── Step 2: Remove duplicate domains ──
    print("🔧 Step 2: Removing duplicate domains...")
    before = len(df)
    df = df.drop_duplicates(subset=['domain'], keep='first')
    print(f"   Removed {before - len(df)} duplicates → {len(df)} rows")

    # ── Step 3: Clean text ──
    print("🔧 Step 3: Cleaning text content...")
    df['text'] = df['text'].apply(clean_text)

    # ── Step 4: Remove rows with short text ──
    print("🔧 Step 4: Removing pages with too little text...")
    before = len(df)
    df = df[df['text'].str.len() >= MIN_TEXT_LEN]
    print(f"   Removed {before - len(df)} short pages → {len(df)} rows")

    # ── Step 5: Filter English only ──
    print("🔧 Step 5: Keeping English pages only...")
    before = len(df)
    df = df[df['text'].apply(is_english)]
    print(f"   Removed {before - len(df)} non-English pages → {len(df)} rows")

    # ── Step 6: Clean links and timestamps ──
    print("🔧 Step 6: Cleaning links and timestamps...")
    df['links']     = df['links'].apply(clean_links)
    df['timestamp'] = df['timestamp'].apply(parse_timestamp)

    # ── Step 7: Add text length column ──
    df['text_len'] = df['text'].str.len()

    # ── Step 8: Reset index ──
    df = df.reset_index(drop=True)

    # ── Save ──
    print()
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # ── Summary ──
    print("=" * 55)
    print(f"✅ Final clean rows:  {len(df)}")
    print(f"✅ Unique domains:    {df['domain'].nunique()}")
    print(f"✅ Saved to:          {OUTPUT_FILE}")
    print()
    print("Columns:", list(df.columns))
    print()
    print("Text length stats:")
    print(f"  Min:  {df['text_len'].min()} chars")
    print(f"  Max:  {df['text_len'].max()} chars")
    print(f"  Mean: {df['text_len'].mean():.0f} chars")
    print()
    print("Sample clean domains:")
    print(df[['domain', 'text_len', 'timestamp']].head(5).to_string(index=False))
    print()
    print("=" * 55)
    print("🎉 Cleaning complete! data/domains_clean.csv is ready.")
    print("=" * 55)


if __name__ == "__main__":
    clean_data()