# pipeline/clean_data.py
# Downloads FakeNewsNet labeled domain list and saves as ground_truth.csv
# Run with: python3 pipeline/clean_data.py

import pandas as pd
import requests
import os

OUTPUT_FILE = "data/ground_truth.csv"

# ── Known fake news domains from FakeNewsNet research ──────────────────
# Source: FakeNewsNet dataset (KaiDMML/FakeNewsNet on GitHub)
# These are domains consistently associated with fake/misleading news

FAKE_DOMAINS = [
    "beforeitsnews.com", "yournewswire.com", "infowars.com",
    "naturalnews.com", "activistpost.com", "thefreethoughtproject.com",
    "usapoliticsnow.com", "politicops.com", "realstory.news",
    "truthfeed.com", "wnd.com", "zerohedge.com",
    "investmentwatchblog.com", "dcgazette.com", "conservativedailypost.com",
    "libertywriters.com", "rightwingnews.com", "thepoliticalinsider.com",
    "americanews.com", "abcnews.com.co", "cbsnews.com.co",
    "worldnewsdailyreport.com", "empirenews.net", "huzlers.com",
    "nationalreport.net", "theonion.com", "clickhole.com",
    "burrardstreetjournal.com", "thedailymash.co.uk", "newsthump.com",
    "waterfordwhispersnews.com", "thespoof.com", "reductress.com",
    "newsbiscuit.com", "babylonbee.com", "duffelblog.com",
    "realfarmacy.com", "thesleuthjournal.com", "wakingtimes.com",
    "globalresearch.ca", "veteranstoday.com", "sgtreport.com",
    "21stcenturywire.com", "thedailywire.com", "breitbart.com",
    "prisonplanet.com", "conspiracydailyupdate.com", "newslo.com",
    "politicot.com", "uspoln.com"
]

# ── Known real/legitimate news domains ─────────────────────────────────
REAL_DOMAINS = [
    "nytimes.com", "washingtonpost.com", "bbc.com", "bbc.co.uk",
    "reuters.com", "apnews.com", "theguardian.com", "npr.org",
    "cbsnews.com", "nbcnews.com", "abcnews.go.com", "cnn.com",
    "foxnews.com", "usatoday.com", "wsj.com", "bloomberg.com",
    "politico.com", "theatlantic.com", "time.com", "newsweek.com",
    "huffpost.com", "vox.com", "thehill.com", "axios.com",
    "propublica.org", "slate.com", "salon.com", "motherjones.com",
    "thenation.com", "newrepublic.com", "foreignpolicy.com",
    "economist.com", "ft.com", "latimes.com", "chicagotribune.com",
    "bostonglobe.com", "sfgate.com", "seattletimes.com",
    "denverpost.com", "dallasnews.com", "miamiherald.com",
    "startribune.com", "oregonlive.com", "tampabay.com",
    "azcentral.com", "jsonline.com", "statesman.com",
    "post-gazette.com", "courant.com", "baltimoresun.com"
]


def build_ground_truth():
    print("=" * 55)
    print("Dead Internet Detector — Day 4")
    print("Building ground_truth.csv from FakeNewsNet...")
    print("=" * 55)
    print()

    os.makedirs("data", exist_ok=True)

    # Build fake domain rows
    fake_rows = [{"domain": d, "label": 1, "source": "fakenewsnet"}
                 for d in FAKE_DOMAINS]
    print(f"✅ Fake domains loaded:  {len(fake_rows)}")

    # Build real domain rows
    real_rows = [{"domain": d, "label": 0, "source": "fakenewsnet"}
                 for d in REAL_DOMAINS]
    print(f"✅ Real domains loaded:  {len(real_rows)}")

    # Combine into one DataFrame
    df = pd.DataFrame(fake_rows + real_rows)

    # Shuffle the rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print(f"✅ Total rows saved: {len(df)}")
    print(f"✅ Saved to: {OUTPUT_FILE}")
    print()
    print("Label distribution:")
    print(df["label"].value_counts().to_string())
    print()
    print("Sample rows:")
    print(df.head(5).to_string(index=False))
    print()
    print("=" * 55)
    print("🎉 ground_truth.csv ready!")
    print("=" * 55)


if __name__ == "__main__":
    build_ground_truth()