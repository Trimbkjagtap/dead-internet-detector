# pipeline/generate_synthetic.py
# Uses GPT-4o-mini to generate synthetic fake content ecosystems
# Run with: python3 pipeline/generate_synthetic.py

import os
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OUTPUT_FILE    = "data/synthetic_ecosystem.csv"
ARTICLES_EACH  = 5     # articles per fake domain
NUM_DOMAINS    = 10    # number of fake domains to generate

# ── 10 fake domain names for our synthetic ecosystem ──
FAKE_DOMAIN_NAMES = [
    "breaking-truth-daily.com",
    "real-news-network.net",
    "patriot-updates-now.com",
    "freedom-press-daily.org",
    "truth-insider-news.com",
    "americafirst-updates.net",
    "liberty-news-wire.com",
    "expose-the-truth.org",
    "national-alert-news.com",
    "thepeoplesvoice-news.net",
]

# ── Topics the fake network will push ──
FAKE_TOPICS = [
    "election fraud claims",
    "vaccine misinformation",
    "immigration fearmongering",
    "deep state conspiracy",
    "media bias narratives",
]


def generate_articles_for_domain(client, domain, topic, domain_num):
    """Ask GPT-4o-mini to generate fake coordinated articles."""

    prompt = f"""You are simulating a coordinated fake news network for AI research purposes.
Generate {ARTICLES_EACH} short news article paragraphs (2-3 sentences each) that:
1. All push the same misleading narrative about: {topic}
2. Sound like they come from different websites but repeat the same core claim
3. Use slightly different wording each time but convey the same message
4. Sound like real news articles

Format your response as {ARTICLES_EACH} paragraphs separated by '---'
Do NOT include any headlines or labels — just the paragraph text.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.8,
    )

    content = response.choices[0].message.content
    articles = [a.strip() for a in content.split("---") if a.strip()]
    return articles[:ARTICLES_EACH]


def generate_synthetic_ecosystem():
    print("=" * 55)
    print("Dead Internet Detector — Day 4")
    print("Generating synthetic fake ecosystem with GPT-4...")
    print("=" * 55)
    print()

    os.makedirs("data", exist_ok=True)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    all_rows = []

    for i, domain in enumerate(FAKE_DOMAIN_NAMES):
        # Cycle through topics
        topic = FAKE_TOPICS[i % len(FAKE_TOPICS)]

        print(f"  Generating domain {i+1}/{NUM_DOMAINS}: {domain}")
        print(f"  Topic: {topic}")

        try:
            articles = generate_articles_for_domain(client, domain, topic, i+1)

            for j, article_text in enumerate(articles):
                all_rows.append({
                    "domain":     domain,
                    "article_id": j + 1,
                    "topic":      topic,
                    "text":       article_text,
                    "label":      1,       # 1 = synthetic/fake
                    "source":     "gpt4_generated",
                })

            print(f"  ✅ {len(articles)} articles generated")
            print()

            # Small delay to avoid rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"  ❌ Error for {domain}: {e}")
            continue

    # Save to CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 55)
    print(f"✅ Total articles generated: {len(df)}")
    print(f"✅ Total fake domains:       {df['domain'].nunique()}")
    print(f"✅ Saved to: {OUTPUT_FILE}")
    print()
    print("Sample of generated content:")
    print()
    sample = df.iloc[0]
    print(f"Domain: {sample['domain']}")
    print(f"Topic:  {sample['topic']}")
    print(f"Text:   {sample['text'][:200]}...")
    print()
    print("=" * 55)
    print("🎉 Synthetic ecosystem data ready!")
    print("=" * 55)


if __name__ == "__main__":
    generate_synthetic_ecosystem()