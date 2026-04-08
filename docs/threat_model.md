# Dead Internet Detector — Threat Model

**Version:** 5.0  
**Author:** INFO 7390 Advanced Data Science Project  
**Last Updated:** April 2026

---

## 1. System Purpose

The Dead Internet Detector identifies **coordinated synthetic content networks** — clusters of websites that appear to be independent news outlets but are secretly operated by a single actor to distribute coordinated narratives at scale. The system uses 7 independent signals and a 3-of-7 convergence rule to distinguish genuine coordination from coincidental similarity.

---

## 2. Adversary Definition

### 2.1 Primary Adversary: Medium-Budget Coordinated Actor

| Property | Description |
|----------|-------------|
| **Goal** | Make a network of 5–50 fake news domains appear as independent organic outlets to evade platform takedowns and automated detection |
| **Motivation** | Political influence, ad fraud, state-adjacent propaganda, commercial astroturfing |
| **Budget** | $500–$5,000/month — enough for bulk domain registration, shared cloud hosting, and templated CMS deployment. Not enough for per-domain custom infrastructure |
| **Technical sophistication** | Medium — uses off-the-shelf tools (WordPress, Cloudflare, GoDaddy bulk registration). Can write scripts. Cannot afford dedicated ML engineers |
| **Content strategy** | Templated article spinning, RSS feed reposting, minimal human editing, cron-based publishing schedules |

### 2.2 Out-of-Scope Adversaries

| Adversary Type | Why Out of Scope |
|----------------|-----------------|
| **State-sponsored (Tier 1)** | Unlimited infrastructure budget — can register domains through different registrars months apart, host on separate ASNs, hire human writers. Our signal set cannot reliably detect this at acceptable FPR |
| **Single-domain misinformation** | System is designed for network-level detection. A lone bad actor on one domain leaves no cross-domain fingerprints |
| **Human-written coordinated networks** | If content is genuinely written by different humans covering the same narrative independently, content similarity will not trigger. This is a deliberate design choice — we flag coordination, not viewpoint |
| **Social media bot networks** | Operates on platforms, not web domains — different attack surface |

---

## 3. Attack Surface and Signal Robustness

### 3.1 The 7 Signals and Their Evasion Cost

| Signal | What It Detects | Evasion Strategy | Evasion Cost | Our Mitigation |
|--------|----------------|-----------------|--------------|---------------|
| **S1 Content Similarity** | Near-identical article text | Use LLM paraphrasing per domain | High ($0.01–$0.10/article × scale) | Authority whitelist + threshold elevation; body-text-only embeddings |
| **S2 Cadence Anomaly** | Machine-like posting schedule | Add random jitter to cron jobs | Low — but reveals technical sophistication gap | Isolation Forest detects irregular-regular patterns, not just regularity |
| **S3 WHOIS Age** | Domains registered recently | Register domains 3+ months early (speculation) | Medium — increases lead time and cost | Combined with other signals; authority whitelist exempts established outlets |
| **S4 Shared Hosting** | Multiple domains on same server | Host each domain on separate accounts | High — multiplies cloud cost 10–50× | Same-day + same-registrar gate already required for CDN pairs |
| **S5 Insular Link Network** | Domains only linking to each other | Link to external mainstream sites | Low — easy to fake | Ratio threshold; external links don't require reciprocity |
| **S6 Wayback History** | No archive history (new site) | Pre-register domains and wait | Medium — requires 3–6 month lead time | Combined with S3 (WHOIS age corroborates) |
| **S7 Author Bylines** | Same fake author across domains | Use different fake names per domain | Low per-domain, high at scale (name generation + consistency) | GPT persona analysis; intersection requires same normalized name |

### 3.2 Multi-Signal Evasion Analysis

**The key insight:** The 3-of-7 rule requires the adversary to evade **at least 5 signals simultaneously** to avoid detection. The marginal cost of each additional evasion increases superlinearly:

```
Evading 1 signal:  Low cost   (e.g., randomize posting schedule)
Evading 2 signals: Medium cost (+ use different author names)
Evading 3 signals: High cost  (+ host on separate ASNs)
Evading 4 signals: Very high  (+ stagger domain registration dates)
Evading 5 signals: Prohibitive (+ generate unique content per domain)
```

A medium-budget adversary ($2,000/month) can realistically evade 2–3 signals. Evading all 7 requires behavior indistinguishable from legitimate independent journalism — at which point the network is effectively not harmful.

---

## 4. False Positive Controls

The system implements multiple protections against flagging legitimate news operations:

### 4.1 Authority Whitelist
Wire services (AP, Reuters, AFP, BBC, NYT, etc.) and established local outlets are in `AUTHORITY_DOMAINS`. For pairs involving these domains:
- Similarity threshold raised **50%** (both authority) or **20%** (one authority)
- Effective similarity score discounted by **50%** or **25%**

### 4.2 Syndication Detection
Explicit wire-service attribution text ("Associated Press", "Reuters —", "Copyright AP") in page content suppresses the similarity signal entirely, regardless of score.

### 4.3 Strict 1-of-7 Rule
A single signal firing — regardless of which signal or how high its score — never produces a REVIEW or SYNTHETIC verdict. Minimum 2 signals required for REVIEW; 3 for SYNTHETIC.

### 4.4 CDN Hosting Gate
For domains hosted on major providers (AWS, Google, Cloudflare, Akamai), Signal 4 (shared hosting) only fires if both domains share the **same WHOIS creation date AND same registrar**. Legitimate enterprises using the same CDN are never flagged.

### 4.5 Syndication vs Cloning Classifier
Each similarity pair is classified as `structural_clone`, `shared_journalistic`, or `topic_overlap` based on author byline intersection and similarity score. Only `structural_clone` contributes toward flag counts.

---

## 5. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|-----------|
| WHOIS budget (15 queries/day) | New domains may not get age-checked | Wayback Machine first-seen used as fallback |
| Single-domain evaluation | Signals 1, 5, 7 are weaker in isolation | Multi-domain cluster analysis is primary use case |
| Content language | Embeddings optimized for English | Non-English content may score lower similarity even when cloned |
| Dynamic content | JavaScript-rendered content not crawled | Affects Signal 1 for SPA-heavy sites |
| Ground truth quality | FakeNewsNet labels include satire sites (The Onion, Babylon Bee) labeled as "fake" | Satire ≠ synthetic network; may inflate recall artificially |

---

## 6. Ethical Guidelines

1. **Human review required** before any publication or enforcement action
2. **Research and journalism use only** — not for automated content removal
3. **No personally identifiable information** is stored about authors or readers
4. **Verdicts are probabilistic**, not definitive — confidence scores must be communicated
5. **Appeals process**: any flagged domain can be submitted for manual review

---

## 7. Scope Statement

The system is designed to detect **medium-budget coordinated inauthentic behavior** at the **network level**. It is not a general-purpose misinformation detector and makes no claims about the truthfulness of individual articles. A domain can be organic (LOOKS LEGIT) and still publish false information; a domain can be part of a synthetic network and occasionally publish accurate content.

**Detection target:** Infrastructure-level coordination, not content accuracy.
