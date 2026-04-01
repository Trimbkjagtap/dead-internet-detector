# Dead Internet Detector

Detects coordinated synthetic content ecosystems on the web using graph analysis, Graph Neural Networks, and multi-agent AI.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://dead-internet-detector-64strrdbhdelggehvnsu4x.streamlit.app)
[![API](https://img.shields.io/badge/API-Render-46E3B7?style=for-the-badge&logo=render)](https://dead-internet-detector.onrender.com)
[![API Docs](https://img.shields.io/badge/API_Docs-Swagger-85EA2D?style=for-the-badge&logo=swagger)](https://dead-internet-detector.onrender.com/docs)

INFO 7390 - Data Science Discovery & Architecture  
Northeastern University | MS Information Systems  
Student: Trimbkeshwar Jagtap

---

## What This Project Does

This project detects synthetic website ecosystems as a **network-level problem**, not an article-level problem.

Rather than judging one article in isolation, it checks whether a group of domains shows converging signals of coordination — shared infrastructure, shared authors, nearly identical content, and suspicious registration patterns — and produces an evidence dossier a journalist can follow up on.

---

## What Changed In V3

V3 expands the detection system from 3 signals to 7 and adds a journalist-facing evidence UI:

- **4 new detection signals**: IP/hosting overlap, link network insularity, Wayback Machine archive history, and cross-domain author byline matching
- **3-of-7 verdict rule**: SYNTHETIC only when 3 or more independent signals converge
- **Evidence tab** split into three journalist-facing sections: Content Similarity, Shared Hosting Infrastructure, Shared Author Bylines — each with actionable follow-up prompts
- **Article crawling**: crawler now fetches up to 3 article pages per domain to extract author bylines
- **IP + ASN enrichment**: resolves DNS → IP → hosting org via ipinfo.io with CDN allowlist (Cloudflare, Fastly, AWS, etc. are excluded)
- **Wayback CDX API integration**: checks archive history for brand-new sites or recent archive spikes using `matchType=domain` to correctly count all snapshots under a domain (not just the homepage URL)
- **7-signal heatmap** in Signal Analysis tab with plain-English explanation table
- **Per-domain breakdown** with help tooltips on all 7 signals and recomputed live verdict

---

## What Changed In V2

V2 added continuous monitoring and operational visibility:

- Budgeted WHOIS enrichment service with strict daily query limits and on-disk cache
- Live feed ingestion from WHOISDS and Reddit
- Batch monitor cycle that runs the analysis pipeline over incoming domains
- Async background monitor jobs to avoid frontend timeouts
- Timeline and feed status APIs for dashboard charts
- Streamlit monitor preview panel

---

## Live Demo

| Component | URL |
|-----------|-----|
| Frontend (Streamlit) | [dead-internet-detector.streamlit.app](https://dead-internet-detector-64strrdbhdelggehvnsu4x.streamlit.app) |
| Backend API (FastAPI) | [dead-internet-detector.onrender.com](https://dead-internet-detector.onrender.com) |
| API Documentation | [Swagger UI](https://dead-internet-detector.onrender.com/docs) |

Note: Render free tier can sleep after inactivity. First request after sleep may take 30–60 seconds.

---

## Architecture

### Main Analysis Path

1. Streamlit sends a seed domain to the FastAPI backend.
2. A fast heuristic preliminary verdict is returned immediately; a full background analysis is queued.
3. The 4-agent crew pipeline runs: Crawl → Fingerprint → Graph → Verdict.
4. Neo4j stores domain nodes, similarity edges, hosting edges, and author edges.
5. The API returns cluster verdict, per-domain signal breakdown, and evidence for the journalist.

### Monitoring Path

1. WHOISDS and Reddit feeds are ingested on a schedule.
2. Domains are merged, deduplicated, and chunked into monitor batches.
3. Each batch runs through the existing analysis pipeline.
4. Run summaries are appended to `data/monitor_runs.jsonl`.
5. Streamlit visualizes feed counts and timeline trends.

---

## Detection Signals (3-of-7 Rule)

A domain is flagged **SYNTHETIC** when at least 3 of 7 signals converge. 1–2 signals = **REVIEW**. 0 = **ORGANIC**.

| # | Signal | Method | What It Detects |
|---|--------|--------|-----------------|
| 1 | Content Similarity | Sentence Transformers (`all-MiniLM-L6-v2`) + cosine similarity | Near-identical homepage text across domains |
| 2 | Publishing Cadence | Isolation Forest anomaly detection | Inhuman or highly coordinated posting patterns |
| 3 | WHOIS Registration | WhoisXML API + heuristic fallback | Very new domains or suspicious registrar patterns |
| 4 | Shared Hosting | DNS → IP via socket + ipinfo.io ASN lookup | Domains sharing an IP address or hosting provider (CDNs excluded) |
| 5 | Link Network | Outbound link parsing + insularity scoring | Domains that mutually link to each other or link primarily within the cluster |
| 6 | Archive History | Wayback Machine CDX API | Sites with almost no archive history (new_site) or a sudden recent snapshot spike |
| 7 | Author Overlap | Article page crawling + regex byline extraction | The same author name appearing on multiple different domains |

---

## Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Multi-Agent Pipeline | 4 custom agents (Crawler, Fingerprint, Graph Builder, Verdict) | End-to-end analysis orchestration |
| LLM Analysis | OpenAI GPT-4o | Journalist-facing AI assessment |
| Domain Classification | Graph Convolutional Network (GCN) | Cluster-level synthetic/organic classification |
| Anomaly Detection | Isolation Forest | Cadence anomaly scoring |
| Content Embeddings | Sentence Transformers `all-MiniLM-L6-v2` | Semantic similarity |
| IP / ASN Enrichment | ipinfo.io + socket DNS | Hosting overlap detection |
| Archive History | Wayback Machine CDX API | Site age and activity pattern |
| Graph Database | Neo4j AuraDB | Domain nodes + SIMILAR_TO / SAME_HOST / SHARED_AUTHOR edges |
| Cache | `data/enrichment_cache.json` | 24-hour TTL for IP and Wayback lookups |

---

## Local Quick Start

### Prerequisites

- Python 3.12+
- Neo4j AuraDB instance
- OpenAI API key
- Optional: WhoisXML API key (recommended for stronger Signal 3)
- Optional: ipinfo.io token (recommended for Signal 4; free tier works without it)

### 1. Clone

```bash
git clone https://github.com/Trimbkjagtap/dead-internet-detector.git
cd dead-internet-detector
```

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create or edit `.env`:

```env
# Required
OPENAI_API_KEY=sk-...
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# Signal 3 — WHOIS (optional but recommended)
WHOIS_API_KEY=at_your-key-here
WHOIS_STRICT_BUDGET_MODE=true
WHOIS_DAILY_QUERY_LIMIT=15
WHOIS_CACHE_FILE=data/whois_cache.json
WHOIS_YOUNG_DOMAIN_DAYS=90

# Signal 4 — Hosting overlap (optional)
IPINFO_TOKEN=your-ipinfo-token

# Signal 5 — Link network (optional tuning)
INSULAR_SCORE_THRESHOLD=0.5
MIN_LINKS_FOR_INSULAR=3
MIN_DOMAINS_FOR_INSULAR=5

# Signal 6 — Wayback Machine (optional tuning)
WAYBACK_MIN_SNAPSHOTS=5
WAYBACK_RECENT_DAYS=30
WAYBACK_SPIKE_RATIO=0.5

# Content similarity threshold
SIM_THRESHOLD=0.45
SIM_EDGE_FLAG_THRESHOLD=0.45

# Monitoring (optional)
WHOISDS_DAILY_URL=
WHOISDS_SUSPICIOUS_TLDS=.xyz,.top,.click,.online,.site,.info,.buzz,.icu
WHOISDS_MAX_DOMAINS=500
MONITOR_BATCH_SIZE=10
MONITOR_MAX_ANALYZE_PER_RUN=40
MONITOR_INTERVAL_HOURS=6
```

### 5. Start backend API

```bash
uvicorn main:app --reload --port 8000
```

### 6. Start frontend UI (new terminal)

```bash
streamlit run app.py
```

Open http://localhost:8501

---

## Monitoring Operations

### Run a monitor cycle once

```bash
python -m pipeline.monitor
```

### Start recurring scheduler

```bash
python -m pipeline.schedule_monitor
```

### Monitor output files

- `data/incoming/whoisds_domains.json`
- `data/incoming/reddit_domains.json`
- `data/monitor_runs.jsonl`
- `data/whois_cache.json`
- `data/enrichment_cache.json`

---

## API Endpoints

### Core

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root metadata |
| GET | `/health` | Health check |
| POST | `/analyze` | Run full analysis pipeline on seed domains |
| POST | `/lookup` | Fast single-domain lookup (preliminary + background full analysis) |
| GET | `/lookup/job/{job_id}` | Poll background lookup job |
| GET | `/graph` | Graph nodes and edges for visualization |
| GET | `/graph/neighborhood/{domain}` | Focused neighborhood graph for one domain |
| GET | `/stats` | Graph summary stats |

### Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/feed-status` | Latest feed ingestion summary |
| GET | `/timeline?limit=20` | Recent monitor run timeline |
| POST | `/monitor/run` | Synchronous monitor run |
| POST | `/monitor/start` | Start async monitor run; returns `job_id` |
| GET | `/monitor/job/{job_id}` | Poll async monitor job status |

### Example: Analyze domains

```bash
curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"domains": ["suspicious-news.com", "fake-updates.net"]}'
```

### Example: Single-domain lookup

```bash
curl -X POST http://localhost:8000/lookup \
     -H "Content-Type: application/json" \
     -d '{"domain": "infowars.com"}'
```

---

## Deployment (Render)

### Backend

- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Set all required environment variables in Render dashboard

### Frontend (Streamlit Cloud)

- Set `BACKEND_URL` to your deployed API URL (e.g. `https://dead-internet-detector.onrender.com`)

---

## Project Structure

```text
dead-internet-detector/
├── agents/
│   ├── crawler_agent.py        # Agent 1 — crawls homepage + article pages, extracts authors
│   ├── fingerprint_agent.py    # Agent 2 — computes all 7 signals
│   ├── graph_builder_agent.py  # Agent 3 — upserts Neo4j nodes + edges
│   ├── verdict_agent.py        # Agent 4 — GCN inference + 3-of-7 rule verdict
│   └── crew.py                 # Wires agents into full pipeline
├── config/
│   └── signal_config.py        # All signal thresholds and env vars
├── services/
│   ├── whois_service.py        # Budgeted WHOIS enrichment
│   └── enrichment_service.py   # IP/ASN resolution + Wayback CDX API
├── pipeline/
│   ├── ingest_reddit.py
│   ├── ingest_whoisds.py
│   ├── monitor.py
│   └── schedule_monitor.py
├── database/
│   └── neo4j_client.py
├── models/
│   └── gnn_model.pt            # Trained GCN model
├── data/
├── main.py                     # FastAPI backend
├── app.py                      # Streamlit frontend
├── requirements.txt
├── Procfile
└── README.md
```

---

## Architectural Decisions and Trade-offs

### Why the GCN is a supporting signal, not the primary classifier

The original proposal positioned a Graph Neural Network as the core classifier. In implementation, the GCN ended up receiving only 5% weight in confidence scoring for one honest reason: the training dataset is too small to trust its outputs at inference time on unseen domains. A GCN generalizes from labeled graph structure — it needs many labeled examples of synthetic clusters to learn meaningful patterns. With fewer than a few hundred labeled domains, it overfits.

The architectural shift from "GNN as primary classifier" to "rule-based convergence of 7 independent signals, with GNN as a tiebreaker" is actually more defensible for the use case:

- **No black box verdicts.** A journalist can see exactly which signals fired and why. A GNN confidence score is uninterpretable.
- **Each signal is falsifiable.** You can check the Wayback snapshot count, verify the shared IP in WHOIS, confirm the author name on both sites. A GNN score cannot be independently verified.
- **Convergence is the thesis.** The project's core claim is that synthetic networks are detectable because *multiple independent signals converge* — not because any one ML model says so. The 7-signal rule makes that thesis explicit.

The GCN remains in the pipeline as a learned prior that can be improved with more labeled data. Retraining it on all 7 feature columns is the next step.

### Why REVIEW ≠ fake, and how the UI communicates this

The 3-of-7 convergence rule intentionally holds back from a SYNTHETIC verdict when only 1–2 signals fire. Content similarity at 54% between two right-wing outlets (infowars ↔ breitbart) is real signal but has an innocent explanation: topical and stylistic overlap between outlets covering the same ideological beat. The system surfaces it as evidence for a journalist to evaluate — it does not call the site fake.

The UI now explicitly explains *why* a REVIEW verdict did not cross the SYNTHETIC threshold, rather than leaving the user to wonder why the confidence is 68% but no signals appear confirmed.

---

## Known Limitations

1. The GCN model was trained on 3-signal features; retraining on all 7 is planned but not yet done. It receives 5% weight in confidence scoring.
2. Cadence detection works best with multiple domains in a batch; single-domain analysis has limited cadence signal.
3. Author extraction relies on regex byline patterns and may miss non-standard article layouts.
4. Async background jobs are stored in-memory and lost if the API process restarts.
5. WHOISDS ingestion requires a valid daily URL from a paid WHOISDS subscription.
6. WHOIS API usage is budget-constrained and falls back to heuristic mode when the daily limit is reached.
7. Render free tier cold starts can delay the first response by 30–60 seconds.

---

## Security Note

If any API keys were exposed during development or testing, rotate them immediately and update deployment secrets.

---

Built by Trimbkeshwar Jagtap | Northeastern University
