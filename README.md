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

This project detects synthetic website ecosystems as a network-level problem rather than an article-level problem.

It combines:
- Content similarity analysis
- Publishing cadence anomaly detection
- WHOIS-based domain registration signals
- Graph-based cluster classification

Domains are flagged as high risk only when multiple independent signals converge.

---

## What Changed In V2

V2 adds continuous monitoring and operational visibility:

- Budgeted WHOIS enrichment service with strict daily query limits and on-disk cache
- Live feed ingestion from WHOISDS and Reddit
- Batch monitor cycle that runs the same analysis pipeline over incoming domains
- Async background monitor jobs to avoid frontend timeouts
- Timeline and feed status APIs for dashboard charts and monitoring
- Streamlit monitor preview panel to start and poll monitor jobs

---

## Live Demo

| Component | URL |
|-----------|-----|
| Frontend (Streamlit) | [dead-internet-detector.streamlit.app](https://dead-internet-detector-64strrdbhdelggehvnsu4x.streamlit.app) |
| Backend API (FastAPI) | [dead-internet-detector.onrender.com](https://dead-internet-detector.onrender.com) |
| API Documentation | [Swagger UI](https://dead-internet-detector.onrender.com/docs) |

Note: Render free tier can sleep after inactivity. First request after sleep may take around 30-60 seconds.

---

## Architecture

### Main Analysis Path

1. Streamlit sends seed domains to FastAPI.
2. Crew pipeline runs domain crawl, fingerprinting, graph updates, and verdict.
3. Neo4j stores domain nodes and similarity edges.
4. API returns cluster verdict and per-domain signal details.

### Monitoring Path (V2)

1. WHOISDS and Reddit feeds are ingested.
2. Domains are merged, deduplicated, and chunked into monitor batches.
3. Each batch runs through the existing analysis pipeline.
4. Run summaries are appended to `data/monitor_runs.jsonl`.
5. Streamlit visualizes feed counts and timeline trends.

---

## Detection Signals (2-of-3 Rule)

A domain is treated as synthetic risk when at least 2 of 3 signals converge:

| Signal | Method | What It Detects |
|--------|--------|-----------------|
| Content Similarity | Sentence Transformers + cosine similarity | Near-identical cross-domain content |
| Cadence Anomaly | Isolation Forest | Inhuman or highly regular publishing behavior |
| WHOIS Registration | Budgeted WHOIS enrichment + heuristics fallback | Very new or suspicious domain registration patterns |

---

## Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Multi-Agent Orchestration | CrewAI | Pipeline execution |
| LLM Reasoning | OpenAI GPT-4o | Agent-level reasoning and analysis support |
| Domain Classification | Graph Neural Network | Cluster-level synthetic/organic classification |
| Anomaly Detection | Isolation Forest | Cadence anomaly scoring |
| Content Embeddings | Sentence Transformers | Semantic similarity |
| Graph Database | Neo4j AuraDB | Domain and relationship storage |

---

## Local Quick Start

### Prerequisites
- Python 3.12+
- Neo4j AuraDB credentials
- OpenAI API key
- Optional: WHOIS XML API key (recommended for stronger Signal 3)

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

Create or edit `.env` with these values:

```env
OPENAI_API_KEY=sk-...
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
WHOIS_API_KEY=at_your-key-here

WHOIS_STRICT_BUDGET_MODE=true
WHOIS_DAILY_QUERY_LIMIT=15
WHOIS_CACHE_FILE=data/whois_cache.json
WHOIS_YOUNG_DOMAIN_DAYS=90

WHOISDS_DAILY_URL=
WHOISDS_SUSPICIOUS_TLDS=.xyz,.top,.click,.online,.site,.info,.buzz,.icu
WHOISDS_MAX_DOMAINS=500

MONITOR_BATCH_SIZE=10
MONITOR_MAX_ANALYZE_PER_RUN=40
MONITOR_INTERVAL_HOURS=6

SIM_THRESHOLD=0.45
SIM_EDGE_FLAG_THRESHOLD=0.45
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

## Monitoring Operations (V2)

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

---

## API Endpoints

### Core

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root metadata |
| GET | `/health` | Health check |
| POST | `/analyze` | Run full analysis pipeline on seed domains |
| GET | `/graph` | Read graph nodes and edges for visualization |
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

### Example: Start and poll monitor job

```bash
curl -X POST http://localhost:8000/monitor/start

curl http://localhost:8000/monitor/job/<job_id>
```

---

## Render Deployment Notes

### Backend

- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Ensure all required environment variables are configured on Render

### Frontend

- Set `BACKEND_URL` to your deployed API URL (for example `https://dead-internet-detector.onrender.com`)

---

## Project Structure

```text
dead-internet-detector/
├── agents/
│   ├── crawler_agent.py
│   ├── fingerprint_agent.py
│   ├── graph_builder_agent.py
│   ├── verdict_agent.py
│   └── crew.py
├── config/
│   └── signal_config.py
├── services/
│   └── whois_service.py
├── pipeline/
│   ├── ingest_reddit.py
│   ├── ingest_whoisds.py
│   ├── monitor.py
│   ├── schedule_monitor.py
│   └── ...
├── database/
│   └── neo4j_client.py
├── data/
├── models/
├── main.py
├── app.py
├── requirements.txt
├── Procfile
└── README.md
```

---

## Known Limitations

1. Async monitor jobs are stored in-memory and are lost if the API process restarts.
2. WHOISDS ingestion requires a valid daily URL.
3. WHOIS API usage is budget-constrained and can fall back to heuristic mode.
4. Render free tier cold starts can delay first response.
5. Model calibration and retraining work is still ongoing for full v2 roadmap completion.

---

## Security Note

If any real API keys were exposed during development or testing, rotate them immediately and update deployment secrets.

---

Built by Trimbkeshwar Jagtap | Northeastern University