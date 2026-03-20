# 🕸️ Dead Internet Detector

> Detects coordinated synthetic content ecosystems on the web using graph analysis, Graph Neural Networks, and multi-agent AI.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://dead-internet-detector-64strrdbhdelggehvnsu4x.streamlit.app)
[![API](https://img.shields.io/badge/API-Render-46E3B7?style=for-the-badge&logo=render)](https://dead-internet-detector.onrender.com)
[![API Docs](https://img.shields.io/badge/API_Docs-Swagger-85EA2D?style=for-the-badge&logo=swagger)](https://dead-internet-detector.onrender.com/docs)

**INFO 7390 — Data Science Discovery & Architecture**
**Northeastern University | MS Information Systems**
**Student:** Trimbkeshwar Jagtap

---

## 🎯 What This Does

The internet is increasingly populated by AI-generated fake website networks — coordinated ecosystems of synthetic domains designed to manipulate search rankings and public opinion. Existing tools check individual articles for AI content. This system asks the **network-level question**: is this entire corner of the internet a coordinated synthetic ecosystem?

The Dead Internet Detector treats synthetic content as a **graph problem** — mapping relationships between domains, detecting publishing cadence anomalies, and using a Graph Neural Network to classify domain clusters as SYNTHETIC or ORGANIC.

---

## 🌐 Live Demo

| Component | URL |
|-----------|-----|
| **Frontend (Streamlit)** | [dead-internet-detector.streamlit.app](https://dead-internet-detector-64strrdbhdelggehvnsu4x.streamlit.app) |
| **Backend API (FastAPI)** | [dead-internet-detector.onrender.com](https://dead-internet-detector.onrender.com) |
| **API Documentation** | [Swagger UI](https://dead-internet-detector.onrender.com/docs) |

> ⚠️ **Note:** The Render free tier spins down after 15 minutes of inactivity. The first request after sleep may take ~50 seconds to wake up. Please wait for it to respond.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER (Browser)                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              STREAMLIT FRONTEND (Streamlit Cloud)             │
│  • Domain input form      • Network graph visualization      │
│  • Signal heatmaps        • GPT-4o AI analysis               │
│  • Verdict display        • JSON report download             │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTPS (REST API)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               FASTAPI BACKEND (Render.com)                    │
│  /health  /analyze  /graph  /stats  /docs                    │
└──────┬───────────┬──────────────┬───────────────────────────┘
       │           │              │
       ▼           ▼              ▼
┌──────────┐ ┌──────────┐ ┌────────────────┐
│ CrewAI   │ │ OpenAI   │ │ Neo4j AuraDB   │
│ 4 Agents │ │ GPT-4o   │ │ Graph Database │
└──────────┘ └──────────┘ └────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    4-AGENT PIPELINE                           │
│                                                              │
│  Agent 1: Crawler        → Fetch content from domains        │
│  Agent 2: Fingerprint    → Compute 3 detection signals       │
│  Agent 3: Graph Builder  → Update Neo4j domain network       │
│  Agent 4: Verdict        → GNN inference → SYNTHETIC/ORGANIC │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Detection Signals (2-of-3 Rule)

A domain is flagged SYNTHETIC only if **2 or more** signals converge:

| Signal | Method | What It Detects |
|--------|--------|-----------------|
| 🔵 Content Similarity | Sentence Transformers + Cosine Similarity | Domains publishing near-identical content |
| 🟣 Cadence Anomaly | Isolation Forest (unsupervised ML) | Inhuman publishing schedules (e.g., every 47 min) |
| 🟢 WHOIS Registration | Domain age heuristics | Domains registered within days of each other |

---

## 🤖 AI & ML Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Multi-Agent Orchestration | **CrewAI** | 4 agents in sequence |
| LLM Reasoning | **OpenAI GPT-4o** | Agent reasoning + cluster analysis |
| Domain Classification | **Graph Neural Network (GNN)** | Classify domains as synthetic/organic |
| Anomaly Detection | **Isolation Forest** | Detect cadence anomalies |
| Content Embeddings | **Sentence Transformers** | Compute semantic similarity |
| Graph Database | **Neo4j AuraDB** | Store domain relationships |

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.12+
- Neo4j AuraDB account (free tier)
- OpenAI API key

### 1. Clone the repository
```bash
git clone https://github.com/Trimbkjagtap/dead-internet-detector.git
cd dead-internet-detector
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your actual keys:
#   OPENAI_API_KEY=sk-...
#   NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
#   NEO4J_USERNAME=neo4j
#   NEO4J_PASSWORD=your-password
```

### 5. Start the backend
```bash
uvicorn main:app --reload --port 8000
```

### 6. Start the frontend (new terminal)
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check — returns ok if API is running |
| POST | `/analyze` | Analyze seed domains (main pipeline) |
| GET | `/graph` | Get Neo4j graph data for visualization |
| GET | `/stats` | Graph summary statistics |
| GET | `/docs` | Interactive Swagger API documentation |

### Example: Analyze domains
```bash
curl -X POST https://dead-internet-detector.onrender.com/analyze \
  -H "Content-Type: application/json" \
  -d '{"domains": ["suspicious-news.com", "fake-updates.net"]}'
```

---

## 🧪 Running Tests
```bash
python3 -m pytest test_api.py -v
```
All 7 tests should pass.

---

## 📁 Project Structure
```
dead-internet-detector/
├── agents/                      # CrewAI agent definitions
│   ├── crawler_agent.py         # Agent 1: web crawling
│   ├── fingerprint_agent.py     # Agent 2: signal computation
│   ├── graph_builder_agent.py   # Agent 3: Neo4j graph updates
│   ├── verdict_agent.py         # Agent 4: GNN verdict
│   └── crew.py                  # Full pipeline orchestration
├── ml/                          # Machine learning models
│   ├── train_gnn.py             # GNN training script (Colab)
│   └── evaluate_gnn.py          # Model evaluation
├── pipeline/                    # Data processing pipeline
│   ├── download_common_crawl.py # Common Crawl data fetcher
│   ├── preprocess.py            # Data cleaning
│   ├── compute_embeddings.py    # Sentence Transformer embeddings
│   ├── cadence_analysis.py      # Isolation Forest anomaly detection
│   ├── whois_features.py        # WHOIS domain age features
│   └── merge_features.py        # Combine all features
├── database/
│   └── neo4j_client.py          # Neo4j graph database client
├── data/                        # Datasets (not in git)
├── models/                      # Trained models (not in git)
├── docs/
│   └── screenshots/             # App screenshots for README
├── main.py                      # FastAPI backend server
├── app.py                       # Streamlit frontend UI
├── test_api.py                  # API tests (7 tests)
├── requirements.txt             # Frontend dependencies (Streamlit Cloud)
├── requirements-backend.txt     # Backend dependencies (Render)
├── Procfile                     # Render deployment config
├── .env.example                 # Environment variable template
└── README.md                    # This file
```

---

## 📊 Dataset Sources

| Dataset | Source | Records | Use |
|---------|--------|---------|-----|
| Common Crawl | commoncrawl.org (CC-MAIN-2024-51) | ~2,000 domains | Raw web domain data |
| FakeNewsNet | GitHub/KaiDMML | ~100 domains | Ground truth (labeled fake/real) |
| GPT-4 Synthetic | Generated via GPT-4 | ~50 domains | Synthetic ecosystem training data |
| WHOIS Data | whoisxmlapi.com | Per domain | Domain registration ages |

---

## ⚠️ Known Limitations

1. **Render cold starts** — Free tier backend sleeps after 15 min; first request takes ~50 seconds
2. **CrewAI version** — Pinned to 0.11.2 for Python compatibility
3. **GNN training** — Must be done on Google Colab (requires GPU)
4. **Rate limits** — OpenAI API calls are subject to rate limits
5. **Synthetic domains** — Demo domains are fabricated for testing; real-world detection requires live WHOIS data

---

## ⚖️ Ethical Considerations

- Designed for **research and journalism** purposes only
- **2-of-3 signal rule** prevents false positives — no domain flagged by a single signal
- **Human review required** before any domain is publicly accused
- No personal data about website visitors is collected or stored
- Should not be used to target legitimate publishers without evidence

---

## 🎓 Course Context

This project was built for **INFO 7390: Data Science Discovery & Architecture** at Northeastern University.

It demonstrates:
- **Agentic AI frameworks** (CrewAI with 4 specialized agents)
- **Graph Neural Networks** for domain classification
- **Causal reasoning** (2-of-3 signal convergence rule)
- **Synthetic data generation** and validation
- **Full-stack deployment** (Streamlit Cloud + Render + Neo4j AuraDB)
- **Modern ML tooling** (Sentence Transformers, Isolation Forest, GNN)

---

*Built by Trimbkeshwar Jagtap | Northeastern University 2025*