# 🕸️ Dead Internet Detector

> Detects coordinated synthetic content ecosystems on the web using graph analysis, Graph Neural Networks, and multi-agent AI.

**INFO 7390 — Data Science Discovery & Architecture**  
**Northeastern University | MS Information Systems**  
**Student:** Trimbkeshwar Jagtap

---

## 🎯 What This Does

The internet is increasingly populated by AI-generated fake website networks — coordinated ecosystems of synthetic domains designed to manipulate search rankings and public opinion. Existing tools check individual articles for AI content. This system asks the **network-level question**: is this entire corner of the internet a coordinated synthetic ecosystem?

The Dead Internet Detector treats synthetic content as a **graph problem** — mapping relationships between domains, detecting publishing cadence anomalies, and using a Graph Neural Network to classify domain clusters as SYNTHETIC or ORGANIC.

---

## 🏗️ Architecture

```
User → Streamlit Frontend → FastAPI Backend → CrewAI Agents (×4)
                                                    ↓
                                              Neo4j AuraDB
                                                    ↓
                                           GNN Model → Verdict
```

### The 4 AI Agents
| Agent | Role |
|-------|------|
| 🕷️ Crawler Agent | Fetches domain content and discovers linked sites |
| 🔬 Fingerprint Analyst | Computes 3 detection signals |
| 🏗️ Graph Builder | Creates/updates Neo4j domain graph |
| ⚖️ Verdict Agent | Runs GNN inference → final verdict |

### The 3 Signals
| Signal | Method | Flag Threshold |
|--------|--------|----------------|
| Content Similarity | Sentence Transformers + Cosine | > 0.85 |
| Cadence Anomaly | Isolation Forest | score < -0.1 |
| WHOIS Registration | Domain age + TLD heuristics | suspicious TLD |

**Verdict rule:** SYNTHETIC requires 2-of-3 signals to converge.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend | FastAPI + uvicorn |
| AI Agents | CrewAI |
| LLM | OpenAI GPT-4o |
| Graph Database | Neo4j AuraDB |
| ML Models | GCN (PyTorch) + Isolation Forest |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Data | Common Crawl, FakeNewsNet, GPT-4 synthetic |
| Deployment | Render (backend) + Streamlit Cloud (frontend) |

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.11+
- Neo4j AuraDB account (free)
- OpenAI API key (~$10 credit)

### 1. Clone the repo
```bash
git clone https://github.com/Trimbkjagtap/dead-internet-detector.git
cd dead-internet-detector
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Your `.env` file needs:
```
OPENAI_API_KEY=sk-proj-...
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=your-username
NEO4J_PASSWORD=your-password
WHOIS_API_KEY=at_...
```

### 5. Run the data pipeline (first time only)
```bash
python3 pipeline/download_common_crawl.py   # ~10 min
python3 pipeline/preprocess.py
python3 pipeline/compute_embeddings.py      # ~10 min
python3 pipeline/cadence_analysis.py
python3 pipeline/whois_features.py
python3 pipeline/merge_features.py
python3 database/neo4j_client.py            # builds graph
```

### 6. Train the GNN (first time only)
Upload `data/domain_features.csv` and `data/similarity_edges.csv` to Google Colab, run `ml/train_gnn.py`, download `gnn_model.pt` to `models/`.

### 7. Start the backend
```bash
uvicorn main:app --reload --port 8000
```

### 8. Start the frontend
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/analyze` | Analyze seed domains |
| GET | `/graph` | Get graph data for visualization |
| GET | `/stats` | Graph statistics |
| GET | `/docs` | Swagger UI documentation |

### Example: Analyze domains
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"domains": ["suspicious-news.com", "fake-updates.net"]}'
```

---

## 🧪 Running Tests
```bash
python3 -m pytest test_api.py -v
```

---

## 📁 Project Structure
```
dead-internet-detector/
├── agents/                  # CrewAI agent definitions
│   ├── crawler_agent.py     # Agent 1: web crawling
│   ├── fingerprint_agent.py # Agent 2: signal computation
│   ├── graph_builder_agent.py # Agent 3: Neo4j graph
│   ├── verdict_agent.py     # Agent 4: GNN verdict
│   └── crew.py              # Full pipeline orchestration
├── ml/                      # Machine learning
│   ├── train_gnn.py         # GNN training (run on Colab)
│   └── evaluate_gnn.py      # Model evaluation
├── pipeline/                # Data processing
│   ├── download_common_crawl.py
│   ├── preprocess.py
│   ├── compute_embeddings.py
│   ├── cadence_analysis.py
│   ├── whois_features.py
│   └── merge_features.py
├── database/
│   └── neo4j_client.py      # Neo4j graph builder
├── data/                    # Datasets (not in git)
├── models/                  # Trained models (not in git)
├── main.py                  # FastAPI backend
├── app.py                   # Streamlit frontend
├── test_api.py              # API tests
├── Procfile                 # Render deployment
└── requirements.txt
```

---

## 🌐 Live Demo

- **Frontend:** https://dead-internet-detector.streamlit.app *(coming Day 22)*
- **Backend API:** https://dead-internet-detector.onrender.com *(coming Day 21)*
- **API Docs:** https://dead-internet-detector.onrender.com/docs

---

## ⚠️ Ethical Considerations

- This tool is designed for research and journalism purposes
- False positives are mitigated by the 2-of-3 signal convergence rule
- Human review is required before any domain is publicly accused
- The system does not store personal data about website visitors

---

## 📊 Dataset Sources

| Dataset | Source | Use |
|---------|--------|-----|
| Common Crawl | commoncrawl.org | Raw web domain data |
| FakeNewsNet | GitHub/KaiDMML | Labeled fake/real domains |
| GPT-4 Synthetic | Generated | Training data for GNN |
| WHOIS XML API | whoisxmlapi.com | Domain registration data |

---

## 🎓 Course Context

This project was built for **INFO 7390: Data Science Discovery & Architecture** at Northeastern University. It demonstrates:
- Agentic AI frameworks (CrewAI)
- Graph Neural Networks
- Causal reasoning (2-of-3 signal convergence)
- Synthetic data generation and validation
- Full-stack deployment

---

*Built with ❤️ by Trimbkeshwar Jagtap | Northeastern University 2025*