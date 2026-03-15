# main.py
# FastAPI backend for the Dead Internet Detector
# Run locally with: uvicorn main:app --reload --port 8000
# Deploy to Render with: uvicorn main:app --host 0.0.0.0 --port $PORT

import os
import sys
import json
from typing import List
from datetime import datetime

sys.path.insert(0, '.')

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── App setup ────────────────────────────────────────
app = FastAPI(
    title="Dead Internet Detector API",
    description=(
        "Detects coordinated synthetic content ecosystems on the web "
        "using graph analysis, GNN models, and multi-agent AI."
    ),
    version="1.0.0",
)

# Allow all origins (needed for Streamlit frontend to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response Models ──────────────────────────

class AnalyzeRequest(BaseModel):
    domains: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "domains": ["example.com", "suspicious-news.net", "fake-updates.org"]
            }
        }


class HealthResponse(BaseModel):
    status:    str
    message:   str
    timestamp: str


class StatsResponse(BaseModel):
    total_nodes:       int
    total_edges:       int
    synthetic_domains: int
    organic_domains:   int


# ── In-memory job store (simple, no Redis needed) ────
_jobs = {}


# ── Helper: get Neo4j driver ─────────────────────────
def get_neo4j_driver():
    from neo4j import GraphDatabase
    uri      = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD", "").strip('"').strip("'")
    return GraphDatabase.driver(uri, auth=(username, password))


# ══════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — basic info."""
    return {
        "name":        "Dead Internet Detector API",
        "version":     "1.0.0",
        "description": "Detects coordinated synthetic content ecosystems",
        "endpoints": {
            "health":  "GET  /health",
            "analyze": "POST /analyze",
            "graph":   "GET  /graph",
            "stats":   "GET  /stats",
            "docs":    "GET  /docs",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns ok if the API is running.
    Render uses this to confirm the service is alive.
    """
    return {
        "status":    "ok",
        "message":   "Dead Internet Detector is running",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/analyze", tags=["Analysis"])
async def analyze_domains(request: AnalyzeRequest):
    """
    Run the full 4-agent pipeline on a list of seed domains.

    - Agent 1 crawls the domains
    - Agent 2 computes 3 signals (similarity, cadence, WHOIS)
    - Agent 3 updates the Neo4j graph
    - Agent 4 runs GNN inference and returns verdict

    Returns verdict JSON with confidence score and explanation.
    """
    # Validate input
    if not request.domains:
        raise HTTPException(status_code=400, detail="No domains provided")

    if len(request.domains) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 domains per request"
        )

    # Clean domain inputs
    domains = []
    for d in request.domains:
        d = d.strip().lower()
        # Remove protocol if included
        d = d.replace("https://", "").replace("http://", "")
        # Remove trailing slashes
        d = d.rstrip("/")
        if d:
            domains.append(d)

    if not domains:
        raise HTTPException(status_code=400, detail="No valid domains after cleaning")

    try:
        # Run the full pipeline
        from agents.crew import run_analysis
        result = run_analysis(domains)

        # Add metadata
        result['analyzed_at'] = datetime.utcnow().isoformat()
        result['api_version']  = "1.0.0"

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/graph", tags=["Graph"])
async def get_graph():
    """
    Returns the current Neo4j graph as JSON for visualization.
    Returns nodes (domains) and edges (similarity relationships).
    """
    try:
        driver = get_neo4j_driver()

        with driver.session() as session:
            # Get all domain nodes
            nodes_result = session.run("""
                MATCH (n:Domain)
                RETURN n.domain AS domain,
                       n.preliminary_verdict AS verdict,
                       n.signals_triggered AS signals,
                       n.avg_similarity AS avg_sim,
                       n.anomaly_score AS anomaly_score,
                       n.max_similarity AS max_sim
                LIMIT 200
            """)
            nodes = []
            for record in nodes_result:
                nodes.append({
                    "id":             record["domain"],
                    "domain":         record["domain"],
                    "verdict":        record["verdict"] or "ORGANIC",
                    "signals":        record["signals"] or 0,
                    "avg_similarity": round(float(record["avg_sim"] or 0), 4),
                    "anomaly_score":  round(float(record["anomaly_score"] or 0), 4),
                })

            # Get similarity edges
            edges_result = session.run("""
                MATCH (a:Domain)-[r:SIMILAR_TO]->(b:Domain)
                RETURN a.domain AS source,
                       b.domain AS target,
                       r.similarity AS similarity,
                       r.flagged AS flagged
                LIMIT 300
            """)
            edges = []
            for record in edges_result:
                edges.append({
                    "source":     record["source"],
                    "target":     record["target"],
                    "similarity": round(float(record["similarity"] or 0), 4),
                    "flagged":    bool(record["flagged"]),
                })

        driver.close()

        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Graph query failed: {str(e)}"
        )


@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    """
    Returns summary statistics of the current graph database.
    """
    try:
        driver = get_neo4j_driver()

        with driver.session() as session:
            total_nodes = session.run(
                "MATCH (n:Domain) RETURN count(n) AS c"
            ).single()['c']

            total_edges = session.run(
                "MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) AS c"
            ).single()['c']

            synthetic = session.run(
                "MATCH (n:Domain) WHERE n.preliminary_verdict='SYNTHETIC' "
                "RETURN count(n) AS c"
            ).single()['c']

        driver.close()

        return {
            "total_nodes":       total_nodes,
            "total_edges":       total_edges,
            "synthetic_domains": synthetic,
            "organic_domains":   total_nodes - synthetic,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Stats query failed: {str(e)}"
        )


# ── Run directly ─────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)