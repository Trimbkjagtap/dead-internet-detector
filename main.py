# main.py
# FastAPI backend for the Dead Internet Detector
# Run locally with: uvicorn main:app --reload --port 8000
# Deploy to Render with: uvicorn main:app --host 0.0.0.0 --port $PORT

import os
import sys
import json
import traceback
import uuid
from typing import Dict, List
from datetime import datetime, timezone
from pathlib import Path

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


class LookupRequest(BaseModel):
    domain: str

    class Config:
        json_schema_extra = {
            "example": {
                "domain": "patriot-updates-now.com"
            }
        }


# ── In-memory job store (simple, no Redis needed) ────
_jobs = {}
MONITOR_LOG_FILE = Path("data/monitor_runs.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_domain(domain: str) -> str:
    d = (domain or "").strip().lower()
    d = d.replace("https://", "").replace("http://", "")
    d = d.split("/")[0].strip()
    return d


def _risk_copy(verdict: str) -> Dict[str, str]:
    if verdict == "SYNTHETIC":
        return {
            "risk_level": "HIGH_RISK",
            "headline": "High Risk - Likely part of a fake network",
        }
    if verdict == "REVIEW":
        return {
            "risk_level": "SUSPICIOUS",
            "headline": "Suspicious - Some warning signs detected",
        }
    return {
        "risk_level": "LOW_RISK",
        "headline": "Looks Legitimate - No warning signs",
    }


def _heuristic_preliminary_verdict(domain: str) -> Dict:
    from services.whois_service import WhoisService

    whois = WhoisService()
    whois_features = whois.get_domain_features(domain)

    tld_flags = [".xyz", ".top", ".click", ".online", ".site", ".info", ".buzz", ".icu"]
    keyword_flags = [
        "truth", "patriot", "freedom", "liberty", "alert", "insider",
        "expose", "breaking", "real-news", "updates-now", "daily-truth",
        "peoples-voice", "wire", "report", "first-news", "national-alert",
    ]

    reasons = []
    score = 0

    age_days = int(whois_features.get("domain_age_days", -1))
    if age_days >= 0 and age_days <= 90:
        score += 1
        reasons.append("Domain was registered very recently")

    if any(domain.endswith(tld) for tld in tld_flags):
        score += 1
        reasons.append("Domain uses a higher-risk TLD commonly seen in low-trust campaigns")

    if any(k in domain for k in keyword_flags) or domain.count("-") >= 2:
        score += 1
        reasons.append("Domain name pattern looks promotional or synthetic")

    whois_flagged = int(whois_features.get("whois_flagged", 0))
    if whois_flagged:
        score += 1
        reasons.append("WHOIS pattern triggered the registration-risk heuristic")

    if score >= 3:
        verdict = "SYNTHETIC"
        confidence = 0.74
    elif score >= 1:
        verdict = "REVIEW"
        confidence = 0.58
    else:
        verdict = "ORGANIC"
        confidence = 0.76

    base = _risk_copy(verdict)
    explanation = (
        "; ".join(reasons)
        if reasons else
        "No strong warning signs were found in quick domain-registration heuristics"
    )

    return {
        "source": "preliminary",
        "domain": domain,
        "cluster_verdict": verdict,
        "max_confidence": confidence,
        "risk_level": base["risk_level"],
        "headline": base["headline"],
        "summary": (
            "Preliminary check only. "
            "A full network analysis has been queued in the background."
        ),
        "domain_verdicts": {
            domain: {
                "verdict": verdict,
                "confidence": confidence,
                "signals_triggered": min(score, 3),
                "signal_1_similarity": 0,
                "signal_2_cadence": 0,
                "signal_3_whois": 1 if whois_flagged else 0,
                "explanation": explanation,
            }
        },
        "analysis_type": "preliminary",
        "analyzed_at": _now_iso(),
    }


def _run_lookup_analysis_job(job_id: str, domain: str) -> None:
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["started_at"] = _now_iso()
    try:
        from agents.crew import run_analysis
        result = run_analysis([domain])
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["finished_at"] = _now_iso()
        _jobs[job_id]["result"] = {
            "cluster_verdict": result.get("cluster_verdict", "UNKNOWN"),
            "summary": result.get("summary", ""),
            "max_confidence": result.get("max_confidence", 0),
        }
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["finished_at"] = _now_iso()
        _jobs[job_id]["error"] = str(e)
        _jobs[job_id]["traceback"] = traceback.format_exc(limit=5)


def _lookup_cached_domain(domain: str) -> Dict:
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            node = session.run(
                """
                MATCH (d:Domain {domain: $domain})
                OPTIONAL MATCH (d)-[r:SIMILAR_TO]-(other:Domain)
                RETURN
                    d.domain AS domain,
                    d.preliminary_verdict AS preliminary_verdict,
                    d.signals_triggered AS signals_triggered,
                    d.avg_similarity AS avg_similarity,
                    d.max_similarity AS max_similarity,
                    d.anomaly_score AS anomaly_score,
                    d.burst_score AS burst_score,
                    d.domain_age_days AS domain_age_days,
                    d.registrar AS registrar,
                    d.updated_at AS updated_at,
                    collect(DISTINCT other.domain)[0..8] AS related_domains,
                    count(DISTINCT other) AS related_count
                LIMIT 1
                """,
                domain=domain,
            ).single()
    finally:
        driver.close()

    if not node:
        return {}

    verdict = (node.get("preliminary_verdict") or "ORGANIC").upper()
    confidence = min(0.97, max(0.35, (int(node.get("signals_triggered") or 0) / 3.0) + 0.35))
    related_count = int(node.get("related_count") or 0)
    reasons = []

    if float(node.get("max_similarity") or 0) >= 0.82:
        reasons.append("High content similarity was observed with related domains")
    if int(node.get("domain_age_days") or -1) >= 0 and int(node.get("domain_age_days") or -1) <= 90:
        reasons.append("Domain registration is recent")
    if float(node.get("burst_score") or 0) >= 0.6:
        reasons.append("Publishing cadence appears coordinated")
    if related_count > 0:
        reasons.append(f"Connected to {related_count} related domain(s) in the network graph")

    if not reasons:
        reasons.append("No strong warning signals were found in stored graph features")

    base = _risk_copy(verdict)
    summary = ". ".join(reasons) + "."

    return {
        "source": "cache",
        "domain": domain,
        "cluster_verdict": verdict,
        "max_confidence": round(confidence, 2),
        "risk_level": base["risk_level"],
        "headline": base["headline"],
        "summary": summary,
        "analysis_type": "cached",
        "related_domains": [d for d in (node.get("related_domains") or []) if d and d != domain],
        "domain_verdicts": {
            domain: {
                "verdict": verdict,
                "confidence": round(confidence, 2),
                "signals_triggered": int(node.get("signals_triggered") or 0),
                "signal_1_similarity": 1 if float(node.get("max_similarity") or 0) >= 0.82 else 0,
                "signal_2_cadence": 1 if float(node.get("burst_score") or 0) >= 0.6 else 0,
                "signal_3_whois": 1 if int(node.get("domain_age_days") or -1) in range(0, 91) else 0,
                "explanation": summary,
            }
        },
        "analyzed_at": _now_iso(),
        "cached_updated_at": node.get("updated_at"),
    }


def _run_monitor_job(job_id: str) -> None:
    """
    Background worker for monitor runs. Updates in-memory job status.
    """
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
    try:
        from pipeline.monitor import run_monitor_cycle
        summary = run_monitor_cycle()
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
        _jobs[job_id]["summary"] = summary
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
        _jobs[job_id]["error"] = str(e)
        _jobs[job_id]["traceback"] = traceback.format_exc(limit=5)


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
            "lookup":  "POST /lookup",
            "lookup_job": "GET /lookup/job/{job_id}",
            "graph":   "GET  /graph",
            "stats":   "GET  /stats",
            "recently_detected": "GET /recently-detected",
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
        "timestamp": _now_iso(),
    }


@app.post("/lookup", tags=["Lookup"])
async def lookup_domain(request: LookupRequest, background_tasks: BackgroundTasks):
    """
    Fast single-domain lookup for real users.

    Flow:
    1) Check Neo4j cache for prior analysis and return instantly if found.
    2) If missing, return lightweight preliminary verdict in seconds.
    3) Queue full pipeline analysis in background for future cache hits.
    """
    domain = _normalize_domain(request.domain)
    if not domain:
        raise HTTPException(status_code=400, detail="Please provide a valid domain")

    try:
        cached = _lookup_cached_domain(domain)
        if cached:
            return {
                "status": "cached",
                **cached,
            }

        preliminary = _heuristic_preliminary_verdict(domain)
        job_id = str(uuid.uuid4())
        _jobs[job_id] = {
            "job_id": job_id,
            "job_type": "lookup_full_analysis",
            "domain": domain,
            "status": "queued",
            "created_at": _now_iso(),
            "result": None,
        }
        background_tasks.add_task(_run_lookup_analysis_job, job_id, domain)

        return {
            "status": "queued",
            **preliminary,
            "job_id": job_id,
            "job_status_url": f"/lookup/job/{job_id}",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lookup failed: {str(e)}")


@app.get("/lookup/job/{job_id}", tags=["Lookup"])
async def get_lookup_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Lookup job not found")
    if job.get("job_type") != "lookup_full_analysis":
        raise HTTPException(status_code=400, detail="Job is not a lookup analysis job")
    return job


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
        result['analyzed_at'] = datetime.now(timezone.utc).isoformat()
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


@app.get("/feed-status", tags=["Monitoring"])
async def get_feed_status():
    """
    Returns latest feed ingestion summary from monitor runs.
    """
    if not MONITOR_LOG_FILE.exists():
        return {
            "status": "no-data",
            "message": "No monitor runs found yet",
            "latest": None,
        }

    lines = MONITOR_LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return {
            "status": "no-data",
            "message": "No monitor runs found yet",
            "latest": None,
        }

    latest = json.loads(lines[-1])
    return {
        "status": "ok",
        "latest": latest,
    }


@app.get("/timeline", tags=["Monitoring"])
async def get_timeline(limit: int = 20):
    """
    Returns recent monitor runs as a timeline for dashboard charts.
    """
    if not MONITOR_LOG_FILE.exists():
        return {"timeline": []}

    lines = MONITOR_LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in lines[-max(limit, 1):]:
        try:
            entries.append(json.loads(line))
        except Exception:
            continue

    return {"timeline": entries}


@app.get("/recently-detected", tags=["Monitoring"])
async def recently_detected(limit: int = 10):
    """
    Returns recently updated suspicious/synthetic domains for homepage feed.
    """
    if limit < 1:
        limit = 1
    if limit > 50:
        limit = 50

    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (d:Domain)
                WHERE d.preliminary_verdict IN ['SYNTHETIC', 'REVIEW']
                RETURN
                    d.domain AS domain,
                    d.preliminary_verdict AS verdict,
                    d.signals_triggered AS signals,
                    d.max_similarity AS max_similarity,
                    d.domain_age_days AS domain_age_days,
                    d.updated_at AS updated_at
                ORDER BY d.updated_at DESC
                LIMIT $limit
                """,
                limit=limit,
            )
            results = []
            for r in rows:
                verdict = (r.get("verdict") or "REVIEW").upper()
                base = _risk_copy(verdict)
                reason_bits = []
                if float(r.get("max_similarity") or 0) >= 0.82:
                    reason_bits.append("high content similarity")
                if int(r.get("signals") or 0) >= 2:
                    reason_bits.append("multiple detection signals")
                if int(r.get("domain_age_days") or -1) in range(0, 91):
                    reason_bits.append("very recent domain registration")
                reason = ", ".join(reason_bits) if reason_bits else "suspicious network indicators"

                results.append({
                    "domain": r.get("domain"),
                    "verdict": verdict,
                    "risk_level": base["risk_level"],
                    "headline": base["headline"],
                    "signals_triggered": int(r.get("signals") or 0),
                    "updated_at": r.get("updated_at"),
                    "reason": reason,
                })
        driver.close()

        return {
            "status": "ok",
            "count": len(results),
            "items": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recently detected query failed: {str(e)}")


@app.post("/monitor/run", tags=["Monitoring"])
async def run_monitor_once():
    """
    Runs one ingest -> analyze monitor cycle and returns summary.
    """
    try:
        from pipeline.monitor import run_monitor_cycle
        summary = run_monitor_cycle()
        return {
            "status": "ok",
            "summary": summary,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Monitor run failed: {str(e)}",
        )


@app.post("/monitor/start", tags=["Monitoring"])
async def start_monitor_once(background_tasks: BackgroundTasks):
    """
    Starts one monitor run in the background and returns a job id immediately.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": None,
    }
    background_tasks.add_task(_run_monitor_job, job_id)
    return {
        "status": "accepted",
        "job_id": job_id,
    }


@app.get("/monitor/job/{job_id}", tags=["Monitoring"])
async def get_monitor_job(job_id: str):
    """
    Poll monitor job status and final summary.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Monitor job not found")
    return job


# ── Run directly ─────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)