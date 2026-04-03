# app.py
# Streamlit frontend for the Dead Internet Detector v3
# Run with: streamlit run app.py

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import time
import os
import math

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="Dead Internet Detector",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Backend URL ──────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
try:
    BACKEND_URL = st.secrets.get("BACKEND_URL", BACKEND_URL)
except Exception:
    pass

PUBLIC_BACKEND_URL = os.getenv("PUBLIC_BACKEND_URL", BACKEND_URL)
try:
    PUBLIC_BACKEND_URL = st.secrets.get("PUBLIC_BACKEND_URL", PUBLIC_BACKEND_URL)
except Exception:
    pass


# ── Custom CSS ───────────────────────────────────────
st.markdown("""
<style>
    /* ════════════════════════════════════════════════
       GLOBAL BACKGROUND  — deep space with dot grid
       ════════════════════════════════════════════════ */
    .stApp {
        background-color: #070b14 !important;
    }

    /* Layered background via the main scrollable area */
    [data-testid="stAppViewContainer"] {
        background-color: #070b14;
        background-image:
            radial-gradient(ellipse 80% 50% at 15% 0%, rgba(59,130,246,0.13) 0%, transparent 55%),
            radial-gradient(ellipse 60% 45% at 85% 100%, rgba(139,92,246,0.11) 0%, transparent 50%),
            radial-gradient(circle, rgba(255,255,255,0.035) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 30px 30px;
        background-attachment: fixed;
    }

    /* Remove default white padding Streamlit adds */
    .block-container {
        background: transparent !important;
        padding-top: 1.5rem !important;
        max-width: 1200px;
    }

    /* Header bar */
    [data-testid="stHeader"] { background: transparent !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1120 0%, #080c18 100%) !important;
        border-right: 1px solid rgba(59,130,246,0.18) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        border-radius: 10px; padding: 4px; gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; color: #8899bb !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(59,130,246,0.18) !important;
        color: #93c5fd !important;
    }

    /* Metric boxes */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px; padding: 14px 18px;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.025) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 12px !important;
    }

    /* Input */
    [data-testid="stTextInput"] input {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(59,130,246,0.30) !important;
        border-radius: 10px !important;
        color: #e0e8ff !important;
        font-size: 15px !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: rgba(59,130,246,0.70) !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
        border: none !important; border-radius: 10px !important;
        color: white !important; font-weight: 600 !important;
        box-shadow: 0 4px 18px rgba(37,99,235,0.40) !important;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 24px rgba(37,99,235,0.55) !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important; color: #c0d0ee !important;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.07) !important; }

    /* ── Hero box ── */
    .hero-box {
        background: linear-gradient(135deg,
            rgba(37,99,235,0.15) 0%,
            rgba(124,58,237,0.10) 50%,
            rgba(7,11,20,0.0) 100%);
        border: 1px solid rgba(59,130,246,0.25);
        border-radius: 20px;
        padding: 40px 44px;
        margin-bottom: 28px;
        text-align: center;
        backdrop-filter: blur(6px);
        position: relative; overflow: hidden;
    }
    .hero-box::before {
        content: "";
        position: absolute; top: -60px; left: 50%; transform: translateX(-50%);
        width: 320px; height: 120px;
        background: radial-gradient(ellipse, rgba(59,130,246,0.25), transparent 70%);
        pointer-events: none;
    }
    .hero-box h1 {
        font-size: 2.4rem; margin: 0 0 10px;
        background: linear-gradient(90deg, #93c5fd, #c4b5fd, #93c5fd);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; font-weight: 800; letter-spacing: -0.5px;
    }
    .hero-box p { font-size: 1.05rem; color: #8899bb; margin: 0; line-height: 1.7; }

    /* ── Verdict banners — glassmorphism style ── */
    .verdict-synthetic {
        background: linear-gradient(135deg,
            rgba(127,29,29,0.85) 0%, rgba(185,28,28,0.90) 50%, rgba(220,38,38,0.85) 100%);
        backdrop-filter: blur(10px);
        color: white; padding: 26px 36px; border-radius: 16px;
        text-align: center; font-size: 26px; font-weight: 700;
        margin: 16px 0; border: 1px solid rgba(239,68,68,0.50);
        box-shadow: 0 8px 32px rgba(220,38,38,0.40), inset 0 1px 0 rgba(255,255,255,0.10);
    }
    .verdict-organic {
        background: linear-gradient(135deg,
            rgba(5,46,22,0.85) 0%, rgba(21,128,61,0.90) 50%, rgba(34,197,94,0.80) 100%);
        backdrop-filter: blur(10px);
        color: white; padding: 26px 36px; border-radius: 16px;
        text-align: center; font-size: 26px; font-weight: 700;
        margin: 16px 0; border: 1px solid rgba(34,197,94,0.40);
        box-shadow: 0 8px 32px rgba(34,197,94,0.30), inset 0 1px 0 rgba(255,255,255,0.10);
    }
    .verdict-review {
        background: linear-gradient(135deg,
            rgba(69,26,3,0.85) 0%, rgba(180,83,9,0.90) 50%, rgba(234,88,12,0.80) 100%);
        backdrop-filter: blur(10px);
        color: white; padding: 26px 36px; border-radius: 16px;
        text-align: center; font-size: 26px; font-weight: 700;
        margin: 16px 0; border: 1px solid rgba(234,88,12,0.45);
        box-shadow: 0 8px 32px rgba(234,88,12,0.35), inset 0 1px 0 rgba(255,255,255,0.10);
    }
    .verdict-sub {
        font-size: 14px; font-weight: 400; opacity: 0.88;
        margin-top: 8px; display: block; letter-spacing: 0.1px;
    }

    /* ── Confidence bar — animated glow ── */
    .conf-bar-wrap {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; overflow: hidden;
        height: 20px; width: 100%; margin: 12px 0 6px;
    }
    .conf-bar-fill-red {
        height: 100%; border-radius: 12px;
        background: linear-gradient(90deg, #991b1b, #ef4444, #fca5a5);
        box-shadow: 0 0 12px rgba(239,68,68,0.60);
    }
    .conf-bar-fill-green {
        height: 100%; border-radius: 12px;
        background: linear-gradient(90deg, #14532d, #22c55e, #86efac);
        box-shadow: 0 0 12px rgba(34,197,94,0.50);
    }
    .conf-bar-fill-amber {
        height: 100%; border-radius: 12px;
        background: linear-gradient(90deg, #7c2d12, #f97316, #fdba74);
        box-shadow: 0 0 12px rgba(249,115,22,0.55);
    }
    .conf-label { font-size: 12px; color: #6b7a99; text-align: right; margin-top: 2px; }

    /* ── Glass info card ── */
    .info-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px; padding: 18px 22px; margin: 10px 0;
        backdrop-filter: blur(4px);
    }
    .info-card-title { font-weight: 700; font-size: 15px; color: #c8d8f0; margin-bottom: 6px; }
    .info-card-body  { font-size: 13px; color: #7a90b0; line-height: 1.7; }

    /* ── Signal pills ── */
    .signal-pill-on {
        display: inline-block;
        background: rgba(239,68,68,0.15); color: #fca5a5;
        border: 1px solid rgba(239,68,68,0.45); border-radius: 20px;
        padding: 4px 13px; font-size: 12px; font-weight: 600; margin: 3px 2px;
        box-shadow: 0 0 8px rgba(239,68,68,0.20);
    }
    .signal-pill-off {
        display: inline-block;
        background: rgba(34,197,94,0.10); color: #86efac;
        border: 1px solid rgba(34,197,94,0.30); border-radius: 20px;
        padding: 4px 13px; font-size: 12px; font-weight: 600; margin: 3px 2px;
    }

    /* ── Sidebar step card ── */
    .step-card {
        background: rgba(59,130,246,0.06);
        border-left: 3px solid #3b82f6;
        border-radius: 10px; padding: 11px 15px; margin: 6px 0;
    }
    .step-num  { color: #60a5fa; font-weight: 700; font-size: 13px; }
    .step-body { color: #8899bb; font-size: 13px; margin-top: 3px; line-height: 1.5; }

    /* ── Evidence card ── */
    .evidence-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px; padding: 16px 20px; margin: 8px 0;
    }
    .evidence-high { border-left: 4px solid #ef4444; }
    .evidence-med  { border-left: 4px solid #f97316; }
    .evidence-low  { border-left: 4px solid #3b82f6; }

    /* ── Monitor feed cards ── */
    .monitor-card {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px; padding: 16px; margin: 8px 0;
    }
    .feed-ok   { border-left: 4px solid #22c55e; }
    .feed-warn { border-left: 4px solid #f97316; }
    .feed-off  { border-left: 4px solid #ef4444; }

    /* ── Summary banner ── */
    .summary-banner {
        background: rgba(59,130,246,0.06);
        border: 1px solid rgba(59,130,246,0.20);
        border-left: 4px solid #3b82f6;
        border-radius: 12px; padding: 14px 20px; margin: 12px 0;
        font-size: 14px; color: #b0c4e0; line-height: 1.8;
    }

    /* ── Why-not box ── */
    .why-not-box {
        background: rgba(249,115,22,0.06);
        border: 1px dashed rgba(249,115,22,0.40);
        border-radius: 12px; padding: 14px 18px; margin: 10px 0;
        font-size: 13px; color: #fdba74; line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: safe backend GET ─────────────────────────
def api_get(path, timeout=15):
    try:
        r = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def confidence_bar(pct, color="red"):
    css_class = {"red": "conf-bar-fill-red", "green": "conf-bar-fill-green", "amber": "conf-bar-fill-amber"}.get(color, "conf-bar-fill-red")
    return (
        f'<div class="conf-bar-wrap">'
        f'<div class="{css_class}" style="width:{int(pct*100)}%"></div>'
        f'</div>'
        f'<div class="conf-label">{pct:.0%} confidence</div>'
    )


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🕸️ Dead Internet Detector")
    st.caption("INFO 7390 · Northeastern University")
    st.divider()

    # Backend status — first thing a user needs to know
    health = api_get("/health")
    if health and health.get("status") == "ok":
        st.success("✅ Backend online")
    else:
        st.error("❌ Backend offline")
        st.caption(f"Expected at: {BACKEND_URL}")

    st.divider()

    st.markdown("### 🧭 How It Works")
    steps = [
        ("1", "Crawl", "Fetches homepage + article text from the domain and up to 10 linked sites"),
        ("2", "Analyze", "Checks 7 independent signals — from content similarity to author names"),
        ("3", "Graph", "Builds a network map in Neo4j connecting related domains"),
        ("4", "Verdict", "GNN + 3-of-7 rule decides: Fake Network, Suspicious, or Legitimate"),
    ]
    for num, title, body in steps:
        st.markdown(
            f'<div class="step-card">'
            f'<div class="step-num">Step {num} · {title}</div>'
            f'<div class="step-body">{body}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### 🔬 The 7 Detection Signals")
    st.markdown("""
| # | Signal | What it checks |
|---|--------|---------------|
| 1 | Content Similarity | Is the text nearly identical to other domains? |
| 2 | Cadence Anomaly | Is the publishing schedule suspicious? |
| 3 | WHOIS | Is the domain brand-new or oddly registered? |
| 4 | Shared Hosting | Does it share a server with other suspect domains? |
| 5 | Link Network | Do the sites only link to each other? |
| 6 | Archive History | Does Wayback Machine barely know it exists? |
| 7 | Author Overlap | Do the same author names appear everywhere? |
""")
    st.caption("**3-of-7 rule:** A cluster is only flagged HIGH RISK when 3 or more independent signals fire. One or two signals can have innocent explanations.")

    st.divider()
    st.markdown("### 🚦 What the Verdicts Mean")
    st.markdown("""
<div style="margin:4px 0">🔴 <b>High Risk</b> — 3+ signals converge. Likely a coordinated fake network.</div>
<div style="margin:4px 0">🟡 <b>Suspicious</b> — 1–2 signals. Worth investigating, not confirmed.</div>
<div style="margin:4px 0">🟢 <b>Looks Legit</b> — No signals fired. Appears organic.</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════

# ── Hero ─────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
  <h1>🕸️ Dead Internet Detector</h1>
  <p>Paste any domain below. The system checks 7 independent signals to decide whether<br>
  the site is part of a <b>coordinated fake content network</b> — or just a normal website.</p>
</div>
""", unsafe_allow_html=True)

tab_analyze, tab_monitor = st.tabs(["🔍 Check a Domain", "📡 Live Monitor Dashboard"])


# ══════════════════════════════════════════════════════
# TAB 1 — ANALYZE DOMAIN
# ══════════════════════════════════════════════════════
with tab_analyze:

    # ── Recently flagged strip ────────────────────────
    recent = api_get("/recently-detected?limit=5")
    recent_items = recent.get("items", []) if recent else []

    if recent_items:
        rc1, rc2 = st.columns([1, 3])
        with rc1:
            st.metric("Recently Flagged", len(recent_items))
        with rc2:
            top = ", ".join([item.get("domain", "") for item in recent_items[:3]])
            st.caption(f"Latest flagged: **{top}**")
        with st.expander("See all recently detected domains"):
            for item in recent_items:
                icon = "🔴" if item.get("verdict") == "SYNTHETIC" else "🟡"
                st.markdown(
                    f"{icon} **{item.get('domain', '')}** — {item.get('headline', 'Suspicious activity')}  \n"
                    f"_Reason: {item.get('reason', 'Network indicators detected')}_"
                )

    # ── Search bar ───────────────────────────────────
    st.markdown("#### Enter a domain to check")
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        domain_input = st.text_input(
            "domain",
            value=st.session_state.get("lookup_domain", ""),
            placeholder="e.g.  naturalnews.com  or  suspicious-updates.xyz",
            label_visibility="collapsed",
        )
    with col_btn:
        check_clicked = st.button(
            "Check",
            type="primary",
            use_container_width=True,
            disabled=(not domain_input.strip()),
        )

    st.caption("ℹ️ Results are cached — first lookup triggers a full pipeline (≈60 s). Repeat checks are instant.")

    # ── Trigger lookup ────────────────────────────────
    if check_clicked and domain_input.strip():
        new_domain = domain_input.strip()
        if new_domain != st.session_state.get("lookup_domain", ""):
            st.session_state.pop("result", None)
            st.session_state.pop("ai_analysis", None)
            st.session_state.pop("report_id", None)
            st.session_state["lookup_job_id"] = None
        st.session_state["lookup_domain"] = new_domain
        try:
            with st.spinner("Connecting to backend…"):
                t0 = time.time()
                resp = requests.post(
                    f"{BACKEND_URL}/lookup",
                    json={"domain": new_domain},
                    timeout=20,
                )
            if resp.status_code != 200:
                st.error(f"Lookup failed: {resp.text}")
            else:
                lookup = resp.json()
                st.session_state["result"] = lookup
                if lookup.get("status") == "queued" and lookup.get("job_id"):
                    st.session_state["lookup_job_id"] = lookup.get("job_id")
                else:
                    st.session_state["lookup_job_id"] = None
                st.rerun()
        except requests.exceptions.Timeout:
            st.error("Lookup timed out. Try again in a few seconds.")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to backend at {BACKEND_URL}")
        except Exception as e:
            st.error(f"Error: {e}")

    # ── Auto-poll background job ──────────────────────
    job_id = st.session_state.get("lookup_job_id")
    if job_id:
        job = api_get(f"/lookup/job/{job_id}", timeout=10)
        job_status = job.get("status") if job else None

        if job_status in ("queued", "running"):
            st.info("⏳ **Deep analysis in progress…** The page will refresh automatically when it's done.")
            time.sleep(5)
            st.rerun()
        elif job_status == "completed":
            full_result = job.get("full_result")
            if full_result:
                full_result["analysis_type"] = "fresh"
                full_result["analyzed_at"] = job.get("finished_at", "")
                st.session_state["result"] = full_result
            else:
                domain_for_lookup = job.get("domain", "")
                if domain_for_lookup:
                    try:
                        cached_resp = requests.post(
                            f"{BACKEND_URL}/lookup",
                            json={"domain": domain_for_lookup},
                            timeout=15,
                        )
                        if cached_resp.status_code == 200:
                            st.session_state["result"] = cached_resp.json()
                    except Exception:
                        pass
            st.session_state["lookup_job_id"] = None
            st.rerun()
        elif job_status == "failed":
            st.error(f"Analysis failed: {job.get('error', 'unknown error')}")
            st.session_state["lookup_job_id"] = None

    # ══════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════
    if "result" in st.session_state:
        result          = st.session_state["result"]
        verdict         = result.get("cluster_verdict", "UNKNOWN")
        confidence      = result.get("max_confidence", 0.0)
        summary         = result.get("summary", "")
        headline        = result.get("headline", "Verdict")
        analysis_type   = result.get("analysis_type", "")
        domain_verdicts = result.get("domain_verdicts", {})

        st.divider()

        # ── Verdict banner ────────────────────────────
        VERDICT_CSS   = {"SYNTHETIC": "verdict-synthetic", "REVIEW": "verdict-review"}.get(verdict, "verdict-organic")
        VERDICT_ICON  = {"SYNTHETIC": "🔴", "REVIEW": "🟡"}.get(verdict, "🟢")
        VERDICT_COLOR = {"SYNTHETIC": "red", "REVIEW": "amber"}.get(verdict, "green")
        VERDICT_PLAIN = {
            "SYNTHETIC": "This site shows strong signs of belonging to a coordinated fake news network.",
            "REVIEW":    "Some warning signs were found. Worth investigating further — not confirmed fake.",
        }.get(verdict, "No suspicious patterns detected. This site appears to be a normal, independent domain.")

        col_v, col_c = st.columns([3, 2])
        with col_v:
            st.markdown(
                f'<div class="{VERDICT_CSS}">'
                f'{VERDICT_ICON} {headline}'
                f'<span class="verdict-sub">{VERDICT_PLAIN}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_c:
            st.markdown("#### Confidence Score")
            st.markdown(confidence_bar(confidence, VERDICT_COLOR), unsafe_allow_html=True)
            st.caption(
                "**How confident is this verdict?** "
                "Based on the number of signals that fired, content similarity score, and GNN output. "
                "Higher = more evidence of coordination."
            )

        # ── Summary box ───────────────────────────────
        if analysis_type != "preliminary":
            st.markdown(f'<div class="summary-banner">📋 {summary}</div>', unsafe_allow_html=True)
            if verdict == "REVIEW":
                max_signals = max(
                    (dv.get("signals_triggered", 0) for dv in domain_verdicts.values()),
                    default=0
                )
                st.markdown(
                    f'<div class="why-not-box">'
                    f'<b>Why "Suspicious" and not "High Risk"?</b><br>'
                    f'The detector requires 3 or more independent signals to confirm a coordinated fake network '
                    f'(the 3-of-7 convergence rule). The highest signal count here is <b>{max_signals}/7</b>. '
                    f'One or two signals can have innocent explanations — for example, two right-leaning outlets '
                    f'will naturally share similar vocabulary without being coordinated. '
                    f'Treat this as a lead worth investigating, not a confirmed verdict.'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Metrics row ───────────────────────────────
        _display_domain = result.get("domain") or (
            result.get("seed_domains", [""])[0] if result.get("seed_domains") else "—"
        )
        if isinstance(_display_domain, str) and _display_domain.startswith("www."):
            _display_domain = _display_domain[4:]

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1: st.metric("Domain checked", _display_domain)
        with mc2: st.metric("🔴 High Risk",   result.get("synthetic_domains", 1 if verdict == "SYNTHETIC" else 0))
        with mc3: st.metric("🟡 Suspicious",  result.get("review_domains",   1 if verdict == "REVIEW"    else 0))
        with mc4: st.metric("🟢 Looks Legit", result.get("organic_domains",  1 if verdict == "ORGANIC"   else 0))

        st.divider()

        # ── Result sub-tabs ───────────────────────────
        evidence_pairs   = result.get("evidence_pairs",   [])
        hosting_evidence = result.get("hosting_evidence", [])
        author_evidence  = result.get("author_evidence",  [])
        total_evidence   = len(evidence_pairs) + len(hosting_evidence) + len(author_evidence)

        r_tab1, r_tab2, r_tab3, r_tab4, r_tab5 = st.tabs([
            "🕸️ Network Graph",
            "📊 Signal Heatmap",
            "🤖 AI Analysis",
            "📋 Per-Domain Details",
            f"🔍 Evidence ({total_evidence})",
        ])

        # ══════════════════════════════════════════════
        # SUB-TAB 1 — NETWORK GRAPH
        # ══════════════════════════════════════════════
        with r_tab1:
            st.markdown("#### Domain Similarity Network")
            st.markdown(
                '<div class="info-card">'
                '<div class="info-card-title">How to read this chart</div>'
                '<div class="info-card-body">'
                'The <b>large center node</b> is the domain you searched. '
                'Each surrounding node is a domain that shares suspiciously similar content. '
                '<b>Lines (edges)</b> connect two domains above the similarity threshold. '
                'Hover over any node to see its verdict and signal count. '
                '<span style="color:#e74c3c">● Red</span> = High Risk &nbsp; '
                '<span style="color:#e67e22">● Amber</span> = Suspicious &nbsp; '
                '<span style="color:#27ae60">● Green</span> = Looks Legit'
                '</div></div>',
                unsafe_allow_html=True,
            )

            graph_domain = st.session_state.get("lookup_domain", "")
            graph_data   = api_get(f"/graph/neighborhood/{graph_domain}") if graph_domain else None
            cmap         = {"SYNTHETIC": "#e74c3c", "REVIEW": "#e67e22", "ORGANIC": "#27ae60"}

            if graph_data and graph_data.get("nodes"):
                nodes          = graph_data["nodes"]
                edges          = graph_data["edges"]
                seed_nodes     = [nd for nd in nodes if nd.get("is_seed")]
                neighbor_nodes = [nd for nd in nodes if not nd.get("is_seed")]

                if not neighbor_nodes:
                    seed_verdict = seed_nodes[0].get("verdict", "ORGANIC") if seed_nodes else "ORGANIC"
                    color_icon   = {"SYNTHETIC": "🔴", "REVIEW": "🟡"}.get(seed_verdict, "🟢")
                    st.markdown(f"### {color_icon} {graph_domain}")
                    st.info(
                        f"**{graph_domain}** is in the database but has no similar domains connected to it yet. "
                        "This means no other stored domain exceeded the similarity threshold (0.45). "
                        "Check the **Evidence** tab — content comparisons may have found partial matches."
                    )
                else:
                    pos = {}
                    if seed_nodes:
                        pos[seed_nodes[0]["id"]] = (0.0, 0.0)
                    nb_count = len(neighbor_nodes)
                    for idx, nd in enumerate(neighbor_nodes):
                        angle = 2 * math.pi * idx / max(nb_count, 1)
                        pos[nd["id"]] = (math.cos(angle), math.sin(angle))

                    ex, ey = [], []
                    for e in edges:
                        s, t = pos.get(e["source"]), pos.get(e["target"])
                        if s is not None and t is not None:
                            ex += [s[0], t[0], None]
                            ey += [s[1], t[1], None]

                    placed_nodes = [nd for nd in nodes if nd["id"] in pos]
                    node_x       = [pos[nd["id"]][0] for nd in placed_nodes]
                    node_y       = [pos[nd["id"]][1] for nd in placed_nodes]
                    node_colors  = [cmap.get(nd.get("verdict", "ORGANIC"), "#888") for nd in placed_nodes]
                    node_sizes   = [28 if nd.get("is_seed") else 14 for nd in placed_nodes]
                    node_labels  = [nd["domain"] for nd in placed_nodes]
                    hover_texts  = [
                        f"<b>{nd['domain']}</b><br>"
                        f"Verdict: {nd.get('verdict','ORGANIC')}<br>"
                        f"Signals: {nd.get('signals', 0)}/7"
                        for nd in placed_nodes
                    ]

                    fig = go.Figure(
                        data=[
                            go.Scatter(
                                x=ex, y=ey, mode="lines",
                                line=dict(width=1.5, color="#3d4f6b"),
                                hoverinfo="none",
                            ),
                            go.Scatter(
                                x=node_x, y=node_y,
                                mode="markers+text",
                                text=node_labels,
                                textposition="top center",
                                textfont=dict(size=10, color="#d0d8e8"),
                                hovertemplate="%{customdata}<extra></extra>",
                                customdata=hover_texts,
                                marker=dict(
                                    size=node_sizes,
                                    color=node_colors,
                                    line=dict(width=2, color="#ffffff"),
                                    opacity=0.92,
                                ),
                            ),
                        ],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode="closest",
                            paper_bgcolor="#0e1117",
                            plot_bgcolor="#0e1117",
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.55, 1.55]),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.55, 1.55]),
                            height=540,
                            margin=dict(l=20, r=20, t=20, b=20),
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        f"**{graph_domain}** shares similar content with **{len(neighbor_nodes)}** "
                        f"other domain(s). Lines connect domains above the 0.45 similarity threshold."
                    )

            elif graph_domain:
                st.info(
                    f"**{graph_domain}** has not been stored in the graph database yet. "
                    "Run an analysis first, then return here."
                )
            else:
                st.info("Search for a domain above — the similarity network will appear here.")

        # ══════════════════════════════════════════════
        # SUB-TAB 2 — SIGNAL HEATMAP
        # ══════════════════════════════════════════════
        with r_tab2:
            st.markdown("#### Signal Heatmap — Which signals fired for each domain?")
            st.markdown(
                '<div class="info-card">'
                '<div class="info-card-title">How to read this chart</div>'
                '<div class="info-card-body">'
                'Each row is one signal. Each column is one domain. '
                '<b style="color:#e74c3c">Red cell</b> = that signal fired (suspicious). '
                '<b style="color:#27ae60">Dark cell</b> = signal is clear (normal). '
                'A domain needs <b>3+ red cells</b> to be classified as High Risk.'
                '</div></div>',
                unsafe_allow_html=True,
            )

            if domain_verdicts:
                rows = []
                for d, dv in domain_verdicts.items():
                    rows.append({
                        "Domain":        d[:35],
                        "Verdict":       dv.get("verdict", "ORGANIC"),
                        "Signals":       dv.get("signals_triggered", 0),
                        "1·Similarity":  dv.get("signal_1_similarity", 0),
                        "2·Cadence":     dv.get("signal_2_cadence", 0),
                        "3·WHOIS":       dv.get("signal_3_whois", 0),
                        "4·Hosting":     dv.get("signal_4_hosting", 0),
                        "5·Links":       dv.get("signal_5_link_network", 0),
                        "6·Wayback":     dv.get("signal_6_wayback", 0),
                        "7·Authors":     dv.get("signal_7_authors", 0),
                    })
                df_sig     = pd.DataFrame(rows)
                heat_cols  = ["1·Similarity","2·Cadence","3·WHOIS","4·Hosting","5·Links","6·Wayback","7·Authors"]
                heat       = df_sig[["Domain"] + heat_cols].set_index("Domain")
                fig2 = px.imshow(
                    heat.T,
                    color_continuous_scale=["#0d2218", "#e74c3c"],
                    zmin=0, zmax=1,
                    title="Signal Heatmap — Red = Triggered, Dark = Clear",
                    aspect="auto",
                    labels={"x": "Domain", "y": "Signal", "color": "Triggered"},
                )
                fig2.update_layout(
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font_color="#e0e0e0", height=360,
                )
                st.plotly_chart(fig2, use_container_width=True)

                sm1, sm2, sm3 = st.columns(3)
                with sm1: st.metric("Max signals fired", f"{int(df_sig['Signals'].max())}/7")
                with sm2: st.metric("High risk domains",  len(df_sig[df_sig["Signals"] >= 3]))
                with sm3: st.metric("Needs review",        len(df_sig[(df_sig["Signals"] >= 1) & (df_sig["Signals"] < 3)]))

                st.markdown("#### What each signal checks")
                st.markdown(
                    "| # | Signal | What it checks |\n"
                    "|---|--------|----------------|\n"
                    "| 1 | Content Similarity | Homepage text is unusually similar to other known domains |\n"
                    "| 2 | Cadence Anomaly | Publishing time pattern looks machine-generated |\n"
                    "| 3 | WHOIS | Domain is very new or has suspicious registration data |\n"
                    "| 4 | Shared Hosting | Shares IP address or hosting provider with another domain |\n"
                    "| 5 | Link Network | Domains in the cluster mutually link to each other (insular) |\n"
                    "| 6 | Wayback Archive | Site has almost no archive history — very new or recently faked |\n"
                    "| 7 | Author Overlap | Same author name appears on articles across multiple domains |"
                )

        # ══════════════════════════════════════════════
        # SUB-TAB 3 — AI ANALYSIS
        # ══════════════════════════════════════════════
        with r_tab3:
            st.markdown("#### 🤖 GPT-4o Journalist Briefing")
            st.markdown(
                '<div class="info-card">'
                '<div class="info-card-body">'
                'Feeds all signal results into GPT-4o and asks it to act as a journalist '
                'assessing whether this looks like a real news operation or a coordinated fake network. '
                'This is <b>advisory only</b> — always apply human judgment.'
                '</div></div>',
                unsafe_allow_html=True,
            )

            if st.button("Generate AI Analysis", type="primary"):
                with st.spinner("GPT-4o analyzing…"):
                    try:
                        from openai import OpenAI
                        from dotenv import load_dotenv
                        load_dotenv()
                        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                        domain_summary = "\n".join([
                            f"- {d}: {v.get('verdict')} "
                            f"(signals={v.get('signals_triggered',0)}/7 — "
                            f"similarity={v.get('signal_1_similarity',0)}, "
                            f"hosting={v.get('signal_4_hosting',0)}, "
                            f"wayback={v.get('signal_6_wayback',0)}, "
                            f"authors={v.get('signal_7_authors',0)})"
                            for d, v in domain_verdicts.items()
                        ])
                        prompt = f"""You are helping a journalist investigate a potential coordinated fake news network.

Domain analysis results (7 signals checked per domain):
{domain_summary}

Overall verdict: {verdict}

Signals explained: 1=content similarity, 2=publishing cadence, 3=WHOIS registration, 4=shared hosting IP, 5=insular link network, 6=Wayback Machine history, 7=shared author names.

Provide: (1) what the triggered signals suggest about coordination, (2) whether this looks like a real news operation or a fake network, (3) the most suspicious specific findings, (4) what a journalist should investigate next (e.g. check ownership records, interview sources). Be direct and concise."""

                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=350, temperature=0.3,
                        )
                        st.session_state["ai_analysis"] = resp.choices[0].message.content
                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")

            if "ai_analysis" in st.session_state:
                st.markdown("---")
                st.markdown("##### Assessment")
                st.info(st.session_state["ai_analysis"])
                st.caption("⚠️ AI output is advisory — never publish solely based on automated analysis.")

        # ══════════════════════════════════════════════
        # SUB-TAB 4 — PER-DOMAIN DETAILS
        # ══════════════════════════════════════════════
        with r_tab4:
            st.markdown("#### Per-Domain Breakdown")
            st.markdown(
                '<div class="info-card">'
                '<div class="info-card-body">'
                'Expand each domain card to see exactly which signals fired and why. '
                'Each domain is checked independently against all 7 signals. '
                '<b>3 or more red signals = High Risk.</b>'
                '</div></div>',
                unsafe_allow_html=True,
            )

            if domain_verdicts:
                for d, dv in domain_verdicts.items():
                    s1 = int(dv.get("signal_1_similarity",  0))
                    s2 = int(dv.get("signal_2_cadence",     0))
                    s3 = int(dv.get("signal_3_whois",       0))
                    s4 = int(dv.get("signal_4_hosting",     0))
                    s5 = int(dv.get("signal_5_link_network",0))
                    s6 = int(dv.get("signal_6_wayback",     0))
                    s7 = int(dv.get("signal_7_authors",     0))
                    signals_fired = s1 + s2 + s3 + s4 + s5 + s6 + s7

                    v = "SYNTHETIC" if signals_fired >= 3 else ("REVIEW" if signals_fired >= 1 else "ORGANIC")

                    max_sim  = float(dv.get("max_similarity", 0) or 0)
                    _burst   = float(dv.get("burst_score",    0) or 0)
                    if signals_fired >= 3:
                        recomputed_conf = round(max(0.65, min(0.97,
                            0.65 + min((signals_fired - 3) / 4.0, 1.0) * 0.20
                            + min(max_sim * 0.20, 0.10) + min(_burst * 0.15, 0.06))), 2)
                    elif signals_fired >= 1:
                        recomputed_conf = round(max(0.02, min(0.64,
                            0.40 + (signals_fired - 1) * 0.10
                            + min(max_sim * 0.20, 0.10) + min(_burst * 0.15, 0.06))), 2)
                    else:
                        recomputed_conf = round(max(0.70, min(1.0 - min(max_sim * 0.20, 0.20), 0.97)), 2)

                    icon = "🔴" if v == "SYNTHETIC" else ("🟡" if v == "REVIEW" else "🟢")

                    with st.expander(
                        f"{icon}  {d}  —  {v}  —  {signals_fired}/7 signals triggered  —  Confidence: {recomputed_conf:.0%}",
                        expanded=(v != "ORGANIC"),
                    ):
                        # Signal pills row
                        signal_defs = [
                            (s1, "1·Similarity"), (s2, "2·Cadence"), (s3, "3·WHOIS"),
                            (s4, "4·Hosting"),    (s5, "5·Links"),   (s6, "6·Wayback"), (s7, "7·Authors"),
                        ]
                        pills_html = "".join(
                            f'<span class="signal-pill-on">🚨 {label}</span>' if fired
                            else f'<span class="signal-pill-off">✅ {label}</span>'
                            for fired, label in signal_defs
                        )
                        st.markdown(pills_html, unsafe_allow_html=True)
                        st.markdown("")

                        # Row 1: signals 1–3
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            sim_val = float(dv.get("max_similarity") or 0)
                            st.metric("1 · Content Similarity",
                                      "🚨 Triggered" if s1 else "✅ Clear",
                                      delta=f"score {sim_val:.3f} (threshold 0.45)" if sim_val else "no match found",
                                      delta_color="inverse" if s1 else "off",
                                      help="Is the homepage text nearly identical to another domain in the database?")
                        with c2:
                            burst = float(dv.get("burst_score") or 0)
                            st.metric("2 · Publishing Cadence",
                                      "🚨 Triggered" if s2 else "✅ Clear",
                                      delta=f"anomaly score {burst:.3f}" if burst else None,
                                      delta_color="inverse" if s2 else "off",
                                      help="Does the publishing schedule look machine-generated or abnormal?")
                        with c3:
                            age = int(dv.get("domain_age_days") or -1)
                            age_str = f"{age} days old" if age > 0 else "age unknown"
                            st.metric("3 · WHOIS Registration",
                                      "🚨 Triggered" if s3 else "✅ Clear",
                                      delta=age_str,
                                      delta_color="inverse" if s3 else "off",
                                      help="Is the domain brand-new or registered in a suspicious pattern?")

                        # Row 2: signals 4–7
                        c4, c5, c6, c7 = st.columns(4)
                        with c4:
                            org = dv.get("hosting_org", "") or ""
                            ip  = dv.get("ip_address",  "") or ""
                            st.metric("4 · Shared Hosting",
                                      "🚨 Triggered" if s4 else "✅ Clear",
                                      delta=org[:28] if org else (ip or "no data"),
                                      delta_color="inverse" if s4 else "off",
                                      help="Does this domain share an IP address or hosting provider with another suspect domain?")
                        with c5:
                            insular = float(dv.get("insular_score") or 0)
                            st.metric("5 · Link Network",
                                      "🚨 Triggered" if s5 else "✅ Clear",
                                      delta=f"{insular:.0%} links within cluster" if insular else None,
                                      delta_color="inverse" if s5 else "off",
                                      help="Do the domains in this cluster mostly link to each other and nobody else?")
                        with c6:
                            snaps    = int(dv.get("wayback_snapshot_count") or 0)
                            wb_reason= dv.get("wayback_flag_reason", "") or ""
                            wb_label = "new site" if wb_reason == "new_site" else ("archive spike" if wb_reason else f"{snaps} snapshots")
                            st.metric("6 · Archive History",
                                      "🚨 Triggered" if s6 else "✅ Clear",
                                      delta=wb_label,
                                      delta_color="inverse" if s6 else "off",
                                      help="Does the Wayback Machine barely know this site exists? Very new = suspicious.")
                        with c7:
                            try:
                                import json as _json
                                shared = _json.loads(dv.get("shared_authors", "[]") or "[]")
                            except Exception:
                                shared = []
                            auth_label = shared[0] if shared else ("shared author found" if s7 else "no overlap")
                            st.metric("7 · Author Overlap",
                                      "🚨 Triggered" if s7 else "✅ Clear",
                                      delta=str(auth_label)[:36],
                                      delta_color="inverse" if s7 else "off",
                                      help="Does the same author name appear on multiple different domains?")

                        st.divider()
                        if signals_fired == 0:
                            st.success(f"✅ **{d}** passed all 7 checks. No suspicious patterns detected.")
                        elif v == "REVIEW":
                            st.info(f"💬 {dv.get('explanation', 'No explanation available')}")
                            st.markdown(
                                f'<div class="why-not-box">'
                                f'<b>Why not High Risk?</b> The 3-of-7 rule requires at least 3 independent signals '
                                f'to confirm a coordinated fake network. <b>{d}</b> triggered {signals_fired}/7 — '
                                f'enough to flag for investigation, but not enough to confirm synthetic coordination. '
                                f'A journalist should verify the flagged signal(s) before drawing conclusions.'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.info(f"💬 {dv.get('explanation', 'No explanation available')}")

        # ══════════════════════════════════════════════
        # SUB-TAB 5 — EVIDENCE
        # ══════════════════════════════════════════════
        with r_tab5:
            st.markdown("#### Evidence Dossier")
            st.markdown(
                '<div class="info-card">'
                '<div class="info-card-body">'
                'Everything the detector found that a journalist can follow up on. '
                'Three types of evidence: <b>content matches</b> (similar text), '
                '<b>shared infrastructure</b> (same server), and <b>shared author names</b>. '
                'Click any row to expand and read the actual excerpts side by side.'
                '</div></div>',
                unsafe_allow_html=True,
            )

            # ── Section 1: Content Similarity ─────────
            st.markdown("---")
            st.markdown("### 📄 Content Similarity Matches")
            st.caption(
                "These domain pairs have suspiciously similar homepage text (above 0.45 cosine similarity). "
                "Read both excerpts and decide: is this copied content, or just coincidental topic overlap?"
            )
            if not evidence_pairs:
                st.info("No content similarity matches found above the threshold for this domain.")
            else:
                for ep in evidence_pairs:
                    sim_pct   = ep['similarity']
                    ev_class  = "evidence-high" if sim_pct >= 0.70 else ("evidence-med" if sim_pct >= 0.50 else "evidence-low")
                    sim_label = f"{sim_pct:.0%}"
                    with st.expander(
                        f"{'🔴' if sim_pct >= 0.70 else '🟡' if sim_pct >= 0.50 else '🔵'}  "
                        f"**{ep['domain_a']}** ↔ **{ep['domain_b']}** — {sim_label} similar",
                        expanded=(sim_pct >= 0.70),
                    ):
                        st.markdown(
                            f'<div class="evidence-card {ev_class}">'
                            f'Similarity score: <b>{ep["similarity"]:.4f}</b> &nbsp;·&nbsp; '
                            f'Threshold: 0.45 &nbsp;·&nbsp; '
                            f'{"🔴 Very high — likely copied or templated content" if sim_pct >= 0.70 else "🟡 Moderately high — worth investigating"}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**{ep['domain_a']}**")
                            st.text_area(
                                label="excerpt_a", label_visibility="collapsed",
                                value=ep.get("excerpt_a") or "(no content captured)",
                                height=200, disabled=True,
                                key=f"ea_{ep['domain_a']}_{ep['domain_b']}",
                            )
                        with col_b:
                            st.markdown(f"**{ep['domain_b']}**")
                            st.text_area(
                                label="excerpt_b", label_visibility="collapsed",
                                value=ep.get("excerpt_b") or "(no content captured)",
                                height=200, disabled=True,
                                key=f"eb_{ep['domain_a']}_{ep['domain_b']}",
                            )

            # ── Section 2: Shared Hosting ──────────────
            st.markdown("---")
            st.markdown("### 🖥️ Shared Hosting Infrastructure")
            st.caption(
                "These domains share the same IP address or hosting provider. "
                "Unrelated news outlets rarely share servers — this is a strong indicator of common ownership. "
                "Check WHOIS and hosting records to confirm."
            )
            if not hosting_evidence:
                st.info("No shared hosting detected between domains in this cluster.")
            else:
                for he in hosting_evidence:
                    edge_label = "same IP address" if he["edge_type"] == "SAME_IP" else "same hosting provider (ASN)"
                    st.warning(
                        f"**{he['domain_a']}** and **{he['domain_b']}** share the **{edge_label}**. "
                        f"→ Investigate: who registered both domains? Are they at the same address?"
                    )

            # ── Section 3: Shared Authors ──────────────
            st.markdown("---")
            st.markdown("### ✍️ Shared Author Bylines")
            st.caption(
                "The same author name appeared on articles from multiple different domains. "
                "This is one of the strongest signals of a coordinated network — "
                "real independent outlets almost never share bylined staff."
            )
            if not author_evidence:
                st.info("No shared authors detected. (Requires article pages to have been successfully crawled.)")
            else:
                for ae in author_evidence:
                    st.warning(
                        f"Author **\"{ae['author']}\"** was found on both **{ae['domain_a']}** and **{ae['domain_b']}**. "
                        f"→ Investigate: is this person real? Do they have a social media presence? Are they listed on both sites?"
                    )

        # ── Export & Share ────────────────────────────
        st.divider()
        st.markdown("#### Export & Share")

        report = {
            "project":         "Dead Internet Detector",
            "analyzed_at":     result.get("analyzed_at", ""),
            "seed_domains":    result.get("seed_domains", [result.get("domain", "")]),
            "cluster_verdict": verdict,
            "confidence":      confidence,
            "summary":         summary,
            "domain_verdicts": domain_verdicts,
            "evidence_pairs":  evidence_pairs,
            "analysis_type":   analysis_type,
        }

        exp_col, share_col = st.columns(2)
        with exp_col:
            st.download_button(
                "📥 Download Full Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name="dead_internet_report.json",
                mime="application/json",
                use_container_width=True,
            )
        with share_col:
            if st.button("🔗 Generate Shareable Link", use_container_width=True):
                try:
                    save_resp = requests.post(
                        f"{BACKEND_URL}/report/save",
                        json={"report": report},
                        timeout=10,
                    )
                    if save_resp.status_code == 200:
                        st.session_state["report_id"] = save_resp.json().get("report_id", "")
                    else:
                        st.error(f"Could not save report: {save_resp.text}")
                except Exception as e:
                    st.error(f"Share failed: {e}")

        if st.session_state.get("report_id"):
            report_id  = st.session_state["report_id"]
            report_url = f"{PUBLIC_BACKEND_URL}/report/{report_id}"
            st.success("Report saved. Share this link:")
            st.code(report_url, language=None)
            st.caption("Anyone with this link can retrieve the full analysis JSON.")

        with st.expander("🔧 Raw API Response (for debugging)"):
            st.json(result)


# ══════════════════════════════════════════════════════
# TAB 2 — LIVE MONITOR DASHBOARD
# ══════════════════════════════════════════════════════
with tab_monitor:

    st.markdown("#### 📡 Live Monitor Dashboard")
    st.markdown(
        '<div class="info-card">'
        '<div class="info-card-body">'
        'The monitor automatically ingests newly registered domains (from WHOISDS) and domains '
        'shared on Reddit, then runs them through the same 4-agent detection pipeline. '
        'Use <b>Start Monitor Cycle</b> below to trigger a run manually.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Feed status ───────────────────────────────────
    st.markdown("### Feed Status")
    feed   = api_get("/feed-status")
    latest = feed.get("latest") if feed else None

    col_w, col_r, col_q, col_b = st.columns(4)

    if latest:
        whoisds_n   = latest.get("whoisds_count",    0)
        reddit_n    = latest.get("reddit_count",      0)
        queued_n    = latest.get("queued_unique",      0)
        batches_n   = latest.get("batches",            0)
        syn_batches = latest.get("synthetic_batches",  0)
        ran_at      = latest.get("ran_at", "never")

        with col_w:
            st.metric("WHOISDS Domains", whoisds_n, help="Newly registered domains from WHOISDS feed")
            st.caption("✅ Feed active" if whoisds_n > 0 else "⚠️ Feed not configured")
        with col_r:
            st.metric("Reddit Domains", reddit_n, help="Domains extracted from Reddit posts")
            st.caption("✅ Feed active" if reddit_n > 0 else "⚠️ No domains found")
        with col_q:
            st.metric("Queued for Analysis", queued_n, help="Unique domains sent to the detection pipeline")
        with col_b:
            st.metric("Synthetic Batches", f"{syn_batches}/{batches_n}", help="Batches where synthetic content was detected")

        try:
            from datetime import datetime
            dt        = datetime.fromisoformat(ran_at.replace("Z", "+00:00"))
            nice_time = dt.strftime("%b %d, %Y at %H:%M UTC")
        except Exception:
            nice_time = ran_at
        st.caption(f"Last monitor run: **{nice_time}**")
    else:
        with col_w: st.metric("WHOISDS Domains", "—")
        with col_r: st.metric("Reddit Domains",  "—")
        with col_q: st.metric("Queued",           "—")
        with col_b: st.metric("Synthetic",        "—")
        st.info("No monitor runs yet. Click **Start Monitor Cycle** below to begin.")

    st.divider()

    # ── Timeline chart ────────────────────────────────
    st.markdown("### Monitor Run Timeline")
    timeline_data = api_get("/timeline?limit=20")
    timeline      = timeline_data.get("timeline", []) if timeline_data else []

    if timeline and len(timeline) >= 1:
        df_tl = pd.DataFrame(timeline)
        if "ran_at" in df_tl.columns:
            df_tl["ran_at"] = pd.to_datetime(df_tl["ran_at"], utc=True, errors="coerce")
            df_tl = df_tl.dropna(subset=["ran_at"]).sort_values("ran_at")

            rename_map = {}
            if "queued_unique"      in df_tl.columns: rename_map["queued_unique"]      = "Queued"
            if "reddit_count"       in df_tl.columns: rename_map["reddit_count"]       = "Reddit"
            if "whoisds_count"      in df_tl.columns: rename_map["whoisds_count"]      = "WHOISDS"
            if "synthetic_batches"  in df_tl.columns: rename_map["synthetic_batches"]  = "Synthetic Batches"
            df_tl = df_tl.rename(columns=rename_map)
            y_cols = [c for c in ["Queued","Reddit","WHOISDS","Synthetic Batches"] if c in df_tl.columns]

            if not df_tl.empty and y_cols:
                fig_tl = px.line(
                    df_tl, x="ran_at", y=y_cols,
                    title="Domains Ingested & Analyzed Over Time",
                    markers=True,
                    color_discrete_sequence=["#3498db","#e74c3c","#e67e22","#9b59b6"],
                )
                fig_tl.update_layout(
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font_color="#e0e0e0", legend_title_text="",
                    xaxis_title="Time",
                    xaxis=dict(type="date", tickformat="%b %d %H:%M"),
                    yaxis_title="Count", height=350,
                )
                st.plotly_chart(fig_tl, use_container_width=True)
    else:
        st.info("No timeline data yet. Run a monitor cycle to start collecting data points.")

    st.divider()

    # ── Monitor controls ──────────────────────────────
    st.markdown("### Run Monitor Cycle")
    st.caption(
        "Each cycle fetches fresh domains from Reddit (and WHOISDS if configured), "
        "then analyzes them in batches of 10 through the full 4-agent detection pipeline."
    )

    if "monitor_job_id" not in st.session_state:
        st.session_state["monitor_job_id"] = None

    col_start, col_poll = st.columns(2)
    with col_start:
        if st.button("▶️ Start Monitor Cycle", use_container_width=True, type="primary"):
            try:
                r = requests.post(f"{BACKEND_URL}/monitor/start", timeout=20)
                if r.status_code == 200:
                    job_id = r.json().get("job_id", "")
                    st.session_state["monitor_job_id"] = job_id
                    st.success(f"Monitor started — Job ID: `{job_id[:12]}…`")
                else:
                    st.error(f"Failed to start: {r.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
    with col_poll:
        if st.button("🔄 Check Job Status", use_container_width=True):
            st.rerun()

    job_id = st.session_state.get("monitor_job_id")
    if job_id:
        job = api_get(f"/monitor/job/{job_id}")
        if job:
            status = job.get("status", "unknown")
            if status in ("queued", "running"):
                st.warning(f"⏳ Job `{job_id[:12]}…` is **{status}**. Click 'Check Job Status' to refresh.")
            elif status == "completed":
                st.success(f"✅ Job `{job_id[:12]}…` completed!")
                s = job.get("summary", {})
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1: st.metric("Reddit Domains",    s.get("reddit_count",      0))
                with mc2: st.metric("WHOISDS Domains",   s.get("whoisds_count",     0))
                with mc3: st.metric("Queued",             s.get("queued_unique",     0))
                with mc4: st.metric("Synthetic Batches",  s.get("synthetic_batches", 0))
            elif status == "failed":
                st.error(f"❌ Job failed: {job.get('error', 'unknown')}")
                tb = job.get("traceback")
                if tb:
                    with st.expander("Error details"):
                        st.code(tb)
        else:
            st.info("Job status unavailable — backend may have restarted.")

    st.divider()

    with st.expander("ℹ️ How does the monitor work?"):
        st.markdown("""
        **Data Sources:**
        - **Reddit** — Scans r/worldnews, r/conspiracy, r/politics, r/news for external domain links
        - **WHOISDS** — Fetches newly registered domains with suspicious TLDs (.xyz, .top, .click, etc.)

        **Pipeline:**
        1. Domains from both feeds are merged and deduplicated
        2. Up to 40 unique domains are queued per cycle
        3. Domains are split into batches of 10
        4. Each batch runs through the full 4-agent analysis pipeline
        5. Results are logged to `data/monitor_runs.jsonl`

        **Scheduling:**
        - Manual: Click "Start Monitor Cycle" above
        - Automatic: Run `python -m pipeline.schedule_monitor` (every 6 hours by default)
        """)

    with st.expander("ℹ️ About This Project"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **The Dead Internet Detector** detects coordinated synthetic content
            ecosystems — networks of fake websites designed to manipulate search
            rankings and public opinion.

            Unlike tools that check individual articles, this system asks the
            **network-level question**: is this entire corner of the internet
            artificially coordinated?
            """)
        with col2:
            st.markdown("""
            **Tech Stack:**
            - **FastAPI** — backend API
            - **Graph Neural Network (GCN)** — domain cluster classification
            - **Neo4j AuraDB** — graph database
            - **Sentence Transformers** — content similarity embeddings
            - **Isolation Forest** — cadence anomaly detection
            - **GPT-4o** — journalist briefing

            **Ethical Guidelines:**
            Research and journalism only. Human review required before publication.
            3-of-7 convergence required. No personal data collected.
            """)
