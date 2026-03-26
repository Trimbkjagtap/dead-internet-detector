# app.py
# Streamlit frontend for the Dead Internet Detector v2
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


# ── Custom CSS ───────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .verdict-synthetic {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white; padding: 20px 30px; border-radius: 12px;
        text-align: center; font-size: 28px; font-weight: 700;
        margin: 20px 0;
    }
    .verdict-organic {
        background: linear-gradient(135deg, #1a7a2a, #27ae60);
        color: white; padding: 20px 30px; border-radius: 12px;
        text-align: center; font-size: 28px; font-weight: 700;
        margin: 20px 0;
    }
    .verdict-review {
        background: linear-gradient(135deg, #c07a00, #e67e22);
        color: white; padding: 20px 30px; border-radius: 12px;
        text-align: center; font-size: 28px; font-weight: 700;
        margin: 20px 0;
    }
    .monitor-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 16px; margin: 8px 0;
    }
    .feed-ok { border-left: 4px solid #27ae60; }
    .feed-warn { border-left: 4px solid #e67e22; }
    .feed-off { border-left: 4px solid #e74c3c; }
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


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🕸️")
    st.title("Dead Internet Detector")
    st.caption("INFO 7390 | Northeastern University")

    st.divider()

    st.subheader("📖 How It Works")
    st.markdown("""
    1. **Crawl** — fetch content from seed domains + linked sites
    2. **Analyze** — compute 3 detection signals
    3. **Graph** — build domain network in Neo4j
    4. **Verdict** — GNN classifies clusters

    **2-of-3 rule:** flagged SYNTHETIC only when 2+ signals converge.
    """)

    st.divider()

    st.subheader("🔬 Detection Signals")
    st.markdown("""
    | Signal | Method |
    |--------|--------|
    | Content Similarity | Sentence Transformers |
    | Cadence Anomaly | Isolation Forest |
    | WHOIS Registration | Domain age + heuristics |
    """)

    st.divider()

    st.subheader("🔌 Backend Status")
    health = api_get("/health")
    if health and health.get("status") == "ok":
        st.success("✅ Backend connected")
    else:
        st.error("❌ Backend offline")
        st.caption(f"Expected at: {BACKEND_URL}")


# ══════════════════════════════════════════════════════
# MAIN PAGE — Two top-level tabs
# ══════════════════════════════════════════════════════

st.title("🕸️ Dead Internet Detector")
st.caption("Detect coordinated synthetic content ecosystems using graph analysis and AI")

tab_analyze, tab_monitor = st.tabs(["🔍 Check a Domain", "📡 Live Monitor Dashboard"])


# ══════════════════════════════════════════════════════
# TAB 1 — ANALYZE DOMAINS
# ══════════════════════════════════════════════════════
with tab_analyze:
    st.subheader("Paste a suspicious domain")
    st.caption("Type one domain and get a fast trust check. Example: patriot-updates-now.com")

    recent = api_get("/recently-detected?limit=5")
    recent_items = recent.get("items", []) if recent else []

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        st.metric("Recently Flagged", len(recent_items))
    with rc2:
        if recent_items:
            top = ", ".join([item.get("domain", "") for item in recent_items[:3]])
            st.caption(f"Latest flagged domains: {top}")
        else:
            st.caption("No recently flagged domains yet")

    with st.expander("See recently detected domains"):
        if recent_items:
            for item in recent_items:
                icon = "🔴" if item.get("verdict") == "SYNTHETIC" else "🟡"
                st.markdown(
                    f"{icon} **{item.get('domain', '')}** - {item.get('headline', 'Suspicious activity')}  \n"
                    f"Reason: {item.get('reason', 'Network indicators detected')}"
                )
        else:
            st.info("Run monitor cycles to populate this feed.")

    domain_input = st.text_input(
        "Paste a domain to check",
        value=st.session_state.get("lookup_domain", ""),
        placeholder="suspicious-news.com",
    )

    c_action, c_refresh = st.columns([2, 1])
    with c_action:
        check_clicked = st.button(
            "Check This Site",
            type="primary",
            use_container_width=True,
            disabled=(not domain_input.strip()),
        )
    with c_refresh:
        refresh_job = st.button("Refresh Full Analysis", use_container_width=True)

    if check_clicked and domain_input.strip():
        st.session_state["lookup_domain"] = domain_input.strip()
        try:
            with st.spinner("Running instant lookup..."):
                t0 = time.time()
                resp = requests.post(
                    f"{BACKEND_URL}/lookup",
                    json={"domain": domain_input.strip()},
                    timeout=20,
                )
                elapsed = time.time() - t0

            if resp.status_code != 200:
                st.error(f"Lookup failed: {resp.text}")
            else:
                lookup = resp.json()
                st.session_state["result"] = lookup
                st.success(f"Lookup complete in {elapsed:.1f} seconds")

                if lookup.get("status") == "queued" and lookup.get("job_id"):
                    st.session_state["lookup_job_id"] = lookup.get("job_id")
                    st.info("Full analysis is running in the background. Refresh to check status.")
                else:
                    st.session_state["lookup_job_id"] = None
        except requests.exceptions.Timeout:
            st.error("Lookup timed out. Try again in a few seconds.")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to backend at {BACKEND_URL}")
        except Exception as e:
            st.error(f"Error: {e}")

    if refresh_job and st.session_state.get("lookup_job_id"):
        job_id = st.session_state.get("lookup_job_id")
        job = api_get(f"/lookup/job/{job_id}", timeout=10)
        if not job:
            st.warning("Job status unavailable. Backend may have restarted.")
        else:
            status = job.get("status")
            if status in ("queued", "running"):
                st.info(f"Full analysis is still {status}.")
            elif status == "completed":
                domain = st.session_state.get("lookup_domain", "")
                if domain:
                    latest_resp = requests.post(
                        f"{BACKEND_URL}/lookup",
                        json={"domain": domain},
                        timeout=20,
                    )
                    if latest_resp.status_code == 200:
                        st.session_state["result"] = latest_resp.json()
                        st.success("Full analysis is complete. Showing updated cached result.")
                st.session_state["lookup_job_id"] = None
            elif status == "failed":
                st.error(f"Background analysis failed: {job.get('error', 'unknown error')}")
                st.session_state["lookup_job_id"] = None

    # ── Show results ─────────────────────────────────
    if "result" in st.session_state:
        result = st.session_state["result"]
        verdict = result.get("cluster_verdict", "UNKNOWN")
        confidence = result.get("max_confidence", 0.0)
        summary = result.get("summary", "")
        headline = result.get("headline", "Verdict")
        analysis_type = result.get("analysis_type", "")
        domain_verdicts = result.get("domain_verdicts", {})

        st.divider()

        # Verdict badge
        st.subheader("Trust Verdict")
        if verdict == "SYNTHETIC":
            st.markdown(
                f'<div class="verdict-synthetic">🔴 {headline}'
                f'<br/><span style="font-size:16px">Confidence: {confidence:.0%}</span></div>',
                unsafe_allow_html=True,
            )
        elif verdict == "REVIEW":
            st.markdown(
                f'<div class="verdict-review">🟡 {headline}'
                f'<br/><span style="font-size:16px">Confidence: {confidence:.0%}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="verdict-organic">🟢 {headline}'
                f'<br/><span style="font-size:16px">Confidence: {confidence:.0%}</span></div>',
                unsafe_allow_html=True,
            )

        if analysis_type == "preliminary":
            st.warning("Preliminary result shown. Full network analysis is running in the background.")
        st.info(f"{summary}")

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Domain", result.get("domain") or (result.get("seed_domains", [""])[0] if result.get("seed_domains") else "—"))
        with c2: st.metric("🔴 High Risk", result.get("synthetic_domains", 1 if verdict == "SYNTHETIC" else 0))
        with c3: st.metric("🟡 Suspicious", result.get("review_domains", 1 if verdict == "REVIEW" else 0))
        with c4: st.metric("🟢 Looks Legit", result.get("organic_domains", 1 if verdict == "ORGANIC" else 0))

        st.divider()

        # Result sub-tabs
        r_tab1, r_tab2, r_tab3, r_tab4 = st.tabs([
            "🕸️ Network Graph",
            "📊 Signal Analysis",
            "🤖 GPT-4 Analysis",
            "📋 Domain Details",
        ])

        # ── Network Graph ────────────────────────────
        with r_tab1:
            st.subheader("Domain Network Graph")
            graph_data = api_get("/graph")
            if graph_data:
                nodes = graph_data.get("nodes", [])
                edges = graph_data.get("edges", [])
                if nodes:
                    n = len(nodes)
                    node_x = [math.cos(2 * math.pi * i / max(n, 1)) for i in range(n)]
                    node_y = [math.sin(2 * math.pi * i / max(n, 1)) for i in range(n)]
                    pos = {nodes[i]["id"]: (node_x[i], node_y[i]) for i in range(n)}

                    ex, ey = [], []
                    for e in edges[:100]:
                        s, t = pos.get(e["source"]), pos.get(e["target"])
                        if s and t:
                            ex += [s[0], t[0], None]
                            ey += [s[1], t[1], None]

                    cmap = {"SYNTHETIC": "#e74c3c", "REVIEW": "#e67e22", "ORGANIC": "#27ae60"}
                    fig = go.Figure(
                        data=[
                            go.Scatter(x=ex, y=ey, mode="lines",
                                       line=dict(width=0.5, color="#444"),
                                       hoverinfo="none"),
                            go.Scatter(
                                x=node_x, y=node_y, mode="markers",
                                hovertemplate="<b>%{customdata[0]}</b><br>Verdict: %{customdata[1]}<br>Signals: %{customdata[2]}<extra></extra>",
                                customdata=[[nd["domain"], nd.get("verdict", "ORGANIC"), nd.get("signals", 0)] for nd in nodes],
                                marker=dict(size=10, color=[cmap.get(nd.get("verdict", "ORGANIC"), "#888") for nd in nodes],
                                            line=dict(width=1, color="#fff")),
                            ),
                        ],
                        layout=go.Layout(
                            showlegend=False, hovermode="closest",
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=500, margin=dict(l=20, r=20, t=20, b=20),
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("🔴 Synthetic  🟠 Review  🟢 Organic — hover for details")
                else:
                    st.info("Graph is empty — run an analysis first")
            else:
                st.warning("Graph visualization unavailable")

        # ── Signal Analysis ──────────────────────────
        with r_tab2:
            st.subheader("Signal Analysis")
            if domain_verdicts:
                rows = []
                for d, dv in domain_verdicts.items():
                    rows.append({
                        "Domain": d[:35],
                        "Synthetic Prob": round(dv.get("confidence", 0), 4),
                        "Signals": dv.get("signals_triggered", 0),
                        "Verdict": dv.get("verdict", "ORGANIC"),
                        "Similarity": dv.get("signal_1_similarity", 0),
                        "Cadence": dv.get("signal_2_cadence", 0),
                        "WHOIS": dv.get("signal_3_whois", 0),
                    })
                df_sig = pd.DataFrame(rows)

                cmap = {"SYNTHETIC": "#e74c3c", "REVIEW": "#e67e22", "ORGANIC": "#27ae60"}

                fig1 = px.bar(df_sig, x="Domain", y="Synthetic Prob", color="Verdict",
                              color_discrete_map=cmap, title="Synthetic Probability by Domain")
                fig1.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                   font_color="#e0e0e0", xaxis_tickangle=-45)
                st.plotly_chart(fig1, use_container_width=True)

                heat = df_sig[["Domain", "Similarity", "Cadence", "WHOIS"]].set_index("Domain")
                fig2 = px.imshow(heat.T, color_continuous_scale=["#1a4a1a", "#e74c3c"],
                                 title="Signal Heatmap — Red = Triggered", aspect="auto")
                fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                   font_color="#e0e0e0")
                st.plotly_chart(fig2, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Avg Synthetic Prob", f"{df_sig['Synthetic Prob'].mean():.1%}")
                with c2: st.metric("Max Signals", f"{df_sig['Signals'].max()}/3")
                with c3: st.metric("Above 50%", len(df_sig[df_sig["Synthetic Prob"] > 0.5]))
                st.dataframe(df_sig, use_container_width=True)

        # ── GPT-4 Analysis ───────────────────────────
        with r_tab3:
            st.subheader("🤖 GPT-4o AI Analysis")
            if st.button("Generate AI Analysis", type="primary"):
                with st.spinner("GPT-4o analyzing..."):
                    try:
                        from openai import OpenAI
                        from dotenv import load_dotenv
                        load_dotenv()
                        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                        domain_summary = "\n".join([
                            f"- {d}: {v.get('verdict')} "
                            f"(confidence={v.get('confidence',0):.0%}, "
                            f"signals={v.get('signals_triggered',0)}/3)"
                            for d, v in domain_verdicts.items()
                        ])
                        prompt = f"""Analyze this domain cluster assessment:

Domains:
{domain_summary}

Overall: {verdict} at {confidence:.0%} confidence.

Provide: (1) what signal patterns suggest, (2) whether this looks organic or coordinated and why, (3) key indicators, (4) what to investigate next. Be concise."""

                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=350, temperature=0.3,
                        )
                        st.session_state["ai_analysis"] = resp.choices[0].message.content
                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")

            if "ai_analysis" in st.session_state:
                st.markdown("### Assessment")
                st.info(st.session_state["ai_analysis"])
                st.caption("⚠️ AI analysis is advisory — always apply human judgment")

        # ── Domain Details ───────────────────────────
        with r_tab4:
            st.subheader("Per-Domain Breakdown")
            if domain_verdicts:
                for d, dv in domain_verdicts.items():
                    v = dv.get("verdict", "ORGANIC")
                    icon = "🔴" if v == "SYNTHETIC" else "🟡" if v == "REVIEW" else "🟢"
                    with st.expander(f"{icon} {d} — {v} ({dv.get('confidence',0):.0%})"):
                        c1, c2, c3 = st.columns(3)
                        with c1: st.metric("Similarity", "🚨 Triggered" if dv.get("signal_1_similarity") else "✅ Clear")
                        with c2: st.metric("Cadence", "🚨 Triggered" if dv.get("signal_2_cadence") else "✅ Clear")
                        with c3: st.metric("WHOIS", "🚨 Triggered" if dv.get("signal_3_whois") else "✅ Clear")
                        st.info(f"💬 {dv.get('explanation', 'No explanation available')}")

        st.divider()

        # Download report
        report = {
            "project": "Dead Internet Detector",
            "analyzed_at": result.get("analyzed_at", ""),
            "seed_domains": result.get("seed_domains", [result.get("domain", "")]),
            "cluster_verdict": verdict,
            "confidence": confidence,
            "summary": summary,
            "domain_verdicts": domain_verdicts,
            "analysis_type": analysis_type,
        }
        st.download_button(
            "📥 Download Full Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="dead_internet_report.json",
            mime="application/json",
            use_container_width=True,
        )

        with st.expander("🔧 Raw API Response"):
            st.json(result)


# ══════════════════════════════════════════════════════
# TAB 2 — LIVE MONITOR DASHBOARD
# ══════════════════════════════════════════════════════
with tab_monitor:

    st.subheader("📡 Live Monitor Dashboard")
    st.caption(
        "The monitor ingests newly registered domains (WHOISDS) and "
        "domains shared on Reddit, then runs them through the same "
        "detection pipeline automatically."
    )

    st.divider()

    # ── Feed status cards ────────────────────────────
    st.markdown("### Feed Status")
    feed = api_get("/feed-status")
    latest = feed.get("latest") if feed else None

    col_w, col_r, col_q, col_b = st.columns(4)

    if latest:
        whoisds_n = latest.get("whoisds_count", 0)
        reddit_n = latest.get("reddit_count", 0)
        queued_n = latest.get("queued_unique", 0)
        batches_n = latest.get("batches", 0)
        syn_batches = latest.get("synthetic_batches", 0)
        ran_at = latest.get("ran_at", "never")

        with col_w:
            st.metric("WHOISDS Domains", whoisds_n,
                       help="Newly registered domains from WHOISDS feed")
            if whoisds_n == 0:
                st.caption("⚠️ Feed not configured")
            else:
                st.caption("✅ Feed active")

        with col_r:
            st.metric("Reddit Domains", reddit_n,
                       help="Domains extracted from Reddit posts")
            st.caption("✅ Feed active" if reddit_n > 0 else "⚠️ No domains found")

        with col_q:
            st.metric("Queued for Analysis", queued_n,
                       help="Unique domains sent to the detection pipeline")

        with col_b:
            st.metric("Synthetic Batches", f"{syn_batches}/{batches_n}",
                       help="Batches where synthetic content was detected")

        # Last run timestamp
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(ran_at.replace("Z", "+00:00"))
            nice_time = dt.strftime("%b %d, %Y at %H:%M UTC")
        except Exception:
            nice_time = ran_at
        st.caption(f"Last monitor run: **{nice_time}**")

    else:
        with col_w: st.metric("WHOISDS Domains", "—")
        with col_r: st.metric("Reddit Domains", "—")
        with col_q: st.metric("Queued", "—")
        with col_b: st.metric("Synthetic", "—")
        st.info("No monitor runs yet. Click **Run Monitor Cycle** below to start.")

    st.divider()

    # ── Timeline chart ───────────────────────────────
    st.markdown("### Monitor Run Timeline")
    timeline_data = api_get("/timeline?limit=20")
    timeline = timeline_data.get("timeline", []) if timeline_data else []

    if timeline and len(timeline) >= 1:
        df_tl = pd.DataFrame(timeline)
        if "ran_at" in df_tl.columns:
            # Parse timestamps — handle both timezone-aware and naive formats
            df_tl["ran_at"] = pd.to_datetime(df_tl["ran_at"], utc=True, errors="coerce")
            df_tl = df_tl.dropna(subset=["ran_at"])
            df_tl = df_tl.sort_values("ran_at")

            # Rename columns for readable legend
            rename_map = {}
            if "queued_unique" in df_tl.columns:
                rename_map["queued_unique"] = "Queued"
            if "reddit_count" in df_tl.columns:
                rename_map["reddit_count"] = "Reddit"
            if "whoisds_count" in df_tl.columns:
                rename_map["whoisds_count"] = "WHOISDS"
            if "synthetic_batches" in df_tl.columns:
                rename_map["synthetic_batches"] = "Synthetic Batches"
            df_tl = df_tl.rename(columns=rename_map)

            y_cols = [c for c in ["Queued", "Reddit", "WHOISDS", "Synthetic Batches"] if c in df_tl.columns]

            if not df_tl.empty and y_cols:
                fig_tl = px.line(
                    df_tl, x="ran_at", y=y_cols,
                    title="Domains Ingested & Analyzed Over Time",
                    markers=True,
                    color_discrete_sequence=["#3498db", "#e74c3c", "#e67e22", "#9b59b6"],
                )
                fig_tl.update_layout(
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font_color="#e0e0e0",
                    legend_title_text="",
                    xaxis_title="Time",
                    xaxis=dict(type="date", tickformat="%b %d %H:%M"),
                    yaxis_title="Count",
                    height=350,
                )
                st.plotly_chart(fig_tl, use_container_width=True)
            else:
                st.info("Timeline data could not be parsed. Run more monitor cycles to populate.")
    else:
        st.info("No timeline data yet. Run a monitor cycle to start collecting data points.")

    st.divider()

    # ── Monitor controls ─────────────────────────────
    st.markdown("### Run Monitor")
    st.caption(
        "Each cycle fetches fresh domains from Reddit (and WHOISDS if configured), "
        "then analyzes them in batches of 10 through the full detection pipeline."
    )

    col_start, col_poll = st.columns(2)

    # Initialize session state
    if "monitor_job_id" not in st.session_state:
        st.session_state["monitor_job_id"] = None
    if "monitor_status" not in st.session_state:
        st.session_state["monitor_status"] = None

    with col_start:
        if st.button("▶️ Start Monitor Cycle", use_container_width=True, type="primary"):
            try:
                r = requests.post(f"{BACKEND_URL}/monitor/start", timeout=20)
                if r.status_code == 200:
                    job_id = r.json().get("job_id", "")
                    st.session_state["monitor_job_id"] = job_id
                    st.session_state["monitor_status"] = "running"
                    st.success(f"Monitor started — Job ID: `{job_id[:12]}...`")
                else:
                    st.error(f"Failed to start: {r.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

    with col_poll:
        if st.button("🔄 Check Job Status", use_container_width=True):
            st.rerun()

    # Show job result
    job_id = st.session_state.get("monitor_job_id")
    if job_id:
        job = api_get(f"/monitor/job/{job_id}")
        if job:
            status = job.get("status", "unknown")

            if status in ("queued", "running"):
                st.warning(f"⏳ Job `{job_id[:12]}...` is **{status}**. Click 'Check Job Status' to refresh.")

            elif status == "completed":
                st.session_state["monitor_status"] = "completed"
                st.success(f"✅ Job `{job_id[:12]}...` completed!")

                s = job.get("summary", {})
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1: st.metric("Reddit Domains", s.get("reddit_count", 0))
                with mc2: st.metric("WHOISDS Domains", s.get("whoisds_count", 0))
                with mc3: st.metric("Queued", s.get("queued_unique", 0))
                with mc4: st.metric("Synthetic Batches", s.get("synthetic_batches", 0))

            elif status == "failed":
                st.session_state["monitor_status"] = "failed"
                st.error(f"❌ Job failed: {job.get('error', 'unknown')}")
                tb = job.get("traceback")
                if tb:
                    with st.expander("Error details"):
                        st.code(tb)
        else:
            st.info("Job status unavailable — backend may have restarted.")

    st.divider()

    # ── How monitoring works ─────────────────────────
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

    # ── About section ────────────────────────────────
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
            - **CrewAI** — 4 agents in sequence
            - **GPT-4o** — agent reasoning
            - **Graph Neural Network** — domain classification
            - **Neo4j AuraDB** — graph database
            - **Sentence Transformers** — content embeddings
            - **Isolation Forest** — cadence anomaly detection

            **Ethical Guidelines:**
            Research and journalism only. Human review required.
            2-of-3 signals required. No personal data collected.
            """)