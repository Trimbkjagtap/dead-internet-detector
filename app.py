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

# Public-facing URL used in shareable report links (may differ from internal BACKEND_URL)
PUBLIC_BACKEND_URL = os.getenv("PUBLIC_BACKEND_URL", BACKEND_URL)
try:
    PUBLIC_BACKEND_URL = st.secrets.get("PUBLIC_BACKEND_URL", PUBLIC_BACKEND_URL)
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
    2. **Analyze** — compute 7 detection signals
    3. **Graph** — build domain network in Neo4j
    4. **Verdict** — GNN classifies clusters

    **3-of-7 rule:** flagged SYNTHETIC only when 3+ signals converge.
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

    check_clicked = st.button(
        "🔍 Check This Site",
        type="primary",
        use_container_width=True,
        disabled=(not domain_input.strip()),
    )

    if check_clicked and domain_input.strip():
        new_domain = domain_input.strip()
        # If different domain, clear previous state
        if new_domain != st.session_state.get("lookup_domain", ""):
            st.session_state.pop("result", None)
            st.session_state.pop("ai_analysis", None)
            st.session_state.pop("report_id", None)
            st.session_state["lookup_job_id"] = None
        st.session_state["lookup_domain"] = new_domain
        try:
            with st.spinner("Checking domain..."):
                t0 = time.time()
                resp = requests.post(
                    f"{BACKEND_URL}/lookup",
                    json={"domain": new_domain},
                    timeout=20,
                )
                elapsed = time.time() - t0

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
            st.info("⏳ **Deep analysis running...** Results will appear automatically.")
            time.sleep(5)
            st.rerun()
        elif job_status == "completed":
            full_result = job.get("full_result")
            if full_result:
                full_result["analysis_type"] = "fresh"
                full_result["analyzed_at"] = job.get("finished_at", "")
                st.session_state["result"] = full_result
            else:
                # Old job without full_result — re-fetch from cache which
                # now pulls evidence_pairs from the job store.
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
            pass  # The job-poll banner above already tells the user analysis is running
        else:
            st.info(f"📋 {summary}")
            if verdict == "REVIEW" and analysis_type != "preliminary":
                max_signals = max(
                    (dv.get("signals_triggered", 0) for dv in domain_verdicts.values()),
                    default=0
                )
                st.caption(
                    f"**Why Suspicious and not High Risk?** "
                    f"The detector requires 3 or more independent signals to confirm a coordinated fake network "
                    f"(the 3-of-7 convergence rule). The highest signal count here is {max_signals}/7. "
                    f"One or two signals can have innocent explanations — for example, two right-leaning outlets "
                    f"will naturally share similar vocabulary without being coordinated. "
                    f"Treat this as a lead worth investigating, not a confirmed verdict."
                )

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        _display_domain = result.get("domain") or (result.get("seed_domains", [""])[0] if result.get("seed_domains") else "—")
        if isinstance(_display_domain, str) and _display_domain.startswith("www."):
            _display_domain = _display_domain[4:]
        with c1: st.metric("Domain", _display_domain)
        with c2: st.metric("🔴 High Risk", result.get("synthetic_domains", 1 if verdict == "SYNTHETIC" else 0))
        with c3: st.metric("🟡 Suspicious", result.get("review_domains", 1 if verdict == "REVIEW" else 0))
        with c4: st.metric("🟢 Looks Legit", result.get("organic_domains", 1 if verdict == "ORGANIC" else 0))

        st.divider()

        evidence_pairs   = result.get("evidence_pairs", [])
        hosting_evidence = result.get("hosting_evidence", [])
        author_evidence  = result.get("author_evidence", [])
        total_evidence   = len(evidence_pairs) + len(hosting_evidence) + len(author_evidence)

        # Result sub-tabs
        r_tab1, r_tab2, r_tab3, r_tab4, r_tab5 = st.tabs([
            "🕸️ Network Graph",
            "📊 Signal Analysis",
            "🤖 GPT-4 Analysis",
            "📋 Domain Details",
            f"🔍 Evidence ({total_evidence})",
        ])

        # ── Network Graph ────────────────────────────
        with r_tab1:
            st.subheader("Domain Similarity Network")
            st.caption(
                "Shows the queried domain (large center node) and every domain "
                "that shares similar content with it. Hover a node to see details."
            )

            # Get the queried domain from session state
            graph_domain = st.session_state.get("lookup_domain", "")
            graph_data = api_get(f"/graph/neighborhood/{graph_domain}") if graph_domain else None

            cmap = {"SYNTHETIC": "#e74c3c", "REVIEW": "#e67e22", "ORGANIC": "#27ae60"}

            if graph_data and graph_data.get("nodes"):
                nodes = graph_data["nodes"]
                edges = graph_data["edges"]

                seed_nodes = [nd for nd in nodes if nd.get("is_seed")]
                neighbor_nodes = [nd for nd in nodes if not nd.get("is_seed")]

                # If no neighbors, just show a centered message — no chart needed
                if not neighbor_nodes:
                    seed_verdict = seed_nodes[0].get("verdict", "ORGANIC") if seed_nodes else "ORGANIC"
                    color = {"SYNTHETIC": "🔴", "REVIEW": "🟡"}.get(seed_verdict, "🟢")
                    st.markdown(f"### {color} {graph_domain}")
                    st.info(
                        f"**{graph_domain}** is in the database but has no similar domains connected to it yet. "
                        "This means no other stored domain exceeded the similarity threshold. "
                        "Check the **Evidence** tab — corpus comparisons may have found matches there."
                    )
                else:
                # Position: seed at center, neighbors in a circle around it
                    pos = {}
                    if seed_nodes:
                        pos[seed_nodes[0]["id"]] = (0.0, 0.0)

                    nb_count = len(neighbor_nodes)
                    for idx, nd in enumerate(neighbor_nodes):
                        angle = 2 * math.pi * idx / max(nb_count, 1)
                        pos[nd["id"]] = (math.cos(angle), math.sin(angle))

                    # Draw edges
                    ex, ey = [], []
                    for e in edges:
                        s, t = pos.get(e["source"]), pos.get(e["target"])
                        if s is not None and t is not None:
                            ex += [s[0], t[0], None]
                            ey += [s[1], t[1], None]

                    # Draw nodes
                    placed_nodes = [nd for nd in nodes if nd["id"] in pos]
                    node_x      = [pos[nd["id"]][0] for nd in placed_nodes]
                    node_y      = [pos[nd["id"]][1] for nd in placed_nodes]
                    node_colors = [cmap.get(nd.get("verdict", "ORGANIC"), "#888") for nd in placed_nodes]
                    node_sizes  = [22 if nd.get("is_seed") else 12 for nd in placed_nodes]
                    node_labels = [nd["domain"] for nd in placed_nodes]
                    hover_texts = [
                        f"<b>{nd['domain']}</b><br>"
                        f"Verdict: {nd.get('verdict','ORGANIC')}<br>"
                        f"Signals triggered: {nd.get('signals', 0)}"
                        for nd in placed_nodes
                    ]

                    fig = go.Figure(
                        data=[
                            go.Scatter(
                                x=ex, y=ey, mode="lines",
                                line=dict(width=1, color="#555"),
                                hoverinfo="none",
                            ),
                            go.Scatter(
                                x=node_x, y=node_y,
                                mode="markers+text",
                                text=node_labels,
                                textposition="top center",
                                textfont=dict(size=10, color="#cccccc"),
                                hovertemplate="%{customdata}<extra></extra>",
                                customdata=hover_texts,
                                marker=dict(
                                    size=node_sizes,
                                    color=node_colors,
                                    line=dict(width=1.5, color="#ffffff"),
                                ),
                            ),
                        ],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode="closest",
                            paper_bgcolor="#0e1117",
                            plot_bgcolor="#0e1117",
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 1.4]),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 1.4]),
                            height=520,
                            margin=dict(l=20, r=20, t=20, b=20),
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    col_l, col_m, col_r = st.columns(3)
                    col_l.markdown("🔴 **Synthetic**")
                    col_m.markdown("🟡 **Review**")
                    col_r.markdown("🟢 **Organic**")
                    st.caption(
                        f"**{graph_domain}** shares similar content with **{len(neighbor_nodes)}** other domain(s). "
                        "Lines connect domains above the similarity threshold. Hover nodes for details."
                    )
            elif graph_domain:
                st.info(
                    f"**{graph_domain}** has not been stored in the graph database yet. "
                    "Run an analysis first, then return to this tab."
                )
            else:
                st.info("Search for a domain first — the graph will show its similarity network here.")

        # ── Signal Analysis ──────────────────────────
        with r_tab2:
            st.subheader("Signal Analysis")
            st.caption("Each of the 7 signals is an independent check. Red = triggered. 3 or more triggers = high risk.")
            if domain_verdicts:
                rows = []
                for d, dv in domain_verdicts.items():
                    rows.append({
                        "Domain":      d[:35],
                        "Verdict":     dv.get("verdict", "ORGANIC"),
                        "Signals":     dv.get("signals_triggered", 0),
                        "1·Similarity": dv.get("signal_1_similarity", 0),
                        "2·Cadence":   dv.get("signal_2_cadence", 0),
                        "3·WHOIS":     dv.get("signal_3_whois", 0),
                        "4·Hosting":   dv.get("signal_4_hosting", 0),
                        "5·Links":     dv.get("signal_5_link_network", 0),
                        "6·Wayback":   dv.get("signal_6_wayback", 0),
                        "7·Authors":   dv.get("signal_7_authors", 0),
                    })
                df_sig = pd.DataFrame(rows)
                cmap = {"SYNTHETIC": "#e74c3c", "REVIEW": "#e67e22", "ORGANIC": "#27ae60"}

                heat_cols = ["1·Similarity","2·Cadence","3·WHOIS","4·Hosting","5·Links","6·Wayback","7·Authors"]
                heat = df_sig[["Domain"] + heat_cols].set_index("Domain")
                fig2 = px.imshow(
                    heat.T,
                    color_continuous_scale=["#1a2a1a", "#e74c3c"],
                    zmin=0, zmax=1,
                    title="Signal Heatmap — Red = Triggered, Green = Clear",
                    aspect="auto",
                    labels={"x": "Domain", "y": "Signal", "color": "Triggered"},
                )
                fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#e0e0e0")
                st.plotly_chart(fig2, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Max signals fired", f"{int(df_sig['Signals'].max())}/7")
                with c2: st.metric("High risk domains", len(df_sig[df_sig["Signals"] >= 3]))
                with c3: st.metric("Needs review", len(df_sig[(df_sig["Signals"] >= 1) & (df_sig["Signals"] < 3)]))

                st.caption("**What each signal means:**")
                st.markdown(
                    "| # | Signal | What it checks |\n"
                    "|---|--------|----------------|\n"
                    "| 1 | Similarity | Homepage text is unusually similar to other known domains |\n"
                    "| 2 | Cadence | Publishing time pattern looks anomalous |\n"
                    "| 3 | WHOIS | Domain is very new or has suspicious registration pattern |\n"
                    "| 4 | Hosting | Shares IP address or hosting provider with another domain |\n"
                    "| 5 | Links | Domains in the cluster mutually link to each other |\n"
                    "| 6 | Wayback | Site has almost no archive history (very new or recently faked) |\n"
                    "| 7 | Authors | Same author name appears on multiple different domains |"
                )

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
                st.markdown("### Assessment")
                st.info(st.session_state["ai_analysis"])
                st.caption("⚠️ AI analysis is advisory — always apply human judgment")

        # ── Domain Details ───────────────────────────
        with r_tab4:
            st.subheader("Per-Domain Breakdown")
            st.caption("Each domain is checked against 7 independent signals. 3 or more = High Risk.")
            if domain_verdicts:
                for d, dv in domain_verdicts.items():
                    s1 = int(dv.get("signal_1_similarity", 0))
                    s2 = int(dv.get("signal_2_cadence", 0))
                    s3 = int(dv.get("signal_3_whois", 0))
                    s4 = int(dv.get("signal_4_hosting", 0))
                    s5 = int(dv.get("signal_5_link_network", 0))
                    s6 = int(dv.get("signal_6_wayback", 0))
                    s7 = int(dv.get("signal_7_authors", 0))
                    signals_fired = s1 + s2 + s3 + s4 + s5 + s6 + s7

                    if signals_fired >= 3:
                        v = "SYNTHETIC"
                    elif signals_fired >= 1:
                        v = "REVIEW"
                    else:
                        v = "ORGANIC"

                    max_sim = float(dv.get("max_similarity", 0) or 0)
                    _burst  = float(dv.get("burst_score", 0) or 0)
                    if signals_fired >= 3:
                        recomputed_conf = round(max(0.65, min(0.97, 0.65
                            + min((signals_fired - 3) / 4.0, 1.0) * 0.20
                            + min(max_sim * 0.20, 0.10)
                            + min(_burst * 0.15, 0.06))), 2)
                    elif signals_fired >= 1:
                        recomputed_conf = round(max(0.02, min(0.64, 0.40
                            + (signals_fired - 1) * 0.10
                            + min(max_sim * 0.20, 0.10)
                            + min(_burst * 0.15, 0.06))), 2)
                    else:
                        recomputed_conf = round(max(0.70, min(1.0 - min(max_sim * 0.20, 0.20), 0.97)), 2)

                    icon = "🔴" if v == "SYNTHETIC" else "🟡" if v == "REVIEW" else "🟢"

                    with st.expander(
                        f"{icon} {d} — {v} — {signals_fired}/7 signals triggered",
                        expanded=(v != "ORGANIC"),
                    ):
                        # Row 1: original 3 signals
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            sim_val = float(dv.get("max_similarity") or 0)
                            st.metric("1 · Content Similarity",
                                      "🚨 Triggered" if s1 else "✅ Clear",
                                      delta=f"score {sim_val:.3f} (threshold 0.45)" if sim_val else "no match found",
                                      delta_color="inverse" if s1 else "off",
                                      help="Checks if this site's text is unusually similar to other known domains.")
                        with c2:
                            burst = float(dv.get("burst_score") or 0)
                            st.metric("2 · Publishing Cadence",
                                      "🚨 Triggered" if s2 else "✅ Clear",
                                      delta=f"anomaly score {burst:.3f}" if burst else None,
                                      delta_color="inverse" if s2 else "off",
                                      help="Detects unusual publishing time patterns compared to other domains.")
                        with c3:
                            age = int(dv.get("domain_age_days") or -1)
                            age_str = f"{age} days old" if age > 0 else "age unknown"
                            st.metric("3 · WHOIS Registration",
                                      "🚨 Triggered" if s3 else "✅ Clear",
                                      delta=age_str,
                                      delta_color="inverse" if s3 else "off",
                                      help="Flags very new domains or suspicious registration patterns.")

                        # Row 2: new 4 signals
                        c4, c5, c6, c7 = st.columns(4)
                        with c4:
                            org = dv.get("hosting_org", "") or ""
                            ip  = dv.get("ip_address", "") or ""
                            st.metric("4 · Shared Hosting",
                                      "🚨 Triggered" if s4 else "✅ Clear",
                                      delta=org[:30] if org else (ip or "no data"),
                                      delta_color="inverse" if s4 else "off",
                                      help="Flags domains that share an IP address or hosting provider.")
                        with c5:
                            insular = float(dv.get("insular_score") or 0)
                            st.metric("5 · Link Network",
                                      "🚨 Triggered" if s5 else "✅ Clear",
                                      delta=f"{insular:.0%} links within cluster" if insular else None,
                                      delta_color="inverse" if s5 else "off",
                                      help="Flags domains that mutually link to each other (insular cluster).")
                        with c6:
                            snaps = int(dv.get("wayback_snapshot_count") or 0)
                            wb_reason = dv.get("wayback_flag_reason", "") or ""
                            wb_label = "new site" if wb_reason == "new_site" else ("archive spike" if wb_reason else f"{snaps} snapshots")
                            st.metric("6 · Archive History",
                                      "🚨 Triggered" if s6 else "✅ Clear",
                                      delta=wb_label,
                                      delta_color="inverse" if s6 else "off",
                                      help="Checks Wayback Machine — very new sites or sudden archive spikes are suspicious.")
                        with c7:
                            try:
                                import json as _json
                                shared = _json.loads(dv.get("shared_authors", "[]") or "[]")
                            except Exception:
                                shared = []
                            auth_label = shared[0] if shared else ("shared author found" if s7 else "no overlap")
                            st.metric("7 · Author Overlap",
                                      "🚨 Triggered" if s7 else "✅ Clear",
                                      delta=str(auth_label)[:40],
                                      delta_color="inverse" if s7 else "off",
                                      help="Flags when the same author name appears on multiple different domains.")

                        st.divider()
                        if signals_fired == 0:
                            st.success(f"**{d}** passed all 7 checks. No suspicious patterns detected.")
                        elif v == "REVIEW":
                            st.info(f"💬 {dv.get('explanation', 'No explanation available')}")
                            st.caption(
                                f"**Why not High Risk?** The 3-of-7 rule requires at least 3 independent signals "
                                f"to confirm a coordinated fake network. {d} triggered {signals_fired}/7 — "
                                f"enough to flag for investigation, but not enough to confirm synthetic coordination on its own. "
                                f"A journalist should verify the flagged signal(s) before drawing conclusions."
                            )
                        else:
                            st.info(f"💬 {dv.get('explanation', 'No explanation available')}")

        # ── Evidence ─────────────────────────────────
        with r_tab5:
            st.subheader("Evidence")
            st.caption(
                "Everything the detector found that a journalist can follow up on. "
                "Three types: content matches, shared infrastructure, and shared authors."
            )

            # ── Section 1: Content Similarity ───────────
            st.markdown("### 📄 Content Similarity")
            st.caption(
                "These domain pairs have suspiciously similar homepage text. "
                "Read both excerpts and judge whether the similarity looks like "
                "copied content or just coincidental topic overlap."
            )
            if not evidence_pairs:
                st.info("No content similarity matches found above the threshold.")
            else:
                for ep in evidence_pairs:
                    sim_pct = f"{ep['similarity']:.0%}"
                    with st.expander(
                        f"**{ep['domain_a']}** ↔ **{ep['domain_b']}** — {sim_pct} similar",
                        expanded=(ep['similarity'] >= 0.7),
                    ):
                        st.markdown(f"**Text similarity score: `{ep['similarity']}`** (threshold: 0.45 — higher = more similar)")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**{ep['domain_a']}**")
                            st.text_area(
                                label="excerpt_a",
                                value=ep.get("excerpt_a") or "(no content captured)",
                                height=200, disabled=True,
                                label_visibility="collapsed",
                                key=f"ea_{ep['domain_a']}_{ep['domain_b']}",
                            )
                        with col_b:
                            st.markdown(f"**{ep['domain_b']}**")
                            st.text_area(
                                label="excerpt_b",
                                value=ep.get("excerpt_b") or "(no content captured)",
                                height=200, disabled=True,
                                label_visibility="collapsed",
                                key=f"eb_{ep['domain_a']}_{ep['domain_b']}",
                            )

            st.divider()

            # ── Section 2: Shared Hosting Infrastructure ─
            st.markdown("### 🖥️ Shared Hosting Infrastructure")
            st.caption(
                "These domains share the same IP address or hosting provider. "
                "Unrelated news outlets rarely share servers — this is a strong "
                "indicator of common ownership. Check WHOIS and hosting records to confirm."
            )
            if not hosting_evidence:
                st.info("No shared hosting detected.")
            else:
                for he in hosting_evidence:
                    edge_label = "same IP address" if he["edge_type"] == "SAME_IP" else "same hosting provider (ASN)"
                    st.warning(
                        f"**{he['domain_a']}** and **{he['domain_b']}** share the **{edge_label}**. "
                        f"→ Check: who registered both domains? Are they at the same address?"
                    )

            st.divider()

            # ── Section 3: Shared Author Names ──────────
            st.markdown("### ✍️ Shared Author Bylines")
            st.caption(
                "The same author name appeared on articles from multiple different domains. "
                "This is one of the strongest signals of a coordinated network — "
                "real independent outlets rarely share staff."
            )
            if not author_evidence:
                st.info("No shared authors detected. (Requires article pages to have been crawled successfully.)")
            else:
                for ae in author_evidence:
                    st.warning(
                        f"Author **\"{ae['author']}\"** was found on both **{ae['domain_a']}** and **{ae['domain_b']}**. "
                        f"→ Check: is this person real? Do they have a social media presence? Are they listed on both sites?"
                    )

        st.divider()

        # ── Export & Share ────────────────────────────
        report = {
            "project": "Dead Internet Detector",
            "analyzed_at": result.get("analyzed_at", ""),
            "seed_domains": result.get("seed_domains", [result.get("domain", "")]),
            "cluster_verdict": verdict,
            "confidence": confidence,
            "summary": summary,
            "domain_verdicts": domain_verdicts,
            "evidence_pairs": evidence_pairs,
            "analysis_type": analysis_type,
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
                        report_id = save_resp.json().get("report_id", "")
                        st.session_state["report_id"] = report_id
                    else:
                        st.error(f"Could not save report: {save_resp.text}")
                except Exception as e:
                    st.error(f"Share failed: {e}")

        if st.session_state.get("report_id"):
            report_id = st.session_state["report_id"]
            report_url = f"{PUBLIC_BACKEND_URL}/report/{report_id}"
            st.success("Report saved. Share this link:")
            st.code(report_url, language=None)
            st.caption(
                "Paste this URL into an email, document, or browser. "
                "Anyone with the link can retrieve the full analysis JSON."
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