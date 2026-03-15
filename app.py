# app.py
# Streamlit frontend for the Dead Internet Detector
# Run with: streamlit run app.py

import streamlit as st
import requests
import plotly.graph_objects as go
import json
import time
import os

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="Dead Internet Detector",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Backend URL ──────────────────────────────────────
# Locally: http://localhost:8000
# Deployed: your Render URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Try Streamlit secrets if available
try:
    BACKEND_URL = st.secrets.get("BACKEND_URL", BACKEND_URL)
except:
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
    .domain-card {
        background: #1a1a2e; border-radius: 8px;
        padding: 12px 16px; margin: 8px 0;
        border-left: 4px solid #555;
    }
    .domain-synthetic { border-left-color: #e74c3c !important; }
    .domain-organic   { border-left-color: #27ae60 !important; }
    .domain-review    { border-left-color: #e67e22 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/spider-web.png", width=80)
    st.title("Dead Internet Detector")
    st.caption("INFO 7390 | Northeastern University")

    st.divider()

    st.subheader("⚙️ Settings")
    max_domains = st.slider(
        "Max domains to analyze", 1, 20, 5,
        help="More domains = more thorough but slower"
    )

    st.divider()

    st.subheader("📖 How It Works")
    st.markdown("""
    1. **Crawl** — fetch content from seed domains + linked sites
    2. **Analyze** — compute 3 signals:
       - 🔵 Content similarity
       - 🟣 Publishing cadence
       - 🟢 Domain registration
    3. **Graph** — build domain relationship network in Neo4j
    4. **Verdict** — GNN classifies cluster as SYNTHETIC or ORGANIC

    **2-of-3 rule:** A domain is flagged SYNTHETIC only if at least 2 signals converge.
    """)

    st.divider()

    st.subheader("🔌 Backend Status")
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if r.status_code == 200:
            st.success("✅ Backend connected")
        else:
            st.error("❌ Backend error")
    except:
        st.error("❌ Backend offline")
        st.caption(f"Expected at: {BACKEND_URL}")


# ══════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════

st.title("🕸️ Dead Internet Detector")
st.caption("Detect coordinated synthetic content ecosystems using graph analysis and AI")

st.divider()

# ── Input section ────────────────────────────────────
st.subheader("🎯 Enter Seed Domains to Investigate")
col1, col2 = st.columns([3, 1])

with col1:
    domains_input = st.text_area(
        "Enter domains (one per line)",
        value="example.com\nbbc.com\nwikipedia.org",
        height=120,
        help="Enter 2-5 domains you want to investigate. "
             "The system will crawl these and discover connected domains.",
        placeholder="suspicious-news.com\nfake-updates.net\nbreaking-truth-daily.com"
    )

with col2:
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("**Try these examples:**")
    if st.button("🟢 Legit sites", width='stretch'):
        st.session_state['preset'] = "bbc.com\nreuters.com\nnytimes.com"
    if st.button("🔴 Suspicious", width='stretch'):
        st.session_state['preset'] = "breaking-truth-daily.com\nreal-news-network.net\npatriot-updates-now.com"

# Apply preset if selected
if 'preset' in st.session_state:
    domains_input = st.session_state.pop('preset')

# Parse domains
raw_domains = [d.strip() for d in domains_input.strip().split('\n') if d.strip()]
domains = raw_domains[:max_domains]

if len(raw_domains) > max_domains:
    st.warning(f"⚠️ Only analyzing first {max_domains} domains (adjust in sidebar)")

st.caption(f"Domains to analyze: **{', '.join(domains)}**")

# ── Analyze button ───────────────────────────────────
analyze_clicked = st.button(
    "🔍 Analyze Domains",
    type="primary",
    width='stretch',
    disabled=len(domains) == 0
)


# ══════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════

if analyze_clicked and domains:

    # Progress display
    progress_bar = st.progress(0)
    status_text  = st.empty()

    status_text.text("🕷️ Step 1/4: Crawler Agent crawling domains...")
    progress_bar.progress(10)

    try:
        # Call backend
        status_text.text("🔬 Step 2/4: Fingerprint Analyst computing signals...")
        progress_bar.progress(30)

        with st.spinner("AI agents working... (1-3 minutes)"):
            start_time = time.time()
            response   = requests.post(
                f"{BACKEND_URL}/analyze",
                json={"domains": domains},
                timeout=180
            )
            elapsed = time.time() - start_time

        progress_bar.progress(70)
        status_text.text("🏗️ Step 3/4: Graph Builder updating Neo4j...")
        time.sleep(0.5)
        progress_bar.progress(90)
        status_text.text("⚖️ Step 4/4: Verdict Agent running GNN inference...")
        time.sleep(0.5)
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()

        if response.status_code != 200:
            st.error(f"❌ Analysis failed: {response.text}")
            st.stop()

        result = response.json()
        st.success(f"✅ Analysis complete in {elapsed:.0f} seconds")

        # ── Store result in session ──
        st.session_state['result'] = result

    except requests.exceptions.Timeout:
        progress_bar.empty()
        status_text.empty()
        st.error("❌ Request timed out. The analysis is taking too long. Try fewer domains.")
        st.stop()
    except requests.exceptions.ConnectionError:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Cannot connect to backend at {BACKEND_URL}. Make sure the FastAPI server is running.")
        st.stop()
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error: {str(e)}")
        st.stop()


# ══════════════════════════════════════════════════════
# RESULTS DISPLAY
# ══════════════════════════════════════════════════════

if 'result' in st.session_state:
    result = st.session_state['result']

    verdict    = result.get('cluster_verdict', 'UNKNOWN')
    confidence = result.get('max_confidence', 0.0)
    summary    = result.get('summary', '')

    st.divider()

    # ── Verdict badge ────────────────────────────────
    st.subheader("🎯 Verdict")

    if verdict == 'SYNTHETIC':
        st.markdown(
            f'<div class="verdict-synthetic">'
            f'🚨 SYNTHETIC ECOSYSTEM DETECTED<br/>'
            f'<span style="font-size:16px">Confidence: {confidence:.0%}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    elif verdict == 'REVIEW':
        st.markdown(
            f'<div class="verdict-review">'
            f'⚠️ REVIEW RECOMMENDED<br/>'
            f'<span style="font-size:16px">Confidence: {confidence:.0%}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="verdict-organic">'
            f'✅ ORGANIC<br/>'
            f'<span style="font-size:16px">Confidence: {confidence:.0%}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.info(f"📋 {summary}")

    # ── Metrics row ──────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Domains Analyzed", result.get('domains_analyzed', 0))
    with col2:
        st.metric("🔴 Synthetic", result.get('synthetic_domains', 0))
    with col3:
        st.metric("🟡 Review", result.get('review_domains', 0))
    with col4:
        st.metric("🟢 Organic", result.get('organic_domains', 0))

    st.divider()

    # ── Graph visualization ──────────────────────────
    st.subheader("🕸️ Domain Network Graph")

    try:
        graph_response = requests.get(f"{BACKEND_URL}/graph", timeout=15)
        if graph_response.status_code == 200:
            graph_data = graph_response.json()
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])

            if nodes:
                # Build Plotly network graph
                import math

                # Position nodes in a circle
                n = len(nodes)
                node_x, node_y = [], []
                for i, node in enumerate(nodes):
                    angle = 2 * math.pi * i / max(n, 1)
                    node_x.append(math.cos(angle))
                    node_y.append(math.sin(angle))

                node_positions = {
                    node['id']: (node_x[i], node_y[i])
                    for i, node in enumerate(nodes)
                }

                # Edge traces
                edge_x, edge_y = [], []
                for edge in edges[:100]:  # limit for performance
                    src = node_positions.get(edge['source'])
                    tgt = node_positions.get(edge['target'])
                    if src and tgt:
                        edge_x += [src[0], tgt[0], None]
                        edge_y += [src[1], tgt[1], None]

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(width=0.5, color='#444'),
                    hoverinfo='none',
                    name='Similarity edges'
                )

                # Node colors by verdict
                color_map = {
                    'SYNTHETIC': '#e74c3c',
                    'REVIEW':    '#e67e22',
                    'ORGANIC':   '#27ae60',
                }
                node_colors = [
                    color_map.get(n.get('verdict', 'ORGANIC'), '#888')
                    for n in nodes
                ]

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hovertemplate=(
                        '<b>%{customdata[0]}</b><br>'
                        'Verdict: %{customdata[1]}<br>'
                        'Signals: %{customdata[2]}<br>'
                        'Avg Similarity: %{customdata[3]:.3f}'
                        '<extra></extra>'
                    ),
                    customdata=[
                        [n['domain'], n.get('verdict','ORGANIC'),
                         n.get('signals',0), n.get('avg_similarity',0)]
                        for n in nodes
                    ],
                    marker=dict(
                        size=10,
                        color=node_colors,
                        line=dict(width=1, color='#fff'),
                    ),
                    name='Domains'
                )

                fig = go.Figure(
                    data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text=f'Domain Network — {len(nodes)} nodes, {len(edges)} edges',
                            font=dict(color='#fff', size=14)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        paper_bgcolor='#0e1117',
                        plot_bgcolor='#0e1117',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=500,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                )

                st.plotly_chart(fig, width='stretch')
                st.caption("🔴 Synthetic  🟠 Review  🟢 Organic — hover over nodes for details")
            else:
                st.info("Graph is empty — run an analysis first")
    except Exception as e:
        st.warning(f"Graph visualization unavailable: {str(e)}")

    st.divider()

    # ── Per-domain breakdown ─────────────────────────
    st.subheader("📋 Per-Domain Breakdown")

    domain_verdicts = result.get('domain_verdicts', {})
    if domain_verdicts:
        for domain, dv in domain_verdicts.items():
            v = dv.get('verdict', 'ORGANIC')
            icon = "🔴" if v == 'SYNTHETIC' else "🟡" if v == 'REVIEW' else "🟢"
            css_class = f"domain-{v.lower()}"

            with st.expander(f"{icon} {domain} — {v} (confidence: {dv.get('confidence',0):.0%})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    s1 = dv.get('signal_1_similarity', 0)
                    st.metric("Signal 1: Similarity",
                              "🚨 Triggered" if s1 else "✅ Clear")
                with col2:
                    s2 = dv.get('signal_2_cadence', 0)
                    st.metric("Signal 2: Cadence",
                              "🚨 Triggered" if s2 else "✅ Clear")
                with col3:
                    s3 = dv.get('signal_3_whois', 0)
                    st.metric("Signal 3: WHOIS",
                              "🚨 Triggered" if s3 else "✅ Clear")

                st.info(f"💬 {dv.get('explanation', 'No explanation available')}")
    else:
        st.info("Run an analysis to see per-domain results")

    st.divider()

    # ── Raw JSON expander ────────────────────────────
    with st.expander("🔧 Raw API Response (for debugging)"):
        st.json(result)

        # ── Download report ─────────────────────────────
    st.divider()
    st.subheader("📥 Download Report")

    import json as _json
    report = {
        "project":        "Dead Internet Detector",
        "analyzed_at":    result.get("analyzed_at", ""),
        "seed_domains":   result.get("seed_domains", []),
        "cluster_verdict": verdict,
        "confidence":     confidence,
        "summary":        summary,
        "domain_verdicts": domain_verdicts,
        "graph_stats":    result.get("graph_stats", {}),
    }
    report_json = _json.dumps(report, indent=2)

    st.download_button(
        label="📥 Download Full Report (JSON)",
        data=report_json,
        file_name="dead_internet_report.json",
        mime="application/json",
        width='stretch',
    )
