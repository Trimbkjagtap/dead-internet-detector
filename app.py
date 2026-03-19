# app.py
# Streamlit frontend for the Dead Internet Detector
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
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🕸️")
    st.title("Dead Internet Detector")
    st.caption("INFO 7390 | Northeastern University")

    st.divider()

    st.subheader("⚙️ Settings")
    max_domains = st.slider("Max domains to analyze", 1, 20, 5)

    st.divider()

    st.subheader("📖 How It Works")
    st.markdown("""
    1. **Crawl** — fetch content from seed domains + linked sites
    2. **Analyze** — compute 3 signals:
       - 🔵 Content similarity
       - 🟣 Publishing cadence
       - 🟢 Domain registration
    3. **Graph** — build domain network in Neo4j
    4. **Verdict** — GNN classifies as SYNTHETIC or ORGANIC

    **2-of-3 rule:** flagged SYNTHETIC only if 2+ signals converge.
    """)

    st.divider()

    st.subheader("⚖️ Ethical Use")
    st.markdown("""
    - Research & journalism purposes only
    - Human review required before any accusations
    - 2-of-3 rule prevents false positives
    - No personal data collected or stored
    - Do not use to target legitimate publishers
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

# ── About section ────────────────────────────────────
with st.expander("ℹ️ About This Project", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **The Dead Internet Detector** is an AI-powered system that detects coordinated
        synthetic content ecosystems — networks of fake websites designed to manipulate
        search rankings and public opinion.

        Unlike tools that check individual articles, this system asks the **network-level
        question**: is this entire corner of the internet artificially coordinated?

        ### 🔬 Detection Signals
        | Signal | Method |
        |--------|--------|
        | 🔵 Content Similarity | Sentence Transformers + Cosine |
        | 🟣 Cadence Anomaly | Isolation Forest ML |
        | 🟢 WHOIS Registration | Domain age heuristics |
        """)
    with col2:
        st.markdown("""
        ### 🤖 AI Components
        - **CrewAI** — 4 agents in sequence
        - **GPT-4o** — agent reasoning + cluster analysis
        - **Graph Neural Network** — domain classification
        - **Neo4j AuraDB** — graph database

        ### ⚖️ Ethical Guidelines
        - Research and journalism only
        - Human review before accusations
        - 2-of-3 signals required for verdict
        - No personal data collected
        """)

st.divider()

# ── Input section ────────────────────────────────────
st.subheader("🎯 Enter Seed Domains to Investigate")

col1, col2 = st.columns([3, 1])

with col1:
    domains_input = st.text_area(
        "🔍 Enter seed domains to investigate (one per line)",
        value="breaking-truth-daily.com\nreal-news-network.net\npatriot-updates-now.com",
        height=120,
        help="Enter 2-5 domains. The system will crawl these and discover connected domains automatically. Minimum 2 domains recommended for meaningful graph analysis.",
        placeholder="suspicious-news.com\nfake-updates.net\nbreaking-truth-daily.com"
    )

with col2:
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("**Try these examples:**")
    if st.button("🟢 Legit sites", use_container_width=True):
        st.session_state['preset'] = "bbc.com\nreuters.com\nnytimes.com"
    if st.button("🔴 Suspicious", use_container_width=True):
        st.session_state['preset'] = "breaking-truth-daily.com\nreal-news-network.net\npatriot-updates-now.com"

if 'preset' in st.session_state:
    domains_input = st.session_state.pop('preset')

raw_domains = [d.strip() for d in domains_input.strip().split('\n') if d.strip()]
domains = raw_domains[:max_domains]

if len(raw_domains) > max_domains:
    st.warning(f"⚠️ Only analyzing first {max_domains} domains (adjust slider in sidebar)")

if len(domains) < 2:
    st.info("💡 Tip: Enter at least 2 domains for meaningful network analysis")

st.caption(f"Domains to analyze: **{', '.join(domains)}**")

analyze_clicked = st.button(
    "🔍 Analyze Domains",
    type="primary",
    use_container_width=True,
    disabled=len(domains) == 0
)


# ══════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════

if analyze_clicked and domains:
    progress_bar = st.progress(0)
    status_text  = st.empty()

    status_text.text("🕷️ Step 1/4: Crawler Agent crawling domains...")
    progress_bar.progress(10)

    try:
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
        st.session_state['result'] = result

    except requests.exceptions.Timeout:
        progress_bar.empty()
        status_text.empty()
        st.error("❌ Request timed out. Try fewer domains.")
        st.stop()
    except requests.exceptions.ConnectionError:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Cannot connect to backend at {BACKEND_URL}")
        st.stop()
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error: {str(e)}")
        st.stop()


# ══════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════

if 'result' in st.session_state:
    result         = st.session_state['result']
    verdict        = result.get('cluster_verdict', 'UNKNOWN')
    confidence     = result.get('max_confidence', 0.0)
    summary        = result.get('summary', '')
    domain_verdicts = result.get('domain_verdicts', {})

    st.divider()

    # ── Verdict badge ────────────────────────────────
    st.subheader("🎯 Verdict")

    if verdict == 'SYNTHETIC':
        st.markdown(
            f'<div class="verdict-synthetic">🚨 SYNTHETIC ECOSYSTEM DETECTED<br/>'
            f'<span style="font-size:16px">Confidence: {confidence:.0%}</span></div>',
            unsafe_allow_html=True)
    elif verdict == 'REVIEW':
        st.markdown(
            f'<div class="verdict-review">⚠️ REVIEW RECOMMENDED<br/>'
            f'<span style="font-size:16px">Confidence: {confidence:.0%}</span></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="verdict-organic">✅ ORGANIC<br/>'
            f'<span style="font-size:16px">Confidence: {confidence:.0%}</span></div>',
            unsafe_allow_html=True)

    st.info(f"📋 {summary}")

    # ── Metrics ──────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Domains Analyzed", result.get('domains_analyzed', 0))
    with c2: st.metric("🔴 Synthetic", result.get('synthetic_domains', 0))
    with c3: st.metric("🟡 Review", result.get('review_domains', 0))
    with c4: st.metric("🟢 Organic", result.get('organic_domains', 0))

    st.divider()

    # ── Tabs ─────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🕸️ Network Graph",
        "📊 Signal Analysis",
        "🤖 GPT-4 Analysis",
        "📋 Domain Details"
    ])

    # ── TAB 1: Network Graph ──────────────────────────
    with tab1:
        st.subheader("Domain Network Graph")
        try:
            graph_response = requests.get(f"{BACKEND_URL}/graph", timeout=15)
            if graph_response.status_code == 200:
                graph_data = graph_response.json()
                nodes = graph_data.get('nodes', [])
                edges = graph_data.get('edges', [])

                if nodes:
                    n = len(nodes)
                    node_x, node_y = [], []
                    for i in range(n):
                        angle = 2 * math.pi * i / max(n, 1)
                        node_x.append(math.cos(angle))
                        node_y.append(math.sin(angle))

                    node_positions = {
                        nodes[i]['id']: (node_x[i], node_y[i])
                        for i in range(n)
                    }

                    edge_x, edge_y = [], []
                    for edge in edges[:100]:
                        src = node_positions.get(edge['source'])
                        tgt = node_positions.get(edge['target'])
                        if src and tgt:
                            edge_x += [src[0], tgt[0], None]
                            edge_y += [src[1], tgt[1], None]

                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y, mode='lines',
                        line=dict(width=0.5, color='#444'),
                        hoverinfo='none', name='Similarity edges'
                    )

                    color_map = {'SYNTHETIC': '#e74c3c', 'REVIEW': '#e67e22', 'ORGANIC': '#27ae60'}
                    node_colors = [color_map.get(nd.get('verdict', 'ORGANIC'), '#888') for nd in nodes]

                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers',
                        hovertemplate=(
                            '<b>%{customdata[0]}</b><br>'
                            'Verdict: %{customdata[1]}<br>'
                            'Signals: %{customdata[2]}<br>'
                            '<extra></extra>'
                        ),
                        customdata=[[nd['domain'], nd.get('verdict','ORGANIC'), nd.get('signals',0)] for nd in nodes],
                        marker=dict(size=10, color=node_colors, line=dict(width=1, color='#fff')),
                        name='Domains'
                    )

                    fig = go.Figure(
                        data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=dict(text=f'Domain Network — {len(nodes)} nodes, {len(edges)} edges', font=dict(color='#fff', size=14)),
                            showlegend=False, hovermode='closest',
                            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=500, margin=dict(l=20, r=20, t=40, b=20),
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("🔴 Synthetic  🟠 Review  🟢 Organic — hover over nodes for details")
                else:
                    st.info("Graph is empty — run an analysis first")
        except Exception as e:
            st.warning(f"Graph visualization unavailable: {str(e)}")

    # ── TAB 2: Signal Analysis ────────────────────────
    with tab2:
        st.subheader("Signal Analysis Charts")

        if domain_verdicts:
            signal_data = []
            for domain, dv in domain_verdicts.items():
                signal_data.append({
                    'Domain':                domain[:35],
                    'Synthetic Probability': round(dv.get('confidence', 0), 4),
                    'Signals Triggered':     dv.get('signals_triggered', 0),
                    'Verdict':               dv.get('verdict', 'ORGANIC'),
                    'Similarity':            dv.get('signal_1_similarity', 0),
                    'Cadence':               dv.get('signal_2_cadence', 0),
                    'WHOIS':                 dv.get('signal_3_whois', 0),
                })
            df_sig = pd.DataFrame(signal_data)

            # Chart 1 — Synthetic probability
            cmap = {'SYNTHETIC': '#e74c3c', 'REVIEW': '#e67e22', 'ORGANIC': '#27ae60'}
            fig1 = px.bar(
                df_sig, x='Domain', y='Synthetic Probability',
                color='Verdict', color_discrete_map=cmap,
                title='Synthetic Probability by Domain',
            )
            fig1.update_layout(
                paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                font_color='#e0e0e0', xaxis_tickangle=-45,
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Chart 2 — Signal heatmap
            heat_df = df_sig[['Domain', 'Similarity', 'Cadence', 'WHOIS']].set_index('Domain')
            fig2 = px.imshow(
                heat_df.T,
                color_continuous_scale=['#1a4a1a', '#e74c3c'],
                title='Signal Heatmap — Red = Triggered',
                aspect='auto',
                labels=dict(color='Triggered'),
            )
            fig2.update_layout(
                paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                font_color='#e0e0e0',
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Summary stats
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Avg Synthetic Probability",
                          f"{df_sig['Synthetic Probability'].mean():.1%}")
            with c2:
                st.metric("Max Signals Triggered",
                          f"{df_sig['Signals Triggered'].max()}/3")
            with c3:
                st.metric("Domains Above 50% Confidence",
                          len(df_sig[df_sig['Synthetic Probability'] > 0.5]))

            # Data table
            st.dataframe(df_sig, use_container_width=True)

    # ── TAB 3: GPT-4 Analysis ─────────────────────────
    with tab3:
        st.subheader("🤖 GPT-4o AI Analysis")
        st.caption("Powered by OpenAI GPT-4o via CrewAI agents")

        if st.button("Generate AI Analysis", type="primary"):
            with st.spinner("GPT-4o analyzing the domain cluster..."):
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

                    prompt = f"""You are an expert in detecting coordinated inauthentic behavior on the internet.

Analyze this domain cluster and provide a structured assessment:

Domains analyzed:
{domain_summary}

Overall cluster verdict: {verdict}
Maximum synthetic probability: {confidence:.0%}

Provide:
1. What the signal pattern suggests about this network
2. Whether this looks like organic or coordinated behavior and why
3. Key indicators that influenced this assessment
4. What a journalist or researcher should investigate next

Be concise, technical but accessible. Use bullet points."""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=350,
                        temperature=0.3,
                    )

                    ai_analysis = response.choices[0].message.content
                    st.session_state['ai_analysis'] = ai_analysis

                except Exception as e:
                    st.error(f"AI analysis failed: {str(e)}")

        if 'ai_analysis' in st.session_state:
            st.markdown("### GPT-4o Assessment")
            st.info(st.session_state['ai_analysis'])
            st.caption("⚠️ AI analysis is advisory only — always apply human judgment")

    # ── TAB 4: Domain Details ─────────────────────────
    with tab4:
        st.subheader("Per-Domain Breakdown")

        if domain_verdicts:
            for domain, dv in domain_verdicts.items():
                v    = dv.get('verdict', 'ORGANIC')
                icon = "🔴" if v == 'SYNTHETIC' else "🟡" if v == 'REVIEW' else "🟢"

                with st.expander(f"{icon} {domain} — {v} (confidence: {dv.get('confidence',0):.0%})"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        s1 = dv.get('signal_1_similarity', 0)
                        st.metric("Signal 1: Similarity", "🚨 Triggered" if s1 else "✅ Clear")
                    with c2:
                        s2 = dv.get('signal_2_cadence', 0)
                        st.metric("Signal 2: Cadence", "🚨 Triggered" if s2 else "✅ Clear")
                    with c3:
                        s3 = dv.get('signal_3_whois', 0)
                        st.metric("Signal 3: WHOIS", "🚨 Triggered" if s3 else "✅ Clear")
                    st.info(f"💬 {dv.get('explanation', 'No explanation available')}")
        else:
            st.info("Run an analysis to see per-domain results")

    st.divider()

    # ── Download report ──────────────────────────────
    st.subheader("📥 Download Report")
    report = {
        "project":         "Dead Internet Detector",
        "analyzed_at":     result.get("analyzed_at", ""),
        "seed_domains":    result.get("seed_domains", []),
        "cluster_verdict": verdict,
        "confidence":      confidence,
        "summary":         summary,
        "domain_verdicts": domain_verdicts,
        "graph_stats":     result.get("graph_stats", {}),
    }
    st.download_button(
        label="📥 Download Full Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="dead_internet_report.json",
        mime="application/json",
        use_container_width=True,
    )

    st.divider()

    # ── Raw JSON ─────────────────────────────────────
    with st.expander("🔧 Raw API Response (for debugging)"):
        st.json(result)