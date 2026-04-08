# agents/verdict_agent.py
# Agent 4 — Verdict Agent
# Runs GNN inference and produces final verdict with explanation
# Confidence is computed from actual signal strength, not just GNN output

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH  = "models/gnn_model.pt"
THRESHOLD   = 0.5


# ── Re-define GNN architecture ───────────────────────
class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)

    def forward(self, x, edge_index, n_nodes):
        row, col = edge_index
        agg = torch.zeros(n_nodes, x.shape[1], device=x.device)
        agg.scatter_add_(0, col.unsqueeze(1).expand(-1, x.shape[1]), x[row])
        deg = torch.bincount(col, minlength=n_nodes).float().clamp(min=1)
        return self.linear(agg / deg.unsqueeze(1))


class GCN(nn.Module):
    def __init__(self, in_f, hidden, n_classes, dropout=0.3):
        super().__init__()
        self.conv1   = GCNLayer(in_f, hidden)
        self.conv2   = GCNLayer(hidden, hidden // 2)
        self.linear  = nn.Linear(hidden // 2, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, n_nodes):
        x = F.relu(self.conv1(x, edge_index, n_nodes))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, n_nodes))
        x = self.dropout(x)
        return self.linear(x)


def load_gnn_model():
    """Load the trained GNN model from disk."""
    checkpoint    = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    feature_cols  = checkpoint['feature_cols']
    n_features    = checkpoint['n_features']
    scaler_mean   = np.array(checkpoint['scaler_mean'])
    scaler_scale  = np.array(checkpoint['scaler_scale'])

    model = GCN(in_f=n_features, hidden=64, n_classes=2, dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, feature_cols, scaler_mean, scaler_scale


def build_graph_tensors(features: dict, sim_edges: dict):
    """Convert feature dict into PyTorch tensors for GNN inference."""
    domain_list   = list(features.keys())
    domain_to_idx = {d: i for i, d in enumerate(domain_list)}
    n_nodes       = len(domain_list)

    _, feature_cols, scaler_mean, scaler_scale = load_gnn_model()

    X = []
    for domain in domain_list:
        feats = features[domain]
        row   = [float(feats.get(c, 0)) for c in feature_cols]
        X.append(row)

    X = np.array(X, dtype=np.float32)
    safe_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)
    X = (X - scaler_mean) / safe_scale
    X_tensor = torch.tensor(X, dtype=torch.float32)

    edge_src, edge_dst = [], []
    for key in sim_edges:
        if "|||" in str(key):
            a, b = str(key).split("|||")
        elif isinstance(key, (list, tuple)) and len(key) == 2:
            a, b = key
        else:
            continue
        if a in domain_to_idx and b in domain_to_idx:
            i, j = domain_to_idx[a], domain_to_idx[b]
            edge_src.extend([i, j])
            edge_dst.extend([j, i])

    for i in range(n_nodes):
        edge_src.append(i)
        edge_dst.append(i)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    return X_tensor, edge_index, n_nodes, domain_list


def compute_confidence(feats: dict, syn_prob: float) -> float:
    """Delegates to the canonical formula in config.signal_config."""
    from config.signal_config import compute_confidence_from_signals
    return compute_confidence_from_signals(
        signals  = int(feats.get('signals_triggered', 0)),
        max_sim  = float(feats.get('max_similarity', 0)),
        burst    = float(feats.get('burst_score', 0)),
        gnn_prob = float(syn_prob),
    )


def _generate_explanation_template(domain: str, feats: dict) -> str:
    """Fallback template-based explanation (used when OpenAI is unavailable)."""
    reasons = []

    if feats.get('similarity_flag', 0):
        reasons.append(
            f"content is unusually similar to other domains "
            f"(similarity {feats.get('max_similarity',0):.2f})"
        )
    if feats.get('cadence_flagged', 0):
        reasons.append("anomalous publishing time pattern detected")
    if feats.get('whois_flagged', 0):
        reasons.append("suspicious domain registration pattern")
    if feats.get('hosting_flagged', 0):
        reasons.append("shares hosting infrastructure with other analyzed domains")
    if feats.get('link_network_flagged', 0):
        mutual  = int(feats.get('mutual_link_count', 0))
        insular = float(feats.get('insular_score', 0))
        if mutual > 0:
            reasons.append(f"mutually links with {mutual} other domain(s) in the cluster")
        else:
            reasons.append(f"{insular:.0%} of outgoing links stay within the analyzed cluster")
    if feats.get('wayback_flagged', 0):
        reason = feats.get('wayback_flag_reason', '')
        count  = int(feats.get('wayback_snapshot_count', 0))
        if reason == 'new_site':
            reasons.append(f"only {count} Wayback Machine snapshot(s) — site appears very new")
        else:
            recent = int(feats.get('wayback_recent_count', 0))
            reasons.append(f"archive spike: {recent} of {count} snapshots are from the last 30 days")
    if feats.get('author_overlap_flagged', 0):
        try:
            shared = json.loads(feats.get('shared_authors', '[]'))
        except Exception:
            shared = []
        names = ", ".join(f'"{a}"' for a in shared[:3])
        reasons.append(f"author name(s) {names} appear across multiple domains")

    signals = feats.get('signals_triggered', 0)
    if not reasons:
        return (
            f"{domain} passed all 7 checks — no suspicious patterns detected. "
            f"Content, hosting, link structure, archive history, and registration data all appear organic."
        )
    return (
        f"{domain} triggered {signals}/7 signals: "
        + "; ".join(reasons) + ". "
        + (f"With {signals} signals converging, the 3-of-7 rule classifies this as a synthetic network."
           if signals >= 3 else
           f"{signals} signal(s) detected — flagged for human review. Not enough to confirm a fake network alone.")
    )


def generate_explanation(domain: str, feats: dict) -> str:
    """
    Generate plain-English explanation for the verdict.
    Uses GPT-4o-mini when OPENAI_API_KEY is set and signals > 0.
    Falls back to template strings gracefully if GPT is unavailable.
    """
    signals = int(feats.get('signals_triggered', 0))
    if signals == 0:
        return _generate_explanation_template(domain, feats)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return _generate_explanation_template(domain, feats)

    # Build a compact signal summary with only triggered signals + their raw scores
    triggered_lines = []
    if feats.get('similarity_flag', 0):
        triggered_lines.append(
            f"Signal 1 (Content Similarity): TRIGGERED — max cosine similarity = {feats.get('max_similarity',0):.3f}"
        )
    if feats.get('cadence_flagged', 0):
        triggered_lines.append(
            f"Signal 2 (Cadence Anomaly): TRIGGERED — burst_score = {feats.get('burst_score',0):.3f}"
        )
    if feats.get('whois_flagged', 0):
        triggered_lines.append(
            f"Signal 3 (WHOIS Age): TRIGGERED — domain_age = {feats.get('domain_age_days','?')} days"
        )
    if feats.get('hosting_flagged', 0):
        triggered_lines.append(
            f"Signal 4 (Shared Hosting): TRIGGERED — hosting_org = {feats.get('hosting_org','?')}, "
            f"ip = {feats.get('ip_address','?')}"
        )
    if feats.get('link_network_flagged', 0):
        triggered_lines.append(
            f"Signal 5 (Link Network): TRIGGERED — insular_score = {feats.get('insular_score',0):.2f}, "
            f"mutual_links = {feats.get('mutual_link_count',0)}"
        )
    if feats.get('wayback_flagged', 0):
        triggered_lines.append(
            f"Signal 6 (Wayback History): TRIGGERED — snapshots = {feats.get('wayback_snapshot_count',0)}, "
            f"reason = {feats.get('wayback_flag_reason','?')}"
        )
    if feats.get('author_overlap_flagged', 0):
        triggered_lines.append(
            f"Signal 7 (Author Overlap): TRIGGERED — shared_authors = {feats.get('shared_authors','[]')}"
        )

    signal_block = "\n".join(triggered_lines)
    verdict_tier = (
        "SYNTHETIC (3+ signals — classified as coordinated fake network)"
        if signals >= 3 else
        "REVIEW (1-2 signals — flagged for human investigation)"
    )

    system_prompt = (
        "You are a forensic analyst explaining disinformation detection results to an investigative journalist. "
        "For each triggered signal, write exactly one clear sentence stating what the specific numbers mean "
        "and why they are suspicious. Use the actual values. Never hedge with 'may' or 'could' — "
        "say what the data shows. Be concise.\n\n"
        "IMPORTANT — when Signal 1 (Content Similarity) is triggered, you MUST explicitly distinguish "
        "between two scenarios:\n"
        "  A) CLONED CONTENT (bot network): near-identical HTML structure, same article text verbatim, "
        "same bylines, no attribution — indicates coordinated content farming.\n"
        "  B) SHARED JOURNALISTIC REPORTING (wire service): similar story framing, different wording, "
        "AP/Reuters attribution, legitimate local outlet context — this is expected and NOT suspicious.\n"
        "State which scenario the similarity score and context suggests, and why."
    )
    user_prompt = (
        f"Domain: {domain}\n"
        f"Verdict: {verdict_tier}\n\n"
        f"Triggered signals ({signals}/{7}):\n{signal_block}\n\n"
        f"Write a numbered list — one sentence per triggered signal explaining what it means. "
        f"For Signal 1 (if present), explicitly state whether this looks like CLONED CONTENT or "
        f"SHARED JOURNALISTIC REPORTING based on the similarity score and any available context. "
        f"End with one sentence summarising the overall risk tier."
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=250,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"   ⚠️  GPT explanation failed for {domain}: {e} — using template fallback")
        return _generate_explanation_template(domain, feats)


def gpt_calibration_check(domain: str, feats: dict, verdict: str, confidence: float) -> str:
    """
    Day 4: Ask GPT to sanity-check the verdict against the raw signal scores.
    Returns a short string starting with AGREE, CHALLENGE, or UNCERTAIN.
    Only called when signals >= 1. Returns empty string on failure.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return ""

    signals = int(feats.get('signals_triggered', 0))
    summary = (
        f"Domain: {domain} | Verdict: {verdict} | Confidence: {confidence:.0%} | "
        f"Signals fired: {signals}/7\n"
        f"max_similarity={feats.get('max_similarity',0):.3f}, "
        f"burst_score={feats.get('burst_score',0):.3f}, "
        f"domain_age_days={feats.get('domain_age_days','?')}, "
        f"insular_score={feats.get('insular_score',0):.2f}, "
        f"wayback_snapshots={feats.get('wayback_snapshot_count','?')}, "
        f"hosting_org={feats.get('hosting_org','?')}\n"
        f"Signal flags: sim={feats.get('similarity_flag',0)}, cadence={feats.get('cadence_flagged',0)}, "
        f"whois={feats.get('whois_flagged',0)}, hosting={feats.get('hosting_flagged',0)}, "
        f"links={feats.get('link_network_flagged',0)}, wayback={feats.get('wayback_flagged',0)}, "
        f"authors={feats.get('author_overlap_flagged',0)}"
    )

    system_prompt = (
        "You are a disinformation detection calibration assistant. "
        "Evaluate the raw signal scores and say whether the verdict looks well-supported. "
        "Reply with exactly one of: AGREE, CHALLENGE, or UNCERTAIN — then one sentence of reasoning. "
        "Example: 'AGREE — three independent signals with strong numeric scores support the SYNTHETIC verdict.'"
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": summary},
            ],
            max_tokens=80,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"   ⚠️  GPT calibration check failed for {domain}: {e}")
        return ""


def gpt_author_analysis(domain: str, shared_authors_list: list) -> str:
    """
    Day 7: Assess whether shared author names look like real people or pseudonyms.
    Returns LIKELY_REAL / POSSIBLY_FAKE / LIKELY_FAKE + 2 sentences. Empty string on failure.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or not shared_authors_list:
        return ""

    names_str = ", ".join(f'"{a}"' for a in shared_authors_list[:10])
    system_prompt = (
        "You are an investigative journalist trained to spot fake author personas used in disinformation networks. "
        "Analyze a list of author names that appear across multiple domains. "
        "Assess: do these look like real people (varied, culturally plausible) or likely pseudonyms "
        "(generic, alliterative, pattern-matched, or suspiciously similar)? "
        "Output exactly: LIKELY_REAL, POSSIBLY_FAKE, or LIKELY_FAKE — then exactly two sentences."
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Shared author names on {domain}: {names_str}"},
            ],
            max_tokens=100,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"   ⚠️  GPT author analysis failed for {domain}: {e}")
        return ""


def gpt_intent_classify(domain: str, excerpt: str) -> str:
    """
    Day 8: Classify domain intent from homepage text.
    Returns one of: NEWS / PROPAGANDA / COMMERCIAL / SATIRICAL / AGGREGATOR / UNKNOWN
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or not excerpt or len(excerpt.strip()) < 20:
        return "UNKNOWN"

    system_prompt = (
        "Classify this website's intent based on its homepage text. "
        "Output exactly one word from: NEWS, PROPAGANDA, COMMERCIAL, SATIRICAL, AGGREGATOR, UNKNOWN. "
        "No explanation, no punctuation — just the single word."
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": excerpt[:600]},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        result = resp.choices[0].message.content.strip().upper()
        valid = {"NEWS", "PROPAGANDA", "COMMERCIAL", "SATIRICAL", "AGGREGATOR", "UNKNOWN"}
        return result if result in valid else "UNKNOWN"
    except Exception as e:
        print(f"   ⚠️  GPT intent classification failed for {domain}: {e}")
        return "UNKNOWN"


def produce_verdict(features: dict, sim_edges: dict, excerpts: dict = None) -> dict:
    """
    Main verdict function.
    Runs GNN inference and produces final verdict for all domains.
    
    Verdict logic (3-of-7 rule):
    - SYNTHETIC: 3 or more signals triggered
    - REVIEW: 1–2 signals triggered
    - ORGANIC: no signals triggered
    
    Confidence is computed from signal strength + similarity scores,
    not solely from GNN output (which overfits on small training data).
    """
    print(f"⚖️  Verdict Agent starting...")
    print(f"   Domains to evaluate: {len(features)}")

    if not features:
        return {'error': 'No domain features provided', 'verdicts': {}}

    try:
        model, feature_cols, scaler_mean, scaler_scale = load_gnn_model()
        print(f"   ✅ GNN model loaded")

        X_tensor, edge_index, n_nodes, domain_list = build_graph_tensors(
            features, sim_edges
        )

        # Run GNN inference (used as one input to confidence)
        with torch.no_grad():
            out  = model(X_tensor, edge_index, n_nodes)
            prob = F.softmax(out, dim=1)
            synthetic_probs = prob[:, 1].numpy()

        print(f"   ✅ GNN inference complete")

        verdicts = {}
        for i, domain in enumerate(domain_list):
            feats    = features[domain]
            syn_prob = float(synthetic_probs[i])
            signals  = int(feats.get('signals_triggered', 0))

            confidence = compute_confidence(feats, syn_prob)

            # ── Strict 3-of-7 enforcement ─────────────────────────────────────
            # A single signal (1/7) is NEVER enough to flag a domain as Suspicious
            # or High Risk. A lone signal could always be explained by legitimate
            # journalism, geographic co-coverage, or generic CDN usage.
            # Any domain with signals == 1 is automatically ORGANIC.
            # REVIEW requires 2 signals; SYNTHETIC requires 3+.

            # Verdict
            if signals >= 3:
                verdict = 'SYNTHETIC'
            elif signals == 2:
                verdict = 'REVIEW'
            else:
                # 0 or 1 signal → ORGANIC
                verdict = 'ORGANIC'
                sim_boost_org     = min(float(feats.get('max_similarity', 0)) * 0.20, 0.20)
                cadence_boost_org = min(float(feats.get('burst_score', 0)) * 0.15, 0.10)
                gnn_boost_org     = syn_prob * 0.05
                confidence = round(max(0.70, min(1.0 - (sim_boost_org + cadence_boost_org + gnn_boost_org), 0.97)), 2)

            explanation = generate_explanation(domain, feats)

            # When exactly 1 signal fired, append a clear "single-signal" disclaimer
            if signals == 1:
                fired_name = (
                    "content similarity"    if feats.get('similarity_flag') else
                    "cadence anomaly"       if feats.get('cadence_flagged') else
                    "WHOIS age"             if feats.get('whois_flagged') else
                    "shared hosting"        if feats.get('hosting_flagged') else
                    "insular link network"  if feats.get('link_network_flagged') else
                    "Wayback history"       if feats.get('wayback_flagged') else
                    "shared author bylines" if feats.get('author_overlap_flagged') else
                    "unknown signal"
                )
                explanation = (
                    f"{domain} — {fired_name} signal fired (1/7), but no other signals "
                    f"corroborate it. The 3-of-7 convergence rule requires at least 2 independent "
                    f"signals before flagging a domain as Suspicious. A single signal may reflect "
                    f"legitimate syndication, regional topic overlap, or common hosting. "
                    f"Verdict: Looks Legitimate."
                )

            # Day 4: GPT confidence calibration check (skip clean ORGANIC to save cost)
            gpt_review = ""
            if signals >= 1:
                gpt_review = gpt_calibration_check(domain, feats, verdict, confidence)

            # Day 7: GPT author pseudonym analysis (only when Signal 7 fires)
            author_gpt_note = ""
            if feats.get('author_overlap_flagged', 0):
                try:
                    shared_list = json.loads(feats.get('shared_authors', '[]'))
                except Exception:
                    shared_list = []
                if shared_list:
                    author_gpt_note = gpt_author_analysis(domain, shared_list)

            # Day 8: GPT domain intent classification from homepage excerpt
            excerpt_text = (excerpts or {}).get(domain, "")
            gpt_intent = gpt_intent_classify(domain, excerpt_text)

            verdicts[domain] = {
                'verdict':               verdict,
                'confidence':            confidence,
                'gnn_raw_score':         round(syn_prob, 4),
                'gpt_confidence_review': gpt_review,
                'author_gpt_note':       author_gpt_note,
                'gpt_domain_intent':     gpt_intent,
                'signals_triggered':     signals,
                'signal_1_similarity':   int(feats.get('similarity_flag', 0)),
                'signal_2_cadence':      int(feats.get('cadence_flagged', 0)),
                'signal_3_whois':        int(feats.get('whois_flagged', 0)),
                'signal_4_hosting':      int(feats.get('hosting_flagged', 0)),
                'signal_5_link_network': int(feats.get('link_network_flagged', 0)),
                'signal_6_wayback':      int(feats.get('wayback_flagged', 0)),
                'signal_7_authors':      int(feats.get('author_overlap_flagged', 0)),
                # Pass through display fields
                'max_similarity':        float(feats.get('max_similarity', 0)),
                'burst_score':           float(feats.get('burst_score', 0)),
                'domain_age_days':       int(feats.get('domain_age_days', -1)),
                'domain_age_source':     str(feats.get('domain_age_source', 'unknown')),
                'ip_address':            str(feats.get('ip_address', '')),
                'hosting_org':           str(feats.get('hosting_org', '')),
                'insular_score':         float(feats.get('insular_score', 0)),
                'wayback_snapshot_count': int(feats.get('wayback_snapshot_count', 0)),
                'wayback_flag_reason':   str(feats.get('wayback_flag_reason', '')),
                'shared_authors':        str(feats.get('shared_authors', '[]')),
                'explanation':           explanation,
            }

        synthetic_count = sum(1 for v in verdicts.values() if v['verdict'] == 'SYNTHETIC')
        review_count    = sum(1 for v in verdicts.values() if v['verdict'] == 'REVIEW')
        max_confidence  = max(v['confidence'] for v in verdicts.values()) if verdicts else 0

        if synthetic_count > 0:
            cluster_verdict = 'SYNTHETIC'
        elif review_count > 0:
            cluster_verdict = 'REVIEW'
        else:
            cluster_verdict = 'ORGANIC'

        result = {
            'cluster_verdict':   cluster_verdict,
            'max_confidence':    round(max_confidence, 4),
            'synthetic_domains': synthetic_count,
            'review_domains':    review_count,
            'organic_domains':   len(verdicts) - synthetic_count - review_count,
            'domain_verdicts':   verdicts,
        }

        print(f"✅ Verdict Agent complete!")
        print(f"   Cluster verdict: {cluster_verdict}")
        print(f"   Max confidence:  {max_confidence:.2%}")
        
        # Print per-domain summary
        for domain, v in verdicts.items():
            icon = "🔴" if v['verdict'] == 'SYNTHETIC' else "🟡" if v['verdict'] == 'REVIEW' else "🟢"
            print(f"   {icon} {domain}: {v['verdict']} "
                  f"(conf={v['confidence']:.0%}, signals={v['signals_triggered']}, "
                  f"gnn={v['gnn_raw_score']:.2f})")
        
        return result

    except Exception as e:
        print(f"   ❌ Verdict Agent error: {e}")
        return {'error': str(e), 'verdicts': {}}


def produce_verdict_tool(input_json: str) -> str:
    """CrewAI tool wrapper."""
    data      = json.loads(input_json)
    features  = data.get('features', {})
    sim_edges = data.get('sim_edges', {})
    result    = produce_verdict(features, sim_edges)
    return json.dumps(result)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from agents.crawler_agent import crawl_domains
    from agents.fingerprint_agent import analyze_domains

    print("Testing Verdict Agent...")
    data     = crawl_domains(["example.com", "bbc.com"])
    analysis = analyze_domains(data)
    verdict  = produce_verdict(
        analysis['features'],
        analysis.get('sim_edges', {})
    )

    print()
    print("Cluster verdict:", verdict['cluster_verdict'])
    print("Max confidence: ", verdict['max_confidence'])
    for domain, v in verdict['domain_verdicts'].items():
        print(f"  {domain}: {v['verdict']} (conf={v['confidence']:.0%}, signals={v['signals_triggered']})")
    print("\n🎉 Verdict Agent test passed!")