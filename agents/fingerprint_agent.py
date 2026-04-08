# agents/fingerprint_agent.py
# Agent 2 — Fingerprint Analyst Agent
# Takes crawled domain data, computes all 3 signals
# Excludes subdomain-to-parent comparisons from similarity

import json
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from sentence_transformers import SentenceTransformer
from config.signal_config import (
    MODEL_NAME,
    SIM_THRESHOLD,
    bounded_contamination,
    IPINFO_TOKEN,
    INSULAR_SCORE_THRESHOLD,
    MIN_LINKS_FOR_INSULAR,
    MIN_DOMAINS_FOR_INSULAR,
    WAYBACK_MIN_SNAPSHOTS,
    WAYBACK_SPIKE_RATIO,
    AUTHORITY_DOMAINS,
    AUTHORITY_PAIR_WEIGHT,
    AUTHORITY_MIXED_WEIGHT,
    AUTHORITY_PAIR_THRESHOLD_MULTIPLIER,
    AUTHORITY_MIXED_THRESHOLD_MULTIPLIER,
    SYNDICATION_MARKERS,
)
from services.whois_service import WhoisService
from services.enrichment_service import resolve_ip_and_asn, get_wayback_data

# ── Config ───────────────────────────────────────────
CADENCE_THRESHOLD = -0.1

# Load model once at module level (cached after first load)
_model = None

def get_model():
    global _model
    if _model is None:
        print("  Loading Sentence Transformer model...")
        _model = SentenceTransformer(MODEL_NAME)
        print("  ✅ Model loaded")
    return _model


def get_parent_domain(domain: str) -> str:
    """Extract parent domain from subdomain."""
    parts = domain.split('.')
    if len(parts) >= 3 and parts[-2] in ['co', 'com', 'org', 'net', 'gov']:
        return '.'.join(parts[-3:])
    elif len(parts) >= 2:
        return '.'.join(parts[-2:])
    return domain


def is_same_site(domain_a: str, domain_b: str) -> bool:
    """
    Check if two domains are from the same site.
    e.g. bostonglobe.com and sponsored.bostonglobe.com are the same site.
    e.g. twitter.com and x.com are different sites (handled by redirect detection).
    """
    return get_parent_domain(domain_a) == get_parent_domain(domain_b)


def _classify_similarity(
    raw_sim: float,
    authors_a: list,
    authors_b: list,
    text_a: str,
    text_b: str,
) -> str:
    """
    Classify a content similarity pair as one of:
      'structural_clone'         — identical boilerplate/nav + body, strong cloning signal
      'shared_journalistic'      — similar body text but unique bylines and structure
      'topic_overlap'            — moderate sim, no shared elements beyond topic

    Logic (requirement 3):
    - If both domains have non-empty, completely disjoint author sets → NOT a structural
      clone (real personas differ across independent outlets).
    - If sim >= 0.90 (near-verbatim) AND authors overlap → structural_clone.
    - If sim >= 0.75 but authors are disjoint → shared_journalistic (wire service pattern).
    - Otherwise → topic_overlap.
    """
    # Normalise author sets (lowercase, strip)
    set_a = {a.lower().strip() for a in authors_a if a.strip()}
    set_b = {b.lower().strip() for b in authors_b if b.strip()}
    authors_overlap = bool(set_a & set_b) if (set_a and set_b) else None  # None = unknown

    # Near-verbatim body text with shared authors → structural clone
    if raw_sim >= 0.90 and authors_overlap:
        return 'structural_clone'

    # High-sim but completely different bylines → wire-service syndication
    if raw_sim >= 0.75 and authors_overlap is False:
        return 'shared_journalistic'

    # High-sim, author data unknown (one/both had no bylines captured)
    # Check for boilerplate nav phrases duplicated across texts as a proxy
    _NAV_PHRASES = [
        "home", "about us", "contact", "privacy policy", "terms of service",
        "subscribe", "newsletter", "advertise", "cookies",
    ]
    text_a_l = text_a.lower()[:600]
    text_b_l = text_b.lower()[:600]
    nav_matches = sum(1 for p in _NAV_PHRASES if p in text_a_l and p in text_b_l)
    # Many nav phrases in common (≥5) at 80%+ sim → likely templated clone
    if raw_sim >= 0.80 and nav_matches >= 5:
        return 'structural_clone'

    if raw_sim >= 0.65:
        return 'shared_journalistic'

    return 'topic_overlap'


def _load_corpus() -> list:
    """Load fallback CSV domains with text for corpus comparison."""
    import os
    import pandas as pd
    corpus = []
    for path in ['data/domains_clean.csv', 'data/synthetic_ecosystem.csv',
                 'data/domains_raw.csv', 'data/ground_truth.csv']:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                domain = str(row.get('domain', '')).strip().lower()
                if not domain:
                    continue
                text = ''
                for col in ['text', 'content', 'page_text', 'body']:
                    if col in df.columns and pd.notna(row.get(col)):
                        text = str(row[col])[:1000]
                        break
                if text and len(text) > 50:
                    corpus.append({'domain': domain, 'text': text})
        except Exception:
            continue
    # deduplicate by domain
    seen = set()
    deduped = []
    for d in corpus:
        if d['domain'] not in seen:
            seen.add(d['domain'])
            deduped.append(d)
    return deduped


def _compute_signal1_against_corpus(domains_data: list) -> dict:
    """
    When only one domain was crawled, compare it against the fallback corpus
    to surface similar domains as evidence pairs.
    """
    corpus = _load_corpus()
    if not corpus:
        return {'domain_scores': {}, 'edges': {}, 'excerpts': {}}

    seed = domains_data[0]
    seed_domain = seed['domain']
    seed_text   = seed.get('text', '')[:1000]
    excerpts    = {seed_domain: seed.get('text', '')[:400].strip()}

    # Domains that contain content from everywhere — comparing against them is meaningless
    _NOISE_DOMAINS = {
        'archive.is', 'archive.today', 'archive.org', 'web.archive.org',
        'webcache.googleusercontent.com', 'cached.google.com',
        'translate.google.com', 'amp.google.com',
        'google.com', 'bing.com', 'yahoo.com', 'duckduckgo.com',
        'reddit.com', 'twitter.com', 'x.com', 'facebook.com',
        'linkedin.com', 'pinterest.com', 'tumblr.com',
        'wikipedia.org', 'wikimedia.org',
        'pastebin.com', 'hastebin.com',
    }

    # Exclude the seed domain itself and noise domains from corpus
    corpus = [d for d in corpus if d['domain'] != seed_domain and d['domain'] not in _NOISE_DOMAINS]
    if not corpus:
        return {'domain_scores': {}, 'edges': {}, 'excerpts': excerpts}

    model = get_model()
    all_texts   = [seed_text] + [d['text'] for d in corpus]
    all_domains = [seed_domain] + [d['domain'] for d in corpus]
    embeddings  = model.encode(all_texts, show_progress_bar=False)
    sim_matrix  = cosine_similarity(embeddings)

    edges = {}
    flagged_pairs = []
    for j in range(1, len(all_domains)):
        raw_sim = float(sim_matrix[0][j])
        corpus_domain = all_domains[j]
        d_auth = seed_domain in AUTHORITY_DOMAINS
        c_auth = corpus_domain in AUTHORITY_DOMAINS
        if d_auth and c_auth:
            effective_sim  = raw_sim * AUTHORITY_PAIR_WEIGHT
            pair_threshold = SIM_THRESHOLD * AUTHORITY_PAIR_THRESHOLD_MULTIPLIER
        elif d_auth or c_auth:
            effective_sim  = raw_sim * AUTHORITY_MIXED_WEIGHT
            pair_threshold = SIM_THRESHOLD * AUTHORITY_MIXED_THRESHOLD_MULTIPLIER
        else:
            effective_sim  = raw_sim
            pair_threshold = SIM_THRESHOLD
        if effective_sim >= pair_threshold:
            pair_key = "|||".join(sorted([seed_domain, corpus_domain]))
            if pair_key not in edges:
                edges[pair_key] = {
                    'score':               round(raw_sim, 4),
                    'effective':           round(effective_sim, 4),
                    'authority_discounted': (d_auth or c_auth),
                    'syndicated':          False,
                }
                excerpts[corpus_domain] = corpus[j - 1]['text'][:400].strip()
                flagged_pairs.append((corpus_domain, effective_sim))

    max_sim = max((float(sim_matrix[0][j]) for j in range(1, len(all_domains))), default=0.0)
    avg_sim = float(np.mean([float(sim_matrix[0][j]) for j in range(1, len(all_domains))]))

    print(f"  ✅ Signal 1 (corpus): {len(edges)} similar pairs found for {seed_domain}")
    return {
        'domain_scores': {
            seed_domain: {
                'avg_similarity':       round(avg_sim, 4),
                'max_similarity':       round(max_sim, 4),
                'similarity_flag':      1 if max_sim >= SIM_THRESHOLD else 0,
                'similar_domain_count': len(flagged_pairs),
            }
        },
        'edges':    edges,
        'excerpts': excerpts,
    }


def compute_signal1_similarity(domains_data: list) -> dict:
    """
    Signal 1: Content similarity using Sentence Transformers.
    Excludes subdomain-to-parent comparisons to avoid false positives.
    """
    print("  📐 Computing Signal 1 (content similarity)...")

    if len(domains_data) < 2:
        # Single domain — compare against fallback corpus to find similar domains
        return _compute_signal1_against_corpus(domains_data)


    model   = get_model()
    # Prefer body_text (article body only, nav/sidebar stripped) over full text.
    # Falls back to text[:1000] for records crawled before body_text was added.
    texts   = [d.get('body_text') or d.get('text', '')[:1000] for d in domains_data]
    domains = [d['domain'] for d in domains_data]
    # author lists per domain for syndication classification
    authors_by_domain = {d['domain']: d.get('article_authors', []) for d in domains_data}

    # Compute embeddings
    embeddings = model.encode(texts, show_progress_bar=False)

    # Compute pairwise cosine similarity
    sim_matrix = cosine_similarity(embeddings)

    # Build edges and per-domain scores
    edges         = {}
    domain_scores = {}
    # Map domain -> short excerpt for evidence display (body text preferred)
    excerpts      = {d['domain']: (d.get('body_text') or d.get('text', ''))[:400].strip()
                    for d in domains_data}

    for i in range(len(domains)):
        similarities = []
        flagged_pairs = []

        for j in range(len(domains)):
            if i == j:
                continue

            # Skip same-site comparisons (subdomains of same parent)
            if is_same_site(domains[i], domains[j]):
                continue

            raw_sim = float(sim_matrix[i][j])

            # ── Authority weighting + elevated threshold (Requirement 1) ──────
            # Wire services publish near-identical content by design — that is
            # syndication, not coordination. Two protections apply:
            #   (a) effective_sim is discounted (display/scoring weight)
            #   (b) the flag threshold itself is raised 20–50% so the signal
            #       is far harder to trigger for known credible outlets.
            d_i_auth = domains[i] in AUTHORITY_DOMAINS
            d_j_auth = domains[j] in AUTHORITY_DOMAINS
            if d_i_auth and d_j_auth:
                effective_sim    = raw_sim * AUTHORITY_PAIR_WEIGHT
                pair_threshold   = SIM_THRESHOLD * AUTHORITY_PAIR_THRESHOLD_MULTIPLIER
            elif d_i_auth or d_j_auth:
                effective_sim    = raw_sim * AUTHORITY_MIXED_WEIGHT
                pair_threshold   = SIM_THRESHOLD * AUTHORITY_MIXED_THRESHOLD_MULTIPLIER
            else:
                effective_sim    = raw_sim
                pair_threshold   = SIM_THRESHOLD

            # ── Syndication marker suppression ───────────────────────────────
            # Explicit wire-service attribution in the text suppresses the pair
            # entirely — score is zeroed regardless of numeric similarity.
            exc_i = excerpts.get(domains[i], '').lower()
            exc_j = excerpts.get(domains[j], '').lower()
            is_syndicated = any(
                marker in exc_i or marker in exc_j
                for marker in SYNDICATION_MARKERS
            )
            if is_syndicated:
                effective_sim = 0.0  # suppress entirely

            similarities.append(effective_sim)

            if effective_sim >= pair_threshold:
                pair_key = "|||".join(sorted([domains[i], domains[j]]))
                if pair_key not in edges:
                    sim_type = _classify_similarity(
                        raw_sim,
                        authors_by_domain.get(domains[i], []),
                        authors_by_domain.get(domains[j], []),
                        texts[i],
                        texts[j],
                    )
                    # Syndication marker text overrides classification
                    if is_syndicated:
                        sim_type = 'shared_journalistic'
                    edges[pair_key] = {
                        'score':               round(raw_sim, 4),
                        'effective':           round(effective_sim, 4),
                        'authority_discounted': (d_i_auth or d_j_auth),
                        'syndicated':          is_syndicated,
                        'similarity_type':     sim_type,
                    }
                flagged_pairs.append((domains[j], effective_sim))

        avg_sim = float(np.mean(similarities)) if similarities else 0.0
        max_sim = float(max(similarities)) if similarities else 0.0

        domain_scores[domains[i]] = {
            'avg_similarity':        round(avg_sim, 4),
            'max_similarity':        round(max_sim, 4),
            'similarity_flag':       1 if max_sim >= SIM_THRESHOLD else 0,
            'similar_domain_count':  len(flagged_pairs),
            'is_authority_domain':   1 if domains[i] in AUTHORITY_DOMAINS else 0,
        }

    flagged = sum(1 for v in domain_scores.values() if v['similarity_flag'] == 1)
    print(f"  ✅ Signal 1: {len(edges)} similar pairs found, {flagged} domains flagged")

    return {'domain_scores': domain_scores, 'edges': edges, 'excerpts': excerpts}


def compute_signal2_cadence(domains_data: list) -> dict:
    """
    Signal 2: Publishing cadence anomaly detection using Isolation Forest.
    """
    print("  ⏰ Computing Signal 2 (cadence anomaly)...")

    cadence_features = []
    domain_names     = []

    for d in domains_data:
        domain    = d['domain']
        timestamp = d.get('timestamp', '')

        try:
            dt     = datetime.strptime(timestamp[:19], '%Y-%m-%d %H:%M:%S')
            hour   = dt.hour
            minute = dt.minute
            dow    = dt.weekday()
        except:
            hour = minute = dow = 0

        text_len = len(d.get('text', ''))
        has_links = 1 if d.get('links', '') else 0

        cadence_features.append([hour, minute, dow, text_len, has_links])
        domain_names.append(domain)

    if len(cadence_features) < 3:
        result = {}
        for domain in domain_names:
            result[domain] = {
                'anomaly_score':   0.0,
                'cadence_flagged': 0,
                'burst_score':     0.0,
            }
        return result

    X   = np.array(cadence_features, dtype=float)
    contamination = bounded_contamination(len(cadence_features))
    clf = IsolationForest(n_estimators=50, contamination=contamination, random_state=42)
    clf.fit(X)
    scores      = clf.decision_function(X)
    predictions = clf.predict(X)

    result = {}
    for i, domain in enumerate(domain_names):
        result[domain] = {
            'anomaly_score':   round(float(scores[i]), 4),
            'cadence_flagged': 1 if predictions[i] == -1 else 0,
            'burst_score':     round(float(abs(scores[i])), 4),
        }

    flagged = sum(1 for v in result.values() if v['cadence_flagged'] == 1)
    print(f"  ✅ Signal 2: {flagged} domains with anomalous cadence")
    return result


def compute_signal3_whois(domains_data: list) -> dict:
    """
    Signal 3: Domain registration pattern analysis.
    Flags domains with suspicious naming patterns and TLDs.
    """
    print("  🔍 Computing Signal 3 (WHOIS enrichment + budgeted fallback)...")
    whois_service = WhoisService()

    result = {}
    for d in domains_data:
        domain = d['domain'].lower()
        parent = get_parent_domain(domain)
        whois = whois_service.get_domain_features(parent)

        result[domain] = {
            'domain_age_days':      int(whois.get('domain_age_days', -1)),
            'whois_flagged':        int(whois.get('whois_flagged', 0)),
            'registrar':            whois.get('registrar', 'unknown'),
            'registration_country': whois.get('registration_country', ''),
            'whois_source':         whois.get('source', 'heuristic'),
        }

    flagged = sum(1 for v in result.values() if v['whois_flagged'] == 1)
    budget = whois_service.budget_status()
    print(
        f"  ✅ Signal 3: {flagged} domains flagged, WHOIS budget "
        f"{budget['queries_today']}/{budget['daily_limit']} used"
    )
    return result


_CDN_ASNS = {"AS13335", "AS54113", "AS16509", "AS15169", "AS20940", "AS209242", "AS22120", "AS14618", "AS8075"}

from config.signal_config import NOISE_AUTHORS as _NOISE_AUTHORS


def compute_signal4_hosting(domains_data: list, whois_data: dict = None) -> dict:
    """
    Signal 4: IP address and ASN/hosting overlap between domains.

    False-positive guard for major CDN providers (AWS, Google, Cloudflare, Akamai):
    If the shared IP/ASN belongs to a large generic hosting provider, the pair is
    only flagged when BOTH domains were registered on the same calendar day via the
    same registrar — a near-impossible coincidence for legitimate independent outlets.
    """
    print("  🌐 Computing Signal 4 (hosting overlap)...")

    whois_data = whois_data or {}

    live_domains = [d for d in domains_data if d.get('status') == 'ok']
    if len(live_domains) < 2:
        result = {}
        for d in domains_data:
            result[d['domain']] = {
                'ip_address': '', 'asn': '', 'hosting_org': '',
                'hosting_flagged': 0, 'shared_ip_domains': [],
                'shared_asn_domains': [], 'is_cdn': False,
            }
        return {'domain_scores': result, 'host_edges': {}}

    ip_info     = {}
    ip_to_doms  = {}
    asn_to_doms = {}

    for d in live_domains:
        domain = d['domain']
        info   = resolve_ip_and_asn(domain, IPINFO_TOKEN)
        ip_info[domain] = info
        ip  = info.get('ip_address', '')
        asn = info.get('asn', '')
        if ip:
            ip_to_doms.setdefault(ip, []).append(domain)
        if asn and asn not in _CDN_ASNS:
            asn_to_doms.setdefault(asn, []).append(domain)

    result     = {}
    host_edges = {}

    def _same_day_same_registrar(dom_a: str, dom_b: str) -> bool:
        """Return True only if both domains share creation date (day) AND registrar."""
        w_a = whois_data.get(dom_a, {})
        w_b = whois_data.get(dom_b, {})
        date_a = w_a.get('created_date', '')
        date_b = w_b.get('created_date', '')
        reg_a  = w_a.get('registrar', '').lower().strip()
        reg_b  = w_b.get('registrar', '').lower().strip()
        if not date_a or not date_b:
            return False  # unknown dates → don't flag CDN pairs
        if date_a[:10] != date_b[:10]:
            return False
        if not reg_a or not reg_b or reg_a == 'unknown' or reg_b == 'unknown':
            return False
        return reg_a == reg_b

    for d in domains_data:
        domain = d['domain']
        if domain not in ip_info:
            result[domain] = {
                'ip_address': '', 'asn': '', 'hosting_org': '',
                'hosting_flagged': 0, 'shared_ip_domains': [],
                'shared_asn_domains': [], 'is_cdn': False,
            }
            continue

        info   = ip_info[domain]
        ip     = info.get('ip_address', '')
        asn    = info.get('asn', '')
        is_cdn = info.get('is_cdn', False)

        shared_ip  = [x for x in ip_to_doms.get(ip, [])  if x != domain]
        shared_asn = [x for x in asn_to_doms.get(asn, []) if x != domain]

        # For CDN-hosted domains: require same-day + same-registrar as corroboration.
        # For non-CDN shared IPs: flag as before (exact IP match is suspicious regardless).
        if is_cdn:
            # CDN shared-IP: only flag if same-day/same-registrar confirmed
            cdn_confirmed_ip  = [x for x in shared_ip  if _same_day_same_registrar(domain, x)]
            cdn_confirmed_asn = [x for x in shared_asn if _same_day_same_registrar(domain, x)]
            flagged = 1 if (cdn_confirmed_ip or cdn_confirmed_asn) else 0
            flagging_ip  = cdn_confirmed_ip
            flagging_asn = cdn_confirmed_asn
        else:
            flagged      = 1 if (shared_ip or shared_asn) else 0
            flagging_ip  = shared_ip
            flagging_asn = shared_asn

        for other in flagging_ip:
            key = "|||".join(sorted([domain, other]))
            host_edges[key] = "SAME_IP"
        for other in flagging_asn:
            key = "|||".join(sorted([domain, other]))
            if key not in host_edges:
                host_edges[key] = "SAME_ASN"

        result[domain] = {
            'ip_address':         ip,
            'asn':                asn,
            'hosting_org':        info.get('hosting_org', ''),
            'hosting_flagged':    flagged,
            'shared_ip_domains':  shared_ip,
            'shared_asn_domains': shared_asn,
            'is_cdn':             is_cdn,
        }

    flagged_count = sum(1 for v in result.values() if v['hosting_flagged'])
    print(f"  ✅ Signal 4: {flagged_count} domains flagged for hosting overlap, {len(host_edges)} host edges")
    return {'domain_scores': result, 'host_edges': host_edges}


def compute_signal5_link_network(domains_data: list) -> dict:
    """Signal 5: Mutual linking and insular network detection."""
    print("  🔗 Computing Signal 5 (link network)...")

    analyzed_set = {d['domain'] for d in domains_data}

    outlinks = {}
    for d in domains_data:
        raw   = d.get('links', '') or ''
        links = {l.strip() for l in raw.split('|') if l.strip()}
        outlinks[d['domain']] = links

    result       = {}
    mutual_edges = {}

    for d in domains_data:
        domain = d['domain']
        my_links = outlinks.get(domain, set())

        mutual = [
            other for other in analyzed_set
            if other != domain and domain in outlinks.get(other, set()) and other in my_links
        ]

        overlap = my_links & analyzed_set - {domain}
        insular_score = len(overlap) / max(len(my_links), 1) if len(my_links) >= MIN_LINKS_FOR_INSULAR else 0.0

        insular_flag = (
            insular_score >= INSULAR_SCORE_THRESHOLD
            and len(my_links) >= MIN_LINKS_FOR_INSULAR
            and len(analyzed_set) >= MIN_DOMAINS_FOR_INSULAR
        )

        flagged = 1 if (mutual or insular_flag) else 0

        for other in mutual:
            key = "|||".join(sorted([domain, other]))
            mutual_edges[key] = True

        result[domain] = {
            'mutual_link_count':   len(mutual),
            'insular_score':       round(insular_score, 4),
            'link_network_flagged': flagged,
        }

    flagged_count = sum(1 for v in result.values() if v['link_network_flagged'])
    print(f"  ✅ Signal 5: {flagged_count} domains flagged, {len(mutual_edges)} mutual link pairs")
    return {'domain_scores': result, 'mutual_link_edges': mutual_edges}


def compute_signal6_wayback(domains_data: list) -> dict:
    """Signal 6: Wayback Machine history — new or spiking sites."""
    print("  📚 Computing Signal 6 (Wayback Machine history)...")

    result = {}
    for d in domains_data:
        domain = d['domain']
        if d.get('status') == 'fallback':
            result[domain] = {
                'wayback_snapshot_count': 0, 'wayback_first_seen': '',
                'wayback_recent_count': 0,  'wayback_age_days': -1,
                'wayback_flagged': 0, 'wayback_flag_reason': '',
            }
            continue

        wb = get_wayback_data(domain)
        count  = wb.get('wayback_snapshot_count', 0)
        recent = wb.get('wayback_recent_count', 0)

        new_site_flag = (not wb.get('wayback_error') and count < WAYBACK_MIN_SNAPSHOTS)
        spike_flag    = (
            count >= WAYBACK_MIN_SNAPSHOTS and recent >= 2
            and (recent / max(count, 1)) >= WAYBACK_SPIKE_RATIO
        )

        if new_site_flag:
            reason = 'new_site'
        elif spike_flag:
            reason = 'recent_spike'
        else:
            reason = ''

        result[domain] = {
            'wayback_snapshot_count': count,
            'wayback_first_seen':     wb.get('wayback_first_seen', ''),
            'wayback_recent_count':   recent,
            'wayback_age_days':       wb.get('wayback_age_days', -1),
            'wayback_flagged':        1 if (new_site_flag or spike_flag) else 0,
            'wayback_flag_reason':    reason,
        }

    flagged_count = sum(1 for v in result.values() if v['wayback_flagged'])
    print(f"  ✅ Signal 6: {flagged_count} domains flagged by Wayback history")
    return result


def compute_signal7_authors(domains_data: list) -> dict:
    """Signal 7: Shared author names across domains."""
    print("  ✍️  Computing Signal 7 (shared authors)...")

    author_to_domains = {}
    for d in domains_data:
        domain  = d['domain']
        authors = d.get('article_authors', []) or []
        for name in authors:
            norm = name.lower().strip()
            if len(norm) < 5 or norm in _NOISE_AUTHORS:
                continue
            author_to_domains.setdefault(norm, set()).add(domain)

    result       = {}
    author_edges = {}

    for d in domains_data:
        domain  = d['domain']
        authors = d.get('article_authors', []) or []

        shared = []
        for name in authors:
            norm = name.lower().strip()
            if norm in author_to_domains and len(author_to_domains[norm]) > 1:
                shared.append(name)
                # Build edges between all domains sharing this author
                doms = list(author_to_domains[norm])
                for i in range(len(doms)):
                    for j in range(i + 1, len(doms)):
                        key = "|||".join(sorted([doms[i], doms[j]]))
                        author_edges[key] = name

        result[domain] = {
            'article_authors':        authors,
            'shared_authors':         list(dict.fromkeys(shared)),
            'author_overlap_flagged': 1 if shared else 0,
        }

    flagged_count = sum(1 for v in result.values() if v['author_overlap_flagged'])
    print(f"  ✅ Signal 7: {flagged_count} domains share author names, {len(author_edges)} author edges")
    return {'domain_scores': result, 'author_edges': author_edges}


def analyze_domains(domains_data: list) -> dict:
    """
    Main analysis function.
    Takes crawled domain data, computes all 3 signals.
    """
    print(f"🔬 Fingerprint Analyst starting...")
    print(f"   Analyzing {len(domains_data)} domains...")
    print()

    if not domains_data:
        return {}

    sig1 = compute_signal1_similarity(domains_data)
    sig2 = compute_signal2_cadence(domains_data)
    sig3 = compute_signal3_whois(domains_data)
    # Pass WHOIS data to Signal 4 so it can gate CDN pairs on same-day/same-registrar
    sig4 = compute_signal4_hosting(domains_data, whois_data=sig3)
    sig5 = compute_signal5_link_network(domains_data)
    sig6 = compute_signal6_wayback(domains_data)
    sig7 = compute_signal7_authors(domains_data)

    features = {}
    for d in domains_data:
        domain = d['domain']

        s1 = sig1.get('domain_scores', {}).get(domain, {})
        s2 = sig2.get(domain, {})
        s3 = sig3.get(domain, {})
        s4 = sig4.get('domain_scores', {}).get(domain, {})
        s5 = sig5.get('domain_scores', {}).get(domain, {})
        s6 = sig6.get(domain, {})
        s7 = sig7.get('domain_scores', {}).get(domain, {})

        similarity_flag        = s1.get('similarity_flag', 0)
        cadence_flagged        = s2.get('cadence_flagged', 0)
        whois_flagged          = s3.get('whois_flagged', 0)
        hosting_flagged        = s4.get('hosting_flagged', 0)
        link_network_flagged   = s5.get('link_network_flagged', 0)
        wayback_flagged        = s6.get('wayback_flagged', 0)
        author_overlap_flagged = s7.get('author_overlap_flagged', 0)

        signals_triggered = (
            similarity_flag + cadence_flagged + whois_flagged +
            hosting_flagged + link_network_flagged +
            wayback_flagged + author_overlap_flagged
        )

        # Signal 3 age fallback: if WHOIS returned -1 (unknown), use Wayback first-seen
        # as a lower-bound age estimate. This prevents established domains from showing "-1d".
        whois_age = int(s3.get('domain_age_days', -1))
        wb_age    = int(s6.get('wayback_age_days', -1))
        effective_age      = whois_age if whois_age >= 0 else wb_age
        age_source         = 'whois' if whois_age >= 0 else ('wayback_estimate' if wb_age >= 0 else 'unknown')

        features[domain] = {
            # Signal 1
            'avg_similarity':   s1.get('avg_similarity', 0.0),
            'max_similarity':   s1.get('max_similarity', 0.0),
            'similarity_flag':  similarity_flag,
            # Signal 2
            'anomaly_score':    s2.get('anomaly_score', 0.0),
            'cadence_flagged':  cadence_flagged,
            'burst_score':      s2.get('burst_score', 0.0),
            'hour_variance':    0.0,
            # Signal 3
            'domain_age_days':  effective_age,
            'domain_age_source': age_source,
            'registrar':        s3.get('registrar', 'unknown'),
            'whois_flagged':    whois_flagged,
            # Signal 4
            'ip_address':          s4.get('ip_address', ''),
            'asn':                 s4.get('asn', ''),
            'hosting_org':         s4.get('hosting_org', ''),
            'hosting_flagged':     hosting_flagged,
            'shared_ip_domains':   json.dumps(s4.get('shared_ip_domains', [])),
            'shared_asn_domains':  json.dumps(s4.get('shared_asn_domains', [])),
            # Signal 5
            'mutual_link_count':    s5.get('mutual_link_count', 0),
            'insular_score':        s5.get('insular_score', 0.0),
            'link_network_flagged': link_network_flagged,
            # Signal 6
            'wayback_snapshot_count': s6.get('wayback_snapshot_count', 0),
            'wayback_first_seen':     s6.get('wayback_first_seen', ''),
            'wayback_recent_count':   s6.get('wayback_recent_count', 0),
            'wayback_age_days':       s6.get('wayback_age_days', -1),
            'wayback_flagged':        wayback_flagged,
            'wayback_flag_reason':    s6.get('wayback_flag_reason', ''),
            # Signal 7
            'article_authors':        json.dumps(s7.get('article_authors', [])),
            'shared_authors':         json.dumps(s7.get('shared_authors', [])),
            'author_overlap_flagged': author_overlap_flagged,
            # Summary
            'signals_triggered': signals_triggered,
        }

    print()
    print(f"✅ Fingerprint Analyst complete!")
    print(f"   Features computed for {len(features)} domains")

    return {
        'features':      features,
        'sim_edges':     sig1.get('edges', {}),
        'excerpts':      sig1.get('excerpts', {}),
        'host_edges':    sig4.get('host_edges', {}),
        'author_edges':  sig7.get('author_edges', {}),
    }


def analyze_domains_tool(domains_json: str) -> str:
    """CrewAI tool wrapper."""
    domains_data = json.loads(domains_json)
    result = analyze_domains(domains_data)
    result['sim_edges'] = {
        f"{k[0]}|||{k[1]}": v
        for k, v in result.get('sim_edges', {}).items()
    }
    return json.dumps(result)


if __name__ == "__main__":
    from agents.crawler_agent import crawl_domains

    print("Testing Fingerprint Analyst Agent...")
    test_data = crawl_domains(["example.com", "wikipedia.org"])
    result = analyze_domains(test_data)

    print("\nSample features:")
    for domain, feats in list(result['features'].items())[:2]:
        print(f"\n  {domain}:")
        for k, v in feats.items():
            print(f"    {k}: {v}")
    print("\n🎉 Fingerprint Analyst Agent test passed!")