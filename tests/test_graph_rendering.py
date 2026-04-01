"""
tests/test_graph_rendering.py
Tests for network graph edge rendering logic.

The bug: pos.get() returns (0.0, 0.0) for the center (seed) node.
`if s and t:` evaluates (0.0, 0.0) as falsy → edge to/from center node is never drawn.
Fix: use `if s is not None and t is not None:`

Run with:
    python -m pytest tests/test_graph_rendering.py -v
"""

import math
import pytest


def build_positions(seed_nodes, neighbor_nodes):
    """Mirrors the position logic in app.py."""
    pos = {}
    if seed_nodes:
        pos[seed_nodes[0]["id"]] = (0.0, 0.0)
    nb_count = len(neighbor_nodes)
    for idx, nd in enumerate(neighbor_nodes):
        angle = 2 * math.pi * idx / max(nb_count, 1)
        pos[nd["id"]] = (math.cos(angle), math.sin(angle))
    return pos


def draw_edges_buggy(edges, pos):
    """The old broken version: `if s and t` — drops edges when pos.get returns None."""
    ex, ey = [], []
    for e in edges:
        s, t = pos.get(e["source"]), pos.get(e["target"])
        if s and t:
            ex += [s[0], t[0], None]
            ey += [s[1], t[1], None]
    return ex, ey


def build_edges_buggy(records):
    """Old main.py: drops edge if similarity IS None."""
    edges = []
    for sd, nb, sim in records:
        if sim is not None:
            edges.append({"source": sd, "target": nb, "similarity": round(float(sim), 4)})
    return edges


def build_edges_fixed(records):
    """Fixed main.py: always adds edge, defaults similarity to 0.0."""
    edges = []
    for sd, nb, sim in records:
        edges.append({
            "source": sd,
            "target": nb,
            "similarity": round(float(sim), 4) if sim is not None else 0.0,
        })
    return edges


def draw_edges_fixed(edges, pos):
    """The fixed version: `if s is not None and t is not None`."""
    ex, ey = [], []
    for e in edges:
        s, t = pos.get(e["source"]), pos.get(e["target"])
        if s is not None and t is not None:
            ex += [s[0], t[0], None]
            ey += [s[1], t[1], None]
    return ex, ey


class TestGraphEdgeRendering:

    def setup_method(self):
        self.seed = [{"id": "infowars.com", "domain": "infowars.com", "is_seed": True, "verdict": "REVIEW"}]
        self.neighbors = [
            {"id": "banned.video",   "domain": "banned.video",   "is_seed": False, "verdict": "REVIEW"},
            {"id": "breitbart.com",  "domain": "breitbart.com",  "is_seed": False, "verdict": "ORGANIC"},
        ]
        self.pos = build_positions(self.seed, self.neighbors)
        self.edges = [
            {"source": "infowars.com", "target": "banned.video",  "similarity": 0.52},
            {"source": "infowars.com", "target": "breitbart.com", "similarity": 0.54},
        ]

    def test_seed_is_at_origin(self):
        """Seed node should always be placed at (0.0, 0.0)."""
        assert self.pos["infowars.com"] == (0.0, 0.0)

    def test_null_similarity_buggy_drops_edge(self):
        """
        The real bug: Neo4j returns r.similarity=None for some edges.
        Old code: `if sim is not None` → edge silently dropped → node appears
        in graph with no connecting line.
        """
        records = [
            ("infowars.com", "banned.video",  None),   # similarity not stored
            ("infowars.com", "breitbart.com", 0.54),
        ]
        buggy = build_edges_buggy(records)
        fixed = build_edges_fixed(records)

        assert len(buggy) == 1, "Buggy: None similarity drops the edge"
        assert len(fixed)  == 2, "Fixed: None similarity defaults to 0.0, edge kept"

    def test_null_similarity_fixed_defaults_to_zero(self):
        """Fixed version should use 0.0 for missing similarity, not drop the edge."""
        records = [("infowars.com", "banned.video", None)]
        fixed = build_edges_fixed(records)
        assert fixed[0]["similarity"] == 0.0

    def test_fixed_version_renders_all_edges(self):
        """Fixed version should render all edges regardless of (0.0, 0.0) position."""
        ex, ey = draw_edges_fixed(self.edges, self.pos)
        # Each edge produces 3 points (x1, x2, None), so 2 edges = 6 points
        assert len(ex) == 6, f"Expected 6 edge x-points for 2 edges, got {len(ex)}"
        assert len(ey) == 6, f"Expected 6 edge y-points for 2 edges, got {len(ey)}"

    def test_fixed_version_includes_none_separators(self):
        """Plotly requires None between each edge segment to avoid connecting them."""
        ex, ey = draw_edges_fixed(self.edges, self.pos)
        none_count = sum(1 for v in ex if v is None)
        assert none_count == 2, f"Expected 2 None separators for 2 edges, got {none_count}"

    def test_missing_node_in_pos_is_skipped(self):
        """If an edge references a node not in pos, it should be silently skipped."""
        edges_with_unknown = self.edges + [
            {"source": "infowars.com", "target": "unknown-domain.com", "similarity": 0.6}
        ]
        ex, ey = draw_edges_fixed(edges_with_unknown, self.pos)
        # Still just 2 edges rendered (unknown domain is skipped)
        assert len(ex) == 6

    def test_single_neighbor_renders_one_edge(self):
        """Single neighbor graph should still draw the connecting line."""
        seed = [{"id": "example.com", "domain": "example.com", "is_seed": True}]
        neighbors = [{"id": "similar.com", "domain": "similar.com", "is_seed": False}]
        pos = build_positions(seed, neighbors)
        edges = [{"source": "example.com", "target": "similar.com", "similarity": 0.7}]

        ex, ey = draw_edges_fixed(edges, pos)
        assert len(ex) == 3  # x1, x2, None
