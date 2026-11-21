"""ph.py against independent brute-force references on small random graphs."""
import networkx as nx
import numpy as np
import pytest
import torch

from conftest import random_facts
from photformer import Graph
from ph import _ols, merge_tree, topo_scan, toper_query


def make_case(seed, n=18, n_rel=2, n_facts=30, drop=0.2):
    """graph + symmetric random edge mask (both directions of a base fact masked together, as in query_masks)"""
    rng = np.random.default_rng(seed)
    facts = random_facts(n, n_rel, n_facts, seed, self_loops=True)
    G = Graph(facts, n, n_rel)
    keep_base = rng.random(G.F) > drop
    em_edge = np.concatenate([keep_base, keep_base])          # augmented edge i and i + F are the two directions
    indptr, nbr, perm = G.csr
    emask = em_edge[perm.numpy()]
    adj = [set() for _ in range(n)]                           # undirected simple graph of live facts
    for i, (a, _, b) in enumerate(facts):
        if keep_base[i] and a != b:
            adj[a].add(int(b)); adj[b].add(int(a))
    f = rng.uniform(-1, 1, n)
    h = int(rng.integers(n))
    return f, h, indptr, nbr, emask, adj


def ref_merge_tree(f, h, adj):
    n = len(f)
    seq = [h] + [int(v) for v in np.argsort(-f, kind="stable") if v != h]
    key = lambda v: np.inf if v == h else f[v]                # h is the oldest vertex of all
    label = list(range(n)); members = {v: {v} for v in range(n)}; elder = {v: v for v in range(n)}
    present = set()
    join, death = [-1] * n, [-1] * n
    cyc, csize = [0] * n, [0] * n
    join[h] = h
    for v in seq:
        present.add(v)
        nb = [u for u in adj[v] if u in present]
        comps = {label[u] for u in nb}
        cyc[v] = len(nb) - len(comps)
        groups = [label[v]] + sorted(comps)
        old = max(groups, key=lambda g: key(elder[g]))
        for g in groups:
            if g != old:
                death[elder[g]] = v
        h_in = label[h] in groups
        merged = set().union(*(members[g] for g in groups))
        if h_in:
            for m in merged:
                if join[m] == -1:
                    join[m] = v
        for g in groups:
            if g != old:
                del members[g]
        members[old] = merged
        for m in merged:
            label[m] = old
        csize[v] = len(merged)
    return [np.array(a) for a in (join, death, cyc, csize)]


def maximin(f, h, adj, v):
    """max over h-v paths of the min f on the path (h excluded): brute force over thresholds"""
    if v == h:
        return np.inf
    best = -np.inf
    for thr in sorted(set(f)):
        keep = {u for u in range(len(f)) if u == h or f[u] >= thr}
        g = nx.Graph(); g.add_nodes_from(keep)
        g.add_edges_from((a, b) for a in keep for b in adj[a] if b in keep)
        if v in keep and nx.has_path(g, h, v):
            best = thr
    return best


@pytest.mark.parametrize("seed", range(25))
def test_merge_tree_matches_reference(seed):
    f, h, indptr, nbr, emask, adj = make_case(seed)
    got = merge_tree(f, h, indptr, nbr, emask)
    want = ref_merge_tree(f, h, adj)
    for name, g, w in zip(("join", "death", "cyc", "size"), got, want):
        np.testing.assert_array_equal(g, w, err_msg=name)


@pytest.mark.parametrize("seed", range(10))
def test_merge_tree_invariants(seed):
    f, h, indptr, nbr, emask, adj = make_case(seed)
    join, death, cyc, size = merge_tree(f, h, indptr, nbr, emask)
    g = nx.Graph(); g.add_nodes_from(range(len(f)))
    g.add_edges_from((a, b) for a in range(len(f)) for b in adj[a])
    b0 = nx.number_connected_components(g)
    # total H1 born = cycle rank of the final graph
    assert cyc.sum() == g.number_of_edges() - g.number_of_nodes() + b0
    # one essential H0 bar per final component; h never dies
    assert (death == -1).sum() == b0 and death[h] == -1
    # join = the weakest vertex on the best h-v path; unreachable -> -1
    comp_h = nx.node_connected_component(g, h)
    for v in range(len(f)):
        if v not in comp_h:
            assert join[v] == -1
        elif v != h:
            assert f[join[v]] == pytest.approx(maximin(f, h, adj, v))
    # bars die at a vertex no higher than where they were born
    for v in range(len(f)):
        if death[v] >= 0 and v != h:
            assert f[death[v]] <= f[v]


def test_merge_tree_ignores_masked_edges():
    # path 0 - 1 - 2; masking 1-2 must disconnect 2 from h=0
    G = Graph(np.array([[0, 0, 1], [1, 0, 2]]), 3, 1)
    indptr, nbr, perm = G.csr
    f = np.array([0.0, 0.5, 0.9])
    em = np.array([True, False, True, False])[perm.numpy()]
    join, *_ = merge_tree(f, 0, indptr, nbr, em)
    assert join[2] == -1 and join[1] == 1
    join, *_ = merge_tree(f, 0, indptr, nbr, np.ones(4, bool))
    assert join[2] == 1                                       # weakest link of 0-1-2 is vertex 1


@pytest.mark.parametrize("seed", range(10))
def test_topo_scan_matches_networkx(seed):
    f, h, indptr, nbr, emask, adj = make_case(seed, n=25, n_facts=50)
    rng = np.random.default_rng(100 + seed)
    reach = rng.random(len(f)) > 0.2
    n_thr, width = 7, 2
    out = topo_scan(f, reach, indptr, nbr, emask, n_thr, width)
    vals = np.sort(f[reach])
    thr = [vals[min(len(vals) - 1, int(j * (len(vals) - 1) / (n_thr - 1)))] for j in range(n_thr)]
    for t in range(n_thr - width):
        lo, hi = thr[t], thr[t + width]
        keep = [v for v in range(len(f)) if reach[v] and lo <= f[v] <= hi]
        g = nx.Graph(); g.add_nodes_from(keep)
        g.add_edges_from((a, b) for a in keep for b in adj[a] if b in keep)
        V, E, b0 = g.number_of_nodes(), g.number_of_edges(), nx.number_connected_components(g)
        np.testing.assert_array_equal(out[t], [b0, E - V + b0, V, E], err_msg=f"window {t}")


def test_ols_exact_line():
    xs = np.array([1.0, 2.0, 3.0, 4.0]); ys = 2.0 + 3.0 * xs
    a, b = _ols(xs, ys, 4)
    assert a == pytest.approx(2.0) and b == pytest.approx(3.0)
    assert _ols(xs, ys, 1) == (0.0, 0.0)


def test_toper_query_complete_graph():
    n = 6
    facts = np.array([[i, 0, j] for i in range(n) for j in range(i + 1, n)])
    G = Graph(facts, n, 1)
    indptr, nbr, _ = G.csr
    f = np.linspace(1, 0, n)
    a, b = toper_query(f, np.ones(n, bool), indptr, nbr, np.ones(len(nbr), bool), n)
    # |E_i| = i(i-1)/2 is convex in |V_i|, so the fitted growth must be positive
    assert b > 0 and np.isfinite(a)
