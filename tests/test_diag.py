"""diag.py: ranks agree with evaluate(), distances agree with networkx and with the model's own reach."""
import networkx as nx
import numpy as np
import pytest
import torch

from diag import bfs_hops, breakdown, dist_bucket, query_table, rank_queries, summarise
from photformer import VARIANTS, PHOTformer, evaluate, set_seed, split_queries


@pytest.fixture
def tiny(toy_split):
    set_seed(0)
    return PHOTformer(toy_split.n_rel, **dict(VARIANTS["PHOT"], d=8, ent_layers=2))


@pytest.mark.parametrize("split", ["test", "test_merged", "valid"])
def test_ranks_match_evaluate(toy_split, tiny, split):
    ranks = rank_queries(tiny, toy_split, split)
    ev = evaluate(tiny, toy_split, split)["final"]["full"]
    assert float((1 / ranks).mean()) == pytest.approx(ev["MRR"])
    assert float((ranks <= 10).mean()) == pytest.approx(ev["H10"])


def test_bfs_matches_networkx(toy_split):
    G = toy_split.G_test
    g = nx.Graph(); g.add_nodes_from(range(G.n_ent)); g.add_edges_from(zip(G.src_np, G.dst_np))
    for s, d in bfs_hops(G, range(G.n_ent)).items():
        sp = nx.single_source_shortest_path_length(g, s)
        assert all(d[v] == sp.get(v, -1) for v in range(G.n_ent))


@torch.no_grad()
def test_distance_agrees_with_model_reach(toy_split, tiny):
    """answers more than ent_layers hops away (or disconnected) are exactly those the model cannot reach"""
    L = len(tiny.eenc.layers)
    rows = query_table(toy_split, "test_merged")
    G, Q, _ = split_queries(toy_split, "test_merged")
    tiny.eval()
    tiny.encode(G, torch.as_tensor(Q[:, 0]), torch.as_tensor(Q[:, 1]), None, None, np.random.default_rng(0))
    reach = tiny.eenc.reach[torch.arange(len(Q)), torch.as_tensor(Q[:, 2])].numpy()
    far = np.array([r["dist"] < 0 or r["dist"] > L for r in rows])
    assert far.any() and (~far).any(), "toy case should contain both kinds"
    np.testing.assert_array_equal(reach, ~far)


def test_summaries_partition(toy_split, tiny):
    rows = query_table(toy_split, "test")
    ranks = rank_queries(tiny, toy_split, "test")
    for title, tab in breakdown(rows, ranks, 2)[:4]:              # the relation table is truncated to the top 10
        assert sum(r["n"] for r in tab) == len(rows), title
        assert sum(r["lost"] for r in tab) == pytest.approx(1.0), title
    assert dist_bucket(-1) == "disconnected" and dist_bucket(7, 5) == "6+" and dist_bucket(5, 5) == "5"
    assert [r["group"] for r in summarise([{"g": "6+"}, {"g": "1"}, {"g": "disconnected"}, {"g": "2"}],
                                          lambda r: r["g"])] == ["1", "2", "6+", "disconnected"]
