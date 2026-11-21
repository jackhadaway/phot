"""No-leakage: with the query edge masked during training, every component must see exactly what it would see on a
graph from which that fact was physically deleted."""
import numpy as np
import pytest
import torch

from conftest import random_facts
from photformer import VARIANTS, EntityEncoder, Graph, PHOTformer, query_masks, set_seed

N, R = 30, 3


def tiny_model(**kw):
    set_seed(0)
    cfg = dict(VARIANTS["PHOT"], d=8, max_head=10_000, **kw)    # no head-atom sampling -> deterministic
    return PHOTformer(R, **cfg).eval()


def pick_case(seed):
    facts = random_facts(N, R, 120, seed)
    i = int(np.random.default_rng(seed).integers(len(facts)))
    h, r, t = facts[i]
    G_full = Graph(facts, N, R)
    G_red = Graph(np.delete(facts, i, 0), N, R)
    Q = np.array([[h, r, t], [t, r + R, h]])                  # the query and its reciprocal
    return G_full, G_red, Q


@pytest.mark.parametrize("seed", range(5))
@torch.no_grad()
def test_components_see_no_query_edge(seed):
    model = tiny_model()
    G_full, G_red, Q = pick_case(seed)
    em, banned = query_masks(G_full, Q, R)
    assert (em == 0).sum() == 4                                # 2 queries x 2 directions
    h, q = torch.as_tensor(Q[:, 0]), torch.as_tensor(Q[:, 1])
    zr = model.relations(G_full, q); zq = zr[torch.arange(2), q]   # same relation vectors on both sides

    def run(G, m, b):
        x = model.eenc(G, h, zq, zr, m)
        reach, hop = model.eenc.reach.clone(), model.eenc.hop.clone()
        link = model.link_profile(G, h, zr, m)
        ph = model.ph_features(G, h, x, m)
        ot = model.transport_readout(G, h, q, x, zr, zq, m, b, np.random.default_rng(0))
        return dict(x=x, reach=reach, hop=hop, link=link, ph=ph[0], scan=ph[1], ot=ot)

    a, b = run(G_full, em, banned), run(G_red, None, None)
    for k in a:
        if a[k].dtype == torch.bool or a[k].dtype == torch.long:
            assert torch.equal(a[k], b[k]), k
        else:
            torch.testing.assert_close(a[k], b[k], atol=1e-5, rtol=1e-4, msg=k)


@torch.no_grad()
def test_full_score_sees_no_query_edge():
    """end-to-end s1, on a case where deleting the fact leaves the relation graph unchanged"""
    model = tiny_model()
    for seed in range(50):
        G_full, G_red, Q = pick_case(seed)
        if torch.equal(G_full.rel_adj, G_red.rel_adj):
            break
    else:
        pytest.skip("no case with identical relation graph")
    em, banned = query_masks(G_full, Q, R)
    h, q = torch.as_tensor(Q[:, 0]), torch.as_tensor(Q[:, 1])
    s_full = model.encode(G_full, h, q, em, banned)[0]
    s_red = model.encode(G_red, h, q, None, None)[0]
    torch.testing.assert_close(s_full, s_red, atol=1e-5, rtol=1e-4)


def test_relation_graph_includes_query_edge():
    """Known, minor: the relation graph (ULTRA-style, as in ULTRA itself) is built from all training facts,
    query edges included. Documented here so a change in behaviour is noticed."""
    facts = np.array([[0, 0, 1], [1, 1, 2], [3, 0, 4], [4, 2, 5]])   # fact 1 is the only r0-tail / r1-head meeting
    G_full = Graph(facts, 6, R)
    G_red = Graph(np.delete(facts, 1, 0), 6, R)
    em, _ = query_masks(G_full, facts[1:2], R)
    assert em.sum() == G_full.E - 2                                     # the edge is masked for message passing...
    assert not torch.equal(G_full.rel_adj, G_red.rel_adj)               # ...but still shapes the relation graph


@torch.no_grad()
def test_sqrt_aggregation_degree():
    """sum/sqrt(deg) (older variants, not PHOT) normalises by the FULL degree, masked query edge included."""
    G_full, G_red, Q = pick_case(0)
    em, _ = query_masks(G_full, Q, R)
    set_seed(0); enc = EntityEncoder(8, 3, agg="sqrt").eval()
    h = torch.as_tensor(Q[:, 0]); zq = torch.randn(2, 8); zr = torch.randn(2, 2 * R, 8)
    same = torch.allclose(enc(G_full, h, zq, zr, em), enc(G_red, h, zq, zr, None), atol=1e-5)
    assert not same, "sqrt aggregation no longer leaks the masked degree: update this test and HISTORY"
