"""Graph construction, and sanity checks of the real GraIL v1 splits (skipped if data/grail_repo is missing)."""
import numpy as np
import pytest

from conftest import GRAIL, grail_available, random_facts
from photformer import Graph, InductiveSplit


def test_graph_augmentation_and_csr():
    facts = random_facts(20, 3, 40, 0)
    G = Graph(facts, 20, 3)
    assert G.E == 2 * G.F == 80
    for i, (h, r, t) in enumerate(facts):                     # edge i + F is the inverse of base fact i
        assert (G.src_np[i + G.F], G.rel_np[i + G.F], G.dst_np[i + G.F]) == (t, r + 3, h)
    indptr, nbr, perm = G.csr
    assert indptr[-1] == G.E
    for v in range(20):
        row = nbr[indptr[v]:indptr[v + 1]]
        assert np.all(np.diff(row) >= 0)                     # rows sorted (needed by ph._adjacent)
        assert np.all(G.src_np[perm[indptr[v]:indptr[v + 1]].numpy()] == v)
        np.testing.assert_array_equal(G.dst_np[perm[indptr[v]:indptr[v + 1]].numpy()], row)


def _names(path):
    ents, rels, trip = set(), set(), set()
    for line in open(path):
        p = line.strip().split("\t")
        if len(p) == 3:
            ents |= {p[0], p[2]}; rels.add(p[1]); trip.add(tuple(p))
    return ents, rels, trip


@pytest.mark.parametrize("name", ["fb237_v1", "WN18RR_v1", "nell_v1"])
def test_grail_split_is_inductive(name):
    if not grail_available(name):
        pytest.skip("GraIL data not downloaded")
    e_tr, r_tr, t_tr = _names(f"{GRAIL}/{name}/train.txt")
    e_ig, r_ig, t_ig = _names(f"{GRAIL}/{name}_ind/train.txt")
    _, r_te, t_te = _names(f"{GRAIL}/{name}_ind/test.txt")
    _, _, t_iv = _names(f"{GRAIL}/{name}_ind/valid.txt")
    assert not (e_tr & e_ig), "inference-graph entities must be unseen in training"
    assert not (t_te & t_ig) and not (t_iv & t_ig), "test/valid answers must not be in the observed inference graph"
    assert r_te <= r_tr, "test relations must be in the training vocabulary (relation ids are learned)"
    d = InductiveSplit(GRAIL, name)
    assert d.G_test.n_ent == len(e_ig | _names(f"{GRAIL}/{name}_ind/test.txt")[0]
                                 | _names(f"{GRAIL}/{name}_ind/valid.txt")[0])
