"""Metrics, filtered ranking, gradients, and a short end-to-end training run on a synthetic split."""
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from photformer import VARIANTS, PHOTformer, evaluate, fit, query_masks, reciprocal, set_seed, tie_rank, train_epoch


# ---- ranking ----------------------------------------------------------------------------------------------------
def test_tie_rank():
    assert tie_rank(np.array([3.0, 1.0, 2.0]), 0) == 1.0
    assert tie_rank(np.array([3.0, 1.0, 2.0]), 1) == 3.0
    assert tie_rank(np.array([1.0, 1.0, 1.0, 0.0]), 0) == 2.0          # 3-way tie for first -> mean rank 2
    assert tie_rank(np.array([5.0, 1.0, 1.0]), 2) == 2.5


class FakeModel:
    """scores from a fixed table: scores[(h, q)] -> (N,)"""
    reranker = False

    def __init__(self, table, n):
        self.table, self.n = table, n

    def eval(self): pass

    def train(self): pass

    def encode(self, G, h, q, *a):
        s = torch.stack([torch.as_tensor(self.table[(int(a_), int(b_))], dtype=torch.float) for a_, b_ in zip(h, q)])
        return s, None, None, None


def test_filtered_ranking():
    N, R = 6, 1
    test = np.array([[0, 0, 1]])
    known = np.array([[0, 0, 2]])                                       # another true answer of (0, r0, ?)
    filt = {}
    for h, r, t in np.concatenate([test, known]):
        filt.setdefault((h, r), set()).add(t); filt.setdefault((t, r + R), set()).add(h)
    data = types.SimpleNamespace(G_test=types.SimpleNamespace(n_ent=N), ind_test=test, n_rel=R,
                                 filt_test=__import__("collections").defaultdict(set, filt))
    table = {(0, 0): [0, 5, 9, 7, 1, 1],    # gold 1 scores 5; 2 (filtered) 9; 3 (not filtered) 7 -> rank 2
             (1, 1): [4, 0, 0, 0, 0, 0]}    # reciprocal (1, r0^-1, ?) gold 0 -> rank 1
    out = evaluate(FakeModel(table, N), data, "test")["final"]["full"]
    assert out["MRR"] == pytest.approx((1 / 2 + 1) / 2)
    assert out["H1"] == pytest.approx(0.5) and out["H3"] == 1.0


def test_filter_contains_every_known_answer(toy_split):
    d = toy_split
    for arr, filt in ((d.train, d.filt_train), (d.valid, d.filt_train),
                      (d.ind_graph, d.filt_test), (d.ind_test, d.filt_test)):
        for h, r, t in arr:
            assert t in filt[(h, r)] and h in filt[(t, r + d.n_rel)]


def test_query_masks_cover_both_directions(toy_split):
    G, R = toy_split.G_train, toy_split.n_rel
    Q = reciprocal(toy_split.train[:5], R)
    m, banned = query_masks(G, Q, R)
    for i, (h, r, t) in enumerate(Q):
        rinv = (r + R) % (2 * R)
        assert banned[i] == {G.eid[(h, r, t)], G.eid[(t, rinv, h)]}
        assert (m[i] == 0).sum() == 2


# ---- gradients --------------------------------------------------------------------------------------------------
def test_gradients_reach_ph_and_ot(toy_split):
    set_seed(0)
    d = toy_split
    model = PHOTformer(d.n_rel, **dict(VARIANTS["PHOT"], d=8))
    Q = reciprocal(d.train[:8], d.n_rel)
    h, q, t = (torch.as_tensor(Q[:, i]) for i in range(3))
    em, banned = query_masks(d.G_train, Q, d.n_rel)
    s1 = model.encode(d.G_train, h, q, em, banned)[0]
    ok = model.eenc.reach[torch.arange(len(t)), t]
    F.cross_entropy(s1[ok], t[ok]).backward()
    for name in ("filt.weight", "ph_enc.0.weight", "scan_in.weight", "ot_w", "fact.0.weight", "link_enc.0.weight",
                 "rel_id_emb", "renc.type_emb"):
        g = dict(model.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, name


# ---- training -----------------------------------------------------------------------------------------------------
def test_train_epoch_finite_and_decreasing(toy_split):
    set_seed(0)
    model = PHOTformer(toy_split.n_rel, **dict(VARIANTS["PHOT"], d=8))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = [train_epoch(model, toy_split, opt, batch=16, rng=np.random.default_rng(0), max_steps=10) for _ in range(5)]
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]


def test_fit_uses_variant_schedule(toy_split):
    model = PHOTformer(toy_split.n_rel, **dict(VARIANTS["PHOT"], d=8, epochs=2, max_steps=2))
    out = fit(model, toy_split, verbose=False)
    assert len(out["hist"]) == 2
    ev = evaluate(model, toy_split, "test_merged")
    assert 0 < ev["final"]["full"]["MRR"] <= 1 and ev["n"] == 2 * (len(toy_split.ind_valid) + len(toy_split.ind_test))
