"""Feasibility check for an 'unreachable-answer prior' (no training, no model).

PHOT scores every entity more than `layers` hops from the head (or disconnected) at -1e4, so they tie below every
reachable entity. For the test queries whose answer is unreachable, this compares the answer's filtered rank under:
  now        : as PHOT ranks it: after every reachable candidate, mid-tie among the unreachable ones
  floor+prior: unreachable entities ordered by a prior, but still below every reachable candidate
               (the simplest fix: replace the constant -1e4 by -1e4 + prior)
  prior only : the prior over ALL candidates, ignoring reachability: how informative the prior is on its own, and
               roughly what a model could gain if the prior were allowed to beat weak reachable candidates
Priors (computed on the observed inference graph; test answers are not in it):
  type   : number of observed facts (x, q, v), i.e. how often v is already a target of the query relation
  degree : number of facts incident to v (popularity)
  type+deg: type, ties broken by degree
Reported: MRR / H@10 of the unreachable queries under each, and the change in overall MRR that would follow.

usage: python prior_check.py [split=test_merged] [layers=5]
"""
import sys

import numpy as np

from diag import GRAIL, bfs_hops
from photformer import InductiveSplit, split_queries, tie_rank


def priors(G, n_rel):
    deg = np.bincount(G.src_np, minlength=G.n_ent).astype(float)
    typ = np.zeros((2 * n_rel, G.n_ent))
    np.add.at(typ, (G.rel_np, G.dst_np), 1.0)          # typ[q, v] = #facts (x, q, v), inverse relations included
    return deg, typ


def check(data, split, layers):
    G, Q, filt = split_queries(data, split)
    hops = bfs_hops(G, Q[:, 0])
    deg, typ = priors(G, data.n_rel)
    rows = []
    for h, q, t in Q:
        h, q, t = int(h), int(q), int(t)
        d = hops[h]
        if 0 <= d[t] <= layers:
            continue
        cand = np.ones(G.n_ent, bool)
        cand[[o for o in filt[(h, q)] if o != t]] = False   # filtered setting: other true answers removed
        reach = (d >= 0) & (d <= layers)
        scores = {"type": typ[q], "degree": deg, "type+deg": typ[q] + deg / (deg.max() + 1)}
        r = {"now": tie_rank(np.where(reach, 1.0, 0.0)[cand], int(cand[:t].sum()))}
        for name, p in scores.items():
            p = p / (p.max() + 1)                             # in [0, 1): stays below any reachable bonus of 1
            r[f"floor+{name}"] = tie_rank(np.where(reach, 2.0, p)[cand], int(cand[:t].sum()))
            r[f"only {name}"] = tie_rank(p[cand], int(cand[:t].sum()))
        r["n_reach"] = int((reach & cand).sum()); r["n_cand"] = int(cand.sum())
        rows.append(r)
    return rows, len(Q)


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test_merged"
    layers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    for ds in ["WN18RR_v1", "fb237_v1", "nell_v1"]:
        data = InductiveSplit(GRAIL, ds)
        rows, n = check(data, split, layers)
        print(f"\n{ds} {split}: {len(rows)} of {n} queries ({len(rows) / n:.1%}) have an unreachable answer")
        if not rows:
            continue
        nr = np.array([r["n_reach"] for r in rows]); nc = np.array([r["n_cand"] for r in rows])
        print(f"  reachable candidates ahead of them: median {np.median(nr):.0f} of {np.median(nc):.0f}")
        print(f"  {'ranking':>18s}   MRR    H@10   overall MRR change")
        base = np.array([r["now"] for r in rows])
        for k in [k for k in rows[0] if k not in ("n_reach", "n_cand")]:
            rk = np.array([r[k] for r in rows])
            gain = ((1 / rk) - (1 / base)).sum() / n
            print(f"  {k:>18s}  {np.mean(1 / rk):.3f}  {np.mean(rk <= 10):.3f}   {gain:+.4f}")
