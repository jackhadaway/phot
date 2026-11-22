"""Error breakdown of a trained PHOT model: where is MRR lost?

Each test query (h, q, ?) is ranked exactly as evaluate() ranks it (filtered, full ranking, ties split fairly), then
grouped by
  - distance: shortest-path hops from h to the answer in the observed inference graph (undirected). The model passes
    messages for `ent_layers` (5) hops, so answers further away (or disconnected) get no evidence at all;
  - direction: tail prediction (h, r, ?) or head prediction (t, r^-1, ?);
  - relation;
  - hub: whether the head or the answer is the inference graph's highest-degree entity (NELL v1 is a star);
  - head degree (quartiles).
For each group: count, share of queries, MRR, H@1, H@10, and "lost": the group's share of the total MRR shortfall
sum(1 - 1/rank) / sum over all queries. "lost" shows where improving would pay most.

usage: python diag.py ckpt/<file>.pt [split=test]        breakdown of a trained model -> results/diag_<ckpt>_<split>.csv
       python diag.py --static <dataset> [split=test]     answer-distance distribution only (no model needed)
"""
import csv
import os
import sys
from collections import deque

import numpy as np
import torch

from photformer import DEVICE, InductiveSplit, PHOTformer, split_queries, tie_rank

GRAIL = "data/grail_repo/data"


def bfs_hops(G, sources):
    """{source: (N,) undirected hop distance in the graph, -1 = disconnected}"""
    nb = [[] for _ in range(G.n_ent)]
    for a, b in zip(G.src_np, G.dst_np):               # augmented edges: both directions already present
        nb[a].append(b)
    out = {}
    for s in set(int(x) for x in sources):
        d = np.full(G.n_ent, -1, dtype=np.int64); d[s] = 0; dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in nb[u]:
                if d[v] < 0:
                    d[v] = d[u] + 1; dq.append(v)
        out[s] = d
    return out


def query_table(data, split="test"):
    """one row per query with every grouping attribute (no model needed)"""
    G, Q, _ = split_queries(data, split)
    hops = bfs_hops(G, Q[:, 0])
    deg = np.bincount(G.src_np, minlength=G.n_ent)    # = undirected degree (every fact counted from both ends)
    hub = int(deg.argmax())
    rows = []
    for h, q, t in Q:
        h, q, t = int(h), int(q), int(t)
        dist = int(hops[h][t])
        rows.append(dict(h=h, q=q, t=t, rel=q % data.n_rel, direction="tail" if q < data.n_rel else "head",
                         dist=dist, hub="answer is hub" if t == hub else ("head is hub" if h == hub else "no hub"),
                         head_deg=int(deg[h])))
    return rows


@torch.no_grad()
def rank_queries(model, data, split="test", batch=16):
    """filtered full-ranking rank of every query, identical to evaluate()['final']['full'] for a model without the
    CBLiP reranker (PHOT has none)"""
    assert not model.reranker, "rank_queries covers the stage-1 score only"
    model.eval()
    G, Q, filt = split_queries(data, split)
    ranks = []
    for s in range(0, len(Q), batch):
        qb = Q[s:s + batch]
        h, q, t = [torch.as_tensor(qb[:, i], device=DEVICE) for i in range(3)]
        s1 = model.encode(G, h, q, None, None, np.random.default_rng(0))[0].cpu()
        for i, (hh, rr, tt) in enumerate(qb):
            others = [o for o in filt[(int(hh), int(rr))] if o != tt]
            s1[i, others] = -float("inf")
            ranks.append(tie_rank(s1[i].numpy(), int(tt)))
    model.train()
    return np.array(ranks)


def dist_bucket(d, layers=5):
    if d < 0:
        return "disconnected"
    return f"{layers + 1}+" if d > layers else str(d)


def summarise(rows, key, ranks=None):
    """[(group, n, share, MRR, H1, H10, lost)] sorted by group; without ranks only n and share"""
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(key(r), []).append(i)
    total_lost = float((1 - 1 / ranks).sum()) if ranks is not None else None
    out = []
    for g in sorted(groups, key=_order):
        idx = groups[g]; n = len(idx)
        row = dict(group=g, n=n, share=n / len(rows))
        if ranks is not None:
            r = ranks[idx]
            row.update(MRR=float((1 / r).mean()), H1=float((r <= 1).mean()), H10=float((r <= 10).mean()),
                       lost=float((1 - 1 / r).sum()) / total_lost if total_lost > 0 else 0.0)
        out.append(row)
    return out


def _order(g):
    """numeric groups first in numeric order, then the rest alphabetically"""
    s = str(g).rstrip("+")
    return (0, int(s), str(g)) if s.isdigit() else (1, 0, str(g))


def print_table(title, table):
    print(f"\n{title}")
    has = "MRR" in table[0]
    print(f"  {'group':>16s} {'n':>5s} {'share':>6s}" + ("   MRR    H@1   H@10  lost" if has else ""))
    for r in table:
        line = f"  {str(r['group']):>16s} {r['n']:5d} {r['share']:6.1%}"
        if has:
            line += f" {r['MRR']:.3f}  {r['H1']:.3f}  {r['H10']:.3f} {r['lost']:5.1%}"
        print(line)


def load(ck_path):
    ck = torch.load(ck_path, weights_only=False)
    data = InductiveSplit(GRAIL, ck["dataset"])
    model = PHOTformer(data.n_rel, **ck["cfg"]).to(DEVICE)
    model.load_state_dict(ck["state"])
    return model, data


def breakdown(rows, ranks, layers=5, top_rel=10):
    tabs = [("by answer distance (hops from head, inference graph)", summarise(rows, lambda r: dist_bucket(r["dist"], layers), ranks)),
            ("by direction", summarise(rows, lambda r: r["direction"], ranks)),
            ("by hub", summarise(rows, lambda r: r["hub"], ranks))]
    if ranks is not None:
        q = np.quantile([r["head_deg"] for r in rows], [0.25, 0.5, 0.75])
        lab = lambda d: ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"][int(np.searchsorted(q, d, side="right"))]
        tabs.append(("by head degree quartile", summarise(rows, lambda r: lab(r["head_deg"]), ranks)))
        rel = sorted(summarise(rows, lambda r: r["rel"], ranks), key=lambda r: -r["lost"])[:top_rel]
        tabs.append((f"by relation (top {top_rel} by share of lost MRR)", rel))
    return tabs


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    if sys.argv[1] == "--static":
        ds = sys.argv[2]; split = sys.argv[3] if len(sys.argv) > 3 else "test"
        data = InductiveSplit(GRAIL, ds); rows = query_table(data, split)
        print(f"{ds} {split}: {len(rows)} queries (both directions)")
        for title, tab in breakdown(rows, None):
            print_table(title, tab)
        sys.exit()
    ck = sys.argv[1]; split = sys.argv[2] if len(sys.argv) > 2 else "test"
    model, data = load(ck)
    rows = query_table(data, split); ranks = rank_queries(model, data, split)
    print(f"{data.name} {split}: {len(rows)} queries, MRR {float((1 / ranks).mean()):.4f}, H@10 {float((ranks <= 10).mean()):.4f}")
    for title, tab in breakdown(rows, ranks, model.eenc.layers.__len__()):
        print_table(title, tab)
    os.makedirs("results", exist_ok=True)
    out = f"results/diag_{os.path.splitext(os.path.basename(ck))[0]}_{split}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) + ["rank"])
        w.writeheader()
        for r, k in zip(rows, ranks):
            w.writerow(dict(r, rank=k))
    print("\nper-query ranks:", out)
