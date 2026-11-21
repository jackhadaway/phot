"""Query-conditioned persistent homology for link prediction (numba union-find, exact, one pass per query).

Filtration. For a query (h, q, ?) the model scores every entity with a learned function f(v) = tanh(w . x_v), where
x_v is the query-conditioned message-passing state. Entities enter in DECREASING f (superlevel filtration),
h first; an edge enters when both its ends are in. The edge of the query itself is removed (no leakage).

0-dim persistence (merge tree), per candidate v, as INDICES of the vertices whose f gives the value:
  join[v]  : vertex at whose entry v's component merges with h's component.  f(join[v]) = max over h-v paths of the
             min f on the path (the "weakest link of the best evidence path" from h to v); -1 if never connected.
  death[v] : vertex at whose entry the H0 bar born at v dies (elder rule); v itself if v is not a local maximum
             (persistence 0); -1 if the bar never dies (essential).
  cyc[v]   : number of independent 1-cycles (H1 classes of the graph) born when v enters (redundant evidence loops).
  size[v]  : size of v's component right after v enters.
Values are gathered from f, so gradients flow into the filtration (as in TOGL, Horn et al. ICLR 2022).

Topo-Scan (Uddin et al., ICLR 2026) along the same filtration: thresholds at quantiles of f over the entities
reachable from h; sliding windows [a_t, a_{t+2}]; for each window slice: b0, b1 (graph cycle rank E - V + b0), |V|, |E|.
TopoFormer only defines this for whole graphs; here it is a query-level token sequence for link prediction.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _find(parent, x):
    r = x
    while parent[r] != r:
        r = parent[r]
    while parent[x] != r:
        nx = parent[x]; parent[x] = r; x = nx
    return r


@njit(cache=True)
def merge_tree(f, h, indptr, nbr, emask):
    n = f.shape[0]
    order = np.argsort(-f)
    parent = np.arange(n); size = np.ones(n, np.int64); elder = np.arange(n)
    nxt = -np.ones(n, np.int64); tail = np.arange(n)          # member linked lists per root
    present = np.zeros(n, np.bool_); stamp = -np.ones(n, np.int64)
    join = -np.ones(n, np.int64); death = -np.ones(n, np.int64)
    cyc = np.zeros(n, np.int64); csize = np.zeros(n, np.int64)
    # h enters first
    pos = 0
    seq = np.empty(n, np.int64); seq[0] = h; pos = 1
    for i in range(n):
        if order[i] != h:
            seq[pos] = order[i]; pos += 1
    join[h] = h
    for i in range(n):
        v = seq[i]
        present[v] = True
        for k in range(indptr[v], indptr[v + 1]):
            if not emask[k]:
                continue
            u = nbr[k]
            if u == v or not present[u] or stamp[u] == v:
                continue
            stamp[u] = v
            ru = _find(parent, u); rv = _find(parent, v)
            if ru == rv:
                cyc[v] += 1
                continue
            rh = _find(parent, h)
            # elder rule: the component with the younger (lower-f) eldest vertex dies at v
            eu, ev = elder[ru], elder[rv]
            if eu == h or (ev != h and f[eu] >= f[ev]):
                young, old = rv, ru
            else:
                young, old = ru, rv
            death[elder[young]] = v
            # joining h's component: every member of the other component gets join = v
            if old == rh or young == rh:
                other = young if old == rh else old
                m = other
                while m != -1:
                    join[m] = v
                    m = nxt[m]
            # union (attach young root under old root, keep old's elder)
            parent[young] = old
            size[old] += size[young]
            nxt[tail[old]] = young; tail[old] = tail[young]
        csize[v] = size[_find(parent, v)]
    return join, death, cyc, csize


@njit(cache=True)
def topo_scan(f, reach, indptr, nbr, emask, n_thr, width):
    """Betti tokens of window slices of the filtration (only entities reachable from h)."""
    n = f.shape[0]
    idx = np.where(reach)[0]
    vals = np.sort(f[idx])
    S = n_thr - width
    out = np.zeros((S, 4))
    if len(vals) < 2:
        return out
    thr = np.empty(n_thr)
    for j in range(n_thr):
        thr[j] = vals[min(len(vals) - 1, int(j * (len(vals) - 1) / (n_thr - 1)))]
    parent = np.arange(n); stamp = -np.ones(n, np.int64)
    for t in range(S):
        lo, hi = thr[t], thr[t + width]
        for v in idx:
            parent[v] = v
        V = 0; E = 0; comps = 0
        for v in idx:
            if f[v] < lo or f[v] > hi:
                continue
            V += 1
        for v in idx:
            if f[v] < lo or f[v] > hi:
                continue
            for k in range(indptr[v], indptr[v + 1]):
                if not emask[k]:
                    continue
                u = nbr[k]
                if u <= v or not reach[u] or f[u] < lo or f[u] > hi or stamp[u] == v + t * n:
                    continue
                stamp[u] = v + t * n
                E += 1
                ru = _find(parent, u); rv = _find(parent, v)
                if ru != rv:
                    parent[ru] = rv; comps += 1
        b0 = V - comps
        out[t, 0] = b0; out[t, 1] = E - V + b0; out[t, 2] = V; out[t, 3] = E
    return out


# ---------------------------------------------------------------------------------------------------------------
# TopER (Tola et al., "TopER: Topological Embeddings in Graph Representation Learning", NeurIPS 2025) along the
# query-conditioned filtration. TopER grows a graph along a filtration, records (x_i, y_i) = (|V_i|, |E_i|) and fits
# y = a + b x by least squares: a = pivot, b = growth. Here the filtration is the learned, query-conditioned f, so the
# embedding describes how the EVIDENCE for this query densifies, not a fixed property of the graph.
#   toper_nodes : per candidate v, v's 1-hop neighbourhood grown in decreasing f (v first)  -> (a_v, b_v, n_v)
#   toper_query : the reachable subgraph grown in decreasing f, sampled at quantile thresholds -> (a, b)
# Rows of nbr must be sorted (binary search for adjacency).
# ---------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _ols(xs, ys, k):
    if k < 2:
        return 0.0, 0.0
    mx = 0.0; my = 0.0
    for i in range(k):
        mx += xs[i]; my += ys[i]
    mx /= k; my /= k
    sxx = 0.0; sxy = 0.0
    for i in range(k):
        sxx += (xs[i] - mx) ** 2; sxy += (xs[i] - mx) * (ys[i] - my)
    b = sxy / sxx if sxx > 0 else 0.0
    return my - b * mx, b


@njit(cache=True)
def _adjacent(indptr, nbr, emask, u, w):
    lo = indptr[u]; hi = indptr[u + 1]
    while lo < hi:                                   # lower bound of w in the sorted row of u
        mid = (lo + hi) // 2
        if nbr[mid] < w:
            lo = mid + 1
        else:
            hi = mid
    while lo < indptr[u + 1] and nbr[lo] == w:
        if emask[lo]:
            return True
        lo += 1
    return False


@njit(cache=True)
def toper_nodes(f, reach, indptr, nbr, emask, cap):
    n = f.shape[0]
    out = np.zeros((n, 3))
    buf = np.empty(cap, np.int64); xs = np.empty(cap + 1); ys = np.empty(cap + 1)
    stamp = -np.ones(n, np.int64)
    for v in range(n):
        if not reach[v]:
            continue
        # unique, unmasked neighbours of v
        m = 0
        cand = np.empty(indptr[v + 1] - indptr[v], np.int64)
        for k in range(indptr[v], indptr[v + 1]):
            u = nbr[k]
            if emask[k] and u != v and stamp[u] != v:
                stamp[u] = v; cand[m] = u; m += 1
        cand = cand[:m]
        if m > cap:                                       # keep the cap highest-f neighbours
            cand = cand[np.argsort(-f[cand])[:cap]]; m = cap
        else:
            cand = cand[np.argsort(-f[cand])]
        xs[0] = 1.0; ys[0] = 0.0; E = 0
        for i in range(m):
            u = cand[i]
            E += 1                                         # edge v-u
            for j in range(i):
                if _adjacent(indptr, nbr, emask, u, cand[j]):
                    E += 1
            xs[i + 1] = i + 2.0; ys[i + 1] = E
        a, b = _ols(xs, ys, m + 1)
        out[v, 0] = a; out[v, 1] = b; out[v, 2] = m
    return out


@njit(cache=True)
def toper_query(f, reach, indptr, nbr, emask, n_thr):
    n = f.shape[0]
    idx = np.where(reach)[0]
    if len(idx) < 3:
        return np.zeros(2)
    order = idx[np.argsort(-f[idx])]
    present = np.zeros(n, np.bool_); stamp = -np.ones(n, np.int64)
    K = len(order); xs = np.empty(n_thr); ys = np.empty(n_thr); t = 0; E = 0
    for i in range(K):
        v = order[i]; present[v] = True
        for k in range(indptr[v], indptr[v + 1]):
            u = nbr[k]
            if emask[k] and u != v and present[u] and stamp[u] != v:
                stamp[u] = v; E += 1
        while t < n_thr and i + 1 >= int(np.ceil((t + 1) * K / n_thr)):
            xs[t] = (i + 1.0) / K; ys[t] = E / K; t += 1   # size-normalised so it transfers across graphs
    a, b = _ols(xs, ys, t)
    return np.array([a, b])
