"""PHOT = Persistent Homology + Optimal Transport, for inductive knowledge-graph completion (no Transformer).
Recommended variant: VARIANTS["PHOT"]. For a query (h, q, ?):
   (1) relation vectors = ULTRA-style structural encoding on the graph of relations (4 interaction types)
       + a learned identity vector per relation (rel_ids; GraIL splits share the relation vocabulary),
   (2) query-conditioned message passing over entities (NBFNet-style, DistMult messages) with MEAN aggregation
       (degree-robust: sum/sqrt(deg) collapsed on unseen hubs and over long training),
   (3) relation-agnostic link features per candidate v: #facts h-v, #distinct relations h-v, #common neighbours,
   (4) query-conditioned persistent homology (ph.py): a learned filtration f = tanh(w.x), per-candidate merge-tree
       features + a Topo-Scan token,
   (5) HOT-KG's K=0 optimal-transport readout between the head's fact atoms (pushed by the query relation) and
       every candidate's fact atoms (one atom per incident fact).
   (3) and (4) are CONCATENATED to the entity state (x || link || PE) before the scoring MLP; (5) is added to the score.
No entity parameters anywhere -> works on unseen entities.
The class is still called PHOTformer (the earlier model with a head Transformer); PHOT and HOTformer are aliases.
Ablations kept here: HOTformer v2, PHOTformer (+ Transformer), Hodge / harmonic / TopER encodings, absolute PE, TDA,
CBLiP reranker, sum / PNA aggregation, link profiles, test-time adaptation, DropEdge.
"""
import math, os, time, json, copy, random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class Graph:
    """One graph (entities local to it). facts: (N,3) base triples used for message passing."""

    def __init__(self, facts, n_ent, n_rel):
        self.n_ent, self.n_rel = n_ent, n_rel
        f = np.asarray(facts, dtype=np.int64).reshape(-1, 3)
        aug = np.concatenate([f, np.stack([f[:, 2], f[:, 1] + n_rel, f[:, 0]], 1)], 0) if len(f) else f
        self.src = torch.as_tensor(aug[:, 0]); self.rel = torch.as_tensor(aug[:, 1]); self.dst = torch.as_tensor(aug[:, 2])
        self.E = len(aug)
        self.src_np, self.rel_np, self.dst_np = aug[:, 0].copy(), aug[:, 1].copy(), aug[:, 2].copy()
        self.eid = {(int(a), int(b), int(c)): i for i, (a, b, c) in enumerate(aug)}
        # incident (outgoing, in the augmented graph) edges per entity -> fact tokens
        out = defaultdict(list)
        for i, a in enumerate(aug[:, 0]):
            out[int(a)].append(i)
        self.out = out
        self.rel_adj = self._relation_graph(aug)
        self.F = len(f)                                   # base facts are augmented edges 0..F-1
        self.base = f
        self._tri = None
        self._pe = None
        self._parts = None
        self._tda = None

    def pe(self, normalise=True):
        """structural / topological encodings (2-hop ego topology + METIS-cluster spectra, RWSE), see pe.py"""
        if self._pe is None:
            from pe import structural_pe
            P, parts = structural_pe(self.base, self.n_ent)
            self._parts = torch.as_tensor(parts, dtype=torch.long)
            if normalise:                                  # per-graph z-score: train and test graphs differ in scale
                P = (P - P.mean(0)) / P.std(0).clamp_min(1e-6)
            self._pe = P.float()
        return self._pe

    @property
    def csr(self):
        """augmented edges sorted by source: (indptr, neighbour, perm) with perm[k] = edge id of CSR slot k"""
        if getattr(self, "_csr", None) is None:
            perm = np.lexsort((self.dst_np, self.src_np))          # by source, rows sorted by neighbour
            indptr = np.zeros(self.n_ent + 1, np.int64)
            np.add.at(indptr, self.src_np + 1, 1)
            self._csr = (np.cumsum(indptr), self.dst_np[perm].astype(np.int64), torch.as_tensor(perm))
        return self._csr

    def tda(self):
        """persistent Betti numbers of every entity's 2-hop neighbourhood (ripser), per-graph z-scored, see tda.py"""
        if self._tda is None:
            from tda import tda_features
            T = tda_features(self.base, self.n_ent)
            self._tda = ((T - T.mean(0)) / T.std(0).clamp_min(1e-6)).float()
        return self._tda

    @property
    def parts(self):
        if self._parts is None:
            self.pe()
        return self._parts

    @property
    def triangles(self):
        """2-cells: triangles witnessed by a mined composition rule r1(a,b) & r2(b,c) => r3(a,c).
        Returned as (T, 3) indices of the base edges (a->b, b->c, a->c)."""
        if self._tri is None:
            self._tri = mine_triangles(self.base, self.eid)
        return self._tri

    @property
    def cells2(self):
        """2-cells of the fact complex for the full Hodge decomposition: every clique triangle, and every 2-cycle formed
        by parallel facts on the same entity pair. Returned as (C, 3) base-edge indices and (C, 3) boundary signs
        (2-cycles pad the third slot with sign 0). The harmonic space = cycles of length >= 4 not filled by these."""
        if getattr(self, "_cells2", None) is None:
            a, b = self.src_np[:self.F], self.dst_np[:self.F]
            rep_, grp = {}, {}
            for e in range(self.F):
                if a[e] == b[e]:
                    continue
                p = (min(a[e], b[e]), max(a[e], b[e]))
                grp.setdefault(p, []).append(e)
            sgn = lambda e, u: 1.0 if a[e] == u else -1.0             # orientation of edge e w.r.t. u -> v
            I, S = [], []
            for (u, v), es in grp.items():
                rep_[(u, v)] = es[0]
                for e in es[1:]:                                       # 2-cycle: u -e0-> v -e-> u
                    I.append([es[0], e, 0]); S.append([sgn(es[0], u), -sgn(e, u), 0.0])
            nb = {}
            for (u, v) in grp:
                nb.setdefault(u, set()).add(v); nb.setdefault(v, set()).add(u)
            for (u, v) in grp:                                         # triangles u < v < w
                for w in nb[u] & nb[v]:
                    if w > v:
                        e1, e2, e3 = rep_[(u, v)], rep_[(v, w)], rep_[(u, w)]
                        I.append([e1, e2, e3]); S.append([sgn(e1, u), sgn(e2, v), -sgn(e3, u)])
            I = np.array(I, np.int64).reshape(-1, 3); S = np.array(S, np.float32).reshape(-1, 3)
            self._cells2 = (torch.as_tensor(I), torch.as_tensor(S))
        return self._cells2

    def _relation_graph(self, aug):
        """4 normalised (2R x 2R) adjacencies: h2h, t2t, h2t, t2h  (ULTRA's graph of relations)."""
        R2 = 2 * self.n_rel
        Eh = torch.zeros(self.n_ent, R2); Et = torch.zeros(self.n_ent, R2)
        if len(aug):
            Eh[aug[:, 0], aug[:, 1]] = 1.0; Et[aug[:, 2], aug[:, 1]] = 1.0
        mats = [Eh.T @ Eh, Et.T @ Et, Eh.T @ Et, Et.T @ Eh]
        out = []
        for A in mats:
            A = (A > 0).float(); A.fill_diagonal_(0)
            out.append(A / A.sum(1, keepdim=True).clamp_min(1))
        return torch.stack(out)                      # (4, 2R, 2R)


def mine_triangles(base, eid, min_support=10, min_conf=0.2, top_k=16, max_tri=50000, seed=0):
    byh = defaultdict(list); pair = defaultdict(set)
    for h, r, t in base:
        byh[int(h)].append((int(r), int(t))); pair[(int(h), int(t))].add(int(r))
    sup, hit, paths = defaultdict(int), defaultdict(int), []
    for h, outs in byh.items():
        for r1, b in outs:
            for r2, c in byh.get(b, ()):
                if c == h or c == b:
                    continue
                sup[(r1, r2)] += 1
                for r3 in pair.get((h, c), ()):
                    hit[(r1, r2, r3)] += 1
                    paths.append((h, r1, b, r2, c, r3))
    rules = sorted(((hit[k] / sup[k[:2]], k) for k in hit if sup[k[:2]] >= min_support and hit[k] / sup[k[:2]] >= min_conf),
                   reverse=True)[:top_k]
    keep = {k for _, k in rules}
    tri = [(eid[(h, r1, b)], eid[(b, r2, c)], eid[(h, r3, c)]) for h, r1, b, r2, c, r3 in paths if (r1, r2, r3) in keep]
    tri = np.array(tri, dtype=np.int64).reshape(-1, 3)
    if len(tri) > max_tri:
        tri = tri[np.random.default_rng(seed).permutation(len(tri))[:max_tri]]
    return torch.as_tensor(tri)


def read_triples(path, ent, rel, add_rel=True):
    out = []
    for line in open(path):
        p = line.strip().split("\t")
        if len(p) != 3:
            continue
        h, r, t = p
        if r not in rel:
            if not add_rel:
                continue
            rel[r] = len(rel)
        for x in (h, t):
            if x not in ent:
                ent[x] = len(ent)
        out.append((ent[h], rel[r], ent[t]))
    return np.array(out, dtype=np.int64).reshape(-1, 3)


class InductiveSplit:
    """GraIL-style split: <name>/ (train graph + train/valid queries) and <name>_ind/ (unseen-entity test graph)."""

    def __init__(self, root, name):
        rel = {}
        e_tr = {}
        self.train = read_triples(f"{root}/{name}/train.txt", e_tr, rel)
        self.valid = read_triples(f"{root}/{name}/valid.txt", e_tr, rel)
        e_te = {}
        self.ind_graph = read_triples(f"{root}/{name}_ind/train.txt", e_te, rel)
        self.ind_valid = read_triples(f"{root}/{name}_ind/valid.txt", e_te, rel)
        self.ind_test = read_triples(f"{root}/{name}_ind/test.txt", e_te, rel)
        self.name, self.n_rel = name, len(rel)
        self.G_train = Graph(self.train, len(e_tr), self.n_rel)
        self.G_test = Graph(self.ind_graph, len(e_te), self.n_rel)
        self.filt_train = self._filter([self.train, self.valid])
        self.filt_test = self._filter([self.ind_graph, self.ind_valid, self.ind_test])

    def _filter(self, arrs):
        f = defaultdict(set)
        for a in arrs:
            for h, r, t in a:
                f[(int(h), int(r))].add(int(t)); f[(int(t), int(r) + self.n_rel)].add(int(h))
        return f

    def summary(self):
        return dict(name=self.name, relations=self.n_rel, train_entities=self.G_train.n_ent, train_facts=len(self.train),
                    test_entities=self.G_test.n_ent, test_graph_facts=len(self.ind_graph), test_queries=len(self.ind_test))


def reciprocal(tr, n_rel):
    tr = np.asarray(tr)
    return np.concatenate([tr, np.stack([tr[:, 2], tr[:, 1] + n_rel, tr[:, 0]], 1)], 0)


class Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(2 * d, d); self.ln = nn.LayerNorm(d)

    def forward(self, x, agg):
        return self.ln(x + F.relu(self.lin(torch.cat([x, agg], -1))))


class RelationEncoder(nn.Module):
    """Query-conditioned relation representations on the graph of relations (no relation ids needed)."""

    def __init__(self, d, layers):
        super().__init__()
        self.type_emb = nn.Parameter(torch.randn(layers, 4, d) * 0.1 + 1.0)
        self.layers = nn.ModuleList(Layer(d) for _ in range(layers))
        self.d = d
        self.rw_steps = 0

    def forward(self, adj, q):                       # adj (4, 2R, 2R), q (B,)
        B, R2 = len(q), adj.shape[1]
        z = torch.zeros(B, R2, self.d, device=q.device)
        z[torch.arange(B), q] = 1.0
        bnd = z
        for l, layer in enumerate(self.layers):
            agg = sum(torch.einsum("ij,bjd->bid", adj[k], z * self.type_emb[l, k]) for k in range(4))
            z = layer(z, agg + bnd)
        return z                                     # (B, 2R, d)


class EntityEncoder(nn.Module):
    """NBFNet-style conditional message passing from the query head, DistMult messages."""

    def __init__(self, d, layers, agg="sqrt"):
        super().__init__()
        self.rel_proj = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.layers = nn.ModuleList(Layer(d) for _ in range(layers))
        self.d = d
        self.rw_steps = 0
        self.agg = agg   # "sqrt": sum/sqrt(deg); "mean"; "pna": [sum/sqrt(deg), mean, max] -> linear (degree-robust)
        if agg == "pna":
            self.pna = nn.ModuleList(nn.Linear(3 * d, d) for _ in range(layers))

    def forward(self, G, h, zq, zr, edge_mask=None, node_feat=None):
        """h (B,), zq (B,d) query relation vector, zr (B,2R,d) relation vectors, edge_mask (B,E) or None,
        node_feat (N,d) optional structural encoding added to every entity's initial state"""
        B, N = len(h), G.n_ent
        x = torch.zeros(B, N, self.d, device=zq.device)
        x[torch.arange(B), h] = zq
        if node_feat is not None:
            x = x + node_feat.unsqueeze(0)
        bnd = x
        hop = torch.full((B, N), len(self.layers) + 1, dtype=torch.long, device=zq.device)
        hop[torch.arange(B), h] = 0
        src, rel, dst = G.src.to(zq.device), G.rel.to(zq.device), G.dst.to(zq.device)
        reach = torch.zeros(B, N, device=zq.device); reach[torch.arange(B), h] = 1.0
        em = edge_mask if edge_mask is not None else torch.ones(B, len(src), device=zq.device)
        if self.rw_steps:
            odeg = torch.zeros(B, N, device=zq.device).index_add_(1, src, em).clamp_min(1)
            p = torch.zeros(B, N, device=zq.device); p[torch.arange(B), h] = 1.0
            rw = []
            for _ in range(self.rw_steps):
                p = torch.zeros_like(p).index_add_(1, dst, (p / odeg)[:, src] * em)
                rw.append(p)
            self.rw = torch.stack(rw, -1)                # (B, N, T) landing probability after t steps from h
        self.states = []
        for l, layer in enumerate(self.layers):
            new = (reach + torch.zeros_like(reach).index_add_(1, dst, reach[:, src] * em)).clamp_max(1.0)
            hop = torch.where((new > 0) & (reach == 0), torch.full_like(hop, l + 1), hop)
            reach = new
            r = self.rel_proj[l](zr)                 # (B, 2R, d)
            msg = x[:, src] * r[:, rel]              # (B, E, d)
            if edge_mask is not None:
                msg = msg * edge_mask.unsqueeze(-1)
            agg = torch.zeros_like(x).index_add_(1, dst, msg)
            deg = torch.zeros(N, device=zq.device).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
            if self.agg == "sqrt":
                a = agg / deg.clamp_min(1).sqrt().view(1, N, 1)
            else:
                emd = em.unsqueeze(-1) if edge_mask is not None else 1.0
                dm = torch.zeros(B, N, device=zq.device).index_add_(1, dst, em).clamp_min(1).unsqueeze(-1)  # live degree
                mean = agg / dm
                if self.agg == "mean":
                    a = mean
                else:
                    mx = torch.full_like(x, -1e4).scatter_reduce(1, dst.view(1, -1, 1).expand(B, -1, self.d),
                                                                 msg - (1 - emd) * 1e4, "amax", include_self=True)
                    mx = torch.where(mx > -1e3, mx, torch.zeros_like(mx))
                    a = self.pna[l](torch.cat([agg / deg.clamp_min(1).sqrt().view(1, N, 1), mean, mx], -1))
            x = layer(x, a + bnd)
            self.states.append(x)
        self.reach = reach > 0                        # entities within `layers` hops of h
        self.hop = hop                                # shortest hop distance from h (layers+1 = unreachable)
        return x                                     # (B, N, d)


# connection types between two oriented facts (a_i, b_i), (a_j, b_j)   [CBLiP's 7 + self]
def connection_types(A, Bt, valid):
    """A, Bt (n, T) entity ids of token heads / tails; valid (n, T). Returns (n, T, T) ints in 0..8 (8 = padding)."""
    ai, aj = A.unsqueeze(2), A.unsqueeze(1); bi, bj = Bt.unsqueeze(2), Bt.unsqueeze(1)
    t = torch.full(ai.shape[:1] + (A.shape[1], A.shape[1]), 6, dtype=torch.long, device=A.device)   # 6 = not touching
    t = torch.where(ai == bj, torch.full_like(t, 3), t)        # i.head = j.tail
    t = torch.where(bi == aj, torch.full_like(t, 2), t)        # i.tail = j.head
    t = torch.where(bi == bj, torch.full_like(t, 1), t)        # same tail
    t = torch.where(ai == aj, torch.full_like(t, 0), t)        # same head
    t = torch.where((ai == bj) & (bi == aj), torch.full_like(t, 5), t)   # inverse
    t = torch.where((ai == aj) & (bi == bj), torch.full_like(t, 4), t)   # parallel
    eye = torch.eye(A.shape[1], dtype=torch.bool, device=A.device).unsqueeze(0)
    t = torch.where(eye, torch.full_like(t, 7), t)             # self
    pad = ~(valid.unsqueeze(2) & valid.unsqueeze(1))
    return torch.where(pad, torch.full_like(t, 8), t)


class BiasedAttention(nn.Module):
    def __init__(self, d, heads, n_types=9):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.kb = nn.Embedding(n_types, heads); self.vb = nn.Embedding(n_types, d)
        nn.init.zeros_(self.kb.weight); nn.init.zeros_(self.vb.weight)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, x, types, valid):
        n, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(n, T, 3, self.h, self.dk).unbind(2)
        logit = torch.einsum("nihd,njhd->nhij", q, k) / math.sqrt(self.dk) + self.kb(types).permute(0, 3, 1, 2)
        logit = logit.masked_fill(~valid.view(n, 1, 1, T), -1e9)
        att = logit.softmax(-1)                                          # (n, h, T, T)
        out = torch.einsum("nhij,njhd->nihd", att, v).reshape(n, T, d)
        vbias = self.vb(types).view(n, T, T, self.h, self.dk)           # value bias per connection type
        out = out + torch.einsum("nhij,nijhd->nihd", att, vbias).reshape(n, T, d)
        x = x + self.o(out)
        return x + self.ff(self.ln2(x))


class PHOTformer(nn.Module):
    def __init__(self, n_rel, d=32, rel_layers=4, ent_layers=5, tf_layers=2, heads=4, k=20, max_tok=12,
                 rel_encoder=True, reranker=True, path_tokens=True, transport=True, eps_learn=True,
                 ot_readout=False, head_tf=False, max_head=16, rel_ids=False,
                 hodge=False, holo=0.0, cg_iters=25, ridge=1e-2,
                 pe=False, pe_mp=False, pe_norm=True, pe_struct=True, pe_gate=False, pe_drop=0.0,
                 rel_pe=False, rw_steps=6, tda=None, tda_drop=0.2, ph=None, ph_dim=16, scan_thr=13, ph_filt="learned",
                 toper=False, toper_cap=32, drop_edge=0.0, tta=0.0, agg="sqrt", jk=False, link=False, ema=0.0, lr=5e-3, epochs=10, max_steps=120):
        super().__init__()
        self.drop_edge, self.tta, self.ema, self._ema = drop_edge, tta, ema, None
        self.train_cfg = dict(lr=lr, epochs=epochs, max_steps=max_steps)   # read by run_v4.py
        # positional / topological encodings: structural PE (pe.py) + hop distance from h
        self.pe, self.pe_mp, self.pe_norm = pe, pe_mp, pe_norm
        self.pe_struct, self.pe_gate_on, self.pe_drop = pe_struct, pe_gate, pe_drop
        if pe and pe_gate:
            self.pe_gate = nn.Parameter(torch.zeros(()))      # ReZero-style gate: starts as HOTformer v2
        self.rel_pe = rel_pe
        # Query-conditioned persistent homology (ph.py): learned filtration f = tanh(w.x); per-candidate merge-tree
        # features are CONCATENATED to x (x || PE); ph="pe+scan" adds a Topo-Scan token sequence along f.
        self.ph, self.scan_thr, self.ph_filt = ph, scan_thr, ph_filt
        extra = 0
        if ph:
            self.filt = nn.Linear(d, 1)
            self.ph_enc = nn.Sequential(nn.Linear(6, ph_dim), nn.ReLU(), nn.Linear(ph_dim, ph_dim))
            extra = ph_dim
            if ph == "pe+scan":
                S = scan_thr - 2
                self.scan_in = nn.Linear(4, d); self.scan_pos = nn.Parameter(torch.zeros(S, d))
                self.scan_tf = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True)
                extra += d
        # TopER along the same learned filtration: per-candidate (pivot, growth) of its neighbourhood + query-level TopER
        self.toper, self.toper_cap = toper, toper_cap
        if toper:
            assert ph and ph_filt == "learned", "TopER uses the PH filtration"
            self.toper_node = nn.Sequential(nn.Linear(3, ph_dim), nn.ReLU(), nn.Linear(ph_dim, ph_dim))
            self.toper_q = nn.Sequential(nn.Linear(2, ph_dim), nn.ReLU(), nn.Linear(ph_dim, ph_dim))
            extra += 2 * ph_dim
        # TDA: persistent Betti numbers of 2-hop neighbourhoods. tda="head": only the head's topology conditions the
        # query (cannot fingerprint candidates); tda="both": also a gated, dropped-out per-candidate term.
        self.tda, self.tda_drop = tda, tda_drop
        if tda:
            from tda import TDA_DIM
            self.tda_head = nn.Sequential(nn.Linear(TDA_DIM, d), nn.ReLU(), nn.Linear(d, d))
            nn.init.zeros_(self.tda_head[2].weight); nn.init.zeros_(self.tda_head[2].bias)
            if tda == "both":
                self.tda_node = nn.Sequential(nn.Linear(TDA_DIM, d), nn.ReLU(), nn.Linear(d, d))
                self.tda_gate = nn.Parameter(torch.zeros(()))
        if rel_pe:   # query-relative encoding: RW landing probs from h (masked graph) + same-METIS-cluster-as-h
            self.rel_pe_enc = nn.Sequential(nn.Linear(rw_steps + 1, d), nn.ReLU(), nn.Linear(d, d))
            nn.init.zeros_(self.rel_pe_enc[2].weight); nn.init.zeros_(self.rel_pe_enc[2].bias)
        if pe:
            from pe import PE_DIM
            self.pe_enc = nn.Sequential(nn.Linear(PE_DIM, d), nn.ReLU(), nn.Linear(d, d))
            self.hop_emb = nn.Embedding(ent_layers + 2, d)
            nn.init.zeros_(self.hop_emb.weight)
        # Hodge readout: a query-conditioned edge flow omega; its GRADIENT part (least-squares potential phi,
        # HodgeRank) scores every entity; the curl around observed rule 2-cells is penalised (holonomy loss).
        self.hodge, self.holo, self.cg_iters, self.ridge = hodge, holo, cg_iters, ridge
        if hodge:
            self.flow = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, 1))
            self.hodge_w = nn.Parameter(torch.tensor(1.0))
        if hodge == "pe":   # Hodge decomposition of the flow, per node, CONCATENATED to x (x || PE_hodge)
            self.hodge_enc = nn.Sequential(nn.Linear(7, ph_dim), nn.ReLU(), nn.Linear(ph_dim, ph_dim))
        if hodge == "harmonic":   # full 3-way decomposition: gradient + curl + HARMONIC, per node, concatenated
            self.hodge_enc = nn.Sequential(nn.Linear(10, ph_dim), nn.ReLU(), nn.Linear(ph_dim, ph_dim))
        if hodge == "harmonic_only":   # same decomposition, but ONLY the harmonic channel reaches the model
            self.hodge_enc = nn.Sequential(nn.Linear(3, ph_dim), nn.ReLU(), nn.Linear(ph_dim, ph_dim))
        self.rel_ids = rel_ids and rel_encoder   # structural relation encoding + learned identity (shared vocabulary)
        self.cfg = dict(d=d, rel_layers=rel_layers, ent_layers=ent_layers, tf_layers=tf_layers, heads=heads, k=k,
                        max_tok=max_tok, rel_encoder=rel_encoder, reranker=reranker, path_tokens=path_tokens,
                        transport=transport, ot_readout=ot_readout, head_tf=head_tf, max_head=max_head)
        self.ot_readout, self.head_tf, self.max_head = ot_readout, head_tf, max_head
        if ot_readout:
            # HOT readout: entity measure = query-conditioned embeddings of its incident facts (one atom per fact)
            self.fact = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
            self.qpush = nn.Linear(d, d)                       # T_q: diagonal push-forward from the query relation
            self.wa, self.wb = nn.Linear(d, 1), nn.Linear(d, 1)  # atom weights (head side / candidate side)
            self.ot_temp = nn.Parameter(torch.zeros(()))
            self.ot_w = nn.Parameter(torch.tensor(1.0))
            if head_tf:
                self.qtok = nn.Linear(d, d)
                self.hrole = nn.Embedding(3, d)
                self.htf = nn.ModuleList(BiasedAttention(d, heads) for _ in range(tf_layers))
        self.d, self.k, self.max_tok, self.n_rel = d, k, max_tok, n_rel
        self.rel_encoder, self.reranker, self.path_tokens, self.transport = rel_encoder, reranker, path_tokens, transport
        if rel_encoder:
            self.renc = RelationEncoder(d, rel_layers)
            if rel_ids and rel_encoder:
                self.rel_id_emb = nn.Parameter(torch.zeros(2 * n_rel, d))
        else:                                         # NBFNet-style learned relation embeddings (seen relations only)
            self.rel_emb = nn.Parameter(torch.randn(2 * n_rel, d) * 0.1)
            self.q_emb = nn.Parameter(torch.randn(2 * n_rel, d) * 0.1)
        self.eenc = EntityEncoder(d, ent_layers, agg)
        self.link = link   # direct-link relation profile: sum of query-conditioned relation vectors on edges h -> v
        if link == "count":   # relation-agnostic: #facts h-v, #distinct relations h-v, #common neighbours
            self.link_enc = nn.Sequential(nn.Linear(3, d), nn.ReLU(), nn.Linear(d, d))
        elif link == "both":
            self.link_enc = nn.Sequential(nn.Linear(d + 3, d), nn.ReLU(), nn.Linear(d, d))
        elif link:
            self.link_enc = nn.Sequential(nn.Linear(d + 1, d), nn.ReLU(), nn.Linear(d, d))
        self.jk = jk     # jumping-knowledge readout: the scorer sees every layer's state (1-hop evidence is not diluted)
        if jk:
            self.jk_proj = nn.Linear(ent_layers * d, d)
        self.eenc.rw_steps = rw_steps if rel_pe else 0
        s1_in = 2 * d + (d if link else 0) + (extra if (ph and ph_filt == "learned") else 0) + (ph_dim if hodge in ("pe", "harmonic", "harmonic_only") else 0)
        self.s1 = nn.Sequential(nn.Linear(s1_in, 2 * d), nn.ReLU(), nn.Linear(2 * d, 1))
        if ph and ph_filt == "score":   # filtration = the model's own score; PH gives a residual correction
            self.s_ph = nn.Sequential(nn.Linear(2 * d + extra, 2 * d), nn.ReLU(), nn.Linear(2 * d, 1))
            nn.init.zeros_(self.s_ph[2].weight); nn.init.zeros_(self.s_ph[2].bias)
        if reranker:
            self.role = nn.Embedding(3, d)                   # head / candidate / other
            self.role_b = nn.Embedding(3, d)
            self.target = nn.Parameter(torch.zeros(d))
            self.tok = nn.Linear(3 * d if path_tokens else d, d)
            self.tf = nn.ModuleList(BiasedAttention(d, heads) for _ in range(tf_layers))
            self.s2 = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
            if transport:
                self.qmap = nn.Linear(d, d)                  # query relation -> diagonal push-forward
                self.wpool = nn.Linear(d, 1)                 # atom weights
                self.log_temp = nn.Parameter(torch.zeros(()))
                self.w_ot = nn.Parameter(torch.tensor(0.5))

    # ---- stage 1+2 --------------------------------------------------------------------------
    def relations(self, G, q):
        if self.rel_encoder:
            z = self.renc(G.rel_adj.to(q.device), q)
            return z + self.rel_id_emb.unsqueeze(0) if self.rel_ids else z
        B = len(q)
        z = self.rel_emb.unsqueeze(0).expand(B, -1, -1)
        return z

    def encode(self, G, h, q, edge_mask=None, banned=None, rng=None):
        zr = self.relations(G, q)
        zq = zr[torch.arange(len(q)), q] if self.rel_encoder else self.q_emb[q]
        if self.tda:
            Tg = G.tda().to(zq.device)
            if self.training and self.tda_drop > 0:
                Tg = F.dropout(Tg, self.tda_drop)
            zq = zq + self.tda_head(Tg[h])              # the head's neighbourhood topology conditions the query
        P = None
        if self.pe and self.pe_struct:
            raw = G.pe(self.pe_norm).to(zq.device)
            if self.training and self.pe_drop > 0:
                raw = F.dropout(raw, self.pe_drop)
            P = self.pe_enc(raw)
            if self.pe_gate_on:
                P = self.pe_gate * P
        x = self.eenc(G, h, zq, zr, edge_mask, P if (self.pe_mp and P is not None) else None)
        if self.pe:
            x = x + self.hop_emb(self.eenc.hop) + (P.unsqueeze(0) if P is not None else 0)
        if self.tda == "both":
            x = x + self.tda_gate * self.tda_node(Tg).unsqueeze(0)
        if self.rel_pe:
            parts = G.parts.to(zq.device)
            same = (parts.unsqueeze(0) == parts[h].unsqueeze(1)).float().unsqueeze(-1)       # (B, N, 1)
            rw = torch.log(self.eenc.rw * G.n_ent + 1.0)                                     # scale-free
            x = x + self.rel_pe_enc(torch.cat([rw, same], -1))
        feats = [self.jk_proj(torch.cat(self.eenc.states, -1)) if self.jk else x, zq.unsqueeze(1).expand_as(x)]
        if self.link:
            feats.append(self.link_profile(G, h, zr, edge_mask))
        if self.ph and self.ph_filt == "learned":
            feats += self.ph_features(G, h, x, edge_mask)
        if self.hodge == "pe":
            feats.append(self.hodge_features(G, h, x, zr, edge_mask))
        elif self.hodge in ("harmonic", "harmonic_only"):
            feats.append(self.harmonic_features(G, h, x, zr, edge_mask))
        s1 = self.s1(torch.cat(feats, -1)).squeeze(-1)
        if self.ot_readout:
            s1 = s1 + self.ot_w * self.transport_readout(G, h, q, x, zr, zq, edge_mask, banned, rng)
        if self.ph and self.ph_filt == "score":
            r = self.eenc.reach.float()
            mu = (s1 * r).sum(1, keepdim=True) / r.sum(1, keepdim=True)
            sd = ((((s1 - mu) ** 2) * r).sum(1, keepdim=True).div(r.sum(1, keepdim=True)) + 1e-4).sqrt()
            f = torch.tanh((s1 - mu) / sd)                  # standardised per query: transfers across graphs
            s1 = s1 + self.s_ph(torch.cat(feats + self.ph_features(G, h, x, edge_mask, f), -1)).squeeze(-1)
        if self.hodge and self.hodge not in ("pe", "harmonic", "harmonic_only"):
            phi = self.hodge_readout(G, h, x, zr, edge_mask)
            s1 = s1 + self.hodge_w * (phi - phi[torch.arange(len(h)), h].unsqueeze(1))
        # no path within L hops = no evidence: unreachable entities rank below every reachable one (ties split fairly)
        s1 = torch.where(self.eenc.reach, s1, torch.full_like(s1, -1e4))
        return s1, x, zr, zq

    def link_profile(self, G, h, zr, edge_mask):
        """For every candidate v: the relations of the facts that link h to v directly, as query-conditioned relation
        vectors (sum) + their count. Exact 1-hop rule evidence that deep message passing tends to dilute (NELL hubs)."""
        B, dev = len(h), zr.device
        hn = h.cpu().numpy()
        bi, ei = [], []
        for b in range(B):
            es = G.out.get(int(hn[b]), ())
            bi += [b] * len(es); ei += list(es)
        bi = torch.as_tensor(bi, dtype=torch.long, device=dev); ei = torch.as_tensor(ei, dtype=torch.long, device=dev)
        w = edge_mask[bi, ei] if edge_mask is not None else torch.ones(len(ei), device=dev)
        rel, dst = G.rel.to(dev)[ei], G.dst.to(dev)[ei]
        N, d = G.n_ent, zr.shape[-1]
        P = torch.zeros(B * N, d, device=dev).index_add_(0, bi * N + dst, zr[bi, rel] * w.unsqueeze(-1)).view(B, N, d)
        C = torch.zeros(B * N, device=dev).index_add_(0, bi * N + dst, w).view(B, N, 1)
        if self.link in ("count", "both"):
            R2 = zr.shape[1]
            K = torch.zeros(B * N * R2, device=dev).index_add_(0, (bi * N + dst) * R2 + rel, w).view(B, N, R2)
            nrel = (K > 0).float().sum(-1, keepdim=True)
            # common neighbours: entities adjacent to both h and v (live edges only)
            src_all, dst_all = G.src.to(dev), G.dst.to(dev)
            em = edge_mask if edge_mask is not None else torch.ones(B, G.E, device=dev)
            nb_h = (torch.zeros(B * N, device=dev).index_add_(0, bi * N + dst, w).view(B, N) > 0).float()   # N(h)
            cn = torch.zeros(B, N, device=dev).index_add_(1, dst_all, nb_h[:, src_all] * em)              # |N(h) & N(v)| (with multiplicity)
            cnt = [torch.log1p(C), nrel, torch.log1p(cn).unsqueeze(-1)]
            return self.link_enc(torch.cat(cnt if self.link == "count" else [P] + cnt, -1))
        return self.link_enc(torch.cat([P, torch.log1p(C)], -1))

    def ph_features(self, G, h, x, edge_mask, f=None):
        from ph import merge_tree, topo_scan
        B, N, _ = x.shape
        if f is None:
            f = torch.tanh(self.filt(x).squeeze(-1))                                  # (B, N) learned filtration
        indptr, nbr, perm = G.csr
        em = (edge_mask[:, perm.to(x.device)] > 0) if edge_mask is not None else torch.ones(B, len(nbr), dtype=torch.bool)
        fn, emn, reach = f.detach().cpu().double().numpy(), em.cpu().numpy(), self.eenc.reach.cpu().numpy()
        J, D, C, Z, T = [], [], [], [], []
        for b in range(B):
            j, d_, c, z = merge_tree(fn[b], int(h[b]), indptr, nbr, emn[b])
            J.append(j); D.append(d_); C.append(c); Z.append(z)
            if self.ph == "pe+scan":
                T.append(topo_scan(fn[b], reach[b], indptr, nbr, emn[b], self.scan_thr, 2))
        dev = x.device
        J, D = torch.as_tensor(np.stack(J), device=dev), torch.as_tensor(np.stack(D), device=dev)
        C, Z = torch.as_tensor(np.stack(C), device=dev).float(), torch.as_tensor(np.stack(Z), device=dev).float()
        fj = f.gather(1, J.clamp_min(0)); joined = (J >= 0).float()
        fd = f.gather(1, D.clamp_min(0))
        pers = torch.where(D >= 0, f - fd, f + 1.0)                                  # essential bars: down to -1
        pe = torch.stack([f, torch.where(J >= 0, fj, torch.full_like(f, -2.0)), joined, pers,
                          torch.log1p(C), torch.log1p(Z) / math.log(N + 1)], -1)
        out = [self.ph_enc(pe)]
        if self.ph == "pe+scan":
            tok = torch.log1p(torch.as_tensor(np.stack(T), device=dev).float())          # (B, S, 4)
            z = self.scan_tf(self.scan_in(tok) + self.scan_pos).mean(1)                    # (B, d)
            out.append(z.unsqueeze(1).expand(B, N, -1))
        if self.toper:
            from ph import toper_nodes, toper_query
            TN, TQ = [], []
            for b in range(B):
                TN.append(toper_nodes(fn[b], reach[b], indptr, nbr, emn[b], self.toper_cap))
                TQ.append(toper_query(fn[b], reach[b], indptr, nbr, emn[b], self.scan_thr))
            tn = torch.as_tensor(np.stack(TN), device=dev).float()                       # (B, N, 3) pivot, growth, deg
            tn = torch.stack([torch.sign(tn[..., 0]) * torch.log1p(tn[..., 0].abs()), tn[..., 1] / math.sqrt(self.toper_cap),
                              torch.log1p(tn[..., 2]) / math.log(self.toper_cap + 1)], -1)
            tq = torch.as_tensor(np.stack(TQ), device=dev).float()                       # (B, 2)
            out.append(self.toper_node(tn))
            out.append(self.toper_q(tq).unsqueeze(1).expand(B, N, -1))
        return out

    def hodge_features(self, G, h, x, zr, edge_mask):
        """Hodge decomposition of the query-conditioned flow omega = B1 phi (gradient) + residual (curl + harmonic),
        summarised per entity: [phi_v - phi_h (standardised), divergence, gradient / residual / total energy of the
        incident edges, curl energy on observed 2-cells, gradient fraction]. Fully differentiable (CG via autograd)."""
        B, N, dev = x.shape[0], G.n_ent, x.device
        phi = self.hodge_readout(G, h, x, zr, edge_mask)                                   # sets self._omega
        omega = self._omega
        w = edge_mask[:, :G.F] if edge_mask is not None else torch.ones_like(omega)
        a, b = G.src[:G.F].to(dev), G.dst[:G.F].to(dev)
        sc = lambda y: torch.zeros(B, N, device=dev).index_add_(1, a, y).index_add_(1, b, y)  # edge -> both ends
        deg = sc(w) + 1.0
        g = phi[:, b] - phi[:, a]; res = omega - g
        div = (torch.zeros(B, N, device=dev).index_add_(1, b, w * omega).index_add_(1, a, -w * omega)) / deg
        eg, er = sc(w * g * g) / deg, sc(w * res * res) / deg
        T = G.triangles
        if len(T):
            T = T.to(dev)
            tw = w[:, T[:, 0]] * w[:, T[:, 1]] * w[:, T[:, 2]]
            circ2 = tw * (omega[:, T[:, 0]] + omega[:, T[:, 1]] - omega[:, T[:, 2]]).pow(2)
            va, vb, vc = a[T[:, 0]], b[T[:, 0]], b[T[:, 1]]
            cnt = torch.zeros(B, N, device=dev).index_add_(1, va, tw).index_add_(1, vb, tw).index_add_(1, vc, tw) + 1.0
            curl = torch.zeros(B, N, device=dev).index_add_(1, va, circ2).index_add_(1, vb, circ2).index_add_(1, vc, circ2) / cnt
        else:
            curl = torch.zeros(B, N, device=dev)
        r = self.eenc.reach.float()
        dphi = phi - phi[torch.arange(B, device=dev), h].unsqueeze(1)
        sd = ((dphi.pow(2) * r).sum(1, keepdim=True) / r.sum(1, keepdim=True) + 1e-4).sqrt()
        feats = torch.stack([dphi / sd, torch.tanh(div), torch.log1p(eg), torch.log1p(er), torch.log1p(eg + er),
                             torch.log1p(curl), eg / (eg + er + 1e-6)], -1)
        return self.hodge_enc(feats)

    def transport_readout(self, G, h, q, x, zr, zq, edge_mask, banned, rng):
        """K=0 transport between T_q # mu_h and mu_v for EVERY entity v, where mu_e = weighted query-conditioned
        embeddings of e's incident facts.  log sum_i sum_j a_i b_j exp(<T_q f_i, g_j> / tau)."""
        B, dev = len(h), x.device
        src, rel, dst = G.src.to(dev), G.rel.to(dev), G.dst.to(dev)
        g = self.fact(torch.cat([x[:, src], x[:, dst], zr[:, rel]], -1))          # (B, E, d) one atom per fact
        em = edge_mask if edge_mask is not None else torch.ones(B, G.E, device=dev)
        # head-side atoms: h's incident facts (minus the query's own edge during training)
        M = self.max_head
        idx = np.zeros((B, M), dtype=np.int64); val = np.zeros((B, M), dtype=bool)
        hn = h.cpu().numpy(); rng = rng or np.random.default_rng(0)
        for b in range(B):
            es = [e for e in G.out.get(int(hn[b]), ()) if not banned or e not in banned[b]]
            if len(es) > M:
                es = list(rng.choice(es, M, replace=False))
            idx[b, :len(es)] = es; val[b, :len(es)] = True
        idx, val = torch.as_tensor(idx, device=dev), torch.as_tensor(val, device=dev)
        f = g[torch.arange(B, device=dev).unsqueeze(1), idx]                           # (B, M, d)
        if self.head_tf:                                                               # connection-biased Transformer
            qt = self.qtok(zq).unsqueeze(1)                                            # query token (h, q, ?)
            toks = torch.cat([qt, f], 1)
            A = torch.cat([h.to(dev).unsqueeze(1), src[idx]], 1)
            Bt = torch.cat([torch.full((B, 1), -2, device=dev, dtype=torch.long), dst[idx]], 1)
            vv = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), val], 1)
            role = torch.where(Bt == h.to(dev).unsqueeze(1), 0, torch.where(Bt == -2, 1, 2))
            toks = (toks + self.hrole(role)) * vv.unsqueeze(-1)
            types = connection_types(A, Bt, vv)
            for layer in self.htf:
                toks = layer(toks, types, vv)
            f, val = toks, vv
        fp = f * torch.tanh(self.qpush(zq)).unsqueeze(1)                               # T_q push-forward
        la = self.wa(f).squeeze(-1).masked_fill(~val, -1e9).log_softmax(-1)            # (B, M)
        inner = torch.einsum("bmd,bed->bem", fp, g) * torch.exp(self.ot_temp) / math.sqrt(self.d)
        L = torch.logsumexp(inner + la.unsqueeze(1), -1)                               # (B, E): sum_i a_i exp(..)
        wb = self.wb(g).squeeze(-1).masked_fill(em == 0, -1e9)                         # candidate-side weights
        N = G.n_ent

        def scatter_lse(v):                                                             # logsumexp over facts of each entity
            mx = torch.full((B, N), -1e9, device=dev).scatter_reduce(1, src.expand(B, -1), v, "amax", include_self=True)
            sm = torch.zeros(B, N, device=dev).index_add_(1, src, torch.exp(v - mx[:, src]))
            return mx + torch.log(sm.clamp_min(1e-30))
        ot = scatter_lse(wb + L) - scatter_lse(wb)                                     # = log sum_j b_j sum_i a_i exp(..)
        has = torch.zeros(B, N, device=dev).index_add_(1, src, em) > 0
        return torch.where(has, ot, torch.zeros_like(ot))

    def edge_flow(self, G, x, zr):
        """antisymmetric query-conditioned 1-cochain on base edges: omega(a->b) = m(x_a, x_b, z_r) - m(x_b, x_a, z_r^-1)"""
        dev = x.device
        a, r, b = G.src[:G.F].to(dev), G.rel[:G.F].to(dev), G.dst[:G.F].to(dev)
        fwd = self.flow(torch.cat([x[:, a], x[:, b], zr[:, r]], -1)).squeeze(-1)
        bwd = self.flow(torch.cat([x[:, b], x[:, a], zr[:, r + self.n_rel]], -1)).squeeze(-1)
        return fwd - bwd                                                                   # (B, F)

    def hodge_readout(self, G, h, x, zr, edge_mask):
        """gradient component of the Hodge decomposition: phi = argmin |B1 phi - omega|^2 (+ ridge), by batched CG."""
        B, N, dev = x.shape[0], G.n_ent, x.device
        omega = self.edge_flow(G, x, zr)
        w = edge_mask[:, :G.F] if edge_mask is not None else torch.ones_like(omega)
        self._omega, self._omega_G = omega, G
        a, b = G.src[:G.F].to(dev), G.dst[:G.F].to(dev)

        def B1T(y):                                                                        # (B,F) -> (B,N)
            return torch.zeros(B, N, device=dev).index_add_(1, b, y).index_add_(1, a, -y)

        def mv(p):                                                                         # (B1^T W B1 + ridge) p
            return B1T(w * (p[:, b] - p[:, a])) + self.ridge * p
        rhs = B1T(w * omega)
        phi = torch.zeros_like(rhs); r_ = rhs; p = r_; rs = (r_ * r_).sum(1, keepdim=True)
        for _ in range(self.cg_iters):
            Ap = mv(p)
            alpha = rs / (p * Ap).sum(1, keepdim=True).clamp_min(1e-12)
            phi = phi + alpha * p; r_ = r_ - alpha * Ap
            rs_new = (r_ * r_).sum(1, keepdim=True)
            p = r_ + (rs_new / rs.clamp_min(1e-12)) * p; rs = rs_new
        return phi

    def harmonic_features(self, G, h, x, zr, edge_mask):
        """Full Hodge decomposition of the query-conditioned flow on the fact complex (cells2):
             omega = B1 phi  (gradient)  +  B2^T psi  (curl)  +  eta  (harmonic: flow around unfilled holes)
        phi and psi are least-squares solutions by batched CG (differentiable). Per entity, over its incident facts:
        [phi_v - phi_h (standardised), divergence, log-energies of gradient / curl / harmonic / total,
         gradient / curl / harmonic shares, harmonic circulation through v (signed, standardised)]."""
        B, N, dev = x.shape[0], G.n_ent, x.device
        phi = self.hodge_readout(G, h, x, zr, edge_mask)
        omega = self._omega
        w = edge_mask[:, :G.F] if edge_mask is not None else torch.ones_like(omega)
        a, b = G.src[:G.F].to(dev), G.dst[:G.F].to(dev)
        g = phi[:, b] - phi[:, a]; res = (omega - g) * w
        I, S = G.cells2
        I, S = I.to(dev), S.to(dev)
        if len(I):
            wc = w[:, I] * (S != 0) + (S == 0).float()               # (B, C, 3); a cell is open if any face is masked
            cw = wc.prod(-1)                                           # (B, C)
            Sw = S.unsqueeze(0) * cw.unsqueeze(-1)                     # masked boundary operator

            def B2(y):                                                 # (B, F) -> (B, C)
                return (y[:, I] * Sw).sum(-1)

            def B2T(p):                                                # (B, C) -> (B, F)
                return torch.zeros(B, G.F, device=dev).index_add_(1, I.reshape(-1), (p.unsqueeze(-1) * Sw).reshape(B, -1))
            rhs = B2(res); psi = torch.zeros_like(rhs); r_ = rhs; p = r_; rs = (r_ * r_).sum(1, keepdim=True)
            for _ in range(self.cg_iters):
                Ap = B2(B2T(p)) + self.ridge * p
                alpha = rs / (p * Ap).sum(1, keepdim=True).clamp_min(1e-12)
                psi = psi + alpha * p; r_ = r_ - alpha * Ap
                rs_new = (r_ * r_).sum(1, keepdim=True)
                p = r_ + (rs_new / rs.clamp_min(1e-12)) * p; rs = rs_new
            c = B2T(psi)
        else:
            c = torch.zeros_like(res)
        eta = res - c                                                  # harmonic component
        self._hodge_parts = (g * w, c, eta)
        sc = lambda y: torch.zeros(B, N, device=dev).index_add_(1, a, y).index_add_(1, b, y)
        deg = sc(w) + 1.0
        div = (torch.zeros(B, N, device=dev).index_add_(1, b, w * omega).index_add_(1, a, -w * omega)) / deg
        eg, ec, eh = sc(w * g * g) / deg, sc(c * c) / deg, sc(eta * eta) / deg
        tot = eg + ec + eh + 1e-6
        circ = sc(eta.abs()) / deg                                     # harmonic throughput at v
        r = self.eenc.reach.float()
        dphi = phi - phi[torch.arange(B, device=dev), h].unsqueeze(1)
        sd = ((dphi.pow(2) * r).sum(1, keepdim=True) / r.sum(1, keepdim=True) + 1e-4).sqrt()
        cs = ((circ.pow(2) * r).sum(1, keepdim=True) / r.sum(1, keepdim=True) + 1e-4).sqrt()
        feats = torch.stack([dphi / sd, torch.tanh(div), torch.log1p(eg), torch.log1p(ec), torch.log1p(eh),
                             torch.log1p(tot), eg / tot, ec / tot, eh / tot, circ / cs], -1)
        self._hodge_shares = (eg.mean().item(), ec.mean().item(), eh.mean().item())
        if self.hodge == "harmonic_only":                              # harmonic energy, harmonic share, throughput
            feats = feats[..., [4, 8, 9]]
        return self.hodge_enc(feats)

    def holonomy_loss(self, G, n=4096):
        """curl of omega around OBSERVED 2-cells (rule-witnessed triangles) -> 0  (corrected Cor. 4.12)"""
        T = G.triangles
        if not self.hodge or self.holo <= 0 or len(T) == 0 or getattr(self, "_omega_G", None) is not G:
            return torch.zeros((), device=next(self.parameters()).device)
        idx = T[torch.randint(len(T), (min(n, len(T)),))].to(self._omega.device)
        circ = self._omega[:, idx[:, 0]] + self._omega[:, idx[:, 1]] - self._omega[:, idx[:, 2]]
        return circ.pow(2).mean()

    # ---- stage 3: tokens for (h, q, c) ------------------------------------------------------
    def build_tokens(self, G, h, q, C, banned, rng):
        """token arrays for the B*k (head, candidate) pairs. side: 0 = candidate fact, 1 = head side, 2 = candidate side."""
        B, k = C.shape; T = 1 + 2 * self.max_tok
        A = np.full((B, k, T), -1, dtype=np.int64); Bt = np.full((B, k, T), -1, dtype=np.int64)
        R = np.zeros((B, k, T), dtype=np.int64); side = np.zeros((B, k, T), dtype=np.int64)
        valid = np.zeros((B, k, T), dtype=bool)
        hn, qn, Cn = h.cpu().numpy(), q.cpu().numpy(), C.cpu().numpy()

        def pick(e, bset):
            es = [i for i in G.out.get(e, ()) if i not in bset]
            if len(es) > self.max_tok:
                es = list(rng.choice(es, self.max_tok, replace=False))
            return np.asarray(es, dtype=np.int64)
        for b in range(B):
            hside = pick(int(hn[b]), banned[b]); nh = len(hside)
            A[b, :, 0], R[b, :, 0], Bt[b, :, 0], valid[b, :, 0] = hn[b], qn[b], Cn[b], True
            if nh:
                A[b, :, 1:1 + nh] = G.src_np[hside]; R[b, :, 1:1 + nh] = G.rel_np[hside]
                Bt[b, :, 1:1 + nh] = G.dst_np[hside]; side[b, :, 1:1 + nh] = 1; valid[b, :, 1:1 + nh] = True
            off = 1 + self.max_tok
            for j in range(k):
                cs = pick(int(Cn[b, j]), banned[b]); nc = len(cs)
                if nc:
                    A[b, j, off:off + nc] = G.src_np[cs]; R[b, j, off:off + nc] = G.rel_np[cs]
                    Bt[b, j, off:off + nc] = G.dst_np[cs]; side[b, j, off:off + nc] = 2; valid[b, j, off:off + nc] = True
        return [torch.as_tensor(a) for a in (A, R, Bt, side, valid)]

    def rerank(self, G, h, q, C, x, zr, zq, banned, rng):
        B, k = C.shape; dev = x.device
        A, R, Bt, side, valid = [t.to(dev) for t in self.build_tokens(G, h, q, C, banned, rng)]
        n, T = B * k, A.shape[-1]
        A, R, Bt, side, valid = [t.view(n, T) for t in (A, R, Bt, side, valid)]
        bidx = torch.arange(B, device=dev).repeat_interleave(k).unsqueeze(1).expand(n, T)
        hh = h.to(dev).repeat_interleave(k).unsqueeze(1); cc = C.to(dev).reshape(n, 1)
        role = lambda E: torch.where(E == hh, 0, torch.where(E == cc, 1, 2))
        Ac, Bc = A.clamp_min(0), Bt.clamp_min(0)
        zrel = zr[bidx, R]                                                   # (n, T, d)
        feats = [zrel]
        if self.path_tokens:
            feats += [x[bidx, Ac], x[bidx, Bc]]
        tok = self.tok(torch.cat(feats, -1)) + self.role(role(A)) + self.role_b(role(Bt))
        is_target = torch.zeros(n, T, 1, device=dev); is_target[:, 0] = 1.0
        tok = (tok + is_target * self.target) * valid.unsqueeze(-1)
        types = connection_types(A, Bt, valid)
        for layer in self.tf:
            tok = layer(tok, types, valid)
        s2 = self.s2(tok[:, 0]).squeeze(-1)
        if self.transport:                                                   # HOT-KG K=0 score, atoms = facts
            push = tok * torch.tanh(self.qmap(zq)).repeat_interleave(k, 0).unsqueeze(1)
            lw = self.wpool(tok).squeeze(-1)
            mh, mc = valid & (side == 1), valid & (side == 2)
            la = lw.masked_fill(~mh, -1e9).log_softmax(-1); lb = lw.masked_fill(~mc, -1e9).log_softmax(-1)
            G_ = torch.einsum("nid,njd->nij", push, tok) * torch.exp(self.log_temp) / math.sqrt(self.d)
            ot = torch.logsumexp((la.unsqueeze(2) + lb.unsqueeze(1) + G_).flatten(1), -1)
            has = mh.any(1) & mc.any(1)
            s2 = s2 + self.w_ot * torch.where(has, ot, torch.zeros_like(ot))
        return s2.view(B, k)


def query_masks(G, Q, n_rel):
    """edge mask removing each training query's own edge (both directions) from message passing."""
    m = torch.ones(len(Q), G.E)
    banned = []
    for i, (h, r, t) in enumerate(Q):
        ids = set()
        for key in ((int(h), int(r), int(t)), (int(t), int((r + n_rel) % (2 * n_rel)), int(h))):
            if key in G.eid:
                ids.add(G.eid[key])
        for e in ids:
            m[i, e] = 0.0
        banned.append(ids)
    return m, banned


def _device(model):
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return DEVICE


def train_epoch(model, data, opt, batch=16, rng=None, ls=0.0, max_steps=None):
    n_rel = data.n_rel
    Q = reciprocal(data.train, n_rel)
    perm = np.random.permutation(len(Q)); tot = 0.0; seen = 0
    if max_steps:
        perm = perm[:max_steps * batch]
    # test-time adaptation (tta > 0): a fraction of batches are self-supervised queries built from the OBSERVED facts of
    # the inference graph (each query's own edge removed). Only the inference graph's facts are used: its valid/test
    # triples never enter training, and its training filter is built from the observed facts alone.
    tta = getattr(model, "tta", 0.0)
    if tta > 0 and not hasattr(data, "filt_ind_graph"):
        data.filt_ind_graph = data._filter([data.ind_graph])
    Q_ind = reciprocal(data.ind_graph, n_rel) if tta > 0 else None
    model.train()
    dev = _device(model)
    for s in range(0, len(perm), batch):
        if tta > 0 and np.random.rand() < tta:
            G, filt = data.G_test, data.filt_ind_graph
            qb = Q_ind[np.random.choice(len(Q_ind), min(batch, len(Q_ind)), replace=False)]
        else:
            G, filt = data.G_train, data.filt_train
            qb = Q[perm[s:s + batch]]
        h, q, t = [torch.as_tensor(qb[:, i], device=dev) for i in range(3)]
        emask, banned = query_masks(G, qb, n_rel)
        if getattr(model, "drop_edge", 0) > 0:          # DropEdge: each query sees a random sub-graph
            emask = emask * (torch.rand_like(emask) > model.drop_edge).float()
        s1, x, zr, zq = model.encode(G, h, q, emask.to(dev), banned, rng)
        # filtered 1-vs-all CE: other true answers in the training graph are masked out
        fm = torch.zeros_like(s1, dtype=torch.bool)
        for i, (hh, rr, tt) in enumerate(qb):
            for o in filt[(int(hh), int(rr))]:
                if o != tt:
                    fm[i, o] = True
        s1m = s1.masked_fill(fm, -1e9)
        ok = model.eenc.reach[torch.arange(len(t)), t]           # gold reachable without its own edge
        loss = F.cross_entropy(s1m[ok], t[ok], label_smoothing=ls) if ok.any() else s1.sum() * 0
        if model.reranker:
            with torch.no_grad():
                C = s1m.topk(min(model.k, G.n_ent), 1).indices
                miss = ~(C == t.unsqueeze(1)).any(1)
                C[miss, -1] = t[miss]                                    # training only: make sure gold is present
            s2 = model.rerank(G, h, q, C, x, zr, zq, banned, rng)
            fused = s1m.gather(1, C).detach() + s2
            gold = (C == t.unsqueeze(1)).float().argmax(1)
            loss = loss + F.cross_entropy(fused, gold)
        if model.hodge and model.holo > 0:
            loss = loss + model.holo * model.holonomy_loss(G)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); tot += loss.item() * len(qb); seen += len(qb)
        ema = getattr(model, "_ema", None)
        if ema is not None:                                # exponential moving average of the weights
            with torch.no_grad():
                for k, p in model.named_parameters():
                    ema[k].mul_(model.ema).add_(p.detach(), alpha=1 - model.ema)
    return tot / seen


def tie_rank(scores, gold_idx):
    g = scores[gold_idx]
    return 1.0 + float((scores > g).sum()) + 0.5 * float((scores == g).sum() - 1)


@torch.no_grad()
def evaluate(model, data, split="test", batch=16, n_neg=50, seed=0, max_q=None):
    """Full filtered ranking (MRR, H@1/3/10) and the GraIL/CBLiP protocol (rank vs 50 random negatives, H@10)."""
    model.eval()
    if split == "test":
        G, Q, filt = data.G_test, reciprocal(data.ind_test, data.n_rel), data.filt_test
    elif split == "test_merged":        # standard GraIL protocol (NBFNet, A*Net, ULTRA...): inductive valid + test
        G, Q, filt = data.G_test, reciprocal(np.concatenate([data.ind_valid, data.ind_test]), data.n_rel), data.filt_test
    elif split == "ind_valid":          # validation triples of the inductive graph (model selection as NBFNet/ULTRA)
        G, Q, filt = data.G_test, reciprocal(data.ind_valid, data.n_rel), data.filt_test
    else:
        G, Q, filt = data.G_train, reciprocal(data.valid, data.n_rel), data.filt_train
    if max_q and len(Q) > max_q:
        Q = Q[np.random.default_rng(seed).permutation(len(Q))[:max_q]]
    rng = np.random.default_rng(seed)
    full1, full2, neg1, neg2 = [], [], [], []
    dev = _device(model)
    for s in range(0, len(Q), batch):
        qb = Q[s:s + batch]
        h, q, t = [torch.as_tensor(qb[:, i], device=dev) for i in range(3)]
        s1, x, zr, zq = model.encode(G, h, q, None, None, np.random.default_rng(0))
        s1 = s1.cpu()
        for i, (hh, rr, tt) in enumerate(qb):
            others = [o for o in filt[(int(hh), int(rr))] if o != tt]
            s1[i, others] = -float("inf")
        # ---- full ranking ----
        ranks1 = [tie_rank(s1[i].numpy(), int(t[i])) for i in range(len(qb))]
        full1 += ranks1
        # ---- 50 negatives (GraIL / CBLiP protocol) ----
        negs = []
        for i, (hh, rr, tt) in enumerate(qb):
            bad = filt[(int(hh), int(rr))] | {int(hh)}
            pool = np.array([e for e in range(G.n_ent) if e not in bad])
            # 50 distinct negatives when available; tiny graphs (e.g. NELL v1 test: 225 entities) top up with replacement
            nn_ = rng.choice(pool, size=n_neg, replace=len(pool) < n_neg)
            negs.append(np.concatenate([[int(tt)], nn_]))
        negs = torch.as_tensor(np.stack(negs))
        s1n = s1.gather(1, negs)
        neg1 += [tie_rank(s1n[i].numpy(), 0) for i in range(len(qb))]
        if model.reranker:
            # full ranking: rerank the top-k of the filtered stage-1 scores (gold is never injected)
            k = min(model.k, G.n_ent)
            C = s1.topk(k, 1).indices
            s2 = model.rerank(G, h, q, C.to(DEVICE), x, zr, zq, [set()] * len(qb), rng).cpu()
            fused = s1.gather(1, C) + s2
            for i in range(len(qb)):
                pos = (C[i] == t[i].cpu()).nonzero()
                full2.append(tie_rank(fused[i].numpy(), int(pos[0, 0])) if len(pos) else ranks1[i])
            # 50-negative protocol: rerank the top-k of the 51 candidates
            kn = min(k, negs.shape[1])
            top = s1n.topk(kn, 1).indices
            Cn = negs.gather(1, top)
            s2n = model.rerank(G, h, q, Cn.to(DEVICE), x, zr, zq, [set()] * len(qb), rng).cpu()
            fn = s1n.gather(1, top) + s2n
            for i in range(len(qb)):
                pos = (top[i] == 0).nonzero()
                neg2.append(tie_rank(fn[i].numpy(), int(pos[0, 0])) if len(pos) else neg1[len(neg2)])
    model.train()

    def summ(r):
        r = np.array(r)
        return dict(MRR=float((1 / r).mean()), H1=float((r <= 1).mean()), H3=float((r <= 3).mean()), H10=float((r <= 10).mean()))
    out = dict(n=len(Q), stage1=dict(full=summ(full1), neg50=summ(neg1)))
    if model.reranker:
        out["final"] = dict(full=summ(full2), neg50=summ(neg2))
    else:
        out["final"] = out["stage1"]
    return out


def fit(model, data, epochs=None, lr=None, batch=16, seed=0, valid_q=200, verbose=True, time_limit=None, max_steps=None,
        select="valid"):
    # None = use the variant's own schedule (model.train_cfg), e.g. 40 epochs for VARIANTS["PHOT"]
    cfg = getattr(model, "train_cfg", {})
    epochs = cfg.get("epochs", 10) if epochs is None else epochs
    lr = cfg.get("lr", 5e-3) if lr is None else lr
    max_steps = cfg.get("max_steps", 120) if max_steps is None else max_steps
    set_seed(seed); rng = np.random.default_rng(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, hist, t0 = -1, None, [], time.time()
    if getattr(model, "ema", 0) > 0:
        model._ema = {k: p.detach().clone() for k, p in model.named_parameters()}
    for ep in range(1, epochs + 1):
        loss = train_epoch(model, data, opt, batch, rng, max_steps=max_steps)
        sched.step()
        v = evaluate(model, data, select, max_q=valid_q, seed=0)["final"]["full"]["MRR"]
        if getattr(model, "_ema", None) is not None:       # also validate the averaged weights; keep whichever is better
            raw = {k: p.detach().clone() for k, p in model.named_parameters()}
            with torch.no_grad():
                for k, p in model.named_parameters():
                    p.copy_(model._ema[k])
            ve = evaluate(model, data, select, max_q=valid_q, seed=0)["final"]["full"]["MRR"]
            if ve > best and ve >= v:
                best, best_state = ve, copy.deepcopy(model.state_dict())
            with torch.no_grad():
                for k, p in model.named_parameters():
                    p.copy_(raw[k])
            v_raw, v = v, max(v, ve)
        hist.append(dict(epoch=ep, loss=loss, valid=v, sec=time.time() - t0))
        if verbose:
            print(f"  ep {ep:3d} loss {loss:.4f} valid MRR {v:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if v > best:
            best, best_state = v, copy.deepcopy(model.state_dict())
        if time_limit and time.time() - t0 > time_limit:
            break
    model.load_state_dict(best_state)
    return dict(best_valid=best, hist=hist, sec=time.time() - t0)


VARIANTS = {
    "NBFNet-style":        dict(rel_encoder=False, reranker=False),
    "ULTRA-style":         dict(rel_encoder=True, reranker=False),
    "+ CBLiP reranker":    dict(rel_encoder=True, reranker=True, path_tokens=False, transport=False),
    "+ path tokens":       dict(rel_encoder=True, reranker=True, path_tokens=True, transport=False),
    "HOTformer (full)":    dict(rel_encoder=True, reranker=True, path_tokens=True, transport=True),
    "ULTRA + HOT readout": dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=False),
    "HOTformer v2":        dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True),
    "HOTformer v2 + rel ids": dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, rel_ids=True),
    "ULTRA-style + rel ids":  dict(rel_encoder=True, reranker=False, rel_ids=True),
    "HOTformer v2 + PE":      dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, pe=True),
    "HOTformer v2 + PE in MP": dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, pe=True, pe_mp=True),
    "HOTformer v2 + hop":     dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, pe=True, pe_struct=False),
    "HOTformer v2 + gated PE": dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, pe=True,
                                    pe_gate=True, pe_drop=0.2),
    "HOTformer v2 + rel PE":  dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, pe=True,
                                   pe_struct=False, rel_pe=True),
    "HOTformer v2 + TDA head": dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, tda="head"),
    "HOTformer v2 + TDA":     dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, tda="both"),
    "HOTformer-PH":           dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, ph="pe+scan"),
    "HOTformer-PH + Hodge PE": dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, ph="pe+scan",
                                    hodge="pe"),
    "HOTformer-PH + TopER":    dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, ph="pe+scan",
                                    toper=True),
    "HOTformer-PHT":           dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, ph="pe+scan",
                                    hodge="pe", toper=True),
    "HOTformer v2 + Hodge PE": dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, hodge="pe"),
    "HOTformer v2 + harmonic": dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, hodge="harmonic"),
    "HOTformer-PH + harmonic": dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, ph="pe+scan",
                                    hodge="harmonic"),
    "HOTformer-PH (no scan)": dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, ph="pe"),
    "HOTformer-PH score":     dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, ph="pe+scan",
                                   ph_filt="score"),
    "HOTformer v2 + Hodge":   dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True, hodge=True),
    "HOTformer v2 + Hodge + holonomy": dict(rel_encoder=True, reranker=False, ot_readout=True, head_tf=True,
                                            hodge=True, holo=0.1),
}
# ---- PHOT: no Transformer (PH + OT on conditional message passing) ----
_PHOT = dict(reranker=False, ot_readout=True, head_tf=False, ph="pe+scan")
VARIANTS.update({
    "PHOT-U":        dict(_PHOT, rel_encoder=True),                     # ULTRA structural relation encoder
    "PHOT-N":        dict(_PHOT, rel_encoder=False),                    # learned relation embeddings (NBFNet)
    "PHOT-UN":       dict(_PHOT, rel_encoder=True, rel_ids=True),       # structural + learned identity
    "PHOT-U + DE":   dict(_PHOT, rel_encoder=True, drop_edge=0.2),
    "PHOT-N + DE":   dict(_PHOT, rel_encoder=False, drop_edge=0.2),
    "NBFNet-style + DE": dict(rel_encoder=False, reranker=False, drop_edge=0.2),
    "PHOT-N L6":     dict(_PHOT, rel_encoder=False, ent_layers=6),
    "PHOT-U lr1e-3": dict(_PHOT, rel_encoder=True, lr=1e-3),
    "PHOT-U mean":   dict(_PHOT, rel_encoder=True, agg="mean"),
    "PHOT-U pna":    dict(_PHOT, rel_encoder=True, agg="pna"),
    "PHOT-U mean jk": dict(_PHOT, rel_encoder=True, agg="mean", jk=True),
    "PHOT-U jk":     dict(_PHOT, rel_encoder=True, jk=True),
    "PHOT-U link":   dict(_PHOT, rel_encoder=True, link=True),
    "PHOT-U mean link": dict(_PHOT, rel_encoder=True, agg="mean", link=True),
    "PHOT-N link":   dict(_PHOT, rel_encoder=False, link=True),
    "PHOT-U count":  dict(_PHOT, rel_encoder=True, link="count"),
    "PHOT-U mean count": dict(_PHOT, rel_encoder=True, agg="mean", link="count"),
    "PHOT-U mean both": dict(_PHOT, rel_encoder=True, agg="mean", link="both"),
    "PHOT-N mean count": dict(_PHOT, rel_encoder=False, agg="mean", link="count"),
    "PHOT-UN mean count": dict(_PHOT, rel_encoder=True, rel_ids=True, agg="mean", link="count"),
    "NBFNet-style mean count": dict(rel_encoder=False, reranker=False, agg="mean", link="count"),
    "PHOT-UN mean count x2": dict(_PHOT, rel_encoder=True, rel_ids=True, agg="mean", link="count", epochs=20),
    "PHOT-UN mean count x4": dict(_PHOT, rel_encoder=True, rel_ids=True, agg="mean", link="count", epochs=40),
    "UN mean count x4 (no PH, no OT)": dict(rel_encoder=True, rel_ids=True, reranker=False, agg="mean", link="count", epochs=40),
    "PHOT x4 ema": dict(_PHOT, rel_encoder=True, rel_ids=True, agg="mean", link="count", epochs=40, ema=0.998),
    "PHOT x4 L7": dict(_PHOT, rel_encoder=True, rel_ids=True, agg="mean", link="count", epochs=40, ent_layers=7),
    "PHOT x2 d64": dict(_PHOT, rel_encoder=True, rel_ids=True, agg="mean", link="count", epochs=20, d=64),
    "PHOT-U mean count x2": dict(_PHOT, rel_encoder=True, agg="mean", link="count", epochs=20),
    "PHOT-U mean count L6": dict(_PHOT, rel_encoder=True, agg="mean", link="count", ent_layers=6),
    "PHOT-N pna L6": dict(_PHOT, rel_encoder=False, agg="pna", ent_layers=6),
    "PHOT-U pna L6": dict(_PHOT, rel_encoder=True, agg="pna", ent_layers=6),
    "PHOT-N lr1e-3": dict(_PHOT, rel_encoder=False, lr=1e-3),
    "PHOT-U lr5e-4 x2": dict(_PHOT, rel_encoder=True, lr=5e-4, epochs=20),
    "PH-N":          dict(rel_encoder=False, reranker=False, ph="pe+scan"),
    "PHOT-N + TTA":  dict(_PHOT, rel_encoder=False, tta=0.5),
    "PHOT-U + TTA":  dict(_PHOT, rel_encoder=True, tta=0.5),
    "PHOT-N L6 d64": dict(_PHOT, rel_encoder=False, ent_layers=6, d=64),
})
VARIANTS["PHOT"] = VARIANTS["PHOT-UN mean count x4"]          # recommended: 40 epochs x 120 batches of 16
VARIANTS["PHOT (2x budget)"] = VARIANTS["PHOT-UN mean count x2"]
VARIANTS["PHOT (1x budget)"] = VARIANTS["PHOT-UN mean count"]
HOTformer = PHOT = PHOTformer   # earlier names (HOT-KG lineage); kept so old scripts and logged results still load
# PHOTformer = Persistent Homology + Optimal Transport + Transformer (the recommended model; = "HOTformer-PH")
VARIANTS = {"PHOTformer": VARIANTS["HOTformer-PH"],
            "PHOTformer + Hodge PE + TopER": VARIANTS["HOTformer-PHT"],
    "PHOTformer + harmonic only": dict(VARIANTS["HOTformer-PH"], hodge="harmonic_only"),
            **{k.replace("HOTformer-PH", "PHOTformer"): v for k, v in VARIANTS.items() if k.startswith("HOTformer-PH") and k != "HOTformer-PHT"},
            **VARIANTS}
