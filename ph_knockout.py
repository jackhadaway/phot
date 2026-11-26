"""Which PH features does a trained full-PHOT model actually use? Inference-time knockouts, no retraining.

Each of the 6 merge-tree inputs to ph_enc is replaced by its mean over the query's reachable entities (mean-ablation:
the value stays in range, only the per-candidate information is removed). Also: all five topological inputs at once
(only the raw filtration value f kept), all six, and the Topo-Scan token. P1 test MRR; results/ph_knockout.jsonl.

usage: python ph_knockout.py <dataset> <seeds=0,1,2>
"""
import json
import os
import sys

import torch
from torch import nn

FEATS = ["f", "f_join", "joined", "pers", "cycles", "size"]          # columns of pe in PHOTformer.ph_features
CONDS = {"none": [], **{k: [i] for i, k in enumerate(FEATS)}, "topology (keep f)": [1, 2, 3, 4, 5],
         "all merge-tree": [0, 1, 2, 3, 4, 5]}


class MeanAblate(nn.Module):
    def __init__(self, enc, model):
        super().__init__()
        self.enc, self.model, self.cols = enc, [model], []

    def forward(self, pe):
        if self.cols:
            r = self.model[0].eenc.reach.unsqueeze(-1).float()                     # (B, N, 1)
            mu = (pe * r).sum(1, keepdim=True) / r.sum(1, keepdim=True).clamp_min(1)
            pe = pe.clone()
            pe[..., self.cols] = mu[..., self.cols].expand_as(pe[..., self.cols])
        return self.enc(pe)


def main(ds, seeds):
    from photformer import InductiveSplit, PHOTformer, evaluate
    torch.set_num_threads(1)
    data = InductiveSplit("data/grail_repo/data", ds)
    for s in seeds:
        ck = torch.load(f"ckpt/abl_full_{ds}_s{s}.pt", weights_only=False)
        m = PHOTformer(ck["n_rel"], **ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
        wrap = MeanAblate(m.ph_enc, m); m.ph_enc = wrap
        orig = m.ph_features
        res = {}
        for name, cols in CONDS.items():
            wrap.cols = cols
            res[name] = evaluate(m, data, "test")["final"]["full"]["MRR"]
        wrap.cols = []
        m.ph_features = lambda *a, **k: (lambda o: [o[0], torch.zeros_like(o[1])])(orig(*a, **k))
        res["scan token"] = evaluate(m, data, "test")["final"]["full"]["MRR"]
        with open("results/ph_knockout.jsonl", "a") as fh:
            fh.write(json.dumps(dict(dataset=ds, seed=s, mrr=res)) + "\n")
        print(ds, s, {k: round(v, 4) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1], [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2").split(",")])
