"""Does the inductive-graph validation split (ind_valid: unseen entities, disjoint from ind_test) track test where the
training-graph validation set does not? Scores every saved ablation checkpoint on ind_valid; appends to
results/ind_valid.jsonl. Training-graph valid and test MRR come from results/ablation.jsonl.

usage: python ind_valid_check.py [workers=3]      evaluate every checkpoint not yet scored
       python ind_valid_check.py report           per-variant means of the three numbers, and seed-level agreement
"""
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

OUT = "results/ind_valid.jsonl"


def one(path):
    import torch
    from photformer import InductiveSplit, PHOTformer, evaluate
    torch.set_num_threads(1)
    ck = torch.load(path, weights_only=False)
    data = InductiveSplit("data/grail_repo/data", ck["dataset"])
    m = PHOTformer(ck["n_rel"], **ck["cfg"]); m.load_state_dict(ck["state"])
    mrr = evaluate(m, data, "ind_valid")["final"]["full"]["MRR"]
    ds = ck["dataset"]
    v, s = os.path.basename(path)[4:-3].split(f"_{ds}_s")
    s = int(s)
    with open(OUT, "a") as fh:
        fh.write(json.dumps(dict(variant=v, dataset=ds, seed=s, ind_valid=mrr)) + "\n")


def todo():
    have = {(r["variant"], r["dataset"], r["seed"]) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
    paths = []
    for p in sorted(glob.glob("ckpt/abl_*.pt")):
        stem = os.path.basename(p)[4:-3]
        for ds in ("fb237_v1", "WN18RR_v1", "nell_v1"):
            if f"_{ds}_s" in stem:
                v, s = stem.split(f"_{ds}_s")
                if (v, ds, int(s)) not in have:
                    paths.append(p)
    return paths


def report():
    T = {(r["variant"], r["dataset"], r["seed"]): r for r in map(json.loads, open("results/ablation.jsonl"))}
    rows = [dict(r, test=T[k]["test"]["test"]["MRR"], valid=T[k]["train"]["best_valid"])
            for r in map(json.loads, open(OUT)) if (k := (r["variant"], r["dataset"], r["seed"])) in T]
    for ds in ("fb237_v1", "WN18RR_v1", "nell_v1"):
        R = [r for r in rows if r["dataset"] == ds]
        if not R:
            continue
        print(f"\n{ds}: per-variant means (seeds) | train-graph valid | ind_valid | test")
        vs = sorted({r["variant"] for r in R}, key=lambda v: -np.mean([r["test"] for r in R if r["variant"] == v]))
        M = {}
        for v in vs:
            a = np.array([[r["valid"], r["ind_valid"], r["test"]] for r in R if r["variant"] == v]).mean(0)
            M[v] = a
            print(f"  {v:10s} ({sum(r['variant'] == v for r in R)})  {a[0]:.3f}  {a[1]:.3f}  {a[2]:.3f}")
        A = np.array(list(M.values()))
        rk = lambda x: np.argsort(np.argsort(x))
        print(f"  rank corr. with test over variants: train-graph valid {np.corrcoef(rk(A[:, 0]), rk(A[:, 2]))[0, 1]:+.2f}"
              f", ind_valid {np.corrcoef(rk(A[:, 1]), rk(A[:, 2]))[0, 1]:+.2f}")
        X = np.array([[r["valid"], r["ind_valid"], r["test"]] for r in R])
        print(f"  over all {len(X)} checkpoints: train-graph valid {np.corrcoef(rk(X[:, 0]), rk(X[:, 2]))[0, 1]:+.2f}"
              f", ind_valid {np.corrcoef(rk(X[:, 1]), rk(X[:, 2]))[0, 1]:+.2f}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    elif len(sys.argv) > 1 and sys.argv[1] == "one":
        one(sys.argv[2])
    else:
        paths = todo()
        print(f"{len(paths)} checkpoints to score", flush=True)
        with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as ex:
            list(ex.map(lambda p: subprocess.run([sys.executable, __file__, "one", p], check=False), paths))
        print("done", flush=True)
