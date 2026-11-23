"""PH / OT ablation: train one (variant, dataset, seed) and log P1/P2 to results/ablation.jsonl.

Variants differ from PHOT only in the PH / OT switches; everything else (relations, mean aggregation, link counts,
40 epochs x 120 batches of 16) is VARIANTS["PHOT"]. The epoch is picked on the FULL validation set of the training
graph (see the WN18RR epoch-selection finding), never on test.

usage: python ablate.py run <variant> <dataset> <seed>     one run (THREADS env, default 1)
       python ablate.py queue [workers=3] [seeds=0,1,2]    every run not yet logged, seed-major, in parallel
       python ablate.py table                              mean +- std per variant and dataset, with paired deltas
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ABLATIONS = {
    "full":    dict(),                              # = PHOT: merge-tree + Topo-Scan + OT
    "no-scan": dict(ph="pe"),                       # merge-tree features only (no Topo-Scan token)
    "no-PH":   dict(ph=None),                       # OT only
    "no-OT":   dict(ot_readout=False),              # PH (merge-tree + scan) only
    "neither": dict(ph=None, ot_readout=False),     # message passing + link counts
}
DATASETS = ["nell_v1", "fb237_v1", "WN18RR_v1"]
LOG = "results/ablation.jsonl"


def config(variant):
    from photformer import VARIANTS
    return dict(VARIANTS["PHOT"], **ABLATIONS[variant])


def run(variant, ds, seed):
    import torch
    from photformer import DEVICE, InductiveSplit, PHOTformer, evaluate, fit, set_seed
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    data = InductiveSplit("data/grail_repo/data", ds)
    cfg = config(variant)
    set_seed(seed)
    model = PHOTformer(data.n_rel, **cfg).to(DEVICE)
    tr = fit(model, data, seed=seed, verbose=True, valid_q=None, select="valid")
    res = {s: evaluate(model, data, s)["final"]["full"] for s in ("test", "test_merged")}
    os.makedirs("ckpt", exist_ok=True)
    torch.save(dict(dataset=ds, n_rel=data.n_rel, cfg=cfg, state=model.state_dict()),
               f"ckpt/abl_{variant}_{ds}_s{seed}.pt")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    rec = dict(variant=variant, dataset=ds, seed=seed, test=res, train=tr, commit=commit,
               threads=torch.get_num_threads())
    with open(LOG, "a") as fh:                       # one short line per run: safe to append from parallel workers
        fh.write(json.dumps(rec) + "\n")
    print(f"{variant} {ds} s{seed}: P1 {res['test']['MRR']:.4f} P2 {res['test_merged']['MRR']:.4f} ({tr['sec']:.0f}s)")


def done():
    if not os.path.exists(LOG):
        return set()
    return {(r["variant"], r["dataset"], r["seed"]) for r in map(json.loads, open(LOG))}


def queue(workers=3, seeds=(0, 1, 2)):
    os.makedirs("results/ablation_logs", exist_ok=True)
    have = done()
    jobs = [(v, d, s) for s in seeds for d in DATASETS for v in ABLATIONS if (v, d, s) not in have]
    print(f"{len(jobs)} runs to do, {len(have)} already logged, {workers} workers", flush=True)

    def one(job):
        v, d, s = job
        t0 = time.time()
        with open(f"results/ablation_logs/{v}_{d}_s{s}.log", "w") as fh:
            rc = subprocess.run([sys.executable, "-u", __file__, "run", v, d, str(s)], stdout=fh,
                                stderr=subprocess.STDOUT, env=dict(os.environ, THREADS="1")).returncode
        print(f"{time.strftime('%H:%M')} {'ok  ' if rc == 0 else 'FAIL'} {v} {d} s{s} ({(time.time() - t0) / 60:.0f} min)",
              flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))
    print("queue finished", flush=True)


def table():
    R = {}
    for r in map(json.loads, open(LOG)):
        R[(r["variant"], r["dataset"], r["seed"])] = r["test"]
    for proto, key in (("P1 (inductive test)", "test"), ("P2 (valid+test merged)", "test_merged")):
        print(f"\n{proto}: MRR mean +- std (seeds) | paired delta vs full, mean +- std")
        for d in DATASETS:
            print(f"  {d}")
            for v in ABLATIONS:
                seeds = sorted(s for (vv, dd, s) in R if vv == v and dd == d)
                if not seeds:
                    continue
                m = np.array([R[(v, d, s)][key]["MRR"] for s in seeds])
                both = [s for s in seeds if ("full", d, s) in R]
                delta = np.array([R[(v, d, s)][key]["MRR"] - R[("full", d, s)][key]["MRR"] for s in both])
                dtxt = f" | {delta.mean():+.3f} +- {delta.std():.3f}" if v != "full" and len(delta) else ""
                print(f"    {v:8s} {m.mean():.3f} +- {m.std():.3f} ({len(m)}){dtxt}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif sys.argv[1] == "queue":
        w = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        seeds = tuple(int(s) for s in sys.argv[3].split(",")) if len(sys.argv) > 3 else (0, 1, 2)
        queue(w, seeds)
    else:
        table()
