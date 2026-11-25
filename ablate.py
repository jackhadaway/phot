"""PH / OT ablation: train one (variant, dataset, seed) and log P1/P2 to results/ablation.jsonl.

Variants differ from PHOT only in the PH / OT switches; everything else (relations, mean aggregation, link counts,
40 epochs x 120 batches of 16) is VARIANTS["PHOT"]. The epoch is picked on the FULL validation set of the training
graph (see the WN18RR epoch-selection finding), never on test.

usage: python ablate.py run <variant> <dataset> <seed>     one run (THREADS env, default 1)
       python ablate.py queue [workers=3] [seeds=0,1,2] [group=phot|backbone|noid]   every run not yet logged, in parallel
       python ablate.py table                              mean +- std per variant and dataset, with paired deltas
"""
import json
import math
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
# backbone ablation: starts from "neither" (no PH, no OT) and removes the backbone's own extras.
BACKBONE = {
    "bb-noid":   dict(ABLATIONS["neither"], rel_ids=False),               # no learned relation identities
    "bb-nolink": dict(ABLATIONS["neither"], link=False),                  # no direct-link counts
    "bb-plain":  dict(ABLATIONS["neither"], rel_ids=False, link=False),   # ~ ULTRA architecture trained from scratch
}
# full PHOT without relation identities: do PH + OT still add on top of the no-ID backbone (bb-noid)?
NOID = {"full-noid": dict(rel_ids=False)}
# group -> (variants, datasets, reference variant for paired deltas). NELL v1 is too noisy for these effects.
GROUPS = {
    "phot":     (ABLATIONS, DATASETS, "full"),
    "backbone": ({"neither": ABLATIONS["neither"], **BACKBONE}, ["fb237_v1", "WN18RR_v1"], "neither"),
    "noid":     ({"bb-noid": BACKBONE["bb-noid"], **NOID}, ["fb237_v1", "WN18RR_v1"], "bb-noid"),
}
LOG = "results/ablation.jsonl"


def config(variant):
    from photformer import VARIANTS
    return dict(VARIANTS["PHOT"], **{**ABLATIONS, **BACKBONE, **NOID}[variant])


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


def queue(workers=3, seeds=(0, 1, 2), group="phot"):
    os.makedirs("results/ablation_logs", exist_ok=True)
    have = done()
    variants, datasets, _ = GROUPS[group]
    jobs = [(v, d, s) for s in seeds for d in datasets for v in variants if (v, d, s) not in have]
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


def t_pvalue(t, df):
    """two-sided p-value of Student's t with df degrees of freedom (numerical integral of the density; no scipy)"""
    if not np.isfinite(t):
        return 0.0
    x = np.linspace(0.0, abs(t), 20001)
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    y = c * (1 + x ** 2 / df) ** (-(df + 1) / 2)
    return float(max(0.0, 1.0 - 2.0 * np.sum((y[1:] + y[:-1]) / 2 * np.diff(x))))


def paired(delta):
    """mean, sample std (ddof=1), paired t and two-sided p for per-seed differences"""
    n = len(delta)
    if n < 2:
        return delta.mean(), float("nan"), float("nan"), float("nan")
    sd = delta.std(ddof=1)
    t = delta.mean() / (sd / math.sqrt(n)) if sd > 0 else math.copysign(float("inf"), delta.mean())
    return delta.mean(), sd, t, t_pvalue(t, n - 1)


def table():
    """mean +- sample std over seeds, and the paired (same-seed) difference to the group's reference variant with a
    paired t-test. * = p < 0.05, ** = p < 0.01. With 3 seeds (df = 2), p < 0.05 needs |t| > 4.30."""
    R = {}
    for r in map(json.loads, open(LOG)):
        R[(r["variant"], r["dataset"], r["seed"])] = r["test"]
    for group, (variants, datasets, ref) in GROUPS.items():
        for proto, key in (("P1 (inductive test)", "test"), ("P2 (valid+test merged)", "test_merged")):
            print(f"\n[{group}] {proto}: MRR mean +- sd (seeds) | paired difference vs {ref}: mean +- sd, t, p")
            for d in datasets:
                print(f"  {d}")
                for v in variants:
                    seeds = sorted(s for (vv, dd, s) in R if vv == v and dd == d)
                    if not seeds:
                        continue
                    m = np.array([R[(v, d, s)][key]["MRR"] for s in seeds])
                    sd = f"{m.std(ddof=1):.3f}" if len(m) > 1 else "  -  "
                    line = f"    {v:9s} {m.mean():.3f} +- {sd} ({len(m)})"
                    both = [s for s in seeds if (ref, d, s) in R]
                    if v != ref and both:
                        delta = np.array([R[(v, d, s)][key]["MRR"] - R[(ref, d, s)][key]["MRR"] for s in both])
                        mu, dsd, t, pv = paired(delta)
                        if len(delta) > 1:
                            star = "**" if pv < 0.01 else "*" if pv < 0.05 else ""
                            line += f" | {mu:+.3f} +- {dsd:.3f}  t {t:+6.1f}  p {pv:.3f} {star}"
                        else:
                            line += f" | {mu:+.3f} (1 seed)"
                    print(line)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif sys.argv[1] == "queue":
        w = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        seeds = tuple(int(s) for s in sys.argv[3].split(",")) if len(sys.argv) > 3 else (0, 1, 2)
        queue(w, seeds, sys.argv[4] if len(sys.argv) > 4 else "phot")
    else:
        table()
