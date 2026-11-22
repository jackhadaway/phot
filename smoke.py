"""End-to-end smoke test: the notebook demo (PHOT on NELL-995 v1, standard protocol, short budget).

usage: python smoke.py [dataset=nell_v1] [epochs=3]
3 epochs only checks that the pipeline runs; NELL at 3 epochs is still in the 'hub answers only' regime (MRR ~0.56).
The reported numbers use the variant's 40 epochs: python smoke.py nell_v1 40
"""
import json
import os
import subprocess
import sys
import time

import torch

from photformer import DEVICE, VARIANTS, InductiveSplit, PHOTformer, evaluate, fit, set_seed

ds = sys.argv[1] if len(sys.argv) > 1 else "nell_v1"
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
data = InductiveSplit("data/grail_repo/data", ds); print(data.summary())
set_seed(0)
model = PHOTformer(data.n_rel, **VARIANTS["PHOT"]).to(DEVICE)
# VALID_Q: validation queries used to pick the epoch ("all" = the full validation set; the notebook default is 200).
# TRACK=1: also log, every epoch, the 200-query validation MRR and the test MRR. Diagnostic only, never used to select.
vq = os.environ.get("VALID_Q", "200"); valid_q = None if vq == "all" else int(vq)
track = os.environ.get("TRACK") == "1"


def diagnostics(m, ep):
    return {"v200": evaluate(m, data, "valid", max_q=200, seed=0)["final"]["full"]["MRR"],
            "P1": evaluate(m, data, "test")["final"]["full"]["MRR"],
            "P2": evaluate(m, data, "test_merged")["final"]["full"]["MRR"]}


t0 = time.time()
tr = fit(model, data, seed=0, verbose=True, epochs=epochs, select="valid", valid_q=valid_q,
         on_epoch=diagnostics if track else None)
res = {}
for name, split in (("P1 (inductive test)", "test"), ("P2 (valid+test merged)", "test_merged")):
    f = evaluate(model, data, split)["final"]["full"]; res[split] = f
    print(f"{name}: MRR {f['MRR']:.4f}  H@1 {f['H1']:.4f}  H@10 {f['H10']:.4f}")
print(f"total {time.time() - t0:.0f}s")
os.makedirs("ckpt", exist_ok=True)
ck = f"ckpt/{ds}_s0_e{epochs}_v{vq}.pt"
torch.save(dict(dataset=ds, n_rel=data.n_rel, cfg=VARIANTS["PHOT"], state=model.state_dict()), ck)
print("saved", ck)
os.makedirs("results", exist_ok=True)
with open("results/runs.jsonl", "a") as fh:
    fh.write(json.dumps(dict(dataset=ds, variant="PHOT", seed=0, epochs=epochs, valid_q=vq, test=res, train=tr,
                             commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                                   text=True).stdout.strip())) + "\n")
