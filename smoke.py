"""End-to-end smoke test: the notebook demo (PHOT on NELL-995 v1, standard protocol, short budget).

usage: python smoke.py [dataset=nell_v1] [epochs=3]
3 epochs only checks that the pipeline runs; NELL at 3 epochs is still in the 'hub answers only' regime (MRR ~0.56).
The reported numbers use the variant's 40 epochs: python smoke.py nell_v1 40
"""
import sys
import time

import torch

from photformer import DEVICE, VARIANTS, InductiveSplit, PHOTformer, evaluate, fit, set_seed

ds = sys.argv[1] if len(sys.argv) > 1 else "nell_v1"
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
torch.set_num_threads(2)
data = InductiveSplit("data/grail_repo/data", ds); print(data.summary())
set_seed(0)
model = PHOTformer(data.n_rel, **VARIANTS["PHOT"]).to(DEVICE)
t0 = time.time()
fit(model, data, seed=0, verbose=True, epochs=epochs, select="valid")
for name, split in (("P1 (inductive test)", "test"), ("P2 (valid+test merged)", "test_merged")):
    f = evaluate(model, data, split)["final"]["full"]
    print(f"{name}: MRR {f['MRR']:.4f}  H@1 {f['H1']:.4f}  H@10 {f['H10']:.4f}")
print(f"total {time.time() - t0:.0f}s")
