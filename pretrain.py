"""PHOT-P: pre-train one structural (relation-id-free) PHOT on several graphs, then fine-tune per dataset.

Pre-training graphs (no inference-graph information of any target):
  - the TRAINING graphs of FB15k-237 v1, WN18RR v1, NELL-995 v1 (GraIL),
  - CoDEx-S (Wikidata; a different knowledge graph), as random induced sub-graphs.
Fine-tuning: add a learned identity vector per relation (zero-initialised, so fine-tuning starts exactly at the
pre-trained model) and train on the target's training graph; epoch picked on its validation triples.

usage: python pretrain.py pre  [rounds=40] [codex sub-graphs=6] [out=ckpt/pretrain.pt]
       python pretrain.py tune <dataset> [seed=0] [epochs=10] [lr=2e-3] [ema=0]   -> results/pretrain.jsonl
"""
import sys, os, json, time, copy, types, numpy as np, torch
from collections import defaultdict
from photformer import *

ROOT = "data/grail_repo/data"
DS = ["fb237_v1", "WN18RR_v1", "nell_v1"]
CFG = dict(VARIANTS["PHOT-U mean count"])           # structural relations only: works on any graph


def codex_subgraphs(n_sub=6, n_ent=700, seed=0):
    e, r = {}, {}
    tr = read_triples("data/codex_s/train.txt", e, r)
    rng = np.random.default_rng(seed); out = []
    for i in range(n_sub):
        keep = rng.choice(len(e), n_ent, replace=False); idx = -np.ones(len(e), dtype=np.int64); idx[keep] = np.arange(n_ent)
        m = (idx[tr[:, 0]] >= 0) & (idx[tr[:, 2]] >= 0)
        sub = np.stack([idx[tr[m, 0]], tr[m, 1], idx[tr[m, 2]]], 1)
        d = types.SimpleNamespace(name=f"codex_s#{i}", n_rel=len(r), train=sub, G_train=Graph(sub, n_ent, len(r)))
        f = defaultdict(set)
        for h, rr, t in sub:
            f[(int(h), int(rr))].add(int(t)); f[(int(t), int(rr) + len(r))].add(int(h))
        d.filt_train = f; out.append(d)
    return out


def both(model, data):
    return {k: evaluate(model, data, s)["final"]["full"] for k, s in (("P1", "test"), ("P2", "test_merged"))}


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    mode = sys.argv[1]
    if mode == "pre":
        rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 40
        n_sub = int(sys.argv[3]) if len(sys.argv) > 3 else 6; out = sys.argv[4] if len(sys.argv) > 4 else "ckpt/pretrain.pt"
        grail = [InductiveSplit(ROOT, d) for d in DS]; cx = codex_subgraphs(n_sub)
        print("pre-training graphs:", [(d.name, len(d.train)) for d in grail + cx], flush=True)
        set_seed(0); rng = np.random.default_rng(0)
        model = PHOTformer(1, **CFG)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, rounds)
        best, t0, start = -1, time.time(), 1
        if os.path.exists(out + ".resume"):               # the machine may restart: continue from the last finished round
            rs = torch.load(out + ".resume", weights_only=False)
            model.load_state_dict(rs["model"]); opt.load_state_dict(rs["opt"]); sched.load_state_dict(rs["sched"])
            best, start = rs["best"], rs["round"] + 1; rng = np.random.default_rng(rs["round"])
            np.random.seed(rs["round"]); torch.manual_seed(rs["round"]); print("resumed at round", start, flush=True)
        for ep in range(start, rounds + 1):
            losses = [train_epoch(model, d, opt, 16, rng, max_steps=30) for d in grail]
            losses += [train_epoch(model, cx[(2 * ep + j) % len(cx)], opt, 16, rng, max_steps=30) for j in range(2)]
            sched.step()
            v = [evaluate(model, d, "valid", max_q=100, seed=0)["final"]["full"]["MRR"] for d in grail]
            print(f"round {ep:3d} loss {np.mean(losses):.3f} valid {[round(x, 3) for x in v]} mean {np.mean(v):.4f} ({time.time()-t0:.0f}s)", flush=True)
            if np.mean(v) > best:
                best = float(np.mean(v)); torch.save(dict(cfg=CFG, state=copy.deepcopy(model.state_dict())), out)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), best=best, round=ep), out + ".tmp")
            os.replace(out + ".tmp", out + ".resume")
        print("best mean valid", best)
    else:
        ds = sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 10; lr = float(sys.argv[5]) if len(sys.argv) > 5 else 2e-3
        ema = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
        PRE = os.environ.get("PRE", "ckpt/pretrain.pt"); PTAG = os.environ.get("PTAG", "PHOT-P")
        data = InductiveSplit(ROOT, ds); ck = torch.load(PRE, weights_only=False)
        set_seed(seed)
        model = PHOTformer(data.n_rel, **dict(ck["cfg"], rel_ids=True, ema=ema))
        missing = model.load_state_dict(ck["state"], strict=False); print("not loaded:", missing.missing_keys)
        zero = both(model, data)
        print(f"{ds} zero-shot: P1 {zero['P1']['MRR']:.4f} P2 {zero['P2']['MRR']:.4f}", flush=True)
        tr = fit(model, data, seed=seed, verbose=False, select="valid", epochs=epochs, lr=lr)
        ft = both(model, data)
        tag = f"{PTAG} ft{epochs} lr{lr:g}" + (" ema" if ema else "")
        print(f"{ds} {tag} seed {seed}: P1 MRR {ft['P1']['MRR']:.4f} H@10 {ft['P1']['H10']:.4f} | P2 MRR {ft['P2']['MRR']:.4f} "
              f"H@10 {ft['P2']['H10']:.4f} | valid {tr['best_valid']:.4f} ({tr['sec']:.0f}s)", flush=True)
        open("results/pretrain.jsonl", "a").write(json.dumps(dict(dataset=ds, variant=tag, seed=seed, zero_shot=zero, test=ft, train=tr)) + "\n")
        torch.save(dict(cfg=dict(ck["cfg"], rel_ids=True, ema=ema), state=model.state_dict()), f"ckpt/{ds}__{tag.replace(' ', '_')}__{seed}.pt")
