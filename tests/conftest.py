import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
GRAIL = os.path.join(ROOT, "data", "grail_repo", "data")

import torch  # noqa: E402

torch.set_num_threads(1)


def random_facts(n_ent, n_rel, n_facts, seed, self_loops=False):
    """random unique (h, r, t) triples"""
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    while len(out) < n_facts:
        h, r, t = int(rng.integers(n_ent)), int(rng.integers(n_rel)), int(rng.integers(n_ent))
        if (h == t and not self_loops) or (h, r, t) in seen:
            continue
        seen.add((h, r, t)); out.append((h, r, t))
    return np.array(out, dtype=np.int64)


def write_split(root, name, train, valid, ind_graph, ind_valid, ind_test):
    """write a GraIL-format split with string ids (train entities 'a*', inference entities 'b*')"""
    def dump(path, arr, pre):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            for h, r, t in arr:
                fh.write(f"{pre}{h}\tr{r}\t{pre}{t}\n")
    dump(f"{root}/{name}/train.txt", train, "a")
    dump(f"{root}/{name}/valid.txt", valid, "a")
    dump(f"{root}/{name}_ind/train.txt", ind_graph, "b")
    dump(f"{root}/{name}_ind/valid.txt", ind_valid, "b")
    dump(f"{root}/{name}_ind/test.txt", ind_test, "b")


@pytest.fixture
def toy_split(tmp_path):
    from photformer import InductiveSplit
    R = 3
    tr = random_facts(30, R, 110, seed=1)
    ind = random_facts(25, R, 90, seed=2)
    write_split(str(tmp_path), "toy", tr[:100], tr[100:], ind[:70], ind[70:80], ind[80:])
    return InductiveSplit(str(tmp_path), "toy")


def grail_available(name):
    return os.path.exists(os.path.join(GRAIL, name, "train.txt"))
