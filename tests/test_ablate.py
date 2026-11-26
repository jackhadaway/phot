"""Every ablation variant builds, switches off exactly what it says, and trains."""
import numpy as np
import pytest
import torch

from ablate import ABLATIONS, BACKBONE, config
from photformer import VARIANTS, PHOTformer, set_seed, train_epoch


@pytest.mark.parametrize("variant", list(ABLATIONS))
def test_variant_trains(toy_split, variant):
    cfg = dict(config(variant), d=8)
    set_seed(0)
    model = PHOTformer(toy_split.n_rel, **cfg)
    assert bool(model.ph) == (variant not in ("no-PH", "neither"))
    assert hasattr(model, "scan_tf") == (variant in ("full", "no-OT"))
    assert model.ot_readout == (variant in ("full", "no-scan", "no-PH"))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss = train_epoch(model, toy_split, opt, batch=16, rng=np.random.default_rng(0), max_steps=2)
    assert np.isfinite(loss)


def test_only_ph_ot_switches_differ():
    base = VARIANTS["PHOT"]
    for v in ABLATIONS:
        diff = {k for k in set(base) | set(config(v)) if base.get(k) != config(v).get(k)}
        assert diff <= {"ph", "ot_readout"}, (v, diff)


@pytest.mark.parametrize("variant", list(BACKBONE))
def test_backbone_variant_trains(toy_split, variant):
    cfg = dict(config(variant), d=8)
    set_seed(0)
    model = PHOTformer(toy_split.n_rel, **cfg)
    assert not model.ph and not model.ot_readout and model.rel_encoder
    assert model.rel_ids == (variant == "bb-nolink")
    assert bool(model.link) == (variant == "bb-noid")
    assert hasattr(model, "link_enc") == bool(model.link)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss = train_epoch(model, toy_split, opt, batch=16, rng=np.random.default_rng(0), max_steps=2)
    assert np.isfinite(loss)


def test_backbone_differs_from_neither_only_in_extras():
    base = config("neither")
    for v in BACKBONE:
        diff = {k for k in set(base) | set(config(v)) if base.get(k) != config(v).get(k)}
        assert diff and diff <= {"rel_ids", "link"}, (v, diff)


def test_paired_t_matches_tables():
    from ablate import paired, t_pvalue
    assert t_pvalue(4.303, 2) == pytest.approx(0.05, abs=1e-3)       # textbook critical values
    assert t_pvalue(2.776, 4) == pytest.approx(0.05, abs=1e-3)
    assert t_pvalue(0.0, 2) == pytest.approx(1.0)
    mu, sd, t, p = paired(np.array([-0.04, -0.035, -0.042]))
    assert sd == pytest.approx(np.std([-0.04, -0.035, -0.042], ddof=1))
    assert t == pytest.approx(mu / (sd / np.sqrt(3))) and p < 0.01


def test_full_noid_is_phot_without_ids(toy_split):
    from ablate import NOID
    base = config("full")
    diff = {k for k in set(base) | set(config("full-noid")) if base.get(k) != config("full-noid").get(k)}
    assert diff == {"rel_ids"} and set(NOID) == {"full-noid"}
    model = PHOTformer(toy_split.n_rel, **dict(config("full-noid"), d=8))
    assert model.ph and model.ot_readout and not model.rel_ids and not hasattr(model, "rel_id_emb")


def test_multifilt_variants(toy_split):
    from ablate import MULTIFILT
    base = config("full")
    for v in MULTIFILT:
        diff = {k for k in set(base) | set(config(v)) if base.get(k) != config(v).get(k)}
        assert diff == {"n_filt"}, (v, diff)


@pytest.mark.parametrize("K", [1, 2, 4])
def test_multifilt_trains_and_each_filtration_gets_gradient(toy_split, K):
    cfg = dict(config("full"), d=8, n_filt=K)
    set_seed(0)
    model = PHOTformer(toy_split.n_rel, **cfg)
    assert model.filt.out_features == K and hasattr(model, "filt_emb") == (K > 1)
    assert model.s1[0].in_features == 2 * 8 + 8 + K * 16 + 8          # x, zq, link, K merge-tree blocks, scan
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss = train_epoch(model, toy_split, opt, batch=16, rng=np.random.default_rng(0), max_steps=2)
    assert np.isfinite(loss)
    g = model.filt.weight.grad
    assert g is not None and (g.abs().sum(1) > 0).all(), "every filtration must receive gradient"


def test_one_filtration_is_unchanged_model(toy_split):
    """n_filt=1 builds exactly the old parameter set, so earlier checkpoints still load"""
    set_seed(0); a = PHOTformer(toy_split.n_rel, **dict(config("full"), d=8))
    set_seed(0); b = PHOTformer(toy_split.n_rel, **dict(config("full"), d=8, n_filt=1))
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    ck = torch.load("ckpt/abl_full_fb237_v1_s0.pt", weights_only=False) if __import__("os").path.exists(
        "ckpt/abl_full_fb237_v1_s0.pt") else None
    if ck is not None:
        PHOTformer(ck["n_rel"], **ck["cfg"]).load_state_dict(ck["state"])


def test_ind_valid_selection_variants():
    from ablate import GROUPS, selection
    variants, datasets, ref = GROUPS["indsel"]
    assert set(variants) == {"full@iv", "full-noid@iv"} and ref == "full@iv"
    assert config("full@iv") == config("full") and config("full-noid@iv") == config("full-noid")
    assert selection("full@iv") == "ind_valid" and selection("full") == "valid" and selection("full-noid") == "valid"


def test_ind_valid_split_is_disjoint_from_test(toy_split):
    """selecting on ind_valid never sees a test triple"""
    from photformer import split_queries
    _, Qv, _ = split_queries(toy_split, "ind_valid")
    _, Qt, _ = split_queries(toy_split, "test")
    assert not ({tuple(q) for q in Qv.tolist()} & {tuple(q) for q in Qt.tolist()})
