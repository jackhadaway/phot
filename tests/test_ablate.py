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
