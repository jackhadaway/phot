# PHOT: Persistent Homology + Optimal Transport for Inductive Knowledge-Graph Completion

PHOT predicts missing facts in knowledge graphs whose **entities were never seen during training**. It uses no learned entity embeddings; every representation is computed from graph structure conditioned on the query `(h, q, ?)`.

The recommended configuration (`VARIANTS["PHOT"]` in `photformer.py`) removes the Transformer used in earlier `HOTformer` / `PHOTformer` iterations and combines query-conditioned message passing, relation-agnostic link counts, query-conditioned persistent homology (`ph.py`), and entropic optimal transport.

## Method

For a query `(h, q, ?)`, `PHOT` (`PHOT-UN mean count x4`, `d=32`, 40 epochs × 120 batches of 16) computes:

1. **Relation vectors (`RelationEncoder` + `rel_id_emb`):** An ULTRA-style structural relation encoder (message passing over the 4-type relation interaction graph starting from `q`) summed with a learned identity vector per relation (for inductive splits that share the relation vocabulary).
2. **Query-conditioned message passing (`EntityEncoder`):** 5-layer NBFNet-style message passing with DistMult relation messages and **mean aggregation** (`agg="mean"`), producing a query-conditioned state `x_v` for every entity `v`.
3. **Link features (`link="count"`):** Relation-agnostic 1-hop and common-neighbour counts between `h` and each candidate `v` (`#facts(h, v)`, `#distinct relations(h, v)`, and `#common neighbours`). During training, both directions of the query's own edge are masked out (`query_masks`) so there is no leakage.
4. **Query-conditioned persistent homology (`ph.py`):** A learned superlevel filtration `f(v) = tanh(w · x_v)` orders entities (`h` first). One exact Numba union-find pass (`merge_tree`) computes per-candidate merge-tree features (`join`, `death`, `cyc`, `csize`), and `topo_scan` computes Betti/size summaries (`b0`, `b1`, `|V|`, `|E|`) over sliding quantile windows of `f`.
5. **HOT-KG transport readout (`transport_readout`):** `K=0` entropic optimal transport between `h`'s incident fact atoms (pushed forward by the query relation) and every candidate's incident fact atoms, computed in one scatter log-sum-exp pass.

The final score for candidate `v` is:

```text
MLP[x_v || z_q || link_v || PH_v || scan] + ot_w * transport(h, v)
```

Unreachable entities within `ent_layers` hops are clamped to `-1e4`. Training minimizes filtered 1-vs-all cross-entropy loss with reciprocal queries `(t, q^-1, h)`.

## Installation

Python 3.10+ with the pinned dependencies in `requirements.txt` (note that `numba==0.60.0` requires `numpy==2.0.2` and is incompatible with NumPy 2.4+):

```bash
python -m venv .venv
# Activate the environment, then install dependencies:
pip install -r requirements.txt
```

Dependencies in `requirements.txt`:
- `torch==2.8.0`
- `numpy==2.0.2`
- `numba==0.60.0`
- `pytest==8.4.2`
- `networkx==3.2.1`

## Data Setup

Standard GraIL inductive splits are loaded by `InductiveSplit` from `data/grail_repo/data/` (ignored by git via `.gitignore`). Each split `<name>` expects:
- `data/grail_repo/data/<name>/{train,valid}.txt` (training graph)
- `data/grail_repo/data/<name>_ind/{train,valid,test}.txt` (disjoint-entity inference graph)

For multi-graph pre-training with `pretrain.py`, CoDEx-S triples are additionally expected at `data/codex_s/train.txt`.

## Usage

### Smoke Test & Training (`smoke.py`)

Run an end-to-end training and evaluation pass on a GraIL split (defaults to `nell_v1` for 3 epochs):

```bash
python smoke.py
```

Pass a dataset name and epoch count as positional arguments; the full `VARIANTS["PHOT"]` schedule uses 40 epochs:

```bash
python smoke.py nell_v1 40
```

`smoke.py` selects the best checkpoint on the training graph's validation set (`select="valid"`), evaluates both **P1** (`test`) and **P2** (`test_merged`), and appends the run metrics and git commit hash to `results/runs.jsonl`.

### Programmatic API (`photformer.py`)

```python
from photformer import DEVICE, VARIANTS, InductiveSplit, PHOTformer, evaluate, fit, set_seed

data = InductiveSplit("data/grail_repo/data", "fb237_v1")
set_seed(0)
model = PHOTformer(data.n_rel, **VARIANTS["PHOT"]).to(DEVICE)

# Uses the variant's built-in schedule (40 epochs, lr=5e-3, max_steps=120)
train_stats = fit(model, data, seed=0, select="valid")

p1 = evaluate(model, data, "test")["final"]["full"]
p2 = evaluate(model, data, "test_merged")["final"]["full"]
```

### Multi-Graph Pre-Training & Fine-Tuning (`pretrain.py`, PHOT-P)

`pretrain.py` pre-trains a structural, relation-ID-free PHOT model (`VARIANTS["PHOT-U mean count"]`) jointly across the training graphs of `fb237_v1`, `WN18RR_v1`, `nell_v1`, and induced subgraphs of CoDEx-S, then fine-tunes with zero-initialized per-relation identity vectors:

```bash
# Pre-train (creates ckpt/pretrain.pt and .resume checkpoints)
python pretrain.py pre 40 6 ckpt/pretrain.pt

# Fine-tune on a target split (logs to results/pretrain.jsonl, saves to ckpt/)
python pretrain.py tune fb237_v1 0 10 2e-3
```

## Evaluation Protocol & Results

`evaluate()` reports filtered full ranking (`MRR`, `H1`, `H3`, `H10`) over all entities in the inference graph (predicting both heads and tails) as well as 50-negative ranking (`neg50`):
- **P1 (`split="test"`):** Inductive test triples (NBFNet / A*Net convention).
- **P2 (`split="test_merged"`):** Inductive validation + test triples merged (ULTRA / KG-ICL convention).
- **Model selection (`select="valid"`):** Best epoch selected strictly on the training graph's validation triples.

Benchmark numbers reported in `PHOT.ipynb` (GraIL v1 splits, mean ± std over 3 seeds, `d=32`, 40 epochs × 120 batches of 16, MRR / Hits@10):

| Model | Training Data | FB15k-237 v1 | WN18RR v1 | NELL-995 v1 |
|---|---|---|---|---|
| **PHOT (P1)** | Target dataset only | 0.460 ± 0.009 / 0.619 | 0.731 ± 0.004 / 0.837 | **0.812 ± 0.022** / **0.923** |
| PHOT, 3-seed ensemble (P1) | Target dataset only | 0.467 / 0.615 | 0.738 / 0.840 | **0.820** / **0.945** |
| **PHOT (P2)** | Target dataset only | 0.470 ± 0.005 / 0.625 | 0.730 ± 0.001 / 0.831 | **0.807 ± 0.016** / **0.920** |
| PHOT, 3-seed ensemble (P2) | Target dataset only | 0.481 / 0.623 | 0.733 / 0.832 | **0.816** / **0.933** |
| PHOT-P fine-tuned (P1 / P2) | Multi-graph + target | **0.481** / **0.486** (MRR) | 0.692 / 0.693 (MRR) | 0.699 / 0.703 (MRR) |

> **Note on NELL-995 v1:** As documented in `PHOT.ipynb`, NELL v1's inductive test set is small (100 triples in P1; 201 in P2) and its inference graph is star-shaped around a single hub entity.

## Tests

Run the test suite with `pytest` (configured via `pytest.ini`):

```bash
pytest
```

Test coverage in `tests/`:
- `tests/test_ph.py`: Verifies `merge_tree`, `topo_scan`, `_ols`, and `toper_query` against independent brute-force and NetworkX references.
- `tests/test_leakage.py`: Verifies that masking the query edge via `query_masks` produces identical states and scores to physically deleting the fact from the graph.
- `tests/test_eval_train.py`: Tests `tie_rank`, filtered ranking in `evaluate`, `query_masks`, gradient flow into PH/OT parameters, `train_epoch`, and `fit`.
- `tests/test_data.py`: Tests `Graph` augmentation/CSR invariants and validates the GraIL v1 inductive splits (automatically skipped if `data/grail_repo/data` is not present).

## Repository Structure

| File / Directory | Description |
|---|---|
| `photformer.py` | Data loading (`Graph`, `InductiveSplit`), model (`PHOTformer` / `PHOT` / `HOTformer`), `VARIANTS`, training (`train_epoch`, `fit`), and evaluation (`evaluate`). |
| `ph.py` | Numba-accelerated query-conditioned persistent homology (`merge_tree`, `topo_scan`, `toper_nodes`, `toper_query`). |
| `smoke.py` | End-to-end training/evaluation script logging to `results/runs.jsonl`. |
| `pretrain.py` | Multi-graph structural pre-training and per-dataset fine-tuning (`PHOT-P`). |
| `PHOT.ipynb` | Research notebook with method notes, benchmark tables, ablation history, inlined code, and demo. |
| `tests/` | `pytest` unit and integration suite (`conftest.py`, `test_ph.py`, `test_leakage.py`, `test_eval_train.py`, `test_data.py`). |
| `requirements.txt` | Pinned Python package versions. |
| `pytest.ini` | Pytest configuration (`testpaths = tests`). |
| `results/` | Output directory for run logs (`inductive.jsonl`, `runs.jsonl`, `pretrain.jsonl`). |
