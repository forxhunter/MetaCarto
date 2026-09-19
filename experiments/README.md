# Abandoned experiments

Nothing in this directory is on a production path. It is kept because the negative results are
worth recording, not because any of it runs.

**Do not read this as the design of MetaCarto.** The method is described in
`../layout_algorithm.md` and implemented in `../src/layout/`.

## The learned-layout branch

`gnn_model.py`, `train_gnn.py`, `data_loader.py`, `rl_env.py`, `train_rl.py`, `stress_test.py`

The original plan was to learn layout: a graph attention network to classify reactions into
pathway communities, then a reinforcement-learning agent placing nodes on a discrete grid
against a reward combining orthogonality, backbone straightness and node proximity
(`../plan.md` records the intended architecture).

It was abandoned before producing a usable map, for a reason that turned out to be the central
insight of the project: **the hard part of drawing a metabolic network is not placement, it is
deciding what to draw.** Once each reaction is reduced to a single edge between its main
substrate/product pair — with cofactor-ness applied as a tier rather than a score penalty — a
classical layered drawing produces clean backbones directly, deterministically, with no
training data and no random seed. The learned agent was being asked to rediscover, from a
reward signal, a structure that chemistry already specifies.

What survives from this branch is the evaluation function. The reward terms became the
acceptance metrics in `../src/layout/metrics.py`, which is now the quality gate.

These files import `torch`, `torch_geometric` and `stable_baselines3`, none of which are
installed. They will not import as-is.

## Superseded modules

- `fba.py` — replaced by `../src/layout/direction.py`, which uses parsimonious FBA and then a
  back-edge-minimising arrangement for zero-flux reactions.
- `chemistry.py` — the RDKit-based chemistry helpers. `../src/layout/formula.py` computes the
  conserved-moiety score from molecular formulas alone, which needs no cheminformatics
  dependency and is enough to pick a reaction's main pair.

## Demo drivers

- `run_pipeline.py` — lays out a hardcoded 7-node mock graph. Never read a real model.
- `process_bigg.py` — an early batch driver superseded by `../layout_v2.py`.

The v1 simulated-annealing pipeline itself is **not** here: `../process_subsystems.py` and
`../src/refinement.py` are still in the tree, because v1 is the internal baseline the method is
measured against.
