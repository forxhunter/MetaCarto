# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research pipeline that reads genome-scale metabolic models (BiGG / SBML) and emits
[Escher](https://escher.github.io)-compatible `.json` pathway maps with automatically synthesized
node coordinates. The goal is layouts that look like curated KEGG/Escher maps (linear backbones,
cycles drawn as rings, cofactors as side-branches) rather than force-directed hairballs.
`templates/t*.{webp,png,jpg}` are positive visual targets; `templates/nt*.{jpeg,png}` are negative
examples (spring-layout hairballs).

There are **two independent pipelines**. Know which one you are touching:

| | v1 | v2 |
|---|---|---|
| entry point | `process_subsystems.py` | `layout_v2.py` |
| engine | `src/refinement.py`, inline SA | `src/layout/` |
| placement | simulated annealing from random init, then grid snapping | primary-compound reduction → Sugiyama layering → Brandes-Köpf |
| Escher output | schema-invalid (see below) | schema-correct |
| determinism | random seed, varies per run | fully deterministic |

v2 is newer and measurably better (see *Metrics* below). Neither is deleted.
`layout_algorithm.md` is the v2 design document and explains why v1's approach cannot reach the
target aesthetic.

## Environment

`requirements.txt` is aspirational. The **actual** working interpreter on this machine is the
`endosymbiont-fba` conda env (Python 3.11, `cobra 0.29.1`, `networkx 3.6.1`, `numpy 1.26.4`).
Base conda has no cobra.

```bash
C:/Users/shiel/miniconda3/envs/endosymbiont-fba/python.exe layout_v2.py --model e_coli_core
```

`torch_geometric`, `escher`, `rdkit`, `bioservices` and `pytest` are not installed anywhere.
`src/parsing.py` imports torch lazily inside `convert_to_pyg` for exactly this reason — do not
hoist those imports back to module scope, it breaks every driver that loads an SBML model.

**matplotlib cannot render on this machine.** Its Agg backend aborts the interpreter
(`0xc06d007f`, a delay-loaded DLL failure) inside `transforms`, `bezier` and the PIL writer, in
both conda envs that have it. `src/layout/raster.py` is therefore a self-contained PNG writer
(stdlib `zlib` only, with a 5x7 bitmap font) and `src/layout/preview.py` uses it. Do not
reintroduce a matplotlib dependency for previews.

There is no test suite and no linter config. `scripts/verify_setup.py`, `scripts/check_canvas.py`,
`scripts/verify_occupancy.py` and `scripts/analyze_density.py` inspect emitted v1 JSON;
`src/layout/metrics.py` is the v2 equivalent and is the one to extend.

## Commands

```bash
# v2 (preferred). One map per cluster; --combined adds a whole-model map.
python layout_v2.py --model e_coli_core --preview --combined
python layout_v2.py --model e_coli_core --subsystem "Citric Acid Cycle"
python layout_v2.py --all
python layout_v2.py --model iAF1260 --raw-subsystems   # skip constraints.md size limits
python layout_v2.py --model iAF1260 --no-fba           # skip pFBA orientation

# v1 (legacy). Batch-processes every model in data/bigg/models/
python process_subsystems.py

# Fetch inputs (data/ is gitignored and must be populated locally; 109 models present)
python scripts/fetch_bigg.py                      # -> data/bigg/models/
python scripts/fetch_kegg.py --orgs eco hsa sce   # -> data/kegg/ KGML files
python scripts/fetch_kegg_mapping.py              # -> data/kegg/kegg_mapping.json

# Publish: copy data/bigg/{ModelID}/*.json into the escher_maps_BiGG/ mirror
python scripts/prepare_repo.py
```

`process_bigg.py` and `run_pipeline.py` are earlier demo drivers, not a production path
(`run_pipeline.py` lays out a hardcoded 7-node mock graph).

## v2 architecture (`src/layout/`)

```
cobra model + reaction subset
  compound.build_compound_graph()   one directed edge per reaction between its MAIN pair
  direction.orient_compound_graph() pFBA flux sign, then a linear arrangement for zero-flux
  motifs.find_rings/contract_rings()  cycles become super-nodes with a reserved box
  sugiyama.layered_layout()         FAS -> layering -> crossing reduction -> Brandes-Köpf
  motifs.expand_rings()             members onto a circle, rotated to face entry/exit
  render.build_escher_map()         markers, cofactor arcs, orthogonal routing, labels
  compose.compose()                 --combined only: meta-tiling of the per-cluster maps
  metrics.score()                   acceptance metrics
```

Four ideas carry almost all the quality; change them only deliberately:

1. **Primary-compound reduction** (`compound.py`). Each reaction contributes *one* edge, between
   the substrate/product pair sharing the most molecular skeleton (`formula.moiety_score`); every
   other participant becomes a per-reaction side node. Without this every reaction node has degree
   4-8 and no orthogonal drawing exists. Cofactor-ness is applied as a **tier**, not a score
   penalty: a curated/high-degree cofactor is never a main-pair member while a non-cofactor
   alternative exists on its side. This is what stops CoA→acetyl-CoA (21 shared carbons) from
   beating pyruvate→acetyl-CoA. When *every* participant is currency the whole reaction collapses
   into one tier and raw chemistry decides, so transport of water and ion exchange still work.
2. **Orientation** (`direction.py`). BiGG stores PGM as `2pg -> 3pg` and PGK as `3pg -> 13dpg`,
   i.e. glycolysis partly backwards; drawn as stored it fragments into three short chains.
   pFBA decides direction where there is flux. Antiparallel pairs (PFK/FBP, PYK/PPS) are collapsed
   onto one axis — left as a 2-cycle they force the layering to put FDP above G6P.
3. **Brandes-Köpf** (`sugiyama.py`). This is what produces the straight vertical backbone. A
   barycenter x-assignment does not. Its balancing step takes a per-node median over four
   alignment runs, which is *not* a convex combination and can overlap nodes, so
   `_enforce_separation` runs afterwards — that guarantee is load-bearing, don't drop it.
4. **Deterministic cofactor geometry** (`render.py`). Stubs fan off the multimarkers at a fixed
   angle, both members of a pair on the same side, joined by a Bezier that leaves along the
   reaction axis. The perpendicular is derived from the *unflipped* direction; deriving it after
   negating for the substrate side flips the bank too and scatters ATP and ADP to opposite sides.

`compose.py` builds the whole-model map (`--combined`). It runs the *same* layered algorithm a
second time one level up: each cluster drawing becomes a box, the boxes are laid out by flow
(cluster A above B when A produces what B consumes), then shelf-packed in that flow order into a
poster-shaped block with a caption per pathway. Drawing the whole model in one Sugiyama pass
instead is what the naive `--combined` used to do, and it is 24x worse on crossings: iAF1260 goes
from 2.598 crossings per edge as one drawing to 0.106 as a composed one. Shared metabolites are
left duplicated per tile rather than wired across tiles, which is what `templates/t4` does.

`decompose.py` enforces the `constraints.md` cluster limits (≤60 reactions, ≥6) and falls back to
greedy-modularity communities on the currency-stripped graph. Many BiGG models — iAF1260 among
them — carry no `subsystem` annotation at all, so this is the normal path, not an edge case.

## Metrics

`src/layout/metrics.py` computes the acceptance metrics from `layout_algorithm.md` §9 and
`layout_v2.py` prints them per map with a `!` against anything outside target. Current v2 results
over 1795 maps from four models (e_coli_core, iAB_RBC_283, iAF692, iCHOv1):

| metric | p10 | median | p90 |
|---|---|---|---|
| axis_aligned | 1.000 | 1.000 | 1.000 |
| crossings_per_edge | 0.000 | 0.000 | 0.100 |
| min_separation_ratio | 1.444 | 1.444 | 1.444 |
| label_overlaps | 0 | 0 | 0 |
| hairball_index | 1.000 | 1.000 | 1.875 |

Use these as a regression gate: a change that improves one pathway by eye but degrades these
across the corpus is not an improvement. Two caveats when reading a report:

- `aspect_ratio` legitimately sits low for an unbranched pathway — a ten-step chain drawn as a
  tall column is what a reader expects, and folding it would be worse. `sugiyama._fold_columns`
  only wraps a stack past `fold_after` (26 layers), where it stops fitting on a page.
- `hairball_index` is not meaningful on a composed whole-model map. It measures density variance
  across the canvas, and a tiled poster is dense tiles separated by empty gutters by construction
  (iAF1260 scores 7.8). Read it per cluster, not on `*_Combined.json`.
- Subsystem `groups` trade crossings for clustering. On the e_coli_core combined map: no groups
  gives 0.163 crossings/edge and hairball 3.42; subsystem groups give 0.204 and 3.07. Grouping is
  the default because keeping a pathway in one place is the biological point; `--no-groups` opts out.

## Escher schema

`node_type` accepts exactly three values: `metabolite`, `multimarker`, `midmarker`. There is no
reaction node type — a reaction is a **midmarker** plus **multimarkers**, chained
`metabolite → multimarker → midmarker → multimarker → metabolite`, and arrowheads are derived
from the *signed* stoichiometry in `reactions[*].metabolites[*].coefficient`.

v2 (`src/layout/render.py`) emits this correctly. **v1 (`src/export.py`) does not**, and Escher
will not render its output properly regardless of how good the coordinates are:

- `node_type` is set to `"reaction"`, which is not a valid value;
- every segment has `"b1": null, "b2": null`, so all edges draw as straight diagonals;
- `coefficient` is hardcoded to `1`, discarding direction;
- `reversibility` is always `False`.

## v1 coordinate spaces — the main source of v1 bugs

v1 rescales coordinates four times and each stage assumes a different unit. When a v1 map comes
out empty, off-canvas, or piled at one point, this chain is why:

1. `optimize_layout_sa` works in **normalized [0,1]**.
2. `snap_to_grid(..., grid_spacing=0.2)` is called on that normalized space but internally
   **re-centers to (2500, 2500)**; the caller then manually subtracts the min to undo it. A
   centering side effect inside a snapping function is a layering mistake.
3. `pos_multiplier = 5.0` expands to roughly [0,5] "units".
4. `PIXELS_PER_UNIT = 800.0` converts to Escher pixels; `PADDING = 1000.0` separates tiles.

`snap_to_grid`'s Manhattan pass is a sequential per-edge averaging loop run 10 times. A degree-4
reaction node cannot axis-align all four edges, so later edges undo earlier ones and nodes collapse
together; the spiral anti-overlap step then scatters the collisions to arbitrary free cells,
destroying any alignment achieved. It is not a working orthogonalizer.

## Aspirational vs. live code

`plan.md` describes a GNN + RL architecture. None of it is on either production path:
`src/gnn_model.py`, `src/train_gnn.py`, `src/rl_env.py`, `src/train_rl.py`, `src/chemistry.py`,
`src/stress_test.py` and `src/data_loader.py` are unused by both `process_subsystems.py` and
`layout_v2.py`. `checkpoints/gnn_model.pth` and `ppo_metabolic_layout.zip` are artifacts of those
experiments. `src/fba.py` is superseded by `src/layout/direction.py`, which uses pFBA.

Treat `plan.md` as a roadmap, `constraints.md` as the binding requirements, and
`layout_algorithm.md` as the v2 design.

## Conventions

- v1 matches currency metabolites with a hardcoded prefix list duplicated in
  `process_subsystems.py` and `process_bigg.py` — keep both in sync. v2 uses
  `compound.compute_cofactor_scores`, which combines that list (as a prior) with whole-model
  connectivity percentiles, so it also catches `q8/q8h2`, `fad`, `ppi`, `amp` and `thf` variants.
- v1 duplicated currency nodes are named `{met}__dup_{subsystem}_{n}`; the `"__dup_"` substring is
  load-bearing in the combined-map assembly. v2 does not use this scheme.
- Every map carries a `"Created by Tianyu Wu (GitHub: forxhunter)"` label (`constraints.md` §4).
- `data/`, `validation_outputs/`, `*.json`, `*.xml`, `*.pth`, `*.zip` are gitignored.
  `escher_maps_BiGG/` is the intended publishable artifact.
- Both drivers wrap each model in a broad `try/except` that prints a traceback and continues.
  Failures are easy to miss in batch runs — grep the log for `FAILED` (v2) or `Error:` (v1).
