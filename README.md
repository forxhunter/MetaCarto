# MetaCarto

**Constructive layout synthesis for genome-scale metabolic networks**

MetaCarto reads a genome-scale metabolic model (BiGG / SBML) and draws it the way a curator
would — linear pathway backbones, cycles as rings, cofactors as side branches — rather than the
way a force-directed algorithm does. Output is schema-correct
[Escher](https://escher.github.io) JSON, loadable in any Escher viewer.

The layout is **constructive and deterministic**: no annealing, no random seed, no user
intervention. The same model always produces the same map.

## The generated collection

All 108 models in the [BiGG database](http://bigg.ucsd.edu/), drawn as **2,623 pathway maps**,
are published under CC BY 4.0 at
[forxhunter/escher_maps_BiGG](https://github.com/forxhunter/escher_maps_BiGG) — including
Recon3D (10,600 reactions, 93 maps).

Browse them in the viewer at **[forxhunter.github.io/escher](https://forxhunter.github.io/escher/)**
via *Map ▸ Load map from library…*, which reads the collection directly; nothing to download.

## How it works

Five ideas carry most of the quality. `layout_algorithm.md` is the design document.

1. **Primary-compound reduction** (`src/layout/compound.py`). A reaction such as
   `pyruvate + CoA + NAD⁺ → acetyl-CoA + CO₂ + NADH` contributes *one* directed edge, between
   the substrate/product pair sharing the most molecular skeleton — here pyruvate → acetyl-CoA.
   Every other participant becomes a side branch. Without this, each reaction node has degree
   4–8 and no clean orthogonal drawing exists, which is why naive layouts come out as
   hairballs. Cofactor-ness is applied as a **tier**, not a score penalty: a curated or
   high-degree cofactor is never chosen while a non-cofactor alternative exists on its side.
   That is what stops CoA → acetyl-CoA (21 shared carbons) beating pyruvate → acetyl-CoA.

2. **Direction from flux** (`src/layout/direction.py`). Reconstructions store reversible
   reactions in whichever direction the curator wrote them, so glycolysis is often recorded
   partly backwards. Parsimonious FBA decides the drawn direction wherever a reaction carries
   flux; antiparallel pairs are collapsed onto one axis.

3. **Cycles drawn as cycles** (`src/layout/motifs.py`). Rings are detected on the whole-model
   graph, contracted for layering, then expanded onto a circle rotated so the entry arc faces
   the pathway feeding it.

4. **Layered placement** (`src/layout/sugiyama.py`). Greedy feedback-arc-set, layer assignment,
   cluster-constrained crossing reduction, then **Brandes–Köpf** coordinate assignment. The
   last step is what produces straight vertical backbones; a barycentre assignment does not.

5. **Two scales** (`src/layout/compose.py`). The whole-model map runs the same layered pass one
   level up: each cluster drawing becomes a tile, tiles are ordered by metabolic flow and
   packed into a captioned poster.

**Decomposition** (`src/layout/decompose.py`) assigns reactions to maps from the model's own
`subsystem` annotation where it has one, then from a KEGG pathway lookup, and only failing both
from network structure — `networkx` greedy-modularity communities on the currency-stripped
graph. Many BiGG models carry no subsystem annotation at all, so the fallback is a normal path,
not an edge case. Maps are then named after the metabolic function they cover, following KEGG
BRITE top-level categories.

## Install and run

The map collection and the Escher fork that serves it are **submodules**, so the parent
repository records which commit of each this work was built against:

```bash
git clone --recurse-submodules https://github.com/forxhunter/MetaCarto.git
# already cloned without them:
git submodule update --init
```

The working interpreter is a conda environment with `cobra`, `networkx` and `numpy`:

```bash
python layout_v2.py --model e_coli_core --preview
python layout_v2.py --model Recon3D --group-function --max-cluster 120

# The published collection, in one command. `--max-cluster 120` is part of it:
# without it the same models give roughly a third as many, larger maps, and
# the collection on disk was built with a mix of the two.
python layout_v2.py --all --group-function --max-cluster 120 --out data/bigg
```

Useful flags: `--group-function` merges pathway clusters into functional maps, `--subsystem`
draws one named cluster, `--combined` adds a whole-model map, `--no-fba` skips pFBA
orientation, `--preview` writes PNG, SVG and vector PDF alongside the JSON.

`data/` is not tracked. Populate it with `scripts/fetch_bigg.py` and `scripts/fetch_kegg.py`.

## Quality gates

`src/layout/metrics.py` scores every emitted map — edge orthogonality, crossings per edge,
longest straight run, node separation, three kinds of label collision, local density
(`hairball_index`), occupancy and aspect ratio. Targets and their definitions are in
`layout_algorithm.md` §9.

```bash
python scripts/iterate.py --model e_coli_core   # exits non-zero if any gate fails
```

`iterate.py` adds three gates `metrics.py` does not have: print legibility (label size in
points once the map is fitted to a journal column), title/caption collisions, and canvas
overflow.

## Known limitations

- A few reactions per genome-scale model have no drawable primary pair — typically small
  inorganic chemistry such as catalase or CO₂ transport — and are omitted.
- Whole-model composed maps (`--combined`) are not print figures. Tiling 10,600 reactions onto
  one canvas leaves each reaction so little area that labels land around 0.14 pt. The
  per-cluster maps are the readable artifact.
- `hairball_index` still exceeds its target on most large merged function maps.
- H⁺ and H₂O are suppressed from every map.
- Compartments are not drawn as envelopes: the Escher schema has no region primitive.

## Legacy v1 pipeline

`process_subsystems.py` and `src/refinement.py` are the original simulated-annealing pipeline.
They are kept for comparison and are **not** the production path. Their Escher output is
schema-invalid — `node_type` is set to `"reaction"`, which is not a legal value; every segment
has null `b1`/`b2` so edges draw as straight diagonals; stoichiometric coefficients are
hardcoded to 1, discarding direction. `run_pipeline.py` lays out a hardcoded 7-node mock graph
and is a demo, not a driver.

`plan.md` describes a GNN + reinforcement-learning architecture that was explored and
abandoned. None of it is on either production path; see `experiments/README.md`.

## Licence and citation

**[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**, the same licence as the map
collection. You may use, modify and redistribute this for any purpose including commercially,
**on the condition that you give attribution** — attribution is a term of the licence, not a
courtesy.

If you use MetaCarto, or maps it generated, in a paper, figure, talk, poster, database or
derived software, please cite this repository. `CITATION.cff` carries the machine-readable
form and GitHub renders it as *Cite this repository*.

```bibtex
@software{Wu_MetaCarto_constructive_layout,
  author  = {Wu, Tianyu},
  license = {CC-BY-4.0},
  title   = {{MetaCarto: constructive layout synthesis for genome-scale metabolic networks}},
  url     = {https://github.com/forxhunter/MetaCarto}
}
```

Please also cite the underlying model from [BiGG Models](http://bigg.ucsd.edu/) and, where the
maps are displayed, [Escher](https://doi.org/10.1371/journal.pcbi.1004321).

A note for anyone reusing the code: CC BY is a content licence rather than a software licence,
so it carries no patent grant and is not OSI-approved. It is used here deliberately, to keep
the citation requirement identical across the software and the maps.

## Credits

**Created by Tianyu Wu (GitHub: [forxhunter](https://github.com/forxhunter))**, University of
Illinois Urbana-Champaign.
