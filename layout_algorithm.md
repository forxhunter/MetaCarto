# Biologically-Structured Layout Algorithm (v2)

> **Status: implemented.** `src/layout/` and `layout_v2.py`. Sections 1-5, 7, 8 (as
> `_enforce_separation`), 9, and §6's meta-tiling (`src/layout/compose.py`) are live; §6's
> compartment hulls are not. Measured results are in `benchmarks/results/`. Where the design
> below turned out to be wrong once tested, the correction is marked **[revised]**.

Target: given **any** subset of a metabolic model, produce a drawing that reads the way a
curated KEGG or Escher map does — linear pathway backbones, cycles as rings, cofactors as side
branches — rather than the way a force-directed layout does.

## Requirements

The binding requirements the implementation is held to. Code that enforces one of these cites
this section.

**Decomposition.** Cluster from the model's own pathway annotation where it has one, and only
fall back to structural community detection where it does not. Clusters hold at most 60
reactions and at least 6; a cluster below the floor is merged rather than drawn.

**Geometry.** Edges run horizontally or vertically. Two nodes joined by an edge share an x or a
y coordinate. Nodes do not overlap, and edges overlap as little as the drawing allows.

**Labels.** A label never overlaps a node, an edge, or another label. Where a dense cluster
leaves no free position at any size in the font ladder, a cofactor label is dropped rather than
forced, and the collision is counted (§9) rather than hidden.

**Output.** One map per cluster, plus an optional whole-model map that keeps global
coordinates. Per-cluster maps are centred and trimmed so the canvas fits the content and its
padding. Every map carries the attribution label
`Created by Tianyu Wu (GitHub: forxhunter)`.

## 0. Diagnosis: why v1 fails

The current engine (`process_subsystems.py`) minimizes a continuous energy function
(`calculate_energy`) by simulated annealing over `[0,1]²` from a **random** initial position, then
tries to repair the result with `snap_to_grid`. Three structural reasons this cannot produce t1:

1. **It optimizes the wrong object.** It lays out the raw bipartite reaction–metabolite graph.
   Curated maps lay out the *primary-compound digraph*: one directed edge per reaction, between
   that reaction's main substrate and main product. Everything else is decoration. Without that
   reduction, every reaction node has degree 4–8 and no planar/orthogonal drawing exists.
2. **No direction.** `nx.Graph` is undirected throughout, and stoichiometric signs are discarded
   (`src/export.py` hardcodes `coefficient: 1`). Every positive template has a global flow axis:
   catabolism down, inputs at the top/periphery. Flow direction is not an aesthetic preference,
   it is the thing that makes a map readable.
3. **Continuous optimization cannot produce discrete structure.** A ring, a straight 8-node
   backbone, and a cofactor stub are combinatorial objects. SA over positions finds a local
   minimum that is a blob, and `snap_to_grid`'s per-edge averaging then collapses nodes together
   (a degree-4 node cannot axis-align all four edges) and the spiral de-overlap scatters them.

The fix is not a better energy function. It is to make the layout **constructive**: recognize
biological motifs, place them with a motif-specific rule, and use combinatorial layering
(Sugiyama) for the glue. Optimization is demoted to a final local polish.

## 1. Primary-compound reduction

The single highest-leverage step. Turns the hairball into a sparse digraph.

For each reaction `r`, pick one **main substrate → main product** pair; classify all other
participants as *side metabolites*.

```
def main_pair(rxn):
    subs = [m for m,c in rxn.metabolites.items() if c < 0]
    prods = [m for m,c in rxn.metabolites.items() if c > 0]
    # score each (s, p) candidate pair by conserved-moiety similarity
    best = argmax over (s,p) of carbon_overlap(s, p)
    return best
```

`carbon_overlap(s, p)` without atom mapping, in priority order:

1. If both have `met.formula`, compute shared heavy-atom counts —
   `min(C_s, C_p) + 0.3 * min(N_s, N_p) + 0.1 * min(P_s, P_p)`. The pair transferring the most
   carbon is the KEGG "RPAIR main" pair. This alone gets glycolysis right.
2. Tie-break by lower global degree (a metabolite appearing in 200 reactions is a cofactor, not
   a backbone member).
3. Tie-break by same compartment.

**[revised]** A multiplicative or additive cofactor penalty is not enough on its own: CoA →
acetyl-CoA shares 21 carbons and outscores pyruvate → acetyl-CoA under any weighting that does not
also break other reactions. The implementation applies cofactor-ness as a **tier** — a cofactor is
never a main-pair member while a non-cofactor alternative exists on its side — and falls back to
raw chemistry when every participant is currency. Chemistry also cannot break the citrate-synthase
tie (acetyl-CoA → citrate vs. oxaloacetate → citrate score within 2%), so near-tied candidates are
re-scored by what they do to the backbone: close a cycle, join two components, or continue an
existing chain. That is the criterion a curator applies, and it picks oxaloacetate → citrate.

Replace `is_currency`'s hardcoded prefix list with a **learned/computed** cofactor set:
`degree(m) > percentile_90(degree)` **or** `m` in the curated list **or** `m` was never selected
as a main-pair member. The prefix list stays as a prior, not as the rule — it currently misses
`q8/q8h2`, `fad/fadh2`, `amp`, `ppi`, `thf` variants, `so4`, `nad(p)` compartment variants.

Output: `D = DiGraph` over primary metabolites, one edge per reaction, plus a
`side[r] = (consumed[], produced[])` table. For a 60-reaction subsystem `D` typically has
~50 nodes and ~60 edges — sparse and drawable.

**Edge direction for reversible reactions.** In priority order: (a) FBA flux sign from
`src/fba.py` (revive it — `cobra.flux_analysis.pfba` on the parent model, cache per model);
(b) direction that minimizes back-edges in a global topological order; (c) `rxn.lower_bound >= 0`.

## 2. Motif decomposition

Run on `D`, in this order, each consuming its nodes:

| Motif | Detection | Drawing rule |
|---|---|---|
| **Ring** | chordless directed cycle, length 3–12, ≥2 external connections | circle/rounded-rect; radius ∝ length; entry node faces incoming layer, exit faces outgoing |
| **Backbone chain** | maximal path where every interior node has in-deg = out-deg = 1 | straight vertical run, uniform pitch |
| **Fan** | node with out-deg ≥ 4 into leaves (aminoacyl-tRNA, biomass) | radial fan, leaves on an arc |
| **Grid/ladder** | repeated 2-step motifs (fatty-acid elongation, nucleotide series) | serpentine rows, as in `t4` |
| **Residual** | everything else | handed to Sugiyama as ordinary nodes |

Contract each detected motif into a **super-node** with a known bounding box and a set of
labelled ports (entry/exit). `t1`'s TCA ring and `t2`'s cycle collection are exactly this.

Detection notes: use `nx.simple_cycles(D, length_bound=12)` (NetworkX ≥3.1), keep cycles that are
chordless and node-disjoint, prefer longer ones. `cycle_basis` on the undirected graph — what v1
uses in `get_cycle_nodes` — returns a basis, not the biologically meaningful rings, and v1 then
drops them into five hardcoded canvas slots regardless of connectivity.

## 3. Sugiyama layering on the condensed DAG

This replaces `optimize_layout_sa` entirely. It is deterministic, O(V·E)-ish, and its known
output shape *is* the target aesthetic: layered flow + long straight vertical chains.

1. **Cycle removal.** Greedy feedback-arc-set (Eades–Lin–Smyth) on the condensed graph; reversed
   edges are restored at the end and drawn with reversed arrowheads.
2. **Layer assignment.** Network-simplex (`nx.drawing.nx_agraph` if Graphviz is available, else
   implement longest-path + tightening). Layer index = **y** coordinate. Constraints:
   - exchange / extracellular metabolites pinned to layer 0 (sources) or layer max (sinks) —
     this is the hard-constraint version of v1's soft `peripheral_energy`;
   - same-compartment nodes get a layer-span penalty so compartments stay contiguous.
3. **Crossing minimization.** Median/barycenter sweeps + adjacent-transpose, 8–16 iterations,
   **cluster-constrained** so nodes of the same pathway/compartment stay adjacent within a layer.
4. **X-coordinate assignment.** Brandes–Köpf (four-pass alignment + median balancing). This is the
   step that produces the straight vertical glycolysis spine — it explicitly aligns chains of
   degree-1-in/degree-1-out nodes. Do not substitute a simple barycenter; it will not straighten.

Layer pitch and in-layer pitch are set from node label widths so labels never collide by
construction, rather than being repaired afterwards.

## 4. Motif expansion and port matching

Expand each super-node at its assigned position:

- **Ring**: place members on a circle of radius `r = pitch * n / (2π)`. Rotate the ring so the
  angular position of the entry port points at the incoming edge's direction and the exit port at
  the outgoing edge's. This is why TCA in `t1` and `t2` always reads correctly.
- **Chain**: emit at uniform pitch along the layer axis.
- Re-route the edges that entered the super-node to the corresponding member node.

## 5. Cofactor decoration (deterministic, not random)

v1 places duplicated currency metabolites at `random.random() * 2π` around the reaction node,
which is why the output looks like noise. Replace with a fixed convention, which is what Escher
itself and every positive template use:

For reaction edge `s → p` with unit direction `u` and left normal `n`:

```
midmarker      at  m = (pos[s] + pos[p]) / 2
consumed side  at  m - 0.30*L*u + 0.45*L*n      # e.g. ATP,  H2O in
produced side  at  m + 0.30*L*u + 0.45*L*n      # e.g. ADP,  Pi out
```

Both members of a cofactor pair go on the **same** side `n`, producing the classic curved
double-arrow. Side selection for `n`: alternate per layer, or choose the side with more free space
(cheap: sample a 3×3 occupancy grid). Each cofactor instance is a fresh duplicate node — never a
shared hub. Segments get Bezier control points `b1`, `b2` at ±⅓ along a quadratic bulge so the
stub is drawn as an arc.

## 6. Compartments and modules

- Compartment = the `_c` / `_m` / `_e` / `_p` suffix. After coordinates are fixed, draw a rounded
  convex hull (Escher has no group primitive — emit it as a `text_label` plus a reserved margin,
  or post-process the SVG). Transport reactions are the only edges allowed to cross a hull.
- Module (subsystem) packing **[implemented, revised]**: `compose.py` runs the layered pass on the
  meta-graph, but then *discards* its x-coordinates. Brandes-Koepf aligns nodes into columns, which
  is what straightens a pathway backbone and what wastes space at tile scale -- a column sized for
  the widest tile leaves a hole wherever a small tile sits in it. What is worth keeping is the
  layer assignment and the within-layer order (the flow, and the crossing-minimised arrangement),
  so tiles are shelf-packed in that order into a poster-shaped block, top-aligned per row. Original
  plan, kept for the record: keep v1's variable row-height/column-width grid
  but replace `kamada_kawai` + gravity compaction with Sugiyama on the meta-graph — pathways have
  a real hierarchy (central carbon in the middle, biosynthesis radiating outward), and a layered
  meta-layout reproduces it. Route inter-module edges as orthogonal buses down the gutters, as in
  `t2`.

## 7. Orthogonal routing

Every segment becomes a polyline with ≤2 bends, encoded in Escher's `b1`/`b2`. For same-layer or
adjacent-layer edges the bends are trivial; for long edges introduced by layering, insert dummy
nodes during step 3 (standard Sugiyama) and emit them as `multimarker` nodes — which is also
exactly what Escher's schema expects.

## 8. Local polish (this is where annealing belongs)

Only after the constructive layout exists, run a short, **position-constrained** refinement:
- nodes may move at most ±½ pitch;
- objective = label overlaps + edge crossings + edge length, with a hard barrier on node overlap;
- 100–200 steps, not 1000 from random init.

## 9. Acceptance metric (also the RL reward, if that branch is revived)

Compute on every emitted map. **This table is the code** (`metrics.py:14-25`); where it once
differed from the code, the code won and this was corrected. Metric names are the dict keys
`score()` returns.

| Key | Definition | Target |
|---|---|---|
| `axis_aligned` | fraction of *straight* segments within 1.0 unit of horizontal or vertical | > 0.90 |
| `crossings_per_edge` | segment crossings / segments | < 0.05 |
| `longest_run_ratio` | longest merged axis-aligned run, **in length units**, / total segment length | > 0.15 |
| `min_separation_ratio` | min pairwise distance between *primary* metabolites / pitch | ≥ 1.0 |
| `label_overlaps` | overlapping label-box pairs, boxes eroded 3.0 units | 0 |
| `label_on_node` | label boxes intersecting a node disc, excluding their own anchors | 0 |
| `label_on_edge` | label boxes struck by a straight segment | 0 |
| `hairball_index` | peak / mean bin count over *occupied* bins of a 20×20 histogram of metabolites | < 3 |
| `occupancy` | content bbox area / canvas area | 0.10 – 0.85 |
| `aspect_ratio` | content width / content height | 0.35 – 3.0 |

Also computed, deliberately **ungated** and reported for context only: `nodes`, `reactions`,
`connectors`, `label_shrunk`.

Three corrections to what this section claimed before, all in the direction of the code:

- *Axis-aligned* is measured over **straight segments only** — a cofactor stub and a ring arc are
  curved on purpose, and charging them with being non-orthogonal measures the design rather than
  a defect. Inter-tile `link_` connectors are curved too but *are* measured, by their chord,
  because a connector crossing the network is exactly what the gate exists to catch. The old
  wording, "axis-aligned or ≤1 bend", described neither.
- *Longest straight chain* is normalised by **total segment length**, not node count.
- *Label overlaps* is **three** metrics, not one: box-vs-box, box-vs-node, box-vs-edge.

**Calibration is the weak point of this table, and is being replaced.** Every threshold above
was set from two images — `nt1`/`nt2` score ~0.1 axis-aligned and hairball > 15, the `t*` set
scores well — with no sensitivity analysis and no sample to speak of. A threshold justified by
two JPEGs is not publishable. `data/kegg/` holds 1,010 KGML pathways carrying human-drawn
coordinates, which gives an empirical distribution of what curated layout actually scores;
these targets should be re-derived from it and this note removed once they are.

**[revised]** Two of these targets were also wrong as first written:

- *Aspect ratio* legitimately sits near 0.15 for an unbranched pathway. Glycolysis is a tall
  column in every textbook; folding a ten-step chain to hit a target would make it worse. Folding
  is applied only past 26 layers, where a drawing stops fitting on a page.
- *Minimum separation* has to be enforced, not just measured. Brandes-Köpf's balancing step takes
  a per-node median across four alignment runs; because different nodes take their median from
  different runs it is not a convex combination, and it does produce overlaps on large models. A
  single left-to-right sweep per layer afterwards makes the guarantee unconditional.

Implemented in `src/layout/metrics.py` and reported per map by `layout_v2.py`.

## Implementation order

Each step is independently shippable and visibly improves output:

1. **Blockers first** — guard the `torch_geometric` import in `src/parsing.py`; emit
   `midmarker`/`multimarker` node types and signed coefficients in `src/export.py`; strip the
   `(2500, 2500)` centering side effect out of `snap_to_grid`. Until (2) is done, Escher renders
   these maps incorrectly no matter how good the coordinates are.
2. **§1 primary-compound reduction** + §5 deterministic cofactor stubs. Biggest visual gain per
   line of code; works even with the existing SA placement.
3. **§3 Sugiyama** replacing `optimize_layout_sa` and `snap_to_grid`.
4. **§2/§4 motifs** (rings first — TCA, urea, Calvin are the most recognizable).
5. **§7 routing**, **§6 compartments**, **§9 metrics gate**, **§8 polish**.
