# Metabolic AutoLayout

**Generative Layout Synthesis for Large-Scale Metabolic Networks**

This repository contains the official implementation of the "AutoLayout" algorithm, designed to automatically generate aesthetically pleasing, biologically meaningful, and Escher-compatible layouts for genome-scale metabolic models (GEMs).

## Key Features

*   **Global Meta-Layout**: Tiles subsystems (Glycolysis, TCA, etc.) on a variable-size grid using Gravity Compaction to minimize whitespace.
*   **Automated Decomposition**: Automatically performs community detection (Louvain) on unannotated "hairball" models to create functional clusters.
*   **Coordinate Normalization**: Ensures zero packing drift for tight layouts.
*   **Hybrid Optimization**: Combines Simulated Annealing (local node placement) with Grid Snapping (global alignment).
*   **Escher Compatibility**: Exports directly to `.json` maps loadable in [Escher-FBA](https://escher.github.io).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/forxhunter/AutoLayout.git
    cd AutoLayout
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### v2 engine (`layout_v2.py`)

Structure-driven layout: each reaction is reduced to one directed edge between its main
substrate/product pair, reversible steps are oriented by pFBA flux, cycles are drawn as rings,
and placement is a layered (Sugiyama + Brandes-Köpf) drawing rather than an optimiser. Output is
schema-correct Escher JSON with midmarkers, signed stoichiometry and curved cofactor arcs.

```bash
python layout_v2.py --model e_coli_core --preview --combined
python layout_v2.py --all
```

Every map is scored against the acceptance metrics in `layout_algorithm.md` (edge orthogonality,
crossings, node separation, label overlaps, hairball index). Across 1795 maps from four models the
median map is 100% axis-aligned with zero crossings and zero label overlaps.

See `layout_algorithm.md` for the algorithm and why the v1 approach below cannot reach it.

### v1: Batch Processing
To generate layouts for all BiGG models in `data/bigg/models`:

```bash
python process_subsystems.py
```
*   **Output**: `data/bigg/{ModelID}/{ModelID}_Combined.json`

### v1: Single Model Pipeline
To process a specific model manually:
```bash
python run_pipeline.py --model e_coli_core
```

## Algorithm Details

### Functional Decomposition
For models with `subsystem` metadata, the algorithm respects biological boundaries. For "Uncategorized" models, it uses **Greedy Modularity Community Detection** to infer functional modules from the bipartite reaction-metabolite graph.

### Layout Engine
1.  **Backbone Optimization**: simulated annealing optimizes reaction alignment.
2.  **Hub Duplication**: Duplicate high-degree currency metabolites (ATP, NADH) to prevent "hairballs".
3.  **Meta-Tiling**: Subsets are placed on a meta-grid.
4.  **Gravity Compaction**: Grid tiles are pulled Up-Left to remove gaps.

## Credits

**Created by Tianyu Wu (GitHub: forxhunter)**
Developed at University of Illinois Urbana-Champaign (UIUC).
