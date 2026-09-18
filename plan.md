# Autonomous Implementation Plan: KEGG-Style Metabolic Map Generation

## 1. Project Overview

This project aims to develop an end-to-end autonomous pipeline that ingests a Systems Biology Markup Language (SBML) file and generates a functionally-grouped, KEGG-style metabolic map in the Escher-JSON format. The system utilizes Graph Neural Networks (GNNs) for biological module detection and Reinforcement Learning (RL) for aesthetic spatial optimization.

## 2. Technical Requirements

### 2.2 Software Environment

- **Core Libraries:** `Python 3.9+`, `COBRApy` (for FBA), `libRoadRunner` (for JIT model simulation).1
- **AI Frameworks:** `PyTorch Geometric` (for GNNs), `Ray RLlib` or `Stable Baselines3` (for Reinforcement Learning).
- **I/O Utilities:** `sbml2escher.py` script for schema validation and initial conversion.3

### 2.3 Required Training Data

- **Ground Truth:** ~160 KEGG KGML files (Kyoto Encyclopedia of Genes and Genomes) to learn coordinate hierarchies.
- **Functional Metadata:** KEGG COMPOUND database (~19,000 entries) and BiGG Models (~100 GEMs).

## 3. Implementation Workflow

### Phase 1: Knowledge Extraction & Feature Engineering

1. **Bipartite Graph Construction:** Transform the SBML model into a graph $G = (V, E)$ where metabolites and reactions are nodes.
2. **Flux Weighting:** Perform FBA to determine steady-state flux $v$ ($S \cdot v = 0$). Use these values to weight the "centrality" of nodes.
3. **Molecular Descriptors:** Extract SMILES strings for metabolites and convert them into MACCS fingerprints. This allows the AI to recognize chemical shapes.

### Phase 2: Functional Grouping (GNN Architecture)

1. **Model Selection:** Deploy a Graph Attention Network (GAT) to learn node embeddings based on stoichiometric connectivity.5
2. **Community Detection:** Train the GNN to predict metabolic pathway classes (e.g., Carbohydrate vs. Lipid metabolism). Target a predictive accuracy of $>95\%$.6
3. **Subsystem Partitioning:** Segment the global graph into functional modules that match the clusters seen in curated KEGG maps.

### Phase 3: Layout Synthesis (Reinforcement Learning)

1. **RL Environment:** Model the map canvas as a discrete grid. The agent’s "actions" are the placement of nodes and the routing of reaction lines.
2. **The Reward Function ($R$):**
   - **Orthogonality ($R_{ortho}$):** Positive reward for edges that are perfectly horizontal or vertical.
   - **Backbone Logic:** High reward for placing "major" metabolites (primary flux carriers) in a linear sequence while duplicating "minor" currency metabolites (ATP, $CO_2$) to prevent clutter.8
   - **Functional Proximity:** Reward placing nodes from the same GNN-defined cluster in the same spatial quadrant.

### Phase 4: Aesthetic KEGG-Style Refinement

1. **Node Primitives:** Automatically map reaction nodes to rectangular boxes (enzymes) and metabolites to circular nodes.5
2. **Grid Snapping:** Use a post-processing cost minimization algorithm to snap all coordinates to integer grid points.
3. **Annotation Overlay:** Auto-populate enzyme boxes with EC numbers and gene IDs derived from the SBML metadata.

### Phase 5: Synthesis and Escher-JSON Export

1. **Schema Mapping:** Pack the generated $(x, y)$ coordinates into the Escher-JSON schema (Metadata, Canvas, Nodes, and Reactions).10
2. **Validation:** Run `python -m escher.validate output.json` to ensure the file is ready for the web viewer.12

## 4. Success Metrics

- **Biological Integrity:** All reactant-product relationships from the SBML must be preserved.
- **Visual Clarity:** Edge crossings should be minimized by $>70\%$ compared to force-directed layouts.
- **KEGG Similarity:** The resulting map should visually correlate with traditional pathway hierarchies (biosynthetic on left, degradative on right).11