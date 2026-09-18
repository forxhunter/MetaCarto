# Strict Constraints Checklist

This document tracks all strict requirements defined by the user for the metabolic map generation.

## 1. Biological Decomposition (Priority 1)
- [x] **Primary Method**: Use KEGG Pathway or EC Number annotations (`kegg.reaction`, `ec-code`).
- [ ] **Fallback**: Use Structural Clustering (Louvain) ONLY if biological annotations are missing.
- [ ] **Ignore Reaction Constraints**: Biological grouping takes precedence over topological density.

## 2. Cluster Size & Count Limits
- [x] **Max Clusters**: 15 (Extended from 10).
- [x] **Max Reaction Size**: 60 reactions per cluster.
    - *Action*: Split larger clusters into smaller chunks using Community Detection.
- [x] **Min Reaction Size**: 6 reactions per cluster.
    - *Action*: Merge smaller clusters into "Uncategorized".

## 3. Geometric Layout (Strict)
- [x] **Orthogonality**: All edges must be Horizontal or Vertical lines.
- [x] **Alignment**: Nodes connected by an edge must share an X or Y coordinate.
- [x] **Minimal Overlap**:
    - Edges should overlap as little as possible.
    - Nodes must not overlap.
- [x] **Label Safety**: Labels (Metabolite/Reaction names) must **NEVER** overlap with:
    - Nodes
    - Edges
    - Other Labels

## 4. Export & Presentation
- [x] **Dual Export**:
    1.  **Combined Map**: Retains global coordinates, tiled layout.
    2.  **Individual Maps**: Separate JSON for each subsystem.
- [x] **Formatting (Individual Maps)**:
    - **Centered**: Content centered on canvas.
    - **Trimmed**: White blank area removed (Canvas fits content + padding).
- [x] **Attribution**: "Created by Tianyu Wu (GitHub: forxhunter)" label on all maps.
