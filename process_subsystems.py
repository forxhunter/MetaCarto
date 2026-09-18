import os
import glob
import torch
import numpy as np
import networkx as nx
import copy
import random
import math
import cobra
import json

from src.parsing import load_sbml_model, build_bipartite_graph
from src.refinement import snap_to_grid
from src.export import serialize_results

# Parameters
MODEL_DIR = "data/bigg/models"
OUTPUT_DIR = "data/bigg"
HUB_DEGREE_THRESHOLD = 15

# Common currency metabolites
CURRENCY_PREFIXES = [
    'h2o', 'atp', 'adp', 'nad', 'nadh', 'nadp', 'nadph', 
    'pi', 'h', 'co2', 'o2', 'nh4', 'coa', 'accoa'
]

def is_currency(node_id):
    base = node_id.lower()
    for p in CURRENCY_PREFIXES:
        if base == p or base.startswith(f"{p}_"):
            return True
    return False


# KEGG Mapping Cache
KEGG_MAPPING_FILE = "data/kegg/kegg_mapping.json"
_KEGG_MAPPING = None

def load_kegg_mapping():
    global _KEGG_MAPPING
    if _KEGG_MAPPING is None:
        if os.path.exists(KEGG_MAPPING_FILE):
             try:
                 with open(KEGG_MAPPING_FILE, 'r') as f:
                     _KEGG_MAPPING = json.load(f)
                 print(f"  Loaded KEGG mapping for {len(_KEGG_MAPPING)} reactions.")
             except Exception as e:
                 print(f"  Failed to load KEGG mapping: {e}")
                 _KEGG_MAPPING = {}
        else:
             print("  Warning: KEGG mapping file not found. Skipping biological decomposition.")
             _KEGG_MAPPING = {}
    return _KEGG_MAPPING

def get_subsystems(cobra_model):
    """
    Extracts a dictionary of {subsystem_name: [list of reactions]}
    Priorities:
    1. Explicit BiGG 'subsystem' attribute.
    2. KEGG Pathway annotation (Biological Decomposition).
    3. Louvain Community Detection (Structural Decomposition).
    """
    subsystems = {}
    uncategorized_rxns = []
    
    for rxn in cobra_model.reactions:
        # Some models use 'subsystem', others might be different. BiGG uses 'subsystem'.
        sub = getattr(rxn, 'subsystem', 'Uncategorized')
        if not sub: sub = 'Uncategorized'
        # Some BiGG models define empty strings or "Uncategorized" explicitly
        
        if sub == 'Uncategorized' or sub == '':
             uncategorized_rxns.append(rxn)
             continue
             
        if sub not in subsystems:
            subsystems[sub] = []
        subsystems[sub].append(rxn)
        
    # If we found explicit subsystems, but also have Uncategorized, add them
    if subsystems and uncategorized_rxns:
         # Optional: Try to map uncategorized ones via KEGG? 
         # For now, just keep them as "Uncategorized" as BiGG intended, unless user wants full re-annotation.
         # But the user asked to FIX bad layouts. So let's keep it simple: if valid subsystems exist, use them.
         subsystems['Uncategorized'] = uncategorized_rxns
         
    # FALLBACK: If NO explicit subsystems found (or only Uncategorized), 
    # run Biological Decomposition (KEGG) then Structural (Louvain).
    if len(subsystems) == 0 and uncategorized_rxns:
         print("  No explicit subsystems. Attempting Biological Decomposition (KEGG)...")
         kegg_map = load_kegg_mapping()
         
         biological_subsystems = {}
         remaining_uncategorized = []
         
         for rxn in uncategorized_rxns:
             # Check annotations
             # Priority 1: Direct KEGG Reaction ID
             kegg_ids = []
             if 'kegg.reaction' in rxn.annotation:
                 val = rxn.annotation['kegg.reaction']
                 if isinstance(val, list): kegg_ids.extend(val)
                 else: kegg_ids.append(val)
                 
             # Priority 2: EC Number (Enzyme Commission)
             # This is a strong proxy for function
             if not kegg_ids and 'ec-code' in rxn.annotation:
                 val = rxn.annotation['ec-code']
                 if isinstance(val, list): kegg_ids.extend(val)
                 else: kegg_ids.append(val)
                 
             # Future: could add 'rhea', 'metanetx.reaction' if we had mappings for them
             
             assigned = False
             if kegg_map:
                 for kid in kegg_ids:
                     # Clean up ID (sometimes 1.1.1.1 is stored as simple string)
                     if kid in kegg_map:
                         # Found a pathway!
                         pathways = kegg_map[kid]
                         if pathways:
                             primary_pathway = pathways[0]
                             if primary_pathway not in biological_subsystems:
                                 biological_subsystems[primary_pathway] = []
                             biological_subsystems[primary_pathway].append(rxn)
                             assigned = True
                             break 
             
             if not assigned:
                 remaining_uncategorized.append(rxn)
                 
         # Evaluate Biological Decomposition
         if len(biological_subsystems) > 1:
             print(f"  Success: Decomposed into {len(biological_subsystems)} KEGG pathways.")
             subsystems = biological_subsystems
             if remaining_uncategorized:
                 subsystems['Uncategorized'] = remaining_uncategorized
                 
         else:
            print("  KEGG Decomposition failed (insufficient annotations/mapping).")
            print("  Running Structural Decomposition (Community Detection)...")
            
            # --- STRUCTURAL DECOMPOSITION (Louvain) ---
            
            # Build temp graph for decomposition
            G_temp = nx.Graph()
            rxn_map = {r.id: r for r in uncategorized_rxns}
            
            for r in uncategorized_rxns:
                 G_temp.add_node(r.id, type='reaction')
                 for m in r.metabolites:
                     if not is_currency(m.id): # Skip hubs for clustering
                         G_temp.add_node(m.id, type='metabolite')
                         G_temp.add_edge(r.id, m.id)
            
            # Project to Reaction-Reaction Graph ?
            try:
                 from networkx.algorithms.community import greedy_modularity_communities
                 communities = greedy_modularity_communities(G_temp)
                 
                 print(f"  Detected {len(communities)} structural communities.")
                 
                 for i, comm in enumerate(communities):
                      comm_rxns = []
                      for node_id in comm:
                          if node_id in rxn_map:
                               comm_rxns.append(rxn_map[node_id])
                      
                      if comm_rxns:
                          subsystems[f"Cluster_{i}"] = comm_rxns
                          
            except Exception as e:
                 print(f"  Community Detection Failed: {e}. Fallback to single block.")
                 subsystems['Uncategorized'] = uncategorized_rxns


    return subsystems


def split_large_subsystems(subsystems, max_size=60, cobra_model=None):
    """
    Recursively splits subsystems larger than max_size using community detection.
    """
    new_subsystems = {}
    
    for name, rxns in subsystems.items():
        if len(rxns) <= max_size:
            new_subsystems[name] = rxns
            continue
            
        print(f"  Splitting large cluster '{name}' ({len(rxns)} reactions > {max_size})...")
        
        # Build temp graph for this subsystem
        G_temp = nx.Graph()
        rxn_map = {r.id: r for r in rxns}
        
        for r in rxns:
             G_temp.add_node(r.id, type='reaction')
             for m in r.metabolites:
                 if not is_currency(m.id): 
                     G_temp.add_node(m.id, type='metabolite')
                     G_temp.add_edge(r.id, m.id)
                     
        try:
             from networkx.algorithms.community import greedy_modularity_communities
             # split into 2 or more communities
             communities = greedy_modularity_communities(G_temp, cutoff=1, best_n=None) 
             # Note: greedy_modularity_communities returns a list of sets of nodes
             
             # Map back to reactions
             sub_clusters = []
             for comm in communities:
                  c_rxns = []
                  for node in comm:
                      if node in rxn_map:
                          c_rxns.append(rxn_map[node])
                  if c_rxns:
                      sub_clusters.append(c_rxns)
                      
             # If split failed to reduce size (e.g. one giant component), force split?
             # For now, just add them as separate clusters
             if len(sub_clusters) > 1:
                 for i, c_rxns in enumerate(sub_clusters):
                     # Recursive check? For now assume one split is enough or it will be caught next pass if we looped?
                     # Let's just name them and trust they are better.
                     new_subsystems[f"{name}_{i}"] = c_rxns
             else:
                 # Could not split
                 new_subsystems[name] = rxns
                 
        except Exception as e:
             print(f"  Split failed for {name}: {e}")
             new_subsystems[name] = rxns
             
    return new_subsystems


def enforce_cluster_limits(subsystems, min_size=6, max_clusters=15, max_rxn_size=60, cobra_model=None):
    """
    Enforces constraints:
    1. Max Size 60 (Split)
    2. Min Size 6 (Merge)
    3. Max Count 15 (Cap)
    """
    if len(subsystems) <= 1:
        return subsystems 
        
    # 0. Pre-process: Merge strict 'Uncategorized' if small? 
    # Actually, step 1 is SPLIT.
    subsystems = split_large_subsystems(subsystems, max_size=max_rxn_size, cobra_model=cobra_model)
        
    # 1. Inspect sizes for Merging
    uncat_key = 'Uncategorized'
    
    valid_clusters = [] 
    small_clusters = [] 
    final_uncat_rxns = []
    
    if uncat_key in subsystems:
        final_uncat_rxns.extend(subsystems[uncat_key])

    for name, rxns in subsystems.items():
        if name == uncat_key: continue
        
        if len(rxns) < min_size:
            small_clusters.append((name, rxns))
        else:
            valid_clusters.append((name, rxns))
            
    # 2. Merge small clusters
    if small_clusters:
        for name, rxns in small_clusters:
            final_uncat_rxns.extend(rxns)
            
    # 3. Check Max Cluster Count
    has_uncat = len(final_uncat_rxns) > 0
    max_named_slots = max_clusters - 1 if has_uncat else max_clusters
    
    excess_clusters = []
    if len(valid_clusters) > max_named_slots:
        valid_clusters.sort(key=lambda x: len(x[1]), reverse=True)
        
        keep = valid_clusters[:max_named_slots]
        excess = valid_clusters[max_named_slots:]
        
        valid_clusters = keep
        excess_clusters = excess
        
        print(f"  Cap extended to {max_clusters}. Merging {len(excess_clusters)} excess clusters.")
        for name, rxns in excess_clusters:
            final_uncat_rxns.extend(rxns)
            
    # Rebuild Dictionary
    new_subsystems = {}
    for name, rxns in valid_clusters:
        new_subsystems[name] = rxns
        
    if final_uncat_rxns:
        new_subsystems[uncat_key] = final_uncat_rxns
        
    return new_subsystems

def build_subsystem_graph(cobra_model, rxn_list):
    """
    Builds a bipartite graph for JUST the specified reactions.
    Metabolites that connect to reactions OUTSIDE this subsystem are marked as 'external'.
    """
    G = nx.Graph()
    subset_rxn_ids = set([r.id for r in rxn_list])
    
    for rxn in rxn_list:
        if not rxn.id: continue
        
        # Add Reaction Node
        G.add_node(rxn.id, type='reaction', name=rxn.name)
        
        # Add Metabolites & Edges
        for met, coeff in rxn.metabolites.items():
            met_id = met.id
            if not G.has_node(met_id):
                G.add_node(met_id, type='metabolite', name=met.name)
            
            # Add Edge
            G.add_edge(rxn.id, met_id)
            
            # Check if this metabolite connects to OTHER subsystems
            # If so, mark it as a potential peripheral input/output
            # This is implicit: if it has edges in the full model that aren't in this subgraph.
            # We can check met.reactions
            
            # is_boundary = False
            # for r in met.reactions:
            #     if r.id not in subset_rxn_ids:
            #         is_boundary = True
            #         break
            # if is_boundary:
            #     G.nodes[met_id]['boundary'] = True
                
    return G

# --- Layout Optimization Functions (Reused) ---

def calculate_energy(G, pos, exchange_nodes=None):
    total_energy = 0.0
    if exchange_nodes is None: exchange_nodes = set()
    
    # 1. Orthogonality & Edge Length
    ortho_penalty = 0.0
    length_penalty = 0.0
    epsilon = 0.02 # Tighter tolerance
    
    for u, v in G.edges:
        p1 = pos[u]
        p2 = pos[v]
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Penalize Area (Manhattan Preference)
        # Minimize dx * dy. If one is 0, penalty is 0.
        ortho_penalty += (dx * dy) 
        
        length_penalty += dist
        
    num_edges = G.number_of_edges()
    if num_edges > 0:
        ortho_penalty /= num_edges
        length_penalty /= num_edges
        
    # 2. Overlap & Repulsion
    nodes = list(G.nodes)
    num_nodes = len(nodes)
    overlap_penalty = 0.0
    repulsion_energy = 0.0
    min_dist = 0.12 # Increase minimum distance
    
    sample_size = min(num_nodes * 20, 5000) 
    
    for _ in range(sample_size):
        idx1 = random.randint(0, num_nodes-1)
        idx2 = random.randint(0, num_nodes-1)
        if idx1 == idx2: continue
        
        n1 = nodes[idx1]
        n2 = nodes[idx2]
        p1 = pos[n1]
        p2 = pos[n2]
        
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dist_sq = dx*dx + dy*dy
        
        if dist_sq < min_dist*min_dist:
            overlap_penalty += 1.0
            
        safe_dist_sq = max(dist_sq, 0.0001)
        repulsion_energy += 0.01 / safe_dist_sq

    # 3. Peripheral Force
    peripheral_energy = 0.0
    if exchange_nodes:
        for node in exchange_nodes:
            if node in pos: 
                p = pos[node]
                dist_to_edge = min(p[0], 1.0 - p[0], p[1], 1.0 - p[1])
                peripheral_energy += dist_to_edge
        peripheral_energy /= len(exchange_nodes)

    w_ortho = 500.0 # Increase weight significantly
    w_len = 2.0
    w_overlap = 5000.0 # Strict overlap avoidance
    w_repul = 20.0 
    w_peripheral = 200.0
    
    total_energy = (ortho_penalty * w_ortho) + \
                   (length_penalty * w_len) + \
                   (overlap_penalty * w_overlap) - \
                   (repulsion_energy * w_repul) + \
                   (peripheral_energy * w_peripheral)
                   
    return total_energy

def get_cycle_nodes(G):
    try:
        cycles = nx.cycle_basis(G)
    except nx.NetworkXNotImplemented:
        return set(), {}

    relevant_cycles = [c for c in cycles if 4 < len(c) < 20]
    relevant_cycles.sort(key=len, reverse=True)
    
    fixed_pos = {}
    locked_nodes = set()
    
    slots = [
        ((0.5, 0.5), 0.25),
        ((0.2, 0.2), 0.15),
        ((0.8, 0.2), 0.15),
        ((0.2, 0.8), 0.15),
        ((0.8, 0.8), 0.15)
    ]
    slot_idx = 0
    
    for cycle in relevant_cycles:
        cycle_set = set(cycle)
        if not cycle_set.isdisjoint(locked_nodes):
            continue
            
        if slot_idx >= len(slots): break
        center, radius = slots[slot_idx]
        slot_idx += 1
        
        n = len(cycle)
        for i, node in enumerate(cycle):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            fixed_pos[node] = np.array([x, y])
            locked_nodes.add(node)
            
    return locked_nodes, fixed_pos

def optimize_layout_sa(G, steps=1000):
    # Determine "Peripherals" for this subsystem
    # 1. Exchange Reactions (EX_)
    # 2. Metabolites that are 'boundary' (connect to outside) - difficult to pass here without context
    # For now, stick to EX_ and DM_
    exchange_nodes = set()
    for n, data in G.nodes(data=True):
        if data.get('type') == 'reaction' and (n.startswith("EX_") or n.startswith("DM_")):
            exchange_nodes.add(n)

    # Detect Cycles
    cycle_nodes, fixed_positions = get_cycle_nodes(G)
    
    pos = {n: np.array([random.random(), random.random()]) for n in G.nodes}
    for n, p in fixed_positions.items():
        pos[n] = p
        
    for n in exchange_nodes:
        if n not in fixed_positions: 
            if random.random() < 0.5:
                 pos[n][0] = random.choice([0.05, 0.95])
            else:
                 pos[n][1] = random.choice([0.05, 0.95])
    
    current_energy = calculate_energy(G, pos, exchange_nodes)
    best_energy = current_energy
    best_pos = copy.deepcopy(pos)
    temp = 1.0
    cooling_rate = 0.99
    nodes_list = list(G.nodes)
    
    print(f"    Optimizing ({len(nodes_list)} nodes)...")
    
    for i in range(steps):
        candidate_pos = copy.deepcopy(pos)
        n_perturb = max(1, len(nodes_list) // 10)
        
        for _ in range(n_perturb):
             node = random.choice(nodes_list)
             if node in fixed_positions: continue
             delta = np.random.normal(0, 0.05 * temp, 2)
             candidate_pos[node] = np.clip(candidate_pos[node] + delta, 0, 1)
             
        new_energy = calculate_energy(G, candidate_pos, exchange_nodes)
        if new_energy - current_energy < 0 or random.random() < math.exp(-(new_energy - current_energy) / (temp + 1e-9)):
            pos = candidate_pos
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = copy.deepcopy(pos)
        temp *= cooling_rate
            
    return best_pos


def build_meta_graph(subsystems, cobra_model):
    """
    Constructs a graph where nodes = subsystem names, 
    edges = weight (number of shared metabolites).
    """
    meta_graph = nx.Graph()
    sub_names = list(subsystems.keys())
    
    # Add nodes
    for name in sub_names:
        meta_graph.add_node(name)
        
    # Add edges
    # This is O(N^2) on subsystems (usually < 20), so it's fast.
    for i in range(len(sub_names)):
        for j in range(i + 1, len(sub_names)):
            sub1 = sub_names[i]
            sub2 = sub_names[j]
            
            rxns1 = subsystems[sub1]
            rxns2 = subsystems[sub2]
            
            # Get metabolites for each
            mets1 = set()
            for r in rxns1: mets1.update([m.id for m in r.metabolites])
            
            mets2 = set()
            for r in rxns2: mets2.update([m.id for m in r.metabolites])
            
            # Intersection (excluding currency)
            shared = mets1.intersection(mets2)
            non_currency_shared = [m for m in shared if not is_currency(m)]
            
            weight = len(non_currency_shared)
            if weight > 0:
                meta_graph.add_edge(sub1, sub2, weight=weight)
                
    return meta_graph

def optimize_meta_layout(meta_graph):
    """
    Arranges subsystems on a grid.
    Returns dict: {subsystem_name: (row, col)}
    """
    # Use spectral layout or Kamada-Kawai to get relative positions
    if len(meta_graph.nodes) == 0:
        return {}
        
    pos = nx.kamada_kawai_layout(meta_graph)
    
    # Snap to Grid
    # Normalize to [0, N]
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    
    if not xs: return {}
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = max_x - min_x + 1e-9
    height = max_y - min_y + 1e-9
    
    # Scale to integer grid approx equal to sqrt(N)
    n = len(meta_graph.nodes)
    grid_dim = int(math.ceil(math.sqrt(n))) # Tight grid (was n*2)
    
    grid_pos = {}
    occupied = set()
    
    sorted_nodes = sorted(meta_graph.nodes, key=lambda n: meta_graph.degree(n), reverse=True)
    
    # Quantitative scaling
    for node in sorted_nodes:
        # Scale to grid
        norm_x = (pos[node][0] - min_x) / width
        norm_y = (pos[node][1] - min_y) / height
        
        c = int(round(norm_x * (grid_dim - 1)))
        r = int(round(norm_y * (grid_dim - 1)))
        
        # Spiraling search for nearest empty slot if occupied
        # (BFS on grid)
        queue = [(r, c)]
        visited_search = set([(r, c)])
        found = False
        
        while queue:
            curr_r, curr_c = queue.pop(0)
            if (curr_r, curr_c) not in occupied:
                grid_pos[node] = (curr_r, curr_c)
                occupied.add((curr_r, curr_c))
                found = True
                break
                
    # --- GRID COMPRESSION STEP (Gravity) ---
    # Iteratively move tiles UP and LEFT to fill gaps (Tetris style)
    if not grid_pos: return {}
    
    # Convert to list of mutable objects
    items = []
    for node, (r, c) in grid_pos.items():
        items.append({'node': node, 'r': r, 'c': c})
        
    changed = True
    while changed:
        changed = False
        # Sort by position (top-left priority)
        items.sort(key=lambda x: (x['r'], x['c']))
        
        # Build occupancy map
        occupied_set = set((i['r'], i['c']) for i in items)
        
        for item in items:
            # Try move UP
            current_r, current_c = item['r'], item['c']
            if current_r > 0:
                target_r = current_r - 1
                if (target_r, current_c) not in occupied_set:
                    # Move!
                    occupied_set.remove((current_r, current_c))
                    occupied_set.add((target_r, current_c))
                    item['r'] = target_r
                    changed = True
                    continue # Moved, check next
                    
            # Try move LEFT
            if current_c > 0:
                target_c = current_c - 1
                if (current_r, target_c) not in occupied_set:
                    # Move!
                    occupied_set.remove((current_r, current_c))
                    occupied_set.add((current_r, target_c))
                    item['c'] = target_c
                    changed = True
    
    # Re-normalize to 0-index just in case
    min_r = min(x['r'] for x in items)
    min_c = min(x['c'] for x in items)
    
    final_pos = {}
    for item in items:
        final_pos[item['node']] = (item['r'] - min_r, item['c'] - min_c)
        
    # Calculate simplified dimensions for debug
    final_rows = set(r for r, c in final_pos.values())
    final_cols = set(c for r, c in final_pos.values())
    print(f"  Compacted Grid: {len(final_rows)} rows, {len(final_cols)} cols")
    
    return final_pos

def main():
    print(f"Loading Models from {MODEL_DIR}")
    # Prefer JSON if available, else XML
    files = glob.glob(os.path.join(MODEL_DIR, "*.json")) + glob.glob(os.path.join(MODEL_DIR, "*.xml"))
    
    unique_ids = set()
    
    for f in files:
        file_id = os.path.splitext(os.path.basename(f))[0]
        # if file_id != 'e_coli_core': continue # ONLY TEST E. COLI CORE (Disabled for Batch Processing)
        
        if file_id in unique_ids: continue
        unique_ids.add(file_id)
        
        # Create folder for this model
        model_out_dir = os.path.join(OUTPUT_DIR, file_id)
        os.makedirs(model_out_dir, exist_ok=True)
            
        print(f"Processing Model: {file_id} ({f})")
        
        try:
            if f.endswith('.json'):
                cobra_model = cobra.io.load_json_model(f)
            else:
                cobra_model = load_sbml_model(f)
                
            if not cobra_model: continue
            
            # 1. Get Subsystems
            subsystems = get_subsystems(cobra_model)
        
            # Enforce User Constraints
            subsystems = enforce_cluster_limits(subsystems)
        
            print(f"  Found {len(subsystems)} subsystems.")
            
            # Allow single-subsystem models (e.g. just Uncategorized)
            if len(subsystems) == 0:
                 print("  Warning: Strictly NO subsystems detected (empty model?). Skipping.")
                 continue

            # 2. Build Meta-Graph & Layout
            print("  Building Meta-Graph...")
            meta_graph = build_meta_graph(subsystems, cobra_model)
            tile_positions = optimize_meta_layout(meta_graph)
            
            # 3. Process Each Subsystem & Calculate Bounding Boxes
            subsystem_layouts = {}  # {sub_name: {'coords': {}, 'graph': G, 'bbox': (w, h)}}
            
            for sub_name, rxns in subsystems.items():
                # Only skip "Uncategorized" if we have other valid subsystems to show.
                # If "Uncategorized" is the ONLY bucket, we MUST show it.
                if sub_name == "Uncategorized" and len(subsystems) > 1:
                    continue

                print(f"  > Subsystem: {sub_name} ({len(rxns)} reactions)")
                
                # Build Graph
                G_full = build_subsystem_graph(cobra_model, rxns)
                if len(G_full.nodes) < 2: continue
                
                # Identify Currency
                currency_nodes = []
                for n in G_full.nodes():
                    if is_currency(n): currency_nodes.append(n)
                
                # Backbone Optimize
                G_backbone = G_full.copy()
                G_backbone.remove_nodes_from(currency_nodes)
                
                normalized_pos = {}
                if len(G_backbone.nodes) > 0:
                    normalized_pos = optimize_layout_sa(G_backbone, steps=1000)
                    
                    # --- MANHATTAN ENFORCEMENT ---
                    # Strict post-processing to force horizontal/vertical alignment
                    # effectively "snapping" the SA result to a virtual grid
                    normalized_pos = snap_to_grid(normalized_pos, G_backbone, grid_spacing=0.2) 
                    
                    # Normalization Fix: snap_to_grid adds +2500 offset. We must remove it 
                    # to keep coords in relative [0, ~1] space for the multiplier below.
                    # Otherwise, 2500 * 5 = 12500, pushing everything off-canvas.
                    xs = [p[0] for p in normalized_pos.values()]
                    ys = [p[1] for p in normalized_pos.values()]
                    if xs:
                        min_x, min_y = min(xs), min(ys)
                        for n in normalized_pos:
                            normalized_pos[n] = (normalized_pos[n][0] - min_x, normalized_pos[n][1] - min_y)
                
                # Local Coords in [0, 1] space -> Expand to [0, 10] effectively
                # Force nodes APART so edge length > node size
                # Force nodes APART so edge length > node size
                pos_multiplier = 5.0  
                
                local_coords = {}
                
                # Backbone
                for n, pos in normalized_pos.items():
                    local_coords[n] = (pos[0] * pos_multiplier, pos[1] * pos_multiplier)
                    
                # Hubs (in normalized space)
                for n, data in list(G_full.nodes(data=True)):
                    if data.get('type') == 'reaction':
                        if n not in local_coords: continue
                        rxn_pos = local_coords[n]
                        neighbors = list(G_full.neighbors(n))
                        
                        dup_count = 0 
                        for met in neighbors:
                            if met in currency_nodes:
                                dup_id = f"{met}__dup_{sub_name}_{dup_count}"
                                dup_count += 1
                                angle = random.random() * 2 * math.pi
                                # Increased offset significantly so duplicates don't overlap reaction node
                                offset = 0.25 * pos_multiplier 
                                dx = math.cos(angle) * offset
                                dy = math.sin(angle) * offset
                                local_coords[dup_id] = (rxn_pos[0] + dx, rxn_pos[1] + dy)
                                
                                G_full.add_node(dup_id, original_id=met, type='metabolite', name=met)
                                G_full.add_edge(n, dup_id)
                                
                # CRITICAL Fix for KeyError: Remove original currency nodes from G_full
                # They have been replaced by duplicates and have no coords in local_coords
                if currency_nodes:
                    G_full.remove_nodes_from(currency_nodes)

                # Calculate actual bounding box
                # Calculate actual bounding box
                if local_coords:
                    xs = [c[0] for c in local_coords.values()]
                    ys = [c[1] for c in local_coords.values()]
                    bbox_w = max(xs) - min(xs) + 0.1  # Add padding
                    bbox_h = max(ys) - min(ys) + 0.1
                else:
                    bbox_w, bbox_h = 0.5, 0.5
                
                subsystem_layouts[sub_name] = {
                    'coords': local_coords,
                    'graph': G_full,
                    'bbox': (bbox_w, bbox_h)
                }
                
                # --- SAVE INDIVIDUAL MAP ---
                # Sanitize Name
                safe_name = sub_name.replace('/', '_').replace(' ', '_').replace(':', '').replace('\\', '_')
                sub_out_path = os.path.join(model_out_dir, f"{file_id}_{safe_name}.json")
                
                # We need to SNAP the local_coords to grid for export
                # Or just export floating point? Escher handles float.
                # But our serialize expects 'snapped' dict-like nodes?
                # Actually serialize_results takes (graph, coords). 
                # Coords dict: {id: (x,y)}.
                # Let's verify serialize_results can handle floats. Yes, Escher floats are fine.
                # But we might want to scale up from small [0, 5] range to Escher [0, 5000].
                # The 'pos_multiplier' was 5.0. 
                # Let's scale up by 100 for visibility in individual map.
                
                export_coords = {}
                export_coords = {}
                for nid, (x, y) in local_coords.items():
                    # Scale to match Combined Map's spacious usage (800x)
                    export_coords[nid] = (x * 800.0, y * 800.0)
                    
                serialize_results(G_full, export_coords, sub_out_path, author="Tianyu Wu (GitHub: forxhunter)", center_map=True)
                # print(f"    Saved individual map: {safe_name}")
                # ---------------------------
            
            # 4. Calculate Variable Grid Sizing
            # We want uniform visual density, so we use a CONSTANT pixel multiplier per unit.
            # Local coords are roughly [0, 5] (due to pos_multiplier=5.0)
            # 1 unit = 300px seems reasonable for edge length
            # So 0.25 (offset) * 300 = 75px
            
            # So 0.25 (offset) * 300 = 75px
            
            PIXELS_PER_UNIT = 800.0 # visual scale factor
            
            # Step A: Calculate Pixel Dimensions for each subsystem
            sub_dims = {}
            for sub_name, layout in subsystem_layouts.items():
                w, h = layout['bbox']
                pixel_w = w * PIXELS_PER_UNIT
                pixel_h = h * PIXELS_PER_UNIT
                sub_dims[sub_name] = (pixel_w, pixel_h)
                
            # Step B: Determine Row Heights and Col Widths
            # Map grid (r, c) -> max dimensions
            rows = set()
            cols = set()
            
            grid_content = {} # (r, c) -> sub_name
            
            for sub_name, (r, c) in tile_positions.items():
                if sub_name not in sub_dims: continue # Was filtered out
                rows.add(r)
                cols.add(c)
                grid_content[(r, c)] = sub_name
            
            sorted_rows = sorted(list(rows))
            sorted_cols = sorted(list(cols))
            
            row_heights = {}
            col_widths = {}
            
            PADDING = 1000.0 # Buffer between tiles
            
            for r in sorted_rows:
                max_h = 100.0
                for c in sorted_cols:
                    sub_name = grid_content.get((r, c))
                    if sub_name:
                         max_h = max(max_h, sub_dims[sub_name][1])
                row_heights[r] = max_h + PADDING
                
            for c in sorted_cols:
                max_w = 100.0
                for r in sorted_rows:
                    sub_name = grid_content.get((r, c))
                    if sub_name:
                         max_w = max(max_w, sub_dims[sub_name][0])
                col_widths[c] = max_w + PADDING
            
            # Step C: Calculate Grid Offsets
            # row_y[r] = sum(row_heights[0..r-1])
            row_y = {}
            curr_y = 0.0
            for r in sorted_rows:
                row_y[r] = curr_y
                curr_y += row_heights[r]
                
            col_x = {}
            curr_x = 0.0
            for c in sorted_cols:
                col_x[c] = curr_x
                curr_x += col_widths[c]
                
            # 5. Merge with Variable Grid Positioning
            full_combined_nodes = {}
            full_combined_graph = nx.Graph()
            
            for sub_name, layout_data in subsystem_layouts.items():
                if sub_name not in tile_positions: continue
                
                local_coords = layout_data['coords']
                G_full = layout_data['graph']
                bbox_w, bbox_h = layout_data['bbox']
                
                # Get grid position
                tr, tc = tile_positions[sub_name]
                
                # Global Position uses the computed variable grid offsets
                gx_start = col_x[tc]
                gy_start = row_y[tr]
                
                # Cell Dimensions
                cell_w = col_widths[tc] - PADDING # Actual content area
                cell_h = row_heights[tr] - PADDING
                
                # Center the subsystem within its variable cell
                my_pixel_w = sub_dims[sub_name][0]
                my_pixel_h = sub_dims[sub_name][1]
                
                center_offset_x = (cell_w - my_pixel_w) / 2
                center_offset_y = (cell_h - my_pixel_h) / 2
                
                global_offset_x = gx_start + center_offset_x
                global_offset_y = gy_start + center_offset_y
                
                scale_factor = PIXELS_PER_UNIT # Uniform scale for consistent density!
                
                safe_sub = "".join([c if c.isalnum() else "" for c in sub_name])
                local_id_map = {}

                # Calculate local min offsets to normalize to (0,0)
                lxs = [c[0] for c in local_coords.values()]
                lys = [c[1] for c in local_coords.values()]
                local_min_x = min(lxs)
                local_min_y = min(lys)

                # 1. Transfer Nodes & Layout with Scaling
                for n, coords in local_coords.items():
                    # Generate Unique Global ID
                    if "__dup_" in n:
                        unique_global_id = n 
                    else:
                        unique_global_id = f"{n}_{safe_sub}"
                    
                    local_id_map[n] = unique_global_id
                    
                    # Normalize first! (shift to 0,0)
                    norm_x = coords[0] - local_min_x
                    norm_y = coords[1] - local_min_y
                    
                    # Transform Coordinates with SCALING
                    gx = norm_x * scale_factor + global_offset_x
                    gy = norm_y * scale_factor + global_offset_y
                    
                    # Store as Tuple (x, y) for snap_to_grid
                    full_combined_nodes[unique_global_id] = (gx, gy)
                    
                    # Add Node to Global Graph
                    if n in G_full.nodes:
                         data = G_full.nodes[n].copy()
                         data['id'] = unique_global_id
                         full_combined_graph.add_node(unique_global_id, **data)

                # 2. Transfer Edges
                for u, v in G_full.edges:
                    node_u = None
                    node_v = None
                    
                    if u in local_id_map:
                         node_u = local_id_map[u]
                    
                    if v in local_id_map:
                         node_v = local_id_map[v]
                         
                    if node_u and node_v:
                        full_combined_graph.add_edge(node_u, node_v)

            # 6. Final Global Scaling Checks
            # We don't force 80% occupancy anymore because we packed it tightly based on content.
            # But we should ensure positive coordinates and maybe a nice margin.
            if full_combined_nodes:
                xs = [c[0] for c in full_combined_nodes.values()]
                ys = [c[1] for c in full_combined_nodes.values()]
                
                min_x = min(xs)
                min_y = min(ys)
                
                # Shift to (500, 500)
                shift_x = 500 - min_x
                shift_y = 500 - min_y
                
                shifted_nodes = {}
                for node_id, (x, y) in full_combined_nodes.items():
                     shifted_nodes[node_id] = (x + shift_x, y + shift_y)
                full_combined_nodes = shifted_nodes

            # Export Combined
            combined_out = os.path.join(model_out_dir, f"{file_id}_Combined.json")
            if full_combined_nodes:
                snapped = snap_to_grid(full_combined_nodes, full_combined_graph, grid_spacing=100)
                serialize_results(full_combined_graph, snapped, combined_out, author="Tianyu Wu (GitHub: forxhunter)")
                print(f"  Saved Combined Meta-Map to {combined_out}")
            else:
                print("  Warning: Empty combined map.")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
