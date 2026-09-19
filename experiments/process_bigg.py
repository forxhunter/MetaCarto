import os
import glob
import torch
import numpy as np
import networkx as nx
import copy
import random
import math

from src.parsing import load_sbml_model, build_bipartite_graph
from experiments.gnn_model import get_model
from src.refinement import snap_to_grid
from src.export import serialize_results

# Parameters
MODEL_DIR = "data/bigg/test_model"
OUTPUT_DIR = "data/bigg"
HUB_DEGREE_THRESHOLD = 15

def get_node_features(G):
    x = []
    for n, data in G.nodes(data=True):
        feat = [1.0] if data.get('type') == 'reaction' else [0.0]
        x.append(feat + [0.0]*9)
    return torch.tensor(x, dtype=torch.float)

def get_edge_index(G):
    node_map = {n: i for i, n in enumerate(G.nodes)}
    edge_index = []
    for u, v in G.edges:
        edge_index.append([node_map[u], node_map[v]])
        edge_index.append([node_map[v], node_map[u]])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def calculate_energy(G, pos, exchange_nodes=None):
    """
    Energy function for Simulated Annealing.
    Encourages: Orthogonality, Spacing, Non-Overlap, Peripheral Exchanges.
    """
    total_energy = 0.0
    if exchange_nodes is None: exchange_nodes = set()
    
    # 1. Orthogonality & Edge Length
    ortho_penalty = 0.0
    length_penalty = 0.0
    epsilon = 0.05
    
    for u, v in G.edges:
        p1 = pos[u]
        p2 = pos[v]
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        dist = math.sqrt(dx*dx + dy*dy)
        if dx > epsilon and dy > epsilon:
             ortho_penalty += 1.0 
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
    min_dist = 0.08 # ~400px on 5000px canvas 
    
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

    # 3. Peripheral Force for Exchange Reactions
    peripheral_energy = 0.0
    if exchange_nodes:
        for node in exchange_nodes:
            if node in pos: # Might have been filtered if hub? Unlikely for EX
                p = pos[node]
                # Distance to nearest edge (0 or 1 in x or y)
                dist_to_edge = min(p[0], 1.0 - p[0], p[1], 1.0 - p[1])
                # We want to MINIMIZE this distance (push to edge)
                peripheral_energy += dist_to_edge
        # Average it
        peripheral_energy /= len(exchange_nodes)

    # Weights
    w_ortho = 100.0
    w_len = 2.0
    w_overlap = 1000.0 # Strict No-Overlap
    w_repul = 10.0 # Force Spacing
    w_peripheral = 200.0 # Strong force to edges
    
    total_energy = (ortho_penalty * w_ortho) + \
                   (length_penalty * w_len) + \
                   (overlap_penalty * w_overlap) - \
                   (repulsion_energy * w_repul) + \
                   (peripheral_energy * w_peripheral)
                   
    return total_energy


def get_cycle_nodes(G):
    """
    Detects distinct cycles (e.g. TCA, Pentose Phosphate).
    Returns set of all locked nodes and their fixed positions.
    """
    try:
        cycles = nx.cycle_basis(G)
    except nx.NetworkXNotImplemented:
        return set(), {}

    # 1. Filter useful cycles
    relevant_cycles = [c for c in cycles if 4 < len(c) < 20]
    relevant_cycles.sort(key=len, reverse=True) # Largest first
    
    fixed_pos = {}
    locked_nodes = set()
    
    # 2. Place Cycles
    # Strategy: Place largest cycle in center. 
    # Place subsequent cycles in corners if they don't share nodes.
    
    # Pre-defined "slots" for cycles: Center, Top-Left, Top-Right, Bot-Left, Bot-Right
    slots = [
        ((0.5, 0.5), 0.20), # Center, Radius
        ((0.2, 0.2), 0.15),
        ((0.8, 0.2), 0.15),
        ((0.2, 0.8), 0.15),
        ((0.8, 0.8), 0.15)
    ]
    slot_idx = 0
    
    for cycle in relevant_cycles:
        # Check orthogonality/overlap with already locked nodes
        cycle_set = set(cycle)
        if not cycle_set.isdisjoint(locked_nodes):
            # This cycle shares nodes with an already locked cycle.
            # Skip valid strict locking to avoid conflict. 
            # (Or merge? Merging is hard. Skipping lets SA handle the bridge nodes)
            continue
            
        if slot_idx >= len(slots):
            break
            
        center, radius = slots[slot_idx]
        slot_idx += 1
        
        n = len(cycle)
        for i, node in enumerate(cycle):
            angle = 2 * math.pi * i / n
            # Start angle -90 to put first node at top?
            angle -= math.pi / 2
            
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            fixed_pos[node] = np.array([x, y])
            locked_nodes.add(node)
            
    return locked_nodes, fixed_pos

def optimize_layout_sa(G, steps=1000):
    """
    Simulated Annealing optimization for the backbone graph.
    """
    # 1. Identify Exchange Nodes
    exchange_nodes = set()
    for n, data in G.nodes(data=True):
        if data.get('type') == 'reaction' and (n.startswith("EX_") or n.startswith("DM_")):
            exchange_nodes.add(n)

    # 2. Detect & Lock Cycles
    cycle_nodes, fixed_positions = get_cycle_nodes(G)
    if cycle_nodes:
        print(f"    Locked {len(cycle_nodes)} nodes in main cycle (e.g. {list(cycle_nodes)[:3]}).")

    # 3. Initialize Positions
    pos = {n: np.array([random.random(), random.random()]) for n in G.nodes}
    
    # Apply Fixed Positions (Cycle)
    for n, p in fixed_positions.items():
        pos[n] = p
        
    # Pre-place exchanges at edges
    for n in exchange_nodes:
        if n not in fixed_positions: # Don't move if valid cycle (unlikely for EX)
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
    
    print(f"  > Optimizing Backbone ({len(nodes_list)} nodes) for {steps} steps...")
    
    for i in range(steps):
        candidate_pos = copy.deepcopy(pos)
        n_perturb = max(1, len(nodes_list) // 10)
        
        for _ in range(n_perturb):
             node = random.choice(nodes_list)
             
             # CRITICAL: Do NOT move fixed cycle nodes
             if node in fixed_positions:
                 continue
                 
             delta = np.random.normal(0, 0.05 * temp, 2)
             candidate_pos[node] = np.clip(candidate_pos[node] + delta, 0, 1)
             
        new_energy = calculate_energy(G, candidate_pos, exchange_nodes)
        delta_E = new_energy - current_energy
        if delta_E < 0:
            accept = True
        else:
            prob = math.exp(-delta_E / (temp + 1e-9))
            accept = random.random() < prob
            
        if accept:
            pos = candidate_pos
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = copy.deepcopy(pos)
        temp *= cooling_rate
        
        if i % 100 == 0:
            print(f"    Step {i}: Energy {current_energy:.4f} (Temp {temp:.4f})")
            
    return best_pos


# Common currency metabolites in BiGG models
# We will match these prefixes (e.g. 'atp_c', 'atp_e', 'h2o_c')
CURRENCY_PREFIXES = [
    'h2o', 'atp', 'adp', 'nad', 'nadh', 'nadp', 'nadph', 
    'pi', 'h', 'co2', 'o2', 'nh4', 'coa', 'accoa'
]

def is_currency(node_id):
    # Check if node_id starts with any currency prefix followed by _
    # or is exactly the prefix (unlikely in bigg)
    base = node_id.lower()
    for p in CURRENCY_PREFIXES:
        if base == p or base.startswith(f"{p}_"):
            return True
    return False

def main():
    # 1. Models
    print(f"Loading Models from {MODEL_DIR}")
    files = glob.glob(os.path.join(MODEL_DIR, "*.xml"))
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for f in files:
        file_id = os.path.splitext(os.path.basename(f))[0]
        output_file = os.path.join(OUTPUT_DIR, f"{file_id}.json")
        
        # if os.path.exists(output_file):
        #     print(f"Skipping {file_id} (already done)")
        #     continue
            
        print(f"Processing {file_id}...")
        
        try:
            cobra_model = load_sbml_model(f)
            if not cobra_model: continue
            
            G_full = build_bipartite_graph(cobra_model)
            if len(G_full.nodes) < 2: continue
            
            # --- 1. Identify Currency Hubs ---
            # Instead of just degree, we use the explicit list AND degree check (safer)
            # User wants "h20, atp adp..." specifically.
            currency_nodes = []
            for n in G_full.nodes():
                if is_currency(n):
                    currency_nodes.append(n)
            
            print(f"  Found {len(currency_nodes)} currency nodes (e.g. {currency_nodes[:3]}).")
            
            # --- 2. Create Backbone Graph (Currency Removed) ---
            G_backbone = G_full.copy()
            G_backbone.remove_nodes_from(currency_nodes)
            
            # Remove isolated nodes? 
            # If we remove ATP, the reaction node might still be connected to other things.
            # If a reaction ONLY connected to currency (rare), it becomes isolated.
            # We keep isolated backbone nodes so they don't disappear.
            
            # --- 3. Minimize Backbone ---
            if len(G_backbone.nodes) > 0:
                normalized_pos = optimize_layout_sa(G_backbone, steps=2000) # Increased steps for quality
            else:
                normalized_pos = {}
                
            # --- 4. Re-Assemble & Duplicate Hubs ---
            final_coords = {}
            visual_graph = nx.Graph() # For export
            
            # Scale up
            SCALE = 5000.0
            
            # A. Place Backbone Nodes
            for n, pos in normalized_pos.items():
                final_coords[n] = (float(pos[0] * SCALE), float(pos[1] * SCALE))
                # Add node data to visuals
                data = G_full.nodes[n]
                visual_graph.add_node(n, **data)
                
            # Add backbone edges
            for u, v in G_backbone.edges:
                visual_graph.add_edge(u, v)
                
            # B. Re-attach Currency Nodes (Duplication Logic)
            print("  Re-attaching currency nodes locally...")
            dup_count = 0
            
            # Iterate through all reactions in the original graph
            for n, data in G_full.nodes(data=True):
                if data.get('type') == 'reaction':
                    
                    # If reaction was optimized in backbone, we have its position
                    if n not in final_coords: continue
                    rxn_pos = final_coords[n]
                    
                    # Check ALL neighbors in original graph
                    neighbors = list(G_full.neighbors(n))
                    
                    for met in neighbors:
                        if met in currency_nodes:
                            # This is a currency metabolite!
                            # Create a UNIQUE DUPLICATE for this specific reaction
                            dup_id = f"{met}__dup_{dup_count}" # Unique ID
                            dup_count += 1
                            
                            # Place locally near reaction
                            # Can we be smarter? Inputs on left, outputs on right?
                            # We don't have direction easily in undirected graph unless we check stoichiometry (weights)
                            # Let's try to place them based on "Role" if possible, or just random circle
                            
                            angle = random.random() * 2 * math.pi
                            offset = 120.0 
                            
                            # Try to avoid overlapping with existing backbone edges?
                            # For now, simple radial offset
                            dx = math.cos(angle) * offset
                            dy = math.sin(angle) * offset
                            
                            hub_pos = (rxn_pos[0] + dx, rxn_pos[1] + dy)
                            
                            final_coords[dup_id] = hub_pos
                            
                            # Add Node
                            hub_data = G_full.nodes[met].copy()
                            hub_data['original_id'] = met
                            # Make label visible as original name
                            hub_data['name'] = hub_data.get('name', met) 
                            visual_graph.add_node(dup_id, **hub_data)
                            
                            # Add Edge (Reaction -> Duplicate)
                            # CRITICAL: Do NOT connect duplicates to each other.
                            # Only connect Reaction <-> Duplicate
                            visual_graph.add_edge(n, dup_id)
                            
            # --- 5. Final Snap & Export ---
            # Don't snap hubs? Or snap everything?
            # Snap everything for clean lines
            snapped = snap_to_grid(final_coords, visual_graph, grid_spacing=100)
            
            serialize_results(visual_graph, snapped, output_file)
            print("  Done.")
            
        except Exception as e:
            print(f"  Error: {e}")
            # import traceback
            # traceback.print_exc()

if __name__ == "__main__":
    main()
