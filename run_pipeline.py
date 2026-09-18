import os
import networkx as nx
import numpy as np
from src.parsing import build_bipartite_graph
from src.export import serialize_results
import argparse

def create_mock_sbml_graph():
    """
    Creates a mock bipartite graph since we might not have an SBML file.
    """
    G = nx.Graph()
    # Mock Glycolysis-like structure
    nodes = [
        ("Glc", "metabolite"), ("ATP", "metabolite"), ("ADP", "metabolite"), ("G6P", "metabolite"),
        ("HK", "reaction"), 
        ("PGI", "reaction"), ("F6P", "metabolite")
    ]
    edges = [
        ("Glc", "HK"), ("ATP", "HK"), ("HK", "G6P"), ("HK", "ADP"),
        ("G6P", "PGI"), ("PGI", "F6P")
    ]
    
    for n, t in nodes:
        G.add_node(n, type=t, name=n)
    for u, v in edges:
        G.add_edge(u, v)
    return G

def run_pipeline(output_file="output.json"):
    print("Step 1: Loading Model/Graph...")
    # In real use: model = load_sbml(args.input)
    # G = build_bipartite_graph(model)
    G = create_mock_sbml_graph()
    print(f"Graph created: {len(G.nodes)} nodes, {len(G.edges)} edges.")

    print("Step 2: Functional Grouping (GNN)...")
    # placeholder: assign random groups & flux weights
    for n in G.nodes:
        G.nodes[n]['group'] = np.random.randint(0, 3)
    
    for u, v in G.edges:
        G.edges[u, v]['weight'] = np.random.choice([0.1, 1.0]) # Mock FBA flux
    
    print("Step 3: Layout Synthesis (RL)...")
    # Demonstrate RL integration
    from src.rl_env import MetabolicLayoutEnv
    env = MetabolicLayoutEnv(G)
    # Just run a reset/step to prove it works
    env.reset()
    action = env.action_space.sample()
    new_state, _, _, _, _ = env.step(action)
    
    # Extract coords from RL state
    # State is normalized [0,1], scale to Canvas
    coords = {}
    for idx, node in env.idx_to_node.items():
        coords[node] = (float(new_state[idx][0] * 5000), float(new_state[idx][1] * 5000))
        
    # pos = nx.spring_layout(G, scale=1000)
    # coords = {n: (float(x), float(y)) for n, (x, y) in pos.items()}
    
    print("Step 4: Refinement & Export...")
    from src.refinement import snap_to_grid
    snapped_coords = snap_to_grid(coords, G, grid_spacing=200)
    
    serialize_results(G, snapped_coords, output_file)
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
