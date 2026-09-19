import os
import glob
import numpy as np
import networkx as nx
from src.parsing import parse_kgml
from experiments.rl_env import MetabolicLayoutEnv
from src.refinement import snap_to_grid
from stable_baselines3 import PPO

def evaluate_layout(graph, coords):
    """
    Calculates quality metrics for a given layout.
    """
    ortho_edges = 0
    total_edges = 0
    total_length = 0
    
    epsilon = 1.0 # Pixel tolerance
    
    for u, v in graph.edges:
        if u not in coords or v not in coords:
            continue
        p1 = coords[u]
        p2 = coords[v]
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        
        total_length += np.sqrt(dx**2 + dy**2)
        total_edges += 1
        
        # Check strict orthogonality
        if dx < epsilon or dy < epsilon:
            ortho_edges += 1
            
    # Check Collisions
    # Simple O(N^2)
    nodes = list(coords.keys())
    collisions = 0
    min_dist = 50.0 # Standard node size buffer
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            n1, n2 = nodes[i], nodes[j]
            p1 = coords[n1]
            p2 = coords[n2]
            dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            if dist < min_dist:
                collisions += 1
                
    return {
        "ortho_pct": (ortho_edges / total_edges * 100) if total_edges > 0 else 0,
        "collisions": collisions,
        "avg_edge_len": (total_length / total_edges) if total_edges > 0 else 0
    }

def run_stress_test(data_dir="data/kegg", model_path="ppo_metabolic_layout.zip"):
    # Load Model
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = PPO.load(model_path)
    else:
        print("Model not found. Running with random agent (baseline).")
        model = None

    files = glob.glob(os.path.join(data_dir, "*.xml"))
    # Pick a few representative ones if too many
    # Sort by size? 
    test_files = files[:5] # Test first 5 found
    output_dir = "validation_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running stress test on {len(test_files)} maps...")
    print(f"Saving outputs to {output_dir}/")
    
    results = []
    
    for f in test_files:
        filename = os.path.basename(f)
        try:
            G = parse_kgml(f)
            if G is None or len(G.nodes) < 5:
                continue
                
            print(f"\nTesting {filename} (Nodes: {len(G.nodes)}, Edges: {len(G.edges)})")
            
            # 1. RL Layout
            env = MetabolicLayoutEnv(G)
            obs, _ = env.reset()
            
            # Simple inference loop
            # PPO internal predict loop isn't stateful enough for refinement if we don't track it
            # But specific to our Env: Step updates self.state.
            
            # We need to run the agent for N steps to let it settle? 
            # OR typically we run inference until "convergence".
            # For PPO, we usually just predict NEXT action.
            # But the environment is continuous layout.
            # Let's run 100 steps of inference.
            
            state = env.state
            for _ in range(100):
                if model:
                    action, _ = model.predict(obs)
                else:
                    action = env.action_space.sample()
                obs, _, _, _, _ = env.step(action)
                state = env.state
                
            # Denormalize
            raw_coords = {}
            for idx, n in env.idx_to_node.items():
                # Map [0,1] to [0, 5000] canvas
                raw_coords[n] = (float(state[idx][0] * 5000), float(state[idx][1] * 5000))
            
            # 2. Refinement (Manhattan Enforcer)
            snapped_coords = snap_to_grid(raw_coords, G, grid_spacing=200)
            
            # 3. Evaluate
            metrics = evaluate_layout(G, snapped_coords)
            metrics['file'] = filename
            results.append(metrics)
            
            # 4. Save JSON
            from src.export import serialize_results
            output_name = os.path.splitext(filename)[0] + ".json"
            serialize_results(G, snapped_coords, os.path.join(output_dir, output_name))
            
            print(f"  -> Orthogonality: {metrics['ortho_pct']:.1f}%")
            print(f"  -> Collisions: {metrics['collisions']}")
            
        except Exception as e:
            print(f"  FAILED: {str(e)}")
            
    # Summary
    if results:
        avg_ortho = sum(r['ortho_pct'] for r in results) / len(results)
        total_col = sum(r['collisions'] for r in results)
        print("\n=== STRESS TEST SUMMARY ===")
        print(f"Average Orthogonality: {avg_ortho:.2f}% (Target: >95%)")
        print(f"Total Collisions: {total_col} (Target: 0)")
        
if __name__ == "__main__":
    run_stress_test()
