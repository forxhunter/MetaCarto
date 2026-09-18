import numpy as np

def snap_to_grid(coords, graph, grid_spacing=200):
    """
    Manhattan Enforcer:
    Strictly forces every edge to be perfectly horizontal or vertical logic.
    """
    # 1. Initial Quantization with Position Normalization
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    if not xs: return {}

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    
    # If map is huge (> 6000), it's likely a Meta-Layout. 
    # Just shift to 0,0 (plus padding) to ensure it fits positive canvas.
    # If map is small (e.g. single e_coli subsystem), center it at 2500, 2500 for aesthetics.
    if width > 6000 or height > 6000:
        center_offset_x = -min_x + 500
        center_offset_y = -min_y + 500
    else:
        # Standard centered layout for single maps
        mean_x, mean_y = sum(xs)/len(xs), sum(ys)/len(ys)
        center_offset_x = 2500 - mean_x
        center_offset_y = 2500 - mean_y
    
    snapped = {}
    for n, (x, y) in coords.items():
        snapped[n] = [x + center_offset_x, y + center_offset_y]
        
    # 2. Manhattan Pass (Iterative Constraint Satisfaction)
    # Force alignment on edges.
    for _ in range(10): # Aggressive iteration
        for u, v in graph.edges:
            x1, y1 = snapped[u]
            x2, y2 = snapped[v]
            
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            
            # Decision: Force X or Force Y?
            # If dx is smaller, it's easier to make vertical line (x1=x2).
            if dx < dy:
                # Force Vertical
                avg_x = (x1 + x2) / 2
                snapped[u][0] = avg_x
                snapped[v][0] = avg_x
            else:
                # Force Horizontal
                avg_y = (y1 + y2) / 2
                snapped[u][1] = avg_y
                snapped[v][1] = avg_y
                
    # 3. Final Grid Snap
    for n in snapped:
        snapped[n][0] = round(snapped[n][0] / grid_spacing) * grid_spacing
        snapped[n][1] = round(snapped[n][1] / grid_spacing) * grid_spacing
        
    # 4. Anti-Overlap (Spiral)
    # ... (Reuse overlap logic)
    sorted_nodes = sorted(snapped.keys())
    occupied = {}
    collisions = []
    for n in sorted_nodes:
        pos = tuple(snapped[n])
        if pos in occupied: collisions.append(n)
        else: occupied[pos] = n
        
    for n in collisions:
        x, y = snapped[n]
        found = False
        radius = 1
        while not found:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) < radius and abs(dy) < radius: continue
                    nx, ny = x + dx*grid_spacing, y + dy*grid_spacing
                    if (nx, ny) not in occupied:
                        snapped[n] = [nx, ny]
                        occupied[(nx, ny)] = n
                        found = True
                        break
                if found: break
            radius += 1
            
    final_coords = {n: (float(p[0]), float(p[1])) for n, p in snapped.items()}
    return final_coords

def generate_node_primitives(graph, coords):
    """
    Converts graph nodes and coordinates into Escher-compatible node schemas.
    """
    nodes = {}
    for node_id in graph.nodes:
        x, y = coords[node_id]
        node_type = graph.nodes[node_id].get('type', 'metabolite')
        
        nodes[node_id] = {
            "node_type": node_type,
            "x": float(x),
            "y": float(y),
            "bigg_id": node_id,
            "name": graph.nodes[node_id].get('name', node_id),
            "label_x": float(x) + 20, # Offset label
            "label_y": float(y) + 10,
            "node_is_primary": True # Heuristic
        }
    return nodes
