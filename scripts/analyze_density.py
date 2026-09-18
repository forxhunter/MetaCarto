import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_density(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    nodes = data[1]['nodes']
    if not nodes:
        print("No nodes found.")
        return

    xs = [n['x'] for n in nodes.values()]
    ys = [n['y'] for n in nodes.values()]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = max_x - min_x
    height = max_y - min_y
    area = width * height
    
    print(f"Total Nodes: {len(nodes)}")
    print(f"Bounding Box: {min_x:.0f},{min_y:.0f} to {max_x:.0f},{max_y:.0f}")
    print(f"Dimensions: {width:.0f} x {height:.0f}")
    
    # Calculate nearest neighbor distances to measure local density
    points = np.column_stack((xs, ys))
    
    # Simple grid density
    # Divide bbox into 50x50 grid
    grid_size = 50
    x_bins = np.linspace(min_x, max_x, grid_size+1)
    y_bins = np.linspace(min_y, max_y, grid_size+1)
    
    hist, _, _ = np.histogram2d(xs, ys, bins=[x_bins, y_bins])
    
    non_empty_cells = np.sum(hist > 0)
    total_cells = grid_size * grid_size
    grid_occupancy = non_empty_cells / total_cells
    
    print(f"Grid Density ({grid_size}x{grid_size}): {grid_occupancy*100:.1f}% occupied cells")
    
    # Check for outliers
    # Nodes in the 1st and 99th percentile
    p01_x, p99_x = np.percentile(xs, [1, 99])
    p01_y, p99_y = np.percentile(ys, [1, 99])
    
    print(f"1st-99th Percentile Range X: {p01_x:.0f} to {p99_x:.0f} (Span: {p99_x-p01_x:.0f})")
    print(f"1st-99th Percentile Range Y: {p01_y:.0f} to {p99_y:.0f} (Span: {p99_y-p01_y:.0f})")
    
    if (width > (p99_x - p01_x) * 1.5) or (height > (p99_y - p01_y) * 1.5):
        print("WARNING: Significant outliers detected! Bounding box is much larger than core content.")

if __name__ == "__main__":
    analyze_density('data/bigg/e_coli_core/e_coli_core_Combined.json')
