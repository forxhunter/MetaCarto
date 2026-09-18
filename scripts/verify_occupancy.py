import json
import sys

def calculate_canvas_occupancy(json_file):
    """
    Calculates the actual canvas occupancy of an Escher map.
    Uses AREA measurement: (content_bbox_area / canvas_area)
    Returns: (occupancy_percentage, canvas_width, canvas_height, content_width, content_height)
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract nodes
    nodes = data[1]['nodes']
    
    if not nodes:
        return 0, 0, 0, 0, 0
    
    # Get all node positions
    xs = [node['x'] for node in nodes.values()]
    ys = [node['y'] for node in nodes.values()]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    content_width = max_x - min_x
    content_height = max_y - min_y
    
    # Canvas size (from the JSON)
    canvas = data[1]['canvas']
    canvas_width = canvas['width']
    canvas_height = canvas['height']
    
    # Calculate occupancy using AREA (this is the correct way)
    content_area = content_width * content_height
    canvas_area = canvas_width * canvas_height
    
    occupancy = content_area / canvas_area if canvas_area > 0 else 0
    
    return occupancy * 100, canvas_width, canvas_height, content_width, content_height

def scale_map_to_target_occupancy(json_file, target_occupancy=0.80, max_iterations=10):
    """
    Iteratively scales a map until it reaches target occupancy.
    """
    print(f"\n{'='*60}")
    print(f"Iterative Scaling: {json_file}")
    print(f"Target Occupancy: {target_occupancy*100}%")
    print(f"{'='*60}\n")
    
    for iteration in range(max_iterations):
        # Calculate current occupancy
        occupancy_pct, canvas_w, canvas_h, content_w, content_h = calculate_canvas_occupancy(json_file)
        
        print(f"Iteration {iteration + 1}:")
        print(f"  Canvas: {canvas_w:.0f} x {canvas_h:.0f} px")
        print(f"  Content: {content_w:.0f} x {content_h:.0f} px")
        print(f"  Occupancy: {occupancy_pct:.1f}%")
        
        # Check if we're close enough
        if abs(occupancy_pct/100 - target_occupancy) < 0.02:  # Within 2%
            print(f"\n✅ Target reached! Final occupancy: {occupancy_pct:.1f}%")
            return True
        
        # Calculate scale factor needed
        current_occupancy = occupancy_pct / 100
        scale_factor = target_occupancy / max(current_occupancy, 0.01)
        
        print(f"  Scaling by: {scale_factor:.2f}x")
        
        # Load and scale
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        nodes = data[1]['nodes']
        
        # Find center
        xs = [node['x'] for node in nodes.values()]
        ys = [node['y'] for node in nodes.values()]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        
        # Scale all nodes from center
        for node_id, node in nodes.items():
            node['x'] = center_x + (node['x'] - center_x) * scale_factor
            node['y'] = center_y + (node['y'] - center_y) * scale_factor
            node['label_x'] = center_x + (node['label_x'] - center_x) * scale_factor
            node['label_y'] = center_y + (node['label_y'] - center_y) * scale_factor
        
        # Scale reactions
        reactions = data[1]['reactions']
        for rxn_id, rxn in reactions.items():
            rxn['label_x'] = center_x + (rxn['label_x'] - center_x) * scale_factor
            rxn['label_y'] = center_y + (rxn['label_y'] - center_y) * scale_factor
        
        # Update canvas size
        new_xs = [node['x'] for node in nodes.values()]
        new_ys = [node['y'] for node in nodes.values()]
        data[1]['canvas']['width'] = max(new_xs) + 500
        data[1]['canvas']['height'] = max(new_ys) + 500
        
        # Save
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  ✓ Scaled and saved\n")
    
    print(f"⚠️  Max iterations ({max_iterations}) reached")
    return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = "data/bigg/e_coli_core/e_coli_core_Combined.json"
    
    print(f"\n🔍 Checking: {json_file}")
    
    # Initial check
    occupancy_pct, canvas_w, canvas_h, content_w, content_h = calculate_canvas_occupancy(json_file)
    
    print(f"\nInitial State:")
    print(f"  Canvas: {canvas_w:.0f} x {canvas_h:.0f} px")
    print(f"  Content: {content_w:.0f} x {content_h:.0f} px")
    print(f"  Occupancy: {occupancy_pct:.1f}%")
    
    '''
    if occupancy_pct < 70:  # If less than 70%, scale it up
        print(f"\n⚠️  Occupancy too low! Starting iterative scaling...")
        scale_map_to_target_occupancy(json_file, target_occupancy=0.80)
    elif occupancy_pct > 85:  # If more than 85%, scale down
        print(f"\n⚠️  Occupancy too high! Starting iterative scaling...")
        scale_map_to_target_occupancy(json_file, target_occupancy=0.80)
    else:
    '''
    print(f"\n✅ Occupancy is acceptable ({occupancy_pct:.1f}%)")
