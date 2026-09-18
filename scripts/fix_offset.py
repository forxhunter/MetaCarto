import json

# Load the JSON
with open('data/bigg/e_coli_core/e_coli_core_Combined.json', 'r') as f:
    data = json.load(f)

nodes = data[1]['nodes']
reactions = data[1]['reactions']

# Find current bounds
xs = [n['x'] for n in nodes.values()]
ys = [n['y'] for n in nodes.values()]

min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)

print(f"Before shift:")
print(f"  Nodes: ({min_x:.0f}, {min_y:.0f}) to ({max_x:.0f}, {max_y:.0f})")
print(f"  Canvas: {data[1]['canvas']}")

# Calculate shift to move content to start at (500, 500)
target_start = 500
shift_x = target_start - min_x
shift_y = target_start - min_y

print(f"\nShifting by: ({shift_x:.0f}, {shift_y:.0f})")

# Shift all nodes
for node in nodes.values():
    node['x'] += shift_x
    node['y'] += shift_y
    node['label_x'] += shift_x
    node['label_y'] += shift_y

# Shift all reactions
for rxn in reactions.values():
    rxn['label_x'] += shift_x
    rxn['label_y'] += shift_y

# Recalculate canvas
new_xs = [n['x'] for n in nodes.values()]
new_ys = [n['y'] for n in nodes.values()]

new_min_x, new_max_x = min(new_xs), max(new_xs)
new_min_y, new_max_y = min(new_ys), max(new_ys)

content_width = new_max_x - new_min_x
content_height = new_max_y - new_min_y

padding = 500
data[1]['canvas']['width'] = content_width + 2 * padding
data[1]['canvas']['height'] = content_height + 2 * padding

print(f"\nAfter shift:")
print(f"  Nodes: ({new_min_x:.0f}, {new_min_y:.0f}) to ({new_max_x:.0f}, {new_max_y:.0f})")
print(f"  Canvas: {data[1]['canvas']}")

# Calculate occupancy
occupancy = (content_width * content_height) / (data[1]['canvas']['width'] * data[1]['canvas']['height']) * 100
print(f"\nOccupancy: {occupancy:.1f}%")

# Save
with open('data/bigg/e_coli_core/e_coli_core_Combined.json', 'w') as f:
    json.dump(data, f, indent=2)

print("\n✅ Fixed and saved!")
