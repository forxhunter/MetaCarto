import json

with open('data/bigg/e_coli_core/e_coli_core_Combined.json', 'r') as f:
    d = json.load(f)

nodes = d[1]['nodes']
xs = [n['x'] for n in nodes.values()]
ys = [n['y'] for n in nodes.values()]
canvas = d[1]['canvas']

print(f"Canvas size: {canvas['width']:.0f} x {canvas['height']:.0f} px")
print(f"Content X: {min(xs):.0f} to {max(xs):.0f} (width={max(xs)-min(xs):.0f})")
print(f"Content Y: {min(ys):.0f} to {max(ys):.0f} (height={max(ys)-min(ys):.0f})")
print(f"\nArea occupancy: {((max(xs)-min(xs))*(max(ys)-min(ys)))/(canvas['width']*canvas['height'])*100:.1f}%")
print(f"Width occupancy: {(max(xs)-min(xs))/canvas['width']*100:.1f}%")
print(f"Height occupancy: {(max(ys)-min(ys))/canvas['height']*100:.1f}%")

# Check if content is centered or pushed to corner
print(f"\nContent starts at: ({min(xs):.0f}, {min(ys):.0f})")
print(f"Content ends at: ({max(xs):.0f}, {max(ys):.0f})")
