"""Top-level layout engine: cobra reactions in, Escher map out.

Wires together the stages described in layout_algorithm.md. Everything here is
constructive and deterministic -- there is no annealing and no random seed, so
the same model always produces the same map.
"""

import math

from .compound import build_compound_graph
from .direction import flux_directions, orient_compound_graph
from .motifs import contract_rings, expand_rings, find_rings
from .render import build_escher_map
from .sugiyama import layered_layout

NODE_HEIGHT = 80.0         # drawn height of a metabolite node, with its label
LAYER_GAP = 180.0          # vertical pitch between layers
X_GAP = 120.0              # minimum horizontal clearance between nodes
CHAR_WIDTH = 20.0
MIN_NODE_WIDTH = 160.0
RING_PITCH = 300.0         # arc length between consecutive ring members


class LayoutResult:
    def __init__(self, escher_map, cgraph, pos, rings, layering):
        self.escher_map = escher_map
        self.cgraph = cgraph
        self.pos = pos
        self.rings = rings
        self.layering = layering


def layout_reactions(model, reactions, map_name, author="AutoLayout",
                     use_fba=True, verbose=False, groups=None):
    """Lay out an arbitrary subset of a model. Returns a LayoutResult, or None
    if the subset has no drawable primary-compound structure.

    `groups` optionally maps a metabolite id to a cluster key (subsystem or
    compartment). Supplying it keeps each cluster contiguous through crossing
    minimisation, which is what stops a whole-model map from interleaving
    pathways that a reader expects to find in one place.
    """
    cgraph = build_compound_graph(model, reactions)
    if cgraph.D.number_of_nodes() == 0:
        return None

    orient_compound_graph(cgraph, model=model, use_fba=use_fba, verbose=verbose)

    rings = find_rings(cgraph.D)
    contracted, ring_records, ring_of = contract_rings(cgraph.D, rings, RING_PITCH)
    if verbose and rings:
        print(f"    rings: {[len(r) for r in rings]}")

    width, height = {}, {}
    for node in contracted.nodes:
        if node in ring_records:
            width[node] = ring_records[node]["size"]
            height[node] = ring_records[node]["size"]
        else:
            width[node] = max(MIN_NODE_WIDTH, CHAR_WIDTH * len(str(node)))
            height[node] = NODE_HEIGHT

    sinks = {ring_of.get(n, n) for n, role in cgraph.roles.items() if role == "sink"}
    flux = flux_directions(model) if use_fba else {}

    def reversal_cost(u, v):
        rxns = contracted.edges[u, v]["rxns"]
        if any(r in flux for r in rxns):
            return 8.0          # never draw a flux-carrying step backwards
        if all(cgraph.reactions[r].reversible for r in rxns if r in cgraph.reactions):
            return 1.0
        return 3.0

    contracted_groups = None
    if groups:
        contracted_groups = {n: groups.get(n, "") for n in contracted.nodes}
        for super_id, record in ring_records.items():
            keys = [groups.get(m, "") for m in record["members"]]
            contracted_groups[super_id] = max(set(keys), key=keys.count) if keys else ""

    pos_contracted, layering = layered_layout(
        contracted, width, height, sinks=sinks, groups=contracted_groups,
        x_gap=X_GAP, y_gap=LAYER_GAP, reversal_cost=reversal_cost,
    )
    pos_contracted = _pack_components(contracted, layering, pos_contracted, width, height)
    pos = expand_rings(pos_contracted, ring_records, ring_of, cgraph.D)
    routes = _build_routes(cgraph, layering, pos, ring_of,
                           {sid: rec["members"] for sid, rec in ring_records.items()})

    escher_map = build_escher_map(
        cgraph, {n: p for n, p in pos.items() if not str(n).startswith("__dummy__")},
        map_name, author=author, routes=routes,
    )
    return LayoutResult(escher_map, cgraph, pos, rings, layering)


def _pack_components(contracted, layering, pos, width, height,
                     target_aspect=1.0, gap=420.0):
    """Shelf-pack disconnected pieces instead of leaving them in one row.

    Layering places every component side by side, so a subsystem of nineteen
    independent transport steps comes out seventeen times wider than it is
    tall. Packing them into a roughly square block is the component-level
    version of the meta-tiling the whole-model map does.
    """
    import networkx as nx

    component_of, members = {}, {}
    for index, component in enumerate(nx.weakly_connected_components(contracted)):
        members[index] = set(component)
        for node in component:
            component_of[node] = index
    for dummy, (u, v) in layering.dummies.items():
        index = component_of.get(u, component_of.get(v))
        if index is not None:
            members[index].add(dummy)
            component_of[dummy] = index

    if len(members) < 2:
        return pos

    boxes = []
    for index, nodes in members.items():
        extents = [(pos[n], width.get(n, 40.0), height.get(n, 40.0))
                   for n in nodes if n in pos]
        if not extents:
            continue
        left = min(p[0] - w / 2.0 for p, w, _ in extents)
        right = max(p[0] + w / 2.0 for p, w, _ in extents)
        top = min(p[1] - h / 2.0 for p, _, h in extents)
        bottom = max(p[1] + h / 2.0 for p, _, h in extents)
        boxes.append([index, left, top, right - left, bottom - top])
    if not boxes:
        return pos

    # Size the shelf from *padded* areas: with many small components the gaps
    # dominate the footprint, and ignoring them produces a block as badly
    # proportioned as the single row it replaced, only the other way round.
    padded_area = sum((b[3] + gap) * (b[4] + gap) for b in boxes)
    row_width = max(math.sqrt(padded_area * target_aspect),
                    max(b[3] for b in boxes) + gap)

    boxes.sort(key=lambda b: -b[4])
    offsets, cursor_x, cursor_y, row_height = {}, 0.0, 0.0, 0.0
    for index, left, top, box_w, box_h in boxes:
        if cursor_x > 0.0 and cursor_x + box_w > row_width:
            cursor_x = 0.0
            cursor_y += row_height + gap
            row_height = 0.0
        offsets[index] = (cursor_x - left, cursor_y - top)
        cursor_x += box_w + gap
        row_height = max(row_height, box_h)

    packed = {}
    for node, (x, y) in pos.items():
        dx, dy = offsets.get(component_of.get(node, -1), (0.0, 0.0))
        packed[node] = (x + dx, y + dy)
    return packed


def _build_routes(cgraph, layering, pos, ring_of, ring_members=None):
    """Polyline per reaction, following the layered pass's dummy chain."""
    ring_members = ring_members or {}
    centres = {}
    for super_id, members in ring_members.items():
        placed = [pos[m] for m in members if m in pos]
        if placed:
            centres[super_id] = (sum(p[0] for p in placed) / len(placed),
                                 sum(p[1] for p in placed) / len(placed))

    routes = {}
    for u, v, data in cgraph.D.edges(data=True):
        a, b = ring_of.get(u, u), ring_of.get(v, v)

        if a == b:
            # An arc inside a ring is drawn as an arc, not as a chord. A
            # heptagon reads as a polygon; a curated TCA cycle is a circle.
            # It also stops the orthogonality metric from charging a ring with
            # being non-orthogonal, which a ring is by definition.
            points, orthogonal = [pos[u], pos[v]], False
        else:
            chain = layering.chains.get((a, b))
            if chain is None:
                chain = layering.chains.get((b, a))
                chain = chain[::-1] if chain else []
            points = [pos[u]] + [pos[d] for d in chain] + [pos[v]]
            orthogonal = True

        for rid in data["rxns"]:
            record = cgraph.reactions.get(rid)
            if record is None:
                continue
            forward = record.main_sub == u
            route = {
                "points": points if forward else points[::-1],
                "orthogonal": orthogonal,
            }
            if a == b and a in centres:
                route["arc_centre"] = centres[a]
            routes[rid] = route
    return routes
