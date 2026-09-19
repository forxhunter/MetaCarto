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
                     use_fba=True, verbose=False, groups=None, render=True):
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

    # Cap the drawing width so a broad, shallow network wraps into a block
    # instead of one enormous row.
    #
    # Sized for a roughly square result rather than from the node count. If the
    # nodes were laid end to end the run would be `total_width`; wrapping it at
    # W gives total_width/W rows of height `row_height`, and setting the two
    # sides equal gives W = sqrt(total_width * row_height).
    #
    # A node-count threshold gets this wrong at both ends. e_coli's transport
    # cluster is only 45 reactions, so a count-based cap never fired and two of
    # its components came out 5040 x 304 -- flat strips that stretched the map
    # to six times the node density of the carbohydrate map beside it.
    total_width = sum(width.get(v, 160.0) + X_GAP for v in contracted.nodes)
    row_height = NODE_HEIGHT + LAYER_GAP
    max_width = max(2000.0, math.sqrt(max(total_width, 1.0) * row_height))

    pos_contracted, layering = layered_layout(
        contracted, width, height, sinks=sinks, groups=contracted_groups,
        x_gap=X_GAP, y_gap=LAYER_GAP, reversal_cost=reversal_cost,
        max_width=max_width,
    )
    pos_contracted = _pack_components(contracted, layering, pos_contracted, width, height)
    pos = expand_rings(pos_contracted, ring_records, ring_of, cgraph.D)
    routes = _build_routes(cgraph, layering, pos, ring_of,
                           {sid: rec["members"] for sid, rec in ring_records.items()})

    # `render=False` returns the placement without drawing it. The benchmark
    # needs that: it renders separately, under its own conditions, and timing
    # the discarded render here inflated MetaCarto's reported seconds-per-map
    # by about 45% against baselines that render once.
    escher_map = None
    if render:
        escher_map = build_escher_map(
            cgraph, {n: p for n, p in pos.items()
                     if not str(n).startswith("__dummy__")},
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

    # Scale the gap to what is being packed, instead of a flat 420 everywhere.
    #
    # A transport cluster is mostly two-node components about 300 units across,
    # so a fixed 420-unit gutter is wider than the components it separates and
    # the map comes out six times sparser than a connected pathway of the same
    # reaction count -- measured on e_coli: 9.0 nodes per million square units
    # for transport against 58.1 for carbohydrate metabolism, at 45 and 38
    # reactions respectively. Tying the gutter to the median component keeps the
    # visual density comparable across maps, which is what makes a set of
    # figures look like a set.
    dims = sorted(max(b[3], b[4]) for b in boxes)
    median_dim = dims[len(dims) // 2] if dims else gap
    gap = min(max(0.30 * median_dim, 120.0), gap)

    # Then cut the reserved air to a tenth of that.
    #
    # The gutter above is sized to separate whole pathway components, and using
    # it as the spacing for everything is what made these maps read as mostly
    # white: two one-reaction stubs ended up ~880 units apart while the stub
    # itself is about 400 across. Scaled down, the pieces sit next to each
    # other instead.
    #
    # Floored at the 120 that was already this function's own lower bound.
    # That is not a guess at what is safe, it is the value the gutter took on
    # every small-component map before any of this, and those maps had no
    # inter-component overlaps -- so 120 is known-sufficient in practice, and
    # anything above it was the thing worth cutting.
    gap = max(gap * WHITESPACE_SCALE, 120.0)

    # Size the shelf from *padded* areas: with many small components the gaps
    # dominate the footprint, and ignoring them produces a block as badly
    # proportioned as the single row it replaced, only the other way round.
    padded_area = sum((b[3] + gap) * (b[4] + gap) for b in boxes)
    row_width = max(math.sqrt(padded_area * target_aspect),
                    max(b[3] for b in boxes) + gap)

    # Skyline, not shelf, for the *cores*. A shelf makes every row as tall as
    # its tallest member, so one big component reserves a full-height row and
    # leaves the space beside it empty -- measured on Recon3D's carbohydrate
    # map, 62% of a 16x16 grid was empty while the busiest cell held 15x the
    # mean.
    from .compose import _skyline_pack

    # Two tiers, drawn in that order: the pathway cores are placed first and
    # keep the shape the layering gave them, then the fragments -- one- and
    # two-reaction stubs with no pathway context of their own -- go into
    # whatever space is left, holes inside the cores first and the margin
    # around them after. Packing both tiers together instead treats a core as
    # a solid rectangle, so its interior holes survive untouched while the
    # stubs accumulate in a block beside it, and that second sparse block is
    # most of why these maps read as mostly white.
    #
    # "Fragment" is relative to the map, not an absolute node count. These are
    # nodes of the *contracted compound graph* -- one per metabolite, rings
    # already collapsed -- so a whole pathway is 26 nodes and a lone reaction
    # is 2. An absolute threshold of twelve, sized as if these were rendered
    # nodes, classified entire pathways as fragments on one map and nothing at
    # all on the next: across e_coli's three function maps it fired zero times.
    biggest = max(len(members[b[0]]) for b in boxes)
    cutoff = max(FRAGMENT_FLOOR, FRAGMENT_SHARE * biggest)
    cores = [b for b in boxes if len(members[b[0]]) >= cutoff]
    fragments = [b for b in boxes if len(members[b[0]]) < cutoff]
    # A map of uniformly small pieces -- 120 two-node transport steps -- has no
    # core to fill around, and packing it as a regular grid is already its best
    # drawing (those maps score 1.0-1.9 on hairball, the best in the corpus).
    # Leave them to the plain skyline.
    if not cores or not fragments:
        cores, fragments = boxes, []

    cores.sort(key=lambda b: -b[4])
    order = [b[0] for b in cores]
    box_width = {b[0]: b[3] + gap for b in cores}
    box_height = {b[0]: b[4] + gap for b in cores}
    # Packed to the width of the *whole* drawing, fragments included, not to
    # the width the cores alone need. Sizing this to the cores is what broke
    # the first version of the two-tier packer: the core block came out exactly
    # as wide as the cores, so no fragment had anywhere to go, every one of
    # them fell through to the overflow path, and the map turned into a column
    # of stubs taller than the pathway they were meant to fill in around --
    # aspect ratio 0.86 to 0.20 across the 93 Recon3D function maps.
    placed = _skyline_pack(order, box_width, box_height, gap, row_width)

    origin = {b[0]: (b[1], b[2]) for b in boxes}
    offsets = {}
    for index, (px, py) in placed.items():
        left, top = origin[index]
        offsets[index] = (px - left, py - top)

    if fragments:
        offsets.update(_fill_gaps(fragments, cores, offsets, members, pos,
                                  width, height, layering, gap, padded_area,
                                  target_aspect))

    packed = {}
    for node, (x, y) in pos.items():
        dx, dy = offsets.get(component_of.get(node, -1), (0.0, 0.0))
        packed[node] = (x + dx, y + dy)
    return packed


# A component is a fragment when it is small *relative to the largest piece on
# the same map*: under a quarter of it, and under six compound nodes (three
# reactions) in absolute terms. Both halves matter -- the share alone would
# call a 40-node pathway a fragment beside a 200-node one, and the floor alone
# would not fire at all on a map whose pieces are all small.
FRAGMENT_SHARE = 0.25
FRAGMENT_FLOOR = 6
GAP_CELL = 40.0           # occupancy grid resolution, map units
MAX_GRID = 220            # cap on grid rows/cols, so placement cost is bounded
FRAGMENT_AIR = 40.0       # air reserved around a fragment, map units

# Fraction of the component gutter kept as air between packed pieces. The
# gutter is sized for separating whole pathways; at full size it is most of
# the white on a map whose pieces are small.
WHITESPACE_SCALE = 0.10

# How far a drawn component reaches past the box the packer reserved for it.
#
# The layered pass sizes a component from the primary metabolites it placed;
# the renderer then adds cofactor stubs and markers. Measured over 510
# components: the overhang past that box is *zero* for 75% of them, 152 units
# at the 90th percentile and 200 at the 95th. So this covers p90.
#
# An earlier version used 240, from measuring how far a stub sits from its
# reaction axis (median 175, p90 236). That was the wrong quantity: most stubs
# are drawn between the component's own nodes, inside the box, and reserving
# for them padded every component by six times what it needed.
NODE_PAD = 160.0


def _fill_gaps(fragments, cores, offsets, members, pos, width, height,
               layering, gap, total_area, target_aspect):
    """Place the small components into the free space around the placed cores.

    The cores are rasterised -- each node with its extent plus the gutter, and
    every proper-layer edge as a line, so a long edge reserves the corridor it
    is drawn through -- into a cell grid. The grid covers a canvas big enough
    for the whole drawing rather than just the cores, because a fragment can
    only be filled into space the grid actually addresses.

    Each fragment, largest first, takes the free block nearest the centre of
    the cores. A hole inside a core is nearer than the margin outside it, so
    the holes are consumed first and the remainder wraps around the outside
    instead of forming a sparse block of its own.
    """
    cell = GAP_CELL
    # One cell of clearance, fixed -- NOT the map-wide gutter.
    #
    # The gutter that separates whole pathway components is up to 420 units,
    # and every fragment was reserving it on both of its sides, so two
    # one-reaction stubs ended up ~880 units apart while the stub itself is
    # about 400 across. That is the white space: it is not left over from
    # anything, it was reserved on purpose, by a constant meant for something
    # much bigger. Small pieces sit close together instead.
    #
    # What the clearance must still cover is that the grid is rasterised from
    # the layout's own `width`/`height`, which know nothing about the label
    # text and cofactor stubs the renderer adds afterwards. That shortfall is
    # handled honestly by NODE_PAD below, applied to the drawn material, and
    # not by inflating the space between fragments.
    margin_cells = 1

    xs, ys = [], []
    for b in cores:
        dx, dy = offsets[b[0]]
        for n in members[b[0]]:
            if n in pos:
                w = width.get(n, 40.0) / 2.0
                h = height.get(n, 40.0) / 2.0
                xs += [pos[n][0] + dx - w, pos[n][0] + dx + w]
                ys += [pos[n][1] + dy - h, pos[n][1] + dy + h]
    if not xs:
        return {}
    cx0, cx1, cy0, cy1 = min(xs), max(xs), min(ys), max(ys)

    # The canvas is the core block grown to hold the whole drawing at the
    # target shape, with the cores anchored at its top-left corner. Centring
    # them instead looks tidier and packs worse: it splits the free space into
    # two half-width side strips, and a fragment's bounding box is routinely
    # wider than either, so it fits in neither and falls through to the
    # overflow. Anchored in a corner the free space stays one contiguous L,
    # which is the shape that actually holds things.
    #
    # Being generous with the canvas costs nothing. Fragments are placed at the
    # free block nearest the *cores*, so they pack inward against them and the
    # unused remainder is simply never drawn on.
    span_w, span_h = cx1 - cx0, cy1 - cy0
    canvas_w = max(span_w + 2 * gap, math.sqrt(total_area * 1.1 * target_aspect))
    canvas_h = max(span_h + 2 * gap, total_area * 1.1 / max(canvas_w, 1.0))
    gx0, gy0 = cx0 - gap, cy0 - gap
    # Coarsen the grid rather than let the cell count explode. The resolution
    # is a spacing choice (GAP_CELL), but the cost is quadratic in it: at 40
    # units a poster-sized canvas is 600x600 cells, and scanning that per
    # fragment was 83% of the whole run time on iAF692 -- a 690-reaction model
    # that should take seconds, not minutes.
    cell = max(cell, canvas_w / MAX_GRID, canvas_h / MAX_GRID)
    margin_cells = max(1, int(round(FRAGMENT_AIR / cell)))
    cols = max(1, int(math.ceil(canvas_w / cell)))
    rows = max(1, int(math.ceil(canvas_h / cell)))
    occupied = [bytearray(cols) for _ in range(rows)]

    def mark_rect(x0, y0, x1, y1):
        c0 = max(0, int((x0 - gx0) // cell) - margin_cells)
        c1 = min(cols - 1, int((x1 - gx0) // cell) + margin_cells)
        r0 = max(0, int((y0 - gy0) // cell) - margin_cells)
        r1 = min(rows - 1, int((y1 - gy0) // cell) + margin_cells)
        for r in range(r0, r1 + 1):
            row = occupied[r]
            for c in range(c0, c1 + 1):
                row[c] = 1

    def mark_line(x0, y0, x1, y1):
        steps = max(1, int(math.hypot(x1 - x0, y1 - y0) / (cell * 0.5)))
        for k in range(steps + 1):
            t = k / steps
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            mark_rect(x, y, x, y)

    core_ids = {}
    for b in cores:
        dx, dy = offsets[b[0]]
        for n in members[b[0]]:
            if n not in pos:
                continue
            core_ids[n] = (dx, dy)
            w = width.get(n, 40.0) / 2.0 + NODE_PAD
            h = height.get(n, 40.0) / 2.0 + NODE_PAD
            x, y = pos[n][0] + dx, pos[n][1] + dy
            mark_rect(x - w, y - h, x + w, y + h)
    # Edges: every proper-layer adjacency, dummy chains included, so a long
    # edge reserves the corridor it is drawn through instead of looking like
    # empty space a fragment may be dropped onto.
    for n, (dx, dy) in core_ids.items():
        for m in layering.next.get(n, ()):
            if m in core_ids:
                mark_line(pos[n][0] + dx, pos[n][1] + dy,
                          pos[m][0] + dx, pos[m][1] + dy)

    # The visit order is computed once. Rebuilding and re-sorting it per
    # fragment is the obvious way to write this and is O(fragments x cells
    # log cells), which on a poster-sized grid is minutes.
    # Ordered by distance from the centre of the *cores*, not of the canvas.
    # That makes a fragment take the free cell that hugs the drawn material --
    # an interior hole if one is big enough, otherwise the margin immediately
    # beside it -- instead of drifting out to the middle of empty space.
    core_r = (cy0 + cy1) / 2.0
    core_c = (cx0 + cx1) / 2.0

    def visit_order(rows, cols):
        cr = (core_r - gy0) / cell
        cc = (core_c - gx0) / cell
        return sorted(range(rows * cols),
                      key=lambda k: (k // cols + 0.5 - cr) ** 2
                                    + (k % cols + 0.5 - cc) ** 2)

    order_cells = visit_order(rows, cols)
    # Where the scan for a given fragment size got to last time. Cells only
    # ever go from free to occupied, never back, so a position rejected for one
    # fragment of a given size is rejected for every later fragment of that
    # size too -- the pointer is safe to carry forward, and it turns a
    # per-fragment scan of the whole grid into one scan per distinct size.
    cursor = {}

    # Each fragment reserves its own bounding box plus the standard gutter,
    # and nothing more. Padding them out to match the cores' average node
    # density was tried and reverted: the density has to be measured over the
    # cores' *combined* bounding box, which includes the empty space between
    # separate cores, so it comes out far too low and every fragment is
    # inflated by up to six times its area. The canvas then has to grow to
    # hold them and the fragments end up strewn across it -- which is the same
    # white space, just evenly distributed. The dense-brick worry that
    # motivated the padding was misdiagnosed anyway: the densest cell on the
    # worst map holds 154 nodes and every one of them belongs to a single
    # core component, so the peak that hairball_index reports is not coming
    # from how the fragments are packed at all.
    out = {}
    for index, left, top, fw, fh in sorted(fragments, key=lambda b: -(b[3] * b[4])):
        # Reserve the fragment's true drawn extent -- its box plus the room the
        # renderer will want for labels and stubs -- then one cell of air. Two
        # neighbours are therefore about 2 cells apart, not two gutters.
        pw, ph = fw + 2 * NODE_PAD, fh + 2 * NODE_PAD
        need_c = int(math.ceil(pw / cell)) + 2 * margin_cells
        need_r = int(math.ceil(ph / cell)) + 2 * margin_cells
        slot = None
        key = (need_r, need_c)
        for _attempt in range(3):
            for scan in range(cursor.get(key, 0), len(order_cells)):
                r, c = divmod(order_cells[scan], cols)
                if occupied[r][c] or r + need_r > rows or c + need_c > cols:
                    continue
                if not any(any(occupied[rr][c:c + need_c])
                           for rr in range(r, r + need_r)):
                    slot = (r, c)
                    cursor[key] = scan + 1
                    break
            if slot is not None:
                break
            # Genuinely full. Extend on whichever axis keeps the drawing
            # closest to the target shape -- always extending downward is what
            # turned the overflow into a column.
            if need_c > cols:
                for row in occupied:
                    row.extend(bytearray(need_c - cols))
                cols = need_c
            if need_r > rows or float(cols) / max(rows, 1) > target_aspect:
                for _ in range(need_r):
                    occupied.append(bytearray(cols))
                rows = len(occupied)
            else:
                for row in occupied:
                    row.extend(bytearray(need_c))
                cols += need_c
            order_cells = visit_order(rows, cols)
            cursor.clear()          # those indices referred to the old grid
        if slot is None:
            continue
        r, c = slot
        # Centred in the block it reserved, so the density padding sits around
        # the fragment rather than all on one side of it.
        inner_w = max((need_c - 2 * margin_cells) * cell, pw)
        inner_h = max((need_r - 2 * margin_cells) * cell, ph)
        x = gx0 + (c + margin_cells) * cell + (inner_w - fw) / 2.0
        y = gy0 + (r + margin_cells) * cell + (inner_h - fh) / 2.0
        out[index] = (x - left, y - top)
        for rr in range(r, min(rows, r + need_r)):
            row = occupied[rr]
            for cc in range(c, min(cols, c + need_c)):
                row[cc] = 1
    return out


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
