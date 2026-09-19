"""Layered (Sugiyama) drawing of the primary-compound digraph
(layout_algorithm.md S3).

Replaces the v1 simulated-annealing placement. The four classical stages are:

    1. cycle removal        greedy feedback arc set, biased to reverse
                            reversible reactions rather than irreversible ones
    2. layer assignment     longest-path, with exchange metabolites pinned to
                            the top and bottom edges of the drawing
    3. crossing reduction   median heuristic + adjacent transpose sweeps
    4. x-coordinates        Brandes-Koepf

Stage 4 is the one that matters aesthetically: it explicitly aligns chains of
degree-1 nodes into straight vertical runs, which is what makes glycolysis read
as a spine instead of a staircase. A barycenter x-assignment does not do this.
"""

import math
import sys
from dataclasses import dataclass, field

INF = float("inf")


@dataclass
class ProperLayering:
    layers: list = field(default_factory=list)      # list[list[node]], ordered
    layer_of: dict = field(default_factory=dict)    # node -> layer index
    prev: dict = field(default_factory=dict)        # node -> [nodes in layer-1]
    next: dict = field(default_factory=dict)        # node -> [nodes in layer+1]
    dummies: dict = field(default_factory=dict)     # dummy id -> (u, v) original edge
    chains: dict = field(default_factory=dict)      # (u, v) -> [dummy ids, top to bottom]
    reversed_edges: set = field(default_factory=set)


# --------------------------------------------------------------------------
# 1. cycle removal
# --------------------------------------------------------------------------

def greedy_feedback_arc_set(D, reversal_cost=None):
    """Eades-Lin-Smyth greedy linear arrangement; returns edges to reverse.

    `reversal_cost(u, v) -> float` lets an irreversible reaction resist being
    drawn against its thermodynamic direction while a reversible one flips
    freely. Metabolic networks are full of cycles, so which arcs get reversed
    decides whether the drawing reads with the biology or against it.
    """
    if reversal_cost is None:
        reversal_cost = lambda u, v: 1.0

    remaining = set(D.nodes)
    out_w = {v: 0.0 for v in remaining}
    in_w = {v: 0.0 for v in remaining}
    for u, v in D.edges:
        w = reversal_cost(u, v)
        out_w[u] += w
        in_w[v] += w

    succ = {v: set(D.successors(v)) for v in remaining}
    pred = {v: set(D.predecessors(v)) for v in remaining}

    head, tail = [], []

    def drop(v):
        remaining.discard(v)
        for w in succ[v]:
            if w in remaining:
                in_w[w] -= reversal_cost(v, w)
                pred[w].discard(v)
        for w in pred[v]:
            if w in remaining:
                out_w[w] -= reversal_cost(w, v)
                succ[w].discard(v)

    while remaining:
        moved = True
        while moved:
            moved = False
            for v in [v for v in remaining if out_w[v] <= 1e-12]:
                if v in remaining:
                    tail.append(v)
                    drop(v)
                    moved = True
            for v in [v for v in remaining if in_w[v] <= 1e-12]:
                if v in remaining:
                    head.append(v)
                    drop(v)
                    moved = True
        if remaining:
            v = max(remaining, key=lambda n: (out_w[n] - in_w[n], str(n)))
            head.append(v)
            drop(v)

    order = {v: i for i, v in enumerate(head + tail[::-1])}
    return {(u, v) for u, v in D.edges if order[u] > order[v]}


# --------------------------------------------------------------------------
# 2. layer assignment
# --------------------------------------------------------------------------

def assign_layers(acyclic_edges, nodes, sinks=()):
    """Longest-path layering. Returns {node: layer}."""
    succ, pred = {n: [] for n in nodes}, {n: [] for n in nodes}
    for u, v in acyclic_edges:
        succ[u].append(v)
        pred[v].append(u)

    layer, visiting = {}, set()
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

    def depth(v):
        if v in layer:
            return layer[v]
        if v in visiting:           # defensive: should be acyclic by now
            return 0
        visiting.add(v)
        layer[v] = max((depth(u) + 1 for u in pred[v]), default=0)
        visiting.discard(v)
        return layer[v]

    for n in nodes:
        depth(n)

    # Secreted / demanded metabolites belong at the bottom edge of the drawing,
    # the way curated maps put extracellular products on the border. Only pull
    # down true dead ends, so nothing mid-pathway gets stretched.
    if layer:
        bottom = max(layer.values())
        for n in sinks:
            if n in layer and not succ.get(n) and (len(pred.get(n, ())) > 1):
                layer[n] = bottom

    _tighten_leaves(layer, pred, succ)
    if layer:
        base = min(layer.values())
        if base:
            for n in layer:
                layer[n] -= base
    return layer


def _tighten_leaves(layer, pred, succ, passes=4):
    """Pull degree-1 nodes next to their single neighbour.

    Longest-path layering puts every source on layer 0, which for a metabolic
    model means every uptake metabolite sits in one row at the top with a
    column of empty space running down to wherever it is actually consumed --
    that is most of the whitespace in a naive layered map. A transported
    metabolite has exactly one neighbour, so it can simply sit beside it, which
    is also where a curated map draws it.
    """
    for _ in range(passes):
        changed = False
        for node in list(layer):
            ups, downs = pred.get(node, ()), succ.get(node, ())
            if len(ups) + len(downs) != 1:
                continue
            target = (layer[ups[0]] + 1) if ups else (layer[downs[0]] - 1)
            if target != layer[node]:
                layer[node] = target
                changed = True
        if not changed:
            break


# --------------------------------------------------------------------------
# 3. proper layering (dummy nodes) + crossing reduction
# --------------------------------------------------------------------------

def build_proper_layering(edges, layer, reversed_edges):
    """Split multi-layer edges with dummy nodes; returns a ProperLayering."""
    pl = ProperLayering(reversed_edges=set(reversed_edges))
    height = (max(layer.values()) + 1) if layer else 0
    pl.layers = [[] for _ in range(height)]
    pl.layer_of = dict(layer)

    for n, i in layer.items():
        pl.layers[i].append(n)
        pl.prev[n] = []
        pl.next[n] = []

    for u, v in edges:
        span = layer[v] - layer[u]
        if span <= 0:
            continue
        chain = []
        upper = u
        for i in range(layer[u] + 1, layer[v]):
            d = f"__dummy__{u}__{v}__{i}"
            pl.dummies[d] = (u, v)
            pl.layer_of[d] = i
            pl.layers[i].append(d)
            pl.prev[d] = [upper]
            pl.next[d] = []
            pl.next[upper].append(d)
            chain.append(d)
            upper = d
        pl.next[upper].append(v)
        pl.prev[v].append(upper)
        pl.chains[(u, v)] = chain

    return pl


def _crossings_between(pl, i, index):
    """Crossing count between layers i and i+1 for the current ordering."""
    pairs = []
    for v in pl.layers[i]:
        for w in pl.next[v]:
            pairs.append((index[v], index[w]))
    pairs.sort()
    count = 0
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            if pairs[b][1] < pairs[a][1]:
                count += 1
    return count


def total_crossings(pl):
    index = {v: k for layer in pl.layers for k, v in enumerate(layer)}
    return sum(_crossings_between(pl, i, index) for i in range(len(pl.layers) - 1))


def reduce_crossings(pl, groups=None, sweeps=8):
    """Median heuristic + adjacent transpose.

    `groups` maps a node to a cluster key (compartment, or subsystem in the
    meta-layout). Nodes are kept contiguous within their cluster during
    ordering, which is how compartments stay in one piece instead of being
    interleaved by the crossing minimiser.
    """
    if not pl.layers:
        return pl
    groups = groups or {}

    def group_key(v):
        if v in pl.dummies:
            u, w = pl.dummies[v]
            return groups.get(u, groups.get(w, ""))
        return groups.get(v, "")

    def median_value(v, neighbours, index):
        positions = sorted(index[n] for n in neighbours if n in index)
        if not positions:
            return -1.0
        m = len(positions) // 2
        if len(positions) % 2 == 1:
            return float(positions[m])
        return (positions[m - 1] + positions[m]) / 2.0

    def reorder(target_layer, neighbour_of, index):
        medians = {}
        for k, v in enumerate(pl.layers[target_layer]):
            med = median_value(v, neighbour_of[v], index)
            medians[v] = med if med >= 0 else float(k)

        # Groups are ordered by where their members actually want to be, not
        # by name. Sorting on the key itself keeps clusters contiguous but
        # imposes an arbitrary (alphabetical) left-to-right order on the
        # pathways, which costs more crossings than the clustering saves.
        buckets = {}
        for v in pl.layers[target_layer]:
            buckets.setdefault(group_key(v), []).append(medians[v])
        anchor = {g: sum(vals) / len(vals) for g, vals in buckets.items()}

        keyed = [(anchor[group_key(v)], group_key(v), medians[v], k, v)
                 for k, v in enumerate(pl.layers[target_layer])]
        keyed.sort()
        pl.layers[target_layer] = [v for _, _, _, _, v in keyed]

    best_order = [list(l) for l in pl.layers]
    best_cost = total_crossings(pl)

    for sweep in range(sweeps):
        index = {v: k for layer in pl.layers for k, v in enumerate(layer)}
        if sweep % 2 == 0:
            for i in range(1, len(pl.layers)):
                reorder(i, pl.prev, index)
                index = {v: k for layer in pl.layers for k, v in enumerate(layer)}
        else:
            for i in range(len(pl.layers) - 2, -1, -1):
                reorder(i, pl.next, index)
                index = {v: k for layer in pl.layers for k, v in enumerate(layer)}

        _transpose(pl, group_key)
        cost = total_crossings(pl)
        if cost < best_cost:
            best_cost = cost
            best_order = [list(l) for l in pl.layers]

    pl.layers = best_order
    return pl


def _transpose(pl, group_key):
    improved = True
    while improved:
        improved = False
        for i, layer in enumerate(pl.layers):
            index = {v: k for l in pl.layers for k, v in enumerate(l)}
            for k in range(len(layer) - 1):
                v, w = layer[k], layer[k + 1]
                if group_key(v) != group_key(w):
                    continue
                before = _pair_crossings(pl, v, w, index)
                index[v], index[w] = index[w], index[v]
                after = _pair_crossings(pl, v, w, index)
                if after < before:
                    layer[k], layer[k + 1] = w, v
                    improved = True
                else:
                    index[v], index[w] = index[w], index[v]


def _pair_crossings(pl, v, w, index):
    count = 0
    for direction in (pl.prev, pl.next):
        a = sorted(index[n] for n in direction[v] if n in index)
        b = sorted(index[n] for n in direction[w] if n in index)
        for x in a:
            for y in b:
                if y < x:
                    count += 1
    return count


# --------------------------------------------------------------------------
# 4. x-coordinates: Brandes-Koepf
# --------------------------------------------------------------------------

def brandes_koepf(pl, width, gap):
    """Assign an x coordinate per node. `width[v]` is the node's drawn width."""
    if not pl.layers:
        return {}

    marked = _mark_type1_conflicts(pl)
    runs = []
    for vertical in ("down", "up"):
        for horizontal in ("left", "right"):
            x = _aligned_layout(pl, marked, width, gap, vertical, horizontal)
            if horizontal == "right":
                x = {v: -c for v, c in x.items()}
            runs.append(x)
    return _enforce_separation(pl, _balance(runs), width, gap)


def _enforce_separation(pl, x, width, gap):
    """Guarantee the minimum gap between neighbours in a layer.

    Each of the four alignment runs satisfies the separation constraints, but
    the balancing step takes a per-node median across them, and a median picks
    a different run for different nodes -- so it is not a convex combination
    and can produce overlapping nodes. One left-to-right sweep per layer costs
    nothing and makes the guarantee unconditional.
    """
    for layer in pl.layers:
        for i in range(1, len(layer)):
            left, right = layer[i - 1], layer[i]
            needed = (width.get(left, 0.0) + width.get(right, 0.0)) / 2.0 + gap
            if x[right] - x[left] < needed:
                x[right] = x[left] + needed
    return x


def _has_inner_segment(pl, v):
    return v in pl.dummies and any(u in pl.dummies for u in pl.prev[v])


def _mark_type1_conflicts(pl):
    """Type-1 conflicts: a segment crossing an inner (dummy-to-dummy) segment.

    These are the edges Brandes-Koepf refuses to straighten, so that long
    edges -- which in a metabolic map are exactly the pathway backbones
    spanning several reactions -- win the competition for a straight line.
    """
    marked = set()
    index = {v: k for layer in pl.layers for k, v in enumerate(layer)}

    for i in range(1, len(pl.layers) - 1):
        lower = pl.layers[i + 1]
        k0, l = 0, 0
        for l1, v in enumerate(lower):
            inner = _has_inner_segment(pl, v)
            if l1 == len(lower) - 1 or inner:
                k1 = len(pl.layers[i]) - 1
                if inner:
                    for u in pl.prev[v]:
                        if u in pl.dummies:
                            k1 = index[u]
                            break
                while l <= l1:
                    for u in pl.prev[lower[l]]:
                        k = index[u]
                        if k < k0 or k > k1:
                            marked.add(frozenset((u, lower[l])))
                    l += 1
                k0 = k1
    return marked


def _aligned_layout(pl, marked, width, gap, vertical, horizontal):
    layers = [list(l) for l in pl.layers]
    if vertical == "up":
        layers.reverse()
    if horizontal == "right":
        layers = [l[::-1] for l in layers]

    upper_of = {}
    for i, layer in enumerate(layers):
        for v in layer:
            if i == 0:
                upper_of[v] = []
            else:
                above = set(layers[i - 1])
                nbrs = (pl.prev[v] if vertical == "down" else pl.next[v])
                upper_of[v] = [u for u in nbrs if u in above]

    index = {v: k for layer in layers for k, v in enumerate(layer)}
    for layer in layers:
        for v in layer:
            upper_of[v].sort(key=lambda u: index[u])

    # --- vertical alignment: build blocks of nodes that share an x ---
    root = {v: v for layer in layers for v in layer}
    align = {v: v for layer in layers for v in layer}

    for i in range(1, len(layers)):
        r = -1
        for v in layers[i]:
            ups = upper_of[v]
            if not ups:
                continue
            m = (len(ups) - 1) / 2.0
            for mi in {int(m), int(m + 0.5)}:
                if align[v] != v:
                    break
                u = ups[mi]
                if frozenset((u, v)) in marked:
                    continue
                if r < index[u]:
                    align[u] = v
                    root[v] = root[u]
                    align[v] = root[v]
                    r = index[u]

    # --- horizontal compaction ---
    sink = {v: v for v in root}
    shift = {v: INF for v in root}
    x = {}
    left_of = {}
    for layer in layers:
        for k, v in enumerate(layer):
            left_of[v] = layer[k - 1] if k > 0 else None

    def separation(a, b):
        return (width.get(a, 0.0) + width.get(b, 0.0)) / 2.0 + gap

    def place_block(v):
        if v in x:
            return
        x[v] = 0.0
        w = v
        while True:
            left = left_of[w]
            if left is not None:
                u = root[left]
                place_block(u)
                if sink[v] == v:
                    sink[v] = sink[u]
                if sink[v] != sink[u]:
                    shift[sink[u]] = min(shift[sink[u]], x[v] - x[u] - separation(left, w))
                else:
                    x[v] = max(x[v], x[u] + separation(left, w))
            w = align[w]
            if w == v:
                break

    for v in root:
        if root[v] == v:
            place_block(v)

    result = {}
    for v in root:
        base = x[root[v]]
        s = shift[sink[root[v]]]
        result[v] = base + (s if s < INF else 0.0)
    return result


def _balance(runs):
    """Combine the four alignment runs: align their extents, take the median."""
    widths = [max(r.values()) - min(r.values()) if r else 0.0 for r in runs]
    reference = runs[widths.index(min(widths))]
    ref_lo, ref_hi = min(reference.values()), max(reference.values())

    aligned = []
    for i, run in enumerate(runs):
        lo, hi = min(run.values()), max(run.values())
        # runs 0 and 2 are left-biased, 1 and 3 right-biased (already negated)
        delta = (ref_lo - lo) if i % 2 == 0 else (ref_hi - hi)
        aligned.append({v: c + delta for v, c in run.items()})

    out = {}
    for v in runs[0]:
        values = sorted(run[v] for run in aligned)
        out[v] = (values[1] + values[2]) / 2.0
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def layered_layout(D, width, height, sinks=(), groups=None, x_gap=40.0, y_gap=120.0,
                   fold_after=26, max_width=None,
                   reversal_cost=None, sweeps=8):
    """Full Sugiyama pass. Returns (positions, ProperLayering).

    `width`/`height` are per-node drawn extents; layer pitch adapts to the
    tallest node in each layer so that a ring motif can occupy a whole band
    without colliding with its neighbours.
    """
    reverse = greedy_feedback_arc_set(D, reversal_cost)
    acyclic = [(v, u) if (u, v) in reverse else (u, v) for u, v in D.edges]
    acyclic = [(u, v) for u, v in acyclic if u != v]

    layer = assign_layers(acyclic, list(D.nodes), sinks=sinks)
    pl = build_proper_layering(acyclic, layer, reverse)
    reduce_crossings(pl, groups=groups, sweeps=sweeps)

    dummy_width = 1.0
    full_width = dict(width)
    for d in pl.dummies:
        full_width[d] = dummy_width

    xs = brandes_koepf(pl, full_width, x_gap)

    ys, cursor = {}, 0.0
    for layer_nodes in pl.layers:
        tallest = max((height.get(v, 0.0) for v in layer_nodes), default=0.0)
        cursor += tallest / 2.0
        for v in layer_nodes:
            ys[v] = cursor
        cursor += tallest / 2.0 + y_gap

    # Wrapping runs before folding: it changes the layer stack's height, which
    # is the quantity folding then decides on.
    xs, ys = wrap_wide_layers(pl, xs, ys, full_width, height, x_gap, y_gap,
                              max_width)
    xs, ys = _fold_columns(pl, xs, ys, full_width, height, x_gap, y_gap, fold_after)

    pos = {v: (xs.get(v, 0.0), ys.get(v, 0.0)) for v in xs}
    return pos, pl


def wrap_wide_layers(pl, xs, ys, width, height, x_gap, y_gap, max_width):
    """Re-flow layers that are far wider than they are tall into sub-rows.

    `_fold_columns` handles the opposite problem -- a pathway so long it becomes
    an unreadably tall column. A merged functional map has the reverse shape:
    few layers, each enormously wide. Lipid metabolism in Recon3D is ~3000
    reactions at a shallow topological depth, so one layer holds a thousand
    nodes and the drawing comes out 500,000 units wide and 135,000 tall -- a
    horizontal smear with no readable structure, measured at 18.8:1 for the
    worst tile.

    Nodes keep their within-layer order, so the crossing reduction that ran
    earlier is not thrown away; each over-wide layer is simply cut into
    consecutive chunks and stacked. Layers below shift down to make room.

    This is what a curated global map does with a wide pathway family -- KEGG's
    map01100 wraps them into rectangular blocks rather than drawing one long
    line.
    """
    if not max_width:
        return xs, ys

    def node_width(v):
        return width.get(v, 0.0)

    # Only re-flow components that are themselves too wide.
    #
    # Re-flowing a layer rebuilds its x-coordinates from the within-layer
    # order, starting at x=0, which throws away the Brandes-Koepf assignment
    # that aligns a reaction's substrate above its product. Done to every layer
    # that is a fair trade on one enormous component and pure damage on a map
    # of small ones: a substrate at order-position i in layer L and its product
    # at position j in layer L+1 get unrelated x, so the edge becomes a
    # diagonal and the router draws it as a wide Z. On RECON1's transport map
    # -- 66 components, the largest 4% of nodes -- single transport reactions
    # came out 2460 x 1820 units, and only 4 of 96 reactions were drawn
    # vertically.
    #
    # Components that fit are left exactly where Brandes-Koepf put them.
    # `_pack_components` arranges them against each other afterwards, so
    # nothing here needs to lay them out relative to one another.
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for layer in pl.layers:
        for v in layer:
            find(v)
            for w in pl.next.get(v, ()):
                ra, rb = find(v), find(w)
                if ra != rb:
                    parent[ra] = rb

    span = {}
    for layer in pl.layers:
        per_component = {}
        for v in layer:
            root = find(v)
            per_component[root] = per_component.get(root, 0.0) + node_width(v) + x_gap
        for root, used in per_component.items():
            span[root] = max(span.get(root, 0.0), used)

    wide = {root for root, used in span.items() if used > max_width}
    if not wide:
        return xs, ys

    new_xs, new_ys = dict(xs), dict(ys)
    cursor = None

    for layer in pl.layers:
        layer = [v for v in layer if find(v) in wide]
        if not layer:
            continue
        ordered = sorted(layer, key=lambda v: xs[v])
        layer_top = min(ys[v] for v in layer) if cursor is None else cursor

        # Cut into chunks that each fit the width budget.
        chunks, current, used = [], [], 0.0
        for v in ordered:
            w = node_width(v) + x_gap
            if current and used + w > max_width:
                chunks.append(current)
                current, used = [], 0.0
            current.append(v)
            used += w
        if current:
            chunks.append(current)

        row_y = layer_top
        for chunk in chunks:
            x = 0.0
            row_height = max((height.get(v, 0.0) for v in chunk), default=0.0)
            for v in chunk:
                w = node_width(v)
                new_xs[v] = x + w / 2.0
                new_ys[v] = row_y + row_height / 2.0
                x += w + x_gap
            row_y += row_height + y_gap
        cursor = row_y

    return new_xs, new_ys


def _fold_columns(pl, xs, ys, width, height, x_gap, y_gap, fold_after):
    """Wrap a very tall stack of layers into boustrophedon columns.

    A forty-step linear pathway is correct as one column and unreadable as one
    column. Curated maps fold long runs back on themselves -- the fatty-acid
    elongation ladders in `templates/t4` are exactly this -- so past a height
    where the drawing stops fitting on a page, break the layer stack into
    columns and reverse every other one, which keeps the two layers either side
    of a fold adjacent instead of a page apart.

    Left alone below the threshold: a ten-step pathway drawn as a tall column
    is what a reader expects, and folding it would be worse.
    """
    count = len(pl.layers)
    if not fold_after or count <= fold_after:
        return xs, ys

    spans = [
        (min(xs[v] - width.get(v, 0.0) / 2.0 for v in layer),
         max(xs[v] + width.get(v, 0.0) / 2.0 for v in layer))
        if layer else (0.0, 0.0)
        for layer in pl.layers
    ]
    column_width = max(hi - lo for lo, hi in spans) + 4.0 * x_gap
    left_edge = min(lo for lo, _ in spans)

    layer_heights = [max((height.get(v, 0.0) for v in layer), default=0.0)
                     for layer in pl.layers]
    total_height = sum(layer_heights) + y_gap * max(count - 1, 0)

    best, best_cost = 1, None
    for columns in range(1, count + 1):
        rows = math.ceil(count / columns)
        block_h = total_height * rows / count
        block_w = column_width * columns
        cost = abs(math.log(max(block_w, 1.0) / max(block_h, 1.0)))
        if best_cost is None or cost < best_cost:
            best_cost, best = cost, columns
    if best <= 1:
        return xs, ys

    rows_per_column = math.ceil(count / best)
    placement = {}
    for index in range(count):
        column, row = divmod(index, rows_per_column)
        if column % 2 == 1:
            row = rows_per_column - 1 - row
        placement[index] = (column, row)

    row_height = {}
    for index, (_, row) in placement.items():
        row_height[row] = max(row_height.get(row, 0.0), layer_heights[index])

    row_centre, cursor = {}, 0.0
    for row in sorted(row_height):
        row_centre[row] = cursor + row_height[row] / 2.0
        cursor += row_height[row] + y_gap

    new_xs, new_ys = dict(xs), dict(ys)
    for index, layer in enumerate(pl.layers):
        column, row = placement[index]
        for v in layer:
            new_xs[v] = xs[v] + column * column_width - left_edge
            new_ys[v] = row_centre[row]
    return new_xs, new_ys
