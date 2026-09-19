"""Compose per-cluster drawings into one whole-model map
(layout_algorithm.md S6, meta-tiling).

Laying a genome-scale model out as a single layered drawing is correct and
unreadable: iAF1260 comes out at 2.6 crossings per edge with a hairball index
of 7.3, because one Sugiyama pass has no reason to keep pathways apart.

So the same algorithm runs twice, at two scales. Each cluster is drawn on its
own, then each drawing becomes a *box* and the boxes are laid out by the same
layered pass -- pathways have a real flow between them (central carbon in the
middle, biosynthesis radiating outward), so a layered meta-layout reproduces
that arrangement instead of dropping tiles on an arbitrary grid.

Metabolites shared between clusters are left duplicated rather than wired
across tiles. That is what curated maps do: `templates/t4` repeats currency and
branch-point compounds in every pathway it appears in, because a few hundred
cross-map connector lines would bury the pathways they connect.
"""

import math

import networkx as nx

from .render import (CHAR_WIDTH_RATIO, ESCHER_DEFAULT_FONT_BASE, LINE_HEIGHT_RATIO,
                     METABOLITE_FONT_FACTOR, REACTION_FONT_FACTOR)
from . import taxonomy
from .sugiyama import layered_layout

TILE_GAP = 260.0           # clear space between tiles
TITLE_OFFSET = 150.0       # cluster caption above its tile
REGION_GAP = 480.0         # clear space between biological regions
REGION_TITLE_OFFSET = 420.0
REGION_ASPECT = 1.5        # shape of a single region block
CANVAS_ASPECT = 1.4        # shape of the whole poster
REGION_FONT_BASE = 30.0
CANVAS_PADDING = 700.0


def build_meta_graph(clusters, cofactor_score, cutoff=0.5):
    """Directed graph over cluster names; A -> B when A makes what B consumes.

    Currency is excluded, for the same reason it is excluded from community
    detection: ATP and water link every pathway to every other, and an edge
    that is always present carries no layout information.
    """
    produced, consumed = {}, {}
    for name, reactions in clusters.items():
        makes, takes = set(), set()
        for reaction in reactions:
            for metabolite, coefficient in reaction.metabolites.items():
                if cofactor_score.get(metabolite.id, 0.0) >= cutoff:
                    continue
                (makes if coefficient > 0 else takes).add(metabolite.id)
                if reaction.lower_bound < 0 < reaction.upper_bound:
                    (takes if coefficient > 0 else makes).add(metabolite.id)
        produced[name] = makes
        consumed[name] = takes

    meta = nx.DiGraph()
    meta.add_nodes_from(clusters)
    for a in clusters:
        for b in clusters:
            if a == b:
                continue
            shared = produced[a] & consumed[b]
            if shared:
                # The shared metabolites themselves, not just how many. Tile
                # placement only needs the count, but drawing a connector needs
                # to know which compound to hang it on.
                meta.add_edge(a, b, weight=len(shared), rxns=[],
                              mets=sorted(shared))
    return meta


def _bbox(escher_map):
    """Extent of a tile, including the space its labels actually occupy.

    `label_x`/`label_y` is the *left edge and vertical centre* of a label, not
    its extent. Treating it as a point understates the tile by most of a label
    width, so tiles packed edge to edge end up with their captions overlapping
    the neighbouring tile -- which is where this map's remaining label
    collisions were coming from.
    """
    body = escher_map[1]
    nodes = body["nodes"]
    if not nodes:
        return 0.0, 0.0, 1.0, 1.0

    xs, ys = [], []

    def add_label(holder, text, factor):
        size = holder.get("font_size_base", ESCHER_DEFAULT_FONT_BASE) * factor
        width = max(len(str(text)), 1) * size * CHAR_WIDTH_RATIO
        height = size * LINE_HEIGHT_RATIO
        xs.extend((holder["label_x"], holder["label_x"] + width))
        ys.extend((holder["label_y"] - height / 2.0, holder["label_y"] + height / 2.0))

    for node in nodes.values():
        xs.append(node["x"])
        ys.append(node["y"])
        if "label_x" in node:
            add_label(node, node.get("bigg_id", ""), METABOLITE_FONT_FACTOR)
    for reaction in body["reactions"].values():
        add_label(reaction, reaction.get("bigg_id", ""), REACTION_FONT_FACTOR)

    return min(xs), min(ys), max(xs), max(ys)


def _offset_tile(escher_map, dx, dy, prefix, out_nodes, out_reactions):
    """Copy one tile's nodes and reactions into the combined map, shifted."""
    remap = {}
    for node_id, node in escher_map[1]["nodes"].items():
        new_id = f"{prefix}_{node_id}"
        remap[node_id] = new_id
        moved = dict(node)
        moved["x"] += dx
        moved["y"] += dy
        if "label_x" in moved:
            moved["label_x"] += dx
            moved["label_y"] += dy
        out_nodes[new_id] = moved

    for reaction_id, reaction in escher_map[1]["reactions"].items():
        moved = dict(reaction)
        moved["label_x"] += dx
        moved["label_y"] += dy
        moved["segments"] = {
            f"{prefix}_{segment_id}": {
                "from_node_id": remap[segment["from_node_id"]],
                "to_node_id": remap[segment["to_node_id"]],
                "b1": _shift_point(segment.get("b1"), dx, dy),
                "b2": _shift_point(segment.get("b2"), dx, dy),
            }
            for segment_id, segment in reaction["segments"].items()
            if segment["from_node_id"] in remap and segment["to_node_id"] in remap
        }
        # Reaction ids are unique within a model, but a reaction split across
        # two clusters would silently overwrite one of them.
        key = reaction_id if reaction_id not in out_reactions else f"{reaction_id}__{prefix}"
        out_reactions[key] = moved


def _shift_point(point, dx, dy):
    if not point:
        return None
    return {"x": point["x"] + dx, "y": point["y"] + dy}


# --------------------------------------------------------------------------
# stitching tiles back together
# --------------------------------------------------------------------------

STITCH_MIN_WEIGHT = 4      # shared primary metabolites before a link is drawn
STITCH_PER_TILE = 1        # strongest links kept per tile, each direction
STITCH_CAP = 400           # absolute ceiling on connectors drawn
STITCH_MAX_SPAN = 0.18     # longest connector, as a fraction of canvas diagonal


def select_links(meta_graph, per_tile=STITCH_PER_TILE,
                 min_weight=STITCH_MIN_WEIGHT, cap=STITCH_CAP):
    """The inter-tile links worth drawing.

    Drawing every meta edge is not an option and never was: Recon3D's 307 tiles
    share 4224 directed links, 13.8 per tile, and half of them rest on a single
    shared metabolite. Rendered, that is precisely the hairball the two-scale
    layout exists to prevent -- which is why `compose` originally drew none at
    all and left shared compounds duplicated, as `templates/t4` does.

    The middle position is to draw the *trunk*. Each tile keeps only its
    strongest incoming and outgoing link, and only when the two tiles share
    enough chemistry for the link to mean something. On Recon3D that is ~500
    connectors rather than 4224, and what survives is the backbone of flow
    between pathways -- the thing a metro map draws and a KEGG map implies.
    """
    keep = {}
    for node in meta_graph.nodes:
        for edges in (meta_graph.out_edges(node, data=True),
                      meta_graph.in_edges(node, data=True)):
            ranked = sorted(edges, key=lambda e: -e[2].get("weight", 1))
            for u, v, data in ranked[:per_tile]:
                if data.get("weight", 1) < min_weight:
                    continue
                keep[(u, v)] = data
    if len(keep) <= cap:
        return keep
    strongest = sorted(keep.items(), key=lambda kv: -kv[1].get("weight", 1))
    return dict(strongest[:cap])


def _primary_indices(nodes):
    """{tile prefix: {bigg_id: node id}} over every tile at once.

    Built in a single pass. Scanning the node table once per link instead looks
    harmless at ten tiles and is quadratic at three hundred: the Recon3D poster
    has 67k nodes and ~500 links, which is 54M dictionary probes for an index
    that never changes.
    """
    indices = {}
    for node_id, node in nodes.items():
        if node.get("node_type") != "metabolite":
            continue
        bigg = node.get("bigg_id")
        if bigg is None:
            continue
        prefix, _, _ = node_id.partition("_")
        index = indices.setdefault(prefix, {})
        # Prefer a primary instance; a cofactor stub is a poor anchor because
        # it sits off the backbone and the connector would point at nothing.
        if bigg not in index or node.get("node_is_primary"):
            index[bigg] = node_id
    return indices


def stitch(links, tile_prefix, nodes, reactions):
    """Emit one connector per selected link, anchored on a shared metabolite.

    The connector is a real Escher reaction with a single segment, so Escher and
    the preview both draw it without special-casing, and it is bowed with Bezier
    controls so a long link reads as a connector rather than as another backbone
    edge running through the poster.
    """
    # A connector is only worth drawing between tiles the eye can associate.
    #
    # Measured the hard way: drawing every selected link regardless of distance
    # put 232 diagonals across the Recon3D poster and pushed crossings per edge
    # from 0.139 to 0.190. At ten tiles that reads as a flow diagram; at three
    # hundred it is a veil over the whole figure, which is why `compose`
    # originally drew nothing. Capping the span keeps the local stitching, where
    # a reader can actually follow the line to the other end, and drops the
    # poster-spanning ones that only add ink.
    span_limit = None
    if nodes:
        xs = [n["x"] for n in nodes.values()]
        ys = [n["y"] for n in nodes.values()]
        diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        span_limit = diagonal * STITCH_MAX_SPAN

    indices = _primary_indices(nodes)
    drawn = 0
    for (source, target), data in links.items():
        from_prefix = tile_prefix.get(source)
        to_prefix = tile_prefix.get(target)
        if from_prefix is None or to_prefix is None:
            continue
        from_index = indices.get(from_prefix, {})
        to_index = indices.get(to_prefix, {})

        anchor = None
        for metabolite in data.get("mets", ()):
            if metabolite in from_index and metabolite in to_index:
                anchor = metabolite
                break
        if anchor is None:
            continue

        a_id, b_id = from_index[anchor], to_index[anchor]
        a, b = nodes[a_id], nodes[b_id]
        mid_x, mid_y = (a["x"] + b["x"]) / 2.0, (a["y"] + b["y"]) / 2.0
        # Bow perpendicular to the run, a tenth of its length, so parallel
        # connectors between the same pair of regions stay distinguishable.
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        span = math.hypot(dx, dy) or 1.0
        if span_limit is not None and span > span_limit:
            continue
        bow_x, bow_y = -dy / span * span * 0.1, dx / span * span * 0.1

        key = f"link_{drawn}_{anchor}"
        reactions[key] = {
            "name": f"{source} -> {target} ({anchor})",
            "bigg_id": anchor,
            "reversibility": False,
            "label_x": mid_x + bow_x,
            "label_y": mid_y + bow_y,
            "gene_reaction_rule": "",
            "genes": [],
            "metabolites": [
                {"bigg_id": anchor, "coefficient": -1.0},
                {"bigg_id": anchor, "coefficient": 1.0},
            ],
            "segments": {
                f"{key}_s1": {
                    "from_node_id": a_id,
                    "to_node_id": b_id,
                    "b1": {"x": a["x"] + dx * 0.25 + bow_x,
                           "y": a["y"] + dy * 0.25 + bow_y},
                    "b2": {"x": a["x"] + dx * 0.75 + bow_x,
                           "y": a["y"] + dy * 0.75 + bow_y},
                }
            },
        }
        drawn += 1
    return drawn


def _skyline_pack(order, width, height, gap, target_width):
    """Pack tiles into a block, filling the holes a shelf leaves behind.

    Shelf packing makes every row as tall as its tallest tile, so a row holding
    one tall pathway and three short ones is mostly empty. A skyline tracks the
    current occupied height across the block and drops each tile into the
    lowest place it fits, which closes those holes.

    `order` is kept as the placement order, so the metabolic flow the meta-
    layout worked out still determines which tile goes where first.
    """
    columns = []          # (x_start, x_end, height) skyline segments
    placed = {}

    def height_at(x0, x1):
        top = 0.0
        for start, end, level in columns:
            if start < x1 and x0 < end:
                top = max(top, level)
        return top

    for name in order:
        w, h = width[name] + gap, height[name] + gap
        # Candidate positions are the left *and* right edge of every placed
        # tile. Taking only the right edges, which is the easy mistake, means a
        # tile can never be tucked against the left side of a taller neighbour,
        # and the packer leaves half the block empty.
        candidates = sorted({0.0}
                            | {start for start, _, _ in columns}
                            | {end for _, end, _ in columns})
        best_x, best_y = 0.0, None
        for x in candidates:
            if x + w > target_width and x > 0.0:
                continue
            y = height_at(x, x + w)
            if best_y is None or y < best_y - 1e-9 or (
                    abs(y - best_y) <= 1e-9 and x < best_x):
                best_x, best_y = x, y
        if best_y is None:
            best_x, best_y = 0.0, height_at(0.0, w)

        placed[name] = (best_x + w / 2.0, best_y + h / 2.0)
        columns.append((best_x, best_x + w, best_y + h))

    return placed


def _block_extent(placements, width, height, gap):
    right = max(x + (width[n] + gap) / 2.0 for n, (x, _) in placements.items())
    bottom = max(y + (height[n] + gap) / 2.0 for n, (_, y) in placements.items())
    return right, bottom


def _pack_regions(layering, centres, width, height, gap, names):
    """Lay the tiles out as labelled biological regions, packed two levels deep.

    Grouping pathway tiles into KEGG-style superclasses is how every curated
    global map is organised -- a reader finds lipid metabolism by region, not by
    reading every caption.

    The packing has to be two-level to be worth anything. Giving each region a
    full-width band and stacking the bands leaves a region holding one tile
    wasting the whole width beside it: measured at 23% of the canvas covered,
    77% whitespace. So each region is packed into a block sized to its own
    content, and the blocks are then packed against each other.

    Within a region the meta-layout's flow order is preserved, so the
    arrangement is still metabolically ordered, just locally.
    """
    flow_order = []
    for layer in layering.layers:
        members = [n for n in layer if n in width]
        flow_order.extend(sorted(members, key=lambda n: centres.get(n, (0.0, 0.0))[0]))
    for name in names:
        if name not in flow_order and name in width:
            flow_order.append(name)
    if not flow_order:
        return centres, {}

    regions = taxonomy.group(flow_order)
    rank = {name: index for index, name in enumerate(flow_order)}

    # Level 1: pack each region against its own tiles.
    blocks, block_width, block_height = {}, {}, {}
    for label, members in regions.items():
        members = sorted(members, key=lambda n: rank.get(n, 0))
        area = sum((width[n] + gap) * (height[n] + gap) for n in members)
        target = max(math.sqrt(area * REGION_ASPECT),
                     max(width[n] for n in members) + gap)
        local = _skyline_pack(members, width, height, gap, target)
        right, bottom = _block_extent(local, width, height, gap)
        blocks[label] = local
        block_width[label] = right
        block_height[label] = bottom + REGION_TITLE_OFFSET

    # Level 2: pack the region blocks against each other.
    labels = list(blocks)
    total = sum((block_width[l] + REGION_GAP) * (block_height[l] + REGION_GAP)
                for l in labels)
    target = max(math.sqrt(total * CANVAS_ASPECT),
                 max(block_width[l] for l in labels) + REGION_GAP)
    block_pos = _skyline_pack(labels, block_width, block_height, REGION_GAP, target)

    placed, captions = {}, {}
    for label in labels:
        cx, cy = block_pos[label]
        origin_x = cx - (block_width[label] + REGION_GAP) / 2.0
        origin_y = cy - (block_height[label] + REGION_GAP) / 2.0
        captions[label] = (origin_x, origin_y + REGION_TITLE_OFFSET * 0.35,
                           block_width[label])
        for name, (x, y) in blocks[label].items():
            placed[name] = (origin_x + x, origin_y + REGION_TITLE_OFFSET + y)

    return placed, captions


def compose(tiles, meta_graph, map_name, author="AutoLayout", description="",
            gap=TILE_GAP, show_titles=True, stitch_links=True):
    """Merge per-cluster Escher maps into one.

    `tiles` is [(cluster name, escher map)]; `meta_graph` is the directed graph
    over those names from `build_meta_graph`.
    """
    tiles = [(name, m) for name, m in tiles if m and m[1]["nodes"]]
    if not tiles:
        return None
    if len(tiles) == 1:
        return tiles[0][1]

    boxes = {name: _bbox(m) for name, m in tiles}
    width, height = {}, {}
    for name, (left, top, right, bottom) in boxes.items():
        width[name] = (right - left) + gap
        height[name] = (bottom - top) + gap + TITLE_OFFSET

    placed = nx.DiGraph()
    placed.add_nodes_from(name for name, _ in tiles)
    for u, v, data in meta_graph.edges(data=True):
        if u in width and v in width:
            placed.add_edge(u, v, **data)

    def reversal_cost(u, v):
        # Reversing a strongly shared link costs more: the arrangement should
        # bend a one-metabolite connection rather than a twenty-metabolite one.
        return 1.0 + placed.edges[u, v].get("weight", 1)

    centres, layering = layered_layout(
        placed, width, height,
        x_gap=gap * 0.5, y_gap=gap * 0.6,
        reversal_cost=reversal_cost, fold_after=None,
    )
    centres, captions = _pack_regions(layering, centres, width, height, gap,
                                      [name for name, _ in tiles])

    nodes, reactions, labels = {}, {}, {}
    tile_prefix = {}
    for index, (name, tile) in enumerate(tiles):
        left, top, right, bottom = boxes[name]
        centre_x, centre_y = centres.get(name, (0.0, 0.0))
        dx = centre_x - (left + right) / 2.0
        dy = centre_y - (top + bottom) / 2.0 + TITLE_OFFSET / 2.0
        _offset_tile(tile, dx, dy, f"t{index}", nodes, reactions)
        tile_prefix[name] = f"t{index}"

        if show_titles:
            labels[f"title_{index}"] = {
                "x": left + dx,
                "y": top + dy - TITLE_OFFSET * 0.55,
                "text": name,
                "font_size_base": 12.0,
            }

    # Tiles are placed; now put the flow between them back. This runs after
    # placement because a connector is drawn between final node positions.
    if stitch_links:
        stitch(select_links(placed), tile_prefix, nodes, reactions)

    if show_titles:
        for label, (x, y, _block_width) in captions.items():
            labels[f"region_{label}"] = {
                "x": x,
                "y": y,
                "text": label.upper(),
                "font_size_base": REGION_FONT_BASE,
            }

    xs = [n["x"] for n in nodes.values()] + [l["x"] for l in labels.values()]
    ys = [n["y"] for n in nodes.values()] + [l["y"] for l in labels.values()]
    canvas = {
        "x": min(xs) - CANVAS_PADDING,
        "y": min(ys) - CANVAS_PADDING,
        "width": (max(xs) - min(xs)) + 2 * CANVAS_PADDING,
        "height": (max(ys) - min(ys)) + 2 * CANVAS_PADDING,
    }
    labels["attribution"] = {
        "x": canvas["x"] + 80.0,
        "y": canvas["y"] + canvas["height"] - 80.0,
        "text": f"Created by {author}",
    }

    return [
        {
            "map_name": map_name,
            "map_id": map_name.replace(" ", "_"),
            "map_description": description or f"Generated by {author}",
            "homepage": "https://escher.github.io",
            "schema": "https://escher.github.io/escher/jsonschema/1-0-0#",
        },
        {
            "reactions": reactions,
            "nodes": nodes,
            "text_labels": labels,
            "canvas": canvas,
        },
    ]
