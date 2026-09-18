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
            weight = len(produced[a] & consumed[b])
            if weight:
                meta.add_edge(a, b, weight=weight, rxns=[])
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
            gap=TILE_GAP, show_titles=True):
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
    for index, (name, tile) in enumerate(tiles):
        left, top, right, bottom = boxes[name]
        centre_x, centre_y = centres.get(name, (0.0, 0.0))
        dx = centre_x - (left + right) / 2.0
        dy = centre_y - (top + bottom) / 2.0 + TITLE_OFFSET / 2.0
        _offset_tile(tile, dx, dy, f"t{index}", nodes, reactions)

        if show_titles:
            labels[f"title_{index}"] = {
                "x": left + dx,
                "y": top + dy - TITLE_OFFSET * 0.55,
                "text": name,
                "font_size_base": 12.0,
            }

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
