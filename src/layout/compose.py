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
from .sugiyama import layered_layout

TILE_GAP = 900.0           # clear space between tiles
TITLE_OFFSET = 190.0       # cluster caption above its tile
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


def _pack_rows(layering, centres, width, height, gap):
    """Close up the gaps the meta-layout leaves between tiles.

    Brandes-Koepf aligns nodes into columns, which is exactly what makes a
    pathway backbone straight and exactly what wastes space at tile scale: a
    column sized for the tallest, widest tile leaves a hole wherever a small
    tile sits in it, and a nine-tile model ends up mostly whitespace.

    The layer assignment and the within-layer order are worth keeping -- they
    are the metabolic flow and the crossing-minimised arrangement. So keep
    both, and re-pack each layer as a row butted up against its neighbours.
    """
    order = []
    for layer in layering.layers:
        members = [n for n in layer if n in width]
        order.extend(sorted(members, key=lambda n: centres.get(n, (0.0, 0.0))[0]))
    if not order:
        return centres

    # One row per meta-layer would be faithful to the flow but badly
    # proportioned -- a deep pathway DAG gives many rows of one or two tiles.
    # Shelf-pack the flow order instead: related pathways stay adjacent in
    # reading order, and the block comes out roughly poster-shaped, which is
    # how KEGG's global map (templates/t4) is arranged.
    area = sum((width[n] + gap) * (height[n] + gap) for n in order)
    target_width = max(math.sqrt(area * 1.3), max(width[n] for n in order))

    rows, current, current_width = [], [], 0.0
    for node in order:
        if current and current_width + width[node] + gap > target_width:
            rows.append(current)
            current, current_width = [], 0.0
        current.append(node)
        current_width += width[node] + gap
    if current:
        rows.append(current)

    row_widths = [sum(width[n] for n in row) + gap * (len(row) - 1) for row in rows]
    widest = max(row_widths)

    packed, cursor_y = {}, 0.0
    for row, row_width in zip(rows, row_widths):
        row_height = max(height[n] for n in row)
        cursor_x = (widest - row_width) / 2.0        # centre each row
        for node in row:
            # Top-align within the row rather than centring. A row is as tall
            # as its tallest tile, and centring a short tile in that band puts
            # empty space both above and below it; top-aligning collects the
            # slack in one place and lines the captions up across the row.
            packed[node] = (cursor_x + width[node] / 2.0, cursor_y + height[node] / 2.0)
            cursor_x += width[node] + gap
        cursor_y += row_height + gap
    return packed


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
    centres = _pack_rows(layering, centres, width, height, gap)

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
