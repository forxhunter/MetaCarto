"""Write an Escher map dict to SVG.

The PNG preview answers "what does this look like"; it cannot answer "what does
this look like zoomed in", because a raster has one resolution and this
rasteriser holds the whole image in Python bytearrays -- the 17,000-unit lipid
map already costs ~400 MB at 12,000 px, and doubling the zoom quadruples that.

SVG has no such ceiling. The same geometry becomes a few hundred KB of vector
primitives that stay sharp at any magnification and print at any DPI, which is
what a large pathway map actually needs: nobody reads a 10,000-node figure at
fit-to-page, they zoom into one corner.

Text is real text, so it is also selectable and searchable -- you can find a
metabolite with ctrl-F instead of hunting for it.
"""

from .render import (CHAR_WIDTH_RATIO, ESCHER_DEFAULT_FONT_BASE,
                     METABOLITE_FONT_FACTOR, REACTION_FONT_FACTOR)

BACKBONE = "#2f3b47"
STUB = "#9aa6b2"
PRIMARY_FILL = "#e8833a"
PRIMARY_EDGE = "#2f3b47"
SECONDARY_FILL = "#c8d2dc"
SECONDARY_EDGE = "#6b7885"
MIDMARKER = "#3a7bd5"
MULTIMARKER = "#7f8c99"
METABOLITE_TEXT = "#1c252e"
REACTION_TEXT = "#2b5fa8"
TITLE_TEXT = "#a83a6e"

PRIMARY_R = 30.0
SECONDARY_R = 16.0
MID_R = 11.0
MULTI_R = 6.0
BACKBONE_W = 6.0
STUB_W = 3.0
ARROW_LEN = 46.0
ARROW_HALF = 16.0


def _esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _is_secondary(node):
    return (node["node_type"] == "metabolite"
            and not node.get("node_is_primary", True))


def _arrow(parts, tail, head, radius, colour):
    dx, dy = head[0] - tail[0], head[1] - tail[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    tip_x, tip_y = head[0] - ux * radius, head[1] - uy * radius
    bx, by = tip_x - ux * ARROW_LEN, tip_y - uy * ARROW_LEN
    px, py = -uy * ARROW_HALF, ux * ARROW_HALF
    parts.append(
        '<path d="M%.1f %.1f L%.1f %.1f L%.1f %.1f Z" fill="%s"/>'
        % (tip_x, tip_y, bx + px, by + py, bx - px, by - py, colour))


def render(escher_map, path, show_labels=True):
    body = escher_map[1]
    nodes, reactions = body["nodes"], body["reactions"]
    box = body["canvas"]

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'viewBox="%.1f %.1f %.1f %.1f" width="%.0f" height="%.0f">'
        % (box["x"], box["y"], box["width"], box["height"],
           box["width"], box["height"]),
        '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#ffffff"/>'
        % (box["x"], box["y"], box["width"], box["height"]),
        '<g stroke-linecap="round" fill="none">',
    ]

    arrows = []
    for reaction in reactions.values():
        for segment in reaction["segments"].values():
            a = nodes.get(segment["from_node_id"])
            b = nodes.get(segment["to_node_id"])
            if a is None or b is None:
                continue
            stub = _is_secondary(a) or _is_secondary(b)
            colour = STUB if stub else BACKBONE
            weight = STUB_W if stub else BACKBONE_W
            p0, p3 = (a["x"], a["y"]), (b["x"], b["y"])

            if segment.get("b1") and segment.get("b2"):
                b1, b2 = segment["b1"], segment["b2"]
                parts.append(
                    '<path d="M%.1f %.1f C%.1f %.1f %.1f %.1f %.1f %.1f" '
                    'stroke="%s" stroke-width="%.1f"/>'
                    % (p0[0], p0[1], b1["x"], b1["y"], b2["x"], b2["y"],
                       p3[0], p3[1], colour, weight))
                tail = (b2["x"], b2["y"])
            else:
                parts.append(
                    '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                    'stroke="%s" stroke-width="%.1f"/>'
                    % (p0[0], p0[1], p3[0], p3[1], colour, weight))
                tail = p0

            if b["node_type"] == "metabolite":
                radius = PRIMARY_R if b.get("node_is_primary", True) else SECONDARY_R
                arrows.append((tail, p3, radius, colour))

    parts.append('</g><g>')
    for tail, head, radius, colour in arrows:
        _arrow(parts, tail, head, radius, colour)

    for node in nodes.values():
        kind = node["node_type"]
        if kind == "metabolite":
            if node.get("node_is_primary", True):
                r, fill, edge = PRIMARY_R, PRIMARY_FILL, PRIMARY_EDGE
            else:
                r, fill, edge = SECONDARY_R, SECONDARY_FILL, SECONDARY_EDGE
            parts.append(
                '<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" stroke="%s" '
                'stroke-width="2"/>' % (node["x"], node["y"], r, fill, edge))
        elif kind == "midmarker":
            parts.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>'
                         % (node["x"], node["y"], MID_R, MIDMARKER))
        else:
            parts.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>'
                         % (node["x"], node["y"], MULTI_R, MULTIMARKER))
    parts.append('</g>')

    if show_labels:
        parts.append('<g font-family="Helvetica,Arial,sans-serif">')

        def label(holder, factor, text, colour):
            size = holder.get("font_size_base", ESCHER_DEFAULT_FONT_BASE) * factor
            parts.append(
                '<text x="%.1f" y="%.1f" font-size="%.1f" fill="%s" '
                'dominant-baseline="middle">%s</text>'
                % (holder["label_x"], holder["label_y"], size, colour, _esc(text)))

        for node in nodes.values():
            if node["node_type"] != "metabolite" or node.get("label_hidden"):
                continue
            colour = (METABOLITE_TEXT if node.get("node_is_primary", True)
                      else SECONDARY_EDGE)
            label(node, METABOLITE_FONT_FACTOR,
                  node.get("label_text", node["bigg_id"]), colour)
        for reaction in reactions.values():
            label(reaction, REACTION_FONT_FACTOR, reaction["bigg_id"], REACTION_TEXT)
        for free in body.get("text_labels", {}).values():
            size = free.get("font_size_base", ESCHER_DEFAULT_FONT_BASE) * 3.0
            parts.append(
                '<text x="%.1f" y="%.1f" font-size="%.1f" fill="%s" '
                'dominant-baseline="middle">%s</text>'
                % (free["x"], free["y"], size, TITLE_TEXT,
                   _esc(free.get("text", ""))))
        parts.append('</g>')

    parts.append('</svg>')
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))
    return path
