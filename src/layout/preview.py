"""Render an Escher map dict to PNG for visual inspection.

Reads the emitted map rather than the internal layout state, so a preview that
looks right is also evidence the exported JSON is structurally right.
"""

from .raster import Canvas, text_width

MAX_PIXELS = 2200.0

BACKBONE = (47, 59, 71)
STUB = (154, 166, 178)
PRIMARY_FILL = (232, 131, 58)
PRIMARY_EDGE = (47, 59, 71)
SECONDARY_FILL = (200, 210, 220)
SECONDARY_EDGE = (107, 120, 133)
MIDMARKER = (58, 123, 213)
MULTIMARKER = (127, 140, 153)
METABOLITE_TEXT = (28, 37, 46)
REACTION_TEXT = (43, 95, 168)
TITLE_TEXT = (168, 58, 110)


def _bezier(p0, b1, b2, p3, steps=14):
    points = []
    for i in range(steps + 1):
        t = i / steps
        mt = 1.0 - t
        points.append((
            mt ** 3 * p0[0] + 3 * mt * mt * t * b1[0] + 3 * mt * t * t * b2[0] + t ** 3 * p3[0],
            mt ** 3 * p0[1] + 3 * mt * mt * t * b1[1] + 3 * mt * t * t * b2[1] + t ** 3 * p3[1],
        ))
    return points


def render(escher_map, path, show_labels=True, max_pixels=MAX_PIXELS):
    body = escher_map[1]
    nodes, reactions, canvas_box = body["nodes"], body["reactions"], body["canvas"]

    span_x = max(canvas_box["width"], 1.0)
    span_y = max(canvas_box["height"], 1.0)
    scale = min(max_pixels / span_x, max_pixels / span_y, 0.5)
    width = max(240, int(span_x * scale))
    height = max(240, int(span_y * scale))
    canvas = Canvas(width, height)

    def to_px(x, y):
        return ((x - canvas_box["x"]) * scale, (y - canvas_box["y"]) * scale)

    thick = max(1, int(round(3.0 * scale)))
    thin = max(1, int(round(1.6 * scale)))

    for reaction in reactions.values():
        for segment in reaction["segments"].values():
            a = nodes.get(segment["from_node_id"])
            b = nodes.get(segment["to_node_id"])
            if a is None or b is None:
                continue
            p0, p3 = (a["x"], a["y"]), (b["x"], b["y"])
            if segment.get("b1") and segment.get("b2"):
                curve = _bezier(p0, (segment["b1"]["x"], segment["b1"]["y"]),
                                (segment["b2"]["x"], segment["b2"]["y"]), p3)
                canvas.polyline([to_px(*p) for p in curve], STUB, thin)
            else:
                canvas.line(*to_px(*p0), *to_px(*p3), BACKBONE, thick)

    for node in nodes.values():
        x, y = to_px(node["x"], node["y"])
        kind = node["node_type"]
        if kind == "metabolite":
            if node.get("node_is_primary", True):
                canvas.disc(x, y, max(3.0, 30.0 * scale), PRIMARY_FILL, PRIMARY_EDGE)
            else:
                canvas.disc(x, y, max(2.0, 16.0 * scale), SECONDARY_FILL, SECONDARY_EDGE)
        elif kind == "midmarker":
            canvas.disc(x, y, max(2.0, 11.0 * scale), MIDMARKER)
        else:
            canvas.disc(x, y, max(1.0, 6.0 * scale), MULTIMARKER)

    if show_labels:
        # Match the per-character width the layout reserved, so a preview that
        # looks like it has overlapping labels really does have them.
        from .raster import GLYPH_WIDTH
        from .render import _CHAR_WIDTH

        font_scale = max(1, int(round(_CHAR_WIDTH * scale / GLYPH_WIDTH)))
        for node in nodes.values():
            if node["node_type"] != "metabolite":
                continue
            lx, ly = to_px(node["label_x"], node["label_y"])
            colour = METABOLITE_TEXT if node.get("node_is_primary", True) else SECONDARY_EDGE
            canvas.text(lx, ly, node["bigg_id"], colour, font_scale)
        for reaction in reactions.values():
            lx, ly = to_px(reaction["label_x"], reaction["label_y"])
            canvas.text(lx, ly, reaction["bigg_id"], REACTION_TEXT, font_scale)

        # Free-standing captions: cluster titles on a composed whole-model map,
        # and the attribution line. Without these a meta-tiled map looks like
        # unlabelled islands.
        for label in body.get("text_labels", {}).values():
            lx, ly = to_px(label["x"], label["y"])
            canvas.text(lx, ly, label.get("text", ""), TITLE_TEXT, font_scale + 1)

    return canvas.save(path)


def label_pixel_width(text, scale=1):
    return text_width(text, scale)
