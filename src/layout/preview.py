"""Render an Escher map dict to PNG for visual inspection.

Reads the emitted map rather than the internal layout state, so a preview that
looks right is also evidence the exported JSON is structurally right.
"""

from .raster import Canvas, text_width

MAX_PIXELS = 6000.0
POSTER_PIXELS = 12000.0    # a whole-model map needs more canvas to be readable at all
MAX_SCALE = 2.0            # glyphs at 4x; past this the file grows for nothing
DEFAULT_DPI = 300

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


def render(escher_map, path, show_labels=True, max_pixels=None, dpi=DEFAULT_DPI):
    body = escher_map[1]
    nodes, reactions, canvas_box = body["nodes"], body["reactions"], body["canvas"]

    span_x = max(canvas_box["width"], 1.0)
    span_y = max(canvas_box["height"], 1.0)
    if max_pixels is None:
        max_pixels = POSTER_PIXELS if max(span_x, span_y) > 12000.0 else MAX_PIXELS

    # Scale so the smallest label still lands on real glyphs rather than the
    # "too small to read" bar, then let the pixel budget veto it.
    #
    # The old 0.5 ceiling was the binding constraint on every large map: at 0.5
    # a metabolite label is drawn at exactly 1x, i.e. a 6x7 pixel glyph, which
    # survives a screenshot and disintegrates the moment anyone zooms. Raising
    # the ceiling is what makes a zoom sharp; `max_pixels` still bounds the
    # memory, because this rasteriser holds the whole image as Python bytearrays.
    scale = min(max_pixels / span_x, max_pixels / span_y, MAX_SCALE)
    width = max(240, int(span_x * scale))
    height = max(240, int(span_y * scale))
    canvas = Canvas(width, height)

    def to_px(x, y):
        return ((x - canvas_box["x"]) * scale, (y - canvas_box["y"]) * scale)

    thick = max(1, int(round(3.0 * scale)))
    thin = max(1, int(round(1.6 * scale)))

    def node_radius(node):
        if node["node_type"] == "metabolite":
            return 30.0 if node.get("node_is_primary", True) else 16.0
        return 11.0 if node["node_type"] == "midmarker" else 6.0

    # Arrowheads. The segment chain is emitted substrate -> product, so a
    # segment's own from/to order is the flux direction. An arrowhead short of
    # each target metabolite is the only direction cue a rendered figure
    # carries: Escher draws its own from the signed stoichiometry, but these
    # PNGs are what anyone actually looks at, and without heads a reader cannot
    # tell which way the pathway runs.
    arrows = []

    for reaction in reactions.values():
        for segment in reaction["segments"].values():
            a = nodes.get(segment["from_node_id"])
            b = nodes.get(segment["to_node_id"])
            if a is None or b is None:
                continue
            p0, p3 = (a["x"], a["y"]), (b["x"], b["y"])

            # Weight follows what the segment *is*, not whether it happens to
            # be curved. Ring arcs are curved by construction, and colouring by
            # curvature drew the TCA backbone in the thin grey reserved for
            # cofactor stubs. A stub is the segment that touches a secondary
            # metabolite; everything else is backbone.
            def is_secondary(node):
                return (node["node_type"] == "metabolite"
                        and not node.get("node_is_primary", True))

            stub = is_secondary(a) or is_secondary(b)
            colour = STUB if stub else BACKBONE
            weight = thin if stub else thick

            if segment.get("b1") and segment.get("b2"):
                curve = _bezier(p0, (segment["b1"]["x"], segment["b1"]["y"]),
                                (segment["b2"]["x"], segment["b2"]["y"]), p3)
                canvas.polyline([to_px(*p) for p in curve], colour, weight)
                if b["node_type"] == "metabolite":
                    arrows.append((curve[-2], p3, node_radius(b), colour))
            else:
                canvas.line(*to_px(*p0), *to_px(*p3), colour, weight)
                if b["node_type"] == "metabolite":
                    arrows.append((p0, p3, node_radius(b), colour))

    for node in nodes.values():
        x, y = to_px(node["x"], node["y"])
        kind = node["node_type"]
        if kind == "metabolite":
            if node.get("node_is_primary", True):
                canvas.disc(x, y, max(1.0, 30.0 * scale), PRIMARY_FILL, PRIMARY_EDGE)
            else:
                canvas.disc(x, y, max(1.0, 16.0 * scale), SECONDARY_FILL, SECONDARY_EDGE)
        elif kind == "midmarker":
            canvas.disc(x, y, max(1.0, 11.0 * scale), MIDMARKER)
        else:
            canvas.disc(x, y, max(1.0, 6.0 * scale), MULTIMARKER)

    for tail, head, radius, colour in arrows:
        dx, dy = head[0] - tail[0], head[1] - tail[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length
        tip = (head[0] - ux * radius, head[1] - uy * radius)
        canvas.arrowhead(to_px(*tip), (ux, uy),
                         max(4.0, 46.0 * scale), max(3.0, 32.0 * scale), colour)

    if show_labels:
        # Render each label at the size the layout reserved for it, so a
        # preview that looks crowded really is crowded.
        from .raster import GLYPH_WIDTH
        from .render import (CHAR_WIDTH_RATIO, ESCHER_DEFAULT_FONT_BASE,
                             METABOLITE_FONT_FACTOR, REACTION_FONT_FACTOR)

        def draw_label(holder, factor, text, lx, ly, colour):
            """Glyphs when they would be the right size, a bar when they would
            not. Clamping the glyph scale to 1x is what made poster previews
            unreadable *and* unfaithful at the same time."""
            base = holder.get("font_size_base", ESCHER_DEFAULT_FONT_BASE)
            size_px = base * factor * scale
            per_char = size_px * CHAR_WIDTH_RATIO
            steps = per_char / GLYPH_WIDTH
            if steps >= 0.75:
                canvas.text(lx, ly, text, colour, max(1, int(round(steps))))
            else:
                canvas.bar(lx, ly, max(1.0, per_char * len(str(text))),
                           max(1.0, size_px), colour)

        for node in nodes.values():
            if node["node_type"] != "metabolite" or node.get("label_hidden"):
                continue
            lx, ly = to_px(node["label_x"], node["label_y"])
            colour = METABOLITE_TEXT if node.get("node_is_primary", True) else SECONDARY_EDGE
            draw_label(node, METABOLITE_FONT_FACTOR,
                       node.get("label_text", node["bigg_id"]), lx, ly, colour)
        for reaction in reactions.values():
            lx, ly = to_px(reaction["label_x"], reaction["label_y"])
            draw_label(reaction, REACTION_FONT_FACTOR, reaction["bigg_id"],
                       lx, ly, REACTION_TEXT)

        # Free-standing captions: cluster titles on a composed whole-model map,
        # and the attribution line. Without these a meta-tiled map looks like
        # unlabelled islands.
        for label in body.get("text_labels", {}).values():
            lx, ly = to_px(label["x"], label["y"])
            draw_label(label, 3.0, label.get("text", ""), lx, ly, TITLE_TEXT)

    return canvas.save(path, dpi=dpi)


def label_pixel_width(text, scale=1):
    return text_width(text, scale)
