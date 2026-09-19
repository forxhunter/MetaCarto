"""Write an Escher map dict to a vector PDF, for use as a journal figure.

SVG is the right interchange format but LaTeX cannot read it, and none of
inkscape, rsvg-convert or cairosvg is installed on this machine -- the same
reason `raster.py` exists instead of matplotlib. Rather than add a converter
and a system dependency, this emits PDF directly from the same primitives
`svgout.render` draws: lines, cubic Beziers, circles, filled triangles and
text.

Text needs no font embedding because Helvetica is one of the fourteen PDF
base fonts every reader is required to have, so a figure stays real selectable
text at any magnification and the file stays small.

The page is sized in millimetres, so a figure can be placed at its final print
width and the label sizes `scripts/iterate.py` reports mean what they say:

    pdfout.render(doc, "fig1.pdf", width_mm=180.0)   # full text width
"""

import zlib

from .render import (CHAR_WIDTH_RATIO, ESCHER_DEFAULT_FONT_BASE,
                     LINE_HEIGHT_RATIO, METABOLITE_FONT_FACTOR,
                     REACTION_FONT_FACTOR)
from .svgout import (ARROW_HALF, ARROW_LEN, BACKBONE, BACKBONE_W, MID_R,
                     MIDMARKER, MULTI_R, MULTIMARKER, METABOLITE_TEXT,
                     PRIMARY_EDGE, PRIMARY_FILL, PRIMARY_R, REACTION_TEXT,
                     SECONDARY_EDGE, SECONDARY_FILL, SECONDARY_R, STUB,
                     STUB_W, TITLE_TEXT, _is_secondary)

PT_PER_MM = 72.0 / 25.4

# Bezier circle constant: 4/3 * (sqrt(2) - 1).
KAPPA = 0.5522847498307936

# A Helvetica baseline sits about this fraction of the font size below the
# vertical centre of a glyph box, which is what SVG's
# dominant-baseline="middle" positions against. Matching it keeps the PDF and
# the SVG exports aligned on the same label geometry.
BASELINE_DROP = 0.35


def content_box(body, margin=60.0):
    """The box the drawing actually occupies, labels included.

    The declared `canvas` is generous -- on the Recon3D mevalonate map about a
    quarter of its width is margin -- and a figure placed at a fixed column
    width pays for that margin in font size: the same map is 5.4 pt of
    metabolite label against the canvas and 7.2 pt against its content. So the
    page is cut to the ink, which is also what `scripts/iterate.py` measures
    against.
    """
    xs, ys = [], []
    for node in body["nodes"].values():
        xs.append(node["x"])
        ys.append(node["y"])
        if node["node_type"] != "metabolite" or node.get("label_hidden"):
            continue
        size = (node.get("font_size_base", ESCHER_DEFAULT_FONT_BASE)
                * METABOLITE_FONT_FACTOR)
        text = node.get("label_text", node["bigg_id"])
        xs.append(node["label_x"])
        xs.append(node["label_x"] + len(text) * size * CHAR_WIDTH_RATIO)
        ys.append(node["label_y"] - size * LINE_HEIGHT_RATIO / 2.0)
        ys.append(node["label_y"] + size * LINE_HEIGHT_RATIO / 2.0)
    for reaction in body["reactions"].values():
        size = (reaction.get("font_size_base", ESCHER_DEFAULT_FONT_BASE)
                * REACTION_FONT_FACTOR)
        xs.append(reaction["label_x"])
        xs.append(reaction["label_x"]
                  + len(reaction["bigg_id"]) * size * CHAR_WIDTH_RATIO)
        ys.append(reaction["label_y"] - size * LINE_HEIGHT_RATIO / 2.0)
        ys.append(reaction["label_y"] + size * LINE_HEIGHT_RATIO / 2.0)
    for free in body.get("text_labels", {}).values():
        size = free.get("font_size_base", ESCHER_DEFAULT_FONT_BASE) * 3.0
        xs.append(free["x"])
        xs.append(free["x"] + len(free.get("text", "")) * size * CHAR_WIDTH_RATIO)
        ys.append(free["y"] - size * LINE_HEIGHT_RATIO / 2.0)
        ys.append(free["y"] + size * LINE_HEIGHT_RATIO / 2.0)
    if not xs:
        return dict(body["canvas"])
    left, right = min(xs) - margin, max(xs) + margin
    top, bottom = min(ys) - margin, max(ys) + margin
    return {"x": left, "y": top,
            "width": right - left, "height": bottom - top}


def _rgb(colour):
    """Turn '#rrggbb' into the three 0..1 components PDF wants."""
    value = colour.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _pdf_text(text):
    out = str(text).replace("\\", "\\\\")
    return out.replace("(", "\\(").replace(")", "\\)")


class _Canvas(object):
    """Accumulates a PDF content stream in page points, y-up."""

    def __init__(self, box, scale):
        self.parts = []
        self.scale = scale
        self.x0 = box["x"]
        # The y flip happens here rather than through a `cm` matrix, because a
        # mirrored CTM mirrors the glyphs too and would need undoing at every
        # label.
        self.y1 = box["y"] + box["height"]

    def xy(self, x, y):
        return (x - self.x0) * self.scale, (self.y1 - y) * self.scale

    def stroke_colour(self, colour):
        self.parts.append("%.3f %.3f %.3f RG" % _rgb(colour))

    def fill_colour(self, colour):
        self.parts.append("%.3f %.3f %.3f rg" % _rgb(colour))

    def line(self, p0, p1, colour, weight):
        a, b = self.xy(*p0), self.xy(*p1)
        self.stroke_colour(colour)
        self.parts.append("%.2f w" % (weight * self.scale))
        self.parts.append("%.2f %.2f m %.2f %.2f l S" % (a[0], a[1], b[0], b[1]))

    def bezier(self, p0, c1, c2, p3, colour, weight):
        a, b, c, d = (self.xy(*p) for p in (p0, c1, c2, p3))
        self.stroke_colour(colour)
        self.parts.append("%.2f w" % (weight * self.scale))
        self.parts.append("%.2f %.2f m %.2f %.2f %.2f %.2f %.2f %.2f c S"
                          % (a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]))

    def circle(self, centre, radius, fill, edge=None, weight=2.0):
        cx, cy = self.xy(*centre)
        r = radius * self.scale
        k = r * KAPPA
        self.parts.append("%.2f %.2f m" % (cx + r, cy))
        for arc in ((cx + r, cy + k, cx + k, cy + r, cx, cy + r),
                    (cx - k, cy + r, cx - r, cy + k, cx - r, cy),
                    (cx - r, cy - k, cx - k, cy - r, cx, cy - r),
                    (cx + k, cy - r, cx + r, cy - k, cx + r, cy)):
            self.parts.append("%.2f %.2f %.2f %.2f %.2f %.2f c" % arc)
        self.fill_colour(fill)
        if edge is None:
            self.parts.append("f")
        else:
            self.stroke_colour(edge)
            self.parts.append("%.2f w B" % (weight * self.scale))

    def triangle(self, p0, p1, p2, colour):
        a, b, c = (self.xy(*p) for p in (p0, p1, p2))
        self.fill_colour(colour)
        self.parts.append("%.2f %.2f m %.2f %.2f l %.2f %.2f l h f"
                          % (a[0], a[1], b[0], b[1], c[0], c[1]))

    def text(self, anchor, size, body, colour):
        x, y = self.xy(*anchor)
        pt = size * self.scale
        self.fill_colour(colour)
        self.parts.append(
            "BT /F1 %.2f Tf %.2f %.2f Td (%s) Tj ET"
            % (pt, x, y - pt * BASELINE_DROP, _pdf_text(body)))


def _arrow(canvas, tail, head, radius, colour):
    dx, dy = head[0] - tail[0], head[1] - tail[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    tip = (head[0] - ux * radius, head[1] - uy * radius)
    base = (tip[0] - ux * ARROW_LEN, tip[1] - uy * ARROW_LEN)
    px, py = -uy * ARROW_HALF, ux * ARROW_HALF
    canvas.triangle(tip, (base[0] + px, base[1] + py),
                    (base[0] - px, base[1] - py), colour)


def _draw(escher_map, canvas, show_labels):
    body = escher_map[1]
    nodes, reactions = body["nodes"], body["reactions"]

    canvas.parts.append("1 J")          # round caps, as the SVG has
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
                canvas.bezier(p0, (b1["x"], b1["y"]), (b2["x"], b2["y"]),
                              p3, colour, weight)
                tail = (b2["x"], b2["y"])
            else:
                canvas.line(p0, p3, colour, weight)
                tail = p0
            if b["node_type"] == "metabolite":
                radius = (PRIMARY_R if b.get("node_is_primary", True)
                          else SECONDARY_R)
                arrows.append((tail, p3, radius, colour))

    for tail, head, radius, colour in arrows:
        _arrow(canvas, tail, head, radius, colour)

    for node in nodes.values():
        kind = node["node_type"]
        centre = (node["x"], node["y"])
        if kind == "metabolite":
            if node.get("node_is_primary", True):
                canvas.circle(centre, PRIMARY_R, PRIMARY_FILL, PRIMARY_EDGE)
            else:
                canvas.circle(centre, SECONDARY_R, SECONDARY_FILL,
                              SECONDARY_EDGE)
        elif kind == "midmarker":
            canvas.circle(centre, MID_R, MIDMARKER)
        else:
            canvas.circle(centre, MULTI_R, MULTIMARKER)

    if not show_labels:
        return
    for node in nodes.values():
        if node["node_type"] != "metabolite" or node.get("label_hidden"):
            continue
        size = (node.get("font_size_base", ESCHER_DEFAULT_FONT_BASE)
                * METABOLITE_FONT_FACTOR)
        colour = (METABOLITE_TEXT if node.get("node_is_primary", True)
                  else SECONDARY_EDGE)
        canvas.text((node["label_x"], node["label_y"]), size,
                    node.get("label_text", node["bigg_id"]), colour)
    for reaction in reactions.values():
        size = (reaction.get("font_size_base", ESCHER_DEFAULT_FONT_BASE)
                * REACTION_FONT_FACTOR)
        canvas.text((reaction["label_x"], reaction["label_y"]), size,
                    reaction["bigg_id"], REACTION_TEXT)
    for free in body.get("text_labels", {}).values():
        size = free.get("font_size_base", ESCHER_DEFAULT_FONT_BASE) * 3.0
        canvas.text((free["x"], free["y"]), size,
                    free.get("text", ""), TITLE_TEXT)


def render(escher_map, path, show_labels=True, width_mm=180.0,
           max_height_mm=None, crop=True, compress=True):
    """Write `escher_map` to `path` as a vector PDF sized for the page.

    `width_mm` is the placed width. If `max_height_mm` is given and the map is
    too tall for it, the figure is fitted to the height instead -- what a
    production editor does, and what `scripts/iterate.py` assumes when it
    reports a label size.
    """
    body = escher_map[1]
    box = content_box(body) if crop else body["canvas"]
    if box["width"] <= 0 or box["height"] <= 0:
        raise ValueError("map has an empty canvas")

    scale = (width_mm * PT_PER_MM) / box["width"]
    if max_height_mm is not None:
        scale = min(scale, (max_height_mm * PT_PER_MM) / box["height"])
    page_w = box["width"] * scale
    page_h = box["height"] * scale

    canvas = _Canvas(box, scale)
    canvas.fill_colour("#ffffff")
    canvas.parts.append("0 0 %.2f %.2f re f" % (page_w, page_h))
    _draw(escher_map, canvas, show_labels)

    stream = ("\n".join(canvas.parts)).encode("latin-1", "replace")
    if compress:
        stream = zlib.compress(stream)
        extra = "/Filter /FlateDecode "
    else:
        extra = ""

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        ("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] "
         "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
         % (page_w, page_h)).encode("ascii"),
        ("<< /Length %d %s>>\nstream\n" % (len(stream), extra)).encode("ascii")
        + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>",
    ]

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, payload in enumerate(objects, start=1):
        offsets.append(len(out))
        out += ("%d 0 obj\n" % number).encode("ascii") + payload + b"\nendobj\n"

    xref = len(out)
    out += ("xref\n0 %d\n" % (len(objects) + 1)).encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += ("%010d 00000 n \n" % offset).encode("ascii")
    out += ("trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, xref)).encode("ascii")

    with open(path, "wb") as handle:
        handle.write(out)
    return path
