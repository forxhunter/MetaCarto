"""Minimal dependency-free PNG rasteriser (stdlib zlib only).

Previews exist so a layout can be judged by eye, which means they have to work
everywhere. matplotlib's Agg backend aborts the interpreter on this machine
(delay-loaded DLL failure inside the transform and PIL paths), and a dev tool
that depends on a fragile native stack is not a dev tool. Lines, discs and a
5x7 bitmap font are all a layout preview needs.
"""

import struct
import zlib

# Classic 5x7 console font, column-major, bit 0 = top row.
_FONT = {
    " ": (0x00, 0x00, 0x00, 0x00, 0x00),
    "(": (0x00, 0x1C, 0x22, 0x41, 0x00), ")": (0x00, 0x41, 0x22, 0x1C, 0x00),
    "+": (0x08, 0x08, 0x3E, 0x08, 0x08), ",": (0x00, 0x50, 0x30, 0x00, 0x00),
    "-": (0x08, 0x08, 0x08, 0x08, 0x08), ".": (0x00, 0x60, 0x60, 0x00, 0x00),
    "/": (0x20, 0x10, 0x08, 0x04, 0x02), ":": (0x00, 0x36, 0x36, 0x00, 0x00),
    "[": (0x00, 0x7F, 0x41, 0x41, 0x00), "]": (0x00, 0x41, 0x41, 0x7F, 0x00),
    "_": (0x40, 0x40, 0x40, 0x40, 0x40),
    "0": (0x3E, 0x51, 0x49, 0x45, 0x3E), "1": (0x00, 0x42, 0x7F, 0x40, 0x00),
    "2": (0x42, 0x61, 0x51, 0x49, 0x46), "3": (0x21, 0x41, 0x45, 0x4B, 0x31),
    "4": (0x18, 0x14, 0x12, 0x7F, 0x10), "5": (0x27, 0x45, 0x45, 0x45, 0x39),
    "6": (0x3C, 0x4A, 0x49, 0x49, 0x30), "7": (0x01, 0x71, 0x09, 0x05, 0x03),
    "8": (0x36, 0x49, 0x49, 0x49, 0x36), "9": (0x06, 0x49, 0x49, 0x29, 0x1E),
    "A": (0x7E, 0x11, 0x11, 0x11, 0x7E), "B": (0x7F, 0x49, 0x49, 0x49, 0x36),
    "C": (0x3E, 0x41, 0x41, 0x41, 0x22), "D": (0x7F, 0x41, 0x41, 0x22, 0x1C),
    "E": (0x7F, 0x49, 0x49, 0x49, 0x41), "F": (0x7F, 0x09, 0x09, 0x09, 0x01),
    "G": (0x3E, 0x41, 0x49, 0x49, 0x7A), "H": (0x7F, 0x08, 0x08, 0x08, 0x7F),
    "I": (0x00, 0x41, 0x7F, 0x41, 0x00), "J": (0x20, 0x40, 0x41, 0x3F, 0x01),
    "K": (0x7F, 0x08, 0x14, 0x22, 0x41), "L": (0x7F, 0x40, 0x40, 0x40, 0x40),
    "M": (0x7F, 0x02, 0x0C, 0x02, 0x7F), "N": (0x7F, 0x04, 0x08, 0x10, 0x7F),
    "O": (0x3E, 0x41, 0x41, 0x41, 0x3E), "P": (0x7F, 0x09, 0x09, 0x09, 0x06),
    "Q": (0x3E, 0x41, 0x51, 0x21, 0x5E), "R": (0x7F, 0x09, 0x19, 0x29, 0x46),
    "S": (0x46, 0x49, 0x49, 0x49, 0x31), "T": (0x01, 0x01, 0x7F, 0x01, 0x01),
    "U": (0x3F, 0x40, 0x40, 0x40, 0x3F), "V": (0x1F, 0x20, 0x40, 0x20, 0x1F),
    "W": (0x7F, 0x20, 0x18, 0x20, 0x7F), "X": (0x63, 0x14, 0x08, 0x14, 0x63),
    "Y": (0x03, 0x04, 0x78, 0x04, 0x03), "Z": (0x61, 0x51, 0x49, 0x45, 0x43),
}
_UNKNOWN = (0x7F, 0x41, 0x41, 0x41, 0x7F)

GLYPH_WIDTH = 6      # 5 columns plus one of spacing
GLYPH_HEIGHT = 7


class Canvas:
    def __init__(self, width, height, background=(255, 255, 255)):
        self.width = int(width)
        self.height = int(height)
        row = bytes(background) * self.width
        self.rows = [bytearray(row) for _ in range(self.height)]

    def _blend(self, x, y, colour):
        if 0 <= x < self.width and 0 <= y < self.height:
            i = x * 3
            self.rows[y][i:i + 3] = bytes(colour)

    def disc(self, cx, cy, radius, fill, outline=None):
        cx, cy, radius = int(round(cx)), int(round(cy)), int(round(radius))
        r2 = radius * radius
        inner2 = max(0, radius - 2) ** 2
        for dy in range(-radius, radius + 1):
            y = cy + dy
            if not (0 <= y < self.height):
                continue
            span = int((r2 - dy * dy) ** 0.5) if r2 >= dy * dy else -1
            for dx in range(-span, span + 1):
                d2 = dx * dx + dy * dy
                colour = fill if (outline is None or d2 <= inner2) else outline
                self._blend(cx + dx, y, colour)

    def line(self, x0, y0, x1, y1, colour, thickness=1):
        x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        half = thickness // 2
        while True:
            for ox in range(-half, half + 1):
                for oy in range(-half, half + 1):
                    self._blend(x0 + ox, y0 + oy, colour)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def triangle(self, apex, base_a, base_b, colour):
        """Filled triangle by scanline. Needed for arrowheads: without a filled
        primitive the rendered figures carry no direction cue at all."""
        pts = [apex, base_a, base_b]
        ys = [int(round(p[1])) for p in pts]
        top, bottom = max(0, min(ys)), min(self.height - 1, max(ys))

        def edge_x(p0, p1, y):
            if abs(p1[1] - p0[1]) < 1e-9:
                return None
            t = (y - p0[1]) / (p1[1] - p0[1])
            if t < 0.0 or t > 1.0:
                return None
            return p0[0] + t * (p1[0] - p0[0])

        for y in range(top, bottom + 1):
            crossings = []
            for a, b in ((pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])):
                x = edge_x(a, b, y + 0.5)
                if x is not None:
                    crossings.append(x)
            if len(crossings) < 2:
                continue
            x0, x1 = int(round(min(crossings))), int(round(max(crossings)))
            for x in range(x0, x1 + 1):
                self._blend(x, y, colour)

    def arrowhead(self, tip, direction, length, width, colour):
        """Arrowhead at `tip` pointing along the unit vector `direction`."""
        ux, uy = direction
        norm = (ux * ux + uy * uy) ** 0.5
        if norm < 1e-9:
            return
        ux, uy = ux / norm, uy / norm
        px, py = -uy, ux
        back = (tip[0] - ux * length, tip[1] - uy * length)
        self.triangle(tip,
                      (back[0] + px * width / 2.0, back[1] + py * width / 2.0),
                      (back[0] - px * width / 2.0, back[1] - py * width / 2.0),
                      colour)

    def polyline(self, points, colour, thickness=1):
        for i in range(len(points) - 1):
            self.line(points[i][0], points[i][1],
                      points[i + 1][0], points[i + 1][1], colour, thickness)

    def bar(self, x, y, width, height, colour):
        """Solid block standing in for text too small to draw as glyphs.

        A preview that clamps its font to a 1x minimum draws poster text an
        order of magnitude oversized relative to the geometry, which makes the
        one view of a whole-model map unusable for judging it. A bar of the
        label's true length and height reports the ink honestly: it says "text
        of this footprint, illegible at this size", which is the fact that
        matters.
        """
        left, right = int(round(x)), int(round(x + width))
        top, bottom = int(round(y - height / 2.0)), int(round(y + height / 2.0))
        if right <= left:
            right = left + 1
        if bottom <= top:
            bottom = top + 1
        for py in range(max(0, top), min(self.height, bottom)):
            for px in range(max(0, left), min(self.width, right)):
                self._blend(px, py, colour)

    def text(self, x, y, message, colour, scale=1):
        """Draw `message` with its left edge at x and vertically centred on y."""
        cursor = int(round(x))
        top = int(round(y)) - (GLYPH_HEIGHT * scale) // 2
        for char in str(message).upper():
            glyph = _FONT.get(char, _UNKNOWN)
            for col in range(5):
                bits = glyph[col]
                for row in range(GLYPH_HEIGHT):
                    if bits & (1 << row):
                        for sx in range(scale):
                            for sy in range(scale):
                                self._blend(cursor + col * scale + sx,
                                            top + row * scale + sy, colour)
            cursor += GLYPH_WIDTH * scale

    def save(self, path, dpi=300):
        """Write the PNG, declaring `dpi` so viewers report the right print size.

        The pHYs chunk is metadata only -- it does not add detail. What makes a
        zoom sharp is the pixel count, which `preview.render` decides. Without
        pHYs, though, every consumer assumes 72 dpi and reports a figure four
        times too large on paper, so both halves are needed.
        """
        raw = b"".join(b"\x00" + bytes(row) for row in self.rows)
        compressed = zlib.compress(raw, 6)

        def chunk(tag, payload):
            return (struct.pack(">I", len(payload)) + tag + payload
                    + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

        header = struct.pack(">2I5B", self.width, self.height, 8, 2, 0, 0, 0)
        # pHYs: pixels per metre, unit 1 = metre.
        per_metre = int(round(dpi / 0.0254))
        phys = struct.pack(">2IB", per_metre, per_metre, 1)
        with open(path, "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n")
            handle.write(chunk(b"IHDR", header))
            handle.write(chunk(b"pHYs", phys))
            handle.write(chunk(b"IDAT", compressed))
            handle.write(chunk(b"IEND", b""))
        return path


def text_width(message, scale=1):
    return len(str(message)) * GLYPH_WIDTH * scale
