"""A small vector chart writer, for the manuscript's plots.

matplotlib is installed in this project's environment and cannot render in it:
every backend -- Agg, pdf and svg -- aborts the interpreter with
`0xc06d007f`, a delay-load failure, on matplotlib 3.8.4 and 3.11.2 alike, with
or without any text in the figure. The C extensions import cleanly and fail
when called, so it is the environment rather than the library.

Rather than depend on it, this draws the one chart the paper needs with the
same PDF primitives `pdfout` already uses for the maps: lines, filled shapes
and Helvetica text, which need no font embedding. The output is vector and
sized in millimetres, so a chart is placed at its final width and the point
sizes in the axis labels are the sizes on the page.

Only what the manuscript uses is implemented. This is not a plotting library.
"""

from . import pdfout

PT_PER_MM = pdfout.PT_PER_MM

TEXT = "#1c252e"
AXIS = "#8a8f96"
GRID = "#e3e7ea"


def _rgb(colour):
    return pdfout._rgb(colour)


class Chart(object):
    """A single axes in page points, y up, with millimetre framing."""

    def __init__(self, width_mm, height_mm, margin=(13.0, 4.0, 10.0, 3.0)):
        # margins: left, right, bottom, top, in millimetres
        self.w = width_mm * PT_PER_MM
        self.h = height_mm * PT_PER_MM
        self.ml, self.mr, self.mb, self.mt = (m * PT_PER_MM for m in margin)
        self.parts = ["1 J", "1 j"]
        self.xlim = (0.0, 1.0)
        self.ylim = (0.0, 1.0)

    # -- coordinate mapping -------------------------------------------------

    @property
    def plot_w(self):
        return self.w - self.ml - self.mr

    @property
    def plot_h(self):
        return self.h - self.mb - self.mt

    def xy(self, x, y):
        x0, x1 = self.xlim
        y0, y1 = self.ylim
        return (self.ml + (x - x0) / (x1 - x0) * self.plot_w,
                self.mb + (y - y0) / (y1 - y0) * self.plot_h)

    # -- primitives ---------------------------------------------------------

    def _stroke(self, colour, width, dash=None):
        self.parts.append("%.3f %.3f %.3f RG" % _rgb(colour))
        self.parts.append("%.2f w" % width)
        self.parts.append("[%s] 0 d" % (" ".join("%.1f" % d for d in dash)
                                        if dash else ""))

    def polyline(self, points, colour, width=1.1, dash=None):
        if len(points) < 2:
            return
        self._stroke(colour, width, dash)
        first = self.xy(*points[0])
        self.parts.append("%.2f %.2f m" % first)
        for point in points[1:]:
            self.parts.append("%.2f %.2f l" % self.xy(*point))
        self.parts.append("S")

    def dot(self, x, y, colour, radius=1.7):
        cx, cy = self.xy(x, y)
        k = radius * pdfout.KAPPA
        self.parts.append("%.2f %.2f m" % (cx + radius, cy))
        for arc in ((cx + radius, cy + k, cx + k, cy + radius, cx, cy + radius),
                    (cx - k, cy + radius, cx - radius, cy + k, cx - radius, cy),
                    (cx - radius, cy - k, cx - k, cy - radius, cx, cy - radius),
                    (cx + k, cy - radius, cx + radius, cy - k, cx + radius, cy)):
            self.parts.append("%.2f %.2f %.2f %.2f %.2f %.2f c" % arc)
        self.parts.append("%.3f %.3f %.3f rg f" % _rgb(colour))

    def text(self, x_pt, y_pt, size, body, colour=TEXT, anchor="left"):
        width = len(body) * size * 0.5          # Helvetica average advance
        if anchor == "middle":
            x_pt -= width / 2.0
        elif anchor == "right":
            x_pt -= width
        self.parts.append("%.3f %.3f %.3f rg" % _rgb(colour))
        self.parts.append("BT /F1 %.2f Tf %.2f %.2f Td (%s) Tj ET"
                          % (size, x_pt, y_pt, pdfout._pdf_text(body)))

    # -- axes ---------------------------------------------------------------

    def axes(self, xticks, yticks, xlabel, ylabel, size=6.5,
             xfmt="%g", yfmt="%g"):
        for value in xticks:
            x, _ = self.xy(value, self.ylim[0])
            self.polyline([(value, self.ylim[0]), (value, self.ylim[1])],
                          GRID, 0.4)
            self.text(x, self.mb - 5.2, size, xfmt % value, AXIS, "middle")
        for value in yticks:
            _, y = self.xy(self.xlim[0], value)
            self.polyline([(self.xlim[0], value), (self.xlim[1], value)],
                          GRID, 0.4)
            self.text(self.ml - 3.0, y - size * 0.34, size, yfmt % value,
                      AXIS, "right")
        self._stroke(AXIS, 0.6)
        self.parts.append("%.2f %.2f m %.2f %.2f l %.2f %.2f l S"
                          % (self.ml, self.mb + self.plot_h,
                             self.ml, self.mb,
                             self.ml + self.plot_w, self.mb))
        self.text(self.ml + self.plot_w / 2.0, 1.5, size + 0.5, xlabel,
                  TEXT, "middle")
        # Rotated 90 degrees: the text matrix carries the rotation, so the
        # glyphs turn with the baseline.
        self.parts.append("%.3f %.3f %.3f rg" % _rgb(TEXT))
        self.parts.append(
            "BT /F1 %.2f Tf 0 1 -1 0 %.2f %.2f Tm (%s) Tj ET"
            % (size + 0.5, 7.0,
               self.mb + self.plot_h / 2.0 - len(ylabel) * (size + 0.5) * 0.25,
               pdfout._pdf_text(ylabel)))

    def legend(self, entries, x_pt, y_pt, size=6.0, leading=7.5):
        """entries: [(label, colour, dash)] drawn top-down from y_pt."""
        for index, (label, colour, dash) in enumerate(entries):
            y = y_pt - index * leading
            self._stroke(colour, 1.1, dash)
            self.parts.append("%.2f %.2f m %.2f %.2f l S"
                              % (x_pt, y + size * 0.3, x_pt + 13.0,
                                 y + size * 0.3))
            self.text(x_pt + 16.0, y, size, label, TEXT)

    def save(self, path):
        return pdfout.write_pdf(self.parts, self.w, self.h, path)


def ecdf(values):
    """Sorted values and their cumulative fraction, as step vertices."""
    ordered = sorted(values)
    n = len(ordered)
    points = [(ordered[0], 0.0)]
    for index, value in enumerate(ordered):
        points.append((value, (index + 1) / n))
        if index + 1 < n:
            points.append((ordered[index + 1], (index + 1) / n))
    return points
