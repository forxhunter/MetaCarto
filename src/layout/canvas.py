"""One canvas per species: every pathway of a model on a single drawing.

`compose.py` already puts a whole model on one map, and it is mostly white:
55-60% of its area has nothing drawn within a node spacing, and the largest
empty square is up to a fifth of the drawing (`metrics.blank_space`). The
reason is structural. Each pathway is drawn on its own, reduced to its bounding
rectangle, and the rectangles are packed. A rectangle is the wrong abstraction
twice over: the space *inside* it between a pathway's pieces is lost, and
rectangles of unequal shape never tile, so the space between them is lost too.

So the merge is planned on the drawings' real shapes, before anything is
placed. Every pathway drawing is split into its connected pieces and each
piece is rasterised into the cells its ink covers -- nodes, edges, labels --
with enclosed holes filled, so nothing is ever dropped inside a TCA ring. A
pathway's pieces are packed into three candidate arrangements (tall, square,
wide); the canvas picks whichever fits the space left when the pathway's turn
comes. Then, in two stages:

  the core      every region but transport, grown outward from central
                carbon, one KEGG superclass at a time. Each pathway takes the
                position needing the least canvas at the page's aspect, snug
                against what is there, pulled to its own region and to the
                pathways it exchanges metabolites with (`build_meta_graph`).
                Regions come out cohesive and free-form; the core is round.

  the membrane  transport and exchange. Their pathways are one-reaction
                carriers with no shape of their own to keep, so they are laid
                piece by piece into a frame sized for them around the core,
                against its edges first -- filling the corners a round core
                leaves on a rectangular page, and putting transport where a
                cell has it, at the boundary. Each pathway's pieces stay
                together and start beside the metabolism they serve.

Clearance grows with the grouping -- pieces of one pathway 1 cell apart,
pathways 3, regions and the membrane 5 -- so grouping is carried by proximity,
and every pathway and region is captioned. Collision is tested on the grid
with that clearance, so overlap is impossible by construction rather than
checked for afterwards; `cross_overlaps` is the check anyway.

Positions are evaluated all at once by cross-correlating the occupancy grid
with the shape (numpy FFT) over a window near where the shape belongs, which is
what makes a free-form search affordable on a ten-thousand-reaction model.

Alternatives that were built and measured, and lost: a fixed-width strip filled
from the top (flat sides, but regions smeared into bands and a large empty
wedge at the bottom), and a fixed frame with the membrane laid first (the core
then grew round inside a square ring and left a moat that the larger lipid and
glycan pathways could not fit into, so they were pushed outside the frame).
"""

import math

import numpy as np

from .compose import _offset_tile
from .metrics import _bezier_points, _label_boxes
from .render import CHAR_WIDTH_RATIO, LINE_HEIGHT_RATIO
from . import taxonomy

CELL = 80.0              # map units per grid cell
PIECE_GAP = 1            # clearance in cells between pieces of one pathway
PATHWAY_GAP = 3          # ... between pathways of one region
REGION_GAP = 5           # ... between regions
CAPTION_FONT = 12.0      # font_size_base of a pathway caption
REGION_FONT = 26.0       # ... of a region caption
TITLE_FONT = 40.0        # ... of the canvas title
TEXT_SCALE = 3.0         # text labels render at font_size_base * 3
CANVAS_ASPECT = 1.414    # landscape A-series: what a poster is printed on
CANVAS_PADDING = 400.0

# Scoring weights for a core placement. Area dominates: a position that grows
# the canvas loses to one that does not. Among those that fit, contact fills
# holes, and the anchor keeps related pathways next to each other.
CONTACT_WEIGHT = 0.35
ANCHOR_WEIGHT = 0.6
CONTACT_REACH = 2        # cells beyond the clearance that count as contact

_RADIUS = {"multimarker": 6.0, "midmarker": 11.0}


# --------------------------------------------------------------------------
# grid primitives
# --------------------------------------------------------------------------

def _dilate(mask, k):
    """Grow a boolean mask by `k` cells on every side (Chebyshev).

    The result is `2k` larger in each dimension; its cell (0, 0) is the
    input's (-k, -k).
    """
    if k <= 0:
        return mask.copy()
    padded = np.pad(mask, 2 * k).astype(np.int32)
    summed = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    n = 2 * k + 1
    window = (summed[n:, n:] - summed[:-n, n:]
              - summed[n:, :-n] + summed[:-n, :-n])
    return window > 0


def _fill_holes(mask):
    """Mark as occupied every free cell that cannot reach the border."""
    try:
        from scipy.ndimage import binary_fill_holes
        return binary_fill_holes(mask)
    except ImportError:
        pass
    free = ~mask
    reach = np.zeros_like(mask)
    reach[0, :], reach[-1, :] = free[0, :], free[-1, :]
    reach[:, 0], reach[:, -1] = free[:, 0], free[:, -1]
    while True:
        grown = np.zeros_like(reach)
        grown[1:, :] |= reach[:-1, :]
        grown[:-1, :] |= reach[1:, :]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown = (grown | reach) & free
        if (grown == reach).all():
            break
        reach = grown
    return mask | (free & ~reach)


def _correlate(grid, kernel):
    """sum(grid[r:r+h, c:c+w] * kernel) for every valid (r, c), via FFT."""
    rows, cols = grid.shape
    h, w = kernel.shape
    shape = (rows, cols)
    spectrum = np.fft.rfft2(grid.astype(np.float64), shape)
    padded = np.zeros(shape)
    padded[:h, :w] = kernel
    result = np.fft.irfft2(spectrum * np.conj(np.fft.rfft2(padded, shape)), shape)
    return result[:rows - h + 1, :cols - w + 1]


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------

def _text_box(text, font):
    size = font * TEXT_SCALE
    return len(str(text)) * size * CHAR_WIDTH_RATIO, size * LINE_HEIGHT_RATIO


def rasterise(body, cell=CELL, fill=True):
    """Cells covered by a drawing's ink: (mask, origin_x, origin_y).

    The origin is the map position of cell (0, 0)'s corner, aligned to the
    grid, so translating the drawing by whole cells keeps it aligned.
    """
    nodes = body["nodes"]
    discs, rects = [], []
    for node in nodes.values():
        if node["node_type"] == "metabolite":
            radius = 30.0 if node.get("node_is_primary", True) else 16.0
        else:
            radius = _RADIUS.get(node["node_type"], 6.0)
        discs.append((node["x"], node["y"], radius))
    for reaction in body["reactions"].values():
        for segment in reaction["segments"].values():
            a, b = nodes.get(segment["from_node_id"]), nodes.get(segment["to_node_id"])
            if a is None or b is None:
                continue
            p0, p3 = (a["x"], a["y"]), (b["x"], b["y"])
            steps = max(1, int(math.hypot(p3[0] - p0[0], p3[1] - p0[1]) / (cell / 3.0)))
            if segment.get("b1") and segment.get("b2"):
                pts = _bezier_points(p0, (segment["b1"]["x"], segment["b1"]["y"]),
                                     (segment["b2"]["x"], segment["b2"]["y"]), p3, steps)
            else:
                pts = ((p0[0] + (p3[0] - p0[0]) * i / steps,
                        p0[1] + (p3[1] - p0[1]) * i / steps) for i in range(steps + 1))
            discs.extend((x, y, 4.0) for x, y in pts)
    for left, top, right, bottom, _ in _label_boxes(body):
        rects.append((left, top, right, bottom))
    for x, y, r in discs:
        rects.append((x - r, y - r, x + r, y + r))

    if not rects:
        return np.zeros((1, 1), dtype=bool), 0.0, 0.0
    ox = math.floor(min(r[0] for r in rects) / cell) * cell
    oy = math.floor(min(r[1] for r in rects) / cell) * cell
    cols = int((max(r[2] for r in rects) - ox) // cell) + 1
    rows = int((max(r[3] for r in rects) - oy) // cell) + 1
    mask = np.zeros((rows, cols), dtype=bool)
    for left, top, right, bottom in rects:
        mask[int((top - oy) // cell):int((bottom - oy) // cell) + 1,
             int((left - ox) // cell):int((right - ox) // cell) + 1] = True
    return (_fill_holes(mask) if fill else mask), ox, oy


def split_pieces(escher_map):
    """A drawing's connected pieces, each a {nodes, reactions} body.

    Two nodes are in one piece when a segment joins them, so a reaction --
    whose segments all run through its own markers -- always lands whole in
    one piece. The drawing's own title and attribution are dropped; the canvas
    captions every pathway itself.
    """
    body = escher_map[1]
    parent = {node_id: node_id for node_id in body["nodes"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for reaction in body["reactions"].values():
        for segment in reaction["segments"].values():
            a, b = segment["from_node_id"], segment["to_node_id"]
            if a in parent and b in parent:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    pieces = {}
    for node_id, node in body["nodes"].items():
        pieces.setdefault(find(node_id), {"nodes": {}, "reactions": {}})["nodes"][node_id] = node
    for reaction_id, reaction in body["reactions"].items():
        for segment in reaction["segments"].values():
            if segment["from_node_id"] in parent:
                pieces[find(segment["from_node_id"])]["reactions"][reaction_id] = reaction
                break
    # Deterministic order: largest first, then by the smallest node id.
    return sorted(pieces.values(), key=lambda p: (-len(p["nodes"]), min(p["nodes"])))


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------

class Packer:
    """Places free-form shapes on a growing grid without overlap.

    Shapes are placed in groups. Within the current group they keep `gap`
    cells of clearance, and `outer_gap` from everything placed in earlier
    groups -- so a region's pathways sit closer to each other than to the next
    region's, and the grouping is visible without drawing a boundary. `freeze`
    sets everything placed so far behind a third clearance, `fixed_gap`, for a
    stage that packs at a finer grain around it.

    `evaluate` finds the best position for a shape; `commit` places it. They
    are separate so the caller can compare several arrangements of the same
    pathway and keep whichever fits the space that is left.
    """

    def __init__(self, gap, aspect=1.0, outer_gap=None, room=64):
        self.gap = gap
        self.outer_gap = gap if outer_gap is None else outer_gap
        self.aspect = aspect
        self.done = np.zeros((room, room), dtype=bool)    # earlier groups
        self.group = np.zeros((room, room), dtype=bool)   # the current group
        self.fixed = np.zeros((room, room), dtype=bool)   # an earlier stage
        self.fixed_gap = 0
        self.offsets = {}
        self.box = None                                   # all placed ink
        self.group_box = None
        # Any fixed lines the caller works against (a strip's edges), as
        # grid coordinates: {name: (row or None, col or None)}. Padding the
        # grid moves them with everything else.
        self.marks = {}

    def new_group(self):
        self.done |= self.group
        self.group[:] = False
        self.group_box = None

    def freeze(self, fixed_gap):
        """Hold everything placed so far `fixed_gap` cells away from what follows."""
        self.new_group()
        self.fixed |= self.done
        self.done[:] = False
        self.fixed_gap = fixed_gap

    def reserve(self, masks):
        """Make room for the largest of `masks` before any position is computed.

        Making room can pad the grid, which moves every coordinate. Done
        between two candidate evaluations, it leaves the first one's answer --
        and any anchor computed before it -- in the old coordinates.
        """
        if self.box is not None:
            self._ensure_room(max(m.shape[0] for m in masks), max(m.shape[1] for m in masks))

    def _margin(self, h, w):
        return max(h, w) + 2 * max(self.gap, self.outer_gap, self.fixed_gap) + CONTACT_REACH + 2

    def _ensure_room(self, h, w):
        """Keep a margin around the placed ink wider than the next shape."""
        need = self._margin(h, w)
        rows, cols = self.done.shape
        r0, c0, r1, c1 = self.box
        pads = [max(0, need - r0), max(0, need - (rows - 1 - r1)),
                max(0, need - c0), max(0, need - (cols - 1 - c1))]
        if not any(pads):
            return
        # Grow generously, so a long run of placements does not re-pad (and
        # re-copy) the grid every time.
        top, bottom, left, right = (x and x + need for x in pads)
        widths = ((top, bottom), (left, right))
        self.done = np.pad(self.done, widths)
        self.group = np.pad(self.group, widths)
        self.fixed = np.pad(self.fixed, widths)
        self.offsets = {k: (r + top, c + left) for k, (r, c) in self.offsets.items()}
        self.marks = {k: (None if r is None else r + top, None if c is None else c + left)
                      for k, (r, c) in self.marks.items()}
        self.box = (r0 + top, c0 + left, r1 + top, c1 + left)
        if self.group_box is not None:
            g0, g1, g2, g3 = self.group_box
            self.group_box = (g0 + top, g1 + left, g2 + top, g3 + left)

    def _window(self, which, pad, kernel, rows, cols):
        """Correlation of a grid, padded by `pad`, with `kernel`, for kernel
        top-left positions rows[0]..rows[1] x cols[0]..cols[1] (padded coords).

        Only the slice of the grid those positions can touch is transformed,
        which is what keeps a placement near its region cheap on a canvas the
        size of Recon3D's.
        """
        grid = {"group": self.group, "done": self.done, "fixed": self.fixed}.get(which)
        if grid is None:
            grid = self.group | self.done | self.fixed
        h, w = kernel.shape
        r0, r1 = rows[0] - pad, rows[1] + h - pad          # grid rows, half-open
        c0, c1 = cols[0] - pad, cols[1] + w - pad
        sub = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        gr0, gr1 = max(r0, 0), min(r1, grid.shape[0])
        gc0, gc1 = max(c0, 0), min(c1, grid.shape[1])
        if gr0 < gr1 and gc0 < gc1:
            sub[gr0 - r0:gr1 - r0, gc0 - c0:gc1 - c0] = grid[gr0:gr1, gc0:gc1]
        if not sub.any():
            return np.zeros((rows[1] - rows[0] + 1, cols[1] - cols[0] + 1))
        return _correlate(sub, kernel)

    @staticmethod
    def _ink_box(mask):
        rows = np.flatnonzero(mask.any(1))
        cols = np.flatnonzero(mask.any(0))
        return rows[0], rows[-1], cols[0], cols[-1]

    def candidates(self, mask, gap=None, window=None):
        """Every position `mask` may take, with how snugly it would sit.

        Returns (feasible, contact, rows, cols, ink box): `rows` and `cols`
        broadcast to the grid position of the mask's cell (0, 0) for each
        candidate. `window` = (row_lo, row_hi, col_lo, col_hi) limits that
        position; None searches the whole grid.
        """
        h, w = mask.shape
        self._ensure_room(h, w)
        g_in = self.gap if gap is None else gap
        g_out = max(self.outer_gap, g_in)
        rows, cols = self.done.shape

        # Positions are indexed by the top-left of `mask` dilated by g_in.
        lo_r, hi_r = 0, rows - (h + 2 * g_in)
        lo_c, hi_c = 0, cols - (w + 2 * g_in)
        if window is not None:
            lo_r = max(lo_r, window[0] - g_in)
            hi_r = min(hi_r, window[1] - g_in)
            lo_c = max(lo_c, window[2] - g_in)
            hi_c = min(hi_c, window[3] - g_in)
        if hi_r < lo_r or hi_c < lo_c:
            return None
        span_r, span_c = (lo_r, hi_r), (lo_c, hi_c)

        # A kernel dilated by more than g_in lines up with the same index once
        # its grid is padded by the difference.
        feasible = self._window("group", 0, _dilate(mask, g_in), span_r, span_c) < 0.5
        feasible &= self._window("done", g_out - g_in, _dilate(mask, g_out),
                                 span_r, span_c) < 0.5
        if self.fixed_gap:
            g_fix = max(self.fixed_gap, g_in)
            feasible &= self._window("fixed", g_fix - g_in, _dilate(mask, g_fix),
                                     span_r, span_c) < 0.5
        halo = _dilate(mask, g_in + CONTACT_REACH)
        contact = self._window("all", CONTACT_REACH, halo, span_r, span_c)
        contact = contact / max(1.0, float(halo.sum()))

        pr = np.arange(lo_r, hi_r + 1)[:, None] + g_in
        pc = np.arange(lo_c, hi_c + 1)[None, :] + g_in
        return feasible, contact, pr, pc, self._ink_box(mask)

    def evaluate(self, mask, anchor=None, anchor_weight=ANCHOR_WEIGHT, gap=None,
                 window=None, terms=None):
        """(score, row, col) of the best position for `mask`; lower is better.

        The default score is compactness: the least canvas at `aspect`, snug,
        and near `anchor`. `terms(pr, pc, box, contact)` replaces it.
        """
        h, w = mask.shape
        if self.box is None:
            need = 2 * self._margin(h, w)
            if self.done.shape[0] < h + need or self.done.shape[1] < w + need:
                self.done = np.zeros((h + need, w + need), dtype=bool)
                self.group = np.zeros_like(self.done)
                self.fixed = np.zeros_like(self.done)
            return 0.0, need // 2, need // 2

        found = self.candidates(mask, gap, window)
        if found is None:
            return math.inf, 0, 0
        feasible, contact, pr, pc, box = found
        if terms is not None:
            score = terms(pr, pc, box, contact)
        else:
            ir0, ir1, ic0, ic1 = box
            r0, c0, r1, c1 = self.box
            new_h = np.maximum(r1, pr + ir1) - np.minimum(r0, pr + ir0) + 1
            new_w = np.maximum(c1, pc + ic1) - np.minimum(c0, pc + ic0) + 1
            side = np.maximum(new_w / self.aspect, new_h)
            current = max((c1 - c0 + 1) / self.aspect, r1 - r0 + 1)
            score = (side / current) ** 2 - CONTACT_WEIGHT * contact
            if anchor is not None:
                distance = np.hypot(pr + (ir0 + ir1) / 2.0 - anchor[0],
                                    pc + (ic0 + ic1) / 2.0 - anchor[1])
                score = score + anchor_weight * distance / (current * math.sqrt(self.aspect))
        score = np.where(feasible, score, np.inf)

        flat = int(np.argmin(score))          # first minimum: top-left wins ties
        gr, gc = divmod(flat, score.shape[1])
        return float(score[gr, gc]), int(pr[gr, 0]), int(pc[0, gc])

    def commit(self, key, mask, r, c):
        h, w = mask.shape
        ir0, ir1, ic0, ic1 = self._ink_box(mask)
        self.group[r:r + h, c:c + w] |= mask
        self.offsets[key] = (r, c)
        box = (r + ir0, c + ic0, r + ir1, c + ic1)

        def union(a, b):
            if a is None:
                return b
            return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

        self.box = union(self.box, box)
        self.group_box = union(self.group_box, box)

    def place(self, key, mask, anchor=None, anchor_weight=ANCHOR_WEIGHT, gap=None):
        _, r, c = self.evaluate(mask, anchor, anchor_weight, gap)
        self.commit(key, mask, r, c)
        return r, c

    def centre_of(self, key, mask):
        r, c = self.offsets[key]
        return r + mask.shape[0] / 2.0, c + mask.shape[1] / 2.0

    def finish(self):
        """(mask, offsets) cropped to the placed ink."""
        r0, c0, r1, c1 = self.box
        mask = (self.done | self.group | self.fixed)[r0:r1 + 1, c0:c1 + 1].copy()
        offsets = {k: (r - r0, c - c0) for k, (r, c) in self.offsets.items()}
        return mask, offsets


def _caption_mask(text, font):
    width, height = _text_box(text, font)
    cw = int(math.ceil(width / CELL)) + 1
    ch = int(math.ceil(height / CELL))
    return np.ones((ch, cw), dtype=bool)


def _with_caption(mask, text, font, gap=1):
    """Put a caption above a shape's top-left: (mask, caption cells, shape offset).

    The caption is part of the shape from here on, so the next level up packs
    around it as it does around ink.
    """
    caption = _caption_mask(text, font)
    ch, cw = caption.shape
    rows, cols = mask.shape
    out = np.zeros((rows + ch + gap, max(cols, cw)), dtype=bool)
    out[ch + gap:, :cols] = mask
    out[:ch, :cw] = True
    # Enclosed gaps between this shape's parts are its own; the level above
    # must not drop a neighbour into them.
    return _fill_holes(out), (0, 0, ch), (ch + gap, 0)


def _pack(shapes, gap, aspect, order):
    packer = Packer(gap, aspect)
    for key in order:
        packer.place(key, shapes[key])
    return packer.finish()


# --------------------------------------------------------------------------
# the canvas
# --------------------------------------------------------------------------

# Shapes tried for each pathway. One arrangement per pathway would fix its
# outline before anyone knew where it was going; three let the canvas take the
# tall one beside a column and the wide one under a row.
PATHWAY_ASPECTS = (0.6, 1.0, 1.7)

# How strongly a pathway is drawn towards the rest of its region, against the
# pull of the pathways it exchanges metabolites with: a region has to read as
# one place before the links between places matter.
REGION_PULL = 2.0


def _arrangements(name, tile):
    """A pathway's pieces, their shapes, and its distinct candidate arrangements."""
    pieces = split_pieces(tile)
    shapes, origin = {}, {}
    for p, piece in enumerate(pieces):
        shapes[p], ox, oy = rasterise(piece)
        origin[p] = (ox, oy)
    order = sorted(shapes, key=lambda p: (-int(shapes[p].sum()), p))
    options = []
    for aspect in PATHWAY_ASPECTS:
        inner, offsets = _pack(shapes, PIECE_GAP, aspect, order)
        if any(np.array_equal(inner, o["inner"]) for o in options):
            continue
        mask, caption, shift = _with_caption(inner, name, CAPTION_FONT)
        options.append({"inner": inner, "mask": mask, "offsets": offsets,
                        "shift": shift, "caption": caption})
    return {"pieces": pieces, "origin": origin, "shapes": shapes, "options": options}


def _weighted_centre(points):
    total = sum(w for _, w in points)
    if not total:
        return None
    return (sum(p[0] * w for p, w in points) / total,
            sum(p[1] * w for p, w in points) / total)


class _Placement:
    """What has been placed where, for anchors and captions."""

    def __init__(self, packer, drawings, neighbours):
        self.packer, self.drawings, self.neighbours = packer, drawings, neighbours
        self.chosen = {}          # rigid pathway -> option index
        self.fluid = {}           # fluid pathway -> placed piece indices

    def placed(self, name):
        return name in self.chosen or bool(self.fluid.get(name))

    def centre(self, name):
        if name in self.chosen:
            mask = self.drawings[name]["options"][self.chosen[name]]["mask"]
            return self.packer.centre_of(name, mask)
        shapes = self.drawings[name]["shapes"]
        return _weighted_centre([(self.packer.centre_of((name, p), shapes[p]),
                                  int(shapes[p].sum())) for p in self.fluid[name]])

    def weight(self, name):
        if name in self.chosen:
            return int(self.drawings[name]["options"][self.chosen[name]]["mask"].sum())
        return sum(int(self.drawings[name]["shapes"][p].sum()) for p in self.fluid[name])

    def region_centre(self, members):
        return _weighted_centre([(self.centre(m), self.weight(m)) for m in members])

    def linked_centre(self, name, only=None):
        return _weighted_centre([(self.centre(o), w) for o, w in self.neighbours.get(name, ())
                                 if self.placed(o) and (only is None or o in only)])

    def caption(self, key, text, font, anchor_box, gap=1):
        """A caption just above the top-left of `anchor_box`, as a heading sits."""
        packer = self.packer
        caption = _caption_mask(text, font)
        packer.reserve([caption])
        ch, cw = caption.shape
        g0, g1, _, _ = anchor_box()
        anchor = (g0 - ch / 2.0 - 1, g1 + cw / 2.0)
        window = (g0 - ch - 8, g0 + 12, g1 - 8, g1 + cw + 8)
        score, r, c = packer.evaluate(caption, anchor, anchor_weight=8.0, gap=gap, window=window)
        if not math.isfinite(score):
            score, r, c = packer.evaluate(caption, anchor, anchor_weight=8.0, gap=gap)
        packer.commit(key, caption, r, c)
        return ch


def _place_rigid(state, regions):
    """Grow regions outward from the first, each pathway as one drawn shape.

    Every placement takes the least canvas at the target aspect, sitting snug
    and pulled towards the rest of its region and the pathways it exchanges
    metabolites with. Regions come out cohesive and free-form, central carbon
    in the middle; on its own the whole grows as a round blob.
    """
    packer = state.packer
    for label, members in regions.items():
        packer.new_group()
        placed = []
        options_of = {n: state.drawings[n]["options"] for n in members}
        size = {n: int(options_of[n][0]["inner"].sum()) for n in members}
        for name in sorted(members, key=lambda n: (-size[n], n)):
            packer.reserve([o["mask"] for o in options_of[name]])
            points = []
            if placed:
                pull = REGION_PULL * max(1.0, sum(w for o, w in state.neighbours.get(name, ())
                                                  if state.placed(o)))
                points.append((state.region_centre(placed), pull))
            points.extend((state.centre(o), w) for o, w in state.neighbours.get(name, ())
                          if state.placed(o))
            anchor = _weighted_centre(points) if points else None
            best = None
            for i, option in enumerate(options_of[name]):
                score, r, c = packer.evaluate(option["mask"], anchor)
                if best is None or score < best[0]:
                    best = (score, i, r, c)
            _, i, r, c = best
            packer.commit(name, options_of[name][i]["mask"], r, c)
            state.chosen[name] = i
            placed.append(name)
        state.caption(("region", label), label.upper(), REGION_FONT,
                      lambda: packer.group_box)


# The membrane. Transport and exchange pathways are made of independent
# one-reaction pieces -- a carrier moving one compound -- so unlike a pathway
# they have no shape of their own to keep, and can fill whatever space the
# rigid regions leave. Laid last, piece by piece, between the rigid core and a
# frame sized for them, they fill the page out to its corners. That is also
# where they belong: at the cell's boundary, around the metabolism inside.
BOUNDARY = "Transport and exchange"
MEMBRANE_FILL = 0.80      # expected share of its space the membrane will fill
MEMBRANE_EDGE = 2.0       # per frame width a piece sits in from the frame
MEMBRANE_COHESION = 3.0   # pull towards the rest of the piece's own pathway
MEMBRANE_LINKS = 0.6      # a pathway starts beside the metabolism it serves
MEMBRANE_OUTSIDE = 6.0    # per frame width a piece reaches past the frame
MEMBRANE_MIN_CORE = 12    # rigid pathways before a membrane is worth laying


def _place_membrane(state, label, members):
    packer = state.packer
    drawings = state.drawings
    core = set(state.chosen)

    # Size the frame: the core as it stands, plus room for every piece.
    packer.freeze(REGION_GAP)
    r0, c0, r1, c1 = packer.box
    held = int(_dilate(packer.fixed, 1).sum())
    fluid = sum(int(_dilate(drawings[n]["shapes"][p], 1).sum())
                for n in members for p in drawings[n]["shapes"])
    fluid += sum(int(_dilate(_caption_mask(n, CAPTION_FONT), 1).sum()) for n in members)
    fluid += int(_dilate(_caption_mask(label.upper(), REGION_FONT), 1).sum())
    area = held + fluid / MEMBRANE_FILL
    width = max(int(round(math.sqrt(area * CANVAS_ASPECT))), (c1 - c0 + 1) + 2 * REGION_GAP + 2)
    height = max(int(round(area / width)), (r1 - r0 + 1) + 2 * REGION_GAP + 2)
    centre = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
    top = int(round(centre[0] - height / 2.0))
    left = int(round(centre[1] - width / 2.0))
    packer.marks["frame-top-left"] = (top, left)
    packer.marks["frame-bottom-right"] = (top + height - 1, left + width - 1)
    # Make the grid cover the whole frame before anything is placed in it.
    # Growing it can shift every coordinate; the frame's mark says by how much.
    packer.box = (min(r0, top), min(c0, left), max(r1, top + height - 1), max(c1, left + width - 1))
    before = packer.marks["frame-top-left"]
    packer.reserve([np.ones((1, 1), dtype=bool)])
    after = packer.marks["frame-top-left"]
    dr, dc = after[0] - before[0], after[1] - before[1]
    packer.box = (r0 + dr, c0 + dc, r1 + dr, c1 + dc)
    # Within the membrane: pieces of one pathway close, pathways apart.
    packer.gap, packer.outer_gap = PIECE_GAP, PATHWAY_GAP

    def frame():
        (t, l), (b, r) = packer.marks["frame-top-left"], packer.marks["frame-bottom-right"]
        return t, l, b, r

    def terms_for(own, linked):
        def terms(pr, pc, box, contact):
            t, l, b, r = frame()
            ir0, ir1, ic0, ic1 = box
            top_d, bottom_d = pr + ir0 - t, b - (pr + ir1)
            left_d, right_d = pc + ic0 - l, r - (pc + ic1)
            inward = np.maximum(0, np.minimum(np.minimum(top_d, bottom_d),
                                              np.minimum(left_d, right_d)))
            outside = (np.maximum(0, -top_d) + np.maximum(0, -bottom_d)
                       + np.maximum(0, -left_d) + np.maximum(0, -right_d))
            score = ((MEMBRANE_EDGE * inward + MEMBRANE_OUTSIDE * outside) / width
                     - CONTACT_WEIGHT * contact)
            centre_r, centre_c = pr + (ir0 + ir1) / 2.0, pc + (ic0 + ic1) / 2.0
            if own is not None:
                score = score + MEMBRANE_COHESION * np.hypot(
                    centre_r - own[0], centre_c - own[1]) / width
            elif linked is not None:
                score = score + MEMBRANE_LINKS * np.hypot(
                    centre_r - linked[0], centre_c - linked[1]) / width
            return score
        return terms

    # The region's caption at the frame's top-left corner, where the membrane
    # begins.
    caption = _caption_mask(label.upper(), REGION_FONT)
    t, l, _, _ = frame()
    score, r, c = packer.evaluate(caption, (t + caption.shape[0] / 2.0, l + caption.shape[1] / 2.0),
                                  anchor_weight=8.0, gap=1,
                                  window=(t - 2, t + 30, l - 2, l + 30))
    if not math.isfinite(score):
        score, r, c = packer.evaluate(caption, (t, l), anchor_weight=8.0, gap=1)
    packer.commit(("region", label), caption, r, c)

    pieces_count = {n: len(drawings[n]["shapes"]) for n in members}
    for name in sorted(members, key=lambda n: (-pieces_count[n], n)):
        packer.new_group()
        shapes = drawings[name]["shapes"]
        state.fluid[name] = []
        linked = state.linked_centre(name, only=core)
        for p in sorted(shapes, key=lambda p: (-int(shapes[p].sum()), p)):
            mask = shapes[p]
            packer.reserve([mask])
            own = state.centre(name) if state.fluid[name] else None
            h, w = mask.shape
            if packer.group_box is not None:
                g0, g1, g2, g3 = packer.group_box
                reach = max(h, w) + 6
                window = (g0 - reach, g2 + reach, g1 - reach, g3 + reach)
            else:
                t, l, b, r = frame()
                window = (t - h, b + 1, l - w, r + 1)
            score, rr, cc = packer.evaluate(mask, window=window, terms=terms_for(own, linked))
            if not math.isfinite(score):
                score, rr, cc = packer.evaluate(mask, terms=terms_for(own, linked))
            packer.commit((name, p), mask, rr, cc)
            state.fluid[name].append(p)
        state.caption(("caption", name), name, CAPTION_FONT, lambda: packer.group_box)


def compose_canvas(tiles, labels, meta_graph, map_name, author="AutoLayout",
                   description="", membrane=True):
    """Every pathway drawing of one model on a single, densely packed canvas.

    `tiles` is [(pathway name, escher map)], `labels` {pathway name: region}
    (from `taxonomy.classify`), `meta_graph` the flow between pathways from
    `compose.build_meta_graph`. With `membrane`, transport and exchange are
    laid last, piece by piece, around the rest (`_place_membrane`).
    """
    tiles = [(name, m) for name, m in tiles if m and m[1]["nodes"]]
    if not tiles:
        return None

    index_of = {name: i for i, (name, _) in enumerate(tiles)}
    drawings = {name: _arrangements(name, tile) for name, tile in tiles}

    weight = {}
    if meta_graph is not None:
        for u, v, data in meta_graph.edges(data=True):
            key = tuple(sorted((u, v)))
            weight[key] = weight.get(key, 0) + data.get("weight", 1)
    neighbours = {}
    for (a, b), w in weight.items():
        neighbours.setdefault(a, []).append((b, w))
        neighbours.setdefault(b, []).append((a, w))

    regions = {}
    for name, _ in tiles:
        regions.setdefault(labels.get(name) or taxonomy.superclass(name), []).append(name)
    # Canonical order: central carbon first, at the centre, everything else
    # around it.
    regions = dict(sorted(regions.items(), key=lambda kv: taxonomy.order_index(kv[0])))
    boundary = regions.get(BOUNDARY, []) if membrane else []
    rigid_count = sum(len(m) for label, m in regions.items() if label != BOUNDARY)
    # A membrane of one or two pathways is not a membrane, and a core of a few
    # pathways leaves corners too small to be worth breaking transport up for:
    # e_coli_core's four rigid pathways read better as one organic cluster.
    if len(boundary) < 3 or rigid_count < MEMBRANE_MIN_CORE:
        boundary = []
    rigid = {label: members for label, members in regions.items()
             if not (boundary and label == BOUNDARY)}

    packer = Packer(PATHWAY_GAP, CANVAS_ASPECT, outer_gap=REGION_GAP)
    state = _Placement(packer, drawings, neighbours)
    if rigid:
        _place_rigid(state, rigid)
    if boundary:
        if packer.box is None:
            # Nothing rigid to wrap around: lay the membrane as ordinary shapes.
            _place_rigid(state, {BOUNDARY: boundary})
        else:
            _place_membrane(state, BOUNDARY, boundary)

    canvas_mask, offsets = packer.finish()
    canvas_mask, title_cells, (top, left) = _with_caption(canvas_mask, map_name,
                                                          TITLE_FONT, gap=2)

    # Assemble: every piece moved from where it was drawn to where it was put.
    nodes, reactions, text = {}, {}, {}
    text["title"] = _caption_label(map_name, TITLE_FONT, 0, 0, title_cells)
    for label in regions:
        if ("region", label) not in offsets:
            continue
        rr, rc = offsets[("region", label)]
        ch = _caption_mask(label.upper(), REGION_FONT).shape[0]
        text[f"region_{label}"] = _caption_label(label.upper(), REGION_FONT,
                                                 rr + top, rc + left, (0, 0, ch))
    for name, _ in tiles:
        drawing = drawings[name]
        index = index_of[name]
        if name in state.chosen:
            option = drawing["options"][state.chosen[name]]
            base_r, base_c = offsets[name]
            base_r, base_c = base_r + top, base_c + left
            text[f"title_{index}"] = _caption_label(name, CAPTION_FONT, base_r, base_c,
                                                    option["caption"])
            cells = {p: (base_r + option["shift"][0] + qr, base_c + option["shift"][1] + qc)
                     for p, (qr, qc) in option["offsets"].items()}
        else:
            cr, cc = offsets[("caption", name)]
            ch = _caption_mask(name, CAPTION_FONT).shape[0]
            text[f"title_{index}"] = _caption_label(name, CAPTION_FONT, cr + top, cc + left,
                                                    (0, 0, ch))
            cells = {p: (offsets[(name, p)][0] + top, offsets[(name, p)][1] + left)
                     for p in drawing["shapes"]}
        for p, piece in enumerate(drawing["pieces"]):
            row, col = cells[p]
            ox, oy = drawing["origin"][p]
            _offset_tile([None, piece], col * CELL - ox, row * CELL - oy, f"t{index}",
                         nodes, reactions)

    rows, cols = canvas_mask.shape
    canvas = {
        "x": -CANVAS_PADDING,
        "y": -CANVAS_PADDING,
        "width": cols * CELL + 2 * CANVAS_PADDING,
        "height": rows * CELL + 2 * CANVAS_PADDING,
    }
    text["attribution"] = {
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
        {"reactions": reactions, "nodes": nodes, "text_labels": text, "canvas": canvas},
    ]


def _caption_label(text, font, row, col, caption_cells):
    """A text label filling the caption cells reserved at (row, col)."""
    r, c, ch = caption_cells
    return {
        "x": (col + c) * CELL,
        "y": (row + r) * CELL + ch * CELL / 2.0,
        "text": text,
        "font_size_base": font,
    }


def cross_overlaps(escher_map, cell=20.0):
    """Cells drawn on by two different pathways (or a pathway and a caption).

    The composition claims that no two pathways overlap; this is the check.
    Overlaps *within* one pathway drawing are the renderer's business and are
    reported by `metrics.score`; this counts only what composing introduced.
    Pathways are told apart by the tile prefix `_offset_tile` gives node ids.
    """
    body = escher_map[1]
    by_tile = {}
    for node_id, node in body["nodes"].items():
        tile = node_id.split("_", 1)[0]
        by_tile.setdefault(tile, {"nodes": {}, "reactions": {}})["nodes"][node_id] = node
    for reaction_id, reaction in body["reactions"].items():
        for segment in reaction["segments"].values():
            tile = segment["from_node_id"].split("_", 1)[0]
            by_tile.setdefault(tile, {"nodes": {}, "reactions": {}})["reactions"][reaction_id] = reaction
            break
    for key, label in body.get("text_labels", {}).items():
        by_tile[f"text:{key}"] = {"nodes": {}, "reactions": {}, "text_labels": {key: label}}

    claimed = {}
    clashes = 0
    for tile, part in by_tile.items():
        mask, ox, oy = rasterise(part, cell, fill=False)
        r0, c0 = int(round(oy / cell)), int(round(ox / cell))
        for r, c in zip(*np.nonzero(mask)):
            key = (r + r0, c + c0)
            owner = claimed.setdefault(key, tile)
            if owner != tile:
                clashes += 1
    return clashes


def organisation(escher_map, labels, meta_graph, names, reach=1500.0):
    """How well a canvas keeps biology together.

    `names` lists pathway names in tile order (tile `t{i}` is names[i]).
    Returns:
      region_cohesion  for every metabolite, whether the nearest metabolite of
                       a *different* pathway (within `reach`) is in the same
                       superclass. Measured locally rather than between
                       pathway centres, because a region drawn as a ring --
                       the transport membrane -- has its centre in the middle
                       of everything else.
      link_ratio       mean distance between pathways that exchange
                       metabolites over the mean distance between any two --
                       below 1 when related pathways are drawn near each other.
    """
    tile_of, xy = [], []
    for node_id, node in escher_map[1]["nodes"].items():
        if node["node_type"] != "metabolite":
            continue
        tile = node_id.split("_", 1)[0]
        if tile.startswith("t") and tile[1:].isdigit():
            tile_of.append(int(tile[1:]))
            xy.append((node["x"], node["y"]))
    if not xy:
        return {"region_cohesion": 1.0, "link_ratio": 1.0}
    xy = np.array(xy)
    tile_of = np.array(tile_of)
    region_of = {i: labels.get(names[i]) or taxonomy.superclass(names[i])
                 for i in set(tile_of.tolist())}

    buckets = {}
    for k, (x, y) in enumerate(xy):
        buckets.setdefault((int(x // reach), int(y // reach)), []).append(k)
    same = counted = 0
    for k, (x, y) in enumerate(xy):
        bx, by = int(x // reach), int(y // reach)
        best, best_d = None, reach
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((bx + dx, by + dy), ()):
                    if tile_of[j] == tile_of[k]:
                        continue
                    d = math.hypot(xy[j, 0] - x, xy[j, 1] - y)
                    if d < best_d:
                        best, best_d = j, d
        if best is not None:
            counted += 1
            same += region_of[tile_of[best]] == region_of[tile_of[k]]
    cohesion = same / counted if counted else 1.0

    centres = {names[i]: xy[tile_of == i].mean(axis=0) for i in set(tile_of.tolist())}
    keys = sorted(centres)
    if len(keys) < 2:
        return {"region_cohesion": cohesion, "link_ratio": 1.0}
    pts = np.array([centres[k] for k in keys])
    dist = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    index = {k: i for i, k in enumerate(keys)}
    linked, weights = [], []
    if meta_graph is not None:
        for u, v, data in meta_graph.edges(data=True):
            if u in index and v in index:
                linked.append(dist[index[u], index[v]])
                weights.append(data.get("weight", 1))
    mean_all = dist.sum() / (len(keys) * (len(keys) - 1))
    ratio = float(np.average(linked, weights=weights) / mean_all) if linked else 1.0
    return {"region_cohesion": float(cohesion), "link_ratio": ratio}
