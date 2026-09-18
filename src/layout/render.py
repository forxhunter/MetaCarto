"""Escher rendering: markers, cofactor stubs, orthogonal routing, labels
(layout_algorithm.md S5 and S7).

This module owns everything between "primary metabolites have coordinates" and
"a file Escher will open". It also fixes the v1 schema defects: Escher accepts
only `metabolite`, `multimarker` and `midmarker` node types, a reaction is a
marker chain rather than a node, and arrowheads are derived from *signed*
stoichiometry, which v1 discarded.
"""

import json
import math

from .compound import NEVER_PRIMARY, SUPPRESSED, strip_compartment

# All distances are Escher canvas pixels.
STUB_RADIUS = 160.0        # cofactor distance from the reaction axis
MARKER_OFFSET = 36.0       # multimarker distance from the midmarker
STUB_SPREAD = math.radians(24.0)
STUB_ANGLE = math.radians(72.0)     # mostly perpendicular: stubs must not crowd the axis
PARALLEL_GAP = 150.0       # lateral offset between opposed reactions
LABEL_DX = 24.0
LABEL_DY = -12.0


def _unit(dx, dy):
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0, -1.0, 0.0
    return dx / length, dy / length, length


def _rotate(vx, vy, angle):
    c, s = math.cos(angle), math.sin(angle)
    return vx * c - vy * s, vx * s + vy * c


def _orthogonal_path(p0, p1):
    """Vertical-horizontal-vertical route; a single segment when x already
    matches, which is the case for every backbone edge Brandes-Koepf aligned."""
    (x0, y0), (x1, y1) = p0, p1
    if abs(x0 - x1) < 1.0:
        return [(x0, y0), (x1, y1)]
    if abs(y0 - y1) < 1.0:
        return [(x0, y0), (x1, y1)]
    ym = (y0 + y1) / 2.0
    return [(x0, y0), (x0, ym), (x1, ym), (x1, y1)]


def _offset_channel(path, shift):
    """Re-route a path down a parallel channel `shift` to one side.

    Used for opposed reaction pairs. It stays orthogonal: nudging the midpoint
    sideways and drawing three straight-line legs, which is the obvious thing
    to do, produces two diagonals and breaks the one property the whole layered
    pass exists to deliver.
    """
    start, end = path[0], path[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]

    if abs(dy) >= abs(dx):
        inset = min(70.0, abs(dy) / 4.0) if dy else 70.0
        y_a = start[1] + math.copysign(inset, dy or 1.0)
        y_b = end[1] - math.copysign(inset, dy or 1.0)
        lane = start[0] + shift
        return [start, (start[0], y_a), (lane, y_a), (lane, y_b), (end[0], y_b), end]

    inset = min(70.0, abs(dx) / 4.0) if dx else 70.0
    x_a = start[0] + math.copysign(inset, dx or 1.0)
    x_b = end[0] - math.copysign(inset, dx or 1.0)
    lane = start[1] + shift
    return [start, (x_a, start[1]), (x_a, lane), (x_b, lane), (x_b, end[1]), end]


def _arc_controls(start, end, centre):
    """Cubic controls approximating the circular arc start -> end about centre.

    Standard circular-arc-to-Bezier: controls sit on the tangents at each end,
    at distance 4/3 * tan(theta/4) * r.
    """
    ax, ay = start[0] - centre[0], start[1] - centre[1]
    bx, by = end[0] - centre[0], end[1] - centre[1]
    radius = (math.hypot(ax, ay) + math.hypot(bx, by)) / 2.0
    if radius < 1e-6:
        return None, None

    theta = math.atan2(ax * by - ay * bx, ax * bx + ay * by)
    if abs(theta) < 1e-6:
        return None, None
    k = 4.0 / 3.0 * math.tan(theta / 4.0)

    return ({"x": start[0] - k * ay, "y": start[1] + k * ax},
            {"x": end[0] + k * by, "y": end[1] - k * bx})


def _arc_points(start, end, centre, along=0.0):
    """Marker positions and local tangent on the circular arc start -> end."""
    cx, cy = centre
    r0 = math.hypot(start[0] - cx, start[1] - cy)
    r1 = math.hypot(end[0] - cx, end[1] - cy)
    radius = (r0 + r1) / 2.0
    if radius < 1e-6:
        return None

    a0 = math.atan2(start[1] - cy, start[0] - cx)
    a1 = math.atan2(end[1] - cy, end[0] - cx)
    sweep = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
    if abs(sweep) < 1e-6:
        return None

    span = abs(sweep) * radius
    shift = max(-0.25, min(0.25, along / span)) if span > 1e-6 else 0.0
    delta = min(MARKER_OFFSET / span, 0.2) if span > 1e-6 else 0.1

    def at(t):
        angle = a0 + sweep * t
        return (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

    def tangent(t):
        angle = a0 + sweep * t
        sign = 1.0 if sweep > 0 else -1.0
        return (-math.sin(angle) * sign, math.cos(angle) * sign)

    mid_t = 0.5 + shift
    return at(mid_t - delta), at(mid_t), at(mid_t + delta), tangent(mid_t)


def _route_polyline(points):
    """Orthogonalise a routed polyline leg by leg, dropping repeated points."""
    out = []
    for i in range(len(points) - 1):
        for point in _orthogonal_path(points[i], points[i + 1]):
            if not out or math.hypot(point[0] - out[-1][0], point[1] - out[-1][1]) > 1.0:
                out.append(point)
    return out or list(points)


def _walk(path, distance):
    """Point and local direction at `distance` along a polyline."""
    remaining = distance
    for i in range(len(path) - 1):
        (x0, y0), (x1, y1) = path[i], path[i + 1]
        ux, uy, length = _unit(x1 - x0, y1 - y0)
        if remaining <= length or i == len(path) - 2:
            t = max(0.0, min(length, remaining))
            return (x0 + ux * t, y0 + uy * t), (ux, uy)
        remaining -= length
    return path[-1], (0.0, 1.0)


def _path_length(path):
    return sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
               for i in range(len(path) - 1))


class EscherBuilder:
    def __init__(self, author="AutoLayout"):
        self.nodes = {}
        self.reactions = {}
        self.text_labels = {}
        self.author = author
        self._counter = 0

    def _new_id(self):
        self._counter += 1
        return str(self._counter)

    def add_metabolite(self, bigg_id, name, x, y, primary=True):
        node_id = self._new_id()
        self.nodes[node_id] = {
            "node_type": "metabolite",
            "x": float(x),
            "y": float(y),
            "bigg_id": bigg_id,
            "name": name or bigg_id,
            "label_x": float(x) + LABEL_DX,
            "label_y": float(y) + LABEL_DY,
            "node_is_primary": bool(primary),
        }
        return node_id

    def add_marker(self, kind, x, y):
        node_id = self._new_id()
        self.nodes[node_id] = {"node_type": kind, "x": float(x), "y": float(y)}
        return node_id

    def to_escher(self, map_name, description, canvas):
        return [
            {
                "map_name": map_name,
                "map_id": map_name.replace(" ", "_"),
                "map_description": description,
                "homepage": "https://escher.github.io",
                "schema": "https://escher.github.io/escher/jsonschema/1-0-0#",
            },
            {
                "reactions": self.reactions,
                "nodes": self.nodes,
                "text_labels": self.text_labels,
                "canvas": canvas,
            },
        ]


def _cofactor_side(axis_point, direction, occupied):
    """Put the cofactor fan on the emptier side of the reaction axis.

    Curated maps keep cofactors on a consistent side so the eye can ignore
    them; the only reason to flip is a neighbour already sitting there.
    """
    px, py = -direction[1], direction[0]
    scores = {}
    for side in (1, -1):
        probe = (axis_point[0] + side * px * STUB_RADIUS,
                 axis_point[1] + side * py * STUB_RADIUS)
        scores[side] = sum(
            1 for ox, oy in occupied
            if (ox - probe[0]) ** 2 + (oy - probe[1]) ** 2 < (1.4 * STUB_RADIUS) ** 2
        )
    return 1 if scores[1] <= scores[-1] else -1


def _stub_positions(anchor, direction, side, count, forward):
    """Fan `count` cofactors off one multimarker, all on the same side."""
    if count == 0:
        return []
    ux, uy = direction
    # The perpendicular must come from the *unflipped* axis: deriving it after
    # negating the direction flips the side too, which would scatter ATP and
    # ADP onto opposite banks instead of pairing them into the curved double
    # arrow every curated map uses.
    px, py = -uy, ux
    if not forward:
        ux, uy = -ux, -uy
    base_x = ux * math.cos(STUB_ANGLE) + side * px * math.sin(STUB_ANGLE)
    base_y = uy * math.cos(STUB_ANGLE) + side * py * math.sin(STUB_ANGLE)

    out = []
    for j in range(count):
        offset = (j - (count - 1) / 2.0) * STUB_SPREAD
        dx, dy = _rotate(base_x, base_y, offset)
        out.append((anchor[0] + dx * STUB_RADIUS, anchor[1] + dy * STUB_RADIUS))
    return out


def _bezier_to_stub(anchor, stub, direction, forward):
    """Cubic control points for an arc that leaves the axis tangentially.

    Modelled as a quadratic whose control point sits straight back along the
    reaction axis, then raised to a cubic -- that is the shape Escher itself
    draws for cofactor pairs, and it reads as "this leaves the arrow" rather
    than as another backbone edge.
    """
    ux, uy = direction
    if not forward:
        ux, uy = -ux, -uy
    span = math.hypot(stub[0] - anchor[0], stub[1] - anchor[1])
    qx = anchor[0] + ux * 0.85 * span
    qy = anchor[1] + uy * 0.85 * span
    b1 = {"x": anchor[0] + (2.0 / 3.0) * (qx - anchor[0]),
          "y": anchor[1] + (2.0 / 3.0) * (qy - anchor[1])}
    b2 = {"x": stub[0] + (2.0 / 3.0) * (qx - stub[0]),
          "y": stub[1] + (2.0 / 3.0) * (qy - stub[1])}
    return b1, b2


def build_escher_map(cgraph, pos, map_name, author="AutoLayout", description="",
                     routes=None):
    """Turn primary-metabolite coordinates into a complete Escher map.

    `routes` maps a reaction id to {"points": [...], "orthogonal": bool}: the
    polyline the layered pass computed for that edge, so multi-layer edges
    follow their dummy chain instead of cutting across intervening rows, and
    ring arcs stay as straight chords instead of being turned into staircases.
    """
    builder = EscherBuilder(author=author)
    metabolite_nodes = {}
    occupied = [pos[n] for n in pos]
    routes = routes or {}

    for met_id, (x, y) in pos.items():
        info = cgraph.metabolites.get(met_id, {})
        metabolite_nodes[met_id] = builder.add_metabolite(
            met_id, info.get("name", met_id), x, y, primary=True
        )

    for rid, rec in cgraph.reactions.items():
        sub, prod = rec.main_sub, rec.main_prod
        if sub not in pos and prod not in pos:
            continue
        _draw_reaction(builder, cgraph, rec, pos, metabolite_nodes, occupied,
                       routes.get(rid))

    _unify_primary(builder)
    _enforce_secondary(builder)
    apply_display_labels(builder)
    _place_labels(builder)
    canvas = _canvas(builder)
    _title(builder, canvas, map_name)
    _attribution(builder, canvas, author)
    canvas = _canvas(builder)          # title and credit need room too
    return builder.to_escher(map_name, description or f"Generated by {author}", canvas)


def _draw_reaction(builder, cgraph, rec, pos, metabolite_nodes, occupied, route=None):
    sub, prod = rec.main_sub, rec.main_prod

    arc_centre = route.get("arc_centre") if route else None
    if route is not None:
        points = route["points"]
        path = _route_polyline(points) if route.get("orthogonal", True) else list(points)
        start_node = metabolite_nodes[sub]
        end_node = metabolite_nodes[prod]
    elif sub in pos and prod in pos:
        path = _orthogonal_path(pos[sub], pos[prod])
        start_node = metabolite_nodes[sub]
        end_node = metabolite_nodes[prod]
    else:
        # Boundary exchange: a short stub leaving the drawing.
        anchored = sub if sub in pos else prod
        x, y = pos[anchored]
        outward = 1.0 if anchored == sub else -1.0
        path = [(x, y), (x, y + outward * 150.0)]
        start_node = metabolite_nodes[anchored]
        end_node = None

    # Opposed reactions (kinase/phosphatase, dehydrogenase/reductase) share an
    # axis. Routing them down two parallel channels draws a closed rectangle
    # with no nodes at its corners -- and a closed quadrilateral is the
    # universal notation for a cycle, so a reversible pair ends up claiming to
    # be one. Worst case, SUCDi/FRD7 drew a false ring right where the real TCA
    # ring belongs. They share one axis instead, each with its own midmarker
    # offset along it, and the arrowheads (which follow each reaction's own
    # direction) make the pair read as the standard double arrow.
    siblings = []
    if sub in pos and prod in pos and cgraph.D.has_edge(sub, prod):
        siblings = cgraph.D.edges[sub, prod]["rxns"]
    along = 0.0
    if len(siblings) > 1 and rec.rid in siblings:
        index = siblings.index(rec.rid)
        along = (index - (len(siblings) - 1) / 2.0) * MARKER_OFFSET * 2.4

    total = _path_length(path)
    centre = min(max(total / 2.0 + along, total * 0.25), total * 0.75)
    midpoint, direction = _walk(path, centre)
    offset = min(MARKER_OFFSET, total / 5.0)
    p_in, _ = _walk(path, centre - offset)
    p_out, _ = _walk(path, centre + offset)

    arc_mid = arc_in = arc_out = None
    if arc_centre is not None and len(path) == 2:
        arc = _arc_points(path[0], path[1], arc_centre, along)
        if arc is not None:
            arc_in, arc_mid, arc_out, direction = arc
            midpoint, p_in, p_out = arc_mid, arc_in, arc_out
        else:
            arc_centre = None

    mid_id = builder.add_marker("midmarker", *midpoint)
    in_id = builder.add_marker("multimarker", *p_in)
    out_id = builder.add_marker("multimarker", *p_out)

    segments = {}
    seq = [0]

    def segment(a, b, b1=None, b2=None):
        seq[0] += 1
        segments[f"{rec.rid}_s{seq[0]}"] = {
            "from_node_id": a, "to_node_id": b, "b1": b1, "b2": b2,
        }

    # Backbone chain, with routing bends as multimarkers. Bends have to be
    # split by arc length around the midmarker: emitting them all before it
    # would chain a bend that lies past the midpoint back to the start, which
    # draws the edge as a diagonal across the layers it was routed around.
    arc = [0.0]
    for i in range(len(path) - 1):
        arc.append(arc[-1] + math.hypot(path[i + 1][0] - path[i][0],
                                        path[i + 1][1] - path[i][1]))
    interior = list(zip(path[1:-1], arc[1:-1]))
    before = [p for p, d in interior if d < centre - offset]
    after = [p for p, d in interior if d > centre + offset]

    if arc_centre is not None and end_node is not None and len(path) == 2:
        # Markers ride the arc, not the chord. Walking the two-point path puts
        # them on the straight line between the two ring members, i.e. inside
        # the circle, and the four chain segments then draw three diagonal
        # chords per arc -- which is both wrong and what was dragging the
        # orthogonality metric down on every ring.
        for marker_id, point in ((mid_id, arc_mid), (in_id, arc_in), (out_id, arc_out)):
            builder.nodes[marker_id]["x"], builder.nodes[marker_id]["y"] = point
        chain = ((start_node, in_id, path[0], arc_in),
                 (in_id, mid_id, arc_in, arc_mid),
                 (mid_id, out_id, arc_mid, arc_out),
                 (out_id, end_node, arc_out, path[1]))
        for a_id, b_id, a_pt, b_pt in chain:
            b1, b2 = _arc_controls(a_pt, b_pt, arc_centre)
            segment(a_id, b_id, b1, b2)
    else:
        upstream = start_node
        for point in before:
            bend = builder.add_marker("multimarker", *point)
            segment(upstream, bend)
            upstream = bend
        segment(upstream, in_id)
        segment(in_id, mid_id)
        segment(mid_id, out_id)

        if end_node is not None:
            downstream = out_id
            for point in after:
                bend = builder.add_marker("multimarker", *point)
                segment(downstream, bend)
                downstream = bend
            segment(downstream, end_node)

    # Cofactor fans, both sides of the axis chosen once per reaction.
    side = _cofactor_side(midpoint, direction, occupied)
    metabolite_entries = []

    for met_id, forward, anchor_id, anchor_pt in (
        (rec.consumed, False, in_id, p_in),
        (rec.produced, True, out_id, p_out),
    ):
        # Protons and water are never drawn. They take part in most reactions
        # and say nothing when they do; they were roughly 30% of every disc.
        met_id = [m for m in met_id if strip_compartment(m) not in SUPPRESSED]
        stubs = _stub_positions(anchor_pt, direction, side, len(met_id), forward)
        for met, point in zip(met_id, stubs):
            info = cgraph.metabolites.get(met, {})
            node_id = builder.add_metabolite(
                met, info.get("name", met), point[0], point[1], primary=False
            )
            occupied.append(point)
            b1, b2 = _bezier_to_stub(anchor_pt, point, direction, forward)
            if forward:
                segment(anchor_id, node_id, b1, b2)
            else:
                segment(node_id, anchor_id, b1, b2)

    for met, coefficient in rec.stoichiometry.items():
        metabolite_entries.append({"bigg_id": met, "coefficient": float(coefficient)})

    builder.reactions[rec.rid] = {
        "name": rec.name,
        "bigg_id": rec.rid,
        "reversibility": bool(rec.reversible),
        "label_x": float(midpoint[0]) + LABEL_DX,
        "label_y": float(midpoint[1]) + LABEL_DY,
        "gene_reaction_rule": rec.genes,
        "genes": [],
        "metabolites": metabolite_entries,
        "segments": segments,
        "_anchor": (float(midpoint[0]), float(midpoint[1])),
    }


# Curated maps do not print database identifiers. They print short
# conventional names, and they do not print the compartment unless the same
# compound appears in two of them. Switching to the model's full `name` field
# instead ("D-Glucose 6-phosphate") is the obvious move and the wrong one: it
# multiplies label width by four and the collision count explodes.
_ABBREVIATIONS = {
    "accoa": "AcCoA", "succoa": "SucCoA", "coa": "CoA", "acon-c": "cis-Acon",
    "glx": "Glyox", "akg": "aKG", "oaa": "OAA", "pep": "PEP", "pyr": "Pyr",
    "cit": "Cit", "icit": "IsoCit", "succ": "Succ", "fum": "Fum",
    "mal-l": "Mal", "asp-l": "Asp", "glu-l": "Glu", "gln-l": "Gln",
    "ala-l": "Ala", "ser-l": "Ser", "thr-l": "Thr", "lac-d": "Lac",
    "glc-d": "Glc", "g6p": "G6P", "f6p": "F6P", "fdp": "FBP", "dhap": "DHAP",
    "g3p": "G3P", "13dpg": "1,3BPG", "3pg": "3PG", "2pg": "2PG",
    "6pgl": "6PGL", "6pgc": "6PGC", "ru5p-d": "Ru5P", "xu5p-d": "Xu5P",
    "r5p": "R5P", "s7p": "S7P", "e4p": "E4P", "etoh": "EtOH", "acald": "AcAld",
    "actp": "AcP", "ac": "Ac", "for": "Form", "q8": "Q8", "q8h2": "Q8H2",
}
_COMPARTMENT_NAMES = {"c": "c", "e": "e", "p": "p", "m": "m", "x": "x",
                      "r": "r", "l": "l", "n": "n", "g": "g", "v": "v"}


def display_label(bigg_id, ambiguous=()):
    """Short, human-facing label for a metabolite id.

    `mal__L_c` -> `Mal`, or `Mal[c]` when malate also appears in another
    compartment on the same map. BiGG's double-underscore escaping never
    reaches the figure.
    """
    base = strip_compartment(bigg_id)
    compartment = bigg_id[len(base) + 1:] if len(bigg_id) > len(base) else ""
    key = base.replace("__", "-").replace("_", "-").lower()
    text = _ABBREVIATIONS.get(key, base.replace("__", "-").replace("_", "-"))
    if key in ambiguous and compartment in _COMPARTMENT_NAMES:
        text = f"{text}[{compartment}]"
    return text


def ambiguous_bases(builder):
    """Metabolite bases that appear in more than one compartment on this map."""
    seen = {}
    for node in builder.nodes.values():
        if node["node_type"] != "metabolite":
            continue
        identifier = node["bigg_id"]
        base = strip_compartment(identifier)
        key = base.replace("__", "-").replace("_", "-").lower()
        seen.setdefault(key, set()).add(identifier[len(base) + 1:])
    return {key for key, comps in seen.items() if len(comps) > 1}


def apply_display_labels(builder):
    ambiguous = ambiguous_bases(builder)
    for node in builder.nodes.values():
        if node["node_type"] == "metabolite":
            node["label_text"] = display_label(node["bigg_id"], ambiguous)
    for reaction in builder.reactions.values():
        reaction["label_text"] = reaction["bigg_id"]


def _unify_primary(builder):
    """One compound, one visual code, per figure.

    Primary/secondary is decided per reaction, so acetyl-CoA is the terminal
    intermediate of glycolysis and a grey stub on citrate synthase -- the
    carbon input to the TCA cycle drawn with the same weight as water. If a
    compound is primary anywhere on a map it is primary everywhere on it.
    """
    primary = {n["bigg_id"] for n in builder.nodes.values()
               if n["node_type"] == "metabolite" and n.get("node_is_primary")}
    for node in builder.nodes.values():
        if node["node_type"] == "metabolite" and node["bigg_id"] in primary:
            node["node_is_primary"] = True


def _enforce_secondary(builder):
    """Currency is never drawn as a primary node.

    Main-pair selection is per reaction, so ATP can win the pair in one
    reaction and be a side metabolite in the next -- and the same compound then
    appears in one figure as both a large coloured intermediate and a small grey
    stub. A big orange ATP reads as a pathway intermediate. The rule is global
    per figure, with one escape: if demoting everything would leave no primary
    node at all, the map really is about that chemistry (water transport, ion
    exchange) and the demotion is skipped.
    """
    metabolites = [n for n in builder.nodes.values() if n["node_type"] == "metabolite"]
    demote = [n for n in metabolites
              if n.get("node_is_primary") and strip_compartment(n["bigg_id"]) in NEVER_PRIMARY]
    if not demote:
        return
    survivors = sum(1 for n in metabolites
                    if n.get("node_is_primary") and n not in demote)
    if survivors == 0:
        return
    for node in demote:
        node["node_is_primary"] = False


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------

# Label geometry.
#
# Escher renders a label at `font_size_base` (or the global `gene_font_size`,
# default 18) times a per-kind factor set in Draw.js: 1.1 for a metabolite
# label, 1.5 for a reaction label. Emitting `font_size_base` per label is what
# lets the collision model here be truthful -- the box this code reserves is
# the box the renderer actually draws.
ESCHER_DEFAULT_FONT_BASE = 18.0
METABOLITE_FONT_FACTOR = 1.1
REACTION_FONT_FACTOR = 1.5

# Helvetica/Arial lowercase-and-digits averages a little under 0.6 em.
CHAR_WIDTH_RATIO = 0.58
LINE_HEIGHT_RATIO = 1.25

# Shrinking beats exile. A label two node-widths from its node annotates
# nothing, so when the full size does not fit, step down the ladder rather than
# widen the search. 9 is the floor: below that the text stops being legible at
# the zoom level where a whole pathway is on screen.
FONT_LADDER = (18.0, 15.0, 12.5, 10.5, 9.0)
MIN_FONT_BASE = FONT_LADDER[-1]

_PRIMARY_CLEARANCE = 34.0      # node radius 30 plus a hair
_SECONDARY_CLEARANCE = 20.0
_MARKER_CLEARANCE = 13.0
_SEGMENT_CLEARANCE = 13.0
_STUB_CLEARANCE = 7.0
# Labels may sit closer to each other than to the drawing: two adjacent labels
# read fine with a hairline between them, and the alternative is pushing one of
# them away from the thing it names. Text does not fill its own box -- ascender
# and descender space is mostly empty -- so a few pixels of box overlap is not
# visible overlap. `metrics` applies the same tolerance, otherwise it reports a
# failure for something deliberately allowed here.
LABEL_OVERLAP_TOLERANCE = 3.0


def label_box(text, font_base, factor):
    """(width, height) of a label as the renderer will draw it."""
    size = font_base * factor
    return (max(len(str(text)), 1) * size * CHAR_WIDTH_RATIO,
            size * LINE_HEIGHT_RATIO)


# Candidate label-box centres relative to the anchor, nearest first. The
# candidates ring the anchor on an ellipse inflated by the label's own half
# extent, so every offset puts the box just clear of the anchor whichever way
# it points. Three rings, all inside one layer pitch: past that the label stops
# reading as a caption for this node, and the font ladder is the better lever.
# Sideways placements come first -- a label beside a node reads better than one
# above or below it -- but the ring is dense enough (12 directions) that a
# crowded region still has somewhere to go.
_CANDIDATE_ARMS = 12
_CANDIDATE_RINGS = (0.0, 26.0, 70.0)


def _candidates(h, v, gap):
    out = []
    for ring in _CANDIDATE_RINGS:
        arms = []
        for index in range(_CANDIDATE_ARMS):
            angle = 2.0 * math.pi * index / _CANDIDATE_ARMS
            cos, sin = math.cos(angle), math.sin(angle)
            arms.append((abs(sin), (h + gap + ring) * cos, (v + gap + ring) * sin))
        arms.sort(key=lambda a: a[0])       # horizontal offsets first
        out.extend((dx, dy) for _, dx, dy in arms)
    return out


class _Obstacles:
    """Uniform-grid index over nodes, edges and already-placed labels.

    Label placement is quadratic without it -- a genome-scale map has thousands
    of labels and tens of thousands of segments -- and the whole point of the
    pass is that it can afford to check every obstacle rather than guessing.
    """

    CELL = 220.0

    def __init__(self):
        self.cells = {}

    def _key(self, x, y):
        return (int(x // self.CELL), int(y // self.CELL))

    def _insert(self, x, y, item):
        self.cells.setdefault(self._key(x, y), []).append(item)

    def add_point(self, x, y, clearance):
        self._insert(x, y, ("point", x, y, clearance))

    def add_segment(self, x1, y1, x2, y2, clearance):
        item = ("segment", (x1, y1, x2, y2), clearance)
        steps = max(1, int(math.hypot(x2 - x1, y2 - y1) / self.CELL) + 1)
        seen = set()
        for i in range(steps + 1):
            t = i / steps
            key = self._key(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
            if key not in seen:
                seen.add(key)
                self.cells.setdefault(key, []).append(item)

    def add_box(self, left, top, right, bottom):
        item = ("box", left, top, right, bottom)
        for cx in range(int(left // self.CELL), int(right // self.CELL) + 1):
            for cy in range(int(top // self.CELL), int(bottom // self.CELL) + 1):
                self.cells.setdefault((cx, cy), []).append(item)

    def hits(self, left, top, right, bottom):
        checked = set()
        for cx in range(int((left - self.CELL) // self.CELL),
                        int((right + self.CELL) // self.CELL) + 1):
            for cy in range(int((top - self.CELL) // self.CELL),
                            int((bottom + self.CELL) // self.CELL) + 1):
                for item in self.cells.get((cx, cy), ()):
                    if id(item) in checked:
                        continue
                    checked.add(id(item))
                    if item[0] == "point":
                        _, x, y, clearance = item
                        if (left - clearance < x < right + clearance
                                and top - clearance < y < bottom + clearance):
                            return True
                    elif item[0] == "segment":
                        _, (x1, y1, x2, y2), clearance = item
                        if _segment_hits_box(x1, y1, x2, y2,
                                             left - clearance, top - clearance,
                                             right + clearance, bottom + clearance):
                            return True
                    else:
                        _, bl, bt, br, bb = item
                        if not (right < bl or left > br or bottom < bt or top > bb):
                            return True
        return False


def _segment_hits_box(x1, y1, x2, y2, left, top, right, bottom):
    """Liang-Barsky clip test."""
    if max(x1, x2) < left or min(x1, x2) > right:
        return False
    if max(y1, y2) < top or min(y1, y2) > bottom:
        return False
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x1 - left), (dx, right - x1), (-dy, y1 - top), (dy, bottom - y1)):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return False
            t0 = max(t0, r)
        else:
            if r < t0:
                return False
            t1 = min(t1, r)
    return t0 <= t1


def _place_labels(builder):
    """Place every label near the thing it names, shrinking it if it must.

    Nodes and edges are final by this point; labels are the only thing allowed
    to move. `constraints.md` asks that a label never overlap a node, an edge,
    or another label, and the previous version satisfied that by widening the
    search until something was free -- which left labels hundreds of pixels
    from their node, annotating nothing. Distance is capped instead, and the
    font ladder absorbs the pressure. Only if a label cannot fit anywhere even
    at the minimum size is an overlap accepted, and `metrics.score` counts it.

    Longest labels go first: they are both hardest to fit and most damaging
    when they land on something.
    """
    obstacles = _Obstacles()
    for node in builder.nodes.values():
        if node["node_type"] == "metabolite":
            clearance = (_PRIMARY_CLEARANCE if node.get("node_is_primary", True)
                         else _SECONDARY_CLEARANCE)
        else:
            clearance = _MARKER_CLEARANCE
        obstacles.add_point(node["x"], node["y"], clearance)
    for reaction in builder.reactions.values():
        for segment in reaction["segments"].values():
            a = builder.nodes.get(segment["from_node_id"])
            b = builder.nodes.get(segment["to_node_id"])
            if a is None or b is None:
                continue
            clearance = _STUB_CLEARANCE if segment.get("b1") else _SEGMENT_CLEARANCE
            obstacles.add_segment(a["x"], a["y"], b["x"], b["y"], clearance)

    targets = []
    for node in builder.nodes.values():
        if node["node_type"] == "metabolite":
            targets.append((node, node.get("label_text", node["bigg_id"]),
                            node["x"], node["y"], METABOLITE_FONT_FACTOR))
    for reaction in builder.reactions.values():
        anchor_x, anchor_y = reaction.pop("_anchor")
        targets.append((reaction, reaction["bigg_id"], anchor_x, anchor_y,
                        REACTION_FONT_FACTOR))
    targets.sort(key=lambda t: -len(t[1]))

    crowded = 0
    for holder, text, anchor_x, anchor_y, factor in targets:
        placed = None
        fallback = None

        # Nearest placement wins over largest font: a caption that has drifted
        # away from its node is a worse failure than a slightly smaller one.
        for font_base in FONT_LADDER:
            width, height = label_box(text, font_base, factor)
            half_w, half_v = width / 2.0, height / 2.0
            gap = max(6.0, 0.35 * height)
            for dx, dy in _candidates(half_w, half_v, gap):
                cx, cy = anchor_x + dx, anchor_y + dy
                if fallback is None:
                    fallback = (cx, cy, half_w, half_v, font_base)
                if not obstacles.hits(cx - half_w, cy - half_v,
                                      cx + half_w, cy + half_v):
                    placed = (cx, cy, half_w, half_v, font_base)
                    break
            if placed:
                break

        if placed is None:
            placed = fallback
            crowded += 1

        cx, cy, half_w, half_v, font_base = placed
        holder["label_x"] = cx - half_w
        holder["label_y"] = cy
        if font_base != ESCHER_DEFAULT_FONT_BASE:
            holder["font_size_base"] = font_base
        obstacles.add_box(cx - half_w + LABEL_OVERLAP_TOLERANCE,
                          cy - half_v + LABEL_OVERLAP_TOLERANCE,
                          cx + half_w - LABEL_OVERLAP_TOLERANCE,
                          cy + half_v - LABEL_OVERLAP_TOLERANCE)

    return crowded


TEXT_LABEL_FONT_FACTOR = 3.0      # Draw.js scales a free text label by 3


def _canvas(builder, padding=None):
    """Canvas that contains every node and every label.

    Sizing from node coordinates alone clips whatever hangs past them -- which
    is every label, and most visibly the attribution line along the bottom.
    """
    xs, ys = [], []
    for node in builder.nodes.values():
        xs.append(node["x"])
        ys.append(node["y"])
        if "label_x" in node:
            size = node.get("font_size_base", ESCHER_DEFAULT_FONT_BASE) * METABOLITE_FONT_FACTOR
            xs.extend((node["label_x"],
                       node["label_x"] + len(node["bigg_id"]) * size * CHAR_WIDTH_RATIO))
            ys.extend((node["label_y"] - size * LINE_HEIGHT_RATIO / 2.0,
                       node["label_y"] + size * LINE_HEIGHT_RATIO / 2.0))
    for reaction in builder.reactions.values():
        size = reaction.get("font_size_base", ESCHER_DEFAULT_FONT_BASE) * REACTION_FONT_FACTOR
        xs.extend((reaction["label_x"],
                   reaction["label_x"] + len(reaction["bigg_id"]) * size * CHAR_WIDTH_RATIO))
        ys.extend((reaction["label_y"] - size * LINE_HEIGHT_RATIO / 2.0,
                   reaction["label_y"] + size * LINE_HEIGHT_RATIO / 2.0))
    for label in builder.text_labels.values():
        size = label.get("font_size_base", ESCHER_DEFAULT_FONT_BASE) * TEXT_LABEL_FONT_FACTOR
        xs.extend((label["x"],
                   label["x"] + len(str(label.get("text", ""))) * size * CHAR_WIDTH_RATIO))
        ys.extend((label["y"] - size * LINE_HEIGHT_RATIO / 2.0,
                   label["y"] + size * LINE_HEIGHT_RATIO / 2.0))

    if not xs:
        return {"x": 0.0, "y": 0.0, "width": 1000.0, "height": 1000.0}
    if padding is None:
        # Proportional, not fixed: a flat 400 on each side is a sensible margin
        # for a pathway map and most of the canvas for a two-node cluster.
        padding = max(150.0, 0.06 * max(max(xs) - min(xs), max(ys) - min(ys)))
    return {
        "x": min(xs) - padding,
        "y": min(ys) - padding,
        "width": (max(xs) - min(xs)) + 2 * padding,
        "height": (max(ys) - min(ys)) + 2 * padding,
    }


TITLE_FONT_BASE = 26.0             # map title
ATTRIBUTION_FONT_BASE = 8.0        # a credit line, not a title


def _title(builder, canvas, text):
    """Name the figure.

    Every emitted map previously carried exactly one text label and it was the
    author credit -- a figure with a byline and no title, which is precisely
    backwards. Title, region caption, reaction, metabolite gives the three-level
    size hierarchy a reader needs to enter the figure.
    """
    if not text:
        return
    builder.text_labels["map_title"] = {
        "x": canvas["x"] + 60.0,
        "y": canvas["y"] + 120.0,
        "text": text,
        "font_size_base": TITLE_FONT_BASE,
    }


def _attribution(builder, canvas, author):
    # Free text labels render at 3x their base size, so the default 18 would
    # make a 38-character credit line wider than a single-pathway map and drag
    # the canvas out with it.
    builder.text_labels["attribution"] = {
        "x": canvas["x"] + 60.0,
        "y": canvas["y"] + canvas["height"] - 60.0,
        "text": f"Created by {author}",
        "font_size_base": ATTRIBUTION_FONT_BASE,
    }


def save(escher_map, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(escher_map, handle, indent=2)
