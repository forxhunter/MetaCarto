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

from . import identity
from .compound import NEVER_PRIMARY, SUPPRESSED, strip_compartment

# All distances are Escher canvas pixels.
STUB_RADIUS = 160.0        # cofactor distance from the reaction axis
MARKER_OFFSET = 36.0       # multimarker distance from the midmarker
STUB_SPREAD = math.radians(24.0)
STUB_ANGLE = math.radians(72.0)     # mostly perpendicular: stubs must not crowd the axis
PARALLEL_GAP = 150.0       # lateral offset between opposed reactions
LANE_GAP = 110.0           # between the lanes of reactions joining one metabolite pair
LANE_FAN = 60.0            # how much further along its axis each inner lane steps out
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


def _bow_centre(start, end, sagitta):
    """Centre of the circle through `start` and `end` whose arc bulges
    `sagitta` units to the left of start -> end (negative: to the right)."""
    ux, uy, length = _unit(end[0] - start[0], end[1] - start[1])
    if length < 1e-6 or abs(sagitta) < 1e-6:
        return None
    nx, ny = -uy, ux
    radius = (length * length / 4.0 + sagitta * sagitta) / (2.0 * abs(sagitta))
    mx, my = (start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0
    back = sagitta - math.copysign(radius, sagitta)
    return (mx + nx * back, my + ny * back)


def _signed_sagitta(start, end, centre):
    """How far the arc about `centre` from start to end bulges left of the chord."""
    ux, uy, length = _unit(end[0] - start[0], end[1] - start[1])
    nx, ny = -uy, ux
    mx, my = (start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0
    radius = (math.hypot(start[0] - centre[0], start[1] - centre[1])
              + math.hypot(end[0] - centre[0], end[1] - centre[1])) / 2.0
    towards = (mx - centre[0]) * nx + (my - centre[1]) * ny
    # The short arc lies on the side of the chord away from the centre.
    return math.copysign(radius - abs(towards), towards) if towards else radius


def _offset_polyline(path, shift, rank=0):
    """The orthogonal polyline `path` moved `shift` units to its left.

    Its two ends stay on the metabolites they join, so every lane needs a
    step sideways into its track, and no two steps may share a line. The
    outermost lane on each side (`rank` 0) -- every lane when two or three
    reactions join the pair, by far the common case -- steps out at a right
    angle at the metabolite. Inner lanes fan out diagonally, each `LANE_FAN`
    further along than the lane outside it, so their angles all differ.

    Fanning every lane out diagonally kept lanes apart but cost a fifth of the
    maps their orthogonality; following the axis before stepping out kept it
    orthogonal but put both inner lanes on the same stretch of axis.
    """
    normals, units, lengths = [], [], []
    for i in range(len(path) - 1):
        ux, uy, length = _unit(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        normals.append((-uy, ux))
        units.append((ux, uy))
        lengths.append(length)

    def stepped(point, unit, normal, inset, sign):
        # The outermost lane steps out at a right angle at the metabolite.
        # An inner lane cannot do the same without running along its outer
        # neighbour's step, and cannot follow the axis first either: the
        # inner lane on the other side would follow the same stretch of axis.
        # So inner lanes, which only exist with four or more reactions between
        # one pair, fan out at their own angle instead.
        lane = (point[0] + sign * unit[0] * inset + shift * normal[0],
                point[1] + sign * unit[1] * inset + shift * normal[1])
        return [lane]

    head_inset = min(rank * LANE_FAN, lengths[0] / 3.0)
    tail_inset = min(rank * LANE_FAN, lengths[-1] / 3.0)
    out = [path[0]] + stepped(path[0], units[0], normals[0], head_inset, 1.0)
    for i in range(1, len(path) - 1):
        n1, n2 = normals[i - 1], normals[i]
        if abs(n1[0] - n2[0]) < 1e-6 and abs(n1[1] - n2[1]) < 1e-6:
            continue
        out.append((path[i][0] + shift * (n1[0] + n2[0]), path[i][1] + shift * (n1[1] + n2[1])))
    out += list(reversed(stepped(path[-1], units[-1], normals[-1], tail_inset, -1.0)))
    out.append(path[-1])
    return out


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
                     routes=None, rings=()):
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

    # Sorted, not in `pos` order. Node ids come from a counter, so they follow
    # insertion order, and `pos` inherits its order from the layered pass --
    # which passes through a set somewhere upstream and therefore varies with
    # PYTHONHASHSEED between processes. The drawing was identical either way
    # (same coordinates, same pairs, same metrics) but the emitted JSON was not
    # byte-identical across runs, which is weaker than the determinism this
    # project claims. Sorting here pins the numbering wherever the upstream
    # order came from.
    for met_id, (x, y) in sorted(pos.items()):
        info = cgraph.metabolites.get(met_id, {})
        node_id = builder.add_metabolite(
            met_id, info.get("name", met_id), x, y, primary=True
        )
        # Placed by the layered pass, which already guaranteed its separation.
        builder.nodes[node_id]["_anchored"] = True
        metabolite_nodes[met_id] = node_id

    # Ring members say which ring they are on. A ring's middle is empty by
    # design -- a curated TCA cycle is a circle with nothing inside -- and
    # `metrics.blank_space` reads this to count that middle as drawing rather
    # than as a hole in the page. Outside the Escher schema, like label_text.
    for index, ring in enumerate(rings):
        for met_id in ring:
            if met_id in metabolite_nodes:
                builder.nodes[metabolite_nodes[met_id]]["ring"] = f"ring{index}"

    for rid, rec in cgraph.reactions.items():
        sub, prod = rec.main_sub, rec.main_prod
        if sub not in pos and prod not in pos:
            continue
        _draw_reaction(builder, cgraph, rec, pos, metabolite_nodes, occupied,
                       routes.get(rid))

    _nudge_overlaps(builder)
    _slide_markers(builder)
    _unify_primary(builder)
    _enforce_secondary(builder)
    _separate_nodes(builder)
    # Again, now that every node is final. Separation moves cofactor stubs,
    # and promoted duplicates, onto lines the first pass had already cleared;
    # over four models this second pass takes edges through unrelated
    # metabolites from 41 to 37 per thousand reactions, through unrelated
    # cofactors from 61 to 49.
    _nudge_overlaps(builder)
    _slide_markers(builder)
    apply_display_labels(builder)
    _place_labels(builder)
    canvas = _canvas(builder)
    _title(builder, canvas, map_name)
    _attribution(builder, canvas, author)
    canvas = _canvas(builder)          # title and credit need room too
    for node in builder.nodes.values():
        node.pop("_anchored", None)
    return builder.to_escher(map_name, description or f"Generated by {author}", canvas)


NUDGE = 30.0               # spacing between edges pulled off a shared line
NUDGE_PASSES = 8
NODE_CLEARANCE = 14.0      # gap kept between an edge and an unrelated node


def _runs(builder):
    """Every reaction's straight edges merged into maximal collinear runs.

    A run is the unit an edge moves by: shifting one segment of a straight
    stretch alone would leave its neighbour diagonal, while shifting the whole
    stretch only lengthens or shortens the perpendicular legs at its ends.
    Returns [(rid, orientation, coordinate, lo, hi, node ids)].
    """
    nodes = builder.nodes
    out = []
    for rid, reaction in builder.reactions.items():
        adjacency = {}
        for segment in reaction["segments"].values():
            if segment.get("b1") or segment.get("b2"):
                continue
            a, b = segment["from_node_id"], segment["to_node_id"]
            na, nb = nodes[a], nodes[b]
            if abs(na["y"] - nb["y"]) < 1.0 and abs(na["x"] - nb["x"]) >= 1.0:
                orient = "h"
            elif abs(na["x"] - nb["x"]) < 1.0 and abs(na["y"] - nb["y"]) >= 1.0:
                orient = "v"
            else:
                continue
            adjacency.setdefault((a, orient), []).append(b)
            adjacency.setdefault((b, orient), []).append(a)
        seen = set()
        for (start, orient), _ in adjacency.items():
            if (start, orient) in seen:
                continue
            # Walk the connected same-orientation chain through `start`.
            chain, stack = set(), [start]
            while stack:
                node = stack.pop()
                if node in chain:
                    continue
                chain.add(node)
                seen.add((node, orient))
                stack.extend(adjacency.get((node, orient), ()))
            coord = nodes[start]["y"] if orient == "h" else nodes[start]["x"]
            along = [nodes[n]["x"] if orient == "h" else nodes[n]["y"] for n in chain]
            out.append((rid, orient, coord, min(along), max(along), chain))
    return out


def _movable(builder, chain):
    return all(builder.nodes[n]["node_type"] != "metabolite" for n in chain)


def _shift_run(builder, rid, orient, chain, shift, moved):
    """Move a run `shift` across its own line; return False if it cannot.

    A run that ends on a metabolite gets a new bend a short way out from it,
    so the metabolite stays put and the rest of the run takes the new track.
    """
    nodes = builder.nodes
    fixed = [n for n in chain if nodes[n]["node_type"] == "metabolite"]
    if len(fixed) > 1:
        return False
    if fixed:
        # The metabolite stays put; the run leaves it through a right-angle
        # step at the metabolite itself, then takes its new track.
        m = fixed[0]
        reaction = builder.reactions[rid]
        for sid, segment in list(reaction["segments"].items()):
            if segment.get("b1") or segment.get("b2"):
                continue
            a, b = segment["from_node_id"], segment["to_node_id"]
            if m not in (a, b):
                continue
            other = b if a == m else a
            if other not in chain:
                continue
            mx, my = nodes[m]["x"], nodes[m]["y"]
            jog = builder.add_marker("multimarker", mx, my)
            del reaction["segments"][sid]
            pairs = ((a, jog), (jog, b))
            for k, (p, q) in enumerate(pairs):
                reaction["segments"][f"{sid}_n{k}"] = {
                    "from_node_id": p, "to_node_id": q, "b1": None, "b2": None}
            chain = (chain - {m}) | {jog}
            break
    for node_id in chain:
        dx, dy = (0.0, shift) if orient == "h" else (shift, 0.0)
        nodes[node_id]["x"] += dx
        nodes[node_id]["y"] += dy
        moved[node_id] = (moved.get(node_id, (0.0, 0.0))[0] + dx,
                          moved.get(node_id, (0.0, 0.0))[1] + dy)
    return True


def _nudge_overlaps(builder):
    """Pull apart edges of unrelated reactions drawn on top of each other.

    Orthogonal routing sends long edges down shared channels, so two
    reactions with nothing in common can end up on one line, or one edge can
    run straight through another reaction's markers or metabolite and appear
    to join it. The remedy from orthogonal connector routing is nudging: move
    each offending straight run a little to the side.

      collinear  unrelated runs sharing a stretch of one line each get a
                 track of their own, NUDGE apart;
      through    a run passing over an unrelated node, or over any other
                 reaction's midmarker, moves clear of it.

    Reactions that share a metabolite are left on one track. Edges converging
    on the compound they share are how the sharing is drawn, and they still
    read as two edges because they part where the sharing ends; splitting
    them cost a tenth of the drawing's orthogonality for nothing.
    """
    nodes = builder.nodes
    touches, owners = {}, {}
    for rid, reaction in builder.reactions.items():
        for segment in reaction["segments"].values():
            for end in (segment["from_node_id"], segment["to_node_id"]):
                owners.setdefault(end, set()).add(rid)
                if nodes[end]["node_type"] == "metabolite":
                    touches.setdefault(rid, set()).add(end)

    def related(r1, r2):
        return r1 == r2 or bool(touches.get(r1, set()) & touches.get(r2, set()))

    def radius(node):
        if node["node_type"] == "metabolite":
            return PRIMARY_RADIUS if node.get("node_is_primary", True) else SECONDARY_RADIUS
        return 11.0 if node["node_type"] == "midmarker" else 6.0

    for _ in range(NUDGE_PASSES):
        moved = {}
        runs = _runs(builder)

        # Collinear: group runs on one line whose spans overlap.
        lines = {}
        for run in runs:
            rid, orient, coord, lo, hi, chain = run
            lines.setdefault((orient, round(coord / 4.0)), []).append(run)
        for spans in lines.values():
            spans.sort(key=lambda r: r[3])
            groups, current, reach = [], [], None
            for run in spans:
                if current and run[3] < reach - 12.0:
                    current.append(run)
                    reach = max(reach, run[4])
                else:
                    if current:
                        groups.append(current)
                    current, reach = [run], run[4]
            if current:
                groups.append(current)
            for group in groups:
                # One track per set of mutually related reactions.
                tracks = []
                for run in sorted(group, key=lambda r: r[0]):
                    for track in tracks:
                        if any(related(run[0], other[0]) for other in track):
                            track.append(run)
                            break
                    else:
                        tracks.append([run])
                if len(tracks) < 2:
                    continue
                for k, track in enumerate(tracks):
                    shift = (k - (len(tracks) - 1) / 2.0) * NUDGE
                    if not shift:
                        continue
                    for rid, orient, _, _, _, chain in track:
                        if not any(n in moved for n in chain):
                            _shift_run(builder, rid, orient, chain, shift, moved)

        # Through: a run passing over a node of an unrelated reaction.
        cell = 120.0
        grid = {}
        for node_id, node in nodes.items():
            grid.setdefault((int(node["x"] // cell), int(node["y"] // cell)), []).append(node_id)
        for rid, orient, coord, lo, hi, chain in _runs(builder):
            if any(n in moved for n in chain):
                continue
            worst = None
            steps = int((hi - lo) // cell) + 2
            for s in range(steps + 1):
                t = lo + (hi - lo) * min(1.0, s / steps)
                px, py = (t, coord) if orient == "h" else (coord, t)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for other in grid.get((int(px // cell) + dx, int(py // cell) + dy), ()):
                            if other in chain:
                                continue
                            node = nodes[other]
                            # A midmarker is where a reaction is read -- its
                            # arrow, its label. Another reaction's line over it,
                            # or its bend on it, reads as one reaction even when
                            # the two do share a metabolite.
                            mid = node["node_type"] == "midmarker"
                            if mid:
                                if rid in owners.get(other, ()):
                                    continue
                            elif any(related(rid, r) for r in owners.get(other, ())):
                                continue
                            along = node["x"] if orient == "h" else node["y"]
                            across = node["y"] if orient == "h" else node["x"]
                            need = radius(node) + NODE_CLEARANCE
                            # Past the ends too: a node on a run's corner sits
                            # on the bend marker there, which is as much an
                            # overlap as one halfway along.
                            # Past the ends too, for a node a corner may not
                            # touch: a metabolite or midmarker on a run's bend
                            # is as much an overlap as one halfway along. Two
                            # bends touching is not; they are dots on lines.
                            margin = need if mid or node["node_type"] == "metabolite" else 0.0
                            if not lo - margin < along < hi + margin:
                                continue
                            if abs(across - coord) < need:
                                side = 1.0 if coord >= across else -1.0
                                shift = across + side * need - coord
                                if worst is None or abs(shift) > abs(worst):
                                    worst = shift
            if worst:
                _shift_run(builder, rid, orient, chain, worst, moved)

        if not moved:
            return
        # Curves anchored on a moved marker keep their shape.
        for reaction in builder.reactions.values():
            for segment in reaction["segments"].values():
                for end, control in (("from_node_id", "b1"), ("to_node_id", "b2")):
                    point = segment.get(control)
                    move = moved.get(segment[end])
                    if point and move:
                        point["x"] += move[0]
                        point["y"] += move[1]


SLIDE_STEP = 12.0
SLIDE_GAP = 4.0            # clearance between a moved marker and anything else


def _node_reach(node):
    if node["node_type"] == "metabolite":
        return PRIMARY_RADIUS if node.get("node_is_primary", True) else SECONDARY_RADIUS
    return 11.0 if node["node_type"] == "midmarker" else 6.0


def _slide_markers(builder):
    """Move a reaction's arrow off any node that is not its own.

    Nudging moves straight runs sideways, which cannot help a reaction drawn
    as one diagonal or curved stroke -- an inner sibling lane, a transport
    step between two columns -- whose midmarker happens to land on another
    reaction's metabolite. The reaction then reads as passing through that
    compound. Sliding the arrow (midmarker, its two multimarkers, and the
    cofactor fan hanging off them) along its own straight axis changes
    nothing else about the drawing and clears it.
    """
    nodes = builder.nodes
    owners, members = {}, {}
    for rid, reaction in builder.reactions.items():
        for segment in reaction["segments"].values():
            for end in (segment["from_node_id"], segment["to_node_id"]):
                owners.setdefault(end, set()).add(rid)
                members.setdefault(rid, set()).add(end)
    cell = 120.0
    grid = {}
    for node_id, node in nodes.items():
        grid.setdefault((int(node["x"] // cell), int(node["y"] // cell)), []).append(node_id)

    def clashes(rid, points, own):
        for (x, y), reach in points:
            cx, cy = int(x // cell), int(y // cell)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for other in grid.get((cx + dx, cy + dy), ()):
                        if other in own:
                            continue
                        node = nodes[other]
                        if math.hypot(node["x"] - x, node["y"] - y) < reach + _node_reach(node) + SLIDE_GAP:
                            return True
        return False

    for rid, reaction in builder.reactions.items():
        segments = list(reaction["segments"].values())
        mid = next((n for s in segments for n in (s["from_node_id"], s["to_node_id"])
                    if nodes[n]["node_type"] == "midmarker"), None)
        if mid is None:
            continue
        straight = {}
        for segment in segments:
            a, b = segment["from_node_id"], segment["to_node_id"]
            if not (segment.get("b1") or segment.get("b2")):
                straight.setdefault(a, []).append(b)
                straight.setdefault(b, []).append(a)
        flank = [n for n in straight.get(mid, ()) if nodes[n]["node_type"] == "multimarker"]
        if len(flank) != 2:
            continue
        outer = []
        for marker in flank:
            beyond = [n for n in straight.get(marker, ()) if n != mid]
            if len(beyond) != 1:
                break
            outer.append(beyond[0])
        if len(outer) != 2:
            continue
        p, q = nodes[outer[0]], nodes[outer[1]]
        ux, uy, length = _unit(q["x"] - p["x"], q["y"] - p["y"])
        if length < 1.0:
            continue
        body = [mid] + flank
        # The axis has to be straight through all five points to slide along it.
        if any(abs((nodes[n]["x"] - p["x"]) * uy - (nodes[n]["y"] - p["y"]) * ux) > 1.0 for n in body):
            continue
        # Cofactor stubs hang off the two multimarkers and move with them.
        stubs = sorted({s["from_node_id"] if s["to_node_id"] in flank else s["to_node_id"]
                        for s in segments
                        if (s["from_node_id"] in flank) != (s["to_node_id"] in flank)})
        stubs = [n for n in stubs if owners.get(n) == {rid}
                 and nodes[n]["node_type"] == "metabolite"
                 and not nodes[n].get("node_is_primary", True)]
        moving = body + stubs
        mine = members[rid]

        def placed(t):
            return [((nodes[n]["x"] + ux * t, nodes[n]["y"] + uy * t), _node_reach(nodes[n]))
                    for n in moving]

        if not clashes(rid, placed(0.0), mine):
            continue
        along = [(nodes[n]["x"] - p["x"]) * ux + (nodes[n]["y"] - p["y"]) * uy for n in body]
        keep = 24.0
        low, high = keep - min(along), (length - keep) - max(along)
        steps = sorted((k * SLIDE_STEP for k in range(int(low // SLIDE_STEP), int(high // SLIDE_STEP) + 1)),
                       key=abs)
        for t in steps:
            if t and not clashes(rid, placed(t), mine):
                for n in moving:
                    old = (nodes[n]["x"], nodes[n]["y"])
                    grid[(int(old[0] // cell), int(old[1] // cell))].remove(n)
                    nodes[n]["x"] += ux * t
                    nodes[n]["y"] += uy * t
                    grid.setdefault((int(nodes[n]["x"] // cell), int(nodes[n]["y"] // cell)), []).append(n)
                for segment in segments:
                    for end, control in (("from_node_id", "b1"), ("to_node_id", "b2")):
                        point = segment.get(control)
                        if point and segment[end] in moving:
                            point["x"] += ux * t
                            point["y"] += uy * t
                reaction["label_x"] += ux * t
                reaction["label_y"] += uy * t
                break


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

    # Reactions joining the same two metabolites -- an opposed pair such as
    # PFK/FBP, or two enzymes for one conversion such as NADH16 and CYTBD --
    # each get a lane of their own, bowed apart.
    #
    # They used to share one axis, each midmarker offset along it, because
    # two parallel *straight* channels draw a closed rectangle with no nodes at
    # its corners, which reads as a cycle (SUCDi/FRD7 drew a false ring beside
    # the real TCA ring). Sharing the axis was worse in a different way: the
    # two reactions are drawn on top of each other, and Q8 -> NADH16 -> CYTBD
    # -> Q8H2 reads as a two-step chain through an intermediate that does not
    # exist. Bowed lanes are a lens, which no one reads as a cycle, and every
    # reaction has a line of its own.
    #
    # Lanes are shared out over the *unordered* pair. Reactions drawn in
    # opposite directions sit on the two opposite edges, and laning each edge
    # on its own put them on the same tracks -- a left lane drawn backwards is
    # the other edge's right lane -- so four iron transporters still drew as
    # two lines.
    siblings = []
    canonical = (sub, prod) if str(sub) <= str(prod) else (prod, sub)
    if sub in pos and prod in pos:
        for u, v in (canonical, canonical[::-1]):
            if cgraph.D.has_edge(u, v):
                siblings.extend(r for r in cgraph.D.edges[u, v]["rxns"] if r not in siblings)
    along = 0.0
    lane = 0.0
    if len(siblings) > 1 and rec.rid in siblings and end_node is not None:
        index = siblings.index(rec.rid)
        lane = (index - (len(siblings) - 1) / 2.0) * LANE_GAP
        if (sub, prod) != canonical:
            lane = -lane
    if lane:
        if len(path) == 2:
            chord = _signed_sagitta(path[0], path[1], arc_centre) if arc_centre else 0.0
            arc_centre = _bow_centre(path[0], path[1], chord + lane)
        else:
            lanes = [(k - (len(siblings) - 1) / 2.0) * LANE_GAP for k in range(len(siblings))]
            rank = sum(1 for other in lanes
                       if other * lane > 0 and abs(other) > abs(lane) + 1e-6)
            path = _offset_polyline(path, lane, rank)

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
    if lane:
        # Outward from the pair, so one lane's cofactors never cross the other.
        side = 1 if lane > 0 else -1
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
                # Escher's b1 is the control beside from_node. A substrate stub
                # is the from_node here, so the pair swaps; passed through
                # unswapped, the curve kinked back past the anchor.
                segment(node_id, anchor_id, b2, b1)

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

_LABEL_LIMIT = 28


def _label_base(identifier):
    """The species part of an id, as a reader should see it, and its key."""
    base = identity.species(identifier)
    if base.startswith("name:") or identity.opaque(base):
        return None, base
    readable = base.replace("__", "-").replace("_", "-")
    return readable, readable.lower()


def display_label(bigg_id, ambiguous=(), name=None):
    """Short, human-facing label for a metabolite id.

    `mal__L_c` -> `Mal`, or `Mal[c]` when malate also appears in another
    compartment on the same map. BiGG's double-underscore escaping never
    reaches the figure. An id that means nothing to a reader -- Human-GEM's
    `MAM01371c`, Yeast-GEM's `s_0434` -- is labelled by the currency key
    (`atp`) or the metabolite's name instead.
    """
    readable, key = _label_base(bigg_id)
    compartment = identity.compartment(bigg_id)
    if readable is None:
        text = identity.currency(bigg_id)
        if not text:
            text = str(name or bigg_id).strip()
            if compartment and text.endswith("_" + compartment):
                text = text[:-len(compartment) - 1]
            text = text.split(" [")[0].strip() or bigg_id
        if len(text) > _LABEL_LIMIT:
            text = text[:_LABEL_LIMIT - 1].rstrip() + "…"
    else:
        text = _ABBREVIATIONS.get(key, readable)
    if key in ambiguous and compartment and len(compartment) <= 3:
        text = f"{text}[{compartment}]"
    return text


def ambiguous_bases(builder):
    """Metabolite bases that appear in more than one compartment on this map."""
    seen = {}
    for node in builder.nodes.values():
        if node["node_type"] != "metabolite":
            continue
        identifier = node["bigg_id"]
        seen.setdefault(_label_base(identifier)[1], set()).add(identity.compartment(identifier))
    return {key for key, comps in seen.items() if len(comps) > 1}


def apply_display_labels(builder):
    ambiguous = ambiguous_bases(builder)
    for node in builder.nodes.values():
        if node["node_type"] == "metabolite":
            node["label_text"] = display_label(node["bigg_id"], ambiguous, node.get("name"))
    for reaction in builder.reactions.values():
        reaction["label_text"] = reaction["bigg_id"]


PRIMARY_RADIUS = 30.0
SECONDARY_RADIUS = 16.0


def _separate_nodes(builder, passes=14):
    """Guarantee drawn nodes do not overlap.

    `sugiyama._enforce_separation` guarantees this for the layout, but the
    renderer adds nodes the layout never saw -- one cofactor stub per reaction
    participant -- so the guarantee did not reach the figure. Two reactions
    sharing a main pair can be drawn on identical geometry and stack their
    stubs exactly, which is how 41% of Recon3D cluster maps ended up with
    coincident nodes at distance 0.

    Only nodes the renderer itself created are moved. Backbone nodes come from
    the layered pass, which already separated them; if two of those collide it
    is a layout bug and pushing them apart here would hide it.

    Movability keys off where a node came from, not off its primary flag.
    `_unify_primary` promotes a duplicate stub to primary when the compound is
    primary elsewhere on the map, and keying off the flag meant those promoted
    stubs became untouchable -- converting 558 fixable stub overlaps into
    unfixable "primary" ones.
    """
    nodes = [n for n in builder.nodes.values() if n["node_type"] == "metabolite"]
    if len(nodes) < 2:
        return 0
    by_id = [(node_id, n) for node_id, n in builder.nodes.items()
             if n["node_type"] == "metabolite"]
    joined = set()
    for reaction in builder.reactions.values():
        for segment in reaction["segments"].values():
            a, b = segment["from_node_id"], segment["to_node_id"]
            joined.update(((a, b), (b, a)))

    cell = 2.2 * PRIMARY_RADIUS
    moved = 0
    for _ in range(passes):
        grid = {}
        for node in nodes:
            grid.setdefault((int(node["x"] // cell), int(node["y"] // cell)), []).append(node)

        collisions = 0
        for (cx, cy), members in grid.items():
            neighbours = [other
                          for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                          for other in grid.get((cx + dx, cy + dy), ())]
            for node in members:
                for other in neighbours:
                    if node is other:
                        continue
                    r1 = PRIMARY_RADIUS if node.get("node_is_primary", True) else SECONDARY_RADIUS
                    r2 = PRIMARY_RADIUS if other.get("node_is_primary", True) else SECONDARY_RADIUS
                    needed = r1 + r2 + 6.0
                    dx = node["x"] - other["x"]
                    dy = node["y"] - other["y"]
                    distance = math.hypot(dx, dy)
                    if distance >= needed:
                        continue
                    collisions += 1
                    # Deterministic tie-break when exactly coincident, so the
                    # same input always produces the same output.
                    if distance < 1e-6:
                        angle = (hash(node["bigg_id"]) % 360) * math.pi / 180.0
                        dx, dy, distance = math.cos(angle), math.sin(angle), 1.0
                    push = (needed - distance)
                    ux, uy = dx / distance, dy / distance
                    if not node.get("_anchored"):
                        node["x"] += ux * push
                        node["y"] += uy * push
                        moved += 1
                    elif not other.get("_anchored"):
                        other["x"] -= ux * push
                        other["y"] -= uy * push
                        moved += 1
        collisions += _clear_markers(builder, by_id, joined, cell)
        if not collisions:
            break
    return moved


MARKER_RADIUS = {"multimarker": 6.0, "midmarker": 11.0}


def _clear_markers(builder, metabolites, joined, cell):
    """Push movable stubs off the markers of reactions they are not joined to.

    Separation used to compare metabolites only with metabolites, so a
    cofactor stub could sit on another reaction's arrow: in e_coli_core the
    CO2 of PEP carboxylase landed on aconitase's midmarker once the
    anaplerotic reactions were drawn beside the TCA ring. Markers belong to
    the layout and stay put; the stub moves.
    """
    grid = {}
    for node_id, node in builder.nodes.items():
        if node["node_type"] in MARKER_RADIUS:
            grid.setdefault((int(node["x"] // cell), int(node["y"] // cell)), []).append(node_id)
    hits = 0
    for node_id, node in metabolites:
        if node.get("_anchored"):
            continue
        r = PRIMARY_RADIUS if node.get("node_is_primary", True) else SECONDARY_RADIUS
        cx, cy = int(node["x"] // cell), int(node["y"] // cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for marker_id in grid.get((cx + dx, cy + dy), ()):
                    if (node_id, marker_id) in joined:
                        continue
                    marker = builder.nodes[marker_id]
                    needed = r + MARKER_RADIUS[marker["node_type"]] + 6.0
                    ux, uy = node["x"] - marker["x"], node["y"] - marker["y"]
                    distance = math.hypot(ux, uy)
                    if distance >= needed:
                        continue
                    hits += 1
                    if distance < 1e-6:
                        angle = (hash(node["bigg_id"]) % 360) * math.pi / 180.0
                        ux, uy, distance = math.cos(angle), math.sin(angle), 1.0
                    node["x"] += ux / distance * (needed - distance)
                    node["y"] += uy / distance * (needed - distance)
    return hits


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
BOUNDARY_MARGIN = 0.5          # see _Obstacles.hits
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
                        # A hair of margin: a label placed exactly on the
                        # clearance boundary passes or fails on float rounding,
                        # and `metrics` recomputes the box from the stored
                        # left edge and width, so it came out the other way --
                        # one Yeast-GEM label read as sitting on a node.
                        clearance += BOUNDARY_MARGIN
                        if (left - clearance < x < right + clearance
                                and top - clearance < y < bottom + clearance):
                            return True
                    elif item[0] == "segment":
                        _, (x1, y1, x2, y2), clearance = item
                        clearance += BOUNDARY_MARGIN
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


def _segment_points(a, b, segment, steps=10):
    """A segment as a polyline: its two ends, or its Bezier sampled."""
    p0, p3 = (a["x"], a["y"]), (b["x"], b["y"])
    if not (segment.get("b1") and segment.get("b2")):
        return [p0, p3]
    b1 = (segment["b1"]["x"], segment["b1"]["y"])
    b2 = (segment["b2"]["x"], segment["b2"]["y"])
    out = []
    for i in range(steps + 1):
        t = i / steps
        m = 1.0 - t
        out.append((m ** 3 * p0[0] + 3 * m * m * t * b1[0] + 3 * m * t * t * b2[0] + t ** 3 * p3[0],
                    m ** 3 * p0[1] + 3 * m * m * t * b1[1] + 3 * m * t * t * b2[1] + t ** 3 * p3[1]))
    return out


FAR_STEP = 60.0            # ring spacing when a label has to look further out
FAR_LIMIT = 20000.0


def _place_far(obstacles, text, factor, anchor_x, anchor_y):
    """The nearest free place for a label at the smallest size, at any distance."""
    width, height = label_box(text, MIN_FONT_BASE, factor)
    half_w, half_v = width / 2.0, height / 2.0
    radius = FAR_STEP * 2
    while radius < FAR_LIMIT:
        arms = max(16, int(2.0 * math.pi * radius / FAR_STEP))
        ring = []
        for index in range(arms):
            angle = 2.0 * math.pi * index / arms
            ring.append((abs(math.sin(angle)), radius * math.cos(angle),
                         radius * math.sin(angle)))
        ring.sort(key=lambda a: a[0])           # beside before above or below
        for _, dx, dy in ring:
            cx, cy = anchor_x + dx, anchor_y + dy
            if not obstacles.hits(cx - half_w, cy - half_v, cx + half_w, cy + half_v):
                return (cx, cy, half_w, half_v, MIN_FONT_BASE)
        radius += FAR_STEP
    return (anchor_x + FAR_LIMIT, anchor_y, half_w, half_v, MIN_FONT_BASE)


def _place_labels(builder):
    """Place every label near the thing it names, shrinking it if it must.

    Nodes and edges are final by this point; labels are the only thing allowed
    to move. `layout_algorithm.md` asks that a label never overlap a node, an edge,
    or another label, and the previous version satisfied that by widening the
    search until something was free -- which left labels hundreds of pixels
    from their node, annotating nothing. So the near search comes first and
    the font ladder absorbs the pressure there; only a label that fits nowhere
    near, even at the minimum size, goes further out (`_place_far`). It never
    overlaps: text on a node or an edge makes both unreadable.

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
            stub = any(n["node_type"] == "metabolite" and not n.get("node_is_primary", True)
                       for n in (a, b))
            clearance = _STUB_CLEARANCE if stub else _SEGMENT_CLEARANCE
            # A curve is an obstacle along the curve. Its chord, which is what
            # this used to add, runs through the empty middle of a bowed lane
            # or a ring and kept labels off exactly the wrong place.
            points = _segment_points(a, b, segment)
            for (x1, y1), (x2, y2) in zip(points, points[1:]):
                obstacles.add_segment(x1, y1, x2, y2, clearance)

    targets = []
    for node in builder.nodes.values():
        if node["node_type"] == "metabolite":
            targets.append((node, node.get("label_text", node["bigg_id"]),
                            node["x"], node["y"], METABOLITE_FONT_FACTOR))
    # Primaries first: the backbone must be readable, and in a dense cluster
    # there is not room for everything. Within each class, longest first.
    targets.sort(key=lambda t: (not t[0].get("node_is_primary", True), -len(t[1])))
    reaction_targets = []
    for reaction in builder.reactions.values():
        anchor_x, anchor_y = reaction.pop("_anchor")
        reaction_targets.append((reaction, reaction["bigg_id"], anchor_x, anchor_y,
                                 REACTION_FONT_FACTOR))
    reaction_targets.sort(key=lambda t: -len(t[1]))
    # Reaction labels rank between primary and secondary metabolites: naming
    # the step matters more than naming its cofactors.
    primaries = [t for t in targets if t[0].get("node_is_primary", True)]
    secondaries = [t for t in targets if not t[0].get("node_is_primary", True)]
    targets = primaries + reaction_targets + secondaries

    crowded = 0
    for holder, text, anchor_x, anchor_y, factor in targets:
        placed = None

        # Nearest placement wins over largest font: a caption that has drifted
        # away from its node is a worse failure than a slightly smaller one.
        for font_base in FONT_LADDER:
            width, height = label_box(text, font_base, factor)
            half_w, half_v = width / 2.0, height / 2.0
            gap = max(6.0, 0.35 * height)
            for dx, dy in _candidates(half_w, half_v, gap):
                cx, cy = anchor_x + dx, anchor_y + dy
                if not obstacles.hits(cx - half_w, cy - half_v,
                                      cx + half_w, cy + half_v):
                    placed = (cx, cy, half_w, half_v, font_base)
                    break
            if placed:
                break

        if placed is None:
            # Nothing free close by, even at the smallest size. Text over a
            # node or an edge is never acceptable -- it makes both unreadable --
            # so look further out, ring by ring, until there is room. There
            # always is, eventually, past the edge of the drawing.
            #
            # This used to accept the overlap for backbone and reaction labels
            # and hide cofactor labels instead. Hiding only works in our own
            # renderers: Escher does not know `label_hidden` and drew those
            # labels where they were, on top of whatever was there.
            placed = _place_far(obstacles, text, factor, anchor_x, anchor_y)
            crowded += 1

        cx, cy, half_w, half_v, font_base = placed
        holder["label_x"] = cx - half_w
        holder["label_y"] = cy
        if font_base != ESCHER_DEFAULT_FONT_BASE:
            holder["font_size_base"] = font_base
        holder.pop("label_hidden", None)
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
                       node["label_x"] + len(node.get("label_text", node["bigg_id"]))
                       * size * CHAR_WIDTH_RATIO))
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
TITLE_MIN_FONT_BASE = 12.0         # still a title, beside 18-point metabolites x 1.1
TITLE_MIN_WIDTH = 900.0            # a single node's map may still carry a title
TITLE_CLEARANCE = 50.0             # title's lower edge to the top row's centres
ATTRIBUTION_FONT_BASE = 8.0        # a credit line, not a title


def _content_top(builder):
    tops = [n["y"] for n in builder.nodes.values()]
    for node in builder.nodes.values():
        if "label_y" in node:
            tops.append(node["label_y"])
    for reaction in builder.reactions.values():
        tops.append(reaction["label_y"])
    return min(tops) if tops else 0.0


def _title(builder, canvas, text):
    """Name the figure.

    Every emitted map previously carried exactly one text label and it was the
    author credit -- a figure with a byline and no title, which is precisely
    backwards. Title, region caption, reaction, metabolite gives the three-level
    size hierarchy a reader needs to enter the figure.
    """
    if not text:
        return
    # Above the content, not inside the canvas padding. Placing it at a fixed
    # offset from the canvas edge put it on top of the network whenever the
    # proportional padding was smaller than the title itself -- free text was
    # 20 of 207 flagged label collisions.
    #
    # No wider than the drawing. A title set at full size across a narrow
    # pathway widens the canvas to fit it, and everything right of the drawing
    # is then empty: a two-pathway column under a 55-character title was half
    # blank. The title shrinks instead, down to a size that still reads as one.
    xs = [n["x"] for n in builder.nodes.values()]
    room = max(max(xs) - min(xs) if xs else 0.0, TITLE_MIN_WIDTH)
    per_point = len(text) * TEXT_LABEL_FONT_FACTOR * CHAR_WIDTH_RATIO
    font = max(TITLE_MIN_FONT_BASE, min(TITLE_FONT_BASE, room / max(per_point, 1e-9)))
    height = font * TEXT_LABEL_FONT_FACTOR * LINE_HEIGHT_RATIO
    # The gap is measured from the title's lower edge, not its size: offset by
    # its own height, a shrunken title came down onto the top row of nodes.
    builder.text_labels["map_title"] = {
        "x": canvas["x"] + 60.0,
        "y": _content_top(builder) - height / 2.0 - TITLE_CLEARANCE,
        "text": text,
        "font_size_base": round(font, 1),
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


def annotate_pathways(escher_map, parts, region=None):
    """Record which pathway each reaction of a map belongs to.

    `parts` is [(pathway name, reactions or reaction ids)]. Written once, in
    the map header, as `pathways`: [{name, region, reactions}] with reaction
    keys as they appear in this map. A viewer uses it to select or move a
    whole pathway at once; nothing in the drawing depends on it.
    """
    present = escher_map[1]["reactions"]
    entries = []
    for name, reactions in parts:
        ids = [getattr(r, "id", r) for r in reactions]
        keys = [i for i in ids if i in present]
        if keys:
            entries.append({"name": name, "region": region, "reactions": keys})
    if entries:
        escher_map[0]["pathways"] = entries
    return escher_map


def save(escher_map, path):
    """Write one Escher map, compactly.

    These files are read by software, not by people -- Escher loads them, and
    the map picker fetches them over HTTP one at a time. Indenting them costs
    43% of the file size for nothing: across the 2623-map BiGG collection that
    is 841 MB against 480 MB, doubling both the clone and every fetch the
    viewer makes.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(escher_map, handle, separators=(",", ":"))
