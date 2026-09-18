"""Acceptance metrics (layout_algorithm.md S9).

These are what separate the positive templates from the force-directed
negatives numerically, so they double as a regression gate: a change that
improves one pathway by eye but wrecks these on the other hundred models is
not an improvement.

Computed on the emitted Escher map, not on internal state, so they measure
what a reader actually sees.
"""

import math

TARGETS = {
    "axis_aligned": (0.90, None),
    "crossings_per_edge": (None, 0.05),
    "longest_run_ratio": (0.15, None),
    "min_separation_ratio": (1.0, None),
    "label_overlaps": (None, 0),
    "hairball_index": (None, 3.0),
    "occupancy": (0.10, 0.85),
    "aspect_ratio": (0.35, 3.0),
}

_LABEL_CHAR_WIDTH = 17.0
_LABEL_HEIGHT = 40.0


def _straight_segments(body):
    nodes = body["nodes"]
    out = []
    for reaction in body["reactions"].values():
        for segment in reaction["segments"].values():
            if segment.get("b1") or segment.get("b2"):
                continue                      # cofactor arc, curved by design
            a = nodes.get(segment["from_node_id"])
            b = nodes.get(segment["to_node_id"])
            if a is None or b is None:
                continue
            out.append((a["x"], a["y"], b["x"], b["y"]))
    return out


def _segments_cross(s1, s2):
    (x1, y1, x2, y2), (x3, y3, x4, y4) = s1, s2
    if (abs(x1 - x3) < 1e-6 and abs(y1 - y3) < 1e-6) or (abs(x1 - x4) < 1e-6 and abs(y1 - y4) < 1e-6):
        return False
    if (abs(x2 - x3) < 1e-6 and abs(y2 - y3) < 1e-6) or (abs(x2 - x4) < 1e-6 and abs(y2 - y4) < 1e-6):
        return False

    def orient(ax, ay, bx, by, cx, cy):
        v = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)

    d1 = orient(x1, y1, x2, y2, x3, y3)
    d2 = orient(x1, y1, x2, y2, x4, y4)
    d3 = orient(x3, y3, x4, y4, x1, y1)
    d4 = orient(x3, y3, x4, y4, x2, y2)
    return d1 != d2 and d3 != d4


def _count_crossings(segments, cell=400.0):
    """Bucketed pair test; exhaustive comparison is quadratic and a
    genome-scale map has tens of thousands of segments."""
    buckets = {}
    for i, (x1, y1, x2, y2) in enumerate(segments):
        steps = max(1, int(math.hypot(x2 - x1, y2 - y1) / cell) + 1)
        for k in range(steps + 1):
            t = k / steps
            key = (int((x1 + (x2 - x1) * t) // cell), int((y1 + (y2 - y1) * t) // cell))
            buckets.setdefault(key, set()).add(i)

    seen, crossings = set(), 0
    for members in buckets.values():
        members = sorted(members)
        for a_index in range(len(members)):
            for b_index in range(a_index + 1, len(members)):
                pair = (members[a_index], members[b_index])
                if pair in seen:
                    continue
                seen.add(pair)
                if _segments_cross(segments[pair[0]], segments[pair[1]]):
                    crossings += 1
    return crossings


def _longest_straight_run(segments, tolerance=1.0):
    """Longest chain of collinear, axis-aligned, touching segments."""
    by_line = {}
    for x1, y1, x2, y2 in segments:
        if abs(x1 - x2) < tolerance:
            by_line.setdefault(("v", round(x1)), []).append((min(y1, y2), max(y1, y2)))
        elif abs(y1 - y2) < tolerance:
            by_line.setdefault(("h", round(y1)), []).append((min(x1, x2), max(x1, x2)))

    best = 0.0
    for spans in by_line.values():
        spans.sort()
        current_start, current_end = spans[0]
        for start, end in spans[1:]:
            if start <= current_end + tolerance:
                current_end = max(current_end, end)
            else:
                best = max(best, current_end - current_start)
                current_start, current_end = start, end
        best = max(best, current_end - current_start)
    return best


def _label_boxes(body):
    boxes = []
    for node in body["nodes"].values():
        if node["node_type"] != "metabolite":
            continue
        width = max(90.0, len(node["bigg_id"]) * _LABEL_CHAR_WIDTH)
        boxes.append((node["label_x"], node["label_y"] - _LABEL_HEIGHT / 2.0,
                      node["label_x"] + width, node["label_y"] + _LABEL_HEIGHT / 2.0))
    for reaction in body["reactions"].values():
        width = max(90.0, len(reaction["bigg_id"]) * _LABEL_CHAR_WIDTH)
        boxes.append((reaction["label_x"], reaction["label_y"] - _LABEL_HEIGHT / 2.0,
                      reaction["label_x"] + width, reaction["label_y"] + _LABEL_HEIGHT / 2.0))
    return boxes


def _count_box_overlaps(boxes, cell=400.0):
    buckets = {}
    for i, (left, top, right, bottom) in enumerate(boxes):
        for cx in range(int(left // cell), int(right // cell) + 1):
            for cy in range(int(top // cell), int(bottom // cell) + 1):
                buckets.setdefault((cx, cy), set()).add(i)

    seen, overlaps = set(), 0
    for members in buckets.values():
        members = sorted(members)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pair = (members[a], members[b])
                if pair in seen:
                    continue
                seen.add(pair)
                l1, t1, r1, b1 = boxes[pair[0]]
                l2, t2, r2, b2 = boxes[pair[1]]
                if not (r1 <= l2 or l1 >= r2 or b1 <= t2 or t1 >= b2):
                    overlaps += 1
    return overlaps


def _hairball_index(points, bins=20):
    """Peak local node density over mean occupied density.

    A spring layout piles everything into one blob, which shows up here as a
    large ratio; `nt1`/`nt2` score above 15, a curated map below 3.
    """
    if len(points) < 4:
        return 1.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    width = max(max(xs) - min(xs), 1e-6)
    height = max(max(ys) - min(ys), 1e-6)

    counts = {}
    for x, y in points:
        key = (min(bins - 1, int((x - min(xs)) / width * bins)),
               min(bins - 1, int((y - min(ys)) / height * bins)))
        counts[key] = counts.get(key, 0) + 1
    mean = sum(counts.values()) / len(counts)
    return max(counts.values()) / mean if mean else 1.0


def score(escher_map, pitch=180.0):
    """Compute the acceptance metrics for one emitted map."""
    body = escher_map[1]
    nodes = body["nodes"]
    segments = _straight_segments(body)

    aligned = sum(
        1 for x1, y1, x2, y2 in segments
        if abs(x1 - x2) < 1.0 or abs(y1 - y2) < 1.0
    )
    metabolites = [(n["x"], n["y"]) for n in nodes.values()
                   if n["node_type"] == "metabolite"]
    # Separation is measured between *primary* metabolites only. Cofactor stubs
    # sit deliberately close to their reaction axis, so including them would
    # measure the stub radius rather than whether the backbone is crowded.
    primaries = [(n["x"], n["y"]) for n in nodes.values()
                 if n["node_type"] == "metabolite" and n.get("node_is_primary", True)]

    min_separation = math.inf
    cell = pitch
    buckets = {}
    for x, y in primaries:
        buckets.setdefault((int(x // cell), int(y // cell)), []).append((x, y))
    for (cx, cy), members in buckets.items():
        neighbours = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbours.extend(buckets.get((cx + dx, cy + dy), ()))
        for x, y in members:
            for ox, oy in neighbours:
                if (x, y) == (ox, oy):
                    continue
                min_separation = min(min_separation, math.hypot(x - ox, y - oy))
    if min_separation is math.inf:
        min_separation = pitch

    xs = [p[0] for p in metabolites] or [0.0]
    ys = [p[1] for p in metabolites] or [0.0]
    # Clamp to one node footprint. A two-metabolite cluster is a vertical pair
    # with zero measured width, and reporting aspect 0.000 for a drawing that
    # is exactly right is a false alarm that hides the real ones.
    node_extent = 160.0
    content_w = max(max(xs) - min(xs), node_extent)
    content_h = max(max(ys) - min(ys), node_extent)
    canvas = body["canvas"]

    total_length = sum(math.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in segments)

    return {
        "nodes": len(nodes),
        "reactions": len(body["reactions"]),
        "axis_aligned": aligned / len(segments) if segments else 1.0,
        "crossings_per_edge": (_count_crossings(segments) / len(segments)) if segments else 0.0,
        "longest_run_ratio": (_longest_straight_run(segments) / total_length) if total_length else 0.0,
        "min_separation_ratio": min_separation / pitch,
        "label_overlaps": _count_box_overlaps(_label_boxes(body)),
        "hairball_index": _hairball_index(metabolites),
        "occupancy": (content_w * content_h) / max(canvas["width"] * canvas["height"], 1.0),
        "aspect_ratio": content_w / content_h,
    }


def format_report(values):
    lines = []
    for key, value in values.items():
        low, high = TARGETS.get(key, (None, None))
        flag = " "
        if low is not None and value < low:
            flag = "!"
        if high is not None and value > high:
            flag = "!"
        rendered = f"{value:.3f}" if isinstance(value, float) else str(value)
        lines.append(f"  {flag} {key:22} {rendered}")
    return "\n".join(lines)
