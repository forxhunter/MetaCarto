"""Squeeze oversized empty bands out of a finished map.

Average density can be fine while the drawing still looks wrong, because the
eye reads the *distribution* of white, not its total. The Recon3D functional
maps score a perfectly reasonable 0.66 occupancy and still have a void down the
middle and a crowd in one corner: hairball_index -- max local density over mean
local density -- sits at 4.0 against a target of 3.

The cause is structural rather than a bug in any one step. Wrapping a wide
layer leaves a ragged last row; disconnected components get parked side by
side; a single long edge reserves a corridor the width of the page and puts
nothing in it.

Rather than chase each source, this runs at the end on the emitted map: find
the empty bands, and compress the ones bigger than a sensible gap. It is a
monotone, piecewise-linear remap of each axis, so the drawing's order and
structure are untouched -- nothing crosses anything it did not cross before,
and no pair of nodes gets closer than the gap it is compressed to.

Everything positional has to move together or the map tears: node centres,
label anchors, and the Bezier control points of every segment.
"""


def _coords(body):
    xs, ys = [], []
    for node in body["nodes"].values():
        xs.append(node["x"])
        ys.append(node["y"])
        if "label_x" in node:
            xs.append(node["label_x"])
            ys.append(node["label_y"])
    for reaction in body["reactions"].values():
        if "label_x" in reaction:
            xs.append(reaction["label_x"])
            ys.append(reaction["label_y"])
        for segment in reaction["segments"].values():
            for key in ("b1", "b2"):
                point = segment.get(key)
                if point:
                    xs.append(point["x"])
                    ys.append(point["y"])
    for label in body.get("text_labels", {}).values():
        xs.append(label["x"])
        ys.append(label["y"])
    return xs, ys


def _build_remap(values, max_gap):
    """Piecewise-linear map collapsing every empty run longer than `max_gap`.

    Returns None when nothing needs collapsing, so the caller can skip the walk
    over every coordinate in the map.
    """
    if not values:
        return None
    points = sorted(set(round(v, 3) for v in values))
    shift, breaks = 0.0, []
    for previous, current in zip(points, points[1:]):
        gap = current - previous
        if gap > max_gap:
            shift += gap - max_gap
            breaks.append((current, shift))
    if not breaks:
        return None

    def remap(value):
        # Total shift accumulated at or before `value`.
        total = 0.0
        for position, accumulated in breaks:
            if value >= position:
                total = accumulated
            else:
                break
        return value - total

    return remap


def compact(escher_map, max_gap=900.0):
    """Collapse empty bands wider than `max_gap`. Returns (dx_saved, dy_saved).

    `max_gap` is deliberately larger than the node pitch: the goal is to remove
    the corridors nothing lives in, not to close the breathing room between
    pathways, which is doing real work for the reader.
    """
    body = escher_map[1]
    xs, ys = _coords(body)
    if not xs:
        return 0.0, 0.0

    remap_x = _build_remap(xs, max_gap)
    remap_y = _build_remap(ys, max_gap)
    if remap_x is None and remap_y is None:
        return 0.0, 0.0

    fx = remap_x or (lambda v: v)
    fy = remap_y or (lambda v: v)

    before_w = max(xs) - min(xs)
    before_h = max(ys) - min(ys)

    for node in body["nodes"].values():
        node["x"] = fx(node["x"])
        node["y"] = fy(node["y"])
        if "label_x" in node:
            node["label_x"] = fx(node["label_x"])
            node["label_y"] = fy(node["label_y"])
    for reaction in body["reactions"].values():
        if "label_x" in reaction:
            reaction["label_x"] = fx(reaction["label_x"])
            reaction["label_y"] = fy(reaction["label_y"])
        for segment in reaction["segments"].values():
            for key in ("b1", "b2"):
                point = segment.get(key)
                if point:
                    point["x"] = fx(point["x"])
                    point["y"] = fy(point["y"])
    for label in body.get("text_labels", {}).values():
        label["x"] = fx(label["x"])
        label["y"] = fy(label["y"])

    xs2, ys2 = _coords(body)
    pad_x = (body["canvas"]["width"] - before_w) / 2.0
    pad_y = (body["canvas"]["height"] - before_h) / 2.0
    body["canvas"] = {
        "x": min(xs2) - pad_x,
        "y": min(ys2) - pad_y,
        "width": (max(xs2) - min(xs2)) + 2 * pad_x,
        "height": (max(ys2) - min(ys2)) + 2 * pad_y,
    }
    return before_w - (max(xs2) - min(xs2)), before_h - (max(ys2) - min(ys2))
