"""Read a KEGG KGML pathway as an Escher-shaped map, so curated human layout
can be scored by exactly the same code that scores ours.

Why this exists
---------------
`layout_algorithm.md` §9 sets every acceptance threshold -- crossings < 0.05,
hairball < 3, and the rest -- and justifies all of them with one sentence:
that these are the numbers separating `templates/t*` from `templates/nt*`. That
is a calibration against two images. It is not a defensible basis for the
thresholds a paper reports against.

`data/kegg/` holds 1,010 KGML pathways whose coordinates were drawn by KEGG's
curators. Scoring those gives an empirical distribution of what curated
metabolic layout actually measures, which is what the thresholds should be
derived from.

The conversion
--------------
KGML and Escher describe the same picture differently, so the mapping is
chosen to preserve *geometry*, which is what the metrics read:

  <entry type="compound">        -> a metabolite node at its graphics x/y
  <entry> carrying reaction=...  -> the midmarker, at the enzyme box's x/y
  <reaction><substrate>/<product> -> segments substrate -> midmarker -> product

Straight segments, no Bezier controls: KEGG draws straight connectors, and
giving them curvature would exclude them from `_straight_segments` and quietly
measure nothing.

What does NOT transfer
----------------------
Only the geometry metrics are comparable. KEGG labels compounds with tiny
C-numbers and enzymes with EC boxes, which is a different labelling problem
from Escher's, so `label_on_node`, `label_on_edge` and `label_overlaps` are
meaningless here and callers must drop them. `min_separation_ratio` is scaled
by a pitch that is an Escher convention, so it does not transfer either.
`comparable_metrics()` is the authoritative list.
"""

import os
import xml.etree.ElementTree as ET

# The metrics that mean the same thing on a KEGG drawing and on ours.
#
# This list is short on purpose, and three obvious candidates are excluded
# because comparing them would be measuring the conversion rather than the
# curation:
#
#   axis_aligned, longest_run_ratio -- KGML stores node positions and nothing
#     else. There is no `coords` attribute anywhere in these 1,010 files, so
#     the connector route KEGG actually draws is not in the data. Scoring the
#     straight substrate -> enzyme -> product chords instead reports how the
#     *nodes* line up, which is a different quantity. Measured that way KEGG
#     scores about 0.21 axis-aligned against our 0.99, and reading that as
#     "curators do not draw orthogonally" would be wrong: their rendered maps
#     plainly do.
#
#   occupancy -- content bbox over canvas, and KEGG has no canvas. The one in
#     `load()` is this module's invention, so the ratio would measure the
#     padding constant chosen here.
#
# crossings_per_edge is kept with a caveat: it is route-dependent too, so a
# KEGG figure here is the straight-line crossing number for that placement,
# which is the standard graph-drawing measure but is not what KEGG's own
# renderer would produce. Treat it as a placement-quality comparison.
COMPARABLE = (
    "crossings_per_edge",
    "hairball_index",
    "aspect_ratio",
)

# Computed and reported, but not comparable -- see above.
NOT_COMPARABLE = ("axis_aligned", "longest_run_ratio", "occupancy",
                  "min_separation_ratio", "label_overlaps", "label_on_node",
                  "label_on_edge")


def comparable_metrics():
    return COMPARABLE


def _graphics(entry):
    g = entry.find("graphics")
    if g is None:
        return None
    try:
        return float(g.get("x")), float(g.get("y"))
    except (TypeError, ValueError):
        return None


def load(path, canvas_padding=40.0):
    """One KGML file as `[header, body]`, or None if it draws no reactions.

    Many KGML files in a genome download are regulatory or signalling maps with
    no `<reaction>` elements at all, and a few metabolic ones reference entries
    that carry no graphics. Both return None rather than an empty map, so the
    caller's denominator is the number of pathways actually measured.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None

    entries = {}
    for entry in root.findall("entry"):
        xy = _graphics(entry)
        if xy is not None:
            entries[entry.get("id")] = (entry, xy)

    nodes, reactions = {}, {}
    used = set()

    for reaction in root.findall("reaction"):
        box = entries.get(reaction.get("id"))
        if box is None:
            continue
        subs = [s.get("id") for s in reaction.findall("substrate")]
        prods = [p.get("id") for p in reaction.findall("product")]
        subs = [s for s in subs if s in entries]
        prods = [p for p in prods if p in entries]
        if not subs or not prods:
            continue

        mid_id = "mid_%s" % reaction.get("id")
        mx, my = box[1]
        nodes[mid_id] = {"node_type": "midmarker", "x": mx, "y": my}

        segments = {}
        for i, member in enumerate(subs + prods):
            entry, (x, y) = entries[member]
            nodes.setdefault(member, {
                "node_type": "metabolite",
                "x": x,
                "y": y,
                "bigg_id": (entry.get("name") or "").replace("cpd:", ""),
                "name": (entry.get("name") or ""),
                "node_is_primary": True,
                # score() requires these. KEGG does write the compound id at
                # the circle, so the position is honest -- but the label
                # metrics derived from it are not comparable and the caller
                # drops them. See the module docstring.
                "label_x": x,
                "label_y": y,
            })
            used.add(member)
            # substrate -> midmarker -> product, matching how the metrics walk
            # an Escher reaction.
            a, b = (member, mid_id) if member in subs else (mid_id, member)
            segments["s%d" % i] = {
                "from_node_id": a, "to_node_id": b, "b1": None, "b2": None,
            }

        reactions[reaction.get("id")] = {
            "bigg_id": (reaction.get("name") or "").replace("rn:", ""),
            "name": reaction.get("name") or "",
            "reversibility": reaction.get("type") == "reversible",
            "label_x": mx,
            "label_y": my,
            "segments": segments,
            "metabolites": [],
        }

    if not reactions:
        return None

    # Compounds KEGG draws but this pathway does not react -- map cross-links,
    # mostly. They are on the canvas, so they count towards density and extent.
    for key, (entry, (x, y)) in entries.items():
        if key in used or entry.get("type") != "compound":
            continue
        nodes[key] = {
            "node_type": "metabolite", "x": x, "y": y,
            "bigg_id": (entry.get("name") or "").replace("cpd:", ""),
            "name": (entry.get("name") or ""), "node_is_primary": True,
            "label_x": x, "label_y": y,
        }

    xs = [n["x"] for n in nodes.values()]
    ys = [n["y"] for n in nodes.values()]
    canvas = {
        "x": min(xs) - canvas_padding,
        "y": min(ys) - canvas_padding,
        "width": (max(xs) - min(xs)) + 2 * canvas_padding,
        "height": (max(ys) - min(ys)) + 2 * canvas_padding,
    }

    header = {
        "map_name": root.get("title") or os.path.basename(path),
        "map_id": root.get("name") or "",
        "map_description": "KEGG KGML, curated coordinates",
        "homepage": "https://www.kegg.jp/",
        "schema": "https://escher.github.io/escher/jsonschema/1-0-0#",
    }
    body = {
        "reactions": reactions,
        "nodes": nodes,
        "text_labels": {},
        "canvas": canvas,
    }
    return [header, body]


def load_all(directory, limit=None):
    """Every KGML file in `directory` that draws at least one reaction."""
    out = []
    names = sorted(n for n in os.listdir(directory) if n.endswith(".xml"))
    for name in names:
        chart = load(os.path.join(directory, name))
        if chart is not None:
            out.append((name, chart))
            if limit and len(out) >= limit:
                break
    return out
