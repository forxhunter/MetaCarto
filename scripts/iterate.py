"""Generate-then-evaluate loop for a model's maps.

The acceptance loop the layout work actually needs: run the pipeline, score
every map it emitted, and print one table saying which gates fail and where.
Exit code is 0 only when every gate passes, so this can drive a fix/regenerate
cycle or a CI check.

    python scripts/iterate.py --model Recon3D                 # evaluate what is on disk
    python scripts/iterate.py --model Recon3D --generate      # regenerate first
    python scripts/iterate.py --model Recon3D --top 15

Two things it checks that src/layout/metrics.py does not:

1. **Print legibility.** A map is only a figure if its labels survive the page.
   Fitting the content box to a journal width and a 240 mm page height, the
   metabolite label must land at or above `--min-pt` (Nature's floor is ~5 pt).
   This is the gate the composed posters fail hardest and the one the geometry
   metrics are blind to.
2. **Title/caption collisions.** `metrics._label_boxes` walks only `nodes` and
   `reactions`, so every region-title and cluster-caption overlap on a composed
   map is invisible to `label_overlaps`. Composed maps are exactly where those
   labels live, so they are measured here instead.
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.layout import metrics
from src.layout.render import (CHAR_WIDTH_RATIO, ESCHER_DEFAULT_FONT_BASE,
                               LINE_HEIGHT_RATIO, METABOLITE_FONT_FACTOR,
                               TEXT_LABEL_FONT_FACTOR)

MM_PER_PT = 25.4 / 72.0
PAGE_HEIGHT_MM = 240.0

# Gates from layout_algorithm.md S9, as (lower, upper); None means unbounded.
GATES = dict(metrics.TARGETS)


def content_box(body):
    xs = [n["x"] for n in body["nodes"].values()]
    ys = [n["y"] for n in body["nodes"].values()]
    if not xs:
        return 0.0, 0.0
    return max(xs) - min(xs), max(ys) - min(ys)


def label_pt(body, width_mm, page_mm=PAGE_HEIGHT_MM):
    """Metabolite label size in points once the map is fitted to the page.

    Fit to width, then fall back to fitting height if the map is too tall --
    which is what a production editor does, and what turns a tall pathway
    column into unreadably small type.
    """
    w, h = content_box(body)
    if w <= 0 or h <= 0:
        return 0.0
    scale = width_mm / w
    if h * scale > page_mm:
        scale = page_mm / h
    units = ESCHER_DEFAULT_FONT_BASE * METABOLITE_FONT_FACTOR
    return units * scale / MM_PER_PT


def text_label_collisions(body):
    """Overlapping pairs among free text labels (titles, captions, regions)."""
    boxes = []
    for label in body.get("text_labels", {}).values():
        size = label.get("font_size_base", ESCHER_DEFAULT_FONT_BASE) * TEXT_LABEL_FONT_FACTOR
        width = max(len(label.get("text", "")), 1) * size * CHAR_WIDTH_RATIO
        height = size * LINE_HEIGHT_RATIO
        boxes.append((label["x"], label["y"] - height / 2.0,
                      label["x"] + width, label["y"] + height / 2.0))
    hits = 0
    for i in range(len(boxes)):
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            bx0, by0, bx1, by1 = boxes[j]
            if not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0):
                hits += 1
    return hits


def canvas_overflow(body):
    """Units by which node+label ink escapes the declared canvas rectangle."""
    canvas = body["canvas"]
    xs, ys = [], []
    for node in body["nodes"].values():
        xs.append(node["x"])
        ys.append(node["y"])
        if node["node_type"] == "metabolite":
            size = node.get("font_size_base", ESCHER_DEFAULT_FONT_BASE) * METABOLITE_FONT_FACTOR
            xs.append(node["label_x"] + len(node["bigg_id"]) * size * CHAR_WIDTH_RATIO)
            ys.append(node["label_y"])
    if not xs:
        return 0.0
    return max(canvas["x"] - min(xs), min(xs) - min(xs),
               max(xs) - (canvas["x"] + canvas["width"]),
               canvas["y"] - min(ys),
               max(ys) - (canvas["y"] + canvas["height"]), 0.0)


def evaluate(path, width_mm, min_pt):
    body = json.load(open(path))[1]
    escher = json.load(open(path))
    values = metrics.score(escher)
    values["label_pt"] = label_pt(body, width_mm)
    values["title_collisions"] = text_label_collisions(body)
    values["canvas_overflow"] = canvas_overflow(body)
    failures = []
    for key, (lo, hi) in GATES.items():
        v = values.get(key)
        if v is None:
            continue
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            failures.append(key)
    if values["label_pt"] < min_pt:
        failures.append("label_pt")
    if values["title_collisions"] > 0:
        failures.append("title_collisions")
    if values["canvas_overflow"] > 0:
        failures.append("canvas_overflow")
    return values, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="layout_output")
    parser.add_argument("--generate", action="store_true",
                        help="run layout_v2 before evaluating")
    parser.add_argument("--width-mm", type=float, default=180.0,
                        help="journal figure width the map must fit (default full page)")
    parser.add_argument("--min-pt", type=float, default=5.0,
                        help="smallest acceptable metabolite label (default Nature's floor)")
    parser.add_argument("--top", type=int, default=10,
                        help="how many worst offenders to list per gate")
    args = parser.parse_args(argv)

    out_dir = os.path.join(args.out, args.model)

    if args.generate:
        import layout_v2
        argv2 = ["--model", args.model, "--combined", "--preview", "--quiet"]
        print("generating: layout_v2.py " + " ".join(argv2))
        layout_v2.main(argv2)

    paths = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    if not paths:
        print("no maps under " + out_dir)
        return 2

    combined_path = None
    cluster_paths = []
    for p in paths:
        if "_Combined" in os.path.basename(p):
            combined_path = p
        else:
            cluster_paths.append(p)

    print("model %s: %d cluster maps%s" % (
        args.model, len(cluster_paths),
        ", 1 combined map" if combined_path else ", NO COMBINED MAP"))
    print("fitting to %.0f mm wide / %.0f mm page, label floor %.1f pt\n"
          % (args.width_mm, PAGE_HEIGHT_MM, args.min_pt))

    scored = []
    for p in cluster_paths:
        try:
            values, failures = evaluate(p, args.width_mm, args.min_pt)
            scored.append((os.path.basename(p), values, failures))
        except Exception as exc:
            print("  %s: FAILED to score (%s)" % (os.path.basename(p), exc))

    keys = ["axis_aligned", "crossings_per_edge", "longest_run_ratio",
            "min_separation_ratio", "label_overlaps", "label_on_node",
            "label_on_edge", "hairball_index", "occupancy", "aspect_ratio",
            "label_pt", "title_collisions", "canvas_overflow"]

    print("=== cluster maps (n=%d) ===" % len(scored))
    print("  %-22s %8s %8s %8s   %s" % ("gate", "p10", "median", "p90", "failing"))
    for key in keys:
        vals = sorted(v[key] for _, v, _ in scored if key in v)
        if not vals:
            continue
        n = len(vals)
        p10, med, p90 = vals[int(0.1 * n)], vals[n // 2], vals[int(0.9 * (n - 1))]
        bad = sum(1 for _, _, f in scored if key in f)
        flag = "!" if bad else " "
        print("%s %-22s %8.3f %8.3f %8.3f   %d/%d (%.0f%%)"
              % (flag, key, p10, med, p90, bad, n, 100.0 * bad / n))

    worst = sorted(scored, key=lambda r: -len(r[2]))[:args.top]
    print("\n  worst maps by failed-gate count:")
    for name, values, failures in worst:
        if not failures:
            continue
        print("    %-52s %d gates: %s" % (name[:52], len(failures), ", ".join(sorted(failures))))

    total_fail = sum(1 for _, _, f in scored if f)
    print("\n  %d/%d cluster maps pass every gate" % (len(scored) - total_fail, len(scored)))

    combined_ok = True
    if combined_path:
        values, failures = evaluate(combined_path, args.width_mm, args.min_pt)
        print("\n=== combined map: %s ===" % os.path.basename(combined_path))
        for key in keys:
            if key not in values:
                continue
            flag = "!" if key in failures else " "
            print("%s   %-22s %12.3f" % (flag, key, values[key]))
        combined_ok = not failures
        print("\n  combined map %s (%d gates failing)"
              % ("PASSES" if combined_ok else "FAILS", len(failures)))
    else:
        combined_ok = False
        print("\n=== combined map MISSING - run with --generate ===")

    ok = combined_ok and total_fail == 0
    print("\nRESULT: %s" % ("all gates pass" if ok else "gates failing - not ready"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
