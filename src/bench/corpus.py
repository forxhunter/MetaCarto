"""Score every map in the released collection, not a sample of it.

The baseline comparison runs on 676 maps from eight models because each of the
six methods has to draw each of them. Nothing forces the same restriction on
\\mytool's own output: all 2621 published maps can simply be measured, and a
claim about the collection should be made over the collection.

Reports, per metric, the median and the 10th/90th percentiles over every map,
and the share of maps inside the acceptance band from `layout_algorithm.md`
S9. Print legibility is included because it is the measure that decides
whether a map is usable as a figure, and it is the one the large merged maps
fail.

    python -m src.bench.corpus
    python -m src.bench.corpus --root data/bigg --out benchmarks/results
"""

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.iterate import label_pt                   # noqa: E402
from src.layout import metrics                         # noqa: E402

SKIP_DIRS = {"models", "test_model"}

REPORTED = ("crossings_per_edge", "axis_aligned", "longest_run_ratio",
            "min_separation_ratio", "label_overlaps", "label_on_node",
            "label_on_edge", "hairball_index", "occupancy", "aspect_ratio")

# The width a figure is placed at, for the print-legibility column.
FIGURE_MM = 180.0
LEGIBLE_PT = 5.0


def percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def within(value, band):
    low, high = band
    if low is not None and value < low:
        return False
    if high is not None and value > high:
        return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.join("data", "bigg"))
    parser.add_argument("--out", default=os.path.join("benchmarks", "results"))
    args = parser.parse_args(argv)

    pooled = {key: [] for key in REPORTED}
    pooled["label_pt"] = []
    per_model = {}
    maps = reactions = failed = 0

    for model_dir in sorted(glob.glob(os.path.join(args.root, "*"))):
        model = os.path.basename(model_dir)
        if not os.path.isdir(model_dir) or model in SKIP_DIRS:
            continue
        rows = 0
        for path in sorted(glob.glob(os.path.join(model_dir, "*.json"))):
            try:
                with open(path, encoding="utf-8") as handle:
                    doc = json.load(handle)
                values = metrics.score(doc)
            except Exception:                           # noqa: BLE001
                failed += 1
                continue
            if not isinstance(doc, list) or len(doc) < 2:
                continue
            for key in REPORTED:
                pooled[key].append(values[key])
            pooled["label_pt"].append(label_pt(doc[1], FIGURE_MM))
            reactions += values["reactions"]
            maps += 1
            rows += 1
        if rows:
            per_model[model] = rows

    if not maps:
        raise SystemExit("no maps found under %s" % args.root)

    summary = {}
    print("%-22s%10s%10s%10s%12s" % ("metric", "p10", "median", "p90", "in band"))
    for key in REPORTED:
        values = pooled[key]
        band = metrics.TARGETS.get(key)
        share = (sum(1 for v in values if within(v, band)) / len(values)
                 if band else None)
        summary[key] = {
            "median": statistics.median(values),
            "p10": percentile(values, 0.10),
            "p90": percentile(values, 0.90),
            "in_band": share,
            "n": len(values),
        }
        print("%-22s%10.3f%10.3f%10.3f%11s"
              % (key, summary[key]["p10"], summary[key]["median"],
                 summary[key]["p90"],
                 "%.1f%%" % (100 * share) if share is not None else "--"))

    pts = pooled["label_pt"]
    legible = sum(1 for v in pts if v >= LEGIBLE_PT) / len(pts)
    summary["label_pt"] = {
        "median": statistics.median(pts),
        "p10": percentile(pts, 0.10),
        "p90": percentile(pts, 0.90),
        "in_band": legible,
        "n": len(pts),
    }
    print("%-22s%10.2f%10.2f%10.2f%11s"
          % ("label_pt at %.0f mm" % FIGURE_MM, summary["label_pt"]["p10"],
             summary["label_pt"]["median"], summary["label_pt"]["p90"],
             "%.1f%%" % (100 * legible)))

    print()
    print("%d maps over %d models, %d reactions drawn; %d unreadable"
          % (maps, len(per_model), reactions, failed))

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                   # noqa: BLE001
        sha = "unknown"
    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, "corpus.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump({"commit": sha, "maps": maps, "models": len(per_model),
                   "reactions": reactions, "figure_width_mm": FIGURE_MM,
                   "legible_pt": LEGIBLE_PT, "metrics": summary,
                   "maps_per_model": per_model}, handle, indent=2)
    print("wrote %s (commit %s)" % (target, sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
