"""Derive the acceptance thresholds from curated maps instead of two images.

`layout_algorithm.md` §9 sets every gate and justifies them in one sentence:
these are the numbers separating `templates/t*` from `templates/nt*`. That is a
sample of two. This scores KEGG's 1,010 curated KGML pathways with the same
`metrics.score` used on our output, and reports where MetaCarto's maps fall in
that distribution.

    python -m src.bench.calibrate
    python -m src.bench.calibrate --maps escher_maps_BiGG --sample 200
"""

import argparse
import glob
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.bench import kgml            # noqa: E402
from src.layout import metrics        # noqa: E402


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return {}
    def q(f):
        return ordered[min(len(ordered) - 1, int(f * len(ordered)))]
    return {
        "n": len(ordered),
        "p10": q(0.10),
        "median": statistics.median(ordered),
        "p90": q(0.90),
        "p95": q(0.95),
    }


def score_kegg(directory, limit=None):
    rows = []
    for name, chart in kgml.load_all(directory, limit=limit):
        try:
            values = metrics.score(chart, pitch=180.0)
        except Exception:
            continue
        values["_name"] = name
        values["_reactions"] = len(chart[1]["reactions"])
        values["_metabolites"] = sum(1 for n in chart[1]["nodes"].values()
                                     if n["node_type"] == "metabolite")
        rows.append(values)
    return rows


def score_ours(root, sample=None, seed=0):
    files = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if not os.path.isdir(path):
            continue
        files += [f for f in glob.glob(os.path.join(path, "*.json"))
                  if not f.endswith("model_index.json")]
    if sample and len(files) > sample:
        random.Random(seed).shuffle(files)
        files = files[:sample]
    rows = []
    for path in files:
        try:
            with open(path, encoding="utf-8") as handle:
                blob = json.load(handle)
            values = metrics.score(blob, pitch=180.0)
            values["_metabolites"] = sum(1 for n in blob[1]["nodes"].values()
                                         if n["node_type"] == "metabolite")
            rows.append(values)
        except Exception:
            continue
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kegg", default=os.path.join("data", "kegg"))
    parser.add_argument("--maps", default="escher_maps_BiGG")
    parser.add_argument("--sample", type=int, default=250,
                        help="how many of our maps to score (0 = all)")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap on KGML pathways (0 = all)")
    parser.add_argument("--out", default=os.path.join("benchmarks", "results"),
                        help="where to write the machine-readable result")
    args = parser.parse_args(argv)

    curated = score_kegg(args.kegg, limit=args.limit or None)
    ours = score_ours(args.maps, sample=args.sample or None)
    if not curated:
        parser.error("no scorable KGML pathways in %s" % args.kegg)
    if not ours:
        parser.error("no scorable maps in %s" % args.maps)
    print(f"curated KEGG pathways scored: {len(curated)}")
    print(f"MetaCarto maps scored:        {len(ours)}\n")

    gates = metrics.TARGETS
    header = f"{'metric':22}{'KEGG p10':>10}{'KEGG med':>10}{'KEGG p90':>10}" \
             f"{'ours med':>10}{'current gate':>14}"
    print(header)
    print("-" * len(header))
    for key in kgml.comparable_metrics():
        k = quantiles([r[key] for r in curated])
        o = quantiles([r[key] for r in ours])
        low, high = gates.get(key, (None, None))
        gate = ("> %.2f" % low) if high is None else (
            "< %.2f" % high if low is None else "%.2f-%.2f" % (low, high))
        print(f"{key:22}{k['p10']:10.3f}{k['median']:10.3f}{k['p90']:10.3f}"
              f"{o['median']:10.3f}{gate:>14}")

    print("\nNot comparable (reported for completeness, do not read across):")
    for key in kgml.NOT_COMPARABLE:
        if key not in curated[0]:
            continue
        k = quantiles([r[key] for r in curated])
        o = quantiles([r[key] for r in ours])
        print(f"  {key:22} KEGG median {k['median']:8.3f}   ours {o['median']:8.3f}")

    if args.out:
        import subprocess
        try:
            sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                          stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            sha = "unknown"
        os.makedirs(args.out, exist_ok=True)
        payload = {
            "commit": sha,
            "kegg_pathways_scored": len(curated),
            "our_maps_scored": len(ours),
            "comparable": {k: {"kegg": quantiles([r[k] for r in curated]),
                               "ours": quantiles([r[k] for r in ours])}
                           for k in kgml.comparable_metrics()},
            "not_comparable": {k: {"kegg": quantiles([r[k] for r in curated]),
                                   "ours": quantiles([r[k] for r in ours])}
                               for k in kgml.NOT_COMPARABLE if k in curated[0]},
        }
        target = os.path.join(args.out, "kegg_calibration.json")
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print("wrote %s (commit %s)" % (target, sha))

    # Banded, because the pooled comparison is confounded by drawing size.
    #
    # hairball_index is max-bin over mean-occupied-bin on a fixed 20x20 grid,
    # so any drawing with fewer than ~400 well-spread nodes puts one node per
    # bin and scores exactly 1.0 by construction. KEGG pathways carry a median
    # of ~47 metabolites against our ~260, because we draw a duplicate cofactor
    # stub per participant and KEGG does not. Comparing pooled medians reports
    # that size difference as a quality difference.
    bands = [(0, 40), (40, 70), (70, 110), (110, 200), (200, 10 ** 9)]
    print()
    print("hairball_index by drawing size (metabolite nodes):")
    print("  %-12s%20s%20s" % ("band", "KEGG", "ours"))
    for low, high in bands:
        k = [r["hairball_index"] for r in curated
             if low <= r.get("_metabolites", 0) < high]
        o = [r["hairball_index"] for r in ours
             if low <= r.get("_metabolites", 0) < high]
        label = ("%d-%d" % (low, high)) if high < 10 ** 9 else ("%d+" % low)
        ktxt = ("%.3f (n=%d)" % (statistics.median(k), len(k))) if k else "none"
        otxt = ("%.3f (n=%d)" % (statistics.median(o), len(o))) if o else "none"
        print("  %-12s%20s%20s" % (label, ktxt, otxt))
    print("  A band with no KEGG pathways cannot support a comparison.")

    print()
    print("What the curated 90th percentile would imply if adopted as gates.")
    print("Nothing below has been applied: metrics.TARGETS is unchanged.")
    for key in kgml.comparable_metrics():
        k = quantiles([r[key] for r in curated])
        low, high = gates.get(key, (None, None))
        if high is not None and low is None:
            print(f"  {key:22} < {k['p90']:.3f}   (currently < {high})")
        elif low is not None and high is not None:
            print(f"  {key:22} {k['p10']:.3f} - {k['p90']:.3f}"
                  f"   (currently {low}-{high})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
