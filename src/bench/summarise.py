"""Pool the per-model baseline runs and test the differences properly.

Every number reported so far has been a bare median. For a paper the claim
"MetaCarto beats X" needs a paired test on the same maps, a correction for
testing several metrics at once, and an effect size -- a significant difference
of no practical size is not a result.

    python -m src.bench.summarise

Reads benchmarks/results/baselines_*.json and writes baseline_summary.json.
"""

import argparse
import glob
import json
import math
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# Lower is better for these; higher is better for the rest.
LOWER_BETTER = {"crossings_per_edge", "hairball_index"}
# Distance from 1.0 is what matters, not the raw value: a map twice as wide as
# tall and one twice as tall as wide are equally far from square.
BALANCED = {"aspect_ratio"}


def cliffs_delta(a, b):
    """Non-parametric effect size: P(a > b) - P(a < b), in [-1, 1].

    Computed by sorting rather than the O(n^2) double loop, which matters at a
    few thousand pairs.
    """
    if not a or not b:
        return 0.0
    ordered = sorted(b)
    n = len(ordered)
    greater = less = 0
    import bisect
    for value in a:
        greater += bisect.bisect_left(ordered, value)
        less += n - bisect.bisect_right(ordered, value)
    total = len(a) * n
    return (greater - less) / total if total else 0.0


def wilcoxon(pairs):
    """Paired Wilcoxon signed-rank p-value, or None if scipy is unavailable."""
    try:
        from scipy.stats import wilcoxon as _w
    except ImportError:                                   # pragma: no cover
        return None
    a = [x for x, y in pairs if x != y]
    b = [y for x, y in pairs if x != y]
    if len(a) < 5:
        return None
    try:
        return float(_w(a, b).pvalue)
    except Exception:
        return None


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    indexed = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    out = list(pvalues)
    m = len(indexed)
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        adjusted = min(1.0, (m - rank) * p)
        running = max(running, adjusted)
        out[i] = running
    return out


def transform(key, value):
    """Put every metric on a 'lower is better' scale."""
    if key in BALANCED:
        return abs(math.log(value)) if value > 0 else float("inf")
    return value if key in LOWER_BETTER else -value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=os.path.join("benchmarks", "results"))
    parser.add_argument("--reference", default="metacarto")
    args = parser.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.results, "baselines_*.json")))
    if not files:
        parser.error("no baselines_*.json in %s" % args.results)

    # metric -> method -> {cluster key: value}
    pooled, models = {}, []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            blob = json.load(handle)
        models.append(blob["model"])
        for method, payload in blob["methods"].items():
            for row in payload["per_map"]:
                key = (blob["model"], row["cluster"])
                for metric, value in row.items():
                    if metric == "cluster":
                        continue
                    pooled.setdefault(metric, {}).setdefault(method, {})[key] = value

    methods = sorted({m for per in pooled.values() for m in per})
    methods.remove(args.reference)
    methods.insert(0, args.reference)
    metrics_order = [m for m in ("crossings_per_edge", "hairball_index",
                                 "aspect_ratio", "occupancy",
                                 "min_separation_ratio") if m in pooled]

    print("models: %s" % ", ".join(models))
    print("maps per method: %d\n" % len(next(iter(pooled.values()))[args.reference]))

    summary = {"models": models, "reference": args.reference, "metrics": {}}
    for metric in metrics_order:
        per = pooled[metric]
        ref = per[args.reference]
        print("%s" % metric)
        print("  %-14s%10s%12s%12s%10s" %
              ("method", "median", "vs ref", "p (Holm)", "delta"))
        rows, praw = [], []
        for method in methods:
            values = per.get(method, {})
            shared = [k for k in ref if k in values]
            med = statistics.median(values[k] for k in shared) if shared else float("nan")
            if method == args.reference:
                rows.append((method, med, None, None))
                praw.append(None)
                continue
            pairs = [(transform(metric, ref[k]), transform(metric, values[k]))
                     for k in shared]
            praw.append(wilcoxon(pairs))
            delta = cliffs_delta([p[0] for p in pairs], [p[1] for p in pairs])
            rows.append((method, med, med - rows[0][1], delta))
        adjusted = holm(praw)
        for (method, med, diff, delta), p in zip(rows, adjusted):
            if method == args.reference:
                print("  %-14s%10.3f%12s%12s%10s" % (method, med, "--", "--", "--"))
            else:
                ptxt = "n/a" if p is None else ("%.1e" % p if p < 1e-3 else "%.3f" % p)
                print("  %-14s%10.3f%+12.3f%12s%10.2f" % (method, med, diff, ptxt, delta))
        summary["metrics"][metric] = {
            method: {"median": med, "p_holm": p, "cliffs_delta": delta}
            for (method, med, _d, delta), p in zip(rows, adjusted)
        }
        print()

    print("Cliff's delta is computed on a lower-is-better scale, so a negative")
    print("value means %s is better than that method." % args.reference)

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "unknown"
    summary["commit"] = sha
    target = os.path.join(args.results, "baseline_summary.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("\nwrote %s (commit %s)" % (target, sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
