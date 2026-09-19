"""Pool the per-model baseline runs and test the differences properly.

Rewritten after an audit found four defects that changed conclusions:

  * Holm was applied within each metric (5 tests) while 25 were being run, so
    at least one "significant" result did not survive the real family.
  * The effect size was Cliff's delta, computed on the two *marginal*
    distributions while the test was paired. On data where the reference wins
    every pair it returned -0.10 instead of -1.0.
  * `occupancy` was scored as higher-is-better although `metrics.TARGETS`
    defines it as the band (0.10, 0.85), so 0.99 -- a failing value -- scored
    better than 0.40. `min_separation_ratio` had the same shape of error.
  * Only the favourable rows were being reported. Every comparison is printed
    now, wins and losses, and the counts are summarised at the end.

    python -m src.bench.summarise

Reads benchmarks/results/baselines_*.json and writes baseline_summary.json.

What this does NOT measure
--------------------------
The benchmark corpus is not the shipped corpus. It draws the raw
`decompose.clusters` output, while `escher_maps_BiGG` publishes maps merged by
`merge_by_function`, so the two populations differ -- e_coli_core is 6 clusters
here and 3 published maps there. And every method is rendered with straight
edges, so MetaCarto's orthogonal routing, which the shipped maps do use, is
absent. Numbers here describe placement quality under identical conditions;
they are not a description of the product.
"""

import argparse
import glob
import json
import math
import os
import random
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# Lower is better.
LOWER_BETTER = {"crossings_per_edge", "crossings_per_graph_edge",
                "hairball_index"}
HIGHER_BETTER = {"axis_aligned"}
# Distance from 1.0 is what matters: a map twice as wide as tall and one twice
# as tall as wide are equally far from square.
BALANCED = {"aspect_ratio"}
# Targets that are bands or bounds, scored by distance *outside* the target and
# 0 anywhere inside it. Taken from metrics.TARGETS.
BANDS = {"occupancy": (0.10, 0.85), "min_separation_ratio": (1.0, None)}

ORDER = ("crossings_per_graph_edge", "axis_aligned", "hairball_index",
         "aspect_ratio", "occupancy", "min_separation_ratio")


def rank_biserial(pairs):
    """Matched-pairs rank-biserial correlation, in [-1, 1].

    The effect size that matches the Wilcoxon signed-rank test actually run:
    computed from the signed ranks of the paired differences, so -1.0 means
    the reference wins every pair. Negative means the reference is better,
    because `transform()` puts everything on a lower-is-better scale.
    """
    diffs = [a - b for a, b in pairs if a != b]
    if not diffs:
        return 0.0
    ranked = sorted(range(len(diffs)), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * len(diffs)
    i = 0
    while i < len(ranked):
        j = i
        while (j + 1 < len(ranked)
               and abs(diffs[ranked[j + 1]]) == abs(diffs[ranked[i]])):
            j += 1
        shared = (i + j) / 2.0 + 1.0          # average rank across ties
        for k in range(i, j + 1):
            ranks[ranked[k]] = shared
        i = j + 1
    total = sum(ranks)
    positive = sum(r for d, r in zip(diffs, ranks) if d > 0)
    negative = sum(r for d, r in zip(diffs, ranks) if d < 0)
    return (positive - negative) / total if total else 0.0


def wilcoxon(pairs):
    """Paired Wilcoxon signed-rank p-value, and how many pairs it used."""
    try:
        from scipy.stats import wilcoxon as _w
    except ImportError:                                   # pragma: no cover
        return None, 0
    a = [x for x, y in pairs if x != y]
    b = [y for x, y in pairs if x != y]
    if len(a) < 5:
        return None, len(a)
    try:
        return float(_w(a, b).pvalue), len(a)
    except Exception:
        return None, len(a)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, order preserved, None passed through."""
    indexed = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    out = list(pvalues)
    m = len(indexed)
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        out[i] = running
    return out


def transform(key, value):
    """Put every metric on a 'lower is better' scale."""
    if key in BALANCED:
        return abs(math.log(value)) if value > 0 else float("inf")
    if key in BANDS:
        low, high = BANDS[key]
        return max(0.0,
                   (low - value) if low is not None else 0.0,
                   (value - high) if high is not None else 0.0)
    return value if key in LOWER_BETTER else -value


def bootstrap_ci(diffs, rounds=2000, seed=0):
    """Percentile CI for the median paired difference."""
    if not diffs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(diffs)
    medians = []
    for _ in range(rounds):
        medians.append(statistics.median(diffs[rng.randrange(n)] for _ in range(n)))
    medians.sort()
    return medians[int(0.025 * rounds)], medians[int(0.975 * rounds)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=os.path.join("benchmarks", "results"))
    parser.add_argument("--reference", default="metacarto")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.results, "baselines_*.json")))
    if not files:
        parser.error("no baselines_*.json in %s" % args.results)

    pooled, models = {}, []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            blob = json.load(handle)
        models.append(blob["model"])
        for method, payload in blob["methods"].items():
            for row in payload["per_map"]:
                key = (blob["model"], row["cluster"])
                for metric, value in row.items():
                    if metric.startswith("_") or metric == "cluster":
                        continue
                    pooled.setdefault(metric, {}).setdefault(method, {})[key] = value

    methods = sorted({m for per in pooled.values() for m in per})
    if args.reference not in methods:
        parser.error("reference %r not present; have %s"
                     % (args.reference, ", ".join(methods)))
    methods.remove(args.reference)
    methods.insert(0, args.reference)
    metric_names = [m for m in ORDER if m in pooled]

    print("models : %s" % ", ".join(models))
    print("maps   : %d per method" % len(pooled[metric_names[0]][args.reference]))
    print("tests  : %d metrics x %d methods = %d, Holm-corrected together\n"
          % (len(metric_names), len(methods) - 1,
             len(metric_names) * (len(methods) - 1)))

    # Every test first, so Holm sees the family that was actually run.
    cells, praw = [], []
    for metric in metric_names:
        per = pooled[metric]
        ref = per[args.reference]
        for method in methods:
            if method == args.reference:
                continue
            values = per.get(method, {})
            shared = [k for k in ref if k in values]
            pairs = [(transform(metric, ref[k]), transform(metric, values[k]))
                     for k in shared]
            p, used = wilcoxon(pairs)
            diffs = [a - b for a, b in pairs]
            cells.append({
                "metric": metric, "method": method,
                "median": statistics.median(values[k] for k in shared) if shared else float("nan"),
                "ref_median": statistics.median(ref[k] for k in shared) if shared else float("nan"),
                "median_diff": statistics.median(diffs) if diffs else float("nan"),
                "ci": bootstrap_ci(diffs),
                "effect": rank_biserial(pairs),
                "pairs": len(shared), "pairs_used": used,
            })
            praw.append(p)
    adjusted = holm(praw)

    wins = losses = nsig = 0
    summary = {"models": models, "reference": args.reference,
               "tests": len(cells), "metrics": {}}
    for metric in metric_names:
        ref_med = next(c["ref_median"] for c in cells if c["metric"] == metric)
        print("%s   (%s = %.3f)" % (metric, args.reference, ref_med))
        print("  %-14s%10s%12s%22s%9s%10s%7s" %
              ("method", "median", "d(median)", "95% CI of d", "effect",
               "p (Holm)", "pairs"))
        for cell, p in zip(cells, adjusted):
            if cell["metric"] != metric:
                continue
            if p is not None and p < args.alpha:
                verdict = "better" if cell["effect"] < 0 else "WORSE"
                if cell["effect"] < 0:
                    wins += 1
                else:
                    losses += 1
            else:
                verdict = "n.s."
                nsig += 1
            ptxt = "n/a" if p is None else ("%.1e" % p if p < 1e-3 else "%.3f" % p)
            lo, hi = cell["ci"]
            print("  %-14s%10.3f%12.4f   [%8.4f,%8.4f]%9.2f%10s%7d  %s"
                  % (cell["method"], cell["median"], cell["median_diff"],
                     lo, hi, cell["effect"], ptxt, cell["pairs_used"], verdict))
            summary["metrics"].setdefault(metric, {})[cell["method"]] = {
                "median": cell["median"], "median_paired_diff": cell["median_diff"],
                "ci95": list(cell["ci"]), "rank_biserial": cell["effect"],
                "p_holm": p, "pairs": cell["pairs"], "pairs_used": cell["pairs_used"],
                "verdict": verdict,
            }
        print()

    print("d(median) and the effect size are on a lower-is-better scale, so")
    print("negative favours %s. 'WORSE' rows are losses and are reported in" % args.reference)
    print("full -- there are %d of them." % losses)
    print("\nacross all %d tests at alpha=%.2f (Holm): %d better, %d worse, %d not significant"
          % (len(cells), args.alpha, wins, losses, nsig))
    print()
    print("A non-significant row is a failure to detect a difference, not")
    print("evidence of equivalence. Read the 95% CI of the paired difference")
    print("instead: it is the interval the data are consistent with, and it is")
    print("the honest equivalence statement. No arbitrary equivalence margin")
    print("has been invented to turn a null result into a positive claim.")

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "unknown"
    summary["commit"] = sha
    summary["counts"] = {"better": wins, "worse": losses, "not_significant": nsig}
    target = os.path.join(args.results, "baseline_summary.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("\nwrote %s (commit %s)" % (target, sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
