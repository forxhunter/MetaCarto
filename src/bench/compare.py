"""Compare MetaCarto against the layout engines the prior tools use.

    python -m src.bench.compare --model e_coli_core
    python -m src.bench.compare --model Recon3D --max-cluster 120 --limit 20

Every method draws the same compound graph and is rendered by the same
`build_escher_map`, so the only variable is where the nodes go. See
`adapters.py` for the two ways that comparison is deliberately tilted against
MetaCarto.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.bench import adapters                       # noqa: E402
from src.layout import metrics                       # noqa: E402
from src.layout.compound import build_compound_graph  # noqa: E402
from src.layout.direction import orient_compound_graph  # noqa: E402
from src.layout.engine import layout_reactions       # noqa: E402
from src.layout.render import build_escher_map       # noqa: E402

# Only the metrics that survive being compared across methods. `axis_aligned`
# and `longest_run_ratio` describe edge routing, and no route hints are passed
# here, so they would measure the shared renderer rather than the placement.
REPORTED = ("crossings_per_edge", "hairball_index", "aspect_ratio",
            "occupancy", "min_separation_ratio")

METHODS = ("metacarto", "dot", "neato", "fdp", "spring", "kamada_kawai")


def draw(model, reactions, name, method, use_fba=True):
    """One map, drawn by one method. Returns (escher_map, seconds)."""
    start = time.perf_counter()
    if method == "metacarto":
        result = layout_reactions(model, reactions, name, use_fba=use_fba)
        if result is None:
            return None, 0.0
        # Rendered from positions alone, with no route hints, so MetaCarto is
        # judged on placement under the same renderer as everything else.
        pos = {n: p for n, p in result.pos.items()
               if not str(n).startswith("__dummy__")}
        chart = build_escher_map(result.cgraph, pos, name)
        return chart, time.perf_counter() - start

    cgraph = build_compound_graph(model, reactions)
    if cgraph.D.number_of_nodes() == 0:
        return None, 0.0
    orient_compound_graph(cgraph, model=model, use_fba=use_fba, verbose=False)
    pos = adapters.place(cgraph.D, method)
    if not pos:
        return None, 0.0
    chart = build_escher_map(cgraph, pos, name)
    return chart, time.perf_counter() - start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="e_coli_core")
    parser.add_argument("--model-dir", default=os.path.join("data", "bigg", "models"))
    parser.add_argument("--max-cluster", type=int, default=None)
    parser.add_argument("--min-cluster", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="cap on the number of clusters drawn (0 = all)")
    parser.add_argument("--group-function", action="store_true")
    parser.add_argument("--no-fba", action="store_true")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--out", default=os.path.join("benchmarks", "results"))
    args = parser.parse_args(argv)

    import cobra
    from src.layout.compound import compute_cofactor_scores
    from src.layout.decompose import clusters
    import layout_v2

    path = None
    for ext in (".xml", ".json"):
        candidate = os.path.join(args.model_dir, args.model + ext)
        if os.path.exists(candidate):
            path = candidate
            break
    if path is None:
        parser.error("no model file for %s in %s" % (args.model, args.model_dir))

    print("loading %s ..." % path)
    model = (cobra.io.read_sbml_model(path) if path.endswith(".xml")
             else cobra.io.load_json_model(path))
    size_limits = {}
    if args.max_cluster:
        size_limits["max_size"] = args.max_cluster
    if args.min_cluster:
        size_limits["min_size"] = args.min_cluster
    groups = clusters(model, compute_cofactor_scores(model), **size_limits)
    if args.group_function:
        groups = layout_v2.merge_by_function(groups, max_size=args.max_cluster)
    names = sorted(groups)
    if args.limit:
        names = names[:args.limit]
    print("%s: %d clusters, drawing %d" % (args.model, len(groups), len(names)))

    methods = [m for m in args.methods.split(",") if m]
    ready = adapters.available()
    for method in list(methods):
        if method != "metacarto" and not ready.get(method):
            print("  skipping %s (not installed)" % method)
            methods.remove(method)

    rows = {m: [] for m in methods}
    timings = {m: [] for m in methods}
    for i, name in enumerate(names, 1):
        reactions = groups[name]
        for method in methods:
            try:
                chart, seconds = draw(model, reactions, name, method,
                                      use_fba=not args.no_fba)
            except Exception as exc:
                print("  %-14s %-34s FAILED %s" % (method, name[:34], exc))
                continue
            if chart is None:
                continue
            try:
                values = metrics.score(chart, pitch=180.0)
            except Exception as exc:
                print("  %-14s %-34s SCORE FAILED %s" % (method, name[:34], exc))
                continue
            values["_cluster"] = name
            rows[method].append(values)
            timings[method].append(seconds)
        print("  [%d/%d] %s" % (i, len(names), name[:48]))

    print()
    header = "%-14s%8s" % ("method", "maps") + "".join(
        "%20s" % k[:19] for k in REPORTED) + "%10s" % "sec/map"
    print(header)
    print("-" * len(header))
    for method in methods:
        if not rows[method]:
            continue
        cells = "".join("%20.3f" % statistics.median(r[k] for r in rows[method])
                        for k in REPORTED)
        print("%-14s%8d%s%10.2f" % (method, len(rows[method]), cells,
                                    statistics.median(timings[method])))

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "unknown"
    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, "baselines_%s.json" % args.model)
    payload = {
        "commit": sha,
        "model": args.model,
        "clusters": len(names),
        "methods": {
            m: {
                "maps": len(rows[m]),
                "median_seconds": (statistics.median(timings[m])
                                   if timings[m] else None),
                "metrics": {k: {
                    "median": statistics.median(r[k] for r in rows[m]),
                    "mean": statistics.fmean(r[k] for r in rows[m]),
                } for k in REPORTED},
                "per_map": [{"cluster": r["_cluster"],
                             **{k: r[k] for k in REPORTED}} for r in rows[m]],
            }
            for m in methods if rows[m]
        },
    }
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("\nwrote %s (commit %s)" % (target, sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
