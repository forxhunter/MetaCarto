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

import contextlib                                    # noqa: E402

from src.bench import adapters                       # noqa: E402
from src.layout import metrics                       # noqa: E402
from src.layout import render as _render             # noqa: E402
from src.layout.compound import build_compound_graph  # noqa: E402
from src.layout.direction import orient_compound_graph  # noqa: E402
from src.layout.engine import layout_reactions       # noqa: E402
from src.layout.render import build_escher_map       # noqa: E402

# Only the metrics that survive being compared across methods. `axis_aligned`
# and `longest_run_ratio` describe edge routing, and no route hints are passed
# here, so they would measure the shared renderer rather than the placement.
# axis_aligned is back in, and is meaningful now. It was excluded while the
# shared orthogonal router was applied to everyone, because it then measured
# the router: a kamada-kawai drawing scored 0.97. With straight edges it
# measures what it should -- the share of edges whose endpoints the placement
# actually lined up -- and force-directed layouts score ~0.00 as they should.
REPORTED = ("crossings_per_graph_edge", "axis_aligned", "hairball_index",
            "aspect_ratio", "occupancy", "min_separation_ratio")

METHODS = ("metacarto", "dot", "neato", "fdp", "spring", "kamada_kawai")


@contextlib.contextmanager
def straight_edges():
    """Draw every edge as a straight segment, for every method.

    `render._orthogonal_path` returns a vertical-horizontal-vertical dogleg
    whenever an edge's endpoints are not already aligned, adding two bend
    multimarkers and two segments. Brandes-Koepf aligns MetaCarto's nodes so it
    mostly takes the 2-point branch; continuous force-directed coordinates
    never do. Two things went wrong as a result.

    The shared renderer *orthogonalised the competitors*: on iAF1260 a
    kamada-kawai drawing scored 0.969 axis-aligned, which is not a
    kamada-kawai drawing and is not what MetExploreViz would put on screen.
    And the segment count became method-dependent -- a force-directed drawing
    of the same graph carried ~20% more segments than a layered one -- so
    `crossings_per_edge` was divided by a denominator that grew with how badly
    a method placed its nodes.

    Straight edges for everyone removes both. It also removes MetaCarto's
    orthogonal routing, which is a real feature of the shipped maps and is not
    represented here at all: this measures placement, not the product.
    """
    original = _render._orthogonal_path
    _render._orthogonal_path = lambda p0, p1: [p0, p1]
    try:
        yield
    finally:
        _render._orthogonal_path = original


def draw(model, reactions, name, method, use_fba=True):
    """One map, drawn by one method. Returns (chart, seconds, graph_edges).

    Every method -- MetaCarto included -- goes through `adapters._normalise`.
    That symmetry is not cosmetic. `min_separation_ratio` is
    `min_pairwise_distance / pitch` with pitch = 180, `_normalise` puts every
    layout's *median* nearest-neighbour distance at exactly 180, and the
    minimum can never exceed the median. So any normalised layout is capped
    below 1.0 by construction, and the first version of this exempted
    MetaCarto: it scored up to 1.78 while no competitor could pass ~0.9. The
    "decisive win on node separation" was that exemption, not a property of
    the layouts.
    """
    start = time.perf_counter()
    if method == "metacarto":
        result = layout_reactions(model, reactions, name, use_fba=use_fba,
                                  render=False)
        if result is None:
            return None, 0.0, 0
        pos = {n: p for n, p in result.pos.items()
               if not str(n).startswith("__dummy__")}
        pos = adapters.normalise(pos)
        with straight_edges():
            chart = build_escher_map(result.cgraph, pos, name)
        return chart, time.perf_counter() - start, result.cgraph.D.number_of_edges()

    cgraph = build_compound_graph(model, reactions)
    if cgraph.D.number_of_nodes() == 0:
        return None, 0.0, 0
    orient_compound_graph(cgraph, model=model, use_fba=use_fba, verbose=False)
    pos = adapters.place(cgraph.D, method)
    if not pos:
        return None, 0.0, 0
    with straight_edges():
        chart = build_escher_map(cgraph, pos, name)
    return chart, time.perf_counter() - start, cgraph.D.number_of_edges()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="e_coli_core")
    parser.add_argument("--model-dir", default=os.path.join("data", "bigg", "models"))
    parser.add_argument("--max-cluster", type=int, default=None)
    parser.add_argument("--min-cluster", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="draw a random sample of this many clusters "
                             "(0 = every cluster, which is the default and "
                             "what should be reported)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for --limit sampling, so it is reproducible")
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
    population = sorted(groups)
    names = population
    if args.limit and args.limit < len(population):
        # A seeded random sample, not the alphabetical head.
        #
        # This used to be `population[:limit]`, which for iAF1260 meant every
        # cluster from "Acetaldehyde transport periplasm" to "N-c 3" and not
        # one from O-Z -- no oxidative phosphorylation, no pentose phosphate,
        # no purine or pyrimidine metabolism. "We evaluated on the
        # alphabetically first 40 clusters" is not a sampling statement anyone
        # should accept, and the saved result recorded only the sample size, so
        # it did not even disclose that sampling had happened.
        import random as _random
        names = sorted(_random.Random(args.seed).sample(population, args.limit))
    sampled = len(names) < len(population)
    print("%s: %d clusters, drawing %d%s" % (
        args.model, len(population), len(names),
        " (random sample, seed %d)" % args.seed if sampled else " (all)"))

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
                chart, seconds, graph_edges = draw(model, reactions, name,
                                                   method,
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
            # Per *graph* edge, not per rendered segment. The shared
            # renderer emits a 4-point dogleg with two extra multimarkers
            # whenever an edge is not already axis-aligned, so a force-directed
            # drawing of the same graph carries ~20% more segments than a
            # layered one. Dividing by segments makes the denominator depend on
            # the method being measured.
            segs = len(metrics._straight_segments(chart[1]))
            raw = values["crossings_per_edge"] * max(segs, 1)
            values["crossings_per_graph_edge"] = raw / max(graph_edges, 1)
            values["_graph_edges"] = graph_edges
            values["_segments"] = segs
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
        "clusters_in_model": len(population),
        "clusters_drawn": len(names),
        "sampled": sampled,
        "seed": args.seed if sampled else None,
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
