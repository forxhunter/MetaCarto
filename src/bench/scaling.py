"""How does the method scale, and where does the time actually go?

There was no timing instrumentation anywhere in the project -- `perf_counter`,
`timeit` and `time.time` appear nowhere in src/ -- so "it handles Recon3D"
was an assertion with no number behind it. A referee will ask for the number.

This times each stage separately, because the total is not the interesting
part. The pipeline is:

    build_compound_graph     chemistry, once per cluster
    orient_compound_graph    pFBA once per model, then per cluster
    find/contract rings      cycle detection
    layered_layout           the Sugiyama pass
    pack + expand            component packing and ring expansion
    build_escher_map         rendering

Reported per cluster against reaction count, so the growth is visible rather
than inferred from one aggregate.

    python -m src.bench.scaling --models e_coli_core,iYO844,iJO1366,RECON1
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

from src.layout import metrics                        # noqa: E402
from src.layout.compound import (build_compound_graph,  # noqa: E402
                                 compute_cofactor_scores)
from src.layout.direction import orient_compound_graph  # noqa: E402
from src.layout.engine import layout_reactions        # noqa: E402


def time_model(model, groups, peak=True):
    """Per-cluster timings and sizes for one model."""
    rows = []
    for name, reactions in groups.items():
        sizes = {"reactions": len(reactions)}
        t0 = time.perf_counter()
        cgraph = build_compound_graph(model, reactions)
        t1 = time.perf_counter()
        if cgraph.D.number_of_nodes() == 0:
            continue
        orient_compound_graph(cgraph, model=model, use_fba=True, verbose=False)
        t2 = time.perf_counter()
        result = layout_reactions(model, reactions, name, use_fba=True)
        t3 = time.perf_counter()
        if result is None:
            continue
        sizes.update({
            "compound_nodes": cgraph.D.number_of_nodes(),
            "compound_edges": cgraph.D.number_of_edges(),
            "drawn_nodes": len(result.escher_map[1]["nodes"]),
        })
        rows.append({
            **sizes,
            "cluster": name,
            "compound_graph_s": t1 - t0,
            "orientation_s": t2 - t1,
            "layout_and_render_s": t3 - t2,
            "total_s": t3 - t0,
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="e_coli_core,iYO844,iJO1366,RECON1")
    parser.add_argument("--model-dir", default=os.path.join("data", "bigg", "models"))
    parser.add_argument("--out", default=os.path.join("benchmarks", "results"))
    args = parser.parse_args(argv)

    import cobra
    from src.layout.decompose import clusters

    everything = {}
    print("%-12s%10s%9s%10s%12s%12s%12s"
          % ("model", "reactions", "maps", "load s", "decompose s",
             "draw s tot", "s / map"))
    for name in args.models.split(","):
        path = None
        for ext in (".json", ".xml"):
            candidate = os.path.join(args.model_dir, name + ext)
            if os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            print("%-12s  (no model file)" % name)
            continue

        t0 = time.perf_counter()
        model = (cobra.io.load_json_model(path) if path.endswith(".json")
                 else cobra.io.read_sbml_model(path))
        load_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        groups = clusters(model, compute_cofactor_scores(model))
        decompose_s = time.perf_counter() - t0

        rows = time_model(model, groups)
        draw_s = sum(r["total_s"] for r in rows)
        everything[name] = {
            "reactions": len(model.reactions),
            "maps": len(rows),
            "load_s": load_s,
            "decompose_s": decompose_s,
            "draw_total_s": draw_s,
            "per_map": rows,
        }
        print("%-12s%10d%9d%10.1f%12.1f%12.1f%12.3f"
              % (name, len(model.reactions), len(rows), load_s, decompose_s,
                 draw_s, draw_s / max(len(rows), 1)))

    print()
    print("where the drawing time goes (median share per cluster)")
    print("  %-24s%12s" % ("stage", "share"))
    stages = ("compound_graph_s", "orientation_s", "layout_and_render_s")
    pooled = [r for v in everything.values() for r in v["per_map"]]
    if pooled:
        for stage in stages:
            share = statistics.median(r[stage] / r["total_s"]
                                      for r in pooled if r["total_s"] > 0)
            print("  %-24s%11.0f%%" % (stage, 100 * share))

    # Growth: seconds per map against cluster size.
    if pooled:
        print()
        print("draw time by cluster size")
        print("  %-18s%10s%14s" % ("reactions", "clusters", "median s"))
        for low, high in ((0, 20), (20, 50), (50, 100), (100, 10 ** 9)):
            band = [r for r in pooled if low <= r["reactions"] < high]
            if not band:
                continue
            label = "%d-%d" % (low, high) if high < 10 ** 9 else "%d+" % low
            print("  %-18s%10d%14.3f"
                  % (label, len(band),
                     statistics.median(r["total_s"] for r in band)))

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "unknown"
    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, "scaling.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump({"commit": sha, "models": everything}, handle, indent=2)
    print("\nwrote %s (commit %s)" % (target, sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
