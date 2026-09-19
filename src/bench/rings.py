"""Are cycles actually drawn as cycles?

"The TCA cycle, the urea cycle and the Calvin cycle are detected and placed on
a ring" is one of the five claims the method makes, and it was the one with no
evidence behind it. The ablation study could not test it: disabling ring
detection changed 9 clusters of 100-odd on the models tried, because most
clusters contain no cycle at all, and a null result from a corpus with nothing
to detect says nothing.

This measures the claim directly and only where it applies. For every cycle
`motifs.find_rings` detects, it asks how close the drawn members are to lying
on a common circle:

    radial CV = stdev(distance to centroid) / mean(distance to centroid)

0.0 is a perfect circle. A ring drawn as a chain has members strung along a
line, and the distances to their centroid vary widely, so the CV is large. The
same rings are then measured with ring placement disabled, which is the honest
comparison -- against how those same cycles would otherwise have been drawn,
not against an arbitrary threshold.

    python -m src.bench.rings --model iJO1366
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.bench import ablate                          # noqa: E402
from src.layout import engine as _engine              # noqa: E402
from src.layout.motifs import find_rings              # noqa: E402


def radial_cv(points):
    """Coefficient of variation of the radii about the centroid."""
    if len(points) < 3:
        return None
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    radii = [math.hypot(x - cx, y - cy) for x, y in points]
    mean = sum(radii) / len(radii)
    if mean <= 0:
        return None
    return statistics.pstdev(radii) / mean


def measure(model, reactions, name, variant):
    """[(ring size, radial CV)] for every cycle in this cluster."""
    result = None
    manager = ablate.VARIANTS.get(variant)
    import contextlib
    ctx = (contextlib.nullcontext() if manager is None or variant == "no_fba"
           else manager())
    with ctx:
        # pFBA orientation is required, not optional, for this measurement.
        # The acyclic orientation used when there is no flux removes every
        # directed cycle before find_rings ever runs -- measured across
        # e_coli_core, iJO1366 and iMM904, use_fba=False leaves zero rings in
        # zero clusters, while pFBA leaves the TCA cycle (7 members in
        # e_coli_core) and 3 cycles in iJO1366. So `--no-fba` silently turns
        # ring drawing off, which is worth knowing independently of this test.
        result = _engine.layout_reactions(model, reactions, name,
                                          use_fba=True, render=False)
    if result is None:
        return []

    # Rings are found on the compound graph, independently of whether the
    # layout then placed them on a circle -- so the same cycles are measured
    # in both variants.
    rings = find_rings(result.cgraph.D)
    out = []
    for ring in rings:
        points = [result.pos[m] for m in ring if m in result.pos]
        if len(points) < 3:
            continue
        cv = radial_cv(points)
        if cv is not None:
            out.append((len(ring), cv))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="iJO1366")
    parser.add_argument("--model-dir", default=os.path.join("data", "bigg", "models"))
    parser.add_argument("--out", default=os.path.join("benchmarks", "results"))
    args = parser.parse_args(argv)

    import cobra
    from src.layout.compound import compute_cofactor_scores
    from src.layout.decompose import clusters

    path = os.path.join(args.model_dir, args.model + ".xml")
    if not os.path.exists(path):
        parser.error("no model at %s" % path)
    print("loading %s ..." % path)
    model = cobra.io.read_sbml_model(path)
    groups = clusters(model, compute_cofactor_scores(model))
    names = sorted(groups)

    rows = {"full": [], "no_rings": []}
    per_cluster = []
    for i, name in enumerate(names, 1):
        full = measure(model, groups[name], name, "full")
        if not full:
            continue
        ablated = measure(model, groups[name], name, "no_rings")
        rows["full"].extend(cv for _s, cv in full)
        rows["no_rings"].extend(cv for _s, cv in ablated)
        per_cluster.append({
            "cluster": name, "rings": len(full),
            "full_cv": statistics.median(cv for _s, cv in full),
            "ablated_cv": (statistics.median(cv for _s, cv in ablated)
                           if ablated else None),
        })
    print("clusters containing at least one cycle: %d of %d"
          % (len(per_cluster), len(names)))
    if not per_cluster:
        print("no cycles in this model's clusters; nothing to measure.")
        return 0

    print("cycles measured: %d" % len(rows["full"]))
    print()
    print("%-16s%14s%14s" % ("variant", "median CV", "CV < 0.10"))
    for variant in ("full", "no_rings"):
        values = rows[variant]
        if not values:
            continue
        tight = sum(1 for v in values if v < 0.10) / len(values)
        print("%-16s%14.4f%13.0f%%" % (variant, statistics.median(values),
                                       100 * tight))
    print()
    print("CV 0.0 is a perfect circle. 'CV < 0.10' is the share of cycles")
    print("drawn within 10% radial variation, i.e. recognisably round.")

    worst = sorted(per_cluster, key=lambda r: -r["full_cv"])[:5]
    print("\nleast circular clusters under the full method:")
    for row in worst:
        print("  %-38s %d cycles  CV %.3f" % (row["cluster"][:38], row["rings"],
                                              row["full_cv"]))

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "unknown"
    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, "rings_%s.json" % args.model)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump({"commit": sha, "model": args.model,
                   "clusters_with_cycles": len(per_cluster),
                   "cycles": len(rows["full"]),
                   "median_cv": {k: (statistics.median(v) if v else None)
                                 for k, v in rows.items()},
                   "per_cluster": per_cluster}, handle, indent=2)
    print("\nwrote %s (commit %s)" % (target, sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
