"""Turn each claimed contribution off and measure what it was worth.

The baseline comparison showed MetaCarto is not a better general-purpose
layout engine than Graphviz dot. So the case for the method rests on the
domain-specific parts -- the ones dot has no equivalent of -- and a claim that
a component matters is only worth making if removing it degrades something
measurable.

Each ablation disables exactly one component and redraws the same clusters.
Everything else, including the straight-edge rendering used for the baselines,
is held fixed, and each variant is paired against the full method on the same
cluster.

    python -m src.bench.ablate --model iJO1366
    python -m src.bench.ablate --model iJO1366 --only no_rings,no_bk

An ablation that changes nothing is reported as changing nothing. That is a
result about the component, not a failure of the experiment.
"""

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.bench import adapters                        # noqa: E402
from src.bench.compare import straight_edges          # noqa: E402
from src.bench.summarise import (holm, rank_biserial,  # noqa: E402
                                 transform, wilcoxon)
from src.layout import compound as _compound          # noqa: E402
from src.layout import engine as _engine              # noqa: E402
from src.layout import metrics                        # noqa: E402
from src.layout import sugiyama as _sugiyama          # noqa: E402
from src.layout.render import build_escher_map        # noqa: E402

REPORTED = ("crossings_per_graph_edge", "axis_aligned", "hairball_index",
            "aspect_ratio", "min_separation_ratio")


def _barycenter(pl, width, gap):
    """Barycentre x-assignment: what Brandes-Koepf replaced.

    The design document claims Brandes-Koepf "is what produces the straight
    vertical backbone; a barycenter x-assignment does not". This is that
    barycentre, so the claim can be checked rather than asserted: iterate each
    node to the mean x of its neighbours, then re-space within the layer to
    restore separation while keeping the crossing-minimised order.
    """
    x = {}
    for layer in pl.layers:
        cursor = 0.0
        for v in layer:
            w = width.get(v, 40.0)
            x[v] = cursor + w / 2.0
            cursor += w + gap

    for _ in range(8):
        for layer in pl.layers:
            for v in layer:
                neighbours = list(pl.prev.get(v, ())) + list(pl.next.get(v, ()))
                seen = [x[u] for u in neighbours if u in x]
                if seen:
                    x[v] = sum(seen) / len(seen)
            # Restore separation without reordering.
            ordered = sorted(layer, key=lambda v: x[v])
            cursor = None
            for v in ordered:
                w = width.get(v, 40.0)
                if cursor is not None and x[v] - w / 2.0 < cursor:
                    x[v] = cursor + w / 2.0
                cursor = x[v] + w / 2.0 + gap
    return x


@contextlib.contextmanager
def _patch(target, name, value):
    original = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, original)


@contextlib.contextmanager
def no_cofactor_tiering():
    """Cofactor-ness as a score penalty only, which is what the tier replaced.

    Both of these patched `_COFACTOR_CUTOFF` until it was noticed that nothing
    reads it: the tier is taken from membership of the curated `NEVER_PRIMARY`
    list, and the cutoff has been a dead constant since that became true. So
    `no_cofactor_tiering` was a no-op -- it reported agreement identical to the
    full method on iJO1366 and iYO844 to the reaction, which is what exposed it
    -- and `no_cofactor_handling` was removing the damping alone while leaving
    the tier in place. Emptying the list is what actually drops every candidate
    into one tier and leaves damped chemistry to decide.
    """
    with _patch(_compound, "NEVER_PRIMARY", frozenset()):
        yield


@contextlib.contextmanager
def no_cofactor_handling():
    """No tier and no damping: the main pair chosen on raw shared chemistry."""
    with _patch(_compound, "NEVER_PRIMARY", frozenset()), \
         _patch(_compound, "_COFACTOR_DAMPING", 0.0):
        yield


@contextlib.contextmanager
def no_rings():
    """No cycle detection, so the TCA cycle is drawn as a chain."""
    with _patch(_engine, "find_rings", lambda *a, **k: []):
        yield


@contextlib.contextmanager
def no_brandes_koepf():
    with _patch(_sugiyama, "brandes_koepf", _barycenter):
        yield


@contextlib.contextmanager
def no_gap_filling():
    """Every component packed as a core, i.e. the state before two-tier packing."""
    with _patch(_engine, "FRAGMENT_SHARE", 0.0), \
         _patch(_engine, "FRAGMENT_FLOOR", 0):
        yield


VARIANTS = {
    "full": None,
    "no_cofactor_tiering": no_cofactor_tiering,
    "no_cofactor_handling": no_cofactor_handling,
    "no_rings": no_rings,
    "no_bk": no_brandes_koepf,
    "no_gap_filling": no_gap_filling,
    "no_fba": "use_fba",          # handled as a parameter, not a patch
}


def draw(model, reactions, name, variant):
    use_fba = variant != "no_fba"
    ctx = VARIANTS.get(variant)
    manager = contextlib.nullcontext() if (ctx is None or variant == "no_fba") else ctx()
    with manager:
        result = _engine.layout_reactions(model, reactions, name,
                                          use_fba=use_fba, render=False)
        if result is None:
            return None, 0
        pos = {n: p for n, p in result.pos.items()
               if not str(n).startswith("__dummy__")}
        pos = adapters.normalise(pos)
        with straight_edges():
            chart = build_escher_map(result.cgraph, pos, name)
    return chart, result.cgraph.D.number_of_edges()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="iJO1366")
    parser.add_argument("--model-dir", default=os.path.join("data", "bigg", "models"))
    parser.add_argument("--only", default="")
    parser.add_argument("--limit", type=int, default=0)
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
    if args.limit:
        names = names[:args.limit]
    print("%s: %d clusters" % (args.model, len(names)))

    wanted = [v for v in VARIANTS if v == "full"
              or not args.only or v in args.only.split(",")]
    rows = {v: {} for v in wanted}
    for i, name in enumerate(names, 1):
        for variant in wanted:
            try:
                chart, edges = draw(model, groups[name], name, variant)
            except Exception as exc:
                print("  %-22s %-28s FAILED %s" % (variant, name[:28], exc))
                continue
            if chart is None:
                continue
            values = metrics.score(chart, pitch=180.0)
            segs = len(metrics._straight_segments(chart[1]))
            values["crossings_per_graph_edge"] = (
                values["crossings_per_edge"] * max(segs, 1) / max(edges, 1))
            rows[variant][name] = values
        if i % 10 == 0 or i == len(names):
            print("  [%d/%d]" % (i, len(names)))

    full = rows["full"]
    cells, praw = [], []
    for metric in REPORTED:
        for variant in wanted:
            if variant == "full":
                continue
            shared = [k for k in full if k in rows[variant]]
            pairs = [(transform(metric, full[k][metric]),
                      transform(metric, rows[variant][k][metric]))
                     for k in shared]
            p, used = wilcoxon(pairs)
            cells.append({
                "metric": metric, "variant": variant,
                "full": statistics.median(full[k][metric] for k in shared) if shared else float("nan"),
                "ablated": statistics.median(rows[variant][k][metric] for k in shared) if shared else float("nan"),
                "effect": rank_biserial(pairs), "pairs": used,
            })
            praw.append(p)
    adjusted = holm(praw)

    print()
    for metric in REPORTED:
        shown = [(c, p) for c, p in zip(cells, adjusted) if c["metric"] == metric]
        if not shown:
            continue
        print("%s   (full = %.3f)" % (metric, shown[0][0]["full"]))
        print("  %-22s%12s%10s%12s%8s" %
              ("ablation", "ablated", "effect", "p (Holm)", "pairs"))
        for cell, p in shown:
            ptxt = "n/a" if p is None else ("%.1e" % p if p < 1e-3 else "%.3f" % p)
            if p is not None and p < 0.05:
                verdict = "component helps" if cell["effect"] < 0 else "component HURTS"
            else:
                verdict = "no detected effect"
            print("  %-22s%12.3f%10.2f%12s%8d  %s"
                  % (cell["variant"], cell["ablated"], cell["effect"], ptxt,
                     cell["pairs"], verdict))
        print()
    print("effect < 0 means the full method is better, i.e. removing the")
    print("component made the drawing worse and the component earns its place.")

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "unknown"
    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, "ablations_%s.json" % args.model)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump({"commit": sha, "model": args.model, "clusters": len(names),
                   "cells": [dict(c, p_holm=p) for c, p in zip(cells, adjusted)]},
                  handle, indent=2)
    print("\nwrote %s (commit %s)" % (target, sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
