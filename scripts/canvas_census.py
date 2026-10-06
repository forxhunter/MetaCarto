"""Draw every model on one canvas and measure it against `--combined`.

For each model: the default decomposition, every pathway laid out, then both
whole-model maps -- the species canvas (`canvas.compose_canvas`) and the
rectangle-packed poster (`compose.compose`) -- scored on the same tiles:

  blank_share      area with nothing drawn within a node spacing
  largest_blank    the largest empty square, as a share of the drawing
  largest_blank_rect  the largest empty rectangle: a long empty band
  cross_overlaps   cells where two different pathways' ink meet (must be 0)
  region_cohesion  local share of neighbours in the same superclass
  link_ratio       distance between pathways that exchange metabolites over
                   the distance between any two (below 1 = related nearby)

    python scripts/canvas_census.py --jobs 16 --out canvas_census.json
    python scripts/canvas_census.py --models e_coli_core iML1515 --save maps/
"""

import argparse
import glob
import json
import os
import statistics
import sys
import time
import traceback
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join("data", "bigg", "models")


def model_paths(only=None):
    paths = sorted(glob.glob(os.path.join(MODEL_DIR, "*.json"))
                   + glob.glob(os.path.join(MODEL_DIR, "*.xml")))
    seen, unique = set(), []
    for path in paths:                       # JSON first: it has subsystems
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem not in seen and (not only or stem in only):
            seen.add(stem)
            unique.append(path)
    return unique


def measure(escher_map, labels, meta, names):
    from src.layout import canvas, metrics
    blank = metrics.blank_space(escher_map)
    organisation = canvas.organisation(escher_map, labels, meta, names)
    return {
        "blank_share": blank["blank_share"],
        "largest_blank": blank["largest_blank_share"],
        "largest_blank_rect": blank["largest_blank_rect_share"],
        "cross_overlaps": canvas.cross_overlaps(escher_map),
        "region_cohesion": organisation["region_cohesion"],
        "link_ratio": organisation["link_ratio"],
        "width": escher_map[1]["canvas"]["width"],
        "height": escher_map[1]["canvas"]["height"],
    }


def run(args):
    path, save = args
    os.chdir(ROOT)
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        import layout_v2
        from src.layout import render
        from src.layout.compose import compose
        from src.layout.engine import layout_reactions

        start = time.time()
        model = layout_v2.load_model(path)
        from src.layout.compound import compute_cofactor_scores
        from src.layout.decompose import clusters
        groups = clusters(model, compute_cofactor_scores(model))
        tiles = []
        for name, reactions in sorted(groups.items()):
            result = layout_reactions(model, reactions, name, author=layout_v2.AUTHOR)
            if result is not None and result.escher_map[1]["nodes"]:
                tiles.append((name, result.escher_map))
        laid_out = time.time() - start
        if not tiles:
            return {"model": stem, "error": "no drawable pathway"}

        from src.layout import taxonomy
        from src.layout.compose import build_meta_graph
        from src.layout.decompose import load_kegg_mapping
        mapping = load_kegg_mapping()
        labels = {n: taxonomy.classify(n, groups[n], mapping) for n, _ in tiles}
        meta = build_meta_graph(groups, compute_cofactor_scores(model))
        names = [n for n, _ in tiles]

        from src.layout.canvas import BOUNDARY, MEMBRANE_MIN_CORE
        boundary = sum(1 for n in names if labels[n] == BOUNDARY)
        membrane = boundary >= 3 and len(names) - boundary >= MEMBRANE_MIN_CORE

        start = time.time()
        sheet = layout_v2.species_canvas(model, stem, groups, tiles)
        composed = time.time() - start
        old = compose(tiles, meta, f"{stem}_Combined", author=layout_v2.AUTHOR)

        if save:
            os.makedirs(save, exist_ok=True)
            render.save(sheet, os.path.join(save, f"{stem}_Canvas.json"))

        return {
            "model": stem,
            "reactions": len(model.reactions),
            "pathways": len(tiles),
            "membrane": membrane,
            "layout_seconds": laid_out,
            "canvas_seconds": composed,
            "canvas": measure(sheet, labels, meta, names),
            "combined": measure(old, labels, meta, names) if old else None,
        }
    except Exception as exc:                        # noqa: BLE001
        return {"model": stem, "error": f"{exc}", "trace": traceback.format_exc()}


def pct(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    k = q * (len(values) - 1)
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="*", help="model ids (default: all)")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out", help="write per-model results to this JSON")
    parser.add_argument("--save", help="also write each canvas map into this directory")
    args = parser.parse_args(argv)

    os.chdir(ROOT)
    jobs = [(p, args.save) for p in model_paths(args.models)]
    results = []
    with Pool(args.jobs) as pool:
        for result in pool.imap_unordered(run, jobs):
            results.append(result)
            if "error" in result:
                print(f"{result['model']:24s} FAILED {result['error']}", flush=True)
            else:
                c, o = result["canvas"], result["combined"] or {}
                print(f"{result['model']:24s} {result['reactions']:6d} rx "
                      f"blank {o.get('blank_share', float('nan')):.3f} -> {c['blank_share']:.3f}  "
                      f"largest {o.get('largest_blank', float('nan')):.3f} -> {c['largest_blank']:.3f}  "
                      f"cross {c['cross_overlaps']}  link {c['link_ratio']:.2f}  "
                      f"{result['layout_seconds'] + result['canvas_seconds']:.0f}s", flush=True)
    results.sort(key=lambda r: r["model"])

    ok = [r for r in results if "error" not in r]
    print(f"\n{len(ok)} of {len(results)} models drawn; "
          f"{len(results) - len(ok)} failed")
    if ok:
        print(f"{'':18s}{'combined p10/med/p90':>26s}{'canvas p10/med/p90':>26s}")
        for key in ("blank_share", "largest_blank", "region_cohesion", "link_ratio",
                    "cross_overlaps"):
            row = []
            for side in ("combined", "canvas"):
                values = [r[side][key] for r in ok if r.get(side)]
                row.append("%.3f / %.3f / %.3f" % (pct(values, .1), statistics.median(values),
                                                   pct(values, .9)))
            print(f"{key:18s}{row[0]:>26s}{row[1]:>26s}")
        better = sum(1 for r in ok if r["combined"]
                     and r["canvas"]["blank_share"] < r["combined"]["blank_share"])
        clean = sum(1 for r in ok if r["canvas"]["cross_overlaps"] == 0)
        print(f"\ncanvas emptier than --combined in {better} of {len(ok)} models; "
              f"no pathway overlap in {clean} of {len(ok)}")
        times = [r["layout_seconds"] + r["canvas_seconds"] for r in ok]
        print(f"seconds per model: median {statistics.median(times):.0f}, max {max(times):.0f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
