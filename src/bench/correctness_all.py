"""Score main-pair choice against KEGG for every model that can be scored.

`correctness.py` does one model per invocation and re-parses the whole KGML
mirror each time, which is fine for four models and wasteful for a hundred.
This loads the curated drawings once and walks the corpus.

A model is scorable only where its reactions and metabolites carry KEGG
identifiers, so the number of models that yield anything is itself a result:
BiGG annotation is uneven, and a model contributing three scorable reactions
should not be averaged with one contributing seven hundred. Models below
`--min-reactions` are recorded as skipped, with their counts, rather than
dropped silently.

Resumable: a model already present in the output file is not recomputed unless
--force is given, because loading Recon3D's SBML is minutes on its own.

    python -m src.bench.correctness_all
    python -m src.bench.correctness_all --min-reactions 30
    python -m src.bench.correctness_all --models iJO1366,iMM904 --force
"""

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.bench.correctness import (evaluate, evaluate_metdraw,  # noqa: E402
                                   kegg_drawn_pairs)

VARIANTS = ("full", "no_cofactor_tiering", "no_cofactor_handling")
THRESHOLDS = (0.80, 0.90, 0.95)

# Below this many scorable reactions a rate is noise: one disagreement moves
# it by tens of points. Such models are still reported, but excluded from the
# pooled figures.
MIN_SCORABLE = 20


def model_paths(model_dir, only=None):
    """Model id -> file, preferring .xml, which is what BiGG ships."""
    found = {}
    for ext in (".json", ".xml"):
        for path in glob.glob(os.path.join(model_dir, "*" + ext)):
            found[os.path.basename(path)[:-len(ext)]] = path
    if only:
        wanted = [m.strip() for m in only.split(",") if m.strip()]
        found = {k: v for k, v in found.items() if k in wanted}
    return dict(sorted(found.items()))


def score_model(path, drawn):
    import cobra
    model = (cobra.io.load_json_model(path) if path.endswith(".json")
             else cobra.io.read_sbml_model(path))
    row = {"reactions_in_model": len(model.reactions), "variants": {}}
    for variant in VARIANTS:
        considered, agree, _missed = evaluate(model, drawn, variant)
        row["variants"][variant] = {
            "considered": considered,
            "agree": agree,
            "rate": agree / considered if considered else None,
        }
    row["scored"] = row["variants"]["full"]["considered"]
    if row["scored"]:
        row["metdraw"] = {
            "p%d" % int(p * 100): evaluate_metdraw(model, drawn, p)
            for p in THRESHOLDS
        }
    return row


def summarise(rows, floor):
    """Pooled and per-model figures over the models worth pooling."""
    usable = {m: r for m, r in rows.items() if r.get("scored", 0) >= floor}
    out = {"models_scored": len(usable), "models_skipped": len(rows) - len(usable),
           "min_scorable": floor}
    if not usable:
        return out

    out["reactions_scored"] = sum(r["scored"] for r in usable.values())
    for variant in VARIANTS:
        rates = [r["variants"][variant]["rate"] for r in usable.values()
                 if r["variants"][variant]["rate"] is not None]
        agree = sum(r["variants"][variant]["agree"] for r in usable.values())
        total = sum(r["variants"][variant]["considered"] for r in usable.values())
        out[variant] = {
            # Pooled counts every reaction once; the median weights every
            # model equally. They answer different questions, so both.
            "pooled_rate": agree / total if total else None,
            "median_rate": statistics.median(rates) if rates else None,
            "min_rate": min(rates) if rates else None,
            "max_rate": max(rates) if rates else None,
            "models": len(rates),
        }

    md = {}
    for key in ("p%d" % int(p * 100) for p in THRESHOLDS):
        recalls, precisions, epr = [], [], []
        for r in usable.values():
            stats = r.get("metdraw", {}).get(key)
            if stats:
                recalls.append(stats["recall"])
                precisions.append(stats["precision"])
                epr.append(stats["edges_per_reaction"])
        if recalls:
            md[key] = {
                "recall_median": statistics.median(recalls),
                "recall_min": min(recalls), "recall_max": max(recalls),
                "precision_median": statistics.median(precisions),
                "precision_min": min(precisions), "precision_max": max(precisions),
                "edges_median": statistics.median(epr),
                "edges_min": min(epr), "edges_max": max(epr),
                "models": len(recalls),
            }
    out["metdraw"] = md
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=os.path.join("data", "bigg", "models"))
    parser.add_argument("--kegg", default=os.path.join("data", "kegg"))
    parser.add_argument("--out", default=os.path.join("benchmarks", "results"))
    parser.add_argument("--models", help="comma-separated subset")
    parser.add_argument("--min-reactions", type=int, default=MIN_SCORABLE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    target = os.path.join(args.out, "correctness_all.json")
    rows = {}
    if os.path.exists(target) and not args.force:
        with open(target, encoding="utf-8") as handle:
            rows = json.load(handle).get("per_model", {})
        print("resuming: %d models already scored" % len(rows))

    print("reading KGML reaction drawings ...")
    drawn = kegg_drawn_pairs(args.kegg)
    print("  %d KEGG reactions with a drawn substrate/product pair" % len(drawn))

    paths = model_paths(args.model_dir, args.models)
    print("%d models on disk" % len(paths))
    print()
    print("%-22s%9s%9s%9s%9s%8s" % ("model", "rxns", "scored", "full", "-tier", "s"))

    failed = []
    for index, (model_id, path) in enumerate(paths.items(), 1):
        if model_id in rows and not args.force:
            continue
        started = time.time()
        try:
            row = score_model(path, drawn)
        except Exception:                                # noqa: BLE001
            failed.append(model_id)
            print("%-22s FAILED" % model_id)
            traceback.print_exc(limit=1)
            continue
        row["seconds"] = time.time() - started
        rows[model_id] = row
        full = row["variants"]["full"]["rate"]
        tier = row["variants"]["no_cofactor_tiering"]["rate"]
        print("%-22s%9d%9d%9s%9s%8.0f"
              % (model_id, row["reactions_in_model"], row["scored"],
                 "%.1f%%" % (100 * full) if full is not None else "--",
                 "%.1f%%" % (100 * tier) if tier is not None else "--",
                 row["seconds"]))

        # Written every model, so an interrupted run resumes rather than
        # restarts -- the corpus takes long enough that it will be interrupted.
        os.makedirs(args.out, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump({"per_model": rows}, handle, indent=1)

    summary = summarise(rows, args.min_reactions)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                    # noqa: BLE001
        sha = "unknown"
    with open(target, "w", encoding="utf-8") as handle:
        json.dump({"commit": sha, "kegg_reactions_drawn": len(drawn),
                   "summary": summary, "failed": failed,
                   "per_model": rows}, handle, indent=1)

    print()
    print("models with >= %d scorable reactions: %d of %d"
          % (args.min_reactions, summary.get("models_scored", 0), len(rows)))
    print("reactions scored: %d" % summary.get("reactions_scored", 0))
    for variant in VARIANTS:
        s = summary.get(variant)
        if s:
            print("  %-22s pooled %.1f%%  median %.1f%%  range %.1f-%.1f%%"
                  % (variant, 100 * s["pooled_rate"], 100 * s["median_rate"],
                     100 * s["min_rate"], 100 * s["max_rate"]))
    if failed:
        print("failed to load: %s" % ", ".join(failed))
    print("wrote %s (commit %s)" % (target, sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
